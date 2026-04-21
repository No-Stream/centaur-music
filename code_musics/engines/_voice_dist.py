"""Per-note voice distortion helper for VA-family engines.

This module provides :func:`apply_voice_dist` — a single entry point that
dispatches across six distortion modes plus an ``off`` passthrough.  It is
intended to be called **inside** the per-note render loop of the
``polyblep`` / ``va`` / ``filtered_stack`` engines, after the VCA stage and
before the note buffer is summed into the voice-output buffer.  Distorting
each note independently *before* summing is the RePro-5 polyphonic-
distortion idiom — chord notes retain their harmonic identity instead of
collapsing into IMD mud.

Design notes:

* Fast paths: ``mode="off"`` and ``drive <= 0.0`` return the input
  untouched.  This matters because the helper is called per-note, per-
  voice — skipping the ``drive=0`` case is the difference between
  bit-identical renders and unintended spectral coloration on existing
  pieces.
* Saturation-blend coefficient (Track B.2 idiom): when the drive passes
  through zero during automation, the wet/dry crossfade is driven by a
  smoothstep of ``drive`` in ``[0, epsilon]`` rather than a hard gate.
  This avoids stepping / click artifacts.
* Deferred imports: ``saturation`` and ``preamp`` modes dispatch to
  :func:`code_musics.synth.apply_saturation` and
  :func:`code_musics.synth.apply_preamp`.  These must be imported inside
  the function body — importing at module top triggers the
  ``synth.py -> engines/__init__.py`` cycle.  See
  :mod:`code_musics.engines.drum_voice` (``_apply_layer_shaper``) for the
  same dispatch pattern.
* Per-algorithm oversampling: the cheap shapers use first-order ADAA
  where available (``soft_clip`` / ``hard_clip``) and auto-upgrade to 2x
  for the digital algorithms (``corrode`` = bit_crush + rate_reduce).
  ``saturation`` / ``preamp`` run with their own oversample_factor=2
  override.  No extra oversampling is added on top.
"""

from __future__ import annotations

import numpy as np

from code_musics.engines._dsp_utils import (
    alpha_from_cutoff,
    iir_lowpass_1pole,
    smoothstep_blend,
)
from code_musics.engines._waveshaper import apply_waveshaper

# Mirrors ``code_musics.synth.SAMPLE_RATE`` as a module-local constant so
# we can avoid importing from ``synth.py`` at module top (the circular
# import from that direction is what the deferred-import pattern exists
# to sidestep).
_DEFAULT_SAMPLE_RATE: int = 44_100

# Supported modes.  Kept as a frozenset for cheap membership checks.
_VALID_MODES: frozenset[str] = frozenset(
    {"off", "soft_clip", "hard_clip", "foldback", "corrode", "saturation", "preamp"}
)

# Waveshaper-algorithm mapping for the "cheap" modes.
_MODE_TO_WAVESHAPER_ALGO: dict[str, str] = {
    "soft_clip": "tanh",
    "hard_clip": "hard_clip",
    "foldback": "linear_fold",
}

# Blend-coefficient epsilon.  When 0 < drive < epsilon, wet/dry is
# crossfaded via smoothstep(drive / epsilon).  Chosen to match the
# "drive across zero has no audible step" success criterion in the
# test suite.  Larger values would smear perceived onset of the slot;
# smaller values would expose FFT quantization in the continuity test.
_BLEND_EPSILON: float = 1e-3

# Tone-tilt corner frequencies (mirroring the ``_apply_saturation_legacy``
# idiom at synth.py:5162-5165 where the legacy path uses a 2.8 kHz one-pole
# lowpass split).  Asymmetric corners keep the "bright" vs "dark" side
# perceptually distinct.
_TONE_HIGH_PIVOT_HZ: float = 1_000.0
_TONE_LOW_PIVOT_HZ: float = 2_000.0


def _onepole_lowpass(
    signal: np.ndarray, cutoff_hz: float, sample_rate: int
) -> np.ndarray:
    """Single-pole lowpass via the shared RC-alpha + IIR helper."""
    alpha = alpha_from_cutoff(cutoff_hz, sample_rate)
    return iir_lowpass_1pole(signal, alpha)


def _map_drive(drive: float) -> float:
    """User-facing drive (0.0-2.0) -> internal drive for shaper algos.

    Linear mapping ``drive * 0.5`` gives:

    * drive=0.0 -> 0.0 (continuous with the fast path)
    * drive=0.5 -> 0.25 (subtle per AGENTS.md loudness ladder)
    * drive=1.0 -> 0.5 (moderate)
    * drive=2.0 -> 1.0 (max drive for the cheap shapers; meaningful for
      ``apply_saturation`` / ``apply_preamp`` on the gentler side)

    NOTE: the plan-doc suggested ``0.5 + 2.0 * drive``, but that mapping
    is discontinuous at drive->0+ (jumps from fast-path dry to internal
    drive 0.5 which is already aggressively shaping) — defeats the
    saturation-blend continuity idiom.  Using ``drive * 0.5`` instead
    keeps the mapping zero-continuous and stays inside the natural
    ``[0, 1]`` range that :func:`apply_waveshaper` expects.  Reported as
    API feedback in the agent summary.
    """
    return drive * 0.5


def _blend_coefficient(drive: float) -> float:
    """Smoothstep(drive / epsilon) for drive in [0, epsilon], 1.0 above."""
    return smoothstep_blend(drive, _BLEND_EPSILON)


def _apply_tone_stage(signal: np.ndarray, tone: float, sample_rate: int) -> np.ndarray:
    """Pre-shaper 1-pole tilt.

    Mirrors the ``tone_tilt`` idiom used by
    :func:`code_musics.synth._apply_saturation_legacy` (synth.py:5162-5165):
    add a scaled copy of ``signal - lowpass(signal)`` to brighten, or
    subtract the mirror dual to darken.

    tone=0.0 is a no-op (bit-identical passthrough).
    """
    if tone == 0.0:
        return signal
    if tone > 0.0:
        emphasized = _onepole_lowpass(signal, _TONE_HIGH_PIVOT_HZ, sample_rate)
        # signal + tone * (highpass-ish residue)
        return signal + tone * (signal - emphasized)
    # tone < 0: bias toward lows by subtracting the high-frequency residue.
    emphasized = _onepole_lowpass(signal, _TONE_LOW_PIVOT_HZ, sample_rate)
    return signal - abs(tone) * (signal - emphasized)


def _apply_corrode(signal: np.ndarray, internal_drive: float) -> np.ndarray:
    """Sequential bit_crush -> rate_reduce.

    Maps the user drive (already passed through :func:`_map_drive`) onto
    bit depth (16 -> ~6) and rate reduction (1 -> ~8) so corrode deepens
    smoothly with drive.
    """
    # drive range here is ~[0.5, 4.5].  Normalize to [0, 1] span for the
    # bit/rate parameterization.  At internal_drive=0.5 we want minimal
    # crushing; at internal_drive=4.5 we want heavy crushing.
    normalized = float(np.clip((internal_drive - 0.5) / 4.0, 0.0, 1.0))
    bit_depth = 16.0 - 10.0 * normalized  # 16 -> 6
    reduce_ratio = 1.0 + 7.0 * normalized  # 1 -> 8
    # The waveshaper's drive param controls pre-gain; keep it mild since
    # the quantization itself is the audible effect.
    waveshaper_drive = float(np.clip(normalized * 0.5, 0.0, 1.0))
    crushed = apply_waveshaper(
        signal,
        algorithm="bit_crush",
        drive=waveshaper_drive,
        mix=1.0,
        bit_depth=bit_depth,
    )
    return apply_waveshaper(
        crushed,
        algorithm="rate_reduce",
        drive=waveshaper_drive,
        mix=1.0,
        reduce_ratio=reduce_ratio,
    )


def apply_voice_dist(
    signal: np.ndarray,
    *,
    mode: str,
    drive: float = 0.5,
    mix: float = 1.0,
    tone: float = 0.0,
    sample_rate: int = _DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """Apply per-note voice distortion to a mono buffer.

    Args:
        signal: Mono per-note buffer, float64.  Shape is preserved.
        mode: One of ``{"off", "soft_clip", "hard_clip", "foldback",
            "corrode", "saturation", "preamp"}``.  Unknown modes raise
            ``ValueError``.
        drive: 0.0-2.0.  ``drive <= 0.0`` is a fast-path passthrough.
        mix: 0.0-1.0 wet/dry ratio.  0 returns dry, 1 returns fully wet.
        tone: -1.0..1.0 pre-stage 1-pole tilt.  0.0 is a no-op.
        sample_rate: Audio sample rate (used by tone stage and preamp).

    Returns:
        Processed buffer, same dtype/shape as ``signal``.
    """
    # Validate mode up-front so misspellings fail loudly.
    if mode not in _VALID_MODES:
        raise ValueError(f"Unsupported voice_dist mode: {mode!r}")

    dry = np.asarray(signal, dtype=np.float64)

    # Fast paths: off mode or drive<=0 -> exact passthrough.
    # Preserves bit-identical output for existing pieces when the slot
    # defaults are untouched.
    if mode == "off" or drive <= 0.0:
        return dry

    internal_drive = _map_drive(drive)

    # Pre-shaper tone stage (skipped when tone == 0.0).
    toned = _apply_tone_stage(dry, tone, sample_rate)

    # Dispatch to the per-mode shaper.
    if mode in _MODE_TO_WAVESHAPER_ALGO:
        wet = apply_waveshaper(
            toned,
            algorithm=_MODE_TO_WAVESHAPER_ALGO[mode],
            drive=float(np.clip(internal_drive, 0.0, 1.0)),
            mix=1.0,
        )
    elif mode == "corrode":
        wet = _apply_corrode(toned, internal_drive)
    elif mode == "saturation":
        # Deferred import breaks the synth.py <-> engines/__init__.py cycle.
        # See drum_voice.py:367-369 for the established pattern.
        from code_musics.synth import apply_saturation

        wet_result = apply_saturation(
            toned,
            drive=internal_drive,
            mix=1.0,
            oversample_factor=2,
            compensation_mode="none",
            return_analysis=False,
        )
        wet = np.asarray(wet_result, dtype=np.float64)
    else:  # mode == "preamp"
        from code_musics.synth import apply_preamp

        wet_result = apply_preamp(
            toned,
            drive=internal_drive,
            mix=1.0,
            oversample_factor=2,
            compensation_mode="none",
            sample_rate=sample_rate,
            return_analysis=False,
        )
        wet = np.asarray(wet_result, dtype=np.float64)

    # Saturation-blend coefficient: smoothstep crossfade near drive=0 so
    # automating drive through zero is continuous.  For drive above the
    # epsilon floor this is a no-op (blend == 1.0) and collapses to the
    # standard wet/dry mix.
    blend = _blend_coefficient(drive)
    wet_weight = float(mix) * blend
    out = (1.0 - wet_weight) * dry + wet_weight * wet

    # Final numerical safety net.  The underlying shapers have their own
    # guards; this catches the rare case where tone + corrode hits a
    # pathological input.  Cheap insurance.
    return np.nan_to_num(out, copy=False)
