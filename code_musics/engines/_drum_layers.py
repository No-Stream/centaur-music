"""Four composable layer generator functions for the unified drum_voice engine.

Each function returns a raw mono float64 numpy array BEFORE envelope/level
scaling — the main engine handles envelopes and mixing.

Layer types are ported from existing engines:
- Exciter: click (kick_tom), impulse, multi_tap (clap)
- Tone: oscillator (kick_tom), resonator (kick_tom), fm (kick_tom/snare), additive
- Noise: white, colored (snare), bandpass, comb (snare)
- Metallic: partials (metallic_perc)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from code_musics.engines._drum_utils import (
    bandpass_noise,
    bandpass_noise_windowed,
    integrated_phase,
)
from code_musics.engines._dsp_utils import (
    fm_modulate,
    fm_modulate_2op,
    phase_modulate_nop,
)
from code_musics.engines._envelopes import render_envelope
from code_musics.engines._filters import apply_zdf_svf
from code_musics.engines._modal import render_modal_bank
from code_musics.engines._oscillators import polyblep_square as _polyblep_square
from code_musics.engines.kick_tom import _resonator_body
from code_musics.engines.sample import _load_sample, render_sample_segment
from code_musics.engines.snare import _comb_filter
from code_musics.spectra import get_mode_table

logger: logging.Logger = logging.getLogger(__name__)

_VALID_EXCITER_TYPES = frozenset(
    {"click", "impulse", "multi_tap", "fm_burst", "noise_burst", "sample"}
)
_VALID_TONE_TYPES = frozenset(
    {"oscillator", "resonator", "fm", "additive", "efm", "modal"}
)
_VALID_NOISE_TYPES = frozenset({"white", "colored", "bandpass", "comb"})
_VALID_METALLIC_TYPES = frozenset(
    {"partials", "ring_mod", "fm_cluster", "efm_cymbal", "modal_bank"}
)

# Named EFM cymbal operator-ratio sets for the metallic efm_cymbal renderer.
# The bar / plate variants draw from spectra.get_mode_table() at call time so
# they track the canonical mode tables.
_EFM_CYMBAL_RATIO_SETS: dict[str, list[float]] = {
    "tr808": [1.0, 1.35, 1.81, 2.37, 2.86, 3.51],
    "tr909": [1.0, 1.47, 1.98, 2.59, 3.14, 3.83],
}


# ---------------------------------------------------------------------------
# Exciter layer
# ---------------------------------------------------------------------------


def render_exciter(
    *,
    n_samples: int,
    freq: float,
    sample_rate: int,
    rng: np.random.Generator,
    exciter_type: str,
    params: dict[str, Any],
) -> np.ndarray:
    """Render the exciter layer (transient/attack energy)."""
    if exciter_type not in _VALID_EXCITER_TYPES:
        raise ValueError(
            f"exciter_type must be one of {sorted(_VALID_EXCITER_TYPES)}, "
            f"got {exciter_type!r}"
        )

    if exciter_type == "click":
        return _exciter_click(
            n_samples=n_samples,
            sample_rate=sample_rate,
            rng=rng,
            center_hz=float(params.get("exciter_center_hz", 3200.0)),
            decay_s=float(params.get("exciter_decay_s", 0.007)),
            emphasis=float(params.get("exciter_emphasis", 2.4)),
        )

    if exciter_type == "impulse":
        return _exciter_impulse(
            n_samples=n_samples,
            width_samples=int(params.get("exciter_width_samples", 1)),
        )

    if exciter_type == "fm_burst":
        return _exciter_fm_burst(
            n_samples=n_samples,
            freq=freq,
            sample_rate=sample_rate,
            params=params,
        )

    if exciter_type == "noise_burst":
        return _exciter_noise_burst(
            n_samples=n_samples,
            freq=freq,
            sample_rate=sample_rate,
            rng=rng,
            params=params,
        )

    if exciter_type == "sample":
        return _exciter_sample(
            n_samples=n_samples,
            freq=freq,
            sample_rate=sample_rate,
            params=params,
        )

    return _exciter_multi_tap(
        n_samples=n_samples,
        freq=freq,
        sample_rate=sample_rate,
        rng=rng,
        params=params,
    )


def _exciter_click(
    *,
    n_samples: int,
    sample_rate: int,
    rng: np.random.Generator,
    center_hz: float,
    decay_s: float,
    emphasis: float,
) -> np.ndarray:
    """Bandpass-filtered noise burst — ported from kick_tom._transient_noise."""
    raw = rng.standard_normal(n_samples)
    shaped = bandpass_noise(raw, sample_rate=sample_rate, center_hz=center_hz)
    time = np.arange(n_samples, dtype=np.float64) / sample_rate
    envelope = np.exp(-time / decay_s)
    envelope[: max(1, n_samples // 512)] *= emphasis
    return shaped * envelope


def _exciter_impulse(*, n_samples: int, width_samples: int) -> np.ndarray:
    """Single-sample or very short pulse for exciting resonators."""
    signal = np.zeros(n_samples, dtype=np.float64)
    w = max(1, min(width_samples, n_samples))
    signal[:w] = 1.0
    return signal


def _exciter_multi_tap(
    *,
    n_samples: int,
    freq: float,
    sample_rate: int,
    rng: np.random.Generator,
    params: dict[str, Any],
) -> np.ndarray:
    """Multiple rapid noise micro-bursts — ported from clap.render."""
    n_taps = int(params.get("exciter_n_taps", 4))
    tap_spacing_s = float(params.get("exciter_tap_spacing_s", 0.005))
    tap_decay_s = float(params.get("exciter_tap_decay_s", 0.003))
    tap_crescendo = float(params.get("exciter_tap_crescendo", 0.3))
    tap_acceleration = float(params.get("exciter_tap_acceleration", 0.0))
    tap_freq_spread = float(params.get("exciter_tap_freq_spread", 0.0))
    tap_bandwidth_ratio = float(params.get("exciter_tap_bandwidth_ratio", 2.0))

    center_hz = freq
    signal = np.zeros(n_samples, dtype=np.float64)
    attack_len = max(1, int(0.0001 * sample_rate))
    cumulative_offset_s = 0.0

    for i in range(n_taps):
        if i == 0:
            tap_offset = 0
        else:
            gap_s = tap_spacing_s * (
                1.0 - tap_acceleration * (i - 1) / max(1, n_taps - 2)
            )
            gap_s = max(0.0005, gap_s)
            cumulative_offset_s += gap_s
            tap_offset = int(cumulative_offset_s * sample_rate)

        tap_length = n_samples - tap_offset
        if tap_length <= 0:
            continue

        if tap_freq_spread > 0:
            tap_center = center_hz * (
                1.0 + tap_freq_spread * 0.3 * rng.uniform(-1.0, 1.0)
            )
        else:
            tap_center = center_hz

        tap_noise = rng.standard_normal(tap_length)
        tap_noise = bandpass_noise_windowed(
            tap_noise,
            sample_rate=sample_rate,
            center_hz=tap_center,
            width_ratio=tap_bandwidth_ratio,
        )

        t_tap = np.arange(tap_length, dtype=np.float64) / sample_rate
        tap_env = np.exp(-t_tap / tap_decay_s)
        tap_env[:attack_len] *= np.linspace(0.0, 1.0, attack_len)

        tap_amp = 1.0 + tap_crescendo * (i / max(1, n_taps - 1))
        signal[tap_offset : tap_offset + tap_length] += tap_amp * tap_noise * tap_env

    return signal


def _exciter_fm_burst(
    *,
    n_samples: int,
    freq: float,
    sample_rate: int,
    params: dict[str, Any],
) -> np.ndarray:
    """Short FM-modulated oscillator burst for harmonically rich transients.

    The FM index decays with the same time constant as the burst envelope,
    so the burst starts bright and rapidly simplifies to a near-sine.
    The main engine applies the amplitude envelope — this returns the raw FM signal.
    """
    fm_ratio = float(params.get("exciter_fm_ratio", 1.5))
    fm_index = float(params.get("exciter_fm_index", 4.0))
    fm_feedback = float(params.get("exciter_fm_feedback", 0.0))
    decay_s = float(params.get("exciter_decay_s", 0.007))

    time = np.arange(n_samples, dtype=np.float64) / sample_rate
    index_env = np.exp(-time / max(decay_s, 1e-6))
    freq_profile = np.full(n_samples, freq, dtype=np.float64)

    return fm_modulate(
        freq_profile,
        mod_ratio=fm_ratio,
        mod_index=fm_index,
        sample_rate=sample_rate,
        feedback=fm_feedback,
        index_envelope=index_env,
    )


def _exciter_noise_burst(
    *,
    n_samples: int,
    freq: float,
    sample_rate: int,
    rng: np.random.Generator,
    params: dict[str, Any],
) -> np.ndarray:
    """Wider-band noise burst with optional post-burst lowpass filter.

    Uses a wider bandwidth ratio than the click exciter for a broader,
    more diffuse transient.
    """
    bandwidth_ratio = float(params.get("exciter_bandwidth_ratio", 3.0))
    filter_cutoff_hz = params.get("exciter_filter_cutoff_hz")
    filter_q = float(params.get("exciter_filter_q", 0.707))

    raw = rng.standard_normal(n_samples)
    signal = bandpass_noise_windowed(
        raw,
        sample_rate=sample_rate,
        center_hz=freq,
        width_ratio=bandwidth_ratio,
    )

    if filter_cutoff_hz is not None:
        cutoff_profile = np.full(n_samples, float(filter_cutoff_hz), dtype=np.float64)
        signal = apply_zdf_svf(
            signal,
            cutoff_profile=cutoff_profile,
            resonance_q=filter_q,
            sample_rate=sample_rate,
            filter_mode="lowpass",
            filter_drive=0.0,
        )

    return signal


def _exciter_sample(
    *,
    n_samples: int,
    freq: float,
    sample_rate: int,
    params: dict[str, Any],
) -> np.ndarray:
    """Sample-playback exciter — loads + plays a WAV as the transient layer.

    Reads ``exciter_sample_path`` (required) and any ``exciter_sample_*``
    prefixed params, mapping them onto ``render_sample_segment``'s param
    surface.  The returned signal is unnormalized so the caller's envelope and
    ``exciter_level`` apply on top as with any other exciter.
    """
    sample_path = params.get("exciter_sample_path")
    if not sample_path:
        raise ValueError(
            "exciter_type='sample' requires 'exciter_sample_path' in params"
        )

    buffer = _load_sample(str(sample_path), sample_rate)

    sample_params: dict[str, Any] = {
        "root_freq": float(params.get("exciter_sample_root_freq", freq)),
        "pitch_shift": bool(params.get("exciter_sample_pitch_shift", True)),
        "start_offset_ms": float(params.get("exciter_sample_start_offset_ms", 0.0)),
        "start_jitter_ms": float(params.get("exciter_sample_start_jitter_ms", 0.0)),
        "reverse": bool(params.get("exciter_sample_reverse", False)),
    }

    # Optional pass-through params (omit keys when absent so the underlying
    # engine's own defaults apply).
    bend_envelope = params.get("exciter_sample_bend_envelope")
    if bend_envelope is not None:
        sample_params["bend_envelope"] = bend_envelope

    ring_freq_hz = float(params.get("exciter_sample_ring_freq_hz", 0.0))
    ring_depth = float(params.get("exciter_sample_ring_depth", 0.0))
    if ring_freq_hz > 0.0 and ring_depth > 0.0:
        sample_params["ring_freq_hz"] = ring_freq_hz
        sample_params["ring_depth"] = ring_depth

    pitch_shift_semitones = float(
        params.get("exciter_sample_pitch_shift_semitones", 0.0)
    )
    if pitch_shift_semitones != 0.0:
        # Translate semitones into the sample engine's bend-envelope surface
        # as a constant cents offset.  Two-point flat envelope.
        cents = pitch_shift_semitones * 100.0
        sample_params["bend_envelope"] = [
            {"time": 0.0, "value": cents},
            {"time": 1.0, "value": cents},
        ]

    duration = n_samples / sample_rate
    return render_sample_segment(
        buffer,
        freq=freq,
        duration=duration,
        amp=1.0,
        sample_rate=sample_rate,
        params=sample_params,
    )


# ---------------------------------------------------------------------------
# Tone layer
# ---------------------------------------------------------------------------


def render_tone(
    *,
    n_samples: int,
    freq_profile: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    tone_type: str,
    exciter_signal: np.ndarray | None,
    params: dict[str, Any],
) -> np.ndarray:
    """Render the tone layer (pitched periodic content / body)."""
    if tone_type not in _VALID_TONE_TYPES:
        raise ValueError(
            f"tone_type must be one of {sorted(_VALID_TONE_TYPES)}, got {tone_type!r}"
        )

    if tone_type == "oscillator":
        return _tone_oscillator(
            n_samples=n_samples,
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            params=params,
        )

    if tone_type == "resonator":
        return _tone_resonator(
            n_samples=n_samples,
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            rng=rng,
            exciter_signal=exciter_signal,
            params=params,
        )

    if tone_type == "fm":
        return _tone_fm(
            n_samples=n_samples,
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            params=params,
        )

    if tone_type == "efm":
        return _tone_efm(
            n_samples=n_samples,
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            params=params,
        )

    if tone_type == "modal":
        return _tone_modal(
            n_samples=n_samples,
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            rng=rng,
            exciter_signal=exciter_signal,
            params=params,
        )

    return _tone_additive(
        n_samples=n_samples,
        freq_profile=freq_profile,
        sample_rate=sample_rate,
        params=params,
    )


def _tone_oscillator(
    *,
    n_samples: int,
    freq_profile: np.ndarray,
    sample_rate: int,
    params: dict[str, Any],
) -> np.ndarray:
    """Waveform oscillator with punch — ported from kick_tom oscillator mode."""
    wave = str(params.get("tone_wave", "sine")).lower()
    punch = float(params.get("tone_punch", 0.20))
    second_harmonic = float(params.get("tone_second_harmonic", 0.16))

    phase = integrated_phase(freq_profile, sample_rate=sample_rate)
    fundamental = _oscillator_wave(wave=wave, phase=phase)

    time = np.arange(n_samples, dtype=np.float64) / sample_rate
    punch_env = 1.0 + punch * np.exp(-time / 0.018)

    harmonic_phase = integrated_phase(freq_profile, sample_rate=sample_rate)
    second = np.sin(2.0 * harmonic_phase)

    body = (
        (1.0 - second_harmonic) * fundamental + second_harmonic * second
    ) * punch_env
    return body


def _oscillator_wave(*, wave: str, phase: np.ndarray) -> np.ndarray:
    """Generate oscillator waveform from phase — ported from kick_tom._oscillator."""
    if wave == "sine":
        return np.sin(phase)
    if wave == "triangle":
        return (2.0 / np.pi) * np.arcsin(np.sin(phase))
    if wave == "sine_clip":
        return np.tanh(1.8 * np.sin(phase))
    raise ValueError(
        f"tone_wave must be 'sine', 'triangle', or 'sine_clip', got {wave!r}"
    )


def _tone_resonator(
    *,
    n_samples: int,
    freq_profile: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    exciter_signal: np.ndarray | None,
    params: dict[str, Any],
) -> np.ndarray:
    """Resonator body driven by exciter — ported from kick_tom resonator mode."""
    tone_decay_s = float(params.get("tone_decay_s", 0.26))
    punch = float(params.get("tone_punch", 0.20))

    if exciter_signal is not None and np.any(exciter_signal != 0):
        excitation = exciter_signal.copy()
    else:
        impulse_samples = max(1, int(0.001 * sample_rate))
        excitation = np.zeros(n_samples, dtype=np.float64)
        impulse_noise = rng.standard_normal(impulse_samples)
        punch_env_impulse = 1.0 + punch * np.exp(
            -np.arange(impulse_samples, dtype=np.float64) / (0.001 * sample_rate)
        )
        excitation[:impulse_samples] = impulse_noise * punch_env_impulse

    early_end = max(1, n_samples // 4)
    mean_freq = float(np.mean(freq_profile[:early_end]))
    resonator_q = max(1.0, np.pi * mean_freq * tone_decay_s)

    return _resonator_body(
        excitation,
        freq_profile.astype(np.float64),
        resonator_q,
        sample_rate,
    )


def _tone_fm(
    *,
    n_samples: int,
    freq_profile: np.ndarray,
    sample_rate: int,
    params: dict[str, Any],
) -> np.ndarray:
    """FM synthesis body — ported from kick_tom/snare FM path."""
    fm_ratio = float(params.get("tone_fm_ratio", 1.41))
    fm_index = float(params.get("tone_fm_index", 3.0))
    fm_feedback = float(params.get("tone_fm_feedback", 0.0))
    fm_index_decay_s = float(params.get("tone_fm_index_decay_s", 0.05))

    time = np.arange(n_samples, dtype=np.float64) / sample_rate
    index_env = np.exp(-time / fm_index_decay_s)

    return fm_modulate(
        freq_profile,
        mod_ratio=fm_ratio,
        mod_index=fm_index,
        sample_rate=sample_rate,
        feedback=fm_feedback,
        index_envelope=index_env,
    )


def _tone_efm(
    *,
    n_samples: int,
    freq_profile: np.ndarray,
    sample_rate: int,
    params: dict[str, Any],
) -> np.ndarray:
    """Two-op DX-style FM tone using ``fm_modulate_2op``.

    Required-ish params:
        ``efm_ratio`` (float, default 1.5): first modulator ratio.
        ``efm_index_peak`` (float, default 3.0): first modulator peak index.
        ``efm_feedback`` (float in [0, 1], default 0.0): mod1 self-feedback.

    Optional second modulator (disabled by default):
        ``efm_ratio_2`` (float, default 0.0 => disabled): second modulator ratio.
        ``efm_index_2`` (float, default 0.0): second modulator peak index.
        ``efm_feedback_2`` (float, default 0.0): mod2 self-feedback.

    Envelope and carrier:
        ``efm_index_envelope`` (list[dict] | None): per-sample index multiplier,
            applied to both modulators.  When None, falls back to an exponential
            index decay with time constant ``efm_index_decay_s`` (default 0.05).
        ``efm_carrier_feedback`` (float in [0, 1], default 0.0): carrier-self
            feedback for added growl.
    """
    efm_ratio = float(params.get("efm_ratio", 1.5))
    efm_index_peak = float(params.get("efm_index_peak", 3.0))
    efm_feedback = float(params.get("efm_feedback", 0.0))

    efm_ratio_2 = float(params.get("efm_ratio_2", 0.0))
    efm_index_2 = float(params.get("efm_index_2", 0.0))
    efm_feedback_2 = float(params.get("efm_feedback_2", 0.0))

    efm_index_decay_s = float(params.get("efm_index_decay_s", 0.05))
    efm_carrier_feedback = float(params.get("efm_carrier_feedback", 0.0))
    index_envelope_raw = params.get("efm_index_envelope")

    if index_envelope_raw is not None:
        index_env = render_envelope(index_envelope_raw, n_samples, default_value=1.0)
    else:
        time = np.arange(n_samples, dtype=np.float64) / sample_rate
        index_env = np.exp(-time / max(efm_index_decay_s, 1e-6))

    # mod2 is disabled when either efm_ratio_2 or efm_index_2 is zero.
    # fm_modulate_2op treats mod2_ratio=0 as disabled (mod2_index must also be
    # 0 in that case); we also zero index/feedback to match.
    use_second_mod = efm_ratio_2 > 0.0 and efm_index_2 > 0.0
    mod2_ratio = efm_ratio_2 if use_second_mod else 0.0
    mod2_index = efm_index_2 if use_second_mod else 0.0
    mod2_feedback = efm_feedback_2 if use_second_mod else 0.0

    return fm_modulate_2op(
        freq_profile,
        mod1_ratio=efm_ratio,
        mod1_index=efm_index_peak,
        mod2_ratio=mod2_ratio,
        mod2_index=mod2_index,
        sample_rate=sample_rate,
        mod1_feedback=efm_feedback,
        mod2_feedback=mod2_feedback,
        carrier_feedback=efm_carrier_feedback,
        index_envelope=index_env,
    )


def _tone_modal(
    *,
    n_samples: int,
    freq_profile: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    exciter_signal: np.ndarray | None,
    params: dict[str, Any],
) -> np.ndarray:
    """Modal resonator bank driven by the exciter signal.

    Resolves a mode table (named via ``modal_mode_table``, or explicit
    custom arrays) plus per-mode decay / damping / position shaping, and
    hands it to :func:`render_modal_bank`.  When no exciter is present a
    tiny white-noise burst seeds the modes.
    """
    mode_ratios, mode_amps, mode_decays_s = _resolve_modal_bank_params(
        prefix="modal", params=params
    )
    return _render_modal_bank_with_fallback(
        exciter_signal=exciter_signal,
        freq_profile=freq_profile,
        mode_ratios=mode_ratios,
        mode_amps=mode_amps,
        mode_decays_s=mode_decays_s,
        n_samples=n_samples,
        sample_rate=sample_rate,
        rng=rng,
    )


def _tone_additive(
    *,
    n_samples: int,
    freq_profile: np.ndarray,
    sample_rate: int,
    params: dict[str, Any],
) -> np.ndarray:
    """Additive partial set for pitched/harmonic use."""
    raw_ratios = params.get("tone_partial_ratios")
    n_partials = int(params.get("tone_n_partials", 6))
    brightness = float(params.get("tone_brightness", 0.7))

    if raw_ratios is not None:
        partial_ratios = [float(r) for r in raw_ratios]
    else:
        partial_ratios = [float(i + 1) for i in range(n_partials)]

    nyquist = sample_rate / 2.0
    base_freq = float(np.mean(freq_profile[: max(1, n_samples // 10)]))
    signal = np.zeros(n_samples, dtype=np.float64)
    num_partials = len(partial_ratios)

    for i, ratio in enumerate(partial_ratios):
        if base_freq * ratio >= nyquist:
            continue
        weight = brightness ** (i / max(1, num_partials - 1)) if i > 0 else 1.0
        partial_freq_profile = freq_profile * ratio
        phase = integrated_phase(partial_freq_profile, sample_rate=sample_rate)
        signal += weight * np.sin(phase)

    return signal


# ---------------------------------------------------------------------------
# Noise layer
# ---------------------------------------------------------------------------


def render_noise(
    *,
    n_samples: int,
    freq: float,
    sample_rate: int,
    rng: np.random.Generator,
    noise_type: str,
    params: dict[str, Any],
) -> np.ndarray:
    """Render the noise layer (aperiodic texture)."""
    if noise_type not in _VALID_NOISE_TYPES:
        raise ValueError(
            f"noise_type must be one of {sorted(_VALID_NOISE_TYPES)}, "
            f"got {noise_type!r}"
        )

    if noise_type == "white":
        return rng.standard_normal(n_samples)

    if noise_type == "colored":
        return _noise_colored(
            n_samples=n_samples,
            sample_rate=sample_rate,
            rng=rng,
            pre_hp_hz=float(params.get("noise_pre_hp_hz", 500.0)),
        )

    if noise_type == "bandpass":
        return _noise_bandpass(
            n_samples=n_samples,
            freq=freq,
            sample_rate=sample_rate,
            rng=rng,
            center_ratio=float(params.get("noise_center_ratio", 1.0)),
            width_ratio=float(params.get("noise_width_ratio", 0.75)),
        )

    return _noise_comb(
        n_samples=n_samples,
        freq=freq,
        sample_rate=sample_rate,
        rng=rng,
        comb_feedback=float(params.get("noise_comb_feedback", 0.45)),
        pre_noise_mode=str(params.get("noise_pre_noise_mode", "white")).lower(),
        pre_hp_hz=float(params.get("noise_pre_hp_hz", 500.0)),
    )


def _noise_colored(
    *,
    n_samples: int,
    sample_rate: int,
    rng: np.random.Generator,
    pre_hp_hz: float,
) -> np.ndarray:
    """White noise + highpass — ported from snare colored wire mode."""
    raw = rng.standard_normal(n_samples)
    hp_cutoff = np.full(n_samples, pre_hp_hz, dtype=np.float64)
    return apply_zdf_svf(
        raw,
        cutoff_profile=hp_cutoff,
        resonance_q=0.707,
        sample_rate=sample_rate,
        filter_mode="highpass",
        filter_drive=0.0,
    )


def _noise_bandpass(
    *,
    n_samples: int,
    freq: float,
    sample_rate: int,
    rng: np.random.Generator,
    center_ratio: float,
    width_ratio: float,
) -> np.ndarray:
    """FFT-domain bandpass noise."""
    raw = rng.standard_normal(n_samples)
    center_hz = freq * center_ratio
    return bandpass_noise_windowed(
        raw,
        sample_rate=sample_rate,
        center_hz=center_hz,
        width_ratio=width_ratio,
    )


def _noise_comb(
    *,
    n_samples: int,
    freq: float,
    sample_rate: int,
    rng: np.random.Generator,
    comb_feedback: float,
    pre_noise_mode: str,
    pre_hp_hz: float,
) -> np.ndarray:
    """Noise through comb filter at note freq — ported from snare wire path."""
    raw = rng.standard_normal(n_samples)
    if pre_noise_mode == "colored":
        hp_cutoff = np.full(n_samples, pre_hp_hz, dtype=np.float64)
        raw = apply_zdf_svf(
            raw,
            cutoff_profile=hp_cutoff,
            resonance_q=0.707,
            sample_rate=sample_rate,
            filter_mode="highpass",
            filter_drive=0.0,
        )
    delay_samples = max(1, int(sample_rate / freq))
    return _comb_filter(raw, delay_samples, comb_feedback)


# ---------------------------------------------------------------------------
# Metallic layer
# ---------------------------------------------------------------------------


def render_metallic(
    *,
    n_samples: int,
    freq_profile: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    metallic_type: str,
    params: dict[str, Any],
    exciter_signal: np.ndarray | None = None,
) -> np.ndarray:
    """Render the metallic layer (inharmonic periodic content)."""
    if metallic_type not in _VALID_METALLIC_TYPES:
        raise ValueError(
            f"metallic_type must be one of {sorted(_VALID_METALLIC_TYPES)}, "
            f"got {metallic_type!r}"
        )

    if metallic_type == "partials":
        return _metallic_partials(
            n_samples=n_samples,
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            rng=rng,
            params=params,
        )

    if metallic_type == "ring_mod":
        return _metallic_ring_mod(
            n_samples=n_samples,
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            rng=rng,
            params=params,
        )

    if metallic_type == "efm_cymbal":
        return _metallic_efm_cymbal(
            n_samples=n_samples,
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            params=params,
        )

    if metallic_type == "modal_bank":
        return _metallic_modal_bank(
            n_samples=n_samples,
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            rng=rng,
            exciter_signal=exciter_signal,
            params=params,
        )

    return _metallic_fm_cluster(
        n_samples=n_samples,
        freq_profile=freq_profile,
        sample_rate=sample_rate,
        rng=rng,
        params=params,
    )


def _metallic_partials(
    *,
    n_samples: int,
    freq_profile: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    params: dict[str, Any],
) -> np.ndarray:
    """Additive inharmonic partials — ported from metallic_perc.render."""
    raw_ratios = params.get("metallic_partial_ratios")
    n_partials = int(params.get("metallic_n_partials", 6))
    oscillator_mode = str(params.get("metallic_oscillator_mode", "sine")).lower()
    brightness = float(params.get("metallic_brightness", 0.7))
    density = float(params.get("metallic_density", 0.5))

    if raw_ratios is not None:
        partial_ratios = [float(r) for r in raw_ratios]
    else:
        partial_ratios = [math.sqrt(float(i + 1)) for i in range(n_partials)]

    jittered_ratios = list(partial_ratios)
    if density > 0:
        for i in range(len(jittered_ratios)):
            jitter = density * 0.03 * rng.uniform(-1.0, 1.0)
            jittered_ratios[i] = jittered_ratios[i] * (1.0 + jitter)

    base_freq = float(np.mean(freq_profile[: max(1, n_samples // 10)]))
    nyquist = sample_rate / 2.0
    num_partials = len(jittered_ratios)
    signal = np.zeros(n_samples, dtype=np.float64)

    for i, ratio in enumerate(jittered_ratios):
        partial_freq = base_freq * ratio
        if partial_freq >= nyquist:
            continue
        weight = brightness ** (i / max(1, num_partials - 1)) if i > 0 else 1.0

        if oscillator_mode == "square":
            norm_phase_inc = np.full(
                n_samples, partial_freq / sample_rate, dtype=np.float64
            )
            norm_cumphase = np.cumsum(norm_phase_inc)
            norm_phase = norm_cumphase % 1.0
            signal += weight * _polyblep_square(
                norm_phase, norm_phase_inc, norm_cumphase, pulse_width=0.5
            )
        else:
            phase_inc = 2.0 * np.pi * partial_freq / sample_rate
            phase = np.cumsum(np.full(n_samples, phase_inc, dtype=np.float64))
            signal += weight * np.sin(phase)

    return signal


def _metallic_ring_mod(
    *,
    n_samples: int,
    freq_profile: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    params: dict[str, Any],
) -> np.ndarray:
    """Ring modulator on a small additive partial set for metallic shimmer.

    Generates harmonic partials, then ring-modulates them with an oscillator
    at an inharmonic ratio of the base frequency.  The ``amount`` parameter
    blends between the dry partials and the ring-modulated result.
    """
    ring_mod_freq_ratio = float(params.get("metallic_ring_mod_freq_ratio", 1.48))
    ring_mod_amount = float(params.get("metallic_ring_mod_amount", 0.5))
    n_partials = int(params.get("metallic_n_partials", 4))
    brightness = float(params.get("metallic_brightness", 0.7))
    density = float(params.get("metallic_density", 0.0))

    base_freq = float(np.mean(freq_profile[: max(1, n_samples // 10)]))
    nyquist = sample_rate / 2.0

    # Build base partial ratios (harmonic series)
    partial_ratios = [float(i + 1) for i in range(n_partials)]
    if density > 0:
        for i in range(len(partial_ratios)):
            jitter = density * 0.03 * rng.uniform(-1.0, 1.0)
            partial_ratios[i] *= 1.0 + jitter

    # Sum harmonic partials with brightness rolloff
    partials_signal = np.zeros(n_samples, dtype=np.float64)
    for i, ratio in enumerate(partial_ratios):
        if base_freq * ratio >= nyquist:
            continue
        weight = brightness ** (i / max(1, n_partials - 1)) if i > 0 else 1.0
        partial_freq_profile = freq_profile * ratio
        phase = integrated_phase(partial_freq_profile, sample_rate=sample_rate)
        partials_signal += weight * np.sin(phase)

    # Ring modulator oscillator at inharmonic ratio
    ring_mod_freq_profile = freq_profile * ring_mod_freq_ratio
    ring_mod_phase = integrated_phase(ring_mod_freq_profile, sample_rate=sample_rate)
    ring_mod_osc = np.sin(ring_mod_phase)

    # Blend: (1 - amount) * dry + amount * ring_modulated
    ring_mod_amount = max(0.0, min(1.0, ring_mod_amount))
    return (1.0 - ring_mod_amount) * partials_signal + ring_mod_amount * (
        partials_signal * ring_mod_osc
    )


def _metallic_fm_cluster(
    *,
    n_samples: int,
    freq_profile: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    params: dict[str, Any],
) -> np.ndarray:
    """Multiple FM operators at inharmonic ratios for dense metallic textures.

    Each operator is an FM pair: carrier at ``base_freq * ratio``, modulator at
    ``base_freq * ratio * 1.41`` (metallic ratio).  Operators are weighted by
    brightness rolloff and summed.  Density jitter on ratios produces thicker
    textures when enabled.
    """
    n_operators = int(params.get("metallic_n_operators", 4))
    raw_ratios = params.get("metallic_fm_ratios")
    fm_index = float(params.get("metallic_fm_index", 3.0))
    fm_feedback = float(params.get("metallic_fm_feedback", 0.0))
    brightness = float(params.get("metallic_brightness", 0.7))
    density = float(params.get("metallic_density", 0.0))

    # Default ratios: sqrt-spaced inharmonic series
    if raw_ratios is not None:
        operator_ratios = [float(r) for r in raw_ratios]
    else:
        operator_ratios = [math.sqrt(float(i + 1)) for i in range(n_operators)]

    # Apply density jitter to ratios
    if density > 0:
        for i in range(len(operator_ratios)):
            jitter = density * 0.03 * rng.uniform(-1.0, 1.0)
            operator_ratios[i] *= 1.0 + jitter

    base_freq = float(np.mean(freq_profile[: max(1, n_samples // 10)]))
    nyquist = sample_rate / 2.0
    n_ops = len(operator_ratios)
    signal = np.zeros(n_samples, dtype=np.float64)

    metallic_mod_ratio = 1.41  # sqrt(2) approx -- classic metallic interval

    for i, ratio in enumerate(operator_ratios):
        carrier_freq = base_freq * ratio
        if carrier_freq >= nyquist:
            continue

        weight = brightness ** (i / max(1, n_ops - 1)) if i > 0 else 1.0
        carrier_freq_profile = freq_profile * ratio

        signal += weight * fm_modulate(
            carrier_freq_profile,
            mod_ratio=metallic_mod_ratio,
            mod_index=fm_index,
            sample_rate=sample_rate,
            feedback=fm_feedback,
        )

    return signal


def _metallic_efm_cymbal(
    *,
    n_samples: int,
    freq_profile: np.ndarray,
    sample_rate: int,
    params: dict[str, Any],
) -> np.ndarray:
    """EFM-style cymbal: N parallel PM operators into a sine carrier.

    Reads:
        ``cymbal_op_count`` (int, default 6): number of PM operators.
        ``cymbal_ratio_set`` (str, default ``"tr808"``): named ratio set.
            Valid: ``"tr808"``, ``"tr909"``, ``"bar"``, ``"plate"``.
        ``cymbal_index`` (float, default 2.5): peak PM index (broadcast to all ops).
        ``cymbal_feedback`` (float, default 0.0): per-op self-feedback.
        ``cymbal_index_envelope``: optional envelope; applied identically to all ops.
    """
    n_ops = int(params.get("cymbal_op_count", 6))
    if n_ops <= 0:
        raise ValueError(f"cymbal_op_count must be >= 1, got {n_ops}")
    ratio_set_name = str(params.get("cymbal_ratio_set", "tr808")).lower()
    cymbal_index = float(params.get("cymbal_index", 2.5))
    cymbal_feedback = float(params.get("cymbal_feedback", 0.0))
    index_envelope_raw = params.get("cymbal_index_envelope")

    if ratio_set_name in _EFM_CYMBAL_RATIO_SETS:
        full_ratios = list(_EFM_CYMBAL_RATIO_SETS[ratio_set_name])
    elif ratio_set_name == "bar":
        full_ratios = list(get_mode_table("bar_metal"))
    elif ratio_set_name == "plate":
        full_ratios = list(get_mode_table("plate"))
    else:
        raise ValueError(
            f"cymbal_ratio_set must be one of "
            f"{sorted(list(_EFM_CYMBAL_RATIO_SETS) + ['bar', 'plate'])}, "
            f"got {ratio_set_name!r}"
        )

    # Clip / pad ratio list to exactly n_ops entries.
    if len(full_ratios) >= n_ops:
        op_ratios_list = full_ratios[:n_ops]
    else:
        tail = [
            full_ratios[-1] * (1.0 + 0.1 * (k + 1))
            for k in range(n_ops - len(full_ratios))
        ]
        op_ratios_list = full_ratios + tail

    op_ratios = np.asarray(op_ratios_list, dtype=np.float64)
    op_indices = np.full(n_ops, cymbal_index, dtype=np.float64)
    op_feedbacks = np.full(n_ops, cymbal_feedback, dtype=np.float64)

    if index_envelope_raw is not None:
        per_op_env = render_envelope(index_envelope_raw, n_samples, default_value=1.0)
        # phase_modulate_nop calls ascontiguousarray() on op_envelopes, so a
        # broadcast view is fine here — no need for an explicit .copy().
        op_envelopes = np.broadcast_to(per_op_env, (n_ops, n_samples))
    else:
        op_envelopes = None

    base_freq = float(np.mean(freq_profile[: max(1, n_samples // 10)]))
    carrier_profile = np.full(n_samples, base_freq, dtype=np.float64)

    return phase_modulate_nop(
        carrier_profile,
        op_ratios=op_ratios,
        op_indices=op_indices,
        op_feedbacks=op_feedbacks,
        sample_rate=sample_rate,
        op_envelopes=op_envelopes,
    )


def _metallic_modal_bank(
    *,
    n_samples: int,
    freq_profile: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    exciter_signal: np.ndarray | None,
    params: dict[str, Any],
) -> np.ndarray:
    """Modal resonator bank driven by exciter, in the metallic layer.

    Mirrors ``_tone_modal`` but reads ``metallic_*`` prefixed params and
    defaults to the bar_metal table.
    """
    mode_ratios, mode_amps, mode_decays_s = _resolve_modal_bank_params(
        prefix="metallic",
        params=params,
        default_table="bar_metal",
        default_decay_s=0.4,
    )
    return _render_modal_bank_with_fallback(
        exciter_signal=exciter_signal,
        freq_profile=freq_profile,
        mode_ratios=mode_ratios,
        mode_amps=mode_amps,
        mode_decays_s=mode_decays_s,
        n_samples=n_samples,
        sample_rate=sample_rate,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# Shared modal-bank parameter resolution (used by _tone_modal + _metallic_modal_bank)
# ---------------------------------------------------------------------------


def _resolve_modal_bank_params(
    *,
    prefix: str,
    params: dict[str, Any],
    default_table: str = "membrane",
    default_decay_s: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve mode ratios / amps / decays from ``{prefix}_*`` params.

    Supports both a named mode table (via ``{prefix}_mode_table``) and
    explicit custom arrays (via ``{prefix}_ratios`` / ``{prefix}_amps`` /
    ``{prefix}_decays_s``).  Applies tension / damping / damping tilt /
    position / hardness macros and returns concrete float64 arrays.
    """
    explicit_ratios = params.get(f"{prefix}_ratios")
    explicit_amps = params.get(f"{prefix}_amps")
    explicit_decays = params.get(f"{prefix}_decays_s")

    if explicit_ratios is not None:
        base_ratios = [float(r) for r in explicit_ratios]
        n_modes = len(base_ratios)
    else:
        table_name = str(params.get(f"{prefix}_mode_table", default_table))
        if table_name == "custom":
            raise ValueError(
                f"{prefix}_mode_table='custom' requires {prefix}_ratios to be "
                "set to an explicit list of mode ratios"
            )
        table_ratios = get_mode_table(table_name)
        n_modes_cap = int(params.get(f"{prefix}_n_modes", len(table_ratios)))
        if n_modes_cap <= 0:
            raise ValueError(f"{prefix}_n_modes must be >= 1, got {n_modes_cap}")
        base_ratios = list(table_ratios[:n_modes_cap])
        n_modes = len(base_ratios)

    tension = float(params.get("modal_tension", params.get(f"{prefix}_tension", 0.0)))
    tension = max(-1.0, min(1.0, tension))
    stretch_exp = 1.0 + 0.3 * tension
    shaped_ratios = [float(r) ** stretch_exp for r in base_ratios]

    if explicit_amps is not None:
        if len(explicit_amps) != n_modes:
            raise ValueError(
                f"{prefix}_amps length ({len(explicit_amps)}) must match "
                f"ratios length ({n_modes})"
            )
        base_amps = [float(a) for a in explicit_amps]
    else:
        # Mild 1/k rolloff to keep upper modes from dominating.
        base_amps = [1.0 / math.sqrt(float(i + 1)) for i in range(n_modes)]

    position = float(
        params.get("modal_position", params.get(f"{prefix}_position", 0.0))
    )
    position = max(0.0, min(1.0, position))
    shaped_amps: list[float] = []
    for i, amp in enumerate(base_amps):
        window = math.cos(math.pi * position * (i + 1) / max(1, n_modes)) ** 2
        shaped_amps.append(amp * window)

    global_decay_s = float(params.get(f"{prefix}_decay_s", default_decay_s))
    damping_mult = float(
        params.get("modal_damping", params.get(f"{prefix}_damping", 1.0))
    )
    damping_tilt = float(
        params.get("modal_damping_tilt", params.get(f"{prefix}_damping_tilt", 0.0))
    )
    damping_tilt = max(-1.0, min(1.0, damping_tilt))

    if explicit_decays is not None:
        if len(explicit_decays) != n_modes:
            raise ValueError(
                f"{prefix}_decays_s length ({len(explicit_decays)}) must match "
                f"ratios length ({n_modes})"
            )
        shaped_decays = [float(d) * damping_mult for d in explicit_decays]
    else:
        shaped_decays = [global_decay_s * damping_mult for _ in range(n_modes)]

    # Apply damping tilt: positive tilt shortens high-mode decay.
    for i in range(n_modes):
        shaped_decays[i] = shaped_decays[i] * math.exp(
            -damping_tilt * i / max(1, n_modes)
        )

    return (
        np.asarray(shaped_ratios, dtype=np.float64),
        np.asarray(shaped_amps, dtype=np.float64),
        np.asarray(shaped_decays, dtype=np.float64),
    )


def _render_modal_bank_with_fallback(
    *,
    exciter_signal: np.ndarray | None,
    freq_profile: np.ndarray,
    mode_ratios: np.ndarray,
    mode_amps: np.ndarray,
    mode_decays_s: np.ndarray,
    n_samples: int,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Excite a modal resonator bank, synthesising a 1 ms noise burst when the
    caller did not pass an exciter.

    Shared between the tone and metallic modal renderers so both follow the
    same fallback path: if an exciter signal is present and non-silent, it is
    used verbatim (copy-free cast to float64); otherwise a short white-noise
    impulse at the start of the buffer seeds the modes.
    """
    if exciter_signal is not None and np.any(exciter_signal != 0):
        excitation = exciter_signal.astype(np.float64, copy=False)
    else:
        impulse_samples = max(1, int(0.001 * sample_rate))
        excitation = np.zeros(n_samples, dtype=np.float64)
        excitation[:impulse_samples] = rng.standard_normal(impulse_samples)

    early_end = max(1, n_samples // 4)
    base_freq = float(np.mean(freq_profile[:early_end]))
    return render_modal_bank(
        excitation,
        mode_ratios=mode_ratios,
        mode_amps=mode_amps,
        mode_decays_s=mode_decays_s,
        freq_hz=base_freq,
        sample_rate=sample_rate,
    )
