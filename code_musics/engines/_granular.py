"""Granular synthesis: grain-cloud oscillator over an internal source buffer.

Source-agnostic — ``grain_source`` selects what fills the internal source
buffer (osc / partials / fm / noise / sample); grains are then windowed and
sprayed with jittered position, pitch, and size.

Modes:
- ``cloud`` — dense overlapping grains over the full source buffer.
- ``time_freeze`` — grains read from a short source window around
  ``grain_window_start``, Fennesz-style frozen-time.
- ``texture`` — sparse, large, heavily-jittered grains (density < 1/size
  so grains audibly punctuate).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from code_musics.engines._dsp_utils import bandpass_noise, flow_exciter
from code_musics.engines._oscillators import render_polyblep_oscillator
from code_musics.engines.fm import _fm_sample_loop
from code_musics.engines.sample import _load_sample
from code_musics.engines.va import _render_partial_bank, _render_supersaw_bank

_VALID_GRAIN_TYPES = frozenset({"cloud", "time_freeze", "texture"})
_VALID_GRAIN_SOURCES = frozenset({"osc", "partials", "fm", "noise", "sample"})
_VALID_GRAIN_WINDOW_SHAPES = frozenset({"hann", "gaussian", "rectangular"})

# Per-mode defaults applied only when the caller did not set the param.
# Cloud is the "basic" default (no overrides); texture favors sparse big grains;
# time_freeze leans on ``source_duration_s`` adjustment, not param overrides.
_MODE_DEFAULTS: dict[str, dict[str, float]] = {
    "cloud": {},
    "time_freeze": {},
    "texture": {
        "grain_density": 8.0,
        "grain_size_ms": 150.0,
        "grain_jitter": 0.8,
    },
}


def _build_source_buffer(
    *,
    source: str,
    freq: float,
    duration_s: float,
    sample_rate: int,
    rng: np.random.Generator,
    params: dict[str, Any],
) -> np.ndarray:
    """Render a mono source buffer to granulate.  ``duration_s`` is
    independent of the output length — the granulator advances its own
    read pointer.
    """
    if source not in _VALID_GRAIN_SOURCES:
        raise ValueError(
            f"grain_source must be one of {sorted(_VALID_GRAIN_SOURCES)}, "
            f"got {source!r}"
        )

    n_samples = max(1, int(duration_s * sample_rate))
    freq_profile = np.full(n_samples, freq, dtype=np.float64)

    if source == "osc":
        waveform = str(params.get("grain_osc_wave", "saw")).lower()
        if waveform == "supersaw":
            return _render_supersaw_bank(
                freq_profile=freq_profile,
                sample_rate=sample_rate,
                detune=float(params.get("grain_osc_spread_cents", 18.0)) / 100.0,
                mix=float(params.get("grain_osc_mix", 0.5)),
                sync=False,
                rng=rng,
                spread_profile=None,
                phase_noise_amount=0.0,
                phase_noise_freq=0.0,
                phase_noise_duration=0.0,
                phase_noise_amp=0.0,
                phase_noise_voice_name="",
            )
        sig, _ = render_polyblep_oscillator(
            waveform=waveform,
            pulse_width=float(params.get("grain_osc_pulse_width", 0.5)),
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            start_phase=0.0,
            phase_noise=None,
        )
        return sig

    if source == "partials":
        n_harmonics = int(params.get("grain_partials_n_harmonics", 12))
        rolloff = float(params.get("grain_partials_rolloff", 0.7))
        partials = [
            {
                "ratio": float(k),
                "amp": rolloff ** (k - 1),
                "phase": 0.0,
            }
            for k in range(1, n_harmonics + 1)
        ]
        return _render_partial_bank(
            partials=partials,
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            start_phase=0.0,
        )

    if source == "fm":
        mod_ratio = float(params.get("grain_fm_ratio", 2.0))
        mod_index = float(params.get("grain_fm_index", 2.5))
        feedback = float(params.get("grain_fm_feedback", 0.0))
        carrier_phase_inc = 2.0 * np.pi * freq_profile / sample_rate
        mod_phase_inc = 2.0 * np.pi * freq_profile * mod_ratio / sample_rate
        signal = np.empty(n_samples, dtype=np.float64)
        _fm_sample_loop(
            signal,
            carrier_phase_inc,
            mod_phase_inc,
            mod_index,
            feedback,
            0,  # no index decay inside the source buffer
            1.0,
            n_samples,
        )
        return signal

    if source == "noise":
        mode = str(params.get("grain_noise_mode", "white")).lower()
        if mode == "white":
            return rng.standard_normal(n_samples).astype(np.float64)
        if mode == "pink":
            white = rng.standard_normal(n_samples).astype(np.float64)
            spectrum = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
            scale = np.ones_like(freqs)
            scale[1:] = 1.0 / np.sqrt(freqs[1:])
            spectrum = spectrum * scale
            spectrum[0] = 0.0
            return np.fft.irfft(spectrum, n=n_samples).real
        if mode == "bandpass":
            return bandpass_noise(
                rng.standard_normal(n_samples).astype(np.float64),
                sample_rate=sample_rate,
                center_hz=float(params.get("grain_noise_center_hz", freq)),
                width_ratio=float(params.get("grain_noise_width_ratio", 0.5)),
            )
        if mode == "flow":
            return flow_exciter(
                n_samples=n_samples,
                param=float(params.get("grain_noise_flow_density", 0.3)),
                rng=rng,
            )
        raise ValueError(
            f"grain_noise_mode must be one of white|pink|bandpass|flow, got {mode!r}"
        )

    sample_path = params.get("grain_sample_path")
    if not sample_path:
        raise ValueError("grain_source='sample' requires grain_sample_path in params")
    buffer = _load_sample(str(sample_path), sample_rate)
    return np.asarray(buffer, dtype=np.float64)


def _window(*, shape: str, n: int) -> np.ndarray:
    """Build a grain amplitude window of the requested shape, length ``n``."""
    if n <= 1:
        return np.ones(max(0, n), dtype=np.float64)
    if shape == "hann":
        return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))).astype(
            np.float64
        )
    if shape == "gaussian":
        sigma = n / 6.0
        center = (n - 1) / 2.0
        idx = np.arange(n, dtype=np.float64)
        return np.exp(-0.5 * ((idx - center) / sigma) ** 2)
    if shape == "rectangular":
        return np.ones(n, dtype=np.float64)
    raise ValueError(
        f"grain_window_shape must be one of {sorted(_VALID_GRAIN_WINDOW_SHAPES)}, "
        f"got {shape!r}"
    )


def _pick_grain_pitch(
    *,
    base_pitch_ratio: float,
    pitch_spread: float,
    jitter_direction: float,
    ji_lattice: tuple[float, ...] | None,
) -> float:
    """Return a per-grain pitch multiplier.

    ``base_pitch_ratio`` is the nominal grain playback rate (1.0 = source
    natural pitch).  ``pitch_spread`` in ``[0, 1]`` controls how far pitch
    can deviate from the base.  ``jitter_direction`` is a uniform
    ``[-1, 1]`` random value chosen by the caller.  When ``ji_lattice`` is
    provided, the deviation is quantized to its ratios — grain pitches
    sprinkle onto harmonic intervals instead of smearing off-key.
    """
    if pitch_spread <= 0.0 or ji_lattice is None or len(ji_lattice) == 0:
        # Continuous deviation up to pitch_spread octaves.
        return base_pitch_ratio * (2.0 ** (pitch_spread * jitter_direction))

    # Quantize onto a JI ratio.  The lattice is typically a small set of
    # fractions; pick one weighted by jitter_direction's sign.
    # Positive jitter → ratios > 1, negative → ratios < 1 (or inverted).
    if jitter_direction >= 0.0:
        eligible = [r for r in ji_lattice if r >= 1.0]
    else:
        eligible = [r for r in ji_lattice if r <= 1.0]
    if not eligible:
        eligible = list(ji_lattice)
    # Pick deterministically by jitter magnitude.
    idx = int(abs(jitter_direction) * len(eligible))
    idx = min(idx, len(eligible) - 1)
    return base_pitch_ratio * float(eligible[idx]) ** pitch_spread


def render_granular(
    *,
    n_samples: int,
    freq: float,
    sample_rate: int,
    seed: int,
    grain_type: str,
    grain_source: str,
    params: dict[str, Any],
) -> np.ndarray:
    """Render one note's worth of granular texture.

    Reads ``grain_density``, ``grain_size_ms``, ``grain_jitter``,
    ``grain_pitch_spread``, ``grain_window_shape``, and mode-specific
    params from ``params``.  ``grain_source`` selects what fills the
    internal source buffer.  A fresh deterministic RNG is built from
    ``seed`` internally — callers in ``synth_voice`` derive the seed from
    their per-note rng the same way pluck / scanned / chaotic dispatches do.

    ``grain_ji_lattice`` (if present) must be a tuple or list of positive
    ``float`` ratios (e.g. ``(1.0, 5/4, 4/3, 3/2)``).

    Output is peak-normalized to <= 1.0.
    """
    if grain_type not in _VALID_GRAIN_TYPES:
        raise ValueError(
            f"grain_type must be one of {sorted(_VALID_GRAIN_TYPES)}, "
            f"got {grain_type!r}"
        )
    if n_samples <= 0:
        return np.zeros(max(0, n_samples), dtype=np.float64)

    rng = np.random.default_rng(seed)

    # --- Core grain params ---
    density = float(params.get("grain_density", 30.0))  # grains per second
    size_ms = float(params.get("grain_size_ms", 50.0))
    jitter = float(params.get("grain_jitter", 0.3))  # 0..1
    pitch_spread = float(params.get("grain_pitch_spread", 0.0))  # 0..1
    base_pitch_ratio = float(params.get("grain_pitch_ratio", 1.0))
    window_shape = str(params.get("grain_window_shape", "hann")).lower()
    ji_lattice: tuple[float, ...] | None = params.get("grain_ji_lattice")

    if density <= 0.0:
        return np.zeros(n_samples, dtype=np.float64)
    if size_ms <= 0.0:
        raise ValueError(f"grain_size_ms must be positive, got {size_ms}")
    if jitter < 0.0 or jitter > 1.0:
        raise ValueError(f"grain_jitter must be in [0, 1], got {jitter}")
    if pitch_spread < 0.0 or pitch_spread > 1.0:
        raise ValueError(f"grain_pitch_spread must be in [0, 1], got {pitch_spread}")

    # Mode-specific defaults fill in unset params (user values always win).
    mode_overrides = _MODE_DEFAULTS.get(grain_type, {})
    if "grain_density" not in params and "grain_density" in mode_overrides:
        density = mode_overrides["grain_density"]
    if "grain_size_ms" not in params and "grain_size_ms" in mode_overrides:
        size_ms = mode_overrides["grain_size_ms"]
    if "grain_jitter" not in params and "grain_jitter" in mode_overrides:
        jitter = mode_overrides["grain_jitter"]

    # --- Render source buffer ---
    source_duration_s = max(0.5, n_samples / float(sample_rate))
    if grain_type == "time_freeze":
        # Very short source buffer — granulator reads from a small window
        # repeatedly.  ``grain_window_start`` selects which fraction of a
        # notional longer render to freeze (0 = start, 1 = end).
        source_duration_s = 0.2
    source_buffer = _build_source_buffer(
        source=grain_source,
        freq=freq,
        duration_s=source_duration_s,
        sample_rate=sample_rate,
        rng=rng,
        params=params,
    )
    source_len = source_buffer.shape[0]
    if source_len <= 1:
        return np.zeros(n_samples, dtype=np.float64)

    # Peak-normalize source for predictable grain amplitudes.
    src_peak = float(np.max(np.abs(source_buffer)))
    if src_peak > 0.0:
        source_buffer = source_buffer / src_peak

    # --- Grain scheduling ---
    out = np.zeros(n_samples, dtype=np.float64)
    grain_samples = max(2, int(size_ms * 0.001 * sample_rate))
    window = _window(shape=window_shape, n=grain_samples)

    # Expected inter-grain gap (samples).
    mean_gap = max(1.0, sample_rate / density)
    # Number of grains we'll schedule, with a small safety margin.
    est_n_grains = int(1.5 * n_samples / mean_gap) + 2

    # Pre-draw all random numbers we need in one go for determinism + speed.
    gap_jitter = rng.uniform(-0.5, 0.5, size=est_n_grains) * jitter
    position_jitter = rng.uniform(-1.0, 1.0, size=est_n_grains)
    pitch_jitter = rng.uniform(-1.0, 1.0, size=est_n_grains)
    size_jitter = rng.uniform(-0.5, 0.5, size=est_n_grains) * jitter

    window_start_frac = float(params.get("grain_window_start", 0.5))
    window_start_frac = max(0.0, min(1.0, window_start_frac))
    freeze_center = int(window_start_frac * (source_len - grain_samples))

    out_pos = 0
    grain_idx = 0
    while out_pos < n_samples and grain_idx < est_n_grains:
        size_mult = 1.0 + size_jitter[grain_idx]
        this_grain_samples = max(2, int(grain_samples * size_mult))
        this_window = (
            window
            if this_grain_samples == grain_samples
            else _window(shape=window_shape, n=this_grain_samples)
        )

        if grain_type == "time_freeze":
            read_spread = int(jitter * 0.5 * this_grain_samples)
            read_jitter = int(position_jitter[grain_idx] * read_spread)
            read_pos = (freeze_center + read_jitter) % source_len
        else:
            read_pos = int((position_jitter[grain_idx] + 1.0) * 0.5 * source_len)

        rate = _pick_grain_pitch(
            base_pitch_ratio=base_pitch_ratio,
            pitch_spread=pitch_spread,
            jitter_direction=pitch_jitter[grain_idx],
            ji_lattice=ji_lattice,
        )

        src_indices = read_pos + rate * np.arange(this_grain_samples, dtype=np.float64)
        src_indices_wrapped = src_indices % source_len
        base_idx = np.floor(src_indices_wrapped).astype(np.int64)
        frac = src_indices_wrapped - base_idx
        next_idx = (base_idx + 1) % source_len
        grain = (1.0 - frac) * source_buffer[base_idx] + frac * source_buffer[next_idx]
        grain *= this_window

        start = out_pos
        end = min(n_samples, start + this_grain_samples)
        if end > start:
            out[start:end] += grain[: end - start]

        gap = mean_gap * (1.0 + gap_jitter[grain_idx])
        out_pos += max(1, int(gap))
        grain_idx += 1

    peak = float(np.max(np.abs(out)))
    if peak > 0.0:
        out = out / peak
    return out
