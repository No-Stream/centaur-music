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
from code_musics.engines._dsp_utils import fm_modulate
from code_musics.engines._filters import apply_zdf_svf
from code_musics.engines.kick_tom import _resonator_body
from code_musics.engines.polyblep import _polyblep_square
from code_musics.engines.snare import _comb_filter

logger: logging.Logger = logging.getLogger(__name__)

_VALID_EXCITER_TYPES = frozenset(
    {"click", "impulse", "multi_tap", "fm_burst", "noise_burst"}
)
_VALID_TONE_TYPES = frozenset({"oscillator", "resonator", "fm", "additive"})
_VALID_NOISE_TYPES = frozenset({"white", "colored", "bandpass", "comb"})
_VALID_METALLIC_TYPES = frozenset({"partials", "ring_mod", "fm_cluster"})


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
