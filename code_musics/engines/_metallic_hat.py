"""Noise-forward metallic hat voicing shared by drum engines."""

from __future__ import annotations

from typing import Any

import numpy as np

from code_musics.engines._drum_utils import integrated_phase
from code_musics.engines._filters import apply_zdf_svf
from code_musics.engines._oscillators import polyblep_square

_DEFAULT_HAT_RATIOS: tuple[float, ...] = (
    1.00,
    1.17,
    1.31,
    1.47,
    1.64,
    1.82,
    2.03,
    2.27,
    2.53,
    2.81,
)

_VALID_HAT_OSCILLATOR_MODES = {"sine", "square"}


def render_hat_noise_metallic(
    *,
    n_samples: int,
    freq_profile: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    params: dict[str, Any],
    prefix: str,
) -> np.ndarray:
    """Render a hi-hat metallic layer as noisy metal wash, not a small bell bank.

    The regular ``partials`` metallic path is deliberately resonant and useful
    for bells, cowbells, claves, and gamelan colors. Hats need the opposite:
    many close, unstable sources with fast modal decay, plus a broadband
    high-frequency noise bed that dominates the tail.
    """
    if n_samples == 0:
        return np.zeros(0, dtype=np.float64)

    raw_ratios = params.get(f"{prefix}partial_ratios")
    n_partials = int(params.get(f"{prefix}n_partials", 24))
    brightness = float(params.get(f"{prefix}brightness", 0.55))
    density = float(params.get(f"{prefix}density", 0.85))
    oscillator_mode = str(params.get(f"{prefix}oscillator_mode", "square")).lower()
    detune_cents = float(params.get(f"{prefix}hat_detune_cents", 42.0))
    noise_mix = float(params.get(f"{prefix}hat_noise_mix", 0.72))
    partial_decay_spread = float(params.get(f"{prefix}hat_partial_decay_spread", 0.78))
    bank_hp_hz = float(params.get(f"{prefix}hat_bank_hp_hz", 3_600.0))
    noise_hp_hz = float(params.get(f"{prefix}hat_noise_hp_hz", 3_200.0))
    noise_bp_hz = float(params.get(f"{prefix}hat_noise_bp_hz", 7_500.0))
    noise_bp_q = float(params.get(f"{prefix}hat_noise_bp_q", 0.75))

    if n_partials <= 0:
        raise ValueError(f"{prefix}n_partials must be positive")
    if not 0.0 <= brightness <= 1.0:
        raise ValueError(f"{prefix}brightness must be between 0 and 1")
    if not 0.0 <= density <= 1.0:
        raise ValueError(f"{prefix}density must be between 0 and 1")
    if oscillator_mode not in _VALID_HAT_OSCILLATOR_MODES:
        raise ValueError(
            f"{prefix}oscillator_mode must be one of "
            f"{sorted(_VALID_HAT_OSCILLATOR_MODES)}, got {oscillator_mode!r}"
        )
    if detune_cents < 0.0:
        raise ValueError(f"{prefix}hat_detune_cents must be non-negative")
    if not 0.0 <= noise_mix <= 1.0:
        raise ValueError(f"{prefix}hat_noise_mix must be between 0 and 1")
    if not 0.0 <= partial_decay_spread <= 1.0:
        raise ValueError(f"{prefix}hat_partial_decay_spread must be between 0 and 1")
    if bank_hp_hz <= 0.0 or noise_hp_hz <= 0.0 or noise_bp_hz <= 0.0:
        raise ValueError(f"{prefix}hat filter frequencies must be positive")
    if noise_bp_q < 0.5:
        raise ValueError(f"{prefix}hat_noise_bp_q must be >= 0.5")

    base_ratios = (
        [float(r) for r in raw_ratios]
        if raw_ratios is not None
        else list(_DEFAULT_HAT_RATIOS)
    )
    if any(r <= 0.0 for r in base_ratios):
        raise ValueError(f"all {prefix}partial_ratios must be positive")

    ratios = _expand_hat_ratios(
        base_ratios=base_ratios,
        n_partials=n_partials,
        detune_cents=detune_cents,
        density=density,
        rng=rng,
    )
    partial_cloud = _render_partial_cloud(
        ratios=ratios,
        freq_profile=freq_profile,
        sample_rate=sample_rate,
        brightness=brightness,
        oscillator_mode=oscillator_mode,
        partial_decay_spread=partial_decay_spread,
    )
    partial_cloud = _highpass(
        partial_cloud, sample_rate=sample_rate, cutoff_hz=bank_hp_hz
    )

    raw_noise = rng.standard_normal(n_samples)
    noise = _highpass(raw_noise, sample_rate=sample_rate, cutoff_hz=noise_hp_hz)
    noise = apply_zdf_svf(
        noise,
        cutoff_profile=noise_bp_hz,
        resonance_q=noise_bp_q,
        sample_rate=sample_rate,
        filter_mode="bandpass",
        filter_drive=0.0,
    )

    partial_cloud = _normalize_rms(partial_cloud)
    noise = _normalize_rms(noise)
    signal = (1.0 - noise_mix) * partial_cloud + noise_mix * noise
    peak = float(np.max(np.abs(signal)))
    if peak > 1e-12:
        signal = signal / peak
    return signal


def _expand_hat_ratios(
    *,
    base_ratios: list[float],
    n_partials: int,
    detune_cents: float,
    density: float,
    rng: np.random.Generator,
) -> list[float]:
    detune_factors = [
        2.0 ** (-detune_cents / 1200.0),
        1.0,
        2.0 ** (detune_cents / 1200.0),
    ]
    ratios: list[float] = []
    base_index = 0
    while len(ratios) < n_partials:
        base_ratio = base_ratios[base_index % len(base_ratios)]
        for detune_factor in detune_factors:
            if len(ratios) >= n_partials:
                break
            random_spread = 1.0 + density * 0.045 * rng.uniform(-1.0, 1.0)
            ratios.append(base_ratio * detune_factor * random_spread)
        base_index += 1
    return ratios


def _render_partial_cloud(
    *,
    ratios: list[float],
    freq_profile: np.ndarray,
    sample_rate: int,
    brightness: float,
    oscillator_mode: str,
    partial_decay_spread: float,
) -> np.ndarray:
    n_samples = freq_profile.size
    time = np.arange(n_samples, dtype=np.float64) / sample_rate
    base_freq = float(np.mean(freq_profile[: max(1, n_samples // 10)]))
    nyquist = sample_rate / 2.0
    base_decay_s = max(1e-4, n_samples / sample_rate)
    signal = np.zeros(n_samples, dtype=np.float64)
    n_ratios = len(ratios)

    for index, ratio in enumerate(ratios):
        if base_freq * ratio >= nyquist:
            continue
        position = index / max(1, n_ratios - 1)
        weight = brightness**position if index > 0 else 1.0
        decay_scale = 1.0 - partial_decay_spread * (0.68 * position)
        partial_envelope = np.exp(-time / (base_decay_s * max(0.18, decay_scale)))
        partial_freq_profile = freq_profile * ratio
        if oscillator_mode == "square":
            norm_phase_inc = partial_freq_profile / sample_rate
            norm_cumphase = np.cumsum(norm_phase_inc)
            norm_phase = norm_cumphase % 1.0
            partial = polyblep_square(
                norm_phase, norm_phase_inc, norm_cumphase, pulse_width=0.5
            )
        else:
            phase = integrated_phase(partial_freq_profile, sample_rate=sample_rate)
            partial = np.sin(phase)
        signal += weight * partial * partial_envelope

    return signal


def _highpass(signal: np.ndarray, *, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    return apply_zdf_svf(
        signal,
        cutoff_profile=cutoff_hz,
        resonance_q=0.707,
        sample_rate=sample_rate,
        filter_mode="highpass",
        filter_drive=0.0,
    )


def _normalize_rms(signal: np.ndarray) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(signal))))
    if rms <= 1e-12:
        return signal
    return signal / rms
