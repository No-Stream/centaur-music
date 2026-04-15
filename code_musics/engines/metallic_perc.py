"""Additive/FM metallic percussion engine for hihats, cymbals, cowbell, and clave."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from code_musics.engines._drum_utils import (
    bandpass_noise,
    resolve_velocity_timbre,
    rng_for_note,
)
from code_musics.engines._envelopes import render_envelope
from code_musics.engines._filters import _SUPPORTED_FILTER_MODES, apply_zdf_svf
from code_musics.engines.polyblep import _polyblep_square

logger: logging.Logger = logging.getLogger(__name__)

_VALID_METALLIC_FILTER_MODES = _SUPPORTED_FILTER_MODES - {"notch"}

# Authentic 808 hihat/cymbal Schmitt-trigger oscillator frequency ratios.
_808_RATIOS: list[float] = [1.0, 1.3348, 1.4755, 1.6818, 1.9307, 2.5452]

_VALID_OSCILLATOR_MODES = {"sine", "square"}


def render(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
    freq_trajectory: np.ndarray | None = None,
) -> np.ndarray:
    """Render a metallic percussion voice with inharmonic additive partials.

    DSP chain: N sine partials at non-integer frequency ratios -> optional ring
    modulation -> ZDF filter -> decay envelope -> transient click layer -> mix
    and peak-normalize.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if freq <= 0:
        raise ValueError("freq must be positive")
    if freq_trajectory is not None:
        raise ValueError("pitch motion is not supported for the metallic_perc engine")

    # --- Parse and validate parameters ---
    n_partials = int(params.get("n_partials", 6))
    raw_ratios = params.get("partial_ratios")
    oscillator_mode = str(params.get("oscillator_mode", "sine")).lower()
    brightness = float(params.get("brightness", 0.7))
    decay_ms = float(params.get("decay_ms", 80.0))
    ring_mod_amount = float(params.get("ring_mod_amount", 0.0))
    ring_mod_freq_ratio = float(params.get("ring_mod_freq_ratio", 1.48))
    filter_center_ratio = float(params.get("filter_center_ratio", 1.0))
    filter_q = float(params.get("filter_q", 1.2))
    filter_mode = str(params.get("filter_mode", "bandpass"))
    click_amount = float(params.get("click_amount", 0.05))
    click_decay_s = float(params.get("click_decay", 0.003))
    noise_amount = float(params.get("noise_amount", 0.0))
    density = float(params.get("density", 0.5))

    amp_envelope_raw = params.get("amp_envelope")
    filter_envelope_raw = params.get("filter_envelope")

    if n_partials <= 0:
        raise ValueError("n_partials must be positive")
    if oscillator_mode not in _VALID_OSCILLATOR_MODES:
        raise ValueError(
            f"oscillator_mode must be one of {sorted(_VALID_OSCILLATOR_MODES)}, "
            f"got {oscillator_mode!r}"
        )
    if not 0.0 <= brightness <= 1.0:
        raise ValueError("brightness must be between 0 and 1")
    if decay_ms <= 0:
        raise ValueError("decay_ms must be positive")
    if not 0.0 <= ring_mod_amount <= 1.0:
        raise ValueError("ring_mod_amount must be between 0 and 1")
    if ring_mod_freq_ratio <= 0:
        raise ValueError("ring_mod_freq_ratio must be positive")
    if filter_center_ratio <= 0:
        raise ValueError("filter_center_ratio must be positive")
    if filter_q < 0.5:
        raise ValueError("filter_q must be >= 0.5")
    if filter_mode not in _VALID_METALLIC_FILTER_MODES:
        raise ValueError(
            f"filter_mode must be one of {sorted(_VALID_METALLIC_FILTER_MODES)}, "
            f"got {filter_mode!r}"
        )
    if click_amount < 0:
        raise ValueError("click_amount must be non-negative")
    if click_decay_s <= 0:
        raise ValueError("click_decay must be positive")
    if noise_amount < 0:
        raise ValueError("noise_amount must be non-negative")
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be between 0 and 1")

    # --- velocity-to-timbre scaling ---
    timbre = resolve_velocity_timbre(amp, params)
    decay_ms *= timbre.decay_scale
    filter_center_ratio *= timbre.brightness_scale
    brightness = min(1.0, brightness * timbre.brightness_scale)
    ring_mod_amount = min(1.0, ring_mod_amount * timbre.harmonic_scale)

    # --- Resolve partial ratios ---
    if raw_ratios is not None:
        partial_ratios = [float(r) for r in raw_ratios]
        if any(r <= 0 for r in partial_ratios):
            raise ValueError("all partial_ratios must be positive")
    else:
        partial_ratios = [math.sqrt(float(i + 1)) for i in range(n_partials)]

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0, dtype=np.float64)

    time = np.arange(n_samples, dtype=np.float64) / sample_rate

    # --- Deterministic RNG for density jitter and click noise ---
    rng = rng_for_note(
        freq=freq, duration=duration, amp=amp, sample_rate=sample_rate, params=params
    )

    # --- Apply density jitter to partial ratios ---
    jittered_ratios = list(partial_ratios)
    if density > 0:
        for i in range(len(jittered_ratios)):
            jitter = density * 0.03 * rng.uniform(-1.0, 1.0)
            jittered_ratios[i] = jittered_ratios[i] * (1.0 + jitter)

    # --- Additive partials with phase accumulation ---
    nyquist = sample_rate / 2.0
    num_partials = len(jittered_ratios)
    partials_sum = np.zeros(n_samples, dtype=np.float64)
    for i, ratio in enumerate(jittered_ratios):
        partial_freq = freq * ratio
        if partial_freq >= nyquist:
            continue  # skip aliasing partials
        weight = brightness ** (i / max(1, num_partials - 1)) if i > 0 else 1.0
        if oscillator_mode == "square":
            # PolyBLEP uses normalized [0,1) phase, not radians.
            norm_phase_inc = np.full(
                n_samples, partial_freq / sample_rate, dtype=np.float64
            )
            norm_cumphase = np.cumsum(norm_phase_inc)
            norm_phase = norm_cumphase % 1.0
            partials_sum += weight * _polyblep_square(
                norm_phase, norm_phase_inc, norm_cumphase, pulse_width=0.5
            )
        else:
            phase_inc = 2.0 * np.pi * partial_freq / sample_rate
            phase = np.cumsum(np.full(n_samples, phase_inc, dtype=np.float64))
            partials_sum += weight * np.sin(phase)

    # --- Optional ring modulation ---
    if ring_mod_amount > 0:
        ring_freq = freq * ring_mod_freq_ratio
        ring_phase_inc = 2.0 * np.pi * ring_freq / sample_rate
        ring_phase = np.cumsum(np.full(n_samples, ring_phase_inc, dtype=np.float64))
        ring_signal = np.sin(ring_phase)
        partials_sum = (1.0 - ring_mod_amount) * partials_sum + ring_mod_amount * (
            partials_sum * ring_signal
        )

    # --- Filter ---
    max_safe_cutoff = sample_rate * 0.4
    cutoff_hz = min(freq * filter_center_ratio, max_safe_cutoff)
    if filter_envelope_raw is not None:
        cutoff_profile = render_envelope(
            filter_envelope_raw, n_samples, default_value=cutoff_hz
        )
    else:
        cutoff_profile = np.full(n_samples, cutoff_hz, dtype=np.float64)
    partials_sum = apply_zdf_svf(
        partials_sum,
        cutoff_profile=cutoff_profile,
        resonance_q=filter_q,
        sample_rate=sample_rate,
        filter_mode=filter_mode,
        filter_drive=0.0,
    )

    # --- Amplitude envelope: exponential decay with 0.1ms attack ramp ---
    decay_seconds = decay_ms / 1000.0
    if amp_envelope_raw is not None:
        envelope = render_envelope(amp_envelope_raw, n_samples, default_value=0.0)
    else:
        envelope = np.exp(-time / decay_seconds)
    attack_samples = max(1, int(0.0001 * sample_rate))
    envelope[:attack_samples] *= np.linspace(0.0, 1.0, attack_samples)

    # --- Broadband noise layer (follows the main decay envelope) ---
    noise_layer = np.zeros(n_samples, dtype=np.float64)
    if noise_amount > 0:
        raw_noise = rng.standard_normal(n_samples)
        noise_layer = bandpass_noise(
            raw_noise, sample_rate=sample_rate, center_hz=min(freq, sample_rate * 0.35)
        )
        noise_layer *= envelope  # same decay shape as partials

    # --- Transient click: short filtered noise burst ---
    click_noise = rng.standard_normal(n_samples)
    click_noise = bandpass_noise(
        click_noise,
        sample_rate=sample_rate,
        center_hz=min(freq * 2.0, sample_rate * 0.35),
    )
    click_env = np.exp(-time / click_decay_s)
    click = click_noise * click_env

    # --- Mix, normalize, scale by amp ---
    signal = (
        (partials_sum * envelope)
        + (noise_amount * noise_layer)
        + (click_amount * click)
    )

    peak = np.max(np.abs(signal))
    if peak > 1e-9:
        signal = signal / peak
    return (amp * signal).astype(np.float64)
