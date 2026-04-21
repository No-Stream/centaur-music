"""Hybrid noise-and-tone percussion engine."""

from __future__ import annotations

from typing import Any

import numpy as np

from code_musics.engines._drum_utils import (
    bandpass_noise_windowed,
    resolve_velocity_timbre,
    rng_for_note,
)
from code_musics.engines._envelopes import render_envelope


def render(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
    freq_trajectory: np.ndarray | None = None,
) -> np.ndarray:
    """Render a short hybrid percussion voice.

    The sound combines a pitched strike, a noise burst, and a small click layer.
    It stays frequency-first so it can be driven from the same score model as the
    melodic engines.

    Optional multi-point envelope params:

    - ``tone_amp_envelope``: replaces the exponential tonal component decay.
    - ``noise_amp_envelope``: replaces the exponential noise component decay.
    - ``overall_amp_envelope``: applied to the final mixed signal before
      peak normalization.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if freq <= 0:
        raise ValueError("freq must be positive")
    if freq_trajectory is not None:
        raise ValueError("pitch motion is not supported for the noise_perc engine")

    noise_mix = float(params.get("noise_mix", 0.5))
    pitch_decay = float(params.get("pitch_decay", 0.08))
    tone_decay = float(params.get("tone_decay", 0.18))
    noise_decay = float(params.get("noise_decay", pitch_decay))
    bandpass_ratio = float(params.get("bandpass_ratio", 1.0))
    bandpass_width_ratio = float(params.get("bandpass_width_ratio", 0.75))
    click_amount = float(params.get("click_amount", 0.12))
    click_attack = float(params.get("click_attack", 0.0002))

    tone_amp_envelope_raw = params.get("tone_amp_envelope")
    noise_amp_envelope_raw = params.get("noise_amp_envelope")
    overall_amp_envelope_raw = params.get("overall_amp_envelope")

    if not 0.0 <= noise_mix <= 1.0:
        raise ValueError("noise_mix must be between 0 and 1")
    if pitch_decay <= 0 or tone_decay <= 0:
        raise ValueError("pitch_decay and tone_decay must be positive")
    if noise_decay <= 0:
        raise ValueError("noise_decay must be positive")
    if bandpass_width_ratio <= 0:
        raise ValueError("bandpass_width_ratio must be positive")
    if bandpass_ratio <= 0:
        raise ValueError("bandpass_ratio must be positive")
    if click_amount < 0:
        raise ValueError("click_amount must be non-negative")
    if click_attack < 0:
        raise ValueError("click_attack must be non-negative")

    # --- velocity-to-timbre scaling ---
    timbre = resolve_velocity_timbre(amp, params)
    tone_decay *= timbre.decay_scale
    noise_decay *= timbre.decay_scale
    noise_mix = max(0.0, min(1.0, noise_mix + timbre.noise_balance))

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0)

    t = np.arange(n_samples, dtype=np.float64) / sample_rate

    if tone_amp_envelope_raw is not None:
        tone_env = render_envelope(tone_amp_envelope_raw, n_samples, default_value=0.0)
    else:
        tone_env = np.exp(-t / tone_decay)

    tone = np.sin(2.0 * np.pi * freq * t) * tone_env
    tone += 0.35 * np.sin(2.0 * np.pi * freq * 2.0 * t) * tone_env**1.4

    rng = rng_for_note(
        freq=freq, duration=duration, amp=amp, sample_rate=sample_rate, params=params
    )

    noise = rng.standard_normal(n_samples)
    noise = bandpass_noise_windowed(
        noise,
        sample_rate=sample_rate,
        center_hz=freq * bandpass_ratio,
        width_ratio=bandpass_width_ratio,
    )
    if noise_amp_envelope_raw is not None:
        noise_env = render_envelope(
            noise_amp_envelope_raw, n_samples, default_value=0.0
        )
    else:
        noise_env = np.exp(-t / noise_decay)
    noise *= noise_env

    click_attack_samples = int(click_attack * sample_rate)
    click = _click_envelope(
        n_samples, attack_samples=click_attack_samples
    ) * rng.standard_normal(n_samples)
    click = bandpass_noise_windowed(
        click, sample_rate=sample_rate, center_hz=min(sample_rate * 0.25, freq * 3.0)
    )

    signal = (1.0 - noise_mix) * tone + noise_mix * noise + click_amount * click

    # --- optional overall amplitude envelope ---
    if overall_amp_envelope_raw is not None:
        overall_env = render_envelope(
            overall_amp_envelope_raw, n_samples, default_value=1.0
        )
        signal *= overall_env

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak
    return amp * signal.astype(np.float64)


def _click_envelope(n_samples: int, *, attack_samples: int = 0) -> np.ndarray:
    """Short asymmetric burst for the initial transient."""
    if n_samples == 0:
        return np.zeros(0)
    envelope = np.exp(
        -np.arange(n_samples, dtype=np.float64) / max(1.0, n_samples * 0.015)
    )
    envelope[: max(1, n_samples // 256)] *= 2.5
    if attack_samples > 0:
        ramp_len = min(attack_samples, n_samples)
        envelope[:ramp_len] *= np.linspace(0.0, 1.0, ramp_len)
    return envelope
