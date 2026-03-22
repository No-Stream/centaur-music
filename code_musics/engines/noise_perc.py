"""Hybrid noise-and-tone percussion engine."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import numpy as np


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
    # noise_decay controls the noise body envelope independently from pitch_decay.
    # Defaults to pitch_decay when omitted (backward-compatible with old presets).
    noise_decay = float(params.get("noise_decay", pitch_decay))
    bandpass_ratio = float(params.get("bandpass_ratio", 1.0))
    bandpass_width_ratio = float(params.get("bandpass_width_ratio", 0.75))
    click_amount = float(params.get("click_amount", 0.12))

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

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0)

    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    tone_env = np.exp(-t / tone_decay)

    tone = np.sin(2.0 * np.pi * freq * t) * tone_env
    tone += (
        0.35 * np.sin(2.0 * np.pi * freq * 2.0 * t) * np.exp(-t / (tone_decay * 0.7))
    )

    rng = _rng_for_note(
        freq=freq, duration=duration, amp=amp, sample_rate=sample_rate, params=params
    )

    noise = rng.standard_normal(n_samples)
    noise = _bandpass_noise(
        noise,
        sample_rate=sample_rate,
        center_hz=freq * bandpass_ratio,
        width_ratio=bandpass_width_ratio,
    )
    noise_env = np.exp(-t / noise_decay)
    noise *= noise_env

    click = _click_envelope(n_samples) * rng.standard_normal(n_samples)
    click = _bandpass_noise(
        click, sample_rate=sample_rate, center_hz=min(sample_rate * 0.25, freq * 3.0)
    )

    signal = (1.0 - noise_mix) * tone + noise_mix * noise + click_amount * click
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak
    return amp * signal.astype(np.float64)


def _click_envelope(n_samples: int) -> np.ndarray:
    """Short asymmetric burst for the initial transient."""
    if n_samples == 0:
        return np.zeros(0)
    envelope = np.exp(
        -np.arange(n_samples, dtype=np.float64) / max(1.0, n_samples * 0.015)
    )
    envelope[: max(1, n_samples // 256)] *= 2.5
    return envelope


def _bandpass_noise(
    signal: np.ndarray,
    *,
    sample_rate: int,
    center_hz: float,
    width_ratio: float = 0.75,
) -> np.ndarray:
    """Shape white noise with a spectral band around `center_hz`.

    Args:
        width_ratio: Width of the band as a multiple of `center_hz`.
            Default 0.75 gives a moderately focused band. Higher values
            (e.g. 2.5–3.0) produce broad, natural-sounding noise.
    """
    if signal.size == 0:
        return signal

    nyquist = sample_rate / 2.0
    center_hz = float(np.clip(center_hz, 30.0, nyquist * 0.95))
    width_hz = max(80.0, center_hz * width_ratio)
    low_hz = max(20.0, center_hz - width_hz / 2.0)
    high_hz = min(nyquist * 0.98, center_hz + width_hz / 2.0)

    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    mask = np.exp(-0.5 * ((freqs - center_hz) / max(1.0, width_hz / 2.5)) ** 2)
    mask *= (freqs >= low_hz).astype(np.float64)
    mask *= (freqs <= high_hz).astype(np.float64)
    shaped = np.fft.irfft(spectrum * mask, n=signal.size)
    return shaped.real


def _rng_for_note(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
) -> np.random.Generator:
    """Build a deterministic RNG seed from the render inputs."""
    seed_material = repr(
        (
            round(freq, 6),
            round(duration, 6),
            round(amp, 6),
            sample_rate,
            tuple(sorted(params.items())),
        )
    ).encode("utf-8")
    seed_bytes = sha256(seed_material).digest()[:8]
    seed = int.from_bytes(seed_bytes, byteorder="big", signed=False)
    return np.random.default_rng(seed)
