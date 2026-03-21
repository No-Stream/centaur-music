"""Hybrid kick/tom drum synthesis engine."""

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
    """Render an electro-style kick/tom voice with internal sweep and transient."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if freq <= 0:
        raise ValueError("freq must be positive")

    decay_ms = float(params.get("body_decay_ms", params.get("decay_ms", 260.0)))
    pitch_sweep_amount_ratio = float(params.get("pitch_sweep_amount_ratio", 2.5))
    pitch_sweep_decay_ms = float(params.get("pitch_sweep_decay_ms", 42.0))
    body_wave = str(params.get("body_wave", "sine")).lower()
    body_tone_ratio = float(params.get("body_tone_ratio", 0.16))
    body_punch_ratio = float(params.get("body_punch_ratio", 0.20))
    overtone_amount = float(params.get("overtone_amount", 0.10))
    overtone_ratio = float(params.get("overtone_ratio", 1.9))
    overtone_decay_ms = float(params.get("overtone_decay_ms", 110.0))
    click_amount = float(params.get("click_amount", 0.08))
    click_decay_ms = float(params.get("click_decay_ms", 7.0))
    click_tone_hz = float(params.get("click_tone_hz", 3_200.0))
    noise_amount = float(params.get("noise_amount", 0.02))
    noise_decay_ms = float(params.get("noise_decay_ms", 28.0))
    noise_bandpass_hz = float(params.get("noise_bandpass_hz", 1_100.0))
    drive_ratio = float(params.get("drive_ratio", 0.10))
    post_lowpass_hz = float(params.get("post_lowpass_hz", 14_000.0))

    if decay_ms <= 0:
        raise ValueError("decay_ms must be positive")
    if pitch_sweep_amount_ratio <= 0:
        raise ValueError("pitch_sweep_amount_ratio must be positive")
    if pitch_sweep_decay_ms <= 0:
        raise ValueError("pitch_sweep_decay_ms must be positive")
    if body_wave not in {"sine", "triangle", "sine_clip"}:
        raise ValueError("body_wave must be 'sine', 'triangle', or 'sine_clip'")
    if not 0.0 <= body_tone_ratio <= 1.0:
        raise ValueError("body_tone_ratio must be between 0 and 1")
    if body_punch_ratio < 0:
        raise ValueError("body_punch_ratio must be non-negative")
    if overtone_amount < 0:
        raise ValueError("overtone_amount must be non-negative")
    if overtone_ratio <= 0:
        raise ValueError("overtone_ratio must be positive")
    if overtone_decay_ms <= 0:
        raise ValueError("overtone_decay_ms must be positive")
    if click_amount < 0:
        raise ValueError("click_amount must be non-negative")
    if click_decay_ms <= 0:
        raise ValueError("click_decay_ms must be positive")
    if click_tone_hz <= 0:
        raise ValueError("click_tone_hz must be positive")
    if noise_amount < 0:
        raise ValueError("noise_amount must be non-negative")
    if noise_decay_ms <= 0:
        raise ValueError("noise_decay_ms must be positive")
    if noise_bandpass_hz <= 0:
        raise ValueError("noise_bandpass_hz must be positive")
    if not 0.0 <= drive_ratio <= 1.0:
        raise ValueError("drive_ratio must be between 0 and 1")
    if post_lowpass_hz <= 0:
        raise ValueError("post_lowpass_hz must be positive")

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0, dtype=np.float64)

    time = np.arange(n_samples, dtype=np.float64) / sample_rate
    body_decay_seconds = decay_ms / 1000.0
    sweep_decay_seconds = pitch_sweep_decay_ms / 1000.0
    overtone_decay_seconds = overtone_decay_ms / 1000.0
    click_decay_seconds = click_decay_ms / 1000.0
    noise_decay_seconds = noise_decay_ms / 1000.0

    base_freq_profile = _resolve_base_freq_profile(
        freq=freq,
        n_samples=n_samples,
        freq_trajectory=freq_trajectory,
    )
    sweep_profile = 1.0 + (pitch_sweep_amount_ratio - 1.0) * np.exp(
        -time / sweep_decay_seconds
    )
    freq_profile = base_freq_profile * sweep_profile

    fundamental_phase = _integrated_phase(freq_profile, sample_rate=sample_rate)
    fundamental = _oscillator(body_wave=body_wave, phase=fundamental_phase)
    body_env = np.exp(-time / body_decay_seconds)
    punch_env = 1.0 + body_punch_ratio * np.exp(-time / 0.018)
    second_harmonic = np.sin(2.0 * fundamental_phase) * np.exp(
        -time / max(0.02, body_decay_seconds * 0.55)
    )
    body = (
        (((1.0 - body_tone_ratio) * fundamental) + (body_tone_ratio * second_harmonic))
        * body_env
        * punch_env
    )

    overtone_phase = _integrated_phase(
        freq_profile * overtone_ratio, sample_rate=sample_rate
    )
    overtone = (
        overtone_amount
        * np.sin(overtone_phase)
        * np.exp(-time / overtone_decay_seconds)
    )

    rng = _rng_for_note(
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=sample_rate,
        params=params,
    )
    click = _transient_noise(
        rng=rng,
        n_samples=n_samples,
        sample_rate=sample_rate,
        center_hz=click_tone_hz,
        decay_seconds=click_decay_seconds,
        emphasis=2.4,
    )
    noise = _transient_noise(
        rng=rng,
        n_samples=n_samples,
        sample_rate=sample_rate,
        center_hz=noise_bandpass_hz,
        decay_seconds=noise_decay_seconds,
        emphasis=1.0,
    )

    signal = body + overtone + (click_amount * click) + (noise_amount * noise)
    signal = _apply_drive(signal, drive_ratio=drive_ratio)
    signal = _one_pole_lowpass(
        signal,
        cutoff_hz=min(post_lowpass_hz, sample_rate * 0.48),
        sample_rate=sample_rate,
    )

    peak = np.max(np.abs(signal))
    if peak > 1e-9:
        signal = signal / peak
    return amp * signal.astype(np.float64)


def _resolve_base_freq_profile(
    *,
    freq: float,
    n_samples: int,
    freq_trajectory: np.ndarray | None,
) -> np.ndarray:
    if freq_trajectory is None:
        return np.full(n_samples, freq, dtype=np.float64)

    resolved = np.asarray(freq_trajectory, dtype=np.float64)
    if resolved.ndim != 1:
        raise ValueError("freq_trajectory must be one-dimensional")
    if resolved.size != n_samples:
        raise ValueError("freq_trajectory length must match note duration")
    if np.any(resolved <= 0):
        raise ValueError("freq_trajectory values must be positive")
    return resolved


def _integrated_phase(freq_profile: np.ndarray, *, sample_rate: int) -> np.ndarray:
    phase_increment = 2.0 * np.pi * freq_profile / sample_rate
    return np.cumsum(phase_increment)


def _oscillator(*, body_wave: str, phase: np.ndarray) -> np.ndarray:
    if body_wave == "sine":
        return np.sin(phase)
    if body_wave == "triangle":
        return (2.0 / np.pi) * np.arcsin(np.sin(phase))
    return np.tanh(1.8 * np.sin(phase))


def _transient_noise(
    *,
    rng: np.random.Generator,
    n_samples: int,
    sample_rate: int,
    center_hz: float,
    decay_seconds: float,
    emphasis: float,
) -> np.ndarray:
    raw = rng.standard_normal(n_samples)
    shaped = _bandpass_noise(raw, sample_rate=sample_rate, center_hz=center_hz)
    time = np.arange(n_samples, dtype=np.float64) / sample_rate
    envelope = np.exp(-time / decay_seconds)
    envelope[: max(1, n_samples // 512)] *= emphasis
    return shaped * envelope


def _bandpass_noise(
    signal: np.ndarray,
    *,
    sample_rate: int,
    center_hz: float,
) -> np.ndarray:
    nyquist = sample_rate / 2.0
    bounded_center_hz = float(np.clip(center_hz, 40.0, nyquist * 0.95))
    width_hz = max(140.0, bounded_center_hz * 0.8)
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    mask = np.exp(-0.5 * ((freqs - bounded_center_hz) / max(1.0, width_hz / 2.7)) ** 2)
    return np.fft.irfft(spectrum * mask, n=signal.size).real


def _apply_drive(signal: np.ndarray, *, drive_ratio: float) -> np.ndarray:
    drive = 1.0 + (7.0 * drive_ratio)
    driven = np.tanh(drive * signal)
    return driven / np.tanh(drive)


def _one_pole_lowpass(
    signal: np.ndarray,
    *,
    cutoff_hz: float,
    sample_rate: int,
) -> np.ndarray:
    if cutoff_hz >= sample_rate * 0.49:
        return signal

    dt = 1.0 / sample_rate
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    alpha = dt / (rc + dt)
    filtered = np.empty_like(signal)
    filtered[0] = alpha * signal[0]
    for sample_index in range(1, signal.size):
        filtered[sample_index] = filtered[sample_index - 1] + (
            alpha * (signal[sample_index] - filtered[sample_index - 1])
        )
    return filtered


def _rng_for_note(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
) -> np.random.Generator:
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
