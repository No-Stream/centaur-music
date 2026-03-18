"""Additive synthesis engine."""

from __future__ import annotations

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
    """Render a richer additive voice with compatibility defaults."""
    n_harmonics = int(params.get("n_harmonics", 6))
    harmonic_rolloff = float(params.get("harmonic_rolloff", 0.5))
    brightness_tilt = float(params.get("brightness_tilt", 0.0))
    odd_even_balance = float(params.get("odd_even_balance", 0.0))
    detune_cents = float(params.get("detune_cents", 0.0))
    unison_voices = max(1, int(params.get("unison_voices", 1)))

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0)

    if freq_trajectory is not None:
        freq_trajectory = np.asarray(freq_trajectory, dtype=np.float64)
        if freq_trajectory.ndim != 1:
            raise ValueError("freq_trajectory must be one-dimensional")
        if freq_trajectory.size != n_samples:
            raise ValueError("freq_trajectory length must match note duration")

    signal = np.zeros(n_samples)
    voice_detunes = _unison_detunes(unison_voices, detune_cents)

    for detune_offset_cents in voice_detunes:
        voice_freq = freq * (2.0 ** (detune_offset_cents / 1200.0))
        if freq_trajectory is None:
            t = np.linspace(0.0, duration, n_samples, endpoint=False)
            signal += _render_partial_bank(
                t=t,
                freq=voice_freq,
                sample_rate=sample_rate,
                n_harmonics=n_harmonics,
                harmonic_rolloff=harmonic_rolloff,
                brightness_tilt=brightness_tilt,
                odd_even_balance=odd_even_balance,
            )
        else:
            signal += _render_partial_bank_with_trajectory(
                freq_trajectory=freq_trajectory
                * (2.0 ** (detune_offset_cents / 1200.0)),
                sample_rate=sample_rate,
                n_harmonics=n_harmonics,
                harmonic_rolloff=harmonic_rolloff,
                brightness_tilt=brightness_tilt,
                odd_even_balance=odd_even_balance,
            )

    signal /= float(len(voice_detunes))
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak
    return amp * signal


def _render_partial_bank(
    *,
    t: np.ndarray,
    freq: float,
    sample_rate: int,
    n_harmonics: int,
    harmonic_rolloff: float,
    brightness_tilt: float,
    odd_even_balance: float,
) -> np.ndarray:
    signal = np.zeros_like(t)
    total_amp = 0.0
    clamped_odd_even_balance = max(-0.95, min(0.95, odd_even_balance))

    for harmonic_index in range(1, n_harmonics + 1):
        partial_freq = freq * harmonic_index
        if partial_freq >= sample_rate / 2:
            break

        partial_amp = harmonic_rolloff ** (harmonic_index - 1)
        if brightness_tilt != 0.0:
            partial_amp *= harmonic_index ** brightness_tilt

        if harmonic_index % 2 == 0:
            partial_amp *= 1.0 - clamped_odd_even_balance
        else:
            partial_amp *= 1.0 + clamped_odd_even_balance

        if partial_amp <= 0:
            continue

        signal += partial_amp * np.sin(2.0 * np.pi * partial_freq * t)
        total_amp += partial_amp

    if total_amp == 0.0:
        return signal
    return signal / total_amp


def _render_partial_bank_with_trajectory(
    *,
    freq_trajectory: np.ndarray,
    sample_rate: int,
    n_harmonics: int,
    harmonic_rolloff: float,
    brightness_tilt: float,
    odd_even_balance: float,
) -> np.ndarray:
    if freq_trajectory.ndim != 1:
        raise ValueError("freq_trajectory must be one-dimensional")
    if freq_trajectory.size == 0:
        return np.zeros(0)

    signal = np.zeros_like(freq_trajectory, dtype=np.float64)
    total_amp = 0.0
    clamped_odd_even_balance = max(-0.95, min(0.95, odd_even_balance))

    for harmonic_index in range(1, n_harmonics + 1):
        partial_freq_trajectory = freq_trajectory * harmonic_index
        if np.max(partial_freq_trajectory) >= sample_rate / 2:
            break

        partial_amp = harmonic_rolloff ** (harmonic_index - 1)
        if brightness_tilt != 0.0:
            partial_amp *= harmonic_index ** brightness_tilt

        if harmonic_index % 2 == 0:
            partial_amp *= 1.0 - clamped_odd_even_balance
        else:
            partial_amp *= 1.0 + clamped_odd_even_balance

        if partial_amp <= 0:
            continue

        phase = np.cumsum(
            np.concatenate(
                [
                    np.zeros(1, dtype=np.float64),
                    2.0 * np.pi * partial_freq_trajectory[:-1] / sample_rate,
                ]
            )
        )
        signal += partial_amp * np.sin(phase)
        total_amp += partial_amp

    if total_amp == 0.0:
        return signal
    return signal / total_amp


def _unison_detunes(unison_voices: int, detune_cents: float) -> list[float]:
    if unison_voices <= 1 or detune_cents == 0.0:
        return [0.0]
    if unison_voices == 2:
        return [-detune_cents / 2.0, detune_cents / 2.0]

    return [
        ((voice_index / (unison_voices - 1)) - 0.5) * detune_cents
        for voice_index in range(unison_voices)
    ]
