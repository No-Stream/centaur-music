"""Filtered-stack synthesis engine."""

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
    """Render a harmonic-rich source shaped by low-pass style spectral weighting."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    waveform = str(params.get("waveform", "saw")).lower()
    n_harmonics = int(params.get("n_harmonics", 12))
    cutoff_ratio = float(params.get("cutoff_ratio", 8.0))
    resonance = float(params.get("resonance", 0.0))
    filter_env_amount = float(params.get("filter_env_amount", 0.0))
    filter_env_decay = float(params.get("filter_env_decay", 0.18))
    pulse_width = float(params.get("pulse_width", 0.5))

    if n_harmonics < 1:
        raise ValueError("n_harmonics must be at least 1")
    if cutoff_ratio <= 0:
        raise ValueError("cutoff_ratio must be positive")
    if filter_env_decay <= 0:
        raise ValueError("filter_env_decay must be positive")
    if not 0.0 < pulse_width < 1.0:
        raise ValueError("pulse_width must be between 0 and 1")

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0)

    t = np.linspace(0.0, duration, n_samples, endpoint=False)
    if freq_trajectory is not None:
        freq_trajectory = np.asarray(freq_trajectory, dtype=np.float64)
        if freq_trajectory.ndim != 1:
            raise ValueError("freq_trajectory must be one-dimensional")
        if freq_trajectory.size != n_samples:
            raise ValueError("freq_trajectory length must match note duration")
        freq_profile = freq_trajectory
    else:
        freq_profile = np.full(n_samples, freq, dtype=np.float64)

    signal = np.zeros(n_samples)

    cutoff_envelope = 1.0 + filter_env_amount * np.exp(-t / filter_env_decay)
    cutoff_envelope = np.maximum(cutoff_envelope, 0.05)
    cutoff_hz = freq_profile * cutoff_ratio * cutoff_envelope

    for harmonic_index in range(1, n_harmonics + 1):
        partial_freq_profile = freq_profile * harmonic_index
        if np.max(partial_freq_profile) >= sample_rate / 2:
            break

        harmonic_weight = _waveform_weight(waveform, harmonic_index, pulse_width)
        if harmonic_weight == 0.0:
            continue

        lowpass_weight = 1.0 / (
            1.0 + np.power(partial_freq_profile / np.maximum(cutoff_hz, 1e-9), 8.0)
        )

        if resonance != 0.0:
            resonance_width = np.maximum(cutoff_hz * 0.18, 1.0)
            resonance_bump = np.exp(
                -0.5 * np.square((partial_freq_profile - cutoff_hz) / resonance_width)
            )
            lowpass_weight = lowpass_weight + resonance * resonance_bump

        phase = np.cumsum(
            np.concatenate(
                [
                    np.zeros(1, dtype=np.float64),
                    2.0 * np.pi * partial_freq_profile[:-1] / sample_rate,
                ]
            )
        )
        signal += harmonic_weight * lowpass_weight * np.sin(phase)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak
    return amp * signal


def _waveform_weight(waveform: str, harmonic_index: int, pulse_width: float) -> float:
    """Return a signed harmonic weight for a basic oscillator source."""
    if waveform == "saw":
        return 1.0 / harmonic_index
    if waveform == "square":
        if harmonic_index % 2 == 0:
            return 0.0
        return 1.0 / harmonic_index
    if waveform == "pulse":
        return np.sin(np.pi * harmonic_index * pulse_width) / (
            np.pi * harmonic_index
        )
    if waveform == "triangle":
        if harmonic_index % 2 == 0:
            return 0.0
        sign = -1.0 if ((harmonic_index - 1) // 2) % 2 else 1.0
        return sign / (harmonic_index * harmonic_index)

    raise ValueError(f"Unsupported waveform: {waveform}")
