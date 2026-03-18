"""Frequency modulation synthesis engine."""

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
    """Render a compact 2-operator FM voice.

    The implementation is frequency-first: the carrier and modulator operate
    directly on Hertz-derived phase increments, which keeps it compatible with
    alternate tuning systems.
    """
    if duration <= 0:
        return np.zeros(0)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if freq <= 0:
        raise ValueError("freq must be positive")

    carrier_ratio = float(params.get("carrier_ratio", 1.0))
    mod_ratio = float(params.get("mod_ratio", 1.0))
    mod_index = float(params.get("mod_index", 1.5))
    feedback = float(params.get("feedback", 0.0))
    index_decay = float(params.get("index_decay", 0.0))
    index_sustain = float(params.get("index_sustain", 0.5))

    if carrier_ratio <= 0:
        raise ValueError("carrier_ratio must be positive")
    if mod_ratio <= 0:
        raise ValueError("mod_ratio must be positive")
    if mod_index < 0:
        raise ValueError("mod_index must be non-negative")
    if index_decay < 0:
        raise ValueError("index_decay must be non-negative")

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0)

    if freq_trajectory is None:
        carrier_phase_increment = np.full(
            n_samples, 2.0 * np.pi * freq * carrier_ratio / sample_rate, dtype=np.float64
        )
        mod_phase_increment = np.full(
            n_samples, 2.0 * np.pi * freq * mod_ratio / sample_rate, dtype=np.float64
        )
    else:
        freq_trajectory = np.asarray(freq_trajectory, dtype=np.float64)
        if freq_trajectory.ndim != 1:
            raise ValueError("freq_trajectory must be one-dimensional")
        if freq_trajectory.size != n_samples:
            raise ValueError("freq_trajectory length must match note duration")
        carrier_phase_increment = 2.0 * np.pi * freq_trajectory * carrier_ratio / sample_rate
        mod_phase_increment = 2.0 * np.pi * freq_trajectory * mod_ratio / sample_rate

    signal = np.empty(n_samples, dtype=np.float64)
    carrier_phase = 0.0
    mod_phase = 0.0
    previous_feedback_sample = 0.0

    index_decay_samples = int(index_decay * sample_rate)
    index_decay_samples = min(max(index_decay_samples, 0), n_samples)
    sustain_scale = max(0.0, index_sustain)

    for sample_index in range(n_samples):
        modulation_envelope = _modulation_envelope(
            sample_index=sample_index,
            decay_samples=index_decay_samples,
            sustain_scale=sustain_scale,
        )
        current_index = mod_index * modulation_envelope

        modulator_sample = np.sin(mod_phase + feedback * previous_feedback_sample)
        previous_feedback_sample = modulator_sample

        carrier_sample = np.sin(carrier_phase + current_index * modulator_sample)
        signal[sample_index] = carrier_sample

        carrier_phase += carrier_phase_increment[sample_index]
        mod_phase += mod_phase_increment[sample_index]

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak
    return amp * signal


def _modulation_envelope(
    *,
    sample_index: int,
    decay_samples: int,
    sustain_scale: float,
) -> float:
    """Return the modulation-index envelope multiplier for a sample."""
    if decay_samples <= 0:
        return sustain_scale
    if sample_index >= decay_samples:
        return sustain_scale

    decay_progress = sample_index / max(1, decay_samples - 1)
    return 1.0 + (sustain_scale - 1.0) * decay_progress
