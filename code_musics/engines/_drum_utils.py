"""Shared DSP helpers for drum/percussion synthesis engines.

Consolidates velocity-timbre scaling, narrow bandpass noise, and phase
integration that were previously duplicated across kick_tom, snare, clap,
noise_perc, and metallic_perc engines.

Deterministic RNG seeding and the windowed bandpass noise variant are
canonical in :mod:`code_musics.engines._dsp_utils` and re-exported here
for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from code_musics.engines._dsp_utils import bandpass_noise as bandpass_noise_windowed
from code_musics.engines._dsp_utils import rng_for_note

# Re-export so existing ``from _drum_utils import rng_for_note`` keeps working.
__all__ = ["rng_for_note", "bandpass_noise_windowed"]


@dataclass(frozen=True)
class TimbreScaling:
    """Computed timbre modifiers for a given amplitude/velocity level."""

    decay_scale: float  # multiply decay times by this (1.0 = no change)
    brightness_scale: float  # multiply filter cutoff / brightness by this
    harmonic_scale: float  # multiply FM index / overtone / distortion amount
    noise_balance: float  # additive offset to noise/tonal balance (0.0 = no change)


_NEUTRAL_TIMBRE = TimbreScaling(
    decay_scale=1.0, brightness_scale=1.0, harmonic_scale=1.0, noise_balance=0.0
)

_VELOCITY_TIMBRE_KEYS = frozenset(
    {
        "velocity_timbre_decay",
        "velocity_timbre_brightness",
        "velocity_timbre_harmonics",
        "velocity_timbre_noise",
    }
)


def resolve_velocity_timbre(amp: float, params: dict[str, Any]) -> TimbreScaling:
    """Compute timbre scaling from amplitude and optional velocity_timbre_* params.

    Returns neutral scaling when no velocity_timbre_* params are present (zero cost path).
    """
    if not _VELOCITY_TIMBRE_KEYS & params.keys():
        return _NEUTRAL_TIMBRE

    deviation = amp - 1.0

    decay_sens = float(params.get("velocity_timbre_decay", 0.0))
    brightness_sens = float(params.get("velocity_timbre_brightness", 0.0))
    harmonic_sens = float(params.get("velocity_timbre_harmonics", 0.0))
    noise_sens = float(params.get("velocity_timbre_noise", 0.0))

    decay_scale = max(0.25, min(4.0, 1.0 + decay_sens * deviation))
    brightness_scale = max(0.25, min(4.0, 1.0 + brightness_sens * deviation))
    harmonic_scale = max(0.25, min(4.0, 1.0 + harmonic_sens * deviation))
    noise_balance = max(-0.5, min(0.5, noise_sens * deviation))

    return TimbreScaling(
        decay_scale=decay_scale,
        brightness_scale=brightness_scale,
        harmonic_scale=harmonic_scale,
        noise_balance=noise_balance,
    )


def bandpass_noise(
    signal: np.ndarray,
    *,
    sample_rate: int,
    center_hz: float,
) -> np.ndarray:
    """FFT-domain Gaussian bandpass shaping (narrow variant).

    Used by kick_tom, snare, and metallic_perc engines.  For the wider
    variant with explicit ``width_ratio`` and hard band edges, see
    :func:`bandpass_noise_windowed` (canonical in ``_dsp_utils``).
    """
    nyquist = sample_rate / 2.0
    bounded_center_hz = float(np.clip(center_hz, 40.0, nyquist * 0.95))
    width_hz = max(140.0, bounded_center_hz * 0.8)
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    mask = np.exp(-0.5 * ((freqs - bounded_center_hz) / max(1.0, width_hz / 2.7)) ** 2)
    return np.fft.irfft(spectrum * mask, n=signal.size).real


def integrated_phase(freq_profile: np.ndarray, *, sample_rate: int) -> np.ndarray:
    """Cumulative phase from a per-sample frequency profile."""
    phase_increment = 2.0 * np.pi * freq_profile / sample_rate
    return np.cumsum(phase_increment)
