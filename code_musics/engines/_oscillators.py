"""Shared bandlimited oscillator primitives.

Time-domain PolyBLEP saw/square/triangle/sine generation, extracted from the
polyblep engine so other engines (e.g. ``va``) can reuse the bandlimited saw
bank. The rendering path is phase-aware so callers can cross-correlate phase
with sync, hard-sync, or spectral manipulation downstream.
"""

from __future__ import annotations

import numpy as np


def render_polyblep_oscillator(
    *,
    waveform: str,
    pulse_width: float | np.ndarray,
    freq_profile: np.ndarray,
    sample_rate: int,
    start_phase: float = 0.0,
    phase_noise: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a PolyBLEP oscillator, returning ``(signal, phase)``.

    ``phase`` is the wrapped 0-1 phase at each sample so callers (e.g. a
    supersaw bank with hard-sync) can inspect or re-drive downstream stages.

    ``pulse_width`` may be a scalar (legacy, default path) or a per-sample
    ``np.ndarray`` of the same length as ``freq_profile``.  Per-sample values
    are used only by the ``square`` waveform and are ignored by saw, sine,
    and triangle.

    ``phase_noise`` is an optional per-sample offset (in cycles) added to
    the integrated phase *before* wrapping and *before* the PolyBLEP
    discontinuity correction.  Callers scale the noise to their desired
    amount; a value of ``None`` (default) preserves bit-identical legacy
    behavior.
    """
    phase_inc = freq_profile / sample_rate
    cumphase = np.cumsum(phase_inc) + start_phase / (2.0 * np.pi)
    if phase_noise is not None:
        cumphase = cumphase + phase_noise
    phase = cumphase % 1.0

    if waveform == "sine":
        return np.sin(2.0 * np.pi * phase), phase
    if waveform == "saw":
        return polyblep_saw(phase, phase_inc), phase
    if waveform == "square":
        return polyblep_square(phase, phase_inc, cumphase, pulse_width), phase
    if waveform == "triangle":
        return polyblep_triangle(phase, phase_inc, cumphase, sample_rate), phase
    raise ValueError(f"Unknown waveform: {waveform!r}")


def polyblep_saw(phase: np.ndarray, phase_inc: np.ndarray) -> np.ndarray:
    """Generate a bandlimited sawtooth via PolyBLEP correction."""
    saw = 2.0 * phase - 1.0

    mask_pre = phase > (1.0 - phase_inc)
    t_pre = (phase[mask_pre] - 1.0) / phase_inc[mask_pre]
    saw[mask_pre] -= t_pre * t_pre + 2.0 * t_pre + 1.0

    mask_post = phase < phase_inc
    t_post = phase[mask_post] / phase_inc[mask_post]
    saw[mask_post] -= 2.0 * t_post - t_post * t_post - 1.0

    return saw


def polyblep_square(
    phase: np.ndarray,
    phase_inc: np.ndarray,
    cumphase: np.ndarray,
    pulse_width: float | np.ndarray,
) -> np.ndarray:
    """Generate a bandlimited square/pulse wave as the difference of two saws.

    ``pulse_width`` may be a scalar or a per-sample array.  Per-sample
    arrays enable audio-rate PWM; scalars preserve the legacy path.
    """
    saw1 = polyblep_saw(phase, phase_inc)
    phase2 = (cumphase + pulse_width) % 1.0
    saw2 = polyblep_saw(phase2, phase_inc)
    square = (saw1 - saw2) / 2.0
    square -= square.mean()
    return square


def polyblep_triangle(
    phase: np.ndarray,
    phase_inc: np.ndarray,
    cumphase: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Bandlimited triangle via BLAMP: cumulative sum of the polyblep square."""
    square = polyblep_square(phase, phase_inc, cumphase, pulse_width=0.5)
    triangle = np.cumsum(square) / sample_rate
    triangle -= triangle.mean()
    peak = np.max(np.abs(triangle))
    if peak > 1e-9:
        triangle /= peak
    return triangle
