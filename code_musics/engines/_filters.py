"""Shared DSP filter primitives used by multiple synth engines."""

from __future__ import annotations

import numpy as np

_SUPPORTED_FILTER_MODES = {"lowpass", "bandpass", "highpass", "notch"}


def _soft_clip(value: float) -> float:
    """Return a stable soft-clipped sample."""
    return float(np.tanh(value))


def _select_filter_output(
    *, filter_mode: str, low: float, band: float, high: float
) -> float:
    """Return the requested state-variable filter output."""
    if filter_mode == "lowpass":
        return low
    if filter_mode == "bandpass":
        return band
    if filter_mode == "highpass":
        return high
    return low + high


def _apply_linear_zdf_svf(
    signal: np.ndarray,
    *,
    cutoff_profile: np.ndarray,
    damping: float,
    sample_rate: int,
    filter_mode: str,
) -> np.ndarray:
    """Apply the fully linear ZDF/TPT state-variable filter."""
    filtered = np.empty_like(signal, dtype=np.float64)
    low_state = 0.0
    band_state = 0.0

    for index, sample in enumerate(signal):
        g = np.tan(np.pi * float(cutoff_profile[index]) / sample_rate)
        feedback = low_state + damping * band_state
        high = (float(sample) - feedback) / (1.0 + damping * g + g * g)
        band = g * high + band_state
        low = g * band + low_state
        band_state = g * high + band
        low_state = g * band + low
        filtered[index] = _select_filter_output(
            filter_mode=filter_mode,
            low=low,
            band=band,
            high=high,
        )

    return filtered


def _apply_driven_zdf_svf(
    signal: np.ndarray,
    *,
    cutoff_profile: np.ndarray,
    damping: float,
    sample_rate: int,
    filter_mode: str,
    filter_drive: float,
) -> np.ndarray:
    """Apply a moderated nonlinear ZDF/TPT state-variable filter."""
    filtered = np.empty_like(signal, dtype=np.float64)
    low_state = 0.0
    band_state = 0.0

    drive_gain = 1.0 + (2.0 * filter_drive)
    feedback_gain = 1.0 + filter_drive
    output_gain = 1.0 + (0.25 * filter_drive)
    compensation = 1.0 / (1.0 + (0.2 * filter_drive))

    for index, sample in enumerate(signal):
        g = np.tan(np.pi * float(cutoff_profile[index]) / sample_rate)
        driven_input = _soft_clip(float(sample) * drive_gain)
        feedback = _soft_clip((low_state + (damping * band_state)) * feedback_gain)
        high = (driven_input - feedback) / (1.0 + (damping * g) + (g * g))
        band = (g * high) + band_state
        low = (g * band) + low_state
        band_state = (g * high) + band
        low_state = (g * band) + low
        output = _select_filter_output(
            filter_mode=filter_mode,
            low=low,
            band=band,
            high=high,
        )
        filtered[index] = _soft_clip(output * output_gain)

    return filtered * compensation


def apply_zdf_svf(
    signal: np.ndarray,
    *,
    cutoff_profile: np.ndarray,
    resonance: float,
    sample_rate: int,
    filter_mode: str,
    filter_drive: float,
) -> np.ndarray:
    """Apply a per-sample ZDF/TPT state-variable filter.

    When ``filter_drive=0`` the filter runs a fully linear path — no soft-clipping
    anywhere in the loop.  Nonlinear processing only activates for ``filter_drive>0``.

    Args:
        signal: Input audio array.
        cutoff_profile: Per-sample cutoff frequency in Hz (same length as signal).
        resonance: Non-negative resonance amount.
        sample_rate: Audio sample rate in Hz.
        filter_mode: One of ``"lowpass"``, ``"bandpass"``, ``"highpass"``, ``"notch"``.
        filter_drive: Non-negative drive amount; 0.0 means fully linear/clean.
    """
    resonance_bounded = max(0.0, float(resonance))
    q = 0.707 + (11.293 * resonance_bounded)
    damping = 1.0 / q
    linear_filtered = _apply_linear_zdf_svf(
        signal,
        cutoff_profile=cutoff_profile,
        damping=damping,
        sample_rate=sample_rate,
        filter_mode=filter_mode,
    )

    if filter_drive <= 0.0:
        return linear_filtered

    driven_filtered = _apply_driven_zdf_svf(
        signal,
        cutoff_profile=cutoff_profile,
        damping=damping,
        sample_rate=sample_rate,
        filter_mode=filter_mode,
        filter_drive=filter_drive,
    )
    drive_blend = 0.75 * (filter_drive**1.3)
    return ((1.0 - drive_blend) * linear_filtered) + (drive_blend * driven_filtered)
