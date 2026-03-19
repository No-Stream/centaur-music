"""Shared DSP filter primitives used by multiple synth engines."""

from __future__ import annotations

import numpy as np

_SUPPORTED_FILTER_MODES = {"lowpass", "bandpass", "highpass", "notch"}


def _soft_clip(value: float) -> float:
    """Return a stable soft-clipped sample."""
    return float(np.tanh(value))


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
    filtered = np.empty_like(signal, dtype=np.float64)
    low_state = 0.0
    band_state = 0.0

    resonance_bounded = max(0.0, float(resonance))
    q = 0.707 + (11.293 * resonance_bounded)
    damping = 1.0 / q

    if filter_drive <= 0.0:
        # Fully linear path — no soft-clipping anywhere.
        for index, sample in enumerate(signal):
            g = np.tan(np.pi * float(cutoff_profile[index]) / sample_rate)
            feedback = low_state + damping * band_state
            high = (float(sample) - feedback) / (1.0 + damping * g + g * g)
            band = g * high + band_state
            low = g * band + low_state
            band_state = g * high + band
            low_state = g * band + low
            if filter_mode == "lowpass":
                filtered[index] = low
            elif filter_mode == "bandpass":
                filtered[index] = band
            elif filter_mode == "highpass":
                filtered[index] = high
            else:
                filtered[index] = low + high
        return filtered

    # Nonlinear path with soft-clipping and drive — only reached when filter_drive > 0.
    # Normalize input to ±1 before driving so that filter_drive has consistent meaning
    # regardless of the incoming signal amplitude.
    input_peak = float(np.max(np.abs(signal)))
    working_signal = signal / input_peak if input_peak > 1e-9 else signal

    drive_gain = 1.0 + (2.0 * filter_drive)
    for index, sample in enumerate(working_signal):
        g = np.tan(np.pi * float(cutoff_profile[index]) / sample_rate)
        driven_input = _soft_clip(float(sample) * drive_gain)
        feedback = _soft_clip(
            (low_state + (damping * band_state)) * (1.0 + filter_drive)
        )
        high = (driven_input - feedback) / (1.0 + (damping * g) + (g * g))
        band = (g * high) + band_state
        low = (g * band) + low_state
        band_state = (g * high) + band
        low_state = (g * band) + low
        if filter_mode == "lowpass":
            output = low
        elif filter_mode == "bandpass":
            output = band
        elif filter_mode == "highpass":
            output = high
        else:
            output = low + high
        filtered[index] = _soft_clip(output * (1.0 + (0.35 * filter_drive)))

    compensation = 1.0 / (1.0 + (0.6 * filter_drive))
    return filtered * compensation
