"""Shared DSP filter primitives used by multiple synth engines."""

from __future__ import annotations

import math

import numba
import numpy as np

_SUPPORTED_FILTER_MODES = {"lowpass", "bandpass", "highpass", "notch"}

# Integer constants for filter mode selection inside numba-compiled loops.
_LP: int = 0
_BP: int = 1
_HP: int = 2
_NOTCH: int = 3

_MODE_STR_TO_INT: dict[str, int] = {
    "lowpass": _LP,
    "bandpass": _BP,
    "highpass": _HP,
    "notch": _NOTCH,
}


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


@numba.njit(cache=True)
def _apply_linear_zdf_svf(
    signal: np.ndarray,
    cutoff_profile: np.ndarray,
    damping: float,
    sample_rate: int,
    mode_int: int,
    precomputed_g: float,
) -> np.ndarray:
    """Apply the fully linear ZDF/TPT state-variable filter (numba-compiled)."""
    n = signal.shape[0]
    filtered = np.empty(n, dtype=np.float64)
    low_state = 0.0
    band_state = 0.0
    use_precomputed = precomputed_g >= 0.0

    for i in range(n):
        if use_precomputed:
            g = precomputed_g
        else:
            g = math.tan(math.pi * cutoff_profile[i] / sample_rate)
        sample = signal[i]
        feedback = low_state + damping * band_state
        high = (sample - feedback) / (1.0 + damping * g + g * g)
        band = g * high + band_state
        low = g * band + low_state
        band_state = g * high + band
        low_state = g * band + low

        if mode_int == _LP:
            filtered[i] = low
        elif mode_int == _BP:
            filtered[i] = band
        elif mode_int == _HP:
            filtered[i] = high
        else:
            filtered[i] = low + high

    return filtered


@numba.njit(cache=True)
def _apply_driven_zdf_svf(
    signal: np.ndarray,
    cutoff_profile: np.ndarray,
    damping: float,
    sample_rate: int,
    mode_int: int,
    filter_drive: float,
    precomputed_g: float,
) -> np.ndarray:
    """Apply a moderated nonlinear ZDF/TPT state-variable filter (numba-compiled)."""
    n = signal.shape[0]
    filtered = np.empty(n, dtype=np.float64)
    low_state = 0.0
    band_state = 0.0
    use_precomputed = precomputed_g >= 0.0

    drive_gain = 1.0 + (2.0 * filter_drive)
    feedback_gain = 1.0 + filter_drive
    output_gain = 1.0 + (0.25 * filter_drive)
    compensation = 1.0 / (1.0 + (0.2 * filter_drive))

    for i in range(n):
        if use_precomputed:
            g = precomputed_g
        else:
            g = math.tan(math.pi * cutoff_profile[i] / sample_rate)
        sample = signal[i]
        driven_input = math.tanh(sample * drive_gain)
        feedback = math.tanh((low_state + (damping * band_state)) * feedback_gain)
        high = (driven_input - feedback) / (1.0 + (damping * g) + (g * g))
        band = (g * high) + band_state
        low = (g * band) + low_state
        band_state = (g * high) + band
        low_state = (g * band) + low

        if mode_int == _LP:
            output = low
        elif mode_int == _BP:
            output = band
        elif mode_int == _HP:
            output = high
        else:
            output = low + high

        filtered[i] = math.tanh(output * output_gain)

    for i in range(n):
        filtered[i] *= compensation

    return filtered


def apply_zdf_svf(
    signal: np.ndarray,
    *,
    cutoff_profile: np.ndarray,
    resonance_q: float = 0.707,
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
        resonance_q: Filter Q value (>= 0.5). Q=0.707 is Butterworth (no resonance
            peak). Q=1 is a gentle peak; Q=4+ approaches self-oscillation depending
            on drive.
        sample_rate: Audio sample rate in Hz.
        filter_mode: One of ``"lowpass"``, ``"bandpass"``, ``"highpass"``, ``"notch"``.
        filter_drive: Non-negative drive amount; 0.0 means fully linear/clean.
    """
    q = max(0.5, float(resonance_q))
    damping = 1.0 / q
    mode_int = _MODE_STR_TO_INT.get(filter_mode, _LP)

    sig = np.asarray(signal, dtype=np.float64)
    cutoff = np.asarray(cutoff_profile, dtype=np.float64)

    # Pre-compute g when cutoff is constant to avoid per-sample tan() calls.
    if cutoff.size > 0 and np.all(cutoff == cutoff[0]):
        precomputed_g = math.tan(math.pi * float(cutoff[0]) / sample_rate)
    else:
        precomputed_g = -1.0  # sentinel: compute per-sample

    linear_filtered = _apply_linear_zdf_svf(
        sig,
        cutoff,
        damping,
        sample_rate,
        mode_int,
        precomputed_g,
    )

    if filter_drive <= 0.0:
        return linear_filtered

    driven_filtered = _apply_driven_zdf_svf(
        sig,
        cutoff,
        damping,
        sample_rate,
        mode_int,
        filter_drive,
        precomputed_g,
    )
    drive_blend = 0.75 * (filter_drive**1.3)
    return ((1.0 - drive_blend) * linear_filtered) + (drive_blend * driven_filtered)
