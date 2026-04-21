"""Parallel modal resonator bank for physically-informed drum tones.

A mono exciter signal is run through N independent time-invariant 2-pole
resonant bandpass biquads, one per mode.  Mode centers come from
``freq_hz * mode_ratios[i]``, amplitudes from ``mode_amps[i]``, and Q from
``mode_decays_s[i]`` via ``Q = pi * f_c * decay_s`` (equivalent to ~-60 dB
energy decay over ``decay_s``).

This bridges ``code_musics.spectra`` mode tables (membrane, bar, bowl,
plate, stopped pipe) into drum synthesis by letting an exciter ring
through a physically-informed resonance structure.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numba
import numpy as np


@numba.njit(cache=True)
def _render_modal_bank_loop(
    out: np.ndarray,
    exciter: np.ndarray,
    b0_coeffs: np.ndarray,
    b2_coeffs: np.ndarray,
    a1_coeffs: np.ndarray,
    a2_coeffs: np.ndarray,
    mode_amps: np.ndarray,
) -> None:
    """Sum N parallel time-invariant bandpass biquads into ``out``."""
    n_samples = exciter.shape[0]
    n_modes = b0_coeffs.shape[0]

    for mode_idx in range(n_modes):
        amp = mode_amps[mode_idx]
        if amp == 0.0:
            continue

        b0 = b0_coeffs[mode_idx]
        b2 = b2_coeffs[mode_idx]
        a1 = a1_coeffs[mode_idx]
        a2 = a2_coeffs[mode_idx]

        x1 = 0.0
        x2 = 0.0
        y1 = 0.0
        y2 = 0.0

        for i in range(n_samples):
            x0 = exciter[i]
            y0 = b0 * x0 + b2 * x2 - a1 * y1 - a2 * y2
            out[i] += amp * y0

            x2 = x1
            x1 = x0
            y2 = y1
            y1 = y0


def render_modal_bank(
    exciter: np.ndarray | Sequence[float],
    mode_ratios: np.ndarray | Sequence[float],
    mode_amps: np.ndarray | Sequence[float],
    mode_decays_s: np.ndarray | Sequence[float],
    freq_hz: float,
    sample_rate: int,
) -> np.ndarray:
    """Render a parallel modal resonator bank driven by a mono exciter.

    Args:
        exciter: Mono excitation signal.  Arbitrary-shape float sequence
            accepted; coerced to ``float64``.
        mode_ratios: Per-mode frequency ratios.  Center frequency of mode
            ``i`` is ``freq_hz * mode_ratios[i]``.
        mode_amps: Per-mode output amplitudes.  Not internally normalized.
        mode_decays_s: Per-mode -60 dB decay time in seconds.  Q is
            derived as ``Q = pi * f_c * decay_s``.
        freq_hz: Fundamental frequency in Hz (positive).
        sample_rate: Audio sample rate (positive integer).

    Returns:
        Mono ``float64`` array the same length as ``exciter``, the sum of
        all mode outputs.

    Modes whose center frequency lies at or above Nyquist, or whose decay
    is non-positive, are silently skipped.  Length mismatches between
    ``mode_ratios``, ``mode_amps``, and ``mode_decays_s`` raise ValueError.
    """
    exciter_arr = np.asarray(exciter, dtype=np.float64)
    ratios_arr = np.asarray(mode_ratios, dtype=np.float64)
    amps_arr = np.asarray(mode_amps, dtype=np.float64)
    decays_arr = np.asarray(mode_decays_s, dtype=np.float64)

    if exciter_arr.ndim != 1:
        raise ValueError("exciter must be a 1-D array")
    if ratios_arr.ndim != 1 or amps_arr.ndim != 1 or decays_arr.ndim != 1:
        raise ValueError("mode_ratios, mode_amps, mode_decays_s must be 1-D arrays")
    if not (ratios_arr.shape[0] == amps_arr.shape[0] == decays_arr.shape[0]):
        raise ValueError(
            "mode_ratios, mode_amps, mode_decays_s must all have the same length; "
            f"got {ratios_arr.shape[0]}, {amps_arr.shape[0]}, {decays_arr.shape[0]}"
        )
    if freq_hz <= 0.0:
        raise ValueError(f"freq_hz must be positive, got {freq_hz}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    if exciter_arr.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)

    nyquist = 0.5 * sample_rate
    two_pi_over_sr = 2.0 * math.pi / sample_rate

    active_b0: list[float] = []
    active_b2: list[float] = []
    active_a1: list[float] = []
    active_a2: list[float] = []
    active_amps: list[float] = []

    for i in range(ratios_arr.shape[0]):
        f_c = freq_hz * ratios_arr[i]
        decay_s = decays_arr[i]

        if f_c >= nyquist:
            continue
        if f_c <= 0.0:
            continue
        if decay_s <= 0.0:
            continue

        q = math.pi * f_c * decay_s
        if q <= 0.0:
            continue

        w0 = two_pi_over_sr * f_c
        sin_w0 = math.sin(w0)
        cos_w0 = math.cos(w0)
        alpha = sin_w0 / (2.0 * q)

        a0 = 1.0 + alpha
        b0 = alpha / a0
        b2 = -alpha / a0
        a1 = (-2.0 * cos_w0) / a0
        a2 = (1.0 - alpha) / a0

        active_b0.append(b0)
        active_b2.append(b2)
        active_a1.append(a1)
        active_a2.append(a2)
        active_amps.append(float(amps_arr[i]))

    out = np.zeros(exciter_arr.shape[0], dtype=np.float64)

    if not active_b0:
        return out

    _render_modal_bank_loop(
        out,
        exciter_arr,
        np.asarray(active_b0, dtype=np.float64),
        np.asarray(active_b2, dtype=np.float64),
        np.asarray(active_a1, dtype=np.float64),
        np.asarray(active_a2, dtype=np.float64),
        np.asarray(active_amps, dtype=np.float64),
    )
    return out
