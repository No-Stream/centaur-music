"""Modal resonator bank with optional mode coupling and allpass dispersion.

Runs a mono exciter through N parallel bandpass biquads (one per mode).
``coupling`` adds inter-mode energy exchange for rolling beats / living decays;
``dispersion`` adds post-bank allpass frequency-smearing for piano-stiffness
and bell-warp character.  Defaults (``coupling=0``, ``dispersion=0``) are
bit-identical to the plain-parallel legacy path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numba
import numpy as np

_VALID_COUPLING_TOPOLOGIES = frozenset({"chain", "ring", "all"})
# Public ceiling for the ``coupling`` parameter. 0.35 is the empirical knee
# where the cheapest chain topology still stays stable for the highest-Q
# presets we ship (long bowl decays with small n_modes). Values above this
# are rejected as a ValueError rather than silently clamped.
COUPLING_MAX = 0.35
# Allpass dispersion peaks at 0.85 for strong bell-warp without going
# unstable. Callers pass [0, 1]; we scale to [0, 0.85] internally.
_DISPERSION_ALPHA_MAX = 0.85


@numba.njit(cache=True)
def _render_modal_bank_parallel_loop(
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


@numba.njit(cache=True)
def _render_modal_bank_coupled_loop(
    out: np.ndarray,
    exciter: np.ndarray,
    b0_coeffs: np.ndarray,
    b2_coeffs: np.ndarray,
    a1_coeffs: np.ndarray,
    a2_coeffs: np.ndarray,
    mode_amps: np.ndarray,
    coupling_weights: np.ndarray,
) -> None:
    """Sum N coupled bandpass biquads into ``out``.

    Mode i's input each sample is ``exciter[n] + sum_j coupling_weights[i,j] *
    y_prev[j]`` — a one-sample delayed output feedback, the standard coupled-
    modal-array approximation.
    """
    n_samples = exciter.shape[0]
    n_modes = b0_coeffs.shape[0]

    x1 = np.zeros(n_modes, dtype=np.float64)
    x2 = np.zeros(n_modes, dtype=np.float64)
    y1 = np.zeros(n_modes, dtype=np.float64)
    y2 = np.zeros(n_modes, dtype=np.float64)
    y_prev = np.zeros(n_modes, dtype=np.float64)

    for n in range(n_samples):
        ex = exciter[n]
        for i in range(n_modes):
            coupling_in = 0.0
            for j in range(n_modes):
                coupling_in += coupling_weights[i, j] * y_prev[j]
            x0 = ex + coupling_in

            y0 = (
                b0_coeffs[i] * x0
                + b2_coeffs[i] * x2[i]
                - a1_coeffs[i] * y1[i]
                - a2_coeffs[i] * y2[i]
            )
            out[n] += mode_amps[i] * y0

            x2[i] = x1[i]
            x1[i] = x0
            y2[i] = y1[i]
            y1[i] = y0

        for i in range(n_modes):
            y_prev[i] = y1[i]


@numba.njit(cache=True)
def _apply_allpass_cascade(
    signal: np.ndarray,
    alpha: float,
    n_stages: int,
) -> np.ndarray:
    """Apply ``n_stages`` of first-order allpass filters for dispersion.

    Each stage has transfer function ``H(z) = (-alpha + z^-1) / (1 - alpha z^-1)``
    — unity magnitude, frequency-dependent phase delay.  Cascading stages
    accumulates phase distortion and produces audible piano-stiffness /
    bell-warp character on transient-driven content.
    """
    n_samples = signal.shape[0]
    out = np.empty_like(signal)
    buf = signal.copy()
    for _ in range(n_stages):
        x_prev = 0.0
        y_prev = 0.0
        for n in range(n_samples):
            x = buf[n]
            y = -alpha * x + x_prev + alpha * y_prev
            out[n] = y
            x_prev = x
            y_prev = y
        buf = out.copy()
    return buf


def _build_coupling_weights(
    *, n_modes: int, amount: float, topology: str
) -> np.ndarray:
    """Build the N×N coupling matrix for the requested topology.

    ``amount`` is already validated to ``[0, COUPLING_MAX]`` by the public
    ``render_modal_bank`` entry point.  Entries are real-valued, symmetric,
    with zero diagonal.
    """
    if n_modes <= 1 or amount <= 0.0:
        return np.zeros((max(n_modes, 1), max(n_modes, 1)), dtype=np.float64)

    amount_clamped = float(amount)
    weights = np.zeros((n_modes, n_modes), dtype=np.float64)

    if topology == "chain":
        half = 0.5 * amount_clamped
        for i in range(n_modes):
            if i > 0:
                weights[i, i - 1] = half
            if i < n_modes - 1:
                weights[i, i + 1] = half
        return weights

    if topology == "ring":
        half = 0.5 * amount_clamped
        for i in range(n_modes):
            weights[i, (i - 1) % n_modes] = half
            weights[i, (i + 1) % n_modes] = half
        return weights

    if topology == "all":
        for i in range(n_modes):
            row_norm = 0.0
            for j in range(n_modes):
                if i == j:
                    continue
                weights[i, j] = 1.0 / float(abs(i - j))
                row_norm += weights[i, j]
            if row_norm > 0.0:
                for j in range(n_modes):
                    if i != j:
                        weights[i, j] = weights[i, j] * amount_clamped / row_norm
        return weights

    raise ValueError(
        f"coupling_topology must be one of {sorted(_VALID_COUPLING_TOPOLOGIES)}, "
        f"got {topology!r}"
    )


def render_modal_bank(
    exciter: np.ndarray | Sequence[float],
    mode_ratios: np.ndarray | Sequence[float],
    mode_amps: np.ndarray | Sequence[float],
    mode_decays_s: np.ndarray | Sequence[float],
    freq_hz: float,
    sample_rate: int,
    *,
    coupling: float = 0.0,
    coupling_topology: str = "chain",
    dispersion: float = 0.0,
    dispersion_n_stages: int = 4,
) -> np.ndarray:
    """Render a modal resonator bank driven by a mono exciter.

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
        coupling: Inter-mode coupling amount in ``[0, COUPLING_MAX]`` (0.35).
            ``0`` (default) runs the legacy parallel path bit-identical to
            prior behavior.  Values >0 inject a weighted sum of other modes'
            previous-sample outputs into each mode's input.  The 0.35
            ceiling is where the cheapest chain topology still stays stable
            for the highest-Q presets we ship; values above raise
            ``ValueError``.
        coupling_topology: ``"chain"`` (default), ``"ring"``, or ``"all"``.
            Ignored when ``coupling == 0``.  ``"chain"`` is cheapest and
            most physical; ``"all"`` is richest with ``1/|i-j|`` falloff.
        dispersion: Post-bank allpass dispersion amount in ``[0, 1]``.
            ``0`` (default) is bypass.  Produces piano-stiffness /
            bell-warp character — frequency-dependent phase smearing that
            makes the tail warble instead of ring straight.
        dispersion_n_stages: Number of cascaded first-order allpass stages.
            Default 4.  Ignored when ``dispersion == 0``.

    Returns:
        Mono ``float64`` array the same length as ``exciter``.

    Modes whose center frequency lies at or above Nyquist, or whose decay
    is non-positive, are silently skipped.  Length mismatches between
    ``mode_ratios``, ``mode_amps``, and ``mode_decays_s`` raise ValueError,
    as does an unrecognized ``coupling_topology``.
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
    if coupling < 0.0 or coupling > COUPLING_MAX:
        raise ValueError(f"coupling must be in [0, {COUPLING_MAX}], got {coupling}")
    if dispersion < 0.0 or dispersion > 1.0:
        raise ValueError(f"dispersion must be in [0, 1], got {dispersion}")
    if dispersion_n_stages < 1:
        raise ValueError(f"dispersion_n_stages must be >= 1, got {dispersion_n_stages}")

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

    b0_arr = np.asarray(active_b0, dtype=np.float64)
    b2_arr = np.asarray(active_b2, dtype=np.float64)
    a1_arr = np.asarray(active_a1, dtype=np.float64)
    a2_arr = np.asarray(active_a2, dtype=np.float64)
    amps_active = np.asarray(active_amps, dtype=np.float64)
    n_active = b0_arr.shape[0]

    if coupling <= 0.0 or n_active <= 1:
        _render_modal_bank_parallel_loop(
            out, exciter_arr, b0_arr, b2_arr, a1_arr, a2_arr, amps_active
        )
    else:
        weights = _build_coupling_weights(
            n_modes=n_active, amount=coupling, topology=coupling_topology
        )
        _render_modal_bank_coupled_loop(
            out, exciter_arr, b0_arr, b2_arr, a1_arr, a2_arr, amps_active, weights
        )

    if dispersion > 0.0:
        alpha_disp = _DISPERSION_ALPHA_MAX * float(dispersion)
        out = _apply_allpass_cascade(out, alpha_disp, int(dispersion_n_stages))

    return out
