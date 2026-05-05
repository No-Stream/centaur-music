"""Shared second-order ADAA (AD2) utilities for tanh nonlinearities.

Both :mod:`code_musics.engines._filters` and
:mod:`code_musics.engines._waveshaper` need F2(tanh) — the second
antiderivative of ``tanh`` — for Bilbao/Esqueda-style second-order
antiderivative anti-aliasing.  The closed form requires the dilogarithm
``Li2``, which numba does not ship, so we precompute ``F2`` on a dense
uniform grid at module load and linearly interpolate inside the numba
kernels.  Outside the grid range, ``F1 = log(cosh(x))`` is exactly
``|x| - ln(2)`` to many digits, so ``F2`` continues analytically as a
quadratic on each side with ``C⁰`` continuity at the edge.

Historically the filter and waveshaper modules each carried their own copy
of this infrastructure (same 20001-point grid, same ``[-20, 20]`` range,
same trapezoidal integration, same asymptotic extension).  The two copies
were bit-identical but diverged in naming.  This module is the single
source of truth; both callers import from here.

Exports:
    * :data:`_LN2` — ``ln(2)`` as a float constant shared by numba kernels.
    * :data:`_ADAA_DX_THRESHOLD` — fallback threshold for divided-difference
      denominators (``1e-5``).  Used by both AD1 and AD2 dispatchers.
    * :data:`_AD2_TANH_LIMIT`, :data:`_AD2_TANH_GRID`, :data:`_AD2_TANH_STEP` —
      grid geometry for the F2(tanh) table.
    * :data:`_AD2_TANH_TABLE` — precomputed F2 values on the grid.
    * :func:`_ad2_tanh_lut` — numba-compiled F2(tanh) evaluator that combines
      linear interpolation inside the grid with the exact quadratic
      asymptote outside.  The full AD2 Bilbao/Esqueda kernel
      (three-sample divided-difference form) lives in :mod:`_waveshaper`
      as ``_adaa2_sample`` for the general multi-algorithm case, and in
      :mod:`_filters` as ``_adaa2_tanh`` for the tanh-only filter-feedback
      path.  The filter's ``_adaa2_tanh`` is effectively a tanh-only
      specialization of the waveshaper's ``_adaa2_sample`` — both use this
      shared LUT evaluator for their F2 calls.
"""

from __future__ import annotations

import numba
import numpy as np

# ``ln(2)`` — used by both the F2(tanh) asymptote extension and by callers'
# own ``log(cosh)`` helpers.  Kept here so the table build and the runtime
# asymptote evaluation see exactly the same constant.
_LN2: float = 0.6931471805599453

# Fallback threshold for divided-difference denominators in ADAA dispatch.
# When ``|x_curr - x_prev|`` (AD1) or any of the AD2 denominators falls
# below this, callers fall back to direct midpoint evaluation to avoid
# catastrophic cancellation in the ``(F(x_n) - F(x_{n-1})) / dx`` ratio.
# Centralised here so AD1 and AD2 in both files stay in lock-step.
_ADAA_DX_THRESHOLD: float = 1e-5

# Grid geometry for the F2(tanh) lookup table.  The grid spans
# ``[-_AD2_TANH_LIMIT, +_AD2_TANH_LIMIT]`` with ``_AD2_TANH_GRID`` points,
# chosen odd so that ``x = 0`` lands exactly on a grid point.  ``20001``
# points over ``[-20, 20]`` gives 2 ms step at ``sample_rate = 44.1 kHz``
# equivalent precision (far finer than any realistic driven-input
# excursion).  Outside ``±20`` ``F1 = log(cosh(x))`` is within 1e-17 of
# ``|x| - ln(2)`` so the asymptotic continuation is exact to double
# precision.
_AD2_TANH_LIMIT: float = 20.0
_AD2_TANH_GRID: int = 20001


def _build_ad2_tanh_table() -> np.ndarray:
    """Build a dense table of ``F2(log(cosh(x)))`` for ``x ∈ [-limit, +limit]``.

    ``F2`` is computed via cumulative trapezoidal integration of ``F1``
    starting from ``x=0`` (where ``F2(0) = 0`` by the odd-symmetry choice of
    integration constant).  At load time this costs ~20k float ops; the
    returned array is consumed by the numba kernel via linear
    interpolation.
    """
    limit = _AD2_TANH_LIMIT
    grid = _AD2_TANH_GRID
    xs = np.linspace(-limit, limit, grid, dtype=np.float64)
    # F1 = log(cosh(x)) — numerically stable form.
    ax = np.abs(xs)
    f1 = ax + np.log1p(np.exp(-2.0 * ax)) - _LN2
    # Trapezoidal cumulative integral from the center (x=0) outward.  F1 is
    # even, so F2 is odd and F2(0) = 0 by choice of integration constant.
    dx = xs[1] - xs[0]
    mid = (grid - 1) // 2  # index of x=0
    f2 = np.zeros(grid, dtype=np.float64)
    # Positive side: cumulative trapz from x=0 outward.
    for i in range(mid + 1, grid):
        f2[i] = f2[i - 1] + 0.5 * (f1[i - 1] + f1[i]) * dx
    # Negative side: by odd symmetry F2(-x) = -F2(x).
    for i in range(mid - 1, -1, -1):
        f2[i] = -f2[grid - 1 - i]
    return f2


_AD2_TANH_TABLE: np.ndarray = _build_ad2_tanh_table()
_AD2_TANH_STEP: float = 2.0 * _AD2_TANH_LIMIT / (_AD2_TANH_GRID - 1)


@numba.njit(cache=True)
def _ad2_tanh_lut(x: float, table: np.ndarray) -> float:
    """Second antiderivative of tanh, via lookup + asymptotic extension.

    ``F2(tanh)`` requires the dilogarithm ``Li2``, which numba does not
    ship.  We precompute ``F2`` on a dense uniform grid in
    ``[-_AD2_TANH_LIMIT, +limit]`` and linearly interpolate here.  Outside
    that range, ``F1 = log(cosh(x))`` is exactly ``|x| - ln(2)`` to many
    digits, so ``F2`` on the right (``x > +limit``) continues as
    ``x²/2 - ln(2)·x + C_r`` with ``C_r`` chosen for ``C⁰`` continuity at
    the edge, and the left side uses odd symmetry ``F2(-x) = -F2(x)``.

    The ``table`` argument is taken explicitly (rather than referenced as a
    module global) so numba can specialize on its dtype / shape per
    call-site and so callers can pass the same compiled kernel the same
    module-level :data:`_AD2_TANH_TABLE` without numba having to reach
    through Python globals.
    """
    limit = _AD2_TANH_LIMIT
    if x <= -limit:
        # Use odd symmetry from the positive-side asymptote.
        edge = float(table[table.shape[0] - 1])  # F2 at +limit
        # F2_asym(x) for x >> 0: x²/2 - ln(2)*x + const.  At x=limit this
        # should equal ``edge``, so const = edge - limit²/2 + ln(2)*limit.
        const = edge - 0.5 * limit * limit + _LN2 * limit
        neg_x = -x  # positive-side evaluation
        f2_pos = 0.5 * neg_x * neg_x - _LN2 * neg_x + const
        return -f2_pos
    if x >= limit:
        edge = float(table[table.shape[0] - 1])
        const = edge - 0.5 * limit * limit + _LN2 * limit
        return 0.5 * x * x - _LN2 * x + const
    # Linear interpolation inside the grid.
    pos = (x + limit) / _AD2_TANH_STEP
    idx = int(pos)
    frac = pos - idx
    n = table.shape[0]
    if idx >= n - 1:
        return float(table[n - 1])
    a = float(table[idx])
    b = float(table[idx + 1])
    return a + (b - a) * frac
