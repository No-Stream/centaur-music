"""Direct unit tests for the filter DSP primitives in _filters.py."""

from __future__ import annotations

import math

import pytest

from code_musics.engines._filters import _adaa_tanh, _algebraic_sat, _log_cosh

# Access the Python fallback for numba-compiled functions.
log_cosh = _log_cosh.py_func
adaa_tanh = _adaa_tanh.py_func
algebraic_sat = _algebraic_sat.py_func


# ---------------------------------------------------------------------------
# _log_cosh
# ---------------------------------------------------------------------------


class TestLogCosh:
    def test_zero(self) -> None:
        assert log_cosh(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_small_quadratic_approx(self) -> None:
        x = 0.1
        expected = x * x / 2.0  # quadratic approximation for small x
        assert log_cosh(x) == pytest.approx(expected, rel=0.01)

    def test_large_positive(self) -> None:
        expected = 100.0 - math.log(2.0)
        assert log_cosh(100.0) == pytest.approx(expected, rel=1e-6)

    def test_large_negative_symmetric(self) -> None:
        expected = 100.0 - math.log(2.0)
        assert log_cosh(-100.0) == pytest.approx(expected, rel=1e-6)

    def test_monotonic(self) -> None:
        xs = [0.0, 1.0, 2.0, 5.0, 10.0]
        values = [log_cosh(x) for x in xs]
        for i in range(len(values) - 1):
            assert values[i] < values[i + 1]


# ---------------------------------------------------------------------------
# _adaa_tanh
# ---------------------------------------------------------------------------


class TestAdaaTanh:
    def test_small_dx_fallback(self) -> None:
        result = adaa_tanh(0.5, 0.5)
        assert result == pytest.approx(math.tanh(0.5), rel=1e-6)

    def test_bounded(self) -> None:
        test_pairs = [
            (0.0, 0.0),
            (1.0, -1.0),
            (5.0, 3.0),
            (-3.0, -5.0),
            (10.0, 9.0),
            (-10.0, -9.0),
        ]
        for x_curr, x_prev in test_pairs:
            result = adaa_tanh(x_curr, x_prev)
            assert -1.0 <= result <= 1.0, (
                f"out of bounds for ({x_curr}, {x_prev}): {result}"
            )

    def test_approximates_tanh_for_small_dx(self) -> None:
        x_prev = 1.0
        x_curr = 1.001
        midpoint = 0.5 * (x_curr + x_prev)
        result = adaa_tanh(x_curr, x_prev)
        assert result == pytest.approx(math.tanh(midpoint), rel=0.01)

    def test_large_inputs(self) -> None:
        result = adaa_tanh(10.0, 9.0)
        assert math.isfinite(result)
        assert -1.0 <= result <= 1.0

    def test_sign_change(self) -> None:
        result = adaa_tanh(1.0, -1.0)
        assert math.isfinite(result)
        assert -1.0 <= result <= 1.0

    def test_identical_inputs(self) -> None:
        result = adaa_tanh(2.0, 2.0)
        assert result == pytest.approx(math.tanh(2.0), rel=1e-6)


# ---------------------------------------------------------------------------
# _algebraic_sat
# ---------------------------------------------------------------------------


class TestAlgebraicSat:
    def test_transparent_small(self) -> None:
        for x in [0.1, -0.1, 0.2, -0.2, 0.3, -0.3]:
            result = algebraic_sat(x)
            assert result == pytest.approx(x, rel=0.05), (
                f"not transparent at {x}: {result}"
            )

    def test_saturates_large(self) -> None:
        for x in [3.0, 5.0, 10.0]:
            result = algebraic_sat(x)
            assert result <= 1.0, f"positive saturation failed at {x}: {result}"
        for x in [-3.0, -5.0, -10.0]:
            result = algebraic_sat(x)
            assert result >= -1.0, f"negative saturation failed at {x}: {result}"

    def test_boundary_continuity(self) -> None:
        x_below = 1.999
        x_at = 2.0
        x_above = 2.001
        val_below = algebraic_sat(x_below)
        val_at = algebraic_sat(x_at)
        val_above = algebraic_sat(x_above)
        assert abs(val_at - val_below) < 0.01
        assert abs(val_above - val_at) < 0.01

    def test_odd_symmetry(self) -> None:
        for x in [0.5, 1.0, 2.0, 3.0, 5.0]:
            assert algebraic_sat(-x) == pytest.approx(-algebraic_sat(x), abs=1e-12)

    def test_monotonic(self) -> None:
        xs = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
        values = [algebraic_sat(x) for x in xs]
        for i in range(len(values) - 1):
            assert values[i] < values[i + 1]
