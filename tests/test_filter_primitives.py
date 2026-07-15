"""Direct unit tests for the filter DSP primitives in _filters.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from code_musics.engines._dsp_utils import apply_filter_oversampled
from code_musics.engines._filters import (
    _adaa_tanh,
    _algebraic_sat,
    _log_cosh,
    apply_filter,
    apply_zdf_svf,
)

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


class TestApplyZdfSvf:
    def test_scalar_cutoff_matches_constant_profile(self) -> None:
        rng = np.random.default_rng(123)
        signal = rng.standard_normal(2048)
        constant_profile = np.full(signal.size, 3_200.0, dtype=np.float64)

        scalar = apply_zdf_svf(
            signal,
            cutoff_profile=3_200.0,
            resonance_q=0.707,
            sample_rate=44_100,
            filter_mode="highpass",
            filter_drive=0.0,
        )
        profiled = apply_zdf_svf(
            signal,
            cutoff_profile=constant_profile,
            resonance_q=0.707,
            sample_rate=44_100,
            filter_mode="highpass",
            filter_drive=0.0,
        )

        np.testing.assert_allclose(scalar, profiled, rtol=0.0, atol=0.0)


class TestFilterControlProfiles:
    @pytest.mark.parametrize(
        "filter_topology",
        [
            "svf",
            "ladder",
            "sallen_key",
            "cascade",
            "sem",
            "jupiter",
            "k35",
            "diode",
        ],
    )
    def test_topologies_accept_dynamic_control_profiles(
        self, filter_topology: str
    ) -> None:
        rng = np.random.default_rng(456)
        sample_rate = 44_100
        time = np.arange(4096, dtype=np.float64) / sample_rate
        signal = (
            0.12 * np.sin(2.0 * np.pi * 110.0 * time)
            + 0.08 * np.sin(2.0 * np.pi * 660.0 * time)
            + 0.03 * rng.standard_normal(time.size)
        )
        cutoff = np.linspace(500.0, 5_500.0, signal.size, dtype=np.float64)
        resonance = np.linspace(0.55, 5.0, signal.size, dtype=np.float64)
        drive = np.linspace(0.0, 0.9, signal.size, dtype=np.float64)
        morph = np.linspace(0.0, 1.5, signal.size, dtype=np.float64)
        feedback = np.linspace(0.0, 0.18, signal.size, dtype=np.float64)

        dynamic = apply_filter(
            signal,
            cutoff_profile=cutoff,
            resonance_q=resonance,
            sample_rate=sample_rate,
            filter_topology=filter_topology,
            filter_drive=drive,
            filter_morph=morph,
            feedback_amount=feedback,
        )
        scalar = apply_filter(
            signal,
            cutoff_profile=float(np.mean(cutoff)),
            resonance_q=float(np.mean(resonance)),
            sample_rate=sample_rate,
            filter_topology=filter_topology,
            filter_drive=float(np.mean(drive)),
            filter_morph=0.0,
            feedback_amount=0.0,
        )

        assert dynamic.shape == signal.shape
        assert np.all(np.isfinite(dynamic))
        mean_abs_delta = float(np.mean(np.abs(dynamic - scalar)))
        assert mean_abs_delta > 1e-5

    def test_hpf_profile_runs_when_any_sample_is_active(self) -> None:
        sample_rate = 44_100
        time = np.arange(4096, dtype=np.float64) / sample_rate
        signal = np.sin(2.0 * np.pi * 110.0 * time)
        cutoff = np.full(signal.size, 5_000.0, dtype=np.float64)
        hpf = np.zeros(signal.size, dtype=np.float64)
        hpf[signal.size // 2 :] = 800.0

        no_hpf = apply_filter(signal, cutoff_profile=cutoff, sample_rate=sample_rate)
        with_hpf = apply_filter(
            signal,
            cutoff_profile=cutoff,
            sample_rate=sample_rate,
            hpf_cutoff_hz=hpf,
        )

        assert with_hpf.shape == signal.shape
        assert np.all(np.isfinite(with_hpf))
        assert np.max(np.abs(with_hpf[signal.size // 2 :])) < np.max(
            np.abs(no_hpf[signal.size // 2 :])
        )

    def test_non_svf_constant_profiles_fold_to_scalar(self) -> None:
        rng = np.random.default_rng(789)
        signal = rng.standard_normal(2048) * 0.1
        cutoff = np.full(signal.size, 2_800.0, dtype=np.float64)

        scalar = apply_filter(
            signal,
            cutoff_profile=cutoff,
            resonance_q=1.25,
            sample_rate=44_100,
            filter_topology="ladder",
            filter_drive=0.2,
            feedback_amount=0.05,
        )
        folded = apply_filter(
            signal,
            cutoff_profile=cutoff,
            resonance_q=np.full(signal.size, 1.25, dtype=np.float64),
            sample_rate=44_100,
            filter_topology="ladder",
            filter_drive=np.full(signal.size, 0.2, dtype=np.float64),
            feedback_amount=np.full(signal.size, 0.05, dtype=np.float64),
        )

        np.testing.assert_allclose(scalar, folded, rtol=0.0, atol=0.0)

    def test_profiled_cascade_drive_preserves_scalar_stage_saturation(self) -> None:
        rng = np.random.default_rng(8642)
        signal = rng.standard_normal(2048).astype(np.float64) * 0.18
        cutoff = np.full(signal.size, 1_700.0, dtype=np.float64)
        resonance_profile = np.full(signal.size, 2.2, dtype=np.float64)
        resonance_profile[-1] = 2.21

        scalar = apply_filter(
            signal,
            cutoff_profile=cutoff,
            resonance_q=2.2,
            sample_rate=44_100,
            filter_topology="cascade",
            filter_drive=0.75,
            filter_solver="newton",
        )
        profiled = apply_filter(
            signal,
            cutoff_profile=cutoff,
            resonance_q=resonance_profile,
            sample_rate=44_100,
            filter_topology="cascade",
            filter_drive=0.75,
            filter_solver="newton",
        )

        stable_prefix = slice(0, signal.size - 1)
        diff_rms = float(
            np.sqrt(np.mean((scalar[stable_prefix] - profiled[stable_prefix]) ** 2))
        )
        ref_rms = float(np.sqrt(np.mean(scalar[stable_prefix] ** 2))) + 1e-12
        assert diff_rms < 1e-3 * ref_rms

    def test_non_svf_accepts_dynamic_main_control_profiles(self) -> None:
        rng = np.random.default_rng(123)
        signal = rng.standard_normal(256).astype(np.float64) * 0.1
        cutoff = np.full(signal.size, 2_000.0, dtype=np.float64)
        resonance = np.linspace(0.7, 2.0, signal.size, dtype=np.float64)

        baseline = apply_filter(
            signal,
            cutoff_profile=cutoff,
            resonance_q=0.7,
            sample_rate=44_100,
            filter_topology="ladder",
            filter_solver="adaa",
        )
        result = apply_filter(
            signal,
            cutoff_profile=cutoff,
            resonance_q=resonance,
            sample_rate=44_100,
            filter_topology="ladder",
            filter_solver="adaa",
        )

        assert result.shape == signal.shape
        assert np.all(np.isfinite(result))
        assert float(np.mean(np.abs(result - baseline))) > 1e-5

    @pytest.mark.parametrize(
        "filter_topology",
        ["ladder", "sallen_key", "cascade", "sem", "jupiter", "k35", "diode"],
    )
    def test_dynamic_non_svf_profiles_use_default_newton_solver(
        self, filter_topology: str
    ) -> None:
        rng = np.random.default_rng(1357)
        sample_rate = 44_100
        time = np.arange(2048, dtype=np.float64) / sample_rate
        signal = (
            0.15 * np.sin(2.0 * np.pi * 90.0 * time)
            + 0.06 * np.sin(2.0 * np.pi * 1_100.0 * time)
            + 0.01 * rng.standard_normal(time.size)
        )
        cutoff = np.linspace(450.0, 4_500.0, signal.size, dtype=np.float64)
        resonance = np.linspace(0.7, 7.0, signal.size, dtype=np.float64)
        drive = np.linspace(0.0, 0.8, signal.size, dtype=np.float64)
        morph = np.linspace(0.0, 1.25, signal.size, dtype=np.float64)
        feedback = np.linspace(0.0, 0.22, signal.size, dtype=np.float64)

        default = apply_filter(
            signal,
            cutoff_profile=cutoff,
            resonance_q=resonance,
            sample_rate=sample_rate,
            filter_topology=filter_topology,
            filter_drive=drive,
            filter_morph=morph,
            feedback_amount=feedback,
        )
        explicit_newton = apply_filter(
            signal,
            cutoff_profile=cutoff,
            resonance_q=resonance,
            sample_rate=sample_rate,
            filter_topology=filter_topology,
            filter_drive=drive,
            filter_morph=morph,
            feedback_amount=feedback,
            filter_solver="newton",
        )

        assert default.shape == signal.shape
        assert np.all(np.isfinite(default))
        np.testing.assert_allclose(default, explicit_newton, rtol=0.0, atol=0.0)

    @pytest.mark.parametrize(
        "filter_topology",
        ["ladder", "sallen_key", "cascade", "sem", "jupiter", "k35", "diode"],
    )
    def test_dynamic_feedback_topologies_keep_newton_distinct_from_adaa(
        self, filter_topology: str
    ) -> None:
        sample_rate = 44_100
        time = np.arange(3072, dtype=np.float64) / sample_rate
        signal = 0.12 * np.sin(2.0 * np.pi * 130.0 * time) + 0.04 * np.sin(
            2.0 * np.pi * 390.0 * time
        )
        cutoff = np.linspace(700.0, 2_400.0, signal.size, dtype=np.float64)
        resonance = np.linspace(3.0, 14.0, signal.size, dtype=np.float64)
        feedback = np.linspace(0.0, 0.35, signal.size, dtype=np.float64)

        newton = apply_filter(
            signal,
            cutoff_profile=cutoff,
            resonance_q=resonance,
            sample_rate=sample_rate,
            filter_topology=filter_topology,
            feedback_amount=feedback,
            filter_solver="newton",
        )
        adaa = apply_filter(
            signal,
            cutoff_profile=cutoff,
            resonance_q=resonance,
            sample_rate=sample_rate,
            filter_topology=filter_topology,
            feedback_amount=feedback,
            filter_solver="adaa",
        )

        assert np.all(np.isfinite(newton))
        assert np.all(np.isfinite(adaa))
        mean_abs_delta = float(np.mean(np.abs(newton - adaa)))
        assert mean_abs_delta > 1e-6

    @pytest.mark.parametrize(
        "filter_topology",
        ["ladder", "sallen_key", "cascade", "sem", "jupiter", "k35", "diode"],
    )
    def test_dynamic_non_svf_newton_profiles_converge_with_more_iterations(
        self, filter_topology: str
    ) -> None:
        sample_rate = 44_100
        time = np.arange(2048, dtype=np.float64) / sample_rate
        signal = 0.1 * np.sin(2.0 * np.pi * 160.0 * time) + 0.04 * np.sin(
            2.0 * np.pi * 640.0 * time
        )
        cutoff = np.linspace(600.0, 3_000.0, signal.size, dtype=np.float64)
        resonance = np.linspace(1.0, 8.0, signal.size, dtype=np.float64)
        drive = np.linspace(0.0, 0.45, signal.size, dtype=np.float64)
        feedback = np.linspace(0.0, 0.25, signal.size, dtype=np.float64)

        fast = apply_filter(
            signal,
            cutoff_profile=cutoff,
            resonance_q=resonance,
            sample_rate=sample_rate,
            filter_topology=filter_topology,
            filter_drive=drive,
            feedback_amount=feedback,
            filter_solver="newton",
            max_newton_iters=2,
        )
        divine = apply_filter(
            signal,
            cutoff_profile=cutoff,
            resonance_q=resonance,
            sample_rate=sample_rate,
            filter_topology=filter_topology,
            filter_drive=drive,
            feedback_amount=feedback,
            filter_solver="newton",
            max_newton_iters=8,
        )

        assert np.all(np.isfinite(fast))
        assert np.all(np.isfinite(divine))
        diff_rms = float(np.sqrt(np.mean((fast - divine) ** 2)))
        ref_rms = float(np.sqrt(np.mean(divine**2))) + 1e-12
        assert diff_rms < 1e-3 * ref_rms

    def test_profile_validation_rejects_bad_shape_and_nonfinite_values(self) -> None:
        signal = np.zeros(128, dtype=np.float64)
        cutoff = np.full(signal.size, 2_000.0, dtype=np.float64)
        bad_resonance = np.full(signal.size, 0.7, dtype=np.float64)
        bad_resonance[10] = np.nan

        with pytest.raises(ValueError, match="resonance_q.*finite"):
            apply_filter(
                signal,
                cutoff_profile=cutoff,
                resonance_q=bad_resonance,
                sample_rate=44_100,
            )

        with pytest.raises(ValueError, match="filter_drive.*length"):
            apply_filter(
                signal,
                cutoff_profile=cutoff,
                sample_rate=44_100,
                filter_drive=np.zeros(signal.size - 1, dtype=np.float64),
            )

    def test_oversampled_filter_threads_dynamic_profiles(self) -> None:
        rng = np.random.default_rng(246)
        signal = rng.standard_normal(512) * 0.1
        cutoff = np.linspace(800.0, 4_000.0, signal.size, dtype=np.float64)
        resonance = np.linspace(0.7, 3.0, signal.size, dtype=np.float64)
        hpf = np.linspace(0.0, 600.0, signal.size, dtype=np.float64)

        result = apply_filter_oversampled(
            signal,
            cutoff_profile=cutoff,
            sample_rate=44_100,
            oversample_factor=2,
            resonance_q=resonance,
            hpf_cutoff_hz=hpf,
        )

        assert result.shape == signal.shape
        assert np.all(np.isfinite(result))
