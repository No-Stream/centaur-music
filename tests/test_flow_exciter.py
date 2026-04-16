"""Tests for the Brush/Flow exciter rare-event S&H noise primitive."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._dsp_utils import flow_exciter


class TestFlowExciterContract:
    """Basic shape, dtype, and finite-value invariants."""

    def test_returns_requested_length(self) -> None:
        rng = np.random.default_rng(0)
        out = flow_exciter(n_samples=1024, param=0.5, rng=rng)
        assert out.shape == (1024,)

    def test_returns_float64(self) -> None:
        rng = np.random.default_rng(0)
        out = flow_exciter(n_samples=512, param=0.5, rng=rng)
        assert out.dtype == np.float64

    def test_finite_values(self) -> None:
        rng = np.random.default_rng(0)
        out = flow_exciter(n_samples=44_100, param=0.5, rng=rng)
        assert np.all(np.isfinite(out)), "output must be free of NaN/Inf"

    def test_bounded(self) -> None:
        """Output must be bounded; rand - 0.5 is in [-0.5, 0.5]."""
        rng = np.random.default_rng(0)
        out = flow_exciter(n_samples=44_100, param=0.7, rng=rng)
        assert np.max(np.abs(out)) <= 1.0, "output should be bounded below 1.0"

    def test_zero_samples(self) -> None:
        rng = np.random.default_rng(0)
        out = flow_exciter(n_samples=0, param=0.5, rng=rng)
        assert out.shape == (0,)


class TestFlowExciterParamBehavior:
    """Behavior across the param range.

    At ``param=0`` the blend weight is zero, so output is the raw held
    S&H state (piecewise constant, rare flips).  At ``param=1`` it is
    pure uniform(-0.5, 0.5) noise.  Intermediate values cross-blend.
    """

    def test_param_zero_is_piecewise_constant(self) -> None:
        """At param=0, output is piecewise-constant with very few unique values.

        Threshold = 0.0001 → ~N*0.0001 flips total, so the output has at
        most a handful of distinct values over a 44.1k render.
        """
        rng = np.random.default_rng(42)
        n = 44_100
        out = flow_exciter(n_samples=n, param=0.0, rng=rng)
        unique_count = len(np.unique(out))
        assert unique_count < 40, (
            f"expected very few unique values at param=0, got {unique_count}"
        )
        # Nearly all sample-to-sample diffs should be exactly zero.
        flat_fraction = np.sum(np.diff(out) == 0.0) / (n - 1)
        assert flat_fraction > 0.99, (
            f"expected > 99% held samples at param=0, got {flat_fraction:.4f}"
        )

    def test_param_one_is_uniform_noise(self) -> None:
        """At param=1, output is pure uniform(-0.5, 0.5).

        Expected RMS is sqrt(1/12) ~ 0.289.
        """
        rng = np.random.default_rng(42)
        out = flow_exciter(n_samples=44_100, param=1.0, rng=rng)
        rms = np.sqrt(np.mean(out**2))
        assert 0.25 < rms < 0.32, f"expected uniform-noise RMS at param=1, got {rms}"
        assert np.min(out) < -0.3 and np.max(out) > 0.3, (
            "expected wide dynamic range at param=1"
        )
        # Unlike param=0, almost every sample should differ from its neighbor.
        flat_fraction = np.sum(np.diff(out) == 0.0) / (out.size - 1)
        assert flat_fraction < 0.001, (
            f"expected near-zero held samples at param=1, got {flat_fraction:.4f}"
        )

    def test_density_increases_with_param(self) -> None:
        """Event density should grow monotonically with param.

        Measured as the number of distinct "segments" (state changes)
        in the held component of the signal.  param=0.1 should have
        many fewer events than param=0.9.
        """
        rng_a = np.random.default_rng(0)
        sparse = flow_exciter(n_samples=44_100, param=0.1, rng=rng_a)
        rng_b = np.random.default_rng(0)
        dense = flow_exciter(n_samples=44_100, param=0.9, rng=rng_b)

        # At low param, mix_weight is tiny so the signal is dominated by
        # "state"; counting exact-equal adjacent samples measures held runs.
        sparse_events = np.sum(np.abs(np.diff(sparse)) > 0.01)
        dense_events = np.sum(np.abs(np.diff(dense)) > 0.01)
        assert sparse_events < dense_events, (
            f"sparse events {sparse_events} should be below dense {dense_events}"
        )

    def test_sparse_param_has_many_held_samples(self) -> None:
        """At low param, output should show flat stretches between flips."""
        rng = np.random.default_rng(0)
        out = flow_exciter(n_samples=44_100, param=0.1, rng=rng)
        # At param=0.1, mix_weight=0.0001, so sample-to-sample diff is
        # bounded by |r - state| * 0.0001 ~ 0.0001 in magnitude between
        # flips. Count how many adjacent-sample diffs are small.
        small_diffs = np.sum(np.abs(np.diff(out)) < 0.001) / (out.size - 1)
        assert small_diffs > 0.95, (
            f"expected >95% small diffs at low param, got {small_diffs:.4f}"
        )


class TestFlowExciterDeterminism:
    """Determinism under fixed seed."""

    def test_deterministic_same_rng_seed(self) -> None:
        rng_a = np.random.default_rng(12345)
        rng_b = np.random.default_rng(12345)
        a = flow_exciter(n_samples=2048, param=0.4, rng=rng_a)
        b = flow_exciter(n_samples=2048, param=0.4, rng=rng_b)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_produce_different_output(self) -> None:
        a = flow_exciter(n_samples=2048, param=0.4, rng=np.random.default_rng(1))
        b = flow_exciter(n_samples=2048, param=0.4, rng=np.random.default_rng(2))
        assert not np.array_equal(a, b)


class TestFlowExciterValidation:
    """Input validation."""

    def test_param_out_of_range_raises(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            flow_exciter(n_samples=100, param=-0.1, rng=rng)
        with pytest.raises(ValueError):
            flow_exciter(n_samples=100, param=1.5, rng=rng)

    def test_negative_n_samples_raises(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            flow_exciter(n_samples=-10, param=0.5, rng=rng)


class TestFlowExciterAdditiveIntegration:
    """Integration with the additive engine."""

    def test_additive_renders_with_flow_noise_mode(self) -> None:
        from code_musics.engines import additive

        params = {
            "partials": [{"ratio": 1.0, "amp": 1.0, "noise": 0.6}],
            "noise_mode": "flow",
            "noise_amount": 0.6,
            "noise_bandwidth_hz": 200.0,
            "flow_density": 0.4,
        }
        signal = additive.render(
            freq=220.0,
            duration=0.5,
            amp=0.5,
            sample_rate=44_100,
            params=params,
        )
        assert signal.shape == (int(44_100 * 0.5),)
        assert np.all(np.isfinite(signal))
        assert np.max(np.abs(signal)) > 0.0

    def test_additive_default_noise_mode_still_works(self) -> None:
        """Default noise mode (white) must remain unchanged as the default."""
        from code_musics.engines import additive

        params = {
            "partials": [{"ratio": 1.0, "amp": 1.0, "noise": 0.4}],
            "noise_amount": 0.4,
            "noise_bandwidth_hz": 150.0,
        }
        signal = additive.render(
            freq=220.0,
            duration=0.3,
            amp=0.5,
            sample_rate=44_100,
            params=params,
        )
        assert signal.shape == (int(44_100 * 0.3),)
        assert np.all(np.isfinite(signal))

    def test_flow_noise_mode_sounds_different_from_white(self) -> None:
        """The flow noise mode should produce a distinguishable output."""
        from code_musics.engines import additive

        base_params = {
            "partials": [{"ratio": 1.0, "amp": 0.01, "noise": 1.0}],
            "noise_amount": 1.0,
            "noise_bandwidth_hz": 400.0,
        }
        white = additive.render(
            freq=440.0,
            duration=0.3,
            amp=0.5,
            sample_rate=44_100,
            params={**base_params, "noise_mode": "white"},
        )
        flow = additive.render(
            freq=440.0,
            duration=0.3,
            amp=0.5,
            sample_rate=44_100,
            params={**base_params, "noise_mode": "flow", "flow_density": 0.2},
        )
        assert not np.array_equal(white, flow), (
            "flow noise mode should differ from white noise"
        )


class TestFlowExciterInvalidNoiseMode:
    """Additive engine validates noise_mode values."""

    def test_unknown_noise_mode_raises(self) -> None:
        from code_musics.engines import additive

        params = {
            "partials": [{"ratio": 1.0, "amp": 1.0, "noise": 0.3}],
            "noise_amount": 0.3,
            "noise_mode": "not_a_mode",
        }
        with pytest.raises(ValueError):
            additive.render(
                freq=220.0,
                duration=0.2,
                amp=0.5,
                sample_rate=44_100,
                params=params,
            )
