"""Tests for Ornstein-Uhlenbeck cutoff drift in _dsp_utils and engine integration."""

from __future__ import annotations

import numpy as np

from code_musics.engines._dsp_utils import build_cutoff_drift

# ---------------------------------------------------------------------------
# build_cutoff_drift unit tests
# ---------------------------------------------------------------------------


class TestCutoffDriftEdgeCases:
    """Zero/degenerate inputs return all-ones (no modulation)."""

    def test_ou_returns_ones_when_zero(self) -> None:
        rng = np.random.default_rng(42)
        result = build_cutoff_drift(
            1024,
            amount_cents=0.0,
            rate_hz=0.3,
            rng=rng,
            sample_rate=44100,
        )
        np.testing.assert_array_equal(result, np.ones(1024, dtype=np.float64))

    def test_ou_returns_empty_for_zero_samples(self) -> None:
        rng = np.random.default_rng(42)
        result = build_cutoff_drift(
            0,
            amount_cents=30.0,
            rate_hz=0.3,
            rng=rng,
            sample_rate=44100,
        )
        assert result.shape == (0,)


class TestCutoffDriftMeanReverting:
    """Over a long trajectory at high spring rate, mean should be near zero."""

    def test_ou_mean_reverting(self) -> None:
        sample_rate = 44100
        duration_s = 10.0
        n_samples = int(sample_rate * duration_s)
        rng = np.random.default_rng(99)

        result = build_cutoff_drift(
            n_samples,
            amount_cents=30.0,
            rate_hz=2.0,  # high spring rate for faster reversion
            rng=rng,
            sample_rate=sample_rate,
        )

        # Convert multiplicative ratio back to cents
        cents = 1200.0 * np.log2(result)
        assert abs(np.mean(cents)) < 5.0, (
            f"O-U mean = {np.mean(cents):.2f} cents, expected near 0 (±5)"
        )


class TestCutoffDriftDeterminism:
    """Same rng seed must produce identical trajectory."""

    def test_ou_deterministic(self) -> None:
        kwargs: dict = dict(
            amount_cents=30.0,
            rate_hz=0.3,
            sample_rate=44100,
        )
        rng1 = np.random.default_rng(77)
        rng2 = np.random.default_rng(77)

        a = build_cutoff_drift(8192, rng=rng1, **kwargs)
        b = build_cutoff_drift(8192, rng=rng2, **kwargs)
        np.testing.assert_array_equal(a, b)


class TestCutoffDriftBounded:
    """Max excursion should stay within approximately 3x the amount_cents (O-U tails)."""

    def test_ou_bounded(self) -> None:
        sample_rate = 44100
        n_samples = int(sample_rate * 10.0)
        rng = np.random.default_rng(42)
        amount_cents = 30.0

        result = build_cutoff_drift(
            n_samples,
            amount_cents=amount_cents,
            rate_hz=0.3,
            rng=rng,
            sample_rate=sample_rate,
        )

        cents = 1200.0 * np.log2(result)
        max_excursion = np.max(np.abs(cents))
        # O-U is not hard-clamped; 4x RMS is a generous tail bound
        # (for a Gaussian-stationary O-U, 4-sigma events are ~1 in 16k)
        assert max_excursion < 4.0 * amount_cents, (
            f"max excursion = {max_excursion:.1f} cents, expected < {4.0 * amount_cents:.1f}"
        )


class TestCutoffDriftMultiplicative:
    """Output should be a ratio array centered near 1.0."""

    def test_ou_multiplicative(self) -> None:
        rng = np.random.default_rng(42)
        result = build_cutoff_drift(
            44100,
            amount_cents=30.0,
            rate_hz=0.3,
            rng=rng,
            sample_rate=44100,
        )

        # All values should be positive (multiplicative ratio)
        assert np.all(result > 0), "multiplicative ratios must be positive"
        # Mean should be near 1.0 (within ~1% for 30 cents RMS)
        assert abs(np.mean(result) - 1.0) < 0.02, (
            f"mean ratio = {np.mean(result):.4f}, expected near 1.0"
        )
        # Should not be exactly ones (there is actual modulation)
        assert not np.allclose(result, 1.0), "drift should produce actual modulation"


# ---------------------------------------------------------------------------
# Engine integration tests
# ---------------------------------------------------------------------------


class TestPolyBLEPCutoffDrift:
    """PolyBLEP engine should render finite audio with cutoff_drift enabled."""

    def test_polyblep_cutoff_drift(self) -> None:
        from code_musics.engines.polyblep import render

        signal = render(
            freq=220.0,
            duration=0.5,
            amp=0.8,
            sample_rate=44100,
            params={
                "waveform": "saw",
                "cutoff_hz": 1200.0,
                "cutoff_drift": 0.5,
            },
        )
        assert len(signal) == int(0.5 * 44100)
        assert np.all(np.isfinite(signal))
        assert np.max(np.abs(signal)) > 0

    def test_cutoff_drift_zero_is_static(self) -> None:
        """cutoff_drift=0 produces deterministic, unmodulated output distinct from drifted."""
        from code_musics.engines.polyblep import render

        kwargs: dict = dict(
            freq=220.0,
            duration=0.3,
            amp=0.7,
            sample_rate=44100,
        )
        static_params = {"waveform": "saw", "cutoff_hz": 1500.0, "cutoff_drift": 0.0}
        drifted_params = {"waveform": "saw", "cutoff_hz": 1500.0, "cutoff_drift": 1.0}

        a = render(**kwargs, params=static_params)
        b = render(**kwargs, params=static_params)
        np.testing.assert_allclose(a, b, atol=1e-12)

        drifted = render(**kwargs, params=drifted_params)
        assert not np.allclose(a, drifted, atol=1e-6)


class TestFilteredStackCutoffDrift:
    """Filtered-stack engine should render finite audio with cutoff_drift enabled."""

    def test_filtered_stack_cutoff_drift(self) -> None:
        from code_musics.engines.filtered_stack import render

        signal = render(
            freq=220.0,
            duration=0.5,
            amp=0.8,
            sample_rate=44100,
            params={
                "waveform": "saw",
                "n_harmonics": 12,
                "cutoff_hz": 1200.0,
                "cutoff_drift": 0.5,
            },
        )
        assert len(signal) == int(0.5 * 44100)
        assert np.all(np.isfinite(signal))
        assert np.max(np.abs(signal)) > 0

    def test_cutoff_drift_zero_is_static_filtered_stack(self) -> None:
        """cutoff_drift=0 produces deterministic, unmodulated output distinct from drifted."""
        from code_musics.engines.filtered_stack import render

        kwargs: dict = dict(
            freq=220.0,
            duration=0.3,
            amp=0.7,
            sample_rate=44100,
        )
        static_params = {
            "waveform": "saw",
            "n_harmonics": 12,
            "cutoff_hz": 1500.0,
            "cutoff_drift": 0.0,
        }
        drifted_params = {
            "waveform": "saw",
            "n_harmonics": 12,
            "cutoff_hz": 1500.0,
            "cutoff_drift": 1.0,
        }

        a = render(**kwargs, params=static_params)
        b = render(**kwargs, params=static_params)
        np.testing.assert_allclose(a, b, atol=1e-12)

        drifted = render(**kwargs, params=drifted_params)
        assert not np.allclose(a, drifted, atol=1e-6)
