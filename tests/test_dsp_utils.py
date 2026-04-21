"""Tests for code_musics.engines._dsp_utils: build_drift and render_noise_floor."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._dsp_utils import (
    MAX_DRIFT_CENTS,
    build_drift,
    render_noise_floor,
)

# ---------------------------------------------------------------------------
# build_drift tests
# ---------------------------------------------------------------------------


class TestBuildDriftEdgeCases:
    """Zero-drift and backward-compatibility guards."""

    def test_drift_zero_returns_ones(self) -> None:
        result = build_drift(
            n_samples=1024,
            drift_amount=0.0,
            drift_rate_hz=0.3,
            duration=1.0,
            phase_offset=0.0,
        )
        np.testing.assert_array_equal(result, np.ones(1024, dtype=np.float64))

    def test_drift_zero_samples_returns_empty(self) -> None:
        result = build_drift(
            n_samples=0,
            drift_amount=0.5,
            drift_rate_hz=0.3,
            duration=1.0,
            phase_offset=0.0,
        )
        assert result.shape == (0,)

    def test_drift_backward_compatible_no_rng(self) -> None:
        """Calling without rng should not raise TypeError."""
        result = build_drift(
            n_samples=512,
            drift_amount=0.5,
            drift_rate_hz=0.3,
            duration=1.0,
            phase_offset=0.0,
        )
        assert result.shape == (512,)
        assert result.dtype == np.float64


class TestBuildDriftBoundedExcursion:
    """Peak cents excursion must stay within MAX_DRIFT_CENTS * drift_amount."""

    @pytest.mark.parametrize("drift_amount", [0.1, 0.5, 1.0, 2.0])
    def test_bounded_excursion_without_rng(self, drift_amount: float) -> None:
        result = build_drift(
            n_samples=44100,
            drift_amount=drift_amount,
            drift_rate_hz=0.3,
            duration=1.0,
            phase_offset=0.0,
        )
        cents = 1200.0 * np.log2(result)
        max_expected_cents = MAX_DRIFT_CENTS * drift_amount
        assert np.max(np.abs(cents)) <= max_expected_cents + 1e-6

    @pytest.mark.parametrize("drift_amount", [0.1, 0.5, 1.0, 2.0])
    def test_bounded_excursion_with_rng(self, drift_amount: float) -> None:
        rng = np.random.default_rng(42)
        result = build_drift(
            n_samples=44100,
            drift_amount=drift_amount,
            drift_rate_hz=0.3,
            duration=1.0,
            phase_offset=0.0,
            rng=rng,
        )
        cents = 1200.0 * np.log2(result)
        max_expected_cents = MAX_DRIFT_CENTS * drift_amount
        assert np.max(np.abs(cents)) <= max_expected_cents + 1e-6


class TestBuildDriftDeterminism:
    """Same inputs must produce identical output."""

    COMMON_KWARGS: dict = dict(
        n_samples=8192,
        drift_amount=0.7,
        drift_rate_hz=0.25,
        duration=2.0,
        phase_offset=1.5,
    )

    def test_deterministic_with_rng(self) -> None:
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        a = build_drift(**self.COMMON_KWARGS, rng=rng1)
        b = build_drift(**self.COMMON_KWARGS, rng=rng2)
        np.testing.assert_array_equal(a, b)

    def test_deterministic_without_rng(self) -> None:
        a = build_drift(**self.COMMON_KWARGS)
        b = build_drift(**self.COMMON_KWARGS)
        np.testing.assert_array_equal(a, b)


class TestBuildDriftAperiodic:
    """Multi-rate drift should not be a pure sinusoid."""

    def test_autocorrelation_below_threshold(self) -> None:
        """Autocorrelation at the base drift period should be < 0.95."""
        sample_rate = 44100
        drift_rate_hz = 0.3
        duration = 10.0
        n_samples = int(sample_rate * duration)

        rng = np.random.default_rng(42)
        result = build_drift(
            n_samples=n_samples,
            drift_amount=1.0,
            drift_rate_hz=drift_rate_hz,
            duration=duration,
            phase_offset=0.0,
            rng=rng,
        )
        cents = 1200.0 * np.log2(result)
        cents_normed = cents - np.mean(cents)

        period_samples = int(sample_rate / drift_rate_hz)
        if period_samples >= n_samples:
            pytest.skip("period longer than signal")

        autocorr_at_period = float(
            np.sum(cents_normed[:-period_samples] * cents_normed[period_samples:])
        ) / float(np.sum(cents_normed**2))

        assert autocorr_at_period < 0.95, (
            f"autocorrelation at drift period = {autocorr_at_period:.3f}, "
            "drift looks too periodic (pure sinusoid)"
        )


# ---------------------------------------------------------------------------
# render_noise_floor tests
# ---------------------------------------------------------------------------


class TestNoiseFloorSpectrum:
    """Pink noise floor spectral characteristics."""

    def test_pink_noise_spectrum_rolloff(self) -> None:
        """Spectral power should roll off roughly 3 dB/octave."""
        sample_rate = 44100
        n_samples = sample_rate * 2  # 2 seconds
        rng = np.random.default_rng(42)

        # Constant-amplitude signal so envelope following is uniform
        signal = np.full(n_samples, 0.5, dtype=np.float64)

        noise_floor = render_noise_floor(
            signal=signal,
            sample_rate=sample_rate,
            n_samples=n_samples,
            rng=rng,
        )

        spectrum = np.abs(np.fft.rfft(noise_floor))
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)

        # Compare energy in two octave bands: 250-500 Hz vs 1000-2000 Hz
        low_mask = (freqs >= 250) & (freqs < 500)
        high_mask = (freqs >= 1000) & (freqs < 2000)

        low_power_db = 10.0 * np.log10(np.mean(spectrum[low_mask] ** 2) + 1e-30)
        high_power_db = 10.0 * np.log10(np.mean(spectrum[high_mask] ** 2) + 1e-30)

        # Pink noise: power spectral density ~ 1/f, so 1 octave up => ~3 dB down.
        # With 2 octaves separation, expect ~6 dB difference. Allow generous margin.
        rolloff_db = low_power_db - high_power_db
        assert rolloff_db > 2.0, (
            f"expected rolloff between low and high bands, got {rolloff_db:.1f} dB"
        )


class TestNoiseFloorLevel:
    """Noise floor output level should be around -60 dBFS relative to signal peak."""

    def test_noise_floor_level(self) -> None:
        sample_rate = 44100
        n_samples = sample_rate
        rng = np.random.default_rng(7)

        signal = np.full(n_samples, 0.5, dtype=np.float64)
        noise_floor = render_noise_floor(
            signal=signal,
            sample_rate=sample_rate,
            n_samples=n_samples,
            rng=rng,
        )

        signal_peak = np.max(np.abs(signal))
        noise_peak = np.max(np.abs(noise_floor))
        ratio_db = 20.0 * np.log10(noise_peak / signal_peak + 1e-30)

        # Expect around -60 dBFS; allow range of -70 to -45 dBFS
        assert -70.0 < ratio_db < -45.0, (
            f"noise floor level {ratio_db:.1f} dBFS outside expected range"
        )

    def test_custom_level_scales_output(self) -> None:
        """Passing a higher level should produce proportionally louder noise."""
        sample_rate = 44100
        n_samples = sample_rate
        signal = np.full(n_samples, 0.5, dtype=np.float64)

        rng_default = np.random.default_rng(7)
        noise_default = render_noise_floor(
            signal=signal,
            sample_rate=sample_rate,
            n_samples=n_samples,
            rng=rng_default,
        )

        rng_loud = np.random.default_rng(7)
        noise_loud = render_noise_floor(
            signal=signal,
            sample_rate=sample_rate,
            n_samples=n_samples,
            rng=rng_loud,
            level=0.01,
        )

        default_rms = float(np.sqrt(np.mean(noise_default**2)))
        loud_rms = float(np.sqrt(np.mean(noise_loud**2)))

        # 0.01 / 0.001 = 10x louder; check within a factor-of-2 tolerance
        ratio = loud_rms / max(default_rms, 1e-30)
        assert 5.0 < ratio < 15.0, f"expected ~10x ratio, got {ratio:.2f}"


class TestNoiseFloorEnvelopeFollowing:
    """Noise should be louder where the signal is louder."""

    def test_envelope_following(self) -> None:
        sample_rate = 44100
        n_samples = sample_rate * 2  # 2 seconds
        rng = np.random.default_rng(55)

        # Signal: loud first half, quiet second half
        signal = np.zeros(n_samples, dtype=np.float64)
        signal[: n_samples // 2] = 0.8
        signal[n_samples // 2 :] = 0.05

        noise_floor = render_noise_floor(
            signal=signal,
            sample_rate=sample_rate,
            n_samples=n_samples,
            rng=rng,
        )

        first_half_rms = float(np.sqrt(np.mean(noise_floor[: n_samples // 2] ** 2)))
        second_half_rms = float(np.sqrt(np.mean(noise_floor[n_samples // 2 :] ** 2)))

        assert first_half_rms > second_half_rms * 2.0, (
            f"noise in loud section ({first_half_rms:.6f}) should be > 2x "
            f"noise in quiet section ({second_half_rms:.6f})"
        )
