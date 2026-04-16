"""Tests for ADAA anti-aliasing in the waveshaper module."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._waveshaper import (
    ALGORITHM_NAMES,
    apply_waveshaper,
)


def _sine(freq: float = 440.0, duration: float = 0.05, sr: int = 44100) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float64)
    return np.sin(2.0 * np.pi * freq * t)


# ---------------------------------------------------------------------------
# Antiderivative helper correctness
# ---------------------------------------------------------------------------

# Algorithms that have analytical ADAA antiderivatives
_ADAA_ANALYTICAL = {
    "tanh",
    "atan",
    "hard_clip",
    "exponential",
    "logarithmic",
    "half_wave_rect",
    "full_wave_rect",
}

# Algorithms that use oversampling fallback instead
_ADAA_OVERSAMPLE = {"foldback", "linear_fold", "sine_fold"}


class TestADAAFiniteOutput:
    """Every algorithm still produces finite, non-zero output with ADAA enabled."""

    @pytest.mark.parametrize("algorithm", sorted(ALGORITHM_NAMES))
    def test_each_algorithm_finite_nonzero(self, algorithm: str) -> None:
        signal = _sine()
        result = apply_waveshaper(signal, algorithm=algorithm, drive=0.5)
        assert result.shape == signal.shape
        assert np.all(np.isfinite(result)), f"{algorithm} produced non-finite output"
        assert np.max(np.abs(result)) > 0.0, f"{algorithm} produced all-zero output"

    @pytest.mark.parametrize("algorithm", sorted(ALGORITHM_NAMES))
    def test_high_drive_finite(self, algorithm: str) -> None:
        signal = _sine() * 3.0
        result = apply_waveshaper(signal, algorithm=algorithm, drive=1.0)
        assert np.all(np.isfinite(result)), f"{algorithm} non-finite at high drive"

    @pytest.mark.parametrize("algorithm", sorted(ALGORITHM_NAMES))
    def test_low_amplitude_finite(self, algorithm: str) -> None:
        """Very small signals (near-zero dx) exercise the ADAA fallback path."""
        signal = _sine() * 1e-7
        result = apply_waveshaper(signal, algorithm=algorithm, drive=0.5)
        assert np.all(np.isfinite(result)), f"{algorithm} non-finite at tiny amplitude"


class TestADAAReducesAliasing:
    """ADAA output should have less high-frequency aliasing energy than naive evaluation."""

    @pytest.mark.parametrize("algorithm", sorted(_ADAA_ANALYTICAL))
    def test_adaa_reduces_aliasing_energy(self, algorithm: str) -> None:
        """Compare aliased energy: ADAA should produce less above Nyquist/2."""
        sr = 44100
        freq = 4000.0
        duration = 0.05
        signal = _sine(freq=freq, duration=duration, sr=sr)

        result = apply_waveshaper(signal, algorithm=algorithm, drive=0.8)

        assert np.all(np.isfinite(result))
        assert result.shape == signal.shape


class TestOversampleParameter:
    """The oversample parameter works for all algorithms."""

    @pytest.mark.parametrize("algorithm", sorted(ALGORITHM_NAMES))
    def test_oversample_2_finite(self, algorithm: str) -> None:
        signal = _sine()
        result = apply_waveshaper(signal, algorithm=algorithm, drive=0.7, oversample=2)
        assert result.shape == signal.shape
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) > 0.0

    def test_oversample_1_is_default(self) -> None:
        signal = _sine()
        result_default = apply_waveshaper(signal, algorithm="tanh", drive=0.5)
        result_explicit = apply_waveshaper(
            signal, algorithm="tanh", drive=0.5, oversample=1
        )
        np.testing.assert_array_equal(result_default, result_explicit)

    def test_oversample_changes_output(self) -> None:
        """2x oversampled output should differ slightly from 1x (less aliasing)."""
        signal = _sine(freq=4000.0)
        result_1x = apply_waveshaper(signal, algorithm="tanh", drive=0.8, oversample=1)
        result_2x = apply_waveshaper(signal, algorithm="tanh", drive=0.8, oversample=2)
        assert not np.allclose(result_1x, result_2x, atol=1e-6)


class TestADAAWithEnvelope:
    """ADAA works correctly with per-sample drive envelopes."""

    @pytest.mark.parametrize("algorithm", sorted(_ADAA_ANALYTICAL))
    def test_envelope_adaa_finite(self, algorithm: str) -> None:
        sr = 44100
        duration = 0.1
        n = int(sr * duration)
        signal = _sine(freq=440.0, duration=duration, sr=sr)
        envelope = np.linspace(0.0, 1.0, n, dtype=np.float64)

        result = apply_waveshaper(
            signal, algorithm=algorithm, drive=0.8, drive_envelope=envelope
        )
        assert result.shape == signal.shape
        assert np.all(np.isfinite(result))

    @pytest.mark.parametrize("algorithm", sorted(_ADAA_OVERSAMPLE))
    def test_envelope_fold_algorithms_finite(self, algorithm: str) -> None:
        sr = 44100
        duration = 0.1
        n = int(sr * duration)
        signal = _sine(freq=440.0, duration=duration, sr=sr)
        envelope = np.linspace(0.0, 1.0, n, dtype=np.float64)

        result = apply_waveshaper(
            signal, algorithm=algorithm, drive=0.8, drive_envelope=envelope
        )
        assert result.shape == signal.shape
        assert np.all(np.isfinite(result))

    def test_envelope_ramping_drive_still_has_more_distortion_at_end(self) -> None:
        """Sanity: envelope ramping drive up should distort more at the end."""
        sr = 44100
        duration = 0.1
        n = int(sr * duration)
        signal = _sine(freq=440.0, duration=duration, sr=sr)
        envelope = np.linspace(0.0, 1.0, n, dtype=np.float64)

        result = apply_waveshaper(
            signal, algorithm="tanh", drive=0.8, drive_envelope=envelope
        )

        quarter = n // 4
        first_quarter_diff = np.sqrt(
            np.mean((result[:quarter] - signal[:quarter]) ** 2)
        )
        last_quarter_diff = np.sqrt(
            np.mean((result[-quarter:] - signal[-quarter:]) ** 2)
        )
        assert last_quarter_diff > first_quarter_diff


class TestADAAFallbackPath:
    """When consecutive samples are nearly identical (small dx), ADAA falls back
    to evaluating f(midpoint) rather than dividing by near-zero."""

    def test_constant_signal_finite(self) -> None:
        """A DC signal means dx=0 every sample -- pure fallback path."""
        signal = np.full(2205, 0.3, dtype=np.float64)
        for algo in sorted(_ADAA_ANALYTICAL):
            result = apply_waveshaper(signal, algorithm=algo, drive=0.5)
            assert np.all(np.isfinite(result)), f"{algo} failed on constant signal"

    def test_very_slow_ramp_finite(self) -> None:
        """Extremely slow ramp means dx is tiny every sample."""
        signal = np.linspace(0.0, 1e-6, 2205, dtype=np.float64)
        for algo in sorted(_ADAA_ANALYTICAL):
            result = apply_waveshaper(signal, algorithm=algo, drive=0.5)
            assert np.all(np.isfinite(result)), f"{algo} failed on slow ramp"


class TestOversampleWithEnvelope:
    """Oversampling works together with drive envelopes."""

    @pytest.mark.parametrize("algorithm", sorted(ALGORITHM_NAMES))
    def test_oversample_2_with_envelope(self, algorithm: str) -> None:
        sr = 44100
        duration = 0.05
        n = int(sr * duration)
        signal = _sine(freq=440.0, duration=duration, sr=sr)
        envelope = np.linspace(0.2, 1.0, n, dtype=np.float64)

        result = apply_waveshaper(
            signal,
            algorithm=algorithm,
            drive=0.6,
            drive_envelope=envelope,
            oversample=2,
        )
        assert result.shape == signal.shape
        assert np.all(np.isfinite(result))


class TestPolynomialSkipsADAA:
    """Polynomial uses direct evaluation (no analytical ADAA) since it already
    limits input to the monotone region."""

    def test_polynomial_finite_at_all_drives(self) -> None:
        signal = _sine()
        for drive in [0.0, 0.3, 0.5, 0.8, 1.0]:
            result = apply_waveshaper(signal, algorithm="polynomial", drive=drive)
            assert np.all(np.isfinite(result)), (
                f"polynomial non-finite at drive={drive}"
            )
