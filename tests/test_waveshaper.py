"""Tests for per-oscillator waveshaping distortion algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._waveshaper import ALGORITHM_NAMES, apply_waveshaper


def _sine(freq: float = 440.0, duration: float = 0.05, sr: int = 44100) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float64)
    return np.sin(2.0 * np.pi * freq * t)


@pytest.mark.parametrize("algorithm", sorted(ALGORITHM_NAMES))
def test_each_algorithm_produces_nonzero_output(algorithm: str) -> None:
    signal = _sine()
    result = apply_waveshaper(signal, algorithm=algorithm, drive=0.5)
    assert result.shape == signal.shape
    assert np.all(np.isfinite(result))
    assert np.max(np.abs(result)) > 0.0


def test_drive_zero_is_near_passthrough() -> None:
    # Use a low-amplitude signal where tanh(x) ~ x is a good approximation.
    # At peak=0.1 and drive_gain=1.0, tanh distortion is negligible.
    signal = _sine() * 0.1
    result = apply_waveshaper(signal, algorithm="tanh", drive=0.0)
    np.testing.assert_allclose(result, signal, atol=0.005)


def test_mix_zero_is_passthrough() -> None:
    signal = _sine()
    result = apply_waveshaper(signal, algorithm="hard_clip", drive=0.8, mix=0.0)
    np.testing.assert_array_equal(result, signal)


def test_mix_one_is_fully_wet() -> None:
    signal = _sine()
    wet_only = apply_waveshaper(signal, algorithm="tanh", drive=0.6, mix=1.0)
    blended = apply_waveshaper(signal, algorithm="tanh", drive=0.6, mix=0.5)
    # Blended should differ from both dry and wet
    assert not np.allclose(wet_only, signal, atol=1e-6)
    assert not np.allclose(blended, wet_only, atol=1e-6)
    assert not np.allclose(blended, signal, atol=1e-6)


def test_foldback_adds_harmonics() -> None:
    signal = _sine(freq=440.0, duration=0.1)
    result = apply_waveshaper(signal, algorithm="foldback", drive=0.7)

    # Compare spectral content: foldback should add harmonics beyond the fundamental
    input_spectrum = np.abs(np.fft.rfft(signal))
    output_spectrum = np.abs(np.fft.rfft(result))

    # Energy above 2x fundamental should increase
    sr = 44100
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sr)
    high_mask = freqs > 880.0
    input_high_energy = float(np.sum(input_spectrum[high_mask] ** 2))
    output_high_energy = float(np.sum(output_spectrum[high_mask] ** 2))
    assert output_high_energy > input_high_energy * 2.0


def test_tanh_stays_bounded() -> None:
    # Loud signal with drive=1.0
    signal = _sine() * 5.0
    result = apply_waveshaper(signal, algorithm="tanh", drive=1.0)
    assert np.all(np.isfinite(result))
    # tanh output before compensation is bounded by [-1, 1].
    # After RMS compensation it can exceed 1 but should still be reasonable.
    assert np.max(np.abs(result)) < 20.0


def test_drive_envelope_modulates_over_time() -> None:
    sr = 44100
    duration = 0.1
    n = int(sr * duration)
    signal = _sine(freq=440.0, duration=duration, sr=sr)

    # Envelope ramps from 0 to 1
    envelope = np.linspace(0.0, 1.0, n, dtype=np.float64)
    result = apply_waveshaper(
        signal, algorithm="hard_clip", drive=0.8, drive_envelope=envelope
    )

    # First quarter should be cleaner (less distortion) than last quarter.
    # Measure by comparing to dry signal in each segment.
    quarter = n // 4
    first_quarter_diff = np.sqrt(np.mean((result[:quarter] - signal[:quarter]) ** 2))
    last_quarter_diff = np.sqrt(np.mean((result[-quarter:] - signal[-quarter:]) ** 2))
    assert last_quarter_diff > first_quarter_diff


def test_invalid_algorithm_raises() -> None:
    signal = _sine()
    with pytest.raises(ValueError, match="Unknown waveshaper algorithm"):
        apply_waveshaper(signal, algorithm="nonexistent", drive=0.5)


# ---------------------------------------------------------------------------
# linear_fold wavefolder tests
# ---------------------------------------------------------------------------


def test_linear_fold_identity_low_drive() -> None:
    """At drive~1 (gain=1), small signals pass through mostly unchanged."""
    signal = _sine() * 0.1
    result = apply_waveshaper(signal, algorithm="linear_fold", drive=0.0)
    np.testing.assert_allclose(result, signal, atol=0.05)


def test_linear_fold_folds_at_high_drive() -> None:
    """At high drive, output stays bounded in [-1, 1] before RMS compensation."""
    signal = _sine()
    result = apply_waveshaper(signal, algorithm="linear_fold", drive=0.8, mix=1.0)
    assert np.all(np.isfinite(result))
    # After RMS compensation it can exceed 1 but must remain reasonable
    assert np.max(np.abs(result)) < 20.0


# ---------------------------------------------------------------------------
# sine_fold wavefolder tests
# ---------------------------------------------------------------------------


def test_sine_fold_identity_low_drive() -> None:
    """sin(x * 1.0 * pi) ~ x * pi for small x; after RMS compensation ~ input."""
    signal = _sine() * 0.05
    result = apply_waveshaper(signal, algorithm="sine_fold", drive=0.0)
    np.testing.assert_allclose(result, signal, atol=0.05)


def test_sine_fold_bounded() -> None:
    """sin() is bounded [-1, 1] before compensation, regardless of input."""
    signal = _sine() * 5.0
    result = apply_waveshaper(signal, algorithm="sine_fold", drive=1.0)
    assert np.all(np.isfinite(result))
    assert np.max(np.abs(result)) < 20.0


# ---------------------------------------------------------------------------
# Shared wavefolder tests
# ---------------------------------------------------------------------------


def test_wavefolders_in_algorithms_dict() -> None:
    """Both new wavefolders are registered in ALGORITHM_NAMES."""
    assert "linear_fold" in ALGORITHM_NAMES
    assert "sine_fold" in ALGORITHM_NAMES


@pytest.mark.parametrize("algorithm", ["linear_fold", "sine_fold"])
def test_apply_waveshaper_with_folds(algorithm: str) -> None:
    """apply_waveshaper() works end-to-end with both wavefolder algorithms."""
    signal = _sine()
    result = apply_waveshaper(signal, algorithm=algorithm, drive=0.5)
    assert result.shape == signal.shape
    assert np.all(np.isfinite(result))
    assert np.max(np.abs(result)) > 0.0


@pytest.mark.parametrize("algorithm", ["linear_fold", "sine_fold"])
def test_wavefolder_with_envelope(algorithm: str) -> None:
    """Both wavefolders work with per-sample drive envelopes."""
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

    # First quarter (low drive) should differ less from dry than last quarter (high drive)
    quarter = n // 4
    first_quarter_diff = np.sqrt(np.mean((result[:quarter] - signal[:quarter]) ** 2))
    last_quarter_diff = np.sqrt(np.mean((result[-quarter:] - signal[-quarter:]) ** 2))
    assert last_quarter_diff > first_quarter_diff
