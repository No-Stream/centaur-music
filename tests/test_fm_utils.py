"""Tests for the fm_modulate() reusable DSP primitive."""

from __future__ import annotations

import numpy as np

from code_musics.engines._dsp_utils import fm_modulate


def test_fm_modulate_zero_index_is_clean_sine() -> None:
    """With mod_index=0, output should be a clean sine wave."""
    sample_rate = 44_100
    duration = 0.1
    freq = 440.0
    n_samples = int(sample_rate * duration)
    carrier_freq_profile = np.full(n_samples, freq, dtype=np.float64)

    result = fm_modulate(
        carrier_freq_profile,
        mod_ratio=1.5,
        mod_index=0.0,
        sample_rate=sample_rate,
    )

    assert result.shape == (n_samples,)
    assert np.isfinite(result).all()

    # With zero index the modulator has no effect -- output should be a pure sine.
    # The loop writes sin(phase) then increments, so sample[0] = sin(0).
    phase_inc = 2.0 * np.pi * freq / sample_rate
    phase = phase_inc * np.arange(n_samples, dtype=np.float64)
    expected = np.sin(phase)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_fm_modulate_nonzero_index_adds_harmonics() -> None:
    """Spectral content should increase with higher modulation index."""
    sample_rate = 44_100
    duration = 0.2
    freq = 200.0
    n_samples = int(sample_rate * duration)
    carrier_freq_profile = np.full(n_samples, freq, dtype=np.float64)

    clean = fm_modulate(
        carrier_freq_profile,
        mod_ratio=1.0,
        mod_index=0.0,
        sample_rate=sample_rate,
    )
    modulated = fm_modulate(
        carrier_freq_profile,
        mod_ratio=1.0,
        mod_index=4.0,
        sample_rate=sample_rate,
    )

    # Count spectral bins above a threshold to measure harmonic richness
    def spectral_spread(signal: np.ndarray) -> int:
        spectrum = np.abs(np.fft.rfft(signal))
        threshold = np.max(spectrum) * 0.01
        return int(np.sum(spectrum > threshold))

    assert spectral_spread(modulated) > spectral_spread(clean)


def test_fm_modulate_index_envelope_decays() -> None:
    """Index envelope should shape the spectral evolution over time."""
    sample_rate = 44_100
    duration = 0.2
    freq = 200.0
    n_samples = int(sample_rate * duration)
    carrier_freq_profile = np.full(n_samples, freq, dtype=np.float64)

    # Decaying envelope: rich attack, clean tail
    index_envelope = np.exp(
        -np.arange(n_samples, dtype=np.float64) / (0.02 * sample_rate)
    )

    result = fm_modulate(
        carrier_freq_profile,
        mod_ratio=1.41,
        mod_index=5.0,
        sample_rate=sample_rate,
        index_envelope=index_envelope,
    )

    assert result.shape == (n_samples,)
    assert np.isfinite(result).all()

    # The first quarter should have more spectral content than the last quarter
    quarter = n_samples // 4
    first_quarter = result[:quarter]
    last_quarter = result[-quarter:]

    def spectral_energy_above_fundamental(signal: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(signal))
        fund_bin = int(round(freq / (sample_rate / len(signal))))
        above_fund = spectrum[fund_bin + 3 :]
        return float(np.sum(above_fund**2))

    attack_energy = spectral_energy_above_fundamental(first_quarter)
    tail_energy = spectral_energy_above_fundamental(last_quarter)
    assert attack_energy > tail_energy


def test_fm_modulate_feedback_adds_complexity() -> None:
    """Non-zero feedback should produce a different signal than zero feedback."""
    sample_rate = 44_100
    duration = 0.1
    freq = 300.0
    n_samples = int(sample_rate * duration)
    carrier_freq_profile = np.full(n_samples, freq, dtype=np.float64)

    no_fb = fm_modulate(
        carrier_freq_profile,
        mod_ratio=1.0,
        mod_index=2.0,
        sample_rate=sample_rate,
        feedback=0.0,
    )
    with_fb = fm_modulate(
        carrier_freq_profile,
        mod_ratio=1.0,
        mod_index=2.0,
        sample_rate=sample_rate,
        feedback=0.5,
    )

    assert not np.allclose(no_fb, with_fb)
    assert np.isfinite(with_fb).all()
