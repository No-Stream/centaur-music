"""Tests for code_musics.engines._drum_utils shared drum DSP helpers."""

from __future__ import annotations

import numpy as np

from code_musics.engines._drum_utils import (
    bandpass_noise,
    bandpass_noise_windowed,
    integrated_phase,
    rng_for_note,
)


class TestRngForNote:
    """Deterministic RNG seeding from note parameters."""

    COMMON_KWARGS = dict(
        freq=110.0,
        duration=0.25,
        amp=0.8,
        sample_rate=44100,
        params={"decay_ms": 200.0},
    )

    def test_deterministic(self) -> None:
        rng1 = rng_for_note(**self.COMMON_KWARGS)
        rng2 = rng_for_note(**self.COMMON_KWARGS)
        seq1 = rng1.standard_normal(64)
        seq2 = rng2.standard_normal(64)
        np.testing.assert_array_equal(seq1, seq2)

    def test_different_params_different_seed(self) -> None:
        rng_a = rng_for_note(**self.COMMON_KWARGS)
        altered = {**self.COMMON_KWARGS, "params": {"decay_ms": 300.0}}
        rng_b = rng_for_note(**altered)
        seq_a = rng_a.standard_normal(64)
        seq_b = rng_b.standard_normal(64)
        assert not np.array_equal(seq_a, seq_b)

    def test_different_freq_different_seed(self) -> None:
        rng_a = rng_for_note(**self.COMMON_KWARGS)
        rng_b = rng_for_note(**{**self.COMMON_KWARGS, "freq": 220.0})
        assert not np.array_equal(rng_a.standard_normal(32), rng_b.standard_normal(32))


class TestBandpassNoise:
    """FFT-domain Gaussian bandpass shaping (narrow variant)."""

    def test_shapes_spectrum(self) -> None:
        rng = np.random.default_rng(42)
        sr = 44100
        center_hz = 2000.0
        signal = rng.standard_normal(sr)  # 1 second of noise

        shaped = bandpass_noise(signal, sample_rate=sr, center_hz=center_hz)

        spectrum = np.abs(np.fft.rfft(shaped))
        freqs = np.fft.rfftfreq(len(shaped), d=1.0 / sr)

        near_mask = np.abs(freqs - center_hz) < 400
        far_mask = np.abs(freqs - center_hz) > 4000
        near_energy = np.mean(spectrum[near_mask] ** 2)
        far_energy = np.mean(spectrum[far_mask] ** 2)

        assert near_energy > far_energy * 10, "energy near center should dominate"

    def test_output_length_matches_input(self) -> None:
        signal = np.random.default_rng(0).standard_normal(1024)
        out = bandpass_noise(signal, sample_rate=44100, center_hz=1000.0)
        assert out.shape == signal.shape


class TestBandpassNoiseWindowed:
    """FFT-domain Gaussian bandpass with hard band edges."""

    def test_shapes_spectrum(self) -> None:
        rng = np.random.default_rng(42)
        sr = 44100
        center_hz = 2000.0
        signal = rng.standard_normal(sr)

        shaped = bandpass_noise_windowed(
            signal, sample_rate=sr, center_hz=center_hz, width_ratio=0.75
        )

        spectrum = np.abs(np.fft.rfft(shaped))
        freqs = np.fft.rfftfreq(len(shaped), d=1.0 / sr)

        near_mask = np.abs(freqs - center_hz) < 300
        far_mask = np.abs(freqs - center_hz) > 4000
        near_energy = np.mean(spectrum[near_mask] ** 2)
        far_energy = np.mean(spectrum[far_mask] ** 2)

        assert near_energy > far_energy * 10

    def test_empty_signal_passthrough(self) -> None:
        out = bandpass_noise_windowed(
            np.array([], dtype=np.float64), sample_rate=44100, center_hz=1000.0
        )
        assert out.size == 0


class TestIntegratedPhase:
    """Cumulative phase from frequency profile."""

    def test_matches_cumsum(self) -> None:
        sr = 44100
        freq_profile = np.full(1000, 440.0, dtype=np.float64)
        result = integrated_phase(freq_profile, sample_rate=sr)
        expected = np.cumsum(2.0 * np.pi * freq_profile / sr)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_varying_frequency(self) -> None:
        sr = 44100
        freq_profile = np.linspace(100.0, 1000.0, 500, dtype=np.float64)
        result = integrated_phase(freq_profile, sample_rate=sr)
        expected = np.cumsum(2.0 * np.pi * freq_profile / sr)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_monotonically_increasing(self) -> None:
        sr = 44100
        freq_profile = np.full(256, 200.0, dtype=np.float64)
        result = integrated_phase(freq_profile, sample_rate=sr)
        assert np.all(np.diff(result) > 0)
