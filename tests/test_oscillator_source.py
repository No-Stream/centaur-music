"""Tests for ``OscillatorSource`` and the extended per-sample destination set.

``OscillatorSource`` is the audio-rate sibling of :class:`LFOSource`: it has
no upper rate cap and is the intended modulation source for per-sample
destinations like ``pulse_width``, ``osc2_detune_cents``, and
``osc_spread_cents``.  These tests cover basic spectral correctness,
determinism, stereo decorrelation, and the widened allowlist.
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.modulation import (
    _PER_SAMPLE_SYNTH_DESTINATIONS,
    OscillatorSource,
    SourceSamplingContext,
)
from code_musics.synth import SAMPLE_RATE


def _ctx(total_dur: float = 1.0) -> SourceSamplingContext:
    return SourceSamplingContext(sample_rate=SAMPLE_RATE, total_dur=total_dur)


def _fft_peak_hz(signal: np.ndarray, sample_rate: int) -> float:
    """Return the frequency of the strongest positive-frequency FFT bin."""
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    # Ignore DC to avoid trivially matching bin 0 for saws.
    spectrum[0] = 0.0
    return float(freqs[int(np.argmax(spectrum))])


class TestOscillatorSourceSine:
    def test_sine_at_low_rate(self) -> None:
        """2 Hz sine over 1 s at SAMPLE_RATE has FFT peak at 2 Hz."""
        source = OscillatorSource(rate_hz=2.0, waveshape="sine")
        duration_s = 1.0
        n_samples = int(SAMPLE_RATE * duration_s)
        times = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
        curve = source.sample(times, _ctx(total_dur=duration_s))

        assert curve.shape == times.shape
        assert np.all(np.isfinite(curve))
        assert curve.min() >= -1.0 - 1e-9
        assert curve.max() <= 1.0 + 1e-9
        peak_hz = _fft_peak_hz(curve, SAMPLE_RATE)
        assert abs(peak_hz - 2.0) < 1.0  # within one FFT bin (1 Hz here)

    def test_audio_rate(self) -> None:
        """200 Hz sine (above the LFO cap) renders cleanly with FFT peak at 200 Hz."""
        source = OscillatorSource(rate_hz=200.0, waveshape="sine")
        duration_s = 1.0
        n_samples = int(SAMPLE_RATE * duration_s)
        times = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
        curve = source.sample(times, _ctx(total_dur=duration_s))

        assert curve.shape == times.shape
        assert np.all(np.isfinite(curve))
        assert curve.min() >= -1.0 - 1e-9
        assert curve.max() <= 1.0 + 1e-9
        peak_hz = _fft_peak_hz(curve, SAMPLE_RATE)
        assert abs(peak_hz - 200.0) < 2.0

    def test_high_audio_rate_800_hz(self) -> None:
        """800 Hz sine (well above any LFO cap) still tracks its fundamental."""
        source = OscillatorSource(rate_hz=800.0, waveshape="sine")
        duration_s = 0.5
        n_samples = int(SAMPLE_RATE * duration_s)
        times = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
        curve = source.sample(times, _ctx(total_dur=duration_s))

        assert curve.shape == times.shape
        assert np.all(np.isfinite(curve))
        peak_hz = _fft_peak_hz(curve, SAMPLE_RATE)
        assert abs(peak_hz - 800.0) < 4.0


class TestOscillatorSourceSawAndTriangle:
    def test_saw_range_and_spectrum(self) -> None:
        """50 Hz saw is in [-1, 1] and has harmonics with ~1/n amplitude falloff."""
        rate_hz = 50.0
        source = OscillatorSource(rate_hz=rate_hz, waveshape="saw")
        duration_s = 1.0
        n_samples = int(SAMPLE_RATE * duration_s)
        times = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
        curve = source.sample(times, _ctx(total_dur=duration_s))

        assert curve.shape == times.shape
        assert np.all(np.isfinite(curve))
        assert curve.min() >= -1.0 - 1e-9
        assert curve.max() <= 1.0 + 1e-9

        spectrum = np.abs(np.fft.rfft(curve))
        freqs = np.fft.rfftfreq(curve.size, d=1.0 / SAMPLE_RATE)

        # Pick the bin closest to each harmonic, normalize by fundamental amp.
        def bin_at(hz: float) -> float:
            idx = int(np.argmin(np.abs(freqs - hz)))
            return float(spectrum[idx])

        a1 = bin_at(rate_hz)
        a2 = bin_at(2.0 * rate_hz)
        a3 = bin_at(3.0 * rate_hz)
        a4 = bin_at(4.0 * rate_hz)
        assert a1 > 0
        # Saw spectrum: a_n ~ 1/n.  Allow generous tolerance for naive saw.
        assert a2 / a1 == pytest.approx(1.0 / 2.0, rel=0.25)
        assert a3 / a1 == pytest.approx(1.0 / 3.0, rel=0.3)
        assert a4 / a1 == pytest.approx(1.0 / 4.0, rel=0.35)

    def test_triangle_range_and_spectrum(self) -> None:
        """50 Hz triangle is in [-1, 1] with odd harmonics falling ~1/n^2."""
        rate_hz = 50.0
        source = OscillatorSource(rate_hz=rate_hz, waveshape="triangle")
        duration_s = 1.0
        n_samples = int(SAMPLE_RATE * duration_s)
        times = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
        curve = source.sample(times, _ctx(total_dur=duration_s))

        assert curve.shape == times.shape
        assert np.all(np.isfinite(curve))
        assert curve.min() >= -1.0 - 1e-9
        assert curve.max() <= 1.0 + 1e-9

        spectrum = np.abs(np.fft.rfft(curve))
        freqs = np.fft.rfftfreq(curve.size, d=1.0 / SAMPLE_RATE)

        def bin_at(hz: float) -> float:
            idx = int(np.argmin(np.abs(freqs - hz)))
            return float(spectrum[idx])

        a1 = bin_at(rate_hz)
        a3 = bin_at(3.0 * rate_hz)
        a5 = bin_at(5.0 * rate_hz)
        a2 = bin_at(2.0 * rate_hz)
        a4 = bin_at(4.0 * rate_hz)
        assert a1 > 0
        # Triangle: odd harmonics fall as 1/n^2; even harmonics ~ 0.
        assert a3 / a1 == pytest.approx(1.0 / 9.0, rel=0.35)
        assert a5 / a1 == pytest.approx(1.0 / 25.0, rel=0.5)
        # Even harmonics should be dramatically weaker than the fundamental.
        assert a2 < a1 * 0.05
        assert a4 < a1 * 0.05


class TestOscillatorSourceStereo:
    def test_stereo_phase_offset(self) -> None:
        """stereo=True returns a (2, n) buffer with L != R."""
        source = OscillatorSource(
            rate_hz=100.0, waveshape="sine", stereo=True, stereo_phase_offset=0.25
        )
        duration_s = 0.1
        n_samples = int(SAMPLE_RATE * duration_s)
        times = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
        curve = source.sample(times, _ctx(total_dur=duration_s))

        assert curve.shape == (2, n_samples)
        assert np.all(np.isfinite(curve))
        left = curve[0]
        right = curve[1]
        assert not np.allclose(left, right)
        # Both channels still track the fundamental.
        peak_left = _fft_peak_hz(left, SAMPLE_RATE)
        peak_right = _fft_peak_hz(right, SAMPLE_RATE)
        assert abs(peak_left - 100.0) < 2.0
        assert abs(peak_right - 100.0) < 2.0

    def test_stereo_false_returns_mono(self) -> None:
        """Default (stereo=False) keeps the 1D mono contract used by the matrix."""
        source = OscillatorSource(rate_hz=100.0, waveshape="sine")
        times = np.arange(256, dtype=np.float64) / SAMPLE_RATE
        curve = source.sample(times, _ctx(total_dur=1.0))
        assert curve.ndim == 1
        assert curve.shape == times.shape


class TestOscillatorSourceDeterminism:
    def test_identical_outputs_for_identical_params(self) -> None:
        """Phase accumulator only — two renders must be bit-for-bit identical."""
        times = np.arange(4096, dtype=np.float64) / SAMPLE_RATE
        ctx = _ctx(total_dur=1.0)
        source_a = OscillatorSource(
            rate_hz=173.0, waveshape="saw", phase=0.3, stereo=False
        )
        source_b = OscillatorSource(
            rate_hz=173.0, waveshape="saw", phase=0.3, stereo=False
        )
        np.testing.assert_array_equal(
            source_a.sample(times, ctx), source_b.sample(times, ctx)
        )

    def test_repeated_calls_same_output(self) -> None:
        source = OscillatorSource(rate_hz=42.0, waveshape="triangle", phase=0.1)
        times = np.arange(2048, dtype=np.float64) / SAMPLE_RATE
        ctx = _ctx(total_dur=1.0)
        np.testing.assert_array_equal(
            source.sample(times, ctx), source.sample(times, ctx)
        )


class TestOscillatorSourceValidation:
    def test_zero_rate_rejected(self) -> None:
        with pytest.raises(ValueError, match="rate_hz"):
            OscillatorSource(rate_hz=0.0)

    def test_negative_rate_rejected(self) -> None:
        with pytest.raises(ValueError, match="rate_hz"):
            OscillatorSource(rate_hz=-5.0)

    def test_unsupported_waveshape_rejected(self) -> None:
        with pytest.raises(ValueError, match="waveshape"):
            OscillatorSource(rate_hz=1.0, waveshape="square")  # type: ignore[arg-type]

    def test_empty_times_returns_empty(self) -> None:
        source = OscillatorSource(rate_hz=10.0, waveshape="sine")
        curve = source.sample(np.zeros(0, dtype=np.float64), _ctx())
        assert curve.shape == (0,)

    def test_empty_times_stereo_returns_empty_2d(self) -> None:
        source = OscillatorSource(rate_hz=10.0, waveshape="sine", stereo=True)
        curve = source.sample(np.zeros(0, dtype=np.float64), _ctx())
        assert curve.shape == (2, 0)


class TestPerSampleDestinations:
    def test_extended_allowlist(self) -> None:
        """The expanded per-sample destination allowlist matches the plan."""
        expected = {
            "cutoff_hz",
            "pulse_width",
            "osc2_detune_cents",
            "osc2_freq_ratio",
            "osc_spread_cents",
        }
        assert expected == _PER_SAMPLE_SYNTH_DESTINATIONS
