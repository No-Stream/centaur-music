"""Tests for the Machinedrum-E12-inspired extensions to the sample engine.

These tests exercise ``render_sample_segment`` directly with synthetic in-memory
buffers so they don't touch the filesystem-cached WAV loader in ``render``.
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines.sample import render_sample_segment

SR = 44100
DUR = 0.5
ROOT_FREQ = 440.0


@pytest.fixture()
def sine_buffer() -> np.ndarray:
    """A 0.5 s, 440 Hz sine at the test sample rate."""
    n = int(SR * DUR)
    t = np.arange(n, dtype=np.float64) / SR
    return np.sin(2.0 * np.pi * ROOT_FREQ * t)


@pytest.fixture()
def short_sine_buffer() -> np.ndarray:
    """A short (30 ms) 440 Hz sine — used for retrigger tests so each hit is
    naturally separated from the next."""
    n = int(SR * 0.03)
    t = np.arange(n, dtype=np.float64) / SR
    return np.sin(2.0 * np.pi * ROOT_FREQ * t)


def _spectral_centroid(signal: np.ndarray, sample_rate: int) -> float:
    if len(signal) == 0:
        return 0.0
    mag = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    total = float(mag.sum())
    if total < 1e-12:
        return 0.0
    return float((mag * freqs).sum() / total)


def _amplitude_envelope(signal: np.ndarray, window: int = 220) -> np.ndarray:
    """Short-window RMS envelope used for visually finding retrigger bumps."""
    abs_sig = np.abs(signal)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(abs_sig, kernel, mode="same")


def _dominant_freq(signal: np.ndarray, sample_rate: int) -> float:
    mag = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    return float(freqs[int(np.argmax(mag))])


class TestRetrigger:
    def test_retrigger_count_produces_n_bumps(
        self, short_sine_buffer: np.ndarray
    ) -> None:
        duration = 0.25
        interval_ms = 50.0
        result = render_sample_segment(
            short_sine_buffer,
            freq=ROOT_FREQ,
            duration=duration,
            amp=1.0,
            sample_rate=SR,
            params={
                "pitch_shift": False,
                "retrigger_count": 4,
                "retrigger_interval_ms": interval_ms,
            },
        )

        # With a 30 ms buffer and 50 ms retrigger interval, each trigger lives
        # in its own slot. Sample the envelope inside each expected window and
        # assert it's meaningfully above the silence between.
        env = _amplitude_envelope(result)
        interval_samples = int(interval_ms / 1000.0 * SR)
        buffer_samples = len(short_sine_buffer)
        threshold = 0.05 * float(env.max())

        bumps_present = 0
        for i in range(4):
            start = i * interval_samples
            end = min(start + buffer_samples, len(env))
            if start < len(env) and float(env[start:end].max()) > threshold:
                bumps_present += 1
        assert bumps_present >= 4

    def test_retrigger_pitch_step_shifts_each_trigger(
        self, short_sine_buffer: np.ndarray
    ) -> None:
        duration = 0.4
        interval_ms = 100.0
        result = render_sample_segment(
            short_sine_buffer,
            freq=ROOT_FREQ,
            duration=duration,
            amp=1.0,
            sample_rate=SR,
            params={
                "pitch_shift": False,
                "retrigger_count": 3,
                "retrigger_interval_ms": interval_ms,
                "retrigger_pitch_step_cents": 300.0,
                "retrigger_decay_curve": "linear",
            },
        )

        # With a short 30 ms buffer and 100 ms interval, each trigger is clean.
        # Analyze the first ~25 ms of each trigger window.
        interval_samples = int(interval_ms / 1000.0 * SR)
        seg_len = int(0.025 * SR)
        seg0 = result[0:seg_len]
        seg1 = result[interval_samples : interval_samples + seg_len]
        seg2 = result[2 * interval_samples : 2 * interval_samples + seg_len]

        f0 = _dominant_freq(seg0, SR)
        f1 = _dominant_freq(seg1, SR)
        f2 = _dominant_freq(seg2, SR)

        assert f1 > f0, f"expected f1 > f0, got {f0} -> {f1}"
        assert f2 > f1, f"expected f2 > f1, got {f1} -> {f2}"

    def test_no_retrigger_by_default(self, sine_buffer: np.ndarray) -> None:
        result = render_sample_segment(
            sine_buffer,
            freq=ROOT_FREQ,
            duration=DUR,
            amp=1.0,
            sample_rate=SR,
            params={"pitch_shift": False},
        )
        # With no retrigger and no decay, the playback is just the buffer
        # truncated/padded, then peak-normalized + scaled by amp. For a unit-amp
        # sine that peak-normalizes back to itself.
        assert np.max(np.abs(result)) == pytest.approx(1.0, abs=0.01)
        # And the envelope shouldn't show multiple bumps: first half and second
        # half should have similar mean energy.
        half = len(result) // 2
        first_energy = float(np.mean(np.abs(result[:half])))
        second_energy = float(np.mean(np.abs(result[half:])))
        assert abs(first_energy - second_energy) < 0.1


class TestBendEnvelope:
    def test_bend_envelope_shifts_pitch(self, sine_buffer: np.ndarray) -> None:
        duration = DUR
        result = render_sample_segment(
            sine_buffer,
            freq=ROOT_FREQ,
            duration=duration,
            amp=1.0,
            sample_rate=SR,
            params={
                "pitch_shift": False,
                "bend_envelope": [
                    {"time": 0.0, "value": 0.0},
                    {"time": 1.0, "value": 1200.0, "curve": "linear"},
                ],
            },
        )

        n = len(result)
        q = n // 4
        first_quarter = result[:q]
        last_quarter = result[-q:]

        c_first = _spectral_centroid(first_quarter, SR)
        c_last = _spectral_centroid(last_quarter, SR)
        assert c_last > c_first, (
            f"expected centroid to rise with upward bend, got {c_first} -> {c_last}"
        )


class TestRingMod:
    def test_ring_mod_alters_spectrum(self, sine_buffer: np.ndarray) -> None:
        clean = render_sample_segment(
            sine_buffer,
            freq=ROOT_FREQ,
            duration=DUR,
            amp=1.0,
            sample_rate=SR,
            params={"pitch_shift": False},
        )
        ringed = render_sample_segment(
            sine_buffer,
            freq=ROOT_FREQ,
            duration=DUR,
            amp=1.0,
            sample_rate=SR,
            params={"pitch_shift": False, "ring_freq_hz": 77.0, "ring_depth": 0.7},
        )

        n = len(ringed)
        freqs = np.fft.rfftfreq(n, d=1.0 / SR)

        def bin_mag(signal: np.ndarray, target_hz: float) -> float:
            mag = np.abs(np.fft.rfft(signal))
            idx = int(np.argmin(np.abs(freqs - target_hz)))
            # Take a small neighborhood to be robust to bin leakage.
            lo = max(0, idx - 2)
            hi = min(len(mag), idx + 3)
            return float(mag[lo:hi].max())

        # Sideband at 440 + 77 = 517 Hz should have non-trivial energy relative
        # to the clean baseline at the same bin.
        sideband_clean = bin_mag(clean, ROOT_FREQ + 77.0)
        sideband_ringed = bin_mag(ringed, ROOT_FREQ + 77.0)
        carrier_ringed = bin_mag(ringed, ROOT_FREQ)
        assert sideband_ringed > 5.0 * sideband_clean + 1.0
        # Sideband should be a meaningful fraction of the carrier.
        assert sideband_ringed > 0.05 * carrier_ringed


class TestRateReduce:
    def test_rate_reduce_creates_plateaus(self, sine_buffer: np.ndarray) -> None:
        step_size = 8
        result = render_sample_segment(
            sine_buffer,
            freq=ROOT_FREQ,
            duration=DUR,
            amp=1.0,
            sample_rate=SR,
            params={"pitch_shift": False, "rate_reduce_ratio": float(step_size)},
        )
        # Phase-independent invariant: over a long stretch the number of
        # distinct sample values is bounded by roughly len(window) / step_size
        # (one unique value per plateau). Allow slack for fractional boundary
        # effects at either end of the window.
        window = result[1000:2600]
        unique_values = np.unique(np.round(window, 10))
        max_expected_unique = len(window) // step_size + 10
        assert len(unique_values) <= max_expected_unique, (
            f"expected <= {max_expected_unique} unique values for "
            f"step_size={step_size}, got {len(unique_values)}"
        )


class TestBitDepth:
    def test_bit_depth_quantizes(self, sine_buffer: np.ndarray) -> None:
        result = render_sample_segment(
            sine_buffer,
            freq=ROOT_FREQ,
            duration=DUR,
            amp=1.0,
            sample_rate=SR,
            params={"pitch_shift": False, "bit_depth": 2.0},
        )
        # 2-bit quantization => ~4 levels (2^2). Post peak-normalize + amp scaling
        # preserves the level structure; allow for a couple extra unique values
        # from the normalization scaling edge.
        unique_values = np.unique(np.round(result, 6))
        assert len(unique_values) <= 6, (
            f"expected <= 6 unique values for 2-bit depth, got {len(unique_values)}: "
            f"{unique_values}"
        )


class TestStartJitter:
    def test_start_jitter_is_deterministic(self, sine_buffer: np.ndarray) -> None:
        params = {
            "pitch_shift": False,
            "start_jitter_ms": 5.0,
        }
        a = render_sample_segment(
            sine_buffer,
            freq=ROOT_FREQ,
            duration=0.1,
            amp=1.0,
            sample_rate=SR,
            params=params,
            note_seed=1234,
        )
        b = render_sample_segment(
            sine_buffer,
            freq=ROOT_FREQ,
            duration=0.1,
            amp=1.0,
            sample_rate=SR,
            params=params,
            note_seed=1234,
        )
        c = render_sample_segment(
            sine_buffer,
            freq=ROOT_FREQ,
            duration=0.1,
            amp=1.0,
            sample_rate=SR,
            params=params,
            note_seed=5678,
        )
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)


class TestSmoke:
    def test_render_sample_segment_importable_and_clean_output(
        self, sine_buffer: np.ndarray
    ) -> None:
        result = render_sample_segment(
            sine_buffer,
            freq=ROOT_FREQ,
            duration=0.25,
            amp=0.8,
            sample_rate=SR,
            params={
                "pitch_shift": False,
                "retrigger_count": 2,
                "retrigger_interval_ms": 30.0,
                "bend_envelope": [
                    {"time": 0.0, "value": 0.0},
                    {"time": 1.0, "value": 100.0, "curve": "linear"},
                ],
                "ring_freq_hz": 50.0,
                "ring_depth": 0.3,
                "rate_reduce_ratio": 2.0,
                "bit_depth": 8.0,
            },
        )
        assert result.dtype == np.float64
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) > 0.0
