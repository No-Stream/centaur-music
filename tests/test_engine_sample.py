"""Tests for the sample playback engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from code_musics.engines.sample import _load_sample, render


@pytest.fixture()
def sample_wav(tmp_path: Path) -> Path:
    """Create a short synthetic WAV for testing."""
    sr = 44100
    dur = 0.1
    t = np.arange(int(sr * dur), dtype=np.float64) / sr
    signal = np.sin(2.0 * np.pi * 440.0 * t)
    path = tmp_path / "test_tone.wav"
    sf.write(str(path), signal, sr, subtype="FLOAT")
    return path


class TestSampleEngineBasic:
    def test_render_basic(self, sample_wav: Path) -> None:
        result = render(
            freq=440.0,
            duration=0.1,
            amp=0.8,
            sample_rate=44100,
            params={"sample_path": str(sample_wav), "pitch_shift": False},
        )
        assert result.dtype == np.float64
        assert len(result) == int(44100 * 0.1)
        assert np.max(np.abs(result)) > 0

    def test_render_zero_duration(self, sample_wav: Path) -> None:
        with pytest.raises(ValueError, match="duration must be positive"):
            render(
                freq=440.0,
                duration=0.0,
                amp=0.8,
                sample_rate=44100,
                params={"sample_path": str(sample_wav)},
            )

    def test_missing_sample_path(self) -> None:
        with pytest.raises(ValueError, match="sample_path"):
            render(
                freq=440.0,
                duration=0.1,
                amp=0.8,
                sample_rate=44100,
                params={},
            )

    def test_invalid_filter_mode(self, sample_wav: Path) -> None:
        with pytest.raises(ValueError, match="filter_mode"):
            render(
                freq=440.0,
                duration=0.1,
                amp=0.8,
                sample_rate=44100,
                params={
                    "sample_path": str(sample_wav),
                    "filter_mode": "notch",
                    "pitch_shift": False,
                },
            )

    def test_freq_trajectory_rejected(self, sample_wav: Path) -> None:
        with pytest.raises(ValueError, match="freq_trajectory"):
            render(
                freq=440.0,
                duration=0.1,
                amp=0.8,
                sample_rate=44100,
                params={"sample_path": str(sample_wav)},
                freq_trajectory=np.ones(4410),
            )


class TestSampleEngineFeatures:
    def test_pitch_shift(self, sample_wav: Path) -> None:
        base = render(
            freq=440.0,
            duration=0.1,
            amp=0.8,
            sample_rate=44100,
            params={"sample_path": str(sample_wav), "root_freq": 440.0},
        )
        shifted = render(
            freq=880.0,
            duration=0.1,
            amp=0.8,
            sample_rate=44100,
            params={"sample_path": str(sample_wav), "root_freq": 440.0},
        )
        assert not np.allclose(base, shifted)

    def test_reverse(self, sample_wav: Path) -> None:
        fwd = render(
            freq=440.0,
            duration=0.1,
            amp=0.8,
            sample_rate=44100,
            params={"sample_path": str(sample_wav), "pitch_shift": False},
        )
        rev = render(
            freq=440.0,
            duration=0.1,
            amp=0.8,
            sample_rate=44100,
            params={
                "sample_path": str(sample_wav),
                "pitch_shift": False,
                "reverse": True,
            },
        )
        assert not np.allclose(fwd, rev)

    def test_decay_envelope(self, sample_wav: Path) -> None:
        no_decay = render(
            freq=440.0,
            duration=0.1,
            amp=0.8,
            sample_rate=44100,
            params={"sample_path": str(sample_wav), "pitch_shift": False},
        )
        with_decay = render(
            freq=440.0,
            duration=0.1,
            amp=0.8,
            sample_rate=44100,
            params={
                "sample_path": str(sample_wav),
                "pitch_shift": False,
                "decay_ms": 20.0,
            },
        )
        assert not np.allclose(no_decay, with_decay)
        # Decayed signal should be quieter in the tail
        tail_start = len(with_decay) * 3 // 4
        assert np.mean(np.abs(with_decay[tail_start:])) < np.mean(
            np.abs(no_decay[tail_start:])
        )

    def test_filter(self, sample_wav: Path) -> None:
        unfiltered = render(
            freq=440.0,
            duration=0.1,
            amp=0.8,
            sample_rate=44100,
            params={"sample_path": str(sample_wav), "pitch_shift": False},
        )
        filtered = render(
            freq=440.0,
            duration=0.1,
            amp=0.8,
            sample_rate=44100,
            params={
                "sample_path": str(sample_wav),
                "pitch_shift": False,
                "filter_mode": "lowpass",
                "filter_cutoff_hz": 500.0,
            },
        )
        assert not np.allclose(unfiltered, filtered)

    def test_start_offset(self, sample_wav: Path) -> None:
        no_offset = render(
            freq=440.0,
            duration=0.05,
            amp=0.8,
            sample_rate=44100,
            params={"sample_path": str(sample_wav), "pitch_shift": False},
        )
        with_offset = render(
            freq=440.0,
            duration=0.05,
            amp=0.8,
            sample_rate=44100,
            params={
                "sample_path": str(sample_wav),
                "pitch_shift": False,
                "start_offset_ms": 30.0,
            },
        )
        assert not np.allclose(no_offset, with_offset)

    def test_amp_scales_output(self, sample_wav: Path) -> None:
        loud = render(
            freq=440.0,
            duration=0.1,
            amp=1.0,
            sample_rate=44100,
            params={"sample_path": str(sample_wav), "pitch_shift": False},
        )
        quiet = render(
            freq=440.0,
            duration=0.1,
            amp=0.5,
            sample_rate=44100,
            params={"sample_path": str(sample_wav), "pitch_shift": False},
        )
        assert np.max(np.abs(loud)) == pytest.approx(1.0, abs=0.01)
        assert np.max(np.abs(quiet)) == pytest.approx(0.5, abs=0.01)


class TestSampleCache:
    def test_cache_returns_same_data(self, sample_wav: Path) -> None:
        _load_sample.cache_clear()
        a = _load_sample(str(sample_wav), 44100)
        b = _load_sample(str(sample_wav), 44100)
        assert a is b  # exact same object from cache

    def test_stereo_mixdown(self, tmp_path: Path) -> None:
        sr = 44100
        n = 4410
        stereo = np.column_stack(
            [np.ones(n, dtype=np.float64), -np.ones(n, dtype=np.float64)]
        )
        path = tmp_path / "stereo.wav"
        sf.write(str(path), stereo, sr, subtype="FLOAT")
        _load_sample.cache_clear()
        mono = _load_sample(str(path), sr)
        assert mono.ndim == 1
        # Mean of [1, -1] = 0
        assert np.allclose(mono, 0.0)
