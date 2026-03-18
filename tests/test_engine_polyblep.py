"""Tests for the PolyBLEP synthesis engine."""

from __future__ import annotations

import numpy as np

from code_musics.engines.polyblep import render
from code_musics.engines.registry import render_note_signal


class TestPolyBLEPSmoke:
    def test_smoke_saw(self) -> None:
        signal = render(
            freq=110.0,
            duration=0.25,
            amp=0.8,
            sample_rate=44100,
            params={"waveform": "saw"},
        )
        assert len(signal) == int(0.25 * 44100)
        assert np.all(np.isfinite(signal))
        assert np.max(np.abs(signal)) > 0

    def test_smoke_square(self) -> None:
        signal = render(
            freq=110.0,
            duration=0.25,
            amp=0.8,
            sample_rate=44100,
            params={"waveform": "square"},
        )
        assert len(signal) == int(0.25 * 44100)
        assert np.all(np.isfinite(signal))
        assert np.max(np.abs(signal)) > 0

    def test_length_matches_sample_math(self) -> None:
        signal = render(
            freq=220.0,
            duration=0.3,
            amp=0.5,
            sample_rate=32000,
            params={},
        )
        assert len(signal) == int(0.3 * 32000)


class TestPolyBLEPSpectral:
    def test_cutoff_affects_spectrum(self) -> None:
        dur = 0.5
        sr = 44100
        low_cut = render(
            freq=220.0,
            duration=dur,
            amp=0.8,
            sample_rate=sr,
            params={"cutoff_hz": 400.0},
        )
        high_cut = render(
            freq=220.0,
            duration=dur,
            amp=0.8,
            sample_rate=sr,
            params={"cutoff_hz": 4000.0},
        )
        assert not np.allclose(low_cut, high_cut)

        spectrum_low = np.abs(np.fft.rfft(low_cut))
        spectrum_high = np.abs(np.fft.rfft(high_cut))
        mid = len(spectrum_low) // 2
        low_upper_energy = np.sum(spectrum_low[mid:] ** 2)
        high_upper_energy = np.sum(spectrum_high[mid:] ** 2)
        assert high_upper_energy > low_upper_energy

    def test_pulse_width_affects_square(self) -> None:
        dur = 0.4
        sr = 44100
        pw_30 = render(
            freq=220.0,
            duration=dur,
            amp=0.8,
            sample_rate=sr,
            params={"waveform": "square", "pulse_width": 0.3},
        )
        pw_50 = render(
            freq=220.0,
            duration=dur,
            amp=0.8,
            sample_rate=sr,
            params={"waveform": "square", "pulse_width": 0.5},
        )
        assert not np.allclose(pw_30, pw_50)


class TestPolyBLEPFreqTrajectory:
    def test_freq_trajectory_constant_is_valid(self) -> None:
        sr = 44100
        dur = 0.3
        n = int(dur * sr)
        constant_traj = np.full(n, 220.0)
        signal = render(
            freq=220.0,
            duration=dur,
            amp=0.7,
            sample_rate=sr,
            params={},
            freq_trajectory=constant_traj,
        )
        assert len(signal) == n
        assert np.all(np.isfinite(signal))
        assert np.max(np.abs(signal)) > 0

        sweep_traj = np.linspace(220.0, 440.0, n)
        swept = render(
            freq=220.0,
            duration=dur,
            amp=0.7,
            sample_rate=sr,
            params={},
            freq_trajectory=sweep_traj,
        )
        assert len(swept) == n
        assert np.all(np.isfinite(swept))
        assert np.max(np.abs(swept)) > 0


class TestPolyBLEPDeterminism:
    def test_deterministic(self) -> None:
        kwargs = dict(
            freq=330.0,
            duration=0.2,
            amp=0.6,
            sample_rate=44100,
            params={"waveform": "saw", "cutoff_hz": 2000.0, "resonance": 0.1},
        )
        a = render(**kwargs)  # type: ignore[arg-type]
        b = render(**kwargs)  # type: ignore[arg-type]
        assert np.allclose(a, b)


class TestPolyBLEPRegistry:
    def test_warm_lead_preset_via_registry(self) -> None:
        signal = render_note_signal(
            freq=220.0,
            duration=0.3,
            amp=0.6,
            sample_rate=44100,
            params={"engine": "polyblep", "preset": "warm_lead"},
        )
        assert np.all(np.isfinite(signal))
        assert np.max(np.abs(signal)) > 0
