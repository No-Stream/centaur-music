"""Tests for the PolyBLEP synthesis engine."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines.polyblep import render
from code_musics.engines.registry import render_note_signal


class TestPolyBLEPSmoke:
    @pytest.mark.parametrize("waveform", ["saw", "square", "sine", "triangle"])
    def test_smoke_waveform(self, waveform: str) -> None:
        signal = render(
            freq=110.0,
            duration=0.25,
            amp=0.8,
            sample_rate=44100,
            params={"waveform": waveform},
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
    def test_second_oscillator_materially_changes_the_sound(self) -> None:
        base = render(
            freq=110.0,
            duration=0.4,
            amp=0.8,
            sample_rate=44100,
            params={"waveform": "saw", "cutoff_hz": 1_200.0},
        )
        layered = render(
            freq=110.0,
            duration=0.4,
            amp=0.8,
            sample_rate=44100,
            params={
                "waveform": "saw",
                "cutoff_hz": 1_200.0,
                "osc2_level": 0.8,
                "osc2_waveform": "square",
                "osc2_detune_cents": 7.0,
            },
        )

        assert np.isfinite(layered).all()
        assert not np.allclose(base, layered)
        assert np.linalg.norm(base - layered) > 1.0

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
        # Measure energy between the two cutoffs (600–3000 Hz), where the filter
        # difference is meaningful.  Bin resolution = sr / n_samples = 2 Hz/bin.
        n_samples = int(dur * sr)
        hz_per_bin = sr / n_samples
        bin_600 = int(600 / hz_per_bin)
        bin_3000 = int(3000 / hz_per_bin)
        low_mid_energy = np.sum(spectrum_low[bin_600:bin_3000] ** 2)
        high_mid_energy = np.sum(spectrum_high[bin_600:bin_3000] ** 2)
        assert high_mid_energy > low_mid_energy

    def test_filter_modes_materially_change_the_sound(self) -> None:
        base_params = {
            "waveform": "saw",
            "cutoff_hz": 1_200.0,
            "resonance_q": 2.74,
            "filter_env_amount": 0.3,
        }
        lowpass = render(
            freq=220.0,
            duration=0.4,
            amp=0.8,
            sample_rate=44100,
            params={**base_params, "filter_mode": "lowpass"},
        )
        bandpass = render(
            freq=220.0,
            duration=0.4,
            amp=0.8,
            sample_rate=44100,
            params={**base_params, "filter_mode": "bandpass"},
        )
        notch = render(
            freq=220.0,
            duration=0.4,
            amp=0.8,
            sample_rate=44100,
            params={**base_params, "filter_mode": "notch"},
        )

        assert not np.allclose(lowpass, bandpass)
        assert not np.allclose(lowpass, notch)
        assert np.linalg.norm(lowpass - bandpass) > 1.0
        assert np.linalg.norm(lowpass - notch) > 1.0

    def test_filter_drive_changes_the_sound_without_instability(self) -> None:
        clean = render(
            freq=110.0,
            duration=0.4,
            amp=0.8,
            sample_rate=44100,
            params={
                "waveform": "saw",
                "cutoff_hz": 900.0,
                "resonance_q": 3.53,
                "filter_drive": 0.0,
            },
        )
        driven = render(
            freq=110.0,
            duration=0.4,
            amp=0.8,
            sample_rate=44100,
            params={
                "waveform": "saw",
                "cutoff_hz": 900.0,
                "resonance_q": 3.53,
                "filter_drive": 0.8,
            },
        )

        assert np.isfinite(driven).all()
        assert np.max(np.abs(driven)) > 0.0
        assert not np.allclose(clean, driven)
        assert np.linalg.norm(clean - driven) > 1.0

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


class TestPolyBLEPStability:
    def test_very_high_resonance_low_cutoff_remains_finite(self) -> None:
        signal = render(
            freq=55.0,
            duration=0.75,
            amp=0.8,
            sample_rate=44100,
            params={
                "waveform": "square",
                "cutoff_hz": 140.0,
                "resonance_q": 23.29,
                "filter_drive": 0.6,
                "filter_mode": "lowpass",
                "filter_env_amount": 0.2,
                "filter_env_decay": 0.3,
            },
        )
        assert np.isfinite(signal).all()
        assert np.max(np.abs(signal)) > 0.0
        assert np.max(np.abs(signal)) < 2.0


class TestPolyBLEPDeterminism:
    def test_deterministic(self) -> None:
        kwargs = dict(
            freq=330.0,
            duration=0.2,
            amp=0.6,
            sample_rate=44100,
            params={
                "waveform": "saw",
                "cutoff_hz": 2000.0,
                "resonance_q": 1.84,
                "filter_drive": 0.2,
                "filter_mode": "lowpass",
            },
        )
        a = render(**kwargs)  # type: ignore[arg-type]
        b = render(**kwargs)  # type: ignore[arg-type]
        assert np.allclose(a, b)

    def test_second_oscillator_rejects_negative_level(self) -> None:
        with np.testing.assert_raises(ValueError):
            render(
                freq=220.0,
                duration=0.2,
                amp=0.6,
                sample_rate=44100,
                params={"osc2_level": -0.1},
            )


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
