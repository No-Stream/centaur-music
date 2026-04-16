"""Tests for the Moog-style 4-pole ladder filter implementation."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._filters import apply_filter, apply_zdf_svf

SR = 44100


def _test_signal(dur: float = 0.5) -> np.ndarray:
    """White noise test signal."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(int(SR * dur))


def _band_energy(
    signal: np.ndarray, low_hz: float, high_hz: float, sr: int = SR
) -> float:
    """RMS energy in a frequency band."""
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1 / sr)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    return float(np.sqrt(np.mean(spectrum[mask] ** 2)))


class TestLadderTopology:
    def test_svf_default_unchanged(self) -> None:
        """Default topology='svf' matches direct apply_zdf_svf call."""
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        via_dispatch = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_mode="lowpass",
            filter_drive=0.0,
        )
        via_direct = apply_zdf_svf(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_mode="lowpass",
            filter_drive=0.0,
        )
        np.testing.assert_allclose(via_dispatch, via_direct)

    def test_ladder_produces_output(self) -> None:
        """Ladder topology produces non-zero, finite output."""
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_mode="lowpass",
            filter_topology="ladder",
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) > 0.01

    def test_ladder_steeper_than_svf(self) -> None:
        """Ladder (24dB/oct) should have more HF attenuation than SVF (12dB/oct)."""
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        svf = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_mode="lowpass",
            filter_topology="svf",
        )
        ladder = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_mode="lowpass",
            filter_topology="ladder",
        )
        svf_hf = _band_energy(svf, 4000, 20000)
        ladder_hf = _band_energy(ladder, 4000, 20000)
        assert ladder_hf < svf_hf * 0.5  # at least 6dB steeper

    def test_unknown_topology_raises(self) -> None:
        """Unknown filter_topology raises ValueError."""
        sig = _test_signal(0.1)
        cutoff = np.full(len(sig), 1000.0)
        with pytest.raises(ValueError, match="filter_topology"):
            apply_filter(
                sig,
                cutoff_profile=cutoff,
                sample_rate=SR,
                filter_topology="invalid",
            )


class TestBassCompensation:
    def test_compensation_restores_bass(self) -> None:
        """Bass compensation at 1.0 should restore low-frequency energy.

        Use very high resonance (Q=12) so the bass-scooping effect of ladder
        feedback is pronounced enough that compensation is clearly measurable.
        """
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        no_comp = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            resonance_q=12.0,
            bass_compensation=0.0,
        )
        with_comp = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            resonance_q=12.0,
            bass_compensation=1.0,
        )
        no_comp_bass = _band_energy(no_comp, 20, 200)
        with_comp_bass = _band_energy(with_comp, 20, 200)
        assert with_comp_bass > no_comp_bass * 1.01  # measurable restoration


class TestLadderDrive:
    @pytest.mark.parametrize("drive", [0.0, 0.2, 0.5, 1.0])
    def test_drive_produces_finite_output(self, drive: float) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_drive=drive,
        )
        assert np.all(np.isfinite(result))

    def test_drive_adds_harmonics(self) -> None:
        """Higher drive should increase harmonic content (THD)."""
        n = int(SR * 0.5)
        t = np.arange(n) / SR
        sig = np.sin(2 * np.pi * 200 * t)
        cutoff = np.full(n, 2000.0)
        clean = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_drive=0.0,
        )
        driven = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_drive=0.5,
        )
        clean_hf = _band_energy(clean, 400, 5000)
        driven_hf = _band_energy(driven, 400, 5000)
        assert driven_hf > clean_hf * 1.5


class TestLadderModes:
    @pytest.mark.parametrize("mode", ["lowpass", "bandpass", "highpass"])
    def test_mode_produces_output(self, mode: str) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_mode=mode,
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) > 0.001
