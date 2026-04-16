"""Tests for the serial highpass filter stage in apply_filter."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._filters import apply_filter

SR = 44100


def _noise(dur: float = 0.5) -> np.ndarray:
    return np.random.default_rng(42).standard_normal(int(SR * dur))


def _band_energy(signal: np.ndarray, low_hz: float, high_hz: float) -> float:
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1 / SR)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    return float(np.sqrt(np.mean(spectrum[mask] ** 2)))


class TestSerialHPF:
    def test_zero_hpf_is_passthrough(self) -> None:
        """hpf_cutoff_hz=0 should not affect the output."""
        sig = _noise()
        cutoff = np.full(len(sig), 2000.0)
        no_hpf = apply_filter(sig, cutoff_profile=cutoff, sample_rate=SR)
        with_zero = apply_filter(
            sig, cutoff_profile=cutoff, sample_rate=SR, hpf_cutoff_hz=0.0
        )
        np.testing.assert_allclose(no_hpf, with_zero)

    def test_hpf_reduces_bass(self) -> None:
        """HPF should attenuate low frequencies."""
        sig = _noise()
        cutoff = np.full(len(sig), 8000.0)
        no_hpf = apply_filter(sig, cutoff_profile=cutoff, sample_rate=SR)
        with_hpf = apply_filter(
            sig, cutoff_profile=cutoff, sample_rate=SR, hpf_cutoff_hz=500.0
        )
        bass_no_hpf = _band_energy(no_hpf, 20, 200)
        bass_with_hpf = _band_energy(with_hpf, 20, 200)
        assert bass_with_hpf < bass_no_hpf * 0.5

    @pytest.mark.parametrize("topology", ["svf", "ladder"])
    def test_hpf_works_both_topologies(self, topology: str) -> None:
        sig = _noise()
        cutoff = np.full(len(sig), 2000.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            hpf_cutoff_hz=300.0,
        )
        assert np.all(np.isfinite(result))
