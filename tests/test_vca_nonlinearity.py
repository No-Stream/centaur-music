"""Tests for VCA nonlinearity (envelope-coupled saturation)."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics import synth

SR = 44100


def _test_tone(dur: float = 1.0) -> np.ndarray:
    """1 second tone -- long enough that ADSR sustain is well populated."""
    t = np.arange(int(SR * dur)) / SR
    return np.sin(2 * np.pi * 440 * t)


class TestVcaNonlinearity:
    def test_zero_is_linear(self) -> None:
        """vca_nonlinearity=0 should give pure linear VCA."""
        sig = _test_tone()
        linear = synth.adsr(
            sig,
            attack=0.01,
            decay=0.1,
            sustain_level=0.7,
            release=0.1,
            sample_rate=SR,
            vca_nonlinearity=0.0,
        )
        default = synth.adsr(
            sig,
            attack=0.01,
            decay=0.1,
            sustain_level=0.7,
            release=0.1,
            sample_rate=SR,
        )
        np.testing.assert_allclose(linear, default, atol=1e-12)

    @pytest.mark.parametrize("amount", [0.1, 0.3, 0.5, 0.8, 1.0])
    def test_produces_finite_output(self, amount: float) -> None:
        sig = _test_tone()
        result = synth.adsr(sig, vca_nonlinearity=amount, sample_rate=SR)
        assert np.all(np.isfinite(result))

    def test_nonlinearity_adds_harmonics(self) -> None:
        """VCA saturation should add harmonic content."""
        sig = _test_tone()
        linear = synth.adsr(
            sig, attack=0.01, sustain_level=1.0, sample_rate=SR, vca_nonlinearity=0.0
        )
        saturated = synth.adsr(
            sig, attack=0.01, sustain_level=1.0, sample_rate=SR, vca_nonlinearity=0.7
        )
        lin_spec = np.abs(np.fft.rfft(linear))
        sat_spec = np.abs(np.fft.rfft(saturated))
        # Sum energy above fundamental (skip first 5 bins)
        assert np.sum(sat_spec[5:]) > np.sum(lin_spec[5:]) * 1.1

    def test_peak_amplitude_preserved(self) -> None:
        """Level compensation should keep peak amplitude similar."""
        sig = _test_tone()
        linear = synth.adsr(
            sig, attack=0.01, sustain_level=1.0, sample_rate=SR, vca_nonlinearity=0.0
        )
        saturated = synth.adsr(
            sig, attack=0.01, sustain_level=1.0, sample_rate=SR, vca_nonlinearity=0.5
        )
        lin_peak = float(np.max(np.abs(linear)))
        sat_peak = float(np.max(np.abs(saturated)))
        # Peaks should be within ~2dB
        assert abs(20 * np.log10(sat_peak / lin_peak)) < 2.0
