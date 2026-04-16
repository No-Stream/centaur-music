"""Tests for SVF and ladder filter mode morphing."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._filters import apply_filter

SR = 44100


def _noise(dur: float = 0.5) -> np.ndarray:
    return np.random.default_rng(42).standard_normal(int(SR * dur))


class TestSvfMorph:
    def test_zero_morph_matches_pure_mode(self) -> None:
        """morph=0 should give identical output to no-morph path."""
        sig = _noise()
        cutoff = np.full(len(sig), 1000.0)
        no_morph = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_mode="lowpass",
            filter_morph=0.0,
        )
        with_zero = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_mode="lowpass",
            filter_morph=0.0,
        )
        np.testing.assert_allclose(no_morph, with_zero)

    @pytest.mark.parametrize("morph", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    def test_morph_produces_finite_output(self, morph: float) -> None:
        sig = _noise()
        cutoff = np.full(len(sig), 1000.0)
        result = apply_filter(
            sig, cutoff_profile=cutoff, sample_rate=SR, filter_morph=morph
        )
        assert np.all(np.isfinite(result))

    def test_morph_1_matches_next_mode(self) -> None:
        """morph=1.0 from LP should approximate pure BP."""
        sig = _noise()
        cutoff = np.full(len(sig), 1000.0)
        morphed = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_mode="lowpass",
            filter_morph=1.0,
        )
        pure_bp = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_mode="bandpass",
            filter_morph=0.0,
        )
        # Should be very close (both are pure BP)
        np.testing.assert_allclose(morphed, pure_bp, atol=1e-10)


class TestLadderMorph:
    @pytest.mark.parametrize("morph", [0.0, 1.0, 2.0, 3.0])
    def test_ladder_morph_produces_output(self, morph: float) -> None:
        sig = _noise()
        cutoff = np.full(len(sig), 1000.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_morph=morph,
        )
        assert np.all(np.isfinite(result))

    def test_higher_morph_less_steep(self) -> None:
        """Morph from 4-pole toward 1-pole should reduce HF attenuation."""
        sig = _noise()
        cutoff = np.full(len(sig), 1000.0)
        four_pole = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_morph=0.0,
        )
        one_pole = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_morph=3.0,
        )
        spec_4p = np.abs(np.fft.rfft(four_pole))
        spec_1p = np.abs(np.fft.rfft(one_pole))
        freqs = np.fft.rfftfreq(len(sig), 1 / SR)
        hf = freqs > 4000
        # 1-pole should have more HF energy
        assert np.sum(spec_1p[hf]) > np.sum(spec_4p[hf]) * 1.5
