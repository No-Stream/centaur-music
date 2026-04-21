"""Tests for native stereo width (mid/side) effect."""

from __future__ import annotations

import numpy as np

from code_musics.score import EffectSpec
from code_musics.synth import SAMPLE_RATE, apply_effect_chain, apply_stereo_width


def _stereo_signal(duration_s: float = 0.1) -> np.ndarray:
    """A short stereo signal with distinct L and R content."""
    n_samples = int(SAMPLE_RATE * duration_s)
    t = np.linspace(0.0, duration_s, n_samples, endpoint=False)
    left = np.sin(2.0 * np.pi * 440.0 * t)
    right = np.sin(2.0 * np.pi * 660.0 * t)
    return np.array([left, right])


class TestStereoWidth:
    def test_stereo_width_unity(self) -> None:
        """width=1.0 returns input unchanged."""
        signal = _stereo_signal()
        result = apply_stereo_width(signal, width=1.0)
        np.testing.assert_allclose(result, signal, atol=1e-12)

    def test_stereo_width_mono(self) -> None:
        """width=0.0 collapses to mono (identical L and R)."""
        signal = _stereo_signal()
        result = apply_stereo_width(signal, width=0.0)
        np.testing.assert_allclose(result[0], result[1], atol=1e-12)

    def test_stereo_width_wider(self) -> None:
        """width=2.0 increases the L-R difference relative to the input."""
        signal = _stereo_signal()
        input_side_energy = float(np.sum((signal[0] - signal[1]) ** 2))

        result = apply_stereo_width(signal, width=2.0)
        output_side_energy = float(np.sum((result[0] - result[1]) ** 2))

        assert output_side_energy > input_side_energy, (
            f"width=2.0 should widen: input side energy={input_side_energy:.6f}, "
            f"output={output_side_energy:.6f}"
        )

    def test_stereo_width_mono_passthrough(self) -> None:
        """Mono (1D) input passes through unchanged regardless of width."""
        mono_signal = np.sin(
            np.linspace(0.0, 4.0 * np.pi, SAMPLE_RATE // 10, endpoint=False)
        )
        result = apply_stereo_width(mono_signal, width=0.5)
        np.testing.assert_array_equal(result, mono_signal)

    def test_stereo_width_in_effect_chain(self) -> None:
        """EffectSpec(kind='stereo_width') works through apply_effect_chain."""
        signal = _stereo_signal(duration_s=0.05)
        result = apply_effect_chain(
            signal,
            [EffectSpec("stereo_width", {"width": 0.0})],
        )
        # Should collapse to mono
        np.testing.assert_allclose(result[0], result[1], atol=1e-12)
