"""Tests for external feedback path on the filter."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._filters import apply_filter

SR = 44100


def _sine(freq: float = 200, dur: float = 0.5) -> np.ndarray:
    t = np.arange(int(SR * dur)) / SR
    return np.sin(2 * np.pi * freq * t)


class TestFeedbackPath:
    def test_zero_feedback_is_baseline(self) -> None:
        """feedback_amount=0 should match no-feedback output."""
        sig = _sine()
        cutoff = np.full(len(sig), 2000.0)
        baseline = apply_filter(
            sig, cutoff_profile=cutoff, sample_rate=SR, feedback_amount=0.0
        )
        explicit = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            feedback_amount=0.0,
            feedback_saturation=0.5,
        )
        np.testing.assert_allclose(baseline, explicit)

    @pytest.mark.parametrize("amount", [0.1, 0.3, 0.5, 0.8])
    def test_feedback_produces_finite_output(self, amount: float) -> None:
        sig = _sine()
        cutoff = np.full(len(sig), 2000.0)
        result = apply_filter(
            sig, cutoff_profile=cutoff, sample_rate=SR, feedback_amount=amount
        )
        assert np.all(np.isfinite(result))

    def test_feedback_adds_harmonics(self) -> None:
        """Feedback should add harmonic content to a sine input."""
        sig = _sine()
        cutoff = np.full(len(sig), 2000.0)
        clean = apply_filter(
            sig, cutoff_profile=cutoff, sample_rate=SR, feedback_amount=0.0
        )
        feedback = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            feedback_amount=0.5,
            feedback_saturation=0.5,
        )
        clean_hf = float(np.sum(np.abs(np.fft.rfft(clean))[5:]))
        feedback_hf = float(np.sum(np.abs(np.fft.rfft(feedback))[5:]))
        assert feedback_hf > clean_hf * 1.2

    @pytest.mark.parametrize(
        "topology",
        ["svf", "ladder", "sallen_key", "cascade", "sem", "jupiter", "k35", "diode"],
    )
    def test_feedback_works_all_topologies(self, topology: str) -> None:
        sig = _sine()
        cutoff = np.full(len(sig), 2000.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            feedback_amount=0.3,
            feedback_saturation=0.4,
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) > 0.01
