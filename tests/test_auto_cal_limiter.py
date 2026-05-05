"""Tests for piece-aware auto-calibration on ``apply_native_limiter``.

Mirrors the ``max_shave_db`` auto-cal tests for ``apply_clipper`` but targets
the limiter's headroom-below-peak surface (``headroom_db``).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from code_musics.synth import (
    SAMPLE_RATE,
    apply_native_limiter,
)


def _sine_wave(freq_hz: float, duration_s: float, amp: float = 1.0) -> np.ndarray:
    n = int(duration_s * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float64)


class TestLimiterAutoCalibration:
    """``headroom_db`` sets threshold from input peak statistic."""

    def test_threshold_set_to_peak_minus_headroom(self) -> None:
        """p99.9 peak near -3 dBFS with headroom 1 dB -> threshold ~ -4 dBFS."""
        amp = 10.0 ** (-3.0 / 20.0)
        signal = _sine_wave(440.0, 1.0, amp=amp)
        _, metrics = apply_native_limiter(
            signal,
            headroom_db=1.0,
            return_analysis=True,
        )
        threshold = float(metrics["calibrated_threshold_db"])
        assert abs(threshold - (-4.0)) < 0.1, (
            f"expected ~-4 dBFS; got {threshold:.3f} dBFS "
            f"(reference_peak={metrics['reference_peak_dbfs']}, "
            f"headroom={metrics['headroom_db']})"
        )
        assert "reference_peak_dbfs" in metrics
        assert "calibration_percentile" in metrics

    def test_headroom_none_is_backwards_compatible(self) -> None:
        """Default path (``headroom_db=None``) matches pre-change behavior bit-for-bit."""
        signal = _sine_wave(440.0, 0.5, amp=0.9)
        out_default = apply_native_limiter(signal, threshold_db=-0.5)
        out_none = apply_native_limiter(signal, threshold_db=-0.5, headroom_db=None)
        np.testing.assert_allclose(out_default, out_none, atol=0.0, rtol=0.0)
        _, metrics = apply_native_limiter(
            signal,
            threshold_db=-0.5,
            return_analysis=True,
        )
        assert "calibrated_threshold_db" not in metrics
        assert "reference_peak_dbfs" not in metrics
        assert "headroom_db" not in metrics

    def test_conflict_warning_when_both_set(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Non-default ``threshold_db`` + ``headroom_db`` logs a WARNING."""
        signal = _sine_wave(440.0, 0.5, amp=0.9)
        with caplog.at_level(logging.WARNING, logger="code_musics.synth"):
            apply_native_limiter(signal, threshold_db=-1.0, headroom_db=1.0)
        warning_texts = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("headroom_db wins" in msg for msg in warning_texts), (
            f"expected warning about headroom_db winning; got {warning_texts}"
        )

    def test_negative_headroom_raises(self) -> None:
        """``headroom_db`` must be non-negative when set."""
        signal = _sine_wave(440.0, 0.5, amp=0.9)
        with pytest.raises(ValueError, match="headroom_db must be non-negative"):
            apply_native_limiter(signal, headroom_db=-1.0)
