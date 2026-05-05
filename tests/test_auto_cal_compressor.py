"""Tests for piece-aware auto-calibration on ``apply_compressor``.

Mirrors the ``max_shave_db`` auto-cal tests for ``apply_clipper`` but targets
the compressor's active-region average gain-reduction surface
(``target_avg_gr_db``).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from code_musics.synth import (
    SAMPLE_RATE,
    apply_compressor,
)


def _sine_wave(freq_hz: float, duration_s: float, amp: float = 1.0) -> np.ndarray:
    n = int(duration_s * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float64)


def _kick_burst_signal(
    duration_s: float,
    *,
    burst_amp: float,
    burst_len_s: float = 0.05,
    stride_s: float = 0.5,
) -> np.ndarray:
    """Build a sparse signal: short loud bursts separated by silence."""
    n = int(duration_s * SAMPLE_RATE)
    signal = np.zeros(n, dtype=np.float64)
    burst_len = int(burst_len_s * SAMPLE_RATE)
    stride = int(stride_s * SAMPLE_RATE)
    n_bursts = n // stride
    t_burst = np.arange(burst_len) / SAMPLE_RATE
    shape = burst_amp * np.sin(2.0 * np.pi * 80.0 * t_burst)
    for i in range(n_bursts):
        start = i * stride
        signal[start : start + burst_len] = shape
    return signal


def _half_on_half_off_burst(duration_s: float, amp: float = 0.7) -> np.ndarray:
    """Signal that is a sine tone for half its duration, silent for the other half.

    Used to verify the solver targets avg GR over the *active* half, not the
    (noise-floor-dragged) piece mean.
    """
    n = int(duration_s * SAMPLE_RATE)
    signal = np.zeros(n, dtype=np.float64)
    active_n = n // 2
    t = np.arange(active_n) / SAMPLE_RATE
    signal[:active_n] = amp * np.sin(2.0 * np.pi * 220.0 * t)
    return signal


class TestCompressorAutoCalibration:
    """``target_avg_gr_db`` binary-searches threshold for active-region avg GR."""

    def test_target_avg_gr_tracks_on_half_active_signal(self) -> None:
        """Solver lands within 0.75 dB of target on a 50%-active signal.

        The active half is a steady 220 Hz sine tone; the silent half should
        be excluded from the avg-GR metric by the 40 dB active-region gate.
        """
        signal = _half_on_half_off_burst(duration_s=4.0, amp=0.7)
        result = apply_compressor(
            signal,
            target_avg_gr_db=4.0,
            ratio=4.0,
            attack_ms=5.0,
            release_ms=120.0,
            knee_db=6.0,
            return_analysis=True,
        )
        assert isinstance(result, tuple)
        _, metrics = result
        measured_avg_gr = float(metrics["measured_avg_gr_db"])
        assert abs(measured_avg_gr - 4.0) < 0.75, (
            f"solver did not converge: target=4.0, measured={measured_avg_gr:.2f} dB"
        )

    def test_target_avg_gr_tracks_on_sparse_kick_pattern(self) -> None:
        """Sparse kick pattern: active-region GR lands within 0.75 dB of target.

        On sparse bursty content the active-region gate isolates the hits, so
        the solver should still land close to target even though the piece
        mean GR is dominated by silent samples.
        """
        signal = _kick_burst_signal(
            duration_s=4.0,
            burst_amp=0.7,
            burst_len_s=0.2,
            stride_s=0.5,
        )
        result = apply_compressor(
            signal,
            target_avg_gr_db=6.0,
            ratio=4.0,
            attack_ms=10.0,
            release_ms=80.0,
            knee_db=6.0,
            return_analysis=True,
        )
        assert isinstance(result, tuple)
        _, metrics = result
        measured_avg_gr = float(metrics["measured_avg_gr_db"])
        assert abs(measured_avg_gr - 6.0) < 0.75, (
            f"sparse-content solver did not converge: target=6.0, "
            f"measured={measured_avg_gr:.2f} dB"
        )

    def test_target_avg_gr_none_is_backwards_compatible(self) -> None:
        """Default path matches pre-change behavior bit-for-bit."""
        signal = 1.1 * _sine_wave(220.0, 1.0, amp=1.0)
        out_default = apply_compressor(
            signal,
            threshold_db=-18.0,
            ratio=4.0,
            attack_ms=0.5,
            release_ms=120.0,
            knee_db=6.0,
        )
        out_none = apply_compressor(
            signal,
            threshold_db=-18.0,
            ratio=4.0,
            attack_ms=0.5,
            release_ms=120.0,
            knee_db=6.0,
            target_avg_gr_db=None,
        )
        assert isinstance(out_default, np.ndarray)
        assert isinstance(out_none, np.ndarray)
        np.testing.assert_array_equal(out_default, out_none)
        result = apply_compressor(
            signal,
            threshold_db=-18.0,
            ratio=4.0,
            attack_ms=0.5,
            release_ms=120.0,
            knee_db=6.0,
            return_analysis=True,
        )
        assert isinstance(result, tuple)
        _, metrics = result
        assert "calibrated_threshold_db" not in metrics
        assert "target_avg_gr_db" not in metrics
        assert "measured_avg_gr_db" not in metrics
        assert "solver_iterations" not in metrics

    def test_conflict_warning_when_both_set(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Non-default ``threshold_db`` + ``target_avg_gr_db`` logs a WARNING."""
        signal = _sine_wave(220.0, 1.0, amp=0.5)
        with caplog.at_level(logging.WARNING, logger="code_musics.synth"):
            apply_compressor(
                signal,
                threshold_db=-10.0,
                target_avg_gr_db=4.0,
                ratio=4.0,
                attack_ms=5.0,
                release_ms=120.0,
                knee_db=6.0,
            )
        warning_texts = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("target_avg_gr_db wins" in msg for msg in warning_texts), (
            f"expected warning about target_avg_gr_db winning; got {warning_texts}"
        )

    def test_analysis_dict_contains_calibration_fields(self) -> None:
        """``return_analysis=True`` surfaces calibration metadata."""
        signal = _sine_wave(220.0, 2.0, amp=0.5)
        result = apply_compressor(
            signal,
            target_avg_gr_db=3.0,
            ratio=4.0,
            attack_ms=5.0,
            release_ms=120.0,
            knee_db=6.0,
            return_analysis=True,
        )
        assert isinstance(result, tuple)
        _, metrics = result
        assert "calibrated_threshold_db" in metrics
        assert "target_avg_gr_db" in metrics
        assert "measured_avg_gr_db" in metrics
        assert "solver_iterations" in metrics
        assert float(metrics["target_avg_gr_db"]) == 3.0
        assert int(metrics["solver_iterations"]) >= 1

    def test_silent_input_warns_and_falls_back(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Pure-zeros input logs a WARNING and falls back to static threshold."""
        signal = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float64)
        with caplog.at_level(logging.WARNING, logger="code_musics.synth"):
            result = apply_compressor(
                signal,
                target_avg_gr_db=4.0,
                ratio=4.0,
                attack_ms=5.0,
                release_ms=120.0,
                knee_db=6.0,
                return_analysis=True,
            )
        assert isinstance(result, tuple)
        out, _metrics = result
        assert np.all(np.isfinite(out))
        warning_texts = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("no active content" in msg for msg in warning_texts), (
            f"expected silent-input fallback warning; got {warning_texts}"
        )

    def test_quiet_input_does_not_crash_or_nan(self) -> None:
        """Very quiet input (~-60 dBFS env) still produces finite output."""
        signal = _sine_wave(220.0, 1.0, amp=10.0 ** (-60.0 / 20.0))
        result = apply_compressor(
            signal,
            target_avg_gr_db=4.0,
            ratio=4.0,
            attack_ms=5.0,
            release_ms=120.0,
            knee_db=6.0,
            return_analysis=True,
        )
        assert isinstance(result, tuple)
        out, metrics = result
        assert np.isfinite(out).all()
        assert np.isfinite(float(metrics["calibrated_threshold_db"]))
