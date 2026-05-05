"""Tests for piece-aware auto-calibration on ``apply_drive``.

Mirrors the ``max_shave_db`` / ``target_avg_gr_db`` auto-cal tests but targets
the drive shaper's characteristic THD surface (``target_thd_pct``).

The solver uses ``_saturation_thd`` (440 Hz sine probe, harmonics 2-10) so the
measurement is content-independent — the goal is predictable perceptual
categories (gentle / musical / fuzz) regardless of the incoming signal.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from code_musics.synth import (
    SAMPLE_RATE,
    apply_drive,
)


def _sine_wave(freq_hz: float, duration_s: float, amp: float = 1.0) -> np.ndarray:
    n = int(duration_s * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    return (amp * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float64)


class TestDriveAutoCalibration:
    """``target_thd_pct`` binary-searches ``drive`` for the characteristic THD."""

    def test_target_thd_tracks_on_sine_probe(self) -> None:
        """Solver lands within 1.5% (abs) of a 5% THD target."""
        signal = _sine_wave(220.0, 1.0, amp=0.5)
        result = apply_drive(
            signal,
            target_thd_pct=5.0,
            multiband=False,
            return_analysis=True,
        )
        assert isinstance(result, tuple)
        _, metrics = result
        measured = float(metrics["measured_thd_pct"])
        assert abs(measured - 5.0) < 1.5, (
            f"solver did not converge: target=5.0, measured={measured:.2f}%"
        )

    def test_target_thd_is_monotone_in_solved_drive(self) -> None:
        """Higher target THD -> higher solved drive."""
        signal = _sine_wave(220.0, 0.5, amp=0.5)
        solved_drives = []
        for target in [1.0, 3.0, 7.0, 12.0]:
            result = apply_drive(
                signal,
                target_thd_pct=target,
                multiband=False,
                return_analysis=True,
            )
            assert isinstance(result, tuple)
            _, metrics = result
            solved_drives.append(float(metrics["solved_drive"]))
        diffs = np.diff(solved_drives)
        assert np.all(diffs > 0.0), (
            f"solved_drive should increase with target_thd_pct; got {solved_drives}"
        )

    def test_target_thd_none_is_backwards_compatible(self) -> None:
        """Default path (``target_thd_pct=None``) matches pre-change behavior."""
        signal = _sine_wave(220.0, 0.5, amp=0.5)
        out_default = apply_drive(signal, 1.18, 0.34, multiband=False)
        out_none = apply_drive(signal, 1.18, 0.34, multiband=False, target_thd_pct=None)
        assert isinstance(out_default, np.ndarray)
        assert isinstance(out_none, np.ndarray)
        np.testing.assert_array_equal(out_default, out_none)
        result = apply_drive(signal, 1.18, 0.34, multiband=False, return_analysis=True)
        assert isinstance(result, tuple)
        _, metrics = result
        assert "solved_drive" not in metrics
        assert "target_thd_pct" not in metrics
        assert "measured_thd_pct" not in metrics
        assert "solver_iterations" not in metrics

    def test_conflict_warning_when_both_set(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Non-default ``drive`` + ``target_thd_pct`` logs a WARNING."""
        signal = _sine_wave(220.0, 0.5, amp=0.5)
        with caplog.at_level(logging.WARNING, logger="code_musics.synth"):
            apply_drive(
                signal,
                drive=2.0,
                target_thd_pct=5.0,
                multiband=False,
            )
        warning_texts = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("target_thd_pct wins" in msg for msg in warning_texts), (
            f"expected warning about target_thd_pct winning; got {warning_texts}"
        )

    def test_analysis_dict_contains_calibration_fields(self) -> None:
        """``return_analysis=True`` surfaces calibration metadata."""
        signal = _sine_wave(220.0, 0.5, amp=0.5)
        result = apply_drive(
            signal,
            target_thd_pct=4.0,
            multiband=False,
            return_analysis=True,
        )
        assert isinstance(result, tuple)
        _, metrics = result
        assert "solved_drive" in metrics
        assert "target_thd_pct" in metrics
        assert "measured_thd_pct" in metrics
        assert "solver_iterations" in metrics
        assert float(metrics["target_thd_pct"]) == 4.0
        assert int(metrics["solver_iterations"]) >= 1
        assert float(metrics["solved_drive"]) > 0.0

    def test_negative_target_raises(self) -> None:
        """``target_thd_pct`` must be non-negative when set."""
        signal = _sine_wave(220.0, 0.2, amp=0.5)
        with pytest.raises(ValueError, match="target_thd_pct must be non-negative"):
            apply_drive(signal, target_thd_pct=-1.0)

    def test_zero_target_is_allowed(self) -> None:
        """``target_thd_pct=0.0`` snaps to the lowest drive (near-clean)."""
        signal = _sine_wave(220.0, 0.5, amp=0.5)
        result = apply_drive(
            signal,
            target_thd_pct=0.0,
            multiband=False,
            return_analysis=True,
        )
        assert isinstance(result, tuple)
        out, metrics = result
        assert np.all(np.isfinite(out))
        assert float(metrics["solved_drive"]) >= 0.0

    def test_legacy_algorithm_auto_mode(self) -> None:
        """``algorithm='legacy'`` + ``target_thd_pct`` also converges."""
        signal = _sine_wave(220.0, 0.5, amp=0.5)
        result = apply_drive(
            signal,
            target_thd_pct=5.0,
            algorithm="legacy",
            return_analysis=True,
        )
        assert isinstance(result, tuple)
        out, metrics = result
        assert np.all(np.isfinite(out))
        measured = float(metrics["measured_thd_pct"])
        assert abs(measured - 5.0) < 2.0, (
            f"legacy solver did not converge: target=5.0, measured={measured:.2f}%"
        )

    def test_multiband_auto_mode_converges(self) -> None:
        """``multiband=True`` + ``target_thd_pct`` produces finite output and solved drive."""
        signal = _sine_wave(220.0, 0.5, amp=0.5)
        result = apply_drive(
            signal,
            target_thd_pct=5.0,
            multiband=True,
            return_analysis=True,
        )
        assert isinstance(result, tuple)
        out, metrics = result
        assert np.all(np.isfinite(out))
        assert float(metrics["solved_drive"]) > 0.0
        assert int(metrics["solver_iterations"]) >= 1

    def test_solver_respects_iteration_cap(self) -> None:
        """Solver does not exceed its iteration cap; returns finite result."""
        signal = _sine_wave(220.0, 0.5, amp=0.5)
        result = apply_drive(
            signal,
            target_thd_pct=99.0,
            multiband=False,
            return_analysis=True,
        )
        assert isinstance(result, tuple)
        out, metrics = result
        assert np.all(np.isfinite(out))
        assert 1 <= int(metrics["solver_iterations"]) <= 20
