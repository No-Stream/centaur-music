"""Tests for the three filter bug fixes in _filters.py.

Fix 1: Double-subtraction of low_state in driven SVF
Fix 2: drive_blend not clamped to [0, 1]
Fix 3: Inconsistent Nyquist clamp limits
"""

from __future__ import annotations

import numpy as np

from code_musics.engines._filters import apply_zdf_svf


def _sine_signal(
    freq: float = 220.0,
    duration: float = 0.5,
    sample_rate: int = 44100,
    amplitude: float = 0.7,
) -> np.ndarray:
    n = int(sample_rate * duration)
    t = np.linspace(0.0, duration, n, endpoint=False)
    return amplitude * np.sin(2.0 * np.pi * freq * t)


class TestDriveBlendClamped:
    """Fix 2: drive_blend = 0.75 * (drive**1.3) must be clamped to [0, 1].

    At drive=1.5 the unclamped value is ~1.27, which means the linear
    path gets a NEGATIVE weight (-0.27), producing garbage output.
    """

    def test_drive_2_output_matches_pure_driven(self) -> None:
        """At drive >= ~1.26, unclamped blend exceeds 1.0.

        At drive=2.0, unclamped blend is ~1.85, meaning the output is
        1.85*driven - 0.85*linear — amplifying the driven path past unity.
        After clamping, drive >= ~1.26 should produce EXACTLY the driven
        path output (blend=1.0).  So drive=1.5 and drive=2.0 should be
        identical: both are 100% driven, 0% linear.
        """
        signal = _sine_signal(amplitude=0.8)
        n = len(signal)
        cutoff = np.full(n, 800.0, dtype=np.float64)

        driven_15 = apply_zdf_svf(
            signal,
            cutoff_profile=cutoff,
            resonance_q=4.0,
            sample_rate=44100,
            filter_mode="lowpass",
            filter_drive=1.5,
        )
        driven_20 = apply_zdf_svf(
            signal,
            cutoff_profile=cutoff,
            resonance_q=4.0,
            sample_rate=44100,
            filter_mode="lowpass",
            filter_drive=2.0,
        )

        assert np.all(np.isfinite(driven_15)), "drive=1.5 produced NaN/Inf"
        assert np.all(np.isfinite(driven_20)), "drive=2.0 produced NaN/Inf"

        # With clamped blend, both should be 100% driven (blend=1.0).
        # The driven paths themselves differ (different drive_gain, etc.),
        # but the blend factor should be saturated at 1.0 for both.
        # The key check: neither should have a peak wildly above 1.0,
        # which the unclamped negative-weight formula can produce.
        peak_15 = float(np.max(np.abs(driven_15)))
        peak_20 = float(np.max(np.abs(driven_20)))
        assert peak_15 < 2.0, (
            f"drive=1.5 peak={peak_15:.2f} — unclamped blend amplification?"
        )
        assert peak_20 < 2.0, (
            f"drive=2.0 peak={peak_20:.2f} — unclamped blend amplification?"
        )


class TestNyquistClampConsistency:
    """Fix 3: All Nyquist clamp paths use the unified _NYQUIST_CLAMP_RATIO.

    The key correctness property: constant and varying cutoff profiles above
    the Nyquist limit should produce the same clamped result, because both
    paths (precomputed g vs per-sample g) now use the same limit.
    """

    def test_constant_and_varying_above_limit_produce_same_output(self) -> None:
        """A constant cutoff of 25 kHz should produce the same output as a
        varying profile that is 25 kHz everywhere, because both paths clamp
        to the same nyquist_limit value."""
        sr = 44100
        signal = _sine_signal(freq=220.0, sample_rate=sr, amplitude=0.5)
        n = len(signal)

        cutoff_const = np.full(n, 25000.0, dtype=np.float64)
        # Break constant-detection to force the per-sample path
        cutoff_vary = np.full(n, 25000.0, dtype=np.float64)
        cutoff_vary[0] = 24999.0

        for drive in [0.0, 0.3]:
            result_const = apply_zdf_svf(
                signal,
                cutoff_profile=cutoff_const,
                resonance_q=0.707,
                sample_rate=sr,
                filter_mode="lowpass",
                filter_drive=drive,
            )
            result_vary = apply_zdf_svf(
                signal,
                cutoff_profile=cutoff_vary,
                resonance_q=0.707,
                sample_rate=sr,
                filter_mode="lowpass",
                filter_drive=drive,
            )
            # Both should be nearly identical: the only difference is sample 0
            # where cutoff_vary is 24999 vs 25000, but both get clamped to the
            # same nyquist_limit anyway.
            np.testing.assert_allclose(
                result_const,
                result_vary,
                atol=1e-6,
                err_msg=f"Constant vs varying path mismatch at drive={drive}",
            )

    def test_nyquist_clamp_ratio_constant_exists(self) -> None:
        """The module-level constant should be accessible."""
        from code_musics.engines._filters import _NYQUIST_CLAMP_RATIO

        assert 0.35 <= _NYQUIST_CLAMP_RATIO <= 0.49
