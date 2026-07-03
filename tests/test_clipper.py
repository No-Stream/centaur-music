"""Tests for the apply_clipper native effect."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.synth import (
    SAMPLE_RATE,
    apply_clipper,
    apply_native_limiter,
    build_chain_summary_from_dicts,
)


def _sine(freq_hz: float, duration_s: float, amp: float = 1.0) -> np.ndarray:
    n = int(duration_s * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    return amp * np.sin(2.0 * np.pi * freq_hz * t)


class TestClipperLevels:
    """Peak-clipping semantics: threshold_db sets the ceiling."""

    def test_below_threshold_is_near_passthrough(self) -> None:
        # Signal peaks at 0.3 (~-10 dBFS), threshold at -3 dBFS — well below.
        # Resample roundtrip produces tiny (<0.5 dB) peak overshoot and
        # sub-sample shifts near the edges; compare mid-signal RMS.
        signal = _sine(440.0, 0.1, amp=0.3)
        out = apply_clipper(signal, threshold_db=-3.0, knee_width_db=0.0)
        input_peak = float(np.max(np.abs(signal)))
        output_peak = float(np.max(np.abs(out)))
        assert output_peak <= input_peak * 10.0 ** (0.5 / 20.0), (
            f"passthrough overshoot {20 * np.log10(output_peak / input_peak):.2f} dB"
        )
        n = signal.shape[0]
        mid = slice(n // 4, 3 * n // 4)
        in_rms = float(np.sqrt(np.mean(signal[mid] ** 2)))
        out_rms = float(np.sqrt(np.mean(out[mid] ** 2)))
        assert abs(out_rms - in_rms) / in_rms < 0.01

    def test_hard_clip_peak_at_threshold(self) -> None:
        signal = _sine(440.0, 0.1, amp=1.0)
        threshold_db = -6.0
        out = apply_clipper(
            signal,
            threshold_db=threshold_db,
            knee_width_db=0.0,
            oversample_factor=4,
        )
        threshold_lin = 10.0 ** (threshold_db / 20.0)
        peak = float(np.max(np.abs(out)))
        # Brickwall poly knee + OS roundtrip can leave ~0.3 dB overshoot at
        # the discontinuity from sub-sample reconstruction; tolerate <=0.5 dB.
        assert peak <= threshold_lin * 10.0 ** (0.5 / 20.0), (
            f"peak {20 * np.log10(peak):.2f} dB exceeds "
            f"threshold {threshold_db} dB by >0.5 dB"
        )

    def test_soft_knee_reduces_peak(self) -> None:
        # A wide soft knee still attenuates peaks above threshold relative
        # to the dry input, even though it doesn't strictly bound at the
        # ceiling for samples inside the knee region.
        signal = _sine(440.0, 0.1, amp=1.0)
        out = apply_clipper(
            signal, threshold_db=-6.0, knee_width_db=6.0, oversample_factor=4
        )
        input_peak = float(np.max(np.abs(signal)))
        output_peak = float(np.max(np.abs(out)))
        assert output_peak < input_peak, "soft knee should reduce peaks"

    def test_output_length_matches_input(self) -> None:
        signal = _sine(440.0, 0.1, amp=1.0)
        for os in (1, 2, 4, 8):
            out = apply_clipper(signal, oversample_factor=os)
            assert out.shape == signal.shape, f"length mismatch at OS={os}"


class TestClipperKnee:
    """knee_width_db controls soft/hard character monotonically."""

    def test_narrower_knee_has_more_hf_energy(self) -> None:
        # Narrow knee (brickwall) generates more high-harmonic content than
        # a wide knee at the same shave amount; intermediate widths sit
        # between them.  This replaces the old hardness crossfade test.
        signal = _sine(440.0, 0.2, amp=1.0)
        threshold_db = -12.0

        def hf_energy(x: np.ndarray) -> float:
            spec = np.abs(np.fft.rfft(x))
            freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / SAMPLE_RATE)
            return float(np.sum(spec[freqs > 2000.0] ** 2))

        wide = apply_clipper(signal, threshold_db=threshold_db, knee_width_db=6.0)
        mid = apply_clipper(signal, threshold_db=threshold_db, knee_width_db=2.0)
        hard = apply_clipper(signal, threshold_db=threshold_db, knee_width_db=0.0)
        e_wide = hf_energy(wide)
        e_mid = hf_energy(mid)
        e_hard = hf_energy(hard)
        assert e_wide < e_hard, (
            f"hard knee should be brighter than wide knee: "
            f"e_wide={e_wide:.3g}, e_hard={e_hard:.3g}"
        )
        assert e_wide <= e_mid <= e_hard

    def test_below_knee_is_near_passthrough(self) -> None:
        # Musical promise of the polynomial knee: samples well below
        # threshold pass through unshaped.  AD2 on a linear region introduces
        # a small inter-sample averaging (inherent to any divided-difference
        # antialiasing — f'(x) is approximated by (F(x_n) - F(x_{n-1}))/dx,
        # which for a linear f returns the mid-point average).  The
        # *waveform* is not bit-identical, but mid-signal RMS is preserved
        # to within the near-sample-delay tolerance, and the spectrum
        # below Nyquist/2 is preserved.
        n = 4096
        t = np.arange(n) / SAMPLE_RATE
        signal = 0.30 * np.sin(2.0 * np.pi * 220.0 * t)
        out = apply_clipper(
            signal, threshold_db=-3.0, knee_width_db=6.0, oversample_factor=1
        )
        # Mid-signal RMS (skip endpoints to avoid AD2 init transient).
        mid = slice(n // 4, 3 * n // 4)
        in_rms = float(np.sqrt(np.mean(signal[mid] ** 2)))
        out_rms = float(np.sqrt(np.mean(out[mid] ** 2)))
        assert abs(out_rms - in_rms) / in_rms < 0.01, (
            f"below-knee passthrough RMS drift {(out_rms / in_rms - 1) * 100:.2f}%"
        )

    def test_zero_knee_matches_algorithm_hard(self) -> None:
        # knee_width_db=0 with algorithm="poly_knee" should be functionally
        # equivalent to algorithm="hard" (both are a pure clamp).
        signal = _sine(440.0, 0.1, amp=1.0)
        out_poly = apply_clipper(
            signal,
            threshold_db=-6.0,
            knee_width_db=0.0,
            algorithm="poly_knee",
            oversample_factor=4,
        )
        out_hard = apply_clipper(
            signal,
            threshold_db=-6.0,
            algorithm="hard",
            oversample_factor=4,
        )
        # Both run through the same OS resample but different kernels
        # (AD2 poly-knee vs direct np.clip).  Peak magnitudes should
        # match within a small tolerance set by AD2's linear-region
        # averaging plus resample overshoot.
        np.testing.assert_allclose(
            np.max(np.abs(out_poly)), np.max(np.abs(out_hard)), atol=5e-4
        )

    def test_mix_zero_is_dry(self) -> None:
        signal = _sine(440.0, 0.1, amp=1.0)
        out = apply_clipper(signal, threshold_db=-6.0, mix=0.0)
        np.testing.assert_allclose(out, signal)


class TestClipperAlgorithms:
    """algorithm = 'poly_knee' (default) vs 'hard'."""

    def test_hard_algorithm_ignores_knee(self) -> None:
        # algorithm="hard" is a literal clamp; knee_width_db is silently
        # ignored.  Output should be identical whether knee=0 or knee=6.
        signal = _sine(440.0, 0.1, amp=1.0)
        out_a = apply_clipper(
            signal, threshold_db=-6.0, algorithm="hard", knee_width_db=0.0
        )
        out_b = apply_clipper(
            signal, threshold_db=-6.0, algorithm="hard", knee_width_db=6.0
        )
        np.testing.assert_array_equal(out_a, out_b)

    def test_unknown_algorithm_raises(self) -> None:
        signal = _sine(440.0, 0.1)
        with pytest.raises(ValueError, match="algorithm"):
            apply_clipper(signal, algorithm="transition_only")


class TestClipperAutomation:
    """Per-sample threshold_db, knee_width_db, and mix arrays."""

    def test_per_sample_threshold_array(self) -> None:
        # Ramp the threshold from -3 dB at the start to -12 dB at the end.
        # The front half should be near-passthrough, the back half should
        # be heavily clipped.
        signal = _sine(440.0, 0.5, amp=0.5)
        n = signal.shape[0]
        threshold_db_arr = np.linspace(-3.0, -12.0, n)
        out = apply_clipper(
            signal,
            threshold_db=threshold_db_arr,
            knee_width_db=0.0,
            oversample_factor=4,
        )
        # Peak in back half should be lower than back-half dry peak.
        back = out[n // 2 :]
        back_dry = signal[n // 2 :]
        assert float(np.max(np.abs(back))) < float(np.max(np.abs(back_dry)))

    def test_per_sample_mix_array(self) -> None:
        # mix array ramps from 0.0 to 1.0; output front half should equal
        # dry, output back half should equal wet.
        signal = _sine(440.0, 0.3, amp=1.0)
        n = signal.shape[0]
        mix_arr = np.linspace(0.0, 1.0, n)
        out = apply_clipper(
            signal,
            threshold_db=-6.0,
            knee_width_db=0.0,
            mix=mix_arr,
            oversample_factor=4,
        )
        # First few samples should match dry (mix ~ 0).
        np.testing.assert_allclose(out[:10], signal[:10], atol=5e-3)

    def test_threshold_array_length_mismatch_raises(self) -> None:
        signal = _sine(440.0, 0.1, amp=1.0)
        short = np.full(signal.shape[0] - 10, -6.0)
        with pytest.raises(ValueError, match="threshold_db"):
            apply_clipper(signal, threshold_db=short)


class TestClipperOversampling:
    """Higher OS reduces aliasing on content with fast slew."""

    def test_clipper_reduces_aliasing_vs_naive_clip(self) -> None:
        signal = _sine(5000.0, 0.2, amp=1.2)  # hot sine, forces clipping
        threshold_lin = 10.0 ** (-6.0 / 20.0)
        naive = np.clip(signal, -threshold_lin, threshold_lin)
        clean = apply_clipper(
            signal, threshold_db=-6.0, knee_width_db=0.0, oversample_factor=4
        )

        def sub_fundamental_energy(x: np.ndarray, fundamental: float) -> float:
            spec = np.abs(np.fft.rfft(x))
            freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / SAMPLE_RATE)
            return float(np.sum(spec[freqs < fundamental - 100.0] ** 2))

        alias_naive = sub_fundamental_energy(naive, 5000.0)
        alias_clean = sub_fundamental_energy(clean, 5000.0)
        assert alias_clean < 0.5 * alias_naive, (
            f"OS poly-knee clipper should have <half naive alias energy: "
            f"{alias_naive=:.3g}, {alias_clean=:.3g}"
        )


class TestClipperValidation:
    def test_invalid_oversample_factor_raises(self) -> None:
        signal = _sine(440.0, 0.01)
        with pytest.raises(ValueError, match="oversample_factor"):
            apply_clipper(signal, oversample_factor=3)

    def test_handles_stereo(self) -> None:
        mono = _sine(440.0, 0.1, amp=1.0)
        stereo = np.stack([mono, 0.5 * mono])
        out = apply_clipper(stereo, threshold_db=-6.0, knee_width_db=0.0)
        assert out.shape == stereo.shape

    def test_handles_empty_signal(self) -> None:
        out = apply_clipper(np.zeros(0), threshold_db=-6.0)
        assert out.shape == (0,)

    def test_os_16_works(self) -> None:
        signal = _sine(440.0, 0.1, amp=1.0)
        out = apply_clipper(signal, threshold_db=-6.0, oversample_factor=16)
        assert out.shape == signal.shape
        assert np.all(np.isfinite(out))


class TestClipperStereoLink:
    """Linked-envelope + common-attenuation stereo topology."""

    def test_mono_duplicated_matches_mono_path(self) -> None:
        # When both channels are identical, per-channel gains match and the
        # link collapses to a no-op — stereo output matches mono.
        signal = _sine(440.0, 0.1, amp=1.0)
        stereo = np.stack([signal, signal])

        mono_out = apply_clipper(
            signal, threshold_db=-6.0, knee_width_db=0.0, oversample_factor=8
        )
        stereo_out = apply_clipper(
            stereo, threshold_db=-6.0, knee_width_db=0.0, oversample_factor=8
        )

        np.testing.assert_allclose(stereo_out[0], mono_out, atol=1e-2)
        np.testing.assert_allclose(stereo_out[1], mono_out, atol=1e-2)

    def test_silent_right_channel_stays_silent(self) -> None:
        n = int(0.1 * SAMPLE_RATE)
        t = np.arange(n) / SAMPLE_RATE
        loud_l = np.sin(2.0 * np.pi * 440.0 * t)
        silent_r = np.zeros_like(loud_l)
        stereo = np.stack([loud_l, silent_r])

        out = apply_clipper(
            stereo, threshold_db=-6.0, knee_width_db=0.0, oversample_factor=8
        )

        assert float(np.max(np.abs(out[1]))) < 1e-10, (
            "stereo link bled L content into a silent R channel"
        )
        threshold_lin = 10.0 ** (-6.0 / 20.0)
        peak_l = float(np.max(np.abs(out[0])))
        assert peak_l <= threshold_lin * 10.0 ** (0.5 / 20.0), (
            f"L peak {20 * np.log10(peak_l):.2f} dB overshoots threshold -6 dB"
        )

    def test_stereo_link_propagates_attenuation_to_quiet_channel(self) -> None:
        n = int(0.2 * SAMPLE_RATE)
        t = np.arange(n) / SAMPLE_RATE
        loud_l = 1.5 * np.sin(2.0 * np.pi * 220.0 * t)
        quiet_r = 0.2 * np.sin(2.0 * np.pi * 330.0 * t)
        stereo = np.stack([loud_l, quiet_r])

        linked_out = apply_clipper(
            stereo, threshold_db=-6.0, knee_width_db=0.0, oversample_factor=8
        )
        per_ch_r = apply_clipper(
            quiet_r, threshold_db=-6.0, knee_width_db=0.0, oversample_factor=8
        )

        rms_linked_r = float(np.sqrt(np.mean(linked_out[1] ** 2)))
        rms_per_ch_r = float(np.sqrt(np.mean(per_ch_r**2)))

        assert rms_linked_r < 0.95 * rms_per_ch_r, (
            f"linked R's RMS should be pulled down by L's clipping: "
            f"linked RMS {rms_linked_r:.4f} vs per-channel {rms_per_ch_r:.4f}"
        )

        threshold_lin = 10.0 ** (-6.0 / 20.0)
        assert float(np.max(np.abs(linked_out[0]))) <= threshold_lin * 10.0 ** (
            1.0 / 20.0
        )


class TestClipperAnalysis:
    """return_analysis=True returns native metrics matching the log output."""

    def test_return_analysis_includes_shaved_db(self) -> None:
        signal = _sine(440.0, 0.1, amp=1.0)
        result = apply_clipper(
            signal,
            threshold_db=-6.0,
            knee_width_db=0.0,
            oversample_factor=4,
            return_analysis=True,
        )
        out, metrics = result
        assert out.shape == signal.shape
        assert "shaved_db" in metrics
        assert "active_fraction" in metrics
        assert "knee_width_db" in metrics
        assert "algorithm" in metrics
        assert "threshold_db" in metrics
        assert float(metrics["shaved_db"]) > 3.0
        assert float(metrics["active_fraction"]) > 0.3

    def test_inactive_clipper_reports_zero_active_fraction(self) -> None:
        signal = _sine(440.0, 0.1, amp=0.1)
        _, metrics = apply_clipper(
            signal, threshold_db=-3.0, knee_width_db=0.0, return_analysis=True
        )
        assert float(metrics["active_fraction"]) < 0.001
        assert float(metrics["shaved_db"]) < 0.5


class TestClipperAutoCalibration:
    """max_shave_db sets threshold from input peak statistic, bounding work."""

    def _signal_with_p99_peak_db(
        self, peak_db: float, duration_s: float = 0.5
    ) -> np.ndarray:
        amp = 10.0 ** (peak_db / 20.0)
        return _sine(440.0, duration_s, amp=amp)

    def test_zero_shave_is_unity_on_peak(self) -> None:
        signal = self._signal_with_p99_peak_db(-6.0)
        _, metrics = apply_clipper(
            signal,
            max_shave_db=0.0,
            knee_width_db=0.0,
            oversample_factor=4,
            return_analysis=True,
        )
        assert abs(float(metrics["calibrated_threshold_db"]) - (-6.0)) < 0.3
        assert float(metrics["shaved_db"]) < 0.3

    def test_target_shave_is_hit(self) -> None:
        signal = self._signal_with_p99_peak_db(-6.0)
        _, metrics = apply_clipper(
            signal,
            max_shave_db=2.0,
            knee_width_db=0.0,
            oversample_factor=4,
            return_analysis=True,
        )
        threshold = float(metrics["calibrated_threshold_db"])
        assert abs(threshold - (-8.0)) < 0.3, f"expected ~-8 dBFS, got {threshold}"
        shaved = float(metrics["shaved_db"])
        assert abs(shaved - 2.0) < 0.5, f"expected ~2.0 dB shave, got {shaved}"

    def test_sparse_transient_tracked_by_percentile(self) -> None:
        n = 2 * SAMPLE_RATE
        signal = np.zeros(n)
        burst_amp = 10.0 ** (-3.0 / 20.0)
        burst_len = int(0.02 * SAMPLE_RATE)
        stride = int(0.05 * SAMPLE_RATE)
        for i in range(40):
            start = i * stride
            signal[start : start + burst_len] = burst_amp
        _, metrics = apply_clipper(
            signal,
            max_shave_db=1.0,
            knee_width_db=0.0,
            oversample_factor=4,
            calibration_percentile=70.0,
            return_analysis=True,
        )
        ref_peak_db = float(metrics["reference_peak_dbfs"])
        assert ref_peak_db > -6.0, (
            f"p70 should track loud transients; got {ref_peak_db:.2f} dBFS"
        )
        thresh = float(metrics["calibrated_threshold_db"])
        assert abs(thresh - (-4.0)) < 1.0

    def test_max_shave_wins_over_threshold(self) -> None:
        signal = self._signal_with_p99_peak_db(-6.0)
        _, metrics = apply_clipper(
            signal,
            threshold_db=-20.0,
            max_shave_db=1.0,
            knee_width_db=0.0,
            oversample_factor=4,
            return_analysis=True,
        )
        thresh = float(metrics["threshold_db"])
        assert abs(thresh - (-7.0)) < 0.5, (
            f"max_shave_db should win; got threshold {thresh}"
        )
        assert "calibrated_threshold_db" in metrics

    def test_max_shave_none_is_backwards_compatible(self) -> None:
        signal = _sine(440.0, 0.1, amp=1.0)
        _, metrics = apply_clipper(
            signal,
            threshold_db=-6.0,
            knee_width_db=0.0,
            oversample_factor=4,
            return_analysis=True,
        )
        assert "calibrated_threshold_db" not in metrics
        assert "reference_peak_dbfs" not in metrics
        assert float(metrics["threshold_db"]) == -6.0
        assert float(metrics["shaved_db"]) > 3.0

    def test_percentile_changes_calibrated_threshold(self) -> None:
        n = 5 * SAMPLE_RATE
        t = np.arange(n) / SAMPLE_RATE
        sine = np.sin(2.0 * np.pi * 440.0 * t)
        quiet_amp = 10.0 ** (-20.0 / 20.0)
        loud_amp = 10.0 ** (-3.0 / 20.0)
        signal = quiet_amp * sine
        loud_start = int(0.80 * n)
        signal[loud_start:] = loud_amp * sine[loud_start:]
        _, metrics_50 = apply_clipper(
            signal,
            max_shave_db=1.0,
            knee_width_db=0.0,
            oversample_factor=4,
            calibration_percentile=50.0,
            return_analysis=True,
        )
        _, metrics_95 = apply_clipper(
            signal,
            max_shave_db=1.0,
            knee_width_db=0.0,
            oversample_factor=4,
            calibration_percentile=95.0,
            return_analysis=True,
        )
        thresh_50 = float(metrics_50["calibrated_threshold_db"])
        thresh_95 = float(metrics_95["calibrated_threshold_db"])
        assert thresh_50 < thresh_95 - 5.0, (
            f"p50 threshold should be well below p95: p50={thresh_50}, p95={thresh_95}"
        )

    def test_max_shave_with_threshold_array_raises(self) -> None:
        # The new per-sample threshold array is incompatible with
        # max_shave_db calibration.
        signal = _sine(440.0, 0.1, amp=1.0)
        n = signal.shape[0]
        arr = np.full(n, -6.0)
        with pytest.raises(ValueError, match="max_shave_db"):
            apply_clipper(signal, threshold_db=arr, max_shave_db=1.0)


class TestLimiterAnalysis:
    """apply_native_limiter return_analysis surfaces GR metrics."""

    def test_hot_signal_triggers_limiter(self) -> None:
        hot = 2.0 * _sine(220.0, 0.2, amp=1.0)
        _, metrics = apply_native_limiter(hot, threshold_db=-1.0, return_analysis=True)
        assert float(metrics["max_gain_reduction_db"]) < -3.0
        assert float(metrics["active_gain_reduction_fraction"]) > 0.5

    def test_quiet_signal_limiter_inactive(self) -> None:
        quiet = _sine(220.0, 0.2, amp=0.1)
        _, metrics = apply_native_limiter(
            quiet, threshold_db=-1.0, return_analysis=True
        )
        assert float(metrics["active_gain_reduction_fraction"]) < 0.01


class TestChainSummary:
    """build_chain_summary_from_dicts aggregates per-effect IO deltas."""

    def test_single_effect_chain_returns_none(self) -> None:
        entry = [
            {
                "index": 0,
                "kind": "compressor",
                "display_name": "compressor",
                "metrics": {"avg_gain_reduction_db": 2.0},
                "warnings": [],
            }
        ]
        assert build_chain_summary_from_dicts(entry) is None

    def test_two_stage_chain_returns_summary(self) -> None:
        entries = [
            {
                "index": 0,
                "kind": "compressor",
                "display_name": "compressor",
                "metrics": {
                    "avg_gain_reduction_db": 2.0,
                    "thd_delta_pct": 1.0,
                    "spectral_centroid_delta_hz": 50.0,
                    "high_band_delta_db": 0.5,
                    "peak_delta_db": -0.5,
                    "crest_factor_delta_db": 0.3,
                },
                "warnings": [],
            },
            {
                "index": 1,
                "kind": "drive",
                "display_name": "drive",
                "metrics": {
                    "thd_delta_pct": 12.0,
                    "spectral_centroid_delta_hz": 400.0,
                    "high_band_delta_db": 3.5,
                    "peak_delta_db": -0.3,
                    "crest_factor_delta_db": -0.4,
                },
                "warnings": [],
            },
        ]
        summary = build_chain_summary_from_dicts(entries, chain_label="test_bus")
        assert summary is not None
        assert summary.metrics["stage_count"] == 2
        assert summary.metrics["total_thd_growth_pct"] == 13.0
        assert summary.metrics["total_centroid_lift_hz"] == 450.0
        assert summary.metrics["total_high_band_lift_db"] == 4.0
        assert summary.metrics["total_peak_shave_db"] == 0.8

    def test_papery_chain_warns(self) -> None:
        entries = [
            {
                "index": 0,
                "kind": "drive",
                "display_name": "drive",
                "metrics": {"high_band_delta_db": 3.0, "thd_delta_pct": 8.0},
                "warnings": [],
            },
            {
                "index": 1,
                "kind": "clipper",
                "display_name": "clipper",
                "metrics": {"high_band_delta_db": 2.5, "thd_delta_pct": 0.0},
                "warnings": [],
            },
        ]
        summary = build_chain_summary_from_dicts(entries, chain_label="drum_bus")
        assert summary is not None
        codes = {w.code for w in summary.warnings}
        assert "chain_papery" in codes

    def test_papery_warning_suppressed_on_pure_linear_chain(self) -> None:
        """An EQ + delay bus can sum high-band deltas without nonlinearity.

        Regression guard for the bell_delay false-positive seen in
        colundi_arps_study: a feedback delay on a bright source replicates
        existing high-band content into each repeat, which reads as
        `total_high_band_lift_db` > 4 without any stage actually introducing
        harmonic buildup.  The chain_papery warning should not fire in this
        case.
        """
        entries = [
            {
                "index": 0,
                "kind": "eq",
                "display_name": "eq",
                "metrics": {"high_band_delta_db": 2.0, "thd_delta_pct": 0.0},
                "warnings": [],
            },
            {
                "index": 1,
                "kind": "delay",
                "display_name": "delay",
                "metrics": {"high_band_delta_db": 5.0, "thd_delta_pct": 0.0},
                "warnings": [],
            },
        ]
        summary = build_chain_summary_from_dicts(entries, chain_label="bell_delay")
        assert summary is not None
        assert summary.metrics["total_high_band_lift_db"] == 7.0  # still measured
        codes = {w.code for w in summary.warnings}
        assert "chain_papery" not in codes

    def test_papery_warning_fires_when_measured_imd_growth_present(self) -> None:
        """Even non-saturator kinds trip the warning when IMD growth is measured.

        A plugin effect whose ``kind`` isn't in the static nonlinear set
        (e.g. a new effect) still qualifies the chain as nonlinear when the
        per-stage two-tone IMD measurement grew by >= 20%.
        """
        entries = [
            {
                "index": 0,
                "kind": "airwindows",
                "display_name": "airwindows",
                "metrics": {
                    "high_band_delta_db": 3.0,
                    "imd_detection": "two_tone",
                    "imd_ratio_input": 1.0,
                    "imd_ratio_output": 1.5,
                },
                "warnings": [],
            },
            {
                "index": 1,
                "kind": "delay",
                "display_name": "delay",
                "metrics": {"high_band_delta_db": 2.0, "thd_delta_pct": 0.0},
                "warnings": [],
            },
        ]
        summary = build_chain_summary_from_dicts(entries, chain_label="test_bus")
        assert summary is not None
        codes = {w.code for w in summary.warnings}
        assert "chain_papery" in codes

    def test_over_compressed_chain_warns(self) -> None:
        entries = [
            {
                "index": 0,
                "kind": "compressor",
                "display_name": "compressor",
                "metrics": {"avg_gain_reduction_db": 7.0},
                "warnings": [],
            },
            {
                "index": 1,
                "kind": "limiter",
                "display_name": "limiter",
                "metrics": {"max_gain_reduction_db": -3.0},
                "warnings": [],
            },
        ]
        summary = build_chain_summary_from_dicts(entries)
        assert summary is not None
        codes = {w.code for w in summary.warnings}
        assert "chain_over_compressed" in codes

    def test_clean_chain_produces_no_warnings(self) -> None:
        entries = [
            {
                "index": 0,
                "kind": "compressor",
                "display_name": "compressor",
                "metrics": {
                    "avg_gain_reduction_db": 2.0,
                    "thd_delta_pct": 1.0,
                    "spectral_centroid_delta_hz": 30.0,
                    "high_band_delta_db": 0.3,
                    "peak_delta_db": -0.3,
                    "crest_factor_delta_db": 0.2,
                },
                "warnings": [],
            },
            {
                "index": 1,
                "kind": "drive",
                "display_name": "drive",
                "metrics": {
                    "thd_delta_pct": 3.0,
                    "spectral_centroid_delta_hz": 80.0,
                    "high_band_delta_db": 1.0,
                    "peak_delta_db": -0.2,
                    "crest_factor_delta_db": -0.3,
                },
                "warnings": [],
            },
        ]
        summary = build_chain_summary_from_dicts(entries)
        assert summary is not None
        assert summary.warnings == []

    def test_chain_brightness_creep_fires_severe_on_dense_input(self) -> None:
        entries = [
            {
                "index": 0,
                "kind": "preamp",
                "display_name": "preamp",
                "metrics": {
                    "spectral_centroid_delta_hz": 700.0,
                    "imd_detection": "two_tone",
                    "imd_ratio_input": 1.0,
                    "imd_ratio_output": 1.3,
                    "input_active_window_fraction": 0.9,
                },
                "warnings": [],
            },
            {
                "index": 1,
                "kind": "clipper",
                "display_name": "clipper",
                "metrics": {"spectral_centroid_delta_hz": 600.0},
                "warnings": [],
            },
        ]
        summary = build_chain_summary_from_dicts(entries, chain_label="drum_bus")
        assert summary is not None
        codes = {w.code: w for w in summary.warnings}
        assert "chain_brightness_creep" in codes
        assert codes["chain_brightness_creep"].severity == "severe"
        assert summary.metrics["chain_input_active_fraction"] == 0.9

    def test_chain_brightness_creep_capped_to_warning_on_sparse_input(self) -> None:
        """A drum bus that's silent most of the piece shouldn't escalate to severe.

        Regression guard for a real render where drums dropped out for whole
        sections: total_centroid_lift_hz landed at ~1240 (well past the 1200
        severe threshold) purely because the handful of active blocks skewed
        the relative measurement, not because anything was audibly wrong.
        """
        entries = [
            {
                "index": 0,
                "kind": "preamp",
                "display_name": "preamp",
                "metrics": {
                    "spectral_centroid_delta_hz": 700.0,
                    "imd_detection": "two_tone",
                    "imd_ratio_input": 1.0,
                    "imd_ratio_output": 1.3,
                    "input_active_window_fraction": 0.1,
                },
                "warnings": [],
            },
            {
                "index": 1,
                "kind": "clipper",
                "display_name": "clipper",
                "metrics": {"spectral_centroid_delta_hz": 600.0},
                "warnings": [],
            },
        ]
        summary = build_chain_summary_from_dicts(entries, chain_label="drum_bus")
        assert summary is not None
        codes = {w.code: w for w in summary.warnings}
        assert "chain_brightness_creep" in codes, (
            "warning should still fire, just capped"
        )
        assert codes["chain_brightness_creep"].severity == "warning"
        assert summary.metrics["chain_input_active_fraction"] == 0.1
        assert (
            codes["chain_brightness_creep"].metrics["chain_input_active_fraction"]
            == 0.1
        )

    def test_chain_papery_capped_to_warning_on_sparse_input(self) -> None:
        entries = [
            {
                "index": 0,
                "kind": "preamp",
                "display_name": "preamp",
                "metrics": {
                    "high_band_delta_db": 5.0,
                    "imd_detection": "two_tone",
                    "imd_ratio_input": 1.0,
                    "imd_ratio_output": 1.3,
                    "input_active_window_fraction": 0.05,
                },
                "warnings": [],
            },
            {
                "index": 1,
                "kind": "clipper",
                "display_name": "clipper",
                "metrics": {"high_band_delta_db": 4.0},
                "warnings": [],
            },
        ]
        summary = build_chain_summary_from_dicts(entries, chain_label="drum_bus")
        assert summary is not None
        codes = {w.code: w for w in summary.warnings}
        assert "chain_papery" in codes
        assert codes["chain_papery"].severity == "warning"
