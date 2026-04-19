"""Tests for JI-aware harmonic drift tool."""

import math

import numpy as np
import pytest

from code_musics.automation import AutomationSpec
from code_musics.tuning import enumerate_ji_ratios, tenney_height


class TestTenneyHeight:
    """Verify Tenney height computation for known JI intervals."""

    def test_perfect_fifth(self) -> None:
        assert math.isclose(tenney_height(3 / 2), math.log2(6), rel_tol=1e-9)

    def test_major_third(self) -> None:
        assert math.isclose(tenney_height(5 / 4), math.log2(20), rel_tol=1e-9)

    def test_octave(self) -> None:
        assert math.isclose(tenney_height(2 / 1), math.log2(2), rel_tol=1e-9)

    def test_unison(self) -> None:
        assert math.isclose(tenney_height(1 / 1), math.log2(1), rel_tol=1e-9)

    def test_septimal_seventh(self) -> None:
        assert math.isclose(tenney_height(7 / 4), math.log2(28), rel_tol=1e-9)


class TestEnumerateJIRatios:
    """Verify JI ratio enumeration within a pitch range."""

    def test_5_limit_contains_known_ratios(self) -> None:
        ratios = enumerate_ji_ratios(1.0, 2.0, prime_limit=5)
        ratio_set = set(ratios)
        for expected in [1.0, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 2.0]:
            assert any(math.isclose(r, expected, rel_tol=1e-9) for r in ratio_set), (
                f"Expected {expected} in 5-limit ratios"
            )

    def test_respects_prime_limit(self) -> None:
        ratios_5 = enumerate_ji_ratios(1.0, 2.0, prime_limit=5)
        ratios_7 = enumerate_ji_ratios(1.0, 2.0, prime_limit=7)
        has_7_4_in_5_limit = any(math.isclose(r, 7 / 4, rel_tol=1e-9) for r in ratios_5)
        has_7_4_in_7_limit = any(math.isclose(r, 7 / 4, rel_tol=1e-9) for r in ratios_7)
        assert not has_7_4_in_5_limit, "7/4 should not appear in 5-limit"
        assert has_7_4_in_7_limit, "7/4 should appear in 7-limit"

    def test_sorted_ascending(self) -> None:
        ratios = enumerate_ji_ratios(1.0, 2.0, prime_limit=7)
        for i in range(len(ratios) - 1):
            assert ratios[i] <= ratios[i + 1], "Ratios should be sorted ascending"

    def test_respects_range_bounds(self) -> None:
        ratios = enumerate_ji_ratios(1.2, 1.6, prime_limit=7)
        for r in ratios:
            assert 1.2 <= r <= 1.6, f"Ratio {r} outside [1.2, 1.6]"

    def test_respects_max_height(self) -> None:
        ratios = enumerate_ji_ratios(1.0, 2.0, prime_limit=7, max_height=5.0)
        for r in ratios:
            assert tenney_height(r) <= 5.0 + 1e-9, (
                f"Ratio {r} has Tenney height {tenney_height(r)} > 5.0"
            )

    def test_no_duplicates(self) -> None:
        ratios = enumerate_ji_ratios(1.0, 2.0, prime_limit=7)
        for i in range(len(ratios)):
            for j in range(i + 1, len(ratios)):
                assert not math.isclose(ratios[i], ratios[j], rel_tol=1e-9), (
                    f"Duplicate ratios: {ratios[i]} and {ratios[j]}"
                )


class TestHarmonicDrift:
    """Verify harmonic_drift generates correct automation lanes."""

    def test_returns_automation_lanes(self) -> None:
        from code_musics.harmonic_drift import harmonic_drift

        start_chord = [1.0, 5 / 4, 3 / 2]
        end_chord = [1.0, 9 / 7, 7 / 4]
        lanes = harmonic_drift(
            start_chord=start_chord,
            end_chord=end_chord,
            duration=4.0,
        )
        assert isinstance(lanes, list)
        assert len(lanes) == len(start_chord)
        for lane in lanes:
            assert isinstance(lane, AutomationSpec)
            assert lane.target.kind == "pitch_ratio"
            assert lane.mode == "multiply"

    def test_zero_attraction_is_approximately_linear(self) -> None:
        from code_musics.harmonic_drift import harmonic_drift

        start_chord = [3 / 2]
        end_chord = [7 / 4]
        # Pass glide_ms=None to drift across the entire note duration so the
        # linear-in-log check can compare against uniform progress.
        lanes = harmonic_drift(
            start_chord=start_chord,
            end_chord=end_chord,
            duration=4.0,
            attraction=0.0,
            smoothness=0.0,
            wander=0.0,
            glide_ms=None,
        )
        lane = lanes[0]
        n_segments = len(lane.segments)
        assert n_segments > 0

        # Sample the trajectory by evaluating segments at their midpoints.
        # With zero attraction, the path should be linear in log-pitch space.
        log_start = np.log2(1.0)  # pitch_ratio starts at 1.0 (no change)
        expected_end_ratio = (7 / 4) / (3 / 2)
        log_end = np.log2(expected_end_ratio)

        max_deviation = 0.0
        for segment in lane.segments:
            assert segment.start_value is not None and segment.end_value is not None
            mid_time = (segment.start + segment.end) / 2.0
            progress = mid_time / 4.0
            expected_log = log_start + progress * (log_end - log_start)
            mid_value = segment.start_value + 0.5 * (
                segment.end_value - segment.start_value
            )
            actual_log = np.log2(mid_value)
            max_deviation = max(max_deviation, abs(actual_log - expected_log))

        assert max_deviation < 0.01, (
            f"Zero-attraction path deviates {max_deviation:.4f} from linear (log-pitch)"
        )

    def test_glide_window_holds_unity_before_glide(self) -> None:
        """With default glide_ms, the note holds start pitch until the glide window."""
        from code_musics.harmonic_drift import harmonic_drift

        lanes = harmonic_drift(
            start_chord=[1.0],
            end_chord=[5 / 4],  # major third — under the interval cap
            duration=4.0,
            attraction=0.0,
            smoothness=0.0,
            wander=0.0,
            glide_ms=250.0,
        )
        lane = lanes[0]
        # Leading segment holds unity until glide start (3.75s in).
        first = lane.segments[0]
        assert first.shape == "hold"
        assert math.isclose(first.value or 0.0, 1.0, abs_tol=1e-9)
        assert math.isclose(first.start, 0.0, abs_tol=1e-9)
        assert math.isclose(first.end, 3.75, abs_tol=1e-3)
        # Subsequent glide segments land in the last 250 ms.
        for seg in lane.segments[1:]:
            assert seg.start >= 3.75 - 1e-6
            assert seg.end <= 4.0 + 1e-6
        # Final end value reaches the target ratio.
        assert math.isclose(lane.segments[-1].end_value or 0.0, 5 / 4, rel_tol=1e-3)

    def test_large_interval_skips_drift(self) -> None:
        """Voices whose interval exceeds max_interval_cents emit flat unity lanes."""
        from code_musics.harmonic_drift import harmonic_drift

        # 1.0 -> 4.0 is 2 octaves = 2400 cents, well over the 700-cent default.
        lanes = harmonic_drift(
            start_chord=[1.0, 3 / 2],
            end_chord=[4.0, 7 / 4],  # first voice: leap; second: under cap
            duration=4.0,
            attraction=0.0,
            smoothness=0.0,
            wander=0.0,
            glide_ms=None,
        )
        # First voice skipped: single hold segment at 1.0.
        skipped = lanes[0]
        assert len(skipped.segments) == 1
        assert skipped.segments[0].shape == "hold"
        assert math.isclose(skipped.segments[0].value or 0.0, 1.0, abs_tol=1e-9)
        # Second voice still drifts: ends near 7/6 (= (7/4) / (3/2)).
        assert math.isclose(
            lanes[1].segments[-1].end_value or 0.0, (7 / 4) / (3 / 2), rel_tol=1e-3
        )

    def test_high_attraction_lingers_near_ji(self) -> None:
        from code_musics.harmonic_drift import harmonic_drift

        # Drift from 1.0 to 2.0 -- passes through many JI ratios.
        start_chord = [1.0]
        end_chord = [2.0]
        duration = 4.0

        # Use glide_ms=None and a permissive max_interval_cents because
        # start=1.0 -> end=2.0 is a whole octave (1200 cents) and would
        # otherwise skip drift under the default 700-cent cap.
        lanes_low = harmonic_drift(
            start_chord=start_chord,
            end_chord=end_chord,
            duration=duration,
            attraction=0.0,
            smoothness=0.0,
            wander=0.0,
            glide_ms=None,
            max_interval_cents=10_000.0,
        )
        lanes_high = harmonic_drift(
            start_chord=start_chord,
            end_chord=end_chord,
            duration=duration,
            attraction=0.9,
            smoothness=0.0,
            wander=0.0,
            glide_ms=None,
            max_interval_cents=10_000.0,
        )

        ji_ratios = enumerate_ji_ratios(1.0, 2.0, prime_limit=7, max_height=6.0)
        threshold_cents = 5.0  # tight threshold to distinguish attraction levels

        def fraction_near_ji(lane: AutomationSpec) -> float:
            count_near = 0
            total = 0
            for segment in lane.segments:
                assert segment.start_value is not None and segment.end_value is not None
                for frac in [0.0, 0.5, 1.0]:
                    value = segment.start_value + frac * (
                        segment.end_value - segment.start_value
                    )
                    # value is a pitch_ratio multiplier relative to start_chord[0]=1.0
                    # so the actual ratio IS the value.
                    actual_ratio = value
                    near_any = any(
                        abs(1200.0 * np.log2(actual_ratio / ji_r)) < threshold_cents
                        for ji_r in ji_ratios
                        if ji_r > 0
                    )
                    if near_any:
                        count_near += 1
                    total += 1
            return count_near / max(total, 1)

        frac_low = fraction_near_ji(lanes_low[0])
        frac_high = fraction_near_ji(lanes_high[0])

        assert frac_high > frac_low, (
            f"High attraction ({frac_high:.2%}) should spend more time near JI "
            f"than low attraction ({frac_low:.2%})"
        )

    def test_preserves_endpoints(self) -> None:
        from code_musics.harmonic_drift import harmonic_drift

        start_chord = [5 / 4, 3 / 2]
        end_chord = [9 / 7, 7 / 4]
        # glide_ms=None so the trajectory starts at t=0 and the first segment
        # is the initial linear step (not a leading hold).
        lanes = harmonic_drift(
            start_chord=start_chord,
            end_chord=end_chord,
            duration=4.0,
            glide_ms=None,
        )

        for i, lane in enumerate(lanes):
            expected_end_ratio = end_chord[i] / start_chord[i]
            first_seg = lane.segments[0]
            last_seg = lane.segments[-1]
            assert first_seg.start_value is not None
            assert last_seg.end_value is not None
            assert math.isclose(first_seg.start_value, 1.0, rel_tol=1e-3), (
                f"Voice {i}: first segment start_value={first_seg.start_value}, expected ~1.0"
            )
            assert math.isclose(last_seg.end_value, expected_end_ratio, rel_tol=1e-3), (
                f"Voice {i}: last segment end_value={last_seg.end_value}, "
                f"expected ~{expected_end_ratio}"
            )

    def test_deterministic_with_seed(self) -> None:
        from code_musics.harmonic_drift import harmonic_drift

        lanes_a = harmonic_drift(
            start_chord=[1.0, 5 / 4],
            end_chord=[3 / 2, 7 / 4],
            duration=4.0,
            attraction=0.5,
            wander=0.3,
            seed=42,
        )
        lanes_b = harmonic_drift(
            start_chord=[1.0, 5 / 4],
            end_chord=[3 / 2, 7 / 4],
            duration=4.0,
            attraction=0.5,
            wander=0.3,
            seed=42,
        )

        for la, lb in zip(lanes_a, lanes_b, strict=True):
            assert len(la.segments) == len(lb.segments)
            for sa, sb in zip(la.segments, lb.segments, strict=True):
                assert sa.start_value == sb.start_value
                assert sa.end_value == sb.end_value

    def test_chord_length_mismatch_raises(self) -> None:
        from code_musics.harmonic_drift import harmonic_drift

        with pytest.raises(ValueError, match="same length"):
            harmonic_drift(
                start_chord=[1.0, 5 / 4],
                end_chord=[3 / 2],
                duration=4.0,
            )

    def test_static_voice_produces_near_unity_automation(self) -> None:
        """When start == end for a voice, automation should stay near 1.0."""
        from code_musics.harmonic_drift import harmonic_drift

        lanes = harmonic_drift(
            start_chord=[3 / 2],
            end_chord=[3 / 2],
            duration=4.0,
            attraction=0.5,
            wander=0.0,
        )
        for segment in lanes[0].segments:
            if segment.shape == "hold":
                assert segment.value is not None
                assert math.isclose(segment.value, 1.0, abs_tol=1e-6)
            else:
                assert segment.start_value is not None and segment.end_value is not None
                assert math.isclose(segment.start_value, 1.0, abs_tol=1e-6)
                assert math.isclose(segment.end_value, 1.0, abs_tol=1e-6)
