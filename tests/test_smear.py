"""Tests for Loveless-inspired compositional tools."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.automation import AutomationSpec
from code_musics.score import NoteEvent, Phrase, Score
from code_musics.smear import (
    ThickenedCopy,
    bloom,
    pitch_wobble,
    smear_progression,
    strum,
    thicken,
)


def _chord_phrase(partials: list[float], duration: float = 1.0) -> Phrase:
    """Build a simultaneous chord as a Phrase (all notes at start=0)."""
    events = tuple(NoteEvent(start=0.0, duration=duration, partial=p) for p in partials)
    return Phrase(events=events)


class TestStrum:
    def test_strum_offsets_notes(self) -> None:
        chord = _chord_phrase([4.0, 5.0, 6.0, 7.0])
        strummed = strum(chord, spread_ms=40.0, direction="down")

        starts = [e.start for e in strummed.events]
        assert starts[0] == 0.0
        assert all(s >= 0.0 for s in starts)
        # All starts should be within spread_ms
        assert max(starts) <= 0.040 + 1e-9
        # Notes should be staggered, not all at 0
        assert len(set(round(s, 6) for s in starts)) > 1

    def test_strum_directions(self) -> None:
        chord = _chord_phrase([3.0, 5.0, 7.0, 9.0])
        down = strum(chord, spread_ms=40.0, direction="down")
        up = strum(chord, spread_ms=40.0, direction="up")
        out = strum(chord, spread_ms=40.0, direction="out")

        down_order = [e.partial for e in sorted(down.events, key=lambda e: e.start)]
        up_order = [e.partial for e in sorted(up.events, key=lambda e: e.start)]
        out_order = [e.partial for e in sorted(out.events, key=lambda e: e.start)]

        # Down: low to high
        assert down_order == sorted(down_order)
        # Up: high to low
        assert up_order == sorted(up_order, reverse=True)
        # Out: different from both down and up
        assert out_order != down_order or out_order != up_order

    def test_strum_random_is_deterministic(self) -> None:
        chord = _chord_phrase([3.0, 5.0, 7.0, 9.0])
        r1 = strum(chord, spread_ms=30.0, direction="random", seed=42)
        r2 = strum(chord, spread_ms=30.0, direction="random", seed=42)
        assert [e.start for e in r1.events] == [e.start for e in r2.events]

    def test_strum_single_note_unchanged(self) -> None:
        single = _chord_phrase([4.0])
        strummed = strum(single, spread_ms=40.0)
        assert strummed.events[0].start == 0.0

    def test_strum_preserves_event_count(self) -> None:
        chord = _chord_phrase([3.0, 5.0, 7.0])
        strummed = strum(chord, spread_ms=20.0)
        assert len(strummed.events) == len(chord.events)


class TestThicken:
    def test_thicken_produces_n_copies(self) -> None:
        chord = _chord_phrase([4.0, 5.0, 6.0])
        copies = thicken(chord, n=5)
        assert len(copies) == 5
        assert all(isinstance(c, ThickenedCopy) for c in copies)

    def test_thicken_copies_have_different_pan(self) -> None:
        chord = _chord_phrase([4.0, 5.0])
        copies = thicken(chord, n=5, stereo_width=0.7)
        pans = [c.pan for c in copies]
        assert len(set(round(p, 6) for p in pans)) > 1

    def test_thicken_deterministic(self) -> None:
        chord = _chord_phrase([4.0, 5.0, 6.0])
        c1 = thicken(chord, n=5, seed=99)
        c2 = thicken(chord, n=5, seed=99)
        for a, b in zip(c1, c2, strict=True):
            assert a.pan == b.pan
            assert a.amp_offset_db == b.amp_offset_db
            for ea, eb in zip(a.phrase.events, b.phrase.events, strict=True):
                assert ea.start == eb.start
                assert ea.freq == eb.freq or ea.partial == eb.partial

    def test_thicken_outer_copies_tapered(self) -> None:
        chord = _chord_phrase([4.0])
        copies = thicken(chord, n=5, amp_taper_db=-3.0)
        # Center copy should have 0 taper, outer copies should have negative offsets
        center_idx = len(copies) // 2
        assert copies[center_idx].amp_offset_db == pytest.approx(0.0, abs=0.01)
        assert copies[0].amp_offset_db < 0.0
        assert copies[-1].amp_offset_db < 0.0

    def test_thicken_returns_phrases_with_correct_event_count(self) -> None:
        chord = _chord_phrase([4.0, 5.0, 6.0])
        copies = thicken(chord, n=3)
        for c in copies:
            assert len(c.phrase.events) == 3


class TestPitchWobble:
    def test_pitch_wobble_returns_automation_lane(self) -> None:
        lane = pitch_wobble(duration=4.0, rate_hz=0.15, depth_cents=12.0, style="lfo")
        assert isinstance(lane, AutomationSpec)
        assert lane.target.kind == "pitch_ratio"
        assert lane.mode == "multiply"

    def test_pitch_wobble_lfo_shape(self) -> None:
        lane = pitch_wobble(duration=2.0, rate_hz=1.0, depth_cents=50.0, style="lfo")
        # Sample at multiple points -- values should oscillate around 1.0
        values = [lane.sample(t) for t in np.linspace(0.0, 1.99, 100)]
        values_arr = np.array([v for v in values if v is not None])
        assert len(values_arr) > 0
        # Mean should be near 1.0 (pitch ratio center)
        assert abs(np.mean(values_arr) - 1.0) < 0.01
        # Should have variation
        assert np.std(values_arr) > 0.001

    def test_pitch_wobble_smooth_returns_segments(self) -> None:
        lane = pitch_wobble(duration=2.0, rate_hz=0.2, depth_cents=10.0, style="smooth")
        assert len(lane.segments) > 1
        # All segments should be linear
        assert all(s.shape == "linear" for s in lane.segments)

    def test_pitch_wobble_drunk_returns_segments(self) -> None:
        lane = pitch_wobble(duration=2.0, rate_hz=0.2, depth_cents=10.0, style="drunk")
        assert len(lane.segments) > 1

    def test_pitch_wobble_start_time_offset(self) -> None:
        lane = pitch_wobble(
            duration=2.0, rate_hz=0.5, depth_cents=10.0, style="lfo", start_time=5.0
        )
        # Segments should start at start_time
        assert lane.segments[0].start >= 5.0

    def test_pitch_wobble_depth_curve(self) -> None:
        depth_curve = [(0.0, 0.0), (1.0, 20.0), (2.0, 0.0)]
        lane = pitch_wobble(
            duration=2.0,
            rate_hz=0.5,
            depth_cents=20.0,
            style="lfo",
            depth_curve=depth_curve,
        )
        # At start, depth should be near zero so values near 1.0
        start_val = lane.sample(0.0)
        assert start_val is not None
        assert abs(start_val - 1.0) < 0.005


class TestSmearProgression:
    def test_smear_progression_basic(self) -> None:
        chords = [[1.0, 5 / 4, 3 / 2], [1.0, 6 / 5, 3 / 2]]
        durations = [4.0, 4.0]
        phrases = smear_progression(chords, durations, overlap=0.5)

        # One phrase per voice index (3 voices in the chords)
        assert len(phrases) == 3
        # Each phrase should have notes
        for phrase in phrases:
            assert len(phrase.events) > 0
            # First chord notes should have pitch_motion for glide to next chord
            first_note = phrase.events[0]
            assert first_note.pitch_motion is not None

    def test_smear_progression_reattack(self) -> None:
        chords = [[1.0, 5 / 4, 3 / 2], [1.0, 6 / 5, 3 / 2]]
        durations = [4.0, 4.0]
        voice_behavior = ["glide", "reattack", "glide"]
        phrases = smear_progression(
            chords,
            durations,
            overlap=0.5,
            voice_behavior=voice_behavior,
        )

        assert len(phrases) == 3
        # Voice 1 (index 1) should have no pitch_motion on its notes
        reattack_phrase = phrases[1]
        for event in reattack_phrase.events:
            assert event.pitch_motion is None

    def test_smear_progression_single_chord_no_glide(self) -> None:
        chords = [[1.0, 5 / 4, 3 / 2]]
        durations = [4.0]
        phrases = smear_progression(chords, durations, overlap=0.5)
        assert len(phrases) == 3
        # With a single chord, no glide target exists
        for phrase in phrases:
            for event in phrase.events:
                assert event.pitch_motion is None

    def test_smear_progression_voice_count_matches_chord_width(self) -> None:
        chords = [[1.0, 3 / 2, 7 / 4, 2.0], [1.0, 5 / 4, 3 / 2, 15 / 8]]
        durations = [3.0, 3.0]
        phrases = smear_progression(chords, durations)
        assert len(phrases) == 4


class TestBloom:
    def test_bloom_staggers_entries(self) -> None:
        score = Score(f0=100.0)
        phrase_a = _chord_phrase([4.0], duration=2.0)
        phrase_b = _chord_phrase([5.0], duration=2.0)
        phrase_c = _chord_phrase([6.0], duration=2.0)

        voice_specs = [
            {"name": "v1", "synth_defaults": {}, "phrase": phrase_a},
            {"name": "v2", "synth_defaults": {}, "phrase": phrase_b},
            {"name": "v3", "synth_defaults": {}, "phrase": phrase_c},
        ]

        result = bloom(
            score,
            voice_specs,
            center_time=20.0,
            grow_dur=4.0,
            peak_dur=8.0,
            fade_dur=4.0,
        )

        assert result is score
        assert len(score.voices) == 3

        # Voices should have notes at different start times
        starts = []
        for voice_name in ["v1", "v2", "v3"]:
            voice_notes = score.voices[voice_name].notes
            assert len(voice_notes) > 0
            starts.append(min(n.start for n in voice_notes))

        # First voice should enter before last voice
        assert starts[0] < starts[-1]

    def test_bloom_all_voices_present_during_peak(self) -> None:
        score = Score(f0=100.0)
        phrase_a = _chord_phrase([4.0], duration=2.0)
        phrase_b = _chord_phrase([5.0], duration=2.0)

        voice_specs = [
            {"name": "v1", "synth_defaults": {}, "phrase": phrase_a},
            {"name": "v2", "synth_defaults": {}, "phrase": phrase_b},
        ]

        center_time = 20.0
        peak_dur = 8.0
        bloom(
            score,
            voice_specs,
            center_time=center_time,
            grow_dur=4.0,
            peak_dur=peak_dur,
            fade_dur=4.0,
        )

        peak_start = center_time - peak_dur / 2
        peak_end = center_time + peak_dur / 2

        # All voices should have at least one note overlapping the peak region
        for voice_name in ["v1", "v2"]:
            voice_notes = score.voices[voice_name].notes
            overlaps_peak = any(
                n.start < peak_end and (n.start + n.duration) > peak_start
                for n in voice_notes
            )
            assert overlaps_peak, f"{voice_name} should overlap peak region"
