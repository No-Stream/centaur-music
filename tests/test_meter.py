"""Musical timeline helper tests."""

from __future__ import annotations

import pytest

from code_musics.meter import B, E, Groove, M, Q, S, Timeline, dotted, triplet, tuplet


def test_timeline_converts_common_note_values_to_seconds() -> None:
    timeline = Timeline(bpm=60.0)

    assert timeline.duration(Q) == pytest.approx(1.0)
    assert timeline.duration(E) == pytest.approx(0.5)
    assert timeline.duration(dotted(Q)) == pytest.approx(1.5)
    assert timeline.duration(triplet(Q)) == pytest.approx(2.0 / 3.0)


def test_timeline_supports_bar_beat_positions_and_measure_lengths() -> None:
    timeline = Timeline(bpm=120.0, meter=(6, 8))

    assert timeline.measures(1.0) == pytest.approx(1.5)
    assert timeline.at(bar=3, beat=1.0) == pytest.approx(3.5)
    assert timeline.position(M(2.5)) == pytest.approx(2.25)
    assert timeline.position(B(2.0)) == pytest.approx(1.0)


def test_timeline_locate_resolves_bar_and_beat() -> None:
    timeline = Timeline(bpm=120.0, meter=(4, 4))

    location = timeline.locate(4.5)

    assert location.bar == 3
    assert location.beat_within_bar == pytest.approx(1.0)
    assert location.absolute_beats == pytest.approx(9.0)


def test_timeline_validates_bad_inputs() -> None:
    with pytest.raises(ValueError, match="bpm must be positive"):
        Timeline(bpm=0.0)
    with pytest.raises(ValueError, match="meter denominator"):
        Timeline(bpm=90.0, meter=(4, 3))
    with pytest.raises(ValueError, match="bar must be at least 1"):
        Timeline(bpm=90.0).at(bar=0)
    with pytest.raises(ValueError, match="measure must be finite and at least 1.0"):
        M(0.5)
    with pytest.raises(ValueError, match="amount must be finite"):
        Groove.eighths_swing(1.0)


def test_timeline_supports_eighth_groove_positions_and_round_trip_locate() -> None:
    timeline = Timeline(bpm=120.0, groove=Groove.eighths_swing(2.0 / 3.0))

    assert timeline.position(B(0.5)) == pytest.approx(1.0 / 3.0)
    assert timeline.position(B(1.0)) == pytest.approx(0.5)
    assert timeline.duration(E) == pytest.approx(0.25)

    offbeat_location = timeline.locate(1.0 / 3.0)
    assert offbeat_location.absolute_beats == pytest.approx(0.5)
    assert offbeat_location.bar == 1
    assert offbeat_location.beat_within_bar == pytest.approx(0.5)


def test_timeline_supports_sixteenth_groove_positions() -> None:
    timeline = Timeline(bpm=120.0, groove=Groove.sixteenths_swing(2.0 / 3.0))

    assert timeline.position(B(0.25)) == pytest.approx(1.0 / 6.0)
    assert timeline.position(B(0.5)) == pytest.approx(0.25)
    assert timeline.position(B(0.75)) == pytest.approx(5.0 / 12.0)


def test_timeline_accepts_explicit_straight_groove() -> None:
    timeline = Timeline(bpm=120.0, groove=Groove.eighths_swing(0.5))

    assert timeline.position(B(0.5)) == pytest.approx(0.25)
    assert timeline.locate(0.25).absolute_beats == pytest.approx(0.5)


# --- Groove validation ---


def test_groove_rejects_empty_timing_offsets() -> None:
    with pytest.raises(ValueError, match="timing_offsets must be non-empty"):
        Groove(subdivision="eighth", timing_offsets=(), velocity_weights=(1.0,))


def test_groove_rejects_out_of_range_offsets() -> None:
    with pytest.raises(ValueError, match="timing_offsets must all be in the range"):
        Groove(subdivision="eighth", timing_offsets=(0.0, 1.0), velocity_weights=(1.0,))
    with pytest.raises(ValueError, match="timing_offsets must all be in the range"):
        Groove(subdivision="eighth", timing_offsets=(-1.0,), velocity_weights=(1.0,))


def test_groove_rejects_empty_velocity_weights() -> None:
    with pytest.raises(ValueError, match="velocity_weights must be non-empty"):
        Groove(subdivision="eighth", timing_offsets=(0.0,), velocity_weights=())


def test_groove_rejects_non_positive_velocity_weights() -> None:
    with pytest.raises(ValueError, match="velocity_weights must all be positive"):
        Groove(subdivision="eighth", timing_offsets=(0.0,), velocity_weights=(0.0,))
    with pytest.raises(ValueError, match="velocity_weights must all be positive"):
        Groove(subdivision="eighth", timing_offsets=(0.0,), velocity_weights=(-0.5,))


def test_groove_rejects_bad_subdivision() -> None:
    with pytest.raises(ValueError, match="subdivision must be"):
        Groove(subdivision="quarter", timing_offsets=(0.0,), velocity_weights=(1.0,))  # type: ignore[arg-type]


# --- Groove accessors ---


def test_groove_timing_offset_at_cycles() -> None:
    groove = Groove(
        subdivision="sixteenth",
        timing_offsets=(0.0, 0.1, 0.2),
        velocity_weights=(1.0,),
    )
    assert groove.timing_offset_at(0) == pytest.approx(0.0)
    assert groove.timing_offset_at(1) == pytest.approx(0.1)
    assert groove.timing_offset_at(2) == pytest.approx(0.2)
    assert groove.timing_offset_at(3) == pytest.approx(0.0)
    assert groove.timing_offset_at(4) == pytest.approx(0.1)


def test_groove_velocity_weight_at_cycles() -> None:
    groove = Groove(
        subdivision="eighth",
        timing_offsets=(0.0,),
        velocity_weights=(1.0, 0.6),
    )
    assert groove.velocity_weight_at(0) == pytest.approx(1.0)
    assert groove.velocity_weight_at(1) == pytest.approx(0.6)
    assert groove.velocity_weight_at(2) == pytest.approx(1.0)


def test_groove_step_size_beats() -> None:
    assert Groove.eighths_swing().step_size_beats == pytest.approx(0.5)
    assert Groove.sixteenths_swing().step_size_beats == pytest.approx(0.25)


# --- Groove presets ---


def test_groove_presets_construct_without_error() -> None:
    presets = [
        Groove.eighths_swing(),
        Groove.sixteenths_swing(),
        Groove.mpc_tight(),
        Groove.dilla_lazy(),
        Groove.motown_pocket(),
        Groove.bossa(),
        Groove.tr808_swing(),
    ]
    for preset in presets:
        assert preset.name
        assert preset.timing_offsets
        assert preset.velocity_weights


def test_timeline_with_named_groove_presets() -> None:
    for groove in (Groove.mpc_tight(), Groove.dilla_lazy(), Groove.motown_pocket()):
        timeline = Timeline(bpm=120.0, groove=groove)
        assert timeline.position(B(0.0)) == pytest.approx(0.0)
        pos = timeline.position(B(4.0))
        assert pos > 0


# --- tuplet ---


def test_tuplet_quintuplet_quarter() -> None:
    result = tuplet(5, 4, Q)
    assert result.beats == pytest.approx(4.0 / 5.0)


def test_tuplet_septuplet_eighth() -> None:
    result = tuplet(7, 4, E)
    assert result.beats == pytest.approx(0.5 * 4.0 / 7.0)


def test_tuplet_triplet_equivalence() -> None:
    result = tuplet(3, 2, Q)
    expected = triplet(Q)
    assert result.beats == pytest.approx(expected.beats)


def test_tuplet_returns_beatspan() -> None:
    from code_musics.meter import BeatSpan

    result = tuplet(5, 4, Q)
    assert isinstance(result, BeatSpan)


def test_tuplet_rejects_non_positive_counts() -> None:
    with pytest.raises(ValueError, match="tuplet counts must be positive"):
        tuplet(0, 4, Q)
    with pytest.raises(ValueError, match="tuplet counts must be positive"):
        tuplet(5, 0, Q)


def test_tuplet_works_with_sixteenth() -> None:
    result = tuplet(5, 4, S)
    assert result.beats == pytest.approx(0.25 * 4.0 / 5.0)
