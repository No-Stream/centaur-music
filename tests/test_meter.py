"""Musical timeline helper tests."""

from __future__ import annotations

import pytest

from code_musics.meter import B, E, M, Q, SwingSpec, Timeline, dotted, triplet


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
    with pytest.raises(ValueError, match="offbeat_position"):
        SwingSpec.eighths(1.0)


def test_timeline_supports_eighth_swing_positions_and_round_trip_locate() -> None:
    timeline = Timeline(bpm=120.0, swing=SwingSpec.eighths(2.0 / 3.0))

    assert timeline.position(B(0.5)) == pytest.approx(1.0 / 3.0)
    assert timeline.position(B(1.0)) == pytest.approx(0.5)
    assert timeline.duration(E) == pytest.approx(0.25)

    offbeat_location = timeline.locate(1.0 / 3.0)
    assert offbeat_location.absolute_beats == pytest.approx(0.5)
    assert offbeat_location.bar == 1
    assert offbeat_location.beat_within_bar == pytest.approx(0.5)


def test_timeline_supports_sixteenth_swing_positions() -> None:
    timeline = Timeline(bpm=120.0, swing=SwingSpec.sixteenths(2.0 / 3.0))

    assert timeline.position(B(0.25)) == pytest.approx(1.0 / 6.0)
    assert timeline.position(B(0.5)) == pytest.approx(0.25)
    assert timeline.position(B(0.75)) == pytest.approx(5.0 / 12.0)


def test_timeline_accepts_explicit_straight_swing_position() -> None:
    timeline = Timeline(bpm=120.0, swing=SwingSpec.eighths(0.5))

    assert timeline.position(B(0.5)) == pytest.approx(0.25)
    assert timeline.locate(0.25).absolute_beats == pytest.approx(0.5)
