"""Musical timeline helper tests."""

from __future__ import annotations

import pytest

from code_musics.meter import B, E, M, Q, Timeline, dotted, triplet


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
