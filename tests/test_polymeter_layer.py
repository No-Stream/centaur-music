"""Tests for polymeter_layer and polymeter_alignment."""

from __future__ import annotations

import pytest

from code_musics.composition import (
    RhythmCell,
    line,
    polymeter_alignment,
    polymeter_layer,
)
from code_musics.score import NoteEvent, Phrase


def _single_cycle_phrase(cycle: float, onsets: int) -> Phrase:
    step = cycle / onsets
    return line(
        tones=[float(i + 1) for i in range(onsets)],
        rhythm=RhythmCell(spans=tuple(step for _ in range(onsets))),
    )


def test_polymeter_layer_tiles_cycle_over_total() -> None:
    phrase = _single_cycle_phrase(cycle=4.0, onsets=4)  # 4 evenly-spaced notes
    tiled = polymeter_layer(phrase, cycle=4.0, total=12.0)

    assert len(tiled.events) == 12
    starts = [event.start for event in tiled.events]
    expected = [i * 1.0 for i in range(12)]
    assert starts == pytest.approx(expected)


def test_polymeter_layer_phases_against_longer_cycle() -> None:
    """7 vs 16: two full cycles of 7 plus a partial 2-of-7 fit inside [0, 16)."""
    phrase = _single_cycle_phrase(cycle=7.0, onsets=7)
    tiled = polymeter_layer(phrase, cycle=7.0, total=16.0)

    starts = [event.start for event in tiled.events]
    assert starts == pytest.approx(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    )


def test_polymeter_layer_drops_starts_past_total() -> None:
    phrase = _single_cycle_phrase(cycle=4.0, onsets=4)
    tiled = polymeter_layer(phrase, cycle=4.0, total=6.0)

    # Notes at 4.0 and 5.0 fit; notes at 6.0, 7.0 in the second cycle fall at
    # or beyond total and are dropped.
    starts = sorted(event.start for event in tiled.events)
    assert starts == pytest.approx([0, 1, 2, 3, 4, 5])


def test_polymeter_layer_honours_start_offset() -> None:
    phrase = _single_cycle_phrase(cycle=4.0, onsets=2)
    tiled = polymeter_layer(phrase, cycle=4.0, total=10.0, start=2.0)

    starts = sorted(event.start for event in tiled.events)
    assert starts == pytest.approx([2.0, 4.0, 6.0, 8.0])


def test_polymeter_layer_preserves_durations_and_pitch() -> None:
    step = 0.5
    phrase = Phrase(
        events=(
            NoteEvent(start=0.0, duration=step * 0.8, partial=1.0, velocity=1.0),
            NoteEvent(start=step, duration=step * 0.8, partial=5 / 4, velocity=1.0),
        )
    )
    tiled = polymeter_layer(phrase, cycle=1.0, total=3.0)

    durations = {round(event.duration, 6) for event in tiled.events}
    assert durations == {round(step * 0.8, 6)}

    partials = [event.partial for event in tiled.events]
    assert partials == [1.0, 5 / 4, 1.0, 5 / 4, 1.0, 5 / 4]


def test_polymeter_layer_rejects_events_outside_cycle() -> None:
    phrase = Phrase(
        events=(NoteEvent(start=5.0, duration=0.1, partial=1.0),)  # past cycle
    )
    with pytest.raises(ValueError, match="outside one cycle"):
        polymeter_layer(phrase, cycle=4.0, total=12.0)


def test_polymeter_layer_rejects_bad_params() -> None:
    phrase = _single_cycle_phrase(cycle=4.0, onsets=4)
    with pytest.raises(ValueError):
        polymeter_layer(phrase, cycle=0.0, total=12.0)
    with pytest.raises(ValueError):
        polymeter_layer(phrase, cycle=4.0, total=-1.0)
    with pytest.raises(ValueError):
        polymeter_layer(phrase, cycle=4.0, total=12.0, start=-1.0)


def test_polymeter_layer_empty_phrase_returns_empty_phrase() -> None:
    tiled = polymeter_layer(Phrase(events=()), cycle=4.0, total=12.0)
    assert tiled.events == ()


def test_polymeter_alignment_basic_pairs() -> None:
    assert polymeter_alignment([7, 16]) == 112.0
    assert polymeter_alignment([3, 4]) == 12.0
    assert polymeter_alignment([16, 20]) == 80.0


def test_polymeter_alignment_four_way_polymeter() -> None:
    """The piece's actual polymeter stack — kick bar 16 vs hat 11 vs glitch 9 vs tick 7."""
    assert polymeter_alignment([16, 11, 9, 7]) == 11088.0


def test_polymeter_alignment_accepts_floats_close_to_integers() -> None:
    # Beat counts sometimes arrive as floats from upstream math.
    assert polymeter_alignment([7.0, 16.0000001]) == 112.0


def test_polymeter_alignment_rejects_non_integer_floats() -> None:
    with pytest.raises(ValueError, match="not close to a positive integer"):
        polymeter_alignment([7.3, 16.0])


def test_polymeter_alignment_rejects_empty_or_non_positive() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        polymeter_alignment([])
    with pytest.raises(ValueError, match="must be positive"):
        polymeter_alignment([7, 0])
    with pytest.raises(ValueError, match="must be positive"):
        polymeter_alignment([7, -3])
