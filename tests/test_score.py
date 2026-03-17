"""Score abstraction tests."""

from pathlib import Path

import numpy as np

from code_musics.render import render_piece
from code_musics.score import NoteEvent, Phrase, Score


def test_total_duration_is_derived_from_note_endpoints() -> None:
    score = Score(f0=55.0)
    score.add_note("a", start=1.0, duration=2.5, partial=4)
    score.add_note("b", start=0.5, duration=5.0, partial=6)

    assert score.total_dur == 5.5


def test_phrase_and_direct_note_have_matching_timing() -> None:
    score = Score(f0=55.0)
    phrase = Phrase(events=(NoteEvent(start=0.0, duration=1.2, partial=5, amp=0.4),))

    placed = score.add_phrase("lead", phrase, start=3.0)
    direct = score.add_note("lead", start=3.0, duration=1.2, partial=5, amp=0.4)

    assert placed[0].start == direct.start
    assert placed[0].duration == direct.duration
    assert placed[0].partial == direct.partial
    assert placed[0].amp == direct.amp


def test_phrase_transforms_do_not_mutate_original() -> None:
    phrase = Phrase.from_partials([4, 5, 6], note_dur=1.0, step=0.8, amp=0.5)
    original_partials = [event.partial for event in phrase.events]

    transformed = phrase.transformed(start=10.0, partial_shift=2.0, amp_scale=0.5, reverse=True)

    assert [event.partial for event in phrase.events] == original_partials
    assert [event.partial for event in transformed] == [6.0, 7.0, 8.0]
    assert transformed[0].start > 10.0


def test_render_overlapping_voices_returns_audio() -> None:
    score = Score(f0=55.0)
    score.add_note("a", start=0.0, duration=1.0, partial=4, amp=0.3)
    score.add_note("b", start=0.5, duration=1.0, partial=5, amp=0.3)

    audio = score.render()

    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert len(audio) == int(1.5 * score.sample_rate)
    assert np.max(np.abs(audio)) > 0


def test_plot_piano_roll_writes_file(tmp_path: Path) -> None:
    score = Score(f0=55.0)
    score.add_note("a", start=0.0, duration=1.0, partial=4, amp=0.3)

    output_path = tmp_path / "roll.png"
    figure, _ = score.plot_piano_roll(output_path)

    assert output_path.exists()
    figure.clf()


def test_render_piece_writes_audio_and_plot(tmp_path: Path) -> None:
    audio_path, plot_path = render_piece("chord_4567", output_dir=tmp_path, save_plot=True)

    assert audio_path.exists()
    assert plot_path is not None
    assert plot_path.exists()
