"""Composition-helper tests."""

from __future__ import annotations

import pytest

from code_musics.composition import (
    ArticulationSpec,
    RhythmCell,
    legato,
    line,
    staccato,
    with_accent_pattern,
    with_tail_breath,
)
from code_musics.pitch_motion import PitchMotionSpec


def test_line_builds_expected_starts_durations_and_accents() -> None:
    phrase = line(
        tones=[4.0, 5.0, 6.0],
        rhythm=RhythmCell(spans=(0.5, 0.75, 1.0)),
        amp=0.4,
        articulation=ArticulationSpec(
            gate=(0.5, 1.0, 1.2),
            accent_pattern=(1.0, 0.8, 1.25),
            tail_breath=0.1,
        ),
    )

    assert [event.start for event in phrase.events] == [0.0, 0.5, 1.25]
    assert [round(event.duration, 4) for event in phrase.events] == [0.25, 0.75, 1.1]
    assert [round(event.amp, 4) for event in phrase.events] == [0.4, 0.32, 0.5]


def test_line_can_attach_pitch_motion_per_event() -> None:
    motions = (
        None,
        PitchMotionSpec.linear_bend(target_partial=6.0),
        PitchMotionSpec.ratio_glide(start_ratio=1.0, end_ratio=5 / 4),
    )
    phrase = line(
        tones=[4.0, 5.0, 6.0],
        rhythm=(0.5, 0.5, 0.5),
        pitch_motion=motions,
    )

    assert phrase.events[0].pitch_motion is None
    assert phrase.events[1].pitch_motion == motions[1]
    assert phrase.events[2].pitch_motion == motions[2]


def test_line_supports_absolute_frequency_mode() -> None:
    phrase = line(
        tones=[220.0, 247.5, 330.0],
        rhythm=(0.25, 0.5, 0.75),
        pitch_kind="freq",
        amp=0.2,
    )

    assert [event.freq for event in phrase.events] == [220.0, 247.5, 330.0]
    assert [event.partial for event in phrase.events] == [None, None, None]
    assert [event.start for event in phrase.events] == [0.0, 0.25, 0.75]


def test_articulation_scales_attack_and_release_when_defaults_exist() -> None:
    phrase = line(
        tones=[4.0, 5.0],
        rhythm=(1.0, 1.0),
        synth_defaults={"attack": 0.2, "release": 0.6},
        articulation=ArticulationSpec(attack_scale=0.5, release_scale=1.5),
    )

    assert phrase.events[0].synth == {
        "attack": 0.2,
        "release": 0.6,
        "attack_scale": 0.5,
        "release_scale": 1.5,
    }


def test_rhythm_cell_gates_and_articulation_gate_compose_multiplicatively() -> None:
    phrase = line(
        tones=[4.0, 5.0],
        rhythm=RhythmCell(spans=(1.0, 2.0), gates=(0.5, 1.25)),
        articulation=ArticulationSpec(gate=(0.5, 0.8)),
    )

    assert [event.duration for event in phrase.events] == pytest.approx([0.25, 2.0])


def test_staccato_and_legato_leave_starts_unchanged() -> None:
    phrase = line(tones=[4.0, 5.0], rhythm=(1.0, 1.0))

    clipped = staccato(phrase)
    connected = legato(phrase)

    assert [event.start for event in clipped.events] == [0.0, 1.0]
    assert [event.start for event in connected.events] == [0.0, 1.0]
    assert clipped.events[0].duration < phrase.events[0].duration
    assert connected.events[0].duration > phrase.events[0].duration


def test_phrase_transforms_do_not_mutate_composition_helpers() -> None:
    phrase = line(tones=[4.0, 5.0, 6.0], rhythm=(0.5, 0.75, 1.0), amp=0.3)
    accented = with_accent_pattern(phrase, (1.0, 0.7, 1.2))
    breathed = with_tail_breath(accented, 0.2)

    assert [event.amp for event in phrase.events] == [0.3, 0.3, 0.3]
    assert [round(event.amp, 3) for event in accented.events] == [0.3, 0.21, 0.36]
    assert breathed.events[-1].duration == pytest.approx(accented.events[-1].duration - 0.2)


def test_line_rejects_length_mismatch_and_non_positive_values() -> None:
    with pytest.raises(ValueError, match="same length"):
        line(tones=[4.0, 5.0], rhythm=(1.0,))

    with pytest.raises(ValueError, match="positive"):
        RhythmCell(spans=(1.0, 0.0))

    with pytest.raises(ValueError, match="positive duration"):
        line(
            tones=[4.0],
            rhythm=(0.2,),
            articulation=ArticulationSpec(gate=0.5, tail_breath=0.2),
        )
