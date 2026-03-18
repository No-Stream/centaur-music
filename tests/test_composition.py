"""Composition-helper tests."""

from __future__ import annotations

import pytest

from code_musics.composition import (
    ArticulationSpec,
    ContextSectionSpec,
    HarmonicContext,
    RhythmCell,
    build_context_sections,
    legato,
    line,
    place_ratio_chord,
    place_ratio_line,
    ratio_line,
    resolve_ratios,
    staccato,
    with_accent_pattern,
    with_synth_ramp,
    with_tail_breath,
)
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import Score


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


def test_harmonic_context_resolves_ratios_and_supports_drift() -> None:
    context = HarmonicContext(tonic=220.0, name="root")
    drifted = context.drifted(by_ratio=80 / 81, name="drifted")

    assert resolve_ratios(context, [1.0, 5 / 4, 3 / 2]) == pytest.approx(
        [220.0, 275.0, 330.0]
    )
    assert drifted.tonic == pytest.approx(220.0 * (80 / 81))
    assert drifted.name == "drifted"


def test_build_context_sections_places_windows_from_base_tonic() -> None:
    sections = build_context_sections(
        base_tonic=220.0,
        start=4.0,
        specs=(
            ContextSectionSpec(name="A", duration=2.5),
            ContextSectionSpec(name="B", duration=2.5, tonic_ratio=80 / 81),
        ),
    )

    assert [section.start for section in sections] == pytest.approx([4.0, 6.5])
    assert [section.duration for section in sections] == pytest.approx([2.5, 2.5])
    assert [section.context.tonic for section in sections] == pytest.approx(
        [220.0, 220.0 * (80 / 81)]
    )


def test_ratio_line_resolves_local_context_into_frequency_phrase() -> None:
    phrase = ratio_line(
        tones=[1.0, 5 / 4, 3 / 2],
        rhythm=(0.5, 0.5, 0.5),
        context=HarmonicContext(tonic=220.0),
        amp=0.2,
    )

    assert [event.freq for event in phrase.events] == pytest.approx(
        [220.0, 275.0, 330.0]
    )
    assert [event.partial for event in phrase.events] == [None, None, None]


def test_place_ratio_helpers_emit_shifted_concrete_notes() -> None:
    score = Score(f0=220.0)
    section_a, section_b = build_context_sections(
        base_tonic=220.0,
        specs=(
            ContextSectionSpec(name="A", duration=1.5),
            ContextSectionSpec(name="B", duration=1.5, tonic_ratio=80 / 81),
        ),
    )

    placed_a = place_ratio_line(
        score,
        "melody",
        section=section_a,
        tones=[1.0, 5 / 4],
        rhythm=(0.5, 0.5),
        amp=0.2,
    )
    placed_b = place_ratio_line(
        score,
        "melody",
        section=section_b,
        tones=[1.0, 5 / 4],
        rhythm=(0.5, 0.5),
        amp=0.2,
    )
    placed_chord = place_ratio_chord(
        score,
        "chord",
        section=section_b,
        ratios=[1.0, 3 / 2],
        duration=1.0,
        amp=[0.3, 0.15],
        gap=0.1,
    )

    assert [note.freq for note in placed_a] == pytest.approx([220.0, 275.0])
    assert [note.freq for note in placed_b] == pytest.approx(
        [220.0 * (80 / 81), 275.0 * (80 / 81)]
    )
    assert [note.start for note in placed_b] == pytest.approx([1.5, 2.0])
    assert [note.freq for note in placed_chord] == pytest.approx(
        [220.0 * (80 / 81), 330.0 * (80 / 81)]
    )
    assert [note.start for note in placed_chord] == pytest.approx([1.5, 1.6])


def test_context_drift_helpers_match_manual_ji_comma_drift_frequencies() -> None:
    drifted_section = build_context_sections(
        base_tonic=220.0,
        specs=(ContextSectionSpec(name="drift", duration=10.0, tonic_ratio=80 / 81),),
    )[0]

    phrase = place_ratio_line(
        Score(f0=220.0),
        "melody",
        section=drifted_section,
        tones=[2.0, 9 / 4, 5 / 2],
        rhythm=(0.375, 0.375, 0.375),
        amp=0.38,
    )

    manual_freqs = [220.0 * (1 / (81 / 80)) * ratio for ratio in [2.0, 9 / 4, 5 / 2]]
    assert [note.freq for note in phrase] == pytest.approx(manual_freqs)


def test_line_supports_amp_db() -> None:
    phrase = line(
        tones=[4.0, 5.0],
        rhythm=(0.5, 0.5),
        amp_db=-18.0,
        articulation=ArticulationSpec(accent_pattern=(1.0, 2.0)),
    )

    assert phrase.events[0].amp == pytest.approx(10 ** (-18.0 / 20.0))
    assert phrase.events[1].amp == pytest.approx((10 ** (-18.0 / 20.0)) * 2.0)


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
    assert breathed.events[-1].duration == pytest.approx(
        accented.events[-1].duration - 0.2
    )


def test_with_synth_ramp_interpolates_per_event_params() -> None:
    phrase = line(
        tones=[220.0, 247.5, 330.0],
        rhythm=(0.5, 0.5, 0.5),
        pitch_kind="freq",
        synth_defaults={"engine": "fm", "mod_index": 0.6},
    )

    ramped = with_synth_ramp(
        phrase,
        start={"mod_index": 0.6, "release": 0.25},
        end={"mod_index": 1.2, "release": 0.7},
    )

    assert ramped.events[0].synth is not None
    assert ramped.events[0].synth["mod_index"] == pytest.approx(0.6)
    assert ramped.events[1].synth["mod_index"] == pytest.approx(0.9)
    assert ramped.events[2].synth["mod_index"] == pytest.approx(1.2)
    assert ramped.events[2].synth["release"] == pytest.approx(0.7)


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

    with pytest.raises(ValueError, match="amp or amp_db"):
        line(tones=[4.0], rhythm=(1.0,), amp=0.5, amp_db=-6.0)
