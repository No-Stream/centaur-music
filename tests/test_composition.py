"""Composition-helper tests."""

from __future__ import annotations

import pytest

from code_musics.composition import (
    ArticulationSpec,
    ContextSectionSpec,
    HarmonicContext,
    RhythmCell,
    build_context_sections,
    canon,
    legato,
    line,
    place_ratio_chord,
    place_ratio_line,
    progression,
    ratio_line,
    recontextualize_phrase,
    resolve_ratios,
    sequence,
    staccato,
    voiced_ratio_chord,
    with_accent_pattern,
    with_synth_ramp,
    with_tail_breath,
)
from code_musics.pieces.sketches import (
    build_composition_tools_consonant_score,
    build_composition_tools_showcase_score,
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


def test_recontextualize_phrase_retargets_local_tonic() -> None:
    source_phrase = line(
        tones=[1.0, 5 / 4, 3 / 2],
        rhythm=(0.5, 0.5, 0.5),
        amp=0.2,
    )

    recontextualized = recontextualize_phrase(
        source_phrase,
        target_context=HarmonicContext(tonic=220.0),
    )

    assert [event.freq for event in recontextualized.events] == pytest.approx(
        [220.0, 275.0, 330.0]
    )
    assert [event.partial for event in recontextualized.events] == [None, None, None]


def test_recontextualize_phrase_supports_frequency_authored_source_and_validation() -> (
    None
):
    source_phrase = line(
        tones=[220.0, 275.0],
        rhythm=(0.5, 0.5),
        pitch_kind="freq",
        amp=0.2,
    )

    recontextualized = recontextualize_phrase(
        source_phrase,
        target_context=HarmonicContext(tonic=110.0),
        source_tonic=220.0,
    )

    assert [event.freq for event in recontextualized.events] == pytest.approx(
        [110.0, 137.5]
    )
    with pytest.raises(ValueError, match="source_tonic must be positive"):
        recontextualize_phrase(
            source_phrase,
            target_context=HarmonicContext(tonic=110.0),
            source_tonic=0.0,
        )


def test_sequence_places_repeated_entries_with_context_shift() -> None:
    score = Score(f0=55.0)
    phrase = line(tones=[1.0, 5 / 4], rhythm=(0.5, 0.5), amp=0.2)
    sections = build_context_sections(
        base_tonic=220.0,
        specs=(
            ContextSectionSpec(name="A", duration=1.0),
            ContextSectionSpec(name="B", duration=1.0, tonic_ratio=80 / 81),
        ),
    )

    placed = sequence(
        score,
        "lead",
        phrase,
        starts=(0.0, 0.25),
        amp_scales=(1.0, 0.5),
        sections=sections,
    )

    assert [note.start for note in placed[0]] == pytest.approx([0.0, 0.5])
    assert [note.start for note in placed[1]] == pytest.approx([1.25, 1.75])
    assert [note.freq for note in placed[1]] == pytest.approx(
        [220.0 * (80 / 81), 275.0 * (80 / 81)]
    )
    assert placed[1][0].amp == pytest.approx(0.1)


def test_sequence_applies_partial_shift_before_recontextualizing() -> None:
    score = Score(f0=55.0)
    phrase = line(tones=[1.0, 5 / 4], rhythm=(0.5, 0.5), amp=0.2)
    section = build_context_sections(
        base_tonic=220.0,
        specs=(ContextSectionSpec(name="A", duration=1.0),),
    )[0]

    placed = sequence(
        score,
        "lead",
        phrase,
        starts=(0.0,),
        partial_shifts=(1.0,),
        sections=(section,),
    )[0]

    assert [note.freq for note in placed] == pytest.approx([440.0, 495.0])


def test_sequence_supports_reverse_and_time_scale_without_sections() -> None:
    score = Score(f0=55.0)
    phrase = line(tones=[4.0, 5.0], rhythm=(0.5, 1.0), amp=0.2)

    placed = sequence(
        score,
        "lead",
        phrase,
        starts=(1.0,),
        time_scales=(2.0,),
        reverses=(True,),
        amp_scales=(0.5,),
    )[0]

    assert [note.start for note in placed] == pytest.approx([3.0, 1.0])
    assert [note.duration for note in placed] == pytest.approx([1.0, 2.0])
    assert [note.amp for note in placed] == pytest.approx([0.1, 0.1])


def test_sequence_rejects_invalid_lengths_and_negative_start() -> None:
    score = Score(f0=55.0)
    phrase = line(tones=[4.0], rhythm=(0.5,), amp=0.2)

    with pytest.raises(ValueError, match="starts must not be empty"):
        sequence(score, "lead", phrase, starts=())
    with pytest.raises(ValueError, match="starts must be non-negative"):
        sequence(score, "lead", phrase, starts=(-0.1,))
    with pytest.raises(ValueError, match="sections length must match starts"):
        sequence(
            score,
            "lead",
            phrase,
            starts=(0.0,),
            sections=build_context_sections(
                base_tonic=220.0,
                specs=(
                    ContextSectionSpec(name="A", duration=1.0),
                    ContextSectionSpec(name="B", duration=1.0),
                ),
            ),
        )


def test_canon_places_delayed_entries_across_voices() -> None:
    score = Score(f0=55.0)
    phrase = line(tones=[4.0, 5.0], rhythm=(0.5, 0.5), amp=0.2)

    placed = canon(
        score,
        voice_names=("lead", "answer"),
        phrase=phrase,
        start=0.0,
        delays=(0.75,),
        amp_scales=(1.0, 0.6),
        partial_shifts=(0.0, 1.0),
    )

    assert [note.start for note in placed["lead"]] == pytest.approx([0.0, 0.5])
    assert [note.start for note in placed["answer"]] == pytest.approx([0.75, 1.25])
    assert [note.partial for note in placed["answer"]] == pytest.approx([5.0, 6.0])
    assert placed["answer"][0].amp == pytest.approx(0.12)


def test_canon_supports_scalar_delay_and_rejects_invalid_inputs() -> None:
    score = Score(f0=55.0)
    phrase = line(tones=[4.0], rhythm=(0.5,), amp=0.2)

    placed = canon(
        score,
        voice_names=("a", "b", "c"),
        phrase=phrase,
        start=0.5,
        delays=1.0,
    )

    assert [placed["a"][0].start, placed["b"][0].start, placed["c"][0].start] == (
        pytest.approx([0.5, 1.5, 2.5])
    )
    with pytest.raises(ValueError, match="voice_names must not be empty"):
        canon(score, voice_names=(), phrase=phrase, start=0.0, delays=1.0)
    with pytest.raises(ValueError, match="start must be non-negative"):
        canon(score, voice_names=("a",), phrase=phrase, start=-0.1, delays=1.0)
    with pytest.raises(ValueError, match="delays must have length"):
        canon(
            score,
            voice_names=("a", "b"),
            phrase=phrase,
            start=0.0,
            delays=(1.0, 2.0),
        )
    with pytest.raises(ValueError, match="sections length must match voice_names"):
        canon(
            score,
            voice_names=("a", "b"),
            phrase=phrase,
            start=0.0,
            delays=(1.0,),
            sections=(),
        )


def test_voiced_ratio_chord_supports_voicing_inversion_and_register() -> None:
    freqs = voiced_ratio_chord(
        [1.0, 5 / 4, 3 / 2],
        context=HarmonicContext(tonic=110.0),
        voicing="open",
        inversion=1,
        low_hz=150.0,
        high_hz=500.0,
    )

    assert freqs == pytest.approx(sorted(freqs))
    assert min(freqs) >= 150.0
    assert max(freqs) < 500.0
    assert len(freqs) == 3


def test_voiced_ratio_chord_supports_spread_and_validation() -> None:
    freqs = voiced_ratio_chord(
        [1.0, 5 / 4, 3 / 2],
        context=HarmonicContext(tonic=110.0),
        voicing="spread",
    )

    assert freqs == pytest.approx([110.0, 275.0, 660.0])
    with pytest.raises(ValueError, match="ratios must not be empty"):
        voiced_ratio_chord([], context=HarmonicContext(tonic=110.0))
    with pytest.raises(ValueError, match="voicing must be"):
        voiced_ratio_chord(
            [1.0],
            context=HarmonicContext(tonic=110.0),
            voicing="cluster",
        )
    with pytest.raises(ValueError, match="inversion must be non-negative"):
        voiced_ratio_chord(
            [1.0],
            context=HarmonicContext(tonic=110.0),
            inversion=-1,
        )
    with pytest.raises(ValueError, match="low_hz must be less than high_hz"):
        voiced_ratio_chord(
            [1.0],
            context=HarmonicContext(tonic=110.0),
            low_hz=200.0,
            high_hz=100.0,
        )


def test_progression_places_block_and_arpeggio_patterns() -> None:
    score = Score(f0=55.0)
    sections = build_context_sections(
        base_tonic=110.0,
        specs=(
            ContextSectionSpec(name="I", duration=1.0),
            ContextSectionSpec(name="V", duration=1.2, tonic_ratio=3 / 2),
        ),
    )

    block_notes = progression(
        score,
        "pad",
        sections=sections,
        chords=([1.0, 5 / 4, 3 / 2], [1.0, 5 / 4, 3 / 2]),
        pattern="block",
        amp=(0.3, 0.3),
    )
    arp_notes = progression(
        score,
        "arp",
        sections=sections[:1],
        chords=([1.0, 5 / 4, 3 / 2],),
        pattern="arpeggio",
        amp=0.15,
    )

    assert len(block_notes) == 6
    assert [note.start for note in block_notes[:3]] == pytest.approx([0.0, 0.0, 0.0])
    assert [note.start for note in arp_notes] == pytest.approx([0.0, 1 / 3, 2 / 3])
    assert [round(note.duration, 4) for note in arp_notes] == pytest.approx(
        [round(1 / 3, 4), round(1 / 3, 4), round(1 / 3, 4)]
    )


def test_progression_supports_pedal_upper_pattern_and_validation() -> None:
    score = Score(f0=55.0)
    sections = build_context_sections(
        base_tonic=110.0,
        specs=(ContextSectionSpec(name="I", duration=2.0),),
    )

    notes = progression(
        score,
        "pad",
        sections=sections,
        chords=([1.0, 5 / 4, 3 / 2],),
        pattern="pedal_upper",
        amp=0.3,
        duration_scale=0.5,
    )

    assert len(notes) == 3
    assert notes[0].duration == pytest.approx(1.0)
    assert [note.freq for note in notes[1:]] == pytest.approx([137.5, 165.0])
    assert [note.duration for note in notes[1:]] == pytest.approx([0.82, 0.82])
    with pytest.raises(ValueError, match="sections must not be empty"):
        progression(score, "pad", sections=(), chords=(), pattern="block")
    with pytest.raises(ValueError, match="same length"):
        progression(
            score,
            "pad",
            sections=sections,
            chords=([1.0], [1.0]),
            pattern="block",
        )
    with pytest.raises(ValueError, match="duration_scale must be positive"):
        progression(
            score,
            "pad",
            sections=sections,
            chords=([1.0],),
            pattern="block",
            duration_scale=0.0,
        )
    with pytest.raises(ValueError, match="pattern must be"):
        progression(
            score,
            "pad",
            sections=sections,
            chords=([1.0],),
            pattern="strum",
        )


def test_composition_tools_showcase_piece_builds_and_renders() -> None:
    score = build_composition_tools_showcase_score()
    rendered = score.render()

    assert score.total_dur >= 16.0
    assert "lead" in score.voices
    assert "pad" in score.voices
    assert rendered.size > 0


def test_composition_tools_consonant_piece_builds_and_renders() -> None:
    score = build_composition_tools_consonant_score()
    rendered = score.render()

    assert score.total_dur >= 16.0
    assert "lead" in score.voices
    assert "answer" in score.voices
    assert "bells" in score.voices
    assert rendered.size > 0


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
