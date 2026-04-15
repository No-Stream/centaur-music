"""Composition-helper tests."""

from __future__ import annotations

import pytest

from code_musics.automation import AutomationSegment
from code_musics.composition import (
    ArticulationSpec,
    ContextSectionSpec,
    HarmonicContext,
    MeteredSectionSpec,
    RhythmCell,
    augment,
    bar_automation,
    build_context_sections,
    canon,
    concat,
    cross_rhythm,
    diminish,
    displace,
    echo,
    grid_canon,
    grid_line,
    grid_ratio_line,
    grid_sequence,
    legato,
    line,
    metered_sections,
    overlay,
    place_ratio_chord,
    place_ratio_line,
    polyrhythm,
    progression,
    ratio_line,
    recontextualize_phrase,
    resolve_ratios,
    rhythmic_retrograde,
    rotate,
    sequence,
    staccato,
    voiced_ratio_chord,
    with_accent_pattern,
    with_synth_ramp,
    with_tail_breath,
)
from code_musics.meter import B, E, Groove, M, Q, S, Timeline, dotted
from code_musics.pieces.composition_showcases import (
    build_composition_tools_consonant_score,
    build_composition_tools_showcase_score,
)
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import Phrase, Score


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


def test_grid_line_matches_seconds_based_phrase_timing() -> None:
    timeline = Timeline(bpm=120.0)

    phrase = grid_line(
        tones=[4.0, 5.0, 6.0],
        durations=[Q, E, dotted(Q)],
        timeline=timeline,
        amp=0.2,
    )

    assert [event.start for event in phrase.events] == pytest.approx([0.0, 0.5, 0.75])
    assert [event.duration for event in phrase.events] == pytest.approx(
        [0.5, 0.25, 0.75]
    )


def test_grid_ratio_line_resolves_context_with_beat_durations() -> None:
    phrase = grid_ratio_line(
        tones=[1.0, 5 / 4, 3 / 2],
        durations=[Q, Q, E],
        context=HarmonicContext(tonic=220.0),
        timeline=Timeline(bpm=120.0),
        amp=0.2,
    )

    assert [event.freq for event in phrase.events] == pytest.approx(
        [220.0, 275.0, 330.0]
    )
    assert [event.start for event in phrase.events] == pytest.approx([0.0, 0.5, 1.0])


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


def test_metered_sections_build_context_windows_from_bars() -> None:
    sections = metered_sections(
        timeline=Timeline(bpm=120.0),
        base_tonic=220.0,
        start=M(2),
        specs=(
            MeteredSectionSpec(name="A", bars=1.0),
            MeteredSectionSpec(name="B", bars=2.0, tonic_ratio=3 / 2),
        ),
    )

    assert [section.start for section in sections] == pytest.approx([2.0, 4.0])
    assert [section.duration for section in sections] == pytest.approx([2.0, 4.0])
    assert [section.context.tonic for section in sections] == pytest.approx(
        [220.0, 330.0]
    )


def test_bar_automation_builds_linear_segments_from_bar_points() -> None:
    timeline = Timeline(bpm=120.0, meter=(4, 4))

    automation = bar_automation(
        target="cutoff_hz",
        timeline=timeline,
        points=((1, 0.0, 600.0), (3, 0.0, 1200.0), (4, 2.0, 900.0)),
        clamp_min=400.0,
    )

    assert automation.default_value == pytest.approx(600.0)
    assert automation.clamp_min == pytest.approx(400.0)
    assert automation.target.name == "cutoff_hz"
    assert automation.segments == (
        AutomationSegment(
            start=0.0,
            end=4.0,
            shape="linear",
            start_value=600.0,
            end_value=1200.0,
        ),
        AutomationSegment(
            start=4.0,
            end=7.0,
            shape="linear",
            start_value=1200.0,
            end_value=900.0,
        ),
    )


def test_bar_automation_rejects_non_increasing_points() -> None:
    timeline = Timeline(bpm=120.0, meter=(4, 4))

    with pytest.raises(ValueError, match="strictly increasing"):
        bar_automation(
            target="cutoff_hz",
            timeline=timeline,
            points=((2, 0.0, 600.0), (2, 0.0, 900.0)),
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
    score = Score(f0_hz=220.0)
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


def test_place_ratio_chord_amp_db_and_velocity() -> None:
    score = Score(f0_hz=220.0)
    section = build_context_sections(
        base_tonic=220.0,
        specs=(ContextSectionSpec(name="a", duration=4.0, tonic_ratio=1.0),),
    )[0]

    # amp_db path
    notes_db = place_ratio_chord(
        score,
        "pad",
        section=section,
        ratios=[1.0, 5 / 4, 3 / 2],
        duration=2.0,
        amp_db=-12.0,
        velocity=0.7,
    )
    for note in notes_db:
        assert note.amp_db == -12.0
        assert note.amp == pytest.approx(10 ** (-12.0 / 20))
        assert note.velocity == pytest.approx(0.7)

    # amp path still works and velocity defaults to 1.0
    notes_amp = place_ratio_chord(
        score,
        "pad",
        section=section,
        ratios=[1.0, 3 / 2],
        duration=1.0,
        amp=0.5,
    )
    for note in notes_amp:
        assert note.amp == pytest.approx(0.5)
        assert note.amp_db is None
        assert note.velocity == pytest.approx(1.0)


def test_context_drift_helpers_match_manual_ji_comma_drift_frequencies() -> None:
    drifted_section = build_context_sections(
        base_tonic=220.0,
        specs=(ContextSectionSpec(name="drift", duration=10.0, tonic_ratio=80 / 81),),
    )[0]

    phrase = place_ratio_line(
        Score(f0_hz=220.0),
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


def test_line_cycles_shorter_rhythm_cells_across_longer_tone_sequences() -> None:
    phrase = line(
        tones=[4.0, 5.0, 6.0, 7.0, 8.0],
        rhythm=RhythmCell(spans=(0.5, 0.75), gates=(0.8, 1.2)),
        amp=0.25,
    )

    assert [event.start for event in phrase.events] == pytest.approx(
        [0.0, 0.5, 1.25, 1.75, 2.5]
    )
    assert [event.duration for event in phrase.events] == pytest.approx(
        [0.4, 0.9, 0.4, 0.9, 0.4]
    )


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


def test_echo_rejects_partial_shift_for_freq_phrases_and_supports_freq_scale() -> None:
    phrase = ratio_line(
        tones=[1.0, 5 / 4],
        rhythm=(0.5, 0.5),
        context=HarmonicContext(tonic=220.0),
        amp=0.2,
    )

    with pytest.raises(ValueError, match="partial_shift has no effect"):
        echo(phrase, delay=0.25, partial_shift=1.0)

    echoed = echo(phrase, delay=0.25, amp_scale=0.5, freq_scale=3 / 2)
    assert [event.start for event in echoed.events] == pytest.approx([0.25, 0.75])
    assert [event.freq for event in echoed.events] == pytest.approx([330.0, 412.5])
    assert [event.amp for event in echoed.events] == pytest.approx([0.1, 0.1])


def test_echo_rejects_mixed_phrase_when_both_pitch_shift_modes_are_used() -> None:
    mixed_phrase = overlay(
        line(tones=[4.0], rhythm=(0.5,), amp=0.2),
        ratio_line(
            tones=[1.0],
            rhythm=(0.5,),
            context=HarmonicContext(tonic=220.0),
            amp=0.2,
        ),
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        echo(
            mixed_phrase,
            delay=0.25,
            partial_shift=1.0,
            freq_scale=3 / 2,
        )


def test_concat_and_overlay_build_composite_phrases_before_score_placement() -> None:
    first = line(tones=[4.0, 5.0], rhythm=(0.5, 0.5), amp=0.2)
    second = line(tones=[6.0], rhythm=(0.75,), amp=0.25)

    concatenated = concat(first, second)
    layered = overlay(first, second, offset=0.25)

    assert concatenated.duration == pytest.approx(1.75)
    assert [event.start for event in concatenated.events] == pytest.approx(
        [0.0, 0.5, 1.0]
    )
    assert [event.start for event in layered.events] == pytest.approx([0.0, 0.25, 0.5])
    assert layered.duration == pytest.approx(1.0)


def test_with_synth_ramp_interpolates_per_event_params() -> None:
    phrase = line(
        tones=[220.0, 247.5, 330.0],
        rhythm=(0.5, 0.5, 0.5),
        pitch_kind="freq",
        synth_defaults={"engine": "fm", "mod_index": 0.6},
    )

    ramped = with_synth_ramp(
        phrase,
        start_params={"mod_index": 0.6, "release": 0.25},
        end_params={"mod_index": 1.2, "release": 0.7},
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
    score = Score(f0_hz=55.0)
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
    score = Score(f0_hz=55.0)
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
    score = Score(f0_hz=55.0)
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
    score = Score(f0_hz=55.0)
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
    score = Score(f0_hz=55.0)
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

    assert len(placed["lead"]) == 1
    assert [note.start for note in placed["lead"][0]] == pytest.approx([0.0, 0.5])
    assert [note.start for note in placed["answer"][0]] == pytest.approx([0.75, 1.25])
    assert [note.partial for note in placed["answer"][0]] == pytest.approx([5.0, 6.0])
    assert placed["answer"][0][0].amp == pytest.approx(0.12)


def test_canon_supports_repeats_and_scalar_delay_and_rejects_invalid_inputs() -> None:
    score = Score(f0_hz=55.0)
    phrase = line(tones=[4.0], rhythm=(0.5,), amp=0.2)

    placed = canon(
        score,
        voice_names=("a", "b", "c"),
        phrase=phrase,
        start=0.5,
        delays=1.0,
        repeats=2,
        repeat_gap=0.25,
    )

    assert [
        placed["a"][0][0].start,
        placed["b"][0][0].start,
        placed["c"][0][0].start,
    ] == (pytest.approx([0.5, 1.5, 2.5]))
    assert [
        placed["a"][1][0].start,
        placed["b"][1][0].start,
        placed["c"][1][0].start,
    ] == (pytest.approx([1.25, 2.25, 3.25]))
    with pytest.raises(ValueError, match="voice_names must not be empty"):
        canon(score, voice_names=(), phrase=phrase, start=0.0, delays=1.0)
    with pytest.raises(ValueError, match="start must be non-negative"):
        canon(score, voice_names=("a",), phrase=phrase, start=-0.1, delays=1.0)
    with pytest.raises(ValueError, match="repeats must be positive"):
        canon(
            score, voice_names=("a",), phrase=phrase, start=0.0, delays=1.0, repeats=0
        )
    with pytest.raises(ValueError, match="repeat_gap must be non-negative"):
        canon(
            score,
            voice_names=("a",),
            phrase=phrase,
            start=0.0,
            delays=1.0,
            repeat_gap=-0.1,
        )
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


def test_grid_sequence_places_entries_from_bar_and_beat_positions() -> None:
    score = Score(f0_hz=55.0)
    phrase = grid_line(
        tones=[4.0, 5.0],
        durations=[Q, Q],
        timeline=Timeline(bpm=120.0),
        amp=0.2,
    )

    placed = grid_sequence(
        score,
        "lead",
        phrase,
        timeline=Timeline(bpm=120.0),
        at=[M(1), B(6.0)],
    )

    assert [note.start for note in placed[0]] == pytest.approx([0.0, 0.5])
    assert [note.start for note in placed[1]] == pytest.approx([3.0, 3.5])


def test_grid_canon_uses_beat_delays_and_repeat_gap() -> None:
    timeline = Timeline(bpm=120.0)
    score = Score(f0_hz=55.0)
    phrase = grid_line(
        tones=[4.0],
        durations=[Q],
        timeline=timeline,
        amp=0.2,
    )

    placed = grid_canon(
        score,
        voice_names=("a", "b"),
        phrase=phrase,
        timeline=timeline,
        start=M(1),
        delays=[B(2.0)],
        repeats=2,
        repeat_gap=B(1.0),
    )

    assert [entry[0].start for entry in placed["a"]] == pytest.approx([0.0, 1.0])
    assert [entry[0].start for entry in placed["b"]] == pytest.approx([1.0, 2.0])


def test_grid_helpers_build_renderable_score() -> None:
    timeline = Timeline(bpm=96.0)
    score = Score(f0_hz=55.0)
    score.add_voice("lead", synth_defaults={"engine": "additive", "preset": "organ"})
    score.add_voice("answer", synth_defaults={"engine": "additive", "preset": "organ"})

    motif = grid_line(
        tones=[1.0, 5 / 4, 3 / 2, 5 / 4],
        durations=[Q, Q, Q, Q],
        timeline=timeline,
        amp=0.12,
    )

    grid_sequence(score, "lead", motif, timeline=timeline, at=[M(1), M(2)])
    grid_canon(
        score,
        voice_names=("answer",),
        phrase=motif,
        timeline=timeline,
        start=B(2.0),
        delays=B(0.0),
    )

    rendered = score.render()

    assert score.total_dur == pytest.approx(5.0)
    assert rendered.size > 0


def test_grid_line_supports_eighth_swing() -> None:
    timeline = Timeline(bpm=120.0, groove=Groove.eighths_swing(2.0 / 3.0))

    phrase = grid_line(
        tones=[4.0, 5.0, 6.0, 7.0],
        durations=[E, E, E, E],
        timeline=timeline,
        amp=0.2,
    )

    assert [event.start for event in phrase.events] == pytest.approx(
        [0.0, 1.0 / 3.0, 0.5, 5.0 / 6.0]
    )
    assert [event.duration for event in phrase.events] == pytest.approx(
        [1.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0, 1.0 / 6.0]
    )


def test_grid_line_supports_sixteenth_swing() -> None:
    timeline = Timeline(bpm=120.0, groove=Groove.sixteenths_swing(2.0 / 3.0))

    phrase = grid_line(
        tones=[4.0, 5.0, 6.0, 7.0],
        durations=[S, S, S, S],
        timeline=timeline,
        amp=0.2,
    )

    assert [event.start for event in phrase.events] == pytest.approx(
        [0.0, 1.0 / 6.0, 0.25, 5.0 / 12.0]
    )
    assert [event.duration for event in phrase.events] == pytest.approx(
        [1.0 / 6.0, 1.0 / 12.0, 1.0 / 6.0, 1.0 / 12.0]
    )


def test_grid_sequence_places_swung_offbeat_phrase_entries() -> None:
    timeline = Timeline(bpm=120.0, groove=Groove.eighths_swing(2.0 / 3.0))
    score = Score(f0_hz=55.0)
    phrase = grid_line(
        tones=[4.0, 5.0],
        durations=[E, E],
        timeline=timeline,
        amp=0.2,
    )

    placed = grid_sequence(
        score,
        "lead",
        phrase,
        timeline=timeline,
        at=[B(0.5)],
    )

    assert [note.start for note in placed[0]] == pytest.approx([1.0 / 3.0, 0.5])
    assert [note.duration for note in placed[0]] == pytest.approx(
        [1.0 / 6.0, 1.0 / 3.0]
    )


def test_grid_canon_uses_swung_grid_for_delays_and_repeats() -> None:
    timeline = Timeline(bpm=120.0, groove=Groove.eighths_swing(2.0 / 3.0))
    score = Score(f0_hz=55.0)
    phrase = grid_line(
        tones=[4.0],
        durations=[E],
        timeline=timeline,
        amp=0.2,
    )

    placed = grid_canon(
        score,
        voice_names=("a", "b"),
        phrase=phrase,
        timeline=timeline,
        start=B(0.5),
        delays=[E],
        repeats=2,
        repeat_gap=E,
    )

    assert [entry[0].start for entry in placed["a"]] == pytest.approx(
        [1.0 / 3.0, 5.0 / 6.0]
    )
    assert [entry[0].start for entry in placed["b"]] == pytest.approx([0.5, 1.0])


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


def test_voiced_ratio_chord_supports_drop_voicings() -> None:
    drop2 = voiced_ratio_chord(
        [1.0, 5 / 4, 3 / 2, 7 / 4],
        context=HarmonicContext(tonic=100.0),
        voicing="drop2",
    )
    drop3 = voiced_ratio_chord(
        [1.0, 5 / 4, 3 / 2, 7 / 4],
        context=HarmonicContext(tonic=100.0),
        voicing="drop3",
    )

    assert drop2 == pytest.approx([75.0, 100.0, 125.0, 175.0])
    assert drop3 == pytest.approx([62.5, 100.0, 150.0, 175.0])


def test_progression_places_block_and_arpeggio_patterns() -> None:
    score = Score(f0_hz=55.0)
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


def test_progression_supports_custom_arpeggio_orders() -> None:
    score = Score(f0_hz=55.0)
    sections = build_context_sections(
        base_tonic=110.0,
        specs=(ContextSectionSpec(name="I", duration=1.0),),
    )

    descending = progression(
        score,
        "arp_desc",
        sections=sections,
        chords=([1.0, 5 / 4, 3 / 2, 7 / 4],),
        pattern="arpeggio",
        arpeggio_order="descending",
    )
    inside_out = progression(
        score,
        "arp_inside",
        sections=sections,
        chords=([1.0, 5 / 4, 3 / 2, 7 / 4],),
        pattern="arpeggio",
        arpeggio_order="inside_out",
    )

    assert [note.freq for note in descending] == pytest.approx(
        [192.5, 165.0, 137.5, 110.0]
    )
    assert [note.freq for note in inside_out] == pytest.approx(
        [165.0, 137.5, 192.5, 110.0]
    )

    with pytest.raises(ValueError, match="custom arpeggio_order length must match"):
        progression(
            score,
            "arp_bad",
            sections=sections,
            chords=([1.0, 5 / 4, 3 / 2],),
            pattern="arpeggio",
            arpeggio_order=(0, 2),
        )


def test_progression_supports_pedal_upper_pattern_and_validation() -> None:
    score = Score(f0_hz=55.0)
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
    with pytest.raises(ValueError, match="same length or a shorter rhythm"):
        line(tones=[4.0, 5.0], rhythm=(1.0, 1.0, 1.0))

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


# --- augment / diminish ---


def test_augment_doubles_starts_and_durations() -> None:
    phrase = line(tones=[4.0, 5.0, 6.0], rhythm=(0.5, 0.75, 1.0), amp=0.3)
    stretched = augment(phrase, 2.0)

    assert [e.start for e in stretched.events] == pytest.approx([0.0, 1.0, 2.5])
    assert [e.duration for e in stretched.events] == pytest.approx([1.0, 1.5, 2.0])
    # Pitches unchanged
    assert [e.partial for e in stretched.events] == [4.0, 5.0, 6.0]


def test_diminish_halves_starts_and_durations() -> None:
    phrase = line(tones=[4.0, 5.0, 6.0], rhythm=(0.5, 0.75, 1.0), amp=0.3)
    compressed = diminish(phrase, 2.0)

    assert [e.start for e in compressed.events] == pytest.approx([0.0, 0.25, 0.625])
    assert [e.duration for e in compressed.events] == pytest.approx([0.25, 0.375, 0.5])


def test_augment_identity() -> None:
    phrase = line(tones=[4.0, 5.0], rhythm=(0.5, 1.0), amp=0.2)
    same = augment(phrase, 1.0)

    assert [e.start for e in same.events] == [e.start for e in phrase.events]
    assert [e.duration for e in same.events] == [e.duration for e in phrase.events]


def test_augment_empty_phrase() -> None:
    empty = Phrase(events=())
    assert augment(empty, 2.0).events == ()


def test_augment_rejects_non_positive_factor() -> None:
    phrase = line(tones=[4.0], rhythm=(1.0,))
    with pytest.raises(ValueError, match="factor must be positive"):
        augment(phrase, 0.0)
    with pytest.raises(ValueError, match="factor must be positive"):
        augment(phrase, -1.0)
    with pytest.raises(ValueError, match="factor must be positive"):
        diminish(phrase, 0.0)


def test_augment_scales_beat_timings() -> None:
    timeline = Timeline(bpm=120.0)
    phrase = grid_line(
        tones=[4.0, 5.0],
        durations=[Q, Q],
        timeline=timeline,
        amp=0.2,
    )
    stretched = augment(phrase, 2.0)

    assert stretched.beat_timings is not None
    assert [bt.start_beats for bt in stretched.beat_timings] == pytest.approx(
        [0.0, 2.0]
    )
    assert [bt.duration_beats for bt in stretched.beat_timings] == pytest.approx(
        [2.0, 2.0]
    )


# --- rhythmic_retrograde ---


def test_rhythmic_retrograde_swaps_durations_preserves_pitches() -> None:
    phrase = line(tones=[4.0, 5.0, 6.0], rhythm=(0.25, 0.5, 1.0), amp=0.3)
    retro = rhythmic_retrograde(phrase)

    # Pitches stay in original order
    assert [e.partial for e in retro.events] == [4.0, 5.0, 6.0]
    # Durations are reversed
    assert [e.duration for e in retro.events] == pytest.approx([1.0, 0.5, 0.25])


def test_rhythmic_retrograde_single_note_is_identity() -> None:
    phrase = line(tones=[4.0], rhythm=(1.0,), amp=0.2)
    retro = rhythmic_retrograde(phrase)

    assert retro.events[0].partial == 4.0
    assert retro.events[0].duration == pytest.approx(1.0)


def test_rhythmic_retrograde_empty_phrase() -> None:
    empty = Phrase(events=())
    assert rhythmic_retrograde(empty).events == ()


# --- displace ---


def test_displace_shifts_all_starts() -> None:
    phrase = line(tones=[4.0, 5.0, 6.0], rhythm=(0.5, 0.5, 0.5), amp=0.2)
    shifted = displace(phrase, 1.5)

    assert [e.start for e in shifted.events] == pytest.approx([1.5, 2.0, 2.5])
    # Durations unchanged
    assert [e.duration for e in shifted.events] == [e.duration for e in phrase.events]


def test_displace_negative_offset() -> None:
    # Start phrase at t=2.0 so negative offset still yields non-negative starts
    phrase = line(tones=[4.0, 5.0], rhythm=(1.0, 1.0), amp=0.2)
    shifted_forward = displace(phrase, 2.0)
    shifted_back = displace(shifted_forward, -0.5)

    assert [e.start for e in shifted_back.events] == pytest.approx([1.5, 2.5])


def test_displace_empty_phrase() -> None:
    empty = Phrase(events=())
    assert displace(empty, 1.0).events == ()


# --- rotate ---


def test_rotate_by_one_moves_first_event_to_end() -> None:
    phrase = line(tones=[4.0, 5.0, 6.0, 7.0], rhythm=(0.5, 0.5, 0.5, 0.5), amp=0.2)
    rotated = rotate(phrase, 1)

    # Pitches rotated: first event goes to end
    assert [e.partial for e in rotated.events] == [5.0, 6.0, 7.0, 4.0]
    # IOIs are preserved cyclically, total span unchanged
    total_original = phrase.duration
    total_rotated = rotated.duration
    assert total_rotated == pytest.approx(total_original)


def test_rotate_by_zero_is_identity() -> None:
    phrase = line(tones=[4.0, 5.0], rhythm=(0.5, 1.0), amp=0.2)
    rotated = rotate(phrase, 0)

    assert [e.partial for e in rotated.events] == [4.0, 5.0]
    assert [e.start for e in rotated.events] == [e.start for e in phrase.events]


def test_rotate_full_cycle_is_identity() -> None:
    phrase = line(tones=[4.0, 5.0, 6.0], rhythm=(0.5, 0.5, 0.5), amp=0.2)
    rotated = rotate(phrase, 3)

    assert [e.partial for e in rotated.events] == [4.0, 5.0, 6.0]
    assert [e.start for e in rotated.events] == pytest.approx(
        [e.start for e in phrase.events]
    )


def test_rotate_single_note_is_identity() -> None:
    phrase = line(tones=[4.0], rhythm=(1.0,), amp=0.2)
    rotated = rotate(phrase, 1)

    assert [e.partial for e in rotated.events] == [4.0]


# --- polyrhythm ---


def test_polyrhythm_produces_correct_divisions() -> None:
    r3, r4 = polyrhythm(3, 4, 1.0)

    assert len(r3.spans) == 3
    assert len(r4.spans) == 4
    assert sum(r3.spans) == pytest.approx(1.0)
    assert sum(r4.spans) == pytest.approx(1.0)
    # Each span is equal within its cell
    assert all(s == pytest.approx(1.0 / 3) for s in r3.spans)
    assert all(s == pytest.approx(0.25) for s in r4.spans)


def test_polyrhythm_rejects_non_positive_counts() -> None:
    with pytest.raises(ValueError, match="division counts must be positive"):
        polyrhythm(0, 4, 1.0)
    with pytest.raises(ValueError, match="division counts must be positive"):
        polyrhythm(3, -1, 1.0)


def test_polyrhythm_rejects_non_positive_span() -> None:
    with pytest.raises(ValueError, match="span must be positive"):
        polyrhythm(3, 4, 0.0)
    with pytest.raises(ValueError, match="span must be positive"):
        polyrhythm(3, 4, -1.0)


# --- cross_rhythm ---


def test_cross_rhythm_produces_correct_phrases() -> None:
    phrases = cross_rhythm(
        layers=[(3, [1.0, 5 / 4, 3 / 2]), (4, [2.0, 7 / 4, 3 / 2, 5 / 3])],
        span=2.0,
    )

    assert len(phrases) == 2
    # First layer: 3 divisions of 2.0s = 3 events
    assert len(phrases[0].events) == 3
    assert phrases[0].duration == pytest.approx(2.0)
    # Second layer: 4 divisions of 2.0s = 4 events
    assert len(phrases[1].events) == 4
    assert phrases[1].duration == pytest.approx(2.0)


def test_cross_rhythm_rejects_empty_layers() -> None:
    with pytest.raises(ValueError, match="layers must not be empty"):
        cross_rhythm(layers=[], span=1.0)


def test_cross_rhythm_rejects_non_positive_span() -> None:
    with pytest.raises(ValueError, match="span must be positive"):
        cross_rhythm(layers=[(3, [1.0])], span=0.0)


def test_cross_rhythm_rejects_non_positive_divisions() -> None:
    with pytest.raises(ValueError, match="division counts must be positive"):
        cross_rhythm(layers=[(0, [1.0])], span=1.0)


# --- groove velocity weights ---


def test_grid_line_applies_groove_velocity_weights() -> None:
    timeline = Timeline(bpm=120.0, groove=Groove.mpc_tight())

    phrase = grid_line(
        tones=[4.0, 5.0, 6.0, 7.0],
        durations=[S, S, S, S],
        timeline=timeline,
        amp=0.2,
    )

    expected_weights = (1.0, 0.65, 0.85, 0.55)
    for event, weight in zip(phrase.events, expected_weights, strict=True):
        assert event.velocity == pytest.approx(weight)


def test_grid_ratio_line_applies_groove_velocity_weights() -> None:
    timeline = Timeline(bpm=120.0, groove=Groove.mpc_tight())

    phrase = grid_ratio_line(
        tones=[1.0, 5 / 4, 3 / 2, 7 / 4],
        durations=[S, S, S, S],
        context=HarmonicContext(tonic=220.0),
        timeline=timeline,
        amp=0.2,
    )

    expected_weights = (1.0, 0.65, 0.85, 0.55)
    for event, weight in zip(phrase.events, expected_weights, strict=True):
        assert event.velocity == pytest.approx(weight)


# --- line() velocity parameter ---


def test_line_scalar_velocity() -> None:
    phrase = line(
        tones=[4.0, 5.0, 6.0],
        rhythm=(0.5, 0.5, 0.5),
        velocity=0.7,
    )

    for event in phrase.events:
        assert event.velocity == pytest.approx(0.7)


def test_line_per_note_velocity() -> None:
    phrase = line(
        tones=[4.0, 5.0, 6.0],
        rhythm=(0.5, 0.5, 0.5),
        velocity=[0.5, 0.8, 1.0],
    )

    assert [event.velocity for event in phrase.events] == pytest.approx([0.5, 0.8, 1.0])


def test_line_wrong_length_velocity_raises() -> None:
    with pytest.raises(ValueError, match="velocity sequence must have the same length"):
        line(
            tones=[4.0, 5.0, 6.0],
            rhythm=(0.5, 0.5, 0.5),
            velocity=[0.5, 0.8],
        )
