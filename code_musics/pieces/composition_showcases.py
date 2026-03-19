"""Composition-helper showcase pieces and articulation study."""

from __future__ import annotations

from code_musics.composition import (
    ArticulationSpec,
    ContextSectionSpec,
    RhythmCell,
    build_context_sections,
    canon,
    line,
    progression,
    recontextualize_phrase,
    sequence,
    voiced_ratio_chord,
)
from code_musics.pieces._shared import SOFT_REVERB_EFFECT, WARM_SATURATION_EFFECT
from code_musics.pieces.registry import PieceDefinition
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import EffectSpec, Score


def build_articulation_study_sketch() -> Score:
    """Short study contrasting clipped JI rhythm with ratio glides."""
    score = Score(f0=110.0, master_effects=[SOFT_REVERB_EFFECT])
    score.add_voice(
        "drone",
        synth_defaults={
            "engine": "additive",
            "preset": "drone",
            "attack": 0.6,
            "release": 2.0,
        },
    )
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "fm",
            "preset": "glass_lead",
            "attack": 0.03,
            "release": 0.35,
        },
        effects=[EffectSpec("chorus", {"preset": "juno_subtle", "mix": 0.30})],
    )

    score.add_note("drone", start=0.0, duration=16.0, partial=1.0, amp=0.18)
    score.add_note("drone", start=4.0, duration=10.0, partial=3 / 2, amp=0.12)

    phrase = line(
        tones=[6.0, 7.0, 8.0, 9.0, 8.0, 7.0],
        rhythm=RhythmCell(spans=(0.75, 0.75, 1.1, 0.65, 0.9, 1.4)),
        amp=0.30,
        articulation=ArticulationSpec(
            gate=(0.55, 0.55, 0.92, 0.5, 0.72, 1.08),
            accent_pattern=(1.0, 0.9, 1.18, 0.92, 1.05, 1.2),
            tail_breath=0.12,
        ),
        pitch_motion=(
            None,
            PitchMotionSpec.linear_bend(target_partial=8.0),
            None,
            PitchMotionSpec.ratio_glide(start_ratio=1.0, end_ratio=7 / 6),
            None,
            PitchMotionSpec.vibrato(depth_ratio=0.015, rate_hz=6.2),
        ),
    )
    score.add_phrase("lead", phrase, start=1.5)
    score.add_phrase("lead", phrase, start=8.0, partial_shift=2.0, amp_scale=0.92)
    return score


def build_composition_tools_showcase_score() -> Score:
    """Short piece built primarily from the composition-helper layer."""
    score = Score(
        f0=55.0,
        master_effects=[WARM_SATURATION_EFFECT, SOFT_REVERB_EFFECT],
    )
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "filtered_stack",
            "preset": "round_bass",
            "attack": 0.05,
            "release": 0.55,
        },
        pan=-0.08,
    )
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "filtered_stack",
            "preset": "warm_pad",
            "attack": 0.28,
            "release": 1.6,
        },
        pan=-0.18,
    )
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "fm",
            "preset": "glass_lead",
            "attack": 0.03,
            "release": 0.38,
        },
        pan=0.10,
    )
    score.add_voice(
        "answer_a",
        synth_defaults={
            "engine": "additive",
            "preset": "soft_pad",
            "attack": 0.05,
            "release": 0.8,
        },
        pan=0.08,
    )
    score.add_voice(
        "answer_b",
        synth_defaults={
            "engine": "additive",
            "preset": "soft_pad",
            "attack": 0.05,
            "release": 0.8,
        },
        pan=0.18,
    )
    score.add_voice(
        "answer_c",
        synth_defaults={
            "engine": "additive",
            "preset": "soft_pad",
            "attack": 0.05,
            "release": 0.8,
        },
        pan=0.28,
    )

    sections = build_context_sections(
        base_tonic=110.0,
        specs=(
            ContextSectionSpec(name="I", duration=4.0),
            ContextSectionSpec(name="V", duration=4.0, tonic_ratio=3 / 2),
            ContextSectionSpec(name="IV", duration=4.0, tonic_ratio=4 / 3),
            ContextSectionSpec(name="I_return", duration=4.0),
        ),
    )

    progression(
        score,
        "bass",
        sections=sections,
        chords=([1.0, 3 / 2], [1.0, 3 / 2], [1.0, 3 / 2], [1.0, 3 / 2]),
        pattern="pedal_upper",
        amp=(0.16, 0.15, 0.15, 0.18),
        voicing="close",
        low_hz=70.0,
        high_hz=220.0,
    )
    progression(
        score,
        "pad",
        sections=sections,
        chords=(
            [1.0, 5 / 4, 3 / 2],
            [1.0, 5 / 4, 3 / 2],
            [1.0, 5 / 4, 3 / 2],
            [1.0, 5 / 4, 3 / 2],
        ),
        pattern="block",
        amp=(0.08, 0.075, 0.07, 0.09),
        voicing="open",
        inversion=(1, 1, 1, 1),
        low_hz=240.0,
        high_hz=620.0,
        duration_scale=0.82,
    )

    motif = line(
        tones=[1.0, 5 / 4, 3 / 2, 5 / 4],
        rhythm=RhythmCell(spans=(0.75, 0.75, 1.0, 1.5)),
        amp_db=-14.5,
        articulation=ArticulationSpec(
            gate=(0.92, 0.88, 0.94, 0.78),
            accent_pattern=(1.0, 0.92, 1.08, 0.9),
            tail_breath=0.08,
        ),
    )

    sequence(
        score,
        "lead",
        motif,
        starts=(0.35, 0.35, 0.35, 0.35),
        amp_scales=(1.0, 0.92, 0.86, 1.04),
        sections=sections,
    )
    canon(
        score,
        voice_names=("answer_a", "answer_b", "answer_c"),
        phrase=motif,
        start=2.0,
        delays=(4.0, 6.0),
        amp_scales=(0.32, 0.28, 0.36),
        partial_shifts=(0.0, 0.0, 0.0),
        sections=(sections[0], sections[1], sections[3]),
    )
    return score


def build_composition_tools_consonant_score() -> Score:
    """Sparse consonant demonstration piece that exercises all composition helpers."""
    score = Score(f0=55.0, master_effects=[SOFT_REVERB_EFFECT])
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "filtered_stack",
            "preset": "round_bass",
            "attack": 0.04,
            "release": 0.45,
        },
        pan=-0.08,
    )
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "filtered_stack",
            "preset": "warm_pad",
            "attack": 0.22,
            "release": 1.4,
        },
        pan=-0.18,
    )
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "fm",
            "preset": "glass_lead",
            "attack": 0.025,
            "release": 0.32,
        },
        pan=0.04,
    )
    score.add_voice(
        "answer",
        synth_defaults={
            "engine": "additive",
            "preset": "soft_pad",
            "attack": 0.05,
            "release": 0.75,
        },
        pan=0.18,
    )
    score.add_voice(
        "bells",
        synth_defaults={
            "engine": "additive",
            "preset": "bright_pluck",
            "attack": 0.01,
            "release": 1.2,
        },
        pan=0.24,
    )

    sections = build_context_sections(
        base_tonic=110.0,
        specs=(
            ContextSectionSpec(name="I_a", duration=4.0),
            ContextSectionSpec(name="V", duration=4.0, tonic_ratio=3 / 2),
            ContextSectionSpec(name="IV", duration=4.0, tonic_ratio=4 / 3),
            ContextSectionSpec(name="I_b", duration=4.0),
        ),
    )

    progression(
        score,
        "bass",
        sections=sections,
        chords=([1.0], [1.0], [1.0], [1.0]),
        pattern="block",
        amp=(0.08, 0.08, 0.075, 0.09),
        voicing="close",
        low_hz=70.0,
        high_hz=150.0,
        duration_scale=0.55,
    )
    progression(
        score,
        "pad",
        sections=sections,
        chords=(
            [1.0, 5 / 4, 3 / 2],
            [1.0, 5 / 4, 3 / 2],
            [1.0, 5 / 4, 3 / 2],
            [1.0, 5 / 4, 3 / 2],
        ),
        pattern="block",
        amp=(0.032, 0.03, 0.028, 0.036),
        voicing="close",
        inversion=(0, 0, 0, 0),
        low_hz=220.0,
        high_hz=420.0,
        duration_scale=0.52,
    )

    motif = line(
        tones=[1.0, 5 / 4, 3 / 2, 1.0],
        rhythm=RhythmCell(spans=(0.95, 0.95, 1.1, 1.5)),
        amp_db=-18.0,
        articulation=ArticulationSpec(
            gate=(0.9, 0.9, 0.95, 0.82),
            accent_pattern=(1.0, 0.94, 1.04, 1.02),
            tail_breath=0.08,
        ),
    )
    sequence(
        score,
        "lead",
        motif,
        starts=(0.55, 0.55, 0.55, 0.55),
        amp_scales=(1.0, 0.94, 0.88, 1.02),
        sections=sections,
    )
    canon(
        score,
        voice_names=("answer", "answer"),
        phrase=motif,
        start=2.6,
        delays=(8.0,),
        amp_scales=(0.18, 0.2),
        partial_shifts=(0.0, 0.0),
        sections=(sections[0], sections[2]),
    )

    lifted_phrase = recontextualize_phrase(motif, target_context=sections[3].context)
    score.add_phrase("bells", lifted_phrase, start=13.6, time_scale=0.7, amp_scale=0.34)

    for section in sections:
        bell_freqs = voiced_ratio_chord(
            [1.0, 3 / 2],
            context=section.context,
            voicing="open",
            inversion=0,
            low_hz=330.0,
            high_hz=700.0,
        )
        for index, freq in enumerate(bell_freqs):
            score.add_note(
                "bells",
                start=section.start + 0.12 + (index * 0.18),
                duration=1.25 - (index * 0.12),
                freq=freq,
                amp=0.02 - (index * 0.003),
            )

    return score


PIECES: dict[str, PieceDefinition] = {
    "sketch_articulation_study": PieceDefinition(
        name="sketch_articulation_study",
        output_name="20_sketch_articulation_study.wav",
        build_score=build_articulation_study_sketch,
    ),
    "composition_tools_showcase": PieceDefinition(
        name="composition_tools_showcase",
        output_name="21_composition_tools_showcase.wav",
        build_score=build_composition_tools_showcase_score,
    ),
    "composition_tools_consonant": PieceDefinition(
        name="composition_tools_consonant",
        output_name="22_composition_tools_consonant.wav",
        build_score=build_composition_tools_consonant_score,
    ),
}
