"""Markov chain generative study: hand-authored transitions and phrase learning."""

from __future__ import annotations

from code_musics.composition import (
    ArticulationSpec,
    HarmonicContext,
    RhythmCell,
    ratio_line,
    sequence,
)
from code_musics.generative.markov import RatioMarkov
from code_musics.humanize import EnvelopeHumanizeSpec, TimingHumanizeSpec
from code_musics.pieces._shared import REVERB_EFFECT, SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Score

# -- Musical constants --

TONIC_HZ = 185.0  # roughly F#3, a warm middle register

# 5-limit and septimal ratios used throughout
UNISON = 1 / 1
MAJOR_SECOND = 9 / 8
MINOR_THIRD = 6 / 5
MAJOR_THIRD = 5 / 4
FOURTH = 4 / 3
FIFTH = 3 / 2
SEPTIMAL_MINOR_SEVENTH = 7 / 4
OCTAVE = 2 / 1


def _build_section_a(score: Score, ctx: HarmonicContext, start: float) -> float:
    """Section A: hand-crafted septimal transition table with stepwise bias."""

    # Transition table favoring stepwise motion (seconds, thirds) with occasional
    # leaps to the fifth or septimal seventh for color.
    chain = RatioMarkov.from_transitions(
        {
            UNISON: {MAJOR_SECOND: 3, MINOR_THIRD: 2, FIFTH: 1},
            MAJOR_SECOND: {UNISON: 2, MINOR_THIRD: 3, MAJOR_THIRD: 1},
            MINOR_THIRD: {MAJOR_SECOND: 2, MAJOR_THIRD: 2, FOURTH: 1},
            MAJOR_THIRD: {MINOR_THIRD: 2, FOURTH: 3, FIFTH: 1},
            FOURTH: {MAJOR_THIRD: 2, FIFTH: 2, MINOR_THIRD: 1},
            FIFTH: {FOURTH: 2, SEPTIMAL_MINOR_SEVENTH: 2, MAJOR_THIRD: 1},
            SEPTIMAL_MINOR_SEVENTH: {FIFTH: 3, OCTAVE: 2, FOURTH: 1},
            OCTAVE: {SEPTIMAL_MINOR_SEVENTH: 3, FIFTH: 2, MAJOR_THIRD: 1},
        }
    )

    note_dur = 0.52
    rhythm = RhythmCell(spans=(note_dur,))
    articulation = ArticulationSpec(gate=0.88, tail_breath=0.04)

    melody_a = chain.to_phrase(
        16,
        rhythm,
        seed=42,
        start=UNISON,
        context=ctx,
        articulation=articulation,
    )

    sequence(score, "melody_a", melody_a, starts=[start])

    section_dur = melody_a.duration
    # Drone pedal on the tonic underneath
    score.add_note(
        "drone",
        start=start,
        duration=section_dur + 1.0,
        freq=ctx.resolve_ratio(UNISON),
        amp_db=-12.0,
    )
    # Quiet fifth reinforcement
    score.add_note(
        "drone",
        start=start + 0.5,
        duration=section_dur,
        freq=ctx.resolve_ratio(FIFTH / 2),
        amp_db=-18.0,
    )

    return start + section_dur


def _build_section_b(score: Score, ctx: HarmonicContext, start: float) -> float:
    """Section B: learn from a seed phrase, then generate variations."""

    # A short memorable motif -- a rising-then-falling septimal gesture.
    seed_ratios = [
        UNISON,
        MAJOR_SECOND,
        MAJOR_THIRD,
        FIFTH,
        SEPTIMAL_MINOR_SEVENTH,
        FIFTH,
        FOURTH,
        MAJOR_THIRD,
        MAJOR_SECOND,
        UNISON,
        MINOR_THIRD,
        UNISON,
    ]
    seed_rhythm = RhythmCell(spans=(0.42,))
    seed_articulation = ArticulationSpec(gate=0.9, tail_breath=0.03)

    seed_phrase = ratio_line(
        seed_ratios,
        seed_rhythm,
        context=ctx,
        articulation=seed_articulation,
    )

    # Train a second-order chain so it captures two-note context from the motif.
    learned_chain = RatioMarkov.from_phrase(seed_phrase, order=2, context=ctx)

    # Place the seed phrase itself as an opening statement.
    gap = 0.3
    sequence(score, "melody_b", seed_phrase, starts=[start])
    cursor = start + seed_phrase.duration + gap

    # Generate two variations from different seeds -- same intervallic character,
    # different melodic paths.
    for variation_seed in (7, 31):
        variation = learned_chain.to_phrase(
            10,
            seed_rhythm,
            seed=variation_seed,
            context=ctx,
            articulation=seed_articulation,
        )
        sequence(score, "melody_b", variation, starts=[cursor])
        cursor += variation.duration + gap

    section_dur = cursor - start

    # Low drone on the minor third for a warmer harmonic bed
    score.add_note(
        "drone",
        start=start,
        duration=section_dur,
        freq=ctx.resolve_ratio(MINOR_THIRD / 2),
        amp_db=-14.0,
    )
    score.add_note(
        "drone",
        start=start + 0.3,
        duration=section_dur - 0.3,
        freq=ctx.resolve_ratio(UNISON),
        amp_db=-16.0,
    )

    return cursor


def build_study_markov_score() -> Score:
    """Build the Markov chain generative study."""

    score = Score(
        f0_hz=TONIC_HZ,
        timing_humanize=TimingHumanizeSpec(preset="chamber"),
        master_effects=[SOFT_REVERB_EFFECT],
    )

    ctx = HarmonicContext(tonic=TONIC_HZ)

    # -- Voices --

    # Section A melody: FM bell for a clear, ringing quality
    score.add_voice(
        "melody_a",
        synth_defaults={"engine": "fm", "preset": "glass_lead"},
        effects=[
            EffectSpec("delay", {"delay_seconds": 0.28, "feedback": 0.18, "mix": 0.12}),
        ],
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        mix_db=-3.0,
        pan=-0.2,
    )

    # Section B melody: filtered stack for a warmer, breathier feel
    score.add_voice(
        "melody_b",
        synth_defaults={"engine": "filtered_stack", "preset": "reed_lead"},
        effects=[
            EffectSpec("delay", {"delay_seconds": 0.22, "feedback": 0.15, "mix": 0.10}),
        ],
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        mix_db=-3.0,
        pan=0.2,
    )

    # Drone voice: additive soft pad, wide and present
    score.add_voice(
        "drone",
        synth_defaults={"engine": "additive", "preset": "soft_pad"},
        effects=[REVERB_EFFECT],
        velocity_humanize=None,
        mix_db=-2.0,
    )

    # -- Compose sections --

    section_a_end = _build_section_a(score, ctx, start=0.5)
    gap_between_sections = 1.2
    _build_section_b(score, ctx, start=section_a_end + gap_between_sections)

    return score


PIECES: dict[str, PieceDefinition] = {
    "study_markov": PieceDefinition(
        name="study_markov",
        output_name="study_markov",
        build_score=build_study_markov_score,
        study=True,
    ),
}
