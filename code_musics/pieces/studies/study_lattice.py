"""Study: harmonic lattice walker — drifting through JI prime-factor space.

Two contrasting walks on the 3-5-7 lattice:

  Section A — a slow, gentle walk biased toward the 3-axis (fifths/fourths)
  with moderate gravity.  Feels like wandering through diatonic territory
  with occasional septimal detours.

  Section B — a faster walk with heavier 7-axis weight and weaker gravity,
  letting the harmony drift further into unfamiliar septimal territory
  before gravity gently pulls it back.

A sustained drone anchors both sections.  The two walks overlap briefly
during the crossfade region.
"""

from __future__ import annotations

from code_musics.composition import HarmonicContext, RhythmCell, ratio_line, sequence
from code_musics.generative.lattice import LatticeWalker
from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    VelocityHumanizeSpec,
)
from code_musics.pieces._shared import SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Score, VoiceSend

F0_HZ = 110.0
CONTEXT = HarmonicContext(tonic=F0_HZ, name="A2")

SECTION_A_STEPS = 24
SECTION_B_STEPS = 32
NOTE_DUR_A = 0.55
NOTE_DUR_B = 0.35
CROSSFADE_OVERLAP = 2.0


def build_study_lattice_score() -> Score:
    score = Score(
        f0=F0_HZ,
        timing_humanize=TimingHumanizeSpec(preset="chamber"),
    )

    score.add_send_bus(
        "hall",
        effects=[
            EffectSpec(
                "reverb", {"room_size": 0.85, "damping": 0.55, "wet_level": 0.6}
            ),
        ],
    )

    # ── drone voice ──────────────────────────────────────────────────
    score.add_voice(
        "drone",
        synth_defaults={
            "engine": "additive",
            "env": {
                "attack_ms": 2000.0,
                "decay_ms": 500.0,
                "sustain_ratio": 0.85,
                "release_ms": 3000.0,
            },
            "params": {"harmonic_rolloff": 0.65, "n_harmonics": 8},
        },
        sends=[VoiceSend(target="hall", send_db=-6.0)],
        mix_db=-6.0,
    )

    section_a_dur = SECTION_A_STEPS * NOTE_DUR_A
    section_b_start = section_a_dur - CROSSFADE_OVERLAP
    section_b_dur = SECTION_B_STEPS * NOTE_DUR_B
    total_dur = section_b_start + section_b_dur + 3.0

    score.add_note("drone", start=0.0, duration=total_dur, partial=1.0, amp_db=-8.0)
    score.add_note(
        "drone", start=0.5, duration=total_dur - 0.5, partial=2.0, amp_db=-14.0
    )

    # ── walk A: gentle, diatonic-leaning ─────────────────────────────
    walker_a = LatticeWalker(
        axes=(3, 5, 7),
        step_weights={3: 0.55, 5: 0.30, 7: 0.15},
        gravity=0.35,
        max_distance=2,
        seed=17,
    )
    ratios_a = walker_a.walk(SECTION_A_STEPS)
    rhythm_a = RhythmCell(spans=(NOTE_DUR_A,) * SECTION_A_STEPS)
    phrase_a = ratio_line(
        ratios_a,
        rhythm_a,
        context=CONTEXT,
        amp_db=-4.0,
    )

    score.add_voice(
        "walk_a",
        synth_defaults={
            "engine": "filtered_stack",
            "preset": "warm_pad",
        },
        effects=[SOFT_REVERB_EFFECT],
        sends=[VoiceSend(target="hall", send_db=-10.0)],
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        mix_db=-2.0,
        pan=-0.2,
    )
    sequence(score, "walk_a", phrase_a, starts=[0.0])

    # ── walk B: septimal, more adventurous ───────────────────────────
    walker_b = LatticeWalker(
        axes=(3, 5, 7),
        step_weights={3: 0.25, 5: 0.25, 7: 0.50},
        gravity=0.15,
        max_distance=3,
        seed=53,
    )
    ratios_b = walker_b.walk(SECTION_B_STEPS)
    rhythm_b = RhythmCell(spans=(NOTE_DUR_B,) * SECTION_B_STEPS)
    phrase_b = ratio_line(
        ratios_b,
        rhythm_b,
        context=CONTEXT,
        amp_db=-3.0,
    )

    score.add_voice(
        "walk_b",
        synth_defaults={
            "engine": "fm",
            "preset": "bell",
            "params": {"mod_index": 1.4},
        },
        effects=[SOFT_REVERB_EFFECT],
        sends=[VoiceSend(target="hall", send_db=-8.0)],
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble"),
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        mix_db=-1.0,
        pan=0.2,
    )
    sequence(score, "walk_b", phrase_b, starts=[section_b_start])

    return score


PIECES: dict[str, PieceDefinition] = {
    "study_lattice": PieceDefinition(
        name="study_lattice",
        output_name="study_lattice",
        build_score=build_study_lattice_score,
        study=True,
    ),
}
