"""Harpsichord engine demo — a chorale-like invention in 7-limit JI.

Demonstrates register blending, velocity expression, spectral morphing,
sympathetic resonance, and xenharmonic partial ratios.

Tuning: 7-limit JI from A3 (f0 = 220 Hz).
Tempo: ~96 BPM (beat = 0.625 s).

Voice layout:
  upper  — baroque preset (clear front 8'), melodic line
  lower  — concert preset (front + back 8' chorus), harmonic foundation
  color  — glass preset (higher-prime partials), sustained harmonic tones

Harmonic language: primarily 5-limit consonance (thirds, fifths, octaves)
with 7-limit touches (7/4, 7/6) as passing color.
"""

from __future__ import annotations

from code_musics.composition import HarmonicContext, ratio_line
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Score

# ── Tempo and grid ──────────────────────────────────────────────────────

BPM: float = 96.0
BEAT: float = 60.0 / BPM
HALF: float = BEAT * 2
QUARTER: float = BEAT
EIGHTH: float = BEAT / 2
DOT_QUARTER: float = BEAT * 1.5
BAR: float = BEAT * 4

F0_HZ: float = 220.0
CONTEXT = HarmonicContext(tonic=F0_HZ, name="A3 7-limit")

# ── JI ratios (7-limit from A) ─────────────────────────────────────────
# Written as intervals from A so vertical harmony is easy to reason about.

R1 = 1 / 1  # A
R9_8 = 9 / 8  # B
R6_5 = 6 / 5  # C (just minor third)
R5_4 = 5 / 4  # C#
R4_3 = 4 / 3  # D
R3_2 = 3 / 2  # E
R5_3 = 5 / 3  # F#
R7_4 = 7 / 4  # G↓7 (septimal seventh — color)
R15_8 = 15 / 8  # G#
R2 = 2 / 1  # A'


def build_study_harpsichord_score() -> Score:
    score = Score(
        f0_hz=F0_HZ,
        sample_rate=44_100,
        master_effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 55, "slope_db_per_oct": 12},
                    ]
                },
            ),
        ],
    )

    # ── Upper voice — baroque, clear and articulate ─────────────────
    score.add_voice(
        "upper",
        synth_defaults={"engine": "harpsichord", "preset": "baroque"},
        normalize_lufs=-21.0,
        pan=-0.2,
        sympathetic_amount=0.15,
        sympathetic_decay_s=1.8,
    )

    # ── Lower voice — concert, richer with back 8' chorus ──────────
    score.add_voice(
        "lower",
        synth_defaults={"engine": "harpsichord", "preset": "concert"},
        normalize_lufs=-21.0,
        pan=0.2,
        sympathetic_amount=0.15,
        sympathetic_decay_s=1.8,
    )

    # ── Color voice — glass partials for sustained harmonic shimmer ─
    score.add_voice(
        "color",
        synth_defaults={
            "engine": "harpsichord",
            "preset": "glass",
            "morph_time": 0.6,
        },
        normalize_lufs=-28.0,
        pan=0.0,
        sympathetic_amount=0.2,
        sympathetic_decay_s=2.5,
    )

    t = 0.0

    # ================================================================
    # Section A: Opening — voices in consonant thirds and fifths
    # Vertical intervals labeled at each beat alignment.
    # ================================================================

    # Bar 1-2: A pedal in bass, upper descends through A major chord
    #   U: A'(H)       E(H)         C#(Q)  D(Q)   E(H)
    #   L: A(H)        A(H)         A(Q)   A(Q)   C#(H)
    #   ↕: 8ve         5th          M3     4th    M3
    score.add_phrase(
        "upper",
        ratio_line(
            [R2, R3_2, R5_4, R4_3, R3_2],
            [HALF, HALF, QUARTER, QUARTER, HALF],
            context=CONTEXT,
            amp_db=-6.0,
        ),
        start=t,
    )
    score.add_phrase(
        "lower",
        ratio_line(
            [R1 / 2, R1 / 2, R1 / 2, R1 / 2, R5_4 / 2],
            [HALF, HALF, QUARTER, QUARTER, HALF],
            context=CONTEXT,
            amp_db=-8.0,
        ),
        start=t,
    )
    t += BAR * 2

    # Bar 3-4: Rising bass meets descending melody
    #   U: F#(H)       E(Q)  D(Q)   C#(H+Q)       A(Q)
    #   L: A(H)        C#(Q) D(Q)   E(H+Q)        A(H)
    #   ↕: M6          M3    8ve    m6→            8ve
    score.add_phrase(
        "upper",
        ratio_line(
            [R5_3, R3_2, R4_3, R5_4, R1],
            [HALF, QUARTER, QUARTER, DOT_QUARTER, QUARTER + HALF],
            context=CONTEXT,
            amp_db=-6.0,
        ),
        start=t,
    )
    score.add_phrase(
        "lower",
        ratio_line(
            [R1 / 2, R5_4 / 2, R4_3 / 2, R3_2 / 2, R1 / 2],
            [HALF, QUARTER, QUARTER, DOT_QUARTER, QUARTER + HALF],
            context=CONTEXT,
            amp_db=-8.0,
        ),
        start=t,
    )
    t += BAR * 2

    # ================================================================
    # Section B: Development — richer harmony, 7-limit color enters
    # ================================================================

    # Bar 5-6: Upper echoes lower's rising line; color voice adds 7/4
    #   U: E(Q)  F#(Q)   A'(H)       G#(Q)  E(Q)   C#(H)
    #   L: A(Q)  D(Q)    E(H)        E(Q)   C#(Q)  A(H)
    #   ↕: 5th   M3      4th→8ve     M3     M3     M3
    score.add_phrase(
        "upper",
        ratio_line(
            [R3_2, R5_3, R2, R15_8, R3_2, R5_4],
            [QUARTER, QUARTER, HALF, QUARTER, QUARTER, HALF],
            context=CONTEXT,
            amp_db=-5.0,
        ),
        start=t,
    )
    score.add_phrase(
        "lower",
        ratio_line(
            [R1 / 2, R4_3 / 2, R3_2 / 2, R3_2 / 2, R5_4 / 2, R1 / 2],
            [QUARTER, QUARTER, HALF, QUARTER, QUARTER, HALF],
            context=CONTEXT,
            amp_db=-7.0,
        ),
        start=t,
    )
    # Color: septimal seventh — brief touch of the alien
    score.add_phrase(
        "color",
        ratio_line(
            [R7_4],
            [BAR * 1.5],
            context=CONTEXT,
            amp_db=-16.0,
        ),
        start=t,
    )
    t += BAR * 2

    # Bar 7-8: Call and response — lower imitates upper's motif
    #   U: D(Q)  E(Q)  F#(Q) E(Q)   hold A'(BAR)
    #   L:            rest           D(Q) E(Q) F#(Q) E(Q)  A(H)
    #   ↕: (solo)                    8ve  8ve  8ve   8ve   8ve
    score.add_phrase(
        "upper",
        ratio_line(
            [R4_3, R3_2, R5_3, R3_2, R2],
            [QUARTER, QUARTER, QUARTER, QUARTER, BAR],
            context=CONTEXT,
            amp_db=-5.0,
        ),
        start=t,
    )
    score.add_phrase(
        "lower",
        ratio_line(
            [R4_3 / 2, R3_2 / 2, R5_3 / 2, R3_2 / 2, R1 / 2],
            [QUARTER, QUARTER, QUARTER, QUARTER, BAR],
            context=CONTEXT,
            amp_db=-7.0,
        ),
        start=t + BAR,  # delayed entry — imitation
    )
    t += BAR * 2

    # ================================================================
    # Section C: Recapitulation — fuller texture, voices interlock
    # ================================================================

    # Bar 9-10: Both voices in parallel thirds — warm, consonant
    #   U: A'(Q)  G#(Q)  F#(Q)  E(Q)    D(Q)  C#(Q)  D(Q)  E(Q)
    #   L: F#(Q)  E(Q)   D(Q)   C#(Q)   A(Q)  A(Q)   A(Q)  C#(Q)
    #   ↕: M3     M3     M3     M3      4th   M3     4th   M3
    score.add_phrase(
        "upper",
        ratio_line(
            [R2, R15_8, R5_3, R3_2, R4_3, R5_4, R4_3, R3_2],
            [QUARTER] * 8,
            context=CONTEXT,
            amp_db=-5.0,
        ),
        start=t,
    )
    score.add_phrase(
        "lower",
        ratio_line(
            [R5_3 / 2, R3_2 / 2, R4_3 / 2, R5_4 / 2, R1 / 2, R1 / 2, R1 / 2, R5_4 / 2],
            [QUARTER] * 8,
            context=CONTEXT,
            amp_db=-7.0,
        ),
        start=t,
    )
    # Color: sustained fifth — simple, grounding
    score.add_phrase(
        "color",
        ratio_line(
            [R3_2],
            [BAR * 2],
            context=CONTEXT,
            amp_db=-18.0,
        ),
        start=t,
    )
    t += BAR * 2

    # ================================================================
    # Section D: Cadence — resolving to open A
    # ================================================================

    # Bar 11-12: V → I cadence with suspension
    #   U: G#(Q) F#(Q) E(H)        D(Q)  C#(Q)  A(BAR)
    #   L: E(Q)  D(Q)  C#(H)       A(Q)  E(Q)   A(BAR)
    #   ↕: M3    M3    M3          4th   m6→    8ve
    score.add_phrase(
        "upper",
        ratio_line(
            [R15_8, R5_3, R3_2, R4_3, R5_4, R2],
            [QUARTER, QUARTER, HALF, QUARTER, QUARTER, BAR],
            context=CONTEXT,
            amp_db=-6.0,
        ),
        start=t,
    )
    score.add_phrase(
        "lower",
        ratio_line(
            [R3_2 / 2, R4_3 / 2, R5_4 / 2, R1 / 2, R3_2 / 2, R1 / 2],
            [QUARTER, QUARTER, HALF, QUARTER, QUARTER, BAR],
            context=CONTEXT,
            amp_db=-8.0,
        ),
        start=t,
    )
    # Final color: 7/4 → A (septimal resolution)
    score.add_phrase(
        "color",
        ratio_line(
            [R7_4, R2],
            [BAR, BAR],
            context=CONTEXT,
            amp_db=-18.0,
        ),
        start=t,
    )

    return score


PIECES: dict[str, PieceDefinition] = {
    "study_harpsichord": PieceDefinition(
        name="study_harpsichord",
        output_name="study_harpsichord",
        build_score=build_study_harpsichord_score,
        study=True,
    ),
}
