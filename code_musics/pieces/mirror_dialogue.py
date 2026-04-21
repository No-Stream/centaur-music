"""Mirror Dialogue — a conversation between otonal and utonal worlds.

Otonal (overtone-based, 4:5:6:7) = bright, warm, open.
Utonal (undertone-based, 1/(4:5:6:7)) = dark, hollow, introspective.

The piece is structured as a dialogue: an otonal phrase, answered by its
utonal mirror. Each answer slightly transforms the previous phrase. The
conversation accelerates — shorter phrases, more overlap — until the two
merge into a single texture where you can no longer tell them apart.

Structure:
  Phase 1 (0-18s)   — clear call-and-response with silence between
  Phase 2 (18-34s)  — gaps shrink, phrases begin to overlap
  Phase 3 (34-50s)  — simultaneous, interleaved, converging
  Coda    (50-56s)  — a single merged chord, both worlds at once

Voices:
  otonal  — polyblep saw, warm and bright (panned slightly left)
  utonal  — additive with darker partials (panned slightly right)
  drone   — additive sub-drone on 1/1, anchoring both worlds
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from code_musics.composition import ArticulationSpec, RhythmCell, line
from code_musics.humanize import VelocityHumanizeSpec
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Score

if TYPE_CHECKING:
    from code_musics.score import Phrase

F0 = 110.0  # A2

# ── Interval material ────────────────────────────────────────────────────────
# Otonal: ratios from the harmonic series above 1/1
# Utonal: the undertone mirror — each interval inverted through the octave
#   5/4 -> 8/5,  3/2 -> 4/3,  7/4 -> 8/7

# Melodic cells — each phrase is a short gesture through these intervals.
# The otonal phrases ascend through brightness; the utonal answers descend
# through darkness, mirroring the same intervallic motion in undertone space.

OTONAL_CELL_A = [1.0, 5 / 4, 3 / 2, 7 / 4]  # root -> M3 -> P5 -> H7
UTONAL_CELL_A = [
    1.0,
    8 / 7,
    4 / 3,
    8 / 5,
]  # root -> mirror of H7 -> mirror of P5 -> mirror of M3

OTONAL_CELL_B = [7 / 4, 3 / 2, 5 / 4, 1.0]  # descending otonal
UTONAL_CELL_B = [8 / 5, 4 / 3, 8 / 7, 1.0]  # descending utonal

OTONAL_CELL_C = [1.0, 3 / 2, 7 / 4, 2.0]  # root -> fifth -> seventh -> octave
UTONAL_CELL_C = [1.0, 4 / 3, 8 / 5, 2.0]  # utonal mirror

# Merged/converged cell: otonal and utonal ratios interleaved
MERGED_CELL = [1.0, 8 / 7, 5 / 4, 4 / 3, 3 / 2, 8 / 5, 7 / 4, 2.0]


def _otonal_phrase(
    tones: list[float],
    note_dur: float,
    *,
    gate: float = 0.82,
    amp_db: float = -10.0,
) -> Phrase:
    """Build a polyblep-voiced otonal phrase."""
    return line(
        tones=[F0 * r for r in tones],
        rhythm=RhythmCell(spans=tuple([note_dur] * len(tones)), gates=gate),
        pitch_kind="freq",
        amp_db=amp_db,
        articulation=ArticulationSpec(tail_breath=note_dur * 0.05),
    )


def _utonal_phrase(
    tones: list[float],
    note_dur: float,
    *,
    gate: float = 0.88,
    amp_db: float = -10.0,
) -> Phrase:
    """Build an additive-voiced utonal phrase."""
    return line(
        tones=[F0 * r for r in tones],
        rhythm=RhythmCell(spans=tuple([note_dur] * len(tones)), gates=gate),
        pitch_kind="freq",
        amp_db=amp_db,
        articulation=ArticulationSpec(tail_breath=note_dur * 0.05),
    )


def build_score() -> Score:
    """Build the Mirror Dialogue score."""
    score = Score(
        f0_hz=F0,
        master_effects=[
            EffectSpec(
                "reverb", {"room_size": 0.72, "damping": 0.35, "wet_level": 0.22}
            ),
            EffectSpec("delay", {"delay_seconds": 0.38, "feedback": 0.18, "mix": 0.12}),
        ],
    )

    # ── Voices ────────────────────────────────────────────────────────────────

    # Otonal: polyblep saw — warm, bright, slightly filtered
    score.add_voice(
        "otonal",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "saw",
            "cutoff_hz": 2200.0,
            "keytrack": 0.08,
            "resonance": 0.06,
            "filter_env_amount": 0.15,
            "filter_env_decay": 0.5,
            "attack": 0.04,
            "decay": 0.6,
            "sustain_level": 0.55,
            "release": 0.8,
        },
        pan=-0.3,
        mix_db=-1.0,
        velocity_humanize=VelocityHumanizeSpec(seed=7),
    )

    # Utonal: additive with darker, hollower timbre — emphasis on odd partials
    score.add_voice(
        "utonal",
        synth_defaults={
            "engine": "additive",
            "n_harmonics": 6,
            "harmonic_rolloff": 0.48,
            "odd_even_balance": 0.25,
            "brightness_tilt": -0.08,
            "attack": 0.06,
            "decay": 0.8,
            "sustain_level": 0.50,
            "release": 1.2,
        },
        pan=0.3,
        mix_db=-1.0,
        velocity_humanize=VelocityHumanizeSpec(seed=13),
    )

    # Drone: low additive, barely audible, felt more than heard
    score.add_voice(
        "drone",
        synth_defaults={
            "engine": "additive",
            "n_harmonics": 3,
            "harmonic_rolloff": 0.60,
            "attack": 3.0,
            "decay": 1.0,
            "sustain_level": 0.85,
            "release": 5.0,
        },
        pan=0.0,
        mix_db=-8.0,
        velocity_humanize=None,
    )

    # ── Drone ─────────────────────────────────────────────────────────────────
    # 1/1 root drone throughout, plus a quiet fifth that enters later
    score.add_note("drone", start=0.0, duration=58.0, freq=F0, amp_db=-18.0)
    score.add_note("drone", start=4.0, duration=52.0, freq=F0 * 3 / 2, amp_db=-24.0)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: Clear call-and-response (0-18s)
    # Otonal speaks, silence, utonal answers. Each pair uses different material.
    # ══════════════════════════════════════════════════════════════════════════

    t = 1.5  # leave a breath for the drone to establish

    # --- Pair 1: ascending otonal, ascending utonal mirror ---
    note_dur_1 = 0.75
    p1_otonal = _otonal_phrase(OTONAL_CELL_A, note_dur_1)
    score.add_phrase("otonal", p1_otonal, start=t)
    t += p1_otonal.duration + 1.0  # 1s silence

    p1_utonal = _utonal_phrase(UTONAL_CELL_A, note_dur_1, amp_db=-11.0)
    score.add_phrase("utonal", p1_utonal, start=t)
    t += p1_utonal.duration + 1.2

    # --- Pair 2: descending otonal, descending utonal mirror ---
    note_dur_2 = 0.65
    p2_otonal = _otonal_phrase(OTONAL_CELL_B, note_dur_2, amp_db=-9.0)
    score.add_phrase("otonal", p2_otonal, start=t)
    t += p2_otonal.duration + 0.8

    p2_utonal = _utonal_phrase(UTONAL_CELL_B, note_dur_2, amp_db=-10.0)
    score.add_phrase("utonal", p2_utonal, start=t)
    t += p2_utonal.duration + 1.0

    # --- Pair 3: wider otonal arc, utonal answers with its mirror ---
    note_dur_3 = 0.70
    p3_otonal = _otonal_phrase(OTONAL_CELL_C, note_dur_3, amp_db=-9.0)
    score.add_phrase("otonal", p3_otonal, start=t)
    t += p3_otonal.duration + 0.6

    p3_utonal = _utonal_phrase(UTONAL_CELL_C, note_dur_3, amp_db=-10.0)
    score.add_phrase("utonal", p3_utonal, start=t)
    t += p3_utonal.duration + 0.4

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: Gaps shrink, overlap begins (18-34s)
    # Phrases get shorter, silences compress, utonal starts before otonal ends.
    # ══════════════════════════════════════════════════════════════════════════

    # --- Pair 4: otonal cell A faster, utonal overlaps by 0.3s ---
    note_dur_4 = 0.55
    p4_otonal = _otonal_phrase(OTONAL_CELL_A, note_dur_4, amp_db=-9.0)
    score.add_phrase("otonal", p4_otonal, start=t)
    overlap_start = t + p4_otonal.duration - 0.3
    t += p4_otonal.duration + 0.15

    p4_utonal = _utonal_phrase(UTONAL_CELL_A, note_dur_4, amp_db=-10.0)
    score.add_phrase("utonal", p4_utonal, start=overlap_start)

    # --- Pair 5: descending, more overlap ---
    note_dur_5 = 0.50
    p5_otonal = _otonal_phrase(OTONAL_CELL_B, note_dur_5, amp_db=-8.5)
    score.add_phrase("otonal", p5_otonal, start=t)
    overlap_start = t + p5_otonal.duration * 0.5  # utonal enters halfway through
    t += p5_otonal.duration + 0.1

    p5_utonal = _utonal_phrase(UTONAL_CELL_B, note_dur_5, amp_db=-9.5)
    score.add_phrase("utonal", p5_utonal, start=overlap_start)

    # --- Pair 6: wide arc, nearly simultaneous ---
    note_dur_6 = 0.45
    p6_otonal = _otonal_phrase(OTONAL_CELL_C, note_dur_6, amp_db=-8.5)
    score.add_phrase("otonal", p6_otonal, start=t)
    # utonal starts just 0.2s after otonal
    p6_utonal = _utonal_phrase(UTONAL_CELL_C, note_dur_6, amp_db=-9.5)
    score.add_phrase("utonal", p6_utonal, start=t + 0.2)
    t += max(p6_otonal.duration, p6_utonal.duration + 0.2) + 0.15

    # --- Pair 7: rapid exchange ---
    note_dur_7 = 0.40
    p7_otonal = _otonal_phrase(OTONAL_CELL_A, note_dur_7, amp_db=-8.0, gate=0.75)
    p7_utonal = _utonal_phrase(UTONAL_CELL_A, note_dur_7, amp_db=-9.0, gate=0.80)
    score.add_phrase("otonal", p7_otonal, start=t)
    score.add_phrase("utonal", p7_utonal, start=t + 0.12)
    t += p7_otonal.duration + 0.1

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3: Simultaneous, interleaved, converging (34-50s)
    # Both voices play together. The otonal voice starts incorporating utonal
    # intervals; the utonal voice starts incorporating otonal intervals.
    # ══════════════════════════════════════════════════════════════════════════

    # --- Simultaneous pair with cross-pollinated material ---
    note_dur_8 = 0.38

    # Otonal borrows one utonal interval (4/3 instead of pure 5/4)
    otonal_contaminated_a = [1.0, 4 / 3, 3 / 2, 7 / 4]
    p8_otonal = _otonal_phrase(otonal_contaminated_a, note_dur_8, amp_db=-8.0)
    score.add_phrase("otonal", p8_otonal, start=t)

    # Utonal borrows one otonal interval (5/4 instead of 8/7)
    utonal_contaminated_a = [1.0, 5 / 4, 4 / 3, 8 / 5]
    p8_utonal = _utonal_phrase(utonal_contaminated_a, note_dur_8, amp_db=-8.5)
    score.add_phrase("utonal", p8_utonal, start=t + 0.08)
    t += max(p8_otonal.duration, p8_utonal.duration + 0.08) + 0.1

    # --- More contamination: both voices share more intervals ---
    note_dur_9 = 0.35
    otonal_contaminated_b = [1.0, 5 / 4, 4 / 3, 7 / 4]
    utonal_contaminated_b = [1.0, 8 / 7, 3 / 2, 8 / 5]
    p9_otonal = _otonal_phrase(otonal_contaminated_b, note_dur_9, amp_db=-7.5)
    p9_utonal = _utonal_phrase(utonal_contaminated_b, note_dur_9, amp_db=-8.0)
    score.add_phrase("otonal", p9_otonal, start=t)
    score.add_phrase("utonal", p9_utonal, start=t + 0.05)
    t += max(p9_otonal.duration, p9_utonal.duration + 0.05) + 0.08

    # --- Rapid interleaved notes, both voices on converging material ---
    note_dur_10 = 0.30
    # Now each voice plays a mix of otonal and utonal intervals
    converging_a = [1.0, 8 / 7, 5 / 4, 3 / 2, 7 / 4]
    converging_b = [1.0, 5 / 4, 4 / 3, 8 / 5, 2.0]
    p10_otonal = _otonal_phrase(converging_a, note_dur_10, amp_db=-7.5, gate=0.78)
    p10_utonal = _utonal_phrase(converging_b, note_dur_10, amp_db=-8.0, gate=0.82)
    score.add_phrase("otonal", p10_otonal, start=t)
    score.add_phrase("utonal", p10_utonal, start=t + note_dur_10 * 0.5)
    t += max(p10_otonal.duration, p10_utonal.duration + note_dur_10 * 0.5) + 0.08

    # --- Near-unison: both voices play the same merged scale ---
    note_dur_11 = 0.28
    p11_otonal = _otonal_phrase(MERGED_CELL, note_dur_11, amp_db=-8.0, gate=0.80)
    p11_utonal = _utonal_phrase(MERGED_CELL, note_dur_11, amp_db=-8.5, gate=0.84)
    score.add_phrase("otonal", p11_otonal, start=t)
    score.add_phrase("utonal", p11_utonal, start=t + 0.04)
    t += max(p11_otonal.duration, p11_utonal.duration + 0.04) + 0.06

    # --- One more pass through merged, even tighter ---
    note_dur_12 = 0.25
    p12_otonal = _otonal_phrase(MERGED_CELL, note_dur_12, amp_db=-8.5, gate=0.85)
    p12_utonal = _utonal_phrase(MERGED_CELL, note_dur_12, amp_db=-8.5, gate=0.85)
    score.add_phrase("otonal", p12_otonal, start=t)
    score.add_phrase("utonal", p12_utonal, start=t + 0.02)
    t += max(p12_otonal.duration, p12_utonal.duration + 0.02)

    # ══════════════════════════════════════════════════════════════════════════
    # CODA: A single merged chord — both worlds simultaneously (50-56s)
    # Full otonal+utonal tetrad sounding together, then fading into the drone.
    # ══════════════════════════════════════════════════════════════════════════

    coda_start = t + 0.5
    coda_dur = 5.5

    # Otonal chord: 1/1, 5/4, 3/2, 7/4 — voiced across the register
    for ratio, db in [(1.0, -12.0), (5 / 4, -13.0), (3 / 2, -13.0), (7 / 4, -14.0)]:
        score.add_note(
            "otonal",
            start=coda_start,
            duration=coda_dur,
            freq=F0 * ratio,
            amp_db=db,
        )

    # Utonal chord: 1/1, 8/7, 4/3, 8/5 — the dark mirror
    for ratio, db in [(1.0, -12.0), (8 / 7, -13.0), (4 / 3, -13.0), (8 / 5, -14.0)]:
        score.add_note(
            "utonal",
            start=coda_start,
            duration=coda_dur,
            freq=F0 * ratio,
            amp_db=db,
        )

    return score


PIECES: dict[str, PieceDefinition] = {
    "mirror_dialogue": PieceDefinition(
        name="mirror_dialogue",
        output_name="31_mirror_dialogue.wav",
        build_score=build_score,
    ),
}
