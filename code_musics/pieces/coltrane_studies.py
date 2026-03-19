"""Coltrane-cycle studies — JI and EDO riffs on Giant Steps root motion.

Giant Steps works in 12-TET because (2^(4/12))^3 = 2: three major thirds
close the octave exactly, thanks to the diesis (128/125 ≈ 41 ¢) being
tempered out. In JI, (5/4)^3 = 125/64 — 41 ¢ *flat* of 2, so the cycle
never closes. These three sketches explore that idea and its analogues.

1. ji_spiral_steps:    5/4 root motion — each lap returns 41 ¢ flat (diesis)
2. septimal_changes:   7/5 root motion — each cycle returns 35 ¢ flat (50/49)
3. giant_steps_15edo:  15-EDO major thirds — closes like 12-TET, but the
                        "fifth" is 640 ¢ instead of 702 ¢
"""

from __future__ import annotations

import logging
import math

from code_musics.humanize import EnvelopeHumanizeSpec, TimingHumanizeSpec
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Score

logger = logging.getLogger(__name__)

_STEP_15EDO = 2 ** (1 / 15)  # 80 cents per step


def _normalize_to_range(freq: float, low: float, high: float) -> float:
    """Shift freq by octaves until it falls in [low, high)."""
    while freq >= high:
        freq /= 2
    while freq < low:
        freq *= 2
    return freq


# ---------------------------------------------------------------------------
# Piece 1 — ji_spiral_steps
# ---------------------------------------------------------------------------


def build_ji_spiral_steps_score() -> Score:
    """JI Giant Steps: three key centers each a just major third (5/4) apart.

    In 12-TET, (2^(1/3))^3 = 2 — the cycle closes.
    In JI, (5/4)^3 = 125/64, which is the 5-limit diesis (128/125 ≈ 41 ¢)
    flat of a true octave. Each lap through B → Eb → G returns the "B"
    41 ¢ flat against the fixed B2 drone. Three laps = ~123 ¢ total drift.
    """
    chord_dur = 4.0
    laps = 3
    n_steps = 3  # B → Eb → G each lap
    reverb_tail = 5.0
    total_dur = laps * n_steps * chord_dur + reverb_tail

    score = Score(
        f0=110.0,
        master_effects=[
            EffectSpec(
                "reverb", {"room_size": 0.60, "damping": 0.50, "wet_level": 0.25}
            ),
            EffectSpec("delay", {"delay_seconds": 0.33, "feedback": 0.14, "mix": 0.09}),
        ],
    )
    score.add_voice(
        "drone",
        synth_defaults={
            "harmonic_rolloff": 0.58,
            "n_harmonics": 7,
            "attack": 2.0,
            "decay": 0.5,
            "sustain_level": 0.84,
            "release": 3.5,
        },
    )
    score.add_voice(
        "bass",
        synth_defaults={
            "harmonic_rolloff": 0.50,
            "n_harmonics": 8,
            "attack": 0.14,
            "decay": 0.22,
            "sustain_level": 0.72,
            "release": 0.90,
        },
    )
    score.add_voice(
        "chord",
        synth_defaults={
            "harmonic_rolloff": 0.36,
            "n_harmonics": 6,
            "attack": 0.20,
            "decay": 0.18,
            "sustain_level": 0.65,
            "release": 0.85,
        },
        pan=0.10,
    )

    # Fixed drone: B2 = 9/8 × 110 Hz = 123.75 Hz — the ear's anchor.
    # As the spiral chords drift flat against it, dissonance accumulates.
    b2 = 110.0 * 9 / 8  # 123.75 Hz
    score.add_note("drone", start=0.0, duration=total_dur, freq=b2, amp=0.16)

    # Spiral root: starts at b2, ×(5/4) each step.
    # Normalized to [120, 240) for the bass register so every chord sounds
    # grounded; the drift is heard as the bass/chord relationship to the drone
    # sinking ~41 ¢ per lap.
    f_root = b2
    t = 0.0
    for lap in range(laps):
        for step in range(n_steps):
            logger.info(
                "lap=%d step=%d  f_root=%.2f Hz  (%.1f ¢ drift from B2 drone)",
                lap,
                step,
                f_root,
                1200.0 * math.log2(f_root / b2),
            )
            f_bass = _normalize_to_range(f_root, 120.0, 240.0)
            f_chord_root = f_bass * 2  # one octave above bass
            f_third = f_chord_root * 5 / 4  # just major third
            f_fifth = f_chord_root * 3 / 2  # just perfect fifth

            note_dur = chord_dur + 0.30
            score.add_note("bass", start=t, duration=note_dur, freq=f_bass, amp=0.26)
            # Gently arpeggiated triad
            score.add_note(
                "chord", start=t, duration=note_dur, freq=f_chord_root, amp=0.18
            )
            score.add_note(
                "chord",
                start=t + 0.08,
                duration=note_dur - 0.08,
                freq=f_third,
                amp=0.15,
            )
            score.add_note(
                "chord",
                start=t + 0.16,
                duration=note_dur - 0.16,
                freq=f_fifth,
                amp=0.12,
            )

            t += chord_dur
            f_root *= 5 / 4  # advance to next key center

    return score


# ---------------------------------------------------------------------------
# Piece 2 — septimal_changes
# ---------------------------------------------------------------------------

# Melodic gestures per block (8 total), as (ratio × f_chord_root, seconds).
# Ratios draw from the 4:5:6:7 series and the septimal steps between them.
# Block 0: silent. Block 1: chord enters, melody waits.
# Blocks 2-4: melody joins and builds. Block 4: peak. Blocks 5-7: settle.
_SEPTIMAL_MELODY: list[list[tuple[float, float]]] = [
    [],  # 0: establishing — bass + drone only
    [],  # 1: chord enters, melody holds back
    [(2.0, 0.7), (7 / 4, 0.8), (3 / 2, 2.0)],  # 2: from octave, descend
    [(3 / 2, 0.5), (7 / 4, 0.7), (2.0, 2.3)],  # 3: rise through 7th to octave
    [  # 4: peak — climbing then back down through the series
        (5 / 4, 0.4),
        (3 / 2, 0.4),
        (7 / 4, 0.5),
        (2.0, 0.5),
        (7 / 4, 0.5),
        (3 / 2, 1.2),
    ],
    [(2.0, 0.5), (7 / 4, 0.6), (3 / 2, 0.7), (7 / 4, 1.7)],  # 5: descend, linger on 7th
    [(3 / 2, 1.0), (7 / 4, 0.8), (3 / 2, 1.7)],  # 6: gentle arc, settling
    [(5 / 4, 1.8), (1.0, 1.8)],  # 7: final — thirds to root
]

# Chord/bass amplitude arc and melody amplitude, indexed by block (0–7).
# Arc peaks at block 4, melody enters block 2.
_CHORD_AMP_ARC = [0.68, 0.76, 0.85, 0.92, 1.00, 0.94, 0.84, 0.70]
_MEL_AMP_ARC = [0.00, 0.00, 0.24, 0.30, 0.36, 0.32, 0.26, 0.20]


def build_septimal_changes_score() -> Score:
    """Septimal tritone (7/5) root motion — 7-limit riff on Giant Steps.

    Two key centers a septimal tritone (7/5 ≈ 582.5 ¢) apart. Two of them
    almost close the octave: (7/5)^2 = 49/25 ≈ 35 ¢ flat (the 50/49 comma).

    Chord voicing: 4:5:6:7 — root, just major third, fifth, harmonic 7th.
    Melody traces the same harmonic series above the chord, with the
    septimal intervals (7/6, 8/7) appearing as characteristic melodic steps.

    Structure: sparse opening (bass+drone) → chord entry → melody builds →
    peak at block 4 → quiet resolution. Four drift cycles = ~140 ¢ total.
    """
    chord_dur = 3.5
    n_cycles = 4  # 4 × (A, B) pairs = 8 chord blocks
    reverb_tail = 5.0
    total_dur = n_cycles * 2 * chord_dur + reverb_tail

    score = Score(
        f0=110.0,
        timing_humanize=TimingHumanizeSpec(preset="chamber"),
        master_effects=[
            EffectSpec(
                "reverb", {"room_size": 0.62, "damping": 0.48, "wet_level": 0.26}
            ),
            EffectSpec("delay", {"delay_seconds": 0.28, "feedback": 0.14, "mix": 0.09}),
            EffectSpec(
                "saturation", {"preset": "neve_gentle", "mix": 0.18, "drive": 1.08}
            ),
        ],
    )

    score.add_voice(
        "drone",
        synth_defaults={
            "harmonic_rolloff": 0.56,
            "n_harmonics": 7,
            "attack": 2.0,
            "decay": 0.4,
            "sustain_level": 0.84,
            "release": 3.5,
        },
    )
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "filtered_stack",
            "preset": "round_bass",
            "attack": 0.08,
            "decay": 0.22,
            "sustain_level": 0.70,
            "release": 0.90,
        },
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        pan=-0.10,
    )
    score.add_voice(
        "chord",
        synth_defaults={
            "engine": "filtered_stack",
            "preset": "warm_pad",
            "attack": 0.28,
            "decay": 0.25,
            "sustain_level": 0.72,
            "release": 1.2,
        },
        effects=[EffectSpec("chorus", {"preset": "juno_subtle", "mix": 0.22})],
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        pan=0.08,
    )
    score.add_voice(
        "melody",
        synth_defaults={
            "engine": "filtered_stack",
            "preset": "reed_lead",
            "attack": 0.04,
            "decay": 0.14,
            "sustain_level": 0.64,
            "release": 0.30,
        },
        envelope_humanize=EnvelopeHumanizeSpec(preset="loose_pluck"),
        pan=-0.08,
    )

    # Fixed A2=110 Hz drone — the anchor the spiral slowly drifts away from.
    score.add_note("drone", start=0.0, duration=total_dur, freq=110.0, amp=0.14)

    f_root = 110.0
    t = 0.0
    for cycle in range(n_cycles):
        for center_idx in range(2):
            block_idx = cycle * 2 + center_idx
            chord_amp = _CHORD_AMP_ARC[block_idx]
            mel_amp = _MEL_AMP_ARC[block_idx]

            logger.info(
                "block=%d  f_root=%.2f Hz  (%.1f ¢ from A2)  chord_amp=%.2f",
                block_idx,
                f_root,
                1200.0 * math.log2(f_root / 110.0),
                chord_amp,
            )

            f_bass = _normalize_to_range(f_root, 100.0, 200.0)
            f_chord_root = f_bass * 2
            f_third = f_chord_root * 5 / 4  # just major third
            f_fifth = f_chord_root * 3 / 2  # just perfect fifth
            f_seventh = f_chord_root * 7 / 4  # harmonic seventh

            note_dur = chord_dur + 0.30

            # Bass: main note + mid-block re-attack for rhythmic pulse.
            score.add_note(
                "bass", start=t, duration=note_dur, freq=f_bass, amp=0.24 * chord_amp
            )
            score.add_note(
                "bass",
                start=t + 1.9,
                duration=1.3,
                freq=f_bass,
                amp=0.16 * chord_amp,
            )

            # Chord: enter from block 1 onward, arpeggiated through 4:5:6:7.
            if block_idx >= 1:
                score.add_note(
                    "chord",
                    start=t,
                    duration=note_dur,
                    freq=f_chord_root,
                    amp=0.17 * chord_amp,
                )
                score.add_note(
                    "chord",
                    start=t + 0.07,
                    duration=note_dur - 0.07,
                    freq=f_third,
                    amp=0.14 * chord_amp,
                )
                score.add_note(
                    "chord",
                    start=t + 0.14,
                    duration=note_dur - 0.14,
                    freq=f_fifth,
                    amp=0.12 * chord_amp,
                )
                score.add_note(
                    "chord",
                    start=t + 0.21,
                    duration=note_dur - 0.21,
                    freq=f_seventh,
                    amp=0.10 * chord_amp,
                )

            # Melody: traces the harmonic series, present from block 2 onward.
            if mel_amp > 0:
                mt = t + 0.12  # slight pickup breath before first note
                for ratio, dur in _SEPTIMAL_MELODY[block_idx]:
                    score.add_note(
                        "melody",
                        start=mt,
                        duration=dur * 0.88,  # articulation gap
                        freq=f_chord_root * ratio,
                        amp=mel_amp,
                    )
                    mt += dur

            t += chord_dur
            f_root *= 7 / 5  # septimal tritone to next center

    return score


# ---------------------------------------------------------------------------
# Piece 3 — giant_steps_15edo
# ---------------------------------------------------------------------------


def build_giant_steps_15edo_score() -> Score:
    """Giant Steps in 15-EDO: closes like 12-TET, but fifths are 640 ¢.

    15-EDO step = 80 ¢. Its major third (5 steps = 400 ¢) is identical to
    12-TET, so three still close the octave exactly — same symmetry as the
    original Giant Steps. But the fifth (8 steps = 640 ¢) is 62 ¢ wider
    than just, giving each chord an alien, wide-open sound.

    No drift here — it's the closed cycle, but with a different timbre than
    either 12-TET or JI. The three key centers repeat identically across laps.
    """
    chord_dur = 3.0
    laps = 3
    n_steps = 3

    score = Score(
        f0=110.0,
        master_effects=[
            EffectSpec(
                "reverb", {"room_size": 0.55, "damping": 0.52, "wet_level": 0.24}
            ),
            EffectSpec("delay", {"delay_seconds": 0.30, "feedback": 0.13, "mix": 0.08}),
        ],
    )
    score.add_voice(
        "bass",
        synth_defaults={
            "harmonic_rolloff": 0.50,
            "n_harmonics": 8,
            "attack": 0.12,
            "decay": 0.20,
            "sustain_level": 0.72,
            "release": 0.90,
        },
    )
    score.add_voice(
        "chord",
        synth_defaults={
            "harmonic_rolloff": 0.34,
            "n_harmonics": 6,
            "attack": 0.18,
            "decay": 0.18,
            "sustain_level": 0.64,
            "release": 0.85,
        },
        pan=0.10,
    )

    # 15-EDO interval ratios from root
    third_15 = _STEP_15EDO**5  # 5 steps = 400 ¢ (same as 12-TET major third)
    fifth_15 = _STEP_15EDO**8  # 8 steps = 640 ¢ (vs 702 ¢ just — noticeably wide)

    # Three key centers, each 5 EDO-steps (400 ¢) apart.
    # Starting root: B2 ≈ 123.75 Hz. The cycle closes: (2^(5/15))^3 = 2.
    b2 = 110.0 * 9 / 8  # 123.75 Hz
    key_freqs = [b2 * _STEP_15EDO ** (5 * k) for k in range(n_steps)]

    t = 0.0
    for lap in range(laps):
        for step_idx, f_root in enumerate(key_freqs):
            f_bass = _normalize_to_range(f_root, 120.0, 240.0)
            f_chord_root = f_bass * 2
            f_third = f_chord_root * third_15
            f_fifth = f_chord_root * fifth_15

            logger.info(
                "lap=%d step=%d  f_root=%.2f Hz  third=%.2f (400 ¢)  fifth=%.2f (640 ¢)",
                lap,
                step_idx,
                f_root,
                f_third,
                f_fifth,
            )
            note_dur = chord_dur + 0.30
            score.add_note("bass", start=t, duration=note_dur, freq=f_bass, amp=0.26)
            score.add_note(
                "chord", start=t, duration=note_dur, freq=f_chord_root, amp=0.18
            )
            score.add_note(
                "chord",
                start=t + 0.08,
                duration=note_dur - 0.08,
                freq=f_third,
                amp=0.15,
            )
            score.add_note(
                "chord",
                start=t + 0.16,
                duration=note_dur - 0.16,
                freq=f_fifth,
                amp=0.12,
            )

            t += chord_dur

    return score


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PIECES: dict[str, PieceDefinition] = {
    "ji_spiral_steps": PieceDefinition(
        name="ji_spiral_steps",
        output_name="20_ji_spiral_steps.wav",
        build_score=build_ji_spiral_steps_score,
    ),
    "septimal_changes": PieceDefinition(
        name="septimal_changes",
        output_name="21_septimal_changes.wav",
        build_score=build_septimal_changes_score,
    ),
    "giant_steps_15edo": PieceDefinition(
        name="giant_steps_15edo",
        output_name="22_giant_steps_15edo.wav",
        build_score=build_giant_steps_15edo_score,
    ),
}
