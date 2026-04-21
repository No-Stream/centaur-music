"""`ji_comma_drift` piece builder."""

from __future__ import annotations

import logging
import math

from code_musics.composition import (
    ContextSection,
    ContextSectionSpec,
    HarmonicContext,
    build_context_sections,
    place_ratio_chord,
    place_ratio_line,
)
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Score

logger = logging.getLogger(__name__)

_MELODY_V1: list[tuple[float, float]] = [
    (2.0, 0.50),
    (5 / 2, 0.75),
    (2.0, 0.50),
    (15 / 8, 0.75),
    (5 / 3, 0.50),
    (2.0, 1.00),
    (5 / 3, 1.00),
    (4 / 3, 0.50),
    (5 / 3, 0.50),
    (2.0, 0.75),
    (5 / 3, 0.375),
    (4 / 3, 0.375),
    (3 / 2, 0.50),
    (15 / 8, 0.50),
    (9 / 4, 1.00),
    (15 / 8, 0.50),
]
_MELODY_V2: list[tuple[float, float]] = [
    (2.0, 0.375),
    (9 / 4, 0.375),
    (5 / 2, 0.375),
    (9 / 4, 0.375),
    (2.0, 0.375),
    (15 / 8, 0.375),
    (2.0, 0.25),
    (5 / 3, 0.75),
    (2.0, 0.50),
    (5 / 2, 0.50),
    (2.0, 0.25),
    (5 / 3, 0.50),
    (10 / 9, 0.50),
    (4 / 3, 0.375),
    (5 / 3, 0.375),
    (2.0, 0.50),
    (5 / 3, 0.375),
    (4 / 3, 0.375),
    (9 / 4, 1.00),
    (5 / 2, 0.75),
    (9 / 4, 0.375),
    (15 / 8, 0.375),
]
_MELODY_V3: list[tuple[float, float]] = [
    (5 / 2, 1.00),
    (9 / 4, 0.50),
    (2.0, 0.50),
    (15 / 8, 0.50),
    (5 / 3, 0.50),
    (2.0, 0.50),
    (5 / 3, 0.375),
    (3 / 2, 0.375),
    (5 / 3, 0.75),
    (4 / 3, 0.75),
    (5 / 3, 0.50),
    (2.0, 0.75),
    (5 / 3, 0.50),
    (15 / 8, 0.50),
    (9 / 4, 0.75),
    (15 / 8, 0.50),
    (3 / 2, 0.75),
]
_MELODY_SNAP: list[tuple[float, float]] = [
    (2.0, 1.00),
    (5 / 2, 0.75),
    (9 / 4, 0.50),
    (2.0, 0.75),
]
_MELODY_SNAP_LONG: list[tuple[float, float]] = [
    (2.0, 1.25),
    (15 / 8, 0.50),
    (2.0, 0.75),
    (9 / 4, 0.75),
    (5 / 2, 0.75),
    (9 / 4, 0.50),
    (2.0, 0.50),
]


def _place_melody(
    score: Score,
    section: ContextSection,
    notes: list[tuple[float, float]],
    amp: float = 0.38,
) -> None:
    place_ratio_line(
        score,
        "melody",
        section=section,
        tones=[ratio for ratio, _ in notes],
        rhythm=[duration for _, duration in notes],
        amp=amp,
    )


def _place_chord_cycle(
    score: Score,
    sections: tuple[ContextSection, ...],
    chord_dur: float,
) -> None:
    """One I–IV–ii–V cycle tuned purely from tonic f_c."""
    note_dur = chord_dur + 0.35
    bass_dur = chord_dur * 0.84
    arp_gap = 0.08

    cycle: list[tuple[float, list[tuple[float, float]]]] = [
        (1 / 2, [(1.0, 0.20), (5 / 4, 0.16), (3 / 2, 0.13)]),
        (4 / 3 / 2, [(4 / 3, 0.18), (5 / 3, 0.14), (2.0, 0.11)]),
        (10 / 9 / 2, [(10 / 9, 0.17), (4 / 3, 0.13), (5 / 3, 0.11)]),
        (3 / 2 / 2, [(3 / 2, 0.20), (15 / 8, 0.15), (9 / 4, 0.13)]),
    ]
    for section, (bass_ratio, harmony_notes) in zip(sections, cycle, strict=True):
        place_ratio_chord(
            score,
            "bass",
            section=section,
            ratios=[bass_ratio],
            duration=bass_dur,
            amp=0.26,
        )
        place_ratio_chord(
            score,
            "chord",
            section=section,
            ratios=[ratio for ratio, _ in harmony_notes],
            duration=note_dur,
            amp=[amp for _, amp in harmony_notes],
            gap=arp_gap,
        )


def _place_snap(
    score: Score,
    section: ContextSection,
    melody: list[tuple[float, float]],
) -> None:
    """Sustained I chord at drone_freq — the snap-back moment."""
    note_dur = section.duration - 0.4
    place_ratio_chord(
        score,
        "bass",
        section=section,
        ratios=[1 / 2],
        duration=note_dur,
        amp=0.30,
    )
    place_ratio_chord(
        score,
        "chord",
        section=section,
        ratios=[1.0, 5 / 4, 3 / 2],
        duration=note_dur,
        amp=[0.22, 0.18, 0.15],
    )
    _place_melody(score, section, melody, amp=0.42)


def build_ji_comma_drift_score() -> Score:
    """Syntonic comma pump with an anthemic drifting melody."""
    drone_freq = 220.0
    chord_dur = 2.5
    cycle_dur = 4 * chord_dur
    comma = 81 / 80

    intro_dur = 4.0
    snap_dur = 3.0
    snap_long_dur = 5.0
    n_blocks = 3

    total_dur = (
        intro_dur
        + (n_blocks - 1) * (cycle_dur + snap_dur)
        + cycle_dur
        + snap_long_dur
        + cycle_dur
        + 3.0
    )

    score = Score(
        f0_hz=drone_freq,
        master_effects=[
            EffectSpec("delay", {"delay_seconds": 0.26, "feedback": 0.14, "mix": 0.09}),
            EffectSpec(
                "reverb",
                {"room_size": 0.68, "damping": 0.44, "wet_level": 0.24},
            ),
        ],
    )
    score.add_voice(
        "drone",
        synth_defaults={
            "harmonic_rolloff": 0.48,
            "n_harmonics": 8,
            "attack": 1.8,
            "decay": 0.4,
            "sustain_level": 0.82,
            "release": 3.5,
        },
    )
    score.add_voice(
        "chord",
        synth_defaults={
            "harmonic_rolloff": 0.34,
            "n_harmonics": 6,
            "attack": 0.35,
            "decay": 0.20,
            "sustain_level": 0.70,
            "release": 0.90,
        },
    )
    score.add_voice(
        "bass",
        synth_defaults={
            "harmonic_rolloff": 0.50,
            "n_harmonics": 8,
            "attack": 0.14,
            "decay": 0.20,
            "sustain_level": 0.76,
            "release": 0.70,
        },
    )
    score.add_voice(
        "melody",
        synth_defaults={
            "harmonic_rolloff": 0.24,
            "n_harmonics": 5,
            "attack": 0.05,
            "decay": 0.12,
            "sustain_level": 0.72,
            "release": 0.45,
        },
    )

    score.add_note(
        "drone",
        start=0.0,
        duration=total_dur,
        freq=drone_freq,
        amp=0.22,
        label="A=220 drone",
    )

    melody_variants = [_MELODY_V1, _MELODY_V2, _MELODY_V3]
    snap_melodies = [_MELODY_SNAP, _MELODY_SNAP, _MELODY_SNAP_LONG]
    snap_durations = [snap_dur, snap_dur, snap_long_dur]

    current_time = intro_dur
    for block_idx in range(n_blocks):
        drifted_tonic = drone_freq * (1 / comma)
        logger.info(
            "block %d  tonic=%.2f Hz  (%.2f ¢ from drone)",
            block_idx,
            drifted_tonic,
            1200 * math.log2(drifted_tonic / drone_freq),
        )
        drift_sections = build_context_sections(
            base_tonic=drone_freq,
            start=current_time,
            specs=(
                ContextSectionSpec(name="I", duration=chord_dur, tonic_ratio=1 / comma),
                ContextSectionSpec(
                    name="IV", duration=chord_dur, tonic_ratio=1 / comma
                ),
                ContextSectionSpec(
                    name="ii", duration=chord_dur, tonic_ratio=1 / comma
                ),
                ContextSectionSpec(name="V", duration=chord_dur, tonic_ratio=1 / comma),
            ),
        )
        _place_chord_cycle(score, drift_sections, chord_dur)
        _place_melody(
            score,
            ContextSection(
                start=current_time,
                duration=cycle_dur,
                context=HarmonicContext(
                    tonic=drifted_tonic,
                    name=f"block_{block_idx}_drift",
                ),
            ),
            melody_variants[block_idx],
        )
        current_time += cycle_dur

        _place_snap(
            score,
            ContextSection(
                start=current_time,
                duration=snap_durations[block_idx],
                context=HarmonicContext(
                    tonic=drone_freq,
                    name=f"block_{block_idx}_snap",
                ),
            ),
            snap_melodies[block_idx],
        )
        current_time += snap_durations[block_idx]

    coda_sections = build_context_sections(
        base_tonic=drone_freq,
        start=current_time,
        specs=(
            ContextSectionSpec(name="I", duration=chord_dur),
            ContextSectionSpec(name="IV", duration=chord_dur),
            ContextSectionSpec(name="ii", duration=chord_dur),
            ContextSectionSpec(name="V", duration=chord_dur),
        ),
    )
    _place_chord_cycle(score, coda_sections, chord_dur)
    _place_melody(
        score,
        ContextSection(
            start=current_time,
            duration=cycle_dur,
            context=HarmonicContext(tonic=drone_freq, name="coda"),
        ),
        _MELODY_V1,
        amp=0.40,
    )

    return score


PIECES: dict[str, PieceDefinition] = {
    "ji_comma_drift": PieceDefinition(
        name="ji_comma_drift",
        output_name="19_ji_comma_drift",
        build_score=build_ji_comma_drift_score,
    )
}
