"""Septimal JI explorations rendered through the score abstractions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from code_musics.score import EffectSpec, Phrase, Score
from code_musics.synth import sequence
from code_musics.tuning import ratio_to_cents

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
ROOT = 110.0


@dataclass(frozen=True)
class PieceDefinition:
    """Named renderable piece."""

    name: str
    output_name: str
    build_score: Callable[[], Score] | None = None
    render_audio: Callable[[], np.ndarray] | None = None


def render_interval_demo() -> np.ndarray:
    """Render each septimal interval against a drone."""
    intervals: list[tuple[str, float]] = [
        ("unison", 1 / 1),
        ("8/7 whole tone", 8 / 7),
        ("7/6 min 3rd", 7 / 6),
        ("9/7 maj 3rd", 9 / 7),
        ("7/5 tritone", 7 / 5),
        ("3/2 fifth", 3 / 2),
        ("7/4 harm 7th", 7 / 4),
        ("2/1 octave", 2 / 1),
    ]

    note_dur = 3.5
    segments: list[np.ndarray] = []

    for name, ratio in intervals:
        score = Score(f0=ROOT)
        score.add_voice(
            "drone",
            synth_defaults={
                "attack": 0.06,
                "release": 0.4,
            },
        )
        score.add_voice(
            "upper",
            synth_defaults={
                "attack": 0.08,
                "release": 0.5,
            },
        )
        score.add_note("drone", start=0.0, duration=note_dur, partial=1.0, amp=0.6)
        score.add_note("upper", start=0.0, duration=note_dur, freq=ROOT * ratio, amp=0.6)
        logger.info("  %s  %.2f Hz  (%.1f cents)", name, ROOT * ratio, ratio_to_cents(ratio))
        segments.append(score.render())

    return sequence(*segments, gap=0.25)


def build_chord_4567_score() -> Score:
    """Build the 4:5:6:7 chord buildup score."""
    score = Score(f0=ROOT)
    ratios = [4 / 4, 5 / 4, 6 / 4, 7 / 4]
    names = ["4 - root", "5 - major third", "6 - perfect fifth", "7 - harmonic seventh"]
    entry_offsets = [0.0, 3.0, 6.0, 9.0]
    hold_until = 18.0

    score.add_voice(
        "chord",
        synth_defaults={
            "attack": 0.5,
            "decay": 0.3,
            "sustain_level": 0.8,
            "release": 1.5,
        },
    )

    for ratio, name, offset in zip(ratios, names, entry_offsets):
        freq = ROOT * ratio
        note_dur = hold_until - offset
        logger.info("  %s  %.2f Hz  enters at %.1fs", name, freq, offset)
        score.add_note("chord", start=offset, duration=note_dur, freq=freq, amp=0.55, label=name)

    return score


def build_harmonic_drift_score() -> Score:
    """Build the large-form septimal piece using phrases and direct notes."""
    score = Score(
        f0=55.0,
        master_effects=[
            EffectSpec("delay", {"delay_seconds": 0.38, "feedback": 0.28, "mix": 0.22}),
            EffectSpec("bricasti", {"ir_name": "1 Halls 07 Large & Dark", "wet": 0.35}),
        ],
    )

    score.add_voice(
        "drone",
        synth_defaults={
            "harmonic_rolloff": 0.45,
            "attack": 1.0,
            "decay": 0.5,
            "sustain_level": 0.85,
            "release": 4.0,
        },
    )
    score.add_note("drone", start=0.0, duration=85.0, partial=1.0, amp=0.55, label="bass drone")

    score.add_voice(
        "mid_chord",
        synth_defaults={
            "harmonic_rolloff": 0.4,
            "attack": 1.5,
            "decay": 0.5,
            "sustain_level": 0.7,
            "release": 4.0,
        },
    )
    for partial, offset in [(4, 6.0), (6, 9.0), (5, 11.0), (7, 14.0)]:
        score.add_note("mid_chord", start=offset, duration=30.0, partial=partial, amp=0.35)

    score.add_voice("melody_a")
    score.add_voice("melody_b")
    score.add_voice("pedal")
    score.add_voice("resolution")

    ascending_phrase = Phrase.from_partials(
        [8, 9, 11, 12, 14],
        note_dur=1.6,
        step=1.6 * 0.85,
        amp=0.45,
        synth_defaults={
            "harmonic_rolloff": 0.35,
            "attack": 0.05,
            "decay": 0.15,
            "sustain_level": 0.7,
            "release": 0.6,
        },
    )
    score.add_phrase("melody_a", ascending_phrase, start=18.0)

    weird_main_phrase = Phrase.from_partials(
        [14, 13, 11, 13, 14, 11, 13],
        note_dur=2.4,
        step=2.4 * 0.80,
        amp=0.38,
        synth_defaults={
            "harmonic_rolloff": 0.3,
            "attack": 0.12,
            "decay": 0.2,
            "sustain_level": 0.65,
            "release": 1.0,
        },
    )
    weird_start = 18.0 + ascending_phrase.duration
    score.add_phrase("melody_a", weird_main_phrase, start=weird_start)

    high_phrase = Phrase.from_partials(
        [16, 17, 19, 18, 17, 16, 19, 17],
        note_dur=2.0,
        step=2.0 * 0.75,
        amp=0.22,
        synth_defaults={
            "harmonic_rolloff": 0.25,
            "attack": 0.3,
            "decay": 0.2,
            "sustain_level": 0.55,
            "release": 1.2,
        },
    )
    weird_high_start = 18.0 + ascending_phrase.duration + 1.0
    score.add_phrase("melody_b", high_phrase, start=weird_high_start)

    descent_phrase = Phrase.from_partials(
        [14, 12, 9, 7],
        note_dur=2.0,
        step=2.0 * 0.82,
        amp=0.40,
        synth_defaults={
            "harmonic_rolloff": 0.35,
            "attack": 0.15,
            "decay": 0.2,
            "sustain_level": 0.7,
            "release": 0.9,
        },
    )
    score.add_phrase("melody_a", descent_phrase, start=40.0)

    canon_phrase = Phrase.from_partials(
        [4, 5, 6, 7, 6, 5, 6, 7, 6, 5, 4],
        note_dur=1.6,
        step=1.6 * 0.78,
        amp=0.42,
        synth_defaults={
            "harmonic_rolloff": 0.38,
            "attack": 0.06,
            "decay": 0.12,
            "sustain_level": 0.72,
            "release": 0.55,
        },
    )
    score.add_phrase("melody_a", canon_phrase, start=46.0)
    score.add_phrase("melody_b", canon_phrase, start=49.0, partial_shift=4, amp_scale=0.34 / 0.42)

    score.add_voice(
        "pedal",
        synth_defaults={
            "harmonic_rolloff": 0.35,
            "attack": 1.0,
            "decay": 0.3,
            "sustain_level": 0.7,
            "release": 3.0,
        },
    )
    score.add_note("pedal", start=54.0, duration=18.0, partial=4.0, amp=0.28)

    call_phrase = Phrase.from_partials(
        [5, 7, 6, 5],
        note_dur=1.2,
        step=1.2 * 0.80,
        amp=0.38,
        synth_defaults={
            "harmonic_rolloff": 0.36,
            "attack": 0.05,
            "decay": 0.10,
            "sustain_level": 0.70,
            "release": 0.50,
        },
    )
    response_phrase = Phrase.from_partials(
        [8, 7, 6, 4],
        note_dur=1.2,
        step=1.2 * 0.80,
        amp=0.33,
        synth_defaults={
            "harmonic_rolloff": 0.36,
            "attack": 0.05,
            "decay": 0.10,
            "sustain_level": 0.70,
            "release": 0.50,
        },
    )
    for call_start, shift in [(54.0, 0.0), (60.0, 1.0), (66.0, 2.0)]:
        score.add_phrase("melody_a", call_phrase, start=call_start, partial_shift=shift)
        score.add_phrase("melody_b", response_phrase, start=call_start + 2.5, partial_shift=shift)

    contrary_up = Phrase.from_partials(
        [4, 5, 6, 7, 8, 9, 10, 12, 14],
        note_dur=1.0,
        step=1.0 * 0.82,
        amp=0.38,
        synth_defaults={
            "harmonic_rolloff": 0.34,
            "attack": 0.04,
            "decay": 0.08,
            "sustain_level": 0.68,
            "release": 0.40,
        },
    )
    contrary_down = Phrase.from_partials(
        [14, 12, 10, 9, 8, 7, 6, 5, 4],
        note_dur=1.0,
        step=1.0 * 0.82,
        amp=0.34,
        synth_defaults={
            "harmonic_rolloff": 0.34,
            "attack": 0.04,
            "decay": 0.08,
            "sustain_level": 0.68,
            "release": 0.40,
        },
    )
    score.add_phrase("melody_a", contrary_up, start=71.0)
    score.add_phrase("melody_b", contrary_down, start=71.0)

    score.add_voice(
        "resolution",
        synth_defaults={
            "harmonic_rolloff": 0.18,
            "attack": 2.5,
            "decay": 0.5,
            "sustain_level": 0.75,
            "release": 5.0,
        },
    )
    resolution_duration = 85.0 - 78.0 - 2.0
    for partial, bloom_offset, amp in [
        (4, 0.0, 0.14),
        (5, 0.7, 0.12),
        (6, 1.4, 0.12),
        (7, 2.1, 0.14),
        (8, 0.4, 0.11),
        (9, 1.8, 0.09),
        (10, 1.1, 0.10),
        (12, 2.5, 0.10),
        (14, 3.2, 0.12),
    ]:
        score.add_note(
            "resolution",
            start=78.0 + bloom_offset,
            duration=resolution_duration,
            partial=partial,
            amp=amp,
        )

    return score


PIECES: dict[str, PieceDefinition] = {
    "interval_demo": PieceDefinition(
        name="interval_demo",
        output_name="01_interval_demo.wav",
        render_audio=render_interval_demo,
    ),
    "chord_4567": PieceDefinition(
        name="chord_4567",
        output_name="02_chord_4567.wav",
        build_score=build_chord_4567_score,
    ),
    "harmonic_drift": PieceDefinition(
        name="harmonic_drift",
        output_name="03_harmonic_drift.wav",
        build_score=build_harmonic_drift_score,
    ),
}
