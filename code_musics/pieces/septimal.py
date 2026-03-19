"""Septimal JI explorations rendered through the score abstractions."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Phrase, Score
from code_musics.synth import sequence
from code_musics.tuning import otonal, ratio_to_cents, utonal

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output")
ROOT = 110.0


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
        score.add_note(
            "upper", start=0.0, duration=note_dur, freq=ROOT * ratio, amp=0.6
        )
        logger.info(
            "  %s  %.2f Hz  (%.1f cents)", name, ROOT * ratio, ratio_to_cents(ratio)
        )
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

    for ratio, name, offset in zip(ratios, names, entry_offsets, strict=True):
        freq = ROOT * ratio
        note_dur = hold_until - offset
        logger.info("  %s  %.2f Hz  enters at %.1fs", name, freq, offset)
        score.add_note(
            "chord", start=offset, duration=note_dur, freq=freq, amp=0.55, label=name
        )

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
    score.add_note(
        "drone", start=0.0, duration=85.0, partial=1.0, amp=0.55, label="bass drone"
    )

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
        score.add_note(
            "mid_chord", start=offset, duration=30.0, partial=partial, amp=0.35
        )

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
    score.add_phrase(
        "melody_b", canon_phrase, start=49.0, partial_shift=4, amp_scale=0.34 / 0.42
    )

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
        score.add_phrase(
            "melody_b", response_phrase, start=call_start + 2.5, partial_shift=shift
        )

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


def build_harmonic_window_score() -> Score:
    """Build a study that slides one phrase through harmonic windows."""
    score = Score(
        f0=55.0,
        master_effects=[
            EffectSpec("delay", {"delay_seconds": 0.32, "feedback": 0.22, "mix": 0.18}),
            EffectSpec("bricasti", {"ir_name": "1 Halls 07 Large & Dark", "wet": 0.25}),
        ],
    )

    score.add_voice(
        "drone",
        synth_defaults={
            "harmonic_rolloff": 0.42,
            "attack": 1.2,
            "decay": 0.5,
            "sustain_level": 0.85,
            "release": 4.0,
        },
    )
    score.add_note(
        "drone", start=0.0, duration=52.0, partial=1.0, amp=0.42, label="root drone"
    )
    score.add_note(
        "drone", start=12.0, duration=30.0, partial=2.0, amp=0.14, label="upper drone"
    )

    score.add_voice(
        "window_low",
        synth_defaults={
            "harmonic_rolloff": 0.34,
            "attack": 0.06,
            "decay": 0.15,
            "sustain_level": 0.70,
            "release": 0.7,
        },
    )
    score.add_voice(
        "window_high",
        synth_defaults={
            "harmonic_rolloff": 0.26,
            "attack": 0.10,
            "decay": 0.18,
            "sustain_level": 0.62,
            "release": 1.0,
        },
    )
    score.add_voice(
        "punctuation",
        synth_defaults={
            "harmonic_rolloff": 0.30,
            "attack": 0.25,
            "decay": 0.20,
            "sustain_level": 0.65,
            "release": 1.4,
        },
    )

    core_phrase = Phrase.from_partials(
        [6, 7, 9, 8, 7],
        note_dur=1.9,
        step=1.55,
        amp=0.34,
        synth_defaults={
            "harmonic_rolloff": 0.33,
            "attack": 0.05,
            "decay": 0.12,
            "sustain_level": 0.72,
            "release": 0.7,
        },
    )
    echo_phrase = Phrase.from_partials(
        [9, 8, 7, 6],
        note_dur=1.6,
        step=1.25,
        amp=0.22,
        synth_defaults={
            "harmonic_rolloff": 0.25,
            "attack": 0.09,
            "decay": 0.14,
            "sustain_level": 0.60,
            "release": 0.9,
        },
    )

    windows: list[tuple[float, float, float, bool]] = [
        (4.0, 0.0, 1.00, False),
        (12.0, 2.0, 0.92, False),
        (20.0, 4.0, 0.86, True),
        (28.0, 6.0, 0.80, False),
        (36.0, 8.0, 0.76, True),
    ]
    for start, partial_shift, amp_scale, reverse in windows:
        score.add_phrase(
            "window_low",
            core_phrase,
            start=start,
            partial_shift=partial_shift,
            amp_scale=amp_scale,
            reverse=reverse,
        )
        score.add_phrase(
            "window_high",
            echo_phrase,
            start=start + 2.4,
            partial_shift=partial_shift + 6.0,
            amp_scale=amp_scale,
            reverse=not reverse,
        )

    for start, partial in [
        (10.0, 5.0),
        (18.0, 7.0),
        (26.0, 9.0),
        (34.0, 11.0),
        (42.0, 13.0),
    ]:
        score.add_note(
            "punctuation", start=start, duration=5.2, partial=partial, amp=0.18
        )

    return score


def build_otonal_utonal_mirror_score() -> Score:
    """Build a study that mirrors otonal chords with subharmonic answers."""
    score = Score(
        f0=110.0,
        master_effects=[
            EffectSpec("delay", {"delay_seconds": 0.28, "feedback": 0.18, "mix": 0.14}),
            EffectSpec("bricasti", {"ir_name": "1 Halls 07 Large & Dark", "wet": 0.28}),
        ],
    )

    score.add_voice(
        "pedal",
        synth_defaults={
            "harmonic_rolloff": 0.48,
            "attack": 0.9,
            "decay": 0.3,
            "sustain_level": 0.84,
            "release": 3.5,
        },
    )
    score.add_voice(
        "otonal_chord",
        synth_defaults={
            "harmonic_rolloff": 0.28,
            "attack": 0.5,
            "decay": 0.25,
            "sustain_level": 0.72,
            "release": 2.0,
        },
    )
    score.add_voice(
        "utonal_chord",
        synth_defaults={
            "harmonic_rolloff": 0.24,
            "attack": 0.8,
            "decay": 0.25,
            "sustain_level": 0.66,
            "release": 2.4,
        },
    )
    score.add_voice(
        "bridge",
        synth_defaults={
            "harmonic_rolloff": 0.32,
            "attack": 0.08,
            "decay": 0.12,
            "sustain_level": 0.68,
            "release": 0.8,
        },
    )

    score.add_note(
        "pedal", start=0.0, duration=44.0, partial=1.0, amp=0.30, label="pedal"
    )
    score.add_note(
        "pedal", start=22.0, duration=18.0, partial=0.5, amp=0.12, label="sub pedal"
    )

    otonal_bases = [55.0, 73.3333333333, 82.5]
    utonal_bases = [330.0, 275.0, 220.0]
    chord_partials = [4, 5, 6, 7]

    section_starts = [2.0, 14.0, 26.0]
    for start, otonal_base, utonal_base in zip(
        section_starts, otonal_bases, utonal_bases, strict=True
    ):
        for freq in otonal(otonal_base, chord_partials):
            score.add_note(
                "otonal_chord", start=start, duration=6.5, freq=freq, amp=0.18
            )

        bridge_phrase = Phrase.from_partials(
            [6, 7, 6, 5, 4],
            note_dur=0.9,
            step=0.75,
            amp=0.22,
            synth_defaults={
                "harmonic_rolloff": 0.30,
                "attack": 0.04,
                "decay": 0.10,
                "sustain_level": 0.72,
                "release": 0.45,
            },
        )
        score.add_phrase("bridge", bridge_phrase, start=start + 5.0, partial_shift=1.0)

        for freq in sorted(utonal(utonal_base, chord_partials)):
            score.add_note(
                "utonal_chord", start=start + 7.0, duration=7.5, freq=freq, amp=0.16
            )

    closing_phrase = Phrase.from_partials(
        [4, 5, 6, 7, 6, 5, 4],
        note_dur=1.2,
        step=0.95,
        amp=0.20,
        synth_defaults={
            "harmonic_rolloff": 0.28,
            "attack": 0.06,
            "decay": 0.10,
            "sustain_level": 0.68,
            "release": 0.55,
        },
    )
    score.add_phrase("bridge", closing_phrase, start=36.0)

    return score


def build_otonal_utonal_mirror_expanded_score() -> Score:
    """Build a longer-form mirror study with clearer harmonic landmarks."""
    score = Score(
        f0=110.0,
        master_effects=[
            EffectSpec("delay", {"delay_seconds": 0.30, "feedback": 0.20, "mix": 0.15}),
            EffectSpec("bricasti", {"ir_name": "1 Halls 07 Large & Dark", "wet": 0.28}),
        ],
    )

    score.add_voice(
        "pedal",
        synth_defaults={
            "harmonic_rolloff": 0.48,
            "attack": 1.2,
            "decay": 0.3,
            "sustain_level": 0.84,
            "release": 4.5,
        },
    )
    score.add_voice(
        "memory",
        synth_defaults={
            "harmonic_rolloff": 0.24,
            "attack": 1.2,
            "decay": 0.2,
            "sustain_level": 0.70,
            "release": 4.2,
        },
    )
    score.add_voice(
        "otonal_chord",
        synth_defaults={
            "harmonic_rolloff": 0.27,
            "attack": 0.55,
            "decay": 0.25,
            "sustain_level": 0.72,
            "release": 2.2,
        },
    )
    score.add_voice(
        "utonal_chord",
        synth_defaults={
            "harmonic_rolloff": 0.23,
            "attack": 0.85,
            "decay": 0.25,
            "sustain_level": 0.66,
            "release": 2.8,
        },
    )
    score.add_voice(
        "bridge",
        synth_defaults={
            "harmonic_rolloff": 0.30,
            "attack": 0.07,
            "decay": 0.12,
            "sustain_level": 0.68,
            "release": 0.8,
        },
    )
    score.add_voice(
        "high_echo",
        synth_defaults={
            "harmonic_rolloff": 0.22,
            "attack": 0.15,
            "decay": 0.18,
            "sustain_level": 0.58,
            "release": 0.95,
        },
    )

    bridge_phrase = Phrase.from_partials(
        [5, 6, 7, 6, 5, 4],
        note_dur=0.9,
        step=0.74,
        amp=0.20,
        synth_defaults={
            "harmonic_rolloff": 0.30,
            "attack": 0.04,
            "decay": 0.10,
            "sustain_level": 0.72,
            "release": 0.45,
        },
    )
    echo_phrase = Phrase.from_partials(
        [8, 7, 6, 5],
        note_dur=1.0,
        step=0.80,
        amp=0.12,
        synth_defaults={
            "harmonic_rolloff": 0.24,
            "attack": 0.08,
            "decay": 0.12,
            "sustain_level": 0.62,
            "release": 0.60,
        },
    )
    closing_phrase = Phrase.from_partials(
        [4, 5, 6, 7, 8, 7, 6, 5, 4],
        note_dur=1.05,
        step=0.84,
        amp=0.18,
        synth_defaults={
            "harmonic_rolloff": 0.28,
            "attack": 0.06,
            "decay": 0.10,
            "sustain_level": 0.68,
            "release": 0.55,
        },
    )

    sections: list[dict[str, float | bool]] = [
        {
            "start": 2.0,
            "otonal_base": 55.0,
            "utonal_base": 330.0,
            "pedal_freq": 110.0,
            "memory_freq": 165.0,
            "partial_shift": 0.0,
            "reverse_echo": False,
        },
        {
            "start": 16.0,
            "otonal_base": 73.3333333333,
            "utonal_base": 440.0,
            "pedal_freq": 146.6666666666,
            "memory_freq": 220.0,
            "partial_shift": 1.0,
            "reverse_echo": False,
        },
        {
            "start": 30.0,
            "otonal_base": 82.5,
            "utonal_base": 495.0,
            "pedal_freq": 165.0,
            "memory_freq": 247.5,
            "partial_shift": 2.0,
            "reverse_echo": False,
        },
        {
            "start": 44.0,
            "otonal_base": 73.3333333333,
            "utonal_base": 440.0,
            "pedal_freq": 146.6666666666,
            "memory_freq": 220.0,
            "partial_shift": 1.0,
            "reverse_echo": True,
        },
        {
            "start": 58.0,
            "otonal_base": 55.0,
            "utonal_base": 330.0,
            "pedal_freq": 110.0,
            "memory_freq": 165.0,
            "partial_shift": 0.0,
            "reverse_echo": True,
        },
    ]

    chord_partials = [4, 5, 6, 7]

    for index, section in enumerate(sections):
        start = float(section["start"])
        otonal_base = float(section["otonal_base"])
        utonal_base = float(section["utonal_base"])
        pedal_freq = float(section["pedal_freq"])
        memory_freq = float(section["memory_freq"])
        partial_shift = float(section["partial_shift"])
        reverse_echo = bool(section["reverse_echo"])

        score.add_note(
            "pedal", start=start - 2.0, duration=14.5, freq=pedal_freq, amp=0.24
        )

        if index > 0:
            score.add_note(
                "memory",
                start=start - 1.0,
                duration=8.0,
                freq=memory_freq,
                amp=0.08,
                label="memory tone",
            )

        for chord_index, freq in enumerate(otonal(otonal_base, chord_partials)):
            score.add_note(
                "otonal_chord",
                start=start + (0.35 * chord_index),
                duration=6.8,
                freq=freq,
                amp=0.17 + (0.01 * chord_index),
            )

        score.add_phrase(
            "bridge", bridge_phrase, start=start + 5.2, partial_shift=partial_shift
        )
        score.add_phrase(
            "high_echo",
            echo_phrase,
            start=start + 6.5,
            partial_shift=partial_shift + 4.0,
            reverse=reverse_echo,
        )

        for chord_index, freq in enumerate(sorted(utonal(utonal_base, chord_partials))):
            score.add_note(
                "utonal_chord",
                start=start + 8.0 + (0.2 * chord_index),
                duration=6.8,
                freq=freq,
                amp=0.13 + (0.01 * chord_index),
            )

    score.add_note(
        "memory", start=69.0, duration=9.0, freq=165.0, amp=0.10, label="return"
    )
    score.add_note(
        "pedal", start=68.0, duration=16.0, freq=110.0, amp=0.26, label="final pedal"
    )
    score.add_phrase("bridge", closing_phrase, start=70.0)
    score.add_phrase(
        "high_echo", closing_phrase, start=71.0, partial_shift=3.0, amp_scale=0.55
    )

    for chord_index, freq in enumerate(otonal(55.0, chord_partials)):
        score.add_note(
            "otonal_chord",
            start=74.0 + (0.25 * chord_index),
            duration=9.5,
            freq=freq,
            amp=0.11 + (0.01 * chord_index),
        )
    for chord_index, freq in enumerate(sorted(utonal(330.0, chord_partials))):
        score.add_note(
            "utonal_chord",
            start=75.6 + (0.18 * chord_index),
            duration=8.8,
            freq=freq,
            amp=0.09 + (0.01 * chord_index),
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
    "harmonic_window": PieceDefinition(
        name="harmonic_window",
        output_name="04_harmonic_window.wav",
        build_score=build_harmonic_window_score,
    ),
    "otonal_utonal_mirror": PieceDefinition(
        name="otonal_utonal_mirror",
        output_name="05_otonal_utonal_mirror.wav",
        build_score=build_otonal_utonal_mirror_score,
    ),
    "otonal_utonal_mirror_expanded": PieceDefinition(
        name="otonal_utonal_mirror_expanded",
        output_name="06_otonal_utonal_mirror_expanded.wav",
        build_score=build_otonal_utonal_mirror_expanded_score,
    ),
}
