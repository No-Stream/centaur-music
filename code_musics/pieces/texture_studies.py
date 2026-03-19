"""Texture, interference, and spiral-form study pieces."""

from __future__ import annotations

from code_musics.pieces._shared import (
    DELAY_EFFECT,
    REVERB_EFFECT,
    SOFT_REVERB_EFFECT,
    WARM_SATURATION_EFFECT,
)
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import Score


def build_spiral_sketch() -> Score:
    """Same melodic arch at four fundamentals rising by JI fifths (3/2)."""
    score = Score(f0=55.0, master_effects=[DELAY_EFFECT, REVERB_EFFECT])
    score.add_voice(
        "melody",
        synth_defaults={
            "harmonic_rolloff": 0.30,
            "attack": 0.06,
            "decay": 0.14,
            "sustain_level": 0.70,
            "release": 0.85,
        },
    )
    score.add_voice(
        "drone",
        synth_defaults={
            "harmonic_rolloff": 0.48,
            "attack": 1.2,
            "decay": 0.25,
            "sustain_level": 0.80,
            "release": 3.5,
        },
    )

    shape_ratios = [1.0, 1.25, 1.5, 1.75, 1.5, 1.25, 1.0]
    note_dur = 1.4
    step = 1.1
    section_gap = 5.0
    melody_dur = (len(shape_ratios) - 1) * step + note_dur
    section_dur = melody_dur + section_gap

    fundamentals = [55.0]
    for _ in range(3):
        fundamentals.append(fundamentals[-1] * 3 / 2)

    for section_idx, section_f0 in enumerate(fundamentals):
        section_start = section_idx * section_dur
        score.add_note(
            "drone",
            start=section_start,
            duration=melody_dur + 2.0,
            freq=section_f0 * 2.0,
            amp=0.28,
        )
        for note_idx, ratio in enumerate(shape_ratios):
            score.add_note(
                "melody",
                start=section_start + note_idx * step,
                duration=note_dur,
                freq=section_f0 * 4.0 * ratio,
                amp=0.36,
            )

    return score


def build_interference_sketch() -> Score:
    """Two harmonic series 0.5 Hz apart, creating layered beating patterns."""
    f0_a = 110.0
    f0_b = 110.5

    score = Score(f0=f0_a, master_effects=[SOFT_REVERB_EFFECT])
    score.add_voice(
        "series_a",
        synth_defaults={
            "harmonic_rolloff": 0.40,
            "n_harmonics": 3,
            "attack": 1.8,
            "decay": 0.25,
            "sustain_level": 0.84,
            "release": 5.0,
        },
    )
    score.add_voice(
        "series_b",
        synth_defaults={
            "harmonic_rolloff": 0.38,
            "n_harmonics": 3,
            "attack": 2.2,
            "decay": 0.25,
            "sustain_level": 0.80,
            "release": 5.0,
        },
    )
    score.add_voice(
        "solo",
        synth_defaults={
            "harmonic_rolloff": 0.22,
            "n_harmonics": 4,
            "attack": 0.10,
            "decay": 0.30,
            "sustain_level": 0.50,
            "release": 1.8,
        },
    )

    held_dur = 40.0
    for partial in range(2, 9):
        amp_a = max(0.04, 0.16 - partial * 0.01)
        amp_b = max(0.03, 0.14 - partial * 0.01)
        score.add_note(
            "series_a",
            start=0.0,
            duration=held_dur,
            freq=f0_a * partial,
            amp=amp_a,
        )
        score.add_note(
            "series_b",
            start=0.6,
            duration=held_dur - 1.0,
            freq=f0_b * partial,
            amp=amp_b,
        )

    solo_events: list[tuple[float, float, float]] = [
        (6.0, f0_a * 6, 0.26),
        (14.0, f0_a * 8, 0.22),
        (22.0, f0_a * 7, 0.28),
        (30.0, f0_a * 5, 0.30),
        (37.0, f0_a * 4, 0.32),
    ]
    for start, freq, amp in solo_events:
        score.add_note("solo", start=start, duration=4.0, freq=freq, amp=amp)

    return score


def build_spiral_arch_sketch() -> Score:
    """Spiral up 3 JI fifths then back down — arch shape inverts on the return."""
    score = Score(f0=55.0, master_effects=[DELAY_EFFECT, REVERB_EFFECT])
    score.add_voice(
        "melody",
        synth_defaults={
            "harmonic_rolloff": 0.30,
            "attack": 0.06,
            "decay": 0.14,
            "sustain_level": 0.70,
            "release": 0.85,
        },
    )
    score.add_voice(
        "drone",
        synth_defaults={
            "harmonic_rolloff": 0.48,
            "attack": 1.0,
            "decay": 0.25,
            "sustain_level": 0.80,
            "release": 3.5,
        },
    )

    ascending_shape = [1.0, 1.25, 1.5, 1.75, 1.5, 1.25, 1.0]
    descending_shape = [1.75, 1.5, 1.25, 1.0, 1.25, 1.5, 1.75]
    ascending_ioi = [0.90, 0.65, 0.65, 1.50, 0.65, 0.85]
    ascending_durations = [1.10, 0.80, 0.80, 2.00, 0.80, 1.00, 1.20]
    ascending_amps = [0.32, 0.28, 0.30, 0.42, 0.30, 0.28, 0.32]
    descending_ioi = [0.85, 0.65, 1.50, 0.65, 0.65, 0.90]
    descending_durations = [1.20, 1.00, 0.80, 2.00, 0.80, 0.80, 1.10]
    descending_amps = [0.42, 0.32, 0.28, 0.30, 0.30, 0.28, 0.32]

    def ioi_to_onsets(iois: list[float]) -> list[float]:
        onsets = [0.0]
        for interval in iois:
            onsets.append(onsets[-1] + interval)
        return onsets

    ascending_onsets = ioi_to_onsets(ascending_ioi)
    descending_onsets = ioi_to_onsets(descending_ioi)

    section_gap = 2.0
    ascending_phrase_dur = ascending_onsets[-1] + ascending_durations[-1]
    descending_phrase_dur = descending_onsets[-1] + descending_durations[-1]

    ascending_fundamentals = [55.0]
    for _ in range(3):
        ascending_fundamentals.append(ascending_fundamentals[-1] * 3 / 2)
    sections = ascending_fundamentals + ascending_fundamentals[-2::-1]
    shapes = [ascending_shape] * 4 + [descending_shape] * 3
    onsets_per_section = [ascending_onsets] * 4 + [descending_onsets] * 3
    durations_per_section = [ascending_durations] * 4 + [descending_durations] * 3
    amps_per_section = [ascending_amps] * 4 + [descending_amps] * 3
    phrase_durations = [ascending_phrase_dur] * 4 + [descending_phrase_dur] * 3

    cursor = 0.0
    for section_f0, shape, onsets, durations, amps, phrase_dur in zip(
        sections,
        shapes,
        onsets_per_section,
        durations_per_section,
        amps_per_section,
        phrase_durations,
        strict=True,
    ):
        score.add_note(
            "drone",
            start=cursor,
            duration=phrase_dur + 1.5,
            freq=section_f0 * 2.0,
            amp=0.26,
        )
        for ratio, onset, duration, amp in zip(
            shape, onsets, durations, amps, strict=True
        ):
            score.add_note(
                "melody",
                start=cursor + onset,
                duration=duration,
                freq=section_f0 * 4.0 * ratio,
                amp=amp,
            )
        cursor += phrase_dur + section_gap

    return score


def build_interference_v2_sketch() -> Score:
    """Beating texture that shifts gears: slow beating phase, then fast."""
    f0_a = 110.0
    f0_b_slow = 110.5
    f0_b_fast = 113.0

    score = Score(f0=f0_a, master_effects=[SOFT_REVERB_EFFECT])
    score.add_voice(
        "series_a",
        synth_defaults={
            "harmonic_rolloff": 0.40,
            "n_harmonics": 3,
            "attack": 1.8,
            "decay": 0.25,
            "sustain_level": 0.84,
            "release": 5.0,
        },
    )
    score.add_voice(
        "series_b",
        synth_defaults={
            "harmonic_rolloff": 0.38,
            "n_harmonics": 3,
            "attack": 2.0,
            "decay": 0.25,
            "sustain_level": 0.80,
            "release": 5.0,
        },
    )
    score.add_voice(
        "melody",
        synth_defaults={
            "harmonic_rolloff": 0.22,
            "n_harmonics": 4,
            "attack": 0.10,
            "decay": 0.30,
            "sustain_level": 0.50,
            "release": 1.8,
        },
    )

    for partial in range(2, 9):
        amp_a = max(0.04, 0.15 - partial * 0.01)
        score.add_note(
            "series_a", start=0.0, duration=55.0, freq=f0_a * partial, amp=amp_a
        )

    for partial in range(2, 9):
        amp_b = max(0.03, 0.12 - partial * 0.01)
        score.add_note(
            "series_b",
            start=0.5,
            duration=32.0,
            freq=f0_b_slow * partial,
            amp=amp_b,
        )

    for partial in range(2, 9):
        amp_b = max(0.03, 0.12 - partial * 0.01)
        score.add_note(
            "series_b",
            start=25.0,
            duration=30.0,
            freq=f0_b_fast * partial,
            amp=amp_b,
        )

    melody_events: list[tuple[float, float, float]] = [
        (3.0, f0_a * 7, 0.28),
        (7.0, f0_a * 8, 0.24),
        (11.0, f0_a * 9, 0.22),
        (15.0, f0_a * 8, 0.26),
        (19.0, f0_a * 7, 0.28),
        (23.0, f0_a * 6, 0.30),
        (27.0, f0_a * 8, 0.22),
        (31.0, f0_a * 7, 0.26),
        (35.0, f0_a * 6, 0.28),
        (39.0, f0_a * 5, 0.30),
        (44.0, f0_a * 4, 0.34),
    ]
    for start, freq, amp in melody_events:
        score.add_note("melody", start=start, duration=3.5, freq=freq, amp=amp)

    return score


def build_interference_ji_sketch() -> Score:
    """Three JI-related drone series entering in sequence."""
    f0_a = 110.0
    f0_b = 110.0 * 3 / 2
    f0_c = 110.0 * 7 / 4

    score = Score(
        f0=f0_a,
        master_effects=[WARM_SATURATION_EFFECT, SOFT_REVERB_EFFECT],
    )
    score.add_voice(
        "series_a",
        synth_defaults={
            "harmonic_rolloff": 0.45,
            "n_harmonics": 3,
            "attack": 2.0,
            "decay": 0.20,
            "sustain_level": 0.85,
            "release": 5.0,
        },
    )
    score.add_voice(
        "series_b",
        synth_defaults={
            "harmonic_rolloff": 0.40,
            "n_harmonics": 3,
            "attack": 2.5,
            "decay": 0.20,
            "sustain_level": 0.82,
            "release": 5.0,
        },
    )
    score.add_voice(
        "series_c",
        synth_defaults={
            "harmonic_rolloff": 0.35,
            "n_harmonics": 3,
            "attack": 3.0,
            "decay": 0.20,
            "sustain_level": 0.78,
            "release": 5.0,
        },
    )
    score.add_voice(
        "melody",
        synth_defaults={
            "harmonic_rolloff": 0.22,
            "n_harmonics": 4,
            "attack": 0.10,
            "decay": 0.30,
            "sustain_level": 0.50,
            "release": 1.8,
        },
    )

    held_dur = 55.0
    for partial in range(1, 7):
        amp = max(0.04, 0.18 - partial * 0.02)
        score.add_note(
            "series_a", start=0.0, duration=held_dur, freq=f0_a * partial, amp=amp
        )
    for partial in range(1, 5):
        amp = max(0.04, 0.16 - partial * 0.02)
        score.add_note(
            "series_b",
            start=5.0,
            duration=held_dur - 5.0,
            freq=f0_b * partial,
            amp=amp,
        )
    for partial in range(1, 4):
        amp = max(0.03, 0.12 - partial * 0.02)
        score.add_note(
            "series_c",
            start=18.0,
            duration=held_dur - 18.0,
            freq=f0_c * partial,
            amp=amp,
        )

    melody_events: list[tuple[float, float, float]] = [
        (2.0, f0_a * 6, 0.28),
        (6.0, f0_a * 8, 0.24),
        (10.0, f0_a * 7, 0.28),
        (14.0, f0_b * 4, 0.26),
        (18.0, f0_c * 2, 0.26),
        (22.0, f0_a * 8, 0.22),
        (26.0, f0_b * 3, 0.26),
        (30.0, f0_a * 6, 0.28),
        (34.0, f0_c * 3, 0.24),
        (38.0, f0_a * 5, 0.30),
        (42.0, f0_b * 2, 0.28),
        (47.0, f0_a * 4, 0.34),
    ]
    for start, freq, amp in melody_events:
        score.add_note("melody", start=start, duration=4.0, freq=freq, amp=amp)

    return score


PIECES: dict[str, PieceDefinition] = {
    "sketch_spiral": PieceDefinition(
        name="sketch_spiral",
        output_name="11_sketch_spiral.wav",
        build_score=build_spiral_sketch,
    ),
    "sketch_interference": PieceDefinition(
        name="sketch_interference",
        output_name="12_sketch_interference.wav",
        build_score=build_interference_sketch,
    ),
    "sketch_spiral_arch": PieceDefinition(
        name="sketch_spiral_arch",
        output_name="14_sketch_spiral_arch.wav",
        build_score=build_spiral_arch_sketch,
    ),
    "sketch_interference_v2": PieceDefinition(
        name="sketch_interference_v2",
        output_name="15_sketch_interference_v2.wav",
        build_score=build_interference_v2_sketch,
    ),
    "sketch_interference_ji": PieceDefinition(
        name="sketch_interference_ji",
        output_name="16_sketch_interference_ji.wav",
        build_score=build_interference_ji_sketch,
    ),
}
