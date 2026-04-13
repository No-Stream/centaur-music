"""Harmonic-space and arpeggio-oriented study pieces."""

from __future__ import annotations

from code_musics.pieces._shared import SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import Score


def build_arpeggios_sketch() -> Score:
    """Sparse high-partial melody drifting downward — simple and tender."""
    score = Score(f0=55.0, master_effects=[SOFT_REVERB_EFFECT])
    score.add_voice(
        "drone",
        synth_defaults={
            "harmonic_rolloff": 0.60,
            "n_harmonics": 3,
            "attack": 2.5,
            "decay": 0.30,
            "sustain_level": 0.65,
            "release": 6.0,
        },
    )
    score.add_voice(
        "solo",
        synth_defaults={
            "harmonic_rolloff": 0.18,
            "n_harmonics": 4,
            "attack": 0.12,
            "decay": 0.40,
            "sustain_level": 0.30,
            "release": 3.5,
        },
    )

    score.add_note(
        "drone", start=0.0, duration=72.0, partial=1.0, amp=0.15, label="root"
    )
    score.add_note(
        "drone", start=8.0, duration=57.0, partial=2.0, amp=0.08, label="octave"
    )

    melody_events: list[tuple[float, float, float]] = [
        (0.0, 14, 0.28),
        (5.5, 12, 0.30),
        (10.5, 14, 0.22),
        (15.5, 12, 0.28),
        (20.0, 10, 0.32),
        (24.5, 9, 0.30),
        (29.5, 12, 0.20),
        (34.0, 10, 0.26),
        (38.5, 9, 0.28),
        (43.0, 8, 0.32),
        (47.5, 10, 0.24),
        (52.0, 9, 0.28),
        (56.5, 8, 0.30),
        (61.0, 7, 0.34),
        (65.5, 8, 0.26),
        (69.5, 6, 0.36),
    ]
    for start, partial, amp in melody_events:
        score.add_note("solo", start=start, duration=5.0, partial=partial, amp=amp)

    return score


def build_arpeggios_cross_sketch() -> Score:
    """Two voices in contrary motion: one descends, one ascends, weaving JI chords."""
    score = Score(f0=55.0, master_effects=[SOFT_REVERB_EFFECT])
    shared_synth: dict = {
        "harmonic_rolloff": 0.18,
        "n_harmonics": 4,
        "attack": 0.10,
        "decay": 0.35,
        "sustain_level": 0.28,
        "release": 3.5,
    }
    score.add_voice(
        "drone",
        synth_defaults={
            "harmonic_rolloff": 0.60,
            "n_harmonics": 3,
            "attack": 2.5,
            "decay": 0.30,
            "sustain_level": 0.65,
            "release": 6.0,
        },
    )
    score.add_voice("voice_a", synth_defaults=shared_synth)
    score.add_voice(
        "voice_b",
        synth_defaults={**shared_synth, "harmonic_rolloff": 0.22},
    )

    score.add_note("drone", start=0.0, duration=44.0, partial=1.0, amp=0.12)
    score.add_note("drone", start=4.0, duration=38.0, partial=2.0, amp=0.06)

    note_dur = 4.5
    voice_a_events: list[tuple[float, float, float]] = [
        (0.0, 14, 0.26),
        (2.5, 12, 0.28),
        (5.0, 14, 0.22),
        (7.5, 11, 0.26),
        (10.0, 10, 0.28),
        (12.5, 9, 0.30),
        (15.0, 11, 0.22),
        (17.5, 10, 0.26),
        (20.0, 9, 0.26),
        (22.5, 8, 0.30),
        (25.0, 10, 0.22),
        (27.5, 9, 0.26),
        (30.0, 8, 0.28),
        (32.5, 7, 0.32),
        (35.0, 8, 0.24),
        (38.0, 6, 0.34),
    ]
    voice_b_events: list[tuple[float, float, float]] = [
        (1.2, 6, 0.28),
        (3.7, 7, 0.26),
        (6.2, 6, 0.24),
        (8.7, 8, 0.28),
        (11.2, 7, 0.28),
        (13.7, 9, 0.26),
        (16.2, 8, 0.24),
        (18.7, 10, 0.26),
        (21.2, 9, 0.26),
        (23.7, 10, 0.26),
        (26.2, 9, 0.24),
        (28.7, 11, 0.26),
        (31.2, 10, 0.24),
        (33.7, 12, 0.28),
        (36.2, 11, 0.24),
        (39.2, 14, 0.30),
    ]

    for start, partial, amp in voice_a_events:
        score.add_note(
            "voice_a", start=start, duration=note_dur, partial=partial, amp=amp
        )
    for start, partial, amp in voice_b_events:
        score.add_note(
            "voice_b", start=start, duration=note_dur, partial=partial, amp=amp
        )

    return score


PIECES: dict[str, PieceDefinition] = {
    "sketch_arpeggios": PieceDefinition(
        name="sketch_arpeggios",
        output_name="09_sketch_arpeggios",
        build_score=build_arpeggios_sketch,
    ),
    "sketch_arpeggios_cross": PieceDefinition(
        name="sketch_arpeggios_cross",
        output_name="13_sketch_arpeggios_cross",
        build_score=build_arpeggios_cross_sketch,
    ),
}
