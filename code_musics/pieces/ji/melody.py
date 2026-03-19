"""`ji_melody` piece builder."""

from __future__ import annotations

from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Score


def build_ji_melody_score() -> Score:
    """A lyrical melodic line in 5-limit JI A major over a bass pedal."""
    f0 = 220.0
    bass_f0 = 110.0

    A3 = f0 * 1
    B3 = f0 * 9 / 8
    Cs4 = f0 * 5 / 4
    D4 = f0 * 4 / 3
    E4 = f0 * 3 / 2
    Fs4 = f0 * 5 / 3
    Gs4 = f0 * 15 / 8
    A4 = f0 * 2
    B4 = f0 * 9 / 4
    Cs5 = f0 * 5 / 2

    score = Score(
        f0=bass_f0,
        master_effects=[
            EffectSpec(
                "reverb",
                {"room_size": 0.60, "damping": 0.50, "wet_level": 0.24},
            ),
            EffectSpec("delay", {"delay_seconds": 0.34, "feedback": 0.16, "mix": 0.10}),
        ],
    )
    score.add_voice(
        "bass",
        synth_defaults={
            "harmonic_rolloff": 0.50,
            "n_harmonics": 8,
            "attack": 1.0,
            "decay": 0.4,
            "sustain_level": 0.80,
            "release": 3.0,
        },
    )
    score.add_voice(
        "bass_fifth",
        synth_defaults={
            "harmonic_rolloff": 0.42,
            "n_harmonics": 6,
            "attack": 1.2,
            "decay": 0.3,
            "sustain_level": 0.72,
            "release": 3.0,
        },
    )
    score.add_voice(
        "melody",
        synth_defaults={
            "harmonic_rolloff": 0.28,
            "n_harmonics": 5,
            "attack": 0.04,
            "decay": 0.12,
            "sustain_level": 0.74,
            "release": 0.45,
        },
    )

    melody_notes: list[tuple[float, float]] = [
        (A4, 1.50),
        (Gs4, 0.75),
        (Fs4, 0.75),
        (E4, 1.125),
        (D4, 0.375),
        (Cs4, 0.75),
        (B3, 0.75),
        (A3, 1.50),
        (B3, 0.375),
        (Cs4, 0.375),
        (D4, 3.00),
        (E4, 0.75),
        (Fs4, 0.75),
        (Gs4, 0.75),
        (A4, 1.50),
        (B4, 0.75),
        (A4, 0.75),
        (Gs4, 1.50),
        (Fs4, 0.75),
        (E4, 3.00),
        (Cs5, 1.50),
        (B4, 0.375),
        (A4, 0.375),
        (Gs4, 0.75),
        (A4, 0.375),
        (Gs4, 0.375),
        (Fs4, 1.50),
        (E4, 0.75),
        (D4, 0.75),
        (Cs4, 0.75),
        (B3, 0.375),
        (A3, 4.50),
    ]

    current_time = 0.0
    for freq, duration in melody_notes:
        score.add_note(
            "melody", start=current_time, duration=duration, freq=freq, amp=0.42
        )
        current_time += duration

    score.add_note(
        "bass", start=0.0, duration=current_time + 1.0, freq=bass_f0, amp=0.32
    )
    score.add_note(
        "bass_fifth",
        start=0.0,
        duration=current_time + 1.0,
        freq=bass_f0 * 3 / 2,
        amp=0.16,
    )
    return score


PIECES: dict[str, PieceDefinition] = {
    "ji_melody": PieceDefinition(
        name="ji_melody",
        output_name="18_ji_melody.wav",
        build_score=build_ji_melody_score,
    )
}
