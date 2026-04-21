"""Counterpoint-leaning and variation-based study pieces."""

from __future__ import annotations

from code_musics.pieces._shared import DELAY_EFFECT, REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import Phrase, Score
from code_musics.tuning import utonal


def build_passacaglia_sketch() -> Score:
    """Descending bass ostinato (8-7-6-5-4) with five accumulating variations."""
    score = Score(f0_hz=55.0, master_effects=[DELAY_EFFECT, REVERB_EFFECT])
    score.add_voice(
        "bass",
        synth_defaults={
            "harmonic_rolloff": 0.52,
            "attack": 0.04,
            "decay": 0.12,
            "sustain_level": 0.85,
            "release": 0.75,
        },
    )
    score.add_voice(
        "upper",
        synth_defaults={
            "harmonic_rolloff": 0.28,
            "attack": 0.07,
            "decay": 0.14,
            "sustain_level": 0.68,
            "release": 0.95,
        },
    )
    score.add_voice(
        "mid",
        synth_defaults={
            "harmonic_rolloff": 0.36,
            "attack": 0.80,
            "decay": 0.20,
            "sustain_level": 0.75,
            "release": 2.5,
        },
    )
    score.add_voice(
        "alto",
        synth_defaults={
            "harmonic_rolloff": 0.34,
            "attack": 0.05,
            "decay": 0.10,
            "sustain_level": 0.73,
            "release": 0.60,
        },
    )

    ground = Phrase.from_partials(
        [8, 7, 6, 5, 4],
        duration=2.2,
        onset_interval=2.0,
        amp=0.44,
        synth_defaults={
            "harmonic_rolloff": 0.52,
            "attack": 0.04,
            "decay": 0.12,
            "sustain_level": 0.85,
            "release": 0.75,
        },
    )
    ground_dur = ground.duration

    for repetition in range(5):
        score.add_phrase("bass", ground, start=repetition * ground_dur)

    upper_phrase = Phrase.from_partials(
        [12, 14, 13, 12, 11, 12],
        duration=1.6,
        onset_interval=1.45,
        amp=0.28,
        synth_defaults={
            "harmonic_rolloff": 0.26,
            "attack": 0.07,
            "decay": 0.14,
            "sustain_level": 0.66,
            "release": 0.95,
        },
    )
    score.add_phrase("upper", upper_phrase, start=ground_dur)
    score.add_phrase("upper", upper_phrase, start=ground_dur * 2, partial_shift=2.0)

    for partial, offset in [(5, 0.0), (6, 1.0), (7, 2.2)]:
        score.add_note(
            "mid",
            start=ground_dur * 2 + offset,
            duration=8.5,
            partial=partial,
            amp=0.20,
        )

    alto_phrase = Phrase.from_partials(
        [9, 10, 9, 8, 10, 9],
        duration=1.4,
        onset_interval=1.2,
        amp=0.32,
        synth_defaults={
            "harmonic_rolloff": 0.34,
            "attack": 0.05,
            "decay": 0.10,
            "sustain_level": 0.74,
            "release": 0.58,
        },
    )
    score.add_phrase("alto", alto_phrase, start=ground_dur * 3)
    score.add_phrase(
        "alto",
        alto_phrase,
        start=ground_dur * 3 + 5.0,
        partial_shift=-1.0,
        amp_scale=0.80,
    )

    score.add_phrase(
        "upper",
        upper_phrase,
        start=ground_dur * 4,
        partial_shift=4.0,
        amp_scale=0.68,
    )
    score.add_phrase("alto", alto_phrase, start=ground_dur * 4, amp_scale=1.05)
    for partial, offset in [(8, 0.0), (10, 0.7), (12, 1.4), (7, 0.4)]:
        score.add_note(
            "mid",
            start=ground_dur * 4 + offset,
            duration=10.5,
            partial=partial,
            amp=0.14,
        )

    return score


def build_invention_sketch() -> Score:
    """Two-voice imitative counterpoint on a six-note JI subject."""
    score = Score(f0_hz=110.0, master_effects=[DELAY_EFFECT, REVERB_EFFECT])
    score.add_voice(
        "voice_a",
        synth_defaults={
            "harmonic_rolloff": 0.34,
            "attack": 0.05,
            "decay": 0.12,
            "sustain_level": 0.72,
            "release": 0.65,
        },
    )
    score.add_voice(
        "voice_b",
        synth_defaults={
            "harmonic_rolloff": 0.30,
            "attack": 0.06,
            "decay": 0.14,
            "sustain_level": 0.68,
            "release": 0.70,
        },
    )
    score.add_voice(
        "pedal",
        synth_defaults={
            "harmonic_rolloff": 0.48,
            "attack": 0.90,
            "decay": 0.20,
            "sustain_level": 0.80,
            "release": 3.0,
        },
    )

    subject = Phrase.from_partials(
        [6, 7, 8, 9, 8, 7],
        duration=1.3,
        onset_interval=1.0,
        amp=0.38,
        synth_defaults={
            "harmonic_rolloff": 0.34,
            "attack": 0.05,
            "decay": 0.12,
            "sustain_level": 0.72,
            "release": 0.65,
        },
    )
    subject_dur = subject.duration

    score.add_phrase("voice_a", subject, start=0.0)
    answer_start = 4.0
    score.add_phrase("voice_b", subject, start=answer_start, partial_shift=4)

    head = Phrase.from_partials(
        [6, 7, 8, 9],
        duration=1.1,
        onset_interval=0.85,
        amp=0.35,
        synth_defaults={
            "harmonic_rolloff": 0.34,
            "attack": 0.05,
            "decay": 0.10,
            "sustain_level": 0.70,
            "release": 0.55,
        },
    )
    tail = Phrase.from_partials(
        [9, 8, 7, 6],
        duration=1.1,
        onset_interval=0.85,
        amp=0.33,
        synth_defaults={
            "harmonic_rolloff": 0.34,
            "attack": 0.05,
            "decay": 0.10,
            "sustain_level": 0.70,
            "release": 0.55,
        },
    )

    development_start = answer_start + subject_dur + 1.0
    score.add_phrase("voice_a", head, start=development_start)
    score.add_phrase("voice_b", tail, start=development_start + 0.6, partial_shift=3)
    score.add_phrase(
        "voice_b",
        head,
        start=development_start + head.duration + 0.8,
        partial_shift=4,
    )
    score.add_phrase(
        "voice_a",
        tail,
        start=development_start + head.duration + 1.4,
        partial_shift=-1,
    )

    stretto_start = development_start + head.duration + tail.duration + 2.0
    score.add_phrase("voice_a", subject, start=stretto_start)
    score.add_phrase("voice_b", subject, start=stretto_start + 2.0, partial_shift=4)

    score.add_note(
        "pedal",
        start=stretto_start - 1.0,
        duration=subject_dur + 3.5,
        partial=4,
        amp=0.24,
    )
    score.add_note(
        "pedal",
        start=stretto_start - 1.0,
        duration=subject_dur + 3.5,
        partial=6,
        amp=0.16,
    )

    return score


def build_variations_sketch() -> Score:
    """One JI theme heard through five transform lenses."""
    f0 = 110.0
    score = Score(f0_hz=f0, master_effects=[DELAY_EFFECT, REVERB_EFFECT])
    score.add_voice(
        "melody",
        synth_defaults={
            "harmonic_rolloff": 0.32,
            "attack": 0.06,
            "decay": 0.12,
            "sustain_level": 0.72,
            "release": 0.75,
        },
    )
    score.add_voice(
        "harmony",
        synth_defaults={
            "harmonic_rolloff": 0.40,
            "attack": 0.60,
            "decay": 0.20,
            "sustain_level": 0.70,
            "release": 2.5,
        },
    )

    theme = Phrase.from_partials(
        [4, 5, 6, 7, 8, 7, 6],
        duration=1.1,
        onset_interval=0.9,
        amp=0.38,
        synth_defaults={
            "harmonic_rolloff": 0.32,
            "attack": 0.06,
            "decay": 0.12,
            "sustain_level": 0.72,
            "release": 0.75,
        },
    )
    theme_dur = theme.duration
    gap = 2.0
    cursor = 0.0

    score.add_phrase("melody", theme, start=cursor)
    cursor += theme_dur + gap
    score.add_phrase("melody", theme, start=cursor, time_scale=2.0)
    cursor += theme_dur * 2.0 + gap
    score.add_phrase("melody", theme, start=cursor, reverse=True)
    cursor += theme_dur + gap
    score.add_phrase("melody", theme, start=cursor, partial_shift=4.0)
    cursor += theme_dur + gap

    score.add_phrase("melody", theme, start=cursor)
    for freq in utonal(f0 * 8.0, [4, 5, 6, 7]):
        score.add_note(
            "harmony",
            start=cursor,
            duration=theme_dur + 1.5,
            freq=freq,
            amp=0.16,
        )
    cursor += theme_dur + gap

    score.add_phrase("melody", theme, start=cursor, time_scale=0.5)
    score.add_phrase(
        "melody",
        theme,
        start=cursor,
        time_scale=0.5,
        partial_shift=4.0,
        amp_scale=0.65,
    )

    return score


PIECES: dict[str, PieceDefinition] = {
    "sketch_passacaglia": PieceDefinition(
        name="sketch_passacaglia",
        output_name="07_sketch_passacaglia",
        build_score=build_passacaglia_sketch,
    ),
    "sketch_invention": PieceDefinition(
        name="sketch_invention",
        output_name="08_sketch_invention",
        build_score=build_invention_sketch,
    ),
    "sketch_variations": PieceDefinition(
        name="sketch_variations",
        output_name="10_sketch_variations",
        build_score=build_variations_sketch,
    ),
}
