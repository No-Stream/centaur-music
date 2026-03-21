"""Pieces built to show explicit spectral additive writing."""

from __future__ import annotations

from code_musics.composition import RhythmCell, line
from code_musics.humanize import VelocityHumanizeSpec
from code_musics.pieces._shared import SOFT_REVERB_EFFECT, WARM_SATURATION_EFFECT
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score
from code_musics.spectra import ratio_spectrum


def build_spectral_consonance_score() -> Score:
    """Consonant explicit-spectrum additive piece in a 7/11-flavored JI world."""
    score = Score(
        f0=55.0,
        master_effects=[WARM_SATURATION_EFFECT, SOFT_REVERB_EFFECT],
    )

    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "additive",
            "attack": 0.45,
            "release": 2.6,
            "params": {
                "partials": ratio_spectrum(
                    [1.0, 3 / 2, 7 / 4, 7 / 6],
                    [1.0, 0.34, 0.18, 0.08],
                ),
                "attack_partials": ratio_spectrum(
                    [1.0, 3 / 2, 7 / 4, 7 / 6, 11 / 8],
                    [1.0, 0.38, 0.22, 0.1, 0.06],
                ),
                "spectral_morph_time": 0.18,
                "partial_decay_tilt": 0.18,
                "upper_partial_drift_cents": 1.2,
            },
        },
        pan=-0.06,
    )
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "additive",
            "attack": 0.32,
            "release": 2.8,
            "params": {
                "partials": ratio_spectrum(
                    [1.0, 5 / 4, 3 / 2, 7 / 4, 11 / 8],
                    [1.0, 0.34, 0.24, 0.18, 0.12],
                ),
                "attack_partials": ratio_spectrum(
                    [1.0, 5 / 4, 3 / 2, 7 / 4, 11 / 8, 11 / 4],
                    [1.0, 0.4, 0.3, 0.2, 0.15, 0.08],
                ),
                "spectral_morph_time": 0.28,
                "partial_decay_tilt": 0.26,
                "upper_partial_drift_cents": 1.8,
                "upper_partial_drift_min_ratio": 1.6,
                "unison_voices": 2,
                "detune_cents": 2.0,
            },
        },
        effects=[
            EffectSpec("chorus", {"preset": "juno_subtle", "mix": 0.18}),
        ],
        pan=0.04,
    )
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "additive",
            "attack": 0.03,
            "decay": 0.16,
            "sustain_level": 0.54,
            "release": 0.85,
            "params": {
                "partials": ratio_spectrum(
                    [1.0, 11 / 8, 3 / 2, 7 / 4, 11 / 4],
                    [1.0, 0.3, 0.24, 0.16, 0.08],
                ),
                "attack_partials": ratio_spectrum(
                    [1.0, 9 / 8, 11 / 8, 3 / 2, 7 / 4, 11 / 4],
                    [1.0, 0.24, 0.34, 0.28, 0.2, 0.1],
                ),
                "spectral_morph_time": 0.1,
                "partial_decay_tilt": 0.48,
                "upper_partial_drift_cents": 2.4,
                "upper_partial_drift_min_ratio": 1.4,
            },
        },
        effects=[
            EffectSpec("delay", {"delay_seconds": 0.36, "feedback": 0.2, "mix": 0.14}),
        ],
        velocity_humanize=VelocityHumanizeSpec(seed=17),
        pan=0.14,
    )
    score.add_voice(
        "spark",
        synth_defaults={
            "engine": "additive",
            "preset": "eleven_limit_glass",
            "attack": 0.01,
            "release": 0.6,
        },
        pan=0.22,
    )

    bass_notes = (
        (0.0, 8.0, 1.0, -24.0),
        (8.0, 8.0, 3 / 2, -25.0),
        (16.0, 8.0, 1.0, -24.0),
        (24.0, 10.0, 7 / 4, -25.0),
        (34.0, 8.0, 1.0, -24.0),
        (42.0, 10.0, 3 / 2, -25.5),
    )
    for start, duration, partial, amp_db in bass_notes:
        score.add_note(
            "bass", start=start, duration=duration, partial=partial, amp_db=amp_db
        )

    pad_chords = (
        (2.0, 12.0, (2.0, 5 / 2, 3.0, 7 / 2)),
        (16.0, 8.0, (2.0, 11 / 4, 3.0)),
        (24.0, 10.0, (7 / 4, 11 / 4, 7 / 2)),
        (34.0, 10.0, (2.0, 5 / 2, 3.0, 7 / 2)),
        (42.0, 8.0, (2.0, 11 / 4, 3.0)),
    )
    for start, duration, chord in pad_chords:
        for index, partial in enumerate(chord):
            score.add_note(
                "pad",
                start=start + (index * 0.12),
                duration=duration - (index * 0.08),
                partial=partial,
                amp_db=-23.0 - (index * 1.4),
                velocity=0.88 - (index * 0.06),
            )

    melody_a = line(
        tones=[4.0, 5.0, 6.0, 7.0, 6.0, 5.0, 4.0],
        rhythm=RhythmCell(
            spans=(0.7, 0.7, 0.85, 1.1, 0.8, 0.7, 2.2),
            gates=(0.74, 0.72, 0.8, 0.92, 0.78, 0.72, 1.0),
        ),
        amp_db=-15.0,
    )
    melody_b = line(
        tones=[4.0, 11 / 2, 6.0, 7.0, 11 / 2, 5.0, 4.0],
        rhythm=RhythmCell(
            spans=(0.65, 0.9, 0.7, 1.0, 0.9, 0.7, 2.0),
            gates=(0.72, 0.9, 0.74, 0.88, 0.9, 0.72, 1.0),
        ),
        amp_db=-14.0,
    )
    melody_c = line(
        tones=[6.0, 7.0, 11 / 2, 6.0, 5.0, 4.0],
        rhythm=RhythmCell(
            spans=(0.7, 0.95, 1.15, 0.8, 0.7, 3.1),
            gates=(0.72, 0.86, 0.94, 0.76, 0.74, 1.0),
        ),
        amp_db=-15.5,
    )

    score.add_phrase("lead", melody_a, start=5.5)
    score.add_phrase("lead", melody_b, start=18.0, amp_scale=1.04)
    score.add_phrase("lead", melody_c, start=27.5, amp_scale=0.96)
    score.add_phrase("lead", melody_b, start=36.0, amp_scale=0.94)

    spark_notes = (
        (12.4, 0.6, 7.0, -24.0),
        (14.1, 0.5, 11 / 2, -25.5),
        (21.0, 0.5, 11 / 2, -24.5),
        (23.4, 0.55, 7.0, -25.0),
        (31.2, 0.6, 6.0, -24.8),
        (39.6, 0.48, 11 / 2, -24.5),
        (47.0, 0.7, 7.0, -25.2),
    )
    for start, duration, partial, amp_db in spark_notes:
        score.add_note(
            "spark", start=start, duration=duration, partial=partial, amp_db=amp_db
        )

    return score


PIECES: dict[str, PieceDefinition] = {
    "spectral_consonance": PieceDefinition(
        name="spectral_consonance",
        output_name="spectral_consonance",
        build_score=build_spectral_consonance_score,
        sections=(
            PieceSection("Arrival", 0.0, 16.0),
            PieceSection("Opening", 16.0, 34.0),
            PieceSection("Return", 34.0, 52.0),
        ),
    ),
}
