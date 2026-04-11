"""Euclidean rhythm study — layered polyrhythmic patterns over JI pitch material.

Three voices with interlocking euclidean patterns at different densities:
  1. bass     — E(3,8) on a filtered_stack round_bass, cycling through a
                 root-fifth-septimal-seventh figure (1/1, 3/2, 7/4)
  2. melody   — E(5,8) on an fm bell, cycling harmonic partials 8-13
                 for an open, slightly alien colour
  3. tick     — E(7,16) on noise_perc tick, high and tight rhythmic texture

All three run at the same step grid (~130 BPM sixteenths) so the euclidean
interference is audible as interlocking accent patterns.  Phrases repeat via
sequence() to fill ~20 seconds.
"""

from __future__ import annotations

from code_musics.composition import HarmonicContext, concat, sequence
from code_musics.generative.euclidean import euclidean_line
from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    VelocityHumanizeSpec,
)
from code_musics.pieces._shared import SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Score, SendBusSpec, VoiceSend

BPM = 130.0
SIXTEENTH = 60.0 / BPM / 4.0
F0 = 55.0

BASS_SYNTH: dict = {
    "engine": "filtered_stack",
    "preset": "round_bass",
    "env": {
        "attack_ms": 8.0,
        "decay_ms": 300.0,
        "sustain_ratio": 0.4,
        "release_ms": 200.0,
    },
}

MELODY_SYNTH: dict = {
    "engine": "fm",
    "preset": "bell",
    "env": {
        "attack_ms": 3.0,
        "decay_ms": 400.0,
        "sustain_ratio": 0.15,
        "release_ms": 350.0,
    },
}

TICK_SYNTH: dict = {
    "engine": "noise_perc",
    "preset": "tick",
}

BASS_TONES: list[float] = [1.0, 3 / 2, 7 / 4]
MELODY_TONES: list[float] = [p / 8 for p in [8, 9, 10, 11, 13]]
TICK_TONES: list[float] = [4.0]


def build_study_euclidean_score() -> Score:
    ctx = HarmonicContext(tonic=F0, name="root")

    bass_phrase_a = euclidean_line(
        BASS_TONES,
        hits=3,
        steps=8,
        span=SIXTEENTH,
        pitch_kind="freq",
        gate=0.85,
        amp_db=-6.0,
        synth=BASS_SYNTH,
        context=ctx,
    )
    bass_phrase_b = euclidean_line(
        BASS_TONES,
        hits=3,
        steps=8,
        span=SIXTEENTH,
        rotation=2,
        pitch_kind="freq",
        gate=0.85,
        amp_db=-6.0,
        synth=BASS_SYNTH,
        context=ctx.drifted(by_ratio=4 / 3, name="IV"),
    )
    bass_phrase = concat(bass_phrase_a, bass_phrase_b)

    melody_phrase_a = euclidean_line(
        MELODY_TONES,
        hits=5,
        steps=8,
        span=SIXTEENTH,
        pitch_kind="freq",
        gate=0.7,
        amp_db=-8.0,
        synth=MELODY_SYNTH,
        context=ctx,
    )
    melody_phrase_b = euclidean_line(
        MELODY_TONES,
        hits=5,
        steps=8,
        span=SIXTEENTH,
        rotation=1,
        pitch_kind="freq",
        gate=0.7,
        amp_db=-8.0,
        synth=MELODY_SYNTH,
        context=ctx.drifted(by_ratio=4 / 3, name="IV"),
    )
    melody_phrase = concat(melody_phrase_a, melody_phrase_b)

    tick_phrase = euclidean_line(
        TICK_TONES,
        hits=7,
        steps=16,
        span=SIXTEENTH,
        pitch_kind="partial",
        gate=0.5,
        amp_db=-10.0,
        synth=TICK_SYNTH,
    )

    bar_dur = SIXTEENTH * 16
    repetitions = 5
    starts = [i * bar_dur for i in range(repetitions)]

    score = Score(
        f0=F0,
        timing_humanize=TimingHumanizeSpec(preset="tight_ensemble", seed=42),
        send_buses=[
            SendBusSpec(
                name="room",
                effects=[SOFT_REVERB_EFFECT],
            ),
        ],
    )

    score.add_voice(
        "bass",
        synth_defaults=BASS_SYNTH,
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living", seed=10),
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog", seed=11),
        pan=-0.3,
        mix_db=-2.0,
        sends=[VoiceSend(target="room", send_db=-8.0)],
    )

    score.add_voice(
        "melody",
        synth_defaults=MELODY_SYNTH,
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living", seed=20),
        envelope_humanize=EnvelopeHumanizeSpec(preset="loose_pluck", seed=21),
        pan=0.35,
        mix_db=-1.0,
        effects=[
            EffectSpec(
                "delay", {"delay_seconds": SIXTEENTH * 3, "feedback": 0.25, "mix": 0.18}
            )
        ],
        sends=[VoiceSend(target="room", send_db=-6.0)],
    )

    score.add_voice(
        "tick",
        synth_defaults=TICK_SYNTH,
        normalize_peak_db=-6.0,
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living", seed=30),
        pan=0.1,
        mix_db=-4.0,
        sends=[VoiceSend(target="room", send_db=-12.0)],
    )

    sequence(score, "bass", bass_phrase, starts=starts)
    sequence(score, "melody", melody_phrase, starts=starts)
    sequence(score, "tick", tick_phrase, starts=starts)

    return score


PIECES: dict[str, PieceDefinition] = {
    "study_euclidean": PieceDefinition(
        name="study_euclidean",
        output_name="study_euclidean",
        build_score=build_study_euclidean_score,
        study=True,
    ),
}
