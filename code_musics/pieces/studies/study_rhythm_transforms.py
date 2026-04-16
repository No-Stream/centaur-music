"""Rhythm transforms study — motif development through canonic texture.

A single two-bar melodic motif is developed through every rhythmic phrase
transform, building polyphonic texture like a canonic fugue:

  Bars  1-8:   Statement — original motif repeated on harpsichord + CA percussion
  Bars  9-16:  Augmentation — organ enters at half speed
  Bars 17-22:  Diminution — FM glass lead enters at double speed
  Bars 23-30:  Retro+Displace — piano (retrograde) and additive pad (displaced)
  Bars 31-38:  Rotation — thinned to rotated variants on harpsichord
  Bars 39-40:  Coda — original statement alone

~110 s at 108 BPM, straight time (rhythmic interest comes from the transforms
themselves rather than groove).  Harmonic material is a descending harmonic-
series line in JI.
"""

from __future__ import annotations

from code_musics.composition import (
    augment,
    diminish,
    displace,
    grid_line,
    line,
    rhythmic_retrograde,
    rotate,
    sequence,
)
from code_musics.generative.ca_rhythm import ca_rhythm_layers
from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    VelocityHumanizeSpec,
)
from code_musics.meter import E, Q, S, Timeline
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS, SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Score, SendBusSpec, VoiceSend

# ── Timeline ──────────────────────────────────────────────────────────

BPM = 108.0
F0 = 55.0  # A1
TL = Timeline(bpm=BPM)

BAR_DUR = TL.measures(1)

# ── Motif ─────────────────────────────────────────────────────────────
# 2 bars of 4/4 — descending harmonic series line with rhythmic variety.

MOTIF_TONES: list[float] = [1.0, 9 / 8, 5 / 4, 3 / 2, 7 / 4, 5 / 3, 3 / 2, 1.0]
MOTIF_DURATIONS = [Q, E, E, Q, E, E, Q, Q]

# ── Synth voices ──────────────────────────────────────────────────────

V1_SYNTH: dict = {"engine": "harpsichord", "preset": "baroque"}
V2_SYNTH: dict = {"engine": "organ", "preset": "warm"}
V3_SYNTH: dict = {"engine": "fm", "preset": "glass_lead"}
V4_SYNTH: dict = {"engine": "piano", "preset": "warm"}
V5_SYNTH: dict = {"engine": "additive", "preset": "soft_pad"}

# Percussion voices — three CA layers with distinct timbres.
PERC_HAT_SYNTH: dict = {"engine": "drum_voice", "preset": "closed_hat"}
PERC_TICK_SYNTH: dict = {"engine": "drum_voice", "preset": "tick"}
PERC_COWBELL_SYNTH: dict = {
    "engine": "drum_voice",
    "preset": "cowbell",
    "env": {
        "attack_ms": 0.5,
        "decay_ms": 200.0,
        "sustain_ratio": 0.0,
        "release_ms": 80.0,
    },
}


def _bar_start(bar: int) -> float:
    """Return the absolute time (seconds) at the start of a 1-indexed bar."""
    return TL.at(bar=bar)


def build_study_rhythm_transforms_score() -> Score:
    # ── Build motif and transforms ────────────────────────────────────
    motif = grid_line(
        MOTIF_TONES,
        MOTIF_DURATIONS,
        timeline=TL,
        pitch_kind="partial",
        amp_db=-6.0,
        synth_defaults=V1_SYNTH,
    )

    motif_aug = augment(motif, 2.0)
    motif_dim = diminish(motif, 2.0)
    motif_retro = rhythmic_retrograde(motif)
    motif_disp = displace(motif, TL.duration(E))
    motif_rot1 = rotate(motif, 1)
    motif_rot2 = rotate(motif, 2)
    motif_rot3 = rotate(motif, 3)

    # ── CA percussion layers ──────────────────────────────────────────
    ca_layers = ca_rhythm_layers(rule=30, steps=16, layers=3, span=TL.duration(S))

    # ── Score setup ───────────────────────────────────────────────────
    score = Score(
        f0_hz=F0,
        timing_humanize=TimingHumanizeSpec(preset="chamber", seed=42),
        master_effects=DEFAULT_MASTER_EFFECTS,
        send_buses=[
            SendBusSpec(name="room", effects=[SOFT_REVERB_EFFECT]),
        ],
    )

    # ── Tonal voices ──────────────────────────────────────────────────

    score.add_voice(
        "v1_harpsichord",
        synth_defaults=V1_SYNTH,
        normalize_lufs=-24.0,
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living", seed=10),
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog", seed=11),
        pan=0.0,
        mix_db=0.0,
        sends=[VoiceSend(target="room", send_db=-8.0)],
    )

    score.add_voice(
        "v2_organ",
        synth_defaults=V2_SYNTH,
        normalize_lufs=-24.0,
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living", seed=20),
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog", seed=21),
        pan=-0.3,
        mix_db=-1.0,
        sends=[VoiceSend(target="room", send_db=-6.0)],
    )

    score.add_voice(
        "v3_fm",
        synth_defaults=V3_SYNTH,
        normalize_lufs=-24.0,
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living", seed=30),
        envelope_humanize=EnvelopeHumanizeSpec(preset="loose_pluck", seed=31),
        pan=0.3,
        mix_db=-1.0,
        effects=[
            EffectSpec(
                "delay", {"delay_seconds": TL.duration(E), "feedback": 0.2, "mix": 0.14}
            ),
        ],
        sends=[VoiceSend(target="room", send_db=-8.0)],
    )

    score.add_voice(
        "v4_piano",
        synth_defaults=V4_SYNTH,
        normalize_lufs=-24.0,
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living", seed=40),
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog", seed=41),
        pan=-0.5,
        mix_db=-2.0,
        sends=[VoiceSend(target="room", send_db=-6.0)],
    )

    score.add_voice(
        "v5_pad",
        synth_defaults=V5_SYNTH,
        normalize_lufs=-24.0,
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living", seed=50),
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog", seed=51),
        pan=0.5,
        mix_db=-2.0,
        sends=[VoiceSend(target="room", send_db=-4.0)],
    )

    # ── Percussion voices ─────────────────────────────────────────────

    score.add_voice(
        "perc_hat",
        synth_defaults=PERC_HAT_SYNTH,
        normalize_peak_db=-6.0,
        pan=-0.2,
        mix_db=-6.0,
        sends=[VoiceSend(target="room", send_db=-14.0)],
    )

    score.add_voice(
        "perc_tick",
        synth_defaults=PERC_TICK_SYNTH,
        normalize_peak_db=-6.0,
        pan=0.2,
        mix_db=-8.0,
        sends=[VoiceSend(target="room", send_db=-16.0)],
    )

    score.add_voice(
        "perc_bell",
        synth_defaults=PERC_COWBELL_SYNTH,
        normalize_peak_db=-6.0,
        pan=0.4,
        mix_db=-10.0,
        sends=[VoiceSend(target="room", send_db=-12.0)],
    )

    # ── Section placement ─────────────────────────────────────────────
    #
    # Motif is 2 bars long; compute repetition starts per section.

    motif_dur = motif.duration

    # Section 1: Statement (bars 1-8) — V1 original × 4
    sec1_starts = [_bar_start(bar) for bar in range(1, 8, 2)]
    sequence(score, "v1_harpsichord", motif, starts=sec1_starts)

    # Section 2: Augmentation (bars 9-16) — V1 continues, V2 augmented × 2
    sec2_v1_starts = [_bar_start(bar) for bar in range(9, 16, 2)]
    sequence(score, "v1_harpsichord", motif, starts=sec2_v1_starts)

    # Augmented motif is 4 bars long → fits twice in 8 bars
    sec2_v2_starts = [_bar_start(9), _bar_start(13)]
    sequence(score, "v2_organ", motif_aug, starts=sec2_v2_starts)

    # Section 3: Diminution (bars 17-22) — V1 + V2 + V3 diminished
    sec3_v1_starts = [_bar_start(bar) for bar in range(17, 22, 2)]
    sequence(score, "v1_harpsichord", motif, starts=sec3_v1_starts)

    # Augmented motif spans 4 bars; place one starting at bar 17
    sec3_v2_starts = [_bar_start(17)]
    sequence(score, "v2_organ", motif_aug, starts=sec3_v2_starts)

    # Diminished motif is 1 bar long → fits 6 times in 6 bars
    sec3_v3_starts = [_bar_start(17) + i * (motif_dur / 2.0) for i in range(6)]
    sequence(score, "v3_fm", motif_dim, starts=sec3_v3_starts)

    # Section 4: Retrograde + Displace (bars 23-30) — peak texture, all 5 voices
    sec4_v1_starts = [_bar_start(bar) for bar in range(23, 30, 2)]
    sequence(score, "v1_harpsichord", motif, starts=sec4_v1_starts)

    sec4_v2_starts = [_bar_start(23), _bar_start(27)]
    sequence(score, "v2_organ", motif_aug, starts=sec4_v2_starts)

    sec4_v3_starts = [_bar_start(23) + i * (motif_dur / 2.0) for i in range(8)]
    sequence(score, "v3_fm", motif_dim, starts=sec4_v3_starts)

    sec4_v4_starts = [_bar_start(bar) for bar in range(23, 30, 2)]
    sequence(score, "v4_piano", motif_retro, starts=sec4_v4_starts)

    sec4_v5_starts = [_bar_start(bar) for bar in range(23, 30, 2)]
    sequence(score, "v5_pad", motif_disp, starts=sec4_v5_starts)

    # Section 5: Rotation (bars 31-38) — thin to 2 voices with rotated variants
    sec5_starts_1 = [_bar_start(31), _bar_start(33)]
    sec5_starts_2 = [_bar_start(35), _bar_start(37)]
    sequence(score, "v1_harpsichord", motif_rot1, starts=sec5_starts_1)
    sequence(score, "v1_harpsichord", motif_rot2, starts=sec5_starts_2)
    sequence(score, "v3_fm", motif_rot3, starts=[_bar_start(31), _bar_start(35)])

    # Section 6: Coda (bars 39-40) — V1 alone, original statement
    sequence(score, "v1_harpsichord", motif, starts=[_bar_start(39)])

    # ── CA percussion placement ───────────────────────────────────────
    #
    # Each CA layer runs from bar 1 through bar 38 (percussion drops out
    # in the coda).  The RhythmCell durations cycle through repetitions.

    perc_voice_names = ["perc_hat", "perc_tick", "perc_bell"]
    perc_tones: list[list[float]] = [
        [8.0],  # high partial for hat
        [4.0],  # mid partial for tick
        [6.0],  # partial for cowbell
    ]

    perc_end_bar = 39
    for layer_idx, (ca_cell, voice_name, tones) in enumerate(
        zip(ca_layers, perc_voice_names, perc_tones, strict=True)
    ):
        cell_dur = sum(ca_cell.spans)
        n_reps = max(1, int((_bar_start(perc_end_bar) - _bar_start(1)) / cell_dur))
        perc_starts = [_bar_start(1) + i * cell_dur for i in range(n_reps)]

        perc_phrase = line(
            tones * len(ca_cell.spans),
            ca_cell,
            pitch_kind="partial",
            amp_db=-10.0 - layer_idx * 2.0,
        )
        sequence(score, voice_name, perc_phrase, starts=perc_starts)

    return score


PIECES: dict[str, PieceDefinition] = {
    "study_rhythm_transforms": PieceDefinition(
        name="study_rhythm_transforms",
        output_name="study_rhythm_transforms",
        build_score=build_study_rhythm_transforms_score,
        study=True,
    ),
}
