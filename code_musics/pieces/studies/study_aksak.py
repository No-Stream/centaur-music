"""Aksak meter study — "Unequal Pulses"

Balkan 7/8 asymmetric meter (2+2+3) meets septimal JI.  ~105 s at 140
pulses/min.  The piece grows from sparse kick downbeats through full
groove with cross-rhythmic counterpoint, evolving mutated hat patterns,
a polyrhythmic climax, and a spacious release.

Section map (35 aksak bars):
  Bars  1-4   Pulse     — Sparse kick on group downbeats (2+2+3)
  Bars  5-8   Build     — Metallic perc on all pulses, bass enters
  Bars  9-16  Groove    — Full texture, melody + cross-rhythm counter layer
  Bars 17-24  Mutation  — mutate_rhythm evolves drum patterns
  Bars 25-30  Poly peak — polyrhythm(3,7) creates maximum rhythmic tension
  Bars 31-35  Release   — Elements thin, long reverb tail
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from code_musics.composition import cross_rhythm, line, polyrhythm, sequence
from code_musics.generative.aksak import AksakPattern
from code_musics.generative.mutation import mutate_rhythm
from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    VelocityHumanizeSpec,
)
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS, SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Score, SendBusSpec, VoiceSend

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Timing grid
# ---------------------------------------------------------------------------

PULSE = 60.0 / 140.0  # one pulse at 140 bpm
AKSAK = AksakPattern.balkan_7(pulse=PULSE)  # (2, 2, 3) grouping
BAR_DUR = AKSAK.total_duration  # 7 pulses

# Section boundaries (bar indices → seconds)
SEC_PULSE_START = 0.0  # bar 1
SEC_BUILD_START = 4 * BAR_DUR  # bar 5
SEC_GROOVE_START = 8 * BAR_DUR  # bar 9
SEC_MUTATION_START = 16 * BAR_DUR  # bar 17
SEC_POLY_START = 24 * BAR_DUR  # bar 25
SEC_RELEASE_START = 30 * BAR_DUR  # bar 31
SEC_END = 35 * BAR_DUR  # bar 36 (end)

# ---------------------------------------------------------------------------
# Pitch material — 7-limit JI
# ---------------------------------------------------------------------------

F0 = 110.0
SCALE_RATIOS: list[float] = [1 / 1, 7 / 6, 5 / 4, 3 / 2, 7 / 4]

# Frequency pool derived from F0
BASS_TONES: list[float] = [F0 * r for r in [1 / 1, 3 / 2, 7 / 4]]
MELODY_TONES: list[float] = [F0 * r for r in SCALE_RATIOS]
COUNTER_TONES: list[float] = [F0 * r for r in [7 / 6, 5 / 4, 3 / 2]]
PAD_TONES: list[float] = [F0 * r for r in [1 / 1, 5 / 4, 3 / 2]]

# ---------------------------------------------------------------------------
# Synth definitions
# ---------------------------------------------------------------------------

KICK_SYNTH: dict = {"engine": "kick_tom", "preset": "909_house"}

METAL_SYNTH: dict = {"engine": "metallic_perc", "preset": "closed_hat"}

BASS_SYNTH: dict = {
    "engine": "polyblep",
    "preset": "moog_bass",
    "env": {
        "attack_ms": 5.0,
        "decay_ms": 250.0,
        "sustain_ratio": 0.5,
        "release_ms": 120.0,
    },
}

MELODY_SYNTH: dict = {
    "engine": "fm",
    "preset": "bell",
    "env": {
        "attack_ms": 3.0,
        "decay_ms": 350.0,
        "sustain_ratio": 0.2,
        "release_ms": 300.0,
    },
}

COUNTER_SYNTH: dict = {
    "engine": "additive",
    "preset": "soft_pad",
    "env": {
        "attack_ms": 60.0,
        "decay_ms": 400.0,
        "sustain_ratio": 0.4,
        "release_ms": 500.0,
    },
}

PAD_SYNTH: dict = {
    "engine": "filtered_stack",
    "preset": "warm_pad",
    "env": {
        "attack_ms": 300.0,
        "decay_ms": 600.0,
        "sustain_ratio": 0.7,
        "release_ms": 1200.0,
    },
}

# ---------------------------------------------------------------------------
# Phrase helpers
# ---------------------------------------------------------------------------


def _bar_starts(section_start: float, n_bars: int) -> list[float]:
    """Return bar-onset times for *n_bars* starting at *section_start*."""
    return [section_start + i * BAR_DUR for i in range(n_bars)]


def _kick_phrase():
    """Kick on group downbeats (2+2+3)."""
    rhythm = AKSAK.to_rhythm()  # 3 spans matching the 3 groups
    return line(
        tones=[F0, F0, F0],
        rhythm=rhythm,
        pitch_kind="freq",
        amp_db=-4.0,
        synth_defaults=KICK_SYNTH,
    )


def _hat_phrase():
    """Hi-hat on every pulse."""
    rhythm = AKSAK.to_pulses()  # 7 equal pulses
    hat_freq = F0 * 4.0  # high frequency for hats
    return line(
        tones=[hat_freq] * 7,
        rhythm=rhythm,
        pitch_kind="freq",
        amp_db=-8.0,
        velocity=[0.9, 0.6, 0.9, 0.6, 0.9, 0.6, 0.7],
        synth_defaults=METAL_SYNTH,
    )


def _bass_phrase():
    """Bass follows group downbeats cycling through root-fifth-seventh."""
    rhythm = AKSAK.to_rhythm()
    return line(
        tones=BASS_TONES,
        rhythm=rhythm,
        pitch_kind="freq",
        amp_db=-6.0,
        synth_defaults=BASS_SYNTH,
    )


# ---------------------------------------------------------------------------
# Score builder
# ---------------------------------------------------------------------------


def build_study_aksak_score() -> Score:
    score = Score(
        f0_hz=F0,
        timing_humanize=TimingHumanizeSpec(preset="chamber", seed=42),
        master_effects=DEFAULT_MASTER_EFFECTS,
        send_buses=[
            SendBusSpec(name="room", effects=[SOFT_REVERB_EFFECT]),
        ],
    )

    # -- Voices --

    score.add_voice(
        "kick",
        synth_defaults=KICK_SYNTH,
        normalize_peak_db=-6.0,
        velocity_humanize=None,
        pan=0.0,
        mix_db=-1.0,
        sends=[VoiceSend(target="room", send_db=-14.0)],
    )

    score.add_voice(
        "hat",
        synth_defaults=METAL_SYNTH,
        normalize_peak_db=-6.0,
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living", seed=50),
        pan=0.25,
        mix_db=-4.0,
        sends=[VoiceSend(target="room", send_db=-10.0)],
    )

    score.add_voice(
        "bass",
        synth_defaults=BASS_SYNTH,
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living", seed=60),
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog", seed=61),
        pan=-0.2,
        mix_db=-2.0,
        sends=[VoiceSend(target="room", send_db=-12.0)],
    )

    score.add_voice(
        "melody",
        synth_defaults=MELODY_SYNTH,
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living", seed=70),
        envelope_humanize=EnvelopeHumanizeSpec(preset="loose_pluck", seed=71),
        pan=0.3,
        mix_db=-1.0,
        effects=[
            EffectSpec(
                "delay", {"delay_seconds": PULSE * 2, "feedback": 0.2, "mix": 0.15}
            ),
        ],
        sends=[VoiceSend(target="room", send_db=-6.0)],
    )

    score.add_voice(
        "counter",
        synth_defaults=COUNTER_SYNTH,
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living", seed=80),
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog", seed=81),
        pan=-0.35,
        mix_db=-3.0,
        sends=[VoiceSend(target="room", send_db=-8.0)],
    )

    score.add_voice(
        "pad",
        synth_defaults=PAD_SYNTH,
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living", seed=90),
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog", seed=91),
        pan=0.0,
        mix_db=-5.0,
        sends=[VoiceSend(target="room", send_db=-4.0)],
    )

    # -----------------------------------------------------------------------
    # Section 1: Pulse (bars 1-4) — kick on group downbeats
    # -----------------------------------------------------------------------

    kick_ph = _kick_phrase()
    sequence(score, "kick", kick_ph, starts=_bar_starts(SEC_PULSE_START, 4))

    # -----------------------------------------------------------------------
    # Section 2: Build (bars 5-8) — add hats + bass
    # -----------------------------------------------------------------------

    hat_ph = _hat_phrase()
    bass_ph = _bass_phrase()
    build_starts = _bar_starts(SEC_BUILD_START, 4)

    sequence(score, "kick", kick_ph, starts=build_starts)
    sequence(score, "hat", hat_ph, starts=build_starts)
    sequence(score, "bass", bass_ph, starts=build_starts)

    # -----------------------------------------------------------------------
    # Section 3: Groove (bars 9-16) — full texture with cross-rhythm
    # -----------------------------------------------------------------------

    groove_starts = _bar_starts(SEC_GROOVE_START, 8)

    sequence(score, "kick", kick_ph, starts=groove_starts)
    sequence(score, "hat", hat_ph, starts=groove_starts)
    sequence(score, "bass", bass_ph, starts=groove_starts)

    # Cross-rhythm: 7-division melody against 3-division counter per bar
    cross_phrases = cross_rhythm(
        layers=[
            (5, MELODY_TONES),
            (3, COUNTER_TONES),
        ],
        span=BAR_DUR,
        pitch_kind="freq",
        amp_db=-6.0,
    )
    melody_cross, counter_cross = cross_phrases

    sequence(score, "melody", melody_cross, starts=groove_starts)
    sequence(score, "counter", counter_cross, starts=groove_starts)

    # Pad enters softly in the groove
    pad_phrase = line(
        tones=PAD_TONES,
        rhythm=AKSAK.to_rhythm(),
        pitch_kind="freq",
        amp_db=-10.0,
        synth_defaults=PAD_SYNTH,
    )
    sequence(score, "pad", pad_phrase, starts=groove_starts)

    # -----------------------------------------------------------------------
    # Section 4: Mutation (bars 17-24) — evolving hat patterns
    # -----------------------------------------------------------------------

    mutation_starts = _bar_starts(SEC_MUTATION_START, 8)

    sequence(score, "kick", kick_ph, starts=mutation_starts)
    sequence(score, "bass", bass_ph, starts=mutation_starts)
    sequence(score, "melody", melody_cross, starts=mutation_starts)

    # Mutate the hat phrase progressively across bars
    for i, bar_start in enumerate(mutation_starts):
        intensity = (i + 1) / len(mutation_starts)
        mutated_hat = mutate_rhythm(
            hat_ph,
            drop_prob=0.15 * intensity,
            shift_amount=PULSE * 0.08 * intensity,
            subdivide_prob=0.1 * intensity,
            accent_drift=0.2 * intensity,
            seed=100 + i,
        )
        sequence(score, "hat", mutated_hat, starts=[bar_start])

    # Counter continues with slight variation
    sequence(score, "counter", counter_cross, starts=mutation_starts)
    sequence(score, "pad", pad_phrase, starts=mutation_starts)

    # -----------------------------------------------------------------------
    # Section 5: Poly peak (bars 25-30) — polyrhythmic tension
    # -----------------------------------------------------------------------

    poly_starts = _bar_starts(SEC_POLY_START, 6)

    sequence(score, "kick", kick_ph, starts=poly_starts)
    sequence(score, "bass", bass_ph, starts=poly_starts)

    # polyrhythm(3, 7) over one bar — two competing grids
    poly_3, poly_7 = polyrhythm(3, 7, span=BAR_DUR)

    melody_poly = line(
        tones=MELODY_TONES[:3],
        rhythm=poly_3,
        pitch_kind="freq",
        amp_db=-5.0,
        synth_defaults=MELODY_SYNTH,
    )
    counter_tones_7 = [F0 * r for r in (SCALE_RATIOS[::-1] + SCALE_RATIOS[:2])]
    counter_poly = line(
        tones=counter_tones_7,
        rhythm=poly_7,
        pitch_kind="freq",
        amp_db=-7.0,
        synth_defaults=COUNTER_SYNTH,
    )

    sequence(score, "melody", melody_poly, starts=poly_starts)
    sequence(score, "counter", counter_poly, starts=poly_starts)

    # Intensified mutated hats
    for i, bar_start in enumerate(poly_starts):
        mutated_hat = mutate_rhythm(
            hat_ph,
            drop_prob=0.1,
            shift_amount=PULSE * 0.12,
            subdivide_prob=0.2,
            accent_drift=0.3,
            seed=200 + i,
        )
        sequence(score, "hat", mutated_hat, starts=[bar_start])

    sequence(score, "pad", pad_phrase, starts=poly_starts)

    # -----------------------------------------------------------------------
    # Section 6: Release (bars 31-35) — thinning out, reverb tail
    # -----------------------------------------------------------------------

    release_starts = _bar_starts(SEC_RELEASE_START, 5)

    # Sparse kick, fading
    for i, bar_start in enumerate(release_starts):
        fade = 1.0 - i / len(release_starts)
        fading_kick = line(
            tones=[F0, F0, F0],
            rhythm=AKSAK.to_rhythm(),
            pitch_kind="freq",
            amp_db=-4.0,
            velocity=[fade * 0.8, fade * 0.5, fade * 0.6],
            synth_defaults=KICK_SYNTH,
        )
        sequence(score, "kick", fading_kick, starts=[bar_start])

    # Sustained pad with long notes to ring out
    release_pad = line(
        tones=PAD_TONES,
        rhythm=AKSAK.to_rhythm(),
        pitch_kind="freq",
        amp_db=-8.0,
        synth_defaults=PAD_SYNTH,
    )
    sequence(score, "pad", release_pad, starts=release_starts[:3])

    return score


# ---------------------------------------------------------------------------
# Piece registration
# ---------------------------------------------------------------------------

PIECES: dict[str, PieceDefinition] = {
    "study_aksak": PieceDefinition(
        name="study_aksak",
        output_name="study_aksak",
        build_score=build_study_aksak_score,
        study=True,
    ),
}
