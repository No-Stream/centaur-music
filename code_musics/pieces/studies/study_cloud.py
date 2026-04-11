"""Stochastic cloud study — ambient texture built from overlapping harmonic-series clouds.

Three cloud layers drift in and out across ~28 seconds:
  - A low, slow subharmonic bed (additive drone engine, long notes)
  - A mid-range swell that builds from sparse to dense and back (filtered_stack pad)
  - A high, bright overtone sparkle (FM bell engine, short notes)

All pitch material comes from harmonic-series TonePools rooted on a shared
fundamental, so the clouds reinforce each other harmonically even though note
placement is fully stochastic.
"""

from __future__ import annotations

from code_musics.generative.cloud import stochastic_cloud
from code_musics.generative.tone_pool import TonePool
from code_musics.humanize import EnvelopeHumanizeSpec, TimingHumanizeSpec
from code_musics.pieces._shared import REVERB_EFFECT, SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Score, VoiceSend

DURATION_S = 28.0
F0_HZ = 55.0


def build_score() -> Score:
    """Build the stochastic cloud study score."""
    score = Score(
        f0=F0_HZ,
        timing_humanize=TimingHumanizeSpec(preset="loose_late_night"),
        master_effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 30.0, "slope_db_per_oct": 24},
                        {"kind": "high_shelf", "freq_hz": 9000.0, "gain_db": -1.5},
                    ],
                },
            ),
        ],
    )

    # ── shared reverb bus ──────────────────────────────────────────────
    score.add_send_bus(
        "hall",
        effects=[
            EffectSpec("reverb", {"room_size": 0.9, "damping": 0.5, "wet_level": 0.7}),
        ],
    )

    # ── layer 1: subharmonic bed ──────────────────────────────────────
    # Low partials (1-4) of the harmonic series, weighted toward the fundamental.
    # Long, slow notes that provide a warm foundation.
    low_pool = TonePool.weighted(
        {
            1.0: 4.0,  # fundamental — heaviest weight
            2.0: 2.0,  # octave
            3 / 2: 1.5,  # fifth
            3.0: 1.0,  # twelfth
        }
    )
    low_cloud = stochastic_cloud(
        tones=low_pool,
        duration=DURATION_S,
        density=[(0.0, 0.8), (0.3, 1.2), (0.7, 1.2), (1.0, 0.6)],
        amp_db_range=(-18.0, -10.0),
        note_dur_range=(3.0, 6.0),
        pitch_kind="partial",
        seed=42,
    )

    score.add_voice(
        "low_bed",
        synth_defaults={
            "engine": "additive",
            "preset": "drone",
            "env": {
                "attack_ms": 800.0,
                "decay_ms": 400.0,
                "sustain_ratio": 0.80,
                "release_ms": 2000.0,
            },
            "params": {
                "harmonic_rolloff": 0.55,
                "n_harmonics": 4,
            },
        },
        effects=[SOFT_REVERB_EFFECT],
        sends=[VoiceSend(target="hall", send_db=-6.0)],
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        mix_db=-2.0,
        pan=-0.15,
    )
    score.add_phrase("low_bed", low_cloud, start=0.0)

    # ── layer 2: mid-range density swell ──────────────────────────────
    # Partials 4-10 of the harmonic series — the main textural body.
    # Density swells from sparse to dense and back.
    mid_pool = TonePool.from_harmonics([4, 5, 6, 7, 8, 9, 10])
    mid_cloud = stochastic_cloud(
        tones=mid_pool,
        duration=DURATION_S,
        density=[
            (0.0, 1.0),
            (0.15, 2.0),
            (0.35, 8.0),
            (0.55, 10.0),
            (0.75, 6.0),
            (0.90, 2.0),
            (1.0, 1.0),
        ],
        amp_db_range=(-20.0, -9.0),
        note_dur_range=(0.4, 1.8),
        pitch_kind="partial",
        seed=137,
    )

    score.add_voice(
        "mid_swell",
        synth_defaults={
            "engine": "filtered_stack",
            "preset": "warm_pad",
            "env": {
                "attack_ms": 200.0,
                "decay_ms": 300.0,
                "sustain_ratio": 0.65,
                "release_ms": 800.0,
            },
            "params": {
                "cutoff_hz": 1800.0,
                "resonance_q": 1.38,
                "filter_env_amount": 0.3,
                "filter_env_decay": 0.4,
            },
        },
        effects=[
            EffectSpec("chorus", {"preset": "juno_subtle", "mix": 0.22}),
        ],
        sends=[VoiceSend(target="hall", send_db=-3.0)],
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        mix_db=0.0,
        pan=0.0,
    )
    score.add_phrase("mid_swell", mid_cloud, start=0.0)

    # ── layer 3: high overtone sparkle ────────────────────────────────
    # Upper partials (8-16), short FM bell notes that glint through the texture.
    high_pool = TonePool.from_harmonics([8, 9, 10, 11, 12, 13, 14, 16])
    high_cloud = stochastic_cloud(
        tones=high_pool,
        duration=DURATION_S,
        density=[(0.0, 0.3), (0.25, 1.5), (0.5, 2.5), (0.75, 1.5), (1.0, 0.5)],
        amp_db_range=(-24.0, -14.0),
        note_dur_range=(0.08, 0.35),
        pitch_kind="partial",
        seed=271,
    )

    score.add_voice(
        "high_sparkle",
        synth_defaults={
            "engine": "fm",
            "preset": "bell",
            "env": {
                "attack_ms": 5.0,
                "decay_ms": 120.0,
                "sustain_ratio": 0.15,
                "release_ms": 400.0,
            },
        },
        effects=[REVERB_EFFECT],
        sends=[VoiceSend(target="hall", send_db=-8.0)],
        mix_db=-3.0,
        pan=0.2,
    )
    score.add_phrase("high_sparkle", high_cloud, start=0.0)

    return score


PIECES: dict[str, PieceDefinition] = {
    "study_cloud": PieceDefinition(
        name="study_cloud",
        output_name="study_cloud",
        build_score=build_score,
        study=True,
    ),
}
