"""Groove study — same beat, different feel.

~100s at 92 BPM (~34 bars).  One beat, one chord loop, five groove lenses.
The musical material stays constant; only the groove changes.

Section map:
  1-6    Straight — kick + hat, clinical, quantized
  7-12   MPC tight — same beat, now pocket
  13-20  Dilla lazy — behind-the-beat, pad + bass enter
  21-26  Bossa — anticipated offbeats, light and airy
  27-34  808 swing — full energy, classic shuffle, clap enters

Showcases: Groove presets (velocity weighting + timing displacement),
prob_rhythm for drum patterns, mutate_rhythm for evolving hats.
"""

from __future__ import annotations

from code_musics.composition import line, sequence
from code_musics.generative.mutation import mutate_rhythm
from code_musics.generative.prob_rhythm import prob_rhythm
from code_musics.humanize import TimingHumanizeSpec, VelocityHumanizeSpec
from code_musics.meter import Groove, Q, S, Timeline
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS, SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Score, SendBusSpec, VoiceSend

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BPM = 92.0
BEAT = 60.0 / BPM
BAR = 4.0 * BEAT
S16 = BEAT / 4.0
F0 = 55.0  # A1

SECTIONS: list[tuple[str, int, int, Groove | None]] = [
    ("straight", 1, 6, None),
    ("mpc", 7, 12, Groove.mpc_tight()),
    ("dilla", 13, 20, Groove.dilla_lazy()),
    ("bossa", 21, 26, Groove.bossa()),
    ("808", 27, 34, Groove.tr808_swing()),
]

# 7-limit otonal / utonal two-chord vamp
CHORD_A = [1.0, 5 / 4, 3 / 2, 7 / 4]  # otonal
CHORD_B = [1.0, 8 / 7, 4 / 3, 8 / 5]  # utonal colour

# ---------------------------------------------------------------------------
# Synth specs
# ---------------------------------------------------------------------------

KICK_SYNTH: dict = {"engine": "drum_voice", "preset": "808_hiphop"}
HAT_SYNTH: dict = {"engine": "drum_voice", "preset": "closed_hat"}
PAD_SYNTH: dict = {
    "engine": "organ",
    "preset": "septimal",
    "env": {
        "attack_ms": 120.0,
        "decay_ms": 600.0,
        "sustain_ratio": 0.7,
        "release_ms": 400.0,
    },
}
BASS_SYNTH: dict = {
    "engine": "polyblep",
    "preset": "sub_bass",
    "env": {
        "attack_ms": 5.0,
        "decay_ms": 200.0,
        "sustain_ratio": 0.6,
        "release_ms": 150.0,
    },
}
CLAP_SYNTH: dict = {"engine": "drum_voice", "preset": "909_clap"}


def _bar_start(bar: int) -> float:
    """1-based bar number → seconds."""
    return (bar - 1) * BAR


# ---------------------------------------------------------------------------
# Pattern builders
# ---------------------------------------------------------------------------


def _build_kick_phrase(tl: Timeline):
    """Four-on-the-floor with probabilistic ghost hits."""
    kick_rhythm = prob_rhythm(
        16,
        onset_weights=[
            1.0,
            0.0,
            0.0,
            0.0,
            0.8,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.6,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        span=tl.duration(S),
        seed=1,
    )
    return line(
        tones=[1.0] * len(kick_rhythm.spans),
        rhythm=kick_rhythm,
        pitch_kind="partial",
        amp_db=-6.0,
        synth_defaults=KICK_SYNTH,
    )


def _build_hat_phrase(tl: Timeline, section_idx: int):
    """Dense hat pattern, mutated more aggressively for later sections."""
    hat_rhythm = prob_rhythm(
        16,
        onset_weights=[0.95, 0.4, 0.7, 0.4],
        accent_weights=[1.0, 0.5, 0.8, 0.5],
        span=tl.duration(S),
        seed=2,
    )
    base = line(
        tones=[4.0] * len(hat_rhythm.spans),
        rhythm=hat_rhythm,
        pitch_kind="partial",
        amp_db=-8.0,
        synth_defaults=HAT_SYNTH,
    )
    if section_idx == 0:
        return base
    return mutate_rhythm(
        base,
        add_prob=0.04 * section_idx,
        shift_amount=0.002 * section_idx,
        accent_drift=0.05 * section_idx,
        seed=100 + section_idx,
    )


def _build_clap_phrase(tl: Timeline):
    """Backbeat clap on beats 2 and 4."""
    return line(
        tones=[1.0, 1.0],
        rhythm=[tl.duration(Q)] * 2,
        pitch_kind="partial",
        amp_db=-8.0,
        synth_defaults=CLAP_SYNTH,
    )


# ---------------------------------------------------------------------------
# Score builder
# ---------------------------------------------------------------------------


def build_score() -> Score:
    score = Score(
        f0_hz=F0,
        timing_humanize=TimingHumanizeSpec(preset="tight_ensemble", seed=42),
        send_buses=[
            SendBusSpec(name="room", effects=[SOFT_REVERB_EFFECT]),
        ],
        master_effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 35.0, "slope_db_per_oct": 12}
                    ]
                },
            ),
            *DEFAULT_MASTER_EFFECTS,
        ],
    )

    # --- Voices ---

    score.add_voice(
        "kick",
        synth_defaults=KICK_SYNTH,
        normalize_peak_db=-6.0,
        pan=0.0,
        mix_db=-1.0,
    )
    score.add_voice(
        "hat",
        synth_defaults=HAT_SYNTH,
        normalize_peak_db=-6.0,
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living", seed=10),
        pan=0.2,
        mix_db=-4.0,
        sends=[VoiceSend(target="room", send_db=-14.0)],
    )
    score.add_voice(
        "clap",
        synth_defaults=CLAP_SYNTH,
        normalize_peak_db=-6.0,
        pan=-0.1,
        mix_db=-5.0,
        sends=[VoiceSend(target="room", send_db=-10.0)],
    )
    score.add_voice(
        "pad",
        synth_defaults=PAD_SYNTH,
        normalize_lufs=-24.0,
        pan=0.0,
        mix_db=-3.0,
        sends=[VoiceSend(target="room", send_db=-6.0)],
    )
    score.add_voice(
        "bass",
        synth_defaults=BASS_SYNTH,
        normalize_lufs=-24.0,
        pan=0.0,
        mix_db=-2.0,
    )

    # --- Section loop ---

    for section_idx, (_name, bar_start, bar_end, groove) in enumerate(SECTIONS):
        tl = Timeline(bpm=BPM, groove=groove)
        n_bars = bar_end - bar_start + 1
        section_offset = _bar_start(bar_start)

        # Kick — present in all sections
        kick_phrase = _build_kick_phrase(tl)
        kick_starts = [section_offset + i * BAR for i in range(n_bars)]
        sequence(score, "kick", kick_phrase, starts=kick_starts)

        # Hat — present in all sections, mutating
        hat_phrase = _build_hat_phrase(tl, section_idx)
        hat_starts = [section_offset + i * BAR for i in range(n_bars)]
        sequence(score, "hat", hat_phrase, starts=hat_starts)

        # Pad — enters in Dilla section (idx 2) and stays
        if section_idx >= 2:
            for bar_offset in range(0, n_bars, 2):
                t = section_offset + bar_offset * BAR
                for i, partial in enumerate(CHORD_A):
                    score.add_note(
                        "pad",
                        start=t,
                        duration=BAR * 2,
                        partial=partial,
                        amp_db=-10.0 - i * 1.5,
                    )
                t2 = t + BAR * 2
                if bar_offset + 2 < n_bars:
                    for i, partial in enumerate(CHORD_B):
                        score.add_note(
                            "pad",
                            start=t2,
                            duration=BAR * 2,
                            partial=partial,
                            amp_db=-10.0 - i * 1.5,
                        )

        # Bass — enters in Dilla section (idx 2) and stays
        if section_idx >= 2:
            for bar_i in range(n_bars):
                t = section_offset + bar_i * BAR
                root = 1.0 if bar_i % 4 < 2 else 8 / 7
                score.add_note(
                    "bass", start=t, duration=BAR * 0.85, partial=root, amp_db=-6.0
                )
                score.add_note(
                    "bass",
                    start=t + BEAT * 2.5,
                    duration=BEAT * 1.0,
                    partial=root * 3 / 2,
                    amp_db=-9.0,
                )

        # Clap — enters in 808 section (idx 4)
        if section_idx >= 4:
            clap_phrase = _build_clap_phrase(tl)
            for bar_i in range(n_bars):
                t = section_offset + bar_i * BAR + BEAT
                sequence(score, "clap", clap_phrase, starts=[t])

    return score


PIECES: dict[str, PieceDefinition] = {
    "study_groove": PieceDefinition(
        name="study_groove",
        output_name="study_groove",
        build_score=build_score,
        study=True,
    ),
}
