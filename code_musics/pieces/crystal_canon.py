"""Crystal Canon — three voices tracing the same JI phrase at golden-ratio
tempo proportions (1 : φ : φ²).

φ is irrational, so the voices never phase-align.  They share harmonic
material — a simple 8-note arch through pure 5-limit intervals — but each
hears it at a different speed.  The polytemporal loop is the Escher staircase:
always in motion, never resolving, yet built from the same few pure stones.

Arc (oblique card: "take away in order of apparent non-importance"):
  § 1  slow alone        — patient crystalline growth; sets the harmonic world
  § 2  slow + fast       — the two extremes in dialogue; maximum divergence
  § 3  all three         — fullest texture; voices briefly approach alignment
  § 4  slow + mid        — fast removed first (most active, apparently loudest)
  § 5  slow alone (coda) — mid removed; one voice remains in open space

Timbre differentiation keeps the voices spatially and texturally distinct:
  slow  — pad, long attack/release, many harmonics bleeding together; centre
  fast  — crystalline, near-sine, short articulation; left
  mid   — medium, balanced; right
"""

from __future__ import annotations

import logging
import math

from code_musics.composition import RhythmCell, line
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Phrase, Score

logger = logging.getLogger(__name__)

PHI: float = (1.0 + math.sqrt(5.0)) / 2.0  # ≈ 1.618

# ── Phrase definition ─────────────────────────────────────────────────────────
# 8-note arch through pure 5-limit intervals:
#   root → maj3 → 5th → octave → maj6 → 5th → maj3 → root
# At the φ-related tempos, the phrase lasts 4.8s / 7.76s / 2.97s respectively.
_NOTE_DUR: float = 0.60  # seconds per note at time_scale=1.0
_PHRASE_PARTIALS: tuple[float, ...] = (1.0, 5 / 4, 3 / 2, 2.0, 5 / 3, 3 / 2, 5 / 4, 1.0)
_N_NOTES: int = len(_PHRASE_PARTIALS)
_PHRASE_DUR_BASE: float = _NOTE_DUR * _N_NOTES  # 4.80 s at ×1.0

# ── Time scales ───────────────────────────────────────────────────────────────
_TS_SLOW: float = PHI  # ≈ 1.618 — one phrase ≈ 7.76 s
_TS_MID: float = 1.0  #           one phrase = 4.80 s
_TS_FAST: float = 1.0 / PHI  # ≈ 0.618 — one phrase ≈ 2.97 s

_PDur: dict[str, float] = {
    "slow": _PHRASE_DUR_BASE * _TS_SLOW,
    "mid": _PHRASE_DUR_BASE * _TS_MID,
    "fast": _PHRASE_DUR_BASE * _TS_FAST,
}


def _build_phrase(f0: float, amp_db: float) -> Phrase:
    freqs = [f0 * r for r in _PHRASE_PARTIALS]
    return line(
        tones=freqs,
        rhythm=RhythmCell(spans=tuple([_NOTE_DUR] * _N_NOTES)),
        pitch_kind="freq",
        amp_db=amp_db,
    )


def _fill_voice(
    score: Score,
    voice: str,
    phrase: Phrase,
    time_scale: float,
    t_start: float,
    t_end: float,
    *,
    fade_in_phrases: int = 2,
    fade_out_phrases: int = 3,
) -> None:
    """Repeat phrase across [t_start, t_end] with gentle amplitude fade-in/out."""
    phrase_dur = _PHRASE_DUR_BASE * time_scale
    total_reps = math.ceil((t_end - t_start) / phrase_dur)
    t = t_start
    for rep in range(total_reps):
        if rep < fade_in_phrases:
            amp_scale = (rep + 1) / fade_in_phrases
        elif rep >= total_reps - fade_out_phrases:
            reps_remaining = total_reps - rep
            amp_scale = max(reps_remaining / fade_out_phrases, 0.15)
        else:
            amp_scale = 1.0
        score.add_phrase(
            voice, phrase, start=t, time_scale=time_scale, amp_scale=amp_scale
        )
        t += phrase_dur


def build_crystal_canon_score() -> Score:
    """Three-voice golden-ratio canon on a pure 5-limit arch phrase."""
    f0 = 220.0  # A3

    # ── Section boundaries ────────────────────────────────────────────────────
    # Each boundary is chosen at a round number of the relevant voice's phrase
    # so that entries and exits happen at phrase downbeats (per that voice).
    # Because φ is irrational, these boundaries are rarely simultaneous across
    # voices — the phase relationships stay perpetually shifting.

    T0 = 0.0
    T1 = T0 + 4 * _PDur["slow"]  # ≈  31.1 s  — fast enters
    T2 = T1 + 8 * _PDur["fast"]  # ≈  54.8 s  — mid enters
    T3 = T2 + 6 * _PDur["mid"]  # ≈  83.6 s  — fast exits
    T4 = T3 + 4 * _PDur["slow"]  # ≈ 114.7 s  — mid exits
    T5 = T4 + 4 * _PDur["slow"]  # ≈ 145.8 s  — end / fade

    logger.info(
        "crystal_canon sections: T1=%.1f T2=%.1f T3=%.1f T4=%.1f T5=%.1f",
        T1,
        T2,
        T3,
        T4,
        T5,
    )

    # ── Score ─────────────────────────────────────────────────────────────────
    score = Score(
        f0=f0,
        master_effects=[
            EffectSpec(
                "reverb", {"room_size": 0.78, "damping": 0.38, "wet_level": 0.22}
            ),
            EffectSpec("delay", {"delay_seconds": 0.31, "feedback": 0.12, "mix": 0.07}),
        ],
    )

    # slow — pad; long attack blurs adjacent notes together into a wash
    score.add_voice(
        "slow",
        synth_defaults={
            "engine": "additive",
            "params": {"n_harmonics": 8, "harmonic_rolloff": 0.50},
            "env": {
                "attack_ms": 700.0,
                "decay_ms": 400.0,
                "sustain_ratio": 0.78,
                "release_ms": 1400.0,
            },
        },
        mix_db=0.0,
        pan=0.0,
        velocity_humanize=None,
    )

    # fast — crystalline; near-sine, short articulation, each note distinct
    score.add_voice(
        "fast",
        synth_defaults={
            "engine": "additive",
            "params": {"n_harmonics": 3, "harmonic_rolloff": 0.72},
            "env": {
                "attack_ms": 10.0,
                "decay_ms": 140.0,
                "sustain_ratio": 0.38,
                "release_ms": 260.0,
            },
        },
        mix_db=-3.0,
        pan=-0.42,
        velocity_humanize=None,
    )

    # mid — balanced; the structural middle ground
    score.add_voice(
        "mid",
        synth_defaults={
            "engine": "additive",
            "params": {"n_harmonics": 5, "harmonic_rolloff": 0.60},
            "env": {
                "attack_ms": 90.0,
                "decay_ms": 260.0,
                "sustain_ratio": 0.62,
                "release_ms": 560.0,
            },
        },
        mix_db=-2.0,
        pan=0.36,
        velocity_humanize=None,
    )

    # drone — sub-octave anchor; barely audible, felt more than heard
    score.add_voice(
        "drone",
        synth_defaults={
            "engine": "additive",
            "params": {"n_harmonics": 3, "harmonic_rolloff": 0.70},
            "env": {
                "attack_ms": 2800.0,
                "decay_ms": 400.0,
                "sustain_ratio": 0.88,
                "release_ms": 4000.0,
            },
        },
        mix_db=-14.0,
        pan=0.0,
        velocity_humanize=None,
    )

    # ── Phrases (same material, three dynamic levels matching mix positions) ──
    phrase_slow = _build_phrase(f0, amp_db=-10.0)
    phrase_fast = _build_phrase(f0, amp_db=-12.0)
    phrase_mid = _build_phrase(f0, amp_db=-11.0)

    # ── Fill voices across their active windows ───────────────────────────────
    # § 1 + § 2 + § 3 + § 4 + § 5 : slow runs throughout
    _fill_voice(
        score,
        "slow",
        phrase_slow,
        _TS_SLOW,
        T0,
        T5,
        fade_in_phrases=1,
        fade_out_phrases=4,
    )

    # § 2 + § 3 : fast active
    _fill_voice(
        score,
        "fast",
        phrase_fast,
        _TS_FAST,
        T1,
        T3,
        fade_in_phrases=3,
        fade_out_phrases=4,
    )

    # § 3 + § 4 : mid active
    _fill_voice(
        score, "mid", phrase_mid, _TS_MID, T2, T4, fade_in_phrases=2, fade_out_phrases=3
    )

    # Drone: two overlapping long notes — one at the root, one at the sub-octave.
    # The sub enters a little late so the harmonic opening is gradual.
    score.add_note("drone", start=T0, duration=T5 + 4.0, freq=f0, amp_db=-16.0)
    score.add_note(
        "drone", start=T0 + 8.0, duration=T5 - 4.0, freq=f0 / 2.0, amp_db=-20.0
    )

    return score


PIECES: dict[str, PieceDefinition] = {
    "crystal_canon": PieceDefinition(
        name="crystal_canon",
        output_name="30_crystal_canon",
        build_score=build_crystal_canon_score,
    ),
}
