"""Turing Machine sequencer study — quasi-looping patterns that gradually evolve.

Three sections demonstrate the shift-register's behavior:
  1. Locked loop (flip_probability=0): an 8-step pattern repeats exactly,
     establishing a recognizable motif over a harmonic-series tone pool.
  2. Gentle mutation (flip_probability=0.08): the loop starts drifting —
     most steps repeat but occasional bit-flips introduce new tones.
  3. Dissolving (flip_probability=0.35): mutations accelerate, the pattern
     loses its identity, and the sequence wanders freely.

A sustained additive drone provides harmonic grounding while the polyblep
arp voice steps through the Turing Machine output.
"""

from __future__ import annotations

from code_musics.composition import RhythmCell, line, sequence
from code_musics.generative.euclidean import euclidean_rhythm
from code_musics.generative.tone_pool import TonePool
from code_musics.generative.turing import TuringMachine
from code_musics.humanize import EnvelopeHumanizeSpec, TimingHumanizeSpec
from code_musics.pieces._shared import DELAY_EFFECT, SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Phrase, Score, VoiceSend

# ── constants ─────────────────────────────────────────────────────────

F0_HZ = 55.0  # A1 fundamental
STEP_DUR = 0.22  # sixteenth-ish at ~136 bpm
REGISTER_LENGTH = 8

# Harmonic-series partials 4-11 give a bright, major-flavored pool
# spanning two octaves above the fundamental.
TONE_POOL = TonePool.from_harmonics([4, 5, 6, 7, 8, 9, 10, 11])

# Euclidean rhythm: 5 hits in 8 steps — a tresillo-adjacent groove
_rhythm_maybe = euclidean_rhythm(5, 8, span=STEP_DUR)
if _rhythm_maybe is None:
    raise ValueError("euclidean_rhythm(5, 8) must produce hits")
RHYTHM: RhythmCell = _rhythm_maybe

# Section durations in pattern repeats
LOCKED_REPEATS = 4  # 4 full cycles of the 8-step pattern
DRIFT_REPEATS = 5
DISSOLVE_REPEATS = 3
STEPS_PER_REPEAT = 8

LOCKED_STEPS = LOCKED_REPEATS * STEPS_PER_REPEAT
DRIFT_STEPS = DRIFT_REPEATS * STEPS_PER_REPEAT
DISSOLVE_STEPS = DISSOLVE_REPEATS * STEPS_PER_REPEAT


def build_score() -> Score:
    """Build the Turing Machine study score."""
    score = Score(
        f0=F0_HZ,
        timing_humanize=TimingHumanizeSpec(preset="tight_ensemble"),
        master_effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 35.0, "slope_db_per_oct": 12},
                        {"kind": "high_shelf", "freq_hz": 8500.0, "gain_db": -1.0},
                    ],
                },
            ),
        ],
    )

    # ── shared reverb bus ─────────────────────────────────────────────
    score.add_send_bus("hall", effects=[SOFT_REVERB_EFFECT])

    # ── drone voice ───────────────────────────────────────────────────
    score.add_voice(
        "drone",
        synth_defaults={
            "engine": "additive",
            "preset": "drone",
        },
        effects=[],
        sends=[VoiceSend(target="hall", send_db=-6.0)],
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        mix_db=-4.0,
    )

    # ── arp voice (polyblep with filter envelope) ─────────────────────
    score.add_voice(
        "arp",
        synth_defaults={
            "engine": "polyblep",
            "preset": "synth_pluck",
            "params": {
                "cutoff_hz": 1400.0,
                "resonance_ratio": 0.14,
                "filter_env_depth_ratio": 1.4,
                "filter_env_decay_ms": 120.0,
            },
        },
        effects=[DELAY_EFFECT],
        sends=[VoiceSend(target="hall", send_db=-9.0)],
        envelope_humanize=EnvelopeHumanizeSpec(preset="loose_pluck"),
        mix_db=-1.0,
        max_polyphony=1,
        legato=True,
    )

    # ── drone notes ───────────────────────────────────────────────────
    total_dur = (LOCKED_STEPS + DRIFT_STEPS + DISSOLVE_STEPS) * STEP_DUR + 2.0
    score.add_note("drone", start=0.0, duration=total_dur, partial=1.0, amp_db=-6.0)
    score.add_note(
        "drone", start=0.5, duration=total_dur - 0.5, partial=2.0, amp_db=-12.0
    )
    # Subtle fifth drone enters in drift section
    locked_dur = LOCKED_STEPS * STEP_DUR
    score.add_note(
        "drone",
        start=locked_dur,
        duration=total_dur - locked_dur,
        partial=3.0,
        amp_db=-16.0,
    )

    # ── section 1: locked loop ────────────────────────────────────────
    tm_locked = TuringMachine(
        length=REGISTER_LENGTH,
        flip_probability=0.0,
        tones=TONE_POOL,
        seed=42,
    )
    phrase_locked = tm_locked.to_phrase(LOCKED_STEPS, RHYTHM, amp_db=-3.0)

    sequence(score, "arp", phrase_locked, starts=[0.0])

    # ── section 2: gentle drift ───────────────────────────────────────
    drift_start = locked_dur
    tm_drift = TuringMachine(
        length=REGISTER_LENGTH,
        flip_probability=0.08,
        tones=TONE_POOL,
        seed=42,  # same seed so initial register state matches
    )
    # Generate enough steps to skip through the locked portion, then take drift steps
    drift_ratios = tm_drift.generate(LOCKED_STEPS + DRIFT_STEPS)[LOCKED_STEPS:]
    phrase_drift = _ratios_to_phrase(drift_ratios, RHYTHM, amp_db=-2.0)
    sequence(score, "arp", phrase_drift, starts=[drift_start])

    # ── section 3: dissolving ─────────────────────────────────────────
    dissolve_start = drift_start + DRIFT_STEPS * STEP_DUR
    tm_dissolve = TuringMachine(
        length=REGISTER_LENGTH,
        flip_probability=0.35,
        tones=TONE_POOL,
        seed=77,  # different seed for a fresh register
    )
    phrase_dissolve = tm_dissolve.to_phrase(DISSOLVE_STEPS, RHYTHM, amp_db=-1.0)
    sequence(score, "arp", phrase_dissolve, starts=[dissolve_start])

    return score


def _ratios_to_phrase(
    ratios: list[float],
    rhythm: RhythmCell,
    *,
    amp_db: float = 0.0,
) -> Phrase:
    """Build a phrase from pre-generated ratios and a rhythm cell."""

    return line(ratios, rhythm, amp_db=amp_db)


# ── registration ──────────────────────────────────────────────────────

PIECES: dict[str, PieceDefinition] = {
    "study_turing": PieceDefinition(
        name="study_turing",
        output_name="study_turing",
        build_score=build_score,
        study=True,
    ),
}
