"""Study: probability gate — pattern dissolution and reformation.

Demonstrates how `prob_gate` creates variation from repetition.  A clear
melodic phrase is repeated ten times; each repetition passes through
`prob_gate` with a different seed and a density envelope that thins the
pattern out and brings it back.  `position_weights` keep downbeats more
likely, and `accent_bias` favours louder notes for survival so rhythmic
shape degrades gracefully.

A steady low pulse on the root runs underneath, ungated, providing an
anchor while the melody dissolves and reforms above it.

~25 seconds, JI ratios over a 110 Hz tonic.
"""

from __future__ import annotations

from code_musics.composition import HarmonicContext, RhythmCell, ratio_line
from code_musics.generative.prob_gate import prob_gate
from code_musics.humanize import EnvelopeHumanizeSpec, TimingHumanizeSpec
from code_musics.pieces._shared import DELAY_EFFECT, SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Phrase, Score, VoiceSend

F0_HZ = 110.0
BPM = 92.0
BEAT_S = 60.0 / BPM

# Density envelope across 10 repetitions: full -> sparse -> full.
DENSITY_CURVE: tuple[float, ...] = (
    1.0,
    0.85,
    0.65,
    0.45,
    0.30,
    0.30,
    0.45,
    0.65,
    0.85,
    1.0,
)
NUM_REPS = len(DENSITY_CURVE)

# Position weights (per-beat within the 8-note phrase).
# Downbeat and strong beats survive more; offbeats thin first.
POSITION_WEIGHTS: tuple[float, ...] = (
    1.0,  # beat 1 — strongest
    0.60,
    0.85,  # beat 3
    0.55,
    0.90,  # beat 5
    0.50,
    0.75,  # beat 7
    0.45,
)


def _build_source_phrase(
    ctx: HarmonicContext,
) -> Phrase:
    """An 8-note JI phrase: stepwise with a couple of leaps for character."""
    ratios = [
        1 / 1,  # root
        9 / 8,  # major second
        5 / 4,  # major third
        4 / 3,  # fourth
        3 / 2,  # fifth
        7 / 4,  # septimal seventh — colour tone
        5 / 3,  # major sixth
        2 / 1,  # octave
    ]
    rhythm = RhythmCell(
        spans=(BEAT_S,) * 8,
        gates=(0.85, 0.80, 0.90, 0.80, 0.85, 0.92, 0.80, 0.70),
    )
    return ratio_line(
        tones=ratios,
        rhythm=rhythm,
        context=ctx,
        amp_db=-9.0,
    )


def build_study_prob_gate_score() -> Score:
    """Build the prob-gate study score."""
    ctx = HarmonicContext(tonic=F0_HZ, name="root")

    score = Score(
        f0=F0_HZ,
        timing_humanize=TimingHumanizeSpec(preset="chamber"),
        master_effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 40.0, "slope_db_per_oct": 12},
                        {"kind": "high_shelf", "freq_hz": 8000.0, "gain_db": -1.0},
                    ],
                },
            ),
        ],
    )

    # ── shared reverb bus ────────────────────────────────────────────
    score.add_send_bus(
        "room",
        effects=[SOFT_REVERB_EFFECT],
    )

    # ── steady pulse: ungated root, one note per phrase length ───────
    phrase_dur = BEAT_S * 8

    score.add_voice(
        "pulse",
        synth_defaults={
            "engine": "additive",
            "env": {
                "attack_ms": 60.0,
                "decay_ms": 200.0,
                "sustain_ratio": 0.70,
                "release_ms": 400.0,
            },
            "params": {
                "harmonic_rolloff": 0.50,
                "n_harmonics": 5,
            },
        },
        sends=[VoiceSend(target="room", send_db=-6.0)],
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        mix_db=-3.0,
        pan=-0.10,
    )
    for rep_idx in range(NUM_REPS):
        t = rep_idx * phrase_dur
        score.add_note(
            "pulse",
            start=t,
            duration=phrase_dur * 0.92,
            partial=1.0,
            amp_db=-12.0,
            label=f"pulse_{rep_idx}",
        )

    # ── gated melody ─────────────────────────────────────────────────
    source = _build_source_phrase(ctx)

    score.add_voice(
        "melody",
        synth_defaults={
            "engine": "polyblep",
            "preset": "warm_lead",
            "env": {
                "attack_ms": 18.0,
                "decay_ms": 140.0,
                "sustain_ratio": 0.55,
                "release_ms": 320.0,
            },
            "params": {
                "cutoff_hz": 2400.0,
                "resonance_ratio": 0.10,
                "filter_env_depth_ratio": 0.55,
                "filter_env_decay_ms": 200.0,
            },
        },
        effects=[DELAY_EFFECT],
        sends=[VoiceSend(target="room", send_db=-3.0)],
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        mix_db=0.0,
        pan=0.12,
    )

    for rep_idx in range(NUM_REPS):
        density = DENSITY_CURVE[rep_idx]
        seed = 100 + rep_idx * 7  # distinct seed per repetition

        gated = prob_gate(
            source,
            density=density,
            accent_bias=0.35,
            position_weights=POSITION_WEIGHTS,
            seed=seed,
        )

        if gated.events:
            score.add_phrase(
                "melody",
                gated,
                start=rep_idx * phrase_dur,
            )

    return score


PIECES: dict[str, PieceDefinition] = {
    "study_prob_gate": PieceDefinition(
        name="study_prob_gate",
        output_name="study_prob_gate",
        build_score=build_study_prob_gate_score,
        study=True,
    ),
}
