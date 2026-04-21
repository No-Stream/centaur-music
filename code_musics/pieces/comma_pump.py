"""Comma Pump — a 50-second exploration of syntonic comma drift.

Each chord cycle plays a 4:5:6:7 otonal tetrad, then drifts upward by one
syntonic comma (81/80 ~ 21.5 cents). After four cycles the harmony has
shifted ~86 cents from where it started — the "same" chord now sounds
noticeably wrong, yet you arrived there by imperceptible steps.

The melody voice stays anchored in the original tuning. At first it blends
perfectly with the harmony; by the final cycle the two worlds audibly
disagree. The central question: when does the listener notice?

f0 = 110 Hz (A2). Three voices: pad (drifting chords), melody (fixed
tuning), bass (drifts with the chords).
"""

from __future__ import annotations

from code_musics.composition import RhythmCell, line
from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    VelocityHumanizeSpec,
)
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Score

# ---------------------------------------------------------------------------
# Synth voices
# ---------------------------------------------------------------------------

_PAD_VOICE: dict = {
    "engine": "additive",
    "n_harmonics": 6,
    "harmonic_rolloff": 0.50,
    "brightness_tilt": -0.03,
    "attack": 1.8,
    "release": 4.0,
}

_MELODY_VOICE: dict = {
    "engine": "polyblep",
    "waveform": "triangle",
    "attack": 0.04,
    "release": 2.0,
}

_BASS_VOICE: dict = {
    "engine": "additive",
    "n_harmonics": 3,
    "harmonic_rolloff": 0.40,
    "attack": 2.5,
    "release": 5.0,
}

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

COMMA = 81 / 80  # syntonic comma ~ 21.5 cents

# 4:5:6:7 otonal tetrad as partials of f0_hz=110
# partial 1.0 = 110 Hz  root
# partial 5/4 = 137.5 Hz  major third
# partial 3/2 = 165 Hz  fifth
# partial 7/4 = 192.5 Hz  septimal seventh
CHORD_PARTIALS = (1.0, 5 / 4, 3 / 2, 7 / 4)


def build_score() -> Score:
    """Build the comma pump score."""
    score = Score(
        f0_hz=110.0,
        master_effects=[
            EffectSpec("reverb", {"room_size": 0.7, "damping": 0.5, "wet_level": 0.25}),
            EffectSpec(
                "saturation", {"preset": "tube_warm", "mix": 0.15, "drive": 1.08}
            ),
        ],
        timing_humanize=TimingHumanizeSpec(preset="chamber", seed=81),
    )

    # --- Voices ---------------------------------------------------------------

    score.add_voice(
        "pad",
        synth_defaults=_PAD_VOICE,
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad", seed=80),
        pan=0.0,
        mix_db=-2.0,
    )

    score.add_voice(
        "melody",
        synth_defaults=_MELODY_VOICE,
        velocity_humanize=VelocityHumanizeSpec(seed=5),
        pan=0.15,
        mix_db=0.0,
        effects=[
            EffectSpec("delay", {"delay_seconds": 0.38, "feedback": 0.25, "mix": 0.18}),
        ],
    )

    score.add_voice(
        "bass",
        synth_defaults=_BASS_VOICE,
        pan=0.0,
        mix_db=-3.0,
    )

    # --- Layout ---------------------------------------------------------------
    # 4 cycles, each ~11 seconds, plus a short coda.
    # cycle structure: 2s chord swell -> 4s melody over chord -> 2s chord fade
    #                  + 3s transition/breath before next cycle
    cycle_dur = 11.0
    n_cycles = 4

    for i in range(n_cycles):
        drift = COMMA**i
        t0 = 2.0 + i * cycle_dur  # cycle start time

        # --- Pad: drifting chord block ----------------------------------------
        # Each chord tone drifts by the accumulated comma.
        # Stagger entries slightly for a strummed feel.
        for j, base_partial in enumerate(CHORD_PARTIALS):
            shifted = base_partial * drift
            stagger = j * 0.15
            # Chord sustains for most of the cycle
            score.add_note(
                "pad",
                start=t0 + stagger,
                duration=8.5 - stagger,
                partial=shifted,
                amp_db=-17.0 - j * 1.0,  # higher tones slightly quieter
                velocity=0.85 + i * 0.04,  # slightly more intensity each cycle
            )

        # --- Bass: root and fifth, drifting -----------------------------------
        bass_root = 0.5 * drift  # A1, one octave below f0, drifted
        bass_fifth = 0.5 * (3 / 2) * drift  # E2, drifted
        score.add_note(
            "bass",
            start=t0 - 0.5,
            duration=cycle_dur - 0.5,
            partial=bass_root,
            amp_db=-22.0,
        )
        score.add_note(
            "bass",
            start=t0 + 1.5,
            duration=cycle_dur - 3.0,
            partial=bass_fifth,
            amp_db=-26.0,
        )

        # --- Melody: stays in the original tuning ----------------------------
        # The melody is always in the un-drifted partial space.
        # This creates increasing tension with the drifting harmony.
        _place_melody_cycle(score, t0, cycle_index=i)

    # --- Coda: final chord, drifted all four commas, fading slowly -----------
    coda_t = 2.0 + n_cycles * cycle_dur
    full_drift = COMMA**n_cycles
    for j, base_partial in enumerate(CHORD_PARTIALS):
        shifted = base_partial * full_drift
        score.add_note(
            "pad",
            start=coda_t + j * 0.2,
            duration=6.0,
            partial=shifted,
            amp_db=-19.0 - j * 1.0,
            velocity=0.75,
            synth={"attack": 2.5, "release": 6.0},
        )

    # Coda bass
    score.add_note(
        "bass",
        start=coda_t,
        duration=6.0,
        partial=0.5 * full_drift,
        amp_db=-22.0,
        synth={"attack": 2.5, "release": 7.0},
    )

    # Coda melody: one long note on the original root, held against the
    # drifted chord. The 86-cent discrepancy is fully exposed.
    score.add_note(
        "melody",
        start=coda_t + 1.0,
        duration=5.0,
        partial=2.0,  # A3, original tuning
        amp_db=-12.0,
        velocity=0.7,
    )

    return score


def _place_melody_cycle(score: Score, t0: float, cycle_index: int) -> None:
    """Write a short melody for one cycle, always in the original tuning.

    Each cycle's melody is a variation — same harmonic materials, different
    shape and emphasis. The melody uses the 4:5:6:7 partials in the upper
    octave (partials 2-4 of f0_hz=110, i.e. 220-440 Hz range).
    """
    # Partial shorthands (original, un-drifted)
    A3 = 2.0  # noqa: N806  — 220 Hz
    Cs4 = 5 / 2  # noqa: N806  — 275 Hz, pure major third
    E4 = 3.0  # noqa: N806  — 330 Hz, pure fifth
    Gb7 = 7 / 2  # noqa: N806  — 385 Hz, septimal seventh
    A4 = 4.0  # noqa: N806  — 440 Hz

    melody_start = t0 + 1.5  # melody enters after the chord has bloomed

    if cycle_index == 0:
        # Cycle 1: simple, gentle. Establish the tonality.
        # Descend from fifth through the septimal seventh to the root.
        phrase = line(
            tones=[E4, Gb7, Cs4, A3],
            rhythm=RhythmCell(
                spans=(1.0, 1.3, 1.0, 2.5),
                gates=(0.75, 0.85, 0.75, 1.0),
            ),
            amp_db=-11.0,
        )
        score.add_phrase("melody", phrase, start=melody_start)

    elif cycle_index == 1:
        # Cycle 2: ascend. Drift is barely noticeable (~22 cents).
        # Rise to the octave, linger on the seventh.
        phrase = line(
            tones=[A3, Cs4, E4, Gb7, A4],
            rhythm=RhythmCell(
                spans=(0.8, 0.7, 0.9, 1.6, 2.0),
                gates=(0.70, 0.72, 0.70, 0.90, 1.0),
            ),
            amp_db=-10.5,
        )
        score.add_phrase("melody", phrase, start=melody_start)

    elif cycle_index == 2:
        # Cycle 3: wider, more restless. Drift is ~43 cents now.
        # The melody reaches higher and pulls back harder.
        phrase = line(
            tones=[A4, Gb7, E4, Gb7, Cs4, A3],
            rhythm=RhythmCell(
                spans=(0.7, 0.9, 0.6, 1.2, 0.8, 2.5),
                gates=(0.68, 0.82, 0.65, 0.88, 0.72, 1.0),
            ),
            amp_db=-10.0,
        )
        score.add_phrase("melody", phrase, start=melody_start)

    else:
        # Cycle 4: sparse, exposed. Drift is ~64 cents — clearly audible.
        # Just two notes: the septimal seventh and the root.
        # Maximum exposure of the comma gap.
        phrase = line(
            tones=[Gb7, A3],
            rhythm=RhythmCell(
                spans=(2.2, 3.5),
                gates=(0.90, 1.0),
            ),
            amp_db=-11.0,
        )
        score.add_phrase("melody", phrase, start=melody_start)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PIECES: dict[str, PieceDefinition] = {
    "comma_pump": PieceDefinition(
        name="comma_pump",
        output_name="comma_pump",
        build_score=build_score,
    ),
}
