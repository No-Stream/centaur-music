"""Study for Surge XT — slow septimal chord progression.

Demonstrates the Surge XT instrument engine with MPE per-note pitch bend
for xenharmonic accuracy.  Surge XT's Classic oscillator with LP Vintage
Ladder filter and unison detune creates a warm analog pad.  The piece moves
through four chords in 7-limit JI, lingering on the septimal seventh for
its distinctive bittersweet colour.

Voice layout:
  - pad: Surge XT (Classic saw + LP Vintage Ladder, 3-voice unison, slow env)

Harmonic language: 7-limit JI centred on A2 (110 Hz).
"""

from __future__ import annotations

from code_musics.pieces._shared import SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import Score

F0_HZ: float = 110.0

# Surge XT parameter raw_values for a warm analog pad patch.
# Classic saw -> LP Vintage Ladder -> slow amp envelope.
_PAD_SURGE_PARAMS: dict[str, float] = {
    # Oscillator: Classic saw with 3-voice unison for chorus warmth
    "a_osc_1_type": 0.0,  # Classic
    "a_osc_1_shape": 0.5,  # 50% (saw position)
    "a_osc_1_unison_voices": 0.15,  # 3 voices
    "a_osc_1_unison_detune": 0.12,  # ~12 cents -- gentle chorus
    # Filter: Vintage Ladder LP, warm cutoff, mild resonance
    "a_filter_1_type": 0.295,  # LP Vintage Ladder
    "a_filter_1_cutoff": 0.52,  # ~500 Hz -- dark and warm
    "a_filter_1_resonance": 0.12,  # 12% -- just enough colour
    # Amp envelope: slow pad
    "a_amp_eg_attack": 0.50,  # ~350 ms
    "a_amp_eg_decay": 0.55,  # ~550 ms
    "a_amp_eg_sustain": 0.80,  # 80%
    "a_amp_eg_release": 0.58,  # ~700 ms
}


def build_score() -> Score:
    score = Score(
        f0_hz=F0_HZ,
        master_effects=[SOFT_REVERB_EFFECT],
    )

    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "surge_xt",
            "surge_params": _PAD_SURGE_PARAMS,
            "tail_seconds": 2.5,
        },
        normalize_lufs=-20.0,
    )

    # -- Chord progression in 7-limit JI -----------------------------------
    #
    # Each chord is a list of (partial, duration) pairs.  Partials are ratios
    # of f0 = 110 Hz, voiced across two octaves for openness.
    #
    # Chord I:   root major       1 — 5/2 — 3
    # Chord iv7: septimal fourth  4/3 — 7/3 — 8/3
    # Chord vi:  submediant       5/3 — 2 — 7/2
    # Chord I7:  septimal close   1 — 5/4 — 3/2 — 7/4

    chords: list[tuple[float, list[float]]] = [
        # (start_time, [partials])
        (0.0, [1, 5 / 2, 3]),  # I — open A major
        (4.0, [4 / 3, 7 / 3, 8 / 3]),  # iv7 — septimal D minor colour
        (8.5, [5 / 3, 2, 7 / 2]),  # vi — warm, with harmonic 7th peeking in
        (13.0, [1, 5 / 4, 3 / 2, 7 / 4]),  # I7 — septimal seventh, bittersweet
    ]

    chord_dur = 4.5  # seconds per chord (overlap with release tail)

    for start, partials in chords:
        for partial in partials:
            score.add_note(
                "pad",
                partial=partial,
                start=start,
                duration=chord_dur,
                amp_db=-8.0,
                velocity=0.75,
            )

    # A simple descending melody threaded through the upper register.
    melody: list[tuple[float, float, float]] = [
        # (start, partial, duration)
        (0.5, 4.0, 1.8),  # A4 — opens with the octave
        (2.5, 7 / 2, 1.3),  # septimal seventh
        (4.2, 3.0, 2.0),  # E4 — settles on the fifth
        (6.5, 8 / 3, 1.8),  # D4
        (8.8, 7 / 2, 1.5),  # septimal seventh again
        (10.5, 3.0, 2.2),  # E4
        (13.3, 5 / 2, 1.5),  # C#4
        (15.0, 7 / 4, 2.5),  # septimal seventh, low — final
    ]

    for start, partial, dur in melody:
        score.add_note(
            "pad",
            partial=partial,
            start=start,
            duration=dur,
            amp_db=-6.0,
            velocity=0.85,
        )

    return score


PIECES: dict[str, PieceDefinition] = {
    "study_surge_xt": PieceDefinition(
        name="study_surge_xt",
        output_name="study_surge_xt",
        build_score=build_score,
        study=True,
    ),
}
