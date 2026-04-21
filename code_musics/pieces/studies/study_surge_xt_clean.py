"""Study for Surge XT (clean) — pure JI clarity through analog synthesis.

Same septimal chord progression as study_surge_xt, but with a single
oscillator voice (no unison detune) and a slightly brighter filter to
let the just intervals speak for themselves.  This is the "control" —
proof that the MPE pitch bend path delivers clean, accurate JI tuning
through Surge XT without the warble of the detuned version.

Voice layout:
  - pad: Surge XT (Classic saw, 1 voice, LP Vintage Ladder, slow env)

Harmonic language: 7-limit JI centred on A2 (110 Hz).
"""

from __future__ import annotations

from code_musics.pieces._shared import SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import Score

F0_HZ: float = 110.0

# Clean patch: single oscillator, no unison, slightly brighter filter.
_CLEAN_SURGE_PARAMS: dict[str, float] = {
    # Oscillator: single Classic saw — no chorus, no beating
    "a_osc_1_type": 0.0,  # Classic
    "a_osc_1_shape": 0.5,  # saw
    "a_osc_1_unison_voices": 0.0,  # 1 voice
    "a_osc_1_unison_detune": 0.0,  # no detune
    # Filter: Vintage Ladder LP, slightly brighter to let harmonics breathe
    "a_filter_1_type": 0.295,  # LP Vintage Ladder
    "a_filter_1_cutoff": 0.57,  # ~700 Hz
    "a_filter_1_resonance": 0.08,  # 8% — minimal colouration
    # Amp envelope: same slow pad shape
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
            "surge_params": _CLEAN_SURGE_PARAMS,
            "tail_seconds": 2.5,
        },
        normalize_lufs=-20.0,
    )

    # Same chord progression as the detuned study.
    chords: list[tuple[float, list[float]]] = [
        (0.0, [1, 5 / 2, 3]),  # I — open A major
        (4.0, [4 / 3, 7 / 3, 8 / 3]),  # iv7 — septimal D minor colour
        (8.5, [5 / 3, 2, 7 / 2]),  # vi — with harmonic 7th
        (13.0, [1, 5 / 4, 3 / 2, 7 / 4]),  # I7 — septimal seventh
    ]

    chord_dur = 4.5

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

    # Same melody.
    melody: list[tuple[float, float, float]] = [
        (0.5, 4.0, 1.8),
        (2.5, 7 / 2, 1.3),
        (4.2, 3.0, 2.0),
        (6.5, 8 / 3, 1.8),
        (8.8, 7 / 2, 1.5),
        (10.5, 3.0, 2.2),
        (13.3, 5 / 2, 1.5),
        (15.0, 7 / 4, 2.5),
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
    "study_surge_xt_clean": PieceDefinition(
        name="study_surge_xt_clean",
        output_name="study_surge_xt_clean",
        build_score=build_score,
        study=True,
    ),
}
