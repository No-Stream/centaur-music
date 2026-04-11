"""BWV 846 Fugue -- Bach's Fugue in C Major (WTC Book I) in just intonation.

4-voice fugue retuned to JI using the same customized tuning table as the
Prelude.  Uses the Krüger MIDI which has separate tracks for each fugue
voice (soprano, alto, tenor, bass) and real velocity data.

The C major fugue is a good first test for JI counterpoint because:
  - The subject and answer stay within C major and G major
  - F# (45/32) is the only heavily-used chromatic pitch class
  - No enharmonic respellings are needed

The potential pitfalls (which we'll hear):
  - The D (9/8) serves as both the 5th of G and the root of Dm -- in
    strict 5-limit these would be different pitches (9/8 vs 10/9), but
    our static table picks one.  The Dm chord gets a slightly sharp
    minor 3rd (32/27 instead of 6/5).
  - Any passage where the subject overlaps with itself in stretto may
    expose comma-level beating between voices.
"""

from __future__ import annotations

import logging
from pathlib import Path

from code_musics.humanize import EnvelopeHumanizeSpec, VelocityHumanizeSpec
from code_musics.midi_import import read_midi
from code_musics.pieces._shared import REVERB_EFFECT, bwv_846_tuning
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import Score

logger: logging.Logger = logging.getLogger(__name__)

_MIDI_PATH = Path("midi_references/bach/bach_846.mid")

_VOICE_SEED_OFFSET: dict[str, int] = {"soprano": 0, "alto": 1, "tenor": 2, "bass": 3}

_F0 = 261.63  # C4
_ROOT_MIDI = 60

# Fugue voices live on tracks 3-6 in the Krüger MIDI.
_VOICE_TRACKS: dict[str, int] = {
    "soprano": 3,
    "alto": 4,
    "tenor": 5,
    "bass": 6,
}


def _fugue_voice_synth(n_harmonics: int, rolloff: float) -> dict:  # type: ignore[type-arg]
    """Additive synth for a fugue voice.  More sustained than the
    Prelude's harpsichord -- fugue voices need to sing through held notes
    and overlapping entries.
    """
    return {
        "engine": "additive",
        "params": {
            "n_harmonics": n_harmonics,
            "harmonic_rolloff": rolloff,
        },
        "env": {
            "attack_ms": 10.0,
            "decay_ms": 350.0,
            "sustain_ratio": 0.50,
            "release_ms": 300.0,
        },
    }


def build_bwv_846_fugue_score() -> Score:
    """Build a JI-retuned score of Bach's C Major Fugue from the Krüger MIDI."""
    midi = read_midi(_MIDI_PATH)
    table = bwv_846_tuning()

    # Collect fugue notes per voice, offset timing so the fugue starts at t=0.
    voice_notes: dict[str, list] = {}
    earliest = float("inf")
    for voice_name, track_idx in _VOICE_TRACKS.items():
        notes = [n for n in midi.notes if n.track == track_idx]
        if notes:
            earliest = min(earliest, min(n.start for n in notes))
        voice_notes[voice_name] = notes

    total_notes = sum(len(ns) for ns in voice_notes.values())
    logger.info(
        f"BWV 846 Fugue: {total_notes} notes across 4 voices, offset={earliest:.1f}s"
    )
    logger.info(f"Tuning table:\n{table.describe(root_midi_note=_ROOT_MIDI)}")

    score = Score(
        f0=_F0,
        master_effects=[REVERB_EFFECT],
    )

    # Voice-specific synth settings: soprano brightest, bass warmest.
    voice_config: dict[str, dict] = {
        "soprano": {
            "synth": _fugue_voice_synth(n_harmonics=10, rolloff=0.36),
            "mix_db": -1.0,
            "pan": 0.25,
        },
        "alto": {
            "synth": _fugue_voice_synth(n_harmonics=8, rolloff=0.40),
            "mix_db": -2.0,
            "pan": -0.15,
        },
        "tenor": {
            "synth": _fugue_voice_synth(n_harmonics=7, rolloff=0.44),
            "mix_db": -2.0,
            "pan": 0.15,
        },
        "bass": {
            "synth": _fugue_voice_synth(n_harmonics=6, rolloff=0.48),
            "mix_db": 0.0,
            "pan": -0.25,
        },
    }

    for voice_name, config in voice_config.items():
        score.add_voice(
            voice_name,
            synth_defaults=config["synth"],
            mix_db=config["mix_db"],
            pan=config["pan"],
            normalize_lufs=-20.0,
            velocity_humanize=VelocityHumanizeSpec(
                preset="subtle_living",
                seed=846 + _VOICE_SEED_OFFSET[voice_name],
            ),
            envelope_humanize=EnvelopeHumanizeSpec(
                preset="subtle_analog",
                seed=847 + _VOICE_SEED_OFFSET[voice_name],
            ),
        )

    for voice_name, notes in voice_notes.items():
        for note in notes:
            freq = table.resolve(note.midi_note, _F0, root_midi_note=_ROOT_MIDI)
            vel = note.velocity / 64.0  # MIDI velocity centered around 1.0
            score.add_note(
                voice_name,
                start=note.start - earliest,
                duration=note.duration,
                freq=freq,
                velocity=vel,
            )

    return score


PIECES: dict[str, PieceDefinition] = {
    "bwv_846_fugue_ji": PieceDefinition(
        name="bwv_846_fugue_ji",
        output_name="32_bwv_846_fugue_ji",
        build_score=build_bwv_846_fugue_score,
    ),
}
