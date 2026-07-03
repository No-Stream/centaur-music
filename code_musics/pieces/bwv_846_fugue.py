"""BWV 846 Fugue -- Bach's Fugue in C Major (WTC Book I) in just intonation.

4-voice fugue retuned to JI using the same customized tuning table as the
Prelude.  Prefers a local Krüger MIDI when available because it has separate
tracks for each fugue voice (soprano, alto, tenor, bass).  Fresh checkouts use
the verified Sankey BWV 846 MIDI reference and infer voice lanes from the fugue
section.

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
from code_musics.midi_import import MidiImportResult, MidiNote, read_midi
from code_musics.pieces._shared import REVERB_EFFECT, bwv_846_tuning
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import Score

logger: logging.Logger = logging.getLogger(__name__)

_MIDI_PATH = Path("midi_references/bach/bach_846.mid")
_SANKEY_MIDI_PATH = Path(
    "midi_references/bach/well-tempered-clavier-i_bwv-846_(c)sankey.mid"
)

_VOICE_SEED_OFFSET: dict[str, int] = {"soprano": 0, "alto": 1, "tenor": 2, "bass": 3}
_VOICE_NAMES: tuple[str, ...] = ("soprano", "alto", "tenor", "bass")
_VOICE_PITCH_CENTERS: dict[str, int] = {
    "soprano": 72,
    "alto": 65,
    "tenor": 58,
    "bass": 50,
}

_F0 = 261.63  # C4
_ROOT_MIDI = 60

# Fugue voices live on tracks 3-6 in the Krüger MIDI.
_VOICE_TRACKS: dict[str, int] = {
    "soprano": 3,
    "alto": 4,
    "tenor": 5,
    "bass": 6,
}

# In the Sankey MIDI, the Prelude ends before a short break; the first fugue
# note starts around 116.4 s in the current upstream file.
_SANKEY_FUGUE_START_SECONDS = 113.0


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


def _read_bwv_846_midi() -> MidiImportResult:
    if _MIDI_PATH.exists():
        return read_midi(_MIDI_PATH)
    if _SANKEY_MIDI_PATH.exists():
        logger.warning(
            "Krüger BWV 846 fugue MIDI not found at %s; using Sankey MIDI "
            "with inferred fugue voice lanes.",
            _MIDI_PATH,
        )
        return read_midi(_SANKEY_MIDI_PATH)
    raise FileNotFoundError(
        f"Reference MIDI not found: {_MIDI_PATH} or {_SANKEY_MIDI_PATH}. "
        "Run `make fetch-midi-references` to install the verified Sankey MIDI asset."
    )


def _split_sankey_fugue_voices(notes: list[MidiNote]) -> dict[str, list[MidiNote]]:
    """Infer stable fugue voice lanes from the single-track Sankey MIDI."""
    fugue_notes = [note for note in notes if note.start >= _SANKEY_FUGUE_START_SECONDS]
    if not fugue_notes:
        raise ValueError("Sankey BWV 846 MIDI has no notes in the fugue section")

    voice_notes: dict[str, list[MidiNote]] = {
        voice_name: [] for voice_name in _VOICE_NAMES
    }
    voice_last_pitch = dict(_VOICE_PITCH_CENTERS)
    voice_available_at = {voice_name: 0.0 for voice_name in _VOICE_NAMES}

    for note in fugue_notes:
        candidate_voices = [
            voice_name
            for voice_name in _VOICE_NAMES
            if voice_available_at[voice_name] <= note.start + 0.02
        ]
        if not candidate_voices:
            candidate_voices = list(_VOICE_NAMES)

        voice_name = min(
            candidate_voices,
            key=lambda candidate: (
                abs(note.midi_note - voice_last_pitch[candidate])
                + 0.35 * abs(note.midi_note - _VOICE_PITCH_CENTERS[candidate]),
                _VOICE_NAMES.index(candidate),
            ),
        )
        voice_notes[voice_name].append(note)
        voice_last_pitch[voice_name] = note.midi_note
        voice_available_at[voice_name] = note.start + note.duration

    return voice_notes


def _collect_voice_notes(midi: MidiImportResult) -> dict[str, list[MidiNote]]:
    voice_notes = {
        voice_name: [note for note in midi.notes if note.track == track_idx]
        for voice_name, track_idx in _VOICE_TRACKS.items()
    }
    if any(voice_notes.values()):
        return voice_notes
    return _split_sankey_fugue_voices(midi.notes)


def build_bwv_846_fugue_score() -> Score:
    """Build a JI-retuned score of Bach's C Major Fugue."""
    midi = _read_bwv_846_midi()
    table = bwv_846_tuning()

    # Collect fugue notes per voice, offset timing so the fugue starts at t=0.
    voice_notes = _collect_voice_notes(midi)
    earliest = float("inf")
    for notes in voice_notes.values():
        if notes:
            earliest = min(earliest, min(n.start for n in notes))
    if earliest == float("inf"):
        raise ValueError("BWV 846 fugue MIDI did not produce any voice notes")

    total_notes = sum(len(ns) for ns in voice_notes.values())
    logger.info(
        f"BWV 846 Fugue: {total_notes} notes across 4 voices, offset={earliest:.1f}s"
    )
    logger.info(f"Tuning table:\n{table.describe(root_midi_note=_ROOT_MIDI)}")

    score = Score(
        f0_hz=_F0,
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
