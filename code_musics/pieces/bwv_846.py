"""BWV 846 -- Bach's Prelude in C Major (WTC Book I) in just intonation.

Reads the Sankey MIDI performance and retunes it from 12-TET to JI,
using TuningTable.five_limit_major() as a baseline with hand-tuned
overrides for the chromatic moments.

The retuning workflow:
  1. read_midi() extracts note events from the MIDI file
  2. A customized TuningTable maps 12-TET pitch classes to JI ratios
  3. Two creative overrides refine the table's defaults:
     - C# -> 25/24 (augmented unison rather than the table's 16/15 minor 2nd)
     - Bb -> 7/4  (septimal minor 7th for resonant dominant 7th chords)
  4. Only the Prelude is imported (bars 1-56); the Fugue is a separate project.

The Sankey MIDI file contains both the Prelude and Fugue.  The Prelude
occupies roughly the first 112 seconds (a section break with empty bars
separates it from the Fugue).  We filter by time to extract just the
Prelude material.

The MIDI has flat velocity (all 127) so dynamics are shaped programmatically:
  - A smooth arc swells toward the harmonic peak (~bars 20-35) and recedes.
  - Bass (held) notes sit slightly louder to anchor the harmony.
  - Arpeggio notes carry a subtle downbeat accent within each group.

Harmonic analysis of chromatic moments in the Prelude:
  Bars  9-10, 15-16, 45:  F# (V/V region) -- 45/32 from the 5-limit table
  Bars 18-19:              C# + Bb -- C# overridden to 25/24, Bb to 7/4
  Bars 21-22, 36:          Ab -- 8/5 from the table
  Bars 31-32, 50-51:       Bb -- 7/4 override
  Bars 34-35, 43-44:       Eb -- 6/5 from the table
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from code_musics.automation import AutomationSegment, AutomationSpec, AutomationTarget
from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    VelocityHumanizeSpec,
)
from code_musics.midi_import import MidiNote, read_midi
from code_musics.pieces._shared import REVERB_EFFECT, bwv_846_tuning
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Score

logger: logging.Logger = logging.getLogger(__name__)

_MIDI_PATH = Path("midi_references/bach/well-tempered-clavier-i_bwv-846_(c)sankey.mid")

_F0 = 261.63  # C4 -- tonic of BWV 846
_ROOT_MIDI = 60  # MIDI note number for C4

# The Prelude ends around 112s; a clear section break (empty bars) precedes the Fugue.
_PRELUDE_END_SECONDS = 113.0

# Duration threshold separating bass (held) notes from arpeggio (short) notes.
# Bass notes are ~1.0-6.2s, arpeggio notes are ~0.2-1.0s.
_BASS_DURATION_THRESHOLD = 1.0


def _bass_synth() -> dict:  # type: ignore[type-arg]
    """Warm additive synth for the held bass and inner-voice notes.

    Fewer harmonics with gentler rolloff for warmth.  Longer sustain
    so the held notes breathe underneath the arpeggio.
    """
    return {
        "engine": "additive",
        "params": {
            "n_harmonics": 6,
            "harmonic_rolloff": 0.50,
        },
        "env": {
            "attack_ms": 12.0,
            "decay_ms": 400.0,
            "sustain_ratio": 0.45,
            "release_ms": 500.0,
        },
    }


def _arpeggio_synth() -> dict:  # type: ignore[type-arg]
    """Brighter additive synth for the arpeggiated notes.

    More harmonics with moderate rolloff for the characteristic
    harpsichord brightness.  Sharp attack, quick decay so the
    arpeggio stays articulate and the JI tuning rings clearly.
    """
    return {
        "engine": "additive",
        "params": {
            "n_harmonics": 10,
            "harmonic_rolloff": 0.38,
        },
        "env": {
            "attack_ms": 5.0,
            "decay_ms": 220.0,
            "sustain_ratio": 0.18,
            "release_ms": 350.0,
        },
    }


def _dynamic_arc(t: float, total_dur: float) -> float:
    """Smooth dynamic arc across the piece.

    Returns a velocity multiplier in ~[0.72, 1.0]:
      - Opens gently at ~0.78
      - Swells to 1.0 around 35-45% through (the secondary-dominant peak)
      - Settles back to ~0.82 for the closing
      - Final 8 seconds dim gently to ~0.65
    """
    progress = t / total_dur
    arc = 0.78 + 0.22 * math.sin(progress * math.pi * 0.95)
    if progress > 0.92:
        fade = (progress - 0.92) / 0.08
        arc *= 1.0 - 0.2 * fade
    return arc


def _is_group_downbeat(note: MidiNote, prev_note: MidiNote | None) -> bool:
    """Heuristic: a note is a 'downbeat' in the arpeggio if there's a gap or register drop."""
    if prev_note is None:
        return True
    gap = note.start - (prev_note.start + prev_note.duration)
    if gap > 0.05:
        return True
    return note.midi_note < prev_note.midi_note - 3


def build_bwv_846_score() -> Score:
    """Build a JI-retuned score of Bach's C Major Prelude from the Sankey MIDI."""
    midi = read_midi(_MIDI_PATH)
    table = bwv_846_tuning()

    logger.info(f"BWV 846: read {len(midi.notes)} notes, duration {midi.duration:.1f}s")
    logger.info(f"Tuning table:\n{table.describe(root_midi_note=_ROOT_MIDI)}")

    prelude_notes = [n for n in midi.notes if n.start < _PRELUDE_END_SECONDS]
    bass_notes = [n for n in prelude_notes if n.duration > _BASS_DURATION_THRESHOLD]
    arp_notes = [n for n in prelude_notes if n.duration <= _BASS_DURATION_THRESHOLD]

    logger.info(
        f"BWV 846: {len(prelude_notes)} prelude notes "
        f"({len(bass_notes)} bass, {len(arp_notes)} arp) "
        f"from {len(midi.notes)} total"
    )

    total_dur = max(n.start + n.duration for n in prelude_notes)

    score = Score(
        f0_hz=_F0,
        master_effects=[REVERB_EFFECT],
    )

    # Bass voice: warm, sustained, anchors the harmony.
    score.add_voice(
        "bass",
        synth_defaults=_bass_synth(),
        mix_db=0.0,
        normalize_lufs=-20.0,
        velocity_humanize=VelocityHumanizeSpec(
            preset="subtle_living",
            note_jitter=0.02,
            seed=846,
        ),
        envelope_humanize=EnvelopeHumanizeSpec(
            preset="subtle_analog",
            seed=847,
        ),
    )

    # Arpeggio voice: brighter, articulate, carries the melodic motion.
    score.add_voice(
        "arpeggio",
        synth_defaults=_arpeggio_synth(),
        mix_db=-2.0,
        normalize_lufs=-20.0,
        velocity_humanize=VelocityHumanizeSpec(
            preset="subtle_living",
            note_jitter=0.025,
            seed=848,
        ),
        envelope_humanize=EnvelopeHumanizeSpec(
            preset="subtle_analog",
            seed=849,
        ),
    )

    for note in bass_notes:
        freq = table.resolve(note.midi_note, _F0, root_midi_note=_ROOT_MIDI)
        vel = _dynamic_arc(note.start, total_dur) * 0.90
        score.add_note(
            "bass",
            start=note.start,
            duration=note.duration,
            freq=freq,
            velocity=vel,
        )

    prev_arp: MidiNote | None = None
    for note in arp_notes:
        freq = table.resolve(note.midi_note, _F0, root_midi_note=_ROOT_MIDI)
        arc = _dynamic_arc(note.start, total_dur)
        accent = 1.06 if _is_group_downbeat(note, prev_arp) else 1.0
        vel = arc * 0.82 * accent
        score.add_note(
            "arpeggio",
            start=note.start,
            duration=note.duration,
            freq=freq,
            velocity=vel,
        )
        prev_arp = note

    return score


# ---------------------------------------------------------------------------
# Piano version — same JI tuning, piano engine, richer humanization
# ---------------------------------------------------------------------------


def _piano_dynamic_arc(t: float, total_dur: float) -> float:
    """Wider dynamic arc for the piano version.

    Returns a velocity multiplier in ~[0.55, 1.0] — bigger range than the
    additive version, letting the piano's velocity-dependent hammer response
    do its thing.  The climax region (bars 20-35, ~35-45% through) pushes
    to forte; the opening and closing sit in piano-mezzo piano.
    """
    progress = t / total_dur
    arc = 0.62 + 0.38 * math.sin(progress * math.pi * 0.95)
    if progress > 0.92:
        fade = (progress - 0.92) / 0.08
        arc *= 1.0 - 0.35 * fade
    return arc


def _is_arp_top_note(note: MidiNote, idx: int, arp_notes: list[MidiNote]) -> bool:
    """Heuristic: is this the highest note in its local arpeggio group?

    Checks whether the note is higher than both its neighbors within
    a tight time window.  The top notes carry the implied melody.
    """
    window_sec = 0.6
    pitch = note.midi_note
    for other_idx in (idx - 1, idx + 1):
        if 0 <= other_idx < len(arp_notes):
            other = arp_notes[other_idx]
            if abs(other.start - note.start) < window_sec and other.midi_note > pitch:
                return False
    for other_idx in (idx - 2, idx + 2):
        if 0 <= other_idx < len(arp_notes):
            other = arp_notes[other_idx]
            if abs(other.start - note.start) < window_sec and other.midi_note > pitch:
                return False
    return pitch > 60


def build_bwv_846_piano_score() -> Score:
    """BWV 846 in JI with the piano engine.

    Key differences from the additive version:
    - Piano engine with inharmonicity=0 for pure JI partial alignment
    - Wider dynamic arc (pp to f) exploiting velocity-dependent hammer
    - Top-note melodic accents in the arpeggio
    - Hammer hardness automation following the piece's harmonic intensity
    - Chamber timing humanization for natural rubato
    - Richer reverb (Bricasti large hall, slightly wetter)
    """
    midi = read_midi(_MIDI_PATH)
    table = bwv_846_tuning()

    prelude_notes = [n for n in midi.notes if n.start < _PRELUDE_END_SECONDS]
    bass_notes = [n for n in prelude_notes if n.duration > _BASS_DURATION_THRESHOLD]
    arp_notes = [n for n in prelude_notes if n.duration <= _BASS_DURATION_THRESHOLD]

    total_dur = max(n.start + n.duration for n in prelude_notes)

    # Slightly wetter reverb for piano — the space is part of the instrument.
    piano_reverb = EffectSpec(
        "reverb", {"room_size": 0.82, "damping": 0.55, "wet_level": 0.38}
    )

    score = Score(
        f0_hz=_F0,
        master_effects=[piano_reverb],
        timing_humanize=TimingHumanizeSpec(
            preset="chamber",
            micro_jitter_ms=0.5,
            seed=8460,
        ),
    )

    # Bass voice: warm felt-ish piano, soft hammer, sustained.
    # Tiny inharmonicity (much less than a real piano) keeps the JI intervals
    # clean while breaking the pure-additive comb-filter quality.
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "piano",
            "preset": "warm",
            "inharmonicity": 0.0005,
            "decay_base": 5.0,
            "soundboard_color": 0.55,
            "attack": 0.003,
            "sustain_level": 1.0,
            "release": 0.15,
        },
        mix_db=-1.0,
        normalize_lufs=-22.0,
        velocity_humanize=VelocityHumanizeSpec(
            preset="breathing_ensemble",
            note_jitter=0.02,
            seed=8461,
        ),
        envelope_humanize=EnvelopeHumanizeSpec(
            preset="subtle_analog",
            seed=8462,
        ),
    )

    # Arpeggio voice: brighter piano, articulate attack.
    # Hammer hardness is automated to follow the piece's intensity.
    score.add_voice(
        "arpeggio",
        synth_defaults={
            "engine": "piano",
            "preset": "grand",
            "inharmonicity": 0.0005,
            "soundboard_color": 0.45,
            "attack": 0.002,
            "sustain_level": 1.0,
            "release": 0.12,
        },
        mix_db=0.0,
        normalize_lufs=-22.0,
        velocity_humanize=VelocityHumanizeSpec(
            preset="breathing_ensemble",
            note_jitter=0.03,
            seed=8463,
        ),
        envelope_humanize=EnvelopeHumanizeSpec(
            preset="subtle_analog",
            seed=8464,
        ),
        automation=[
            # Hammer hardness follows the piece's harmonic intensity:
            # soft and warm in the opening, brightens into the climax,
            # settles back for the coda.
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="hammer_hardness"),
                segments=(
                    AutomationSegment(
                        start=0.0,
                        end=total_dur * 0.15,
                        shape="linear",
                        start_value=0.35,
                        end_value=0.40,
                    ),
                    AutomationSegment(
                        start=total_dur * 0.15,
                        end=total_dur * 0.40,
                        shape="linear",
                        start_value=0.40,
                        end_value=0.62,
                    ),
                    AutomationSegment(
                        start=total_dur * 0.40,
                        end=total_dur * 0.55,
                        shape="linear",
                        start_value=0.62,
                        end_value=0.55,
                    ),
                    AutomationSegment(
                        start=total_dur * 0.55,
                        end=total_dur * 0.85,
                        shape="linear",
                        start_value=0.55,
                        end_value=0.42,
                    ),
                    AutomationSegment(
                        start=total_dur * 0.85,
                        end=total_dur,
                        shape="linear",
                        start_value=0.42,
                        end_value=0.30,
                    ),
                ),
            ),
        ],
    )

    for note in bass_notes:
        freq = table.resolve(note.midi_note, _F0, root_midi_note=_ROOT_MIDI)
        vel = _piano_dynamic_arc(note.start, total_dur) * 0.85
        score.add_note(
            "bass",
            start=note.start,
            duration=note.duration,
            freq=freq,
            velocity=vel,
        )

    prev_arp: MidiNote | None = None
    for idx, note in enumerate(arp_notes):
        freq = table.resolve(note.midi_note, _F0, root_midi_note=_ROOT_MIDI)
        arc = _piano_dynamic_arc(note.start, total_dur)

        downbeat_accent = 1.08 if _is_group_downbeat(note, prev_arp) else 1.0
        top_note_accent = 1.10 if _is_arp_top_note(note, idx, arp_notes) else 1.0
        vel = arc * 0.78 * downbeat_accent * top_note_accent

        score.add_note(
            "arpeggio",
            start=note.start,
            duration=note.duration,
            freq=freq,
            velocity=vel,
        )
        prev_arp = note

    return score


PIECES: dict[str, PieceDefinition] = {
    "bwv_846_ji": PieceDefinition(
        name="bwv_846_ji",
        output_name="31_bwv_846_ji",
        build_score=build_bwv_846_score,
    ),
    "bwv_846_piano": PieceDefinition(
        name="bwv_846_piano",
        output_name="32_bwv_846_piano",
        build_score=build_bwv_846_piano_score,
    ),
}
