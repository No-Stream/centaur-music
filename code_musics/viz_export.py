"""Generic visualization-JSON exporter for score-backed pieces.

Produces a single machine-readable JSON payload combining note timing,
voice metadata, section boundaries, an RMS loudness envelope, and
free-form per-piece annotations. Intended as the common data source for
downstream visualization/video tooling.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from code_musics.pieces.registry import PieceSection
from code_musics.score import Score

logger = logging.getLogger(__name__)

VIZ_SCHEMA_VERSION = 1

_SILENCE_FLOOR_DB = -120.0


def parse_viz_label(label: str | None) -> dict[str, Any] | None:
    """Parse a note label into structured viz semantics.

    Format is semicolon-separated tokens: the first token is the ``kind``,
    remaining tokens are ``key=value`` pairs (coerced to int, then float,
    else left as str) collected into ``tags``. A bare token with no ``=``
    becomes ``{token: True}`` in ``tags``. A legacy plain label (no ``;``
    and no ``=``) is treated as a bare ``kind`` with empty tags.

    Never raises: malformed segments are skipped silently. Labels are a
    creative authoring surface and viz export must not break renders.
    """
    if not label:
        return None

    segments = label.split(";")
    kind = segments[0].strip()
    if not kind:
        return None

    tags: dict[str, Any] = {}
    for segment in segments[1:]:
        segment = segment.strip()
        if not segment:
            continue
        if "=" in segment:
            key, _, raw_value = segment.partition("=")
            key = key.strip()
            raw_value = raw_value.strip()
            if not key or not raw_value:
                continue
            tags[key] = _coerce_tag_value(raw_value)
        else:
            tags[segment] = True

    return {"kind": kind, "tags": tags}


def _coerce_tag_value(raw_value: str) -> int | float | str:
    """Coerce a raw tag value string to int, then float, else str."""
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        pass
    return raw_value


@dataclass(frozen=True)
class VizExportSpec:
    """Configuration for a single viz-JSON export."""

    piece_name: str
    output_name: str
    envelope_hop_seconds: float = 0.025


@dataclass(frozen=True)
class VizExportResult:
    """Paths and summary counts emitted by a viz export."""

    viz_path: Path
    note_count: int
    envelope_frame_count: int


def build_rms_envelope(
    signal: np.ndarray, *, sample_rate: int, hop_seconds: float
) -> dict[str, Any]:
    """Build a framewise RMS loudness envelope from a (possibly stereo) signal."""
    if hop_seconds <= 0:
        raise ValueError("hop_seconds must be positive")

    mono = signal.mean(axis=0) if signal.ndim == 2 else signal

    hop_samples = max(1, int(round(hop_seconds * sample_rate)))
    frame_samples = 2 * hop_samples

    total_samples = mono.shape[0]
    frame_count = max(0, (total_samples - 1) // hop_samples + 1) if total_samples else 0

    pad_amount = max(0, (frame_count - 1) * hop_samples + frame_samples - total_samples)
    padded = np.pad(mono, (0, pad_amount))

    if frame_count > 0:
        frame_starts = np.arange(frame_count) * hop_samples
        frame_indices = frame_starts[:, None] + np.arange(frame_samples)[None, :]
        frames = padded[frame_indices]
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    else:
        rms = np.zeros(0, dtype=np.float64)

    with np.errstate(divide="ignore"):
        rms_db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    rms_db = np.maximum(rms_db, _SILENCE_FLOOR_DB)

    return {
        "hop_seconds": hop_seconds,
        "frame_count": int(frame_count),
        "rms": [round(float(value), 5) for value in rms],
        "rms_db": [round(float(value), 3) for value in rms_db],
    }


def build_viz_payload(
    *,
    score: Score,
    sections: tuple[PieceSection, ...] = (),
    annotations: dict[str, Any] | None = None,
    mix_wav_path: Path,
    spec: VizExportSpec,
) -> dict[str, Any]:
    """Build the full viz-JSON payload for a score-backed piece."""
    if not mix_wav_path.exists():
        raise FileNotFoundError(
            f"Mix WAV not found at {mix_wav_path}. Render the piece first, e.g. "
            f"`make render PIECE={spec.piece_name}`."
        )

    signal, sample_rate = sf.read(str(mix_wav_path), always_2d=False)
    signal_array = np.asarray(signal)
    if signal_array.ndim == 2:
        signal_array = signal_array.T

    envelope = build_rms_envelope(
        signal_array, sample_rate=sample_rate, hop_seconds=spec.envelope_hop_seconds
    )

    voice_note_counts: dict[str, int] = {
        voice_name: len(voice.notes) for voice_name, voice in score.voices.items()
    }
    voices = {
        voice_name: {
            "pan": voice.pan,
            "mix_db": voice.mix_db,
            "is_percussive": voice.is_percussive(),
            "note_count": voice_note_counts[voice_name],
        }
        for voice_name, voice in score.voices.items()
    }

    resolved_notes = sorted(
        score.resolved_timing_notes(),
        key=lambda note: (note.resolved_start, note.voice_name, note.note_index),
    )
    notes = []
    for note in resolved_notes:
        source_note = score.voices[note.voice_name].notes[note.note_index]
        note_payload = {
            "voice_name": note.voice_name,
            "note_index": note.note_index,
            "start_seconds": round(note.resolved_start, 4),
            "end_seconds": round(note.resolved_end, 4),
            "duration_seconds": round(note.duration, 4),
            "authored_start_seconds": round(note.authored_start, 4),
            "freq_hz": round(note.freq_hz, 4),
            "partial": note.partial,
            "velocity": round(source_note.velocity, 5),
            "amp": round(source_note.amp, 5) if source_note.amp is not None else None,
            "amp_db": (
                round(source_note.amp_db, 3) if source_note.amp_db is not None else None
            ),
            "label": note.label,
            "semantics": parse_viz_label(note.label),
        }
        if score.timeline is not None:
            note_payload["musical_location"] = _musical_location_metadata(
                score.timeline,
                note.resolved_start,
            )
            note_payload["authored_musical_location"] = _musical_location_metadata(
                score.timeline,
                note.authored_start,
            )
        notes.append(note_payload)

    payload: dict[str, Any] = {
        "schema_version": VIZ_SCHEMA_VERSION,
        "piece_name": spec.piece_name,
        "total_duration_seconds": round(score.total_dur, 4),
        "sample_rate": int(sample_rate),
        "f0_hz": score.f0_hz,
        "sections": [
            {
                "label": section.label,
                "start_seconds": section.start_seconds,
                "end_seconds": section.end_seconds,
            }
            for section in sections
        ],
        "voices": voices,
        "notes": notes,
        "annotations": annotations if annotations is not None else {},
        "envelope": envelope,
    }
    if score.timeline is not None:
        payload["musical_time"] = score.timeline.to_metadata()
    return payload


def _musical_location_metadata(timeline: Any, seconds: float) -> dict[str, float | int]:
    location = timeline.locate(seconds)
    return {
        "bar": location.bar,
        "beat": location.beat,
        "absolute_beats": location.absolute_beats,
        "seconds": location.seconds,
    }


def export_viz_json(payload: dict[str, Any], output_path: Path) -> VizExportResult:
    """Write a viz payload to disk as compact JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as viz_file:
        json.dump(payload, viz_file, separators=(",", ":"))

    return VizExportResult(
        viz_path=output_path,
        note_count=len(payload["notes"]),
        envelope_frame_count=payload["envelope"]["frame_count"],
    )
