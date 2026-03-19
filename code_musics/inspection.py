"""Timestamp-oriented score inspection helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from code_musics.analysis import analyze_score
from code_musics.pieces import PIECES
from code_musics.pieces.registry import PieceSection
from code_musics.score import Score


def parse_timestamp_seconds(value: str) -> float:
    """Parse ``SS``, ``MM:SS``, or ``HH:MM:SS`` into seconds."""
    normalized = value.strip()
    if not normalized:
        raise ValueError("timestamp cannot be empty")
    if ":" not in normalized:
        seconds = float(normalized)
        if seconds < 0:
            raise ValueError("timestamp must be non-negative")
        return seconds

    parts = normalized.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError("timestamp must be SS, MM:SS, or HH:MM:SS")
    multipliers = [1.0, 60.0, 3_600.0]
    total_seconds = 0.0
    for index, part in enumerate(reversed(parts)):
        if not part:
            raise ValueError("timestamp contains an empty field")
        total_seconds += float(part) * multipliers[index]
    if total_seconds < 0:
        raise ValueError("timestamp must be non-negative")
    return total_seconds


def inspect_piece_timestamp(
    *,
    piece_name: str,
    timestamp_seconds: float,
    window_seconds: float = 8.0,
    output_dir: str | Path = "output",
) -> dict[str, Any]:
    """Inspect a timestamp inside a registered score-backed piece."""
    if piece_name not in PIECES:
        raise ValueError(f"Unknown piece: {piece_name}")

    definition = PIECES[piece_name]
    if definition.build_score is None:
        raise ValueError(f"Piece {piece_name} does not expose a Score for inspection")

    score = definition.build_score()
    return inspect_score_timestamp(
        score=score,
        timestamp_seconds=timestamp_seconds,
        sections=definition.sections,
        window_seconds=window_seconds,
        artifact_paths=_discover_latest_artifacts(
            output_name=definition.output_name,
            output_dir=output_dir,
        ),
        piece_name=piece_name,
    )


def inspect_score_timestamp(
    *,
    score: Score,
    timestamp_seconds: float,
    sections: tuple[PieceSection, ...] = (),
    window_seconds: float = 8.0,
    artifact_paths: dict[str, str] | None = None,
    piece_name: str = "score",
) -> dict[str, Any]:
    """Build a compact inspection record for one score timestamp."""
    if timestamp_seconds < 0:
        raise ValueError("timestamp_seconds must be non-negative")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    analysis = analyze_score(score)
    resolved_notes = score.resolved_timing_notes()
    window_start = max(0.0, timestamp_seconds - (window_seconds / 2.0))
    window_end = min(score.total_dur, timestamp_seconds + (window_seconds / 2.0))
    section = _section_for_timestamp(
        sections=sections, timestamp_seconds=timestamp_seconds
    )
    active_notes = [
        note
        for note in resolved_notes
        if note.resolved_start <= timestamp_seconds < note.resolved_end
    ]
    nearby_onsets = [
        note
        for note in resolved_notes
        if window_start <= note.resolved_start < window_end
    ]
    drift_window = _drift_window_for_timestamp(
        windows=analysis.timing_drift_windows,
        timestamp_seconds=timestamp_seconds,
    )

    return {
        "piece_name": piece_name,
        "timestamp_seconds": timestamp_seconds,
        "timestamp_label": _format_timestamp(timestamp_seconds),
        "window_seconds": window_seconds,
        "window_start_seconds": window_start,
        "window_end_seconds": window_end,
        "section": section,
        "active_voice_names": sorted({note.voice_name for note in active_notes}),
        "active_notes": [_serialize_note(note) for note in active_notes],
        "nearby_onsets": [_serialize_note(note) for note in nearby_onsets],
        "global_drift": analysis.timing_drift_summary,
        "local_drift_window": drift_window,
        "artifacts": artifact_paths or {},
    }


def format_inspection_summary(inspection: dict[str, Any]) -> str:
    """Render an inspection payload as readable CLI text."""
    section = inspection["section"]
    section_text = (
        f"{section['label']} ({section['start_seconds']:.1f}-{section['end_seconds']:.1f}s)"
        if section is not None
        else "unknown"
    )
    lines = [
        f"Piece: {inspection['piece_name']}",
        f"At: {inspection['timestamp_label']} ({inspection['timestamp_seconds']:.2f}s)",
        f"Section: {section_text}",
        (
            "Window: "
            f"{inspection['window_start_seconds']:.2f}-{inspection['window_end_seconds']:.2f}s"
        ),
        "Active voices: " + (", ".join(inspection["active_voice_names"]) or "none"),
    ]

    global_drift = inspection["global_drift"]
    lines.append(
        "Drift: "
        f"mean abs offset {global_drift['mean_absolute_offset_ms']:.1f} ms, "
        f"p95 spread {global_drift['p95_inter_voice_spread_ms']:.1f} ms, "
        f"max spread {global_drift['max_inter_voice_spread_ms']:.1f} ms"
    )
    local_drift = inspection["local_drift_window"]
    if local_drift is not None:
        lines.append(
            "Local drift window: "
            f"{local_drift['start_seconds']:.1f}-{local_drift['end_seconds']:.1f}s, "
            f"mean spread {local_drift['mean_inter_voice_spread_ms']:.1f} ms, "
            f"max spread {local_drift['max_inter_voice_spread_ms']:.1f} ms"
        )

    lines.append("Active notes:")
    if inspection["active_notes"]:
        lines.extend(
            f"- {_format_note_line(note)}" for note in inspection["active_notes"]
        )
    else:
        lines.append("- none")

    lines.append("Nearby onsets:")
    if inspection["nearby_onsets"]:
        lines.extend(
            f"- {_format_note_line(note)}" for note in inspection["nearby_onsets"][:12]
        )
        remaining_onsets = len(inspection["nearby_onsets"]) - 12
        if remaining_onsets > 0:
            lines.append(f"- ... {remaining_onsets} more onsets in window")
    else:
        lines.append("- none")

    if inspection["artifacts"]:
        lines.append("Artifacts:")
        for artifact_name, artifact_path in sorted(inspection["artifacts"].items()):
            lines.append(f"- {artifact_name}: {artifact_path}")

    warnings = global_drift.get("warnings", [])
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines)


def _serialize_note(note: Any) -> dict[str, Any]:
    return {
        "voice_name": note.voice_name,
        "note_index": note.note_index,
        "authored_start_seconds": note.authored_start,
        "resolved_start_seconds": note.resolved_start,
        "resolved_end_seconds": note.resolved_end,
        "duration_seconds": note.duration,
        "timing_offset_ms": note.timing_offset_seconds * 1_000.0,
        "freq_hz": note.freq_hz,
        "partial": note.partial,
        "label": note.label,
    }


def _format_note_line(note: dict[str, Any]) -> str:
    pitch_text = (
        f"partial {note['partial']:.3f}"
        if note["partial"] is not None
        else f"{note['freq_hz']:.2f} Hz"
    )
    label_suffix = f' label="{note["label"]}"' if note["label"] else ""
    return (
        f"{note['voice_name']}[{note['note_index']}] "
        f"{note['resolved_start_seconds']:.2f}-{note['resolved_end_seconds']:.2f}s "
        f"offset {note['timing_offset_ms']:+.1f} ms {pitch_text}{label_suffix}"
    )


def _section_for_timestamp(
    *,
    sections: tuple[PieceSection, ...],
    timestamp_seconds: float,
) -> dict[str, Any] | None:
    for section in sections:
        if section.start_seconds <= timestamp_seconds < section.end_seconds:
            return {
                "label": section.label,
                "start_seconds": section.start_seconds,
                "end_seconds": section.end_seconds,
            }
    return None


def _drift_window_for_timestamp(
    *,
    windows: list[dict[str, Any]],
    timestamp_seconds: float,
) -> dict[str, Any] | None:
    for window in windows:
        if window["start_seconds"] <= timestamp_seconds < window["end_seconds"]:
            return window
    return None


def _discover_latest_artifacts(
    *,
    output_name: str,
    output_dir: str | Path,
) -> dict[str, str]:
    output_path = Path(output_dir) / output_name
    artifact_candidates = {
        "audio": output_path,
        "plot": output_path.with_suffix(".png"),
        "analysis_manifest": output_path.with_suffix(".analysis.json"),
        "timeline": output_path.with_suffix(".timeline.json"),
        "render_metadata": output_path.with_suffix(".render.json"),
    }
    artifacts = {
        artifact_name: str(path)
        for artifact_name, path in artifact_candidates.items()
        if path.exists()
    }
    analysis_manifest_path = output_path.with_suffix(".analysis.json")
    if analysis_manifest_path.exists():
        manifest = json.loads(analysis_manifest_path.read_text(encoding="utf-8"))
        timeline_path = manifest.get("score", {}).get("artifacts", {}).get("timeline")
        if timeline_path is not None:
            artifacts["timeline"] = str(timeline_path)
    return artifacts


def _format_timestamp(value_seconds: float) -> str:
    hours = int(value_seconds // 3_600)
    minutes = int((value_seconds % 3_600) // 60)
    seconds = value_seconds - (hours * 3_600) - (minutes * 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{seconds:05.2f}"
    if abs(seconds - round(seconds)) < 1e-9:
        return f"{minutes:d}:{int(round(seconds)):02d}"
    return f"{minutes:d}:{seconds:05.2f}"
