"""Named piece rendering workflow."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from code_musics.analysis import save_analysis_artifacts
from code_musics.pieces import PIECES
from code_musics.score import Score
from code_musics.synth import SAMPLE_RATE, write_wav

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderResult:
    """Paths and metadata emitted by a render."""

    audio_path: Path
    plot_path: Path | None = None
    analysis_manifest_path: Path | None = None
    analysis_artifacts: dict | None = None
    render_metadata_path: Path | None = None
    version_audio_path: Path | None = None
    version_plot_path: Path | None = None
    version_analysis_manifest_path: Path | None = None
    version_metadata_path: Path | None = None

    def __iter__(self) -> Iterator[Path | None]:
        """Preserve tuple-style unpacking for legacy callers."""
        yield self.audio_path
        yield self.plot_path


def list_pieces() -> list[str]:
    """Return the registered piece names."""
    return sorted(PIECES)


def render_piece(
    piece_name: str,
    *,
    output_dir: str | Path = "output",
    save_plot: bool = False,
    save_analysis: bool = True,
) -> RenderResult:
    """Render a registered piece by name."""
    if piece_name not in PIECES:
        raise ValueError(f"Unknown piece: {piece_name}")

    definition = PIECES[piece_name]
    output_path = Path(output_dir) / definition.output_name
    version_timestamp = _build_version_timestamp()
    version_output_path = _build_version_audio_path(
        piece_name=piece_name,
        output_path=output_path,
        version_timestamp=version_timestamp,
    )
    plot_path: Path | None = None
    analysis_manifest_path: Path | None = None
    analysis_artifacts: dict | None = None
    version_plot_path: Path | None = None
    version_analysis_manifest_path: Path | None = None
    version_analysis_artifacts: dict[str, Any] | None = None
    score: Score | None = None

    if definition.build_score is not None:
        score = definition.build_score()
        audio = score.render()
        if save_plot:
            version_plot_path = version_output_path.with_suffix(".png")
            figure, _ = score.plot_piano_roll(version_plot_path)
            plt.close(figure)
        if save_analysis:
            version_analysis_artifacts = save_analysis_artifacts(
                output_prefix=version_output_path.with_suffix(""),
                mix_signal=audio,
                sample_rate=score.sample_rate,
                stems=score.render_stems(),
                score=score,
            )
            version_analysis_manifest_path = Path(
                str(version_analysis_artifacts["manifest_path"])
            )
    elif definition.render_audio is not None:
        audio = definition.render_audio()
        if save_analysis:
            version_analysis_artifacts = save_analysis_artifacts(
                output_prefix=version_output_path.with_suffix(""),
                mix_signal=audio,
                sample_rate=SAMPLE_RATE,
            )
            version_analysis_manifest_path = Path(
                str(version_analysis_artifacts["manifest_path"])
            )
    else:
        raise ValueError(f"Piece {piece_name} has no render path configured")

    write_wav(version_output_path, audio)
    shutil.copy2(version_output_path, output_path)

    if version_plot_path is not None:
        plot_path = output_path.with_suffix(".png")
        shutil.copy2(version_plot_path, plot_path)

    if version_analysis_artifacts is not None:
        analysis_artifacts = _copy_analysis_artifacts_to_latest(
            version_analysis_artifacts=version_analysis_artifacts,
            version_prefix=version_output_path.with_suffix(""),
            latest_prefix=output_path.with_suffix(""),
        )
        analysis_manifest_path = Path(str(analysis_artifacts["manifest_path"]))

    version_metadata_path = version_output_path.with_suffix(".render.json")
    render_metadata_path = output_path.with_suffix(".render.json")
    versioned_artifacts = _artifact_paths(
        audio_path=version_output_path,
        plot_path=version_plot_path,
        analysis_manifest_path=version_analysis_manifest_path,
    )
    latest_artifacts = _artifact_paths(
        audio_path=output_path,
        plot_path=plot_path,
        analysis_manifest_path=analysis_manifest_path,
    )

    version_metadata = _build_render_metadata(
        piece_name=piece_name,
        definition=definition,
        output_dir=Path(output_dir),
        version_timestamp=version_timestamp,
        save_plot=save_plot,
        save_analysis=save_analysis,
        score=score,
        latest_artifacts=latest_artifacts,
        versioned_artifacts=versioned_artifacts,
    )
    version_metadata["metadata_path"] = str(version_metadata_path)
    version_metadata["version_metadata_path"] = str(version_metadata_path)
    version_metadata_path.write_text(
        json.dumps(version_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    latest_metadata = dict(version_metadata)
    latest_metadata["metadata_path"] = str(render_metadata_path)
    latest_metadata["version_metadata_path"] = str(version_metadata_path)
    render_metadata_path.write_text(
        json.dumps(latest_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logger.info("Rendered %s", piece_name)
    if analysis_manifest_path is not None:
        logger.info("Analysis manifest: %s", analysis_manifest_path)
    logger.info("Versioned render: %s", version_output_path)
    return RenderResult(
        audio_path=output_path,
        plot_path=plot_path,
        analysis_manifest_path=analysis_manifest_path,
        analysis_artifacts=analysis_artifacts,
        render_metadata_path=render_metadata_path,
        version_audio_path=version_output_path,
        version_plot_path=version_plot_path,
        version_analysis_manifest_path=version_analysis_manifest_path,
        version_metadata_path=version_metadata_path,
    )


def _build_version_timestamp() -> str:
    """Return a second-level UTC render timestamp."""
    return datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")


def _build_version_audio_path(
    *,
    piece_name: str,
    output_path: Path,
    version_timestamp: str,
) -> Path:
    """Return the archive location for a versioned audio render."""
    version_dir = output_path.parent / "versions" / piece_name
    version_stem = f"{output_path.stem}__{version_timestamp}"
    return version_dir / f"{version_stem}{output_path.suffix}"


def _copy_analysis_artifacts_to_latest(
    *,
    version_analysis_artifacts: dict[str, Any],
    version_prefix: Path,
    latest_prefix: Path,
) -> dict[str, Any]:
    """Mirror versioned analysis files into the stable latest filenames."""
    latest_analysis_artifacts = _rewrite_prefixed_paths(
        payload=version_analysis_artifacts,
        source_prefix_name=version_prefix.name,
        target_prefix_name=latest_prefix.name,
    )
    for source_path, target_path in _collect_rewritten_paths(
        source_payload=version_analysis_artifacts,
        rewritten_payload=latest_analysis_artifacts,
    ):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
    return latest_analysis_artifacts


def _rewrite_prefixed_paths(
    *,
    payload: Any,
    source_prefix_name: str,
    target_prefix_name: str,
) -> Any:
    """Replace a versioned artifact prefix with the stable output prefix."""
    if isinstance(payload, dict):
        return {
            key: _rewrite_prefixed_paths(
                payload=value,
                source_prefix_name=source_prefix_name,
                target_prefix_name=target_prefix_name,
            )
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [
            _rewrite_prefixed_paths(
                payload=value,
                source_prefix_name=source_prefix_name,
                target_prefix_name=target_prefix_name,
            )
            for value in payload
        ]
    if isinstance(payload, str):
        path = Path(payload)
        if path.name.startswith(source_prefix_name):
            return str(
                path.with_name(
                    path.name.replace(source_prefix_name, target_prefix_name, 1)
                )
            )
    return payload


def _collect_rewritten_paths(
    *,
    source_payload: Any,
    rewritten_payload: Any,
) -> list[tuple[Path, Path]]:
    """Collect source/target file pairs from two parallel JSON payloads."""
    path_pairs: set[tuple[Path, Path]] = set()

    def _visit(source_value: Any, target_value: Any) -> None:
        if isinstance(source_value, dict) and isinstance(target_value, dict):
            for key in source_value:
                if key in target_value:
                    _visit(source_value[key], target_value[key])
            return
        if isinstance(source_value, list) and isinstance(target_value, list):
            for source_item, target_item in zip(
                source_value,
                target_value,
                strict=True,
            ):
                _visit(source_item, target_item)
            return
        if isinstance(source_value, str) and isinstance(target_value, str):
            source_path = Path(source_value)
            target_path = Path(target_value)
            if source_path.exists():
                path_pairs.add((source_path, target_path))

    _visit(source_payload, rewritten_payload)
    return sorted(path_pairs, key=lambda pair: str(pair[1]))


def _artifact_paths(
    *,
    audio_path: Path,
    plot_path: Path | None,
    analysis_manifest_path: Path | None,
) -> dict[str, str]:
    """Build the artifact path block stored in render metadata."""
    artifact_paths: dict[str, str] = {"audio_path": str(audio_path)}
    if plot_path is not None:
        artifact_paths["plot_path"] = str(plot_path)
    if analysis_manifest_path is not None:
        artifact_paths["analysis_manifest_path"] = str(analysis_manifest_path)
    return artifact_paths


def _build_render_metadata(
    *,
    piece_name: str,
    definition: Any,
    output_dir: Path,
    version_timestamp: str,
    save_plot: bool,
    save_analysis: bool,
    score: Score | None,
    latest_artifacts: dict[str, str],
    versioned_artifacts: dict[str, str],
) -> dict[str, Any]:
    """Build a small JSON record for the current render request."""
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "rendered_at_utc": version_timestamp,
        "piece_name": piece_name,
        "output_name": definition.output_name,
        "metadata_path": "",
        "version_metadata_path": "",
        "request": {
            "output_dir": str(output_dir),
            "save_plot": save_plot,
            "save_analysis": save_analysis,
        },
        "artifacts": {
            "latest": latest_artifacts,
            "versioned": versioned_artifacts,
        },
        "provenance": _build_provenance(definition),
    }
    if score is not None:
        metadata["score_summary"] = _build_score_summary(score)
        metadata["score_snapshot"] = _serialize_value(score)
    return metadata


def _build_provenance(definition: Any) -> dict[str, Any]:
    """Capture enough code provenance to relate renders to source changes."""
    render_callable = definition.build_score or definition.render_audio
    if render_callable is None:
        return {}

    try:
        source_text = inspect.getsource(render_callable)
    except (OSError, TypeError):
        source_text = None

    provenance: dict[str, Any] = {
        "callable_name": render_callable.__name__,
        "callable_kind": "build_score"
        if definition.build_score is not None
        else "render_audio",
        "module": render_callable.__module__,
        "source_file": inspect.getsourcefile(render_callable),
    }
    if source_text is not None:
        provenance["source_sha256"] = hashlib.sha256(
            source_text.encode("utf-8")
        ).hexdigest()

    git_commit = _git_output(["git", "rev-parse", "HEAD"])
    git_status = _git_output(["git", "status", "--short", "--untracked-files=no"])
    if git_commit is not None:
        provenance["git_commit"] = git_commit
    if git_status is not None:
        provenance["git_is_dirty"] = bool(git_status)
    return provenance


def _git_output(command: list[str]) -> str | None:
    """Run a small git command and return trimmed stdout when available."""
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _build_score_summary(score: Score) -> dict[str, Any]:
    """Return a compact human-readable score summary."""
    voice_summaries: dict[str, dict[str, Any]] = {}
    total_notes = 0
    for voice_name, voice in score.voices.items():
        total_notes += len(voice.notes)
        labels = [note.label for note in voice.notes if note.label is not None]
        partials = [
            float(note.partial) for note in voice.notes if note.partial is not None
        ]
        freqs = [float(note.freq) for note in voice.notes if note.freq is not None]
        voice_summaries[voice_name] = {
            "note_count": len(voice.notes),
            "pan": voice.pan,
            "synth_defaults": voice.synth_defaults,
            "effects": _serialize_value(voice.effects),
            "labels": sorted(set(labels)),
            "partial_range": [min(partials), max(partials)] if partials else None,
            "frequency_range_hz": [min(freqs), max(freqs)] if freqs else None,
        }
    return {
        "f0_hz": score.f0,
        "sample_rate": score.sample_rate,
        "total_duration_seconds": score.total_dur,
        "voice_count": len(score.voices),
        "note_count": total_notes,
        "voice_names": list(score.voices),
        "master_effects": _serialize_value(score.master_effects),
        "voices": voice_summaries,
    }


def _serialize_value(value: Any) -> Any:
    """Convert score-related objects into JSON-serializable structures."""
    if is_dataclass(value):
        return {
            field_name: _serialize_value(field_value)
            for field_name, field_value in value.__dict__.items()
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
