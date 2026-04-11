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
from code_musics.midi_export import (
    ALL_STEM_FORMATS,
    MidiBundleExportResult,
    MidiBundleExportSpec,
    MidiStemFormat,
    export_midi_bundle,
)
from code_musics.pieces import PIECES
from code_musics.score import Score
from code_musics.synth import (
    SAMPLE_RATE,
    db_to_amp,
    finalize_master,
    gain_stage_for_master_bus,
    write_wav,
)

logger = logging.getLogger(__name__)
_EXPORT_TARGET_LUFS = -18.0
_EXPORT_TRUE_PEAK_CEILING_DBFS = -0.5


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


@dataclass(frozen=True)
class RenderWindow:
    """Requested snippet window and hidden render margins."""

    start_seconds: float
    duration_seconds: float
    pre_margin_seconds: float = 0.5
    post_margin_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.start_seconds < 0:
            raise ValueError("start_seconds must be non-negative")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.pre_margin_seconds < 0:
            raise ValueError("pre_margin_seconds must be non-negative")
        if self.post_margin_seconds < 0:
            raise ValueError("post_margin_seconds must be non-negative")

    @property
    def end_seconds(self) -> float:
        return self.start_seconds + self.duration_seconds

    @property
    def render_start_seconds(self) -> float:
        return max(0.0, self.start_seconds - self.pre_margin_seconds)

    @property
    def render_end_seconds(self) -> float:
        return self.end_seconds + self.post_margin_seconds

    @property
    def trim_start_seconds(self) -> float:
        return self.start_seconds - self.render_start_seconds


def list_pieces() -> list[str]:
    """Return the registered piece names."""
    return sorted(PIECES)


def export_piece_midi(
    piece_name: str,
    *,
    output_dir: str | Path = "output/midi",
    render_window: RenderWindow | None = None,
    stem_formats: tuple[MidiStemFormat, ...] = ALL_STEM_FORMATS,
) -> MidiBundleExportResult:
    """Export a registered score-backed piece as a MIDI bundle."""
    if piece_name not in PIECES:
        raise ValueError(f"Unknown piece: {piece_name}")

    definition = PIECES[piece_name]
    if definition.build_score is None:
        raise ValueError(
            f"Piece {piece_name} does not support MIDI export because it uses "
            "render_audio instead of build_score"
        )

    score = definition.build_score()

    bundle_dir = _build_output_path(
        output_dir=output_dir,
        output_name=definition.output_name,
        piece_name=piece_name,
        study=definition.study,
        render_window=render_window,
    ).with_suffix("")
    spec = MidiBundleExportSpec(
        piece_name=piece_name,
        output_name=bundle_dir.name,
        stem_formats=stem_formats,
    )
    return export_midi_bundle(
        score,
        bundle_dir,
        spec=spec,
        window_start_seconds=(
            render_window.start_seconds if render_window is not None else None
        ),
        window_end_seconds=(
            render_window.end_seconds if render_window is not None else None
        ),
    )


def render_piece(
    piece_name: str,
    *,
    output_dir: str | Path = "output",
    save_plot: bool = False,
    save_analysis: bool = True,
    render_window: RenderWindow | None = None,
) -> RenderResult:
    """Render a registered piece by name."""
    if piece_name not in PIECES:
        raise ValueError(f"Unknown piece: {piece_name}")

    definition = PIECES[piece_name]
    output_path = _build_output_path(
        output_dir=output_dir,
        output_name=definition.output_name,
        piece_name=piece_name,
        study=definition.study,
        render_window=render_window,
    )
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
    render_score: Score | None = None
    rendered_stems: dict[str, np.ndarray] | None = None
    effect_analysis: dict[str, Any] | None = None
    pre_master_mix: np.ndarray | None = None

    if definition.build_score is not None:
        score = definition.build_score()
        render_score = score
        if render_window is not None:
            render_score = score.extract_window(
                start_seconds=render_window.render_start_seconds,
                end_seconds=render_window.render_end_seconds,
            )
        audio, rendered_stems, effect_analysis = (
            render_score.render_with_effect_analysis()
        )
        if rendered_stems:
            _dry_stems, send_returns, _, _ = (
                render_score._render_mix_components_internal(
                    collect_effect_analysis=False
                )
            )
            pre_master_mix_inputs = [*rendered_stems.values(), *send_returns.values()]
            pre_master_mix = render_score._stack_signals(pre_master_mix_inputs)
            if render_score.auto_master_gain_stage:
                pre_master_mix = gain_stage_for_master_bus(
                    pre_master_mix,
                    sample_rate=render_score.sample_rate,
                    target_lufs=render_score.master_bus_target_lufs,
                    max_true_peak_dbfs=render_score.master_bus_max_true_peak_dbfs,
                )
            if render_score.master_input_gain_db != 0.0:
                pre_master_mix = pre_master_mix * db_to_amp(
                    render_score.master_input_gain_db
                )
        if render_window is not None:
            audio = _trim_rendered_audio(
                audio=audio,
                sample_rate=render_score.sample_rate,
                start_seconds=render_window.trim_start_seconds,
                duration_seconds=render_window.duration_seconds,
            )
        if save_plot:
            version_plot_path = version_output_path.with_suffix(".png")
            figure, _ = render_score.plot_piano_roll(version_plot_path)
            plt.close(figure)
    elif definition.render_audio is not None:
        if render_window is not None:
            raise ValueError(
                f"Piece {piece_name} does not support snippet rendering because it "
                "uses render_audio instead of build_score"
            )
        audio = definition.render_audio()
    else:
        raise ValueError(f"Piece {piece_name} has no render path configured")

    mastering_result = finalize_master(
        audio,
        sample_rate=render_score.sample_rate
        if render_score is not None
        else SAMPLE_RATE,
        target_lufs=_EXPORT_TARGET_LUFS,
        true_peak_ceiling_dbfs=_EXPORT_TRUE_PEAK_CEILING_DBFS,
    )
    export_audio = mastering_result.signal

    if save_analysis:
        version_analysis_artifacts = save_analysis_artifacts(
            output_prefix=version_output_path.with_suffix(""),
            mix_signal=export_audio,
            pre_master_mix_signal=pre_master_mix,
            pre_export_mix_signal=audio,
            sample_rate=render_score.sample_rate
            if render_score is not None
            else SAMPLE_RATE,
            stems=rendered_stems,
            effect_analysis=effect_analysis,
            score=render_score,
            piece_sections=definition.sections,
        )
        version_analysis_manifest_path = Path(
            str(version_analysis_artifacts["manifest_path"])
        )

    write_wav(version_output_path, export_audio)
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
        score=render_score,
        render_window=render_window,
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
    version_dir = output_path.parent / "versions"
    version_stem = f"{output_path.stem}__{version_timestamp}"
    return version_dir / f"{version_stem}{output_path.suffix}"


def _build_output_path(
    *,
    output_dir: str | Path,
    output_name: str,
    piece_name: str,
    study: bool = False,
    render_window: RenderWindow | None,
) -> Path:
    """Return the stable output path for the render request.

    Renders land in per-piece subdirectories:
      output/{piece_name}/{output_name}.wav          (regular pieces)
      output/studies/{piece_name}/{output_name}.wav   (studies)
    """
    if study:
        base_dir = Path(output_dir) / "studies" / piece_name
    else:
        base_dir = Path(output_dir) / piece_name
    base_output_path = base_dir / f"{output_name}.wav"
    if render_window is None:
        return base_output_path
    snippet_suffix = (
        "__snippet_"
        f"{_format_filename_seconds(render_window.start_seconds)}_"
        f"{_format_filename_seconds(render_window.duration_seconds)}"
    )
    return base_output_path.with_name(
        f"{base_output_path.stem}{snippet_suffix}{base_output_path.suffix}"
    )


def _format_filename_seconds(value_seconds: float) -> str:
    """Return a short filesystem-friendly second marker."""
    normalized = f"{value_seconds:.3f}".rstrip("0").rstrip(".")
    return normalized.replace(".", "p")


def _trim_rendered_audio(
    *,
    audio: np.ndarray,
    sample_rate: int,
    start_seconds: float,
    duration_seconds: float,
) -> np.ndarray:
    """Trim rendered audio to the exact requested snippet bounds."""
    start_sample = max(0, int(round(start_seconds * sample_rate)))
    duration_samples = max(0, int(round(duration_seconds * sample_rate)))
    end_sample = start_sample + duration_samples
    if audio.ndim == 1:
        trimmed = audio[start_sample:end_sample]
    else:
        trimmed = audio[:, start_sample:end_sample]
    if trimmed.shape[-1] >= duration_samples:
        return trimmed

    pad_width = duration_samples - trimmed.shape[-1]
    if audio.ndim == 1:
        return np.pad(trimmed, (0, pad_width))
    return np.pad(trimmed, ((0, 0), (0, pad_width)))


def _copy_analysis_artifacts_to_latest(
    *,
    version_analysis_artifacts: dict[str, Any],
    version_prefix: Path,
    latest_prefix: Path,
) -> dict[str, Any]:
    """Mirror versioned analysis files into the stable latest filenames."""
    latest_analysis_artifacts = _rewrite_prefixed_paths(
        payload=version_analysis_artifacts,
        source_prefix=version_prefix,
        target_prefix=latest_prefix,
    )
    for source_path, target_path in _collect_rewritten_paths(
        source_payload=version_analysis_artifacts,
        rewritten_payload=latest_analysis_artifacts,
    ):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
    manifest_path = Path(str(latest_analysis_artifacts["manifest_path"]))
    manifest_path.write_text(
        json.dumps(latest_analysis_artifacts, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return latest_analysis_artifacts


def _rewrite_prefixed_paths(
    *,
    payload: Any,
    source_prefix: Path,
    target_prefix: Path,
) -> Any:
    """Replace a versioned artifact prefix with the stable output prefix."""
    if isinstance(payload, dict):
        return {
            key: _rewrite_prefixed_paths(
                payload=value,
                source_prefix=source_prefix,
                target_prefix=target_prefix,
            )
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [
            _rewrite_prefixed_paths(
                payload=value,
                source_prefix=source_prefix,
                target_prefix=target_prefix,
            )
            for value in payload
        ]
    if isinstance(payload, str):
        normalized_payload = payload.replace("\\", "/")
        normalized_source = str(source_prefix).replace("\\", "/")
        if normalized_payload.startswith(normalized_source):
            return str(target_prefix) + payload[len(str(source_prefix)) :]
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
    render_window: RenderWindow | None,
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
            "export_target_lufs": _EXPORT_TARGET_LUFS,
            "export_true_peak_ceiling_dbfs": _EXPORT_TRUE_PEAK_CEILING_DBFS,
            "render_window": _serialize_render_window(render_window),
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


def _serialize_render_window(
    render_window: RenderWindow | None,
) -> dict[str, Any] | None:
    """Convert a render window into JSON-friendly metadata."""
    if render_window is None:
        return None
    return {
        "mode": "snippet",
        "start_seconds": render_window.start_seconds,
        "duration_seconds": render_window.duration_seconds,
        "end_seconds": render_window.end_seconds,
        "pre_margin_seconds": render_window.pre_margin_seconds,
        "post_margin_seconds": render_window.post_margin_seconds,
        "render_start_seconds": render_window.render_start_seconds,
        "render_end_seconds": render_window.render_end_seconds,
    }


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
