"""Standard MIDI and tuning-file export for score-backed pieces."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from code_musics.midi_export_stems import collect_stem_notes, write_stem_files
from code_musics.midi_export_tuning import (
    analyze_tuning,
    build_chromatic_tuning,
    write_tuning_files,
)
from code_musics.midi_export_types import (
    ALL_STEM_FORMATS,
    DEFAULT_EXPORT_BPM,
    DEFAULT_PERIOD_RATIO,
    DEFAULT_REFERENCE_MIDI_NOTE,
    DEFAULT_STATIC_SCALE_MAX_SIZE,
    DEFAULT_TICKS_PER_BEAT,
    PITCH_CLASS_QUANTIZATION_CENTS,
    ChromaticTuningResult,
    MidiBundleExportResult,
    MidiBundleExportSpec,
    MidiBundleManifest,
    MidiStemExportResult,
    MidiStemFormat,
    MidiStemNote,
    ResolvedTuningMode,
    TuningAnalysisResult,
    TuningMode,
)
from code_musics.score import Score

logger: logging.Logger = logging.getLogger(__name__)


def export_midi_bundle(
    score: Score,
    out_dir: str | Path,
    *,
    spec: MidiBundleExportSpec,
    window_start_seconds: float | None = None,
    window_end_seconds: float | None = None,
) -> MidiBundleExportResult:
    bundle_dir = Path(out_dir)
    tuning_dir = bundle_dir / "tuning"
    stems_dir = bundle_dir / "stems"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    stems_dir.mkdir(parents=True, exist_ok=True)

    stem_notes = collect_stem_notes(
        score,
        window_start_seconds=window_start_seconds,
        window_end_seconds=window_end_seconds,
    )
    tuning_analysis = analyze_tuning(score=score, stem_notes=stem_notes, spec=spec)
    tuning_files = write_tuning_files(
        tuning_dir=tuning_dir,
        tuning_analysis=tuning_analysis,
        spec=spec,
    )
    chromatic_result = build_chromatic_tuning(
        tuning_dir=tuning_dir,
        tuning_analysis=tuning_analysis,
        spec=spec,
    )
    chromatic_tuning_dict = _chromatic_result_to_dict(chromatic_result)
    if chromatic_result.scl_path:
        tuning_files["chromatic_scl"] = chromatic_result.scl_path
    if chromatic_result.kbm_path:
        tuning_files["chromatic_kbm"] = chromatic_result.kbm_path

    stem_results = write_stem_files(
        stems_dir=stems_dir,
        stem_notes=stem_notes,
        tuning_analysis=tuning_analysis,
        spec=spec,
    )

    base_warnings: list[str] = []
    if tuning_analysis.is_approximate:
        base_warnings.append("Shared tuning files are approximate convenience exports.")
    base_warnings.extend(chromatic_result.warnings)

    manifest = MidiBundleManifest(
        schema_version=1,
        piece_name=spec.piece_name,
        output_name=spec.output_name,
        tuning_mode=tuning_analysis.tuning_mode,
        shared_tuning_status=(
            "approximate" if tuning_analysis.is_approximate else "exact"
        ),
        warning_suffix=tuning_analysis.warning_suffix,
        requested_stem_formats=list(spec.stem_formats),
        timing_encoding={
            "ticks_per_beat": spec.ticks_per_beat,
            "export_bpm": spec.export_bpm,
            "seconds_per_beat": 60.0 / spec.export_bpm,
        },
        tuning={
            "period_ratio": tuning_analysis.period_ratio,
            "period_cents": tuning_analysis.period_cents,
            "reference_midi_note": tuning_analysis.reference_midi_note,
            "reference_frequency_hz": tuning_analysis.reference_frequency_hz,
            "pitch_class_cents": list(tuning_analysis.pitch_class_cents),
            "scale_entry_cents": list(tuning_analysis.scale_entry_cents),
            "quantization_cents": tuning_analysis.quantization_cents,
        },
        stem_exports=stem_results,
        tuning_files=tuning_files,
        chromatic_tuning=chromatic_tuning_dict,
        warnings=base_warnings,
    )
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    readme_path = bundle_dir / "README.md"
    readme_path.write_text(
        _build_bundle_readme(tuning_analysis, chromatic_result), encoding="utf-8"
    )
    logger.info("Exported MIDI bundle for %s to %s", spec.piece_name, bundle_dir)
    return MidiBundleExportResult(
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        readme_path=readme_path,
        tuning_dir=tuning_dir,
        stems_dir=stems_dir,
        manifest=manifest,
    )


def _chromatic_result_to_dict(
    result: ChromaticTuningResult,
) -> dict[str, object] | None:
    if result.skipped_reason:
        return {"skipped_reason": result.skipped_reason}
    return {
        "scl": result.scl_path,
        "kbm": result.kbm_path,
        "slot_assignments": [asdict(a) for a in result.slot_assignments],
        "warnings": result.warnings,
    }


def _build_bundle_readme(
    tuning_analysis: TuningAnalysisResult,
    chromatic_result: ChromaticTuningResult,
) -> str:
    warning_line = ""
    if tuning_analysis.is_approximate:
        warning_line = (
            "\nApproximate shared tuning files are suffixed with WARNING_APPROX.\n"
        )
    chromatic_line = ""
    if chromatic_result.scl_path:
        chromatic_line = (
            "\nChromatic-fill SCL (*_chromatic.scl + *_chromatic.kbm) maps this\n"
            "tuning onto a standard 12-note piano layout for interactive composition\n"
            "in MTS-ESP / Ableton. Empty slots are filled from the nearest scale tone.\n"
        )
    elif chromatic_result.skipped_reason:
        chromatic_line = f"\nChromatic-fill SCL was not generated: {chromatic_result.skipped_reason}\n"
    return (
        "MIDI bundle\n\n"
        "Score seconds are encoded as MIDI at 60 BPM, so 1 beat equals 1 second.\n"
        + warning_line
        + "Requested stem formats fail fast if they cannot be emitted correctly.\n"
        + "Load tuning/*.scl with tuning/*.kbm for *_scala.mid.\n"
        + "Load tuning/*.tun for *_tun.mid.\n"
        + "Use 48 semitone bend for *_mpe_48st.mid.\n"
        + "Use 12 semitone bend for *_poly_bend_12st.mid and *_mono_bend_12st.mid.\n"
        + chromatic_line
    )


__all__ = [
    "ALL_STEM_FORMATS",
    "DEFAULT_EXPORT_BPM",
    "DEFAULT_PERIOD_RATIO",
    "DEFAULT_REFERENCE_MIDI_NOTE",
    "DEFAULT_STATIC_SCALE_MAX_SIZE",
    "DEFAULT_TICKS_PER_BEAT",
    "MidiBundleExportResult",
    "MidiBundleExportSpec",
    "MidiBundleManifest",
    "MidiStemExportResult",
    "MidiStemFormat",
    "MidiStemNote",
    "PITCH_CLASS_QUANTIZATION_CENTS",
    "ResolvedTuningMode",
    "TuningAnalysisResult",
    "TuningMode",
    "export_midi_bundle",
]
