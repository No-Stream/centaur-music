"""Tuning analysis and tuning-file writers for MIDI export."""

from __future__ import annotations

import logging
import math
from pathlib import Path

from code_musics.midi_export_types import (
    CHROMATIC_SLOT_CENTS,
    CHROMATIC_SLOT_NAMES,
    TUNING_WARNING_SUFFIX,
    ChromaticSlotAssignment,
    ChromaticTuningResult,
    MidiBundleExportSpec,
    ResolvedTuningMode,
    TuningAnalysisResult,
    TuningMode,
)
from code_musics.score import Score

logger: logging.Logger = logging.getLogger(__name__)

_NUM_CHROMATIC_SLOTS = 12


def analyze_tuning(
    *,
    score: Score,
    stem_notes: dict[str, list],
    spec: MidiBundleExportSpec,
) -> TuningAnalysisResult:
    reference_frequency_hz = (
        score.f0_hz
        if spec.reference_frequency_hz is None
        else spec.reference_frequency_hz
    )
    if reference_frequency_hz <= 0:
        raise ValueError("reference_frequency_hz must be positive")
    period_cents = 1200.0 * math.log2(spec.period_ratio)
    pitch_class_values = _collect_pitch_classes(
        stem_notes=stem_notes,
        reference_frequency_hz=reference_frequency_hz,
        period_cents=period_cents,
        quantization_cents=spec.quantization_cents,
    )
    resolved_mode = _resolve_tuning_mode(
        requested_mode=spec.tuning_mode,
        pitch_class_count=len(pitch_class_values),
        max_static_scale_size=spec.static_scale_max_size,
    )
    pitch_class_cents = tuple(_ensure_unison_pitch_class(pitch_class_values))
    scale_entry_cents = tuple(
        [value for value in pitch_class_cents if value > 0.0] + [period_cents]
    )
    return TuningAnalysisResult(
        tuning_mode=resolved_mode,
        is_approximate=resolved_mode == "exact_note_tuning",
        period_ratio=spec.period_ratio,
        period_cents=period_cents,
        reference_midi_note=spec.reference_midi_note,
        reference_frequency_hz=reference_frequency_hz,
        pitch_class_cents=pitch_class_cents,
        scale_entry_cents=scale_entry_cents,
        quantization_cents=spec.quantization_cents,
        warning_suffix=(
            TUNING_WARNING_SUFFIX if resolved_mode == "exact_note_tuning" else None
        ),
    )


def _collect_pitch_classes(
    *,
    stem_notes: dict[str, list],
    reference_frequency_hz: float,
    period_cents: float,
    quantization_cents: float,
) -> list[float]:
    quantized_pitch_classes: set[float] = set()
    for voice_notes in stem_notes.values():
        for note in voice_notes:
            cents = 1200.0 * math.log2(note.freq_hz / reference_frequency_hz)
            pitch_class = cents % period_cents
            if math.isclose(pitch_class, period_cents, abs_tol=1e-9):
                pitch_class = 0.0
            quantized_pitch_classes.add(
                round(pitch_class / quantization_cents) * quantization_cents
            )
    return sorted(quantized_pitch_classes)


def _ensure_unison_pitch_class(pitch_class_values: list[float]) -> list[float]:
    if any(math.isclose(value, 0.0, abs_tol=1e-9) for value in pitch_class_values):
        return sorted(pitch_class_values)
    return [0.0, *sorted(pitch_class_values)]


def _resolve_tuning_mode(
    *,
    requested_mode: TuningMode,
    pitch_class_count: int,
    max_static_scale_size: int,
) -> ResolvedTuningMode:
    if requested_mode == "static_periodic_tuning":
        return "static_periodic_tuning"
    if requested_mode == "exact_note_tuning":
        return "exact_note_tuning"
    if pitch_class_count <= max_static_scale_size:
        return "static_periodic_tuning"
    return "exact_note_tuning"


def write_tuning_files(
    *,
    tuning_dir: Path,
    tuning_analysis: TuningAnalysisResult,
    spec: MidiBundleExportSpec,
) -> dict[str, str]:
    suffix = (
        f"_{tuning_analysis.warning_suffix}" if tuning_analysis.warning_suffix else ""
    )
    base_name = f"{spec.output_name}{suffix}"
    scl_path = tuning_dir / f"{base_name}.scl"
    kbm_path = tuning_dir / f"{base_name}.kbm"
    tun_path = tuning_dir / f"{base_name}.tun"

    scl_path.write_text(build_scl_text(tuning_analysis, spec), encoding="utf-8")
    kbm_path.write_text(build_kbm_text(tuning_analysis), encoding="utf-8")
    tun_path.write_text(build_tun_text(tuning_analysis, spec), encoding="utf-8")
    return {
        "scl": str(scl_path),
        "kbm": str(kbm_path),
        "tun": str(tun_path),
    }


def build_scl_text(
    tuning_analysis: TuningAnalysisResult,
    spec: MidiBundleExportSpec,
) -> str:
    lines = [
        f"! {spec.output_name}.scl",
        f"{spec.output_name} {'approximate' if tuning_analysis.is_approximate else 'exact'}",
        str(len(tuning_analysis.scale_entry_cents)),
        "!",
    ]
    lines.extend(f"{value:.6f}" for value in tuning_analysis.scale_entry_cents)
    return "\n".join(lines) + "\n"


def build_kbm_text(tuning_analysis: TuningAnalysisResult) -> str:
    mapping_size = len(tuning_analysis.pitch_class_cents)
    lines = [
        "! generated keyboard mapping",
        str(mapping_size),
        "!",
        "0",
        "127",
        str(tuning_analysis.reference_midi_note),
        str(tuning_analysis.reference_midi_note),
        f"{tuning_analysis.reference_frequency_hz:.12f}",
        str(mapping_size),
    ]
    lines.extend(str(index) for index in range(mapping_size))
    return "\n".join(lines) + "\n"


def build_tun_text(
    tuning_analysis: TuningAnalysisResult,
    spec: MidiBundleExportSpec,
) -> str:
    base_freq = mapped_frequency_for_midi_note(
        midi_note=0,
        tuning_analysis=tuning_analysis,
    )
    description = (
        f"{spec.output_name} "
        f"{'approximate' if tuning_analysis.is_approximate else 'exact'} tuning"
    )
    lines = [
        "; generated by centaur-music",
        "[Scale Begin]",
        "[Info]",
        f'Name="{spec.output_name}"',
        f'Description="{description}"',
        "[Exact Tuning]",
        f"BaseFreq={base_freq:.12f}",
    ]
    for midi_note in range(128):
        freq_hz = mapped_frequency_for_midi_note(
            midi_note=midi_note,
            tuning_analysis=tuning_analysis,
        )
        cents = 1200.0 * math.log2(freq_hz / base_freq)
        lines.append(f"note {midi_note}={cents:.12f}")
    lines.append("[Scale End]")
    return "\n".join(lines) + "\n"


def mapped_frequency_for_midi_note(
    *,
    midi_note: int,
    tuning_analysis: TuningAnalysisResult,
) -> float:
    scale_size = len(tuning_analysis.pitch_class_cents)
    degree_offset = midi_note - tuning_analysis.reference_midi_note
    period_index = math.floor(degree_offset / scale_size)
    degree_index = degree_offset % scale_size
    pitch_class_ratio = 2.0 ** (
        tuning_analysis.pitch_class_cents[degree_index] / 1200.0
    )
    return (
        tuning_analysis.reference_frequency_hz
        * (tuning_analysis.period_ratio**period_index)
        * pitch_class_ratio
    )


# ---------------------------------------------------------------------------
# Chromatic-fill SCL for MTS-ESP / Ableton interactive composition
# ---------------------------------------------------------------------------


def assign_chromatic_slots(
    pitch_class_cents: tuple[float, ...],
) -> list[ChromaticSlotAssignment]:
    """Greedy-by-distance assignment of pitch classes to 12-TET chromatic slots.

    Each pitch class is mapped to the nearest available chromatic slot.
    Pairs are sorted by absolute cent distance; closest wins first.
    """
    available_slots: set[int] = set(range(_NUM_CHROMATIC_SLOTS))
    remaining_cents: dict[int, float] = {}
    for index, cents in enumerate(pitch_class_cents):
        if not math.isclose(cents, 0.0, abs_tol=1e-9):
            remaining_cents[index] = cents
        # 0.0 (unison) always maps to slot 0

    assignments: dict[int, ChromaticSlotAssignment] = {}

    # Unison → slot 0
    assignments[0] = ChromaticSlotAssignment(
        slot=0,
        slot_name=CHROMATIC_SLOT_NAMES[0],
        cents=0.0,
        source="scale",
        error_cents=0.0,
    )
    available_slots.discard(0)

    # Build candidate pairs: (abs_error, pitch_class_index, slot)
    candidates: list[tuple[float, int, int]] = []
    for pc_index, cents in remaining_cents.items():
        for slot in range(_NUM_CHROMATIC_SLOTS):
            slot_cents = CHROMATIC_SLOT_CENTS[slot]
            error = abs(cents - slot_cents)
            candidates.append((error, pc_index, slot))

    candidates.sort()

    assigned_pcs: set[int] = set()
    for error, pc_index, slot in candidates:
        if pc_index in assigned_pcs or slot not in available_slots:
            continue
        assignments[slot] = ChromaticSlotAssignment(
            slot=slot,
            slot_name=CHROMATIC_SLOT_NAMES[slot],
            cents=pitch_class_cents[pc_index],
            source="scale",
            error_cents=error,
        )
        available_slots.discard(slot)
        assigned_pcs.add(pc_index)
        if not available_slots or len(assigned_pcs) == len(remaining_cents):
            break

    return [assignments[slot] for slot in sorted(assignments)]


def _nearest_neighbor_fill(
    assignments: list[ChromaticSlotAssignment],
) -> list[ChromaticSlotAssignment]:
    """Fill unoccupied chromatic slots by copying the nearest occupied slot's cents.

    Ties go to the lower slot.
    """
    occupied: dict[int, ChromaticSlotAssignment] = {a.slot: a for a in assignments}
    result: list[ChromaticSlotAssignment] = []

    for slot in range(_NUM_CHROMATIC_SLOTS):
        if slot in occupied:
            result.append(occupied[slot])
            continue

        # Find nearest occupied slot; ties favor lower
        best_distance = _NUM_CHROMATIC_SLOTS
        best_slot = -1
        for occ_slot in occupied:
            distance = abs(slot - occ_slot)
            if distance < best_distance or (
                distance == best_distance and occ_slot < best_slot
            ):
                best_distance = distance
                best_slot = occ_slot

        donor = occupied[best_slot]
        result.append(
            ChromaticSlotAssignment(
                slot=slot,
                slot_name=CHROMATIC_SLOT_NAMES[slot],
                cents=donor.cents,
                source=f"fill_from_{donor.slot_name}",
                error_cents=0.0,
            )
        )

    return result


def build_chromatic_tuning(
    *,
    tuning_dir: Path,
    tuning_analysis: TuningAnalysisResult,
    spec: MidiBundleExportSpec,
) -> ChromaticTuningResult:
    """Build a 12-note chromatic-fill SCL for MTS-ESP / Ableton composition."""
    if not spec.chromatic_scl:
        return ChromaticTuningResult(
            slot_assignments=(),
            warnings=[],
            skipped_reason="chromatic_scl disabled in export spec",
        )

    if not math.isclose(tuning_analysis.period_ratio, 2.0, rel_tol=1e-9):
        reason = (
            f"non-octave period ratio ({tuning_analysis.period_ratio:.4f}) "
            "does not map to chromatic layout"
        )
        logger.info("Skipping chromatic SCL: %s", reason)
        return ChromaticTuningResult(
            slot_assignments=(),
            warnings=[],
            skipped_reason=reason,
        )

    n_pitch_classes = len(tuning_analysis.pitch_class_cents)
    if n_pitch_classes > _NUM_CHROMATIC_SLOTS:
        reason = f"{n_pitch_classes} pitch classes exceeds 12 chromatic slots"
        logger.info("Skipping chromatic SCL: %s", reason)
        return ChromaticTuningResult(
            slot_assignments=(),
            warnings=[],
            skipped_reason=reason,
        )

    raw_assignments = assign_chromatic_slots(tuning_analysis.pitch_class_cents)
    filled_assignments = _nearest_neighbor_fill(raw_assignments)

    warnings: list[str] = []
    threshold = spec.chromatic_warning_threshold_cents
    for assignment in filled_assignments:
        if assignment.source == "scale" and assignment.error_cents > threshold:
            warnings.append(
                f"{assignment.slot_name}: {assignment.cents:.1f}c mapped to "
                f"{CHROMATIC_SLOT_CENTS[assignment.slot]:.0f}c slot "
                f"(error {assignment.error_cents:.1f}c > {threshold:.0f}c threshold)"
            )

    for warning in warnings:
        logger.warning("Chromatic SCL: %s", warning)

    chromatic_name = f"{spec.output_name}_chromatic"
    scl_path = tuning_dir / f"{chromatic_name}.scl"
    kbm_path = tuning_dir / f"{chromatic_name}.kbm"

    scl_path.write_text(
        build_chromatic_scl_text(filled_assignments, spec), encoding="utf-8"
    )
    kbm_path.write_text(build_chromatic_kbm_text(tuning_analysis), encoding="utf-8")

    return ChromaticTuningResult(
        slot_assignments=tuple(filled_assignments),
        warnings=warnings,
        scl_path=str(scl_path),
        kbm_path=str(kbm_path),
    )


def build_chromatic_scl_text(
    assignments: list[ChromaticSlotAssignment],
    spec: MidiBundleExportSpec,
) -> str:
    """Build a 12-note Scala file from chromatic slot assignments."""
    name = f"{spec.output_name}_chromatic"
    lines = [
        f"! {name}.scl",
        f"{name} chromatic-fill for MTS-ESP",
        str(_NUM_CHROMATIC_SLOTS),
        "!",
    ]
    # SCL entries are intervals above 1/1 — skip slot 0 (unison), end with octave
    for assignment in assignments[1:]:
        lines.append(f"{assignment.cents:.6f}")
    lines.append("1200.000000")
    return "\n".join(lines) + "\n"


def build_chromatic_kbm_text(
    tuning_analysis: TuningAnalysisResult,
) -> str:
    """Build a standard 12-note keyboard mapping for the chromatic SCL."""
    lines = [
        "! chromatic-fill keyboard mapping",
        str(_NUM_CHROMATIC_SLOTS),
        "!",
        "0",
        "127",
        str(tuning_analysis.reference_midi_note),
        str(tuning_analysis.reference_midi_note),
        f"{tuning_analysis.reference_frequency_hz:.12f}",
        str(_NUM_CHROMATIC_SLOTS),
    ]
    lines.extend(str(index) for index in range(_NUM_CHROMATIC_SLOTS))
    return "\n".join(lines) + "\n"
