"""Shared types and constants for MIDI export."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

DEFAULT_TICKS_PER_BEAT = 960
DEFAULT_EXPORT_BPM = 60.0
DEFAULT_REFERENCE_MIDI_NOTE = 60
DEFAULT_PERIOD_RATIO = 2.0
DEFAULT_STATIC_SCALE_MAX_SIZE = 24
PITCH_CLASS_QUANTIZATION_CENTS = 0.01
MPE_MEMBER_CHANNELS = tuple(range(1, 16))
MPE_GLOBAL_CHANNEL = 0
POLY_BEND_CHANNELS = tuple(range(0, 15))
MONO_BEND_CHANNEL = 0
TUNING_WARNING_SUFFIX = "WARNING_APPROX"

MidiStemFormat = Literal[
    "scala",
    "tun",
    "mpe_48st",
    "poly_bend_12st",
    "mono_bend_12st",
]
TuningMode = Literal["auto", "static_periodic_tuning", "exact_note_tuning"]
ResolvedTuningMode = Literal["static_periodic_tuning", "exact_note_tuning"]

ALL_STEM_FORMATS: tuple[MidiStemFormat, ...] = (
    "scala",
    "tun",
    "mpe_48st",
    "poly_bend_12st",
    "mono_bend_12st",
)


@dataclass(frozen=True)
class MidiStemNote:
    voice_name: str
    note_index: int
    start_seconds: float
    duration_seconds: float
    end_seconds: float
    freq_hz: float
    velocity: int
    label: str | None


@dataclass(frozen=True)
class TuningAnalysisResult:
    tuning_mode: ResolvedTuningMode
    is_approximate: bool
    period_ratio: float
    period_cents: float
    reference_midi_note: int
    reference_frequency_hz: float
    pitch_class_cents: tuple[float, ...]
    scale_entry_cents: tuple[float, ...]
    quantization_cents: float
    warning_suffix: str | None = None


@dataclass(frozen=True)
class MidiStemExportResult:
    voice_name: str
    note_count: int
    emitted_files: dict[str, str] = field(default_factory=dict)
    max_simultaneous_notes: int = 0


@dataclass(frozen=True)
class MidiBundleManifest:
    schema_version: int
    piece_name: str
    output_name: str
    tuning_mode: ResolvedTuningMode
    shared_tuning_status: Literal["exact", "approximate"]
    warning_suffix: str | None
    requested_stem_formats: list[MidiStemFormat]
    timing_encoding: dict[str, float | int]
    tuning: dict[str, Any]
    stem_exports: list[MidiStemExportResult]
    tuning_files: dict[str, str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stem_exports"] = [asdict(result) for result in self.stem_exports]
        return payload


@dataclass(frozen=True)
class MidiBundleExportResult:
    bundle_dir: Path
    manifest_path: Path
    readme_path: Path
    tuning_dir: Path
    stems_dir: Path
    manifest: MidiBundleManifest


def _normalize_stem_formats(
    stem_formats: tuple[str, ...],
) -> tuple[MidiStemFormat, ...]:
    if not stem_formats:
        raise ValueError("stem_formats must not be empty")
    normalized_formats: list[MidiStemFormat] = []
    seen_formats: set[MidiStemFormat] = set()
    for stem_format in stem_formats:
        if stem_format not in ALL_STEM_FORMATS:
            raise ValueError(f"Unsupported MIDI stem format: {stem_format!r}")
        typed_format = cast(MidiStemFormat, stem_format)
        if typed_format in seen_formats:
            continue
        normalized_formats.append(typed_format)
        seen_formats.add(typed_format)
    return tuple(normalized_formats)


@dataclass(frozen=True)
class MidiBundleExportSpec:
    piece_name: str
    output_name: str
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT
    export_bpm: float = DEFAULT_EXPORT_BPM
    tuning_mode: TuningMode = "auto"
    reference_midi_note: int = DEFAULT_REFERENCE_MIDI_NOTE
    reference_frequency_hz: float | None = None
    period_ratio: float = DEFAULT_PERIOD_RATIO
    static_scale_max_size: int = DEFAULT_STATIC_SCALE_MAX_SIZE
    quantization_cents: float = PITCH_CLASS_QUANTIZATION_CENTS
    stem_formats: tuple[MidiStemFormat, ...] = ALL_STEM_FORMATS

    def __post_init__(self) -> None:
        if self.ticks_per_beat <= 0:
            raise ValueError("ticks_per_beat must be positive")
        if self.export_bpm <= 0:
            raise ValueError("export_bpm must be positive")
        if not 0 <= self.reference_midi_note <= 127:
            raise ValueError("reference_midi_note must be in the range [0, 127]")
        if self.reference_frequency_hz is not None and self.reference_frequency_hz <= 0:
            raise ValueError("reference_frequency_hz must be positive when provided")
        if self.period_ratio <= 1.0:
            raise ValueError("period_ratio must be greater than 1.0")
        if self.static_scale_max_size < 1:
            raise ValueError("static_scale_max_size must be >= 1")
        if self.quantization_cents <= 0:
            raise ValueError("quantization_cents must be positive")
        object.__setattr__(
            self,
            "stem_formats",
            _normalize_stem_formats(tuple(self.stem_formats)),
        )
