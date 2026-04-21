"""Types for audio stem WAV export."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class StemExportSpec:
    """Configuration for a stem export bundle."""

    piece_name: str
    output_name: str
    bit_depth: int = 24
    include_mix: bool = True
    dry: bool = False

    def __post_init__(self) -> None:
        if self.bit_depth not in (16, 24, 32):
            raise ValueError(f"bit_depth must be 16, 24, or 32, got {self.bit_depth}")


@dataclass(frozen=True)
class StemFileInfo:
    """Metadata for a single exported stem WAV file."""

    name: str
    kind: Literal["voice", "send"]
    path: str  # relative to bundle dir
    channels: int
    sample_count: int
    peak_dbfs: float


@dataclass(frozen=True)
class StemBundleManifest:
    """Manifest describing an exported stem bundle."""

    schema_version: int
    piece_name: str
    output_name: str
    sample_rate: int
    bit_depth: int
    dry: bool
    total_duration_seconds: float
    total_samples: int
    voice_stems: list[StemFileInfo]
    send_stems: list[StemFileInfo]
    mix_path: str | None
    processing_notes: str
    stem_gain_db: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["voice_stems"] = [asdict(s) for s in self.voice_stems]
        payload["send_stems"] = [asdict(s) for s in self.send_stems]
        return payload


@dataclass(frozen=True)
class StemBundleResult:
    """Result of exporting a stem bundle."""

    bundle_dir: Path
    manifest_path: Path
    manifest: StemBundleManifest
