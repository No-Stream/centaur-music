"""Shared piece registry helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from code_musics.score import Score


@dataclass(frozen=True)
class PieceSection:
    """Named section boundary metadata for a piece timeline."""

    label: str
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if self.start_seconds < 0:
            raise ValueError("section start_seconds must be non-negative")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("section end_seconds must be greater than start_seconds")


@dataclass(frozen=True)
class PieceDefinition:
    """Named renderable piece.

    ``export_target_lufs`` overrides the export-stage LUFS target for
    pieces whose natural integrated loudness falls well below the
    project-wide default (-18 LUFS).  Quiet/sparse/dynamic pieces
    (solo piano meditations, ambient textures with long silences)
    otherwise force the final mastering stage into heavy limiting as
    it tries to pull their LUFS up.  Leave as ``None`` for the default.
    """

    name: str
    output_name: str
    build_score: Callable[[], Score] | None = None
    render_audio: Callable[[], np.ndarray] | None = None
    sections: tuple[PieceSection, ...] = field(default_factory=tuple)
    study: bool = False
    export_target_lufs: float | None = None
    build_viz_annotations: Callable[[], dict[str, Any]] | None = None


type PieceMap = Mapping[str, PieceDefinition]


def merge_piece_maps(*piece_maps: PieceMap) -> dict[str, PieceDefinition]:
    """Merge multiple piece maps and fail fast on duplicate names."""
    merged: dict[str, PieceDefinition] = {}
    for piece_map in piece_maps:
        for piece_name, definition in piece_map.items():
            if piece_name in merged:
                raise ValueError(f"Duplicate piece registered: {piece_name}")
            merged[piece_name] = definition
    return merged
