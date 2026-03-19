"""Shared piece registry helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

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
    """Named renderable piece."""

    name: str
    output_name: str
    build_score: Callable[[], Score] | None = None
    render_audio: Callable[[], np.ndarray] | None = None
    sections: tuple[PieceSection, ...] = field(default_factory=tuple)


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
