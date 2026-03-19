"""Compatibility re-exports for former `ji_studies.py` builders."""

from code_musics.pieces.ji import (
    PIECES,
    build_ji_chorale_score,
    build_ji_comma_drift_score,
    build_ji_melody_score,
)

__all__ = [
    "PIECES",
    "build_ji_chorale_score",
    "build_ji_comma_drift_score",
    "build_ji_melody_score",
]
