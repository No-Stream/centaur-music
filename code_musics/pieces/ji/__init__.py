"""Just-intonation study pieces."""

from code_musics.pieces.ji.chorale import (
    PIECES as _CHORALE_PIECES,
)
from code_musics.pieces.ji.chorale import (
    build_ji_chorale_score,
)
from code_musics.pieces.ji.chorale_v2 import (
    PIECES as _CHORALE_V2_PIECES,
)
from code_musics.pieces.ji.chorale_v2 import (
    build_ji_chorale_v2_score,
)
from code_musics.pieces.ji.comma_drift import (
    PIECES as _COMMA_DRIFT_PIECES,
)
from code_musics.pieces.ji.comma_drift import (
    build_ji_comma_drift_score,
)
from code_musics.pieces.ji.melody import (
    PIECES as _MELODY_PIECES,
)
from code_musics.pieces.ji.melody import (
    build_ji_melody_score,
)
from code_musics.pieces.registry import merge_piece_maps

PIECES = merge_piece_maps(
    _CHORALE_PIECES,
    _CHORALE_V2_PIECES,
    _MELODY_PIECES,
    _COMMA_DRIFT_PIECES,
)

__all__ = [
    "PIECES",
    "build_ji_chorale_score",
    "build_ji_chorale_v2_score",
    "build_ji_comma_drift_score",
    "build_ji_melody_score",
]
