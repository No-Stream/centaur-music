"""Piece registry exports."""

from code_musics.pieces.coltrane_studies import PIECES as _COLTRANE_PIECES
from code_musics.pieces.effects_showcase import PIECES as _EFFECTS_SHOWCASE_PIECES
from code_musics.pieces.ji_studies import PIECES as _JI_PIECES
from code_musics.pieces.septimal import PIECES as _SEPTIMAL_PIECES
from code_musics.pieces.sketches import PIECES as _SKETCH_PIECES

PIECES = {
    **_SEPTIMAL_PIECES,
    **_SKETCH_PIECES,
    **_JI_PIECES,
    **_COLTRANE_PIECES,
    **_EFFECTS_SHOWCASE_PIECES,
}

__all__ = ["PIECES"]
