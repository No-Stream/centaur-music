"""Piece registry exports."""

from code_musics.pieces.ji_studies import PIECES as _JI_PIECES
from code_musics.pieces.septimal import PIECES as _SEPTIMAL_PIECES
from code_musics.pieces.sketches import PIECES as _SKETCH_PIECES

PIECES = {**_SEPTIMAL_PIECES, **_SKETCH_PIECES, **_JI_PIECES}

__all__ = ["PIECES"]
