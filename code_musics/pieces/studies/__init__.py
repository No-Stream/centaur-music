"""Study pieces exploring generative composition tools."""

from code_musics.pieces.registry import merge_piece_maps
from code_musics.pieces.studies.study_cloud import PIECES as _CLOUD_PIECES
from code_musics.pieces.studies.study_euclidean import PIECES as _EUCLIDEAN_PIECES
from code_musics.pieces.studies.study_harpsichord import PIECES as _HARPSICHORD_PIECES
from code_musics.pieces.studies.study_lattice import PIECES as _LATTICE_PIECES
from code_musics.pieces.studies.study_markov import PIECES as _MARKOV_PIECES
from code_musics.pieces.studies.study_prob_gate import PIECES as _PROB_GATE_PIECES
from code_musics.pieces.studies.study_surge_xt import PIECES as _SURGE_XT_PIECES
from code_musics.pieces.studies.study_surge_xt_clean import (
    PIECES as _SURGE_XT_CLEAN_PIECES,
)
from code_musics.pieces.studies.study_tone_pool import PIECES as _TONE_POOL_PIECES
from code_musics.pieces.studies.study_turing import PIECES as _TURING_PIECES

PIECES = merge_piece_maps(
    _CLOUD_PIECES,
    _EUCLIDEAN_PIECES,
    _HARPSICHORD_PIECES,
    _LATTICE_PIECES,
    _MARKOV_PIECES,
    _PROB_GATE_PIECES,
    _SURGE_XT_PIECES,
    _SURGE_XT_CLEAN_PIECES,
    _TONE_POOL_PIECES,
    _TURING_PIECES,
)

__all__ = ["PIECES"]
