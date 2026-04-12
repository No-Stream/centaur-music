"""Piece registry exports."""

from code_musics.pieces.bwv_846 import PIECES as _BWV_846_PIECES
from code_musics.pieces.bwv_846_fugue import PIECES as _BWV_846_FUGUE_PIECES
from code_musics.pieces.coltrane_studies import PIECES as _COLTRANE_PIECES
from code_musics.pieces.colundi_sequence import PIECES as _COLUNDI_SEQUENCE_PIECES
from code_musics.pieces.composition_showcases import (
    PIECES as _COMPOSITION_SHOWCASE_PIECES,
)
from code_musics.pieces.counterpoint_studies import PIECES as _COUNTERPOINT_PIECES
from code_musics.pieces.crystal_canon import PIECES as _CRYSTAL_CANON_PIECES
from code_musics.pieces.effects_showcase import PIECES as _EFFECTS_SHOWCASE_PIECES
from code_musics.pieces.harmonic_studies import PIECES as _HARMONIC_STUDY_PIECES
from code_musics.pieces.ji import PIECES as _JI_PIECES
from code_musics.pieces.justly_intoned_synth import PIECES as _JUSTLY_INTONED_PIECES
from code_musics.pieces.mellow_studies import PIECES as _MELLOW_PIECES
from code_musics.pieces.natural_steps import PIECES as _NATURAL_STEPS_PIECES
from code_musics.pieces.organ_passacaglia import PIECES as _ORGAN_CHORALE_PIECES
from code_musics.pieces.registry import merge_piece_maps
from code_musics.pieces.septimal import PIECES as _SEPTIMAL_PIECES
from code_musics.pieces.septimal_bloom import PIECES as _SEPTIMAL_BLOOM_PIECES
from code_musics.pieces.slow_glass import PIECES as _SLOW_GLASS_PIECES
from code_musics.pieces.spectral_studies import PIECES as _SPECTRAL_STUDY_PIECES
from code_musics.pieces.studies import PIECES as _STUDIES_PIECES
from code_musics.pieces.techno_studies import PIECES as _TECHNO_PIECES
from code_musics.pieces.texture_studies import PIECES as _TEXTURE_STUDY_PIECES
from code_musics.pieces.trance_studies import PIECES as _TRANCE_PIECES
from code_musics.pieces.wtc_sketches import PIECES as _WTC_PIECES

PIECES = merge_piece_maps(
    _BWV_846_PIECES,
    _BWV_846_FUGUE_PIECES,
    _SEPTIMAL_PIECES,
    _SEPTIMAL_BLOOM_PIECES,
    _COUNTERPOINT_PIECES,
    _HARMONIC_STUDY_PIECES,
    _TEXTURE_STUDY_PIECES,
    _COMPOSITION_SHOWCASE_PIECES,
    _JI_PIECES,
    _COLTRANE_PIECES,
    _EFFECTS_SHOWCASE_PIECES,
    _MELLOW_PIECES,
    _WTC_PIECES,
    _JUSTLY_INTONED_PIECES,
    _NATURAL_STEPS_PIECES,
    _CRYSTAL_CANON_PIECES,
    _SLOW_GLASS_PIECES,
    _SPECTRAL_STUDY_PIECES,
    _TECHNO_PIECES,
    _COLUNDI_SEQUENCE_PIECES,
    _STUDIES_PIECES,
    _ORGAN_CHORALE_PIECES,
    _TRANCE_PIECES,
)

__all__ = ["PIECES"]
