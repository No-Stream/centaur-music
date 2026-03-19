"""Compatibility re-exports for former `sketches.py` builders."""

from code_musics.pieces.composition_showcases import (
    PIECES as _COMPOSITION_SHOWCASE_PIECES,
)
from code_musics.pieces.composition_showcases import (
    build_articulation_study_sketch,
    build_composition_tools_consonant_score,
    build_composition_tools_showcase_score,
)
from code_musics.pieces.counterpoint_studies import (
    PIECES as _COUNTERPOINT_PIECES,
)
from code_musics.pieces.counterpoint_studies import (
    build_invention_sketch,
    build_passacaglia_sketch,
    build_variations_sketch,
)
from code_musics.pieces.harmonic_studies import (
    PIECES as _HARMONIC_STUDY_PIECES,
)
from code_musics.pieces.harmonic_studies import (
    build_arpeggios_cross_sketch,
    build_arpeggios_sketch,
)
from code_musics.pieces.registry import merge_piece_maps
from code_musics.pieces.texture_studies import (
    PIECES as _TEXTURE_STUDY_PIECES,
)
from code_musics.pieces.texture_studies import (
    build_interference_ji_sketch,
    build_interference_sketch,
    build_interference_v2_sketch,
    build_spiral_arch_sketch,
    build_spiral_sketch,
)

PIECES = merge_piece_maps(
    _COUNTERPOINT_PIECES,
    _HARMONIC_STUDY_PIECES,
    _TEXTURE_STUDY_PIECES,
    _COMPOSITION_SHOWCASE_PIECES,
)

__all__ = [
    "PIECES",
    "build_arpeggios_cross_sketch",
    "build_arpeggios_sketch",
    "build_articulation_study_sketch",
    "build_composition_tools_consonant_score",
    "build_composition_tools_showcase_score",
    "build_invention_sketch",
    "build_interference_ji_sketch",
    "build_interference_sketch",
    "build_interference_v2_sketch",
    "build_passacaglia_sketch",
    "build_spiral_arch_sketch",
    "build_spiral_sketch",
    "build_variations_sketch",
]
