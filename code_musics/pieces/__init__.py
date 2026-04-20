"""Piece registry exports."""

from code_musics.pieces.additive_studies import PIECES as _ADDITIVE_STUDY_PIECES
from code_musics.pieces.amber_room import PIECES as _AMBER_ROOM_PIECES
from code_musics.pieces.beating_light import PIECES as _BEATING_LIGHT_PIECES
from code_musics.pieces.bell_pulse import PIECES as _BELL_PULSE_PIECES
from code_musics.pieces.breath_study import PIECES as _BREATH_STUDY_PIECES
from code_musics.pieces.bwv_846 import PIECES as _BWV_846_PIECES
from code_musics.pieces.bwv_846_fugue import PIECES as _BWV_846_FUGUE_PIECES
from code_musics.pieces.coltrane_studies import PIECES as _COLTRANE_PIECES
from code_musics.pieces.colundi_sequence import PIECES as _COLUNDI_SEQUENCE_PIECES
from code_musics.pieces.comma_pump import PIECES as _COMMA_PUMP_PIECES
from code_musics.pieces.composition_showcases import (
    PIECES as _COMPOSITION_SHOWCASE_PIECES,
)
from code_musics.pieces.counterpoint_studies import PIECES as _COUNTERPOINT_PIECES
from code_musics.pieces.crystal_canon import PIECES as _CRYSTAL_CANON_PIECES
from code_musics.pieces.diva_study import PIECES as _DIVA_STUDY_PIECES
from code_musics.pieces.effects_showcase import PIECES as _EFFECTS_SHOWCASE_PIECES
from code_musics.pieces.emergence_reverse import PIECES as _EMERGENCE_REVERSE_PIECES
from code_musics.pieces.filter_palette_study import (
    PIECES as _FILTER_PALETTE_STUDY_PIECES,
)
from code_musics.pieces.forge import PIECES as _FORGE_PIECES
from code_musics.pieces.harmonic_studies import PIECES as _HARMONIC_STUDY_PIECES
from code_musics.pieces.iron_pulse import PIECES as _IRON_PULSE_PIECES
from code_musics.pieces.iron_pulse_v2 import PIECES as _IRON_PULSE_V2_PIECES
from code_musics.pieces.ji import PIECES as _JI_PIECES
from code_musics.pieces.justly_intoned_synth import PIECES as _JUSTLY_INTONED_PIECES
from code_musics.pieces.md_study import PIECES as _MD_STUDY_PIECES
from code_musics.pieces.mellow_studies import PIECES as _MELLOW_PIECES
from code_musics.pieces.mirror_dialogue import PIECES as _MIRROR_DIALOGUE_PIECES
from code_musics.pieces.mod_matrix_study import PIECES as _MOD_MATRIX_STUDY_PIECES
from code_musics.pieces.natural_steps import PIECES as _NATURAL_STEPS_PIECES
from code_musics.pieces.newton_bloom import PIECES as _NEWTON_BLOOM_PIECES
from code_musics.pieces.night_lattice import PIECES as _NIGHT_LATTICE_PIECES
from code_musics.pieces.organ_passacaglia import PIECES as _ORGAN_CHORALE_PIECES
from code_musics.pieces.phase_garden import PIECES as _PHASE_GARDEN_PIECES
from code_musics.pieces.registry import merge_piece_maps
from code_musics.pieces.septimal import PIECES as _SEPTIMAL_PIECES
from code_musics.pieces.septimal_bloom import PIECES as _SEPTIMAL_BLOOM_PIECES
from code_musics.pieces.seventh_window import PIECES as _SEVENTH_WINDOW_PIECES
from code_musics.pieces.slow_glass import PIECES as _SLOW_GLASS_PIECES
from code_musics.pieces.slow_glass_v2 import PIECES as _SLOW_GLASS_V2_PIECES
from code_musics.pieces.spectral_passage import PIECES as _SPECTRAL_PASSAGE_PIECES
from code_musics.pieces.spectral_studies import PIECES as _SPECTRAL_STUDY_PIECES
from code_musics.pieces.studies import PIECES as _STUDIES_PIECES
from code_musics.pieces.tape_hymn import PIECES as _TAPE_HYMN_PIECES
from code_musics.pieces.techno_studies import PIECES as _TECHNO_PIECES
from code_musics.pieces.texture_studies import PIECES as _TEXTURE_STUDY_PIECES
from code_musics.pieces.trance_studies import PIECES as _TRANCE_PIECES
from code_musics.pieces.va_showcase import PIECES as _VA_SHOWCASE_PIECES
from code_musics.pieces.velvet_wall import PIECES as _VELVET_WALL_PIECES
from code_musics.pieces.warming_up import PIECES as _WARMING_UP_PIECES
from code_musics.pieces.wtc_sketches import PIECES as _WTC_PIECES

PIECES = merge_piece_maps(
    _ADDITIVE_STUDY_PIECES,
    _AMBER_ROOM_PIECES,
    _BWV_846_PIECES,
    _BWV_846_FUGUE_PIECES,
    _SEPTIMAL_PIECES,
    _SEPTIMAL_BLOOM_PIECES,
    _COUNTERPOINT_PIECES,
    _HARMONIC_STUDY_PIECES,
    _IRON_PULSE_PIECES,
    _IRON_PULSE_V2_PIECES,
    _TEXTURE_STUDY_PIECES,
    _COMPOSITION_SHOWCASE_PIECES,
    _JI_PIECES,
    _COLTRANE_PIECES,
    _EFFECTS_SHOWCASE_PIECES,
    _FILTER_PALETTE_STUDY_PIECES,
    _MELLOW_PIECES,
    _WTC_PIECES,
    _JUSTLY_INTONED_PIECES,
    _NATURAL_STEPS_PIECES,
    _CRYSTAL_CANON_PIECES,
    _SLOW_GLASS_PIECES,
    _SLOW_GLASS_V2_PIECES,
    _SPECTRAL_PASSAGE_PIECES,
    _SPECTRAL_STUDY_PIECES,
    _TECHNO_PIECES,
    _COLUNDI_SEQUENCE_PIECES,
    _STUDIES_PIECES,
    _ORGAN_CHORALE_PIECES,
    _PHASE_GARDEN_PIECES,
    _TRANCE_PIECES,
    _VELVET_WALL_PIECES,
    _BEATING_LIGHT_PIECES,
    _COMMA_PUMP_PIECES,
    _EMERGENCE_REVERSE_PIECES,
    _MIRROR_DIALOGUE_PIECES,
    _SEVENTH_WINDOW_PIECES,
    _TAPE_HYMN_PIECES,
    _BELL_PULSE_PIECES,
    _NIGHT_LATTICE_PIECES,
    _WARMING_UP_PIECES,
    _FORGE_PIECES,
    _BREATH_STUDY_PIECES,
    _DIVA_STUDY_PIECES,
    _MD_STUDY_PIECES,
    _VA_SHOWCASE_PIECES,
    _MOD_MATRIX_STUDY_PIECES,
    _NEWTON_BLOOM_PIECES,
)

__all__ = ["PIECES"]
