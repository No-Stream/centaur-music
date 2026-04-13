"""Study: weighted tone pools shaping melodic character.

Demonstrates how TonePool weight distributions create contrasting melodic
personalities over three sections:

  A  Consonant — root, fifth, and major third dominate.
  B  Septimal  — 7-limit color tones dominate.
  C  Mixed     — a broad pool blending both palettes.

A sustained drone on the root anchors the ear throughout so the shifting
pool character is clearly audible.
"""

from __future__ import annotations

from code_musics.composition import (
    ArticulationSpec,
    HarmonicContext,
    ratio_line,
    sequence,
)
from code_musics.generative.tone_pool import TonePool
from code_musics.humanize import EnvelopeHumanizeSpec, TimingHumanizeSpec
from code_musics.pieces._shared import SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Score

# ---------------------------------------------------------------------------
# Tonic and timing constants
# ---------------------------------------------------------------------------

F0_HZ: float = 220.0
NOTE_DUR: float = 0.55
SECTION_NOTES: int = 12
SECTION_DUR: float = NOTE_DUR * SECTION_NOTES
GAP_BETWEEN_SECTIONS: float = 0.4
DRONE_TAIL: float = 1.5

# ---------------------------------------------------------------------------
# Weighted tone pools
# ---------------------------------------------------------------------------

CONSONANT_POOL: TonePool = TonePool.weighted(
    {
        1.0: 5.0,  # root — strong anchor
        5 / 4: 3.0,  # major third (5-limit)
        3 / 2: 4.0,  # perfect fifth
        2.0: 2.0,  # octave
        3 / 4: 1.0,  # fifth below (adds a lower neighbor)
    }
)

SEPTIMAL_POOL: TonePool = TonePool.weighted(
    {
        1.0: 2.0,  # root — lighter anchor
        7 / 4: 4.0,  # harmonic seventh
        7 / 6: 3.5,  # septimal minor third
        7 / 5: 3.0,  # septimal tritone
        9 / 7: 2.5,  # septimal major third
    }
)

MIXED_POOL: TonePool = TonePool.weighted(
    {
        1.0: 3.0,  # root
        5 / 4: 2.0,  # major third
        3 / 2: 2.5,  # fifth
        7 / 4: 2.0,  # harmonic seventh
        7 / 6: 1.5,  # septimal minor third
        9 / 7: 1.5,  # septimal major third
        2.0: 1.0,  # octave
    }
)

# ---------------------------------------------------------------------------
# Piece builder
# ---------------------------------------------------------------------------


def build_score() -> Score:
    """Build a ~25-second tone-pool study."""
    total_dur = (SECTION_DUR * 3) + (GAP_BETWEEN_SECTIONS * 2) + DRONE_TAIL
    score = Score(
        f0=F0_HZ,
        timing_humanize=TimingHumanizeSpec(preset="tight_ensemble"),
        master_effects=[SOFT_REVERB_EFFECT],
    )

    context = HarmonicContext(tonic=F0_HZ, name="root")

    # -- drone voice ---------------------------------------------------------
    score.add_voice(
        "drone",
        synth_defaults={
            "engine": "additive",
            "preset": "drone",
            "n_harmonics": 6,
            "harmonic_rolloff": 0.55,
        },
        mix_db=-3.0,
    )
    score.add_note(
        "drone",
        start=0.0,
        duration=total_dur,
        freq=F0_HZ,
        amp_db=-10.0,
        label="root_drone",
    )
    # quiet octave layer for body
    score.add_note(
        "drone",
        start=0.0,
        duration=total_dur,
        freq=F0_HZ * 2.0,
        amp_db=-18.0,
        label="octave_drone",
    )

    # -- melody voice --------------------------------------------------------
    melody_synth: dict[str, object] = {
        "engine": "additive",
        "n_harmonics": 5,
        "harmonic_rolloff": 0.35,
        "attack": 0.06,
        "decay": 0.18,
        "sustain_level": 0.50,
        "release": 0.40,
    }
    score.add_voice(
        "melody",
        synth_defaults=melody_synth,
        effects=[
            EffectSpec("delay", {"delay_seconds": 0.28, "feedback": 0.15, "mix": 0.12}),
        ],
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        mix_db=-1.0,
    )

    articulation = ArticulationSpec(gate=0.88)
    rhythm = [NOTE_DUR] * SECTION_NOTES

    # Section A — consonant pool
    tones_a = CONSONANT_POOL.draw(SECTION_NOTES, seed=42)
    phrase_a = ratio_line(
        tones_a,
        rhythm,
        context=context,
        amp_db=-6.0,
        articulation=articulation,
    )

    # Section B — septimal pool
    tones_b = SEPTIMAL_POOL.draw(SECTION_NOTES, seed=77)
    phrase_b = ratio_line(
        tones_b,
        rhythm,
        context=context,
        amp_db=-6.0,
        articulation=articulation,
    )

    # Section C — mixed pool
    tones_c = MIXED_POOL.draw(SECTION_NOTES, seed=123)
    phrase_c = ratio_line(
        tones_c,
        rhythm,
        context=context,
        amp_db=-6.0,
        articulation=articulation,
    )

    # Place each section with gaps between them
    section_starts = [
        0.0,
        SECTION_DUR + GAP_BETWEEN_SECTIONS,
        (SECTION_DUR + GAP_BETWEEN_SECTIONS) * 2,
    ]
    sequence(
        score,
        "melody",
        phrase_a,
        starts=[section_starts[0]],
    )
    sequence(
        score,
        "melody",
        phrase_b,
        starts=[section_starts[1]],
    )
    sequence(
        score,
        "melody",
        phrase_c,
        starts=[section_starts[2]],
    )

    return score


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

PIECES: dict[str, PieceDefinition] = {
    "study_tone_pool": PieceDefinition(
        name="study_tone_pool",
        output_name="study_tone_pool",
        build_score=build_score,
        study=True,
    ),
}
