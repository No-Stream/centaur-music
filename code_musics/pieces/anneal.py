"""anneal — Colundi stretch-drift piece and its audition studies.

Sethares co-design: every tonal voice's partials are drawn from the scale's
own degrees, so the scale's intervals are consonant by construction and the
Act II pseudo-octave stretch warps scale and spectrum together.

Design spec: docs/plans/2026-07-05-anneal-design.md
Plan:        docs/plans/2026-07-05-anneal-plan.md

`anneal_fusion_sketch` is audition study 1: a fixed A/B chord ladder proving
(a) skeleton-spectrum fusion, (b) 19-/49-family color-partial fusion, and
(c) that a matched stretch at P=2.07 reads as "warped world", not "out of
tune". Four chords at home tuning, then the same four fully stretched.
"""

from __future__ import annotations

from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS, SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import Score
from code_musics.spectra import scale_fused_spectrum
from code_musics.tuning import stretch_ratio

F0 = 98.0  # G2; kick fundamental (later tasks) at 49 Hz

HOME_PSEUDO_OCTAVE = 2.0
PEAK_PSEUDO_OCTAVE = 2.07

# 3/7-limit skeleton degrees -> integer partials {1,2,3,3.5,4,6,7,8,12,14}.
# Deliberately NO 5th/10th harmonic: the scale's third is 19/16, and a 5/4
# partial would beat against every third in the piece.
SKELETON_DEGREES = [1.0, 3 / 2, 7 / 4]
SKELETON_PARTIALS = scale_fused_spectrum(SKELETON_DEGREES, octaves=3)

# Color partials: octave transpositions of the scale's color degrees.
COLOR_19_RATIOS = [19 / 8, 19 / 4]  # 2.375, 4.75
COLOR_49_RATIOS = [49 / 15, 98 / 15]  # ~3.267, ~6.533

# Chord spellings (scale degrees; 2.0 entries stretch to the pseudo-octave).
TONIC_4_6_7 = [1.0, 3 / 2, 7 / 4]
MINOR_16_19_24 = [1.0, 19 / 16, 3 / 2]
NEUTRAL_SUBDOMINANT = [4 / 3, 49 / 30, 2.0]


def _with_color(
    base: list[dict[str, float]], color_ratios: list[float], weight: float = 0.6
) -> list[dict[str, float]]:
    """Blend color-degree partials into a fused spectrum at reduced weight."""
    extra = [{"ratio": ratio, "amp": weight / ratio} for ratio in color_ratios]
    return sorted(base + extra, key=lambda partial: partial["ratio"])


def _stretched_partials(
    partials: list[dict[str, float]], pseudo_octave: float
) -> list[dict[str, float]]:
    """Map a partial list through the stretch law (identity at 2.0)."""
    return [
        {"ratio": stretch_ratio(partial["ratio"], pseudo_octave), "amp": partial["amp"]}
        for partial in partials
    ]


def _fusion_sketch_score() -> Score:
    score = Score(
        f0_hz=F0,
        master_effects=DEFAULT_MASTER_EFFECTS,
    )
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "additive",
            "attack": 0.8,
            "release": 1.6,
            "decay_power": 2.0,
            "spectral_morph_type": "phase_disperse",
            "spectral_morph_amount": 0.35,
        },
        effects=[SOFT_REVERB_EFFECT],
        pan=0.0,
    )

    # Order matches _VARIANT_LABELS / the registered sections.
    variants: list[tuple[list[float], list[dict[str, float]]]] = [
        (TONIC_4_6_7, SKELETON_PARTIALS),
        (MINOR_16_19_24, SKELETON_PARTIALS),
        (MINOR_16_19_24, _with_color(SKELETON_PARTIALS, COLOR_19_RATIOS)),
        (NEUTRAL_SUBDOMINANT, _with_color(SKELETON_PARTIALS, COLOR_49_RATIOS)),
    ]

    slot_dur = 6.0
    hold_dur = 5.0
    for pass_index, pseudo_octave in enumerate(
        [HOME_PSEUDO_OCTAVE, PEAK_PSEUDO_OCTAVE]
    ):
        for variant_index, (chord_degrees, partials) in enumerate(variants):
            start = (pass_index * len(variants) + variant_index) * slot_dur
            stretched = _stretched_partials(partials, pseudo_octave)
            for degree in chord_degrees:
                score.add_note(
                    "pad",
                    start=start,
                    duration=hold_dur,
                    partial=stretch_ratio(degree, pseudo_octave),
                    amp_db=-15.0 if degree == chord_degrees[0] else -18.0,
                    synth={"partials": stretched},
                )
    return score


_VARIANT_LABELS = [
    "tonic 4:6:7 skeleton",
    "16:19:24 skeleton only",
    "16:19:24 + 19-color",
    "neutral 4/3 + 49-color",
]

_FUSION_SECTIONS = tuple(
    PieceSection(
        f"{label} ({tuning_label})",
        (pass_index * len(_VARIANT_LABELS) + variant_index) * 6.0,
        (pass_index * len(_VARIANT_LABELS) + variant_index) * 6.0 + 6.0,
    )
    for pass_index, tuning_label in enumerate(["home", "stretched 2.07"])
    for variant_index, label in enumerate(_VARIANT_LABELS)
)

PIECES = {
    "anneal_fusion_sketch": PieceDefinition(
        name="anneal_fusion_sketch",
        output_name="anneal_fusion_sketch",
        build_score=_fusion_sketch_score,
        sections=_FUSION_SECTIONS,
        study=True,
    ),
}
