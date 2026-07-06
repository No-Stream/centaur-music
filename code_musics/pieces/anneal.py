"""anneal — Colundi stretch-drift piece and its audition studies.

Sethares co-design: every tonal voice's partials are drawn from the scale's
own degrees, so the scale's intervals are consonant by construction and the
Act II pseudo-octave stretch warps scale and spectrum together.

Design spec: docs/plans/2026-07-05-anneal-design.md
Plan:        docs/plans/2026-07-05-anneal-plan.md

`anneal_fusion_sketch` is audition study 1: a chord ladder proving
(a) skeleton-spectrum fusion, (b) chord-role-aware color-partial fusion, and
(c) that a matched stretch at P=2.07 reads as "warped world", not "out of
tune". Four chords at home tuning, then a continuous stretch ramp on the
tonic (the piece's Act II gesture in miniature) with a slow anneal home.
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

# Color partials are CHORD-ROLE-AWARE: a note carries a color partial only
# when it is an exact octave-multiple of a chord-internal interval, so it
# lands on another chord tone's partial. Giving every note the same color
# list creates comma collisions instead of fusion (audition 1 finding: the
# fifth's 19/8 partial at 3/2*19/8 = 3.5625 beat against the root's 7/4
# partial at 3.5 — a 57/56 clash, ~31 cents / ~6 Hz of roughness).
COLOR_19_RATIOS = [19 / 8, 19 / 4]  # on the ROOT of 16:19:24 -> third's partials
COLOR_49_RATIOS = [49 / 40 * 2, 49 / 40 * 4]  # 2.45, 4.9: on the lower two
# notes of the neutral triad -> the 49/30-note's partials (plus a 0.7-cent
# slow shimmer against the octave note).

# Chords as (degree, color_ratios) pairs; 2.0 degrees stretch to the
# pseudo-octave like everything else.
TONIC_4_6_7: list[tuple[float, list[float]]] = [
    (1.0, []),
    (3 / 2, []),
    (7 / 4, []),
]
MINOR_16_19_24: list[tuple[float, list[float]]] = [
    (1.0, COLOR_19_RATIOS),
    (19 / 16, []),
    (3 / 2, []),
]
NEUTRAL_SUBDOMINANT: list[tuple[float, list[float]]] = [
    (4 / 3, COLOR_49_RATIOS),
    (49 / 30, COLOR_49_RATIOS),
    (2.0, []),
]


def _with_color(
    base: list[dict[str, float]], color_ratios: list[float], weight: float = 0.4
) -> list[dict[str, float]]:
    """Blend color-degree partials into a fused spectrum at reduced weight."""
    if not color_ratios:
        return base
    extra = [{"ratio": ratio, "amp": weight / ratio} for ratio in color_ratios]
    return sorted(base + extra, key=lambda partial: partial["ratio"])


def _smoothstep(x: float) -> float:
    clamped = min(max(x, 0.0), 1.0)
    return 3 * clamped**2 - 2 * clamped**3


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

    def add_chord(
        chord: list[tuple[float, list[float]]],
        start: float,
        duration: float,
        pseudo_octave: float,
    ) -> None:
        for note_index, (degree, color_ratios) in enumerate(chord):
            partials = _stretched_partials(
                _with_color(SKELETON_PARTIALS, color_ratios), pseudo_octave
            )
            score.add_note(
                "pad",
                start=start,
                duration=duration,
                partial=stretch_ratio(degree, pseudo_octave),
                amp_db=-15.0 if note_index == 0 else -18.0,
                synth={"partials": partials},
            )

    # Chord ladder at home tuning (role-aware color partials).
    add_chord(TONIC_4_6_7, 0.0, 5.0, HOME_PSEUDO_OCTAVE)
    add_chord(MINOR_16_19_24, 6.0, 5.0, HOME_PSEUDO_OCTAVE)
    add_chord(NEUTRAL_SUBDOMINANT, 12.0, 5.0, HOME_PSEUDO_OCTAVE)
    add_chord(TONIC_4_6_7, 18.0, 5.0, HOME_PSEUDO_OCTAVE)

    # Continuous stretch ramp — the piece's Act II gesture in miniature.
    # Tonic re-struck every 3 s while S climbs 2.00 -> 2.07 (smoothstep).
    stretch_span = PEAK_PSEUDO_OCTAVE - HOME_PSEUDO_OCTAVE
    for strike in range(6):
        onset = 24.0 + strike * 3.0
        pseudo_octave = HOME_PSEUDO_OCTAVE + stretch_span * _smoothstep(
            (onset - 24.0) / 18.0
        )
        add_chord(TONIC_4_6_7, onset, 2.8, pseudo_octave)

    # Hold at full stretch.
    add_chord(TONIC_4_6_7, 42.0, 5.0, PEAK_PSEUDO_OCTAVE)

    # Anneal home (ease-out, slower feel than the climb).
    for strike in range(3):
        onset = 48.0 + strike * 3.0
        pseudo_octave = HOME_PSEUDO_OCTAVE + stretch_span * (
            1.0 - _smoothstep((onset - 48.0) / 12.0)
        )
        add_chord(TONIC_4_6_7, onset, 2.8, pseudo_octave)
    add_chord(TONIC_4_6_7, 57.0, 6.0, HOME_PSEUDO_OCTAVE)
    return score


_FUSION_SECTIONS = (
    PieceSection("tonic 4:6:7 (home)", 0.0, 6.0),
    PieceSection("16:19:24, root 19-color (home)", 6.0, 12.0),
    PieceSection("neutral triad, 49-fusion (home)", 12.0, 18.0),
    PieceSection("tonic again (home)", 18.0, 24.0),
    PieceSection("stretch ramp 2.00 -> 2.07", 24.0, 42.0),
    PieceSection("held at 2.07", 42.0, 48.0),
    PieceSection("anneal home", 48.0, 63.0),
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
