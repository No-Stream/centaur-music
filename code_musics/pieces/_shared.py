"""Small shared constants for study-piece modules."""

from __future__ import annotations

from code_musics.score import EffectSpec
from code_musics.synth import BRICASTI_IR_DIR
from code_musics.tuning import TuningTable


def bwv_846_tuning() -> TuningTable:
    """5-limit major table customized for BWV 846's chromatic moments.

    Shared by both the Prelude and Fugue modules.  Two overrides from the
    stock five_limit_major table:

    C# (pitch class 1): 25/24 instead of 16/15.
    Bb (pitch class 10): 7/4 instead of 9/5 (septimal minor 7th).
    """
    base = TuningTable.five_limit_major()
    ratios = list(base.ratios)
    labels = list(base.labels)
    ratios[1] = 25 / 24
    labels[1] = "25/24"
    ratios[10] = 7 / 4
    labels[10] = "7/4"
    return TuningTable(
        ratios=tuple(ratios),
        labels=tuple(labels),
        name="BWV 846 (5-limit + septimal Bb)",
    )


def _make_reverb(wet: float) -> EffectSpec:
    if BRICASTI_IR_DIR.exists():
        return EffectSpec(
            "bricasti", {"ir_name": "1 Halls 07 Large & Dark", "wet": wet}
        )
    return EffectSpec("reverb", {"room_size": 0.75, "damping": 0.6, "wet_level": wet})


REVERB_EFFECT = _make_reverb(0.32)
SOFT_REVERB_EFFECT = _make_reverb(0.25)
DELAY_EFFECT = EffectSpec(
    "delay",
    {"delay_seconds": 0.32, "feedback": 0.22, "mix": 0.16},
)
WARM_SATURATION_EFFECT = EffectSpec(
    "saturation",
    {"preset": "tube_warm", "mix": 0.24, "drive": 1.14},
)
