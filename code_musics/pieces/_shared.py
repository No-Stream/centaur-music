"""Small shared constants for study-piece modules."""

from __future__ import annotations

from code_musics.score import EffectSpec

REVERB_EFFECT = EffectSpec(
    "bricasti",
    {"ir_name": "1 Halls 07 Large & Dark", "wet": 0.32},
)
SOFT_REVERB_EFFECT = EffectSpec(
    "bricasti",
    {"ir_name": "1 Halls 07 Large & Dark", "wet": 0.25},
)
DELAY_EFFECT = EffectSpec(
    "delay",
    {"delay_seconds": 0.32, "feedback": 0.22, "mix": 0.16},
)
WARM_SATURATION_EFFECT = EffectSpec(
    "saturation",
    {"preset": "tube_warm", "mix": 0.24, "drive": 1.14},
)
