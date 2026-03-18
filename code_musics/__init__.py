"""Composition and rendering tools for code-musics."""

from code_musics.composition import ArticulationSpec, RhythmCell, echo, legato, line, staccato
from code_musics.pitch_motion import PitchMotionSpec

__all__ = [
    "ArticulationSpec",
    "PitchMotionSpec",
    "RhythmCell",
    "echo",
    "legato",
    "line",
    "staccato",
]
