"""Composition and rendering tools for code-musics."""

from code_musics.composition import (
    ArticulationSpec,
    RhythmCell,
    echo,
    legato,
    line,
    staccato,
)
from code_musics.humanize import DriftSpec, EnvelopeHumanizeSpec, TimingHumanizeSpec
from code_musics.pitch_motion import PitchMotionSpec

__all__ = [
    "ArticulationSpec",
    "DriftSpec",
    "EnvelopeHumanizeSpec",
    "PitchMotionSpec",
    "RhythmCell",
    "TimingHumanizeSpec",
    "echo",
    "legato",
    "line",
    "staccato",
]
