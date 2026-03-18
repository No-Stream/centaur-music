"""Composition and rendering tools for code-musics."""

from code_musics.composition import (
    ArticulationSpec,
    RhythmCell,
    canon,
    echo,
    legato,
    line,
    progression,
    recontextualize_phrase,
    sequence,
    staccato,
    voiced_ratio_chord,
)
from code_musics.humanize import (
    DriftSpec,
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    VelocityHumanizeSpec,
)
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import VelocityParamMap

__all__ = [
    "ArticulationSpec",
    "DriftSpec",
    "EnvelopeHumanizeSpec",
    "PitchMotionSpec",
    "RhythmCell",
    "TimingHumanizeSpec",
    "VelocityHumanizeSpec",
    "VelocityParamMap",
    "canon",
    "echo",
    "legato",
    "line",
    "progression",
    "recontextualize_phrase",
    "sequence",
    "staccato",
    "voiced_ratio_chord",
]
