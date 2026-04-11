"""Generative composition tools: stochastic, algorithmic, and lattice-based generators."""

from __future__ import annotations

from code_musics.generative._rng import make_rng
from code_musics.generative.cloud import stochastic_cloud
from code_musics.generative.euclidean import (
    euclidean_line,
    euclidean_pattern,
    euclidean_rhythm,
)
from code_musics.generative.lattice import LatticeWalker
from code_musics.generative.markov import RatioMarkov
from code_musics.generative.prob_gate import prob_gate
from code_musics.generative.tone_pool import TonePool
from code_musics.generative.turing import TuringMachine

__all__ = [
    "LatticeWalker",
    "RatioMarkov",
    "TonePool",
    "TuringMachine",
    "euclidean_line",
    "euclidean_pattern",
    "euclidean_rhythm",
    "make_rng",
    "prob_gate",
    "stochastic_cloud",
]
