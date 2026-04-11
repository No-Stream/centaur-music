from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from code_musics.composition import HarmonicContext, RhythmCell
from code_musics.generative._rng import _ratios_to_phrase, make_rng
from code_musics.score import Phrase


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _octave_reduce(ratio: float) -> float:
    while ratio < 1.0:
        ratio *= 2.0
    while ratio >= 2.0:
        ratio /= 2.0
    return ratio


@dataclass(frozen=True)
class LatticeWalker:
    """Random walk on the JI prime-factor lattice.

    Each step moves one unit along a randomly chosen prime axis, with optional
    gravitational pull toward the origin (1/1).
    """

    axes: tuple[int, ...] = (3, 5, 7)
    step_weights: dict[int, float] | None = None
    gravity: float = 0.0
    max_distance: int = 3
    octave_reduce: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.axes:
            raise ValueError("axes must be non-empty")
        for p in self.axes:
            if not _is_prime(p):
                raise ValueError(f"all axes must be primes > 1, got {p}")
        if self.gravity < 0:
            raise ValueError(f"gravity must be non-negative, got {self.gravity}")
        if self.max_distance <= 0:
            raise ValueError(f"max_distance must be positive, got {self.max_distance}")
        if self.step_weights is not None:
            for key, val in self.step_weights.items():
                if key not in self.axes:
                    raise ValueError(f"step_weights key {key} not in axes {self.axes}")
                if val <= 0:
                    raise ValueError(
                        f"step_weights values must be positive, got {val} for {key}"
                    )

    def walk(self, n: int, *, start: dict[int, int] | None = None) -> list[float]:
        """Generate *n* ratios by walking the lattice."""
        rng = make_rng(self.seed)
        position: dict[int, int] = {p: 0 for p in self.axes}
        if start is not None:
            for p, exp in start.items():
                if p not in position:
                    raise ValueError(f"start key {p} not in axes {self.axes}")
                position[p] = max(-self.max_distance, min(self.max_distance, exp))

        weights = [
            self.step_weights[p]
            if self.step_weights is not None and p in self.step_weights
            else 1.0
            for p in self.axes
        ]

        results: list[float] = []
        for _ in range(n):
            axis = rng.choices(self.axes, weights=weights, k=1)[0]

            exp = position[axis]
            if exp > 0:
                prob_minus = 0.5 + self.gravity * 0.5
            elif exp < 0:
                prob_minus = 0.5 - self.gravity * 0.5
            else:
                prob_minus = 0.5
            direction = -1 if rng.random() < prob_minus else 1

            new_exp = max(-self.max_distance, min(self.max_distance, exp + direction))
            position[axis] = new_exp

            ratio = math.prod(float(p) ** position[p] for p in self.axes)
            if self.octave_reduce:
                ratio = _octave_reduce(ratio)
            results.append(ratio)
        return results

    def to_phrase(
        self,
        n: int,
        rhythm: RhythmCell | Sequence[float],
        *,
        context: HarmonicContext | None = None,
        start: dict[int, int] | None = None,
        **line_kwargs: Any,
    ) -> Phrase:
        """Walk the lattice and build a :class:`Phrase`."""
        ratios = self.walk(n, start=start)
        return _ratios_to_phrase(ratios, rhythm, context=context, **line_kwargs)
