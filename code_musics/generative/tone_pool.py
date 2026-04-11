from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from code_musics.generative._rng import make_rng


@dataclass(frozen=True)
class TonePool:
    """Weighted pitch pool for stochastic pitch selection.

    Ratios are frequency ratios (e.g. 1.0, 1.25, 1.5 for 4:5:6).
    Weights are normalized to sum to 1.0.
    """

    ratios: tuple[float, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.ratios) == 0:
            raise ValueError("ratios must be non-empty")
        if len(self.ratios) != len(self.weights):
            raise ValueError(
                f"ratios and weights must have the same length, "
                f"got {len(self.ratios)} and {len(self.weights)}"
            )
        if any(w <= 0 for w in self.weights):
            raise ValueError("all weights must be positive")
        total = sum(self.weights)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"weights must sum to 1.0, got {total}")

    @classmethod
    def uniform(cls, ratios: Sequence[float]) -> TonePool:
        """Create a pool with equal weights for all ratios."""
        if len(ratios) == 0:
            raise ValueError("ratios must be non-empty")
        n = len(ratios)
        w = 1.0 / n
        return cls(ratios=tuple(ratios), weights=tuple(w for _ in ratios))

    @classmethod
    def weighted(cls, mapping: dict[float, float]) -> TonePool:
        """Create a pool from a {ratio: weight} mapping, auto-normalized."""
        if len(mapping) == 0:
            raise ValueError("mapping must be non-empty")
        if any(w <= 0 for w in mapping.values()):
            raise ValueError("all weights must be positive")
        total = sum(mapping.values())
        ratios = tuple(mapping.keys())
        weights = tuple(v / total for v in mapping.values())
        return cls(ratios=ratios, weights=weights)

    @classmethod
    def from_harmonics(cls, partials: Sequence[int]) -> TonePool:
        """Create a pool from harmonic series partial numbers.

        Ratios are partial/min(partials), with uniform weights.
        E.g. [4, 5, 6, 7] -> ratios (1.0, 1.25, 1.5, 1.75).
        """
        if len(partials) == 0:
            raise ValueError("partials must be non-empty")
        if any(p <= 0 for p in partials):
            raise ValueError("all partials must be positive integers")
        base = min(partials)
        ratios = tuple(p / base for p in partials)
        return cls.uniform(ratios)

    def draw(self, n: int, *, seed: int, replace: bool = True) -> list[float]:
        """Draw n ratios according to weights.

        Args:
            n: Number of draws.
            seed: Deterministic seed for reproducibility.
            replace: If True, draw with replacement. If False, draw without
                replacement (n must not exceed pool size).
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if not replace and n > len(self.ratios):
            raise ValueError(
                f"cannot draw {n} without replacement from pool of {len(self.ratios)}"
            )
        rng = make_rng(seed)
        if replace:
            return rng.choices(self.ratios, weights=self.weights, k=n)
        else:
            remaining_ratios = list(self.ratios)
            remaining_weights = list(self.weights)
            results: list[float] = []
            for _ in range(n):
                chosen = rng.choices(remaining_ratios, weights=remaining_weights, k=1)[
                    0
                ]
                idx = remaining_ratios.index(chosen)
                results.append(chosen)
                remaining_ratios.pop(idx)
                remaining_weights.pop(idx)
            return results

    def draw_one(self, *, rng: random.Random) -> float:
        """Draw a single ratio using an external RNG instance."""
        return rng.choices(self.ratios, weights=self.weights, k=1)[0]
