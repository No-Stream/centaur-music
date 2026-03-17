"""Thin tuning helpers for just intonation and EDO work."""

import math


def harmonic_series(f0: float, n_partials: int) -> list[float]:
    """Return the first n partials of the harmonic series."""
    if n_partials < 1:
        raise ValueError("n_partials must be at least 1")
    return [f0 * partial for partial in range(1, n_partials + 1)]


def ji_chord(f0: float, ratios: list[float]) -> list[float]:
    """Resolve JI ratios over a fundamental."""
    return [f0 * ratio for ratio in ratios]


def edo_scale(f0: float, divisions: int, octaves: int = 1) -> list[float]:
    """Build an equal-division scale across one or more octaves."""
    if divisions < 1:
        raise ValueError("divisions must be at least 1")
    if octaves < 1:
        raise ValueError("octaves must be at least 1")

    n_steps = divisions * octaves
    return [f0 * (2 ** (step / divisions)) for step in range(n_steps + 1)]


def otonal(f0: float, partials: list[float]) -> list[float]:
    """Resolve harmonic partials over a fundamental."""
    return [f0 * partial for partial in partials]


def utonal(f0: float, partials: list[float]) -> list[float]:
    """Resolve subharmonic partials over a fundamental."""
    return [f0 / partial for partial in partials]


def cents_to_ratio(cents: float) -> float:
    """Convert cents to a frequency ratio."""
    return 2 ** (cents / 1200.0)


def ratio_to_cents(ratio: float) -> float:
    """Convert a frequency ratio to cents."""
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    return 1200.0 * math.log2(ratio)
