"""Thin tuning helpers for just intonation and EDO work."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

_NOTE_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


@dataclass(frozen=True)
class TuningTable:
    """Maps 12-TET pitch classes to JI ratios. A starting point for retuning, not a final answer."""

    ratios: tuple[float, ...]
    labels: tuple[str, ...]
    name: str

    def __post_init__(self) -> None:
        if len(self.ratios) != 12:
            raise ValueError(
                f"TuningTable requires exactly 12 ratios, got {len(self.ratios)}"
            )
        if len(self.labels) != 12:
            raise ValueError(
                f"TuningTable requires exactly 12 labels, got {len(self.labels)}"
            )
        if any(r <= 0 for r in self.ratios):
            raise ValueError("All TuningTable ratios must be positive")

    def ratio_for(self, midi_note: int, root_midi_note: int = 60) -> float:
        """Return the JI ratio for a MIDI note, including octave displacement.

        root_midi_note defines which MIDI note maps to 1/1 (the unison).
        """
        degree = (midi_note - root_midi_note) % 12
        octave = (midi_note - root_midi_note) // 12
        return self.ratios[degree] * (2.0**octave)

    def resolve(self, midi_note: int, f0: float, root_midi_note: int = 60) -> float:
        """Map a MIDI note to a JI frequency. f0 is the frequency of root_midi_note."""
        return f0 * self.ratio_for(midi_note, root_midi_note)

    def label_for(self, midi_note: int, root_midi_note: int = 60) -> str:
        """Human-readable ratio label for a MIDI note."""
        degree = (midi_note - root_midi_note) % 12
        return self.labels[degree]

    def describe(self, root_midi_note: int = 60) -> str:
        """Return a readable string showing the full 12-class mapping."""
        lines = [f"TuningTable: {self.name}"]
        for i in range(12):
            pc = (root_midi_note + i) % 12
            cents = ratio_to_cents(self.ratios[i])
            lines.append(
                f"  {_NOTE_NAMES[pc]:>2s} -> {self.labels[i]:>7s}  ({cents:7.1f}c)"
            )
        return "\n".join(lines)

    @classmethod
    def five_limit_major(cls) -> TuningTable:
        """Standard 5-limit JI chromatic scale rooted on the unison."""
        return cls(
            ratios=(
                1 / 1,
                16 / 15,
                9 / 8,
                6 / 5,
                5 / 4,
                4 / 3,
                45 / 32,
                3 / 2,
                8 / 5,
                5 / 3,
                9 / 5,
                15 / 8,
            ),
            labels=(
                "1/1",
                "16/15",
                "9/8",
                "6/5",
                "5/4",
                "4/3",
                "45/32",
                "3/2",
                "8/5",
                "5/3",
                "9/5",
                "15/8",
            ),
            name="5-limit major",
        )

    @classmethod
    def seven_limit(cls) -> TuningTable:
        """7-limit JI chromatic scale with septimal intervals."""
        return cls(
            ratios=(
                1 / 1,
                15 / 14,
                9 / 8,
                7 / 6,
                5 / 4,
                4 / 3,
                7 / 5,
                3 / 2,
                8 / 5,
                5 / 3,
                7 / 4,
                15 / 8,
            ),
            labels=(
                "1/1",
                "15/14",
                "9/8",
                "7/6",
                "5/4",
                "4/3",
                "7/5",
                "3/2",
                "8/5",
                "5/3",
                "7/4",
                "15/8",
            ),
            name="7-limit",
        )

    @classmethod
    def pythagorean(cls) -> TuningTable:
        """Pure 3-limit Pythagorean tuning from chain of fifths."""
        return cls(
            ratios=(
                1 / 1,
                256 / 243,
                9 / 8,
                32 / 27,
                81 / 64,
                4 / 3,
                729 / 512,
                3 / 2,
                128 / 81,
                27 / 16,
                16 / 9,
                243 / 128,
            ),
            labels=(
                "1/1",
                "256/243",
                "9/8",
                "32/27",
                "81/64",
                "4/3",
                "729/512",
                "3/2",
                "128/81",
                "27/16",
                "16/9",
                "243/128",
            ),
            name="pythagorean",
        )


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


def otonal(f0: float, partials: list[float] | list[int]) -> list[float]:
    """Resolve harmonic partials over a fundamental."""
    return [f0 * partial for partial in partials]


def utonal(f0: float, partials: list[float] | list[int]) -> list[float]:
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


def tenney_height(ratio: float) -> float:
    """Tenney height of a JI ratio: log2(p * q) where p/q is in lowest terms.

    Lower values indicate more consonant intervals (e.g. 3/2 -> ~2.58, 7/4 -> ~4.81).
    """
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    frac = Fraction(ratio).limit_denominator(10000)
    return math.log2(frac.numerator * frac.denominator)


def _prime_factors(n: int) -> set[int]:
    """Return the set of prime factors of a positive integer."""
    factors: set[int] = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def _within_prime_limit(n: int, prime_limit: int) -> bool:
    """Check whether all prime factors of n are <= prime_limit."""
    if n <= 1:
        return True
    return all(p <= prime_limit for p in _prime_factors(n))


def enumerate_ji_ratios(
    low: float,
    high: float,
    prime_limit: int = 7,
    max_height: float = 15.0,
) -> list[float]:
    """Enumerate JI ratios in [low, high] within the given prime limit and Tenney height.

    Returns sorted ascending by pitch. Only ratios whose numerator and denominator
    have all prime factors <= prime_limit are included.
    """
    if low > high:
        raise ValueError("low must be <= high")
    if prime_limit < 2:
        raise ValueError("prime_limit must be at least 2")

    max_denom = 128
    max_term = int(2**max_height) + 1

    seen: set[tuple[int, int]] = set()
    results: list[float] = []

    for q in range(1, max_denom + 1):
        if not _within_prime_limit(q, prime_limit):
            continue
        p_low = max(1, math.ceil(low * q))
        p_high = min(max_term, math.floor(high * q))
        for p in range(p_low, p_high + 1):
            if not _within_prime_limit(p, prime_limit):
                continue
            frac = Fraction(p, q)
            canonical = (frac.numerator, frac.denominator)
            if canonical in seen:
                continue
            seen.add(canonical)
            ratio_value = p / q
            if ratio_value < low or ratio_value > high:
                continue
            height = math.log2(frac.numerator * frac.denominator)
            if height > max_height:
                continue
            results.append(ratio_value)

    results.sort()
    return results
