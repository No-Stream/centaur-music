"""Aksak (additive meter) pattern builder."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from code_musics.composition import RhythmCell

if TYPE_CHECKING:
    from code_musics.meter import Timeline


@dataclass(frozen=True)
class AksakPattern:
    """Additive meter built from unequal pulse groups."""

    grouping: tuple[int, ...]  # e.g., (3, 3, 2) for 8-pulse aksak
    pulse: float  # duration of one pulse unit (seconds)

    def __post_init__(self) -> None:
        if not self.grouping:
            raise ValueError("grouping must be non-empty")
        if any(g <= 0 for g in self.grouping):
            raise ValueError("all group sizes must be positive")
        if self.pulse <= 0 or not math.isfinite(self.pulse):
            raise ValueError("pulse must be positive and finite")

    @classmethod
    def from_timeline(cls, grouping: Sequence[int], tl: Timeline) -> AksakPattern:
        """Derive pulse duration from a Timeline's sixteenth-note duration."""
        from code_musics.meter import S

        return cls(grouping=tuple(grouping), pulse=tl.duration(S))

    def to_rhythm(self) -> RhythmCell:
        """Convert to a RhythmCell.

        Each group produces one span equal to group_size * pulse.
        For per-pulse accenting, use ``to_pulses(accent_first=True)``.
        """
        spans = tuple(g * self.pulse for g in self.grouping)
        return RhythmCell(spans=spans)

    def to_pulses(self, *, accent_first: bool = True) -> RhythmCell:
        """Expand all pulses as individual equal steps.

        If *accent_first*, the first pulse of each group gets ``gate=1.0``
        and subsequent pulses get ``gate=0.7``.  When False, all pulses
        are uniform ``gate=1.0``.
        """
        total = sum(self.grouping)
        spans = tuple(self.pulse for _ in range(total))
        if not accent_first:
            return RhythmCell(spans=spans)
        gates: list[float] = []
        for group_size in self.grouping:
            gates.append(1.0)
            gates.extend(0.7 for _ in range(group_size - 1))
        return RhythmCell(spans=spans, gates=tuple(gates))

    def to_meter(self) -> tuple[int, int]:
        """Return the equivalent time signature as (numerator, 8)."""
        return (sum(self.grouping), 8)

    @property
    def total_pulses(self) -> int:
        return sum(self.grouping)

    @property
    def total_duration(self) -> float:
        return self.total_pulses * self.pulse

    # Named presets
    @classmethod
    def turkish_9(cls, pulse: float) -> AksakPattern:
        """9/8 as 2+2+2+3 (Turkish zeybek)."""
        return cls(grouping=(2, 2, 2, 3), pulse=pulse)

    @classmethod
    def balkan_7(cls, pulse: float) -> AksakPattern:
        """7/8 as 2+2+3 (common Balkan)."""
        return cls(grouping=(2, 2, 3), pulse=pulse)

    @classmethod
    def take_five(cls, pulse: float) -> AksakPattern:
        """5/4 as 3+2 (Dave Brubeck)."""
        return cls(grouping=(3, 2), pulse=pulse)
