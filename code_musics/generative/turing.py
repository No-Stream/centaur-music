from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from code_musics.composition import HarmonicContext, RhythmCell
from code_musics.generative._rng import _ratios_to_phrase, make_rng
from code_musics.generative.tone_pool import TonePool
from code_musics.score import Phrase


@dataclass(frozen=True)
class TuringMachine:
    """Shift-register sequencer inspired by the Music Thing Turing Machine.

    When flip_probability is 0 the register loops with period ``length``.
    Small flip values introduce gradual mutations; 1.0 is fully random.
    """

    length: int = 8
    flip_probability: float = 0.0
    tones: TonePool | Sequence[float] = ()
    seed: int = 0

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError(f"length must be positive, got {self.length}")
        if not (0.0 <= self.flip_probability <= 1.0):
            raise ValueError(
                f"flip_probability must be in [0, 1], got {self.flip_probability}"
            )
        tone_count = (
            len(self.tones.ratios)
            if isinstance(self.tones, TonePool)
            else len(self.tones)
        )
        if tone_count == 0:
            raise ValueError("tones must be non-empty")

    def _tone_at(self, index: int) -> float:
        if isinstance(self.tones, TonePool):
            return self.tones.ratios[index % len(self.tones.ratios)]
        return self.tones[index % len(self.tones)]

    def generate(self, n: int) -> list[float]:
        """Generate *n* ratios from the shift-register process."""
        rng = make_rng(self.seed)
        register = [rng.randint(0, 1) for _ in range(self.length)]
        results: list[float] = []
        for _ in range(n):
            value = 0
            for bit in register:
                value = (value << 1) | bit
            tone_count = (
                len(self.tones.ratios)
                if isinstance(self.tones, TonePool)
                else len(self.tones)
            )
            results.append(self._tone_at(value % tone_count))
            msb = register[0]
            register = register[1:]
            coin = 1 if rng.random() < self.flip_probability else 0
            register.append(msb ^ coin)
        return results

    def to_phrase(
        self,
        n: int,
        rhythm: RhythmCell | Sequence[float],
        *,
        context: HarmonicContext | None = None,
        **line_kwargs: Any,
    ) -> Phrase:
        """Generate ratios and build a :class:`Phrase`."""
        ratios = self.generate(n)
        return _ratios_to_phrase(ratios, rhythm, context=context, **line_kwargs)
