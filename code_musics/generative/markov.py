from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from code_musics.composition import HarmonicContext, RhythmCell
from code_musics.generative._rng import _ratios_to_phrase, make_rng
from code_musics.score import Phrase


def _normalize_weights(raw: dict[float, float]) -> list[tuple[float, float]]:
    if not raw:
        raise ValueError("transition targets must be non-empty")
    if any(w <= 0 for w in raw.values()):
        raise ValueError("all weights must be positive")
    total = sum(raw.values())
    return [(ratio, w / total) for ratio, w in raw.items()]


@dataclass(frozen=True)
class RatioMarkov:
    """Markov chain over JI ratios with configurable order."""

    _table: dict[tuple[float, ...], list[tuple[float, float]]]
    order: int

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError("order must be at least 1")
        if not self._table:
            raise ValueError("transition table must be non-empty")
        for state, targets in self._table.items():
            if len(state) != self.order:
                raise ValueError(
                    f"state {state} has length {len(state)}, expected {self.order}"
                )
            if not targets:
                raise ValueError(f"state {state} has no transition targets")
            total = sum(w for _, w in targets)
            if abs(total - 1.0) > 1e-9:
                raise ValueError(
                    f"weights for state {state} sum to {total}, expected 1.0"
                )

    def generate(
        self,
        n: int,
        *,
        start: float | tuple[float, ...] | None = None,
        seed: int = 0,
    ) -> list[float]:
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if n == 0:
            return []

        rng = make_rng(seed)
        states = list(self._table.keys())

        if start is None:
            state = states[rng.randint(0, len(states) - 1)]
        elif isinstance(start, (int, float)):
            state = (float(start),)
        else:
            state = tuple(float(x) for x in start)

        if len(state) != self.order:
            raise ValueError(f"start has length {len(state)}, expected {self.order}")

        result: list[float] = list(state)

        while len(result) < n:
            targets = self._table.get(state)
            if targets is None:
                state = states[rng.randint(0, len(states) - 1)]
                targets = self._table[state]
            ratios_list = [r for r, _w in targets]
            weights_list = [w for _r, w in targets]
            next_ratio = rng.choices(ratios_list, weights=weights_list, k=1)[0]
            result.append(next_ratio)
            state = (*state[1:], next_ratio)

        return result[:n]

    def to_phrase(
        self,
        n: int,
        rhythm: RhythmCell | Sequence[float],
        *,
        seed: int = 0,
        start: float | tuple[float, ...] | None = None,
        context: HarmonicContext | None = None,
        **line_kwargs: Any,
    ) -> Phrase:
        ratios = self.generate(n, start=start, seed=seed)
        line_kwargs.setdefault("pitch_kind", "partial")
        return _ratios_to_phrase(ratios, rhythm, context=context, **line_kwargs)

    @classmethod
    def from_transitions(
        cls, transitions: dict[float, dict[float, float]]
    ) -> RatioMarkov:
        if not transitions:
            raise ValueError("transitions must be non-empty")
        table: dict[tuple[float, ...], list[tuple[float, float]]] = {}
        for src, targets in transitions.items():
            table[(float(src),)] = _normalize_weights(targets)
        return cls(_table=table, order=1)

    @classmethod
    def from_table(
        cls,
        transitions: dict[tuple[float, ...], dict[float, float]],
        *,
        order: int = 1,
    ) -> RatioMarkov:
        if not transitions:
            raise ValueError("transitions must be non-empty")
        if order < 1:
            raise ValueError("order must be at least 1")
        table: dict[tuple[float, ...], list[tuple[float, float]]] = {}
        for state, targets in transitions.items():
            if len(state) != order:
                raise ValueError(
                    f"state {state} has length {len(state)}, expected {order}"
                )
            table[tuple(float(x) for x in state)] = _normalize_weights(targets)
        return cls(_table=table, order=order)

    @classmethod
    def from_phrase(
        cls,
        phrase: Phrase,
        *,
        order: int = 1,
        context: HarmonicContext | None = None,
    ) -> RatioMarkov:
        if order < 1:
            raise ValueError("order must be at least 1")

        ratios: list[float] = []
        for event in phrase.events:
            if event.partial is not None:
                ratios.append(event.partial)
            elif event.freq is not None:
                if context is not None:
                    ratios.append(event.freq / context.tonic)
                else:
                    ratios.append(event.freq)
            else:
                raise ValueError("event has neither partial nor freq")

        min_events = order + 1
        if len(ratios) < min_events:
            raise ValueError(
                f"phrase must have at least {min_events} events for order {order}, "
                f"got {len(ratios)}"
            )

        counts: dict[tuple[float, ...], dict[float, float]] = {}
        for i in range(len(ratios) - order):
            state = tuple(ratios[i : i + order])
            next_ratio = ratios[i + order]
            if state not in counts:
                counts[state] = {}
            counts[state][next_ratio] = counts[state].get(next_ratio, 0.0) + 1.0

        table: dict[tuple[float, ...], list[tuple[float, float]]] = {}
        for state, targets in counts.items():
            table[state] = _normalize_weights(targets)
        return cls(_table=table, order=order)
