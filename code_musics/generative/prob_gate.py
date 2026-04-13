from __future__ import annotations

from collections.abc import Sequence

from code_musics.generative._rng import make_rng
from code_musics.score import NoteEvent, Phrase


def prob_gate(
    phrase: Phrase,
    *,
    density: float = 0.7,
    accent_bias: float = 0.0,
    position_weights: Sequence[float] | None = None,
    seed: int = 0,
) -> Phrase:
    """Probabilistically filter notes from a phrase, preserving original timing."""
    if not (0.0 <= density <= 1.0):
        raise ValueError("density must be in [0.0, 1.0]")
    if not (0.0 <= accent_bias <= 1.0):
        raise ValueError("accent_bias must be in [0.0, 1.0]")
    if position_weights is not None:
        if not position_weights:
            raise ValueError("position_weights must be non-empty")
        if any(w < 0 for w in position_weights):
            raise ValueError("position_weights must be non-negative")

    if not phrase.events:
        return phrase

    max_amp = max(_resolved_amp(event) for event in phrase.events)

    rng = make_rng(seed)
    surviving: list[NoteEvent] = []

    for i, event in enumerate(phrase.events):
        position_weight = (
            position_weights[i % len(position_weights)]
            if position_weights is not None
            else 1.0
        )
        normalized_amp = _resolved_amp(event) / max_amp if max_amp > 0 else 1.0
        accent_factor = 1.0 + accent_bias * (normalized_amp - 0.5)
        p = max(0.0, min(1.0, density * position_weight * accent_factor))

        if rng.random() < p:
            surviving.append(event)

    return Phrase(events=tuple(surviving))


def _resolved_amp(event: NoteEvent) -> float:
    if event.amp is None:
        raise ValueError("event amp unexpectedly missing")
    return float(event.amp)
