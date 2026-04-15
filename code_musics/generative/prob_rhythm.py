"""Probabilistic rhythm generation with per-step onset and accent weights."""

from __future__ import annotations

from collections.abc import Sequence

from code_musics.composition import RhythmCell
from code_musics.generative._rng import make_rng


def prob_rhythm(
    steps: int,
    *,
    onset_weights: Sequence[float] | float = 0.7,
    accent_weights: Sequence[float] | float = 1.0,
    span: float = 0.25,
    seed: int = 0,
) -> RhythmCell:
    """Generate a rhythm from per-step onset probabilities.

    onset_weights cycle if shorter than steps — [1.0, 0.3, 0.5, 0.3]
    naturally emphasizes beat positions in a 16th grid.
    accent_weights set the gate (articulation) of surviving onsets.

    Returns a RhythmCell with at least one onset. If the random draw
    produces zero onsets, the first step is forced on.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if span <= 0:
        raise ValueError("span must be positive")

    onset_list = _expand_weights(onset_weights, steps, "onset_weights")
    accent_list = _expand_weights(accent_weights, steps, "accent_weights")

    rng = make_rng(seed)
    hits = [rng.random() < w for w in onset_list]

    if not any(hits):
        hits[0] = True

    # Rest-absorption: silent steps merge into the preceding sounding step's span.
    sounding_spans: list[float] = []
    sounding_gates: list[float] = []
    current_span = 0.0

    for i, is_hit in enumerate(hits):
        current_span += span
        if is_hit:
            if sounding_spans:
                sounding_spans.append(current_span)
            else:
                # First hit — include any leading rest spans
                sounding_spans.append(current_span)
            sounding_gates.append(accent_list[i])
            current_span = 0.0

    # Trailing rest after the last hit gets added to the last sounding span.
    if current_span > 0.0 and sounding_spans:
        sounding_spans[-1] += current_span

    gates: float | tuple[float, ...] = (
        sounding_gates[0] if len(set(sounding_gates)) == 1 else tuple(sounding_gates)
    )
    return RhythmCell(spans=tuple(sounding_spans), gates=gates)


def _expand_weights(
    weights: Sequence[float] | float,
    length: int,
    name: str,
) -> list[float]:
    """Expand scalar or short sequence to *length* by cycling, validating non-negative."""
    if isinstance(weights, (int, float)):
        if weights < 0:
            raise ValueError(f"{name} must be non-negative")
        return [float(weights)] * length
    if not weights:
        raise ValueError(f"{name} must be non-empty")
    expanded = [float(weights[i % len(weights)]) for i in range(length)]
    if any(w < 0 for w in expanded):
        raise ValueError(f"{name} must be non-negative")
    return expanded
