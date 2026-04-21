"""Cellular automata rhythm generators."""

from __future__ import annotations

from code_musics.composition import RhythmCell
from code_musics.generative._rng import make_rng


def ca_rhythm(
    rule: int,
    steps: int,
    *,
    init: int | None = None,
    span: float = 0.25,
    row: int = -1,
    seed: int = 0,
) -> RhythmCell:
    """1D elementary cellular automaton as a rhythm generator.

    rule: Wolfram rule number (0-255)
    steps: width of the CA (number of cells/time steps)
    init: initial state as a bit pattern. None = seeded random pattern.
    span: duration per step (seconds)
    row: which generation to use (-1 = last, after ``steps`` generations)
    seed: deterministic seed for the initial state when *init* is None.
        Ignored when *init* is provided (the initial state is fully determined).

    The binary pattern maps to onsets. Silent steps are absorbed into
    the preceding sounding step (same strategy as euclidean rhythms).
    If the row is all zeros, falls back to a single hit covering the full span.
    """
    if not (0 <= rule <= 255):
        raise ValueError("rule must be in 0-255")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if span <= 0:
        raise ValueError("span must be positive")

    grid = _evolve(rule, steps, init=init, seed=seed, generations=abs(row))
    target_row = grid[row]

    return _pattern_to_rhythm(target_row, span)


def ca_rhythm_layers(
    rule: int,
    steps: int,
    *,
    layers: int = 3,
    init: int | None = None,
    span: float = 0.25,
    seed: int = 0,
) -> list[RhythmCell]:
    """Multiple rows from the same CA as layered rhythm patterns.

    Picks *layers* evenly-spaced rows from the evolution history.

    When *init* is None, the initial state is generated from *seed*.
    When *init* is provided, *seed* is ignored.
    """
    if layers <= 0:
        raise ValueError("layers must be positive")
    if not (0 <= rule <= 255):
        raise ValueError("rule must be in 0-255")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if span <= 0:
        raise ValueError("span must be positive")

    generations = max(steps, layers)
    grid = _evolve(rule, steps, init=init, seed=seed, generations=generations)

    # Pick evenly-spaced rows (excluding row 0 which is just the init state).
    total_rows = len(grid)
    indices = [
        1 + round(i * (total_rows - 2) / (layers - 1)) if layers > 1 else total_rows - 1
        for i in range(layers)
    ]
    return [_pattern_to_rhythm(grid[idx], span) for idx in indices]


# --- internal ---


def _evolve(
    rule: int,
    width: int,
    *,
    init: int | None,
    seed: int,
    generations: int,
) -> list[list[bool]]:
    """Run an elementary CA for *generations* steps and return the full grid."""
    if init is not None:
        state = [(init >> i) & 1 == 1 for i in range(width)]
    else:
        rng = make_rng(seed)
        state = [rng.random() < 0.5 for _ in range(width)]

    rule_bits = [(rule >> i) & 1 == 1 for i in range(8)]
    grid: list[list[bool]] = [state[:]]

    for _ in range(generations):
        new_state: list[bool] = []
        for j in range(width):
            left = state[(j - 1) % width]
            center = state[j]
            right = state[(j + 1) % width]
            neighborhood = (int(left) << 2) | (int(center) << 1) | int(right)
            new_state.append(rule_bits[neighborhood])
        state = new_state
        grid.append(state[:])

    return grid


def _pattern_to_rhythm(pattern: list[bool], span: float) -> RhythmCell:
    """Convert a boolean onset pattern to a RhythmCell with rest absorption."""
    if not any(pattern):
        return RhythmCell(spans=(span * len(pattern),))

    sounding_spans: list[float] = []
    current_span = 0.0

    for is_hit in pattern:
        current_span += span
        if is_hit:
            if sounding_spans:
                sounding_spans.append(current_span)
            else:
                sounding_spans.append(current_span)
            current_span = 0.0

    # Trailing rest merges into the last sounding span.
    if current_span > 0.0 and sounding_spans:
        sounding_spans[-1] += current_span

    return RhythmCell(spans=tuple(sounding_spans))
