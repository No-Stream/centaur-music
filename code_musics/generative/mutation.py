"""Stochastic rhythm mutation for evolving groove patterns."""

from __future__ import annotations

import random
from dataclasses import replace

from code_musics.generative._rng import make_rng
from code_musics.score import NoteEvent, Phrase

_RNG = random.Random


def mutate_rhythm(
    phrase: Phrase,
    *,
    add_prob: float = 0.0,
    drop_prob: float = 0.0,
    shift_amount: float = 0.0,
    subdivide_prob: float = 0.0,
    merge_prob: float = 0.0,
    accent_drift: float = 0.0,
    seed: int = 0,
) -> Phrase:
    """Stochastic variation of an existing phrase's rhythm.

    add_prob: probability of inserting a ghost note between events
    drop_prob: probability of removing an event
    shift_amount: max onset shift in seconds (uniform random [-shift, +shift])
    subdivide_prob: probability of splitting a note into two at its midpoint
    merge_prob: probability of merging a note with the next (keeps first pitch, combined duration)
    accent_drift: max velocity change per note (uniform random)

    Each mutation is independent. Apply repeatedly with different seeds
    for evolving grooves across sections.
    """
    _validate_prob(add_prob, "add_prob")
    _validate_prob(drop_prob, "drop_prob")
    _validate_prob(subdivide_prob, "subdivide_prob")
    _validate_prob(merge_prob, "merge_prob")
    if shift_amount < 0:
        raise ValueError("shift_amount must be non-negative")
    if accent_drift < 0:
        raise ValueError("accent_drift must be non-negative")

    if not phrase.events:
        return phrase

    rng = make_rng(seed)
    events = list(phrase.events)

    # 1. Drop (keep at least one event)
    events = _apply_drop(events, drop_prob, rng)

    # 2. Merge adjacent events
    events = _apply_merge(events, merge_prob, rng)

    # 3. Subdivide events
    events = _apply_subdivide(events, subdivide_prob, rng)

    # 4. Add ghost notes between events
    events = _apply_add(events, add_prob, rng)

    # 5. Shift onsets
    events = _apply_shift(events, shift_amount, rng)

    # 6. Accent drift
    events = _apply_accent_drift(events, accent_drift, rng)

    return Phrase(events=tuple(events))


# --- mutation passes ---


def _apply_drop(
    events: list[NoteEvent],
    prob: float,
    rng: _RNG,
) -> list[NoteEvent]:
    if prob <= 0 or len(events) <= 1:
        return events
    surviving = [e for e in events if rng.random() >= prob]
    return surviving if surviving else [events[0]]


def _apply_merge(
    events: list[NoteEvent],
    prob: float,
    rng: _RNG,
) -> list[NoteEvent]:
    if prob <= 0 or len(events) < 2:
        return events
    result: list[NoteEvent] = []
    i = 0
    while i < len(events):
        if i + 1 < len(events) and rng.random() < prob:
            e = events[i]
            enext = events[i + 1]
            merged_dur = e.duration + enext.duration
            synth_dict = dict(e.synth) if e.synth is not None else None
            result.append(replace(e, duration=merged_dur, synth=synth_dict))
            i += 2
        else:
            result.append(events[i])
            i += 1
    return result


def _apply_subdivide(
    events: list[NoteEvent],
    prob: float,
    rng: _RNG,
) -> list[NoteEvent]:
    if prob <= 0:
        return events
    result: list[NoteEvent] = []
    for e in events:
        if rng.random() < prob:
            half_dur = e.duration / 2.0
            synth_a = dict(e.synth) if e.synth is not None else None
            synth_b = dict(e.synth) if e.synth is not None else None
            result.append(replace(e, duration=half_dur, synth=synth_a))
            result.append(
                replace(e, start=e.start + half_dur, duration=half_dur, synth=synth_b)
            )
        else:
            result.append(e)
    return result


def _apply_add(
    events: list[NoteEvent],
    prob: float,
    rng: _RNG,
) -> list[NoteEvent]:
    if prob <= 0 or len(events) < 2:
        return events
    result: list[NoteEvent] = [events[0]]
    for i in range(1, len(events)):
        if rng.random() < prob:
            prev = events[i - 1]
            curr = events[i]
            mid_start = (prev.start + prev.duration + curr.start) / 2.0
            mid_dur = (
                min(curr.start - mid_start, 0.05) if curr.start > mid_start else 0.01
            )
            mid_dur = max(mid_dur, 0.001)
            # Ghost note: use the previous event's pitch, low velocity
            ghost_vel = min(0.3, prev.velocity * 0.4)
            ghost_vel = max(0.01, ghost_vel)
            synth_dict = dict(prev.synth) if prev.synth is not None else None
            ghost = replace(
                prev,
                start=mid_start,
                duration=mid_dur,
                velocity=ghost_vel,
                synth=synth_dict,
            )
            result.append(ghost)
        result.append(events[i])
    return result


def _apply_shift(
    events: list[NoteEvent],
    amount: float,
    rng: _RNG,
) -> list[NoteEvent]:
    if amount <= 0:
        return events
    result: list[NoteEvent] = []
    for e in events:
        offset = rng.uniform(-amount, amount)
        new_start = max(0.0, e.start + offset)
        synth_dict = dict(e.synth) if e.synth is not None else None
        result.append(replace(e, start=new_start, synth=synth_dict))
    return result


def _apply_accent_drift(
    events: list[NoteEvent],
    drift: float,
    rng: _RNG,
) -> list[NoteEvent]:
    if drift <= 0:
        return events
    result: list[NoteEvent] = []
    for e in events:
        delta = rng.uniform(-drift, drift)
        new_vel = max(0.01, min(2.0, e.velocity + delta))
        synth_dict = dict(e.synth) if e.synth is not None else None
        result.append(replace(e, velocity=new_vel, synth=synth_dict))
    return result


def _validate_prob(value: float, name: str) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0.0, 1.0]")
