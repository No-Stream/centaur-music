from __future__ import annotations

from collections.abc import Sequence
from itertools import cycle, islice
from typing import TYPE_CHECKING, Any

from code_musics.composition import RhythmCell, line, ratio_line

if TYPE_CHECKING:
    from code_musics.composition import HarmonicContext, PitchKind
    from code_musics.score import Phrase


def euclidean_pattern(hits: int, steps: int, *, rotation: int = 0) -> tuple[bool, ...]:
    """Bjorklund's algorithm: distribute *hits* onsets as evenly as possible across *steps*."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if hits < 0 or hits > steps:
        raise ValueError("hits must satisfy 0 <= hits <= steps")

    if hits == 0:
        return (False,) * steps
    if hits == steps:
        return (True,) * steps

    groups: list[list[bool]] = [[True] for _ in range(hits)] + [
        [False] for _ in range(steps - hits)
    ]

    while True:
        remainder_count = len(groups) - hits
        if remainder_count <= 1:
            break
        distribute_count = min(hits, remainder_count)
        new_groups: list[list[bool]] = []
        for i in range(distribute_count):
            new_groups.append(groups[i] + groups[hits + i])
        for i in range(distribute_count, hits):
            new_groups.append(groups[i])
        for i in range(hits + distribute_count, len(groups)):
            new_groups.append(groups[i])
        groups = new_groups
        hits = distribute_count
        if hits <= 1:
            break

    flat = [step for group in groups for step in group]

    if rotation != 0:
        r = rotation % len(flat)
        flat = flat[-r:] + flat[:-r]

    return tuple(flat)


def euclidean_rhythm(
    hits: int,
    steps: int,
    *,
    span: float = 0.25,
    rotation: int = 0,
) -> RhythmCell | None:
    """Convert a euclidean pattern into a RhythmCell.

    Silent steps are absorbed into the preceding sounding step's span.
    Returns None when hits is 0 (no sounding positions).
    """
    if span <= 0:
        raise ValueError("span must be positive")

    pattern = euclidean_pattern(hits, steps, rotation=rotation)

    if hits == 0:
        return None

    sounding_spans: list[float] = []
    current_span = 0.0
    started = False

    leading_rests = 0
    for step in pattern:
        if step:
            break
        leading_rests += 1

    rotated = list(pattern[leading_rests:]) + list(pattern[:leading_rests])

    for step in rotated:
        current_span += span
        if step and started:
            sounding_spans.append(current_span - span)
            current_span = span
        elif step:
            started = True

    sounding_spans.append(current_span)

    return RhythmCell(spans=tuple(sounding_spans))


def euclidean_line(
    tones: Sequence[float],
    hits: int,
    steps: int,
    *,
    span: float = 0.25,
    rotation: int = 0,
    pitch_kind: PitchKind = "partial",
    amp: float | None = None,
    amp_db: float | None = None,
    gate: float = 0.9,
    synth: dict[str, Any] | None = None,
    context: HarmonicContext | None = None,
) -> Phrase:
    """Build a phrase with tones distributed across a euclidean rhythm.

    Tones cycle through sounding positions. When *context* is provided the tones
    are treated as ratios resolved against the context tonic.
    """
    if not tones:
        raise ValueError("tones must not be empty")

    rhythm = euclidean_rhythm(hits, steps, span=span, rotation=rotation)
    if rhythm is None:
        raise ValueError("euclidean_line requires at least one hit")

    sounding_count = len(rhythm.spans)
    cycled_tones = list(islice(cycle(tones), sounding_count))

    rhythm_with_gate = RhythmCell(spans=rhythm.spans, gates=gate)

    if context is not None:
        return ratio_line(
            tones=cycled_tones,
            rhythm=rhythm_with_gate,
            context=context,
            amp=amp,
            amp_db=amp_db,
            synth_defaults=synth,
        )

    return line(
        tones=cycled_tones,
        rhythm=rhythm_with_gate,
        pitch_kind=pitch_kind,
        amp=amp,
        amp_db=amp_db,
        synth_defaults=synth,
    )
