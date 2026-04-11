from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from code_musics.composition import HarmonicContext
from code_musics.generative._rng import make_rng
from code_musics.generative.tone_pool import TonePool
from code_musics.score import NoteEvent, Phrase


def _integrate_density(
    breakpoints: Sequence[tuple[float, float]], duration: float
) -> float:
    """Integrate the piecewise-linear density curve over [0, duration]."""
    total = 0.0
    for i in range(len(breakpoints) - 1):
        t0_frac, d0 = breakpoints[i]
        t1_frac, d1 = breakpoints[i + 1]
        t0 = t0_frac * duration
        t1 = t1_frac * duration
        segment_dur = t1 - t0
        total += 0.5 * (d0 + d1) * segment_dur
    return total


def _cumulative_density(
    breakpoints: Sequence[tuple[float, float]], duration: float
) -> list[tuple[float, float]]:
    """Build cumulative integral breakpoints: [(time, cumulative_area), ...]."""
    points: list[tuple[float, float]] = []
    acc = 0.0
    for i, (t_frac, _d) in enumerate(breakpoints):
        t = t_frac * duration
        if i > 0:
            prev_t_frac, prev_d = breakpoints[i - 1]
            cur_d = breakpoints[i][1]
            prev_t = prev_t_frac * duration
            segment_dur = t - prev_t
            acc += 0.5 * (prev_d + cur_d) * segment_dur
        points.append((t, acc))
    return points


def _invert_cdf(u: float, cumulative: list[tuple[float, float]]) -> float:
    """Invert the cumulative density to find the time for a given uniform sample."""
    if u <= 0.0:
        return cumulative[0][0]
    if u >= cumulative[-1][1]:
        return cumulative[-1][0]
    for i in range(len(cumulative) - 1):
        t0, c0 = cumulative[i]
        t1, c1 = cumulative[i + 1]
        if c0 <= u <= c1:
            segment_area = c1 - c0
            if segment_area < 1e-15:
                return t0
            frac = (u - c0) / segment_area
            return t0 + frac * (t1 - t0)
    return cumulative[-1][0]


def _validate_breakpoints(breakpoints: Sequence[tuple[float, float]]) -> None:
    if len(breakpoints) < 2:
        raise ValueError("density breakpoints must have at least 2 entries")
    for i, (t, d) in enumerate(breakpoints):
        if d < 0:
            raise ValueError(f"density must be non-negative at breakpoint {i}, got {d}")
        if i > 0 and t < breakpoints[i - 1][0]:
            raise ValueError("density breakpoint time fractions must be non-decreasing")
    if breakpoints[0][0] != 0.0:
        raise ValueError("first breakpoint time fraction must be 0.0")
    if breakpoints[-1][0] != 1.0:
        raise ValueError("last breakpoint time fraction must be 1.0")


def _draw_tone(pool: TonePool | Sequence[float], rng: random.Random) -> float:
    if isinstance(pool, TonePool):
        return pool.draw_one(rng=rng)
    return rng.choice(pool)


def stochastic_cloud(
    *,
    tones: TonePool | Sequence[float],
    duration: float,
    density: float | Sequence[tuple[float, float]] = 5.0,
    amp_db_range: tuple[float, float] = (-18.0, -6.0),
    note_dur_range: tuple[float, float] = (0.1, 0.5),
    pitch_kind: str = "partial",
    context: HarmonicContext | None = None,
    seed: int = 0,
    synth: dict[str, Any] | None = None,
) -> Phrase:
    """Generate a cloud of stochastic notes as a Phrase."""
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    if amp_db_range[0] > amp_db_range[1]:
        raise ValueError(
            f"amp_db_range[0] must be <= amp_db_range[1], got {amp_db_range}"
        )
    if note_dur_range[0] > note_dur_range[1]:
        raise ValueError(
            f"note_dur_range[0] must be <= note_dur_range[1], got {note_dur_range}"
        )
    if note_dur_range[0] <= 0:
        raise ValueError(
            f"note_dur_range values must be positive, got {note_dur_range}"
        )
    if pitch_kind not in ("partial", "freq"):
        raise ValueError(f"pitch_kind must be 'partial' or 'freq', got {pitch_kind!r}")
    if not isinstance(tones, TonePool) and len(tones) == 0:
        raise ValueError("tones must not be empty")

    rng = make_rng(seed)

    if isinstance(density, (int, float)):
        if density <= 0:
            raise ValueError(f"density must be positive, got {density}")
        n_notes = round(float(density) * duration)
        start_times = sorted(rng.uniform(0.0, duration) for _ in range(n_notes))
    else:
        breakpoints = list(density)
        _validate_breakpoints(breakpoints)
        total_integral = _integrate_density(breakpoints, duration)
        n_notes = round(total_integral)
        cumulative = _cumulative_density(breakpoints, duration)
        uniforms = sorted(rng.uniform(0.0, total_integral) for _ in range(n_notes))
        start_times = [_invert_cdf(u, cumulative) for u in uniforms]

    events: list[NoteEvent] = []
    for t in start_times:
        tone = _draw_tone(tones, rng)
        amp_db = rng.uniform(amp_db_range[0], amp_db_range[1])
        dur = rng.uniform(note_dur_range[0], note_dur_range[1])

        if context is not None:
            freq_val = context.resolve_ratio(tone)
            event = NoteEvent(
                start=t,
                duration=dur,
                amp_db=amp_db,
                freq=freq_val,
                synth=synth,
            )
        elif pitch_kind == "freq":
            event = NoteEvent(
                start=t,
                duration=dur,
                amp_db=amp_db,
                freq=tone,
                synth=synth,
            )
        else:
            event = NoteEvent(
                start=t,
                duration=dur,
                amp_db=amp_db,
                partial=tone,
                synth=synth,
            )
        events.append(event)

    return Phrase(events=tuple(events))
