"""High-level musical-time helpers that compile to score seconds."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

_VALID_METER_DENOMINATORS = {1, 2, 4, 8, 16, 32}


@dataclass(frozen=True)
class BeatValue:
    """Named rhythmic value expressed in quarter-note beats."""

    beats: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.beats) or self.beats <= 0:
            raise ValueError("beats must be positive and finite")


@dataclass(frozen=True)
class BeatSpan:
    """Beat distance or duration expressed in quarter-note beats."""

    beats: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.beats) or self.beats < 0:
            raise ValueError("beats must be non-negative and finite")


@dataclass(frozen=True)
class MeasurePosition:
    """Absolute bar position, where `1.0` is the start of bar one."""

    measure: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.measure) or self.measure < 1.0:
            raise ValueError("measure must be finite and at least 1.0")


@dataclass(frozen=True)
class MusicalLocation:
    """Resolved musical location for a second offset on a timeline."""

    bar: int
    beat: float
    absolute_beats: float
    seconds: float

    @property
    def beat_within_bar(self) -> float:
        """Alias for the beat offset within the resolved bar."""
        return self.beat


@dataclass(frozen=True)
class Groove:
    """Rhythmic feel specification: per-step timing + velocity patterns."""

    subdivision: Literal["eighth", "sixteenth"]
    timing_offsets: tuple[float, ...]
    velocity_weights: tuple[float, ...]
    name: str = ""

    def __post_init__(self) -> None:
        if self.subdivision not in {"eighth", "sixteenth"}:
            raise ValueError("subdivision must be 'eighth' or 'sixteenth'")
        if not self.timing_offsets:
            raise ValueError("timing_offsets must be non-empty")
        if any(not math.isfinite(o) for o in self.timing_offsets):
            raise ValueError("timing_offsets must all be finite")
        if any(o <= -1.0 or o >= 1.0 for o in self.timing_offsets):
            raise ValueError("timing_offsets must all be in the range (-1.0, 1.0)")
        if not self.velocity_weights:
            raise ValueError("velocity_weights must be non-empty")
        if any(not math.isfinite(w) for w in self.velocity_weights):
            raise ValueError("velocity_weights must all be finite")
        if any(w <= 0 for w in self.velocity_weights):
            raise ValueError("velocity_weights must all be positive")

    @property
    def step_size_beats(self) -> float:
        """Return the beat span of one groove step."""
        return 0.5 if self.subdivision == "eighth" else 0.25

    def timing_offset_at(self, step_index: int) -> float:
        """Return the timing offset for a step index, cycling the pattern."""
        return self.timing_offsets[step_index % len(self.timing_offsets)]

    def velocity_weight_at(self, step_index: int) -> float:
        """Return the velocity weight for a step index, cycling the pattern."""
        return self.velocity_weights[step_index % len(self.velocity_weights)]

    # --- Classic swing factories ---

    @classmethod
    def eighths_swing(cls, amount: float = 2.0 / 3.0) -> Groove:
        """Return an eighth-note swing groove.

        *amount* is the offbeat position in [0.5, 1.0).
        """
        if not math.isfinite(amount) or amount < 0.5 or amount >= 1.0:
            raise ValueError("amount must be finite and in [0.5, 1.0)")
        offset = 2.0 * amount - 1.0
        return cls(
            subdivision="eighth",
            timing_offsets=(0.0, offset),
            velocity_weights=(1.0, 1.0),
            name="eighths_swing",
        )

    @classmethod
    def sixteenths_swing(cls, amount: float = 2.0 / 3.0) -> Groove:
        """Return a sixteenth-note swing groove.

        *amount* is the offbeat position in [0.5, 1.0).
        """
        if not math.isfinite(amount) or amount < 0.5 or amount >= 1.0:
            raise ValueError("amount must be finite and in [0.5, 1.0)")
        offset = 2.0 * amount - 1.0
        return cls(
            subdivision="sixteenth",
            timing_offsets=(0.0, offset),
            velocity_weights=(1.0, 1.0),
            name="sixteenths_swing",
        )

    # --- Named groove presets ---

    @classmethod
    def mpc_tight(cls) -> Groove:
        """MPC-style tight sixteenth groove."""
        return cls(
            subdivision="sixteenth",
            timing_offsets=(0.0, 0.10, 0.0, -0.05),
            velocity_weights=(1.0, 0.65, 0.85, 0.55),
            name="mpc_tight",
        )

    @classmethod
    def dilla_lazy(cls) -> Groove:
        """J Dilla-style lazy sixteenth groove."""
        return cls(
            subdivision="sixteenth",
            timing_offsets=(0.0, 0.22, 0.05, 0.18),
            velocity_weights=(1.0, 0.55, 0.9, 0.5),
            name="dilla_lazy",
        )

    @classmethod
    def motown_pocket(cls) -> Groove:
        """Motown-style eighth-note pocket groove."""
        return cls(
            subdivision="eighth",
            timing_offsets=(0.0, 0.15),
            velocity_weights=(1.0, 0.7),
            name="motown_pocket",
        )

    @classmethod
    def bossa(cls) -> Groove:
        """Bossa nova eighth-note groove with anticipated offbeats."""
        return cls(
            subdivision="eighth",
            timing_offsets=(0.0, -0.08),
            velocity_weights=(1.0, 0.85),
            name="bossa",
        )

    @classmethod
    def tr808_swing(cls) -> Groove:
        """TR-808-style sixteenth swing groove."""
        return cls(
            subdivision="sixteenth",
            timing_offsets=(0.0, 0.25, 0.0, 0.0),
            velocity_weights=(1.0, 0.6, 0.75, 0.5),
            name="tr808_swing",
        )


type DurationLike = float | BeatValue | BeatSpan
type TimePointLike = float | BeatValue | BeatSpan | MeasurePosition

W = BeatValue(4.0)
H = BeatValue(2.0)
Q = BeatValue(1.0)
E = BeatValue(0.5)
S = BeatValue(0.25)


def B(value: float) -> BeatSpan:
    """Return a beat-relative span or absolute beat offset."""
    return BeatSpan(float(value))


def M(value: float) -> MeasurePosition:
    """Return a bar-position reference, where `1.0` is the first bar."""
    return MeasurePosition(float(value))


def dotted(value: DurationLike) -> BeatSpan:
    """Return a dotted version of a beat-relative duration."""
    return BeatSpan(_coerce_duration_beats(value) * 1.5)


def triplet(value: DurationLike) -> BeatSpan:
    """Return a triplet-scaled version of a beat-relative duration."""
    return BeatSpan(_coerce_duration_beats(value) * (2.0 / 3.0))


def tuplet(n: int, in_space_of: int, value: DurationLike) -> BeatSpan:
    """Duration of one note in an n:m tuplet.

    tuplet(5, 4, Q) -> one quintuplet quarter (4/5 of Q)
    tuplet(7, 4, E) -> one septuplet eighth (4/7 of E)
    """
    if n <= 0 or in_space_of <= 0:
        raise ValueError("tuplet counts must be positive")
    base_beats = _coerce_duration_beats(value)
    return BeatSpan(base_beats * in_space_of / n)


@dataclass(frozen=True)
class Timeline:
    """Single-tempo musical timeline for high-level authoring."""

    bpm: float
    meter: tuple[int, int] = (4, 4)
    pickup_beats: float = 0.0
    groove: Groove | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.bpm) or self.bpm <= 0:
            raise ValueError("bpm must be positive and finite")

        numerator, denominator = self.meter
        if numerator <= 0:
            raise ValueError("meter numerator must be positive")
        if denominator not in _VALID_METER_DENOMINATORS:
            raise ValueError("meter denominator must be one of 1, 2, 4, 8, 16, or 32")
        if not math.isfinite(self.pickup_beats) or self.pickup_beats < 0:
            raise ValueError("pickup_beats must be non-negative and finite")

    @property
    def seconds_per_beat(self) -> float:
        """Return seconds per quarter-note beat."""
        return 60.0 / self.bpm

    @property
    def beats_per_bar(self) -> float:
        """Return bar length measured in quarter-note beats."""
        numerator, denominator = self.meter
        return numerator * (4.0 / denominator)

    def measures(self, value: float) -> float:
        """Resolve a measure span into seconds."""
        if not math.isfinite(value) or value < 0:
            raise ValueError("measure span must be non-negative and finite")
        return value * self.beats_per_bar * self.seconds_per_beat

    def duration(self, value: DurationLike) -> float:
        """Resolve a beat-relative duration into seconds."""
        return _coerce_duration_beats(value) * self.seconds_per_beat

    def at(self, *, bar: int, beat: float = 0.0) -> float:
        """Return the second offset for a bar/beat location."""
        if bar < 1:
            raise ValueError("bar must be at least 1")
        if not math.isfinite(beat) or beat < 0:
            raise ValueError("beat must be non-negative and finite")
        absolute_beats = self.pickup_beats + ((bar - 1) * self.beats_per_bar) + beat
        return self._beats_to_seconds(absolute_beats)

    def position(self, value: TimePointLike) -> float:
        """Resolve a musical position reference into seconds."""
        return self._beats_to_seconds(self.absolute_beats(value))

    def absolute_beats(self, value: TimePointLike) -> float:
        """Resolve a position reference into absolute quarter-note beats."""
        if isinstance(value, MeasurePosition):
            return self.pickup_beats + ((value.measure - 1.0) * self.beats_per_bar)

        if isinstance(value, (BeatSpan, BeatValue)):
            return value.beats

        beat_value = float(value)
        if not math.isfinite(beat_value) or beat_value < 0:
            raise ValueError("time point beats must be non-negative and finite")
        return beat_value

    def locate(self, seconds: float) -> MusicalLocation:
        """Resolve a second offset into bar/beat information."""
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("seconds must be non-negative and finite")

        absolute_beats = self._seconds_to_beats(seconds)
        if absolute_beats < self.pickup_beats:
            return MusicalLocation(
                bar=0,
                beat=absolute_beats,
                absolute_beats=absolute_beats,
                seconds=seconds,
            )

        adjusted_beats = absolute_beats - self.pickup_beats
        bar = int(math.floor(adjusted_beats / self.beats_per_bar)) + 1
        beat_within_bar = adjusted_beats - ((bar - 1) * self.beats_per_bar)
        return MusicalLocation(
            bar=bar,
            beat=beat_within_bar,
            absolute_beats=absolute_beats,
            seconds=seconds,
        )

    def _beats_to_seconds(self, absolute_beats: float) -> float:
        if self.groove is None:
            return absolute_beats * self.seconds_per_beat
        step_size = self.groove.step_size_beats
        step_float = absolute_beats / step_size
        step_index = math.floor(step_float)
        phase = step_float - step_index
        o_i = self.groove.timing_offset_at(step_index)
        o_next = self.groove.timing_offset_at(step_index + 1)
        warped_steps = step_index + o_i + phase * (1.0 + o_next - o_i)
        return warped_steps * step_size * self.seconds_per_beat

    def _seconds_to_beats(self, seconds: float) -> float:
        warped_beats = seconds / self.seconds_per_beat
        if self.groove is None:
            return warped_beats
        step_size = self.groove.step_size_beats
        warped_steps = warped_beats / step_size
        step_index = math.floor(warped_steps)
        for candidate in (step_index, step_index - 1, step_index + 1):
            if candidate < 0:
                continue
            o_i = self.groove.timing_offset_at(candidate)
            o_next = self.groove.timing_offset_at(candidate + 1)
            warped_start = candidate + o_i
            warped_end = candidate + 1 + o_next
            if warped_start <= warped_steps < warped_end or (
                candidate == 0 and warped_steps < warped_end
            ):
                span = warped_end - warped_start
                phase = (warped_steps - warped_start) / span if span > 0 else 0.0
                return (candidate + phase) * step_size
        raise ValueError(
            f"groove warp inversion failed for {seconds:.6f}s "
            f"(warped_steps={warped_steps:.6f})"
        )


def _coerce_duration_beats(value: DurationLike) -> float:
    beats = value.beats if isinstance(value, (BeatSpan, BeatValue)) else float(value)

    if not math.isfinite(beats) or beats <= 0:
        raise ValueError("duration beats must be positive and finite")
    return beats


__all__ = [
    "B",
    "BeatSpan",
    "BeatValue",
    "DurationLike",
    "E",
    "Groove",
    "H",
    "M",
    "MeasurePosition",
    "MusicalLocation",
    "Q",
    "S",
    "Timeline",
    "TimePointLike",
    "W",
    "dotted",
    "triplet",
    "tuplet",
]
