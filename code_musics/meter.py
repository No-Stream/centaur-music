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
class SwingSpec:
    """Timing warp for a swung musical grid."""

    subdivision: Literal["eighth", "sixteenth"]
    offbeat_position: float

    def __post_init__(self) -> None:
        if self.subdivision not in {"eighth", "sixteenth"}:
            raise ValueError("subdivision must be 'eighth' or 'sixteenth'")
        if (
            not math.isfinite(self.offbeat_position)
            or self.offbeat_position < 0.5
            or self.offbeat_position >= 1.0
        ):
            raise ValueError(
                "offbeat_position must be finite and in the range [0.5, 1.0)"
            )

    @classmethod
    def eighths(cls, offbeat_position: float = 2.0 / 3.0) -> SwingSpec:
        """Return an eighth-note swing specification."""
        return cls(subdivision="eighth", offbeat_position=offbeat_position)

    @classmethod
    def sixteenths(cls, offbeat_position: float = 2.0 / 3.0) -> SwingSpec:
        """Return a sixteenth-note swing specification."""
        return cls(subdivision="sixteenth", offbeat_position=offbeat_position)


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


@dataclass(frozen=True)
class Timeline:
    """Single-tempo musical timeline for high-level authoring."""

    bpm: float
    meter: tuple[int, int] = (4, 4)
    pickup_beats: float = 0.0
    swing: SwingSpec | None = None

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

    def beats(self, value: DurationLike) -> float:
        """Resolve a beat-relative duration into seconds."""
        return _coerce_duration_beats(value) * self.seconds_per_beat

    def measures(self, value: float) -> float:
        """Resolve a measure span into seconds."""
        if not math.isfinite(value) or value < 0:
            raise ValueError("measure span must be non-negative and finite")
        return value * self.beats_per_bar * self.seconds_per_beat

    def duration(self, value: DurationLike) -> float:
        """Alias for converting a beat-relative duration into seconds."""
        return self.beats(value)

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

    @property
    def swing_unit_beats(self) -> float | None:
        """Return the beat span of one swing unit, if swing is enabled."""
        if self.swing is None:
            return None
        if self.swing.subdivision == "eighth":
            return 1.0
        return 0.5

    def _beats_to_seconds(self, absolute_beats: float) -> float:
        if self.swing is None:
            return absolute_beats * self.seconds_per_beat

        unit_beats = self.swing_unit_beats
        assert unit_beats is not None  # guaranteed: swing is not None here
        warped_beats = self._warp_beats_within_units(
            absolute_beats,
            unit_beats=unit_beats,
            offbeat_position=self.swing.offbeat_position,
        )
        return warped_beats * self.seconds_per_beat

    def _seconds_to_beats(self, seconds: float) -> float:
        warped_beats = seconds / self.seconds_per_beat
        if self.swing is None:
            return warped_beats

        unit_beats = self.swing_unit_beats
        assert unit_beats is not None  # guaranteed: swing is not None here
        return self._unwarp_beats_within_units(
            warped_beats,
            unit_beats=unit_beats,
            offbeat_position=self.swing.offbeat_position,
        )

    @staticmethod
    def _warp_beats_within_units(
        absolute_beats: float,
        *,
        unit_beats: float,
        offbeat_position: float,
    ) -> float:
        unit_index = math.floor(absolute_beats / unit_beats)
        phase = (absolute_beats - (unit_index * unit_beats)) / unit_beats

        if phase <= 0.5:
            warped_phase = phase * (offbeat_position / 0.5)
        else:
            warped_phase = offbeat_position + (
                ((phase - 0.5) / 0.5) * (1.0 - offbeat_position)
            )
        return (unit_index + warped_phase) * unit_beats

    @staticmethod
    def _unwarp_beats_within_units(
        warped_beats: float,
        *,
        unit_beats: float,
        offbeat_position: float,
    ) -> float:
        unit_index = math.floor(warped_beats / unit_beats)
        warped_phase = (warped_beats - (unit_index * unit_beats)) / unit_beats

        if warped_phase <= offbeat_position:
            phase = (
                0.0
                if offbeat_position == 0.0
                else warped_phase * (0.5 / offbeat_position)
            )
        else:
            phase = 0.5 + (
                ((warped_phase - offbeat_position) / (1.0 - offbeat_position)) * 0.5
            )
        return (unit_index + phase) * unit_beats


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
    "H",
    "M",
    "MeasurePosition",
    "MusicalLocation",
    "Q",
    "S",
    "SwingSpec",
    "Timeline",
    "TimePointLike",
    "W",
    "dotted",
    "triplet",
]
