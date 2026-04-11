"""Score-level automation specs and render helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

AutomationTargetKind = Literal["synth", "pitch_ratio", "control"]
AutomationMode = Literal["replace", "add", "multiply"]
AutomationShape = Literal["hold", "linear", "exp", "sine_lfo"]

_SUPPORTED_SYNTH_AUTOMATION_PARAMS = {
    "attack",
    "brightness",
    "brightness_tilt",
    "cutoff_hz",
    "decay",
    "filter_drive",
    "filter_env_amount",
    "hammer_hardness",
    "hammer_noise",
    "mod_index",
    "release",
    "resonance_q",
    "soundboard_color",
    "sustain_level",
}

_SUPPORTED_CONTROL_AUTOMATION_PARAMS = {
    "mix",
    "mix_db",
    "pan",
    "pre_fx_gain_db",
    "return_db",
    "send_db",
    "wet",
    "wet_level",
}


@dataclass(frozen=True)
class AutomationTarget:
    """Declarative automation destination."""

    kind: AutomationTargetKind
    name: str

    def __post_init__(self) -> None:
        if self.kind == "pitch_ratio" and self.name != "pitch_ratio":
            raise ValueError(
                "pitch_ratio automation target must be named 'pitch_ratio'"
            )
        if self.kind == "synth" and self.name not in _SUPPORTED_SYNTH_AUTOMATION_PARAMS:
            raise ValueError(f"Unsupported synth automation target: {self.name!r}")
        if (
            self.kind == "control"
            and self.name not in _SUPPORTED_CONTROL_AUTOMATION_PARAMS
        ):
            raise ValueError(f"Unsupported control automation target: {self.name!r}")


@dataclass(frozen=True)
class AutomationSegment:
    """One time-bounded automation curve segment."""

    start: float
    end: float
    shape: AutomationShape
    start_value: float | None = None
    end_value: float | None = None
    value: float | None = None
    freq_hz: float | None = None
    phase: float = 0.0
    depth: float | None = None
    offset: float = 0.0

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("automation segment start must be non-negative")
        if self.end <= self.start:
            raise ValueError("automation segment end must be greater than start")

        if self.shape == "hold":
            if self.value is None:
                raise ValueError("hold automation segment requires value")
            return

        if self.shape == "linear":
            if self.start_value is None or self.end_value is None:
                raise ValueError(
                    "linear automation segment requires start_value and end_value"
                )
            return

        if self.shape == "exp":
            if self.start_value is None or self.end_value is None:
                raise ValueError(
                    "exp automation segment requires start_value and end_value"
                )
            if self.start_value <= 0 or self.end_value <= 0:
                raise ValueError("exp automation segment values must be positive")
            return

        if self.shape == "sine_lfo":
            if self.freq_hz is None or self.depth is None:
                raise ValueError(
                    "sine_lfo automation segment requires freq_hz and depth"
                )
            if self.freq_hz <= 0:
                raise ValueError("sine_lfo freq_hz must be positive")
            return

        raise ValueError(f"Unsupported automation segment shape: {self.shape!r}")


@dataclass(frozen=True)
class AutomationSpec:
    """Complete automation lane for a single target."""

    target: AutomationTarget
    segments: tuple[AutomationSegment, ...] = field(default_factory=tuple)
    default_value: float | None = None
    clamp_min: float | None = None
    clamp_max: float | None = None
    mode: AutomationMode = "replace"

    def __post_init__(self) -> None:
        if self.mode not in {"replace", "add", "multiply"}:
            raise ValueError(f"Unsupported automation mode: {self.mode!r}")
        if (
            self.clamp_min is not None
            and self.clamp_max is not None
            and self.clamp_min > self.clamp_max
        ):
            raise ValueError("automation clamp_min must be <= clamp_max")
        previous_end = 0.0
        for index, segment in enumerate(self.segments):
            if index == 0:
                previous_end = segment.end
                continue
            if segment.start < previous_end:
                raise ValueError(
                    "automation segments must be ordered and non-overlapping"
                )
            previous_end = segment.end

    def sample(self, time: float) -> float | None:
        value = self.default_value
        for segment in self.segments:
            if _segment_contains_time(segment, time):
                value = _sample_segment(segment, time)
                break
        if value is None:
            return None
        if self.clamp_min is not None:
            value = max(self.clamp_min, value)
        if self.clamp_max is not None:
            value = min(self.clamp_max, value)
        return float(value)

    def sample_many(self, times: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                self.apply_to_base(base_value=np.nan, time=float(time))
                for time in np.asarray(times, dtype=np.float64)
            ],
            dtype=np.float64,
        )

    def apply_to_base(self, *, base_value: float, time: float) -> float:
        sampled_value = self.sample(time)
        if sampled_value is None:
            return float(base_value)
        return _apply_mode(
            mode=self.mode,
            base_value=float(base_value),
            sampled_value=sampled_value,
        )


def build_pitch_ratio_trajectory(
    *,
    base_freq: float,
    duration: float,
    sample_rate: int,
    voice_automation: list[AutomationSpec],
    note_automation: list[AutomationSpec],
    note_start: float,
) -> np.ndarray | None:
    """Build a per-sample pitch trajectory from pitch_ratio automation lanes."""
    n_samples = int(duration * sample_rate)
    if n_samples == 0:
        return np.zeros(0, dtype=np.float64)

    pitch_specs = [
        spec
        for spec in [*voice_automation, *note_automation]
        if spec.target.kind == "pitch_ratio"
    ]
    if not pitch_specs:
        return None

    local_times = np.arange(n_samples, dtype=np.float64) / sample_rate
    absolute_times = note_start + local_times
    ratio = np.ones(n_samples, dtype=np.float64)

    for spec in voice_automation:
        if spec.target.kind != "pitch_ratio":
            continue
        ratio = np.asarray(
            [
                spec.apply_to_base(base_value=float(current), time=float(time))
                for current, time in zip(ratio, absolute_times, strict=True)
            ],
            dtype=np.float64,
        )

    for spec in note_automation:
        if spec.target.kind != "pitch_ratio":
            continue
        ratio = np.asarray(
            [
                spec.apply_to_base(base_value=float(current), time=float(time))
                for current, time in zip(ratio, local_times, strict=True)
            ],
            dtype=np.float64,
        )

    if np.any(ratio <= 0):
        raise ValueError("pitch_ratio automation must produce strictly positive ratios")
    return base_freq * ratio


def apply_synth_automation(
    *,
    params: dict[str, float | int | str],
    voice_automation: list[AutomationSpec],
    note_automation: list[AutomationSpec],
    note_start: float,
) -> dict[str, float | int | str]:
    """Return synth params after note-start automation has been applied."""
    automated_params = dict(params)

    for spec in voice_automation:
        if spec.target.kind != "synth":
            continue
        param_name = spec.target.name
        base_value = float(automated_params.get(param_name, 0.0))
        automated_params[param_name] = spec.apply_to_base(
            base_value=base_value,
            time=note_start,
        )

    for spec in note_automation:
        if spec.target.kind != "synth":
            continue
        param_name = spec.target.name
        base_value = float(automated_params.get(param_name, 0.0))
        automated_params[param_name] = spec.apply_to_base(
            base_value=base_value,
            time=0.0,
        )

    return automated_params


def has_pitch_ratio_automation(specs: list[AutomationSpec]) -> bool:
    """Return whether any lane targets pitch ratio."""
    return any(spec.target.kind == "pitch_ratio" for spec in specs)


def apply_control_automation(
    *,
    base_value: float,
    specs: list[AutomationSpec],
    target_name: str,
    times: np.ndarray,
) -> np.ndarray:
    """Return a per-sample control curve for one control target."""
    resolved_times = np.asarray(times, dtype=np.float64)
    if resolved_times.ndim != 1:
        raise ValueError("times must be a one-dimensional array")

    values = np.full(resolved_times.shape, float(base_value), dtype=np.float64)
    for spec in specs:
        if spec.target.kind != "control" or spec.target.name != target_name:
            continue
        values = np.asarray(
            [
                spec.apply_to_base(base_value=float(current), time=float(time))
                for current, time in zip(values, resolved_times, strict=True)
            ],
            dtype=np.float64,
        )
    return values


def _segment_contains_time(segment: AutomationSegment, time: float) -> bool:
    if segment.start <= time < segment.end:
        return True
    return bool(np.isclose(time, segment.end))


def _sample_segment(segment: AutomationSegment, time: float) -> float:
    if segment.shape == "hold":
        if segment.value is None:
            raise ValueError("hold automation segment requires value")
        return float(segment.value)

    progress = (time - segment.start) / (segment.end - segment.start)
    progress = float(np.clip(progress, 0.0, 1.0))

    if segment.shape == "linear":
        if segment.start_value is None or segment.end_value is None:
            raise ValueError(
                "linear automation segment requires start_value and end_value"
            )
        return float(
            segment.start_value + (progress * (segment.end_value - segment.start_value))
        )

    if segment.shape == "exp":
        if segment.start_value is None or segment.end_value is None:
            raise ValueError(
                "exp automation segment requires start_value and end_value"
            )
        return float(
            np.exp(
                np.log(segment.start_value)
                + (progress * (np.log(segment.end_value) - np.log(segment.start_value)))
            )
        )

    if segment.shape == "sine_lfo":
        if segment.freq_hz is None or segment.depth is None:
            raise ValueError("sine_lfo automation segment requires freq_hz and depth")
        local_time = time - segment.start
        return float(
            segment.offset
            + segment.depth
            * np.sin((2.0 * np.pi * segment.freq_hz * local_time) + segment.phase)
        )

    raise ValueError(f"Unsupported automation segment shape: {segment.shape!r}")


def _apply_mode(
    *, mode: AutomationMode, base_value: float, sampled_value: float
) -> float:
    if mode == "replace":
        return float(sampled_value)
    if mode == "add":
        return float(base_value + sampled_value)
    if mode == "multiply":
        return float(base_value * sampled_value)
    raise ValueError(f"Unsupported automation mode: {mode!r}")
