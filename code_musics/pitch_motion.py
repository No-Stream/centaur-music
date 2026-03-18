"""Pitch-motion helpers for score-level note metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

PitchMotionKind = Literal["linear_bend", "ratio_glide", "vibrato"]


@dataclass(frozen=True)
class PitchMotionSpec:
    """Declarative pitch motion attached to a note event."""

    kind: PitchMotionKind
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def linear_bend(
        cls,
        *,
        target_freq: float | None = None,
        target_partial: float | None = None,
        target_ratio: float | None = None,
    ) -> PitchMotionSpec:
        """Create a bend toward a target pitch in absolute frequency space."""
        return cls(
            kind="linear_bend",
            params={
                "target_freq": target_freq,
                "target_partial": target_partial,
                "target_ratio": target_ratio,
            },
        )

    @classmethod
    def ratio_glide(
        cls,
        *,
        start_ratio: float = 1.0,
        end_ratio: float = 1.0,
    ) -> PitchMotionSpec:
        """Create a glide that interpolates in multiplicative ratio space."""
        return cls(
            kind="ratio_glide",
            params={
                "start_ratio": start_ratio,
                "end_ratio": end_ratio,
            },
        )

    @classmethod
    def vibrato(
        cls,
        *,
        depth_ratio: float = 0.01,
        rate_hz: float = 5.5,
        phase: float = 0.0,
    ) -> PitchMotionSpec:
        """Create a small deterministic vibrato around the base frequency."""
        return cls(
            kind="vibrato",
            params={
                "depth_ratio": depth_ratio,
                "rate_hz": rate_hz,
                "phase": phase,
            },
        )

    def validate(self) -> None:
        """Validate the motion spec eagerly."""
        if self.kind == "linear_bend":
            target_freq = self.params.get("target_freq")
            target_partial = self.params.get("target_partial")
            target_ratio = self.params.get("target_ratio")
            target_count = sum(
                value is not None
                for value in (target_freq, target_partial, target_ratio)
            )
            if target_count != 1:
                raise ValueError(
                    "linear_bend requires exactly one of target_freq, target_partial, "
                    "or target_ratio"
                )
            if target_freq is not None and float(target_freq) <= 0:
                raise ValueError("target_freq must be positive")
            if target_partial is not None and float(target_partial) <= 0:
                raise ValueError("target_partial must be positive")
            if target_ratio is not None and float(target_ratio) <= 0:
                raise ValueError("target_ratio must be positive")
            return

        if self.kind == "ratio_glide":
            start_ratio = float(self.params.get("start_ratio", 1.0))
            end_ratio = float(self.params.get("end_ratio", 1.0))
            if start_ratio <= 0 or end_ratio <= 0:
                raise ValueError("start_ratio and end_ratio must be positive")
            return

        if self.kind == "vibrato":
            depth_ratio = float(self.params.get("depth_ratio", 0.0))
            rate_hz = float(self.params.get("rate_hz", 0.0))
            if depth_ratio <= 0 or depth_ratio >= 0.25:
                raise ValueError("depth_ratio must be positive and less than 0.25")
            if rate_hz <= 0:
                raise ValueError("rate_hz must be positive")
            return

        raise ValueError(f"Unsupported pitch motion kind: {self.kind}")

    def target_frequency(self, score_f0: float) -> float:
        """Resolve the target frequency for a bend against the score root."""
        if self.kind != "linear_bend":
            raise ValueError("target_frequency is only valid for linear_bend motion")

        target_freq = self.params.get("target_freq")
        if target_freq is not None:
            return float(target_freq)

        target_partial = self.params.get("target_partial")
        if target_partial is not None:
            return score_f0 * float(target_partial)

        target_ratio = self.params.get("target_ratio")
        if target_ratio is not None:
            return score_f0 * float(target_ratio)

        raise ValueError("linear_bend requires a target pitch")


def build_frequency_trajectory(
    *,
    base_freq: float,
    duration: float,
    sample_rate: int,
    motion: PitchMotionSpec,
    score_f0: float,
) -> np.ndarray:
    """Build a strictly positive per-sample frequency trajectory."""
    if base_freq <= 0:
        raise ValueError("base_freq must be positive")
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    n_samples = int(duration * sample_rate)
    if n_samples == 0:
        return np.zeros(0, dtype=np.float64)

    if motion.kind == "linear_bend":
        target_freq = motion.target_frequency(score_f0=score_f0)
        trajectory = np.linspace(base_freq, target_freq, n_samples, endpoint=True)
    elif motion.kind == "ratio_glide":
        start_ratio = float(motion.params.get("start_ratio", 1.0))
        end_ratio = float(motion.params.get("end_ratio", 1.0))
        ratio_trajectory = np.exp(
            np.linspace(
                np.log(start_ratio), np.log(end_ratio), n_samples, endpoint=True
            )
        )
        trajectory = base_freq * ratio_trajectory
    elif motion.kind == "vibrato":
        depth_ratio = float(motion.params.get("depth_ratio", 0.0))
        rate_hz = float(motion.params.get("rate_hz", 0.0))
        phase = float(motion.params.get("phase", 0.0))
        time = np.arange(n_samples, dtype=np.float64) / sample_rate
        trajectory = base_freq * (
            1.0 + depth_ratio * np.sin(2.0 * np.pi * rate_hz * time + phase)
        )
    else:
        raise ValueError(f"Unsupported pitch motion kind: {motion.kind}")

    trajectory = np.asarray(trajectory, dtype=np.float64)
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("pitch motion must produce finite frequencies")
    if np.any(trajectory <= 0):
        raise ValueError("pitch motion must produce strictly positive frequencies")
    return trajectory


def phase_from_frequency_trajectory(
    freq_trajectory: np.ndarray,
    *,
    sample_rate: int,
) -> np.ndarray:
    """Integrate frequency samples into a phase trajectory starting at zero."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if freq_trajectory.ndim != 1:
        raise ValueError("freq_trajectory must be one-dimensional")

    if freq_trajectory.size == 0:
        return np.zeros(0, dtype=np.float64)

    increments = 2.0 * np.pi * freq_trajectory / sample_rate
    return np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(increments[:-1], dtype=np.float64)]
    )
