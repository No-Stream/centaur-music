"""Score-level automation specs and render helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

AutomationTargetKind = Literal["synth", "pitch_ratio", "control"]
AutomationMode = Literal["replace", "add", "multiply"]
AutomationShape = Literal["hold", "linear", "exp", "sine_lfo"]

_SUPPORTED_SYNTH_AUTOMATION_PARAMS = {
    "analog_jitter",
    "attack",
    "attack_brightness",
    "attack_power",
    "attack_target",
    "bass_compensation",
    "body_distortion_drive",
    "body_fm_index",
    "brightness",
    "brightness_tilt",
    "click_amount",
    "cutoff_drift",
    "cutoff_hz",
    "decay",
    "decay_power",
    "drift",
    "feedback",
    "feedback_amount",
    "feedback_saturation",
    "filter_drive",
    "filter_env_amount",
    "filter_env_decay",
    "filter_even_harmonics",
    "filter_morph",
    "hammer_hardness",
    "hammer_noise",
    "hpf_cutoff_hz",
    "index_decay",
    "leakage",
    "mod_index",
    "morph_time",
    "noise_amount",
    "noise_floor",
    "osc2_detune_cents",
    "osc2_level",
    "osc_asymmetry",
    "osc_dc_offset",
    "osc_shape_drift",
    "osc_softness",
    "overtone_amount",
    "pitch_drift",
    "pluck_hardness",
    "pluck_noise",
    "release",
    "release_power",
    "resonance_q",
    "soundboard_brightness",
    "soundboard_color",
    "sustain_level",
    "vca_nonlinearity",
    "vibrato_chorus",
    "vibrato_depth",
    "voice_card_spread",
    "voice_card_pitch_spread",
    "voice_card_filter_spread",
    "voice_card_envelope_spread",
    "voice_card_osc_spread",
    "voice_card_level_spread",
    # va engine: supersaw / spectralwave / drive / comb params
    "supersaw_detune",
    "supersaw_mix",
    "spectral_position",
    "spectral_morph_amount",
    "drive_amount",
    "comb_feedback",
    "comb_damping",
    "comb_keytrack",
    "comb_mix",
    "comb_delay_ms",
    "filter1_cutoff_hz",
    "filter1_resonance_q",
    "filter2_cutoff_hz",
    "filter2_resonance_q",
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
    phase_rad: float = 0.0
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

    def sample_many_raw(self, times: np.ndarray) -> np.ndarray:
        """Vectorized sampling — returns raw values without mode application.

        Uses np.searchsorted on ordered, non-overlapping segments for O(n log m)
        total instead of O(n * m).  Returns default_value for times outside any
        segment, or NaN if default_value is None.
        """
        return _sample_segments_vectorized(
            self.segments,
            times,
            self.default_value,
            self.clamp_min,
            self.clamp_max,
        )

    def sample_many(self, times: np.ndarray) -> np.ndarray:
        """Vectorized sampling with mode applied against NaN base (replace-only legacy path)."""
        return self.sample_many_raw(times)

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
        sampled = spec.sample_many_raw(absolute_times)
        ratio = _apply_mode_vectorized(spec.mode, ratio, sampled)

    for spec in note_automation:
        if spec.target.kind != "pitch_ratio":
            continue
        sampled = spec.sample_many_raw(local_times)
        ratio = _apply_mode_vectorized(spec.mode, ratio, sampled)

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
        sampled = spec.sample_many_raw(resolved_times)
        values = _apply_mode_vectorized(spec.mode, values, sampled)
    return values


def apply_mode_vectorized(
    mode: AutomationMode,
    base: np.ndarray,
    contribution: np.ndarray,
    *,
    nan_mask_contribution: bool = False,
) -> np.ndarray:
    """Apply a combine mode to a contribution against a base curve.

    ``nan_mask_contribution=True`` preserves the automation-segment
    semantics where NaN in ``contribution`` means "outside any segment —
    keep base value."  With the default ``False``, every sample of
    ``contribution`` is treated as defined (modulation-matrix semantics).
    """
    if not nan_mask_contribution:
        if mode == "replace":
            return contribution.astype(np.float64, copy=True)
        if mode == "add":
            return base + contribution
        if mode == "multiply":
            return base * contribution
        raise ValueError(f"Unsupported combine mode: {mode!r}")

    valid = ~np.isnan(contribution)
    result = base.copy()
    if mode == "replace":
        result[valid] = contribution[valid]
    elif mode == "add":
        result[valid] = base[valid] + contribution[valid]
    elif mode == "multiply":
        result[valid] = base[valid] * contribution[valid]
    else:
        raise ValueError(f"Unsupported combine mode: {mode!r}")
    return result


def _apply_mode_vectorized(
    mode: AutomationMode, base: np.ndarray, sampled: np.ndarray
) -> np.ndarray:
    return apply_mode_vectorized(mode, base, sampled, nan_mask_contribution=True)


def _sample_segments_vectorized(
    segments: tuple[AutomationSegment, ...],
    times: np.ndarray,
    default_value: float | None,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
) -> np.ndarray:
    """Evaluate ordered, non-overlapping segments for an array of times.

    Uses np.searchsorted for O(n log m) total instead of O(n * m).
    """
    times = np.asarray(times, dtype=np.float64)
    n = len(times)
    default = default_value if default_value is not None else np.nan
    result = np.full(n, default, dtype=np.float64)

    if not segments:
        return result

    starts = np.array([s.start for s in segments], dtype=np.float64)
    ends = np.array([s.end for s in segments], dtype=np.float64)

    # For each time, find the segment whose start is <= time.
    # searchsorted('right') on starts gives the index of the first start > time,
    # so idx-1 is the candidate segment.
    indices = np.searchsorted(starts, times, side="right") - 1

    for seg_idx in range(len(segments)):
        seg = segments[seg_idx]
        mask = (
            (indices == seg_idx) & (times >= starts[seg_idx]) & (times <= ends[seg_idx])
        )
        if not np.any(mask):
            continue

        seg_times = times[mask]

        if seg.shape == "hold":
            result[mask] = float(seg.value)  # type: ignore[arg-type]
        elif seg.shape == "linear":
            progress = (seg_times - seg.start) / (seg.end - seg.start)
            np.clip(progress, 0.0, 1.0, out=progress)
            result[mask] = seg.start_value + progress * (  # type: ignore[reportOptionalOperand]
                seg.end_value - seg.start_value
            )  # type: ignore[operator]
        elif seg.shape == "exp":
            progress = (seg_times - seg.start) / (seg.end - seg.start)
            np.clip(progress, 0.0, 1.0, out=progress)
            log_start = np.log(seg.start_value)  # type: ignore[arg-type]
            log_end = np.log(seg.end_value)  # type: ignore[arg-type]
            result[mask] = np.exp(log_start + progress * (log_end - log_start))
        elif seg.shape == "sine_lfo":
            local_t = seg_times - seg.start
            result[mask] = seg.offset + seg.depth * np.sin(  # type: ignore[operator]
                2.0 * np.pi * seg.freq_hz * local_t + seg.phase_rad  # type: ignore[operator]
            )

    if clamp_min is not None:
        np.maximum(result, clamp_min, out=result)
    if clamp_max is not None:
        np.minimum(result, clamp_max, out=result)

    return result


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
            * np.sin((2.0 * np.pi * segment.freq_hz * local_time) + segment.phase_rad)
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
