"""Humanization and drift helpers for render-time performance variation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

_SUPPORTED_DRIFT_STYLES = {"random_walk", "smooth_noise", "lfo", "sample_hold"}

_TIMING_PRESETS: dict[str, dict[str, Any]] = {
    "tight_ensemble": {
        "ensemble_drift": {"style": "random_walk", "rate_hz": 0.05, "smoothness": 0.82},
        "ensemble_amount_ms": 10.0,
        "follow_strength": 0.96,
        "voice_spread_ms": 1.8,
        "micro_jitter_ms": 0.35,
        "chord_spread_ms": 1.0,
    },
    "chamber": {
        "ensemble_drift": {
            "style": "random_walk",
            "rate_hz": 0.035,
            "smoothness": 0.84,
        },
        "ensemble_amount_ms": 18.0,
        "follow_strength": 0.92,
        "voice_spread_ms": 3.0,
        "micro_jitter_ms": 0.7,
        "chord_spread_ms": 2.2,
    },
    "relaxed_band": {
        "ensemble_drift": {
            "style": "random_walk",
            "rate_hz": 0.028,
            "smoothness": 0.88,
        },
        "ensemble_amount_ms": 26.0,
        "follow_strength": 0.88,
        "voice_spread_ms": 4.8,
        "micro_jitter_ms": 1.0,
        "chord_spread_ms": 3.6,
    },
    "loose_late_night": {
        "ensemble_drift": {"style": "random_walk", "rate_hz": 0.022, "smoothness": 0.9},
        "ensemble_amount_ms": 38.0,
        "follow_strength": 0.82,
        "voice_spread_ms": 7.5,
        "micro_jitter_ms": 1.6,
        "chord_spread_ms": 5.0,
    },
}

_ENVELOPE_PRESETS: dict[str, dict[str, Any]] = {
    "subtle_analog": {
        "drift": {"style": "smooth_noise", "rate_hz": 0.09, "smoothness": 0.75},
        "attack_amount_pct": 0.08,
        "decay_amount_pct": 0.06,
        "sustain_amount_pct": 0.04,
        "release_amount_pct": 0.08,
    },
    "breathing_pad": {
        "drift": {"style": "smooth_noise", "rate_hz": 0.06, "smoothness": 0.85},
        "attack_amount_pct": 0.16,
        "decay_amount_pct": 0.12,
        "sustain_amount_pct": 0.06,
        "release_amount_pct": 0.18,
    },
    "loose_pluck": {
        "drift": {"style": "smooth_noise", "rate_hz": 0.12, "smoothness": 0.62},
        "attack_amount_pct": 0.12,
        "decay_amount_pct": 0.14,
        "sustain_amount_pct": 0.05,
        "release_amount_pct": 0.12,
    },
}


@dataclass(frozen=True)
class DriftSpec:
    """Reusable drift generator definition."""

    style: str = "random_walk"
    rate_hz: float = 0.035
    smoothness: float = 0.84
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.style not in _SUPPORTED_DRIFT_STYLES:
            raise ValueError(f"Unsupported drift style: {self.style}")
        if self.rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        if not 0.0 <= self.smoothness <= 1.0:
            raise ValueError("smoothness must be between 0 and 1")


@dataclass(frozen=True)
class TimingTarget:
    """Resolved note timing target used during render-time humanization."""

    key: tuple[str, int]
    voice_name: str
    start: float


@dataclass(frozen=True)
class TimingHumanizeSpec:
    """High-level ensemble timing humanization."""

    preset: str | None = None
    ensemble_drift: DriftSpec | None = None
    ensemble_amount_ms: float | None = None
    follow_strength: float | None = None
    voice_spread_ms: float | None = None
    micro_jitter_ms: float | None = None
    chord_spread_ms: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        preset_name = self.preset or "chamber"
        if preset_name not in _TIMING_PRESETS:
            raise ValueError(f"Unknown timing humanize preset: {preset_name!r}")
        preset = _TIMING_PRESETS[preset_name]

        drift_value = self.ensemble_drift or DriftSpec(**preset["ensemble_drift"])
        ensemble_amount_ms = float(
            preset["ensemble_amount_ms"]
            if self.ensemble_amount_ms is None
            else self.ensemble_amount_ms
        )
        follow_strength = float(
            preset["follow_strength"]
            if self.follow_strength is None
            else self.follow_strength
        )
        voice_spread_ms = float(
            preset["voice_spread_ms"]
            if self.voice_spread_ms is None
            else self.voice_spread_ms
        )
        micro_jitter_ms = float(
            preset["micro_jitter_ms"]
            if self.micro_jitter_ms is None
            else self.micro_jitter_ms
        )
        chord_spread_ms = float(
            preset["chord_spread_ms"]
            if self.chord_spread_ms is None
            else self.chord_spread_ms
        )

        object.__setattr__(self, "preset", preset_name)
        object.__setattr__(self, "ensemble_drift", drift_value)
        object.__setattr__(self, "ensemble_amount_ms", ensemble_amount_ms)
        object.__setattr__(self, "follow_strength", follow_strength)
        object.__setattr__(self, "voice_spread_ms", voice_spread_ms)
        object.__setattr__(self, "micro_jitter_ms", micro_jitter_ms)
        object.__setattr__(self, "chord_spread_ms", chord_spread_ms)

        if ensemble_amount_ms < 0:
            raise ValueError("ensemble_amount_ms must be non-negative")
        if not 0.0 <= follow_strength <= 1.0:
            raise ValueError("follow_strength must be between 0 and 1")
        if voice_spread_ms < 0:
            raise ValueError("voice_spread_ms must be non-negative")
        if micro_jitter_ms < 0:
            raise ValueError("micro_jitter_ms must be non-negative")
        if chord_spread_ms < 0:
            raise ValueError("chord_spread_ms must be non-negative")


@dataclass(frozen=True)
class EnvelopeHumanizeSpec:
    """Smooth ADSR variation over score time."""

    preset: str | None = None
    drift: DriftSpec | None = None
    attack_amount_pct: float | None = None
    decay_amount_pct: float | None = None
    sustain_amount_pct: float | None = None
    release_amount_pct: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        preset_name = self.preset or "subtle_analog"
        if preset_name not in _ENVELOPE_PRESETS:
            raise ValueError(f"Unknown envelope humanize preset: {preset_name!r}")
        preset = _ENVELOPE_PRESETS[preset_name]

        drift_value = self.drift or DriftSpec(**preset["drift"])
        attack_amount_pct = float(
            preset["attack_amount_pct"]
            if self.attack_amount_pct is None
            else self.attack_amount_pct
        )
        decay_amount_pct = float(
            preset["decay_amount_pct"]
            if self.decay_amount_pct is None
            else self.decay_amount_pct
        )
        sustain_amount_pct = float(
            preset["sustain_amount_pct"]
            if self.sustain_amount_pct is None
            else self.sustain_amount_pct
        )
        release_amount_pct = float(
            preset["release_amount_pct"]
            if self.release_amount_pct is None
            else self.release_amount_pct
        )

        object.__setattr__(self, "preset", preset_name)
        object.__setattr__(self, "drift", drift_value)
        object.__setattr__(self, "attack_amount_pct", attack_amount_pct)
        object.__setattr__(self, "decay_amount_pct", decay_amount_pct)
        object.__setattr__(self, "sustain_amount_pct", sustain_amount_pct)
        object.__setattr__(self, "release_amount_pct", release_amount_pct)

        if attack_amount_pct < 0:
            raise ValueError("attack_amount_pct must be non-negative")
        if decay_amount_pct < 0:
            raise ValueError("decay_amount_pct must be non-negative")
        if sustain_amount_pct < 0:
            raise ValueError("sustain_amount_pct must be non-negative")
        if release_amount_pct < 0:
            raise ValueError("release_amount_pct must be non-negative")


def build_timing_offsets(
    *,
    targets: list[TimingTarget],
    humanize: TimingHumanizeSpec | None,
    total_dur: float,
) -> dict[tuple[str, int], float]:
    """Return render-time start offsets in seconds for each note target."""
    if humanize is None or not targets:
        return {}

    global_seed = _seed_or_default(humanize.seed, "timing", humanize.preset)
    ensemble_drift = humanize.ensemble_drift
    if ensemble_drift is None:
        raise ValueError("ensemble_drift must be resolved")
    ensemble_amount_ms = humanize.ensemble_amount_ms
    if ensemble_amount_ms is None:
        raise ValueError("ensemble_amount_ms must be resolved")
    follow_strength = humanize.follow_strength
    if follow_strength is None:
        raise ValueError("follow_strength must be resolved")
    voice_spread_ms = humanize.voice_spread_ms
    if voice_spread_ms is None:
        raise ValueError("voice_spread_ms must be resolved")
    micro_jitter_limit_ms = humanize.micro_jitter_ms
    if micro_jitter_limit_ms is None:
        raise ValueError("micro_jitter_ms must be resolved")
    chord_spread_ms = humanize.chord_spread_ms
    if chord_spread_ms is None:
        raise ValueError("chord_spread_ms must be resolved")

    start_times = np.asarray([target.start for target in targets], dtype=np.float64)
    global_curve = _sample_drift_curve(
        ensemble_drift,
        times=start_times,
        total_dur=total_dur,
        seed=global_seed,
    )
    global_offsets_ms = ensemble_amount_ms * global_curve

    per_voice_static_ms: dict[str, float] = {}
    per_voice_dynamic: dict[str, np.ndarray] = {}
    for voice_name in {target.voice_name for target in targets}:
        voice_seed = _stable_seed(global_seed, "voice", voice_name)
        voice_rng = np.random.default_rng(voice_seed)
        per_voice_static_ms[voice_name] = float(
            voice_rng.uniform(-voice_spread_ms, voice_spread_ms)
        )
        per_voice_dynamic[voice_name] = (
            voice_spread_ms
            * (1.0 - follow_strength)
            * (
                _sample_drift_curve(
                    DriftSpec(
                        style="smooth_noise",
                        rate_hz=max(ensemble_drift.rate_hz * 1.4, 0.01),
                        smoothness=min(1.0, ensemble_drift.smoothness + 0.05),
                        seed=None,
                    ),
                    times=start_times,
                    total_dur=total_dur,
                    seed=_stable_seed(voice_seed, "dynamic"),
                )
            )
        )

    chord_offsets_ms = _build_chord_spread_offsets(
        targets=targets,
        chord_spread_ms=chord_spread_ms,
        seed=_stable_seed(global_seed, "chords"),
    )

    offsets: dict[tuple[str, int], float] = {}
    for index, target in enumerate(targets):
        note_seed = _stable_seed(global_seed, "note", target.voice_name, target.key[1])
        note_rng = np.random.default_rng(note_seed)
        note_jitter_ms = 0.0
        if micro_jitter_limit_ms > 0:
            note_jitter_ms = float(
                np.clip(
                    note_rng.normal(loc=0.0, scale=micro_jitter_limit_ms / 2.5),
                    -micro_jitter_limit_ms,
                    micro_jitter_limit_ms,
                )
            )

        offset_ms = (
            global_offsets_ms[index]
            + per_voice_static_ms[target.voice_name]
            + per_voice_dynamic[target.voice_name][index]
            + chord_offsets_ms[target.key]
            + note_jitter_ms
        )
        offsets[target.key] = float(offset_ms / 1_000.0)

    return offsets


def resolve_envelope_params(
    *,
    base_attack: float,
    base_decay: float,
    base_sustain_level: float,
    base_release: float,
    note_start: float,
    humanize: EnvelopeHumanizeSpec | None,
    total_dur: float,
    voice_name: str,
) -> tuple[float, float, float, float]:
    """Return ADSR parameters after smooth drift is applied."""
    if humanize is None:
        return base_attack, base_decay, base_sustain_level, base_release

    drift = humanize.drift
    if drift is None:
        raise ValueError("drift must be resolved")
    attack_amount_pct = humanize.attack_amount_pct
    if attack_amount_pct is None:
        raise ValueError("attack_amount_pct must be resolved")
    decay_amount_pct = humanize.decay_amount_pct
    if decay_amount_pct is None:
        raise ValueError("decay_amount_pct must be resolved")
    sustain_amount_pct = humanize.sustain_amount_pct
    if sustain_amount_pct is None:
        raise ValueError("sustain_amount_pct must be resolved")
    release_amount_pct = humanize.release_amount_pct
    if release_amount_pct is None:
        raise ValueError("release_amount_pct must be resolved")

    shared_seed = _seed_or_default(
        humanize.seed, "envelope", humanize.preset, voice_name
    )
    shared_curve = _sample_drift_curve(
        drift,
        times=np.asarray([note_start], dtype=np.float64),
        total_dur=total_dur,
        seed=shared_seed,
    )[0]

    def _parameter_curve(parameter_name: str) -> float:
        local_curve = _sample_drift_curve(
            drift,
            times=np.asarray([note_start], dtype=np.float64),
            total_dur=total_dur,
            seed=_stable_seed(shared_seed, parameter_name),
        )[0]
        return float((0.6 * shared_curve) + (0.4 * local_curve))

    attack = max(
        0.0,
        base_attack * (1.0 + (attack_amount_pct * _parameter_curve("attack"))),
    )
    decay = max(
        0.0,
        base_decay * (1.0 + (decay_amount_pct * _parameter_curve("decay"))),
    )
    sustain_level = float(
        np.clip(
            base_sustain_level
            * (1.0 + (sustain_amount_pct * _parameter_curve("sustain"))),
            0.0,
            1.0,
        )
    )
    release = max(
        0.0,
        base_release * (1.0 + (release_amount_pct * _parameter_curve("release"))),
    )
    return attack, decay, sustain_level, release


def _build_chord_spread_offsets(
    *,
    targets: list[TimingTarget],
    chord_spread_ms: float,
    seed: int,
) -> dict[tuple[str, int], float]:
    if chord_spread_ms <= 0:
        return {target.key: 0.0 for target in targets}

    grouped_targets: dict[float, list[TimingTarget]] = {}
    for target in targets:
        grouped_targets.setdefault(target.start, []).append(target)

    offsets: dict[tuple[str, int], float] = {}
    for start_time, group in grouped_targets.items():
        if len(group) == 1:
            offsets[group[0].key] = 0.0
            continue

        spread_values = np.linspace(
            -chord_spread_ms / 2.0, chord_spread_ms / 2.0, len(group)
        )
        ordering = sorted(
            group,
            key=lambda target: _stable_seed(
                seed, "group", start_time, target.voice_name, target.key[1]
            ),
        )
        for target, spread in zip(ordering, spread_values, strict=True):
            offsets[target.key] = float(spread)

    return offsets


def _sample_drift_curve(
    spec: DriftSpec,
    *,
    times: np.ndarray,
    total_dur: float,
    seed: int,
) -> np.ndarray:
    if times.size == 0:
        return np.zeros(0, dtype=np.float64)
    if total_dur <= 0:
        return np.zeros_like(times, dtype=np.float64)

    bounded_times = np.clip(np.asarray(times, dtype=np.float64), 0.0, total_dur)
    rng = np.random.default_rng(
        _stable_seed(seed, spec.seed if spec.seed is not None else 0)
    )

    if spec.style == "lfo":
        phase = rng.uniform(0.0, 2.0 * np.pi)
        curve = np.sin((2.0 * np.pi * spec.rate_hz * bounded_times) + phase)
        return _normalize_curve(curve)

    anchor_step = min(total_dur, max(1.0 / spec.rate_hz, total_dur / 64.0))
    anchor_times = np.arange(
        0.0, total_dur + anchor_step, anchor_step, dtype=np.float64
    )
    if anchor_times.size < 2:
        anchor_times = np.array([0.0, total_dur], dtype=np.float64)

    if spec.style == "random_walk":
        increments = rng.normal(loc=0.0, scale=1.0, size=anchor_times.size)
        anchor_values = np.cumsum(increments)
    else:
        anchor_values = rng.normal(loc=0.0, scale=1.0, size=anchor_times.size)

    anchor_values = _smooth_anchor_values(anchor_values, spec.smoothness)

    if spec.style == "sample_hold":
        indices = np.searchsorted(anchor_times, bounded_times, side="right") - 1
        indices = np.clip(indices, 0, anchor_times.size - 1)
        return _normalize_curve(anchor_values[indices])

    curve = np.interp(bounded_times, anchor_times, anchor_values)
    return _normalize_curve(curve)


def _smooth_anchor_values(values: np.ndarray, smoothness: float) -> np.ndarray:
    window = max(1, min(len(values), int(round(1.0 + (smoothness * 8.0)))))
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def _normalize_curve(curve: np.ndarray) -> np.ndarray:
    max_abs = float(np.max(np.abs(curve)))
    if max_abs <= 1e-12:
        return np.zeros_like(curve, dtype=np.float64)
    return np.asarray(curve / max_abs, dtype=np.float64)


def _seed_or_default(seed: int | None, *parts: object) -> int:
    if seed is not None:
        return int(seed)
    return _stable_seed(*parts)


def _stable_seed(*parts: object) -> int:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)
