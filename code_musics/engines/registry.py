"""Engine registry and preset resolution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from code_musics.engines import additive, filtered_stack, fm, noise_perc, polyblep

EngineRenderer = Callable[..., np.ndarray]

_ENGINES: dict[str, EngineRenderer] = {
    "additive": additive.render,
    "fm": fm.render,
    "filtered_stack": filtered_stack.render,
    "noise_perc": noise_perc.render,
    "polyblep": polyblep.render,
}

_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "additive": {
        "soft_pad": {
            "n_harmonics": 8,
            "harmonic_rolloff": 0.62,
            "brightness_tilt": -0.15,
            "attack": 0.45,
            "release": 1.2,
            "unison_voices": 2,
            "detune_cents": 3.0,
        },
        "drone": {
            "n_harmonics": 10,
            "harmonic_rolloff": 0.52,
            "brightness_tilt": -0.05,
            "attack": 1.2,
            "release": 3.0,
        },
        "bright_pluck": {
            "n_harmonics": 7,
            "harmonic_rolloff": 0.4,
            "brightness_tilt": 0.2,
            "attack": 0.01,
            "decay": 0.12,
            "sustain_level": 0.45,
            "release": 0.22,
        },
    },
    "fm": {
        "bell": {
            "carrier_ratio": 1.0,
            "mod_ratio": 2.0,
            "mod_index": 4.5,
            "index_decay": 0.25,
            "index_sustain": 0.12,
            "attack": 0.01,
            "decay": 0.18,
            "sustain_level": 0.22,
            "release": 0.8,
        },
        "glass_lead": {
            "carrier_ratio": 1.0,
            "mod_ratio": 7 / 4,
            "mod_index": 2.2,
            "index_decay": 0.08,
            "index_sustain": 0.55,
            "attack": 0.02,
            "release": 0.35,
        },
        "metal_bass": {
            "carrier_ratio": 0.5,
            "mod_ratio": 1.5,
            "mod_index": 3.0,
            "feedback": 0.18,
            "index_decay": 0.15,
            "index_sustain": 0.3,
            "attack": 0.01,
            "release": 0.45,
        },
    },
    "filtered_stack": {
        "warm_pad": {
            "waveform": "saw",
            "n_harmonics": 14,
            "cutoff_hz": 900.0,
            "keytrack": 0.15,
            "filter_env_amount": 0.8,
            "filter_env_decay": 0.35,
            "attack": 0.4,
            "release": 1.4,
        },
        "reed_lead": {
            "waveform": "square",
            "n_harmonics": 10,
            "cutoff_hz": 2_200.0,
            "keytrack": 0.2,
            "resonance": 0.18,
            "filter_env_amount": 0.5,
            "filter_env_decay": 0.18,
            "attack": 0.02,
            "release": 0.25,
        },
        "round_bass": {
            "waveform": "triangle",
            "n_harmonics": 9,
            "cutoff_hz": 450.0,
            "keytrack": 0.1,
            "resonance": 0.08,
            "attack": 0.01,
            "release": 0.3,
        },
    },
    "noise_perc": {
        "kickish": {
            "noise_mix": 0.18,
            "pitch_decay": 0.06,
            "tone_decay": 0.2,
            "bandpass_ratio": 0.8,
            "click_amount": 0.04,
        },
        "snareish": {
            "noise_mix": 0.75,
            "pitch_decay": 0.03,
            "tone_decay": 0.11,
            "bandpass_ratio": 1.6,
            "click_amount": 0.12,
        },
        "tick": {
            "noise_mix": 0.88,
            "pitch_decay": 0.02,
            "tone_decay": 0.05,
            "bandpass_ratio": 2.4,
            "click_amount": 0.28,
        },
    },
    "polyblep": {
        "warm_lead": {
            "waveform": "saw",
            "cutoff_hz": 3000.0,
            "resonance": 0.08,
            "filter_env_amount": 0.45,
            "filter_env_decay": 0.90,
            "keytrack": 0.05,
        },
    },
}


def register_engine(name: str, renderer: EngineRenderer) -> None:
    """Register a renderer under the given engine name."""
    _ENGINES[name] = renderer


def register_presets(engine_name: str, presets: dict[str, dict[str, Any]]) -> None:
    """Register presets for a specific engine."""
    engine_presets = _PRESETS.setdefault(engine_name, {})
    engine_presets.update(presets)


def resolve_synth_params(params: dict[str, Any]) -> dict[str, Any]:
    """Resolve preset-backed synth params into a single parameter dictionary."""
    resolved = dict(params)
    engine_name = str(resolved.get("engine", "additive"))
    preset_name = resolved.pop("preset", None)
    if preset_name is None:
        resolved["engine"] = engine_name
        return resolved

    engine_presets = _PRESETS.get(engine_name, {})
    if preset_name not in engine_presets:
        raise ValueError(f"Unknown preset {preset_name!r} for engine {engine_name!r}")

    merged = dict(engine_presets[preset_name])
    merged.update(resolved)
    merged["engine"] = engine_name
    return merged


def render_note_signal(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
    freq_trajectory: np.ndarray | None = None,
) -> np.ndarray:
    """Render a note through the requested engine."""
    resolved = resolve_synth_params(params)
    engine_name = str(resolved.get("engine", "additive"))
    if engine_name not in _ENGINES:
        raise ValueError(f"Unsupported synth engine: {engine_name}")
    if freq_trajectory is not None and engine_name == "noise_perc":
        raise ValueError("pitch motion is not supported for the noise_perc engine")

    renderer_kwargs: dict[str, Any] = {
        "freq": freq,
        "duration": duration,
        "amp": amp,
        "sample_rate": sample_rate,
        "params": resolved,
    }
    if freq_trajectory is not None:
        renderer_kwargs["freq_trajectory"] = freq_trajectory

    return _ENGINES[engine_name](**renderer_kwargs)
