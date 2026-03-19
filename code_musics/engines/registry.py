"""Engine registry, synth-spec normalization, and preset resolution."""

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

_ENV_ALIAS_TO_CANONICAL: dict[str, str] = {
    "attack_ms": "attack",
    "attack": "attack",
    "decay_ms": "decay",
    "decay": "decay",
    "sustain_ratio": "sustain_level",
    "sustain_level": "sustain_level",
    "release_ms": "release",
    "release": "release",
    "attack_scale": "attack_scale",
    "release_scale": "release_scale",
}

_SECONDS_FROM_MS_KEYS = {
    "attack_ms",
    "decay_ms",
    "release_ms",
}

_PARAM_ALIAS_TO_CANONICAL: dict[str, str] = {
    "filter_env_depth_ratio": "filter_env_amount",
    "filter_env_amount": "filter_env_amount",
    "filter_env_decay_ms": "filter_env_decay",
    "filter_env_decay": "filter_env_decay",
    "index_decay_ms": "index_decay",
    "index_decay": "index_decay",
    "index_sustain_ratio": "index_sustain",
    "index_sustain": "index_sustain",
    "pitch_decay_ms": "pitch_decay",
    "pitch_decay": "pitch_decay",
    "tone_decay_ms": "tone_decay",
    "tone_decay": "tone_decay",
    "noise_mix_ratio": "noise_mix",
    "noise_mix": "noise_mix",
    "feedback_ratio": "feedback",
    "feedback": "feedback",
    "resonance_ratio": "resonance",
    "resonance": "resonance",
    "filter_drive_ratio": "filter_drive",
    "filter_drive": "filter_drive",
}

_SECONDS_FROM_MS_PARAM_KEYS = {
    "filter_env_decay_ms",
    "index_decay_ms",
    "pitch_decay_ms",
    "tone_decay_ms",
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
        "organ": {
            "n_harmonics": 9,
            "harmonic_rolloff": 0.82,
            "brightness_tilt": -0.05,
            "odd_even_balance": -0.1,
            "attack": 0.01,
            "decay": 0.08,
            "sustain_level": 0.92,
            "release": 0.12,
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
        "dx_piano": {
            "carrier_ratio": 1.0,
            "mod_ratio": 3.0,
            "mod_index": 3.4,
            "feedback": 0.06,
            "index_decay": 0.22,
            "index_sustain": 0.16,
            "attack": 0.008,
            "decay": 0.28,
            "sustain_level": 0.2,
            "release": 0.75,
        },
        "lately_bass": {
            "carrier_ratio": 0.5,
            "mod_ratio": 1.0,
            "mod_index": 2.8,
            "feedback": 0.3,
            "index_decay": 0.1,
            "index_sustain": 0.42,
            "attack": 0.005,
            "decay": 0.16,
            "sustain_level": 0.58,
            "release": 0.18,
        },
        "fm_clav": {
            "carrier_ratio": 1.0,
            "mod_ratio": 4.0,
            "mod_index": 2.6,
            "feedback": 0.04,
            "index_decay": 0.06,
            "index_sustain": 0.08,
            "attack": 0.002,
            "decay": 0.14,
            "sustain_level": 0.08,
            "release": 0.12,
        },
        "fm_mallet": {
            "carrier_ratio": 1.0,
            "mod_ratio": 3.5,
            "mod_index": 4.2,
            "feedback": 0.02,
            "index_decay": 0.09,
            "index_sustain": 0.06,
            "attack": 0.003,
            "decay": 0.22,
            "sustain_level": 0.1,
            "release": 0.36,
        },
        "chorused_ep": {
            "carrier_ratio": 1.0,
            "mod_ratio": 2.0,
            "mod_index": 2.4,
            "feedback": 0.04,
            "index_decay": 0.18,
            "index_sustain": 0.18,
            "attack": 0.01,
            "decay": 0.34,
            "sustain_level": 0.3,
            "release": 0.95,
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
            "cutoff_hz": 1_200.0,
            "keytrack": 0.2,
            "resonance": 0.15,
            "filter_env_amount": 0.35,
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
        "saw_pad": {
            "waveform": "saw",
            "n_harmonics": 16,
            "cutoff_hz": 1_100.0,
            "keytrack": 0.18,
            "resonance": 0.12,
            "filter_env_amount": 0.5,
            "filter_env_decay": 0.7,
            "attack": 0.55,
            "release": 1.8,
        },
        "string_pad": {
            "waveform": "saw",
            "n_harmonics": 18,
            "cutoff_hz": 780.0,
            "keytrack": 0.12,
            "resonance": 0.06,
            "filter_env_amount": 0.32,
            "filter_env_decay": 1.0,
            "attack": 0.75,
            "decay": 0.45,
            "sustain_level": 0.82,
            "release": 2.4,
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
            "filter_mode": "lowpass",
            "filter_drive": 0.2,
        },
        "synth_pluck": {
            "waveform": "square",
            "pulse_width": 0.42,
            "cutoff_hz": 1_700.0,
            "resonance": 0.16,
            "filter_env_amount": 1.7,
            "filter_env_decay": 0.14,
            "keytrack": 0.08,
            "filter_mode": "lowpass",
            "filter_drive": 0.28,
            "attack": 0.005,
            "decay": 0.16,
            "sustain_level": 0.18,
            "release": 0.18,
        },
        "analog_brass": {
            "waveform": "saw",
            "cutoff_hz": 1_450.0,
            "resonance": 0.14,
            "filter_env_amount": 1.05,
            "filter_env_decay": 0.32,
            "keytrack": 0.1,
            "filter_mode": "lowpass",
            "filter_drive": 0.26,
            "attack": 0.03,
            "decay": 0.3,
            "sustain_level": 0.68,
            "release": 0.28,
        },
        "square_lead": {
            "waveform": "square",
            "pulse_width": 0.5,
            "cutoff_hz": 2_200.0,
            "resonance": 0.1,
            "filter_env_amount": 0.7,
            "filter_env_decay": 0.24,
            "keytrack": 0.06,
            "filter_mode": "lowpass",
            "filter_drive": 0.22,
            "attack": 0.01,
            "decay": 0.18,
            "sustain_level": 0.62,
            "release": 0.22,
        },
        "hoover": {
            "waveform": "square",
            "pulse_width": 0.32,
            "cutoff_hz": 2_600.0,
            "resonance": 0.2,
            "filter_env_amount": 1.35,
            "filter_env_decay": 0.38,
            "keytrack": 0.08,
            "filter_mode": "bandpass",
            "filter_drive": 0.34,
            "attack": 0.015,
            "decay": 0.24,
            "sustain_level": 0.66,
            "release": 0.32,
        },
        "moog_bass": {
            "waveform": "saw",
            "cutoff_hz": 520.0,
            "resonance": 0.22,
            "filter_env_amount": 1.6,
            "filter_env_decay": 0.16,
            "keytrack": 0.05,
            "filter_mode": "lowpass",
            "filter_drive": 0.5,
            "attack": 0.004,
            "decay": 0.18,
            "sustain_level": 0.5,
            "release": 0.12,
        },
        "sync_lead": {
            "waveform": "saw",
            "cutoff_hz": 2_400.0,
            "resonance": 0.18,
            "filter_env_amount": 1.0,
            "filter_env_decay": 0.22,
            "keytrack": 0.07,
            "filter_mode": "highpass",
            "filter_drive": 0.26,
            "attack": 0.006,
            "decay": 0.18,
            "sustain_level": 0.56,
            "release": 0.2,
        },
        "acid_bass": {
            "waveform": "square",
            "pulse_width": 0.46,
            "cutoff_hz": 680.0,
            "resonance": 0.42,
            "filter_env_amount": 1.85,
            "filter_env_decay": 0.14,
            "keytrack": 0.04,
            "filter_mode": "lowpass",
            "filter_drive": 0.44,
            "attack": 0.003,
            "decay": 0.14,
            "sustain_level": 0.36,
            "release": 0.1,
        },
        "sub_bass": {
            "waveform": "square",
            "pulse_width": 0.5,
            "cutoff_hz": 260.0,
            "resonance": 0.04,
            "filter_env_amount": 0.4,
            "filter_env_decay": 0.22,
            "keytrack": 0.02,
            "filter_mode": "lowpass",
            "filter_drive": 0.24,
            "attack": 0.004,
            "decay": 0.12,
            "sustain_level": 0.78,
            "release": 0.12,
        },
        "resonant_sweep": {
            "waveform": "saw",
            "cutoff_hz": 900.0,
            "resonance": 0.55,
            "filter_env_amount": 2.4,
            "filter_env_decay": 0.65,
            "keytrack": 0.08,
            "filter_mode": "bandpass",
            "filter_drive": 0.3,
            "attack": 0.012,
            "decay": 0.35,
            "sustain_level": 0.5,
            "release": 0.3,
        },
        "soft_square_pad": {
            "waveform": "square",
            "pulse_width": 0.48,
            "cutoff_hz": 1_000.0,
            "resonance": 0.08,
            "filter_env_amount": 0.45,
            "filter_env_decay": 0.95,
            "keytrack": 0.06,
            "filter_mode": "lowpass",
            "filter_drive": 0.2,
            "attack": 0.42,
            "decay": 0.5,
            "sustain_level": 0.74,
            "release": 1.6,
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


def normalize_synth_spec(params: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize structured/unit-bearing synth params into the flat legacy form."""
    if params is None:
        return {}

    normalized = dict(params)
    env = normalized.pop("env", None)
    engine_params = normalized.pop("params", None)

    if env is not None:
        if not isinstance(env, dict):
            raise ValueError("synth env must be a dict when provided")
        _merge_alias_group(
            destination=normalized,
            source=env,
            alias_map=_ENV_ALIAS_TO_CANONICAL,
            ms_keys=_SECONDS_FROM_MS_KEYS,
        )

    if engine_params is not None:
        if not isinstance(engine_params, dict):
            raise ValueError("synth params must be a dict when provided")
        _merge_alias_group(
            destination=normalized,
            source=engine_params,
            alias_map=_PARAM_ALIAS_TO_CANONICAL,
            ms_keys=_SECONDS_FROM_MS_PARAM_KEYS,
        )

    _rewrite_aliases_in_place(
        destination=normalized,
        alias_map=_ENV_ALIAS_TO_CANONICAL,
        ms_keys=_SECONDS_FROM_MS_KEYS,
    )
    _rewrite_aliases_in_place(
        destination=normalized,
        alias_map=_PARAM_ALIAS_TO_CANONICAL,
        ms_keys=_SECONDS_FROM_MS_PARAM_KEYS,
    )
    return normalized


def resolve_synth_params(params: dict[str, Any]) -> dict[str, Any]:
    """Resolve preset-backed synth params into a single parameter dictionary."""
    resolved = normalize_synth_spec(params)
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


def _merge_alias_group(
    *,
    destination: dict[str, Any],
    source: dict[str, Any],
    alias_map: dict[str, str],
    ms_keys: set[str],
) -> None:
    rewritten = dict(source)
    _rewrite_aliases_in_place(
        destination=rewritten,
        alias_map=alias_map,
        ms_keys=ms_keys,
    )
    destination.update(rewritten)


def _rewrite_aliases_in_place(
    *,
    destination: dict[str, Any],
    alias_map: dict[str, str],
    ms_keys: set[str],
) -> None:
    resolved_canonical_keys: set[str] = set()
    for alias_name, canonical_name in alias_map.items():
        if alias_name not in destination:
            continue
        if canonical_name in resolved_canonical_keys:
            if alias_name != canonical_name:
                destination.pop(alias_name)
            continue
        raw_value = destination.pop(alias_name)
        if alias_name in ms_keys:
            destination[canonical_name] = float(raw_value) / 1000.0
        else:
            destination[canonical_name] = raw_value
        resolved_canonical_keys.add(canonical_name)


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
