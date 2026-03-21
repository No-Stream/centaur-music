"""Engine registry, synth-spec normalization, and preset resolution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from code_musics.engines import (
    additive,
    filtered_stack,
    fm,
    kick_tom,
    noise_perc,
    polyblep,
)

EngineRenderer = Callable[..., np.ndarray]

_ENGINES: dict[str, EngineRenderer] = {
    "additive": additive.render,
    "fm": fm.render,
    "filtered_stack": filtered_stack.render,
    "noise_perc": noise_perc.render,
    "kick_tom": kick_tom.render,
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
    "body_punch": "body_punch_ratio",
    "body_punch_ratio": "body_punch_ratio",
    "body_tone": "body_tone_ratio",
    "body_tone_ratio": "body_tone_ratio",
    "drive": "drive_ratio",
    "drive_ratio": "drive_ratio",
    "sweep_amount_ratio": "pitch_sweep_amount_ratio",
    "pitch_sweep_amount_ratio": "pitch_sweep_amount_ratio",
    "sweep_decay_ms": "pitch_sweep_decay_ms",
    "pitch_sweep_decay_ms": "pitch_sweep_decay_ms",
    "body_decay_ms": "body_decay_ms",
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
    "kick_tom": {
        "808_hiphop": {
            "body_decay_ms": 720.0,
            "pitch_sweep_amount_ratio": 3.2,
            "pitch_sweep_decay_ms": 64.0,
            "body_wave": "sine",
            "body_tone_ratio": 0.06,
            "body_punch_ratio": 0.12,
            "overtone_amount": 0.02,
            "overtone_ratio": 1.6,
            "overtone_decay_ms": 160.0,
            "click_amount": 0.015,
            "click_decay_ms": 8.0,
            "click_tone_hz": 2_200.0,
            "noise_amount": 0.01,
            "noise_decay_ms": 24.0,
            "noise_bandpass_hz": 700.0,
            "drive_ratio": 0.08,
            "post_lowpass_hz": 7_500.0,
        },
        "808_house": {
            "body_decay_ms": 360.0,
            "pitch_sweep_amount_ratio": 2.8,
            "pitch_sweep_decay_ms": 42.0,
            "body_wave": "sine",
            "body_tone_ratio": 0.10,
            "body_punch_ratio": 0.20,
            "overtone_amount": 0.04,
            "overtone_ratio": 1.9,
            "overtone_decay_ms": 120.0,
            "click_amount": 0.06,
            "click_decay_ms": 7.0,
            "click_tone_hz": 3_200.0,
            "noise_amount": 0.02,
            "noise_decay_ms": 22.0,
            "noise_bandpass_hz": 1_100.0,
            "drive_ratio": 0.12,
            "post_lowpass_hz": 11_000.0,
        },
        "808_tape": {
            "body_decay_ms": 520.0,
            "pitch_sweep_amount_ratio": 2.9,
            "pitch_sweep_decay_ms": 54.0,
            "body_wave": "sine_clip",
            "body_tone_ratio": 0.12,
            "body_punch_ratio": 0.18,
            "overtone_amount": 0.06,
            "overtone_ratio": 1.8,
            "overtone_decay_ms": 130.0,
            "click_amount": 0.04,
            "click_decay_ms": 8.0,
            "click_tone_hz": 2_800.0,
            "noise_amount": 0.018,
            "noise_decay_ms": 26.0,
            "noise_bandpass_hz": 900.0,
            "drive_ratio": 0.22,
            "post_lowpass_hz": 8_800.0,
        },
        "909_techno": {
            "body_decay_ms": 300.0,
            "pitch_sweep_amount_ratio": 2.7,
            "pitch_sweep_decay_ms": 34.0,
            "body_wave": "triangle",
            "body_tone_ratio": 0.22,
            "body_punch_ratio": 0.28,
            "overtone_amount": 0.12,
            "overtone_ratio": 2.1,
            "overtone_decay_ms": 95.0,
            "click_amount": 0.18,
            "click_decay_ms": 6.0,
            "click_tone_hz": 4_400.0,
            "noise_amount": 0.03,
            "noise_decay_ms": 18.0,
            "noise_bandpass_hz": 2_200.0,
            "drive_ratio": 0.20,
            "post_lowpass_hz": 12_500.0,
        },
        "909_house": {
            "body_decay_ms": 240.0,
            "pitch_sweep_amount_ratio": 2.4,
            "pitch_sweep_decay_ms": 30.0,
            "body_wave": "triangle",
            "body_tone_ratio": 0.18,
            "body_punch_ratio": 0.24,
            "overtone_amount": 0.10,
            "overtone_ratio": 2.0,
            "overtone_decay_ms": 85.0,
            "click_amount": 0.22,
            "click_decay_ms": 5.0,
            "click_tone_hz": 5_200.0,
            "noise_amount": 0.025,
            "noise_decay_ms": 16.0,
            "noise_bandpass_hz": 2_800.0,
            "drive_ratio": 0.16,
            "post_lowpass_hz": 14_000.0,
        },
        "909_crunch": {
            "body_decay_ms": 320.0,
            "pitch_sweep_amount_ratio": 2.6,
            "pitch_sweep_decay_ms": 36.0,
            "body_wave": "sine_clip",
            "body_tone_ratio": 0.24,
            "body_punch_ratio": 0.30,
            "overtone_amount": 0.16,
            "overtone_ratio": 2.2,
            "overtone_decay_ms": 100.0,
            "click_amount": 0.20,
            "click_decay_ms": 6.0,
            "click_tone_hz": 4_600.0,
            "noise_amount": 0.04,
            "noise_decay_ms": 20.0,
            "noise_bandpass_hz": 2_400.0,
            "drive_ratio": 0.34,
            "post_lowpass_hz": 11_500.0,
        },
        "distorted_hardkick": {
            "body_decay_ms": 260.0,
            "pitch_sweep_amount_ratio": 3.0,
            "pitch_sweep_decay_ms": 28.0,
            "body_wave": "sine_clip",
            "body_tone_ratio": 0.28,
            "body_punch_ratio": 0.34,
            "overtone_amount": 0.22,
            "overtone_ratio": 2.4,
            "overtone_decay_ms": 110.0,
            "click_amount": 0.24,
            "click_decay_ms": 5.0,
            "click_tone_hz": 5_000.0,
            "noise_amount": 0.05,
            "noise_decay_ms": 18.0,
            "noise_bandpass_hz": 2_600.0,
            "drive_ratio": 0.48,
            "post_lowpass_hz": 10_000.0,
        },
        "zap_kick": {
            "body_decay_ms": 180.0,
            "pitch_sweep_amount_ratio": 5.0,
            "pitch_sweep_decay_ms": 14.0,
            "body_wave": "sine",
            "body_tone_ratio": 0.20,
            "body_punch_ratio": 0.22,
            "overtone_amount": 0.16,
            "overtone_ratio": 2.8,
            "overtone_decay_ms": 55.0,
            "click_amount": 0.12,
            "click_decay_ms": 4.0,
            "click_tone_hz": 4_800.0,
            "noise_amount": 0.02,
            "noise_decay_ms": 12.0,
            "noise_bandpass_hz": 1_800.0,
            "drive_ratio": 0.18,
            "post_lowpass_hz": 13_000.0,
        },
        "round_tom": {
            "body_decay_ms": 420.0,
            "pitch_sweep_amount_ratio": 1.28,
            "pitch_sweep_decay_ms": 62.0,
            "body_wave": "triangle",
            "body_tone_ratio": 0.30,
            "body_punch_ratio": 0.08,
            "overtone_amount": 0.38,
            "overtone_ratio": 2.05,
            "overtone_decay_ms": 210.0,
            "click_amount": 0.03,
            "click_decay_ms": 8.0,
            "click_tone_hz": 2_400.0,
            "noise_amount": 0.01,
            "noise_decay_ms": 26.0,
            "noise_bandpass_hz": 1_100.0,
            "drive_ratio": 0.14,
            "post_lowpass_hz": 12_000.0,
        },
        "floor_tom": {
            "body_decay_ms": 520.0,
            "pitch_sweep_amount_ratio": 1.55,
            "pitch_sweep_decay_ms": 76.0,
            "body_wave": "triangle",
            "body_tone_ratio": 0.18,
            "body_punch_ratio": 0.12,
            "overtone_amount": 0.20,
            "overtone_ratio": 1.48,
            "overtone_decay_ms": 220.0,
            "click_amount": 0.03,
            "click_decay_ms": 8.0,
            "click_tone_hz": 1_800.0,
            "noise_amount": 0.008,
            "noise_decay_ms": 30.0,
            "noise_bandpass_hz": 600.0,
            "drive_ratio": 0.10,
            "post_lowpass_hz": 8_000.0,
        },
        "electro_tom": {
            "body_decay_ms": 320.0,
            "pitch_sweep_amount_ratio": 1.7,
            "pitch_sweep_decay_ms": 48.0,
            "body_wave": "triangle",
            "body_tone_ratio": 0.24,
            "body_punch_ratio": 0.18,
            "overtone_amount": 0.16,
            "overtone_ratio": 2.0,
            "overtone_decay_ms": 130.0,
            "click_amount": 0.08,
            "click_decay_ms": 6.0,
            "click_tone_hz": 3_000.0,
            "noise_amount": 0.015,
            "noise_decay_ms": 20.0,
            "noise_bandpass_hz": 1_200.0,
            "drive_ratio": 0.16,
            "post_lowpass_hz": 10_500.0,
        },
        "ring_tom": {
            "body_decay_ms": 460.0,
            "pitch_sweep_amount_ratio": 1.35,
            "pitch_sweep_decay_ms": 56.0,
            "body_wave": "sine",
            "body_tone_ratio": 0.20,
            "body_punch_ratio": 0.10,
            "overtone_amount": 0.28,
            "overtone_ratio": 2.45,
            "overtone_decay_ms": 210.0,
            "click_amount": 0.03,
            "click_decay_ms": 7.0,
            "click_tone_hz": 2_400.0,
            "noise_amount": 0.008,
            "noise_decay_ms": 20.0,
            "noise_bandpass_hz": 900.0,
            "drive_ratio": 0.12,
            "post_lowpass_hz": 9_500.0,
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
