"""Pluck-excitation + modal-resonator harpsichord synthesis engine.

Captures the crisp, immediate attack and bright decay of a plucked-string
harpsichord while going beyond physical constraints: velocity expression,
per-note spectral morphing, continuous register blending, and custom partial
ratios for xenharmonic tuning.

The synthesis chain:
  1. Shaped pluck impulse determines initial mode amplitudes
  2. Modal resonator bank (decaying sinusoids) produces the string tone
  3. Multiple registers (8', 4', lute) are blended with independent pluck
     character and decay
  4. Per-note spectral morphing fades from bright attack to warmer sustain
  5. Post-processing: drift, soundboard, saturation, release noise
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from code_musics.engines._dsp_utils import (
    GOLDEN_RATIO_FRAC,
    apply_body_saturation,
    apply_soundboard,
    bandpass_noise,
    build_drift,
    compute_mode_ratios,
    nyquist_fade,
    render_damper_thump,
    render_noise_floor,
    rng_for_note,
)

logger: logging.Logger = logging.getLogger(__name__)

# --- Default parameter values ---

_DEFAULT_N_MODES = 40
_DEFAULT_INHARMONICITY = 0.00005
_DEFAULT_DECAY_BASE = 1.5
_DEFAULT_DECAY_TILT = 1.2

_DEFAULT_PLUCK_POSITION = 0.15
_DEFAULT_PLUCK_HARDNESS = 0.6
_DEFAULT_PLUCK_NOISE = 0.3
_DEFAULT_VELOCITY_TILT = 0.4

_DEFAULT_ATTACK_BRIGHTNESS = 1.5
_DEFAULT_MORPH_TIME = 0.3

_DEFAULT_DRIFT = 0.06
_DEFAULT_DRIFT_RATE_HZ = 0.04
_DEFAULT_BODY_SATURATION = 0.10
_DEFAULT_SOUNDBOARD_COLOR = 0.25
_DEFAULT_SOUNDBOARD_BRIGHTNESS = 0.65
_DEFAULT_RELEASE_NOISE = 0.06

# --- Default register definitions ---

_DEFAULT_REGISTERS: list[dict[str, Any]] = [
    {
        "name": "front_8",
        "pitch_mult": 1.0,
        "pluck_position": 0.15,
        "pluck_hardness": 0.6,
        "pluck_noise": 0.3,
        "brightness_tilt": 0.0,
        "decay_scale": 1.0,
        "partial_ratios": None,
        "blend": 1.0,
    },
    {
        "name": "back_8",
        "pitch_mult": 1.0,
        "pluck_position": 0.22,
        "pluck_hardness": 0.55,
        "pluck_noise": 0.25,
        "brightness_tilt": -0.1,
        "decay_scale": 1.05,
        "partial_ratios": None,
        "blend": 0.0,
    },
    {
        "name": "four_foot",
        "pitch_mult": 2.0,
        "pluck_position": 0.12,
        "pluck_hardness": 0.7,
        "pluck_noise": 0.35,
        "brightness_tilt": 0.15,
        "decay_scale": 0.7,
        "partial_ratios": None,
        "blend": 0.0,
    },
    {
        "name": "lute",
        "pitch_mult": 1.0,
        "pluck_position": 0.18,
        "pluck_hardness": 0.4,
        "pluck_noise": 0.15,
        "brightness_tilt": -0.3,
        "decay_scale": 0.6,
        "partial_ratios": None,
        "blend": 0.0,
    },
]


def render(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
    freq_trajectory: np.ndarray | None = None,
) -> np.ndarray:
    """Render a harpsichord note via pluck excitation + modal resonator bank."""
    if freq <= 0:
        raise ValueError("freq must be positive")
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0)

    # --- Extract top-level params ---
    n_modes = int(params.get("n_modes", _DEFAULT_N_MODES))
    inharmonicity = float(params.get("inharmonicity", _DEFAULT_INHARMONICITY))
    decay_base = float(params.get("decay_base", _DEFAULT_DECAY_BASE))
    decay_tilt = float(params.get("decay_tilt", _DEFAULT_DECAY_TILT))

    velocity_tilt = float(params.get("velocity_tilt", _DEFAULT_VELOCITY_TILT))

    attack_brightness = float(
        params.get("attack_brightness", _DEFAULT_ATTACK_BRIGHTNESS)
    )
    morph_time = float(params.get("morph_time", _DEFAULT_MORPH_TIME))

    drift = float(params.get("drift", _DEFAULT_DRIFT))
    drift_rate_hz = float(params.get("drift_rate_hz", _DEFAULT_DRIFT_RATE_HZ))
    body_saturation = float(params.get("body_saturation", _DEFAULT_BODY_SATURATION))
    soundboard_color = float(params.get("soundboard_color", _DEFAULT_SOUNDBOARD_COLOR))
    soundboard_brightness = float(
        params.get("soundboard_brightness", _DEFAULT_SOUNDBOARD_BRIGHTNESS)
    )
    release_noise = float(params.get("release_noise", _DEFAULT_RELEASE_NOISE))
    partial_ratios_raw = params.get("partial_ratios")

    if inharmonicity < 0:
        raise ValueError("inharmonicity must be non-negative")

    if freq_trajectory is not None:
        freq_trajectory = np.asarray(freq_trajectory, dtype=np.float64)
        if freq_trajectory.ndim != 1:
            raise ValueError("freq_trajectory must be one-dimensional")
        if freq_trajectory.size != n_samples:
            raise ValueError("freq_trajectory length must match note duration")

    # --- Resolve registers ---
    registers = _resolve_registers(params)

    active_registers = [r for r in registers if r["blend"] > 0]
    if not active_registers:
        raise ValueError("at least one register must have blend > 0")

    rng = rng_for_note(freq=freq, duration=duration, amp=amp, sample_rate=sample_rate)

    # --- Render each active register ---
    signal = np.zeros(n_samples, dtype=np.float64)

    for reg in active_registers:
        reg_freq = freq * reg["pitch_mult"]
        reg_hardness = np.clip(
            reg["pluck_hardness"] + velocity_tilt * (amp - 0.5), 0.05, 0.99
        )
        reg_position = reg["pluck_position"]
        reg_noise = reg["pluck_noise"]
        reg_brightness_tilt = reg["brightness_tilt"]
        reg_decay_scale = reg["decay_scale"]
        reg_partial_ratios = reg.get("partial_ratios") or partial_ratios_raw

        # Build freq_trajectory for this register's pitch multiplier
        reg_freq_trajectory = None
        if freq_trajectory is not None:
            reg_freq_trajectory = freq_trajectory * reg["pitch_mult"]

        # Compute mode ratios
        mode_ratios, mode_amps = compute_mode_ratios(
            freq=reg_freq,
            n_modes=n_modes,
            inharmonicity=inharmonicity,
            partial_ratios=reg_partial_ratios,
            sample_rate=sample_rate,
            amp_rolloff_exp=0.8,
        )

        if mode_ratios.size == 0:
            continue

        # Build drift trajectory for this register
        phase_offset = hash(reg["name"]) * GOLDEN_RATIO_FRAC * 2.0 * np.pi
        drift_trajectory = build_drift(
            n_samples=n_samples,
            drift_amount=drift,
            drift_rate_hz=drift_rate_hz,
            duration=duration,
            phase_offset=phase_offset,
        )

        # Render modal string for this register
        reg_signal = _render_register_string(
            freq=reg_freq,
            mode_ratios=mode_ratios,
            mode_amps=mode_amps,
            n_samples=n_samples,
            sample_rate=sample_rate,
            decay_base=decay_base * reg_decay_scale,
            decay_tilt=decay_tilt,
            pluck_position=reg_position,
            pluck_hardness=float(reg_hardness),
            pluck_noise=reg_noise,
            brightness_tilt=reg_brightness_tilt,
            attack_brightness=attack_brightness,
            morph_time=morph_time,
            drift_trajectory=drift_trajectory,
            freq_trajectory=reg_freq_trajectory,
            amp=amp,
            rng=rng,
        )

        signal += reg["blend"] * reg_signal

    # --- Post-processing ---
    if body_saturation > 0:
        signal = apply_body_saturation(signal, body_saturation)

    signal += render_noise_floor(
        signal=signal,
        sample_rate=sample_rate,
        n_samples=n_samples,
        rng=rng,
    )

    mixed = apply_soundboard(
        signal=signal,
        soundboard_color=soundboard_color,
        soundboard_brightness=soundboard_brightness,
        sample_rate=sample_rate,
    )

    # Release noise (brighter than piano's damper thump)
    if release_noise > 0:
        string_rms = float(np.sqrt(np.mean(mixed**2)))
        release_center_hz = min(freq * 4.0, sample_rate * 0.4)
        release_burst = render_damper_thump(
            freq=freq,
            damper_noise=release_noise,
            n_samples=n_samples,
            sample_rate=sample_rate,
            rng=rng,
            level_scale=string_rms,
            burst_duration_s=0.012,
            center_hz=release_center_hz,
            width_ratio=1.2,
        )
        mixed += release_burst

    # Peak-normalize and scale by amp
    peak = np.max(np.abs(mixed))
    if peak <= 0.0:
        raise ValueError("harpsichord render produced no audible output")
    return amp * (mixed / peak)


def _render_register_string(
    *,
    freq: float,
    mode_ratios: np.ndarray,
    mode_amps: np.ndarray,
    n_samples: int,
    sample_rate: int,
    decay_base: float,
    decay_tilt: float,
    pluck_position: float,
    pluck_hardness: float,
    pluck_noise: float,
    brightness_tilt: float,
    attack_brightness: float,
    morph_time: float,
    drift_trajectory: np.ndarray,
    freq_trajectory: np.ndarray | None,
    amp: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render one register's string via pluck excitation + vectorized modal decay."""
    n_modes = mode_ratios.size
    if n_modes == 0:
        return np.zeros(n_samples, dtype=np.float64)

    mode_freqs_hz = mode_ratios * freq
    nyquist = sample_rate / 2.0

    # --- Decay rates per mode ---
    sigma_base = 1.0 / max(0.01, decay_base)
    sigma_tilt_scale = decay_tilt * 50.0
    ks = np.arange(1, n_modes + 1, dtype=np.float64)
    sigma = sigma_base + sigma_tilt_scale * (ks / n_modes) ** 2

    # --- Pluck excitation weights ---
    # Position comb filter: nodes at pluck_position multiples
    position_weights = np.sin(np.pi * ks * pluck_position)

    # Hardness lowpass: harder pluck excites more high partials
    hardness_cutoff_hz = freq * (4.0 + 40.0 * pluck_hardness)
    hardness_atten = 1.0 / (1.0 + (mode_freqs_hz / hardness_cutoff_hz) ** 1.5)

    # Brightness tilt: per-register spectral shaping
    if brightness_tilt != 0.0:
        brightness_factor = ks**brightness_tilt
    else:
        brightness_factor = np.ones(n_modes, dtype=np.float64)

    # Combined excitation amplitudes
    excitation_amps = (
        mode_amps * np.abs(position_weights) * hardness_atten * brightness_factor
    )

    # Nyquist fade
    mode_freq_arr = np.full(n_modes, 0.0, dtype=np.float64)
    mode_freq_arr[:] = mode_freqs_hz
    nq_fade = nyquist_fade(mode_freq_arr, nyquist)
    excitation_amps *= nq_fade

    # --- Spectral morph envelope ---
    # attack_brightness > 1 means brighter attack; the morph fades from bright to steady
    morph_samples = max(1, int(morph_time * sample_rate))
    morph_samples = min(morph_samples, n_samples)

    # Per-mode attack brightness factor: higher modes get more boost
    if attack_brightness != 1.0:
        # Scale factor increases with mode index: fundamental gets less boost
        morph_boost = 1.0 + (attack_brightness - 1.0) * (ks / n_modes)
    else:
        morph_boost = np.ones(n_modes, dtype=np.float64)

    # --- Vectorized modal synthesis with spectral morphing ---
    n_arr = np.arange(n_samples, dtype=np.float64)

    # Frequency scaling for pitch automation
    freq_scale = np.ones(n_samples, dtype=np.float64)
    if freq_trajectory is not None:
        freq_scale = freq_trajectory / freq

    signal = np.zeros(n_samples, dtype=np.float64)

    # Build morph envelope (time axis): 1.0 at onset, fading to 0.0 at morph_samples
    morph_env = np.zeros(n_samples, dtype=np.float64)
    morph_env[:morph_samples] = (
        1.0 - np.arange(morph_samples, dtype=np.float64) / morph_samples
    )

    # Precompute cumulative phase-increment and decay base outside the mode loop.
    # Each mode multiplies by its own scalar theta, avoiding n_modes cumsum calls.
    cumsum_scaled_drift = np.cumsum(freq_scale * drift_trajectory)
    base_decay = np.exp(-n_arr / sample_rate)

    for k in range(n_modes):
        if excitation_amps[k] < 1e-10:
            continue

        # Base angular frequency
        theta = 2.0 * np.pi * mode_freqs_hz[k] / sample_rate

        # Phase from precomputed cumulative drift (scaled by per-mode theta)
        cumulative_phase = theta * cumsum_scaled_drift

        # Random initial phase for more natural onset
        init_phase = rng.uniform(0.0, 2.0 * np.pi)

        # Exponential decay from precomputed base
        decay = base_decay ** sigma[k]

        # Time-varying amplitude with spectral morph
        # steady amplitude + morph envelope * (boost - 1)
        mode_amp_env = excitation_amps[k] * (1.0 + morph_env * (morph_boost[k] - 1.0))

        signal += mode_amp_env * decay * np.sin(cumulative_phase + init_phase)

    # --- Mix pluck noise ---
    if pluck_noise > 0:
        noise_duration_samples = max(1, int(0.002 * sample_rate))  # ~2ms
        noise_burst = rng.standard_normal(n_samples)

        # Shape noise to be very short
        noise_env = np.zeros(n_samples, dtype=np.float64)
        noise_len = min(noise_duration_samples * 3, n_samples)
        t_noise = np.arange(noise_len, dtype=np.float64)
        noise_env[:noise_len] = np.exp(-t_noise / max(1.0, noise_duration_samples))

        # Bandpass around the pluck frequency area
        noise_center = min(freq * (2.0 + 6.0 * pluck_hardness), sample_rate * 0.4)
        noise_burst = bandpass_noise(
            noise_burst,
            sample_rate=sample_rate,
            center_hz=noise_center,
            width_ratio=1.5,
        )

        noise_peak = np.max(np.abs(noise_burst))
        if noise_peak > 0:
            noise_burst /= noise_peak

        signal_peak = np.max(np.abs(signal))
        if signal_peak > 0:
            signal += pluck_noise * amp * signal_peak * noise_env * noise_burst

    return signal


def _resolve_registers(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve register definitions from params, applying blend overrides."""
    custom_registers = params.get("registers")
    if custom_registers is not None:
        return [dict(r) for r in custom_registers]

    # Start from defaults, apply convenience blend params
    registers = [dict(r) for r in _DEFAULT_REGISTERS]

    blend_keys = {
        "front_8_blend": "front_8",
        "back_8_blend": "back_8",
        "four_foot_blend": "four_foot",
        "lute_blend": "lute",
    }

    for param_key, reg_name in blend_keys.items():
        if param_key in params:
            for reg in registers:
                if reg["name"] == reg_name:
                    reg["blend"] = float(params[param_key])
                    break

    return registers
