"""Modal piano synthesis engine with physical hammer-string interaction."""

from __future__ import annotations

from typing import Any

import numba
import numpy as np

from code_musics.engines._dsp_utils import (
    GOLDEN_RATIO_FRAC,
    apply_body_saturation,
    apply_soundboard,
    build_drift,
    compute_mode_ratios,
    render_damper_thump,
    render_noise_floor,
    rng_for_note,
)

_DEFAULT_N_MODES = 32
_DEFAULT_INHARMONICITY = 0.0005
_DEFAULT_DECAY_BASE = 3.5
_DEFAULT_DECAY_TILT = 0.5
_DEFAULT_HAMMER_MASS = 0.01
_DEFAULT_HAMMER_STIFFNESS = 1e8
_DEFAULT_HAMMER_EXPONENT = 2.5
_DEFAULT_HAMMER_POSITION = 0.12
_DEFAULT_BRIDGE_POSITION = 0.95
_DEFAULT_MAX_HAMMER_VELOCITY = 4.0

_DEFAULT_UNISON_DETUNE = 3.0
_DEFAULT_UNISON_DRIFT = 0.15
_DEFAULT_DRIFT = 0.08
_DEFAULT_DRIFT_RATE_HZ = 0.05
_DEFAULT_BODY_SATURATION = 0.15
_DEFAULT_SOUNDBOARD_COLOR = 0.4
_DEFAULT_SOUNDBOARD_BRIGHTNESS = 0.5
_DEFAULT_DAMPER_NOISE = 0.08

_OVERSAMPLE_FACTOR = 2
_MAX_CONTACT_MS = 30.0


def render(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
    freq_trajectory: np.ndarray | None = None,
) -> np.ndarray:
    """Render a piano note via modal synthesis with physical hammer-string interaction."""
    if freq <= 0:
        raise ValueError("freq must be positive")
    if duration < 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0)

    n_modes = int(params.get("n_modes", params.get("n_partials", _DEFAULT_N_MODES)))
    inharmonicity = float(params.get("inharmonicity", _DEFAULT_INHARMONICITY))
    decay_base = float(params.get("decay_base", _DEFAULT_DECAY_BASE))
    decay_tilt = float(
        params.get("decay_tilt", params.get("decay_partial_tilt", _DEFAULT_DECAY_TILT))
    )
    hammer_mass = float(params.get("hammer_mass", _DEFAULT_HAMMER_MASS))
    hammer_stiffness = float(params.get("hammer_stiffness", _DEFAULT_HAMMER_STIFFNESS))
    hammer_exponent = float(params.get("hammer_exponent", _DEFAULT_HAMMER_EXPONENT))
    hammer_position = float(params.get("hammer_position", _DEFAULT_HAMMER_POSITION))
    bridge_position = float(params.get("bridge_position", _DEFAULT_BRIDGE_POSITION))
    max_hammer_velocity = float(
        params.get("max_hammer_velocity", _DEFAULT_MAX_HAMMER_VELOCITY)
    )

    unison_count_raw = params.get("unison_count")
    unison_detune = float(params.get("unison_detune", _DEFAULT_UNISON_DETUNE))
    unison_drift = float(params.get("unison_drift", _DEFAULT_UNISON_DRIFT))
    drift = float(params.get("drift", _DEFAULT_DRIFT))
    drift_rate_hz = float(params.get("drift_rate_hz", _DEFAULT_DRIFT_RATE_HZ))
    body_saturation = float(params.get("body_saturation", _DEFAULT_BODY_SATURATION))
    soundboard_color = float(params.get("soundboard_color", _DEFAULT_SOUNDBOARD_COLOR))
    soundboard_brightness = float(
        params.get("soundboard_brightness", _DEFAULT_SOUNDBOARD_BRIGHTNESS)
    )
    damper_noise = float(params.get("damper_noise", _DEFAULT_DAMPER_NOISE))
    partial_ratios_raw = params.get("partial_ratios")

    if inharmonicity < 0:
        raise ValueError("inharmonicity must be non-negative")

    if freq_trajectory is not None:
        freq_trajectory = np.asarray(freq_trajectory, dtype=np.float64)
        if freq_trajectory.ndim != 1:
            raise ValueError("freq_trajectory must be one-dimensional")
        if freq_trajectory.size != n_samples:
            raise ValueError("freq_trajectory length must match note duration")

    if unison_count_raw is not None:
        unison_count = int(unison_count_raw)
    elif freq < 200.0:
        unison_count = 1
    elif freq < 400.0:
        unison_count = 2
    else:
        unison_count = 3

    mode_ratios, _ = compute_mode_ratios(
        freq=freq,
        n_modes=n_modes,
        inharmonicity=inharmonicity,
        partial_ratios=partial_ratios_raw,
        sample_rate=sample_rate,
    )

    rng = rng_for_note(freq=freq, duration=duration, amp=amp, sample_rate=sample_rate)

    hammer_velocity = amp * max_hammer_velocity

    string_signal = np.zeros(n_samples, dtype=np.float64)
    for string_idx in range(unison_count):
        phase_offset = string_idx * GOLDEN_RATIO_FRAC * 2.0 * np.pi
        drift_trajectory = build_drift(
            n_samples=n_samples,
            drift_amount=drift + unison_drift * (1 if string_idx > 0 else 0),
            drift_rate_hz=drift_rate_hz * (1.0 + 0.15 * string_idx),
            duration=duration,
            phase_offset=phase_offset,
        )

        detune_cents = 0.0
        if unison_count > 1:
            spread = unison_detune
            detune_cents = spread * ((string_idx / (unison_count - 1)) * 2.0 - 1.0)
            if unison_count == 2:
                detune_cents = spread * (1.0 if string_idx == 1 else -1.0)
        detune_ratio = 2.0 ** (detune_cents / 1200.0)

        string_signal += _render_modal_string(
            freq=freq,
            mode_ratios=mode_ratios,
            detune_ratio=detune_ratio,
            decay_base=decay_base,
            decay_tilt=decay_tilt,
            hammer_mass=hammer_mass,
            hammer_stiffness=hammer_stiffness,
            hammer_exponent=hammer_exponent,
            hammer_position=hammer_position,
            bridge_position=bridge_position,
            hammer_velocity=hammer_velocity,
            drift_trajectory=drift_trajectory,
            n_samples=n_samples,
            sample_rate=sample_rate,
            freq_trajectory=freq_trajectory,
        )

    if body_saturation > 0:
        string_signal = apply_body_saturation(string_signal, body_saturation)

    string_signal += render_noise_floor(
        signal=string_signal,
        sample_rate=sample_rate,
        n_samples=n_samples,
        rng=rng,
    )

    mixed = apply_soundboard(
        signal=string_signal,
        soundboard_color=soundboard_color,
        soundboard_brightness=soundboard_brightness,
        sample_rate=sample_rate,
    )

    if damper_noise > 0:
        string_rms = float(np.sqrt(np.mean(mixed**2)))
        mixed += render_damper_thump(
            freq=freq,
            damper_noise=damper_noise,
            n_samples=n_samples,
            sample_rate=sample_rate,
            rng=rng,
            level_scale=string_rms,
        )

    peak = np.max(np.abs(mixed))
    if peak <= 0.0:
        raise ValueError("piano render produced no audible output")
    return amp * (mixed / peak)


def _render_modal_string(
    *,
    freq: float,
    mode_ratios: np.ndarray,
    detune_ratio: float,
    decay_base: float,
    decay_tilt: float,
    hammer_mass: float,
    hammer_stiffness: float,
    hammer_exponent: float,
    hammer_position: float,
    bridge_position: float,
    hammer_velocity: float,
    drift_trajectory: np.ndarray,
    n_samples: int,
    sample_rate: int,
    freq_trajectory: np.ndarray | None,
) -> np.ndarray:
    """Render one string via modal hammer-string contact + vectorized free decay."""
    n_modes = mode_ratios.size
    if n_modes == 0:
        return np.zeros(n_samples, dtype=np.float64)

    mode_freqs_hz = mode_ratios * freq * detune_ratio

    sigma_base = 1.0 / max(0.01, decay_base)
    sigma_tilt = decay_tilt * 80.0
    ks = np.arange(1, n_modes + 1, dtype=np.float64)
    sigma = sigma_base + sigma_tilt * (ks / n_modes) ** 2

    os_sr = sample_rate * _OVERSAMPLE_FACTOR
    theta = 2.0 * np.pi * mode_freqs_hz / os_sr
    r = np.exp(-sigma / os_sr)
    a1 = 2.0 * r * np.cos(theta)
    a2 = -(r**2)

    felt_cutoff_hz = freq * (8.0 + 40.0 * (hammer_stiffness / 1e9))
    felt_atten = 1.0 / (1.0 + (mode_freqs_hz / felt_cutoff_hz) ** 1.5)
    b_in = np.sin(np.pi * ks * hammer_position) * felt_atten
    w_out = np.sin(np.pi * ks * bridge_position) / ks

    max_contact_samples = int(_MAX_CONTACT_MS / 1000.0 * os_sr)
    max_contact_samples = min(max_contact_samples, n_samples * _OVERSAMPLE_FACTOR)

    _contact_audio, y_final, y_prev_final, _contact_len = _simulate_hammer_contact(
        a1=a1,
        a2=a2,
        b_in=b_in,
        w_out=w_out,
        hammer_mass=hammer_mass,
        hammer_stiffness=hammer_stiffness,
        hammer_exponent=hammer_exponent,
        hammer_velocity=hammer_velocity,
        sample_rate=os_sr,
        max_contact_samples=max_contact_samples,
    )

    theta_sr = 2.0 * np.pi * mode_freqs_hz / sample_rate
    r_sr = np.exp(-sigma / sample_rate)

    signal = _synthesize_free_decay(
        y_final=y_final,
        y_prev_final=y_prev_final,
        theta_os=theta,
        r_os=r,
        theta_sr=theta_sr,
        r_sr=r_sr,
        w_out=w_out,
        remaining_samples=n_samples,
        freq_trajectory=freq_trajectory,
        contact_samples_at_sr=0,
        freq=freq,
    )

    onset_fade_ms = 8.0
    onset_fade_samples = min(int(onset_fade_ms / 1000 * sample_rate), n_samples)
    if onset_fade_samples > 1:
        signal[:onset_fade_samples] *= 0.5 * (
            1.0 - np.cos(np.linspace(0, np.pi, onset_fade_samples))
        )

    signal *= drift_trajectory[:n_samples]
    return signal[:n_samples]


@numba.njit(cache=True)
def _simulate_hammer_contact(
    a1: np.ndarray,
    a2: np.ndarray,
    b_in: np.ndarray,
    w_out: np.ndarray,
    hammer_mass: float,
    hammer_stiffness: float,
    hammer_exponent: float,
    hammer_velocity: float,
    sample_rate: int,
    max_contact_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Simulate hammer-string contact via Stoermer-Verlet integration.

    Returns (output_signal, y_final, y_prev_final, contact_length).
    """
    n_modes = a1.shape[0]
    dt = 1.0 / sample_rate

    y = np.zeros(n_modes)
    y_prev = np.zeros(n_modes)
    output = np.zeros(max_contact_samples)

    hammer_pos = 0.0001
    hammer_vel = hammer_velocity
    release_count = 0
    contact_length = max_contact_samples

    for n in range(max_contact_samples):
        y_at_hammer = 0.0
        for k in range(n_modes):
            y_at_hammer += b_in[k] * y[k]

        delta = hammer_pos - y_at_hammer
        if delta > 0.0:
            force = hammer_stiffness * (delta**hammer_exponent)
            release_count = 0
        else:
            force = 0.0
            release_count += 1

        if release_count >= 3 and n > 10:
            contact_length = n
            break

        hammer_accel = -force / hammer_mass
        hammer_vel += hammer_accel * dt
        hammer_pos += hammer_vel * dt

        for k in range(n_modes):
            y_new = a1[k] * y[k] + a2[k] * y_prev[k] + b_in[k] * force / sample_rate
            y_prev[k] = y[k]
            y[k] = y_new

        out_sample = 0.0
        for k in range(n_modes):
            out_sample += w_out[k] * y[k]
        output[n] = out_sample

    return output, y, y_prev, contact_length


def _synthesize_free_decay(
    *,
    y_final: np.ndarray,
    y_prev_final: np.ndarray,
    theta_os: np.ndarray,
    r_os: np.ndarray,
    theta_sr: np.ndarray,
    r_sr: np.ndarray,
    w_out: np.ndarray,
    remaining_samples: int,
    freq_trajectory: np.ndarray | None,
    contact_samples_at_sr: int,
    freq: float,
) -> np.ndarray:
    """Vectorized free-decay synthesis from final mode states.

    Phase extraction uses oversampled coefficients (theta_os, r_os) since the
    mode states come from the oversampled contact simulation.  Free-decay
    synthesis uses native-rate coefficients (theta_sr, r_sr).
    """
    n_modes = y_final.size
    signal = np.zeros(remaining_samples, dtype=np.float64)
    n_arr = np.arange(remaining_samples, dtype=np.float64)

    freq_scale = np.ones(remaining_samples, dtype=np.float64)
    if freq_trajectory is not None and contact_samples_at_sr < freq_trajectory.size:
        traj_remaining = freq_trajectory[contact_samples_at_sr:]
        use_len = min(traj_remaining.size, remaining_samples)
        freq_scale[:use_len] = traj_remaining[:use_len] / freq

    for k in range(n_modes):
        sin_theta_os = np.sin(theta_os[k])
        if abs(sin_theta_os) < 1e-12:
            continue

        y_k = y_final[k]
        y_k_prev = y_prev_final[k]

        cos_psi = (y_k * np.cos(theta_os[k]) - r_os[k] * y_k_prev) / sin_theta_os
        sin_psi = y_k
        amplitude = np.sqrt(sin_psi**2 + cos_psi**2)
        if amplitude < 1e-15:
            continue

        phase = np.arctan2(sin_psi, cos_psi)

        effective_theta = theta_sr[k] * freq_scale
        cumulative_phase = np.cumsum(effective_theta)
        decay = r_sr[k] ** n_arr
        signal += w_out[k] * amplitude * decay * np.sin(cumulative_phase + phase)

    return signal
