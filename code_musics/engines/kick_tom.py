"""Hybrid kick/tom drum synthesis engine."""

from __future__ import annotations

import logging
from typing import Any

import numba
import numpy as np

from code_musics.engines._drum_utils import (
    bandpass_noise,
    integrated_phase,
    resolve_velocity_timbre,
    rng_for_note,
)
from code_musics.engines._dsp_utils import fm_modulate
from code_musics.engines._envelopes import render_envelope
from code_musics.engines._filters import _SUPPORTED_FILTER_MODES, apply_filter
from code_musics.engines._waveshaper import ALGORITHM_NAMES, apply_waveshaper

logger: logging.Logger = logging.getLogger(__name__)

_VALID_BODY_FILTER_MODES = _SUPPORTED_FILTER_MODES - {"notch"}
_VALID_BODY_MODES = {"oscillator", "resonator"}


@numba.njit(cache=True)
def _resonator_body(
    excitation: np.ndarray,
    freq_profile: np.ndarray,
    q: float,
    sample_rate: int,
) -> np.ndarray:
    """Time-varying 2-pole bandpass resonator for kick/tom body synthesis.

    Uses the ZDF/TPT state-variable topology (same integrator structure as
    the project's main SVF) with per-sample coefficient updates from
    ``freq_profile``.  The trapezoidal integrators carry state cleanly under
    pitch modulation — unlike bilinear biquads, whose internal `x1/x2/y1/y2`
    memory accumulates phase error during fast sweeps.

    Parameter mapping to match the previous bilinear-biquad interface:
    - Cutoff: ``freq_profile[i]``
    - Damping (1/Q): ``1.0 / q``  (matches the `alpha = sin_w0 / (2q)`
      RBJ bandpass Q in the linear regime near cutoff)
    - Gain normalization: scale output by ``sin(w0)/(2q)`` so the constant-
      0dB-peak behavior of the previous implementation is preserved —
      a raw SVF bandpass has unity gain at its peak regardless of Q,
      whereas the RBJ constant-0dB-peak form has peak gain = Q.  We
      re-apply the `sin(w0)/(2q)` envelope to match the kick/tom body
      loudness calibration the surrounding code expects.
    """
    n = excitation.shape[0]
    out = np.zeros(n, dtype=np.float64)
    low_state = 0.0
    band_state = 0.0
    damping = 1.0 / max(q, 1e-6)
    pi_over_sr = np.pi / sample_rate

    for i in range(n):
        fc = freq_profile[i]
        # Clamp cutoff to a safe fraction of Nyquist (matches _filters
        # convention) so tan() stays well-behaved even on edge-case sweeps.
        fc_clamped = fc if fc < 0.45 * sample_rate else 0.45 * sample_rate
        g = np.tan(pi_over_sr * fc_clamped)

        x = excitation[i]
        high = (x - (2.0 * damping + g) * band_state - low_state) / (
            1.0 + 2.0 * damping * g + g * g
        )
        band = g * high + band_state
        low = g * band + low_state
        band_state = band + g * high
        low_state = low + g * band

        # Rescale band output to match the constant-0dB-peak gain of the
        # original RBJ biquad (peak gain = Q there; unity in raw SVF).
        # alpha = sin(w0)/(2q) where w0 = 2*pi*fc/sr — compute directly from g.
        # At small fc, g ≈ pi*fc/sr, so 2*g/(1+g²) ≈ 2*pi*fc/sr ≈ sin(w0) to 1st order.
        sin_w0_approx = 2.0 * g / (1.0 + g * g)
        alpha_rbj = sin_w0_approx / (2.0 * q)
        out[i] = band * alpha_rbj

    return out


def render(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
    freq_trajectory: np.ndarray | None = None,
) -> np.ndarray:
    """Render an electro-style kick/tom voice with internal sweep and transient."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if freq <= 0:
        raise ValueError("freq must be positive")

    body_mode = str(params.get("body_mode", "oscillator")).lower()
    body_decay_s = float(params.get("body_decay", params.get("decay", 0.26)))
    pitch_sweep_amount_ratio = float(params.get("pitch_sweep_amount_ratio", 2.5))
    pitch_sweep_decay_s = float(params.get("pitch_sweep_decay", 0.042))
    body_wave = str(params.get("body_wave", "sine")).lower()
    body_tone_ratio = float(params.get("body_tone_ratio", 0.16))
    body_punch_ratio = float(params.get("body_punch_ratio", 0.20))
    overtone_amount = float(params.get("overtone_amount", 0.10))
    overtone_ratio = float(params.get("overtone_ratio", 1.9))
    overtone_decay_s = float(params.get("overtone_decay", 0.11))
    click_amount = float(params.get("click_amount", 0.08))
    click_decay_s = float(params.get("click_decay", 0.007))
    click_tone_hz = float(params.get("click_tone_hz", 3_200.0))
    noise_amount = float(params.get("noise_amount", 0.02))
    noise_decay_s = float(params.get("noise_decay", 0.028))
    noise_bandpass_hz = float(params.get("noise_bandpass_hz", 1_100.0))

    body_amp_envelope_raw = params.get("body_amp_envelope")
    pitch_envelope_raw = params.get("pitch_envelope")
    overtone_amp_envelope_raw = params.get("overtone_amp_envelope")

    body_filter_mode: str | None = params.get("body_filter_mode")
    body_filter_cutoff_hz = float(params.get("body_filter_cutoff_hz", 2000.0))
    body_filter_q = float(params.get("body_filter_q", 0.707))
    body_filter_drive = float(params.get("body_filter_drive", 0.0))
    body_filter_topology = str(params.get("body_filter_topology", "svf")).lower()
    body_bass_compensation = float(params.get("body_bass_compensation", 0.5))
    body_filter_envelope_raw = params.get("body_filter_envelope")

    # FM body params
    body_fm_ratio_raw = params.get("body_fm_ratio")
    body_fm_ratio: float | None = (
        float(body_fm_ratio_raw) if body_fm_ratio_raw is not None else None
    )
    body_fm_index = float(params.get("body_fm_index", 2.0))
    body_fm_feedback = float(params.get("body_fm_feedback", 0.0))
    body_fm_index_envelope_raw = params.get("body_fm_index_envelope")

    # Waveshaper body params
    body_distortion_name: str | None = params.get("body_distortion")
    body_distortion_drive = float(params.get("body_distortion_drive", 0.5))
    body_distortion_mix = float(params.get("body_distortion_mix", 1.0))
    body_distortion_drive_envelope_raw = params.get("body_distortion_drive_envelope")

    # Deprecation warnings for removed internal drive/lowpass stages.
    if "drive_ratio" in params or "drive" in params:
        logger.warning(
            "kick_tom: drive_ratio is deprecated; use EffectSpec('saturation', ...) "
            "on the voice instead"
        )
    if "post_lowpass_hz" in params:
        logger.warning(
            "kick_tom: post_lowpass_hz is deprecated; use EffectSpec('eq', ...) "
            "on the voice instead"
        )

    if body_mode not in _VALID_BODY_MODES:
        raise ValueError(
            f"body_mode must be one of {sorted(_VALID_BODY_MODES)}, got {body_mode!r}"
        )
    if body_decay_s <= 0:
        raise ValueError("body_decay must be positive")
    if pitch_sweep_amount_ratio <= 0:
        raise ValueError("pitch_sweep_amount_ratio must be positive")
    if pitch_sweep_decay_s <= 0:
        raise ValueError("pitch_sweep_decay must be positive")
    if body_wave not in {"sine", "triangle", "sine_clip"}:
        raise ValueError("body_wave must be 'sine', 'triangle', or 'sine_clip'")
    if not 0.0 <= body_tone_ratio <= 1.0:
        raise ValueError("body_tone_ratio must be between 0 and 1")
    if body_punch_ratio < 0:
        raise ValueError("body_punch_ratio must be non-negative")
    if overtone_amount < 0:
        raise ValueError("overtone_amount must be non-negative")
    if overtone_ratio <= 0:
        raise ValueError("overtone_ratio must be positive")
    if overtone_decay_s <= 0:
        raise ValueError("overtone_decay must be positive")
    if click_amount < 0:
        raise ValueError("click_amount must be non-negative")
    if click_decay_s <= 0:
        raise ValueError("click_decay must be positive")
    if click_tone_hz <= 0:
        raise ValueError("click_tone_hz must be positive")
    if noise_amount < 0:
        raise ValueError("noise_amount must be non-negative")
    if noise_decay_s <= 0:
        raise ValueError("noise_decay must be positive")
    if noise_bandpass_hz <= 0:
        raise ValueError("noise_bandpass_hz must be positive")
    if (
        body_filter_mode is not None
        and body_filter_mode not in _VALID_BODY_FILTER_MODES
    ):
        raise ValueError(
            f"body_filter_mode must be one of {sorted(_VALID_BODY_FILTER_MODES)} "
            f"or None, got {body_filter_mode!r}"
        )
    if body_filter_cutoff_hz <= 0:
        raise ValueError("body_filter_cutoff_hz must be positive")
    if body_filter_q < 0.5:
        raise ValueError("body_filter_q must be >= 0.5")
    if body_filter_drive < 0:
        raise ValueError("body_filter_drive must be non-negative")
    if body_fm_ratio is not None and body_fm_ratio <= 0:
        raise ValueError("body_fm_ratio must be positive")
    if body_fm_index < 0:
        raise ValueError("body_fm_index must be non-negative")
    if body_distortion_name is not None and body_distortion_name not in ALGORITHM_NAMES:
        raise ValueError(
            f"body_distortion must be one of {sorted(ALGORITHM_NAMES)} "
            f"or None, got {body_distortion_name!r}"
        )
    if not 0.0 <= body_distortion_drive <= 1.0:
        raise ValueError("body_distortion_drive must be in [0, 1]")
    if not 0.0 <= body_distortion_mix <= 1.0:
        raise ValueError("body_distortion_mix must be in [0, 1]")

    # --- velocity-to-timbre scaling ---
    timbre = resolve_velocity_timbre(amp, params)
    body_decay_s *= timbre.decay_scale
    click_tone_hz *= timbre.brightness_scale
    noise_bandpass_hz *= timbre.brightness_scale
    if body_fm_ratio is not None:
        body_fm_index *= timbre.harmonic_scale
    overtone_amount *= timbre.harmonic_scale

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0, dtype=np.float64)

    time = np.arange(n_samples, dtype=np.float64) / sample_rate

    base_freq_profile = _resolve_base_freq_profile(
        freq=freq,
        n_samples=n_samples,
        freq_trajectory=freq_trajectory,
    )

    if pitch_envelope_raw is not None:
        sweep_profile = render_envelope(
            pitch_envelope_raw, n_samples, default_value=1.0
        )
    else:
        sweep_profile = 1.0 + (pitch_sweep_amount_ratio - 1.0) * np.exp(
            -time / pitch_sweep_decay_s
        )
    freq_profile = base_freq_profile * sweep_profile

    # Body synthesis: resonator mode or oscillator mode
    if body_mode == "resonator":
        if body_fm_ratio is not None:
            logger.warning(
                "kick_tom: body_fm_ratio is ignored in resonator body_mode; "
                "FM synthesis is only available in oscillator mode"
            )

        # Build excitation: short noise impulse shaped by punch envelope
        rng_exc = rng_for_note(
            freq=freq,
            duration=duration,
            amp=amp,
            sample_rate=sample_rate,
            params=params,
        )
        impulse_samples = max(1, int(0.001 * sample_rate))
        excitation = np.zeros(n_samples, dtype=np.float64)
        impulse_noise = rng_exc.standard_normal(impulse_samples)
        punch_env_impulse = 1.0 + body_punch_ratio * np.exp(
            -np.arange(impulse_samples, dtype=np.float64) / (0.001 * sample_rate)
        )
        excitation[:impulse_samples] = impulse_noise * punch_env_impulse

        # Compute Q from decay time: higher Q = longer ring
        early_end = max(1, n_samples // 4)
        mean_freq = float(np.mean(freq_profile[:early_end]))
        resonator_q = max(1.0, np.pi * mean_freq * body_decay_s)

        body_signal = _resonator_body(
            excitation,
            freq_profile.astype(np.float64),
            resonator_q,
            sample_rate,
        )

        # Apply body amp envelope
        if body_amp_envelope_raw is not None:
            body_env = render_envelope(
                body_amp_envelope_raw, n_samples, default_value=0.0
            )
        else:
            body_env = np.exp(-time / body_decay_s)

        body = body_signal * body_env

    else:
        # Oscillator mode: FM-modulated or standard oscillator
        if body_fm_ratio is not None:
            if body_fm_index_envelope_raw is not None:
                index_env = render_envelope(
                    body_fm_index_envelope_raw, n_samples, default_value=1.0
                )
            else:
                index_env = np.exp(-time / 0.05)
            fundamental = fm_modulate(
                freq_profile,
                mod_ratio=body_fm_ratio,
                mod_index=body_fm_index,
                sample_rate=sample_rate,
                feedback=body_fm_feedback,
                index_envelope=index_env,
            )
        else:
            fundamental_phase = integrated_phase(freq_profile, sample_rate=sample_rate)
            fundamental = _oscillator(body_wave=body_wave, phase=fundamental_phase)

        if body_amp_envelope_raw is not None:
            body_env = render_envelope(
                body_amp_envelope_raw, n_samples, default_value=0.0
            )
        else:
            body_env = np.exp(-time / body_decay_s)

        punch_env = 1.0 + body_punch_ratio * np.exp(-time / 0.018)

        harmonic_phase = integrated_phase(freq_profile, sample_rate=sample_rate)
        second_harmonic = np.sin(2.0 * harmonic_phase) * np.exp(
            -time / max(0.02, body_decay_s * 0.55)
        )
        body = (
            (
                ((1.0 - body_tone_ratio) * fundamental)
                + (body_tone_ratio * second_harmonic)
            )
            * body_env
            * punch_env
        )

    # Waveshaper distortion on body (before filter)
    if body_distortion_name is not None:
        if body_distortion_drive_envelope_raw is not None:
            drive_env = render_envelope(
                body_distortion_drive_envelope_raw, n_samples, default_value=1.0
            )
        else:
            drive_env = None
        body = apply_waveshaper(
            body,
            algorithm=body_distortion_name,
            drive=body_distortion_drive,
            drive_envelope=drive_env,
            mix=body_distortion_mix,
        )

    if body_filter_mode is not None:
        if body_filter_envelope_raw is not None:
            cutoff_profile = render_envelope(
                body_filter_envelope_raw,
                n_samples,
                default_value=body_filter_cutoff_hz,
            )
        else:
            cutoff_profile = np.full(n_samples, body_filter_cutoff_hz)
        body = apply_filter(
            body,
            cutoff_profile=cutoff_profile,
            resonance_q=body_filter_q,
            sample_rate=sample_rate,
            filter_mode=body_filter_mode,
            filter_drive=body_filter_drive,
            filter_topology=body_filter_topology,
            bass_compensation=body_bass_compensation,
        )

    overtone_phase = integrated_phase(
        freq_profile * overtone_ratio, sample_rate=sample_rate
    )

    if overtone_amp_envelope_raw is not None:
        overtone_env = render_envelope(
            overtone_amp_envelope_raw, n_samples, default_value=0.0
        )
    else:
        overtone_env = np.exp(-time / overtone_decay_s)

    overtone = overtone_amount * np.sin(overtone_phase) * overtone_env

    rng = rng_for_note(
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=sample_rate,
        params=params,
    )
    click = _transient_noise(
        rng=rng,
        n_samples=n_samples,
        sample_rate=sample_rate,
        center_hz=click_tone_hz,
        decay_seconds=click_decay_s,
        emphasis=2.4,
    )
    noise = _transient_noise(
        rng=rng,
        n_samples=n_samples,
        sample_rate=sample_rate,
        center_hz=noise_bandpass_hz,
        decay_seconds=noise_decay_s,
        emphasis=1.0,
    )

    signal = body + overtone + (click_amount * click) + (noise_amount * noise)

    peak = np.max(np.abs(signal))
    if peak > 1e-9:
        signal = signal / peak
    return amp * signal.astype(np.float64)


def _resolve_base_freq_profile(
    *,
    freq: float,
    n_samples: int,
    freq_trajectory: np.ndarray | None,
) -> np.ndarray:
    if freq_trajectory is None:
        return np.full(n_samples, freq, dtype=np.float64)

    resolved = np.asarray(freq_trajectory, dtype=np.float64)
    if resolved.ndim != 1:
        raise ValueError("freq_trajectory must be one-dimensional")
    if resolved.size != n_samples:
        raise ValueError("freq_trajectory length must match note duration")
    if np.any(resolved <= 0):
        raise ValueError("freq_trajectory values must be positive")
    return resolved


def _oscillator(*, body_wave: str, phase: np.ndarray) -> np.ndarray:
    if body_wave == "sine":
        return np.sin(phase)
    if body_wave == "triangle":
        return (2.0 / np.pi) * np.arcsin(np.sin(phase))
    return np.tanh(1.8 * np.sin(phase))


def _transient_noise(
    *,
    rng: np.random.Generator,
    n_samples: int,
    sample_rate: int,
    center_hz: float,
    decay_seconds: float,
    emphasis: float,
) -> np.ndarray:
    raw = rng.standard_normal(n_samples)
    shaped = bandpass_noise(raw, sample_rate=sample_rate, center_hz=center_hz)
    time = np.arange(n_samples, dtype=np.float64) / sample_rate
    envelope = np.exp(-time / decay_seconds)
    envelope[: max(1, n_samples // 512)] *= emphasis
    return shaped * envelope
