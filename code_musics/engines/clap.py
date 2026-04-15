"""Multi-tap noise burst percussion engine for clap / snap sounds."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from code_musics.engines._drum_utils import (
    bandpass_noise_windowed,
    resolve_velocity_timbre,
    rng_for_note,
)
from code_musics.engines._envelopes import render_envelope
from code_musics.engines._filters import apply_zdf_svf

logger: logging.Logger = logging.getLogger(__name__)


def render(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
    freq_trajectory: np.ndarray | None = None,
) -> np.ndarray:
    """Render a multi-tap clap/snap percussion voice.

    The characteristic clap sound comes from several rapid micro-bursts
    (taps) before a longer noise body tail.  Each tap is a short
    bandpass-filtered noise burst with exponential decay.

    Optional multi-point envelope params:

    - ``body_amp_envelope``: replaces the exponential body tail decay.
      Scaled to the body-tail portion only (not the individual taps).
    - ``overall_amp_envelope``: applied to the final mixed signal before
      peak normalization — useful for gated clap shapes.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if freq <= 0:
        raise ValueError("freq must be positive")
    if freq_trajectory is not None:
        raise ValueError("pitch motion is not supported for the clap engine")

    # --- parse params ---
    n_taps = int(params.get("n_taps", 4))
    tap_spacing_s = float(params.get("tap_spacing", 0.005))
    tap_decay_s = float(params.get("tap_decay", 0.003))
    tap_crescendo = float(params.get("tap_crescendo", 0.3))
    tap_acceleration = float(params.get("tap_acceleration", 0.0))
    tap_freq_spread = float(params.get("tap_freq_spread", 0.0))
    body_decay_s = float(params.get("body_decay", 0.06))
    filter_center_ratio = float(params.get("filter_center_ratio", 1.0))
    filter_width_ratio = float(params.get("filter_width_ratio", 2.0))
    click_amount = float(params.get("click_amount", 0.08))
    click_decay_s = float(params.get("click_decay", 0.002))
    tail_filter_cutoff_raw = params.get("tail_filter_cutoff_hz")
    tail_filter_q = float(params.get("tail_filter_q", 1.5))

    body_amp_envelope_raw = params.get("body_amp_envelope")
    overall_amp_envelope_raw = params.get("overall_amp_envelope")

    # --- validate ---
    if not 1 <= n_taps <= 8:
        raise ValueError("n_taps must be between 1 and 8")
    if tap_spacing_s <= 0:
        raise ValueError("tap_spacing must be positive")
    if tap_decay_s <= 0:
        raise ValueError("tap_decay must be positive")
    if not 0.0 <= tap_crescendo <= 1.0:
        raise ValueError("tap_crescendo must be between 0 and 1")
    if not 0.0 <= tap_acceleration <= 1.0:
        raise ValueError("tap_acceleration must be between 0 and 1")
    if not 0.0 <= tap_freq_spread <= 1.0:
        raise ValueError("tap_freq_spread must be between 0 and 1")
    if body_decay_s <= 0:
        raise ValueError("body_decay must be positive")
    if filter_center_ratio <= 0:
        raise ValueError("filter_center_ratio must be positive")
    if filter_width_ratio <= 0:
        raise ValueError("filter_width_ratio must be positive")
    if click_amount < 0:
        raise ValueError("click_amount must be non-negative")
    if click_decay_s <= 0:
        raise ValueError("click_decay must be positive")
    if tail_filter_q < 0.5:
        raise ValueError("tail_filter_q must be >= 0.5")

    # --- velocity-to-timbre scaling ---
    timbre = resolve_velocity_timbre(amp, params)
    body_decay_s *= timbre.decay_scale
    filter_center_ratio *= timbre.brightness_scale

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0, dtype=np.float64)

    rng = rng_for_note(
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=sample_rate,
        params=params,
    )

    center_hz = freq * filter_center_ratio
    signal = np.zeros(n_samples, dtype=np.float64)

    # --- multi-tap generation ---
    attack_len = max(1, int(0.0001 * sample_rate))
    cumulative_offset_s = 0.0

    for i in range(n_taps):
        if i == 0:
            tap_offset = 0
        else:
            gap_s = tap_spacing_s * (
                1.0 - tap_acceleration * (i - 1) / max(1, n_taps - 2)
            )
            gap_s = max(0.0005, gap_s)
            cumulative_offset_s += gap_s
            tap_offset = int(cumulative_offset_s * sample_rate)

        tap_length = n_samples - tap_offset
        if tap_length <= 0:
            continue

        if tap_freq_spread > 0:
            tap_center = center_hz * (
                1.0 + tap_freq_spread * 0.3 * rng.uniform(-1.0, 1.0)
            )
        else:
            tap_center = center_hz

        tap_noise = rng.standard_normal(tap_length)
        tap_noise = bandpass_noise_windowed(
            tap_noise,
            sample_rate=sample_rate,
            center_hz=tap_center,
            width_ratio=filter_width_ratio,
        )

        t_tap = np.arange(tap_length, dtype=np.float64) / sample_rate
        tap_env = np.exp(-t_tap / tap_decay_s)
        tap_env[:attack_len] *= np.linspace(0.0, 1.0, attack_len)

        tap_amp = 1.0 + tap_crescendo * (i / max(1, n_taps - 1))
        signal[tap_offset : tap_offset + tap_length] += tap_amp * tap_noise * tap_env

    # --- body tail ---
    body_offset = int(cumulative_offset_s * sample_rate)
    body_length = n_samples - body_offset
    if body_length > 0:
        body_noise = rng.standard_normal(body_length)
        body_noise = bandpass_noise_windowed(
            body_noise,
            sample_rate=sample_rate,
            center_hz=center_hz,
            width_ratio=filter_width_ratio,
        )
        if tail_filter_cutoff_raw is not None:
            cutoff_hz = float(tail_filter_cutoff_raw)
            cutoff_profile = np.full(
                body_length, min(cutoff_hz, sample_rate * 0.4), dtype=np.float64
            )
            body_noise = apply_zdf_svf(
                body_noise,
                cutoff_profile=cutoff_profile,
                resonance_q=tail_filter_q,
                sample_rate=sample_rate,
                filter_mode="bandpass",
                filter_drive=0.0,
            )
        if body_amp_envelope_raw is not None:
            body_env = render_envelope(
                body_amp_envelope_raw, body_length, default_value=0.0
            )
        else:
            t_body = np.arange(body_length, dtype=np.float64) / sample_rate
            body_env = np.exp(-t_body / body_decay_s)
        signal[body_offset : body_offset + body_length] += body_noise * body_env

    # --- optional initial click ---
    if click_amount > 0:
        click_noise = rng.standard_normal(n_samples)
        click_noise = bandpass_noise_windowed(
            click_noise,
            sample_rate=sample_rate,
            center_hz=min(center_hz * 3.0, sample_rate * 0.45),
            width_ratio=1.5,
        )
        t_click = np.arange(n_samples, dtype=np.float64) / sample_rate
        click_env = np.exp(-t_click / click_decay_s)
        signal += click_amount * click_noise * click_env

    # --- optional overall amplitude envelope ---
    if overall_amp_envelope_raw is not None:
        overall_env = render_envelope(
            overall_amp_envelope_raw, n_samples, default_value=1.0
        )
        signal *= overall_env

    # --- peak-normalize and scale by amp ---
    peak = np.max(np.abs(signal))
    if peak > 1e-9:
        signal = signal / peak
    return amp * signal.astype(np.float64)
