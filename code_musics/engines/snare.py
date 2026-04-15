"""909-inspired snare drum synthesis engine.

Three-layer architecture: pitched body with sweep, comb-filtered wire buzz,
and broadband transient click.
"""

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
from code_musics.engines._filters import apply_zdf_svf
from code_musics.engines._waveshaper import ALGORITHM_NAMES, apply_waveshaper

logger: logging.Logger = logging.getLogger(__name__)

_VALID_WIRE_NOISE_MODES: frozenset[str] = frozenset({"white", "colored"})


def render(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
    freq_trajectory: np.ndarray | None = None,
) -> np.ndarray:
    """Render a snare drum with body tone, comb-filtered wire, and click transient."""
    if freq_trajectory is not None:
        raise ValueError("snare engine does not support freq_trajectory (pitch motion)")
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if freq <= 0:
        raise ValueError("freq must be positive")

    # --- parse params ---
    body_decay_s = float(params.get("body_decay", 0.12))
    body_overtone_ratio = float(params.get("body_overtone_ratio", 1.6))
    body_sweep_ratio = float(params.get("body_sweep_ratio", 1.8))
    body_sweep_decay_s = float(params.get("body_sweep_decay", 0.015))
    wire_decay_s = float(params.get("wire_decay", 0.18))
    wire_center_ratio = float(params.get("wire_center_ratio", 3.0))
    wire_q = float(params.get("wire_q", 0.8))
    comb_amount = float(params.get("comb_amount", 0.45))
    body_mix = float(params.get("body_mix", 0.5))
    wire_mix = float(params.get("wire_mix", 0.5))
    click_amount = float(params.get("click_amount", 0.15))
    click_decay_s = float(params.get("click_decay", 0.005))

    body_amp_envelope_raw = params.get("body_amp_envelope")
    wire_amp_envelope_raw = params.get("wire_amp_envelope")
    body_pitch_envelope_raw = params.get("body_pitch_envelope")
    wire_filter_envelope_raw = params.get("wire_filter_envelope")

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

    # Wire noise mode
    wire_noise_mode = str(params.get("wire_noise_mode", "white")).lower()

    # --- validate ---
    if body_decay_s <= 0:
        raise ValueError("body_decay must be positive")
    if body_overtone_ratio <= 0:
        raise ValueError("body_overtone_ratio must be positive")
    if body_sweep_ratio <= 0:
        raise ValueError("body_sweep_ratio must be positive")
    if body_sweep_decay_s <= 0:
        raise ValueError("body_sweep_decay must be positive")
    if wire_decay_s <= 0:
        raise ValueError("wire_decay must be positive")
    if wire_center_ratio <= 0:
        raise ValueError("wire_center_ratio must be positive")
    if wire_q < 0.5:
        raise ValueError("wire_q must be >= 0.5")
    if not 0.0 <= comb_amount <= 1.0:
        raise ValueError("comb_amount must be between 0 and 1")
    if body_mix < 0:
        raise ValueError("body_mix must be non-negative")
    if wire_mix < 0:
        raise ValueError("wire_mix must be non-negative")
    if click_amount < 0:
        raise ValueError("click_amount must be non-negative")
    if click_decay_s <= 0:
        raise ValueError("click_decay must be positive")
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
    if wire_noise_mode not in _VALID_WIRE_NOISE_MODES:
        raise ValueError(
            f"wire_noise_mode must be one of {sorted(_VALID_WIRE_NOISE_MODES)}, "
            f"got {wire_noise_mode!r}"
        )

    # --- velocity-to-timbre scaling ---
    timbre = resolve_velocity_timbre(amp, params)
    body_decay_s *= timbre.decay_scale
    wire_decay_s *= timbre.decay_scale
    wire_center_ratio *= timbre.brightness_scale
    comb_amount = min(1.0, comb_amount * timbre.harmonic_scale)
    if body_fm_ratio is not None:
        body_fm_index *= timbre.harmonic_scale

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0, dtype=np.float64)

    time = np.arange(n_samples, dtype=np.float64) / sample_rate
    rng = rng_for_note(
        freq=freq, duration=duration, amp=amp, sample_rate=sample_rate, params=params
    )

    # --- 1. Body: pitched tone with sweep ---
    if body_pitch_envelope_raw is not None:
        sweep_profile = render_envelope(
            body_pitch_envelope_raw, n_samples, default_value=1.0
        )
    else:
        sweep_profile = 1.0 + (body_sweep_ratio - 1.0) * np.exp(
            -time / body_sweep_decay_s
        )
    freq_profile = freq * sweep_profile

    # Body fundamental: FM-modulated or standard sine
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
        fundamental = np.sin(fundamental_phase)

    overtone_phase = integrated_phase(
        freq_profile * body_overtone_ratio, sample_rate=sample_rate
    )
    overtone = np.sin(overtone_phase)

    body = fundamental + 0.35 * overtone

    # Waveshaper distortion on body (before envelope)
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

    if body_amp_envelope_raw is not None:
        body_env = render_envelope(body_amp_envelope_raw, n_samples, default_value=0.0)
    else:
        body_env = np.exp(-time / body_decay_s)
    body = body * body_env

    # --- 2. Wire: comb-filtered noise ---
    raw_noise = rng.standard_normal(n_samples)

    # Colored wire noise: highpass filter before comb for more realistic character
    if wire_noise_mode == "colored":
        hp_cutoff = np.full(n_samples, 500.0, dtype=np.float64)
        raw_noise = apply_zdf_svf(
            raw_noise,
            cutoff_profile=hp_cutoff,
            resonance_q=0.707,
            sample_rate=sample_rate,
            filter_mode="highpass",
            filter_drive=0.0,
        )

    delay_samples = max(1, int(sample_rate / freq))
    wire = _comb_filter(raw_noise, delay_samples, comb_amount)

    cutoff_hz = min(freq * wire_center_ratio, sample_rate * 0.45)
    if wire_filter_envelope_raw is not None:
        cutoff_profile = render_envelope(
            wire_filter_envelope_raw, n_samples, default_value=cutoff_hz
        )
    else:
        cutoff_profile = np.full(n_samples, cutoff_hz, dtype=np.float64)
    wire = apply_zdf_svf(
        wire,
        cutoff_profile=cutoff_profile,
        resonance_q=wire_q,
        sample_rate=sample_rate,
        filter_mode="bandpass",
        filter_drive=0.0,
    )

    if wire_amp_envelope_raw is not None:
        wire_env = render_envelope(wire_amp_envelope_raw, n_samples, default_value=0.0)
    else:
        wire_env = np.exp(-time / wire_decay_s)
    wire = wire * wire_env

    # --- 3. Click: broadband transient ---
    click_noise = rng.standard_normal(n_samples)
    click = bandpass_noise(click_noise, sample_rate=sample_rate, center_hz=3200.0)
    click_env = np.exp(-time / click_decay_s)
    # 0.1ms attack ramp to prevent pops
    attack_samples = max(1, int(sample_rate * 0.0001))
    attack_ramp = np.ones(n_samples, dtype=np.float64)
    attack_ramp[:attack_samples] = np.linspace(0.0, 1.0, attack_samples)
    click = click * click_env * attack_ramp

    # --- 4. Mix and output ---
    signal = body_mix * body + wire_mix * wire + click_amount * click

    peak = np.max(np.abs(signal))
    if peak > 1e-9:
        signal = signal / peak
    return (amp * signal).astype(np.float64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _comb_filter(signal: np.ndarray, delay_samples: int, feedback: float) -> np.ndarray:
    """Comb filter for wire resonance at the snare body pitch."""
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        if i < delay_samples:
            out[i] = signal[i]
        else:
            out[i] = signal[i] + feedback * out[i - delay_samples]
    return out
