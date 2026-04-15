"""Frequency modulation synthesis engine."""

from __future__ import annotations

import math
from typing import Any

import numba
import numpy as np

from code_musics.engines._dsp_utils import (
    apply_analog_post_processing,
    apply_note_jitter,
    apply_voice_card,
    build_drift,
    extract_analog_params,
    rng_for_note,
)


@numba.njit(cache=True)
def _fm_sample_loop(
    signal: np.ndarray,
    carrier_phase_increment: np.ndarray,
    mod_phase_increment: np.ndarray,
    mod_index: float,
    feedback: float,
    index_decay_samples: int,
    sustain_scale: float,
    n_samples: int,
) -> None:
    carrier_phase = 0.0
    mod_phase = 0.0
    previous_feedback_sample = 0.0
    decay_denom = max(1, index_decay_samples - 1)

    for i in range(n_samples):
        if index_decay_samples <= 0 or i >= index_decay_samples:
            mod_envelope = sustain_scale
        else:
            mod_envelope = 1.0 + (sustain_scale - 1.0) * (i / decay_denom)

        current_index = mod_index * mod_envelope

        modulator_sample = math.sin(mod_phase + feedback * previous_feedback_sample)
        previous_feedback_sample = modulator_sample

        signal[i] = math.sin(carrier_phase + current_index * modulator_sample)

        carrier_phase += carrier_phase_increment[i]
        mod_phase += mod_phase_increment[i]


def render(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
    freq_trajectory: np.ndarray | None = None,
) -> np.ndarray:
    """Render a compact 2-operator FM voice.

    The implementation is frequency-first: the carrier and modulator operate
    directly on Hertz-derived phase increments, which keeps it compatible with
    alternate tuning systems.
    """
    if duration <= 0:
        return np.zeros(0)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if freq <= 0:
        raise ValueError("freq must be positive")

    carrier_ratio = float(params.get("carrier_ratio", 1.0))
    mod_ratio = float(params.get("mod_ratio", 1.0))
    mod_index = float(params.get("mod_index", 1.5))
    feedback = float(params.get("feedback", 0.0))
    index_decay = float(params.get("index_decay", 0.0))
    index_sustain = float(params.get("index_sustain", 0.5))
    analog = extract_analog_params(params)
    pitch_drift = analog["pitch_drift"]
    analog_jitter = analog["analog_jitter"]
    noise_floor_level = analog["noise_floor"]
    drift_rate_hz = analog["drift_rate_hz"]

    if carrier_ratio <= 0:
        raise ValueError("carrier_ratio must be positive")
    if mod_ratio <= 0:
        raise ValueError("mod_ratio must be positive")
    if mod_index < 0:
        raise ValueError("mod_index must be non-negative")
    if index_decay < 0:
        raise ValueError("index_decay must be non-negative")

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0)

    # --- Analog character: RNG, jitter, drift ---
    rng = rng_for_note(
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=sample_rate,
    )
    jittered = apply_note_jitter(params, rng, analog_jitter)
    amp_jitter_db = float(jittered.get("_amp_jitter_db", 0.0))

    # Build base frequency profile
    if freq_trajectory is None:
        freq_profile = np.full(n_samples, freq, dtype=np.float64)
    else:
        freq_trajectory = np.asarray(freq_trajectory, dtype=np.float64)
        if freq_trajectory.ndim != 1:
            raise ValueError("freq_trajectory must be one-dimensional")
        if freq_trajectory.size != n_samples:
            raise ValueError("freq_trajectory length must match note duration")
        freq_profile = freq_trajectory

    # Voice card calibration — persistent per-voice character (no filter in FM)
    freq_profile, amp, _ = apply_voice_card(
        params,
        voice_card_amount=analog["voice_card"],
        freq_profile=freq_profile,
        amp=amp,
    )

    # Apply pitch drift to both carrier and modulator
    if pitch_drift > 0:
        start_phase = float(jittered.get("_phase_offset", 0.0))
        drift_multiplier = build_drift(
            n_samples=n_samples,
            drift_amount=pitch_drift,
            drift_rate_hz=drift_rate_hz,
            duration=duration,
            phase_offset=start_phase,
            rng=rng,
        )
        freq_profile = freq_profile * drift_multiplier

    carrier_phase_increment = 2.0 * np.pi * freq_profile * carrier_ratio / sample_rate
    mod_phase_increment = 2.0 * np.pi * freq_profile * mod_ratio / sample_rate

    signal = np.empty(n_samples, dtype=np.float64)

    index_decay_samples = int(index_decay * sample_rate)
    index_decay_samples = min(max(index_decay_samples, 0), n_samples)
    sustain_scale = max(0.0, index_sustain)

    _fm_sample_loop(
        signal,
        carrier_phase_increment,
        mod_phase_increment,
        mod_index,
        feedback,
        index_decay_samples,
        sustain_scale,
        n_samples,
    )

    signal = apply_analog_post_processing(
        signal,
        rng=rng,
        amp_jitter_db=amp_jitter_db,
        noise_floor_level=noise_floor_level,
        sample_rate=sample_rate,
        n_samples=n_samples,
    )

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak
    return amp * signal
