"""Filtered-stack synthesis engine."""

from __future__ import annotations

from typing import Any

import numpy as np

from code_musics.engines._dsp_utils import (
    apply_analog_post_processing,
    apply_note_jitter,
    apply_voice_card,
    build_cutoff_drift,
    build_drift,
    extract_analog_params,
    nyquist_fade,
    rng_for_note,
)
from code_musics.engines._filters import _SUPPORTED_FILTER_MODES, apply_zdf_svf


def render(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
    freq_trajectory: np.ndarray | None = None,
) -> np.ndarray:
    """Render a harmonic-rich source shaped by a ZDF state-variable filter."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    waveform = str(params.get("waveform", "saw")).lower()
    n_harmonics = int(params.get("n_harmonics", 12))
    cutoff_hz = float(params.get("cutoff_hz", 1_800.0))
    keytrack = float(params.get("keytrack", 0.0))
    reference_freq_hz = float(params.get("reference_freq_hz", 220.0))
    resonance_q = float(params.get("resonance_q", 0.707))
    filter_env_amount = float(params.get("filter_env_amount", 0.0))
    filter_env_decay = float(params.get("filter_env_decay", 0.18))
    pulse_width = float(params.get("pulse_width", 0.5))
    filter_mode = str(params.get("filter_mode", "lowpass")).lower()
    filter_drive = float(params.get("filter_drive", 0.0))
    filter_even_harmonics = float(params.get("filter_even_harmonics", 0.0))
    analog = extract_analog_params(params)
    pitch_drift = analog["pitch_drift"]
    analog_jitter = analog["analog_jitter"]
    noise_floor_level = analog["noise_floor"]
    drift_rate_hz = analog["drift_rate_hz"]
    cutoff_drift_amount = analog["cutoff_drift"]

    if n_harmonics < 1:
        raise ValueError("n_harmonics must be at least 1")
    if cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be positive")
    if reference_freq_hz <= 0:
        raise ValueError("reference_freq_hz must be positive")
    if filter_env_decay <= 0:
        raise ValueError("filter_env_decay must be positive")
    if not 0.0 < pulse_width < 1.0:
        raise ValueError("pulse_width must be between 0 and 1")
    if filter_mode not in _SUPPORTED_FILTER_MODES:
        raise ValueError(
            f"Unsupported filter_mode: {filter_mode!r}. "
            "Use 'lowpass', 'bandpass', 'highpass', or 'notch'."
        )
    if filter_drive < 0:
        raise ValueError("filter_drive must be non-negative")

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0)

    t = np.linspace(0.0, duration, n_samples, endpoint=False)
    if freq_trajectory is not None:
        freq_trajectory = np.asarray(freq_trajectory, dtype=np.float64)
        if freq_trajectory.ndim != 1:
            raise ValueError("freq_trajectory must be one-dimensional")
        if freq_trajectory.size != n_samples:
            raise ValueError("freq_trajectory length must match note duration")
        freq_profile = freq_trajectory
    else:
        freq_profile = np.full(n_samples, freq, dtype=np.float64)

    # --- Analog character: RNG, jitter, drift ---
    rng = rng_for_note(
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=sample_rate,
    )
    # Voice card calibration — persistent per-voice character
    freq_profile, amp, cutoff_hz = apply_voice_card(
        params,
        voice_card_amount=analog["voice_card"],
        freq_profile=freq_profile,
        amp=amp,
        cutoff_hz=cutoff_hz,
    )

    jittered = apply_note_jitter(params, rng, analog_jitter)
    cutoff_hz = float(jittered.get("cutoff_hz", cutoff_hz))
    resonance_q = float(jittered.get("resonance_q", resonance_q))
    filter_env_decay = float(jittered.get("filter_env_decay", filter_env_decay))
    start_phase = float(jittered.get("_phase_offset", 0.0))
    amp_jitter_db = float(jittered.get("_amp_jitter_db", 0.0))

    if pitch_drift > 0:
        drift_multiplier = build_drift(
            n_samples=n_samples,
            drift_amount=pitch_drift,
            drift_rate_hz=drift_rate_hz,
            duration=duration,
            phase_offset=start_phase,
            rng=rng,
        )
        freq_profile = freq_profile * drift_multiplier

    # Build the raw additive signal (unfiltered sum of harmonics).
    signal = np.zeros(n_samples, dtype=np.float64)
    power_estimate = 0.0
    nyquist_hz = sample_rate / 2.0

    for harmonic_index in range(1, n_harmonics + 1):
        partial_freq_profile = freq_profile * harmonic_index
        if np.min(partial_freq_profile) >= nyquist_hz:
            break

        harmonic_weight = _waveform_weight(waveform, harmonic_index, pulse_width)
        if harmonic_weight == 0.0:
            continue

        anti_alias_weight = nyquist_fade(partial_freq_profile, nyquist_hz)
        if np.max(anti_alias_weight) <= 0.0:
            continue

        phase = (
            np.cumsum(
                np.concatenate(
                    [
                        np.zeros(1, dtype=np.float64),
                        2.0 * np.pi * partial_freq_profile[:-1] / sample_rate,
                    ]
                )
            )
            + start_phase * harmonic_index
        )
        partial_weight = harmonic_weight * anti_alias_weight
        signal += partial_weight * np.sin(phase)
        power_estimate += 0.5 * float(np.mean(np.square(partial_weight)))

    # RMS-normalize the additive stack before filtering.
    if power_estimate > 0.0:
        signal = signal / np.sqrt(power_estimate)

    # Build per-sample cutoff profile with keytracking and filter envelope.
    cutoff_envelope = 1.0 + filter_env_amount * np.exp(-t / filter_env_decay)
    cutoff_envelope = np.maximum(cutoff_envelope, 0.05)
    keytracked_cutoff = cutoff_hz * np.power(freq_profile / reference_freq_hz, keytrack)
    cutoff_profile = np.clip(
        keytracked_cutoff * cutoff_envelope, 20.0, nyquist_hz * 0.98
    )

    if cutoff_drift_amount > 0:
        cutoff_rng = rng_for_note(
            freq=freq,
            duration=duration,
            amp=amp,
            sample_rate=sample_rate,
            extra_seed="cutoff_drift",
        )
        cutoff_modulation = build_cutoff_drift(
            n_samples,
            amount_cents=30.0 * cutoff_drift_amount,
            rate_hz=0.3,
            rng=cutoff_rng,
            sample_rate=sample_rate,
        )
        cutoff_profile = np.clip(
            cutoff_profile * cutoff_modulation, 20.0, nyquist_hz * 0.98
        )

    filtered = apply_zdf_svf(
        signal,
        cutoff_profile=cutoff_profile,
        resonance_q=resonance_q,
        sample_rate=sample_rate,
        filter_mode=filter_mode,
        filter_drive=filter_drive,
        filter_even_harmonics=filter_even_harmonics,
    )

    filtered = apply_analog_post_processing(
        filtered,
        rng=rng,
        amp_jitter_db=amp_jitter_db,
        noise_floor_level=noise_floor_level,
        sample_rate=sample_rate,
        n_samples=n_samples,
    )

    # Peak-normalize after filtering (filter sweep causes uneven amplitude).
    peak = np.max(np.abs(filtered))
    if peak > 1e-9:
        filtered /= peak

    return amp * filtered


def _waveform_weight(waveform: str, harmonic_index: int, pulse_width: float) -> float:
    """Return a signed harmonic weight for a basic oscillator source."""
    if waveform == "saw":
        return 1.0 / harmonic_index
    if waveform == "square":
        if harmonic_index % 2 == 0:
            return 0.0
        return 1.0 / harmonic_index
    if waveform == "pulse":
        return np.sin(np.pi * harmonic_index * pulse_width) / (np.pi * harmonic_index)
    if waveform == "triangle":
        if harmonic_index % 2 == 0:
            return 0.0
        sign = -1.0 if ((harmonic_index - 1) // 2) % 2 else 1.0
        return sign / (harmonic_index * harmonic_index)

    raise ValueError(f"Unsupported waveform: {waveform}")
