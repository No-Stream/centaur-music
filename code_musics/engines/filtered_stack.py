"""Filtered-stack synthesis engine."""

from __future__ import annotations

from typing import Any

import numpy as np

from code_musics.engines._dsp_utils import (
    apply_analog_post_processing,
    apply_cutoff_cv_dither,
    apply_filter_oversampled,
    apply_note_jitter,
    apply_pitch_cv_dither,
    apply_voice_card,
    apply_voice_card_post_offsets,
    build_cutoff_drift,
    build_drift,
    build_keytracked_cutoff_profile,
    extract_analog_params,
    nyquist_fade,
    resolve_quality_mode,
    rng_for_note,
)
from code_musics.engines._filters import _SUPPORTED_FILTER_MODES


def render(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
    freq_trajectory: np.ndarray | None = None,
) -> np.ndarray:
    """Render a harmonic-rich source shaped by a ZDF state-variable filter.

    The ``quality`` param selects the ladder solver + oversampling factor
    applied to the filter + feedback + dither block:

    - ``draft`` — ADAA ladder, no oversampling
    - ``fast`` — Newton ladder (2 iters), 2x oversampling
    - ``great`` — Newton ladder (4 iters), 2x oversampling (default)
    - ``divine`` — Newton ladder (8 iters), 4x oversampling
    """
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
    filter_topology = str(params.get("filter_topology", "svf")).lower()
    bass_compensation = float(params.get("bass_compensation", 0.0))
    filter_morph = float(params.get("filter_morph", 0.0))
    hpf_cutoff_hz = float(params.get("hpf_cutoff_hz", 0.0))
    hpf_resonance_q = float(params.get("hpf_resonance_q", 0.707))
    analog = extract_analog_params(params)
    pitch_drift = analog["pitch_drift"]
    analog_jitter = analog["analog_jitter"]
    noise_floor_level = analog["noise_floor"]
    drift_rate_hz = analog["drift_rate_hz"]
    cutoff_drift_amount = analog["cutoff_drift"]
    quality_config = resolve_quality_mode(str(analog["quality"]))

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
    freq_profile, amp, cutoff_hz, vc_offsets = apply_voice_card(
        params,
        voice_card_spread=analog["voice_card_spread"],
        pitch_spread=analog["voice_card_pitch_spread"],
        filter_spread=analog["voice_card_filter_spread"],
        envelope_spread=analog["voice_card_envelope_spread"],
        osc_spread=analog["voice_card_osc_spread"],
        level_spread=analog["voice_card_level_spread"],
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

    attack = float(params.get("attack", 0.04)) * vc_offsets["attack_scale"]
    release = float(params.get("release", 0.1)) * vc_offsets["release_scale"]
    _ = attack, release  # available for ADSR when wired

    if waveform == "pulse":
        pulse_width = min(
            0.99, max(0.01, pulse_width + vc_offsets["pulse_width_offset"])
        )
    resonance_q, drift_rate_hz = apply_voice_card_post_offsets(
        resonance_q, drift_rate_hz, vc_offsets
    )

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

    # OB-Xd-style per-sample CV dither on pitch (layered on top of stable
    # voice_card offsets).  Shares the ``analog_jitter`` knob with the
    # per-note jitter so there's no new surface to reason about.
    freq_profile = apply_pitch_cv_dither(
        freq_profile,
        analog_jitter=analog_jitter,
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=sample_rate,
        n_samples=n_samples,
    )

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
    cutoff_profile = build_keytracked_cutoff_profile(
        cutoff_hz=cutoff_hz,
        keytrack=keytrack,
        reference_freq_hz=reference_freq_hz,
        filter_env_amount=filter_env_amount,
        filter_env_decay=filter_env_decay,
        duration=duration,
        n_samples=n_samples,
        freq_profile=freq_profile,
        nyquist=nyquist_hz,
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

    # OB-Xd-style per-sample CV dither on the filter cutoff.  Same knob as
    # the pitch dither (``analog_jitter``), different target.
    cutoff_profile = apply_cutoff_cv_dither(
        cutoff_profile,
        analog_jitter=analog_jitter,
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=sample_rate,
        n_samples=n_samples,
        nyquist=nyquist_hz,
    )

    filtered = apply_filter_oversampled(
        signal,
        cutoff_profile=cutoff_profile,
        resonance_q=resonance_q,
        sample_rate=sample_rate,
        oversample_factor=quality_config.oversample_factor,
        filter_mode=filter_mode,
        filter_drive=filter_drive,
        filter_even_harmonics=filter_even_harmonics,
        filter_topology=filter_topology,
        bass_compensation=bass_compensation,
        filter_morph=filter_morph,
        hpf_cutoff_hz=hpf_cutoff_hz,
        hpf_resonance_q=hpf_resonance_q,
        filter_solver=quality_config.solver,
        max_newton_iters=quality_config.max_newton_iters,
        newton_tolerance=quality_config.newton_tolerance,
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
