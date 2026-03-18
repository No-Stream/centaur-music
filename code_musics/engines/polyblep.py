"""PolyBLEP synthesis engine.

Generates waveforms in the time domain with polynomial bandlimiting corrections
at discontinuities. Produces smooth analog character with correct 1/n harmonic
spectrum and no Gibbs phenomenon, unlike additive-truncated engines.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


def render(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
    freq_trajectory: np.ndarray | None = None,
) -> np.ndarray:
    """Render a bandlimited oscillator with a time-domain LP filter sweep."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    waveform = str(params.get("waveform", "saw")).lower()
    pulse_width = float(params.get("pulse_width", 0.5))
    cutoff_hz = float(params.get("cutoff_hz", 3000.0))
    keytrack = float(params.get("keytrack", 0.0))
    reference_freq_hz = float(params.get("reference_freq_hz", 220.0))
    resonance = float(params.get("resonance", 0.0))
    filter_env_amount = float(params.get("filter_env_amount", 0.0))
    filter_env_decay = float(params.get("filter_env_decay", 0.18))
    n_filter_segments = int(params.get("n_filter_segments", 8))

    if cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be positive")
    if reference_freq_hz <= 0:
        raise ValueError("reference_freq_hz must be positive")
    if filter_env_decay <= 0:
        raise ValueError("filter_env_decay must be positive")
    if not 0.0 < pulse_width < 1.0:
        raise ValueError("pulse_width must be between 0 and 1")
    if n_filter_segments < 1:
        raise ValueError("n_filter_segments must be at least 1")
    if waveform not in {"saw", "square"}:
        raise ValueError(f"Unsupported waveform: {waveform!r}. Use 'saw' or 'square'.")

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

    # Phase accumulation in normalized [0, 1) domain
    phase_inc = freq_profile / sample_rate
    cumphase = np.cumsum(phase_inc)
    phase = cumphase % 1.0

    if waveform == "saw":
        raw_signal = _polyblep_saw(phase, phase_inc)
    else:
        raw_signal = _polyblep_square(phase, phase_inc, cumphase, pulse_width)

    # Cutoff envelope (identical pattern to filtered_stack)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)
    nyquist = sample_rate / 2.0
    cutoff_envelope = 1.0 + filter_env_amount * np.exp(-t / filter_env_decay)
    cutoff_envelope = np.maximum(cutoff_envelope, 0.05)
    keytracked_cutoff = cutoff_hz * np.power(freq_profile / reference_freq_hz, keytrack)
    cutoff_profile = np.clip(keytracked_cutoff * cutoff_envelope, 20.0, nyquist * 0.98)

    filtered = _apply_segmented_filter(
        raw_signal,
        cutoff_profile=cutoff_profile,
        resonance=resonance,
        sample_rate=sample_rate,
        n_segments=n_filter_segments,
        pre_filter_signal=raw_signal,
    )

    # Peak-normalize like fm.py (filter sweep causes uneven amplitude)
    peak = np.max(np.abs(filtered))
    if peak > 1e-9:
        filtered /= peak

    return amp * filtered


def _polyblep_saw(phase: np.ndarray, phase_inc: np.ndarray) -> np.ndarray:
    """Generate a bandlimited sawtooth via PolyBLEP correction."""
    saw = 2.0 * phase - 1.0  # new array; in-place assignment below is safe

    # Pre-discontinuity: phase approaching 1 (within one phase_inc)
    mask_pre = phase > (1.0 - phase_inc)
    t_pre = (phase[mask_pre] - 1.0) / phase_inc[mask_pre]  # in (-1, 0]
    saw[mask_pre] -= t_pre * t_pre + 2.0 * t_pre + 1.0

    # Post-discontinuity: phase just wrapped (within one phase_inc of 0)
    mask_post = phase < phase_inc
    t_post = phase[mask_post] / phase_inc[mask_post]  # in [0, 1)
    saw[mask_post] -= 2.0 * t_post - t_post * t_post - 1.0

    return saw


def _polyblep_square(
    phase: np.ndarray,
    phase_inc: np.ndarray,
    cumphase: np.ndarray,
    pulse_width: float,
) -> np.ndarray:
    """Generate a bandlimited square/pulse wave as the difference of two saws."""
    saw1 = _polyblep_saw(phase, phase_inc)
    phase2 = (cumphase + pulse_width) % 1.0
    saw2 = _polyblep_saw(phase2, phase_inc)
    square = (saw1 - saw2) / 2.0
    square -= square.mean()  # remove DC from pulse_width asymmetry (no-op at pw=0.5)
    return square


def _apply_segmented_filter(
    signal: np.ndarray,
    *,
    cutoff_profile: np.ndarray,
    resonance: float,
    sample_rate: int,
    n_segments: int,
    pre_filter_signal: np.ndarray,
) -> np.ndarray:
    """Apply a piecewise Butterworth low-pass, carrying filter state across segments."""
    nyquist = sample_rate / 2.0
    n_samples = len(signal)
    filtered = np.empty(n_samples, dtype=np.float64)
    segment_bounds = np.linspace(0, n_samples, n_segments + 1, dtype=int)

    zi: np.ndarray | None = None
    for seg_idx in range(n_segments):
        start = int(segment_bounds[seg_idx])
        end = int(segment_bounds[seg_idx + 1])
        if start >= end:
            continue
        mid = (start + end) // 2
        cutoff_norm = (
            float(cutoff_profile[mid]) / nyquist
        )  # already clipped to [0, 0.98]
        sos = butter(2, cutoff_norm, btype="low", output="sos")
        if zi is None:
            zi = np.asarray(sosfilt_zi(sos)) * signal[start]  # pyright: ignore[reportOperatorIssue]
        filtered[start:end], zi = sosfilt(sos, signal[start:end], zi=zi)

    if resonance > 0.0:
        fc = float(np.median(cutoff_profile))
        if fc > 80.0:  # guard against unstable narrow bandpass at very low fc
            low = np.clip(fc * 0.7 / nyquist, 1e-4, 0.49)
            high = np.clip(fc * 1.3 / nyquist, low + 1e-4, 0.9999)
            sos_bp = butter(2, [low, high], btype="band", output="sos")
            bp_out = np.asarray(sosfilt(sos_bp, pre_filter_signal))  # pyright: ignore[reportArgumentType]
            filtered = filtered + resonance * 2.0 * bp_out

    return filtered
