"""PolyBLEP synthesis engine.

Generates waveforms in the time domain with polynomial bandlimiting corrections
at discontinuities. Produces smooth analog character with correct 1/n harmonic
spectrum and no Gibbs phenomenon, unlike additive-truncated engines.

Supported waveforms:
- ``saw``    — bandlimited sawtooth via direct PolyBLEP correction
- ``square`` — bandlimited square/pulse as the difference of two saws
- ``triangle`` — bandlimited triangle obtained by integrating the square wave
  (BLAMP approach); ``pulse_width`` is ignored for triangle
"""

from __future__ import annotations

from typing import Any

import numpy as np

from code_musics.engines._filters import _SUPPORTED_FILTER_MODES, apply_zdf_svf


def _polyblep_triangle(
    phase: np.ndarray,
    phase_inc: np.ndarray,
    cumphase: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Generate a bandlimited triangle wave by integrating the polyblep square.

    The integral of a bandlimited square wave is a bandlimited triangle (BLAMP),
    so this gives alias-free triangle essentially for free.  ``pulse_width`` is
    not meaningful for a symmetric triangle and is always fixed at 0.5.
    """
    square = _polyblep_square(phase, phase_inc, cumphase, pulse_width=0.5)

    # Integrate via cumulative sum, scaled so that the integral of a unit square
    # at frequency f has peak amplitude ~1/(2f) in samples → we want unit amplitude.
    # Dividing by sample_rate converts from sample-domain to seconds-domain;
    # the resulting triangle has amplitude proportional to 1/freq, so we
    # normalize afterward.
    triangle = np.cumsum(square) / sample_rate

    # DC-block: remove any accumulated offset from the running sum
    triangle -= triangle.mean()

    # Peak-normalize to match saw/square amplitude range (~1.0)
    peak = np.max(np.abs(triangle))
    if peak > 1e-9:
        triangle /= peak

    return triangle


def render(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
    freq_trajectory: np.ndarray | None = None,
) -> np.ndarray:
    """Render a bandlimited oscillator with a driven ZDF/TPT filter sweep."""
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
    filter_mode = str(params.get("filter_mode", "lowpass")).lower()
    filter_drive = float(params.get("filter_drive", 0.0))

    if cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be positive")
    if reference_freq_hz <= 0:
        raise ValueError("reference_freq_hz must be positive")
    if filter_env_decay <= 0:
        raise ValueError("filter_env_decay must be positive")
    if not 0.0 < pulse_width < 1.0:
        raise ValueError("pulse_width must be between 0 and 1")
    if waveform not in {"saw", "square", "triangle"}:
        raise ValueError(
            f"Unsupported waveform: {waveform!r}. Use 'saw', 'square', or 'triangle'."
        )
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

    # Phase accumulation in normalized [0, 1) domain
    phase_inc = freq_profile / sample_rate
    cumphase = np.cumsum(phase_inc)
    phase = cumphase % 1.0

    if waveform == "saw":
        raw_signal = _polyblep_saw(phase, phase_inc)
    elif waveform == "square":
        raw_signal = _polyblep_square(phase, phase_inc, cumphase, pulse_width)
    else:
        raw_signal = _polyblep_triangle(phase, phase_inc, cumphase, sample_rate)

    # Cutoff envelope (identical pattern to filtered_stack)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)
    nyquist = sample_rate / 2.0
    cutoff_envelope = 1.0 + filter_env_amount * np.exp(-t / filter_env_decay)
    cutoff_envelope = np.maximum(cutoff_envelope, 0.05)
    keytracked_cutoff = cutoff_hz * np.power(freq_profile / reference_freq_hz, keytrack)
    cutoff_profile = np.clip(keytracked_cutoff * cutoff_envelope, 20.0, nyquist * 0.98)

    filtered = apply_zdf_svf(
        raw_signal,
        cutoff_profile=cutoff_profile,
        resonance=resonance,
        sample_rate=sample_rate,
        filter_mode=filter_mode,
        filter_drive=filter_drive,
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


def render_polyblep(
    freq: float,
    duration: float,
    sample_rate: int,
    params: dict[str, Any],
    freq_trajectory: np.ndarray | None = None,
) -> np.ndarray:
    """Convenience wrapper around :func:`render` with positional arguments.

    Renders at unit amplitude (``amp=1.0``); use the returned array directly or
    scale afterward.
    """
    return render(
        freq=freq,
        duration=duration,
        amp=1.0,
        sample_rate=sample_rate,
        params=params,
        freq_trajectory=freq_trajectory,
    )
