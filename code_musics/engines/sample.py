"""Sample playback engine — loads WAV files and plays them with pitch/decay/filter control."""

from __future__ import annotations

import functools
import logging
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from code_musics.engines._envelopes import render_envelope
from code_musics.engines._filters import _SUPPORTED_FILTER_MODES, apply_filter

logger: logging.Logger = logging.getLogger(__name__)

_VALID_FILTER_MODES = _SUPPORTED_FILTER_MODES - {"notch"}

# Project root for resolving relative sample paths.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@functools.lru_cache(maxsize=50)
def _load_sample(path: str, target_sr: int) -> np.ndarray:
    """Load a WAV file, convert to mono float64, resample to *target_sr*. Cached."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = _PROJECT_ROOT / resolved

    data, file_sr = sf.read(str(resolved), dtype="float64", always_2d=True)

    # Mix to mono if multi-channel.
    mono: np.ndarray = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]

    if file_sr != target_sr:
        g = gcd(target_sr, int(file_sr))
        up = target_sr // g
        down = int(file_sr) // g
        # Limit resampling ratio to avoid huge intermediate arrays.
        if up > 256 or down > 256:
            n_out = max(1, int(len(mono) * target_sr / file_sr))
            mono = np.interp(
                np.linspace(0, len(mono) - 1, n_out),
                np.arange(len(mono)),
                mono,
            )
        else:
            mono = resample_poly(mono, up, down).astype(np.float64)

    return mono


def render(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
    freq_trajectory: np.ndarray | None = None,
) -> np.ndarray:
    """Render a sample playback voice.

    Required params:
        sample_path: Path to a WAV file (absolute or relative to project root).

    Optional params:
        root_freq (float): The natural pitch of the sample in Hz. Default 440.0.
        pitch_shift (bool): Whether to repitch to *freq*. Default True.
        start_offset_ms (float): Skip into the sample before playback. Default 0.0.
        decay_ms (float | None): Exponential decay time constant in ms. Default None.
        reverse (bool): Reverse the sample before playback. Default False.
        amp_envelope: Multi-point amplitude envelope (list of dicts). Default None.
        filter_mode (str | None): 'lowpass', 'bandpass', or 'highpass'. Default None.
        filter_cutoff_hz (float): Filter cutoff in Hz. Default 5000.0.
        filter_q (float): Filter resonance Q. Default 0.707.
    """
    if freq_trajectory is not None:
        raise ValueError(
            "sample engine does not support freq_trajectory (pitch motion)"
        )
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    sample_path = params.get("sample_path")
    if not sample_path:
        raise ValueError("sample engine requires a 'sample_path' parameter")

    root_freq = float(params.get("root_freq", 440.0))
    do_pitch_shift = bool(params.get("pitch_shift", True))
    start_offset_ms = float(params.get("start_offset_ms", 0.0))
    decay_ms_raw = params.get("decay_ms")
    reverse = bool(params.get("reverse", False))
    amp_envelope_raw = params.get("amp_envelope")

    filter_mode: str | None = params.get("filter_mode")
    filter_cutoff_hz = float(params.get("filter_cutoff_hz", 5000.0))
    filter_q = float(params.get("filter_q", 0.707))
    filter_topology = str(params.get("filter_topology", "svf")).lower()
    bass_compensation = float(params.get("bass_compensation", 0.0))

    if root_freq <= 0:
        raise ValueError("root_freq must be positive")
    if start_offset_ms < 0:
        raise ValueError("start_offset_ms must be non-negative")

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0, dtype=np.float64)

    # Load and cache the sample at our target sample rate.
    raw = _load_sample(str(sample_path), sample_rate)

    # Apply start offset.
    offset_samples = int(start_offset_ms / 1000.0 * sample_rate)
    if offset_samples > 0:
        raw = raw[offset_samples:]
    if len(raw) == 0:
        return np.zeros(n_samples, dtype=np.float64)

    # Pitch shift via resampling: higher freq -> shorter period -> fewer samples.
    signal = raw
    if do_pitch_shift and freq > 0 and abs(freq - root_freq) > 0.01:
        ratio = root_freq / freq
        new_len = max(1, int(len(signal) * ratio))
        signal = np.interp(
            np.linspace(0, len(signal) - 1, new_len),
            np.arange(len(signal)),
            signal,
        )

    if reverse:
        signal = signal[::-1].copy()

    # Truncate or zero-pad to target duration.
    if len(signal) >= n_samples:
        signal = signal[:n_samples].copy()
    else:
        padded = np.zeros(n_samples, dtype=np.float64)
        padded[: len(signal)] = signal
        signal = padded

    # Apply exponential decay envelope.
    if decay_ms_raw is not None:
        decay_ms = float(decay_ms_raw)
        if decay_ms > 0:
            t = np.arange(n_samples, dtype=np.float64) / sample_rate
            signal *= np.exp(-t / (decay_ms / 1000.0))

    # Apply custom multi-point amplitude envelope.
    if amp_envelope_raw is not None:
        env = render_envelope(amp_envelope_raw, n_samples, default_value=1.0)
        signal *= env

    # Apply filter.
    if filter_mode is not None:
        if filter_mode not in _VALID_FILTER_MODES:
            raise ValueError(
                f"filter_mode must be one of {sorted(_VALID_FILTER_MODES)}, got {filter_mode!r}"
            )
        cutoff_profile = np.full(
            n_samples, min(filter_cutoff_hz, sample_rate * 0.4), dtype=np.float64
        )
        signal = apply_filter(
            signal,
            cutoff_profile=cutoff_profile,
            resonance_q=filter_q,
            sample_rate=sample_rate,
            filter_mode=filter_mode,
            filter_drive=0.0,
            filter_topology=filter_topology,
            bass_compensation=bass_compensation,
        )

    # Peak-normalize then scale by amp.
    peak = np.max(np.abs(signal))
    if peak > 1e-9:
        signal = signal / peak
    return (amp * signal).astype(np.float64)
