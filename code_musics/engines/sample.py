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

from code_musics.engines._drum_utils import rng_for_note
from code_musics.engines._envelopes import render_envelope
from code_musics.engines._filters import _SUPPORTED_FILTER_MODES, apply_filter

logger: logging.Logger = logging.getLogger(__name__)

_VALID_FILTER_MODES = _SUPPORTED_FILTER_MODES - {"notch"}
_VALID_RETRIGGER_CURVES = frozenset({"linear", "exp"})

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


def _render_single_trigger(
    buffer: np.ndarray,
    *,
    pitch_ratio: float,
    bend_cents_env: np.ndarray | None,
    reverse: bool,
    n_samples: int,
    sample_rate: int,
) -> np.ndarray:
    """Play *buffer* once into an n_samples-long float64 output.

    *pitch_ratio* is a static ratio (new_len = len(buffer) * root/freq * ratio-relative).
    When *bend_cents_env* is provided, playback rate becomes per-sample:
    ``rate(t) = pitch_ratio^-1 * 2^(bend_cents_env(t) / 1200)`` — i.e. the bend
    multiplies on top of the static pitch ratio. When both are neutral, the buffer
    is copied verbatim (truncated/padded).

    *reverse* flips the source buffer before playback.
    """
    src = buffer[::-1] if reverse else buffer

    if n_samples <= 0 or len(src) == 0:
        return np.zeros(max(0, n_samples), dtype=np.float64)

    if bend_cents_env is None and abs(pitch_ratio - 1.0) < 1e-6:
        out = np.zeros(n_samples, dtype=np.float64)
        copy_n = min(len(src), n_samples)
        out[:copy_n] = src[:copy_n]
        return out

    if bend_cents_env is None:
        # Static ratio: new_len = len(src) * pitch_ratio. Higher ratio -> longer
        # (sounds lower). Caller sets ratio = root_freq / freq for the standard
        # pitch-shift-to-freq behavior.
        new_len = max(1, int(len(src) * pitch_ratio))
        shifted = np.interp(
            np.linspace(0, len(src) - 1, new_len),
            np.arange(len(src)),
            src,
        )
        out = np.zeros(n_samples, dtype=np.float64)
        copy_n = min(len(shifted), n_samples)
        out[:copy_n] = shifted[:copy_n]
        return out

    # Variable-rate playback. Build per-output-sample source position by
    # integrating the per-sample rate. rate(t) = (1/pitch_ratio) * 2^(cents/1200)
    # so positive cents advance faster (higher pitch) and static pitch_ratio > 1
    # slows playback (lower pitch).
    bend_env = np.asarray(bend_cents_env, dtype=np.float64)
    if len(bend_env) != n_samples:
        # Resample envelope to n_samples via linear interp.
        bend_env = np.interp(
            np.linspace(0.0, 1.0, n_samples),
            np.linspace(0.0, 1.0, len(bend_env)),
            bend_env,
        )
    rate_per_sample = (1.0 / max(pitch_ratio, 1e-9)) * np.power(2.0, bend_env / 1200.0)
    positions = np.concatenate(([0.0], np.cumsum(rate_per_sample[:-1])))
    # np.interp with left=0.0, right=0.0 zeros any position that walks off
    # either edge of the source buffer — no extra clamp needed.
    return np.interp(
        positions,
        np.arange(len(src), dtype=np.float64),
        src,
        left=0.0,
        right=0.0,
    )


def render_sample_segment(
    buffer: np.ndarray,
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
    note_seed: int | None = None,
) -> np.ndarray:
    """Render an already-loaded sample buffer through the full sample-engine signal chain.

    This is the refactored core used by :func:`render` (which loads + caches a
    WAV first), and reusable by other engines (e.g. a ``drum_voice`` sample
    exciter) that already hold a buffer.

    Parameters mirror :func:`render`. *buffer* must be a 1-D mono float64 array
    at *sample_rate*. *note_seed*, when provided, seeds any deterministic
    randomness (currently just ``start_jitter_ms``). When ``None``, the seed is
    derived from (freq, duration, amp, sample_rate) via
    :func:`rng_for_note`.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    root_freq = float(params.get("root_freq", 440.0))
    do_pitch_shift = bool(params.get("pitch_shift", True))
    start_offset_ms = float(params.get("start_offset_ms", 0.0))
    start_jitter_ms = float(params.get("start_jitter_ms", 0.0))
    decay_ms_raw = params.get("decay_ms")
    reverse = bool(params.get("reverse", False))
    amp_envelope_raw = params.get("amp_envelope")

    filter_mode: str | None = params.get("filter_mode")
    filter_cutoff_hz = float(params.get("filter_cutoff_hz", 5000.0))
    filter_q = float(params.get("filter_q", 0.707))
    filter_topology = str(params.get("filter_topology", "svf")).lower()
    bass_compensation = float(params.get("bass_compensation", 0.5))

    retrigger_count = int(params.get("retrigger_count", 1))
    retrigger_interval_ms = float(params.get("retrigger_interval_ms", 0.0))
    retrigger_pitch_step_cents = float(params.get("retrigger_pitch_step_cents", 0.0))
    retrigger_decay_curve = str(params.get("retrigger_decay_curve", "exp")).lower()

    bend_envelope_raw = params.get("bend_envelope")

    ring_freq_hz = float(params.get("ring_freq_hz", 0.0))
    ring_depth = float(params.get("ring_depth", 0.0))

    rate_reduce_ratio = float(params.get("rate_reduce_ratio", 1.0))
    bit_depth = float(params.get("bit_depth", 16.0))

    if root_freq <= 0:
        raise ValueError("root_freq must be positive")
    if start_offset_ms < 0:
        raise ValueError("start_offset_ms must be non-negative")
    if start_jitter_ms < 0:
        raise ValueError("start_jitter_ms must be non-negative")
    if retrigger_count < 1:
        raise ValueError("retrigger_count must be >= 1")
    if retrigger_count > 1 and retrigger_interval_ms <= 0:
        raise ValueError("retrigger_interval_ms must be > 0 when retrigger_count > 1")
    if retrigger_decay_curve not in _VALID_RETRIGGER_CURVES:
        raise ValueError(
            f"retrigger_decay_curve must be one of {sorted(_VALID_RETRIGGER_CURVES)}, "
            f"got {retrigger_decay_curve!r}"
        )
    if ring_freq_hz < 0:
        raise ValueError("ring_freq_hz must be non-negative")
    if not 0.0 <= ring_depth <= 1.0:
        raise ValueError("ring_depth must be in [0, 1]")
    if rate_reduce_ratio < 1.0:
        raise ValueError("rate_reduce_ratio must be >= 1.0")
    if not 1.0 <= bit_depth <= 16.0:
        raise ValueError("bit_depth must be in [1.0, 16.0]")
    if buffer.ndim != 1:
        raise ValueError("buffer must be 1-D mono float64")

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0, dtype=np.float64)

    # Deterministic jitter RNG.
    if start_jitter_ms > 0.0:
        if note_seed is not None:
            rng = np.random.default_rng(int(note_seed))
        else:
            rng = rng_for_note(
                freq=freq,
                duration=duration,
                amp=amp,
                sample_rate=sample_rate,
                extra_seed="sample_engine",
            )
        jitter_samples = int(rng.uniform(0.0, start_jitter_ms) / 1000.0 * sample_rate)
    else:
        jitter_samples = 0

    base_offset = int(start_offset_ms / 1000.0 * sample_rate) + jitter_samples
    src = buffer[base_offset:] if base_offset > 0 else buffer
    if len(src) == 0:
        return np.zeros(n_samples, dtype=np.float64)

    # Base pitch ratio (static). When pitch_shift disabled, ratio = 1.0.
    if do_pitch_shift and freq > 0 and abs(freq - root_freq) > 0.01:
        base_pitch_ratio = root_freq / freq
    else:
        base_pitch_ratio = 1.0

    # Optional bend envelope in cents.
    if bend_envelope_raw is not None:
        bend_env = render_envelope(bend_envelope_raw, n_samples, default_value=0.0)
    else:
        bend_env = None

    # Render retrigger layers (single pass when retrigger_count == 1).
    interval_samples = int(retrigger_interval_ms / 1000.0 * sample_rate)
    if retrigger_count > 1 and interval_samples < 1:
        raise ValueError(
            f"retrigger_interval_ms={retrigger_interval_ms} is too small at "
            f"sample_rate={sample_rate}: resolves to interval_samples="
            f"{interval_samples}, need >= 1 to avoid stacking every retrigger "
            "at offset 0"
        )
    signal = np.zeros(n_samples, dtype=np.float64)
    for i in range(retrigger_count):
        trigger_offset = i * interval_samples
        if trigger_offset >= n_samples:
            break

        if retrigger_decay_curve == "exp":
            trigger_amp = 0.7**i
        else:
            trigger_amp = max(0.0, 1.0 - i / retrigger_count)

        # Each successive retrigger shifts by pitch_step_cents. Positive
        # cents => higher pitch => shorter ratio.
        trigger_cents = i * retrigger_pitch_step_cents
        trigger_ratio = base_pitch_ratio * (2.0 ** (-trigger_cents / 1200.0))

        trigger_n = n_samples - trigger_offset
        # When bend_env is set and this is a retrigger, slice the tail of bend_env
        # so it continues from the current time (retrigger bend follows the
        # score-time envelope, not per-trigger).
        trigger_bend = bend_env[trigger_offset:] if bend_env is not None else None

        trigger_out = _render_single_trigger(
            src,
            pitch_ratio=trigger_ratio,
            bend_cents_env=trigger_bend,
            reverse=reverse,
            n_samples=trigger_n,
            sample_rate=sample_rate,
        )
        signal[trigger_offset : trigger_offset + trigger_n] += trigger_amp * trigger_out

    # Apply exponential decay envelope (score-time).
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

    # Ring modulation (amplitude ring, post-playback).
    if ring_freq_hz > 0.0 and ring_depth > 0.0:
        t = np.arange(n_samples, dtype=np.float64) / sample_rate
        ring = (1.0 - ring_depth) + ring_depth * np.sin(2.0 * np.pi * ring_freq_hz * t)
        signal *= ring

    # Rate reduction (sample-and-hold). Deliberately naive grit, no anti-aliasing
    # here — this is a character knob, not a clean decimator.
    if rate_reduce_ratio > 1.0:
        step = int(round(rate_reduce_ratio))
        if step > 1:
            held = np.repeat(signal[::step], step)[:n_samples]
            signal = np.ascontiguousarray(held, dtype=np.float64)

    # Bit-depth quantization.
    if bit_depth < 16.0:
        levels = float(2**bit_depth)
        # Scale to full-range quantizer, round, scale back.
        step_size = 2.0 / levels
        signal = np.round(signal / step_size) * step_size
        signal = np.clip(signal, -1.0, 1.0 - step_size)

    # Peak-normalize then scale by amp.
    peak = np.max(np.abs(signal))
    if peak > 1e-9:
        signal = signal / peak
    return (amp * signal).astype(np.float64)


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

    Core playback params:
        root_freq (float): The natural pitch of the sample in Hz. Default 440.0.
        pitch_shift (bool): Whether to repitch to *freq*. Default True.
        start_offset_ms (float): Skip into the sample before playback. Default 0.0.
        start_jitter_ms (float): Deterministic per-note random extra start
            offset on top of *start_offset_ms*, drawn uniformly from
            ``[0, start_jitter_ms]``. Default 0.0 (disabled).
        decay_ms (float | None): Exponential decay time constant in ms. Default None.
        reverse (bool): Reverse the sample before playback. Default False.
        amp_envelope: Multi-point amplitude envelope (list of dicts). Default None.

    Filter params:
        filter_mode (str | None): 'lowpass', 'bandpass', or 'highpass'. Default None.
        filter_cutoff_hz (float): Filter cutoff in Hz. Default 5000.0.
        filter_q (float): Filter resonance Q. Default 0.707.

    Machinedrum-E12-style macros (all default to disabled):
        retrigger_count (int >= 1): Number of flam hits. Default 1.
        retrigger_interval_ms (float > 0): Spacing between hits.
        retrigger_pitch_step_cents (float): Cents shift per successive hit.
        retrigger_decay_curve ("linear" | "exp"): Per-hit amplitude shape. Default "exp".
        bend_envelope (list[dict]): Cents-valued pitch-bend envelope; applied
            as per-sample playback-rate multiplier.
        ring_freq_hz (float), ring_depth (0..1): Post-playback amplitude ring.
        rate_reduce_ratio (float >= 1.0): Integer-floor sample-and-hold grit.
        bit_depth (1.0..16.0): Quantization grit.
    """
    if freq_trajectory is not None:
        raise ValueError(
            "sample engine does not support freq_trajectory (pitch motion)"
        )

    sample_path = params.get("sample_path")
    if not sample_path:
        raise ValueError("sample engine requires a 'sample_path' parameter")

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    buffer = _load_sample(str(sample_path), sample_rate)
    return render_sample_segment(
        buffer,
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=sample_rate,
        params=params,
    )
