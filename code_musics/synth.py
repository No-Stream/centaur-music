"""Core synthesis utilities."""

from __future__ import annotations

import ctypes
import logging
import math
import os
import re
import struct
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast, overload

import numba
import numpy as np
import pedalboard
import soundfile as sf
from scipy.signal import butter, lfilter, resample_poly, sosfilt, tf2sos

from code_musics.automation import apply_control_automation
from code_musics.engines._dsp_utils import (
    apply_filter_oversampled_preupsampled,
    classify_thd,
    compute_signal_thd,
    resolve_quality_mode,
    upsample_cutoff_profile,
)
from code_musics.engines._filters import (
    _SUPPORTED_FILTER_MODES,
    _SUPPORTED_FILTER_TOPOLOGIES,
)
from code_musics.engines._instrumentation import (
    PerHitSummary,
    a_weighted_mean_band_energy_db,
    detect_percussive_onsets,
    intermodulation_ratio,
    per_hit_transient_metrics,
)
from code_musics.engines._waveshaper import (
    _apply_adaa2_poly_knee,
    _apply_adaa2_poly_knee_scalar,
    _biased_shape,
    _koren_triode_shape,
    _pentode_shape,
)
from code_musics.modulation import combine_connections_on_curve

# Effect-level filter mode aliases.  The filter kernels use "lowpass"/"bandpass"/
# "highpass"/"notch" internally; on the effect surface we accept the short
# aliases (lp/bp/hp/notch) that match the documented parameter ranges.
_ANALOG_FILTER_MODE_ALIAS: dict[str, str] = {
    "lp": "lowpass",
    "bp": "bandpass",
    "hp": "highpass",
    "notch": "notch",
    "lowpass": "lowpass",
    "bandpass": "bandpass",
    "highpass": "highpass",
}

logger: logging.Logger = logging.getLogger(__name__)

_PEDALBOARD_CLS: Any = getattr(pedalboard, "Pedalboard")  # noqa: B009
_DELAY_CLS: Any = getattr(pedalboard, "Delay")  # noqa: B009
_REVERB_CLS: Any = getattr(pedalboard, "Reverb")  # noqa: B009
_CONVOLUTION_CLS: Any = getattr(pedalboard, "Convolution")  # noqa: B009

SAMPLE_RATE = 44100


@dataclass(frozen=True)
class ExternalPluginSpec:
    """Declarative external plugin definition."""

    name: str
    path: Path
    format: str = "vst3"
    host: str = "pedalboard"
    bundle_plugin_name: str | None = None
    preload_libraries: tuple[Path, ...] = ()


PluginConfigurer = Callable[[Any, dict[str, Any]], None]


@dataclass(frozen=True)
class MasteringResult:
    """Finalized export audio plus its summary measurements."""

    signal: np.ndarray
    integrated_lufs: float
    true_peak_dbfs: float


@dataclass(frozen=True)
class SignalLevelDiagnostics:
    """Peak and loudness measurements for a rendered signal."""

    peak_dbfs: float
    true_peak_dbfs: float
    integrated_lufs: float
    active_window_fraction: float


@dataclass(frozen=True)
class EffectAnalysisWarning:
    """Structured warning for an effect stage that looks inactive or aggressive."""

    severity: str
    code: str
    message: str
    metrics: dict[str, float | int | str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class EffectAnalysisEntry:
    """Machine-readable diagnostics for one effect in a chain."""

    index: int
    kind: str
    display_name: str
    metrics: dict[str, float | int | str]
    warnings: list[EffectAnalysisWarning]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "index": self.index,
            "kind": self.kind,
            "display_name": self.display_name,
            "metrics": dict(self.metrics),
            "warnings": [warning.to_dict() for warning in self.warnings],
        }


@dataclass(frozen=True)
class ChainSummary:
    """Cumulative diagnostics across a full effect chain.

    Surfaces "death by a thousand cuts" failure modes where no single
    effect is over-driven but their stacked contribution is. Metrics
    aggregate per-effect IO-deltas across the chain and flag drum-bus-
    style failure patterns (papery brightness creep, transient
    flattening, harmonic content piling up across serial saturation
    stages).
    """

    metrics: dict[str, float | int | str]
    warnings: list[EffectAnalysisWarning]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "warnings": [w.to_dict() for w in self.warnings],
        }


_LOW_EXPORT_PEAK_WARNING_DBFS = -3.0
_NEAR_FULL_SCALE_THRESHOLD = 0.98
_EFFECT_ANALYSIS_EPSILON = 1e-12


def db_to_amp(db: float) -> float:
    """Convert decibels to a linear amplitude multiplier."""
    return float(10.0 ** (db / 20.0))


def amp_to_db(amp: float) -> float:
    """Convert a linear amplitude multiplier to decibels."""
    if amp <= 0:
        raise ValueError("amp must be positive")
    return float(20.0 * np.log10(amp))


def to_mono_reference(signal: np.ndarray) -> np.ndarray:
    """Return a mono analysis reference for mono or stereo audio."""
    normalized = np.asarray(signal, dtype=np.float64)
    if normalized.ndim == 1:
        return normalized
    if normalized.ndim == 2:
        return np.asarray(
            np.mean(normalized, axis=0, dtype=np.float64), dtype=np.float64
        )
    raise ValueError("signal must be mono or stereo")


def to_analysis_channels(signal: np.ndarray) -> np.ndarray:
    """Return analysis channels with shape (channels, samples)."""
    normalized = np.asarray(signal, dtype=np.float64)
    if normalized.ndim == 1:
        return normalized[np.newaxis, :]
    if normalized.ndim == 2 and normalized.shape[0] == 2:
        return normalized
    if normalized.ndim == 2 and normalized.shape[1] == 2:
        return normalized.T
    raise ValueError("signal must be mono or stereo")


def estimate_true_peak_amplitude(
    signal: np.ndarray,
    *,
    oversample_factor: int = 4,
) -> float:
    """Estimate inter-sample peak amplitude via simple oversampling."""
    channels = to_analysis_channels(signal)
    if channels.shape[-1] == 0:
        return 0.0
    if oversample_factor <= 1:
        return float(np.max(np.abs(channels)))

    channel_peaks = [
        float(
            np.max(
                np.abs(
                    np.asarray(
                        resample_poly(channel, oversample_factor, 1),
                        dtype=np.float64,
                    )
                )
            )
        )
        for channel in channels
    ]
    return float(max(channel_peaks, default=0.0))


def measure_signal_levels(
    signal: np.ndarray,
    *,
    sample_rate: int,
    oversample_factor: int = 4,
) -> SignalLevelDiagnostics:
    """Measure peak and loudness diagnostics for mono or stereo audio."""
    normalized_signal = np.asarray(signal, dtype=np.float64)
    channels = to_analysis_channels(normalized_signal)
    if channels.shape[-1] == 0:
        return SignalLevelDiagnostics(
            peak_dbfs=float("-inf"),
            true_peak_dbfs=float("-inf"),
            integrated_lufs=float("-inf"),
            active_window_fraction=0.0,
        )

    peak_amplitude = float(np.max(np.abs(channels)))
    true_peak_amplitude = estimate_true_peak_amplitude(
        normalized_signal,
        oversample_factor=oversample_factor,
    )
    integrated_loudness_lufs, active_window_fraction = integrated_lufs(
        normalized_signal,
        sample_rate=sample_rate,
    )
    return SignalLevelDiagnostics(
        peak_dbfs=amp_to_db(max(peak_amplitude, 1e-12)),
        true_peak_dbfs=amp_to_db(max(true_peak_amplitude, 1e-12)),
        integrated_lufs=integrated_loudness_lufs,
        active_window_fraction=active_window_fraction,
    )


def _safe_amp_to_db(amp: float) -> float:
    return amp_to_db(max(float(amp), _EFFECT_ANALYSIS_EPSILON))


def _signal_peak_dbfs(signal: np.ndarray) -> float:
    if signal.size == 0:
        return float("-inf")
    return _safe_amp_to_db(float(np.max(np.abs(signal))))


def _signal_rms_dbfs(signal: np.ndarray) -> float:
    mono_signal = to_mono_reference(signal)
    if mono_signal.size == 0:
        return float("-inf")
    return _safe_amp_to_db(float(np.sqrt(np.mean(np.square(mono_signal)))))


def _signal_crest_factor_db(signal: np.ndarray) -> float:
    peak_dbfs = _signal_peak_dbfs(signal)
    rms_dbfs = _signal_rms_dbfs(signal)
    if not np.isfinite(peak_dbfs) or not np.isfinite(rms_dbfs):
        return 0.0
    return max(0.0, peak_dbfs - rms_dbfs)


def _clipped_sample_fraction(signal: np.ndarray) -> float:
    mono_signal = to_mono_reference(signal)
    if mono_signal.size == 0:
        return 0.0
    return float(np.count_nonzero(np.abs(mono_signal) >= 1.0) / mono_signal.size)


def _near_full_scale_fraction(
    signal: np.ndarray,
    *,
    threshold: float = _NEAR_FULL_SCALE_THRESHOLD,
) -> float:
    mono_signal = to_mono_reference(signal)
    if mono_signal.size == 0:
        return 0.0
    return float(np.count_nonzero(np.abs(mono_signal) >= threshold) / mono_signal.size)


def _average_spectrum_db(
    signal: np.ndarray,
    *,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    mono_signal = to_mono_reference(signal)
    if mono_signal.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    n_fft = min(
        8192,
        max(2048, int(2 ** np.ceil(np.log2(max(mono_signal.size, 2_048))))),
    )
    window = np.hanning(n_fft)
    if mono_signal.size < n_fft:
        padded = np.zeros(n_fft, dtype=np.float64)
        padded[: mono_signal.size] = mono_signal
        spectrum = np.abs(np.fft.rfft(padded * window))
    else:
        step = max(n_fft // 2, 1)
        magnitudes: list[np.ndarray] = []
        for start in range(0, mono_signal.size - n_fft + 1, step):
            frame = mono_signal[start : start + n_fft] * window
            magnitudes.append(np.abs(np.fft.rfft(frame)))
        spectrum = np.mean(magnitudes, axis=0)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    magnitude_db = 20.0 * np.log10(
        np.maximum(np.asarray(spectrum, dtype=np.float64), _EFFECT_ANALYSIS_EPSILON)
    )
    valid = freqs > 0
    return freqs[valid], magnitude_db[valid]


def _mean_band_energy_db(
    magnitude_db: np.ndarray,
    *,
    freqs: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return float("-inf")
    return float(np.mean(magnitude_db[mask]))


def _spectral_centroid_hz(signal: np.ndarray, *, sample_rate: int) -> float:
    freqs, magnitude_db = _average_spectrum_db(signal, sample_rate=sample_rate)
    if freqs.size == 0:
        return 0.0
    magnitudes = np.power(10.0, magnitude_db / 20.0)
    magnitude_sum = np.sum(magnitudes)
    if magnitude_sum <= 0:
        return 0.0
    return float(np.sum(freqs * magnitudes) / magnitude_sum)


def _seconds_for_mask(mask: np.ndarray, *, sample_rate: int) -> float:
    if mask.size == 0:
        return 0.0
    bool_mask = np.asarray(mask, dtype=bool)
    if not np.any(bool_mask):
        return 0.0
    # Pad with False on both sides so diff always captures run boundaries.
    padded = np.concatenate(([False], bool_mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    # Rising edges (+1) mark run starts; falling edges (-1) mark run ends.
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    longest_run = int(np.max(ends - starts))
    return float(longest_run / sample_rate)


def gated_rms_dbfs(
    signal: np.ndarray,
    *,
    sample_rate: int,
    window_seconds: float = 0.4,
    hop_seconds: float = 0.1,
    gate_dbfs: float = -50.0,
) -> tuple[float, float]:
    """Return a silence-aware RMS proxy and the active-window fraction."""
    mono_reference = to_mono_reference(signal)
    if mono_reference.size == 0:
        return float("-inf"), 0.0

    window_samples = max(1, int(round(window_seconds * sample_rate)))
    hop_samples = max(1, int(round(hop_seconds * sample_rate)))
    frame_rms_values: list[float] = []
    for start in range(0, mono_reference.size, hop_samples):
        frame = mono_reference[start : start + window_samples]
        if frame.size == 0:
            continue
        frame_rms_values.append(float(np.sqrt(np.mean(np.square(frame)))))

    if not frame_rms_values:
        return float("-inf"), 0.0

    frame_rms = np.asarray(frame_rms_values, dtype=np.float64)
    active_mask = _amplitude_to_db_array(frame_rms) > gate_dbfs
    if not np.any(active_mask):
        active_mask = frame_rms > 1e-12
    if not np.any(active_mask):
        return float("-inf"), 0.0

    gated_rms = float(np.sqrt(np.mean(np.square(frame_rms[active_mask]))))
    return amp_to_db(max(gated_rms, 1e-12)), float(
        np.mean(active_mask.astype(np.float64))
    )


def integrated_lufs(
    signal: np.ndarray,
    *,
    sample_rate: int,
    block_seconds: float = 0.4,
    hop_seconds: float = 0.1,
    absolute_gate_lufs: float = -70.0,
) -> tuple[float, float]:
    """Return BS.1770-style integrated loudness and gated-block fraction."""
    channels = to_analysis_channels(signal)
    if channels.shape[-1] == 0:
        return float("-inf"), 0.0

    weighted_channels = _apply_k_weighting(channels, sample_rate=sample_rate)
    block_samples = max(1, int(round(block_seconds * sample_rate)))
    hop_samples = max(1, int(round(hop_seconds * sample_rate)))
    block_energies = _block_channel_energies(
        weighted_channels,
        block_samples=block_samples,
        hop_samples=hop_samples,
    )
    if block_energies.size == 0:
        return float("-inf"), 0.0

    channel_weights = np.ones(block_energies.shape[1], dtype=np.float64)
    weighted_block_energy = np.sum(block_energies * channel_weights, axis=1)
    block_loudness = -0.691 + 10.0 * np.log10(np.maximum(weighted_block_energy, 1e-12))

    absolute_mask = block_loudness >= absolute_gate_lufs
    if not np.any(absolute_mask):
        return float("-inf"), 0.0

    preliminary_energy = float(np.mean(weighted_block_energy[absolute_mask]))
    preliminary_lufs = -0.691 + 10.0 * np.log10(max(preliminary_energy, 1e-12))
    relative_gate_lufs = preliminary_lufs - 10.0
    final_mask = absolute_mask & (block_loudness >= relative_gate_lufs)
    if not np.any(final_mask):
        final_mask = absolute_mask

    integrated_energy = float(np.mean(weighted_block_energy[final_mask]))
    return (
        -0.691 + 10.0 * np.log10(max(integrated_energy, 1e-12)),
        float(np.mean(final_mask.astype(np.float64))),
    )


def normalize_to_gated_rms(
    signal: np.ndarray,
    *,
    sample_rate: int,
    target_dbfs: float,
    gate_dbfs: float = -50.0,
    max_gain_db: float = 18.0,
) -> np.ndarray:
    """Apply a uniform gain so the signal reaches the target gated RMS level."""
    current_dbfs, active_window_fraction = gated_rms_dbfs(
        signal,
        sample_rate=sample_rate,
        gate_dbfs=gate_dbfs,
    )
    if not np.isfinite(current_dbfs) or active_window_fraction <= 0.0:
        return np.asarray(signal, dtype=np.float64)

    required_gain_db = float(
        np.clip(target_dbfs - current_dbfs, -max_gain_db, max_gain_db)
    )
    return np.asarray(signal, dtype=np.float64) * db_to_amp(required_gain_db)


def normalize_to_lufs(
    signal: np.ndarray,
    *,
    sample_rate: int,
    target_lufs: float,
    max_gain_db: float = 18.0,
) -> np.ndarray:
    """Apply a uniform gain so the signal reaches the target integrated LUFS."""
    current_lufs, active_window_fraction = integrated_lufs(
        signal,
        sample_rate=sample_rate,
    )
    if not np.isfinite(current_lufs) or active_window_fraction <= 0.0:
        return np.asarray(signal, dtype=np.float64)

    required_gain_db = float(
        np.clip(target_lufs - current_lufs, -max_gain_db, max_gain_db)
    )
    return np.asarray(signal, dtype=np.float64) * db_to_amp(required_gain_db)


def gain_stage_for_master_bus(
    signal: np.ndarray,
    *,
    sample_rate: int,
    target_lufs: float = -24.0,
    max_true_peak_dbfs: float = -6.0,
    max_boost_db: float = 18.0,
    oversample_factor: int = 4,
) -> np.ndarray:
    """Apply one gain step so the premaster mix hits the master bus sensibly.

    This is intentionally not export mastering. It preserves the authored
    balance, raises or lowers the summed post-fader mix toward a reasonable
    working LUFS, and caps any upward move so the premaster true peak stays
    under a safety ceiling before master effects.
    """
    staged = np.asarray(signal, dtype=np.float64)
    if staged.size == 0:
        return staged

    current_lufs, active_window_fraction = integrated_lufs(
        staged,
        sample_rate=sample_rate,
    )
    if not np.isfinite(current_lufs) or active_window_fraction <= 0.0:
        return staged

    target_gain_db = float(target_lufs - current_lufs)
    target_gain_db = min(target_gain_db, max_boost_db)

    true_peak_amplitude = estimate_true_peak_amplitude(
        staged,
        oversample_factor=oversample_factor,
    )
    peak_limit_gain_db = float(
        max_true_peak_dbfs - amp_to_db(max(true_peak_amplitude, 1e-12))
    )
    applied_gain_db = min(target_gain_db, peak_limit_gain_db)
    chosen = "LUFS" if target_gain_db <= peak_limit_gain_db else "peak-limit"
    logger.info(
        f"Master gain stage: input LUFS {current_lufs:.1f}, "
        f"LUFS-gain {target_gain_db:.2f} dB, peak-limit-gain {peak_limit_gain_db:.2f} dB, "
        f"applied {applied_gain_db:.2f} dB ({chosen})"
    )
    return staged * db_to_amp(applied_gain_db)


def _amplitude_to_db_array(amplitudes: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.asarray(amplitudes, dtype=np.float64), 1e-12))


def _apply_k_weighting(channels: np.ndarray, *, sample_rate: int) -> np.ndarray:
    high_shelf_sos = tf2sos(
        *_design_high_shelf_biquad(
            sample_rate=sample_rate,
            center_hz=1_681.974450955533,
            q=0.7071752369554196,
            gain_db=4.0,
        )
    )
    high_pass_sos = tf2sos(
        *_design_highpass_biquad(
            sample_rate=sample_rate,
            cutoff_hz=38.13547087602444,
            q=0.5003270373238773,
        )
    )
    weighted = np.asarray(channels, dtype=np.float64)
    weighted = sosfilt(high_shelf_sos, weighted, axis=-1)
    weighted = sosfilt(high_pass_sos, weighted, axis=-1)
    return np.asarray(weighted, dtype=np.float64)


def _block_channel_energies(
    channels: np.ndarray,
    *,
    block_samples: int,
    hop_samples: int,
) -> np.ndarray:
    if channels.shape[-1] == 0:
        return np.zeros((0, channels.shape[0]), dtype=np.float64)

    block_energy_values: list[np.ndarray] = []
    if channels.shape[-1] <= block_samples:
        block_energy_values.append(np.mean(np.square(channels), axis=1))
    else:
        for start in range(0, channels.shape[-1] - block_samples + 1, hop_samples):
            block = channels[:, start : start + block_samples]
            block_energy_values.append(np.mean(np.square(block), axis=1))

    if not block_energy_values:
        return np.zeros((0, channels.shape[0]), dtype=np.float64)
    return np.asarray(block_energy_values, dtype=np.float64)


def _design_highpass_biquad(
    *,
    sample_rate: int,
    cutoff_hz: float,
    q: float,
) -> tuple[np.ndarray, np.ndarray]:
    omega = 2.0 * np.pi * cutoff_hz / sample_rate
    alpha = np.sin(omega) / (2.0 * q)
    cosine = np.cos(omega)

    b = np.array(
        [
            (1.0 + cosine) / 2.0,
            -(1.0 + cosine),
            (1.0 + cosine) / 2.0,
        ],
        dtype=np.float64,
    )
    a = np.array(
        [
            1.0 + alpha,
            -2.0 * cosine,
            1.0 - alpha,
        ],
        dtype=np.float64,
    )
    return b / a[0], a / a[0]


def _design_high_shelf_biquad(
    *,
    sample_rate: int,
    center_hz: float,
    q: float,
    gain_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    amplitude = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * np.pi * center_hz / sample_rate
    alpha = np.sin(omega) / (2.0 * q)
    cosine = np.cos(omega)
    sqrt_amplitude = np.sqrt(amplitude)

    b = np.array(
        [
            amplitude
            * (
                (amplitude + 1.0)
                + (amplitude - 1.0) * cosine
                + 2.0 * sqrt_amplitude * alpha
            ),
            -2.0 * amplitude * ((amplitude - 1.0) + (amplitude + 1.0) * cosine),
            amplitude
            * (
                (amplitude + 1.0)
                + (amplitude - 1.0) * cosine
                - 2.0 * sqrt_amplitude * alpha
            ),
        ],
        dtype=np.float64,
    )
    a = np.array(
        [
            (amplitude + 1.0)
            - (amplitude - 1.0) * cosine
            + 2.0 * sqrt_amplitude * alpha,
            2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * cosine),
            (amplitude + 1.0)
            - (amplitude - 1.0) * cosine
            - 2.0 * sqrt_amplitude * alpha,
        ],
        dtype=np.float64,
    )
    return b / a[0], a / a[0]


def _design_low_shelf_biquad(
    *,
    sample_rate: int,
    center_hz: float,
    q: float,
    gain_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    amplitude = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * np.pi * center_hz / sample_rate
    alpha = np.sin(omega) / (2.0 * q)
    cosine = np.cos(omega)
    sqrt_amplitude = np.sqrt(amplitude)

    b = np.array(
        [
            amplitude
            * (
                (amplitude + 1.0)
                - (amplitude - 1.0) * cosine
                + 2.0 * sqrt_amplitude * alpha
            ),
            2.0 * amplitude * ((amplitude - 1.0) - (amplitude + 1.0) * cosine),
            amplitude
            * (
                (amplitude + 1.0)
                - (amplitude - 1.0) * cosine
                - 2.0 * sqrt_amplitude * alpha
            ),
        ],
        dtype=np.float64,
    )
    a = np.array(
        [
            (amplitude + 1.0)
            + (amplitude - 1.0) * cosine
            + 2.0 * sqrt_amplitude * alpha,
            -2.0 * ((amplitude - 1.0) + (amplitude + 1.0) * cosine),
            (amplitude + 1.0)
            + (amplitude - 1.0) * cosine
            - 2.0 * sqrt_amplitude * alpha,
        ],
        dtype=np.float64,
    )
    return b / a[0], a / a[0]


def _design_peaking_biquad(
    *,
    sample_rate: int,
    center_hz: float,
    q: float,
    gain_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    amplitude = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * np.pi * center_hz / sample_rate
    alpha = np.sin(omega) / (2.0 * q)
    cosine = np.cos(omega)

    b = np.array(
        [
            1.0 + (alpha * amplitude),
            -2.0 * cosine,
            1.0 - (alpha * amplitude),
        ],
        dtype=np.float64,
    )
    a = np.array(
        [
            1.0 + (alpha / amplitude),
            -2.0 * cosine,
            1.0 - (alpha / amplitude),
        ],
        dtype=np.float64,
    )
    return b / a[0], a / a[0]


def _validate_eq_frequency(
    *, frequency_hz: float, sample_rate: int, label: str
) -> None:
    nyquist = sample_rate / 2.0
    if not 0.0 < frequency_hz < nyquist:
        raise ValueError(f"{label} must be between 0 and Nyquist")


def _validate_eq_q(q: float) -> None:
    if q <= 0.0:
        raise ValueError("q must be positive")


@dataclass(frozen=True)
class _PassBandSpec:
    """Dispatch entry for Butterworth highpass / lowpass EQ bands."""

    btype: str
    allowed_keys: frozenset[str] = frozenset({"kind", "cutoff_hz", "slope_db_per_oct"})


@dataclass(frozen=True)
class _ParametricBandSpec:
    """Dispatch entry for bell / shelf EQ bands."""

    design_fn: Callable[..., tuple[np.ndarray, np.ndarray]]
    default_q: float | None
    allowed_keys: frozenset[str] = frozenset({"kind", "freq_hz", "gain_db", "q"})


_EQ_BAND_DISPATCH: dict[str, _PassBandSpec | _ParametricBandSpec] = {
    "highpass": _PassBandSpec(btype="highpass"),
    "lowpass": _PassBandSpec(btype="lowpass"),
    "bell": _ParametricBandSpec(design_fn=_design_peaking_biquad, default_q=None),
    "low_shelf": _ParametricBandSpec(
        design_fn=_design_low_shelf_biquad, default_q=0.707
    ),
    "high_shelf": _ParametricBandSpec(
        design_fn=_design_high_shelf_biquad, default_q=0.707
    ),
}


def _design_eq_band_sos(
    *,
    band: dict[str, Any],
    sample_rate: int,
) -> np.ndarray:
    band_kind = str(band.get("kind", "")).lower()
    spec = _EQ_BAND_DISPATCH.get(band_kind)
    if spec is None:
        raise ValueError(f"Unsupported EQ band kind: {band_kind!r}")

    unknown_keys = set(band) - spec.allowed_keys
    if unknown_keys:
        raise ValueError(
            f"Unsupported parameters for {band_kind} EQ band: {sorted(unknown_keys)}"
        )

    if isinstance(spec, _PassBandSpec):
        cutoff_hz = float(band["cutoff_hz"])
        slope_db_per_oct = int(band["slope_db_per_oct"])
        _validate_eq_frequency(
            frequency_hz=cutoff_hz,
            sample_rate=sample_rate,
            label="cutoff_hz",
        )
        if slope_db_per_oct not in {12, 24}:
            raise ValueError("slope_db_per_oct must be 12 or 24")
        order = slope_db_per_oct // 6
        return np.asarray(
            butter(
                order,
                cutoff_hz / (sample_rate / 2.0),
                btype=spec.btype,
                output="sos",
            ),
            dtype=np.float64,
        )

    freq_hz = float(band["freq_hz"])
    gain_db = float(band["gain_db"])
    if spec.default_q is None:
        q = float(band["q"])
    else:
        q = float(band.get("q", spec.default_q))
    _validate_eq_frequency(
        frequency_hz=freq_hz,
        sample_rate=sample_rate,
        label="freq_hz",
    )
    _validate_eq_q(q)
    return tf2sos(
        *spec.design_fn(
            sample_rate=sample_rate,
            center_hz=freq_hz,
            q=q,
            gain_db=gain_db,
        )
    )


# ---------------------------------------------------------------------------
# Chow Tape Model VST3 (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_LSP_PRELOAD_LIBRARIES: tuple[Path, ...] = (
    (
        Path.home() / ".local" / "lib" / "lsp-runtime" / "libpixman-1.so.0.38.4",
        Path.home() / ".local" / "lib" / "lsp-runtime" / "libxcb-render.so.0.0.0",
        Path.home() / ".local" / "lib" / "lsp-runtime" / "libcairo.so.2.11600.0",
    )
    if sys.platform == "linux"
    else ()
)

_PLUGIN_SPECS: dict[str, ExternalPluginSpec] = {
    "lsp_compressor_stereo": ExternalPluginSpec(
        name="lsp_compressor_stereo",
        path=Path.home() / ".vst3" / "lsp-plugins.vst3",
        format="vst3",
        bundle_plugin_name="Compressor Stereo",
        preload_libraries=_LSP_PRELOAD_LIBRARIES,
    ),
    "lsp_limiter_stereo": ExternalPluginSpec(
        name="lsp_limiter_stereo",
        path=Path.home() / ".vst3" / "lsp-plugins.vst3",
        format="vst3",
        bundle_plugin_name="Limiter Stereo",
        preload_libraries=_LSP_PRELOAD_LIBRARIES,
    ),
    "lsp_compressor_stereo_vst2": ExternalPluginSpec(
        name="lsp_compressor_stereo_vst2",
        path=Path.home() / ".vst" / "lsp-plugins-vst-compressor-stereo.so",
        format="vst2",
    ),
    "chow_tape": ExternalPluginSpec(
        name="chow_tape",
        path=Path.home() / ".vst3" / "CHOWTapeModel.vst3",
    ),
    "tal_chorus_lx": ExternalPluginSpec(
        name="tal_chorus_lx",
        path=Path.home() / ".vst3" / "TAL-Chorus-LX.vst3",
    ),
    "tal_reverb2": ExternalPluginSpec(
        name="tal_reverb2",
        path=Path.home() / ".vst3" / "TAL-Reverb-2.vst3",
    ),
    "dragonfly_plate": ExternalPluginSpec(
        name="dragonfly_plate",
        path=Path.home() / ".vst3" / "DragonflyPlateReverb.vst3",
    ),
    "dragonfly_room": ExternalPluginSpec(
        name="dragonfly_room",
        path=Path.home() / ".vst3" / "DragonflyRoomReverb.vst3",
    ),
    "dragonfly_hall": ExternalPluginSpec(
        name="dragonfly_hall",
        path=Path.home() / ".vst3" / "DragonflyHallReverb.vst3",
    ),
    "dragonfly_early": ExternalPluginSpec(
        name="dragonfly_early",
        path=Path.home() / ".vst3" / "DragonflyEarlyReflections.vst3",
    ),
    "byod": ExternalPluginSpec(
        name="byod",
        path=Path.home() / ".vst3" / "BYOD.vst3",
    ),
    "chow_matrix": ExternalPluginSpec(
        name="chow_matrix",
        path=Path.home() / ".vst3" / "ChowMatrix.vst3",
    ),
    "airwindows": ExternalPluginSpec(
        name="airwindows",
        path=Path.home() / ".vst3" / "Airwindows Consolidated.vst3",
    ),
    "surge_xt": ExternalPluginSpec(
        name="surge_xt",
        path=Path.home() / ".vst3" / "Surge XT.vst3",
    ),
    "chow_centaur": ExternalPluginSpec(
        name="chow_centaur",
        path=Path.home() / ".vst3" / "ChowCentaur.vst3",
    ),
    "chow_kick": ExternalPluginSpec(
        name="chow_kick",
        path=Path.home() / ".vst3" / "ChowKick.vst3",
    ),
    "chow_multi_tool": ExternalPluginSpec(
        name="chow_multi_tool",
        path=Path.home() / ".vst3" / "ChowMultiTool.vst3",
    ),
    "chow_phaser_mono": ExternalPluginSpec(
        name="chow_phaser_mono",
        path=Path.home() / ".vst3" / "ChowPhaserMono.vst3",
    ),
    "chow_phaser_stereo": ExternalPluginSpec(
        name="chow_phaser_stereo",
        path=Path.home() / ".vst3" / "ChowPhaserStereo.vst3",
    ),
    "tal_reverb3": ExternalPluginSpec(
        name="tal_reverb3",
        path=Path.home() / ".vst3" / "TAL-Reverb-3.vst3",
    ),
    "valhalla_supermassive": ExternalPluginSpec(
        name="valhalla_supermassive",
        path=Path.home() / ".vst3" / "ValhallaSupermassive.vst3",
    ),
    "valhalla_freq_echo": ExternalPluginSpec(
        name="valhalla_freq_echo",
        path=Path.home() / ".vst3" / "ValhallaFreqEcho.vst3",
    ),
    "valhalla_space_mod": ExternalPluginSpec(
        name="valhalla_space_mod",
        path=Path.home() / ".vst3" / "ValhallaSpaceModulator.vst3",
    ),
    "tdr_kotelnikov": ExternalPluginSpec(
        name="tdr_kotelnikov",
        path=Path.home() / ".vst3" / "TDR Kotelnikov.vst3",
    ),
    "mjuc_jr": ExternalPluginSpec(
        name="mjuc_jr",
        path=Path.home() / ".vst3" / "MJUCjr.vst3",
    ),
    "ivgi": ExternalPluginSpec(
        name="ivgi",
        path=Path.home() / ".vst3" / "IVGI2.vst3",
    ),
    "fetish": ExternalPluginSpec(
        name="fetish",
        path=Path.home() / ".vst3" / "FETish.vst3",
    ),
    "lala": ExternalPluginSpec(
        name="lala",
        path=Path.home() / ".vst3" / "LALA.vst3",
    ),
    "brit_channel": ExternalPluginSpec(
        name="brit_channel",
        path=Path.home() / ".vst3" / "BritChannel.vst3",
    ),
    "rare_se": ExternalPluginSpec(
        name="rare_se",
        path=Path.home() / ".vst3" / "RareSE.vst3",
    ),
    "brit_pre": ExternalPluginSpec(
        name="brit_pre",
        path=Path.home() / ".vst3" / "BritPre.vst3",
    ),
    "britpressor": ExternalPluginSpec(
        name="britpressor",
        path=Path.home() / ".vst3" / "Britpressor.vst3",
    ),
    "distox": ExternalPluginSpec(
        name="distox",
        path=Path.home() / ".vst3" / "Distox.vst3",
    ),
    "fet_drive": ExternalPluginSpec(
        name="fet_drive",
        path=Path.home() / ".vst3" / "FetDrive.vst3",
    ),
    "kolin": ExternalPluginSpec(
        name="kolin",
        path=Path.home() / ".vst3" / "Kolin.vst3",
    ),
    "laea": ExternalPluginSpec(
        name="laea",
        path=Path.home() / ".vst3" / "LAEA.vst3",
    ),
    "merica": ExternalPluginSpec(
        name="merica",
        path=Path.home() / ".vst3" / "MERICA.vst3",
    ),
    "prebox": ExternalPluginSpec(
        name="prebox",
        path=Path.home() / ".vst3" / "PreBOX.vst3",
    ),
    "tuba": ExternalPluginSpec(
        name="tuba",
        path=Path.home() / ".vst3" / "TUBA.vst3",
    ),
    "vital": ExternalPluginSpec(
        name="vital",
        path=Path.home() / ".vst3" / "Vital.vst3",
    ),
    "surge_xt_fx": ExternalPluginSpec(
        name="surge_xt_fx",
        path=Path.home() / ".vst3" / "Surge XT Effects.vst3",
    ),
}
_loaded_external_plugins: dict[tuple[str, str, Path, str | None], Any] = {}
_airwindows_base_preset: bytes | None = None


def _normalize_plugin_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _get_external_plugin_spec(
    plugin_name: str | None = None,
    plugin_path: str | Path | None = None,
    plugin_format: str = "vst3",
    host: str = "pedalboard",
) -> ExternalPluginSpec:
    if plugin_name is not None:
        if plugin_name not in _PLUGIN_SPECS:
            raise ValueError(
                f"Unknown plugin {plugin_name!r}. Choose from: {sorted(_PLUGIN_SPECS)}"
            )
        return _PLUGIN_SPECS[plugin_name]

    if plugin_path is None:
        raise ValueError("Provide either plugin_name or plugin_path")

    return ExternalPluginSpec(
        name=str(plugin_path),
        path=_normalize_plugin_path(plugin_path),
        format=plugin_format,
        host=host,
    )


def has_external_plugin(plugin_name: str) -> bool:
    """Return whether a registered external plugin and its runtime deps exist."""
    spec = _get_external_plugin_spec(plugin_name=plugin_name)
    if not spec.path.exists():
        return False
    return all(path.exists() for path in spec.preload_libraries)


def _preload_shared_libraries(library_paths: tuple[Path, ...]) -> None:
    for library_path in library_paths:
        if not library_path.exists():
            raise FileNotFoundError(
                f"Required shared library not found at {library_path}"
            )
        ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)


def load_external_plugin(
    plugin_name: str | None = None,
    plugin_path: str | Path | None = None,
    plugin_format: str = "vst3",
    host: str = "pedalboard",
) -> Any:
    spec = _get_external_plugin_spec(
        plugin_name=plugin_name,
        plugin_path=plugin_path,
        plugin_format=plugin_format,
        host=host,
    )
    cache_key = (spec.host, spec.format, spec.path, spec.bundle_plugin_name)
    if cache_key in _loaded_external_plugins:
        return _loaded_external_plugins[cache_key]

    if spec.host != "pedalboard":
        raise ValueError(
            f"Unsupported plugin host: {spec.host!r}. Only 'pedalboard' is currently supported."
        )
    if spec.format != "vst3":
        raise ValueError(
            "Unsupported plugin format for the current backend: "
            f"{spec.format!r}. The 'pedalboard' backend here supports VST3 only, "
            "so Linux `.so` VST2 plugins such as LSP cannot be loaded until we add "
            "a separate VST2-capable host."
        )
    if not spec.path.exists():
        raise FileNotFoundError(f"Plugin not found at {spec.path}")

    os.environ.setdefault("DISPLAY", "")
    from pedalboard import load_plugin  # noqa: PLC0415  # type: ignore[attr-defined]

    if spec.preload_libraries:
        _preload_shared_libraries(spec.preload_libraries)

    plugin = load_plugin(str(spec.path), plugin_name=spec.bundle_plugin_name)
    _loaded_external_plugins[cache_key] = plugin
    return plugin


def _configure_plugin_attributes(plugin: Any, params: dict[str, Any]) -> None:
    for key, value in params.items():
        if not hasattr(plugin, key):
            raise ValueError(f"Plugin {type(plugin).__name__} has no parameter {key!r}")
        setattr(plugin, key, value)


def _apply_plugin_processor(
    signal: np.ndarray,
    *,
    plugin_name: str | None = None,
    plugin_path: str | Path | None = None,
    plugin_format: str = "vst3",
    host: str = "pedalboard",
    params: dict[str, Any] | None = None,
    configurer: PluginConfigurer | None = None,
) -> np.ndarray:
    plugin = load_external_plugin(
        plugin_name=plugin_name,
        plugin_path=plugin_path,
        plugin_format=plugin_format,
        host=host,
    )
    if hasattr(plugin, "reset"):
        plugin.reset()
    resolved_params = dict(params or {})
    if configurer is not None:
        configurer(plugin, resolved_params)
    else:
        _configure_plugin_attributes(plugin, resolved_params)

    stereo_in = _ensure_stereo(signal).astype(np.float32)
    stereo_out = plugin(stereo_in, SAMPLE_RATE)
    return _match_input_layout(_coerce_signal_layout(stereo_out), signal)


def tone(
    freq: float,
    duration: float,
    amp: float = 1.0,
    harmonic_rolloff: float = 0.5,
    n_harmonics: int = 6,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Additive synthesis: fundamental + harmonics with geometric rolloff."""
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    signal = np.zeros(n_samples)
    total_amp = 0.0
    for n in range(1, n_harmonics + 1):
        partial_freq = freq * n
        if partial_freq >= sample_rate / 2:
            break
        partial_amp = harmonic_rolloff ** (n - 1)
        signal += partial_amp * np.sin(2 * np.pi * partial_freq * t)
        total_amp += partial_amp
    return amp * signal / total_amp


def adsr(
    signal: np.ndarray,
    attack: float = 0.04,
    decay: float = 0.1,
    sustain_level: float = 0.75,
    release: float = 0.3,
    sample_rate: int = SAMPLE_RATE,
    hold_duration: float | None = None,
    vca_nonlinearity: float = 0.0,
    attack_power: float = 1.0,
    decay_power: float = 1.0,
    release_power: float = 1.0,
    attack_target: float = 1.0,
) -> np.ndarray:
    """Apply ADSR amplitude envelope with optional per-stage curve shaping.

    Each stage transitions from start to end with a shaped position
    ``s = p**power`` where ``p`` is the linear ramp ``[0, 1)`` across the
    stage. ``power == 1.0`` yields the linear behavior of the classic ADSR.
    Values below 1.0 give concave (fast-start) curves; values above 1.0 give
    convex (slow-start) curves. Per-stage powers are clamped to ``[0.1, 8.0]``.

    ``attack_target`` is a VCV Fundamental-style trick (``ATT_TARGET``):
    the attack stage shapes a ramp toward ``attack_target`` and then clips at
    ``1.0``. With ``attack_target == 1.0`` and ``attack_power == 1.0`` the
    output is identical to the original linear envelope. With
    ``attack_target > 1.0`` (typical: 1.2) the attack reaches 1.0 with more
    slope at the top — a "pokey" analog-feeling attack instead of an
    asymptotic approach. ``attack_target`` is clamped to ``[1.0, 1.5]``.

    The per-stage exponent math (``y = p**power``) is public-domain. The
    overshoot-target idiom comes from Vital DAHDSR, OB-Xd's exponential ADSR,
    and VCV Fundamental ``ADSR.cpp``; this is a clean re-implementation from
    the algorithmic description.
    """
    _MIN_ATTACK_S = 0.003  # 3 ms floor — prevents onset clicks
    _POWER_MIN = 0.1
    _POWER_MAX = 8.0
    _ATTACK_TARGET_MIN = 1.0
    _ATTACK_TARGET_MAX = 1.5
    attack = max(attack, _MIN_ATTACK_S)
    release = max(release, _MIN_ATTACK_S)
    attack_power = float(np.clip(attack_power, _POWER_MIN, _POWER_MAX))
    decay_power = float(np.clip(decay_power, _POWER_MIN, _POWER_MAX))
    release_power = float(np.clip(release_power, _POWER_MIN, _POWER_MAX))
    attack_target = float(
        np.clip(attack_target, _ATTACK_TARGET_MIN, _ATTACK_TARGET_MAX)
    )
    n = len(signal)
    n_attack = int(attack * sample_rate)
    n_decay = int(decay * sample_rate)
    n_release = int(release * sample_rate)
    hold_samples = (
        max(0, n - n_release)
        if hold_duration is None
        else max(0, int(hold_duration * sample_rate))
    )
    hold_samples = min(hold_samples, n)
    release_samples = max(0, min(n_release, n - hold_samples))

    envelope = np.zeros(n, dtype=np.float64)
    cursor = 0

    attack_samples = min(n_attack, hold_samples)
    if attack_samples > 0:
        p = np.linspace(0.0, 1.0, attack_samples, endpoint=False)
        shaped_position = p if attack_power == 1.0 else p**attack_power
        # Ramp toward attack_target then clamp at 1.0 (VCV ATT_TARGET idiom).
        attack_ramp = shaped_position * attack_target
        if attack_target > 1.0:
            np.minimum(attack_ramp, 1.0, out=attack_ramp)
        envelope[cursor : cursor + attack_samples] = attack_ramp
        cursor += attack_samples

    decay_samples = min(n_decay, hold_samples - cursor)
    if decay_samples > 0:
        p = np.linspace(0.0, 1.0, decay_samples, endpoint=False)
        shaped_position = p if decay_power == 1.0 else p**decay_power
        envelope[cursor : cursor + decay_samples] = (
            1.0 + (sustain_level - 1.0) * shaped_position
        )
        cursor += decay_samples

    sustain_samples = hold_samples - cursor
    if sustain_samples > 0:
        envelope[cursor : cursor + sustain_samples] = sustain_level
        cursor += sustain_samples

    release_start_level = float(envelope[cursor - 1]) if cursor > 0 else 0.0

    if release_samples > 0:
        if release_power == 1.0:
            envelope[cursor : cursor + release_samples] = np.linspace(
                release_start_level,
                0.0,
                release_samples,
                endpoint=True,
            )
        else:
            p = np.linspace(0.0, 1.0, release_samples, endpoint=True)
            shaped_position = p**release_power
            envelope[cursor : cursor + release_samples] = release_start_level * (
                1.0 - shaped_position
            )
        cursor += release_samples

    shaped = signal * envelope
    if vca_nonlinearity > 0.0:
        drive = 1.0 + vca_nonlinearity * 3.0 * envelope
        shaped = np.where(
            drive > 1e-6,
            np.tanh(drive * shaped) / np.tanh(drive),
            shaped,
        )
    return shaped


def stack(*signals: np.ndarray) -> np.ndarray:
    """Sum signals of potentially different lengths into one array."""
    max_len = max(len(s) for s in signals)
    out = np.zeros(max_len)
    for signal in signals:
        out[: len(signal)] += signal
    return out


def at(signal: np.ndarray, offset_seconds: float) -> np.ndarray:
    """Pad signal with silence at the front so it starts at offset_seconds."""
    pad = np.zeros(int(offset_seconds * SAMPLE_RATE))
    return np.concatenate([pad, signal])


def at_sample_rate(
    signal: np.ndarray, offset_seconds: float, sample_rate: int
) -> np.ndarray:
    """Pad signal with silence at the front using an explicit sample rate."""
    pad = np.zeros(int(offset_seconds * sample_rate))
    return np.concatenate([pad, signal])


def sequence(*segments: np.ndarray, gap: float = 0.05) -> np.ndarray:
    """Concatenate segments with a short silence between each."""
    silence = np.zeros(int(gap * SAMPLE_RATE))
    parts: list[np.ndarray] = []
    for index, segment in enumerate(segments):
        parts.append(segment)
        if index < len(segments) - 1:
            parts.append(silence)
    return np.concatenate(parts)


def lowpass(
    signal: np.ndarray, cutoff_hz: float, sample_rate: int, order: int = 2
) -> np.ndarray:
    """Apply a stable low-pass filter to a mono signal."""
    nyquist = sample_rate / 2.0
    if cutoff_hz <= 0:
        return np.zeros_like(signal)
    if cutoff_hz >= nyquist * 0.995:
        return signal
    sos = butter(order, cutoff_hz / nyquist, btype="lowpass", output="sos")
    return np.asarray(sosfilt(sos, signal))


def highpass(
    signal: np.ndarray, cutoff_hz: float, sample_rate: int, order: int = 2
) -> np.ndarray:
    """Apply a stable high-pass filter to a mono signal."""
    nyquist = sample_rate / 2.0
    if cutoff_hz <= 0:
        return signal
    if cutoff_hz >= nyquist * 0.995:
        return np.zeros_like(signal)
    sos = butter(order, cutoff_hz / nyquist, btype="highpass", output="sos")
    return np.asarray(sosfilt(sos, signal))


def _linkwitz_riley_lowpass(
    signal: np.ndarray, cutoff_hz: float, sample_rate: int
) -> np.ndarray:
    """Linkwitz-Riley 4th-order lowpass (two cascaded 2nd-order Butterworth).

    Used for complementary multiband crossovers that sum to a flat magnitude
    response when the two halves are recombined.
    """
    nyquist = sample_rate / 2.0
    if cutoff_hz <= 0.0:
        return np.zeros_like(signal)
    if cutoff_hz >= nyquist * 0.995:
        return np.asarray(signal, dtype=np.float64)
    sos = butter(2, cutoff_hz / nyquist, btype="lowpass", output="sos")
    y = sosfilt(sos, signal)
    y = sosfilt(sos, y)
    return np.asarray(y, dtype=np.float64)


def _linkwitz_riley_highpass(
    signal: np.ndarray, cutoff_hz: float, sample_rate: int
) -> np.ndarray:
    """Linkwitz-Riley 4th-order highpass (two cascaded 2nd-order Butterworth)."""
    nyquist = sample_rate / 2.0
    if cutoff_hz <= 0.0:
        return np.asarray(signal, dtype=np.float64)
    if cutoff_hz >= nyquist * 0.995:
        return np.zeros_like(signal)
    sos = butter(2, cutoff_hz / nyquist, btype="highpass", output="sos")
    y = sosfilt(sos, signal)
    y = sosfilt(sos, y)
    return np.asarray(y, dtype=np.float64)


def _apply_tilt_eq(
    signal: np.ndarray,
    *,
    sample_rate: int,
    tilt_db: float,
    pivot_hz: float,
    q: float = 0.707,
) -> np.ndarray:
    """Apply a simple tilt EQ around a pivot using complementary shelving filters."""
    if tilt_db == 0.0:
        return np.asarray(signal, dtype=np.float64)

    nyquist = sample_rate / 2.0
    if pivot_hz <= 0.0 or pivot_hz >= nyquist * 0.995:
        raise ValueError("tilt_pivot_hz must be between 0 and Nyquist")

    low_shelf_sos = tf2sos(
        *_design_low_shelf_biquad(
            sample_rate=sample_rate,
            center_hz=pivot_hz,
            q=q,
            gain_db=-(tilt_db / 2.0),
        )
    )
    high_shelf_sos = tf2sos(
        *_design_high_shelf_biquad(
            sample_rate=sample_rate,
            center_hz=pivot_hz,
            q=q,
            gain_db=tilt_db / 2.0,
        )
    )
    tilted = sosfilt(low_shelf_sos, np.asarray(signal, dtype=np.float64))
    tilted = sosfilt(high_shelf_sos, tilted)
    return np.asarray(tilted, dtype=np.float64)


def _shape_reverb_return(
    signal: np.ndarray,
    *,
    sample_rate: int,
    highpass_hz: float = 0.0,
    lowpass_hz: float = 0.0,
    tilt_db: float = 0.0,
    tilt_pivot_hz: float = 1_500.0,
) -> np.ndarray:
    """Apply basic tone shaping to a wet reverb return."""
    if highpass_hz < 0.0:
        raise ValueError("highpass_hz must be non-negative")
    if lowpass_hz < 0.0:
        raise ValueError("lowpass_hz must be non-negative")
    if highpass_hz > 0.0 and lowpass_hz > 0.0 and highpass_hz >= lowpass_hz:
        raise ValueError("highpass_hz must be lower than lowpass_hz")

    def _process_channel(channel: np.ndarray) -> np.ndarray:
        shaped = np.asarray(channel, dtype=np.float64)
        if highpass_hz > 0.0:
            shaped = highpass(shaped, cutoff_hz=highpass_hz, sample_rate=sample_rate)
        if lowpass_hz > 0.0:
            shaped = lowpass(shaped, cutoff_hz=lowpass_hz, sample_rate=sample_rate)
        if tilt_db != 0.0:
            shaped = _apply_tilt_eq(
                shaped,
                sample_rate=sample_rate,
                tilt_db=tilt_db,
                pivot_hz=tilt_pivot_hz,
            )
        return np.asarray(shaped, dtype=np.float64)

    return _apply_per_channel(signal, _process_channel)


def apply_eq(
    signal: np.ndarray,
    *,
    bands: list[dict[str, Any]],
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Apply an ordered minimum-phase EQ made from native IIR bands."""
    if not bands:
        raise ValueError("bands must be a non-empty list")

    def _process_channel(channel: np.ndarray) -> np.ndarray:
        processed = np.asarray(channel, dtype=np.float64)
        for band in bands:
            band_sos = _design_eq_band_sos(band=band, sample_rate=sample_rate)
            processed = sosfilt(band_sos, processed)
        return np.asarray(processed, dtype=np.float64)

    return _apply_per_channel(signal, _process_channel)


def _time_constant_to_coeff(time_ms: float, sample_rate: int) -> float:
    if time_ms <= 0.0:
        raise ValueError("time constant must be positive")
    return float(np.exp(-1.0 / (0.001 * time_ms * sample_rate)))


def _linked_detector_signal(signal: np.ndarray) -> np.ndarray:
    normalized = _coerce_signal_layout(signal)
    if normalized.ndim == 1:
        return np.abs(normalized)
    return np.max(np.abs(normalized), axis=0)


def _compute_detector_rms_envelope(
    *,
    detector_source: np.ndarray,
    sample_rate: int,
    window_ms: float = 50.0,
) -> np.ndarray:
    """Compute a 50 ms non-overlapping RMS envelope of the detector signal.

    The detector source should already have any detector-band EQ applied.
    Mono or stereo input is reduced to a linked detector (max across
    channels); the squared signal is divided into non-overlapping
    ``window_ms`` windows and each window RMS is returned. The envelope
    length is ``n_samples // window_samples`` (i.e., windowed, not
    per-sample — see :func:`_expand_windowed_envelope_to_samples` for
    expansion).
    """
    linked = _linked_detector_signal(detector_source)
    n = linked.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    window_samples = max(1, int(round(window_ms * 1e-3 * sample_rate)))
    squared = linked.astype(np.float64) ** 2
    if n < window_samples:
        return np.array([float(np.sqrt(np.mean(squared)))], dtype=np.float64)
    n_windows = n // window_samples
    trimmed = squared[: n_windows * window_samples]
    windowed = trimmed.reshape(n_windows, window_samples)
    return np.sqrt(np.mean(windowed, axis=1)).astype(np.float64)


def _expand_windowed_envelope_to_samples(
    windowed_envelope: np.ndarray,
    *,
    sample_count: int,
    window_samples: int,
) -> np.ndarray:
    """Broadcast a per-window envelope back to sample-rate by repetition.

    The last partial window (if the signal doesn't divide evenly) repeats
    the final window value. Returns ``sample_count`` samples.
    """
    if windowed_envelope.size == 0:
        return np.zeros(sample_count, dtype=np.float64)
    expanded = np.repeat(windowed_envelope, window_samples)
    if expanded.size < sample_count:
        tail = np.full(sample_count - expanded.size, windowed_envelope[-1])
        expanded = np.concatenate([expanded, tail])
    elif expanded.size > sample_count:
        expanded = expanded[:sample_count]
    return expanded


def _simulate_compressor_gr_trace_db(
    *,
    detector_trace: np.ndarray,
    attack_coeff: float,
    release_coeff: float,
    threshold_db: float,
    ratio: float,
    knee_db: float,
    is_rms: bool,
    release_stage1_coeff: float,
    release_stage2_coeff: float,
) -> np.ndarray:
    """Simulate the feedforward compressor and return the GR trace in dB.

    Shares the same kernel as the main feedforward path so the solver and
    the runtime use bit-identical math. Returns a trace where each sample
    is the instantaneous attenuation in dB as a non-negative value (so
    ``gr_db[i] = 4.0`` means the compressor is attenuating by 4 dB at
    sample ``i``).
    """
    smoothed_level = _compressor_detector_smooth(
        detector_trace, attack_coeff, release_coeff, is_rms
    )
    level_db = 20.0 * np.log10(np.maximum(smoothed_level, 1e-12))
    target_gain_db = _compressor_gain_db_vec(level_db, threshold_db, ratio, knee_db)
    smoothed_gain_db = _compressor_smooth_gain_loop(
        target_gain_db, attack_coeff, release_stage1_coeff, release_stage2_coeff
    )
    return np.maximum(0.0, -smoothed_gain_db)


def _solve_compressor_threshold_for_target_avg_gr(
    *,
    detector_trace: np.ndarray,
    active_mask: np.ndarray,
    attack_coeff: float,
    release_coeff: float,
    ratio: float,
    knee_db: float,
    is_rms: bool,
    release_stage1_coeff: float,
    release_stage2_coeff: float,
    target_avg_gr_db: float,
    search_low_db: float,
    search_high_db: float,
    max_iterations: int = 10,
    tolerance_db: float = 0.25,
) -> tuple[float, float, int]:
    """Binary-search ``threshold_db`` for target average GR on active samples.

    Returns ``(threshold_db, measured_avg_gr_db, iterations)``. The measure
    is "mean attenuation in dB over samples where ``active_mask`` is True,"
    matching the musician intuition that "4 dB of compression" means "you'll
    see 4 dB avg GR on the parts the compressor is doing work."

    Higher threshold -> less GR (monotonic), so standard bisection converges.
    If ``active_mask`` is empty the caller is responsible for falling back;
    we short-circuit and return ``(search_high_db, 0.0, 0)``.
    """
    if not np.any(active_mask):
        return float(search_high_db), 0.0, 0

    low = float(search_low_db)
    high = float(search_high_db)
    best_threshold = high
    best_measured = 0.0
    best_abs_error = float("inf")
    iterations = 0
    for _ in range(max_iterations):
        iterations += 1
        mid = 0.5 * (low + high)
        gr_trace = _simulate_compressor_gr_trace_db(
            detector_trace=detector_trace,
            attack_coeff=attack_coeff,
            release_coeff=release_coeff,
            threshold_db=mid,
            ratio=ratio,
            knee_db=knee_db,
            is_rms=is_rms,
            release_stage1_coeff=release_stage1_coeff,
            release_stage2_coeff=release_stage2_coeff,
        )
        measured_avg_gr = float(np.mean(gr_trace[active_mask]))
        abs_error = abs(measured_avg_gr - target_avg_gr_db)
        if abs_error < best_abs_error:
            best_abs_error = abs_error
            best_threshold = mid
            best_measured = measured_avg_gr
        if abs_error < tolerance_db:
            return float(mid), measured_avg_gr, iterations
        if measured_avg_gr > target_avg_gr_db:
            # Too much GR — raise threshold (less compression).
            low = mid
        else:
            # Too little GR — lower threshold (more compression).
            high = mid
    return float(best_threshold), float(best_measured), iterations


def _resolve_compressor_release_times(
    *,
    release_ms: float,
    release_tail_ms: float | None,
) -> tuple[float, float]:
    if release_ms <= 0.0:
        raise ValueError("release_ms must be positive")
    if release_tail_ms is None:
        return float(release_ms), float(release_ms)
    if release_tail_ms <= 0.0:
        raise ValueError("release_tail_ms must be positive")
    if release_tail_ms < release_ms:
        raise ValueError("release_tail_ms must be greater than or equal to release_ms")
    return float(release_ms), float(release_tail_ms)


def _lookahead_samples(lookahead_ms: float, sample_rate: int) -> int:
    if not np.isfinite(lookahead_ms):
        raise ValueError("lookahead_ms must be finite")
    if lookahead_ms < 0.0:
        raise ValueError("lookahead_ms must be non-negative")
    return int(round((lookahead_ms / 1000.0) * sample_rate))


def _build_effect_warning(
    *,
    severity: str,
    code: str,
    message: str,
    **metrics: float | int | str,
) -> EffectAnalysisWarning:
    return EffectAnalysisWarning(
        severity=severity,
        code=code,
        message=message,
        metrics=metrics,
    )


def _populate_per_hit_effect_metrics(
    *,
    metrics: dict[str, float | int | str],
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    sample_rate: int,
) -> None:
    """Measure per-hit transient preservation for percussive voice effects.

    Detects onsets on the input (pre-effect) signal and reuses those onset
    indices on the output so the comparison is apples-to-apples. Writes
    ``transient_kill_db`` (input p50 transient-peak ratio -> output p50),
    ``hit_count``, and ``transient_peak_ratio_{input,output}`` to *metrics*.
    """
    mono_input = to_mono_reference(input_signal)
    mono_output = to_mono_reference(output_signal)
    if mono_input.size == 0 or mono_output.size == 0:
        metrics["hit_count"] = 0
        metrics["transient_kill_db"] = 0.0
        return
    onset_indices = detect_percussive_onsets(mono_input, sample_rate=sample_rate)
    if onset_indices.size == 0:
        metrics["hit_count"] = 0
        metrics["transient_kill_db"] = 0.0
        return
    input_summary: PerHitSummary = per_hit_transient_metrics(
        mono_input,
        sample_rate=sample_rate,
        onset_sample_indices=onset_indices,
    )
    output_summary: PerHitSummary = per_hit_transient_metrics(
        mono_output,
        sample_rate=sample_rate,
        onset_sample_indices=onset_indices,
    )
    metrics["hit_count"] = int(input_summary.hit_count)
    metrics["transient_peak_ratio_input_p50"] = round(
        input_summary.transient_peak_ratio_p50, 4
    )
    metrics["transient_peak_ratio_output_p50"] = round(
        output_summary.transient_peak_ratio_p50, 4
    )
    input_ratio = max(input_summary.transient_peak_ratio_p50, 1e-6)
    output_ratio = max(output_summary.transient_peak_ratio_p50, 1e-6)
    metrics["transient_kill_db"] = round(
        20.0 * math.log10(output_ratio / input_ratio), 2
    )


def _build_effect_analysis_entry(
    *,
    index: int,
    kind: str,
    display_name: str,
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    sample_rate: int,
    native_metrics: dict[str, float | int | str] | None = None,
    percussive: bool = False,
) -> EffectAnalysisEntry:
    input_peak_dbfs = _signal_peak_dbfs(input_signal)
    output_peak_dbfs = _signal_peak_dbfs(output_signal)
    input_true_peak_dbfs = _safe_amp_to_db(estimate_true_peak_amplitude(input_signal))
    output_true_peak_dbfs = _safe_amp_to_db(estimate_true_peak_amplitude(output_signal))
    input_crest_factor_db = _signal_crest_factor_db(input_signal)
    output_crest_factor_db = _signal_crest_factor_db(output_signal)
    input_clipped_fraction = _clipped_sample_fraction(input_signal)
    output_clipped_fraction = _clipped_sample_fraction(output_signal)
    input_near_full_scale_fraction = _near_full_scale_fraction(input_signal)
    output_near_full_scale_fraction = _near_full_scale_fraction(output_signal)
    input_freqs, input_magnitude_db = _average_spectrum_db(
        input_signal,
        sample_rate=sample_rate,
    )
    output_freqs, output_magnitude_db = _average_spectrum_db(
        output_signal,
        sample_rate=sample_rate,
    )
    input_high_band_db = _mean_band_energy_db(
        input_magnitude_db,
        freqs=input_freqs,
        low_hz=2_000.0,
        high_hz=8_000.0,
    )
    output_high_band_db = _mean_band_energy_db(
        output_magnitude_db,
        freqs=output_freqs,
        low_hz=2_000.0,
        high_hz=8_000.0,
    )
    input_a_weighted_high_band_db = a_weighted_mean_band_energy_db(
        input_freqs,
        input_magnitude_db,
        low_hz=2_000.0,
        high_hz=8_000.0,
    )
    output_a_weighted_high_band_db = a_weighted_mean_band_energy_db(
        output_freqs,
        output_magnitude_db,
        low_hz=2_000.0,
        high_hz=8_000.0,
    )
    spectral_centroid_delta_hz = _spectral_centroid_hz(
        output_signal,
        sample_rate=sample_rate,
    ) - _spectral_centroid_hz(
        input_signal,
        sample_rate=sample_rate,
    )

    input_imd = intermodulation_ratio(input_freqs, input_magnitude_db)
    output_imd = intermodulation_ratio(
        output_freqs,
        output_magnitude_db,
        f1_override_hz=input_imd.f1_hz if input_imd.f1_hz > 0.0 else None,
        f2_override_hz=input_imd.f2_hz if input_imd.f2_hz > 0.0 else None,
    )

    # Signal-level THD: use the input dominant frequency as reference for both
    # so the delta isolates distortion the effect introduced, not spectral reshaping.
    input_dominant_hz = (
        float(input_freqs[np.argmax(input_magnitude_db)])
        if input_freqs.size > 0
        else 0.0
    )
    input_thd_pct, input_thd_character = compute_signal_thd(
        input_freqs, input_magnitude_db, input_dominant_hz
    )
    output_thd_pct, output_thd_character = compute_signal_thd(
        output_freqs, output_magnitude_db, input_dominant_hz
    )
    thd_delta_pct = output_thd_pct - input_thd_pct

    metrics: dict[str, float | int | str] = {
        "input_peak_dbfs": round(input_peak_dbfs, 2),
        "output_peak_dbfs": round(output_peak_dbfs, 2),
        "peak_delta_db": round(output_peak_dbfs - input_peak_dbfs, 2),
        "input_true_peak_dbfs": round(input_true_peak_dbfs, 2),
        "output_true_peak_dbfs": round(output_true_peak_dbfs, 2),
        "true_peak_delta_db": round(output_true_peak_dbfs - input_true_peak_dbfs, 2),
        "input_crest_factor_db": round(input_crest_factor_db, 2),
        "output_crest_factor_db": round(output_crest_factor_db, 2),
        "crest_factor_delta_db": round(
            output_crest_factor_db - input_crest_factor_db,
            2,
        ),
        "input_clipped_sample_fraction": round(input_clipped_fraction, 6),
        "output_clipped_sample_fraction": round(output_clipped_fraction, 6),
        "clipped_sample_fraction_delta": round(
            output_clipped_fraction - input_clipped_fraction,
            6,
        ),
        "input_near_full_scale_fraction": round(input_near_full_scale_fraction, 6),
        "output_near_full_scale_fraction": round(output_near_full_scale_fraction, 6),
        "near_full_scale_fraction_delta": round(
            output_near_full_scale_fraction - input_near_full_scale_fraction,
            6,
        ),
        "high_band_delta_db": round(output_high_band_db - input_high_band_db, 2),
        "a_weighted_high_band_delta_db": round(
            output_a_weighted_high_band_db - input_a_weighted_high_band_db, 2
        ),
        "spectral_centroid_delta_hz": round(spectral_centroid_delta_hz, 1),
        "input_thd_pct": round(input_thd_pct, 2),
        "input_thd_character": input_thd_character,
        "output_thd_pct": round(output_thd_pct, 2),
        "output_thd_character": output_thd_character,
        "thd_delta_pct": round(thd_delta_pct, 2),
        "imd_ratio_input": round(input_imd.ratio, 4),
        "imd_ratio_output": round(output_imd.ratio, 4),
        "imd_ratio_delta": round(output_imd.ratio - input_imd.ratio, 4),
        "imd_detection": output_imd.detection,
        # Fraction of the *input* signal's loudness-gated blocks that are
        # audibly active (see measure_signal_levels / integrated_lufs).
        # Used by _build_chain_summary to put an energy floor under the
        # chain_papery / chain_brightness_creep / perceptual_brightness_lift
        # warnings: a stage's relative centroid/high-band lift can look huge
        # when its input is silent for most of the piece (e.g. a drum bus
        # during a break), even though nothing audible happened.
        "input_active_window_fraction": round(
            measure_signal_levels(
                input_signal, sample_rate=sample_rate
            ).active_window_fraction,
            4,
        ),
    }
    if percussive:
        _populate_per_hit_effect_metrics(
            metrics=metrics,
            input_signal=input_signal,
            output_signal=output_signal,
            sample_rate=sample_rate,
        )
    if native_metrics is not None:
        metrics.update(native_metrics)

    warnings = _build_effect_analysis_warnings(
        kind=kind,
        metrics=metrics,
        signal_duration_seconds=(output_signal.shape[-1] / sample_rate)
        if output_signal.size > 0
        else 0.0,
    )
    return EffectAnalysisEntry(
        index=index,
        kind=kind,
        display_name=display_name,
        metrics=metrics,
        warnings=warnings,
    )


def _build_effect_analysis_warnings(
    *,
    kind: str,
    metrics: dict[str, float | int | str],
    signal_duration_seconds: float,
) -> list[EffectAnalysisWarning]:
    warnings: list[EffectAnalysisWarning] = []
    if kind == "compressor":
        avg_gain_reduction_db = float(metrics.get("avg_gain_reduction_db", 0.0))
        max_gain_reduction_db = float(metrics.get("max_gain_reduction_db", 0.0))
        p95_gain_reduction_db = float(metrics.get("p95_gain_reduction_db", 0.0))
        longest_run_above_1db_seconds = float(
            metrics.get("longest_run_above_1db_seconds", 0.0)
        )
        if avg_gain_reduction_db < 0.5 and max_gain_reduction_db < 1.5:
            warnings.append(
                _build_effect_warning(
                    severity="warning",
                    code="effect_mostly_inactive",
                    message="compressor appears mostly inactive",
                    avg_gain_reduction_db=round(avg_gain_reduction_db, 2),
                    max_gain_reduction_db=round(max_gain_reduction_db, 2),
                )
            )
        if avg_gain_reduction_db >= 8.0 or max_gain_reduction_db >= 14.0:
            warnings.append(
                _build_effect_warning(
                    severity="severe",
                    code="aggressive_compression",
                    message="compressor gain reduction is very aggressive",
                    avg_gain_reduction_db=round(avg_gain_reduction_db, 2),
                    max_gain_reduction_db=round(max_gain_reduction_db, 2),
                    p95_gain_reduction_db=round(p95_gain_reduction_db, 2),
                )
            )
        elif avg_gain_reduction_db >= 6.0 or max_gain_reduction_db >= 10.0:
            warnings.append(
                _build_effect_warning(
                    severity="warning",
                    code="aggressive_compression",
                    message="compressor is clamping down fairly hard",
                    avg_gain_reduction_db=round(avg_gain_reduction_db, 2),
                    max_gain_reduction_db=round(max_gain_reduction_db, 2),
                    p95_gain_reduction_db=round(p95_gain_reduction_db, 2),
                )
            )
        if signal_duration_seconds >= 15.0 and longest_run_above_1db_seconds >= 20.0:
            warnings.append(
                _build_effect_warning(
                    severity="severe",
                    code="continuous_gain_reduction",
                    message="compressor stays above 1 dB of gain reduction for unusually long stretches",
                    longest_run_above_1db_seconds=round(
                        longest_run_above_1db_seconds,
                        2,
                    ),
                )
            )
        elif signal_duration_seconds >= 10.0 and longest_run_above_1db_seconds >= 10.0:
            warnings.append(
                _build_effect_warning(
                    severity="warning",
                    code="continuous_gain_reduction",
                    message="compressor rarely relaxes below 1 dB of gain reduction",
                    longest_run_above_1db_seconds=round(
                        longest_run_above_1db_seconds,
                        2,
                    ),
                )
            )
        return warnings

    if kind == "clipper":
        shaved_db = float(metrics.get("shaved_db", 0.0))
        active_fraction = float(metrics.get("active_fraction", 0.0))
        high_band_delta_db = float(metrics.get("high_band_delta_db", 0.0))
        # New API exposes knee_width_db; fall back to the legacy hardness key
        # for any cached analysis dicts written before the rewrite.
        knee_width_db = float(metrics.get("knee_width_db", 0.0))
        legacy_hardness = float(metrics.get("hardness", -1.0))
        if legacy_hardness >= 0.0 and "knee_width_db" not in metrics:
            # Roughly inverse-map old hardness into a knee proxy so the
            # brittle-character detector still fires on legacy metrics.
            knee_width_db = (1.0 - legacy_hardness) * 4.0
        # Thresholds calibrated for the poly-knee clipper (April 2026 rewrite).
        # The old 3.0 / 6.0 pair was tuned for the hardness-crossfade kernel,
        # which added ~6.5 dB of 2-8 kHz lift at 3 dB shave on a kick
        # transient.  The poly-knee kernel adds <0.2 dB of HB lift at the
        # same depth (measured across va_trance, amber_room, forge,
        # iron_pulse, iron_pulse_v2); 3-5 dB shave is cleanly musical on
        # real drum-bus content.  Thresholds bumped so the warning fires
        # when the clipper is genuinely working hard, not on routine glue.
        if shaved_db >= 8.0:
            warnings.append(
                _build_effect_warning(
                    severity="severe",
                    code="aggressive_clipping",
                    message="clipper is shaving peaks very aggressively",
                    shaved_db=round(shaved_db, 2),
                    active_fraction=round(active_fraction, 4),
                )
            )
        elif shaved_db >= 5.0:
            warnings.append(
                _build_effect_warning(
                    severity="warning",
                    code="aggressive_clipping",
                    message="clipper is shaving substantial peak energy",
                    shaved_db=round(shaved_db, 2),
                    active_fraction=round(active_fraction, 4),
                )
            )
        # Brittle-kick detector: a narrow-knee clipper pushed into wave-shaping
        # territory sounds papery when it lifts the 2-8 kHz band noticeably.
        # knee_width_db <= 1 dB is functionally hard; the failure mode is the
        # combination of narrow knee, non-trivial shave, and HF lift.
        if knee_width_db <= 1.0 and shaved_db >= 1.5 and high_band_delta_db >= 2.5:
            warnings.append(
                _build_effect_warning(
                    severity="warning",
                    code="clipper_brittle_character",
                    message="clipper may be creating brittle/papery high-frequency character",
                    knee_width_db=round(knee_width_db, 2),
                    shaved_db=round(shaved_db, 2),
                    high_band_delta_db=round(high_band_delta_db, 2),
                )
            )
        if active_fraction < 0.001 and shaved_db < 0.2:
            warnings.append(
                _build_effect_warning(
                    severity="warning",
                    code="effect_mostly_inactive",
                    message="clipper is essentially inactive (input not reaching threshold)",
                    shaved_db=round(shaved_db, 2),
                    active_fraction=round(active_fraction, 4),
                )
            )
        return warnings

    if kind == "limiter":
        max_gain_reduction_db = float(metrics.get("max_gain_reduction_db", 0.0))
        avg_gr_when_active_db = float(
            metrics.get("avg_gain_reduction_when_active_db", 0.0)
        )
        active_fraction = float(metrics.get("active_gain_reduction_fraction", 0.0))
        # max_gain_reduction_db is negative when limiting is active
        if max_gain_reduction_db <= -6.0 or active_fraction >= 0.20:
            warnings.append(
                _build_effect_warning(
                    severity="severe",
                    code="aggressive_limiting",
                    message="limiter is doing heavy work — input is significantly hotter than threshold",
                    max_gain_reduction_db=round(max_gain_reduction_db, 2),
                    avg_gain_reduction_when_active_db=round(avg_gr_when_active_db, 2),
                    active_fraction=round(active_fraction, 4),
                )
            )
        elif max_gain_reduction_db <= -3.0 or active_fraction >= 0.05:
            warnings.append(
                _build_effect_warning(
                    severity="warning",
                    code="aggressive_limiting",
                    message="limiter is regularly active — consider easing the input level",
                    max_gain_reduction_db=round(max_gain_reduction_db, 2),
                    avg_gain_reduction_when_active_db=round(avg_gr_when_active_db, 2),
                    active_fraction=round(active_fraction, 4),
                )
            )
        return warnings

    crest_factor_delta_db = float(metrics.get("crest_factor_delta_db", 0.0))
    clipped_sample_fraction_delta = float(
        metrics.get("clipped_sample_fraction_delta", 0.0)
    )
    near_full_scale_fraction_delta = float(
        metrics.get("near_full_scale_fraction_delta", 0.0)
    )
    spectral_centroid_delta_hz = float(metrics.get("spectral_centroid_delta_hz", 0.0))
    peak_delta_db = float(metrics.get("peak_delta_db", 0.0))
    if (
        abs(crest_factor_delta_db) < 0.3
        and abs(spectral_centroid_delta_hz) < 50.0
        and abs(peak_delta_db) < 0.2
        and near_full_scale_fraction_delta < 0.001
        and clipped_sample_fraction_delta <= 0.0
    ):
        warnings.append(
            _build_effect_warning(
                severity="warning",
                code="effect_mostly_inactive",
                message="effect appears to be doing very little",
                crest_factor_delta_db=round(crest_factor_delta_db, 2),
                spectral_centroid_delta_hz=round(spectral_centroid_delta_hz, 1),
                peak_delta_db=round(peak_delta_db, 2),
            )
        )

    if clipped_sample_fraction_delta >= 0.001 or near_full_scale_fraction_delta >= 0.05:
        warnings.append(
            _build_effect_warning(
                severity="severe",
                code="aggressive_drive",
                message="effect is introducing obvious clipping-like output density",
                clipped_sample_fraction_delta=round(
                    clipped_sample_fraction_delta,
                    6,
                ),
                near_full_scale_fraction_delta=round(
                    near_full_scale_fraction_delta,
                    6,
                ),
                crest_factor_delta_db=round(crest_factor_delta_db, 2),
            )
        )
    elif (
        clipped_sample_fraction_delta > 0.0
        or near_full_scale_fraction_delta >= 0.01
        or crest_factor_delta_db <= -4.0
    ):
        warnings.append(
            _build_effect_warning(
                severity="warning",
                code="aggressive_drive",
                message="effect is adding substantial density or clipping-like behavior",
                clipped_sample_fraction_delta=round(
                    clipped_sample_fraction_delta,
                    6,
                ),
                near_full_scale_fraction_delta=round(
                    near_full_scale_fraction_delta,
                    6,
                ),
                crest_factor_delta_db=round(crest_factor_delta_db, 2),
            )
        )

    # effect_introduced_distortion is driven by IMD-ratio relative growth
    # (not THD delta), which stays content-invariant on harmonically-dense
    # material. See the identical pattern in analysis.py
    # _build_mastering_analysis_risks for the rationale.
    imd_input = float(metrics.get("imd_ratio_input", 0.0))
    imd_output = float(metrics.get("imd_ratio_output", 0.0))
    imd_delta = imd_output - imd_input
    imd_growth_factor = imd_delta / max(imd_input, 0.1)
    imd_detection = str(metrics.get("imd_detection", "single_tone"))
    if imd_detection == "two_tone" and imd_growth_factor >= 1.5:
        warnings.append(
            _build_effect_warning(
                severity="severe",
                code="effect_introduced_distortion",
                message=(
                    "effect grew IMD sharply — output has much more nonlinear "
                    "content than input (likely overdriven)"
                ),
                imd_ratio_delta=round(imd_delta, 4),
                imd_growth_factor=round(imd_growth_factor, 3),
                imd_ratio_input=round(imd_input, 4),
                imd_ratio_output=round(imd_output, 4),
                # Retained for reference; not gating.
                thd_delta_pct=round(float(metrics.get("thd_delta_pct", 0.0)), 2),
            )
        )
    elif imd_detection == "two_tone" and imd_growth_factor >= 0.5:
        warnings.append(
            _build_effect_warning(
                severity="warning",
                code="effect_introduced_distortion",
                message=(
                    "effect added noticeable nonlinear coloration "
                    "(fine if intentional warmth; check the audio)"
                ),
                imd_ratio_delta=round(imd_delta, 4),
                imd_growth_factor=round(imd_growth_factor, 3),
                imd_ratio_input=round(imd_input, 4),
                imd_ratio_output=round(imd_output, 4),
                thd_delta_pct=round(float(metrics.get("thd_delta_pct", 0.0)), 2),
            )
        )

    return warnings


# Effect kinds whose DSP can add harmonic content. Used to gate the
# cumulative-brightness chain warnings (chain_papery,
# perceptual_brightness_lift). Linear / time-based effects (eq, delay,
# reverb, chorus, phaser, bbd_chorus, mod_delay) can lift 2-8 kHz energy at
# the chain output vs. input simply by adding wet signal, which is not
# the "stacked brittleness" failure mode those warnings are meant to
# catch. Measured IMD growth on any stage (tracked separately via
# nonlinear_stage_count) also qualifies a chain as potentially nonlinear.
_NONLINEAR_EFFECT_KINDS: frozenset[str] = frozenset(
    {
        "clipper",
        "preamp",
        "tube",
        "transistor",
        "airwindows",
        "byod",
        "chow_centaur",
    }
)

# Below this fraction of loudness-gated-active input blocks, cap the
# chain_papery / chain_brightness_creep / perceptual_brightness_lift
# severities at "warning" (never "severe"). A chain whose input is mostly
# silence (e.g. a drum bus that drops out for whole sections) can rack up a
# huge *relative* centroid/high-band lift number from a handful of active
# blocks without anything audibly wrong happening across the piece.
_CHAIN_SPARSE_INPUT_ACTIVE_FRACTION = 0.25


def _cap_chain_brightness_severity(
    severity: str,
    *,
    input_active_fraction: float,
) -> str:
    """Cap brightness-family chain severities when the input is mostly silent."""
    if (
        severity == "severe"
        and input_active_fraction < _CHAIN_SPARSE_INPUT_ACTIVE_FRACTION
    ):
        return "warning"
    return severity


def build_chain_summary_from_dicts(
    entries: list[dict[str, Any]],
    *,
    chain_label: str | None = None,
) -> ChainSummary | None:
    """Build a chain summary from serialized per-effect entry dicts.

    Used when effect analysis has already been materialized into the
    manifest (post-render) rather than held as live
    :class:`EffectAnalysisEntry` instances.
    """
    rebuilt = [
        EffectAnalysisEntry(
            index=int(e.get("index", i)),
            kind=str(e.get("kind", "")),
            display_name=str(e.get("display_name", e.get("kind", ""))),
            metrics=dict(e.get("metrics", {})),
            warnings=[],
        )
        for i, e in enumerate(entries)
    ]
    return _build_chain_summary(rebuilt, chain_label=chain_label)


def _build_chain_summary(
    entries: list[EffectAnalysisEntry],
    *,
    chain_label: str | None = None,
) -> ChainSummary | None:
    """Aggregate per-effect metrics into chain-level diagnostics.

    Returns None for chains of fewer than two effects (no stacking to
    measure). Sums positive THD, centroid-lift, and high-band-lift deltas;
    sums negative peak + crest deltas; sums compressor + limiter GR.
    Emits leveled warnings when cumulative totals cross musical-failure
    thresholds even though no single stage may have tripped individually.
    """
    if len(entries) < 2:
        return None

    total_thd_growth_pct = 0.0  # kept as reference metric; no longer drives warnings
    total_imd_growth_factor = 0.0
    total_centroid_lift_hz = 0.0
    total_high_band_lift_db = 0.0
    total_peak_shave_db = 0.0
    total_crest_loss_db = 0.0
    total_compressor_gr_db = 0.0
    total_limiter_gr_db = 0.0
    total_clipper_shave_db = 0.0
    nonlinear_stage_count = 0
    kinds: list[str] = []
    # Reference the *original* chain input activity (first stage's input),
    # not each stage's input, since we care whether the source material was
    # sparse overall — not whether an intermediate stage happened to see a
    # quiet moment.
    chain_input_active_fraction = float(
        entries[0].metrics.get("input_active_window_fraction", 1.0)
    )

    for entry in entries:
        m = entry.metrics
        kinds.append(entry.kind)

        thd_delta = float(m.get("thd_delta_pct", 0.0))
        if thd_delta > 0.0:
            total_thd_growth_pct += thd_delta

        # Track cumulative IMD growth for chain_over_saturated, and count
        # nonlinear stages for the chain_papery / perceptual_brightness_lift
        # gate.  A stage counts as "nonlinear" when its effect kind can
        # physically produce harmonic buildup (see _NONLINEAR_EFFECT_KINDS)
        # *and* its measured IMD growth is substantive (>= 20%).
        #
        # The kind guard matters: reverb and long delay tails routinely
        # shift the two-tone IMD ratio measurement by more than 20% just by
        # redistributing spectral balance in the wet signal, which is a
        # measurement artifact rather than real nonlinearity.  Without the
        # kind guard, pure-linear send buses trip the brightness warnings
        # as false positives.
        imd_input = float(m.get("imd_ratio_input", 0.0))
        imd_output = float(m.get("imd_ratio_output", 0.0))
        stage_imd_growth = (imd_output - imd_input) / max(imd_input, 0.1)
        if (
            str(m.get("imd_detection", "single_tone")) == "two_tone"
            and stage_imd_growth > 0.0
        ):
            total_imd_growth_factor += stage_imd_growth
            if stage_imd_growth >= 0.2 and entry.kind in _NONLINEAR_EFFECT_KINDS:
                nonlinear_stage_count += 1

        centroid_delta = float(m.get("spectral_centroid_delta_hz", 0.0))
        if centroid_delta > 0.0:
            total_centroid_lift_hz += centroid_delta

        high_band_delta = float(m.get("high_band_delta_db", 0.0))
        if high_band_delta > 0.0:
            total_high_band_lift_db += high_band_delta

        peak_delta = float(m.get("peak_delta_db", 0.0))
        if peak_delta < 0.0:
            total_peak_shave_db += -peak_delta

        crest_delta = float(m.get("crest_factor_delta_db", 0.0))
        if crest_delta < 0.0:
            total_crest_loss_db += -crest_delta

        if entry.kind == "compressor":
            total_compressor_gr_db += float(m.get("avg_gain_reduction_db", 0.0))
        elif entry.kind == "limiter":
            # max_gain_reduction_db is negative when active; track as positive
            total_limiter_gr_db += -float(m.get("max_gain_reduction_db", 0.0))
        elif entry.kind == "clipper":
            total_clipper_shave_db += float(m.get("shaved_db", 0.0))

    metrics: dict[str, float | int | str] = {
        "stage_count": len(entries),
        "kinds": ",".join(kinds),
        # Reference metric retained for historical comparison. IMD-ratio
        # growth is the reliable signal; THD shifts on harmonically-rich
        # content without any nonlinearity actually being introduced.
        "total_thd_growth_pct": round(total_thd_growth_pct, 2),
        "total_imd_growth_factor": round(total_imd_growth_factor, 3),
        "total_centroid_lift_hz": round(total_centroid_lift_hz, 1),
        "total_high_band_lift_db": round(total_high_band_lift_db, 2),
        "total_peak_shave_db": round(total_peak_shave_db, 2),
        "total_crest_loss_db": round(total_crest_loss_db, 2),
        "total_compressor_gr_db": round(total_compressor_gr_db, 2),
        "total_limiter_gr_db": round(total_limiter_gr_db, 2),
        "total_clipper_shave_db": round(total_clipper_shave_db, 2),
        "nonlinear_stage_count": nonlinear_stage_count,
        "chain_input_active_fraction": round(chain_input_active_fraction, 4),
    }

    warnings: list[EffectAnalysisWarning] = []
    label_prefix = f"{chain_label}: " if chain_label else ""

    # chain_over_saturated fires on cumulative IMD growth, not THD.
    # Thresholds: >= 3.0 cumulative growth factor = severe (e.g. two stages
    # each doubling IMD), >= 1.0 = warning (a single stage adding meaningful
    # nonlinearity, or two moderate ones).
    if total_imd_growth_factor >= 3.0:
        warnings.append(
            _build_effect_warning(
                severity="severe",
                code="chain_over_saturated",
                message=f"{label_prefix}cumulative distortion is very high across the chain",
                total_imd_growth_factor=round(total_imd_growth_factor, 3),
                total_thd_growth_pct=round(total_thd_growth_pct, 2),
                nonlinear_stage_count=nonlinear_stage_count,
            )
        )
    elif total_imd_growth_factor >= 1.0:
        warnings.append(
            _build_effect_warning(
                severity="warning",
                code="chain_over_saturated",
                message=f"{label_prefix}cumulative distortion is elevated across the chain",
                total_imd_growth_factor=round(total_imd_growth_factor, 3),
                total_thd_growth_pct=round(total_thd_growth_pct, 2),
                nonlinear_stage_count=nonlinear_stage_count,
            )
        )

    # Papery-kick detector — the failure we actually saw in va_trance.
    # Triggers when high-band energy is being lifted cumulatively by the
    # chain, indicating stacked saturation + clipping pushing brittle
    # upper-midrange harmonics.
    #
    # Gate: only fire when the chain contains at least one nonlinear stage.
    # A pure linear/time-based chain (eq + delay, reverb, chorus) can sum
    # high-band deltas stage-over-stage without producing brittleness —
    # it just replicates existing content as wet signal.
    chain_has_nonlinear_stage = nonlinear_stage_count > 0 or any(
        kind in _NONLINEAR_EFFECT_KINDS for kind in kinds
    )
    if chain_has_nonlinear_stage and total_high_band_lift_db >= 8.0:
        severity = _cap_chain_brightness_severity(
            "severe", input_active_fraction=chain_input_active_fraction
        )
        warnings.append(
            _build_effect_warning(
                severity=severity,
                code="chain_papery",
                message=f"{label_prefix}chain is adding brittle high-frequency content (>8 dB of 2-8 kHz lift)",
                total_high_band_lift_db=round(total_high_band_lift_db, 2),
                total_centroid_lift_hz=round(total_centroid_lift_hz, 1),
                chain_input_active_fraction=round(chain_input_active_fraction, 4),
            )
        )
    elif chain_has_nonlinear_stage and total_high_band_lift_db >= 4.0:
        warnings.append(
            _build_effect_warning(
                severity="warning",
                code="chain_papery",
                message=f"{label_prefix}chain may be adding papery high-frequency character",
                total_high_band_lift_db=round(total_high_band_lift_db, 2),
                total_centroid_lift_hz=round(total_centroid_lift_hz, 1),
                chain_input_active_fraction=round(chain_input_active_fraction, 4),
            )
        )

    if total_centroid_lift_hz >= 1200.0:
        severity = _cap_chain_brightness_severity(
            "severe", input_active_fraction=chain_input_active_fraction
        )
        warnings.append(
            _build_effect_warning(
                severity=severity,
                code="chain_brightness_creep",
                message=f"{label_prefix}cumulative spectral centroid lift is very high",
                total_centroid_lift_hz=round(total_centroid_lift_hz, 1),
                chain_input_active_fraction=round(chain_input_active_fraction, 4),
            )
        )
    elif total_centroid_lift_hz >= 500.0:
        warnings.append(
            _build_effect_warning(
                severity="warning",
                code="chain_brightness_creep",
                message=f"{label_prefix}cumulative spectral centroid is drifting bright",
                total_centroid_lift_hz=round(total_centroid_lift_hz, 1),
                chain_input_active_fraction=round(chain_input_active_fraction, 4),
            )
        )

    # Thresholds reflect poly-knee clipper + comp glue behavior (April 2026).
    # A finished drum bus with comp (3-6 dB GR) + poly-knee clipper (1-3 dB
    # shave) routinely lands in the 3-4.5 dB cumulative crest-loss band
    # without sounding flattened — that's what "glued" means.  Warning
    # bumped 3->4.5 dB and severe 6->7 dB so the signal fires on real
    # transient-killing chains, not on healthy bus glue.
    if total_crest_loss_db >= 7.0:
        warnings.append(
            _build_effect_warning(
                severity="severe",
                code="chain_transient_flattening",
                message=f"{label_prefix}chain is flattening transients substantially",
                total_crest_loss_db=round(total_crest_loss_db, 2),
            )
        )
    elif total_crest_loss_db >= 4.5:
        warnings.append(
            _build_effect_warning(
                severity="warning",
                code="chain_transient_flattening",
                message=f"{label_prefix}chain is flattening transients",
                total_crest_loss_db=round(total_crest_loss_db, 2),
            )
        )

    total_gr = total_compressor_gr_db + total_limiter_gr_db
    if total_gr >= 14.0:
        warnings.append(
            _build_effect_warning(
                severity="severe",
                code="chain_over_compressed",
                message=f"{label_prefix}cumulative gain reduction across comp+limit stages is very high",
                total_compressor_gr_db=round(total_compressor_gr_db, 2),
                total_limiter_gr_db=round(total_limiter_gr_db, 2),
            )
        )
    elif total_gr >= 8.0:
        warnings.append(
            _build_effect_warning(
                severity="warning",
                code="chain_over_compressed",
                message=f"{label_prefix}cumulative gain reduction across comp+limit stages is elevated",
                total_compressor_gr_db=round(total_compressor_gr_db, 2),
                total_limiter_gr_db=round(total_limiter_gr_db, 2),
            )
        )

    # A-weighted perceptual brightness lift across the chain.
    # Gated on nonlinear-stage presence for the same reason as chain_papery
    # above — linear time-based effects (delay, reverb, eq, chorus) can
    # accumulate a_weighted_high_band_delta without harmonic buildup.
    total_a_weighted_lift_db = 0.0
    for entry in entries:
        a_delta = float(entry.metrics.get("a_weighted_high_band_delta_db", 0.0))
        if a_delta > 0.0:
            total_a_weighted_lift_db += a_delta
    metrics["total_a_weighted_high_band_lift_db"] = round(total_a_weighted_lift_db, 2)
    if chain_has_nonlinear_stage and total_a_weighted_lift_db >= 8.0:
        severity = _cap_chain_brightness_severity(
            "severe", input_active_fraction=chain_input_active_fraction
        )
        warnings.append(
            _build_effect_warning(
                severity=severity,
                code="perceptual_brightness_lift",
                message=f"{label_prefix}A-weighted high-band energy is rising sharply across the chain",
                total_a_weighted_high_band_lift_db=round(total_a_weighted_lift_db, 2),
                chain_input_active_fraction=round(chain_input_active_fraction, 4),
            )
        )
    elif chain_has_nonlinear_stage and total_a_weighted_lift_db >= 4.0:
        warnings.append(
            _build_effect_warning(
                severity="warning",
                code="perceptual_brightness_lift",
                message=f"{label_prefix}A-weighted high-band energy is drifting upward across the chain",
                total_a_weighted_high_band_lift_db=round(total_a_weighted_lift_db, 2),
                chain_input_active_fraction=round(chain_input_active_fraction, 4),
            )
        )

    return ChainSummary(metrics=metrics, warnings=warnings)


def build_voice_stem_delta(
    *,
    voice_name: str,
    dry_signal: np.ndarray,
    wet_signal: np.ndarray,
    sample_rate: int,
    percussive: bool,
) -> dict[str, Any]:
    """Compute dry->wet delta metrics for a single voice-chain stem.

    Mirrors the per-effect IO deltas but at the voice-chain level: synth
    output pre-effects (*dry_signal*) vs post-voice-effects (*wet_signal*).
    Catches failure modes invisible to per-effect or global-mix diagnostics:
    "how did my kick change between voice synth and post-FX?"

    When *percussive* is True, onset detection runs on the dry signal and
    transient preservation is reported per hit. Emits new warning codes
    ``percussive_transient_killed``, ``percussive_crunch_character``, and
    ``transient_brightness_decoupled``.
    """
    entry = _build_effect_analysis_entry(
        index=0,
        kind="voice_stem",
        display_name=f"voice_stem:{voice_name}",
        input_signal=dry_signal,
        output_signal=wet_signal,
        sample_rate=sample_rate,
        percussive=percussive,
    )
    metrics: dict[str, float | int | str] = dict(entry.metrics)
    warnings: list[EffectAnalysisWarning] = []
    if percussive:
        warnings.extend(_build_voice_stem_percussive_warnings(metrics))
    a_weighted_delta = float(metrics.get("a_weighted_high_band_delta_db", 0.0))
    if a_weighted_delta >= 8.0:
        warnings.append(
            _build_effect_warning(
                severity="severe",
                code="perceptual_brightness_lift",
                message=(
                    f"voice_stem:{voice_name} A-weighted high-band energy "
                    "rose sharply through the voice chain"
                ),
                a_weighted_high_band_delta_db=round(a_weighted_delta, 2),
            )
        )
    elif a_weighted_delta >= 4.0:
        warnings.append(
            _build_effect_warning(
                severity="warning",
                code="perceptual_brightness_lift",
                message=(
                    f"voice_stem:{voice_name} A-weighted high-band energy "
                    "drifted upward through the voice chain"
                ),
                a_weighted_high_band_delta_db=round(a_weighted_delta, 2),
            )
        )
    return {
        "voice_name": voice_name,
        "percussive": bool(percussive),
        "metrics": metrics,
        "warnings": [w.to_dict() for w in warnings],
    }


def _build_voice_stem_percussive_warnings(
    metrics: dict[str, float | int | str],
) -> list[EffectAnalysisWarning]:
    """Warning codes specific to percussive voice-chain deltas.

    Codes
    -----
    ``percussive_transient_killed``
        Fires when the voice chain's ``transient_kill_db`` crosses
        warning/severe thresholds. Severe at ~-30% kill (-3.1 dB), warning
        at ~-15% (-1.4 dB).
    ``percussive_crunch_character``
        Fires when the voice chain exhibits BOTH elevated IMD ratio
        (>0.4) AND substantial A-weighted brightness lift (>2 dB). The
        "crunchy kick" detector.
    ``transient_brightness_decoupled``
        Fires when A-weighted brightness lift is high (>3 dB) but
        transients are killed (input p50 - output p50 ratio loss). The
        "papery without punch" detector — brightness is the surface
        symptom, flattened transients are the root cause.
    """
    warnings: list[EffectAnalysisWarning] = []
    transient_kill_db = float(metrics.get("transient_kill_db", 0.0))
    imd_ratio_output = float(metrics.get("imd_ratio_output", 0.0))
    a_weighted_delta = float(metrics.get("a_weighted_high_band_delta_db", 0.0))
    input_p50 = float(metrics.get("transient_peak_ratio_input_p50", 0.0))
    output_p50 = float(metrics.get("transient_peak_ratio_output_p50", 0.0))

    # percussive_transient_killed: severe when >30% kill (-3.1 dB),
    # warning when >15% (-1.4 dB).
    if transient_kill_db <= -3.1:
        warnings.append(
            _build_effect_warning(
                severity="severe",
                code="percussive_transient_killed",
                message="voice chain crushes the per-hit transient by more than 30%",
                transient_kill_db=round(transient_kill_db, 2),
                transient_peak_ratio_input_p50=round(input_p50, 4),
                transient_peak_ratio_output_p50=round(output_p50, 4),
            )
        )
    elif transient_kill_db <= -1.4:
        warnings.append(
            _build_effect_warning(
                severity="warning",
                code="percussive_transient_killed",
                message="voice chain is softening per-hit transients by >15%",
                transient_kill_db=round(transient_kill_db, 2),
                transient_peak_ratio_input_p50=round(input_p50, 4),
                transient_peak_ratio_output_p50=round(output_p50, 4),
            )
        )

    # percussive_crunch_character: IMD + brightness combo (the THD-invisible
    # crunch flavor).
    if imd_ratio_output >= 0.4 and a_weighted_delta >= 2.0:
        warnings.append(
            _build_effect_warning(
                severity="warning",
                code="percussive_crunch_character",
                message=(
                    "voice chain is adding crunchy character "
                    "(intermodulation products dominate over harmonic distortion)"
                ),
                imd_ratio_output=round(imd_ratio_output, 4),
                a_weighted_high_band_delta_db=round(a_weighted_delta, 2),
            )
        )

    # transient_brightness_decoupled: brightness without punch.
    # Only fires when per-hit data is available (hit_count > 0).
    hit_count = int(metrics.get("hit_count", 0))
    if hit_count > 0 and a_weighted_delta >= 3.0 and transient_kill_db <= -1.0:
        warnings.append(
            _build_effect_warning(
                severity="warning",
                code="transient_brightness_decoupled",
                message=(
                    "voice chain adds brightness without preserving punch "
                    "(papery symptom, flattened transients are the root cause)"
                ),
                a_weighted_high_band_delta_db=round(a_weighted_delta, 2),
                transient_kill_db=round(transient_kill_db, 2),
            )
        )

    return warnings


@numba.njit(cache=True)
def _compressor_gain_db_vec(
    level_db: np.ndarray,
    threshold_db: float,
    ratio: float,
    knee_db: float,
) -> np.ndarray:
    """Vectorized compressor gain curve (numba for knee math)."""
    n = level_db.shape[0]
    gain_db = np.zeros(n, dtype=np.float64)
    if knee_db <= 0.0:
        for i in range(n):
            if level_db[i] > threshold_db:
                compressed = threshold_db + ((level_db[i] - threshold_db) / ratio)
                gain_db[i] = compressed - level_db[i]
    else:
        lower_knee = threshold_db - (knee_db / 2.0)
        upper_knee = threshold_db + (knee_db / 2.0)
        inv_ratio_minus_one = (1.0 / ratio) - 1.0
        for i in range(n):
            lev = level_db[i]
            if lev <= lower_knee:
                pass
            elif lev >= upper_knee:
                compressed = threshold_db + ((lev - threshold_db) / ratio)
                gain_db[i] = compressed - lev
            else:
                knee_progress = lev - lower_knee
                gain_db[i] = inv_ratio_minus_one * (knee_progress**2) / (2.0 * knee_db)
    return gain_db


@numba.njit(cache=True)
def _compressor_smooth_gain_loop(
    target_gain_db: np.ndarray,
    attack_coeff: float,
    release_stage1_coeff: float,
    release_stage2_coeff: float,
) -> np.ndarray:
    """Smooth the compressor gain curve with attack/release ballistics."""
    n = target_gain_db.shape[0]
    smoothed = np.empty(n, dtype=np.float64)
    state = 0.0
    for i in range(n):
        tgt = target_gain_db[i]
        if tgt < state:
            coeff = attack_coeff
        else:
            abs_state = -state if state < 0.0 else state
            abs_tgt = -tgt if tgt < 0.0 else tgt
            denom = abs_tgt if abs_tgt > 1.0 else 1.0
            progress = abs_state / denom
            if progress < 0.0:
                progress = 0.0
            elif progress > 1.0:
                progress = 1.0
            coeff = (progress * release_stage2_coeff) + (
                (1.0 - progress) * release_stage1_coeff
            )
        state = (coeff * state) + ((1.0 - coeff) * tgt)
        smoothed[i] = state
    return smoothed


@numba.njit(cache=True)
def _compressor_detector_smooth(
    detector_trace: np.ndarray,
    attack_coeff: float,
    release_coeff: float,
    is_rms: bool,
) -> np.ndarray:
    """One-pole attack/release smoothing of the detector signal."""
    n = detector_trace.shape[0]
    smoothed = np.empty(n, dtype=np.float64)
    state = 0.0
    for i in range(n):
        inp = detector_trace[i]
        target = inp * inp if is_rms else inp
        coeff = attack_coeff if target > state else release_coeff
        state = (coeff * state) + ((1.0 - coeff) * target)
        if is_rms:
            val = state**0.5
            smoothed[i] = val if val > 1e-12 else 1e-12
        else:
            smoothed[i] = state if state > 1e-12 else 1e-12
    return smoothed


@numba.njit(cache=True)
def _compressor_feedback_loop(
    delayed_input_2d: np.ndarray,
    is_rms: bool,
    attack_coeff: float,
    detector_release_coeff: float,
    threshold_db: float,
    ratio: float,
    knee_db: float,
    release_stage1_coeff: float,
    release_stage2_coeff: float,
    makeup_gain: float,
    mix: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample feedback compressor loop (inherently sequential).

    delayed_input_2d is always (channels, samples) -- mono is (1, samples).
    """
    n_channels = delayed_input_2d.shape[0]
    aligned_n = delayed_input_2d.shape[1]
    gain_reduction_trace = np.zeros(aligned_n, dtype=np.float64)
    output = np.zeros_like(delayed_input_2d)

    detector_state = 0.0
    smoothed_gain_db = 0.0
    output_detector_state = 0.0

    lower_knee = threshold_db - (knee_db / 2.0)
    upper_knee = threshold_db + (knee_db / 2.0)
    inv_ratio_minus_one = (1.0 / ratio) - 1.0

    for i in range(aligned_n):
        detector_input = output_detector_state

        target_level = detector_input * detector_input if is_rms else detector_input
        det_coeff = (
            attack_coeff if target_level > detector_state else detector_release_coeff
        )
        detector_state = (det_coeff * detector_state) + (
            (1.0 - det_coeff) * target_level
        )

        if is_rms:
            level_amp = detector_state**0.5
            if level_amp < 1e-12:
                level_amp = 1e-12
        else:
            level_amp = detector_state if detector_state > 1e-12 else 1e-12
        level_db = 20.0 * np.log10(level_amp)

        if knee_db <= 0.0:
            if level_db <= threshold_db:
                target_gain_db = 0.0
            else:
                target_gain_db = (
                    threshold_db + ((level_db - threshold_db) / ratio)
                ) - level_db
        elif level_db <= lower_knee:
            target_gain_db = 0.0
        elif level_db >= upper_knee:
            target_gain_db = (
                threshold_db + ((level_db - threshold_db) / ratio)
            ) - level_db
        else:
            knee_progress = level_db - lower_knee
            target_gain_db = inv_ratio_minus_one * (knee_progress**2) / (2.0 * knee_db)

        if target_gain_db < smoothed_gain_db:
            gain_coeff = attack_coeff
        else:
            abs_sg = -smoothed_gain_db if smoothed_gain_db < 0.0 else smoothed_gain_db
            abs_tg = -target_gain_db if target_gain_db < 0.0 else target_gain_db
            denom = abs_tg if abs_tg > 1.0 else 1.0
            progress = abs_sg / denom
            if progress < 0.0:
                progress = 0.0
            elif progress > 1.0:
                progress = 1.0
            gain_coeff = (progress * release_stage2_coeff) + (
                (1.0 - progress) * release_stage1_coeff
            )
        smoothed_gain_db = (gain_coeff * smoothed_gain_db) + (
            (1.0 - gain_coeff) * target_gain_db
        )
        gain_reduction_trace[i] = max(0.0, -smoothed_gain_db)

        gain = 10.0 ** (smoothed_gain_db / 20.0) * makeup_gain
        max_abs = 0.0
        for ch in range(n_channels):
            wet_val = delayed_input_2d[ch, i] * gain
            output[ch, i] = ((1.0 - mix) * delayed_input_2d[ch, i]) + (mix * wet_val)
            abs_val = -output[ch, i] if output[ch, i] < 0.0 else output[ch, i]
            if abs_val > max_abs:
                max_abs = abs_val
        output_detector_state = max_abs

    return output, gain_reduction_trace


def apply_compressor(
    signal: np.ndarray,
    *,
    threshold_db: float = -20.0,
    ratio: float = 3.0,
    attack_ms: float = 15.0,
    release_ms: float = 180.0,
    release_tail_ms: float | None = None,
    knee_db: float = 6.0,
    makeup_gain_db: float = 0.0,
    mix: float = 1.0,
    topology: str = "feedforward",
    detector_mode: str = "rms",
    detector_bands: list[dict[str, Any]] | None = None,
    sidechain_signal: np.ndarray | None = None,
    lookahead_ms: float = 0.0,
    sample_rate: int = SAMPLE_RATE,
    target_avg_gr_db: float | None = None,
    return_analysis: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int | str]]:
    """Apply a native stereo-linked compressor with optional detector EQ.

    Auto-calibration (piece-aware threshold targeting)
    --------------------------------------------------
    When ``target_avg_gr_db`` is set (default ``None``), the compressor
    binary-searches ``threshold_db`` so that the *average gain reduction on
    active samples* matches the target. "Active" means the 50 ms RMS detector
    envelope at that sample sits within 40 dB of the detector envelope's p99
    peak — i.e., the parts of the piece the compressor is doing real work on.

    This matches musician intuition: "4 dB of compression" means "you'll
    measure 4 dB avg GR on the parts the compressor is working." Unlike a
    piece-level mean GR (which silent gaps on sparse drum content drag down)
    or a single-point closed-form solve (which picks a threshold and hopes
    the ballistics deliver the intended feel), the active-region metric
    reports the compressor's *character*.

    The solver invokes the same feedforward GR kernel the main path uses, so
    it's bit-consistent with the runtime kernel (modulo the feedforward vs.
    feedback topology difference — see Notes below). The binary search runs
    up to 10 iterations over candidate thresholds; each iteration costs one
    envelope-kernel pass on the detector trace, which is cheap.

    This is the sibling of :func:`apply_clipper`'s ``max_shave_db`` pattern:
    both make a compressor/clipper *piece-aware* by letting the user set
    target behavior in dB rather than tuning threshold for every piece.

    When both ``target_avg_gr_db`` and a non-default ``threshold_db`` are
    set, a warning is logged and ``target_avg_gr_db`` wins.

    Edge cases
    ~~~~~~~~~~

    - **Silent / DC-only input**: no active samples exist, so the solver
      logs a WARNING and falls back to the caller's ``threshold_db`` (or
      the ``-20.0`` default). No crash, no NaN.
    - **Solver doesn't converge**: if |measured - target| is still more
      than 0.75 dB after 10 iterations, a WARNING is logged and the best
      candidate found is applied anyway.

    Notes
    -----
    - The solver uses feedforward simulation even when ``topology="feedback"``
      is configured. Feedback-loop-coupled GR can differ from the feedforward
      approximation by ~0.5-1.5 dB on loud material; the design intent is
      "this is approximately how much glue we want," not bit-exact GR.

    Parameters
    ----------
    target_avg_gr_db : float | None
        Desired average gain reduction in dB on active samples (detector
        envelope within 40 dB of its p99 peak). Default ``None`` — no
        auto-calibration; ``threshold_db`` is used directly.
    """
    if ratio < 1.0:
        raise ValueError("ratio must be at least 1")
    if attack_ms <= 0.0:
        raise ValueError("attack_ms must be positive")
    if knee_db < 0.0:
        raise ValueError("knee_db must be non-negative")
    if not 0.0 <= mix <= 1.0:
        raise ValueError("mix must be between 0 and 1")

    normalized_topology = topology.lower()
    if normalized_topology not in {"feedforward", "feedback"}:
        raise ValueError("topology must be 'feedforward' or 'feedback'")

    normalized_detector_mode = detector_mode.lower()
    if normalized_detector_mode not in {"peak", "rms"}:
        raise ValueError("detector_mode must be 'peak' or 'rms'")

    lookahead_samples = _lookahead_samples(lookahead_ms, sample_rate)
    resolved_release_stage1_ms, resolved_release_stage2_ms = (
        _resolve_compressor_release_times(
            release_ms=release_ms,
            release_tail_ms=release_tail_ms,
        )
    )

    input_signal = _coerce_signal_layout(signal)
    detector_source = (
        _coerce_signal_layout(sidechain_signal)
        if sidechain_signal is not None
        else input_signal
    )
    if detector_source.shape[-1] != input_signal.shape[-1]:
        raise ValueError("sidechain_signal must match the program signal length")
    if detector_bands is not None:
        detector_source = apply_eq(
            detector_source,
            bands=detector_bands,
            sample_rate=sample_rate,
        )

    attack_coeff = _time_constant_to_coeff(attack_ms, sample_rate)
    detector_release_coeff = _time_constant_to_coeff(
        resolved_release_stage2_ms, sample_rate
    )
    release_stage1_coeff = _time_constant_to_coeff(
        resolved_release_stage1_ms, sample_rate
    )
    release_stage2_coeff = _time_constant_to_coeff(
        resolved_release_stage2_ms, sample_rate
    )
    makeup_gain = db_to_amp(makeup_gain_db)
    is_rms = normalized_detector_mode == "rms"

    # Track whether user explicitly overrode the default threshold_db; this
    # lets us warn when both target_avg_gr_db and a non-default threshold_db
    # were set.  Mirrors the clipper's sentinel pattern.
    threshold_db_user_set = threshold_db != -20.0

    auto_calibrated = target_avg_gr_db is not None
    calibrated_measured_avg_gr_db: float | None = None
    solver_iterations: int | None = None
    if auto_calibrated:
        assert target_avg_gr_db is not None  # for type narrowing
        if threshold_db_user_set:
            logger.warning(
                "Compressor: both threshold_db and target_avg_gr_db set — "
                "target_avg_gr_db wins; threshold_db ignored."
            )

        detector_trace_for_solver = _linked_detector_signal(detector_source)
        window_ms = 50.0
        window_samples = max(1, int(round(window_ms * 1e-3 * sample_rate)))
        windowed_env = _compute_detector_rms_envelope(
            detector_source=detector_source,
            sample_rate=sample_rate,
            window_ms=window_ms,
        )
        if windowed_env.size == 0:
            envelope_p99_linear = 0.0
        else:
            envelope_p99_linear = float(np.percentile(windowed_env, 99.0))
        # Silence floor: an envelope peak below -120 dBFS is effectively
        # silence (DC, zero input, denormals). No useful calibration target.
        silence_floor_linear = db_to_amp(-120.0)
        if envelope_p99_linear < silence_floor_linear:
            active_mask = np.zeros(detector_trace_for_solver.shape[0], dtype=bool)
            envelope_p99_dbfs = -120.0
        else:
            envelope_p99_dbfs = 20.0 * np.log10(envelope_p99_linear)
            active_threshold_linear = envelope_p99_linear * db_to_amp(-40.0)
            env_per_sample = _expand_windowed_envelope_to_samples(
                windowed_env,
                sample_count=detector_trace_for_solver.shape[0],
                window_samples=window_samples,
            )
            active_mask = env_per_sample >= active_threshold_linear

        if not np.any(active_mask):
            logger.warning(
                "Compressor: target_avg_gr_db requested but input has no "
                "active content (envelope below calibration floor everywhere). "
                f"Falling back to static threshold_db={threshold_db:.2f} dBFS."
            )
        else:
            env_p50_linear = float(np.percentile(windowed_env, 50.0))
            env_p50_db = 20.0 * np.log10(max(env_p50_linear, 1e-12))
            search_low_db = env_p50_db - 30.0
            search_high_db = envelope_p99_dbfs
            (
                threshold_db,
                calibrated_measured_avg_gr_db,
                solver_iterations,
            ) = _solve_compressor_threshold_for_target_avg_gr(
                detector_trace=detector_trace_for_solver,
                active_mask=active_mask,
                attack_coeff=attack_coeff,
                release_coeff=detector_release_coeff,
                ratio=ratio,
                knee_db=knee_db,
                is_rms=is_rms,
                release_stage1_coeff=release_stage1_coeff,
                release_stage2_coeff=release_stage2_coeff,
                target_avg_gr_db=float(target_avg_gr_db),
                search_low_db=search_low_db,
                search_high_db=search_high_db,
            )
            residual = abs(calibrated_measured_avg_gr_db - float(target_avg_gr_db))
            if residual > 0.75:
                logger.warning(
                    "Compressor: auto-cal did not converge — "
                    f"target avg GR on active {float(target_avg_gr_db):.2f} dB, "
                    f"measured {calibrated_measured_avg_gr_db:.2f} dB, "
                    f"{solver_iterations} iter, threshold={threshold_db:.2f} dBFS"
                )
            logger.info(
                f"Compressor: auto-calibrated threshold to {threshold_db:.2f} "
                f"dBFS (target avg GR on active "
                f"{float(target_avg_gr_db):.2f} dB, "
                f"measured {calibrated_measured_avg_gr_db:.2f} dB, "
                f"{solver_iterations} iter)"
            )

    detector_trace = _linked_detector_signal(detector_source)
    sample_count = input_signal.shape[-1]
    aligned_sample_count = sample_count + lookahead_samples

    if input_signal.ndim == 1:
        delayed_input_signal = np.pad(
            input_signal,
            (lookahead_samples, 0),
            mode="constant",
        )
    else:
        delayed_input_signal = np.pad(
            input_signal,
            ((0, 0), (lookahead_samples, 0)),
            mode="constant",
        )

    was_mono = input_signal.ndim == 1

    if normalized_topology == "feedback":
        delayed_2d = (
            delayed_input_signal[np.newaxis, :] if was_mono else delayed_input_signal
        )
        output_2d, gain_reduction_trace_db = _compressor_feedback_loop(
            delayed_2d,
            is_rms,
            attack_coeff,
            detector_release_coeff,
            threshold_db,
            ratio,
            knee_db,
            release_stage1_coeff,
            release_stage2_coeff,
            makeup_gain,
            mix,
        )
        output_signal = output_2d[0] if was_mono else output_2d
    else:
        # Vectorized feedforward path.  The old per-sample loop used
        # min(sample_index, sample_count - 1) to index the detector trace,
        # which means indices beyond sample_count clamp to the last value.
        aligned_detector = np.pad(detector_trace, (0, lookahead_samples), mode="edge")[
            :aligned_sample_count
        ]

        smoothed_level = _compressor_detector_smooth(
            aligned_detector, attack_coeff, detector_release_coeff, is_rms
        )
        level_db = 20.0 * np.log10(smoothed_level)

        target_gain_db = _compressor_gain_db_vec(level_db, threshold_db, ratio, knee_db)

        smoothed_gain_db = _compressor_smooth_gain_loop(
            target_gain_db, attack_coeff, release_stage1_coeff, release_stage2_coeff
        )
        gain_reduction_trace_db = np.maximum(0.0, -smoothed_gain_db)

        gain_linear = 10.0 ** (smoothed_gain_db / 20.0) * makeup_gain
        if was_mono:
            wet_signal = delayed_input_signal * gain_linear
            output_signal = ((1.0 - mix) * delayed_input_signal) + (mix * wet_signal)
        else:
            wet_signal = delayed_input_signal * gain_linear[np.newaxis, :]
            output_signal = ((1.0 - mix) * delayed_input_signal) + (mix * wet_signal)

    if was_mono:
        processed_signal = np.asarray(
            output_signal[lookahead_samples : lookahead_samples + sample_count],
            dtype=np.float64,
        )
    else:
        processed_signal = np.asarray(
            output_signal[:, lookahead_samples : lookahead_samples + sample_count],
            dtype=np.float64,
        )
    aligned_gain_reduction_trace_db = gain_reduction_trace_db[
        lookahead_samples : lookahead_samples + sample_count
    ]
    if not return_analysis:
        return processed_signal

    active_mask = aligned_gain_reduction_trace_db >= 1.0
    below_1db_mask = aligned_gain_reduction_trace_db < 1.0
    analysis: dict[str, float | int | str] = {
        "avg_gain_reduction_db": round(
            float(np.mean(aligned_gain_reduction_trace_db)),
            2,
        ),
        "max_gain_reduction_db": round(
            float(np.max(aligned_gain_reduction_trace_db)),
            2,
        ),
        "p95_gain_reduction_db": round(
            float(np.percentile(aligned_gain_reduction_trace_db, 95.0)),
            2,
        ),
        "active_gain_reduction_fraction": round(
            float(
                np.count_nonzero(active_mask)
                / max(aligned_gain_reduction_trace_db.size, 1)
            ),
            4,
        ),
        "avg_gain_reduction_when_active_db": round(
            float(np.mean(aligned_gain_reduction_trace_db[active_mask]))
            if np.any(active_mask)
            else 0.0,
            2,
        ),
        "below_1db_fraction": round(
            float(
                np.count_nonzero(below_1db_mask)
                / max(aligned_gain_reduction_trace_db.size, 1)
            ),
            4,
        ),
        "longest_run_above_1db_seconds": round(
            _seconds_for_mask(active_mask, sample_rate=sample_rate),
            2,
        ),
    }
    if auto_calibrated:
        assert target_avg_gr_db is not None
        analysis["calibrated_threshold_db"] = round(threshold_db, 2)
        analysis["target_avg_gr_db"] = round(float(target_avg_gr_db), 2)
        if calibrated_measured_avg_gr_db is not None:
            analysis["measured_avg_gr_db"] = round(
                float(calibrated_measured_avg_gr_db), 2
            )
        if solver_iterations is not None:
            analysis["solver_iterations"] = int(solver_iterations)
    return processed_signal, analysis


def _is_stereo(signal: np.ndarray) -> bool:
    """Return True when the signal uses the repo's stereo layout."""
    return signal.ndim == 2 and signal.shape[0] == 2


def _ensure_stereo(signal: np.ndarray) -> np.ndarray:
    """Promote mono signal to stereo, preserving stereo input."""
    if _is_stereo(signal):
        return np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"Unsupported signal shape: {signal.shape}")
    mono_signal = np.asarray(signal, dtype=np.float64)
    return np.stack([mono_signal, mono_signal])


def _match_signal_length(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or trim mono/stereo signal to a target duration in samples."""
    normalized = _coerce_signal_layout(signal)
    current_length = normalized.shape[-1]
    if current_length == target_length:
        return normalized
    if current_length > target_length:
        if normalized.ndim == 1:
            return np.asarray(normalized[:target_length], dtype=np.float64)
        return np.asarray(normalized[:, :target_length], dtype=np.float64)

    pad_width = target_length - current_length
    if normalized.ndim == 1:
        return np.pad(normalized, (0, pad_width), mode="constant")
    return np.pad(normalized, ((0, 0), (0, pad_width)), mode="constant")


def _match_input_layout(processed: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return processed signal in the same channel layout as reference."""
    if _is_stereo(reference):
        return _ensure_stereo(processed)
    if processed.ndim == 1:
        return np.asarray(processed, dtype=np.float64)
    if _is_stereo(processed):
        return np.asarray(processed.mean(axis=0), dtype=np.float64)
    raise ValueError(f"Unsupported processed signal shape: {processed.shape}")


def _coerce_signal_layout(signal: np.ndarray) -> np.ndarray:
    """Normalize effect outputs into supported mono/stereo ndarray layouts."""
    normalized = np.asarray(signal, dtype=np.float64)
    if normalized.ndim == 1:
        return normalized
    if normalized.ndim != 2:
        raise ValueError(f"Unsupported signal shape: {normalized.shape}")
    if normalized.shape[0] == 2:
        return normalized
    if normalized.shape[1] == 2:
        return normalized.T
    raise ValueError(f"Unsupported signal shape: {normalized.shape}")


def _apply_per_channel(
    signal: np.ndarray,
    processor: Any,
) -> np.ndarray:
    """Apply a mono processor independently to each channel when needed."""
    if signal.ndim == 1:
        return _coerce_signal_layout(processor(np.asarray(signal, dtype=np.float64)))
    if not _is_stereo(signal):
        raise ValueError(f"Unsupported signal shape: {signal.shape}")
    processed_channels = [
        np.asarray(
            processor(np.asarray(signal[channel], dtype=np.float64)), dtype=np.float64
        )
        for channel in range(signal.shape[0])
    ]
    return np.stack(processed_channels)


def _fractional_delay(signal: np.ndarray, delay_samples: np.ndarray) -> np.ndarray:
    """Apply a time-varying fractional delay using linear interpolation."""
    sample_positions = np.arange(signal.shape[-1], dtype=np.float64)
    delayed_positions = np.clip(
        sample_positions - delay_samples, 0.0, signal.shape[-1] - 1.0
    )
    return np.interp(delayed_positions, sample_positions, signal)


def apply_pan(
    signal: np.ndarray,
    pan: float = 0.0,
) -> np.ndarray:
    """Apply equal-power panning, returning stereo output."""
    if not -1.0 <= pan <= 1.0:
        raise ValueError("pan must be between -1 and 1")

    stereo_signal = _ensure_stereo(signal)
    mono_reference = stereo_signal.mean(axis=0)
    pan_angle = (pan + 1.0) * (np.pi / 4.0)
    left_gain = np.cos(pan_angle)
    right_gain = np.sin(pan_angle)
    return np.stack([mono_reference * left_gain, mono_reference * right_gain]).astype(
        np.float64
    )


def apply_pan_automation(
    signal: np.ndarray,
    *,
    pan_curve: np.ndarray,
) -> np.ndarray:
    """Apply equal-power panning with a per-sample pan curve."""
    resolved_pan_curve = np.asarray(pan_curve, dtype=np.float64)
    if resolved_pan_curve.ndim != 1:
        raise ValueError("pan_curve must be one-dimensional")
    if np.any((resolved_pan_curve < -1.0) | (resolved_pan_curve > 1.0)):
        raise ValueError("pan values must be between -1 and 1")

    stereo_signal = _ensure_stereo(signal)
    if stereo_signal.shape[-1] != resolved_pan_curve.size:
        raise ValueError("pan_curve length must match the signal length")

    mono_reference = stereo_signal.mean(axis=0)
    pan_angles = (resolved_pan_curve + 1.0) * (np.pi / 4.0)
    left_gain = np.cos(pan_angles)
    right_gain = np.sin(pan_angles)
    return np.stack([mono_reference * left_gain, mono_reference * right_gain]).astype(
        np.float64
    )


_DEFAULT_BRICASTI_IR_DIR = (
    "/mnt/c/Music Production/Convolution Impulses"
    "/Samplicity - Bricasti IRs version 2023-10"
    "/Samplicity - Bricasti IRs version 2023-10, left-right files, 44.1 Khz"
)
BRICASTI_IR_DIR = Path(os.environ.get("BRICASTI_IR_DIR", _DEFAULT_BRICASTI_IR_DIR))


def normalize(signal: np.ndarray, peak: float = 0.85) -> np.ndarray:
    """Normalize mono or stereo signal. Stereo shape: (2, samples)."""
    max_val = np.max(np.abs(signal))
    return signal * peak / max_val if max_val > 0 else signal


def normalize_true_peak(
    signal: np.ndarray,
    *,
    target_peak_dbfs: float,
    oversample_factor: int = 4,
) -> np.ndarray:
    """Apply gain so the estimated true peak lands at or below the ceiling."""
    true_peak_amplitude = estimate_true_peak_amplitude(
        signal,
        oversample_factor=oversample_factor,
    )
    if true_peak_amplitude <= 0:
        return np.asarray(signal, dtype=np.float64)

    target_amplitude = db_to_amp(target_peak_dbfs)
    required_gain = target_amplitude / true_peak_amplitude
    return np.asarray(signal, dtype=np.float64) * required_gain


def _float_to_int16_pcm(
    signal: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Convert float audio in [-1, 1] to dithered int16 PCM."""
    normalized = np.asarray(signal, dtype=np.float64)
    quantization_step = 1.0 / 32768.0
    clipped = np.clip(normalized, -1.0, 1.0 - quantization_step)
    resolved_rng = rng if rng is not None else np.random.default_rng()
    dither = (
        resolved_rng.random(clipped.shape, dtype=np.float64)
        + resolved_rng.random(clipped.shape, dtype=np.float64)
        - 1.0
    ) * quantization_step
    return np.rint((clipped + dither) * 32767.0).astype(np.int16)


@numba.njit(cache=True)
def _gate_gain_smoothing_loop(
    gate_target: np.ndarray,
    attack_coeff: float,
    release_coeff: float,
    floor_lin: float,
) -> np.ndarray:
    n = gate_target.shape[0]
    gain = np.empty(n, dtype=np.float64)
    g = floor_lin
    for i in range(n):
        t = gate_target[i]
        if t > g:
            g += attack_coeff * (t - g)
        else:
            g += release_coeff * (t - g)
        gain[i] = g
    return gain


def apply_gate(
    signal: np.ndarray,
    *,
    threshold_db: float = -40.0,
    attack_ms: float = 0.5,
    hold_ms: float = 40.0,
    release_ms: float = 20.0,
    floor_db: float = -80.0,
    sample_rate: int = SAMPLE_RATE,
    return_analysis: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int | str]]:
    """Noise gate with attack/hold/release gain envelope.

    Args:
        threshold_db: Gate opens when signal exceeds this level.
        attack_ms: Time to ramp from floor to full gain once gate opens.
        hold_ms: Minimum time gate stays open after signal drops below threshold.
        release_ms: Time to ramp from full gain to floor once hold expires.
        floor_db: Attenuation when gate is fully closed. Default -80 dB ≈ silence.
    """
    if attack_ms <= 0:
        raise ValueError("attack_ms must be positive")
    if release_ms <= 0:
        raise ValueError("release_ms must be positive")
    if hold_ms < 0:
        raise ValueError("hold_ms must be non-negative")

    threshold_lin = 10.0 ** (threshold_db / 20.0)
    floor_lin = 10.0 ** (floor_db / 20.0)
    attack_samples = max(1, int(attack_ms * sample_rate / 1000.0))
    hold_samples = max(0, int(hold_ms * sample_rate / 1000.0))
    release_samples = max(1, int(release_ms * sample_rate / 1000.0))

    # One-pole IIR coefficients — exponential approach to target
    attack_coeff = 1.0 - np.exp(-1.0 / attack_samples)
    release_coeff = 1.0 - np.exp(-1.0 / release_samples)

    input_signal = np.asarray(signal, dtype=np.float64)

    # Stereo-linked detection: compute a single gain envelope from the louder of
    # L/R (max-abs-linked key signal) and apply the same gain to both channels.
    # This prevents stereo tearing where one channel gates while the other
    # doesn't.  For mono input the linked key is identical to the channel, so
    # behavior is bit-identical to per-channel processing.
    def _gate_gain_envelope(key: np.ndarray) -> np.ndarray:
        n = len(key)
        if n == 0:
            return np.zeros(0, dtype=np.float64)

        # 2 ms RMS envelope for smooth key signal (avoids flicker on zero crossings)
        detect_window = max(1, int(0.002 * sample_rate))
        kernel = np.ones(detect_window, dtype=np.float64) / detect_window
        smoothed_level = np.sqrt(np.convolve(key**2, kernel, mode="same"))

        # Gate open = level above threshold; hold extends open regions forward
        gate_open = (smoothed_level >= threshold_lin).astype(np.float64)
        if hold_samples > 0:
            hold_kernel = np.ones(hold_samples + 1, dtype=np.float64)
            gate_open = np.convolve(gate_open, hold_kernel, mode="full")[:n]
        gate_target = np.where(gate_open > 0, 1.0, floor_lin)

        return _gate_gain_smoothing_loop(
            gate_target, attack_coeff, release_coeff, floor_lin
        )

    if input_signal.ndim == 1:
        gain_envelope = _gate_gain_envelope(input_signal)
        output = (input_signal * gain_envelope).astype(np.float64)
    elif _is_stereo(input_signal):
        linked_key = np.max(np.abs(input_signal), axis=0)
        gain_envelope = _gate_gain_envelope(linked_key)
        output = (input_signal * gain_envelope[np.newaxis, :]).astype(np.float64)
    else:
        raise ValueError(f"Unsupported signal shape: {input_signal.shape}")

    if not return_analysis:
        return output

    analysis: dict[str, float | int | str] = {
        "threshold_db": round(threshold_db, 1),
        "hold_ms": round(hold_ms, 1),
        "release_ms": round(release_ms, 1),
        "floor_db": round(floor_db, 1),
    }
    return output, analysis


def apply_delay(
    signal: np.ndarray,
    delay_seconds: float = 0.35,
    feedback: float = 0.35,
    mix: float = 0.30,
) -> np.ndarray:
    """Apply pedalboard Delay to a mono or stereo signal."""
    board = _PEDALBOARD_CLS(
        [_DELAY_CLS(delay_seconds=delay_seconds, feedback=feedback, mix=mix)]
    )
    return _coerce_signal_layout(
        board(np.asarray(signal, dtype=np.float32), SAMPLE_RATE)
    )


def apply_reverb(
    signal: np.ndarray,
    room_size: float = 0.75,
    damping: float = 0.4,
    wet_level: float = 0.25,
) -> np.ndarray:
    """Apply pedalboard's built-in algorithmic reverb to mono or stereo."""
    board = _PEDALBOARD_CLS(
        [
            _REVERB_CLS(
                room_size=room_size,
                damping=damping,
                wet_level=wet_level,
                dry_level=1.0 - wet_level,
            )
        ]
    )
    return _coerce_signal_layout(
        board(np.asarray(signal, dtype=np.float32), SAMPLE_RATE)
    )


def apply_chow_tape(
    signal: np.ndarray,
    drive: float = 0.5,
    saturation: float = 0.5,
    bias: float = 0.5,
    mix: float = 70.0,
) -> np.ndarray:
    """Apply Chow Tape Model tape saturation to a mono or stereo signal.

    Parameters
    ----------
    drive:      Tape drive amount (0–1). Higher = more harmonic saturation.
    saturation: Tape saturation density (0–1).
    bias:       Tape bias (0–1). Controls harmonic balance / even vs. odd character.
    mix:        Dry/wet blend in percent (0–100).
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="chow_tape",
        params={
            "tape_drive": drive,
            "tape_saturation": saturation,
            "tape_bias": bias,
            "dry_wet": mix,
            # Keep this wrapper focused on glue/saturation, not motion/degradation.
            "wow_flutter_on_off": False,
            "loss_on_off": False,
        },
    )


# ---------------------------------------------------------------------------
# Airwindows Consolidated
# ---------------------------------------------------------------------------


def _switch_airwindows_algorithm(plugin: Any, algorithm: str) -> None:
    """Switch the Airwindows Consolidated plugin to a named algorithm.

    The plugin stores its current algorithm in VST3 preset XML.  To switch we
    grab the preset bytes, patch the ``currentProcessorName`` attribute and
    reset the ten generic ``awp_0``..``awp_9`` knobs to 0.5, write the result
    to a temporary ``.vstpreset`` file, then ``load_preset`` it back.
    """
    global _airwindows_base_preset  # noqa: PLW0603

    preset_data: bytes = plugin.preset_data
    if _airwindows_base_preset is None:
        _airwindows_base_preset = preset_data

    # --- Parse the VST3 preset container ---
    # Header: "VST3"(4) | version(u32 LE) | classID(32 ASCII) | state_size(u64 LE)
    header_size = 48
    if len(preset_data) < header_size:
        raise ValueError("Airwindows preset_data too short to contain a VST3 header")

    state_data = preset_data[header_size:]
    state_str = state_data.decode("latin-1")

    xml_start = state_str.find("<?xml")
    if xml_start == -1:
        raise ValueError("Could not locate XML block in Airwindows preset state")

    xml_end = state_str.find("/>", xml_start)
    if xml_end == -1:
        raise ValueError("Could not locate XML end in Airwindows preset state")
    xml_end += 2  # include the "/>"

    old_xml = state_str[xml_start:xml_end]

    # Patch the algorithm name
    new_xml = re.sub(
        r'currentProcessorName="[^"]*"',
        f'currentProcessorName="{algorithm}"',
        old_xml,
    )
    # Reset awp_0..awp_9 to 0.5
    for i in range(10):
        new_xml = re.sub(rf'awp_{i}="[^"]*"', f'awp_{i}="0.5"', new_xml)

    # Rebuild state bytes
    new_state_str = state_str[:xml_start] + new_xml + state_str[xml_end:]
    new_state_bytes = new_state_str.encode("latin-1")

    # Rebuild the full preset with updated state_size
    header_prefix = preset_data[:40]  # magic + version + classID
    new_state_size = struct.pack("<Q", len(new_state_bytes))
    new_preset = header_prefix + new_state_size + new_state_bytes

    fd, tmp_path = tempfile.mkstemp(suffix=".vstpreset")
    try:
        os.write(fd, new_preset)
        os.close(fd)
        plugin.load_preset(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _configure_airwindows(plugin: Any, params: dict[str, Any]) -> None:
    """Custom configurer for Airwindows Consolidated.

    Extracts the ``algorithm`` key, switches the plugin to that algorithm,
    then applies any remaining parameter overrides as plugin attributes.
    """
    algorithm = params.pop("algorithm", "Density")
    _switch_airwindows_algorithm(plugin, algorithm)

    input_level = params.pop("input_level", None)
    output_level = params.pop("output_level", None)
    if input_level is not None:
        plugin.input_level = float(input_level)
    if output_level is not None:
        plugin.output_level = float(output_level)

    for key, value in params.items():
        setattr(plugin, key, value)


def apply_airwindows(
    signal: np.ndarray,
    algorithm: str = "Density",
    *,
    input_level: float = 0.0,
    output_level: float = 0.0,
    **algo_params: float,
) -> np.ndarray:
    """Apply an Airwindows Consolidated algorithm to a mono or stereo signal.

    Parameters
    ----------
    algorithm:    Algorithm name (e.g. ``"Density"``, ``"ToTape6"``, ``"Tube"``).
    input_level:  Input trim in dB.
    output_level: Output trim in dB.
    **algo_params: Algorithm-specific parameters set as plugin attributes after
                   the algorithm switch.
    """
    merged: dict[str, Any] = {
        "algorithm": algorithm,
        "input_level": input_level,
        "output_level": output_level,
        **algo_params,
    }
    return _apply_plugin_processor(
        signal,
        plugin_name="airwindows",
        params=merged,
        configurer=_configure_airwindows,
    )


# ---------------------------------------------------------------------------
# BYOD (Build Your Own Distortion)
# ---------------------------------------------------------------------------


def _configure_byod(plugin: Any, params: dict[str, Any]) -> None:
    """Custom configurer for BYOD — sets program first, then params."""
    program = params.pop("program", "Tube Screamer")
    plugin.program = program

    for key, value in params.items():
        setattr(plugin, key, value)


def apply_byod(
    signal: np.ndarray,
    program: str = "Tube Screamer",
    *,
    in_gain: float = 0.0,
    out_gain: float = 0.0,
    dry_wet: float = 100.0,
    mode: str = "Stereo",
    **program_params: float | str | bool,
) -> np.ndarray:
    """Apply the BYOD multi-effect distortion/overdrive pedal.

    Parameters
    ----------
    program:        Preset program name (e.g. ``"Tube Screamer"``, ``"Centaur"``).
    in_gain:        Input gain in dB.
    out_gain:       Output gain in dB.
    dry_wet:        Dry/wet blend 0–100 (percent).
    mode:           Processing mode — ``"Stereo"`` or ``"Mono"``.
    **program_params: Additional program-specific parameters set as plugin
                      attributes after the program switch.
    """
    merged: dict[str, Any] = {
        "program": program,
        "in_gain": in_gain,
        "out_gain": out_gain,
        "dry_wet": dry_wet,
        "mode": mode,
        **program_params,
    }
    return _apply_plugin_processor(
        signal,
        plugin_name="byod",
        params=merged,
        configurer=_configure_byod,
    )


# ---------------------------------------------------------------------------
# ChowCentaur (Klon overdrive)
# ---------------------------------------------------------------------------


def apply_chow_centaur(
    signal: np.ndarray,
    gain: float = 0.3,
    treble: float = 0.5,
    level: float = 0.7,
    mode: str = "Neural",
) -> np.ndarray:
    """Apply the ChowCentaur Klon-style overdrive.

    Parameters
    ----------
    gain:   Drive amount 0–1.
    treble: Treble/tone control 0–1.
    level:  Output level 0–1.
    mode:   Model mode — ``"Neural"`` or ``"Traditional"``.
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="chow_centaur",
        params={
            "gain": float(gain),
            "treble": float(treble),
            "level": float(level),
            "mode": mode,
        },
    )


# ---------------------------------------------------------------------------
# Valhalla Supermassive (reverb / delay / shimmer)
# ---------------------------------------------------------------------------


def apply_valhalla_supermassive(
    signal: np.ndarray,
    mix: float = 50.0,
    delay_ms: float = 300.0,
    feedback: float = 50.0,
    density: float = 0.0,
    width: float = 100.0,
    low_cut: float = 10.0,
    high_cut: float = 20000.0,
    mod_rate: float = 0.5,
    mod_depth: float = 0.0,
) -> np.ndarray:
    """Apply Valhalla Supermassive reverb/delay.

    Parameters
    ----------
    mix:       Dry/wet blend 0–100 (percent).
    delay_ms:  Delay time in milliseconds.
    feedback:  Feedback amount 0–100.
    density:   Density control 0–100.
    width:     Stereo width 0–100.
    low_cut:   Low-cut filter frequency in Hz.
    high_cut:  High-cut filter frequency in Hz.
    mod_rate:  Modulation rate in Hz.
    mod_depth: Modulation depth 0–100.
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="valhalla_supermassive",
        params={
            "mix": float(mix),
            "delay_ms": float(delay_ms),
            "feedback": float(feedback),
            "density": float(density),
            "width": float(width),
            "lowcut": float(low_cut),
            "highcut": float(high_cut),
            "modrate": float(mod_rate),
            "moddepth": float(mod_depth),
        },
    )


# ---------------------------------------------------------------------------
# Valhalla FreqEcho (frequency-shifting delay)
# ---------------------------------------------------------------------------


def apply_valhalla_freq_echo(
    signal: np.ndarray,
    mix: float = 50.0,
    shift: float = 0.0,
    delay: float = 0.01,
    feedback: float = 50.0,
    low_cut: float = 200.0,
    high_cut: float = 5000.0,
) -> np.ndarray:
    """Apply Valhalla FreqEcho frequency-shifting delay.

    Parameters
    ----------
    mix:      Dry/wet blend 0–100 (percent).
    shift:    Frequency shift amount (semitones or plugin units).
    delay:    Delay time (seconds or plugin units).
    feedback: Feedback amount 0–100.
    low_cut:  Low-cut filter frequency in Hz.
    high_cut: High-cut filter frequency in Hz.
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="valhalla_freq_echo",
        params={
            "wetdry": float(mix),
            "shift": float(shift),
            "delay": float(delay),
            "feedback": float(feedback),
            "lowcut": float(low_cut),
            "highcut": float(high_cut),
        },
    )


# ---------------------------------------------------------------------------
# Valhalla Space Modulator (flanging / modulation)
# ---------------------------------------------------------------------------


def apply_valhalla_space_mod(
    signal: np.ndarray,
    mix: float = 50.0,
    rate: float = 0.1,
    depth: float = 10.0,
    feedback: float = 0.0,
) -> np.ndarray:
    """Apply Valhalla SpaceModulator flanging/modulation effect.

    Parameters
    ----------
    mix:      Dry/wet blend 0–100 (percent).
    rate:     Modulation rate in Hz.
    depth:    Modulation depth 0–100.
    feedback: Feedback amount 0–100.
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="valhalla_space_mod",
        params={
            "wetdry": float(mix),
            "rate": float(rate),
            "depth": float(depth),
            "feedback": float(feedback),
        },
    )


# ---------------------------------------------------------------------------
# TDR Kotelnikov (transparent mastering compressor)
# ---------------------------------------------------------------------------


def apply_tdr_kotelnikov(
    signal: np.ndarray,
    threshold_db: float = 0.0,
    ratio: float = 2.0,
    attack_ms: float = 6.0,
    release_rms_ms: float = 220.0,
    makeup_db: float = 0.0,
    soft_knee_db: float = 1.0,
    peak_crest_db: float = 3.0,
    dry_wet: float = 0.0,
) -> np.ndarray:
    """Apply TDR Kotelnikov transparent compressor.

    Parameters
    ----------
    threshold_db:   Compression threshold in dB.
    ratio:          Compression ratio (e.g. 2.0 = 2:1).
    attack_ms:      Attack time in milliseconds.
    release_rms_ms: RMS release time in milliseconds.
    makeup_db:      Makeup gain in dB.
    soft_knee_db:   Soft knee width in dB.
    peak_crest_db:  Peak/crest balance in dB.
    dry_wet:        Dry/wet mix — 0 = fully compressed, higher = more dry.
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="tdr_kotelnikov",
        params={
            "threshold_db": float(threshold_db),
            "ratio": float(ratio),
            "attack_ms": float(attack_ms),
            "release_rms_ms": float(release_rms_ms),
            "makeup_db": float(makeup_db),
            "soft_knee_db": float(soft_knee_db),
            "peak_crest_db": float(peak_crest_db),
            "dry_wet": float(dry_wet),
        },
    )


# ---------------------------------------------------------------------------
# MJUCjr (vari-mu compressor)
# ---------------------------------------------------------------------------


def apply_mjuc_jr(
    signal: np.ndarray,
    compress: float = 0.0,
    makeup: float = 0.0,
    timing: str = "slow",
) -> np.ndarray:
    """Apply MJUCjr vari-mu compressor.

    Parameters
    ----------
    compress: Compression amount in dB (0–48).
    makeup:   Makeup gain in dB (0–48).
    timing:   Timing mode — ``"slow"`` or ``"fast"``.
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="mjuc_jr",
        params={
            "compress": float(compress),
            "makeup": float(makeup),
            "timing": timing,
        },
    )


# ---------------------------------------------------------------------------
# FETish (1176-style FET compressor)
# ---------------------------------------------------------------------------


def apply_fetish(
    signal: np.ndarray,
    input_db: float = -20.0,
    output_db: float = 0.0,
    attack_us: float = 500.0,
    release_ms: float = 400.0,
    ratio: float = 4.0,
    mix: float = 100.0,
    hpf_hz: float = 20.0,
) -> np.ndarray:
    """Apply FETish 1176-style FET compressor.

    Parameters
    ----------
    input_db:   Input level in dB.
    output_db:  Output level in dB.
    attack_us:  Attack time in microseconds.
    release_ms: Release time in milliseconds.
    ratio:      Compression ratio (e.g. 4.0 = 4:1).
    mix:        Dry/wet blend 0–100 (percent).
    hpf_hz:     Sidechain high-pass filter frequency in Hz.
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="fetish",
        params={
            "input_db": float(input_db),
            "output_db": float(output_db),
            "attack_us": float(attack_us),
            "release_ms": float(release_ms),
            "ratio": float(ratio),
            "mix": float(mix),
            "hpf_hz": float(hpf_hz),
        },
    )


# ---------------------------------------------------------------------------
# LALA (LA-2A-style optical compressor)
# ---------------------------------------------------------------------------


def apply_lala(
    signal: np.ndarray,
    gain: float = 30.0,
    peak_reduction: float = 0.0,
    hf: float = 100.0,
    mode: float = 1.0,
) -> np.ndarray:
    """Apply LALA LA-2A-style optical compressor.

    Parameters
    ----------
    gain:           Output gain 0–100.
    peak_reduction: Peak reduction (compression depth) 0–100.
    hf:             High-frequency tilt control 0–100.
    mode:           Operating mode — 1.0 = compress, other values = limit.
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="lala",
        params={
            "gain": float(gain),
            "peak_reduction": float(peak_reduction),
            "hf": float(hf),
            "mode": float(mode),
        },
    )


# ---------------------------------------------------------------------------
# IVGI (saturation / distortion)
# ---------------------------------------------------------------------------


def apply_ivgi(
    signal: np.ndarray,
    drive: float = 0.0,
    trim: float = 0.0,
    output: float = 0.0,
    asymmetry: float = 5.0,
    freq_response: float = 0.0,
) -> np.ndarray:
    """Apply IVGI saturation/distortion.

    Parameters
    ----------
    drive:         Drive amount (0–10).
    trim:          Input trim (0–10).
    output:        Output level (0–10).
    asymmetry:     Asymmetry control (0–10).
    freq_response: Frequency response tilt (0–10).
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="ivgi",
        params={
            "drive": float(drive),
            "trim": float(trim),
            "output": float(output),
            "asymmetry": float(asymmetry),
            "freqresponse": float(freq_response),
        },
    )


# ---------------------------------------------------------------------------
# BritChannel (Neve 1073 channel strip)
# ---------------------------------------------------------------------------


def apply_brit_channel(
    signal: np.ndarray,
    preamp_gain_db: float = 0.0,
    output_trim_db: float = 0.0,
    highpass: str = "OFF",
    low_freq: str = "OFF",
    low_gain_db: float = 0.0,
    mid_freq: str = "OFF",
    mid_gain_db: float = 0.0,
    high_gain_db: float = 0.0,
) -> np.ndarray:
    """Apply BritChannel Neve 1073-style channel strip.

    Parameters
    ----------
    preamp_gain_db: Preamp gain -24 to 24 dB.
    output_trim_db: Output trim -24 to 24 dB.
    highpass:       High-pass filter — "OFF"|"50Hz"|"80Hz"|"160Hz"|"300Hz".
    low_freq:       Low EQ frequency — "OFF"|"35Hz"|"60Hz"|"110Hz"|"220Hz".
    low_gain_db:    Low EQ gain -15 to 15 dB.
    mid_freq:       Mid EQ frequency — "OFF"|".36kHz"|".7kHz"|"1.6kHz"|"3.2kHz"|"4.8kHz"|"7.2kHz".
    mid_gain_db:    Mid EQ gain -15 to 15 dB.
    high_gain_db:   High EQ gain -15 to 15 dB.
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="brit_channel",
        params={
            "preamp_gain_db": float(preamp_gain_db),
            "output_trim_db": float(output_trim_db),
            "highpass": highpass,
            "low_freq": low_freq,
            "low_gain_db": float(low_gain_db),
            "mid_freq": mid_freq,
            "mid_gain_db": float(mid_gain_db),
            "high_gain_db": float(high_gain_db),
        },
    )


# ---------------------------------------------------------------------------
# BritPre (Neve preamp)
# ---------------------------------------------------------------------------


def apply_brit_pre(
    signal: np.ndarray,
    gain: float = 0.0,
    output_db: float = 0.0,
    highpass_filter: str = "OFF",
    lowpass_filter: str = "OFF",
) -> np.ndarray:
    """Apply BritPre Neve-style preamp.

    Parameters
    ----------
    gain:            Preamp gain in 5 dB steps, -20 to 40 dB.
    output_db:       Output level -24 to 24 dB.
    highpass_filter: High-pass — "OFF"|"45Hz"|"70Hz"|"160Hz"|"360Hz".
    lowpass_filter:  Low-pass — "OFF"|"8kHz"|"6kHz"|"4kHz"|"2kHz".
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="brit_pre",
        params={
            "gain": float(gain),
            "output_db": float(output_db),
            "highpass_filter": highpass_filter,
            "lowpass_filter": lowpass_filter,
        },
    )


# ---------------------------------------------------------------------------
# Britpressor (Neve 2254 compressor/limiter)
# ---------------------------------------------------------------------------


def apply_britpressor(
    signal: np.ndarray,
    compressor_threshold: float = 10.0,
    ratio: str = "3:1",
    gain: float = 0.0,
    mix: str = "0/100",
    compressor_recovery_time: str = "400ms",
    limit_level: float = 15.0,
    level_recovery_time: str = "100ms",
    high: float = 0.0,
    mid: float = 0.0,
    high_pass_filter: str = "OFF",
) -> np.ndarray:
    """Apply Britpressor Neve 2254-style compressor/limiter.

    Parameters
    ----------
    compressor_threshold: Threshold -20 to 10 dB.
    ratio:                Ratio — "1.5:1"|"2:1"|"3:1"|"4:1"|"6:1".
    gain:                 Makeup gain 0 to 20 dB.
    mix:                  Dry/wet blend (dry/wet) — 23 steps from "100/0" to "0/100".
    compressor_recovery_time: Recovery — "100ms"|"400ms"|"800ms"|"1500ms"|"Auto-1"|"Auto-2".
    limit_level:          Limiter level 4 to 15.
    level_recovery_time:  Limiter recovery — "50ms"|"100ms"|"200ms"|"800ms"|"Auto-1"|"Auto-2".
    high:                 High EQ -6 to 6 dB (discrete 3 dB steps).
    mid:                  Mid EQ -6 to 6 dB (discrete 3 dB steps).
    high_pass_filter:     Sidechain HPF — "OFF"|"50Hz"|"80Hz"|"160Hz"|"360Hz".
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="britpressor",
        params={
            "compressor_threshold": float(compressor_threshold),
            "ratio": ratio,
            "gain": float(gain),
            "mix": mix,
            "compressor_recovery_time": compressor_recovery_time,
            "limit_level": float(limit_level),
            "level_recovery_time": level_recovery_time,
            "high": float(high),
            "mid": float(mid),
            "high_pass_filter": high_pass_filter,
        },
    )


# ---------------------------------------------------------------------------
# Distox (multi-mode distortion)
# ---------------------------------------------------------------------------


def apply_distox(
    signal: np.ndarray,
    input_db: float = 0.0,
    output_db: float = 0.0,
    mix: float = 100.0,
    hpf_hz: float = 5.0,
    lpf_khz: float = 20.0,
    mode: str = "Op-Amp 1",
) -> np.ndarray:
    """Apply Distox multi-mode distortion.

    Parameters
    ----------
    input_db:  Input level -30 to 30 dB.
    output_db: Output level -30 to 30 dB.
    mix:       Dry/wet blend 0–100 (percent).
    hpf_hz:    High-pass filter 5–2000 Hz.
    lpf_khz:   Low-pass filter 10–20 kHz.
    mode:      Distortion mode — "Op-Amp 1"|"Op-Amp 2"|"Op-Amp 3"|"Tube 1"|"Tube 2"|"Tube 3"|"Tube 4".
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="distox",
        params={
            "input_db": float(input_db),
            "output_db": float(output_db),
            "mix": float(mix),
            "hpf_hz": float(hpf_hz),
            "lpf_khz": float(lpf_khz),
            "mode": mode,
        },
    )


# ---------------------------------------------------------------------------
# FetDrive (FET saturation)
# ---------------------------------------------------------------------------


def apply_fet_drive(
    signal: np.ndarray,
    drive_db: float = 0.0,
    tone: float = 50.0,
    output_db: float = 0.0,
    mix: float = 100.0,
) -> np.ndarray:
    """Apply FetDrive FET saturation.

    Parameters
    ----------
    drive_db:  Drive amount 0–50 dB.
    tone:      Tone control 0–100.
    output_db: Output level -15 to 15 dB.
    mix:       Dry/wet blend 0–100 (percent).
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="fet_drive",
        params={
            "drive_db": float(drive_db),
            "tone": float(tone),
            "output_db": float(output_db),
            "mix": float(mix),
        },
    )


# ---------------------------------------------------------------------------
# Kolin (SSL-style bus compressor)
# ---------------------------------------------------------------------------


def apply_kolin(
    signal: np.ndarray,
    input_db: float = 0.0,
    output_db: float = 0.0,
    attack_ms: float = 10.0,
    release_ms: float = 400.0,
    mix: float = 100.0,
    hpf_hz: float = 20.0,
) -> np.ndarray:
    """Apply Kolin SSL-style bus compressor.

    Parameters
    ----------
    input_db:   Input level 0–40 dB.
    output_db:  Output level -20 to 20 dB.
    attack_ms:  Attack time 1–50 ms.
    release_ms: Release time 100–3000 ms.
    mix:        Dry/wet blend 0–100 (percent).
    hpf_hz:     Sidechain high-pass filter 20–500 Hz.
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="kolin",
        params={
            "input_db": float(input_db),
            "output_db": float(output_db),
            "attack_ms": float(attack_ms),
            "release_ms": float(release_ms),
            "mix": float(mix),
            "hpf_hz": float(hpf_hz),
        },
    )


# ---------------------------------------------------------------------------
# LAEA (leveling amplifier / LA-3A style)
# ---------------------------------------------------------------------------


def apply_laea(
    signal: np.ndarray,
    gain: float = 0.0,
    reduction: float = 30.0,
    mix: float = 100.0,
    limit: bool = False,
) -> np.ndarray:
    """Apply LAEA LA-3A-style leveling amplifier.

    Parameters
    ----------
    gain:      Output gain 0–100.
    reduction: Peak reduction (compression depth) 0–100.
    mix:       Dry/wet blend 0–100 (percent).
    limit:     Limiter mode — True for limit, False for compress.
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="laea",
        params={
            "gain": float(gain),
            "reduction": float(reduction),
            "mix": float(mix),
            "limit": 1.0 if limit else 0.0,
        },
    )


# ---------------------------------------------------------------------------
# MERICA (American EQ)
# ---------------------------------------------------------------------------


def apply_merica(
    signal: np.ndarray,
    low_gain_db: float = 0.0,
    mid_gain_db: float = 0.0,
    high_gain_db: float = 0.0,
    low_freq: float = 50.0,
    mid_freq: float = 400.0,
    high_freq: float = 5000.0,
    input_db: float = 0.0,
    output_db: float = 0.0,
) -> np.ndarray:
    """Apply MERICA American-style 3-band EQ.

    Parameters
    ----------
    low_gain_db:  Low band gain -12 to 12 dB.
    mid_gain_db:  Mid band gain -12 to 12 dB.
    high_gain_db: High band gain -12 to 12 dB.
    low_freq:     Low band frequency (discrete) — 50|100|200|300|400 Hz.
    mid_freq:     Mid band frequency (discrete) — 400|800|1600|3000|5000 Hz.
    high_freq:    High band frequency (discrete) — 5000|7000|10000|12500|15000 Hz.
    input_db:     Input level -12 to 12 dB.
    output_db:    Output level -12 to 12 dB.
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="merica",
        params={
            "low_gain_db": float(low_gain_db),
            "mid_gain_db": float(mid_gain_db),
            "high_gain_db": float(high_gain_db),
            "low_freq": float(low_freq),
            "mid_freq": float(mid_freq),
            "high_freq": float(high_freq),
            "input_db": float(input_db),
            "output_db": float(output_db),
        },
    )


# ---------------------------------------------------------------------------
# PreBOX (preamp saturation)
# ---------------------------------------------------------------------------


def apply_prebox(
    signal: np.ndarray,
    input_db: float = 0.0,
    output_db: float = 0.0,
    model: float = 0.0,
    hpf: float = 0.0,
    lpf: float = 0.0,
    agc: str = "AGC On",
) -> np.ndarray:
    """Apply PreBOX preamp saturation.

    Parameters
    ----------
    input_db:  Input level -24 to 24 dB.
    output_db: Output level -24 to 24 dB.
    model:     Preamp model 0–10 (11 discrete models).
    hpf:       High-pass filter setting 0|1|2|3.
    lpf:       Low-pass filter setting 0|1.
    agc:       Auto gain compensation — "AGC Off"|"AGC On".
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="prebox",
        params={
            "input_db": float(input_db),
            "output_db": float(output_db),
            "model": float(model),
            "hpf": float(hpf),
            "lpf": float(lpf),
            "agc": agc,
        },
    )


# ---------------------------------------------------------------------------
# RareSE (Pultec-style EQ)
# ---------------------------------------------------------------------------


def apply_rare_se(
    signal: np.ndarray,
    low_boost: float = 0.0,
    low_atten: float = 0.0,
    high_boost: float = 0.0,
    high_atten: float = 0.0,
    high_bandwidth: float = 10.0,
    output_db: float = 0.0,
    low_frequency: float = 60.0,
    high_frequency: float = 8000.0,
    high_atten_frequency: float = 20000.0,
) -> np.ndarray:
    """Apply RareSE Pultec-style passive EQ (L/M section).

    Parameters
    ----------
    low_boost:            Low boost 0–10.
    low_atten:            Low attenuation 0–10.
    high_boost:           High boost 0–10.
    high_atten:           High attenuation 0–10.
    high_bandwidth:       High boost bandwidth 1–10.
    output_db:            Output level in dB.
    low_frequency:        Low band frequency in Hz (20–100).
    high_frequency:       High boost frequency in Hz (3000–16000).
    high_atten_frequency: High attenuation frequency in Hz (up to 20000).
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="rare_se",
        params={
            "l_m_low_boost": float(low_boost),
            "l_m_low_atten": float(low_atten),
            "l_m_high_boost": float(high_boost),
            "l_m_high_atten": float(high_atten),
            "l_m_high_bandwidth": float(high_bandwidth),
            "l_m_output_db": float(output_db),
            "l_m_low_frequency": float(low_frequency),
            "l_m_high_freqency": float(high_frequency),
            "l_m_high_atten_freqency": float(high_atten_frequency),
        },
    )


# ---------------------------------------------------------------------------
# TUBA (tube amplifier)
# ---------------------------------------------------------------------------


def apply_tuba(
    signal: np.ndarray,
    level: float = 7.0,
    output_db: float = 0.0,
    gain: str = "Low Gain",
    high_gain: float = 0.0,
    low_gain: float = 0.0,
) -> np.ndarray:
    """Apply TUBA tube amplifier.

    Parameters
    ----------
    level:     Drive level 1–20.
    output_db: Output level -96 to 12 dB.
    gain:      Gain structure — "Low Gain"|"High Gain".
    high_gain: High EQ -6|0|3|6 dB.
    low_gain:  Low EQ -6|0|6 dB.
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="tuba",
        params={
            "level": float(level),
            "output_db": float(output_db),
            "gain": gain,
            "high_gain": float(high_gain),
            "low_gain": float(low_gain),
        },
    )


def apply_bricasti(
    signal: np.ndarray,
    ir_name: str,
    wet: float = 0.35,
    highpass_hz: float = 0.0,
    lowpass_hz: float = 0.0,
    tilt_db: float = 0.0,
    tilt_pivot_hz: float = 1_500.0,
) -> np.ndarray:
    """Convolve a mono or stereo signal with a Bricasti stereo impulse response."""
    if not 0.0 <= wet <= 1.0:
        raise ValueError("wet must be between 0 and 1")

    ir_l = BRICASTI_IR_DIR / f"{ir_name}, 44K L.wav"
    ir_r = BRICASTI_IR_DIR / f"{ir_name}, 44K R.wav"
    if not ir_l.exists() or not ir_r.exists():
        raise FileNotFoundError(f"IR not found: {ir_name!r} - check BRICASTI_IR_DIR")

    stereo_signal = _ensure_stereo(signal).astype(np.float32)
    left = _PEDALBOARD_CLS([_CONVOLUTION_CLS(str(ir_l), mix=1.0)])(
        stereo_signal[0], SAMPLE_RATE
    )
    right = _PEDALBOARD_CLS([_CONVOLUTION_CLS(str(ir_r), mix=1.0)])(
        stereo_signal[1], SAMPLE_RATE
    )
    n_samples = stereo_signal.shape[-1]
    wet_signal = np.stack([left[:n_samples], right[:n_samples]]).astype(np.float64)
    wet_signal = _shape_reverb_return(
        wet_signal,
        sample_rate=SAMPLE_RATE,
        highpass_hz=highpass_hz,
        lowpass_hz=lowpass_hz,
        tilt_db=tilt_db,
        tilt_pivot_hz=tilt_pivot_hz,
    )
    blended = ((1.0 - wet) * stereo_signal.astype(np.float64)) + (wet * wet_signal)
    return _match_input_layout(blended.astype(np.float64), signal)


def apply_tal_chorus_lx(
    signal: np.ndarray,
    mix: float = 0.5,
    chorus_1: bool = True,
    chorus_2: bool = False,
    stereo: float = 1.0,
) -> np.ndarray:
    """Apply TAL-Chorus-LX (Roland Juno-60 BBD chorus emulation).

    Parameters
    ----------
    mix:      Dry/wet blend 0–1. Mapped to plugin's 0–10 dry_wet control.
    chorus_1: Enable chorus mode I (subtle, slower LFO).
    chorus_2: Enable chorus mode II (wider, faster LFO). Both can be on together.
    stereo:   Stereo width 0–1. Mapped to plugin's 0–10 stereo control.
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="tal_chorus_lx",
        params={
            "chorus_1": 1.0 if chorus_1 else 0.0,
            "chorus_2": 1.0 if chorus_2 else 0.0,
            "dry_wet": float(mix) * 10.0,
            "stereo": float(stereo) * 10.0,
            "volume": 5.0,
        },
    )


def apply_tal_reverb2(
    signal: np.ndarray,
    wet: float = 0.3,
    room_size: float = 0.75,
    pre_delay: float = 0.13,
    stereo: float = 1.0,
) -> np.ndarray:
    """Apply TAL-Reverb-2 algorithmic reverb.

    Parameters
    ----------
    wet:        Wet level 0–1.
    room_size:  Room size 0–1.
    pre_delay:  Pre-delay 0–1 (plugin's normalized range).
    stereo:     Stereo width 0–1.
    """
    return _apply_plugin_processor(
        signal,
        plugin_name="tal_reverb2",
        params={
            "dry": 1.0,
            "wet": float(wet),
            "room_size": float(room_size),
            "pre_delay": float(pre_delay),
            "stereo": float(stereo),
        },
    )


def apply_dragonfly(
    signal: np.ndarray,
    variant: str = "plate",
    wet_level: float = 20.0,
    dry_level: float = 100.0,
    decay_s: float = 0.4,
    width: float = 100.0,
    predelay_ms: float = 0.0,
    low_cut_hz: float = 200.0,
    high_cut_hz: float = 16000.0,
    dampen_hz: float = 13000.0,
    size_m: float = 12.0,
    diffuse: float = 70.0,
) -> np.ndarray:
    """Apply a Dragonfly Reverb plugin.

    Parameters
    ----------
    variant:      "plate", "room", "hall", or "early".
    wet_level:    Wet level 0–100 (percent).
    dry_level:    Dry level 0–100 (percent). Set to 0 for send-bus (100% wet) usage.
    decay_s:      Reverb decay in seconds.
    width:        Stereo width 0–100.
    predelay_ms:  Pre-delay in milliseconds.
    low_cut_hz:   Low-cut frequency (plate, room, hall).
    high_cut_hz:  High-cut frequency (plate, room, hall).
    dampen_hz:    Damping cutoff — plate only; ignored for other variants.
    size_m:       Room/hall size in metres — room and hall only.
    diffuse:      Diffusion 0–100 — room and hall only.
    """
    plugin_name = f"dragonfly_{variant}"
    if plugin_name not in _PLUGIN_SPECS:
        raise ValueError(
            f"Unknown Dragonfly variant {variant!r}. Choose from: ['early', 'hall', 'plate', 'room']"
        )
    return _apply_plugin_processor(
        signal,
        plugin_name=plugin_name,
        params={
            "dry_level": float(dry_level),
            "wet_level": float(wet_level),
            "decay_s": float(decay_s),
            "width": float(width),
            "predelay_ms": float(predelay_ms),
            "low_cut_hz": float(low_cut_hz),
            "high_cut_hz": float(high_cut_hz),
            "dampen_hz": float(dampen_hz),
            "size_m": float(size_m),
            "diffuse": float(diffuse),
        },
        configurer=_configure_dragonfly_plugin,
    )


def _configure_dragonfly_plugin(plugin: Any, params: dict[str, Any]) -> None:
    plugin.dry_level = params["dry_level"]
    plugin.wet_level = params["wet_level"]
    plugin.decay_s = params["decay_s"]
    plugin.width = params["width"]
    if hasattr(plugin, "predelay_ms"):
        plugin.predelay_ms = params["predelay_ms"]
    if hasattr(plugin, "low_cut_hz"):
        plugin.low_cut_hz = params["low_cut_hz"]
    if hasattr(plugin, "high_cut_hz"):
        plugin.high_cut_hz = params["high_cut_hz"]
    if hasattr(plugin, "dampen_hz"):
        plugin.dampen_hz = params["dampen_hz"]
    if hasattr(plugin, "size_m"):
        plugin.size_m = params["size_m"]
    if hasattr(plugin, "diffuse"):
        plugin.diffuse = params["diffuse"]


def apply_plugin(
    signal: np.ndarray,
    *,
    plugin_name: str | None = None,
    plugin_path: str | Path | None = None,
    plugin_format: str = "vst3",
    host: str = "pedalboard",
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply an external audio plugin via the configured plugin host backend."""
    return _apply_plugin_processor(
        signal,
        plugin_name=plugin_name,
        plugin_path=plugin_path,
        plugin_format=plugin_format,
        host=host,
        params=params,
    )


def apply_lsp_limiter(
    signal: np.ndarray,
    *,
    threshold_db: float = -0.5,
    input_gain_db: float = 0.0,
    output_gain_db: float = 0.0,
) -> np.ndarray:
    """Apply the LSP stereo limiter with the exposed VST3 parameters."""
    return _apply_plugin_processor(
        signal,
        plugin_name="lsp_limiter_stereo",
        params={
            "threshold_db": float(threshold_db),
            "input_gain_db": float(input_gain_db),
            "output_gain_db": float(output_gain_db),
        },
    )


@numba.njit(cache=True)
def _limiter_smoothing_loop(
    gain_reduction_db: np.ndarray,
    attack_coeff: float,
    release_coeff: float,
) -> np.ndarray:
    n = gain_reduction_db.shape[0]
    smoothed = np.empty(n, dtype=np.float64)
    smoothed[0] = gain_reduction_db[0]
    for i in range(1, n):
        target = gain_reduction_db[i]
        coeff = attack_coeff if target < smoothed[i - 1] else release_coeff
        smoothed[i] = coeff * smoothed[i - 1] + (1.0 - coeff) * target
    return smoothed


@overload
def apply_native_limiter(
    signal: np.ndarray,
    *,
    threshold_db: float = ...,
    input_gain_db: float = ...,
    output_gain_db: float = ...,
    sample_rate: int = ...,
    lookahead_ms: float = ...,
    release_ms: float = ...,
    oversample_factor: int = ...,
    headroom_db: float | None = ...,
    calibration_percentile: float = ...,
    return_analysis: Literal[False] = ...,
) -> np.ndarray: ...


@overload
def apply_native_limiter(
    signal: np.ndarray,
    *,
    threshold_db: float = ...,
    input_gain_db: float = ...,
    output_gain_db: float = ...,
    sample_rate: int = ...,
    lookahead_ms: float = ...,
    release_ms: float = ...,
    oversample_factor: int = ...,
    headroom_db: float | None = ...,
    calibration_percentile: float = ...,
    return_analysis: Literal[True],
) -> tuple[np.ndarray, dict[str, float | int | str]]: ...


def apply_native_limiter(
    signal: np.ndarray,
    *,
    threshold_db: float = -0.5,
    input_gain_db: float = 0.0,
    output_gain_db: float = 0.0,
    sample_rate: int = SAMPLE_RATE,
    lookahead_ms: float = 1.5,
    release_ms: float = 50.0,
    oversample_factor: int = 4,
    headroom_db: float | None = None,
    calibration_percentile: float = 99.9,
    return_analysis: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int | str]]:
    """Native true-peak lookahead brickwall limiter.

    Oversamples for inter-sample peak detection, applies gain reduction with
    lookahead and smooth attack/release, then downsamples back.

    Auto-calibration (piece-aware headroom targeting)
    -------------------------------------------------
    When ``headroom_db`` is set (default ``None``), the limiter computes the
    ``calibration_percentile``-th percentile (default p99.9) of a windowed
    peak distribution of the input signal and sets
    ``threshold_db = reference_peak_dbfs - headroom_db``.  This mirrors
    :func:`apply_clipper`'s ``max_shave_db`` auto-cal, but applied to the
    limiter's ceiling rather than the clipper's shave target.  When both
    ``headroom_db`` and a non-default ``threshold_db`` are set, a warning is
    logged and ``headroom_db`` wins.

    Parameters
    ----------
    headroom_db : float | None
        When set, auto-calibrate ``threshold_db`` so that the limiter's
        ceiling sits ``headroom_db`` below the ``calibration_percentile``-th
        percentile of the windowed input peaks.  Default ``None`` (no
        auto-calibration; ``threshold_db`` is used directly).
    calibration_percentile : float
        Percentile of the windowed peak distribution used for calibration
        (bounded to ``[50, 100]``).  Default ``99.9`` — robust to isolated
        transients while ignoring near-silence.
    """
    attack_ms = 0.1

    sig = np.asarray(signal, dtype=np.float64)
    was_mono = sig.ndim == 1
    if was_mono:
        sig = sig[np.newaxis, :]
    if sig.shape[-1] == 0:
        return sig[0] if was_mono else sig
    original_length = sig.shape[1]

    if headroom_db is not None and headroom_db < 0:
        raise ValueError(
            f"headroom_db must be non-negative when set, got {headroom_db}"
        )

    # Auto-calibrate threshold_db from windowed peak statistics before the
    # input gain is applied — the calibration targets the *pre-input-gain*
    # signal, matching the clipper's ``reference_peak_dbfs`` convention.
    threshold_db_user_set = threshold_db != -0.5
    auto_calibrated = headroom_db is not None
    reference_peak_dbfs: float | None = None
    effective_percentile: float | None = None
    if auto_calibrated:
        assert headroom_db is not None
        if threshold_db_user_set:
            logger.warning(
                "Limiter: both threshold_db and headroom_db set — "
                "headroom_db wins; threshold_db ignored."
            )
        effective_percentile = float(np.clip(calibration_percentile, 50.0, 100.0))
        calibration_window_ms = 50.0
        window_samples = max(1, int(round(calibration_window_ms * 1e-3 * sample_rate)))
        peak_track = np.max(np.abs(sig), axis=0)
        n = peak_track.shape[0]
        if n < window_samples:
            reference_peak = float(np.max(peak_track)) if n > 0 else 1e-12
        else:
            n_windows = n // window_samples
            trimmed = peak_track[: n_windows * window_samples]
            windowed = trimmed.reshape(n_windows, window_samples)
            window_peaks = np.max(windowed, axis=1)
            reference_peak = float(np.percentile(window_peaks, effective_percentile))
        reference_peak = max(reference_peak, 1e-12)
        reference_peak_dbfs = float(20.0 * np.log10(reference_peak))
        threshold_db = reference_peak_dbfs - float(headroom_db)
        logger.info(
            f"Limiter: auto-calibrated threshold to {threshold_db:.2f} dBFS "
            f"(p{effective_percentile:g} peak {reference_peak_dbfs:.2f} dBFS, "
            f"headroom {float(headroom_db):.2f} dB)"
        )

    sig = sig * db_to_amp(input_gain_db)

    os_rate = sample_rate * oversample_factor
    oversampled = np.stack(
        [
            np.asarray(resample_poly(ch, oversample_factor, 1), dtype=np.float64)
            for ch in sig
        ]
    )

    threshold_amp = db_to_amp(threshold_db)
    linked_peak = np.max(np.abs(oversampled), axis=0)

    gain_reduction_db = np.zeros(linked_peak.shape[0], dtype=np.float64)
    above_mask = linked_peak > threshold_amp
    gain_reduction_db[above_mask] = threshold_db - 20.0 * np.log10(
        linked_peak[above_mask]
    )

    lookahead_samples = int(lookahead_ms * 0.001 * os_rate)
    if lookahead_samples > 0:
        shifted = np.empty_like(gain_reduction_db)
        shifted[:(-lookahead_samples)] = gain_reduction_db[lookahead_samples:]
        shifted[(-lookahead_samples):] = 0.0
        reversed_shifted = shifted[::-1]
        propagated = np.minimum.accumulate(reversed_shifted)
        gain_reduction_db = np.minimum(gain_reduction_db, propagated[::-1])

    attack_coeff = _time_constant_to_coeff(attack_ms, os_rate)
    release_coeff = _time_constant_to_coeff(release_ms, os_rate)

    gain_reduction_db = _limiter_smoothing_loop(
        gain_reduction_db, attack_coeff, release_coeff
    )

    gain_linear = 10.0 ** (gain_reduction_db / 20.0)
    limited_oversampled = oversampled * gain_linear[np.newaxis, :]

    limited = np.stack(
        [
            np.asarray(resample_poly(ch, 1, oversample_factor), dtype=np.float64)
            for ch in limited_oversampled
        ]
    )

    if limited.shape[1] > original_length:
        limited = limited[:, :original_length]
    elif limited.shape[1] < original_length:
        pad_width = original_length - limited.shape[1]
        limited = np.pad(limited, ((0, 0), (0, pad_width)))

    limited = limited * db_to_amp(output_gain_db)

    max_gr_db = float(np.min(gain_reduction_db))
    active_mask = gain_reduction_db < -0.01
    active_fraction = float(np.mean(active_mask))
    avg_gr_when_active_db = (
        float(np.mean(gain_reduction_db[active_mask])) if active_fraction > 0 else 0.0
    )
    logger.info(
        f"Native limiter: max gain reduction {max_gr_db:.2f} dB, "
        f"limiting active on {active_fraction * 100.0:.1f}% of samples"
    )
    if max_gr_db < -6.0:
        logger.warning(
            f"Native limiter: heavy limiting ({max_gr_db:.1f} dB max GR) — "
            "input is significantly hotter than threshold"
        )

    output_signal = limited[0] if was_mono else limited

    if not return_analysis:
        return output_signal

    analysis: dict[str, float | int | str] = {
        "threshold_db": round(threshold_db, 2),
        "lookahead_ms": round(lookahead_ms, 2),
        "release_ms": round(release_ms, 1),
        "oversample_factor": int(oversample_factor),
        "max_gain_reduction_db": round(max_gr_db, 2),
        "avg_gain_reduction_when_active_db": round(avg_gr_when_active_db, 2),
        "active_gain_reduction_fraction": round(active_fraction, 4),
    }
    if auto_calibrated:
        assert headroom_db is not None
        assert reference_peak_dbfs is not None
        assert effective_percentile is not None
        analysis["calibrated_threshold_db"] = round(threshold_db, 2)
        analysis["reference_peak_dbfs"] = round(reference_peak_dbfs, 2)
        analysis["calibration_percentile"] = round(effective_percentile, 2)
        analysis["headroom_db"] = round(float(headroom_db), 2)
    return output_signal, analysis


_CLIPPER_ALGORITHMS: frozenset[str] = frozenset({"poly_knee", "hard"})


@overload
def apply_clipper(
    signal: np.ndarray,
    *,
    threshold_db: float | np.ndarray = ...,
    knee_width_db: float | np.ndarray = ...,
    algorithm: str = ...,
    oversample_factor: int = ...,
    mix: float | np.ndarray = ...,
    makeup_gain_db: float = ...,
    max_shave_db: float | None = ...,
    calibration_percentile: float = ...,
    calibration_window_ms: float = ...,
    return_analysis: Literal[False] = ...,
) -> np.ndarray: ...


@overload
def apply_clipper(
    signal: np.ndarray,
    *,
    threshold_db: float | np.ndarray = ...,
    knee_width_db: float | np.ndarray = ...,
    algorithm: str = ...,
    oversample_factor: int = ...,
    mix: float | np.ndarray = ...,
    makeup_gain_db: float = ...,
    max_shave_db: float | None = ...,
    calibration_percentile: float = ...,
    calibration_window_ms: float = ...,
    return_analysis: Literal[True],
) -> tuple[np.ndarray, dict[str, float | int | str]]: ...


def apply_clipper(
    signal: np.ndarray,
    *,
    threshold_db: float | np.ndarray = -3.0,
    knee_width_db: float | np.ndarray = 2.0,
    algorithm: str = "poly_knee",
    oversample_factor: int = 8,
    mix: float | np.ndarray = 1.0,
    makeup_gain_db: float = 0.0,
    max_shave_db: float | None = None,
    calibration_percentile: float = 99.0,
    calibration_window_ms: float = 10.0,
    return_analysis: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int | str]]:
    """Oversampled, stereo-linked peak clipper with a monotone polynomial knee.

    Transfer function (per channel, after oversampling):

    * ``algorithm="poly_knee"`` (default): monotone cubic-Hermite soft knee.
      ``|x|`` below ``threshold_db - knee_width_db/2`` passes through
      bit-exact; between ``threshold_db - knee_width_db/2`` and
      ``threshold_db + knee_width_db/2`` the transfer follows a C¹ cubic
      Hermite blend to a flat ceiling; above that it is clamped to
      ``threshold_db`` exactly.  ``knee_width_db=0`` collapses to a pure
      brickwall.  AD2-antialiased.
    * ``algorithm="hard"``: literal ``numpy.clip`` on the oversampled signal.
      ``knee_width_db`` is ignored.  The AD2 hard-clip kernel was measured
      to add <0.01% IMD over naive ``np.clip`` at OS=8 on transient content,
      so we skip ADAA here for speed and predictable null-test behavior.

    This replaces the previous ``hardness`` crossfade of
    ``hard * 0.85 + tanh * 0.15`` which, per the ``clipper_bisect``
    diagnostic, added +64% IMD on a two-tone stimulus and +6.5 dB of
    2-8 kHz brightness on a kick transient vs pure hard-clip at the same
    shave amount.  The polynomial knee fixes that: at
    ``knee_width_db=2.0`` it nulls to naive hard-clip on sample-below-knee
    content and reduces kick 2-8 kHz lift *below* naive hard-clip while
    doing the same peak shave.

    **Stereo linking.** Each channel is shaped on its own signed waveform,
    then a shared attenuation-gain curve ``min(|clipped_i| / |ch_i|)``
    across channels is applied to both.  Keeps clipping HF artifacts
    phase-coherent across L/R.  Mono input bypasses the link and returns
    the per-channel shaped waveform directly.

    Typical use: drum-bus / master-bus peak shaver doing 1-3 dB of real
    work above ``threshold_db``.  Follow with :func:`apply_native_limiter`
    if a strict true-peak ceiling matters — inter-sample peaks can still
    exist after downsampling.

    Parameters
    ----------
    threshold_db : float | np.ndarray
        Ceiling in dBFS (default -3.0, typical range -12..0).  May be a
        per-sample array matching the input length for automated rides.
        Ignored when ``max_shave_db`` is set.
    knee_width_db : float | np.ndarray
        Total knee width in dB.  ``0.0`` is a brickwall; ``2.0`` is a
        mild musical soft-clip; ``6.0`` becomes audibly gentle.  Default
        2.0.  May also be a per-sample array.  Only used by
        ``algorithm="poly_knee"``.
    algorithm : str
        ``"poly_knee"`` (default) or ``"hard"``.  See transfer-function
        description above.
    oversample_factor : int
        1, 2, 4, 8, or 16.  Default 8.  Uses a sharp Kaiser (β=14) polyphase
        resampler on both legs for ~-100 dB stopband.
    mix : float | np.ndarray
        Dry/wet blend, default 1.0 (fully wet).  Per-sample arrays
        supported for automated mix rides.
    makeup_gain_db : float
        Post-clip gain applied to the wet signal before dry/wet blend.
    max_shave_db : float | None
        When set, auto-calibrate a *scalar* ``threshold_db`` so the peak
        shave on this input is approximately ``max_shave_db``.  Incompatible
        with per-sample ``threshold_db`` arrays (use one or the other).
        Default ``None``.
    calibration_percentile : float
        Percentile of windowed peak distribution used for calibration,
        bounded to [50, 100].  Default 99.0.
    calibration_window_ms : float
        Window length in ms for the calibration peak envelope.  Default 10.0.
    """
    if oversample_factor not in (1, 2, 4, 8, 16):
        raise ValueError(
            f"oversample_factor must be 1, 2, 4, 8, or 16; got {oversample_factor}"
        )
    if algorithm not in _CLIPPER_ALGORITHMS:
        raise ValueError(
            f"algorithm must be one of {sorted(_CLIPPER_ALGORITHMS)}; got {algorithm!r}"
        )

    sig = np.asarray(signal, dtype=np.float64)
    was_mono = sig.ndim == 1
    if was_mono:
        sig = sig[np.newaxis, :]
    if sig.shape[-1] == 0:
        return sig[0] if was_mono else sig
    original_length = sig.shape[1]

    # --- Resolve threshold / knee / mix to scalar or per-sample arrays. ---
    #
    # Keep scalar controls scalar.  Long bus/master clippers are usually static,
    # and expanding three scalar controls to full-length oversampled arrays can
    # cost multiple GB before the audio arrays are even counted.
    def _is_array_like(value: float | np.ndarray) -> bool:
        return np.asarray(value).ndim > 0

    def _as_pre_os_array(value: float | np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(original_length, float(arr))
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a scalar or 1-D array")
        if arr.shape[0] != original_length:
            raise ValueError(
                f"{name} array length {arr.shape[0]} must match input length "
                f"{original_length}"
            )
        return arr

    threshold_is_array = _is_array_like(threshold_db)
    knee_is_array = _is_array_like(knee_width_db)
    mix_is_array = _is_array_like(mix)

    threshold_db_scalar = 0.0
    knee_width_db_scalar = 0.0
    mix_scalar = 0.0
    threshold_db_arr: np.ndarray | None = None
    knee_width_db_arr: np.ndarray | None = None
    mix_arr: np.ndarray | None = None

    if threshold_is_array:
        threshold_db_arr = _as_pre_os_array(threshold_db, "threshold_db")
    else:
        threshold_db_scalar = float(np.asarray(threshold_db, dtype=np.float64).item())

    if knee_is_array:
        knee_width_db_arr = np.maximum(
            _as_pre_os_array(knee_width_db, "knee_width_db"), 0.0
        )
    else:
        knee_width_db_scalar = max(
            float(np.asarray(knee_width_db, dtype=np.float64).item()), 0.0
        )

    if mix_is_array:
        mix_arr = np.clip(_as_pre_os_array(mix, "mix"), 0.0, 1.0)
    else:
        mix_scalar = float(
            np.clip(float(np.asarray(mix, dtype=np.float64).item()), 0.0, 1.0)
        )

    # --- Auto-calibration via max_shave_db (scalar threshold only). ---
    auto_calibrated = max_shave_db is not None
    reference_peak_dbfs: float | None = None
    effective_percentile: float | None = None
    if auto_calibrated:
        assert max_shave_db is not None
        if threshold_is_array:
            raise ValueError(
                "max_shave_db is incompatible with a per-sample threshold_db array; "
                "pick one calibration mode"
            )
        if threshold_db_scalar != -3.0:
            logger.warning(
                "Clipper: both threshold_db and max_shave_db set — "
                "max_shave_db wins; threshold_db ignored."
            )
        effective_percentile = float(np.clip(calibration_percentile, 50.0, 100.0))
        window_samples = max(1, int(round(calibration_window_ms * 1e-3 * SAMPLE_RATE)))
        peak_track = np.max(np.abs(sig), axis=0)
        n = peak_track.shape[0]
        if n < window_samples:
            reference_peak = float(np.max(peak_track)) if n > 0 else 1e-12
        else:
            n_windows = n // window_samples
            trimmed = peak_track[: n_windows * window_samples]
            windowed = trimmed.reshape(n_windows, window_samples)
            window_peaks = np.max(windowed, axis=1)
            reference_peak = float(np.percentile(window_peaks, effective_percentile))
        reference_peak = max(reference_peak, 1e-12)
        reference_peak_dbfs = float(20.0 * np.log10(reference_peak))
        calibrated_db = reference_peak_dbfs - float(max_shave_db)
        threshold_db_scalar = calibrated_db
        logger.info(
            f"Clipper: auto-calibrated threshold to {calibrated_db:.2f} dBFS "
            f"(p{effective_percentile:g} peak {reference_peak_dbfs:.2f} dBFS, "
            f"target shave {max_shave_db:.2f} dB)"
        )

    # Mix=0 at every sample is a pure bypass; skip the OS round-trip.
    if (mix_arr is None and mix_scalar <= 0.0) or (
        mix_arr is not None and np.all(mix_arr <= 0.0)
    ):
        return sig[0].copy() if was_mono else sig.copy()

    makeup_lin = db_to_amp(makeup_gain_db)

    # Sharp Kaiser polyphase: β=14 gives ~-100 dB stopband, vs scipy's default
    # β=5 at ~-50 dB.  The lower stopband leaked folded alias energy back
    # into the audio band — audible as papery HF residue on drum content.
    resample_window = ("kaiser", 14.0)
    effective_sr_hz = float(SAMPLE_RATE * oversample_factor)

    def _upsample(ch: np.ndarray) -> np.ndarray:
        if oversample_factor > 1:
            return np.asarray(
                resample_poly(ch, oversample_factor, 1, window=resample_window),
                dtype=np.float64,
            )
        return ch.astype(np.float64, copy=False)

    def _downsample_to_length(ch: np.ndarray, target_length: int) -> np.ndarray:
        if oversample_factor > 1:
            out = np.asarray(
                resample_poly(ch, 1, oversample_factor, window=resample_window),
                dtype=np.float64,
            )
        else:
            out = ch
        if out.shape[0] > target_length:
            return out[:target_length]
        if out.shape[0] < target_length:
            return np.concatenate([out, np.zeros(target_length - out.shape[0])])
        return out

    use_scalar_shape_controls = threshold_db_arr is None and knee_width_db_arr is None
    threshold_lin_scalar = db_to_amp(threshold_db_scalar)
    knee_half_lin_scalar = threshold_lin_scalar * (
        1.0 - 10.0 ** (-knee_width_db_scalar * 0.5 / 20.0)
    )

    threshold_lin_up: np.ndarray | None = None
    knee_half_lin_up: np.ndarray | None = None
    if not use_scalar_shape_controls:
        if threshold_db_arr is None:
            threshold_db_arr = np.full(original_length, threshold_db_scalar)
        if knee_width_db_arr is None:
            knee_width_db_arr = np.full(original_length, knee_width_db_scalar)
        threshold_db_profile = cast(np.ndarray, threshold_db_arr)
        knee_width_db_profile = cast(np.ndarray, knee_width_db_arr)
        # Lift threshold / knee profiles into the OS domain.  We upsample them
        # the same way we upsample the signal — this is a mild-Gibbs concern
        # at sharp control-rate edges, but for the smooth control curves that
        # AutomationSpec produces it's faithful.  Knee stays positive.
        threshold_db_up = (
            _upsample(threshold_db_profile)
            if oversample_factor > 1
            else threshold_db_profile
        )
        knee_width_db_up = (
            _upsample(knee_width_db_profile)
            if oversample_factor > 1
            else knee_width_db_profile
        )
        # Array-safe dB-to-linear (db_to_amp is scalar-only).
        threshold_lin_up = np.power(10.0, threshold_db_up / 20.0)
        # knee_width_db is total knee width; knee_half_lin is half-width in linear.
        # knee_half_lin = threshold_lin * (1 - 10^(-knee_width_db/2 / 20))
        knee_half_lin_up = threshold_lin_up * (
            1.0 - np.power(10.0, -np.maximum(knee_width_db_up, 0.0) * 0.5 / 20.0)
        )

    def _clip_signed(ch_up: np.ndarray) -> np.ndarray:
        """Apply the selected transfer to one oversampled signed channel."""
        if use_scalar_shape_controls:
            if algorithm == "hard":
                return np.clip(ch_up, -threshold_lin_scalar, threshold_lin_scalar)
            return _apply_adaa2_poly_knee_scalar(
                ch_up,
                threshold_lin_scalar,
                knee_half_lin_scalar,
                effective_sr_hz,
            )
        assert threshold_lin_up is not None
        assert knee_half_lin_up is not None
        if algorithm == "hard":
            return np.clip(ch_up, -threshold_lin_up, threshold_lin_up)
        return _apply_adaa2_poly_knee(
            ch_up, threshold_lin_up, knee_half_lin_up, effective_sr_hz
        )

    # Per-channel shape at OS.  For stereo we build a common attenuation
    # gain curve and apply it to both — preserves phase, kills papery
    # widening from uncorrelated per-channel HF residue.
    if sig.shape[0] == 1:
        ch_up = _upsample(sig[0])
        clipped = _clip_signed(ch_up)
        wet_channels = [_downsample_to_length(clipped, sig.shape[1])]
    else:
        # Silence floor tracks the current threshold per-sample.  At samples
        # where the input channel is well below threshold the clipper is a
        # no-op, so we force the linked gain to 1.0 there rather than
        # letting tiny divisions dominate.
        silence_floor_up: float | np.ndarray
        if use_scalar_shape_controls:
            silence_floor_up = threshold_lin_scalar * 1e-4
        else:
            assert threshold_lin_up is not None
            silence_floor_up = threshold_lin_up * 1e-4
        estimated_upsampled_samples = sig.shape[1] * oversample_factor
        recompute_upsampled_channels = (
            oversample_factor > 1
            and estimated_upsampled_samples * sig.shape[0] >= 20_000_000
        )
        upsampled_channels = (
            [] if recompute_upsampled_channels else [_upsample(ch) for ch in sig]
        )
        gain_curve: np.ndarray | None = None
        gain_input_channels = (
            sig if recompute_upsampled_channels else upsampled_channels
        )
        for channel in gain_input_channels:
            ch_up = _upsample(channel) if recompute_upsampled_channels else channel
            clipped_up = _clip_signed(ch_up)
            abs_in = np.abs(ch_up)
            safe_abs_in = np.maximum(abs_in, silence_floor_up)
            g = np.clip(np.abs(clipped_up) / safe_abs_in, 0.0, 1.0)
            g = np.where(abs_in < silence_floor_up, 1.0, g)
            if gain_curve is None:
                gain_curve = g
            else:
                np.minimum(gain_curve, g, out=gain_curve)

        assert gain_curve is not None
        wet_channels = []
        wet_input_channels = sig if recompute_upsampled_channels else upsampled_channels
        for channel in wet_input_channels:
            ch_up = _upsample(channel) if recompute_upsampled_channels else channel
            attenuated = ch_up * gain_curve
            wet_channels.append(_downsample_to_length(attenuated, sig.shape[1]))

    wet = np.stack(wet_channels)
    wet = wet * makeup_lin

    # Dry/wet blend with per-sample mix.
    if mix_arr is None:
        result = (
            wet if mix_scalar >= 1.0 else (1.0 - mix_scalar) * sig + mix_scalar * wet
        )
    else:
        result = (
            wet if np.all(mix_arr >= 1.0) else (1.0 - mix_arr) * sig + mix_arr * wet
        )

    if result.shape[1] != original_length:
        if result.shape[1] > original_length:
            result = result[:, :original_length]
        else:
            pad = original_length - result.shape[1]
            result = np.pad(result, ((0, 0), (0, pad)))

    peak_before_db = float(20.0 * np.log10(max(np.max(np.abs(sig)), 1e-12)))
    peak_after_db = float(20.0 * np.log10(max(np.max(np.abs(result)), 1e-12)))
    shaved_db = peak_before_db - peak_after_db
    if use_scalar_shape_controls:
        active_fraction = float(
            np.mean(np.max(np.abs(sig), axis=0) >= threshold_lin_scalar)
        )
        threshold_summary_db = threshold_db_scalar
        knee_summary_db = knee_width_db_scalar
    else:
        assert threshold_lin_up is not None
        threshold_for_input_rate = (
            threshold_lin_up[::oversample_factor][:original_length]
            if oversample_factor > 1
            else threshold_lin_up
        )
        active_fraction = float(
            np.mean(np.max(np.abs(sig), axis=0) >= threshold_for_input_rate)
        )
        assert threshold_db_arr is not None
        assert knee_width_db_arr is not None
        threshold_summary_db = float(np.mean(threshold_db_arr))
        knee_summary_db = float(np.mean(knee_width_db_arr))
    mix_summary = mix_scalar if mix_arr is None else float(np.mean(mix_arr))
    logger.info(
        f"Clipper: algorithm={algorithm}, threshold {threshold_summary_db:.1f} dB, "
        f"knee {knee_summary_db:.2f} dB, OS {oversample_factor}x, "
        f"shaved {shaved_db:.2f} dB (peak {peak_before_db:.2f} -> {peak_after_db:.2f} dBFS)"
    )

    output_signal = result[0] if was_mono else result

    if not return_analysis:
        return output_signal

    analysis: dict[str, float | int | str] = {
        "threshold_db": round(threshold_summary_db, 2),
        "knee_width_db": round(knee_summary_db, 3),
        "algorithm": algorithm,
        "oversample_factor": int(oversample_factor),
        "mix": round(mix_summary, 3),
        "makeup_gain_db": round(makeup_gain_db, 2),
        "shaved_db": round(shaved_db, 2),
        "active_fraction": round(active_fraction, 4),
    }
    if auto_calibrated:
        assert max_shave_db is not None
        assert reference_peak_dbfs is not None
        assert effective_percentile is not None
        analysis["calibrated_threshold_db"] = round(threshold_summary_db, 2)
        analysis["reference_peak_dbfs"] = round(reference_peak_dbfs, 2)
        analysis["calibration_percentile"] = round(effective_percentile, 2)
        analysis["max_shave_db"] = round(float(max_shave_db), 2)
    return output_signal, analysis


def finalize_master(
    signal: np.ndarray,
    *,
    sample_rate: int,
    target_lufs: float = -18.0,
    true_peak_ceiling_dbfs: float = -0.5,
    oversample_factor: int = 4,
    max_iterations: int = 6,
    loudness_tolerance_lufs: float = 0.2,
) -> MasteringResult:
    """Finalize a mix to a LUFS target with an LSP true-peak limiter ceiling."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    source_signal = np.asarray(signal, dtype=np.float64)
    mastered = source_signal
    if mastered.size == 0:
        return MasteringResult(
            signal=mastered,
            integrated_lufs=float("-inf"),
            true_peak_dbfs=float("-inf"),
        )

    use_lsp_limiter = has_external_plugin("lsp_limiter_stereo")
    if not use_lsp_limiter:
        logger.info(
            "LSP limiter unavailable — using native lookahead limiter fallback."
        )

    current_lufs, active_window_fraction = integrated_lufs(
        mastered,
        sample_rate=sample_rate,
    )
    if not np.isfinite(current_lufs) or active_window_fraction <= 0.0:
        return MasteringResult(
            signal=mastered,
            integrated_lufs=current_lufs,
            true_peak_dbfs=amp_to_db(
                max(
                    estimate_true_peak_amplitude(
                        mastered,
                        oversample_factor=oversample_factor,
                    ),
                    1e-12,
                )
            ),
        )

    limiter_input_gain_db = target_lufs - current_lufs
    logger.info(
        f"Mastering: input {current_lufs:.1f} LUFS, "
        f"target {target_lufs:.1f} LUFS, {limiter_input_gain_db=:.2f}, "
        f"limiter={'LSP' if use_lsp_limiter else 'native'}"
    )
    limiter_output_gain_db = 0.0

    def _dispatch_limiter(source: np.ndarray, input_gain_db: float) -> np.ndarray:
        if use_lsp_limiter:
            return apply_lsp_limiter(
                source,
                threshold_db=true_peak_ceiling_dbfs,
                input_gain_db=input_gain_db,
                output_gain_db=limiter_output_gain_db,
            )
        return apply_native_limiter(
            source,
            threshold_db=true_peak_ceiling_dbfs,
            input_gain_db=input_gain_db,
            sample_rate=sample_rate,
        )

    mastered = _dispatch_limiter(source_signal, limiter_input_gain_db)

    previous_lufs: float | None = None
    for _ in range(max_iterations):
        # Measure LUFS directly on the limiter's own output — do NOT
        # re-normalize to the true-peak ceiling here.  Re-normalizing every
        # iteration (as the LSP path used to do) pins the signal's peak to
        # the ceiling regardless of the limiter's input gain, which makes
        # measured LUFS nearly insensitive to `limiter_input_gain_db` and
        # prevents the loop from ever converging on `target_lufs`. The
        # true-peak ceiling is still guaranteed by the final safety pass
        # below, after the loop has converged on loudness.
        current_lufs, active_window_fraction = integrated_lufs(
            mastered,
            sample_rate=sample_rate,
        )
        if not np.isfinite(current_lufs) or active_window_fraction <= 0.0:
            break

        loudness_error = target_lufs - current_lufs
        if abs(loudness_error) <= loudness_tolerance_lufs:
            break
        if previous_lufs is not None and abs(current_lufs - previous_lufs) < 1e-6:
            # Gain changes are no longer moving measured LUFS (e.g. the
            # limiter is fully saturating regardless of input gain) — stop
            # iterating rather than walking gain indefinitely.
            break
        previous_lufs = current_lufs

        logger.info(
            f"Mastering iteration: LUFS error {loudness_error:+.2f} LU, "
            f"new gain {limiter_input_gain_db + loudness_error:.2f} dB"
        )
        limiter_input_gain_db += loudness_error
        mastered = _dispatch_limiter(source_signal, limiter_input_gain_db)

    # Final true-peak safety pass.  For the native limiter path, only
    # attenuate — the limiter already guarantees the ceiling, and boosting a
    # high-crest-factor signal defeats LUFS convergence.
    current_true_peak = estimate_true_peak_amplitude(
        mastered, oversample_factor=oversample_factor
    )
    ceiling_amp = db_to_amp(true_peak_ceiling_dbfs)
    if current_true_peak > ceiling_amp:
        mastered = mastered * (ceiling_amp / current_true_peak)
    elif use_lsp_limiter:
        mastered = normalize_true_peak(
            mastered,
            target_peak_dbfs=true_peak_ceiling_dbfs,
            oversample_factor=oversample_factor,
        )
    final_lufs, _ = integrated_lufs(
        mastered,
        sample_rate=sample_rate,
    )
    final_true_peak_dbfs = amp_to_db(
        max(
            estimate_true_peak_amplitude(
                mastered,
                oversample_factor=oversample_factor,
            ),
            1e-12,
        )
    )
    logger.info(
        f"Mastering final: {final_lufs:.1f} LUFS, "
        f"true peak {final_true_peak_dbfs:.2f} dBFS, "
        f"gain {limiter_input_gain_db:.2f} dB"
    )
    return MasteringResult(
        signal=mastered,
        integrated_lufs=final_lufs,
        true_peak_dbfs=final_true_peak_dbfs,
    )


_CHORUS_PRESETS: dict[str, dict[str, float]] = {
    "juno_subtle": {
        "mix": 0.28,
        "rate_hz": 0.32,
        "depth_ms": 2.4,
        "center_delay_ms": 13.5,
        "stereo_phase_deg": 115.0,
        "feedback": 0.04,
        "wet_lowpass_hz": 6_000.0,
        "wet_highpass_hz": 160.0,
        "drift_amount": 0.12,
        "wet_saturation": 0.06,
    },
    "juno_wide": {
        "mix": 0.33,
        "rate_hz": 0.42,
        "depth_ms": 3.3,
        "center_delay_ms": 14.5,
        "stereo_phase_deg": 130.0,
        "feedback": 0.07,
        "wet_lowpass_hz": 5_600.0,
        "wet_highpass_hz": 170.0,
        "drift_amount": 0.16,
        "wet_saturation": 0.08,
    },
    "ensemble_soft": {
        "mix": 0.30,
        "rate_hz": 0.24,
        "depth_ms": 4.1,
        "center_delay_ms": 16.0,
        "stereo_phase_deg": 95.0,
        "feedback": 0.06,
        "wet_lowpass_hz": 5_200.0,
        "wet_highpass_hz": 150.0,
        "drift_amount": 0.18,
        "wet_saturation": 0.10,
    },
}

# ``_DRIVE_PRESETS`` was retired in the tube-saturation redesign. Preset
# intents migrated as follows: ``tube_warm`` -> ``apply_tube("triode_glow")``;
# ``neve_gentle`` -> ``apply_preamp("neve_warmth")``; ``iron_soft`` ->
# ``apply_preamp("iron_color")``; ``kick_heavy`` / ``snare_bite`` ->
# ``apply_transistor`` presets of the same name (drum saturation is
# op-amp / diode territory, not tube).

_AIRWINDOWS_PRESETS: dict[str, dict[str, Any]] = {
    "density_glue": {
        "algorithm": "Density",
        "density": 1.2,
        "highpass": 0.35,
        "out_level": 0.5,
        "dry_wet": 0.6,
    },
    "iron_warmth": {
        "algorithm": "IronOxide5",
        "input_trim": 0.0,
        "tape_high": 0.5,
        "tape_low": 0.5,
        "flutter": 0.2,
        "noise": 0.0,
    },
    "tape_subtle": {
        "algorithm": "ToTape6",
        "input": 0.0,
        "soften": 0.6,
        "head_b": 0.5,
        "flutter": 0.15,
        "output": 0.0,
        "dry_wet": 0.7,
    },
    "tube_warmth": {
        "algorithm": "Tube",
        "tube": 0.4,
    },
    "coils_xformer": {
        "algorithm": "Coils",
        "saturat": 0.35,
        "core_dc": 0.0,
        "dry_wet": 0.7,
    },
    "channel_ssl": {
        "algorithm": "Channel9",
        "console_type": "SSL",
        "drive": 80.0,
        "output": 0.5,
    },
    "drive_gentle": {
        "algorithm": "Drive",
        "drive": 25.0,
        "highpass": 0.3,
        "out_level": 0.5,
        "dry_wet": 0.5,
    },
}

_BYOD_PRESETS: dict[str, dict[str, Any]] = {
    "tube_screamer": {"program": "Tube Screamer"},
    "centaur": {"program": "Centaur"},
    "american": {"program": "American Sound"},
    "zen_drive": {"program": "ZenDrive"},
    "king_of_tone": {"program": "King Of Tone"},
}

_CHOW_CENTAUR_PRESETS: dict[str, dict[str, Any]] = {
    "subtle_warmth": {"gain": 0.2, "treble": 0.45, "level": 0.75, "mode": "Neural"},
    "light_edge": {"gain": 0.4, "treble": 0.55, "level": 0.65, "mode": "Neural"},
    "traditional_clean": {
        "gain": 0.25,
        "treble": 0.5,
        "level": 0.7,
        "mode": "Traditional",
    },
}

_COMPRESSOR_PRESETS: dict[str, dict[str, float | str | list[dict[str, Any]]]] = {
    "kick_glue": {
        # Calibrated for kicks peaking around -6 dBFS (typical with normalize_lufs=None
        # and a moderate amp_db).  With body_decay ~300 ms the tail drops below -13 dBFS
        # around 241 ms after the hit, leaving ~220 ms of release window at 130 BPM.
        # Delivers ~4 dB GR at peak; use kick_punch if you need more bite.
        "threshold_db": -13.0,
        "ratio": 2.4,
        "attack_ms": 12.0,
        "release_ms": 160.0,
        "knee_db": 5.0,
        "makeup_gain_db": 0.75,
        "mix": 0.82,
        "topology": "feedforward",
        "detector_mode": "rms",
        "detector_bands": [
            {"kind": "highpass", "cutoff_hz": 45.0, "slope_db_per_oct": 12}
        ],
    },
    "kick_punch": {
        # Same input-level assumption as kick_glue (~-6 dBFS peak).  Threshold set
        # so the tail drops below it around 276 ms after the hit, leaving ~185 ms
        # of release time — safely clears at up to ~145 BPM with the 140 ms release.
        # Delivers ~6 dB GR at the transient peak.
        "threshold_db": -14.0,
        "ratio": 3.6,
        "attack_ms": 6.0,
        "release_ms": 140.0,
        "knee_db": 4.0,
        "makeup_gain_db": 1.0,
        "mix": 0.92,
        "topology": "feedforward",
        "detector_mode": "peak",
        "detector_bands": [
            {"kind": "highpass", "cutoff_hz": 55.0, "slope_db_per_oct": 12}
        ],
    },
    "tom_control": {
        # Toms have longer body decays (420–520 ms) and are usually hit less densely
        # than kicks, so continuous compression in dense fills is acceptable.
        # Delivers ~5 dB GR at peak for a tom at -6 dBFS; the longer release tail
        # lets the gain ride down more musically than a single-stage release.
        "threshold_db": -14.0,
        "ratio": 2.2,
        "attack_ms": 14.0,
        "release_ms": 220.0,
        "release_tail_ms": 420.0,
        "knee_db": 6.0,
        "makeup_gain_db": 0.6,
        "mix": 0.74,
        "topology": "feedback",
        "detector_mode": "rms",
        "detector_bands": [
            {"kind": "highpass", "cutoff_hz": 60.0, "slope_db_per_oct": 12}
        ],
    },
    "kick_duck": {
        # Surgical kick-sidechain ducking for non-kick voices.
        # 1 ms lookahead + 1 ms attack = near-instant onset, no transient bleed.
        # 100 ms release clears well before the next 16th at 130 BPM (115 ms).
        # No makeup gain — this is a duck, not a leveller.
        "threshold_db": -20.0,
        "ratio": 3.0,
        "attack_ms": 1.0,
        "release_ms": 100.0,
        "knee_db": 2.0,
        "makeup_gain_db": 0.0,
        "mix": 1.0,
        "topology": "feedforward",
        "detector_mode": "peak",
        "lookahead_ms": 1.0,
    },
    "kick_duck_hard": {
        # Aggressive pumping duck — longer 300 ms release creates audible swell-back.
        # Useful for the classic techno "breathing" effect.
        "threshold_db": -20.0,
        "ratio": 4.0,
        "attack_ms": 1.0,
        "release_ms": 300.0,
        "knee_db": 2.0,
        "makeup_gain_db": 0.0,
        "mix": 1.0,
        "topology": "feedforward",
        "detector_mode": "peak",
        "lookahead_ms": 1.0,
    },
    "snare_punch": {
        # Fast-attack transient control — lets the initial crack through then clamps.
        # ~5 dB GR at peak for a snare at -6 dBFS; detector HP keeps low bleed out.
        "threshold_db": -12.0,
        "ratio": 3.0,
        "attack_ms": 4.0,
        "release_ms": 120.0,
        "knee_db": 4.0,
        "makeup_gain_db": 2.5,
        "mix": 0.88,
        "topology": "feedforward",
        "detector_mode": "peak",
        "detector_bands": [
            {"kind": "highpass", "cutoff_hz": 100.0, "slope_db_per_oct": 12}
        ],
    },
    "snare_body": {
        # Slower attack lets the transient fully through; RMS detection smooths
        # the body sustain for a fatter, more even tail.
        "threshold_db": -16.0,
        "ratio": 2.0,
        "attack_ms": 18.0,
        "release_ms": 200.0,
        "knee_db": 6.0,
        "makeup_gain_db": 1.5,
        "mix": 0.75,
        "topology": "feedback",
        "detector_mode": "rms",
        "detector_bands": [
            {"kind": "highpass", "cutoff_hz": 80.0, "slope_db_per_oct": 12}
        ],
    },
    "hat_control": {
        # Very fast attack tames hi-hat spikes; short release avoids pumping
        # on rapid 16th-note patterns.
        "threshold_db": -16.0,
        "ratio": 2.5,
        "attack_ms": 2.0,
        "release_ms": 60.0,
        "knee_db": 3.0,
        "makeup_gain_db": 1.0,
        "mix": 0.85,
        "topology": "feedforward",
        "detector_mode": "peak",
        "detector_bands": [
            {"kind": "highpass", "cutoff_hz": 200.0, "slope_db_per_oct": 12}
        ],
    },
    "master_glue": {
        "threshold_db": -26.0,
        "ratio": 2.0,
        "attack_ms": 30.0,
        "release_ms": 200.0,
        "knee_db": 6.0,
        "topology": "feedback",
        "detector_mode": "rms",
        "detector_bands": [
            {"kind": "highpass", "cutoff_hz": 80.0, "slope_db_per_oct": 12}
        ],
    },
}


def _resolve_effect_params(
    effect_kind: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Resolve effect presets with explicit parameters taking precedence."""
    resolved_params = dict(params)
    preset = resolved_params.pop("preset", None)
    if preset is None:
        return resolved_params

    preset_map: dict[str, dict[str, Any]]
    if effect_kind == "chorus":
        preset_map = _CHORUS_PRESETS
    elif effect_kind == "bbd_chorus":
        preset_map = _BBD_CHORUS_PRESETS
    elif effect_kind == "compressor":
        preset_map = _COMPRESSOR_PRESETS
    elif effect_kind == "phaser":
        preset_map = _PHASER_PRESETS
    elif effect_kind == "mod_delay":
        preset_map = _MOD_DELAY_PRESETS
    elif effect_kind == "airwindows":
        preset_map = _AIRWINDOWS_PRESETS
    elif effect_kind == "byod":
        preset_map = _BYOD_PRESETS
    elif effect_kind == "chow_centaur":
        preset_map = _CHOW_CENTAUR_PRESETS
    elif effect_kind == "preamp":
        preset_map = _PREAMP_PRESETS
    elif effect_kind == "tube":
        preset_map = _TUBE_PRESETS
    elif effect_kind == "transistor":
        preset_map = _TRANSISTOR_PRESETS
    else:
        raise ValueError(f"Unsupported preset-bearing effect kind: {effect_kind}")

    if preset not in preset_map:
        raise ValueError(f"Unsupported {effect_kind} preset: {preset!r}")

    preset_params = dict(preset_map[preset])
    preset_params.update(resolved_params)
    return preset_params


def apply_chorus(
    signal: np.ndarray,
    mix: float = 0.28,
    rate_hz: float = 0.32,
    depth_ms: float = 2.4,
    center_delay_ms: float = 13.5,
    stereo_phase_deg: float = 115.0,
    feedback: float = 0.04,
    wet_lowpass_hz: float = 6_000.0,
    wet_highpass_hz: float = 160.0,
    drift_amount: float = 0.12,
    wet_saturation: float = 0.06,
) -> np.ndarray:
    """Apply a warm, Juno-inspired stereo chorus."""
    if not 0.0 <= mix <= 1.0:
        raise ValueError("mix must be between 0 and 1")
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")
    if depth_ms < 0 or center_delay_ms <= 0:
        raise ValueError("depth_ms must be non-negative and center_delay_ms positive")

    dry_signal = _ensure_stereo(signal)
    sample_positions = np.arange(dry_signal.shape[-1], dtype=np.float64)
    base_phase = 2.0 * np.pi * rate_hz * sample_positions / SAMPLE_RATE
    drift_phase = 2.0 * np.pi * rate_hz * 0.37 * sample_positions / SAMPLE_RATE
    stereo_phase = np.deg2rad(stereo_phase_deg)

    delayed_channels: list[np.ndarray] = []
    for channel_index, channel_signal in enumerate(dry_signal):
        channel_phase = base_phase + stereo_phase * channel_index
        modulation = np.sin(channel_phase)
        if drift_amount > 0:
            modulation += drift_amount * np.sin(drift_phase + (0.7 * channel_index))
        delay_ms = center_delay_ms + depth_ms * modulation
        delay_samples = np.maximum(delay_ms, 0.0) * SAMPLE_RATE / 1_000.0
        delayed = _fractional_delay(channel_signal, delay_samples)
        if feedback > 0:
            delayed = delayed + feedback * _fractional_delay(
                delayed, delay_samples * 0.5
            )
        delayed_channels.append(delayed)

    wet_signal = np.stack(delayed_channels)
    wet_signal = _apply_per_channel(
        wet_signal,
        lambda channel: highpass(
            channel, cutoff_hz=wet_highpass_hz, sample_rate=SAMPLE_RATE, order=2
        ),
    )
    wet_signal = _apply_per_channel(
        wet_signal,
        lambda channel: lowpass(
            channel, cutoff_hz=wet_lowpass_hz, sample_rate=SAMPLE_RATE, order=2
        ),
    )
    if wet_saturation > 0:
        wet_signal = wet_signal + wet_saturation * np.tanh(1.6 * wet_signal)

    blended = ((1.0 - mix) * dry_signal) + (mix * wet_signal)
    return blended.astype(np.float64)


# ---------------------------------------------------------------------------
# BBD chorus (native, Juno/Dimension-D inspired)
# ---------------------------------------------------------------------------

# Preset parameters are grounded in Juno-106 service-manual BBD clocks and the
# published Dimension-D topology rather than copying proprietary presets. The
# DSP (quadrature LFOs, BBD-style pre/post bandlimiting, gentle compander,
# cross-feedback) is standard textbook technique.

_BBD_CHORUS_PRESETS: dict[str, dict[str, float]] = {
    "juno_i": {
        "mix": 0.30,
        "rate_hz": 0.51,
        "depth_ms": 1.5,
        "center_delay_ms": 3.2,
        "cross_feedback": 0.08,
        "compander_amount": 0.20,
        "pre_lowpass_hz": 6_500.0,
        "wet_lowpass_hz": 6_000.0,
        "wet_highpass_hz": 120.0,
        "stack_count": 1.0,
    },
    "juno_ii": {
        "mix": 0.35,
        "rate_hz": 0.83,
        "depth_ms": 2.8,
        "center_delay_ms": 4.4,
        "cross_feedback": 0.20,
        "compander_amount": 0.25,
        "pre_lowpass_hz": 6_500.0,
        "wet_lowpass_hz": 6_000.0,
        "wet_highpass_hz": 120.0,
        "stack_count": 1.0,
    },
    "juno_i_plus_ii": {
        # Both sections stacked, staggered LFO phases for denser motion.
        "mix": 0.40,
        "rate_hz": 0.67,
        "depth_ms": 2.2,
        "center_delay_ms": 3.8,
        "cross_feedback": 0.15,
        "compander_amount": 0.24,
        "pre_lowpass_hz": 6_500.0,
        "wet_lowpass_hz": 6_000.0,
        "wet_highpass_hz": 120.0,
        "stack_count": 2.0,
    },
    "dimension_wide": {
        # Dimension-D territory: longer delays, deeper modulation, slower rate.
        "mix": 0.45,
        "rate_hz": 0.3,
        "depth_ms": 5.0,
        "center_delay_ms": 10.0,
        "cross_feedback": 0.22,
        "compander_amount": 0.18,
        "pre_lowpass_hz": 5_800.0,
        "wet_lowpass_hz": 5_500.0,
        "wet_highpass_hz": 110.0,
        "stack_count": 1.0,
    },
}


@numba.njit(cache=True)
def _bbd_chorus_stereo_loop(
    input_left: np.ndarray,
    input_right: np.ndarray,
    buffer_size: int,
    delay_base_samples: float,
    mod_depth_samples: float,
    lfo_omega_per_sample: float,
    stage_phases_left: np.ndarray,
    stage_phases_right: np.ndarray,
    cross_feedback: float,
    compander_k: float,
    stage_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Stereo BBD delay with quadrature LFOs, cross-feedback, and compander.

    Walks the delay lines sample-by-sample so cross-feedback stays causal.
    Uses 3-point Lagrange interpolation for the fractional read, which is
    cheap and clean enough to avoid the "digital zipper" artifacts linear
    interpolation creates when the read position is continuously moving.

    Supports an arbitrary number of stacked stages in a single pass: each
    stage has its own pair of LFO phases (``stage_phases_left`` /
    ``stage_phases_right``) and its own pair of delay-line buffers. This
    avoids paying the numba dispatch + buffer-allocation cost once per
    stage when using Juno I+II-style stacked presets.
    """
    n_samples = input_left.shape[0]
    n_stages = stage_phases_left.shape[0]
    out_left = np.zeros(n_samples, dtype=np.float64)
    out_right = np.zeros(n_samples, dtype=np.float64)
    # One pair of delay-line buffers per stage. 2D layout keeps things
    # numba-friendly (no nested Python lists inside the JIT loop).
    buffers_left = np.zeros((n_stages, buffer_size), dtype=np.float64)
    buffers_right = np.zeros((n_stages, buffer_size), dtype=np.float64)
    prev_out_left = np.zeros(n_stages, dtype=np.float64)
    prev_out_right = np.zeros(n_stages, dtype=np.float64)
    write_pos = 0

    # Guardrails on the delay read position: never less than 1 sample,
    # and leave a 2-sample margin before the far edge so Lagrange has
    # neighbors to interpolate from.
    min_delay = 1.0
    max_delay = float(buffer_size - 3)

    for i in range(n_samples):
        dry_sample_left = input_left[i]
        dry_sample_right = input_right[i]
        sample_omega = lfo_omega_per_sample * i
        for s in range(n_stages):
            # Quadrature LFOs (bipolar sine, phase-offset per channel).
            phase_left = sample_omega + stage_phases_left[s]
            phase_right = sample_omega + stage_phases_right[s]
            delay_left = delay_base_samples + mod_depth_samples * math.sin(phase_left)
            delay_right = delay_base_samples + mod_depth_samples * math.sin(phase_right)
            if delay_left < min_delay:
                delay_left = min_delay
            elif delay_left > max_delay:
                delay_left = max_delay
            if delay_right < min_delay:
                delay_right = min_delay
            elif delay_right > max_delay:
                delay_right = max_delay

            # 3-point Lagrange read for left channel.
            read_pos_l = write_pos - delay_left
            if read_pos_l < 0.0:
                read_pos_l += buffer_size
            base_l = int(read_pos_l)
            frac_l = read_pos_l - base_l
            il_minus = (base_l - 1) % buffer_size
            il_zero = base_l % buffer_size
            il_plus = (base_l + 1) % buffer_size
            wl_minus = 0.5 * frac_l * (frac_l - 1.0)
            wl_zero = (1.0 - frac_l) * (1.0 + frac_l)
            wl_plus = 0.5 * frac_l * (frac_l + 1.0)
            delayed_l = (
                wl_minus * buffers_left[s, il_minus]
                + wl_zero * buffers_left[s, il_zero]
                + wl_plus * buffers_left[s, il_plus]
            )

            # 3-point Lagrange read for right channel.
            read_pos_r = write_pos - delay_right
            if read_pos_r < 0.0:
                read_pos_r += buffer_size
            base_r = int(read_pos_r)
            frac_r = read_pos_r - base_r
            ir_minus = (base_r - 1) % buffer_size
            ir_zero = base_r % buffer_size
            ir_plus = (base_r + 1) % buffer_size
            wr_minus = 0.5 * frac_r * (frac_r - 1.0)
            wr_zero = (1.0 - frac_r) * (1.0 + frac_r)
            wr_plus = 0.5 * frac_r * (frac_r + 1.0)
            delayed_r = (
                wr_minus * buffers_right[s, ir_minus]
                + wr_zero * buffers_right[s, ir_zero]
                + wr_plus * buffers_right[s, ir_plus]
            )

            # Optional compander: tanh(k*x)/k gives gentle soft-limiting that
            # mimics BBD input/output companding without a full expander pair.
            # compander_k=0 is pure bypass (identity).
            if compander_k > 0.0:
                delayed_l = math.tanh(compander_k * delayed_l) / compander_k
                delayed_r = math.tanh(compander_k * delayed_r) / compander_k

            # Accumulate this stage's wet contribution (pre-scaled so the
            # summed multi-stage output keeps its energy comparable to a
            # single-stage pass).
            out_left[i] += stage_scale * delayed_l
            out_right[i] += stage_scale * delayed_r

            # Write into buffers AFTER reading. Cross-feedback comes from
            # the previous sample of the opposite channel on THIS stage,
            # guaranteeing stability and keeping stages independent.
            buffers_left[s, write_pos] = (
                dry_sample_left + cross_feedback * prev_out_right[s]
            )
            buffers_right[s, write_pos] = (
                dry_sample_right + cross_feedback * prev_out_left[s]
            )
            prev_out_left[s] = delayed_l
            prev_out_right[s] = delayed_r

        write_pos = (write_pos + 1) % buffer_size

    return out_left, out_right


def apply_bbd_chorus(
    signal: np.ndarray,
    mix: float = 0.3,
    rate_hz: float = 0.51,
    depth_ms: float = 1.5,
    center_delay_ms: float = 3.2,
    cross_feedback: float = 0.08,
    compander_amount: float = 0.2,
    pre_lowpass_hz: float = 6_500.0,
    wet_lowpass_hz: float = 6_000.0,
    wet_highpass_hz: float = 120.0,
    stack_count: int = 1,
    preset: str | None = None,
) -> np.ndarray:
    """Native Juno-faithful BBD-style stereo chorus.

    Bucket-brigade flavor via pre/post bandlimiting, quadrature LFOs for true
    stereo decorrelation, cross-feedback to keep the wet field airy rather
    than metallic, and an optional gentle compander. The wet image is summed
    with the dry signal (not crossfaded) — ``mix`` scales the wet contribution,
    preserving the dry signal's presence.

    Parameters
    ----------
    mix:              Wet level added to dry (0 = dry only; 1 = full wet add).
    rate_hz:          LFO rate in Hz (typical 0.3 - 1.0).
    depth_ms:         Peak modulation depth around the center delay.
    center_delay_ms:  Base delay time. Juno I is ~3 ms; Dimension-D is ~10 ms.
    cross_feedback:   L->R and R->L recirculation 0..0.5. Higher values lock
                      the stereo field into a tight ensemble; very high values
                      get resonant. Self-feedback is deliberately avoided.
    compander_amount: 0..1 gentle soft-limiting on the wet path (tanh).
    pre_lowpass_hz:   Pre-delay input bandlimit (BBD input filter).
    wet_lowpass_hz:   Post-delay wet lowpass (BBD output filter).
    wet_highpass_hz:  Post-delay wet highpass (removes low-end smear).
    stack_count:      1 or 2. Two = Juno I+II-style stacked sections with
                      staggered LFO phases for denser motion.
    preset:           One of the named presets in ``_BBD_CHORUS_PRESETS``.

    References
    ----------
    Algorithm drawn from the published Juno-106 service manual (BBD clock
    rates, chorus I/II parameters) and standard BBD chorus topology
    (pre/post bandlimiting, quadrature LFOs, dry+wet summing). No proprietary
    code was copied; this is a from-concept implementation.
    """
    if preset is not None:
        # Route through the shared resolver so the preset handling matches the
        # rest of the effect surface. Preset values win over the kwargs to
        # preserve the historical behavior of this function.
        preset_vals = _resolve_effect_params("bbd_chorus", {"preset": preset})
        mix = float(preset_vals.get("mix", mix))
        rate_hz = float(preset_vals.get("rate_hz", rate_hz))
        depth_ms = float(preset_vals.get("depth_ms", depth_ms))
        center_delay_ms = float(preset_vals.get("center_delay_ms", center_delay_ms))
        cross_feedback = float(preset_vals.get("cross_feedback", cross_feedback))
        compander_amount = float(preset_vals.get("compander_amount", compander_amount))
        pre_lowpass_hz = float(preset_vals.get("pre_lowpass_hz", pre_lowpass_hz))
        wet_lowpass_hz = float(preset_vals.get("wet_lowpass_hz", wet_lowpass_hz))
        wet_highpass_hz = float(preset_vals.get("wet_highpass_hz", wet_highpass_hz))
        stack_count = int(preset_vals.get("stack_count", stack_count))

    if not 0.0 <= mix <= 1.0:
        raise ValueError("mix must be between 0 and 1")
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")
    if depth_ms < 0 or center_delay_ms <= 0:
        raise ValueError("depth_ms must be non-negative and center_delay_ms positive")
    if not 0.0 <= cross_feedback <= 0.5:
        raise ValueError("cross_feedback must be between 0 and 0.5")
    if not 0.0 <= compander_amount <= 1.0:
        raise ValueError("compander_amount must be between 0 and 1")
    if stack_count not in (1, 2):
        raise ValueError("stack_count must be 1 or 2")
    # center - depth must leave at least 0.5 ms of positive delay so the LFO
    # never flips negative at its trough.
    if depth_ms >= center_delay_ms:
        raise ValueError(
            "depth_ms must be less than center_delay_ms so delay stays positive"
        )

    dry_signal = _ensure_stereo(signal)
    if mix == 0.0:
        return dry_signal

    # Pre-emphasis bandlimit (BBD input filter) applied per-channel.
    pre_filtered = _apply_per_channel(
        dry_signal,
        lambda channel: lowpass(
            channel, cutoff_hz=pre_lowpass_hz, sample_rate=SAMPLE_RATE, order=2
        ),
    )

    # Compander knob maps 0..1 -> effective k in tanh(k*x)/k.
    # At k=0 the path is identity; larger k -> more aggressive soft-limit.
    # k up to ~3 keeps things musical; beyond that starts to dominate.
    compander_k = 3.0 * compander_amount

    # Time-base parameters.
    delay_base_samples = center_delay_ms * SAMPLE_RATE / 1000.0
    mod_depth_samples = depth_ms * SAMPLE_RATE / 1000.0
    # Safety margin: buffer must hold max delay + a few samples for interp.
    max_delay_samples = delay_base_samples + mod_depth_samples + 4.0
    buffer_size = int(max_delay_samples) + 4
    lfo_omega = 2.0 * np.pi * rate_hz / SAMPLE_RATE

    # Stage LFO phase pairs per stack. First stage: L=0, R=pi/2 (true quadrature).
    # Second stage: staggered by pi/3 to keep it decorrelated from stage 1.
    if stack_count == 2:
        stage_phases_left = np.array([0.0, np.pi / 3.0], dtype=np.float64)
        stage_phases_right = np.array(
            [0.5 * np.pi, np.pi / 3.0 + 0.5 * np.pi], dtype=np.float64
        )
    else:
        stage_phases_left = np.array([0.0], dtype=np.float64)
        stage_phases_right = np.array([0.5 * np.pi], dtype=np.float64)

    channel_left = np.asarray(pre_filtered[0], dtype=np.float64)
    channel_right = np.asarray(pre_filtered[1], dtype=np.float64)

    # Pre-scale stages so a multi-stage sum keeps energy comparable to a
    # single-stage pass (historical 1/sqrt(N) stage weighting).
    stage_scale = 1.0 / math.sqrt(stage_phases_left.shape[0])
    wet_left, wet_right = _bbd_chorus_stereo_loop(
        channel_left,
        channel_right,
        buffer_size,
        delay_base_samples,
        mod_depth_samples,
        lfo_omega,
        stage_phases_left,
        stage_phases_right,
        cross_feedback,
        compander_k,
        stage_scale,
    )

    wet_signal = np.stack([wet_left, wet_right])
    # Post-delay bandlimit (BBD output filter) + low-end cleanup.
    wet_signal = _apply_per_channel(
        wet_signal,
        lambda channel: lowpass(
            channel, cutoff_hz=wet_lowpass_hz, sample_rate=SAMPLE_RATE, order=2
        ),
    )
    wet_signal = _apply_per_channel(
        wet_signal,
        lambda channel: highpass(
            channel, cutoff_hz=wet_highpass_hz, sample_rate=SAMPLE_RATE, order=2
        ),
    )

    # Sum, don't crossfade: wet widens the dry image rather than replacing it.
    blended = dry_signal + (mix * wet_signal)
    return blended.astype(np.float64)


# ---------------------------------------------------------------------------
# Phaser (ChowPhaser VST3 wrapper)
# ---------------------------------------------------------------------------

_PHASER_PRESETS: dict[str, dict[str, float]] = {
    "gentle_sweep": {
        "rate_hz": 0.15,
        "depth": 0.4,
        "feedback": 0.25,
        "mix": 0.25,
    },
    "metallic_shimmer": {
        "rate_hz": 0.6,
        "depth": 0.65,
        "feedback": 0.75,
        "mix": 0.35,
    },
}


def apply_phaser(
    signal: np.ndarray,
    rate_hz: float = 0.3,
    depth: float = 0.5,
    feedback: float = 0.4,
    mix: float = 0.35,
    preset: str | None = None,
) -> np.ndarray:
    """Apply ChowPhaser stereo phaser effect via VST3.

    Falls back to returning the signal unchanged if the plugin is not installed.

    Parameters
    ----------
    rate_hz:  LFO rate in Hz (mapped to the plugin's 0-1 normalized range).
    depth:    Modulation depth 0-1.
    feedback: Feedback amount 0-1.
    mix:      Wet/dry blend 0-1.
    preset:   Optional preset name from ``_PHASER_PRESETS``.
    """
    if preset is not None:
        if preset not in _PHASER_PRESETS:
            raise ValueError(f"Unknown phaser preset: {preset!r}")
        preset_vals = _PHASER_PRESETS[preset]
        rate_hz = float(preset_vals.get("rate_hz", rate_hz))
        depth = float(preset_vals.get("depth", depth))
        feedback = float(preset_vals.get("feedback", feedback))
        mix = float(preset_vals.get("mix", mix))

    plugin_name = "chow_phaser_stereo"
    if not has_external_plugin(plugin_name):
        logger.warning(
            "ChowPhaser VST3 not found (chow_phaser_stereo). "
            "Returning signal unchanged. Install ChowPhaser to hear this effect."
        )
        return np.asarray(signal, dtype=np.float64)

    if not 0.0 <= mix <= 1.0:
        raise ValueError("mix must be between 0 and 1")

    dry_signal = _ensure_stereo(signal)

    # ChowPhaser params: lfo_freq (0-16 Hz), lfo_depth (0-0.95),
    # feedback (0-0.95), modulation (0-1)
    wet_signal = _apply_plugin_processor(
        signal,
        plugin_name=plugin_name,
        params={
            "lfo_freq": float(np.clip(rate_hz, 0.0, 16.0)),
            "lfo_depth": float(np.clip(depth * 0.95, 0.0, 0.95)),
            "feedback": float(np.clip(feedback * 0.95, 0.0, 0.95)),
            "modulation": float(np.clip(depth, 0.0, 1.0)),
        },
    )

    wet_stereo = _ensure_stereo(wet_signal)
    blended = (1.0 - mix) * dry_signal + mix * wet_stereo
    return _match_input_layout(blended.astype(np.float64), signal)


# ---------------------------------------------------------------------------
# Modulated delay (native)
# ---------------------------------------------------------------------------

_MOD_DELAY_PRESETS: dict[str, dict[str, float]] = {
    "dream_echo": {
        "delay_ms": 280.0,
        "mod_rate_hz": 0.12,
        "mod_depth_ms": 8.0,
        "feedback": 0.45,
        "feedback_lpf_hz": 2800.0,
        "stereo_offset_deg": 110.0,
        "mix": 0.3,
    },
    "shimmer_slap": {
        "delay_ms": 120.0,
        "mod_rate_hz": 0.5,
        "mod_depth_ms": 3.0,
        "feedback": 0.25,
        "feedback_lpf_hz": 6000.0,
        "stereo_offset_deg": 75.0,
        "mix": 0.25,
    },
    "tape_wander": {
        "delay_ms": 350.0,
        "mod_rate_hz": 0.08,
        "mod_depth_ms": 12.0,
        "feedback": 0.55,
        "feedback_lpf_hz": 2200.0,
        "stereo_offset_deg": 130.0,
        "mix": 0.35,
    },
}

_MAX_FEEDBACK: float = 0.92


@numba.njit(cache=True)
def _mod_delay_channel_loop(
    input_signal: np.ndarray,
    buffer_size: int,
    delay_base_samples: float,
    mod_depth_samples: float,
    lfo_omega: float,
    lfo_phase_offset: float,
    feedback: float,
    lpf_alpha: float,
    sample_rate: int,
) -> np.ndarray:
    """Per-channel modulated delay with feedback and lowpass, compiled via Numba.

    Uses cubic Hermite interpolation for the fractional delay read to avoid
    the high-frequency roll-off that linear interpolation causes when the
    read position is continuously moving.
    """
    n_samples = input_signal.shape[0]
    output = np.empty(n_samples, dtype=np.float64)
    delay_buffer = np.zeros(buffer_size, dtype=np.float64)
    write_pos = 0
    lpf_state = 0.0

    for i in range(n_samples):
        lfo_value = math.sin(lfo_omega * i / sample_rate + lfo_phase_offset)
        current_delay = delay_base_samples + mod_depth_samples * lfo_value
        current_delay = max(1.0, min(current_delay, buffer_size - 2.0))

        read_pos_float = write_pos - current_delay
        if read_pos_float < 0.0:
            read_pos_float += buffer_size

        idx = int(read_pos_float)
        frac = read_pos_float - idx

        # Cubic Hermite interpolation: four neighboring samples
        i0 = (idx - 1) % buffer_size
        i1 = idx % buffer_size
        i2 = (idx + 1) % buffer_size
        i3 = (idx + 2) % buffer_size

        y0 = delay_buffer[i0]
        y1 = delay_buffer[i1]
        y2 = delay_buffer[i2]
        y3 = delay_buffer[i3]

        # Hermite basis functions
        c0 = y1
        c1 = 0.5 * (y2 - y0)
        c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
        c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2)

        delayed_sample = ((c3 * frac + c2) * frac + c1) * frac + c0

        # One-pole lowpass in feedback path
        lpf_state = lpf_alpha * delayed_sample + (1.0 - lpf_alpha) * lpf_state
        filtered_feedback = lpf_state

        # Write to buffer: input + filtered feedback
        delay_buffer[write_pos] = input_signal[i] + feedback * filtered_feedback
        output[i] = delayed_sample

        write_pos = (write_pos + 1) % buffer_size

    return output


def apply_mod_delay(
    signal: np.ndarray,
    delay_ms: float = 200.0,
    mod_rate_hz: float = 0.2,
    mod_depth_ms: float = 5.0,
    feedback: float = 0.35,
    feedback_lpf_hz: float = 4000.0,
    stereo_offset_deg: float = 90.0,
    mix: float = 0.3,
    preset: str | None = None,
) -> np.ndarray:
    """Native modulated delay with filtered feedback and stereo spread.

    A chorus-delay hybrid: longer delay times than chorus, with an LFO-modulated
    read position and a lowpass filter in the feedback path that darkens repeats.

    Parameters
    ----------
    delay_ms:          Base delay time in milliseconds (50-500).
    mod_rate_hz:       LFO speed in Hz (0.03-3.0).
    mod_depth_ms:      LFO depth -- how much the delay time swings (0.5-30 ms).
    feedback:          Feedback amount (0-0.92, hard-clipped).
    feedback_lpf_hz:   Lowpass cutoff in the feedback path (darkens repeats).
    stereo_offset_deg: LFO phase offset between L/R for stereo spread.
    mix:               Wet/dry blend (0 = all dry, 1 = all wet).
    preset:            Optional preset name from ``_MOD_DELAY_PRESETS``.
    """
    if preset is not None:
        if preset not in _MOD_DELAY_PRESETS:
            raise ValueError(f"Unknown mod_delay preset: {preset!r}")
        preset_vals = _MOD_DELAY_PRESETS[preset]
        delay_ms = float(preset_vals.get("delay_ms", delay_ms))
        mod_rate_hz = float(preset_vals.get("mod_rate_hz", mod_rate_hz))
        mod_depth_ms = float(preset_vals.get("mod_depth_ms", mod_depth_ms))
        feedback = float(preset_vals.get("feedback", feedback))
        feedback_lpf_hz = float(preset_vals.get("feedback_lpf_hz", feedback_lpf_hz))
        stereo_offset_deg = float(
            preset_vals.get("stereo_offset_deg", stereo_offset_deg)
        )
        mix = float(preset_vals.get("mix", mix))

    if not 0.0 <= mix <= 1.0:
        raise ValueError("mix must be between 0 and 1")
    if delay_ms <= 0:
        raise ValueError("delay_ms must be positive")

    clamped_feedback = min(float(feedback), _MAX_FEEDBACK)

    dry_signal = _ensure_stereo(signal)

    max_delay_ms = delay_ms + mod_depth_ms + 5.0  # small safety margin
    buffer_size = int(max_delay_ms * SAMPLE_RATE / 1000.0) + 4  # +4 for cubic interp
    delay_base_samples = delay_ms * SAMPLE_RATE / 1000.0
    mod_depth_samples = mod_depth_ms * SAMPLE_RATE / 1000.0
    lfo_omega = 2.0 * np.pi * mod_rate_hz
    stereo_phase_rad = np.deg2rad(stereo_offset_deg)

    # One-pole LPF coefficient: alpha = 1 - exp(-2*pi*fc/sr)
    lpf_alpha = 1.0 - np.exp(-2.0 * np.pi * feedback_lpf_hz / SAMPLE_RATE)

    wet_channels: list[np.ndarray] = []
    for channel_index in range(2):
        channel_phase = stereo_phase_rad * channel_index
        channel_input = np.asarray(dry_signal[channel_index], dtype=np.float64)
        wet_channel = _mod_delay_channel_loop(
            channel_input,
            buffer_size,
            delay_base_samples,
            mod_depth_samples,
            lfo_omega,
            channel_phase,
            clamped_feedback,
            lpf_alpha,
            SAMPLE_RATE,
        )
        wet_channels.append(wet_channel)

    wet_signal = np.stack(wet_channels)
    blended = (1.0 - mix) * dry_signal + mix * wet_signal
    return blended.astype(np.float64)


def _dc_block(
    signal: np.ndarray,
    *,
    sample_rate: int,
    cutoff_hz: float = 12.0,
) -> np.ndarray:
    """Remove static offset with a very low high-pass."""
    return highpass(
        np.asarray(signal, dtype=np.float64),
        cutoff_hz=cutoff_hz,
        sample_rate=sample_rate,
        order=1,
    )


def _apply_drive_compensation(
    signal: np.ndarray,
    *,
    reference_signal: np.ndarray,
    sample_rate: int,
    compensation_mode: str,
    max_gain_db: float = 12.0,
) -> tuple[np.ndarray, str, float]:
    resolved_mode = compensation_mode.lower().strip()
    if resolved_mode == "none":
        return np.asarray(signal, dtype=np.float64), "none", 0.0
    if resolved_mode not in {"auto", "lufs", "rms"}:
        raise ValueError("compensation_mode must be 'none', 'auto', 'lufs', or 'rms'")

    processed = np.asarray(signal, dtype=np.float64)
    reference = np.asarray(reference_signal, dtype=np.float64)
    measurement_mode = resolved_mode
    if resolved_mode == "auto":
        measurement_mode = "lufs"

    if measurement_mode == "lufs":
        reference_lufs, reference_active_fraction = integrated_lufs(
            reference,
            sample_rate=sample_rate,
        )
        processed_lufs, processed_active_fraction = integrated_lufs(
            processed,
            sample_rate=sample_rate,
        )
        if (
            min(reference.shape[-1], processed.shape[-1]) < sample_rate
            or reference_active_fraction < 0.25
            or processed_active_fraction < 0.25
            or not np.isfinite(reference_lufs)
            or not np.isfinite(processed_lufs)
        ):
            if resolved_mode == "lufs":
                return processed, "lufs", 0.0
            measurement_mode = "rms"
        else:
            applied_gain_db = float(
                np.clip(reference_lufs - processed_lufs, -max_gain_db, max_gain_db)
            )
            return processed * db_to_amp(applied_gain_db), "lufs", applied_gain_db

    reference_rms_dbfs, reference_active_fraction = gated_rms_dbfs(
        reference,
        sample_rate=sample_rate,
    )
    processed_rms_dbfs, processed_active_fraction = gated_rms_dbfs(
        processed,
        sample_rate=sample_rate,
    )
    if (
        not np.isfinite(reference_rms_dbfs)
        or not np.isfinite(processed_rms_dbfs)
        or reference_active_fraction <= 0.0
        or processed_active_fraction <= 0.0
    ):
        return processed, measurement_mode, 0.0

    applied_gain_db = float(
        np.clip(reference_rms_dbfs - processed_rms_dbfs, -max_gain_db, max_gain_db)
    )
    return processed * db_to_amp(applied_gain_db), "rms", applied_gain_db


def _saturation_thd(
    processor: Callable[[np.ndarray], np.ndarray],
    test_amp: float = 0.25,
) -> tuple[float, str]:
    """Compute characteristic THD% and classify distortion level.

    Applies the saturation shaper to a 440 Hz reference sine and measures how
    much harmonic energy (2nd–10th) exists relative to the fundamental.
    Independent of actual audio content — characterises the shaper curve itself.

    Returns (thd_pct, label) where label is one of:
    ``"clean"``, ``"subtle_warmth"``, ``"warmth"``, ``"saturation"``,
    ``"distortion"``, ``"fuzz"``.
    """
    sr = 44100
    f0 = 440.0
    n_fft = 4096
    t = np.arange(n_fft, dtype=np.float64) / sr
    x = test_amp * np.sin(2.0 * np.pi * f0 * t)

    y = np.asarray(processor(x), dtype=np.float64)

    spectrum = np.abs(np.fft.rfft(y))
    bin_per_hz = n_fft / sr
    h1_bin = int(round(f0 * bin_per_hz))
    h1 = spectrum[h1_bin]
    harmonics_sq_sum = sum(
        spectrum[int(round(h * f0 * bin_per_hz))] ** 2
        for h in range(2, 11)
        if int(round(h * f0 * bin_per_hz)) < len(spectrum)
    )
    thd_pct = float(np.sqrt(harmonics_sq_sum)) / (h1 + 1e-10) * 100.0
    label = classify_thd(thd_pct)

    return round(thd_pct, 2), label


def _solve_drive_for_target_thd(
    *,
    target_thd_pct: float,
    probe_processor_factory: Callable[[float], Callable[[np.ndarray], np.ndarray]],
    search_low: float = 1.0e-3,
    search_high: float = 10.0,
    max_iterations: int = 12,
    tolerance_pct: float = 0.25,
) -> tuple[float, float, int]:
    """Binary-search ``drive`` for target shaper THD% (sine-probe characteristic).

    Monotonic: higher drive produces higher characteristic THD on both the
    modern and legacy shapers, so standard bisection converges. The probe is
    ``_saturation_thd`` (440 Hz sine, harmonics 2-10) which characterises the
    shaper curve itself, independent of the caller's actual input. That is
    the intended perceptual target — "musical saturation" is a property of
    the shaper, not of the incoming signal.

    If the requested ``target_thd_pct`` is outside the achievable range of
    ``[search_low, search_high]`` the solver snaps to whichever endpoint is
    closest and returns early.

    Returns ``(solved_drive, measured_thd_pct, iterations)``.
    """
    low_thd, _ = _saturation_thd(probe_processor_factory(search_low))
    if target_thd_pct <= low_thd:
        return float(search_low), float(low_thd), 1
    high_thd, _ = _saturation_thd(probe_processor_factory(search_high))
    if target_thd_pct >= high_thd:
        return float(search_high), float(high_thd), 2

    low = float(search_low)
    high = float(search_high)
    best_drive = 0.5 * (low + high)
    best_measured = 0.0
    best_abs_error = float("inf")
    iterations = 0
    for _ in range(max_iterations):
        iterations += 1
        mid = 0.5 * (low + high)
        measured_thd, _ = _saturation_thd(probe_processor_factory(mid))
        abs_error = abs(measured_thd - target_thd_pct)
        if abs_error < best_abs_error:
            best_abs_error = abs_error
            best_drive = mid
            best_measured = measured_thd
        if abs_error < tolerance_pct:
            return float(mid), float(measured_thd), iterations
        if measured_thd > target_thd_pct:
            high = mid
        else:
            low = mid
    return float(best_drive), float(best_measured), iterations


@numba.njit(cache=True)
def _envelope_follower_loop(
    abs_signal: np.ndarray,
    attack_coeff: float,
    release_coeff: float,
) -> np.ndarray:
    n = abs_signal.shape[0]
    envelope = np.empty(n, dtype=np.float64)
    previous = 0.0
    for i in range(n):
        sample = abs_signal[i]
        coeff = attack_coeff if sample > previous else release_coeff
        previous = (coeff * previous) + ((1.0 - coeff) * sample)
        envelope[i] = previous
    return envelope


def _envelope_follower(
    signal: np.ndarray,
    *,
    sample_rate: int,
    attack_ms: float,
    release_ms: float,
) -> np.ndarray:
    attack_coeff = _time_constant_to_coeff(attack_ms, sample_rate)
    release_coeff = _time_constant_to_coeff(release_ms, sample_rate)
    abs_signal = np.abs(np.asarray(signal, dtype=np.float64))
    return _envelope_follower_loop(abs_signal, attack_coeff, release_coeff)


# ---------------------------------------------------------------------------
# Preamp: flux-domain transformer saturation
# ---------------------------------------------------------------------------

_PREAMP_PRESETS: dict[str, dict[str, Any]] = {
    "neve_warmth": {
        "drive": 0.5,
        "mix": 0.30,
        "warmth": 0.5,
        "brightness": 0.0,
        "even_odd": 0.7,
        "harmonic_injection": 0.5,
    },
    "iron_color": {
        "drive": 0.6,
        "mix": 0.35,
        "warmth": 0.55,
        "brightness": 0.0,
        "even_odd": 0.65,
        "harmonic_injection": 0.55,
    },
    "tube_glow": {
        "drive": 0.5,
        "mix": 0.30,
        "warmth": 0.4,
        "brightness": 0.0,
        "even_odd": 0.55,
        "harmonic_injection": 0.6,
    },
    "transformer_drive": {
        "drive": 1.2,
        "mix": 0.50,
        "warmth": 0.6,
        "brightness": 0.0,
        "even_odd": 0.75,
        "harmonic_injection": 0.65,
    },
    "kick_body": {
        # Flux-domain body emphasis for kicks. Replaces the old
        # drive-based `kick_weight` preset, which despite its name added
        # ~5 dB of 2-5 kHz papery brightness. The preamp's V -> dPhi/dt
        # integrator saturates bass more than highs, so this preset adds
        # low-harmonic richness (200-1 kHz +30 dB of wet-sum) while
        # leaving the 2-5 kHz band essentially flat.
        "drive": 0.5,
        "mix": 0.35,
        "warmth": 0.6,
        "brightness": 0.0,
        "even_odd": 0.7,
        "harmonic_injection": 0.55,
    },
    "tom_body": {
        # Sibling of `kick_body` tuned lighter for tom content. Same
        # flux-domain low-harmonic emphasis, but gentler mix to preserve
        # tom sustain and avoid muddying the lowmids in a full kit.
        "drive": 0.4,
        "mix": 0.30,
        "warmth": 0.55,
        "brightness": 0.0,
        "even_odd": 0.7,
        "harmonic_injection": 0.55,
    },
}


def apply_preamp(
    signal: np.ndarray,
    *,
    drive: float = 0.35,
    mix: float = 0.30,
    warmth: float = 0.5,
    brightness: float = 0.0,
    even_odd: float = 0.7,
    flux_cutoff_hz: float = 12.0,
    harmonic_injection: float = 0.5,
    oversample_factor: int = 4,
    compensation_mode: str = "auto",
    preset: str | None = None,
    sample_rate: int = SAMPLE_RATE,
    return_analysis: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int | str]]:
    """Flux-domain transformer saturation modeling real iron-core physics.

    Unlike memoryless waveshaping, this operates in the magnetic flux domain
    where low frequencies naturally saturate more than highs (V = N * dPhi/dt).
    The result is frequency-dependent harmonic generation with a warm,
    analog character and minimal intermodulation on complex material.

    Parameters
    ----------
    signal:
        Mono or stereo input audio.
    drive:
        How hard the transformer core is driven. 0.25 = barely there,
        0.5 = gentle warmth, 1.0 = rich, 1.5+ = crunchy.
    mix:
        Wet/dry blend (0.0--1.0). Default 0.30 for subtle bus color.
    warmth:
        Pre-emphasis bass shelf amount (0.0--1.0). Higher values enrich
        low-frequency harmonics by boosting bass into the nonlinearity.
    brightness:
        Post-processing tilt EQ (-1.0 to 1.0). 0 = neutral, positive =
        brighter, negative = darker.
    even_odd:
        Balance of even vs odd harmonics (0.0--1.0). 0 = odd-dominant
        (symmetric saturation), 1.0 = even-dominant (asymmetric, like a
        real transformer). Default 0.7.
    flux_cutoff_hz:
        Leaky integrator corner frequency for voltage-to-flux conversion.
        Lower = more bass-focused saturation, higher = broader. Default 12 Hz.
    harmonic_injection:
        Amount of parallel Chebyshev harmonic injection (0.0--1.0).
        Adds controlled 2nd/3rd/4th harmonics scaled by signal envelope.
    oversample_factor:
        Internal oversampling for the flux-domain processing path.
    compensation_mode:
        Loudness compensation: ``"auto"``/``"lufs"``/``"rms"``/``"none"``.
    preset:
        Optional named preset (``"neve_warmth"``, ``"iron_color"``,
        ``"tube_glow"``, ``"transformer_drive"``).
    sample_rate:
        Audio sample rate.
    return_analysis:
        When True, return ``(signal, analysis_dict)`` with THD and metrics.
    """
    if preset is not None:
        resolved = _resolve_effect_params("preamp", {"preset": preset})
        if "drive" not in resolved or drive != 0.35:
            resolved["drive"] = drive
        if "mix" not in resolved or mix != 0.30:
            resolved["mix"] = mix
        if "warmth" not in resolved or warmth != 0.5:
            resolved["warmth"] = warmth
        if "brightness" not in resolved or brightness != 0.0:
            resolved["brightness"] = brightness
        if "even_odd" not in resolved or even_odd != 0.7:
            resolved["even_odd"] = even_odd
        if "harmonic_injection" not in resolved or harmonic_injection != 0.5:
            resolved["harmonic_injection"] = harmonic_injection
        drive = float(resolved.get("drive", drive))
        mix = float(resolved.get("mix", mix))
        warmth = float(resolved.get("warmth", warmth))
        brightness = float(resolved.get("brightness", brightness))
        even_odd = float(resolved.get("even_odd", even_odd))
        harmonic_injection = float(
            resolved.get("harmonic_injection", harmonic_injection)
        )

    if not 0.0 <= mix <= 1.0:
        raise ValueError("mix must be between 0 and 1")
    if drive < 0.0:
        raise ValueError("drive must be non-negative")
    if oversample_factor < 1:
        raise ValueError("oversample_factor must be at least 1")

    input_signal = np.asarray(signal, dtype=np.float64)

    def _process_channel(channel: np.ndarray) -> np.ndarray:
        dry_channel = np.asarray(channel, dtype=np.float64)
        os_sr = sample_rate * oversample_factor

        # --- Step 1: Oversample ---
        if oversample_factor > 1:
            oversampled = resample_poly(dry_channel, oversample_factor, 1)
        else:
            oversampled = dry_channel.copy()
        dry_oversampled = oversampled.copy()

        # --- Step 2: Pre-emphasis (gentle bass shelf boost) ---
        # Models transformer coupling impedance: bass gets slightly more
        # energy into the core, matching real iron behavior.
        pre_emphasis_db = warmth * 3.0
        if pre_emphasis_db > 0.01:
            pre_sos = tf2sos(
                *_design_low_shelf_biquad(
                    sample_rate=os_sr,
                    center_hz=200.0,
                    q=0.5,
                    gain_db=pre_emphasis_db,
                )
            )
            oversampled = np.asarray(sosfilt(pre_sos, oversampled), dtype=np.float64)

        # --- Step 3: Leaky integrator (voltage -> flux domain) ---
        # Phi[n] = leak * Phi[n-1] + x[n] / sr
        # This is a 1st-order IIR lowpass: implements Faraday's law where
        # flux is the integral of voltage, with a DC-preventing leak term.
        leak = float(np.exp(-2.0 * np.pi * flux_cutoff_hz / os_sr))
        flux = np.asarray(
            lfilter([1.0 / os_sr], [1.0, -leak], oversampled),
            dtype=np.float64,
        )

        # --- Step 4: Soft nonlinearity in flux domain ---
        # Drive scaling referenced to typical musical content (~200 Hz).
        # The integrator gain at frequency f is ~1/(2*pi*f), so flux at
        # 200 Hz for a 0.45-peak signal is ~0.00036.  The drive_scale
        # pushes that into arctan's nonlinear region.  Bass sees higher
        # flux (1/f) → more saturation, matching real transformer physics.
        drive_scale = max(drive * 2.0 * np.pi * 200.0 * 1.2, 1e-6)

        # DC bias for even-harmonic asymmetry (transformer core asymmetry).
        # The bias is envelope-scaled so it tracks signal level.
        flux_envelope = _envelope_follower(
            flux,
            sample_rate=os_sr,
            attack_ms=5.0,
            release_ms=80.0,
        )
        bias_amount = even_odd * 0.05 * (1.0 + drive)
        flux_biased = flux + bias_amount * flux_envelope

        # Arctan with (2/pi) normalization: generates increasing harmonic
        # content as drive_scale pushes flux deeper into the nonlinear
        # region.  Low frequencies see higher flux → more saturation →
        # real transformer behavior.  Output level is corrected after
        # differentiation via RMS matching.
        flux_saturated = (2.0 / np.pi) * np.arctan(
            (np.pi / 2.0) * drive_scale * flux_biased
        )

        # --- Step 5: Differentiate (flux -> voltage) ---
        # y[n] = (x[n] - x[n-1]) * sr  (Faraday: V = dPhi/dt)
        voltage = np.diff(flux_saturated, prepend=flux_saturated[0]) * os_sr

        # RMS-match voltage output to dry input level.  The arctan
        # generates increasing harmonics with drive (good) but also
        # changes the overall gain.  Normalizing here preserves the
        # harmonic content while ensuring the difference extraction
        # captures only color, not level change.
        dry_rms = float(np.sqrt(np.mean(dry_oversampled**2))) + 1e-12
        voltage_rms = float(np.sqrt(np.mean(voltage**2))) + 1e-12
        voltage = voltage * (dry_rms / voltage_rms)

        # --- Step 6: De-emphasis (invert pre-emphasis) ---
        if pre_emphasis_db > 0.01:
            de_sos = tf2sos(
                *_design_low_shelf_biquad(
                    sample_rate=os_sr,
                    center_hz=200.0,
                    q=0.5,
                    gain_db=-pre_emphasis_db,
                )
            )
            voltage = np.asarray(sosfilt(de_sos, voltage), dtype=np.float64)

        # --- Step 7: Extract harmonic difference ---
        # Work with the difference signal (wet - dry) to preserve the
        # original content's phase and tonality, mixing back only the
        # coloration the transformer added.
        flux_color = voltage - dry_oversampled

        # --- Step 8: Filter difference ---
        # LP ~12kHz: remove ultrasonic artifacts from differentiation.
        # HP ~20Hz: remove DC / sub-bass drift from integration residue.
        flux_color = lowpass(flux_color, cutoff_hz=12_000.0, sample_rate=os_sr, order=2)
        flux_color = highpass(flux_color, cutoff_hz=20.0, sample_rate=os_sr, order=1)

        # --- Step 9: Downsample ---
        if oversample_factor > 1:
            flux_color = resample_poly(flux_color, 1, oversample_factor)
        flux_color = np.asarray(flux_color[: dry_channel.shape[-1]], dtype=np.float64)

        # --- Step 10: Parallel Chebyshev harmonic injection ---
        # Adds controlled harmonics without the intermod cascade that
        # plagues memoryless waveshaping on polyphonic material.  Each
        # Chebyshev polynomial generates a single harmonic order cleanly.
        chebyshev_color = np.zeros_like(dry_channel)
        if harmonic_injection > 0.0:
            env = _envelope_follower(
                dry_channel,
                sample_rate=sample_rate,
                attack_ms=8.0,
                release_ms=120.0,
            )
            # Normalize signal for Chebyshev polynomials (need |x| <= 1)
            peak = float(np.max(np.abs(dry_channel))) + 1e-12
            x_norm = dry_channel / peak

            # T2(x) = 2x^2 - 1  (2nd harmonic / octave)
            t2 = 2.0 * x_norm**2 - 1.0
            # T3(x) = 4x^3 - 3x  (3rd harmonic / octave + fifth)
            t3 = 4.0 * x_norm**3 - 3.0 * x_norm
            # T4(x) = 8x^4 - 8x^2 + 1  (4th harmonic / two octaves)
            t4 = 8.0 * x_norm**4 - 8.0 * x_norm**2 + 1.0

            # Weight even harmonics by even_odd, odd by (1 - even_odd).
            # Scale by envelope so harmonics track dynamics naturally.
            injection_scale = harmonic_injection * drive * 0.12
            h2_weight = even_odd * 0.6
            h3_weight = (1.0 - even_odd) * 0.4
            h4_weight = even_odd * 0.2
            chebyshev_color = (
                injection_scale
                * env
                * (h2_weight * t2 + h3_weight * t3 + h4_weight * t4)
            )
            # Remove DC introduced by even-order Chebyshev terms
            chebyshev_color = _dc_block(
                chebyshev_color, sample_rate=sample_rate, cutoff_hz=15.0
            )

        return flux_color + chebyshev_color

    # --- Process per-channel, mix, compensate ---
    wet_difference = _apply_per_channel(input_signal, _process_channel)
    blended = input_signal + mix * wet_difference

    # Post-processing brightness tilt
    if brightness != 0.0:
        tilt_db = brightness * 2.5
        blended = _apply_per_channel(
            blended,
            lambda ch: _apply_tilt_eq(
                ch, sample_rate=sample_rate, tilt_db=tilt_db, pivot_hz=2_500.0
            ),
        )

    compensated_signal, compensation_mode_used, compensation_gain_db = (
        _apply_drive_compensation(
            blended,
            reference_signal=input_signal,
            sample_rate=sample_rate,
            compensation_mode=compensation_mode,
        )
    )
    processed_signal = np.asarray(compensated_signal, dtype=np.float64)

    if not return_analysis:
        return processed_signal

    thd_pct, thd_character = _saturation_thd(
        lambda x: cast(
            np.ndarray,
            apply_preamp(
                x,
                drive=drive,
                mix=1.0,
                warmth=warmth,
                brightness=0.0,
                even_odd=even_odd,
                flux_cutoff_hz=flux_cutoff_hz,
                harmonic_injection=harmonic_injection,
                oversample_factor=oversample_factor,
                compensation_mode="none",
                sample_rate=sample_rate,
                return_analysis=False,
            ),
        )
    )
    analysis: dict[str, float | int | str] = {
        "algorithm": "flux_transformer",
        "drive": round(float(drive), 2),
        "mix": round(float(mix), 2),
        "warmth": round(float(warmth), 2),
        "even_odd": round(float(even_odd), 2),
        "harmonic_injection": round(float(harmonic_injection), 2),
        "dc_offset": round(float(np.mean(to_mono_reference(processed_signal))), 6),
        "thd_pct": thd_pct,
        "thd_character": thd_character,
        "compensation_mode_used": compensation_mode_used,
        "compensation_gain_db": round(float(compensation_gain_db), 2),
    }
    return processed_signal, analysis


# ---------------------------------------------------------------------------
# Shared helpers for apply_tube / apply_transistor.
#
# Honest, purpose-built versions of the LR4 multiband crossover, tone-tilt EQ,
# highpass, DC-block, and LUFS/RMS compensation used by both effects.  Split
# out from the per-effect bodies so both tube and transistor share the same
# gain-staging / compensation surface.
# ---------------------------------------------------------------------------


def _drive_knob_gain(drive: float) -> float:
    """Map the user-facing 0-1 drive knob to internal gain.

    Mirrors ``code_musics.engines._waveshaper._drive_to_gain`` so the new
    tube / transistor effects share a calibration with the shaper primitives:
    ``drive=0 -> 1.0`` (bypass), ``drive=0.5 -> 13.25``, ``drive=1.0 -> 50.0``.
    """
    return 1.0 + 49.0 * float(drive) * float(drive)


def _tube_multiband_split(
    channel: np.ndarray,
    *,
    low_crossover_hz: float,
    high_crossover_hz: float,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LR4 split a mono channel into (low, mid, high) bands that sum to the input."""
    x = np.asarray(channel, dtype=np.float64)
    low = (
        _linkwitz_riley_lowpass(x, low_crossover_hz, sample_rate)
        if low_crossover_hz > 0.0
        else np.zeros_like(x)
    )
    high = (
        _linkwitz_riley_highpass(x, high_crossover_hz, sample_rate)
        if high_crossover_hz > 0.0
        else np.zeros_like(x)
    )
    mid = x
    if low_crossover_hz > 0.0:
        mid = _linkwitz_riley_highpass(mid, low_crossover_hz, sample_rate)
    if high_crossover_hz > 0.0:
        mid = _linkwitz_riley_lowpass(mid, high_crossover_hz, sample_rate)
    return low, mid, high


def _chebyshev_harmonic_blend(
    signal: np.ndarray,
    *,
    amount: float,
    sample_rate: int,
    even_odd: float = 0.7,
) -> np.ndarray:
    """Parallel Chebyshev harmonic injector (T2 / T3 / T4).

    Lifted from ``apply_preamp`` so ``apply_tube`` can use the same
    HG2-style 12AT7 parallel-color path without forking the math.
    Each Chebyshev polynomial generates a single harmonic order cleanly,
    avoiding the intermodulation cascade that plagues memoryless
    waveshaping on polyphonic material.  DC offset from even-order
    terms is blocked on the way out.
    """
    if amount <= 0.0:
        return np.zeros_like(np.asarray(signal, dtype=np.float64))
    dry = np.asarray(signal, dtype=np.float64)
    envelope = _envelope_follower(
        dry, sample_rate=sample_rate, attack_ms=8.0, release_ms=120.0
    )
    peak = float(np.max(np.abs(dry))) + 1e-12
    x_norm = dry / peak
    t2 = 2.0 * x_norm**2 - 1.0
    t3 = 4.0 * x_norm**3 - 3.0 * x_norm
    t4 = 8.0 * x_norm**4 - 8.0 * x_norm**2 + 1.0
    h2_weight = even_odd * 0.6
    h3_weight = (1.0 - even_odd) * 0.4
    h4_weight = even_odd * 0.2
    color = amount * envelope * (h2_weight * t2 + h3_weight * t3 + h4_weight * t4)
    return _dc_block(color, sample_rate=sample_rate, cutoff_hz=15.0)


def _transistor_diode_shape(signal: np.ndarray, asymmetry: float) -> np.ndarray:
    """Vectorised asymmetric Shockley-like diode shaper.

    Mirrors ``code_musics.engines._filters._diode_shape`` for memoryless
    waveshaping use.  ``asymmetry ∈ [0, 1]``: at 0 the curve is a
    near-symmetric log-soft-clip, at 1 the positive swing saturates
    earlier than the negative (the MS-20 / TS "snarl" direction).
    """
    a = float(np.clip(asymmetry, 0.0, 1.0))
    is_pos = 1.0 - 0.35 * a
    is_neg = 1.0 + 0.5 * a
    x = np.asarray(signal, dtype=np.float64)
    ax = np.abs(x)
    pos = np.log1p(ax / is_pos) / np.log1p(1.0 / is_pos)
    neg = -np.log1p(ax / is_neg) / np.log1p(1.0 / is_neg)
    return np.where(x >= 0.0, pos, neg)


def _soft_clip_algebraic(x: np.ndarray) -> np.ndarray:
    """Algebraic (non-tanh) soft clipper: ``x / (1 + |x|)``.

    Output bounded to ``(-1, 1)``, smooth, symmetric.  The honest
    replacement for tanh when we want op-amp soft-clip character
    without pretending it's tube-flavoured.
    """
    xa = np.asarray(x, dtype=np.float64)
    return xa / (1.0 + np.abs(xa))


def _op_amp_soft_knee(x: np.ndarray, knee: float = 0.12) -> np.ndarray:
    """Hard clip with a log-sum-exp soft knee around ±1.

    ``knee`` controls the radius of the smoothed transition (in the
    same units as ``x``); smaller = harder knee.  The approximation uses
    ``softplus`` to round the corner at ``x=1`` and again at ``x=-1``.
    """
    xa = np.asarray(x, dtype=np.float64)
    k = max(float(knee), 1.0e-3)
    upper = 1.0 - k * np.logaddexp(0.0, (1.0 - xa) / k)
    return -1.0 + k * np.logaddexp(0.0, (upper + 1.0) / k)


def _fuzz_cascade(x: np.ndarray, bias: float) -> np.ndarray:
    """Two cascaded hard-clips with a bias offset between them.

    Fuzz-Face-adjacent: the interstage bias is what makes fuzz gated
    and splattery rather than a clean square wave.  DC-block lives
    downstream; we don't pre-compensate the bias here.
    """
    stage1 = np.clip(np.asarray(x, dtype=np.float64) + 0.5 * bias, -1.0, 1.0)
    return np.clip(1.8 * stage1 + 0.5 * bias, -1.0, 1.0)


# ---------------------------------------------------------------------------
# apply_tube — Koren triode / pentode / HG2 cascade / Culture-Vulture
# ---------------------------------------------------------------------------


_TUBE_PRESETS: dict[str, dict[str, Any]] = {
    "triode_glow": {
        "character": "triode",
        "drive": 0.35,
        "bias": 0.15,
        "mix": 0.35,
    },
    "triode_bloom": {
        "character": "triode",
        "drive": 0.6,
        "bias": 0.25,
        "mix": 0.5,
    },
    "pentode_bite": {
        "character": "pentode",
        "drive": 0.5,
        "sharpness": 5.0,
        "mix": 0.4,
    },
    "hg2_enhancer": {
        "character": "hg2",
        "pentode_drive": 0.25,
        "triode_drive": 0.3,
        "parallel_drive": 0.2,
        "mix": 0.3,
    },
    "hg2_drive": {
        "character": "hg2",
        "pentode_drive": 0.6,
        "triode_drive": 0.7,
        "parallel_drive": 0.35,
        "mix": 0.6,
    },
    "culture_warm": {
        "character": "culture",
        "drive": 0.5,
        "bias": 0.4,
        "mix": 0.5,
    },
    "culture_growl": {
        "character": "culture",
        "drive": 0.8,
        "bias": 0.7,
        "mix": 0.7,
    },
    "culture_starve": {
        "character": "culture",
        "drive": 1.0,
        "bias": 0.95,
        "mix": 0.8,
    },
}


_TUBE_VALID_CHARACTERS: frozenset[str] = frozenset(
    {"triode", "pentode", "hg2", "culture"}
)


def _tube_stage_triode(
    x: np.ndarray,
    *,
    drive: float,
    bias: float,
    mu: float,
    ex: float,
) -> np.ndarray:
    """Apply a Koren-triode stage at the given drive/bias.

    No zero-correction: we rely on a downstream DC-block so that the
    asymmetry signature survives to the audio output.  That is what
    makes the Culture-Vulture starved-bias sound reachable.
    """
    gain = _drive_knob_gain(drive)
    driven = gain * x
    if bias == 0.0:
        return _koren_triode_shape(driven, mu=mu, ex=ex)
    return _biased_shape(driven, _koren_triode_shape, bias=bias, mu=mu, ex=ex)


def _tube_stage_pentode(
    x: np.ndarray,
    *,
    drive: float,
    sharpness: float,
) -> np.ndarray:
    """Apply a sharp-knee pentode stage.  Symmetric by construction."""
    gain = _drive_knob_gain(drive)
    return _pentode_shape(gain * x, sharpness=sharpness)


def apply_tube(
    signal: np.ndarray,
    *,
    character: str = "hg2",
    drive: float = 0.5,
    pentode_drive: float | None = None,
    triode_drive: float | None = None,
    bias: float = 0.0,
    parallel_drive: float = 0.0,
    sharpness: float = 4.0,
    mu: float = 100.0,
    ex: float = 1.4,
    tone: float = 0.0,
    tone_tilt: float = 0.0,
    highpass_hz: float = 30.0,
    mix: float = 0.5,
    compensation_mode: str = "auto",
    oversample_factor: int = 4,
    multiband: bool = True,
    low_crossover_hz: float = 120.0,
    high_crossover_hz: float = 5_000.0,
    preset: str | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Koren-ish softplus-triode / pentode / HG2-cascade / Culture-Vulture shaper.

    Physics-flavoured tube saturation across four characters.  Each
    character below has the harmonic signature the name implies (unlike
    the retired tanh-family ``apply_drive``):

    ``character="triode"``
        Koren softplus triode alone.  H2-dominant, asymmetric, smooth.
        Reach for this to add glow / warmth / "amp in the room" colour.
    ``character="pentode"``
        Sharp-knee pentode (``x / (1+|x|^n)^(1/n)``) alone.  Symmetric,
        H3-dominant, mid-focused edge.  Reach for this for lead bite.
    ``character="hg2"``
        Series cascade pentode → triode + optional parallel Chebyshev
        harmonic blend.  Per-stage drives ``pentode_drive`` and
        ``triode_drive`` (both fall back to ``drive``).  Models the HG2
        euphonic enhancer topology.
    ``character="culture"``
        Single triode with wide-range ``bias``.  No zero-correction —
        the downstream DC-block is what lets the Culture-Vulture
        "starvation" sound (one half-cycle collapsing) reach the
        output.  ``|bias| > 0.8`` enters starvation territory.

    Parameters
    ----------
    signal:
        Mono ``(N,)`` or stereo ``(2, N)`` audio.
    character:
        One of ``"triode"`` / ``"pentode"`` / ``"hg2"`` / ``"culture"``.
    drive:
        User-facing 0-1 drive knob.  ``0.2`` = subtle warmth, ``0.5`` =
        musical saturation, ``1.0`` = top of the musical range, ``>1.0``
        into fuzz territory.  Ignored for ``character="hg2"`` when
        both per-stage drives are provided.
    pentode_drive, triode_drive:
        Per-stage overrides for the HG2 cascade.  ``None`` falls back
        to ``drive``.  Ignored for non-HG2 characters.
    bias:
        Asymmetric-offset for ``triode`` / ``culture`` / ``hg2``.
        ``0`` = symmetric, ``±1`` = starved.  NOT DC-compensated in the
        shaper — the output DC-block handles centering.  Warning:
        ``|bias| > 0.8`` produces heavy asymmetry and is the intended
        Culture-Vulture range.  Ignored for ``character="pentode"``
        (pentode is symmetric by nature).
    parallel_drive:
        HG2-style 12AT7 parallel Chebyshev color amount (0-1).  Only
        active for ``character="hg2"``; ignored elsewhere.
    sharpness:
        Pentode knee hardness, ``3..6``.  Higher = harder clip / more H3.
    mu, ex:
        Koren triode advanced parameters.  Leave at defaults unless you
        know what you're doing.
    tone, tone_tilt:
        ``tone`` is a low/high balance knob (-1..1, positive = brighter);
        ``tone_tilt`` adds a post-shaper tilt EQ in dB (-6..6).
    highpass_hz:
        Pre-shaper highpass corner in Hz.  ``0`` disables.
    mix:
        Wet/dry blend (0-1).  Default 0.5.
    compensation_mode:
        Loudness compensation: ``"auto"`` / ``"lufs"`` / ``"rms"`` / ``"none"``.
    oversample_factor:
        Internal oversampling.  Auto-upgrades to 8 at ``drive >= 1.5``.
    multiband:
        When ``True`` (default) a LR4 multiband split routes only the mid
        band through the shaper; bass and air bypass cleanly.  Disable
        to push the full-range signal through the nonlinearity.
    low_crossover_hz, high_crossover_hz:
        Multiband LR4 corner frequencies.  Default ``120 / 5000`` Hz.
    preset:
        Optional named preset.  Available:
        ``triode_glow`` / ``triode_bloom`` — subtle-to-moderate H2 warmth;
        ``pentode_bite`` — mid-focused edge;
        ``hg2_enhancer`` / ``hg2_drive`` — HG2 euphonic to aggressive;
        ``culture_warm`` / ``culture_growl`` / ``culture_starve`` —
        Culture-Vulture from even-harmonic richness through full starvation.
    sample_rate:
        Audio sample rate.
    """
    if preset is not None:
        resolved_preset = _resolve_effect_params("tube", {"preset": preset})
        # Preset values win over Python kwargs only when the kwarg is at its
        # default; explicit user kwargs otherwise override.
        if character == "hg2" and "character" in resolved_preset:
            character = str(resolved_preset["character"])
        for key in (
            "drive",
            "pentode_drive",
            "triode_drive",
            "bias",
            "parallel_drive",
            "sharpness",
            "mu",
            "ex",
            "tone",
            "tone_tilt",
            "highpass_hz",
            "mix",
            "compensation_mode",
            "oversample_factor",
            "multiband",
            "low_crossover_hz",
            "high_crossover_hz",
        ):
            if key in resolved_preset:
                value = resolved_preset[key]
                if key == "drive" and drive == 0.5:
                    drive = float(value)
                elif key == "pentode_drive" and pentode_drive is None:
                    pentode_drive = float(value)
                elif key == "triode_drive" and triode_drive is None:
                    triode_drive = float(value)
                elif key == "bias" and bias == 0.0:
                    bias = float(value)
                elif key == "parallel_drive" and parallel_drive == 0.0:
                    parallel_drive = float(value)
                elif key == "sharpness" and sharpness == 4.0:
                    sharpness = float(value)
                elif key == "mu" and mu == 100.0:
                    mu = float(value)
                elif key == "ex" and ex == 1.4:
                    ex = float(value)
                elif key == "tone" and tone == 0.0:
                    tone = float(value)
                elif key == "tone_tilt" and tone_tilt == 0.0:
                    tone_tilt = float(value)
                elif key == "highpass_hz" and highpass_hz == 30.0:
                    highpass_hz = float(value)
                elif key == "mix" and mix == 0.5:
                    mix = float(value)
                elif key == "compensation_mode" and compensation_mode == "auto":
                    compensation_mode = str(value)
                elif key == "oversample_factor" and oversample_factor == 4:
                    oversample_factor = int(value)
                elif key == "multiband" and multiband is True:
                    multiband = bool(value)
                elif key == "low_crossover_hz" and low_crossover_hz == 120.0:
                    low_crossover_hz = float(value)
                elif key == "high_crossover_hz" and high_crossover_hz == 5_000.0:
                    high_crossover_hz = float(value)

    if character not in _TUBE_VALID_CHARACTERS:
        raise ValueError(
            f"character must be one of {sorted(_TUBE_VALID_CHARACTERS)}, "
            f"got {character!r}"
        )
    if not 0.0 <= mix <= 1.0:
        raise ValueError("mix must be between 0 and 1")
    if drive < 0.0:
        raise ValueError("drive must be non-negative")
    if not -1.0 <= bias <= 1.0:
        raise ValueError("bias must be between -1 and 1")
    if not 0.0 <= parallel_drive <= 1.0:
        raise ValueError("parallel_drive must be between 0 and 1")
    if not 2.5 <= sharpness <= 10.0:
        raise ValueError("sharpness must be between 2.5 and 10")
    if oversample_factor < 1:
        raise ValueError("oversample_factor must be at least 1")
    if highpass_hz < 0.0:
        raise ValueError("highpass_hz must be non-negative")

    resolved_oversample = oversample_factor
    if drive >= 1.5 and resolved_oversample < 8:
        resolved_oversample = 8

    input_signal = np.asarray(signal, dtype=np.float64)

    if mix <= 0.0:
        return input_signal.copy()

    resolved_pentode_drive = drive if pentode_drive is None else pentode_drive
    resolved_triode_drive = drive if triode_drive is None else triode_drive

    def _shape(channel: np.ndarray) -> np.ndarray:
        x = np.asarray(channel, dtype=np.float64)
        if resolved_oversample > 1:
            x_os = resample_poly(x, resolved_oversample, 1)
        else:
            x_os = x
        if character == "triode":
            shaped = _tube_stage_triode(x_os, drive=drive, bias=bias, mu=mu, ex=ex)
        elif character == "pentode":
            shaped = _tube_stage_pentode(x_os, drive=drive, sharpness=sharpness)
        elif character == "culture":
            shaped = _tube_stage_triode(x_os, drive=drive, bias=bias, mu=mu, ex=ex)
        else:  # "hg2"
            pent = _tube_stage_pentode(
                x_os, drive=resolved_pentode_drive, sharpness=sharpness
            )
            shaped = _tube_stage_triode(
                pent, drive=resolved_triode_drive, bias=bias, mu=mu, ex=ex
            )
            if parallel_drive > 0.0:
                os_sr = sample_rate * resolved_oversample
                cheb = _chebyshev_harmonic_blend(
                    x_os,
                    amount=parallel_drive * 0.3,
                    sample_rate=os_sr,
                    even_odd=0.7,
                )
                shaped = shaped + cheb
        if resolved_oversample > 1:
            shaped = resample_poly(shaped, 1, resolved_oversample)
        return np.asarray(shaped[: x.shape[-1]], dtype=np.float64)

    def _process_channel(channel: np.ndarray) -> np.ndarray:
        dry = np.asarray(channel, dtype=np.float64)
        conditioned = dry
        if highpass_hz > 0.0:
            conditioned = highpass(
                conditioned,
                cutoff_hz=highpass_hz,
                sample_rate=sample_rate,
                order=2,
            )
        pre_tilt_db = 5.0 * float(tone)
        if pre_tilt_db != 0.0:
            conditioned = _apply_tilt_eq(
                conditioned,
                sample_rate=sample_rate,
                tilt_db=pre_tilt_db,
                pivot_hz=2_100.0,
            )
        if multiband:
            low, mid, high = _tube_multiband_split(
                conditioned,
                low_crossover_hz=low_crossover_hz,
                high_crossover_hz=high_crossover_hz,
                sample_rate=sample_rate,
            )
            shaped_mid = _shape(mid)
            wet = low + shaped_mid + high
        else:
            wet = _shape(conditioned)
        # Second-order DC block — the new tube/transistor effects can produce
        # substantial asymmetry (Culture-Vulture bias, diode shaper) and a
        # first-order corner leaves measurable DC residue at sine-wave rates.
        wet = highpass(wet, cutoff_hz=20.0, sample_rate=sample_rate, order=2)
        if tone_tilt != 0.0:
            wet = _apply_tilt_eq(
                wet,
                sample_rate=sample_rate,
                tilt_db=float(tone_tilt),
                pivot_hz=2_500.0,
            )
        return wet

    wet_signal = _apply_per_channel(input_signal, _process_channel)
    blended = (1.0 - mix) * input_signal + mix * wet_signal
    compensated, _mode_used, _gain_db = _apply_drive_compensation(
        blended,
        reference_signal=input_signal,
        sample_rate=sample_rate,
        compensation_mode=compensation_mode,
    )
    return np.asarray(compensated, dtype=np.float64)


# ---------------------------------------------------------------------------
# apply_transistor — honest stompbox / op-amp memoryless waveshaper
# ---------------------------------------------------------------------------


_TRANSISTOR_PRESETS: dict[str, dict[str, Any]] = {
    "op_amp_clean": {
        "character": "soft_clip",
        "drive": 0.2,
        "mix": 0.25,
    },
    "tube_screamer": {
        "character": "diode",
        "drive": 0.5,
        "bias": 0.1,
        "mix": 0.5,
    },
    "rat_crunch": {
        "character": "op_amp",
        "drive": 0.8,
        "mix": 0.7,
    },
    "fuzz_face": {
        "character": "fuzz",
        "drive": 1.0,
        "bias": 0.2,
        "mix": 0.8,
    },
    "neve_gentle": {
        "character": "soft_clip",
        "drive": 0.3,
        "mix": 0.3,
        "tone": 0.14,
        "highpass_hz": 28.0,
        "multiband": True,
    },
    "kick_heavy": {
        "character": "op_amp",
        "drive": 1.0,
        "mix": 0.62,
        "tone": 0.10,
        "highpass_hz": 28.0,
        "low_crossover_hz": 80.0,
        "high_crossover_hz": 3_600.0,
        "multiband": True,
        "compensation_mode": "rms",
    },
    "snare_bite": {
        "character": "diode",
        "drive": 0.7,
        "bias": 0.1,
        "mix": 0.3,
        "tone": 0.06,
        "highpass_hz": 30.0,
        "low_crossover_hz": 150.0,
        "high_crossover_hz": 5_000.0,
        "multiband": True,
        "compensation_mode": "rms",
    },
}


_TRANSISTOR_VALID_CHARACTERS: frozenset[str] = frozenset(
    {"soft_clip", "diode", "op_amp", "fuzz"}
)


def _transistor_shape(
    x: np.ndarray,
    *,
    character: str,
    drive: float,
    bias: float,
) -> np.ndarray:
    """Dispatch to the requested memoryless character shaper."""
    gain = _drive_knob_gain(drive)
    driven = gain * x + bias
    if character == "soft_clip":
        return _soft_clip_algebraic(driven)
    if character == "diode":
        return _transistor_diode_shape(driven, asymmetry=min(1.0, abs(bias) + 0.5))
    if character == "op_amp":
        return _op_amp_soft_knee(driven, knee=0.12)
    if character == "fuzz":
        return _fuzz_cascade(driven, bias=bias)
    raise ValueError(
        f"character must be one of {sorted(_TRANSISTOR_VALID_CHARACTERS)}, "
        f"got {character!r}"
    )


def apply_transistor(
    signal: np.ndarray,
    *,
    character: str = "soft_clip",
    drive: float = 0.5,
    bias: float = 0.0,
    mix: float = 0.5,
    tone: float = 0.0,
    tone_tilt: float = 0.0,
    highpass_hz: float = 30.0,
    multiband: bool = True,
    low_crossover_hz: float = 120.0,
    high_crossover_hz: float = 5_000.0,
    compensation_mode: str = "auto",
    oversample_factor: int = 4,
    target_thd_pct: float | None = None,
    preset: str | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Honest stompbox / op-amp memoryless waveshaper.

    Four honest character shapes — no physics-cosplay.  This is the
    effect to reach for when you want deliberate pedalboard / op-amp
    colour; for actual tube character, use :func:`apply_tube`; for
    iron-core / magnetics warmth, use :func:`apply_preamp`.

    ``character="soft_clip"``
        ``x / (1 + |x|)`` — algebraic clipper, not tanh.  Clean
        op-amp feel.  Symmetric.
    ``character="diode"``
        Asymmetric Shockley log-soft-clip.  Tube-Screamer-ish.
    ``character="op_amp"``
        Hard clip with soft knee (log-sum-exp smoothing).  RAT territory.
    ``character="fuzz"``
        Two cascaded hard-clips with bias between them.  Fuzz-Face adjacent.
        Most musical at ``drive >= 0.7``; still works at lower drive
        as a gentle clipper.

    Parameters
    ----------
    signal:
        Mono ``(N,)`` or stereo ``(2, N)`` audio.
    character:
        One of ``"soft_clip"`` / ``"diode"`` / ``"op_amp"`` / ``"fuzz"``.
    drive:
        User-facing 0-1 drive knob (``>1`` enters fuzz territory).
    bias:
        Asymmetric DC offset into the shaper (``-1..1``).  The final
        DC-block re-centers the output.
    mix:
        Wet/dry blend (``0..1``).
    tone, tone_tilt:
        Pre-shaper spectral tilt (``tone``, ``-1..1``) and post-shaper
        tilt EQ (``tone_tilt``, dB).
    highpass_hz:
        Pre-shaper highpass corner, ``0`` disables.
    multiband:
        LR4 multiband bypass for bass and air bands.  Defaults to ``True``.
    low_crossover_hz, high_crossover_hz:
        Multiband corner frequencies.
    compensation_mode:
        Loudness compensation: ``"auto"`` / ``"lufs"`` / ``"rms"`` / ``"none"``.
    oversample_factor:
        Internal oversampling factor; auto-upgrades to 8 at ``drive >= 1.5``.
    target_thd_pct:
        When set, solve ``drive`` to hit the target shaper THD on a
        440 Hz sine probe (binary search).  ``drive`` is ignored.
    preset:
        Optional named preset: ``op_amp_clean``, ``tube_screamer``,
        ``rat_crunch``, ``fuzz_face``, ``neve_gentle``, ``kick_heavy``,
        ``snare_bite``.
    sample_rate:
        Audio sample rate.
    """
    if preset is not None:
        resolved_preset = _resolve_effect_params("transistor", {"preset": preset})
        if character == "soft_clip" and "character" in resolved_preset:
            character = str(resolved_preset["character"])
        for key in (
            "drive",
            "bias",
            "mix",
            "tone",
            "tone_tilt",
            "highpass_hz",
            "multiband",
            "low_crossover_hz",
            "high_crossover_hz",
            "compensation_mode",
            "oversample_factor",
        ):
            if key in resolved_preset:
                value = resolved_preset[key]
                if key == "drive" and drive == 0.5:
                    drive = float(value)
                elif key == "bias" and bias == 0.0:
                    bias = float(value)
                elif key == "mix" and mix == 0.5:
                    mix = float(value)
                elif key == "tone" and tone == 0.0:
                    tone = float(value)
                elif key == "tone_tilt" and tone_tilt == 0.0:
                    tone_tilt = float(value)
                elif key == "highpass_hz" and highpass_hz == 30.0:
                    highpass_hz = float(value)
                elif key == "multiband" and multiband is True:
                    multiband = bool(value)
                elif key == "low_crossover_hz" and low_crossover_hz == 120.0:
                    low_crossover_hz = float(value)
                elif key == "high_crossover_hz" and high_crossover_hz == 5_000.0:
                    high_crossover_hz = float(value)
                elif key == "compensation_mode" and compensation_mode == "auto":
                    compensation_mode = str(value)
                elif key == "oversample_factor" and oversample_factor == 4:
                    oversample_factor = int(value)

    if character not in _TRANSISTOR_VALID_CHARACTERS:
        raise ValueError(
            f"character must be one of {sorted(_TRANSISTOR_VALID_CHARACTERS)}, "
            f"got {character!r}"
        )
    if not 0.0 <= mix <= 1.0:
        raise ValueError("mix must be between 0 and 1")
    if drive < 0.0:
        raise ValueError("drive must be non-negative")
    if not -1.0 <= bias <= 1.0:
        raise ValueError("bias must be between -1 and 1")
    if oversample_factor < 1:
        raise ValueError("oversample_factor must be at least 1")
    if highpass_hz < 0.0:
        raise ValueError("highpass_hz must be non-negative")
    if target_thd_pct is not None and target_thd_pct < 0.0:
        raise ValueError("target_thd_pct must be non-negative")

    if target_thd_pct is not None:
        resolved_character = character
        resolved_bias = bias

        def _probe_factory(
            candidate_drive: float,
        ) -> Callable[[np.ndarray], np.ndarray]:
            return lambda x: _transistor_shape(
                x,
                character=resolved_character,
                drive=candidate_drive,
                bias=resolved_bias,
            )

        solved_drive, _measured, _iters = _solve_drive_for_target_thd(
            target_thd_pct=float(target_thd_pct),
            probe_processor_factory=_probe_factory,
        )
        drive = float(solved_drive)

    resolved_oversample = oversample_factor
    if drive >= 1.5 and resolved_oversample < 8:
        resolved_oversample = 8

    input_signal = np.asarray(signal, dtype=np.float64)
    if mix <= 0.0:
        return input_signal.copy()

    def _shape(channel: np.ndarray) -> np.ndarray:
        x = np.asarray(channel, dtype=np.float64)
        if resolved_oversample > 1:
            x_os = resample_poly(x, resolved_oversample, 1)
        else:
            x_os = x
        shaped = _transistor_shape(x_os, character=character, drive=drive, bias=bias)
        if resolved_oversample > 1:
            shaped = resample_poly(shaped, 1, resolved_oversample)
        return np.asarray(shaped[: x.shape[-1]], dtype=np.float64)

    def _process_channel(channel: np.ndarray) -> np.ndarray:
        dry = np.asarray(channel, dtype=np.float64)
        conditioned = dry
        if highpass_hz > 0.0:
            conditioned = highpass(
                conditioned,
                cutoff_hz=highpass_hz,
                sample_rate=sample_rate,
                order=2,
            )
        pre_tilt_db = 5.0 * float(tone)
        if pre_tilt_db != 0.0:
            conditioned = _apply_tilt_eq(
                conditioned,
                sample_rate=sample_rate,
                tilt_db=pre_tilt_db,
                pivot_hz=2_100.0,
            )
        if multiband:
            low, mid, high = _tube_multiband_split(
                conditioned,
                low_crossover_hz=low_crossover_hz,
                high_crossover_hz=high_crossover_hz,
                sample_rate=sample_rate,
            )
            shaped_mid = _shape(mid)
            wet = low + shaped_mid + high
        else:
            wet = _shape(conditioned)
        # Second-order DC block — the new tube/transistor effects can produce
        # substantial asymmetry (Culture-Vulture bias, diode shaper) and a
        # first-order corner leaves measurable DC residue at sine-wave rates.
        wet = highpass(wet, cutoff_hz=20.0, sample_rate=sample_rate, order=2)
        if tone_tilt != 0.0:
            wet = _apply_tilt_eq(
                wet,
                sample_rate=sample_rate,
                tilt_db=float(tone_tilt),
                pivot_hz=2_500.0,
            )
        return wet

    wet_signal = _apply_per_channel(input_signal, _process_channel)
    blended = (1.0 - mix) * input_signal + mix * wet_signal
    compensated, _mode_used, _gain_db = _apply_drive_compensation(
        blended,
        reference_signal=input_signal,
        sample_rate=sample_rate,
        compensation_mode=compensation_mode,
    )
    return np.asarray(compensated, dtype=np.float64)


# ---------------------------------------------------------------------------
# Analog filter (bus/master/voice wrapper around the synth filter palette)
# ---------------------------------------------------------------------------


def _coerce_filter_curve(
    value: float | np.ndarray,
    *,
    n: int,
    name: str,
) -> np.ndarray:
    """Return a length-``n`` float64 curve for a scalar or per-sample filter param.

    A scalar becomes a constant array.  A per-sample array must already match
    the signal length ``n``; mismatched lengths raise ``ValueError`` (fail
    fast — silent resampling would mask wiring bugs in automation curves and
    has bitten us historically).

    Non-finite values are rejected outright because the filter kernels run in
    numba and non-finite inputs silently corrupt state.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(n, float(arr), dtype=np.float64)
    elif arr.ndim == 1:
        if arr.shape[0] == 0:
            raise ValueError(f"{name} per-sample curve is empty")
        if arr.shape[0] != n:
            raise ValueError(
                f"{name} per-sample curve length {arr.shape[0]} must match signal length {n}"
            )
    else:
        raise ValueError(f"{name} must be a scalar or 1-D per-sample array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def apply_analog_filter(
    signal: np.ndarray,
    *,
    cutoff_hz: float | np.ndarray = 1000.0,
    resonance_q: float = 0.707,
    filter_topology: str = "svf",
    mode: str = "lp",
    filter_drive: float = 0.0,
    feedback_amount: float = 0.0,
    feedback_saturation: float = 0.3,
    quality: str = "great",
    mix: float = 1.0,
    bass_compensation: float = 0.5,
    filter_morph: float = 0.0,
    filter_even_harmonics: float = 0.0,
    hpf_cutoff_hz: float = 0.0,
    hpf_resonance_q: float = 0.707,
    k35_feedback_asymmetry: float = 0.5,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Apply the analog-modeled filter palette as a bus/master/voice effect.

    Wraps :func:`code_musics.engines._dsp_utils.apply_filter_oversampled` so
    the eight ZDF/TPT topologies used by the synth engines (``svf``,
    ``ladder``, ``sallen_key``, ``cascade``, ``sem``, ``jupiter``, ``k35``,
    ``diode``) are available on ``EffectSpec`` chains.

    Automation surface
    ------------------
    Only ``cutoff_hz`` accepts per-sample arrays; the underlying numba
    filter kernels take cutoff as a per-sample profile but Q / drive /
    feedback as scalars.  Historically this function silently collapsed
    per-sample Q/drive/feedback arrays to their mean with a warning — that
    let the signature lie about what automation actually did.  Rather than
    pretend per-block sampling makes these "work", the signature now
    matches reality: ``resonance_q``, ``filter_drive``, ``feedback_amount``
    are ``float`` only.  Automate ``cutoff_hz`` for sweeps (see the
    ``analog_filter`` entries in :mod:`code_musics.score`'s EffectSpec
    automation resolution — ``AutomationSpec`` curves on ``cutoff_hz``
    are converted to per-sample arrays at render time).

    Parameters
    ----------
    signal:
        Mono ``(N,)`` or stereo ``(2, N)`` audio.  Mono is upmixed to stereo
        so the output always preserves stereo width; left and right channels
        run through independent filter states (no cross-channel summing).
    cutoff_hz:
        Lowpass/bandpass/highpass/notch corner.  Scalar or per-sample array
        (for automation-driven sweeps).  Clamped to ``[20 Hz, ~Nyquist]``.
        Per-sample arrays must match the signal length exactly.
    resonance_q:
        Filter Q.  ``0.707`` is neutral; ``> 4.0`` enters self-oscillation
        territory on ladder / K35 / diode.
    filter_topology:
        One of ``svf``, ``ladder``, ``sallen_key``, ``cascade``, ``sem``,
        ``jupiter``, ``k35``, ``diode``.  See the engine-side topology docs
        for per-topology character.
    mode:
        ``lp`` (lowpass, default), ``bp`` (bandpass), ``hp`` (highpass),
        or ``notch``.  Topology support matches the synth engines: SVF and
        SEM support all four; the ladder/cascade/jupiter/sallen_key/k35/diode
        kernels honour the requested mode where their topology supports it
        and fall back to lowpass otherwise.
    filter_drive:
        Pre-filter saturation drive ``0 – 1``.
    feedback_amount:
        Post-filter to pre-filter external feedback ``0 – 1``.  The
        ``newton`` solver (selected by ``quality`` modes ``fast`` / ``great``
        / ``divine``) closes this loop implicitly on all eight analog
        topologies — SVF, cascade, SEM, Sallen-Key, ladder, Jupiter, K35,
        and diode.  K35 outer-Newton activates only in LP mode with
        ``filter_drive == 0``; diode outer-Newton requires LP + ``morph == 0``.
        Outside those gates the topologies fall back to unit-delay external
        feedback while the internal Newton solve (where present) still runs.
    feedback_saturation:
        Shaping of the feedback tanh (0 – 1).
    quality:
        ``draft`` / ``fast`` / ``great`` / ``divine``.  Sole axis for solver
        + internal oversampling selection: ``draft`` uses the ADAA solver
        with no oversampling (cheapest), ``fast`` / ``great`` / ``divine``
        use Newton-iterated ZDF with progressively higher iteration caps
        and oversampling factors.  Use ``great`` (default) for
        master-bus / send-bus use; ``divine`` for lead-line focal points;
        ``fast`` or ``draft`` for dense mixes where CPU matters.
    mix:
        Wet/dry blend ``0 – 1``.  Default ``1.0`` (fully wet).  Use lower
        values for parallel filtering (e.g. a K35 scream blended with the
        dry signal).
    bass_compensation:
        Ladder-topology bass-suck offset (0 – 1).  Ignored by other topologies.
    filter_morph:
        SVF/ladder filter-mode morph parameter (0 – 4, wraps).  Ignored by
        other topologies.
    filter_even_harmonics:
        Shaping of the per-stage saturation curve (0 – 1).  Ignored by
        non-driven topologies.
    hpf_cutoff_hz:
        Optional serial 2-pole ZDF highpass corner applied *before* the main
        filter.  Models CS80 / Jupiter-8 dual-filter architecture.  ``0.0``
        disables.
    hpf_resonance_q:
        Q for the serial highpass.  Defaults to Butterworth-neutral ``0.707``.
    k35_feedback_asymmetry:
        Korg MS-20 K35 asymmetric-feedback tuning (0 – 1).  Ignored by other
        topologies.
    sample_rate:
        Audio sample rate.
    """
    audio = _ensure_stereo(signal)
    n_samples = int(audio.shape[-1])
    if n_samples == 0:
        return audio

    resolved_topology = filter_topology.strip().lower()
    if resolved_topology not in _SUPPORTED_FILTER_TOPOLOGIES:
        raise ValueError(
            f"Unsupported filter_topology: {filter_topology!r}. "
            f"Supported: {sorted(_SUPPORTED_FILTER_TOPOLOGIES)}"
        )

    resolved_mode_name = _ANALOG_FILTER_MODE_ALIAS.get(mode.strip().lower())
    if resolved_mode_name is None or resolved_mode_name not in _SUPPORTED_FILTER_MODES:
        raise ValueError(
            f"Unsupported filter mode: {mode!r}. Supported: ['lp', 'bp', 'hp', 'notch']"
        )

    if not 0.0 <= mix <= 1.0:
        raise ValueError("mix must be between 0 and 1")
    if not 0.0 <= feedback_saturation <= 1.0:
        raise ValueError("feedback_saturation must be between 0 and 1")
    if hpf_cutoff_hz < 0.0:
        raise ValueError("hpf_cutoff_hz must be non-negative")

    # Quality is the sole solver/oversample axis — no separate filter_solver
    # kwarg to reconcile.  Previously we let an explicit filter_solver override
    # the quality-mode default, which produced a silent no-op when users
    # combined quality="draft" with the default filter_solver="newton" (ADAA
    # signal path was dropped in favour of a Newton path with zero iterations).
    quality_config = resolve_quality_mode(quality)

    cutoff_curve = _coerce_filter_curve(cutoff_hz, n=n_samples, name="cutoff_hz")
    if not np.isfinite(resonance_q):
        raise ValueError("resonance_q must be finite")
    if not np.isfinite(filter_drive):
        raise ValueError("filter_drive must be finite")
    if not np.isfinite(feedback_amount):
        raise ValueError("feedback_amount must be finite")

    # Upsample the cutoff curve once and share it between L/R.  The Kaiser
    # polyphase resample dominates CPU on long master-bus sweeps; doing it
    # twice (once per channel) wasted hundreds of ms on typical renders.
    cutoff_upsampled = upsample_cutoff_profile(
        cutoff_curve,
        oversample_factor=quality_config.oversample_factor,
        sample_rate=sample_rate,
        n_base=n_samples,
    )

    def _filter_channel(channel: np.ndarray) -> np.ndarray:
        channel_f64 = np.ascontiguousarray(channel, dtype=np.float64)
        return apply_filter_oversampled_preupsampled(
            channel_f64,
            cutoff_profile_upsampled=cutoff_upsampled,
            sample_rate=sample_rate,
            oversample_factor=quality_config.oversample_factor,
            resonance_q=float(resonance_q),
            filter_mode=resolved_mode_name,
            filter_drive=float(filter_drive),
            filter_even_harmonics=filter_even_harmonics,
            filter_topology=resolved_topology,
            bass_compensation=bass_compensation,
            filter_morph=filter_morph,
            hpf_cutoff_hz=hpf_cutoff_hz,
            hpf_resonance_q=hpf_resonance_q,
            feedback_amount=float(feedback_amount),
            feedback_saturation=feedback_saturation,
            filter_solver=quality_config.solver,
            max_newton_iters=max(quality_config.max_newton_iters, 1),
            newton_tolerance=quality_config.newton_tolerance
            if quality_config.newton_tolerance > 0.0
            else 1e-9,
            k35_feedback_asymmetry=k35_feedback_asymmetry,
        )

    wet_left = _filter_channel(audio[0])
    wet_right = _filter_channel(audio[1])
    wet_stereo = np.stack([wet_left, wet_right]).astype(np.float64)

    if mix >= 1.0:
        return wet_stereo
    if mix <= 0.0:
        return audio.astype(np.float64)
    return ((1.0 - mix) * audio + mix * wet_stereo).astype(np.float64)


def apply_stereo_width(
    signal: np.ndarray,
    width: float = 1.0,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Mid/side stereo width control.

    Parameters
    ----------
    signal:
        Stereo input (2, N).  Mono (1-D) passes through unchanged.
    width:
        0.0 = mono, 1.0 = unchanged, 2.0 = exaggerated stereo.
        Values > 1.0 boost the side signal; values < 1.0 narrow toward mono.
    """
    if signal.ndim == 1:
        return signal  # mono passthrough, nothing to widen

    mid = (signal[0] + signal[1]) * 0.5
    side = (signal[0] - signal[1]) * 0.5
    side = side * width
    return np.array([mid + side, mid - side])


_SUPPORTED_EFFECT_AMOUNT_AUTOMATION_TARGETS = {"mix", "wet", "wet_level"}

# Per-effect automation targets that resolve to per-sample param arrays rather
# than wet/dry amount curves.  ``analog_filter`` supports ``cutoff_hz`` natively
# because the numba filter kernel reads cutoff as a per-sample profile; Q,
# drive, and feedback remain scalar-only on that effect (see apply_analog_filter
# docstring).  Add entries here for other effects that want AutomationSpec
# surfaces on their own param names.
_PER_EFFECT_PARAM_AUTOMATION_TARGETS: dict[str, frozenset[str]] = {
    "analog_filter": frozenset({"cutoff_hz"}),
    "clipper": frozenset({"threshold_db", "knee_width_db"}),
}


def _blend_signals(
    dry_signal: np.ndarray,
    wet_signal: np.ndarray,
    amount_curve: np.ndarray,
) -> np.ndarray:
    resolved_amount_curve = np.asarray(amount_curve, dtype=np.float64)
    if resolved_amount_curve.ndim != 1:
        raise ValueError("amount_curve must be one-dimensional")
    dry = np.asarray(dry_signal, dtype=np.float64)
    wet = np.asarray(wet_signal, dtype=np.float64)
    if dry.shape[-1] != wet.shape[-1]:
        raise ValueError("dry and wet signals must share the same length")
    if dry.ndim != wet.ndim:
        dry = _ensure_stereo(dry)
        wet = _ensure_stereo(wet)
    if dry.shape != wet.shape:
        raise ValueError("dry and wet signals must share the same shape")
    if dry.shape[-1] != resolved_amount_curve.size:
        raise ValueError("amount_curve length must match the signal length")
    if np.any((resolved_amount_curve < 0.0) | (resolved_amount_curve > 1.0)):
        raise ValueError("effect amount automation must stay within [0, 1]")

    if dry.ndim == 1:
        return np.asarray(
            ((1.0 - resolved_amount_curve) * dry) + (resolved_amount_curve * wet),
            dtype=np.float64,
        )
    return np.asarray(
        ((1.0 - resolved_amount_curve[np.newaxis, :]) * dry)
        + (resolved_amount_curve[np.newaxis, :] * wet),
        dtype=np.float64,
    )


def _resolve_effect_amount_automation(
    *,
    effect: Any,
    params: dict[str, Any],
    signal_length: int,
    start_time_seconds: float,
    matrix_connections: list[Any] | None = None,
    source_sampling_context: Any | None = None,
) -> tuple[str, np.ndarray] | None:
    automation_specs = list(getattr(effect, "automation", []))
    targeted_specs = [
        spec
        for spec in automation_specs
        if spec.target.kind == "control"
        and spec.target.name in _SUPPORTED_EFFECT_AMOUNT_AUTOMATION_TARGETS
    ]
    matrix_conns = [
        connection
        for connection in (matrix_connections or [])
        if connection.target.kind == "control"
        and connection.target.name in _SUPPORTED_EFFECT_AMOUNT_AUTOMATION_TARGETS
    ]
    if not targeted_specs and not matrix_conns:
        return None

    target_names = {spec.target.name for spec in targeted_specs}
    target_names.update(connection.target.name for connection in matrix_conns)
    if len(target_names) != 1:
        raise ValueError(
            "effect automation must target exactly one of mix, wet, wet_level"
        )

    target_name = next(iter(target_names))
    if target_name not in params:
        raise ValueError(
            f"effect automation target {target_name!r} requires that parameter on the effect"
        )

    signal_times = start_time_seconds + (
        np.arange(signal_length, dtype=np.float64) / SAMPLE_RATE
    )
    amount_curve = apply_control_automation(
        base_value=float(params[target_name]),
        specs=targeted_specs,
        target_name=target_name,
        times=signal_times,
    )
    if matrix_conns:
        if source_sampling_context is None:
            raise ValueError(
                "matrix connections on effect wet require source_sampling_context"
            )
        amount_curve = combine_connections_on_curve(
            base=amount_curve,
            connections=matrix_conns,
            times=signal_times,
            context=source_sampling_context,
        )
    if np.any((amount_curve < 0.0) | (amount_curve > 1.0)):
        raise ValueError(
            f"effect automation target {target_name!r} must stay within [0, 1]"
        )
    return target_name, amount_curve


def _resolve_effect_param_automation(
    *,
    effect: Any,
    params: dict[str, Any],
    signal_length: int,
    start_time_seconds: float,
    matrix_connections: list[Any] | None = None,
    source_sampling_context: Any | None = None,
) -> dict[str, Any]:
    """Resolve per-effect AutomationSpec curves to per-sample param arrays.

    Mutates a *copy* of ``params`` by replacing scalar entries with per-sample
    numpy arrays when an ``AutomationSpec`` on the effect targets a supported
    per-effect param (currently ``cutoff_hz`` on ``analog_filter``).  Unknown
    target names are left for :func:`_resolve_effect_amount_automation` to
    handle (mix/wet/wet_level) — per-effect params and wet/dry amount curves
    are independent surfaces.
    """
    supported_targets = _PER_EFFECT_PARAM_AUTOMATION_TARGETS.get(effect.kind)
    if not supported_targets:
        return params

    automation_specs = list(getattr(effect, "automation", []))
    matrix_conns = list(matrix_connections or [])
    resolved = dict(params)

    signal_times: np.ndarray | None = None
    for target_name in supported_targets:
        targeted_specs = [
            spec
            for spec in automation_specs
            if spec.target.kind == "control" and spec.target.name == target_name
        ]
        targeted_conns = [
            connection
            for connection in matrix_conns
            if connection.target.kind == "control"
            and connection.target.name == target_name
        ]
        if not targeted_specs and not targeted_conns:
            continue

        if target_name not in resolved:
            raise ValueError(
                f"effect automation target {target_name!r} requires that parameter "
                f"on the {effect.kind!r} effect"
            )

        base_scalar = resolved[target_name]
        if isinstance(base_scalar, np.ndarray):
            raise ValueError(
                f"effect param {target_name!r} on {effect.kind!r} cannot mix an "
                "explicit per-sample array with AutomationSpec curves — supply "
                "one or the other"
            )

        if signal_times is None:
            signal_times = start_time_seconds + (
                np.arange(signal_length, dtype=np.float64) / SAMPLE_RATE
            )

        curve = apply_control_automation(
            base_value=float(base_scalar),
            specs=targeted_specs,
            target_name=target_name,
            times=signal_times,
        )
        if targeted_conns:
            if source_sampling_context is None:
                raise ValueError(
                    f"matrix connections on {effect.kind}.{target_name} require "
                    "source_sampling_context"
                )
            curve = combine_connections_on_curve(
                base=curve,
                connections=targeted_conns,
                times=signal_times,
                context=source_sampling_context,
            )
        resolved[target_name] = curve

    return resolved


_PLUGIN_BACKED_EFFECTS: dict[str, str] = {
    "chow_tape": "chow_tape",
    "phaser": "chow_phaser_stereo",
    "tal_chorus_lx": "tal_chorus_lx",
    "tal_reverb2": "tal_reverb2",
    "dragonfly": "dragonfly_plate",
    "airwindows": "airwindows",
    "byod": "byod",
    "chow_centaur": "chow_centaur",
    "valhalla_supermassive": "valhalla_supermassive",
    "valhalla_freq_echo": "valhalla_freq_echo",
    "valhalla_space_mod": "valhalla_space_mod",
    "tdr_kotelnikov": "tdr_kotelnikov",
    "mjuc_jr": "mjuc_jr",
    "fetish": "fetish",
    "lala": "lala",
    "ivgi": "ivgi",
    "brit_channel": "brit_channel",
    "brit_pre": "brit_pre",
    "britpressor": "britpressor",
    "distox": "distox",
    "fet_drive": "fet_drive",
    "kolin": "kolin",
    "laea": "laea",
    "merica": "merica",
    "prebox": "prebox",
    "rare_se": "rare_se",
    "tuba": "tuba",
}


def _is_missing_vst3_plugin(plugin_name: str) -> bool:
    """True when a registered VST3 plugin's files are absent (not installed).

    Returns False for non-VST3 specs (e.g. VST2) so that format-mismatch
    errors propagate normally instead of being silently skipped.
    """
    spec = _get_external_plugin_spec(plugin_name=plugin_name)
    if spec.format != "vst3":
        return False
    return not has_external_plugin(plugin_name)


_SIMPLE_EFFECT_DISPATCH: dict[str, Callable[..., np.ndarray]] = {
    "delay": apply_delay,
    "reverb": apply_reverb,
    "chow_tape": apply_chow_tape,
    "bricasti": apply_bricasti,
    "chorus": apply_chorus,
    "bbd_chorus": apply_bbd_chorus,
    "mod_delay": apply_mod_delay,
    "phaser": apply_phaser,
    "tal_chorus_lx": apply_tal_chorus_lx,
    "tal_reverb2": apply_tal_reverb2,
    "airwindows": apply_airwindows,
    "byod": apply_byod,
    "chow_centaur": apply_chow_centaur,
    "valhalla_supermassive": apply_valhalla_supermassive,
    "valhalla_freq_echo": apply_valhalla_freq_echo,
    "valhalla_space_mod": apply_valhalla_space_mod,
    "tdr_kotelnikov": apply_tdr_kotelnikov,
    "mjuc_jr": apply_mjuc_jr,
    "fetish": apply_fetish,
    "lala": apply_lala,
    "ivgi": apply_ivgi,
    "brit_channel": apply_brit_channel,
    "brit_pre": apply_brit_pre,
    "britpressor": apply_britpressor,
    "distox": apply_distox,
    "fet_drive": apply_fet_drive,
    "kolin": apply_kolin,
    "laea": apply_laea,
    "merica": apply_merica,
    "prebox": apply_prebox,
    "rare_se": apply_rare_se,
    "tuba": apply_tuba,
    "stereo_width": apply_stereo_width,
    "analog_filter": apply_analog_filter,
}


def _apply_effect_with_automation(
    apply_fn: Callable[..., np.ndarray],
    processed: np.ndarray,
    effect_input: np.ndarray,
    params: dict[str, Any],
    effect_amount_automation: tuple[str, np.ndarray] | None,
) -> np.ndarray:
    """Apply a simple effect, handling wet/dry automation when present."""
    if effect_amount_automation is None:
        return apply_fn(processed, **params)
    target_name, amount_curve = effect_amount_automation
    wet_params = dict(params)
    wet_params[target_name] = 1.0
    wet_signal = apply_fn(processed, **wet_params)
    return _blend_signals(effect_input, wet_signal, amount_curve)


@overload
def apply_effect_chain(
    signal: np.ndarray,
    effects: list[Any],
    *,
    sidechain_signals: Mapping[str, np.ndarray] | None = ...,
    signal_name: str | None = ...,
    start_time_seconds: float = ...,
    matrix_connections: list[Any] | None = ...,
    source_sampling_context: Any | None = ...,
    return_analysis: Literal[False] = ...,
    percussive: bool = ...,
) -> np.ndarray: ...


@overload
def apply_effect_chain(
    signal: np.ndarray,
    effects: list[Any],
    *,
    sidechain_signals: Mapping[str, np.ndarray] | None = ...,
    signal_name: str | None = ...,
    start_time_seconds: float = ...,
    matrix_connections: list[Any] | None = ...,
    source_sampling_context: Any | None = ...,
    return_analysis: Literal[True],
    percussive: bool = ...,
) -> tuple[np.ndarray, list[EffectAnalysisEntry]]: ...


def apply_effect_chain(
    signal: np.ndarray,
    effects: list[Any],
    *,
    sidechain_signals: Mapping[str, np.ndarray] | None = None,
    signal_name: str | None = None,
    start_time_seconds: float = 0.0,
    matrix_connections: list[Any] | None = None,
    source_sampling_context: Any | None = None,
    return_analysis: bool = False,
    percussive: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[EffectAnalysisEntry]]:
    """Apply a declarative effect chain to mono or stereo audio."""
    processed = _coerce_signal_layout(signal)
    effect_analysis: list[EffectAnalysisEntry] = []
    for effect_index, effect in enumerate(effects):
        effect_input = np.asarray(processed, dtype=np.float64)
        params = dict(effect.params)
        if effect.kind in {
            "chorus",
            "bbd_chorus",
            "compressor",
            "mod_delay",
            "phaser",
            "drive",
            "preamp",
            "tube",
            "transistor",
            "airwindows",
            "byod",
            "chow_centaur",
        }:
            params = _resolve_effect_params(effect.kind, params)
        params = _resolve_effect_param_automation(
            effect=effect,
            params=params,
            signal_length=effect_input.shape[-1],
            start_time_seconds=start_time_seconds,
            matrix_connections=matrix_connections,
            source_sampling_context=source_sampling_context,
        )
        effect_amount_automation = _resolve_effect_amount_automation(
            effect=effect,
            params=params,
            signal_length=effect_input.shape[-1],
            start_time_seconds=start_time_seconds,
            matrix_connections=matrix_connections,
            source_sampling_context=source_sampling_context,
        )
        native_metrics: dict[str, float | int | str] | None = None
        # Guard: skip plugin-backed effects when the plugin isn't installed,
        # logging a loud warning so agents and humans notice.
        if effect.kind == "dragonfly":
            variant = params.get("variant", "plate")
            required_plugin = f"dragonfly_{variant}"
        else:
            required_plugin = _PLUGIN_BACKED_EFFECTS.get(effect.kind)
        if required_plugin is not None and _is_missing_vst3_plugin(required_plugin):
            logger.warning(
                "SKIPPING effect %r (#%d): plugin %r is not installed on this "
                "machine. The signal passes through unprocessed. Install the "
                "plugin to hear this effect.",
                effect.kind,
                effect_index,
                required_plugin,
            )
            continue
        simple_apply_fn = _SIMPLE_EFFECT_DISPATCH.get(effect.kind)
        if effect.kind == "gate":
            processed = cast(np.ndarray, apply_gate(processed, **params))
        elif simple_apply_fn is not None:
            processed = _apply_effect_with_automation(
                simple_apply_fn,
                processed,
                effect_input,
                params,
                effect_amount_automation,
            )
        elif effect.kind in ("drive", "saturation"):
            raise ValueError(
                f"EffectSpec kind {effect.kind!r} was retired in the tube-"
                "saturation redesign. Migrate to apply_tube (tube saturation), "
                "apply_transistor (stompbox/op-amp), or apply_preamp "
                "(transformer warmth). See docs/synth_api.md or the effect "
                "docstrings."
            )
        elif effect.kind == "preamp":
            if effect_amount_automation is None and return_analysis:
                processed_signal, preamp_metrics = cast(
                    tuple[np.ndarray, dict[str, float | int | str]],
                    apply_preamp(
                        processed,
                        **params,
                        return_analysis=True,
                    ),
                )
                processed = processed_signal
                native_metrics = preamp_metrics
            elif effect_amount_automation is None:
                processed = cast(np.ndarray, apply_preamp(processed, **params))
            else:
                target_name, amount_curve = effect_amount_automation
                wet_params = dict(params)
                wet_params[target_name] = 1.0
                if return_analysis:
                    wet_signal, preamp_metrics = cast(
                        tuple[np.ndarray, dict[str, float | int | str]],
                        apply_preamp(
                            processed,
                            **wet_params,
                            return_analysis=True,
                        ),
                    )
                    native_metrics = preamp_metrics
                else:
                    wet_signal = cast(np.ndarray, apply_preamp(processed, **wet_params))
                processed = _blend_signals(effect_input, wet_signal, amount_curve)
        elif effect.kind == "tube":
            if effect_amount_automation is None:
                processed = apply_tube(processed, **params)
            else:
                target_name, amount_curve = effect_amount_automation
                wet_params = dict(params)
                wet_params[target_name] = 1.0
                wet_signal = apply_tube(processed, **wet_params)
                processed = _blend_signals(effect_input, wet_signal, amount_curve)
        elif effect.kind == "transistor":
            if effect_amount_automation is None:
                processed = apply_transistor(processed, **params)
            else:
                target_name, amount_curve = effect_amount_automation
                wet_params = dict(params)
                wet_params[target_name] = 1.0
                wet_signal = apply_transistor(processed, **wet_params)
                processed = _blend_signals(effect_input, wet_signal, amount_curve)
        elif effect.kind == "eq":
            processed = apply_eq(processed, **params)
        elif effect.kind == "clipper":
            # Clipper already accepts per-sample threshold_db / knee_width_db
            # / mix arrays directly, so _resolve_effect_param_automation
            # (via the _PER_EFFECT_PARAM_AUTOMATION_TARGETS["clipper"] entry)
            # has already substituted per-sample arrays into ``params`` if
            # AutomationSpec / mod-matrix connections targeted those names.
            # Amount-automation on mix/wet/wet_level goes through the
            # amount_curve pathway just like other native effects.
            if effect_amount_automation is None:
                if return_analysis:
                    processed_signal, clipper_metrics = apply_clipper(
                        processed, **params, return_analysis=True
                    )
                    processed = processed_signal
                    native_metrics = clipper_metrics
                else:
                    processed = apply_clipper(processed, **params)
            else:
                target_name, amount_curve = effect_amount_automation
                wet_params = dict(params)
                wet_params[target_name] = 1.0
                if return_analysis:
                    wet_signal, clipper_metrics = apply_clipper(
                        processed, **wet_params, return_analysis=True
                    )
                    native_metrics = clipper_metrics
                else:
                    wet_signal = cast(
                        np.ndarray, apply_clipper(processed, **wet_params)
                    )
                processed = _blend_signals(effect_input, wet_signal, amount_curve)
        elif effect.kind == "limiter":
            if return_analysis:
                processed_signal, limiter_metrics = apply_native_limiter(
                    processed, **params, return_analysis=True
                )
                processed = processed_signal
                native_metrics = limiter_metrics
            else:
                processed = apply_native_limiter(processed, **params)
        elif effect.kind == "compressor":
            sidechain_signal: np.ndarray | None = None
            sidechain_source = params.pop("sidechain_source", None)
            if sidechain_source is not None:
                if (
                    not isinstance(sidechain_source, str)
                    or not sidechain_source.strip()
                ):
                    raise ValueError("sidechain_source must be a non-empty string")
                normalized_sidechain_source = sidechain_source.strip()
                if (
                    signal_name is not None
                    and normalized_sidechain_source == signal_name
                ):
                    sidechain_signal = processed
                else:
                    if sidechain_signals is None:
                        raise ValueError(
                            f"sidechain_source {normalized_sidechain_source!r} is unavailable"
                        )
                    if normalized_sidechain_source not in sidechain_signals:
                        raise ValueError(
                            f"Unknown sidechain_source: {normalized_sidechain_source!r}"
                        )
                    sidechain_signal = _match_signal_length(
                        sidechain_signals[normalized_sidechain_source],
                        processed.shape[-1],
                    )
            if effect_amount_automation is None and return_analysis:
                processed_signal, compressor_metrics = cast(
                    tuple[np.ndarray, dict[str, float | int | str]],
                    apply_compressor(
                        processed,
                        sidechain_signal=sidechain_signal,
                        **params,
                        return_analysis=True,
                    ),
                )
                processed = processed_signal
                native_metrics = compressor_metrics
            elif effect_amount_automation is None:
                processed = cast(
                    np.ndarray,
                    apply_compressor(
                        processed,
                        sidechain_signal=sidechain_signal,
                        **params,
                    ),
                )
            else:
                target_name, amount_curve = effect_amount_automation
                wet_params = dict(params)
                wet_params[target_name] = 1.0
                if return_analysis:
                    wet_signal, compressor_metrics = cast(
                        tuple[np.ndarray, dict[str, float | int | str]],
                        apply_compressor(
                            processed,
                            sidechain_signal=sidechain_signal,
                            **wet_params,
                            return_analysis=True,
                        ),
                    )
                    native_metrics = compressor_metrics
                else:
                    wet_signal = cast(
                        np.ndarray,
                        apply_compressor(
                            processed,
                            sidechain_signal=sidechain_signal,
                            **wet_params,
                        ),
                    )
                processed = _blend_signals(effect_input, wet_signal, amount_curve)
        elif effect.kind == "dragonfly":
            processed = apply_dragonfly(processed, **params)
        elif effect.kind == "plugin":
            plugin_ref = params.get("plugin_name") or params.get("plugin_path")
            if (
                isinstance(plugin_ref, str)
                and plugin_ref in _PLUGIN_SPECS
                and _is_missing_vst3_plugin(plugin_ref)
            ):
                logger.warning(
                    "SKIPPING generic plugin effect %r (#%d): plugin %r is "
                    "not installed on this machine. The signal passes through "
                    "unprocessed.",
                    effect.kind,
                    effect_index,
                    plugin_ref,
                )
                continue
            processed = apply_plugin(processed, **params)
        else:
            raise ValueError(f"Unsupported effect kind: {effect.kind}")
        if return_analysis:
            display_name = effect.kind
            if effect.kind == "plugin":
                display_name = str(
                    params.get("plugin_name")
                    or params.get("plugin_path")
                    or effect.kind
                )
            effect_analysis.append(
                _build_effect_analysis_entry(
                    index=effect_index,
                    kind=effect.kind,
                    display_name=display_name,
                    input_signal=effect_input,
                    output_signal=processed,
                    sample_rate=SAMPLE_RATE,
                    native_metrics=native_metrics,
                    percussive=percussive,
                )
            )
    if return_analysis:
        return processed, effect_analysis
    return processed


def write_wav(
    path: str | Path,
    signal: np.ndarray,
    *,
    bit_depth: int = 24,
    warn_low_peak: bool = True,
) -> None:
    """Write mono (1D) or stereo (2, samples) signal to a WAV file.

    Parameters
    ----------
    path : str | Path
        Output file path.
    signal : np.ndarray
        Audio signal. Shape (samples,) for mono or (2, samples) for stereo.
    bit_depth : int
        Bit depth for WAV output: 16 (dithered PCM), 24 (PCM), or 32 (float).
        Defaults to 24.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    level_diagnostics = measure_signal_levels(signal, sample_rate=SAMPLE_RATE)

    # soundfile expects (samples, channels)
    write_signal = signal.T if signal.ndim == 2 else signal

    subtype_map = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}
    subtype = subtype_map.get(bit_depth)
    if subtype is None:
        raise ValueError(f"Unsupported bit_depth={bit_depth}; choose 16, 24, or 32")

    if bit_depth == 16:
        # Apply TPDF dither for 16-bit quantization
        write_signal = _float_to_int16_pcm(write_signal).astype(np.float64) / 32767.0

    sf.write(str(output_path), write_signal, SAMPLE_RATE, subtype=subtype)
    logger.info(
        "Wrote %s (%d-bit %s, peak %.2f dBFS, true peak %.2f dBFS, integrated loudness %.2f LUFS)",
        output_path,
        bit_depth,
        subtype,
        level_diagnostics.peak_dbfs,
        level_diagnostics.true_peak_dbfs,
        level_diagnostics.integrated_lufs,
    )
    if (
        warn_low_peak
        and np.isfinite(level_diagnostics.peak_dbfs)
        and (level_diagnostics.peak_dbfs <= _LOW_EXPORT_PEAK_WARNING_DBFS)
    ):
        logger.warning(
            "Export peak is unexpectedly low at %.2f dBFS for %s; "
            "the mastering/limiting pipeline may not have driven the file "
            "to the expected ceiling.",
            level_diagnostics.peak_dbfs,
            output_path,
        )
