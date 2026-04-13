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
from code_musics.engines._dsp_utils import classify_thd, compute_signal_thd

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
) -> np.ndarray:
    """Apply ADSR amplitude envelope."""
    _MIN_ATTACK_S = 0.003  # 3 ms floor — prevents onset clicks
    attack = max(attack, _MIN_ATTACK_S)
    release = max(release, _MIN_ATTACK_S)
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
        envelope[cursor : cursor + attack_samples] = np.linspace(
            0.0, 1.0, attack_samples, endpoint=False
        )
        cursor += attack_samples

    decay_samples = min(n_decay, hold_samples - cursor)
    if decay_samples > 0:
        envelope[cursor : cursor + decay_samples] = np.linspace(
            1.0,
            sustain_level,
            decay_samples,
            endpoint=False,
        )
        cursor += decay_samples

    sustain_samples = hold_samples - cursor
    if sustain_samples > 0:
        envelope[cursor : cursor + sustain_samples] = sustain_level
        cursor += sustain_samples

    release_start_level = float(envelope[cursor - 1]) if cursor > 0 else 0.0

    if release_samples > 0:
        envelope[cursor : cursor + release_samples] = np.linspace(
            release_start_level,
            0.0,
            release_samples,
            endpoint=True,
        )
        cursor += release_samples

    return signal * envelope


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


def _compressor_gain_db(
    *,
    level_db: float,
    threshold_db: float,
    ratio: float,
    knee_db: float,
) -> float:
    if knee_db <= 0.0:
        if level_db <= threshold_db:
            return 0.0
        compressed_db = threshold_db + ((level_db - threshold_db) / ratio)
        return float(compressed_db - level_db)

    lower_knee_db = threshold_db - (knee_db / 2.0)
    upper_knee_db = threshold_db + (knee_db / 2.0)
    if level_db <= lower_knee_db:
        return 0.0
    if level_db >= upper_knee_db:
        compressed_db = threshold_db + ((level_db - threshold_db) / ratio)
        return float(compressed_db - level_db)

    knee_progress_db = level_db - lower_knee_db
    return float(((1.0 / ratio) - 1.0) * (knee_progress_db**2) / (2.0 * knee_db))


def _linked_detector_signal(signal: np.ndarray) -> np.ndarray:
    normalized = _coerce_signal_layout(signal)
    if normalized.ndim == 1:
        return np.abs(normalized)
    return np.max(np.abs(normalized), axis=0)


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


def _build_effect_analysis_entry(
    *,
    index: int,
    kind: str,
    display_name: str,
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    sample_rate: int,
    native_metrics: dict[str, float | int | str] | None = None,
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
    spectral_centroid_delta_hz = _spectral_centroid_hz(
        output_signal,
        sample_rate=sample_rate,
    ) - _spectral_centroid_hz(
        input_signal,
        sample_rate=sample_rate,
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
        "spectral_centroid_delta_hz": round(spectral_centroid_delta_hz, 1),
        "input_thd_pct": round(input_thd_pct, 2),
        "input_thd_character": input_thd_character,
        "output_thd_pct": round(output_thd_pct, 2),
        "output_thd_character": output_thd_character,
        "thd_delta_pct": round(thd_delta_pct, 2),
    }
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

    thd_delta_pct = float(metrics.get("thd_delta_pct", 0.0))
    if thd_delta_pct >= 20.0:
        warnings.append(
            _build_effect_warning(
                severity="severe",
                code="effect_introduced_distortion",
                message="effect introduced heavy harmonic distortion",
                thd_delta_pct=round(thd_delta_pct, 2),
                input_thd_pct=float(metrics.get("input_thd_pct", 0.0)),
                output_thd_pct=float(metrics.get("output_thd_pct", 0.0)),
            )
        )
    elif thd_delta_pct >= 8.0:
        warnings.append(
            _build_effect_warning(
                severity="warning",
                code="effect_introduced_distortion",
                message="effect introduced noticeable harmonic distortion",
                thd_delta_pct=round(thd_delta_pct, 2),
                input_thd_pct=float(metrics.get("input_thd_pct", 0.0)),
                output_thd_pct=float(metrics.get("output_thd_pct", 0.0)),
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
    return_analysis: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int | str]]:
    """Apply a native stereo-linked compressor with optional detector EQ."""
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

    is_rms = normalized_detector_mode == "rms"
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

    def _process_channel(channel: np.ndarray) -> np.ndarray:
        n = len(channel)
        if n == 0:
            return channel.copy()

        # 2 ms RMS envelope for smooth key signal (avoids flicker on zero crossings)
        detect_window = max(1, int(0.002 * sample_rate))
        kernel = np.ones(detect_window, dtype=np.float64) / detect_window
        smoothed_level = np.sqrt(np.convolve(channel**2, kernel, mode="same"))

        # Gate open = level above threshold; hold extends open regions forward
        gate_open = (smoothed_level >= threshold_lin).astype(np.float64)
        if hold_samples > 0:
            hold_kernel = np.ones(hold_samples + 1, dtype=np.float64)
            gate_open = np.convolve(gate_open, hold_kernel, mode="full")[:n]
        gate_target = np.where(gate_open > 0, 1.0, floor_lin)

        gain = _gate_gain_smoothing_loop(
            gate_target, attack_coeff, release_coeff, floor_lin
        )
        return channel * gain

    output = _apply_per_channel(input_signal, _process_channel).astype(np.float64)

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
) -> np.ndarray:
    """Native true-peak lookahead brickwall limiter.

    Oversamples for inter-sample peak detection, applies gain reduction with
    lookahead and smooth attack/release, then downsamples back.
    """
    attack_ms = 0.1

    sig = np.asarray(signal, dtype=np.float64)
    was_mono = sig.ndim == 1
    if was_mono:
        sig = sig[np.newaxis, :]
    if sig.shape[-1] == 0:
        return sig[0] if was_mono else sig
    original_length = sig.shape[1]

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
    active_fraction = float(np.mean(gain_reduction_db < -0.01))
    logger.info(
        f"Native limiter: max gain reduction {max_gr_db:.2f} dB, "
        f"limiting active on {active_fraction * 100.0:.1f}% of samples"
    )
    if max_gr_db < -6.0:
        logger.warning(
            f"Native limiter: heavy limiting ({max_gr_db:.1f} dB max GR) — "
            "input is significantly hotter than threshold"
        )

    if was_mono:
        return limited[0]
    return limited


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

    for _ in range(max_iterations):
        # The native limiter already guarantees peaks are at or below the
        # ceiling, so normalize_true_peak is only needed for the LSP path
        # (where the plugin's output might not land exactly at the ceiling)
        # or as a safety net.  For the native path, skip it — it would boost
        # a high-crest-factor signal back to the ceiling and defeat LUFS
        # convergence.
        if use_lsp_limiter:
            mastered = normalize_true_peak(
                mastered,
                target_peak_dbfs=true_peak_ceiling_dbfs,
                oversample_factor=oversample_factor,
            )
        current_lufs, active_window_fraction = integrated_lufs(
            mastered,
            sample_rate=sample_rate,
        )
        if not np.isfinite(current_lufs) or active_window_fraction <= 0.0:
            break

        loudness_error = target_lufs - current_lufs
        if abs(loudness_error) <= loudness_tolerance_lufs:
            break

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

_SATURATION_PRESETS: dict[str, dict[str, Any]] = {
    "tube_warm": {
        "algorithm": "modern",
        "mode": "tube",
        "drive": 1.0,
        "mix": 0.34,
        "tone": 0.08,
        "fidelity": 0.76,
        "oversample_factor": 4,
        "highpass_hz": 30.0,
        "preserve_lows_hz": 140.0,
        "preserve_highs_hz": 6_500.0,
        "compensation_mode": "auto",
    },
    "iron_soft": {
        "algorithm": "modern",
        "mode": "iron",
        "drive": 0.7,
        "mix": 0.35,
        "tone": -0.06,
        "fidelity": 0.88,
        "oversample_factor": 4,
        "highpass_hz": 26.0,
        "preserve_lows_hz": 180.0,
        "preserve_highs_hz": 5_200.0,
        "compensation_mode": "auto",
    },
    "neve_gentle": {
        "algorithm": "modern",
        "mode": "triode",
        "drive": 0.3,
        "mix": 0.30,
        "tone": 0.14,
        "fidelity": 0.92,
        "oversample_factor": 4,
        "highpass_hz": 28.0,
        "preserve_lows_hz": 120.0,
        "preserve_highs_hz": 7_000.0,
        "compensation_mode": "auto",
    },
    "kick_weight": {
        "algorithm": "modern",
        "mode": "iron",
        "drive": 2.7,
        "mix": 0.42,
        "tone": 0.04,
        "fidelity": 0.46,
        "oversample_factor": 4,
        "highpass_hz": 24.0,
        "preserve_lows_hz": 90.0,
        "preserve_highs_hz": 4_200.0,
        "compensation_mode": "rms",
    },
    "kick_crunch": {
        "algorithm": "modern",
        "mode": "triode",
        "drive": 6.0,
        "mix": 0.62,
        "tone": 0.10,
        "fidelity": 0.26,
        "oversample_factor": 8,
        "highpass_hz": 28.0,
        "preserve_lows_hz": 80.0,
        "preserve_highs_hz": 3_600.0,
        "compensation_mode": "rms",
    },
    "tom_thicken": {
        "algorithm": "modern",
        "mode": "iron",
        "drive": 2.4,
        "mix": 0.32,
        "tone": -0.03,
        "fidelity": 0.62,
        "oversample_factor": 4,
        "highpass_hz": 30.0,
        "preserve_lows_hz": 110.0,
        "preserve_highs_hz": 5_000.0,
        "compensation_mode": "rms",
    },
}

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
    elif effect_kind == "saturation":
        preset_map = _SATURATION_PRESETS
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


def _saturation_curve(
    signal: np.ndarray,
    *,
    drive: float,
    curve: str,
) -> np.ndarray:
    shaped_input = drive * np.asarray(signal, dtype=np.float64)
    if curve == "tube":
        return np.tanh(shaped_input + (0.08 * np.power(shaped_input, 3)))
    if curve == "triode":
        return np.tanh(shaped_input + (0.16 * np.square(shaped_input)))
    if curve == "iron":
        return (2.0 / np.pi) * np.arctan((np.pi / 2.0) * shaped_input)
    raise ValueError(f"Unsupported saturation curve: {curve!r}")


def _asymmetric_saturation_curve(
    signal: np.ndarray,
    *,
    drive: float,
    curve: str,
    asymmetry: float,
    even_harmonics: float,
) -> np.ndarray:
    clipped_even_harmonics = float(np.clip(even_harmonics, 0.0, 1.0))
    symmetric = _saturation_curve(signal, drive=drive, curve=curve)
    if asymmetry == 0.0 or clipped_even_harmonics == 0.0:
        return symmetric

    asymmetric = _saturation_curve(signal + asymmetry, drive=drive, curve=curve)
    asymmetric_zero = _saturation_curve(
        np.zeros(1, dtype=np.float64) + asymmetry,
        drive=drive,
        curve=curve,
    )[0]
    asymmetric = asymmetric - asymmetric_zero
    return ((1.0 - clipped_even_harmonics) * symmetric) + (
        clipped_even_harmonics * asymmetric
    )


def _apply_saturation_compensation(
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


def _apply_saturation_legacy(
    signal: np.ndarray,
    *,
    drive: float,
    mix: float,
    bias: float,
    even_harmonics: float,
    oversample_factor: int,
    highpass_hz: float,
    tone_tilt: float,
    output_lowpass_hz: float,
    compensation_mode: str,
    output_trim_db: float,
    return_analysis: bool,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int | str]]:
    input_signal = np.asarray(signal, dtype=np.float64)
    shaper_hot_sample_count = 0
    total_wet_sample_count = 0

    def _process_channel(channel: np.ndarray) -> np.ndarray:
        nonlocal shaper_hot_sample_count, total_wet_sample_count
        conditioned = highpass(
            channel, cutoff_hz=highpass_hz, sample_rate=SAMPLE_RATE, order=2
        )
        if tone_tilt != 0.0:
            emphasized = lowpass(
                conditioned, cutoff_hz=2_800.0, sample_rate=SAMPLE_RATE, order=1
            )
            conditioned = conditioned + (tone_tilt * emphasized)

        if oversample_factor > 1:
            conditioned = resample_poly(conditioned, oversample_factor, 1)

        biased_drive_signal = drive * (conditioned + bias)
        symmetric_drive_signal = drive * conditioned
        shaped = np.tanh(biased_drive_signal)
        anti_symmetric = np.tanh(symmetric_drive_signal)
        wet_channel = ((1.0 - even_harmonics) * anti_symmetric) + (
            even_harmonics * shaped
        )
        shaper_hot_sample_count += int(
            np.count_nonzero(
                (np.abs(biased_drive_signal) >= 1.0)
                | (np.abs(symmetric_drive_signal) >= 1.0)
            )
        )
        total_wet_sample_count += int(wet_channel.size)

        if oversample_factor > 1:
            wet_channel = resample_poly(wet_channel, 1, oversample_factor)
        wet_channel = wet_channel[: channel.shape[-1]]
        if output_lowpass_hz > 0.0:
            wet_channel = lowpass(
                wet_channel,
                cutoff_hz=output_lowpass_hz,
                sample_rate=SAMPLE_RATE,
                order=2,
            )
        return np.asarray(wet_channel, dtype=np.float64)

    wet_signal = _apply_per_channel(input_signal, _process_channel)
    blended = ((1.0 - mix) * input_signal) + (mix * wet_signal)
    compensated_signal, compensation_mode_used, compensation_gain_db = (
        _apply_saturation_compensation(
            blended,
            reference_signal=input_signal,
            sample_rate=SAMPLE_RATE,
            compensation_mode=compensation_mode,
        )
    )
    processed_signal = np.asarray(
        compensated_signal * db_to_amp(output_trim_db),
        dtype=np.float64,
    )
    if not return_analysis:
        return processed_signal

    thd_pct, thd_character = _saturation_thd(
        lambda x: cast(
            np.ndarray,
            _apply_saturation_legacy(
                x,
                drive=drive,
                mix=1.0,
                bias=bias,
                even_harmonics=even_harmonics,
                oversample_factor=oversample_factor,
                highpass_hz=max(highpass_hz, 1.0),
                tone_tilt=tone_tilt,
                output_lowpass_hz=output_lowpass_hz,
                compensation_mode="none",
                output_trim_db=0.0,
                return_analysis=False,
            ),
        )
    )
    analysis: dict[str, float | int | str] = {
        "algorithm": "legacy",
        "drive": round(float(drive), 2),
        "mix": round(float(mix), 2),
        "even_harmonics": round(float(even_harmonics), 2),
        "shaper_hot_fraction": round(
            float(shaper_hot_sample_count / max(total_wet_sample_count, 1)),
            4,
        ),
        "dc_offset": round(float(np.mean(to_mono_reference(processed_signal))), 6),
        "thd_pct": thd_pct,
        "thd_character": thd_character,
        "compensation_mode_used": compensation_mode_used,
        "compensation_gain_db": round(float(compensation_gain_db), 2),
    }
    return processed_signal, analysis


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


def _apply_saturation_modern(
    signal: np.ndarray,
    *,
    drive: float,
    mix: float,
    mode: str,
    tone: float,
    fidelity: float,
    bias: float,
    even_harmonics: float,
    oversample_factor: int,
    highpass_hz: float,
    tone_tilt: float,
    output_lowpass_hz: float,
    preserve_lows_hz: float,
    preserve_highs_hz: float,
    compensation_mode: str,
    output_trim_db: float,
    return_analysis: bool,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int | str]]:
    if not 0.0 <= fidelity <= 1.0:
        raise ValueError("fidelity must be between 0 and 1")

    profile_map: dict[str, dict[str, float | str]] = {
        "tube": {
            "curve1": "tube",
            "curve2": "tube",
            "stage1_gain": 0.68,
            "stage2_gain": 0.39,
            "asymmetry": 0.035,
            "even": 0.24,
            "sag": 0.10,
            "interstage_tilt_db": -0.6,
            "low_blend": 0.28,
            "high_blend": 0.52,
        },
        "triode": {
            "curve1": "triode",
            "curve2": "tube",
            "stage1_gain": 0.60,
            "stage2_gain": 0.46,
            "asymmetry": 0.055,
            "even": 0.30,
            "sag": 0.14,
            "interstage_tilt_db": 0.4,
            "low_blend": 0.22,
            "high_blend": 0.44,
        },
        "iron": {
            "curve1": "iron",
            "curve2": "tube",
            "stage1_gain": 0.54,
            "stage2_gain": 0.31,
            "asymmetry": 0.022,
            "even": 0.12,
            "sag": 0.06,
            "interstage_tilt_db": -1.0,
            "low_blend": 0.52,
            "high_blend": 0.64,
        },
    }
    normalized_mode = mode.strip().lower()
    if normalized_mode not in profile_map:
        raise ValueError("mode must be 'tube', 'triode', or 'iron'")
    profile = profile_map[normalized_mode]

    resolved_oversample_factor = oversample_factor
    if drive >= 1.7 and resolved_oversample_factor < 8:
        resolved_oversample_factor = 8

    input_signal = np.asarray(signal, dtype=np.float64)
    shaper_hot_sample_count = 0
    total_wet_sample_count = 0

    resolved_tone = float(np.clip(tone + (4.0 * tone_tilt), -1.0, 1.0))
    resolved_even_harmonics = float(
        np.clip((0.55 * float(profile["even"])) + (0.45 * even_harmonics), 0.0, 1.0)
    )
    resolved_asymmetry = float(profile["asymmetry"]) + bias
    oversampled_sample_rate = SAMPLE_RATE * resolved_oversample_factor

    def _process_channel(channel: np.ndarray) -> np.ndarray:
        nonlocal shaper_hot_sample_count, total_wet_sample_count
        dry_channel = np.asarray(channel, dtype=np.float64)
        conditioned = highpass(
            dry_channel,
            cutoff_hz=highpass_hz,
            sample_rate=SAMPLE_RATE,
            order=2,
        )
        pre_tilt_db = (5.0 * resolved_tone) + float(profile["interstage_tilt_db"])
        if pre_tilt_db != 0.0:
            conditioned = _apply_tilt_eq(
                conditioned,
                sample_rate=SAMPLE_RATE,
                tilt_db=pre_tilt_db,
                pivot_hz=2_100.0,
            )

        if resolved_oversample_factor > 1:
            conditioned = resample_poly(conditioned, resolved_oversample_factor, 1)

        envelope = _envelope_follower(
            conditioned,
            sample_rate=oversampled_sample_rate,
            attack_ms=6.0,
            release_ms=90.0,
        )
        sag_amount = float(profile["sag"]) * (0.65 + (0.35 * drive))
        sag_gain = 1.0 / (1.0 + (sag_amount * envelope))

        stage1_drive = 1.0 + (float(profile["stage1_gain"]) * drive)
        stage1_input = conditioned * stage1_drive * sag_gain
        stage1 = _asymmetric_saturation_curve(
            stage1_input,
            drive=1.0,
            curve=cast(str, profile["curve1"]),
            asymmetry=resolved_asymmetry * (0.7 + (0.3 * drive)),
            even_harmonics=resolved_even_harmonics,
        )
        shaper_hot_sample_count += int(np.count_nonzero(np.abs(stage1_input) >= 1.0))
        total_wet_sample_count += int(stage1_input.size)

        interstage = _dc_block(
            stage1,
            sample_rate=oversampled_sample_rate,
            cutoff_hz=8.0,
        )
        interstage = highpass(
            interstage,
            cutoff_hz=max(highpass_hz * 0.75, 10.0),
            sample_rate=oversampled_sample_rate,
            order=1,
        )
        interstage = lowpass(
            interstage,
            cutoff_hz=16_000.0 - (3_500.0 * (1.0 - fidelity)),
            sample_rate=oversampled_sample_rate,
            order=1,
        )
        if resolved_tone != 0.0:
            interstage = _apply_tilt_eq(
                interstage,
                sample_rate=oversampled_sample_rate,
                tilt_db=2.5 * resolved_tone,
                pivot_hz=3_000.0,
            )

        stage2_drive = 1.0 + (float(profile["stage2_gain"]) * drive)
        stage2_input = interstage * stage2_drive
        stage2 = _asymmetric_saturation_curve(
            stage2_input,
            drive=1.0,
            curve=cast(str, profile["curve2"]),
            asymmetry=resolved_asymmetry * 0.55,
            even_harmonics=min(1.0, resolved_even_harmonics + 0.08),
        )
        shaper_hot_sample_count += int(np.count_nonzero(np.abs(stage2_input) >= 1.0))
        total_wet_sample_count += int(stage2_input.size)

        wet_channel = stage2
        if resolved_oversample_factor > 1:
            wet_channel = resample_poly(wet_channel, 1, resolved_oversample_factor)
        wet_channel = np.asarray(wet_channel[: dry_channel.shape[-1]], dtype=np.float64)

        if output_lowpass_hz > 0.0:
            wet_channel = lowpass(
                wet_channel,
                cutoff_hz=output_lowpass_hz,
                sample_rate=SAMPLE_RATE,
                order=1,
            )

        wet_channel = _dc_block(wet_channel, sample_rate=SAMPLE_RATE, cutoff_hz=12.0)

        low_blend = float(profile["low_blend"]) * fidelity
        high_blend = float(profile["high_blend"]) * fidelity
        if preserve_lows_hz > 0.0 and low_blend > 0.0:
            dry_low = lowpass(
                dry_channel,
                cutoff_hz=preserve_lows_hz,
                sample_rate=SAMPLE_RATE,
                order=2,
            )
            wet_low = lowpass(
                wet_channel,
                cutoff_hz=preserve_lows_hz,
                sample_rate=SAMPLE_RATE,
                order=2,
            )
            wet_channel = wet_channel + (low_blend * (dry_low - wet_low))
        if preserve_highs_hz > 0.0 and high_blend > 0.0:
            dry_high = highpass(
                dry_channel,
                cutoff_hz=preserve_highs_hz,
                sample_rate=SAMPLE_RATE,
                order=2,
            )
            wet_high = highpass(
                wet_channel,
                cutoff_hz=preserve_highs_hz,
                sample_rate=SAMPLE_RATE,
                order=2,
            )
            wet_channel = wet_channel + (high_blend * (dry_high - wet_high))
        return _dc_block(wet_channel, sample_rate=SAMPLE_RATE, cutoff_hz=12.0)

    wet_signal = _apply_per_channel(input_signal, _process_channel)
    blended = ((1.0 - mix) * input_signal) + (mix * wet_signal)
    compensated_signal, compensation_mode_used, compensation_gain_db = (
        _apply_saturation_compensation(
            blended,
            reference_signal=input_signal,
            sample_rate=SAMPLE_RATE,
            compensation_mode=compensation_mode,
        )
    )
    processed_signal = np.asarray(
        compensated_signal * db_to_amp(output_trim_db),
        dtype=np.float64,
    )
    if not return_analysis:
        return processed_signal

    thd_pct, thd_character = _saturation_thd(
        lambda x: cast(
            np.ndarray,
            _apply_saturation_modern(
                x,
                drive=drive,
                mix=1.0,
                mode=normalized_mode,
                tone=tone,
                fidelity=fidelity,
                bias=bias,
                even_harmonics=even_harmonics,
                oversample_factor=resolved_oversample_factor,
                highpass_hz=max(highpass_hz, 10.0),
                tone_tilt=tone_tilt,
                output_lowpass_hz=output_lowpass_hz,
                preserve_lows_hz=0.0,
                preserve_highs_hz=0.0,
                compensation_mode="none",
                output_trim_db=0.0,
                return_analysis=False,
            ),
        )
    )
    analysis: dict[str, float | int | str] = {
        "algorithm": "modern",
        "mode": normalized_mode,
        "drive": round(float(drive), 2),
        "mix": round(float(mix), 2),
        "fidelity": round(float(fidelity), 2),
        "tone": round(float(tone), 2),
        "shaper_hot_fraction": round(
            float(shaper_hot_sample_count / max(total_wet_sample_count, 1)),
            4,
        ),
        "dc_offset": round(float(np.mean(to_mono_reference(processed_signal))), 6),
        "thd_pct": thd_pct,
        "thd_character": thd_character,
        "compensation_mode_used": compensation_mode_used,
        "compensation_gain_db": round(float(compensation_gain_db), 2),
    }
    return processed_signal, analysis


def apply_saturation(
    signal: np.ndarray,
    drive: float = 1.18,
    mix: float = 0.34,
    *,
    algorithm: str = "modern",
    mode: str = "tube",
    tone: float = 0.0,
    fidelity: float = 0.7,
    bias: float = 0.11,
    even_harmonics: float = 0.18,
    oversample_factor: int = 4,
    highpass_hz: float = 30.0,
    tone_tilt: float = 0.10,
    output_lowpass_hz: float = 0.0,
    preserve_lows_hz: float = 120.0,
    preserve_highs_hz: float = 6_000.0,
    compensation_mode: str = "auto",
    output_trim_db: float = 0.0,
    return_analysis: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int | str]]:
    """Apply subtle analog-style saturation to mono or stereo signal."""
    if not 0.0 <= mix <= 1.0:
        raise ValueError("mix must be between 0 and 1")
    if drive <= 0:
        raise ValueError("drive must be positive")
    if oversample_factor < 1:
        raise ValueError("oversample_factor must be at least 1")
    if highpass_hz < 0.0:
        raise ValueError("highpass_hz must be non-negative")
    if preserve_lows_hz < 0.0 or preserve_highs_hz < 0.0:
        raise ValueError("preserve_lows_hz and preserve_highs_hz must be non-negative")

    resolved_algorithm = algorithm.strip().lower()
    if resolved_algorithm == "legacy":
        return _apply_saturation_legacy(
            signal,
            drive=drive,
            mix=mix,
            bias=bias,
            even_harmonics=even_harmonics,
            oversample_factor=oversample_factor,
            highpass_hz=highpass_hz,
            tone_tilt=tone_tilt,
            output_lowpass_hz=output_lowpass_hz,
            compensation_mode=compensation_mode,
            output_trim_db=output_trim_db,
            return_analysis=return_analysis,
        )
    if resolved_algorithm != "modern":
        raise ValueError("algorithm must be 'modern' or 'legacy'")
    return _apply_saturation_modern(
        signal,
        drive=drive,
        mix=mix,
        mode=mode,
        tone=tone,
        fidelity=fidelity,
        bias=bias,
        even_harmonics=even_harmonics,
        oversample_factor=oversample_factor,
        highpass_hz=highpass_hz,
        tone_tilt=tone_tilt,
        output_lowpass_hz=output_lowpass_hz,
        preserve_lows_hz=preserve_lows_hz,
        preserve_highs_hz=preserve_highs_hz,
        compensation_mode=compensation_mode,
        output_trim_db=output_trim_db,
        return_analysis=return_analysis,
    )


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
        _apply_saturation_compensation(
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


_SUPPORTED_EFFECT_AMOUNT_AUTOMATION_TARGETS = {"mix", "wet", "wet_level"}


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
) -> tuple[str, np.ndarray] | None:
    automation_specs = list(getattr(effect, "automation", []))
    targeted_specs = [
        spec
        for spec in automation_specs
        if spec.target.kind == "control"
        and spec.target.name in _SUPPORTED_EFFECT_AMOUNT_AUTOMATION_TARGETS
    ]
    if not targeted_specs:
        return None

    target_names = {spec.target.name for spec in targeted_specs}
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
    if np.any((amount_curve < 0.0) | (amount_curve > 1.0)):
        raise ValueError(
            f"effect automation target {target_name!r} must stay within [0, 1]"
        )
    return target_name, amount_curve


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
    return_analysis: Literal[False] = ...,
) -> np.ndarray: ...


@overload
def apply_effect_chain(
    signal: np.ndarray,
    effects: list[Any],
    *,
    sidechain_signals: Mapping[str, np.ndarray] | None = ...,
    signal_name: str | None = ...,
    start_time_seconds: float = ...,
    return_analysis: Literal[True],
) -> tuple[np.ndarray, list[EffectAnalysisEntry]]: ...


def apply_effect_chain(
    signal: np.ndarray,
    effects: list[Any],
    *,
    sidechain_signals: Mapping[str, np.ndarray] | None = None,
    signal_name: str | None = None,
    start_time_seconds: float = 0.0,
    return_analysis: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[EffectAnalysisEntry]]:
    """Apply a declarative effect chain to mono or stereo audio."""
    processed = _coerce_signal_layout(signal)
    effect_analysis: list[EffectAnalysisEntry] = []
    for effect_index, effect in enumerate(effects):
        effect_input = np.asarray(processed, dtype=np.float64)
        params = dict(effect.params)
        if effect.kind in {
            "chorus",
            "compressor",
            "mod_delay",
            "phaser",
            "saturation",
            "preamp",
            "airwindows",
            "byod",
            "chow_centaur",
        }:
            params = _resolve_effect_params(effect.kind, params)
        effect_amount_automation = _resolve_effect_amount_automation(
            effect=effect,
            params=params,
            signal_length=effect_input.shape[-1],
            start_time_seconds=start_time_seconds,
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
        elif effect.kind == "saturation":
            if effect_amount_automation is None and return_analysis:
                processed_signal, saturation_metrics = cast(
                    tuple[np.ndarray, dict[str, float | int | str]],
                    apply_saturation(
                        processed,
                        **params,
                        return_analysis=True,
                    ),
                )
                processed = processed_signal
                native_metrics = saturation_metrics
            elif effect_amount_automation is None:
                processed = cast(np.ndarray, apply_saturation(processed, **params))
            else:
                target_name, amount_curve = effect_amount_automation
                wet_params = dict(params)
                wet_params[target_name] = 1.0
                if return_analysis:
                    wet_signal, saturation_metrics = cast(
                        tuple[np.ndarray, dict[str, float | int | str]],
                        apply_saturation(
                            processed,
                            **wet_params,
                            return_analysis=True,
                        ),
                    )
                    native_metrics = saturation_metrics
                else:
                    wet_signal = cast(
                        np.ndarray, apply_saturation(processed, **wet_params)
                    )
                processed = _blend_signals(effect_input, wet_signal, amount_curve)
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
        elif effect.kind == "eq":
            processed = apply_eq(processed, **params)
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
    if np.isfinite(level_diagnostics.peak_dbfs) and (
        level_diagnostics.peak_dbfs <= _LOW_EXPORT_PEAK_WARNING_DBFS
    ):
        logger.warning(
            "Export peak is unexpectedly low at %.2f dBFS for %s; "
            "the mastering/limiting pipeline may not have driven the file "
            "to the expected ceiling.",
            level_diagnostics.peak_dbfs,
            output_path,
        )
