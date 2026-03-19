"""Core synthesis utilities."""

from __future__ import annotations

import ctypes
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pedalboard
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt, tf2sos

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


_LOW_EXPORT_PEAK_WARNING_DBFS = -3.0


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


# ---------------------------------------------------------------------------
# Chow Tape Model VST3 (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_PLUGIN_SPECS: dict[str, ExternalPluginSpec] = {
    "lsp_compressor_stereo": ExternalPluginSpec(
        name="lsp_compressor_stereo",
        path=Path.home() / ".vst3" / "lsp-plugins.vst3",
        format="vst3",
        bundle_plugin_name="Compressor Stereo",
        preload_libraries=(
            Path.home() / ".local" / "lib" / "lsp-runtime" / "libpixman-1.so.0.38.4",
            Path.home() / ".local" / "lib" / "lsp-runtime" / "libxcb-render.so.0.0.0",
            Path.home() / ".local" / "lib" / "lsp-runtime" / "libcairo.so.2.11600.0",
        ),
    ),
    "lsp_limiter_stereo": ExternalPluginSpec(
        name="lsp_limiter_stereo",
        path=Path.home() / ".vst3" / "lsp-plugins.vst3",
        format="vst3",
        bundle_plugin_name="Limiter Stereo",
        preload_libraries=(
            Path.home() / ".local" / "lib" / "lsp-runtime" / "libpixman-1.so.0.38.4",
            Path.home() / ".local" / "lib" / "lsp-runtime" / "libxcb-render.so.0.0.0",
            Path.home() / ".local" / "lib" / "lsp-runtime" / "libcairo.so.2.11600.0",
        ),
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
}
_loaded_external_plugins: dict[tuple[str, str, Path, str | None], Any] = {}


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


def _load_external_plugin(
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
    plugin = _load_external_plugin(
        plugin_name=plugin_name,
        plugin_path=plugin_path,
        plugin_format=plugin_format,
        host=host,
    )
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


BRICASTI_IR_DIR = Path(
    "/mnt/c/Music Production/Convolution Impulses"
    "/Samplicity - Bricasti IRs version 2023-10"
    "/Samplicity - Bricasti IRs version 2023-10, left-right files, 44.1 Khz"
)


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
    """Apply attenuation when needed so the estimated true peak stays below ceiling."""
    true_peak_amplitude = estimate_true_peak_amplitude(
        signal,
        oversample_factor=oversample_factor,
    )
    if true_peak_amplitude <= 0:
        return np.asarray(signal, dtype=np.float64)

    target_amplitude = db_to_amp(target_peak_dbfs)
    required_gain = target_amplitude / true_peak_amplitude
    if required_gain >= 1.0:
        return np.asarray(signal, dtype=np.float64)
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
            "dry_level": 100.0,
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

    mastered = np.asarray(signal, dtype=np.float64)
    if mastered.size == 0:
        return MasteringResult(
            signal=mastered,
            integrated_lufs=float("-inf"),
            true_peak_dbfs=float("-inf"),
        )

    if not has_external_plugin("lsp_limiter_stereo"):
        raise FileNotFoundError(
            "LSP limiter is required for export mastering but is not available."
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
    limiter_output_gain_db = 0.0
    mastered = apply_lsp_limiter(
        mastered,
        threshold_db=true_peak_ceiling_dbfs,
        input_gain_db=limiter_input_gain_db,
        output_gain_db=limiter_output_gain_db,
    )

    for _ in range(max_iterations):
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

        limiter_input_gain_db += loudness_error
        mastered = apply_lsp_limiter(
            mastered,
            threshold_db=true_peak_ceiling_dbfs,
            input_gain_db=limiter_input_gain_db,
            output_gain_db=limiter_output_gain_db,
        )

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

_SATURATION_PRESETS: dict[str, dict[str, float | int | bool]] = {
    "tube_warm": {
        "drive": 1.18,
        "mix": 0.34,
        "bias": 0.11,
        "even_harmonics": 0.18,
        "oversample_factor": 4,
        "highpass_hz": 30.0,
        "tone_tilt": 0.10,
        "output_lowpass_hz": 13_500.0,
        "compensation": True,
    },
    "iron_soft": {
        "drive": 1.22,
        "mix": 0.38,
        "bias": 0.07,
        "even_harmonics": 0.13,
        "oversample_factor": 4,
        "highpass_hz": 26.0,
        "tone_tilt": -0.08,
        "output_lowpass_hz": 12_000.0,
        "compensation": True,
    },
    "neve_gentle": {
        "drive": 1.28,
        "mix": 0.36,
        "bias": 0.09,
        "even_harmonics": 0.16,
        "oversample_factor": 4,
        "highpass_hz": 28.0,
        "tone_tilt": 0.16,
        "output_lowpass_hz": 12_500.0,
        "compensation": True,
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


def apply_saturation(
    signal: np.ndarray,
    drive: float = 1.18,
    mix: float = 0.34,
    bias: float = 0.11,
    even_harmonics: float = 0.18,
    oversample_factor: int = 4,
    highpass_hz: float = 30.0,
    tone_tilt: float = 0.10,
    output_lowpass_hz: float = 13_500.0,
    compensation: bool = True,
) -> np.ndarray:
    """Apply subtle analog-style saturation to mono or stereo signal."""
    if not 0.0 <= mix <= 1.0:
        raise ValueError("mix must be between 0 and 1")
    if drive <= 0:
        raise ValueError("drive must be positive")
    if oversample_factor < 1:
        raise ValueError("oversample_factor must be at least 1")

    input_signal = np.asarray(signal, dtype=np.float64)
    input_peak = np.max(np.abs(input_signal))

    def _process_channel(channel: np.ndarray) -> np.ndarray:
        conditioned = highpass(
            channel, cutoff_hz=highpass_hz, sample_rate=SAMPLE_RATE, order=2
        )
        if tone_tilt != 0:
            emphasized = lowpass(
                conditioned, cutoff_hz=2_800.0, sample_rate=SAMPLE_RATE, order=1
            )
            conditioned = conditioned + (tone_tilt * emphasized)

        if oversample_factor > 1:
            conditioned = resample_poly(conditioned, oversample_factor, 1)

        shaped = np.tanh(drive * (conditioned + bias))
        anti_symmetric = np.tanh(drive * conditioned)
        wet_channel = ((1.0 - even_harmonics) * anti_symmetric) + (
            even_harmonics * shaped
        )

        if oversample_factor > 1:
            wet_channel = resample_poly(wet_channel, 1, oversample_factor)
        wet_channel = wet_channel[: channel.shape[-1]]
        wet_channel = lowpass(
            wet_channel,
            cutoff_hz=output_lowpass_hz,
            sample_rate=SAMPLE_RATE,
            order=2,
        )
        return wet_channel

    wet_signal = _apply_per_channel(input_signal, _process_channel)
    blended = ((1.0 - mix) * input_signal) + (mix * wet_signal)

    if compensation:
        output_peak = np.max(np.abs(blended))
        if output_peak > 0 and input_peak > 0:
            blended = blended * (input_peak / output_peak)

    return blended.astype(np.float64)


def apply_effect_chain(
    signal: np.ndarray,
    effects: list[Any],
) -> np.ndarray:
    """Apply a declarative effect chain to mono or stereo audio."""
    processed = _coerce_signal_layout(signal)
    for effect in effects:
        params = dict(effect.params)
        if effect.kind in {"chorus", "saturation"}:
            params = _resolve_effect_params(effect.kind, params)
        if effect.kind == "delay":
            processed = apply_delay(processed, **params)
        elif effect.kind == "reverb":
            processed = apply_reverb(processed, **params)
        elif effect.kind == "chow_tape":
            processed = apply_chow_tape(processed, **params)
        elif effect.kind == "bricasti":
            processed = apply_bricasti(processed, **params)
        elif effect.kind == "chorus":
            processed = apply_chorus(processed, **params)
        elif effect.kind == "saturation":
            processed = apply_saturation(processed, **params)
        elif effect.kind == "tal_chorus_lx":
            processed = apply_tal_chorus_lx(processed, **params)
        elif effect.kind == "tal_reverb2":
            processed = apply_tal_reverb2(processed, **params)
        elif effect.kind == "dragonfly":
            processed = apply_dragonfly(processed, **params)
        elif effect.kind == "plugin":
            processed = apply_plugin(processed, **params)
        else:
            raise ValueError(f"Unsupported effect kind: {effect.kind}")
    return processed


def write_wav(path: str | Path, signal: np.ndarray) -> None:
    """Write mono (1D) or stereo (2, samples) signal to a 16-bit WAV file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    level_diagnostics = measure_signal_levels(signal, sample_rate=SAMPLE_RATE)

    if signal.ndim == 2:
        int16 = _float_to_int16_pcm(signal.T)
    else:
        int16 = _float_to_int16_pcm(signal)
    wavfile.write(str(output_path), SAMPLE_RATE, int16)
    logger.info(
        "Wrote %s (peak %.2f dBFS, true peak %.2f dBFS, integrated loudness %.2f LUFS)",
        output_path,
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
