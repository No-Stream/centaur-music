"""Core synthesis utilities."""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pedalboard
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfilt

logger: logging.Logger = logging.getLogger(__name__)

_PEDALBOARD_CLS: Any = getattr(pedalboard, "Pedalboard")  # noqa: B009
_DELAY_CLS: Any = getattr(pedalboard, "Delay")  # noqa: B009
_REVERB_CLS: Any = getattr(pedalboard, "Reverb")  # noqa: B009
_CONVOLUTION_CLS: Any = getattr(pedalboard, "Convolution")  # noqa: B009

SAMPLE_RATE = 44100


def db_to_amp(db: float) -> float:
    """Convert decibels to a linear amplitude multiplier."""
    return float(10.0 ** (db / 20.0))


def amp_to_db(amp: float) -> float:
    """Convert a linear amplitude multiplier to decibels."""
    if amp <= 0:
        raise ValueError("amp must be positive")
    return float(20.0 * np.log10(amp))


# ---------------------------------------------------------------------------
# Chow Tape Model VST3 (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_CHOW_TAPE_PATH = Path.home() / ".vst3" / "CHOWTapeModel.vst3"
_chow_tape_plugin: Any = None


def _get_chow_tape() -> Any:
    global _chow_tape_plugin
    if _chow_tape_plugin is None:
        os.environ.setdefault("DISPLAY", "")  # avoid X11 crash in headless env
        from pedalboard import (  # noqa: PLC0415
            load_plugin,  # type: ignore[attr-defined]
        )

        if not _CHOW_TAPE_PATH.exists():
            raise FileNotFoundError(
                f"Chow Tape Model VST3 not found at {_CHOW_TAPE_PATH}. "
                "Install from https://github.com/jatinchowdhury18/AnalogTapeModel/releases"
            )
        _chow_tape_plugin = load_plugin(str(_CHOW_TAPE_PATH))
    return _chow_tape_plugin


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
    plugin = _get_chow_tape()
    plugin.tape_drive = drive
    plugin.tape_saturation = saturation
    plugin.tape_bias = bias
    plugin.dry_wet = mix
    plugin.wow_flutter_on_off = False  # pure saturation, no wow/flutter
    plugin.loss_on_off = False  # pure saturation, no tape loss colouring

    stereo_in = _ensure_stereo(signal).astype(np.float32)
    stereo_out = plugin(stereo_in, SAMPLE_RATE)
    return _match_input_layout(_coerce_signal_layout(stereo_out), signal)


def apply_bricasti(
    signal: np.ndarray,
    ir_name: str,
    wet: float = 0.35,
) -> np.ndarray:
    """Convolve a mono or stereo signal with a Bricasti stereo impulse response."""
    ir_l = BRICASTI_IR_DIR / f"{ir_name}, 44K L.wav"
    ir_r = BRICASTI_IR_DIR / f"{ir_name}, 44K R.wav"
    if not ir_l.exists() or not ir_r.exists():
        raise FileNotFoundError(f"IR not found: {ir_name!r} - check BRICASTI_IR_DIR")

    stereo_signal = _ensure_stereo(signal).astype(np.float32)
    left = _PEDALBOARD_CLS([_CONVOLUTION_CLS(str(ir_l), mix=wet)])(
        stereo_signal[0], SAMPLE_RATE
    )
    right = _PEDALBOARD_CLS([_CONVOLUTION_CLS(str(ir_r), mix=wet)])(
        stereo_signal[1], SAMPLE_RATE
    )
    n_samples = stereo_signal.shape[-1]
    return np.stack([left[:n_samples], right[:n_samples]]).astype(np.float64)


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
        else:
            raise ValueError(f"Unsupported effect kind: {effect.kind}")
    return processed


def write_wav(path: str | Path, signal: np.ndarray) -> None:
    """Write mono (1D) or stereo (2, samples) signal to a 16-bit WAV file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    signal = normalize(signal)
    if signal.ndim == 2:
        int16 = (signal.T * 32767).astype(np.int16)
    else:
        int16 = (signal * 32767).astype(np.int16)
    wavfile.write(str(output_path), SAMPLE_RATE, int16)
    logger.info("Wrote %s", output_path)
