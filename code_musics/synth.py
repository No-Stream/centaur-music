"""Core synthesis utilities."""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from pedalboard import Convolution, Delay, Pedalboard, Reverb
from scipy.signal import butter, sosfilt
from scipy.io import wavfile

logger: logging.Logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100

# ---------------------------------------------------------------------------
# Chow Tape Model VST3 (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_CHOW_TAPE_PATH = Path.home() / ".vst3" / "CHOWTapeModel.vst3"
_chow_tape_plugin: Any = None


def _get_chow_tape() -> Any:
    global _chow_tape_plugin
    if _chow_tape_plugin is None:
        os.environ.setdefault("DISPLAY", "")  # avoid X11 crash in headless env
        from pedalboard import load_plugin  # noqa: PLC0415
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
) -> np.ndarray:
    """Apply ADSR amplitude envelope."""
    n = len(signal)
    n_attack = int(attack * sample_rate)
    n_decay = int(decay * sample_rate)
    n_release = int(release * sample_rate)
    n_sustain = max(0, n - n_attack - n_decay - n_release)

    envelope = np.concatenate(
        [
            np.linspace(0.0, 1.0, n_attack),
            np.linspace(1.0, sustain_level, n_decay),
            np.full(n_sustain, sustain_level),
            np.linspace(sustain_level, 0.0, n_release),
        ]
    )
    return signal * envelope[:n]


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


def at_sample_rate(signal: np.ndarray, offset_seconds: float, sample_rate: int) -> np.ndarray:
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


def lowpass(signal: np.ndarray, cutoff_hz: float, sample_rate: int, order: int = 2) -> np.ndarray:
    """Apply a stable low-pass filter to a mono signal."""
    nyquist = sample_rate / 2.0
    if cutoff_hz <= 0:
        return np.zeros_like(signal)
    if cutoff_hz >= nyquist * 0.995:
        return signal
    sos = butter(order, cutoff_hz / nyquist, btype="lowpass", output="sos")
    return sosfilt(sos, signal)


def highpass(signal: np.ndarray, cutoff_hz: float, sample_rate: int, order: int = 2) -> np.ndarray:
    """Apply a stable high-pass filter to a mono signal."""
    nyquist = sample_rate / 2.0
    if cutoff_hz <= 0:
        return signal
    if cutoff_hz >= nyquist * 0.995:
        return np.zeros_like(signal)
    sos = butter(order, cutoff_hz / nyquist, btype="highpass", output="sos")
    return sosfilt(sos, signal)


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
    """Apply pedalboard Delay to a mono signal, returning mono."""
    board = Pedalboard(
        [Delay(delay_seconds=delay_seconds, feedback=feedback, mix=mix)]
    )
    return board(signal.astype(np.float32), SAMPLE_RATE)


def apply_reverb(
    signal: np.ndarray,
    room_size: float = 0.75,
    damping: float = 0.4,
    wet_level: float = 0.25,
) -> np.ndarray:
    """Apply pedalboard's built-in algorithmic reverb to a mono signal."""
    board = Pedalboard(
        [
            Reverb(
                room_size=room_size,
                damping=damping,
                wet_level=wet_level,
                dry_level=1.0 - wet_level,
            )
        ]
    )
    return board(signal.astype(np.float32), SAMPLE_RATE)


def apply_chow_tape(
    signal: np.ndarray,
    drive: float = 0.5,
    saturation: float = 0.5,
    bias: float = 0.5,
    mix: float = 70.0,
) -> np.ndarray:
    """Apply Chow Tape Model tape saturation to a mono signal, returning mono.

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
    plugin.loss_on_off = False         # pure saturation, no tape loss colouring

    # Plugin expects stereo input; duplicate mono channel, then average back down
    stereo_in = np.stack([signal.astype(np.float32), signal.astype(np.float32)])
    stereo_out = plugin(stereo_in, SAMPLE_RATE)
    mono_out = stereo_out.mean(axis=0) if stereo_out.ndim == 2 else stereo_out
    return mono_out.astype(np.float64)


def apply_bricasti(
    signal: np.ndarray,
    ir_name: str,
    wet: float = 0.35,
) -> np.ndarray:
    """Convolve a mono signal with a Bricasti stereo impulse response."""
    ir_l = BRICASTI_IR_DIR / f"{ir_name}, 44K L.wav"
    ir_r = BRICASTI_IR_DIR / f"{ir_name}, 44K R.wav"
    if not ir_l.exists() or not ir_r.exists():
        raise FileNotFoundError(f"IR not found: {ir_name!r} - check BRICASTI_IR_DIR")

    mono_signal = signal.astype(np.float32)
    left = Pedalboard([Convolution(str(ir_l), mix=wet)])(mono_signal, SAMPLE_RATE)
    right = Pedalboard([Convolution(str(ir_r), mix=wet)])(mono_signal, SAMPLE_RATE)
    n_samples = len(signal)
    return np.stack([left[:n_samples], right[:n_samples]])


def apply_saturation(
    signal: np.ndarray,
    drive: float = 2.0,
    mix: float = 0.3,
    harmonics: float = 0.15,
) -> np.ndarray:
    """Apply tanh soft-clip saturation with subtle harmonic enrichment.

    Args:
        signal: Mono input signal.
        drive: Saturation amount — higher values clip more aggressively (default 2.0).
        mix: Wet/dry blend, 0.0 = fully dry, 1.0 = fully wet (default 0.3).
        harmonics: Additive 2nd/3rd harmonic boost on the wet path, 0.0–1.0 (default 0.15).

    Returns:
        Mono ndarray with the same peak level as the input.
    """
    input_peak = np.max(np.abs(signal))

    clipped = np.tanh(drive * signal)

    # Additive harmonic enrichment: 2nd harmonic (even) adds warmth/body,
    # 3rd harmonic (odd) adds brightness. Both are derived from the clipped signal.
    second_harmonic = np.tanh(drive * signal * 2.0) * 0.5  # octave, half amplitude
    third_harmonic = np.tanh(drive * signal * 3.0) / 3.0   # fifth+octave, third amplitude
    wet = clipped + harmonics * (second_harmonic + third_harmonic)

    blended = mix * wet + (1.0 - mix) * signal

    # Preserve input loudness so saturation doesn't change perceived level.
    output_peak = np.max(np.abs(blended))
    if output_peak > 0 and input_peak > 0:
        blended = blended * (input_peak / output_peak)

    return blended


def apply_effect_chain(
    signal: np.ndarray,
    effects: list[Any],
) -> np.ndarray:
    """Apply a declarative effect chain to a mono signal."""
    processed = signal
    for effect in effects:
        params = dict(effect.params)
        if effect.kind == "delay":
            processed = apply_delay(processed, **params)
        elif effect.kind == "reverb":
            processed = apply_reverb(processed, **params)
        elif effect.kind == "chow_tape":
            processed = apply_chow_tape(processed, **params)
        elif effect.kind == "bricasti":
            processed = apply_bricasti(processed, **params)
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
