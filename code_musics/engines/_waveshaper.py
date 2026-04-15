"""Per-oscillator waveshaping distortion algorithms for drum engines.

Nine distortion algorithms inspired by Geonkick, each implemented as a
numba-compiled inner loop.  The public entry point is :func:`apply_waveshaper`,
which handles drive mapping, optional per-sample drive envelopes, dry/wet mix,
and output-level compensation.

This module is for per-oscillator use *inside* drum engines.  The voice-level
saturation effect in ``synth.py`` is a separate post-render concern.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import numba
import numpy as np

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Integer IDs for numba dispatch (no Python strings inside compiled loops)
# ---------------------------------------------------------------------------

_ID_HARD_CLIP: int = 0
_ID_TANH: int = 1
_ID_ATAN: int = 2
_ID_EXPONENTIAL: int = 3
_ID_POLYNOMIAL: int = 4
_ID_LOGARITHMIC: int = 5
_ID_FOLDBACK: int = 6
_ID_HALF_WAVE_RECT: int = 7
_ID_FULL_WAVE_RECT: int = 8
_ID_LINEAR_FOLD: int = 9
_ID_SINE_FOLD: int = 10

# ---------------------------------------------------------------------------
# Drive mapping: user-facing 0-1 -> internal gain
#
# Exponential curve:  drive_gain = 1 + 49 * drive^2
#   drive=0.0  -> 1.0   (passthrough)
#   drive=0.25 -> 4.06  (subtle)
#   drive=0.5  -> 13.25 (moderate)
#   drive=0.75 -> 28.56 (strong)
#   drive=1.0  -> 50.0  (maximum)
# ---------------------------------------------------------------------------

_DRIVE_SCALE: float = 49.0


def _drive_to_gain(drive: float) -> float:
    return 1.0 + _DRIVE_SCALE * drive * drive


# ---------------------------------------------------------------------------
# Individual waveshaper algorithms (numba-compiled, operate on flat arrays)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _hard_clip(signal: np.ndarray, drive_gain: float) -> np.ndarray:
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        x = signal[i] * drive_gain
        if x > 1.0:
            out[i] = 1.0
        elif x < -1.0:
            out[i] = -1.0
        else:
            out[i] = x
    return out


@numba.njit(cache=True)
def _tanh_clip(signal: np.ndarray, drive_gain: float) -> np.ndarray:
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = math.tanh(signal[i] * drive_gain)
    return out


@numba.njit(cache=True)
def _atan_clip(signal: np.ndarray, drive_gain: float) -> np.ndarray:
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)
    scale = 2.0 / math.pi
    for i in range(n):
        out[i] = scale * math.atan(signal[i] * drive_gain)
    return out


@numba.njit(cache=True)
def _exponential_clip(signal: np.ndarray, drive_gain: float) -> np.ndarray:
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        x = signal[i]
        ax = abs(x) * drive_gain
        if x >= 0.0:
            out[i] = 1.0 - math.exp(-ax)
        else:
            out[i] = -(1.0 - math.exp(-ax))
    return out


@numba.njit(cache=True)
def _polynomial_clip(signal: np.ndarray, drive_gain: float) -> np.ndarray:
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)
    # Clamp input to avoid folding artifacts beyond the cubic monotone region.
    limit = math.sqrt(3.0 / drive_gain) if drive_gain > 0.0 else 1e6
    for i in range(n):
        x = signal[i] * drive_gain
        if x > limit:
            x = limit
        elif x < -limit:
            x = -limit
        out[i] = x - (drive_gain * x * x * x) / 3.0
    return out


@numba.njit(cache=True)
def _logarithmic_clip(signal: np.ndarray, drive_gain: float) -> np.ndarray:
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)
    norm = math.log(1.0 + drive_gain) if drive_gain > 0.0 else 1.0
    for i in range(n):
        x = signal[i]
        ax = abs(x) * drive_gain
        val = math.log(1.0 + ax) / norm
        if x >= 0.0:
            out[i] = val
        else:
            out[i] = -val
    return out


@numba.njit(cache=True)
def _foldback(signal: np.ndarray, drive_gain: float) -> np.ndarray:
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)
    threshold = 1.0 / max(drive_gain, 1e-12)
    for i in range(n):
        x = signal[i]
        if threshold <= 0.0:
            out[i] = 0.0
        else:
            # Modulo-style fold: map into [-threshold, threshold] via triangle wave
            # Shift so that 0 maps to 0, then fold.
            x_shifted = x + threshold
            period = 4.0 * threshold
            # Place in [0, period)
            phase = x_shifted - period * math.floor(x_shifted / period)
            # Triangle wave: rises 0->threshold over [0, 2*threshold],
            # falls threshold->-threshold over [2*threshold, 4*threshold]
            half = 2.0 * threshold
            if phase < half:
                out[i] = phase - threshold
            else:
                out[i] = 3.0 * threshold - phase
    return out


@numba.njit(cache=True)
def _half_wave_rect(signal: np.ndarray, drive_gain: float) -> np.ndarray:
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        x = signal[i] * drive_gain
        if x > 0.0:
            out[i] = x
        else:
            out[i] = 0.0
    return out


@numba.njit(cache=True)
def _full_wave_rect(signal: np.ndarray, drive_gain: float) -> np.ndarray:
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = abs(signal[i] * drive_gain)
    return out


@numba.njit(cache=True)
def _linear_fold(signal: np.ndarray, drive_gain: float) -> np.ndarray:
    """Linear wavefolder -- folds the waveform back on itself."""
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        scaled = signal[i] * drive_gain
        out[i] = abs(((scaled * 0.25 + 0.75) % 1.0) * -4.0 + 2.0) - 1.0
    return out


@numba.njit(cache=True)
def _sine_fold(signal: np.ndarray, drive_gain: float) -> np.ndarray:
    """Sine wavefolder -- soft, musical folding via sine transfer function."""
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = math.sin(signal[i] * drive_gain * math.pi)
    return out


# ---------------------------------------------------------------------------
# Per-sample envelope-modulated waveshaping (single numba function)
# ---------------------------------------------------------------------------

_DRIVE_SCALE_NB: float = 49.0  # duplicate constant for numba scope


@numba.njit(cache=True)
def _apply_with_envelope(
    signal: np.ndarray,
    algorithm_id: int,
    base_drive: float,
    envelope: np.ndarray,
) -> np.ndarray:
    """Apply waveshaping with per-sample drive modulation.

    ``base_drive`` is the user-facing 0-1 drive value.  ``envelope`` contains
    per-sample multipliers in [0, 1] that scale the drive.
    """
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)

    for i in range(n):
        effective_drive = base_drive * envelope[i]
        g = 1.0 + _DRIVE_SCALE_NB * effective_drive * effective_drive
        x = signal[i]

        if algorithm_id == 0:  # hard_clip
            xd = x * g
            if xd > 1.0:
                out[i] = 1.0
            elif xd < -1.0:
                out[i] = -1.0
            else:
                out[i] = xd

        elif algorithm_id == 1:  # tanh
            out[i] = math.tanh(x * g)

        elif algorithm_id == 2:  # atan
            out[i] = (2.0 / math.pi) * math.atan(x * g)

        elif algorithm_id == 3:  # exponential
            ax = abs(x) * g
            val = 1.0 - math.exp(-ax)
            out[i] = val if x >= 0.0 else -val

        elif algorithm_id == 4:  # polynomial
            limit = math.sqrt(3.0 / g) if g > 0.0 else 1e6
            xd = x * g
            if xd > limit:
                xd = limit
            elif xd < -limit:
                xd = -limit
            out[i] = xd - (g * xd * xd * xd) / 3.0

        elif algorithm_id == 5:  # logarithmic
            norm = math.log(1.0 + g) if g > 0.0 else 1.0
            ax = abs(x) * g
            val = math.log(1.0 + ax) / norm
            out[i] = val if x >= 0.0 else -val

        elif algorithm_id == 6:  # foldback
            threshold = 1.0 / max(g, 1e-12)
            if threshold <= 0.0:
                out[i] = 0.0
            else:
                x_shifted = x + threshold
                period = 4.0 * threshold
                phase = x_shifted - period * math.floor(x_shifted / period)
                half = 2.0 * threshold
                if phase < half:
                    out[i] = phase - threshold
                else:
                    out[i] = 3.0 * threshold - phase

        elif algorithm_id == 7:  # half_wave_rect
            xd = x * g
            out[i] = xd if xd > 0.0 else 0.0

        elif algorithm_id == 8:  # full_wave_rect
            out[i] = abs(x * g)

        elif algorithm_id == 9:  # linear_fold
            scaled = x * g
            out[i] = abs(((scaled * 0.25 + 0.75) % 1.0) * -4.0 + 2.0) - 1.0

        elif algorithm_id == 10:  # sine_fold
            out[i] = math.sin(x * g * math.pi)

        else:
            out[i] = x

    return out


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

_ALGORITHMS: dict[str, Callable[..., np.ndarray]] = {
    "hard_clip": _hard_clip,
    "tanh": _tanh_clip,
    "atan": _atan_clip,
    "exponential": _exponential_clip,
    "polynomial": _polynomial_clip,
    "logarithmic": _logarithmic_clip,
    "foldback": _foldback,
    "half_wave_rect": _half_wave_rect,
    "full_wave_rect": _full_wave_rect,
    "linear_fold": _linear_fold,
    "sine_fold": _sine_fold,
}

_NAME_TO_ID: dict[str, int] = {
    "hard_clip": _ID_HARD_CLIP,
    "tanh": _ID_TANH,
    "atan": _ID_ATAN,
    "exponential": _ID_EXPONENTIAL,
    "polynomial": _ID_POLYNOMIAL,
    "logarithmic": _ID_LOGARITHMIC,
    "foldback": _ID_FOLDBACK,
    "half_wave_rect": _ID_HALF_WAVE_RECT,
    "full_wave_rect": _ID_FULL_WAVE_RECT,
    "linear_fold": _ID_LINEAR_FOLD,
    "sine_fold": _ID_SINE_FOLD,
}

ALGORITHM_NAMES: frozenset[str] = frozenset(_ALGORITHMS)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply_waveshaper(
    signal: np.ndarray,
    *,
    algorithm: str,
    drive: float,
    drive_envelope: np.ndarray | None = None,
    mix: float = 1.0,
) -> np.ndarray:
    """Apply a waveshaping distortion algorithm to *signal*.

    Args:
        signal: Input audio array (mono, float64).
        algorithm: One of :data:`ALGORITHM_NAMES`.
        drive: User-facing drive amount in [0, 1].  0 is near-passthrough,
            1 is maximum distortion.
        drive_envelope: Optional per-sample envelope (same length as *signal*,
            values in [0, 1]) that modulates the drive over time.
        mix: Dry/wet blend in [0, 1].  0 = fully dry, 1 = fully wet.

    Returns:
        Processed signal, level-compensated to roughly match input RMS.
    """
    if algorithm not in _ALGORITHMS:
        raise ValueError(
            f"Unknown waveshaper algorithm {algorithm!r}. "
            f"Choose from: {sorted(ALGORITHM_NAMES)}"
        )

    drive = float(np.clip(drive, 0.0, 1.0))
    mix = float(np.clip(mix, 0.0, 1.0))

    sig = np.asarray(signal, dtype=np.float64)

    env: np.ndarray | None = None
    if drive_envelope is not None:
        env = np.asarray(drive_envelope, dtype=np.float64)
        if env.shape[0] != sig.shape[0]:
            raise ValueError(
                f"drive_envelope length ({env.shape[0]}) must match "
                f"signal length ({sig.shape[0]})"
            )

    # Near-passthrough fast path
    if mix <= 0.0:
        return sig.copy()

    # Compute wet signal
    if env is not None:
        wet = _apply_with_envelope(sig, _NAME_TO_ID[algorithm], drive, env)
    else:
        drive_gain = _drive_to_gain(drive)
        fn = _ALGORITHMS[algorithm]
        wet = fn(sig, drive_gain)

    # RMS-based level compensation: match the wet signal's RMS to the dry
    dry_rms = float(np.sqrt(np.mean(sig * sig)))
    wet_rms = float(np.sqrt(np.mean(wet * wet)))
    if wet_rms > 1e-12 and dry_rms > 1e-12:
        wet = wet * (dry_rms / wet_rms)

    # Dry/wet blend
    if mix >= 1.0:
        return wet
    return (1.0 - mix) * sig + mix * wet
