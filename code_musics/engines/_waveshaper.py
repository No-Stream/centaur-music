"""Per-oscillator waveshaping distortion algorithms for drum engines.

Eleven distortion algorithms inspired by Geonkick, each implemented as a
numba-compiled inner loop.  The public entry point is :func:`apply_waveshaper`,
which handles drive mapping, optional per-sample drive envelopes, dry/wet mix,
output-level compensation, and first-order ADAA anti-aliasing.

Algorithms with closed-form antiderivatives (tanh, atan, hard_clip, exponential,
logarithmic, half_wave_rect, full_wave_rect) use analytical first-order ADAA.
Folding algorithms (foldback, linear_fold, sine_fold) rely on the ``oversample``
parameter for alias reduction.  The polynomial algorithm uses direct evaluation
since it already limits input to the monotone region.

This module is for per-oscillator use *inside* drum engines.  The voice-level
saturation effect in ``synth.py`` is a separate post-render concern.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import numba
import numpy as np
from scipy.signal import resample_poly

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
# ADAA antiderivative helpers (numba-compiled, per-sample)
# ---------------------------------------------------------------------------

_LN2: float = 0.6931471805599453
_ADAA_DX_THRESHOLD: float = 1e-5


@numba.njit(cache=True)
def _log_cosh(x: float) -> float:
    """Numerically stable log(cosh(x)): |x| + log(1 + exp(-2|x|)) - ln(2)."""
    ax = math.fabs(x)
    return ax + math.log1p(math.exp(-2.0 * ax)) - _LN2


@numba.njit(cache=True)
def _ad1_tanh(x: float) -> float:
    """Antiderivative of tanh: F(x) = log(cosh(x))."""
    return _log_cosh(x)


@numba.njit(cache=True)
def _ad1_atan(x: float) -> float:
    """Antiderivative of (2/pi)*atan(x): F(x) = (2/pi)*(x*atan(x) - 0.5*ln(1+x^2))."""
    return (2.0 / math.pi) * (x * math.atan(x) - 0.5 * math.log(1.0 + x * x))


@numba.njit(cache=True)
def _ad1_hard_clip(x: float) -> float:
    """Antiderivative of clamp(x, -1, 1)."""
    if x > 1.0:
        return x - 0.5
    if x < -1.0:
        return -x - 0.5
    return x * x * 0.5


@numba.njit(cache=True)
def _ad1_exponential(x: float) -> float:
    """Antiderivative of sgn(x)*(1-exp(-|x|)).

    For x >= 0: integral of (1 - exp(-x)) = x + exp(-x)
    For x < 0:  integral of (exp(x) - 1) = exp(x) - x
    Both evaluate to 1 at x=0 (continuous).
    """
    if x >= 0.0:
        return x + math.exp(-x)
    return math.exp(x) - x


@numba.njit(cache=True)
def _ad1_logarithmic(x: float, norm: float) -> float:
    """Antiderivative of sgn(x)*log(1+|x|)/norm.

    F is even: F(x) = ((1+|x|)*ln(1+|x|) - |x|) / norm for all x.
    """
    ax = math.fabs(x)
    return ((1.0 + ax) * math.log(1.0 + ax) - ax) / norm


@numba.njit(cache=True)
def _ad1_half_wave_rect(x: float) -> float:
    """Antiderivative of max(x, 0): F(x) = x^2/2 for x>=0, 0 for x<0."""
    if x > 0.0:
        return x * x * 0.5
    return 0.0


@numba.njit(cache=True)
def _ad1_full_wave_rect(x: float) -> float:
    """Antiderivative of |x|: F(x) = x*|x|/2 (odd function)."""
    return x * math.fabs(x) * 0.5


# ---------------------------------------------------------------------------
# Per-sample ADAA dispatch
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _adaa_sample(
    x_curr: float,
    x_prev: float,
    algorithm_id: int,
    drive_gain: float,
) -> float:
    """Compute one ADAA output sample given current and previous driven values.

    x_curr and x_prev are already multiplied by drive_gain.
    """
    dx = x_curr - x_prev
    mid = 0.5 * (x_curr + x_prev)

    if algorithm_id == _ID_TANH:
        if math.fabs(dx) > _ADAA_DX_THRESHOLD:
            result = (_ad1_tanh(x_curr) - _ad1_tanh(x_prev)) / dx
            if result > 1.0:
                return 1.0
            if result < -1.0:
                return -1.0
            return result
        return math.tanh(mid)

    if algorithm_id == _ID_ATAN:
        if math.fabs(dx) > _ADAA_DX_THRESHOLD:
            return (_ad1_atan(x_curr) - _ad1_atan(x_prev)) / dx
        return (2.0 / math.pi) * math.atan(mid)

    if algorithm_id == _ID_HARD_CLIP:
        if math.fabs(dx) > _ADAA_DX_THRESHOLD:
            return (_ad1_hard_clip(x_curr) - _ad1_hard_clip(x_prev)) / dx
        if mid > 1.0:
            return 1.0
        if mid < -1.0:
            return -1.0
        return mid

    if algorithm_id == _ID_EXPONENTIAL:
        if math.fabs(dx) > _ADAA_DX_THRESHOLD:
            return (_ad1_exponential(x_curr) - _ad1_exponential(x_prev)) / dx
        ax = math.fabs(mid)
        val = 1.0 - math.exp(-ax)
        return val if mid >= 0.0 else -val

    if algorithm_id == _ID_LOGARITHMIC:
        norm = math.log(1.0 + drive_gain) if drive_gain > 0.0 else 1.0
        if math.fabs(dx) > _ADAA_DX_THRESHOLD:
            return (
                _ad1_logarithmic(x_curr, norm) - _ad1_logarithmic(x_prev, norm)
            ) / dx
        ax = math.fabs(mid)
        val = math.log(1.0 + ax) / norm
        return val if mid >= 0.0 else -val

    if algorithm_id == _ID_HALF_WAVE_RECT:
        if math.fabs(dx) > _ADAA_DX_THRESHOLD:
            return (_ad1_half_wave_rect(x_curr) - _ad1_half_wave_rect(x_prev)) / dx
        return mid if mid > 0.0 else 0.0

    if algorithm_id == _ID_FULL_WAVE_RECT:
        if math.fabs(dx) > _ADAA_DX_THRESHOLD:
            return (_ad1_full_wave_rect(x_curr) - _ad1_full_wave_rect(x_prev)) / dx
        return math.fabs(mid)

    # Fallback for algorithms without analytical ADAA (polynomial, folds):
    # direct evaluation at x_curr
    if algorithm_id == _ID_POLYNOMIAL:
        limit = math.sqrt(3.0 / drive_gain) if drive_gain > 0.0 else 1e6
        xd = x_curr
        if xd > limit:
            xd = limit
        elif xd < -limit:
            xd = -limit
        return xd - (drive_gain * xd * xd * xd) / 3.0

    if algorithm_id == _ID_FOLDBACK:
        threshold = 1.0 / max(drive_gain, 1e-12)
        if threshold <= 0.0:
            return 0.0
        x_shifted = x_curr / drive_gain + threshold
        period = 4.0 * threshold
        phase = x_shifted - period * math.floor(x_shifted / period)
        half = 2.0 * threshold
        if phase < half:
            return phase - threshold
        return 3.0 * threshold - phase

    if algorithm_id == _ID_LINEAR_FOLD:
        scaled = x_curr
        return abs(((scaled * 0.25 + 0.75) % 1.0) * -4.0 + 2.0) - 1.0

    if algorithm_id == _ID_SINE_FOLD:
        return math.sin(x_curr * math.pi)

    return x_curr


# ---------------------------------------------------------------------------
# ADAA-aware static loop (no envelope)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _apply_adaa(
    signal: np.ndarray,
    algorithm_id: int,
    drive_gain: float,
) -> np.ndarray:
    """Apply waveshaping with first-order ADAA over the whole signal."""
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)

    prev_driven = signal[0] * drive_gain if n > 0 else 0.0

    for i in range(n):
        curr_driven = signal[i] * drive_gain
        out[i] = _adaa_sample(curr_driven, prev_driven, algorithm_id, drive_gain)
        prev_driven = curr_driven

    return out


# ---------------------------------------------------------------------------
# Per-sample envelope-modulated waveshaping with ADAA
# ---------------------------------------------------------------------------

_DRIVE_SCALE_NB: float = 49.0  # duplicate constant for numba scope


@numba.njit(cache=True)
def _apply_with_envelope(
    signal: np.ndarray,
    algorithm_id: int,
    base_drive: float,
    envelope: np.ndarray,
) -> np.ndarray:
    """Apply waveshaping with per-sample drive modulation and ADAA.

    ``base_drive`` is the user-facing 0-1 drive value.  ``envelope`` contains
    per-sample multipliers in [0, 1] that scale the drive.
    """
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)

    # Initialize prev_driven for ADAA state
    initial_drive = base_drive * envelope[0] if n > 0 else 0.0
    initial_g = 1.0 + _DRIVE_SCALE_NB * initial_drive * initial_drive
    prev_driven = signal[0] * initial_g if n > 0 else 0.0

    for i in range(n):
        effective_drive = base_drive * envelope[i]
        g = 1.0 + _DRIVE_SCALE_NB * effective_drive * effective_drive
        curr_driven = signal[i] * g

        out[i] = _adaa_sample(curr_driven, prev_driven, algorithm_id, g)
        prev_driven = curr_driven

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
    oversample: int = 1,
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
        oversample: Oversampling factor (1 or 2).  When 2, the signal is
            upsampled before processing and downsampled after, reducing
            aliasing for folding algorithms that lack analytical ADAA.

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

    # Upsample if requested
    if oversample > 1:
        sig_proc = resample_poly(sig, oversample, 1).astype(np.float64)
        env_proc: np.ndarray | None = None
        if env is not None:
            env_proc = resample_poly(env, oversample, 1).astype(np.float64)
    else:
        sig_proc = sig
        env_proc = env

    # Compute wet signal (ADAA is built into both paths)
    algo_id = _NAME_TO_ID[algorithm]
    if env_proc is not None:
        wet = _apply_with_envelope(sig_proc, algo_id, drive, env_proc)
    else:
        wet = _apply_adaa(sig_proc, algo_id, _drive_to_gain(drive))

    # Downsample if oversampled
    if oversample > 1:
        wet = resample_poly(wet, 1, oversample).astype(np.float64)
        # Trim or pad to match original length (resample_poly can be off by 1)
        if wet.shape[0] > sig.shape[0]:
            wet = wet[: sig.shape[0]]
        elif wet.shape[0] < sig.shape[0]:
            wet = np.concatenate([wet, np.zeros(sig.shape[0] - wet.shape[0])])

    # RMS-based level compensation: match the wet signal's RMS to the dry
    dry_rms = float(np.sqrt(np.mean(sig * sig)))
    wet_rms = float(np.sqrt(np.mean(wet * wet)))
    if wet_rms > 1e-12 and dry_rms > 1e-12:
        wet = wet * (dry_rms / wet_rms)

    # Dry/wet blend
    if mix >= 1.0:
        return wet
    return (1.0 - mix) * sig + mix * wet
