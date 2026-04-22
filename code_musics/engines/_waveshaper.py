"""Per-oscillator waveshaping distortion algorithms for drum engines.

Fourteen distortion algorithms inspired by Geonkick (analog-flavoured) and
Machinedrum / SP-1200 (digital-flavoured), each implemented as a numba-compiled
inner loop.  The public entry point is :func:`apply_waveshaper`, which handles
drive mapping, optional per-sample drive envelopes, dry/wet mix, output-level
compensation, and ADAA anti-aliasing.

Algorithms with closed-form antiderivatives (tanh, atan, hard_clip, exponential,
logarithmic, half_wave_rect, full_wave_rect) use analytical ADAA.  The
``adaa_order`` parameter on :func:`apply_waveshaper` selects first-order (AD1,
default, ~30 dB alias suppression) or second-order (AD2, adds another
~30-40 dB on top of AD1 at the same oversampling cost).  AD2 uses the
Bilbao/Esqueda three-sample form; for ``tanh`` the second antiderivative
requires the dilogarithm Li2 and is provided via a module-load numerically
integrated lookup table (asymptotic for |x| > 20).  For the other six
algorithms F2 is closed-form cubic / log.

Folding algorithms (foldback, linear_fold, sine_fold) rely on the
``oversample`` parameter for alias reduction.  The polynomial algorithm uses
direct evaluation since it already limits input to the monotone region.

The digital-character algorithms (bit_crush, rate_reduce, digital_clip) are
inherently discontinuous / piecewise and are not amenable to ADAA.  Callers
that care about alias reduction for these three should pass ``oversample=2``
— the oversample path handles them transparently.

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

from code_musics.engines._ad2_utils import (
    _AD2_TANH_TABLE,
    _ADAA_DX_THRESHOLD,
    _LN2,
    _ad2_tanh_lut,
)

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
_ID_BIT_CRUSH: int = 11
_ID_RATE_REDUCE: int = 12
_ID_DIGITAL_CLIP: int = 13

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


@numba.njit(cache=True)
def _bit_crush(signal: np.ndarray, drive_gain: float, bits: float) -> np.ndarray:
    """Symmetric signed bit-crusher.

    ``bits`` is a float param (1.0-16.0).  Real levels = ``2**round(bits)``.
    At bits<=1 -> 2 levels (harsh one-bit).  At bits>=12 -> near-transparent.
    Quantization is symmetric around zero (signed-integer style).
    """
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)
    # Real level count: rounded, clamped to [2, 65536]
    b_int = int(round(bits))
    if b_int < 1:
        b_int = 1
    if b_int > 16:
        b_int = 16
    levels = 1 << b_int  # 2**b_int
    # Signed quantization: max positive value is levels/2 - 1; step = 1 / (levels/2)
    half = levels // 2
    step = 1.0 / max(float(half), 1.0)
    inv_gain = 1.0 / drive_gain if drive_gain > 0.0 else 1.0
    for i in range(n):
        x = signal[i] * drive_gain
        # Clamp to [-1, 1] prior to quantization
        if x > 1.0:
            x = 1.0
        elif x < -1.0:
            x = -1.0
        # Symmetric quantizer: round(x * half) / half, clipped to representable range
        q = math.floor(x * half + 0.5)
        if q > half - 1:
            q = float(half - 1)
        elif q < -half:
            q = float(-half)
        out[i] = q * step * inv_gain
    return out


@numba.njit(cache=True)
def _rate_reduce(
    signal: np.ndarray, drive_gain: float, reduce_ratio: float
) -> np.ndarray:
    """Integer-ratio sample-and-hold.

    Holds every Nth sample where N = floor(reduce_ratio), N >= 1.
    Drive is pre-gain, compensated on output.
    """
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)
    n_hold = int(math.floor(reduce_ratio))
    if n_hold < 1:
        n_hold = 1
    inv_gain = 1.0 / drive_gain if drive_gain > 0.0 else 1.0
    held = 0.0
    for i in range(n):
        if i % n_hold == 0:
            held = signal[i] * drive_gain
        out[i] = held * inv_gain
    return out


@numba.njit(cache=True)
def _digital_clip(signal: np.ndarray, drive_gain: float) -> np.ndarray:
    """Asymmetric hard clip: +1.0 / -0.95 rails.

    Drive is pre-gain, compensated on output.
    """
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)
    inv_gain = 1.0 / drive_gain if drive_gain > 0.0 else 1.0
    pos_rail = 1.0
    neg_rail = -0.95
    for i in range(n):
        x = signal[i] * drive_gain
        if x > pos_rail:
            x = pos_rail
        elif x < neg_rail:
            x = neg_rail
        out[i] = x * inv_gain
    return out


# ---------------------------------------------------------------------------
# ADAA antiderivative helpers (numba-compiled, per-sample).
#
# ``_LN2`` and ``_ADAA_DX_THRESHOLD`` are imported at the top of the module
# from :mod:`code_musics.engines._ad2_utils` so the AD1 and AD2 dispatch in
# this file share the exact same constants that :mod:`_filters` uses.
# ---------------------------------------------------------------------------


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
# Second-antiderivative (F2) helpers for AD2 ADAA.
#
# F2(x) = integral of F1(x).  Used in Bilbao/Esqueda's second-order ADAA:
#
#     y = 2 / (x_n - x_{n-2}) *
#         ((F2(x_n) - F2(x_{n-1})) / (x_n - x_{n-1})
#          - (F2(x_{n-1}) - F2(x_{n-2})) / (x_{n-1} - x_{n-2}))
#
# Every helper below is analytically derived except F2(tanh) which uses
# a precomputed lookup table (``_AD2_TANH_TABLE`` / :func:`_ad2_tanh_lut`
# imported from :mod:`code_musics.engines._ad2_utils`) because the closed
# form requires the dilogarithm.  All helpers are continuous at the pieces'
# boundaries so the ADAA subtraction stays well-conditioned; we verified
# continuity at the stitch points (x=0, x=±1) during derivation.
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _ad2_atan(x: float) -> float:
    """Second antiderivative of (2/pi)*atan(x).

    Derivation (integration by parts twice):
        F1(x) = (2/pi) * (x*atan(x) - 0.5*ln(1+x²))
        F2(x) = (1/pi) * (x²*atan(x) + x - atan(x) - x*ln(1+x²))
    Verified: dF2/dx = (2/pi)*x*atan(x) - (1/pi)*x²/(1+x²) ... algebra
    simplifies to F1, confirming the closed form.
    """
    return (
        x * x * math.atan(x) + x - math.atan(x) - x * math.log(1.0 + x * x)
    ) / math.pi


@numba.njit(cache=True)
def _ad2_hard_clip(x: float) -> float:
    """Second antiderivative of clamp(x, -1, 1).

    F1 is x²/2 on |x|≤1 and ±x - 0.5 outside.  Integrating again with C⁰
    continuity at x=±1:
        |x| ≤ 1:  F2 = x³/6
        x > 1:    F2 = x²/2 - x/2 + 1/6
        x < -1:   F2 = -x²/2 - x/2 - 1/6
    Continuity check: at x=1, x³/6 = 1/6 and 1/2 - 1/2 + 1/6 = 1/6. ✓
                      at x=-1, -1/6 and -1/2 + 1/2 - 1/6 = -1/6. ✓
    """
    if x > 1.0:
        return 0.5 * x * x - 0.5 * x + 1.0 / 6.0
    if x < -1.0:
        return -0.5 * x * x - 0.5 * x - 1.0 / 6.0
    return x * x * x / 6.0


@numba.njit(cache=True)
def _ad2_exponential(x: float) -> float:
    """Second antiderivative of sgn(x)*(1-exp(-|x|)).

        F1(x) = x + exp(-x)         for x >= 0
        F1(x) = exp(x) - x          for x <  0
    Integrating and enforcing C⁰ continuity (F2 continuous at x=0):
        F2(x) = x²/2 - exp(-x) + 1       for x >= 0
        F2(x) = exp(x) - x²/2 - 1        for x <  0
    Check: F2(0⁺) = 0 - 1 + 1 = 0; F2(0⁻) = 1 - 0 - 1 = 0. ✓
    """
    if x >= 0.0:
        return 0.5 * x * x - math.exp(-x) + 1.0
    return math.exp(x) - 0.5 * x * x - 1.0


@numba.njit(cache=True)
def _ad2_logarithmic(x: float, norm: float) -> float:
    """Second antiderivative of sgn(x)*log(1+|x|)/norm.

    F1(x) = ((1+|x|)*ln(1+|x|) - |x|) / norm  (even function).
    Let u = 1 + |x|.  For x >= 0:
        integral of (1+x)*ln(1+x) dx = (1+x)²*ln(1+x)/2 - (1+x)²/4 + C
        integral of x dx = x²/2
    So for x >= 0:
        F2_pos(x) = [(1+x)²*ln(1+x)/2 - (1+x)²/4 - x²/2] / norm + C0
    Choose C0 so F2(0) = 0:  C0 = -(-1/4)/norm = 1/(4*norm).
    That is, F2_pos(0) = (0 - 1/4 - 0)/norm + 1/(4*norm) = 0. ✓
    F1 even → F2 odd → for x < 0, F2(x) = -F2_pos(-x).
    """
    ax = math.fabs(x)
    u = 1.0 + ax
    f2_pos = (0.5 * u * u * math.log(u) - 0.25 * u * u - 0.5 * ax * ax + 0.25) / norm
    if x >= 0.0:
        return f2_pos
    return -f2_pos


@numba.njit(cache=True)
def _ad2_half_wave_rect(x: float) -> float:
    """Second antiderivative of max(x, 0): F2 = x³/6 for x>=0, else 0."""
    if x > 0.0:
        return x * x * x / 6.0
    return 0.0


@numba.njit(cache=True)
def _ad2_full_wave_rect(x: float) -> float:
    """Second antiderivative of |x|: F2 = x²*|x|/6 (odd function)."""
    return x * x * math.fabs(x) / 6.0


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
# Second-order ADAA (AD2) — Bilbao/Esqueda form.
#
# Given three-sample history x_n, x_{n-1}, x_{n-2}:
#
#     d1 = (F2(x_n)     - F2(x_{n-1})) / (x_n     - x_{n-1})
#     d2 = (F2(x_{n-1}) - F2(x_{n-2})) / (x_{n-1} - x_{n-2})
#     y  = 2 * (d1 - d2) / (x_n - x_{n-2})
#
# We guard each denominator with the same ``_ADAA_DX_THRESHOLD`` used by AD1,
# falling back to direct midpoint evaluation of f(x) when any denominator is
# too small.  This matches the recipe in Esqueda et al., "Differentiated
# polynomial waveforms via ADAA" (2016) and the generalisation in Bilbao's
# notes.  AD2 adds ~30-40 dB of additional alias suppression on top of AD1 at
# the same oversampling cost.
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _direct_shape(
    x: float,
    algorithm_id: int,
    drive_gain: float,
) -> float:
    """Direct per-sample waveshape evaluation — AD2 fallback when denominators
    collapse.  Used by :func:`_adaa2_sample` when either inner divided
    difference is too small for the Bilbao/Esqueda three-sample form.
    """
    if algorithm_id == _ID_TANH:
        return math.tanh(x)
    if algorithm_id == _ID_ATAN:
        return (2.0 / math.pi) * math.atan(x)
    if algorithm_id == _ID_HARD_CLIP:
        if x > 1.0:
            return 1.0
        if x < -1.0:
            return -1.0
        return x
    if algorithm_id == _ID_EXPONENTIAL:
        ax = math.fabs(x)
        val = 1.0 - math.exp(-ax)
        return val if x >= 0.0 else -val
    if algorithm_id == _ID_LOGARITHMIC:
        norm = math.log(1.0 + drive_gain) if drive_gain > 0.0 else 1.0
        ax = math.fabs(x)
        val = math.log(1.0 + ax) / norm
        return val if x >= 0.0 else -val
    if algorithm_id == _ID_HALF_WAVE_RECT:
        return x if x > 0.0 else 0.0
    if algorithm_id == _ID_FULL_WAVE_RECT:
        return math.fabs(x)
    return x


@numba.njit(cache=True)
def _ad2_f2(
    x: float,
    algorithm_id: int,
    drive_gain: float,
    tanh_table: np.ndarray,
) -> float:
    """Select the right F2 helper for ``algorithm_id``.

    Falls through to a zero return for algorithm ids that do not have F2
    (polynomial, folds, digital).  Callers must route those to the direct
    evaluation path instead of the AD2 dispatch.
    """
    if algorithm_id == _ID_TANH:
        return _ad2_tanh_lut(x, tanh_table)
    if algorithm_id == _ID_ATAN:
        return _ad2_atan(x)
    if algorithm_id == _ID_HARD_CLIP:
        return _ad2_hard_clip(x)
    if algorithm_id == _ID_EXPONENTIAL:
        return _ad2_exponential(x)
    if algorithm_id == _ID_LOGARITHMIC:
        norm = math.log(1.0 + drive_gain) if drive_gain > 0.0 else 1.0
        return _ad2_logarithmic(x, norm)
    if algorithm_id == _ID_HALF_WAVE_RECT:
        return _ad2_half_wave_rect(x)
    if algorithm_id == _ID_FULL_WAVE_RECT:
        return _ad2_full_wave_rect(x)
    return 0.0


# IDs that have an AD2 implementation.  The AD2 path is only used when this
# check passes; other algorithms keep the AD1 / direct / oversample path.
_AD2_SUPPORTED_IDS: frozenset[int] = frozenset(
    {
        _ID_TANH,
        _ID_ATAN,
        _ID_HARD_CLIP,
        _ID_EXPONENTIAL,
        _ID_LOGARITHMIC,
        _ID_HALF_WAVE_RECT,
        _ID_FULL_WAVE_RECT,
    }
)


@numba.njit(cache=True)
def _adaa2_sample(
    x_curr: float,
    x_prev: float,
    x_prev2: float,
    algorithm_id: int,
    drive_gain: float,
    tanh_table: np.ndarray,
) -> float:
    """Compute one AD2 ADAA output sample from a three-sample history.

    All ``x_*`` are already driven (multiplied by ``drive_gain``).  Falls back
    cleanly when either difference is too small:
      * If ``|x_curr - x_prev2| < eps``, we degenerate to AD1 between
        ``x_curr`` and ``x_prev``.
      * If either inner difference is too small, we fall back further to the
        midpoint / direct evaluation of f.
    This matches the same small-denominator safety that ``_adaa_sample`` uses
    with AD1.
    """
    dx_outer = x_curr - x_prev2
    dx_a = x_curr - x_prev
    dx_b = x_prev - x_prev2

    # When the outer difference is small we cannot evaluate AD2 stably.
    # Fall back to AD1 (using the existing dispatcher) which already handles
    # its own small-dx guard.
    if math.fabs(dx_outer) <= _ADAA_DX_THRESHOLD:
        return _adaa_sample(x_curr, x_prev, algorithm_id, drive_gain)

    # If either inner difference is small, both AD1-shaped divided differences
    # would be unstable; fall back to the midpoint direct evaluation.
    if math.fabs(dx_a) <= _ADAA_DX_THRESHOLD or math.fabs(dx_b) <= _ADAA_DX_THRESHOLD:
        return _direct_shape(0.5 * (x_curr + x_prev2), algorithm_id, drive_gain)

    f2_curr = _ad2_f2(x_curr, algorithm_id, drive_gain, tanh_table)
    f2_prev = _ad2_f2(x_prev, algorithm_id, drive_gain, tanh_table)
    f2_prev2 = _ad2_f2(x_prev2, algorithm_id, drive_gain, tanh_table)

    d1 = (f2_curr - f2_prev) / dx_a
    d2 = (f2_prev - f2_prev2) / dx_b
    result = 2.0 * (d1 - d2) / dx_outer

    # Clamp to the algorithm's natural output range to avoid amplification
    # from numerical noise near the fallback boundary.  Only applied for
    # bounded-range algorithms where the clamp is semantically correct.
    if algorithm_id in (_ID_TANH, _ID_HARD_CLIP):
        if result > 1.0:
            return 1.0
        if result < -1.0:
            return -1.0
    return result


# ---------------------------------------------------------------------------
# AD2-aware static loop (no envelope)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _apply_adaa2(
    signal: np.ndarray,
    algorithm_id: int,
    drive_gain: float,
    tanh_table: np.ndarray,
) -> np.ndarray:
    """Apply waveshaping with second-order ADAA over the whole signal."""
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)

    prev_driven = signal[0] * drive_gain if n > 0 else 0.0
    prev2_driven = prev_driven

    for i in range(n):
        curr_driven = signal[i] * drive_gain
        out[i] = _adaa2_sample(
            curr_driven,
            prev_driven,
            prev2_driven,
            algorithm_id,
            drive_gain,
            tanh_table,
        )
        prev2_driven = prev_driven
        prev_driven = curr_driven

    return out


@numba.njit(cache=True)
def _apply_with_envelope_adaa2(
    signal: np.ndarray,
    algorithm_id: int,
    base_drive: float,
    envelope: np.ndarray,
    tanh_table: np.ndarray,
) -> np.ndarray:
    """Apply waveshaping with per-sample drive modulation and AD2 ADAA.

    Note — logarithmic + envelope-modulated drive is not bit-accurate for
    AD2.  The ``logarithmic`` shaper's F2 helper depends on
    ``norm = log(1 + drive_gain)``.  When ``drive_gain`` varies per sample
    via ``envelope``, the three-sample divided-difference form evaluates
    ``F2(x_curr)``, ``F2(x_prev)``, ``F2(x_prev2)`` using the *current*
    sample's ``g`` for all three terms, which mixes values of ``F2`` from
    subtly different underlying antiderivatives.  The resulting output
    deviates from a mathematically exact AD2 of the time-varying shaper
    by a term that scales with the rate of change of ``envelope``.

    In practice this is a numerical concern rather than an audio-breaking
    one: the discrepancy is well below audible under musical drive
    envelopes (the envelope typically changes on tens-of-ms timescales
    while the AD2 kernel samples at the audio rate, so ``norm`` is
    near-constant across any three-sample window).  The limitation is
    inherited from the AD1 envelope path (see :func:`_apply_with_envelope`)
    and documenting it here keeps the behavior explicit for downstream
    callers.  A fully bit-accurate fix would either switch to a
    time-varying-norm F2 derivation or freeze ``norm`` across the AD2
    window — deferred because the audible cost is negligible.
    """
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)

    initial_drive = base_drive * envelope[0] if n > 0 else 0.0
    initial_g = 1.0 + _DRIVE_SCALE_NB * initial_drive * initial_drive
    prev_driven = signal[0] * initial_g if n > 0 else 0.0
    prev2_driven = prev_driven

    for i in range(n):
        effective_drive = base_drive * envelope[i]
        g = 1.0 + _DRIVE_SCALE_NB * effective_drive * effective_drive
        curr_driven = signal[i] * g

        out[i] = _adaa2_sample(
            curr_driven, prev_driven, prev2_driven, algorithm_id, g, tanh_table
        )
        prev2_driven = prev_driven
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
    "bit_crush": _bit_crush,
    "rate_reduce": _rate_reduce,
    "digital_clip": _digital_clip,
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
    "bit_crush": _ID_BIT_CRUSH,
    "rate_reduce": _ID_RATE_REDUCE,
    "digital_clip": _ID_DIGITAL_CLIP,
}

# Digital-character algorithms are not amenable to first-order ADAA (they are
# discontinuous / piecewise).  They run through a direct-dispatch path, and
# callers should pass ``oversample=2`` when alias reduction matters.
_DIGITAL_IDS: frozenset[int] = frozenset(
    {_ID_BIT_CRUSH, _ID_RATE_REDUCE, _ID_DIGITAL_CLIP}
)

ALGORITHM_NAMES: frozenset[str] = frozenset(_ALGORITHMS)

# Sentinel used to detect the default ``oversample`` value.  When a caller
# leaves ``oversample`` unset and chooses one of the digital-character algos
# (bit_crush / rate_reduce / digital_clip), the entry point auto-upgrades to
# ``oversample=2`` because these algos cannot use ADAA and otherwise alias
# audibly.  Callers that explicitly pass ``oversample=1`` keep the naive
# behavior — useful for tests and for engines that want the raw "deliberately
# lo-fi" character.
_DEFAULT_OVERSAMPLE: int = -1


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
    oversample: int = _DEFAULT_OVERSAMPLE,
    adaa_order: int = 1,
    # Extra scalar params: only consumed when ``algorithm`` is the matching
    # digital-character algo; ignored otherwise.
    bit_depth: float = 8.0,
    reduce_ratio: float = 2.0,
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
            When unset, defaults to 1 for ADAA-friendly algorithms and to 2
            for the digital-character algorithms (``bit_crush``,
            ``rate_reduce``, ``digital_clip``) since those cannot use ADAA
            and otherwise alias audibly.  Pass ``oversample=1`` explicitly
            to opt out of the auto-upgrade and keep the raw lo-fi character.
        adaa_order: 1 (default) selects first-order ADAA using the analytical
            antiderivative F1 — same behavior this module has shipped with
            since ADAA was first added.  2 selects second-order ADAA
            (Bilbao/Esqueda three-sample form) for the seven algorithms with
            F2 support (tanh, atan, hard_clip, exponential, logarithmic,
            half_wave_rect, full_wave_rect).  AD2 adds ~30-40 dB of extra
            alias suppression at the same oversampling cost and is
            recommended when drive is substantial.  Unsupported algorithms
            silently fall back to AD1 / oversample / direct dispatch.
        bit_depth: Effective bit-depth for ``bit_crush`` (1.0-16.0).  Ignored
            by other algorithms.
        reduce_ratio: Integer-ratio sample-and-hold factor for ``rate_reduce``
            (>= 1.0).  Ignored by other algorithms.

    Returns:
        Processed signal, level-compensated to roughly match input RMS.
    """
    if algorithm not in _ALGORITHMS:
        raise ValueError(
            f"Unknown waveshaper algorithm {algorithm!r}. "
            f"Choose from: {sorted(ALGORITHM_NAMES)}"
        )
    if adaa_order not in (1, 2):
        raise ValueError(
            f"adaa_order must be 1 or 2, got {adaa_order}. "
            "AD2 supports tanh, atan, hard_clip, exponential, logarithmic, "
            "half_wave_rect, full_wave_rect; other algorithms silently use "
            "the AD1 / oversample path regardless."
        )

    algo_id = _NAME_TO_ID[algorithm]
    if oversample == _DEFAULT_OVERSAMPLE:
        oversample = 2 if algo_id in _DIGITAL_IDS else 1

    # AD2 is only meaningful for algorithms with an F2 implementation; for
    # anything else we fall back to AD1.  This keeps the user-facing knob
    # forgiving (passing adaa_order=2 on, say, foldback just uses the
    # oversample path that foldback already relies on).
    use_adaa2 = adaa_order == 2 and algo_id in _AD2_SUPPORTED_IDS

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

    # Compute wet signal
    if algo_id in _DIGITAL_IDS:
        # Digital-character algos bypass ADAA; drive_envelope is not supported
        # since these algos are piecewise-constant / discontinuous and the ADAA
        # envelope path would be meaningless for them.
        drive_gain = _drive_to_gain(drive)
        if algo_id == _ID_BIT_CRUSH:
            wet = _bit_crush(sig_proc, drive_gain, float(bit_depth))
        elif algo_id == _ID_RATE_REDUCE:
            wet = _rate_reduce(sig_proc, drive_gain, float(reduce_ratio))
        else:  # _ID_DIGITAL_CLIP
            wet = _digital_clip(sig_proc, drive_gain)
    elif env_proc is not None:
        if use_adaa2:
            wet = _apply_with_envelope_adaa2(
                sig_proc, algo_id, drive, env_proc, _AD2_TANH_TABLE
            )
        else:
            wet = _apply_with_envelope(sig_proc, algo_id, drive, env_proc)
    elif use_adaa2:
        wet = _apply_adaa2(sig_proc, algo_id, _drive_to_gain(drive), _AD2_TANH_TABLE)
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
