"""Shared DSP filter primitives used by multiple synth engines."""

from __future__ import annotations

import math

import numba
import numpy as np

_SUPPORTED_FILTER_MODES = {"lowpass", "bandpass", "highpass", "notch"}

# Integer constants for filter mode selection inside numba-compiled loops.
_LP: int = 0
_BP: int = 1
_HP: int = 2
_NOTCH: int = 3

_MODE_STR_TO_INT: dict[str, int] = {
    "lowpass": _LP,
    "bandpass": _BP,
    "highpass": _HP,
    "notch": _NOTCH,
}

_LN2: float = 0.6931471805599453

# Max cutoff as a fraction of sample rate — ~19.8 kHz at 44.1 kHz.
# Prevents tan(pi * fc / sr) from blowing up near Nyquist.
_NYQUIST_CLAMP_RATIO: float = 0.45


@numba.njit(cache=True)
def _log_cosh(x: float) -> float:
    """Numerically stable log(cosh(x)) — the antiderivative of tanh.

    Uses the identity: log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - ln(2).
    Stable for all x: large |x| → |x| - ln(2), small |x| → x²/2.
    """
    ax = math.fabs(x)
    return ax + math.log1p(math.exp(-2.0 * ax)) - _LN2


@numba.njit(cache=True)
def _adaa_tanh(x_curr: float, x_prev: float) -> float:
    """First-order antiderivative anti-aliased tanh.

    Instead of y = tanh(x), computes y = (AD1(x_n) - AD1(x_{n-1})) / (x_n - x_{n-1})
    where AD1(tanh) = log(cosh).  This eliminates aliasing from the nonlinearity
    at a cost comparable to a single tanh + log call per sample.

    Falls back to tanh(midpoint) when the denominator is too small.
    """
    dx = x_curr - x_prev
    if math.fabs(dx) > 1e-5:
        result = (_log_cosh(x_curr) - _log_cosh(x_prev)) / dx
        # ADAA output is theoretically bounded by [-1, 1] (since tanh is),
        # but numerical edge cases with large inputs can exceed this.
        # Clamp to ensure stability.
        if result > 1.0:
            return 1.0
        if result < -1.0:
            return -1.0
        return result
    return math.tanh(0.5 * (x_curr + x_prev))


@numba.njit(cache=True)
def _algebraic_sat(x: float) -> float:
    """Gentle algebraic saturation for integrator state limiting.

    Transparent for |x| < ~1.0, then transitions to soft clipping.
    Uses algebraicSat (from Vital) for the gentle region, with a tanh
    ceiling to guarantee bounded output — prevents state explosion at
    extreme resonance/drive combinations.
    """
    if math.fabs(x) < 2.0:
        return x - 0.9 * x * x * x / (x * x + 3.0)
    return math.tanh(x)


@numba.njit(cache=True)
def _apply_linear_zdf_svf(
    signal: np.ndarray,
    cutoff_profile: np.ndarray,
    damping: float,
    sample_rate: int,
    mode_int: int,
    precomputed_g: float,
    nyquist_limit: float,
) -> np.ndarray:
    """Apply the fully linear ZDF/TPT state-variable filter (numba-compiled)."""
    n = signal.shape[0]
    filtered = np.empty(n, dtype=np.float64)
    low_state = 0.0
    band_state = 0.0
    use_precomputed = precomputed_g >= 0.0

    for i in range(n):
        if use_precomputed:
            g = precomputed_g
        else:
            # Clamp cutoff to _NYQUIST_CLAMP_RATIO of sample rate
            fc = min(cutoff_profile[i], nyquist_limit)
            g = math.tan(math.pi * fc / sample_rate)
        sample = signal[i]
        high = (sample - (2.0 * damping + g) * band_state - low_state) / (
            1.0 + 2.0 * damping * g + g * g
        )
        band = g * high + band_state
        low = g * band + low_state
        band_state = band + g * high
        low_state = low + g * band

        if mode_int == _LP:
            filtered[i] = low
        elif mode_int == _BP:
            filtered[i] = band
        elif mode_int == _HP:
            filtered[i] = high
        else:
            filtered[i] = low + high

    return filtered


@numba.njit(cache=True)
def _apply_driven_zdf_svf(
    signal: np.ndarray,
    cutoff_profile: np.ndarray,
    damping: float,
    sample_rate: int,
    mode_int: int,
    filter_drive: float,
    precomputed_g: float,
    even_harmonics: float,
    nyquist_limit: float,
) -> np.ndarray:
    """Apply a nonlinear ZDF/TPT SVF with topology-aware saturation.

    Architecture follows Vital's dirty-filter pattern: one strategic
    saturation at the feedback summation point (with ADAA anti-aliasing),
    algebraicSat on integrator states, and bidirectional drive/resonance
    interaction.  This replaces the previous three-independent-tanh design.
    """
    n = signal.shape[0]
    filtered = np.empty(n, dtype=np.float64)
    low_state = 0.0
    band_state = 0.0
    use_precomputed = precomputed_g >= 0.0

    # --- Drive/resonance interaction (Vital pattern) ---
    # Higher resonance reduces effective drive (prevents harsh peaks).
    # Drive adds subtle resonance boost (warm emphasis).
    q_from_damping = 1.0 / max(damping, 1e-6)
    effective_drive = max(0.05, filter_drive) / (
        q_from_damping * q_from_damping * 0.3 + 1.0
    )
    drive_gain = 1.0 + 2.5 * effective_drive
    # Drive reduces damping slightly → subtle resonance boost under drive
    effective_damping = damping * (1.0 - min(0.15, filter_drive * 0.2))

    # Level compensation for loudness consistency
    compensation = 1.0 / (1.0 + 0.25 * filter_drive)

    # Asymmetric bias for even harmonics (envelope-tracked)
    bias_amount = even_harmonics * 0.3

    # ADAA state: previous input to the saturation function
    prev_sat_input = 0.0

    for i in range(n):
        if use_precomputed:
            g = precomputed_g
        else:
            # Clamp cutoff to _NYQUIST_CLAMP_RATIO of sample rate
            fc = min(cutoff_profile[i], nyquist_limit)
            g = math.tan(math.pi * fc / sample_rate)

        sample = signal[i]

        # --- Single strategic saturation at the feedback summation point ---
        # Combines drive + feedback into one nonlinear evaluation.
        # The feedback is compressed by the saturation, creating natural
        # drive/resonance interaction: louder signals see less effective
        # resonance, matching real analog filter behavior.
        feedback_sum = low_state + (2.0 * effective_damping + g) * band_state
        sat_input = drive_gain * sample - feedback_sum

        # Apply asymmetric bias for even harmonics if enabled.
        # Envelope estimate from the absolute sample value.
        if bias_amount > 0.0:
            env_est = math.fabs(sample)
            sat_input = sat_input + bias_amount * env_est

        # Bound sat_input to prevent numerical issues in ADAA at extreme
        # resonance/frequency settings.  ±10 is well into full tanh
        # saturation while keeping log_cosh numerically precise.
        if sat_input > 10.0:
            sat_input = 10.0
        elif sat_input < -10.0:
            sat_input = -10.0

        # First-order ADAA tanh: eliminates aliasing from the nonlinearity
        saturated = _adaa_tanh(sat_input, prev_sat_input)
        prev_sat_input = sat_input

        # Remove the bias DC component from the saturated signal
        if bias_amount > 0.0:
            saturated = saturated - _adaa_tanh(
                bias_amount * math.fabs(sample),
                bias_amount * math.fabs(signal[max(0, i - 1)]),
            )

        # --- SVF state update with algebraicSat state limiting ---
        # saturated already accounts for the full feedback (including
        # low_state), so we must NOT subtract low_state again here.
        high = saturated / (1.0 + 2.0 * effective_damping * g + g * g)
        band = g * high + band_state
        low = g * band + low_state

        # AlgebraicSat on state updates AND outputs: transparent below ~1.0,
        # prevents state explosion at high levels.  The state limiting is
        # the primary stability mechanism; output limiting is a safety net.
        band_state = _algebraic_sat(band + g * high)
        low_state = _algebraic_sat(low + g * band)
        band = _algebraic_sat(band)
        low = _algebraic_sat(low)
        high = _algebraic_sat(high)

        if mode_int == _LP:
            output = low
        elif mode_int == _BP:
            output = band
        elif mode_int == _HP:
            output = high
        else:
            output = low + high

        # Gentle output limiting: algebraicSat prevents runaway at
        # extreme Q + drive without harsh clipping.
        filtered[i] = _algebraic_sat(output)

    for i in range(n):
        filtered[i] *= compensation

    return filtered


def apply_zdf_svf(
    signal: np.ndarray,
    *,
    cutoff_profile: np.ndarray,
    resonance_q: float = 0.707,
    sample_rate: int,
    filter_mode: str,
    filter_drive: float,
    filter_even_harmonics: float = 0.0,
) -> np.ndarray:
    """Apply a per-sample ZDF/TPT state-variable filter.

    When ``filter_drive=0`` the filter runs a fully linear path — no soft-clipping
    anywhere in the loop.  Nonlinear processing only activates for ``filter_drive>0``.

    The driven path uses topology-aware saturation: a single ADAA-antialiased
    tanh at the feedback summation point, algebraicSat on integrator states,
    and bidirectional drive/resonance interaction.

    Args:
        signal: Input audio array.
        cutoff_profile: Per-sample cutoff frequency in Hz (same length as signal).
        resonance_q: Filter Q value (>= 0.5). Q=0.707 is Butterworth (no resonance
            peak). Q=1 is a gentle peak; Q=4+ approaches self-oscillation depending
            on drive.
        sample_rate: Audio sample rate in Hz.
        filter_mode: One of ``"lowpass"``, ``"bandpass"``, ``"highpass"``, ``"notch"``.
        filter_drive: Non-negative drive amount; 0.0 means fully linear/clean.
        filter_even_harmonics: Asymmetric bias amount (0.0-0.5).  Adds even
            harmonics via envelope-tracked DC bias before saturation.  0.0 =
            symmetric/odd harmonics only (default, backward compatible).
    """
    q = max(0.5, float(resonance_q))
    damping = 1.0 / q
    mode_int = _MODE_STR_TO_INT.get(filter_mode, _LP)

    sig = np.asarray(signal, dtype=np.float64)
    cutoff = np.asarray(cutoff_profile, dtype=np.float64)

    # Pre-compute g when cutoff is constant to avoid per-sample tan() calls.
    # Clamp to _NYQUIST_CLAMP_RATIO (~45%) of sample rate to prevent tan()
    # from blowing up near Nyquist and to keep the SVF stable.
    nyquist_limit = sample_rate * _NYQUIST_CLAMP_RATIO
    if cutoff.size > 0 and np.all(cutoff == cutoff[0]):
        clamped_fc = min(float(cutoff[0]), nyquist_limit)
        precomputed_g = math.tan(math.pi * clamped_fc / sample_rate)
    else:
        # Clamp in the profile for non-constant cutoff
        cutoff = np.minimum(cutoff, nyquist_limit)
        precomputed_g = -1.0  # sentinel: compute per-sample

    linear_filtered = _apply_linear_zdf_svf(
        sig,
        cutoff,
        damping,
        sample_rate,
        mode_int,
        precomputed_g,
        nyquist_limit,
    )

    if filter_drive <= 0.0:
        return linear_filtered

    even_h = max(0.0, min(0.5, float(filter_even_harmonics)))

    driven_filtered = _apply_driven_zdf_svf(
        sig,
        cutoff,
        damping,
        sample_rate,
        mode_int,
        filter_drive,
        precomputed_g,
        even_h,
        nyquist_limit,
    )
    drive_blend = min(1.0, 0.75 * (filter_drive**1.3))
    return ((1.0 - drive_blend) * linear_filtered) + (drive_blend * driven_filtered)
