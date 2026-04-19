"""Shared DSP filter primitives used by multiple synth engines."""

from __future__ import annotations

import hashlib
import math

import numba
import numpy as np

_SUPPORTED_FILTER_MODES = {"lowpass", "bandpass", "highpass", "notch"}
_SUPPORTED_FILTER_TOPOLOGIES = {"svf", "ladder", "sallen_key", "cascade"}
_SUPPORTED_LADDER_SOLVERS = {"adaa", "newton"}

# Newton iteration defaults used when the engine does not supply quality-driven
# overrides.  Four iterations with 1e-9 tolerance closes the delay-free
# feedback loop far below audio SNR.  A looser tolerance like 1e-5 prematurely
# short-circuits Newton when the solver is seeded from silence (bootstrap noise
# at 1e-6 is already below the tolerance), which would kill self-oscillation.
_DEFAULT_NEWTON_MAX_ITERS: int = 4
_DEFAULT_NEWTON_TOLERANCE: float = 1e-9

# Amplitude of bootstrap noise injected into feedback summations so that
# high-Q ladders / self-oscillating SVFs can wake up from exact silence.
# 1e-6 is ~120 dB below unity — inaudible, but enough to seed feedback loops.
_BOOTSTRAP_NOISE_AMP: float = 1e-6

# Golden-ratio increment for the Weyl sequence used by filter kernels as an
# inline bootstrap-noise source.  The per-sample update is
#     state = (state + _WEYL_INCREMENT) mod 2^64
#     dither = (float(state) / 2^64 - 0.5) * (2 * _BOOTSTRAP_NOISE_AMP)
# yielding a deterministic, low-correlation sequence in [-amp, +amp] without
# allocating a full-length float64 buffer per filter call.
_WEYL_INCREMENT: np.uint64 = np.uint64(0x9E3779B97F4A7C15)
_WEYL_SCALE: float = 1.0 / float(2**64)


def _bootstrap_seed(signal: np.ndarray, tag: str) -> np.uint64:
    """Return a deterministic uint64 seed for inline Weyl-sequence bootstrap noise.

    The seed is derived from a hash of the signal's shape and a few anchor
    samples plus ``tag``, so identical inputs produce identical seeds
    (seedable / reproducible) while different filter call sites stay
    decorrelated.  Kernels use this scalar seed to drive a per-sample
    golden-ratio Weyl sequence in place of a preallocated noise buffer —
    saves ~8n bytes of float64 allocation per filter call.
    """
    material_parts = [str(signal.shape[0]).encode("utf-8"), tag.encode("utf-8")]
    if signal.size > 0:
        # Hash a small deterministic fingerprint of the signal to avoid
        # hashing megabytes of samples for every filter call.
        stride = max(1, signal.size // 16)
        sample = signal[::stride][:16].tobytes()
        material_parts.append(sample)
    digest = hashlib.sha256(b"|".join(material_parts)).digest()
    return np.uint64(int.from_bytes(digest[:8], byteorder="big", signed=False))


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
    bootstrap_seed: np.uint64,
    has_bootstrap: bool,
    morph: float = 0.0,
    feedback_amount: float = 0.0,
    feedback_saturation: float = 0.0,
) -> np.ndarray:
    """Apply the fully linear ZDF/TPT state-variable filter (numba-compiled)."""
    n = signal.shape[0]
    filtered = np.empty(n, dtype=np.float64)
    low_state = 0.0
    band_state = 0.0
    use_precomputed = precomputed_g >= 0.0

    has_ext_fb = feedback_amount > 0.0
    ext_fb_drive = 1.0 + 3.0 * feedback_saturation
    prev_output = 0.0
    weyl_state = bootstrap_seed

    use_morph = morph != 0.0
    lp_w = 0.0
    bp_w = 0.0
    hp_w = 0.0
    notch_w = 0.0
    if use_morph:
        pos = (mode_int + morph) % 4.0
        idx = int(pos)
        frac = pos - idx
        if idx == 0:
            lp_w = 1.0 - frac
            bp_w = frac
            hp_w = 0.0
            notch_w = 0.0
        elif idx == 1:
            lp_w = 0.0
            bp_w = 1.0 - frac
            hp_w = frac
            notch_w = 0.0
        elif idx == 2:
            lp_w = 0.0
            bp_w = 0.0
            hp_w = 1.0 - frac
            notch_w = frac
        else:
            lp_w = frac
            bp_w = 0.0
            hp_w = 0.0
            notch_w = 1.0 - frac

    for i in range(n):
        if use_precomputed:
            g = precomputed_g
        else:
            fc = min(cutoff_profile[i], nyquist_limit)
            g = math.tan(math.pi * fc / sample_rate)
        sample = signal[i]
        if has_ext_fb:
            sample = sample + feedback_amount * math.tanh(ext_fb_drive * prev_output)
            if has_bootstrap:
                weyl_state = weyl_state + _WEYL_INCREMENT
                dither = (float(weyl_state) * _WEYL_SCALE - 0.5) * (
                    _BOOTSTRAP_NOISE_AMP * 2.0
                )
                sample = sample + dither
        high = (sample - (2.0 * damping + g) * band_state - low_state) / (
            1.0 + 2.0 * damping * g + g * g
        )
        band = g * high + band_state
        low = g * band + low_state
        band_state = band + g * high
        low_state = low + g * band

        if use_morph:
            notch = low + high
            output = lp_w * low + bp_w * band + hp_w * high + notch_w * notch
        elif mode_int == _LP:
            output = low
        elif mode_int == _BP:
            output = band
        elif mode_int == _HP:
            output = high
        else:
            output = low + high

        prev_output = output
        filtered[i] = output

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
    bootstrap_seed: np.uint64,
    has_bootstrap: bool,
    morph: float = 0.0,
    feedback_amount: float = 0.0,
    feedback_saturation: float = 0.0,
) -> np.ndarray:
    """Apply a nonlinear ZDF/TPT SVF with topology-aware saturation.

    Architecture follows Vital's dirty-filter pattern: one strategic
    saturation at the feedback summation point (with ADAA anti-aliasing),
    algebraicSat on integrator states, and bidirectional drive/resonance
    interaction.
    """
    n = signal.shape[0]
    filtered = np.empty(n, dtype=np.float64)
    low_state = 0.0
    band_state = 0.0
    use_precomputed = precomputed_g >= 0.0

    has_ext_fb = feedback_amount > 0.0
    ext_fb_drive = 1.0 + 3.0 * feedback_saturation
    prev_output = 0.0
    weyl_state = bootstrap_seed

    use_morph = morph != 0.0
    lp_w = 0.0
    bp_w = 0.0
    hp_w = 0.0
    notch_w = 0.0
    if use_morph:
        pos = (mode_int + morph) % 4.0
        idx = int(pos)
        frac = pos - idx
        if idx == 0:
            lp_w = 1.0 - frac
            bp_w = frac
        elif idx == 1:
            bp_w = 1.0 - frac
            hp_w = frac
        elif idx == 2:
            hp_w = 1.0 - frac
            notch_w = frac
        else:
            notch_w = 1.0 - frac
            lp_w = frac

    q_from_damping = 1.0 / max(damping, 1e-6)
    effective_drive = max(0.05, filter_drive) / (
        q_from_damping * q_from_damping * 0.3 + 1.0
    )
    drive_gain = 1.0 + 2.5 * effective_drive
    effective_damping = damping * (1.0 - min(0.15, filter_drive * 0.2))

    compensation = 1.0 / (1.0 + 0.25 * filter_drive)

    bias_amount = even_harmonics * 0.3

    prev_sat_input = 0.0

    for i in range(n):
        if use_precomputed:
            g = precomputed_g
        else:
            fc = min(cutoff_profile[i], nyquist_limit)
            g = math.tan(math.pi * fc / sample_rate)

        sample = signal[i]
        if has_ext_fb:
            sample = sample + feedback_amount * math.tanh(ext_fb_drive * prev_output)
            if has_bootstrap:
                weyl_state = weyl_state + _WEYL_INCREMENT
                sample = sample + (float(weyl_state) * _WEYL_SCALE - 0.5) * (
                    _BOOTSTRAP_NOISE_AMP * 2.0
                )

        feedback_sum = low_state + (2.0 * effective_damping + g) * band_state
        sat_input = drive_gain * sample - feedback_sum
        if has_bootstrap:
            weyl_state = weyl_state + _WEYL_INCREMENT
            sat_input = sat_input + (float(weyl_state) * _WEYL_SCALE - 0.5) * (
                _BOOTSTRAP_NOISE_AMP * 2.0
            )

        if bias_amount > 0.0:
            env_est = math.fabs(sample)
            sat_input = sat_input + bias_amount * env_est

        if sat_input > 10.0:
            sat_input = 10.0
        elif sat_input < -10.0:
            sat_input = -10.0

        saturated = _adaa_tanh(sat_input, prev_sat_input)
        prev_sat_input = sat_input

        if bias_amount > 0.0:
            saturated = saturated - _adaa_tanh(
                bias_amount * math.fabs(sample),
                bias_amount * math.fabs(signal[max(0, i - 1)]),
            )

        high = saturated / (1.0 + 2.0 * effective_damping * g + g * g)
        band = g * high + band_state
        low = g * band + low_state

        band_state = _algebraic_sat(band + g * high)
        low_state = _algebraic_sat(low + g * band)
        band = _algebraic_sat(band)
        low = _algebraic_sat(low)
        high = _algebraic_sat(high)

        if use_morph:
            notch = low + high
            output = lp_w * low + bp_w * band + hp_w * high + notch_w * notch
        elif mode_int == _LP:
            output = low
        elif mode_int == _BP:
            output = band
        elif mode_int == _HP:
            output = high
        else:
            output = low + high

        prev_output = _algebraic_sat(output)
        filtered[i] = prev_output

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
    feedback_amount: float = 0.0,
    feedback_saturation: float = 0.0,
) -> np.ndarray:
    """Apply a per-sample ZDF/TPT state-variable filter.

    When ``filter_drive=0`` the filter runs a fully linear path — no soft-clipping
    anywhere in the loop.  Nonlinear processing only activates for ``filter_drive>0``.

    The driven path uses topology-aware saturation: a single ADAA-antialiased
    tanh at the feedback summation point, algebraicSat on integrator states,
    and bidirectional drive/resonance interaction.

    When ``feedback_amount > 0``, post-filter output feeds back to the pre-filter
    input through a saturating tanh stage (Minimoog-style mixer feedback).
    """
    q = max(0.5, float(resonance_q))
    damping = 1.0 / q
    mode_int = _MODE_STR_TO_INT.get(filter_mode, _LP)

    sig = np.asarray(signal, dtype=np.float64)
    cutoff = np.asarray(cutoff_profile, dtype=np.float64)

    nyquist_limit = sample_rate * _NYQUIST_CLAMP_RATIO
    if cutoff.size > 0 and np.all(cutoff == cutoff[0]):
        clamped_fc = min(float(cutoff[0]), nyquist_limit)
        precomputed_g = math.tan(math.pi * clamped_fc / sample_rate)
    else:
        cutoff = np.minimum(cutoff, nyquist_limit)
        precomputed_g = -1.0

    fb_amt = max(0.0, float(feedback_amount))
    fb_sat = max(0.0, float(feedback_saturation))
    bootstrap_seed = _bootstrap_seed(sig, "svf")
    has_bootstrap = sig.shape[0] > 0

    linear_filtered = _apply_linear_zdf_svf(
        sig,
        cutoff,
        damping,
        sample_rate,
        mode_int,
        precomputed_g,
        nyquist_limit,
        bootstrap_seed,
        has_bootstrap,
        0.0,
        fb_amt,
        fb_sat,
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
        bootstrap_seed,
        has_bootstrap,
        0.0,
        fb_amt,
        fb_sat,
    )
    drive_blend = min(1.0, 0.75 * (filter_drive**1.3))
    return ((1.0 - drive_blend) * linear_filtered) + (drive_blend * driven_filtered)


def _apply_svf_with_morph(
    signal: np.ndarray,
    *,
    cutoff_profile: np.ndarray,
    resonance_q: float = 0.707,
    sample_rate: int,
    filter_mode: str,
    filter_drive: float,
    filter_even_harmonics: float = 0.0,
    feedback_amount: float = 0.0,
    feedback_saturation: float = 0.0,
    morph: float,
) -> np.ndarray:
    """SVF path with mode morphing -- calls inner loops directly."""
    q = max(0.5, float(resonance_q))
    damping = 1.0 / q
    mode_int = _MODE_STR_TO_INT.get(filter_mode, _LP)

    sig = np.asarray(signal, dtype=np.float64)
    cutoff = np.asarray(cutoff_profile, dtype=np.float64)

    nyquist_limit = sample_rate * _NYQUIST_CLAMP_RATIO
    if cutoff.size > 0 and np.all(cutoff == cutoff[0]):
        clamped_fc = min(float(cutoff[0]), nyquist_limit)
        precomputed_g = math.tan(math.pi * clamped_fc / sample_rate)
    else:
        cutoff = np.minimum(cutoff, nyquist_limit)
        precomputed_g = -1.0

    fb_amt = max(0.0, float(feedback_amount))
    fb_sat = max(0.0, float(feedback_saturation))
    bootstrap_seed = _bootstrap_seed(sig, "svf_morph")
    has_bootstrap = sig.shape[0] > 0

    linear_filtered = _apply_linear_zdf_svf(
        sig,
        cutoff,
        damping,
        sample_rate,
        mode_int,
        precomputed_g,
        nyquist_limit,
        bootstrap_seed,
        has_bootstrap,
        morph,
        fb_amt,
        fb_sat,
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
        bootstrap_seed,
        has_bootstrap,
        morph,
        fb_amt,
        fb_sat,
    )
    drive_blend = min(1.0, 0.75 * (filter_drive**1.3))
    return ((1.0 - drive_blend) * linear_filtered) + (drive_blend * driven_filtered)


# ---------------------------------------------------------------------------
# Moog-style 4-pole ladder filter (Huovilainen improved model)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _apply_ladder_filter_inner(
    signal: np.ndarray,
    cutoff_profile: np.ndarray,
    resonance_k: float,
    sample_rate: int,
    mode_int: int,
    filter_drive: float,
    bass_compensation: float,
    even_harmonics: float,
    precomputed_g: float,
    nyquist_limit: float,
    bootstrap_seed: np.uint64,
    has_bootstrap: bool,
    feedback_amount: float = 0.0,
    feedback_saturation: float = 0.0,
    morph: float = 0.0,
) -> np.ndarray:
    """Huovilainen improved 4-pole ladder with per-stage ZDF/TPT integrators."""
    n = signal.shape[0]
    filtered = np.empty(n, dtype=np.float64)
    use_precomputed = precomputed_g >= 0.0

    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0

    driven = filter_drive > 0.0
    drive_gain = 1.0 + 2.0 * filter_drive if driven else 1.0
    bias_amount = even_harmonics * 0.3

    has_ext_fb = feedback_amount > 0.0
    ext_fb_drive = 1.0 + 3.0 * feedback_saturation
    prev_ext_output = 0.0
    weyl_state = bootstrap_seed

    prev_fb_input = 0.0

    for i in range(n):
        if use_precomputed:
            g = precomputed_g
        else:
            fc = min(cutoff_profile[i], nyquist_limit)
            g = math.tan(math.pi * fc / sample_rate)

        fb_input = resonance_k * s3
        fb = _adaa_tanh(fb_input, prev_fb_input)
        prev_fb_input = fb_input

        inp = signal[i]
        if has_ext_fb:
            inp = inp + feedback_amount * math.tanh(ext_fb_drive * prev_ext_output)
        if has_bootstrap:
            weyl_state = weyl_state + _WEYL_INCREMENT
            inp = inp + (float(weyl_state) * _WEYL_SCALE - 0.5) * (
                _BOOTSTRAP_NOISE_AMP * 2.0
            )
        inp = inp - fb

        if driven:
            inp = drive_gain * inp

        if bias_amount > 0.0:
            inp = inp + bias_amount * math.fabs(signal[i])

        v0 = g * (inp - s0) / (1.0 + g)
        y0 = v0 + s0
        s0 = y0 + v0
        if driven:
            y0 = _algebraic_sat(y0)
            s0 = _algebraic_sat(s0)

        v1 = g * (y0 - s1) / (1.0 + g)
        y1 = v1 + s1
        s1 = y1 + v1
        if driven:
            y1 = _algebraic_sat(y1)
            s1 = _algebraic_sat(s1)

        v2 = g * (y1 - s2) / (1.0 + g)
        y2 = v2 + s2
        s2 = y2 + v2
        if driven:
            y2 = _algebraic_sat(y2)
            s2 = _algebraic_sat(s2)

        v3 = g * (y2 - s3) / (1.0 + g)
        y3 = v3 + s3
        s3 = y3 + v3
        if driven:
            y3 = _algebraic_sat(y3)
            s3 = _algebraic_sat(s3)

        if bias_amount > 0.0:
            y3 = y3 - bias_amount * math.fabs(signal[i]) * 0.25

        if morph <= 0.0:
            if mode_int == _LP:
                output = y3
            elif mode_int == _BP:
                output = y1 - y3
            else:
                output = inp - y3
        else:
            m = min(morph, 3.0)
            if mode_int == _LP:
                if m <= 1.0:
                    output = (1.0 - m) * y3 + m * y2
                elif m <= 2.0:
                    f = m - 1.0
                    output = (1.0 - f) * y2 + f * y1
                else:
                    f = m - 2.0
                    output = (1.0 - f) * y1 + f * y0
            elif mode_int == _BP:
                bp4 = y1 - y3
                bp3 = y0 - y2
                bp2 = y0 - y1
                if m <= 1.0:
                    output = (1.0 - m) * bp4 + m * bp3
                elif m <= 2.0:
                    f = m - 1.0
                    output = (1.0 - f) * bp3 + f * bp2
                else:
                    f = m - 2.0
                    output = (1.0 - f) * bp2 + f * y0
            else:
                hp4 = inp - y3
                hp3 = inp - y2
                hp2 = inp - y1
                hp1 = inp - y0
                if m <= 1.0:
                    output = (1.0 - m) * hp4 + m * hp3
                elif m <= 2.0:
                    f = m - 1.0
                    output = (1.0 - f) * hp3 + f * hp2
                else:
                    f = m - 2.0
                    output = (1.0 - f) * hp2 + f * hp1

        if bass_compensation > 0.0:
            output = output + bass_compensation * (resonance_k / 4.0) * (y0 - y3)

        prev_ext_output = output
        filtered[i] = output

    return filtered


@numba.njit(cache=True)
def _apply_ladder_filter_newton_inner(
    signal: np.ndarray,
    cutoff_profile: np.ndarray,
    resonance_k: float,
    sample_rate: int,
    mode_int: int,
    filter_drive: float,
    bass_compensation: float,
    even_harmonics: float,
    precomputed_g: float,
    nyquist_limit: float,
    bootstrap_seed: np.uint64,
    has_bootstrap: bool,
    feedback_amount: float,
    feedback_saturation: float,
    morph: float,
    max_iters: int,
    tolerance: float,
) -> np.ndarray:
    """Newton-iterated nonlinear ZDF ladder filter.

    The standard Moog-style 4-stage ladder with tanh on the feedback path has
    a single per-sample implicit equation of the form

        y3 = stage4( stage3( stage2( stage1( x - tanh(k * y3) ) ) ) )

    where each stage is a ZDF 1-pole.  For a ZDF 1-pole with gain ``g`` and
    state ``s``, the input-to-output map is affine in the stage input:

        v  = g * (u - s) / (1 + g)
        y  = v + s
           = alpha * u + (1 - alpha) * s,         alpha = g / (1 + g)

    so the cascade of four such stages, operating on input
    ``u = drive_gain * x_pre - tanh(k * y3)``, is also affine in that input:

        y3 = alpha^4 * u + beta,

    where ``beta`` is the Horner-style sum of ``(1 - alpha)`` times the stage
    state offsets, computed once per sample.  Define ``A = alpha^4`` and
    ``C = drive_gain * A * k`` and let ``B = A * drive_gain * x_pre + beta``
    (with ``x_pre`` already including the external feedback summation and
    bootstrap noise).  Then the delay-free feedback loop collapses to

        F(y3) = y3 + A * tanh(k * y3) - B = 0.

    Newton:  y3_{m+1} = y3_m - F(y3_m) / F'(y3_m), where

        F'(y3) = 1 + A * k * sech^2(k * y3).

    F is strictly monotonically increasing (F' > 0 for all y3), and F'' has
    the sign of ``-y3``, so Newton is globally convergent from any initial
    guess and has no bracketing issues.

    Once y3 is known, we recover the intermediate stage outputs y0..y2 and
    the per-stage new states exactly as the ADAA ladder does.
    """
    n = signal.shape[0]
    filtered = np.empty(n, dtype=np.float64)
    use_precomputed = precomputed_g >= 0.0

    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    y3_prev = 0.0  # warm-start the Newton initial guess across samples.

    driven = filter_drive > 0.0
    drive_gain = 1.0 + 2.0 * filter_drive if driven else 1.0
    bias_amount = even_harmonics * 0.3

    has_ext_fb = feedback_amount > 0.0
    ext_fb_drive = 1.0 + 3.0 * feedback_saturation
    prev_ext_output = 0.0
    weyl_state = bootstrap_seed

    for i in range(n):
        if use_precomputed:
            g = precomputed_g
        else:
            fc = min(cutoff_profile[i], nyquist_limit)
            g = math.tan(math.pi * fc / sample_rate)

        alpha = g / (1.0 + g)
        one_minus_alpha = 1.0 - alpha
        A = alpha * alpha * alpha * alpha

        x_pre = signal[i]
        if has_ext_fb:
            x_pre = x_pre + feedback_amount * math.tanh(ext_fb_drive * prev_ext_output)
        if has_bootstrap:
            weyl_state = weyl_state + _WEYL_INCREMENT
            x_pre = x_pre + (float(weyl_state) * _WEYL_SCALE - 0.5) * (
                _BOOTSTRAP_NOISE_AMP * 2.0
            )
        if bias_amount > 0.0:
            x_pre = x_pre + bias_amount * math.fabs(signal[i])

        u_no_fb = drive_gain * x_pre

        # beta = alpha^3*(1-alpha)*s0 + alpha^2*(1-alpha)*s1
        #      + alpha*(1-alpha)*s2 + (1-alpha)*s3
        beta = one_minus_alpha * (((alpha * s0 + s1) * alpha + s2) * alpha + s3)
        B = A * u_no_fb + beta

        # Newton solve F(y3) = y3 + A * tanh(k * y3) - B = 0.
        y3 = y3_prev
        Ak = A * resonance_k
        for _ in range(max_iters):
            ky = resonance_k * y3
            th = math.tanh(ky)
            sech2 = 1.0 - th * th
            F = y3 + A * th - B
            if math.fabs(F) < tolerance:
                break
            Fp = 1.0 + Ak * sech2
            y3 = y3 - F / Fp

        y3_prev = y3

        # Now recover y0..y2 and update stage states using the resolved y3.
        fb = math.tanh(resonance_k * y3)
        u = u_no_fb - fb

        v0 = g * (u - s0) / (1.0 + g)
        y0 = v0 + s0
        s0 = y0 + v0

        v1 = g * (y0 - s1) / (1.0 + g)
        y1 = v1 + s1
        s1 = y1 + v1

        v2 = g * (y1 - s2) / (1.0 + g)
        y2 = v2 + s2
        s2 = y2 + v2

        v3 = g * (y2 - s3) / (1.0 + g)
        # y3 is already the Newton-solved value; use it for state update too
        # so the implicit equation is exactly satisfied at the state level.
        s3 = y3 + v3

        if driven:
            y0 = _algebraic_sat(y0)
            y1 = _algebraic_sat(y1)
            y2 = _algebraic_sat(y2)
            s0 = _algebraic_sat(s0)
            s1 = _algebraic_sat(s1)
            s2 = _algebraic_sat(s2)
            s3 = _algebraic_sat(s3)

        if bias_amount > 0.0:
            y3 = y3 - bias_amount * math.fabs(signal[i]) * 0.25

        if morph <= 0.0:
            if mode_int == _LP:
                output = y3
            elif mode_int == _BP:
                output = y1 - y3
            else:
                output = u - y3
        else:
            m = min(morph, 3.0)
            if mode_int == _LP:
                if m <= 1.0:
                    output = (1.0 - m) * y3 + m * y2
                elif m <= 2.0:
                    f = m - 1.0
                    output = (1.0 - f) * y2 + f * y1
                else:
                    f = m - 2.0
                    output = (1.0 - f) * y1 + f * y0
            elif mode_int == _BP:
                bp4 = y1 - y3
                bp3 = y0 - y2
                bp2 = y0 - y1
                if m <= 1.0:
                    output = (1.0 - m) * bp4 + m * bp3
                elif m <= 2.0:
                    f = m - 1.0
                    output = (1.0 - f) * bp3 + f * bp2
                else:
                    f = m - 2.0
                    output = (1.0 - f) * bp2 + f * y0
            else:
                hp4 = u - y3
                hp3 = u - y2
                hp2 = u - y1
                hp1 = u - y0
                if m <= 1.0:
                    output = (1.0 - m) * hp4 + m * hp3
                elif m <= 2.0:
                    f = m - 1.0
                    output = (1.0 - f) * hp3 + f * hp2
                else:
                    f = m - 2.0
                    output = (1.0 - f) * hp2 + f * hp1

        if bass_compensation > 0.0:
            output = output + bass_compensation * (resonance_k / 4.0) * (y0 - y3)

        prev_ext_output = output
        filtered[i] = output

    return filtered


def _apply_ladder_filter(
    signal: np.ndarray,
    *,
    cutoff_profile: np.ndarray,
    resonance_q: float,
    sample_rate: int,
    filter_mode: str,
    filter_drive: float,
    bass_compensation: float,
    filter_even_harmonics: float,
    feedback_amount: float = 0.0,
    feedback_saturation: float = 0.0,
    filter_morph: float = 0.0,
    solver: str = "adaa",
    max_newton_iters: int = _DEFAULT_NEWTON_MAX_ITERS,
    newton_tolerance: float = _DEFAULT_NEWTON_TOLERANCE,
) -> np.ndarray:
    """Apply a Moog-style 4-pole ladder filter."""
    q = max(0.5, float(resonance_q))

    # ADAA uses delayed feedback (k*s3_prev through tanh), which destabilizes
    # the discrete-time loop below the analytical self-oscillation threshold
    # of the true ZDF ladder (~k=4.0).  Clamping k to 3.98 keeps that path
    # musically self-oscillating without blowing up.
    #
    # The Newton path solves the instantaneous feedback loop implicitly, so
    # it matches the analytical k_osc of ~4.0 and stays strictly stable below
    # it.  To give the user the same subjective "resonance ramps into self-
    # oscillation at high q" behavior, we scale k slightly higher for Newton:
    # k_newton = 4.2 * (1 - 1/(2q)).  At q=10 this lands just below osc
    # threshold; at q=20+ it pushes past and sings.  The tanh saturates the
    # feedback automatically so the over-threshold k does not blow up.
    k_adaa = max(0.0, min(3.98, 4.0 * (1.0 - 1.0 / (2.0 * q))))
    k_newton = max(0.0, min(4.25, 4.2 * (1.0 - 1.0 / (2.0 * q))))

    mode_int = _MODE_STR_TO_INT.get(filter_mode, _LP)
    if mode_int == _NOTCH:
        mode_int = _LP

    sig = np.asarray(signal, dtype=np.float64)
    cutoff = np.asarray(cutoff_profile, dtype=np.float64)
    even_h = max(0.0, min(0.5, float(filter_even_harmonics)))

    nyquist_limit = sample_rate * _NYQUIST_CLAMP_RATIO
    if cutoff.size > 0 and np.all(cutoff == cutoff[0]):
        clamped_fc = min(float(cutoff[0]), nyquist_limit)
        precomputed_g = math.tan(math.pi * clamped_fc / sample_rate)
    else:
        cutoff = np.minimum(cutoff, nyquist_limit)
        precomputed_g = -1.0

    bootstrap_seed = _bootstrap_seed(sig, "ladder")
    has_bootstrap = sig.shape[0] > 0
    solver_name = str(solver)
    if solver_name not in _SUPPORTED_LADDER_SOLVERS:
        raise ValueError(f"Unknown filter_solver: {solver_name!r}")

    if solver_name == "newton":
        iters = max(1, int(max_newton_iters))
        tol = max(1e-9, float(newton_tolerance))
        return _apply_ladder_filter_newton_inner(
            sig,
            cutoff,
            k_newton,
            sample_rate,
            mode_int,
            max(0.0, float(filter_drive)),
            max(0.0, float(bass_compensation)),
            even_h,
            precomputed_g,
            nyquist_limit,
            bootstrap_seed,
            has_bootstrap,
            max(0.0, float(feedback_amount)),
            max(0.0, float(feedback_saturation)),
            max(0.0, min(3.0, float(filter_morph))),
            iters,
            tol,
        )

    return _apply_ladder_filter_inner(
        sig,
        cutoff,
        k_adaa,
        sample_rate,
        mode_int,
        max(0.0, float(filter_drive)),
        max(0.0, float(bass_compensation)),
        even_h,
        precomputed_g,
        nyquist_limit,
        bootstrap_seed,
        has_bootstrap,
        max(0.0, float(feedback_amount)),
        max(0.0, float(feedback_saturation)),
        max(0.0, min(3.0, float(filter_morph))),
    )


# ---------------------------------------------------------------------------
# Sallen-Key 2-pole (biting CEM-3320 character, "Diva Bite")
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _apply_sallen_key_inner(
    signal: np.ndarray,
    cutoff_profile: np.ndarray,
    damping: float,
    sample_rate: int,
    mode_int: int,
    filter_drive: float,
    precomputed_g: float,
    nyquist_limit: float,
    bootstrap_seed: np.uint64,
    has_bootstrap: bool,
    feedback_amount: float,
    feedback_saturation: float,
) -> np.ndarray:
    """ZDF/TPT 2-pole Sallen-Key with biting CEM-3320 character.

    Implementation note: a structurally-accurate Sallen-Key with per-sample
    implicit positive-feedback solve is mathematically solvable (the
    delay-free loop collapses to a scalar equation), but the cross-sample
    state recursion has an eigenvalue that can cross the unit circle at
    moderate Q even when the instantaneous solve is well-conditioned.  This
    is the same family of issue Vadim Zavalishin discusses in *The Art of VA
    Filter Design* — the SK topology benefits from state-space prewarping
    that a simple trapezoidal discretization doesn't provide.

    In practice, the cleanest way to deliver the Sallen-Key voice in ZDF form
    is to use the proven TPT 2-pole SVF math (same integrator structure as
    ``_apply_linear_zdf_svf``) with two character tweaks:

    1. **Tighter Q mapping**: at a user-facing ``resonance_q`` the SK
       damping is ``1 / (q * 1.4)`` — the peak is narrower and spikier than
       the main SVF path at the same Q.
    2. **Pre-filter asymmetric soft-clip** under drive: models the even-
       harmonic bias of a discrete-transistor SK input stage.

    The low/band/high outputs of the SVF are mapped exactly to SK's LP/BP/HP
    — the frequency response is identical to a 2-pole SK at the same pole
    locations; what differs is the *character* (peak shape, drive behavior).
    This preserves the full ZDF/TPT benefits (proper integrator states,
    warp-free corner at cutoff, no unit-delay feedback) without the
    eigenvalue-drift risk of the literal SK cross-sample recursion.
    """
    n = signal.shape[0]
    filtered = np.empty(n, dtype=np.float64)
    use_precomputed = precomputed_g >= 0.0

    low_state = 0.0
    band_state = 0.0

    driven = filter_drive > 0.0
    drive_gain = 1.0 + 1.5 * filter_drive if driven else 1.0
    # SK bite: positive input bias produces asymmetric waveshaping (even
    # harmonics).  Applied pre-filter so the color rides the signal envelope.
    bias_amount = 0.08 * filter_drive if driven else 0.0

    has_ext_fb = feedback_amount > 0.0
    ext_fb_drive = 1.0 + 3.0 * feedback_saturation
    prev_output = 0.0
    weyl_state = bootstrap_seed

    for i in range(n):
        if use_precomputed:
            g = precomputed_g
        else:
            fc = min(cutoff_profile[i], nyquist_limit)
            g = math.tan(math.pi * fc / sample_rate)

        x = signal[i]
        if has_ext_fb:
            x = x + feedback_amount * math.tanh(ext_fb_drive * prev_output)
        if has_bootstrap:
            weyl_state = weyl_state + _WEYL_INCREMENT
            x = x + (float(weyl_state) * _WEYL_SCALE - 0.5) * (
                _BOOTSTRAP_NOISE_AMP * 2.0
            )

        if driven:
            x = drive_gain * (math.tanh(x + bias_amount) - math.tanh(bias_amount))

        # ZDF/TPT 2-pole SVF update — identical to the main SVF path.
        high = (x - (2.0 * damping + g) * band_state - low_state) / (
            1.0 + 2.0 * damping * g + g * g
        )
        band = g * high + band_state
        low = g * band + low_state
        band_state = band + g * high
        low_state = low + g * band

        if driven:
            band_state = _algebraic_sat(band_state)
            low_state = _algebraic_sat(low_state)

        if mode_int == _LP:
            output = low
        elif mode_int == _BP:
            output = band
        else:  # HP
            output = high

        prev_output = output
        filtered[i] = output

    return filtered


def _apply_sallen_key(
    signal: np.ndarray,
    *,
    cutoff_profile: np.ndarray,
    resonance_q: float,
    sample_rate: int,
    filter_mode: str,
    filter_drive: float,
    feedback_amount: float = 0.0,
    feedback_saturation: float = 0.0,
) -> np.ndarray:
    """Apply a Sallen-Key 2-pole filter (ZDF/TPT, CEM-3320-ish character).

    Uses the same trapezoidal integrator structure as the main SVF path,
    with a tighter Q-to-damping mapping and pre-filter asymmetric soft-clip
    under drive — this delivers the SK "bite" while keeping the ZDF/TPT
    structure and guaranteed stability.  See ``_apply_sallen_key_inner``
    docstring for why we use this form instead of a literal SK positive-
    feedback cross-sample solve.
    """
    q = max(0.5, float(resonance_q))
    # SK damping is tighter than SVF at the same Q → narrower, spikier peak.
    # At q=0.707 matches Butterworth; at q=50+ approaches self-oscillation
    # via the SVF's own ZDF resonance path (no K cross-feedback needed).
    sk_q = q * 1.4
    damping = 1.0 / sk_q

    mode_int = _MODE_STR_TO_INT.get(filter_mode, _LP)
    if mode_int == _NOTCH:
        mode_int = _LP

    sig = np.asarray(signal, dtype=np.float64)
    cutoff = np.asarray(cutoff_profile, dtype=np.float64)

    nyquist_limit = sample_rate * _NYQUIST_CLAMP_RATIO
    if cutoff.size > 0 and np.all(cutoff == cutoff[0]):
        clamped_fc = min(float(cutoff[0]), nyquist_limit)
        precomputed_g = math.tan(math.pi * clamped_fc / sample_rate)
    else:
        cutoff = np.minimum(cutoff, nyquist_limit)
        precomputed_g = -1.0

    bootstrap_seed = _bootstrap_seed(sig, "sallen_key")
    has_bootstrap = sig.shape[0] > 0
    return _apply_sallen_key_inner(
        sig,
        cutoff,
        damping,
        sample_rate,
        mode_int,
        max(0.0, float(filter_drive)),
        precomputed_g,
        nyquist_limit,
        bootstrap_seed,
        has_bootstrap,
        max(0.0, float(feedback_amount)),
        max(0.0, float(feedback_saturation)),
    )


# ---------------------------------------------------------------------------
# Cascade 4-pole (no global feedback + peaking resonance, Prophet/Juno feel)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _apply_cascade_inner(
    signal: np.ndarray,
    cutoff_profile: np.ndarray,
    resonance_boost: float,
    sample_rate: int,
    mode_int: int,
    filter_drive: float,
    precomputed_g: float,
    nyquist_limit: float,
    bootstrap_seed: np.uint64,
    has_bootstrap: bool,
    feedback_amount: float,
    feedback_saturation: float,
    morph: float,
) -> np.ndarray:
    """Cascade 4-pole: four independent ZDF 1-poles followed by a peaking
    resonance filter at cutoff.  No global tanh feedback — smoother
    Prophet-5 rev-2 / Juno character.
    """
    n = signal.shape[0]
    filtered = np.empty(n, dtype=np.float64)
    use_precomputed = precomputed_g >= 0.0

    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    bp_low = 0.0
    bp_band = 0.0

    driven = filter_drive > 0.0
    drive_gain = 1.0 + 1.5 * filter_drive if driven else 1.0

    has_ext_fb = feedback_amount > 0.0
    ext_fb_drive = 1.0 + 3.0 * feedback_saturation
    prev_ext_output = 0.0
    weyl_state = bootstrap_seed

    bp_damping = 1.0 / (0.5 + 0.8 * resonance_boost)

    for i in range(n):
        if use_precomputed:
            g = precomputed_g
        else:
            fc = min(cutoff_profile[i], nyquist_limit)
            g = math.tan(math.pi * fc / sample_rate)

        alpha = g / (1.0 + g)
        one_minus_alpha = 1.0 - alpha

        x = signal[i]
        if has_ext_fb:
            x = x + feedback_amount * math.tanh(ext_fb_drive * prev_ext_output)
        if has_bootstrap:
            weyl_state = weyl_state + _WEYL_INCREMENT
            x = x + (float(weyl_state) * _WEYL_SCALE - 0.5) * (
                _BOOTSTRAP_NOISE_AMP * 2.0
            )
        if driven:
            x = drive_gain * x

        y0 = alpha * x + one_minus_alpha * s0
        v0 = y0 - s0
        s0 = y0 + v0

        y1 = alpha * y0 + one_minus_alpha * s1
        v1 = y1 - s1
        s1 = y1 + v1

        y2 = alpha * y1 + one_minus_alpha * s2
        v2 = y2 - s2
        s2 = y2 + v2

        y3 = alpha * y2 + one_minus_alpha * s3
        v3 = y3 - s3
        s3 = y3 + v3

        if driven:
            y0 = _algebraic_sat(y0)
            y1 = _algebraic_sat(y1)
            y2 = _algebraic_sat(y2)
            y3 = _algebraic_sat(y3)
            s0 = _algebraic_sat(s0)
            s1 = _algebraic_sat(s1)
            s2 = _algebraic_sat(s2)
            s3 = _algebraic_sat(s3)

        bp_high = (y3 - (2.0 * bp_damping + g) * bp_band - bp_low) / (
            1.0 + 2.0 * bp_damping * g + g * g
        )
        bp_band_new = g * bp_high + bp_band
        bp_low_new = g * bp_band_new + bp_low
        bp_band = bp_band_new + g * bp_high
        bp_low = bp_low_new + g * bp_band_new
        bp_out = bp_band_new

        if morph <= 0.0:
            if mode_int == _LP:
                base = y3
            elif mode_int == _BP:
                base = y1 - y3
            else:
                base = x - y3
        else:
            m = min(morph, 3.0)
            if mode_int == _LP:
                if m <= 1.0:
                    base = (1.0 - m) * y3 + m * y2
                elif m <= 2.0:
                    f = m - 1.0
                    base = (1.0 - f) * y2 + f * y1
                else:
                    f = m - 2.0
                    base = (1.0 - f) * y1 + f * y0
            elif mode_int == _BP:
                if m <= 1.0:
                    base = (1.0 - m) * (y1 - y3) + m * (y0 - y2)
                elif m <= 2.0:
                    f = m - 1.0
                    base = (1.0 - f) * (y0 - y2) + f * (y0 - y1)
                else:
                    f = m - 2.0
                    base = (1.0 - f) * (y0 - y1) + f * y0
            else:
                if m <= 1.0:
                    base = (1.0 - m) * (x - y3) + m * (x - y2)
                elif m <= 2.0:
                    f = m - 1.0
                    base = (1.0 - f) * (x - y2) + f * (x - y1)
                else:
                    f = m - 2.0
                    base = (1.0 - f) * (x - y1) + f * (x - y0)

        output = base + resonance_boost * bp_out

        prev_ext_output = output
        filtered[i] = output

    return filtered


def _apply_cascade(
    signal: np.ndarray,
    *,
    cutoff_profile: np.ndarray,
    resonance_q: float,
    sample_rate: int,
    filter_mode: str,
    filter_drive: float,
    feedback_amount: float = 0.0,
    feedback_saturation: float = 0.0,
    filter_morph: float = 0.0,
) -> np.ndarray:
    """Apply a Cascade 4-pole filter (4 independent 1-poles + peaking
    resonance)."""
    q = max(0.5, float(resonance_q))
    resonance_boost = max(0.0, min(1.8, 2.0 * (1.0 - 1.0 / (2.0 * q))))

    mode_int = _MODE_STR_TO_INT.get(filter_mode, _LP)
    if mode_int == _NOTCH:
        mode_int = _LP

    sig = np.asarray(signal, dtype=np.float64)
    cutoff = np.asarray(cutoff_profile, dtype=np.float64)

    nyquist_limit = sample_rate * _NYQUIST_CLAMP_RATIO
    if cutoff.size > 0 and np.all(cutoff == cutoff[0]):
        clamped_fc = min(float(cutoff[0]), nyquist_limit)
        precomputed_g = math.tan(math.pi * clamped_fc / sample_rate)
    else:
        cutoff = np.minimum(cutoff, nyquist_limit)
        precomputed_g = -1.0

    bootstrap_seed = _bootstrap_seed(sig, "cascade")
    has_bootstrap = sig.shape[0] > 0
    return _apply_cascade_inner(
        sig,
        cutoff,
        resonance_boost,
        sample_rate,
        mode_int,
        max(0.0, float(filter_drive)),
        precomputed_g,
        nyquist_limit,
        bootstrap_seed,
        has_bootstrap,
        max(0.0, float(feedback_amount)),
        max(0.0, float(feedback_saturation)),
        max(0.0, min(3.0, float(filter_morph))),
    )


# ---------------------------------------------------------------------------
# Unified filter dispatch
# ---------------------------------------------------------------------------


def apply_filter(
    signal: np.ndarray,
    *,
    cutoff_profile: np.ndarray,
    resonance_q: float = 0.707,
    sample_rate: int,
    filter_mode: str = "lowpass",
    filter_drive: float = 0.0,
    filter_even_harmonics: float = 0.0,
    filter_topology: str = "svf",
    bass_compensation: float = 0.0,
    filter_morph: float = 0.0,
    hpf_cutoff_hz: float = 0.0,
    hpf_resonance_q: float = 0.707,
    feedback_amount: float = 0.0,
    feedback_saturation: float = 0.3,
    filter_solver: str = "adaa",
    max_newton_iters: int = _DEFAULT_NEWTON_MAX_ITERS,
    newton_tolerance: float = _DEFAULT_NEWTON_TOLERANCE,
) -> np.ndarray:
    """Unified filter entry point that dispatches to SVF or ladder topology.

    When ``hpf_cutoff_hz > 0``, a serial 2-pole ZDF SVF highpass is applied
    before the main filter stage.

    When ``feedback_amount > 0``, the post-filter output feeds back to the
    pre-filter input through a saturating tanh stage (Minimoog-style mixer
    feedback).  ``feedback_saturation`` controls the drive on the feedback
    path tanh (default 0.3).

    ``filter_solver`` selects the ladder solver: ``"adaa"`` (default, the
    existing Huovilainen-style one-step-ADAA approach) or ``"newton"`` (an
    implicit per-sample Newton solve of the delay-free feedback loop, closer
    to the Diva / u-he ZDF-NL approach).  SVF topology ignores this knob.
    """
    if filter_topology not in _SUPPORTED_FILTER_TOPOLOGIES:
        raise ValueError(f"Unknown filter_topology: {filter_topology!r}")
    if filter_solver not in _SUPPORTED_LADDER_SOLVERS:
        raise ValueError(f"Unknown filter_solver: {filter_solver!r}")

    sig = np.asarray(signal, dtype=np.float64)

    if hpf_cutoff_hz > 0.0:
        nyquist_limit = sample_rate * _NYQUIST_CLAMP_RATIO
        hpf_cutoff = min(float(hpf_cutoff_hz), nyquist_limit)
        hpf_g = math.tan(math.pi * hpf_cutoff / sample_rate)
        hpf_cutoff_profile = np.full(sig.shape[0], hpf_cutoff, dtype=np.float64)
        hpf_damping = 1.0 / max(0.5, float(hpf_resonance_q))
        # Serial HPF has no feedback path — has_bootstrap=False disables
        # per-sample noise injection in the kernel.
        sig = _apply_linear_zdf_svf(
            sig,
            hpf_cutoff_profile,
            hpf_damping,
            sample_rate,
            _HP,
            hpf_g,
            nyquist_limit,
            np.uint64(0),
            False,
        )

    morph = max(0.0, float(filter_morph))

    if filter_topology == "svf":
        if morph == 0.0:
            return apply_zdf_svf(
                sig,
                cutoff_profile=cutoff_profile,
                resonance_q=resonance_q,
                sample_rate=sample_rate,
                filter_mode=filter_mode,
                filter_drive=filter_drive,
                filter_even_harmonics=filter_even_harmonics,
                feedback_amount=feedback_amount,
                feedback_saturation=feedback_saturation,
            )
        return _apply_svf_with_morph(
            sig,
            cutoff_profile=cutoff_profile,
            resonance_q=resonance_q,
            sample_rate=sample_rate,
            filter_mode=filter_mode,
            filter_drive=filter_drive,
            filter_even_harmonics=filter_even_harmonics,
            feedback_amount=feedback_amount,
            feedback_saturation=feedback_saturation,
            morph=morph,
        )

    if filter_topology == "sallen_key":
        return _apply_sallen_key(
            sig,
            cutoff_profile=cutoff_profile,
            resonance_q=resonance_q,
            sample_rate=sample_rate,
            filter_mode=filter_mode,
            filter_drive=filter_drive,
            feedback_amount=feedback_amount,
            feedback_saturation=feedback_saturation,
        )

    if filter_topology == "cascade":
        return _apply_cascade(
            sig,
            cutoff_profile=cutoff_profile,
            resonance_q=resonance_q,
            sample_rate=sample_rate,
            filter_mode=filter_mode,
            filter_drive=filter_drive,
            feedback_amount=feedback_amount,
            feedback_saturation=feedback_saturation,
            filter_morph=morph,
        )

    return _apply_ladder_filter(
        sig,
        cutoff_profile=cutoff_profile,
        resonance_q=resonance_q,
        sample_rate=sample_rate,
        filter_mode=filter_mode,
        filter_drive=filter_drive,
        bass_compensation=bass_compensation,
        filter_even_harmonics=filter_even_harmonics,
        feedback_amount=feedback_amount,
        feedback_saturation=feedback_saturation,
        filter_morph=morph,
        solver=filter_solver,
        max_newton_iters=max_newton_iters,
        newton_tolerance=newton_tolerance,
    )


@numba.njit(cache=True)
def _apply_comb_inner(
    signal: np.ndarray,
    delay_samples: np.ndarray,
    feedback: float,
    damping: float,
    bootstrap_seed: np.uint64,
    has_bootstrap: bool,
    max_delay: int,
) -> np.ndarray:
    """Resonant feedback comb with 1-pole damping in the feedback path.

    Linearly interpolates the delay read so non-integer per-sample delays
    produce smooth pitch. Feedback path soft-clipped via algebraicSat for
    stability at self-oscillation.
    """
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)
    buf_size = max_delay + 4
    buffer = np.zeros(buf_size, dtype=np.float64)
    lp_state = 0.0
    weyl_state = bootstrap_seed
    damp_coef = 1.0 - damping

    for i in range(n):
        raw_delay = delay_samples[i]
        if raw_delay < 1.0:
            raw_delay = 1.0
        if raw_delay > max_delay:
            raw_delay = float(max_delay)
        read_pos = float(i) - raw_delay
        while read_pos < 0.0:
            read_pos += buf_size
        read_idx = int(read_pos)
        frac = read_pos - read_idx
        a = buffer[read_idx % buf_size]
        b = buffer[(read_idx + 1) % buf_size]
        delayed = a + (b - a) * frac

        lp_state = damp_coef * delayed + damping * lp_state
        fb = _algebraic_sat(feedback * lp_state)

        x_in = signal[i]
        if has_bootstrap:
            weyl_state = weyl_state + _WEYL_INCREMENT
            x_in += (float(weyl_state) * _WEYL_SCALE - 0.5) * (
                _BOOTSTRAP_NOISE_AMP * 2.0
            )
        y = x_in + fb

        buffer[i % buf_size] = y
        out[i] = y

    return out


def apply_comb(
    signal: np.ndarray,
    *,
    delay_samples_profile: np.ndarray,
    feedback: float,
    damping: float,
    mix: float,
    sample_rate: int,
) -> np.ndarray:
    """Resonant comb filter with per-sample delay and damped feedback.

    Parameters
    ----------
    signal:
        Input signal (1-D float array).
    delay_samples_profile:
        Per-sample delay length in samples. Values < 1.0 are clamped to 1.0;
        values > ``sample_rate`` are clamped to one second. A frequency-tracked
        delay (``sample_rate / freq``) yields karplus-strong-ish resonance.
    feedback:
        Feedback gain in [0, 0.99]. Above ~0.95 with low damping produces
        sustained self-oscillation; soft-clipped internally for stability.
    damping:
        Feedback-path 1-pole lowpass amount in [0, 1]. 0 = bright resonance,
        higher values progressively darken and shorten the decay. Values are
        clamped internally to a max of 0.999 so the feedback path always
        receives a nonzero fraction of the delayed signal — damping=1.0
        approaches fully muted feedback but preserves numerical stability
        (a true 1.0 would freeze the 1-pole state and silence the wet path).
    mix:
        Dry/wet balance in [0, 1]. 0 returns input unchanged, 1 returns pure
        wet.
    sample_rate:
        Samples per second. Sets the hard cap on delay length.
    """
    if signal.ndim != 1:
        raise ValueError("apply_comb expects a 1-D signal")
    n = signal.shape[0]
    if n == 0:
        return signal
    if not (0.0 <= feedback <= 0.99):
        raise ValueError("feedback must be in [0, 0.99]")
    if not (0.0 <= damping <= 1.0):
        raise ValueError("damping must be in [0, 1]")
    if not (0.0 <= mix <= 1.0):
        raise ValueError("mix must be in [0, 1]")
    if delay_samples_profile.shape[0] != n:
        raise ValueError("delay_samples_profile must match signal length")

    profile = np.ascontiguousarray(delay_samples_profile, dtype=np.float64)
    sig = np.ascontiguousarray(signal, dtype=np.float64)
    if not np.all(np.isfinite(profile)):
        raise ValueError(
            "apply_comb: delay_samples_profile contains non-finite values (NaN/Inf)"
        )
    max_delay = int(np.clip(float(np.max(profile)) + 4.0, 4.0, float(sample_rate)))

    # Clamp damping below 1.0 so the feedback-path 1-pole always receives a
    # nonzero fraction of the delayed signal. At exactly 1.0 the state would
    # freeze and the wet signal would degenerate to silence.
    damping_clamped = min(damping, 0.999)

    bootstrap_seed = _bootstrap_seed(sig, tag="apply_comb")
    has_bootstrap = n > 0
    wet = _apply_comb_inner(
        sig,
        profile,
        feedback,
        damping_clamped,
        bootstrap_seed,
        has_bootstrap,
        max_delay,
    )

    if mix >= 1.0:
        return wet
    return (1.0 - mix) * sig + mix * wet
