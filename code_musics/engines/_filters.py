"""Shared DSP filter primitives used by multiple synth engines."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numba
import numpy as np

_SUPPORTED_FILTER_MODES = {"lowpass", "bandpass", "highpass", "notch"}
_SUPPORTED_FILTER_TOPOLOGIES = {
    "svf",
    "ladder",
    "sallen_key",
    "cascade",
    "sem",
    "jupiter",
    "k35",
    "diode",
}
_SUPPORTED_LADDER_SOLVERS = {"adaa", "newton"}

# Newton iteration defaults used when the engine does not supply quality-driven
# overrides.  Four iterations with 1e-9 tolerance closes the delay-free
# feedback loop far below audio SNR.  A looser tolerance like 1e-5 prematurely
# short-circuits Newton when the solver is seeded from silence (bootstrap noise
# at 1e-6 is already below the tolerance), which would kill self-oscillation.
_DEFAULT_NEWTON_MAX_ITERS: int = 4
_DEFAULT_NEWTON_TOLERANCE: float = 1e-9


@dataclass(frozen=True)
class FilterParams:
    """Bundles every kwarg accepted by :func:`apply_filter`."""

    resonance_q: float = 0.707
    filter_mode: str = "lowpass"
    filter_drive: float = 0.0
    filter_even_harmonics: float = 0.0
    filter_topology: str = "svf"
    bass_compensation: float = 0.0
    filter_morph: float = 0.0
    hpf_cutoff_hz: float = 0.0
    hpf_resonance_q: float = 0.707
    feedback_amount: float = 0.0
    feedback_saturation: float = 0.3
    filter_solver: str = "newton"
    max_newton_iters: int = _DEFAULT_NEWTON_MAX_ITERS
    newton_tolerance: float = _DEFAULT_NEWTON_TOLERANCE
    # Reserved for future k35 topology asymmetric-feedback tuning.
    k35_feedback_asymmetry: float = 0.0


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
def _solve_ext_feedback_newton(
    y_warm: float,
    affine_const: float,
    fb_scale: float,
    fb_drive: float,
    max_iters: int,
    tol: float,
) -> float:
    """Solve ``y = affine_const + fb_scale * tanh(fb_drive * y)`` for ``y``.

    This is the scalar Newton used to close the external filter feedback
    loop without a one-sample delay.  ``affine_const`` absorbs the linear
    (non-feedback) part of the filter body's instantaneous input-to-output
    map (state contribution + input-only gain times the raw input), and
    ``fb_scale = A * fb_amt`` where ``A`` is the filter body's static input
    gain.  ``fb_drive`` is the inner tanh argument scaling (= 1 + 3*sat).

    Residual:  F(y)  = y - affine_const - fb_scale * tanh(fb_drive * y)
    Jacobian:  F'(y) = 1 - fb_scale * fb_drive * (1 - tanh²(fb_drive * y))

    F is monotonically increasing when ``fb_scale * fb_drive < 1`` — the
    well-posed regime below the instantaneous self-oscillation threshold.
    Newton has a unique root and converges quadratically from any warm
    start.  Above the threshold the inflection point in the tanh sigmoid
    still guarantees the stable branch near the warm start wins, so we
    clamp the Jacobian away from zero to keep the step well-defined.
    """
    y = y_warm
    for _ in range(max_iters):
        ty = math.tanh(fb_drive * y)
        residual = y - affine_const - fb_scale * ty
        if math.fabs(residual) < tol:
            break
        sech2 = 1.0 - ty * ty
        jac = 1.0 - fb_scale * fb_drive * sech2
        if jac < 1e-6 and jac > -1e-6:
            jac = 1e-6 if jac >= 0.0 else -1e-6
        y = y - residual / jac
    return y


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
def _diode_shape(x: float, asymmetry: float) -> float:
    """Asymmetric Shockley-like diode shaper for MS-20 and TB-303 character.

    Returns ``sign(x) * log1p(|x|) / scale`` but with different saturation
    scales for positive and negative swings governed by ``asymmetry``.
    At ``asymmetry=0`` this degenerates to a symmetric log-soft-clip
    (close to tanh at small |x|, slightly gentler at large |x|).  At
    ``asymmetry=1`` the positive swing saturates noticeably earlier than
    the negative — this is what produces the MS-20 "snarl" in the K35
    feedback path and the 303 "bark" in the diode ladder stages.

    Stays in [-1, 1]-ish for reasonable inputs and is smooth (no
    discontinuity at x=0).  Intentionally cheap — no transcendentals
    beyond one log1p per call — so it drops into per-stage kernels
    without measurable cost.
    """
    a = max(0.0, min(1.0, asymmetry))
    # Positive Is shrinks (earlier saturation) as asymmetry rises; negative
    # Is grows (later saturation).  Ratio at a=1 is 1.5/0.65 ≈ 2.3x which
    # matches the audible MS-20 asymmetry without going cartoonish.
    is_pos = 1.0 - 0.35 * a
    is_neg = 1.0 + 0.5 * a
    ax = math.fabs(x)
    if x >= 0.0:
        return math.log1p(ax / is_pos) / math.log1p(1.0 / is_pos)
    return -math.log1p(ax / is_neg) / math.log1p(1.0 / is_neg)


# Width of the smoothstep region used to blend clean-vs-driven code paths on
# drive threshold crossings.  The historical boolean ``driven = drive > 0``
# produced an audible step when drive was automated across zero because the
# integrator-state saturation (``_algebraic_sat``) and the sallen_key pre-
# filter tanh shape snapped from off to fully-on.  With this epsilon, the
# saturation engages smoothly across ``drive ∈ [0, DRIVE_BLEND_EPSILON]`` and
# matches current "full driven" behavior for ``drive >= DRIVE_BLEND_EPSILON``.
# Endpoints (drive=0 and drive large) are preserved exactly.
_DRIVE_BLEND_EPSILON: float = 0.05


def _drive_sat_blend(filter_drive: float) -> float:
    """Smoothstep blend coefficient for driven filter kernels.

    Endpoints preserved exactly: 0 at ``filter_drive <= 0``, 1 at
    ``filter_drive >= _DRIVE_BLEND_EPSILON``.  Body mirrors
    :func:`code_musics.engines._dsp_utils.smoothstep_blend`; duplicated
    here to avoid a circular import (_dsp_utils already imports
    :class:`FilterParams` from this module).
    """
    if filter_drive <= 0.0:
        return 0.0
    if filter_drive >= _DRIVE_BLEND_EPSILON:
        return 1.0
    t = filter_drive / _DRIVE_BLEND_EPSILON
    return t * t * (3.0 - 2.0 * t)


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
    use_newton_feedback: bool = False,
    max_newton_iters: int = _DEFAULT_NEWTON_MAX_ITERS,
    newton_tolerance: float = _DEFAULT_NEWTON_TOLERANCE,
) -> np.ndarray:
    """Apply the fully linear ZDF/TPT state-variable filter (numba-compiled)."""
    n = signal.shape[0]
    filtered = np.empty(n, dtype=np.float64)
    low_state = 0.0
    band_state = 0.0
    use_precomputed = precomputed_g >= 0.0

    has_ext_fb = feedback_amount > 0.0
    newton_feedback_active = has_ext_fb and use_newton_feedback
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

        # Affine coefficients of the ZDF SVF body: output = A_out * x + B_out
        # for whichever mode is selected.  We collapse the instantaneous map
        # once per sample so Newton can solve the implicit feedback equation
        # y = A_out * (raw + fb_amt * tanh(ext_fb_drive * y)) + B_out
        # = affine_const + fb_scale * tanh(ext_fb_drive * y).
        denom = 1.0 + 2.0 * damping * g + g * g
        state_term = (2.0 * damping + g) * band_state + low_state
        high_input_gain = 1.0 / denom
        high_state_offset = -state_term / denom
        band_input_gain = g * high_input_gain
        band_state_offset = g * high_state_offset + band_state
        low_input_gain = g * band_input_gain
        low_state_offset = g * band_state_offset + low_state

        if use_morph:
            notch_input_gain = low_input_gain + high_input_gain
            notch_state_offset = low_state_offset + high_state_offset
            A_out = (
                lp_w * low_input_gain
                + bp_w * band_input_gain
                + hp_w * high_input_gain
                + notch_w * notch_input_gain
            )
            B_out = (
                lp_w * low_state_offset
                + bp_w * band_state_offset
                + hp_w * high_state_offset
                + notch_w * notch_state_offset
            )
        elif mode_int == _LP:
            A_out = low_input_gain
            B_out = low_state_offset
        elif mode_int == _BP:
            A_out = band_input_gain
            B_out = band_state_offset
        elif mode_int == _HP:
            A_out = high_input_gain
            B_out = high_state_offset
        else:
            A_out = low_input_gain + high_input_gain
            B_out = low_state_offset + high_state_offset

        sample = signal[i]
        if has_ext_fb:
            if newton_feedback_active:
                # Collapse the filter body to output = A_out*x + B_out and
                # Newton-solve the closed loop y = A_out*(raw + fb_amt*tanh(g_fb*y))
                # + B_out, then reconstruct the loop input so the integrator
                # state update stays consistent with the solved output.
                raw = sample
                if has_bootstrap:
                    weyl_state = weyl_state + _WEYL_INCREMENT
                    raw = raw + (float(weyl_state) * _WEYL_SCALE - 0.5) * (
                        _BOOTSTRAP_NOISE_AMP * 2.0
                    )
                affine_const = A_out * raw + B_out
                fb_scale = A_out * feedback_amount
                output_solved = _solve_ext_feedback_newton(
                    prev_output,
                    affine_const,
                    fb_scale,
                    ext_fb_drive,
                    max_newton_iters,
                    newton_tolerance,
                )
                sample = raw + feedback_amount * math.tanh(ext_fb_drive * output_solved)
            else:
                sample = sample + feedback_amount * math.tanh(
                    ext_fb_drive * prev_output
                )
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
    solver: str = "newton",
    max_newton_iters: int = _DEFAULT_NEWTON_MAX_ITERS,
    newton_tolerance: float = _DEFAULT_NEWTON_TOLERANCE,
) -> np.ndarray:
    """Apply a per-sample ZDF/TPT state-variable filter.

    When ``filter_drive=0`` the filter runs a fully linear path — no soft-clipping
    anywhere in the loop.  Nonlinear processing only activates for ``filter_drive>0``.

    The driven path uses topology-aware saturation: a single ADAA-antialiased
    tanh at the feedback summation point, algebraicSat on integrator states,
    and bidirectional drive/resonance interaction.

    When ``feedback_amount > 0``, post-filter output feeds back to the pre-filter
    input through a saturating tanh stage (Minimoog-style mixer feedback).

    ``solver="newton"`` closes the external feedback loop implicitly on the
    clean (``filter_drive=0``) linear path.  The driven branch stays on
    unit-delay feedback — its pre-filter tanh shape breaks the affine-collapse
    derivation, and drive+ext-FB is a rare combination in practice.
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
    use_newton_fb = solver == "newton" and filter_drive <= 0.0
    newton_iters = max(1, int(max_newton_iters))
    newton_tol = max(1e-12, float(newton_tolerance))

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
        use_newton_fb,
        newton_iters,
        newton_tol,
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
    solver: str = "newton",
    max_newton_iters: int = _DEFAULT_NEWTON_MAX_ITERS,
    newton_tolerance: float = _DEFAULT_NEWTON_TOLERANCE,
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
    use_newton_fb = solver == "newton" and filter_drive <= 0.0
    newton_iters = max(1, int(max_newton_iters))
    newton_tol = max(1e-12, float(newton_tolerance))

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
        use_newton_fb,
        newton_iters,
        newton_tol,
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
    sat_blend: float,
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
    """Huovilainen improved 4-pole ladder with per-stage ZDF/TPT integrators.

    ``sat_blend`` is the smoothstep coefficient (0..1) that controls how much
    of the nonlinear ``_algebraic_sat`` shaping of the per-stage outputs and
    integrator states is mixed in.  ``sat_blend=0`` gives the exact linear
    path; ``sat_blend=1`` gives the fully driven behavior.  Callers compute
    this via ``_drive_sat_blend`` so small drives fade in smoothly instead of
    stepping from zero saturation to full saturation at ``drive=0``.
    """
    n = signal.shape[0]
    filtered = np.empty(n, dtype=np.float64)
    use_precomputed = precomputed_g >= 0.0

    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0

    # ``drive_gain`` is continuous in filter_drive — at drive=0 it evaluates
    # to 1.0, so we can unconditionally apply it without a clean-vs-driven
    # branch.  Only the ``_algebraic_sat`` saturation needs the blend.
    drive_gain = 1.0 + 2.0 * filter_drive
    apply_saturation = sat_blend > 0.0
    sat_mix_clean = 1.0 - sat_blend
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

        inp = drive_gain * inp

        if bias_amount > 0.0:
            inp = inp + bias_amount * math.fabs(signal[i])

        v0 = g * (inp - s0) / (1.0 + g)
        y0 = v0 + s0
        s0 = y0 + v0
        if apply_saturation:
            y0 = sat_mix_clean * y0 + sat_blend * _algebraic_sat(y0)
            s0 = sat_mix_clean * s0 + sat_blend * _algebraic_sat(s0)

        v1 = g * (y0 - s1) / (1.0 + g)
        y1 = v1 + s1
        s1 = y1 + v1
        if apply_saturation:
            y1 = sat_mix_clean * y1 + sat_blend * _algebraic_sat(y1)
            s1 = sat_mix_clean * s1 + sat_blend * _algebraic_sat(s1)

        v2 = g * (y1 - s2) / (1.0 + g)
        y2 = v2 + s2
        s2 = y2 + v2
        if apply_saturation:
            y2 = sat_mix_clean * y2 + sat_blend * _algebraic_sat(y2)
            s2 = sat_mix_clean * s2 + sat_blend * _algebraic_sat(s2)

        v3 = g * (y2 - s3) / (1.0 + g)
        y3 = v3 + s3
        s3 = y3 + v3
        if apply_saturation:
            y3 = sat_mix_clean * y3 + sat_blend * _algebraic_sat(y3)
            s3 = sat_mix_clean * s3 + sat_blend * _algebraic_sat(s3)

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
    sat_blend: float,
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
    close_ext_feedback: bool,
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

    # ``drive_gain`` is continuous in filter_drive — at drive=0 it evaluates
    # to 1.0, so we can unconditionally apply it without a clean-vs-driven
    # branch.  Only the ``_algebraic_sat`` saturation needs the blend.
    drive_gain = 1.0 + 2.0 * filter_drive
    apply_saturation = sat_blend > 0.0
    sat_mix_clean = 1.0 - sat_blend
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
        if has_bootstrap:
            weyl_state = weyl_state + _WEYL_INCREMENT
            x_pre = x_pre + (float(weyl_state) * _WEYL_SCALE - 0.5) * (
                _BOOTSTRAP_NOISE_AMP * 2.0
            )
        if bias_amount > 0.0:
            x_pre = x_pre + bias_amount * math.fabs(signal[i])

        # For LP with no morph, no bass_comp, no even_harmonics:  output = y3.
        # The external feedback closes on output, so the closed implicit
        # equation reduces to a single scalar Newton on y3:
        #   y3 = A*(drive_gain*(x_pre + fb_amt*tanh(g_fb*y3)) - tanh(k*y3)) + beta
        # Rearranged:
        #   F(y3) = y3 + A*tanh(k*y3) - A*D*tanh(g_fb*y3) - B_raw = 0
        # where D = drive_gain*fb_amt and B_raw = A*drive_gain*x_pre + beta.
        # For other modes / morph / bass_comp, the output is still *linear* in
        # y3 and u, but the proportionality differs; we fall back to the
        # unit-delay feedback for those paths since they are far less common
        # at high feedback (LP with no morph is the bass / lead use case).
        outer_newton = (
            close_ext_feedback
            and has_ext_fb
            and morph <= 0.0
            and mode_int == _LP
            and bass_compensation <= 0.0
            and bias_amount <= 0.0
        )
        if has_ext_fb and not outer_newton:
            x_pre = x_pre + feedback_amount * math.tanh(ext_fb_drive * prev_ext_output)

        u_no_fb = drive_gain * x_pre

        # beta = alpha^3*(1-alpha)*s0 + alpha^2*(1-alpha)*s1
        #      + alpha*(1-alpha)*s2 + (1-alpha)*s3
        beta = one_minus_alpha * (((alpha * s0 + s1) * alpha + s2) * alpha + s3)
        B = A * u_no_fb + beta

        y3 = y3_prev
        Ak = A * resonance_k
        if outer_newton:
            # Closed-loop Newton: F(y3) = y3 + A·tanh(k·y3)
            #                             - A·D·tanh(g_fb·y3) - B_raw
            # with D = drive_gain·fb_amt, B_raw = A·drive_gain·x_pre + beta.
            D_outer = drive_gain * feedback_amount
            AD_outer = A * D_outer
            for _ in range(max_iters):
                ky = resonance_k * y3
                gy = ext_fb_drive * y3
                th_k = math.tanh(ky)
                th_g = math.tanh(gy)
                sech2_k = 1.0 - th_k * th_k
                sech2_g = 1.0 - th_g * th_g
                F = y3 + A * th_k - AD_outer * th_g - B
                if math.fabs(F) < tolerance:
                    break
                Fp = 1.0 + Ak * sech2_k - AD_outer * ext_fb_drive * sech2_g
                # Guard against a vanishing Jacobian (well-posed inputs have
                # Fp >> 0; this only matters at the self-oscillation edge).
                if Fp < 1e-6 and Fp > -1e-6:
                    Fp = 1e-6 if Fp >= 0.0 else -1e-6
                y3 = y3 - F / Fp
            # Replay the loop input with the solved y3 so stage states advance
            # consistent with the implicit solution.
            u_no_fb = drive_gain * (
                x_pre + feedback_amount * math.tanh(ext_fb_drive * y3)
            )
        else:
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

        if apply_saturation:
            y0 = sat_mix_clean * y0 + sat_blend * _algebraic_sat(y0)
            y1 = sat_mix_clean * y1 + sat_blend * _algebraic_sat(y1)
            y2 = sat_mix_clean * y2 + sat_blend * _algebraic_sat(y2)
            s0 = sat_mix_clean * s0 + sat_blend * _algebraic_sat(s0)
            s1 = sat_mix_clean * s1 + sat_blend * _algebraic_sat(s1)
            s2 = sat_mix_clean * s2 + sat_blend * _algebraic_sat(s2)
            s3 = sat_mix_clean * s3 + sat_blend * _algebraic_sat(s3)

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

    clamped_drive = max(0.0, float(filter_drive))
    sat_blend = _drive_sat_blend(clamped_drive)

    if solver_name == "newton":
        iters = max(1, int(max_newton_iters))
        tol = max(1e-9, float(newton_tolerance))
        return _apply_ladder_filter_newton_inner(
            sig,
            cutoff,
            k_newton,
            sample_rate,
            mode_int,
            clamped_drive,
            sat_blend,
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
            True,
        )

    return _apply_ladder_filter_inner(
        sig,
        cutoff,
        k_adaa,
        sample_rate,
        mode_int,
        clamped_drive,
        sat_blend,
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
    sat_blend: float,
    precomputed_g: float,
    nyquist_limit: float,
    bootstrap_seed: np.uint64,
    has_bootstrap: bool,
    feedback_amount: float,
    feedback_saturation: float,
    use_newton_feedback: bool = False,
    max_newton_iters: int = _DEFAULT_NEWTON_MAX_ITERS,
    newton_tolerance: float = _DEFAULT_NEWTON_TOLERANCE,
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

    # ``drive_gain`` and ``bias_amount`` are both continuous in filter_drive
    # (evaluate to 1.0 and 0.0 at drive=0), so the pre-filter shape only
    # diverges from identity once we engage the ``tanh`` branch.  Crossfade
    # with the clean input via ``sat_blend`` so drive automation across zero
    # does not step between identity and ``tanh(x)``.
    apply_saturation = sat_blend > 0.0
    sat_mix_clean = 1.0 - sat_blend
    drive_gain = 1.0 + 1.5 * filter_drive
    # SK bite: positive input bias produces asymmetric waveshaping (even
    # harmonics).  Applied pre-filter so the color rides the signal envelope.
    bias_amount = 0.08 * filter_drive

    has_ext_fb = feedback_amount > 0.0
    # Newton external feedback only in the clean (non-driven) body; the
    # drive branch's pre-filter shape is nonlinear in x.
    newton_feedback_active = has_ext_fb and use_newton_feedback and not apply_saturation
    ext_fb_drive = 1.0 + 3.0 * feedback_saturation
    prev_output = 0.0
    weyl_state = bootstrap_seed

    for i in range(n):
        if use_precomputed:
            g = precomputed_g
        else:
            fc = min(cutoff_profile[i], nyquist_limit)
            g = math.tan(math.pi * fc / sample_rate)

        if newton_feedback_active:
            denom_svf = 1.0 + 2.0 * damping * g + g * g
            state_term = (2.0 * damping + g) * band_state + low_state
            high_in = 1.0 / denom_svf
            high_off = -state_term / denom_svf
            band_in = g * high_in
            band_off = g * high_off + band_state
            low_in = g * band_in
            low_off = g * band_off + low_state
            if mode_int == _LP:
                A_out = low_in
                B_out = low_off
            elif mode_int == _BP:
                A_out = band_in
                B_out = band_off
            else:
                A_out = high_in
                B_out = high_off
            raw = signal[i]
            if has_bootstrap:
                weyl_state = weyl_state + _WEYL_INCREMENT
                raw = raw + (float(weyl_state) * _WEYL_SCALE - 0.5) * (
                    _BOOTSTRAP_NOISE_AMP * 2.0
                )
            affine_const = A_out * raw + B_out
            fb_scale = A_out * feedback_amount
            y_solved = _solve_ext_feedback_newton(
                prev_output,
                affine_const,
                fb_scale,
                ext_fb_drive,
                max_newton_iters,
                newton_tolerance,
            )
            x = raw + feedback_amount * math.tanh(ext_fb_drive * y_solved)
        else:
            x = signal[i]
            if has_ext_fb:
                x = x + feedback_amount * math.tanh(ext_fb_drive * prev_output)
            if has_bootstrap:
                weyl_state = weyl_state + _WEYL_INCREMENT
                x = x + (float(weyl_state) * _WEYL_SCALE - 0.5) * (
                    _BOOTSTRAP_NOISE_AMP * 2.0
                )
            if apply_saturation:
                shaped = drive_gain * (
                    math.tanh(x + bias_amount) - math.tanh(bias_amount)
                )
                x = sat_mix_clean * x + sat_blend * shaped

        # ZDF/TPT 2-pole SVF update — identical to the main SVF path.
        high = (x - (2.0 * damping + g) * band_state - low_state) / (
            1.0 + 2.0 * damping * g + g * g
        )
        band = g * high + band_state
        low = g * band + low_state
        band_state = band + g * high
        low_state = low + g * band

        if apply_saturation:
            band_state = sat_mix_clean * band_state + sat_blend * _algebraic_sat(
                band_state
            )
            low_state = sat_mix_clean * low_state + sat_blend * _algebraic_sat(
                low_state
            )

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
    solver: str = "newton",
    max_newton_iters: int = _DEFAULT_NEWTON_MAX_ITERS,
    newton_tolerance: float = _DEFAULT_NEWTON_TOLERANCE,
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
    clamped_drive = max(0.0, float(filter_drive))
    sat_blend = _drive_sat_blend(clamped_drive)
    return _apply_sallen_key_inner(
        sig,
        cutoff,
        damping,
        sample_rate,
        mode_int,
        clamped_drive,
        sat_blend,
        precomputed_g,
        nyquist_limit,
        bootstrap_seed,
        has_bootstrap,
        max(0.0, float(feedback_amount)),
        max(0.0, float(feedback_saturation)),
        solver == "newton",
        max(1, int(max_newton_iters)),
        max(1e-12, float(newton_tolerance)),
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
    sat_blend: float,
    precomputed_g: float,
    nyquist_limit: float,
    bootstrap_seed: np.uint64,
    has_bootstrap: bool,
    feedback_amount: float,
    feedback_saturation: float,
    morph: float,
    use_newton_feedback: bool = False,
    max_newton_iters: int = _DEFAULT_NEWTON_MAX_ITERS,
    newton_tolerance: float = _DEFAULT_NEWTON_TOLERANCE,
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

    # ``drive_gain`` is continuous in filter_drive (1.0 at drive=0), so we can
    # apply it unconditionally.  Only the ``_algebraic_sat`` saturation needs
    # the blend to avoid a discontinuity at drive=0.
    drive_gain = 1.0 + 1.5 * filter_drive
    apply_saturation = sat_blend > 0.0
    sat_mix_clean = 1.0 - sat_blend

    has_ext_fb = feedback_amount > 0.0
    newton_feedback_active = has_ext_fb and use_newton_feedback
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
            if newton_feedback_active:
                # Collapse the linear body to output = A_out*x_in + B_out.
                # Four 1-poles in series give y3 = alpha^4 * u + state_sum,
                # where u = drive_gain * x_in.  y3 then feeds a linear 2-pole
                # peaking SVF; the BP tap's sensitivity to y3 is
                # d(bp_band_new)/d(y3) = g / (1 + 2*bp_damping*g + g²).
                # At drive > 0 the per-stage state saturation is a mild
                # deformation — we use the linearised map to drive Newton
                # and reconstruct the true loop input from the solved y so
                # the nonlinear body update stays consistent.
                raw = x
                if has_bootstrap:
                    weyl_state = weyl_state + _WEYL_INCREMENT
                    raw = raw + (float(weyl_state) * _WEYL_SCALE - 0.5) * (
                        _BOOTSTRAP_NOISE_AMP * 2.0
                    )
                alpha2 = alpha * alpha
                alpha3 = alpha2 * alpha
                alpha4 = alpha2 * alpha2
                y3_state = (
                    alpha3 * one_minus_alpha * s0
                    + alpha2 * one_minus_alpha * s1
                    + alpha * one_minus_alpha * s2
                    + one_minus_alpha * s3
                )
                bp_denom = 1.0 + 2.0 * bp_damping * g + g * g
                bp_y3_to_band = g / bp_denom
                bp_state_term = (2.0 * bp_damping + g) * bp_band + bp_low
                bp_out_from_state = -g * bp_state_term / bp_denom + bp_band
                # Build base-path coefficients by mode/morph against u.
                # (y0, y1, y2, y3, x) are all affine in u with gains
                # (alpha, alpha², alpha³, alpha⁴, 1) respectively.
                if morph <= 0.0:
                    if mode_int == _LP:
                        base_u_gain = alpha4
                        base_state = y3_state
                    elif mode_int == _BP:
                        base_u_gain = alpha2 - alpha4
                        base_state = (
                            alpha * one_minus_alpha * s0 + one_minus_alpha * s1
                        ) - y3_state
                    else:
                        base_u_gain = 1.0 - alpha4
                        base_state = -y3_state
                else:
                    m_cap = min(morph, 3.0)
                    if mode_int == _LP:
                        if m_cap <= 1.0:
                            base_u_gain = (1.0 - m_cap) * alpha4 + m_cap * alpha3
                            base_state = (1.0 - m_cap) * y3_state + m_cap * (
                                alpha2 * one_minus_alpha * s0
                                + alpha * one_minus_alpha * s1
                                + one_minus_alpha * s2
                            )
                        elif m_cap <= 2.0:
                            f_w = m_cap - 1.0
                            base_u_gain = (1.0 - f_w) * alpha3 + f_w * alpha2
                            base_state = (1.0 - f_w) * (
                                alpha2 * one_minus_alpha * s0
                                + alpha * one_minus_alpha * s1
                                + one_minus_alpha * s2
                            ) + f_w * (
                                alpha * one_minus_alpha * s0 + one_minus_alpha * s1
                            )
                        else:
                            f_w = m_cap - 2.0
                            base_u_gain = (1.0 - f_w) * alpha2 + f_w * alpha
                            base_state = (1.0 - f_w) * (
                                alpha * one_minus_alpha * s0 + one_minus_alpha * s1
                            ) + f_w * one_minus_alpha * s0
                    elif mode_int == _BP:
                        if m_cap <= 1.0:
                            base_u_gain = (1.0 - m_cap) * (alpha2 - alpha4) + m_cap * (
                                alpha - alpha3
                            )
                            bp4_state = (
                                alpha * one_minus_alpha * s0 + one_minus_alpha * s1
                            ) - y3_state
                            bp3_state = one_minus_alpha * s0 - (
                                alpha2 * one_minus_alpha * s0
                                + alpha * one_minus_alpha * s1
                                + one_minus_alpha * s2
                            )
                            base_state = (1.0 - m_cap) * bp4_state + m_cap * bp3_state
                        elif m_cap <= 2.0:
                            f_w = m_cap - 1.0
                            base_u_gain = (1.0 - f_w) * (alpha - alpha3) + f_w * (
                                alpha - alpha2
                            )
                            bp3_state = one_minus_alpha * s0 - (
                                alpha2 * one_minus_alpha * s0
                                + alpha * one_minus_alpha * s1
                                + one_minus_alpha * s2
                            )
                            bp2_state = one_minus_alpha * s0 - (
                                alpha * one_minus_alpha * s0 + one_minus_alpha * s1
                            )
                            base_state = (1.0 - f_w) * bp3_state + f_w * bp2_state
                        else:
                            f_w = m_cap - 2.0
                            base_u_gain = (1.0 - f_w) * (alpha - alpha2) + f_w * alpha
                            bp2_state = one_minus_alpha * s0 - (
                                alpha * one_minus_alpha * s0 + one_minus_alpha * s1
                            )
                            base_state = (
                                1.0 - f_w
                            ) * bp2_state + f_w * one_minus_alpha * s0
                    else:
                        if m_cap <= 1.0:
                            base_u_gain = (1.0 - m_cap) * (1.0 - alpha4) + m_cap * (
                                1.0 - alpha3
                            )
                            hp4_state = -y3_state
                            hp3_state = -(
                                alpha2 * one_minus_alpha * s0
                                + alpha * one_minus_alpha * s1
                                + one_minus_alpha * s2
                            )
                            base_state = (1.0 - m_cap) * hp4_state + m_cap * hp3_state
                        elif m_cap <= 2.0:
                            f_w = m_cap - 1.0
                            base_u_gain = (1.0 - f_w) * (1.0 - alpha3) + f_w * (
                                1.0 - alpha2
                            )
                            hp3_state = -(
                                alpha2 * one_minus_alpha * s0
                                + alpha * one_minus_alpha * s1
                                + one_minus_alpha * s2
                            )
                            hp2_state = -(
                                alpha * one_minus_alpha * s0 + one_minus_alpha * s1
                            )
                            base_state = (1.0 - f_w) * hp3_state + f_w * hp2_state
                        else:
                            f_w = m_cap - 2.0
                            base_u_gain = (1.0 - f_w) * (1.0 - alpha2) + f_w * (
                                1.0 - alpha
                            )
                            hp2_state = -(
                                alpha * one_minus_alpha * s0 + one_minus_alpha * s1
                            )
                            hp1_state = -one_minus_alpha * s0
                            base_state = (1.0 - f_w) * hp2_state + f_w * hp1_state
                # output = base_u_gain * u + base_state
                #        + resonance_boost * (bp_y3_to_band * y3 + bp_out_from_state)
                # With u = drive_gain * x_in and y3 = alpha^4 * u + y3_state:
                out_u_gain = base_u_gain + resonance_boost * bp_y3_to_band * alpha4
                out_state = base_state + resonance_boost * (
                    bp_y3_to_band * y3_state + bp_out_from_state
                )
                A_out = drive_gain * out_u_gain
                B_out = out_state
                affine_const = A_out * raw + B_out
                fb_scale = A_out * feedback_amount
                output_solved = _solve_ext_feedback_newton(
                    prev_ext_output,
                    affine_const,
                    fb_scale,
                    ext_fb_drive,
                    max_newton_iters,
                    newton_tolerance,
                )
                x = raw + feedback_amount * math.tanh(ext_fb_drive * output_solved)
            else:
                x = x + feedback_amount * math.tanh(ext_fb_drive * prev_ext_output)
                if has_bootstrap:
                    weyl_state = weyl_state + _WEYL_INCREMENT
                    x = x + (float(weyl_state) * _WEYL_SCALE - 0.5) * (
                        _BOOTSTRAP_NOISE_AMP * 2.0
                    )
        elif has_bootstrap:
            weyl_state = weyl_state + _WEYL_INCREMENT
            x = x + (float(weyl_state) * _WEYL_SCALE - 0.5) * (
                _BOOTSTRAP_NOISE_AMP * 2.0
            )
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

        if apply_saturation:
            y0 = sat_mix_clean * y0 + sat_blend * _algebraic_sat(y0)
            y1 = sat_mix_clean * y1 + sat_blend * _algebraic_sat(y1)
            y2 = sat_mix_clean * y2 + sat_blend * _algebraic_sat(y2)
            y3 = sat_mix_clean * y3 + sat_blend * _algebraic_sat(y3)
            s0 = sat_mix_clean * s0 + sat_blend * _algebraic_sat(s0)
            s1 = sat_mix_clean * s1 + sat_blend * _algebraic_sat(s1)
            s2 = sat_mix_clean * s2 + sat_blend * _algebraic_sat(s2)
            s3 = sat_mix_clean * s3 + sat_blend * _algebraic_sat(s3)

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
    solver: str = "newton",
    max_newton_iters: int = _DEFAULT_NEWTON_MAX_ITERS,
    newton_tolerance: float = _DEFAULT_NEWTON_TOLERANCE,
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
    clamped_drive = max(0.0, float(filter_drive))
    sat_blend = _drive_sat_blend(clamped_drive)
    return _apply_cascade_inner(
        sig,
        cutoff,
        resonance_boost,
        sample_rate,
        mode_int,
        clamped_drive,
        sat_blend,
        precomputed_g,
        nyquist_limit,
        bootstrap_seed,
        has_bootstrap,
        max(0.0, float(feedback_amount)),
        max(0.0, float(feedback_saturation)),
        max(0.0, min(3.0, float(filter_morph))),
        solver == "newton",
        max(1, int(max_newton_iters)),
        max(1e-12, float(newton_tolerance)),
    )


# ---------------------------------------------------------------------------
# SEM 2-pole SVF (Oberheim gentle OTA character, LP-Notch-HP morph)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _apply_sem_inner(
    signal: np.ndarray,
    cutoff_profile: np.ndarray,
    damping: float,
    sample_rate: int,
    mode_int: int,
    filter_drive: float,
    drive_blend: float,
    precomputed_g: float,
    nyquist_limit: float,
    bootstrap_seed: np.uint64,
    has_bootstrap: bool,
    feedback_amount: float,
    feedback_saturation: float,
    morph: float,
    use_newton_feedback: bool = False,
    max_newton_iters: int = _DEFAULT_NEWTON_MAX_ITERS,
    newton_tolerance: float = _DEFAULT_NEWTON_TOLERANCE,
) -> np.ndarray:
    """Oberheim SEM 2-pole SVF: gentle OTA resonance, LP→Notch→HP morph.

    Same ZDF/TPT integrator skeleton as ``_apply_linear_zdf_svf`` with
    three character choices that make it identifiable as SEM rather than
    plain SVF:

    1. Wider/shallower Q-to-damping curve (``damping = 1 / (q * 0.85)``):
       resonance peak is softer than textbook SVF at the same user Q,
       so the filter "blooms" rather than spikes.  Matches the OTA-based
       SEM's characteristic bass-preserving resonance shape.
    2. Gentle per-integrator ``_algebraic_sat`` cap on the state
       variables (no drive-dependent bidirectional feedback boost) —
       models OTA saturation under high drive without the Moog growl.
    3. Three-stage LP→Notch→HP morph under ``morph ∈ [0, 2]``, matching
       the real SEM's single-knob morph knob (BP is available via
       ``filter_mode="bandpass"`` but sits outside the continuous sweep).
       The notch output uses the classic ``low + high`` combination.
    """
    n = signal.shape[0]
    filtered = np.empty(n, dtype=np.float64)
    use_precomputed = precomputed_g >= 0.0

    low_state = 0.0
    band_state = 0.0

    # Drive stays active for the input stage but its character is softer
    # than SVF — a gentle tanh on the pre-filter signal rather than the
    # bidirectional drive/feedback boost.
    driven = drive_blend > 0.0
    drive_gain = 1.0 + 0.9 * filter_drive if driven else 1.0

    has_ext_fb = feedback_amount > 0.0
    # Newton external feedback is only well-defined in the clean (non-driven)
    # linear body — the input-stage tanh(x) under drive makes the instantaneous
    # map nonlinear in x and breaks the affine-collapse derivation.  Drive +
    # ext FB is a rare combo, so we keep the driven path on the unit-delay
    # branch.  Clean body + ext FB is the common case for pad feedback sweeps.
    newton_feedback_active = has_ext_fb and use_newton_feedback and drive_blend <= 0.0
    ext_fb_drive = 1.0 + 3.0 * feedback_saturation
    prev_output = 0.0
    weyl_state = bootstrap_seed

    # Three-stage LP→Notch→HP morph weights.
    use_morph = morph > 0.0
    m = morph
    lp_w = 1.0
    notch_w = 0.0
    hp_w = 0.0
    if use_morph:
        if m <= 1.0:
            lp_w = 1.0 - m
            notch_w = m
            hp_w = 0.0
        else:
            lp_w = 0.0
            notch_w = 2.0 - m
            hp_w = m - 1.0

    for i in range(n):
        if use_precomputed:
            g = precomputed_g
        else:
            fc = min(cutoff_profile[i], nyquist_limit)
            g = math.tan(math.pi * fc / sample_rate)

        if newton_feedback_active:
            # Affine-collapse the ZDF SVF body so output = A_out·x_in + B_out,
            # then Newton-solve the closed implicit equation
            # y = A_out·(raw + fb_amt·tanh(g_fb·y)) + B_out.
            denom_svf = 1.0 + 2.0 * damping * g + g * g
            state_term = (2.0 * damping + g) * band_state + low_state
            high_in = 1.0 / denom_svf
            high_off = -state_term / denom_svf
            band_in = g * high_in
            band_off = g * high_off + band_state
            low_in = g * band_in
            low_off = g * band_off + low_state
            if use_morph:
                notch_in = low_in + high_in
                notch_off = low_off + high_off
                A_out = lp_w * low_in + notch_w * notch_in + hp_w * high_in
                B_out = lp_w * low_off + notch_w * notch_off + hp_w * high_off
            elif mode_int == _LP:
                A_out = low_in
                B_out = low_off
            elif mode_int == _BP:
                A_out = band_in
                B_out = band_off
            elif mode_int == _HP:
                A_out = high_in
                B_out = high_off
            else:
                A_out = low_in + high_in
                B_out = low_off + high_off
            raw = signal[i]
            if has_bootstrap:
                weyl_state = weyl_state + _WEYL_INCREMENT
                raw = raw + (float(weyl_state) * _WEYL_SCALE - 0.5) * (
                    _BOOTSTRAP_NOISE_AMP * 2.0
                )
            affine_const = A_out * raw + B_out
            fb_scale = A_out * feedback_amount
            y_solved = _solve_ext_feedback_newton(
                prev_output,
                affine_const,
                fb_scale,
                ext_fb_drive,
                max_newton_iters,
                newton_tolerance,
            )
            x = raw + feedback_amount * math.tanh(ext_fb_drive * y_solved)
        else:
            x = signal[i]
            if has_ext_fb:
                x = x + feedback_amount * math.tanh(ext_fb_drive * prev_output)
            if has_bootstrap:
                weyl_state = weyl_state + _WEYL_INCREMENT
                x = x + (float(weyl_state) * _WEYL_SCALE - 0.5) * (
                    _BOOTSTRAP_NOISE_AMP * 2.0
                )
            if driven:
                clean = x
                shaped = drive_gain * math.tanh(x)
                x = clean + drive_blend * (shaped - clean)

        high = (x - (2.0 * damping + g) * band_state - low_state) / (
            1.0 + 2.0 * damping * g + g * g
        )
        band = g * high + band_state
        low = g * band + low_state
        band_state = band + g * high
        low_state = low + g * band

        # Gentle OTA-style state cap — always active, not drive-gated.
        # Keeps self-oscillation musical without letting the integrators
        # run away at high Q.
        band_state = _algebraic_sat(band_state)
        low_state = _algebraic_sat(low_state)

        if use_morph:
            notch = low + high
            output = lp_w * low + notch_w * notch + hp_w * high
        elif mode_int == _LP:
            output = low
        elif mode_int == _BP:
            output = band
        elif mode_int == _HP:
            output = high
        else:
            output = low + high  # notch

        prev_output = output
        filtered[i] = output

    return filtered


def _apply_sem(
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
    solver: str = "newton",
    max_newton_iters: int = _DEFAULT_NEWTON_MAX_ITERS,
    newton_tolerance: float = _DEFAULT_NEWTON_TOLERANCE,
) -> np.ndarray:
    """Apply the SEM 2-pole state-variable filter.

    Gentle OTA-style resonance with bass-preserving behavior, continuous
    LP→Notch→HP morph under ``filter_morph ∈ [0, 2]``.  BP available via
    ``filter_mode="bandpass"`` (outside the morph sweep to match real SEM).
    """
    q = max(0.5, float(resonance_q))
    # Wider Q curve than SVF: damping = 1/(q*0.85).  At q=0.707 this gives
    # damping ≈ 1.66 (noticeably softer than SVF's 1.414).  At q=10 damping
    # ≈ 0.12 — still peaked but less extreme than SVF's 0.1, audibly more
    # "bloomy" and less spiky.
    damping = 1.0 / (q * 0.85)

    mode_int = _MODE_STR_TO_INT.get(filter_mode, _LP)
    # Notch has its own explicit path (low + high), so don't coerce it.

    sig = np.asarray(signal, dtype=np.float64)
    cutoff = np.asarray(cutoff_profile, dtype=np.float64)

    nyquist_limit = sample_rate * _NYQUIST_CLAMP_RATIO
    if cutoff.size > 0 and np.all(cutoff == cutoff[0]):
        clamped_fc = min(float(cutoff[0]), nyquist_limit)
        precomputed_g = math.tan(math.pi * clamped_fc / sample_rate)
    else:
        cutoff = np.minimum(cutoff, nyquist_limit)
        precomputed_g = -1.0

    bootstrap_seed = _bootstrap_seed(sig, "sem")
    has_bootstrap = sig.shape[0] > 0
    morph = max(0.0, min(2.0, float(filter_morph)))
    clamped_drive = max(0.0, float(filter_drive))
    drive_blend = _drive_sat_blend(clamped_drive)
    return _apply_sem_inner(
        sig,
        cutoff,
        damping,
        sample_rate,
        mode_int,
        clamped_drive,
        drive_blend,
        precomputed_g,
        nyquist_limit,
        bootstrap_seed,
        has_bootstrap,
        max(0.0, float(feedback_amount)),
        max(0.0, float(feedback_saturation)),
        morph,
        solver == "newton",
        max(1, int(max_newton_iters)),
        max(1e-12, float(newton_tolerance)),
    )


# ---------------------------------------------------------------------------
# Jupiter 4-pole OTA cascade (Roland IR3109-flavored)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _apply_jupiter_inner(
    signal: np.ndarray,
    cutoff_profile: np.ndarray,
    k_feedback: float,
    sample_rate: int,
    mode_int: int,
    filter_drive: float,
    sat_blend: float,
    precomputed_g: float,
    nyquist_limit: float,
    bootstrap_seed: np.uint64,
    has_bootstrap: bool,
    feedback_amount: float,
    feedback_saturation: float,
    morph: float,
) -> np.ndarray:
    """Jupiter 4-pole OTA cascade with single global tanh feedback (ADAA).

    Structurally: four ZDF 1-poles in series wrapped by ONE global
    feedback loop with a single tanh at the summation.  Distinguishing
    character vs. Moog ladder (which applies tanh per-stage) is
    cleaner saturation and far less bass loss at high Q.  The feedback
    nonlinearity uses first-order ADAA to cut aliasing at audio rates.
    """
    n = signal.shape[0]
    filtered = np.empty(n, dtype=np.float64)
    use_precomputed = precomputed_g >= 0.0

    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    fb_prev = 0.0  # for ADAA on the global feedback tanh

    drive_gain = 1.0 + 1.3 * filter_drive
    apply_saturation = sat_blend > 0.0
    sat_mix_clean = 1.0 - sat_blend

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

        x = signal[i]
        if has_ext_fb:
            x = x + feedback_amount * math.tanh(ext_fb_drive * prev_ext_output)
        if has_bootstrap:
            weyl_state = weyl_state + _WEYL_INCREMENT
            x = x + (float(weyl_state) * _WEYL_SCALE - 0.5) * (
                _BOOTSTRAP_NOISE_AMP * 2.0
            )
        x = drive_gain * x

        # Global feedback summation with single ADAA tanh.  Uses s3 (the
        # previous output state) as the one-sample-delayed feedback signal
        # — characteristic of explicit ADAA ladders when no Newton solve
        # is available.  Aliasing reduction comes from the log-cosh
        # antiderivative, not from cross-sample correction.
        fb_curr = s3
        fb_shaped = _adaa_tanh(fb_curr, fb_prev)
        fb_prev = fb_curr
        u = x - k_feedback * fb_shaped

        y0 = alpha * u + one_minus_alpha * s0
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

        if apply_saturation:
            # Very mild state cap only — Jupiter's character is cleaner
            # than Moog, so we don't distort per-stage.
            s3 = sat_mix_clean * s3 + sat_blend * _algebraic_sat(s3)

        if morph <= 0.0:
            if mode_int == _LP:
                output = y3
            elif mode_int == _BP:
                output = y1 - y3
            else:  # HP
                output = x - y3
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
                if m <= 1.0:
                    output = (1.0 - m) * (y1 - y3) + m * (y0 - y2)
                elif m <= 2.0:
                    f = m - 1.0
                    output = (1.0 - f) * (y0 - y2) + f * (y0 - y1)
                else:
                    f = m - 2.0
                    output = (1.0 - f) * (y0 - y1) + f * y0
            else:
                if m <= 1.0:
                    output = (1.0 - m) * (x - y3) + m * (x - y2)
                elif m <= 2.0:
                    f = m - 1.0
                    output = (1.0 - f) * (x - y2) + f * (x - y1)
                else:
                    f = m - 2.0
                    output = (1.0 - f) * (x - y1) + f * (x - y0)

        prev_ext_output = output
        filtered[i] = output

    return filtered


@numba.njit(cache=True)
def _apply_jupiter_newton_inner(
    signal: np.ndarray,
    cutoff_profile: np.ndarray,
    k_feedback: float,
    sample_rate: int,
    mode_int: int,
    filter_drive: float,
    sat_blend: float,
    precomputed_g: float,
    nyquist_limit: float,
    bootstrap_seed: np.uint64,
    has_bootstrap: bool,
    feedback_amount: float,
    feedback_saturation: float,
    morph: float,
    max_iters: int,
    tolerance: float,
    close_ext_feedback: bool,
) -> np.ndarray:
    """Jupiter 4-pole with per-sample Newton solve of the feedback tanh.

    Closes the delay-free global feedback loop to machine precision.
    The implicit equation per sample is

        y3 = F(x - k * tanh(y3))

    where F is the cascaded 4x 1-pole transfer.  For ZDF 1-poles the
    composition is linear in the input so we can write

        y3 = A * (x - k * tanh(y3)) + B

    with A = alpha^4 (pre-state coefficient) and B folded state
    contribution.  Newton iterates ``y3`` until the residual
    ``y3 - A*(x - k*tanh(y3)) - B`` is below ``tolerance``.
    """
    n = signal.shape[0]
    filtered = np.empty(n, dtype=np.float64)
    use_precomputed = precomputed_g >= 0.0

    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0

    drive_gain = 1.0 + 1.3 * filter_drive
    apply_saturation = sat_blend > 0.0
    sat_mix_clean = 1.0 - sat_blend

    has_ext_fb = feedback_amount > 0.0
    ext_fb_drive = 1.0 + 3.0 * feedback_saturation
    prev_ext_output = 0.0
    weyl_state = bootstrap_seed
    y3_guess = 0.0

    for i in range(n):
        if use_precomputed:
            g = precomputed_g
        else:
            fc = min(cutoff_profile[i], nyquist_limit)
            g = math.tan(math.pi * fc / sample_rate)

        alpha = g / (1.0 + g)
        one_minus_alpha = 1.0 - alpha

        x = signal[i]
        if has_bootstrap:
            weyl_state = weyl_state + _WEYL_INCREMENT
            x = x + (float(weyl_state) * _WEYL_SCALE - 0.5) * (
                _BOOTSTRAP_NOISE_AMP * 2.0
            )
        outer_newton = (
            close_ext_feedback and has_ext_fb and morph <= 0.0 and mode_int == _LP
        )
        if has_ext_fb and not outer_newton:
            x = x + feedback_amount * math.tanh(ext_fb_drive * prev_ext_output)
        x = drive_gain * x

        # Closed-form coefficients for the cascaded ZDF 1-poles relating
        # feedback-loop input u to y3:  y3 = A * u + B  where A = alpha^4
        # and B accumulates the state contributions at each stage.
        a2 = alpha * alpha
        a3 = a2 * alpha
        a4 = a3 * alpha
        b_state = (
            a3 * one_minus_alpha * s0
            + a2 * one_minus_alpha * s1
            + alpha * one_minus_alpha * s2
            + one_minus_alpha * s3
        )

        y = y3_guess
        if outer_newton:
            # Output is y3 in this branch, so the closed implicit equation
            # becomes a single scalar Newton in y3:
            #   F(y3) = y3 - a4·(x_raw + D·tanh(g_fb·y3) - k·tanh(y3)) - b_state
            # with x_raw = drive_gain·(signal[i] + bootstrap),
            # D = drive_gain·fb_amt.
            D_outer = drive_gain * feedback_amount
            a4D = a4 * D_outer
            a4k = a4 * k_feedback
            for _ in range(max_iters):
                ty = math.tanh(y)
                gy = ext_fb_drive * y
                th_g = math.tanh(gy)
                sech2_inner = 1.0 - ty * ty
                sech2_outer = 1.0 - th_g * th_g
                resid = y - a4 * x - a4D * th_g + a4k * ty - b_state
                if math.fabs(resid) < tolerance:
                    break
                deriv = 1.0 + a4k * sech2_inner - a4D * ext_fb_drive * sech2_outer
                if deriv < 1e-6 and deriv > -1e-6:
                    deriv = 1e-6 if deriv >= 0.0 else -1e-6
                y = y - resid / deriv
            # Replay x with the solved outer feedback so stage states advance
            # with the implicit solution.
            x = x + drive_gain * feedback_amount * math.tanh(ext_fb_drive * y)
        else:
            for _ in range(max_iters):
                ty = math.tanh(y)
                resid = y - a4 * (x - k_feedback * ty) - b_state
                if math.fabs(resid) < tolerance:
                    break
                deriv = 1.0 + a4 * k_feedback * (1.0 - ty * ty)
                y = y - resid / deriv
        y3_guess = y

        # Roll the stage states forward given the solved u.
        ty = math.tanh(y)
        u = x - k_feedback * ty

        y0 = alpha * u + one_minus_alpha * s0
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

        if apply_saturation:
            s3 = sat_mix_clean * s3 + sat_blend * _algebraic_sat(s3)

        if morph <= 0.0:
            if mode_int == _LP:
                output = y3
            elif mode_int == _BP:
                output = y1 - y3
            else:
                output = x - y3
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
                if m <= 1.0:
                    output = (1.0 - m) * (y1 - y3) + m * (y0 - y2)
                elif m <= 2.0:
                    f = m - 1.0
                    output = (1.0 - f) * (y0 - y2) + f * (y0 - y1)
                else:
                    f = m - 2.0
                    output = (1.0 - f) * (y0 - y1) + f * y0
            else:
                if m <= 1.0:
                    output = (1.0 - m) * (x - y3) + m * (x - y2)
                elif m <= 2.0:
                    f = m - 1.0
                    output = (1.0 - f) * (x - y2) + f * (x - y1)
                else:
                    f = m - 2.0
                    output = (1.0 - f) * (x - y1) + f * (x - y0)

        prev_ext_output = output
        filtered[i] = output

    return filtered


def _apply_jupiter_filter(
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
    solver: str = "adaa",
    max_newton_iters: int = _DEFAULT_NEWTON_MAX_ITERS,
    newton_tolerance: float = _DEFAULT_NEWTON_TOLERANCE,
) -> np.ndarray:
    """Apply a Jupiter-8-flavored 4-pole OTA cascade filter.

    Single global tanh feedback loop (not per-stage) yields the
    characteristic IR3109 cleanliness and minimal bass loss at high Q.
    Pair with ``hpf_cutoff_hz > 0`` (via ``apply_filter``) for the full
    Jupiter-8 dual-filter architecture; leave it at 0 for a Juno-106 LP.

    The Q→k mapping is softer than Moog's (``k`` saturates at ≈3.2 vs
    Moog's 4.0) so the filter can sit at high Q without sucking out the
    fundamental.
    """
    q = max(0.5, float(resonance_q))
    # Jupiter's OTA cascade has softer Q→feedback than Moog — crucial for
    # the bass-preservation character.  Moog effectively runs ``k → 4`` at
    # self-oscillation; we cap around 2.6 and reach it more gradually.
    # At q=0.707 gives k ≈ 0.74; at q=4 gives k ≈ 2.28; at q=10 gives k ≈ 2.47
    # — noticeably less bass suction than the ladder at matched musical Q.
    k_feedback = max(0.0, min(2.6, 2.6 * (1.0 - 1.0 / (2.0 * q))))

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

    bootstrap_seed = _bootstrap_seed(sig, "jupiter")
    has_bootstrap = sig.shape[0] > 0
    clamped_drive = max(0.0, float(filter_drive))
    sat_blend = _drive_sat_blend(clamped_drive)
    morph = max(0.0, min(3.0, float(filter_morph)))

    if solver == "newton":
        return _apply_jupiter_newton_inner(
            sig,
            cutoff,
            k_feedback,
            sample_rate,
            mode_int,
            clamped_drive,
            sat_blend,
            precomputed_g,
            nyquist_limit,
            bootstrap_seed,
            has_bootstrap,
            max(0.0, float(feedback_amount)),
            max(0.0, float(feedback_saturation)),
            morph,
            max(1, int(max_newton_iters)),
            max(1e-12, float(newton_tolerance)),
            True,
        )
    return _apply_jupiter_inner(
        sig,
        cutoff,
        k_feedback,
        sample_rate,
        mode_int,
        clamped_drive,
        sat_blend,
        precomputed_g,
        nyquist_limit,
        bootstrap_seed,
        has_bootstrap,
        max(0.0, float(feedback_amount)),
        max(0.0, float(feedback_saturation)),
        morph,
    )


# ---------------------------------------------------------------------------
# K35 Sallen-Key (Korg MS-20 style, diode-clipped feedback)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _apply_k35_inner(
    signal: np.ndarray,
    cutoff_profile: np.ndarray,
    k_feedback: float,
    asymmetry: float,
    sample_rate: int,
    mode_int: int,
    filter_drive: float,
    sat_blend: float,
    precomputed_g: float,
    nyquist_limit: float,
    bootstrap_seed: np.uint64,
    has_bootstrap: bool,
    feedback_amount: float,
    feedback_saturation: float,
) -> np.ndarray:
    """Korg35 LP: two ZDF 1-poles inside a positive-feedback resonance loop.

    Hybrid solver: the two series 1-pole stages use ZDF/TPT integration,
    and an ``alpha = 1 + k * g_1p^2`` pre-scale approximates the
    implicit-solve denominator, but the diode-shaped feedback term reads
    ``y_prev`` (one-sample-delayed output) rather than the true current
    ``y``.  Self-oscillation threshold is therefore tuned empirically
    rather than matching the textbook ``k = 2`` of a fully-implicit K35.
    A proper closed-form solve would require Newton iteration on the
    diode nonlinearity inside each sample; the empirical tuning here
    preserves the MS-20 character without that cost.

    Output feedback shape: ``_diode_shape`` (asymmetric) rather than
    ``tanh`` — the defining MS-20 nonlinearity.  Input-stage drive
    adds an asymmetric soft-clip so the filter "crunches" at moderate
    drives.

    HP mode mirrors the LP topology with an HPF stage inside the loop
    and an LPF on the output — MS-20's 12 dB/oct HP character.
    """
    n = signal.shape[0]
    filtered = np.empty(n, dtype=np.float64)
    use_precomputed = precomputed_g >= 0.0

    # One-pole ZDF state variables:
    #   s_a: first stage state
    #   s_b: second stage state
    s_a = 0.0
    s_b = 0.0
    y_prev = 0.0  # previous output for feedback sample delay

    drive_gain = 1.0 + 1.8 * filter_drive  # input stage overdrive
    apply_saturation = sat_blend > 0.0
    sat_mix_clean = 1.0 - sat_blend
    # Input-stage crunch: asymmetric soft-clip before the filter.  Gated by
    # drive so it contributes nothing at drive=0.
    has_input_crunch = filter_drive > 0.0

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

        # ZDF 1-pole integrator coefficient.
        g_1p = g / (1.0 + g)
        one_minus = 1.0 - g_1p
        # Alpha-compensation denominator for the K35 LP positive-feedback loop.
        # alpha = 1 + k * g_1p * g_1p (this is the implicit solve that avoids
        # a cross-sample delay on the feedback path).
        alpha = 1.0 / (1.0 + k_feedback * g_1p * g_1p)

        x = signal[i]
        if has_ext_fb:
            x = x + feedback_amount * math.tanh(ext_fb_drive * prev_ext_output)
        if has_bootstrap:
            weyl_state = weyl_state + _WEYL_INCREMENT
            x = x + (float(weyl_state) * _WEYL_SCALE - 0.5) * (
                _BOOTSTRAP_NOISE_AMP * 2.0
            )
        # Input-stage overdrive — asymmetric for that MS-20 crunch.
        if has_input_crunch:
            clean = x
            shaped = _diode_shape(drive_gain * x, asymmetry * 0.6)
            x = clean + sat_blend * (shaped - clean)

        # Feedback path: diode-shape the previous output.  asymmetry
        # drives the even-harmonic bias that defines MS-20's "snarl".
        fb = _diode_shape(y_prev, asymmetry)

        if mode_int == _HP:
            # K35 HP topology: HPF inside loop, LPF on output.
            u = alpha * (x + k_feedback * fb)
            # HP 1-pole TPT: v_hp = (u - s_a) / (1 + g); hp = v_hp - g*s_a/(1+g)?
            # Standard ZDF HP 1-pole:
            #   v = (u - s_a) / (1 + g)
            #   hp = v
            #   s_a = s_a + 2*g*v
            v = (u - s_a) / (1.0 + g)
            hp1 = v
            s_a = s_a + 2.0 * g * v
            # LPF on output stage:
            v2 = (hp1 - s_b) / (1.0 + g)
            s_b = s_b + 2.0 * g * v2
            y = s_b
        else:  # LP (also BP/notch coerce)
            # K35 LP: two LPs in series inside the positive-feedback loop.
            u = alpha * (x + k_feedback * fb)
            lp1 = g_1p * u + one_minus * s_a
            s_a = 2.0 * lp1 - s_a  # TPT trapezoidal state update
            lp2 = g_1p * lp1 + one_minus * s_b
            s_b = 2.0 * lp2 - s_b
            y = lp2

        y_prev = y

        if apply_saturation:
            s_a = sat_mix_clean * s_a + sat_blend * _algebraic_sat(s_a)
            s_b = sat_mix_clean * s_b + sat_blend * _algebraic_sat(s_b)

        output = y
        prev_ext_output = output
        filtered[i] = output

    return filtered


def _apply_k35(
    signal: np.ndarray,
    *,
    cutoff_profile: np.ndarray,
    resonance_q: float,
    sample_rate: int,
    filter_mode: str,
    filter_drive: float,
    k35_feedback_asymmetry: float = 0.0,
    feedback_amount: float = 0.0,
    feedback_saturation: float = 0.0,
) -> np.ndarray:
    """Apply a Korg35 (MS-20 style) 12 dB/oct Sallen-Key filter.

    Diode-clipped feedback gives the characteristic MS-20 "snarl" —
    push ``k35_feedback_asymmetry`` toward ``1.0`` for deranged character,
    keep near ``0.0`` for a cleaner K35.  ``filter_drive`` engages an
    asymmetric input-stage soft-clip that models the real MS-20's
    famous overloading.
    """
    q = max(0.5, float(resonance_q))
    # K35 self-oscillation threshold is at k ≈ 2; map Q → k with a saturating
    # curve so we reach self-oscillation smoothly at high user Q.
    # At q=0.707 → k ≈ 0.0; at q=4 → k ≈ 1.5; at q=10 → k ≈ 1.85; at q=30 → k ≈ 1.97
    k_feedback = max(0.0, min(1.98, 2.0 * (1.0 - 0.7 / math.sqrt(q))))
    asym = max(0.0, min(1.0, float(k35_feedback_asymmetry)))

    mode_int = _MODE_STR_TO_INT.get(filter_mode, _LP)
    # K35 only exposes LP and HP. BP/notch coerce to LP.
    if mode_int in (_BP, _NOTCH):
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

    bootstrap_seed = _bootstrap_seed(sig, "k35")
    has_bootstrap = sig.shape[0] > 0
    clamped_drive = max(0.0, float(filter_drive))
    sat_blend = _drive_sat_blend(clamped_drive)
    return _apply_k35_inner(
        sig,
        cutoff,
        k_feedback,
        asym,
        sample_rate,
        mode_int,
        clamped_drive,
        sat_blend,
        precomputed_g,
        nyquist_limit,
        bootstrap_seed,
        has_bootstrap,
        max(0.0, float(feedback_amount)),
        max(0.0, float(feedback_saturation)),
    )


# ---------------------------------------------------------------------------
# Diode 3-pole ladder (TB-303 acid character)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _apply_diode_inner(
    signal: np.ndarray,
    cutoff_profile: np.ndarray,
    k_feedback: float,
    sample_rate: int,
    filter_drive: float,
    sat_blend: float,
    precomputed_g: float,
    nyquist_limit: float,
    bootstrap_seed: np.uint64,
    has_bootstrap: bool,
    feedback_amount: float,
    feedback_saturation: float,
    morph: float,
) -> np.ndarray:
    """TB-303 style 3-pole diode ladder, ADAA solver.

    Three ZDF 1-poles in series with asymmetric diode clipping at each
    stage (via ``_diode_shape``) and a feedback tap between stages 2 and
    3 — this tap location is responsible for the 303's characteristic
    bass-suck-with-squelch as Q rises.  Pre-filter input drive amplifies
    the feedback asymmetry so high-drive/high-Q produces the "acid bark".

    LP-only output (BP/HP coerce to LP upstream).  ``morph ∈ [0, 2]``
    blends 3→2→1 pole output for 18→12→6 dB/oct.
    """
    n = signal.shape[0]
    filtered = np.empty(n, dtype=np.float64)
    use_precomputed = precomputed_g >= 0.0

    s0 = 0.0
    s1 = 0.0
    s2 = 0.0

    drive_gain = 1.0 + 1.2 * filter_drive
    # Feedback-path asymmetry is the defining 303 character — per-stage
    # clipping is intentionally much milder so the fundamental survives
    # at moderate drive.  Real 303 diode clipping lives primarily in the
    # resonance feedback.
    feedback_asym = min(1.0, 0.4 + 0.6 * filter_drive)
    # Per-stage state cap only engages under meaningful drive and uses
    # algebraic_sat (transparent below ~1.0) to avoid squashing the
    # fundamental at low-to-moderate drive.
    apply_saturation = sat_blend > 0.0

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

        g_1p = g / (1.0 + g)
        one_minus = 1.0 - g_1p

        x = signal[i]
        if has_ext_fb:
            x = x + feedback_amount * math.tanh(ext_fb_drive * prev_ext_output)
        if has_bootstrap:
            weyl_state = weyl_state + _WEYL_INCREMENT
            x = x + (float(weyl_state) * _WEYL_SCALE - 0.5) * (
                _BOOTSTRAP_NOISE_AMP * 2.0
            )
        x = drive_gain * x

        # Feedback tap between stages 2 and 3 (from s2, not y3).  This
        # tap location is what gives 303s their characteristic bass
        # suck + squelch vs Moog (which taps from y3).  The diode shape
        # here IS the acid character — hold this aggressive even at
        # modest Q.
        fb = _diode_shape(s2, feedback_asym)
        u = x - k_feedback * fb

        y0 = g_1p * u + one_minus * s0
        s0 = 2.0 * y0 - s0

        y1 = g_1p * y0 + one_minus * s1
        s1 = 2.0 * y1 - s1

        y2 = g_1p * y1 + one_minus * s2
        s2 = 2.0 * y2 - s2

        if apply_saturation:
            # Gentle state cap only — prevents runaway at Q→∞, does not
            # meaningfully distort in-band content.
            s0 = _algebraic_sat(s0)
            s1 = _algebraic_sat(s1)
            s2 = _algebraic_sat(s2)

        # Output pole-tap morph: 18→12→6 dB/oct (y2→y1→y0).
        if morph <= 0.0:
            output = y2
        else:
            m = min(morph, 2.0)
            if m <= 1.0:
                output = (1.0 - m) * y2 + m * y1
            else:
                f = m - 1.0
                output = (1.0 - f) * y1 + f * y0

        prev_ext_output = output
        filtered[i] = output

    return filtered


@numba.njit(cache=True)
def _apply_diode_newton_inner(
    signal: np.ndarray,
    cutoff_profile: np.ndarray,
    k_feedback: float,
    sample_rate: int,
    filter_drive: float,
    sat_blend: float,
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
    """TB-303 style 3-pole diode ladder, Newton-iterated feedback solve.

    Closes the delay-free loop per sample so the feedback diode isn't
    sample-lagged.  For diode ladders this is audibly better at high Q
    than ADAA since diodes are steeper than tanh and the one-step
    approximation leaves artifacts at resonance.  Solves for ``y2``
    (the tap output) where

        y2 = A * (u) + B
        u  = x - k * diode(y2)

    with ``A = g_1p^3 * …`` coefficients accumulated through the three
    stages.  The per-stage diode also enters the solve but we linearize
    those around the current state — for the tap feedback we Newton-
    iterate since that's where the characteristic squelch lives.
    """
    n = signal.shape[0]
    filtered = np.empty(n, dtype=np.float64)
    use_precomputed = precomputed_g >= 0.0

    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    y2_guess = 0.0

    drive_gain = 1.0 + 1.2 * filter_drive
    feedback_asym = min(1.0, 0.4 + 0.6 * filter_drive)
    apply_saturation = sat_blend > 0.0

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

        g_1p = g / (1.0 + g)
        one_minus = 1.0 - g_1p

        x = signal[i]
        if has_ext_fb:
            x = x + feedback_amount * math.tanh(ext_fb_drive * prev_ext_output)
        if has_bootstrap:
            weyl_state = weyl_state + _WEYL_INCREMENT
            x = x + (float(weyl_state) * _WEYL_SCALE - 0.5) * (
                _BOOTSTRAP_NOISE_AMP * 2.0
            )
        x = drive_gain * x

        # Linear coefficient mapping u → y2 through the three stages:
        #   y2 = g_1p^3 * u + B_state  (per-stage linear; the diode
        # nonlinearity lives in the feedback tap, not in each stage).
        g3 = g_1p * g_1p * g_1p
        b_state = g_1p * g_1p * one_minus * s0 + g_1p * one_minus * s1 + one_minus * s2

        # Newton-solve y2 = g3 * (x - k*diode(y2)) + b_state
        y = y2_guess
        for _ in range(max_iters):
            a = feedback_asym
            is_pos = 1.0 - 0.35 * a
            is_neg = 1.0 + 0.5 * a
            if y >= 0.0:
                diode_norm = math.log1p(1.0 / is_pos)
                dy = math.log1p(y / is_pos) / diode_norm
                deriv_diode = 1.0 / ((is_pos + y) * diode_norm)
            else:
                diode_norm = math.log1p(1.0 / is_neg)
                dy = -math.log1p(-y / is_neg) / diode_norm
                deriv_diode = 1.0 / ((is_neg - y) * diode_norm)
            resid = y - g3 * (x - k_feedback * dy) - b_state
            if math.fabs(resid) < tolerance:
                break
            deriv = 1.0 + g3 * k_feedback * deriv_diode
            y = y - resid / deriv
        y2_guess = y

        fb = _diode_shape(y, feedback_asym)
        u = x - k_feedback * fb

        y0 = g_1p * u + one_minus * s0
        s0 = 2.0 * y0 - s0

        y1 = g_1p * y0 + one_minus * s1
        s1 = 2.0 * y1 - s1

        y2 = g_1p * y1 + one_minus * s2
        s2 = 2.0 * y2 - s2

        if apply_saturation:
            s0 = _algebraic_sat(s0)
            s1 = _algebraic_sat(s1)
            s2 = _algebraic_sat(s2)

        if morph <= 0.0:
            output = y2
        else:
            m = min(morph, 2.0)
            if m <= 1.0:
                output = (1.0 - m) * y2 + m * y1
            else:
                f = m - 1.0
                output = (1.0 - f) * y1 + f * y0

        prev_ext_output = output
        filtered[i] = output

    return filtered


def _apply_diode_filter(
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
    solver: str = "adaa",
    max_newton_iters: int = _DEFAULT_NEWTON_MAX_ITERS,
    newton_tolerance: float = _DEFAULT_NEWTON_TOLERANCE,
) -> np.ndarray:
    """Apply a TB-303 style 3-pole diode ladder filter.

    18 dB/oct LP with asymmetric per-stage diode shaping and a feedback
    tap between stages 2 and 3.  ``filter_drive`` amplifies the
    asymmetry and enables per-stage diode clipping.  Newton solver
    closes the feedback loop more accurately at high Q — recommended
    for authentic acid squelch.
    """
    _ = filter_mode  # diode is LP-only; argument accepted for API symmetry
    q = max(0.5, float(resonance_q))
    # Diode ladder self-oscillation at k ≈ 1.7 on our normalized path.
    # Steep ramp — squelch intensifies quickly with user Q.
    k_feedback = max(0.0, min(1.7, 1.75 * (1.0 - 0.65 / math.sqrt(q))))

    sig = np.asarray(signal, dtype=np.float64)
    cutoff = np.asarray(cutoff_profile, dtype=np.float64)

    nyquist_limit = sample_rate * _NYQUIST_CLAMP_RATIO
    if cutoff.size > 0 and np.all(cutoff == cutoff[0]):
        clamped_fc = min(float(cutoff[0]), nyquist_limit)
        precomputed_g = math.tan(math.pi * clamped_fc / sample_rate)
    else:
        cutoff = np.minimum(cutoff, nyquist_limit)
        precomputed_g = -1.0

    bootstrap_seed = _bootstrap_seed(sig, "diode")
    has_bootstrap = sig.shape[0] > 0
    clamped_drive = max(0.0, float(filter_drive))
    sat_blend = _drive_sat_blend(clamped_drive)
    morph = max(0.0, min(2.0, float(filter_morph)))

    if solver == "newton":
        return _apply_diode_newton_inner(
            sig,
            cutoff,
            k_feedback,
            sample_rate,
            clamped_drive,
            sat_blend,
            precomputed_g,
            nyquist_limit,
            bootstrap_seed,
            has_bootstrap,
            max(0.0, float(feedback_amount)),
            max(0.0, float(feedback_saturation)),
            morph,
            max(1, int(max_newton_iters)),
            max(1e-12, float(newton_tolerance)),
        )
    return _apply_diode_inner(
        sig,
        cutoff,
        k_feedback,
        sample_rate,
        clamped_drive,
        sat_blend,
        precomputed_g,
        nyquist_limit,
        bootstrap_seed,
        has_bootstrap,
        max(0.0, float(feedback_amount)),
        max(0.0, float(feedback_saturation)),
        morph,
    )


# ---------------------------------------------------------------------------
# Unified filter dispatch
# ---------------------------------------------------------------------------


# Topology wrappers translate a :class:`FilterParams` bundle into the specific
# subset of kwargs their underlying ``_apply_*`` implementation expects.  They
# intentionally keep the same positional/keyword calls that the previous
# if/elif tree used so audio behavior is identical.


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
    filter_solver: str = "newton",
    max_newton_iters: int = _DEFAULT_NEWTON_MAX_ITERS,
    newton_tolerance: float = _DEFAULT_NEWTON_TOLERANCE,
    k35_feedback_asymmetry: float = 0.0,
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

    fp = FilterParams(
        resonance_q=resonance_q,
        filter_mode=filter_mode,
        filter_drive=filter_drive,
        filter_even_harmonics=filter_even_harmonics,
        filter_topology=filter_topology,
        bass_compensation=bass_compensation,
        filter_morph=filter_morph,
        hpf_cutoff_hz=hpf_cutoff_hz,
        hpf_resonance_q=hpf_resonance_q,
        feedback_amount=feedback_amount,
        feedback_saturation=feedback_saturation,
        filter_solver=filter_solver,
        max_newton_iters=max_newton_iters,
        newton_tolerance=newton_tolerance,
        k35_feedback_asymmetry=k35_feedback_asymmetry,
    )

    sig = np.asarray(signal, dtype=np.float64)

    if fp.hpf_cutoff_hz > 0.0:
        nyquist_limit = sample_rate * _NYQUIST_CLAMP_RATIO
        hpf_cutoff = min(float(fp.hpf_cutoff_hz), nyquist_limit)
        hpf_g = math.tan(math.pi * hpf_cutoff / sample_rate)
        hpf_cutoff_profile = np.full(sig.shape[0], hpf_cutoff, dtype=np.float64)
        hpf_damping = 1.0 / max(0.5, float(fp.hpf_resonance_q))
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

    topology = fp.filter_topology
    morph = max(0.0, float(fp.filter_morph))
    if topology == "svf":
        if morph == 0.0:
            return apply_zdf_svf(
                sig,
                cutoff_profile=cutoff_profile,
                resonance_q=fp.resonance_q,
                sample_rate=sample_rate,
                filter_mode=fp.filter_mode,
                filter_drive=fp.filter_drive,
                filter_even_harmonics=fp.filter_even_harmonics,
                feedback_amount=fp.feedback_amount,
                feedback_saturation=fp.feedback_saturation,
                solver=fp.filter_solver,
                max_newton_iters=fp.max_newton_iters,
                newton_tolerance=fp.newton_tolerance,
            )
        return _apply_svf_with_morph(
            sig,
            cutoff_profile=cutoff_profile,
            resonance_q=fp.resonance_q,
            sample_rate=sample_rate,
            filter_mode=fp.filter_mode,
            filter_drive=fp.filter_drive,
            filter_even_harmonics=fp.filter_even_harmonics,
            feedback_amount=fp.feedback_amount,
            feedback_saturation=fp.feedback_saturation,
            morph=morph,
            solver=fp.filter_solver,
            max_newton_iters=fp.max_newton_iters,
            newton_tolerance=fp.newton_tolerance,
        )
    if topology == "ladder":
        return _apply_ladder_filter(
            sig,
            cutoff_profile=cutoff_profile,
            resonance_q=fp.resonance_q,
            sample_rate=sample_rate,
            filter_mode=fp.filter_mode,
            filter_drive=fp.filter_drive,
            bass_compensation=fp.bass_compensation,
            filter_even_harmonics=fp.filter_even_harmonics,
            feedback_amount=fp.feedback_amount,
            feedback_saturation=fp.feedback_saturation,
            filter_morph=morph,
            solver=fp.filter_solver,
            max_newton_iters=fp.max_newton_iters,
            newton_tolerance=fp.newton_tolerance,
        )
    if topology == "sallen_key":
        return _apply_sallen_key(
            sig,
            cutoff_profile=cutoff_profile,
            resonance_q=fp.resonance_q,
            sample_rate=sample_rate,
            filter_mode=fp.filter_mode,
            filter_drive=fp.filter_drive,
            feedback_amount=fp.feedback_amount,
            feedback_saturation=fp.feedback_saturation,
            solver=fp.filter_solver,
            max_newton_iters=fp.max_newton_iters,
            newton_tolerance=fp.newton_tolerance,
        )
    if topology == "cascade":
        return _apply_cascade(
            sig,
            cutoff_profile=cutoff_profile,
            resonance_q=fp.resonance_q,
            sample_rate=sample_rate,
            filter_mode=fp.filter_mode,
            filter_drive=fp.filter_drive,
            feedback_amount=fp.feedback_amount,
            feedback_saturation=fp.feedback_saturation,
            filter_morph=morph,
            solver=fp.filter_solver,
            max_newton_iters=fp.max_newton_iters,
            newton_tolerance=fp.newton_tolerance,
        )
    if topology == "sem":
        return _apply_sem(
            sig,
            cutoff_profile=cutoff_profile,
            resonance_q=fp.resonance_q,
            sample_rate=sample_rate,
            filter_mode=fp.filter_mode,
            filter_drive=fp.filter_drive,
            feedback_amount=fp.feedback_amount,
            feedback_saturation=fp.feedback_saturation,
            filter_morph=fp.filter_morph,
            solver=fp.filter_solver,
            max_newton_iters=fp.max_newton_iters,
            newton_tolerance=fp.newton_tolerance,
        )
    if topology == "jupiter":
        return _apply_jupiter_filter(
            sig,
            cutoff_profile=cutoff_profile,
            resonance_q=fp.resonance_q,
            sample_rate=sample_rate,
            filter_mode=fp.filter_mode,
            filter_drive=fp.filter_drive,
            feedback_amount=fp.feedback_amount,
            feedback_saturation=fp.feedback_saturation,
            filter_morph=fp.filter_morph,
            solver=fp.filter_solver,
            max_newton_iters=fp.max_newton_iters,
            newton_tolerance=fp.newton_tolerance,
        )
    if topology == "k35":
        return _apply_k35(
            sig,
            cutoff_profile=cutoff_profile,
            resonance_q=fp.resonance_q,
            sample_rate=sample_rate,
            filter_mode=fp.filter_mode,
            filter_drive=fp.filter_drive,
            k35_feedback_asymmetry=fp.k35_feedback_asymmetry,
            feedback_amount=fp.feedback_amount,
            feedback_saturation=fp.feedback_saturation,
        )
    if topology == "diode":
        return _apply_diode_filter(
            sig,
            cutoff_profile=cutoff_profile,
            resonance_q=fp.resonance_q,
            sample_rate=sample_rate,
            filter_mode=fp.filter_mode,
            filter_drive=fp.filter_drive,
            feedback_amount=fp.feedback_amount,
            feedback_saturation=fp.feedback_saturation,
            filter_morph=fp.filter_morph,
            solver=fp.filter_solver,
            max_newton_iters=fp.max_newton_iters,
            newton_tolerance=fp.newton_tolerance,
        )
    raise ValueError(f"Unknown filter_topology: {topology!r}")


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
