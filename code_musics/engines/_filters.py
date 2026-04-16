"""Shared DSP filter primitives used by multiple synth engines."""

from __future__ import annotations

import math

import numba
import numpy as np

_SUPPORTED_FILTER_MODES = {"lowpass", "bandpass", "highpass", "notch"}
_SUPPORTED_FILTER_TOPOLOGIES = {"svf", "ladder"}

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

        feedback_sum = low_state + (2.0 * effective_damping + g) * band_state
        sat_input = drive_gain * sample - feedback_sum

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

    linear_filtered = _apply_linear_zdf_svf(
        sig,
        cutoff,
        damping,
        sample_rate,
        mode_int,
        precomputed_g,
        nyquist_limit,
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

    linear_filtered = _apply_linear_zdf_svf(
        sig,
        cutoff,
        damping,
        sample_rate,
        mode_int,
        precomputed_g,
        nyquist_limit,
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
) -> np.ndarray:
    """Apply a Moog-style 4-pole ladder filter."""
    q = max(0.5, float(resonance_q))

    k = 4.0 * (1.0 - 1.0 / (2.0 * q))
    k = max(0.0, min(3.98, k))

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

    return _apply_ladder_filter_inner(
        sig,
        cutoff,
        k,
        sample_rate,
        mode_int,
        max(0.0, float(filter_drive)),
        max(0.0, float(bass_compensation)),
        even_h,
        precomputed_g,
        nyquist_limit,
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
) -> np.ndarray:
    """Unified filter entry point that dispatches to SVF or ladder topology.

    When ``hpf_cutoff_hz > 0``, a serial 2-pole ZDF SVF highpass is applied
    before the main filter stage.

    When ``feedback_amount > 0``, the post-filter output feeds back to the
    pre-filter input through a saturating tanh stage (Minimoog-style mixer
    feedback).  ``feedback_saturation`` controls the drive on the feedback
    path tanh (default 0.3).
    """
    if filter_topology not in _SUPPORTED_FILTER_TOPOLOGIES:
        raise ValueError(f"Unknown filter_topology: {filter_topology!r}")

    sig = np.asarray(signal, dtype=np.float64)

    if hpf_cutoff_hz > 0.0:
        nyquist_limit = sample_rate * _NYQUIST_CLAMP_RATIO
        hpf_cutoff = min(float(hpf_cutoff_hz), nyquist_limit)
        hpf_g = math.tan(math.pi * hpf_cutoff / sample_rate)
        hpf_cutoff_profile = np.full(sig.shape[0], hpf_cutoff, dtype=np.float64)
        hpf_damping = 1.0 / max(0.5, float(hpf_resonance_q))
        sig = _apply_linear_zdf_svf(
            sig,
            hpf_cutoff_profile,
            hpf_damping,
            sample_rate,
            _HP,
            hpf_g,
            nyquist_limit,
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
    )
