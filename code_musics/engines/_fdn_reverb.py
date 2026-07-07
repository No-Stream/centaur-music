"""Feedback Delay Network (FDN) reverb core.

A native, deterministic FDN reverb designed for enormous, unworldly, dark, but
*clean* spaces — a cave well beyond architectural scale — whose tails stay
smooth and chorus-free even at 30-60 s decays.

Design summary
--------------
- **Feedback matrix.** A unitary (energy-preserving) mixing matrix keeps the
  late field lossless so the only decay comes from the per-line loop gains.
  Two matrices are supported: a maximally-diffusive ``householder`` reflection
  ``A = I - (2/N) 11ᵀ`` (O(N), the default) and a normalized Sylvester
  ``hadamard`` matrix ``H/√N`` (O(N²), power-of-two ``N`` only). Both are
  orthogonal, which is what keeps huge tails from either blowing up or
  collapsing.
- **Delay lines.** ``N`` prime delay lengths — a base prime table scaled by
  ``size`` and then *snapped back to distinct primes* so the lengths stay
  pairwise coprime at every ``size``. Distinct primes give a dense,
  non-degenerate mode distribution so the tail is smooth rather than fluttery.
- **Per-line decay (Jot, reference-band normalized).** Each line carries an
  in-loop first-order absorbent filter whose magnitude is calibrated so that
  the loop gain *at the 1 kHz reference band* is exactly the Jot gain
  ``g_i = 10^(-3 D_i / (fs · decay_s))`` — i.e. ``decay_s`` is the reference
  (1 kHz) RT60, not the DC RT60. ``damping_hz`` sets the corner above which the
  tail decays faster (dark tail); the bass band gets its own longer RT60
  (``decay_s · low_decay_mult``). The absorbent filter is realized as a blend of
  two one-poles (a damping lowpass + a bass lowpass) whose two mixing
  coefficients are solved from the DC and reference targets. Because both
  targets are genuine Jot gains (each < 1), the design is inherently bounded;
  an unconditional per-line magnitude clamp over the full spectrum guarantees
  stability across the whole parameter space.
- **Modulation.** Each line's read pointer is modulated by its own slow,
  shallow, decorrelated sine (rates detuned around ``modulation_rate_hz``,
  random seeded phases). This continuously sweeps the modal notches to kill
  metallic ringing without audible chorusing — depth stays a few samples.
- **Input diffusion.** A cascade of Schroeder allpasses (``diffusion``) smears
  the input so the onset is a diffuse wash rather than discrete echoes.
- **Injection / stereo taps.** The diffused input is injected into the ``N``
  lines scaled by ``1/√N`` (energy-preserving), and left/right are read through
  two *orthogonal* Hadamard tap vectors, giving a strongly decorrelated stereo
  field from a mono input at a calibrated wet level.

Everything is seeded/deterministic: identical inputs and params render
bit-identically.
"""

from __future__ import annotations

import numba
import numpy as np

# Base prime delay lengths in samples at the nominal (size ≈ 0.5) scale,
# geometrically spread ~23 ms .. 104 ms at 44.1 kHz. They are re-snapped to
# distinct primes after size scaling (see ``_snap_to_distinct_primes``).
_BASE_PRIME_DELAYS = np.array(
    [
        1009,
        1151,
        1289,
        1453,
        1607,
        1811,
        1999,
        2203,
        2411,
        2647,
        2903,
        3187,
        3491,
        3833,
        4201,
        4591,
    ],
    dtype=np.int64,
)

# Short prime allpass delays (samples) for the input diffusion cascade.
_DIFFUSION_DELAYS = np.array([142, 107, 379, 277], dtype=np.int64)

# Cap on the per-line modulation depth (samples) at modulation_depth == 1.0.
# Kept small so the slow pointer wobble sweeps modal notches without the pitch
# smear that reads as chorus.
_MAX_MOD_SAMPLES = 16.0

# Reference band for RT60 calibration: ``decay_s`` is the 1 kHz RT60.
_REFERENCE_HZ = 1000.0

# Unconditional per-line loop-gain ceiling (magnitude over all frequencies).
# Below 1 with headroom so the unitary FDN cannot self-oscillate anywhere in
# the parameter space.
_LOOP_GAIN_CEILING = 0.9995

# Denormal flush threshold (shared by delay writes and filter states).
_DENORMAL = 1.0e-25

# Overall wet-return calibration. Chosen so that sustained pink noise at a long
# decay returns a wet level on the same order as the input (see docs). Applied
# after the ``1/√N`` energy-preserving injection.
_WET_CALIBRATION = 1.6

# Maximum allowed modulation rate (Hz). Above this the pointer wobble stops
# being a slow notch-sweep and starts audibly pitch-modulating the tail.
_MAX_MODULATION_RATE_HZ = 2.0

# Cached prime table for delay-length snapping. Covers the full scaled range:
# max base prime (4591) × max size scale (1.7) ≈ 7805, with headroom for the
# distinct-prime outward search.
_PRIME_TABLE_MAX = 12000


def _sieve_primes(limit: int) -> np.ndarray:
    """Return all primes ``< limit`` via a simple sieve."""
    sieve = np.ones(limit, dtype=bool)
    sieve[:2] = False
    for candidate in range(2, int(limit**0.5) + 1):
        if sieve[candidate]:
            sieve[candidate * candidate :: candidate] = False
    return np.flatnonzero(sieve).astype(np.int64)


_PRIMES = _sieve_primes(_PRIME_TABLE_MAX)


def _snap_to_distinct_primes(lengths: np.ndarray) -> np.ndarray:
    """Snap each length to the nearest prime, keeping all results distinct.

    Distinct primes are pairwise coprime, so the returned delay lengths have
    pairwise ``gcd == 1`` — the property size-scaling used to destroy.
    """
    used: set[int] = set()
    result: list[int] = []
    for target in lengths:
        order = np.argsort(np.abs(_PRIMES - float(target)))
        for candidate in _PRIMES[order]:
            value = int(candidate)
            if value not in used:
                used.add(value)
                result.append(value)
                break
    return np.array(result, dtype=np.int64)


def _hadamard(order: int) -> np.ndarray:
    """Return a normalized (unit-magnitude rows) Sylvester Hadamard matrix."""
    if order & (order - 1) != 0:
        raise ValueError(f"Hadamard order must be a power of two, got {order}")
    matrix = np.array([[1.0]], dtype=np.float64)
    while matrix.shape[0] < order:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / np.sqrt(order)


def _onepole_response(alpha: float, omega: np.ndarray) -> np.ndarray:
    """Complex frequency response of the one-pole ``y += alpha (x - y)``."""
    return alpha / (1.0 - (1.0 - alpha) * np.exp(-1j * omega))


def _absorbent_coefficients(
    delays: np.ndarray,
    *,
    fs: float,
    decay_s: float,
    low_decay_mult: float,
    damp_a: float,
    bass_a: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve per-line in-loop absorbent-filter coefficients (Jot, ref-normalized).

    The loop feedback for a line is ``c_damp · LP_damp(s) + c_bass · LP_bass(s)``
    where ``LP_damp`` is a one-pole lowpass at ``damping_hz`` and ``LP_bass`` a
    one-pole lowpass at ``low_crossover_hz``. The two coefficients are solved so
    that the loop magnitude equals:

    - ``G_dc = 10^(-3 D/(fs · decay_s · low_decay_mult))`` at DC (bass RT60), and
    - ``G_ref = 10^(-3 D/(fs · decay_s))`` at the 1 kHz reference (mid RT60).

    Both targets are genuine Jot gains (< 1), so the design is inherently
    bounded; a spectral magnitude clamp then guarantees the ``< 1`` loop
    condition everywhere.
    """
    omega_ref = 2.0 * np.pi * _REFERENCE_HZ / fs
    mag_damp_ref = float(np.abs(_onepole_response(damp_a, np.array([omega_ref]))[0]))
    mag_bass_ref = float(np.abs(_onepole_response(bass_a, np.array([omega_ref]))[0]))

    exponent = -3.0 * delays.astype(np.float64) / (fs * decay_s)
    g_ref = np.power(10.0, exponent)
    g_dc = np.power(10.0, exponent / low_decay_mult)

    determinant = mag_damp_ref - mag_bass_ref
    if determinant > 1.0e-3:
        c_damp = (g_ref - g_dc * mag_bass_ref) / determinant
        c_bass = g_dc - c_damp
    else:
        # Damping corner sits at/below the reference band: normalization is
        # degenerate. Fall back to a single damping term (the clamp below keeps
        # it bounded; the reference RT60 will simply be damping-limited).
        c_damp = g_ref / max(mag_damp_ref, 1.0e-6)
        c_bass = np.zeros_like(c_damp)

    # --- unconditional per-line spectral stability clamp ---
    grid_hz = np.geomspace(1.0, 0.499 * fs, 512)
    omega_grid = 2.0 * np.pi * grid_hz / fs
    resp_damp = _onepole_response(damp_a, omega_grid)
    resp_bass = _onepole_response(bass_a, omega_grid)
    loop_mag = np.abs(
        c_damp[:, None] * resp_damp[None, :] + c_bass[:, None] * resp_bass[None, :]
    )
    peak = loop_mag.max(axis=1)
    scale = np.where(peak > _LOOP_GAIN_CEILING, _LOOP_GAIN_CEILING / peak, 1.0)
    return c_damp * scale, c_bass * scale


@numba.njit(cache=True)
def _fdn_kernel(
    mono_in: np.ndarray,
    delays: np.ndarray,
    c_damp: np.ndarray,
    c_bass: np.ndarray,
    damp_a: float,
    bass_a: float,
    mod_depth: np.ndarray,
    mod_inc: np.ndarray,
    mod_phase: np.ndarray,
    tap_l: np.ndarray,
    tap_r: np.ndarray,
    hadamard: np.ndarray,
    use_hadamard: bool,
    diff_delays: np.ndarray,
    diff_g: float,
    wet_gain: float,
    line_buf: np.ndarray,
    line_len: int,
    diff_buf: np.ndarray,
    diff_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample FDN recursion. Returns decorrelated (wet_l, wet_r)."""
    n_samples = mono_in.shape[0]
    n_lines = delays.shape[0]
    n_diff = diff_delays.shape[0]

    wet_l = np.empty(n_samples, dtype=np.float64)
    wet_r = np.empty(n_samples, dtype=np.float64)

    damp_state = np.zeros(n_lines, dtype=np.float64)
    bass_state = np.zeros(n_lines, dtype=np.float64)
    fb = np.zeros(n_lines, dtype=np.float64)
    tap = np.zeros(n_lines, dtype=np.float64)
    phase = mod_phase.copy()

    inv_sqrt_n = 1.0 / np.sqrt(float(n_lines))

    for sample_index in range(n_samples):
        # --- input diffusion: Schroeder allpass cascade ---
        x = mono_in[sample_index]
        for stage in range(n_diff):
            stage_len = diff_delays[stage]
            read_pos = sample_index % diff_len
            # circular read at delay = stage_len
            base = sample_index - stage_len
            delayed = 0.0 if base < 0 else diff_buf[stage, base % diff_len]
            v = x + diff_g * delayed
            y = -diff_g * v + delayed
            diff_buf[stage, read_pos] = v
            x = y
        diffused_input = x * inv_sqrt_n

        # --- read the delay lines (modulated fractional read) ---
        for line in range(n_lines):
            read_delay = delays[line] + mod_depth[line] * np.sin(phase[line])
            read_float = sample_index - read_delay
            base = int(np.floor(read_float))
            frac = read_float - base
            if base < 0:
                s = 0.0
            else:
                idx0 = base % line_len
                idx1 = (base + 1) % line_len
                s = line_buf[line, idx0] * (1.0 - frac) + line_buf[line, idx1] * frac
            tap[line] = s

            # advance the LFO phase
            phase[line] += mod_inc[line]
            if phase[line] > 6.283185307179586:
                phase[line] -= 6.283185307179586

            # in-loop absorbent filter: damping LP + bass LP blend
            damp_state[line] += damp_a * (s - damp_state[line])
            if -_DENORMAL < damp_state[line] < _DENORMAL:
                damp_state[line] = 0.0
            bass_state[line] += bass_a * (s - bass_state[line])
            if -_DENORMAL < bass_state[line] < _DENORMAL:
                bass_state[line] = 0.0
            fb[line] = c_damp[line] * damp_state[line] + c_bass[line] * bass_state[line]

        # --- stereo output taps (orthogonal Hadamard projections) ---
        acc_l = 0.0
        acc_r = 0.0
        for line in range(n_lines):
            acc_l += tap_l[line] * tap[line]
            acc_r += tap_r[line] * tap[line]
        wet_l[sample_index] = acc_l * inv_sqrt_n * wet_gain
        wet_r[sample_index] = acc_r * inv_sqrt_n * wet_gain

        # --- feedback mixing + write new line inputs ---
        write_pos = sample_index % line_len
        if use_hadamard:
            for line in range(n_lines):
                mixed = 0.0
                for col in range(n_lines):
                    mixed += hadamard[line, col] * fb[col]
                value = diffused_input + mixed
                if -_DENORMAL < value < _DENORMAL:
                    value = 0.0
                line_buf[line, write_pos] = value
        else:
            fb_sum = 0.0
            for line in range(n_lines):
                fb_sum += fb[line]
            householder_scale = 2.0 / float(n_lines) * fb_sum
            for line in range(n_lines):
                mixed = fb[line] - householder_scale
                value = diffused_input + mixed
                if -_DENORMAL < value < _DENORMAL:
                    value = 0.0
                line_buf[line, write_pos] = value

    return wet_l, wet_r


def render_fdn_reverb(
    mono_input: np.ndarray,
    *,
    sample_rate: int,
    decay_s: float,
    size: float,
    damping_hz: float,
    low_decay_mult: float,
    low_crossover_hz: float,
    modulation_depth: float,
    modulation_rate_hz: float,
    diffusion: float,
    feedback_matrix: str,
    n_lines: int,
    seed: int,
) -> np.ndarray:
    """Render a mono input through the FDN, returning a stereo wet signal.

    Returns a ``(2, n_samples)`` array (no dry, no predelay, no tone shaping —
    those are handled by the ``synth.py`` wrapper).
    """
    if decay_s <= 0.0:
        raise ValueError("decay_s must be positive")
    if not 0.0 <= size <= 1.0:
        raise ValueError("size must be in [0, 1]")
    if damping_hz <= 0.0:
        raise ValueError("damping_hz must be positive")
    if low_decay_mult <= 0.0:
        raise ValueError("low_decay_mult must be positive")
    if not 0.0 <= modulation_depth <= 1.0:
        raise ValueError("modulation_depth must be in [0, 1]")
    if modulation_rate_hz < 0.0:
        raise ValueError("modulation_rate_hz must be non-negative")
    if modulation_rate_hz > _MAX_MODULATION_RATE_HZ:
        raise ValueError(
            f"modulation_rate_hz must be <= {_MAX_MODULATION_RATE_HZ} Hz "
            "(above this the tail audibly pitch-modulates)"
        )
    if not 0.0 <= diffusion <= 1.0:
        raise ValueError("diffusion must be in [0, 1]")
    if n_lines not in (8, 16):
        raise ValueError("n_lines must be 8 or 16")
    if feedback_matrix not in ("householder", "hadamard"):
        raise ValueError("feedback_matrix must be 'householder' or 'hadamard'")

    mono = np.ascontiguousarray(mono_input, dtype=np.float64)

    # --- delay lengths: scale the prime table by size, re-snap to primes ---
    size_scale = 0.4 + 1.3 * size
    scaled = _BASE_PRIME_DELAYS[:n_lines].astype(np.float64) * size_scale
    delays = _snap_to_distinct_primes(np.maximum(2.0, scaled))

    # --- in-loop filter coefficients ---
    fs = float(sample_rate)
    damp_a = 1.0 - np.exp(-2.0 * np.pi * damping_hz / fs)
    damp_a = float(np.clip(damp_a, 0.0, 1.0))
    bass_a = 1.0 - np.exp(-2.0 * np.pi * low_crossover_hz / fs)
    bass_a = float(np.clip(bass_a, 0.0, 1.0))

    # --- per-line absorbent coefficients (reference-normalized Jot) ---
    c_damp, c_bass = _absorbent_coefficients(
        delays,
        fs=fs,
        decay_s=decay_s,
        low_decay_mult=low_decay_mult,
        damp_a=damp_a,
        bass_a=bass_a,
    )

    # --- modulation: slow, shallow, decorrelated per line ---
    rng = np.random.default_rng(seed)
    depth_jitter = 0.6 + 0.4 * rng.random(n_lines)
    mod_depth = modulation_depth * _MAX_MOD_SAMPLES * depth_jitter
    rate_jitter = 0.7 + 0.6 * rng.random(n_lines)
    mod_rate = modulation_rate_hz * rate_jitter
    mod_inc = 2.0 * np.pi * mod_rate / fs
    mod_phase = 2.0 * np.pi * rng.random(n_lines)

    # --- matrices and orthogonal stereo taps ---
    hadamard = _hadamard(n_lines)
    # Two distinct non-constant Hadamard rows are orthogonal → decorrelated L/R.
    tap_l = hadamard[1].copy() * np.sqrt(float(n_lines))
    tap_r = hadamard[2].copy() * np.sqrt(float(n_lines))
    use_hadamard = feedback_matrix == "hadamard"

    # --- diffusion allpass gain ---
    diff_g = 0.7 * diffusion

    # --- buffers ---
    line_len = int(delays.max()) + int(np.ceil(mod_depth.max())) + 4
    diff_len = int(_DIFFUSION_DELAYS.max()) + 4
    line_buf = np.zeros((n_lines, line_len), dtype=np.float64)
    diff_buf = np.zeros((_DIFFUSION_DELAYS.shape[0], diff_len), dtype=np.float64)

    wet_l, wet_r = _fdn_kernel(
        mono,
        delays,
        np.ascontiguousarray(c_damp),
        np.ascontiguousarray(c_bass),
        damp_a,
        bass_a,
        np.ascontiguousarray(mod_depth),
        np.ascontiguousarray(mod_inc),
        np.ascontiguousarray(mod_phase),
        np.ascontiguousarray(tap_l),
        np.ascontiguousarray(tap_r),
        np.ascontiguousarray(hadamard),
        use_hadamard,
        _DIFFUSION_DELAYS,
        diff_g,
        _WET_CALIBRATION,
        line_buf,
        line_len,
        diff_buf,
        diff_len,
    )
    return np.stack([wet_l, wet_r]).astype(np.float64)
