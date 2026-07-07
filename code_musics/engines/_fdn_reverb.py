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
- **Delay lines.** ``N`` mutually-prime delay lengths (a fixed prime table
  scaled by ``size``) give a dense, non-degenerate mode distribution so the
  tail is smooth rather than fluttery.
- **Per-line decay.** Each line's reference loop gain follows the Jot formula
  ``g_i = 10^(-3 D_i / (fs · RT60))`` so every line — regardless of length —
  decays 60 dB in ``decay_s``. A one-pole in-loop lowpass (``damping_hz``)
  makes highs decay faster (dark tail); a separate bass band gets its own gain
  (``low_decay_mult``) so the low end can ring longer than the mids.
- **Modulation.** Each line's read pointer is modulated by its own slow,
  shallow, decorrelated sine (rates detuned around ``modulation_rate_hz``,
  random seeded phases). This continuously sweeps the modal notches to kill
  metallic ringing without audible chorusing — depth stays a few samples.
- **Input diffusion.** A cascade of Schroeder allpasses (``diffusion``) smears
  the input so the onset is a diffuse wash rather than discrete echoes.
- **Stereo taps.** Left and right are read through two *orthogonal* Hadamard
  tap vectors, giving a strongly decorrelated stereo field from a mono input.

Everything is seeded/deterministic: identical inputs and params render
bit-identically.
"""

from __future__ import annotations

import numba
import numpy as np

# Mutually-prime base delay lengths in samples at the nominal (size ≈ 0.5)
# scale, geometrically spread ~23 ms .. 104 ms at 44.1 kHz. Distinct primes are
# mutually coprime, which spreads the FDN modes and avoids degenerate ringing.
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


def _hadamard(order: int) -> np.ndarray:
    """Return a normalized (unit-magnitude rows) Sylvester Hadamard matrix."""
    if order & (order - 1) != 0:
        raise ValueError(f"Hadamard order must be a power of two, got {order}")
    matrix = np.array([[1.0]], dtype=np.float64)
    while matrix.shape[0] < order:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / np.sqrt(order)


@numba.njit(cache=True)
def _fdn_kernel(
    mono_in: np.ndarray,
    delays: np.ndarray,
    g_mid: np.ndarray,
    g_bass: np.ndarray,
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
        diffused_input = x

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

            # HF damping (one-pole LP) + separate bass band for low_decay_mult
            damp_state[line] += damp_a * (s - damp_state[line])
            bass_state[line] += bass_a * (s - bass_state[line])
            fb[line] = (
                g_mid[line] * damp_state[line]
                + (g_bass[line] - g_mid[line]) * bass_state[line]
            )

        # --- stereo output taps (orthogonal Hadamard projections) ---
        acc_l = 0.0
        acc_r = 0.0
        for line in range(n_lines):
            acc_l += tap_l[line] * tap[line]
            acc_r += tap_r[line] * tap[line]
        wet_l[sample_index] = acc_l * inv_sqrt_n
        wet_r[sample_index] = acc_r * inv_sqrt_n

        # --- feedback mixing + write new line inputs ---
        write_pos = sample_index % line_len
        if use_hadamard:
            for line in range(n_lines):
                mixed = 0.0
                for col in range(n_lines):
                    mixed += hadamard[line, col] * fb[col]
                value = diffused_input + mixed
                if -1.0e-25 < value < 1.0e-25:
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
                if -1.0e-25 < value < 1.0e-25:
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
    if not 0.0 <= diffusion <= 1.0:
        raise ValueError("diffusion must be in [0, 1]")
    if n_lines not in (8, 16):
        raise ValueError("n_lines must be 8 or 16")
    if feedback_matrix not in ("householder", "hadamard"):
        raise ValueError("feedback_matrix must be 'householder' or 'hadamard'")

    mono = np.ascontiguousarray(mono_input, dtype=np.float64)

    # --- delay lengths: scale the prime table by size ---
    size_scale = 0.4 + 1.3 * size
    base = _BASE_PRIME_DELAYS[:n_lines].astype(np.float64) * size_scale
    delays = np.maximum(2, np.round(base)).astype(np.int64)

    # --- per-line reference loop gains (Jot RT60 formula) ---
    fs = float(sample_rate)
    exponent_mid = -3.0 * delays.astype(np.float64) / (fs * decay_s)
    g_mid = np.power(10.0, exponent_mid)
    exponent_bass = -3.0 * delays.astype(np.float64) / (fs * decay_s * low_decay_mult)
    g_bass = np.power(10.0, exponent_bass)
    # Guarantee stability with an unconditional headroom clamp.
    g_mid = np.clip(g_mid, 0.0, 0.9995)
    g_bass = np.clip(g_bass, 0.0, 0.9995)

    # --- in-loop filter coefficients ---
    damp_a = 1.0 - np.exp(-2.0 * np.pi * damping_hz / fs)
    damp_a = float(np.clip(damp_a, 0.0, 1.0))
    bass_a = 1.0 - np.exp(-2.0 * np.pi * low_crossover_hz / fs)
    bass_a = float(np.clip(bass_a, 0.0, 1.0))

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
        np.ascontiguousarray(g_mid),
        np.ascontiguousarray(g_bass),
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
        line_buf,
        line_len,
        diff_buf,
        diff_len,
    )
    return np.stack([wet_l, wet_r]).astype(np.float64)
