"""Karplus-Strong++ plucked-string primitive.

Extensions over vanilla KS: fractional delay line (pitch accurate above
~a few hundred Hz), pick-position comb pre-filter on the excitation burst,
1-pole loop lowpass (spectral darkening), and a tanh soft-clip in the
feedback path (warm/crunchy drive + numerical safety for ``sustain=1``).

Public entry point: ``render_pluck``, used by ``synth_voice`` (``osc_type=
"pluck"``) and ``drum_voice`` (``tone_type="pluck"``).
"""

from __future__ import annotations

import math

import numba
import numpy as np

_MIN_LOOP_GAIN = 0.998
_MAX_LOOP_GAIN = 1.0
_MIN_DELAY_SAMPLES = 2.0


@numba.njit(cache=True)
def _ks_plus_plus_loop(
    out: np.ndarray,
    buffer: np.ndarray,
    excitation: np.ndarray,
    delay_samples: np.ndarray,
    loop_gain: float,
    lp_coeff: float,
    drive: float,
) -> None:
    """Inner sample-accurate Karplus-Strong++ delay-line loop.

    Args:
        out: Output buffer (length N).
        buffer: Circular delay-line buffer (length M >= max(delay_samples)+2).
            Pre-seeded with the excitation burst so the initial period rings
            through the filter naturally — we still add the full excitation
            into the feed path for a clean attack transient.
        excitation: Pre-filtered input excitation (length N).
        delay_samples: Per-sample fractional delay length (length N).  Allows
            the pitch to track a time-varying ``freq_profile``.
        loop_gain: Loop-gain scalar in [_MIN_LOOP_GAIN, _MAX_LOOP_GAIN].
        lp_coeff: Loop lowpass coefficient (exponential smoother).  0 = no
            filtering, 1 = maximum damping.
        drive: Nonlinear termination drive.  0 = linear path, larger values
            apply a tanh soft-clip in the feedback path for warm → crunchy
            character (and keep ``sustain=1`` numerically bounded).
    """
    n_samples = out.shape[0]
    buf_len = buffer.shape[0]

    write_idx = 0
    lp_state = 0.0

    for i in range(n_samples):
        delay = delay_samples[i]
        if delay < _MIN_DELAY_SAMPLES:
            delay = _MIN_DELAY_SAMPLES
        if delay > buf_len - 2.0:
            delay = buf_len - 2.0

        # Fractional read index (1-sample linear interpolation).
        read_pos = float(write_idx) - delay
        while read_pos < 0.0:
            read_pos += buf_len
        read_idx0 = int(read_pos)
        frac = read_pos - float(read_idx0)
        read_idx1 = read_idx0 + 1
        if read_idx1 >= buf_len:
            read_idx1 -= buf_len
        delayed = (1.0 - frac) * buffer[read_idx0] + frac * buffer[read_idx1]

        lp_state = (1.0 - lp_coeff) * delayed + lp_coeff * lp_state
        feedback = loop_gain * lp_state

        # Nonlinear termination (drive=0 → linear KS, drive=1 → full tanh).
        if drive > 0.0:
            saturated = math.tanh(feedback * (1.0 + 3.0 * drive))
            feedback = (1.0 - drive) * feedback + drive * saturated

        new_sample = excitation[i] + feedback

        buffer[write_idx] = new_sample
        out[i] = new_sample

        write_idx += 1
        if write_idx >= buf_len:
            write_idx = 0


@numba.njit(cache=True)
def _shape_excitation_burst(
    noise: np.ndarray,
    hardness: float,
    pick_offset: float,
) -> np.ndarray:
    """Apply the lowpass-softening + comb-cancellation shape to a raw noise burst.

    ``hardness=0`` applies a heavy 1-pole lowpass (soft mallet); ``hardness=1``
    leaves the noise untouched (sharp pick).  ``pick_offset`` in samples is the
    fractional delay of the comb cancellation stage — simulates plucking at
    that fraction along the string length.  A ``pick_offset < 1`` skips the
    comb stage entirely.
    """
    burst_len = noise.shape[0]

    lp_coeff = 1.0 - hardness
    if lp_coeff > 0.0:
        alpha = lp_coeff * 0.9
        state = 0.0
        softened = np.empty(burst_len, dtype=np.float64)
        for i in range(burst_len):
            state = (1.0 - alpha) * noise[i] + alpha * state
            softened[i] = state
        noise = softened

    if pick_offset >= 1.0:
        off_int = int(pick_offset)
        off_frac = pick_offset - float(off_int)
        combed = noise.copy()
        for i in range(off_int + 1, burst_len):
            delayed = (1.0 - off_frac) * noise[i - off_int] + off_frac * noise[
                i - off_int - 1
            ]
            combed[i] = noise[i] - delayed
        noise = combed

    peak = 0.0
    for i in range(burst_len):
        mag = abs(noise[i])
        if mag > peak:
            peak = mag
    if peak > 0.0:
        inv_peak = 1.0 / peak
        for i in range(burst_len):
            noise[i] *= inv_peak
    return noise


def _excitation_burst(
    *,
    delay_samples: float,
    hardness: float,
    position: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a length-L excitation burst ready to seed the delay line.

    - ``hardness=0`` → noise passed through a 1-pole lowpass (soft mallet).
    - ``hardness=1`` → raw white noise (sharp pluck / crisp pick).
    - ``position`` → comb-cancel the ``1/position``-th partial, simulating
      pluck location along the string.  ``0.5`` is string centre (strong
      even-partial cancellation), ``0.1`` is near-bridge (bright, minor
      cancellation).
    """
    burst_len = max(2, int(round(delay_samples)))
    noise = rng.standard_normal(burst_len).astype(np.float64)
    hardness_c = float(np.clip(hardness, 0.0, 1.0))
    position_c = float(np.clip(position, 0.01, 0.99))
    pick_offset = position_c * delay_samples
    return _shape_excitation_burst(noise, hardness_c, pick_offset)


def render_pluck(
    *,
    freq: float,
    duration: float,
    sample_rate: int,
    hardness: float,
    damping: float,
    position: float,
    sustain: float,
    drive: float,
    seed: int,
    freq_profile: np.ndarray | None = None,
) -> np.ndarray:
    """Render a Karplus-Strong++ pluck.

    Args:
        freq: Fundamental frequency in Hz (positive).  When
            ``freq_profile`` is ``None`` the loop delay is held constant
            at ``sample_rate / freq``.
        duration: Note duration in seconds (positive).
        sample_rate: Sample rate in Hz (positive).
        hardness: Excitation brightness in [0, 1].  0 = soft mallet, 1 =
            sharp pick.
        damping: Loop damping in [0, 1].  0 = minimal damping (bright,
            long decay), 1 = heavy damping (dark, short decay).
        position: Pick position along the string in [0, 1].  0.5 = middle,
            0.1 = near bridge = brighter, 0.9 = near nut.
        sustain: Loop gain control in [0, 1].  0 = classic KS natural
            decay, 1 = infinite sustain (bounded by nonlinear termination).
        drive: Nonlinear termination drive in [0, 1].  0 = linear, higher
            = tanh saturation (warm → crunchy).
        seed: RNG seed (integer).  Same seed + params → bit-identical output.
        freq_profile: Optional per-sample fundamental frequency in Hz.
            Length must equal ``int(duration * sample_rate)``.  Values must
            be strictly positive and below the fractional-delay floor.

    Returns:
        Mono ``float64`` array of length ``int(duration * sample_rate)``,
        peak-normalised to ~1.0.
    """
    if freq <= 0.0:
        raise ValueError(f"freq must be positive, got {freq}")
    if duration <= 0.0:
        raise ValueError(f"duration must be positive, got {duration}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    n_samples = int(duration * sample_rate)
    if n_samples == 0:
        return np.zeros(0, dtype=np.float64)

    # The delay line must be at least a few samples — KS cannot produce a
    # meaningful pitch if the loop is shorter than the lowpass impulse
    # response.  Reject anything above ~sample_rate/2/_MIN_DELAY_SAMPLES as
    # silly (equivalent to asking for > Nyquist-scaled tones).
    max_freq = sample_rate / _MIN_DELAY_SAMPLES / 2.0
    if freq > max_freq:
        raise ValueError(
            f"freq {freq} Hz exceeds KS pluck upper bound "
            f"({max_freq:.1f} Hz at sample_rate={sample_rate})"
        )

    hardness = float(np.clip(hardness, 0.0, 1.0))
    damping = float(np.clip(damping, 0.0, 1.0))
    position = float(np.clip(position, 0.0, 1.0))
    sustain = float(np.clip(sustain, 0.0, 1.0))
    drive = float(np.clip(drive, 0.0, 1.0))

    # Resolve per-sample delay profile.
    if freq_profile is None:
        scalar_delay = float(sample_rate) / float(freq)
        delay_samples = np.full(n_samples, scalar_delay, dtype=np.float64)
    else:
        resolved = np.asarray(freq_profile, dtype=np.float64)
        if resolved.ndim != 1:
            raise ValueError("freq_profile must be 1-D")
        if resolved.shape[0] != n_samples:
            raise ValueError(
                f"freq_profile length {resolved.shape[0]} != n_samples {n_samples}"
            )
        if np.any(resolved <= 0.0):
            raise ValueError("freq_profile values must be strictly positive")
        delay_samples = float(sample_rate) / resolved

    # Size the delay buffer to hold max delay + fractional headroom.
    base_delay = float(sample_rate) / float(freq)
    max_delay = float(np.max(delay_samples)) if freq_profile is not None else base_delay
    buf_len = int(max(math.ceil(max_delay) + 4, 8))

    rng = np.random.default_rng(seed)
    burst = _excitation_burst(
        delay_samples=base_delay,
        hardness=hardness,
        position=position,
        rng=rng,
    )

    # Seed the delay buffer with the excitation burst so the first loop pass
    # carries it.  Zero-pad the rest.
    buffer = np.zeros(buf_len, dtype=np.float64)
    burst_into_buf = min(burst.shape[0], buf_len)
    buffer[:burst_into_buf] = burst[:burst_into_buf]

    # Place the excitation in the feed stream too, so the first period is
    # crisp.  Beyond the burst, the feed is silent and the delay line
    # recirculates on its own.
    excitation = np.zeros(n_samples, dtype=np.float64)
    feed_len = min(burst.shape[0], n_samples)
    excitation[:feed_len] = burst[:feed_len]

    loop_gain = _MIN_LOOP_GAIN + (_MAX_LOOP_GAIN - _MIN_LOOP_GAIN) * sustain
    # lp_coeff=0 → no smoothing; =~0.95 → heavy damping.  Map damping knob
    # quadratically so small values feel subtle and the upper half really
    # darkens things.
    lp_coeff = 0.95 * (damping**2)

    out = np.zeros(n_samples, dtype=np.float64)
    _ks_plus_plus_loop(
        out,
        buffer,
        excitation,
        delay_samples,
        loop_gain,
        lp_coeff,
        drive,
    )

    peak = float(np.max(np.abs(out)))
    if peak > 1e-12:
        out = out / peak
    return out
