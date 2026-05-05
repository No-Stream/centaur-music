"""Sustained excitation primitives for the ``drum_voice`` engine.

Three renderers — ``render_bow`` (stick-slip friction), ``render_blow``
(reed-table), and ``render_rub`` (filtered noise + stiction events) — that
produce broadband energy for the full note duration.  Pair with the modal
resonator bank (``tone_type="modal"`` / ``metallic_type="modal_bank"``) for
bowed bells, blown reeds, rubbed skins.

Stability is first-class: a bowed note can last 5-10 seconds, so every
inner loop uses bounded nonlinearities (``tanh``) to guarantee no runaway.
"""

from __future__ import annotations

import math

import numba
import numpy as np

from code_musics.engines._dsp_utils import bandpass_noise

# ---------------------------------------------------------------------------
# Bow — stick-slip friction excitation
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _bow_friction_loop(
    out: np.ndarray,
    noise_source: np.ndarray,
    sample_rate: int,
    freq: float,
    pressure: float,
    speed: float,
    position: float,
    noise_amount: float,
) -> None:
    """Stick-slip friction table driven by a bow-speed target.

    Relative bow-body velocity runs through a two-slope friction curve —
    steep ``tanh`` stick near zero, gentler slip further out.  A one-pole
    lowpass on body velocity gives the loop memory near the note
    fundamental so the excitation carries harmonic structure around
    ``freq`` rather than being flat noise.  ``tanh`` on both the curve
    and the output keeps the loop bounded over arbitrary durations.
    """
    n_samples = out.shape[0]
    if n_samples == 0:
        return

    cutoff_hz = max(freq * (0.5 + position * 1.5), 50.0)
    alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz / sample_rate)
    speed_target = 0.25 + 0.35 * speed
    stick_gain = 4.0 + 6.0 * pressure
    slip_gain = 0.4 + 0.4 * (1.0 - pressure)

    body_vel = 0.0
    for i in range(n_samples):
        bow_vel = speed_target + noise_amount * 0.12 * noise_source[i]
        v_rel = bow_vel - body_vel
        stick_force = math.tanh(stick_gain * v_rel)
        slip_force = slip_gain * math.tanh(0.5 * v_rel)
        friction = 0.7 * stick_force + 0.3 * slip_force
        body_vel = body_vel + alpha * (friction - body_vel)
        out[i] = math.tanh(1.1 * friction)


def render_bow(
    *,
    freq: float,
    duration: float,
    sample_rate: int,
    pressure: float,
    speed: float,
    position: float,
    noise_amount: float,
    seed: int,
) -> np.ndarray:
    """Render a sustained bow excitation.

    Args:
        freq: Note fundamental in Hz (used to center the body's spectral
            response — the modal bank handles the actual pitched resonance).
        duration: Length in seconds.
        sample_rate: Audio sample rate.
        pressure: Bow pressure (0..1).  Higher values produce a steeper stick
            region and a grittier, more "forced" excitation.
        speed: Bow speed (0..1).  Higher speeds bias the friction table
            toward the slip region, giving a brighter, less staccato attack.
        position: Bow position along the string (0..1).  Shapes the spectral
            center of the output — higher values push the spectrum up.
        noise_amount: Rosin / hair noise content (0..1).  Adds broadband
            fizz on top of the stick-slip oscillation.
        seed: RNG seed for deterministic output.

    Returns:
        Mono ``float64`` array of length ``int(sample_rate * duration)``.
    """
    if duration <= 0.0:
        raise ValueError(f"duration must be positive, got {duration}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if freq <= 0.0:
        raise ValueError(f"freq must be positive, got {freq}")

    pressure = float(max(0.0, min(1.0, pressure)))
    speed = float(max(0.0, min(1.0, speed)))
    position = float(max(0.0, min(1.0, position)))
    noise_amount = float(max(0.0, min(1.0, noise_amount)))

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0, dtype=np.float64)

    rng = np.random.default_rng(int(seed))
    noise_source = rng.standard_normal(n_samples)

    out = np.empty(n_samples, dtype=np.float64)
    _bow_friction_loop(
        out,
        noise_source,
        sample_rate,
        float(freq),
        pressure,
        speed,
        position,
        noise_amount,
    )

    # Bandpass-biased noise layer around ``freq * (1 + position)`` so the
    # excitation has spectral energy where the modal bank can latch onto it
    # via coupling. Mixed in under ``noise_amount``.
    if noise_amount > 0.0:
        center_hz = float(freq) * (1.0 + 1.5 * position)
        rosin = bandpass_noise(
            rng.standard_normal(n_samples),
            sample_rate=sample_rate,
            center_hz=center_hz,
            width_ratio=0.9,
        )
        rosin_peak = float(np.max(np.abs(rosin)))
        if rosin_peak > 1e-12:
            rosin = rosin / rosin_peak
        out = out + 0.35 * noise_amount * rosin

    peak = float(np.max(np.abs(out)))
    if peak > 1e-12:
        out = out / peak

    return out * pressure


# ---------------------------------------------------------------------------
# Blow — reed-table nonlinearity
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _blow_reed_loop(
    out: np.ndarray,
    breath_noise_source: np.ndarray,
    pressure_profile: np.ndarray,
    sample_rate: int,
    freq: float,
    embouchure: float,
    breath_noise: float,
) -> None:
    """Reed-table nonlinearity driving a small resonant delay line.

    Waveguide clarinet/sax-style reed lookup: the reed opens when breath
    pressure minus delay-line feedback exceeds the embouchure threshold;
    flow saturates quickly through a tanh-smoothed soft-open.  Output is
    the pre-reed excitation — pitch comes from the modal bank downstream.
    Feedback gain stays at 0.9 (loop unconditionally stable) with a tanh
    on the delay-line write as an additional safety net.
    """
    n_samples = out.shape[0]
    if n_samples == 0:
        return

    delay_samples = int(max(4.0, sample_rate / max(freq, 20.0)))
    if delay_samples >= n_samples:
        delay_samples = max(4, n_samples - 1)
    buffer = np.zeros(delay_samples, dtype=np.float64)
    write_idx = 0

    threshold = 0.15 + 0.55 * embouchure
    feedback = 0.9
    reed_stiffness = 3.0 + 8.0 * embouchure

    for i in range(n_samples):
        back_pressure = buffer[write_idx]
        breath = pressure_profile[i] + breath_noise * 0.15 * breath_noise_source[i]
        dp = breath - feedback * back_pressure
        open_amount = 0.5 * (math.tanh(reed_stiffness * (dp - threshold)) + 1.0)
        flow = open_amount * dp
        buffer[write_idx] = math.tanh(flow + feedback * back_pressure)
        out[i] = flow
        write_idx += 1
        if write_idx >= delay_samples:
            write_idx = 0


def render_blow(
    *,
    freq: float,
    duration: float,
    sample_rate: int,
    pressure: float,
    embouchure: float,
    breath_noise: float,
    wobble_rate_hz: float,
    wobble_depth: float,
    seed: int,
) -> np.ndarray:
    """Render a sustained blown-reed excitation.

    Args:
        freq: Note fundamental in Hz (sets the internal waveguide delay
            length; final pitch still comes from the modal bank).
        duration: Length in seconds.
        sample_rate: Audio sample rate.
        pressure: Breath pressure (0..1).
        embouchure: Reed stiffness / threshold (0..1).  Higher = tighter
            lips, later onset, more overtone content.
        breath_noise: Airy fizz content (0..1).
        wobble_rate_hz: Slow pressure LFO rate in Hz (for organic
            breathing).  Pass 0 to disable.
        wobble_depth: LFO depth as a fraction of pressure (0..1).
        seed: RNG seed for deterministic output.

    Returns:
        Mono ``float64`` array of length ``int(sample_rate * duration)``.
    """
    if duration <= 0.0:
        raise ValueError(f"duration must be positive, got {duration}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if freq <= 0.0:
        raise ValueError(f"freq must be positive, got {freq}")

    pressure = float(max(0.0, min(1.0, pressure)))
    embouchure = float(max(0.0, min(1.0, embouchure)))
    breath_noise = float(max(0.0, min(1.0, breath_noise)))
    wobble_rate_hz = float(max(0.0, wobble_rate_hz))
    wobble_depth = float(max(0.0, min(1.0, wobble_depth)))

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0, dtype=np.float64)

    # Pressure profile: constant plus a slow LFO for breathing motion.
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    base_pressure = 0.5 + 0.5 * pressure  # 0.5..1.0 usable range
    if wobble_rate_hz > 0.0 and wobble_depth > 0.0:
        lfo = np.sin(2.0 * math.pi * wobble_rate_hz * t)
        pressure_profile = base_pressure * (1.0 + wobble_depth * 0.25 * lfo)
    else:
        pressure_profile = np.full(n_samples, base_pressure, dtype=np.float64)

    rng = np.random.default_rng(int(seed))
    breath_noise_source = rng.standard_normal(n_samples)

    out = np.empty(n_samples, dtype=np.float64)
    _blow_reed_loop(
        out,
        breath_noise_source,
        pressure_profile,
        sample_rate,
        float(freq),
        embouchure,
        breath_noise,
    )

    # Add a bandpassed breath layer so the excitation has airy broadband
    # content too (mostly for driving dispersed / high-mode banks).
    if breath_noise > 0.0:
        breath_center = float(freq) * 3.0
        breath = bandpass_noise(
            rng.standard_normal(n_samples),
            sample_rate=sample_rate,
            center_hz=breath_center,
            width_ratio=1.2,
        )
        breath_peak = float(np.max(np.abs(breath)))
        if breath_peak > 1e-12:
            breath = breath / breath_peak
        out = out + 0.25 * breath_noise * breath

    peak = float(np.max(np.abs(out)))
    if peak > 1e-12:
        out = out / peak

    return out * pressure


# ---------------------------------------------------------------------------
# Rub — filtered noise with pressure + stiction
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _rub_stiction_loop(
    out: np.ndarray,
    noise_source: np.ndarray,
    sample_rate: int,
    speed: float,
    stiction: float,
) -> None:
    """Shape a pre-filtered noise source with periodic micro-grabs.

    At low ``speed`` values the output is modulated by a series of sharp
    stiction events (finger sticking then releasing against a surface).
    At higher speeds the events merge into a continuous rub.
    """
    n_samples = noise_source.shape[0]
    if n_samples == 0:
        return

    # speed=0 → ~4 Hz events (~250 ms grabs); speed=1 → ~84 Hz (continuous).
    event_rate_hz = 4.0 + 80.0 * speed
    event_period_samples = max(4, int(sample_rate / event_rate_hz))

    event_depth = 0.9 * stiction
    decay_samples = max(2.0, event_period_samples * 0.5)
    decay_coeff = math.exp(-1.0 / decay_samples)

    event_env = 1.0
    next_event_in = 0
    for i in range(n_samples):
        if next_event_in <= 0:
            event_env = 1.0 - event_depth
            next_event_in = event_period_samples
        else:
            event_env = event_env + (1.0 - event_env) * (1.0 - decay_coeff)

        out[i] = noise_source[i] * event_env
        next_event_in -= 1


def render_rub(
    *,
    freq: float,
    duration: float,
    sample_rate: int,
    pressure: float,
    speed: float,
    roughness: float,
    stiction: float,
    seed: int,
) -> np.ndarray:
    """Render a sustained rub excitation (finger on glass / wet hand on skin).

    Args:
        freq: Note fundamental in Hz (used as the rub's spectral center).
        duration: Length in seconds.
        sample_rate: Audio sample rate.
        pressure: Overall amplitude (0..1).
        speed: Rub speed (0..1).  Modulates filter center motion and the
            rate of stiction events (slow = sparse grabs, fast = smooth rub).
        roughness: Spectral bandwidth (0..1).  Wider at higher values for a
            more "abrasive" timbre.
        stiction: Intensity of periodic micro-grabs at low speed (0..1).
            ``0`` gives a smooth rub; ``1`` gives strong periodic attenuation
            events (like a finger catching on a surface).
        seed: RNG seed for deterministic output.

    Returns:
        Mono ``float64`` array of length ``int(sample_rate * duration)``.
    """
    if duration <= 0.0:
        raise ValueError(f"duration must be positive, got {duration}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if freq <= 0.0:
        raise ValueError(f"freq must be positive, got {freq}")

    pressure = float(max(0.0, min(1.0, pressure)))
    speed = float(max(0.0, min(1.0, speed)))
    roughness = float(max(0.0, min(1.0, roughness)))
    stiction = float(max(0.0, min(1.0, stiction)))

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0, dtype=np.float64)

    rng = np.random.default_rng(int(seed))

    # Bandpassed broadband noise: center modulated up with speed, width
    # opened up with roughness. Single FFT-domain pass per call (cheap).
    noise = rng.standard_normal(n_samples)
    center_hz = float(freq) * (1.0 + 2.0 * speed)
    width_ratio = 0.4 + 1.6 * roughness
    shaped = bandpass_noise(
        noise,
        sample_rate=sample_rate,
        center_hz=center_hz,
        width_ratio=width_ratio,
    )
    shaped_peak = float(np.max(np.abs(shaped)))
    if shaped_peak > 1e-12:
        shaped = shaped / shaped_peak

    out = np.empty(n_samples, dtype=np.float64)
    _rub_stiction_loop(out, shaped, sample_rate, speed, stiction)

    peak = float(np.max(np.abs(out)))
    if peak > 1e-12:
        out = out / peak

    return out * pressure
