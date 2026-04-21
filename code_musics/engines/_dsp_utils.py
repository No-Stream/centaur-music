"""Shared DSP utility functions for synth engines.

Extracted from piano.py, piano_additive.py, organ.py, and filtered_stack.py
to eliminate cross-engine duplication.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

import numba
import numpy as np
from scipy.signal import resample_poly

from code_musics.engines._filters import FilterParams, apply_filter
from code_musics.humanize import build_drift_bus  # re-exported for legacy callers

__all__ = ["build_drift_bus"]

logger: logging.Logger = logging.getLogger(__name__)


NYQUIST_FADE_START = 0.85
MAX_DRIFT_CENTS = 4.0
GOLDEN_RATIO_FRAC = 0.6180339887498949

# --- Analog character tuning constants ---
# Per-note jitter ranges (fraction of parameter value)
_JITTER_CUTOFF_FRAC: float = 0.03  # +/- 3% filter cutoff
_JITTER_DECAY_FRAC: float = 0.12  # +/- 12% filter env decay
_JITTER_Q_FRAC: float = 0.02  # +/- 2% resonance Q
_JITTER_ATTACK_FRAC: float = 0.12  # +/- 12% attack time
_JITTER_AMP_DB: float = 0.3  # +/- 0.3 dB amplitude

# Voice card calibration ranges
_VOICE_CARD_CUTOFF_CENTS: float = 50.0  # +/- 50 cents (~3%)
_VOICE_CARD_ATTACK_SCALE: float = 0.05  # +/- 5% (0.95-1.05)
_VOICE_CARD_RELEASE_SCALE: float = 0.05
_VOICE_CARD_AMP_DB: float = 0.2  # +/- 0.2 dB
_VOICE_CARD_PITCH_CENTS: float = 0.5  # +/- 0.5 cents (conservative for JI)
_VOICE_CARD_PULSE_WIDTH: float = 0.03  # +/- 0.03
_VOICE_CARD_RESONANCE_PCT: float = 5.0  # +/- 5%
_VOICE_CARD_SOFTNESS: float = 0.05  # +/- 0.05
_VOICE_CARD_DRIFT_RATE_PCT: float = 20.0  # +/- 20%

# OB-Xd-style fast per-sample CV dither.  Layered on top of the stable
# voice_card offsets: adds an extra rustle of audio-rate modulation to
# pitch and cutoff so held tones have a subtly living top end.
_CV_DITHER_PITCH_SEMITONES: float = 0.05  # +/- 0.05 semitone at amount=1.0
_CV_DITHER_CUTOFF_FRAC: float = 0.03  # +/- 3% of cutoff at amount=1.0
_CV_DITHER_LOWPASS_HZ: float = 4000.0  # tame out pure whiteness

SOUNDBOARD_MODES = (
    (80.0, 4.0, 0.10),
    (120.0, 5.0, 0.14),
    (180.0, 6.0, 0.18),
    (260.0, 5.0, 0.20),
    (340.0, 4.0, 0.18),
    (480.0, 3.5, 0.14),
    (680.0, 3.0, 0.10),
    (950.0, 2.5, 0.07),
    (1400.0, 2.0, 0.05),
    (2200.0, 1.5, 0.03),
    (3200.0, 1.2, 0.02),
    (4000.0, 1.0, 0.01),
)


def classify_thd(thd_pct: float) -> str:
    """Classify a THD percentage into a human-readable distortion label.

    Returns one of: ``"clean"``, ``"subtle_warmth"``, ``"warmth"``,
    ``"saturation"``, ``"distortion"``, ``"fuzz"``.
    """
    if thd_pct < 0.5:
        return "clean"
    elif thd_pct < 2.0:
        return "subtle_warmth"
    elif thd_pct < 5.0:
        return "warmth"
    elif thd_pct < 15.0:
        return "saturation"
    elif thd_pct < 40.0:
        return "distortion"
    return "fuzz"


def compute_signal_thd(
    freqs: np.ndarray,
    magnitude_db: np.ndarray,
    dominant_frequency_hz: float,
    *,
    bin_tolerance: int = 2,
    max_harmonic: int = 10,
) -> tuple[float, str]:
    """Measure THD of a signal from its averaged spectrum.

    Finds the fundamental bin nearest to *dominant_frequency_hz* and sums
    energy at harmonics 2 through *max_harmonic*.  Returns ``(thd_pct, label)``
    where *label* comes from :func:`classify_thd`.
    """
    if dominant_frequency_hz <= 20.0 or len(freqs) < 2:
        return 0.0, "clean"

    bin_spacing_hz = float(freqs[1] - freqs[0])
    if bin_spacing_hz <= 0:
        return 0.0, "clean"

    # Convert dB back to linear magnitude for power summation.
    magnitude_linear = 10.0 ** (magnitude_db / 20.0)

    def _peak_in_window(center_hz: float) -> float:
        center_idx = int(round((center_hz - float(freqs[0])) / bin_spacing_hz))
        lo = max(center_idx - bin_tolerance, 0)
        hi = min(center_idx + bin_tolerance + 1, len(magnitude_linear))
        if lo >= hi:
            return 0.0
        return float(np.max(magnitude_linear[lo:hi]))

    fundamental_amp = _peak_in_window(dominant_frequency_hz)
    if fundamental_amp <= 0.0:
        return 0.0, "clean"

    harmonic_power_sum = 0.0
    nyquist = float(freqs[-1])
    for h in range(2, max_harmonic + 1):
        h_freq = h * dominant_frequency_hz
        if h_freq > nyquist:
            break
        harmonic_power_sum += _peak_in_window(h_freq) ** 2

    thd_pct = float(np.sqrt(harmonic_power_sum)) / fundamental_amp * 100.0
    label = classify_thd(thd_pct)
    return round(thd_pct, 2), label


def compute_mode_ratios(
    *,
    freq: float,
    n_modes: int,
    inharmonicity: float,
    partial_ratios: list[dict[str, float]] | list[float] | None,
    sample_rate: int,
    amp_rolloff_exp: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mode frequency ratios (relative to f0) and per-mode amplitude weights.

    Returns ``(ratios, amps)`` where ratios are multipliers of the base frequency.
    ``amp_rolloff_exp`` controls the default amplitude rolloff: ``1/k^exp``.
    Piano uses ``1.0`` (default), harpsichord uses ``0.8``.
    """
    nyquist = sample_rate / 2.0

    if partial_ratios is not None:
        ratios_list: list[float] = []
        amps_list: list[float] = []
        for entry in partial_ratios:
            if isinstance(entry, dict):
                ratios_list.append(float(entry["ratio"]))
                amps_list.append(float(entry.get("amp", 1.0)))
            else:
                ratios_list.append(float(entry))
                amps_list.append(1.0)
        ratios_arr = np.array(ratios_list, dtype=np.float64)
        amps_arr = np.array(amps_list, dtype=np.float64)
        below_nyquist = (ratios_arr * freq) < nyquist
        return ratios_arr[below_nyquist], amps_arr[below_nyquist]

    ks = np.arange(1, n_modes + 1, dtype=np.float64)
    ratios_arr = ks * np.sqrt(1.0 + inharmonicity * ks**2)
    amps_arr = 1.0 / ks**amp_rolloff_exp
    below_nyquist = (ratios_arr * freq) < nyquist
    return ratios_arr[below_nyquist], amps_arr[below_nyquist]


def nyquist_fade(freq_profile: np.ndarray, nyquist_hz: float) -> np.ndarray:
    """Smooth fade starting at 85% of Nyquist to avoid brittle spectral edges."""
    fade_start_hz = nyquist_hz * NYQUIST_FADE_START
    if fade_start_hz >= nyquist_hz:
        return (freq_profile < nyquist_hz).astype(np.float64)
    fade_progress = (freq_profile - fade_start_hz) / (nyquist_hz - fade_start_hz)
    fade = 1.0 - np.clip(fade_progress, 0.0, 1.0)
    return np.square(fade)


def rng_for_note(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    extra_seed: str = "",
    params: dict[str, Any] | None = None,
) -> np.random.Generator:
    """Deterministic RNG seeded from note parameters via SHA-256.

    Produces the same Generator for the same inputs, ensuring render
    determinism without requiring a global seed.

    Both ``extra_seed`` and ``params`` are optional and additive -- when
    provided they are folded into the hash so that different engine
    configurations or caller-specific disambiguation strings produce
    distinct streams.
    """
    params_tuple = tuple(sorted(params.items(), key=lambda kv: kv[0])) if params else ()
    seed_material = repr(
        (
            round(freq, 6),
            round(duration, 6),
            round(amp, 6),
            sample_rate,
            extra_seed,
            params_tuple,
        )
    ).encode("utf-8")
    seed_bytes = sha256(seed_material).digest()[:8]
    seed = int.from_bytes(seed_bytes, byteorder="big", signed=False)
    return np.random.default_rng(seed)


@numba.njit(cache=True)
def iir_lowpass_1pole(signal: np.ndarray, alpha: float) -> np.ndarray:
    """Single-pole IIR lowpass: ``y[n] = alpha*x[n] + (1-alpha)*y[n-1]``, y[-1]=0."""
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    out[0] = signal[0] * alpha
    one_minus_alpha = 1.0 - alpha
    for j in range(1, n):
        out[j] = alpha * signal[j] + one_minus_alpha * out[j - 1]
    return out


# Legacy alias retained for older call sites that import the private name.
_iir_lowpass_1pole = iir_lowpass_1pole


def alpha_from_cutoff(cutoff_hz: float, sample_rate: int) -> float:
    """RC-style one-pole lowpass coefficient: ``alpha = dt / (RC + dt)``.

    Matches the coefficient convention used by :func:`iir_lowpass_1pole`
    (where ``alpha`` is the input weight).  ``cutoff_hz`` values <= 0 are
    floored to a tiny positive epsilon to avoid a divide-by-zero.
    """
    dt = 1.0 / max(float(sample_rate), 1.0)
    rc = 1.0 / (2.0 * math.pi * max(cutoff_hz, 1e-6))
    return dt / (rc + dt)


def smoothstep_blend(value: float, epsilon: float) -> float:
    """Smoothstep crossfade coefficient for ``value`` entering a process.

    Returns 0 for ``value <= 0``, the classic ``3t² - 2t³`` smoothstep for
    ``0 < value < epsilon``, and 1 for ``value >= epsilon``.  ``epsilon``
    must be positive.
    """
    if value <= 0.0:
        return 0.0
    if value >= epsilon:
        return 1.0
    t = value / epsilon
    return float(t * t * (3.0 - 2.0 * t))


@numba.njit(cache=True)
def _ou_process(noise: np.ndarray, theta: float, dt: float, sigma: float) -> np.ndarray:
    n = noise.shape[0]
    x = np.zeros(n, dtype=np.float64)
    sqrt_dt = math.sqrt(dt)
    neg_theta_dt = -theta * dt
    sigma_sqrt_dt = sigma * sqrt_dt
    for i in range(1, n):
        x[i] = x[i - 1] + neg_theta_dt * x[i - 1] + sigma_sqrt_dt * noise[i]
    return x


@numba.njit(cache=True)
def _envelope_follow(abs_signal: np.ndarray, follower_coeff: float) -> np.ndarray:
    n = abs_signal.shape[0]
    envelope = np.empty(n, dtype=np.float64)
    envelope[0] = abs_signal[0]
    for i in range(1, n):
        prev = follower_coeff * envelope[i - 1]
        current = abs_signal[i]
        envelope[i] = current if current > prev else prev
    return envelope


def build_drift(
    *,
    n_samples: int,
    drift_amount: float,
    drift_rate_hz: float,
    duration: float,
    phase_offset: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build 1/f multi-rate pitch drift, returns multiplicative trajectory.

    Four octave layers (0.4x, 1x, 3x, 10x base rate) with 1/f amplitude
    falloff produce natural-sounding aperiodic drift.  When *rng* is provided,
    each layer blends a sine with IIR-smoothed noise to further break
    periodicity.  Without *rng*, a deterministic incommensurate second sine
    per layer provides mild aperiodicity.
    """
    if drift_amount <= 0 or n_samples == 0:
        return np.ones(n_samples, dtype=np.float64)

    t = np.linspace(0.0, duration, n_samples, endpoint=False, dtype=np.float64)

    octave_rates = [
        drift_rate_hz * 0.4,
        drift_rate_hz,
        drift_rate_hz * 3.0,
        drift_rate_hz * 10.0,
    ]
    octave_amps = [1.0, 0.5, 0.25, 0.125]

    cents = np.zeros(n_samples, dtype=np.float64)
    sr_effective = n_samples / max(duration, 1e-6)
    two_pi = 2.0 * np.pi

    for i, (rate_i, amp_i) in enumerate(zip(octave_rates, octave_amps, strict=True)):
        phase_i = phase_offset + i * GOLDEN_RATIO_FRAC * two_pi
        sine_component = np.sin(two_pi * rate_i * t + phase_i)

        if rng is not None:
            raw_noise = rng.standard_normal(n_samples)
            cutoff_hz = rate_i * 2.0
            alpha = 1.0 - math.exp(-two_pi * cutoff_hz / sr_effective)
            smoothed = _iir_lowpass_1pole(raw_noise, alpha)
            smooth_peak = np.max(np.abs(smoothed))
            if smooth_peak > 1e-12:
                smoothed /= smooth_peak
            layer = 0.7 * sine_component + 0.3 * smoothed
        else:
            # Deterministic fallback: incommensurate second sine
            layer = 0.7 * sine_component + 0.3 * np.sin(
                two_pi * rate_i * 1.317 * t + phase_i * 0.7
            )

        cents += amp_i * layer

    peak_cents = np.max(np.abs(cents))
    if peak_cents > 1e-12:
        cents = MAX_DRIFT_CENTS * drift_amount * cents / peak_cents

    return np.power(2.0, cents / 1200.0)


def build_cutoff_drift(
    n_samples: int,
    *,
    amount_cents: float = 30.0,
    rate_hz: float = 0.3,
    rng: np.random.Generator,
    sample_rate: int,
) -> np.ndarray:
    """Ornstein-Uhlenbeck mean-reverting cutoff drift.

    Returns a multiplicative ratio array for modulating filter cutoff.
    The O-U process naturally returns to center without hard clamping,
    producing organic filter movement.

    Args:
        n_samples: Output length.
        amount_cents: RMS excursion in cents (30 = subtle wobble).
        rate_hz: Spring-back rate (higher = stays closer to center).
        rng: Deterministic RNG from rng_for_note().
        sample_rate: Audio sample rate.
    """
    if amount_cents <= 0 or n_samples == 0:
        return np.ones(n_samples, dtype=np.float64)

    dt = 1.0 / sample_rate
    theta = rate_hz * 2.0 * np.pi  # spring-back rate
    # Calibrate sigma so RMS excursion ~ amount_cents
    # For O-U: stationary std = sigma / sqrt(2 * theta)
    sigma = amount_cents * np.sqrt(2.0 * theta)

    # Generate the O-U process
    noise = rng.standard_normal(n_samples)
    x = _ou_process(noise, theta, dt, sigma)

    # Convert cents to multiplicative ratio
    return np.power(2.0, x / 1200.0)


def apply_body_saturation(
    signal: np.ndarray,
    amount: float,
    *,
    cubic_amount: float = 0.08,
    even_amount: float = 0.22,
    log_thd: bool = False,
) -> np.ndarray:
    """Gentle asymmetric waveshaping for intermodulation warmth.

    ``cubic_amount`` and ``even_amount`` control the mix of odd (cubic) and
    even (half-wave rectified) harmonic content. piano.py uses the defaults;
    piano_additive.py uses ``cubic_amount=0.15, even_amount=0.35, log_thd=True``.
    """
    if amount <= 0:
        return signal

    peak = np.max(np.abs(signal))
    if peak <= 0:
        return signal

    normalized = signal / peak
    shaped = normalized - (amount * cubic_amount) * normalized**3
    even_harmonics = normalized * np.abs(normalized)
    shaped = shaped + even_amount * amount * even_harmonics

    wet_blend = min(1.0, amount * 0.6)
    result = (1.0 - wet_blend) * normalized + wet_blend * shaped
    result_peak = np.max(np.abs(result))
    if result_peak > 0:
        result *= peak / result_peak

    if log_thd:
        diff_rms = float(np.sqrt(np.mean((result / peak - normalized) ** 2)))
        sig_rms = float(np.sqrt(np.mean(normalized**2)))
        thd_pct = diff_rms / max(sig_rms, 1e-10) * 100.0
        label = classify_thd(thd_pct)
        logger.debug(f"body_saturation {amount=:.3f}: THD {thd_pct:.2f}% ({label})")

    return result


def render_noise_floor(
    *,
    signal: np.ndarray,
    sample_rate: int,
    n_samples: int,
    rng: np.random.Generator,
    level: float = 0.001,
) -> np.ndarray:
    """Add a subtle filtered noise floor that follows the amplitude envelope."""
    peak = np.max(np.abs(signal))
    if peak <= 0:
        return np.zeros(n_samples, dtype=np.float64)

    abs_signal = np.abs(signal)
    follower_coeff = np.exp(-1.0 / (0.02 * sample_rate))
    envelope = _envelope_follow(abs_signal, follower_coeff)

    noise = rng.standard_normal(n_samples)

    # FFT-domain pink noise shaping (100-8000 Hz)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    safe_freqs = np.maximum(freqs, 1.0)
    pink_mask = np.where(freqs > 0, 1.0 / np.sqrt(safe_freqs), 1.0)
    low_fade = np.clip((freqs - 80.0) / 40.0, 0.0, 1.0)
    high_fade = np.clip((10000.0 - freqs) / 4000.0, 0.0, 1.0)
    noise = np.fft.irfft(spectrum * pink_mask * low_fade * high_fade, n=n_samples).real

    noise_peak = np.max(np.abs(noise))
    if noise_peak > 0:
        noise /= noise_peak

    return noise * envelope * level


def apply_soundboard(
    *,
    signal: np.ndarray,
    soundboard_color: float,
    soundboard_brightness: float,
    sample_rate: int,
) -> np.ndarray:
    """Apply soundboard resonance via resonant modes + lowpass body color.

    F7 fix: lowpass resonance now uses [0.5, 0.8] range so it is not clamped
    to the ZDF SVF minimum of 0.5.
    """
    if soundboard_color <= 0:
        return signal

    result = signal.copy()

    for mode_hz, mode_q, mode_level in SOUNDBOARD_MODES:
        if mode_hz >= sample_rate / 2.0:
            continue
        cutoff_profile = np.full(signal.size, mode_hz, dtype=np.float64)
        resonant = apply_filter(
            signal,
            cutoff_profile=cutoff_profile,
            resonance_q=mode_q,
            sample_rate=sample_rate,
            filter_mode="bandpass",
            filter_drive=0.0,
        )
        result += resonant * soundboard_color * mode_level

    cutoff_hz = 300.0 + soundboard_brightness * 4000.0
    lp_resonance = 0.5 + soundboard_color * 0.3
    cutoff_profile = np.full(signal.size, cutoff_hz, dtype=np.float64)
    lp_wet = apply_filter(
        signal,
        cutoff_profile=cutoff_profile,
        resonance_q=lp_resonance,
        sample_rate=sample_rate,
        filter_mode="lowpass",
        filter_drive=0.0,
    )
    blend = soundboard_color * 0.35
    result = result * (1.0 - blend) + lp_wet * blend

    return result


def render_damper_thump(
    *,
    freq: float,
    damper_noise: float,
    n_samples: int,
    sample_rate: int,
    rng: np.random.Generator,
    level_scale: float = 1.0,
    burst_duration_s: float = 0.02,
    center_hz: float | None = None,
    width_ratio: float = 0.75,
) -> np.ndarray:
    """Render a short noise burst near the end of the note.

    ``level_scale`` allows the caller to pass an external amplitude reference
    (e.g. ``string_rms``) that scales the output level.

    ``burst_duration_s`` controls the decay time of the noise burst (piano
    damper thump uses 0.02, harpsichord release noise uses 0.012).

    ``center_hz`` overrides the bandpass center frequency (defaults to *freq*).

    ``width_ratio`` controls the bandpass width (piano uses 0.75, harpsichord
    uses 1.2 for a brighter burst).
    """
    if damper_noise <= 0 or (level_scale != 1.0 and level_scale <= 0):
        return np.zeros(n_samples, dtype=np.float64)

    burst_duration_samples = max(1, int(burst_duration_s * sample_rate))
    burst_start = max(0, n_samples - burst_duration_samples * 3)

    noise = rng.standard_normal(n_samples)
    envelope = np.zeros(n_samples, dtype=np.float64)
    burst_len = min(burst_duration_samples * 3, n_samples - burst_start)
    t_burst = np.arange(burst_len, dtype=np.float64)
    envelope[burst_start : burst_start + burst_len] = np.exp(
        -t_burst / max(1.0, burst_duration_samples)
    )

    bp_center = center_hz if center_hz is not None else freq
    noise = bandpass_noise(
        noise, sample_rate=sample_rate, center_hz=bp_center, width_ratio=width_ratio
    )
    return damper_noise * level_scale * envelope * noise


def bandpass_noise(
    signal: np.ndarray,
    *,
    sample_rate: int,
    center_hz: float,
    width_ratio: float = 0.75,
    min_width_hz: float = 80.0,
    gaussian_sigma_divisor: float = 2.5,
    center_clip_min_hz: float = 30.0,
    hard_edges: bool = True,
) -> np.ndarray:
    """Bandpass-filter a noise signal around a center frequency via FFT.

    Canonical FFT-domain Gaussian bandpass.  ``hard_edges=True`` (the default)
    adds rectangular band cutoffs outside the Gaussian for sharper rolloff;
    ``hard_edges=False`` gives a pure Gaussian shape (used by the drum
    "narrow" variant in ``_drum_utils``).
    """
    if signal.size == 0:
        return signal

    nyquist = sample_rate / 2.0
    center_hz = float(np.clip(center_hz, center_clip_min_hz, nyquist * 0.95))
    width_hz = max(min_width_hz, center_hz * width_ratio)

    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    mask = np.exp(
        -0.5 * ((freqs - center_hz) / max(1.0, width_hz / gaussian_sigma_divisor)) ** 2
    )
    if hard_edges:
        low_hz = max(20.0, center_hz - width_hz / 2.0)
        high_hz = min(nyquist * 0.98, center_hz + width_hz / 2.0)
        mask *= (freqs >= low_hz).astype(np.float64)
        mask *= (freqs <= high_hz).astype(np.float64)
    shaped = np.fft.irfft(spectrum * mask, n=signal.size)
    return shaped.real


def flow_exciter(
    *,
    n_samples: int,
    param: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rare-event sample-and-hold noise generator (MI Elements "Flow" exciter).

    Produces breath/brush-like stochastic noise. Low ``param`` yields very
    sparse events (pauses interrupted by rare "flips"), high ``param``
    yields essentially uniform noise. The transition between regimes is
    continuous -- the same generator covers the gamut from an exhaled-air
    whisper to a sustained brush.

    DSP definition (per-sample):

        threshold  = 0.0001 + 0.125 * param**4
        mix_weight = param**4
        on each sample:
            draw r ~ uniform(-0.5, 0.5)
            if uniform(0, 1) < threshold: state = r   (flip)
            out = state + (r - state) * mix_weight

    Vectorized with numpy. Deterministic given a fixed ``rng``.

    Args:
        n_samples: Number of samples to generate.
        param: Density / sparsity control in ``[0, 1]``.
        rng: Pre-seeded numpy ``Generator`` for deterministic output.

    Returns:
        Float64 array of length ``n_samples``, bounded within ``[-0.5, 0.5]``.

    Notes:
        Algorithm re-implemented in numpy from Mutable Instruments
        ``elements/dsp/exciter.cc::ProcessFlow`` (MIT-licensed). Not a
        verbatim port.
    """
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}")
    if not (0.0 <= param <= 1.0):
        raise ValueError(f"param must be in [0, 1], got {param}")
    if n_samples == 0:
        return np.zeros(0, dtype=np.float64)

    threshold = 0.0001 + 0.125 * (param**4)
    mix_weight = param**4

    uniform_samples = rng.uniform(-0.5, 0.5, size=n_samples)
    flip_trigger = rng.uniform(0.0, 1.0, size=n_samples) < threshold

    # Propagate the S&H state vectorized: initial value is 0.0; at each
    # flip index the state adopts uniform_samples[flip_index] and holds
    # until the next flip.
    state = np.zeros(n_samples, dtype=np.float64)
    if flip_trigger.any():
        flip_indices = np.flatnonzero(flip_trigger)
        segment_starts = np.concatenate(([0], flip_indices))
        segment_values = np.concatenate(([0.0], uniform_samples[flip_indices]))
        segment_lengths = np.diff(np.concatenate((segment_starts, [n_samples])))
        state = np.repeat(segment_values, segment_lengths)

    return state + (uniform_samples - state) * mix_weight


# ---------------------------------------------------------------------------
# Reusable FM modulation primitive for drum engines
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _fm_modulate_loop(
    out: np.ndarray,
    carrier_freq_profile: np.ndarray,
    mod_freq_profile: np.ndarray,
    mod_index_profile: np.ndarray,
    sample_rate: int,
    feedback: float,
) -> None:
    """Per-sample FM synthesis: carrier modulated by a single modulator."""
    carrier_phase = 0.0
    mod_phase = 0.0
    prev_mod = 0.0
    two_pi_over_sr = 2.0 * math.pi / sample_rate

    for i in range(out.shape[0]):
        mod_sample = math.sin(mod_phase + feedback * prev_mod)
        prev_mod = mod_sample
        out[i] = math.sin(carrier_phase + mod_index_profile[i] * mod_sample)
        carrier_phase += carrier_freq_profile[i] * two_pi_over_sr
        mod_phase += mod_freq_profile[i] * two_pi_over_sr


def fm_modulate(
    carrier_freq_profile: np.ndarray,
    *,
    mod_ratio: float,
    mod_index: float,
    sample_rate: int,
    feedback: float = 0.0,
    index_envelope: np.ndarray | None = None,
) -> np.ndarray:
    """Render an FM-modulated oscillator.

    This is a simpler, reusable primitive compared to the full 2-operator FM
    engine.  Designed for drum engines where FM adds harmonic richness to a
    pitched body oscillator (e.g. kick body with decaying mod index for a
    harmonically rich attack that thins to a clean fundamental).

    Args:
        carrier_freq_profile: Per-sample carrier frequency in Hz.
        mod_ratio: Modulator frequency as a ratio of the carrier.
        mod_index: Peak modulation index (controls harmonic richness).
        sample_rate: Audio sample rate.
        feedback: Self-feedback on the modulator (0.0 = none, adds noise/complexity).
        index_envelope: Optional per-sample modulation index multiplier (0-1 range).
            When provided, effective index = mod_index * index_envelope[i].
            When None, a flat envelope of 1.0 is used (constant index).

    Returns:
        FM-modulated signal as float64 array.
    """
    carrier_freq_profile = np.asarray(carrier_freq_profile, dtype=np.float64)
    n_samples = carrier_freq_profile.shape[0]

    if mod_ratio <= 0:
        raise ValueError("mod_ratio must be positive")
    if mod_index < 0:
        raise ValueError("mod_index must be non-negative")
    # Phase-modulation self-feedback: the loop computes
    # sin(mod_phase + feedback * prev_mod), which stays stable and musical for
    # feedback in [0, 1]. Values above 1 push the recursion into chaotic /
    # noise-like territory that callers almost never want; fail fast instead of
    # silently producing garbage.
    if not 0.0 <= feedback <= 1.0:
        raise ValueError(
            f"feedback must be in [0.0, 1.0]; got {feedback!r}. "
            "Higher values push the FM recursion into chaotic noise."
        )
    if not np.all(np.isfinite(carrier_freq_profile)):
        raise ValueError(
            "carrier_freq_profile contains non-finite values (NaN or Inf); "
            "FM output would silently propagate non-finite samples."
        )

    mod_freq_profile = carrier_freq_profile * mod_ratio

    if index_envelope is not None:
        index_env = np.asarray(index_envelope, dtype=np.float64)
        if index_env.shape[0] != n_samples:
            raise ValueError(
                f"index_envelope length ({index_env.shape[0]}) must match "
                f"carrier_freq_profile length ({n_samples})"
            )
        mod_index_profile = mod_index * index_env
    else:
        mod_index_profile = np.full(n_samples, mod_index, dtype=np.float64)

    out = np.empty(n_samples, dtype=np.float64)
    _fm_modulate_loop(
        out,
        carrier_freq_profile,
        mod_freq_profile,
        mod_index_profile,
        sample_rate,
        feedback,
    )
    return out


# ---------------------------------------------------------------------------
# Two-operator (parallel modulators) FM primitive for EFM-style drum tones
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _fm_modulate_2op_loop(
    out: np.ndarray,
    carrier_freq_profile: np.ndarray,
    mod1_freq_profile: np.ndarray,
    mod2_freq_profile: np.ndarray,
    mod1_index_profile: np.ndarray,
    mod2_index_profile: np.ndarray,
    sample_rate: int,
    mod1_feedback: float,
    mod2_feedback: float,
    carrier_feedback: float,
) -> None:
    """Two independent modulators summed into one carrier, DX-style."""
    carrier_phase = 0.0
    mod1_phase = 0.0
    mod2_phase = 0.0
    prev_mod1 = 0.0
    prev_mod2 = 0.0
    prev_carrier = 0.0
    two_pi_over_sr = 2.0 * math.pi / sample_rate

    for i in range(out.shape[0]):
        mod1_sample = math.sin(mod1_phase + mod1_feedback * prev_mod1)
        mod2_sample = math.sin(mod2_phase + mod2_feedback * prev_mod2)
        prev_mod1 = mod1_sample
        prev_mod2 = mod2_sample

        carrier_sample = math.sin(
            carrier_phase
            + mod1_index_profile[i] * mod1_sample
            + mod2_index_profile[i] * mod2_sample
            + carrier_feedback * prev_carrier
        )
        prev_carrier = carrier_sample
        out[i] = carrier_sample

        carrier_phase += carrier_freq_profile[i] * two_pi_over_sr
        mod1_phase += mod1_freq_profile[i] * two_pi_over_sr
        mod2_phase += mod2_freq_profile[i] * two_pi_over_sr


def fm_modulate_2op(
    carrier_freq_profile: np.ndarray,
    *,
    mod1_ratio: float,
    mod1_index: float,
    mod2_ratio: float,
    mod2_index: float,
    sample_rate: int,
    mod1_feedback: float = 0.0,
    mod2_feedback: float = 0.0,
    carrier_feedback: float = 0.0,
    index_envelope: np.ndarray | None = None,
) -> np.ndarray:
    """Two parallel modulators feeding one carrier (DX-style 2-op FM).

    Both modulators contribute to the carrier phase simultaneously
    (additive, not cascaded), which produces a distinct sound from
    chaining two single-modulator calls. Each modulator has independent
    self-feedback, and the carrier can also feed back into itself for
    a ring-like growl.

    Args:
        carrier_freq_profile: Per-sample carrier frequency in Hz.
        mod1_ratio: First modulator frequency as ratio of carrier.
        mod1_index: First modulator peak index.
        mod2_ratio: Second modulator frequency as ratio of carrier.
        mod2_index: Second modulator peak index.
        sample_rate: Audio sample rate.
        mod1_feedback: Self-feedback on modulator 1 (0.0-1.0).
        mod2_feedback: Self-feedback on modulator 2 (0.0-1.0).
        carrier_feedback: Self-feedback on the carrier (0.0-1.0).
        index_envelope: Optional per-sample index multiplier applied
            to BOTH modulator indices.  When None, flat 1.0 is used.

    Returns:
        FM-modulated signal as float64 array.
    """
    carrier_freq_profile = np.asarray(carrier_freq_profile, dtype=np.float64)
    n_samples = carrier_freq_profile.shape[0]

    if mod1_ratio <= 0:
        raise ValueError("mod1_ratio must be positive")
    if mod2_ratio < 0:
        raise ValueError("mod2_ratio must be non-negative")
    if mod2_ratio == 0.0 and mod2_index != 0:
        raise ValueError(
            "mod2_ratio=0 disables the second modulator and requires mod2_index=0; "
            f"got mod2_index={mod2_index!r}"
        )
    if mod1_index < 0:
        raise ValueError("mod1_index must be non-negative")
    if mod2_index < 0:
        raise ValueError("mod2_index must be non-negative")
    if not 0.0 <= mod1_feedback <= 1.0:
        raise ValueError(
            f"mod1_feedback must be in [0.0, 1.0]; got {mod1_feedback!r}. "
            "Higher values push the FM recursion into chaotic noise."
        )
    if not 0.0 <= mod2_feedback <= 1.0:
        raise ValueError(
            f"mod2_feedback must be in [0.0, 1.0]; got {mod2_feedback!r}. "
            "Higher values push the FM recursion into chaotic noise."
        )
    if not 0.0 <= carrier_feedback <= 1.0:
        raise ValueError(
            f"carrier_feedback must be in [0.0, 1.0]; got {carrier_feedback!r}. "
            "Higher values push the FM recursion into chaotic noise."
        )
    if not np.all(np.isfinite(carrier_freq_profile)):
        raise ValueError(
            "carrier_freq_profile contains non-finite values (NaN or Inf); "
            "FM output would silently propagate non-finite samples."
        )

    mod1_freq_profile = carrier_freq_profile * mod1_ratio
    mod2_freq_profile = carrier_freq_profile * mod2_ratio

    if index_envelope is not None:
        index_env = np.asarray(index_envelope, dtype=np.float64)
        if index_env.shape[0] != n_samples:
            raise ValueError(
                f"index_envelope length ({index_env.shape[0]}) must match "
                f"carrier_freq_profile length ({n_samples})"
            )
        mod1_index_profile = mod1_index * index_env
        mod2_index_profile = mod2_index * index_env
    else:
        mod1_index_profile = np.full(n_samples, mod1_index, dtype=np.float64)
        mod2_index_profile = np.full(n_samples, mod2_index, dtype=np.float64)

    out = np.empty(n_samples, dtype=np.float64)
    _fm_modulate_2op_loop(
        out,
        carrier_freq_profile,
        mod1_freq_profile,
        mod2_freq_profile,
        mod1_index_profile,
        mod2_index_profile,
        sample_rate,
        mod1_feedback,
        mod2_feedback,
        carrier_feedback,
    )
    return out


# ---------------------------------------------------------------------------
# N-operator phase-modulation primitive for EFM-style cymbal layers
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _phase_modulate_nop_loop(
    out: np.ndarray,
    carrier_freq_profile: np.ndarray,
    op_freq_profiles: np.ndarray,
    op_index_profiles: np.ndarray,
    op_feedbacks: np.ndarray,
    sample_rate: int,
) -> None:
    """N phase-modulation operators feeding a single sine carrier in parallel."""
    n_ops = op_freq_profiles.shape[0]
    n_samples = out.shape[0]

    carrier_phase = 0.0
    op_phases = np.zeros(n_ops, dtype=np.float64)
    prev_op_samples = np.zeros(n_ops, dtype=np.float64)
    two_pi_over_sr = 2.0 * math.pi / sample_rate

    for i in range(n_samples):
        total_mod = 0.0
        for j in range(n_ops):
            op_sample = math.sin(op_phases[j] + op_feedbacks[j] * prev_op_samples[j])
            prev_op_samples[j] = op_sample
            total_mod += op_index_profiles[j, i] * op_sample

        out[i] = math.sin(carrier_phase + total_mod)

        carrier_phase += carrier_freq_profile[i] * two_pi_over_sr
        for j in range(n_ops):
            op_phases[j] += op_freq_profiles[j, i] * two_pi_over_sr


def phase_modulate_nop(
    carrier_freq_profile: np.ndarray,
    *,
    op_ratios: np.ndarray,
    op_indices: np.ndarray,
    op_feedbacks: np.ndarray,
    sample_rate: int,
    op_envelopes: np.ndarray | None = None,
) -> np.ndarray:
    """N parallel phase-modulation operators feeding one sine carrier.

    Each operator runs at its own ratio of the carrier frequency, with
    optional self-feedback and optional per-op per-sample index
    envelope.  All operators sum into the carrier phase additively.
    Useful for EFM-style cymbal/metallic layers where several
    inharmonic operators build up a dense, noisy spectrum.

    Args:
        carrier_freq_profile: Per-sample carrier frequency in Hz,
            shape (T,).
        op_ratios: Per-op frequency ratios of the carrier, shape (N,).
        op_indices: Per-op peak modulation indices, shape (N,).
        op_feedbacks: Per-op self-feedback amounts in [0.0, 1.0],
            shape (N,).
        sample_rate: Audio sample rate.
        op_envelopes: Optional per-op per-sample index multipliers,
            shape (N, T).  When None, flat unity envelopes are used.

    Returns:
        Phase-modulated signal as float64 array, shape (T,).
    """
    carrier_freq_profile = np.ascontiguousarray(carrier_freq_profile, dtype=np.float64)
    op_ratios = np.ascontiguousarray(op_ratios, dtype=np.float64)
    op_indices = np.ascontiguousarray(op_indices, dtype=np.float64)
    op_feedbacks = np.ascontiguousarray(op_feedbacks, dtype=np.float64)
    n_samples = carrier_freq_profile.shape[0]
    n_ops = op_ratios.shape[0]

    if op_indices.shape[0] != n_ops:
        raise ValueError(
            f"op_indices length ({op_indices.shape[0]}) must match "
            f"op_ratios length ({n_ops})"
        )
    if op_feedbacks.shape[0] != n_ops:
        raise ValueError(
            f"op_feedbacks length ({op_feedbacks.shape[0]}) must match "
            f"op_ratios length ({n_ops})"
        )
    if np.any(op_ratios <= 0):
        raise ValueError("all op_ratios must be positive")
    if np.any(op_indices < 0):
        raise ValueError("all op_indices must be non-negative")
    if np.any((op_feedbacks < 0.0) | (op_feedbacks > 1.0)):
        raise ValueError(
            "all op_feedbacks must be in [0.0, 1.0]; higher values push the "
            "PM recursion into chaotic noise."
        )
    if not np.all(np.isfinite(carrier_freq_profile)):
        raise ValueError(
            "carrier_freq_profile contains non-finite values (NaN or Inf)."
        )

    op_freq_profiles = np.empty((n_ops, n_samples), dtype=np.float64)
    for j in range(n_ops):
        op_freq_profiles[j, :] = carrier_freq_profile * op_ratios[j]

    if op_envelopes is not None:
        env = np.ascontiguousarray(op_envelopes, dtype=np.float64)
        if env.ndim != 2:
            raise ValueError(
                f"op_envelopes must be 2D with shape (n_ops, n_samples); "
                f"got ndim={env.ndim}"
            )
        if env.shape != (n_ops, n_samples):
            raise ValueError(
                f"op_envelopes shape {env.shape} must match "
                f"(n_ops={n_ops}, n_samples={n_samples})"
            )
        op_index_profiles = op_indices.reshape(n_ops, 1) * env
    else:
        op_index_profiles = np.broadcast_to(
            op_indices.reshape(n_ops, 1), (n_ops, n_samples)
        ).copy()

    op_index_profiles = np.ascontiguousarray(op_index_profiles, dtype=np.float64)

    out = np.empty(n_samples, dtype=np.float64)
    _phase_modulate_nop_loop(
        out,
        carrier_freq_profile,
        op_freq_profiles,
        op_index_profiles,
        op_feedbacks,
        sample_rate,
    )
    return out


# ---------------------------------------------------------------------------
# Shared analog character helpers
# ---------------------------------------------------------------------------


def extract_analog_params(params: dict[str, Any]) -> dict[str, Any]:
    """Extract analog character parameters with defaults.

    Returns a flat dict of the common analog-character knobs that
    polyblep, filtered_stack, and fm engines all share.

    ``voice_card_spread`` (0.0-3.0, default 1.0) sets the global inter-voice
    calibration variation.  Per-group overrides let you scale individual
    dimensions independently (e.g., keep pitch tight for JI while opening up
    filter and envelope variation):

    - ``voice_card_pitch_spread`` — pitch offset (default: global)
    - ``voice_card_filter_spread`` — cutoff + resonance (default: global)
    - ``voice_card_envelope_spread`` — attack + release timing (default: global)
    - ``voice_card_osc_spread`` — pulse width, softness, drift rate (default: global)
    - ``voice_card_level_spread`` — amplitude (default: global)

    For backward compatibility, the legacy ``voice_card`` key is accepted
    as a fallback for the global spread.
    """
    if "voice_card_spread" in params:
        spread = float(params["voice_card_spread"])
    else:
        spread = float(params.get("voice_card", 1.0))
    return {
        "pitch_drift": float(params.get("pitch_drift", 0.12)),
        "analog_jitter": float(params.get("analog_jitter", 1.0)),
        "noise_floor": float(params.get("noise_floor", 0.001)),
        "drift_rate_hz": float(params.get("drift_rate_hz", 0.3)),
        "cutoff_drift": float(params.get("cutoff_drift", 0.5)),
        "voice_card_spread": spread,
        "voice_card_pitch_spread": float(params.get("voice_card_pitch_spread", spread)),
        "voice_card_filter_spread": float(
            params.get("voice_card_filter_spread", spread)
        ),
        "voice_card_envelope_spread": float(
            params.get("voice_card_envelope_spread", spread)
        ),
        "voice_card_osc_spread": float(params.get("voice_card_osc_spread", spread)),
        "voice_card_level_spread": float(params.get("voice_card_level_spread", spread)),
        "osc_asymmetry": float(params.get("osc_asymmetry", 0.0)),
        "osc_softness": float(params.get("osc_softness", 0.0)),
        "osc_dc_offset": float(params.get("osc_dc_offset", 0.0)),
        "osc_shape_drift": float(params.get("osc_shape_drift", 0.0)),
        "quality": str(params.get("quality", "great")),
        "transient_mode": str(params.get("transient_mode", "analog")),
    }


# ---------------------------------------------------------------------------
# Engine-level quality modes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityConfig:
    """Engine-level quality configuration.

    Controls the ladder filter solver, Newton iteration budget, and the
    internal oversampling factor applied around the filter + feedback +
    dither block.  Oscillators are generated at the base rate (BLEP
    already handles oscillator aliasing) — oversampling is strictly for
    the filter section.
    """

    solver: str
    max_newton_iters: int
    newton_tolerance: float
    oversample_factor: int


_SUPPORTED_QUALITIES: set[str] = {"draft", "fast", "great", "divine"}


# ---------------------------------------------------------------------------
# Transient / reset modes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransientConfig:
    """Per-note reset policy for oscillator phase and DC offset carry-over.

    Modes: ``analog`` (carry both), ``dc_reset`` (carry phase, reset DC),
    ``osc_reset`` (reset both — per-note random phase).
    """

    reset_phase: bool
    reset_dc: bool


_SUPPORTED_TRANSIENT_MODES: set[str] = {
    "analog",
    "dc_reset",
    "osc_reset",
}


def resolve_transient_mode(mode: str) -> TransientConfig:
    """Return the `TransientConfig` for a named transient mode.

    Supported modes: ``analog`` (default), ``dc_reset``, ``osc_reset``.
    Raises ``ValueError`` for any other value — fail fast rather than
    silently falling back to a default.
    """
    if mode == "analog":
        return TransientConfig(reset_phase=False, reset_dc=False)
    if mode == "dc_reset":
        return TransientConfig(reset_phase=False, reset_dc=True)
    if mode == "osc_reset":
        return TransientConfig(reset_phase=True, reset_dc=True)
    raise ValueError(
        f"Unknown transient_mode: {mode!r}. "
        f"Supported: {sorted(_SUPPORTED_TRANSIENT_MODES)}"
    )


def apply_transient_state(
    voice_state: dict[str, Any] | None,
    *,
    transient_config: TransientConfig,
    fresh_phase: float,
    fresh_dc_signs: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Resolve per-note oscillator phases + DC signs from the voice state.

    Returns ``(start_phase_osc1, start_phase_osc2, dc_sign_osc1, dc_sign_osc2)``
    with both phases in radians and DC signs as ``+/- 1.0`` multipliers on
    ``osc_dc_offset``.

    When ``voice_state`` is ``None`` or lacks a given key, the corresponding
    fresh input is returned.  ``reset_phase`` discards both prior phases;
    ``reset_dc`` discards both prior DC signs.  osc2's phase defaults to
    ``fresh_phase`` when not present in the state dict so callers can
    construct their own osc2 start phase downstream when osc2 isn't active.
    """
    if voice_state is None:
        return (
            fresh_phase,
            fresh_phase,
            fresh_dc_signs[0],
            fresh_dc_signs[1],
        )

    if transient_config.reset_phase or "phase_osc1" not in voice_state:
        start_phase_osc1 = fresh_phase
    else:
        start_phase_osc1 = float(voice_state["phase_osc1"])

    if transient_config.reset_phase or "phase_osc2" not in voice_state:
        start_phase_osc2 = fresh_phase
    else:
        start_phase_osc2 = float(voice_state["phase_osc2"])

    if transient_config.reset_dc or "dc_sign_osc1" not in voice_state:
        dc_sign_osc1 = fresh_dc_signs[0]
    else:
        dc_sign_osc1 = float(voice_state["dc_sign_osc1"])

    if transient_config.reset_dc or "dc_sign_osc2" not in voice_state:
        dc_sign_osc2 = fresh_dc_signs[1]
    else:
        dc_sign_osc2 = float(voice_state["dc_sign_osc2"])

    return start_phase_osc1, start_phase_osc2, dc_sign_osc1, dc_sign_osc2


def snapshot_voice_state(
    voice_state: dict[str, Any] | None,
    *,
    final_phase_osc1: float,
    final_phase_osc2: float,
    dc_sign_osc1: float,
    dc_sign_osc2: float,
) -> None:
    """Persist end-of-note oscillator state back into the voice state dict.

    A no-op when ``voice_state`` is ``None``.  Both phases are stored as
    continuous radian values; callers need not pre-wrap since the oscillator
    takes ``start_phase`` modulo 2π anyway.
    """
    if voice_state is None:
        return
    voice_state["phase_osc1"] = float(final_phase_osc1)
    voice_state["phase_osc2"] = float(final_phase_osc2)
    voice_state["dc_sign_osc1"] = float(dc_sign_osc1)
    voice_state["dc_sign_osc2"] = float(dc_sign_osc2)


def resolve_quality_mode(quality: str) -> QualityConfig:
    """Return the `QualityConfig` for a named quality mode.

    Supported modes: ``draft`` (ADAA, 1x), ``fast`` (Newton 2 iters, 2x),
    ``great`` (Newton 4 iters, 2x), ``divine`` (Newton 8 iters, 4x).
    """
    if quality == "draft":
        return QualityConfig("adaa", 0, 0.0, 1)
    if quality == "fast":
        return QualityConfig("newton", 2, 1e-8, 2)
    if quality == "great":
        return QualityConfig("newton", 4, 1e-9, 2)
    if quality == "divine":
        return QualityConfig("newton", 8, 1e-10, 4)
    raise ValueError(
        f"Unknown quality mode: {quality!r}. Supported: {sorted(_SUPPORTED_QUALITIES)}"
    )


_OVERSAMPLE_RESAMPLE_WINDOW = ("kaiser", 8.6)


def apply_filter_oversampled(
    signal: np.ndarray,
    *,
    cutoff_profile: np.ndarray,
    sample_rate: int,
    oversample_factor: int,
    resonance_q: float = 0.707,
    filter_mode: str = "lowpass",
    filter_drive: float = 0.0,
    filter_even_harmonics: float = 0.0,
    filter_topology: str = "svf",
    bass_compensation: float = 0.5,
    filter_morph: float = 0.0,
    hpf_cutoff_hz: float = 0.0,
    hpf_resonance_q: float = 0.707,
    feedback_amount: float = 0.0,
    feedback_saturation: float = 0.3,
    filter_solver: str = "adaa",
    max_newton_iters: int = 4,
    newton_tolerance: float = 1e-9,
    k35_feedback_asymmetry: float = 0.0,
) -> np.ndarray:
    """Run `apply_filter` at ``oversample_factor * sample_rate``.

    Upsamples ``signal`` and ``cutoff_profile`` via polyphase resampling
    (Kaiser window), runs the filter at the higher rate, then downsamples
    the output back to ``sample_rate``.

    When ``oversample_factor == 1`` this short-circuits to a direct
    `apply_filter` call with no resample roundtrip.
    """
    if oversample_factor < 1:
        raise ValueError("oversample_factor must be >= 1")

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
        k35_feedback_asymmetry=k35_feedback_asymmetry,
        max_newton_iters=max_newton_iters,
        newton_tolerance=newton_tolerance,
    )

    if oversample_factor == 1:
        return apply_filter(
            signal,
            cutoff_profile=cutoff_profile,
            sample_rate=sample_rate,
            **asdict(fp),
        )

    n_base = int(signal.shape[0])
    if n_base == 0:
        return np.zeros(0, dtype=np.float64)

    signal_up = resample_poly(
        signal, up=oversample_factor, down=1, window=_OVERSAMPLE_RESAMPLE_WINDOW
    ).astype(np.float64)
    cutoff_up = resample_poly(
        cutoff_profile,
        up=oversample_factor,
        down=1,
        window=_OVERSAMPLE_RESAMPLE_WINDOW,
    ).astype(np.float64)

    n_up_expected = n_base * oversample_factor
    if signal_up.shape[0] != n_up_expected:
        if signal_up.shape[0] > n_up_expected:
            signal_up = signal_up[:n_up_expected]
        else:
            signal_up = np.concatenate(
                [signal_up, np.zeros(n_up_expected - signal_up.shape[0])]
            )
    if cutoff_up.shape[0] != n_up_expected:
        if cutoff_up.shape[0] > n_up_expected:
            cutoff_up = cutoff_up[:n_up_expected]
        else:
            pad_value = cutoff_profile[-1] if cutoff_profile.shape[0] > 0 else 1000.0
            cutoff_up = np.concatenate(
                [
                    cutoff_up,
                    np.full(n_up_expected - cutoff_up.shape[0], float(pad_value)),
                ]
            )

    nyquist_up = (sample_rate * oversample_factor) / 2.0
    cutoff_up = np.clip(cutoff_up, 20.0, nyquist_up * 0.98)

    filtered_up = apply_filter(
        signal_up,
        cutoff_profile=cutoff_up,
        sample_rate=sample_rate * oversample_factor,
        **asdict(fp),
    )

    filtered = resample_poly(
        filtered_up,
        up=1,
        down=oversample_factor,
        window=_OVERSAMPLE_RESAMPLE_WINDOW,
    ).astype(np.float64)

    if filtered.shape[0] != n_base:
        if filtered.shape[0] > n_base:
            filtered = filtered[:n_base]
        else:
            filtered = np.concatenate([filtered, np.zeros(n_base - filtered.shape[0])])
    return filtered


_NEUTRAL_VC_OFFSETS: dict[str, float] = {
    "attack_scale": 1.0,
    "release_scale": 1.0,
    "pulse_width_offset": 0.0,
    "resonance_offset_pct": 0.0,
    "softness_offset": 0.0,
    "drift_rate_offset_pct": 0.0,
}


def apply_voice_card(
    params: dict[str, Any],
    *,
    voice_card_spread: float,
    pitch_spread: float | None = None,
    filter_spread: float | None = None,
    envelope_spread: float | None = None,
    osc_spread: float | None = None,
    level_spread: float | None = None,
    freq_profile: np.ndarray,
    amp: float,
    cutoff_hz: float | None = None,
) -> tuple[np.ndarray, float, float | None, dict[str, float]]:
    """Apply deterministic per-voice calibration offsets.

    Returns ``(freq_profile, amp, cutoff_hz, extra_offsets)`` with voice card
    offsets applied.  *cutoff_hz* is passed through unchanged when ``None``
    (for engines without a filter).  *extra_offsets* is a dict of additional
    per-voice offsets that engines can optionally consume (attack/release
    scale, pulse width, resonance, softness, drift rate).

    *voice_card_spread* sets the global scale for all offset dimensions.
    Per-group overrides (*pitch_spread*, *filter_spread*, *envelope_spread*,
    *osc_spread*, *level_spread*) replace the global for their dimension
    when provided.  This lets you keep pitch tight for JI while opening up
    filter and envelope variation.
    """
    s_pitch = pitch_spread if pitch_spread is not None else voice_card_spread
    s_filter = filter_spread if filter_spread is not None else voice_card_spread
    s_env = envelope_spread if envelope_spread is not None else voice_card_spread
    s_osc = osc_spread if osc_spread is not None else voice_card_spread
    s_level = level_spread if level_spread is not None else voice_card_spread

    voice_name = str(params.get("_voice_name", ""))
    all_zero = (
        s_pitch <= 0 and s_filter <= 0 and s_env <= 0 and s_osc <= 0 and s_level <= 0
    )
    if all_zero or not voice_name:
        return freq_profile, amp, cutoff_hz, dict(_NEUTRAL_VC_OFFSETS)

    vc = voice_card_offsets(voice_name)

    freq_profile = freq_profile * (2.0 ** (vc["pitch_offset_cents"] * s_pitch / 1200.0))

    if cutoff_hz is not None:
        cutoff_hz = cutoff_hz * (2.0 ** (vc["cutoff_offset_cents"] * s_filter / 1200.0))

    amp = amp * (10.0 ** (vc["amp_offset_db"] * s_level / 20.0))

    extra_offsets = {
        "attack_scale": 1.0 + (vc["attack_scale"] - 1.0) * s_env,
        "release_scale": 1.0 + (vc["release_scale"] - 1.0) * s_env,
        "pulse_width_offset": vc["pulse_width_offset"] * s_osc,
        "resonance_offset_pct": vc["resonance_offset_pct"] * s_filter,
        "softness_offset": vc["softness_offset"] * s_osc,
        "drift_rate_offset_pct": vc["drift_rate_offset_pct"] * s_osc,
    }

    return freq_profile, amp, cutoff_hz, extra_offsets


def apply_analog_post_processing(
    signal: np.ndarray,
    *,
    rng: np.random.Generator,
    amp_jitter_db: float,
    noise_floor_level: float,
    sample_rate: int,
    n_samples: int,
) -> np.ndarray:
    """Apply noise floor and amp jitter -- shared by all analog-character engines.

    This is the common tail of the polyblep, filtered_stack, and fm render
    functions: envelope-following noise floor scaled by *noise_floor_level*,
    then per-note amplitude jitter.
    """
    if noise_floor_level > 0:
        noise = render_noise_floor(
            signal=signal,
            sample_rate=sample_rate,
            n_samples=n_samples,
            rng=rng,
            level=noise_floor_level,
        )
        signal = signal + noise
    return signal * 10.0 ** (amp_jitter_db / 20.0)


# ---------------------------------------------------------------------------
# Per-note analog character utilities
# ---------------------------------------------------------------------------


def apply_note_jitter(
    params: dict[str, Any],
    rng: np.random.Generator,
    jitter_amount: float = 1.0,
) -> dict[str, Any]:
    """Apply subtle per-note parameter variation. Returns new dict.

    ``jitter_amount=0`` returns a copy with no jitter. Default jitter ranges
    are scaled by *jitter_amount*.
    """
    result = dict(params)
    if jitter_amount <= 0:
        result["_amp_jitter_db"] = 0.0
        result["_phase_offset"] = 0.0
        return result

    if "cutoff_hz" in result and result["cutoff_hz"] > 0:
        result["cutoff_hz"] *= 1.0 + jitter_amount * rng.uniform(
            -_JITTER_CUTOFF_FRAC, _JITTER_CUTOFF_FRAC
        )
    if "filter_env_decay" in result and result["filter_env_decay"] > 0:
        result["filter_env_decay"] *= 1.0 + jitter_amount * rng.uniform(
            -_JITTER_DECAY_FRAC, _JITTER_DECAY_FRAC
        )
    if "resonance_q" in result and result.get("resonance_q", 0.707) > 0.707:
        result["resonance_q"] *= 1.0 + jitter_amount * rng.uniform(
            -_JITTER_Q_FRAC, _JITTER_Q_FRAC
        )
    if "attack" in result and result["attack"] > 0:
        result["attack"] *= 1.0 + jitter_amount * rng.uniform(
            -_JITTER_ATTACK_FRAC, _JITTER_ATTACK_FRAC
        )

    result["_amp_jitter_db"] = jitter_amount * rng.uniform(
        -_JITTER_AMP_DB, _JITTER_AMP_DB
    )
    result["_phase_offset"] = float(rng.uniform(0, 2.0 * np.pi))
    return result


def voice_card_offsets(voice_name: str) -> dict[str, float]:
    """Deterministic per-voice calibration offsets from voice name.

    Returns fixed offsets that persist across all notes in a voice.
    New offset dimensions are drawn after the original five so that
    existing per-voice character is preserved.
    """
    seed_bytes = sha256(f"voice_card:{voice_name}".encode()).digest()[:8]
    seed = int.from_bytes(seed_bytes, byteorder="big", signed=False)
    rng = np.random.default_rng(seed)
    offsets: dict[str, float] = {
        "cutoff_offset_cents": float(
            rng.uniform(-_VOICE_CARD_CUTOFF_CENTS, _VOICE_CARD_CUTOFF_CENTS)
        ),
        "attack_scale": float(
            rng.uniform(1.0 - _VOICE_CARD_ATTACK_SCALE, 1.0 + _VOICE_CARD_ATTACK_SCALE)
        ),
        "release_scale": float(
            rng.uniform(
                1.0 - _VOICE_CARD_RELEASE_SCALE, 1.0 + _VOICE_CARD_RELEASE_SCALE
            )
        ),
        "amp_offset_db": float(rng.uniform(-_VOICE_CARD_AMP_DB, _VOICE_CARD_AMP_DB)),
        "pitch_offset_cents": float(
            rng.uniform(-_VOICE_CARD_PITCH_CENTS, _VOICE_CARD_PITCH_CENTS)
        ),
    }
    offsets["pulse_width_offset"] = float(
        rng.uniform(-_VOICE_CARD_PULSE_WIDTH, _VOICE_CARD_PULSE_WIDTH)
    )
    offsets["resonance_offset_pct"] = float(
        rng.uniform(-_VOICE_CARD_RESONANCE_PCT, _VOICE_CARD_RESONANCE_PCT)
    )
    offsets["softness_offset"] = float(
        rng.uniform(-_VOICE_CARD_SOFTNESS, _VOICE_CARD_SOFTNESS)
    )
    offsets["drift_rate_offset_pct"] = float(
        rng.uniform(-_VOICE_CARD_DRIFT_RATE_PCT, _VOICE_CARD_DRIFT_RATE_PCT)
    )
    return offsets


# ---------------------------------------------------------------------------
# OB-Xd-style fast per-sample CV dither
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _one_pole_lowpass_inplace(raw: np.ndarray, a: float) -> np.ndarray:
    """Single-pole IIR lowpass: y[n] = b*x[n] + a*y[n-1], b = 1-a."""
    n = raw.shape[0]
    out = np.empty(n, dtype=np.float64)
    b = 1.0 - a
    y = 0.0
    for i in range(n):
        y = b * raw[i] + a * y
        out[i] = y
    return out


def fast_cv_dither(
    n_samples: int,
    *,
    amount: float,
    rng: np.random.Generator,
    lowpass_cutoff_hz: float = _CV_DITHER_LOWPASS_HZ,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Per-sample CV dither for OB-Xd-style fast pitch/cutoff wobble.

    Returns a length-``n_samples`` array with values approximately in
    ``[-amount, +amount]`` after a gentle one-pole lowpass so the dither is
    warm audible-band noise rather than pure white.  Callers scale the output
    by the desired full-scale range (semitones, fractional cutoff, etc.).

    The dither is a second layer on top of the stable ``voice_card_offsets``
    per-voice calibration: the stable layer shifts the voice's anchor, and
    the dither adds fast rustle around that anchor.  Deterministic given
    ``rng``.
    """
    if amount <= 0.0 or n_samples <= 0:
        return np.zeros(max(n_samples, 0), dtype=np.float64)
    raw = rng.uniform(-1.0, 1.0, size=n_samples).astype(np.float64)
    alpha = math.exp(-2.0 * math.pi * lowpass_cutoff_hz / max(sample_rate, 1))
    smoothed = _one_pole_lowpass_inplace(raw, alpha)
    return smoothed * amount


def apply_fast_cv_dither(
    freq_profile: np.ndarray,
    cutoff_profile: np.ndarray | None,
    *,
    amount: float,
    rng: np.random.Generator,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply per-sample pitch and cutoff dither to shaped profiles.

    ``amount`` is typically ``analog_jitter`` scaled by
    ``voice_card_spread`` — 0 disables, 1.0 gives +/-0.05 semitone pitch
    and +/-3% cutoff at full scale.  When ``cutoff_profile`` is ``None``
    (e.g., the FM engine) the cutoff dither is skipped.
    """
    if amount <= 0.0:
        return freq_profile, cutoff_profile
    n = freq_profile.shape[0]
    pitch_dither = fast_cv_dither(
        n,
        amount=amount * _CV_DITHER_PITCH_SEMITONES,
        rng=rng,
        sample_rate=sample_rate,
    )
    freq_profile = freq_profile * np.power(2.0, pitch_dither / 12.0)
    if cutoff_profile is not None:
        cutoff_dither = fast_cv_dither(
            n,
            amount=amount * _CV_DITHER_CUTOFF_FRAC,
            rng=rng,
            sample_rate=sample_rate,
        )
        cutoff_profile = cutoff_profile * (1.0 + cutoff_dither)
    return freq_profile, cutoff_profile


def apply_pitch_cv_dither(
    freq_profile: np.ndarray,
    *,
    analog_jitter: float,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    n_samples: int,
) -> np.ndarray:
    """Multiply ``freq_profile`` by an OB-Xd-style per-sample pitch dither.

    No-op when ``analog_jitter <= 0``.  Deterministically seeded per note via
    ``rng_for_note(extra_seed='cv_dither_pitch')`` — kept decorrelated from
    the sibling cutoff dither and from any other per-note RNG draw.
    """
    if analog_jitter <= 0.0:
        return freq_profile
    rng = rng_for_note(
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=sample_rate,
        extra_seed="cv_dither_pitch",
    )
    dither = fast_cv_dither(
        n_samples,
        amount=analog_jitter * _CV_DITHER_PITCH_SEMITONES,
        rng=rng,
        sample_rate=sample_rate,
    )
    return freq_profile * np.power(2.0, dither / 12.0)


def apply_cutoff_cv_dither(
    cutoff_profile: np.ndarray,
    *,
    analog_jitter: float,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    n_samples: int,
    nyquist: float,
) -> np.ndarray:
    """Multiply ``cutoff_profile`` by an OB-Xd-style per-sample cutoff dither.

    Clipped to ``[20, nyquist * 0.98]`` after modulation.  No-op when
    ``analog_jitter <= 0``.  Deterministically seeded per note via
    ``rng_for_note(extra_seed='cv_dither_cutoff')``.
    """
    if analog_jitter <= 0.0:
        return cutoff_profile
    rng = rng_for_note(
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=sample_rate,
        extra_seed="cv_dither_cutoff",
    )
    dither = fast_cv_dither(
        n_samples,
        amount=analog_jitter * _CV_DITHER_CUTOFF_FRAC,
        rng=rng,
        sample_rate=sample_rate,
    )
    return np.clip(cutoff_profile * (1.0 + dither), 20.0, nyquist * 0.98)


# ---------------------------------------------------------------------------
# Shared per-engine keytracked cutoff + voice-card post-offset helpers
# ---------------------------------------------------------------------------


def build_keytracked_cutoff_profile(
    *,
    cutoff_hz: float,
    keytrack: float,
    reference_freq_hz: float,
    filter_env_amount: float,
    filter_env_decay: float,
    duration: float,
    n_samples: int,
    freq_profile: np.ndarray,
    nyquist: float,
) -> np.ndarray:
    """Build a per-sample filter cutoff profile with exp-decay envelope + keytrack.

    Shared by the polyblep / filtered_stack / va engines — they all compute

        t = linspace(0, duration, n_samples)
        env = max(1 + filter_env_amount * exp(-t / filter_env_decay), 0.05)
        keytracked = cutoff_hz * (freq_profile / reference_freq_hz) ** keytrack
        profile = clip(keytracked * env, 20, nyquist * 0.98)

    with identical behavior in each engine.  The result is the "base" cutoff
    profile before any cutoff drift or CV dither is applied on top.
    """
    t = np.linspace(0.0, duration, n_samples, endpoint=False)
    env = np.maximum(
        1.0 + filter_env_amount * np.exp(-t / filter_env_decay),
        0.05,
    )
    keytracked = cutoff_hz * np.power(freq_profile / reference_freq_hz, keytrack)
    return np.clip(keytracked * env, 20.0, nyquist * 0.98)


def apply_voice_card_post_offsets(
    resonance_q: float,
    drift_rate_hz: float,
    vc_offsets: dict[str, float],
) -> tuple[float, float]:
    """Apply voice_card's resonance + drift-rate offsets.

    Shared by polyblep / filtered_stack / va — the common post-``apply_voice_card``
    coda where the voice-card spread's residual resonance and drift-rate
    percentages are folded into the final scalar values.  Enforces the
    ``resonance_q >= 0.5`` floor.

    Returns ``(resonance_q, drift_rate_hz)``.
    """
    resonance_q = max(
        0.5, resonance_q * (1.0 + vc_offsets["resonance_offset_pct"] / 100.0)
    )
    drift_rate_hz *= 1.0 + vc_offsets["drift_rate_offset_pct"] / 100.0
    return resonance_q, drift_rate_hz


# ---------------------------------------------------------------------------
# Scalar PolyBLEP step correction (for hard-sync / per-event discontinuities)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def apply_polyblep_step_correction(
    signal: np.ndarray,
    event_sample: int,
    event_fraction: float,
    step_amplitude: float,
) -> None:
    """In-place scalar PolyBLEP step correction at a single discontinuity.

    Replaces a vertical step of size ``step_amplitude`` (``y_post - y_pre``)
    with a bandlimited version using the standard 2-point PolyBLEP kernel.

    ``event_sample`` + ``event_fraction`` is the continuous-sample time of
    the event; ``event_fraction`` lies in ``[0, 1)`` and measures how far
    past ``event_sample`` the exact crossing occurred.  The correction is
    applied to ``signal[event_sample]`` (pre-event sample) and
    ``signal[event_sample + 1]`` (post-event sample).

    Kernel derivation mirrors the vectorized saw correction in
    :func:`code_musics.engines._oscillators.polyblep_saw`.  For arbitrary
    step ``d``:

        pre:  signal[k]   += (d / 2) * (1 - frac) ** 2
        post: signal[k+1] -= (d / 2) * (frac) ** 2
    """
    n = signal.shape[0]
    half_step = 0.5 * step_amplitude
    frac = event_fraction
    one_minus = 1.0 - frac
    if 0 <= event_sample < n:
        signal[event_sample] += half_step * one_minus * one_minus
    post_idx = event_sample + 1
    if 0 <= post_idx < n:
        signal[post_idx] -= half_step * frac * frac


# --- Shared drift bus --------------------------------------------------------
# DSP + builder live in code_musics.humanize next to DriftBusSpec. A top-level
# re-export of build_drift_bus keeps existing ``from _dsp_utils import ...``
# call sites working.
