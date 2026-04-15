"""Shared DSP utility functions for synth engines.

Extracted from piano.py, piano_additive.py, organ.py, and filtered_stack.py
to eliminate cross-engine duplication.
"""

from __future__ import annotations

import logging
import math
from hashlib import sha256
from typing import Any

import numba
import numpy as np

from code_musics.engines._filters import apply_zdf_svf

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
def _iir_lowpass_1pole(raw_noise: np.ndarray, alpha: float) -> np.ndarray:
    n = raw_noise.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[0] = raw_noise[0] * alpha
    one_minus_alpha = 1.0 - alpha
    for j in range(1, n):
        out[j] = alpha * raw_noise[j] + one_minus_alpha * out[j - 1]
    return out


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
        resonant = apply_zdf_svf(
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
    lp_wet = apply_zdf_svf(
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
) -> np.ndarray:
    """Bandpass-filter a noise signal around a center frequency via FFT."""
    if signal.size == 0:
        return signal

    nyquist = sample_rate / 2.0
    center_hz = float(np.clip(center_hz, 30.0, nyquist * 0.95))
    width_hz = max(80.0, center_hz * width_ratio)
    low_hz = max(20.0, center_hz - width_hz / 2.0)
    high_hz = min(nyquist * 0.98, center_hz + width_hz / 2.0)

    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    mask = np.exp(-0.5 * ((freqs - center_hz) / max(1.0, width_hz / 2.5)) ** 2)
    mask *= (freqs >= low_hz).astype(np.float64)
    mask *= (freqs <= high_hz).astype(np.float64)
    shaped = np.fft.irfft(spectrum * mask, n=signal.size)
    return shaped.real


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
# Shared analog character helpers
# ---------------------------------------------------------------------------


def extract_analog_params(params: dict[str, Any]) -> dict[str, float]:
    """Extract analog character parameters with defaults.

    Returns a flat dict of the five common analog-character knobs that
    polyblep, filtered_stack, and fm engines all share.
    """
    return {
        "pitch_drift": float(params.get("pitch_drift", 0.12)),
        "analog_jitter": float(params.get("analog_jitter", 1.0)),
        "noise_floor": float(params.get("noise_floor", 0.001)),
        "drift_rate_hz": float(params.get("drift_rate_hz", 0.3)),
        "cutoff_drift": float(params.get("cutoff_drift", 0.5)),
        "voice_card": float(params.get("voice_card", 1.0)),
    }


def apply_voice_card(
    params: dict[str, Any],
    *,
    voice_card_amount: float,
    freq_profile: np.ndarray,
    amp: float,
    cutoff_hz: float | None = None,
) -> tuple[np.ndarray, float, float | None]:
    """Apply deterministic per-voice calibration offsets.

    Returns ``(freq_profile, amp, cutoff_hz)`` with voice card offsets applied.
    *cutoff_hz* is passed through unchanged when ``None`` (for engines without
    a filter).
    """
    voice_name = str(params.get("_voice_name", ""))
    if voice_card_amount <= 0 or not voice_name:
        return freq_profile, amp, cutoff_hz

    vc = voice_card_offsets(voice_name)

    # Pitch offset applied to freq_profile (before drift)
    pitch_cents = vc["pitch_offset_cents"] * voice_card_amount
    freq_profile = freq_profile * (2.0 ** (pitch_cents / 1200.0))

    # Cutoff offset for engines with filters
    if cutoff_hz is not None:
        cutoff_cents = vc["cutoff_offset_cents"] * voice_card_amount
        cutoff_hz = cutoff_hz * (2.0 ** (cutoff_cents / 1200.0))

    # Amp offset
    amp = amp * (10.0 ** (vc["amp_offset_db"] * voice_card_amount / 20.0))

    return freq_profile, amp, cutoff_hz


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
    """
    seed_bytes = sha256(f"voice_card:{voice_name}".encode()).digest()[:8]
    seed = int.from_bytes(seed_bytes, byteorder="big", signed=False)
    rng = np.random.default_rng(seed)
    return {
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
