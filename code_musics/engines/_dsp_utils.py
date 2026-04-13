"""Shared DSP utility functions for synth engines.

Extracted from piano.py, piano_additive.py, organ.py, and filtered_stack.py
to eliminate cross-engine duplication.
"""

from __future__ import annotations

import logging
from hashlib import sha256

import numpy as np

from code_musics.engines._filters import apply_zdf_svf

logger: logging.Logger = logging.getLogger(__name__)

NYQUIST_FADE_START = 0.85
MAX_DRIFT_CENTS = 4.0
GOLDEN_RATIO_FRAC = 0.6180339887498949

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
) -> np.random.Generator:
    """Deterministic RNG seeded from note parameters.

    The organ engine passes ``extra_seed=str(sorted(params.items()))`` to
    include engine-specific params in the seed.
    """
    seed_material = repr(
        (
            round(freq, 6),
            round(duration, 6),
            round(amp, 6),
            sample_rate,
            extra_seed,
        )
    ).encode("utf-8")
    seed_bytes = sha256(seed_material).digest()[:8]
    seed = int.from_bytes(seed_bytes, byteorder="big", signed=False)
    return np.random.default_rng(seed)


def build_drift(
    *,
    n_samples: int,
    drift_amount: float,
    drift_rate_hz: float,
    duration: float,
    phase_offset: float,
) -> np.ndarray:
    """Build slow sinusoidal pitch drift, returns multiplicative trajectory."""
    if drift_amount <= 0 or n_samples == 0:
        return np.ones(n_samples, dtype=np.float64)

    t = np.linspace(0.0, duration, n_samples, endpoint=False, dtype=np.float64)
    max_cents = MAX_DRIFT_CENTS * drift_amount
    cents = max_cents * np.sin(2.0 * np.pi * drift_rate_hz * t + phase_offset)
    return np.power(2.0, cents / 1200.0)


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
) -> np.ndarray:
    """Add a subtle filtered noise floor that follows the amplitude envelope."""
    peak = np.max(np.abs(signal))
    if peak <= 0:
        return np.zeros(n_samples, dtype=np.float64)

    abs_signal = np.abs(signal)
    follower_coeff = np.exp(-1.0 / (0.02 * sample_rate))
    envelope = np.zeros(n_samples, dtype=np.float64)
    envelope[0] = abs_signal[0]
    for i in range(1, n_samples):
        envelope[i] = max(abs_signal[i], follower_coeff * envelope[i - 1])

    noise = rng.standard_normal(n_samples)
    noise = bandpass_noise(
        noise, sample_rate=sample_rate, center_hz=800.0, width_ratio=2.0
    )

    noise_peak = np.max(np.abs(noise))
    if noise_peak > 0:
        noise /= noise_peak

    return noise * envelope * 0.008


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
