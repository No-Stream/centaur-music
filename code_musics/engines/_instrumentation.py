"""Perceptual-measurement DSP primitives for effect-chain instrumentation.

These helpers extend the existing (flat-band / THD / crest) metrics in
``synth.py`` with measurements that correlate better with how a listener
hears effect-chain failures:

- :func:`a_weighted_spectrum_db` — IEC 61672 A-weighting applied to an
  FFT magnitude spectrum, for perceptually-weighted brightness comparisons.
- :func:`a_weighted_high_band_energy_db` — A-weighted mean energy in a
  specified band (defaults to 2-8 kHz for "papery" / brightness-creep
  detection).
- :func:`intermodulation_ratio` — two-tone IMD-to-harmonic-distortion
  ratio; catches "crunch" character that pure THD misses.
- :func:`detect_percussive_onsets` — simple envelope-derivative onset
  detector for per-hit transient analysis.
- :func:`per_hit_transient_metrics` — aggregates peak, crest, click-band
  energy, and 5 ms-vs-50 ms transient preservation across detected hits.

All functions operate on mono ``float64`` signals. Callers downmix stereo
via :func:`code_musics.synth.to_mono_reference` before passing signals in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.signal import find_peaks

logger: logging.Logger = logging.getLogger(__name__)


_EPS = 1e-12


# ---------------------------------------------------------------------------
# A-weighting — IEC 61672-1:2013 analog filter discretized via bilinear.
#
# Reference poles/zeros:
#   zeros: 0, 0, 0, 0 (four zeros at DC)
#   poles: 2pi * (20.598997, 20.598997, 107.65265, 737.86223, 12194.217,
#                 12194.217) rad/s
#   gain chosen so magnitude at 1 kHz equals 0 dB.
# ---------------------------------------------------------------------------


_A_WEIGHT_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _analog_a_weight_magnitude(freqs_hz: np.ndarray) -> np.ndarray:
    """Analog A-weighting magnitude response, evaluated at ``freqs_hz`` (linear)."""
    f = np.asarray(freqs_hz, dtype=np.float64)
    f2 = f * f
    numerator = (12194.217**2) * (f2 * f2)
    denominator = (
        (f2 + 20.598997**2)
        * np.sqrt((f2 + 107.65265**2) * (f2 + 737.86223**2))
        * (f2 + 12194.217**2)
    )
    magnitude = numerator / np.maximum(denominator, _EPS)
    # 2.00 dB normalization offset so response crosses 0 dB at 1 kHz.
    return magnitude * 1.2589254117941673  # 10**(2.00/20)


def a_weight_db(freqs_hz: np.ndarray) -> np.ndarray:
    """Return per-frequency A-weighting gain in dB (analog IEC 61672 reference)."""
    magnitude = _analog_a_weight_magnitude(freqs_hz)
    return 20.0 * np.log10(np.maximum(magnitude, _EPS))


def a_weighted_spectrum_db(
    freqs_hz: np.ndarray,
    magnitude_db: np.ndarray,
) -> np.ndarray:
    """Add A-weighting to a magnitude spectrum expressed in dB."""
    if freqs_hz.size == 0:
        return magnitude_db
    weight_db = a_weight_db(freqs_hz)
    return magnitude_db + weight_db


def a_weighted_mean_band_energy_db(
    freqs_hz: np.ndarray,
    magnitude_db: np.ndarray,
    *,
    low_hz: float,
    high_hz: float,
) -> float:
    """Mean A-weighted energy in dB over ``[low_hz, high_hz)``.

    Reuses the input's FFT magnitude (dB) so callers can keep a single
    spectrum per signal. Returns ``-inf`` when the band is empty.
    """
    if freqs_hz.size == 0 or magnitude_db.size == 0:
        return float("-inf")
    weighted = a_weighted_spectrum_db(freqs_hz, magnitude_db)
    mask = (freqs_hz >= low_hz) & (freqs_hz < high_hz)
    if not np.any(mask):
        return float("-inf")
    return float(np.mean(weighted[mask]))


# ---------------------------------------------------------------------------
# Intermodulation distortion — two-tone IMD-to-harmonic-distortion ratio.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IMDResult:
    """Result of an IMD probe on a signal.

    ``ratio`` is ``sum(IMD_product_energy) / sum(harmonic_product_energy)``.
    A value of 0.0 means no IMD (pure harmonic distortion); >1.0 means the
    IMD products dominate — "crunchy" character.
    ``detection`` is ``"two_tone"`` when two plausible tones were found,
    ``"single_tone"`` otherwise (in which case ratio is forced to 0.0).
    """

    ratio: float
    detection: str
    f1_hz: float
    f2_hz: float


def _peak_magnitude_linear(
    freqs_hz: np.ndarray,
    magnitude_linear: np.ndarray,
    center_hz: float,
    *,
    bin_tolerance: int,
) -> float:
    if center_hz <= 0.0 or freqs_hz.size < 2 or center_hz >= float(freqs_hz[-1]):
        return 0.0
    bin_spacing = float(freqs_hz[1] - freqs_hz[0])
    if bin_spacing <= 0.0:
        return 0.0
    center_idx = int(round((center_hz - float(freqs_hz[0])) / bin_spacing))
    lo = max(center_idx - bin_tolerance, 0)
    hi = min(center_idx + bin_tolerance + 1, magnitude_linear.size)
    if lo >= hi:
        return 0.0
    return float(np.max(magnitude_linear[lo:hi]))


def _pick_two_tones(
    freqs_hz: np.ndarray,
    magnitude_db: np.ndarray,
    *,
    min_hz: float = 40.0,
    min_separation_hz: float = 50.0,
    min_relative_db: float = -30.0,
) -> tuple[float, float]:
    """Return (f1_hz, f2_hz) for the two strongest peaks in the spectrum.

    Returns ``(0.0, 0.0)`` when no plausible pair can be found. ``f2`` is
    ``0.0`` when a second tone at least ``min_relative_db`` below the
    fundamental and separated by ``min_separation_hz`` cannot be located.
    """
    if freqs_hz.size < 4 or magnitude_db.size < 4:
        return 0.0, 0.0

    mask = freqs_hz >= min_hz
    if not np.any(mask):
        return 0.0, 0.0

    valid_freqs = freqs_hz[mask]
    valid_mag_db = magnitude_db[mask]
    if valid_mag_db.size < 3:
        return 0.0, 0.0

    strongest_idx = int(np.argmax(valid_mag_db))
    f1 = float(valid_freqs[strongest_idx])
    f1_db = float(valid_mag_db[strongest_idx])

    # Mask out the region around f1 and search for f2.
    sep_mask = np.abs(valid_freqs - f1) >= min_separation_hz
    if not np.any(sep_mask):
        return f1, 0.0
    secondary_mag_db = np.where(sep_mask, valid_mag_db, -np.inf)
    f2_idx = int(np.argmax(secondary_mag_db))
    f2_db = float(secondary_mag_db[f2_idx])
    if not np.isfinite(f2_db) or (f2_db - f1_db) < min_relative_db:
        return f1, 0.0

    return f1, float(valid_freqs[f2_idx])


def intermodulation_ratio(
    freqs_hz: np.ndarray,
    magnitude_db: np.ndarray,
    *,
    bin_tolerance: int = 2,
    max_harmonic: int = 10,
    max_imd_order: int = 3,
    f1_override_hz: float | None = None,
    f2_override_hz: float | None = None,
) -> IMDResult:
    """Measure IMD-vs-harmonic energy ratio from an averaged spectrum.

    Algorithm
    ---------
    1. Detect the two strongest in-band tones (or use overrides).
    2. If only one tone is present, return ``ratio=0.0`` and flag
       ``"single_tone"``.
    3. Sum power at integer harmonics of f1 and f2 up to ``max_harmonic``.
    4. Sum power at ``|m*f1 +/- n*f2|`` for ``m, n`` in
       ``[1, max_imd_order]`` excluding pure harmonic cases
       (``m==0 or n==0``).
    5. Return ``ratio = imd_energy / (harmonic_energy + eps)``.

    Returns
    -------
    :class:`IMDResult`
    """
    if freqs_hz.size < 4:
        return IMDResult(ratio=0.0, detection="single_tone", f1_hz=0.0, f2_hz=0.0)

    nyquist = float(freqs_hz[-1])
    magnitude_linear = np.power(10.0, magnitude_db / 20.0)

    if f1_override_hz is not None:
        f1 = float(f1_override_hz)
        f2 = float(f2_override_hz) if f2_override_hz is not None else 0.0
    else:
        f1, f2 = _pick_two_tones(freqs_hz, magnitude_db)

    if f1 <= 0.0:
        return IMDResult(ratio=0.0, detection="single_tone", f1_hz=0.0, f2_hz=0.0)
    if f2 <= 0.0:
        return IMDResult(ratio=0.0, detection="single_tone", f1_hz=f1, f2_hz=0.0)

    def _power(freq: float) -> float:
        amp = _peak_magnitude_linear(
            freqs_hz,
            magnitude_linear,
            freq,
            bin_tolerance=bin_tolerance,
        )
        return amp * amp

    harmonic_energy = 0.0
    for tone in (f1, f2):
        for h in range(2, max_harmonic + 1):
            target = h * tone
            if target >= nyquist:
                break
            harmonic_energy += _power(target)

    imd_energy = 0.0
    for m in range(0, max_imd_order + 1):
        for n in range(0, max_imd_order + 1):
            if m == 0 and n == 0:
                continue
            if m == 0 or n == 0:
                # Pure harmonic — already counted above.
                continue
            # Two IMD products per (m, n): m*f1 + n*f2 and |m*f1 - n*f2|.
            for sign in (1.0, -1.0):
                target = m * f1 + sign * n * f2
                if target <= 20.0 or target >= nyquist:
                    continue
                # Skip products that coincide with f1, f2, or their harmonics
                # to avoid double-counting.
                if _is_harmonic(target, f1, max_harmonic) or _is_harmonic(
                    target, f2, max_harmonic
                ):
                    continue
                imd_energy += _power(target)

    # Cap the ratio at 1e3 — anything beyond is noise-floor arithmetic, not
    # musical information. A cap keeps downstream warning logic stable.
    _RATIO_CAP = 1_000.0
    if harmonic_energy <= _EPS:
        ratio = _RATIO_CAP if imd_energy > 0.0 else 0.0
        return IMDResult(
            ratio=float(ratio),
            detection="two_tone",
            f1_hz=f1,
            f2_hz=f2,
        )
    ratio = float(min(imd_energy / harmonic_energy, _RATIO_CAP))
    return IMDResult(ratio=ratio, detection="two_tone", f1_hz=f1, f2_hz=f2)


def _is_harmonic(
    freq: float,
    fundamental: float,
    max_harmonic: int,
    *,
    tolerance_hz: float = 5.0,
) -> bool:
    if fundamental <= 0.0:
        return False
    for h in range(1, max_harmonic + 1):
        if abs(freq - h * fundamental) <= tolerance_hz:
            return True
    return False


# ---------------------------------------------------------------------------
# Per-hit transient diagnostics for percussive voices.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerHitSummary:
    """Aggregate per-hit transient metrics for a percussive signal."""

    hit_count: int
    onset_sample_indices: tuple[int, ...]
    transient_peak_ratio_p5: float
    transient_peak_ratio_p50: float
    transient_peak_ratio_p95: float
    hit_crest_db_p5: float
    hit_crest_db_p50: float
    hit_crest_db_p95: float
    hit_click_energy_db_p5: float
    hit_click_energy_db_p50: float
    hit_click_energy_db_p95: float


def _rms_envelope(
    signal: np.ndarray,
    *,
    sample_rate: int,
    window_ms: float = 5.0,
) -> np.ndarray:
    """Short-window running RMS envelope for onset detection."""
    if signal.size == 0:
        return signal
    win = max(4, int(round((window_ms / 1000.0) * sample_rate)))
    squared = np.asarray(signal, dtype=np.float64) ** 2
    kernel = np.ones(win, dtype=np.float64) / float(win)
    envelope = np.sqrt(np.maximum(np.convolve(squared, kernel, mode="same"), _EPS))
    return envelope


def detect_percussive_onsets(
    signal: np.ndarray,
    *,
    sample_rate: int,
    min_interval_ms: float = 80.0,
    prominence_db: float = 4.0,
    gate_db: float = -45.0,
    lookahead_ms: float = 8.0,
) -> np.ndarray:
    """Return the sample indices of plausible attack-points.

    Envelope-rise approach: compute a short RMS envelope in dBFS, then for
    each sample measure the rise from the minimum envelope in a
    ``lookahead_ms`` window before it to the current sample. Peaks of that
    rise function correspond to attacks. Tuned so a 4/4 kick at 120-140 BPM
    reliably yields one onset per beat on typical drum content without
    firing on sustained sine tones or flat noise.

    Parameters
    ----------
    min_interval_ms
        Minimum spacing between adjacent detections. 40 ms corresponds to
        ~1500 BPM hi-hat territory; drums at normal tempos easily clear it.
    prominence_db
        Minimum peak prominence in the envelope-rise signal (dB).
    gate_db
        Envelope gate: candidates whose envelope at the detection point
        sits below this level are discarded.
    lookahead_ms
        Window length used when computing the envelope rise. 8 ms captures
        kick-style attacks without smearing into sustained tones.
    """
    if signal.size == 0:
        return np.asarray([], dtype=np.int64)
    envelope = _rms_envelope(signal, sample_rate=sample_rate, window_ms=2.5)
    envelope_db = 20.0 * np.log10(np.maximum(envelope, _EPS))
    lookahead = max(2, int(round((lookahead_ms / 1000.0) * sample_rate)))
    # Rolling minimum of the envelope over the preceding ``lookahead`` samples.
    # Using minimum_filter1d with origin trailing so the window ends at the
    # current sample.
    rolling_min = minimum_filter1d(
        envelope_db,
        size=lookahead,
        mode="nearest",
        origin=lookahead // 2,
    )
    rise = np.maximum(0.0, envelope_db - rolling_min)

    distance = max(1, int(round((min_interval_ms / 1000.0) * sample_rate)))
    peak_indices, _ = find_peaks(
        rise,
        prominence=prominence_db,
        distance=distance,
    )
    if peak_indices.size == 0:
        return np.asarray([], dtype=np.int64)
    gated = np.asarray(
        [int(idx) for idx in peak_indices if envelope_db[idx] > gate_db],
        dtype=np.int64,
    )
    return gated


def per_hit_transient_metrics(
    signal: np.ndarray,
    *,
    sample_rate: int,
    onset_sample_indices: np.ndarray | None = None,
    hit_window_ms: float = 50.0,
    transient_window_ms: float = 5.0,
    click_band_low_hz: float = 2_000.0,
    click_band_high_hz: float = 5_000.0,
) -> PerHitSummary:
    """Aggregate per-hit transient metrics for a percussive voice signal."""
    if onset_sample_indices is None:
        onset_sample_indices = detect_percussive_onsets(signal, sample_rate=sample_rate)
    onset_sample_indices = np.asarray(onset_sample_indices, dtype=np.int64)
    if onset_sample_indices.size == 0 or signal.size == 0:
        return PerHitSummary(
            hit_count=0,
            onset_sample_indices=(),
            transient_peak_ratio_p5=0.0,
            transient_peak_ratio_p50=0.0,
            transient_peak_ratio_p95=0.0,
            hit_crest_db_p5=0.0,
            hit_crest_db_p50=0.0,
            hit_crest_db_p95=0.0,
            hit_click_energy_db_p5=float("-inf"),
            hit_click_energy_db_p50=float("-inf"),
            hit_click_energy_db_p95=float("-inf"),
        )

    hit_samples = max(8, int(round((hit_window_ms / 1000.0) * sample_rate)))
    transient_samples = max(4, int(round((transient_window_ms / 1000.0) * sample_rate)))

    ratios: list[float] = []
    crests_db: list[float] = []
    click_energies_db: list[float] = []

    mono = np.asarray(signal, dtype=np.float64)
    for onset in onset_sample_indices:
        onset = int(onset)
        end = min(onset + hit_samples, mono.size)
        if end - onset < transient_samples:
            continue
        hit_segment = mono[onset:end]
        transient_segment = hit_segment[:transient_samples]

        hit_peak = float(np.max(np.abs(hit_segment)))
        transient_peak = float(np.max(np.abs(transient_segment)))
        if hit_peak <= _EPS:
            continue
        ratios.append(transient_peak / hit_peak)

        rms = float(np.sqrt(np.mean(hit_segment * hit_segment)))
        if rms > _EPS and hit_peak > _EPS:
            crests_db.append(20.0 * np.log10(hit_peak / rms))

        # Click-band energy via goertzel-style FFT subset: simple rfft.
        window = np.hanning(hit_segment.size)
        spectrum = np.abs(np.fft.rfft(hit_segment * window))
        freqs = np.fft.rfftfreq(hit_segment.size, d=1.0 / sample_rate)
        mask = (freqs >= click_band_low_hz) & (freqs < click_band_high_hz)
        if np.any(mask):
            band_energy = float(np.mean(spectrum[mask] ** 2))
            click_energies_db.append(10.0 * np.log10(max(band_energy, _EPS)))

    if not ratios:
        return PerHitSummary(
            hit_count=0,
            onset_sample_indices=tuple(int(i) for i in onset_sample_indices),
            transient_peak_ratio_p5=0.0,
            transient_peak_ratio_p50=0.0,
            transient_peak_ratio_p95=0.0,
            hit_crest_db_p5=0.0,
            hit_crest_db_p50=0.0,
            hit_crest_db_p95=0.0,
            hit_click_energy_db_p5=float("-inf"),
            hit_click_energy_db_p50=float("-inf"),
            hit_click_energy_db_p95=float("-inf"),
        )

    ratio_array = np.asarray(ratios, dtype=np.float64)
    crest_array = np.asarray(crests_db, dtype=np.float64)
    click_array = (
        np.asarray(click_energies_db, dtype=np.float64)
        if click_energies_db
        else np.asarray([float("-inf")], dtype=np.float64)
    )

    return PerHitSummary(
        hit_count=len(ratios),
        onset_sample_indices=tuple(int(i) for i in onset_sample_indices),
        transient_peak_ratio_p5=float(np.percentile(ratio_array, 5.0)),
        transient_peak_ratio_p50=float(np.percentile(ratio_array, 50.0)),
        transient_peak_ratio_p95=float(np.percentile(ratio_array, 95.0)),
        hit_crest_db_p5=float(np.percentile(crest_array, 5.0))
        if crest_array.size
        else 0.0,
        hit_crest_db_p50=float(np.percentile(crest_array, 50.0))
        if crest_array.size
        else 0.0,
        hit_crest_db_p95=float(np.percentile(crest_array, 95.0))
        if crest_array.size
        else 0.0,
        hit_click_energy_db_p5=float(np.percentile(click_array, 5.0))
        if np.isfinite(click_array).any()
        else float("-inf"),
        hit_click_energy_db_p50=float(np.percentile(click_array, 50.0))
        if np.isfinite(click_array).any()
        else float("-inf"),
        hit_click_energy_db_p95=float(np.percentile(click_array, 95.0))
        if np.isfinite(click_array).any()
        else float("-inf"),
    )
