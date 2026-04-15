"""Additive synthesis engine."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt

from code_musics.engines._dsp_utils import nyquist_fade, rng_for_note
from code_musics.engines._envelopes import parse_envelope_points, render_envelope
from code_musics.spectra import formant_weight, harmonic_spectrum, vowel_formants
from code_musics.tuning import tenney_height

_SPECTRAL_DRIFT_RATE_HZ = 0.11
_SPECTRAL_DRIFT_RATE_STEP_HZ = 0.013
_DECAY_TILT_STRENGTH = 4.0
_NOISE_AMOUNT_EPSILON = 1e-6

_DEFAULT_GRAVITY_TARGETS = [1.0, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 7 / 4, 2.0]


def render(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
    freq_trajectory: np.ndarray | None = None,
) -> np.ndarray:
    """Render a richer additive voice with compatibility defaults."""
    n_harmonics = int(params.get("n_harmonics", 6))
    harmonic_rolloff = float(params.get("harmonic_rolloff", 0.5))
    brightness_tilt = float(params.get("brightness_tilt", 0.0))
    odd_even_balance = float(params.get("odd_even_balance", 0.0))
    detune_cents = float(params.get("detune_cents", 0.0))
    unison_voices = max(1, int(params.get("unison_voices", 1)))

    noise_amount = float(params.get("noise_amount", 0.0))
    noise_bandwidth_hz = float(params.get("noise_bandwidth_hz", 60.0))

    if noise_amount < 0.0 or noise_amount > 1.0:
        raise ValueError("noise_amount must be between 0.0 and 1.0")
    if noise_bandwidth_hz <= 0.0:
        raise ValueError("noise_bandwidth_hz must be positive")

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0)

    if freq_trajectory is not None:
        freq_trajectory = np.asarray(freq_trajectory, dtype=np.float64)
        if freq_trajectory.ndim != 1:
            raise ValueError("freq_trajectory must be one-dimensional")
        if freq_trajectory.size != n_samples:
            raise ValueError("freq_trajectory length must match note duration")

    base_freq_trajectory = (
        np.full(n_samples, float(freq), dtype=np.float64)
        if freq_trajectory is None
        else freq_trajectory
    )
    partials_param = params.get("partials")
    attack_partials_param = params.get("attack_partials")
    if partials_param is None and attack_partials_param is None:
        signal = np.zeros(n_samples, dtype=np.float64)
        voice_detunes = _unison_detunes(unison_voices, detune_cents)

        for detune_offset_cents in voice_detunes:
            voice_freq = freq * (2.0 ** (detune_offset_cents / 1200.0))
            if freq_trajectory is None:
                t = np.linspace(0.0, duration, n_samples, endpoint=False)
                signal += _render_partial_bank(
                    t=t,
                    freq=voice_freq,
                    sample_rate=sample_rate,
                    n_harmonics=n_harmonics,
                    harmonic_rolloff=harmonic_rolloff,
                    brightness_tilt=brightness_tilt,
                    odd_even_balance=odd_even_balance,
                )
            else:
                signal += _render_partial_bank_with_trajectory(
                    freq_trajectory=freq_trajectory
                    * (2.0 ** (detune_offset_cents / 1200.0)),
                    sample_rate=sample_rate,
                    n_harmonics=n_harmonics,
                    harmonic_rolloff=harmonic_rolloff,
                    brightness_tilt=brightness_tilt,
                    odd_even_balance=odd_even_balance,
                )

        signal /= float(len(voice_detunes))
        peak = np.max(np.abs(signal))
        if peak > 0:
            signal = signal / peak
        return amp * signal

    sustain_partials = _normalize_partials(
        partials_param
        if partials_param is not None
        else harmonic_spectrum(
            n_partials=n_harmonics,
            harmonic_rolloff=harmonic_rolloff,
            brightness_tilt=brightness_tilt,
            odd_even_balance=odd_even_balance,
        )
    )
    attack_partials = (
        None
        if attack_partials_param is None
        else _normalize_partials(attack_partials_param)
    )

    # --- Render-time formant shaping ---
    formant_param = params.get("formant")
    formant_bw = float(params.get("formant_bandwidth_hz", 100.0))
    if formant_param is not None:
        if isinstance(formant_param, str):
            resolved_formants = [
                (center, gain, formant_bw)
                for center, gain, _bw in vowel_formants(formant_param)
            ]
        else:
            resolved_formants = list(formant_param)
        sustain_partials = [dict(p) for p in sustain_partials]
        for partial in sustain_partials:
            abs_freq = freq * partial["ratio"]
            weight = formant_weight(abs_freq, resolved_formants)
            partial["amp"] *= weight
        if attack_partials is not None:
            attack_partials = [dict(p) for p in attack_partials]
            for partial in attack_partials:
                abs_freq = freq * partial["ratio"]
                weight = formant_weight(abs_freq, resolved_formants)
                partial["amp"] *= weight

    spectral_morph_time = max(0.0, float(params.get("spectral_morph_time", 0.0)))
    partial_decay_tilt = max(0.0, float(params.get("partial_decay_tilt", 0.0)))
    upper_partial_drift_cents = max(
        0.0, float(params.get("upper_partial_drift_cents", 0.0))
    )
    upper_partial_drift_min_ratio = max(
        1.0, float(params.get("upper_partial_drift_min_ratio", 2.0))
    )

    # --- Spectral gravity params ---
    spectral_gravity = max(0.0, min(1.0, float(params.get("spectral_gravity", 0.0))))
    gravity_rate = max(0.0, float(params.get("gravity_rate", 1.0)))
    gravity_targets_param = params.get("gravity_targets")
    gravity_targets: list[float] = (
        [float(t) for t in gravity_targets_param]
        if gravity_targets_param is not None
        else _DEFAULT_GRAVITY_TARGETS
    )

    # --- Spectral flicker params ---
    spectral_flicker = max(0.0, min(1.0, float(params.get("spectral_flicker", 0.0))))
    flicker_rate_hz = max(0.01, float(params.get("flicker_rate_hz", 3.0)))
    flicker_correlation = max(
        0.0, min(1.0, float(params.get("flicker_correlation", 0.3)))
    )

    spectral_partials = _build_spectral_partials(
        sustain_partials=sustain_partials,
        attack_partials=attack_partials,
        n_samples=n_samples,
        duration=duration,
        partial_decay_tilt=partial_decay_tilt,
        spectral_morph_time=spectral_morph_time,
        spectral_flicker=spectral_flicker,
        flicker_rate_hz=flicker_rate_hz,
        flicker_correlation=flicker_correlation,
        sample_rate=sample_rate,
        freq=freq,
        amp=amp,
    )

    signal = np.zeros(n_samples, dtype=np.float64)
    voice_detunes = _unison_detunes(unison_voices, detune_cents)
    nyquist_hz = sample_rate / 2.0

    for partial_index, partial in enumerate(spectral_partials):
        gravity_mult = _gravity_ratio_trajectory(
            ratio=partial["ratio"],
            n_samples=n_samples,
            duration=duration,
            gravity_strength=spectral_gravity,
            gravity_rate=gravity_rate,
            gravity_targets=gravity_targets,
        )
        drift_mult = _drift_ratio_trajectory(
            ratio=partial["ratio"],
            partial_index=partial_index,
            n_samples=n_samples,
            duration=duration,
            drift_cents=upper_partial_drift_cents,
            drift_min_ratio=upper_partial_drift_min_ratio,
        )
        base_partial_trajectory = (
            base_freq_trajectory * partial["ratio"] * drift_mult * gravity_mult
        )
        for detune_offset_cents in voice_detunes:
            detune_ratio = 2.0 ** (detune_offset_cents / 1200.0)
            partial_freq_trajectory = base_partial_trajectory * detune_ratio
            if np.min(partial_freq_trajectory) >= nyquist_hz:
                continue
            anti_alias_weight = nyquist_fade(partial_freq_trajectory, nyquist_hz)
            if np.max(anti_alias_weight) <= 0.0:
                continue

            phase = np.cumsum(
                np.concatenate(
                    [
                        np.zeros(1, dtype=np.float64),
                        2.0 * np.pi * partial_freq_trajectory[:-1] / float(sample_rate),
                    ]
                )
            )
            partial_weight = partial["amp_trajectory"] * anti_alias_weight
            signal += partial_weight * np.sin(phase)

    signal /= float(len(voice_detunes))

    # Add noise bands if requested (before peak normalization).
    noise_bands = _render_noise_bands(
        partials=sustain_partials,
        freq=freq,
        n_samples=n_samples,
        sample_rate=sample_rate,
        noise_amount=noise_amount,
        noise_bandwidth_hz=noise_bandwidth_hz,
        freq_trajectory=freq_trajectory,
        rng=rng_for_note(
            freq=freq, duration=duration, amp=amp, sample_rate=sample_rate
        ),
    )
    signal += noise_bands

    peak = np.max(np.abs(signal))
    if peak <= 0.0:
        raise ValueError("spectral additive params produced no audible partials")
    return amp * (signal / peak)


_PARTIAL_OPTIONAL_KEYS = ("noise", "noise_bw", "envelope")


def _normalize_partials(partials: Any) -> list[dict[str, Any]]:
    if not isinstance(partials, list) or len(partials) == 0:
        raise ValueError("partials must be a non-empty list of ratio/amp dicts")

    normalized: list[dict[str, Any]] = []
    for entry in partials:
        if not isinstance(entry, dict):
            raise ValueError("each partial must be a dict with ratio and amp")
        if "ratio" not in entry or "amp" not in entry:
            raise ValueError("each partial must define ratio and amp")
        ratio = float(entry["ratio"])
        amp = float(entry["amp"])
        if ratio <= 0.0:
            raise ValueError("partial ratios must be strictly positive")
        if amp < 0.0:
            raise ValueError("partial amplitudes must be non-negative")
        item: dict[str, Any] = {"ratio": ratio, "amp": amp}
        for key in _PARTIAL_OPTIONAL_KEYS:
            if key in entry:
                item[key] = entry[key]
        # Validate envelope format eagerly if present.
        if "envelope" in item:
            parse_envelope_points(item["envelope"])
        normalized.append(item)

    if not any(entry["amp"] > 0.0 for entry in normalized):
        raise ValueError("partials must include at least one positive amplitude")
    return normalized


def _render_noise_bands(
    *,
    partials: list[dict[str, Any]],
    freq: float,
    n_samples: int,
    sample_rate: int,
    noise_amount: float,
    noise_bandwidth_hz: float,
    freq_trajectory: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render narrow noise bands centered on each partial's frequency.

    Uses ring modulation: lowpass-filtered white noise multiplied by the
    partial's sine carrier.  This naturally places a noise band centered at
    each partial frequency with bandwidth approximately equal to
    ``noise_bandwidth_hz`` (or per-partial ``noise_bw`` override).
    """
    result = np.zeros(n_samples, dtype=np.float64)
    nyquist_hz = sample_rate / 2.0

    # Check if any partial has noise — skip early if nothing to do.
    has_any_noise = noise_amount > _NOISE_AMOUNT_EPSILON or any(
        p.get("noise", 0.0) > _NOISE_AMOUNT_EPSILON for p in partials
    )
    if not has_any_noise:
        return result

    # Generate a single shared white noise buffer; each partial gets its own
    # filtered copy via ring modulation with the partial's carrier.
    base_noise = rng.standard_normal(n_samples)
    t = np.linspace(
        0.0, n_samples / sample_rate, n_samples, endpoint=False, dtype=np.float64
    )

    sos_cache: dict[float, np.ndarray] = {}
    for partial in partials:
        partial_noise_level = float(partial.get("noise", noise_amount))
        if partial_noise_level <= _NOISE_AMOUNT_EPSILON:
            continue

        partial_bw = float(partial.get("noise_bw", noise_bandwidth_hz))
        partial_freq_hz = freq * partial["ratio"]

        # Skip partials at or above Nyquist.
        if partial_freq_hz >= nyquist_hz:
            continue

        # Lowpass filter the noise at half the bandwidth to create baseband noise.
        lp_cutoff = min(partial_bw / 2.0, nyquist_hz * 0.95)
        if lp_cutoff < 1.0:
            continue

        if lp_cutoff not in sos_cache:
            coeffs: np.ndarray = np.asarray(
                butter(2, lp_cutoff, btype="low", fs=sample_rate, output="sos")
            )
            sos_cache[lp_cutoff] = coeffs
        sos = sos_cache[lp_cutoff]
        filtered_noise: np.ndarray = np.asarray(sosfilt(sos, base_noise))

        # Ring-modulate: multiply by 2x the partial's carrier sine.
        if freq_trajectory is not None:
            carrier_freq_traj = freq_trajectory * partial["ratio"]
            phase = np.cumsum(
                np.concatenate(
                    [
                        np.zeros(1, dtype=np.float64),
                        2.0 * np.pi * carrier_freq_traj[:-1] / float(sample_rate),
                    ]
                )
            )
            carrier = np.sin(phase)
        else:
            carrier = np.sin(2.0 * np.pi * partial_freq_hz * t)

        noise_band = 2.0 * filtered_noise * carrier
        noise_band *= partial["amp"] * partial_noise_level
        result += noise_band

    return result


def _lowpass_noise(
    *,
    n_samples: int,
    sample_rate: int,
    cutoff_hz: float,
    rng: np.random.Generator,
    sos_cache: dict[float, np.ndarray] | None = None,
) -> np.ndarray:
    """Generate lowpass-filtered white noise, normalized to roughly [-1, 1]."""
    raw = rng.standard_normal(n_samples)
    nyquist = sample_rate / 2.0
    safe_cutoff = min(cutoff_hz, nyquist * 0.95)
    if safe_cutoff <= 0.0 or n_samples < 4:
        return raw
    cache_key = safe_cutoff / nyquist
    if sos_cache is not None and cache_key in sos_cache:
        sos = sos_cache[cache_key]
    else:
        sos_coeffs: np.ndarray = np.asarray(
            butter(2, cache_key, btype="low", output="sos")
        )
        sos = sos_coeffs
        if sos_cache is not None:
            sos_cache[cache_key] = sos_coeffs
    filtered: np.ndarray = np.asarray(sosfilt(sos, raw), dtype=np.float64)
    peak = np.max(np.abs(filtered))
    if peak > 0:
        filtered /= peak
    return filtered


def _build_spectral_partials(
    *,
    sustain_partials: list[dict[str, Any]],
    attack_partials: list[dict[str, Any]] | None,
    n_samples: int,
    duration: float,
    partial_decay_tilt: float,
    spectral_morph_time: float,
    spectral_flicker: float = 0.0,
    flicker_rate_hz: float = 3.0,
    flicker_correlation: float = 0.3,
    sample_rate: int = 44100,
    freq: float = 220.0,
    amp: float = 0.5,
) -> list[dict[str, Any]]:
    ratio_union = sorted(
        {partial["ratio"] for partial in sustain_partials}
        | (
            {partial["ratio"] for partial in attack_partials}
            if attack_partials is not None
            else set()
        )
    )
    if not ratio_union:
        raise ValueError("spectral additive requires at least one partial ratio")

    sustain_map = {partial["ratio"]: partial["amp"] for partial in sustain_partials}
    attack_map = (
        {}
        if attack_partials is None
        else {partial["ratio"]: partial["amp"] for partial in attack_partials}
    )
    # Per-partial envelope lookup: only sustain_partials carry envelopes.
    envelope_map: dict[float, list[dict]] = {
        partial["ratio"]: partial["envelope"]
        for partial in sustain_partials
        if "envelope" in partial
    }

    note_progress = np.linspace(0.0, 1.0, n_samples, endpoint=False, dtype=np.float64)
    if spectral_morph_time > 0.0 and attack_partials is not None:
        morph_progress = np.clip(
            np.arange(n_samples, dtype=np.float64)
            / max(1.0, spectral_morph_time * n_samples / max(duration, 1e-12)),
            0.0,
            1.0,
        )
    else:
        morph_progress = np.ones(n_samples, dtype=np.float64)

    max_ratio = max(ratio_union)
    min_ratio = min(ratio_union)
    ratio_span = max(max_ratio - min_ratio, 1e-12)

    # Pre-compute flicker modulation if active.
    flicker_shared: np.ndarray | None = None
    flicker_rng: np.random.Generator | None = None
    flicker_sos_cache: dict[float, np.ndarray] = {}
    if spectral_flicker > 0.0 and n_samples > 0:
        flicker_rng = rng_for_note(
            freq=freq,
            duration=duration,
            amp=amp,
            sample_rate=sample_rate,
            extra_seed="spectral_flicker",
        )
        flicker_shared = _lowpass_noise(
            n_samples=n_samples,
            sample_rate=sample_rate,
            cutoff_hz=flicker_rate_hz,
            rng=flicker_rng,
            sos_cache=flicker_sos_cache,
        )

    partials: list[dict[str, Any]] = []
    for ratio in ratio_union:
        attack_amp = attack_map.get(ratio, 0.0)
        sustain_amp = sustain_map.get(ratio, 0.0)
        amp_trajectory = attack_amp + ((sustain_amp - attack_amp) * morph_progress)
        if partial_decay_tilt > 0.0 and max_ratio > min_ratio:
            decay_strength = (ratio - min_ratio) / ratio_span
            amp_trajectory = amp_trajectory * np.exp(
                -partial_decay_tilt
                * _DECAY_TILT_STRENGTH
                * note_progress
                * decay_strength
            )
        if ratio in envelope_map:
            env_curve = render_envelope(
                envelope_map[ratio], n_samples, default_value=1.0
            )
            amp_trajectory = amp_trajectory * env_curve

        # Apply spectral flicker modulation (after morph, decay_tilt, envelope).
        if (
            spectral_flicker > 0.0
            and flicker_shared is not None
            and flicker_rng is not None
        ):
            independent_noise = _lowpass_noise(
                n_samples=n_samples,
                sample_rate=sample_rate,
                cutoff_hz=flicker_rate_hz,
                rng=flicker_rng,
                sos_cache=flicker_sos_cache,
            )
            blended = (
                flicker_correlation * flicker_shared
                + (1.0 - flicker_correlation) * independent_noise
            )
            amp_mod = (1.0 - spectral_flicker) + spectral_flicker * (
                0.5 + 0.5 * blended
            )
            amp_trajectory = amp_trajectory * amp_mod

        partials.append(
            {
                "ratio": ratio,
                "amp_trajectory": amp_trajectory.astype(np.float64),
            }
        )
    return partials


def _gravity_ratio_trajectory(
    *,
    ratio: float,
    n_samples: int,
    duration: float,
    gravity_strength: float,
    gravity_rate: float,
    gravity_targets: list[float],
) -> np.ndarray:
    """Per-partial frequency multiplier that drifts toward nearby just intervals.

    Returns a per-sample array (1.0 = no change). Simpler intervals (lower Tenney
    height) attract more strongly. Octave equivalents up to ratio 4.0 are searched.
    """
    if gravity_strength <= 0.0 or not gravity_targets:
        return np.ones(n_samples, dtype=np.float64)

    # Build expanded attractor set with octave equivalents up to ratio 4.0.
    expanded_attractors: list[float] = []
    for target in gravity_targets:
        if target <= 0.0:
            continue
        r = target
        while r > 2.0:
            r /= 2.0
        while r < 1.0:
            r *= 2.0
        while r <= 4.0:
            expanded_attractors.append(r)
            r *= 2.0

    if not expanded_attractors:
        return np.ones(n_samples, dtype=np.float64)

    # Find nearest attractor in cents space.
    ratio_cents = 1200.0 * math.log2(ratio) if ratio > 0 else 0.0
    best_attractor = expanded_attractors[0]
    best_distance_cents = abs(ratio_cents - 1200.0 * math.log2(expanded_attractors[0]))
    for attractor in expanded_attractors[1:]:
        dist = abs(ratio_cents - 1200.0 * math.log2(attractor))
        if dist < best_distance_cents:
            best_distance_cents = dist
            best_attractor = attractor

    # If already at the attractor, no drift needed.
    if abs(ratio - best_attractor) < 1e-10:
        return np.ones(n_samples, dtype=np.float64)

    # Attraction weight from Tenney height (simpler = stronger attraction).
    attraction_weight = 1.0 / (1.0 + tenney_height(best_attractor))

    # Exponential approach: ratio(t) = attractor + (r - attractor) * exp(-k*t)
    t = np.linspace(0.0, duration, n_samples, endpoint=False, dtype=np.float64)
    decay_rate = gravity_strength * gravity_rate * attraction_weight
    approached_ratio = best_attractor + (ratio - best_attractor) * np.exp(
        -decay_rate * t
    )

    return approached_ratio / ratio


def _drift_ratio_trajectory(
    *,
    ratio: float,
    partial_index: int,
    n_samples: int,
    duration: float,
    drift_cents: float,
    drift_min_ratio: float,
) -> np.ndarray:
    if drift_cents <= 0.0 or ratio < drift_min_ratio:
        return np.ones(n_samples, dtype=np.float64)

    t = np.linspace(0.0, duration, n_samples, endpoint=False, dtype=np.float64)
    rate_hz = _SPECTRAL_DRIFT_RATE_HZ + (partial_index * _SPECTRAL_DRIFT_RATE_STEP_HZ)
    phase = ratio * np.pi * 0.37
    drift_strength = 1.0 - (drift_min_ratio / ratio)
    cents = drift_cents * drift_strength * np.sin((2.0 * np.pi * rate_hz * t) + phase)
    return np.power(2.0, cents / 1200.0)


def _render_partial_bank(
    *,
    t: np.ndarray,
    freq: float,
    sample_rate: int,
    n_harmonics: int,
    harmonic_rolloff: float,
    brightness_tilt: float,
    odd_even_balance: float,
) -> np.ndarray:
    signal = np.zeros_like(t, dtype=np.float64)
    total_amp = 0.0
    clamped_odd_even_balance = max(-0.95, min(0.95, odd_even_balance))
    nyquist_hz = sample_rate / 2.0

    for harmonic_index in range(1, n_harmonics + 1):
        partial_freq = freq * harmonic_index
        if partial_freq >= nyquist_hz:
            break

        partial_amp = harmonic_rolloff ** (harmonic_index - 1)
        if brightness_tilt != 0.0:
            partial_amp *= harmonic_index**brightness_tilt

        if harmonic_index % 2 == 0:
            partial_amp *= 1.0 - clamped_odd_even_balance
        else:
            partial_amp *= 1.0 + clamped_odd_even_balance

        if partial_amp <= 0:
            continue

        anti_alias_weight = float(
            nyquist_fade(np.array([partial_freq], dtype=np.float64), nyquist_hz)[0]
        )
        if anti_alias_weight <= 0.0:
            continue

        weighted_partial_amp = partial_amp * anti_alias_weight
        signal += weighted_partial_amp * np.sin(2.0 * np.pi * partial_freq * t)
        total_amp += weighted_partial_amp

    if total_amp == 0.0:
        return signal
    return signal / total_amp


def _render_partial_bank_with_trajectory(
    *,
    freq_trajectory: np.ndarray,
    sample_rate: int,
    n_harmonics: int,
    harmonic_rolloff: float,
    brightness_tilt: float,
    odd_even_balance: float,
) -> np.ndarray:
    if freq_trajectory.ndim != 1:
        raise ValueError("freq_trajectory must be one-dimensional")
    if freq_trajectory.size == 0:
        return np.zeros(0)

    signal = np.zeros_like(freq_trajectory, dtype=np.float64)
    total_amp = 0.0
    clamped_odd_even_balance = max(-0.95, min(0.95, odd_even_balance))
    nyquist_hz = sample_rate / 2.0

    for harmonic_index in range(1, n_harmonics + 1):
        partial_freq_trajectory = freq_trajectory * harmonic_index
        if np.min(partial_freq_trajectory) >= nyquist_hz:
            break

        partial_amp = harmonic_rolloff ** (harmonic_index - 1)
        if brightness_tilt != 0.0:
            partial_amp *= harmonic_index**brightness_tilt

        if harmonic_index % 2 == 0:
            partial_amp *= 1.0 - clamped_odd_even_balance
        else:
            partial_amp *= 1.0 + clamped_odd_even_balance

        if partial_amp <= 0:
            continue

        anti_alias_weight = nyquist_fade(partial_freq_trajectory, nyquist_hz)
        if np.max(anti_alias_weight) <= 0.0:
            continue

        phase = np.cumsum(
            np.concatenate(
                [
                    np.zeros(1, dtype=np.float64),
                    2.0 * np.pi * partial_freq_trajectory[:-1] / sample_rate,
                ]
            )
        )
        partial_weight = partial_amp * anti_alias_weight
        signal += partial_weight * np.sin(phase)
        total_amp += float(np.mean(partial_weight))

    if total_amp == 0.0:
        return signal
    return signal / total_amp


def _unison_detunes(unison_voices: int, detune_cents: float) -> list[float]:
    if unison_voices <= 1 or detune_cents == 0.0:
        return [0.0]
    if unison_voices == 2:
        return [-detune_cents / 2.0, detune_cents / 2.0]

    return [
        ((voice_index / (unison_voices - 1)) - 0.5) * detune_cents
        for voice_index in range(unison_voices)
    ]
