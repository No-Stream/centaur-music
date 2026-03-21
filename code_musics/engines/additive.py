"""Additive synthesis engine."""

from __future__ import annotations

from typing import Any

import numpy as np

from code_musics.spectra import harmonic_spectrum

_NYQUIST_FADE_START = 0.85
_SPECTRAL_DRIFT_RATE_HZ = 0.11
_SPECTRAL_DRIFT_RATE_STEP_HZ = 0.013
_DECAY_TILT_STRENGTH = 4.0


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
    spectral_morph_time = max(0.0, float(params.get("spectral_morph_time", 0.0)))
    partial_decay_tilt = max(0.0, float(params.get("partial_decay_tilt", 0.0)))
    upper_partial_drift_cents = max(
        0.0, float(params.get("upper_partial_drift_cents", 0.0))
    )
    upper_partial_drift_min_ratio = max(
        1.0, float(params.get("upper_partial_drift_min_ratio", 2.0))
    )

    spectral_partials = _build_spectral_partials(
        sustain_partials=sustain_partials,
        attack_partials=attack_partials,
        n_samples=n_samples,
        duration=duration,
        partial_decay_tilt=partial_decay_tilt,
        spectral_morph_time=spectral_morph_time,
    )

    signal = np.zeros(n_samples, dtype=np.float64)
    voice_detunes = _unison_detunes(unison_voices, detune_cents)
    nyquist_hz = sample_rate / 2.0

    for detune_offset_cents in voice_detunes:
        detune_ratio = 2.0 ** (detune_offset_cents / 1200.0)
        for partial_index, partial in enumerate(spectral_partials):
            partial_freq_trajectory = (
                base_freq_trajectory
                * detune_ratio
                * partial["ratio"]
                * _drift_ratio_trajectory(
                    ratio=partial["ratio"],
                    partial_index=partial_index,
                    n_samples=n_samples,
                    duration=duration,
                    drift_cents=upper_partial_drift_cents,
                    drift_min_ratio=upper_partial_drift_min_ratio,
                )
            )
            if np.min(partial_freq_trajectory) >= nyquist_hz:
                continue
            anti_alias_weight = _nyquist_fade(partial_freq_trajectory, nyquist_hz)
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
    peak = np.max(np.abs(signal))
    if peak <= 0.0:
        raise ValueError("spectral additive params produced no audible partials")
    return amp * (signal / peak)


def _normalize_partials(partials: Any) -> list[dict[str, float]]:
    if not isinstance(partials, list) or len(partials) == 0:
        raise ValueError("partials must be a non-empty list of ratio/amp dicts")

    normalized: list[dict[str, float]] = []
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
        normalized.append({"ratio": ratio, "amp": amp})

    if not any(entry["amp"] > 0.0 for entry in normalized):
        raise ValueError("partials must include at least one positive amplitude")
    return normalized


def _build_spectral_partials(
    *,
    sustain_partials: list[dict[str, float]],
    attack_partials: list[dict[str, float]] | None,
    n_samples: int,
    duration: float,
    partial_decay_tilt: float,
    spectral_morph_time: float,
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
        partials.append(
            {
                "ratio": ratio,
                "amp_trajectory": amp_trajectory.astype(np.float64),
            }
        )
    return partials


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

        anti_alias_weight = _nyquist_fade_scalar(partial_freq, nyquist_hz)
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

        anti_alias_weight = _nyquist_fade(partial_freq_trajectory, nyquist_hz)
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


def _nyquist_fade(freq_profile: np.ndarray, nyquist_hz: float) -> np.ndarray:
    fade_start_hz = nyquist_hz * _NYQUIST_FADE_START
    if fade_start_hz >= nyquist_hz:
        return (freq_profile < nyquist_hz).astype(np.float64)

    fade_progress = (freq_profile - fade_start_hz) / (nyquist_hz - fade_start_hz)
    fade = 1.0 - np.clip(fade_progress, 0.0, 1.0)
    return np.square(fade)


def _nyquist_fade_scalar(freq_hz: float, nyquist_hz: float) -> float:
    return float(_nyquist_fade(np.array([freq_hz], dtype=np.float64), nyquist_hz)[0])
