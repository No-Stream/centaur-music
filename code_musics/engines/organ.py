"""Tonewheel / pipe organ synthesis engine."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import numpy as np

_NYQUIST_FADE_START = 0.85
_DEFAULT_DRAWBAR_RATIOS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
_DEFAULT_DRAWBARS = [0, 8, 8, 8, 0, 0, 0, 0, 0]
_MAX_DRIFT_CENTS = 4.0
_MAX_VIBRATO_CENTS = 14.0
_CLICK_DECAY_SAMPLES_AT_44100 = 132  # ~3ms at 44100 Hz
_GOLDEN_RATIO_FRAC = 0.6180339887498949
_TONEWHEEL_SHAPE_ROLLOFF = 0.45
_TONEWHEEL_SHAPE_MAX_HARMONICS = 5


def render(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
    freq_trajectory: np.ndarray | None = None,
) -> np.ndarray:
    """Render a tonewheel/pipe organ voice."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if freq <= 0:
        raise ValueError("freq must be positive")

    drawbars = list(params.get("drawbars", _DEFAULT_DRAWBARS))
    drawbar_ratios = list(params.get("drawbar_ratios", _DEFAULT_DRAWBAR_RATIOS))
    click = float(params.get("click", 0.15))
    click_brightness = float(params.get("click_brightness", 0.5))
    vibrato_depth = float(params.get("vibrato_depth", 0.0))
    vibrato_rate_hz = float(params.get("vibrato_rate_hz", 6.8))
    vibrato_chorus = float(params.get("vibrato_chorus", 0.0))
    drift = float(params.get("drift", 0.12))
    drift_rate_hz = float(params.get("drift_rate_hz", 0.07))
    leakage = float(params.get("leakage", 0.08))
    tonewheel_shape = float(params.get("tonewheel_shape", 0.0))

    if len(drawbars) != len(drawbar_ratios):
        raise ValueError(
            f"drawbars length ({len(drawbars)}) must match "
            f"drawbar_ratios length ({len(drawbar_ratios)})"
        )
    for i, db_val in enumerate(drawbars):
        if not (0 <= db_val <= 8):
            raise ValueError(f"drawbar {i} value {db_val} must be 0-8")
    for i, ratio in enumerate(drawbar_ratios):
        if ratio <= 0:
            raise ValueError(f"drawbar_ratios[{i}] must be positive, got {ratio}")
    if not any(db > 0 for db in drawbars):
        raise ValueError("at least one drawbar must be nonzero")

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0)

    if freq_trajectory is not None:
        freq_trajectory = np.asarray(freq_trajectory, dtype=np.float64)
        if freq_trajectory.ndim != 1:
            raise ValueError("freq_trajectory must be one-dimensional")
        if freq_trajectory.size != n_samples:
            raise ValueError("freq_trajectory length must match note duration")

    base_freq_profile = (
        np.full(n_samples, float(freq), dtype=np.float64)
        if freq_trajectory is None
        else freq_trajectory
    )

    nyquist_hz = sample_rate / 2.0
    num_drawbars = len(drawbars)

    drift_trajectories = _build_drift_trajectories(
        num_drawbars=num_drawbars,
        n_samples=n_samples,
        duration=duration,
        drift=drift,
        drift_rate_hz=drift_rate_hz,
    )

    vibrato_lfo = _build_vibrato_lfo(
        num_drawbars=num_drawbars,
        n_samples=n_samples,
        duration=duration,
        vibrato_depth=vibrato_depth,
        vibrato_rate_hz=vibrato_rate_hz,
    )

    signal = np.zeros(n_samples, dtype=np.float64)

    for db_idx in range(num_drawbars):
        db_level = drawbars[db_idx]
        if db_level == 0:
            continue

        db_amp = db_level / 8.0
        ratio = drawbar_ratios[db_idx]
        tonewheel_freq = base_freq_profile * ratio * drift_trajectories[db_idx]

        tonewheel_signal = _render_tonewheel(
            tonewheel_freq=tonewheel_freq,
            vibrato_mod=vibrato_lfo[db_idx],
            vibrato_chorus=vibrato_chorus,
            tonewheel_shape=tonewheel_shape,
            sample_rate=sample_rate,
            nyquist_hz=nyquist_hz,
        )

        signal += db_amp * tonewheel_signal

    if leakage > 0:
        signal += _render_leakage(
            base_freq_profile=base_freq_profile,
            drawbars=drawbars,
            drawbar_ratios=drawbar_ratios,
            drift_trajectories=drift_trajectories,
            leakage=leakage,
            sample_rate=sample_rate,
            nyquist_hz=nyquist_hz,
        )

    if click > 0:
        click_signal = _render_click(
            freq=freq,
            click=click,
            click_brightness=click_brightness,
            n_samples=n_samples,
            sample_rate=sample_rate,
            duration=duration,
            amp=amp,
            params=params,
        )
        signal += click_signal

    peak = np.max(np.abs(signal))
    if peak <= 0.0:
        raise ValueError("organ render produced no audible output")
    return amp * (signal / peak)


def _build_drift_trajectories(
    *,
    num_drawbars: int,
    n_samples: int,
    duration: float,
    drift: float,
    drift_rate_hz: float,
) -> list[np.ndarray]:
    if drift <= 0 or n_samples == 0:
        return [np.ones(n_samples, dtype=np.float64) for _ in range(num_drawbars)]

    t = np.linspace(0.0, duration, n_samples, endpoint=False, dtype=np.float64)
    max_cents = _MAX_DRIFT_CENTS * drift
    trajectories: list[np.ndarray] = []

    for db_idx in range(num_drawbars):
        phase_offset = db_idx * _GOLDEN_RATIO_FRAC * 2.0 * np.pi
        rate_variation = 1.0 + 0.15 * np.sin(db_idx * 1.7)
        effective_rate = drift_rate_hz * rate_variation
        cents = max_cents * np.sin(2.0 * np.pi * effective_rate * t + phase_offset)
        trajectories.append(np.power(2.0, cents / 1200.0))

    return trajectories


def _build_vibrato_lfo(
    *,
    num_drawbars: int,
    n_samples: int,
    duration: float,
    vibrato_depth: float,
    vibrato_rate_hz: float,
) -> list[np.ndarray]:
    if vibrato_depth <= 0 or n_samples == 0:
        return [np.ones(n_samples, dtype=np.float64) for _ in range(num_drawbars)]

    t = np.linspace(0.0, duration, n_samples, endpoint=False, dtype=np.float64)
    max_cents = _MAX_VIBRATO_CENTS * vibrato_depth
    lfos: list[np.ndarray] = []
    phase_spacing = 2.0 * np.pi / max(num_drawbars, 1)

    for db_idx in range(num_drawbars):
        phase_offset = db_idx * phase_spacing
        cents = max_cents * np.sin(2.0 * np.pi * vibrato_rate_hz * t + phase_offset)
        lfos.append(np.power(2.0, cents / 1200.0))

    return lfos


def _render_tonewheel(
    *,
    tonewheel_freq: np.ndarray,
    vibrato_mod: np.ndarray,
    vibrato_chorus: float,
    tonewheel_shape: float,
    sample_rate: int,
    nyquist_hz: float,
) -> np.ndarray:
    has_vibrato = not np.all(vibrato_mod == 1.0)

    if has_vibrato and vibrato_chorus > 0:
        dry_signal = _render_shaped_tonewheel(
            freq_profile=tonewheel_freq,
            tonewheel_shape=tonewheel_shape,
            sample_rate=sample_rate,
            nyquist_hz=nyquist_hz,
        )
        mod_signal = _render_shaped_tonewheel(
            freq_profile=tonewheel_freq * vibrato_mod,
            tonewheel_shape=tonewheel_shape,
            sample_rate=sample_rate,
            nyquist_hz=nyquist_hz,
        )
        # chorus=0 -> pure vibrato (mod only)
        # chorus=1 -> 50/50 dry+mod blend
        return (1.0 - vibrato_chorus) * mod_signal + vibrato_chorus * (
            0.5 * dry_signal + 0.5 * mod_signal
        )
    elif has_vibrato:
        return _render_shaped_tonewheel(
            freq_profile=tonewheel_freq * vibrato_mod,
            tonewheel_shape=tonewheel_shape,
            sample_rate=sample_rate,
            nyquist_hz=nyquist_hz,
        )
    else:
        return _render_shaped_tonewheel(
            freq_profile=tonewheel_freq,
            tonewheel_shape=tonewheel_shape,
            sample_rate=sample_rate,
            nyquist_hz=nyquist_hz,
        )


def _render_shaped_tonewheel(
    *,
    freq_profile: np.ndarray,
    tonewheel_shape: float,
    sample_rate: int,
    nyquist_hz: float,
) -> np.ndarray:
    n_samples = freq_profile.size
    anti_alias = _nyquist_fade(freq_profile, nyquist_hz)
    if np.max(anti_alias) <= 0.0:
        return np.zeros(n_samples, dtype=np.float64)

    phase = np.cumsum(
        np.concatenate(
            [
                np.zeros(1, dtype=np.float64),
                2.0 * np.pi * freq_profile[:-1] / float(sample_rate),
            ]
        )
    )
    signal = anti_alias * np.sin(phase)

    if tonewheel_shape > 0:
        max_h = 4 if tonewheel_shape < 0.5 else _TONEWHEEL_SHAPE_MAX_HARMONICS
        for h in range(2, max_h + 1):
            harmonic_freq = freq_profile * h
            harmonic_aa = _nyquist_fade(harmonic_freq, nyquist_hz)
            if np.max(harmonic_aa) <= 0.0:
                continue
            harmonic_amp = tonewheel_shape * (_TONEWHEEL_SHAPE_ROLLOFF ** (h - 1))
            harmonic_phase = phase * h
            signal += harmonic_amp * harmonic_aa * np.sin(harmonic_phase)

    return signal


def _render_leakage(
    *,
    base_freq_profile: np.ndarray,
    drawbars: list[int],
    drawbar_ratios: list[float],
    drift_trajectories: list[np.ndarray],
    leakage: float,
    sample_rate: int,
    nyquist_hz: float,
) -> np.ndarray:
    n_samples = base_freq_profile.size
    signal = np.zeros(n_samples, dtype=np.float64)
    num_drawbars = len(drawbars)
    leakage_amp_scale = leakage * 0.06

    for db_idx in range(num_drawbars):
        if drawbars[db_idx] == 0:
            continue

        db_amp = drawbars[db_idx] / 8.0

        for neighbor_offset in [-1, 1]:
            neighbor_idx = db_idx + neighbor_offset
            if neighbor_idx < 0 or neighbor_idx >= num_drawbars:
                continue

            neighbor_ratio = drawbar_ratios[neighbor_idx]
            neighbor_freq = (
                base_freq_profile * neighbor_ratio * drift_trajectories[neighbor_idx]
            )
            neighbor_aa = _nyquist_fade(neighbor_freq, nyquist_hz)
            if np.max(neighbor_aa) <= 0.0:
                continue

            phase = np.cumsum(
                np.concatenate(
                    [
                        np.zeros(1, dtype=np.float64),
                        2.0 * np.pi * neighbor_freq[:-1] / float(sample_rate),
                    ]
                )
            )
            signal += leakage_amp_scale * db_amp * neighbor_aa * np.sin(phase)

    return signal


def _render_click(
    *,
    freq: float,
    click: float,
    click_brightness: float,
    n_samples: int,
    sample_rate: int,
    duration: float,
    amp: float,
    params: dict[str, Any],
) -> np.ndarray:
    rng = _rng_for_note(
        freq=freq, duration=duration, amp=amp, sample_rate=sample_rate, params=params
    )

    noise = rng.standard_normal(n_samples)

    center_hz = freq * (2.0 + click_brightness * 8.0)
    center_hz = min(center_hz, sample_rate * 0.4)

    noise = _bandpass_noise(noise, sample_rate=sample_rate, center_hz=center_hz)

    decay_samples = max(1, int(_CLICK_DECAY_SAMPLES_AT_44100 * sample_rate / 44100))
    envelope = np.zeros(n_samples, dtype=np.float64)
    click_len = min(decay_samples * 4, n_samples)
    t_click = np.arange(click_len, dtype=np.float64)
    envelope[:click_len] = np.exp(-t_click / max(1.0, decay_samples))

    return click * envelope * noise


def _nyquist_fade(freq_profile: np.ndarray, nyquist_hz: float) -> np.ndarray:
    fade_start_hz = nyquist_hz * _NYQUIST_FADE_START
    if fade_start_hz >= nyquist_hz:
        return (freq_profile < nyquist_hz).astype(np.float64)
    fade_progress = (freq_profile - fade_start_hz) / (nyquist_hz - fade_start_hz)
    fade = 1.0 - np.clip(fade_progress, 0.0, 1.0)
    return np.square(fade)


def _rng_for_note(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
) -> np.random.Generator:
    seed_material = repr(
        (
            round(freq, 6),
            round(duration, 6),
            round(amp, 6),
            sample_rate,
            tuple(sorted(params.items())),
        )
    ).encode("utf-8")
    seed_bytes = sha256(seed_material).digest()[:8]
    seed = int.from_bytes(seed_bytes, byteorder="big", signed=False)
    return np.random.default_rng(seed)


def _bandpass_noise(
    signal: np.ndarray,
    *,
    sample_rate: int,
    center_hz: float,
    width_ratio: float = 0.75,
) -> np.ndarray:
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
