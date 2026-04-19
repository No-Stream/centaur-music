"""Unified composable drum voice engine.

Four independent, mixable synthesis layers — exciter, tone, noise, metallic —
combined through a shared signal flow with per-layer envelopes, shapers, and
filters.  Replaces the separate kick_tom / snare / clap / metallic_perc /
noise_perc engines with a single composable architecture.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from code_musics.engines._drum_layers import (
    render_exciter,
    render_metallic,
    render_noise,
    render_tone,
)
from code_musics.engines._drum_macros import resolve_macros
from code_musics.engines._drum_utils import resolve_velocity_timbre, rng_for_note
from code_musics.engines._envelopes import render_envelope
from code_musics.engines._filters import _SUPPORTED_FILTER_MODES, apply_zdf_svf
from code_musics.engines._pi_macros import resolve_pi_macros
from code_musics.engines._waveshaper import ALGORITHM_NAMES, apply_waveshaper

logger: logging.Logger = logging.getLogger(__name__)

_VALID_VOICE_FILTER_MODES = _SUPPORTED_FILTER_MODES - {"notch"}


def render(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
    freq_trajectory: np.ndarray | None = None,
) -> np.ndarray:
    """Render a composable drum voice with up to four synthesis layers."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if freq <= 0:
        raise ValueError("freq must be positive")

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0, dtype=np.float64)

    # --- Resolve macros (punch, decay_shape, character) ---
    resolve_macros(params)

    # --- Resolve physical-informed macros (pi_hardness, pi_tension, ...) ---
    resolve_pi_macros(params)

    # --- Extract layer type selectors ---
    tone_type: str | None = params.get("tone_type", "oscillator")
    exciter_type: str | None = params.get("exciter_type")
    noise_type: str | None = params.get("noise_type")
    metallic_type: str | None = params.get("metallic_type")

    # --- Extract levels ---
    tone_level = float(params.get("tone_level", 1.0))
    exciter_level = float(params.get("exciter_level", 0.08))
    noise_level = float(params.get("noise_level", 0.02))
    metallic_level = float(params.get("metallic_level", 0.0))

    # --- Extract decay times ---
    tone_decay_s = float(params.get("tone_decay_s", 0.26))
    exciter_decay_s = float(params.get("exciter_decay_s", 0.007))
    noise_decay_s = float(params.get("noise_decay_s", 0.028))
    metallic_decay_s = float(params.get("metallic_decay_s", 0.08))

    # --- Extract pitch sweep ---
    tone_sweep_ratio = float(params.get("tone_sweep_ratio", 2.5))
    tone_sweep_decay_s = float(params.get("tone_sweep_decay_s", 0.042))

    # --- Extract shapers ---
    exciter_shaper: str | None = params.get("exciter_shaper")
    exciter_shaper_drive = float(params.get("exciter_shaper_drive", 0.5))
    exciter_shaper_mix = float(params.get("exciter_shaper_mix", 1.0))
    exciter_shaper_mode: str = str(params.get("exciter_shaper_mode", "triode"))
    exciter_shaper_fidelity = float(params.get("exciter_shaper_fidelity", 0.5))

    tone_shaper: str | None = params.get("tone_shaper")
    tone_shaper_drive = float(params.get("tone_shaper_drive", 0.5))
    tone_shaper_mix = float(params.get("tone_shaper_mix", 1.0))
    tone_shaper_mode: str = str(params.get("tone_shaper_mode", "triode"))
    tone_shaper_fidelity = float(params.get("tone_shaper_fidelity", 0.5))

    voice_shaper: str | None = params.get("shaper")
    voice_shaper_drive = float(params.get("shaper_drive", 0.5))
    voice_shaper_mix = float(params.get("shaper_mix", 1.0))
    voice_shaper_mode: str = str(params.get("shaper_mode", "triode"))
    voice_shaper_fidelity = float(params.get("shaper_fidelity", 0.5))

    # Digital-character shaper params (consumed when shaper is bit_crush /
    # rate_reduce; ignored by other algorithms).
    voice_shaper_bit_depth = float(params.get("bit_depth", 8.0))
    voice_shaper_reduce_ratio = float(params.get("reduce_ratio", 2.0))

    # --- Extract voice filter ---
    filter_mode: str | None = params.get("filter_mode")
    filter_cutoff_hz = float(params.get("filter_cutoff_hz", 2000.0))
    filter_q = float(params.get("filter_q", 0.707))
    filter_drive = float(params.get("filter_drive", 0.0))
    filter_envelope_raw = params.get("filter_envelope")

    # --- Extract noise/metallic per-layer filter ---
    noise_filter_mode: str | None = params.get("noise_filter_mode")
    noise_filter_cutoff_hz = float(params.get("noise_filter_cutoff_hz", 2000.0))
    noise_filter_q = float(params.get("noise_filter_q", 0.707))

    metallic_filter_mode: str | None = params.get("metallic_filter_mode")
    metallic_filter_cutoff_hz = float(params.get("metallic_filter_cutoff_hz", 2000.0))
    metallic_filter_q = float(params.get("metallic_filter_q", 1.2))

    # --- Extract envelope overrides ---
    tone_envelope_raw = params.get("tone_envelope")
    exciter_envelope_raw = params.get("exciter_envelope")
    noise_envelope_raw = params.get("noise_envelope")
    metallic_envelope_raw = params.get("metallic_envelope")
    tone_pitch_envelope_raw = params.get("tone_pitch_envelope")

    # --- Validate voice filter ---
    if filter_mode is not None and filter_mode not in _VALID_VOICE_FILTER_MODES:
        raise ValueError(
            f"filter_mode must be one of {sorted(_VALID_VOICE_FILTER_MODES)} "
            f"or None, got {filter_mode!r}"
        )

    # --- Velocity-to-timbre scaling ---
    timbre = resolve_velocity_timbre(amp, params)
    tone_decay_s *= timbre.decay_scale
    noise_decay_s *= timbre.decay_scale
    metallic_decay_s *= timbre.decay_scale

    # --- Deterministic RNG ---
    rng = rng_for_note(
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=sample_rate,
        params=params,
    )

    time = np.arange(n_samples, dtype=np.float64) / sample_rate

    # --- Build freq_profile with pitch sweep ---
    base_freq_profile = _resolve_base_freq_profile(
        freq=freq, n_samples=n_samples, freq_trajectory=freq_trajectory
    )

    if tone_pitch_envelope_raw is not None:
        sweep_profile = render_envelope(
            tone_pitch_envelope_raw, n_samples, default_value=1.0
        )
    else:
        sweep_profile = 1.0 + (tone_sweep_ratio - 1.0) * np.exp(
            -time / tone_sweep_decay_s
        )
    freq_profile = base_freq_profile * sweep_profile

    # --- Render layers ---
    signal = np.zeros(n_samples, dtype=np.float64)

    # Exciter
    exciter_signal: np.ndarray | None = None
    if exciter_type is not None and exciter_level > 0:
        exciter_signal = render_exciter(
            n_samples=n_samples,
            freq=freq,
            sample_rate=sample_rate,
            rng=rng,
            exciter_type=exciter_type,
            params=params,
        )
        shaped_exciter = _apply_layer_shaper(
            exciter_signal,
            exciter_shaper,
            exciter_shaper_drive,
            exciter_shaper_mix,
            sample_rate=sample_rate,
            mode=exciter_shaper_mode,
            fidelity=exciter_shaper_fidelity,
        )
        exciter_env = _build_layer_envelope(
            exciter_envelope_raw, n_samples, time, exciter_decay_s
        )
        signal += exciter_level * shaped_exciter * exciter_env

    # Tone
    if tone_type is not None and tone_level > 0:
        tone_signal = render_tone(
            n_samples=n_samples,
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            rng=rng,
            tone_type=tone_type,
            exciter_signal=exciter_signal,
            params=params,
        )
        shaped_tone = _apply_layer_shaper(
            tone_signal,
            tone_shaper,
            tone_shaper_drive,
            tone_shaper_mix,
            sample_rate=sample_rate,
            mode=tone_shaper_mode,
            fidelity=tone_shaper_fidelity,
        )
        tone_env = _build_layer_envelope(
            tone_envelope_raw, n_samples, time, tone_decay_s
        )
        signal += tone_level * shaped_tone * tone_env

    # Noise
    if noise_type is not None and noise_level > 0:
        noise_signal = render_noise(
            n_samples=n_samples,
            freq=freq,
            sample_rate=sample_rate,
            rng=rng,
            noise_type=noise_type,
            params=params,
        )
        if noise_filter_mode is not None:
            cutoff_profile = np.full(
                n_samples, noise_filter_cutoff_hz, dtype=np.float64
            )
            noise_signal = apply_zdf_svf(
                noise_signal,
                cutoff_profile=cutoff_profile,
                resonance_q=noise_filter_q,
                sample_rate=sample_rate,
                filter_mode=noise_filter_mode,
                filter_drive=0.0,
            )
        noise_env = _build_layer_envelope(
            noise_envelope_raw, n_samples, time, noise_decay_s
        )
        signal += noise_level * noise_signal * noise_env

    # Metallic
    if metallic_type is not None and metallic_level > 0:
        metallic_signal = render_metallic(
            n_samples=n_samples,
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            rng=rng,
            metallic_type=metallic_type,
            params=params,
            exciter_signal=exciter_signal,
        )
        if metallic_filter_mode is not None:
            cutoff_profile = np.full(
                n_samples, metallic_filter_cutoff_hz, dtype=np.float64
            )
            metallic_signal = apply_zdf_svf(
                metallic_signal,
                cutoff_profile=cutoff_profile,
                resonance_q=metallic_filter_q,
                sample_rate=sample_rate,
                filter_mode=metallic_filter_mode,
                filter_drive=0.0,
            )
        metallic_env = _build_layer_envelope(
            metallic_envelope_raw, n_samples, time, metallic_decay_s
        )
        signal += metallic_level * metallic_signal * metallic_env

    # --- Voice filter (post-mix) ---
    if filter_mode is not None:
        if filter_envelope_raw is not None:
            cutoff_profile = render_envelope(
                filter_envelope_raw, n_samples, default_value=filter_cutoff_hz
            )
        else:
            cutoff_profile = np.full(n_samples, filter_cutoff_hz, dtype=np.float64)
        signal = apply_zdf_svf(
            signal,
            cutoff_profile=cutoff_profile,
            resonance_q=filter_q,
            sample_rate=sample_rate,
            filter_mode=filter_mode,
            filter_drive=filter_drive,
        )

    # --- Voice shaper (post-mix) ---
    signal = _apply_layer_shaper(
        signal,
        voice_shaper,
        voice_shaper_drive,
        voice_shaper_mix,
        sample_rate=sample_rate,
        mode=voice_shaper_mode,
        fidelity=voice_shaper_fidelity,
        bit_depth=voice_shaper_bit_depth,
        reduce_ratio=voice_shaper_reduce_ratio,
    )

    # --- Peak normalize and scale ---
    peak = np.max(np.abs(signal))
    if peak > 1e-9:
        signal = signal / peak
    return (amp * signal).astype(np.float64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_base_freq_profile(
    *,
    freq: float,
    n_samples: int,
    freq_trajectory: np.ndarray | None,
) -> np.ndarray:
    """Resolve freq_trajectory into a base frequency profile."""
    if freq_trajectory is None:
        return np.full(n_samples, freq, dtype=np.float64)

    resolved = np.asarray(freq_trajectory, dtype=np.float64)
    if resolved.ndim != 1:
        raise ValueError("freq_trajectory must be one-dimensional")
    if resolved.size != n_samples:
        raise ValueError("freq_trajectory length must match note duration")
    if np.any(resolved <= 0):
        raise ValueError("freq_trajectory values must be positive")
    return resolved


def _build_layer_envelope(
    envelope_raw: Any,
    n_samples: int,
    time: np.ndarray,
    decay_s: float,
) -> np.ndarray:
    """Build per-layer amplitude envelope: custom multi-point or exponential decay."""
    if envelope_raw is not None:
        return render_envelope(envelope_raw, n_samples, default_value=0.0)
    return np.exp(-time / max(decay_s, 1e-6))


def _apply_layer_shaper(
    signal: np.ndarray,
    shaper: str | None,
    drive: float,
    mix: float,
    *,
    sample_rate: int = 44_100,
    mode: str = "triode",
    fidelity: float = 0.5,
    bit_depth: float = 8.0,
    reduce_ratio: float = 2.0,
) -> np.ndarray:
    """Apply waveshaper, saturation, or preamp to a layer signal."""
    if shaper is None:
        return signal

    if shaper == "saturation":
        # Deferred import: synth.py -> engines/__init__.py -> drum_voice.py cycle
        from code_musics.synth import apply_saturation

        result = apply_saturation(
            signal,
            drive=drive,
            mix=mix,
            mode=mode,
            fidelity=fidelity,
        )
        return np.asarray(result, dtype=np.float64)

    if shaper == "preamp":
        # Deferred import: synth.py -> engines/__init__.py -> drum_voice.py cycle
        from code_musics.synth import apply_preamp

        result = apply_preamp(
            signal,
            drive=drive,
            mix=mix,
            sample_rate=sample_rate,
        )
        return np.asarray(result, dtype=np.float64)

    if shaper not in ALGORITHM_NAMES:
        raise ValueError(
            f"shaper must be one of {sorted(ALGORITHM_NAMES)}, 'saturation', "
            f"'preamp', or None, got {shaper!r}"
        )

    return apply_waveshaper(
        signal,
        algorithm=shaper,
        drive=drive,
        mix=mix,
        bit_depth=bit_depth,
        reduce_ratio=reduce_ratio,
    )
