"""Unified composable synth voice engine.

Four parallel source slots — ``osc``, ``partials``, ``fm``, ``noise`` — summed
pre-filter through a shared serial post-chain (HPF -> dual filter -> VCA ->
voice shaper).  Designed for cross-pollination: stack a supersaw under
additive partials with a 2-op FM bell on top through a Moog ladder, etc.

Mirrors the architectural conventions of ``drum_voice``: flat-namespaced
params with slot prefixes (``osc_*``, ``partials_*``, ``fm_*``, ``noise_*``),
string-dispatched types, perceptual macros resolved via ``_set_if_absent``
before render-time extraction, and deferred imports for the
``synth -> engines -> synth_voice`` cycle when dispatching saturation/preamp
shapers.

v0 skeleton: orchestrator and registration path wired end-to-end; slot
renderers stubbed to return silence while the four slot implementations land
in parallel subagents.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from code_musics.engines._dsp_utils import rng_for_note
from code_musics.engines._envelopes import render_envelope
from code_musics.engines._filters import (
    _SUPPORTED_FILTER_MODES,
    _SUPPORTED_FILTER_TOPOLOGIES,
    apply_filter,
    apply_zdf_svf,
)
from code_musics.engines._synth_macros import resolve_macros
from code_musics.engines._synth_slots import (
    render_fm,
    render_noise,
    render_osc,
    render_partials,
)
from code_musics.engines._waveshaper import ALGORITHM_NAMES, apply_waveshaper

logger: logging.Logger = logging.getLogger(__name__)

_VALID_VOICE_FILTER_MODES = _SUPPORTED_FILTER_MODES


def render(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
    freq_trajectory: np.ndarray | None = None,
) -> np.ndarray:
    """Render a composable synth voice with up to four parallel source slots."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if freq <= 0:
        raise ValueError("freq must be positive")

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0, dtype=np.float64)

    # --- Resolve macros (brightness, movement, body, dirt) ---
    resolve_macros(params)

    # --- Slot type selectors (None = disabled) ---
    osc_type: str | None = params.get("osc_type")
    partials_type: str | None = params.get("partials_type")
    fm_type: str | None = params.get("fm_type")
    noise_type: str | None = params.get("noise_type")

    # --- Slot levels ---
    osc_level = float(params.get("osc_level", 1.0))
    partials_level = float(params.get("partials_level", 1.0))
    fm_level = float(params.get("fm_level", 1.0))
    noise_level = float(params.get("noise_level", 0.1))

    # --- Per-slot envelope overrides (optional multi-point) ---
    osc_envelope_raw = params.get("osc_envelope")
    partials_envelope_raw = params.get("partials_envelope")
    fm_envelope_raw = params.get("fm_envelope")
    noise_envelope_raw = params.get("noise_envelope")

    # --- Per-slot shapers ---
    osc_shaper: str | None = params.get("osc_shaper")
    partials_shaper: str | None = params.get("partials_shaper")
    fm_shaper: str | None = params.get("fm_shaper")
    noise_shaper: str | None = params.get("noise_shaper")

    osc_shaper_drive = float(params.get("osc_shaper_drive", 0.5))
    osc_shaper_mix = float(params.get("osc_shaper_mix", 1.0))
    partials_shaper_drive = float(params.get("partials_shaper_drive", 0.5))
    partials_shaper_mix = float(params.get("partials_shaper_mix", 1.0))
    fm_shaper_drive = float(params.get("fm_shaper_drive", 0.5))
    fm_shaper_mix = float(params.get("fm_shaper_mix", 1.0))
    noise_shaper_drive = float(params.get("noise_shaper_drive", 0.5))
    noise_shaper_mix = float(params.get("noise_shaper_mix", 1.0))

    # --- Voice post-chain ---
    hpf_cutoff_hz = float(params.get("hpf_cutoff_hz", 0.0))

    filter_mode: str | None = params.get("filter_mode")
    filter_cutoff_hz = float(params.get("filter_cutoff_hz", 2000.0))
    resonance_q = float(params.get("resonance_q", 0.707))
    filter_drive = float(params.get("filter_drive", 0.0))
    filter_envelope_raw = params.get("filter_envelope")
    filter_topology = str(params.get("filter_topology", "svf")).lower()
    filter_morph = float(params.get("filter_morph", 0.0))
    k35_feedback_asymmetry = float(params.get("k35_feedback_asymmetry", 0.0))

    voice_shaper: str | None = params.get("shaper")
    voice_shaper_drive = float(params.get("shaper_drive", 0.5))
    voice_shaper_mix = float(params.get("shaper_mix", 1.0))
    voice_shaper_mode: str = str(params.get("shaper_mode", "triode"))
    voice_shaper_fidelity = float(params.get("shaper_fidelity", 0.5))
    voice_shaper_bit_depth = float(params.get("bit_depth", 8.0))
    voice_shaper_reduce_ratio = float(params.get("reduce_ratio", 2.0))

    # Voice envelope (simple attack/release fade; full ADSR lives at the
    # Score layer and is applied via amp/gain shaping, mirroring fm.py /
    # polyblep.py).
    attack_s = float(params.get("attack", 0.01))
    release_s = float(params.get("release", 0.05))

    # --- Validate filter surface ---
    if filter_mode is not None and filter_mode not in _VALID_VOICE_FILTER_MODES:
        raise ValueError(
            f"filter_mode must be one of {sorted(_VALID_VOICE_FILTER_MODES)} "
            f"or None, got {filter_mode!r}"
        )
    if filter_mode is not None and filter_topology not in _SUPPORTED_FILTER_TOPOLOGIES:
        raise ValueError(
            f"Unsupported filter_topology: {filter_topology!r}. "
            f"Supported: {sorted(_SUPPORTED_FILTER_TOPOLOGIES)}"
        )

    # --- Deterministic RNG + base freq profile ---
    rng = rng_for_note(
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=sample_rate,
        params=params,
    )

    time = np.arange(n_samples, dtype=np.float64) / sample_rate
    freq_profile = _resolve_freq_profile(
        freq=freq, n_samples=n_samples, freq_trajectory=freq_trajectory
    )

    # --- Render slots in parallel and sum ---
    signal = np.zeros(n_samples, dtype=np.float64)

    if osc_type is not None and osc_level > 0:
        osc_signal = render_osc(
            n_samples=n_samples,
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            rng=rng,
            osc_type=osc_type,
            params=params,
        )
        osc_signal = _apply_layer_shaper(
            osc_signal,
            osc_shaper,
            osc_shaper_drive,
            osc_shaper_mix,
            sample_rate=sample_rate,
        )
        osc_env = _build_layer_envelope(osc_envelope_raw, n_samples)
        signal += osc_level * osc_signal * osc_env

    if partials_type is not None and partials_level > 0:
        partials_signal = render_partials(
            n_samples=n_samples,
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            rng=rng,
            partials_type=partials_type,
            params=params,
        )
        partials_signal = _apply_layer_shaper(
            partials_signal,
            partials_shaper,
            partials_shaper_drive,
            partials_shaper_mix,
            sample_rate=sample_rate,
        )
        partials_env = _build_layer_envelope(partials_envelope_raw, n_samples)
        signal += partials_level * partials_signal * partials_env

    if fm_type is not None and fm_level > 0:
        fm_signal = render_fm(
            n_samples=n_samples,
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            rng=rng,
            fm_type=fm_type,
            params=params,
        )
        fm_signal = _apply_layer_shaper(
            fm_signal,
            fm_shaper,
            fm_shaper_drive,
            fm_shaper_mix,
            sample_rate=sample_rate,
        )
        fm_env = _build_layer_envelope(fm_envelope_raw, n_samples)
        signal += fm_level * fm_signal * fm_env

    if noise_type is not None and noise_level > 0:
        noise_signal = render_noise(
            n_samples=n_samples,
            freq=freq,
            sample_rate=sample_rate,
            rng=rng,
            noise_type=noise_type,
            params=params,
        )
        noise_signal = _apply_layer_shaper(
            noise_signal,
            noise_shaper,
            noise_shaper_drive,
            noise_shaper_mix,
            sample_rate=sample_rate,
        )
        noise_env = _build_layer_envelope(noise_envelope_raw, n_samples)
        signal += noise_level * noise_signal * noise_env

    # --- Post-chain: HPF -> main filter -> voice shaper ---
    if hpf_cutoff_hz > 0.0:
        hpf_profile = np.full(n_samples, hpf_cutoff_hz, dtype=np.float64)
        signal = apply_zdf_svf(
            signal,
            cutoff_profile=hpf_profile,
            resonance_q=0.707,
            sample_rate=sample_rate,
            filter_mode="highpass",
            filter_drive=0.0,
        )

    if filter_mode is not None:
        if filter_envelope_raw is not None:
            cutoff_profile = render_envelope(
                filter_envelope_raw, n_samples, default_value=filter_cutoff_hz
            )
        else:
            cutoff_profile = np.full(n_samples, filter_cutoff_hz, dtype=np.float64)

        if filter_topology == "svf":
            signal = apply_zdf_svf(
                signal,
                cutoff_profile=cutoff_profile,
                resonance_q=resonance_q,
                sample_rate=sample_rate,
                filter_mode=filter_mode,
                filter_drive=filter_drive,
            )
        else:
            signal = apply_filter(
                signal,
                cutoff_profile=cutoff_profile,
                resonance_q=resonance_q,
                sample_rate=sample_rate,
                filter_mode=filter_mode,
                filter_drive=filter_drive,
                filter_topology=filter_topology,
                filter_morph=filter_morph,
                k35_feedback_asymmetry=k35_feedback_asymmetry,
            )

    # Simple attack/release fade to suppress clicks; the Score layer applies
    # richer amp shaping on top.
    signal = _apply_attack_release(
        signal, time=time, duration=duration, attack_s=attack_s, release_s=release_s
    )

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

    peak = np.max(np.abs(signal))
    if peak > 1e-9:
        signal = signal / peak
    return (amp * signal).astype(np.float64)


def _resolve_freq_profile(
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


def _build_layer_envelope(envelope_raw: Any, n_samples: int) -> np.ndarray:
    """Build a per-slot amplitude envelope.

    Custom multi-point envelopes are honored; otherwise the slot is unity
    (flat) and the voice-level attack/release + Score-level amp shaping own
    note contour.  This matches tonal-engine convention — the Score layer
    handles ADSR through amp/gain plumbing rather than per-engine envelopes.
    """
    if envelope_raw is not None:
        return render_envelope(envelope_raw, n_samples, default_value=1.0)
    return np.ones(n_samples, dtype=np.float64)


def _apply_attack_release(
    signal: np.ndarray,
    *,
    time: np.ndarray,
    duration: float,
    attack_s: float,
    release_s: float,
) -> np.ndarray:
    """Apply linear attack/release fades to suppress edge clicks."""
    env = np.ones_like(signal)
    if attack_s > 0.0:
        attack_samples = int(min(attack_s, duration) * len(signal) / duration)
        if attack_samples > 1:
            env[:attack_samples] = np.linspace(0.0, 1.0, attack_samples)
    if release_s > 0.0:
        release_samples = int(min(release_s, duration) * len(signal) / duration)
        if release_samples > 1:
            env[-release_samples:] = np.minimum(
                env[-release_samples:], np.linspace(1.0, 0.0, release_samples)
            )
    del time  # reserved for future ADSR plumbing
    return signal * env


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
    """Apply waveshaper, saturation, or preamp to a signal.

    Identical dispatch surface to ``drum_voice._apply_layer_shaper``: any
    registered waveshaper algorithm, the modern ``saturation`` effect, or
    the flux-domain ``preamp`` effect.  ``shaper=None`` is a no-op.
    """
    if shaper is None:
        return signal

    if shaper == "saturation":
        # Deferred import: synth.py -> engines/__init__.py -> synth_voice.py
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
        # Deferred import: synth.py -> engines/__init__.py -> synth_voice.py
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
