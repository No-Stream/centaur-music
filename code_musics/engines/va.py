"""Virtual-Analog (VA) synthesis engine — 90s/00s character.

Captures the voiced-DSP character of the JP-8000, Access Virus, and Waldorf Q
era. The identity isn't "good analog emulation" — it's an opinionated stack of
oscillator voicing, saturation, dual filters, and optional comb resonance that
together sound like the era.

Two oscillator modes:

- ``supersaw``    — 7-voice PolyBLEP saw bank with Szabo-accurate nonlinear
  detune and mix laws (JP-8000 lineage), random starting phase per note, and
  optional hard-sync to the center voice (Virus HyperSaw-adjacent).
- ``spectralwave`` — partial-bank oscillator with continuous ``spectral_position``
  sweep from saw → spectral → square, plus optional ``_spectral_morphs`` layer
  (smear, inharmonic_scale, phase_disperse, shepard, random_amplitudes) for
  Virus Classic and Waldorf Q flavors.

Signal flow: osc_mix → (HPF) → drive (waveshaper) → filter chain (single/serial/
parallel/split with two ZDF filter slots, optional comb) → analog post-proc →
peak-normalize → amp.
"""

from __future__ import annotations

from typing import Any

import numba
import numpy as np

from code_musics.engines._dsp_utils import (
    apply_analog_post_processing,
    apply_cutoff_cv_dither,
    apply_note_jitter,
    apply_pitch_cv_dither,
    apply_voice_card,
    apply_voice_card_post_offsets,
    build_cutoff_drift,
    build_drift,
    build_keytracked_cutoff_profile,
    extract_analog_params,
    nyquist_fade,
    rng_for_note,
    voice_card_offsets,
)
from code_musics.engines._filters import (
    _SUPPORTED_FILTER_MODES,
    _SUPPORTED_FILTER_TOPOLOGIES,
    apply_comb,
    apply_filter,
)
from code_musics.engines._oscillators import (
    polyblep_saw,
)
from code_musics.engines._spectral_morphs import (
    MORPH_TYPES,
    apply_sigma_approximation,
    apply_spectral_morph,
)
from code_musics.engines._waveshaper import (
    apply_waveshaper,
)

_OSC_MODES: frozenset[str] = frozenset({"supersaw", "spectralwave"})
_FILTER_ROUTINGS: frozenset[str] = frozenset({"single", "serial", "parallel", "split"})
_COMB_POSITIONS: frozenset[str] = frozenset(
    {"off", "pre_filter", "post_filter", "parallel"}
)
_DRIVE_ALGORITHMS: frozenset[str] = frozenset({"tanh", "atan", "exponential"})

_SUPERSAW_VOICES: int = 7
_SUPERSAW_CENTER_INDEX: int = 3

_SUPERSAW_MAX_DETUNE_CENTS: float = 100.0

_MAX_PARTIALS: int = 256

# VA-flavored analog defaults (more restrained than the shared defaults tuned
# for polyblep/filtered_stack — JP/Virus/Q don't wobble as hard as an analog
# mono).
_VA_DEFAULT_ANALOG_JITTER: float = 0.15
_VA_DEFAULT_PITCH_DRIFT: float = 0.003
_VA_DEFAULT_VOICE_CARD_SPREAD: float = 1.5

# Default HPF for the supersaw mode — Szabo identified output HPF around
# 150 Hz as part of the JP signature "air."
_SUPERSAW_DEFAULT_HPF_HZ: float = 150.0


def _supersaw_detune_cents(detune: float) -> float:
    """Szabo-reverse-engineered detune polynomial for the JP-8000 supersaw.

    Maps user-facing ``detune`` in [0, 1] to a per-side-voice spread in cents.
    Szabo's polynomial returns a normalized factor in [0, 1]; we scale by a
    maximum cents constant to get the actual detune.
    """
    d = max(0.0, min(1.0, float(detune)))
    normalized = (
        10028.7312891634 * d**11
        - 50818.8652045924 * d**10
        + 111363.4808729368 * d**9
        - 138150.6761080548 * d**8
        + 106649.6679158292 * d**7
        - 53046.9642751875 * d**6
        + 17019.9518580080 * d**5
        - 3425.0836591318 * d**4
        + 404.2703938388 * d**3
        - 24.1878824391 * d**2
        + 0.6717417634 * d
        + 0.0030115596
    )
    return normalized * _SUPERSAW_MAX_DETUNE_CENTS


def _supersaw_side_gain(mix: float) -> float:
    m = max(0.0, min(1.0, float(mix)))
    return -0.73764 * m * m + 1.2841 * m + 0.044372


def _supersaw_center_gain(mix: float) -> float:
    m = max(0.0, min(1.0, float(mix)))
    return -0.55366 * m + 0.99785


@numba.njit(cache=True)
def _synced_phase_trajectory(
    phase_inc: np.ndarray,
    start_phase: float,
    center_cumphase: np.ndarray,
) -> np.ndarray:
    """Per-sample hard-sync phase accumulator.

    Walks the satellite's phase increment, resetting to zero at every sample
    where the center's wrapped phase has just wrapped (detected by a negative
    step in ``center_cumphase % 1.0``). Returns wrapped phase in [0, 1).
    """
    n = phase_inc.shape[0]
    out = np.empty(n, dtype=np.float64)
    phase = start_phase - np.floor(start_phase)
    prev_center_wrapped = center_cumphase[0] - np.floor(center_cumphase[0])
    for i in range(n):
        center_wrapped = center_cumphase[i] - np.floor(center_cumphase[i])
        if center_wrapped < prev_center_wrapped:
            phase = 0.0
        prev_center_wrapped = center_wrapped
        phase += phase_inc[i]
        if phase >= 1.0:
            phase -= np.floor(phase)
        out[i] = phase
    return out


def _supersaw_setup(
    *,
    freq_profile: np.ndarray,
    sample_rate: int,
    detune: float,
    mix: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Shared preamble for both supersaw render paths.

    Returns (cents_per_voice, gains, start_phases, cumphase_base, side_gain, center_gain).
    """
    side_cents = _supersaw_detune_cents(detune)
    side_gain = _supersaw_side_gain(mix)
    center_gain = _supersaw_center_gain(mix)

    cents_per_voice = (
        np.array(
            [-1.0, -0.57735, -0.33333, 0.0, 0.33333, 0.57735, 1.0],
            dtype=np.float64,
        )
        * side_cents
    )
    gains = np.array(
        [side_gain, side_gain, side_gain, center_gain, side_gain, side_gain, side_gain],
        dtype=np.float64,
    )
    start_phases = rng.random(_SUPERSAW_VOICES).astype(np.float64)
    cumphase_base = np.cumsum(freq_profile / sample_rate)
    return cents_per_voice, gains, start_phases, cumphase_base, side_gain, center_gain


def _render_supersaw_voice(
    *,
    freq_profile: np.ndarray,
    sample_rate: int,
    cumphase_base: np.ndarray,
    cents: float,
    start_phase: float,
    sync: bool,
    center_cumphase: np.ndarray | None,
) -> np.ndarray:
    """Render a single supersaw voice using the shared cumphase trajectory."""
    ratio = float(2.0 ** (cents / 1200.0))
    voice_phase_inc = freq_profile * ratio / sample_rate
    if sync and center_cumphase is not None:
        phase = _synced_phase_trajectory(voice_phase_inc, start_phase, center_cumphase)
    else:
        voice_cumphase = cumphase_base * ratio + start_phase
        phase = voice_cumphase - np.floor(voice_cumphase)
    return polyblep_saw(phase, voice_phase_inc)


def _render_supersaw_bank(
    *,
    freq_profile: np.ndarray,
    sample_rate: int,
    detune: float,
    mix: float,
    sync: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render the 7-voice supersaw bank as a single mixed output."""
    (
        cents_per_voice,
        gains,
        start_phases,
        cumphase_base,
        side_gain,
        center_gain,
    ) = _supersaw_setup(
        freq_profile=freq_profile,
        sample_rate=sample_rate,
        detune=detune,
        mix=mix,
        rng=rng,
    )

    center_ratio = float(2.0 ** (cents_per_voice[_SUPERSAW_CENTER_INDEX] / 1200.0))
    center_cumphase = (
        cumphase_base * center_ratio + start_phases[_SUPERSAW_CENTER_INDEX]
    )
    center_phase_inc = freq_profile * center_ratio / sample_rate
    center_phase = center_cumphase - np.floor(center_cumphase)
    center_voice = polyblep_saw(center_phase, center_phase_inc)

    mix_signal = gains[_SUPERSAW_CENTER_INDEX] * center_voice

    for i in range(_SUPERSAW_VOICES):
        if i == _SUPERSAW_CENTER_INDEX:
            continue
        voice = _render_supersaw_voice(
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            cumphase_base=cumphase_base,
            cents=float(cents_per_voice[i]),
            start_phase=float(start_phases[i]),
            sync=sync,
            center_cumphase=center_cumphase,
        )
        mix_signal = mix_signal + gains[i] * voice

    total_gain = center_gain + 6.0 * side_gain
    if total_gain > 1e-6:
        mix_signal = mix_signal / total_gain
    return mix_signal


def _render_supersaw_bank_with_components(
    *,
    freq_profile: np.ndarray,
    sample_rate: int,
    detune: float,
    mix: float,
    sync: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render the supersaw bank and return ``(mix, center_only, sides_only)``.

    All three streams share the same RNG / start phases so downstream split
    routing sees coherent signals (center and sides that sum back to mix).
    """
    (
        cents_per_voice,
        gains,
        start_phases,
        cumphase_base,
        side_gain,
        center_gain,
    ) = _supersaw_setup(
        freq_profile=freq_profile,
        sample_rate=sample_rate,
        detune=detune,
        mix=mix,
        rng=rng,
    )

    center_ratio = float(2.0 ** (cents_per_voice[_SUPERSAW_CENTER_INDEX] / 1200.0))
    center_cumphase = (
        cumphase_base * center_ratio + start_phases[_SUPERSAW_CENTER_INDEX]
    )
    center_phase_inc = freq_profile * center_ratio / sample_rate
    center_phase = center_cumphase - np.floor(center_cumphase)
    center_voice = polyblep_saw(center_phase, center_phase_inc)

    center_only = gains[_SUPERSAW_CENTER_INDEX] * center_voice
    sides_only = np.zeros_like(freq_profile)

    for i in range(_SUPERSAW_VOICES):
        if i == _SUPERSAW_CENTER_INDEX:
            continue
        voice = _render_supersaw_voice(
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            cumphase_base=cumphase_base,
            cents=float(cents_per_voice[i]),
            start_phase=float(start_phases[i]),
            sync=sync,
            center_cumphase=center_cumphase,
        )
        sides_only = sides_only + gains[i] * voice

    total_gain = center_gain + 6.0 * side_gain
    if total_gain > 1e-6:
        center_only = center_only / total_gain
        sides_only = sides_only / total_gain

    mix_signal = center_only + sides_only
    return mix_signal, center_only, sides_only


def _build_spectralwave_partials(
    *,
    position: float,
    n_partials: int,
) -> list[dict[str, Any]]:
    """Build a partial bank that sweeps saw → spectral → square.

    At ``position=0`` the bank approximates a saw (all harmonics, ``1/n``).
    At ``position=1`` the bank approximates a square (odd harmonics only, ``1/n``).
    Intermediate positions emphasize a formant-like family of odd harmonics
    (3, 5, 9, 13, 21) and softens surrounding even harmonics — distinct from
    either endpoint so the morph has audible character at mid positions.
    """
    pos = max(0.0, min(1.0, float(position)))
    formant_boost = 1.0 - abs(pos - 0.5) * 2.0  # 0 at endpoints, 1 at mid
    formant_set = {3, 5, 9, 13, 21}

    partials: list[dict[str, Any]] = []
    for k in range(1, n_partials + 1):
        saw_amp = 1.0 / k
        odd = (k % 2) == 1
        sq_amp = (1.0 / k) if odd else 0.0
        base = (1.0 - pos) * saw_amp + pos * sq_amp
        if k in formant_set and formant_boost > 0.0:
            base += formant_boost * 0.45 / k
        elif (not odd) and formant_boost > 0.0:
            base *= 1.0 - formant_boost * 0.35
        if base < 0.0:
            base = 0.0
        partials.append({"ratio": float(k), "amp": float(base), "phase": 0.0})
    return partials


@numba.njit(cache=True)
def _partial_bank_kernel(
    cumphase_base: np.ndarray,
    mean_freq: float,
    nyquist: float,
    ratios: np.ndarray,
    amps: np.ndarray,
    phases: np.ndarray,
) -> np.ndarray:
    """Accumulate all active partials in a single pass.

    Each partial's per-sample amplitude taper follows ``nyquist_fade``
    (quadratic falloff from 85% of Nyquist to Nyquist, via the partial's
    *mean* frequency — a fixed per-partial scalar, since the main use of
    the taper here is to kill partials whose average frequency is above
    Nyquist while softening those approaching it).
    """
    n = cumphase_base.shape[0]
    n_partials = ratios.shape[0]
    out = np.zeros(n, dtype=np.float64)
    fade_start = nyquist * 0.85

    two_pi = 2.0 * np.pi
    for p in range(n_partials):
        ratio = ratios[p]
        amp_k = amps[p]
        if amp_k <= 1e-6:
            continue
        partial_mean = mean_freq * ratio
        if partial_mean >= nyquist:
            continue
        if partial_mean >= fade_start:
            fade_progress = (partial_mean - fade_start) / (nyquist - fade_start)
            fade = 1.0 - fade_progress
            taper = fade * fade
        else:
            taper = 1.0
        scaled_amp = amp_k * taper
        phase_offset = phases[p]
        for i in range(n):
            out[i] += scaled_amp * np.sin(
                two_pi * cumphase_base[i] * ratio + phase_offset
            )
    return out


def _render_partial_bank(
    *,
    partials: list[dict[str, Any]],
    freq_profile: np.ndarray,
    sample_rate: int,
    start_phase: float,
) -> np.ndarray:
    """Sum-of-sines renderer with Nyquist taper and cumulative phase."""
    n_samples = freq_profile.shape[0]
    nyquist = sample_rate / 2.0
    cumphase_base = np.cumsum(freq_profile / sample_rate) + start_phase

    n_partials = len(partials)
    if n_partials == 0:
        return np.zeros(n_samples, dtype=np.float64)

    ratios = np.empty(n_partials, dtype=np.float64)
    amps = np.empty(n_partials, dtype=np.float64)
    phases = np.empty(n_partials, dtype=np.float64)
    for idx, partial in enumerate(partials):
        ratios[idx] = float(partial["ratio"])
        amps[idx] = float(partial["amp"])
        phases[idx] = float(partial.get("phase", 0.0))

    # Use mean freq for per-partial gating; this matches the old
    # ``nyquist_fade(max(taper))`` early-skip check while keeping the kernel
    # scalar-fast.
    mean_freq = float(np.mean(freq_profile))

    # For per-sample accuracy on sweeping partials, apply the full nyquist
    # fade outside the kernel when any partial is near the fade region. For
    # steady notes (the common case) the kernel's scalar taper is accurate;
    # for trajectories that cross Nyquist mid-note we want the per-sample
    # fade so partials smoothly die out as pitch rises.
    if np.max(freq_profile) * np.max(ratios) >= nyquist * 0.85:
        return _render_partial_bank_with_per_sample_fade(
            cumphase_base=cumphase_base,
            freq_profile=freq_profile,
            nyquist=nyquist,
            ratios=ratios,
            amps=amps,
            phases=phases,
        )

    return _partial_bank_kernel(
        cumphase_base=cumphase_base,
        mean_freq=mean_freq,
        nyquist=nyquist,
        ratios=ratios,
        amps=amps,
        phases=phases,
    )


def _render_partial_bank_with_per_sample_fade(
    *,
    cumphase_base: np.ndarray,
    freq_profile: np.ndarray,
    nyquist: float,
    ratios: np.ndarray,
    amps: np.ndarray,
    phases: np.ndarray,
) -> np.ndarray:
    """Slow-path partial bank with per-sample Nyquist taper.

    Used when a sweeping freq_profile can push partials across the fade
    region mid-note. Still one cumphase, but applies the full vectorised
    taper per-partial rather than the kernel's scalar approximation.
    """
    n_samples = cumphase_base.shape[0]
    signal = np.zeros(n_samples, dtype=np.float64)
    for idx in range(ratios.shape[0]):
        amp_k = float(amps[idx])
        if amp_k <= 1e-6:
            continue
        ratio = float(ratios[idx])
        partial_freq = freq_profile * ratio
        taper = nyquist_fade(partial_freq, nyquist)
        if float(np.max(taper)) <= 1e-6:
            continue
        signal += (
            amp_k
            * taper
            * np.sin(2.0 * np.pi * cumphase_base * ratio + float(phases[idx]))
        )
    return signal


def _resolve_filter_slot_params(
    params: dict[str, Any],
    slot: int,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Resolve per-slot filter params with fallback to top-level defaults."""
    prefix = f"filter{slot}_"
    result: dict[str, Any] = {}
    for key, default_value in defaults.items():
        value = params.get(prefix + key, default_value)
        result[key] = value
    return result


def _apply_va_defaults(params: dict[str, Any]) -> dict[str, Any]:
    """Override shared analog defaults with the VA engine's restrained values.

    Only fills keys the user did not pass. Returns a new dict so the caller's
    ``params`` is not mutated.
    """
    osc_mode = str(params.get("osc_mode", "supersaw")).lower()
    va_defaults: dict[str, Any] = {
        "analog_jitter": _VA_DEFAULT_ANALOG_JITTER,
        "pitch_drift": _VA_DEFAULT_PITCH_DRIFT,
        "voice_card_spread": _VA_DEFAULT_VOICE_CARD_SPREAD,
        # Normalize the drive bypass: whether the caller passed drive_amount=0.0
        # or omitted the key entirely, the render must see the same params
        # tuple (and thus the same RNG seed) so downstream phase/dither draws
        # match.
        "drive_amount": 0.0,
    }
    if osc_mode == "supersaw":
        va_defaults["hpf_cutoff_hz"] = _SUPERSAW_DEFAULT_HPF_HZ
    result = dict(params)
    for key, default in va_defaults.items():
        if key not in result:
            result[key] = default
    return result


def render(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
    freq_trajectory: np.ndarray | None = None,
) -> np.ndarray:
    """Render a VA-flavored note."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    params = _apply_va_defaults(params)

    osc_mode = str(params.get("osc_mode", "supersaw")).lower()
    if osc_mode not in _OSC_MODES:
        raise ValueError(
            f"Unsupported osc_mode: {osc_mode!r}. Use one of {sorted(_OSC_MODES)}."
        )

    filter_routing = str(params.get("filter_routing", "single")).lower()
    if filter_routing not in _FILTER_ROUTINGS:
        raise ValueError(
            f"Unsupported filter_routing: {filter_routing!r}. "
            f"Use one of {sorted(_FILTER_ROUTINGS)}."
        )

    comb_position = str(params.get("comb_position", "off")).lower()
    if comb_position not in _COMB_POSITIONS:
        raise ValueError(
            f"Unsupported comb_position: {comb_position!r}. "
            f"Use one of {sorted(_COMB_POSITIONS)}."
        )

    drive_algorithm = str(params.get("drive_algorithm", "tanh")).lower()
    if drive_algorithm not in _DRIVE_ALGORITHMS:
        raise ValueError(
            f"Unsupported drive_algorithm: {drive_algorithm!r}. "
            f"Use one of {sorted(_DRIVE_ALGORITHMS)}."
        )

    supersaw_detune = float(params.get("supersaw_detune", 0.3))
    supersaw_mix = float(params.get("supersaw_mix", 0.5))
    supersaw_sync = bool(params.get("supersaw_sync", False))

    spectral_position = float(params.get("spectral_position", 0.0))
    n_partials = int(params.get("n_partials", 64))
    spectral_morph_type = str(params.get("spectral_morph_type", "none")).lower()
    if spectral_morph_type not in MORPH_TYPES:
        raise ValueError(
            f"Unsupported spectral_morph_type: {spectral_morph_type!r}. "
            f"Use one of {list(MORPH_TYPES)}."
        )
    spectral_morph_amount = float(params.get("spectral_morph_amount", 0.0))
    spectral_morph_shift = float(params.get("spectral_morph_shift", 0.0))
    spectral_morph_center_k = int(params.get("spectral_morph_center_k", 24))
    spectral_morph_seed = int(params.get("spectral_morph_seed", 0))
    sigma_approximation = bool(params.get("sigma_approximation", False))

    osc2_level = float(params.get("osc2_level", 0.0))
    osc2_semitones = float(params.get("osc2_semitones", 0.0))
    osc2_detune_cents = float(params.get("osc2_detune_cents", 0.0))

    drive_amount = float(params.get("drive_amount", 0.0))

    default_slot_defaults: dict[str, Any] = {
        "cutoff_hz": float(params.get("cutoff_hz", 3000.0)),
        "keytrack": float(params.get("keytrack", 0.0)),
        "reference_freq_hz": float(params.get("reference_freq_hz", 220.0)),
        "resonance_q": float(params.get("resonance_q", 0.707)),
        "filter_env_amount": float(params.get("filter_env_amount", 0.0)),
        "filter_env_decay": float(params.get("filter_env_decay", 0.18)),
        "filter_mode": str(params.get("filter_mode", "lowpass")).lower(),
        "filter_drive": float(params.get("filter_drive", 0.0)),
        "filter_topology": str(params.get("filter_topology", "svf")).lower(),
        "bass_compensation": float(params.get("bass_compensation", 0.0)),
        "filter_morph": float(params.get("filter_morph", 0.0)),
        "hpf_cutoff_hz": float(params.get("hpf_cutoff_hz", 0.0)),
        "hpf_resonance_q": float(params.get("hpf_resonance_q", 0.707)),
        "feedback_amount": float(params.get("feedback_amount", 0.0)),
        "feedback_saturation": float(params.get("feedback_saturation", 0.3)),
    }
    filter1 = _resolve_filter_slot_params(params, 1, default_slot_defaults)
    filter2_defaults = dict(default_slot_defaults)
    filter2_defaults["cutoff_hz"] = float(
        params.get("filter2_cutoff_hz", default_slot_defaults["cutoff_hz"])
    )
    filter2 = _resolve_filter_slot_params(params, 2, filter2_defaults)

    comb_delay_ms = float(params.get("comb_delay_ms", 8.0))
    comb_feedback = float(params.get("comb_feedback", 0.5))
    comb_damping = float(params.get("comb_damping", 0.2))
    comb_keytrack = float(params.get("comb_keytrack", 0.0))
    comb_mix = float(params.get("comb_mix", 0.5))

    for slot_name, slot in (("filter1", filter1), ("filter2", filter2)):
        if slot["cutoff_hz"] <= 0:
            raise ValueError(f"{slot_name}_cutoff_hz must be positive")
        if slot["filter_env_decay"] <= 0:
            raise ValueError(f"{slot_name}_filter_env_decay must be positive")
        if slot["filter_mode"] not in _SUPPORTED_FILTER_MODES:
            raise ValueError(
                f"{slot_name}_filter_mode {slot['filter_mode']!r} is not supported"
            )
        if slot["filter_topology"] not in _SUPPORTED_FILTER_TOPOLOGIES:
            raise ValueError(
                f"{slot_name}_filter_topology {slot['filter_topology']!r} "
                "is not supported"
            )
        if slot["filter_drive"] < 0:
            raise ValueError(f"{slot_name}_filter_drive must be non-negative")

    if not 0.0 <= drive_amount <= 1.0:
        raise ValueError("drive_amount must be in [0, 1]")
    if not 0.0 <= spectral_position <= 1.0:
        raise ValueError("spectral_position must be in [0, 1]")
    if not 0.0 <= spectral_morph_amount <= 1.0:
        raise ValueError("spectral_morph_amount must be in [0, 1]")
    if spectral_morph_center_k < 1:
        raise ValueError("spectral_morph_center_k must be >= 1")
    if n_partials < 1:
        raise ValueError("n_partials must be >= 1")
    if n_partials > _MAX_PARTIALS:
        raise ValueError(f"n_partials must be <= {_MAX_PARTIALS}")
    if not 0.0 <= osc2_level <= 1.0:
        raise ValueError("osc2_level must be in [0, 1]")
    if comb_delay_ms <= 0:
        raise ValueError("comb_delay_ms must be positive")
    if not 0.0 <= comb_feedback <= 0.99:
        raise ValueError("comb_feedback must be in [0, 0.99]")
    if not 0.0 <= comb_damping <= 1.0:
        raise ValueError("comb_damping must be in [0, 1]")
    if not 0.0 <= comb_keytrack <= 1.0:
        raise ValueError("comb_keytrack must be in [0, 1]")
    if not 0.0 <= comb_mix <= 1.0:
        raise ValueError("comb_mix must be in [0, 1]")

    analog = extract_analog_params(params)
    pitch_drift = analog["pitch_drift"]
    analog_jitter = analog["analog_jitter"]
    noise_floor_level = analog["noise_floor"]
    drift_rate_hz = analog["drift_rate_hz"]
    cutoff_drift_amount = analog["cutoff_drift"]

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0)

    if freq_trajectory is not None:
        freq_trajectory = np.asarray(freq_trajectory, dtype=np.float64)
        if freq_trajectory.ndim != 1:
            raise ValueError("freq_trajectory must be one-dimensional")
        if freq_trajectory.size != n_samples:
            raise ValueError("freq_trajectory length must match note duration")
        if np.any(freq_trajectory <= 0):
            raise ValueError("freq_trajectory values must be positive")
        freq_profile = freq_trajectory
    else:
        freq_profile = np.full(n_samples, freq, dtype=np.float64)

    rng = rng_for_note(
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=sample_rate,
        params=params,
    )
    freq_profile, amp, vc_cutoff1, vc_offsets = apply_voice_card(
        params,
        voice_card_spread=analog["voice_card_spread"],
        pitch_spread=analog["voice_card_pitch_spread"],
        filter_spread=analog["voice_card_filter_spread"],
        envelope_spread=analog["voice_card_envelope_spread"],
        osc_spread=analog["voice_card_osc_spread"],
        level_spread=analog["voice_card_level_spread"],
        freq_profile=freq_profile,
        amp=amp,
        cutoff_hz=filter1["cutoff_hz"],
    )
    # apply_voice_card only returns None for cutoff when None is passed in;
    # we always pass a float, so the return is always float.
    filter1["cutoff_hz"] = float(vc_cutoff1) if vc_cutoff1 is not None else 0.0

    # Apply an independent voice-card cutoff offset to filter2 so the two
    # slots drift independently rather than sharing a single multiplicative
    # scale. We derive an independent offset from ``voice_card_offsets``
    # using a per-slot name suffix so each slot gets its own deterministic
    # shift.
    voice_name = str(params.get("_voice_name", ""))
    if voice_name and analog["voice_card_filter_spread"] > 0.0:
        vc_filter2 = voice_card_offsets(f"{voice_name}:filter2")
        filter2["cutoff_hz"] = float(filter2["cutoff_hz"]) * (
            2.0
            ** (
                vc_filter2["cutoff_offset_cents"]
                * analog["voice_card_filter_spread"]
                / 1200.0
            )
        )

    jittered = apply_note_jitter(params, rng, analog_jitter)
    start_phase = float(jittered.get("_phase_offset", 0.0))
    amp_jitter_db = float(jittered.get("_amp_jitter_db", 0.0))

    filter1["resonance_q"], drift_rate_hz = apply_voice_card_post_offsets(
        filter1["resonance_q"], drift_rate_hz, vc_offsets
    )
    if voice_name and analog["voice_card_filter_spread"] > 0.0:
        vc_filter2_res = voice_card_offsets(f"{voice_name}:filter2")
        res2_pct = (
            vc_filter2_res["resonance_offset_pct"] * analog["voice_card_filter_spread"]
        )
        filter2["resonance_q"] = max(
            0.5, filter2["resonance_q"] * (1.0 + res2_pct / 100.0)
        )
    else:
        filter2["resonance_q"] = max(
            0.5,
            filter2["resonance_q"] * (1.0 + vc_offsets["resonance_offset_pct"] / 100.0),
        )

    freq_profile = apply_pitch_cv_dither(
        freq_profile,
        analog_jitter=analog_jitter,
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=sample_rate,
        n_samples=n_samples,
    )

    if pitch_drift > 0:
        drift_multiplier = build_drift(
            n_samples=n_samples,
            drift_amount=pitch_drift,
            drift_rate_hz=drift_rate_hz,
            duration=duration,
            phase_offset=start_phase,
            rng=rng,
        )
        freq_profile = freq_profile * drift_multiplier

    # ---- Oscillator rendering ----
    signal_a: np.ndarray
    signal_b: np.ndarray
    if osc_mode == "supersaw":
        if filter_routing == "split":
            raw_signal, center_only, sides_only = _render_supersaw_bank_with_components(
                freq_profile=freq_profile,
                sample_rate=sample_rate,
                detune=supersaw_detune,
                mix=supersaw_mix,
                sync=supersaw_sync,
                rng=rng,
            )
            signal_a = center_only
            signal_b = sides_only
        else:
            raw_signal = _render_supersaw_bank(
                freq_profile=freq_profile,
                sample_rate=sample_rate,
                detune=supersaw_detune,
                mix=supersaw_mix,
                sync=supersaw_sync,
                rng=rng,
            )
            signal_a = raw_signal
            signal_b = np.zeros_like(raw_signal)
    else:
        partials = _build_spectralwave_partials(
            position=spectral_position,
            n_partials=n_partials,
        )
        if spectral_morph_type != "none" and spectral_morph_amount != 0.0:
            partials = apply_spectral_morph(
                partials,
                morph_type=spectral_morph_type,
                amount=spectral_morph_amount,
                shift=spectral_morph_shift,
                center_k=spectral_morph_center_k,
                seed=spectral_morph_seed,
            )
        if sigma_approximation:
            partials = apply_sigma_approximation(partials)

        raw_signal = _render_partial_bank(
            partials=partials,
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            start_phase=start_phase,
        )

        if osc2_level > 0.0:
            osc2_ratio = float(2.0 ** (osc2_semitones / 12.0)) * float(
                2.0 ** (osc2_detune_cents / 1200.0)
            )
            osc2_signal = _render_partial_bank(
                partials=partials,
                freq_profile=freq_profile * osc2_ratio,
                sample_rate=sample_rate,
                start_phase=start_phase * 1.618,
            )
            signal_a = raw_signal
            signal_b = osc2_signal
            raw_signal = (raw_signal + osc2_level * osc2_signal) / (1.0 + osc2_level)
        else:
            signal_a = raw_signal
            signal_b = np.zeros_like(raw_signal)

    peak = float(np.max(np.abs(raw_signal)))
    if peak > 1e-9:
        raw_signal = raw_signal / peak
        if filter_routing == "split":
            signal_a = signal_a / peak
            signal_b = signal_b / peak

    # ---- Drive stage (pre-filter saturation) ----
    if drive_amount > 0.0:
        raw_signal = apply_waveshaper(
            raw_signal, algorithm=drive_algorithm, drive=drive_amount, mix=1.0
        )
        if filter_routing == "split":
            signal_a = apply_waveshaper(
                signal_a, algorithm=drive_algorithm, drive=drive_amount, mix=1.0
            )
            signal_b = apply_waveshaper(
                signal_b, algorithm=drive_algorithm, drive=drive_amount, mix=1.0
            )

    # ---- Cutoff profiles ----
    nyquist = sample_rate / 2.0
    cutoff1_profile = build_keytracked_cutoff_profile(
        cutoff_hz=filter1["cutoff_hz"],
        keytrack=filter1["keytrack"],
        reference_freq_hz=filter1["reference_freq_hz"],
        filter_env_amount=filter1["filter_env_amount"],
        filter_env_decay=filter1["filter_env_decay"],
        duration=duration,
        n_samples=n_samples,
        freq_profile=freq_profile,
        nyquist=nyquist,
    )
    cutoff2_profile = build_keytracked_cutoff_profile(
        cutoff_hz=filter2["cutoff_hz"],
        keytrack=filter2["keytrack"],
        reference_freq_hz=filter2["reference_freq_hz"],
        filter_env_amount=filter2["filter_env_amount"],
        filter_env_decay=filter2["filter_env_decay"],
        duration=duration,
        n_samples=n_samples,
        freq_profile=freq_profile,
        nyquist=nyquist,
    )

    if cutoff_drift_amount > 0:
        cutoff_rng = rng_for_note(
            freq=freq,
            duration=duration,
            amp=amp,
            sample_rate=sample_rate,
            extra_seed="cutoff_drift",
        )
        cutoff_mod = build_cutoff_drift(
            n_samples,
            amount_cents=30.0 * cutoff_drift_amount,
            rate_hz=0.3,
            rng=cutoff_rng,
            sample_rate=sample_rate,
        )
        cutoff1_profile = np.clip(cutoff1_profile * cutoff_mod, 20.0, nyquist * 0.98)
        cutoff2_profile = np.clip(cutoff2_profile * cutoff_mod, 20.0, nyquist * 0.98)

    cutoff1_profile = apply_cutoff_cv_dither(
        cutoff1_profile,
        analog_jitter=analog_jitter,
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=sample_rate,
        n_samples=n_samples,
        nyquist=nyquist,
    )
    cutoff2_profile = apply_cutoff_cv_dither(
        cutoff2_profile,
        analog_jitter=analog_jitter,
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=sample_rate,
        n_samples=n_samples,
        nyquist=nyquist,
    )

    # ---- Comb delay profile (if enabled) ----
    comb_delay_base_samples = comb_delay_ms * 0.001 * sample_rate
    if comb_keytrack > 0.0:
        keytrack_delay = sample_rate / np.maximum(freq_profile, 20.0)
        delay_profile = (
            1.0 - comb_keytrack
        ) * comb_delay_base_samples + comb_keytrack * keytrack_delay
    else:
        delay_profile = np.full(n_samples, comb_delay_base_samples, dtype=np.float64)

    def _comb(sig: np.ndarray, *, mix: float) -> np.ndarray:
        return apply_comb(
            sig,
            delay_samples_profile=delay_profile,
            feedback=comb_feedback,
            damping=comb_damping,
            mix=mix,
            sample_rate=sample_rate,
        )

    # ---- Filter routing ----
    pre = raw_signal
    if comb_position == "pre_filter":
        pre = _comb(pre, mix=1.0)

    def _f1(sig: np.ndarray) -> np.ndarray:
        return apply_filter(
            sig,
            cutoff_profile=cutoff1_profile,
            resonance_q=filter1["resonance_q"],
            sample_rate=sample_rate,
            filter_mode=filter1["filter_mode"],
            filter_drive=filter1["filter_drive"],
            filter_even_harmonics=0.0,
            filter_topology=filter1["filter_topology"],
            bass_compensation=filter1["bass_compensation"],
            filter_morph=filter1["filter_morph"],
            hpf_cutoff_hz=filter1["hpf_cutoff_hz"],
            hpf_resonance_q=filter1["hpf_resonance_q"],
            feedback_amount=filter1["feedback_amount"],
            feedback_saturation=filter1["feedback_saturation"],
        )

    def _f2(sig: np.ndarray) -> np.ndarray:
        return apply_filter(
            sig,
            cutoff_profile=cutoff2_profile,
            resonance_q=filter2["resonance_q"],
            sample_rate=sample_rate,
            filter_mode=filter2["filter_mode"],
            filter_drive=filter2["filter_drive"],
            filter_even_harmonics=0.0,
            filter_topology=filter2["filter_topology"],
            bass_compensation=filter2["bass_compensation"],
            filter_morph=filter2["filter_morph"],
            hpf_cutoff_hz=filter2["hpf_cutoff_hz"],
            hpf_resonance_q=filter2["hpf_resonance_q"],
            feedback_amount=filter2["feedback_amount"],
            feedback_saturation=filter2["feedback_saturation"],
        )

    if filter_routing == "single":
        filtered = _f1(pre)
    elif filter_routing == "serial":
        filtered = _f2(_f1(pre))
    elif filter_routing == "parallel":
        filtered = 0.5 * (_f1(pre) + _f2(pre))
    else:  # split
        filtered = _f1(signal_a) + _f2(signal_b)

    if comb_position == "post_filter":
        filtered = _comb(filtered, mix=1.0)
    elif comb_position == "parallel":
        comb_out = _comb(raw_signal, mix=1.0)
        filtered = (1.0 - comb_mix) * filtered + comb_mix * comb_out

    filtered = apply_analog_post_processing(
        filtered,
        rng=rng,
        amp_jitter_db=amp_jitter_db,
        noise_floor_level=noise_floor_level,
        sample_rate=sample_rate,
        n_samples=n_samples,
    )

    peak = float(np.max(np.abs(filtered)))
    if peak > 1e-9:
        filtered /= peak
    return amp * filtered
