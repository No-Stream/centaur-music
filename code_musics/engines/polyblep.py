"""PolyBLEP synthesis engine.

Generates waveforms in the time domain with polynomial bandlimiting corrections
at discontinuities. Produces smooth analog character with correct 1/n harmonic
spectrum and no Gibbs phenomenon, unlike additive-truncated engines.

Supported waveforms:
- ``saw``      — bandlimited sawtooth via direct PolyBLEP correction
- ``square``   — bandlimited square/pulse as the difference of two saws
- ``triangle`` — bandlimited triangle obtained by integrating the square wave
  (BLAMP approach); ``pulse_width`` is ignored for triangle
- ``sine``     — pure sine wave (no harmonics, no antialiasing needed)
"""

from __future__ import annotations

import math
from typing import Any

import numba
import numpy as np

from code_musics.engines._dsp_utils import (
    apply_analog_post_processing,
    apply_cutoff_cv_dither,
    apply_filter_oversampled,
    apply_note_jitter,
    apply_pitch_cv_dither,
    apply_voice_card,
    apply_voice_card_post_offsets,
    build_cutoff_drift,
    build_drift,
    build_keytracked_cutoff_profile,
    extract_analog_params,
    resolve_quality_mode,
    rng_for_note,
)
from code_musics.engines._filters import (
    _SUPPORTED_FILTER_MODES,
    _SUPPORTED_FILTER_TOPOLOGIES,
)
from code_musics.engines._oscillators import (
    render_polyblep_oscillator as _render_oscillator_with_phase,
)


@numba.njit(cache=True)
def _apply_one_pole_lowpass(signal: np.ndarray, alpha: float) -> np.ndarray:
    """Single-pole IIR lowpass: y[n] = alpha*x[n] + (1-alpha)*y[n-1]."""
    n = signal.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[0] = signal[0] * alpha
    one_minus_alpha = 1.0 - alpha
    for i in range(1, n):
        out[i] = alpha * signal[i] + one_minus_alpha * out[i - 1]
    return out


def _apply_osc_softness(
    signal: np.ndarray,
    *,
    softness: float,
    freq: float,
    sample_rate: int,
) -> np.ndarray:
    """Bandwidth-limit a waveform via a frequency-tracking one-pole lowpass."""
    if softness <= 0.0:
        return signal
    harmonic_multiplier = 10.0 + 40.0 * (1.0 - softness)
    cutoff_freq = freq * harmonic_multiplier
    cutoff_freq = min(cutoff_freq, sample_rate * 0.45)
    alpha = float(np.clip(2.0 * math.pi * cutoff_freq / sample_rate, 0.0, 1.0))
    if alpha >= 1.0:
        return signal
    return _apply_one_pole_lowpass(signal, alpha)


def _apply_osc_asymmetry(
    signal: np.ndarray,
    phase: np.ndarray,
    *,
    asymmetry: float,
    waveform: str,
) -> np.ndarray:
    """Apply waveform asymmetry: saw reset softening or square PW shift."""
    if asymmetry <= 0.0 or waveform not in {"saw", "square"}:
        return signal
    if waveform == "saw":
        sine_component = np.sin(2.0 * np.pi * phase)
        blend = asymmetry * 0.3
        return signal * (1.0 - blend) + blend * sine_component
    return signal


def _build_shape_drift_profile(
    n_samples: int,
    *,
    shape_drift: float,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a slow O-U drift profile for waveform shape modulation.

    Returns an array in roughly [-1, 1] range that modulates waveform shape
    parameters over time.  At shape_drift=1.0, the peak excursion is ~3%.
    """
    if shape_drift <= 0.0 or n_samples == 0:
        return np.zeros(n_samples, dtype=np.float64)
    drift = build_cutoff_drift(
        n_samples,
        amount_cents=30.0 * shape_drift,
        rate_hz=0.15,
        rng=rng,
        sample_rate=sample_rate,
    )
    centered = drift - 1.0
    peak = np.max(np.abs(centered))
    if peak > 1e-12:
        centered = centered / peak
    return centered * shape_drift * 0.03


def render(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
    freq_trajectory: np.ndarray | None = None,
    param_profiles: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Render a bandlimited oscillator with a driven ZDF/TPT filter sweep.

    The ``quality`` param selects the ladder solver + oversampling factor
    applied to the filter + feedback + dither block (oscillators are
    generated at the base rate since BLEP handles their aliasing):

    - ``draft`` — ADAA ladder, no oversampling
    - ``fast`` — Newton ladder (2 iters), 2x oversampling
    - ``great`` — Newton ladder (4 iters), 2x oversampling (default)
    - ``divine`` — Newton ladder (8 iters), 4x oversampling
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    waveform = str(params.get("waveform", "saw")).lower()
    pulse_width = float(params.get("pulse_width", 0.5))
    osc2_level = float(params.get("osc2_level", 0.0))
    osc2_waveform = str(params.get("osc2_waveform", waveform)).lower()
    osc2_pulse_width = float(params.get("osc2_pulse_width", pulse_width))
    osc2_detune_cents = float(params.get("osc2_detune_cents", 0.0))
    osc2_semitones = float(params.get("osc2_semitones", 0.0))
    osc2_spread_power = float(params.get("osc2_spread_power", 1.0))
    cutoff_hz = float(params.get("cutoff_hz", 3000.0))
    keytrack = float(params.get("keytrack", 0.0))
    reference_freq_hz = float(params.get("reference_freq_hz", 220.0))
    resonance_q = float(params.get("resonance_q", 0.707))
    filter_env_amount = float(params.get("filter_env_amount", 0.0))
    filter_env_decay = float(params.get("filter_env_decay", 0.18))
    filter_mode = str(params.get("filter_mode", "lowpass")).lower()
    filter_drive = float(params.get("filter_drive", 0.0))
    filter_even_harmonics = float(params.get("filter_even_harmonics", 0.0))
    filter_topology = str(params.get("filter_topology", "svf")).lower()
    bass_compensation = float(params.get("bass_compensation", 0.0))
    filter_morph = float(params.get("filter_morph", 0.0))
    hpf_cutoff_hz = float(params.get("hpf_cutoff_hz", 0.0))
    hpf_resonance_q = float(params.get("hpf_resonance_q", 0.707))
    feedback_amount = float(params.get("feedback_amount", 0.0))
    feedback_saturation = float(params.get("feedback_saturation", 0.3))
    analog = extract_analog_params(params)
    pitch_drift = analog["pitch_drift"]
    analog_jitter = analog["analog_jitter"]
    noise_floor_level = analog["noise_floor"]
    drift_rate_hz = analog["drift_rate_hz"]
    cutoff_drift_amount = analog["cutoff_drift"]
    osc_asymmetry = analog["osc_asymmetry"]
    osc_softness = analog["osc_softness"]
    osc_dc_offset = analog["osc_dc_offset"]
    osc_shape_drift = analog["osc_shape_drift"]
    quality_config = resolve_quality_mode(str(analog["quality"]))

    if cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be positive")
    if reference_freq_hz <= 0:
        raise ValueError("reference_freq_hz must be positive")
    if filter_env_decay <= 0:
        raise ValueError("filter_env_decay must be positive")
    if not 0.0 < pulse_width < 1.0:
        raise ValueError("pulse_width must be between 0 and 1")
    if not 0.0 < osc2_pulse_width < 1.0:
        raise ValueError("osc2_pulse_width must be between 0 and 1")
    if waveform not in {"saw", "square", "triangle", "sine"}:
        raise ValueError(
            f"Unsupported waveform: {waveform!r}. Use 'saw', 'square', 'triangle', or 'sine'."
        )
    if osc2_waveform not in {"saw", "square", "triangle", "sine"}:
        raise ValueError(
            "Unsupported osc2_waveform: "
            f"{osc2_waveform!r}. Use 'saw', 'square', 'triangle', or 'sine'."
        )
    if filter_mode not in _SUPPORTED_FILTER_MODES:
        raise ValueError(
            f"Unsupported filter_mode: {filter_mode!r}. "
            "Use 'lowpass', 'bandpass', 'highpass', or 'notch'."
        )
    if filter_drive < 0:
        raise ValueError("filter_drive must be non-negative")
    if filter_topology not in _SUPPORTED_FILTER_TOPOLOGIES:
        raise ValueError(
            f"Unsupported filter_topology: {filter_topology!r}. Use 'svf' or 'ladder'."
        )
    if osc2_level < 0:
        raise ValueError("osc2_level must be non-negative")

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0)

    if freq_trajectory is not None:
        freq_trajectory = np.asarray(freq_trajectory, dtype=np.float64)
        if freq_trajectory.ndim != 1:
            raise ValueError("freq_trajectory must be one-dimensional")
        if freq_trajectory.size != n_samples:
            raise ValueError("freq_trajectory length must match note duration")
        freq_profile = freq_trajectory
    else:
        freq_profile = np.full(n_samples, freq, dtype=np.float64)

    # --- Analog character: RNG, jitter, drift ---
    rng = rng_for_note(
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=sample_rate,
        params=params,
    )
    # Voice card calibration — persistent per-voice character
    freq_profile, amp, cutoff_hz, vc_offsets = apply_voice_card(
        params,
        voice_card_spread=analog["voice_card_spread"],
        pitch_spread=analog["voice_card_pitch_spread"],
        filter_spread=analog["voice_card_filter_spread"],
        envelope_spread=analog["voice_card_envelope_spread"],
        osc_spread=analog["voice_card_osc_spread"],
        level_spread=analog["voice_card_level_spread"],
        freq_profile=freq_profile,
        amp=amp,
        cutoff_hz=cutoff_hz,
    )

    jittered = apply_note_jitter(params, rng, analog_jitter)
    cutoff_hz = float(jittered.get("cutoff_hz", cutoff_hz))
    resonance_q = float(jittered.get("resonance_q", resonance_q))
    resonance_q, drift_rate_hz = apply_voice_card_post_offsets(
        resonance_q, drift_rate_hz, vc_offsets
    )
    filter_env_decay = float(jittered.get("filter_env_decay", filter_env_decay))
    start_phase = float(jittered.get("_phase_offset", 0.0))
    amp_jitter_db = float(jittered.get("_amp_jitter_db", 0.0))

    attack = float(params.get("attack", 0.04)) * vc_offsets["attack_scale"]
    release = float(params.get("release", 0.1)) * vc_offsets["release_scale"]
    _ = attack, release  # available for ADSR when wired

    pulse_width = min(0.99, max(0.01, pulse_width + vc_offsets["pulse_width_offset"]))
    osc_softness = max(0.0, osc_softness + vc_offsets["softness_offset"])

    # OB-Xd-style per-sample CV dither on pitch (layered on top of stable
    # voice_card offsets).  Uses ``analog_jitter`` as the amount knob so we
    # don't introduce a new surface for what is effectively a finer layer of
    # the existing per-note jitter concept.
    freq_profile = apply_pitch_cv_dither(
        freq_profile,
        analog_jitter=analog_jitter,
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=sample_rate,
        n_samples=n_samples,
    )

    # Asymmetry shifts square PW slightly
    if osc_asymmetry > 0.0 and waveform == "square":
        pulse_width = min(0.99, max(0.01, pulse_width + osc_asymmetry * 0.04 - 0.02))
    if osc_asymmetry > 0.0 and osc2_waveform == "square" and osc2_level > 0.0:
        osc2_pulse_width = min(
            0.99, max(0.01, osc2_pulse_width + osc_asymmetry * 0.04 - 0.02)
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

    # Shape drift profile (shared between osc1 and osc2)
    shape_drift_profile: np.ndarray | None = None
    if osc_shape_drift > 0.0:
        shape_rng = rng_for_note(
            freq=freq,
            duration=duration,
            amp=amp,
            sample_rate=sample_rate,
            extra_seed="shape_drift",
        )
        shape_drift_profile = _build_shape_drift_profile(
            n_samples,
            shape_drift=osc_shape_drift,
            sample_rate=sample_rate,
            rng=shape_rng,
        )

    raw_signal, osc1_phase = _render_oscillator_with_phase(
        waveform=waveform,
        pulse_width=pulse_width,
        freq_profile=freq_profile,
        sample_rate=sample_rate,
        start_phase=start_phase,
    )

    # Oscillator imperfections for osc1
    if osc_asymmetry > 0.0:
        raw_signal = _apply_osc_asymmetry(
            raw_signal, osc1_phase, asymmetry=osc_asymmetry, waveform=waveform
        )
    if shape_drift_profile is not None:
        raw_signal = _apply_shape_drift(
            raw_signal, osc1_phase, drift_profile=shape_drift_profile, waveform=waveform
        )
    mean_freq = float(np.mean(freq_profile))
    raw_signal = _apply_osc_softness(
        raw_signal, softness=osc_softness, freq=mean_freq, sample_rate=sample_rate
    )
    if osc_dc_offset > 0.0:
        dc_sign = 1.0 if rng.integers(2) == 0 else -1.0
        raw_signal = raw_signal + osc_dc_offset * 0.05 * dc_sign

    if osc2_level > 0.0:
        effective_detune = osc2_detune_cents * pow(0.5, osc2_spread_power - 1.0)
        osc2_ratio = float(2.0 ** (osc2_semitones / 12.0)) * float(
            2.0 ** (effective_detune / 1200.0)
        )
        osc2_signal, osc2_phase = _render_oscillator_with_phase(
            waveform=osc2_waveform,
            pulse_width=osc2_pulse_width,
            freq_profile=freq_profile * osc2_ratio,
            sample_rate=sample_rate,
            start_phase=start_phase * 1.618,
        )

        # Osc2 imperfections (different DC sign via separate RNG draw)
        if osc_asymmetry > 0.0:
            osc2_signal = _apply_osc_asymmetry(
                osc2_signal, osc2_phase, asymmetry=osc_asymmetry, waveform=osc2_waveform
            )
        if shape_drift_profile is not None:
            osc2_signal = _apply_shape_drift(
                osc2_signal,
                osc2_phase,
                drift_profile=shape_drift_profile,
                waveform=osc2_waveform,
            )
        osc2_mean_freq = mean_freq * osc2_ratio
        osc2_signal = _apply_osc_softness(
            osc2_signal,
            softness=osc_softness,
            freq=osc2_mean_freq,
            sample_rate=sample_rate,
        )
        if osc_dc_offset > 0.0:
            dc_sign_2 = 1.0 if rng.integers(2) == 0 else -1.0
            osc2_signal = osc2_signal + osc_dc_offset * 0.05 * dc_sign_2

        raw_signal = (raw_signal + osc2_level * osc2_signal) / (1.0 + osc2_level)

    # Cutoff envelope (identical pattern to filtered_stack / va).
    # When a per-sample ``cutoff_hz`` profile is supplied via
    # ``param_profiles`` it replaces the scalar base.  This is how the
    # modulation matrix rides cutoff at audio rate; scalar paths below
    # (keytrack, envelope, drift, CV dither) stack on top of it.
    nyquist = sample_rate / 2.0
    if param_profiles is not None and "cutoff_hz" in param_profiles:
        cutoff_profile_base = np.asarray(param_profiles["cutoff_hz"], dtype=np.float64)
        if cutoff_profile_base.shape != (n_samples,):
            raise ValueError(
                "cutoff_hz profile length must match note duration in samples"
            )
        t = np.linspace(0.0, duration, n_samples, endpoint=False)
        cutoff_envelope = np.maximum(
            1.0 + filter_env_amount * np.exp(-t / filter_env_decay), 0.05
        )
        keytracked_cutoff = cutoff_profile_base * np.power(
            freq_profile / reference_freq_hz, keytrack
        )
        cutoff_profile = np.clip(
            keytracked_cutoff * cutoff_envelope, 20.0, nyquist * 0.98
        )
    else:
        cutoff_profile = build_keytracked_cutoff_profile(
            cutoff_hz=cutoff_hz,
            keytrack=keytrack,
            reference_freq_hz=reference_freq_hz,
            filter_env_amount=filter_env_amount,
            filter_env_decay=filter_env_decay,
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
        cutoff_modulation = build_cutoff_drift(
            n_samples,
            amount_cents=30.0 * cutoff_drift_amount,
            rate_hz=0.3,
            rng=cutoff_rng,
            sample_rate=sample_rate,
        )
        cutoff_profile = np.clip(
            cutoff_profile * cutoff_modulation, 20.0, nyquist * 0.98
        )

    # OB-Xd-style per-sample CV dither on the filter cutoff.  Same knob as
    # the pitch dither above (``analog_jitter``), just a different target.
    cutoff_profile = apply_cutoff_cv_dither(
        cutoff_profile,
        analog_jitter=analog_jitter,
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=sample_rate,
        n_samples=n_samples,
        nyquist=nyquist,
    )

    filtered = apply_filter_oversampled(
        raw_signal,
        cutoff_profile=cutoff_profile,
        resonance_q=resonance_q,
        sample_rate=sample_rate,
        oversample_factor=quality_config.oversample_factor,
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
        filter_solver=quality_config.solver,
        max_newton_iters=quality_config.max_newton_iters,
        newton_tolerance=quality_config.newton_tolerance,
    )

    filtered = apply_analog_post_processing(
        filtered,
        rng=rng,
        amp_jitter_db=amp_jitter_db,
        noise_floor_level=noise_floor_level,
        sample_rate=sample_rate,
        n_samples=n_samples,
    )

    # Peak-normalize (filter sweep causes uneven amplitude)
    peak = np.max(np.abs(filtered))
    if peak > 1e-9:
        filtered /= peak

    return amp * filtered


def _apply_shape_drift(
    signal: np.ndarray,
    phase: np.ndarray,
    *,
    drift_profile: np.ndarray,
    waveform: str,
) -> np.ndarray:
    """Apply time-varying shape modulation via a drift profile.

    For saw: modulates the asymmetry blend over time.
    For square: modulates the output by blending with a sine at the drift rate.
    """
    if waveform == "saw":
        sine_component = np.sin(2.0 * np.pi * phase)
        blend = np.abs(drift_profile)
        return signal * (1.0 - blend) + blend * sine_component
    if waveform == "square":
        sine_component = np.sin(2.0 * np.pi * phase)
        blend = np.abs(drift_profile)
        return signal * (1.0 - blend) + blend * sine_component
    return signal
