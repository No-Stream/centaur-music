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
    apply_polyblep_step_correction,
    apply_transient_state,
    apply_voice_card,
    apply_voice_card_post_offsets,
    build_cutoff_drift,
    build_drift,
    build_keytracked_cutoff_profile,
    extract_analog_params,
    resolve_quality_mode,
    resolve_transient_mode,
    rng_for_note,
    snapshot_voice_state,
)
from code_musics.engines._filters import (
    _SUPPORTED_FILTER_MODES,
    _SUPPORTED_FILTER_TOPOLOGIES,
)
from code_musics.engines._oscillators import (
    render_polyblep_oscillator as _render_oscillator_with_phase,
)
from code_musics.engines._voice_dist import apply_voice_dist

# Scale factor for per-sample oscillator phase-noise in cycles.  At
# ``osc_phase_noise=1.0`` the per-sample perturbation has peak amplitude
# ~1e-4 cycles, roughly 1 cent of instantaneous pitch jitter at 220 Hz —
# small enough to read as zero-crossing texture rather than pitch wobble.
# See plans/let-s-think-deeply-about-abundant-clock.md (Track B.3).
_PHASE_NOISE_SCALE: float = 1e-4


@numba.njit(cache=True)
def _find_sync_events_kernel(
    osc1_cumphase: np.ndarray,
    osc1_phase_inc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect osc1 wrap events for hard sync, supporting multi-wrap intervals.

    In extreme FM / pitch scenarios where ``freq > sample_rate/2`` a single
    inter-sample interval can contain multiple integer boundary crossings;
    when that happens we emit one event per crossing, spaced proportionally
    across ``(pre_idx, pre_idx+1]``.

    Returns ``(event_samples, event_fractions)`` where the continuous-sample
    time of event ``k`` is ``event_samples[k] + event_fractions[k]`` with
    ``0 <= fraction < 1``.  ``event_sample`` is the integer index of the
    sample immediately BEFORE the crossing.
    """
    n = osc1_cumphase.shape[0]
    if n < 2:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )

    # First pass: count total events so we can allocate exact-sized output.
    total_events = 0
    for i in range(1, n):
        diff = int(np.floor(osc1_cumphase[i]) - np.floor(osc1_cumphase[i - 1]))
        if diff >= 1:
            total_events += diff

    event_samples = np.empty(total_events, dtype=np.int64)
    event_fractions = np.empty(total_events, dtype=np.float64)

    # Second pass: fill events.  For each interval with ``diff`` crossings,
    # place events proportionally using the instantaneous slope at the
    # post sample.
    idx = 0
    for i in range(1, n):
        prev_floor = np.floor(osc1_cumphase[i - 1])
        cur_floor = np.floor(osc1_cumphase[i])
        diff = int(cur_floor - prev_floor)
        if diff < 1:
            continue
        phase_inc_here = osc1_phase_inc[i]
        if phase_inc_here <= 0.0:
            phase_inc_here = 1.0
        pre_cum = osc1_cumphase[i - 1]
        for k in range(diff):
            boundary = prev_floor + float(k + 1)
            frac = (boundary - pre_cum) / phase_inc_here
            if frac < 0.0:
                frac = 0.0
            elif frac > 0.999999:
                frac = 0.999999
            event_samples[idx] = i - 1
            event_fractions[idx] = frac
            idx += 1

    return event_samples, event_fractions


def _find_sync_events(
    osc1_cumphase: np.ndarray,
    osc1_phase_inc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect osc1 wrap events for hard sync (Python wrapper)."""
    return _find_sync_events_kernel(
        np.ascontiguousarray(osc1_cumphase, dtype=np.float64),
        np.ascontiguousarray(osc1_phase_inc, dtype=np.float64),
    )


@numba.njit(cache=True)
def _render_synced_saw_numba(
    freq_profile: np.ndarray,
    sample_rate: int,
    start_phase: float,
    sync_event_samples: np.ndarray,
    sync_event_fractions: np.ndarray,
) -> np.ndarray:
    """Render a hard-synced PolyBLEP sawtooth.

    osc2 phase is advanced by ``freq_profile[i] / sample_rate`` each sample,
    wrapping naturally at 1.0 and forced-reset to 0 at every sync event.
    BLEP step corrections are injected at both natural wraps and sync resets.

    ``sync_event_samples`` are ascending integer sample indices at which
    osc1 wrapped; ``sync_event_fractions`` gives the fractional offset of
    the wrap inside that sample interval (``0 <= frac < 1``).  The event at
    index ``k`` occurs at continuous-sample time ``k + frac``.
    """
    n = freq_profile.shape[0]
    signal = np.empty(n, dtype=np.float64)
    phase = start_phase % 1.0
    prev_phase = phase
    sync_idx = 0
    n_syncs = sync_event_samples.shape[0]

    for i in range(n):
        phase_inc = freq_profile[i] / sample_rate
        phase = prev_phase + phase_inc
        natural_wrapped = False
        natural_frac = 0.0
        if phase >= 1.0:
            natural_wrapped = True
            if phase_inc > 0.0:
                natural_frac = (1.0 - prev_phase) / phase_inc
                if natural_frac < 0.0:
                    natural_frac = 0.0
                elif natural_frac > 0.999999:
                    natural_frac = 0.999999
            phase -= 1.0
        sync_here = False
        sync_frac = 0.0
        if sync_idx < n_syncs and sync_event_samples[sync_idx] == i - 1:
            # sync event lies in the interval (i-1, i]
            sync_frac = sync_event_fractions[sync_idx]
            sync_here = True
            sync_idx += 1

        if sync_here:
            # osc2 phase just before sync = prev_phase + sync_frac*phase_inc
            # (mod 1, in case osc2 wrapped earlier in this same sample interval).
            # After sync it resets to 0 and advances (1 - sync_frac)*phase_inc.
            pre_sync_value = 2.0 * ((prev_phase + sync_frac * phase_inc) % 1.0) - 1.0
            post_sync_sample_phase = (1.0 - sync_frac) * phase_inc
            phase = post_sync_sample_phase
            naive_value = 2.0 * phase - 1.0
            signal[i] = naive_value
            step = naive_value - pre_sync_value
            apply_polyblep_step_correction(
                signal,
                i - 1,
                sync_frac,
                step,
            )
        else:
            naive_value = 2.0 * phase - 1.0
            signal[i] = naive_value
            if natural_wrapped:
                # Natural wrap step: saw drops by 2 at the wrap
                apply_polyblep_step_correction(
                    signal,
                    i - 1,
                    natural_frac,
                    -2.0,
                )

        prev_phase = phase

    return signal


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
    voice_state: dict[str, Any] | None = None,
) -> np.ndarray:
    """Render a bandlimited oscillator with a driven ZDF/TPT filter sweep.

    The ``quality`` param selects the ladder solver + oversampling factor
    applied to the filter + feedback + dither block (oscillators are
    generated at the base rate since BLEP handles their aliasing):

    - ``draft`` — ADAA ladder, no oversampling
    - ``fast`` — Newton ladder (2 iters), 2x oversampling
    - ``great`` — Newton ladder (4 iters), 2x oversampling (default)
    - ``divine`` — Newton ladder (8 iters), 4x oversampling

    The ``transient_mode`` param controls what oscillator state is carried
    across notes within the same voice.  Supported modes: ``analog``
    (default, carry phase + DC), ``dc_reset`` (carry phase, reset DC), and
    ``osc_reset`` (reset phase + DC, approximating the pre-carry per-note
    random-phase behavior).

    ``voice_state`` is an optional per-voice carry-over state dict; ``None``
    disables carry-over.
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
    osc2_sync_enabled = bool(params.get("osc2_sync", False))
    osc2_ring_mod = float(params.get("osc2_ring_mod", 0.0))
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
    k35_feedback_asymmetry = float(params.get("k35_feedback_asymmetry", 0.0))
    # Per-note voice-distortion slot (RePro-5-style pre-sum saturation).
    # When mode="off" or drive<=0 the helper short-circuits to a passthrough,
    # so defaults are a bit-identical no-op for existing pieces.
    voice_dist_mode = str(params.get("voice_dist_mode", "off")).lower()
    voice_dist_drive = float(params.get("voice_dist_drive", 0.5))
    voice_dist_mix = float(params.get("voice_dist_mix", 1.0))
    voice_dist_tone = float(params.get("voice_dist_tone", 0.0))
    osc_phase_noise = float(params.get("osc_phase_noise", 0.0))
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
    transient_config = resolve_transient_mode(str(analog["transient_mode"]))

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
            f"Unsupported filter_topology: {filter_topology!r}. "
            f"Supported: {sorted(_SUPPORTED_FILTER_TOPOLOGIES)}"
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
    fresh_phase = float(jittered.get("_phase_offset", 0.0))
    amp_jitter_db = float(jittered.get("_amp_jitter_db", 0.0))

    # Dedicated DC-sign RNG stream: keeps main rng untouched for
    # transient-mode carry.  Only draw when DC offset is active so pieces
    # that never touch ``osc_dc_offset`` remain bit-identical with prior
    # main-rng interleaving.
    if osc_dc_offset > 0.0:
        dc_rng = rng_for_note(
            freq=freq,
            duration=duration,
            amp=amp,
            sample_rate=sample_rate,
            extra_seed="polyblep_dc_signs",
        )
        fresh_dc_sign_osc1 = 1.0 if dc_rng.integers(2) == 0 else -1.0
        fresh_dc_sign_osc2 = 1.0 if dc_rng.integers(2) == 0 else -1.0
    else:
        fresh_dc_sign_osc1 = 1.0
        fresh_dc_sign_osc2 = 1.0

    (
        start_phase_osc1,
        start_phase_osc2,
        dc_sign_osc1,
        dc_sign_osc2,
    ) = apply_transient_state(
        voice_state,
        transient_config=transient_config,
        fresh_phase=fresh_phase,
        fresh_dc_signs=(fresh_dc_sign_osc1, fresh_dc_sign_osc2),
    )
    start_phase = start_phase_osc1

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

    # Resolve per-sample param_profiles.  Each destination falls back to
    # the scalar value when no profile is supplied, so existing pieces
    # stay bit-for-bit identical.  The validation mirrors the existing
    # cutoff_hz profile handling: shape must match n_samples.
    pulse_width_per_sample: np.ndarray | float = pulse_width
    if param_profiles is not None and "pulse_width" in param_profiles:
        pw_profile = np.asarray(param_profiles["pulse_width"], dtype=np.float64)
        if pw_profile.shape != (n_samples,):
            raise ValueError(
                "pulse_width profile length must match note duration in samples"
            )
        pulse_width_per_sample = np.clip(pw_profile, 0.01, 0.99)

    # Per-sample phase-noise for osc1.  Seeded via a dedicated
    # ``rng_for_note(extra_seed="osc1_phase_noise")`` so osc1 and osc2
    # draw from independent RNG streams — otherwise their zero-crossing
    # jitter would be correlated and the noise would read as coherent
    # pitch wobble instead of broadband texture.
    osc1_phase_noise_profile: np.ndarray | None = None
    if osc_phase_noise > 0.0:
        osc1_noise_rng = rng_for_note(
            freq=freq,
            duration=duration,
            amp=amp,
            sample_rate=sample_rate,
            extra_seed="osc1_phase_noise",
        )
        osc1_phase_noise_profile = osc1_noise_rng.uniform(
            -1.0, 1.0, size=n_samples
        ).astype(np.float64) * (osc_phase_noise * _PHASE_NOISE_SCALE)

    raw_signal, osc1_phase = _render_oscillator_with_phase(
        waveform=waveform,
        pulse_width=pulse_width_per_sample,
        freq_profile=freq_profile,
        sample_rate=sample_rate,
        start_phase=start_phase,
        phase_noise=osc1_phase_noise_profile,
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
        raw_signal = raw_signal + osc_dc_offset * 0.05 * dc_sign_osc1

    # osc2 is rendered whenever it is audible in the mix OR feeding the ring
    # modulator.  Hard sync still requires osc2 even if ring_mod is the only
    # non-zero path.
    needs_osc2 = osc2_level > 0.0 or osc2_ring_mod > 0.0
    if osc2_sync_enabled and osc2_waveform != "saw":
        raise ValueError(
            "osc2_sync=True is only supported when osc2_waveform='saw'. "
            f"Got osc2_waveform={osc2_waveform!r}."
        )
    if not 0.0 <= osc2_ring_mod <= 1.0:
        raise ValueError("osc2_ring_mod must lie in [0.0, 1.0]")

    if needs_osc2:
        # Per-sample osc2 detune / freq ratio support.  When a profile is
        # supplied for either destination, osc2's effective frequency is
        # computed per-sample; the scalar path is preserved for the
        # common case so existing pieces stay bit-for-bit unchanged.
        osc2_detune_profile: np.ndarray | None = None
        if param_profiles is not None and "osc2_detune_cents" in param_profiles:
            detune_arr = np.asarray(
                param_profiles["osc2_detune_cents"], dtype=np.float64
            )
            if detune_arr.shape != (n_samples,):
                raise ValueError(
                    "osc2_detune_cents profile length must match note duration"
                )
            osc2_detune_profile = detune_arr
        osc2_ratio_profile: np.ndarray | None = None
        if param_profiles is not None and "osc2_freq_ratio" in param_profiles:
            ratio_arr = np.asarray(param_profiles["osc2_freq_ratio"], dtype=np.float64)
            if ratio_arr.shape != (n_samples,):
                raise ValueError(
                    "osc2_freq_ratio profile length must match note duration"
                )
            osc2_ratio_profile = ratio_arr

        effective_detune_scalar = osc2_detune_cents * pow(0.5, osc2_spread_power - 1.0)
        if osc2_detune_profile is not None or osc2_ratio_profile is not None:
            # Spread power scales the detune similarly to the scalar path.
            if osc2_detune_profile is not None:
                effective_detune_per_sample = osc2_detune_profile * pow(
                    0.5, osc2_spread_power - 1.0
                )
            else:
                effective_detune_per_sample = np.full(
                    n_samples, effective_detune_scalar, dtype=np.float64
                )
            base_ratio = float(2.0 ** (osc2_semitones / 12.0))
            detune_factor = np.power(2.0, effective_detune_per_sample / 1200.0)
            if osc2_ratio_profile is not None:
                osc2_ratio_per_sample = osc2_ratio_profile * base_ratio * detune_factor
            else:
                osc2_ratio_per_sample = base_ratio * detune_factor
            osc2_freq_profile = freq_profile * osc2_ratio_per_sample
            # Scalar ratio for downstream scalar uses (osc_softness freq
            # estimate etc.).  Use the mean as a representative value.
            osc2_ratio = float(np.mean(osc2_ratio_per_sample))
        else:
            osc2_ratio = float(2.0 ** (osc2_semitones / 12.0)) * float(
                2.0 ** (effective_detune_scalar / 1200.0)
            )
            osc2_freq_profile = freq_profile * osc2_ratio
        # When voice_state carries a prior osc2 phase (analog / dc_reset on a
        # non-first note), resume from it directly so hard-sync event
        # alignment and dual-osc beating stay continuous.  Otherwise fall
        # back to the historical ``start_phase * 1.618`` decorrelation so
        # fresh notes keep their existing phase relationship.
        if (
            voice_state is not None
            and not transient_config.reset_phase
            and "phase_osc2" in voice_state
        ):
            osc2_start_phase = start_phase_osc2
        else:
            osc2_start_phase = start_phase * 1.618

        # Per-sample phase-noise for osc2.  Independent RNG stream from
        # osc1 — see the osc1 build above.
        osc2_phase_noise_profile: np.ndarray | None = None
        if osc_phase_noise > 0.0:
            osc2_noise_rng = rng_for_note(
                freq=freq,
                duration=duration,
                amp=amp,
                sample_rate=sample_rate,
                extra_seed="osc2_phase_noise",
            )
            osc2_phase_noise_profile = osc2_noise_rng.uniform(
                -1.0, 1.0, size=n_samples
            ).astype(np.float64) * (osc_phase_noise * _PHASE_NOISE_SCALE)

        if osc2_sync_enabled:
            # osc1 sync events: integer sample indices where osc1's cumulative
            # phase crosses an integer boundary, with fractional offset.  Osc1
            # phase is already exposed by ``_render_oscillator_with_phase`` in
            # wrapped (0..1) form; recompute cumulative phase to detect wraps.
            osc1_phase_inc = freq_profile / sample_rate
            osc1_cumphase = np.cumsum(osc1_phase_inc) + start_phase / (2.0 * math.pi)
            sync_samples, sync_fractions = _find_sync_events(
                osc1_cumphase, osc1_phase_inc
            )
            osc2_signal = _render_synced_saw_numba(
                osc2_freq_profile,
                sample_rate,
                osc2_start_phase,
                sync_samples,
                sync_fractions,
            )
            # reconstruct wrapped phase for post-processors that expect it
            osc2_cumphase = np.cumsum(osc2_freq_profile / sample_rate) + (
                osc2_start_phase / (2.0 * math.pi)
            )
            osc2_phase = osc2_cumphase % 1.0
        else:
            osc2_signal, osc2_phase = _render_oscillator_with_phase(
                waveform=osc2_waveform,
                pulse_width=osc2_pulse_width,
                freq_profile=osc2_freq_profile,
                sample_rate=sample_rate,
                start_phase=osc2_start_phase,
                phase_noise=osc2_phase_noise_profile,
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
            osc2_signal = osc2_signal + osc_dc_offset * 0.05 * dc_sign_osc2

        # Dry/ring blend.  When ring_mod == 0 the dry path is byte-identical
        # to the pre-ring-mod code: ``(osc1 + osc2_level*osc2) / (1+osc2_level)``.
        dry_denom = 1.0 + osc2_level
        if dry_denom > 0.0:
            dry_mix = (raw_signal + osc2_level * osc2_signal) / dry_denom
        else:
            dry_mix = raw_signal
        if osc2_ring_mod > 0.0:
            ring_signal = raw_signal * osc2_signal
            raw_signal = (1.0 - osc2_ring_mod) * dry_mix + osc2_ring_mod * ring_signal
        else:
            raw_signal = dry_mix

    # Cutoff envelope (identical pattern to filtered_stack / va).
    # When a per-sample ``cutoff_hz`` profile is supplied via
    # ``param_profiles`` it rides cutoff at audio rate; scalar paths
    # below (keytrack, envelope, drift, CV dither) stack on top of it.
    # We scale the profile by the ratio of the post-voice-card /
    # post-jitter scalar to the authored ``params["cutoff_hz"]`` so the
    # profile's matrix contribution is preserved *relative to the
    # author's base* while the analog-character offsets (filter_spread,
    # cutoff jitter) still apply — exactly as they would on the scalar
    # path.  See docs/score_api.md.
    nyquist = sample_rate / 2.0
    if param_profiles is not None and "cutoff_hz" in param_profiles:
        cutoff_profile_base = np.asarray(param_profiles["cutoff_hz"], dtype=np.float64)
        if cutoff_profile_base.shape != (n_samples,):
            raise ValueError(
                "cutoff_hz profile length must match note duration in samples"
            )
        authored_cutoff = float(params.get("cutoff_hz", 3000.0))
        analog_scale = cutoff_hz / authored_cutoff if authored_cutoff > 0.0 else 1.0
        cutoff_profile_base = cutoff_profile_base * analog_scale
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
        k35_feedback_asymmetry=k35_feedback_asymmetry,
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

    note_buffer = amp * filtered

    # Snapshot end-of-note oscillator state.  ``osc1_phase`` / ``osc2_phase``
    # are in wrapped [0, 1) cycles; convert to radians (``start_phase`` +
    # ``2π * wrapped_cycle``) so the next note's resume phase matches the
    # radian convention ``_render_oscillator_with_phase`` expects.
    if osc1_phase.size > 0:
        final_phase_osc1 = float(start_phase + 2.0 * math.pi * osc1_phase[-1])
    else:
        final_phase_osc1 = float(start_phase)
    if needs_osc2 and osc2_phase.size > 0:
        final_phase_osc2 = float(osc2_start_phase + 2.0 * math.pi * osc2_phase[-1])
    else:
        final_phase_osc2 = final_phase_osc1
    snapshot_voice_state(
        voice_state,
        final_phase_osc1=final_phase_osc1,
        final_phase_osc2=final_phase_osc2,
        dc_sign_osc1=dc_sign_osc1,
        dc_sign_osc2=dc_sign_osc2,
    )

    return apply_voice_dist(
        note_buffer,
        mode=voice_dist_mode,
        drive=voice_dist_drive,
        mix=voice_dist_mix,
        tone=voice_dist_tone,
        sample_rate=sample_rate,
    )


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
