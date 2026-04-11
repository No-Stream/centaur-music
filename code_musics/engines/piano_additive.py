"""Additive piano synthesis engine with physical modeling envelopes."""

from __future__ import annotations

from typing import Any

import numpy as np

from code_musics.engines._dsp_utils import (
    GOLDEN_RATIO_FRAC,
    apply_body_saturation,
    apply_soundboard,
    bandpass_noise,
    build_drift,
    nyquist_fade,
    render_damper_thump,
    render_noise_floor,
    rng_for_note,
)

_DEFAULT_N_PARTIALS = 32
_DEFAULT_INHARMONICITY = 0.0005
_DEFAULT_DECAY_BASE = 3.5
_DEFAULT_DECAY_PARTIAL_TILT = 0.5
_DEFAULT_DECAY_PROMPT = 0.5
_DEFAULT_BRIGHTNESS = 0.5
_DEFAULT_HAMMER_HARDNESS = 0.5
_DEFAULT_HAMMER_NOISE = 0.2
_DEFAULT_HAMMER_VELOCITY_TILT = 0.5
_DEFAULT_HAMMER_POSITION = 0.12
_DEFAULT_BODY_SATURATION = 0.15
_DEFAULT_DRIFT = 0.08
_DEFAULT_DRIFT_RATE_HZ = 0.05
_DEFAULT_UNISON_DETUNE = 3.0
_DEFAULT_UNISON_DRIFT = 0.15
_DEFAULT_SOUNDBOARD_COLOR = 0.4
_DEFAULT_SOUNDBOARD_BRIGHTNESS = 0.5
_DEFAULT_DAMPER_NOISE = 0.08


def render(
    *,
    freq: float,
    duration: float,
    amp: float,
    sample_rate: int,
    params: dict[str, Any],
    freq_trajectory: np.ndarray | None = None,
) -> np.ndarray:
    """Render a piano voice via additive synthesis with physical modeling envelopes."""
    if freq <= 0:
        raise ValueError("freq must be positive")
    if duration < 0:
        raise ValueError("duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    n_partials = int(params.get("n_partials", _DEFAULT_N_PARTIALS))
    inharmonicity = float(params.get("inharmonicity", _DEFAULT_INHARMONICITY))
    decay_base = float(params.get("decay_base", _DEFAULT_DECAY_BASE))
    decay_partial_tilt = float(
        params.get("decay_partial_tilt", _DEFAULT_DECAY_PARTIAL_TILT)
    )
    decay_prompt_raw = params.get("decay_prompt")
    decay_two_stage_raw = params.get("decay_two_stage")
    if decay_prompt_raw is not None:
        decay_prompt = float(decay_prompt_raw)
    elif decay_two_stage_raw is not None:
        decay_prompt = float(decay_two_stage_raw) * 2.0
    else:
        decay_prompt = _DEFAULT_DECAY_PROMPT
    brightness = float(params.get("brightness", _DEFAULT_BRIGHTNESS))
    hammer_hardness = float(params.get("hammer_hardness", _DEFAULT_HAMMER_HARDNESS))
    hammer_noise = float(params.get("hammer_noise", _DEFAULT_HAMMER_NOISE))
    hammer_velocity_tilt = float(
        params.get("hammer_velocity_tilt", _DEFAULT_HAMMER_VELOCITY_TILT)
    )
    hammer_position = float(params.get("hammer_position", _DEFAULT_HAMMER_POSITION))
    body_saturation = float(params.get("body_saturation", _DEFAULT_BODY_SATURATION))
    drift = float(params.get("drift", _DEFAULT_DRIFT))
    drift_rate_hz = float(params.get("drift_rate_hz", _DEFAULT_DRIFT_RATE_HZ))
    unison_count_raw = params.get("unison_count")
    unison_detune = float(params.get("unison_detune", _DEFAULT_UNISON_DETUNE))
    unison_drift = float(params.get("unison_drift", _DEFAULT_UNISON_DRIFT))
    soundboard_color = float(params.get("soundboard_color", _DEFAULT_SOUNDBOARD_COLOR))
    soundboard_brightness = float(
        params.get("soundboard_brightness", _DEFAULT_SOUNDBOARD_BRIGHTNESS)
    )
    damper_noise = float(params.get("damper_noise", _DEFAULT_DAMPER_NOISE))
    partial_ratios_raw = params.get("partial_ratios")

    if inharmonicity < 0:
        raise ValueError("inharmonicity must be non-negative")

    n_samples = int(sample_rate * duration)
    if n_samples == 0:
        return np.zeros(0)

    if freq_trajectory is not None:
        freq_trajectory = np.asarray(freq_trajectory, dtype=np.float64)
        if freq_trajectory.ndim != 1:
            raise ValueError("freq_trajectory must be one-dimensional")
        if freq_trajectory.size != n_samples:
            raise ValueError("freq_trajectory length must match note duration")

    if unison_count_raw is not None:
        unison_count = int(unison_count_raw)
    elif freq < 200.0:
        unison_count = 1
    elif freq < 400.0:
        unison_count = 2
    else:
        unison_count = 3

    partial_freqs, partial_amps = _compute_partial_freqs(
        freq=freq,
        n_partials=n_partials,
        inharmonicity=inharmonicity,
        partial_ratios=partial_ratios_raw,
    )

    rng = rng_for_note(freq=freq, duration=duration, amp=amp, sample_rate=sample_rate)

    effective_hardness = hammer_hardness + hammer_velocity_tilt * (amp - 0.5)
    effective_hardness = float(np.clip(effective_hardness, 0.05, 1.0))

    string_signal = np.zeros(n_samples, dtype=np.float64)
    for string_idx in range(unison_count):
        phase_offset = string_idx * GOLDEN_RATIO_FRAC * 2.0 * np.pi
        drift_trajectory = build_drift(
            n_samples=n_samples,
            drift_amount=drift + unison_drift * (1 if string_idx > 0 else 0),
            drift_rate_hz=drift_rate_hz * (1.0 + 0.15 * string_idx),
            duration=duration,
            phase_offset=phase_offset,
        )

        detune_cents = 0.0
        if unison_count > 1:
            spread = unison_detune
            detune_cents = spread * ((string_idx / (unison_count - 1)) * 2.0 - 1.0)
            if unison_count == 2:
                detune_cents = spread * (1.0 if string_idx == 1 else -1.0)
        detune_ratio = 2.0 ** (detune_cents / 1200.0)

        string_signal += _render_string(
            partial_freqs=partial_freqs * detune_ratio,
            partial_amps=partial_amps,
            brightness=brightness,
            hammer_hardness=effective_hardness,
            hammer_position=hammer_position,
            decay_base=decay_base,
            decay_partial_tilt=decay_partial_tilt,
            decay_prompt=decay_prompt,
            drift_trajectory=drift_trajectory,
            n_samples=n_samples,
            sample_rate=sample_rate,
            freq=freq,
            freq_trajectory=freq_trajectory,
            rng=rng,
            string_idx=string_idx,
        )

    # Hammer noise thump at onset.
    if hammer_noise > 0:
        hammer_signal = _render_onset_noise(
            freq=freq,
            effective_hardness=effective_hardness,
            sample_rate=sample_rate,
            n_samples=n_samples,
            rng=rng,
        )
        string_peak = np.max(np.abs(string_signal))
        hammer_peak = np.max(np.abs(hammer_signal))
        if hammer_peak > 0 and string_peak > 0:
            hammer_signal *= string_peak / hammer_peak
        string_signal += hammer_noise * 0.35 * hammer_signal

    if body_saturation > 0:
        string_signal = apply_body_saturation(
            string_signal,
            body_saturation,
            cubic_amount=0.15,
            even_amount=0.35,
            log_thd=True,
        )

    string_signal += render_noise_floor(
        signal=string_signal,
        sample_rate=sample_rate,
        n_samples=n_samples,
        rng=rng,
    )

    mixed = apply_soundboard(
        signal=string_signal,
        soundboard_color=soundboard_color,
        soundboard_brightness=soundboard_brightness,
        sample_rate=sample_rate,
    )

    if damper_noise > 0:
        mixed += render_damper_thump(
            freq=freq,
            damper_noise=damper_noise,
            n_samples=n_samples,
            sample_rate=sample_rate,
            rng=rng,
        )

    peak = np.max(np.abs(mixed))
    if peak <= 0.0:
        raise ValueError("piano render produced no audible output")
    return amp * (mixed / peak)


def _compute_partial_freqs(
    *,
    freq: float,
    n_partials: int,
    inharmonicity: float,
    partial_ratios: list[dict[str, float]] | list[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute partial frequencies and amplitudes.

    Default amplitudes follow a 1/n rolloff (struck string).  Additional
    spectral shaping (hammer lowpass, hammer position comb, brightness)
    is applied per-partial in _render_string.
    """
    if partial_ratios is not None:
        ratios: list[float] = []
        amps: list[float] = []
        for entry in partial_ratios:
            if isinstance(entry, dict):
                ratios.append(float(entry["ratio"]))
                amps.append(float(entry.get("amp", 1.0)))
            else:
                ratios.append(float(entry))
                amps.append(1.0)
        return (
            np.array(ratios, dtype=np.float64) * freq,
            np.array(amps, dtype=np.float64),
        )

    ns = np.arange(1, n_partials + 1, dtype=np.float64)
    freqs = ns * freq * np.sqrt(1.0 + inharmonicity * ns**2)
    amplitudes = 1.0 / ns
    return freqs, amplitudes


def _render_onset_noise(
    *,
    freq: float,
    effective_hardness: float,
    sample_rate: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render two-component onset noise: felt thump + string chatter."""
    result = np.zeros(n_samples, dtype=np.float64)

    thump_duration = 0.015 + 0.015 * (1.0 - effective_hardness)
    thump_samples = min(max(1, int(thump_duration * sample_rate)), n_samples)
    thump_noise = rng.standard_normal(n_samples)
    thump_env = np.zeros(n_samples, dtype=np.float64)
    thump_t = np.arange(thump_samples, dtype=np.float64) / sample_rate
    thump_env[:thump_samples] = np.sin(np.pi * thump_t / thump_duration)
    thump_center = float(np.clip(100.0 + freq * 0.5, 100.0, 3000.0))
    result += thump_env * bandpass_noise(
        thump_noise, sample_rate=sample_rate, center_hz=thump_center, width_ratio=1.5
    )

    chatter_duration = 0.005 + 0.005 * effective_hardness
    chatter_samples = min(max(1, int(chatter_duration * sample_rate)), n_samples)
    chatter_noise = rng.standard_normal(n_samples)
    chatter_env = np.zeros(n_samples, dtype=np.float64)
    chatter_t = np.arange(chatter_samples, dtype=np.float64)
    chatter_env[:chatter_samples] = np.exp(-chatter_t / max(1.0, chatter_samples * 0.3))
    chatter_center = float(np.clip(500.0 + freq * 2.0, 500.0, 8000.0))
    result += (
        effective_hardness
        * chatter_env
        * bandpass_noise(
            chatter_noise,
            sample_rate=sample_rate,
            center_hz=chatter_center,
            width_ratio=1.0,
        )
    )

    return result


def _build_attack_envelope_batch(
    *,
    partial_indices: np.ndarray,
    n_partials: int,
    hammer_hardness: float,
    freq: float,
    sample_rate: int,
    n_samples: int,
) -> np.ndarray:
    """Vectorized attack envelopes for multiple partials: [n_active, n_samples].

    Equivalent to calling _build_attack_envelope for each partial index, but
    computes all partials simultaneously using 2D broadcasting.
    """
    n_active = partial_indices.size
    register_stretch = float(np.clip(1.0 - (freq - 100.0) / 4000.0, 0.5, 1.2))
    contact_time = (0.002 + 0.006 * (1.0 - hammer_hardness)) * register_stretch
    contact_samples = max(2, int(contact_time * sample_rate))

    partial_norms = partial_indices / max(1, n_partials)  # [n_active]

    t = np.arange(n_samples, dtype=np.float64) / sample_rate  # [n_samples]

    # --- Contact/rise phase: affects samples [0:contact_samples] ---
    envelope = np.ones((n_active, n_samples), dtype=np.float64)

    t_contact = t[:contact_samples]  # [contact_samples]
    onset_fractions = np.maximum(0.3, 1.0 - 0.7 * partial_norms)  # [n_active]
    delayed_starts = contact_time * (1.0 - onset_fractions)  # [n_active]
    rise_durations = contact_time * onset_fractions  # [n_active]

    # [n_active, contact_samples]
    rise_t = np.clip(
        t_contact[np.newaxis, :] - delayed_starts[:, np.newaxis],
        0.0,
        None,
    )
    rise_phase = np.clip(
        rise_t / np.maximum(1e-6, rise_durations[:, np.newaxis]),
        0.0,
        1.0,
    )
    envelope[:, :contact_samples] = np.sin(np.pi * rise_phase / 2.0)

    # --- Post-contact bounce ---
    bounce_depths = (0.06 + 0.10 * hammer_hardness) * (
        0.3 + 0.7 * partial_norms
    )  # [n_active]
    bounce_tau = max(1e-6, contact_time * 1.5)
    post_contact_t = np.clip(t - contact_time, 0.0, None)  # [n_samples]
    # [n_active, n_samples]
    bounce = bounce_depths[:, np.newaxis] * np.exp(
        -post_contact_t[np.newaxis, :] / bounce_tau
    )
    envelope *= 1.0 - bounce

    # --- Post-contact decay for higher partials (partial_norm > 0.4) ---
    # Compute for all partials; strength is zero when partial_norm <= 0.4.
    post_decay_strength = np.clip((partial_norms - 0.4) / 0.6, 0.0, None)  # [n_active]
    post_decay_tau = np.maximum(
        1e-6,
        contact_time * (2.0 + 3.0 * (1.0 - partial_norms)),
    )  # [n_active]
    # [n_active, n_samples]
    post_decay = (
        post_decay_strength[:, np.newaxis]
        * 0.3
        * np.exp(-post_contact_t[np.newaxis, :] / post_decay_tau[:, np.newaxis])
    )
    envelope *= 1.0 - post_decay

    # --- Per-partial peak normalization ---
    norm_end = min(contact_samples * 5, n_samples)
    env_peaks = np.max(envelope[:, :norm_end], axis=1)  # [n_active]
    # Avoid division by zero; leave rows with zero peak unchanged (already ones).
    nonzero_mask = env_peaks > 0
    envelope[nonzero_mask] /= env_peaks[nonzero_mask, np.newaxis]

    return envelope


def _build_partial_decay_batch(
    *,
    partial_indices: np.ndarray,
    n_partials: int,
    decay_base: float,
    decay_partial_tilt: float,
    decay_prompt: float,
    freq: float,
    sample_rate: int,
    n_samples: int,
    jitter_raw: np.ndarray,
    tail_ratio_raw: np.ndarray,
    w_tail_rand: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Vectorized triple-phase decay for multiple partials: [n_active, n_samples].

    Random values are pre-drawn by the caller in the same per-partial order as the
    original scalar loop to preserve RNG determinism.
    """
    jitter = np.clip(1.0 + 0.12 * jitter_raw, 0.7, 1.3)  # [n_active]

    partial_norms = partial_indices / max(1, n_partials)  # [n_active]
    tilt_strength = decay_partial_tilt * 130.0
    tilt_factor = 1.0 / (1.0 + tilt_strength * partial_norms**2)  # [n_active]

    tau_sustain = np.maximum(0.01, jitter * decay_base * tilt_factor)  # [n_active]

    register_scale = float(np.clip(1.0 - (freq - 100.0) / 4000.0, 0.3, 1.0))
    tau_prompt = np.maximum(
        0.01,
        (0.03 + 0.12 * register_scale) * (1.0 - 0.4 * partial_norms),
    )  # [n_active]

    tail_ratio = 6.0 + 4.0 * tail_ratio_raw  # [n_active]
    tau_tail = tau_sustain * tail_ratio  # [n_active]

    w_prompt = (0.05 + 0.65 * partial_norms) * decay_prompt  # [n_active]
    w_tail = 0.08 + 0.04 * w_tail_rand  # [n_active]
    w_sustain = np.maximum(0.0, 1.0 - w_prompt - w_tail)  # [n_active]

    # Normalize weights so they sum to 1.0.
    total = w_prompt + w_sustain + w_tail  # [n_active]
    nonzero = total > 0
    w_prompt = np.where(nonzero, w_prompt / np.maximum(total, 1e-30), w_prompt)
    w_sustain = np.where(nonzero, w_sustain / np.maximum(total, 1e-30), w_sustain)
    w_tail = np.where(nonzero, w_tail / np.maximum(total, 1e-30), w_tail)

    # Build 2D envelope: [n_active, n_samples].
    # Each row = w_prompt*exp(-t/tau_prompt) + w_sustain*exp(-t/tau_sustain) + w_tail*exp(-t/tau_tail)
    t_row = t[np.newaxis, :]  # [1, n_samples]
    return (
        w_prompt[:, np.newaxis] * np.exp(-t_row / tau_prompt[:, np.newaxis])
        + w_sustain[:, np.newaxis] * np.exp(-t_row / tau_sustain[:, np.newaxis])
        + w_tail[:, np.newaxis] * np.exp(-t_row / tau_tail[:, np.newaxis])
    )


def _render_string(
    *,
    partial_freqs: np.ndarray,
    partial_amps: np.ndarray,
    brightness: float,
    hammer_hardness: float,
    hammer_position: float,
    decay_base: float,
    decay_partial_tilt: float,
    decay_prompt: float,
    drift_trajectory: np.ndarray,
    n_samples: int,
    sample_rate: int,
    freq: float,
    freq_trajectory: np.ndarray | None,
    rng: np.random.Generator,
    string_idx: int,
) -> np.ndarray:
    """Render one string's worth of partials with piano-like envelopes.

    Uses 2D [n_active, n_samples] arrays to compute all partials simultaneously,
    avoiding per-partial Python loops and temporary allocations.
    """
    nyquist_hz = sample_rate / 2.0
    n_partials = partial_freqs.size
    t = np.arange(n_samples, dtype=np.float64) / sample_rate

    freq_traj_scale = np.ones(n_samples, dtype=np.float64)
    if freq_trajectory is not None:
        freq_traj_scale = freq_trajectory / freq

    # Brief onset pitch transient (longitudinal precursor).
    pitch_onset = np.ones(n_samples, dtype=np.float64)
    if string_idx == 0:
        onset_cents = 6.0 * hammer_hardness
        onset_time = 0.004
        onset_samples = min(int(onset_time * sample_rate), n_samples)
        if onset_samples > 0:
            onset_t = np.arange(onset_samples, dtype=np.float64) / sample_rate
            pitch_onset[:onset_samples] = np.power(
                2.0,
                onset_cents / 1200.0 * np.exp(-onset_t / (onset_time * 0.3)),
            )

    # Combined modulation trajectory shared by all partials: [n_samples].
    combined_mod = drift_trajectory * freq_traj_scale * pitch_onset

    brightness_weights = _compute_brightness_weights(n_partials, brightness)

    # Hammer lowpass cutoff.
    hammer_cutoff_hz = freq * (3.0 + 14.0 * hammer_hardness)
    hammer_order = 2.5 + hammer_hardness * 1.5

    spectral_fade_time = 0.02 + 0.08 * (1.0 - hammer_hardness)

    # Draw initial phases for ALL partials (matches original RNG order).
    initial_phases = rng.uniform(0.0, 2.0 * np.pi, size=n_partials)

    # --- Pre-compute per-partial scalars to determine which partials are active ---
    partial_numbers = np.arange(1, n_partials + 1, dtype=np.float64)

    # Hammer lowpass: 1D array [n_partials].
    hammer_atten = 1.0 / (1.0 + (partial_freqs / hammer_cutoff_hz) ** hammer_order)

    # Hammer position comb: |sin(pi * k * position)|, floored at 0.08.
    comb = np.abs(np.sin(np.pi * partial_numbers * hammer_position))
    comb = np.maximum(comb, 0.08)

    onset_weights = hammer_atten * comb
    target_weights = brightness_weights

    # First skip condition: amplitude too low (no RNG consumed for skipped partials).
    max_weights = np.maximum(onset_weights, target_weights)
    amp_ok = (partial_amps * max_weights) > 1e-8

    # Second skip condition: above Nyquist. Compute instantaneous_freq for all
    # partials as [n_partials, n_samples] and check nyquist_fade max per row.
    # To avoid allocating the full 2D array just for the check when many partials
    # are already eliminated by amp_ok, only check the amp_ok survivors.
    amp_ok_indices = np.where(amp_ok)[0]
    if amp_ok_indices.size == 0:
        return np.zeros(n_samples, dtype=np.float64)

    # Instantaneous freq for amp_ok partials: [n_amp_ok, n_samples].
    inst_freq_candidates = (
        partial_freqs[amp_ok_indices, np.newaxis] * combined_mod[np.newaxis, :]
    )

    # nyquist_fade works element-wise, so 2D input gives 2D output.
    nq_fade_candidates = nyquist_fade(inst_freq_candidates, nyquist_hz)
    nq_max = np.max(nq_fade_candidates, axis=1)
    nq_ok_local = nq_max > 0.0

    # Active indices: partials that survive both checks (in original index space).
    active_indices = amp_ok_indices[nq_ok_local]
    n_active = active_indices.size
    if n_active == 0:
        return np.zeros(n_samples, dtype=np.float64)

    # Retain only the active rows from the Nyquist fade array.
    nq_fade_2d = nq_fade_candidates[nq_ok_local]

    # --- Draw RNG values for active partials in original per-partial order ---
    # Original loop order per active partial: _build_partial_decay draws
    # [standard_normal(), uniform(), uniform()], then beat params draw
    # [uniform(0.3,2.5), uniform(0.12,0.35), uniform(0,2pi)].
    # We draw them in the same sequential order to preserve determinism.
    decay_jitter_raw = np.empty(n_active, dtype=np.float64)
    decay_tail_ratio = np.empty(n_active, dtype=np.float64)
    decay_w_tail_rand = np.empty(n_active, dtype=np.float64)
    beat_rates = np.empty(n_active, dtype=np.float64)
    beat_depths = np.empty(n_active, dtype=np.float64)
    beat_phases = np.empty(n_active, dtype=np.float64)
    for i in range(n_active):
        decay_jitter_raw[i] = rng.standard_normal()
        decay_tail_ratio[i] = rng.uniform()
        decay_w_tail_rand[i] = rng.uniform()
        beat_rates[i] = rng.uniform(0.3, 2.5)
        beat_depths[i] = rng.uniform(0.12, 0.35)
        beat_phases[i] = rng.uniform(0.0, 2.0 * np.pi)

    # --- Vectorized spectral envelope: [n_active, n_samples] ---
    active_onset_w = onset_weights[active_indices]
    active_target_w = target_weights[active_indices]
    spectral_fade = np.exp(-t / max(1e-6, spectral_fade_time))  # [n_samples]
    spectral_env_2d = (
        active_target_w[:, np.newaxis]
        + (active_onset_w - active_target_w)[:, np.newaxis]
        * spectral_fade[np.newaxis, :]
    )
    partial_amp_2d = partial_amps[active_indices, np.newaxis] * spectral_env_2d

    # --- Vectorized attack envelopes: [n_active, n_samples] ---
    attack_env_2d = _build_attack_envelope_batch(
        partial_indices=active_indices,
        n_partials=n_partials,
        hammer_hardness=hammer_hardness,
        freq=freq,
        sample_rate=sample_rate,
        n_samples=n_samples,
    )

    # --- Vectorized decay envelopes: [n_active, n_samples] ---
    decay_env_2d = _build_partial_decay_batch(
        partial_indices=active_indices,
        n_partials=n_partials,
        decay_base=decay_base,
        decay_partial_tilt=decay_partial_tilt,
        decay_prompt=decay_prompt,
        freq=freq,
        sample_rate=sample_rate,
        n_samples=n_samples,
        jitter_raw=decay_jitter_raw,
        tail_ratio_raw=decay_tail_ratio,
        w_tail_rand=decay_w_tail_rand,
        t=t,
    )

    # --- Vectorized AM beat modulation: [n_active, n_samples] ---
    am_2d = 1.0 - beat_depths[:, np.newaxis] * np.sin(
        2.0 * np.pi * beat_rates[:, np.newaxis] * t[np.newaxis, :]
        + beat_phases[:, np.newaxis]
    )

    # --- Vectorized phase accumulation: [n_active, n_samples] ---
    # inst_freq for active partials (reuse from Nyquist check where possible).
    # nq_fade_2d already corresponds to active partials.
    # Reconstruct inst_freq for active partials.
    inst_freq_2d = (
        partial_freqs[active_indices, np.newaxis] * combined_mod[np.newaxis, :]
    )

    # Phase increment: 2*pi*freq/sr, cumsum along time axis.
    phase_inc = 2.0 * np.pi * inst_freq_2d / float(sample_rate)
    # Prepend zero column, drop last column, then cumsum (matches original concat+cumsum).
    phase_inc[:, 1:] = phase_inc[:, :-1]
    phase_inc[:, 0] = 0.0
    phase_2d = initial_phases[active_indices, np.newaxis] + np.cumsum(phase_inc, axis=1)

    # --- Combine and sum all partials ---
    signal_2d = (
        partial_amp_2d
        * nq_fade_2d
        * attack_env_2d
        * decay_env_2d
        * am_2d
        * np.sin(phase_2d)
    )

    return np.sum(signal_2d, axis=0)


def _compute_brightness_weights(n_partials: int, brightness: float) -> np.ndarray:
    """Spectral tilt on top of the natural 1/n rolloff from partial_amps."""
    ns = np.arange(1, n_partials + 1, dtype=np.float64)
    rolloff_rate = 1.0 + (1.0 - brightness) * 3.0
    return 1.0 / (1.0 + ((ns - 1) / max(1.0, n_partials * 0.3)) ** rolloff_rate)
