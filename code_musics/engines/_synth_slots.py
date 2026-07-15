"""Per-slot renderers for the synth_voice engine.

Four parallel source slots — ``osc``, ``partials``, ``fm``, ``noise`` — each
string-dispatched on ``{slot}_type``.  Each renderer returns a 1-D numpy array
of length ``n_samples`` peak-scaled to roughly <= 1.0; the orchestrator
handles level/envelope/sum/post-chain.

Slot implementations compose existing engine primitives (from ``_oscillators``,
``va``, ``fm``, ``_dsp_utils``) rather than reimplementing DSP.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from code_musics.engines._chaotic import SUPPORTED_SYSTEMS as _CHAOTIC_SYSTEMS
from code_musics.engines._chaotic import render_chaotic
from code_musics.engines._dsp_utils import bandpass_noise, flow_exciter, rain_exciter
from code_musics.engines._oscillators import render_polyblep_oscillator
from code_musics.engines._pluck import render_pluck
from code_musics.engines._scanned import render_scanned
from code_musics.engines.fm import _fm_sample_loop
from code_musics.engines.va import (
    _build_spectralwave_partials,
    _render_partial_bank,
    _render_supersaw_bank,
)
from code_musics.found_sound import FOUND_SOUND_TEXTURES, found_sound
from code_musics.humanize import stable_seed

_VALID_OSC_TYPES = frozenset(
    {"polyblep", "supersaw", "pulse", "pluck", "scanned", "chaotic"}
)
_VALID_PARTIALS_TYPES = frozenset({"additive", "spectralwave", "drawbars"})
_VALID_FM_TYPES = frozenset({"two_op"})
_VALID_NOISE_TYPES = frozenset({"white", "pink", "bandpass", "flow", "rain", "found"})

# Standard Hammond drawbar footages relative to 8' fundamental.
# Order: 16', 5-1/3', 8', 4', 2-2/3', 2', 1-3/5', 1-1/3', 1'
_HAMMOND_DRAWBAR_RATIOS: tuple[float, ...] = (
    0.5,
    1.5,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    8.0,
)
_HAMMOND_DRAWBAR_DEFAULT_AMPS: tuple[float, ...] = (
    0.8,
    0.0,
    1.0,
    0.8,
    0.0,
    0.0,
    0.0,
    0.0,
    0.6,
)


def _render_chaotic_slot(
    *,
    n_samples: int,
    freq_profile: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    params: dict[str, Any],
) -> np.ndarray:
    """Render a chaotic ODE integrator for the ``osc`` slot.

    Reads ``osc_chaos_system``, ``osc_chaos_rate_hz``, ``osc_chaos_amount``,
    ``osc_chaos_symmetry``, and ``osc_chaos_seed`` from ``params``.  The
    default rate tracks the voice's mean fundamental so the attractor's
    perceived pitch is coupled to note selection.
    """
    system = str(params.get("osc_chaos_system", "lorenz")).lower()
    if system not in _CHAOTIC_SYSTEMS:
        raise ValueError(
            f"osc_chaos_system must be one of {sorted(_CHAOTIC_SYSTEMS)}, "
            f"got {system!r}"
        )
    default_rate = float(np.mean(freq_profile))
    rate_hz = float(params.get("osc_chaos_rate_hz", default_rate))
    amount = float(params.get("osc_chaos_amount", 0.5))
    symmetry = float(params.get("osc_chaos_symmetry", 0.0))
    user_seed = params.get("osc_chaos_seed")
    if user_seed is None:
        chaos_seed = stable_seed(
            "chaotic_osc",
            int(rng.integers(0, 2**31 - 1)),
            system,
        )
    else:
        chaos_seed = int(user_seed)
    return render_chaotic(
        system=system,
        rate_hz=rate_hz,
        amount=amount,
        symmetry=symmetry,
        n_samples=n_samples,
        sample_rate=sample_rate,
        seed=chaos_seed,
    )


def render_osc(
    *,
    n_samples: int,
    freq_profile: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    osc_type: str,
    params: dict[str, Any],
) -> np.ndarray:
    """Render the ``osc`` slot: polyblep / supersaw / pulse / pluck / scanned / chaotic.

    Dispatches on ``osc_type`` and composes oscillator primitives from
    :mod:`code_musics.engines._oscillators`, :mod:`code_musics.engines.va`,
    :mod:`code_musics.engines._pluck`, :mod:`code_musics.engines._scanned`,
    or :mod:`code_musics.engines._chaotic`.
    """
    if osc_type not in _VALID_OSC_TYPES:
        raise ValueError(
            f"osc_type must be one of {sorted(_VALID_OSC_TYPES)} or None, "
            f"got {osc_type!r}"
        )
    if n_samples == 0:
        return np.zeros(0, dtype=np.float64)

    if osc_type == "polyblep":
        waveform = str(params.get("osc_wave", "saw")).lower()
        pulse_width = float(params.get("osc_pulse_width", 0.5))
        start_phase = float(params.get("osc_start_phase", 0.0))
        osc2_level = float(params.get("osc2_level", 0.0))

        if bool(params.get("osc_hard_sync", False)) and osc2_level > 0.0:
            raise NotImplementedError(
                "osc_hard_sync is deferred to a later iteration; use the "
                "polyblep engine directly for hard-sync voicings."
            )

        signal, _ = render_polyblep_oscillator(
            waveform=waveform,
            pulse_width=pulse_width,
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            start_phase=start_phase,
            phase_noise=None,
        )

        if osc2_level > 0.0:
            osc2_wave = str(params.get("osc2_wave", "saw")).lower()
            osc2_detune_cents = float(params.get("osc2_detune_cents", 0.0))
            osc2_ratio = float(params.get("osc2_freq_ratio", 1.0)) * float(
                2.0 ** (osc2_detune_cents / 1200.0)
            )
            osc2_signal, _ = render_polyblep_oscillator(
                waveform=osc2_wave,
                pulse_width=pulse_width,
                freq_profile=freq_profile * osc2_ratio,
                sample_rate=sample_rate,
                start_phase=start_phase * 1.618,
                phase_noise=None,
            )
            signal = (signal + osc2_level * osc2_signal) / (1.0 + osc2_level)

    elif osc_type == "supersaw":
        spread_cents = float(params.get("osc_spread_cents", 18.0))
        mix = float(params.get("osc_mix", 0.5))
        detune = max(0.0, min(1.0, spread_cents / 100.0))
        mix = max(0.0, min(1.0, mix))

        signal = _render_supersaw_bank(
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            detune=detune,
            mix=mix,
            sync=False,
            rng=rng,
            spread_profile=None,
            phase_noise_amount=0.0,
            phase_noise_freq=0.0,
            phase_noise_duration=0.0,
            phase_noise_amp=0.0,
            phase_noise_voice_name="",
        )

    elif osc_type == "pulse":
        pulse_width = float(params.get("osc_pulse_width", 0.5))
        pulse_width = max(0.05, min(0.95, pulse_width))
        start_phase = float(params.get("osc_start_phase", 0.0))
        signal, _ = render_polyblep_oscillator(
            waveform="square",
            pulse_width=pulse_width,
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            start_phase=start_phase,
            phase_noise=None,
        )

    elif osc_type == "pluck":  # Karplus-Strong++
        # Keep duration for the API, but pass n_samples explicitly — the
        # n_samples -> duration -> int() round-trip can lose a sample.
        duration = n_samples / float(sample_rate)
        hardness = float(params.get("osc_pluck_hardness", 0.5))
        damping = float(params.get("osc_pluck_damping", 0.3))
        position = float(params.get("osc_pluck_position", 0.25))
        sustain = float(params.get("osc_pluck_sustain", 0.0))
        drive = float(params.get("osc_pluck_drive", 0.0))
        # Derive a deterministic seed from the rng so determinism plumbing
        # (voice_card / note seeding) still propagates into the pluck loop.
        pluck_seed = int(rng.integers(0, 2**31 - 1))
        base_freq = float(freq_profile[0])
        signal = render_pluck(
            freq=base_freq,
            duration=duration,
            sample_rate=sample_rate,
            hardness=hardness,
            damping=damping,
            position=position,
            sustain=sustain,
            drive=drive,
            seed=pluck_seed,
            freq_profile=freq_profile,
            n_samples=n_samples,
        )

    elif osc_type == "scanned":  # Verplank-style mass-spring ring scanner
        n_nodes = int(params.get("osc_scan_n_nodes", 16))
        motion = float(params.get("osc_scan_motion", 0.5))
        tension = float(params.get("osc_scan_tension", 0.5))
        damping = float(params.get("osc_scan_damping", 0.2))
        position = float(params.get("osc_scan_position", 0.5))
        scan_seed = int(rng.integers(0, 2**31 - 1))
        signal = render_scanned(
            freq=float(freq_profile[0]),
            n_samples=n_samples,
            sample_rate=sample_rate,
            n_nodes=n_nodes,
            motion=motion,
            tension=tension,
            damping=damping,
            position=position,
            seed=scan_seed,
            freq_profile=freq_profile,
        )

    else:  # "chaotic" — Lorenz / Rössler / Duffing / Chua
        signal = _render_chaotic_slot(
            n_samples=n_samples,
            freq_profile=freq_profile,
            sample_rate=sample_rate,
            rng=rng,
            params=params,
        )

    peak = float(np.max(np.abs(signal)))
    if peak > 1.0:
        signal = signal / peak
    return signal.astype(np.float64, copy=False)


def _build_harmonic_partials(
    *,
    n_harmonics: int,
    rolloff: float,
    brightness_tilt: float,
    odd_even_balance: float,
) -> list[dict[str, Any]]:
    """Simple harmonic series partial list.

    amp(k) = rolloff^(k-1) * k^brightness_tilt, weighted by an odd/even
    balance in [-1, +1]: -1 = even only, +1 = odd only, 0 = equal.  The
    fundamental is always preserved.
    """
    n_harmonics = max(1, int(n_harmonics))
    rolloff = float(rolloff)
    tilt = float(brightness_tilt)
    balance = float(max(-1.0, min(1.0, odd_even_balance)))
    odd_w = 0.5 * (1.0 + balance)
    even_w = 0.5 * (1.0 - balance)

    partials: list[dict[str, Any]] = []
    for k in range(1, n_harmonics + 1):
        amp = (rolloff ** (k - 1)) * (float(k) ** tilt)
        if k == 1:
            weight = 1.0
        elif k % 2 == 1:
            weight = odd_w
        else:
            weight = even_w
        amp *= weight
        if amp <= 0.0:
            continue
        partials.append({"ratio": float(k), "amp": float(amp), "phase": 0.0})
    return partials


def _build_drawbar_partials(
    *,
    amps: list[float] | tuple[float, ...],
    ratios: list[float] | tuple[float, ...],
) -> list[dict[str, Any]]:
    """Hammond-style drawbar additive partials from (ratio, amp) pairs."""
    partials: list[dict[str, Any]] = []
    for ratio, amp in zip(ratios, amps, strict=True):
        amp_f = float(amp)
        if amp_f <= 0.0:
            continue
        partials.append({"ratio": float(ratio), "amp": amp_f, "phase": 0.0})
    return partials


def render_partials(
    *,
    n_samples: int,
    freq_profile: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    partials_type: str,
    params: dict[str, Any],
) -> np.ndarray:
    """Render the ``partials`` slot: additive / spectralwave / drawbars.

    All three architectures funnel into
    :func:`code_musics.engines.va._render_partial_bank`, which handles the
    Nyquist taper and per-sample fade for sweeping pitch trajectories.
    """
    if partials_type not in _VALID_PARTIALS_TYPES:
        raise ValueError(
            f"partials_type must be one of {sorted(_VALID_PARTIALS_TYPES)} or "
            f"None, got {partials_type!r}"
        )
    del rng  # determinism handled by start_phase=0 in _render_partial_bank

    if freq_profile.shape[0] != n_samples:
        raise ValueError(
            f"freq_profile length {freq_profile.shape[0]} != n_samples {n_samples}"
        )

    if partials_type == "additive":
        explicit = params.get("partials_partials")
        if explicit:
            partials = [
                {
                    "ratio": float(p["ratio"]),
                    "amp": float(p["amp"]),
                    "phase": float(p.get("phase", 0.0)),
                }
                for p in explicit
            ]
        else:
            partials = _build_harmonic_partials(
                n_harmonics=int(params.get("partials_n_harmonics", 16)),
                rolloff=float(params.get("partials_harmonic_rolloff", 0.7)),
                brightness_tilt=float(params.get("partials_brightness_tilt", 0.0)),
                odd_even_balance=float(params.get("partials_odd_even_balance", 0.0)),
            )
    elif partials_type == "spectralwave":
        partials = _build_spectralwave_partials(
            position=float(params.get("partials_spectral_position", 0.5)),
            n_partials=int(params.get("partials_n_harmonics", 24)),
        )
    else:  # "drawbars"
        ratios = params.get("partials_drawbar_ratios") or _HAMMOND_DRAWBAR_RATIOS
        amps = params.get("partials_drawbar_amps") or _HAMMOND_DRAWBAR_DEFAULT_AMPS
        if len(amps) != len(ratios):
            raise ValueError(
                f"partials_drawbar_amps length {len(amps)} != "
                f"partials_drawbar_ratios length {len(ratios)}"
            )
        partials = _build_drawbar_partials(amps=amps, ratios=ratios)

    if not partials:
        return np.zeros(n_samples, dtype=np.float64)

    signal = _render_partial_bank(
        partials=partials,
        freq_profile=freq_profile,
        sample_rate=sample_rate,
        start_phase=0.0,
    )

    peak = float(np.max(np.abs(signal)))
    if peak > 1.0:
        signal = signal / peak
    return signal


def render_fm(
    *,
    n_samples: int,
    freq_profile: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    fm_type: str,
    params: dict[str, Any],
) -> np.ndarray:
    """Render the ``fm`` slot: 2-op FM.

    Thin adapter around :func:`code_musics.engines.fm._fm_sample_loop`;
    the orchestrator owns analog jitter / voice card / drift.
    """
    if fm_type not in _VALID_FM_TYPES:
        raise ValueError(
            f"fm_type must be one of {sorted(_VALID_FM_TYPES)} or None, got {fm_type!r}"
        )
    del rng  # kernel is deterministic

    if n_samples <= 0:
        return np.zeros(0, dtype=np.float64)

    freq_profile = np.asarray(freq_profile, dtype=np.float64)
    if freq_profile.ndim != 1 or freq_profile.size != n_samples:
        raise ValueError(
            f"freq_profile must be 1-D of length {n_samples}; "
            f"got shape {freq_profile.shape}"
        )

    carrier_ratio = float(params.get("fm_carrier_ratio", 1.0))
    # ``fm_ratio`` is primary; ``fm_mod_ratio`` accepted as a legacy alias.
    if "fm_ratio" in params:
        mod_ratio = float(params["fm_ratio"])
    else:
        mod_ratio = float(params.get("fm_mod_ratio", 1.0))
    mod_index = float(params.get("fm_index", 1.5))
    feedback = float(params.get("fm_feedback", 0.0))
    index_decay = float(params.get("fm_index_decay", 0.0))
    index_sustain = float(params.get("fm_index_sustain", 0.5))

    if carrier_ratio <= 0:
        raise ValueError("fm_carrier_ratio must be positive")
    if mod_ratio <= 0:
        raise ValueError("fm_ratio must be positive")
    if mod_index < 0:
        raise ValueError("fm_index must be non-negative")
    if not 0.0 <= feedback <= 1.0:
        raise ValueError(f"fm_feedback must be in [0.0, 1.0]; got {feedback!r}")
    if index_decay < 0:
        raise ValueError("fm_index_decay must be non-negative")

    carrier_phase_increment = 2.0 * np.pi * freq_profile * carrier_ratio / sample_rate
    mod_phase_increment = 2.0 * np.pi * freq_profile * mod_ratio / sample_rate

    index_decay_samples = int(index_decay * sample_rate)
    index_decay_samples = min(max(index_decay_samples, 0), n_samples)
    sustain_scale = max(0.0, index_sustain)

    signal = np.empty(n_samples, dtype=np.float64)
    _fm_sample_loop(
        signal,
        carrier_phase_increment,
        mod_phase_increment,
        mod_index,
        feedback,
        index_decay_samples,
        sustain_scale,
        n_samples,
    )

    peak = float(np.max(np.abs(signal)))
    if peak > 1.0:
        signal = signal / peak
    return signal


def render_noise(
    *,
    n_samples: int,
    freq: float,
    sample_rate: int,
    rng: np.random.Generator,
    noise_type: str,
    params: dict[str, Any],
) -> np.ndarray:
    """Render the ``noise`` slot: white / pink / bandpass / flow / rain / found."""
    if noise_type not in _VALID_NOISE_TYPES:
        raise ValueError(
            f"noise_type must be one of {sorted(_VALID_NOISE_TYPES)} or None, "
            f"got {noise_type!r}"
        )

    if n_samples <= 0:
        return np.zeros(max(0, n_samples), dtype=np.float64)

    if noise_type == "white":
        out = rng.standard_normal(n_samples).astype(np.float64)

    elif noise_type == "pink":
        white = rng.standard_normal(n_samples).astype(np.float64)
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
        scale = np.ones_like(freqs)
        scale[1:] = 1.0 / np.sqrt(freqs[1:])
        spectrum = spectrum * scale
        spectrum[0] = 0.0
        out = np.fft.irfft(spectrum, n=n_samples).real

    elif noise_type == "bandpass":
        center_hz = float(params.get("noise_center_hz", freq))
        width_ratio = float(params.get("noise_bandwidth_ratio", 0.5))
        white = rng.standard_normal(n_samples).astype(np.float64)
        out = bandpass_noise(
            white,
            sample_rate=sample_rate,
            center_hz=center_hz,
            width_ratio=width_ratio,
        )

    elif noise_type == "flow":
        density = float(params.get("noise_flow_density", 0.3))
        density = float(np.clip(density, 0.0, 1.0))
        out = flow_exciter(n_samples=n_samples, param=density, rng=rng)

    elif noise_type == "found":
        texture = params.get("noise_found_texture")
        if texture is None:
            raise ValueError(
                "noise_type='found' requires a 'noise_found_texture' param; "
                f"one of {FOUND_SOUND_TEXTURES}"
            )
        found_seed = int(rng.integers(0, 2**31 - 1))
        out = found_sound(
            str(texture),
            n_samples,
            sample_rate,
            density=float(np.clip(params.get("noise_found_density", 0.5), 0.0, 1.0)),
            brightness=float(
                np.clip(params.get("noise_found_brightness", 0.5), 0.0, 1.0)
            ),
            movement=float(np.clip(params.get("noise_found_movement", 0.5), 0.0, 1.0)),
            seed=found_seed,
        )

    else:  # "rain"
        rain_seed = int(rng.integers(0, 2**31 - 1))
        out = rain_exciter(
            n_samples,
            sample_rate,
            density=float(np.clip(params.get("noise_rain_density", 0.5), 0.0, 1.0)),
            brightness=float(
                np.clip(params.get("noise_rain_brightness", 0.5), 0.0, 1.0)
            ),
            drop_size=float(np.clip(params.get("noise_rain_drop_size", 0.5), 0.0, 1.0)),
            wash=float(np.clip(params.get("noise_rain_wash", 0.5), 0.0, 1.0)),
            seed=rain_seed,
        )

    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float64, copy=False)
