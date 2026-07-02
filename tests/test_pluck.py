"""Karplus-Strong++ pluck primitive + synth_voice / drum_voice integration tests.

Covers:
    - Pitch accuracy across a wide range (55 Hz → 2200 Hz), <1% tolerance
    - Finite, nonzero, deterministic output
    - Off-state preserved when ``osc_type`` / ``tone_type`` is not "pluck"
    - Sustain=1 is numerically stable over 5 s
    - Damping knob actually affects decay length
    - Integration: each ``pluck_*`` preset renders finite audio through
      both ``synth_voice`` and ``drum_voice``
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._pluck import render_pluck
from code_musics.engines.drum_voice import render as render_drum_voice
from code_musics.engines.registry import _PRESETS
from code_musics.engines.synth_voice import render as render_synth_voice

SAMPLE_RATE = 44_100


# ---------------------------------------------------------------------------
# render_pluck primitive
# ---------------------------------------------------------------------------


def _measure_fundamental_hz(signal: np.ndarray, sample_rate: int) -> float:
    """Return the fundamental of ``signal`` via autocorrelation-peak detection.

    Autocorrelation is robust for plucked-string tones where the spectral
    peak is often a harmonic (e.g. pick-position filtering can suppress the
    fundamental's magnitude relative to the second partial).  We skip the
    excitation / attack segment, remove DC, and look for the first
    local-maximum lag in a plausible pitch range.
    """
    n = signal.shape[0]
    if n < 4096:
        raise ValueError("signal too short for fundamental measurement")

    start = n // 4
    segment = signal[start:].astype(np.float64)
    segment = segment - np.mean(segment)

    # Autocorrelation via rFFT power spectrum.
    window_len = segment.shape[0]
    padded = np.concatenate([segment, np.zeros(window_len, dtype=np.float64)])
    spectrum = np.fft.rfft(padded)
    power = spectrum * np.conj(spectrum)
    acf = np.fft.irfft(power, n=padded.shape[0]).real[:window_len]
    if acf[0] <= 0.0:
        raise RuntimeError("autocorrelation has non-positive energy at lag 0")
    acf = acf / acf[0]

    # Pick the first peak beyond the main lobe, within plausible pitch range.
    min_lag = max(2, int(sample_rate / 4000.0))  # up to 4 kHz
    max_lag = min(window_len - 2, int(sample_rate / 20.0))  # down to 20 Hz

    # Find the lag of the highest autocorrelation peak in the window.
    best_lag = -1
    best_val = -np.inf
    for lag in range(min_lag, max_lag):
        is_local_max = (
            acf[lag] > acf[lag - 1] and acf[lag] > acf[lag + 1] and acf[lag] > best_val
        )
        if is_local_max:
            best_val = float(acf[lag])
            best_lag = lag
    if best_lag < 0:
        raise RuntimeError("no autocorrelation peak found")

    # Parabolic interpolation for sub-sample precision.
    y_m1 = float(acf[best_lag - 1])
    y_0 = float(acf[best_lag])
    y_p1 = float(acf[best_lag + 1])
    denom = y_m1 - 2.0 * y_0 + y_p1
    offset = 0.5 * (y_m1 - y_p1) / denom if denom != 0.0 else 0.0
    refined_lag = float(best_lag) + offset
    return float(sample_rate) / refined_lag


@pytest.mark.parametrize("target_freq", [55.0, 220.0, 880.0, 2200.0])
def test_pluck_pitch_accuracy_within_one_percent(target_freq: float) -> None:
    duration = 2.0
    signal = render_pluck(
        freq=target_freq,
        duration=duration,
        sample_rate=SAMPLE_RATE,
        hardness=0.5,
        damping=0.1,  # low damping for a clearly-pitched tone
        position=0.3,
        sustain=0.5,
        drive=0.0,
        seed=12345,
    )
    assert np.isfinite(signal).all()
    assert np.max(np.abs(signal)) > 0.0

    measured = _measure_fundamental_hz(signal, SAMPLE_RATE)
    error_ratio = abs(measured - target_freq) / target_freq
    assert error_ratio < 0.01, (
        f"pitch error {error_ratio * 100:.3f}% at target {target_freq} Hz "
        f"(measured {measured:.2f} Hz) exceeds 1% tolerance"
    )


def test_pluck_output_is_finite_and_nonzero() -> None:
    signal = render_pluck(
        freq=220.0,
        duration=0.25,
        sample_rate=SAMPLE_RATE,
        hardness=0.5,
        damping=0.3,
        position=0.25,
        sustain=0.0,
        drive=0.0,
        seed=7,
    )
    assert signal.dtype == np.float64
    assert np.isfinite(signal).all()
    assert np.max(np.abs(signal)) > 0.0
    assert len(signal) == int(SAMPLE_RATE * 0.25)


def test_pluck_is_deterministic_for_same_seed_and_params() -> None:
    kwargs: dict = {
        "freq": 330.0,
        "duration": 0.4,
        "sample_rate": SAMPLE_RATE,
        "hardness": 0.6,
        "damping": 0.2,
        "position": 0.3,
        "sustain": 0.2,
        "drive": 0.1,
        "seed": 42,
    }
    first = render_pluck(**kwargs)
    second = render_pluck(**kwargs)
    assert np.array_equal(first, second)


def test_pluck_different_seeds_differ() -> None:
    common: dict = {
        "freq": 330.0,
        "duration": 0.4,
        "sample_rate": SAMPLE_RATE,
        "hardness": 0.6,
        "damping": 0.2,
        "position": 0.3,
        "sustain": 0.2,
        "drive": 0.0,
    }
    a = render_pluck(seed=1, **common)
    b = render_pluck(seed=2, **common)
    assert not np.array_equal(a, b)


def test_pluck_sustain_one_does_not_blow_up_over_5_seconds() -> None:
    signal = render_pluck(
        freq=220.0,
        duration=5.0,
        sample_rate=SAMPLE_RATE,
        hardness=0.5,
        damping=0.2,
        position=0.3,
        sustain=1.0,
        drive=0.0,
        seed=101,
    )
    assert np.isfinite(signal).all()
    peak = float(np.max(np.abs(signal)))
    assert peak < 2.0, f"sustain=1 produced peak {peak:.2f}, expected < 2.0"


def test_pluck_damping_zero_rings_longer_than_damping_one() -> None:
    """Higher damping → faster decay → less late-tail energy."""
    common: dict = {
        "freq": 220.0,
        "duration": 2.0,
        "sample_rate": SAMPLE_RATE,
        "hardness": 0.5,
        "position": 0.25,
        "sustain": 0.0,
        "drive": 0.0,
        "seed": 55,
    }
    bright = render_pluck(damping=0.0, **common)
    dark = render_pluck(damping=1.0, **common)

    late_start = int(0.75 * bright.shape[0])
    bright_tail_rms = float(np.sqrt(np.mean(bright[late_start:] ** 2)))
    dark_tail_rms = float(np.sqrt(np.mean(dark[late_start:] ** 2)))

    assert bright_tail_rms > 3.0 * dark_tail_rms, (
        f"expected damping=0 tail RMS to greatly exceed damping=1 tail RMS; "
        f"got bright={bright_tail_rms:.4f} dark={dark_tail_rms:.4f}"
    )


def test_pluck_freq_profile_respected() -> None:
    """Per-sample freq profile overrides the scalar freq for tuning."""
    n = int(SAMPLE_RATE * 0.5)
    freq_profile = np.full(n, 440.0, dtype=np.float64)
    signal = render_pluck(
        freq=110.0,  # deliberately different from profile
        duration=0.5,
        sample_rate=SAMPLE_RATE,
        hardness=0.5,
        damping=0.1,
        position=0.3,
        sustain=0.4,
        drive=0.0,
        seed=9,
        freq_profile=freq_profile,
    )
    measured = _measure_fundamental_hz(signal, SAMPLE_RATE)
    assert abs(measured - 440.0) / 440.0 < 0.02


def test_pluck_invalid_freq_raises() -> None:
    with pytest.raises(ValueError):
        render_pluck(
            freq=0.0,
            duration=0.25,
            sample_rate=SAMPLE_RATE,
            hardness=0.5,
            damping=0.3,
            position=0.25,
            sustain=0.0,
            drive=0.0,
            seed=0,
        )


def test_pluck_nyquist_guard_rejects_too_high_freq() -> None:
    with pytest.raises(ValueError):
        render_pluck(
            freq=30_000.0,
            duration=0.1,
            sample_rate=SAMPLE_RATE,
            hardness=0.5,
            damping=0.3,
            position=0.25,
            sustain=0.0,
            drive=0.0,
            seed=0,
        )


# ---------------------------------------------------------------------------
# synth_voice integration
# ---------------------------------------------------------------------------


def test_synth_voice_pluck_slot_produces_signal() -> None:
    audio = render_synth_voice(
        freq=220.0,
        duration=0.5,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={"osc_type": "pluck"},
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.05


def test_synth_voice_pluck_default_off_when_no_osc_type() -> None:
    """Off-state preserved: no osc_type=pluck → silent osc slot."""
    audio = render_synth_voice(
        freq=220.0,
        duration=0.1,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={},
    )
    assert np.max(np.abs(audio)) == 0.0


def test_synth_voice_pluck_pitch_tracks_freq() -> None:
    audio = render_synth_voice(
        freq=440.0,
        duration=1.5,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={
            "osc_type": "pluck",
            "osc_pluck_damping": 0.1,
            "osc_pluck_sustain": 0.6,
        },
    )
    measured = _measure_fundamental_hz(audio, SAMPLE_RATE)
    assert abs(measured - 440.0) / 440.0 < 0.02


@pytest.mark.parametrize(
    "preset_name",
    ["soft_pluck", "acid_pluck", "harp_pluck", "ebow_pluck", "cavernous_pluck"],
)
def test_synth_voice_pluck_presets_render(preset_name: str) -> None:
    assert preset_name in _PRESETS["synth_voice"], (
        f"preset {preset_name!r} not registered for synth_voice"
    )
    preset = _PRESETS["synth_voice"][preset_name]
    audio = render_synth_voice(
        freq=330.0,
        duration=0.6,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params=dict(preset),
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.0


# ---------------------------------------------------------------------------
# drum_voice integration
# ---------------------------------------------------------------------------


def test_drum_voice_pluck_tone_produces_signal() -> None:
    audio = render_drum_voice(
        freq=220.0,
        duration=0.4,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={
            "tone_type": "pluck",
            "tone_level": 1.0,
            "exciter_level": 0.0,
            "noise_level": 0.0,
            "metallic_level": 0.0,
        },
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.05


def test_drum_voice_pluck_driven_by_exciter_signal() -> None:
    """When an exciter layer is present, pluck should consume it as excitation."""
    audio = render_drum_voice(
        freq=180.0,
        duration=0.4,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={
            "tone_type": "pluck",
            "tone_level": 1.0,
            "exciter_type": "click",
            "exciter_level": 0.3,
            "noise_level": 0.0,
            "metallic_level": 0.0,
        },
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.05


@pytest.mark.parametrize(
    "preset_name",
    ["soft_pluck", "acid_pluck", "harp_pluck", "ebow_pluck", "cavernous_pluck"],
)
def test_drum_voice_pluck_presets_render(preset_name: str) -> None:
    assert preset_name in _PRESETS["drum_voice"], (
        f"preset {preset_name!r} not registered for drum_voice"
    )
    preset = _PRESETS["drum_voice"][preset_name]
    audio = render_drum_voice(
        freq=220.0,
        duration=0.6,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params=dict(preset),
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.0


def test_pluck_freq_profile_survives_duration_rounding() -> None:
    """Regression: n_samples -> duration -> int() round-trip lost a sample.

    92942 / 44100 * 44100 rounds down to 92941, so a caller that derived
    ``duration`` from its sample count got a length-mismatch ValueError
    when passing a matching ``freq_profile``.  The explicit ``n_samples``
    parameter must bypass the round-trip.
    """
    n_samples = 92942
    sample_rate = 44100
    duration = n_samples / float(sample_rate)
    assert int(duration * sample_rate) == n_samples - 1  # the lossy case

    freq_profile = np.full(n_samples, 220.0, dtype=np.float64)
    audio = render_pluck(
        freq=220.0,
        duration=duration,
        sample_rate=sample_rate,
        hardness=0.3,
        damping=0.3,
        position=0.3,
        sustain=0.2,
        drive=0.0,
        seed=7,
        freq_profile=freq_profile,
        n_samples=n_samples,
    )
    assert audio.shape[0] == n_samples
    assert np.isfinite(audio).all()
