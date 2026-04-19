"""Integration tests for the Machinedrum-inspired drum_voice layer modes.

Exercises the Phase-2 wiring:

- Tone: ``efm`` (2-op DX-style FM) and ``modal`` (modal resonator bank).
- Metallic: ``efm_cymbal`` (N-op PM) and ``modal_bank`` (modal resonator bank).
- Exciter: ``sample`` (WAV playback as transient layer).
- Physical-informed macros: ``pi_tension``, ``pi_damping``.

Each test drives ``drum_voice.render()`` end-to-end with realistic param dicts
and asserts finite, non-silent, non-degenerate output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from code_musics.engines.drum_voice import render

SAMPLE_RATE = 44_100


def _render(
    *,
    freq: float = 100.0,
    duration: float = 0.3,
    amp: float = 0.8,
    params: dict | None = None,
) -> np.ndarray:
    return render(
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=SAMPLE_RATE,
        params=params or {},
    )


def _spectral_centroid(signal: np.ndarray, sample_rate: int) -> float:
    if len(signal) == 0:
        return 0.0
    mag = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    total = float(mag.sum())
    if total < 1e-12:
        return 0.0
    return float((mag * freqs).sum() / total)


def _fft_peak_freq(signal: np.ndarray, sample_rate: int) -> float:
    mag = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    return float(freqs[int(np.argmax(mag))])


def _rms(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal**2)))


# ---------------------------------------------------------------------------
# Tone: efm
# ---------------------------------------------------------------------------


def test_drum_voice_tone_efm_non_silent() -> None:
    carrier_freq = 100.0
    audio = _render(
        freq=carrier_freq,
        params={
            "tone_type": "efm",
            "tone_level": 1.0,
            "efm_ratio": 1.5,
            "efm_index_peak": 3.0,
            "exciter_type": None,
            "noise_type": None,
            "metallic_type": None,
        },
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.0
    assert np.max(np.abs(audio)) < 2.0

    # Assert genuine FM behavior: with efm_ratio=1.5, efm_index_peak=3.0 the
    # output must carry substantial sideband energy off the carrier bin. A
    # silent fallback-to-sine path would fail this.
    spec = np.abs(np.fft.rfft(audio))
    n = audio.shape[0]
    carrier_bin = int(round(carrier_freq * n / SAMPLE_RATE))
    carrier_energy = float(spec[carrier_bin])
    off_carrier_energy = float(spec.sum() - carrier_energy)
    assert off_carrier_energy / (carrier_energy + 1e-12) > 0.1, (
        f"expected FM sideband energy; off/carrier ratio="
        f"{off_carrier_energy / (carrier_energy + 1e-12):.3f}"
    )


def test_drum_voice_tone_efm_second_mod_adds_complexity() -> None:
    single_mod = _render(
        params={
            "tone_type": "efm",
            "tone_level": 1.0,
            "efm_ratio": 1.5,
            "efm_index_peak": 3.0,
            "exciter_type": None,
            "noise_type": None,
            "metallic_type": None,
        },
    )
    with_second = _render(
        params={
            "tone_type": "efm",
            "tone_level": 1.0,
            "efm_ratio": 1.5,
            "efm_index_peak": 3.0,
            "efm_ratio_2": 2.73,
            "efm_index_2": 4.0,
            "exciter_type": None,
            "noise_type": None,
            "metallic_type": None,
        },
    )
    assert np.isfinite(with_second).all()
    # A second modulator injects additional sidebands; signals must differ and
    # the spectral centroid should move away from the single-mod baseline.
    assert not np.allclose(single_mod, with_second)
    c_single = _spectral_centroid(single_mod, SAMPLE_RATE)
    c_double = _spectral_centroid(with_second, SAMPLE_RATE)
    assert abs(c_single - c_double) > 20.0


# ---------------------------------------------------------------------------
# Tone: modal
# ---------------------------------------------------------------------------


def test_drum_voice_tone_modal_membrane() -> None:
    audio = _render(
        freq=80.0,
        duration=0.5,
        params={
            "tone_type": "modal",
            "tone_level": 1.0,
            "modal_mode_table": "membrane",
            "modal_n_modes": 6,
            "modal_decay_s": 0.4,
            "exciter_type": "click",
            "exciter_level": 1.0,
            "noise_type": None,
            "metallic_type": None,
        },
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.0


def test_drum_voice_tone_modal_exciter_drives_resonator() -> None:
    # When the exciter is present, the modal bank is driven by it; when
    # absent, the bank is seeded by a tiny fallback impulse.  After
    # drum_voice's peak normalization the overall loudness matches, but the
    # spectral and temporal content must still differ measurably because the
    # excitations have very different shapes.
    base_params: dict = {
        "tone_type": "modal",
        "tone_level": 1.0,
        "modal_mode_table": "membrane",
        "modal_n_modes": 6,
        "modal_decay_s": 0.4,
        "exciter_type": "click",
        "noise_type": None,
        "metallic_type": None,
    }
    undriven = _render(
        freq=80.0,
        duration=0.5,
        params={**base_params, "exciter_level": 0.0},
    )
    driven = _render(
        freq=80.0,
        duration=0.5,
        params={**base_params, "exciter_level": 1.0},
    )
    assert np.isfinite(undriven).all()
    assert np.isfinite(driven).all()
    assert not np.allclose(undriven, driven)
    # Driven excitation is broadband click noise vs tiny fallback impulse;
    # after modal filtering the early-window spectral content must differ.
    early = int(0.1 * SAMPLE_RATE)
    c_undriven = _spectral_centroid(undriven[:early], SAMPLE_RATE)
    c_driven = _spectral_centroid(driven[:early], SAMPLE_RATE)
    assert abs(c_undriven - c_driven) > 50.0, (
        f"expected exciter shape to change spectral centroid; got "
        f"undriven={c_undriven:.1f} driven={c_driven:.1f}"
    )


# ---------------------------------------------------------------------------
# Metallic: efm_cymbal
# ---------------------------------------------------------------------------


def test_drum_voice_metallic_efm_cymbal() -> None:
    audio = _render(
        freq=400.0,
        duration=0.3,
        params={
            "tone_type": None,
            "exciter_type": None,
            "noise_type": None,
            "metallic_type": "efm_cymbal",
            "metallic_level": 1.0,
            "cymbal_op_count": 4,
            "cymbal_ratio_set": "bar",
            "cymbal_index": 2.5,
        },
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.0
    centroid = _spectral_centroid(audio, SAMPLE_RATE)
    assert centroid > 2000.0, (
        f"expected high spectral centroid for metallic cymbal; got {centroid:.1f} Hz"
    )


# ---------------------------------------------------------------------------
# Metallic: modal_bank
# ---------------------------------------------------------------------------


def test_drum_voice_metallic_modal_bank_bell() -> None:
    audio = _render(
        freq=300.0,
        duration=0.5,
        params={
            "tone_type": None,
            "noise_type": None,
            "exciter_type": "click",
            "exciter_level": 1.0,
            "metallic_type": "modal_bank",
            "metallic_level": 1.0,
            "metallic_mode_table": "bar_metal",
            "metallic_n_modes": 5,
            "metallic_decay_s": 0.6,
        },
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.0


# ---------------------------------------------------------------------------
# PI macros
# ---------------------------------------------------------------------------


def test_drum_voice_pi_macros_tension_shifts_modes() -> None:
    base_params: dict = {
        "tone_type": "modal",
        "tone_level": 1.0,
        "modal_mode_table": "bar_metal",
        "modal_n_modes": 5,
        "modal_decay_s": 0.5,
        "exciter_type": "click",
        "exciter_level": 1.0,
        "noise_type": None,
        "metallic_type": None,
    }
    low = _render(
        freq=200.0,
        duration=0.4,
        params={**base_params, "pi_tension": -0.8},
    )
    high = _render(
        freq=200.0,
        duration=0.4,
        params={**base_params, "pi_tension": 0.8},
    )
    assert np.isfinite(low).all()
    assert np.isfinite(high).all()
    low_peak = _fft_peak_freq(low, SAMPLE_RATE)
    high_peak = _fft_peak_freq(high, SAMPLE_RATE)
    # Tension stretches mode ratios, so the dominant peak must move.
    assert abs(low_peak - high_peak) > 1.0, (
        f"expected tension to shift peak frequency; got {low_peak=:.1f} {high_peak=:.1f}"
    )


def test_drum_voice_pi_macros_damping_shortens_decay() -> None:
    # After drum_voice peak normalization, the initial hit will reach the same
    # target level regardless of damping.  Use a long tone_decay_s so the
    # layer envelope does not clip the intrinsic modal decay — otherwise the
    # two renders become indistinguishable after normalization.
    base_params: dict = {
        "tone_type": "modal",
        "tone_level": 1.0,
        "modal_mode_table": "membrane",
        "modal_n_modes": 6,
        "modal_decay_s": 0.5,
        "tone_decay_s": 2.0,
        "exciter_type": "click",
        "exciter_level": 1.0,
        "noise_type": None,
        "metallic_type": None,
    }
    long_ring = _render(
        freq=120.0,
        duration=0.6,
        params={**base_params, "pi_damping": 0.2},
    )
    short_ring = _render(
        freq=120.0,
        duration=0.6,
        params={**base_params, "pi_damping": 0.9},
    )
    assert np.isfinite(long_ring).all()
    assert np.isfinite(short_ring).all()
    # Damping should produce measurably different signals, not just numerical
    # drift.  This is the core behavioral guarantee of the pi_damping macro.
    assert not np.allclose(long_ring, short_ring, atol=1e-3)
    # The spectra also differ: a less-damped modal bank has narrower, more
    # resonant peaks (higher spectral "peakiness") than a heavily damped one.
    # Compare peak-to-mean magnitude ratio as a robust resonance proxy.
    long_spec = np.abs(np.fft.rfft(long_ring))
    short_spec = np.abs(np.fft.rfft(short_ring))
    long_peakiness = float(np.max(long_spec) / max(np.mean(long_spec), 1e-12))
    short_peakiness = float(np.max(short_spec) / max(np.mean(short_spec), 1e-12))
    assert long_peakiness > 1.2 * short_peakiness, (
        f"expected less damping to yield more resonant peaks; "
        f"long_peakiness={long_peakiness:.2f} short_peakiness={short_peakiness:.2f}"
    )


# ---------------------------------------------------------------------------
# Exciter: sample
# ---------------------------------------------------------------------------


def test_drum_voice_exciter_sample_generates_audio(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")

    # Synthesize a short 440 Hz sine WAV for use as the exciter sample.
    n_frames = int(0.05 * SAMPLE_RATE)
    t = np.arange(n_frames, dtype=np.float64) / SAMPLE_RATE
    tone = 0.6 * np.sin(2.0 * np.pi * 440.0 * t)
    sample_path = tmp_path / "exciter_tone.wav"
    soundfile.write(str(sample_path), tone, SAMPLE_RATE, subtype="PCM_16")

    audio = _render(
        freq=440.0,
        duration=0.15,
        params={
            "tone_type": None,
            "noise_type": None,
            "metallic_type": None,
            "exciter_type": "sample",
            "exciter_level": 1.0,
            "exciter_sample_path": str(sample_path),
            "exciter_sample_root_freq": 440.0,
            "exciter_sample_pitch_shift": False,
        },
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_drum_voice_unknown_modal_table_raises() -> None:
    with pytest.raises(ValueError):
        _render(
            freq=100.0,
            duration=0.2,
            params={
                "tone_type": "modal",
                "tone_level": 1.0,
                "modal_mode_table": "bogus",
                "exciter_type": "click",
                "exciter_level": 1.0,
                "noise_type": None,
                "metallic_type": None,
            },
        )


# ---------------------------------------------------------------------------
# Composite combination
# ---------------------------------------------------------------------------


def test_drum_voice_composite_efm_tone_plus_modal_metallic() -> None:
    audio = _render(
        freq=160.0,
        duration=0.4,
        params={
            "tone_type": "efm",
            "tone_level": 0.6,
            "efm_ratio": 1.41,
            "efm_index_peak": 2.5,
            "exciter_type": "click",
            "exciter_level": 0.4,
            "noise_type": None,
            "metallic_type": "modal_bank",
            "metallic_level": 0.6,
            "metallic_mode_table": "bar_metal",
            "metallic_n_modes": 5,
            "metallic_decay_s": 0.4,
        },
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.0
