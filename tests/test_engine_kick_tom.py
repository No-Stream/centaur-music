"""Kick/tom engine tests."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from code_musics.engines.kick_tom import render
from code_musics.engines.registry import resolve_synth_params


def _band_energy(
    signal: np.ndarray,
    *,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
) -> float:
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    mask = (freqs >= low_hz) & (freqs < high_hz)
    return float(np.sum(spectrum[mask]))


def test_kick_tom_render_returns_expected_length_and_is_finite() -> None:
    sample_rate = 44_100
    duration = 0.5

    audio = render(
        freq=58.0,
        duration=duration,
        amp=0.8,
        sample_rate=sample_rate,
        params={},
    )

    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert len(audio) == int(sample_rate * duration)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_pitch_sweep_changes_the_attack_character() -> None:
    base_kwargs = {
        "freq": 60.0,
        "duration": 0.35,
        "amp": 0.7,
        "sample_rate": 44_100,
        "params": {
            "decay_ms": 280.0,
            "body_wave": "sine",
            "click_amount": 0.0,
            "noise_amount": 0.0,
        },
    }

    restrained = render(
        **{
            **base_kwargs,
            "params": {
                **base_kwargs["params"],
                "pitch_sweep_amount_ratio": 1.2,
                "pitch_sweep_decay_ms": 60.0,
            },
        }
    )
    aggressive = render(
        **{
            **base_kwargs,
            "params": {
                **base_kwargs["params"],
                "pitch_sweep_amount_ratio": 4.0,
                "pitch_sweep_decay_ms": 18.0,
            },
        }
    )

    assert not np.allclose(restrained, aggressive)
    assert np.argmax(np.abs(aggressive[:2048])) < np.argmax(np.abs(restrained[:2048]))


def test_transient_and_drive_controls_materially_change_the_sound() -> None:
    base_kwargs = {
        "freq": 64.0,
        "duration": 0.3,
        "amp": 0.6,
        "sample_rate": 44_100,
        "params": {
            "decay_ms": 250.0,
            "pitch_sweep_amount_ratio": 2.5,
            "pitch_sweep_decay_ms": 32.0,
        },
    }

    softer = render(
        **{
            **base_kwargs,
            "params": {
                **base_kwargs["params"],
                "click_amount": 0.02,
                "noise_amount": 0.0,
            },
        }
    )
    harder = render(
        **{
            **base_kwargs,
            "params": {
                **base_kwargs["params"],
                "click_amount": 0.25,
                "noise_amount": 0.05,
            },
        }
    )

    assert not np.allclose(softer, harder)


def test_round_tom_has_more_ring_energy_than_808_hiphop() -> None:
    sample_rate = 44_100
    common_kwargs = {
        "freq": 72.0,
        "duration": 0.55,
        "amp": 0.7,
        "sample_rate": sample_rate,
    }

    kick = render(
        **{
            **common_kwargs,
            "params": resolve_synth_params(
                {"engine": "kick_tom", "preset": "808_hiphop"}
            ),
        }
    )
    tom = render(
        **{
            **common_kwargs,
            "params": resolve_synth_params(
                {"engine": "kick_tom", "preset": "round_tom"}
            ),
        }
    )

    kick_mid_ratio = _band_energy(
        kick, sample_rate=sample_rate, low_hz=180.0, high_hz=900.0
    ) / max(
        _band_energy(kick, sample_rate=sample_rate, low_hz=20.0, high_hz=120.0), 1e-9
    )
    tom_mid_ratio = _band_energy(
        tom, sample_rate=sample_rate, low_hz=180.0, high_hz=900.0
    ) / max(
        _band_energy(tom, sample_rate=sample_rate, low_hz=20.0, high_hz=120.0), 1e-9
    )

    assert tom_mid_ratio > kick_mid_ratio * 0.98


def test_kick_tom_render_is_deterministic_for_identical_inputs() -> None:
    kwargs = {
        "freq": 62.0,
        "duration": 0.4,
        "amp": 0.5,
        "sample_rate": 44_100,
        "params": {
            "decay_ms": 320.0,
            "pitch_sweep_amount_ratio": 2.7,
            "pitch_sweep_decay_ms": 36.0,
            "click_amount": 0.1,
            "noise_amount": 0.03,
        },
    }

    first = render(**kwargs)
    second = render(**kwargs)

    assert np.allclose(first, second)


def test_kick_tom_with_body_amp_envelope_renders_finite() -> None:
    audio = render(
        freq=55.0,
        duration=0.5,
        amp=0.8,
        sample_rate=44_100,
        params={
            "body_amp_envelope": [
                {"time": 0.0, "value": 1.0},
                {"time": 0.6, "value": 0.8, "curve": "linear"},
                {"time": 0.65, "value": 0.0, "curve": "exponential"},
            ],
        },
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_kick_tom_with_pitch_envelope_renders_finite() -> None:
    audio = render(
        freq=60.0,
        duration=0.4,
        amp=0.7,
        sample_rate=44_100,
        params={
            "pitch_envelope": [
                {"time": 0.0, "value": 4.0},
                {"time": 0.05, "value": 1.5, "curve": "exponential"},
                {"time": 0.2, "value": 1.0, "curve": "linear"},
            ],
        },
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_kick_tom_with_body_filter_renders_finite() -> None:
    audio = render(
        freq=58.0,
        duration=0.4,
        amp=0.7,
        sample_rate=44_100,
        params={
            "body_filter_mode": "lowpass",
            "body_filter_cutoff_hz": 1200.0,
            "body_filter_q": 1.2,
            "body_filter_drive": 0.0,
            "body_filter_envelope": [
                {"time": 0.0, "value": 4000.0},
                {"time": 0.1, "value": 1200.0, "curve": "exponential"},
                {"time": 1.0, "value": 400.0, "curve": "linear"},
            ],
        },
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_kick_tom_default_matches_when_no_envelope() -> None:
    shared_params = {
        "decay_ms": 260.0,
        "pitch_sweep_amount_ratio": 2.5,
        "pitch_sweep_decay_ms": 42.0,
        "body_wave": "sine",
        "body_tone_ratio": 0.16,
        "body_punch_ratio": 0.20,
        "overtone_amount": 0.10,
        "overtone_ratio": 1.9,
        "overtone_decay_ms": 110.0,
        "click_amount": 0.0,
        "noise_amount": 0.0,
    }
    kwargs = {
        "freq": 60.0,
        "duration": 0.3,
        "amp": 0.6,
        "sample_rate": 44_100,
    }
    without_envelopes = render(**kwargs, params=shared_params)
    with_none_envelopes = render(
        **kwargs,
        params={
            **shared_params,
            "body_amp_envelope": None,
            "pitch_envelope": None,
            "overtone_amp_envelope": None,
        },
    )
    assert np.allclose(without_envelopes, with_none_envelopes)


def test_gated_808_preset_renders() -> None:
    params = resolve_synth_params({"engine": "kick_tom", "preset": "gated_808"})
    audio = render(
        freq=52.0,
        duration=0.7,
        amp=0.8,
        sample_rate=44_100,
        params=params,
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_pitch_dive_preset_renders() -> None:
    params = resolve_synth_params({"engine": "kick_tom", "preset": "pitch_dive"})
    audio = render(
        freq=56.0,
        duration=0.4,
        amp=0.7,
        sample_rate=44_100,
        params=params,
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_filtered_kick_preset_renders() -> None:
    params = resolve_synth_params({"engine": "kick_tom", "preset": "filtered_kick"})
    audio = render(
        freq=58.0,
        duration=0.45,
        amp=0.7,
        sample_rate=44_100,
        params=params,
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_kick_tom_fm_body_renders_finite() -> None:
    audio = render(
        freq=55.0,
        duration=0.4,
        amp=0.8,
        sample_rate=44_100,
        params={
            "body_fm_ratio": 1.41,
            "body_fm_index": 3.0,
            "body_fm_feedback": 0.0,
        },
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_kick_tom_fm_body_differs_from_standard() -> None:
    common_kwargs = {
        "freq": 60.0,
        "duration": 0.35,
        "amp": 0.7,
        "sample_rate": 44_100,
    }
    standard = render(
        **common_kwargs,
        params={"body_wave": "sine", "click_amount": 0.0, "noise_amount": 0.0},
    )
    fm_body = render(
        **common_kwargs,
        params={
            "body_fm_ratio": 1.41,
            "body_fm_index": 3.5,
            "click_amount": 0.0,
            "noise_amount": 0.0,
        },
    )
    assert not np.allclose(standard, fm_body)


def test_kick_tom_body_distortion_renders_finite() -> None:
    audio = render(
        freq=58.0,
        duration=0.4,
        amp=0.7,
        sample_rate=44_100,
        params={
            "body_distortion": "foldback",
            "body_distortion_drive": 0.45,
            "body_distortion_mix": 0.7,
        },
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_fm_body_kick_preset_renders() -> None:
    params = resolve_synth_params({"engine": "kick_tom", "preset": "fm_body_kick"})
    audio = render(
        freq=52.0,
        duration=0.5,
        amp=0.8,
        sample_rate=44_100,
        params=params,
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_foldback_kick_preset_renders() -> None:
    params = resolve_synth_params({"engine": "kick_tom", "preset": "foldback_kick"})
    audio = render(
        freq=56.0,
        duration=0.4,
        amp=0.7,
        sample_rate=44_100,
        params=params,
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


# --- resonator body mode tests ---


def test_resonator_mode_renders_without_error() -> None:
    audio = render(
        freq=55.0,
        duration=0.5,
        amp=0.8,
        sample_rate=44_100,
        params={"body_mode": "resonator"},
    )
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert len(audio) == int(44_100 * 0.5)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_resonator_mode_differs_from_oscillator() -> None:
    common_kwargs = {
        "freq": 60.0,
        "duration": 0.35,
        "amp": 0.7,
        "sample_rate": 44_100,
    }
    oscillator = render(
        **common_kwargs,
        params={"body_mode": "oscillator", "click_amount": 0.0, "noise_amount": 0.0},
    )
    resonator = render(
        **common_kwargs,
        params={"body_mode": "resonator", "click_amount": 0.0, "noise_amount": 0.0},
    )
    assert not np.allclose(oscillator, resonator)


def test_resonator_presets_render_successfully() -> None:
    for preset_name in ("808_resonant", "808_resonant_long", "resonant_tom"):
        params = resolve_synth_params({"engine": "kick_tom", "preset": preset_name})
        audio = render(
            freq=55.0,
            duration=0.5,
            amp=0.8,
            sample_rate=44_100,
            params=params,
        )
        assert np.isfinite(audio).all(), f"preset {preset_name} produced non-finite"
        assert np.max(np.abs(audio)) > 0, f"preset {preset_name} produced silence"


def test_invalid_body_mode_raises() -> None:
    with pytest.raises(ValueError, match="body_mode"):
        render(
            freq=55.0,
            duration=0.3,
            amp=0.7,
            sample_rate=44_100,
            params={"body_mode": "invalid"},
        )


def test_resonator_with_fm_warns_and_renders(caplog: pytest.LogCaptureFixture) -> None:

    with caplog.at_level(logging.WARNING):
        audio = render(
            freq=55.0,
            duration=0.4,
            amp=0.7,
            sample_rate=44_100,
            params={
                "body_mode": "resonator",
                "body_fm_ratio": 1.41,
                "body_fm_index": 3.0,
            },
        )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0
    assert any("body_fm_ratio is ignored" in msg for msg in caplog.messages)
