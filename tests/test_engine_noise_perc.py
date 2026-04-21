"""Noise/percussion engine tests."""

from __future__ import annotations

import numpy as np

from code_musics.engines.noise_perc import render


def test_noise_perc_render_returns_expected_length_and_is_finite() -> None:
    sample_rate = 44100
    duration = 0.25

    audio = render(
        freq=110.0,
        duration=duration,
        amp=0.8,
        sample_rate=sample_rate,
        params={
            "noise_mix": 0.45,
            "pitch_decay_ms": 60.0,
            "tone_decay_ms": 140.0,
            "bandpass_ratio": 1.2,
            "click_amount": 0.1,
        },
    )

    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert len(audio) == int(sample_rate * duration)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_noise_mix_changes_the_output_character() -> None:
    common_params = {
        "pitch_decay_ms": 80.0,
        "tone_decay_ms": 180.0,
        "bandpass_ratio": 1.0,
        "click_amount": 0.08,
    }
    common_kwargs = {
        "freq": 180.0,
        "duration": 0.3,
        "amp": 0.7,
        "sample_rate": 44100,
        "params": common_params,
    }

    pitched = render(
        **{**common_kwargs, "params": {**common_params, "noise_mix": 0.05}}
    )
    noisy = render(**{**common_kwargs, "params": {**common_params, "noise_mix": 0.9}})

    assert not np.allclose(pitched, noisy)


def test_bandpass_and_click_params_materially_change_the_sound() -> None:
    base_params = {
        "noise_mix": 0.6,
        "pitch_decay_ms": 50.0,
        "tone_decay_ms": 160.0,
    }
    base_kwargs = {
        "freq": 140.0,
        "duration": 0.2,
        "amp": 0.6,
        "sample_rate": 44100,
        "params": base_params,
    }

    darker = render(
        **{
            **base_kwargs,
            "params": {**base_params, "bandpass_ratio": 0.7, "click_amount": 0.02},
        },
    )
    brighter = render(
        **{
            **base_kwargs,
            "params": {**base_params, "bandpass_ratio": 2.0, "click_amount": 0.25},
        },
    )

    assert not np.allclose(darker, brighter)


def test_render_is_deterministic_for_identical_inputs() -> None:
    kwargs = {
        "freq": 132.0,
        "duration": 0.18,
        "amp": 0.5,
        "sample_rate": 44100,
        "params": {
            "noise_mix": 0.4,
            "pitch_decay_ms": 40.0,
            "tone_decay_ms": 120.0,
            "bandpass_ratio": 1.1,
            "click_amount": 0.06,
        },
    }

    first = render(**kwargs)
    second = render(**kwargs)

    assert np.allclose(first, second)


def test_noise_perc_with_tone_amp_envelope_renders_finite() -> None:
    audio = render(
        freq=200.0,
        duration=0.3,
        amp=0.8,
        sample_rate=44_100,
        params={
            "tone_amp_envelope": [
                {"time": 0.0, "value": 1.0},
                {"time": 0.3, "value": 0.5, "curve": "exponential"},
                {"time": 1.0, "value": 0.0, "curve": "linear"},
            ],
        },
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_noise_perc_with_overall_amp_envelope_renders_finite() -> None:
    audio = render(
        freq=200.0,
        duration=0.3,
        amp=0.8,
        sample_rate=44_100,
        params={
            "overall_amp_envelope": [
                {"time": 0.0, "value": 1.0},
                {"time": 0.7, "value": 0.8, "curve": "linear"},
                {"time": 0.75, "value": 0.0, "curve": "exponential"},
            ],
        },
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_noise_perc_default_matches_when_no_envelope() -> None:
    kwargs = {
        "freq": 150.0,
        "duration": 0.2,
        "amp": 0.7,
        "sample_rate": 44_100,
        "params": {"noise_mix": 0.5, "tone_decay_ms": 150.0, "click_amount": 0.1},
    }
    first = render(**kwargs)
    second = render(**kwargs)
    assert np.allclose(first, second)


def test_shaped_hit_preset_renders() -> None:
    from code_musics.engines.registry import resolve_synth_params

    resolved = resolve_synth_params({"engine": "noise_perc", "preset": "shaped_hit"})
    audio = render(
        freq=200.0,
        duration=0.3,
        amp=0.8,
        sample_rate=44_100,
        params=resolved,
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0
