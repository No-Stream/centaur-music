"""Noise/percussion engine tests."""

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
            "pitch_decay": 0.06,
            "tone_decay": 0.14,
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
        "pitch_decay": 0.08,
        "tone_decay": 0.18,
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

    pitched = render(**{**common_kwargs, "params": {**common_params, "noise_mix": 0.05}})
    noisy = render(**{**common_kwargs, "params": {**common_params, "noise_mix": 0.9}})

    assert not np.allclose(pitched, noisy)


def test_bandpass_and_click_params_materially_change_the_sound() -> None:
    base_params = {
        "noise_mix": 0.6,
        "pitch_decay": 0.05,
        "tone_decay": 0.16,
    }
    base_kwargs = {
        "freq": 140.0,
        "duration": 0.2,
        "amp": 0.6,
        "sample_rate": 44100,
        "params": base_params,
    }

    darker = render(
        **{**base_kwargs, "params": {**base_params, "bandpass_ratio": 0.7, "click_amount": 0.02}},
    )
    brighter = render(
        **{**base_kwargs, "params": {**base_params, "bandpass_ratio": 2.0, "click_amount": 0.25}},
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
            "pitch_decay": 0.04,
            "tone_decay": 0.12,
            "bandpass_ratio": 1.1,
            "click_amount": 0.06,
        },
    }

    first = render(**kwargs)
    second = render(**kwargs)

    assert np.allclose(first, second)
