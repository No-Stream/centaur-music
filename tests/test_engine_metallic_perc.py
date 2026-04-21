"""Metallic percussion engine tests."""

from __future__ import annotations

import numpy as np

from code_musics.engines.metallic_perc import render


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


def test_metallic_perc_render_returns_expected_length_and_is_finite() -> None:
    sample_rate = 44_100
    duration = 0.3

    audio = render(
        freq=440.0,
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


def test_brightness_changes_spectral_character() -> None:
    sample_rate = 44_100
    base_kwargs = {
        "freq": 440.0,
        "duration": 0.3,
        "amp": 0.7,
        "sample_rate": sample_rate,
        "params": {
            "decay_ms": 200.0,
            "click_amount": 0.0,
            "density": 0.0,
        },
    }

    dark = render(
        **{**base_kwargs, "params": {**base_kwargs["params"], "brightness": 0.3}}
    )
    bright = render(
        **{**base_kwargs, "params": {**base_kwargs["params"], "brightness": 0.9}}
    )

    assert not np.allclose(dark, bright)

    dark_high = _band_energy(
        dark, sample_rate=sample_rate, low_hz=2000.0, high_hz=8000.0
    )
    bright_high = _band_energy(
        bright, sample_rate=sample_rate, low_hz=2000.0, high_hz=8000.0
    )
    assert bright_high > dark_high


def test_decay_changes_tail_energy() -> None:
    sample_rate = 44_100
    duration = 0.5
    base_kwargs = {
        "freq": 440.0,
        "duration": duration,
        "amp": 0.8,
        "sample_rate": sample_rate,
    }

    short_decay = render(
        **{**base_kwargs, "params": {"decay_ms": 30.0, "density": 0.0}}
    )
    long_decay = render(
        **{**base_kwargs, "params": {"decay_ms": 500.0, "density": 0.0}}
    )

    half = len(short_decay) // 2
    short_tail_energy = float(np.sum(short_decay[half:] ** 2))
    long_tail_energy = float(np.sum(long_decay[half:] ** 2))

    assert long_tail_energy > short_tail_energy


def test_ring_mod_adds_content() -> None:
    sample_rate = 44_100
    base_kwargs = {
        "freq": 440.0,
        "duration": 0.3,
        "amp": 0.7,
        "sample_rate": sample_rate,
        "params": {
            "decay_ms": 150.0,
            "click_amount": 0.0,
            "density": 0.0,
        },
    }

    dry = render(
        **{**base_kwargs, "params": {**base_kwargs["params"], "ring_mod_amount": 0.0}}
    )
    wet = render(
        **{**base_kwargs, "params": {**base_kwargs["params"], "ring_mod_amount": 0.6}}
    )

    assert not np.allclose(dry, wet)


def test_render_is_deterministic_for_identical_inputs() -> None:
    kwargs = {
        "freq": 440.0,
        "duration": 0.3,
        "amp": 0.5,
        "sample_rate": 44_100,
        "params": {
            "decay_ms": 100.0,
            "brightness": 0.6,
            "density": 0.4,
            "click_amount": 0.08,
            "ring_mod_amount": 0.2,
        },
    }

    first = render(**kwargs)
    second = render(**kwargs)

    assert np.allclose(first, second)


def test_high_frequency_render_is_finite() -> None:
    """Render at high freq that previously caused NaN via SVF instability."""
    for freq in [10500.0, 12000.0, 15000.0, 18000.0]:
        audio = render(freq=freq, duration=0.04, amp=0.5, sample_rate=44100, params={})
        assert np.isfinite(audio).all(), f"NaN at freq={freq}"
        assert np.max(np.abs(audio)) > 0


# ---------------------------------------------------------------------------
# Multi-point envelope tests
# ---------------------------------------------------------------------------


def test_metallic_perc_with_amp_envelope_renders_finite() -> None:
    sample_rate = 44_100
    duration = 0.3
    audio = render(
        freq=440.0,
        duration=duration,
        amp=0.8,
        sample_rate=sample_rate,
        params={
            "amp_envelope": [
                {"time": 0.0, "value": 1.0},
                {"time": 0.6, "value": 0.4, "curve": "exponential"},
                {"time": 1.0, "value": 0.0, "curve": "linear"},
            ],
        },
    )
    assert len(audio) == int(sample_rate * duration)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_metallic_perc_with_filter_envelope_renders_finite() -> None:
    sample_rate = 44_100
    duration = 0.3
    audio = render(
        freq=440.0,
        duration=duration,
        amp=0.8,
        sample_rate=sample_rate,
        params={
            "filter_envelope": [
                {"time": 0.0, "value": 800.0},
                {"time": 0.3, "value": 6000.0, "curve": "exponential"},
                {"time": 1.0, "value": 3000.0, "curve": "linear"},
            ],
        },
    )
    assert len(audio) == int(sample_rate * duration)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_metallic_perc_default_matches_when_no_envelope() -> None:
    """Omitting envelope params produces the same output as not having them at all."""
    kwargs = {
        "freq": 440.0,
        "duration": 0.3,
        "amp": 0.7,
        "sample_rate": 44_100,
        "params": {
            "decay_ms": 100.0,
            "brightness": 0.6,
            "density": 0.4,
            "click_amount": 0.08,
        },
    }

    first = render(**kwargs)
    second = render(**kwargs)

    assert np.allclose(first, second)


def test_swept_hat_preset_renders() -> None:
    from code_musics.engines.registry import resolve_synth_params

    resolved = resolve_synth_params({"engine": "metallic_perc", "preset": "swept_hat"})
    audio = render(
        freq=440.0,
        duration=0.3,
        amp=0.8,
        sample_rate=44_100,
        params=resolved,
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_decaying_bell_preset_renders() -> None:
    from code_musics.engines.registry import resolve_synth_params

    resolved = resolve_synth_params(
        {"engine": "metallic_perc", "preset": "decaying_bell"}
    )
    audio = render(
        freq=440.0,
        duration=0.8,
        amp=0.8,
        sample_rate=44_100,
        params=resolved,
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


# ---------------------------------------------------------------------------
# Square oscillator mode tests
# ---------------------------------------------------------------------------


def test_square_mode_renders_without_error() -> None:
    sample_rate = 44_100
    duration = 0.3
    audio = render(
        freq=440.0,
        duration=duration,
        amp=0.8,
        sample_rate=sample_rate,
        params={"oscillator_mode": "square", "density": 0.0},
    )
    assert isinstance(audio, np.ndarray)
    assert len(audio) == int(sample_rate * duration)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_square_mode_differs_from_sine() -> None:
    common = {
        "freq": 440.0,
        "duration": 0.3,
        "amp": 0.8,
        "sample_rate": 44_100,
    }
    fixed_params = {"density": 0.0, "click_amount": 0.0, "noise_amount": 0.0}
    sine_audio = render(**common, params={**fixed_params, "oscillator_mode": "sine"})
    square_audio = render(
        **common, params={**fixed_params, "oscillator_mode": "square"}
    )
    assert not np.allclose(sine_audio, square_audio)


def test_808_preset_renders() -> None:
    from code_musics.engines.registry import resolve_synth_params

    for preset_name in ("808_closed_hat", "808_open_hat", "808_cowbell_square"):
        resolved = resolve_synth_params(
            {"engine": "metallic_perc", "preset": preset_name}
        )
        audio = render(
            freq=440.0,
            duration=0.3,
            amp=0.8,
            sample_rate=44_100,
            params=resolved,
        )
        assert np.isfinite(audio).all(), f"NaN in preset {preset_name}"
        assert np.max(np.abs(audio)) > 0, f"silent output for preset {preset_name}"


def test_invalid_oscillator_mode_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="oscillator_mode"):
        render(
            freq=440.0,
            duration=0.3,
            amp=0.8,
            sample_rate=44_100,
            params={"oscillator_mode": "triangle"},
        )
