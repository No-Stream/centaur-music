"""Kick/tom engine tests."""

from __future__ import annotations

import numpy as np

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
                "drive_ratio": 0.02,
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
                "drive_ratio": 0.40,
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

    assert tom_mid_ratio > kick_mid_ratio


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
            "drive_ratio": 0.2,
        },
    }

    first = render(**kwargs)
    second = render(**kwargs)

    assert np.allclose(first, second)
