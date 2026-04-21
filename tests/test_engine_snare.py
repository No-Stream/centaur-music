"""Snare engine tests."""

from __future__ import annotations

import numpy as np

from code_musics.engines.snare import render


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


def test_snare_render_returns_expected_length_and_is_finite() -> None:
    sample_rate = 44_100
    duration = 0.5

    audio = render(
        freq=180.0,
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


def test_body_wire_mix_changes_spectral_character() -> None:
    sample_rate = 44_100
    common = {
        "freq": 180.0,
        "duration": 0.4,
        "amp": 0.7,
        "sample_rate": sample_rate,
    }

    body_heavy = render(
        **common,
        params={"body_mix": 0.9, "wire_mix": 0.1, "click_amount": 0.0},
    )
    wire_heavy = render(
        **common,
        params={"body_mix": 0.1, "wire_mix": 0.9, "click_amount": 0.0},
    )

    assert not np.allclose(body_heavy, wire_heavy)


def test_comb_filter_adds_pitched_resonance() -> None:
    sample_rate = 44_100
    freq = 200.0
    common = {
        "freq": freq,
        "duration": 0.4,
        "amp": 0.7,
        "sample_rate": sample_rate,
    }

    no_comb = render(
        **common,
        params={
            "body_mix": 0.0,
            "wire_mix": 1.0,
            "click_amount": 0.0,
            "comb_amount": 0.0,
        },
    )
    with_comb = render(
        **common,
        params={
            "body_mix": 0.0,
            "wire_mix": 1.0,
            "click_amount": 0.0,
            "comb_amount": 0.6,
        },
    )

    # The comb version should have more energy near `freq` than the no-comb version.
    band_low = freq * 0.8
    band_high = freq * 1.2
    energy_no_comb = _band_energy(
        no_comb, sample_rate=sample_rate, low_hz=band_low, high_hz=band_high
    )
    energy_with_comb = _band_energy(
        with_comb, sample_rate=sample_rate, low_hz=band_low, high_hz=band_high
    )

    assert energy_with_comb > energy_no_comb


def test_short_snare_decays_faster() -> None:
    sample_rate = 44_100
    common = {
        "freq": 180.0,
        "duration": 0.5,
        "amp": 0.7,
        "sample_rate": sample_rate,
    }

    short = render(
        **common,
        params={"body_decay_ms": 50.0, "wire_decay_ms": 60.0},
    )
    long = render(
        **common,
        params={"body_decay_ms": 200.0, "wire_decay_ms": 250.0},
    )

    # Compare late-to-early RMS ratio: a longer decay should sustain more
    # energy relative to its onset, giving a higher ratio.
    early_end = int(0.1 * len(short))
    late_start = int(0.5 * len(short))

    short_early_rms = float(np.sqrt(np.mean(short[:early_end] ** 2)))
    short_late_rms = float(np.sqrt(np.mean(short[late_start:] ** 2)))
    long_early_rms = float(np.sqrt(np.mean(long[:early_end] ** 2)))
    long_late_rms = float(np.sqrt(np.mean(long[late_start:] ** 2)))

    short_ratio = short_late_rms / max(short_early_rms, 1e-10)
    long_ratio = long_late_rms / max(long_early_rms, 1e-10)

    assert long_ratio > short_ratio


def test_render_is_deterministic_for_identical_inputs() -> None:
    kwargs = {
        "freq": 180.0,
        "duration": 0.4,
        "amp": 0.5,
        "sample_rate": 44_100,
        "params": {
            "body_decay_ms": 130.0,
            "wire_decay_ms": 190.0,
            "comb_amount": 0.5,
            "body_mix": 0.5,
            "wire_mix": 0.5,
            "click_amount": 0.12,
        },
    }

    first = render(**kwargs)
    second = render(**kwargs)

    assert np.allclose(first, second)


# ---------------------------------------------------------------------------
# Multi-point envelope tests
# ---------------------------------------------------------------------------


def test_snare_with_body_amp_envelope_renders_finite() -> None:
    sample_rate = 44_100
    duration = 0.4
    audio = render(
        freq=180.0,
        duration=duration,
        amp=0.8,
        sample_rate=sample_rate,
        params={
            "body_amp_envelope": [
                {"time": 0.0, "value": 1.0},
                {"time": 0.5, "value": 0.8, "curve": "linear"},
                {"time": 1.0, "value": 0.0, "curve": "exponential"},
            ],
        },
    )
    assert len(audio) == int(sample_rate * duration)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_snare_with_wire_filter_envelope_renders_finite() -> None:
    sample_rate = 44_100
    duration = 0.4
    audio = render(
        freq=180.0,
        duration=duration,
        amp=0.8,
        sample_rate=sample_rate,
        params={
            "wire_filter_envelope": [
                {"time": 0.0, "value": 400.0},
                {"time": 0.3, "value": 2000.0, "curve": "exponential"},
                {"time": 1.0, "value": 800.0, "curve": "linear"},
            ],
        },
    )
    assert len(audio) == int(sample_rate * duration)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_snare_default_matches_when_no_envelope() -> None:
    """Omitting envelope params produces the same output as not having them at all."""
    kwargs_a = {
        "freq": 180.0,
        "duration": 0.3,
        "amp": 0.7,
        "sample_rate": 44_100,
        "params": {
            "body_decay_ms": 120.0,
            "wire_decay_ms": 180.0,
            "comb_amount": 0.45,
            "body_mix": 0.5,
            "wire_mix": 0.5,
            "click_amount": 0.15,
        },
    }

    first = render(**kwargs_a)
    second = render(**kwargs_a)

    assert np.allclose(first, second)


def test_gated_snare_preset_renders() -> None:
    from code_musics.engines.registry import resolve_synth_params

    resolved = resolve_synth_params({"engine": "snare", "preset": "gated_snare"})
    audio = render(
        freq=180.0,
        duration=0.4,
        amp=0.8,
        sample_rate=44_100,
        params=resolved,
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


# ---------------------------------------------------------------------------
# FM body tests
# ---------------------------------------------------------------------------


def test_fm_body_renders_without_error() -> None:
    sample_rate = 44_100
    duration = 0.4
    audio = render(
        freq=180.0,
        duration=duration,
        amp=0.8,
        sample_rate=sample_rate,
        params={"body_fm_ratio": 1.5, "body_fm_index": 3.0},
    )
    assert len(audio) == int(sample_rate * duration)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_fm_body_differs_from_sine_body() -> None:
    common = {
        "freq": 180.0,
        "duration": 0.4,
        "amp": 0.8,
        "sample_rate": 44_100,
    }
    sine = render(**common, params={})
    fm = render(**common, params={"body_fm_ratio": 1.5, "body_fm_index": 3.0})
    assert not np.allclose(sine, fm)


def test_fm_body_with_index_envelope_renders() -> None:
    audio = render(
        freq=180.0,
        duration=0.4,
        amp=0.8,
        sample_rate=44_100,
        params={
            "body_fm_ratio": 1.5,
            "body_fm_index": 4.0,
            "body_fm_index_envelope": [
                {"time": 0.0, "value": 1.0},
                {"time": 0.05, "value": 0.1, "curve": "exponential"},
                {"time": 1.0, "value": 0.0, "curve": "linear"},
            ],
        },
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


# ---------------------------------------------------------------------------
# Waveshaper body tests
# ---------------------------------------------------------------------------


def test_waveshaper_renders_without_error() -> None:
    sample_rate = 44_100
    duration = 0.4
    audio = render(
        freq=180.0,
        duration=duration,
        amp=0.8,
        sample_rate=sample_rate,
        params={"body_distortion": "tanh", "body_distortion_drive": 0.5},
    )
    assert len(audio) == int(sample_rate * duration)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_invalid_distortion_algorithm_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="body_distortion must be one of"):
        render(
            freq=180.0,
            duration=0.4,
            amp=0.8,
            sample_rate=44_100,
            params={"body_distortion": "nonexistent_algo"},
        )


# ---------------------------------------------------------------------------
# Colored wire noise tests
# ---------------------------------------------------------------------------


def test_colored_wire_noise_renders_without_error() -> None:
    sample_rate = 44_100
    duration = 0.4
    audio = render(
        freq=180.0,
        duration=duration,
        amp=0.8,
        sample_rate=sample_rate,
        params={"wire_noise_mode": "colored"},
    )
    assert len(audio) == int(sample_rate * duration)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_colored_wire_differs_from_white_wire() -> None:
    common = {
        "freq": 180.0,
        "duration": 0.4,
        "amp": 0.8,
        "sample_rate": 44_100,
    }
    white = render(**common, params={"wire_noise_mode": "white"})
    colored = render(**common, params={"wire_noise_mode": "colored"})
    assert not np.allclose(white, colored)


def test_invalid_wire_noise_mode_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="wire_noise_mode must be one of"):
        render(
            freq=180.0,
            duration=0.4,
            amp=0.8,
            sample_rate=44_100,
            params={"wire_noise_mode": "pink"},
        )


# ---------------------------------------------------------------------------
# New preset tests
# ---------------------------------------------------------------------------


def test_fm_snare_preset_renders() -> None:
    from code_musics.engines.registry import resolve_synth_params

    resolved = resolve_synth_params({"engine": "snare", "preset": "fm_snare"})
    audio = render(
        freq=180.0,
        duration=0.4,
        amp=0.8,
        sample_rate=44_100,
        params=resolved,
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_driven_snare_preset_renders() -> None:
    from code_musics.engines.registry import resolve_synth_params

    resolved = resolve_synth_params({"engine": "snare", "preset": "driven_snare"})
    audio = render(
        freq=180.0,
        duration=0.4,
        amp=0.8,
        sample_rate=44_100,
        params=resolved,
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0
