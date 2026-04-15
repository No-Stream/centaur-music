"""Clap engine tests."""

from __future__ import annotations

import numpy as np

from code_musics.engines.clap import render


def test_clap_render_returns_expected_length_and_is_finite() -> None:
    sample_rate = 44_100
    duration = 0.5

    audio = render(
        freq=1000.0,
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


def test_tap_count_changes_the_sound() -> None:
    base_kwargs = {
        "freq": 1000.0,
        "duration": 0.3,
        "amp": 0.7,
        "sample_rate": 44_100,
    }

    few_taps = render(**base_kwargs, params={"n_taps": 2})
    many_taps = render(**base_kwargs, params={"n_taps": 6})

    assert not np.allclose(few_taps, many_taps)


def test_multi_tap_structure_visible_in_waveform() -> None:
    """With well-separated taps, the envelope should show multiple peaks."""
    sample_rate = 44_100
    n_taps = 4
    tap_spacing_ms = 8.0

    audio = render(
        freq=1000.0,
        duration=0.3,
        amp=0.9,
        sample_rate=sample_rate,
        params={
            "n_taps": n_taps,
            "tap_spacing_ms": tap_spacing_ms,
            "tap_decay_ms": 2.0,
            "body_decay_ms": 10.0,
            "click_amount": 0.0,
        },
    )

    # Smooth the absolute signal for envelope detection.
    window_samples = int(0.001 * sample_rate)  # 1 ms smoothing
    kernel = np.ones(window_samples) / window_samples
    envelope = np.convolve(np.abs(audio), kernel, mode="same")

    # Check the first ~40ms for distinct peaks.
    analysis_end = int(0.040 * sample_rate)
    env_region = envelope[:analysis_end]

    # Split into windows around expected tap positions and find peaks.
    tap_spacing_samples = int(tap_spacing_ms / 1000.0 * sample_rate)
    half_window = tap_spacing_samples // 2
    peak_count = 0
    for i in range(n_taps):
        center = int(i * tap_spacing_ms / 1000.0 * sample_rate)
        win_start = max(0, center - half_window)
        win_end = min(len(env_region), center + half_window)
        if win_end <= win_start:
            continue
        window = env_region[win_start:win_end]
        local_peak = np.max(window)
        # Each tap should produce a noticeable bump above the local minimum.
        if local_peak > 0.01 * np.max(env_region):
            peak_count += 1

    assert peak_count >= 3, f"Expected at least 3 tap peaks, found {peak_count}"


def test_body_decay_changes_tail() -> None:
    sample_rate = 44_100
    duration = 0.4
    base_kwargs = {
        "freq": 1000.0,
        "duration": duration,
        "amp": 0.7,
        "sample_rate": sample_rate,
    }

    short_tail = render(**base_kwargs, params={"body_decay_ms": 20.0})
    long_tail = render(**base_kwargs, params={"body_decay_ms": 200.0})

    # The second half should have less energy with short decay.
    half = len(short_tail) // 2
    short_tail_energy = np.sum(short_tail[half:] ** 2)
    long_tail_energy = np.sum(long_tail[half:] ** 2)

    assert long_tail_energy > short_tail_energy


def test_render_is_deterministic_for_identical_inputs() -> None:
    kwargs = {
        "freq": 1200.0,
        "duration": 0.3,
        "amp": 0.6,
        "sample_rate": 44_100,
        "params": {
            "n_taps": 5,
            "tap_spacing_ms": 6.0,
            "tap_decay_ms": 2.5,
            "body_decay_ms": 80.0,
            "click_amount": 0.1,
        },
    }

    first = render(**kwargs)
    second = render(**kwargs)

    assert np.allclose(first, second)


def test_clap_with_body_amp_envelope_renders_finite() -> None:
    audio = render(
        freq=1000.0,
        duration=0.4,
        amp=0.8,
        sample_rate=44_100,
        params={
            "body_amp_envelope": [
                {"time": 0.0, "value": 1.0},
                {"time": 0.5, "value": 0.6, "curve": "exponential"},
                {"time": 1.0, "value": 0.0, "curve": "exponential"},
            ],
        },
    )

    assert isinstance(audio, np.ndarray)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_clap_with_overall_amp_envelope_renders_finite() -> None:
    audio = render(
        freq=1000.0,
        duration=0.5,
        amp=0.8,
        sample_rate=44_100,
        params={
            "overall_amp_envelope": [
                {"time": 0.0, "value": 1.0},
                {"time": 0.6, "value": 0.9, "curve": "linear"},
                {"time": 0.65, "value": 0.0, "curve": "exponential"},
            ],
        },
    )

    assert isinstance(audio, np.ndarray)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_clap_default_matches_when_no_envelope() -> None:
    """Omitting envelope params produces deterministic output."""
    kwargs = {
        "freq": 1000.0,
        "duration": 0.3,
        "amp": 0.7,
        "sample_rate": 44_100,
        "params": {
            "n_taps": 4,
            "tap_spacing_ms": 5.0,
            "body_decay_ms": 60.0,
            "click_amount": 0.08,
        },
    }

    first = render(**kwargs)
    second = render(**kwargs)

    assert np.allclose(first, second)


def test_gated_clap_preset_renders() -> None:
    from code_musics.engines.registry import resolve_synth_params

    resolved = resolve_synth_params({"engine": "clap", "preset": "gated_clap"})
    audio = render(
        freq=1000.0,
        duration=0.5,
        amp=0.8,
        sample_rate=44_100,
        params=resolved,
    )

    assert isinstance(audio, np.ndarray)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_tap_acceleration_renders_without_error() -> None:
    audio = render(
        freq=1000.0,
        duration=0.3,
        amp=0.8,
        sample_rate=44_100,
        params={"tap_acceleration": 0.5, "n_taps": 4, "tap_spacing_ms": 6.0},
    )

    assert isinstance(audio, np.ndarray)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_tap_acceleration_differs_from_uniform() -> None:
    base_kwargs = {
        "freq": 1000.0,
        "duration": 0.3,
        "amp": 0.8,
        "sample_rate": 44_100,
    }
    uniform = render(
        **base_kwargs,
        params={"n_taps": 4, "tap_spacing_ms": 6.0, "tap_acceleration": 0.0},
    )
    accelerated = render(
        **base_kwargs,
        params={"n_taps": 4, "tap_spacing_ms": 6.0, "tap_acceleration": 0.6},
    )

    assert not np.allclose(uniform, accelerated)


def test_tap_freq_spread_renders_without_error() -> None:
    audio = render(
        freq=1000.0,
        duration=0.3,
        amp=0.8,
        sample_rate=44_100,
        params={"tap_freq_spread": 0.3},
    )

    assert isinstance(audio, np.ndarray)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_tap_freq_spread_differs_from_no_spread() -> None:
    base_kwargs = {
        "freq": 1000.0,
        "duration": 0.3,
        "amp": 0.8,
        "sample_rate": 44_100,
    }
    no_spread = render(**base_kwargs, params={"tap_freq_spread": 0.0})
    with_spread = render(**base_kwargs, params={"tap_freq_spread": 0.5})

    assert not np.allclose(no_spread, with_spread)


def test_tail_filter_renders_without_error() -> None:
    audio = render(
        freq=1000.0,
        duration=0.4,
        amp=0.8,
        sample_rate=44_100,
        params={"tail_filter_cutoff_hz": 1200.0, "tail_filter_q": 1.5},
    )

    assert isinstance(audio, np.ndarray)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_new_presets_render_successfully() -> None:
    from code_musics.engines.registry import resolve_synth_params

    for preset_name in ("909_clap_authentic", "scattered_clap"):
        resolved = resolve_synth_params({"engine": "clap", "preset": preset_name})
        audio = render(
            freq=1000.0,
            duration=0.5,
            amp=0.8,
            sample_rate=44_100,
            params=resolved,
        )

        assert isinstance(audio, np.ndarray), f"{preset_name} failed type check"
        assert np.isfinite(audio).all(), f"{preset_name} has non-finite values"
        assert np.max(np.abs(audio)) > 0, f"{preset_name} is silent"


def test_invalid_tap_acceleration_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="tap_acceleration"):
        render(
            freq=1000.0,
            duration=0.3,
            amp=0.8,
            sample_rate=44_100,
            params={"tap_acceleration": 1.5},
        )

    with pytest.raises(ValueError, match="tap_acceleration"):
        render(
            freq=1000.0,
            duration=0.3,
            amp=0.8,
            sample_rate=44_100,
            params={"tap_acceleration": -0.1},
        )
