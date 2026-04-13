"""Organ engine tests."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines.organ import render
from code_musics.engines.registry import render_note_signal

SAMPLE_RATE = 44_100
DURATION = 0.3
FREQ = 220.0
AMP = 0.7


def test_organ_via_registry() -> None:
    signal = render_note_signal(
        freq=FREQ,
        duration=DURATION,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={"engine": "organ"},
    )

    n_expected = int(SAMPLE_RATE * DURATION)
    assert signal.shape == (n_expected,)
    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(signal)) > 0.0


def _band_energy(signal: np.ndarray, *, sample_rate: int, min_hz: float) -> float:
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    return float(np.sum(spectrum[freqs >= min_hz]))


def _render_default(**overrides: object) -> np.ndarray:
    kwargs: dict = {
        "freq": FREQ,
        "duration": DURATION,
        "amp": AMP,
        "sample_rate": SAMPLE_RATE,
        "params": {},
    }
    kwargs.update(overrides)
    return render(**kwargs)


def test_organ_render_basic() -> None:
    signal = _render_default()

    n_expected = int(SAMPLE_RATE * DURATION)
    assert signal.shape == (n_expected,)
    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(signal)) > 0.0


def test_organ_render_deterministic() -> None:
    params = {
        "drawbars": [8, 8, 4, 4, 2, 2, 1, 1, 1],
        "click": 0.2,
        "vibrato_depth": 0.15,
        "drift": 0.1,
    }
    first = _render_default(params=params)
    second = _render_default(params=params)

    assert np.allclose(first, second)


def test_drawbars_affect_spectral_content() -> None:
    fundamental_only = _render_default(
        params={"drawbars": [0, 8, 0, 0, 0, 0, 0, 0, 0], "click": 0.0, "drift": 0.0}
    )
    full_registration = _render_default(
        params={"drawbars": [8, 8, 8, 8, 8, 8, 8, 8, 8], "click": 0.0, "drift": 0.0}
    )

    threshold_hz = FREQ * 2.5
    fund_high_energy = _band_energy(
        fundamental_only, sample_rate=SAMPLE_RATE, min_hz=threshold_hz
    )
    full_high_energy = _band_energy(
        full_registration, sample_rate=SAMPLE_RATE, min_hz=threshold_hz
    )

    assert full_high_energy > fund_high_energy


def test_click_adds_onset_energy() -> None:
    shared = {"drift": 0.0, "vibrato_depth": 0.0}
    no_click = _render_default(params={**shared, "click": 0.0})
    with_click = _render_default(params={**shared, "click": 0.8})

    diff = np.abs(with_click - no_click)
    onset_samples = int(0.005 * SAMPLE_RATE)
    onset_diff = float(np.mean(diff[:onset_samples]))
    tail_diff = float(np.mean(diff[onset_samples:]))

    assert onset_diff > tail_diff


@pytest.mark.parametrize(
    ("param", "off", "on"),
    [
        ("vibrato_depth", 0.0, 0.3),
        ("drift", 0.0, 0.2),
        ("leakage", 0.0, 0.3),
    ],
)
def test_param_changes_signal(param: str, off: float, on: float) -> None:
    shared = {"click": 0.0, "drift": 0.0, "vibrato_depth": 0.0}
    without = _render_default(params={**shared, param: off})
    with_param = _render_default(params={**shared, param: on})

    assert not np.allclose(without, with_param)


def test_tonewheel_shape_adds_harmonics() -> None:
    pure = _render_default(params={"tonewheel_shape": 0.0, "click": 0.0, "drift": 0.0})
    shaped = _render_default(
        params={"tonewheel_shape": 0.5, "click": 0.0, "drift": 0.0}
    )

    threshold_hz = FREQ * 2.5
    pure_high = _band_energy(pure, sample_rate=SAMPLE_RATE, min_hz=threshold_hz)
    shaped_high = _band_energy(shaped, sample_rate=SAMPLE_RATE, min_hz=threshold_hz)

    assert shaped_high > pure_high


def test_custom_drawbar_ratios() -> None:
    septimal_ratios = [0.5, 1.0, 7 / 4, 7 / 2, 3 / 2, 7 / 6, 21 / 8, 9 / 4, 7 / 3]
    drawbars = [4, 8, 6, 4, 6, 4, 3, 3, 4]

    septimal = _render_default(
        params={
            "drawbars": drawbars,
            "drawbar_ratios": septimal_ratios,
            "click": 0.0,
            "drift": 0.0,
        }
    )
    hammond = _render_default(params={"drawbars": drawbars, "click": 0.0, "drift": 0.0})

    assert np.all(np.isfinite(septimal))
    assert np.max(np.abs(septimal)) > 0.0
    assert not np.allclose(septimal, hammond)


def test_freq_trajectory_support() -> None:
    n_samples = int(SAMPLE_RATE * DURATION)
    sweep = np.linspace(FREQ, FREQ * 1.5, n_samples)

    with_sweep = render(
        freq=FREQ,
        duration=DURATION,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={"click": 0.0, "drift": 0.0, "vibrato_depth": 0.0},
        freq_trajectory=sweep,
    )
    static = _render_default(params={"click": 0.0, "drift": 0.0, "vibrato_depth": 0.0})

    assert np.all(np.isfinite(with_sweep))
    assert not np.allclose(with_sweep, static)


def test_mismatched_drawbar_lengths_raises() -> None:
    with pytest.raises(ValueError, match="drawbars length.*must match"):
        _render_default(params={"drawbars": [8, 8]})


def test_all_zero_drawbars_raises() -> None:
    with pytest.raises(ValueError, match="at least one drawbar must be nonzero"):
        _render_default(params={"drawbars": [0, 0, 0, 0, 0, 0, 0, 0, 0]})


def test_negative_freq_raises() -> None:
    with pytest.raises(ValueError, match="freq must be positive"):
        render(
            freq=-1.0,
            duration=DURATION,
            amp=AMP,
            sample_rate=SAMPLE_RATE,
            params={},
        )
