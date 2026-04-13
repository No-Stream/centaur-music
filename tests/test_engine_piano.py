"""Piano engine tests."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines.piano import render
from code_musics.engines.registry import render_note_signal

SAMPLE_RATE = 44_100
DURATION = 0.3
FREQ = 220.0
AMP = 0.7


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


def test_piano_render_basic() -> None:
    signal = _render_default()

    n_expected = int(SAMPLE_RATE * DURATION)
    assert signal.shape == (n_expected,)
    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(signal)) > 0.0


def test_piano_render_deterministic() -> None:
    params = {
        "n_modes": 24,
        "inharmonicity": 0.0003,
        "hammer_stiffness": 1e8,
        "drift": 0.1,
    }
    first = _render_default(params=params)
    second = _render_default(params=params)

    assert np.allclose(first, second)


def test_inharmonicity_stretches_partials() -> None:
    harmonic = _render_default(
        params={"inharmonicity": 0.0, "hammer_noise": 0.0, "drift": 0.0}
    )
    inharmonic = _render_default(
        params={"inharmonicity": 0.001, "hammer_noise": 0.0, "drift": 0.0}
    )

    assert not np.allclose(harmonic, inharmonic)


def test_hammer_stiffness_affects_brightness() -> None:
    soft = _render_default(params={"hammer_stiffness": 1e7, "drift": 0.0})
    hard = _render_default(params={"hammer_stiffness": 5e8, "drift": 0.0})

    assert not np.allclose(soft, hard)


def test_physical_hammer_produces_onset_energy() -> None:
    signal = _render_default(params={"drift": 0.0})

    onset_samples = int(0.02 * SAMPLE_RATE)
    onset_rms = float(np.sqrt(np.mean(signal[:onset_samples] ** 2)))

    assert onset_rms > 0.0, "physical hammer should produce onset energy"


def test_unison_changes_signal() -> None:
    mono = _render_default(
        params={"unison_count": 1, "drift": 0.0, "hammer_noise": 0.0}
    )
    triple = _render_default(
        params={
            "unison_count": 3,
            "unison_detune": 3.0,
            "drift": 0.0,
            "hammer_noise": 0.0,
        }
    )

    assert not np.allclose(mono, triple)


def test_soundboard_changes_signal() -> None:
    no_board = _render_default(
        params={"soundboard_color": 0.0, "drift": 0.0, "hammer_noise": 0.0}
    )
    with_board = _render_default(
        params={"soundboard_color": 0.8, "drift": 0.0, "hammer_noise": 0.0}
    )

    assert not np.allclose(no_board, with_board)


def test_custom_partial_ratios() -> None:
    septimal_ratios = [
        {"ratio": 1.0, "amp": 1.0},
        {"ratio": 7 / 4, "amp": 0.6},
        {"ratio": 3 / 2, "amp": 0.5},
    ]
    custom = _render_default(
        params={"partial_ratios": septimal_ratios, "drift": 0.0, "hammer_noise": 0.0}
    )
    default = _render_default(params={"drift": 0.0, "hammer_noise": 0.0})

    assert np.all(np.isfinite(custom))
    assert np.max(np.abs(custom)) > 0.0
    assert not np.allclose(custom, default)


def test_freq_trajectory_support() -> None:
    n_samples = int(SAMPLE_RATE * DURATION)
    sweep = np.linspace(FREQ, FREQ * 1.5, n_samples)

    with_sweep = render(
        freq=FREQ,
        duration=DURATION,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={"drift": 0.0, "hammer_noise": 0.0},
        freq_trajectory=sweep,
    )
    static = _render_default(params={"drift": 0.0, "hammer_noise": 0.0})

    assert np.all(np.isfinite(with_sweep))
    assert not np.allclose(with_sweep, static)


def test_decay_tail_quieter_than_onset() -> None:
    signal = render(
        freq=FREQ,
        duration=1.0,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={"decay_base": 2.0},
    )

    n_total = signal.size
    onset_end = int(0.2 * n_total)
    tail_start = int(0.8 * n_total)

    onset_rms = float(np.sqrt(np.mean(signal[:onset_end] ** 2)))
    tail_rms = float(np.sqrt(np.mean(signal[tail_start:] ** 2)))

    assert tail_rms < onset_rms


def test_damper_thump_on_short_notes() -> None:
    no_damper = render(
        freq=FREQ,
        duration=0.15,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={"damper_noise": 0.0, "drift": 0.0},
    )
    with_damper = render(
        freq=FREQ,
        duration=0.15,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={"damper_noise": 0.3, "drift": 0.0},
    )

    tail_start = int(0.7 * no_damper.size)
    tail_diff = float(
        np.mean(np.abs(with_damper[tail_start:] - no_damper[tail_start:]))
    )

    assert tail_diff > 0.0


def test_negative_freq_raises() -> None:
    with pytest.raises(ValueError, match="freq must be positive"):
        render(
            freq=-1.0,
            duration=DURATION,
            amp=AMP,
            sample_rate=SAMPLE_RATE,
            params={},
        )


def test_invalid_inharmonicity_raises() -> None:
    with pytest.raises(ValueError, match="inharmonicity must be non-negative"):
        _render_default(params={"inharmonicity": -0.01})


def test_zero_duration_returns_empty() -> None:
    signal = render(
        freq=FREQ,
        duration=0.0,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={},
    )
    assert signal.shape == (0,)


def test_piano_via_registry() -> None:
    signal = render_note_signal(
        freq=FREQ,
        duration=DURATION,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={"engine": "piano"},
    )

    n_expected = int(SAMPLE_RATE * DURATION)
    assert signal.shape == (n_expected,)
    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(signal)) > 0.0


def test_velocity_changes_signal() -> None:
    """Modal hammer at different velocities should produce different waveform shapes."""
    shared_params = {
        "drift": 0.0,
        "unison_count": 1,
        "body_saturation": 0.0,
        "soundboard_color": 0.0,
        "damper_noise": 0.0,
    }
    soft = render(
        freq=FREQ, duration=0.3, amp=0.15, sample_rate=SAMPLE_RATE, params=shared_params
    )
    loud = render(
        freq=FREQ, duration=0.3, amp=0.9, sample_rate=SAMPLE_RATE, params=shared_params
    )

    soft_norm = soft / np.max(np.abs(soft))
    loud_norm = loud / np.max(np.abs(loud))
    assert not np.array_equal(soft_norm, loud_norm), (
        "velocity should change waveform shape, not just amplitude"
    )


def test_piano_preset_grand() -> None:
    signal = render_note_signal(
        freq=FREQ,
        duration=DURATION,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={"engine": "piano", "preset": "grand"},
    )

    n_expected = int(SAMPLE_RATE * DURATION)
    assert signal.shape == (n_expected,)
    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(signal)) > 0.0
