"""FM engine tests."""

from __future__ import annotations

import numpy as np

from code_musics.engines.fm import render


def test_fm_render_returns_expected_length_and_finite_signal() -> None:
    signal = render(
        freq=110.0,
        duration=0.25,
        amp=0.8,
        sample_rate=44100,
        params={
            "carrier_ratio": 1.0,
            "mod_ratio": 2.0,
            "mod_index": 2.5,
        },
    )

    assert isinstance(signal, np.ndarray)
    assert signal.ndim == 1
    assert len(signal) == int(0.25 * 44100)
    assert np.isfinite(signal).all()
    assert np.max(np.abs(signal)) > 0.0


def test_fm_parameter_changes_affect_output() -> None:
    base_signal = render(
        freq=110.0,
        duration=0.25,
        amp=0.8,
        sample_rate=44100,
        params={
            "carrier_ratio": 1.0,
            "mod_ratio": 2.0,
            "mod_index": 1.5,
        },
    )
    altered_signal = render(
        freq=110.0,
        duration=0.25,
        amp=0.8,
        sample_rate=44100,
        params={
            "carrier_ratio": 1.0,
            "mod_ratio": 3.0,
            "mod_index": 4.0,
        },
    )

    assert not np.allclose(base_signal, altered_signal)


def test_fm_feedback_and_decay_produce_valid_audio() -> None:
    signal = render(
        freq=220.0,
        duration=0.4,
        amp=0.5,
        sample_rate=32000,
        params={
            "carrier_ratio": 1.0,
            "mod_ratio": 7 / 4,
            "mod_index": 3.0,
            "feedback": 0.15,
            "index_decay": 0.08,
            "index_sustain": 0.35,
        },
    )

    assert len(signal) == int(0.4 * 32000)
    assert np.isfinite(signal).all()
    assert np.max(np.abs(signal)) > 0.0
