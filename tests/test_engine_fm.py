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


def test_fm_render_matches_mathematical_reference() -> None:
    """Verify the JIT engine output matches a pure-numpy reference implementation."""
    sr = 44100
    dur = 0.05
    freq = 220.0
    amp = 0.7
    mod_ratio = 2.0
    mod_index = 2.0
    n_samples = int(sr * dur)

    signal = render(
        freq=freq,
        duration=dur,
        amp=amp,
        sample_rate=sr,
        params={
            "carrier_ratio": 1.0,
            "mod_ratio": mod_ratio,
            "mod_index": mod_index,
            "feedback": 0.0,
            "index_decay": 0.0,
            "index_sustain": 1.0,
        },
    )

    # Reference: sin(carrier_phase + mod_index * sin(mod_phase)), normalized
    carrier_inc = 2.0 * np.pi * freq / sr
    mod_inc = 2.0 * np.pi * freq * mod_ratio / sr
    carrier_phase = np.cumsum(np.full(n_samples, carrier_inc))
    mod_phase = np.cumsum(np.full(n_samples, mod_inc))
    # Engine starts at phase 0 and increments AFTER writing, so shift by one
    carrier_phase = np.concatenate([[0.0], carrier_phase[:-1]])
    mod_phase = np.concatenate([[0.0], mod_phase[:-1]])
    reference = np.sin(carrier_phase + mod_index * np.sin(mod_phase))
    peak = np.max(np.abs(reference))
    if peak > 0:
        reference = reference / peak
    reference = amp * reference

    np.testing.assert_allclose(signal, reference, atol=1e-10)
