"""Filtered-stack engine tests."""

from __future__ import annotations

import numpy as np

from code_musics.engines.filtered_stack import render


def test_filtered_stack_renders_expected_length_and_finite_audio() -> None:
    audio = render(
        freq=110.0,
        duration=1.5,
        amp=0.4,
        sample_rate=44100,
        params={
            "waveform": "saw",
            "n_harmonics": 12,
            "cutoff_ratio": 8.0,
            "resonance": 0.1,
            "filter_env_amount": 1.5,
            "filter_env_decay": 0.25,
        },
    )

    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert len(audio) == int(1.5 * 44100)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.0


def test_filtered_stack_cutoff_changes_the_sound() -> None:
    low_cutoff = render(
        freq=110.0,
        duration=1.0,
        amp=0.4,
        sample_rate=44100,
        params={
            "waveform": "saw",
            "n_harmonics": 16,
            "cutoff_ratio": 2.0,
            "resonance": 0.0,
        },
    )
    high_cutoff = render(
        freq=110.0,
        duration=1.0,
        amp=0.4,
        sample_rate=44100,
        params={
            "waveform": "saw",
            "n_harmonics": 16,
            "cutoff_ratio": 14.0,
            "resonance": 0.0,
        },
    )

    assert not np.allclose(low_cutoff, high_cutoff)
    assert np.linalg.norm(low_cutoff - high_cutoff) > 1.0


def test_filtered_stack_waveform_specific_params_work() -> None:
    audio = render(
        freq=220.0,
        duration=0.75,
        amp=0.25,
        sample_rate=44100,
        params={
            "waveform": "pulse",
            "pulse_width": 0.3,
            "n_harmonics": 10,
            "cutoff_ratio": 6.0,
            "filter_env_amount": 0.5,
            "filter_env_decay": 0.12,
        },
    )

    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.0
