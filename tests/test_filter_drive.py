"""Behavioral regression tests for shared subtractive filter drive."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from code_musics.engines.filtered_stack import render as render_filtered_stack
from code_musics.engines.polyblep import render as render_polyblep

EngineRenderer = Callable[..., np.ndarray]


def _diff_rms(clean: np.ndarray, candidate: np.ndarray) -> float:
    """Return RMS distance between two equal-length signals."""
    return float(np.sqrt(np.mean(np.square(candidate - clean))))


@pytest.mark.parametrize(
    ("engine_name", "render_fn", "base_params"),
    [
        (
            "polyblep",
            render_polyblep,
            {
                "waveform": "saw",
                "cutoff_hz": 900.0,
                "resonance_q": 3.53,
                "filter_mode": "lowpass",
            },
        ),
        (
            "filtered_stack",
            render_filtered_stack,
            {
                "waveform": "saw",
                "n_harmonics": 16,
                "cutoff_hz": 900.0,
                "resonance_q": 3.53,
                "filter_mode": "lowpass",
            },
        ),
    ],
)
def test_filter_drive_progression_feels_gradual(
    engine_name: str,
    render_fn: EngineRenderer,
    base_params: dict[str, Any],
) -> None:
    clean = render_fn(
        freq=110.0,
        duration=0.5,
        amp=0.8,
        sample_rate=44100,
        params={**base_params, "filter_drive": 0.0},
    )
    drive_levels = [0.01, 0.1, 0.3, 0.5, 1.0]
    differences = [
        _diff_rms(
            clean,
            render_fn(
                freq=110.0,
                duration=0.5,
                amp=0.8,
                sample_rate=44100,
                params={**base_params, "filter_drive": drive},
            ),
        )
        for drive in drive_levels
    ]

    assert differences == sorted(differences), engine_name
    assert differences[0] < 0.002, engine_name
    assert differences[1] < 0.02, engine_name
    assert differences[2] > 0.02, engine_name
    assert differences[3] > 0.06, engine_name
    assert differences[4] > 0.2, engine_name


@pytest.mark.parametrize(
    ("engine_name", "render_fn", "base_params"),
    [
        (
            "polyblep",
            render_polyblep,
            {
                "waveform": "saw",
                "cutoff_hz": 900.0,
                "resonance_q": 3.53,
                "filter_mode": "lowpass",
            },
        ),
        (
            "filtered_stack",
            render_filtered_stack,
            {
                "waveform": "saw",
                "n_harmonics": 16,
                "cutoff_hz": 900.0,
                "resonance_q": 3.53,
                "filter_mode": "lowpass",
            },
        ),
    ],
)
def test_tiny_filter_drive_keeps_high_similarity_to_clean_signal(
    engine_name: str,
    render_fn: EngineRenderer,
    base_params: dict[str, Any],
) -> None:
    clean = render_fn(
        freq=110.0,
        duration=0.5,
        amp=0.8,
        sample_rate=44100,
        params={**base_params, "filter_drive": 0.0},
    )
    lightly_driven = render_fn(
        freq=110.0,
        duration=0.5,
        amp=0.8,
        sample_rate=44100,
        params={**base_params, "filter_drive": 0.01},
    )

    correlation = float(np.corrcoef(clean, lightly_driven)[0, 1])
    assert correlation > 0.999, engine_name
