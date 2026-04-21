"""Smoke tests for creative drum presets."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines.registry import render_note_signal

_CREATIVE_PRESETS = [
    ("kick_tom", "melodic_resonator"),
    ("kick_tom", "kick_bell"),
    ("metallic_perc", "harmonic_bell"),
    ("metallic_perc", "septimal_bell"),
    ("metallic_perc", "square_gamelan"),
    ("snare", "fm_tom"),
    ("snare", "fm_noise_burst"),
    ("metallic_perc", "beating_hat_a"),
    ("metallic_perc", "beating_hat_b"),
    ("metallic_perc", "beating_hat_c"),
    ("clap", "granular_cascade"),
    ("clap", "micro_burst"),
]


@pytest.mark.parametrize("engine,preset", _CREATIVE_PRESETS)
def test_creative_preset_renders(engine: str, preset: str) -> None:
    """Each creative preset renders finite, non-silent audio."""
    signal = render_note_signal(
        freq=120.0,
        duration=0.5,
        amp=0.8,
        sample_rate=44100,
        params={"engine": engine, "preset": preset},
    )
    assert signal.ndim == 1
    assert signal.size > 0
    assert np.all(np.isfinite(signal)), f"{engine}/{preset} produced non-finite samples"
    assert np.max(np.abs(signal)) > 1e-6, f"{engine}/{preset} produced silence"
