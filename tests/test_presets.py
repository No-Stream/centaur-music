"""Smoke tests for the Diva-inspired polyblep presets.

Each preset is rendered through ``render_note_signal`` at a musical
frequency/duration/amplitude and the output is checked for:

- finite samples (no NaN/Inf leaking out of the solver)
- non-silent output (signal energy present)
- sane peak amplitude (no runaway self-oscillation blowup)
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines.registry import render_note_signal

_DIVA_PRESETS: list[str] = [
    "diva_bass_resonance",
    "cs80_attack",
    "prophet_pad",
    "moog_acid_newton",
    "sk_bite_lead",
    "cascade_bass",
    "sync_screamer",
    "ring_mod_lead",
]


@pytest.mark.parametrize("preset", _DIVA_PRESETS)
def test_diva_preset_renders_finite_audio(preset: str) -> None:
    """Each Diva-inspired polyblep preset renders finite, non-silent audio."""
    signal = render_note_signal(
        freq=220.0,
        duration=0.5,
        amp=0.8,
        sample_rate=44100,
        params={"engine": "polyblep", "preset": preset},
    )
    assert signal.ndim == 1
    assert signal.size > 0
    assert np.all(np.isfinite(signal)), f"{preset} produced non-finite samples"
    assert np.max(np.abs(signal)) > 1e-6, f"{preset} produced silence"
    assert np.max(np.abs(signal)) < 10.0, f"{preset} blew up past safe peak"
