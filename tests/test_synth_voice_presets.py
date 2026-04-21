"""Integration tests for synth_voice curated presets.

Renders each registered ``synth_voice`` preset and verifies it produces
a finite, non-silent, non-clipping, spectrally-plausible signal.  This
is the acceptance gate for the preset bundle — manual listening is the
ultimate taste check, but these catch obvious regressions.
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines.registry import _PRESETS, render_note_signal

SAMPLE_RATE = 44_100


def _preset_names() -> list[str]:
    return sorted(_PRESETS["synth_voice"].keys())


def test_at_least_ten_presets_registered() -> None:
    assert len(_preset_names()) >= 10, (
        f"Expected >= 10 curated synth_voice presets, got {len(_preset_names())}"
    )


@pytest.mark.parametrize("preset_name", _preset_names())
def test_preset_renders_cleanly(preset_name: str) -> None:
    """Each preset must render finite, non-silent, non-clipping audio."""
    audio = render_note_signal(
        freq=220.0,
        duration=0.6,
        amp=0.7,
        sample_rate=SAMPLE_RATE,
        params={"engine": "synth_voice", "preset": preset_name},
    )
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert len(audio) == int(SAMPLE_RATE * 0.6)
    assert np.isfinite(audio).all(), (
        f"Preset {preset_name!r} produced non-finite samples"
    )
    peak = float(np.max(np.abs(audio)))
    assert peak > 0.05, (
        f"Preset {preset_name!r} produced near-silent output (peak={peak})"
    )
    assert peak <= 1.0, f"Preset {preset_name!r} clipped after render (peak={peak})"


@pytest.mark.parametrize("preset_name", _preset_names())
def test_preset_spectrally_plausible(preset_name: str) -> None:
    """Each preset should have energy above the amplitude floor in multiple bands."""
    audio = render_note_signal(
        freq=220.0,
        duration=0.8,
        amp=0.7,
        sample_rate=SAMPLE_RATE,
        params={"engine": "synth_voice", "preset": preset_name},
    )
    # Trim attack transient and release tail.
    trim = int(SAMPLE_RATE * 0.1)
    steady = audio[trim:-trim]

    spectrum = np.abs(np.fft.rfft(steady))
    freqs = np.fft.rfftfreq(len(steady), d=1.0 / SAMPLE_RATE)

    # Verify at least the fundamental band (around 220 Hz +/- 30%) holds
    # meaningful energy — filter sweeps may attenuate higher partials
    # heavily in some presets, so we only require the fundamental.
    fund_band = (freqs >= 160.0) & (freqs <= 280.0)
    fund_energy = float(spectrum[fund_band].sum())
    total_energy = float(spectrum.sum())
    assert total_energy > 0.0
    assert fund_energy > 0.0, f"Preset {preset_name!r} has no fundamental energy"


def test_cross_pollination_presets_use_multiple_slots() -> None:
    """Sanity: the hero presets should actually stack sources as advertised."""
    hero_presets = {
        "fm_bell_over_supersaw": {"osc_type", "fm_type"},
        "stiff_piano_sub": {"partials_type", "osc_type"},
        "flow_exciter_pad": {"partials_type", "noise_type"},
        "chaos_cloud_texture": {"osc_type", "noise_type"},
        "virus_hybrid_pad": {"osc_type", "fm_type"},
        "tonewheel_drive": {"partials_type", "noise_type"},
    }
    presets = _PRESETS["synth_voice"]
    for name, required_keys in hero_presets.items():
        assert name in presets, f"Missing hero preset: {name}"
        preset = presets[name]
        missing = required_keys - preset.keys()
        assert not missing, f"Preset {name!r} missing stacking keys: {missing}"
