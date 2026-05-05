"""Granular synthesis tests — primitive + synth_voice integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from code_musics.engines._granular import render_granular
from code_musics.engines.synth_voice import render as render_voice

SR = 44_100


# ---------------------------------------------------------------------------
# Primitive — coverage across all source × mode combos
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grain_type", ["cloud", "time_freeze", "texture"])
@pytest.mark.parametrize("grain_source", ["osc", "partials", "fm", "noise"])
def test_render_granular_produces_finite_output(
    grain_type: str, grain_source: str
) -> None:
    n = int(SR * 0.5)
    out = render_granular(
        n_samples=n,
        freq=220.0,
        sample_rate=SR,
        seed=42,
        grain_type=grain_type,
        grain_source=grain_source,
        params={
            "grain_density": 25.0,
            "grain_size_ms": 40.0,
            "grain_jitter": 0.3,
        },
    )
    assert out.shape == (n,)
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) > 0


def test_render_granular_deterministic_for_same_seed() -> None:
    n = int(SR * 0.3)
    a = render_granular(
        n_samples=n,
        freq=220.0,
        sample_rate=SR,
        seed=7,
        grain_type="cloud",
        grain_source="partials",
        params={"grain_density": 30.0, "grain_size_ms": 40.0},
    )
    b = render_granular(
        n_samples=n,
        freq=220.0,
        sample_rate=SR,
        seed=7,
        grain_type="cloud",
        grain_source="partials",
        params={"grain_density": 30.0, "grain_size_ms": 40.0},
    )
    assert np.array_equal(a, b)


def test_render_granular_different_seeds_differ() -> None:
    n = int(SR * 0.3)
    a = render_granular(
        n_samples=n,
        freq=220.0,
        sample_rate=SR,
        seed=1,
        grain_type="cloud",
        grain_source="partials",
        params={"grain_density": 30.0},
    )
    b = render_granular(
        n_samples=n,
        freq=220.0,
        sample_rate=SR,
        seed=2,
        grain_type="cloud",
        grain_source="partials",
        params={"grain_density": 30.0},
    )
    assert not np.array_equal(a, b)


def test_render_granular_rejects_bad_grain_type() -> None:
    with pytest.raises(ValueError, match="grain_type"):
        render_granular(
            n_samples=1024,
            freq=220.0,
            sample_rate=SR,
            seed=0,
            grain_type="totally_fake",
            grain_source="partials",
            params={},
        )


def test_render_granular_rejects_bad_source() -> None:
    with pytest.raises(ValueError, match="grain_source"):
        render_granular(
            n_samples=1024,
            freq=220.0,
            sample_rate=SR,
            seed=0,
            grain_type="cloud",
            grain_source="uh_no",
            params={},
        )


def test_render_granular_rejects_negative_size() -> None:
    with pytest.raises(ValueError, match="grain_size_ms"):
        render_granular(
            n_samples=1024,
            freq=220.0,
            sample_rate=SR,
            seed=0,
            grain_type="cloud",
            grain_source="noise",
            params={"grain_size_ms": -1.0},
        )


def test_render_granular_zero_density_produces_silence() -> None:
    out = render_granular(
        n_samples=4410,
        freq=220.0,
        sample_rate=SR,
        seed=0,
        grain_type="cloud",
        grain_source="noise",
        params={"grain_density": 0.0},
    )
    assert np.max(np.abs(out)) == 0.0


def test_render_granular_ji_lattice_quantization() -> None:
    """Providing grain_ji_lattice must not break rendering."""
    n = int(SR * 0.3)
    out = render_granular(
        n_samples=n,
        freq=220.0,
        sample_rate=SR,
        seed=0,
        grain_type="cloud",
        grain_source="partials",
        params={
            "grain_density": 30.0,
            "grain_pitch_spread": 1.0,
            "grain_ji_lattice": [1.0, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 7 / 4, 2.0],
        },
    )
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) > 0


def test_render_granular_time_freeze_reads_narrow_window() -> None:
    """time_freeze mode should not fail when the source buffer is short."""
    n = int(SR * 1.0)
    out = render_granular(
        n_samples=n,
        freq=330.0,
        sample_rate=SR,
        seed=0,
        grain_type="time_freeze",
        grain_source="osc",
        params={"grain_density": 40.0, "grain_window_start": 0.3},
    )
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) > 0


# ---------------------------------------------------------------------------
# Sample source mode
# ---------------------------------------------------------------------------


def test_render_granular_sample_source_requires_path() -> None:
    with pytest.raises(ValueError, match="grain_sample_path"):
        render_granular(
            n_samples=4410,
            freq=220.0,
            sample_rate=SR,
            seed=0,
            grain_type="cloud",
            grain_source="sample",
            params={"grain_density": 20.0},
        )


def test_render_granular_sample_source_renders_from_wav(tmp_path: Path) -> None:
    """Write a tiny sine burst WAV and granulate it through the sample source."""
    tone_duration_s = 1.0
    t = np.arange(int(SR * tone_duration_s), dtype=np.float64) / SR
    tone = 0.3 * np.sin(2.0 * np.pi * 220.0 * t)
    sample_path = tmp_path / "tone.wav"
    sf.write(sample_path, tone, SR, subtype="PCM_24")

    n = int(SR * 0.3)
    out = render_granular(
        n_samples=n,
        freq=220.0,
        sample_rate=SR,
        seed=0,
        grain_type="cloud",
        grain_source="sample",
        params={
            "grain_density": 25.0,
            "grain_size_ms": 40.0,
            "grain_sample_path": str(sample_path),
        },
    )
    assert out.shape == (n,)
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) > 0


# ---------------------------------------------------------------------------
# synth_voice integration
# ---------------------------------------------------------------------------


def test_synth_voice_grain_slot_renders() -> None:
    audio = render_voice(
        freq=220.0,
        duration=0.5,
        amp=0.6,
        sample_rate=SR,
        params={
            "grain_type": "cloud",
            "grain_source": "partials",
            "grain_density": 30.0,
            "grain_size_ms": 50.0,
            "filter_mode": "lowpass",
            "filter_cutoff_hz": 3000.0,
        },
    )
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) > 0
    assert len(audio) == int(SR * 0.5)


def test_synth_voice_grain_slot_off_state_preserves_silence() -> None:
    """No grain_type → grain slot never fires."""
    audio = render_voice(
        freq=220.0,
        duration=0.2,
        amp=0.6,
        sample_rate=SR,
        params={},
    )
    assert np.max(np.abs(audio)) == 0.0


def test_synth_voice_grain_slot_stacks_with_osc() -> None:
    """Grain + osc slots should sum pre-filter."""
    audio = render_voice(
        freq=220.0,
        duration=0.3,
        amp=0.6,
        sample_rate=SR,
        params={
            "osc_type": "polyblep",
            "osc_wave": "saw",
            "osc_level": 0.5,
            "grain_type": "cloud",
            "grain_source": "noise",
            "grain_level": 0.3,
            "grain_density": 20.0,
            "filter_mode": "lowpass",
            "filter_cutoff_hz": 2200.0,
        },
    )
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) > 0


def test_synth_voice_grain_presets_load() -> None:
    from code_musics.engines.registry import render_note_signal

    for preset in (
        "grain_breathing_cloud",
        "grain_frozen_time",
        "grain_glitch_scatter",
        "grain_tape_dust",
        "grain_shimmer_dust",
    ):
        audio = render_note_signal(
            freq=220.0,
            duration=0.3,
            amp=0.6,
            sample_rate=SR,
            params={"engine": "synth_voice", "preset": preset},
        )
        assert isinstance(audio, np.ndarray)
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) > 0
