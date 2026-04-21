"""Tests for drum_voice preset migration — all 63 old-engine presets translated.

Verifies every migrated preset renders successfully, and spot-checks representative
presets from each original engine via A/B comparison (both produce finite, non-zero
audio of similar length).
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines.clap import render as clap_render
from code_musics.engines.drum_voice import render as dv_render
from code_musics.engines.kick_tom import render as kick_tom_render
from code_musics.engines.metallic_perc import render as metallic_render
from code_musics.engines.noise_perc import render as noise_perc_render
from code_musics.engines.registry import _PRESETS, resolve_synth_params
from code_musics.engines.snare import render as snare_render

SAMPLE_RATE = 44_100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_drum_voice(
    preset_name: str,
    freq: float = 50.0,
    duration: float = 0.3,
    amp: float = 0.8,
) -> np.ndarray:
    """Render a drum_voice preset through the registry."""
    resolved = resolve_synth_params({"engine": "drum_voice", "preset": preset_name})
    return dv_render(
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=SAMPLE_RATE,
        params=resolved,
    )


def _render_old_engine(
    engine: str,
    preset_name: str,
    freq: float = 50.0,
    duration: float = 0.3,
    amp: float = 0.8,
) -> np.ndarray:
    """Render an old-engine preset through the registry."""
    resolved = resolve_synth_params({"engine": engine, "preset": preset_name})
    renderers = {
        "kick_tom": kick_tom_render,
        "snare": snare_render,
        "clap": clap_render,
        "metallic_perc": metallic_render,
        "noise_perc": noise_perc_render,
    }
    return renderers[engine](
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=SAMPLE_RATE,
        params=resolved,
    )


# ---------------------------------------------------------------------------
# 1. Every migrated preset renders without error (parametrized)
# ---------------------------------------------------------------------------

ALL_DRUM_VOICE_PRESETS = sorted(_PRESETS.get("drum_voice", {}).keys())


@pytest.mark.parametrize("preset_name", ALL_DRUM_VOICE_PRESETS)
def test_preset_renders_finite_nonzero(preset_name: str) -> None:
    """Each drum_voice preset must produce finite, non-zero audio."""
    # Use appropriate freq for the preset type
    if any(
        tag in preset_name
        for tag in ("hat", "ride", "cowbell", "clave", "gamelan", "bell")
    ):
        freq = 8000.0
    elif any(tag in preset_name for tag in ("snare", "rim", "brush", "fm_tom")):
        freq = 200.0
    elif any(tag in preset_name for tag in ("clap", "snap", "burst", "cascade")):
        freq = 3000.0
    elif any(
        tag in preset_name
        for tag in ("kickish", "snareish", "tick", "chh", "shaped_hit")
    ):
        freq = 200.0 if "kickish" in preset_name else 3000.0
    else:
        freq = 50.0

    audio = _render_drum_voice(preset_name, freq=freq)
    assert audio.dtype == np.float64, f"dtype mismatch for {preset_name}"
    assert np.isfinite(audio).all(), f"non-finite values in {preset_name}"
    assert np.max(np.abs(audio)) > 0, f"silent output for {preset_name}"


# ---------------------------------------------------------------------------
# 2. A/B spot-checks: old engine vs drum_voice for representative presets
# ---------------------------------------------------------------------------


class TestKickTomAB:
    """A/B for kick_tom -> drum_voice migration."""

    @pytest.mark.parametrize(
        "preset_name",
        ["808_hiphop", "909_techno", "round_tom"],
    )
    def test_both_render_successfully(self, preset_name: str) -> None:
        old = _render_old_engine("kick_tom", preset_name, freq=50.0)
        new = _render_drum_voice(preset_name, freq=50.0)
        assert np.isfinite(old).all()
        assert np.isfinite(new).all()
        assert np.max(np.abs(old)) > 0
        assert np.max(np.abs(new)) > 0
        assert len(old) == len(new)

    def test_resonator_preset(self) -> None:
        old = _render_old_engine("kick_tom", "melodic_resonator", freq=80.0)
        new = _render_drum_voice("melodic_resonator", freq=80.0)
        assert np.isfinite(old).all()
        assert np.isfinite(new).all()
        assert np.max(np.abs(old)) > 0
        assert np.max(np.abs(new)) > 0

    def test_fm_preset(self) -> None:
        old = _render_old_engine("kick_tom", "fm_body_kick", freq=50.0)
        new = _render_drum_voice("fm_body_kick", freq=50.0)
        assert np.isfinite(old).all()
        assert np.isfinite(new).all()
        assert np.max(np.abs(old)) > 0
        assert np.max(np.abs(new)) > 0


class TestSnareAB:
    """A/B for snare -> drum_voice migration."""

    @pytest.mark.parametrize(
        "preset_name",
        ["909_tight", "brush", "fm_snare"],
    )
    def test_both_render_successfully(self, preset_name: str) -> None:
        old = _render_old_engine("snare", preset_name, freq=200.0)
        new = _render_drum_voice(preset_name, freq=200.0)
        assert np.isfinite(old).all()
        assert np.isfinite(new).all()
        assert np.max(np.abs(old)) > 0
        assert np.max(np.abs(new)) > 0
        assert len(old) == len(new)


class TestClapAB:
    """A/B for clap -> drum_voice migration."""

    @pytest.mark.parametrize(
        "preset_name",
        ["909_clap", "finger_snap", "granular_cascade"],
    )
    def test_both_render_successfully(self, preset_name: str) -> None:
        old = _render_old_engine("clap", preset_name, freq=3000.0)
        new = _render_drum_voice(preset_name, freq=3000.0)
        assert np.isfinite(old).all()
        assert np.isfinite(new).all()
        assert np.max(np.abs(old)) > 0
        assert np.max(np.abs(new)) > 0
        assert len(old) == len(new)


class TestMetallicPercAB:
    """A/B for metallic_perc -> drum_voice migration."""

    @pytest.mark.parametrize(
        "preset_name",
        ["closed_hat", "ride_bell", "cowbell"],
    )
    def test_both_render_successfully(self, preset_name: str) -> None:
        old = _render_old_engine("metallic_perc", preset_name, freq=8000.0)
        new = _render_drum_voice(preset_name, freq=8000.0)
        assert np.isfinite(old).all()
        assert np.isfinite(new).all()
        assert np.max(np.abs(old)) > 0
        assert np.max(np.abs(new)) > 0
        assert len(old) == len(new)


class TestNoisePercAB:
    """A/B for noise_perc -> drum_voice migration."""

    @pytest.mark.parametrize(
        "preset_name",
        ["kickish", "snareish", "chh"],
    )
    def test_both_render_successfully(self, preset_name: str) -> None:
        if preset_name == "chh":
            freq = 9000.0
        elif preset_name == "kickish":
            freq = 50.0
        else:
            freq = 200.0
        old = _render_old_engine("noise_perc", preset_name, freq=freq)
        new = _render_drum_voice(preset_name, freq=freq)
        assert np.isfinite(old).all()
        assert np.isfinite(new).all()
        assert np.max(np.abs(old)) > 0
        assert np.max(np.abs(new)) > 0
        assert len(old) == len(new)

    def test_clap_renamed_to_clap_noise(self) -> None:
        """noise_perc 'clap' was renamed to 'clap_noise' to avoid collision."""
        old = _render_old_engine("noise_perc", "clap", freq=3000.0)
        new = _render_drum_voice("clap_noise", freq=3000.0)
        assert np.isfinite(old).all()
        assert np.isfinite(new).all()
        assert np.max(np.abs(old)) > 0
        assert np.max(np.abs(new)) > 0
        assert len(old) == len(new)
