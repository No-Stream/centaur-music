"""Tests for the drum_voice ergonomic macro system."""

from __future__ import annotations

import numpy as np

from code_musics.engines._drum_macros import resolve_macros
from code_musics.engines.drum_voice import render

SAMPLE_RATE = 44_100


# ---------------------------------------------------------------------------
# 1. punch macro
# ---------------------------------------------------------------------------


class TestPunchMacro:
    def test_punch_zero_maps_to_low_exciter(self) -> None:
        params = resolve_macros({"punch": 0.0})
        assert abs(params["exciter_level"] - 0.01) < 1e-6
        assert abs(params["exciter_decay_s"] - 0.012) < 1e-6
        assert abs(params["exciter_center_hz"] - 1500.0) < 1e-3
        assert abs(params["tone_punch"] - 0.0) < 1e-6

    def test_punch_one_maps_to_high_exciter(self) -> None:
        params = resolve_macros({"punch": 1.0})
        assert abs(params["exciter_level"] - 0.25) < 1e-6
        assert abs(params["exciter_decay_s"] - 0.003) < 1e-6
        assert abs(params["exciter_center_hz"] - 5000.0) < 1e-3
        assert abs(params["tone_punch"] - 0.35) < 1e-6

    def test_punch_midpoint_interpolates(self) -> None:
        params = resolve_macros({"punch": 0.5})
        assert 0.01 < params["exciter_level"] < 0.25
        assert 0.003 < params["exciter_decay_s"] < 0.012
        assert 1500.0 < params["exciter_center_hz"] < 5000.0
        assert 0.0 < params["tone_punch"] < 0.35

    def test_punch_does_not_override_explicit_exciter_level(self) -> None:
        params = resolve_macros({"punch": 0.9, "exciter_level": 0.01})
        assert params["exciter_level"] == 0.01

    def test_punch_is_popped_from_result(self) -> None:
        params = resolve_macros({"punch": 0.5})
        assert "punch" not in params


# ---------------------------------------------------------------------------
# 2. decay_shape macro
# ---------------------------------------------------------------------------


class TestDecayShapeMacro:
    def test_decay_shape_zero_is_tight(self) -> None:
        params = resolve_macros({"decay_shape": 0.0})
        assert abs(params["tone_decay_s"] - 0.08) < 1e-6
        assert abs(params["noise_decay_s"] - 0.015) < 1e-6
        assert abs(params["metallic_decay_s"] - 0.03) < 1e-6
        assert abs(params["tone_sweep_decay_s"] - 0.02) < 1e-6

    def test_decay_shape_one_is_boomy(self) -> None:
        params = resolve_macros({"decay_shape": 1.0})
        assert abs(params["tone_decay_s"] - 0.9) < 1e-6
        assert abs(params["noise_decay_s"] - 0.3) < 1e-6
        assert abs(params["metallic_decay_s"] - 0.4) < 1e-6
        assert abs(params["tone_sweep_decay_s"] - 0.08) < 1e-6

    def test_decay_shape_does_not_override_explicit_tone_decay(self) -> None:
        params = resolve_macros({"decay_shape": 1.0, "tone_decay_s": 0.26})
        assert params["tone_decay_s"] == 0.26

    def test_decay_shape_is_popped_from_result(self) -> None:
        params = resolve_macros({"decay_shape": 0.5})
        assert "decay_shape" not in params


# ---------------------------------------------------------------------------
# 3. character macro
# ---------------------------------------------------------------------------


class TestCharacterMacro:
    def test_character_zero_is_clean(self) -> None:
        params = resolve_macros({"character": 0.0})
        assert params.get("tone_shaper") is None
        assert abs(params["tone_shaper_drive"] - 0.0) < 1e-6
        assert abs(params["filter_drive"] - 0.0) < 1e-6

    def test_character_mid_sets_tanh(self) -> None:
        params = resolve_macros({"character": 0.5})
        assert params["tone_shaper"] == "tanh"
        assert 0.0 < params["tone_shaper_drive"] < 0.6

    def test_character_high_sets_foldback(self) -> None:
        params = resolve_macros({"character": 0.8})
        assert params["tone_shaper"] == "foldback"
        assert params["tone_shaper_drive"] > 0.3

    def test_character_boosts_noise_level(self) -> None:
        params = resolve_macros({"character": 0.8, "noise_level": 0.1})
        # noise_level should be boosted above the original 0.1
        assert params["noise_level"] > 0.1

    def test_character_does_not_boost_noise_if_not_set(self) -> None:
        """If noise_level is not present at all, character does not add it."""
        params = resolve_macros({"character": 0.8})
        assert "noise_level" not in params

    def test_character_does_not_override_explicit_tone_shaper(self) -> None:
        params = resolve_macros({"character": 0.9, "tone_shaper": "atan"})
        assert params["tone_shaper"] == "atan"

    def test_character_is_popped_from_result(self) -> None:
        params = resolve_macros({"character": 0.5})
        assert "character" not in params


# ---------------------------------------------------------------------------
# 4. General macro behavior
# ---------------------------------------------------------------------------


class TestMacroGeneral:
    def test_no_macros_returns_params_unchanged(self) -> None:
        original = {"tone_type": "oscillator", "tone_level": 0.5}
        result = resolve_macros(dict(original))
        assert result == original

    def test_none_macros_treated_as_inactive(self) -> None:
        original = {"punch": None, "decay_shape": None, "character": None}
        result = resolve_macros(dict(original))
        # Macro keys should be popped even when None
        assert "punch" not in result
        assert "decay_shape" not in result
        assert "character" not in result
        assert len(result) == 0

    def test_multiple_macros_combine(self) -> None:
        params = resolve_macros({"punch": 0.7, "decay_shape": 0.3, "character": 0.4})
        # punch fills exciter params
        assert "exciter_level" in params
        # decay_shape fills decay params
        assert "tone_decay_s" in params
        # character fills shaper params
        assert "tone_shaper_drive" in params
        # all macro keys removed
        assert "punch" not in params
        assert "decay_shape" not in params
        assert "character" not in params

    def test_macros_do_not_mutate_original_beyond_expected(self) -> None:
        """resolve_macros operates on the dict in-place, but let's verify behavior."""
        params: dict = {"punch": 0.5, "tone_type": "oscillator"}
        result = resolve_macros(params)
        # result is the same dict object
        assert result is params
        assert "punch" not in result
        assert "exciter_level" in result


# ---------------------------------------------------------------------------
# 5. Integration: macros through the render path
# ---------------------------------------------------------------------------


class TestMacroIntegration:
    def test_render_with_punch_produces_valid_audio(self) -> None:
        audio = render(
            freq=50.0,
            duration=0.3,
            amp=0.8,
            sample_rate=SAMPLE_RATE,
            params={"punch": 0.8},
        )
        assert np.isfinite(audio).all()
        assert np.max(np.abs(audio)) > 0

    def test_render_with_decay_shape_zero_vs_one_energy_difference(self) -> None:
        """Tight decay (0.0) should have less total energy than boomy (1.0)."""
        tight = render(
            freq=50.0,
            duration=0.5,
            amp=0.8,
            sample_rate=SAMPLE_RATE,
            params={"decay_shape": 0.0},
        )
        boomy = render(
            freq=50.0,
            duration=0.5,
            amp=0.8,
            sample_rate=SAMPLE_RATE,
            params={"decay_shape": 1.0},
        )
        assert np.isfinite(tight).all()
        assert np.isfinite(boomy).all()
        tight_energy = float(np.sum(tight**2))
        boomy_energy = float(np.sum(boomy**2))
        assert boomy_energy > tight_energy

    def test_render_with_character_produces_valid_audio(self) -> None:
        audio = render(
            freq=80.0,
            duration=0.3,
            amp=0.8,
            sample_rate=SAMPLE_RATE,
            params={"character": 0.6},
        )
        assert np.isfinite(audio).all()
        assert np.max(np.abs(audio)) > 0

    def test_render_with_all_macros_produces_valid_audio(self) -> None:
        audio = render(
            freq=60.0,
            duration=0.3,
            amp=0.8,
            sample_rate=SAMPLE_RATE,
            params={"punch": 0.5, "decay_shape": 0.5, "character": 0.3},
        )
        assert np.isfinite(audio).all()
        assert np.max(np.abs(audio)) > 0
