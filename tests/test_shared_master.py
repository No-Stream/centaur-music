"""Tests for the default master chain in code_musics.pieces._shared."""

from __future__ import annotations

from unittest.mock import patch

from code_musics.pieces._shared import (
    DEFAULT_MASTER_EFFECTS,
    _make_bus_comp,
    _make_preamp,
)
from code_musics.score import EffectSpec
from code_musics.synth import _COMPRESSOR_PRESETS


class TestDefaultMasterEffects:
    def test_default_master_effects_is_nonempty_list_of_effect_specs(self) -> None:
        assert isinstance(DEFAULT_MASTER_EFFECTS, list)
        assert len(DEFAULT_MASTER_EFFECTS) > 0
        for effect in DEFAULT_MASTER_EFFECTS:
            assert isinstance(effect, EffectSpec)

    def test_preamp_fallback_returns_native_saturation(self) -> None:
        with patch(
            "code_musics.pieces._shared.has_external_plugin", return_value=False
        ):
            effect = _make_preamp()
        assert effect.kind == "saturation"

    def test_bus_comp_fallback_returns_native_compressor(self) -> None:
        with patch(
            "code_musics.pieces._shared.has_external_plugin", return_value=False
        ):
            effect = _make_bus_comp()
        assert effect.kind == "compressor"
        assert effect.params.get("preset") == "master_glue"


class TestMasterGluePreset:
    def test_master_glue_preset_exists(self) -> None:
        assert "master_glue" in _COMPRESSOR_PRESETS

    def test_master_glue_uses_feedback_topology(self) -> None:
        preset = _COMPRESSOR_PRESETS["master_glue"]
        assert preset["topology"] == "feedback"

    def test_master_glue_uses_rms_detector(self) -> None:
        preset = _COMPRESSOR_PRESETS["master_glue"]
        assert preset["detector_mode"] == "rms"
