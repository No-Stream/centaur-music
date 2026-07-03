"""Tests for the default master chain in code_musics.pieces._shared."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from code_musics.pieces._shared import (
    DEFAULT_MASTER_EFFECTS,
    _make_bus_comp,
    _make_preamp,
    bricasti_or_reverb,
    has_bricasti_ir,
)
from code_musics.score import EffectSpec
from code_musics.synth import _COMPRESSOR_PRESETS


class TestDefaultMasterEffects:
    def test_default_master_effects_is_nonempty_list_of_effect_specs(self) -> None:
        assert isinstance(DEFAULT_MASTER_EFFECTS, list)
        assert len(DEFAULT_MASTER_EFFECTS) > 0
        for effect in DEFAULT_MASTER_EFFECTS:
            assert isinstance(effect, EffectSpec)

    def test_preamp_fallback_returns_native_preamp(self) -> None:
        with patch(
            "code_musics.pieces._shared.has_external_plugin", return_value=False
        ):
            effect = _make_preamp()
        assert effect.kind == "preamp"
        assert effect.params.get("preset") == "neve_warmth"

    def test_bus_comp_fallback_returns_native_compressor(self) -> None:
        with patch(
            "code_musics.pieces._shared.has_external_plugin", return_value=False
        ):
            effect = _make_bus_comp()
        assert effect.kind == "compressor"
        assert effect.params.get("preset") == "master_glue"


class TestBricastiFallback:
    def test_has_bricasti_ir_requires_both_stereo_files(self, tmp_path: Path) -> None:
        ir_name = "1 Halls 07 Large & Dark"
        left = tmp_path / f"{ir_name}, 44K L.wav"
        left.write_bytes(b"left")

        with patch("code_musics.pieces._shared.BRICASTI_IR_DIR", tmp_path):
            assert not has_bricasti_ir(ir_name)

        right = tmp_path / f"{ir_name}, 44K R.wav"
        right.write_bytes(b"right")
        with patch("code_musics.pieces._shared.BRICASTI_IR_DIR", tmp_path):
            assert has_bricasti_ir(ir_name)

    def test_bricasti_or_reverb_uses_native_reverb_when_ir_missing(self) -> None:
        with patch("code_musics.pieces._shared.has_bricasti_ir", return_value=False):
            effect = bricasti_or_reverb(
                "1 Halls 07 Large & Dark",
                0.2,
                highpass_hz=150.0,
            )

        assert effect.kind == "reverb"
        assert effect.params == {
            "room_size": 0.75,
            "damping": 0.6,
            "wet_level": 0.2,
        }

    def test_bricasti_or_reverb_preserves_bricasti_params_when_available(self) -> None:
        with patch("code_musics.pieces._shared.has_bricasti_ir", return_value=True):
            effect = bricasti_or_reverb(
                "1 Halls 07 Large & Dark",
                0.2,
                highpass_hz=150.0,
            )

        assert effect.kind == "bricasti"
        assert effect.params == {
            "ir_name": "1 Halls 07 Large & Dark",
            "wet": 0.2,
            "highpass_hz": 150.0,
        }


class TestMasterGluePreset:
    def test_master_glue_preset_exists(self) -> None:
        assert "master_glue" in _COMPRESSOR_PRESETS

    def test_master_glue_uses_feedback_topology(self) -> None:
        preset = _COMPRESSOR_PRESETS["master_glue"]
        assert preset["topology"] == "feedback"

    def test_master_glue_uses_rms_detector(self) -> None:
        preset = _COMPRESSOR_PRESETS["master_glue"]
        assert preset["detector_mode"] == "rms"
