"""Tests for preset-aware default effects in drum_helpers."""

from __future__ import annotations

from code_musics.drum_helpers import add_drum_voice
from code_musics.score import EffectSpec, Score


def _make_score() -> Score:
    return Score(f0_hz=55.0)


class TestPresetDefaultEffects:
    """add_drum_voice applies recommended effects based on preset category."""

    def test_kick_preset_gets_default_effects(self) -> None:
        score = _make_score()
        add_drum_voice(score, "kick", engine="drum_voice", preset="808_hiphop")
        effects = score.voices["kick"].effects
        assert len(effects) == 2
        assert effects[0] == EffectSpec("compressor", {"preset": "kick_punch"})
        assert effects[1] == EffectSpec("saturation", {"preset": "kick_weight"})

    def test_hat_preset_gets_default_effects(self) -> None:
        score = _make_score()
        add_drum_voice(score, "hat", engine="drum_voice", preset="closed_hat")
        effects = score.voices["hat"].effects
        assert len(effects) == 1
        assert effects[0] == EffectSpec("compressor", {"preset": "hat_control"})

    def test_snare_preset_gets_default_effects(self) -> None:
        score = _make_score()
        add_drum_voice(score, "snare", engine="drum_voice", preset="909_tight")
        effects = score.voices["snare"].effects
        assert len(effects) == 1
        assert effects[0] == EffectSpec("compressor", {"preset": "snare_punch"})

    def test_tom_preset_gets_default_effects(self) -> None:
        score = _make_score()
        add_drum_voice(score, "tom", engine="drum_voice", preset="round_tom")
        effects = score.voices["tom"].effects
        assert len(effects) == 1
        assert effects[0] == EffectSpec("compressor", {"preset": "tom_control"})

    def test_clap_preset_no_default_effects(self) -> None:
        score = _make_score()
        add_drum_voice(score, "clap", engine="drum_voice", preset="909_clap")
        effects = score.voices["clap"].effects
        assert effects == []

    def test_explicit_empty_effects_overrides_defaults(self) -> None:
        score = _make_score()
        add_drum_voice(
            score, "kick", engine="drum_voice", preset="808_hiphop", effects=[]
        )
        effects = score.voices["kick"].effects
        assert effects == []

    def test_explicit_effects_list_overrides_defaults(self) -> None:
        score = _make_score()
        user_effects = [EffectSpec("eq", {"bands": []})]
        add_drum_voice(
            score,
            "kick",
            engine="drum_voice",
            preset="808_hiphop",
            effects=user_effects,
        )
        effects = score.voices["kick"].effects
        assert effects == user_effects

    def test_unknown_preset_no_default_effects(self) -> None:
        score = _make_score()
        add_drum_voice(score, "perc", engine="drum_voice", preset="shaped_hit")
        effects = score.voices["perc"].effects
        assert effects == []

    def test_no_preset_no_default_effects(self) -> None:
        score = _make_score()
        add_drum_voice(score, "perc", engine="drum_voice")
        effects = score.voices["perc"].effects
        assert effects == []
