"""Tests for preset-aware default effects in drum_helpers."""

from __future__ import annotations

import pytest

from code_musics.drum_helpers import DRUM_BUS_STYLES, add_drum_voice, setup_drum_bus
from code_musics.score import EffectSpec, Score, SendBusSpec


def _get_bus(score: Score, name: str) -> SendBusSpec:
    return next(bus for bus in score.send_buses if bus.name == name)


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
        assert effects[1] == EffectSpec("preamp", {"preset": "kick_body"})

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
        assert len(effects) == 2
        assert effects[0] == EffectSpec("compressor", {"preset": "tom_control"})
        assert effects[1] == EffectSpec("preamp", {"preset": "tom_body"})

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


class TestDrumBusStyles:
    """setup_drum_bus resolves style presets to comp -> sat -> limit chains."""

    def test_default_style_is_electronic(self) -> None:
        score = _make_score()
        setup_drum_bus(score)
        kinds = [e.kind for e in _get_bus(score, "drum_bus").effects]
        # Electronic uses preamp (flux-domain warmth) as its middle stage
        # for hi-fi glue without papery treble buildup, then a poly-knee
        # clipper shaves kick peaks musically.  True-peak management lives
        # on the master bus (DEFAULT_MASTER_EFFECTS); the bus clipper just
        # keeps kick peaks in check before the master sees them.
        assert kinds == ["compressor", "preamp", "clipper"]

    @pytest.mark.parametrize("style", sorted(DRUM_BUS_STYLES))
    def test_style_starts_with_comp_then_color(self, style: str) -> None:
        score = _make_score()
        bus_name = f"bus_{style}"
        setup_drum_bus(score, style=style, bus_name=bus_name)
        bus_effects = _get_bus(score, bus_name).effects
        assert 2 <= len(bus_effects) <= 3, (
            f"{style=} should have 2 or 3 effects (comp, color, optional clipper)"
        )
        kinds = [e.kind for e in bus_effects]
        assert kinds[0] == "compressor"
        assert kinds[1] in {"drive", "preamp"}
        # Heavier styles chain a clipper on the end; "light" stays clean.
        if len(bus_effects) == 3:
            assert kinds[2] == "clipper"
            assert style != "light", "light style should remain clipper-free"

    def test_explicit_effects_replaces_style(self) -> None:
        score = _make_score()
        user_effects = [EffectSpec("eq", {"bands": []})]
        setup_drum_bus(score, effects=user_effects)
        assert _get_bus(score, "drum_bus").effects == user_effects

    def test_explicit_empty_effects_gives_bare_bus(self) -> None:
        score = _make_score()
        setup_drum_bus(score, effects=[])
        assert _get_bus(score, "drum_bus").effects == []

    def test_unknown_style_raises(self) -> None:
        score = _make_score()
        with pytest.raises(ValueError, match="Unknown drum bus style"):
            setup_drum_bus(score, style="nonexistent")

    def test_berghain_compresses_harder_than_light(self) -> None:
        """Aggressive styles should squash harder than light ones."""
        score_light = _make_score()
        score_berg = _make_score()
        setup_drum_bus(score_light, style="light", bus_name="l")
        setup_drum_bus(score_berg, style="berghain", bus_name="b")
        light_comp = next(
            e for e in _get_bus(score_light, "l").effects if e.kind == "compressor"
        )
        berg_comp = next(
            e for e in _get_bus(score_berg, "b").effects if e.kind == "compressor"
        )
        assert berg_comp.params["ratio"] > light_comp.params["ratio"]
        # Piece-aware calibration: berghain targets more gain reduction on
        # the loud parts than light, independent of input level.
        assert (
            berg_comp.params["target_avg_gr_db"] > light_comp.params["target_avg_gr_db"]
        )
