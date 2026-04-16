"""Tests for voice card spread parameter and offset dimensions."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._dsp_utils import (
    apply_voice_card,
    extract_analog_params,
    voice_card_offsets,
)


class TestVoiceCardSpread:
    def test_spread_zero_no_variation(self) -> None:
        """spread=0 should return unmodified values."""
        freq = np.ones(100) * 440.0
        fp, amp, cutoff, extras = apply_voice_card(
            {"_voice_name": "test"},
            voice_card_spread=0.0,
            freq_profile=freq.copy(),
            amp=1.0,
            cutoff_hz=3000.0,
        )
        np.testing.assert_allclose(fp, freq)
        assert amp == pytest.approx(1.0)
        assert cutoff == pytest.approx(3000.0)
        assert extras["attack_scale"] == pytest.approx(1.0)
        assert extras["pulse_width_offset"] == pytest.approx(0.0)

    def test_spread_one_matches_original_ranges(self) -> None:
        """spread=1 should give offsets within the defined ranges."""
        offsets = voice_card_offsets("voice_A")
        assert abs(offsets["pitch_offset_cents"]) <= 0.5
        assert abs(offsets["cutoff_offset_cents"]) <= 50.0

    def test_higher_spread_wider_variation(self) -> None:
        """Spread=3 should produce wider offsets than spread=1."""
        freq = np.ones(100) * 440.0
        _, _, _, extras_1 = apply_voice_card(
            {"_voice_name": "test"},
            voice_card_spread=1.0,
            freq_profile=freq.copy(),
            amp=1.0,
        )
        _, _, _, extras_3 = apply_voice_card(
            {"_voice_name": "test"},
            voice_card_spread=3.0,
            freq_profile=freq.copy(),
            amp=1.0,
        )
        # Spread=3 offsets should be ~3x spread=1 offsets
        assert (
            abs(extras_3["pulse_width_offset"])
            > abs(extras_1["pulse_width_offset"]) * 2.5
        )

    def test_new_offset_dimensions_present(self) -> None:
        """New offset dimensions should be in the offsets dict."""
        offsets = voice_card_offsets("any_voice")
        for key in [
            "pulse_width_offset",
            "resonance_offset_pct",
            "softness_offset",
            "drift_rate_offset_pct",
        ]:
            assert key in offsets

    def test_legacy_voice_card_backward_compat(self) -> None:
        """Old 'voice_card' param key should still work via fallback."""
        result = extract_analog_params({"voice_card": 0.5})
        assert "voice_card_spread" in result

    def test_per_group_pitch_override(self) -> None:
        """Per-group pitch spread should override global for pitch only."""
        freq = np.ones(100) * 440.0
        fp_global, _, _, _ = apply_voice_card(
            {"_voice_name": "test_voice"},
            voice_card_spread=2.0,
            freq_profile=freq.copy(),
            amp=1.0,
        )
        fp_tight, _, _, _ = apply_voice_card(
            {"_voice_name": "test_voice"},
            voice_card_spread=2.0,
            pitch_spread=0.1,
            freq_profile=freq.copy(),
            amp=1.0,
        )
        # Tight pitch should deviate less than global
        global_dev = np.abs(fp_global[0] / 440.0 - 1.0)
        tight_dev = np.abs(fp_tight[0] / 440.0 - 1.0)
        assert tight_dev < global_dev

    def test_per_group_filter_override(self) -> None:
        """Per-group filter spread should affect cutoff independently."""
        freq = np.ones(100) * 440.0
        _, _, cut_global, ex_global = apply_voice_card(
            {"_voice_name": "test_voice"},
            voice_card_spread=1.0,
            freq_profile=freq.copy(),
            amp=1.0,
            cutoff_hz=3000.0,
        )
        _, _, cut_wide, ex_wide = apply_voice_card(
            {"_voice_name": "test_voice"},
            voice_card_spread=1.0,
            filter_spread=3.0,
            freq_profile=freq.copy(),
            amp=1.0,
            cutoff_hz=3000.0,
        )
        # Filter-wide should deviate more from 3000 than global
        assert cut_global is not None and cut_wide is not None
        assert abs(cut_wide - 3000.0) > abs(cut_global - 3000.0) * 2.0
        # Resonance offset should also be wider
        assert (
            abs(ex_wide["resonance_offset_pct"])
            > abs(ex_global["resonance_offset_pct"]) * 2.0
        )
