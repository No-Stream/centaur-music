"""Unit tests for synth_voice perceptual macros.

Covers resolution order (preset -> macro fill-in -> user kwargs win),
``_set_if_absent`` semantics, per-macro fan-out, and macro-key pop behavior.
"""

from __future__ import annotations

import pytest

from code_musics.engines._synth_macros import resolve_macros


def test_no_macros_means_no_changes() -> None:
    params: dict = {"osc_type": "polyblep", "filter_cutoff_hz": 1234.0}
    resolved = resolve_macros(params)
    assert resolved is params
    assert resolved == {"osc_type": "polyblep", "filter_cutoff_hz": 1234.0}


def test_brightness_fans_out_to_filter_cutoff() -> None:
    params: dict = {"brightness": 1.0}
    resolve_macros(params)
    assert params["filter_cutoff_hz"] == pytest.approx(8000.0, rel=1e-6)
    assert "brightness" not in params  # key popped


def test_brightness_low_uses_low_cutoff() -> None:
    params: dict = {"brightness": 0.0}
    resolve_macros(params)
    assert params["filter_cutoff_hz"] == pytest.approx(400.0, rel=1e-6)


def test_brightness_exp_midpoint_is_geometric_mean() -> None:
    params: dict = {"brightness": 0.5}
    resolve_macros(params)
    # Geometric mean of 400 and 8000 is sqrt(3.2e6) ~= 1788.85
    assert params["filter_cutoff_hz"] == pytest.approx(1788.854, rel=1e-3)


def test_user_explicit_value_wins_over_macro() -> None:
    params: dict = {"brightness": 1.0, "filter_cutoff_hz": 500.0}
    resolve_macros(params)
    assert params["filter_cutoff_hz"] == 500.0


def test_movement_fans_out_to_filter_env_amount() -> None:
    params: dict = {"movement": 0.5}
    resolve_macros(params)
    assert params["filter_env_amount"] == pytest.approx(0.3, rel=1e-6)
    assert params["partials_smear"] == pytest.approx(0.2, rel=1e-6)


def test_body_inversely_biases_hpf() -> None:
    low = {"body": 0.0}
    high = {"body": 1.0}
    resolve_macros(low)
    resolve_macros(high)
    # Low body -> HPF cuts low (200 Hz); high body -> HPF shelved down to 20 Hz
    assert low["hpf_cutoff_hz"] == pytest.approx(200.0, rel=1e-6)
    assert high["hpf_cutoff_hz"] == pytest.approx(20.0, rel=1e-6)


def test_dirt_low_uses_no_shaper() -> None:
    params: dict = {"dirt": 0.02}
    resolve_macros(params)
    assert params.get("shaper") is None or "shaper" not in params


def test_dirt_mid_picks_preamp() -> None:
    params: dict = {"dirt": 0.5}
    resolve_macros(params)
    assert params["shaper"] == "preamp"


def test_dirt_high_picks_hard_clip() -> None:
    params: dict = {"dirt": 0.9}
    resolve_macros(params)
    assert params["shaper"] == "hard_clip"


def test_dirt_extreme_bumps_feedback() -> None:
    params: dict = {"dirt": 1.0}
    resolve_macros(params)
    assert params["feedback_amount"] == pytest.approx(0.15, rel=1e-6)


def test_dirt_moderate_does_not_touch_feedback() -> None:
    params: dict = {"dirt": 0.5}
    resolve_macros(params)
    assert "feedback_amount" not in params


def test_multiple_macros_compose_without_conflict() -> None:
    params: dict = {"brightness": 0.5, "body": 0.5, "dirt": 0.4}
    resolve_macros(params)
    assert "filter_cutoff_hz" in params
    assert "hpf_cutoff_hz" in params
    assert params["shaper"] == "preamp"
    # All macro keys popped.
    for key in ("brightness", "body", "dirt"):
        assert key not in params


def test_macro_keys_always_popped_even_when_inactive() -> None:
    params: dict = {"brightness": None, "movement": None, "body": None, "dirt": None}
    resolve_macros(params)
    assert params == {}


def test_dirt_shaper_choice_respects_explicit_user_shaper() -> None:
    params: dict = {"dirt": 0.9, "shaper": "foldback"}
    resolve_macros(params)
    assert params["shaper"] == "foldback"
