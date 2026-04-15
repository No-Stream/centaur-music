"""Tests for velocity-to-timbre infrastructure."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._drum_utils import (
    _NEUTRAL_TIMBRE,
    resolve_velocity_timbre,
)


class TestResolveVelocityTimbre:
    def test_neutral_when_no_params(self) -> None:
        result = resolve_velocity_timbre(1.0, {})
        assert result is _NEUTRAL_TIMBRE

    def test_neutral_when_unrelated_params(self) -> None:
        result = resolve_velocity_timbre(1.0, {"body_decay_ms": 200.0})
        assert result is _NEUTRAL_TIMBRE

    def test_neutral_at_unit_amp(self) -> None:
        result = resolve_velocity_timbre(
            1.0, {"velocity_timbre_decay": 0.5, "velocity_timbre_brightness": 0.3}
        )
        assert result.decay_scale == pytest.approx(1.0)
        assert result.brightness_scale == pytest.approx(1.0)
        assert result.harmonic_scale == pytest.approx(1.0)
        assert result.noise_balance == pytest.approx(0.0)

    def test_loud_hit_scales_up(self) -> None:
        result = resolve_velocity_timbre(
            1.5,
            {
                "velocity_timbre_decay": 0.5,
                "velocity_timbre_brightness": 0.4,
                "velocity_timbre_harmonics": 0.6,
                "velocity_timbre_noise": 0.3,
            },
        )
        assert result.decay_scale > 1.0
        assert result.brightness_scale > 1.0
        assert result.harmonic_scale > 1.0
        assert result.noise_balance > 0.0

    def test_soft_hit_scales_down(self) -> None:
        result = resolve_velocity_timbre(
            0.5,
            {
                "velocity_timbre_decay": 0.5,
                "velocity_timbre_brightness": 0.4,
                "velocity_timbre_harmonics": 0.6,
                "velocity_timbre_noise": 0.3,
            },
        )
        assert result.decay_scale < 1.0
        assert result.brightness_scale < 1.0
        assert result.harmonic_scale < 1.0
        assert result.noise_balance < 0.0

    def test_clamping_extreme_amp(self) -> None:
        result = resolve_velocity_timbre(
            10.0, {"velocity_timbre_decay": 1.0, "velocity_timbre_noise": 1.0}
        )
        assert result.decay_scale <= 4.0
        assert result.noise_balance <= 0.5

        result_low = resolve_velocity_timbre(
            0.01, {"velocity_timbre_decay": 1.0, "velocity_timbre_noise": 1.0}
        )
        assert result_low.decay_scale >= 0.25
        assert result_low.noise_balance >= -0.5

    def test_negative_sensitivity(self) -> None:
        result = resolve_velocity_timbre(1.5, {"velocity_timbre_decay": -0.5})
        assert result.decay_scale < 1.0  # louder = shorter with negative sens


class TestEngineIntegration:
    """Verify each engine renders with velocity_timbre params without error."""

    _ENGINES = {
        "kick_tom": {"body_decay_ms": 200.0},
        "snare": {},
        "clap": {},
        "metallic_perc": {},
        "noise_perc": {},
    }

    _TIMBRE_PARAMS = {
        "velocity_timbre_decay": 0.3,
        "velocity_timbre_brightness": 0.2,
        "velocity_timbre_harmonics": 0.2,
        "velocity_timbre_noise": 0.1,
    }

    @pytest.mark.parametrize(
        "engine_name", ["kick_tom", "snare", "clap", "metallic_perc", "noise_perc"]
    )
    def test_renders_with_timbre_params(self, engine_name: str) -> None:
        from code_musics.engines.registry import render_note_signal

        base_params = dict(self._ENGINES.get(engine_name, {}))
        params_with_timbre = {**base_params, **self._TIMBRE_PARAMS}

        result = render_note_signal(
            freq=200.0,
            duration=0.1,
            amp=1.0,
            sample_rate=44100,
            params={"engine": engine_name, **params_with_timbre},
        )
        assert result.dtype == np.float64
        assert len(result) > 0
        assert np.isfinite(result).all()

    @pytest.mark.parametrize("engine_name", ["kick_tom", "snare", "metallic_perc"])
    def test_output_changes_with_timbre(self, engine_name: str) -> None:
        """Verify timbre params actually change the output (not just no-op)."""
        from code_musics.engines.registry import render_note_signal

        base_params = dict(self._ENGINES.get(engine_name, {}))

        without = render_note_signal(
            freq=200.0,
            duration=0.15,
            amp=1.5,  # above reference = timbre should differ
            sample_rate=44100,
            params={"engine": engine_name, **base_params},
        )
        with_timbre = render_note_signal(
            freq=200.0,
            duration=0.15,
            amp=1.5,
            sample_rate=44100,
            params={"engine": engine_name, **base_params, **self._TIMBRE_PARAMS},
        )
        assert not np.allclose(without, with_timbre), (
            f"{engine_name}: output should differ with velocity_timbre params"
        )
