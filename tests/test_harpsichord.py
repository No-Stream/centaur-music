"""Harpsichord engine tests."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines.harpsichord import render
from code_musics.engines.registry import _PRESETS, render_note_signal

SAMPLE_RATE = 44_100
DURATION = 0.3
FREQ = 220.0
AMP = 0.7


def _render_default(**overrides: object) -> np.ndarray:
    kwargs: dict = {
        "freq": FREQ,
        "duration": DURATION,
        "amp": AMP,
        "sample_rate": SAMPLE_RATE,
        "params": {},
    }
    kwargs.update(overrides)
    return render(**kwargs)


def _spectral_centroid(signal: np.ndarray, sample_rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    total = np.sum(spectrum)
    if total <= 0:
        return 0.0
    return float(np.sum(freqs * spectrum) / total)


class TestBasicRendering:
    def test_render_basic(self) -> None:
        signal = _render_default()
        n_expected = int(SAMPLE_RATE * DURATION)
        assert signal.shape == (n_expected,)
        assert np.all(np.isfinite(signal))
        assert np.max(np.abs(signal)) > 0.0

    def test_render_deterministic(self) -> None:
        params = {"pluck_hardness": 0.5, "drift": 0.05}
        first = _render_default(params=params)
        second = _render_default(params=params)
        assert np.allclose(first, second)

    @pytest.mark.parametrize("freq", [55.0, 220.0, 880.0, 4000.0])
    def test_render_various_frequencies(self, freq: float) -> None:
        signal = render(
            freq=freq,
            duration=DURATION,
            amp=AMP,
            sample_rate=SAMPLE_RATE,
            params={},
        )
        n_expected = int(SAMPLE_RATE * DURATION)
        assert signal.shape == (n_expected,)
        assert np.all(np.isfinite(signal))
        assert np.max(np.abs(signal)) > 0.0

    @pytest.mark.parametrize("duration", [0.05, 0.3, 1.0])
    def test_render_various_durations(self, duration: float) -> None:
        signal = render(
            freq=FREQ,
            duration=duration,
            amp=AMP,
            sample_rate=SAMPLE_RATE,
            params={},
        )
        n_expected = int(SAMPLE_RATE * duration)
        assert signal.shape == (n_expected,)
        assert np.all(np.isfinite(signal))

    def test_output_shape_matches_duration(self) -> None:
        for dur in [0.1, 0.5, 1.5]:
            signal = render(
                freq=FREQ,
                duration=dur,
                amp=AMP,
                sample_rate=SAMPLE_RATE,
                params={},
            )
            assert signal.shape == (int(SAMPLE_RATE * dur),)


class TestValidation:
    def test_negative_freq_raises(self) -> None:
        with pytest.raises(ValueError, match="freq must be positive"):
            render(
                freq=-1.0,
                duration=DURATION,
                amp=AMP,
                sample_rate=SAMPLE_RATE,
                params={},
            )

    def test_zero_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="duration must be positive"):
            render(
                freq=FREQ,
                duration=0.0,
                amp=AMP,
                sample_rate=SAMPLE_RATE,
                params={},
            )

    def test_negative_inharmonicity_raises(self) -> None:
        with pytest.raises(ValueError, match="inharmonicity must be non-negative"):
            _render_default(params={"inharmonicity": -0.01})

    def test_all_blends_zero_raises(self) -> None:
        with pytest.raises(
            ValueError, match="at least one register must have blend > 0"
        ):
            _render_default(
                params={
                    "front_8_blend": 0.0,
                    "back_8_blend": 0.0,
                    "four_foot_blend": 0.0,
                    "lute_blend": 0.0,
                }
            )


class TestCustomPartialRatios:
    def test_xenharmonic_ratios_render(self) -> None:
        septimal_ratios = [
            {"ratio": 1.0, "amp": 1.0},
            {"ratio": 7 / 4, "amp": 0.6},
            {"ratio": 3 / 2, "amp": 0.5},
        ]
        signal = _render_default(
            params={"partial_ratios": septimal_ratios, "drift": 0.0}
        )
        assert np.all(np.isfinite(signal))
        assert np.max(np.abs(signal)) > 0.0

    def test_custom_ratios_differ_from_default(self) -> None:
        septimal_ratios = [
            {"ratio": 1.0, "amp": 1.0},
            {"ratio": 7 / 4, "amp": 0.6},
            {"ratio": 3 / 2, "amp": 0.5},
        ]
        custom = _render_default(
            params={"partial_ratios": septimal_ratios, "drift": 0.0}
        )
        default = _render_default(params={"drift": 0.0})
        assert not np.allclose(custom, default)


class TestVelocityResponse:
    def test_velocity_changes_waveform_shape(self) -> None:
        """Different amp values should produce different waveform shapes via velocity tilt."""
        shared = {
            "drift": 0.0,
            "release_noise": 0.0,
            "body_saturation": 0.0,
            "soundboard_color": 0.0,
            "pluck_noise": 0.0,
            "attack_brightness": 1.0,
        }
        soft = render(
            freq=FREQ,
            duration=0.5,
            amp=0.15,
            sample_rate=SAMPLE_RATE,
            params=shared,
        )
        loud = render(
            freq=FREQ,
            duration=0.5,
            amp=0.9,
            sample_rate=SAMPLE_RATE,
            params=shared,
        )
        soft_norm = soft / max(1e-12, np.max(np.abs(soft)))
        loud_norm = loud / max(1e-12, np.max(np.abs(loud)))
        assert not np.allclose(soft_norm, loud_norm), (
            "velocity should change waveform shape, not just amplitude"
        )


class TestRegisterBlending:
    def test_multi_register_non_silent(self) -> None:
        signal = _render_default(
            params={"front_8_blend": 1.0, "back_8_blend": 0.7, "drift": 0.0}
        )
        assert np.max(np.abs(signal)) > 0.0

    def test_multi_register_differs_from_single(self) -> None:
        single = _render_default(
            params={
                "front_8_blend": 1.0,
                "back_8_blend": 0.0,
                "drift": 0.0,
                "release_noise": 0.0,
            }
        )
        multi = _render_default(
            params={
                "front_8_blend": 1.0,
                "back_8_blend": 0.7,
                "drift": 0.0,
                "release_noise": 0.0,
            }
        )
        assert not np.allclose(single, multi)

    def test_four_foot_adds_octave_content(self) -> None:
        without_4ft = _render_default(
            params={
                "front_8_blend": 1.0,
                "four_foot_blend": 0.0,
                "drift": 0.0,
                "release_noise": 0.0,
                "body_saturation": 0.0,
                "soundboard_color": 0.0,
            }
        )
        with_4ft = _render_default(
            params={
                "front_8_blend": 1.0,
                "four_foot_blend": 0.6,
                "drift": 0.0,
                "release_noise": 0.0,
                "body_saturation": 0.0,
                "soundboard_color": 0.0,
            }
        )
        assert not np.allclose(without_4ft, with_4ft)


class TestFreqTrajectory:
    def test_freq_trajectory_changes_signal(self) -> None:
        n_samples = int(SAMPLE_RATE * DURATION)
        sweep = np.linspace(FREQ, FREQ * 1.5, n_samples)
        shared = {"drift": 0.0, "release_noise": 0.0}

        with_sweep = render(
            freq=FREQ,
            duration=DURATION,
            amp=AMP,
            sample_rate=SAMPLE_RATE,
            params=shared,
            freq_trajectory=sweep,
        )
        static = _render_default(params=shared)

        assert np.all(np.isfinite(with_sweep))
        assert not np.allclose(with_sweep, static)


class TestPresets:
    @pytest.mark.parametrize(
        "preset_name", list(_PRESETS.get("harpsichord", {}).keys())
    )
    def test_preset_renders(self, preset_name: str) -> None:
        signal = render_note_signal(
            freq=FREQ,
            duration=DURATION,
            amp=AMP,
            sample_rate=SAMPLE_RATE,
            params={"engine": "harpsichord", "preset": preset_name},
        )
        n_expected = int(SAMPLE_RATE * DURATION)
        assert signal.shape == (n_expected,)
        assert np.all(np.isfinite(signal))
        assert np.max(np.abs(signal)) > 0.0


class TestRegistryIntegration:
    def test_harpsichord_via_registry(self) -> None:
        signal = render_note_signal(
            freq=FREQ,
            duration=DURATION,
            amp=AMP,
            sample_rate=SAMPLE_RATE,
            params={"engine": "harpsichord"},
        )
        n_expected = int(SAMPLE_RATE * DURATION)
        assert signal.shape == (n_expected,)
        assert np.all(np.isfinite(signal))
        assert np.max(np.abs(signal)) > 0.0


class TestScoreIntegration:
    def test_score_with_harpsichord_voice(self) -> None:
        from code_musics.score import Score

        score = Score(f0=220.0, sample_rate=22_050)
        score.add_voice(
            "hpsi",
            synth_defaults={"engine": "harpsichord"},
            normalize_lufs=-24.0,
        )
        score.add_note(
            "hpsi",
            start=0.0,
            duration=0.8,
            partial=1,
            amp=0.7,
        )
        audio = score.render()
        assert audio.size > 0
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) > 0.0
