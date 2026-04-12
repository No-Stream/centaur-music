"""Tests for phaser and modulated delay effects."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from code_musics.synth import (
    _SIMPLE_EFFECT_DISPATCH,
    SAMPLE_RATE,
    apply_mod_delay,
    apply_phaser,
    has_external_plugin,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHOW_PHASER_AVAILABLE = has_external_plugin("chow_phaser_stereo")


def _impulse(*, sample_rate: int = SAMPLE_RATE, duration_s: float = 0.1) -> np.ndarray:
    """A single-sample impulse (click) followed by silence."""
    n = int(sample_rate * duration_s)
    signal = np.zeros(n, dtype=np.float64)
    signal[0] = 1.0
    return signal


def _white_noise(
    *, sample_rate: int = SAMPLE_RATE, duration_s: float = 0.5, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sample_rate * duration_s)
    return rng.standard_normal(n)


def _sine_wave(
    frequency_hz: float = 440.0,
    *,
    sample_rate: int = SAMPLE_RATE,
    duration_s: float = 1.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * t)


# ---------------------------------------------------------------------------
# Phaser tests
# ---------------------------------------------------------------------------


class TestPhaser:
    def test_phaser_loads_or_skips(self, caplog: pytest.LogCaptureFixture) -> None:
        signal = _sine_wave(duration_s=0.5)
        with caplog.at_level(logging.WARNING):
            result = apply_phaser(signal)
        if _CHOW_PHASER_AVAILABLE:
            assert result.shape[-1] == signal.shape[-1]
            assert not np.allclose(result, signal, atol=1e-6), (
                "Phaser should modify the signal when plugin is available"
            )
        else:
            np.testing.assert_array_equal(result, signal)
            assert any("chow_phaser" in r.message.lower() for r in caplog.records)

    def test_phaser_stereo_input(self) -> None:
        mono = _sine_wave(duration_s=0.3)
        stereo = np.stack([mono, mono])
        result = apply_phaser(stereo)
        if _CHOW_PHASER_AVAILABLE:
            assert result.ndim == 2 and result.shape[0] == 2
        else:
            np.testing.assert_array_equal(result, stereo)


# ---------------------------------------------------------------------------
# Modulated delay tests
# ---------------------------------------------------------------------------


class TestModDelay:
    def test_mod_delay_produces_output(self) -> None:
        click = _impulse(duration_s=0.5)
        result = apply_mod_delay(click, delay_ms=100.0, feedback=0.4, mix=0.5)
        result_mono = result.mean(axis=0) if result.ndim == 2 else result
        # The input is a single impulse at sample 0. The delay should produce
        # echoes in the region that was originally silent.
        delay_region_start = int(0.08 * SAMPLE_RATE)
        echo_energy = float(np.sum(result_mono[delay_region_start:] ** 2))
        assert echo_energy > 1e-6, (
            f"Modulated delay should produce audible echoes; echo energy={echo_energy:.6f}"
        )

    def test_mod_delay_feedback_darkens(self) -> None:
        click = _impulse(duration_s=1.0)
        result = apply_mod_delay(
            click,
            delay_ms=200.0,
            feedback=0.6,
            feedback_lpf_hz=2000.0,
            mod_depth_ms=0.0,
            mix=1.0,
        )
        result_mono = result.mean(axis=0) if result.ndim == 2 else result
        delay_samples = int(0.2 * SAMPLE_RATE)
        window_half = 400
        first_echo_start = max(0, delay_samples - window_half)
        first_echo_end = delay_samples + window_half
        second_echo_start = max(0, 2 * delay_samples - window_half)
        second_echo_end = 2 * delay_samples + window_half
        if second_echo_end > len(result_mono):
            pytest.skip("Signal too short for second echo analysis")
        first_echo = result_mono[first_echo_start:first_echo_end]
        second_echo = result_mono[second_echo_start:second_echo_end]
        first_spectrum = np.abs(np.fft.rfft(first_echo))
        second_spectrum = np.abs(np.fft.rfft(second_echo))
        freqs = np.fft.rfftfreq(len(first_echo), d=1.0 / SAMPLE_RATE)
        high_freq_mask = freqs > 4000.0
        if not np.any(high_freq_mask):
            pytest.skip("Not enough frequency resolution")
        first_high_energy = float(np.sum(first_spectrum[high_freq_mask] ** 2))
        second_high_energy = float(np.sum(second_spectrum[high_freq_mask] ** 2))
        first_total_energy = float(np.sum(first_spectrum**2))
        second_total_energy = float(np.sum(second_spectrum**2))
        if first_total_energy < 1e-12 or second_total_energy < 1e-12:
            pytest.skip("Echo energy too low for analysis")
        first_high_ratio = first_high_energy / first_total_energy
        second_high_ratio = second_high_energy / second_total_energy
        assert second_high_ratio < first_high_ratio, (
            f"Feedback LPF should darken later echoes: "
            f"first HF ratio={first_high_ratio:.4f}, second={second_high_ratio:.4f}"
        )

    def test_mod_delay_stereo_spread(self) -> None:
        signal = _sine_wave(duration_s=0.3)
        result = apply_mod_delay(
            signal,
            delay_ms=150.0,
            stereo_offset_deg=90.0,
            mod_depth_ms=5.0,
            mix=0.5,
        )
        assert result.ndim == 2 and result.shape[0] == 2, (
            "Mod delay with stereo offset should produce stereo output"
        )
        left = result[0]
        right = result[1]
        assert not np.allclose(left, right, atol=1e-6), (
            "L and R channels should differ with stereo_offset > 0"
        )

    def test_mod_delay_no_runaway(self) -> None:
        noise = _white_noise(duration_s=1.0)
        input_peak = float(np.max(np.abs(noise)))
        result = apply_mod_delay(noise, feedback=0.92, delay_ms=100.0, mix=0.5)
        assert np.all(np.isfinite(result)), "Output must not contain NaN or Inf"
        output_peak = float(np.max(np.abs(result)))
        max_allowed_peak = input_peak * 2.0  # 6 dB above input peak
        assert output_peak < max_allowed_peak, (
            f"Output peak {output_peak:.2f} exceeds {max_allowed_peak:.2f} "
            f"(6 dB above input peak {input_peak:.2f})"
        )

    def test_mod_delay_presets_valid(self) -> None:
        from code_musics.synth import _MOD_DELAY_PRESETS

        signal = _sine_wave(duration_s=0.2)
        for preset_name in _MOD_DELAY_PRESETS:
            result = apply_mod_delay(signal, preset=preset_name)
            assert result.shape[-1] >= signal.shape[-1], (
                f"Preset {preset_name!r} produced shorter output than input"
            )
            assert np.all(np.isfinite(result)), (
                f"Preset {preset_name!r} produced non-finite values"
            )

    def test_mod_delay_mix_zero_is_dry(self) -> None:
        signal = _sine_wave(duration_s=0.2)
        result = apply_mod_delay(signal, mix=0.0)
        result_mono = result.mean(axis=0) if result.ndim == 2 else result
        np.testing.assert_allclose(result_mono, signal, atol=1e-10)

    def test_mod_delay_in_effect_dispatch(self) -> None:
        assert "mod_delay" in _SIMPLE_EFFECT_DISPATCH


class TestPhaserDispatch:
    def test_phaser_in_plugin_backed_effects(self) -> None:
        from code_musics.synth import _PLUGIN_BACKED_EFFECTS

        assert "phaser" in _PLUGIN_BACKED_EFFECTS
