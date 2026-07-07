"""Tests for the native FDN reverb effect."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from code_musics.score import EffectSpec
from code_musics.synth import SAMPLE_RATE, apply_effect_chain, apply_fdn_reverb


def _octave_band(signal: np.ndarray, center_hz: float) -> np.ndarray:
    """Octave-band bandpass filter around ``center_hz``."""
    low = center_hz / np.sqrt(2.0)
    high = center_hz * np.sqrt(2.0)
    sos = butter(
        4,
        [low / (SAMPLE_RATE / 2.0), high / (SAMPLE_RATE / 2.0)],
        btype="bandpass",
        output="sos",
    )
    return sosfilt(sos, signal)


def _measure_rt60(impulse_response: np.ndarray, center_hz: float) -> float:
    """Estimate RT60 from a band-filtered energy-decay curve (Schroeder)."""
    band = _octave_band(impulse_response, center_hz)
    energy = band**2
    # Backward-integrated (Schroeder) energy decay curve.
    edc = np.cumsum(energy[::-1])[::-1]
    edc_db = 10.0 * np.log10(edc / edc[0] + 1e-20)
    # Fit the slope over a clean portion of the decay.
    fit_mask = (edc_db <= -5.0) & (edc_db >= -35.0)
    times = np.arange(edc_db.size) / SAMPLE_RATE
    slope_db_per_s = np.polyfit(times[fit_mask], edc_db[fit_mask], 1)[0]
    return -60.0 / slope_db_per_s


def _impulse(n_samples: int) -> np.ndarray:
    impulse = np.zeros(n_samples, dtype=np.float64)
    impulse[0] = 1.0
    return impulse


def _stereo_impulse(n_samples: int) -> np.ndarray:
    impulse = np.zeros((2, n_samples), dtype=np.float64)
    impulse[:, 0] = 1.0
    return impulse


class TestRT60Accuracy:
    def test_rt60_within_tolerance(self) -> None:
        decay_s = 3.0
        n_samples = int(decay_s * 1.5 * SAMPLE_RATE)
        wet = apply_fdn_reverb(
            _stereo_impulse(n_samples),
            decay_s=decay_s,
            mix=1.0,
            predelay_ms=0.0,
            damping_hz=12000.0,
            low_decay_mult=1.0,
        )
        measured = _measure_rt60(wet[0], center_hz=1000.0)
        assert abs(measured - decay_s) / decay_s < 0.25


class TestStability:
    def test_60s_render_finite_and_bounded(self) -> None:
        n_samples = 60 * SAMPLE_RATE
        rng = np.random.default_rng(1)
        signal = np.zeros(n_samples, dtype=np.float64)
        signal[: SAMPLE_RATE // 2] = rng.standard_normal(SAMPLE_RATE // 2)
        wet = apply_fdn_reverb(
            signal,
            decay_s=60.0,
            size=1.0,
            modulation_depth=1.0,
            modulation_rate_hz=0.5,
            diffusion=1.0,
            mix=0.5,
            feedback_matrix="hadamard",
        )
        assert np.all(np.isfinite(wet))
        assert np.max(np.abs(wet)) < 100.0

    def test_no_dc_buildup(self) -> None:
        n_samples = int(4.0 * SAMPLE_RATE)
        wet = apply_fdn_reverb(
            _stereo_impulse(n_samples),
            decay_s=3.0,
            mix=1.0,
            predelay_ms=0.0,
        )
        late = wet[:, int(1.0 * SAMPLE_RATE) :]
        # Late-tail DC offset should be negligible relative to the tail level.
        for channel in late:
            dc = abs(float(np.mean(channel)))
            rms = float(np.sqrt(np.mean(channel**2)))
            assert dc < 0.05 * rms


class TestStereoDecorrelation:
    def test_late_tail_lr_correlation_low(self) -> None:
        n_samples = int(3.0 * SAMPLE_RATE)
        wet = apply_fdn_reverb(
            _stereo_impulse(n_samples),
            decay_s=2.5,
            mix=1.0,
            predelay_ms=0.0,
        )
        late = wet[:, int(0.5 * SAMPLE_RATE) :]
        correlation = np.corrcoef(late[0], late[1])[0, 1]
        assert abs(correlation) < 0.4


class TestDeterminism:
    def test_identical_renders(self) -> None:
        rng = np.random.default_rng(7)
        signal = rng.standard_normal(SAMPLE_RATE).astype(np.float64)
        first = apply_fdn_reverb(signal, seed=3)
        second = apply_fdn_reverb(signal, seed=3)
        assert np.array_equal(first, second)

    def test_seed_changes_modulation(self) -> None:
        rng = np.random.default_rng(7)
        signal = rng.standard_normal(SAMPLE_RATE).astype(np.float64)
        first = apply_fdn_reverb(signal, seed=1)
        second = apply_fdn_reverb(signal, seed=2)
        assert not np.array_equal(first, second)


class TestIntegration:
    def test_effect_chain_mix_and_layout(self) -> None:
        rng = np.random.default_rng(2)
        stereo = rng.standard_normal((2, SAMPLE_RATE)).astype(np.float64)
        out = apply_effect_chain(
            stereo,
            [EffectSpec("fdn_reverb", {"decay_s": 4.0, "mix": 0.4})],
        )
        assert out.shape == stereo.shape
        assert np.all(np.isfinite(out))

    def test_hadamard_matrix_option(self) -> None:
        wet = apply_fdn_reverb(
            _impulse(SAMPLE_RATE),
            decay_s=2.0,
            feedback_matrix="hadamard",
            mix=1.0,
        )
        assert np.all(np.isfinite(wet))

    def test_mono_input_returns_mono(self) -> None:
        signal = _impulse(SAMPLE_RATE // 2)
        out = apply_fdn_reverb(signal, mix=0.5)
        assert out.ndim == 1


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"decay_s": 0.0},
            {"size": 1.5},
            {"mix": 1.5},
            {"n_lines": 12},
            {"feedback_matrix": "circulant"},
            {"modulation_depth": 2.0},
        ],
    )
    def test_invalid_params_raise(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            apply_fdn_reverb(_impulse(1000), **kwargs)
