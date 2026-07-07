"""Tests for the native FDN reverb effect."""

from __future__ import annotations

from math import gcd

import numpy as np
import pytest
from scipy.signal import butter, lfilter, sosfilt

from code_musics.engines import _fdn_reverb as fdn
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
    return np.asarray(sosfilt(sos, signal))


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


def _pink_noise(n_samples: int, seed: int = 0) -> np.ndarray:
    """Unit-RMS pink noise (Paul Kellett filter on white noise)."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n_samples)
    pink = np.asarray(
        lfilter(
            [0.049922, -0.095993, 0.050612, -0.004408],
            [1.0, -2.494956, 2.017265, -0.522189],
            white,
        )
    )
    return pink / np.sqrt(np.mean(pink**2))


class TestRT60Accuracy:
    @pytest.mark.parametrize("decay_s", [2.0, 10.0, 30.0, 45.0])
    def test_reference_band_rt60_within_10pct(self, decay_s: float) -> None:
        """decay_s == 1 kHz reference-band RT60 within ±10% across the range.

        Measured with the damping corner well above the reference band and the
        bass extension neutralized, isolating the reference-band decay that
        ``decay_s`` is defined to control.
        """
        n_samples = int(min(decay_s * 1.4, 28.0) * SAMPLE_RATE)
        wet = apply_fdn_reverb(
            _stereo_impulse(n_samples),
            decay_s=decay_s,
            mix=1.0,
            predelay_ms=0.0,
            damping_hz=12000.0,
            low_decay_mult=1.0,
        )
        measured = _measure_rt60(wet[0], center_hz=1000.0)
        assert abs(measured - decay_s) / decay_s < 0.10, (
            f"1 kHz RT60 {measured:.2f}s vs target {decay_s}s"
        )


class TestCoprimeDelays:
    @pytest.mark.parametrize("n_lines", [8, 16])
    @pytest.mark.parametrize("size", [0.3, 0.5, 0.85, 0.95, 1.0])
    def test_delay_lengths_pairwise_coprime(self, n_lines: int, size: float) -> None:
        scaled = fdn._BASE_PRIME_DELAYS[:n_lines].astype(np.float64) * (
            0.4 + 1.3 * size
        )
        delays = fdn._snap_to_distinct_primes(np.maximum(2.0, scaled))
        assert len(set(delays.tolist())) == n_lines, "delays must be distinct"
        for i in range(n_lines):
            for j in range(i + 1, n_lines):
                assert gcd(int(delays[i]), int(delays[j])) == 1


class TestWetLevelCalibration:
    def test_sustained_input_wet_rms_same_order_as_input(self) -> None:
        """5 s sustained pink at decay 45 s returns wet RMS on input's order."""
        pink = _pink_noise(5 * SAMPLE_RATE)
        input_rms = float(np.sqrt(np.mean(pink**2)))
        wet = apply_fdn_reverb(pink, decay_s=45.0, mix=1.0, predelay_ms=0.0)
        wet_rms = float(np.sqrt(np.mean(wet**2)))
        ratio = wet_rms / input_rms
        # "Same order as input" — not the several-times-input buildup of the
        # un-normalized version, and audibly present (not vanishing).
        assert 0.3 < ratio < 3.0, f"wet/input RMS ratio {ratio:.2f} out of range"

    def test_shorter_decay_wet_rms_bounded(self) -> None:
        pink = _pink_noise(5 * SAMPLE_RATE)
        input_rms = float(np.sqrt(np.mean(pink**2)))
        wet = apply_fdn_reverb(pink, decay_s=8.0, mix=1.0, predelay_ms=0.0)
        ratio = float(np.sqrt(np.mean(wet**2))) / input_rms
        assert 0.2 < ratio < 2.0


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

    @pytest.mark.parametrize("low_decay_mult", [0.5, 1.0, 3.0, 10.0])
    @pytest.mark.parametrize("size", [0.3, 1.0])
    def test_stable_across_bass_extension_and_size(
        self, low_decay_mult: float, size: float
    ) -> None:
        """The unconditional loop clamp keeps every config bounded.

        Large ``low_decay_mult`` on short lines is exactly the case a naive
        scalar tonal boost would push above unity DC loop gain.
        """
        wet = apply_fdn_reverb(
            _stereo_impulse(3 * SAMPLE_RATE),
            decay_s=45.0,
            mix=1.0,
            predelay_ms=0.0,
            low_decay_mult=low_decay_mult,
            size=size,
            damping_hz=6000.0,
        )
        assert np.all(np.isfinite(wet))
        assert np.max(np.abs(wet)) < 10.0

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

    def test_sustained_dc_input_no_unbounded_accumulation(self) -> None:
        """A held DC input rings but stays bounded and decays once removed."""
        signal = np.zeros((2, 4 * SAMPLE_RATE), dtype=np.float64)
        signal[:, : 2 * SAMPLE_RATE] = 1.0
        wet = apply_fdn_reverb(
            signal,
            decay_s=30.0,
            low_decay_mult=3.0,
            mix=1.0,
            predelay_ms=0.0,
        )
        assert np.all(np.isfinite(wet))
        assert np.max(np.abs(wet)) < 5.0
        # After the input stops, the DC-band offset must relax toward zero.
        early_dc = abs(float(np.mean(wet[:, : 2 * SAMPLE_RATE])))
        late_dc = abs(float(np.mean(wet[:, int(3.5 * SAMPLE_RATE) :])))
        assert late_dc < early_dc


class TestStereoDecorrelation:
    @pytest.mark.parametrize("feedback_matrix", ["householder", "hadamard"])
    @pytest.mark.parametrize("seed", [0, 3, 9])
    def test_late_tail_lr_correlation_low(
        self, feedback_matrix: str, seed: int
    ) -> None:
        n_samples = int(4.0 * SAMPLE_RATE)
        wet = apply_fdn_reverb(
            _stereo_impulse(n_samples),
            decay_s=4.0,
            mix=1.0,
            predelay_ms=0.0,
            feedback_matrix=feedback_matrix,
            seed=seed,
        )
        late = wet[:, int(1.0 * SAMPLE_RATE) :]
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
            {"modulation_rate_hz": 3.0},
            {"modulation_rate_hz": -0.1},
        ],
    )
    def test_invalid_params_raise(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            apply_fdn_reverb(_impulse(1000), **kwargs)

    def test_modulation_rate_at_cap_ok(self) -> None:
        out = apply_fdn_reverb(_impulse(1000), modulation_rate_hz=2.0)
        assert np.all(np.isfinite(out))
