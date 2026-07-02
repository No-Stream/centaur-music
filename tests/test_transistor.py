"""End-to-end tests for the honest stompbox/op-amp transistor effect."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.score import EffectSpec
from code_musics.synth import (
    _TRANSISTOR_PRESETS,
    SAMPLE_RATE,
    _resolve_effect_params,
    apply_transistor,
)


def _make_mono_sine(
    freq_hz: float,
    duration_s: float = 1.0,
    amplitude: float = 0.25,
) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * duration_s)) / SAMPLE_RATE
    return amplitude * np.sin(2.0 * np.pi * freq_hz * t)


def _make_stereo_sine(
    freq_hz: float,
    duration_s: float = 1.0,
    amplitude: float = 0.25,
) -> np.ndarray:
    mono = _make_mono_sine(freq_hz, duration_s=duration_s, amplitude=amplitude)
    return np.stack([mono, mono])


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))


def _harmonic_magnitudes(
    signal: np.ndarray,
    fundamental_hz: float,
    n_harmonics: int = 10,
    sample_rate: int = SAMPLE_RATE,
) -> list[float]:
    mono = signal if signal.ndim == 1 else signal.mean(axis=0)
    n = len(mono)
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(n)))

    def _peak_near(hz: float) -> float:
        idx = int(round(hz * n / sample_rate))
        window = 3
        lo = max(0, idx - window)
        hi = min(len(spectrum), idx + window + 1)
        return float(np.max(spectrum[lo:hi]))

    return [_peak_near(fundamental_hz * k) for k in range(1, n_harmonics + 1)]


class TestPresetSmoke:
    """Every registered preset renders finite output at reasonable levels."""

    @pytest.mark.parametrize("preset_name", list(_TRANSISTOR_PRESETS.keys()))
    def test_preset_renders_finite(self, preset_name: str) -> None:
        signal = _make_stereo_sine(440.0, duration_s=1.0, amplitude=0.25)
        output = apply_transistor(signal, preset=preset_name)
        assert isinstance(output, np.ndarray)
        assert output.shape == signal.shape
        assert np.all(np.isfinite(output))
        assert _rms(output) > 1e-6


class TestCharacterSpectra:
    """Each character has a distinct harmonic fingerprint."""

    def test_soft_clip_is_symmetric(self) -> None:
        signal = _make_mono_sine(440.0, amplitude=0.35)
        output = apply_transistor(
            signal,
            character="soft_clip",
            drive=0.8,
            mix=1.0,
            multiband=False,
            compensation_mode="none",
        )
        # DC-blocked output on a symmetric shape has near-zero mean.
        assert abs(float(np.mean(output))) < 5e-4

    def test_diode_is_asymmetric_pre_dc_block(self) -> None:
        # The diode character produces an asymmetric curve.  After the
        # DC-block mean is near zero, but the positive/negative peak
        # ratio on a symmetric input sine should differ from 1.
        signal = _make_mono_sine(440.0, amplitude=0.35)
        output = apply_transistor(
            signal,
            character="diode",
            drive=0.8,
            bias=0.0,
            mix=1.0,
            multiband=False,
            compensation_mode="none",
        )
        pos_peak = float(np.max(output))
        neg_peak = float(-np.min(output))
        ratio = max(pos_peak, neg_peak) / max(min(pos_peak, neg_peak), 1e-9)
        assert ratio > 1.1, (
            f"diode character should be asymmetric; got pos={pos_peak:.4f} neg={neg_peak:.4f}"
        )

    def test_op_amp_has_harder_knee_than_soft_clip(self) -> None:
        signal = _make_mono_sine(440.0, amplitude=0.35)
        soft = apply_transistor(
            signal,
            character="soft_clip",
            drive=0.8,
            mix=1.0,
            multiband=False,
            compensation_mode="none",
        )
        opamp = apply_transistor(
            signal,
            character="op_amp",
            drive=0.8,
            mix=1.0,
            multiband=False,
            compensation_mode="none",
        )
        soft_h = _harmonic_magnitudes(soft, 440.0)
        opamp_h = _harmonic_magnitudes(opamp, 440.0)
        # H3 relative to H1.
        soft_h3_ratio = soft_h[2] / max(soft_h[0], 1e-9)
        opamp_h3_ratio = opamp_h[2] / max(opamp_h[0], 1e-9)
        assert opamp_h3_ratio > soft_h3_ratio, (
            f"op_amp H3/H1 ({opamp_h3_ratio:.4f}) should exceed "
            f"soft_clip H3/H1 ({soft_h3_ratio:.4f})"
        )

    def test_fuzz_is_harsher_than_soft_clip(self) -> None:
        signal = _make_mono_sine(440.0, amplitude=0.35)
        soft = apply_transistor(
            signal,
            character="soft_clip",
            drive=1.0,
            mix=1.0,
            multiband=False,
            compensation_mode="none",
        )
        fuzz = apply_transistor(
            signal,
            character="fuzz",
            drive=1.0,
            bias=0.2,
            mix=1.0,
            multiband=False,
            compensation_mode="none",
        )
        # Total harmonic content (2..10) relative to fundamental.
        soft_h = _harmonic_magnitudes(soft, 440.0)
        fuzz_h = _harmonic_magnitudes(fuzz, 440.0)
        soft_thd = sum(h**2 for h in soft_h[1:]) / max(soft_h[0] ** 2, 1e-18)
        fuzz_thd = sum(h**2 for h in fuzz_h[1:]) / max(fuzz_h[0] ** 2, 1e-18)
        assert fuzz_thd > soft_thd * 1.2, (
            f"fuzz THD ({fuzz_thd:.4f}) should exceed soft_clip THD ({soft_thd:.4f})"
        )


class TestDriveMonotonicity:
    def test_output_activity_increases_with_drive(self) -> None:
        signal = _make_mono_sine(440.0, amplitude=0.25)
        dry_rms = _rms(signal)
        diffs = []
        for drive in [0.2, 0.5, 0.8, 1.0]:
            out = apply_transistor(
                signal,
                drive=drive,
                mix=1.0,
                multiband=False,
                compensation_mode="none",
            )
            diffs.append(_rms(out - signal) / max(dry_rms, 1e-9))
        for i in range(len(diffs) - 1):
            assert diffs[i + 1] >= diffs[i] - 1e-6, (
                f"drive monotonicity broken: diffs={diffs}"
            )


class TestDcCleanliness:
    def test_dc_clean_post_block(self) -> None:
        signal = _make_mono_sine(440.0, amplitude=0.35)
        output = apply_transistor(
            signal,
            character="diode",
            drive=0.9,
            bias=0.5,
            mix=1.0,
            multiband=False,
            compensation_mode="none",
        )
        # DC residue relative to peak should be well below -50 dBFS.
        peak = float(np.max(np.abs(output)))
        assert abs(float(np.mean(output))) / max(peak, 1e-9) < 3e-3


class TestMixPassthrough:
    def test_mix_zero_is_passthrough(self) -> None:
        signal = _make_stereo_sine(220.0, amplitude=0.25)
        output = apply_transistor(signal, drive=0.9, mix=0.0)
        np.testing.assert_allclose(output, signal, atol=1e-8)


class TestDeterminism:
    def test_bit_exact_across_calls(self) -> None:
        signal = _make_stereo_sine(440.0, amplitude=0.25)
        a = apply_transistor(signal, preset="tube_screamer")
        b = apply_transistor(signal, preset="tube_screamer")
        np.testing.assert_array_equal(a, b)


class TestTargetThdSolver:
    def test_solver_hits_target_within_tolerance(self) -> None:
        signal = _make_mono_sine(440.0, amplitude=0.25)
        output = apply_transistor(
            signal,
            character="soft_clip",
            target_thd_pct=5.0,
            mix=1.0,
            multiband=False,
            compensation_mode="none",
        )
        assert isinstance(output, np.ndarray)
        assert np.all(np.isfinite(output))
        # Measure the THD that we actually produced.
        mags = _harmonic_magnitudes(output, 440.0)
        thd_pct = 100.0 * np.sqrt(sum(m**2 for m in mags[1:])) / max(mags[0], 1e-9)
        assert abs(thd_pct - 5.0) < 2.5, (
            f"solver missed target: got {thd_pct:.2f}%, wanted 5.0%"
        )


class TestPresetResolution:
    def test_effect_spec_resolves_preset(self) -> None:
        resolved = _resolve_effect_params("transistor", {"preset": "tube_screamer"})
        # Resolved params should contain character, drive, mix etc.
        assert resolved.get("character") == "diode"
        assert "drive" in resolved

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError):
            _resolve_effect_params("transistor", {"preset": "__nope__"})

    def test_effect_spec_construct(self) -> None:
        spec = EffectSpec("transistor", {"preset": "op_amp_clean"})
        assert spec.kind == "transistor"
