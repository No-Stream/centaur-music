"""End-to-end tests for the Koren/pentode/HG2/Culture-Vulture tube effect."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.score import EffectSpec
from code_musics.synth import (
    _TUBE_PRESETS,
    SAMPLE_RATE,
    _resolve_effect_params,
    apply_tube,
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
    """Every preset renders without NaN/Inf on a standard sine probe."""

    @pytest.mark.parametrize("preset_name", list(_TUBE_PRESETS.keys()))
    def test_preset_renders_finite(self, preset_name: str) -> None:
        signal = _make_stereo_sine(440.0, duration_s=1.0, amplitude=0.25)
        output = apply_tube(signal, preset=preset_name)
        assert isinstance(output, np.ndarray)
        assert output.shape == signal.shape
        assert np.all(np.isfinite(output))
        assert _rms(output) > 1e-6


class TestHarmonicSignatures:
    """Each character should produce the expected H2/H3 balance."""

    def test_triode_h2_dominates_h3(self) -> None:
        signal = _make_mono_sine(440.0, amplitude=0.35)
        output = apply_tube(
            signal,
            character="triode",
            drive=0.6,
            bias=0.0,
            mix=1.0,
            multiband=False,
            compensation_mode="none",
        )
        h = _harmonic_magnitudes(output, 440.0)
        assert h[1] > h[2] * 1.1, (
            f"triode should be H2-dominant; got h2={h[1]:.4g} h3={h[2]:.4g}"
        )

    def test_pentode_h3_dominates_h2(self) -> None:
        signal = _make_mono_sine(440.0, amplitude=0.35)
        output = apply_tube(
            signal,
            character="pentode",
            drive=0.6,
            mix=1.0,
            multiband=False,
            compensation_mode="none",
        )
        h = _harmonic_magnitudes(output, 440.0)
        assert h[2] > h[1] * 1.5, (
            f"pentode should be H3-dominant; got h2={h[1]:.4g} h3={h[2]:.4g}"
        )

    def test_culture_biased_is_asymmetric(self) -> None:
        signal = _make_mono_sine(440.0, amplitude=0.35)
        output = apply_tube(
            signal,
            character="culture",
            drive=0.6,
            bias=0.6,
            mix=1.0,
            multiband=False,
            compensation_mode="none",
        )
        h = _harmonic_magnitudes(output, 440.0)
        # H2 should dominate H3 for the asymmetric culture character.
        assert h[1] > h[2] * 1.1, (
            f"biased culture should be H2-dominant; got h2={h[1]:.4g} h3={h[2]:.4g}"
        )
        # Verify time-domain asymmetry.
        pos_peak = float(np.max(output))
        neg_peak = float(-np.min(output))
        ratio = max(pos_peak, neg_peak) / max(min(pos_peak, neg_peak), 1e-9)
        assert ratio > 1.15, (
            f"biased culture should show asymmetric peaks; ratio={ratio:.3f}"
        )


class TestBiasStarvation:
    def test_culture_starve_collapses_one_lobe(self) -> None:
        signal = _make_mono_sine(440.0, amplitude=0.35)
        output = apply_tube(
            signal,
            character="culture",
            drive=0.9,
            bias=0.95,
            mix=1.0,
            multiband=False,
            compensation_mode="none",
        )
        pos_peak = float(np.max(output))
        neg_peak = float(-np.min(output))
        ratio = max(pos_peak, neg_peak) / max(min(pos_peak, neg_peak), 1e-9)
        assert ratio > 2.0, (
            f"Culture-Vulture starvation should collapse one lobe; "
            f"ratio={ratio:.3f} (pos={pos_peak:.4f}, neg={neg_peak:.4f})"
        )


class TestDcCleanliness:
    """Even at extreme bias, post-DC-block output mean is near zero."""

    def test_dc_clean_at_extreme_bias(self) -> None:
        signal = _make_mono_sine(440.0, amplitude=0.35)
        output = apply_tube(
            signal,
            character="culture",
            drive=0.9,
            bias=0.95,
            mix=1.0,
            multiband=False,
            compensation_mode="none",
        )
        # DC residue relative to peak should be well below -50 dBFS
        # even at the extreme starvation bias.
        peak = float(np.max(np.abs(output)))
        assert abs(float(np.mean(output))) / max(peak, 1e-9) < 3e-3


class TestHg2CascadeOrder:
    """Swapping pentode/triode drive on HG2 produces audibly different spectra."""

    def test_cascade_order_affects_spectrum(self) -> None:
        signal = _make_mono_sine(440.0, amplitude=0.35)
        pent_heavy = apply_tube(
            signal,
            character="hg2",
            pentode_drive=0.7,
            triode_drive=0.2,
            parallel_drive=0.0,
            mix=1.0,
            multiband=False,
            compensation_mode="none",
        )
        tri_heavy = apply_tube(
            signal,
            character="hg2",
            pentode_drive=0.2,
            triode_drive=0.7,
            parallel_drive=0.0,
            mix=1.0,
            multiband=False,
            compensation_mode="none",
        )
        pent_h = _harmonic_magnitudes(pent_heavy, 440.0)
        tri_h = _harmonic_magnitudes(tri_heavy, 440.0)

        # Level-normalize H1 to compare harmonic profiles fairly.
        def _profile(hs: list[float]) -> list[float]:
            norm = max(hs[0], 1e-9)
            return [h / norm for h in hs]

        pent_prof = _profile(pent_h)
        tri_prof = _profile(tri_h)
        # Difference in H2/H3 relative profile should be substantial.
        diff = sum(
            abs(p - t) for p, t in zip(pent_prof[1:4], tri_prof[1:4], strict=False)
        )
        assert diff > 0.05, (
            f"pentode-heavy and triode-heavy cascade should differ; "
            f"summed profile delta={diff:.4f}"
        )


class TestMultibandBypass:
    """Multiband splits keep bass/air out of the shaper by default."""

    def test_bass_bypass_preserves_50hz_sine(self) -> None:
        signal = _make_mono_sine(50.0, duration_s=1.0, amplitude=0.35)
        wet_multiband = apply_tube(
            signal,
            character="triode",
            drive=0.8,
            mix=1.0,
            multiband=True,
            low_crossover_hz=120.0,
            high_crossover_hz=5000.0,
            compensation_mode="none",
        )
        wet_full = apply_tube(
            signal,
            character="triode",
            drive=0.8,
            mix=1.0,
            multiband=False,
            compensation_mode="none",
        )
        dry_rms = _rms(signal)
        mb_rms = _rms(wet_multiband)
        full_rms = _rms(wet_full)
        mb_db = 20.0 * np.log10(max(mb_rms, 1e-12) / max(dry_rms, 1e-12))
        # Multiband bass bypass should stay close to the dry level.
        assert abs(mb_db) < 2.0, (
            f"multiband bass should bypass cleanly; mb vs dry {mb_db:.2f} dB"
        )
        # Non-multiband path should produce clearly different shape.
        assert not np.allclose(wet_multiband, wet_full, atol=1e-3)
        assert full_rms > 1e-6


class TestMixPassthrough:
    def test_mix_zero_is_passthrough(self) -> None:
        signal = _make_stereo_sine(220.0, amplitude=0.25)
        output = apply_tube(signal, drive=0.9, mix=0.0)
        np.testing.assert_allclose(output, signal, atol=1e-8)


class TestDeterminism:
    def test_bit_exact_across_calls(self) -> None:
        signal = _make_stereo_sine(440.0, amplitude=0.25)
        a = apply_tube(signal, preset="triode_glow")
        b = apply_tube(signal, preset="triode_glow")
        np.testing.assert_array_equal(a, b)


class TestPresetResolution:
    def test_effect_spec_resolves_preset(self) -> None:
        resolved = _resolve_effect_params("tube", {"preset": "triode_glow"})
        assert resolved.get("character") == "triode"
        assert "drive" in resolved

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError):
            _resolve_effect_params("tube", {"preset": "__nope__"})

    def test_effect_spec_construct(self) -> None:
        spec = EffectSpec("tube", {"preset": "hg2_enhancer"})
        assert spec.kind == "tube"
