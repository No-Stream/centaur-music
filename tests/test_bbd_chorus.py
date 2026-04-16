"""Tests for the native Juno-style BBD chorus effect."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.score import EffectSpec, Score
from code_musics.synth import (
    _BBD_CHORUS_PRESETS,
    _SIMPLE_EFFECT_DISPATCH,
    SAMPLE_RATE,
    apply_bbd_chorus,
    apply_effect_chain,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine_wave(
    frequency_hz: float = 440.0,
    *,
    sample_rate: int = SAMPLE_RATE,
    duration_s: float = 1.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * t)


def _white_noise(
    *, sample_rate: int = SAMPLE_RATE, duration_s: float = 0.5, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sample_rate * duration_s)
    return rng.standard_normal(n) * 0.3


# ---------------------------------------------------------------------------
# BBD chorus tests
# ---------------------------------------------------------------------------


class TestBbdChorus:
    def test_mix_zero_returns_dry(self) -> None:
        signal = _sine_wave(duration_s=0.3)
        result = apply_bbd_chorus(signal, mix=0.0)
        # Output is stereo (mono promoted); both channels should equal dry input
        assert result.ndim == 2 and result.shape[0] == 2
        np.testing.assert_allclose(result[0], signal, atol=1e-9)
        np.testing.assert_allclose(result[1], signal, atol=1e-9)

    def test_produces_stereo_decorrelation_from_mono(self) -> None:
        """Quadrature LFOs must produce distinct L/R wet signals from mono input."""
        signal = _sine_wave(frequency_hz=220.0, duration_s=0.8)
        result = apply_bbd_chorus(signal, mix=0.8)
        assert result.ndim == 2 and result.shape[0] == 2
        left = result[0]
        right = result[1]
        assert not np.allclose(left, right, atol=1e-4), (
            "Quadrature LFOs should decorrelate L/R channels from mono input"
        )

    def test_output_bounded_and_finite(self) -> None:
        """No NaN/Inf and peak bounded even at aggressive settings."""
        noise = _white_noise(duration_s=0.8)
        input_peak = float(np.max(np.abs(noise)))
        result = apply_bbd_chorus(
            noise,
            mix=0.5,
            cross_feedback=0.3,
            depth_ms=4.0,
            center_delay_ms=8.0,
            rate_hz=1.0,
            compander_amount=0.5,
        )
        assert np.all(np.isfinite(result)), "Output must not contain NaN or Inf"
        output_peak = float(np.max(np.abs(result)))
        max_allowed_peak = input_peak * 2.0
        assert output_peak < max_allowed_peak, (
            f"Output peak {output_peak:.3f} exceeds 2x input peak {input_peak:.3f}"
        )

    def test_all_presets_run_cleanly(self) -> None:
        """Every preset must run without error and stay bounded."""
        signal = _sine_wave(frequency_hz=440.0, duration_s=0.4)
        input_peak = float(np.max(np.abs(signal)))
        assert _BBD_CHORUS_PRESETS, "presets dict must not be empty"
        for preset_name in _BBD_CHORUS_PRESETS:
            result = apply_bbd_chorus(signal, preset=preset_name)
            assert np.all(np.isfinite(result)), (
                f"preset {preset_name!r} produced non-finite values"
            )
            assert result.shape[-1] == signal.shape[-1], (
                f"preset {preset_name!r} changed signal length"
            )
            out_peak = float(np.max(np.abs(result)))
            assert out_peak < input_peak * 2.0, (
                f"preset {preset_name!r} exceeded 2x input peak"
            )

    def test_expected_presets_present(self) -> None:
        """Sanity-check that the promised preset names exist."""
        for name in ("juno_i", "juno_ii", "juno_i_plus_ii", "dimension_wide"):
            assert name in _BBD_CHORUS_PRESETS, f"missing preset {name!r}"

    def test_stereo_input_preserved(self) -> None:
        mono = _sine_wave(frequency_hz=330.0, duration_s=0.3)
        stereo = np.stack([mono, mono * 0.9])
        result = apply_bbd_chorus(stereo, mix=0.4)
        assert result.ndim == 2 and result.shape[0] == 2
        assert result.shape[-1] == stereo.shape[-1]

    def test_wet_path_modifies_signal(self) -> None:
        """At non-zero mix, output should differ meaningfully from dry."""
        signal = _sine_wave(frequency_hz=440.0, duration_s=0.5)
        result = apply_bbd_chorus(signal, mix=0.5)
        result_mono = result.mean(axis=0)
        # After the initial offset delay, the wet + dry sum should not equal dry
        settled = result_mono[int(0.05 * SAMPLE_RATE) :]
        dry_settled = signal[int(0.05 * SAMPLE_RATE) :]
        assert not np.allclose(settled, dry_settled, atol=1e-3), (
            "Wet path should audibly modify the signal at mix=0.5"
        )

    def test_rejects_invalid_mix(self) -> None:
        signal = _sine_wave(duration_s=0.2)
        with pytest.raises(ValueError):
            apply_bbd_chorus(signal, mix=-0.1)
        with pytest.raises(ValueError):
            apply_bbd_chorus(signal, mix=1.5)

    def test_rejects_bad_preset(self) -> None:
        signal = _sine_wave(duration_s=0.2)
        with pytest.raises(ValueError):
            apply_bbd_chorus(signal, preset="this_preset_does_not_exist")

    def test_registered_in_simple_dispatch(self) -> None:
        """Effect chain dispatcher must know about the new effect."""
        assert "bbd_chorus" in _SIMPLE_EFFECT_DISPATCH
        assert _SIMPLE_EFFECT_DISPATCH["bbd_chorus"] is apply_bbd_chorus

    def test_integration_via_effect_chain_with_preset(self) -> None:
        """End-to-end: EffectSpec + apply_effect_chain works with a preset."""
        signal = _sine_wave(duration_s=0.4)
        effects = [EffectSpec(kind="bbd_chorus", params={"preset": "juno_i"})]
        result = apply_effect_chain(signal, effects)
        assert np.all(np.isfinite(result))
        assert result.shape[-1] == signal.shape[-1]

    def test_integration_in_score(self) -> None:
        """End-to-end smoke test: a voice with bbd_chorus renders audio."""
        score = Score(f0=220.0, auto_master_gain_stage=False)
        score.add_voice(
            "pad",
            synth_defaults={
                "engine": "polyblep",
                "osc_wave": "saw",
                "cutoff_hz": 1800.0,
            },
            effects=[EffectSpec(kind="bbd_chorus", params={"preset": "juno_ii"})],
        )
        score.add_note("pad", start=0.0, duration=0.8, partial=2.0, amp=0.3)
        audio = score.render()
        assert audio.ndim == 2 and audio.shape[0] == 2
        assert np.all(np.isfinite(audio))
        peak = float(np.max(np.abs(audio)))
        assert peak > 0.0, "rendered audio should not be silent"
        assert peak < 1.5, f"rendered audio peak {peak:.3f} is too hot"

    def test_bandlimiting_trims_high_frequencies(self) -> None:
        """Pre/post LPF should clearly bandlimit the wet contribution.

        Since wet is summed with dry (not crossfaded), the dry HF content is
        preserved in the mix by design. The bandlimiting applies to the wet
        path, so we compare the HF energy of (wet-only) against the dry.
        """
        rng = np.random.default_rng(123)
        n = int(0.5 * SAMPLE_RATE)
        dry = rng.standard_normal(n) * 0.3
        # mix=1 gives dry + wet; subtract dry to recover the wet contribution.
        result = apply_bbd_chorus(dry, mix=1.0, cross_feedback=0.0)
        wet_contrib = result - np.stack([dry, dry])

        def hf_energy(sig: np.ndarray) -> float:
            mono = sig.mean(axis=0) if sig.ndim == 2 else sig
            spec = np.abs(np.fft.rfft(mono))
            freqs = np.fft.rfftfreq(mono.shape[-1], d=1.0 / SAMPLE_RATE)
            mask = freqs > 8000.0
            return float(np.sum(spec[mask] ** 2))

        def total_energy(sig: np.ndarray) -> float:
            mono = sig.mean(axis=0) if sig.ndim == 2 else sig
            spec = np.abs(np.fft.rfft(mono))
            return float(np.sum(spec**2))

        dry_hf_ratio = hf_energy(dry) / total_energy(dry)
        wet_hf_ratio = hf_energy(wet_contrib) / total_energy(wet_contrib)
        # The wet should have a much lower HF ratio than the dry.
        assert wet_hf_ratio < dry_hf_ratio * 0.4, (
            f"BBD wet path should clearly be bandlimited: "
            f"dry HF ratio {dry_hf_ratio:.4f}, wet HF ratio {wet_hf_ratio:.4f}"
        )
