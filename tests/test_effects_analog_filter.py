"""Tests for the native analog filter effect (bus/master/voice wrapper)."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.score import EffectSpec
from code_musics.synth import (
    _SIMPLE_EFFECT_DISPATCH,
    SAMPLE_RATE,
    apply_analog_filter,
    apply_effect_chain,
)

_TOPOLOGIES: tuple[str, ...] = (
    "svf",
    "ladder",
    "sallen_key",
    "cascade",
    "sem",
    "jupiter",
    "k35",
    "diode",
)


def _white_noise(
    *,
    duration_s: float = 1.0,
    sample_rate: int = SAMPLE_RATE,
    amp: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sample_rate * duration_s)
    return (rng.standard_normal(n) * amp).astype(np.float64)


def _sine(
    freq_hz: float = 440.0,
    *,
    duration_s: float = 0.5,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.5,
) -> np.ndarray:
    t = np.linspace(0.0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float64)


def _band_power(signal: np.ndarray, low_hz: float, high_hz: float) -> float:
    """Return the RMS power of ``signal`` inside a frequency band."""
    mono = signal if signal.ndim == 1 else signal.mean(axis=0)
    n = mono.shape[-1]
    spectrum = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return 0.0
    mag = np.abs(spectrum[mask])
    return float(np.sqrt(np.mean(mag * mag)))


class TestAnalogFilterStability:
    """No NaN/clipping explosions on any topology for neutral-ish defaults."""

    @pytest.mark.parametrize("topology", _TOPOLOGIES)
    def test_lp_default_stable(self, topology: str) -> None:
        noise = _white_noise(duration_s=1.0, seed=7)
        out = apply_analog_filter(
            noise,
            cutoff_hz=2_000.0,
            resonance_q=1.2,
            filter_topology=topology,
            mode="lp",
            quality="fast",  # keep the test suite cheap; newton is exercised elsewhere
        )
        assert out.shape == (2, noise.shape[0])
        assert np.all(np.isfinite(out)), f"{topology}: non-finite samples produced"
        peak = float(np.max(np.abs(out)))
        # 0 dBFS ceiling check.  Mild overshoot is fine around resonance, but not
        # an order of magnitude above input — that would indicate instability.
        assert peak <= 1.5, f"{topology}: peak {peak:.3f} exceeds 0 dBFS ceiling"

        # For a clearly-below-Nyquist LP cutoff at 2 kHz against white noise, the
        # output should retain a meaningful fraction of input energy.  Topologies
        # with bass compensation / drive variation will differ somewhat, so we
        # only require a lower bound.
        in_rms = float(np.sqrt(np.mean(noise * noise)))
        out_rms = float(np.sqrt(np.mean(out * out)))
        assert out_rms > 0.05 * in_rms, (
            f"{topology}: output RMS {out_rms:.4f} suspiciously low vs input {in_rms:.4f}"
        )

    @pytest.mark.parametrize("topology", _TOPOLOGIES)
    def test_topology_dispatch_no_crash(self, topology: str) -> None:
        """Iterating all topologies through the effect chain must not crash."""
        noise = _white_noise(duration_s=0.25, seed=11)
        effects = [
            EffectSpec(
                "analog_filter",
                {
                    "filter_topology": topology,
                    "cutoff_hz": 1_500.0,
                    "resonance_q": 1.5,
                    "quality": "fast",
                },
            )
        ]
        out = apply_effect_chain(noise, effects)
        assert out.shape[-1] == noise.shape[0]
        assert np.all(np.isfinite(out))


class TestAnalogFilterStereo:
    """Independent L/R filter states — no cross-channel bleed."""

    def test_asymmetric_stereo_channels_match_independent_mono_renders(self) -> None:
        """Feed two *different* non-zero signals on L/R and assert each
        channel of the stereo output matches an independent mono render
        of the same per-channel input.

        The previous version of this test used R=0, which is a trivial
        case: a zero-input filter produces zero output regardless of
        whether L/R state is shared, so a shared-state bug would slip
        through.  Asymmetric non-zero signals force the per-channel
        state to actually diverge; any coupling would show up as
        measurable difference from the independent-mono reference."""
        left = _sine(freq_hz=220.0, duration_s=0.5, amplitude=0.5)
        right = _sine(freq_hz=880.0, duration_s=0.5, amplitude=0.5)

        # Reference: run each channel's input through apply_analog_filter
        # independently (mono -> stereo upmix — left_ref[0] == left_ref[1]).
        left_ref = apply_analog_filter(
            left,
            cutoff_hz=800.0,
            resonance_q=1.0,
            filter_topology="svf",
            quality="fast",
        )
        right_ref = apply_analog_filter(
            right,
            cutoff_hz=800.0,
            resonance_q=1.0,
            filter_topology="svf",
            quality="fast",
        )

        # Stereo pass: both channels in one call.  Each channel's output
        # must match its independent mono-reference channel to float tolerance.
        stereo_in = np.stack([left, right])
        out = apply_analog_filter(
            stereo_in,
            cutoff_hz=800.0,
            resonance_q=1.0,
            filter_topology="svf",
            quality="fast",
        )
        assert out.shape == (2, left.shape[0])
        np.testing.assert_allclose(
            out[0],
            left_ref[0],
            atol=1e-10,
            err_msg="L channel coupled with R — stereo filter state is not independent",
        )
        np.testing.assert_allclose(
            out[1],
            right_ref[0],
            atol=1e-10,
            err_msg="R channel coupled with L — stereo filter state is not independent",
        )

    def test_mono_passthrough_produces_valid_output(self) -> None:
        """Mono input must produce valid stereo output (upmix preserves stereo API)."""
        noise = _white_noise(duration_s=0.2, seed=17)
        assert noise.ndim == 1
        out = apply_analog_filter(
            noise,
            cutoff_hz=1_500.0,
            filter_topology="ladder",
            quality="fast",
        )
        assert out.ndim == 2 and out.shape[0] == 2
        assert out.shape[1] == noise.shape[0]
        # The two channels should be bit-identical for a mono source — no stereo
        # decorrelation happens inside this effect (that's what chorus is for).
        np.testing.assert_allclose(out[0], out[1], atol=1e-12)


class TestAnalogFilterAutomation:
    """Per-sample cutoff automation migrates spectral energy correctly."""

    def test_cutoff_sweep_raises_high_band_energy(self) -> None:
        """Sweeping cutoff from 100 Hz to 10 kHz over 2 s should move energy up."""
        duration_s = 2.0
        n = int(SAMPLE_RATE * duration_s)
        noise = _white_noise(duration_s=duration_s, seed=23)

        # Exponential sweep (frequency perception is log — sample this like we
        # expect automation to express it in pieces).
        cutoff = np.geomspace(100.0, 10_000.0, n).astype(np.float64)

        out = apply_analog_filter(
            noise,
            cutoff_hz=cutoff,
            resonance_q=0.9,
            filter_topology="svf",
            mode="lp",
            quality="fast",
        )

        # Compare band power of the first half (cutoff 100 -> ~1000 Hz) vs the
        # second half (cutoff ~1000 -> 10000 Hz).  The second-half output
        # should show materially more energy in the 2–8 kHz band than the first.
        half = out[:, : n // 2]
        second = out[:, n // 2 :]

        early_high = _band_power(half, 2_000.0, 8_000.0)
        late_high = _band_power(second, 2_000.0, 8_000.0)

        assert late_high > early_high * 2.0, (
            f"cutoff sweep did not migrate energy upward: "
            f"early_high={early_high:.4f} late_high={late_high:.4f}"
        )


class TestAnalogFilterWiring:
    """Integration with the EffectSpec dispatch surface."""

    def test_registered_in_simple_dispatch(self) -> None:
        assert "analog_filter" in _SIMPLE_EFFECT_DISPATCH

    def test_wet_dry_mix_interpolates(self) -> None:
        """mix=0 returns dry; mix=1 returns wet; intermediate blends cleanly."""
        signal = _sine(freq_hz=2_000.0, duration_s=0.2, amplitude=0.5)
        stereo = np.stack([signal, signal])

        dry_out = apply_analog_filter(
            stereo,
            cutoff_hz=200.0,  # strong LP — aggressive attenuation of 2 kHz
            resonance_q=0.8,
            filter_topology="svf",
            mix=0.0,
            quality="fast",
        )
        wet_out = apply_analog_filter(
            stereo,
            cutoff_hz=200.0,
            resonance_q=0.8,
            filter_topology="svf",
            mix=1.0,
            quality="fast",
        )

        # mix=0 should equal input (fully dry).
        np.testing.assert_allclose(dry_out, stereo, atol=1e-9)
        # mix=1 must be materially different from dry (LP is doing something).
        assert not np.allclose(wet_out, stereo, atol=1e-3)

    def test_quality_modes_all_run(self) -> None:
        """All four quality modes produce valid output."""
        noise = _white_noise(duration_s=0.2, seed=31)
        for quality in ("draft", "fast", "great", "divine"):
            out = apply_analog_filter(
                noise,
                cutoff_hz=1_200.0,
                resonance_q=1.5,
                filter_topology="ladder",
                quality=quality,
            )
            assert np.all(np.isfinite(out)), f"quality={quality} produced non-finite"
            assert out.shape == (2, noise.shape[0])

    def test_k35_asymmetry_accepted(self) -> None:
        """K35-specific k35_feedback_asymmetry must be honoured (no param error)."""
        noise = _white_noise(duration_s=0.2, seed=37)
        out = apply_analog_filter(
            noise,
            cutoff_hz=900.0,
            resonance_q=4.0,
            filter_topology="k35",
            filter_drive=0.4,
            k35_feedback_asymmetry=0.7,
            quality="fast",
        )
        assert np.all(np.isfinite(out))
        assert out.shape == (2, noise.shape[0])

    def test_dual_filter_hpf_slot(self) -> None:
        """CS80/Jupiter-8 serial HPF slot attenuates low-band energy."""
        noise = _white_noise(duration_s=0.4, seed=41)
        without_hpf = apply_analog_filter(
            noise,
            cutoff_hz=6_000.0,
            resonance_q=0.9,
            filter_topology="jupiter",
            hpf_cutoff_hz=0.0,
            quality="fast",
        )
        with_hpf = apply_analog_filter(
            noise,
            cutoff_hz=6_000.0,
            resonance_q=0.9,
            filter_topology="jupiter",
            hpf_cutoff_hz=500.0,
            quality="fast",
        )
        low_without = _band_power(without_hpf, 40.0, 200.0)
        low_with = _band_power(with_hpf, 40.0, 200.0)
        assert low_with < low_without * 0.7, (
            f"serial HPF should cut low-band energy: "
            f"without={low_without:.4f} with={low_with:.4f}"
        )

    def test_invalid_topology_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported filter_topology"):
            apply_analog_filter(
                _white_noise(duration_s=0.1),
                filter_topology="not_a_real_topology",
            )

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported filter mode"):
            apply_analog_filter(
                _white_noise(duration_s=0.1),
                mode="bandreject",
            )

    def test_invalid_mix_raises(self) -> None:
        with pytest.raises(ValueError, match="mix"):
            apply_analog_filter(
                _white_noise(duration_s=0.1),
                mix=1.5,
            )

    def test_mismatched_cutoff_curve_raises(self) -> None:
        """Mismatched-length per-sample cutoff curves must fail fast.

        Silent resampling would mask wiring bugs in automation curves;
        callers must supply a curve of the correct length (or use the
        AutomationSpec surface which generates curves at the exact signal
        length at render time).

        Covers both mono and stereo inputs, and asserts the error message
        calls out the length-mismatch condition explicitly so debugging
        surface-level automation bugs isn't a guessing game."""
        mono_noise = _white_noise(duration_s=0.1, seed=43)
        short_curve = np.full(10, 1_000.0, dtype=np.float64)
        long_curve = np.full(mono_noise.shape[0] * 2, 1_000.0, dtype=np.float64)

        # Mono input with too-short curve.
        with pytest.raises(ValueError, match="must match signal length") as exc_info:
            apply_analog_filter(mono_noise, cutoff_hz=short_curve)
        msg = str(exc_info.value)
        assert "cutoff_hz" in msg or "curve" in msg, (
            f"error message should mention cutoff_hz or curve: {msg!r}"
        )

        # Mono input with too-long curve — same error, symmetric failure mode.
        with pytest.raises(ValueError, match="must match signal length"):
            apply_analog_filter(mono_noise, cutoff_hz=long_curve)

        # Stereo input with mismatched curve length.  The stereo path
        # shares the same curve-length validation as mono — this guards
        # against channel-count branching that skips validation.
        stereo_noise = np.stack([mono_noise, mono_noise])
        with pytest.raises(ValueError, match="must match signal length"):
            apply_analog_filter(stereo_noise, cutoff_hz=short_curve)

    def test_empty_cutoff_curve_raises(self) -> None:
        noise = _white_noise(duration_s=0.1, seed=44)
        with pytest.raises(ValueError, match="cutoff_hz per-sample curve is empty"):
            apply_analog_filter(noise, cutoff_hz=np.array([], dtype=np.float64))

    def test_nonfinite_cutoff_raises(self) -> None:
        noise = _white_noise(duration_s=0.1, seed=45)
        with pytest.raises(ValueError, match="non-finite"):
            apply_analog_filter(noise, cutoff_hz=float("nan"))
