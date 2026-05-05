"""Tests for the Linkwitz-Riley multiband crossover bypass on apply_drive.

See `apply_drive(..., multiband=True, low_crossover_hz=..., high_crossover_hz=...)`.

The multiband path splits the input into three LR4 bands and only the
mid band is routed through the nonlinearity.  Bass and treble bypass the
shaper entirely so the effect does not pile harmonics into the sub or
lift the 2-8 kHz band on drum content.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np
import pytest

from code_musics.synth import SAMPLE_RATE, apply_drive


def _drive(signal: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Type-narrowed wrapper for apply_drive returning an ndarray."""
    kwargs.setdefault("return_analysis", False)
    result = apply_drive(signal, **kwargs)
    return cast(np.ndarray, result)


def _sine(freq_hz: float, duration_s: float, amp: float = 1.0) -> np.ndarray:
    n = int(duration_s * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    return np.asarray(amp * np.sin(2.0 * np.pi * freq_hz * t), dtype=np.float64)


def _band_energy(
    signal: np.ndarray,
    *,
    low_hz: float,
    high_hz: float,
    sample_rate: int = SAMPLE_RATE,
) -> float:
    """Energy in a given band via rFFT."""
    mono = signal if signal.ndim == 1 else np.mean(signal, axis=0)
    spectrum = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(mono.shape[-1], 1.0 / sample_rate)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    return float(np.sum(np.abs(spectrum[mask]) ** 2))


def _thd_in_band(
    signal: np.ndarray,
    fundamental_hz: float,
    *,
    low_hz: float,
    high_hz: float,
    sample_rate: int = SAMPLE_RATE,
) -> float:
    """Approximate in-band harmonic distortion percentage.

    Sums harmonic energy (2f, 3f, ..., up to band edge) vs. fundamental
    energy, restricted to the specified band.
    """
    mono = signal if signal.ndim == 1 else np.mean(signal, axis=0)
    spectrum = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(mono.shape[-1], 1.0 / sample_rate)
    bin_hz = float(freqs[1] - freqs[0])

    def _peak_around(f: float) -> float:
        lo = max(0.0, f - bin_hz * 3.0)
        hi = f + bin_hz * 3.0
        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            return 0.0
        return float(np.max(np.abs(spectrum[mask])))

    fundamental_mag = _peak_around(fundamental_hz)
    if fundamental_mag <= 0.0:
        return 0.0

    harmonic_sq = 0.0
    n = 2
    while fundamental_hz * n < high_hz and n < 40:
        harmonic_freq = fundamental_hz * n
        if low_hz <= harmonic_freq <= high_hz:
            harmonic_sq += _peak_around(harmonic_freq) ** 2
        n += 1
    return 100.0 * float(np.sqrt(harmonic_sq) / fundamental_mag)


def _peak_steady(signal: np.ndarray, skip_samples: int) -> float:
    tail = signal[skip_samples:]
    return float(np.max(np.abs(tail)))


class TestMultibandReconstruction:
    """LR4 multiband split should reconstruct the input at unity when nothing
    is actually processed (mix=0 or drive on mid produces near-unity)."""

    def test_mix_zero_passthrough(self) -> None:
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(SAMPLE_RATE).astype(np.float64) * 0.2
        out = _drive(
            signal,
            drive=2.0,
            mix=0.0,
            multiband=True,
            compensation_mode="none",
        )
        diff = np.subtract(out, signal)
        rms_err = float(np.sqrt(np.mean(diff * diff)))
        assert rms_err < 1e-10, f"mix=0 should be identity, got RMS err {rms_err:.2e}"

    def test_lr4_split_sums_to_flat_before_processing(self) -> None:
        """The LR4 split machinery, on its own, should sum to flat (unity
        magnitude) for sinusoids across the band.  We verify this directly
        against the LR4 helpers without going through the nonlinearity —
        `apply_drive` itself does not have a true unity floor, so we can't
        use it to probe reconstruction at drive=1.
        """
        from code_musics.synth import (
            _linkwitz_riley_highpass,
            _linkwitz_riley_lowpass,
        )

        for f in (50.0, 200.0, 1_000.0, 3_000.0, 8_000.0):
            signal = _sine(f, 0.5, amp=0.1)
            low = _linkwitz_riley_lowpass(signal, 120.0, SAMPLE_RATE)
            mid = _linkwitz_riley_lowpass(
                _linkwitz_riley_highpass(signal, 120.0, SAMPLE_RATE),
                5_000.0,
                SAMPLE_RATE,
            )
            high = _linkwitz_riley_highpass(signal, 5_000.0, SAMPLE_RATE)
            rec = low + mid + high
            skip = int(0.1 * SAMPLE_RATE)
            amp_in = _peak_steady(signal, skip)
            amp_out = _peak_steady(rec, skip)
            db_err = 20.0 * np.log10(amp_out / amp_in)
            assert abs(db_err) < 0.5, f"LR4 sum at {f} Hz: {db_err:.2f} dB"


class TestBassBypass:
    """Bass below low_crossover_hz should bypass the nonlinearity."""

    def test_bass_thd_is_low(self) -> None:
        # 60 Hz sine at -6 dBFS, heavy drive, multiband ON with fc_low=120.
        # drive=1.5 is "musical saturation / fuzz border" under the
        # post-unity-rescale semantics — heavy enough to make THD reduction
        # via bass bypass clearly measurable, but not fuzz territory.
        signal = _sine(60.0, 1.0, amp=0.5)
        out_mb = _drive(
            signal,
            drive=1.5,
            mix=1.0,
            multiband=True,
            low_crossover_hz=120.0,
            high_crossover_hz=5_000.0,
            compensation_mode="none",
        )
        thd_mb = _thd_in_band(out_mb, 60.0, low_hz=40.0, high_hz=200.0)

        out_mono = _drive(
            signal,
            drive=1.5,
            mix=1.0,
            multiband=False,
            preserve_lows_hz=0.0,
            preserve_highs_hz=0.0,
            compensation_mode="none",
        )
        thd_mono = _thd_in_band(out_mono, 60.0, low_hz=40.0, high_hz=200.0)

        assert thd_mb < thd_mono, (
            f"multiband should reduce bass THD: mb={thd_mb:.2f}% vs mono={thd_mono:.2f}%"
        )
        assert thd_mb < 5.0, f"multiband bass THD too high: {thd_mb:.2f}%"


class TestHighBypass:
    """Highs above high_crossover_hz should bypass the nonlinearity."""

    def test_high_sine_preservation(self) -> None:
        """A sine far above the high crossover (12 kHz vs 5 kHz crossover)
        should pass through the multiband path with much less amplitude
        change than the monolithic path.

        `_apply_drive_modern` does not have a true unity floor — even drive=1
        imparts ~+6 dB of voicing gain — so the test compares multiband
        against monolithic rather than against literal unity.
        """
        signal = _sine(12_000.0, 0.5, amp=0.5)
        out_mb = _drive(
            signal,
            drive=3.0,
            mix=1.0,
            multiband=True,
            low_crossover_hz=120.0,
            high_crossover_hz=5_000.0,
            fidelity=0.0,
            compensation_mode="none",
        )
        out_mono = _drive(
            signal,
            drive=3.0,
            mix=1.0,
            multiband=False,
            preserve_lows_hz=0.0,
            preserve_highs_hz=0.0,
            fidelity=0.0,
            compensation_mode="none",
        )
        skip = int(0.1 * SAMPLE_RATE)
        amp_in = _peak_steady(signal, skip)
        db_mb = 20.0 * np.log10(_peak_steady(out_mb, skip) / amp_in)
        db_mono = 20.0 * np.log10(_peak_steady(out_mono, skip) / amp_in)
        # Multiband should be noticeably closer to unity than monolithic.
        assert abs(db_mb) < abs(db_mono) - 3.0, (
            f"multiband should preserve 12 kHz much better than monolithic: "
            f"mb={db_mb:+.2f} dB mono={db_mono:+.2f} dB"
        )
        # And absolute multiband drift should be modest (< 3 dB) since
        # 12 kHz is well into the bypass band.
        assert abs(db_mb) < 3.0, f"multiband 12 kHz drifted: {db_mb:+.2f} dB"


class TestMidBandDrives:
    """The mid band should still be fully driven when multiband=True."""

    def test_mid_sine_thd(self) -> None:
        signal = _sine(1_000.0, 0.5, amp=0.5)
        out = _drive(
            signal,
            drive=3.0,
            mix=1.0,
            multiband=True,
            low_crossover_hz=120.0,
            high_crossover_hz=5_000.0,
            fidelity=0.0,
            compensation_mode="none",
        )
        thd = _thd_in_band(out, 1_000.0, low_hz=500.0, high_hz=10_000.0)
        assert thd > 5.0, f"mid band should distort: THD={thd:.2f}%"


class TestMultibandVsMonolithicDrumContent:
    """On realistic drum-bus content, multiband should show less 2-8 kHz lift."""

    def test_drum_noise_burst(self) -> None:
        rng = np.random.default_rng(1234)
        n = SAMPLE_RATE
        noise = rng.standard_normal(n).astype(np.float64)
        from code_musics.synth import highpass, lowpass

        shaped = highpass(
            lowpass(noise, 2_000.0, SAMPLE_RATE, order=4),
            80.0,
            SAMPLE_RATE,
            order=4,
        )
        peak = float(np.max(np.abs(shaped)))
        shaped = shaped * (10.0 ** (-12.0 / 20.0) / max(peak, 1e-9))

        input_high_band = _band_energy(shaped, low_hz=2_000.0, high_hz=8_000.0)

        out_mb = _drive(
            shaped,
            drive=0.5,
            mix=0.33,
            multiband=True,
            low_crossover_hz=120.0,
            high_crossover_hz=5_000.0,
            compensation_mode="none",
        )
        mb_high_band = _band_energy(out_mb, low_hz=2_000.0, high_hz=8_000.0)

        out_mono = _drive(
            shaped,
            drive=0.5,
            mix=0.33,
            multiband=False,
            preserve_lows_hz=0.0,
            preserve_highs_hz=0.0,
            compensation_mode="none",
        )
        mono_high_band = _band_energy(out_mono, low_hz=2_000.0, high_hz=8_000.0)

        mb_lift = mb_high_band - input_high_band
        mono_lift = mono_high_band - input_high_band
        assert mb_lift < mono_lift, (
            f"multiband should lift 2-8 kHz less than monolithic: "
            f"mb_lift={mb_lift:.4f} mono_lift={mono_lift:.4f}"
        )
        if mono_lift > 0:
            ratio = mb_lift / mono_lift
            assert ratio < 0.8, (
                f"multiband 2-8 kHz lift ratio vs monolithic {ratio:.2f} (want <0.8)"
            )


class TestLegacyPassthrough:
    """multiband=False must give bit-identical output to the pre-multiband path."""

    def test_bit_identical_to_reference(self) -> None:
        from code_musics.synth import _apply_drive_modern

        rng = np.random.default_rng(0xBADD)
        signal = rng.standard_normal(SAMPLE_RATE // 2).astype(np.float64) * 0.2

        ref_raw = _apply_drive_modern(
            signal,
            drive=1.5,
            mix=0.4,
            mode="tube",
            tone=0.1,
            fidelity=0.7,
            bias=0.1,
            even_harmonics=0.2,
            oversample_factor=4,
            highpass_hz=30.0,
            tone_tilt=0.1,
            output_lowpass_hz=0.0,
            preserve_lows_hz=120.0,
            preserve_highs_hz=6_000.0,
            compensation_mode="auto",
            output_trim_db=0.0,
            return_analysis=False,
        )
        ref = cast(np.ndarray, ref_raw)

        via_public = _drive(
            signal,
            drive=1.5,
            mix=0.4,
            multiband=False,
            mode="tube",
            tone=0.1,
            fidelity=0.7,
            bias=0.1,
            even_harmonics=0.2,
            oversample_factor=4,
            highpass_hz=30.0,
            tone_tilt=0.1,
            output_lowpass_hz=0.0,
            preserve_lows_hz=120.0,
            preserve_highs_hz=6_000.0,
            compensation_mode="auto",
            output_trim_db=0.0,
        )
        diff = np.subtract(via_public, ref)
        rms_delta = float(np.sqrt(np.mean(diff * diff)))
        assert rms_delta < 1e-10, (
            f"multiband=False drifted from reference: {rms_delta:.2e}"
        )


class TestDeprecationWarning:
    """preserve_lows_hz / preserve_highs_hz should warn under multiband=True."""

    def test_preserve_lows_warns(self) -> None:
        signal = _sine(200.0, 0.1, amp=0.2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = _drive(
                signal,
                drive=1.5,
                mix=0.3,
                multiband=True,
                preserve_lows_hz=100.0,
                compensation_mode="none",
            )
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert any("preserve_lows_hz" in str(w.message) for w in deprecations), (
            f"expected DeprecationWarning for preserve_lows_hz, got "
            f"{[str(w.message) for w in deprecations]}"
        )

    def test_preserve_highs_warns(self) -> None:
        signal = _sine(200.0, 0.1, amp=0.2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = _drive(
                signal,
                drive=1.5,
                mix=0.3,
                multiband=True,
                preserve_highs_hz=4_500.0,
                compensation_mode="none",
            )
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert any("preserve_highs_hz" in str(w.message) for w in deprecations)

    def test_preserve_lows_forwards_to_low_crossover(self) -> None:
        # With preserve_lows_hz=250 (forwarded), 150 Hz falls under the
        # bypass — should show lower THD than default 120 Hz crossover.
        signal = _sine(150.0, 0.5, amp=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            default_out = _drive(
                signal,
                drive=5.0,
                mix=1.0,
                multiband=True,
                low_crossover_hz=120.0,
                compensation_mode="none",
            )
            forwarded_out = _drive(
                signal,
                drive=5.0,
                mix=1.0,
                multiband=True,
                preserve_lows_hz=250.0,
                compensation_mode="none",
            )
        default_thd = _thd_in_band(default_out, 150.0, low_hz=80.0, high_hz=800.0)
        forwarded_thd = _thd_in_band(forwarded_out, 150.0, low_hz=80.0, high_hz=800.0)
        assert forwarded_thd < default_thd, (
            f"preserve_lows_hz=250 should bypass more bass: "
            f"forwarded_thd={forwarded_thd:.2f}% vs default={default_thd:.2f}%"
        )

    def test_multiband_false_does_not_warn(self) -> None:
        signal = _sine(200.0, 0.1, amp=0.2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = _drive(
                signal,
                drive=1.5,
                mix=0.3,
                multiband=False,
                preserve_lows_hz=100.0,
                preserve_highs_hz=4_500.0,
                compensation_mode="none",
            )
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert not deprecations, (
            f"multiband=False should not warn, got: "
            f"{[str(w.message) for w in deprecations]}"
        )


class TestCrossoverValidation:
    def test_rejects_reversed_crossovers(self) -> None:
        signal = _sine(200.0, 0.1, amp=0.2)
        with pytest.raises(ValueError, match="low_crossover_hz must be below"):
            _ = _drive(
                signal,
                multiband=True,
                low_crossover_hz=5_000.0,
                high_crossover_hz=120.0,
            )

    def test_accepts_zero_disables_bypass(self) -> None:
        signal = _sine(1_000.0, 0.2, amp=0.3)
        out = _drive(
            signal,
            drive=2.0,
            mix=0.5,
            multiband=True,
            low_crossover_hz=0.0,
            high_crossover_hz=0.0,
            compensation_mode="none",
        )
        assert np.isfinite(out).all()
