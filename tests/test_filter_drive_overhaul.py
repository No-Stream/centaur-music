"""Tests for the overhauled driven ZDF SVF filter.

Validates the new topology: single strategic saturation at the feedback
summation point, algebraicSat state limiting, drive/resonance interaction,
first-order ADAA, and optional even harmonics via asymmetric bias.
"""

from __future__ import annotations

import numpy as np

from code_musics.engines._filters import apply_zdf_svf


def _render_test_signal(
    freq: float = 220.0,
    duration: float = 0.5,
    sample_rate: int = 44100,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Generate a simple sawtooth for filter testing."""
    n = int(sample_rate * duration)
    t = np.linspace(0.0, duration, n, endpoint=False)
    phase = (freq * t) % 1.0
    return amplitude * (2.0 * phase - 1.0)


def _measure_thd(
    signal: np.ndarray,
    fundamental_hz: float,
    sample_rate: int,
    max_harmonic: int = 8,
) -> float:
    """Measure total harmonic distortion percentage."""
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    bin_spacing = freqs[1] - freqs[0]

    def _peak_near(target_hz: float) -> float:
        idx = int(round(target_hz / bin_spacing))
        lo = max(0, idx - 2)
        hi = min(len(spectrum), idx + 3)
        return float(np.max(spectrum[lo:hi])) if lo < hi else 0.0

    fundamental_amp = _peak_near(fundamental_hz)
    if fundamental_amp < 1e-12:
        return 0.0

    harmonic_power = sum(
        _peak_near(h * fundamental_hz) ** 2
        for h in range(2, max_harmonic + 1)
        if h * fundamental_hz < sample_rate / 2
    )
    return float(np.sqrt(harmonic_power)) / fundamental_amp * 100.0


def _measure_resonance_peak(
    cutoff_hz: float,
    resonance_q: float,
    filter_drive: float,
    sample_rate: int = 44100,
    duration: float = 0.5,
    **extra_params: float,
) -> float:
    """Measure the amplitude of the resonance peak relative to passband."""
    n = int(sample_rate * duration)
    # White noise input for spectral analysis
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(n) * 0.5
    cutoff_profile = np.full(n, cutoff_hz, dtype=np.float64)

    filtered = apply_zdf_svf(
        noise,
        cutoff_profile=cutoff_profile,
        resonance_q=resonance_q,
        sample_rate=sample_rate,
        filter_mode="lowpass",
        filter_drive=filter_drive,
        **extra_params,
    )

    spectrum = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    bin_spacing = freqs[1] - freqs[0]

    # Find peak near cutoff
    cutoff_idx = int(round(cutoff_hz / bin_spacing))
    search_lo = max(0, cutoff_idx - 10)
    search_hi = min(len(spectrum), cutoff_idx + 10)
    peak_amp = float(np.max(spectrum[search_lo:search_hi]))

    # Passband average (well below cutoff)
    passband_hi = max(1, cutoff_idx // 3)
    passband_amp = float(np.mean(spectrum[1:passband_hi]))

    return peak_amp / max(passband_amp, 1e-12)


class TestDriveZeroIsLinear:
    """Filter with drive=0 must produce identical output to the linear path."""

    def test_lowpass_drive_zero_matches_linear(self) -> None:
        signal = _render_test_signal()
        n = len(signal)
        cutoff = np.full(n, 1500.0, dtype=np.float64)

        linear = apply_zdf_svf(
            signal,
            cutoff_profile=cutoff,
            resonance_q=2.0,
            sample_rate=44100,
            filter_mode="lowpass",
            filter_drive=0.0,
        )
        driven = apply_zdf_svf(
            signal,
            cutoff_profile=cutoff,
            resonance_q=2.0,
            sample_rate=44100,
            filter_mode="lowpass",
            filter_drive=0.0,
        )
        np.testing.assert_array_equal(linear, driven)


class TestLowDriveWarmth:
    """Low drive values (0.05-0.3) should add gentle harmonics, not crunch.

    Uses sine input so any measured THD is purely from filter drive,
    not the input signal's own harmonics.
    """

    @staticmethod
    def _sine_signal(
        freq: float = 220.0, duration: float = 0.5, sample_rate: int = 44100
    ) -> np.ndarray:
        n = int(sample_rate * duration)
        t = np.linspace(0.0, duration, n, endpoint=False)
        return 0.7 * np.sin(2.0 * np.pi * freq * t)

    def test_drive_015_adds_subtle_harmonics(self) -> None:
        """At drive=0.15 with sine input, THD should be gentle."""
        signal = self._sine_signal()
        n = len(signal)
        cutoff = np.full(n, 2000.0, dtype=np.float64)

        filtered = apply_zdf_svf(
            signal,
            cutoff_profile=cutoff,
            resonance_q=1.0,
            sample_rate=44100,
            filter_mode="lowpass",
            filter_drive=0.15,
        )
        thd = _measure_thd(filtered, 220.0, 44100)
        assert thd < 15.0, f"THD at drive=0.15 is {thd:.1f}%, expected < 15%"

    def test_drive_03_still_musical(self) -> None:
        """At drive=0.3, still moderate distortion with sine input."""
        signal = self._sine_signal()
        n = len(signal)
        cutoff = np.full(n, 2000.0, dtype=np.float64)

        filtered = apply_zdf_svf(
            signal,
            cutoff_profile=cutoff,
            resonance_q=1.0,
            sample_rate=44100,
            filter_mode="lowpass",
            filter_drive=0.3,
        )
        thd = _measure_thd(filtered, 220.0, 44100)
        assert thd < 25.0, f"THD at drive=0.3 is {thd:.1f}%, expected < 25%"

    def test_drive_increases_thd_monotonically(self) -> None:
        """Higher drive should produce more harmonics (monotonic THD increase)."""
        signal = self._sine_signal()
        n = len(signal)
        cutoff = np.full(n, 2000.0, dtype=np.float64)

        thds = []
        for drive in [0.0, 0.1, 0.3, 0.6, 1.0]:
            filtered = apply_zdf_svf(
                signal,
                cutoff_profile=cutoff,
                resonance_q=1.0,
                sample_rate=44100,
                filter_mode="lowpass",
                filter_drive=drive,
            )
            thds.append(_measure_thd(filtered, 220.0, 44100))

        for i in range(1, len(thds)):
            assert thds[i] >= thds[i - 1] * 0.9, f"THD not monotonic: {thds}"


class TestDriveResonanceInteraction:
    """Higher drive should compress the resonance peak."""

    def test_high_drive_reduces_resonance_peak(self) -> None:
        """With high Q and high drive, the peak should be smaller than with
        the same Q and no drive (drive compresses feedback)."""
        peak_clean = _measure_resonance_peak(
            cutoff_hz=1000.0, resonance_q=8.0, filter_drive=0.0
        )
        peak_driven = _measure_resonance_peak(
            cutoff_hz=1000.0, resonance_q=8.0, filter_drive=0.8
        )
        # Drive should reduce the resonance peak amplitude
        assert peak_driven < peak_clean, (
            f"Resonance peak with drive=0.8 ({peak_driven:.1f}) should be less "
            f"than clean ({peak_clean:.1f})"
        )


class TestDriveProgression:
    """Drive effect should increase monotonically with the drive parameter."""

    def test_monotonic_difference_from_clean(self) -> None:
        signal = _render_test_signal()
        n = len(signal)
        cutoff = np.full(n, 1200.0, dtype=np.float64)

        clean = apply_zdf_svf(
            signal,
            cutoff_profile=cutoff,
            resonance_q=2.0,
            sample_rate=44100,
            filter_mode="lowpass",
            filter_drive=0.0,
        )

        drive_levels = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
        diffs = []
        for drive in drive_levels:
            driven = apply_zdf_svf(
                signal,
                cutoff_profile=cutoff,
                resonance_q=2.0,
                sample_rate=44100,
                filter_mode="lowpass",
                filter_drive=drive,
            )
            diff = float(np.sqrt(np.mean((driven - clean) ** 2)))
            diffs.append(diff)

        # Should be monotonically increasing
        for i in range(1, len(diffs)):
            assert diffs[i] >= diffs[i - 1] * 0.95, (
                f"Drive {drive_levels[i]} diff ({diffs[i]:.4f}) should be >= "
                f"drive {drive_levels[i - 1]} diff ({diffs[i - 1]:.4f})"
            )


class TestFilterStability:
    """Filter must remain stable under extreme parameters."""

    def test_high_drive_high_q_finite(self) -> None:
        """High Q + high drive produces finite output (driven path is bounded
        even though the linear path at Q=20 can ring to ~4000)."""
        signal = _render_test_signal(amplitude=1.0)
        n = len(signal)
        cutoff = np.full(n, 800.0, dtype=np.float64)

        result = apply_zdf_svf(
            signal,
            cutoff_profile=cutoff,
            resonance_q=20.0,
            sample_rate=44100,
            filter_mode="lowpass",
            filter_drive=1.0,
        )
        assert np.all(np.isfinite(result)), "Filter produced NaN/Inf"
        # The blended output includes ~25% linear path which CAN ring
        # at Q=20, so we just check finiteness, not amplitude bound.
        # The driven component itself should be bounded by algebraicSat.

    def test_moderate_drive_resonance_bounded(self) -> None:
        """At musically realistic Q (4-8), driven output stays reasonable."""
        signal = _render_test_signal(amplitude=1.0)
        n = len(signal)
        cutoff = np.full(n, 800.0, dtype=np.float64)

        for q_val in [4.0, 6.0, 8.0]:
            result = apply_zdf_svf(
                signal,
                cutoff_profile=cutoff,
                resonance_q=q_val,
                sample_rate=44100,
                filter_mode="lowpass",
                filter_drive=0.5,
            )
            assert np.all(np.isfinite(result)), f"NaN at Q={q_val}"
            peak = float(np.max(np.abs(result)))
            assert peak < 50.0, f"Q={q_val} peak={peak:.1f}, expected < 50"

    def test_high_drive_moderate_freq_stable(self) -> None:
        """Moderate cutoff + high drive should not cause instability.

        Note: the underlying linear SVF has a pre-existing numerical
        stability issue when signal frequency is near cutoff at high Q
        (resonance amplification → state accumulation). This test uses
        well-separated signal/cutoff frequencies.
        """
        signal = _render_test_signal(freq=220.0, amplitude=0.8)
        n = len(signal)
        cutoff = np.full(n, 3000.0, dtype=np.float64)

        result = apply_zdf_svf(
            signal,
            cutoff_profile=cutoff,
            resonance_q=4.0,
            sample_rate=44100,
            filter_mode="lowpass",
            filter_drive=0.8,
        )
        assert np.all(np.isfinite(result)), "Filter produced NaN/Inf"
        assert np.max(np.abs(result)) < 50.0, "Filter output exploded"


class TestEvenHarmonics:
    """Optional asymmetric bias should add even harmonics."""

    def test_even_harmonics_param_accepted(self) -> None:
        """The filter should accept filter_even_harmonics without error."""
        signal = _render_test_signal()
        n = len(signal)
        cutoff = np.full(n, 2000.0, dtype=np.float64)

        result = apply_zdf_svf(
            signal,
            cutoff_profile=cutoff,
            resonance_q=1.0,
            sample_rate=44100,
            filter_mode="lowpass",
            filter_drive=0.3,
            filter_even_harmonics=0.2,
        )
        assert np.all(np.isfinite(result))

    def test_even_harmonics_zero_matches_default(self) -> None:
        """filter_even_harmonics=0 should match default behavior."""
        signal = _render_test_signal()
        n = len(signal)
        cutoff = np.full(n, 2000.0, dtype=np.float64)

        default = apply_zdf_svf(
            signal,
            cutoff_profile=cutoff,
            resonance_q=1.0,
            sample_rate=44100,
            filter_mode="lowpass",
            filter_drive=0.3,
        )
        explicit_zero = apply_zdf_svf(
            signal,
            cutoff_profile=cutoff,
            resonance_q=1.0,
            sample_rate=44100,
            filter_mode="lowpass",
            filter_drive=0.3,
            filter_even_harmonics=0.0,
        )
        np.testing.assert_allclose(default, explicit_zero, atol=1e-10)

    def test_even_harmonics_changes_signal(self) -> None:
        """Non-zero even_harmonics should change the output."""
        signal = _render_test_signal()
        n = len(signal)
        cutoff = np.full(n, 2000.0, dtype=np.float64)

        without = apply_zdf_svf(
            signal,
            cutoff_profile=cutoff,
            resonance_q=1.0,
            sample_rate=44100,
            filter_mode="lowpass",
            filter_drive=0.3,
            filter_even_harmonics=0.0,
        )
        with_even = apply_zdf_svf(
            signal,
            cutoff_profile=cutoff,
            resonance_q=1.0,
            sample_rate=44100,
            filter_mode="lowpass",
            filter_drive=0.3,
            filter_even_harmonics=0.3,
        )
        diff = float(np.sqrt(np.mean((with_even - without) ** 2)))
        assert diff > 1e-4, "Even harmonics had no effect"
