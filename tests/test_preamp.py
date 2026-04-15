"""End-to-end tests for the flux-domain transformer preamp effect."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.score import EffectSpec, NoteEvent, Phrase, Score
from code_musics.synth import SAMPLE_RATE, apply_preamp


def _make_sine(
    freq_hz: float,
    duration_s: float = 2.0,
    amplitude: float = 0.25,
) -> np.ndarray:
    """Stereo sine wave at the project sample rate."""
    t = np.arange(int(SAMPLE_RATE * duration_s)) / SAMPLE_RATE
    mono = amplitude * np.sin(2.0 * np.pi * freq_hz * t)
    return np.stack([mono, mono])


def _make_mono_sine(
    freq_hz: float,
    duration_s: float = 2.0,
    amplitude: float = 0.25,
) -> np.ndarray:
    """Mono sine wave at the project sample rate."""
    t = np.arange(int(SAMPLE_RATE * duration_s)) / SAMPLE_RATE
    return amplitude * np.sin(2.0 * np.pi * freq_hz * t)


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))


def _thd_of_sine(
    signal: np.ndarray,
    fundamental_hz: float,
    sample_rate: int = SAMPLE_RATE,
    n_harmonics: int = 8,
) -> float:
    """Measure total harmonic distortion of a (mono) signal.

    Returns the ratio of harmonic energy to fundamental energy.
    """
    n = len(signal)
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(n)))

    def _bin_energy(target_hz: float) -> float:
        idx = int(round(target_hz * n / sample_rate))
        window = 3
        lo = max(0, idx - window)
        hi = min(len(spectrum), idx + window + 1)
        return float(np.sum(spectrum[lo:hi] ** 2))

    fundamental_energy = _bin_energy(fundamental_hz)
    if fundamental_energy < 1e-20:
        return 0.0

    harmonic_energy = sum(
        _bin_energy(fundamental_hz * k) for k in range(2, n_harmonics + 2)
    )
    return float(np.sqrt(harmonic_energy / fundamental_energy))


class TestBasicSignalIntegrity:
    """Verify that the preamp produces valid, non-trivial output."""

    def test_output_is_not_silence(self) -> None:
        signal = _make_sine(200.0, duration_s=2.0, amplitude=0.25)
        output = apply_preamp(signal)
        assert isinstance(output, np.ndarray)
        assert _rms(output) > 1e-6

    def test_output_shape_matches_input(self) -> None:
        signal = _make_sine(200.0, duration_s=1.0)
        output = apply_preamp(signal)
        assert isinstance(output, np.ndarray)
        assert output.shape == signal.shape

    def test_output_differs_from_input(self) -> None:
        signal = _make_sine(200.0, duration_s=1.0, amplitude=0.25)
        output = apply_preamp(signal, drive=0.5, mix=1.0)
        assert isinstance(output, np.ndarray)
        assert not np.allclose(output, signal, atol=1e-8)

    def test_no_nan_or_inf(self) -> None:
        signal = _make_sine(200.0, duration_s=1.0, amplitude=0.25)
        output = apply_preamp(signal, drive=1.0, mix=1.0)
        assert isinstance(output, np.ndarray)
        assert np.all(np.isfinite(output))

    def test_no_gain_explosion(self) -> None:
        signal = _make_sine(200.0, duration_s=1.0, amplitude=0.25)
        output = apply_preamp(signal, drive=1.0, mix=1.0)
        assert isinstance(output, np.ndarray)
        input_peak_db = 20.0 * np.log10(np.max(np.abs(signal)) + 1e-12)
        output_peak_db = 20.0 * np.log10(np.max(np.abs(output)) + 1e-12)
        assert output_peak_db < input_peak_db + 3.0


class TestDriveScalesEffect:
    """Drive parameter should monotonically increase the amount of coloration."""

    def test_difference_rms_increases_with_drive(self) -> None:
        signal = _make_sine(200.0, duration_s=1.0, amplitude=0.25)
        drive_values = [0.25, 0.5, 1.0, 1.5]
        difference_rms_values = []
        for drive in drive_values:
            output = apply_preamp(
                signal, drive=drive, mix=1.0, compensation_mode="none"
            )
            assert isinstance(output, np.ndarray)
            diff = output - signal
            difference_rms_values.append(_rms(diff))

        for i in range(len(difference_rms_values) - 1):
            assert difference_rms_values[i + 1] > difference_rms_values[i], (
                f"Drive {drive_values[i + 1]} (diff RMS {difference_rms_values[i + 1]:.6f}) "
                f"should produce more coloration than drive {drive_values[i]} "
                f"(diff RMS {difference_rms_values[i]:.6f})"
            )


class TestMixControl:
    """Mix parameter should control wet/dry blending."""

    def test_mix_zero_returns_input(self) -> None:
        signal = _make_sine(200.0, duration_s=1.0, amplitude=0.25)
        output = apply_preamp(signal, mix=0.0, drive=0.5)
        assert isinstance(output, np.ndarray)
        np.testing.assert_allclose(output, signal, atol=1e-6)

    def test_difference_increases_with_mix(self) -> None:
        signal = _make_sine(200.0, duration_s=1.0, amplitude=0.25)
        mix_values = [0.0, 0.5, 1.0]
        difference_rms_values = []
        for mix in mix_values:
            output = apply_preamp(signal, mix=mix, drive=0.5, compensation_mode="none")
            assert isinstance(output, np.ndarray)
            diff = output - signal
            difference_rms_values.append(_rms(diff))

        for i in range(len(difference_rms_values) - 1):
            assert difference_rms_values[i + 1] >= difference_rms_values[i]


class TestFrequencyDependentSaturation:
    """The core flux-domain property: bass saturates more than treble."""

    def test_bass_has_higher_thd_than_treble(self) -> None:
        bass_signal = _make_mono_sine(50.0, duration_s=2.0, amplitude=0.25)
        treble_signal = _make_mono_sine(2000.0, duration_s=2.0, amplitude=0.25)

        bass_output = apply_preamp(
            bass_signal, drive=1.0, mix=1.0, compensation_mode="none"
        )
        treble_output = apply_preamp(
            treble_signal, drive=1.0, mix=1.0, compensation_mode="none"
        )
        assert isinstance(bass_output, np.ndarray)
        assert isinstance(treble_output, np.ndarray)

        bass_thd = _thd_of_sine(bass_output, 50.0)
        treble_thd = _thd_of_sine(treble_output, 2000.0)

        assert bass_thd > treble_thd * 1.5, (
            f"Bass THD ({bass_thd:.4f}) should be noticeably higher than "
            f"treble THD ({treble_thd:.4f}) due to flux-domain integration"
        )


class TestEvenOddHarmonicBalance:
    """even_odd parameter should control the 2nd vs 3rd harmonic balance."""

    def test_even_odd_controls_harmonic_character(self) -> None:
        signal = _make_mono_sine(200.0, duration_s=2.0, amplitude=0.25)
        n = len(signal)

        output_odd = apply_preamp(
            signal, drive=1.0, mix=1.0, even_odd=0.0, compensation_mode="none"
        )
        output_even = apply_preamp(
            signal, drive=1.0, mix=1.0, even_odd=1.0, compensation_mode="none"
        )
        assert isinstance(output_odd, np.ndarray)
        assert isinstance(output_even, np.ndarray)

        def _harmonic_amplitude(output: np.ndarray, harmonic: int) -> float:
            spectrum = np.abs(np.fft.rfft(output * np.hanning(n)))
            target_hz = 200.0 * harmonic
            idx = int(round(target_hz * n / SAMPLE_RATE))
            window = 3
            lo = max(0, idx - window)
            hi = min(len(spectrum), idx + window + 1)
            return float(np.max(spectrum[lo:hi]))

        h2_odd = _harmonic_amplitude(output_odd, 2)
        h3_odd = _harmonic_amplitude(output_odd, 3)
        h2_even = _harmonic_amplitude(output_even, 2)
        h3_even = _harmonic_amplitude(output_even, 3)

        even_ratio_at_odd_setting = h2_odd / (h3_odd + 1e-12)
        even_ratio_at_even_setting = h2_even / (h3_even + 1e-12)

        assert even_ratio_at_even_setting > even_ratio_at_odd_setting, (
            f"even_odd=1.0 should produce relatively more 2nd harmonic "
            f"(h2/h3={even_ratio_at_even_setting:.3f}) than even_odd=0.0 "
            f"(h2/h3={even_ratio_at_odd_setting:.3f})"
        )


class TestPresets:
    """Each preset should produce valid, non-trivial output."""

    @pytest.mark.parametrize(
        "preset_name",
        ["neve_warmth", "iron_color", "tube_glow", "transformer_drive"],
    )
    def test_preset_produces_non_trivial_output(self, preset_name: str) -> None:
        signal = _make_sine(200.0, duration_s=1.0, amplitude=0.25)
        output = apply_preamp(signal, preset=preset_name)
        assert isinstance(output, np.ndarray)
        assert _rms(output) > 1e-6
        assert not np.allclose(output, signal, atol=1e-8)
        assert np.all(np.isfinite(output))


class TestScoreIntegration:
    """Preamp should work end-to-end through Score rendering."""

    def test_score_with_preamp_master_effect_renders(self) -> None:
        score = Score(f0_hz=220.0)
        phrase = Phrase(
            events=(
                NoteEvent(start=0.0, duration=0.5, partial=1, amp=0.3),
                NoteEvent(start=0.6, duration=0.5, partial=2, amp=0.3),
            )
        )
        score.add_voice(
            "test_voice",
            synth_defaults={"engine": "additive"},
        )
        score.add_phrase("test_voice", phrase, start=0.0)
        score.master_effects = [EffectSpec("preamp", {"preset": "neve_warmth"})]

        audio = score.render()

        assert audio is not None
        assert audio.size > 0
        assert np.all(np.isfinite(audio))
        assert _rms(audio) > 1e-6


class TestStereoHandling:
    """Verify both channels are processed independently."""

    def test_different_channels_remain_different(self) -> None:
        t = np.arange(int(SAMPLE_RATE * 1.0)) / SAMPLE_RATE
        left = 0.25 * np.sin(2.0 * np.pi * 150.0 * t)
        right = 0.25 * np.sin(2.0 * np.pi * 600.0 * t)
        stereo = np.stack([left, right])

        output = apply_preamp(stereo, drive=0.8, mix=1.0)
        assert isinstance(output, np.ndarray)
        assert output.shape == stereo.shape

        left_rms = _rms(output[0])
        right_rms = _rms(output[1])
        assert left_rms > 1e-6
        assert right_rms > 1e-6
        assert not np.allclose(output[0], output[1], atol=1e-6)


class TestAnalysisOutput:
    """return_analysis=True should return a tuple with diagnostic metrics."""

    def test_returns_tuple_with_expected_keys(self) -> None:
        signal = _make_sine(200.0, duration_s=1.0, amplitude=0.25)
        result = apply_preamp(signal, drive=0.5, return_analysis=True)
        assert isinstance(result, tuple)
        audio, analysis = result

        assert isinstance(audio, np.ndarray)
        assert audio.shape == signal.shape

        assert isinstance(analysis, dict)
        expected_keys = {
            "algorithm",
            "drive",
            "mix",
            "warmth",
            "even_odd",
            "harmonic_injection",
            "dc_offset",
            "thd_pct",
            "thd_character",
            "compensation_mode_used",
            "compensation_gain_db",
        }
        assert expected_keys.issubset(analysis.keys()), (
            f"Missing keys: {expected_keys - analysis.keys()}"
        )

    def test_analysis_thd_increases_with_drive(self) -> None:
        signal = _make_sine(200.0, duration_s=1.0, amplitude=0.25)

        _, analysis_low = apply_preamp(
            signal, drive=0.25, mix=1.0, return_analysis=True
        )  # type: ignore[misc]
        _, analysis_high = apply_preamp(
            signal, drive=1.5, mix=1.0, return_analysis=True
        )  # type: ignore[misc]

        assert float(analysis_high["thd_pct"]) > float(analysis_low["thd_pct"])


class TestInputValidation:
    """Verify parameter validation raises on invalid inputs."""

    def test_rejects_mix_out_of_range(self) -> None:
        signal = _make_sine(200.0, duration_s=0.5)
        with pytest.raises(ValueError, match="mix must be between 0 and 1"):
            apply_preamp(signal, mix=1.5)

    def test_rejects_negative_drive(self) -> None:
        signal = _make_sine(200.0, duration_s=0.5)
        with pytest.raises(ValueError, match="drive must be non-negative"):
            apply_preamp(signal, drive=-0.1)

    def test_rejects_oversample_factor_below_one(self) -> None:
        signal = _make_sine(200.0, duration_s=0.5)
        with pytest.raises(ValueError, match="oversample_factor must be at least 1"):
            apply_preamp(signal, oversample_factor=0)
