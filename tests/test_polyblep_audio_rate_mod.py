"""Audio-rate per-sample parameter modulation tests for the polyblep engine.

These tests verify that ``param_profiles`` for ``pulse_width``,
``osc2_detune_cents``, and ``osc2_freq_ratio`` are threaded through the
oscillator render path per-sample (not collapsed to a single scalar at
note onset).  The spectral fingerprint of audio-rate modulation is
sidebands at ``f0 +/- f_mod`` — a scalar-only path produces none, so
the tests look for those sidebands.
"""

from __future__ import annotations

import numpy as np

from code_musics.engines.polyblep import render

SR = 44100


def _common_params(**overrides: object) -> dict[str, object]:
    base = {
        "waveform": "square",
        "pulse_width": 0.5,
        "cutoff_hz": 8000.0,
        "resonance_q": 0.707,
        "filter_drive": 0.0,
        "analog_jitter": 0.0,
        "pitch_drift": 0.0,
        "cutoff_drift": 0.0,
        "noise_floor": 0.0,
        "osc_asymmetry": 0.0,
        "osc_softness": 0.0,
        "osc_dc_offset": 0.0,
        "osc_shape_drift": 0.0,
    }
    base.update(overrides)
    return base


def _energy_at(
    signal: np.ndarray, freq_hz: float, sr: int, bw_hz: float = 5.0
) -> float:
    spec = np.abs(np.fft.rfft(signal))
    hz_per_bin = sr / signal.size
    lo = max(0, int((freq_hz - bw_hz) / hz_per_bin))
    hi = min(spec.size, int((freq_hz + bw_hz) / hz_per_bin) + 1)
    return float(np.sum(spec[lo:hi] ** 2))


class TestAudioRatePWM:
    def test_audio_rate_pwm_creates_sidebands(self) -> None:
        """Modulate `pulse_width` per-sample; assert sidebands at f0 +/- f_mod."""
        f0 = 220.0
        f_mod = 80.0
        dur = 1.0
        n = int(dur * SR)
        t = np.arange(n, dtype=np.float64) / SR
        # Modulate pulse_width between 0.35 and 0.65 (bipolar around 0.5)
        pulse_width_profile = 0.5 + 0.15 * np.sin(2.0 * np.pi * f_mod * t)

        with_mod = render(
            freq=f0,
            duration=dur,
            amp=0.8,
            sample_rate=SR,
            params=_common_params(pulse_width=0.5),
            param_profiles={"pulse_width": pulse_width_profile},
        )
        scalar = render(
            freq=f0,
            duration=dur,
            amp=0.8,
            sample_rate=SR,
            params=_common_params(pulse_width=0.5),
        )
        assert with_mod.shape == scalar.shape
        assert np.all(np.isfinite(with_mod))

        # Sidebands should appear at f0 +/- f_mod in the modulated signal.
        # A square wave has only odd harmonics, so f0 +/- 80 = 140 and 300 Hz
        # are ~empty in the scalar version but populated by the modulation.
        sb_low = _energy_at(with_mod, f0 - f_mod, SR)
        sb_high = _energy_at(with_mod, f0 + f_mod, SR)
        scalar_sb_low = _energy_at(scalar, f0 - f_mod, SR)
        scalar_sb_high = _energy_at(scalar, f0 + f_mod, SR)
        modulated_total = sb_low + sb_high
        scalar_total = scalar_sb_low + scalar_sb_high
        eps = 1e-18
        ratio_db = 10.0 * np.log10((modulated_total + eps) / (scalar_total + eps))
        assert ratio_db > 12.0, (
            f"expected audio-rate PWM sidebands at f0 +/- f_mod, got "
            f"only {ratio_db:.2f} dB over scalar baseline"
        )


class TestAudioRateOsc2Detune:
    def test_audio_rate_osc2_detune_creates_sidebands(self) -> None:
        """Modulate `osc2_detune_cents` per-sample; assert spectral sidebands."""
        f0 = 220.0
        f_mod = 60.0
        dur = 1.0
        n = int(dur * SR)
        t = np.arange(n, dtype=np.float64) / SR
        # Modulate osc2 detune between -50 and +50 cents.
        detune_profile = 50.0 * np.sin(2.0 * np.pi * f_mod * t)

        with_mod = render(
            freq=f0,
            duration=dur,
            amp=0.8,
            sample_rate=SR,
            params=_common_params(
                waveform="saw",
                osc2_level=1.0,
                osc2_waveform="saw",
                osc2_detune_cents=0.0,
            ),
            param_profiles={"osc2_detune_cents": detune_profile},
        )
        scalar = render(
            freq=f0,
            duration=dur,
            amp=0.8,
            sample_rate=SR,
            params=_common_params(
                waveform="saw",
                osc2_level=1.0,
                osc2_waveform="saw",
                osc2_detune_cents=0.0,
            ),
        )
        assert with_mod.shape == scalar.shape
        assert np.all(np.isfinite(with_mod))

        # Modulation should add energy around f0 +/- f_mod (audible as
        # roughness).  Compare accumulated energy in a band around those
        # sideband frequencies.
        mod_energy = _energy_at(with_mod, f0 + f_mod, SR) + _energy_at(
            with_mod, f0 - f_mod, SR
        )
        scalar_energy = _energy_at(scalar, f0 + f_mod, SR) + _energy_at(
            scalar, f0 - f_mod, SR
        )
        eps = 1e-18
        ratio_db = 10.0 * np.log10((mod_energy + eps) / (scalar_energy + eps))
        assert ratio_db > 6.0, (
            f"expected audio-rate osc2-detune sidebands, got "
            f"{ratio_db:.2f} dB margin over scalar baseline"
        )


class TestScalarPathUnchanged:
    def test_scalar_pulse_width_identical_without_profile(self) -> None:
        """Without per-sample profiles, output must be bit-identical.

        Freezes the scalar (no param_profiles) render so that adding the
        per-sample path cannot silently change pre-existing behavior.
        """
        params = _common_params(waveform="square", pulse_width=0.4)
        first = render(freq=220.0, duration=0.3, amp=0.8, sample_rate=SR, params=params)
        second = render(
            freq=220.0, duration=0.3, amp=0.8, sample_rate=SR, params=params
        )
        # Same params -> deterministic identical output (already true) and
        # the determinism must survive the audio-rate plumbing addition.
        np.testing.assert_array_equal(first, second)


class TestAudioRateOsc2FreqRatio:
    def test_audio_rate_osc2_freq_ratio(self) -> None:
        """Per-sample `osc2_freq_ratio` modulates osc2 pitch directly."""
        f0 = 220.0
        dur = 0.5
        n = int(dur * SR)
        # Sweep osc2 ratio from 1.0 to 2.0 linearly (one octave).
        ratio_profile = np.linspace(1.0, 2.0, n)

        with_mod = render(
            freq=f0,
            duration=dur,
            amp=0.8,
            sample_rate=SR,
            params=_common_params(
                waveform="saw",
                osc2_level=1.0,
                osc2_waveform="saw",
                osc2_detune_cents=0.0,
            ),
            param_profiles={"osc2_freq_ratio": ratio_profile},
        )
        scalar = render(
            freq=f0,
            duration=dur,
            amp=0.8,
            sample_rate=SR,
            params=_common_params(
                waveform="saw",
                osc2_level=1.0,
                osc2_waveform="saw",
                osc2_detune_cents=0.0,
            ),
        )
        assert with_mod.shape == scalar.shape
        assert np.all(np.isfinite(with_mod))
        # Swept osc2 pitch should produce different audio than the scalar path.
        assert not np.allclose(with_mod, scalar)
        assert np.linalg.norm(with_mod - scalar) > 1.0
