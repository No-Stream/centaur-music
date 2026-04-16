"""Tests for oscillator imperfection parameters in the polyblep engine."""

from __future__ import annotations

import numpy as np

from code_musics.engines.registry import render_note_signal

SR = 44100
FREQ = 440.0
DUR = 0.3


def _render(**overrides: object) -> np.ndarray:
    # Disable analog character defaults so we isolate the imperfection under
    # test.  Otherwise per-note jitter and drift mask the effect.
    params: dict[str, object] = {
        "engine": "polyblep",
        "waveform": "saw",
        "cutoff_hz": 8000.0,
        "pitch_drift": 0.0,
        "analog_jitter": 0.0,
        "noise_floor": 0.0,
        "cutoff_drift": 0.0,
        "voice_card_spread": 0.0,
    }
    params.update(overrides)
    return render_note_signal(
        freq=FREQ, duration=DUR, amp=1.0, sample_rate=SR, params=params
    )


class TestOscAsymmetry:
    def test_zero_is_baseline(self) -> None:
        """osc_asymmetry=0 should match default behavior."""
        baseline = _render()
        explicit_zero = _render(osc_asymmetry=0.0)
        np.testing.assert_allclose(baseline, explicit_zero, atol=1e-10)

    def test_nonzero_changes_spectrum(self) -> None:
        """Asymmetry should change the spectral balance."""
        baseline = _render()
        asymmetric = _render(osc_asymmetry=0.5)
        assert not np.allclose(baseline, asymmetric, atol=1e-6)


class TestOscSoftness:
    def test_softness_reduces_harmonics(self) -> None:
        """Higher softness should reduce high-frequency content.

        Use a low base frequency (100 Hz) with many harmonics below Nyquist so
        the one-pole cutoff from softness falls within the audible band.  At
        higher frequencies, the cutoff exceeds the alpha=1.0 ceiling and the
        one-pole lowpass is a no-op.
        """
        low_freq = 100.0
        common_params: dict[str, object] = {
            "engine": "polyblep",
            "waveform": "saw",
            "cutoff_hz": 20000.0,
            "pitch_drift": 0.0,
            "analog_jitter": 0.0,
            "noise_floor": 0.0,
            "cutoff_drift": 0.0,
            "voice_card_spread": 0.0,
        }
        sharp = render_note_signal(
            freq=low_freq,
            duration=DUR,
            amp=1.0,
            sample_rate=SR,
            params={**common_params, "osc_softness": 0.0},
        )
        soft = render_note_signal(
            freq=low_freq,
            duration=DUR,
            amp=1.0,
            sample_rate=SR,
            params={**common_params, "osc_softness": 0.99},
        )
        sharp_spec = np.abs(np.fft.rfft(sharp))
        soft_spec = np.abs(np.fft.rfft(soft))
        freqs = np.fft.rfftfreq(len(sharp), 1 / SR)
        hf_mask = freqs > 4000
        assert np.sum(soft_spec[hf_mask]) < np.sum(sharp_spec[hf_mask]) * 0.9


class TestOscDcOffset:
    def test_offset_shifts_mean(self) -> None:
        """DC offset should shift the mean of the output signal."""
        no_dc = _render(osc_dc_offset=0.0, cutoff_hz=20000.0)
        with_dc = _render(osc_dc_offset=1.0, cutoff_hz=20000.0)
        # The absolute difference in mean should be nonzero
        assert abs(np.mean(with_dc) - np.mean(no_dc)) > 1e-4


class TestOscShapeDrift:
    def test_drift_varies_over_time(self) -> None:
        """Shape drift should cause the waveform to vary across the note."""
        result = _render(osc_shape_drift=1.0, waveform="square")
        # Split into early and late halves -- they should differ
        mid = len(result) // 2
        early_spectrum = np.abs(np.fft.rfft(result[:mid]))
        late_spectrum = np.abs(np.fft.rfft(result[mid:]))
        # The spectra shouldn't be identical (drift changed the shape)
        assert not np.allclose(early_spectrum, late_spectrum, rtol=0.01)


class TestAllImperfectionsSmoke:
    def test_combined_produces_finite_output(self) -> None:
        """All imperfections enabled simultaneously should produce valid audio."""
        result = _render(
            osc_asymmetry=0.3,
            osc_softness=0.3,
            osc_dc_offset=0.5,
            osc_shape_drift=0.4,
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) > 0.01
