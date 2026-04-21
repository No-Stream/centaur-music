"""Per-sample oscillator phase-noise tests for the polyblep engine.

These tests verify that ``osc_phase_noise``:

* defaults to a no-op (existing pieces stay bit-identical),
* raises the out-of-band noise floor when enabled,
* is deterministic given the same note hash,
* uses independent RNG streams for osc1 and osc2 (so their zero-crossing
  jitter is uncorrelated and sums to a broadband, not coherent, noise
  floor).
"""

from __future__ import annotations

import numpy as np

from code_musics.engines.polyblep import render

SR = 44100


def _common_params(**overrides: object) -> dict[str, object]:
    base = {
        "waveform": "saw",
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


class TestPhaseNoiseDefault:
    def test_default_is_bit_identical_to_no_param(self) -> None:
        baseline = render(
            freq=220.0, duration=0.3, amp=0.7, sample_rate=SR, params=_common_params()
        )
        with_default_zero = render(
            freq=220.0,
            duration=0.3,
            amp=0.7,
            sample_rate=SR,
            params=_common_params(osc_phase_noise=0.0),
        )
        np.testing.assert_array_equal(baseline, with_default_zero)


class TestPhaseNoiseRaisesNoiseFloor:
    def test_noise_floor_rises_above_5khz(self) -> None:
        """Enabling osc_phase_noise raises the out-of-band noise floor.

        Use a wide-open cutoff so the phase noise is not masked by the
        filter; for a 220 Hz sawtooth the harmonics above 5 kHz decay
        as 1/n (small).  Phase-noise broadband contribution should
        dominate there.
        """
        dur = 1.0
        sr = SR
        # Open the filter fully (cutoff near Nyquist) so the filter
        # doesn't mask the high-frequency noise.
        clean = render(
            freq=220.0,
            duration=dur,
            amp=0.7,
            sample_rate=sr,
            params=_common_params(cutoff_hz=18000.0, osc_phase_noise=0.0),
        )
        noisy = render(
            freq=220.0,
            duration=dur,
            amp=0.7,
            sample_rate=sr,
            params=_common_params(cutoff_hz=18000.0, osc_phase_noise=0.5),
        )
        # Measure energy in the "between harmonics" zone above 5 kHz:
        # for a 220 Hz saw, harmonics sit at 220n — in any 20-Hz band
        # centered BETWEEN two harmonics, signal energy is near-zero
        # and phase-noise contribution dominates.
        clean_spec = np.abs(np.fft.rfft(clean))
        noisy_spec = np.abs(np.fft.rfft(noisy))
        hz_per_bin = sr / clean.size

        # Pick an off-harmonic frequency band: 5110 Hz is between the
        # 23rd (5060 Hz) and 24th (5280 Hz) harmonics.  Grab a narrow
        # window of ~10 Hz around it — wide enough for several bins,
        # narrow enough to avoid adjacent harmonics.
        def band_energy(
            spec: np.ndarray, center_hz: float, bw_hz: float = 10.0
        ) -> float:
            lo = int((center_hz - bw_hz) / hz_per_bin)
            hi = int((center_hz + bw_hz) / hz_per_bin) + 1
            return float(np.sum(spec[lo:hi] ** 2))

        clean_floor = band_energy(clean_spec, 5110.0) + band_energy(clean_spec, 5330.0)
        noisy_floor = band_energy(noisy_spec, 5110.0) + band_energy(noisy_spec, 5330.0)
        assert noisy_floor > clean_floor * 2.0, (
            f"expected osc_phase_noise to raise off-harmonic HF floor "
            f"substantially; got clean={clean_floor:.3e} noisy={noisy_floor:.3e}"
        )


class TestPhaseNoiseDeterminism:
    def test_identical_renders_are_bit_identical(self) -> None:
        params = _common_params(osc_phase_noise=0.5)
        first = render(freq=220.0, duration=0.3, amp=0.7, sample_rate=SR, params=params)
        second = render(
            freq=220.0, duration=0.3, amp=0.7, sample_rate=SR, params=params
        )
        np.testing.assert_array_equal(first, second)


class TestOsc1Osc2NoiseIndependence:
    def test_osc1_osc2_noise_streams_are_uncorrelated(self) -> None:
        """Osc1 and osc2 phase-noise streams must be independent.

        Strategy: isolate each oscillator's noise contribution by diffing
        the same-params render at ``osc_phase_noise>0`` against the
        ``osc_phase_noise=0`` reference.  The remaining signal is the
        oscillator's noise contribution plus filter coloring (both
        deterministic given the note hash).  For each oscillator rendered
        in isolation (osc2_level=0 vs osc1 silent) we compute its
        phase-noise perturbation, then check those perturbations are
        decorrelated (Pearson |r| below a conservative bound).
        """
        dur = 0.4
        sr = SR
        # Common params for the two "isolate osc1 / osc2" renders.
        # Open the filter so noise survives; keep everything else
        # deterministic.
        iso_params = _common_params(
            waveform="saw",
            cutoff_hz=18000.0,
        )

        # Render osc1 alone with and without phase noise; diff to isolate
        # osc1's phase-noise perturbation.
        osc1_clean = render(
            freq=220.0,
            duration=dur,
            amp=0.7,
            sample_rate=sr,
            params={**iso_params, "osc2_level": 0.0, "osc_phase_noise": 0.0},
        )
        osc1_noisy = render(
            freq=220.0,
            duration=dur,
            amp=0.7,
            sample_rate=sr,
            params={**iso_params, "osc2_level": 0.0, "osc_phase_noise": 0.5},
        )
        osc1_noise = osc1_noisy - osc1_clean

        # For osc2-only, we cannot silence osc1, but with osc2_level huge
        # osc2 dominates the mix.  Simpler approach: render with
        # osc2_level=1.0 (equal mix), isolate the total noise via diff,
        # then subtract the osc1-isolated noise contribution so what
        # remains is dominated by osc2.
        both_clean = render(
            freq=220.0,
            duration=dur,
            amp=0.7,
            sample_rate=sr,
            params={
                **iso_params,
                "osc2_level": 1.0,
                "osc2_waveform": "saw",
                "osc2_detune_cents": 0.0,
                "osc_phase_noise": 0.0,
            },
        )
        both_noisy = render(
            freq=220.0,
            duration=dur,
            amp=0.7,
            sample_rate=sr,
            params={
                **iso_params,
                "osc2_level": 1.0,
                "osc2_waveform": "saw",
                "osc2_detune_cents": 0.0,
                "osc_phase_noise": 0.5,
            },
        )
        both_noise = both_noisy - both_clean

        # If osc1 and osc2 phase-noise streams were identical, both_noise
        # would be ~2x osc1_noise (after accounting for normalization)
        # and strongly correlated with it.  If independent, the osc2
        # component of both_noise is decorrelated from osc1_noise, so
        # the correlation drops well below 1.0.
        def pearson(a: np.ndarray, b: np.ndarray) -> float:
            a = a - a.mean()
            b = b - b.mean()
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-18))

        corr = pearson(osc1_noise, both_noise)
        # With fully shared RNG streams: corr ~= 1.0 (perfectly correlated).
        # With independent streams: corr drops toward ~0.7 (since osc1's
        # noise is still a big component of both_noise, but osc2's
        # decorrelated noise pulls it down).  0.85 is halfway between the
        # independent baseline and 1.0 — tight enough to catch a
        # scaled-shared bug (``osc2 = 0.5 * osc1_noise`` → corr ≈ 0.95)
        # while leaving headroom for natural RNG variance.
        assert corr < 0.85, (
            f"osc1 and osc2 phase-noise streams appear correlated; "
            f"Pearson(osc1_noise, both_noise) = {corr:.4f}"
        )
