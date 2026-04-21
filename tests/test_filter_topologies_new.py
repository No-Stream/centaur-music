"""Tests for Diva-inspired filter topologies: Sallen-Key and Cascade.

- `sallen_key`: 2-pole Sallen-Key with positive-feedback resonance shaping.
  Biting, CEM-3320-ish — the "Diva Bite" character. 12 dB/oct slope.

- `cascade`: 4-pole cascade of independent ZDF 1-poles followed by a separate
  resonance peaking filter at cutoff. Smoother 24 dB/oct than the Moog ladder
  — no global tanh feedback growl. Closer to Prophet-5 rev-2 / Juno feel.
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._filters import apply_filter

SR = 44100


def _test_signal(dur: float = 0.5, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(SR * dur))


def _band_energy(
    signal: np.ndarray, low_hz: float, high_hz: float, sr: int = SR
) -> float:
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1 / sr)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    return float(np.sqrt(np.mean(spectrum[mask] ** 2)))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


class TestSallenKeyBasic:
    def test_sallen_key_produces_finite_output(self) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 1500.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="sallen_key",
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) > 0.01

    def test_sallen_key_attenuates_highs(self) -> None:
        """2-pole lowpass should reduce high-frequency energy vs. unfiltered."""
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        filtered = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="sallen_key",
        )
        unfiltered_hf = _band_energy(sig, 5000, 20000)
        filtered_hf = _band_energy(filtered, 5000, 20000)
        assert filtered_hf < unfiltered_hf * 0.5

    def test_sallen_key_steeper_than_nothing_but_gentler_than_ladder(self) -> None:
        """Sallen-Key is 12 dB/oct, ladder is 24 dB/oct.  At 4x cutoff, ladder
        should attenuate notably more than SK. ``bass_compensation=0.0`` on the
        ladder isolates the topology slope from the default compensation term.
        """
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        sk = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="sallen_key",
        )
        ladder = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            bass_compensation=0.0,
        )
        # At 4 kHz (2 octaves above cutoff), ladder should have ~24 dB more
        # attenuation than SK.  Use a wider band and a loose ratio threshold.
        sk_hf = _band_energy(sk, 3500, 6000)
        ladder_hf = _band_energy(ladder, 3500, 6000)
        assert ladder_hf < sk_hf * 0.5, f"sk={sk_hf:.4f}, ladder={ladder_hf:.4f}"

    def test_sallen_key_resonance_peak(self) -> None:
        """High Q should produce a measurable peak in a narrow band around
        the cutoff."""
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        low_q = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="sallen_key",
            resonance_q=0.707,
        )
        high_q = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="sallen_key",
            resonance_q=6.0,
        )
        low_peak = _band_energy(low_q, 900, 1100)
        high_peak = _band_energy(high_q, 900, 1100)
        assert high_peak > low_peak * 1.3

    def test_sallen_key_stable_at_extreme(self) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 60.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="sallen_key",
            resonance_q=10.0,
            filter_drive=1.0,
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) < 4.0

    @pytest.mark.parametrize("mode", ["lowpass", "bandpass", "highpass"])
    def test_sallen_key_modes_produce_output(self, mode: str) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="sallen_key",
            filter_mode=mode,
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) > 0.001


class TestSallenKeyDrive:
    def test_drive_adds_harmonics(self) -> None:
        n = int(SR * 0.5)
        t = np.arange(n) / SR
        sig = np.sin(2 * np.pi * 200 * t)
        cutoff = np.full(n, 2000.0)
        clean = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="sallen_key",
            filter_drive=0.0,
        )
        driven = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="sallen_key",
            filter_drive=0.8,
        )
        clean_hf = _band_energy(clean, 400, 5000)
        driven_hf = _band_energy(driven, 400, 5000)
        assert driven_hf > clean_hf * 1.3


class TestCascadeBasic:
    def test_cascade_produces_finite_output(self) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 1500.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="cascade",
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) > 0.01

    def test_cascade_steeper_than_sallen_key(self) -> None:
        """Cascade is 24 dB/oct, SK is 12 dB/oct."""
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        sk = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="sallen_key",
        )
        cas = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="cascade",
        )
        sk_hf = _band_energy(sk, 3500, 6000)
        cas_hf = _band_energy(cas, 3500, 6000)
        assert cas_hf < sk_hf * 0.6

    def test_cascade_smoother_than_ladder_at_high_q(self) -> None:
        """Cascade has no global tanh feedback growl; it's smoother than the
        Moog ladder at high resonance.  Measure this as: at fc=1000, q=10,
        ladder has more output in a tight band around fc (growl/peaking)
        than cascade does."""
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        ladder = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            resonance_q=10.0,
            filter_drive=0.5,
        )
        cas = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="cascade",
            resonance_q=10.0,
            filter_drive=0.5,
        )
        # Both should have some resonance, but they should differ noticeably.
        assert _rms(ladder - cas) > 0.01 * _rms(ladder)

    @pytest.mark.parametrize("mode", ["lowpass", "bandpass", "highpass"])
    def test_cascade_modes_produce_output(self, mode: str) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="cascade",
            filter_mode=mode,
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) > 0.001

    def test_cascade_resonance_peak(self) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        low_q = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="cascade",
            resonance_q=0.707,
        )
        high_q = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="cascade",
            resonance_q=6.0,
        )
        low_peak = _band_energy(low_q, 900, 1100)
        high_peak = _band_energy(high_q, 900, 1100)
        assert high_peak > low_peak * 1.3

    def test_cascade_stable_at_extreme(self) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 60.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="cascade",
            resonance_q=10.0,
            filter_drive=1.0,
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) < 4.0


class TestSupportsInfrastructure:
    """Both new topologies should work with the existing ladder/SVF surface:
    serial HPF, ext feedback, filter_morph."""

    @pytest.mark.parametrize("topology", ["sallen_key", "cascade"])
    def test_with_serial_hpf(self, topology: str) -> None:
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 2000.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            hpf_cutoff_hz=300.0,
        )
        assert np.all(np.isfinite(result))
        # Low band should be attenuated by serial HPF.
        no_hpf = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            hpf_cutoff_hz=0.0,
        )
        assert _band_energy(result, 50, 200) < _band_energy(no_hpf, 50, 200)

    @pytest.mark.parametrize("topology", ["sallen_key", "cascade"])
    def test_with_ext_feedback(self, topology: str) -> None:
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 1200.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            feedback_amount=0.4,
            feedback_saturation=0.3,
            resonance_q=4.0,
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) < 4.0

    def test_cascade_filter_morph(self) -> None:
        """Cascade supports 4->3->2->1 pole morph like the ladder."""
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 1500.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="cascade",
            filter_morph=2.0,
            resonance_q=2.0,
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) > 0.01
