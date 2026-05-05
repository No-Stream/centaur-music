"""Tests for analog-inspired filter topologies: Jupiter, SEM, K35, and Diode.

- `jupiter`: Roland IR3109-flavored 4-pole OTA cascade with a single global
  tanh feedback (ADAA + Newton solvers).  Soft Q→k mapping preserves bass
  better than Moog ladder at matched Q.  Supports pole-tap blend
  (24→18→12→6 dB/oct) under ``filter_morph ∈ [0, 3]``.
- `sem`: Oberheim SEM 2-pole SVF with wider Q-to-damping curve and a
  three-stage LP→Notch→HP morph under ``filter_morph ∈ [0, 2]``.
- `k35`: Korg35 2-pole Sallen-Key with diode-clipped feedback.  The defining
  MS-20 "snarl" comes from ``k35_feedback_asymmetry``.  LP + HP modes.
- `diode`: TB-303 style 3-pole diode ladder.  18 dB/oct native; feedback tap
  between stages 2 and 3 gives the characteristic bass-suck-with-squelch.
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


def _sine(freq_hz: float, dur: float = 0.5, amplitude: float = 0.5) -> np.ndarray:
    n = int(SR * dur)
    t = np.arange(n) / SR
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def _bin_amplitude(
    signal: np.ndarray, target_hz: float, half_width_hz: float = 5.0
) -> float:
    """Peak spectral magnitude in a narrow bin around ``target_hz``.

    Used for harmonic measurement on pure-sine tests.  Returns the max
    (not RMS) magnitude inside the bin so a well-defined single harmonic
    reads cleanly without being diluted by neighbouring noise bins.
    """
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1 / SR)
    mask = (freqs >= target_hz - half_width_hz) & (freqs <= target_hz + half_width_hz)
    if not np.any(mask):
        return 0.0
    return float(np.max(spectrum[mask]))


# ---------------------------------------------------------------------------
# Jupiter
# ---------------------------------------------------------------------------


class TestJupiterBasic:
    def test_jupiter_produces_finite_output(self) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 1500.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="jupiter",
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) > 0.01

    def test_jupiter_attenuates_highs(self) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        filtered = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="jupiter",
        )
        unfiltered_hf = _band_energy(sig, 5000, 20000)
        filtered_hf = _band_energy(filtered, 5000, 20000)
        assert filtered_hf < unfiltered_hf * 0.5

    def test_jupiter_resonance_peak(self) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        low_q = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="jupiter",
            resonance_q=0.707,
        )
        high_q = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="jupiter",
            resonance_q=6.0,
        )
        low_peak = _band_energy(low_q, 900, 1100)
        high_peak = _band_energy(high_q, 900, 1100)
        assert high_peak > low_peak * 1.3

    def test_jupiter_stable_at_extreme(self) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 60.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="jupiter",
            resonance_q=10.0,
            filter_drive=1.0,
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) < 4.0

    def test_jupiter_slope_at_least_as_steep_as_ladder(self) -> None:
        """Jupiter is 24 dB/oct, same nominal slope as the Moog ladder."""
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        jup = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="jupiter",
        )
        lad = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
        )
        sk = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="sallen_key",
        )
        jup_hf = _band_energy(jup, 3500, 6000)
        lad_hf = _band_energy(lad, 3500, 6000)
        sk_hf = _band_energy(sk, 3500, 6000)
        # Both 24 dB/oct filters should sit well below the 12 dB/oct SK.
        assert jup_hf < sk_hf * 0.6, f"jup_hf={jup_hf:.4f}, sk_hf={sk_hf:.4f}"
        # Jupiter's HF should be within a reasonable factor of ladder's (same slope).
        assert jup_hf < lad_hf * 2.5, f"jup_hf={jup_hf:.4f}, lad_hf={lad_hf:.4f}"


class TestJupiterCharacter:
    def test_jupiter_preserves_bass_vs_ladder(self) -> None:
        """At high Q, Jupiter's soft Q→k mapping should yield a less extreme
        resonance peak at fc than the Moog ladder's aggressive k→4 curve."""
        sig = _test_signal()
        cutoff = np.full(len(sig), 1500.0)
        jup = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="jupiter",
            resonance_q=8.0,
        )
        lad = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            resonance_q=8.0,
        )
        jup_peak = _band_energy(jup, 1400, 1600)
        lad_peak = _band_energy(lad, 1400, 1600)
        assert jup_peak < lad_peak, f"jup_peak={jup_peak:.4f}, lad_peak={lad_peak:.4f}"

    def test_jupiter_adaa_vs_newton_agree_at_low_q(self) -> None:
        """ADAA and Newton solvers should agree when feedback is mild.

        The two solvers differ fundamentally — Newton closes the delay-free
        loop implicitly to machine precision, ADAA advances an explicit
        one-sample-delayed feedback through a second-order antiderivative
        approximation.  At Q=0.707 (no resonance peak) and moderate input
        amplitude they stay within ~10% RMS of each other on broadband
        noise; a tighter threshold would simply be a calibration of the
        AD2 vs Newton delta rather than a useful guardrail.
        """
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 1500.0)
        adaa = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="jupiter",
            resonance_q=0.707,
            filter_solver="adaa",
        )
        newt = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="jupiter",
            resonance_q=0.707,
            filter_solver="newton",
        )
        ref_rms = max(_rms(adaa), _rms(newt), 1e-9)
        assert _rms(adaa - newt) < 0.10 * ref_rms

    def test_jupiter_newton_stable_at_extreme_q(self) -> None:
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 800.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="jupiter",
            resonance_q=20.0,
            filter_drive=1.0,
            filter_solver="newton",
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) < 4.0


# ---------------------------------------------------------------------------
# SEM
# ---------------------------------------------------------------------------


class TestSemBasic:
    def test_sem_produces_finite_output(self) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 1500.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="sem",
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) > 0.01

    def test_sem_attenuates_highs(self) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        filtered = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="sem",
        )
        unfiltered_hf = _band_energy(sig, 5000, 20000)
        filtered_hf = _band_energy(filtered, 5000, 20000)
        assert filtered_hf < unfiltered_hf * 0.5

    def test_sem_resonance_peak(self) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        low_q = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="sem",
            resonance_q=0.707,
        )
        high_q = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="sem",
            resonance_q=6.0,
        )
        low_peak = _band_energy(low_q, 900, 1100)
        high_peak = _band_energy(high_q, 900, 1100)
        assert high_peak > low_peak * 1.3

    def test_sem_stable_at_extreme(self) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 60.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="sem",
            resonance_q=10.0,
            filter_drive=1.0,
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) < 4.0

    def test_sem_slope_is_12db_per_oct(self) -> None:
        """SEM is 12 dB/oct — should have more HF passthrough than 24 dB/oct
        filters and less than unfiltered. ``bass_compensation=0.0`` on the
        ladder isolates the topology slope from the default compensation term.
        """
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        sem_out = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="sem",
        )
        lad = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            bass_compensation=0.0,
        )
        sem_hf = _band_energy(sem_out, 3500, 6000)
        lad_hf = _band_energy(lad, 3500, 6000)
        sig_hf = _band_energy(sig, 3500, 6000)
        assert sem_hf > lad_hf * 1.3, f"sem_hf={sem_hf:.4f}, lad_hf={lad_hf:.4f}"
        assert sem_hf < sig_hf * 0.7, f"sem_hf={sem_hf:.4f}, sig_hf={sig_hf:.4f}"


class TestSemMorph:
    def test_sem_morph_lp_notch_hp(self) -> None:
        """morph=0 is LP (preserves bass), morph=1 is notch (bass and HF),
        morph=2 is HP (preserves HF)."""
        sig = _test_signal()
        cutoff = np.full(len(sig), 1500.0)
        lp = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="sem",
            filter_morph=0.0,
        )
        notch = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="sem",
            filter_morph=1.0,
        )
        hp = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="sem",
            filter_morph=2.0,
        )
        lp_bass = _band_energy(lp, 50, 400)
        notch_bass = _band_energy(notch, 50, 400)
        hp_bass = _band_energy(hp, 50, 400)
        lp_hf = _band_energy(lp, 3000, 8000)
        notch_hf = _band_energy(notch, 3000, 8000)
        hp_hf = _band_energy(hp, 3000, 8000)

        # morph=0 (LP) keeps more bass than morph=2 (HP).
        assert lp_bass > hp_bass, f"lp_bass={lp_bass:.4f}, hp_bass={hp_bass:.4f}"
        # morph=2 (HP) keeps more HF than morph=0 (LP).
        assert hp_hf > lp_hf, f"hp_hf={hp_hf:.4f}, lp_hf={lp_hf:.4f}"
        # morph=1 (notch) keeps both passbands reasonably high.
        assert notch_bass > 0.8 * lp_bass, (
            f"notch_bass={notch_bass:.4f}, lp_bass={lp_bass:.4f}"
        )
        assert notch_hf > 0.8 * hp_hf, f"notch_hf={notch_hf:.4f}, hp_hf={hp_hf:.4f}"


# ---------------------------------------------------------------------------
# K35
# ---------------------------------------------------------------------------


class TestK35Basic:
    def test_k35_produces_finite_output(self) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 1500.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="k35",
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) > 0.01

    def test_k35_attenuates_highs(self) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        filtered = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="k35",
        )
        unfiltered_hf = _band_energy(sig, 5000, 20000)
        filtered_hf = _band_energy(filtered, 5000, 20000)
        assert filtered_hf < unfiltered_hf * 0.5

    def test_k35_resonance_peak(self) -> None:
        """K35's soft Q→k mapping (k saturates near 2) produces a modest
        resonance hump on noise.  Measure at fc=300 Hz with q=30 where the
        peak-to-passband ratio is most visible."""
        sig = _test_signal()
        cutoff = np.full(len(sig), 300.0)
        low_q = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="k35",
            resonance_q=0.707,
        )
        high_q = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="k35",
            resonance_q=30.0,
        )
        low_peak = _band_energy(low_q, 285, 315)
        high_peak = _band_energy(high_q, 285, 315)
        assert high_peak > low_peak * 1.3, (
            f"low_peak={low_peak:.4f}, high_peak={high_peak:.4f}"
        )

    def test_k35_stable_at_extreme(self) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 60.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="k35",
            resonance_q=10.0,
            filter_drive=1.0,
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) < 4.0

    def test_k35_slope_is_12db_per_oct(self) -> None:
        """K35 is 12 dB/oct (Sallen-Key)."""
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        k35 = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="k35",
        )
        lad = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
        )
        k35_hf = _band_energy(k35, 3500, 6000)
        lad_hf = _band_energy(lad, 3500, 6000)
        # Same-slope-ish comparison: K35 should let noticeably more HF through
        # than the 24 dB/oct ladder.
        assert k35_hf > lad_hf * 1.3, f"k35_hf={k35_hf:.4f}, lad_hf={lad_hf:.4f}"


class TestK35Character:
    def test_k35_asymmetry_produces_even_harmonics(self) -> None:
        """Diode-clipped feedback with asymmetry=1 should produce
        significantly more second-harmonic energy than asymmetry=0."""
        sig = _sine(220.0, dur=0.5)
        cutoff = np.full(len(sig), 1200.0)
        h2_by_asym: list[float] = []
        for asym in (0.0, 0.5, 1.0):
            out = apply_filter(
                sig,
                cutoff_profile=cutoff,
                sample_rate=SR,
                filter_topology="k35",
                resonance_q=3.0,
                filter_drive=0.5,
                k35_feedback_asymmetry=asym,
            )
            h2_by_asym.append(_bin_amplitude(out, 440.0))
        assert h2_by_asym[2] > h2_by_asym[0] * 3.0, (
            f"h2@asym=0.0: {h2_by_asym[0]:.4f}, h2@asym=1.0: {h2_by_asym[2]:.4f}"
        )

    def test_k35_self_oscillates_from_silence(self) -> None:
        """At very high Q, K35 should wake from exact silence through the
        kernel bootstrap-noise injection and ring at the cutoff frequency."""
        n = int(SR * 1.0)
        sig = np.zeros(n)
        cutoff = np.full(n, 440.0)
        out = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="k35",
            resonance_q=50.0,
        )
        assert np.all(np.isfinite(out))
        # Skip the first 100 ms to let the oscillation build up.
        steady = out[int(0.1 * SR) :]
        assert np.max(np.abs(steady)) > 0.01, f"max|out|={np.max(np.abs(steady)):.6f}"
        # Band around cutoff should dominate the spectrum.
        on_cutoff = _band_energy(steady, 420, 460)
        far_off = _band_energy(steady, 1000, 5000)
        assert on_cutoff > far_off, f"on_cutoff={on_cutoff:.4f}, far_off={far_off:.4f}"

    def test_k35_bp_coerces_to_lp(self) -> None:
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 1500.0)
        lp = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="k35",
            filter_mode="lowpass",
        )
        bp = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="k35",
            filter_mode="bandpass",
        )
        lp_rms = _rms(lp)
        assert _rms(lp - bp) < 0.01 * lp_rms, (
            f"||lp - bp||={_rms(lp - bp):.6f}, lp_rms={lp_rms:.4f}"
        )

    def test_k35_notch_coerces_to_lp(self) -> None:
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 1500.0)
        lp = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="k35",
            filter_mode="lowpass",
        )
        notch = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="k35",
            filter_mode="notch",
        )
        lp_rms = _rms(lp)
        assert _rms(lp - notch) < 0.01 * lp_rms


class TestK35NewtonSolver:
    def test_k35_adaa_vs_newton_agree_at_low_q(self) -> None:
        """ADAA and Newton solvers should agree closely when feedback is mild."""
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 1500.0)
        adaa = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="k35",
            resonance_q=0.707,
            filter_solver="adaa",
        )
        newt = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="k35",
            resonance_q=0.707,
            filter_solver="newton",
        )
        ref_rms = max(_rms(adaa), _rms(newt), 1e-9)
        # Newton tracks ADAA tightly at low Q where the one-sample feedback
        # lag has minimal audible effect; tolerance matches the Jupiter
        # low-Q agreement test.
        assert _rms(adaa - newt) < 0.1 * ref_rms

    def test_k35_newton_stable_at_extreme_q(self) -> None:
        """Newton solver stays finite at self-oscillation and heavy drive."""
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 800.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="k35",
            resonance_q=20.0,
            filter_drive=1.0,
            filter_solver="newton",
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) < 4.0

    def test_k35_newton_self_oscillates_from_silence(self) -> None:
        """Newton path bootstraps self-oscillation from exact silence
        through the per-sample Weyl noise injection, ringing at the
        cutoff frequency even without any excitation."""
        n = int(SR * 1.0)
        sig = np.zeros(n)
        cutoff = np.full(n, 440.0)
        out = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="k35",
            resonance_q=50.0,
            filter_solver="newton",
        )
        assert np.all(np.isfinite(out))
        steady = out[int(0.1 * SR) :]
        assert np.max(np.abs(steady)) > 0.01
        on_cutoff = _band_energy(steady, 420, 460)
        far_off = _band_energy(steady, 1000, 5000)
        assert on_cutoff > far_off

    def test_k35_newton_hp_falls_back_to_legacy(self) -> None:
        """K35 HP topology is structurally different — solver="newton"
        should silently fall back to the hybrid legacy path, matching
        the ADAA output bit-exactly (same kernel)."""
        sig = _test_signal(0.2)
        cutoff = np.full(len(sig), 1200.0)
        adaa = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="k35",
            filter_mode="highpass",
            resonance_q=3.0,
            filter_solver="adaa",
        )
        newt = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="k35",
            filter_mode="highpass",
            resonance_q=3.0,
            filter_solver="newton",
        )
        np.testing.assert_allclose(adaa, newt)

    def test_k35_newton_ext_fb_closed_loop(self) -> None:
        """Newton path with external feedback engages the outer-Newton
        branch (no unit-delay on the external tanh).  Output should be
        finite, bounded, and differ from the unit-delay ADAA path."""
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 1500.0)
        newt = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="k35",
            resonance_q=3.0,
            feedback_amount=0.6,
            feedback_saturation=0.4,
            filter_solver="newton",
        )
        adaa = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="k35",
            resonance_q=3.0,
            feedback_amount=0.6,
            feedback_saturation=0.4,
            filter_solver="adaa",
        )
        assert np.all(np.isfinite(newt))
        assert np.max(np.abs(newt)) < 10.0
        # Both paths should be audibly similar (closed vs unit-delay FB)
        # but not bit-identical — that's the whole point of closing the loop.
        diff_rms = _rms(newt - adaa)
        adaa_rms = _rms(adaa)
        assert diff_rms > 0.0, "outer-Newton path should differ from unit-delay ADAA"
        # But it shouldn't be dramatically different at moderate Q.
        assert diff_rms < 0.5 * adaa_rms


# ---------------------------------------------------------------------------
# Diode (TB-303)
# ---------------------------------------------------------------------------


class TestDiodeBasic:
    def test_diode_produces_finite_output(self) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 1500.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="diode",
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) > 0.01

    def test_diode_attenuates_highs(self) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        filtered = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="diode",
        )
        unfiltered_hf = _band_energy(sig, 5000, 20000)
        filtered_hf = _band_energy(filtered, 5000, 20000)
        assert filtered_hf < unfiltered_hf * 0.5

    def test_diode_resonance_peak(self) -> None:
        """High Q should lift the band at cutoff noticeably over unity Q.

        Both solvers show clear resonance growth, but the Newton diode
        resolves the delay-free feedback more tightly at low Q (bringing
        the baseline up) so the high/low ratio shrinks.  We keep the test
        solver-agnostic and tighten the band to 950-1050 Hz so the peak
        signal dominates nearby shoulder energy.
        """
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        low_q = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="diode",
            resonance_q=0.707,
        )
        high_q = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="diode",
            resonance_q=20.0,
        )
        low_peak = _band_energy(low_q, 950, 1050)
        high_peak = _band_energy(high_q, 950, 1050)
        # Diode's k-saturation is intentionally soft (max k≈1.7, self-osc
        # at Q~25), so the resonance peak is modest vs Moog — the 303's
        # character lives in bass suck + asymmetric harmonics, not a
        # towering peak.  Loose threshold catches regressions without
        # demanding Moog-style peak height.
        assert high_peak > low_peak * 1.15, (
            f"low_peak={low_peak:.4f}, high_peak={high_peak:.4f}"
        )

    def test_diode_stable_at_extreme(self) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 60.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="diode",
            resonance_q=10.0,
            filter_drive=1.0,
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) < 4.0

    def test_diode_slope_between_sk_and_ladder(self) -> None:
        """Diode is 18 dB/oct — HF energy should sit between
        Sallen-Key (12 dB/oct, more HF) and the Moog ladder (24 dB/oct,
        less HF) at matched cutoff. ``bass_compensation=0.0`` on the ladder
        isolates the topology slope from the default compensation term."""
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        dio = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="diode",
        )
        sk = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="sallen_key",
        )
        lad = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            bass_compensation=0.0,
        )
        dio_hf = _band_energy(dio, 3500, 6000)
        sk_hf = _band_energy(sk, 3500, 6000)
        lad_hf = _band_energy(lad, 3500, 6000)
        assert dio_hf > lad_hf, f"dio_hf={dio_hf:.4f}, lad_hf={lad_hf:.4f}"
        assert dio_hf < sk_hf, f"dio_hf={dio_hf:.4f}, sk_hf={sk_hf:.4f}"


class TestDiodeCharacter:
    def test_diode_squelch_bass_drops_with_q(self) -> None:
        """303 acid squelch: bass should decrease (or at least Q=10 << Q=1)
        as Q rises because the feedback tap between stages 2 and 3 sucks
        out low frequencies."""
        sig = _test_signal()
        cutoff = np.full(len(sig), 1500.0)
        bass_levels: list[float] = []
        for q in (1.0, 4.0, 10.0):
            out = apply_filter(
                sig,
                cutoff_profile=cutoff,
                sample_rate=SR,
                filter_topology="diode",
                resonance_q=q,
                filter_drive=0.3,
            )
            bass_levels.append(_band_energy(out, 50, 300))
        assert bass_levels[2] < 0.8 * bass_levels[0], (
            f"bass@q=1: {bass_levels[0]:.4f}, bass@q=10: {bass_levels[2]:.4f}"
        )

    def test_diode_asymmetric_harmonics(self) -> None:
        """Diode feedback asymmetry should persist even without drive because
        the feedback tap uses ``_diode_shape`` by default — giving the diode
        significantly more 2nd-harmonic content than a Moog ladder at
        matched settings."""
        sig = _sine(220.0, dur=0.5)
        cutoff = np.full(len(sig), 1200.0)
        dio = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="diode",
            resonance_q=2.0,
            filter_drive=0.0,
        )
        lad = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            resonance_q=2.0,
            filter_drive=0.0,
        )
        dio_h2 = _bin_amplitude(dio, 440.0)
        lad_h2 = _bin_amplitude(lad, 440.0)
        assert dio_h2 > 5.0 * lad_h2, f"dio_h2={dio_h2:.4f}, lad_h2={lad_h2:.4f}"

    def test_diode_bp_hp_notch_coerce_to_lp(self) -> None:
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 1500.0)
        lp = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="diode",
            filter_mode="lowpass",
        )
        lp_rms = _rms(lp)
        for mode in ("bandpass", "highpass", "notch"):
            other = apply_filter(
                sig,
                cutoff_profile=cutoff,
                sample_rate=SR,
                filter_topology="diode",
                filter_mode=mode,
            )
            assert _rms(lp - other) < 0.01 * lp_rms, (
                f"mode={mode}: ||lp - other||={_rms(lp - other):.6f}"
            )

    def test_diode_adaa_vs_newton_agree_at_low_q(self) -> None:
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 1500.0)
        adaa = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="diode",
            resonance_q=0.707,
            filter_solver="adaa",
        )
        newt = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="diode",
            resonance_q=0.707,
            filter_solver="newton",
        )
        ref_rms = max(_rms(adaa), _rms(newt), 1e-9)
        assert _rms(adaa - newt) < 0.05 * ref_rms

    def test_diode_newton_ext_fb_closed_loop(self) -> None:
        """Diode Newton path with external feedback engages the
        outer-Newton branch added in the analog-modeling maturation
        pass — no unit-delay on the external tanh.  Output should be
        finite, bounded, and differ from the ADAA unit-delay path."""
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 1200.0)
        newt = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="diode",
            resonance_q=4.0,
            feedback_amount=0.5,
            feedback_saturation=0.3,
            filter_solver="newton",
        )
        adaa = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="diode",
            resonance_q=4.0,
            feedback_amount=0.5,
            feedback_saturation=0.3,
            filter_solver="adaa",
        )
        assert np.all(np.isfinite(newt))
        assert np.max(np.abs(newt)) < 10.0
        # Closed-loop Newton should differ from unit-delay ADAA.
        diff_rms = _rms(newt - adaa)
        adaa_rms = _rms(adaa)
        assert diff_rms > 0.0
        assert diff_rms < 0.5 * adaa_rms

    def test_diode_newton_ext_fb_morph_falls_back_to_unit_delay(self) -> None:
        """outer-Newton gate requires morph=0; with morph>0 the diode
        Newton path should still solve the internal diode feedback
        (the preexisting win) but keep unit-delay on external FB."""
        sig = _test_signal(0.2)
        cutoff = np.full(len(sig), 1500.0)
        out = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="diode",
            resonance_q=5.0,
            feedback_amount=0.4,
            filter_morph=1.0,
            filter_solver="newton",
        )
        assert np.all(np.isfinite(out))
        assert np.max(np.abs(out)) < 10.0


# ---------------------------------------------------------------------------
# K35/diode joint self-osc + outer-Newton regression.
#
# The existing K35 and diode Newton tests exercise these dimensions
# independently:
# - self-osc + internal-Newton with ``feedback_amount=0`` (outer-Newton
#   branch NOT engaged)
# - outer-Newton closed loop with moderate Q (``resonance_q=3.0``, far from
#   self-osc)
#
# The joint regime — high-Q self-oscillation WITH the outer-Newton loop
# engaged — is where unit-delay damping would differ most from the
# closed Newton solve, and where the Newton solver is most likely to
# diverge (two tanh nonlinearities chained through the feedback loop
# plus near-critical gain).  If the solver ever regresses to single-step
# unit-delay behavior here, either the output explodes or it silently
# goes DC.  Assert finite + bounded output to catch both.
# ---------------------------------------------------------------------------


class TestK35DiodeJointSelfOscOuterNewton:
    """Joint regime: high-Q self-oscillation AND closed-loop external
    feedback on K35 and diode.  The outer-Newton branch eliminates the
    unit-delay damping on the external tanh; in this regime the
    difference is most audible and the solver most likely to diverge."""

    @pytest.mark.parametrize("filter_topology", ["k35", "diode"])
    @pytest.mark.parametrize("resonance_q", [20.0, 30.0])
    def test_joint_high_q_ext_fb_stable(
        self, filter_topology: str, resonance_q: float
    ) -> None:
        """With resonance_q >= 20 (self-osc regime) and feedback_amount > 0
        (outer-Newton engaged), the Newton solver must produce finite,
        bounded output.  This is the regime where naive feedback handling
        (unit-delay or naive fixed-point iteration) tends to explode or
        go DC."""
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 1000.0)
        out = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=filter_topology,
            resonance_q=resonance_q,
            feedback_amount=0.5,
            feedback_saturation=0.4,
            filter_drive=0.5,
            filter_solver="newton",
        )
        assert np.all(np.isfinite(out)), (
            f"{filter_topology} @ Q={resonance_q} + outer-Newton FB produced "
            f"non-finite output — solver diverged"
        )
        # Bounded: output peak must stay under a generous ceiling.  The
        # self-osc ring lands on the filter's cutoff, and the outer-loop
        # tanh soft-saturates runaway, so a healthy solver stays <8.0 even
        # at Q=30.  A runaway would blow past 100 easily.
        peak = float(np.max(np.abs(out)))
        assert peak < 8.0, (
            f"{filter_topology} @ Q={resonance_q} + outer-Newton FB peak "
            f"{peak:.2f} exceeds ceiling — solver unstable"
        )
        # Non-degenerate output: must not go silent or DC-lock.  In the
        # self-osc regime the filter rings at its cutoff, producing
        # substantial AC energy even with modest input.
        rms = _rms(out)
        assert rms > 0.005, (
            f"{filter_topology} @ Q={resonance_q} + outer-Newton FB RMS "
            f"{rms:.4f} suspiciously low — solver may be DC-locked or silent"
        )

    @pytest.mark.parametrize("filter_topology", ["k35", "diode"])
    def test_joint_high_q_ext_fb_from_silence(self, filter_topology: str) -> None:
        """With zero input and high Q + outer-Newton FB, the filter
        should self-oscillate (or bootstrap via seed-noise injection)
        without diverging.  Guards against a solver that produces a
        finite bounded output from a noise-excited signal but goes
        NaN/Inf when starting from clean silence."""
        n = int(SR * 0.5)
        silence = np.zeros(n)
        cutoff = np.full(n, 500.0)
        out = apply_filter(
            silence,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=filter_topology,
            resonance_q=25.0,
            feedback_amount=0.5,
            feedback_saturation=0.3,
            filter_solver="newton",
        )
        assert np.all(np.isfinite(out)), (
            f"{filter_topology} from silence + outer-Newton FB produced "
            f"non-finite output"
        )
        peak = float(np.max(np.abs(out)))
        assert peak < 8.0, (
            f"{filter_topology} from silence + outer-Newton FB peak "
            f"{peak:.2f} exceeds ceiling"
        )


# ---------------------------------------------------------------------------
# Shared infrastructure: serial HPF, external feedback, mode support
# ---------------------------------------------------------------------------


_NEW_TOPOLOGIES = ("jupiter", "sem", "k35", "diode")

_SUPPORTED_MODES: dict[str, tuple[str, ...]] = {
    "jupiter": ("lowpass", "bandpass", "highpass"),
    "sem": ("lowpass", "bandpass", "highpass", "notch"),
    "k35": ("lowpass", "highpass"),
    "diode": ("lowpass",),
}


class TestSupportsInfrastructure:
    """All four new topologies should interoperate with the existing
    filter surface: serial HPF, external feedback, and their own supported
    filter modes."""

    @pytest.mark.parametrize("topology", _NEW_TOPOLOGIES)
    def test_topology_with_serial_hpf(self, topology: str) -> None:
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 2000.0)
        with_hpf = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            hpf_cutoff_hz=300.0,
        )
        without_hpf = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            hpf_cutoff_hz=0.0,
        )
        assert np.all(np.isfinite(with_hpf))
        assert _band_energy(with_hpf, 50, 200) < _band_energy(without_hpf, 50, 200)

    @pytest.mark.parametrize("topology", _NEW_TOPOLOGIES)
    def test_topology_with_ext_feedback(self, topology: str) -> None:
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

    @pytest.mark.parametrize(
        "topology,mode",
        [
            (topology, mode)
            for topology in _NEW_TOPOLOGIES
            for mode in _SUPPORTED_MODES[topology]
        ],
    )
    def test_topology_supports_mode(self, topology: str, mode: str) -> None:
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 1000.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            filter_mode=mode,
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) > 0.001
