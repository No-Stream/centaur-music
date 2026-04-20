"""Tests for the Newton-iterated nonlinear ZDF ladder filter.

The Newton solver resolves the delay-free feedback loop implicitly at each
sample instead of using a one-step ADAA approximation against the previous
sample. Intended to match the Diva-class ZDF-NL behavior described by
Zavalishin 2008 and the u-he RePro write-up: cleaner self-oscillation,
better resonance tracking under fast modulation, and more convincing drive.
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._filters import apply_filter

SR = 44100


def _test_signal(dur: float = 0.5, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(SR * dur))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


def _band_energy(
    signal: np.ndarray, low_hz: float, high_hz: float, sr: int = SR
) -> float:
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1 / sr)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    return float(np.sqrt(np.mean(spectrum[mask] ** 2)))


class TestNewtonSolverBasics:
    def test_newton_produces_finite_output(self) -> None:
        sig = _test_signal()
        cutoff = np.full(len(sig), 1000.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_solver="newton",
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) > 0.01

    def test_newton_at_zero_drive_matches_adaa_reasonably(self) -> None:
        """At drive=0, q=0.707 the feedback tanh operates in its linear regime.
        ADAA and Newton have different topologies (delayed vs instantaneous
        feedback), so they produce measurably different outputs, but at
        unity-ish Q the difference should stay modest (<25% RMS) — both are
        still recognizably the same filter.
        """
        sig = _test_signal()
        cutoff = np.full(len(sig), 1500.0)
        adaa = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_drive=0.0,
            resonance_q=0.707,
            filter_solver="adaa",
        )
        newton = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_drive=0.0,
            resonance_q=0.707,
            filter_solver="newton",
        )
        rms_diff = _rms(adaa - newton)
        ref = _rms(adaa)
        assert rms_diff < 0.25 * ref, (
            f"Newton and ADAA should be in the same ballpark at low drive/q: "
            f"rms_diff={rms_diff:.4f}, ref={ref:.4f}"
        )

    def test_newton_stable_at_extreme_resonance_and_drive(self) -> None:
        """The regression case from test_engine_polyblep.py extended to ladder+Newton.
        At cutoff=60Hz, q=24, drive=1.5 the solver must stay finite and bounded.
        """
        sig = _test_signal()
        cutoff = np.full(len(sig), 60.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_drive=1.5,
            resonance_q=24.0,
            filter_solver="newton",
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) < 4.0

    def test_newton_self_oscillation_wakes_from_silence(self) -> None:
        """With silent input at high Q the bootstrap noise should seed
        a self-oscillating tone. Within 300 ms the RMS should rise above
        a non-trivial threshold."""
        n = int(SR * 0.5)
        sig = np.zeros(n, dtype=np.float64)
        cutoff = np.full(n, 500.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_drive=0.3,
            resonance_q=20.0,
            filter_solver="newton",
        )
        assert np.all(np.isfinite(result))
        late = result[int(SR * 0.3) :]
        assert _rms(late) > 1e-4, (
            f"Expected self-oscillation wake-up; late-RMS={_rms(late):.2e}"
        )


class TestNewtonVsAdaaUnderDrive:
    def test_newton_differs_from_adaa_under_heavy_feedback(self) -> None:
        """At high resonance + drive, Newton and ADAA are expected to produce
        measurably different outputs — the Newton solver does not require the
        one-step-behind approximation."""
        n = int(SR * 0.3)
        t = np.arange(n) / SR
        sig = np.sin(2 * np.pi * 200 * t)
        cutoff = np.full(n, 800.0)
        adaa = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_drive=1.0,
            resonance_q=12.0,
            filter_solver="adaa",
        )
        newton = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_drive=1.0,
            resonance_q=12.0,
            filter_solver="newton",
        )
        diff_rms = _rms(adaa - newton)
        ref_rms = _rms(adaa)
        assert diff_rms > 0.001 * ref_rms, (
            "Newton solver should produce measurably different output from ADAA "
            "under heavy feedback."
        )

    def test_newton_preserves_drive_harmonic_content(self) -> None:
        """Newton ladder should still add harmonics under drive (not linearize
        the nonlinearity away)."""
        n = int(SR * 0.5)
        t = np.arange(n) / SR
        sig = np.sin(2 * np.pi * 200 * t)
        cutoff = np.full(n, 2000.0)
        clean = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_drive=0.0,
            filter_solver="newton",
        )
        driven = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_drive=0.5,
            filter_solver="newton",
        )
        clean_hf = _band_energy(clean, 400, 5000)
        driven_hf = _band_energy(driven, 400, 5000)
        assert driven_hf > clean_hf * 1.5


class TestNewtonExistingSurface:
    """The Newton solver must coexist with the rest of the ladder surface:
    filter_morph, bass_compensation, serial HPF, ext feedback, filter_mode."""

    def test_newton_with_filter_morph(self) -> None:
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 1500.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_solver="newton",
            filter_morph=1.5,
            resonance_q=4.0,
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) > 0.01

    def test_newton_with_bass_compensation(self) -> None:
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 600.0)
        no_comp = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_solver="newton",
            resonance_q=12.0,
            bass_compensation=0.0,
        )
        with_comp = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_solver="newton",
            resonance_q=12.0,
            bass_compensation=1.0,
        )
        assert _band_energy(with_comp, 20, 200) > _band_energy(no_comp, 20, 200) * 1.01

    def test_newton_with_serial_hpf(self) -> None:
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 2000.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_solver="newton",
            hpf_cutoff_hz=300.0,
        )
        assert np.all(np.isfinite(result))

    def test_newton_with_ext_feedback(self) -> None:
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 1200.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_solver="newton",
            feedback_amount=0.4,
            feedback_saturation=0.3,
            resonance_q=6.0,
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) < 4.0

    @pytest.mark.parametrize("mode", ["lowpass", "bandpass", "highpass"])
    def test_newton_all_modes_produce_output(self, mode: str) -> None:
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 1000.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_solver="newton",
            filter_mode=mode,
        )
        assert np.all(np.isfinite(result))
        assert np.max(np.abs(result)) > 0.001


class TestNewtonIterations:
    def test_max_iters_one_still_stable(self) -> None:
        """Even with a single Newton step we should stay finite — it's
        strictly no worse than the ADAA approximation."""
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 600.0)
        result = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_solver="newton",
            filter_drive=1.0,
            resonance_q=10.0,
            max_newton_iters=1,
        )
        assert np.all(np.isfinite(result))

    def test_more_iters_converges_to_stable_limit(self) -> None:
        """The scalar Newton solver is extremely well-conditioned for this
        topology: warm-started from the previous sample's y3, with
        A*k << 1, it reaches machine precision in 1-2 Newton steps. So 4
        vs 16 iterations should give the same answer within numerical noise.

        This is a *property* of the 1-D Newton-for-scalar-tanh-ladder setup —
        not a bug. It means we can safely run quality=fast at max_iters=1 and
        still have converged output at typical drives.
        """
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 500.0)

        def run(iters: int) -> np.ndarray:
            return apply_filter(
                sig,
                cutoff_profile=cutoff,
                sample_rate=SR,
                filter_topology="ladder",
                filter_solver="newton",
                filter_drive=1.2,
                resonance_q=14.0,
                max_newton_iters=iters,
            )

        r4 = run(4)
        r16 = run(16)
        assert _rms(r4 - r16) < 1e-8 * (_rms(r16) + 1e-12)


class TestDispatchBackwardsCompatible:
    def test_default_solver_is_newton(self) -> None:
        """The default solver is Newton ('great' quality) so pieces get the
        delay-free feedback path out of the box.  ``filter_solver="adaa"``
        remains available for the lower-quality draft path."""
        sig = _test_signal(0.3)
        cutoff = np.full(len(sig), 1000.0)
        default = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_drive=0.5,
        )
        explicit_newton = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_drive=0.5,
            filter_solver="newton",
        )
        np.testing.assert_allclose(default, explicit_newton)

    def test_unknown_solver_raises(self) -> None:
        sig = _test_signal(0.1)
        cutoff = np.full(len(sig), 1000.0)
        with pytest.raises(ValueError, match="filter_solver"):
            apply_filter(
                sig,
                cutoff_profile=cutoff,
                sample_rate=SR,
                filter_topology="ladder",
                filter_solver="mystery",
            )
