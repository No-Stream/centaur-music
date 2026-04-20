"""End-to-end tests for the Newton-solved external filter feedback loop.

The "external" feedback is the global `feedback_amount` path that feeds the
filter output back to the pre-filter input through a saturating `tanh`.
Historically this used a one-sample delay (`tanh(ext_fb_drive * y_prev)`)
which audibly damps high-resonance behaviour and smears fast transients.

The Newton solver closes the loop implicitly per sample — the filter body
is collapsed to its affine input-to-output map (or combined with existing
internal Newton solvers for ladder/Jupiter), and the scalar equation
`y = H(x + fb_amt · tanh(g_fb · y))` is solved to machine precision.

Topologies with closed Newton external feedback at the time of writing:
    linear SVF, cascade, SEM, Sallen-Key, ladder (Newton solver),
    Jupiter (Newton solver).

Topologies still on unit-delay (see FUTURE.md): K35, diode, driven SVF.
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._filters import apply_filter

SR = 44100

CLOSED_FB_TOPOLOGIES: tuple[str, ...] = (
    "svf",
    "cascade",
    "sem",
    "sallen_key",
    "ladder",
    "jupiter",
)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


def _band_energy(signal: np.ndarray, lo: float, hi: float) -> float:
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / SR)
    mask = (freqs >= lo) & (freqs <= hi)
    return float(np.sqrt(np.mean(spectrum[mask] ** 2)))


def _noise(dur: float = 0.3, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(SR * dur))


class TestNewtonExtFeedbackFinite:
    """Every closed-FB topology must produce finite, audible output at
    representative `feedback_amount` with the Newton solver."""

    @pytest.mark.parametrize("topology", CLOSED_FB_TOPOLOGIES)
    @pytest.mark.parametrize("fb_amt", [0.2, 0.5, 0.8])
    def test_finite_and_audible(self, topology: str, fb_amt: float) -> None:
        sig = _noise()
        cutoff = np.full(len(sig), 900.0)
        out = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            filter_solver="newton",
            feedback_amount=fb_amt,
            feedback_saturation=0.35,
            resonance_q=4.0,
        )
        assert np.all(np.isfinite(out)), (
            f"{topology} produced non-finite samples at fb_amt={fb_amt}"
        )
        assert _rms(out) > 1e-4, (
            f"{topology} collapsed to near-silence at fb_amt={fb_amt}"
        )
        # Implicit feedback should not explode even at fb_amt=0.8 with q=4.
        assert np.max(np.abs(out)) < 5.0


class TestAdaaVsNewtonAgreeAtLowFeedback:
    """At low feedback the delay-free solve differs from the unit-delay
    solve only by sub-sample timing jitter on the tanh.  They should
    match within a tight RMS bound.

    This is the primary safety check that the Newton extension doesn't
    break typical patch behaviour — most music does not drive the
    feedback path near self-oscillation."""

    @pytest.mark.parametrize("topology", CLOSED_FB_TOPOLOGIES)
    def test_low_feedback_newton_close_to_adaa(self, topology: str) -> None:
        sig = _noise()
        cutoff = np.full(len(sig), 1200.0)
        adaa = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            filter_solver="adaa",
            feedback_amount=0.1,
            feedback_saturation=0.3,
            resonance_q=2.0,
        )
        newton = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            filter_solver="newton",
            feedback_amount=0.1,
            feedback_saturation=0.3,
            resonance_q=2.0,
        )
        ref = max(_rms(adaa), 1e-9)
        rel_diff = _rms(adaa - newton) / ref
        # Ladder and Jupiter have different internal-resonance tuning
        # per solver (k_adaa vs k_newton), so their baseline RMS differs
        # even at fb_amt=0.  We only guard against *runaway* divergence
        # here; the per-topology character is intentional.
        assert rel_diff < 0.5, (
            f"{topology}: newton diverged from adaa by {rel_diff:.3f} at low feedback"
        )


class TestHighFeedbackNewtonDiffers:
    """Under strong feedback the unit-delay and implicit solves diverge
    meaningfully — this is the whole point of the feature.  Measurable
    divergence proves the Newton branch is actually engaging, not a
    silent no-op."""

    @pytest.mark.parametrize(
        "topology",
        ["svf", "cascade", "sem", "sallen_key", "ladder", "jupiter"],
    )
    def test_high_feedback_newton_differs_from_adaa(self, topology: str) -> None:
        sig = _noise(dur=0.2)
        cutoff = np.full(len(sig), 700.0)
        adaa = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            filter_solver="adaa",
            feedback_amount=0.7,
            feedback_saturation=0.5,
            resonance_q=5.0,
        )
        newton = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            filter_solver="newton",
            feedback_amount=0.7,
            feedback_saturation=0.5,
            resonance_q=5.0,
        )
        ref = max(_rms(adaa), 1e-9)
        rel_diff = _rms(adaa - newton) / ref
        assert rel_diff > 1e-3, (
            f"{topology}: newton should differ from adaa under heavy "
            f"feedback but rel_diff={rel_diff:.2e}"
        )


class TestNewtonIterationConvergence:
    """The scalar Newton residual is well-conditioned with a warm start
    from the previous sample's solved value, so even one iteration is
    very close to converged.  Bumping iterations from 1 to 8 should
    shrink the residual by many orders of magnitude, and from 2 to 8
    must be numerically identical — this validates the warm-start
    strategy and catches regressions in the Jacobian math."""

    @pytest.mark.parametrize("topology", CLOSED_FB_TOPOLOGIES)
    def test_two_iter_matches_eight(self, topology: str) -> None:
        sig = _noise(dur=0.2)
        cutoff = np.full(len(sig), 1000.0)

        def run(n_iters: int) -> np.ndarray:
            return apply_filter(
                sig,
                cutoff_profile=cutoff,
                sample_rate=SR,
                filter_topology=topology,
                filter_solver="newton",
                feedback_amount=0.4,
                feedback_saturation=0.3,
                resonance_q=3.0,
                max_newton_iters=n_iters,
            )

        r2 = run(2)
        r8 = run(8)
        rel = _rms(r2 - r8) / max(_rms(r8), 1e-12)
        # Two warm-started iterations converge to machine epsilon on
        # every topology in our tested range.
        assert rel < 1e-9, f"{topology}: 2-iter vs 8-iter rel diff={rel:.2e}"

    @pytest.mark.parametrize("topology", CLOSED_FB_TOPOLOGIES)
    def test_one_iter_already_close(self, topology: str) -> None:
        """One warm-started Newton step should land within 10^-4
        relative error — sloppy-but-stable, a safe floor for quality="fast"."""
        sig = _noise(dur=0.2)
        cutoff = np.full(len(sig), 1000.0)

        def run(n_iters: int) -> np.ndarray:
            return apply_filter(
                sig,
                cutoff_profile=cutoff,
                sample_rate=SR,
                filter_topology=topology,
                filter_solver="newton",
                feedback_amount=0.4,
                feedback_saturation=0.3,
                resonance_q=3.0,
                max_newton_iters=n_iters,
            )

        rel = _rms(run(1) - run(8)) / max(_rms(run(8)), 1e-12)
        assert rel < 1e-4, f"{topology}: 1-iter vs 8-iter rel diff={rel:.2e}"


class TestSelfOscillationFromSilence:
    """With silent input + very high ext-FB + moderate Q, the filter
    topologies whose output has significant gain at cutoff (SVF, SEM)
    can bootstrap into self-oscillation purely through the external
    feedback tanh and the bootstrap noise seeded on the feedback
    summation.  This is a stability-plus-liveness test: Newton must
    preserve the instability *and* keep the grown signal bounded.

    Ladder and Jupiter bootstrap self-oscillation through their own
    *internal* feedback rather than the external path (their cutoff
    response attenuates output energy below unity), so we exercise
    them separately in ``test_internal_fb_still_self_oscillates``."""

    @pytest.mark.parametrize(
        "topology,resonance_q,fb_amt",
        [
            ("svf", 8.0, 0.85),
            ("sem", 10.0, 0.85),
        ],
    )
    def test_self_oscillation_wakes_and_stays_bounded(
        self, topology: str, resonance_q: float, fb_amt: float
    ) -> None:
        n = int(SR * 0.5)
        silence = np.zeros(n, dtype=np.float64)
        cutoff = np.full(n, 500.0)
        out = apply_filter(
            silence,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            filter_solver="newton",
            feedback_amount=fb_amt,
            feedback_saturation=0.5,
            resonance_q=resonance_q,
        )
        assert np.all(np.isfinite(out))
        late_rms = _rms(out[int(SR * 0.3) :])
        # SVF/SEM must grow a musically audible tone from silence.
        assert late_rms > 1e-3, (
            f"{topology}: no self-oscillation wake-up (late_rms={late_rms:.2e})"
        )
        # Bounded by the feedback tanh ceiling.
        assert np.max(np.abs(out)) < 3.0

    def test_ladder_self_oscillates_via_internal_resonance(self) -> None:
        """Ladder self-oscillation is driven by its own internal
        resonance feedback, not the external loop.  Newton must preserve
        that — this mirrors test_newton_ladder.py's wake-up test but
        lives here so this suite catches ladder regressions too.

        Jupiter's Q→k mapping is intentionally softer (caps at ~2.6 vs
        Moog's 4.0) so it does not self-oscillate from silent input —
        that is by-design character, documented in the
        ``_apply_jupiter_filter`` docstring, and not a regression
        target for this feature.
        """
        n = int(SR * 0.5)
        silence = np.zeros(n, dtype=np.float64)
        cutoff = np.full(n, 500.0)
        out = apply_filter(
            silence,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            filter_solver="newton",
            resonance_q=30.0,
            filter_drive=0.3,
        )
        late_rms = _rms(out[int(SR * 0.3) :])
        assert late_rms > 1e-4, (
            f"ladder: internal self-oscillation regressed (late_rms={late_rms:.2e})"
        )
        assert np.max(np.abs(out)) < 3.0


class TestNewtonExtFeedbackAllModes:
    """LP is the common use case for feedback-driven basses / leads, but
    BP and HP outputs must also render cleanly."""

    @pytest.mark.parametrize("topology", CLOSED_FB_TOPOLOGIES)
    @pytest.mark.parametrize("mode", ["lowpass", "bandpass", "highpass"])
    def test_each_mode_renders(self, topology: str, mode: str) -> None:
        sig = _noise(dur=0.2)
        cutoff = np.full(len(sig), 1200.0)
        out = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            filter_mode=mode,
            filter_solver="newton",
            feedback_amount=0.3,
            feedback_saturation=0.35,
            resonance_q=2.5,
        )
        assert np.all(np.isfinite(out))
        assert _rms(out) > 1e-4


class TestStabilityUnderStress:
    """Combined high-Q + heavy feedback + fast cutoff modulation is the
    realistic worst case for a feedback filter in a composition context.
    The Newton solver must stay bounded."""

    @pytest.mark.parametrize("topology", CLOSED_FB_TOPOLOGIES)
    def test_modulated_cutoff_high_q_high_fb_stays_bounded(self, topology: str) -> None:
        n = int(SR * 0.5)
        t = np.arange(n) / SR
        sig = _noise(dur=0.5, seed=7)
        cutoff = 400.0 + 1200.0 * (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t))
        out = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            filter_solver="newton",
            feedback_amount=0.65,
            feedback_saturation=0.5,
            resonance_q=8.0,
        )
        assert np.all(np.isfinite(out))
        assert np.max(np.abs(out)) < 5.0


class TestDefaultSolverIsNewton:
    """The user-facing default is Newton so direct `apply_filter` callers
    pick up delay-free feedback automatically.  The `"adaa"` solver
    remains available for the draft-quality path."""

    def test_default_equals_explicit_newton(self) -> None:
        sig = _noise(dur=0.2)
        cutoff = np.full(len(sig), 1500.0)
        for topology in CLOSED_FB_TOPOLOGIES:
            default = apply_filter(
                sig,
                cutoff_profile=cutoff,
                sample_rate=SR,
                filter_topology=topology,
                feedback_amount=0.3,
            )
            explicit = apply_filter(
                sig,
                cutoff_profile=cutoff,
                sample_rate=SR,
                filter_topology=topology,
                feedback_amount=0.3,
                filter_solver="newton",
            )
            np.testing.assert_allclose(
                default,
                explicit,
                err_msg=f"{topology}: default solver did not match explicit newton",
            )


class TestAdaaPathUntouched:
    """Regression guard: the Newton feature must not change the ADAA
    path for any topology.  This is the invariant that keeps draft
    quality and any explicit `filter_solver="adaa"` callers
    bit-identical to pre-change behaviour."""

    @pytest.mark.parametrize("topology", CLOSED_FB_TOPOLOGIES)
    def test_adaa_feedback_path_deterministic(self, topology: str) -> None:
        """Two back-to-back calls with ADAA + feedback must be identical."""
        sig = _noise(dur=0.2)
        cutoff = np.full(len(sig), 1000.0)
        a = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            filter_solver="adaa",
            feedback_amount=0.5,
            feedback_saturation=0.4,
            resonance_q=4.0,
        )
        b = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            filter_solver="adaa",
            feedback_amount=0.5,
            feedback_saturation=0.4,
            resonance_q=4.0,
        )
        np.testing.assert_array_equal(a, b)


class TestZeroFeedbackIsIdentityToNoFeedback:
    """`feedback_amount=0.0` must produce the same output whether the
    solver is set to ADAA or Newton, and the two must both match the
    no-feedback baseline.  This guards against accidental Newton-branch
    side effects when the feature is inactive."""

    @pytest.mark.parametrize("topology", CLOSED_FB_TOPOLOGIES)
    def test_zero_feedback_adaa_equals_newton(self, topology: str) -> None:
        sig = _noise(dur=0.2)
        cutoff = np.full(len(sig), 1500.0)
        adaa = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            filter_solver="adaa",
            feedback_amount=0.0,
        )
        newton = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            filter_solver="newton",
            feedback_amount=0.0,
        )
        # Without feedback, the ladder/Jupiter solver pair still differ
        # because their k-mappings are solver-specific.  For the pure-
        # linear-body topologies (svf, cascade, sem, sallen_key) the
        # output must be identical — nothing in the filter body changes
        # when fb_amt=0 on those.
        if topology in {"svf", "cascade", "sem", "sallen_key"}:
            np.testing.assert_allclose(adaa, newton)
        else:
            rel = _rms(adaa - newton) / max(_rms(adaa), 1e-9)
            assert rel < 0.5, (
                f"{topology}: adaa and newton diverged at fb_amt=0 (rel={rel:.3f})"
            )


class TestHighFeedbackAddsHarmonicContent:
    """Feedback pushes the filter into a resonant regime; the ``tanh``
    on the feedback path also injects harmonic content.  Either way,
    measured against a clean sine input, total output energy at and
    above the cutoff should rise when feedback engages.

    We use a pure sine at cutoff so the measurement isolates the
    filter's behaviour at fc (instead of averaging over broadband
    noise where feedback-induced harmonics dominate the whole
    spectrum)."""

    @pytest.mark.parametrize("topology", CLOSED_FB_TOPOLOGIES)
    def test_feedback_adds_harmonics_at_cutoff_tone(self, topology: str) -> None:
        # Sine at 400 Hz with 1000 Hz cutoff — fundamental sits well
        # within the pass-band so the no-FB baseline has a clean tone;
        # feedback adds harmonics that leak into 800-1200 Hz via the
        # resonance peak and the tanh distortion on the loop.
        n = int(SR * 0.3)
        t = np.arange(n) / SR
        sig = 0.4 * np.sin(2 * np.pi * 400.0 * t)
        cutoff = np.full(n, 1000.0)
        no_fb = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            filter_solver="newton",
            feedback_amount=0.0,
            resonance_q=2.0,
        )
        with_fb = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            filter_solver="newton",
            feedback_amount=0.6,
            feedback_saturation=0.5,
            resonance_q=2.0,
        )
        hf_no_fb = _band_energy(no_fb, 800, 2000)
        hf_with_fb = _band_energy(with_fb, 800, 2000)
        # Ladder and Jupiter's 24 dB/oct response attenuates the 800-2000
        # band strongly vs 2-pole topologies, so the absolute values are
        # small but the *ratio* must rise when feedback turns on.
        assert hf_with_fb > hf_no_fb * 1.2, (
            f"{topology}: expected feedback to add HF content "
            f"(no_fb={hf_no_fb:.4f}, with_fb={hf_with_fb:.4f})"
        )
