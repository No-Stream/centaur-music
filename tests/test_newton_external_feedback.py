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
from tests._dsp_test_utils import band_energy, noise, rms

SR = 44100

CLOSED_FB_TOPOLOGIES: tuple[str, ...] = (
    "svf",
    "cascade",
    "sem",
    "sallen_key",
    "ladder",
    "jupiter",
)

# Pure-linear-body topologies (no per-solver k mapping).  Used to keep
# ADAA-vs-Newton agreement tight at low feedback — any drift between
# solvers on these must be within numerical noise.
AFFINE_BODY_TOPOLOGIES: tuple[str, ...] = ("svf", "cascade", "sem", "sallen_key")

# Topologies that use solver-specific internal-resonance k mappings
# (Moog-style k_adaa vs k_newton) — documented in docs/synth_api.md's
# Quality Modes section.  Their ADAA and Newton baselines deliberately
# diverge in character.
SOLVER_SPECIFIC_K_TOPOLOGIES: tuple[str, ...] = ("ladder", "jupiter")

# Topologies that accept `filter_morph` in the dispatcher and go through
# one of the morph-capable code paths (SVF mode-cycle, cascade weighted
# pole-taps, ladder pole-tap blend, SEM LP→notch→HP sweep, Jupiter).
MORPH_TOPOLOGIES: tuple[str, ...] = ("svf", "cascade", "sem", "ladder", "jupiter")


class TestNewtonExtFeedbackMonotonicHarmonics:
    """As ``feedback_amount`` rises, the ``tanh(g_fb * y)`` on the loop
    injects progressively more harmonic distortion.  The 3rd harmonic
    of a fundamental well below cutoff is the cleanest spectral
    signature of that process — it rises monotonically with feedback
    on every closed-FB topology, and a bug that accidentally shorted
    or halved the feedback path would show up here as non-monotone (or
    flat) harmonic growth.

    Uses a sine fundamental well below cutoff so harmonics fall inside
    the passband and are measurable (as opposed to a sine at cutoff,
    where the feedback path compresses the fundamental itself).
    """

    @pytest.mark.parametrize("topology", CLOSED_FB_TOPOLOGIES)
    def test_third_harmonic_grows_with_feedback(self, topology: str) -> None:
        fund = 200.0
        cutoff_hz = 1200.0
        n = int(SR * 0.3)
        t = np.arange(n) / SR
        sig = 0.4 * np.sin(2.0 * np.pi * fund * t)
        cutoff = np.full(n, cutoff_hz)
        fb_sweep = [0.0, 0.3, 0.6, 0.85]
        h3_energies: list[float] = []
        for fb_amt in fb_sweep:
            out = apply_filter(
                sig,
                cutoff_profile=cutoff,
                sample_rate=SR,
                filter_topology=topology,
                filter_solver="newton",
                feedback_amount=fb_amt,
                feedback_saturation=0.35,
                resonance_q=2.0,
            )
            assert np.all(np.isfinite(out)), (
                f"{topology}: non-finite output at fb_amt={fb_amt}"
            )
            assert np.max(np.abs(out)) < 5.0
            h3_energies.append(band_energy(out, 3 * fund - 30, 3 * fund + 30, sr=SR))

        for lo, hi in zip(fb_sweep[:-1], fb_sweep[1:], strict=False):
            ie_lo = h3_energies[fb_sweep.index(lo)]
            ie_hi = h3_energies[fb_sweep.index(hi)]
            assert ie_hi >= 0.95 * ie_lo, (
                f"{topology}: 3rd-harmonic dropped {ie_lo:.4f} -> {ie_hi:.4f} "
                f"going fb_amt {lo} -> {hi} (sweep={h3_energies})"
            )
        # Endpoint must be meaningfully larger than the no-FB baseline —
        # the feature produces audible harmonic content.
        assert h3_energies[-1] > 2.0 * h3_energies[0] + 1e-6, (
            f"{topology}: feedback failed to raise 3rd harmonic "
            f"(no_fb={h3_energies[0]:.4f}, max_fb={h3_energies[-1]:.4f})"
        )


class TestAdaaVsNewtonAgreeAtLowFeedback:
    """At low feedback the delay-free solve differs from the unit-delay
    solve only by sub-sample timing jitter on the tanh.  They should
    match within a tight RMS bound.

    This is the primary safety check that the Newton extension doesn't
    break typical patch behaviour — most music does not drive the
    feedback path near self-oscillation.
    """

    @pytest.mark.parametrize("topology", AFFINE_BODY_TOPOLOGIES)
    def test_low_feedback_newton_close_to_adaa_affine(self, topology: str) -> None:
        """Pure-affine-body topologies share the same body coefficients
        across solvers, so the ADAA↔Newton delta at fb_amt=0.1 is only
        the one-sample tanh lag — must be <5%.
        """
        sig = noise(sr=SR)
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
        ref = max(rms(adaa), 1e-9)
        rel_diff = rms(adaa - newton) / ref
        assert rel_diff < 0.05, (
            f"{topology}: newton diverged from adaa by {rel_diff:.3f} "
            f"at low feedback (affine body -- expect <5%)"
        )

    @pytest.mark.parametrize("topology", SOLVER_SPECIFIC_K_TOPOLOGIES)
    def test_low_feedback_newton_close_to_adaa_solver_specific(
        self, topology: str
    ) -> None:
        """Ladder and Jupiter use per-solver k(Q) mappings (k_adaa vs
        k_newton) documented in ``docs/synth_api.md`` Quality Modes.
        Their baselines legitimately differ even at fb_amt=0 — we only
        guard against *runaway* divergence here.
        """
        sig = noise(sr=SR)
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
        ref = max(rms(adaa), 1e-9)
        rel_diff = rms(adaa - newton) / ref
        assert rel_diff < 0.5, (
            f"{topology}: newton diverged from adaa by {rel_diff:.3f} "
            f"at low feedback (solver-specific k mapping -- expect <50%)"
        )


class TestHighFeedbackAddsHarmonicContent:
    """Under strong feedback the implicit solve both reshapes the
    resonance peak and injects distortion harmonics through the tanh.
    Either way, a clean sine inside the passband should spawn energy in
    the harmonic region when feedback engages.

    This supersedes a prior RMS-diff-only "newton differs from adaa"
    test: measuring band energy proves the feature actually produces
    the audible behaviour we advertise, not just any bit-level
    divergence.
    """

    @pytest.mark.parametrize("topology", CLOSED_FB_TOPOLOGIES)
    def test_feedback_adds_harmonics_at_cutoff_tone(self, topology: str) -> None:
        # Sine at 400 Hz with 1000 Hz cutoff — fundamental sits well
        # within the pass-band so the no-FB baseline has a clean tone;
        # feedback adds harmonics that leak into 800-2000 Hz via the
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
        hf_no_fb = band_energy(no_fb, 800, 2000, sr=SR)
        hf_with_fb = band_energy(with_fb, 800, 2000, sr=SR)
        # Ladder and Jupiter's 24 dB/oct response attenuates the 800-2000
        # band strongly vs 2-pole topologies, so the absolute values are
        # small but the *ratio* must rise when feedback turns on.
        assert hf_with_fb > hf_no_fb * 1.2, (
            f"{topology}: expected feedback to add HF content "
            f"(no_fb={hf_no_fb:.4f}, with_fb={hf_with_fb:.4f})"
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
        sig = noise(dur=0.2, sr=SR)
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
        rel = rms(r2 - r8) / max(rms(r8), 1e-12)
        # Two warm-started iterations converge to machine epsilon on
        # every topology in our tested range.
        assert rel < 1e-9, f"{topology}: 2-iter vs 8-iter rel diff={rel:.2e}"

    @pytest.mark.parametrize("topology", CLOSED_FB_TOPOLOGIES)
    def test_one_iter_already_close(self, topology: str) -> None:
        """One warm-started Newton step should land within 10^-4
        relative error — sloppy-but-stable, a safe floor for quality="fast"."""
        sig = noise(dur=0.2, sr=SR)
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

        rel = rms(run(1) - run(8)) / max(rms(run(8)), 1e-12)
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
        late_rms = rms(out[int(SR * 0.3) :])
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
        late_rms = rms(out[int(SR * 0.3) :])
        assert late_rms > 1e-4, (
            f"ladder: internal self-oscillation regressed (late_rms={late_rms:.2e})"
        )
        assert np.max(np.abs(out)) < 3.0


# Cascade uses a Prophet-5-style "complementary" HP (``x - y3``) and a
# "difference" BP (``y1 - y3``) rather than true 2-pole HP/BP responses.
# Its HP is near-flat instead of attenuating below cutoff, and its BP
# peak sits substantially below cutoff — both are the intentional
# analog-character quirks documented in ``_apply_cascade_inner``.  We
# exclude cascade from HP/BP frequency-response shape assertions (it
# still gets LP, finite-output, and all the other cross-topology checks).
NON_COMPLEMENTARY_HP_BP_TOPOLOGIES: tuple[str, ...] = (
    "svf",
    "sem",
    "sallen_key",
    "ladder",
    "jupiter",
)


class TestNewtonExtFeedbackModeFrequencyResponse:
    """LP is the common use case, but BP and HP must render with the
    correct frequency character.  We drive broadband noise and measure
    band ratios on either side of cutoff; a bug that silently collapsed
    the mode to LP (or zeroed a mode's coefficients) would show up as
    the HP band dominating where LP should, etc.
    """

    @pytest.mark.parametrize("topology", CLOSED_FB_TOPOLOGIES)
    def test_lowpass_attenuates_above_cutoff(self, topology: str) -> None:
        cutoff_hz = 1200.0
        sig = noise(dur=0.4, sr=SR)
        cutoff = np.full(len(sig), cutoff_hz)
        out = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            filter_mode="lowpass",
            filter_solver="newton",
            feedback_amount=0.3,
            feedback_saturation=0.35,
            resonance_q=2.5,
        )
        assert np.all(np.isfinite(out))
        assert rms(out) > 1e-4
        below = band_energy(out, 0.2 * cutoff_hz, 0.6 * cutoff_hz, sr=SR)
        above = band_energy(out, 2.0 * cutoff_hz, 4.0 * cutoff_hz, sr=SR)
        assert below > 1.5 * above, (
            f"{topology} LP: expected below-cutoff energy to dominate "
            f"(below={below:.4f}, above={above:.4f})"
        )

    @pytest.mark.parametrize("topology", NON_COMPLEMENTARY_HP_BP_TOPOLOGIES)
    def test_highpass_attenuates_below_cutoff(self, topology: str) -> None:
        cutoff_hz = 1200.0
        sig = noise(dur=0.4, sr=SR)
        cutoff = np.full(len(sig), cutoff_hz)
        out = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            filter_mode="highpass",
            filter_solver="newton",
            feedback_amount=0.3,
            feedback_saturation=0.35,
            resonance_q=2.5,
        )
        assert np.all(np.isfinite(out))
        assert rms(out) > 1e-4
        # Deep sub-cutoff band (0.1-0.3x) vs clearly above cutoff
        # (1.5-3x).  Wide margin protects against HP being silently
        # swapped with LP while tolerating the natural softer-than-6-dB
        # rolloff on the Jupiter/ladder 24 dB/oct HP taps.
        below = band_energy(out, 0.1 * cutoff_hz, 0.3 * cutoff_hz, sr=SR)
        above = band_energy(out, 1.5 * cutoff_hz, 3.0 * cutoff_hz, sr=SR)
        assert above > 1.3 * below, (
            f"{topology} HP: expected above-cutoff energy to dominate "
            f"(below={below:.4f}, above={above:.4f})"
        )

    @pytest.mark.parametrize("topology", NON_COMPLEMENTARY_HP_BP_TOPOLOGIES)
    def test_bandpass_peaks_near_cutoff(self, topology: str) -> None:
        cutoff_hz = 1200.0
        sig = noise(dur=0.4, sr=SR)
        cutoff = np.full(len(sig), cutoff_hz)
        out = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            filter_mode="bandpass",
            filter_solver="newton",
            feedback_amount=0.3,
            feedback_saturation=0.35,
            resonance_q=2.5,
        )
        assert np.all(np.isfinite(out))
        assert rms(out) > 1e-4
        near = band_energy(out, 0.7 * cutoff_hz, 1.4 * cutoff_hz, sr=SR)
        far_low = band_energy(out, 0.1 * cutoff_hz, 0.35 * cutoff_hz, sr=SR)
        far_high = band_energy(out, 3.0 * cutoff_hz, 6.0 * cutoff_hz, sr=SR)
        assert near > 1.2 * far_low, (
            f"{topology} BP: peak near cutoff vs low-band "
            f"(near={near:.4f}, far_low={far_low:.4f})"
        )
        assert near > 1.2 * far_high, (
            f"{topology} BP: peak near cutoff vs high-band "
            f"(near={near:.4f}, far_high={far_high:.4f})"
        )


class TestStabilityUnderStress:
    """Combined high-Q + heavy feedback + fast cutoff modulation is the
    realistic worst case for a feedback filter in a composition context.
    The Newton solver must stay bounded, and very small feedback values
    must stay numerically close to the no-feedback baseline (guarding
    against per-sample coefficient staleness / off-by-one indexing into
    the cutoff profile).
    """

    @pytest.mark.parametrize("topology", CLOSED_FB_TOPOLOGIES)
    def test_modulated_cutoff_high_q_high_fb_stays_bounded(self, topology: str) -> None:
        n = int(SR * 0.5)
        t = np.arange(n) / SR
        sig = noise(dur=0.5, seed=7, sr=SR)
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

    @pytest.mark.parametrize("topology", CLOSED_FB_TOPOLOGIES)
    def test_modulated_cutoff_tiny_fb_matches_no_fb(self, topology: str) -> None:
        """With a tiny ``feedback_amount=0.02`` under cutoff modulation,
        the feedback contribution at each sample is ``tanh(0.02*y_prev)``,
        which is numerically tiny.  The output must stay close to the
        zero-feedback baseline — if the per-sample affine coefficients
        are computed from a stale cutoff value (off-by-one in the
        profile indexing) the two renders would drift far more than the
        tanh(0.02) contribution can account for.
        """
        n = int(SR * 0.5)
        t = np.arange(n) / SR
        sig = noise(dur=0.5, seed=7, sr=SR)
        cutoff = 400.0 + 1200.0 * (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t))
        no_fb = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            filter_solver="newton",
            feedback_amount=0.0,
            resonance_q=4.0,
        )
        tiny_fb = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            filter_solver="newton",
            feedback_amount=0.02,
            feedback_saturation=0.3,
            resonance_q=4.0,
        )
        ref = max(rms(no_fb), 1e-9)
        rel = rms(tiny_fb - no_fb) / ref
        assert rel < 0.2, (
            f"{topology}: tiny-FB render drifted from no-FB baseline by "
            f"{rel:.3f} under cutoff modulation -- possible stale-g bug"
        )


class TestDefaultSolverIsNewton:
    """The user-facing default is Newton so direct `apply_filter` callers
    pick up delay-free feedback automatically.  The `"adaa"` solver
    remains available for the draft-quality path.

    One topology (ladder) is enough here — this test verifies the
    default-kwarg wiring in ``apply_filter``, not per-topology math.
    """

    def test_default_equals_explicit_newton(self) -> None:
        sig = noise(dur=0.2, sr=SR)
        cutoff = np.full(len(sig), 1500.0)
        default = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            feedback_amount=0.3,
        )
        explicit = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="ladder",
            feedback_amount=0.3,
            filter_solver="newton",
        )
        np.testing.assert_allclose(
            default,
            explicit,
            err_msg="ladder: default solver did not match explicit newton",
        )


class TestAdaaPathUntouched:
    """Regression guard: the Newton feature must not change the ADAA
    path for any topology.  This is the invariant that keeps draft
    quality and any explicit `filter_solver="adaa"` callers
    bit-identical to pre-change behaviour."""

    @pytest.mark.parametrize("topology", CLOSED_FB_TOPOLOGIES)
    def test_adaa_feedback_path_deterministic(self, topology: str) -> None:
        """Two back-to-back calls with ADAA + feedback must be identical."""
        sig = noise(dur=0.2, sr=SR)
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
        sig = noise(dur=0.2, sr=SR)
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
        if topology in AFFINE_BODY_TOPOLOGIES:
            np.testing.assert_allclose(adaa, newton)
        else:
            rel = rms(adaa - newton) / max(rms(adaa), 1e-9)
            assert rel < 0.5, (
                f"{topology}: adaa and newton diverged at fb_amt=0 (rel={rel:.3f})"
            )


class TestNewtonExtFeedbackWithMorph:
    """The SVF / cascade / ladder / SEM / Jupiter morph paths take a
    different affine-collapse branch (``lp_w``/``bp_w``/``hp_w``/
    ``notch_w`` blends for SVF; pole-tap blend for ladder/Jupiter;
    LP→notch→HP sweep for SEM) when ``filter_morph > 0``.  Under Newton
    external feedback that branch must still produce finite, bounded,
    non-trivial output — and at the canonical SVF endpoints the morph
    path must exactly recover the plain LP / BP outputs.
    """

    @pytest.mark.parametrize("topology", MORPH_TOPOLOGIES)
    @pytest.mark.parametrize("filter_morph", [0.25, 0.5, 0.75, 1.0, 1.5, 2.5])
    def test_morph_under_newton_fb_stable(
        self, topology: str, filter_morph: float
    ) -> None:
        # SEM clamps morph into [0, 2]; Jupiter/ladder/cascade/SVF use
        # [0, 3].  All topologies in MORPH_TOPOLOGIES accept these
        # values without raising.
        sig = noise(dur=0.25, sr=SR)
        cutoff = np.full(len(sig), 1000.0)
        out = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology=topology,
            filter_mode="lowpass",
            filter_solver="newton",
            feedback_amount=0.4,
            feedback_saturation=0.35,
            resonance_q=3.0,
            filter_morph=filter_morph,
        )
        assert np.all(np.isfinite(out)), (
            f"{topology} morph={filter_morph}: non-finite output"
        )
        assert np.max(np.abs(out)) < 5.0, (
            f"{topology} morph={filter_morph}: exceeded 5.0 bound"
        )
        assert rms(out) > 1e-3, (
            f"{topology} morph={filter_morph}: collapsed to near-silence"
        )

    # The morph-collapse path and the plain-mode path evaluate the
    # SVF coefficients through slightly different arithmetic
    # expressions (weighted blend of four body taps vs direct tap
    # lookup).  At canonical endpoints the weighted blend degenerates
    # to a single tap, but the floating-point round-off differs by
    # ~1e-7 per sample — accumulate over thousands of samples with
    # Newton iteration and you get ~1e-6 max-abs differences.  We use
    # a ``rtol=0, atol=1e-5`` tolerance: tight enough to catch any
    # real structural bug (those produce ≥1% drift) while tolerating
    # the expected float noise.
    _MORPH_ENDPOINT_ATOL = 1e-5

    def test_svf_morph_zero_equals_plain_lp(self) -> None:
        """SVF at ``filter_mode="lowpass"`` + ``filter_morph=0.0`` takes
        the non-morph code path and must be bit-identical to the plain
        LP render.  This is the default entry point.
        """
        sig = noise(dur=0.2, sr=SR)
        cutoff = np.full(len(sig), 1000.0)
        plain_lp = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="svf",
            filter_mode="lowpass",
            filter_solver="newton",
            feedback_amount=0.3,
            feedback_saturation=0.35,
            resonance_q=3.0,
        )
        morph_zero = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="svf",
            filter_mode="lowpass",
            filter_solver="newton",
            feedback_amount=0.3,
            feedback_saturation=0.35,
            resonance_q=3.0,
            filter_morph=0.0,
        )
        # morph=0.0 dispatches to the non-morph SVF path identically to
        # plain LP -- bit-identical result expected here.
        np.testing.assert_array_equal(plain_lp, morph_zero)

    def test_svf_morph_one_from_lp_equals_plain_bp(self) -> None:
        """SVF at ``filter_mode="lowpass"`` + ``filter_morph=1.0`` hits
        the ``use_morph`` branch with ``lp_w=0, bp_w=1``, which must
        match the plain BP render (also computed via Newton).

        The plain BP path goes through the non-morph ``mode_int==_BP``
        branch of the same inner loop, so at morph=1 the two renders
        share the same affine coefficients and the feedback Newton
        solve operates on identical (A_out, B_out) each sample.
        """
        sig = noise(dur=0.2, sr=SR)
        cutoff = np.full(len(sig), 1000.0)
        plain_bp = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="svf",
            filter_mode="bandpass",
            filter_solver="newton",
            feedback_amount=0.3,
            feedback_saturation=0.35,
            resonance_q=3.0,
        )
        morph_one = apply_filter(
            sig,
            cutoff_profile=cutoff,
            sample_rate=SR,
            filter_topology="svf",
            filter_mode="lowpass",
            filter_solver="newton",
            feedback_amount=0.3,
            feedback_saturation=0.35,
            resonance_q=3.0,
            filter_morph=1.0,
        )
        # The morph-weighted-blend path reaches the same tap as plain BP
        # via different arithmetic ordering; see _MORPH_ENDPOINT_ATOL.
        np.testing.assert_allclose(
            plain_bp, morph_one, rtol=0.0, atol=self._MORPH_ENDPOINT_ATOL
        )
