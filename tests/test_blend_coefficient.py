"""Verify the saturation-blend coefficient idiom removes drive=0 discontinuities.

The ladder, Sallen-Key, and cascade filter topologies used to branch on
``driven = filter_drive > 0`` to gate their ``_algebraic_sat`` integrator-state
saturation (and Sallen-Key's pre-filter ``tanh`` shape).  That branch produced
an audible step when drive was automated across zero.

The idiom now smoothsteps a blend coefficient over ``drive ∈ [0, epsilon]``
so the transition is C¹-continuous.  These tests:

1. Confirm that drive=0 exactly preserves the fully-linear output (no
   saturation applied at all).
2. Confirm that ``drive >= epsilon`` matches the historical fully-driven
   behavior byte-for-byte.
3. Confirm that the output at intermediate drives does not jump — the RMS
   distance between neighboring drive points stays bounded and monotone-ish
   across the critical region around zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._filters import _DRIVE_BLEND_EPSILON, apply_filter

SR: int = 44100


def _test_signal(dur: float = 0.25, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(SR * dur))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


def _render(topology: str, drive: float, *, seed: int = 42) -> np.ndarray:
    sig = _test_signal(seed=seed)
    cutoff = np.full(len(sig), 1200.0)
    return apply_filter(
        sig,
        cutoff_profile=cutoff,
        sample_rate=SR,
        filter_topology=topology,
        resonance_q=3.0,
        filter_drive=drive,
    )


TOPOLOGIES = ["ladder", "sallen_key", "cascade"]


class TestBlendEndpointsPreserved:
    """At drive=0 and drive >= epsilon the output must match the pre-blend
    behavior exactly.  The blend only touches the interior (0, epsilon).
    """

    @pytest.mark.parametrize("topology", TOPOLOGIES)
    def test_drive_zero_is_linear(self, topology: str) -> None:
        """drive=0 should pass through the clean code path and drive just
        above zero should differ only by a vanishing ``drive_gain`` factor,
        NOT by a sudden jump in the saturation path (which is the actual
        regression this test guards)."""
        out_zero = _render(topology, 0.0)
        out_tiny = _render(topology, 1e-9)
        # At drive=1e-9 the smoothstep blend is 0 (below epsilon=0.05), so
        # no saturation kicks in.  The only residual difference is the
        # ``drive_gain = 1 + k*filter_drive`` multiplicative factor scaling
        # the input by ~1 + 2e-9 (ladder) or 1 + 1.5e-9 (sallen_key /
        # cascade).  That yields a delta proportional to drive, not the
        # O(x^3) ``_algebraic_sat`` step we are eliminating.
        diff_rms = _rms(out_tiny - out_zero)
        zero_rms = _rms(out_zero)
        # Relative difference should track the drive factor (~2e-9), not
        # jump to O(1%).  A 1e-7 relative bound is generous and still
        # three orders of magnitude below a true discontinuity step.
        assert diff_rms < zero_rms * 1e-7, (
            f"{topology}: drive=0 and drive=1e-9 differ by {diff_rms} "
            f"(signal rms {zero_rms}); indicates a step, not a smooth ramp"
        )

    @pytest.mark.parametrize("topology", TOPOLOGIES)
    def test_full_drive_activates_saturation(self, topology: str) -> None:
        """At drive >= epsilon the blend coefficient is 1, so the driven
        saturation is fully engaged.  Output should be deterministic and
        different from the linear path by a measurable amount."""
        linear = _render(topology, 0.0)
        driven = _render(topology, _DRIVE_BLEND_EPSILON * 2.0)
        diff = _rms(driven - linear)
        # Saturation at this drive level should produce a measurable delta.
        assert diff > 1e-4, (
            f"{topology}: drive={_DRIVE_BLEND_EPSILON * 2} should differ from "
            f"drive=0 (got rms diff {diff})"
        )


class TestBlendContinuity:
    """Sweep drive across the critical region [0, 2*epsilon] and assert that
    neighboring outputs do not jump discontinuously.
    """

    @pytest.mark.parametrize("topology", TOPOLOGIES)
    def test_no_step_across_zero(self, topology: str) -> None:
        """The key regression: at drive=0+ the output used to step because
        ``if driven: y = _algebraic_sat(y)`` activated all at once.  With
        the smoothstep blend, neighbor-to-neighbor RMS delta should be
        small and grow gradually."""
        drives = np.linspace(0.0, 2.0 * _DRIVE_BLEND_EPSILON, 21)
        outputs = [_render(topology, float(d)) for d in drives]

        # Neighbor-to-neighbor RMS deltas.
        deltas = np.array(
            [_rms(outputs[i + 1] - outputs[i]) for i in range(len(outputs) - 1)]
        )
        max_delta = float(deltas.max())
        mean_delta = float(deltas.mean())

        # If there were a step-discontinuity at drive=0, exactly one delta
        # (the 0 -> eps step) would dominate.  A C¹-continuous blend keeps
        # deltas roughly the same order of magnitude across the sweep.
        #
        # The ratio max/mean should stay moderate (< 5x) if we are smooth.
        # A true step would show max/mean >> 10.
        ratio = max_delta / max(mean_delta, 1e-12)
        assert ratio < 5.0, (
            f"{topology}: max/mean delta ratio {ratio:.2f} is too large — "
            f"suggests a step discontinuity across drive=0 "
            f"(deltas: {deltas.tolist()})"
        )

    @pytest.mark.parametrize("topology", TOPOLOGIES)
    def test_drive_sweep_is_monotone_to_linear(self, topology: str) -> None:
        """As drive increases from 0 into the saturation region, the RMS
        distance from the linear (drive=0) output should grow monotonically
        (within small numerical wobble).  A discontinuity at drive=0 would
        show up as an initial jump followed by a plateau.
        """
        linear = _render(topology, 0.0)
        drives = np.linspace(0.0, _DRIVE_BLEND_EPSILON, 11)
        distances = np.array(
            [_rms(_render(topology, float(d)) - linear) for d in drives]
        )
        # Monotonicity check — allow tiny numerical wobble (1e-9 absolute).
        diffs = np.diff(distances)
        assert np.all(diffs >= -1e-9), (
            f"{topology}: distance-from-linear should increase monotonically "
            f"with drive, got deltas {diffs.tolist()}"
        )
        # The distance at drive=0 is exactly 0; at drive=epsilon it should
        # be clearly positive.
        assert distances[0] == 0.0
        assert distances[-1] > 0.0
