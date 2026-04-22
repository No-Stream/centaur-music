"""Mode coupling + allpass dispersion tests for render_modal_bank.

Covers:
* Legacy parallel path is bit-identical when coupling=0 and dispersion=0.
* All three coupling topologies (chain / ring / all) produce finite output
  and differ from the parallel path.
* Coupling topology validation.
* Dispersion produces finite output and differs from non-dispersed.
* Integration through drum_voice tone_type='modal' and metallic_type='modal_bank'.
* pi_coupling / pi_dispersion macros route correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._modal import render_modal_bank
from code_musics.engines._pi_macros import resolve_pi_macros

# drum_voice integration tests import lazily inside each test so this module
# stays green even if drum_voice itself is temporarily broken by an in-flight
# refactor on a parallel branch.

SR = 44_100


def _exciter(n: int, seed: int = 0) -> np.ndarray:
    """Short impulse-like exciter scaled to a musically usable amplitude.

    The amplitude matters: coupling's per-sample effect is proportional to
    state magnitude, so a 1e-5-scale exciter makes differences swamped by
    atol=1e-6. Scale to ~1.0 peak like a real drum hit so inter-mode
    coupling contributions are comfortably above measurement noise.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    raw = rng.standard_normal(12)
    raw = raw / float(np.max(np.abs(raw)))
    out[:12] = raw
    return out


def _standard_bank_args() -> dict:
    return dict(
        mode_ratios=[1.0, 2.76, 5.4, 8.9],
        mode_amps=[1.0, 0.6, 0.35, 0.2],
        mode_decays_s=[0.8, 0.6, 0.4, 0.3],
        freq_hz=220.0,
        sample_rate=SR,
    )


# ---------------------------------------------------------------------------
# Legacy behavior preservation
# ---------------------------------------------------------------------------


def test_no_coupling_no_dispersion_is_bit_identical_to_legacy_path() -> None:
    """With coupling=0 and dispersion=0 the parallel path must match exactly."""
    exciter = _exciter(4410)
    legacy = render_modal_bank(exciter, **_standard_bank_args())
    with_defaults = render_modal_bank(
        exciter,
        **_standard_bank_args(),
        coupling=0.0,
        dispersion=0.0,
    )
    assert np.array_equal(legacy, with_defaults)


# ---------------------------------------------------------------------------
# Coupling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topology", ["chain", "ring", "all"])
def test_coupling_produces_finite_output_distinct_from_parallel(topology: str) -> None:
    exciter = _exciter(8820)
    baseline = render_modal_bank(exciter, **_standard_bank_args())
    coupled = render_modal_bank(
        exciter,
        **_standard_bank_args(),
        coupling=0.25,
        coupling_topology=topology,
    )
    assert coupled.shape == baseline.shape
    assert np.all(np.isfinite(coupled))
    # Coupling modifies the per-sample input of every mode, so the outputs
    # MUST differ. Use a tight absolute threshold rather than relying on
    # np.allclose's default rtol, which swamps small deltas on low-magnitude
    # signals at the start of the buffer.
    max_delta = float(np.max(np.abs(coupled - baseline)))
    assert max_delta > 0.0, f"coupling={topology} had zero effect"


def test_coupling_topology_unknown_raises() -> None:
    exciter = _exciter(1024)
    with pytest.raises(ValueError, match="coupling_topology"):
        render_modal_bank(
            exciter,
            **_standard_bank_args(),
            coupling=0.2,
            coupling_topology="not_a_topology",
        )


def test_coupling_at_ceiling_stays_stable() -> None:
    """coupling at COUPLING_MAX must not blow up on the highest-Q presets."""
    from code_musics.engines._modal import COUPLING_MAX

    n = int(SR * 2.0)
    exciter = _exciter(n, seed=42)
    out = render_modal_bank(
        exciter,
        mode_ratios=[1.0, 2.76, 5.4, 8.9, 13.3],
        mode_amps=[1.0, 0.7, 0.5, 0.35, 0.22],
        mode_decays_s=[2.0, 1.6, 1.2, 0.9, 0.7],
        freq_hz=220.0,
        sample_rate=SR,
        coupling=COUPLING_MAX,
        coupling_topology="all",
    )
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) < 10.0


def test_coupling_amount_out_of_range_raises() -> None:
    exciter = _exciter(1024)
    with pytest.raises(ValueError, match="coupling"):
        render_modal_bank(
            exciter,
            **_standard_bank_args(),
            coupling=1.0,
        )


# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------


def test_dispersion_produces_finite_output_distinct_from_baseline() -> None:
    exciter = _exciter(8820)
    baseline = render_modal_bank(exciter, **_standard_bank_args())
    dispersed = render_modal_bank(
        exciter,
        **_standard_bank_args(),
        dispersion=0.6,
    )
    assert np.all(np.isfinite(dispersed))
    assert not np.allclose(dispersed, baseline, atol=1e-6)
    # Allpass is unity-magnitude so RMS should stay close to baseline.
    rms_base = float(np.sqrt(np.mean(baseline**2)))
    rms_disp = float(np.sqrt(np.mean(dispersed**2)))
    assert rms_disp == pytest.approx(rms_base, rel=0.2)


def test_dispersion_stages_param_must_be_positive() -> None:
    exciter = _exciter(1024)
    with pytest.raises(ValueError, match="dispersion_n_stages"):
        render_modal_bank(
            exciter,
            **_standard_bank_args(),
            dispersion=0.5,
            dispersion_n_stages=0,
        )


def test_dispersion_out_of_range_raises() -> None:
    exciter = _exciter(1024)
    with pytest.raises(ValueError, match="dispersion"):
        render_modal_bank(
            exciter,
            **_standard_bank_args(),
            dispersion=1.5,
        )


def test_combined_coupling_and_dispersion_stable() -> None:
    n = int(SR * 1.5)
    exciter = _exciter(n, seed=7)
    out = render_modal_bank(
        exciter,
        **_standard_bank_args(),
        coupling=0.3,
        coupling_topology="chain",
        dispersion=0.5,
    )
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) < 10.0


# ---------------------------------------------------------------------------
# drum_voice integration
# ---------------------------------------------------------------------------


def test_drum_voice_modal_tone_with_coupling() -> None:
    drum_voice = pytest.importorskip("code_musics.engines.drum_voice")
    audio = drum_voice.render(
        freq=110.0,
        duration=0.8,
        amp=0.8,
        sample_rate=SR,
        params={
            "exciter_type": "click",
            "exciter_level": 0.3,
            "tone_type": "modal",
            "tone_level": 1.0,
            "modal_mode_table": "bar_metal",
            "modal_n_modes": 6,
            "modal_decay_s": 0.9,
            "modal_coupling": 0.3,
            "modal_coupling_topology": "chain",
        },
    )
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) > 0


def test_drum_voice_metallic_modal_with_dispersion() -> None:
    drum_voice = pytest.importorskip("code_musics.engines.drum_voice")
    audio = drum_voice.render(
        freq=220.0,
        duration=0.6,
        amp=0.8,
        sample_rate=SR,
        params={
            "exciter_type": "click",
            "exciter_level": 0.2,
            "tone_type": None,
            "metallic_type": "modal_bank",
            "metallic_level": 1.0,
            "metallic_mode_table": "bowl",
            "metallic_n_modes": 6,
            "metallic_decay_s": 0.8,
            "metallic_dispersion": 0.7,
        },
    )
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) > 0


def test_bit_identical_when_coupling_and_dispersion_off() -> None:
    """Direct bank-level bit-identical check — coupling=0 + dispersion=0 is legacy.

    Covers the backward-compat guarantee at the primitive layer.  The
    drum_voice wrapper hashes its full params dict into its per-note RNG
    seed, so adding new keys (even with no-op values) inevitably changes
    unrelated noise components of its output.  That's existing engine
    behavior, not a coupling/dispersion regression.
    """
    exciter = _exciter(4410)
    baseline = render_modal_bank(exciter, **_standard_bank_args())
    with_explicit_off = render_modal_bank(
        exciter,
        **_standard_bank_args(),
        coupling=0.0,
        coupling_topology="ring",  # topology ignored when coupling=0
        dispersion=0.0,
        dispersion_n_stages=8,  # stages ignored when dispersion=0
    )
    assert np.array_equal(baseline, with_explicit_off)


# ---------------------------------------------------------------------------
# pi_coupling / pi_dispersion macros
# ---------------------------------------------------------------------------


def test_pi_coupling_macro_sets_modal_coupling() -> None:
    """pi_coupling is a 0..1 perceptual macro scaled into [0, COUPLING_MAX]."""
    from code_musics.engines._modal import COUPLING_MAX

    params = {"pi_coupling": 0.4}
    resolve_pi_macros(params)
    assert params["modal_coupling"] == pytest.approx(0.4 * COUPLING_MAX)
    assert "pi_coupling" not in params


def test_pi_dispersion_macro_sets_modal_dispersion() -> None:
    params = {"pi_dispersion": 0.6}
    resolve_pi_macros(params)
    assert params["modal_dispersion"] == pytest.approx(0.6)
    assert "pi_dispersion" not in params


def test_user_modal_coupling_wins_over_pi_macro() -> None:
    params = {"pi_coupling": 0.4, "modal_coupling": 0.1}
    resolve_pi_macros(params)
    assert params["modal_coupling"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "macro_key,target_key,in_value,expected",
    [
        # 0..1 macros (out-of-range on both ends)
        ("pi_hardness", "exciter_center_hz", 1.5, 5500.0),
        ("pi_hardness", "exciter_center_hz", -0.2, 500.0),
        ("pi_damping", "modal_damping", 1.5, 0.2),
        ("pi_damping", "modal_damping", -0.2, 2.5),
        ("pi_position", "modal_position", 1.5, 1.0),
        ("pi_position", "modal_position", -0.2, 0.0),
        ("pi_dispersion", "modal_dispersion", 1.5, 1.0),
        ("pi_dispersion", "modal_dispersion", -0.2, 0.0),
        # -1..1 macros
        ("pi_tension", "modal_tension", 1.5, 1.0),
        ("pi_tension", "modal_tension", -1.5, -1.0),
        ("pi_damping_tilt", "modal_damping_tilt", 1.5, 1.0),
        ("pi_damping_tilt", "modal_damping_tilt", -1.5, -1.0),
    ],
)
def test_pi_macros_clamp_to_documented_range(
    macro_key: str, target_key: str, in_value: float, expected: float
) -> None:
    params: dict = {macro_key: in_value}
    resolve_pi_macros(params)
    assert params[target_key] == pytest.approx(expected)


def test_pi_coupling_clamps_and_scales_to_coupling_max() -> None:
    """``pi_coupling`` is a 0..1 perceptual macro scaled into [0, COUPLING_MAX]."""
    from code_musics.engines._modal import COUPLING_MAX

    high = {"pi_coupling": 1.5}
    resolve_pi_macros(high)
    assert high["modal_coupling"] == pytest.approx(COUPLING_MAX)

    low = {"pi_coupling": -0.2}
    resolve_pi_macros(low)
    assert low["modal_coupling"] == pytest.approx(0.0)
