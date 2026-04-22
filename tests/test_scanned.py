"""Scanned synthesis tests.

Covers the raw ``render_scanned`` primitive and its synth_voice integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._scanned import render_scanned
from code_musics.engines.synth_voice import render as render_voice

SR = 44_100


def _freq_profile(n: int, freq: float = 220.0) -> np.ndarray:
    return np.full(n, freq, dtype=np.float64)


# ---------------------------------------------------------------------------
# Primitive
# ---------------------------------------------------------------------------


def test_render_scanned_produces_finite_correct_length_output() -> None:
    n = int(SR * 0.5)
    out = render_scanned(
        freq=220.0,
        n_samples=n,
        sample_rate=SR,
        n_nodes=16,
        motion=0.3,
        tension=0.5,
        damping=0.2,
        position=0.4,
        seed=1,
    )
    assert out.shape == (n,)
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) > 0


def test_render_scanned_freq_profile_matches_constant_freq() -> None:
    """Passing freq_profile of constant value must match the scalar-freq path."""
    n = int(SR * 0.2)
    scalar = render_scanned(
        freq=220.0,
        n_samples=n,
        sample_rate=SR,
        n_nodes=16,
        motion=0.3,
        tension=0.5,
        damping=0.2,
        position=0.4,
        seed=1,
    )
    profile = render_scanned(
        freq=220.0,
        n_samples=n,
        sample_rate=SR,
        n_nodes=16,
        motion=0.3,
        tension=0.5,
        damping=0.2,
        position=0.4,
        seed=1,
        freq_profile=_freq_profile(n, freq=220.0),
    )
    assert np.array_equal(scalar, profile)


def _call_render_scanned(seed: int) -> np.ndarray:
    return render_scanned(
        freq=220.0,
        n_samples=4410,
        sample_rate=SR,
        n_nodes=16,
        motion=0.5,
        tension=0.5,
        damping=0.2,
        position=0.5,
        seed=seed,
    )


def test_render_scanned_deterministic_for_same_seed() -> None:
    a = _call_render_scanned(seed=42)
    b = _call_render_scanned(seed=42)
    assert np.array_equal(a, b)


def test_render_scanned_different_seeds_differ() -> None:
    a = _call_render_scanned(seed=1)
    b = _call_render_scanned(seed=2)
    assert not np.array_equal(a, b)


def test_render_scanned_stable_over_long_duration() -> None:
    n = int(SR * 8.0)
    out = render_scanned(
        freq=110.0,
        n_samples=n,
        sample_rate=SR,
        n_nodes=24,
        motion=0.9,
        tension=0.8,
        damping=0.05,
        position=0.3,
        seed=7,
    )
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) < 2.0


def test_render_scanned_rejects_too_few_nodes() -> None:
    with pytest.raises(ValueError, match="n_nodes"):
        render_scanned(
            freq=220.0,
            n_samples=1024,
            sample_rate=SR,
            n_nodes=2,
            motion=0.3,
            tension=0.5,
            damping=0.2,
            position=0.4,
            seed=0,
        )


def test_render_scanned_rejects_too_many_nodes() -> None:
    with pytest.raises(ValueError, match="n_nodes"):
        render_scanned(
            freq=220.0,
            n_samples=1024,
            sample_rate=SR,
            n_nodes=200,
            motion=0.3,
            tension=0.5,
            damping=0.2,
            position=0.4,
            seed=0,
        )


def test_render_scanned_rejects_motion_out_of_range() -> None:
    with pytest.raises(ValueError, match="motion"):
        render_scanned(
            freq=220.0,
            n_samples=1024,
            sample_rate=SR,
            n_nodes=16,
            motion=1.5,
            tension=0.5,
            damping=0.2,
            position=0.4,
            seed=0,
        )


def test_render_scanned_rejects_tension_out_of_range() -> None:
    with pytest.raises(ValueError, match="tension"):
        render_scanned(
            freq=220.0,
            n_samples=1024,
            sample_rate=SR,
            n_nodes=16,
            motion=0.3,
            tension=-0.1,
            damping=0.2,
            position=0.4,
            seed=0,
        )


def test_render_scanned_rejects_damping_out_of_range() -> None:
    with pytest.raises(ValueError, match="damping"):
        render_scanned(
            freq=220.0,
            n_samples=1024,
            sample_rate=SR,
            n_nodes=16,
            motion=0.3,
            tension=0.5,
            damping=1.5,
            position=0.4,
            seed=0,
        )


def test_render_scanned_rejects_position_out_of_range() -> None:
    with pytest.raises(ValueError, match="position"):
        render_scanned(
            freq=220.0,
            n_samples=1024,
            sample_rate=SR,
            n_nodes=16,
            motion=0.3,
            tension=0.5,
            damping=0.2,
            position=-0.1,
            seed=0,
        )


def test_render_scanned_rejects_nonpositive_freq_profile() -> None:
    with pytest.raises(ValueError, match="freq_profile"):
        render_scanned(
            freq=220.0,
            n_samples=1024,
            sample_rate=SR,
            n_nodes=16,
            motion=0.3,
            tension=0.5,
            damping=0.2,
            position=0.4,
            seed=0,
            freq_profile=np.zeros(1024, dtype=np.float64),
        )


def test_damping_shortens_rms_tail() -> None:
    """Higher damping should produce a smaller late-buffer RMS than low damping."""
    n = int(SR * 2.0)
    low = render_scanned(
        freq=220.0,
        n_samples=n,
        sample_rate=SR,
        n_nodes=16,
        motion=0.2,
        tension=0.5,
        damping=0.0,
        position=0.5,
        seed=0,
    )
    high = render_scanned(
        freq=220.0,
        n_samples=n,
        sample_rate=SR,
        n_nodes=16,
        motion=0.2,
        tension=0.5,
        damping=0.9,
        position=0.5,
        seed=0,
    )
    late_low = float(np.sqrt(np.mean(low[-SR:] ** 2)))
    late_high = float(np.sqrt(np.mean(high[-SR:] ** 2)))
    # Both peak-normalize to 1 overall, but the late-window RMS should show
    # the damping effect: heavy damping's tail is much smaller relative to
    # its peak than no-damping's tail.
    assert late_high < late_low


# ---------------------------------------------------------------------------
# synth_voice integration
# ---------------------------------------------------------------------------


def test_synth_voice_scanned_osc_renders() -> None:
    audio = render_voice(
        freq=220.0,
        duration=0.3,
        amp=0.6,
        sample_rate=SR,
        params={
            "osc_type": "scanned",
            "osc_scan_n_nodes": 16,
            "osc_scan_motion": 0.5,
            "osc_scan_tension": 0.6,
            "osc_scan_damping": 0.15,
            "osc_scan_position": 0.5,
            "filter_mode": "lowpass",
            "filter_cutoff_hz": 3000.0,
        },
    )
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) > 0
    assert len(audio) == int(SR * 0.3)


def test_synth_voice_invalid_scan_param_raises() -> None:
    with pytest.raises(ValueError, match="tension"):
        render_voice(
            freq=220.0,
            duration=0.1,
            amp=0.5,
            sample_rate=SR,
            params={
                "osc_type": "scanned",
                "osc_scan_tension": 1.5,
            },
        )


def test_synth_voice_scanned_preset_loads() -> None:
    """scanned presets should be registered and render finite audio."""
    from code_musics.engines.registry import render_note_signal

    for preset in (
        "scanned_breathing_string",
        "scanned_singing_loop",
        "scanned_glass_swarm",
        "scanned_taut_wire",
        "scanned_loose_chain",
    ):
        audio = render_note_signal(
            freq=220.0,
            duration=0.25,
            amp=0.6,
            sample_rate=SR,
            params={"engine": "synth_voice", "preset": preset},
        )
        assert isinstance(audio, np.ndarray)
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) > 0
