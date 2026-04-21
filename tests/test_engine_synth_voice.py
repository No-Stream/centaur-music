"""Unified synth_voice engine tests.

Covers orchestrator + dispatch + each slot type + cross-pollination smoke
tests.  Slot-internal DSP correctness is exercised upstream (polyblep,
va, fm, _dsp_utils) — here we verify integration and the synth_voice
surface contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines.synth_voice import render

SAMPLE_RATE = 44_100


def test_empty_voice_renders_silence_without_crashing() -> None:
    audio = render(
        freq=220.0,
        duration=0.1,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={},
    )
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert len(audio) == int(SAMPLE_RATE * 0.1)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) == 0.0


def test_returns_empty_array_for_zero_duration_samples() -> None:
    audio = render(
        freq=220.0,
        duration=1e-6,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={},
    )
    assert len(audio) == 0


def test_registered_in_engine_registry() -> None:
    """synth_voice must be routable via the shared render_note_signal dispatch."""
    from code_musics.engines.registry import render_note_signal

    audio = render_note_signal(
        freq=220.0,
        duration=0.05,
        amp=0.5,
        sample_rate=SAMPLE_RATE,
        params={"engine": "synth_voice"},
    )
    assert isinstance(audio, np.ndarray)
    assert len(audio) == int(SAMPLE_RATE * 0.05)


def test_invalid_osc_type_raises() -> None:
    with pytest.raises(ValueError, match="osc_type"):
        render(
            freq=220.0,
            duration=0.05,
            amp=0.5,
            sample_rate=SAMPLE_RATE,
            params={"osc_type": "not_a_real_type"},
        )


def test_invalid_partials_type_raises() -> None:
    with pytest.raises(ValueError, match="partials_type"):
        render(
            freq=220.0,
            duration=0.05,
            amp=0.5,
            sample_rate=SAMPLE_RATE,
            params={"partials_type": "bogus"},
        )


def test_invalid_fm_type_raises() -> None:
    with pytest.raises(ValueError, match="fm_type"):
        render(
            freq=220.0,
            duration=0.05,
            amp=0.5,
            sample_rate=SAMPLE_RATE,
            params={"fm_type": "three_op"},
        )


def test_invalid_noise_type_raises() -> None:
    with pytest.raises(ValueError, match="noise_type"):
        render(
            freq=220.0,
            duration=0.05,
            amp=0.5,
            sample_rate=SAMPLE_RATE,
            params={"noise_type": "brown_fm"},
        )


def test_invalid_filter_topology_raises() -> None:
    with pytest.raises(ValueError, match="filter_topology"):
        render(
            freq=220.0,
            duration=0.05,
            amp=0.5,
            sample_rate=SAMPLE_RATE,
            params={
                "filter_mode": "lowpass",
                "filter_topology": "space_ladder",
            },
        )


def test_deterministic_output_for_identical_params() -> None:
    kwargs: dict = {
        "freq": 220.0,
        "duration": 0.1,
        "amp": 0.7,
        "sample_rate": SAMPLE_RATE,
        "params": {},
    }
    first = render(**kwargs)
    second = render(**kwargs)
    assert np.allclose(first, second)


def test_filter_surface_accepts_ladder_topology_on_silent_input() -> None:
    """Post-chain filter runs cleanly on zero input (exercises dispatch).

    The ladder topology seeds a 1e-6 bootstrap noise so high-Q
    self-oscillation can wake from silence (see CLAUDE.md); post-normalize
    that can be audible, so we only assert finiteness here.
    """
    audio = render(
        freq=220.0,
        duration=0.1,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={
            "filter_mode": "lowpass",
            "filter_topology": "ladder",
            "filter_cutoff_hz": 800.0,
            "resonance_q": 1.2,
        },
    )
    assert np.isfinite(audio).all()


def test_hpf_runs_cleanly_on_silent_input() -> None:
    audio = render(
        freq=220.0,
        duration=0.1,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={"hpf_cutoff_hz": 80.0},
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) == 0.0


# ---------------------------------------------------------------------------
# Per-slot smoke tests — each slot type should produce non-silent output.
# ---------------------------------------------------------------------------


def _render_voice(**params_overrides: object) -> np.ndarray:
    return render(
        freq=220.0,
        duration=0.2,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params=dict(params_overrides),
    )


@pytest.mark.parametrize("osc_type", ["polyblep", "supersaw", "pulse"])
def test_osc_slot_produces_signal(osc_type: str) -> None:
    audio = _render_voice(osc_type=osc_type)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.1


@pytest.mark.parametrize("partials_type", ["additive", "spectralwave", "drawbars"])
def test_partials_slot_produces_signal(partials_type: str) -> None:
    audio = _render_voice(partials_type=partials_type)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.1


def test_fm_slot_two_op_produces_signal() -> None:
    audio = _render_voice(
        fm_type="two_op",
        fm_ratio=2.0,
        fm_index=1.5,
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.1


@pytest.mark.parametrize("noise_type", ["white", "pink", "bandpass", "flow"])
def test_noise_slot_produces_signal(noise_type: str) -> None:
    audio = _render_voice(noise_type=noise_type, noise_level=1.0)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.01


def test_polyblep_osc2_detune_mixes_cleanly() -> None:
    audio = _render_voice(
        osc_type="polyblep",
        osc_wave="saw",
        osc2_level=0.5,
        osc2_detune_cents=7.0,
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.1


def test_polyblep_hard_sync_with_osc2_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="osc_hard_sync"):
        _render_voice(
            osc_type="polyblep",
            osc2_level=0.5,
            osc_hard_sync=True,
        )


# ---------------------------------------------------------------------------
# Cross-pollination: the design's hero use case.
# ---------------------------------------------------------------------------


def test_cross_pollination_all_four_slots_active() -> None:
    """FM bell over supersaw + additive pad + pink noise, through ladder."""
    audio = _render_voice(
        osc_type="supersaw",
        osc_level=0.5,
        osc_spread_cents=18.0,
        partials_type="additive",
        partials_level=0.4,
        partials_harmonic_rolloff=0.8,
        fm_type="two_op",
        fm_level=0.3,
        fm_ratio=2.0,
        fm_index=1.5,
        noise_type="pink",
        noise_level=0.1,
        filter_mode="lowpass",
        filter_topology="ladder",
        filter_cutoff_hz=1800.0,
        resonance_q=0.9,
        hpf_cutoff_hz=50.0,
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.2


def test_drawbars_through_diode_filter() -> None:
    """Hammond drawbars into a TB-303 diode ladder."""
    audio = _render_voice(
        partials_type="drawbars",
        filter_mode="lowpass",
        filter_topology="diode",
        filter_cutoff_hz=1200.0,
        resonance_q=2.0,
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.1


# ---------------------------------------------------------------------------
# Freq trajectory / pitch motion.
# ---------------------------------------------------------------------------


def test_freq_trajectory_accepted() -> None:
    """A per-sample freq trajectory should render without error."""
    n = int(SAMPLE_RATE * 0.1)
    trajectory = np.linspace(220.0, 440.0, n)
    audio = render(
        freq=220.0,
        duration=0.1,
        amp=0.7,
        sample_rate=SAMPLE_RATE,
        params={"osc_type": "polyblep"},
        freq_trajectory=trajectory,
    )
    assert np.isfinite(audio).all()
    assert len(audio) == n
    assert np.max(np.abs(audio)) > 0.1


def test_freq_trajectory_wrong_length_raises() -> None:
    with pytest.raises(ValueError, match="freq_trajectory length"):
        render(
            freq=220.0,
            duration=0.1,
            amp=0.7,
            sample_rate=SAMPLE_RATE,
            params={"osc_type": "polyblep"},
            freq_trajectory=np.linspace(220.0, 440.0, 999),
        )


# ---------------------------------------------------------------------------
# Voice-level shaper dispatch.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shaper", ["tanh", "saturation", "preamp", "foldback"])
def test_voice_shaper_dispatch(shaper: str) -> None:
    audio = _render_voice(
        osc_type="polyblep",
        shaper=shaper,
        shaper_drive=0.4,
        shaper_mix=1.0,
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.05


def test_unknown_shaper_raises() -> None:
    with pytest.raises(ValueError, match="shaper must be one of"):
        _render_voice(osc_type="polyblep", shaper="wobble_gate_9000")
