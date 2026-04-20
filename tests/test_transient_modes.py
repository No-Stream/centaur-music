"""Tests for transient / reset modes on polyblep and filtered_stack.

Transient modes control what oscillator state (phase + DC offset) carries
across notes within a voice.  See
:func:`code_musics.engines._dsp_utils.resolve_transient_mode` for the
authoritative config table.

The engines accept a per-voice ``voice_state`` dict that the score-level
render loop threads through each note.  When ``voice_state=None`` every
note starts fresh, matching the pre-carry era.
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines import filtered_stack as filtered_stack_engine
from code_musics.engines import polyblep as polyblep_engine
from code_musics.engines._dsp_utils import (
    TransientConfig,
    apply_transient_state,
    resolve_transient_mode,
    snapshot_voice_state,
)
from code_musics.engines.registry import render_note_signal

SR = 44100


def _base_params(engine: str = "polyblep") -> dict:
    return {
        "engine": engine,
        "waveform": "saw",
        "cutoff_hz": 4000.0,
        "resonance_q": 0.8,
        "pitch_drift": 0.0,
        "analog_jitter": 0.0,
        "noise_floor": 0.0,
        "cutoff_drift": 0.0,
        "voice_card_spread": 0.0,
        "osc_softness": 0.0,
        "osc_asymmetry": 0.0,
        "osc_shape_drift": 0.0,
    }


class TestResolveTransientMode:
    def test_analog_default(self) -> None:
        cfg = resolve_transient_mode("analog")
        assert isinstance(cfg, TransientConfig)
        assert cfg.reset_phase is False
        assert cfg.reset_dc is False

    def test_dc_reset(self) -> None:
        cfg = resolve_transient_mode("dc_reset")
        assert cfg.reset_phase is False
        assert cfg.reset_dc is True

    def test_osc_reset(self) -> None:
        cfg = resolve_transient_mode("osc_reset")
        assert cfg.reset_phase is True
        assert cfg.reset_dc is True

    def test_vcf_reset_removed(self) -> None:
        # vcf_reset was removed — it was a stub that never consulted
        # ``reset_filter``.  Filter-state carry-over is a separate future
        # project.
        with pytest.raises(ValueError, match="Unknown transient_mode"):
            resolve_transient_mode("vcf_reset")

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown transient_mode"):
            resolve_transient_mode("bogus")


class TestApplyTransientState:
    def test_none_voice_state_returns_fresh(self) -> None:
        cfg = resolve_transient_mode("analog")
        phase1, phase2, dc1, dc2 = apply_transient_state(
            None, transient_config=cfg, fresh_phase=0.42, fresh_dc_signs=(1.0, -1.0)
        )
        assert phase1 == 0.42
        assert phase2 == 0.42
        assert dc1 == 1.0
        assert dc2 == -1.0

    def test_analog_carries_prior_phase_and_dc(self) -> None:
        cfg = resolve_transient_mode("analog")
        state = {
            "phase_osc1": 1.5,
            "phase_osc2": 2.5,
            "dc_sign_osc1": -1.0,
            "dc_sign_osc2": 1.0,
        }
        phase1, phase2, dc1, dc2 = apply_transient_state(
            state, transient_config=cfg, fresh_phase=0.42, fresh_dc_signs=(1.0, -1.0)
        )
        assert phase1 == 1.5
        assert phase2 == 2.5
        assert dc1 == -1.0
        assert dc2 == 1.0

    def test_dc_reset_keeps_phase_but_drops_dc(self) -> None:
        cfg = resolve_transient_mode("dc_reset")
        state = {
            "phase_osc1": 1.5,
            "phase_osc2": 2.5,
            "dc_sign_osc1": -1.0,
            "dc_sign_osc2": 1.0,
        }
        phase1, phase2, dc1, dc2 = apply_transient_state(
            state, transient_config=cfg, fresh_phase=0.42, fresh_dc_signs=(1.0, -1.0)
        )
        assert phase1 == 1.5
        assert phase2 == 2.5
        assert dc1 == 1.0  # fresh
        assert dc2 == -1.0  # fresh

    def test_osc_reset_drops_both(self) -> None:
        cfg = resolve_transient_mode("osc_reset")
        state = {
            "phase_osc1": 1.5,
            "phase_osc2": 2.5,
            "dc_sign_osc1": -1.0,
            "dc_sign_osc2": 1.0,
        }
        phase1, phase2, dc1, dc2 = apply_transient_state(
            state, transient_config=cfg, fresh_phase=0.42, fresh_dc_signs=(1.0, -1.0)
        )
        assert phase1 == 0.42
        assert phase2 == 0.42
        assert dc1 == 1.0
        assert dc2 == -1.0

    def test_empty_state_dict_uses_fresh(self) -> None:
        """On the first note of a voice, state dict is empty — everything fresh."""
        cfg = resolve_transient_mode("analog")
        state: dict = {}
        phase1, phase2, dc1, dc2 = apply_transient_state(
            state, transient_config=cfg, fresh_phase=0.42, fresh_dc_signs=(1.0, -1.0)
        )
        assert phase1 == 0.42
        assert phase2 == 0.42
        assert dc1 == 1.0
        assert dc2 == -1.0


class TestSnapshotVoiceState:
    def test_persists_fields(self) -> None:
        state: dict = {}
        snapshot_voice_state(
            state,
            final_phase_osc1=2.71,
            final_phase_osc2=3.14,
            dc_sign_osc1=-1.0,
            dc_sign_osc2=1.0,
        )
        assert state["phase_osc1"] == 2.71
        assert state["phase_osc2"] == 3.14
        assert state["dc_sign_osc1"] == -1.0
        assert state["dc_sign_osc2"] == 1.0

    def test_noop_when_state_is_none(self) -> None:
        # Should not raise.
        snapshot_voice_state(
            None,
            final_phase_osc1=2.71,
            final_phase_osc2=3.14,
            dc_sign_osc1=1.0,
            dc_sign_osc2=1.0,
        )


class TestEngineVoiceStateIntegration:
    def test_polyblep_voice_state_none_still_works(self) -> None:
        audio = polyblep_engine.render(
            freq=220.0,
            duration=0.1,
            amp=0.8,
            sample_rate=SR,
            params=_base_params("polyblep"),
        )
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) > 0.01

    def test_filtered_stack_voice_state_none_still_works(self) -> None:
        audio = filtered_stack_engine.render(
            freq=220.0,
            duration=0.1,
            amp=0.8,
            sample_rate=SR,
            params=_base_params("filtered_stack"),
        )
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) > 0.01

    def test_polyblep_osc_reset_matches_voice_state_none(self) -> None:
        """osc_reset explicitly discards prior state, so back-to-back renders
        with osc_reset + a populated voice_state should match renders with
        voice_state=None."""
        params = {**_base_params("polyblep"), "transient_mode": "osc_reset"}
        state = {
            "phase_osc1": 1.23,
            "phase_osc2": 4.56,
            "dc_sign_osc1": -1.0,
            "dc_sign_osc2": -1.0,
        }
        reset_audio = polyblep_engine.render(
            freq=220.0,
            duration=0.15,
            amp=0.8,
            sample_rate=SR,
            params=params,
            voice_state=state,
        )
        fresh_audio = polyblep_engine.render(
            freq=220.0,
            duration=0.15,
            amp=0.8,
            sample_rate=SR,
            params={**_base_params("polyblep"), "transient_mode": "osc_reset"},
            voice_state=None,
        )
        np.testing.assert_array_equal(reset_audio, fresh_audio)

    def test_polyblep_analog_carries_phase_across_notes(self) -> None:
        """Two back-to-back notes under analog mode: the second note's start
        phase should match the first note's final phase (from voice_state).

        Uses a non-period-aligned freq/duration so the wrapped final phase
        doesn't coincidentally land on 0 — otherwise the carry-over is
        observationally indistinguishable from a fresh start.
        """
        params = {**_base_params("polyblep"), "transient_mode": "analog"}
        state: dict = {}
        polyblep_engine.render(
            freq=217.3,
            duration=0.1,
            amp=0.5,
            sample_rate=SR,
            params=params,
            voice_state=state,
        )
        assert state["phase_osc1"] != 0.0  # first render populated state

        # The second note's render reads state["phase_osc1"] as its
        # start_phase when analog mode is active.
        analog_audio = polyblep_engine.render(
            freq=217.3,
            duration=0.1,
            amp=0.5,
            sample_rate=SR,
            params=params,
            voice_state=state,
        )
        reset_audio = polyblep_engine.render(
            freq=217.3,
            duration=0.1,
            amp=0.5,
            sample_rate=SR,
            params={**_base_params("polyblep"), "transient_mode": "osc_reset"},
            voice_state=None,
        )
        assert not np.allclose(analog_audio, reset_audio, atol=1e-9)

    def test_polyblep_dc_reset_keeps_phase_drops_dc(self) -> None:
        """Under dc_reset the phase is carried but DC sign is fresh each note.
        Use osc_dc_offset so the DC sign is observable in the output."""
        params = {
            **_base_params("polyblep"),
            "transient_mode": "dc_reset",
            "osc_dc_offset": 0.8,
        }
        state: dict = {}
        audio_one = polyblep_engine.render(
            freq=220.0,
            duration=0.1,
            amp=0.5,
            sample_rate=SR,
            params=params,
            voice_state=state,
        )
        phase_after_one = state["phase_osc1"]
        dc_after_one = state["dc_sign_osc1"]
        audio_two = polyblep_engine.render(
            freq=220.0,
            duration=0.1,
            amp=0.5,
            sample_rate=SR,
            params=params,
            voice_state=state,
        )
        # Phase should carry (dc_reset keeps phase).
        assert state["phase_osc1"] != phase_after_one
        # DC sign is redrawn fresh each note; with identical RNG seeds the
        # sign ends up the same both times.
        assert state["dc_sign_osc1"] == dc_after_one
        assert np.all(np.isfinite(audio_one))
        assert np.all(np.isfinite(audio_two))

    def test_polyblep_voice_state_carries_osc2_phase(self) -> None:
        """F2: osc2's phase must be snapshotted so dual-osc and hard-sync
        patches don't glitch across note boundaries."""
        params = {
            **_base_params("polyblep"),
            "transient_mode": "analog",
            "osc2_level": 0.7,
            "osc2_detune_cents": 700.0,
        }
        state: dict = {}
        polyblep_engine.render(
            freq=217.3,
            duration=0.1,
            amp=0.5,
            sample_rate=SR,
            params=params,
            voice_state=state,
        )
        assert "phase_osc1" in state
        assert "phase_osc2" in state
        # osc2 runs at a detuned ratio so its final phase differs from osc1's.
        assert state["phase_osc1"] != state["phase_osc2"]

    def test_filtered_stack_analog_mode_carries_phase(self) -> None:
        """Filtered-stack is additive (no DC), so analog mode should carry
        the fundamental phase through voice_state."""
        params = {**_base_params("filtered_stack"), "transient_mode": "analog"}
        state: dict = {}
        filtered_stack_engine.render(
            freq=220.0,
            duration=0.1,
            amp=0.5,
            sample_rate=SR,
            params=params,
            voice_state=state,
        )
        assert "phase_osc1" in state
        # After one note, phase has advanced from 0
        assert state["phase_osc1"] != 0.0

    def test_polyblep_empty_voice_state_matches_none(self) -> None:
        """F24: voice_state={} under analog should produce identical output
        to voice_state=None on the first note (no prior state to carry)."""
        params = {**_base_params("polyblep"), "transient_mode": "analog"}
        empty_audio = polyblep_engine.render(
            freq=220.0,
            duration=0.1,
            amp=0.5,
            sample_rate=SR,
            params=params,
            voice_state={},
        )
        none_audio = polyblep_engine.render(
            freq=220.0,
            duration=0.1,
            amp=0.5,
            sample_rate=SR,
            params=params,
            voice_state=None,
        )
        np.testing.assert_array_equal(empty_audio, none_audio)

    def test_filtered_stack_empty_voice_state_matches_none(self) -> None:
        """F24: same as above for filtered_stack."""
        params = {**_base_params("filtered_stack"), "transient_mode": "analog"}
        empty_audio = filtered_stack_engine.render(
            freq=220.0,
            duration=0.1,
            amp=0.5,
            sample_rate=SR,
            params=params,
            voice_state={},
        )
        none_audio = filtered_stack_engine.render(
            freq=220.0,
            duration=0.1,
            amp=0.5,
            sample_rate=SR,
            params=params,
            voice_state=None,
        )
        np.testing.assert_array_equal(empty_audio, none_audio)


class TestRegistryThreading:
    def test_render_note_signal_accepts_voice_state(self) -> None:
        """The registry's render_note_signal dispatcher must accept
        voice_state and forward it to polyblep / filtered_stack."""
        state: dict = {}
        audio = render_note_signal(
            freq=220.0,
            duration=0.1,
            amp=0.5,
            sample_rate=SR,
            params=_base_params("polyblep"),
            voice_state=state,
        )
        assert np.all(np.isfinite(audio))
        # polyblep should have snapshotted phase into the state dict
        assert "phase_osc1" in state
        assert "phase_osc2" in state

    def test_render_note_signal_ignores_voice_state_for_non_subtractive(self) -> None:
        """Passing voice_state to an engine that doesn't accept it (e.g.
        additive) must not crash — the dispatcher silently drops it."""
        audio = render_note_signal(
            freq=220.0,
            duration=0.1,
            amp=0.5,
            sample_rate=SR,
            params={"engine": "additive"},
            voice_state={"some": "state"},
        )
        assert np.all(np.isfinite(audio))

    def test_render_note_signal_without_voice_state(self) -> None:
        """voice_state is optional."""
        audio = render_note_signal(
            freq=220.0,
            duration=0.1,
            amp=0.5,
            sample_rate=SR,
            params=_base_params("polyblep"),
        )
        assert np.all(np.isfinite(audio))
