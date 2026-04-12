"""Score abstraction tests."""

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from code_musics import synth
from code_musics.automation import AutomationSegment, AutomationSpec, AutomationTarget
from code_musics.composition import line
from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    TimingTarget,
    VelocityHumanizeSpec,
    VelocityTarget,
    build_timing_offsets,
    build_velocity_multipliers,
    resolve_envelope_params,
)
from code_musics.pieces import PIECES
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.render import RenderWindow, render_piece
from code_musics.score import (
    EffectSpec,
    NoteEvent,
    Phrase,
    Score,
    SendBusSpec,
    VoiceSend,
)


def test_total_duration_is_derived_from_note_endpoints() -> None:
    score = Score(f0=55.0)
    score.add_note("a", start=1.0, duration=2.5, partial=4)
    score.add_note("b", start=0.5, duration=5.0, partial=6)

    assert score.total_dur == 5.5


def test_phrase_and_direct_note_have_matching_timing() -> None:
    score = Score(f0=55.0)
    phrase = Phrase(events=(NoteEvent(start=0.0, duration=1.2, partial=5, amp=0.4),))

    placed = score.add_phrase("lead", phrase, start=3.0)
    direct = score.add_note("lead", start=3.0, duration=1.2, partial=5, amp=0.4)

    assert placed[0].start == direct.start
    assert placed[0].duration == direct.duration
    assert placed[0].partial == direct.partial
    assert placed[0].amp == direct.amp


def test_note_event_and_add_note_support_amp_db() -> None:
    score = Score(f0=55.0)
    note = NoteEvent(start=0.0, duration=1.0, partial=4.0, amp_db=-12.0)
    placed = score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp_db=-12.0)

    assert note.amp == pytest.approx(10 ** (-12.0 / 20.0))
    assert note.amp_db == pytest.approx(-12.0)
    assert placed.amp == pytest.approx(note.amp)


def test_note_event_supports_velocity() -> None:
    note = NoteEvent(start=0.0, duration=1.0, partial=4.0, velocity=0.85)

    assert note.velocity == pytest.approx(0.85)


def test_note_event_rejects_invalid_velocity() -> None:
    with pytest.raises(ValueError, match="velocity"):
        NoteEvent(start=0.0, duration=1.0, partial=4.0, velocity=0.0)


def test_note_event_rejects_amp_and_amp_db_together() -> None:
    with pytest.raises(ValueError, match="amp or amp_db"):
        NoteEvent(start=0.0, duration=1.0, partial=4.0, amp=0.5, amp_db=-6.0)


def test_phrase_transforms_do_not_mutate_original() -> None:
    phrase = Phrase.from_partials([4, 5, 6], note_dur=1.0, step=0.8, amp=0.5)
    original_partials = [event.partial for event in phrase.events]

    transformed = phrase.transformed(
        start=10.0, partial_shift=2.0, amp_scale=0.5, reverse=True
    )

    assert [event.partial for event in phrase.events] == original_partials
    assert [event.partial for event in transformed] == [6.0, 7.0, 8.0]
    assert transformed[0].start > 10.0


def test_phrase_from_partials_supports_amp_db() -> None:
    phrase = Phrase.from_partials([4, 5], note_dur=1.0, step=0.5, amp_db=-18.0)

    assert [event.amp_db for event in phrase.events] == [-18.0, -18.0]
    assert [event.amp for event in phrase.events] == pytest.approx(
        [10 ** (-18.0 / 20.0)] * 2
    )


def test_phrase_from_partials_supports_velocity() -> None:
    phrase = Phrase.from_partials([4, 5], note_dur=1.0, step=0.5, velocity=0.92)

    assert [event.velocity for event in phrase.events] == pytest.approx([0.92, 0.92])


def test_phrase_transform_preserves_pitch_motion_through_reverse_and_scale() -> None:
    phrase = line(
        tones=[4.0, 5.0],
        rhythm=(0.5, 1.0),
        pitch_motion=(
            PitchMotionSpec.linear_bend(target_partial=5.0),
            PitchMotionSpec.ratio_glide(start_ratio=1.0, end_ratio=6 / 5),
        ),
    )

    transformed = phrase.transformed(start=2.0, time_scale=2.0, reverse=True)

    assert phrase.events[0].pitch_motion is not None
    assert transformed[0].pitch_motion == phrase.events[0].pitch_motion
    assert transformed[1].pitch_motion == phrase.events[1].pitch_motion
    assert transformed[0].duration == pytest.approx(1.0)
    assert transformed[1].duration == pytest.approx(2.0)


def test_phrase_transform_preserves_velocity() -> None:
    phrase = Phrase(
        events=(NoteEvent(start=0.0, duration=1.0, partial=4.0, velocity=0.8),)
    )

    transformed = phrase.transformed(start=2.0, time_scale=1.5, reverse=True)

    assert transformed[0].velocity == pytest.approx(0.8)


def test_note_event_supports_automation() -> None:
    automation = [
        AutomationSpec(
            target=AutomationTarget(kind="synth", name="cutoff_hz"),
            segments=(
                AutomationSegment(start=0.0, end=1.0, shape="hold", value=800.0),
            ),
        )
    ]

    note = NoteEvent(
        start=0.0,
        duration=1.0,
        partial=4.0,
        automation=automation,
    )

    assert note.automation == automation


def test_automation_segment_rejects_overlapping_ranges() -> None:
    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        AutomationSpec(
            target=AutomationTarget(kind="synth", name="cutoff_hz"),
            segments=(
                AutomationSegment(
                    start=0.0,
                    end=1.0,
                    shape="linear",
                    start_value=300.0,
                    end_value=800.0,
                ),
                AutomationSegment(
                    start=0.5,
                    end=1.5,
                    shape="hold",
                    value=900.0,
                ),
            ),
        )


def test_automation_modes_apply_expected_values() -> None:
    replace = AutomationSpec(
        target=AutomationTarget(kind="synth", name="cutoff_hz"),
        segments=(AutomationSegment(start=0.0, end=1.0, shape="hold", value=900.0),),
        mode="replace",
    )
    add = AutomationSpec(
        target=AutomationTarget(kind="synth", name="cutoff_hz"),
        segments=(AutomationSegment(start=0.0, end=1.0, shape="hold", value=150.0),),
        mode="add",
    )
    multiply = AutomationSpec(
        target=AutomationTarget(kind="synth", name="cutoff_hz"),
        segments=(AutomationSegment(start=0.0, end=1.0, shape="hold", value=1.5),),
        mode="multiply",
    )

    assert replace.apply_to_base(base_value=400.0, time=0.5) == pytest.approx(900.0)
    assert add.apply_to_base(base_value=400.0, time=0.5) == pytest.approx(550.0)
    assert multiply.apply_to_base(base_value=400.0, time=0.5) == pytest.approx(600.0)


def test_control_automation_target_validation() -> None:
    target = AutomationTarget(kind="control", name="send_db")

    assert target.name == "send_db"

    with pytest.raises(ValueError, match="Unsupported control automation target"):
        AutomationTarget(kind="control", name="feedback")


def test_synth_automation_params_accepted() -> None:
    """All supported synth params should be accepted by AutomationTarget."""
    expected_params = [
        "attack",
        "brightness",
        "brightness_tilt",
        "click_amount",
        "cutoff_hz",
        "decay",
        "drive_ratio",
        "feedback",
        "filter_drive",
        "filter_env_amount",
        "filter_env_decay",
        "hammer_hardness",
        "hammer_noise",
        "index_decay",
        "mod_index",
        "noise_amount",
        "osc2_detune_cents",
        "osc2_level",
        "overtone_amount",
        "release",
        "resonance_q",
        "soundboard_color",
        "sustain_level",
    ]
    for param_name in expected_params:
        target = AutomationTarget(kind="synth", name=param_name)
        assert target.kind == "synth"
        assert target.name == param_name


def test_synth_automation_rejects_unsupported_param() -> None:
    with pytest.raises(ValueError, match="Unsupported synth automation target"):
        AutomationTarget(kind="synth", name="waveform")


def test_render_overlapping_voices_returns_audio() -> None:
    score = Score(f0=55.0)
    score.add_note("a", start=0.0, duration=1.0, partial=4, amp=0.3)
    score.add_note("b", start=0.5, duration=1.0, partial=5, amp=0.3)

    audio = score.render()

    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert len(audio) == int(1.8 * score.sample_rate)
    assert np.max(np.abs(audio)) > 0


def test_voice_max_polyphony_one_truncates_previous_note() -> None:
    strict_mono = Score(f0=55.0, auto_master_gain_stage=False)
    strict_mono.add_voice(
        "bass",
        synth_defaults={"engine": "polyblep", "waveform": "saw", "release": 0.12},
        normalize_lufs=None,
        max_polyphony=1,
    )
    strict_mono.add_note("bass", start=0.0, duration=0.6, freq=55.0, amp=0.2)
    strict_mono.add_note("bass", start=0.3, duration=0.4, freq=82.5, amp=0.2)

    manually_truncated = Score(f0=55.0, auto_master_gain_stage=False)
    manually_truncated.add_voice(
        "bass",
        synth_defaults={"engine": "polyblep", "waveform": "saw", "release": 0.12},
        normalize_lufs=None,
    )
    manually_truncated.add_note(
        "bass",
        start=0.0,
        duration=0.3,
        freq=55.0,
        amp=0.2,
        synth={"release": 0.005},
    )
    manually_truncated.add_note("bass", start=0.3, duration=0.4, freq=82.5, amp=0.2)

    assert np.allclose(strict_mono.render(), manually_truncated.render(), atol=1e-8)


def test_voice_legato_skips_attack_retrigger_when_polyphony_is_one() -> None:
    retriggered = Score(f0=55.0, auto_master_gain_stage=False)
    retriggered.add_voice(
        "bass",
        synth_defaults={"engine": "polyblep", "waveform": "saw", "attack": 0.05},
        normalize_lufs=None,
        max_polyphony=1,
        legato=False,
    )
    retriggered.add_note("bass", start=0.0, duration=0.6, freq=55.0, amp=0.2)
    retriggered.add_note("bass", start=0.3, duration=0.4, freq=82.5, amp=0.2)

    legato = Score(f0=55.0, auto_master_gain_stage=False)
    legato.add_voice(
        "bass",
        synth_defaults={"engine": "polyblep", "waveform": "saw", "attack": 0.05},
        normalize_lufs=None,
        max_polyphony=1,
        legato=True,
    )
    legato.add_note("bass", start=0.0, duration=0.6, freq=55.0, amp=0.2)
    legato.add_note("bass", start=0.3, duration=0.4, freq=82.5, amp=0.2)

    retriggered_audio = retriggered.render()
    legato_audio = legato.render()
    onset_sample = int(0.3 * retriggered.sample_rate)
    window_end = onset_sample + int(0.02 * retriggered.sample_rate)
    retriggered_window = retriggered_audio[onset_sample:window_end]
    legato_window = legato_audio[onset_sample:window_end]

    assert not np.allclose(retriggered_audio, legato_audio)
    assert np.sqrt(np.mean(np.square(legato_window))) > np.sqrt(
        np.mean(np.square(retriggered_window))
    )


def test_score_send_bus_adds_shared_return_to_mix() -> None:
    dry_reference = Score(f0=55.0, auto_master_gain_stage=False)
    dry_reference.add_voice("lead", normalize_lufs=None)
    dry_reference.add_note("lead", start=0.0, duration=0.3, partial=4.0, amp=0.2)

    score = Score(f0=55.0, auto_master_gain_stage=False)
    score.add_send_bus(
        "slap",
        effects=[
            EffectSpec(
                "delay",
                {"delay_seconds": 0.12, "feedback": 0.0, "mix": 1.0},
            )
        ],
    )
    score.add_voice("lead", normalize_lufs=None, sends=[VoiceSend("slap", send_db=0.0)])
    score.add_note("lead", start=0.0, duration=0.3, partial=4.0, amp=0.2)

    dry_stem = score.render_stems()["lead"]
    full_mix = score.render()

    assert np.allclose(dry_stem, dry_reference.render_stems()["lead"])
    assert not np.allclose(full_mix, dry_stem)


def test_multiple_voices_can_share_send_bus() -> None:
    score = Score(f0=55.0, auto_master_gain_stage=False)
    score.add_send_bus("room")
    score.add_voice("lead", normalize_lufs=None, sends=[VoiceSend("room", send_db=0.0)])
    score.add_voice("pad", normalize_lufs=None, sends=[VoiceSend("room", send_db=0.0)])
    score.add_note("lead", start=0.0, duration=0.5, partial=4.0, amp=0.1)
    score.add_note("pad", start=0.0, duration=0.5, partial=5.0, amp=0.1)

    rendered = score.render()

    assert rendered.size > 0
    assert np.max(np.abs(rendered)) > 0.15


def test_send_bus_supports_non_reverb_effects() -> None:
    score = Score(f0=55.0, auto_master_gain_stage=False)
    score.add_send_bus(
        "width",
        effects=[EffectSpec("chorus", {"preset": "juno_subtle", "mix": 1.0})],
    )
    score.add_voice(
        "lead",
        normalize_lufs=None,
        sends=[VoiceSend("width", send_db=-3.0)],
    )
    score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.2)

    rendered = score.render()

    assert rendered.ndim == 2
    assert rendered.shape[0] == 2


def test_render_extends_note_past_note_end_for_release_tail() -> None:
    score = Score(f0=55.0)
    score.add_voice(
        "lead",
        synth_defaults={
            "attack": 0.0,
            "decay": 0.0,
            "sustain_level": 1.0,
            "release": 0.25,
        },
        velocity_humanize=None,
    )
    score.add_note("lead", start=0.0, duration=0.5, partial=4.0, amp=0.2)

    audio = score.render()

    assert len(audio) == int(0.75 * score.sample_rate)
    assert np.max(np.abs(audio[int(0.55 * score.sample_rate) :])) > 0.0


def test_render_short_note_release_reaches_zero_in_tail() -> None:
    score = Score(f0=55.0)
    score.add_voice(
        "lead",
        synth_defaults={
            "attack": 0.04,
            "decay": 0.14,
            "sustain_level": 0.64,
            "release": 0.30,
        },
        velocity_humanize=None,
    )
    score.add_note("lead", start=0.0, duration=0.08, partial=4.0, amp=0.2)

    audio = score.render()

    assert len(audio) == int(0.38 * score.sample_rate)
    assert np.abs(audio[-1]) < 1e-9


def test_extract_window_keeps_overlapping_notes_and_shifts_them() -> None:
    score = Score(f0=55.0)
    score.add_note("lead", start=1.0, duration=1.5, partial=4.0, amp=0.2)
    score.add_note("lead", start=3.25, duration=0.75, partial=5.0, amp=0.2)
    score.add_note("lead", start=5.0, duration=0.5, partial=6.0, amp=0.2)

    windowed_score = score.extract_window(start_seconds=2.0, end_seconds=4.0)

    assert list(windowed_score.voices) == ["lead"]
    kept_notes = windowed_score.voices["lead"].notes
    assert len(kept_notes) == 2
    assert kept_notes[0].start == pytest.approx(0.0)
    assert kept_notes[0].duration == pytest.approx(1.5)
    assert kept_notes[1].start == pytest.approx(1.25)
    assert windowed_score.time_origin_seconds == pytest.approx(2.0)
    assert windowed_score.time_reference_total_dur == pytest.approx(score.total_dur)


def test_extract_window_preserves_absolute_timing_context() -> None:
    score = Score(
        f0=55.0,
        time_origin_seconds=1.5,
        time_reference_total_dur=12.0,
    )
    score.add_note("lead", start=4.0, duration=1.0, partial=4.0, amp=0.2)

    windowed_score = score.extract_window(start_seconds=3.0, end_seconds=6.0)
    resolved_note = windowed_score.resolved_timing_notes()[0]

    assert resolved_note.authored_start == pytest.approx(5.5)
    assert resolved_note.resolved_end == pytest.approx(6.5)


def test_chorus_promotes_mono_to_stereo() -> None:
    signal = np.sin(np.linspace(0.0, 8.0 * np.pi, synth.SAMPLE_RATE, endpoint=False))

    processed = synth.apply_effect_chain(
        signal,
        [EffectSpec("chorus", {"preset": "juno_subtle"})],
    )

    assert processed.ndim == 2
    assert processed.shape[0] == 2
    assert processed.shape[1] == signal.shape[0]
    assert not np.allclose(processed[0], processed[1])


def test_effect_presets_allow_explicit_overrides() -> None:
    signal = np.sin(np.linspace(0.0, 4.0 * np.pi, synth.SAMPLE_RATE, endpoint=False))

    default_processed = synth.apply_effect_chain(
        signal,
        [EffectSpec("chorus", {"preset": "juno_subtle"})],
    )
    overridden_processed = synth.apply_effect_chain(
        signal,
        [EffectSpec("chorus", {"preset": "juno_subtle", "mix": 0.12})],
    )

    default_delta = np.mean(np.abs(default_processed - np.stack([signal, signal])))
    overridden_delta = np.mean(
        np.abs(overridden_processed - np.stack([signal, signal]))
    )
    assert overridden_delta < default_delta


def test_eq_effect_runs_through_apply_effect_chain() -> None:
    signal = np.sin(np.linspace(0.0, 4.0 * np.pi, synth.SAMPLE_RATE, endpoint=False))

    processed = synth.apply_effect_chain(
        signal,
        [
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 120.0,
                            "slope_db_per_oct": 12,
                        },
                        {
                            "kind": "bell",
                            "freq_hz": 1_200.0,
                            "gain_db": 3.0,
                            "q": 1.0,
                        },
                    ]
                },
            )
        ],
    )

    assert processed.shape == signal.shape
    assert np.isfinite(processed).all()


def test_compressor_effect_runs_through_apply_effect_chain() -> None:
    signal = 1.1 * np.sin(
        np.linspace(0.0, 6.0 * np.pi, synth.SAMPLE_RATE, endpoint=False)
    )

    processed = synth.apply_effect_chain(
        signal,
        [
            EffectSpec(
                "compressor",
                {
                    "threshold_db": -20.0,
                    "ratio": 3.0,
                    "attack_ms": 6.0,
                    "release_ms": 180.0,
                    "release_tail_ms": 320.0,
                    "detector_bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 140.0,
                            "slope_db_per_oct": 12,
                        }
                    ],
                },
            )
        ],
    )

    assert processed.shape == signal.shape
    assert np.isfinite(processed).all()


def test_compressor_effect_analysis_reports_gain_reduction_metrics() -> None:
    signal = 1.1 * np.sin(
        np.linspace(0.0, 12.0 * np.pi, synth.SAMPLE_RATE, endpoint=False)
    )

    processed, effect_analysis = synth.apply_effect_chain(
        signal,
        [
            EffectSpec(
                "compressor",
                {
                    "threshold_db": -24.0,
                    "ratio": 4.0,
                    "attack_ms": 4.0,
                    "release_ms": 120.0,
                },
            )
        ],
        return_analysis=True,
    )

    assert processed.shape == signal.shape
    compressor_metrics = effect_analysis[0].metrics
    assert compressor_metrics["avg_gain_reduction_db"] > 0.5
    assert (
        compressor_metrics["max_gain_reduction_db"]
        >= (compressor_metrics["avg_gain_reduction_db"])
    )
    assert "longest_run_above_1db_seconds" in compressor_metrics


def test_score_voice_compressor_can_sidechain_from_another_voice() -> None:
    score = Score(f0=55.0, auto_master_gain_stage=False)
    score.add_voice(
        "kick",
        normalize_peak_db=-6.0,
        velocity_humanize=None,
    )
    score.add_voice(
        "bass",
        normalize_lufs=None,
        velocity_humanize=None,
        effects=[
            EffectSpec(
                "compressor",
                {
                    "threshold_db": -30.0,
                    "ratio": 8.0,
                    "attack_ms": 0.5,
                    "release_ms": 120.0,
                    "knee_db": 0.0,
                    "lookahead_ms": 5.0,
                    "sidechain_source": "kick",
                },
            )
        ],
    )
    score.add_note("kick", start=0.20, duration=0.12, freq=55.0, amp=1.0)
    score.add_note("bass", start=0.0, duration=0.8, partial=2.0, amp=0.12)

    rendered_stems = score.render_stems()
    bass_stem = rendered_stems["bass"]

    dry_reference = Score(f0=55.0, auto_master_gain_stage=False)
    dry_reference.add_voice("bass", normalize_lufs=None, velocity_humanize=None)
    dry_reference.add_note("bass", start=0.0, duration=0.8, partial=2.0, amp=0.12)
    dry_bass_stem = dry_reference.render_stems()["bass"]

    duck_window = slice(int(0.20 * score.sample_rate), int(0.30 * score.sample_rate))
    ducked_rms = float(np.sqrt(np.mean(np.square(bass_stem[duck_window]))))
    dry_rms = float(np.sqrt(np.mean(np.square(dry_bass_stem[duck_window]))))

    assert ducked_rms < dry_rms * 0.7


def test_score_sidechain_processing_is_dependency_order_independent() -> None:
    score = Score(f0=55.0, auto_master_gain_stage=False)
    score.add_voice(
        "pad",
        normalize_lufs=None,
        velocity_humanize=None,
        effects=[
            EffectSpec(
                "compressor",
                {
                    "threshold_db": -32.0,
                    "ratio": 5.0,
                    "attack_ms": 0.5,
                    "release_ms": 160.0,
                    "sidechain_source": "kick",
                },
            )
        ],
    )
    score.add_voice("kick", normalize_peak_db=-6.0, velocity_humanize=None)
    score.add_note("kick", start=0.10, duration=0.10, freq=55.0, amp=1.0)
    score.add_note("pad", start=0.0, duration=0.7, partial=3.0, amp=0.10)

    rendered_stems = score.render_stems()

    assert "pad" in rendered_stems
    assert np.max(np.abs(rendered_stems["pad"])) > 0.0


def test_score_sidechain_rejects_unknown_source_voice() -> None:
    score = Score(f0=55.0, auto_master_gain_stage=False)
    score.add_voice(
        "bass",
        normalize_lufs=None,
        velocity_humanize=None,
        effects=[
            EffectSpec(
                "compressor",
                {
                    "threshold_db": -28.0,
                    "ratio": 4.0,
                    "sidechain_source": "missing_kick",
                },
            )
        ],
    )
    score.add_note("bass", start=0.0, duration=0.5, partial=2.0, amp=0.12)

    with pytest.raises(ValueError, match="Unknown sidechain_source"):
        score.render_stems()


def test_score_sidechain_cycle_is_rejected() -> None:
    score = Score(f0=55.0, auto_master_gain_stage=False)
    score.add_voice(
        "a",
        normalize_lufs=None,
        velocity_humanize=None,
        effects=[
            EffectSpec(
                "compressor",
                {"threshold_db": -28.0, "ratio": 4.0, "sidechain_source": "b"},
            )
        ],
    )
    score.add_voice(
        "b",
        normalize_lufs=None,
        velocity_humanize=None,
        effects=[
            EffectSpec(
                "compressor",
                {"threshold_db": -28.0, "ratio": 4.0, "sidechain_source": "a"},
            )
        ],
    )
    score.add_note("a", start=0.0, duration=0.5, partial=2.0, amp=0.12)
    score.add_note("b", start=0.0, duration=0.5, partial=3.0, amp=0.12)

    with pytest.raises(ValueError, match="contains a cycle"):
        score.render_stems()


def test_plugin_effect_sets_named_plugin_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePlugin:
        def __init__(self) -> None:
            self.drive = 0.0
            self.mix = 0.0

        def __call__(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
            assert sample_rate == synth.SAMPLE_RATE
            return signal

    fake_plugin = FakePlugin()
    monkeypatch.setattr(synth, "load_external_plugin", lambda **_: fake_plugin)
    signal = 0.5 * np.sin(
        np.linspace(0.0, 2.0 * np.pi, synth.SAMPLE_RATE, endpoint=False)
    )

    processed = synth.apply_effect_chain(
        signal,
        [
            EffectSpec(
                "plugin",
                {
                    "plugin_name": "my_bus_plugin",
                    "params": {"drive": 0.42, "mix": 0.18},
                },
            )
        ],
    )

    assert processed.shape == signal.shape
    assert fake_plugin.drive == pytest.approx(0.42)
    assert fake_plugin.mix == pytest.approx(0.18)


def test_plugin_effect_analysis_reports_inactive_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePlugin:
        def __init__(self) -> None:
            self.drive = 0.0

        def __call__(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
            assert sample_rate == synth.SAMPLE_RATE
            return signal

    monkeypatch.setattr(synth, "load_external_plugin", lambda **_: FakePlugin())
    signal = 0.5 * np.sin(
        np.linspace(0.0, 2.0 * np.pi, synth.SAMPLE_RATE, endpoint=False)
    )

    _processed, effect_analysis = synth.apply_effect_chain(
        signal,
        [
            EffectSpec(
                "plugin",
                {
                    "plugin_name": "my_bus_plugin",
                    "params": {"drive": 0.42},
                },
            )
        ],
        return_analysis=True,
    )

    warning_codes = {warning.code for warning in effect_analysis[0].warnings}
    assert "effect_mostly_inactive" in warning_codes


def test_plugin_effect_rejects_unknown_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePlugin:
        def __call__(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
            return signal

    monkeypatch.setattr(synth, "load_external_plugin", lambda **_: FakePlugin())
    signal = np.sin(np.linspace(0.0, 2.0 * np.pi, 1024, endpoint=False))

    with pytest.raises(ValueError, match="no parameter"):
        synth.apply_effect_chain(
            signal,
            [
                EffectSpec(
                    "plugin",
                    {"plugin_name": "my_bus_plugin", "params": {"threshold": -18.0}},
                )
            ],
        )


def test_registered_vst2_plugin_explains_backend_limitation() -> None:
    signal = np.sin(np.linspace(0.0, 2.0 * np.pi, 1024, endpoint=False))

    with pytest.raises(ValueError, match="supports VST3 only"):
        synth.apply_effect_chain(
            signal,
            [EffectSpec("plugin", {"plugin_name": "lsp_compressor_stereo_vst2"})],
        )


def test_has_external_plugin_requires_plugin_and_runtime_libs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin_path = tmp_path / "plugin.vst3"
    runtime_a = tmp_path / "runtime-a.so"
    runtime_b = tmp_path / "runtime-b.so"
    plugin_path.touch()
    runtime_a.touch()
    plugin_spec = synth.ExternalPluginSpec(
        name="test_plugin",
        path=plugin_path,
        preload_libraries=(runtime_a, runtime_b),
    )
    monkeypatch.setitem(synth._PLUGIN_SPECS, "test_plugin", plugin_spec)

    assert synth.has_external_plugin("test_plugin") is False
    runtime_b.touch()
    assert synth.has_external_plugin("test_plugin") is True


def test_tal_reverb_uses_shared_plugin_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePlugin:
        def __init__(self) -> None:
            self.dry = 0.0
            self.wet = 0.0
            self.room_size = 0.0
            self.pre_delay = 0.0
            self.stereo = 0.0

        def __call__(self, signal: np.ndarray, sample_rate: int) -> np.ndarray:
            return signal

    fake_plugin = FakePlugin()
    monkeypatch.setattr(synth, "load_external_plugin", lambda **_: fake_plugin)
    signal = np.sin(np.linspace(0.0, 2.0 * np.pi, 1024, endpoint=False))

    processed = synth.apply_tal_reverb2(
        signal,
        wet=0.24,
        room_size=0.61,
        pre_delay=0.17,
        stereo=0.8,
    )

    assert processed.shape == signal.shape
    assert fake_plugin.dry == pytest.approx(1.0)
    assert fake_plugin.wet == pytest.approx(0.24)
    assert fake_plugin.room_size == pytest.approx(0.61)
    assert fake_plugin.pre_delay == pytest.approx(0.17)
    assert fake_plugin.stereo == pytest.approx(0.8)


def test_score_renders_stereo_when_voice_effects_promote_signal() -> None:
    score = Score(f0=55.0)
    score.add_voice(
        "lead",
        effects=[EffectSpec("chorus", {"preset": "juno_subtle"})],
    )
    score.add_note("lead", start=0.0, duration=1.0, partial=4, amp=0.25)
    score.add_note("plain", start=0.25, duration=0.8, partial=5, amp=0.2)

    audio = score.render()

    assert audio.ndim == 2
    assert audio.shape[0] == 2


def test_voice_normalize_lufs_raises_quiet_voice_toward_target() -> None:
    score = Score(f0=55.0)
    score.add_voice("lead")
    score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.05)

    normalized_stem = score.render_stems()["lead"]
    normalized_lufs, _ = synth.integrated_lufs(
        normalized_stem,
        sample_rate=score.sample_rate,
    )

    plain_score = Score(f0=55.0)
    plain_score.add_voice("lead", normalize_lufs=None)
    plain_score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.05)
    plain_stem = plain_score.render_stems()["lead"]
    plain_lufs, _ = synth.integrated_lufs(
        plain_stem,
        sample_rate=plain_score.sample_rate,
    )

    assert normalized_lufs > plain_lufs
    assert np.isclose(normalized_lufs, -24.0, atol=1.5)


def test_voice_normalize_lufs_preserves_silence() -> None:
    score = Score(f0=55.0)
    score.add_voice("empty", normalize_lufs=-24.0)

    assert score.render_stems() == {}


def test_voice_normalize_lufs_can_be_disabled() -> None:
    normalized_score = Score(f0=55.0)
    normalized_score.add_voice("lead")
    normalized_score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.05)

    raw_score = Score(f0=55.0)
    raw_score.add_voice("lead", normalize_lufs=None)
    raw_score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.05)

    normalized_lufs, _ = synth.integrated_lufs(
        normalized_score.render_stems()["lead"],
        sample_rate=normalized_score.sample_rate,
    )
    raw_lufs, _ = synth.integrated_lufs(
        raw_score.render_stems()["lead"],
        sample_rate=raw_score.sample_rate,
    )

    assert normalized_lufs > raw_lufs


def test_voice_pre_fx_gain_db_increases_stem_level() -> None:
    neutral_score = Score(f0=55.0)
    neutral_score.add_voice("lead", normalize_lufs=None)
    neutral_score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.05)

    boosted_score = Score(f0=55.0)
    boosted_score.add_voice("lead", normalize_lufs=None, pre_fx_gain_db=6.0)
    boosted_score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.05)

    neutral_peak = np.max(np.abs(neutral_score.render_stems()["lead"]))
    boosted_peak = np.max(np.abs(boosted_score.render_stems()["lead"]))

    assert boosted_peak == pytest.approx(neutral_peak * synth.db_to_amp(6.0), rel=5e-2)


def test_voice_mix_db_applies_after_voice_effects() -> None:
    base_score = Score(f0=55.0)
    base_score.add_voice(
        "lead",
        normalize_lufs=None,
        effects=[EffectSpec("chorus", {"preset": "juno_subtle"})],
    )
    base_score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.2)

    lowered_score = Score(f0=55.0)
    lowered_score.add_voice(
        "lead",
        normalize_lufs=None,
        mix_db=-6.0,
        effects=[EffectSpec("chorus", {"preset": "juno_subtle"})],
    )
    lowered_score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.2)

    base_peak = np.max(np.abs(base_score.render_stems()["lead"]))
    lowered_peak = np.max(np.abs(lowered_score.render_stems()["lead"]))

    assert lowered_peak == pytest.approx(base_peak * synth.db_to_amp(-6.0), rel=5e-2)


def test_voice_pan_automation_moves_stereo_image_over_time() -> None:
    score = Score(f0=55.0, auto_master_gain_stage=False)
    score.add_voice(
        "lead",
        normalize_lufs=None,
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="control", name="pan"),
                segments=(
                    AutomationSegment(
                        start=0.0,
                        end=1.0,
                        shape="linear",
                        start_value=-1.0,
                        end_value=1.0,
                    ),
                ),
            )
        ],
    )
    score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.2)

    rendered = score.render_stems()["lead"]
    first_window = rendered[:, : int(0.2 * score.sample_rate)]
    last_window = rendered[
        :, int(0.8 * score.sample_rate) : int(1.0 * score.sample_rate)
    ]

    assert rendered.ndim == 2
    assert np.max(np.abs(first_window[0])) > np.max(np.abs(first_window[1])) * 2.0
    assert np.max(np.abs(last_window[1])) > np.max(np.abs(last_window[0])) * 2.0


def test_voice_send_is_post_fader() -> None:
    base_score = Score(f0=55.0, auto_master_gain_stage=False)
    base_score.add_send_bus("room")
    base_score.add_voice(
        "lead",
        normalize_lufs=None,
        sends=[VoiceSend("room", send_db=0.0)],
    )
    base_score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.2)

    lowered_score = Score(f0=55.0, auto_master_gain_stage=False)
    lowered_score.add_send_bus("room")
    lowered_score.add_voice(
        "lead",
        normalize_lufs=None,
        mix_db=-6.0,
        sends=[VoiceSend("room", send_db=0.0)],
    )
    lowered_score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.2)

    base_peak = np.max(np.abs(base_score.render()))
    lowered_peak = np.max(np.abs(lowered_score.render()))

    assert lowered_peak == pytest.approx(base_peak * synth.db_to_amp(-6.0), rel=5e-2)


def test_voice_send_uses_post_insert_signal() -> None:
    base_score = Score(f0=55.0, auto_master_gain_stage=False)
    base_score.add_send_bus("room")
    base_score.add_voice(
        "lead",
        normalize_lufs=None,
        sends=[VoiceSend("room", send_db=0.0)],
    )
    base_score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.1)

    boosted_score = Score(f0=55.0, auto_master_gain_stage=False)
    boosted_score.add_send_bus("room")
    boosted_score.add_voice(
        "lead",
        normalize_lufs=None,
        pre_fx_gain_db=6.0,
        sends=[VoiceSend("room", send_db=0.0)],
    )
    boosted_score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.1)

    base_peak = np.max(np.abs(base_score.render()))
    boosted_peak = np.max(np.abs(boosted_score.render()))

    assert boosted_peak == pytest.approx(base_peak * synth.db_to_amp(6.0), rel=5e-2)


def test_voice_send_db_automation_changes_send_return_level_over_time() -> None:
    score = Score(f0=55.0, auto_master_gain_stage=False)
    score.add_send_bus("room")
    score.add_voice(
        "lead",
        normalize_lufs=None,
        sends=[
            VoiceSend(
                "room",
                send_db=-30.0,
                automation=[
                    AutomationSpec(
                        target=AutomationTarget(kind="control", name="send_db"),
                        segments=(
                            AutomationSegment(
                                start=0.0,
                                end=1.0,
                                shape="linear",
                                start_value=-30.0,
                                end_value=0.0,
                            ),
                        ),
                    )
                ],
            )
        ],
    )
    score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.2)

    full_mix = score.render()
    dry_stem = score.render_stems()["lead"]
    send_return = full_mix - dry_stem

    early_peak = np.max(np.abs(send_return[: int(0.2 * score.sample_rate)]))
    late_peak = np.max(
        np.abs(send_return[int(0.8 * score.sample_rate) : int(1.0 * score.sample_rate)])
    )

    assert late_peak > early_peak * 6.0


def test_send_bus_pan_automation_moves_return_image_over_time() -> None:
    score = Score(f0=55.0, auto_master_gain_stage=False)
    score.add_send_bus(
        "room",
        pan=0.0,
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="control", name="pan"),
                segments=(
                    AutomationSegment(
                        start=0.0,
                        end=1.0,
                        shape="linear",
                        start_value=-1.0,
                        end_value=1.0,
                    ),
                ),
            )
        ],
    )
    score.add_voice("lead", normalize_lufs=None, sends=[VoiceSend("room", send_db=0.0)])
    score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.2)

    full_mix = score.render()
    dry_stem = score.render_stems()["lead"]
    send_return = full_mix - np.stack([dry_stem, dry_stem])
    first_window = send_return[:, : int(0.2 * score.sample_rate)]
    last_window = send_return[
        :, int(0.8 * score.sample_rate) : int(1.0 * score.sample_rate)
    ]

    assert send_return.ndim == 2
    assert np.max(np.abs(first_window[0])) > np.max(np.abs(first_window[1])) * 2.0
    assert np.max(np.abs(last_window[1])) > np.max(np.abs(last_window[0])) * 2.0


def test_effect_mix_automation_changes_insert_wetness_over_time() -> None:
    score = Score(f0=55.0, auto_master_gain_stage=False)
    score.add_voice(
        "lead",
        normalize_lufs=None,
        effects=[
            EffectSpec(
                "chorus",
                {"preset": "juno_subtle", "mix": 0.0},
                automation=[
                    AutomationSpec(
                        target=AutomationTarget(kind="control", name="mix"),
                        segments=(
                            AutomationSegment(
                                start=0.0,
                                end=1.0,
                                shape="linear",
                                start_value=0.0,
                                end_value=1.0,
                            ),
                        ),
                    )
                ],
            )
        ],
    )
    score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.2)

    dry_reference = Score(f0=55.0, auto_master_gain_stage=False)
    dry_reference.add_voice(
        "lead",
        normalize_lufs=None,
        effects=[EffectSpec("chorus", {"preset": "juno_subtle", "mix": 0.0})],
    )
    dry_reference.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.2)

    processed = score.render_stems()["lead"]
    dry = dry_reference.render_stems()["lead"]
    early_difference = np.max(
        np.abs(
            processed[:, : int(0.2 * score.sample_rate)]
            - dry[:, : int(0.2 * score.sample_rate)]
        )
    )
    late_difference = np.max(
        np.abs(
            processed[:, int(0.8 * score.sample_rate) : int(1.0 * score.sample_rate)]
            - dry[
                :,
                int(0.8 * dry_reference.sample_rate) : int(
                    1.0 * dry_reference.sample_rate
                ),
            ]
        )
    )

    assert late_difference > early_difference * 3.0


def test_render_stems_excludes_send_returns() -> None:
    dry_reference = Score(f0=55.0, auto_master_gain_stage=False)
    dry_reference.add_voice("lead", normalize_lufs=None)
    dry_reference.add_note("lead", start=0.0, duration=0.25, partial=4.0, amp=0.2)

    score = Score(f0=55.0, auto_master_gain_stage=False)
    score.add_send_bus(
        "echo",
        effects=[
            EffectSpec(
                "delay",
                {"delay_seconds": 0.16, "feedback": 0.0, "mix": 1.0},
            )
        ],
    )
    score.add_voice("lead", normalize_lufs=None, sends=[VoiceSend("echo", send_db=0.0)])
    score.add_note("lead", start=0.0, duration=0.25, partial=4.0, amp=0.2)

    dry_stem = score.render_stems()["lead"]
    full_mix = score.render()

    assert np.allclose(dry_stem, dry_reference.render_stems()["lead"])
    assert not np.allclose(full_mix, dry_stem)


def test_add_voice_rejects_non_finite_gain_controls() -> None:
    score = Score(f0=55.0)

    with pytest.raises(ValueError, match="pre_fx_gain_db must be finite"):
        score.add_voice("lead", pre_fx_gain_db=float("inf"))

    with pytest.raises(ValueError, match="mix_db must be finite"):
        score.add_voice("lead", mix_db=float("nan"))


def test_send_bus_validation_rejects_invalid_configs() -> None:
    with pytest.raises(ValueError, match="send bus names must be unique"):
        Score(
            f0=55.0,
            send_buses=[SendBusSpec(name="room"), SendBusSpec(name="room")],
        )

    with pytest.raises(ValueError, match="voice send target does not exist on score"):
        Score(f0=55.0).add_voice("lead", sends=[VoiceSend("missing")])

    with pytest.raises(ValueError, match="voice send_db must be finite"):
        VoiceSend("room", send_db=float("inf"))

    with pytest.raises(ValueError, match="send bus return_db must be finite"):
        SendBusSpec(name="room", return_db=float("nan"))


def test_render_with_effect_analysis_includes_send_effects() -> None:
    score = Score(f0=55.0, auto_master_gain_stage=False)
    score.add_send_bus(
        "room",
        effects=[
            EffectSpec(
                "compressor",
                {"threshold_db": -28.0, "ratio": 3.0, "attack_ms": 4.0},
            )
        ],
    )
    score.add_voice("lead", normalize_lufs=None, sends=[VoiceSend("room", send_db=0.0)])
    score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.3)

    _mix, stems, effect_analysis = score.render_with_effect_analysis()

    assert "lead" in stems
    assert "room" in effect_analysis["send_effects"]
    assert effect_analysis["send_effects"]["room"]


def test_score_auto_master_gain_stage_raises_balanced_mix_toward_target() -> None:
    unstaged_score = Score(
        f0=55.0,
        auto_master_gain_stage=False,
    )
    unstaged_score.add_voice("lead", mix_db=-18.0)
    unstaged_score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.05)

    staged_score = Score(f0=55.0)
    staged_score.add_voice("lead", mix_db=-18.0)
    staged_score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.05)

    unstaged_lufs, _ = synth.integrated_lufs(
        unstaged_score.render(),
        sample_rate=unstaged_score.sample_rate,
    )
    staged_lufs, _ = synth.integrated_lufs(
        staged_score.render(),
        sample_rate=staged_score.sample_rate,
    )

    assert staged_lufs > unstaged_lufs
    assert np.isclose(staged_lufs, -24.0, atol=1.5)


def test_score_auto_master_gain_stage_respects_peak_safety_ceiling() -> None:
    score = Score(
        f0=55.0,
        master_bus_target_lufs=-12.0,
        master_bus_max_true_peak_dbfs=-10.0,
    )
    score.add_voice("lead", normalize_lufs=None, mix_db=-30.0)
    score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.5)

    rendered = score.render()
    peak_dbfs = synth.amp_to_db(float(np.max(np.abs(rendered))))

    assert peak_dbfs <= -10.0 + 0.6


def test_score_master_input_gain_db_scales_mix_before_master_effects() -> None:
    dry_score = Score(f0=55.0, auto_master_gain_stage=False, master_input_gain_db=0.0)
    dry_score.add_voice("lead", normalize_lufs=None)
    dry_score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.1)

    boosted_score = Score(
        f0=55.0,
        auto_master_gain_stage=False,
        master_input_gain_db=6.0,
    )
    boosted_score.add_voice("lead", normalize_lufs=None)
    boosted_score.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.1)

    dry_peak = np.max(np.abs(dry_score.render()))
    boosted_peak = np.max(np.abs(boosted_score.render()))

    assert boosted_peak == pytest.approx(dry_peak * synth.db_to_amp(6.0), rel=5e-2)


def test_score_rejects_non_finite_master_input_gain_db() -> None:
    with pytest.raises(ValueError, match="master_input_gain_db must be finite"):
        Score(f0=55.0, master_input_gain_db=float("inf"))

    with pytest.raises(ValueError, match="master_bus_target_lufs must be finite"):
        Score(f0=55.0, master_bus_target_lufs=float("nan"))

    with pytest.raises(
        ValueError,
        match="master_bus_max_true_peak_dbfs must be finite",
    ):
        Score(f0=55.0, master_bus_max_true_peak_dbfs=float("inf"))


def test_extract_window_preserves_master_input_gain_db() -> None:
    score = Score(
        f0=55.0,
        auto_master_gain_stage=False,
        master_bus_target_lufs=-22.0,
        master_bus_max_true_peak_dbfs=-8.0,
        master_input_gain_db=3.0,
    )
    score.add_voice("lead", normalize_lufs=None)
    score.add_note("lead", start=1.0, duration=1.0, partial=4.0, amp=0.1)

    window = score.extract_window(start_seconds=0.5, end_seconds=2.5)

    assert window.auto_master_gain_stage is False
    assert window.master_bus_target_lufs == pytest.approx(-22.0)
    assert window.master_bus_max_true_peak_dbfs == pytest.approx(-8.0)
    assert window.master_input_gain_db == pytest.approx(3.0)


def test_extract_window_preserves_send_buses() -> None:
    score = Score(f0=55.0)
    score.add_send_bus("room", return_db=-3.0)
    score.add_voice("lead", sends=[VoiceSend("room", send_db=-6.0)])
    score.add_note("lead", start=1.0, duration=1.0, partial=4.0, amp=0.1)

    window = score.extract_window(start_seconds=0.5, end_seconds=2.5)

    assert [send_bus.name for send_bus in window.send_buses] == ["room"]
    assert window.voices["lead"].sends == [VoiceSend("room", send_db=-6.0)]


def test_true_peak_estimation_uses_loudest_stereo_channel() -> None:
    duration = synth.SAMPLE_RATE
    time = np.arange(duration, dtype=np.float64) / synth.SAMPLE_RATE
    left = 0.2 * np.sin(2.0 * np.pi * 440.0 * time)
    right = 0.8 * np.sin(2.0 * np.pi * 440.0 * time)
    stereo_signal = np.stack([left, right])

    true_peak = synth.estimate_true_peak_amplitude(
        stereo_signal,
        oversample_factor=1,
    )

    assert true_peak == pytest.approx(0.8, rel=1e-3)


def test_finalize_master_falls_back_when_lsp_limiter_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(synth, "has_external_plugin", lambda plugin_name: False)
    signal = 0.2 * np.sin(
        np.linspace(0.0, 4.0 * np.pi, synth.SAMPLE_RATE, endpoint=False)
    )

    result = synth.finalize_master(signal, sample_rate=synth.SAMPLE_RATE)
    assert isinstance(result, synth.MasteringResult)
    assert np.isfinite(result.integrated_lufs)
    assert np.isfinite(result.true_peak_dbfs)


def test_finalize_master_targets_lufs_and_true_peak_with_limiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(synth, "has_external_plugin", lambda plugin_name: True)

    def fake_apply_lsp_limiter(
        signal: np.ndarray,
        *,
        threshold_db: float,
        input_gain_db: float,
        output_gain_db: float,
    ) -> np.ndarray:
        processed = np.asarray(signal, dtype=np.float64) * synth.db_to_amp(
            input_gain_db + output_gain_db
        )
        ceiling = synth.db_to_amp(threshold_db)
        return np.clip(processed, -ceiling, ceiling)

    monkeypatch.setattr(synth, "apply_lsp_limiter", fake_apply_lsp_limiter)
    monkeypatch.setattr(
        synth,
        "normalize_true_peak",
        lambda signal, **_: np.asarray(signal, dtype=np.float64),
    )
    signal = 0.04 * np.sin(
        np.linspace(0.0, 40.0 * np.pi, synth.SAMPLE_RATE * 2, endpoint=False)
    )

    mastering_result = synth.finalize_master(
        signal,
        sample_rate=synth.SAMPLE_RATE,
        target_lufs=-18.0,
        true_peak_ceiling_dbfs=-0.5,
        max_iterations=8,
    )

    assert mastering_result.integrated_lufs == pytest.approx(-18.0, abs=1.5)
    assert mastering_result.true_peak_dbfs <= -0.5 + 0.1


def test_finalize_master_iterative_limiter_reapplies_gain_from_original_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(synth, "has_external_plugin", lambda plugin_name: True)
    limiter_inputs: list[np.ndarray] = []
    lufs_values = iter(
        [(-30.0, 1.0), (-25.0, 1.0), (-18.0, 1.0), (-18.0, 1.0), (-18.0, 1.0)]
    )

    def fake_integrated_lufs(
        signal: np.ndarray,
        *,
        sample_rate: int,
    ) -> tuple[float, float]:
        del signal, sample_rate
        return next(lufs_values)

    def fake_apply_lsp_limiter(
        signal: np.ndarray,
        *,
        threshold_db: float,
        input_gain_db: float,
        output_gain_db: float,
    ) -> np.ndarray:
        del threshold_db, input_gain_db, output_gain_db
        limiter_inputs.append(np.asarray(signal, dtype=np.float64).copy())
        return np.asarray(signal, dtype=np.float64) * 0.5

    monkeypatch.setattr(synth, "integrated_lufs", fake_integrated_lufs)
    monkeypatch.setattr(synth, "apply_lsp_limiter", fake_apply_lsp_limiter)
    monkeypatch.setattr(
        synth,
        "normalize_true_peak",
        lambda signal, **_: np.asarray(signal, dtype=np.float64) + 0.25,
    )
    monkeypatch.setattr(
        synth, "estimate_true_peak_amplitude", lambda *args, **kwargs: 0.9
    )

    signal = np.ones(16, dtype=np.float64)
    synth.finalize_master(signal, sample_rate=synth.SAMPLE_RATE, max_iterations=4)

    np.testing.assert_allclose(limiter_inputs[0], signal)
    np.testing.assert_allclose(limiter_inputs[1], signal)


def test_normalize_true_peak_boosts_under_ceiling_signal() -> None:
    signal = np.array([0.0, 0.1, -0.1, 0.0], dtype=np.float64)

    normalized = synth.normalize_true_peak(
        signal,
        target_peak_dbfs=-0.5,
        oversample_factor=1,
    )

    assert np.max(np.abs(normalized)) == pytest.approx(synth.db_to_amp(-0.5))


def test_finalize_master_boosts_to_true_peak_ceiling_when_headroom_remains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(synth, "has_external_plugin", lambda plugin_name: True)
    monkeypatch.setattr(
        synth,
        "apply_lsp_limiter",
        lambda signal, **kwargs: (
            np.asarray(signal, dtype=np.float64)
            * synth.db_to_amp(kwargs["input_gain_db"] + kwargs["output_gain_db"])
        ),
    )

    signal = 0.04 * np.sin(
        np.linspace(0.0, 40.0 * np.pi, synth.SAMPLE_RATE * 2, endpoint=False)
    )

    mastering_result = synth.finalize_master(
        signal,
        sample_rate=synth.SAMPLE_RATE,
        target_lufs=-18.0,
        true_peak_ceiling_dbfs=-0.5,
        max_iterations=4,
    )

    assert mastering_result.true_peak_dbfs == pytest.approx(-0.5, abs=0.15)
    assert np.max(np.abs(mastering_result.signal)) == pytest.approx(
        synth.db_to_amp(-0.5),
        abs=1e-3,
    )


def test_voice_pan_promotes_mono_voice_to_stereo() -> None:
    score = Score(f0=55.0)
    score.add_voice("lead", pan=0.25)
    score.add_note("lead", start=0.0, duration=1.0, partial=4, amp=0.25)

    audio = score.render()

    assert audio.ndim == 2
    assert audio.shape[0] == 2
    assert not np.allclose(audio[0], audio[1])


def test_add_voice_rejects_out_of_range_pan() -> None:
    score = Score(f0=55.0)

    with pytest.raises(ValueError, match="pan must be between -1 and 1"):
        score.add_voice("lead", pan=1.5)


def test_timing_humanize_offsets_are_deterministic_for_seed() -> None:
    targets = [
        TimingTarget(key=("a", 0), voice_name="a", start=0.0),
        TimingTarget(key=("a", 1), voice_name="a", start=2.0),
        TimingTarget(key=("b", 0), voice_name="b", start=0.0),
        TimingTarget(key=("b", 1), voice_name="b", start=2.0),
    ]
    humanize = TimingHumanizeSpec(seed=17, micro_jitter_ms=0.0)

    first = build_timing_offsets(targets=targets, humanize=humanize, total_dur=4.0)
    second = build_timing_offsets(targets=targets, humanize=humanize, total_dur=4.0)

    assert first == second


def test_timing_humanize_keeps_voices_strongly_correlated() -> None:
    targets: list[TimingTarget] = []
    for voice_name in ("lead", "alto", "bass"):
        for index, start in enumerate(np.linspace(0.0, 18.0, 10)):
            targets.append(
                TimingTarget(
                    key=(voice_name, index), voice_name=voice_name, start=float(start)
                )
            )

    humanize = TimingHumanizeSpec(
        seed=9,
        ensemble_amount_ms=24.0,
        follow_strength=0.94,
        voice_spread_ms=4.0,
        micro_jitter_ms=0.0,
        chord_spread_ms=0.0,
    )
    offsets = build_timing_offsets(targets=targets, humanize=humanize, total_dur=20.0)

    lead = np.asarray([offsets[("lead", index)] for index in range(10)])
    alto = np.asarray([offsets[("alto", index)] for index in range(10)])
    bass = np.asarray([offsets[("bass", index)] for index in range(10)])

    assert np.corrcoef(lead, alto)[0, 1] > 0.9
    assert np.corrcoef(lead, bass)[0, 1] > 0.9


def test_timing_humanize_chord_spread_is_small_secondary_layer() -> None:
    targets = [
        TimingTarget(key=("lead", 0), voice_name="lead", start=3.0),
        TimingTarget(key=("alto", 0), voice_name="alto", start=3.0),
        TimingTarget(key=("bass", 0), voice_name="bass", start=3.0),
    ]
    humanize = TimingHumanizeSpec(
        seed=3,
        ensemble_amount_ms=0.0,
        voice_spread_ms=0.0,
        micro_jitter_ms=0.0,
        chord_spread_ms=6.0,
    )
    offsets = build_timing_offsets(targets=targets, humanize=humanize, total_dur=6.0)
    offset_values = np.asarray(list(offsets.values()))

    assert np.isclose(offset_values.mean(), 0.0)
    assert np.max(offset_values) == pytest.approx(0.003)
    assert np.min(offset_values) == pytest.approx(-0.003)


def test_envelope_humanize_varies_adsr_within_valid_ranges() -> None:
    humanize = EnvelopeHumanizeSpec(preset="breathing_pad", seed=21)

    early = resolve_envelope_params(
        base_attack=0.4,
        base_decay=0.2,
        base_sustain_level=0.7,
        base_release=1.0,
        note_start=1.0,
        humanize=humanize,
        total_dur=20.0,
        voice_name="pad",
    )
    late = resolve_envelope_params(
        base_attack=0.4,
        base_decay=0.2,
        base_sustain_level=0.7,
        base_release=1.0,
        note_start=15.0,
        humanize=humanize,
        total_dur=20.0,
        voice_name="pad",
    )

    assert early != late
    for attack, decay, sustain_level, release in (early, late):
        assert attack >= 0.0
        assert decay >= 0.0
        assert 0.0 <= sustain_level <= 1.0
        assert release >= 0.0


def test_velocity_humanize_is_deterministic_for_seed() -> None:
    targets = [
        VelocityTarget(
            key=("lead", 0), voice_name="lead", group_name="band", start=0.0
        ),
        VelocityTarget(
            key=("lead", 1), voice_name="lead", group_name="band", start=2.0
        ),
        VelocityTarget(
            key=("alto", 0), voice_name="alto", group_name="band", start=0.0
        ),
        VelocityTarget(
            key=("alto", 1), voice_name="alto", group_name="band", start=2.0
        ),
    ]
    humanize = VelocityHumanizeSpec(seed=17, note_jitter=0.0)

    first = build_velocity_multipliers(
        targets=targets,
        humanize=humanize,
        total_dur=4.0,
    )
    second = build_velocity_multipliers(
        targets=targets,
        humanize=humanize,
        total_dur=4.0,
    )

    assert first == second


def test_velocity_humanize_keeps_grouped_voices_strongly_correlated() -> None:
    targets: list[VelocityTarget] = []
    for voice_name in ("lead", "alto", "bass"):
        for index, start in enumerate(np.linspace(0.0, 18.0, 10)):
            targets.append(
                VelocityTarget(
                    key=(voice_name, index),
                    voice_name=voice_name,
                    group_name="ensemble",
                    start=float(start),
                )
            )

    humanize = VelocityHumanizeSpec(
        seed=9,
        group_amount=0.08,
        follow_strength=0.95,
        voice_spread=0.02,
        note_jitter=0.0,
        chord_spread=0.0,
        min_multiplier=0.85,
        max_multiplier=1.15,
    )
    multipliers = build_velocity_multipliers(
        targets=targets,
        humanize=humanize,
        total_dur=20.0,
    )

    lead = np.asarray([multipliers[("lead", index)] for index in range(10)])
    alto = np.asarray([multipliers[("alto", index)] for index in range(10)])
    bass = np.asarray([multipliers[("bass", index)] for index in range(10)])

    assert np.corrcoef(lead, alto)[0, 1] > 0.9
    assert np.corrcoef(lead, bass)[0, 1] > 0.9


def test_velocity_humanize_default_preset_stays_subtle() -> None:
    targets = [
        VelocityTarget(
            key=("lead", index),
            voice_name="lead",
            group_name="lead",
            start=float(index),
        )
        for index in range(8)
    ]

    multipliers = build_velocity_multipliers(
        targets=targets,
        humanize=VelocityHumanizeSpec(seed=4),
        total_dur=8.0,
    )

    assert all(0.9 <= value <= 1.1 for value in multipliers.values())


def test_score_render_is_deterministic_with_same_humanize_seed() -> None:
    base_timing = TimingHumanizeSpec(seed=12)
    first = Score(f0=55.0, timing_humanize=base_timing)
    second = Score(f0=55.0, timing_humanize=base_timing)
    for score in (first, second):
        score.add_voice("lead", envelope_humanize=EnvelopeHumanizeSpec(seed=5))
        score.add_voice("alto")
        score.add_note("lead", start=0.0, duration=0.8, partial=4.0, amp=0.2)
        score.add_note("lead", start=1.0, duration=0.8, partial=5.0, amp=0.2)
        score.add_note("alto", start=0.0, duration=1.2, partial=3.0, amp=0.15)
        score.add_note("alto", start=1.0, duration=1.0, partial=4.0, amp=0.15)

    assert np.allclose(first.render(), second.render())


def test_score_render_changes_with_different_humanize_seed() -> None:
    neutral = Score(f0=55.0, timing_humanize=TimingHumanizeSpec(seed=10))
    changed = Score(f0=55.0, timing_humanize=TimingHumanizeSpec(seed=11))
    for score in (neutral, changed):
        score.add_voice("lead")
        score.add_note("lead", start=0.0, duration=0.8, partial=4.0, amp=0.2)
        score.add_note("lead", start=1.0, duration=0.8, partial=5.0, amp=0.2)
        score.add_note("lead", start=2.0, duration=0.8, partial=6.0, amp=0.2)

    neutral_audio = neutral.render()
    changed_audio = changed.render()

    if neutral_audio.shape != changed_audio.shape:
        assert neutral_audio.shape != changed_audio.shape
        return

    assert not np.allclose(neutral_audio, changed_audio)


def test_stereo_effect_chain_continues_into_saturation() -> None:
    signal = np.sin(np.linspace(0.0, 6.0 * np.pi, synth.SAMPLE_RATE, endpoint=False))

    processed = synth.apply_effect_chain(
        signal,
        [
            EffectSpec("chorus", {"preset": "juno_subtle"}),
            EffectSpec("saturation", {"preset": "tube_warm"}),
        ],
    )

    assert processed.ndim == 2
    assert processed.shape[0] == 2
    assert np.isfinite(processed).all()


def test_saturation_gain_compensation_keeps_level_reasonable() -> None:
    signal = 0.35 * np.sin(
        np.linspace(0.0, 10.0 * np.pi, synth.SAMPLE_RATE, endpoint=False)
    )

    processed = synth.apply_effect_chain(
        signal,
        [EffectSpec("saturation", {"preset": "neve_gentle"})],
    )

    input_peak = np.max(np.abs(signal))
    output_peak = np.max(np.abs(processed))
    assert output_peak > 0
    assert np.isclose(output_peak, input_peak, rtol=0.25)


def test_saturation_modern_asymmetry_remains_dc_safe() -> None:
    t = np.arange(synth.SAMPLE_RATE, dtype=np.float64) / synth.SAMPLE_RATE
    signal = 0.4 * np.sin(2.0 * np.pi * 220.0 * t)

    processed = synth.apply_saturation(
        signal,
        drive=1.8,
        mix=1.0,
        mode="triode",
        bias=0.2,
        algorithm="modern",
        compensation_mode="none",
    )

    assert abs(float(np.mean(processed))) < 0.005


def test_saturation_effect_analysis_reports_shaper_activity() -> None:
    t = np.arange(synth.SAMPLE_RATE, dtype=np.float64) / synth.SAMPLE_RATE
    signal = 0.7 * np.sin(2.0 * np.pi * 220.0 * t)

    _processed, effect_analysis = synth.apply_effect_chain(
        signal,
        [EffectSpec("saturation", {"drive": 4.0, "mix": 0.9, "bias": 0.2})],
        return_analysis=True,
    )

    saturation_metrics = effect_analysis[0].metrics
    assert saturation_metrics["shaper_hot_fraction"] > 0.0
    assert "crest_factor_delta_db" in saturation_metrics
    assert saturation_metrics["algorithm"] == "modern"


def test_saturation_modern_auto_uses_lufs_for_sustained_signal() -> None:
    t = np.arange(synth.SAMPLE_RATE * 2, dtype=np.float64) / synth.SAMPLE_RATE
    signal = 0.26 * np.sin(2.0 * np.pi * 220.0 * t)

    processed, analysis = synth.apply_saturation(
        signal,
        drive=1.45,
        mix=0.8,
        mode="tube",
        compensation_mode="auto",
        return_analysis=True,
    )
    input_lufs, _ = synth.integrated_lufs(signal, sample_rate=synth.SAMPLE_RATE)
    output_lufs, _ = synth.integrated_lufs(processed, sample_rate=synth.SAMPLE_RATE)

    assert analysis["compensation_mode_used"] == "lufs"
    assert abs(output_lufs - input_lufs) < 1.5


def test_saturation_modern_auto_uses_rms_for_short_signal() -> None:
    t = np.linspace(0.0, 0.15, int(0.15 * synth.SAMPLE_RATE), endpoint=False)
    signal = 0.5 * np.sin(2.0 * np.pi * 110.0 * t) * np.exp(-t / 0.05)

    _processed, analysis = synth.apply_saturation(
        signal,
        drive=1.5,
        mix=0.8,
        mode="tube",
        compensation_mode="auto",
        return_analysis=True,
    )

    assert analysis["compensation_mode_used"] == "rms"


def test_saturation_explicit_lufs_is_strict() -> None:
    t = np.linspace(0.0, 0.15, int(0.15 * synth.SAMPLE_RATE), endpoint=False)
    signal = 0.5 * np.sin(2.0 * np.pi * 110.0 * t) * np.exp(-t / 0.05)

    _processed, analysis = synth.apply_saturation(
        signal,
        drive=1.5,
        mix=0.8,
        mode="tube",
        compensation_mode="lufs",
        return_analysis=True,
    )

    assert analysis["compensation_mode_used"] == "lufs"


def _band_energy(
    signal: np.ndarray,
    *,
    low_hz: float,
    high_hz: float,
    sample_rate: int,
) -> float:
    mono = np.asarray(signal, dtype=np.float64)
    if mono.ndim == 2:
        mono = np.mean(mono, axis=0, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(mono.size))) ** 2
    freqs = np.fft.rfftfreq(mono.size, 1.0 / sample_rate)
    mask = (freqs >= low_hz) & (freqs < high_hz)
    return float(np.sum(spectrum[mask]))


def _alias_proxy(
    signal: np.ndarray,
    *,
    fundamental_hz: float,
    sample_rate: int,
    max_harmonic: int = 12,
    tolerance_hz: float = 30.0,
) -> float:
    mono = np.asarray(signal, dtype=np.float64)
    if mono.ndim == 2:
        mono = np.mean(mono, axis=0, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(mono.size))) ** 2
    freqs = np.fft.rfftfreq(mono.size, 1.0 / sample_rate)
    mask = freqs < 20.0
    for harmonic_index in range(1, max_harmonic + 1):
        harmonic_hz = harmonic_index * fundamental_hz
        if harmonic_hz >= sample_rate / 2.0:
            continue
        mask |= np.abs(freqs - harmonic_hz) <= tolerance_hz
    total_energy = float(np.sum(spectrum))
    return float(np.sum(spectrum[~mask]) / max(total_energy, 1e-12))


def test_saturation_modern_preserve_highs_retains_more_air() -> None:
    t = np.arange(synth.SAMPLE_RATE, dtype=np.float64) / synth.SAMPLE_RATE
    signal = (
        0.25 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.18 * np.sin(2.0 * np.pi * 6_400.0 * t)
        + 0.10 * np.sin(2.0 * np.pi * 8_900.0 * t)
    )

    without_preserve = synth.apply_saturation(
        signal,
        drive=1.7,
        mix=1.0,
        mode="tube",
        fidelity=0.0,
        preserve_highs_hz=0.0,
        compensation_mode="none",
    )
    with_preserve = synth.apply_saturation(
        signal,
        drive=1.7,
        mix=1.0,
        mode="tube",
        fidelity=0.95,
        preserve_highs_hz=6_000.0,
        compensation_mode="none",
    )

    input_high_band = _band_energy(
        signal,
        low_hz=6_000.0,
        high_hz=12_000.0,
        sample_rate=synth.SAMPLE_RATE,
    )
    without_high_band = _band_energy(
        without_preserve,
        low_hz=6_000.0,
        high_hz=12_000.0,
        sample_rate=synth.SAMPLE_RATE,
    )
    with_high_band = _band_energy(
        with_preserve,
        low_hz=6_000.0,
        high_hz=12_000.0,
        sample_rate=synth.SAMPLE_RATE,
    )
    assert abs(with_high_band - input_high_band) < abs(
        without_high_band - input_high_band
    )


def test_saturation_modern_has_lower_alias_proxy_than_legacy() -> None:
    t = np.arange(synth.SAMPLE_RATE, dtype=np.float64) / synth.SAMPLE_RATE
    signal = 0.6 * np.sin(2.0 * np.pi * 9_000.0 * t)

    legacy = synth.apply_saturation(
        signal,
        drive=8.0,
        mix=1.0,
        algorithm="legacy",
        compensation_mode="none",
    )
    modern = synth.apply_saturation(
        signal,
        drive=1.85,
        mix=1.0,
        mode="triode",
        oversample_factor=8,
        fidelity=0.45,
        compensation_mode="none",
    )

    legacy_alias = _alias_proxy(
        legacy,
        fundamental_hz=9_000.0,
        sample_rate=synth.SAMPLE_RATE,
    )
    modern_alias = _alias_proxy(
        modern,
        fundamental_hz=9_000.0,
        sample_rate=synth.SAMPLE_RATE,
    )
    assert modern_alias < legacy_alias


def test_saturation_modern_preserve_lows_keeps_low_end_more_stable() -> None:
    t = np.arange(synth.SAMPLE_RATE, dtype=np.float64) / synth.SAMPLE_RATE
    signal = (
        0.50 * np.sin(2.0 * np.pi * 55.0 * t)
        + 0.16 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.08 * np.sin(2.0 * np.pi * 4_500.0 * t)
    )

    input_low_band = _band_energy(
        signal,
        low_hz=20.0,
        high_hz=120.0,
        sample_rate=synth.SAMPLE_RATE,
    )
    without_preserve = synth.apply_saturation(
        signal,
        drive=1.9,
        mix=1.0,
        mode="iron",
        fidelity=0.0,
        preserve_lows_hz=0.0,
        compensation_mode="none",
    )
    with_preserve = synth.apply_saturation(
        signal,
        drive=1.9,
        mix=1.0,
        mode="iron",
        fidelity=0.95,
        preserve_lows_hz=120.0,
        compensation_mode="none",
    )

    without_low_delta = abs(
        _band_energy(
            without_preserve,
            low_hz=20.0,
            high_hz=120.0,
            sample_rate=synth.SAMPLE_RATE,
        )
        - input_low_band
    )
    with_low_delta = abs(
        _band_energy(
            with_preserve,
            low_hz=20.0,
            high_hz=120.0,
            sample_rate=synth.SAMPLE_RATE,
        )
        - input_low_band
    )
    assert with_low_delta < without_low_delta


def test_kick_punch_preset_compresses_and_recovers() -> None:
    """kick_punch should compress the kick body without silencing it.

    kick_punch uses a 6 ms attack, which intentionally lets the initial
    transient through uncompressed for punch.  The 1 dB makeup gain can push
    the output peak slightly above the input peak, so a peak comparison is
    not meaningful here.  Instead we verify that the body (post-transient) is
    attenuated: the RMS of the post-attack region should be lower in the
    processed signal than in the dry signal.
    """
    # Kick-like burst: exponential decay from ~-6 dBFS peak
    t = np.linspace(0.0, 0.5, synth.SAMPLE_RATE // 2, endpoint=False)
    kick = 0.5 * np.sin(2.0 * np.pi * 62.0 * t) * np.exp(-t / 0.3)

    processed = synth.apply_effect_chain(
        kick,
        [EffectSpec("compressor", {"preset": "kick_punch"})],
    )
    assert isinstance(processed, np.ndarray)
    assert np.isfinite(processed).all()

    # Skip the first ~20 ms (transient) so the attack phase doesn't mask GR.
    body_start = int(0.020 * synth.SAMPLE_RATE)
    input_body_rms = float(np.sqrt(np.mean(kick[body_start:] ** 2)))
    output_body_rms = float(np.sqrt(np.mean(processed[body_start:] ** 2)))

    # Compression should reduce the body RMS but not silence it
    assert output_body_rms < input_body_rms, "kick_punch should compress the body"
    assert output_body_rms > input_body_rms * 0.2, (
        "kick_punch should not crush the signal"
    )


def test_plot_piano_roll_writes_file(tmp_path: Path) -> None:
    score = Score(f0=55.0)
    score.add_note("a", start=0.0, duration=1.0, partial=4, amp=0.3)

    output_path = tmp_path / "roll.png"
    figure, _ = score.plot_piano_roll(output_path)

    assert output_path.exists()
    figure.clf()


def test_render_piece_writes_audio_and_plot(tmp_path: Path) -> None:
    result = render_piece("chord_4567", output_dir=tmp_path, save_plot=True)
    audio_path, plot_path = result

    assert audio_path.exists()
    assert result.version_audio_path is not None
    assert result.version_audio_path.exists()
    assert "chord_4567/versions/" in str(result.version_audio_path)
    assert plot_path is not None
    assert plot_path.exists()
    assert result.version_plot_path is not None
    assert result.version_plot_path.exists()
    assert result.analysis_manifest_path is not None
    assert result.analysis_manifest_path.exists()
    assert result.version_analysis_manifest_path is not None
    assert result.version_analysis_manifest_path.exists()
    assert result.render_metadata_path is not None
    assert result.render_metadata_path.exists()
    assert result.version_metadata_path is not None
    assert result.version_metadata_path.exists()
    manifest = json.loads(result.analysis_manifest_path.read_text(encoding="utf-8"))
    assert Path(manifest["manifest_path"]) == result.analysis_manifest_path
    assert Path(manifest["mix"]["artifacts"]["spectrum"]).exists()
    assert "artifact_risk" in manifest
    assert "versions/" not in str(result.analysis_manifest_path)
    assert "versions/" not in manifest["manifest_path"]
    assert "versions/" not in manifest["mix"]["artifacts"]["spectrum"]
    render_metadata = json.loads(
        result.render_metadata_path.read_text(encoding="utf-8")
    )
    assert render_metadata["piece_name"] == "chord_4567"
    assert render_metadata["request"]["save_plot"] is True
    assert render_metadata["request"]["export_target_lufs"] == -18.0
    assert render_metadata["request"]["export_true_peak_ceiling_dbfs"] == -0.5
    assert render_metadata["score_summary"]["note_count"] == 4
    assert render_metadata["score_summary"]["voice_names"] == ["chord"]
    assert render_metadata["score_snapshot"]["voices"]["chord"]["notes"]
    assert (
        Path(render_metadata["artifacts"]["versioned"]["audio_path"])
        == result.version_audio_path
    )
    assert (
        Path(render_metadata["artifacts"]["latest"]["analysis_manifest_path"])
        == result.analysis_manifest_path
    )
    assert "pre_master_summary" in manifest["mix"]
    assert "pre_export_summary" in manifest["mix"]
    mix_risk_codes = {risk["code"] for risk in manifest["artifact_risk"]["mix"]}
    assert "export_loudness_jump" not in mix_risk_codes


def test_render_piece_snippet_writes_separate_artifacts_and_metadata(
    tmp_path: Path,
) -> None:
    render_window = RenderWindow(start_seconds=0.5, duration_seconds=0.75)

    result = render_piece(
        "chord_4567",
        output_dir=tmp_path,
        save_plot=True,
        render_window=render_window,
    )

    assert result.audio_path.exists()
    assert "__snippet_" in result.audio_path.name
    assert result.version_audio_path is not None
    assert result.version_audio_path.exists()
    assert "__snippet_" in result.version_audio_path.name
    assert result.plot_path is not None
    assert result.plot_path.exists()
    assert result.analysis_manifest_path is not None
    assert result.analysis_manifest_path.exists()
    render_metadata = json.loads(
        result.render_metadata_path.read_text(encoding="utf-8")
    )
    render_request = render_metadata["request"]["render_window"]
    assert render_request["mode"] == "snippet"
    assert render_request["start_seconds"] == pytest.approx(0.5)
    assert render_request["duration_seconds"] == pytest.approx(0.75)
    assert render_request["render_start_seconds"] == pytest.approx(0.0)
    assert render_request["render_end_seconds"] == pytest.approx(2.25)
    assert render_metadata["score_summary"]["note_count"] == 1
    assert "__snippet_" in render_metadata["artifacts"]["latest"]["audio_path"]


def test_render_piece_snippet_rejects_render_audio_piece(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not support snippet rendering"):
        render_piece(
            "interval_demo",
            output_dir=tmp_path,
            render_window=RenderWindow(start_seconds=0.0, duration_seconds=1.0),
        )


def test_render_piece_render_audio_surface_writes_audio_and_analysis(
    tmp_path: Path,
) -> None:
    result = render_piece("interval_demo", output_dir=tmp_path, save_plot=True)
    audio_path, plot_path = result

    assert audio_path.exists()
    assert result.version_audio_path is not None
    assert result.version_audio_path.exists()
    assert plot_path is None
    assert result.analysis_manifest_path is not None
    assert result.analysis_manifest_path.exists()
    manifest = json.loads(result.analysis_manifest_path.read_text(encoding="utf-8"))
    assert Path(manifest["mix"]["artifacts"]["spectrum"]).exists()
    assert result.render_metadata_path is not None
    assert result.render_metadata_path.exists()
    render_metadata = json.loads(
        result.render_metadata_path.read_text(encoding="utf-8")
    )
    assert render_metadata["piece_name"] == "interval_demo"
    assert render_metadata["request"]["save_plot"] is True
    assert render_metadata["request"]["export_target_lufs"] == -18.0
    assert "score_snapshot" not in render_metadata
    assert (
        Path(render_metadata["artifacts"]["latest"]["audio_path"]) == result.audio_path
    )


def test_render_piece_effects_showcase_writes_audio_and_analysis(
    tmp_path: Path,
) -> None:
    result = render_piece("effects_showcase", output_dir=tmp_path, save_plot=True)
    audio_path, plot_path = result

    assert audio_path.exists()
    assert result.version_audio_path is not None
    assert result.version_audio_path.exists()
    assert plot_path is None
    assert result.analysis_manifest_path is not None
    assert result.analysis_manifest_path.exists()
    render_metadata = json.loads(
        result.render_metadata_path.read_text(encoding="utf-8")
    )
    assert render_metadata["piece_name"] == "effects_showcase"
    assert render_metadata["request"]["save_plot"] is True
    assert render_metadata["request"]["export_true_peak_ceiling_dbfs"] == -0.5


def test_piece_registry_definitions_are_complete() -> None:
    output_names = [definition.output_name for definition in PIECES.values()]

    assert output_names
    assert len(output_names) == len(set(output_names))

    for piece_name, definition in PIECES.items():
        assert definition.name == piece_name
        assert bool(definition.build_score) != bool(definition.render_audio)


# ---------------------------------------------------------------------------
# noise_perc: independent noise_decay / pitch_decay
# ---------------------------------------------------------------------------


def test_noise_perc_noise_decay_independent_from_pitch_decay() -> None:
    """Short pitch_decay + long noise_decay should give more noise body than
    using pitch_decay alone (the old behavior)."""
    from code_musics.engines import noise_perc

    sr = 44100
    dur = 0.3
    amp = 0.5
    freq = 3000.0

    # Old-style: pitch_decay=0.004 (short) dominates noise — no body
    short_noise = noise_perc.render(
        freq=freq,
        duration=dur,
        amp=amp,
        sample_rate=sr,
        params={
            "noise_mix": 0.99,
            "pitch_decay": 0.004,
            "tone_decay": 0.003,
            "bandpass_ratio": 0.8,
            "click_amount": 0.06,
        },
    )
    # New-style: noise_decay=0.085 gives proper body while pitch_decay stays short
    long_noise = noise_perc.render(
        freq=freq,
        duration=dur,
        amp=amp,
        sample_rate=sr,
        params={
            "noise_mix": 0.99,
            "pitch_decay": 0.004,
            "noise_decay": 0.085,
            "tone_decay": 0.003,
            "bandpass_ratio": 0.8,
            "click_amount": 0.06,
        },
    )

    # The long_noise version should have more energy in the 50–200 ms window
    # (the noise body) than the short version.
    mid_start = int(0.05 * sr)
    mid_end = int(0.20 * sr)
    short_body_rms = float(np.sqrt(np.mean(short_noise[mid_start:mid_end] ** 2)))
    long_body_rms = float(np.sqrt(np.mean(long_noise[mid_start:mid_end] ** 2)))
    assert long_body_rms > short_body_rms * 2.0, (
        f"Long noise_decay should have significantly more body: "
        f"long={long_body_rms:.6f} short={short_body_rms:.6f}"
    )


def test_noise_perc_noise_decay_defaults_to_pitch_decay() -> None:
    """Omitting noise_decay should give the same result as pitch_decay=noise_decay."""
    from code_musics.engines import noise_perc

    # Note: exact sample equality is not expected because the RNG seed incorporates
    # the params dict, so adding noise_decay changes the seed even when the value
    # matches pitch_decay. We verify behavior equivalence by energy comparison.
    sr = 44100
    dur = 0.25
    pitch_dec = 0.03  # deliberately short so body disappears fast

    without_noise_decay = noise_perc.render(
        freq=300.0,
        duration=dur,
        amp=0.5,
        sample_rate=sr,
        params={
            "noise_mix": 0.95,
            "pitch_decay": pitch_dec,
            "tone_decay": 0.05,
            "bandpass_ratio": 1.0,
            "click_amount": 0.05,
        },
    )
    with_explicit_short = noise_perc.render(
        freq=300.0,
        duration=dur,
        amp=0.5,
        sample_rate=sr,
        params={
            "noise_mix": 0.95,
            "pitch_decay": pitch_dec,
            "noise_decay": pitch_dec,
            "tone_decay": 0.05,
            "bandpass_ratio": 1.0,
            "click_amount": 0.05,
        },
    )
    with_long_noise_decay = noise_perc.render(
        freq=300.0,
        duration=dur,
        amp=0.5,
        sample_rate=sr,
        params={
            "noise_mix": 0.95,
            "pitch_decay": pitch_dec,
            "noise_decay": 0.12,
            "tone_decay": 0.05,
            "bandpass_ratio": 1.0,
            "click_amount": 0.05,
        },
    )

    # Default and explicit-short should both have much less body than long.
    mid_start = int(0.05 * sr)
    mid_end = int(0.18 * sr)
    rms_default = float(np.sqrt(np.mean(without_noise_decay[mid_start:mid_end] ** 2)))
    rms_short = float(np.sqrt(np.mean(with_explicit_short[mid_start:mid_end] ** 2)))
    rms_long = float(np.sqrt(np.mean(with_long_noise_decay[mid_start:mid_end] ** 2)))
    assert rms_long > rms_default * 2.0
    assert rms_long > rms_short * 2.0


# ---------------------------------------------------------------------------
# polyblep: resonance_q parameter
# ---------------------------------------------------------------------------


def test_polyblep_resonance_q_overrides_resonance() -> None:
    """resonance_q=0.707 (Butterworth flat) should produce a less resonant
    filter than resonance=0.5 (~Q=6.4), i.e. no resonant peak."""
    score_q = Score(f0=55.0)
    score_q.add_voice(
        "bass",
        synth_defaults={
            "engine": "polyblep",
            "params": {"waveform": "saw", "cutoff_hz": 800.0, "resonance_q": 0.707},
        },
    )
    score_q.add_note("bass", start=0.0, duration=0.5, freq=110.0, amp_db=-6.0)

    score_res = Score(f0=55.0)
    score_res.add_voice(
        "bass",
        synth_defaults={
            "engine": "polyblep",
            "params": {"waveform": "saw", "cutoff_hz": 800.0, "resonance_q": 6.35},
        },
    )
    score_res.add_note("bass", start=0.0, duration=0.5, freq=110.0, amp_db=-6.0)

    audio_q = score_q.render()
    audio_res = score_res.render()

    # Both should produce finite audio and be different
    assert np.isfinite(audio_q).all()
    assert np.isfinite(audio_res).all()
    assert not np.allclose(audio_q, audio_res)


# ---------------------------------------------------------------------------
# saturation: THD reporting
# ---------------------------------------------------------------------------


def test_saturation_thd_reported_in_analysis() -> None:
    """apply_saturation with return_analysis=True should report thd_pct and
    thd_character in the analysis dict."""
    signal = 0.5 * np.sin(
        np.linspace(0.0, 4.0 * np.pi, synth.SAMPLE_RATE, endpoint=False)
    )
    _, analysis = synth.apply_saturation(
        signal, drive=1.5, mix=0.5, return_analysis=True
    )

    assert isinstance(analysis, dict)
    assert "thd_pct" in analysis
    assert "thd_character" in analysis
    assert isinstance(analysis["thd_pct"], float)
    assert analysis["thd_pct"] >= 0.0
    assert analysis["thd_character"] in {
        "clean",
        "subtle_warmth",
        "warmth",
        "saturation",
        "distortion",
        "fuzz",
    }


def test_saturation_thd_increases_with_drive() -> None:
    """Higher drive should produce higher THD%."""
    signal = 0.3 * np.sin(
        np.linspace(0.0, 4.0 * np.pi, synth.SAMPLE_RATE, endpoint=False)
    )
    _, low_analysis = synth.apply_saturation(
        signal, drive=0.5, mix=0.5, return_analysis=True
    )
    _, high_analysis = synth.apply_saturation(
        signal, drive=3.0, mix=0.5, return_analysis=True
    )

    assert isinstance(low_analysis, dict)
    assert isinstance(high_analysis, dict)
    assert float(high_analysis["thd_pct"]) > float(low_analysis["thd_pct"])


def test_saturation_thd_reported_via_effect_chain() -> None:
    """THD should appear in the effect analysis manifest via apply_effect_chain."""
    signal = 0.5 * np.sin(
        np.linspace(0.0, 4.0 * np.pi, synth.SAMPLE_RATE, endpoint=False)
    )
    _out, effect_analysis = synth.apply_effect_chain(
        signal,
        [EffectSpec("saturation", {"drive": 2.0, "mix": 0.6})],
        return_analysis=True,
    )
    metrics = effect_analysis[0].metrics
    assert "thd_pct" in metrics
    assert "thd_character" in metrics


def test_effect_analysis_includes_signal_thd_metrics() -> None:
    """Every effect stage should report input/output THD and delta."""
    t = np.arange(synth.SAMPLE_RATE, dtype=np.float64) / synth.SAMPLE_RATE
    signal = 0.5 * np.sin(2.0 * np.pi * 220.0 * t)

    _processed, effect_analysis = synth.apply_effect_chain(
        signal,
        [EffectSpec("saturation", {"drive": 2.0, "mix": 0.6})],
        return_analysis=True,
    )

    metrics = effect_analysis[0].metrics
    assert "input_thd_pct" in metrics
    assert "output_thd_pct" in metrics
    assert "thd_delta_pct" in metrics
    assert "input_thd_character" in metrics
    assert "output_thd_character" in metrics
    # A sine input should have very low input THD.
    assert float(metrics["input_thd_pct"]) < 2.0
    # Saturation should raise THD.
    assert float(metrics["output_thd_pct"]) > float(metrics["input_thd_pct"])
    assert float(metrics["thd_delta_pct"]) > 0.0


def test_aggressive_saturation_triggers_thd_warning() -> None:
    """Heavy saturation should trigger the effect_introduced_distortion warning."""
    t = np.arange(synth.SAMPLE_RATE, dtype=np.float64) / synth.SAMPLE_RATE
    signal = 0.7 * np.sin(2.0 * np.pi * 220.0 * t)

    _processed, effect_analysis = synth.apply_effect_chain(
        signal,
        [EffectSpec("saturation", {"drive": 6.0, "mix": 1.0})],
        return_analysis=True,
    )

    warning_codes = {w.code for w in effect_analysis[0].warnings}
    assert "effect_introduced_distortion" in warning_codes


def test_gentle_saturation_does_not_trigger_thd_warning() -> None:
    """Mild saturation should not fire the distortion warning."""
    t = np.arange(synth.SAMPLE_RATE, dtype=np.float64) / synth.SAMPLE_RATE
    signal = 0.3 * np.sin(2.0 * np.pi * 220.0 * t)

    _processed, effect_analysis = synth.apply_effect_chain(
        signal,
        [EffectSpec("saturation", {"drive": 1.2, "mix": 0.3})],
        return_analysis=True,
    )

    warning_codes = {w.code for w in effect_analysis[0].warnings}
    assert "effect_introduced_distortion" not in warning_codes


class TestMasterBusDiagnosticLogging:
    """Verify that the master bus signal chain emits diagnostic log messages."""

    @staticmethod
    def _build_simple_score() -> Score:
        score = Score(f0=55.0)
        score.add_voice("pad", synth_defaults={"engine": "additive"})
        score.add_note("pad", start=0.0, duration=0.5, partial=4, amp_db=-6.0)
        return score

    def test_score_render_logs_pre_master_peak(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        score = self._build_simple_score()
        with caplog.at_level(logging.INFO, logger="code_musics.score"):
            score.render()
        pre_master_messages = [
            r.message for r in caplog.records if "pre-processing" in r.message
        ]
        assert len(pre_master_messages) == 1
        assert "dBFS" in pre_master_messages[0]

    def test_score_render_logs_post_gain_stage_peak(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        score = self._build_simple_score()
        with caplog.at_level(logging.INFO, logger="code_musics.score"):
            score.render()
        post_stage_messages = [
            r.message for r in caplog.records if "auto gain stage" in r.message.lower()
        ]
        assert len(post_stage_messages) == 1
        assert "dBFS" in post_stage_messages[0]

    def test_score_render_logs_ceiling_warning_when_peak_exceeds_threshold(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        score = Score(f0=55.0, auto_master_gain_stage=False)
        score.add_voice(
            "loud", synth_defaults={"engine": "additive"}, normalize_lufs=None
        )
        for i in range(20):
            score.add_note("loud", start=0.0, duration=0.5, partial=4 + i, amp=1.0)
        with caplog.at_level(logging.WARNING, logger="code_musics.score"):
            score.render()
        ceiling_messages = [
            r.message for r in caplog.records if "ceiling activated" in r.message
        ]
        assert len(ceiling_messages) == 1
        assert "attenuated" in ceiling_messages[0]

    def test_score_render_no_ceiling_warning_for_quiet_mix(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        score = self._build_simple_score()
        with caplog.at_level(logging.WARNING, logger="code_musics.score"):
            score.render()
        ceiling_messages = [
            r.message for r in caplog.records if "ceiling activated" in r.message
        ]
        assert len(ceiling_messages) == 0


class TestGainStageDiagnosticLogging:
    """Verify that gain_stage_for_master_bus emits diagnostic log messages."""

    def test_gain_stage_logs_decision(self, caplog: pytest.LogCaptureFixture) -> None:
        signal = 0.1 * np.sin(
            np.linspace(0.0, 2.0 * np.pi * 440.0, synth.SAMPLE_RATE, endpoint=False)
        )
        with caplog.at_level(logging.INFO, logger="code_musics.synth"):
            synth.gain_stage_for_master_bus(signal, sample_rate=synth.SAMPLE_RATE)
        gain_messages = [
            r.message for r in caplog.records if "gain stage" in r.message.lower()
        ]
        assert len(gain_messages) >= 1
        msg = gain_messages[0]
        assert "input LUFS" in msg or "current LUFS" in msg
        assert "applied" in msg.lower()


class TestFinalizeMasterDiagnosticLogging:
    """Verify that finalize_master emits diagnostic log messages."""

    def test_finalize_master_logs_limiter_input_gain(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        signal = 0.3 * np.sin(
            np.linspace(0.0, 2.0 * np.pi * 440.0, synth.SAMPLE_RATE * 2, endpoint=False)
        )
        with caplog.at_level(logging.INFO, logger="code_musics.synth"):
            synth.finalize_master(signal, sample_rate=synth.SAMPLE_RATE)
        limiter_messages = [
            r.message for r in caplog.records if "limiter_input_gain_db" in r.message
        ]
        assert len(limiter_messages) >= 1

    def test_finalize_master_logs_convergence_or_immediate(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        signal = 0.3 * np.sin(
            np.linspace(0.0, 2.0 * np.pi * 440.0, synth.SAMPLE_RATE * 2, endpoint=False)
        )
        with caplog.at_level(logging.INFO, logger="code_musics.synth"):
            result = synth.finalize_master(signal, sample_rate=synth.SAMPLE_RATE)
        # The loop may converge immediately (no iteration messages) or need
        # adjustments (iteration messages). Either is valid — verify we get
        # a final result log regardless.
        final_messages = [
            r.message for r in caplog.records if "Mastering final" in r.message
        ]
        assert len(final_messages) >= 1
        assert np.isfinite(result.integrated_lufs)

    def test_finalize_master_logs_final_result(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        signal = 0.3 * np.sin(
            np.linspace(0.0, 2.0 * np.pi * 440.0, synth.SAMPLE_RATE * 2, endpoint=False)
        )
        with caplog.at_level(logging.INFO, logger="code_musics.synth"):
            synth.finalize_master(signal, sample_rate=synth.SAMPLE_RATE)
        final_messages = [
            r.message for r in caplog.records if "final" in r.message.lower()
        ]
        assert len(final_messages) >= 1
        msg = final_messages[0]
        assert "LUFS" in msg
        assert "true peak" in msg.lower() or "true_peak" in msg.lower()


class TestNativeLimiter:
    """Tests for the native true-peak lookahead limiter."""

    def test_native_limiter_enforces_ceiling(self) -> None:
        """Feed a signal with peaks at +6 dBFS, verify output true peak is at or below threshold."""
        duration_seconds = 0.5
        num_samples = int(synth.SAMPLE_RATE * duration_seconds)
        t = np.linspace(
            0.0, 2.0 * np.pi * 440.0 * duration_seconds, num_samples, endpoint=False
        )
        hot_signal = 2.0 * np.sin(t)
        stereo_hot = np.stack([hot_signal, hot_signal * 0.8])

        threshold_db = -0.5
        limited = synth.apply_native_limiter(stereo_hot, threshold_db=threshold_db)

        true_peak = synth.estimate_true_peak_amplitude(limited, oversample_factor=4)
        true_peak_db = synth.amp_to_db(max(true_peak, 1e-12))
        assert true_peak_db <= threshold_db + 0.1

    def test_native_limiter_transparent_below_threshold(self) -> None:
        """Feed a signal well below threshold, verify output is approximately identical."""
        duration_seconds = 0.5
        num_samples = int(synth.SAMPLE_RATE * duration_seconds)
        t = np.linspace(
            0.0, 2.0 * np.pi * 440.0 * duration_seconds, num_samples, endpoint=False
        )
        quiet_signal = 0.1 * np.sin(t)
        stereo_quiet = np.stack([quiet_signal, quiet_signal * 0.9])

        limited = synth.apply_native_limiter(stereo_quiet, threshold_db=-0.5)

        np.testing.assert_allclose(limited, stereo_quiet, atol=1e-3)

    def test_native_limiter_preserves_shape(self) -> None:
        """Verify mono stays mono, stereo stays stereo, length matches."""
        num_samples = synth.SAMPLE_RATE
        mono_signal = 0.5 * np.sin(
            np.linspace(0.0, 2.0 * np.pi * 440.0, num_samples, endpoint=False)
        )
        stereo_signal = np.stack([mono_signal, mono_signal * 0.8])

        mono_limited = synth.apply_native_limiter(mono_signal)
        stereo_limited = synth.apply_native_limiter(stereo_signal)

        assert mono_limited.ndim == 1
        assert mono_limited.shape[0] == num_samples
        assert stereo_limited.ndim == 2
        assert stereo_limited.shape == (2, num_samples)

    def test_native_limiter_logs_gain_reduction(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify the summary log message appears after processing."""
        num_samples = int(synth.SAMPLE_RATE * 0.5)
        t = np.linspace(0.0, 2.0 * np.pi * 440.0 * 0.5, num_samples, endpoint=False)
        hot_signal = 2.0 * np.sin(t)

        with caplog.at_level(logging.INFO, logger="code_musics.synth"):
            synth.apply_native_limiter(hot_signal, threshold_db=-0.5)

        limiter_messages = [
            r.message for r in caplog.records if "limiter" in r.message.lower()
        ]
        assert len(limiter_messages) >= 1
        msg = limiter_messages[0]
        assert "gain reduction" in msg.lower() or "gr" in msg.lower()


class TestPitchMotionGlideTranslation:
    """Verify PitchMotionSpec translates to correct glide params for instrument engines."""

    @staticmethod
    def _fake_render_voice(
        captured: list[list[dict]],
    ):  # noqa: ANN205
        """Return a callable that captures note dicts passed to surge_xt.render_voice."""

        def fake(
            *, notes: list[dict], total_duration: float, sample_rate: int, params: dict
        ) -> np.ndarray:
            captured.append(notes)
            n_samples = int(total_duration * sample_rate)
            return np.zeros((2, n_samples), dtype=np.float64)

        return fake

    def test_linear_bend_produces_glide_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """linear_bend should swap freq/glide_from so the engine glides TO the target."""
        import code_musics.engines.surge_xt as surge_xt_mod

        captured: list[list[dict]] = []
        monkeypatch.setattr(
            surge_xt_mod, "render_voice", self._fake_render_voice(captured)
        )

        f0 = 100.0
        score = Score(f0=f0, auto_master_gain_stage=False)
        score.add_voice(
            "lead",
            synth_defaults={"engine": "surge_xt"},
            normalize_lufs=None,
        )
        score.add_note(
            "lead",
            start=0.0,
            duration=1.0,
            partial=4,
            pitch_motion=PitchMotionSpec.linear_bend(target_partial=6.0),
        )
        score.render()

        assert len(captured) == 1
        note_dicts = captured[0]
        assert len(note_dicts) == 1
        nd = note_dicts[0]

        note_freq = f0 * 4  # 400 Hz
        target_freq = f0 * 6  # 600 Hz
        assert nd["freq"] == pytest.approx(target_freq)
        assert nd["glide_from"] == pytest.approx(note_freq)
        assert nd["glide_time"] == pytest.approx(1.0)

    def test_ratio_glide_produces_glide_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ratio_glide should compute start/end freqs from ratios."""
        import code_musics.engines.surge_xt as surge_xt_mod

        captured: list[list[dict]] = []
        monkeypatch.setattr(
            surge_xt_mod, "render_voice", self._fake_render_voice(captured)
        )

        f0 = 100.0
        score = Score(f0=f0, auto_master_gain_stage=False)
        score.add_voice(
            "lead",
            synth_defaults={"engine": "surge_xt"},
            normalize_lufs=None,
        )
        score.add_note(
            "lead",
            start=0.0,
            duration=2.0,
            partial=4,
            pitch_motion=PitchMotionSpec.ratio_glide(start_ratio=0.5, end_ratio=1.5),
        )
        score.render()

        assert len(captured) == 1
        note_dicts = captured[0]
        assert len(note_dicts) == 1
        nd = note_dicts[0]

        note_freq = f0 * 4  # 400 Hz
        assert nd["freq"] == pytest.approx(note_freq * 1.5)
        assert nd["glide_from"] == pytest.approx(note_freq * 0.5)
        # glide_duration not set, so defaults to note.duration
        assert nd["glide_time"] == pytest.approx(2.0)
