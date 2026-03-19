"""Score abstraction tests."""

import json
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
from code_musics.score import EffectSpec, NoteEvent, Phrase, Score


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


def test_render_overlapping_voices_returns_audio() -> None:
    score = Score(f0=55.0)
    score.add_note("a", start=0.0, duration=1.0, partial=4, amp=0.3)
    score.add_note("b", start=0.5, duration=1.0, partial=5, amp=0.3)

    audio = score.render()

    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert len(audio) == int(1.8 * score.sample_rate)
    assert np.max(np.abs(audio)) > 0


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
    monkeypatch.setattr(synth, "_load_external_plugin", lambda **_: fake_plugin)
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

    monkeypatch.setattr(synth, "_load_external_plugin", lambda **_: FakePlugin())
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

    monkeypatch.setattr(synth, "_load_external_plugin", lambda **_: FakePlugin())
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
    monkeypatch.setattr(synth, "_load_external_plugin", lambda **_: fake_plugin)
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


def test_add_voice_rejects_non_finite_gain_controls() -> None:
    score = Score(f0=55.0)

    with pytest.raises(ValueError, match="pre_fx_gain_db must be finite"):
        score.add_voice("lead", pre_fx_gain_db=float("inf"))

    with pytest.raises(ValueError, match="mix_db must be finite"):
        score.add_voice("lead", mix_db=float("nan"))


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


def test_finalize_master_raises_when_lsp_limiter_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(synth, "has_external_plugin", lambda plugin_name: False)
    signal = 0.2 * np.sin(
        np.linspace(0.0, 4.0 * np.pi, synth.SAMPLE_RATE, endpoint=False)
    )

    with pytest.raises(FileNotFoundError, match="LSP limiter"):
        synth.finalize_master(signal, sample_rate=synth.SAMPLE_RATE)


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


def test_saturation_effect_analysis_reports_shaper_activity() -> None:
    signal = 0.7 * np.sin(
        np.linspace(0.0, 10.0 * np.pi, synth.SAMPLE_RATE, endpoint=False)
    )

    _processed, effect_analysis = synth.apply_effect_chain(
        signal,
        [EffectSpec("saturation", {"drive": 4.0, "mix": 0.9, "bias": 0.2})],
        return_analysis=True,
    )

    saturation_metrics = effect_analysis[0].metrics
    assert saturation_metrics["shaper_hot_fraction"] > 0.0
    assert "crest_factor_delta_db" in saturation_metrics


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
    assert "versions/chord_4567/" in str(result.version_audio_path)
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
