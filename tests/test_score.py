"""Score abstraction tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from code_musics import synth
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
from code_musics.render import render_piece
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
    assert audio.shape[1] == int(1.35 * score.sample_rate)


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
    assert Path(manifest["mix"]["artifacts"]["spectrum"]).exists()
    render_metadata = json.loads(
        result.render_metadata_path.read_text(encoding="utf-8")
    )
    assert render_metadata["piece_name"] == "chord_4567"
    assert render_metadata["request"]["save_plot"] is True
    assert render_metadata["score_summary"]["note_count"] == 4
    assert render_metadata["score_summary"]["voice_names"] == ["chord"]
    assert render_metadata["score_snapshot"]["voices"]["chord"]["notes"]
    assert (
        Path(render_metadata["artifacts"]["versioned"]["audio_path"])
        == result.version_audio_path
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


def test_piece_registry_definitions_are_complete() -> None:
    output_names = [definition.output_name for definition in PIECES.values()]

    assert output_names
    assert len(output_names) == len(set(output_names))

    for piece_name, definition in PIECES.items():
        assert definition.name == piece_name
        assert bool(definition.build_score) != bool(definition.render_audio)
