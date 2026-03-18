"""Engine registry and score integration tests."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines.registry import render_note_signal, resolve_synth_params
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import Score


def test_unknown_engine_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported synth engine"):
        render_note_signal(
            freq=220.0,
            duration=0.5,
            amp=0.3,
            sample_rate=44100,
            params={"engine": "nope"},
        )


def test_resolve_synth_params_applies_preset_before_explicit_overrides() -> None:
    resolved = resolve_synth_params(
        {
            "engine": "additive",
            "preset": "bright_pluck",
            "attack": 0.2,
            "n_harmonics": 11,
        }
    )

    assert resolved["engine"] == "additive"
    assert resolved["attack"] == 0.2
    assert resolved["n_harmonics"] == 11
    assert "preset" not in resolved


def test_voice_level_engine_and_note_override_both_render() -> None:
    score = Score(f0=110.0)
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "additive",
            "preset": "soft_pad",
            "attack": 0.01,
        },
    )
    score.add_note("lead", start=0.0, duration=0.4, partial=2.0, amp=0.25)
    score.add_note(
        "lead",
        start=0.5,
        duration=0.3,
        partial=3.0,
        amp=0.25,
        synth={"engine": "additive", "preset": "bright_pluck", "attack": 0.02},
    )

    audio = score.render()

    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert np.all(np.isfinite(audio))
    assert len(audio) == int(0.8 * score.sample_rate)
    assert np.max(np.abs(audio)) > 0


def test_score_can_mix_multiple_engine_types() -> None:
    score = Score(f0=110.0)
    score.add_voice("pad", synth_defaults={"engine": "additive", "preset": "soft_pad"})
    score.add_voice("bell", synth_defaults={"engine": "fm", "preset": "bell"})
    score.add_voice("bass", synth_defaults={"engine": "filtered_stack", "preset": "round_bass"})
    score.add_voice("perc", synth_defaults={"engine": "noise_perc", "preset": "snareish"})

    score.add_note("pad", start=0.0, duration=0.8, partial=2.0, amp=0.2)
    score.add_note("bell", start=0.15, duration=0.5, partial=3.0, amp=0.2)
    score.add_note("bass", start=0.0, duration=0.6, partial=1.0, amp=0.2)
    score.add_note("perc", start=0.35, duration=0.2, freq=180.0, amp=0.2)

    audio = score.render()

    assert audio.ndim == 1
    assert np.all(np.isfinite(audio))
    assert len(audio) == int(0.8 * score.sample_rate)
    assert np.max(np.abs(audio)) > 0.0


def test_pitch_motion_renders_through_additive_engine() -> None:
    static_score = Score(f0=110.0)
    static_score.add_voice("lead", synth_defaults={"engine": "additive", "preset": "bright_pluck"})
    static_score.add_note("lead", start=0.0, duration=0.4, partial=2.0, amp=0.25)

    motion_score = Score(f0=110.0)
    motion_score.add_voice("lead", synth_defaults={"engine": "additive", "preset": "bright_pluck"})
    motion_score.add_note(
        "lead",
        start=0.0,
        duration=0.4,
        partial=2.0,
        amp=0.25,
        pitch_motion=PitchMotionSpec.linear_bend(target_partial=3.0),
    )

    static_audio = static_score.render()
    motion_audio = motion_score.render()

    assert static_audio.shape == motion_audio.shape
    assert np.all(np.isfinite(motion_audio))
    assert np.max(np.abs(motion_audio - static_audio)) > 1e-4


def test_pitch_motion_renders_through_fm_engine() -> None:
    static_score = Score(f0=110.0)
    static_score.add_voice("lead", synth_defaults={"engine": "fm", "preset": "glass_lead"})
    static_score.add_note("lead", start=0.0, duration=0.5, partial=2.0, amp=0.25)

    motion_score = Score(f0=110.0)
    motion_score.add_voice("lead", synth_defaults={"engine": "fm", "preset": "glass_lead"})
    motion_score.add_note(
        "lead",
        start=0.0,
        duration=0.5,
        partial=2.0,
        amp=0.25,
        pitch_motion=PitchMotionSpec.ratio_glide(end_ratio=3 / 2),
    )

    static_audio = static_score.render()
    motion_audio = motion_score.render()

    assert static_audio.shape == motion_audio.shape
    assert np.all(np.isfinite(motion_audio))
    assert np.max(np.abs(motion_audio - static_audio)) > 1e-4


def test_pitch_motion_renders_through_filtered_stack_engine() -> None:
    static_score = Score(f0=110.0)
    static_score.add_voice("bass", synth_defaults={"engine": "filtered_stack", "preset": "round_bass"})
    static_score.add_note("bass", start=0.0, duration=0.45, partial=1.0, amp=0.25)

    motion_score = Score(f0=110.0)
    motion_score.add_voice("bass", synth_defaults={"engine": "filtered_stack", "preset": "round_bass"})
    motion_score.add_note(
        "bass",
        start=0.0,
        duration=0.45,
        partial=1.0,
        amp=0.25,
        pitch_motion=PitchMotionSpec.vibrato(depth_ratio=0.02, rate_hz=5.0),
    )

    static_audio = static_score.render()
    motion_audio = motion_score.render()

    assert static_audio.shape == motion_audio.shape
    assert np.all(np.isfinite(motion_audio))
    assert np.max(np.abs(motion_audio - static_audio)) > 1e-4


def test_pitch_motion_is_rejected_for_noise_perc() -> None:
    with pytest.raises(ValueError, match="pitch motion is not supported"):
        render_note_signal(
            freq=180.0,
            duration=0.2,
            amp=0.3,
            sample_rate=44100,
            params={"engine": "noise_perc", "preset": "snareish"},
            freq_trajectory=np.array([180.0, 181.0]),
        )


def test_attack_and_release_scales_change_rendered_envelope() -> None:
    neutral = Score(f0=110.0)
    neutral.add_voice(
        "lead",
        synth_defaults={"engine": "additive", "preset": "bright_pluck", "attack": 0.08, "release": 0.15},
    )
    neutral.add_note("lead", start=0.0, duration=0.5, partial=2.0, amp=0.25)

    shaped = Score(f0=110.0)
    shaped.add_voice(
        "lead",
        synth_defaults={"engine": "additive", "preset": "bright_pluck", "attack": 0.08, "release": 0.15},
    )
    shaped.add_note(
        "lead",
        start=0.0,
        duration=0.5,
        partial=2.0,
        amp=0.25,
        synth={"attack_scale": 2.0, "release_scale": 0.4},
    )

    neutral_audio = neutral.render()
    shaped_audio = shaped.render()

    assert neutral_audio.shape == shaped_audio.shape
    assert np.all(np.isfinite(shaped_audio))
    assert not np.allclose(neutral_audio, shaped_audio)
