"""Engine registry and score integration tests."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines.registry import render_note_signal, resolve_synth_params
from code_musics.humanize import VelocityHumanizeSpec
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import Score, VelocityParamMap


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
    assert len(audio) == int(1.6 * score.sample_rate)
    assert np.max(np.abs(audio)) > 0


def test_score_can_mix_multiple_engine_types() -> None:
    score = Score(f0=110.0)
    score.add_voice("pad", synth_defaults={"engine": "additive", "preset": "soft_pad"})
    score.add_voice("bell", synth_defaults={"engine": "fm", "preset": "bell"})
    score.add_voice(
        "bass", synth_defaults={"engine": "filtered_stack", "preset": "round_bass"}
    )
    score.add_voice(
        "perc", synth_defaults={"engine": "noise_perc", "preset": "snareish"}
    )

    score.add_note("pad", start=0.0, duration=0.8, partial=2.0, amp=0.2)
    score.add_note("bell", start=0.15, duration=0.5, partial=3.0, amp=0.2)
    score.add_note("bass", start=0.0, duration=0.6, partial=1.0, amp=0.2)
    score.add_note("perc", start=0.35, duration=0.2, freq=180.0, amp=0.2)

    audio = score.render()

    assert audio.ndim == 1
    assert np.all(np.isfinite(audio))
    assert len(audio) == int(2.0 * score.sample_rate)
    assert np.max(np.abs(audio)) > 0.0


def test_pitch_motion_renders_through_additive_engine() -> None:
    static_score = Score(f0=110.0)
    static_score.add_voice(
        "lead", synth_defaults={"engine": "additive", "preset": "bright_pluck"}
    )
    static_score.add_note("lead", start=0.0, duration=0.4, partial=2.0, amp=0.25)

    motion_score = Score(f0=110.0)
    motion_score.add_voice(
        "lead", synth_defaults={"engine": "additive", "preset": "bright_pluck"}
    )
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
    static_score.add_voice(
        "lead", synth_defaults={"engine": "fm", "preset": "glass_lead"}
    )
    static_score.add_note("lead", start=0.0, duration=0.5, partial=2.0, amp=0.25)

    motion_score = Score(f0=110.0)
    motion_score.add_voice(
        "lead", synth_defaults={"engine": "fm", "preset": "glass_lead"}
    )
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
    static_score.add_voice(
        "bass", synth_defaults={"engine": "filtered_stack", "preset": "round_bass"}
    )
    static_score.add_note("bass", start=0.0, duration=0.45, partial=1.0, amp=0.25)

    motion_score = Score(f0=110.0)
    motion_score.add_voice(
        "bass", synth_defaults={"engine": "filtered_stack", "preset": "round_bass"}
    )
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
        synth_defaults={
            "engine": "additive",
            "preset": "bright_pluck",
            "attack": 0.08,
            "release": 0.15,
        },
    )
    neutral.add_note("lead", start=0.0, duration=0.5, partial=2.0, amp=0.25)

    shaped = Score(f0=110.0)
    shaped.add_voice(
        "lead",
        synth_defaults={
            "engine": "additive",
            "preset": "bright_pluck",
            "attack": 0.08,
            "release": 0.15,
        },
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

    assert np.all(np.isfinite(shaped_audio))
    min_len = min(len(neutral_audio), len(shaped_audio))
    assert not np.allclose(neutral_audio[:min_len], shaped_audio[:min_len])


def test_velocity_changes_rendered_loudness_on_db_scale() -> None:
    soft = Score(f0=110.0)
    loud = Score(f0=110.0)
    for score in (soft, loud):
        score.add_voice(
            "lead",
            synth_defaults={"engine": "additive", "preset": "bright_pluck"},
            velocity_humanize=None,
        )

    soft.add_note(
        "lead",
        start=0.0,
        duration=0.5,
        partial=2.0,
        amp_db=-18.0,
        velocity=0.8,
    )
    loud.add_note(
        "lead",
        start=0.0,
        duration=0.5,
        partial=2.0,
        amp_db=-18.0,
        velocity=1.2,
    )

    soft_audio = soft.render()
    loud_audio = loud.render()

    assert soft_audio.shape == loud_audio.shape
    assert np.max(np.abs(loud_audio)) > np.max(np.abs(soft_audio))


def test_velocity_can_modulate_synth_parameters() -> None:
    dark = Score(f0=110.0)
    bright = Score(f0=110.0)
    for score in (dark, bright):
        score.add_voice(
            "lead",
            synth_defaults={"engine": "filtered_stack", "preset": "round_bass"},
            velocity_humanize=None,
            velocity_to_params={
                "cutoff_hz": VelocityParamMap(
                    min_value=250.0,
                    max_value=1_600.0,
                    min_velocity=0.8,
                    max_velocity=1.2,
                )
            },
        )

    dark.add_note("lead", start=0.0, duration=0.5, partial=1.0, amp=0.25, velocity=0.8)
    bright.add_note(
        "lead",
        start=0.0,
        duration=0.5,
        partial=1.0,
        amp=0.25,
        velocity=1.2,
    )

    dark_audio = dark.render()
    bright_audio = bright.render()

    assert dark_audio.shape == bright_audio.shape
    assert np.all(np.isfinite(bright_audio))
    assert not np.allclose(dark_audio, bright_audio)


def test_velocity_group_humanize_correlates_across_voices() -> None:
    score = Score(f0=110.0)
    shared_velocity = VelocityHumanizeSpec(
        seed=5,
        group_amount=0.08,
        follow_strength=0.96,
        voice_spread=0.02,
        note_jitter=0.0,
        chord_spread=0.0,
        min_multiplier=0.85,
        max_multiplier=1.15,
    )
    for voice_name in ("lead", "alto"):
        score.add_voice(
            voice_name,
            synth_defaults={"engine": "additive", "preset": "bright_pluck"},
            velocity_humanize=shared_velocity,
            velocity_group="ensemble",
        )
        for index, start in enumerate(np.linspace(0.0, 7.0, 8)):
            score.add_note(
                voice_name,
                start=float(start),
                duration=0.4,
                partial=(2.0 + index) if voice_name == "lead" else (3.0 + index),
                amp=0.2,
            )

    multipliers = score._build_velocity_multiplier_map()
    lead = np.asarray([multipliers[("lead", index)] for index in range(8)])
    alto = np.asarray([multipliers[("alto", index)] for index in range(8)])

    assert np.corrcoef(lead, alto)[0, 1] > 0.9
