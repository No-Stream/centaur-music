"""Engine registry and score integration tests."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.automation import AutomationSegment, AutomationSpec, AutomationTarget
from code_musics.engines.registry import (
    normalize_synth_spec,
    render_note_signal,
    resolve_synth_params,
)
from code_musics.humanize import VelocityHumanizeSpec
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import EffectSpec, Score, VelocityParamMap
from code_musics.spectra import harmonic_spectrum, ratio_spectrum


def _band_energy(signal: np.ndarray, *, sample_rate: int, min_hz: float) -> float:
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    return float(np.sum(spectrum[freqs >= min_hz]))


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


def test_normalize_synth_spec_supports_structured_env_and_params_aliases() -> None:
    normalized = normalize_synth_spec(
        {
            "engine": "polyblep",
            "env": {
                "attack_ms": 30.0,
                "decay_ms": 180.0,
                "sustain_ratio": 0.62,
                "release_ms": 280.0,
            },
            "params": {
                "cutoff_hz": 2200.0,
                "resonance_ratio": 0.12,
                "filter_env_depth_ratio": 0.7,
                "filter_env_decay_ms": 240.0,
                "filter_drive_ratio": 0.18,
            },
        }
    )

    assert normalized["engine"] == "polyblep"
    assert normalized["attack"] == pytest.approx(0.03)
    assert normalized["decay"] == pytest.approx(0.18)
    assert normalized["sustain_level"] == pytest.approx(0.62)
    assert normalized["release"] == pytest.approx(0.28)
    assert normalized["cutoff_hz"] == pytest.approx(2200.0)
    assert normalized["resonance"] == pytest.approx(0.12)
    assert normalized["filter_env_amount"] == pytest.approx(0.7)
    assert normalized["filter_env_decay"] == pytest.approx(0.24)
    assert normalized["filter_drive"] == pytest.approx(0.18)


def test_new_unit_bearing_aliases_override_legacy_flat_names() -> None:
    normalized = normalize_synth_spec(
        {
            "attack": 0.4,
            "attack_ms": 25.0,
            "sustain_level": 0.9,
            "sustain_ratio": 0.55,
            "filter_env_decay": 0.8,
            "filter_env_decay_ms": 120.0,
        }
    )

    assert normalized["attack"] == pytest.approx(0.025)
    assert normalized["sustain_level"] == pytest.approx(0.55)
    assert normalized["filter_env_decay"] == pytest.approx(0.12)


@pytest.mark.parametrize(
    ("engine_name", "preset_name"),
    [
        ("additive", "organ"),
        ("fm", "dx_piano"),
        ("fm", "lately_bass"),
        ("fm", "fm_clav"),
        ("fm", "fm_mallet"),
        ("fm", "chorused_ep"),
        ("filtered_stack", "saw_pad"),
        ("filtered_stack", "string_pad"),
        ("kick_tom", "909_techno"),
        ("kick_tom", "round_tom"),
        ("polyblep", "synth_pluck"),
        ("polyblep", "analog_brass"),
        ("polyblep", "square_lead"),
        ("polyblep", "hoover"),
        ("polyblep", "moog_bass"),
        ("polyblep", "sync_lead"),
        ("polyblep", "acid_bass"),
        ("polyblep", "sub_bass"),
        ("polyblep", "resonant_sweep"),
        ("polyblep", "soft_square_pad"),
    ],
)
def test_new_presets_resolve_and_render(
    engine_name: str,
    preset_name: str,
) -> None:
    resolved = resolve_synth_params({"engine": engine_name, "preset": preset_name})

    assert resolved["engine"] == engine_name
    assert "preset" not in resolved

    signal = render_note_signal(
        freq=220.0,
        duration=0.35,
        amp=0.3,
        sample_rate=44100,
        params={"engine": engine_name, "preset": preset_name},
    )

    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(signal)) > 0.0


@pytest.mark.parametrize(
    "preset_name",
    ["ji_fusion_pad", "septimal_reed", "eleven_limit_glass", "utonal_drone"],
)
def test_new_spectral_additive_presets_resolve_and_render(preset_name: str) -> None:
    resolved = resolve_synth_params({"engine": "additive", "preset": preset_name})

    assert resolved["engine"] == "additive"
    assert "partials" in resolved

    signal = render_note_signal(
        freq=220.0,
        duration=0.45,
        amp=0.3,
        sample_rate=44100,
        params={"engine": "additive", "preset": preset_name},
    )

    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(signal)) > 0.0


def test_additive_explicit_partials_override_harmonic_stack_controls() -> None:
    explicit = render_note_signal(
        freq=220.0,
        duration=0.4,
        amp=0.25,
        sample_rate=44100,
        params={
            "engine": "additive",
            "n_harmonics": 12,
            "harmonic_rolloff": 0.9,
            "partials": [{"ratio": 1.0, "amp": 1.0}],
        },
    )
    harmonic = render_note_signal(
        freq=220.0,
        duration=0.4,
        amp=0.25,
        sample_rate=44100,
        params={
            "engine": "additive",
            "n_harmonics": 12,
            "harmonic_rolloff": 0.9,
        },
    )

    assert _band_energy(explicit, sample_rate=44100, min_hz=800.0) < _band_energy(
        harmonic, sample_rate=44100, min_hz=800.0
    )


def test_additive_rejects_invalid_spectral_partial_lists() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        render_note_signal(
            freq=220.0,
            duration=0.4,
            amp=0.25,
            sample_rate=44100,
            params={
                "engine": "additive",
                "partials": [{"ratio": 0.0, "amp": 1.0}],
            },
        )


def test_attack_partials_and_morph_time_make_the_onset_brighter() -> None:
    signal = render_note_signal(
        freq=220.0,
        duration=0.6,
        amp=0.25,
        sample_rate=44100,
        params={
            "engine": "additive",
            "partials": ratio_spectrum([1.0, 3 / 2], [1.0, 0.18]),
            "attack_partials": ratio_spectrum([1.0, 3 / 2, 11 / 4], [1.0, 0.18, 0.45]),
            "spectral_morph_time": 0.12,
        },
    )

    attack_slice = signal[: int(0.12 * 44100)]
    sustain_slice = signal[-int(0.12 * 44100) :]
    assert _band_energy(attack_slice, sample_rate=44100, min_hz=500.0) > _band_energy(
        sustain_slice, sample_rate=44100, min_hz=500.0
    )


def test_partial_decay_tilt_reduces_upper_band_energy_later_in_note() -> None:
    signal = render_note_signal(
        freq=220.0,
        duration=0.8,
        amp=0.25,
        sample_rate=44100,
        params={
            "engine": "additive",
            "partials": harmonic_spectrum(n_partials=6, harmonic_rolloff=0.8),
            "partial_decay_tilt": 0.8,
        },
    )

    early_slice = signal[: int(0.2 * 44100)]
    late_slice = signal[-int(0.2 * 44100) :]
    assert _band_energy(early_slice, sample_rate=44100, min_hz=1200.0) > _band_energy(
        late_slice, sample_rate=44100, min_hz=1200.0
    )


def test_upper_partial_drift_is_deterministic_across_repeated_renders() -> None:
    params = {
        "engine": "additive",
        "partials": ratio_spectrum(
            [1.0, 11 / 8, 3 / 2, 11 / 4], [1.0, 0.3, 0.22, 0.12]
        ),
        "upper_partial_drift_cents": 4.0,
        "upper_partial_drift_min_ratio": 1.3,
    }

    first = render_note_signal(
        freq=220.0,
        duration=0.5,
        amp=0.25,
        sample_rate=44100,
        params=params,
    )
    second = render_note_signal(
        freq=220.0,
        duration=0.5,
        amp=0.25,
        sample_rate=44100,
        params=params,
    )
    static = render_note_signal(
        freq=220.0,
        duration=0.5,
        amp=0.25,
        sample_rate=44100,
        params={
            "engine": "additive",
            "partials": params["partials"],
        },
    )

    assert np.allclose(first, second)
    assert not np.allclose(first, static)


def test_score_can_render_explicit_spectral_additive_voice() -> None:
    score = Score(f0=110.0)
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "additive",
            "params": {
                "partials": ratio_spectrum([1.0, 7 / 4, 11 / 8], [1.0, 0.28, 0.16]),
                "attack_partials": ratio_spectrum(
                    [1.0, 7 / 4, 11 / 8, 11 / 4],
                    [1.0, 0.28, 0.16, 0.1],
                ),
                "spectral_morph_time": 0.15,
                "partial_decay_tilt": 0.3,
            },
            "env": {"attack_ms": 80.0, "release_ms": 600.0},
        },
    )
    score.add_note("pad", start=0.0, duration=0.7, partial=2.0, amp=0.2)
    score.add_note("pad", start=0.8, duration=0.6, partial=7 / 4, amp=0.18)

    audio = score.render()

    assert audio.ndim == 1
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) > 0.0


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


def test_structured_synth_spec_merges_voice_defaults_with_note_overrides() -> None:
    score = Score(f0=110.0)
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "polyblep",
            "preset": "warm_lead",
            "env": {"attack_ms": 80.0, "release_ms": 500.0},
            "params": {"cutoff_hz": 900.0, "filter_env_decay_ms": 400.0},
        },
    )
    score.add_note(
        "lead",
        start=0.0,
        duration=0.35,
        partial=2.0,
        amp=0.25,
        synth={
            "env": {"release_ms": 120.0},
            "params": {"cutoff_hz": 1800.0},
        },
    )

    audio = score.render()

    assert audio.ndim == 1
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) > 0.0


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
    score.add_voice(
        "kick", synth_defaults={"engine": "kick_tom", "preset": "909_house"}
    )

    score.add_voice(
        "lead", synth_defaults={"engine": "polyblep", "preset": "warm_lead"}
    )

    score.add_note("pad", start=0.0, duration=0.8, partial=2.0, amp=0.2)
    score.add_note("bell", start=0.15, duration=0.5, partial=3.0, amp=0.2)
    score.add_note("bass", start=0.0, duration=0.6, partial=1.0, amp=0.2)
    score.add_note("perc", start=0.35, duration=0.2, freq=180.0, amp=0.2)
    score.add_note("kick", start=0.7, duration=0.4, freq=58.0, amp=0.2)
    score.add_note("lead", start=0.1, duration=0.5, partial=2.0, amp=0.2)

    audio = score.render()

    assert audio.ndim == 1
    assert np.all(np.isfinite(audio))
    assert len(audio) == int(2.0 * score.sample_rate)
    assert np.max(np.abs(audio)) > 0.0


def test_normalize_synth_spec_supports_kick_tom_param_aliases() -> None:
    normalized = normalize_synth_spec(
        {
            "engine": "kick_tom",
            "params": {
                "body_decay_ms": 420.0,
                "sweep_amount_ratio": 1.8,
                "sweep_decay_ms": 66.0,
                "body_punch": 0.35,
                "body_tone": 0.2,
                "drive": 0.28,
            },
        }
    )

    assert normalized["body_decay_ms"] == pytest.approx(420.0)
    assert normalized["pitch_sweep_amount_ratio"] == pytest.approx(1.8)
    assert normalized["pitch_sweep_decay_ms"] == pytest.approx(66.0)
    assert normalized["body_punch_ratio"] == pytest.approx(0.35)
    assert normalized["body_tone_ratio"] == pytest.approx(0.2)
    assert normalized["drive_ratio"] == pytest.approx(0.28)


def test_score_can_render_kick_tom_with_finished_effect_chain() -> None:
    score = Score(f0=110.0)
    score.add_voice(
        "kick",
        synth_defaults={"engine": "kick_tom", "preset": "909_techno"},
        velocity_humanize=None,
        velocity_to_params={
            "click_amount": VelocityParamMap(min_value=0.12, max_value=0.28),
            "drive_ratio": VelocityParamMap(min_value=0.14, max_value=0.32),
            "pitch_sweep_amount_ratio": VelocityParamMap(
                min_value=2.3,
                max_value=3.1,
            ),
        },
        effects=[
            EffectSpec("compressor", {"preset": "kick_punch"}),
            EffectSpec("saturation", {"preset": "kick_weight"}),
        ],
    )
    score.add_note("kick", start=0.0, duration=0.35, freq=55.0, amp=0.22, velocity=0.9)
    score.add_note(
        "kick", start=0.55, duration=0.35, freq=55.0, amp=0.22, velocity=1.15
    )

    audio = score.render()

    assert audio.ndim == 1
    assert np.all(np.isfinite(audio))
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


def test_pitch_motion_renders_through_polyblep_engine() -> None:
    static_score = Score(f0=110.0)
    static_score.add_voice(
        "lead", synth_defaults={"engine": "polyblep", "preset": "warm_lead"}
    )
    static_score.add_note("lead", start=0.0, duration=0.45, partial=2.0, amp=0.25)

    motion_score = Score(f0=110.0)
    motion_score.add_voice(
        "lead", synth_defaults={"engine": "polyblep", "preset": "warm_lead"}
    )
    motion_score.add_note(
        "lead",
        start=0.0,
        duration=0.45,
        partial=2.0,
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


def test_voice_automation_changes_note_start_synth_params() -> None:
    static_score = Score(f0=110.0)
    static_score.add_voice(
        "lead",
        synth_defaults={"engine": "filtered_stack", "preset": "round_bass"},
        velocity_humanize=None,
    )
    static_score.add_note("lead", start=0.0, duration=0.5, partial=1.0, amp=0.25)

    automated_score = Score(f0=110.0)
    automated_score.add_voice(
        "lead",
        synth_defaults={"engine": "filtered_stack", "preset": "round_bass"},
        velocity_humanize=None,
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="cutoff_hz"),
                segments=(
                    AutomationSegment(
                        start=0.0,
                        end=1.0,
                        shape="hold",
                        value=1800.0,
                    ),
                ),
            )
        ],
    )
    automated_score.add_note("lead", start=0.0, duration=0.5, partial=1.0, amp=0.25)

    static_audio = static_score.render()
    automated_audio = automated_score.render()

    assert static_audio.shape == automated_audio.shape
    assert np.max(np.abs(static_audio - automated_audio)) > 1e-4


def test_note_automation_overrides_voice_automation_on_same_target() -> None:
    score = Score(f0=110.0)
    score.add_voice(
        "lead",
        synth_defaults={"engine": "filtered_stack", "preset": "round_bass"},
        velocity_humanize=None,
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="cutoff_hz"),
                segments=(
                    AutomationSegment(start=0.0, end=2.0, shape="hold", value=300.0),
                ),
            )
        ],
    )
    score.add_note(
        "lead",
        start=0.0,
        duration=0.5,
        partial=1.0,
        amp=0.25,
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="cutoff_hz"),
                segments=(
                    AutomationSegment(start=0.0, end=0.5, shape="hold", value=1800.0),
                ),
            )
        ],
    )

    overridden_audio = score.render()

    reference = Score(f0=110.0)
    reference.add_voice(
        "lead",
        synth_defaults={
            "engine": "filtered_stack",
            "preset": "round_bass",
            "cutoff_hz": 1800.0,
        },
        velocity_humanize=None,
    )
    reference.add_note("lead", start=0.0, duration=0.5, partial=1.0, amp=0.25)
    reference_audio = reference.render()

    assert overridden_audio.shape == reference_audio.shape
    assert np.allclose(overridden_audio, reference_audio)


def test_pitch_ratio_automation_renders_per_sample_pitch_motion() -> None:
    static_score = Score(f0=110.0)
    static_score.add_voice(
        "lead", synth_defaults={"engine": "additive", "preset": "bright_pluck"}
    )
    static_score.add_note("lead", start=0.0, duration=0.5, partial=2.0, amp=0.25)

    automated_score = Score(f0=110.0)
    automated_score.add_voice(
        "lead", synth_defaults={"engine": "additive", "preset": "bright_pluck"}
    )
    automated_score.add_note(
        "lead",
        start=0.0,
        duration=0.5,
        partial=2.0,
        amp=0.25,
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="pitch_ratio", name="pitch_ratio"),
                segments=(
                    AutomationSegment(
                        start=0.0,
                        end=0.5,
                        shape="linear",
                        start_value=1.0,
                        end_value=1.5,
                    ),
                ),
            )
        ],
    )

    static_audio = static_score.render()
    automated_audio = automated_score.render()

    assert static_audio.shape == automated_audio.shape
    assert np.max(np.abs(static_audio - automated_audio)) > 1e-4


def test_pitch_ratio_automation_conflicts_with_pitch_motion() -> None:
    score = Score(f0=110.0)
    score.add_voice(
        "lead", synth_defaults={"engine": "additive", "preset": "bright_pluck"}
    )
    score.add_note(
        "lead",
        start=0.0,
        duration=0.5,
        partial=2.0,
        amp=0.25,
        pitch_motion=PitchMotionSpec.linear_bend(target_partial=3.0),
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="pitch_ratio", name="pitch_ratio"),
                segments=(
                    AutomationSegment(
                        start=0.0,
                        end=0.5,
                        shape="hold",
                        value=1.0,
                    ),
                ),
            )
        ],
    )

    with pytest.raises(ValueError, match="pitch_ratio automation"):
        score.render()


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
    soft = Score(f0=110.0, auto_master_gain_stage=False)
    loud = Score(f0=110.0, auto_master_gain_stage=False)
    for score in (soft, loud):
        score.add_voice(
            "lead",
            synth_defaults={"engine": "additive", "preset": "bright_pluck"},
            velocity_humanize=None,
            normalize_lufs=None,
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
