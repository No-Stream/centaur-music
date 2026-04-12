"""Analysis helper tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from code_musics.analysis import (
    analyze_audio,
    analyze_score,
    build_score_timeline,
    compare_analysis_manifests,
    save_analysis_artifacts,
)
from code_musics.humanize import TimingHumanizeSpec
from code_musics.pieces.registry import PieceSection
from code_musics.score import EffectSpec, Score, VelocityParamMap, VoiceSend


def test_analyze_audio_reports_expected_band_bias() -> None:
    sample_rate = 44_100
    duration = 1.0
    time = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    low_heavy_signal = 0.9 * np.sin(2.0 * np.pi * 80.0 * time) + 0.1 * np.sin(
        2.0 * np.pi * 4_000.0 * time
    )

    analysis = analyze_audio(low_heavy_signal, sample_rate=sample_rate)

    assert analysis.duration_seconds == 1.0
    assert analysis.band_energy_db["bass"] > analysis.band_energy_db["high"]
    assert analysis.low_high_balance_db > 0
    assert np.isfinite(analysis.integrated_lufs)
    assert analysis.gated_rms_dbfs >= analysis.rms_dbfs


def test_analyze_audio_warns_for_clipping_and_low_active_level() -> None:
    sample_rate = 44_100
    clipped_signal = np.array([0.0, 1.1, -1.05, 0.2], dtype=np.float64)
    clipped_analysis = analyze_audio(clipped_signal, sample_rate=sample_rate)

    assert clipped_analysis.clipped_sample_count == 2
    assert "sample peak clipping detected" in clipped_analysis.warnings

    quiet_signal = np.concatenate(
        [
            np.zeros(sample_rate, dtype=np.float64),
            np.full(sample_rate, 0.008, dtype=np.float64),
            np.zeros(sample_rate, dtype=np.float64),
        ]
    )
    quiet_analysis = analyze_audio(quiet_signal, sample_rate=sample_rate)

    assert quiet_analysis.active_window_fraction > 0.0
    assert "active passages are very quiet overall" in quiet_analysis.warnings


def test_analyze_audio_reports_bright_modulated_artifact_risks() -> None:
    sample_rate = 44_100
    duration = 4.0
    time = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    # Power-2 modulation at 8 Hz produces ~23 dB depth, clearing the 18 dB
    # warning threshold and the 4 Hz frequency floor.
    modulation = np.power(
        np.maximum(0.001, 0.5 + 0.5 * np.sin(2.0 * np.pi * 8.0 * time)), 2.0
    )
    signal = 0.8 * modulation * np.sin(2.0 * np.pi * 5_200.0 * time)

    analysis = analyze_audio(signal, sample_rate=sample_rate)
    risk_codes = {risk.code for risk in analysis.artifact_risks}

    assert "bright_spectral_centroid" in risk_codes
    assert "strong_amplitude_modulation" in risk_codes
    assert analysis.amplitude_modulation_depth_db >= 18.0
    assert analysis.dominant_amplitude_modulation_hz >= 7.0
    amplitude_modulation_risk = next(
        risk
        for risk in analysis.artifact_risks
        if risk.code == "strong_amplitude_modulation"
    )
    assert "Hz" in amplitude_modulation_risk.message


def test_analyze_audio_does_not_flag_slow_phrase_shape_as_modulation_artifact() -> None:
    sample_rate = 44_100
    duration = 20.0
    time = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    modulation = 0.52 + 0.46 * np.sin(2.0 * np.pi * 0.1 * time)
    signal = modulation * np.sin(2.0 * np.pi * 880.0 * time)

    analysis = analyze_audio(signal, sample_rate=sample_rate)

    assert analysis.amplitude_modulation_depth_db >= 12.0
    assert analysis.dominant_amplitude_modulation_hz < 2.0
    assert all(
        risk.code != "strong_amplitude_modulation" for risk in analysis.artifact_risks
    )


def test_analyze_audio_reports_extreme_compression_artifact_risk() -> None:
    sample_rate = 44_100
    duration = 1.5
    time = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    signal = 0.88 * np.sign(np.sin(2.0 * np.pi * 220.0 * time))

    analysis = analyze_audio(signal, sample_rate=sample_rate)

    assert any(risk.code == "extreme_compression" for risk in analysis.artifact_risks)
    assert analysis.crest_factor_db <= 5.0


def test_analyze_score_reports_density_and_ranges() -> None:
    score = Score(f0=55.0)
    score.add_note("bass", start=0.0, duration=3.0, partial=2.0, amp=0.2)
    score.add_note("lead", start=0.0, duration=1.0, partial=6.0, amp=0.2)
    score.add_note("lead", start=1.0, duration=1.0, partial=7.0, amp=0.2)

    analysis = analyze_score(score)

    assert analysis.note_count == 3
    assert analysis.voice_count == 2
    assert analysis.peak_simultaneous_notes >= 2
    assert analysis.partial_range == (2.0, 7.0)
    assert "lead" in analysis.voice_summaries
    assert analysis.timing_drift_summary["max_absolute_offset_ms"] == 0.0
    assert analysis.timing_drift_summary["max_inter_voice_spread_ms"] == 0.0


def test_analyze_score_reports_timing_drift_stats() -> None:
    score = Score(
        f0=55.0,
        timing_humanize=TimingHumanizeSpec(
            preset="loose_late_night",
            ensemble_amount_ms=24.0,
            voice_spread_ms=8.0,
            micro_jitter_ms=1.0,
            chord_spread_ms=6.0,
            seed=7,
        ),
    )
    for start in (0.0, 1.0, 2.0, 3.0):
        score.add_note("bass", start=start, duration=0.7, partial=2.0, amp=0.2)
        score.add_note("lead", start=start, duration=0.7, partial=6.0, amp=0.2)

    analysis = analyze_score(score)

    assert analysis.timing_drift_summary["max_absolute_offset_ms"] > 0.0
    assert analysis.timing_drift_summary["max_inter_voice_spread_ms"] > 0.0
    assert analysis.timing_drift_windows


def test_build_score_timeline_includes_sections_and_resolved_notes() -> None:
    score = Score(f0=55.0)
    score.add_note("bass", start=0.0, duration=1.0, partial=2.0, amp=0.2)
    score.add_note(
        "lead", start=0.5, duration=0.5, partial=6.0, amp=0.2, label="pickup"
    )

    timeline = build_score_timeline(
        score=score,
        sections=(PieceSection(label="Intro", start_seconds=0.0, end_seconds=2.0),),
    )

    assert timeline["sections"][0]["label"] == "Intro"
    assert timeline["notes"][1]["label"] == "pickup"
    assert timeline["windows"]


def test_save_analysis_artifacts_writes_manifest_and_plots(tmp_path: Path) -> None:
    score = Score(f0=55.0)
    score.add_send_bus(
        "room",
        effects=[
            EffectSpec("delay", {"delay_seconds": 0.1, "feedback": 0.0, "mix": 1.0})
        ],
    )
    score.master_effects = [
        EffectSpec("compressor", {"threshold_db": -30.0, "ratio": 3.0})
    ]
    score.add_voice("bass", sends=[VoiceSend("room", send_db=-6.0)])
    score.add_note("bass", start=0.0, duration=1.0, partial=2.0, amp=0.2)
    score.add_note("lead", start=0.25, duration=0.75, partial=6.0, amp=0.2)

    stems = score.render_stems()
    mix = score.render()
    _mix_with_effects, _stems_with_effects, effect_analysis = (
        score.render_with_effect_analysis()
    )
    manifest = save_analysis_artifacts(
        output_prefix=tmp_path / "example_piece",
        mix_signal=mix,
        sample_rate=score.sample_rate,
        stems=stems,
        effect_analysis=effect_analysis,
        score=score,
        piece_sections=(
            PieceSection(label="Intro", start_seconds=0.0, end_seconds=1.0),
        ),
    )

    manifest_path = Path(manifest["manifest_path"])
    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path.exists()
    assert Path(saved_manifest["mix"]["artifacts"]["spectrum"]).exists()
    assert Path(saved_manifest["mix"]["artifacts"]["spectrogram"]).exists()
    assert Path(saved_manifest["mix"]["artifacts"]["band_energy"]).exists()
    assert "pre_export_summary" not in saved_manifest["mix"]
    assert Path(saved_manifest["score"]["artifacts"]["density"]).exists()
    assert Path(saved_manifest["score"]["artifacts"]["timeline"]).exists()
    assert Path(saved_manifest["voices"]["bass"]["artifacts"]["spectrum"]).exists()
    assert saved_manifest["effect_analysis"] == effect_analysis
    assert saved_manifest["effect_analysis"]["mix_effects"]
    assert saved_manifest["effect_analysis"]["send_effects"]["room"]


def test_save_analysis_artifacts_records_pre_export_mix_summary(tmp_path: Path) -> None:
    signal = np.full(44_100, 0.5, dtype=np.float64)
    normalized_signal = signal * 0.5

    manifest = save_analysis_artifacts(
        output_prefix=tmp_path / "normalized_piece",
        mix_signal=normalized_signal,
        pre_export_mix_signal=signal,
        sample_rate=44_100,
    )

    assert "pre_export_summary" in manifest["mix"]
    assert (
        manifest["mix"]["pre_export_summary"]["peak_dbfs"]
        > manifest["mix"]["summary"]["peak_dbfs"]
    )


def test_save_analysis_artifacts_does_not_flag_intentional_export_normalization(
    tmp_path: Path,
) -> None:
    sample_rate = 44_100
    duration_seconds = 2.0
    time = (
        np.arange(int(sample_rate * duration_seconds), dtype=np.float64) / sample_rate
    )
    pre_export_signal = 0.01 * np.sin(2.0 * np.pi * 220.0 * time)
    export_signal = pre_export_signal * 10.0

    manifest = save_analysis_artifacts(
        output_prefix=tmp_path / "normalized_export_piece",
        mix_signal=export_signal,
        pre_master_mix_signal=pre_export_signal,
        pre_export_mix_signal=pre_export_signal,
        sample_rate=sample_rate,
    )

    mix_risk_codes = {risk["code"] for risk in manifest["artifact_risk"]["mix"]}

    assert "export_loudness_jump" not in mix_risk_codes
    assert "heavy_export_compression" not in mix_risk_codes


def test_save_analysis_artifacts_flags_loudness_jump_with_crest_collapse(
    tmp_path: Path,
) -> None:
    sample_rate = 44_100
    duration_seconds = 2.0
    time = (
        np.arange(int(sample_rate * duration_seconds), dtype=np.float64) / sample_rate
    )
    pre_export_signal = 0.01 * np.sin(2.0 * np.pi * 220.0 * time)
    export_signal = 0.1 * np.sign(np.sin(2.0 * np.pi * 220.0 * time))

    manifest = save_analysis_artifacts(
        output_prefix=tmp_path / "compressed_export_piece",
        mix_signal=export_signal,
        pre_master_mix_signal=pre_export_signal,
        pre_export_mix_signal=export_signal,
        sample_rate=sample_rate,
    )

    mix_risk_codes = {risk["code"] for risk in manifest["artifact_risk"]["mix"]}

    assert (
        "export_loudness_jump" in mix_risk_codes
        or "heavy_export_compression" in mix_risk_codes
    )


def test_save_analysis_artifacts_records_artifact_risk_report(tmp_path: Path) -> None:
    risky_score = Score(f0=55.0)
    risky_score.add_voice(
        "lead",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "triangle",
            "cutoff_hz": 3_600.0,
            "resonance_q": 2.06,
            "filter_env_amount": 0.95,
            "filter_drive": 0.09,
            "attack": 0.05,
            "decay": 0.15,
            "sustain_level": 0.45,
            "release": 0.25,
        },
        velocity_to_params={
            "cutoff_hz": VelocityParamMap(min_value=1_200.0, max_value=2_600.0),
            "filter_env_amount": VelocityParamMap(min_value=0.65, max_value=1.2),
        },
    )
    for index, cutoff_hz in enumerate((2_400.0, 3_400.0, 4_100.0, 4_300.0)):
        risky_score.add_note(
            "lead",
            start=index * 0.6,
            duration=0.5,
            partial=8.0,
            amp_db=-15.0,
            velocity=1.15,
            synth={"cutoff_hz": cutoff_hz},
        )

    risky_manifest = save_analysis_artifacts(
        output_prefix=tmp_path / "risky_piece",
        mix_signal=risky_score.render(),
        sample_rate=risky_score.sample_rate,
        stems=risky_score.render_stems(),
        score=risky_score,
    )

    parameter_risks = risky_manifest["artifact_risk"]["parameter_surfaces"]["lead"]
    parameter_codes = {risk["code"] for risk in parameter_risks}
    assert "aggressive_filter_motion" in parameter_codes
    assert "bright_hot_authoring" in parameter_codes
    assert "velocity_param_out_of_bounds" in parameter_codes
    assert "wide_velocity_filter_env" not in parameter_codes
    assert (
        risky_manifest["artifact_risk"]["summary"]["voice_count_with_parameter_risks"]
        == 1
    )

    safe_score = Score(f0=55.0)
    safe_score.add_voice(
        "lead",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "triangle",
            "cutoff_hz": 2_200.0,
            "resonance_q": 1.50,
            "filter_env_amount": 0.18,
            "filter_drive": 0.04,
            "attack": 0.05,
            "decay": 0.15,
            "sustain_level": 0.45,
            "release": 0.25,
        },
        velocity_to_params={
            "cutoff_hz": VelocityParamMap(min_value=1_900.0, max_value=2_400.0),
        },
    )
    for index, cutoff_hz in enumerate((1_900.0, 2_050.0, 2_200.0, 2_350.0)):
        safe_score.add_note(
            "lead",
            start=index * 0.6,
            duration=0.5,
            partial=8.0,
            amp_db=-18.0,
            velocity=1.0,
            synth={"cutoff_hz": cutoff_hz},
        )

    safe_manifest = save_analysis_artifacts(
        output_prefix=tmp_path / "safe_piece",
        mix_signal=safe_score.render(),
        sample_rate=safe_score.sample_rate,
        stems=safe_score.render_stems(),
        score=safe_score,
    )

    assert safe_manifest["artifact_risk"]["parameter_surfaces"] == {}


def test_save_analysis_artifacts_reports_implausibly_wide_velocity_filter_env_span(
    tmp_path: Path,
) -> None:
    score = Score(f0=55.0)
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "triangle",
            "cutoff_hz": 1_800.0,
            "filter_env_amount": 0.92,
            "filter_drive": 0.02,
        },
        velocity_to_params={
            "filter_env_amount": VelocityParamMap(min_value=0.05, max_value=0.9),
        },
    )
    for index in range(4):
        score.add_note(
            "lead",
            start=index * 0.5,
            duration=0.35,
            partial=6.0,
            amp_db=-20.0,
            velocity=1.0,
        )

    manifest = save_analysis_artifacts(
        output_prefix=tmp_path / "wide_velocity_filter_env_piece",
        mix_signal=score.render(),
        sample_rate=score.sample_rate,
        stems=score.render_stems(),
        score=score,
    )

    parameter_risks = manifest["artifact_risk"]["parameter_surfaces"]["lead"]
    parameter_codes = {risk["code"] for risk in parameter_risks}
    assert "wide_velocity_filter_env" in parameter_codes
    assert "velocity_param_out_of_bounds" not in parameter_codes


def test_compare_analysis_manifests_reports_mix_and_score_deltas(
    tmp_path: Path,
) -> None:
    before_manifest_path = tmp_path / "before.analysis.json"
    after_manifest_path = tmp_path / "after.analysis.json"
    comparison_path = tmp_path / "comparison.analysis.json"

    before_manifest_path.write_text(
        json.dumps(
            {
                "mix": {
                    "summary": {
                        "peak_dbfs": -4.0,
                        "true_peak_dbfs": -3.6,
                        "rms_dbfs": -18.0,
                        "integrated_lufs": -15.0,
                        "spectral_centroid_hz": 300.0,
                        "dominant_frequency_hz": 110.0,
                        "low_high_balance_db": 20.0,
                        "spectral_tilt_db_per_octave": -8.0,
                        "tilt_error_db_per_octave": -5.0,
                        "warnings": ["dark"],
                    }
                },
                "score": {
                    "summary": {
                        "note_count": 10,
                        "notes_per_second": 1.0,
                        "peak_simultaneous_notes": 4,
                        "mean_simultaneous_notes": 3.0,
                        "mean_attack_density_hz": 1.0,
                        "max_attack_density_hz": 2.0,
                        "warnings": ["dense"],
                    }
                },
                "voices": {
                    "lead": {
                        "summary": {
                            "peak_dbfs": -6.0,
                            "true_peak_dbfs": -5.5,
                            "rms_dbfs": -20.0,
                            "integrated_lufs": -18.0,
                            "spectral_centroid_hz": 500.0,
                            "low_high_balance_db": 15.0,
                            "spectral_tilt_db_per_octave": -7.0,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    after_manifest_path.write_text(
        json.dumps(
            {
                "mix": {
                    "summary": {
                        "peak_dbfs": -5.0,
                        "true_peak_dbfs": -4.8,
                        "rms_dbfs": -17.0,
                        "integrated_lufs": -14.0,
                        "spectral_centroid_hz": 420.0,
                        "dominant_frequency_hz": 220.0,
                        "low_high_balance_db": 14.0,
                        "spectral_tilt_db_per_octave": -5.0,
                        "tilt_error_db_per_octave": -2.0,
                        "warnings": [],
                    }
                },
                "score": {
                    "summary": {
                        "note_count": 10,
                        "notes_per_second": 1.0,
                        "peak_simultaneous_notes": 3,
                        "mean_simultaneous_notes": 2.5,
                        "mean_attack_density_hz": 1.0,
                        "max_attack_density_hz": 2.0,
                        "warnings": [],
                    }
                },
                "voices": {
                    "lead": {
                        "summary": {
                            "peak_dbfs": -7.0,
                            "true_peak_dbfs": -6.8,
                            "rms_dbfs": -19.0,
                            "integrated_lufs": -17.0,
                            "spectral_centroid_hz": 650.0,
                            "low_high_balance_db": 10.0,
                            "spectral_tilt_db_per_octave": -5.0,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    comparison = compare_analysis_manifests(
        before_manifest_path,
        after_manifest_path,
        output_path=comparison_path,
    )

    assert comparison["mix_delta"]["spectral_centroid_hz"] == 120.0
    assert comparison["mix_delta"]["low_high_balance_db"] == -6.0
    assert comparison["mix_delta"]["integrated_lufs"] == 1.0
    assert comparison["score_delta"]["peak_simultaneous_notes"] == -1.0
    assert comparison["voice_delta"]["lead"]["spectral_centroid_hz"] == 150.0
    assert comparison_path.exists()


def test_analyze_audio_reports_thd_for_clean_sine() -> None:
    sample_rate = 44_100
    duration = 2.0
    time = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    clean_sine = 0.5 * np.sin(2.0 * np.pi * 440.0 * time)

    analysis = analyze_audio(clean_sine, sample_rate=sample_rate)

    assert analysis.thd_pct < 2.0
    assert analysis.thd_character in ("clean", "subtle_warmth")
    assert all(risk.code != "harmonic_distortion" for risk in analysis.artifact_risks)


def test_analyze_audio_reports_thd_for_distorted_signal() -> None:
    sample_rate = 44_100
    duration = 2.0
    time = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    sine = 0.9 * np.sin(2.0 * np.pi * 220.0 * time)
    distorted = np.clip(sine * 5.0, -1.0, 1.0)

    analysis = analyze_audio(distorted, sample_rate=sample_rate)

    # THD measurement still works — high THD is correctly detected as metadata.
    assert analysis.thd_pct >= 15.0
    assert analysis.thd_character in ("distortion", "fuzz")
    # But absolute THD no longer fires artifact risk warnings (false positive on
    # harmonically rich timbres like saw waves).
    risk_codes = {risk.code for risk in analysis.artifact_risks}
    assert "harmonic_distortion" not in risk_codes


def test_analyze_audio_does_not_flag_saw_wave_thd() -> None:
    sample_rate = 44_100
    duration = 2.0
    time = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    # Band-limited saw wave: first 20 harmonics at 220 Hz.
    saw = np.zeros_like(time)
    for harmonic in range(1, 21):
        saw += (
            ((-1.0) ** (harmonic + 1))
            / harmonic
            * np.sin(2.0 * np.pi * 220.0 * harmonic * time)
        )
    saw *= 0.5 / np.max(np.abs(saw))  # normalize to 0.5 peak

    analysis = analyze_audio(saw, sample_rate=sample_rate)

    assert analysis.thd_pct >= 10.0
    risk_codes = {risk.code for risk in analysis.artifact_risks}
    assert "harmonic_distortion" not in risk_codes


def test_save_analysis_artifacts_flags_mastering_introduced_thd(
    tmp_path: Path,
) -> None:
    sample_rate = 44_100
    duration = 2.0
    time = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    pre_master = 0.3 * np.sin(2.0 * np.pi * 220.0 * time)
    post_master = np.clip(pre_master * 5.0, -1.0, 1.0)

    manifest = save_analysis_artifacts(
        output_prefix=tmp_path / "thd_delta",
        mix_signal=post_master,
        pre_master_mix_signal=pre_master,
        pre_export_mix_signal=post_master,
        sample_rate=sample_rate,
    )

    mix_risk_codes = {risk["code"] for risk in manifest["artifact_risk"]["mix"]}
    assert "harmonic_distortion" in mix_risk_codes


def test_analyze_audio_does_not_flag_rhythmic_pattern_as_modulation_artifact() -> None:
    sample_rate = 44_100
    duration = 4.0
    time = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    # ~3.33 Hz square-ish envelope (below 4 Hz floor): eighth notes at ~100 BPM.
    envelope = 0.5 * (1.0 + np.sign(np.sin(2.0 * np.pi * 3.33 * time)))
    signal = envelope * 0.4 * np.sin(2.0 * np.pi * 440.0 * time)

    analysis = analyze_audio(signal, sample_rate=sample_rate)

    assert analysis.dominant_amplitude_modulation_hz < 4.0
    risk_codes = {risk.code for risk in analysis.artifact_risks}
    assert "strong_amplitude_modulation" not in risk_codes
