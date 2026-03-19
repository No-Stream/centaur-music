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
from code_musics.score import Score


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
    score.add_note("bass", start=0.0, duration=1.0, partial=2.0, amp=0.2)
    score.add_note("lead", start=0.25, duration=0.75, partial=6.0, amp=0.2)

    stems = score.render_stems()
    mix = score.render()
    manifest = save_analysis_artifacts(
        output_prefix=tmp_path / "example_piece",
        mix_signal=mix,
        sample_rate=score.sample_rate,
        stems=stems,
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
