"""Analysis helper tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from code_musics.analysis import (
    analyze_audio,
    analyze_score,
    compare_analysis_manifests,
    save_analysis_artifacts,
)
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
    )

    manifest_path = Path(manifest["manifest_path"])
    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path.exists()
    assert Path(saved_manifest["mix"]["artifacts"]["spectrum"]).exists()
    assert Path(saved_manifest["mix"]["artifacts"]["spectrogram"]).exists()
    assert Path(saved_manifest["mix"]["artifacts"]["band_energy"]).exists()
    assert Path(saved_manifest["score"]["artifacts"]["density"]).exists()
    assert Path(saved_manifest["voices"]["bass"]["artifacts"]["spectrum"]).exists()


def test_compare_analysis_manifests_reports_mix_and_score_deltas(tmp_path: Path) -> None:
    before_manifest_path = tmp_path / "before.analysis.json"
    after_manifest_path = tmp_path / "after.analysis.json"
    comparison_path = tmp_path / "comparison.analysis.json"

    before_manifest_path.write_text(
        json.dumps(
            {
                "mix": {
                    "summary": {
                        "peak_dbfs": -4.0,
                        "rms_dbfs": -18.0,
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
                            "rms_dbfs": -20.0,
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
                        "rms_dbfs": -17.0,
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
                            "rms_dbfs": -19.0,
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
    assert comparison["score_delta"]["peak_simultaneous_notes"] == -1.0
    assert comparison["voice_delta"]["lead"]["spectral_centroid_hz"] == 150.0
    assert comparison_path.exists()
