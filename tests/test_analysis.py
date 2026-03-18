"""Analysis helper tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from code_musics.analysis import analyze_audio, analyze_score, save_analysis_artifacts
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
