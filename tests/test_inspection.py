"""Timestamp inspection helper tests."""

from __future__ import annotations

from code_musics.inspection import (
    format_inspection_summary,
    inspect_score_timestamp,
    parse_timestamp_seconds,
)
from code_musics.pieces.registry import PieceSection
from code_musics.score import Score


def test_parse_timestamp_seconds_supports_common_formats() -> None:
    assert parse_timestamp_seconds("130") == 130.0
    assert parse_timestamp_seconds("2:10") == 130.0
    assert parse_timestamp_seconds("1:02:03") == 3723.0


def test_inspect_score_timestamp_reports_section_and_notes() -> None:
    score = Score(f0=55.0)
    score.add_note("bass", start=0.0, duration=4.0, partial=2.0, amp=0.2)
    score.add_note("lead", start=2.0, duration=1.0, partial=6.0, amp=0.2, label="entry")

    inspection = inspect_score_timestamp(
        score=score,
        timestamp_seconds=2.2,
        sections=(PieceSection(label="Opening", start_seconds=0.0, end_seconds=5.0),),
        window_seconds=4.0,
        piece_name="test_piece",
    )

    assert inspection["section"] is not None
    assert inspection["section"]["label"] == "Opening"
    assert inspection["active_voice_names"] == ["bass", "lead"]
    assert any(note["label"] == "entry" for note in inspection["nearby_onsets"])

    summary = format_inspection_summary(inspection)
    assert "Piece: test_piece" in summary
    assert "Section: Opening" in summary
    assert "entry" in summary
