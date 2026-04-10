"""Smoke tests for all score-backed pieces in the registry."""

import pytest

from code_musics.pieces import PIECES
from code_musics.pieces.registry import PieceDefinition

_SCORE_BACKED_PIECES = [
    (name, definition)
    for name, definition in sorted(PIECES.items())
    if definition.build_score is not None
]


@pytest.mark.parametrize(
    ("piece_name", "definition"),
    _SCORE_BACKED_PIECES,
    ids=[name for name, _ in _SCORE_BACKED_PIECES],
)
def test_score_backed_piece_builds_valid_score(
    piece_name: str,
    definition: PieceDefinition,
) -> None:
    assert definition.build_score is not None
    score = definition.build_score()

    assert score.total_dur > 0, f"{piece_name}: total_dur should be positive"
    voices_with_notes = [
        voice_name for voice_name, voice in score.voices.items() if voice.notes
    ]
    assert len(voices_with_notes) > 0, (
        f"{piece_name}: at least one voice should have notes"
    )
