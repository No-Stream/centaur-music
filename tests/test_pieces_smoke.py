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


def test_septimal_bloom_no_param_curves() -> None:
    """septimal_bloom should use static Surge XT params, not chunked param_curves."""
    definition = PIECES["septimal_bloom"]
    assert definition.build_score is not None
    score = definition.build_score()

    for voice_name, voice in score.voices.items():
        assert "param_curves" not in voice.synth_defaults, (
            f"Voice {voice_name!r} still has param_curves -- "
            "remove chunked automation and use static Surge XT filter cutoffs"
        )
