"""Smoke tests for all score-backed pieces in the registry."""

import numpy as np
import pytest

from code_musics.pieces import PIECES
from code_musics.pieces.registry import PieceDefinition

_SCORE_BACKED_PIECES = [
    (name, definition)
    for name, definition in sorted(PIECES.items())
    if definition.build_score is not None
]

# Curated subset covering diverse engines and effect topologies.
# Avoids any piece that requires external VST/VST3 plugins so this
# test can gate rendering on any machine that passes the main suite.
_RENDER_SMOKE_PIECE_NAMES: tuple[str, ...] = (
    "ji_chorale",
    "amber_room",
    "bwv_846_piano",
    "breath_study",
    "filter_palette_study",
    "va_showcase",
    "vowel_cathedral",
    "struck_light",
    "mod_matrix_study",
    "slow_glass",
)

# Per-piece window overrides for pieces with slow intros or leading silence.
# Maps piece name to (start_seconds, end_seconds). Defaults to (0.0, 2.0).
_PIECE_WINDOW_OVERRIDES: dict[str, tuple[float, float]] = {
    # slow_glass opens with the first note at t=2.0s; shift past the lead-in.
    "slow_glass": (4.0, 6.0),
}

_RENDER_SMOKE_PIECES = [
    (name, PIECES[name])
    for name in _RENDER_SMOKE_PIECE_NAMES
    if name in PIECES and PIECES[name].build_score is not None
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


@pytest.mark.parametrize(
    ("piece_name", "definition"),
    _RENDER_SMOKE_PIECES,
    ids=[name for name, _ in _RENDER_SMOKE_PIECES],
)
def test_score_backed_piece_renders_clean_window(
    piece_name: str,
    definition: PieceDefinition,
) -> None:
    """Render a short window of the piece and assert audio sanity invariants."""
    assert definition.build_score is not None
    score = definition.build_score()

    start_seconds, end_seconds = _PIECE_WINDOW_OVERRIDES.get(piece_name, (0.0, 2.0))
    window = score.extract_window(start_seconds=start_seconds, end_seconds=end_seconds)
    if not window.voices:
        pytest.skip(
            f"{piece_name}: no voices audible in "
            f"{start_seconds:.1f}-{end_seconds:.1f}s window; adjust window or subset"
        )

    audio = window.render()

    assert audio.size > 0, f"{piece_name}: rendered audio is empty"
    assert np.isfinite(audio).all(), f"{piece_name}: rendered audio contains NaN or Inf"

    peak = float(np.abs(audio).max())
    assert peak <= 1.0 + 1e-6, (
        f"{piece_name}: rendered audio peak {peak:.6f} exceeds 1.0 (hard clipping)"
    )
    assert peak > 1e-4, (
        f"{piece_name}: rendered audio peak {peak:.6e} is effectively silent"
    )

    if audio.ndim == 2:
        for channel_index in range(audio.shape[0]):
            channel = audio[channel_index]
            channel_peak = float(np.abs(channel).max())
            assert np.isfinite(channel).all(), (
                f"{piece_name}: channel {channel_index} contains NaN or Inf"
            )
            assert channel_peak <= 1.0 + 1e-6, (
                f"{piece_name}: channel {channel_index} peak {channel_peak:.6f} "
                "exceeds 1.0"
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
