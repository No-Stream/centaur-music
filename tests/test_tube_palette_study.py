"""Smoke test for the tube_palette_study piece."""

from __future__ import annotations

import numpy as np

from code_musics.pieces.tube_palette_study import (
    N_SECTIONS,
    PIECES,
    SECTION_DUR,
    build_tube_palette_study,
)


def test_tube_palette_study_registered() -> None:
    assert "tube_palette_study" in PIECES
    piece = PIECES["tube_palette_study"]
    assert piece.build_score is build_tube_palette_study
    assert len(piece.sections) == N_SECTIONS


def test_tube_palette_study_renders() -> None:
    score = build_tube_palette_study()
    audio = score.render()
    assert np.all(np.isfinite(audio))
    assert audio.size > 0
    assert float(np.max(np.abs(audio))) > 1e-3

    # Sanity: rendered duration should be at least N_SECTIONS * SECTION_DUR
    # worth of samples at the score's sample rate.
    expected_min_samples = int(N_SECTIONS * SECTION_DUR * score.sample_rate)
    n_samples = audio.shape[-1] if audio.ndim > 1 else audio.shape[0]
    assert n_samples >= expected_min_samples * 0.95
