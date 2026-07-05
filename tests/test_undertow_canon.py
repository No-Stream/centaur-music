"""Tests for the Undertow Canon piece."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from code_musics.pieces import PIECES
from code_musics.pieces.undertow_canon import BAR, TOTAL_BARS, build_score


def test_undertow_canon_is_registered() -> None:
    definition = PIECES["undertow_canon"]

    assert definition.name == "undertow_canon"
    assert definition.output_name == "undertow_canon"
    assert definition.build_score is build_score
    assert not definition.study


def test_undertow_canon_builds_expected_voice_layout() -> None:
    score = build_score()

    assert set(score.voices) == {
        "organ_bass",
        "organ_subject",
        "organ_answer",
        "organ_shadow",
        "bell_arp",
        "grain_shimmer",
        "kick",
        "wood",
        "hat",
    }
    for voice in score.voices.values():
        assert voice.notes, f"{voice.name} should contain authored notes"


def test_undertow_canon_sections_cover_full_form() -> None:
    definition = PIECES["undertow_canon"]
    sections = definition.sections

    assert [section.label for section in sections] == [
        "ground",
        "canon_bloom",
        "undertow",
        "collision",
        "return",
    ]
    assert sections[0].start_seconds == pytest.approx(0.0)
    for previous, current in pairwise(sections):
        assert previous.end_seconds == pytest.approx(current.start_seconds)
        assert previous.start_seconds < previous.end_seconds
    assert sections[-1].end_seconds == pytest.approx(TOTAL_BARS * BAR)


def test_undertow_canon_percussion_is_peak_normalized() -> None:
    score = build_score()

    for voice_name in ("kick", "wood", "hat"):
        voice = score.voices[voice_name]
        assert voice.normalize_peak_db == pytest.approx(-6.0)
        assert voice.normalize_lufs is None
        assert voice.is_percussive()


def test_undertow_canon_short_window_renders_finite_audio() -> None:
    score = build_score()
    windowed = score.extract_window(start_seconds=16 * BAR, end_seconds=18 * BAR)

    audio = windowed.render()

    assert audio.size > 0
    assert np.all(np.isfinite(audio))
    peak = float(np.max(np.abs(audio)))
    assert peak > 1e-4
    assert peak <= 1.0 + 1e-6
