"""Viz instrumentation tests for hexany_garden.

The piece carries structured viz labels on every tonal note and exposes
build_viz_annotations() for the video pipeline.  These tests pin:
1. the note stream itself (labels must never perturb generation), and
2. the semantic coverage and internal consistency of labels/annotations.
"""

import hashlib
import json

import pytest

from code_musics.pieces.hexany_garden import (
    _CHORD_SLOTS,
    _POLAR_COMPLEMENT,
    _QUOTE_BARS,
    BAR,
    S3_BAR,
    TOTAL_DUR,
    build_score,
    build_viz_annotations,
)
from code_musics.score import Score
from code_musics.viz_export import parse_viz_label

# Deterministic digest of the full note stream (timing/pitch/velocity/level,
# labels excluded).  If a change to the piece is *intentionally* musical,
# update this hash; if this fails unexpectedly, viz instrumentation has
# perturbed generation (most likely by disturbing RNG draw order).
_NOTE_STREAM_SHA256 = "a6031e3ddaa82a32475a61f0d243d13d032751d03dcb3c1c1769e94488e7192d"
_NOTE_COUNT = 1796

_LABELED_VOICES = (
    "arp",
    "thumb",
    "pad",
    "bass",
    "bell",
    "thread",
    "bow",
    "glint",
    "haze",
)


@pytest.fixture(scope="module")
def score() -> Score:
    return build_score()


def _bar_of(start_seconds: float) -> int:
    return int(start_seconds // BAR) + 1


class TestNoteStreamRegression:
    def test_note_stream_fingerprint_unchanged(self, score: Score) -> None:
        lines: list[str] = []
        for voice_name, voice in score.voices.items():
            for i, note in enumerate(voice.notes):
                lines.append(
                    f"{voice_name}|{i}|{note.start:.9f}|{note.duration:.9f}"
                    f"|{note.partial}|{note.freq}|{note.velocity:.9f}|{note.amp_db}"
                )
        lines.sort()
        digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
        assert len(lines) == _NOTE_COUNT
        assert digest == _NOTE_STREAM_SHA256


class TestLabelCoverage:
    def test_all_tonal_voices_fully_labeled(self, score: Score) -> None:
        for voice_name in _LABELED_VOICES:
            notes = score.voices[voice_name].notes
            assert notes, voice_name
            unlabeled = [i for i, n in enumerate(notes) if not n.label]
            assert not unlabeled, f"{voice_name}: unlabeled notes {unlabeled[:5]}"

    def test_labels_parse_with_expected_tags(self, score: Score) -> None:
        for note in score.voices["arp"].notes:
            parsed = parse_viz_label(note.label)
            assert parsed is not None and parsed["kind"] == "walker"
            tags = parsed["tags"]
            assert tags["deg"] in range(6)
            assert tags["oct"] >= 0
            assert tags["phase"] in ("record", "replay", "free")
            assert tags["leap"] in ("none", "edge", "polar")
            assert tags["grid"] in ("straight", "slow", "tresillo", "seven")

    def test_polar_leaps_exist_and_only_after_the_turn(self, score: Score) -> None:
        polar_bars = [
            _bar_of(note.start)
            for note in score.voices["arp"].notes
            if parse_viz_label(note.label)["tags"]["leap"] == "polar"  # type: ignore[index]
        ]
        assert polar_bars, "expected at least one polar leap"
        assert min(polar_bars) >= S3_BAR

    def test_quote_notes_land_on_quote_phrase_bars(self, score: Score) -> None:
        quote_bars = {
            _bar_of(note.start)
            for note in score.voices["arp"].notes
            if parse_viz_label(note.label)["tags"].get("quote")  # type: ignore[index]
        }
        assert quote_bars
        # A quote cell spans the first two bars of its phrase (record), and
        # the riff replays re-state it at phrase offsets +2..+5.
        allowed = {start + off for start in _QUOTE_BARS for off in range(6)}
        assert quote_bars <= allowed

    def test_polar_mirror_statement_at_bar_49(self, score: Score) -> None:
        mirrors = [
            parse_viz_label(note.label)
            for note in score.voices["bell"].notes
            if note.label and "polar_mirror" in note.label
        ]
        assert len(mirrors) == 5  # the full motif, shadowed
        for parsed in mirrors:
            assert parsed is not None
            tags = parsed["tags"]
            assert _POLAR_COMPLEMENT[tags["mirror_of"]] == tags["deg"]


class TestVizAnnotations:
    def test_annotations_are_json_serializable(self) -> None:
        annotations = build_viz_annotations()
        round_tripped = json.loads(json.dumps(annotations))
        assert round_tripped["bloom"]["bar"] == 93

    def test_chord_slots_are_contiguous_and_complete(self) -> None:
        slots = build_viz_annotations()["chord_slots"]
        assert len(slots) == len(_CHORD_SLOTS)
        assert slots[0]["start_bar"] == 1
        for prev, cur in zip(slots, slots[1:], strict=False):
            assert prev["end_bar"] == cur["start_bar"]
        assert slots[-1]["end_seconds"] == pytest.approx(TOTAL_DUR, abs=1e-3)

    def test_region_polarity(self) -> None:
        regions = {r["name"]: r for r in build_viz_annotations()["regions"]}
        assert len(regions) == 9
        for name, region in regions.items():
            if name == "dyad":
                assert region["otonal"] is None
            elif name.startswith("u_"):
                assert region["otonal"] is False
                assert region["excluded_factor"] in (1, 3, 5, 7)
            else:
                assert region["otonal"] is True
                assert region["fixed_factor"] in (1, 3, 5, 7)

    def test_sections_match_piece_definition(self) -> None:
        from code_musics.pieces.hexany_garden import PIECES

        annotated = build_viz_annotations()["sections"]
        registered = PIECES["hexany_garden"].sections
        assert [s["label"] for s in annotated] == [s.label for s in registered]
        for ann, reg in zip(annotated, registered, strict=True):
            assert ann["start_seconds"] == pytest.approx(reg.start_seconds, abs=1e-3)
            assert ann["end_seconds"] == pytest.approx(reg.end_seconds, abs=1e-3)
