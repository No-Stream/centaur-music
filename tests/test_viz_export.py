"""Tests for the generic visualization-JSON exporter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import code_musics.render as render_module
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.render import export_piece_viz
from code_musics.score import Score
from code_musics.synth import write_wav
from code_musics.viz_export import (
    VIZ_SCHEMA_VERSION,
    VizExportSpec,
    build_rms_envelope,
    build_viz_payload,
    parse_viz_label,
)


def _build_test_score() -> Score:
    """Build a tiny 2-voice score: one tonal, one percussive."""
    score = Score(f0_hz=220.0, auto_master_gain_stage=False, master_effects=[])
    score.add_voice("lead", normalize_lufs=None, velocity_humanize=None)
    score.add_note(
        "lead",
        start=0.0,
        duration=0.2,
        partial=1.0,
        amp=0.2,
        velocity=1.2,
        label="walker;deg=3;oct=1",
    )
    score.add_note(
        "lead",
        start=0.2,
        duration=0.2,
        partial=3 / 2,
        amp_db=-6.0,
        velocity=0.8,
        label="bloom",
    )
    score.add_note(
        "lead",
        start=0.4,
        duration=0.2,
        partial=2.0,
        amp=0.15,
        label=";=x;deg=2",
    )
    score.add_voice(
        "kick",
        normalize_lufs=None,
        normalize_peak_db=-6.0,
        velocity_humanize=None,
    )
    score.add_note("kick", start=0.0, duration=0.1, partial=1.0, amp=0.3)
    return score


def _write_sine_wav(
    path: Path, *, seconds: float = 1.0, sample_rate: int = 44100
) -> None:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    sine = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    write_wav(path, sine)


class TestVizPayloadSchema:
    def test_top_level_keys_and_schema_version(self, tmp_path: Path) -> None:
        score = _build_test_score()
        wav_path = tmp_path / "mix.wav"
        _write_sine_wav(wav_path)
        payload = build_viz_payload(
            score=score,
            mix_wav_path=wav_path,
            spec=VizExportSpec(piece_name="test_piece", output_name="test_piece"),
        )

        assert payload["schema_version"] == VIZ_SCHEMA_VERSION == 1
        for key in (
            "piece_name",
            "total_duration_seconds",
            "sample_rate",
            "f0_hz",
            "sections",
            "voices",
            "notes",
            "annotations",
            "envelope",
        ):
            assert key in payload

    def test_notes_sorted_by_start_then_voice_then_index(self, tmp_path: Path) -> None:
        score = _build_test_score()
        wav_path = tmp_path / "mix.wav"
        _write_sine_wav(wav_path)
        payload = build_viz_payload(
            score=score,
            mix_wav_path=wav_path,
            spec=VizExportSpec(piece_name="test_piece", output_name="test_piece"),
        )

        starts = [note["start_seconds"] for note in payload["notes"]]
        assert starts == sorted(starts)
        assert len(payload["notes"]) == 4


class TestVelocityAmpJoin:
    def test_velocity_and_amp_db_join_matches_source_notes(
        self, tmp_path: Path
    ) -> None:
        score = _build_test_score()
        wav_path = tmp_path / "mix.wav"
        _write_sine_wav(wav_path)
        payload = build_viz_payload(
            score=score,
            mix_wav_path=wav_path,
            spec=VizExportSpec(piece_name="test_piece", output_name="test_piece"),
        )

        lead_notes = [note for note in payload["notes"] if note["voice_name"] == "lead"]
        lead_notes.sort(key=lambda note: note["note_index"])
        source_notes = score.voices["lead"].notes

        for note_record, source_note in zip(lead_notes, source_notes, strict=True):
            assert note_record["velocity"] == pytest.approx(
                source_note.velocity, abs=1e-5
            )
            assert note_record["amp"] == pytest.approx(source_note.amp, abs=1e-5)
            if source_note.amp_db is not None:
                assert note_record["amp_db"] == pytest.approx(
                    source_note.amp_db, abs=1e-3
                )


class TestParseVizLabel:
    def test_structured_label(self) -> None:
        result = parse_viz_label("walker;deg=3;oct=1")
        assert result == {"kind": "walker", "tags": {"deg": 3, "oct": 1}}

    def test_coerces_int_float_str(self) -> None:
        result = parse_viz_label("kind;a=1;b=1.5;c=hello")
        assert result is not None
        assert result["tags"]["a"] == 1
        assert isinstance(result["tags"]["a"], int)
        assert result["tags"]["b"] == pytest.approx(1.5)
        assert isinstance(result["tags"]["b"], float)
        assert result["tags"]["c"] == "hello"

    def test_bare_token_becomes_true(self) -> None:
        result = parse_viz_label("kind;accent")
        assert result == {"kind": "kind", "tags": {"accent": True}}

    def test_legacy_plain_label(self) -> None:
        result = parse_viz_label("bloom")
        assert result == {"kind": "bloom", "tags": {}}

    def test_none_input(self) -> None:
        assert parse_viz_label(None) is None
        assert parse_viz_label("") is None

    def test_malformed_segments_skipped_silently(self) -> None:
        result = parse_viz_label("kind;=x;deg=2;;")
        assert result == {"kind": "kind", "tags": {"deg": 2}}

    def test_never_raises_on_garbage(self) -> None:
        for garbage in (";;;", "=", "==", "a=b=c", ";=", "kind;=", "kind;a="):
            parse_viz_label(garbage)


class TestVoiceMetadata:
    def test_percussive_and_tonal_flags(self, tmp_path: Path) -> None:
        score = _build_test_score()
        wav_path = tmp_path / "mix.wav"
        _write_sine_wav(wav_path)
        payload = build_viz_payload(
            score=score,
            mix_wav_path=wav_path,
            spec=VizExportSpec(piece_name="test_piece", output_name="test_piece"),
        )

        assert payload["voices"]["kick"]["is_percussive"] is True
        assert payload["voices"]["lead"]["is_percussive"] is False
        assert payload["voices"]["lead"]["note_count"] == 3
        assert payload["voices"]["kick"]["note_count"] == 1


class TestEnvelope:
    def test_frame_count_and_flatness_for_constant_sine(self, tmp_path: Path) -> None:
        sample_rate = 44100
        seconds = 1.0
        wav_path = tmp_path / "mix.wav"
        _write_sine_wav(wav_path, seconds=seconds, sample_rate=sample_rate)
        signal, sr = None, sample_rate
        import soundfile as sf

        signal, sr = sf.read(str(wav_path))
        envelope = build_rms_envelope(
            np.asarray(signal), sample_rate=sr, hop_seconds=0.025
        )

        expected_frames = int(seconds / 0.025)
        assert abs(envelope["frame_count"] - expected_frames) <= 1
        rms_values = np.array(envelope["rms"])
        # Interior frames (away from start/end fade due to framing) should be
        # close to the sine's RMS (~0.5 / sqrt(2)).
        interior = rms_values[2:-2]
        assert np.all(interior > 0.2)
        assert np.std(interior) < 0.05

    def test_silence_floors_at_minus_120_db(self) -> None:
        silence = np.zeros(44100)
        envelope = build_rms_envelope(silence, sample_rate=44100, hop_seconds=0.025)
        assert all(value == pytest.approx(-120.0) for value in envelope["rms_db"])


class TestMissingWav:
    def test_missing_wav_raises_with_path_hint(self, tmp_path: Path) -> None:
        score = _build_test_score()
        missing_path = tmp_path / "does_not_exist.wav"
        with pytest.raises(FileNotFoundError, match=str(missing_path)):
            build_viz_payload(
                score=score,
                mix_wav_path=missing_path,
                spec=VizExportSpec(piece_name="test_piece", output_name="test_piece"),
            )


class TestAnnotationsPassthrough:
    def test_annotations_roundtrip(self, tmp_path: Path) -> None:
        score = _build_test_score()
        wav_path = tmp_path / "mix.wav"
        _write_sine_wav(wav_path)
        payload = build_viz_payload(
            score=score,
            mix_wav_path=wav_path,
            annotations={"x": 1, "nested": {"y": [1, 2, 3]}},
            spec=VizExportSpec(piece_name="test_piece", output_name="test_piece"),
        )
        assert payload["annotations"] == {"x": 1, "nested": {"y": [1, 2, 3]}}

    def test_annotations_default_empty_dict(self, tmp_path: Path) -> None:
        score = _build_test_score()
        wav_path = tmp_path / "mix.wav"
        _write_sine_wav(wav_path)
        payload = build_viz_payload(
            score=score,
            mix_wav_path=wav_path,
            spec=VizExportSpec(piece_name="test_piece", output_name="test_piece"),
        )
        assert payload["annotations"] == {}


class TestExportPieceViz:
    def test_unknown_piece_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown piece"):
            export_piece_viz("does_not_exist_piece")

    def test_export_writes_valid_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def build_score() -> Score:
            return _build_test_score()

        piece_definition = PieceDefinition(
            name="test_viz_piece",
            output_name="test_viz_piece",
            build_score=build_score,
            sections=(PieceSection(label="intro", start_seconds=0.0, end_seconds=0.5),),
            build_viz_annotations=lambda: {"x": 1},
        )
        monkeypatch.setitem(render_module.PIECES, "test_viz_piece", piece_definition)

        wav_path = tmp_path / "test_viz_piece" / "test_viz_piece.wav"
        _write_sine_wav(wav_path)

        result = export_piece_viz("test_viz_piece", output_dir=tmp_path)

        assert result.viz_path.exists()
        assert result.viz_path.name == "test_viz_piece.viz.json"
        payload = json.loads(result.viz_path.read_text())
        assert payload["piece_name"] == "test_viz_piece"
        assert payload["annotations"] == {"x": 1}
        assert payload["sections"] == [
            {"label": "intro", "start_seconds": 0.0, "end_seconds": 0.5}
        ]
        assert result.note_count == len(payload["notes"])
