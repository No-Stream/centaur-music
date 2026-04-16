"""MIDI export tests."""

from __future__ import annotations

import json
from pathlib import Path

import mido
import pytest

import code_musics.render as render_module
from code_musics.automation import AutomationSegment, AutomationSpec, AutomationTarget
from code_musics.humanize import TimingHumanizeSpec
from code_musics.midi_export import (
    MidiBundleExportSpec,
    MidiStemNote,
    TuningAnalysisResult,
    export_midi_bundle,
)
from code_musics.midi_export_stems import (
    collect_stem_notes,
    resolve_shared_tuning_midi_note,
)
from code_musics.midi_export_tuning import (
    _nearest_neighbor_fill,
    assign_chromatic_slots,
)
from code_musics.midi_export_types import CHROMATIC_SLOT_NAMES
from code_musics.pieces.registry import PieceDefinition
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.render import RenderWindow, export_piece_midi
from code_musics.score import Score


def _track_messages_with_absolute_ticks(
    midi_path: Path,
) -> list[tuple[int, mido.Message]]:
    midi_file = mido.MidiFile(midi_path)
    absolute_tick = 0
    messages: list[tuple[int, mido.Message]] = []
    for message in midi_file.tracks[0]:
        absolute_tick += message.time
        messages.append((absolute_tick, message))
    return messages


def test_export_midi_bundle_static_piece_writes_expected_midi_contents(
    tmp_path: Path,
) -> None:
    score = Score(f0_hz=220.0)
    score.add_voice("bass", max_polyphony=1)
    score.add_voice("pad")
    score.add_note("bass", start=0.0, duration=1.0, partial=1.0)
    score.add_note("bass", start=1.0, duration=1.0, partial=3 / 2)
    score.add_note("pad", start=0.0, duration=2.0, partial=5 / 4)
    score.add_note("pad", start=0.0, duration=2.0, partial=3 / 2)

    result = export_midi_bundle(
        score,
        tmp_path / "bundle",
        spec=MidiBundleExportSpec(
            piece_name="static_demo",
            output_name="static_demo",
            stem_formats=("scala", "tun", "mpe_48st", "poly_bend_12st"),
        ),
    )

    assert result.manifest.tuning_mode == "static_periodic_tuning"
    assert result.manifest.shared_tuning_status == "exact"
    assert result.manifest.requested_stem_formats == [
        "scala",
        "tun",
        "mpe_48st",
        "poly_bend_12st",
    ]
    assert (result.tuning_dir / "static_demo.scl").exists()
    assert (result.tuning_dir / "static_demo.kbm").exists()
    assert (result.tuning_dir / "static_demo.tun").exists()
    assert (result.stems_dir / "bass_scala.mid").exists()
    assert (result.stems_dir / "bass_tun.mid").exists()
    assert (result.stems_dir / "bass_mpe_48st.mid").exists()
    assert (result.stems_dir / "bass_poly_bend_12st.mid").exists()
    assert not (result.stems_dir / "bass_mono_bend_12st.mid").exists()

    bass_scala_messages = _track_messages_with_absolute_ticks(
        result.stems_dir / "bass_scala.mid"
    )
    bass_scala_note_on = [
        (tick, message.note)
        for tick, message in bass_scala_messages
        if message.type == "note_on"
    ]
    assert bass_scala_note_on == [(0, 60), (960, 62)]

    bass_mpe_messages = _track_messages_with_absolute_ticks(
        result.stems_dir / "bass_mpe_48st.mid"
    )
    bass_mpe_note_on = [
        (tick, message.channel, message.note)
        for tick, message in bass_mpe_messages
        if message.type == "note_on"
    ]
    bass_mpe_pitchwheel = [
        (tick, message.channel)
        for tick, message in bass_mpe_messages
        if message.type == "pitchwheel"
    ]
    assert bass_mpe_note_on == [(0, 1, 57), (960, 1, 64)]
    assert bass_mpe_pitchwheel == [(0, 1), (960, 1)]


def test_export_midi_bundle_dynamic_piece_writes_warning_approx_tuning(
    tmp_path: Path,
) -> None:
    score = Score(f0_hz=220.0)
    score.add_voice("melody", max_polyphony=1)
    for index in range(30):
        cents = float(index * 11.0)
        freq_hz = 220.0 * (2.0 ** (cents / 1200.0))
        score.add_note(
            "melody",
            start=index * 0.25,
            duration=0.2,
            freq=freq_hz,
        )

    result = export_midi_bundle(
        score,
        tmp_path / "bundle",
        spec=MidiBundleExportSpec(
            piece_name="dynamic_demo",
            output_name="dynamic_demo",
            stem_formats=("mpe_48st", "poly_bend_12st"),
        ),
    )

    assert result.manifest.tuning_mode == "exact_note_tuning"
    assert result.manifest.shared_tuning_status == "approximate"
    assert result.manifest.requested_stem_formats == ["mpe_48st", "poly_bend_12st"]
    assert (result.tuning_dir / "dynamic_demo_WARNING_APPROX.scl").exists()
    assert (result.tuning_dir / "dynamic_demo_WARNING_APPROX.kbm").exists()
    assert (result.tuning_dir / "dynamic_demo_WARNING_APPROX.tun").exists()
    assert (result.stems_dir / "melody_mpe_48st.mid").exists()
    assert (result.stems_dir / "melody_poly_bend_12st.mid").exists()
    assert not (result.stems_dir / "melody_scala.mid").exists()

    manifest_payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["warning_suffix"] == "WARNING_APPROX"


def test_export_midi_bundle_rejects_pitch_motion(tmp_path: Path) -> None:
    score = Score(f0_hz=220.0)
    score.add_voice("lead")
    score.add_note(
        "lead",
        start=0.0,
        duration=1.0,
        partial=1.0,
        pitch_motion=PitchMotionSpec.linear_bend(target_partial=3 / 2),
    )

    with pytest.raises(ValueError, match="pitch_motion"):
        export_midi_bundle(
            score,
            tmp_path / "bundle",
            spec=MidiBundleExportSpec(
                piece_name="pitch_motion_demo", output_name="pitch_motion_demo"
            ),
        )


def test_export_midi_bundle_rejects_pitch_ratio_automation(tmp_path: Path) -> None:
    score = Score(f0_hz=220.0)
    score.add_voice(
        "lead",
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="pitch_ratio", name="pitch_ratio"),
                default_value=1.0,
                segments=(
                    AutomationSegment(
                        start=0.0,
                        end=1.0,
                        shape="linear",
                        start_value=1.0,
                        end_value=1.5,
                    ),
                ),
            )
        ],
    )
    score.add_note("lead", start=0.0, duration=1.0, partial=1.0)

    with pytest.raises(ValueError, match="pitch_ratio automation"):
        export_midi_bundle(
            score,
            tmp_path / "bundle",
            spec=MidiBundleExportSpec(
                piece_name="pitch_ratio_demo", output_name="pitch_ratio_demo"
            ),
        )


def test_export_midi_bundle_fails_fast_for_requested_overlapping_mono_export(
    tmp_path: Path,
) -> None:
    score = Score(f0_hz=220.0)
    score.add_voice("pad")
    score.add_note("pad", start=0.0, duration=2.0, partial=5 / 4)
    score.add_note("pad", start=0.5, duration=2.0, partial=3 / 2)

    with pytest.raises(ValueError, match="mono_bend_12st"):
        export_midi_bundle(
            score,
            tmp_path / "bundle",
            spec=MidiBundleExportSpec(
                piece_name="mono_fail",
                output_name="mono_fail",
                stem_formats=("mono_bend_12st",),
            ),
        )


def test_export_midi_bundle_fails_fast_for_requested_channel_overflow(
    tmp_path: Path,
) -> None:
    score = Score(f0_hz=220.0)
    score.add_voice("cluster")
    for index in range(16):
        score.add_note(
            "cluster",
            start=0.0,
            duration=1.0,
            freq=220.0 * (2.0 ** (index / 24.0)),
        )

    with pytest.raises(ValueError, match="mpe_48st supports at most 15"):
        export_midi_bundle(
            score,
            tmp_path / "bundle",
            spec=MidiBundleExportSpec(
                piece_name="cluster",
                output_name="cluster",
                stem_formats=("mpe_48st",),
            ),
        )


def test_export_midi_bundle_selected_formats_allow_large_polyphony(
    tmp_path: Path,
) -> None:
    score = Score(f0_hz=220.0)
    score.add_voice("cluster")
    for index in range(16):
        score.add_note(
            "cluster",
            start=0.0,
            duration=1.0,
            freq=220.0 * (2.0 ** (index / 24.0)),
        )

    result = export_midi_bundle(
        score,
        tmp_path / "bundle",
        spec=MidiBundleExportSpec(
            piece_name="cluster",
            output_name="cluster",
            stem_formats=("scala", "tun"),
        ),
    )

    assert result.manifest.requested_stem_formats == ["scala", "tun"]
    assert (result.stems_dir / "cluster_scala.mid").exists()
    assert (result.stems_dir / "cluster_tun.mid").exists()
    assert not (result.stems_dir / "cluster_mpe_48st.mid").exists()


def test_sequential_chords_at_same_tick_do_not_trigger_false_channel_overflow(
    tmp_path: Path,
) -> None:
    score = Score(f0_hz=220.0)
    score.add_voice("chords")
    for index in range(15):
        score.add_note(
            "chords",
            start=0.0,
            duration=1.0,
            freq=220.0 * (2.0 ** (index / 36.0)),
        )
    for index in range(15):
        score.add_note(
            "chords",
            start=1.0,
            duration=1.0,
            freq=220.0 * (2.0 ** ((index + 1) / 36.0)),
        )

    result = export_midi_bundle(
        score,
        tmp_path / "bundle",
        spec=MidiBundleExportSpec(
            piece_name="chords",
            output_name="chords",
            stem_formats=("mpe_48st",),
        ),
    )

    assert (result.stems_dir / "chords_mpe_48st.mid").exists()


def test_shared_tuning_mapping_wraps_to_next_period_unison() -> None:
    tuning_analysis = TuningAnalysisResult(
        tuning_mode="static_periodic_tuning",
        is_approximate=False,
        period_ratio=2.0,
        period_cents=1200.0,
        reference_midi_note=60,
        reference_frequency_hz=220.0,
        pitch_class_cents=(0.0, 700.0),
        scale_entry_cents=(700.0, 1200.0),
        quantization_cents=0.01,
    )
    near_period_note = MidiStemNote(
        voice_name="lead",
        note_index=0,
        start_seconds=0.0,
        duration_seconds=1.0,
        end_seconds=1.0,
        freq_hz=220.0 * (2.0 ** (1199.9 / 1200.0)),
        velocity=96,
        label=None,
    )

    assert (
        resolve_shared_tuning_midi_note(
            note=near_period_note,
            tuning_analysis=tuning_analysis,
        )
        == 62
    )


def test_collect_stem_notes_clips_window_and_uses_full_score_timing_context() -> None:
    score = Score(
        f0_hz=220.0,
        timing_humanize=TimingHumanizeSpec(
            preset="tight_ensemble",
            seed=17,
            micro_jitter_ms=0.0,
            chord_spread_ms=0.0,
        ),
    )
    score.add_voice("lead")
    score.add_note("lead", start=1.0, duration=2.5, partial=1.0)
    score.add_note("lead", start=2.4, duration=1.0, partial=3 / 2)

    timing_offsets = score.resolve_timing_offsets()
    first_note_offset = timing_offsets[("lead", 0)]
    first_note_start = 1.0 + first_note_offset
    first_note_end = first_note_start + 2.5

    stem_notes = collect_stem_notes(
        score,
        window_start_seconds=2.0,
        window_end_seconds=4.0,
    )

    lead_notes = stem_notes["lead"]
    assert len(lead_notes) == 2
    assert lead_notes[0].start_seconds == pytest.approx(0.0)
    assert lead_notes[0].duration_seconds == pytest.approx(
        min(4.0, first_note_end) - max(2.0, first_note_start)
    )


def test_export_piece_midi_snippet_uses_snippet_suffix_and_clips_crossing_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build_score() -> Score:
        score = Score(f0_hz=220.0)
        score.add_voice("lead", max_polyphony=1)
        score.add_note("lead", start=0.0, duration=2.0, partial=1.0)
        score.add_note("lead", start=1.5, duration=1.0, partial=3 / 2)
        return score

    piece_definition = PieceDefinition(
        name="test_midi_piece",
        output_name="test_midi_piece",
        build_score=build_score,
    )
    monkeypatch.setitem(render_module.PIECES, "test_midi_piece", piece_definition)

    result = export_piece_midi(
        "test_midi_piece",
        output_dir=tmp_path,
        render_window=RenderWindow(start_seconds=1.0, duration_seconds=1.0),
        stem_formats=("scala",),
    )

    assert result.bundle_dir.exists()
    assert "__snippet_" in result.bundle_dir.name
    lead_messages = _track_messages_with_absolute_ticks(
        result.stems_dir / "lead_scala.mid"
    )
    lead_note_on = [
        (tick, message.note)
        for tick, message in lead_messages
        if message.type == "note_on"
    ]
    lead_note_off = [
        (tick, message.note)
        for tick, message in lead_messages
        if message.type == "note_off"
    ]
    assert lead_note_on == [(0, 60), (480, 61)]
    assert lead_note_off == [(960, 60), (960, 61)]


# ---------------------------------------------------------------------------
# Chromatic-fill SCL tests
# ---------------------------------------------------------------------------


class TestAssignChromaticSlots:
    def test_seven_limit_ji_12_notes_no_conflicts(self) -> None:
        """Full 7-limit JI chromatic: all 12 slots filled, no nearest-neighbor fill."""
        # 1/1, 16/15, 9/8, 6/5, 5/4, 4/3, 7/5, 3/2, 8/5, 5/3, 7/4, 15/8
        ji_7limit_cents = (
            0.0,
            111.73,
            203.91,
            315.64,
            386.31,
            498.04,
            582.51,
            701.96,
            813.69,
            884.36,
            968.83,
            1088.27,
        )
        assignments = assign_chromatic_slots(ji_7limit_cents)
        assert len(assignments) == 12
        slot_names = [a.slot_name for a in assignments]
        assert slot_names == list(CHROMATIC_SLOT_NAMES)
        # All from scale
        assert all(a.source == "scale" for a in assignments)
        # Max error should be 7/4 → Bb (~31.2c)
        max_error = max(a.error_cents for a in assignments)
        assert max_error == pytest.approx(31.17, abs=0.5)

    def test_colundi_7_notes_with_fill(self) -> None:
        """Colundi 7-note scale: 7 occupied + 5 nearest-neighbor filled."""
        # 1/1, 11/10, 19/16, 4/3, 3/2, 49/30, 7/4
        colundi_cents = (0.0, 165.00, 297.51, 498.04, 701.96, 849.21, 968.83)
        raw = assign_chromatic_slots(colundi_cents)
        filled = _nearest_neighbor_fill(raw)
        assert len(filled) == 12
        scale_slots = [a for a in filled if a.source == "scale"]
        fill_slots = [a for a in filled if a.source.startswith("fill_from_")]
        assert len(scale_slots) == 7
        assert len(fill_slots) == 5

    def test_simple_triad_nearest_neighbor_fill(self) -> None:
        """A simple major triad: 1/1 (0c), 5/4 (386c), 3/2 (702c)."""
        triad_cents = (0.0, 386.31, 701.96)
        raw = assign_chromatic_slots(triad_cents)
        filled = _nearest_neighbor_fill(raw)
        assert len(filled) == 12
        # Slots 0 (C), 4 (E), 7 (G) should be from scale
        scale_slots = {a.slot for a in filled if a.source == "scale"}
        assert scale_slots == {0, 4, 7}
        # All other slots should be filled from nearest
        for a in filled:
            if a.slot not in scale_slots:
                assert a.source.startswith("fill_from_")

    def test_conflict_two_tones_same_slot(self) -> None:
        """5/4 (386c) and 81/64 (408c) both want E (400c). Closer one wins."""
        cents = (0.0, 386.31, 407.82)
        raw = assign_chromatic_slots(cents)
        filled = _nearest_neighbor_fill(raw)
        # 81/64 (408c) is closer to E=400c (8c) than 5/4 (386c, 14c)
        e_slot = next(a for a in filled if a.slot == 4)
        assert e_slot.cents == pytest.approx(407.82)
        assert e_slot.source == "scale"
        # 5/4 (386c) should be bumped to nearest available — Eb=300c or D#
        bumped = next(
            a for a in filled if abs(a.cents - 386.31) < 0.1 and a.source == "scale"
        )
        assert bumped.slot != 4


class TestChromaticSclIntegration:
    def test_chromatic_files_emitted_for_static_piece(self, tmp_path: Path) -> None:
        """Full integration: export bundle produces chromatic SCL + KBM."""
        score = Score(f0_hz=220.0)
        score.add_voice("bass", max_polyphony=1)
        score.add_note("bass", start=0.0, duration=1.0, partial=1.0)
        score.add_note("bass", start=1.0, duration=1.0, partial=3 / 2)
        score.add_note("bass", start=2.0, duration=1.0, partial=5 / 4)

        result = export_midi_bundle(
            score,
            tmp_path / "bundle",
            spec=MidiBundleExportSpec(
                piece_name="chromatic_test",
                output_name="chromatic_test",
                stem_formats=("scala",),
            ),
        )

        assert (result.tuning_dir / "chromatic_test_chromatic.scl").exists()
        assert (result.tuning_dir / "chromatic_test_chromatic.kbm").exists()
        # Manifest should have chromatic_tuning
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert "chromatic_tuning" in manifest
        assert manifest["chromatic_tuning"]["scl"] is not None
        # SCL should have 12 entries (header says 12)
        scl_text = (result.tuning_dir / "chromatic_test_chromatic.scl").read_text()
        scl_lines = [
            line for line in scl_text.strip().split("\n") if not line.startswith("!")
        ]
        # First non-comment line is description, second is count
        assert scl_lines[1] == "12"

    def test_chromatic_scl_skipped_for_many_pitch_classes(self, tmp_path: Path) -> None:
        """More than 12 pitch classes: chromatic SCL skipped."""
        score = Score(f0_hz=220.0)
        score.add_voice("melody", max_polyphony=1)
        for i in range(15):
            freq = 220.0 * (2.0 ** (i * 80.0 / 1200.0))
            score.add_note("melody", start=i * 0.5, duration=0.4, freq=freq)

        result = export_midi_bundle(
            score,
            tmp_path / "bundle",
            spec=MidiBundleExportSpec(
                piece_name="many_pc",
                output_name="many_pc",
                stem_formats=("mpe_48st",),
            ),
        )

        assert not (result.tuning_dir / "many_pc_chromatic.scl").exists()
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert "skipped_reason" in manifest["chromatic_tuning"]

    def test_chromatic_scl_skipped_for_non_octave_period(self, tmp_path: Path) -> None:
        """Non-octave period: chromatic SCL skipped."""
        score = Score(f0_hz=220.0)
        score.add_voice("lead", max_polyphony=1)
        score.add_note("lead", start=0.0, duration=1.0, partial=1.0)

        result = export_midi_bundle(
            score,
            tmp_path / "bundle",
            spec=MidiBundleExportSpec(
                piece_name="tritave",
                output_name="tritave",
                period_ratio=3.0,
                stem_formats=("scala",),
            ),
        )

        assert not (result.tuning_dir / "tritave_chromatic.scl").exists()
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert "non-octave" in manifest["chromatic_tuning"]["skipped_reason"]

    def test_chromatic_scl_disabled_via_spec(self, tmp_path: Path) -> None:
        """chromatic_scl=False disables generation."""
        score = Score(f0_hz=220.0)
        score.add_voice("lead", max_polyphony=1)
        score.add_note("lead", start=0.0, duration=1.0, partial=1.0)

        result = export_midi_bundle(
            score,
            tmp_path / "bundle",
            spec=MidiBundleExportSpec(
                piece_name="disabled",
                output_name="disabled",
                stem_formats=("scala",),
                chromatic_scl=False,
            ),
        )

        assert not (result.tuning_dir / "disabled_chromatic.scl").exists()
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        assert "disabled" in manifest["chromatic_tuning"]["skipped_reason"]

    def test_chromatic_warnings_above_threshold(self, tmp_path: Path) -> None:
        """Colundi-like scale produces warnings for large slot errors."""
        score = Score(f0_hz=220.0)
        score.add_voice("pad")
        # 49/30 = 849.2c → Ab(800c) = 49.2c error, above 35c threshold
        ratios = [1.0, 11 / 10, 19 / 16, 4 / 3, 3 / 2, 49 / 30, 7 / 4]
        for i, r in enumerate(ratios):
            score.add_note("pad", start=i * 1.0, duration=0.9, partial=r)

        result = export_midi_bundle(
            score,
            tmp_path / "bundle",
            spec=MidiBundleExportSpec(
                piece_name="colundi",
                output_name="colundi",
                stem_formats=("scala",),
            ),
        )

        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        chromatic = manifest["chromatic_tuning"]
        assert len(chromatic["warnings"]) > 0
        # 49/30 should trigger a warning
        assert any("Ab" in w for w in chromatic["warnings"])
