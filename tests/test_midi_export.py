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
