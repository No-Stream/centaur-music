"""CLI argument parsing tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import main as main_module
from code_musics.midi_export import ALL_STEM_FORMATS
from main import _build_render_window, _parse_midi_formats, parse_args


def test_build_render_window_from_centered_snippet_args() -> None:
    args = parse_args(["chord_4567", "--snippet-at", "2:10", "--snippet-window", "12"])

    render_window = _build_render_window(args)

    assert render_window is not None
    assert render_window.start_seconds == pytest.approx(124.0)
    assert render_window.duration_seconds == pytest.approx(12.0)


def test_build_render_window_from_explicit_window_args() -> None:
    args = parse_args(["chord_4567", "--window-start", "1:30", "--window-dur", "4.5"])

    render_window = _build_render_window(args)

    assert render_window is not None
    assert render_window.start_seconds == pytest.approx(90.0)
    assert render_window.duration_seconds == pytest.approx(4.5)


def test_build_render_window_rejects_mixed_snippet_modes() -> None:
    args = parse_args(
        [
            "chord_4567",
            "--snippet-at",
            "2:10",
            "--window-start",
            "130",
            "--window-dur",
            "8",
        ]
    )

    with pytest.raises(ValueError, match="Use either --snippet-at/--snippet-window"):
        _build_render_window(args)


def test_build_render_window_requires_window_pair() -> None:
    args = parse_args(["chord_4567", "--window-start", "130"])

    with pytest.raises(ValueError, match="must be provided together"):
        _build_render_window(args)


def test_build_render_window_requires_snippet_at_when_window_is_overridden() -> None:
    args = parse_args(["chord_4567", "--snippet-window", "8"])

    with pytest.raises(ValueError, match="requires --snippet-at"):
        _build_render_window(args)


def test_parse_args_supports_export_midi_and_midi_formats() -> None:
    args = parse_args(["chord_4567", "--export-midi", "--midi-formats", "scala,tun"])

    assert args.export_midi is True
    assert args.midi_formats == "scala,tun"


def test_parse_midi_formats_defaults_to_all() -> None:
    assert _parse_midi_formats(None) == ALL_STEM_FORMATS


def test_parse_midi_formats_splits_and_strips_values() -> None:
    assert _parse_midi_formats(" scala, tun ,mpe_48st ") == (
        "scala",
        "tun",
        "mpe_48st",
    )


def test_main_passes_selected_midi_formats_to_export(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        main_module,
        "parse_args",
        lambda: argparse.Namespace(
            piece="chord_4567",
            list=False,
            plot=False,
            export_midi=True,
            midi_formats="scala,tun",
            no_analysis=False,
            inspect_at=None,
            inspect_window=8.0,
            snippet_at=None,
            snippet_window=12.0,
            window_start=None,
            window_dur=None,
        ),
    )
    monkeypatch.setattr(main_module, "list_pieces", lambda: ["chord_4567"])

    def fake_export_piece_midi(
        piece_name: str,
        *,
        output_dir: str | Path = "output/midi",
        render_window: object | None = None,
        stem_formats: tuple[str, ...],
    ) -> argparse.Namespace:
        captured["piece_name"] = piece_name
        captured["output_dir"] = output_dir
        captured["render_window"] = render_window
        captured["stem_formats"] = stem_formats
        return argparse.Namespace(manifest_path=tmp_path / "manifest.json")

    monkeypatch.setattr(main_module, "export_piece_midi", fake_export_piece_midi)

    main_module.main()

    assert captured["piece_name"] == "chord_4567"
    assert captured["render_window"] is None
    assert captured["stem_formats"] == ("scala", "tun")
