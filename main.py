"""Command-line entrypoint for rendering registered pieces."""

import argparse
import logging
from typing import cast

from code_musics.inspection import (
    format_inspection_summary,
    inspect_piece_timestamp,
    parse_timestamp_seconds,
)
from code_musics.midi_export import ALL_STEM_FORMATS, MidiStemFormat
from code_musics.render import (
    RenderWindow,
    export_piece_midi,
    list_pieces,
    render_piece,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Render named code-musics pieces.")
    parser.add_argument("piece", nargs="?", help="Registered piece name, or 'all'")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available pieces and exit.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a piano-roll plot when the piece uses the Score abstraction.",
    )
    parser.add_argument(
        "--export-midi",
        action="store_true",
        help="Export a MIDI bundle instead of rendering audio.",
    )
    parser.add_argument(
        "--midi-formats",
        help=(
            "Comma-separated MIDI stem formats to export. "
            f"Choices: {', '.join(ALL_STEM_FORMATS)}"
        ),
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip analysis artifacts and JSON manifest generation.",
    )
    parser.add_argument(
        "--inspect-at",
        help="Inspect what is happening at a timestamp like 130 or 2:10 instead of rendering.",
    )
    parser.add_argument(
        "--inspect-window",
        type=float,
        default=8.0,
        help="Inspection window size in seconds used with --inspect-at.",
    )
    parser.add_argument(
        "--snippet-at",
        help="Render a centered snippet around a timestamp like 130 or 2:10.",
    )
    parser.add_argument(
        "--snippet-window",
        type=float,
        default=12.0,
        help="Snippet duration in seconds used with --snippet-at.",
    )
    parser.add_argument(
        "--window-start",
        help="Render a snippet starting at a timestamp like 130 or 2:10.",
    )
    parser.add_argument(
        "--window-dur",
        type=float,
        help="Snippet duration in seconds used with --window-start.",
    )
    return parser.parse_args(argv)


def _parse_midi_formats(midi_formats: str | None) -> tuple[MidiStemFormat, ...]:
    if midi_formats is None:
        return ALL_STEM_FORMATS
    return tuple(
        cast(MidiStemFormat, stem_format.strip())
        for stem_format in midi_formats.split(",")
        if stem_format.strip()
    )


def _build_render_window(args: argparse.Namespace) -> RenderWindow | None:
    """Return a validated render-window request when snippet flags are present."""
    centered_mode = args.snippet_at is not None or args.snippet_window != 12.0
    explicit_mode = args.window_start is not None or args.window_dur is not None

    if centered_mode and explicit_mode:
        raise ValueError(
            "Use either --snippet-at/--snippet-window or --window-start/--window-dur, not both"
        )
    if args.snippet_at is None and args.snippet_window != 12.0:
        raise ValueError("--snippet-window requires --snippet-at")
    if args.snippet_at is not None:
        if args.snippet_window <= 0:
            raise ValueError("--snippet-window must be positive")
        center_seconds = parse_timestamp_seconds(args.snippet_at)
        start_seconds = max(0.0, center_seconds - (args.snippet_window / 2.0))
        return RenderWindow(
            start_seconds=start_seconds,
            duration_seconds=args.snippet_window,
        )
    if args.window_start is not None or args.window_dur is not None:
        if args.window_start is None or args.window_dur is None:
            raise ValueError(
                "--window-start and --window-dur must be provided together"
            )
        if args.window_dur <= 0:
            raise ValueError("--window-dur must be positive")
        return RenderWindow(
            start_seconds=parse_timestamp_seconds(args.window_start),
            duration_seconds=args.window_dur,
        )
    return None


def main() -> None:
    """Run the piece renderer CLI."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    if args.list or args.piece is None:
        for piece_name in list_pieces():
            print(piece_name)
        return

    if args.inspect_at is not None:
        if args.piece in {None, "all"}:
            raise ValueError("--inspect-at requires a single score-backed piece name")
        inspection = inspect_piece_timestamp(
            piece_name=args.piece,
            timestamp_seconds=parse_timestamp_seconds(args.inspect_at),
            window_seconds=args.inspect_window,
        )
        print(format_inspection_summary(inspection))
        return

    render_window = _build_render_window(args)
    if render_window is not None and args.piece in {None, "all"}:
        raise ValueError("Snippet rendering requires a single piece name")

    piece_names = list_pieces() if args.piece == "all" else [args.piece]
    for piece_name in piece_names:
        if args.export_midi:
            result = export_piece_midi(
                piece_name,
                render_window=render_window,
                stem_formats=_parse_midi_formats(args.midi_formats),
            )
            logging.info("Saved MIDI bundle manifest to %s", result.manifest_path)
            continue

        result = render_piece(
            piece_name,
            save_plot=args.plot,
            save_analysis=not args.no_analysis,
            render_window=render_window,
        )
        if result.analysis_manifest_path is not None:
            logging.info("Saved analysis manifest to %s", result.analysis_manifest_path)


if __name__ == "__main__":
    main()
