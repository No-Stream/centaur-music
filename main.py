"""Command-line entrypoint for rendering registered pieces."""

import argparse
import logging

from code_musics.inspection import (
    format_inspection_summary,
    inspect_piece_timestamp,
    parse_timestamp_seconds,
)
from code_musics.render import list_pieces, render_piece


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


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

    piece_names = list_pieces() if args.piece == "all" else [args.piece]
    for piece_name in piece_names:
        result = render_piece(
            piece_name,
            save_plot=args.plot,
            save_analysis=not args.no_analysis,
        )
        if result.analysis_manifest_path is not None:
            logging.info("Saved analysis manifest to %s", result.analysis_manifest_path)


if __name__ == "__main__":
    main()
