"""Command-line entrypoint for rendering registered pieces."""

import argparse
import logging

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
    return parser.parse_args()


def main() -> None:
    """Run the piece renderer CLI."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    if args.list or args.piece is None:
        for piece_name in list_pieces():
            print(piece_name)
        return

    piece_names = list_pieces() if args.piece == "all" else [args.piece]
    for piece_name in piece_names:
        render_piece(piece_name, save_plot=args.plot)


if __name__ == "__main__":
    main()
