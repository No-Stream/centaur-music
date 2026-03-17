"""Compatibility wrapper for the septimal pieces."""

import logging

from code_musics.render import render_piece


def main() -> None:
    """Render all septimal reference pieces."""
    logging.basicConfig(level=logging.INFO)
    for piece_name in ["interval_demo", "chord_4567", "harmonic_drift"]:
        render_piece(piece_name)


if __name__ == "__main__":
    main()
