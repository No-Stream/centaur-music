"""Named piece rendering workflow."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from code_musics.pieces import PIECES
from code_musics.synth import write_wav

logger = logging.getLogger(__name__)


def list_pieces() -> list[str]:
    """Return the registered piece names."""
    return sorted(PIECES)


def render_piece(
    piece_name: str,
    *,
    output_dir: str | Path = "output",
    save_plot: bool = False,
) -> tuple[Path, Path | None]:
    """Render a registered piece by name."""
    if piece_name not in PIECES:
        raise ValueError(f"Unknown piece: {piece_name}")

    definition = PIECES[piece_name]
    output_path = Path(output_dir) / definition.output_name
    plot_path: Path | None = None

    if definition.build_score is not None:
        score = definition.build_score()
        audio = score.render()
        if save_plot:
            plot_path = output_path.with_suffix(".png")
            figure, _ = score.plot_piano_roll(plot_path)
            plt.close(figure)
    elif definition.render_audio is not None:
        audio = definition.render_audio()
    else:
        raise ValueError(f"Piece {piece_name} has no render path configured")

    write_wav(output_path, audio)
    logger.info("Rendered %s", piece_name)
    return output_path, plot_path
