"""Named piece rendering workflow."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from code_musics.analysis import save_analysis_artifacts
from code_musics.pieces import PIECES
from code_musics.synth import SAMPLE_RATE, write_wav

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderResult:
    """Paths and metadata emitted by a render."""

    audio_path: Path
    plot_path: Path | None = None
    analysis_manifest_path: Path | None = None
    analysis_artifacts: dict | None = None

    def __iter__(self) -> Iterator[Path | None]:
        """Preserve tuple-style unpacking for legacy callers."""
        yield self.audio_path
        yield self.plot_path


def list_pieces() -> list[str]:
    """Return the registered piece names."""
    return sorted(PIECES)


def render_piece(
    piece_name: str,
    *,
    output_dir: str | Path = "output",
    save_plot: bool = False,
    save_analysis: bool = True,
) -> RenderResult:
    """Render a registered piece by name."""
    if piece_name not in PIECES:
        raise ValueError(f"Unknown piece: {piece_name}")

    definition = PIECES[piece_name]
    output_path = Path(output_dir) / definition.output_name
    plot_path: Path | None = None
    analysis_manifest_path: Path | None = None
    analysis_artifacts: dict | None = None

    if definition.build_score is not None:
        score = definition.build_score()
        stems = score.render_stems()
        audio = score.render()
        if save_plot:
            plot_path = output_path.with_suffix(".png")
            figure, _ = score.plot_piano_roll(plot_path)
            plt.close(figure)
        if save_analysis:
            analysis_artifacts = save_analysis_artifacts(
                output_prefix=output_path.with_suffix(""),
                mix_signal=audio,
                sample_rate=score.sample_rate,
                stems=stems,
                score=score,
            )
            analysis_manifest_path = Path(str(analysis_artifacts["manifest_path"]))
    elif definition.render_audio is not None:
        audio = definition.render_audio()
        if save_analysis:
            analysis_artifacts = save_analysis_artifacts(
                output_prefix=output_path.with_suffix(""),
                mix_signal=audio,
                sample_rate=SAMPLE_RATE,
            )
            analysis_manifest_path = Path(str(analysis_artifacts["manifest_path"]))
    else:
        raise ValueError(f"Piece {piece_name} has no render path configured")

    write_wav(output_path, audio)
    logger.info("Rendered %s", piece_name)
    if analysis_manifest_path is not None:
        logger.info("Analysis manifest: %s", analysis_manifest_path)
    return RenderResult(
        audio_path=output_path,
        plot_path=plot_path,
        analysis_manifest_path=analysis_manifest_path,
        analysis_artifacts=analysis_artifacts,
    )
