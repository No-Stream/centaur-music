"""Profile peak RSS for a named piece render without writing artifacts."""

from __future__ import annotations

import argparse
import resource
import time

import numpy as np

from code_musics.pieces import PIECES


def _max_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("piece", choices=sorted(PIECES))
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Profile render_with_effect_analysis instead of the low-memory mix path.",
    )
    args = parser.parse_args()

    definition = PIECES[args.piece]
    start = time.perf_counter()
    if definition.build_score is not None:
        score = definition.build_score()
        if args.analysis:
            audio, stems, sends, _effect_analysis = score.render_with_effect_analysis()
            retained_gb = (
                audio.nbytes
                + sum(stem.nbytes for stem in stems.values())
                + sum(send.nbytes for send in sends.values())
            ) / 1e9
        else:
            audio = score.render()
            retained_gb = audio.nbytes / 1e9
    elif definition.render_audio is not None:
        audio = definition.render_audio()
        retained_gb = audio.nbytes / 1e9
    else:
        raise ValueError(f"Piece {args.piece!r} has no render path configured")

    elapsed = time.perf_counter() - start
    peak = _max_rss_gb()
    peak_dbfs = (
        20.0 * np.log10(max(float(np.max(np.abs(audio))), 1e-12))
        if audio.size
        else float("-inf")
    )
    print(
        f"piece={args.piece} analysis={args.analysis} "
        f"elapsed_s={elapsed:.2f} peak_rss_gb={peak:.2f} "
        f"retained_audio_gb={retained_gb:.3f} peak_dbfs={peak_dbfs:.2f}"
    )


if __name__ == "__main__":
    main()
