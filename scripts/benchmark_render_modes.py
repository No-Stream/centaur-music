"""Benchmark render modes without writing repo-local artifacts."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

from code_musics.pieces import PIECES
from code_musics.render import RenderWindow, render_piece

BenchmarkMode = Literal["none", "full", "summary", "fast", "fast-preview"]


@dataclass(frozen=True)
class BenchmarkResult:
    piece: str
    mode: BenchmarkMode
    repeat_count: int
    warmup_count: int
    elapsed_seconds: list[float]
    median_seconds: float
    min_seconds: float
    max_seconds: float
    peak_rss_gb: float
    audio_path: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pieces",
        nargs="+",
        default=["hexany_garden", "amber_room", "filter_palette_study"],
        choices=sorted(PIECES),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["none", "full", "summary", "fast", "fast-preview"],
        choices=["none", "full", "summary", "fast", "fast-preview"],
    )
    parser.add_argument("--window-start", type=float, default=0.0)
    parser.add_argument("--window-dur", type=float, default=8.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/centaur-render-bench"),
    )
    return parser.parse_args()


def _git_output(args: list[str]) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _max_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def _run_case(
    *,
    piece: str,
    mode: BenchmarkMode,
    output_dir: Path,
    render_window: RenderWindow,
    warmups: int,
    repeats: int,
) -> BenchmarkResult:
    elapsed_values: list[float] = []
    result_path = ""
    total_runs = warmups + repeats
    for run_index in range(total_runs):
        analysis_mode = cast(
            Literal["full", "summary", "fast"],
            "full" if mode == "none" else "fast" if mode == "fast-preview" else mode,
        )
        start = time.perf_counter()
        result = render_piece(
            piece,
            output_dir=output_dir,
            save_plot=False,
            save_analysis=mode != "none",
            analysis_mode=analysis_mode,
            fast_preview=mode == "fast-preview",
            render_window=render_window,
        )
        elapsed = time.perf_counter() - start
        result_path = str(result.audio_path)
        if run_index >= warmups:
            elapsed_values.append(elapsed)

    return BenchmarkResult(
        piece=piece,
        mode=mode,
        repeat_count=repeats,
        warmup_count=warmups,
        elapsed_seconds=elapsed_values,
        median_seconds=float(statistics.median(elapsed_values)),
        min_seconds=float(min(elapsed_values)),
        max_seconds=float(max(elapsed_values)),
        peak_rss_gb=_max_rss_gb(),
        audio_path=result_path,
    )


def main() -> None:
    args = _parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")

    render_window = RenderWindow(
        start_seconds=args.window_start,
        duration_seconds=args.window_dur,
    )
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "created_at": timestamp,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "git_status": _git_output(["status", "--short", "--untracked-files=no"]),
        "thread_env": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "window": {
            "start_seconds": render_window.start_seconds,
            "duration_seconds": render_window.duration_seconds,
        },
    }

    results: list[BenchmarkResult] = []
    for piece in args.pieces:
        for mode_text in args.modes:
            mode = cast(BenchmarkMode, mode_text)
            results.append(
                _run_case(
                    piece=piece,
                    mode=mode,
                    output_dir=output_dir,
                    render_window=render_window,
                    warmups=args.warmups,
                    repeats=args.repeats,
                )
            )

    payload = {
        "metadata": metadata,
        "results": [asdict(result) for result in results],
    }
    result_path = output_dir / "benchmark_render_modes.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
