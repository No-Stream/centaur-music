"""Deterministic headless-Chrome -> ffmpeg video capture driver.

Drives a browser scene (loaded via viz/lib/frame_driver.js) frame-by-frame
through Playwright, pipes PNG screenshots into per-shard ffmpeg encodes
running in parallel worker processes, concatenates the shard segments, and
optionally muxes in an audio track.

Playwright is imported lazily inside the functions that actually launch a
browser so this module (and its pure helper functions) can be imported and
unit-tested before the `viz-setup` bootstrap step has installed it.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread

logger: logging.Logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FFMPEG_PATH = REPO_ROOT / "tools" / "ffmpeg" / "ffmpeg"

CHROME_LAUNCH_ARGS = [
    "--force-color-profile=srgb",
    "--hide-scrollbars",
    "--disable-lcd-text",
    "--enable-unsafe-swiftshader",
]


@dataclass(frozen=True)
class CaptureJob:
    """Fully-resolved parameters for one capture run."""

    scene_dir: Path
    viz_json_path: Path | None
    audio_path: Path | None
    width: int
    height: int
    fps: int
    frame_start: int
    frame_end: int  # exclusive
    workers: int
    output_path: Path
    ffmpeg_path: Path
    crf: int = 17
    audio_bitrate: str = "320k"


def shard_frame_ranges(
    frame_start: int, frame_end: int, workers: int
) -> list[tuple[int, int]]:
    """Split [frame_start, frame_end) into up to `workers` contiguous shards.

    Coverage is complete and non-overlapping. When there are fewer frames
    than workers, returns fewer, single-frame shards rather than empty ones.
    """
    total_frames = frame_end - frame_start
    if total_frames <= 0:
        raise ValueError(
            f"shard_frame_ranges: empty or invalid range [{frame_start}, {frame_end})"
        )

    shard_count = min(workers, total_frames)
    base_size, remainder = divmod(total_frames, shard_count)

    shards: list[tuple[int, int]] = []
    cursor = frame_start
    for shard_index in range(shard_count):
        this_size = base_size + (1 if shard_index < remainder else 0)
        shard_end = cursor + this_size
        shards.append((cursor, shard_end))
        cursor = shard_end

    return shards


def build_segment_encode_cmd(
    ffmpeg: Path, fps: int, segment_path: Path, crf: int
) -> list[str]:
    """ffmpeg command that reads PNGs from stdin (image2pipe) and encodes an
    x264 segment."""
    return [
        str(ffmpeg),
        "-y",
        "-f",
        "image2pipe",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(segment_path),
    ]


def build_concat_cmd(ffmpeg: Path, concat_list: Path, out: Path) -> list[str]:
    """ffmpeg concat-demuxer command that stream-copies shard segments into
    one video file."""
    return [
        str(ffmpeg),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c",
        "copy",
        str(out),
    ]


def build_mux_cmd(
    ffmpeg: Path,
    video: Path,
    audio: Path,
    audio_offset_seconds: float,
    out: Path,
    audio_bitrate: str,
) -> list[str]:
    """ffmpeg command that muxes a stream-copied video against an
    offset-trimmed, AAC-encoded audio track."""
    return [
        str(ffmpeg),
        "-y",
        "-i",
        str(video),
        "-ss",
        str(audio_offset_seconds),
        "-i",
        str(audio),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        "-shortest",
        str(out),
    ]


def _scene_page_url(port: int, scene_dir: Path, viz_json_path: Path | None) -> str:
    scene_relative = scene_dir.resolve().relative_to(REPO_ROOT)
    url = f"http://127.0.0.1:{port}/{scene_relative}/index.html"
    if viz_json_path is not None:
        viz_relative = viz_json_path.resolve().relative_to(REPO_ROOT)
        url = f"{url}?viz=/{viz_relative}"
    return url


class _RepoRootHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)  # type: ignore[arg-type]

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        # Silence per-request access logging; capture progress is logged
        # separately via the `logging` module.
        pass


def _start_repo_http_server() -> tuple[ThreadingHTTPServer, int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RepoRootHTTPRequestHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    return server, port


def capture_segment(
    scene_dir: Path,
    viz_json_path: Path | None,
    width: int,
    height: int,
    fps: int,
    frame_start: int,
    frame_end: int,
    ffmpeg_path: Path,
    crf: int,
    segment_path: Path,
) -> Path:
    """Worker entry point: launch one Chrome page + one ffmpeg encode
    process, render frames [frame_start, frame_end) and pipe PNG screenshots
    into ffmpeg's stdin. Returns the encoded segment path."""
    from playwright.sync_api import sync_playwright  # noqa: PLC0415

    server, port = _start_repo_http_server()
    try:
        page_url = _scene_page_url(port, scene_dir, viz_json_path)

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                channel="chrome", args=CHROME_LAUNCH_ARGS
            )
            try:
                page = browser.new_page(
                    viewport={"width": width, "height": height},
                    device_scale_factor=1,
                )
                page.goto(page_url)
                page.wait_for_function(
                    "window.__viz && window.__viz.ready || window.__viz_error"
                )

                error = page.evaluate("window.__viz_error || null")
                if error is not None:
                    raise RuntimeError(
                        f"scene failed to initialize: {error.get('message')}\n{error.get('stack', '')}"
                    )

                encode_cmd = build_segment_encode_cmd(
                    ffmpeg_path, fps, segment_path, crf
                )
                encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)
                assert encoder.stdin is not None

                try:
                    frame_count = frame_end - frame_start
                    for offset in range(frame_count):
                        frame_index = frame_start + offset
                        page.evaluate(
                            "([i, fps]) => window.__viz.renderFrame(i, fps)",
                            [frame_index, fps],
                        )
                        png_bytes = page.screenshot(type="png")
                        encoder.stdin.write(png_bytes)

                        if offset % 50 == 0:
                            logger.info(
                                "shard [%d, %d): rendered frame %d/%d",
                                frame_start,
                                frame_end,
                                offset + 1,
                                frame_count,
                            )
                finally:
                    encoder.stdin.close()
                    returncode = encoder.wait()

                if returncode != 0:
                    raise RuntimeError(
                        f"ffmpeg segment encode failed with exit code {returncode}: {encode_cmd}"
                    )
            finally:
                browser.close()
    finally:
        server.shutdown()

    return segment_path


def run_capture(job: CaptureJob) -> Path:
    """Shard the frame range across worker processes, encode each shard,
    concatenate the segments, and optionally mux in audio. Returns the final
    output path."""
    shards = shard_frame_ranges(job.frame_start, job.frame_end, job.workers)
    logger.info(
        "capturing %d frames across %d shard(s)",
        job.frame_end - job.frame_start,
        len(shards),
    )

    with TemporaryDirectory(prefix="viz_capture_") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        segment_paths: list[Path | None] = [None] * len(shards)

        with ProcessPoolExecutor(max_workers=len(shards)) as executor:
            future_to_index = {}
            for shard_index, (shard_start, shard_end) in enumerate(shards):
                segment_path = tmp_dir / f"segment_{shard_index:04d}.mp4"
                future = executor.submit(
                    capture_segment,
                    job.scene_dir,
                    job.viz_json_path,
                    job.width,
                    job.height,
                    job.fps,
                    shard_start,
                    shard_end,
                    job.ffmpeg_path,
                    job.crf,
                    segment_path,
                )
                future_to_index[future] = shard_index

            for future in as_completed(future_to_index):
                shard_index = future_to_index[future]
                segment_paths[shard_index] = future.result()

        resolved_segment_paths = [path for path in segment_paths if path is not None]
        assert len(resolved_segment_paths) == len(shards)

        concat_list = tmp_dir / "concat_list.txt"
        concat_list.write_text(
            "".join(f"file '{path}'\n" for path in resolved_segment_paths),
            encoding="utf-8",
        )

        job.output_path.parent.mkdir(parents=True, exist_ok=True)

        if job.audio_path is None:
            concat_cmd = build_concat_cmd(job.ffmpeg_path, concat_list, job.output_path)
            _run_ffmpeg(concat_cmd)
            return job.output_path

        concatenated_video = tmp_dir / "concatenated.mp4"
        concat_cmd = build_concat_cmd(job.ffmpeg_path, concat_list, concatenated_video)
        _run_ffmpeg(concat_cmd)

        audio_offset_seconds = job.frame_start / job.fps
        mux_cmd = build_mux_cmd(
            job.ffmpeg_path,
            concatenated_video,
            job.audio_path,
            audio_offset_seconds,
            job.output_path,
            job.audio_bitrate,
        )
        _run_ffmpeg(mux_cmd)
        return job.output_path


def _run_ffmpeg(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg command failed with exit code {result.returncode}: {cmd}\n{result.stderr}"
        )


def _read_total_duration_seconds(viz_json_path: Path) -> float:
    payload = json.loads(viz_json_path.read_text(encoding="utf-8"))
    if "total_duration_seconds" not in payload:
        raise ValueError(
            f"{viz_json_path}: missing 'total_duration_seconds' key required to infer --end-frame"
        )
    return float(payload["total_duration_seconds"])


def _preflight_check(job: CaptureJob) -> None:
    if not job.ffmpeg_path.exists() or not job.ffmpeg_path.is_file():
        raise FileNotFoundError(
            f"ffmpeg binary not found at {job.ffmpeg_path}. Run `make viz-setup` to install it."
        )

    scene_index = job.scene_dir / "index.html"
    if not scene_index.exists():
        raise FileNotFoundError(f"scene index.html not found at {scene_index}")

    if job.viz_json_path is not None and not job.viz_json_path.exists():
        raise FileNotFoundError(f"viz json not found at {job.viz_json_path}")

    if job.audio_path is not None and not job.audio_path.exists():
        raise FileNotFoundError(f"audio file not found at {job.audio_path}")

    if job.frame_end <= job.frame_start:
        raise ValueError(
            f"frame_end ({job.frame_end}) must be greater than frame_start ({job.frame_start})"
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", required=True, type=Path)
    parser.add_argument("--viz-json", type=Path, default=None)
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--start-seconds", type=float, default=None)
    parser.add_argument("--end-seconds", type=float, default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--crf", type=int, default=17)
    parser.add_argument("--audio-bitrate", default="320k")
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG_PATH)
    return parser


def _resolve_frame_range(args: argparse.Namespace) -> tuple[int, int]:
    if args.start_seconds is not None:
        frame_start = round(args.start_seconds * args.fps)
    elif args.start_frame is not None:
        frame_start = args.start_frame
    else:
        frame_start = 0

    if args.end_seconds is not None:
        frame_end = round(args.end_seconds * args.fps)
    elif args.end_frame is not None:
        frame_end = args.end_frame
    elif args.viz_json is not None:
        total_duration_seconds = _read_total_duration_seconds(args.viz_json)
        frame_end = round(total_duration_seconds * args.fps)
    else:
        raise ValueError(
            "no end point given: pass --end-seconds, --end-frame, or --viz-json "
            "(to infer the end from total_duration_seconds)"
        )

    return frame_start, frame_end


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    frame_start, frame_end = _resolve_frame_range(args)

    job = CaptureJob(
        scene_dir=args.scene_dir,
        viz_json_path=args.viz_json,
        audio_path=args.audio,
        width=args.width,
        height=args.height,
        fps=args.fps,
        frame_start=frame_start,
        frame_end=frame_end,
        workers=args.workers,
        output_path=args.output,
        ffmpeg_path=args.ffmpeg,
        crf=args.crf,
        audio_bitrate=args.audio_bitrate,
    )

    _preflight_check(job)

    output_path = run_capture(job)
    logger.info("wrote %s", output_path)


if __name__ == "__main__":
    main()
