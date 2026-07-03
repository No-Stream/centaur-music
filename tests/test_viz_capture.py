"""Tests for the viz/capture.py headless-Chrome -> ffmpeg capture pipeline."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from viz.capture import (
    DEFAULT_FFMPEG_PATH,
    CaptureJob,
    _build_arg_parser,
    _preflight_check,
    _read_total_duration_seconds,
    _resolve_frame_range,
    build_concat_cmd,
    build_mux_cmd,
    build_segment_encode_cmd,
    run_capture,
    shard_frame_ranges,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SMOKE_SCENE_DIR = REPO_ROOT / "viz" / "_smoke"


class TestShardFrameRanges:
    def test_even_division_covers_full_range_without_overlap(self) -> None:
        shards = shard_frame_ranges(0, 100, 4)
        assert shards[0][0] == 0
        assert shards[-1][1] == 100
        for i in range(len(shards) - 1):
            assert shards[i][1] == shards[i + 1][0]

    def test_uneven_division_distributes_remainder(self) -> None:
        shards = shard_frame_ranges(0, 10, 3)
        sizes = [end - start for start, end in shards]
        assert sum(sizes) == 10
        assert sorted(sizes) == [3, 3, 4]

    def test_nonzero_start_offset_is_preserved(self) -> None:
        shards = shard_frame_ranges(50, 130, 4)
        assert shards[0][0] == 50
        assert shards[-1][1] == 130

    def test_workers_exceeding_frame_count_yields_fewer_single_frame_shards(
        self,
    ) -> None:
        shards = shard_frame_ranges(0, 3, 10)
        assert len(shards) == 3
        assert all(end - start == 1 for start, end in shards)

    def test_zero_length_range_raises(self) -> None:
        with pytest.raises(ValueError):
            shard_frame_ranges(10, 10, 4)

    def test_inverted_range_raises(self) -> None:
        with pytest.raises(ValueError):
            shard_frame_ranges(10, 5, 4)


class TestCommandBuilders:
    def test_segment_encode_cmd_contains_expected_args_in_order(
        self, tmp_path: Path
    ) -> None:
        segment = tmp_path / "seg_0000.mp4"
        cmd = build_segment_encode_cmd(Path("/bin/ffmpeg"), 30, segment, 17)
        assert cmd == [
            "/bin/ffmpeg",
            "-y",
            "-f",
            "image2pipe",
            "-framerate",
            "30",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            str(segment),
        ]

    def test_concat_cmd_contains_expected_args_in_order(self, tmp_path: Path) -> None:
        concat_list = tmp_path / "concat.txt"
        out = tmp_path / "out.mp4"
        cmd = build_concat_cmd(Path("/bin/ffmpeg"), concat_list, out)
        assert cmd == [
            "/bin/ffmpeg",
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

    def test_mux_cmd_contains_expected_args_in_order(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.wav"
        out = tmp_path / "out.mp4"
        cmd = build_mux_cmd(Path("/bin/ffmpeg"), video, audio, 1.5, out, "320k")
        assert cmd == [
            "/bin/ffmpeg",
            "-y",
            "-i",
            str(video),
            "-ss",
            "1.5",
            "-i",
            str(audio),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "320k",
            "-shortest",
            str(out),
        ]


class TestCliFrameRangeResolution:
    def test_seconds_take_precedence_and_round_to_frames(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--scene-dir",
                "viz/_smoke",
                "--output",
                "out.mp4",
                "--fps",
                "30",
                "--start-seconds",
                "1.0",
                "--end-seconds",
                "2.0",
                "--start-frame",
                "999",
                "--end-frame",
                "999",
            ]
        )
        frame_start, frame_end = _resolve_frame_range(args)
        assert (frame_start, frame_end) == (30, 60)

    def test_explicit_frames_used_when_seconds_absent(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--scene-dir",
                "viz/_smoke",
                "--output",
                "out.mp4",
                "--start-frame",
                "12",
                "--end-frame",
                "48",
            ]
        )
        frame_start, frame_end = _resolve_frame_range(args)
        assert (frame_start, frame_end) == (12, 48)

    def test_default_start_is_zero(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(
            ["--scene-dir", "viz/_smoke", "--output", "out.mp4", "--end-frame", "48"]
        )
        frame_start, _ = _resolve_frame_range(args)
        assert frame_start == 0

    def test_end_inferred_from_viz_json_total_duration(self, tmp_path: Path) -> None:
        viz_json = tmp_path / "viz.json"
        viz_json.write_text(
            json.dumps({"total_duration_seconds": 2.0}), encoding="utf-8"
        )
        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--scene-dir",
                "viz/_smoke",
                "--output",
                "out.mp4",
                "--fps",
                "24",
                "--viz-json",
                str(viz_json),
            ]
        )
        frame_start, frame_end = _resolve_frame_range(args)
        assert (frame_start, frame_end) == (0, 48)

    def test_missing_end_source_raises(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args(["--scene-dir", "viz/_smoke", "--output", "out.mp4"])
        with pytest.raises(ValueError):
            _resolve_frame_range(args)


class TestReadTotalDurationSeconds:
    def test_reads_value(self, tmp_path: Path) -> None:
        viz_json = tmp_path / "viz.json"
        viz_json.write_text(
            json.dumps({"total_duration_seconds": 12.5}), encoding="utf-8"
        )
        assert _read_total_duration_seconds(viz_json) == 12.5

    def test_missing_key_raises(self, tmp_path: Path) -> None:
        viz_json = tmp_path / "viz.json"
        viz_json.write_text(json.dumps({}), encoding="utf-8")
        with pytest.raises(ValueError):
            _read_total_duration_seconds(viz_json)


class TestPreflightCheck:
    def _job(self, tmp_path: Path, **overrides: object) -> CaptureJob:
        defaults: dict[str, object] = dict(
            scene_dir=SMOKE_SCENE_DIR,
            viz_json_path=None,
            audio_path=None,
            width=320,
            height=180,
            fps=12,
            frame_start=0,
            frame_end=12,
            workers=2,
            output_path=tmp_path / "out.mp4",
            ffmpeg_path=DEFAULT_FFMPEG_PATH,
        )
        defaults.update(overrides)
        return CaptureJob(**defaults)  # type: ignore[arg-type]

    def test_missing_ffmpeg_raises_with_setup_hint(self, tmp_path: Path) -> None:
        job = self._job(tmp_path, ffmpeg_path=tmp_path / "no_such_ffmpeg")
        with pytest.raises(FileNotFoundError, match="make viz-setup"):
            _preflight_check(job)

    def test_missing_scene_dir_raises(self, tmp_path: Path) -> None:
        job = self._job(
            tmp_path,
            scene_dir=tmp_path / "no_such_scene",
            ffmpeg_path=tmp_path / "ffmpeg",
        )
        (tmp_path / "ffmpeg").write_bytes(b"")
        os.chmod(tmp_path / "ffmpeg", 0o755)
        with pytest.raises(FileNotFoundError, match="index.html"):
            _preflight_check(job)

    def test_missing_viz_json_raises(self, tmp_path: Path) -> None:
        fake_ffmpeg = tmp_path / "ffmpeg"
        fake_ffmpeg.write_bytes(b"")
        os.chmod(fake_ffmpeg, 0o755)
        job = self._job(
            tmp_path,
            ffmpeg_path=fake_ffmpeg,
            viz_json_path=tmp_path / "missing_viz.json",
        )
        with pytest.raises(FileNotFoundError, match="viz json"):
            _preflight_check(job)

    def test_missing_audio_raises(self, tmp_path: Path) -> None:
        fake_ffmpeg = tmp_path / "ffmpeg"
        fake_ffmpeg.write_bytes(b"")
        os.chmod(fake_ffmpeg, 0o755)
        job = self._job(
            tmp_path, ffmpeg_path=fake_ffmpeg, audio_path=tmp_path / "missing_audio.wav"
        )
        with pytest.raises(FileNotFoundError, match="audio"):
            _preflight_check(job)

    def test_inverted_frame_range_raises(self, tmp_path: Path) -> None:
        fake_ffmpeg = tmp_path / "ffmpeg"
        fake_ffmpeg.write_bytes(b"")
        os.chmod(fake_ffmpeg, 0o755)
        job = self._job(tmp_path, ffmpeg_path=fake_ffmpeg, frame_start=10, frame_end=5)
        with pytest.raises(ValueError, match="frame_end"):
            _preflight_check(job)

    def test_valid_job_passes(self, tmp_path: Path) -> None:
        fake_ffmpeg = tmp_path / "ffmpeg"
        fake_ffmpeg.write_bytes(b"")
        os.chmod(fake_ffmpeg, 0o755)
        job = self._job(tmp_path, ffmpeg_path=fake_ffmpeg)
        _preflight_check(job)  # should not raise


def _playwright_importable() -> bool:
    try:
        import playwright  # noqa: F401
    except ImportError:
        return False
    return True


_SMOKE_READY = (
    os.environ.get("VIZ_SMOKE") == "1"
    and DEFAULT_FFMPEG_PATH.exists()
    and _playwright_importable()
    and Path("/usr/bin/google-chrome").exists()
)


@pytest.mark.skipif(
    not _SMOKE_READY, reason="VIZ_SMOKE=1 + ffmpeg + playwright + chrome required"
)
class TestSmokeCapture:
    def test_captures_smoke_scene_end_to_end(self, tmp_path: Path) -> None:
        output = tmp_path / "smoke.mp4"
        job = CaptureJob(
            scene_dir=SMOKE_SCENE_DIR,
            viz_json_path=None,
            audio_path=None,
            width=320,
            height=180,
            fps=12,
            frame_start=0,
            frame_end=12,
            workers=2,
            output_path=output,
            ffmpeg_path=DEFAULT_FFMPEG_PATH,
        )
        _preflight_check(job)
        result_path = run_capture(job)

        assert result_path == output
        assert output.exists()

        ffprobe_path = DEFAULT_FFMPEG_PATH.parent / "ffprobe"
        probe = subprocess.run(
            [
                str(ffprobe_path),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,width,height,duration",
                "-of",
                "json",
                str(output),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        stream_info = json.loads(probe.stdout)["streams"][0]
        assert int(stream_info["width"]) == 320
        assert int(stream_info["height"]) == 180

        nb_frames = stream_info.get("nb_frames")
        if nb_frames is not None and nb_frames != "N/A":
            assert int(nb_frames) == 12
        else:
            assert abs(float(stream_info["duration"]) - 1.0) < 0.1
