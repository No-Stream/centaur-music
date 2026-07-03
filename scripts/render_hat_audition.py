"""Render a short hi-hat/metallic percussion audition WAV.

This is intentionally an artifact generator, not a scratch script: it writes a
WAV plus a timing map under ``output/hat_auditions`` so the hat voicing can be
auditioned outside the test suite.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from code_musics.engines.drum_voice import render as drum_voice_render
from code_musics.engines.metallic_perc import render as metallic_perc_render
from code_musics.engines.registry import resolve_synth_params
from code_musics.synth import SAMPLE_RATE, write_wav


@dataclass(frozen=True)
class AuditionHit:
    label: str
    engine: str
    preset: str
    freq_hz: float
    duration_s: float
    amp: float = 0.72
    repeat_count: int = 3
    repeat_spacing_s: float = 0.42
    params: dict[str, float | str] | None = None


@dataclass(frozen=True)
class RenderedHit:
    label: str
    engine: str
    preset: str
    freq_hz: float
    start_s: float
    end_s: float
    params: dict[str, float | str] | None


_AUDITION_HITS: tuple[AuditionHit, ...] = (
    AuditionHit(
        label="drum_voice closed_hat stock, dark register",
        engine="drum_voice",
        preset="closed_hat",
        freq_hz=4_600.0,
        duration_s=0.08,
    ),
    AuditionHit(
        label="drum_voice open_hat stock, dark register",
        engine="drum_voice",
        preset="open_hat",
        freq_hz=4_600.0,
        duration_s=0.42,
        repeat_spacing_s=0.72,
    ),
    AuditionHit(
        label="drum_voice closed_hat stock, brighter register",
        engine="drum_voice",
        preset="closed_hat",
        freq_hz=6_400.0,
        duration_s=0.08,
    ),
    AuditionHit(
        label="drum_voice open_hat stock, brighter register",
        engine="drum_voice",
        preset="open_hat",
        freq_hz=6_400.0,
        duration_s=0.42,
        repeat_spacing_s=0.72,
    ),
    AuditionHit(
        label="metallic_perc closed_hat stock",
        engine="metallic_perc",
        preset="closed_hat",
        freq_hz=4_600.0,
        duration_s=0.08,
    ),
    AuditionHit(
        label="metallic_perc open_hat stock",
        engine="metallic_perc",
        preset="open_hat",
        freq_hz=4_600.0,
        duration_s=0.42,
        repeat_spacing_s=0.72,
    ),
    AuditionHit(
        label="drum_voice closed_hat extra noise mix",
        engine="drum_voice",
        preset="closed_hat",
        freq_hz=4_600.0,
        duration_s=0.08,
        params={"metallic_hat_noise_mix": 0.9, "noise_level": 0.52},
    ),
    AuditionHit(
        label="drum_voice open_hat extra noise mix",
        engine="drum_voice",
        preset="open_hat",
        freq_hz=4_600.0,
        duration_s=0.42,
        repeat_spacing_s=0.72,
        params={"metallic_hat_noise_mix": 0.92, "noise_level": 0.7},
    ),
    AuditionHit(
        label="drum_voice pedal_hat old partial voicing",
        engine="drum_voice",
        preset="pedal_hat",
        freq_hz=6_000.0,
        duration_s=0.08,
    ),
    AuditionHit(
        label="drum_voice swept_hat old partial voicing",
        engine="drum_voice",
        preset="swept_hat",
        freq_hz=6_000.0,
        duration_s=0.16,
    ),
    AuditionHit(
        label="drum_voice 808_closed_hat square partials",
        engine="drum_voice",
        preset="808_closed_hat",
        freq_hz=6_000.0,
        duration_s=0.08,
    ),
    AuditionHit(
        label="drum_voice 808_open_hat square partials",
        engine="drum_voice",
        preset="808_open_hat",
        freq_hz=6_000.0,
        duration_s=0.42,
        repeat_spacing_s=0.72,
    ),
    AuditionHit(
        label="drum_voice ride_bell tonal control",
        engine="drum_voice",
        preset="ride_bell",
        freq_hz=4_600.0,
        duration_s=0.55,
        repeat_count=2,
        repeat_spacing_s=0.85,
    ),
    AuditionHit(
        label="drum_voice cowbell tonal control",
        engine="drum_voice",
        preset="cowbell",
        freq_hz=800.0,
        duration_s=0.28,
        repeat_count=2,
        repeat_spacing_s=0.56,
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/hat_auditions"),
        help="Directory for the WAV and timing map.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = args.out_dir / "hat_preset_audition.wav"
    timing_path = args.out_dir / "hat_preset_audition_timing.json"

    audio, timing = render_audition()
    write_wav(wav_path, audio, bit_depth=24, warn_low_peak=False)
    timing_path.write_text(
        json.dumps([asdict(hit) for hit in timing], indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"WAV: {wav_path}")
    print(f"Timing map: {timing_path}")
    for hit in timing:
        print(f"{hit.start_s:6.2f}-{hit.end_s:6.2f}s  {hit.label}")


def render_audition() -> tuple[np.ndarray, list[RenderedHit]]:
    chunks: list[np.ndarray] = []
    timing: list[RenderedHit] = []
    cursor_s = 0.0
    section_gap = _silence(0.75)

    for hit in _AUDITION_HITS:
        hit_audio = _render_hit_group(hit)
        start_s = cursor_s
        end_s = start_s + hit_audio.size / SAMPLE_RATE
        chunks.append(hit_audio)
        chunks.append(section_gap)
        timing.append(
            RenderedHit(
                label=hit.label,
                engine=hit.engine,
                preset=hit.preset,
                freq_hz=hit.freq_hz,
                start_s=round(start_s, 3),
                end_s=round(end_s, 3),
                params=hit.params,
            )
        )
        cursor_s = end_s + section_gap.size / SAMPLE_RATE

    audition = np.concatenate(chunks)
    peak = float(np.max(np.abs(audition)))
    if peak > 1e-9:
        audition = audition * (10.0 ** (-1.0 / 20.0) / peak)
    return audition.astype(np.float64), timing


def _render_hit_group(hit: AuditionHit) -> np.ndarray:
    group_len = int(
        SAMPLE_RATE
        * (hit.repeat_spacing_s * (hit.repeat_count - 1) + hit.duration_s + 0.2)
    )
    group = np.zeros(group_len, dtype=np.float64)
    for repeat_index in range(hit.repeat_count):
        start = int(round(repeat_index * hit.repeat_spacing_s * SAMPLE_RATE))
        rendered = _render_one_hit(hit)
        group[start : start + rendered.size] += rendered
    return group


def _render_one_hit(hit: AuditionHit) -> np.ndarray:
    params = resolve_synth_params({"engine": hit.engine, "preset": hit.preset})
    if hit.params is not None:
        params.update(hit.params)
    renderer = drum_voice_render if hit.engine == "drum_voice" else metallic_perc_render
    return renderer(
        freq=hit.freq_hz,
        duration=hit.duration_s,
        amp=hit.amp,
        sample_rate=SAMPLE_RATE,
        params=params,
    )


def _silence(duration_s: float) -> np.ndarray:
    return np.zeros(int(round(duration_s * SAMPLE_RATE)), dtype=np.float64)


if __name__ == "__main__":
    main()
