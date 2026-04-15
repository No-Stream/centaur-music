"""Audio stem WAV export."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np

from code_musics.score import Score
from code_musics.stem_export_types import (
    StemBundleManifest,
    StemBundleResult,
    StemExportSpec,
    StemFileInfo,
)
from code_musics.synth import (
    finalize_master,
    measure_signal_levels,
    write_wav,
)

logger: logging.Logger = logging.getLogger(__name__)

_EXPORT_TARGET_LUFS = -18.0
_EXPORT_TRUE_PEAK_CEILING_DBFS = -0.5
_STEM_PEAK_CEILING_DBFS = -0.5

_WET_PROCESSING_NOTES = (
    "Voice stems include: normalization, pre_fx_gain, pan, voice effects, mix_db fader. "
    "Send stems include: bus summing, bus effects, return_db, bus pan. "
    "Stems sum to the pre-master mix (before auto gain staging and master effects). "
    "The reference mix includes full mastering."
)
_DRY_PROCESSING_NOTES = (
    "Voice stems are post-normalization only (pre-effects, pre-pan, pre-fader, mono). "
    "No send returns are included. "
    "The reference mix is the full wet mastered mix for alignment."
)


def _zero_pad_to_length(signal: np.ndarray, target_length: int) -> np.ndarray:
    """Zero-pad a mono (N,) or stereo (2, N) signal to target_length samples."""
    if signal.ndim == 1:
        current = signal.shape[0]
        if current >= target_length:
            return signal
        return np.pad(signal, (0, target_length - current))
    # stereo: (2, N)
    current = signal.shape[1]
    if current >= target_length:
        return signal
    return np.pad(signal, ((0, 0), (0, target_length - current)))


def _global_peak(signals: list[np.ndarray]) -> float:
    """Return the maximum absolute sample value across all signals."""
    if not signals:
        return 0.0
    return float(max(np.max(np.abs(s)) for s in signals if s.size > 0))


def _compute_ceiling_gain(
    global_peak: float, ceiling_dbfs: float
) -> tuple[float, float]:
    """Return (linear_gain, gain_db) to bring global_peak under ceiling.

    Returns (1.0, 0.0) when no attenuation is needed.
    """
    if global_peak <= 0.0:
        return 1.0, 0.0
    ceiling_linear = 10.0 ** (ceiling_dbfs / 20.0)
    if global_peak <= ceiling_linear:
        return 1.0, 0.0
    gain = ceiling_linear / global_peak
    gain_db = 20.0 * np.log10(gain)
    return float(gain), float(gain_db)


def _write_stem_files(
    stems: dict[str, np.ndarray],
    output_dir: Path,
    kind: Literal["voice", "send"],
    bit_depth: int,
    target_length: int,
    sample_rate: int,
    gain: float = 1.0,
) -> list[StemFileInfo]:
    """Write a dict of named signals to WAV files and return metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[StemFileInfo] = []
    for name, signal in sorted(stems.items()):
        padded = _zero_pad_to_length(signal, target_length)
        if gain != 1.0:
            padded = padded * gain
        wav_path = output_dir / f"{name}.wav"
        write_wav(wav_path, padded, bit_depth=bit_depth, warn_low_peak=False)
        levels = measure_signal_levels(padded, sample_rate=sample_rate)
        channels = 2 if padded.ndim == 2 else 1
        results.append(
            StemFileInfo(
                name=name,
                kind=kind,
                path=str(wav_path.relative_to(output_dir.parent)),
                channels=channels,
                sample_count=padded.shape[-1],
                peak_dbfs=round(levels.peak_dbfs, 2),
            )
        )
    return results


def export_stem_bundle(
    score: Score,
    bundle_dir: str | Path,
    *,
    spec: StemExportSpec,
) -> StemBundleResult:
    """Export audio stem WAVs from a rendered score.

    Returns a StemBundleResult with paths and manifest metadata.
    """
    bundle_path = Path(bundle_dir)
    bundle_path.mkdir(parents=True, exist_ok=True)

    voice_stems, send_returns, mix_audio = score.render_for_stem_export(dry=spec.dry)

    if not voice_stems and not send_returns:
        logger.warning("No stems to export for %s", spec.piece_name)

    # Compute shared target length for zero-padding
    all_signals = [*voice_stems.values(), *send_returns.values()]
    target_length = max(s.shape[-1] for s in all_signals) if all_signals else 0

    sample_rate = score.sample_rate

    # Uniform gain scaling: prevent clipping while preserving stem summation.
    # All stems/sends are scaled by the same factor so relative levels (and
    # the summation property) are preserved exactly.
    peak = _global_peak(all_signals)
    gain, gain_db = _compute_ceiling_gain(peak, _STEM_PEAK_CEILING_DBFS)
    if gain != 1.0:
        logger.info(
            "Stem ceiling gain: %.2f dB (global peak %.2f dBFS → ceiling %.1f dBFS)",
            gain_db,
            20.0 * np.log10(max(peak, 1e-12)),
            _STEM_PEAK_CEILING_DBFS,
        )

    # Write voice stems
    voice_infos = _write_stem_files(
        voice_stems,
        bundle_path / "voices",
        kind="voice",
        bit_depth=spec.bit_depth,
        target_length=target_length,
        sample_rate=sample_rate,
        gain=gain,
    )

    # Write send returns (wet mode only)
    send_infos: list[StemFileInfo] = []
    if send_returns:
        send_infos = _write_stem_files(
            send_returns,
            bundle_path / "sends",
            kind="send",
            bit_depth=spec.bit_depth,
            target_length=target_length,
            sample_rate=sample_rate,
            gain=gain,
        )

    # Write reference mix
    mix_path_str: str | None = None
    if spec.include_mix and mix_audio.size > 0:
        mastering_result = finalize_master(
            mix_audio,
            sample_rate=sample_rate,
            target_lufs=_EXPORT_TARGET_LUFS,
            true_peak_ceiling_dbfs=_EXPORT_TRUE_PEAK_CEILING_DBFS,
        )
        mix_wav_path = bundle_path / "mix.wav"
        write_wav(
            mix_wav_path,
            mastering_result.signal,
            bit_depth=spec.bit_depth,
            warn_low_peak=False,
        )
        mix_path_str = "mix.wav"

    total_duration = target_length / sample_rate if target_length > 0 else 0.0

    manifest = StemBundleManifest(
        schema_version=1,
        piece_name=spec.piece_name,
        output_name=spec.output_name,
        sample_rate=sample_rate,
        bit_depth=spec.bit_depth,
        dry=spec.dry,
        total_duration_seconds=round(total_duration, 4),
        total_samples=target_length,
        voice_stems=voice_infos,
        send_stems=send_infos,
        mix_path=mix_path_str,
        processing_notes=_DRY_PROCESSING_NOTES if spec.dry else _WET_PROCESSING_NOTES,
        stem_gain_db=round(gain_db, 2),
    )

    manifest_path = bundle_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=False),
        encoding="utf-8",
    )
    logger.info(
        "Exported %d voice stems + %d send returns to %s",
        len(voice_infos),
        len(send_infos),
        bundle_path,
    )

    return StemBundleResult(
        bundle_dir=bundle_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )
