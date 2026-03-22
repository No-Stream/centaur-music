"""Render and score analysis helpers."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from code_musics import synth
from code_musics.pieces.registry import PieceSection
from code_musics.score import Score

_EPSILON = 1e-12
logger = logging.getLogger(__name__)

# Artifact risk codes suppressed from log output (still written to JSON manifest).
# These are legitimate checks for melodic voices but are structural false positives
# on any arrangement that includes hi-hats or high-register FM leads.
SUPPRESSED_CODES: frozenset[str] = frozenset(
    {
        "high_band_dominance",
        "bright_spectral_centroid",
        "flat_or_bright_tilt",  # expected on hi-hats and high-register leads
    }
)
_DEFAULT_BANDS: tuple[tuple[str, float, float], ...] = (
    ("sub", 20.0, 60.0),
    ("bass", 60.0, 250.0),
    ("low_mid", 250.0, 500.0),
    ("mid", 500.0, 2_000.0),
    ("high", 2_000.0, 8_000.0),
    ("air", 8_000.0, 20_000.0),
)


@dataclass(frozen=True)
class ArtifactRiskWarning:
    """Structured warning for suspicious rendered audio or parameter surfaces."""

    severity: str
    code: str
    message: str
    source: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class ArtifactRiskReport:
    """Aggregated artifact-risk summary for a render."""

    mix: list[ArtifactRiskWarning]
    voices: dict[str, list[ArtifactRiskWarning]]
    parameter_surfaces: dict[str, list[ArtifactRiskWarning]]
    summary: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class AudioAnalysis:
    """Machine-readable audio analysis summary."""

    duration_seconds: float
    peak_dbfs: float
    true_peak_dbfs: float
    peak_headroom_db: float
    clipped_sample_count: int
    clipped_sample_fraction: float
    clipped_true_peak: bool
    rms_dbfs: float
    gated_rms_dbfs: float
    integrated_lufs: float
    active_window_fraction: float
    crest_factor_db: float
    spectral_centroid_hz: float
    dominant_frequency_hz: float
    low_high_balance_db: float
    high_band_emphasis_db: float
    spectral_tilt_db_per_octave: float
    reference_tilt_db_per_octave: float
    tilt_error_db_per_octave: float
    amplitude_modulation_depth_db: float
    dominant_amplitude_modulation_hz: float
    band_energy_db: dict[str, float]
    warnings: list[str]
    artifact_risks: list[ArtifactRiskWarning]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ScoreAnalysis:
    """Machine-readable symbolic score analysis summary."""

    total_duration_seconds: float
    note_count: int
    voice_count: int
    notes_per_second: float
    peak_simultaneous_notes: int
    mean_simultaneous_notes: float
    mean_attack_density_hz: float
    max_attack_density_hz: float
    partial_range: tuple[float, float] | None
    frequency_range_hz: tuple[float, float] | None
    voice_summaries: dict[str, dict[str, float | int]]
    timing_drift_summary: dict[str, Any]
    timing_drift_windows: list[dict[str, Any]]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)


def analyze_audio(
    signal: np.ndarray,
    *,
    sample_rate: int,
    reference_tilt_db_per_octave: float = -3.0,
) -> AudioAnalysis:
    """Summarize a mono or stereo render for diagnostics."""
    mono_signal = synth.to_mono_reference(signal)
    if mono_signal.size == 0:
        return AudioAnalysis(
            duration_seconds=0.0,
            peak_dbfs=float("-inf"),
            true_peak_dbfs=float("-inf"),
            peak_headroom_db=float("inf"),
            clipped_sample_count=0,
            clipped_sample_fraction=0.0,
            clipped_true_peak=False,
            rms_dbfs=float("-inf"),
            gated_rms_dbfs=float("-inf"),
            integrated_lufs=float("-inf"),
            active_window_fraction=0.0,
            crest_factor_db=0.0,
            spectral_centroid_hz=0.0,
            dominant_frequency_hz=0.0,
            low_high_balance_db=0.0,
            high_band_emphasis_db=0.0,
            spectral_tilt_db_per_octave=0.0,
            reference_tilt_db_per_octave=reference_tilt_db_per_octave,
            tilt_error_db_per_octave=-reference_tilt_db_per_octave,
            amplitude_modulation_depth_db=0.0,
            dominant_amplitude_modulation_hz=0.0,
            band_energy_db={name: float("-inf") for name, _, _ in _DEFAULT_BANDS},
            warnings=["empty render"],
            artifact_risks=[],
        )

    freqs, magnitude_db = _average_spectrum(mono_signal, sample_rate=sample_rate)
    peak_amplitude = float(np.max(np.abs(mono_signal)))
    true_peak_amplitude = synth.estimate_true_peak_amplitude(signal)
    rms_amplitude = float(np.sqrt(np.mean(np.square(mono_signal))))
    gated_rms_dbfs, active_window_fraction = synth.gated_rms_dbfs(
        signal,
        sample_rate=sample_rate,
    )
    integrated_lufs, lufs_active_window_fraction = synth.integrated_lufs(
        signal,
        sample_rate=sample_rate,
    )
    clipped_sample_count = int(np.count_nonzero(np.abs(mono_signal) >= 1.0))
    peak_dbfs = _amplitude_to_db(peak_amplitude)
    true_peak_dbfs = _amplitude_to_db(true_peak_amplitude)
    band_energy_db = _compute_band_energies(
        freqs=freqs,
        magnitude_db=magnitude_db,
    )
    high_band_emphasis_db = _compute_high_band_emphasis_db(band_energy_db)
    low_high_balance_db = _band_mean(
        magnitude_db,
        freqs=freqs,
        low_hz=60.0,
        high_hz=250.0,
    ) - _band_mean(
        magnitude_db,
        freqs=freqs,
        low_hz=2_000.0,
        high_hz=8_000.0,
    )

    spectral_centroid_hz = _spectral_centroid(freqs=freqs, magnitude_db=magnitude_db)
    dominant_frequency_hz = float(freqs[np.argmax(magnitude_db)])
    spectral_tilt = _fit_spectral_tilt(freqs=freqs, magnitude_db=magnitude_db)
    dominant_amplitude_modulation_hz = _dominant_amplitude_modulation_hz(
        mono_signal,
        sample_rate=sample_rate,
    )
    amplitude_modulation_depth_db = _amplitude_modulation_depth_db(
        mono_signal,
        sample_rate=sample_rate,
    )
    warnings = _build_audio_warnings(
        peak_dbfs=peak_dbfs,
        true_peak_dbfs=true_peak_dbfs,
        clipped_sample_count=clipped_sample_count,
        integrated_lufs=integrated_lufs,
        active_window_fraction=lufs_active_window_fraction,
        low_high_balance_db=low_high_balance_db,
        tilt_error_db_per_octave=spectral_tilt - reference_tilt_db_per_octave,
        spectral_centroid_hz=spectral_centroid_hz,
    )
    artifact_risks = _build_audio_artifact_risks(
        peak_dbfs=peak_dbfs,
        true_peak_dbfs=true_peak_dbfs,
        clipped_sample_count=clipped_sample_count,
        integrated_lufs=integrated_lufs,
        crest_factor_db=max(0.0, peak_dbfs - _amplitude_to_db(rms_amplitude)),
        spectral_centroid_hz=spectral_centroid_hz,
        low_high_balance_db=low_high_balance_db,
        high_band_emphasis_db=high_band_emphasis_db,
        tilt_error_db_per_octave=spectral_tilt - reference_tilt_db_per_octave,
        amplitude_modulation_depth_db=amplitude_modulation_depth_db,
        dominant_amplitude_modulation_hz=dominant_amplitude_modulation_hz,
    )

    return AudioAnalysis(
        duration_seconds=float(mono_signal.size / sample_rate),
        peak_dbfs=peak_dbfs,
        true_peak_dbfs=true_peak_dbfs,
        peak_headroom_db=max(0.0, -peak_dbfs),
        clipped_sample_count=clipped_sample_count,
        clipped_sample_fraction=clipped_sample_count / float(mono_signal.size),
        clipped_true_peak=true_peak_amplitude > 1.0,
        rms_dbfs=_amplitude_to_db(rms_amplitude),
        gated_rms_dbfs=gated_rms_dbfs,
        integrated_lufs=integrated_lufs,
        active_window_fraction=lufs_active_window_fraction,
        crest_factor_db=max(0.0, peak_dbfs - _amplitude_to_db(rms_amplitude)),
        spectral_centroid_hz=spectral_centroid_hz,
        dominant_frequency_hz=dominant_frequency_hz,
        low_high_balance_db=low_high_balance_db,
        high_band_emphasis_db=high_band_emphasis_db,
        spectral_tilt_db_per_octave=spectral_tilt,
        reference_tilt_db_per_octave=reference_tilt_db_per_octave,
        tilt_error_db_per_octave=spectral_tilt - reference_tilt_db_per_octave,
        amplitude_modulation_depth_db=amplitude_modulation_depth_db,
        dominant_amplitude_modulation_hz=dominant_amplitude_modulation_hz,
        band_energy_db=band_energy_db,
        warnings=warnings,
        artifact_risks=artifact_risks,
    )


def analyze_score(
    score: Score,
    *,
    window_seconds: float = 1.0,
) -> ScoreAnalysis:
    """Summarize note density and registral spread from symbolic score events."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    total_duration = score.total_dur
    resolved_notes = score.resolved_timing_notes()
    voice_summaries: dict[str, dict[str, float | int]] = {}
    frequencies_hz: list[float] = []
    partials: list[float] = []
    note_count = 0
    onset_counts: list[int] = []
    active_counts: list[int] = []

    if total_duration > 0:
        window_edges = np.arange(0.0, total_duration + window_seconds, window_seconds)
        onset_counts = [0 for _ in range(max(len(window_edges) - 1, 1))]
        active_counts = [0 for _ in range(max(len(window_edges) - 1, 1))]
    else:
        window_edges = np.array([0.0, window_seconds])
        onset_counts = [0]
        active_counts = [0]

    for voice_name, voice in score.voices.items():
        voice_notes = voice.notes
        note_count += len(voice_notes)

        voice_freqs: list[float] = []
        voice_attacks = 0
        for note in voice_notes:
            if note.freq is not None:
                resolved_freq = note.freq
            else:
                if note.partial is None:
                    raise ValueError("note must define partial or freq")
                resolved_freq = score.f0 * note.partial
            voice_freqs.append(resolved_freq)
            frequencies_hz.append(resolved_freq)
            if note.partial is not None:
                partials.append(float(note.partial))
            voice_attacks += 1

            onset_index = min(
                int(note.start / window_seconds),
                len(onset_counts) - 1,
            )
            onset_counts[onset_index] += 1

            start_index = min(int(note.start / window_seconds), len(active_counts) - 1)
            end_index = min(
                max(
                    int(np.ceil((note.start + note.duration) / window_seconds)) - 1,
                    start_index,
                ),
                len(active_counts) - 1,
            )
            for index in range(start_index, end_index + 1):
                active_counts[index] += 1

        voice_summaries[voice_name] = {
            "note_count": len(voice_notes),
            "mean_frequency_hz": float(np.mean(voice_freqs)) if voice_freqs else 0.0,
            "min_frequency_hz": float(np.min(voice_freqs)) if voice_freqs else 0.0,
            "max_frequency_hz": float(np.max(voice_freqs)) if voice_freqs else 0.0,
            "mean_note_duration_seconds": (
                float(np.mean([note.duration for note in voice_notes]))
                if voice_notes
                else 0.0
            ),
            "attack_rate_hz": (voice_attacks / total_duration)
            if total_duration > 0
            else 0.0,
        }

    notes_per_second = note_count / total_duration if total_duration > 0 else 0.0
    mean_attack_density_hz = (
        float(np.mean(onset_counts)) / window_seconds if onset_counts else 0.0
    )
    max_attack_density_hz = (
        float(np.max(onset_counts)) / window_seconds if onset_counts else 0.0
    )
    peak_simultaneous_notes = int(np.max(active_counts)) if active_counts else 0
    mean_simultaneous_notes = float(np.mean(active_counts)) if active_counts else 0.0

    warnings: list[str] = []
    if mean_simultaneous_notes >= 4.0:
        warnings.append("dense texture across much of the score")
    if max_attack_density_hz >= 5.0:
        warnings.append("high attack density may feel busy or percussive")
    if (
        note_count > 0
        and mean_note_duration_hz(score) > 2.5
        and mean_attack_density_hz < 0.8
    ):
        warnings.append("long-note bias may read as drony")

    timing_drift_summary, timing_drift_windows = _analyze_timing_drift(
        resolved_notes=resolved_notes,
        total_duration=total_duration,
        window_seconds=window_seconds,
    )
    warnings.extend(timing_drift_summary["warnings"])

    return ScoreAnalysis(
        total_duration_seconds=total_duration,
        note_count=note_count,
        voice_count=len(score.voices),
        notes_per_second=notes_per_second,
        peak_simultaneous_notes=peak_simultaneous_notes,
        mean_simultaneous_notes=mean_simultaneous_notes,
        mean_attack_density_hz=mean_attack_density_hz,
        max_attack_density_hz=max_attack_density_hz,
        partial_range=(
            (float(np.min(partials)), float(np.max(partials))) if partials else None
        ),
        frequency_range_hz=(
            (float(np.min(frequencies_hz)), float(np.max(frequencies_hz)))
            if frequencies_hz
            else None
        ),
        voice_summaries=voice_summaries,
        timing_drift_summary=timing_drift_summary,
        timing_drift_windows=timing_drift_windows,
        warnings=warnings,
    )


def save_analysis_artifacts(
    *,
    output_prefix: str | Path,
    mix_signal: np.ndarray,
    sample_rate: int,
    pre_master_mix_signal: np.ndarray | None = None,
    pre_export_mix_signal: np.ndarray | None = None,
    stems: dict[str, np.ndarray] | None = None,
    effect_analysis: dict[str, Any] | None = None,
    score: Score | None = None,
    piece_sections: tuple[PieceSection, ...] = (),
    reference_tilt_db_per_octave: float = -3.0,
) -> dict[str, Any]:
    """Write plots and a JSON manifest for a render."""
    prefix_path = Path(output_prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    mix_analysis = analyze_audio(
        mix_signal,
        sample_rate=sample_rate,
        reference_tilt_db_per_octave=reference_tilt_db_per_octave,
    )
    pre_master_analysis: AudioAnalysis | None = None
    pre_export_analysis: AudioAnalysis | None = None
    manifest: dict[str, Any] = {
        "reference_tilt_db_per_octave": reference_tilt_db_per_octave,
        "mix": {
            "summary": mix_analysis.to_dict(),
            "artifacts": {},
        },
        "voices": {},
        "effect_analysis": effect_analysis
        if effect_analysis is not None
        else {"mix_effects": [], "voice_effects": {}, "send_effects": {}},
    }
    if pre_master_mix_signal is not None:
        pre_master_analysis = analyze_audio(
            pre_master_mix_signal,
            sample_rate=sample_rate,
            reference_tilt_db_per_octave=reference_tilt_db_per_octave,
        )
        manifest["mix"]["pre_master_summary"] = pre_master_analysis.to_dict()
    if pre_export_mix_signal is not None:
        pre_export_analysis = analyze_audio(
            pre_export_mix_signal,
            sample_rate=sample_rate,
            reference_tilt_db_per_octave=reference_tilt_db_per_octave,
        )
        manifest["mix"]["pre_export_summary"] = pre_export_analysis.to_dict()

    mix_spectrum_path = prefix_path.with_name(f"{prefix_path.name}.mix_spectrum.png")
    _save_spectrum_plot(
        signal=mix_signal,
        sample_rate=sample_rate,
        path=mix_spectrum_path,
        title="Mix Spectrum",
        reference_tilt_db_per_octave=reference_tilt_db_per_octave,
    )
    mix_spectrogram_path = prefix_path.with_name(
        f"{prefix_path.name}.mix_spectrogram.png"
    )
    _save_spectrogram_plot(
        signal=mix_signal,
        sample_rate=sample_rate,
        path=mix_spectrogram_path,
        title="Mix Spectrogram",
    )
    mix_band_energy_path = prefix_path.with_name(
        f"{prefix_path.name}.mix_band_energy.png"
    )
    _save_band_energy_plot(
        band_energy_db=mix_analysis.band_energy_db,
        path=mix_band_energy_path,
        title="Mix Band Energy",
    )
    manifest["mix"]["artifacts"] = {
        "spectrum": str(mix_spectrum_path),
        "spectrogram": str(mix_spectrogram_path),
        "band_energy": str(mix_band_energy_path),
    }

    if score is not None:
        score_analysis = analyze_score(score)
        score_density_path = prefix_path.with_name(
            f"{prefix_path.name}.score_density.png"
        )
        _save_score_density_plot(score=score, path=score_density_path)
        timeline = build_score_timeline(
            score=score,
            sections=piece_sections,
            window_seconds=1.0,
        )
        timeline_path = prefix_path.with_name(f"{prefix_path.name}.timeline.json")
        timeline_path.write_text(_json_dump(timeline), encoding="utf-8")
        manifest["score"] = {
            "summary": score_analysis.to_dict(),
            "artifacts": {
                "density": str(score_density_path),
                "timeline": str(timeline_path),
            },
        }

    voice_analyses: dict[str, AudioAnalysis] = {}
    for voice_name, stem_signal in (stems or {}).items():
        voice_analysis = analyze_audio(
            stem_signal,
            sample_rate=sample_rate,
            reference_tilt_db_per_octave=reference_tilt_db_per_octave,
        )
        voice_analyses[voice_name] = voice_analysis
        safe_voice_name = _sanitize_name(voice_name)
        voice_spectrum_path = prefix_path.with_name(
            f"{prefix_path.name}.voice_{safe_voice_name}_spectrum.png"
        )
        _save_spectrum_plot(
            signal=stem_signal,
            sample_rate=sample_rate,
            path=voice_spectrum_path,
            title=f"Voice Spectrum: {voice_name}",
            reference_tilt_db_per_octave=reference_tilt_db_per_octave,
        )
        manifest["voices"][voice_name] = {
            "summary": voice_analysis.to_dict(),
            "artifacts": {
                "spectrum": str(voice_spectrum_path),
            },
        }

    artifact_risk_report = _build_artifact_risk_report(
        mix_analysis=mix_analysis,
        voice_analyses=voice_analyses,
        score=score,
        pre_master_mix_analysis=pre_master_analysis,
        pre_export_mix_analysis=pre_export_analysis,
    )
    manifest["artifact_risk"] = artifact_risk_report.to_dict()
    _log_artifact_risk_report(artifact_risk_report)
    _log_effect_analysis_warnings(manifest["effect_analysis"])

    manifest_path = prefix_path.with_name(f"{prefix_path.name}.analysis.json")
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(_json_dump(manifest), encoding="utf-8")
    return manifest


def compare_analysis_manifests(
    before_manifest_path: str | Path,
    after_manifest_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare two analysis manifests and optionally save a delta report."""
    before_manifest = json.loads(Path(before_manifest_path).read_text(encoding="utf-8"))
    after_manifest = json.loads(Path(after_manifest_path).read_text(encoding="utf-8"))

    comparison = {
        "before_manifest_path": str(before_manifest_path),
        "after_manifest_path": str(after_manifest_path),
        "mix_delta": _compare_numeric_dicts(
            before_manifest["mix"]["summary"],
            after_manifest["mix"]["summary"],
            keys=(
                "peak_dbfs",
                "true_peak_dbfs",
                "rms_dbfs",
                "integrated_lufs",
                "spectral_centroid_hz",
                "dominant_frequency_hz",
                "low_high_balance_db",
                "spectral_tilt_db_per_octave",
                "tilt_error_db_per_octave",
            ),
        ),
        "score_delta": _compare_numeric_dicts(
            before_manifest.get("score", {}).get("summary", {}),
            after_manifest.get("score", {}).get("summary", {}),
            keys=(
                "note_count",
                "notes_per_second",
                "peak_simultaneous_notes",
                "mean_simultaneous_notes",
                "mean_attack_density_hz",
                "max_attack_density_hz",
            ),
        ),
        "voice_delta": {},
        "warning_changes": {
            "mix_before": before_manifest["mix"]["summary"].get("warnings", []),
            "mix_after": after_manifest["mix"]["summary"].get("warnings", []),
            "score_before": before_manifest.get("score", {})
            .get("summary", {})
            .get("warnings", []),
            "score_after": after_manifest.get("score", {})
            .get("summary", {})
            .get("warnings", []),
        },
    }

    for voice_name in sorted(
        set(before_manifest.get("voices", {})) | set(after_manifest.get("voices", {}))
    ):
        before_voice = (
            before_manifest.get("voices", {}).get(voice_name, {}).get("summary", {})
        )
        after_voice = (
            after_manifest.get("voices", {}).get(voice_name, {}).get("summary", {})
        )
        comparison["voice_delta"][voice_name] = _compare_numeric_dicts(
            before_voice,
            after_voice,
            keys=(
                "peak_dbfs",
                "true_peak_dbfs",
                "rms_dbfs",
                "integrated_lufs",
                "spectral_centroid_hz",
                "low_high_balance_db",
                "spectral_tilt_db_per_octave",
            ),
        )

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(_json_dump(comparison), encoding="utf-8")
        comparison["output_path"] = str(output_file)

    return comparison


def mean_note_duration_hz(score: Score) -> float:
    """Return the average note duration in seconds for warning heuristics."""
    durations = [
        note.duration for voice in score.voices.values() for note in voice.notes
    ]
    return float(np.mean(durations)) if durations else 0.0


def build_score_timeline(
    *,
    score: Score,
    sections: tuple[PieceSection, ...] = (),
    window_seconds: float = 1.0,
) -> dict[str, Any]:
    """Build a machine-readable timeline artifact for score inspection."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    resolved_notes = sorted(
        score.resolved_timing_notes(),
        key=lambda note: (note.resolved_start, note.voice_name, note.note_index),
    )
    return {
        "total_duration_seconds": score.total_dur,
        "window_seconds": window_seconds,
        "voice_names": list(score.voices),
        "sections": [
            {
                "label": section.label,
                "start_seconds": section.start_seconds,
                "end_seconds": section.end_seconds,
            }
            for section in sections
        ],
        "notes": [
            {
                "voice_name": note.voice_name,
                "note_index": note.note_index,
                "authored_start_seconds": note.authored_start,
                "resolved_start_seconds": note.resolved_start,
                "resolved_end_seconds": note.resolved_end,
                "duration_seconds": note.duration,
                "timing_offset_ms": note.timing_offset_seconds * 1_000.0,
                "freq_hz": note.freq_hz,
                "partial": note.partial,
                "label": note.label,
            }
            for note in resolved_notes
        ],
        "windows": _build_timeline_windows(
            resolved_notes=resolved_notes,
            total_duration=score.total_dur,
            window_seconds=window_seconds,
        ),
    }


def _analyze_timing_drift(
    *,
    resolved_notes: list[Any],
    total_duration: float,
    window_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clusters = _cluster_resolved_notes_by_authored_start(resolved_notes)
    absolute_offsets_ms = np.asarray(
        [abs(note.timing_offset_seconds) * 1_000.0 for note in resolved_notes],
        dtype=np.float64,
    )
    inter_voice_spreads_ms = np.asarray(
        [
            cluster["inter_voice_spread_ms"]
            for cluster in clusters
            if cluster["voice_count"] >= 2
        ],
        dtype=np.float64,
    )
    pairwise_summary = _build_pairwise_spread_summary(clusters)
    drift_windows = _build_timing_drift_windows(
        clusters=clusters,
        total_duration=total_duration,
        window_seconds=window_seconds,
    )

    warnings: list[str] = []
    if _safe_percentile(inter_voice_spreads_ms, 95.0) >= 35.0:
        warnings.append("timing drift regularly exceeds a human ensemble feel")
    if _safe_max(inter_voice_spreads_ms) >= 55.0:
        warnings.append("large inter-voice timing spreads may blur coordinated attacks")
    if any(window["max_inter_voice_spread_ms"] >= 45.0 for window in drift_windows):
        warnings.append("some score windows drift more than the surrounding texture")
    for pair_name, pair_stats in pairwise_summary.items():
        if (
            int(pair_stats["cluster_count"]) >= 4
            and abs(float(pair_stats["mean_signed_spread_ms"])) >= 15.0
        ):
            warnings.append(f"voice pair {pair_name} shows a persistent lead-lag bias")

    return (
        {
            "note_count": len(resolved_notes),
            "cluster_count": len(clusters),
            "mean_absolute_offset_ms": _safe_mean(absolute_offsets_ms),
            "median_absolute_offset_ms": _safe_percentile(absolute_offsets_ms, 50.0),
            "p95_absolute_offset_ms": _safe_percentile(absolute_offsets_ms, 95.0),
            "max_absolute_offset_ms": _safe_max(absolute_offsets_ms),
            "mean_inter_voice_spread_ms": _safe_mean(inter_voice_spreads_ms),
            "median_inter_voice_spread_ms": _safe_percentile(
                inter_voice_spreads_ms,
                50.0,
            ),
            "p95_inter_voice_spread_ms": _safe_percentile(
                inter_voice_spreads_ms,
                95.0,
            ),
            "max_inter_voice_spread_ms": _safe_max(inter_voice_spreads_ms),
            "per_voice_pair": pairwise_summary,
            "warnings": warnings,
        },
        drift_windows,
    )


def _cluster_resolved_notes_by_authored_start(
    resolved_notes: list[Any],
    *,
    authored_start_tolerance_seconds: float = 1e-6,
) -> list[dict[str, Any]]:
    sorted_notes = sorted(
        resolved_notes,
        key=lambda note: (note.authored_start, note.voice_name, note.note_index),
    )
    if not sorted_notes:
        return []

    clusters: list[list[Any]] = [[sorted_notes[0]]]
    for note in sorted_notes[1:]:
        if (
            abs(note.authored_start - clusters[-1][-1].authored_start)
            <= authored_start_tolerance_seconds
        ):
            clusters[-1].append(note)
        else:
            clusters.append([note])

    return [
        {
            "authored_start_seconds": notes[0].authored_start,
            "voice_count": len({note.voice_name for note in notes}),
            "note_count": len(notes),
            "inter_voice_spread_ms": (
                float(
                    (
                        max(note.resolved_start for note in notes)
                        - min(note.resolved_start for note in notes)
                    )
                    * 1_000.0
                )
                if len(notes) >= 2
                else 0.0
            ),
            "notes": notes,
        }
        for notes in clusters
    ]


def _build_pairwise_spread_summary(
    clusters: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    pairwise_values: dict[str, list[float]] = {}
    for cluster in clusters:
        notes = sorted(
            cluster["notes"], key=lambda note: (note.voice_name, note.note_index)
        )
        for left_index, left_note in enumerate(notes):
            for right_note in notes[left_index + 1 :]:
                if left_note.voice_name == right_note.voice_name:
                    continue
                pair_name = f"{left_note.voice_name}|{right_note.voice_name}"
                pairwise_values.setdefault(pair_name, []).append(
                    (right_note.resolved_start - left_note.resolved_start) * 1_000.0
                )

    return {
        pair_name: {
            "cluster_count": len(values),
            "mean_signed_spread_ms": float(np.mean(values)),
            "mean_absolute_spread_ms": float(np.mean(np.abs(values))),
            "max_absolute_spread_ms": float(np.max(np.abs(values))),
        }
        for pair_name, values in sorted(pairwise_values.items())
    }


def _build_timing_drift_windows(
    *,
    clusters: list[dict[str, Any]],
    total_duration: float,
    window_seconds: float,
) -> list[dict[str, Any]]:
    if total_duration <= 0:
        return []

    windows: list[dict[str, Any]] = []
    window_start = 0.0
    while window_start < total_duration:
        window_end = min(total_duration, window_start + window_seconds)
        window_clusters = [
            cluster
            for cluster in clusters
            if window_start <= cluster["authored_start_seconds"] < window_end
        ]
        window_spreads = np.asarray(
            [
                cluster["inter_voice_spread_ms"]
                for cluster in window_clusters
                if cluster["voice_count"] >= 2
            ],
            dtype=np.float64,
        )
        windows.append(
            {
                "start_seconds": window_start,
                "end_seconds": window_end,
                "cluster_count": len(window_clusters),
                "mean_inter_voice_spread_ms": _safe_mean(window_spreads),
                "max_inter_voice_spread_ms": _safe_max(window_spreads),
            }
        )
        window_start = window_end
    return windows


def _build_timeline_windows(
    *,
    resolved_notes: list[Any],
    total_duration: float,
    window_seconds: float,
) -> list[dict[str, Any]]:
    if total_duration <= 0:
        return []

    windows: list[dict[str, Any]] = []
    window_start = 0.0
    while window_start < total_duration:
        window_end = min(total_duration, window_start + window_seconds)
        notes_starting = [
            note
            for note in resolved_notes
            if window_start <= note.resolved_start < window_end
        ]
        notes_active = [
            note
            for note in resolved_notes
            if note.resolved_start < window_end and note.resolved_end > window_start
        ]
        windows.append(
            {
                "start_seconds": window_start,
                "end_seconds": window_end,
                "onset_count": len(notes_starting),
                "active_note_count": len(notes_active),
                "active_voice_names": sorted(
                    {note.voice_name for note in notes_active}
                ),
                "mean_freq_hz": (
                    float(np.mean([note.freq_hz for note in notes_active]))
                    if notes_active
                    else 0.0
                ),
            }
        )
        window_start = window_end
    return windows


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size > 0 else 0.0


def _safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size > 0 else 0.0


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size > 0 else 0.0


def _average_spectrum(
    signal: np.ndarray, *, sample_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    n_fft = min(8192, max(2048, int(2 ** np.ceil(np.log2(max(signal.size, 2_048))))))
    window = np.hanning(n_fft)
    if signal.size < n_fft:
        padded = np.zeros(n_fft, dtype=np.float64)
        padded[: signal.size] = signal
        spectrum = np.abs(np.fft.rfft(padded * window))
    else:
        step = max(n_fft // 2, 1)
        magnitudes: list[np.ndarray] = []
        for start in range(0, signal.size - n_fft + 1, step):
            frame = signal[start : start + n_fft] * window
            magnitudes.append(np.abs(np.fft.rfft(frame)))
        spectrum = np.mean(magnitudes, axis=0)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    magnitude_db = 20.0 * np.log10(
        np.maximum(np.asarray(spectrum, dtype=np.float64), _EPSILON)
    )
    valid = freqs > 0
    return freqs[valid], magnitude_db[valid]


def _compute_band_energies(
    *,
    freqs: np.ndarray,
    magnitude_db: np.ndarray,
) -> dict[str, float]:
    return {
        name: _band_mean(magnitude_db, freqs=freqs, low_hz=low_hz, high_hz=high_hz)
        for name, low_hz, high_hz in _DEFAULT_BANDS
    }


def _band_mean(
    magnitude_db: np.ndarray,
    *,
    freqs: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float:
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return float("-inf")
    return float(np.mean(magnitude_db[mask]))


def _spectral_centroid(*, freqs: np.ndarray, magnitude_db: np.ndarray) -> float:
    magnitudes = np.power(10.0, magnitude_db / 20.0)
    weight_sum = np.sum(magnitudes)
    if weight_sum <= 0:
        return 0.0
    return float(np.sum(freqs * magnitudes) / weight_sum)


def _fit_spectral_tilt(*, freqs: np.ndarray, magnitude_db: np.ndarray) -> float:
    valid = freqs >= 20.0
    if np.count_nonzero(valid) < 4:
        return 0.0
    log2_freqs = np.log2(freqs[valid])
    coefficients = np.polyfit(log2_freqs, magnitude_db[valid], 1)
    return float(coefficients[0])


def _build_audio_warnings(
    *,
    peak_dbfs: float,
    true_peak_dbfs: float,
    clipped_sample_count: int,
    integrated_lufs: float,
    active_window_fraction: float,
    low_high_balance_db: float,
    tilt_error_db_per_octave: float,
    spectral_centroid_hz: float,
) -> list[str]:
    warnings: list[str] = []
    if clipped_sample_count > 0:
        warnings.append("sample peak clipping detected")
    elif true_peak_dbfs > 0.0:
        warnings.append("estimated inter-sample clipping risk")
    elif peak_dbfs >= -0.3:
        warnings.append("very little peak headroom")
    elif peak_dbfs >= -1.0:
        warnings.append("peak headroom is tight")

    if integrated_lufs <= -32.0 and active_window_fraction >= 0.05:
        warnings.append("active passages are very quiet overall")
    elif integrated_lufs <= -27.0 and active_window_fraction >= 0.05:
        warnings.append("active passages are somewhat quiet overall")

    if low_high_balance_db >= 18.0:
        warnings.append("low band dominates high band strongly")
    elif low_high_balance_db >= 10.0:
        warnings.append("mix leans warm and low-forward")
    elif low_high_balance_db <= -14.0:
        warnings.append("high band dominates low band strongly")
    elif low_high_balance_db <= -8.0:
        warnings.append("mix leans bright relative to the low band")

    if tilt_error_db_per_octave <= -6.0:
        warnings.append("spectrum falls off faster than the reference tilt")
    elif tilt_error_db_per_octave <= -3.0:
        warnings.append("spectrum is warmer than most pop-style balances")
    elif tilt_error_db_per_octave >= 6.0:
        warnings.append("spectrum is flatter or brighter than the reference tilt")
    elif tilt_error_db_per_octave >= 3.0:
        warnings.append("spectrum is somewhat brighter than the reference tilt")

    if spectral_centroid_hz < 180.0:
        warnings.append("very dark spectral centroid")
    elif spectral_centroid_hz > 4_000.0:
        warnings.append("very bright spectral centroid")
    return warnings


def _build_audio_artifact_risks(
    *,
    peak_dbfs: float,
    true_peak_dbfs: float,
    clipped_sample_count: int,
    integrated_lufs: float,
    crest_factor_db: float,
    spectral_centroid_hz: float,
    low_high_balance_db: float,
    high_band_emphasis_db: float,
    tilt_error_db_per_octave: float,
    amplitude_modulation_depth_db: float,
    dominant_amplitude_modulation_hz: float,
) -> list[ArtifactRiskWarning]:
    risks: list[ArtifactRiskWarning] = []
    if clipped_sample_count > 0 or true_peak_dbfs > 0.0:
        risks.append(
            _artifact_risk(
                severity="severe",
                code="clipping_risk",
                source="audio_analysis",
                message="render is clipping or at inter-sample clipping risk",
                true_peak_dbfs=round(true_peak_dbfs, 2),
                clipped_sample_count=clipped_sample_count,
            )
        )
    if crest_factor_db <= 3.5 and integrated_lufs >= -16.0:
        risks.append(
            _artifact_risk(
                severity="severe",
                code="extreme_compression",
                source="audio_analysis",
                message="render dynamics are extremely squashed and may indicate over-compression or clipping-like saturation",
                crest_factor_db=round(crest_factor_db, 2),
                integrated_lufs=round(integrated_lufs, 2),
                peak_dbfs=round(peak_dbfs, 2),
            )
        )
    elif crest_factor_db <= 5.0 and integrated_lufs >= -18.0:
        risks.append(
            _artifact_risk(
                severity="warning",
                code="extreme_compression",
                source="audio_analysis",
                message="render dynamics are unusually constrained and may be over-compressed",
                crest_factor_db=round(crest_factor_db, 2),
                integrated_lufs=round(integrated_lufs, 2),
                peak_dbfs=round(peak_dbfs, 2),
            )
        )
    if spectral_centroid_hz >= 5_500.0:
        risks.append(
            _artifact_risk(
                severity="severe",
                code="bright_spectral_centroid",
                source="audio_analysis",
                message="render is extremely bright and may read as harsh or brittle",
                spectral_centroid_hz=round(spectral_centroid_hz, 1),
            )
        )
    elif spectral_centroid_hz >= 4_000.0:
        risks.append(
            _artifact_risk(
                severity="warning",
                code="bright_spectral_centroid",
                source="audio_analysis",
                message="render is very bright and may exaggerate filter artifacts",
                spectral_centroid_hz=round(spectral_centroid_hz, 1),
            )
        )
    if high_band_emphasis_db >= 12.0 or low_high_balance_db <= -16.0:
        risks.append(
            _artifact_risk(
                severity="severe",
                code="high_band_dominance",
                source="audio_analysis",
                message="upper bands dominate strongly and may sound clipped or piercing",
                high_band_emphasis_db=round(high_band_emphasis_db, 2),
                low_high_balance_db=round(low_high_balance_db, 2),
            )
        )
    elif high_band_emphasis_db >= 7.0 or low_high_balance_db <= -10.0:
        risks.append(
            _artifact_risk(
                severity="warning",
                code="high_band_dominance",
                source="audio_analysis",
                message="upper-band energy is elevated relative to the body of the mix",
                high_band_emphasis_db=round(high_band_emphasis_db, 2),
                low_high_balance_db=round(low_high_balance_db, 2),
            )
        )
    if tilt_error_db_per_octave >= 8.0:
        risks.append(
            _artifact_risk(
                severity="severe",
                code="flat_or_bright_tilt",
                source="audio_analysis",
                message="spectral tilt is unusually flat or bright for this render path",
                tilt_error_db_per_octave=round(tilt_error_db_per_octave, 2),
            )
        )
    elif tilt_error_db_per_octave >= 5.0:
        risks.append(
            _artifact_risk(
                severity="warning",
                code="flat_or_bright_tilt",
                source="audio_analysis",
                message="spectral tilt is brighter than the reference balance",
                tilt_error_db_per_octave=round(tilt_error_db_per_octave, 2),
            )
        )
    modulation_is_tremolo_like = dominant_amplitude_modulation_hz >= 2.0
    if amplitude_modulation_depth_db >= 18.0 and modulation_is_tremolo_like:
        risks.append(
            _artifact_risk(
                severity="severe",
                code="strong_amplitude_modulation",
                source="audio_analysis",
                message=(
                    "strong amplitude modulation may read as unintended tremolo "
                    f"({dominant_amplitude_modulation_hz:.2f} Hz)"
                ),
                amplitude_modulation_depth_db=round(amplitude_modulation_depth_db, 2),
                dominant_amplitude_modulation_hz=round(
                    dominant_amplitude_modulation_hz, 2
                ),
            )
        )
    elif amplitude_modulation_depth_db >= 12.0 and modulation_is_tremolo_like:
        risks.append(
            _artifact_risk(
                severity="warning",
                code="strong_amplitude_modulation",
                source="audio_analysis",
                message=(
                    "render shows pronounced amplitude modulation "
                    f"({dominant_amplitude_modulation_hz:.2f} Hz)"
                ),
                amplitude_modulation_depth_db=round(amplitude_modulation_depth_db, 2),
                dominant_amplitude_modulation_hz=round(
                    dominant_amplitude_modulation_hz, 2
                ),
            )
        )
    if integrated_lufs >= -11.0 and peak_dbfs <= -0.8:
        risks.append(
            _artifact_risk(
                severity="warning",
                code="dense_loudness_profile",
                source="audio_analysis",
                message="render is loud relative to its remaining sample-peak headroom",
                integrated_lufs=round(integrated_lufs, 2),
                peak_dbfs=round(peak_dbfs, 2),
            )
        )
    return risks


def _build_artifact_risk_report(
    *,
    mix_analysis: AudioAnalysis,
    voice_analyses: dict[str, AudioAnalysis],
    score: Score | None,
    pre_master_mix_analysis: AudioAnalysis | None,
    pre_export_mix_analysis: AudioAnalysis | None,
) -> ArtifactRiskReport:
    mix_risks = list(mix_analysis.artifact_risks)
    if pre_master_mix_analysis is not None and pre_export_mix_analysis is not None:
        loudness_delta_lufs = (
            pre_export_mix_analysis.integrated_lufs
            - pre_master_mix_analysis.integrated_lufs
        )
        centroid_delta_hz = (
            pre_export_mix_analysis.spectral_centroid_hz
            - pre_master_mix_analysis.spectral_centroid_hz
        )
        crest_delta_db = (
            pre_master_mix_analysis.crest_factor_db
            - pre_export_mix_analysis.crest_factor_db
        )
        export_density_increase = (
            pre_export_mix_analysis.clipped_sample_fraction
            > pre_master_mix_analysis.clipped_sample_fraction
            or (
                pre_export_mix_analysis.clipped_true_peak
                and not pre_master_mix_analysis.clipped_true_peak
            )
        )
        # Warn on large premaster -> post-master jumps only when the master bus
        # also shows signs of added limiting or clipping density. Final export
        # normalization is tracked separately and intentionally does not drive
        # this warning surface.
        if loudness_delta_lufs >= 8.0 and (
            crest_delta_db >= 3.0 or export_density_increase
        ):
            mix_risks.append(
                _artifact_risk(
                    severity="severe",
                    code="export_loudness_jump",
                    source="mastering_analysis",
                    message="export mastering increased loudness with strong compression or clipping-like density",
                    loudness_delta_lufs=round(loudness_delta_lufs, 2),
                    crest_factor_delta_db=round(crest_delta_db, 2),
                    pre_master_lufs=round(
                        pre_master_mix_analysis.integrated_lufs,
                        2,
                    ),
                    post_master_lufs=round(
                        pre_export_mix_analysis.integrated_lufs,
                        2,
                    ),
                    master_density_increase=export_density_increase,
                )
            )
        elif loudness_delta_lufs >= 4.5 and (
            crest_delta_db >= 1.5 or export_density_increase
        ):
            mix_risks.append(
                _artifact_risk(
                    severity="warning",
                    code="heavy_export_compression",
                    source="mastering_analysis",
                    message="export mastering increased loudness with noticeable compression or clipping-like density",
                    loudness_delta_lufs=round(loudness_delta_lufs, 2),
                    crest_factor_delta_db=round(crest_delta_db, 2),
                    pre_master_lufs=round(
                        pre_master_mix_analysis.integrated_lufs,
                        2,
                    ),
                    post_master_lufs=round(
                        pre_export_mix_analysis.integrated_lufs,
                        2,
                    ),
                    master_density_increase=export_density_increase,
                )
            )
        if centroid_delta_hz >= 1_500.0:
            mix_risks.append(
                _artifact_risk(
                    severity="warning",
                    code="export_brightness_jump",
                    source="mastering_analysis",
                    message="export mastering made the mix substantially brighter",
                    centroid_delta_hz=round(centroid_delta_hz, 1),
                    pre_master_centroid_hz=round(
                        pre_master_mix_analysis.spectral_centroid_hz,
                        1,
                    ),
                    post_master_centroid_hz=round(
                        pre_export_mix_analysis.spectral_centroid_hz,
                        1,
                    ),
                )
            )
        if crest_delta_db >= 8.0:
            mix_risks.append(
                _artifact_risk(
                    severity="severe",
                    code="heavy_export_compression",
                    source="mastering_analysis",
                    message="master bus processing collapsed crest factor dramatically",
                    crest_factor_delta_db=round(crest_delta_db, 2),
                    pre_master_crest_factor_db=round(
                        pre_master_mix_analysis.crest_factor_db,
                        2,
                    ),
                    post_master_crest_factor_db=round(
                        pre_export_mix_analysis.crest_factor_db,
                        2,
                    ),
                )
            )
        elif crest_delta_db >= 5.0:
            mix_risks.append(
                _artifact_risk(
                    severity="warning",
                    code="heavy_export_compression",
                    source="mastering_analysis",
                    message="master bus processing reduced crest factor noticeably",
                    crest_factor_delta_db=round(crest_delta_db, 2),
                    pre_master_crest_factor_db=round(
                        pre_master_mix_analysis.crest_factor_db,
                        2,
                    ),
                    post_master_crest_factor_db=round(
                        pre_export_mix_analysis.crest_factor_db,
                        2,
                    ),
                )
            )

    parameter_surface_risks = (
        _analyze_parameter_surface_risks(score) if score is not None else {}
    )
    summary = _summarize_artifact_risks(
        mix_risks=mix_risks,
        voice_risks={
            name: analysis.artifact_risks for name, analysis in voice_analyses.items()
        },
        parameter_surface_risks=parameter_surface_risks,
    )
    return ArtifactRiskReport(
        mix=_sort_risks(mix_risks),
        voices={
            voice_name: _sort_risks(analysis.artifact_risks)
            for voice_name, analysis in voice_analyses.items()
            if analysis.artifact_risks
        },
        parameter_surfaces={
            voice_name: _sort_risks(risks)
            for voice_name, risks in parameter_surface_risks.items()
            if risks
        },
        summary=summary,
    )


def _analyze_parameter_surface_risks(
    score: Score,
) -> dict[str, list[ArtifactRiskWarning]]:
    risks_by_voice: dict[str, list[ArtifactRiskWarning]] = {}
    bounded_velocity_params = {
        "filter_env_amount": (0.0, 1.0),
    }
    for voice_name, voice in score.voices.items():
        risks: list[ArtifactRiskWarning] = []
        authored_params = _collect_voice_param_values(
            score=score, voice_name=voice_name
        )

        cutoff_values = authored_params.get("cutoff_hz", [])
        filter_env_values = authored_params.get("filter_env_amount", [])
        resonance_values = authored_params.get("resonance", [])
        drive_values = authored_params.get("filter_drive", [])
        note_amp_db_values = authored_params.get("note_amp_db", [])

        cutoff_min = min(cutoff_values) if cutoff_values else 0.0
        cutoff_max = max(cutoff_values) if cutoff_values else 0.0
        cutoff_span = cutoff_max - cutoff_min
        filter_env_max = max(filter_env_values) if filter_env_values else 0.0
        resonance_max = max(resonance_values) if resonance_values else 0.0
        drive_max = max(drive_values) if drive_values else 0.0
        hottest_note_amp_db = max(note_amp_db_values) if note_amp_db_values else -120.0

        cutoff_velocity_map = voice.velocity_to_params.get("cutoff_hz")
        velocity_cutoff_span = (
            cutoff_velocity_map.max_value - cutoff_velocity_map.min_value
            if cutoff_velocity_map is not None
            else 0.0
        )
        filter_env_velocity_map = voice.velocity_to_params.get("filter_env_amount")
        velocity_filter_env_span = (
            filter_env_velocity_map.max_value - filter_env_velocity_map.min_value
            if filter_env_velocity_map is not None
            else 0.0
        )

        for param_name, (min_bound, max_bound) in bounded_velocity_params.items():
            velocity_map = voice.velocity_to_params.get(param_name)
            if velocity_map is None:
                continue
            if (
                velocity_map.min_value >= min_bound
                and velocity_map.max_value <= max_bound
            ):
                continue
            risks.append(
                _artifact_risk(
                    severity="warning",
                    code="velocity_param_out_of_bounds",
                    source="parameter_surface",
                    message=(
                        f'velocity map for "{param_name}" exceeds the expected '
                        "authored range"
                    ),
                    param_name=param_name,
                    velocity_param_min=round(velocity_map.min_value, 2),
                    velocity_param_max=round(velocity_map.max_value, 2),
                    expected_min=round(min_bound, 2),
                    expected_max=round(max_bound, 2),
                )
            )

        if (
            cutoff_span >= 1_500.0
            and filter_env_max >= 0.75
            and velocity_cutoff_span >= 700.0
        ):
            severity = (
                "severe"
                if cutoff_span >= 2_000.0 or drive_max >= 0.08 or resonance_max >= 0.12
                else "warning"
            )
            risks.append(
                _artifact_risk(
                    severity=severity,
                    code="aggressive_filter_motion",
                    source="parameter_surface",
                    message=(
                        "large cutoff motion plus strong filter modulation may create "
                        "wah, divebomb, or tremolo-like artifacts"
                    ),
                    cutoff_span_hz=round(cutoff_span, 1),
                    filter_env_amount_max=round(filter_env_max, 2),
                    velocity_cutoff_span_hz=round(velocity_cutoff_span, 1),
                    filter_drive_max=round(drive_max, 2),
                    resonance_max=round(resonance_max, 2),
                )
            )

        if cutoff_max >= 3_200.0 and hottest_note_amp_db >= -16.0:
            risks.append(
                _artifact_risk(
                    severity="warning",
                    code="bright_hot_authoring",
                    source="parameter_surface",
                    message=(
                        "bright cutoff settings combined with relatively hot note levels "
                        "may make harshness much more obvious"
                    ),
                    cutoff_hz_max=round(cutoff_max, 1),
                    hottest_note_amp_db=round(hottest_note_amp_db, 2),
                )
            )

        if drive_max >= 0.08 and filter_env_max >= 0.8 and resonance_max >= 0.1:
            risks.append(
                _artifact_risk(
                    severity="warning",
                    code="drive_resonance_interaction",
                    source="parameter_surface",
                    message=(
                        "filter drive, resonance, and envelope depth are all elevated; "
                        "this combination is easy to push into brittle artifacts"
                    ),
                    filter_drive_max=round(drive_max, 2),
                    resonance_max=round(resonance_max, 2),
                    filter_env_amount_max=round(filter_env_max, 2),
                )
            )

        if velocity_filter_env_span > 0.8 and filter_env_max >= 0.9:
            risks.append(
                _artifact_risk(
                    severity="warning",
                    code="wide_velocity_filter_env",
                    source="parameter_surface",
                    message=(
                        "velocity is driving an unusually wide filter-envelope range, "
                        "which can make accents feel like different presets"
                    ),
                    velocity_filter_env_span=round(velocity_filter_env_span, 2),
                    filter_env_amount_max=round(filter_env_max, 2),
                )
            )

        if risks:
            risks_by_voice[voice_name] = risks
    return risks_by_voice


def _collect_voice_param_values(
    *,
    score: Score,
    voice_name: str,
) -> dict[str, list[float]]:
    voice = score.voices[voice_name]
    param_values: dict[str, list[float]] = {}
    for param_name, param_value in voice.synth_defaults.items():
        if isinstance(param_value, (int, float)) and not isinstance(param_value, bool):
            param_values.setdefault(param_name, []).append(float(param_value))
    for note in voice.notes:
        merged_params = dict(voice.synth_defaults)
        if note.synth is not None:
            merged_params.update(note.synth)
        for param_name, param_value in merged_params.items():
            if isinstance(param_value, (int, float)) and not isinstance(
                param_value, bool
            ):
                param_values.setdefault(param_name, []).append(float(param_value))
        param_values.setdefault("note_amp_db", []).append(
            float(
                note.amp_db
                if note.amp_db is not None
                else synth.amp_to_db(float(note.amp or 1.0))
            )
        )
    return param_values


def _summarize_artifact_risks(
    *,
    mix_risks: list[ArtifactRiskWarning],
    voice_risks: dict[str, list[ArtifactRiskWarning]],
    parameter_surface_risks: dict[str, list[ArtifactRiskWarning]],
) -> dict[str, int]:
    all_risks = list(mix_risks)
    for warnings in voice_risks.values():
        all_risks.extend(warnings)
    for warnings in parameter_surface_risks.values():
        all_risks.extend(warnings)
    severity_counts = {"info": 0, "warning": 0, "severe": 0}
    for risk in all_risks:
        severity_counts[risk.severity] = severity_counts.get(risk.severity, 0) + 1
    return {
        "total_warning_count": len(all_risks),
        "info_count": severity_counts.get("info", 0),
        "warning_count": severity_counts.get("warning", 0),
        "severe_count": severity_counts.get("severe", 0),
        "voice_count_with_risks": sum(
            1 for warnings in voice_risks.values() if warnings
        ),
        "voice_count_with_parameter_risks": sum(
            1 for warnings in parameter_surface_risks.values() if warnings
        ),
    }


def _artifact_risk(
    *,
    severity: str,
    code: str,
    message: str,
    source: str,
    **metrics: Any,
) -> ArtifactRiskWarning:
    return ArtifactRiskWarning(
        severity=severity,
        code=code,
        message=message,
        source=source,
        metrics=metrics,
    )


def _sort_risks(
    risks: list[ArtifactRiskWarning],
) -> list[ArtifactRiskWarning]:
    severity_order = {"severe": 0, "warning": 1, "info": 2}
    return sorted(
        risks,
        key=lambda risk: (
            severity_order.get(risk.severity, 99),
            risk.code,
            risk.message,
        ),
    )


def _log_artifact_risk_report(report: ArtifactRiskReport) -> None:
    for risk in report.mix:
        _log_artifact_risk(scope="mix", risk=risk)
    for voice_name, risks in report.voices.items():
        for risk in risks:
            _log_artifact_risk(scope=f"voice:{voice_name}", risk=risk)
    for voice_name, risks in report.parameter_surfaces.items():
        for risk in risks:
            _log_artifact_risk(scope=f"params:{voice_name}", risk=risk)


def _log_artifact_risk(*, scope: str, risk: ArtifactRiskWarning) -> None:
    if risk.code in SUPPRESSED_CODES:
        return
    metric_items = ", ".join(
        f"{key}={value}" for key, value in sorted(risk.metrics.items())
    )
    logger.warning(
        "Artifact risk [%s] %s/%s: %s%s",
        scope,
        risk.severity,
        risk.code,
        risk.message,
        f" ({metric_items})" if metric_items else "",
    )


def _log_effect_analysis_warnings(effect_analysis: dict[str, Any]) -> None:
    for entry in effect_analysis.get("mix_effects", []):
        _log_effect_entry_warnings(scope="mix_fx", entry=entry)
    for voice_name, entries in effect_analysis.get("voice_effects", {}).items():
        for entry in entries:
            _log_effect_entry_warnings(scope=f"voice_fx:{voice_name}", entry=entry)
    for bus_name, entries in effect_analysis.get("send_effects", {}).items():
        for entry in entries:
            _log_effect_entry_warnings(scope=f"send_fx:{bus_name}", entry=entry)


def _log_effect_entry_warnings(*, scope: str, entry: dict[str, Any]) -> None:
    warnings = entry.get("warnings", [])
    for warning in warnings:
        metric_items = ", ".join(
            f"{key}={value}"
            for key, value in sorted(warning.get("metrics", {}).items())
        )
        log_message = "Effect analysis [%s] %s/%s (%s): %s%s"
        log_args = (
            scope,
            warning.get("severity", "warning"),
            warning.get("code", "unknown"),
            entry.get("display_name", entry.get("kind", "effect")),
            warning.get("message", ""),
            f" ({metric_items})" if metric_items else "",
        )
        if warning.get("severity") == "severe":
            logger.error(log_message, *log_args)
        else:
            logger.warning(log_message, *log_args)


def _compute_high_band_emphasis_db(band_energy_db: dict[str, float]) -> float:
    high_values = [
        band_energy_db.get("high", float("-inf")),
        band_energy_db.get("air", float("-inf")),
    ]
    body_values = [
        band_energy_db.get("low_mid", float("-inf")),
        band_energy_db.get("mid", float("-inf")),
    ]
    return _finite_mean(high_values) - _finite_mean(body_values)


def _finite_mean(values: list[float]) -> float:
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return float("-inf")
    return float(np.mean(finite_values))


def _amplitude_modulation_depth_db(
    signal: np.ndarray,
    *,
    sample_rate: int,
    window_seconds: float = 0.05,
    hop_seconds: float = 0.01,
    gate_dbfs: float = -50.0,
) -> float:
    _, frame_dbfs_values = _frame_rms_envelope(
        signal,
        sample_rate=sample_rate,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
    )
    if frame_dbfs_values.size == 0:
        return 0.0

    active_frame_dbfs_values = frame_dbfs_values[frame_dbfs_values > gate_dbfs]
    if active_frame_dbfs_values.size < 4:
        return 0.0

    return float(
        np.percentile(active_frame_dbfs_values, 95.0)
        - np.percentile(active_frame_dbfs_values, 5.0)
    )


def _dominant_amplitude_modulation_hz(
    signal: np.ndarray,
    *,
    sample_rate: int,
    window_seconds: float = 0.05,
    hop_seconds: float = 0.01,
    gate_dbfs: float = -50.0,
    min_frequency_hz: float = 0.25,
    max_frequency_hz: float = 12.0,
) -> float:
    frame_rate_hz, frame_dbfs_values = _frame_rms_envelope(
        signal,
        sample_rate=sample_rate,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
    )
    if frame_dbfs_values.size < 8:
        return 0.0

    active_mask = frame_dbfs_values > gate_dbfs
    if np.count_nonzero(active_mask) < 8:
        return 0.0

    normalized_envelope = np.maximum(0.0, frame_dbfs_values - gate_dbfs)
    centered_envelope = normalized_envelope - float(np.mean(normalized_envelope))
    if np.allclose(centered_envelope, 0.0):
        return 0.0

    window = np.hanning(centered_envelope.size)
    spectrum = np.fft.rfft(centered_envelope * window)
    frequencies_hz = np.fft.rfftfreq(centered_envelope.size, d=1.0 / frame_rate_hz)
    candidate_mask = (frequencies_hz >= min_frequency_hz) & (
        frequencies_hz <= max_frequency_hz
    )
    if not np.any(candidate_mask):
        return 0.0

    magnitudes = np.abs(spectrum[candidate_mask])
    if np.allclose(magnitudes, 0.0):
        return 0.0

    candidate_frequencies_hz = frequencies_hz[candidate_mask]
    return float(candidate_frequencies_hz[int(np.argmax(magnitudes))])


def _frame_rms_envelope(
    signal: np.ndarray,
    *,
    sample_rate: int,
    window_seconds: float,
    hop_seconds: float,
) -> tuple[float, np.ndarray]:
    if signal.size == 0:
        return 0.0, np.asarray([], dtype=np.float64)

    window_samples = max(1, int(round(window_seconds * sample_rate)))
    hop_samples = max(1, int(round(hop_seconds * sample_rate)))
    frame_dbfs_values: list[float] = []
    for start in range(0, signal.size, hop_samples):
        frame = signal[start : start + window_samples]
        if frame.size == 0:
            continue
        frame_rms = float(np.sqrt(np.mean(np.square(frame))))
        frame_dbfs_values.append(_amplitude_to_db(frame_rms))

    return sample_rate / float(hop_samples), np.asarray(
        frame_dbfs_values,
        dtype=np.float64,
    )


def _save_spectrum_plot(
    *,
    signal: np.ndarray,
    sample_rate: int,
    path: Path,
    title: str,
    reference_tilt_db_per_octave: float,
) -> None:
    freqs, magnitude_db = _average_spectrum(
        synth.to_mono_reference(signal),
        sample_rate=sample_rate,
    )
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.semilogx(freqs, magnitude_db, linewidth=1.2, label="measured")

    reference_curve = magnitude_db[0] + reference_tilt_db_per_octave * (
        np.log2(freqs) - np.log2(freqs[0])
    )
    axis.semilogx(
        freqs,
        reference_curve,
        linestyle="--",
        linewidth=1.0,
        label=f"reference {reference_tilt_db_per_octave:.1f} dB/oct",
    )
    axis.set_title(title)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Magnitude (dB, relative)")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _save_spectrogram_plot(
    *,
    signal: np.ndarray,
    sample_rate: int,
    path: Path,
    title: str,
) -> None:
    mono_signal = synth.to_mono_reference(signal)
    if mono_signal.size == 0:
        mono_signal = np.zeros(256, dtype=np.float64)
    nperseg = min(2048, max(256, mono_signal.size))
    noverlap = min(max(nperseg // 2, 1), nperseg - 1)
    freqs, times, spec = spectrogram(
        mono_signal,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )
    spec_db = 20.0 * np.log10(np.maximum(spec, _EPSILON))
    figure, axis = plt.subplots(figsize=(10, 4))
    mesh = axis.pcolormesh(times, freqs, spec_db, shading="auto")
    axis.set_ylim(20.0, min(sample_rate / 2.0, 12_000.0))
    axis.set_title(title)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Frequency (Hz)")
    figure.colorbar(mesh, ax=axis, label="Magnitude (dB)")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _save_band_energy_plot(
    *,
    band_energy_db: dict[str, float],
    path: Path,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    labels = list(band_energy_db)
    values = [band_energy_db[label] for label in labels]
    axis.bar(labels, values, color="#4c72b0")
    axis.set_title(title)
    axis.set_ylabel("Mean magnitude (dB)")
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _save_score_density_plot(*, score: Score, path: Path) -> None:
    total_duration = score.total_dur
    if total_duration == 0:
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.set_title("Score Density")
        axis.text(0.5, 0.5, "Empty score", ha="center", va="center")
        figure.tight_layout()
        figure.savefig(path, bbox_inches="tight")
        plt.close(figure)
        return

    window_seconds = 1.0
    window_starts = np.arange(0.0, total_duration, window_seconds)
    onset_counts: list[int] = []
    active_counts: list[int] = []
    for window_start in window_starts:
        window_end = window_start + window_seconds
        onsets = 0
        active = 0
        for voice in score.voices.values():
            for note in voice.notes:
                if window_start <= note.start < window_end:
                    onsets += 1
                if (
                    note.start < window_end
                    and (note.start + note.duration) > window_start
                ):
                    active += 1
        onset_counts.append(onsets)
        active_counts.append(active)

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(window_starts, onset_counts, label="attacks / sec", linewidth=1.5)
    axis.plot(window_starts, active_counts, label="active notes", linewidth=1.5)
    axis.set_title("Score Density")
    axis.set_xlabel("Time (seconds)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _amplitude_to_db(amplitude: float) -> float:
    return float(20.0 * np.log10(max(float(amplitude), _EPSILON)))


def _sanitize_name(name: str) -> str:
    return "".join(
        character if character.isalnum() else "_" for character in name
    ).strip("_")


def _json_dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _compare_numeric_dicts(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    keys: tuple[str, ...],
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in keys:
        if key in before and key in after:
            deltas[key] = float(after[key]) - float(before[key])
    return deltas
