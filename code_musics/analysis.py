"""Render and score analysis helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from code_musics import synth
from code_musics.score import Score

_EPSILON = 1e-12
_DEFAULT_BANDS: tuple[tuple[str, float, float], ...] = (
    ("sub", 20.0, 60.0),
    ("bass", 60.0, 250.0),
    ("low_mid", 250.0, 500.0),
    ("mid", 500.0, 2_000.0),
    ("high", 2_000.0, 8_000.0),
    ("air", 8_000.0, 20_000.0),
)


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
    spectral_tilt_db_per_octave: float
    reference_tilt_db_per_octave: float
    tilt_error_db_per_octave: float
    band_energy_db: dict[str, float]
    warnings: list[str]

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
            spectral_tilt_db_per_octave=0.0,
            reference_tilt_db_per_octave=reference_tilt_db_per_octave,
            tilt_error_db_per_octave=-reference_tilt_db_per_octave,
            band_energy_db={name: float("-inf") for name, _, _ in _DEFAULT_BANDS},
            warnings=["empty render"],
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
        spectral_tilt_db_per_octave=spectral_tilt,
        reference_tilt_db_per_octave=reference_tilt_db_per_octave,
        tilt_error_db_per_octave=spectral_tilt - reference_tilt_db_per_octave,
        band_energy_db=band_energy_db,
        warnings=warnings,
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
        warnings=warnings,
    )


def save_analysis_artifacts(
    *,
    output_prefix: str | Path,
    mix_signal: np.ndarray,
    sample_rate: int,
    pre_export_mix_signal: np.ndarray | None = None,
    stems: dict[str, np.ndarray] | None = None,
    score: Score | None = None,
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
    manifest: dict[str, Any] = {
        "reference_tilt_db_per_octave": reference_tilt_db_per_octave,
        "mix": {
            "summary": mix_analysis.to_dict(),
            "artifacts": {},
        },
        "voices": {},
    }
    if pre_export_mix_signal is not None:
        manifest["mix"]["pre_export_summary"] = analyze_audio(
            pre_export_mix_signal,
            sample_rate=sample_rate,
            reference_tilt_db_per_octave=reference_tilt_db_per_octave,
        ).to_dict()

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
        manifest["score"] = {
            "summary": score_analysis.to_dict(),
            "artifacts": {"density": str(score_density_path)},
        }

    for voice_name, stem_signal in (stems or {}).items():
        voice_analysis = analyze_audio(
            stem_signal,
            sample_rate=sample_rate,
            reference_tilt_db_per_octave=reference_tilt_db_per_octave,
        )
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
