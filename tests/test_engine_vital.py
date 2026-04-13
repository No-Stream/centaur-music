"""Tests for the Vital instrument engine -- integration and FFT pitch verification."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from code_musics.engines.vital import render_voice
from code_musics.synth import has_external_plugin

requires_vital = pytest.mark.skipif(
    not has_external_plugin("vital"),
    reason="Vital VST3 not installed",
)


def _cents_error(actual_hz: float, expected_hz: float) -> float:
    return abs(1200.0 * math.log2(actual_hz / expected_hz))


def _detect_fundamental_hz(audio: np.ndarray, sample_rate: int) -> float:
    """Detect fundamental frequency via FFT peak with parabolic interpolation."""
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    skip_samples = int(0.05 * sample_rate)
    segment = mono[skip_samples:]
    windowed = segment * np.hanning(len(segment))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / sample_rate)
    min_bin = max(1, int(20.0 / (sample_rate / len(windowed))))
    peak_bin = min_bin + int(np.argmax(spectrum[min_bin:]))
    if 0 < peak_bin < len(spectrum) - 1:
        alpha = float(spectrum[peak_bin - 1])
        beta = float(spectrum[peak_bin])
        gamma = float(spectrum[peak_bin + 1])
        denom = alpha - 2.0 * beta + gamma
        if denom != 0:
            correction = 0.5 * (alpha - gamma) / denom
            return float(freqs[peak_bin] + correction * (freqs[1] - freqs[0]))
    return float(freqs[peak_bin])


# ===========================================================================
# Group 1: render_voice integration (requires Vital)
# ===========================================================================


class TestRenderVoiceIntegration:
    @requires_vital
    def test_single_note_produces_nonsilent_stereo(self) -> None:
        audio = render_voice(
            notes=[
                {
                    "freq": 440.0,
                    "start": 0.0,
                    "duration": 0.3,
                    "velocity": 0.8,
                    "amp": 1.0,
                }
            ],
            total_duration=0.3,
            params={"tail_seconds": 0.5},
        )
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float64
        assert audio.ndim == 2
        assert audio.shape[0] == 2
        assert audio.shape[1] > 0
        assert np.max(np.abs(audio)) > 0, "audio is silent"
        assert np.all(np.isfinite(audio))

    @requires_vital
    def test_overlapping_ji_chord(self) -> None:
        f0 = 220.0
        notes = [
            {"freq": f0, "start": 0.0, "duration": 0.4, "velocity": 0.8, "amp": 1.0},
            {
                "freq": f0 * 5 / 4,
                "start": 0.0,
                "duration": 0.4,
                "velocity": 0.8,
                "amp": 1.0,
            },
            {
                "freq": f0 * 3 / 2,
                "start": 0.0,
                "duration": 0.4,
                "velocity": 0.8,
                "amp": 1.0,
            },
            {
                "freq": f0 * 7 / 4,
                "start": 0.0,
                "duration": 0.4,
                "velocity": 0.8,
                "amp": 1.0,
            },
        ]
        audio = render_voice(
            notes=notes, total_duration=0.4, params={"tail_seconds": 0.5}
        )
        assert np.max(np.abs(audio)) > 0
        assert np.all(np.isfinite(audio))

    @requires_vital
    def test_empty_notes_no_crash(self) -> None:
        audio = render_voice(notes=[], total_duration=0.5, params={"tail_seconds": 0.3})
        assert isinstance(audio, np.ndarray)

    def test_silence_fallback_when_missing(self) -> None:
        with patch("code_musics.engines.vital.has_external_plugin", return_value=False):
            audio = render_voice(
                notes=[
                    {
                        "freq": 440.0,
                        "start": 0.0,
                        "duration": 0.3,
                        "velocity": 0.8,
                        "amp": 1.0,
                    }
                ],
                total_duration=0.3,
            )
        assert np.allclose(audio, 0.0)
        assert audio.shape[0] == 2


# ===========================================================================
# Group 2: Score-level smoke test (requires Vital)
# ===========================================================================


class TestScoreIntegration:
    @requires_vital
    def test_score_renders_vital_voice(self) -> None:
        from code_musics.score import Score

        score = Score(f0=220.0)
        score.add_voice(
            "vital_pad",
            synth_defaults={"engine": "vital", "tail_seconds": 0.5},
            normalize_lufs=-24.0,
        )
        for partial, start in [(1.0, 0.0), (5 / 4, 0.1), (3 / 2, 0.2)]:
            score.add_note(
                "vital_pad",
                partial=partial,
                start=start,
                duration=0.3,
            )
        audio = score.render()
        assert audio.ndim == 2
        assert audio.shape[0] == 2
        assert audio.shape[1] > 0
        assert np.max(np.abs(audio)) > 0, "rendered audio is silent"


# ===========================================================================
# Group 3: FFT pitch verification (requires Vital)
# ===========================================================================


class TestFFTPitchVerification:
    @requires_vital
    @pytest.mark.parametrize(
        "ratio",
        [1.0, 9 / 8, 5 / 4, 11 / 8, 3 / 2, 7 / 4],
        ids=["1/1", "9/8", "5/4", "11/8", "3/2", "7/4"],
    )
    def test_rendered_pitch_matches_ji_ratio(self, ratio: float) -> None:
        f0 = 220.0
        target_hz = f0 * ratio
        audio = render_voice(
            notes=[
                {
                    "freq": target_hz,
                    "start": 0.0,
                    "duration": 0.8,
                    "velocity": 0.9,
                    "amp": 1.0,
                }
            ],
            total_duration=0.8,
            sample_rate=44100,
            params={"tail_seconds": 0.2},
        )
        detected = _detect_fundamental_hz(audio, 44100)
        error = _cents_error(detected, target_hz)
        assert error < 5.0, (
            f"ratio {ratio}: expected {target_hz:.2f} Hz, "
            f"detected {detected:.2f} Hz, error {error:.1f} cents"
        )

    @requires_vital
    @pytest.mark.parametrize("f0", [55.0, 110.0, 220.0, 440.0, 880.0])
    def test_pitch_accuracy_across_octaves(self, f0: float) -> None:
        target_hz = f0 * 3 / 2
        audio = render_voice(
            notes=[
                {
                    "freq": target_hz,
                    "start": 0.0,
                    "duration": 0.8,
                    "velocity": 0.9,
                    "amp": 1.0,
                }
            ],
            total_duration=0.8,
            sample_rate=44100,
            params={"tail_seconds": 0.2},
        )
        detected = _detect_fundamental_hz(audio, 44100)
        error = _cents_error(detected, target_hz)
        assert error < 5.0, (
            f"f0={f0}: expected {target_hz:.2f} Hz, "
            f"detected {detected:.2f} Hz, error {error:.1f} cents"
        )


# ===========================================================================
# Group 4: High-precision pitch accuracy (requires Vital, expensive)
# ===========================================================================

# Longer notes at higher sample rate for tighter FFT bins.  Theoretical limit
# is ~0.29 cents (one pitch-bend LSB at 24-semitone range).  We assert < 0.5
# cents, well within the MPE quantisation floor.

_PRECISION_RATIOS: list[tuple[str, float]] = [
    ("5/4", 5 / 4),
    ("7/4", 7 / 4),
    ("11/8", 11 / 8),
    ("13/8", 13 / 8),
    ("7/6", 7 / 6),
    ("9/7", 9 / 7),
    ("11/10", 11 / 10),
    ("15/11", 15 / 11),
]


class TestHighPrecisionPitch:
    @requires_vital
    @pytest.mark.parametrize(
        "ratio",
        [r for _, r in _PRECISION_RATIOS],
        ids=[name for name, _ in _PRECISION_RATIOS],
    )
    def test_sub_half_cent_accuracy(self, ratio: float) -> None:
        f0 = 220.0
        target_hz = f0 * ratio
        sr = 96000
        audio = render_voice(
            notes=[
                {
                    "freq": target_hz,
                    "start": 0.0,
                    "duration": 2.0,
                    "velocity": 0.9,
                    "amp": 1.0,
                }
            ],
            total_duration=2.0,
            sample_rate=sr,
            params={"tail_seconds": 0.5},
        )
        detected = _detect_fundamental_hz(audio, sr)
        error = _cents_error(detected, target_hz)
        assert error < 0.5, (
            f"ratio {ratio}: expected {target_hz:.4f} Hz, "
            f"detected {detected:.4f} Hz, error {error:.3f} cents"
        )


# ===========================================================================
# Group 5: Channel pitch isolation (requires Vital)
# ===========================================================================


class TestChannelPitchIsolation:
    """Two overlapping notes at very different pitches should stay independent.

    If MPE channel isolation works, the first note stays at its target even
    after the second note's pitch bend arrives on a different channel.
    """

    @requires_vital
    def test_overlapping_notes_maintain_independent_pitch(self) -> None:
        freq_low = 220.0
        freq_high = 440.0
        sr = 44100

        audio = render_voice(
            notes=[
                {
                    "freq": freq_low,
                    "start": 0.0,
                    "duration": 0.8,
                    "velocity": 0.9,
                    "amp": 1.0,
                },
                {
                    "freq": freq_high,
                    "start": 0.3,
                    "duration": 0.5,
                    "velocity": 0.9,
                    "amp": 1.0,
                },
            ],
            total_duration=0.8,
            sample_rate=sr,
            params={"tail_seconds": 0.3},
        )

        # Analyse the first 0.25s (only the low note is sounding)
        solo_end = int(0.25 * sr)
        solo_segment = audio[:, :solo_end]
        detected_solo = _detect_fundamental_hz(solo_segment, sr)

        # Analyse 0.35-0.75s (both notes overlap)
        overlap_start = int(0.35 * sr)
        overlap_end = int(0.75 * sr)
        overlap_segment = audio[:, overlap_start:overlap_end]
        mono_overlap = overlap_segment.mean(axis=0)
        windowed = mono_overlap * np.hanning(len(mono_overlap))
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(windowed), d=1.0 / sr)

        # Find the two strongest peaks (should be near freq_low and freq_high)
        min_bin = max(1, int(100.0 / (sr / len(windowed))))
        max_bin = int(600.0 / (sr / len(windowed)))
        low_peak_bin = min_bin + int(np.argmax(spectrum[min_bin:max_bin]))
        detected_low_overlap = float(freqs[low_peak_bin])

        # The low note should not have drifted toward the high note
        drift = _cents_error(detected_low_overlap, freq_low)
        assert drift < 50.0, (
            f"Low note drifted {drift:.1f} cents during overlap -- "
            f"MPE channel isolation may be broken "
            f"(solo={detected_solo:.1f} Hz, overlap={detected_low_overlap:.1f} Hz)"
        )

        # Basic sanity: solo segment should be near freq_low
        solo_error = _cents_error(detected_solo, freq_low)
        assert solo_error < 5.0, (
            f"Solo low note off by {solo_error:.1f} cents "
            f"(detected {detected_solo:.1f} Hz, expected {freq_low:.1f} Hz)"
        )


# ===========================================================================
# Group 6: Glide smoke test (requires Vital)
# ===========================================================================


class TestGlideSmoke:
    """Verify that glide_from/glide_time produces a pitch sweep."""

    @requires_vital
    def test_glide_produces_pitch_movement(self) -> None:
        freq_start = 220.0
        freq_end = 330.0
        sr = 44100

        audio = render_voice(
            notes=[
                {
                    "freq": freq_end,
                    "start": 0.0,
                    "duration": 1.0,
                    "velocity": 0.9,
                    "amp": 1.0,
                    "glide_from": freq_start,
                    "glide_time": 0.8,
                }
            ],
            total_duration=1.0,
            sample_rate=sr,
            params={"tail_seconds": 0.3},
        )

        # Detect pitch in the early portion (should be near freq_start)
        early_end = int(0.15 * sr)
        early_segment = audio[:, :early_end]
        detected_early = _detect_fundamental_hz(early_segment, sr)

        # Detect pitch in the late portion (should be near freq_end)
        late_start = int(0.85 * sr)
        late_end = int(1.0 * sr)
        late_segment = audio[:, late_start:late_end]
        detected_late = _detect_fundamental_hz(late_segment, sr)

        # Early pitch should be closer to freq_start than to freq_end
        early_to_start = _cents_error(detected_early, freq_start)
        early_to_end = _cents_error(detected_early, freq_end)
        assert early_to_start < early_to_end, (
            f"Early pitch {detected_early:.1f} Hz is closer to end "
            f"({freq_end} Hz) than start ({freq_start} Hz) -- "
            f"glide may not be working"
        )

        # Late pitch should be closer to freq_end than to freq_start
        late_to_start = _cents_error(detected_late, freq_start)
        late_to_end = _cents_error(detected_late, freq_end)
        assert late_to_end < late_to_start, (
            f"Late pitch {detected_late:.1f} Hz is closer to start "
            f"({freq_start} Hz) than end ({freq_end} Hz) -- "
            f"glide may not be working"
        )
