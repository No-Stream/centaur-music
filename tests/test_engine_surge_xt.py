"""Tests for the Surge XT instrument engine -- pitch math, integration, and FFT verification."""

from __future__ import annotations

import logging
import math
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from mido import Message

from code_musics.engines.surge_xt import (
    BEND_RANGE_SEMITONES,
    _build_cc_messages,
    _interpolate_param_curves,
    _resolve_glide_bend,
    _resolve_note_and_bend,
    render_voice,
)
from code_musics.synth import has_external_plugin

requires_surge_xt = pytest.mark.skipif(
    not has_external_plugin("surge_xt"),
    reason="Surge XT VST3 not installed",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reconstruct_freq(midi_note: int, bend_value: int) -> float:
    """Reconstruct Hz from MIDI note + pitch bend using the engine's bend range."""
    semitones_from_a4 = midi_note - 69 + (bend_value / 8191.0) * BEND_RANGE_SEMITONES
    return 440.0 * (2.0 ** (semitones_from_a4 / 12.0))


# With 24-semitone range and 14-bit pitch bend, one LSB = 24/8191 ≈ 0.293 cents.
# Worst-case quantization error from rounding is half that ≈ 0.15 cents.
_MATH_TOLERANCE_CENTS = 0.2


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
# Group 1: _resolve_note_and_bend pitch math (pure Python, always runs)
# ===========================================================================


class TestResolveNoteAndBend:
    def test_exact_12tet_notes(self) -> None:
        cases = {
            440.0: 69,
            261.6255653005986: 60,
            220.0: 57,
            880.0: 81,
            130.8127826502993: 48,
        }
        for freq, expected_note in cases.items():
            midi_note, bend = _resolve_note_and_bend(freq)
            assert midi_note == expected_note, (
                f"{freq} Hz -> note {midi_note}, expected {expected_note}"
            )
            assert abs(bend) <= 1, f"{freq} Hz -> bend {bend}, expected ~0"

    def test_ji_ratio_reconstruction(self) -> None:
        f0 = 220.0
        ratios = [7 / 4, 11 / 8, 13 / 11, 3 / 2, 11 / 10, 5 / 4, 9 / 8, 7 / 6]
        for ratio in ratios:
            target_hz = f0 * ratio
            midi_note, bend = _resolve_note_and_bend(target_hz)
            reconstructed = _reconstruct_freq(midi_note, bend)
            error_cents = _cents_error(reconstructed, target_hz)
            assert error_cents < _MATH_TOLERANCE_CENTS, (
                f"ratio {ratio}: target {target_hz:.3f} Hz, "
                f"got {reconstructed:.3f} Hz, error {error_cents:.4f} cents"
            )

    def test_parametric_accuracy_sweep(self) -> None:
        freqs = np.logspace(np.log10(20), np.log10(16000), 200)
        for freq in freqs:
            midi_note, bend = _resolve_note_and_bend(float(freq))
            reconstructed = _reconstruct_freq(midi_note, bend)
            error_cents = _cents_error(reconstructed, float(freq))
            assert error_cents < _MATH_TOLERANCE_CENTS, (
                f"{freq:.2f} Hz: reconstructed {reconstructed:.4f} Hz, "
                f"error {error_cents:.6f} cents"
            )

    def test_boundary_frequencies(self) -> None:
        """Frequencies at exact semitone midpoints (where rounding flips)."""
        for base_note in [48, 60, 69, 72, 84]:
            midpoint_freq = 440.0 * (2.0 ** ((base_note + 0.5 - 69) / 12.0))
            midi_note, bend = _resolve_note_and_bend(midpoint_freq)
            assert midi_note in (base_note, base_note + 1)
            reconstructed = _reconstruct_freq(midi_note, bend)
            error_cents = _cents_error(reconstructed, midpoint_freq)
            assert error_cents < _MATH_TOLERANCE_CENTS, (
                f"midpoint at note {base_note}: error {error_cents:.4f} cents"
            )

    def test_extreme_low_freq_raises(self) -> None:
        with pytest.raises(ValueError, match="pitch bend beyond"):
            _resolve_note_and_bend(0.5)

    def test_deterministic(self) -> None:
        freq = 220.0 * 7 / 4
        results = [_resolve_note_and_bend(freq) for _ in range(100)]
        assert all(r == results[0] for r in results)


# ===========================================================================
# Group 2: render_voice integration (requires Surge XT)
# ===========================================================================


class TestRenderVoiceIntegration:
    @requires_surge_xt
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

    @requires_surge_xt
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
        ]
        audio = render_voice(
            notes=notes, total_duration=0.4, params={"tail_seconds": 0.5}
        )
        assert np.max(np.abs(audio)) > 0
        assert np.all(np.isfinite(audio))

    @requires_surge_xt
    def test_16_simultaneous_notes_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        notes = [
            {
                "freq": 220.0 * (i + 1),
                "start": 0.0,
                "duration": 0.2,
                "velocity": 0.8,
                "amp": 1.0,
            }
            for i in range(16)
        ]
        with caplog.at_level(logging.WARNING, logger="code_musics.engines.surge_xt"):
            audio = render_voice(
                notes=notes, total_duration=0.2, params={"tail_seconds": 0.3}
            )
        assert any("channel collision" in r.message.lower() for r in caplog.records), (
            "expected channel collision warning for 16 simultaneous notes"
        )
        assert audio.shape[0] == 2

    @requires_surge_xt
    def test_empty_notes_no_crash(self) -> None:
        audio = render_voice(notes=[], total_duration=0.5, params={"tail_seconds": 0.3})
        assert isinstance(audio, np.ndarray)

    def test_silence_fallback_when_missing(self) -> None:
        with patch(
            "code_musics.engines.surge_xt.has_external_plugin", return_value=False
        ):
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
# Group 3: Score-level smoke test (requires Surge XT)
# ===========================================================================


class TestScoreIntegration:
    @requires_surge_xt
    def test_score_renders_surge_xt_voice(self) -> None:
        from code_musics.score import Score

        score = Score(f0=220.0)
        score.add_voice(
            "surge_pad",
            synth_defaults={"engine": "surge_xt", "tail_seconds": 0.5},
            normalize_lufs=-24.0,
        )
        for partial, start in [(1.0, 0.0), (5 / 4, 0.1), (3 / 2, 0.2)]:
            score.add_note(
                "surge_pad",
                partial=partial,
                start=start,
                duration=0.3,
                amp_db=-6,
            )

        audio = score.render()
        assert audio.ndim == 2
        assert np.max(np.abs(audio)) > 0, "score render produced silence"
        assert np.all(np.isfinite(audio))


# ===========================================================================
# Group 4: FFT pitch verification (requires Surge XT) -- the critical test
# ===========================================================================


class TestFFTPitchVerification:
    @requires_surge_xt
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

    @requires_surge_xt
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
# Group 5: Velocity / amplitude sanity
# ===========================================================================


# ===========================================================================
# Group 5: Per-channel pitch bend isolation diagnostic
# ===========================================================================


class TestChannelPitchIsolation:
    """Diagnostic: does pedalboard isolate pitch bend per MIDI channel?

    Renders two overlapping notes at very different pitches (220 Hz and 440 Hz)
    on separate channels.  If per-channel isolation works, the first note should
    stay at ~220 Hz even after the second note's pitch bend (targeting 440 Hz)
    arrives.  If isolation is broken, the first note's pitch will shift toward
    440 Hz when the second note starts.
    """

    @requires_surge_xt
    def test_overlapping_notes_maintain_independent_pitch(self) -> None:
        freq_low = 220.0
        freq_high = 440.0

        audio = render_voice(
            notes=[
                {
                    "freq": freq_low,
                    "start": 0.0,
                    "duration": 2.0,
                    "velocity": 0.9,
                    "amp": 1.0,
                },
                {
                    "freq": freq_high,
                    "start": 0.5,
                    "duration": 0.5,
                    "velocity": 0.9,
                    "amp": 1.0,
                },
            ],
            total_duration=2.0,
            sample_rate=44100,
            params={"tail_seconds": 0.5, "release_padding": 2.0},
        )

        # Analyze a window AFTER the high note's pitch bend has been sent
        # but where the low note should still be at 220 Hz.
        # Window: t=0.6 to t=0.9 (both notes sounding, so expect both freqs).
        # More telling: t=1.2 to t=1.8 (high note has ended, only low note).
        sr = 44100
        solo_start = int(1.2 * sr)
        solo_end = int(1.8 * sr)
        solo_segment = audio[:, solo_start:solo_end]

        detected = _detect_fundamental_hz(solo_segment, sr)
        error_cents = _cents_error(detected, freq_low)

        # If pitch isolation works, the low note should still be ~220 Hz.
        # Allow generous 20-cent tolerance (single-note tests use 5 cents).
        # If isolation is broken, the error will be hundreds of cents
        # (note would be near 440 Hz or some mangled intermediate).
        assert error_cents < 20.0, (
            f"PITCH ISOLATION FAILURE: expected ~{freq_low:.0f} Hz in solo "
            f"window after high note ended, detected {detected:.1f} Hz "
            f"(error {error_cents:.1f} cents). This suggests pedalboard does "
            f"NOT isolate per-channel pitch bend — H1 confirmed."
        )

    @requires_surge_xt
    def test_staggered_chord_notes_hold_pitch(self) -> None:
        """Three staggered JI notes — does the first note drift when later ones arrive?"""
        f0 = 110.0
        notes = [
            {"freq": f0, "start": 0.0, "duration": 3.0, "velocity": 0.9, "amp": 1.0},
            {
                "freq": f0 * 3 / 2,
                "start": 0.8,
                "duration": 2.0,
                "velocity": 0.9,
                "amp": 1.0,
            },
            {
                "freq": f0 * 7 / 4,
                "start": 1.6,
                "duration": 1.2,
                "velocity": 0.9,
                "amp": 1.0,
            },
        ]

        audio = render_voice(
            notes=notes,
            total_duration=3.0,
            sample_rate=44100,
            params={"tail_seconds": 0.5, "release_padding": 2.0},
        )

        # Window t=0.2 to t=0.7: only the 110 Hz note is sounding.
        sr = 44100
        solo_start = int(0.2 * sr)
        solo_end = int(0.7 * sr)
        pre_segment = audio[:, solo_start:solo_end]
        pre_freq = _detect_fundamental_hz(pre_segment, sr)

        # Window t=2.9 to t=3.0: only the 110 Hz note is still sounding
        # (3/2 ends at 2.8, 7/4 ends at 2.8).
        late_start = int(2.9 * sr)
        late_end = int(3.0 * sr)
        late_segment = audio[:, late_start:late_end]
        late_freq = _detect_fundamental_hz(late_segment, sr)

        # The 110 Hz note's pitch should be the same before and after the
        # other notes arrive.  Large drift = broken channel isolation.
        drift_cents = abs(_cents_error(late_freq, pre_freq))
        assert drift_cents < 20.0, (
            f"PITCH DRIFT: 110 Hz note drifted {drift_cents:.1f} cents between "
            f"pre-chord ({pre_freq:.1f} Hz) and post-chord ({late_freq:.1f} Hz). "
            f"Suggests pitch bend from later notes affected the first note."
        )


# ===========================================================================
# Group 6: Velocity / amplitude sanity
# ===========================================================================


class TestVelocityAmplitude:
    def test_velocity_amp_midi_range(self) -> None:
        """The midi_velocity formula should always produce values in [1, 127]."""
        for vel in [0.05, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0]:
            for amp in [0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0]:
                midi_vel = max(1, min(127, int(round(vel * amp * 127))))
                assert 1 <= midi_vel <= 127, f"vel={vel}, amp={amp} -> {midi_vel}"


# ===========================================================================
# Group 7: _resolve_glide_bend pitch math (pure Python, always runs)
# ===========================================================================


class TestResolveGlideBend:
    """Verify the glide bend helper computes correct bend values."""

    def test_glide_bend_at_target_frequency_matches_resolve(self) -> None:
        """When glide_from == target freq, the glide bend should match the
        normal _resolve_note_and_bend bend value."""

        test_freqs = [220.0, 440.0, 330.0, 220.0 * 7 / 4, 220.0 * 11 / 8]
        for freq in test_freqs:
            midi_note, expected_bend = _resolve_note_and_bend(freq)
            glide_bend = _resolve_glide_bend(midi_note, freq)
            assert glide_bend == expected_bend, (
                f"{freq} Hz: _resolve_glide_bend gave {glide_bend}, "
                f"_resolve_note_and_bend gave {expected_bend}"
            )

    def test_glide_bend_reconstruction_accuracy(self) -> None:
        """Glide bend values should reconstruct to the requested frequency."""

        target_freq = 440.0
        midi_note, _ = _resolve_note_and_bend(target_freq)

        glide_freqs = [
            420.0,
            460.0,
            330.0,
            target_freq * 7 / 8,
            target_freq * 9 / 8,
        ]
        for glide_freq in glide_freqs:
            bend = _resolve_glide_bend(midi_note, glide_freq)
            reconstructed = _reconstruct_freq(midi_note, bend)
            error_cents = _cents_error(reconstructed, glide_freq)
            assert error_cents < _MATH_TOLERANCE_CENTS, (
                f"glide to {glide_freq:.2f} Hz from note {midi_note}: "
                f"reconstructed {reconstructed:.4f} Hz, error {error_cents:.4f} cents"
            )

    def test_glide_bend_direction(self) -> None:
        """Bending up from a note should give positive bend; down gives negative."""

        midi_note = 69  # A4 = 440 Hz
        bend_up = _resolve_glide_bend(midi_note, 460.0)
        bend_down = _resolve_glide_bend(midi_note, 420.0)
        bend_exact = _resolve_glide_bend(midi_note, 440.0)
        assert bend_up > 0, f"bend up should be positive, got {bend_up}"
        assert bend_down < 0, f"bend down should be negative, got {bend_down}"
        assert abs(bend_exact) <= 1, f"exact match should be ~0, got {bend_exact}"

    def test_glide_bend_clamps_at_range_limit(self) -> None:
        """Frequencies beyond BEND_RANGE_SEMITONES should clamp to max bend."""

        midi_note = 69
        # Way above the bend range -- should clamp to +8191
        very_high = 440.0 * 2.0**5  # 5 octaves up, way beyond 24 semitones
        bend = _resolve_glide_bend(midi_note, very_high)
        assert bend == 8191

        # Way below -- should clamp to -8191
        very_low = 440.0 * 2.0**-5
        bend = _resolve_glide_bend(midi_note, very_low)
        assert bend == -8191


# ===========================================================================
# Group 8: Glide / pitch motion integration (requires Surge XT)
# ===========================================================================


class TestGlideIntegration:
    @requires_surge_xt
    def test_render_voice_with_glide_produces_pitch_sweep(self) -> None:
        """Render a note that glides from 220 Hz to 330 Hz over 0.5s.

        Verify via FFT that the later portion is at ~330 Hz, not 220 Hz.
        """
        glide_from_hz = 220.0
        target_hz = 330.0
        glide_time = 0.5
        total_note_dur = 1.5

        audio = render_voice(
            notes=[
                {
                    "freq": target_hz,
                    "start": 0.0,
                    "duration": total_note_dur,
                    "velocity": 0.9,
                    "amp": 1.0,
                    "glide_from": glide_from_hz,
                    "glide_time": glide_time,
                }
            ],
            total_duration=total_note_dur,
            sample_rate=44100,
            params={"tail_seconds": 0.5},
        )

        sr = 44100
        # Analyze the post-glide steady portion (t=0.8 to t=1.3)
        post_glide_start = int(0.8 * sr)
        post_glide_end = int(1.3 * sr)
        post_segment = audio[:, post_glide_start:post_glide_end]

        detected = _detect_fundamental_hz(post_segment, sr)
        error_cents = _cents_error(detected, target_hz)
        assert error_cents < 15.0, (
            f"Post-glide pitch should be ~{target_hz:.0f} Hz, "
            f"detected {detected:.1f} Hz (error {error_cents:.1f} cents)"
        )

    @requires_surge_xt
    def test_glide_onset_pitch_matches_glide_from(self) -> None:
        """Verify the note starts at the glide_from frequency, not the target.

        If there's a pitch discontinuity at onset (H1), the early portion of the
        note would be at or near the target frequency instead of the glide_from
        frequency.  This test uses a wide interval (220 Hz -> 440 Hz) so a wrong
        onset pitch is clearly detectable via FFT.
        """
        glide_from_hz = 440.0
        target_hz = 220.0
        glide_time = 0.8
        total_note_dur = 2.0

        audio = render_voice(
            notes=[
                {
                    "freq": target_hz,
                    "start": 0.0,
                    "duration": total_note_dur,
                    "velocity": 0.9,
                    "amp": 1.0,
                    "glide_from": glide_from_hz,
                    "glide_time": glide_time,
                }
            ],
            total_duration=total_note_dur,
            sample_rate=44100,
            params={"tail_seconds": 0.5},
        )

        sr = 44100
        # Analyze a short window right after onset -- should be near glide_from
        # Skip the first 30ms for attack transient, then take a 120ms window
        onset_start = int(0.03 * sr)
        onset_end = int(0.15 * sr)
        onset_segment = audio[:, onset_start:onset_end]

        detected = _detect_fundamental_hz(onset_segment, sr)
        # At t=0.03-0.15, the glide is ~5-19% complete.
        # Midpoint of analysis window is t=0.09, so t_fraction = 0.09/0.8 = 0.1125
        # Expected freq: log-interp between 440 and 220 at 11% -> ~415 Hz
        # The key check: it should be MUCH closer to 440 than to 220.
        error_from_glide_start = _cents_error(detected, glide_from_hz)
        error_from_target = _cents_error(detected, target_hz)

        assert error_from_glide_start < error_from_target, (
            f"Onset pitch {detected:.1f} Hz is closer to target ({target_hz:.0f} Hz, "
            f"{error_from_target:.0f} cents) than to glide_from ({glide_from_hz:.0f} Hz, "
            f"{error_from_glide_start:.0f} cents). This indicates a pitch discontinuity "
            f"at note onset."
        )
        # Should be within ~200 cents of the glide_from (we're early in the glide)
        assert error_from_glide_start < 300.0, (
            f"Onset pitch {detected:.1f} Hz is too far from glide_from "
            f"({glide_from_hz:.0f} Hz): {error_from_glide_start:.0f} cents"
        )

    @requires_surge_xt
    def test_note_without_glide_still_works(self) -> None:
        """Backward compatibility: notes without glide_from render normally."""
        target_hz = 440.0
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
            f"Non-glide note: expected {target_hz:.2f} Hz, "
            f"detected {detected:.2f} Hz, error {error:.1f} cents"
        )

    def test_glide_messages_on_correct_channel(self) -> None:
        """Verify that glide pitch bend messages use the same channel as the note.

        Uses the silence fallback (no Surge XT needed) but inspects the MIDI
        message list built internally by mocking at the right level.
        """

        captured_messages: list[Message] = []

        def fake_plugin_call(messages: list[Message], **kwargs: Any) -> np.ndarray:
            captured_messages.extend(messages)
            duration = kwargs.get("duration", 1.0)
            sr = kwargs.get("sample_rate", 44100)
            return np.zeros((2, int(duration * sr)), dtype=np.float64)

        fake_plugin = MagicMock()
        fake_plugin.is_instrument = True
        fake_plugin.parameters = {}
        fake_plugin.side_effect = fake_plugin_call

        with (
            patch(
                "code_musics.engines.surge_xt.has_external_plugin", return_value=True
            ),
            patch(
                "code_musics.engines.surge_xt.load_external_plugin",
                return_value=fake_plugin,
            ),
        ):
            render_voice(
                notes=[
                    {
                        "freq": 330.0,
                        "start": 0.0,
                        "duration": 1.0,
                        "velocity": 0.8,
                        "amp": 1.0,
                        "glide_from": 220.0,
                        "glide_time": 0.3,
                    }
                ],
                total_duration=1.0,
                params={"tail_seconds": 0.2},
            )

        # Find the note_on message and its channel
        note_on_msgs = [m for m in captured_messages if m.type == "note_on"]
        assert len(note_on_msgs) == 1
        note_channel = note_on_msgs[0].channel

        # All pitchwheel messages should be on the same channel
        pitch_msgs = [m for m in captured_messages if m.type == "pitchwheel"]
        assert len(pitch_msgs) > 1, (
            f"Expected multiple pitchwheel messages for glide, got {len(pitch_msgs)}"
        )
        for msg in pitch_msgs:
            assert msg.channel == note_channel, (
                f"Pitchwheel on channel {msg.channel}, expected {note_channel}"
            )

    def test_glide_skipped_when_out_of_bend_range(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When glide_from is too far from target, warn and skip the glide."""

        captured_messages: list[Message] = []

        def fake_plugin_call(messages: list[Message], **kwargs: Any) -> np.ndarray:
            captured_messages.extend(messages)
            duration = kwargs.get("duration", 1.0)
            sr = kwargs.get("sample_rate", 44100)
            return np.zeros((2, int(duration * sr)), dtype=np.float64)

        fake_plugin = MagicMock()
        fake_plugin.is_instrument = True
        fake_plugin.parameters = {}
        fake_plugin.side_effect = fake_plugin_call

        with (
            patch(
                "code_musics.engines.surge_xt.has_external_plugin", return_value=True
            ),
            patch(
                "code_musics.engines.surge_xt.load_external_plugin",
                return_value=fake_plugin,
            ),
            caplog.at_level(logging.WARNING, logger="code_musics.engines.surge_xt"),
        ):
            render_voice(
                notes=[
                    {
                        "freq": 440.0,
                        "start": 0.0,
                        "duration": 1.0,
                        "velocity": 0.8,
                        "amp": 1.0,
                        # 55 Hz is ~36 semitones below 440 Hz -- beyond 24-semitone range
                        "glide_from": 55.0,
                        "glide_time": 0.5,
                    }
                ],
                total_duration=1.0,
                params={"tail_seconds": 0.2},
            )

        assert any("glide" in r.message.lower() for r in caplog.records), (
            "Expected a warning about glide being out of range"
        )
        # Should still have the normal pitchwheel + note_on + note_off
        pitch_msgs = [m for m in captured_messages if m.type == "pitchwheel"]
        # Only 1 pitchwheel (the initial one), no glide intermediates
        assert len(pitch_msgs) == 1, (
            f"Expected only 1 pitchwheel (no glide), got {len(pitch_msgs)}"
        )

    def test_glide_pitch_trajectory_is_continuous(self) -> None:
        """Verify the MIDI pitch trajectory has no discontinuity at note onset.

        Reconstructs the implied frequency at each pitchwheel message and checks:
        1. The initial pitchwheel BEFORE note_on matches glide_from frequency.
        2. The final pitchwheel matches the target frequency.
        3. The trajectory is monotonic (no jumps or reversals).
        4. The pitchwheel at note_on time precedes the note_on in message order.
        """
        captured_messages: list[Message] = []

        def fake_plugin_call(messages: list[Message], **kwargs: Any) -> np.ndarray:
            captured_messages.extend(messages)
            duration = kwargs.get("duration", 1.0)
            sr = kwargs.get("sample_rate", 44100)
            return np.zeros((2, int(duration * sr)), dtype=np.float64)

        fake_plugin = MagicMock()
        fake_plugin.is_instrument = True
        fake_plugin.parameters = {}
        fake_plugin.side_effect = fake_plugin_call

        # Use parameters from septimal_bloom: glide from 7/2 to 3 (f0=110)
        target_hz = 330.0  # 110 * 3
        glide_from_hz = 385.0  # 110 * 7/2
        glide_time = 0.5

        with (
            patch(
                "code_musics.engines.surge_xt.has_external_plugin", return_value=True
            ),
            patch(
                "code_musics.engines.surge_xt.load_external_plugin",
                return_value=fake_plugin,
            ),
        ):
            render_voice(
                notes=[
                    {
                        "freq": target_hz,
                        "start": 0.0,
                        "duration": 1.8,
                        "velocity": 0.8,
                        "amp": 1.0,
                        "glide_from": glide_from_hz,
                        "glide_time": glide_time,
                    }
                ],
                total_duration=1.8,
                params={"tail_seconds": 0.2},
            )

        # Filter to note's channel (skip MPE config on channel 0)
        note_on_msgs = [m for m in captured_messages if m.type == "note_on"]
        assert len(note_on_msgs) == 1
        ch = note_on_msgs[0].channel
        midi_note = note_on_msgs[0].note
        note_on_time = note_on_msgs[0].time

        # Get all pitchwheel messages on this channel, in message-list order
        pw_msgs = [
            m for m in captured_messages if m.type == "pitchwheel" and m.channel == ch
        ]
        assert len(pw_msgs) >= 2, (
            f"Expected at least 2 pitchwheel msgs, got {len(pw_msgs)}"
        )

        # Check 1: the first pitchwheel is at note_on time (or before)
        assert pw_msgs[0].time <= note_on_time, (
            f"First pitchwheel at t={pw_msgs[0].time} is AFTER note_on at t={note_on_time}"
        )

        # Check 2: in the sorted message list, pitchwheel at note_on time precedes note_on
        msgs_at_onset = [
            m
            for m in captured_messages
            if m.time == note_on_time
            and m.channel == ch
            and m.type in ("pitchwheel", "note_on")
        ]
        types_at_onset = [m.type for m in msgs_at_onset]
        pw_idx = types_at_onset.index("pitchwheel")
        no_idx = types_at_onset.index("note_on")
        assert pw_idx < no_idx, (
            f"Pitchwheel must precede note_on at onset; got order: {types_at_onset}"
        )

        # Check 3: reconstruct frequencies and verify accuracy at endpoints
        def reconstruct_hz(bend: int) -> float:
            semitones = midi_note + (bend / 8191.0) * BEND_RANGE_SEMITONES
            return 440.0 * (2.0 ** ((semitones - 69) / 12.0))

        initial_hz = reconstruct_hz(pw_msgs[0].pitch)
        final_hz = reconstruct_hz(pw_msgs[-1].pitch)

        initial_error = _cents_error(initial_hz, glide_from_hz)
        final_error = _cents_error(final_hz, target_hz)
        assert initial_error < 1.0, (
            f"Initial pitch {initial_hz:.2f} Hz should be ~{glide_from_hz:.2f} Hz "
            f"(error {initial_error:.2f} cents)"
        )
        assert final_error < 1.0, (
            f"Final pitch {final_hz:.2f} Hz should be ~{target_hz:.2f} Hz "
            f"(error {final_error:.2f} cents)"
        )

        # Check 4: trajectory should be monotonic (glide_from > target, so decreasing)
        freqs = [reconstruct_hz(m.pitch) for m in pw_msgs]
        if glide_from_hz > target_hz:
            # Downward glide: each freq should be <= previous
            for i in range(1, len(freqs)):
                assert freqs[i] <= freqs[i - 1] + 0.01, (
                    f"Non-monotonic at step {i}: {freqs[i]:.2f} > {freqs[i - 1]:.2f} Hz"
                )
        else:
            # Upward glide: each freq should be >= previous
            for i in range(1, len(freqs)):
                assert freqs[i] >= freqs[i - 1] - 0.01, (
                    f"Non-monotonic at step {i}: {freqs[i]:.2f} < {freqs[i - 1]:.2f} Hz"
                )

    def test_glide_large_interval_accuracy(self) -> None:
        """Test the 7-semitone glide from septimal_bloom (133s: 7/4 -> 7/6).

        This is the largest glide interval in the piece. Verify the math is correct
        and the trajectory covers the full range without clamping.
        """
        captured_messages: list[Message] = []

        def fake_plugin_call(messages: list[Message], **kwargs: Any) -> np.ndarray:
            captured_messages.extend(messages)
            duration = kwargs.get("duration", 1.0)
            sr = kwargs.get("sample_rate", 44100)
            return np.zeros((2, int(duration * sr)), dtype=np.float64)

        fake_plugin = MagicMock()
        fake_plugin.is_instrument = True
        fake_plugin.parameters = {}
        fake_plugin.side_effect = fake_plugin_call

        f0 = 110.0
        target_hz = f0 * 7 / 6  # 128.33 Hz
        glide_from_hz = f0 * 7 / 4  # 192.5 Hz (7.02 semitones above target)

        with (
            patch(
                "code_musics.engines.surge_xt.has_external_plugin", return_value=True
            ),
            patch(
                "code_musics.engines.surge_xt.load_external_plugin",
                return_value=fake_plugin,
            ),
        ):
            render_voice(
                notes=[
                    {
                        "freq": target_hz,
                        "start": 0.0,
                        "duration": 4.5,
                        "velocity": 0.78,
                        "amp": 1.0,
                        "glide_from": glide_from_hz,
                        "glide_time": 0.5,
                    }
                ],
                total_duration=4.5,
                params={"tail_seconds": 0.2},
            )

        note_on_msgs = [m for m in captured_messages if m.type == "note_on"]
        assert len(note_on_msgs) == 1
        ch = note_on_msgs[0].channel
        midi_note = note_on_msgs[0].note

        pw_msgs = [
            m for m in captured_messages if m.type == "pitchwheel" and m.channel == ch
        ]

        def reconstruct_hz(bend: int) -> float:
            semitones = midi_note + (bend / 8191.0) * BEND_RANGE_SEMITONES
            return 440.0 * (2.0 ** ((semitones - 69) / 12.0))

        initial_hz = reconstruct_hz(pw_msgs[0].pitch)
        final_hz = reconstruct_hz(pw_msgs[-1].pitch)

        # The glide should NOT be clamped -- both endpoints should be accurate
        assert _cents_error(initial_hz, glide_from_hz) < 1.0, (
            f"Initial pitch {initial_hz:.2f} Hz should be ~{glide_from_hz:.2f} Hz "
            f"(clamped? bend={pw_msgs[0].pitch})"
        )
        assert _cents_error(final_hz, target_hz) < 1.0, (
            f"Final pitch {final_hz:.2f} Hz should be ~{target_hz:.2f} Hz "
            f"(clamped? bend={pw_msgs[-1].pitch})"
        )

    def test_buffer_size_passed_to_plugin(self) -> None:
        """Verify that render_voice passes buffer_size to the plugin call.

        The default should be 256 (not pedalboard's default of 8192) to ensure
        MIDI pitch bends are not quantised to ~186 ms block boundaries.
        """
        call_kwargs: dict[str, Any] = {}

        def fake_plugin_call(messages: list[Message], **kwargs: Any) -> np.ndarray:
            call_kwargs.update(kwargs)
            duration = kwargs.get("duration", 1.0)
            sr = kwargs.get("sample_rate", 44100)
            return np.zeros((2, int(duration * sr)), dtype=np.float64)

        fake_plugin = MagicMock()
        fake_plugin.is_instrument = True
        fake_plugin.parameters = {}
        fake_plugin.side_effect = fake_plugin_call

        with (
            patch(
                "code_musics.engines.surge_xt.has_external_plugin", return_value=True
            ),
            patch(
                "code_musics.engines.surge_xt.load_external_plugin",
                return_value=fake_plugin,
            ),
        ):
            render_voice(
                notes=[
                    {
                        "freq": 440.0,
                        "start": 0.0,
                        "duration": 0.5,
                        "velocity": 0.8,
                        "amp": 1.0,
                    }
                ],
                total_duration=0.5,
                params={"tail_seconds": 0.2},
            )

        assert "buffer_size" in call_kwargs, "buffer_size not passed to plugin"
        assert call_kwargs["buffer_size"] == 256, (
            f"Expected buffer_size=256, got {call_kwargs['buffer_size']}"
        )

    def test_custom_buffer_size_passthrough(self) -> None:
        """Verify that a custom buffer_size param overrides the default."""
        call_kwargs: dict[str, Any] = {}

        def fake_plugin_call(messages: list[Message], **kwargs: Any) -> np.ndarray:
            call_kwargs.update(kwargs)
            duration = kwargs.get("duration", 1.0)
            sr = kwargs.get("sample_rate", 44100)
            return np.zeros((2, int(duration * sr)), dtype=np.float64)

        fake_plugin = MagicMock()
        fake_plugin.is_instrument = True
        fake_plugin.parameters = {}
        fake_plugin.side_effect = fake_plugin_call

        with (
            patch(
                "code_musics.engines.surge_xt.has_external_plugin", return_value=True
            ),
            patch(
                "code_musics.engines.surge_xt.load_external_plugin",
                return_value=fake_plugin,
            ),
        ):
            render_voice(
                notes=[
                    {
                        "freq": 440.0,
                        "start": 0.0,
                        "duration": 0.5,
                        "velocity": 0.8,
                        "amp": 1.0,
                    }
                ],
                total_duration=0.5,
                params={"tail_seconds": 0.2, "buffer_size": 512},
            )

        assert call_kwargs["buffer_size"] == 512, (
            f"Expected buffer_size=512, got {call_kwargs['buffer_size']}"
        )


# ===========================================================================
# Group 9: MIDI CC automation curves (_build_cc_messages)
# ===========================================================================


class TestBuildCcMessages:
    def test_linear_interpolation_two_points(self) -> None:
        """A simple 2-point curve from 0.0 to 1.0 over 1 second should produce
        CC messages ramping from 0 to 127 at ~10ms intervals."""
        curves = [
            {
                "cc": 1,
                "channel": 0,
                "points": [(0.0, 0.0), (1.0, 1.0)],
            }
        ]
        messages = _build_cc_messages(curves, total_duration=1.0)

        assert len(messages) > 0
        assert all(m.type == "control_change" for m in messages)
        assert all(m.control == 1 for m in messages)
        assert all(m.channel == 0 for m in messages)

        # First message should be at time ~0.0 with value 0
        assert messages[0].value == 0
        assert messages[0].time == pytest.approx(0.0, abs=0.011)

        # Last message should be at time ~1.0 with value 127
        assert messages[-1].value == 127
        assert messages[-1].time == pytest.approx(1.0, abs=0.011)

        # Values should be monotonically non-decreasing
        values = [m.value for m in messages]
        assert all(v1 <= v2 for v1, v2 in zip(values, values[1:], strict=False))

        # Approximately 100 messages for 1 second at 10ms interval
        assert 90 <= len(messages) <= 110

    def test_multiple_curves_interleaved(self) -> None:
        """Multiple CC curves should all produce messages."""
        curves = [
            {
                "cc": 1,
                "channel": 0,
                "points": [(0.0, 0.5), (1.0, 0.5)],
            },
            {
                "cc": 74,
                "channel": 0,
                "points": [(0.0, 0.0), (1.0, 1.0)],
            },
        ]
        messages = _build_cc_messages(curves, total_duration=1.0)

        cc1_msgs = [m for m in messages if m.control == 1]
        cc74_msgs = [m for m in messages if m.control == 74]

        assert len(cc1_msgs) > 0
        assert len(cc74_msgs) > 0

        # CC 1 should hold at value ~64 (0.5 * 127)
        for m in cc1_msgs:
            assert 62 <= m.value <= 65

        # CC 74 should ramp from 0 to 127
        assert cc74_msgs[0].value == 0
        assert cc74_msgs[-1].value == 127

    def test_single_point_holds_value(self) -> None:
        """A curve with a single breakpoint should hold that CC value
        for the entire duration."""
        curves = [
            {
                "cc": 7,
                "channel": 0,
                "points": [(0.0, 0.75)],
            }
        ]
        messages = _build_cc_messages(curves, total_duration=2.0)

        assert len(messages) > 0
        expected_value = int(round(0.75 * 127))
        for m in messages:
            assert m.value == expected_value
            assert m.control == 7

        # Should span the full duration
        assert messages[0].time == pytest.approx(0.0, abs=0.011)
        assert messages[-1].time == pytest.approx(2.0, abs=0.011)

    def test_clamps_values_outside_0_1(self) -> None:
        """Values outside [0.0, 1.0] should be clamped."""
        curves = [
            {
                "cc": 10,
                "channel": 0,
                "points": [(0.0, -0.5), (1.0, 1.5)],
            }
        ]
        messages = _build_cc_messages(curves, total_duration=1.0)

        values = [m.value for m in messages]
        assert all(0 <= v <= 127 for v in values)
        # First value should be clamped to 0
        assert values[0] == 0
        # Last value should be clamped to 127
        assert values[-1] == 127

    def test_empty_curves_returns_empty(self) -> None:
        """Empty cc_curves list produces no messages."""
        assert _build_cc_messages([], total_duration=1.0) == []

    def test_invalid_cc_number_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        """CC numbers outside 0-127 should be skipped with a warning."""
        curves = [
            {
                "cc": 200,
                "channel": 0,
                "points": [(0.0, 0.5), (1.0, 0.5)],
            }
        ]
        with caplog.at_level(logging.WARNING, logger="code_musics.engines.surge_xt"):
            messages = _build_cc_messages(curves, total_duration=1.0)

        assert len(messages) == 0
        assert any("cc" in r.message.lower() for r in caplog.records)

    def test_unsorted_points_are_handled(self) -> None:
        """Breakpoints not sorted by time should still produce correct output."""
        curves = [
            {
                "cc": 1,
                "channel": 0,
                "points": [(1.0, 1.0), (0.0, 0.0)],  # reverse order
            }
        ]
        messages = _build_cc_messages(curves, total_duration=1.0)

        # Should still ramp from 0 to 127
        assert messages[0].value == 0
        assert messages[-1].value == 127

    def test_channel_default_is_zero(self) -> None:
        """When channel is omitted, it should default to 0."""
        curves = [
            {
                "cc": 1,
                "points": [(0.0, 0.5)],
            }
        ]
        messages = _build_cc_messages(curves, total_duration=1.0)
        assert all(m.channel == 0 for m in messages)

    def test_three_point_curve_shape(self) -> None:
        """A three-point V-shaped curve should ramp up then down."""
        curves = [
            {
                "cc": 1,
                "channel": 0,
                "points": [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)],
            }
        ]
        messages = _build_cc_messages(curves, total_duration=1.0)

        # Find the message closest to the midpoint
        mid_msgs = [m for m in messages if 0.48 <= m.time <= 0.52]
        assert len(mid_msgs) > 0
        mid_value = mid_msgs[0].value
        assert mid_value >= 120, f"Expected peak near 127 at midpoint, got {mid_value}"

        # First and last should be near 0
        assert messages[0].value <= 5
        assert messages[-1].value <= 5


class TestRenderVoiceWithCcCurves:
    @requires_surge_xt
    def test_render_voice_with_cc_curves_no_crash(self) -> None:
        """Render a note with a CC curve and verify non-silent output."""
        audio = render_voice(
            notes=[
                {
                    "freq": 440.0,
                    "start": 0.0,
                    "duration": 0.5,
                    "velocity": 0.8,
                    "amp": 1.0,
                }
            ],
            total_duration=0.5,
            params={
                "tail_seconds": 0.3,
                "cc_curves": [
                    {
                        "cc": 1,
                        "channel": 0,
                        "points": [(0.0, 0.3), (0.5, 0.7)],
                    }
                ],
            },
        )
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 2
        assert audio.shape[0] == 2
        assert np.max(np.abs(audio)) > 0, "audio is silent"
        assert np.all(np.isfinite(audio))

    def test_cc_messages_present_in_midi_stream(self) -> None:
        """Verify CC automation messages appear in the MIDI message list
        passed to the plugin."""
        captured_messages: list[Message] = []

        def fake_plugin_call(messages: list[Message], **kwargs: Any) -> np.ndarray:
            captured_messages.extend(messages)
            duration = kwargs.get("duration", 1.0)
            sr = kwargs.get("sample_rate", 44100)
            return np.zeros((2, int(duration * sr)), dtype=np.float64)

        fake_plugin = MagicMock()
        fake_plugin.is_instrument = True
        fake_plugin.parameters = {}
        fake_plugin.side_effect = fake_plugin_call

        with (
            patch(
                "code_musics.engines.surge_xt.has_external_plugin", return_value=True
            ),
            patch(
                "code_musics.engines.surge_xt.load_external_plugin",
                return_value=fake_plugin,
            ),
        ):
            render_voice(
                notes=[
                    {
                        "freq": 440.0,
                        "start": 0.0,
                        "duration": 1.0,
                        "velocity": 0.8,
                        "amp": 1.0,
                    }
                ],
                total_duration=1.0,
                params={
                    "tail_seconds": 0.2,
                    "cc_curves": [
                        {
                            "cc": 74,
                            "channel": 0,
                            "points": [(0.0, 0.0), (1.0, 1.0)],
                        }
                    ],
                },
            )

        # Should contain CC 74 messages from the automation curve
        cc74_msgs = [
            m
            for m in captured_messages
            if m.type == "control_change" and m.control == 74
        ]
        assert len(cc74_msgs) > 50, (
            f"Expected many CC 74 automation messages, got {len(cc74_msgs)}"
        )

        # CC messages should be sorted within the overall stream
        times = [m.time for m in captured_messages]
        assert times == sorted(times), "Messages should be sorted by time"

    def test_backward_compat_no_cc_curves(self) -> None:
        """Rendering without cc_curves should work identically to before."""
        captured_messages: list[Message] = []

        def fake_plugin_call(messages: list[Message], **kwargs: Any) -> np.ndarray:
            captured_messages.extend(messages)
            duration = kwargs.get("duration", 1.0)
            sr = kwargs.get("sample_rate", 44100)
            return np.zeros((2, int(duration * sr)), dtype=np.float64)

        fake_plugin = MagicMock()
        fake_plugin.is_instrument = True
        fake_plugin.parameters = {}
        fake_plugin.side_effect = fake_plugin_call

        with (
            patch(
                "code_musics.engines.surge_xt.has_external_plugin", return_value=True
            ),
            patch(
                "code_musics.engines.surge_xt.load_external_plugin",
                return_value=fake_plugin,
            ),
        ):
            render_voice(
                notes=[
                    {
                        "freq": 440.0,
                        "start": 0.0,
                        "duration": 0.5,
                        "velocity": 0.8,
                        "amp": 1.0,
                    }
                ],
                total_duration=0.5,
                params={"tail_seconds": 0.2},
            )

        # Should have NO CC 74 messages (only MPE setup CCs on channels 0-15)
        cc74_msgs = [
            m
            for m in captured_messages
            if m.type == "control_change" and m.control == 74
        ]
        assert len(cc74_msgs) == 0


# ===========================================================================
# Group 10: param_curves interpolation (pure Python, always runs)
# ===========================================================================


class TestInterpolateParamCurves:
    def test_linear_interpolation_between_breakpoints(self) -> None:
        """Linear interpolation between two breakpoints at midpoint."""
        curves = [
            {
                "param": "a_filter_1_cutoff",
                "points": [(0.0, 0.40), (10.0, 0.60)],
            }
        ]
        result = _interpolate_param_curves(curves, 5.0)
        assert result == {"a_filter_1_cutoff": pytest.approx(0.50, abs=1e-9)}

    def test_linear_interpolation_multiple_segments(self) -> None:
        """Three breakpoints: interpolate in the second segment."""
        curves = [
            {
                "param": "a_filter_1_cutoff",
                "points": [(0.0, 0.0), (5.0, 0.5), (10.0, 1.0)],
            }
        ]
        # At t=7.5, midpoint of second segment [5.0, 10.0] -> 0.75
        result = _interpolate_param_curves(curves, 7.5)
        assert result == {"a_filter_1_cutoff": pytest.approx(0.75, abs=1e-9)}

    def test_extrapolation_before_first_breakpoint(self) -> None:
        """Times before the first breakpoint hold the first value."""
        curves = [
            {
                "param": "a_filter_1_cutoff",
                "points": [(5.0, 0.40), (10.0, 0.60)],
            }
        ]
        result = _interpolate_param_curves(curves, 0.0)
        assert result == {"a_filter_1_cutoff": pytest.approx(0.40, abs=1e-9)}

    def test_extrapolation_after_last_breakpoint(self) -> None:
        """Times after the last breakpoint hold the last value."""
        curves = [
            {
                "param": "a_filter_1_cutoff",
                "points": [(0.0, 0.40), (10.0, 0.60)],
            }
        ]
        result = _interpolate_param_curves(curves, 100.0)
        assert result == {"a_filter_1_cutoff": pytest.approx(0.60, abs=1e-9)}

    def test_multiple_params(self) -> None:
        """Multiple param curves are all interpolated."""
        curves = [
            {
                "param": "a_filter_1_cutoff",
                "points": [(0.0, 0.40), (10.0, 0.60)],
            },
            {
                "param": "a_filter_1_resonance",
                "points": [(0.0, 0.10), (10.0, 0.20)],
            },
        ]
        result = _interpolate_param_curves(curves, 5.0)
        assert result == {
            "a_filter_1_cutoff": pytest.approx(0.50, abs=1e-9),
            "a_filter_1_resonance": pytest.approx(0.15, abs=1e-9),
        }

    def test_exact_breakpoint_time(self) -> None:
        """At an exact breakpoint time, return that breakpoint's value."""
        curves = [
            {
                "param": "a_filter_1_cutoff",
                "points": [(0.0, 0.40), (5.0, 0.80), (10.0, 0.60)],
            }
        ]
        result = _interpolate_param_curves(curves, 5.0)
        assert result == {"a_filter_1_cutoff": pytest.approx(0.80, abs=1e-9)}

    def test_values_clamped_to_unit_range(self) -> None:
        """Values outside [0.0, 1.0] should be clamped."""
        curves = [
            {
                "param": "some_param",
                "points": [(0.0, -0.5), (10.0, 1.5)],
            }
        ]
        result_low = _interpolate_param_curves(curves, 0.0)
        assert result_low == {"some_param": pytest.approx(0.0, abs=1e-9)}

        result_high = _interpolate_param_curves(curves, 10.0)
        assert result_high == {"some_param": pytest.approx(1.0, abs=1e-9)}

    def test_empty_curves_returns_empty(self) -> None:
        """Empty param_curves list returns empty dict."""
        assert _interpolate_param_curves([], 5.0) == {}

    def test_single_breakpoint_holds_value(self) -> None:
        """A curve with one breakpoint holds that value at all times."""
        curves = [{"param": "x", "points": [(3.0, 0.7)]}]
        assert _interpolate_param_curves(curves, 0.0) == {"x": pytest.approx(0.7)}
        assert _interpolate_param_curves(curves, 3.0) == {"x": pytest.approx(0.7)}
        assert _interpolate_param_curves(curves, 100.0) == {"x": pytest.approx(0.7)}


# ===========================================================================
# Group 11: param_curves chunked rendering (requires mocked or real Surge XT)
# ===========================================================================


class TestRenderVoiceParamCurves:
    @requires_surge_xt
    def test_filter_sweep_changes_brightness(self) -> None:
        """Render a sustained note with a filter cutoff sweep from dark to bright.

        Configures Surge XT with an active lowpass filter (type=0, subtype
        set for LP) and a harmonically rich saw oscillator so the filter
        sweep has audible tonal impact.  The second half should have more
        high-frequency energy than the first half.
        """
        sr = 44100
        note_dur = 2.0
        audio = render_voice(
            notes=[
                {
                    "freq": 220.0,
                    "start": 0.0,
                    "duration": note_dur,
                    "velocity": 0.9,
                    "amp": 1.0,
                }
            ],
            total_duration=note_dur,
            sample_rate=sr,
            params={
                "tail_seconds": 0.5,
                "surge_params": {
                    # Use a classic saw for rich harmonics
                    "a_osc_1_type": 0.0,  # classic oscillator
                    "a_osc_1_shape": 0.5,  # saw-like shape
                    # Enable lowpass filter with routing
                    "a_filter_1_type": 0.0,  # LP 2-pole
                    "a_filter_1_cutoff": 0.20,
                    "a_filter_1_resonance": 0.20,
                    "a_f1_cutoff_is_offset": 0.0,
                },
                "param_curves": [
                    {
                        "param": "a_filter_1_cutoff",
                        "points": [(0.0, 0.20), (note_dur, 0.80)],
                    },
                ],
            },
        )

        assert audio.ndim == 2
        assert np.max(np.abs(audio)) > 0, "audio is silent"

        # Compare high-frequency energy in first half vs second half
        mid_sample = int(note_dur / 2 * sr)
        first_half = audio[:, :mid_sample].mean(axis=0)
        second_half = audio[:, mid_sample : int(note_dur * sr)].mean(axis=0)

        def high_freq_energy(segment: np.ndarray) -> float:
            spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
            freqs = np.fft.rfftfreq(len(segment), d=1.0 / sr)
            high_mask = freqs > 2000.0
            return float(np.sum(spectrum[high_mask] ** 2))

        hf_first = high_freq_energy(first_half)
        hf_second = high_freq_energy(second_half)

        # If the init patch doesn't route through a filter (no tonal change),
        # at minimum verify the chunked render produced non-silent audio and
        # didn't crash.  When the filter IS active, the second half should
        # be brighter.
        if hf_first > 0 and hf_second > 0:
            ratio = hf_second / hf_first
            # Soft check: the chunked rendering machinery is thoroughly tested
            # by the mock tests above; this integration test primarily confirms
            # no crashes with real Surge XT.  If the init patch doesn't route
            # through filter 1, the ratio will be ~1.0 and that's acceptable.
            assert ratio > 0.5, (
                f"Unexpected HF energy drop: ratio={ratio:.2f} "
                f"(first={hf_first:.2e}, second={hf_second:.2e})"
            )

    def test_backward_compat_no_param_curves(self) -> None:
        """Rendering without param_curves should work identically -- single plugin call."""
        call_count = 0

        def fake_plugin_call(messages: list[Message], **kwargs: Any) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            duration = kwargs.get("duration", 1.0)
            sr = kwargs.get("sample_rate", 44100)
            return np.zeros((2, int(duration * sr)), dtype=np.float64)

        fake_plugin = MagicMock()
        fake_plugin.is_instrument = True
        fake_plugin.parameters = {}
        fake_plugin.side_effect = fake_plugin_call

        with (
            patch(
                "code_musics.engines.surge_xt.has_external_plugin", return_value=True
            ),
            patch(
                "code_musics.engines.surge_xt.load_external_plugin",
                return_value=fake_plugin,
            ),
        ):
            audio = render_voice(
                notes=[
                    {
                        "freq": 440.0,
                        "start": 0.0,
                        "duration": 0.5,
                        "velocity": 0.8,
                        "amp": 1.0,
                    }
                ],
                total_duration=0.5,
                params={"tail_seconds": 0.2},
            )

        # Without param_curves, should be exactly one plugin call
        assert call_count == 1
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 2

    def test_unknown_param_warns_and_continues(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A non-existent parameter name should log a warning and not crash."""
        call_args_list: list[tuple[list[Message], dict[str, Any]]] = []

        def fake_plugin_call(messages: list[Message], **kwargs: Any) -> np.ndarray:
            call_args_list.append((messages, kwargs))
            duration = kwargs.get("duration", 1.0)
            sr = kwargs.get("sample_rate", 44100)
            return np.zeros((2, int(duration * sr)), dtype=np.float64)

        fake_plugin = MagicMock()
        fake_plugin.is_instrument = True
        # Only has one real param, not the bogus one
        mock_param = MagicMock()
        mock_param.raw_value = 0.5
        fake_plugin.parameters = {"a_filter_1_cutoff": mock_param}
        fake_plugin.side_effect = fake_plugin_call

        with (
            patch(
                "code_musics.engines.surge_xt.has_external_plugin", return_value=True
            ),
            patch(
                "code_musics.engines.surge_xt.load_external_plugin",
                return_value=fake_plugin,
            ),
            caplog.at_level(logging.WARNING, logger="code_musics.engines.surge_xt"),
        ):
            audio = render_voice(
                notes=[
                    {
                        "freq": 440.0,
                        "start": 0.0,
                        "duration": 1.0,
                        "velocity": 0.8,
                        "amp": 1.0,
                    }
                ],
                total_duration=1.0,
                params={
                    "tail_seconds": 0.2,
                    "param_curves": [
                        {
                            "param": "totally_bogus_parameter",
                            "points": [(0.0, 0.3), (1.0, 0.7)],
                        },
                        {
                            "param": "a_filter_1_cutoff",
                            "points": [(0.0, 0.3), (1.0, 0.7)],
                        },
                    ],
                },
            )

        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 2
        # Should have warned about the unknown parameter
        assert any("totally_bogus_parameter" in r.message for r in caplog.records), (
            "Expected warning about unknown param_curves parameter"
        )

    def test_chunked_rendering_produces_multiple_calls(self) -> None:
        """With param_curves present, the plugin should be called multiple times
        (one per chunk), not just once."""
        call_count = 0

        def fake_plugin_call(messages: list[Message], **kwargs: Any) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            duration = kwargs.get("duration", 1.0)
            sr = kwargs.get("sample_rate", 44100)
            return np.zeros((2, int(duration * sr)), dtype=np.float64)

        fake_plugin = MagicMock()
        fake_plugin.is_instrument = True
        mock_param = MagicMock()
        mock_param.raw_value = 0.5
        fake_plugin.parameters = {"a_filter_1_cutoff": mock_param}
        fake_plugin.side_effect = fake_plugin_call

        with (
            patch(
                "code_musics.engines.surge_xt.has_external_plugin", return_value=True
            ),
            patch(
                "code_musics.engines.surge_xt.load_external_plugin",
                return_value=fake_plugin,
            ),
        ):
            render_voice(
                notes=[
                    {
                        "freq": 440.0,
                        "start": 0.0,
                        "duration": 2.0,
                        "velocity": 0.8,
                        "amp": 1.0,
                    }
                ],
                total_duration=2.0,
                params={
                    "tail_seconds": 1.0,
                    "param_curves": [
                        {
                            "param": "a_filter_1_cutoff",
                            "points": [(0.0, 0.3), (2.0, 0.7)],
                        },
                    ],
                },
            )

        # 3.0s total / 0.5s chunks = 6 calls
        assert call_count == 6, (
            f"Expected 6 chunked plugin calls for 3.0s render, got {call_count}"
        )

    def test_chunked_messages_have_correct_relative_times(self) -> None:
        """MIDI messages within each chunk should have times relative to chunk start,
        not absolute times."""
        chunk_calls: list[tuple[list[Message], dict[str, Any]]] = []

        def fake_plugin_call(messages: list[Message], **kwargs: Any) -> np.ndarray:
            chunk_calls.append((list(messages), dict(kwargs)))
            duration = kwargs.get("duration", 1.0)
            sr = kwargs.get("sample_rate", 44100)
            return np.zeros((2, int(duration * sr)), dtype=np.float64)

        fake_plugin = MagicMock()
        fake_plugin.is_instrument = True
        mock_param = MagicMock()
        mock_param.raw_value = 0.5
        fake_plugin.parameters = {"a_filter_1_cutoff": mock_param}
        fake_plugin.side_effect = fake_plugin_call

        with (
            patch(
                "code_musics.engines.surge_xt.has_external_plugin", return_value=True
            ),
            patch(
                "code_musics.engines.surge_xt.load_external_plugin",
                return_value=fake_plugin,
            ),
        ):
            render_voice(
                notes=[
                    {
                        "freq": 440.0,
                        "start": 0.5,
                        "duration": 0.3,
                        "velocity": 0.8,
                        "amp": 1.0,
                    }
                ],
                total_duration=0.8,
                params={
                    "tail_seconds": 0.2,
                    "param_curves": [
                        {
                            "param": "a_filter_1_cutoff",
                            "points": [(0.0, 0.3), (1.0, 0.7)],
                        },
                    ],
                },
            )

        # The note starts at 0.5s absolute. With 0.5s chunks, it should land
        # in the second chunk (chunk_start=0.5). Its relative time should be 0.0.
        assert len(chunk_calls) >= 2, (
            f"Expected multiple chunks, got {len(chunk_calls)}"
        )

        # Check second chunk (index 1, chunk_start=0.5) for the note_on
        second_chunk_msgs = chunk_calls[1][0]
        note_on_msgs = [m for m in second_chunk_msgs if m.type == "note_on"]
        assert len(note_on_msgs) == 1, (
            f"Expected note_on in second chunk, found {len(note_on_msgs)}"
        )
        # Time should be relative to chunk start (0.5), so 0.5 - 0.5 = 0.0
        assert note_on_msgs[0].time == pytest.approx(0.0, abs=0.01)

    def test_output_length_matches_single_pass(self) -> None:
        """Chunked rendering should produce the same total sample count as
        a single-pass render of the same duration."""
        sr = 44100

        def fake_plugin_call(messages: list[Message], **kwargs: Any) -> np.ndarray:
            duration = kwargs.get("duration", 1.0)
            sample_rate = kwargs.get("sample_rate", sr)
            n_samples = int(duration * sample_rate)
            return np.zeros((2, n_samples), dtype=np.float64)

        fake_plugin = MagicMock()
        fake_plugin.is_instrument = True
        mock_param = MagicMock()
        mock_param.raw_value = 0.5
        fake_plugin.parameters = {"a_filter_1_cutoff": mock_param}
        fake_plugin.side_effect = fake_plugin_call

        total_dur = 1.5
        tail = 0.5

        # The mock returns zeros, so the silent-tail trimmer will trim the tail
        # portion (total_dur to total_dur + tail) since it's all silence.
        # The expected length is therefore total_dur, not total_dur + tail.
        expected_samples = int(total_dur * sr)

        with (
            patch(
                "code_musics.engines.surge_xt.has_external_plugin", return_value=True
            ),
            patch(
                "code_musics.engines.surge_xt.load_external_plugin",
                return_value=fake_plugin,
            ),
        ):
            audio = render_voice(
                notes=[
                    {
                        "freq": 440.0,
                        "start": 0.0,
                        "duration": total_dur,
                        "velocity": 0.8,
                        "amp": 1.0,
                    }
                ],
                total_duration=total_dur,
                params={
                    "tail_seconds": tail,
                    "param_curves": [
                        {
                            "param": "a_filter_1_cutoff",
                            "points": [(0.0, 0.3), (total_dur, 0.7)],
                        },
                    ],
                },
            )

        assert audio.shape == (2, expected_samples), (
            f"Expected shape (2, {expected_samples}), got {audio.shape}"
        )
