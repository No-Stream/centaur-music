"""Tests for perceptual effect-chain instrumentation.

Covers the five new measurement surfaces added on top of the existing
per-effect IO deltas:

1. A-weighted high-band brightness delta.
2. Two-tone intermodulation distortion ratio.
3. Percussive onset detection + per-hit transient metrics.
4. Voice-stem dry->wet delta analysis (full voice-chain).
5. New warning codes (percussive_transient_killed,
   percussive_crunch_character, transient_brightness_decoupled,
   perceptual_brightness_lift).
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics import synth
from code_musics.drum_helpers import add_drum_voice
from code_musics.engines._instrumentation import (
    a_weight_db,
    a_weighted_mean_band_energy_db,
    detect_percussive_onsets,
    intermodulation_ratio,
    per_hit_transient_metrics,
)
from code_musics.score import EffectSpec, NoteEvent, Score, Voice

SAMPLE_RATE = 44_100


# ---------------------------------------------------------------------------
# A-weighting
# ---------------------------------------------------------------------------


class TestAWeighting:
    def test_a_weight_unity_near_1khz(self) -> None:
        # A-weighting crosses 0 dB at 1 kHz by construction.
        gain_db = a_weight_db(np.asarray([1_000.0]))
        assert abs(float(gain_db[0])) < 0.5

    def test_a_weight_low_freq_suppressed(self) -> None:
        # At 100 Hz A-weighting drops to roughly -19 dB.
        gain_db = a_weight_db(np.asarray([100.0]))
        assert -21.0 < float(gain_db[0]) < -17.0

    def test_a_weight_4khz_slight_boost(self) -> None:
        # At 4 kHz A-weighting is +1 dB.
        gain_db = a_weight_db(np.asarray([4_000.0]))
        assert 0.5 < float(gain_db[0]) < 1.5

    def test_a_weighted_spectrum_band_suppresses_bass(self) -> None:
        freqs = np.asarray([50.0, 100.0, 4_000.0, 8_000.0])
        magnitude_db = np.zeros_like(freqs)
        unweighted = float(np.mean(magnitude_db))
        weighted = a_weighted_mean_band_energy_db(
            freqs, magnitude_db, low_hz=50.0, high_hz=200.0
        )
        weighted_mid = a_weighted_mean_band_energy_db(
            freqs, magnitude_db, low_hz=2_000.0, high_hz=5_000.0
        )
        # Low band is suppressed, mid band is near unity.
        assert weighted < unweighted - 10.0
        assert abs(weighted_mid - unweighted) < 2.0


# ---------------------------------------------------------------------------
# Intermodulation distortion
# ---------------------------------------------------------------------------


def _sine(freq_hz: float, duration: float = 0.5, amp: float = 0.5) -> np.ndarray:
    n = int(duration * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    return amp * np.sin(2.0 * np.pi * freq_hz * t)


def _spectrum(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Reuse synth's averaged spectrum — matches instrumentation path.
    return synth._average_spectrum_db(signal, sample_rate=SAMPLE_RATE)


class TestIMD:
    def test_pure_sine_reports_single_tone(self) -> None:
        signal = _sine(440.0)
        freqs, magnitude_db = _spectrum(signal)
        result = intermodulation_ratio(freqs, magnitude_db)
        assert result.detection == "single_tone"
        assert result.ratio == 0.0

    def test_two_tone_clean_low_imd(self) -> None:
        signal = _sine(440.0) + _sine(660.0)
        signal = signal / float(np.max(np.abs(signal)) + 1e-9) * 0.4
        freqs, magnitude_db = _spectrum(signal)
        result = intermodulation_ratio(freqs, magnitude_db)
        # Clean sum of two sines: no IMD, but also little harmonic energy.
        # Ratio can be small-but-nonzero due to FFT leakage. Gate loosely.
        assert result.detection == "two_tone"
        assert result.ratio < 2.0

    def test_two_tone_soft_clip_produces_imd(self) -> None:
        signal = _sine(440.0) + _sine(660.0)
        # Normalize so the clipper engages.
        signal = signal / float(np.max(np.abs(signal)) + 1e-9) * 0.95
        driven = np.tanh(2.5 * signal)
        freqs, magnitude_db = _spectrum(driven)
        result = intermodulation_ratio(freqs, magnitude_db)
        assert result.detection == "two_tone"
        assert result.ratio > 0.3


# ---------------------------------------------------------------------------
# Percussive onsets + per-hit transient metrics
# ---------------------------------------------------------------------------


def _synthetic_click_heavy_kick(
    n_hits: int = 4, tempo_bpm: float = 120.0, duration: float = 2.0
) -> np.ndarray:
    """Kick train where the 5 ms click dominates the 50 ms body peak.

    Used to verify transient-kill detection: the dry signal has a
    ``transient_peak_ratio`` near 1.0, so any effect that attenuates the
    first ~5 ms produces a visible negative ``transient_kill_db``.
    """
    n = int(duration * SAMPLE_RATE)
    signal = np.zeros(n, dtype=np.float64)
    beat_seconds = 60.0 / tempo_bpm
    for hit_idx in range(n_hits):
        onset_time = hit_idx * beat_seconds
        onset_sample = int(onset_time * SAMPLE_RATE)
        if onset_sample >= n:
            break
        length = min(int(0.050 * SAMPLE_RATE), n - onset_sample)
        t = np.arange(length) / SAMPLE_RATE
        # Click: 3 ms broadband burst with high amplitude. Dominates the
        # first 5 ms comfortably.
        click_len = min(int(0.003 * SAMPLE_RATE), length)
        click_env = np.exp(-np.arange(click_len) / 20.0)
        click = 0.9 * np.sin(2.0 * np.pi * 4_000.0 * t[:click_len]) * click_env
        # Body: low-amplitude 60 Hz swell peaking at 20 ms.
        body_env = np.exp(-((t - 0.020) ** 2) / (2 * 0.010**2))
        body = 0.25 * np.sin(2.0 * np.pi * 60.0 * t) * body_env
        hit = body
        hit[:click_len] += click
        signal[onset_sample : onset_sample + length] += hit
    return signal


def _synthetic_kick_train(
    n_hits: int = 4, tempo_bpm: float = 120.0, duration: float = 2.0
) -> np.ndarray:
    """Build a crude 4/4 kick-like signal with body peak AFTER click.

    A realistic kick has a snappy click (HF transient) riding a slower
    body whose amplitude peaks a few ms into the hit. In this signal the
    5 ms transient window captures mostly click energy; the 50 ms window
    captures the body peak. The ratio is strictly below 1.0 so a fast-
    attack compressor or lowpass filter (which attenuates the click but
    preserves the body) produces a measurable transient_kill.
    """
    n = int(duration * SAMPLE_RATE)
    signal = np.zeros(n, dtype=np.float64)
    beat_seconds = 60.0 / tempo_bpm
    for hit_idx in range(n_hits):
        onset_time = hit_idx * beat_seconds
        onset_sample = int(onset_time * SAMPLE_RATE)
        if onset_sample >= n:
            break
        length = min(int(0.050 * SAMPLE_RATE), n - onset_sample)
        t = np.arange(length) / SAMPLE_RATE
        # Kick body: 60 Hz sine, amplitude envelope that peaks ~15 ms in
        # then decays out by 50 ms.
        body_env = np.exp(-((t - 0.015) ** 2) / (2 * 0.010**2))
        body = 0.6 * np.sin(2.0 * np.pi * 60.0 * t) * body_env
        # Click transient: short broadband burst in first 2 ms, weaker than
        # the body peak so the 5ms/50ms ratio is well below 1.
        click_len = min(int(0.002 * SAMPLE_RATE), length)
        click_env = np.exp(-np.arange(click_len) / 30.0)
        click = 0.4 * np.sin(2.0 * np.pi * 4_000.0 * t[:click_len]) * click_env
        body[:click_len] += click
        signal[onset_sample : onset_sample + length] += body
    return signal


class TestOnsetDetection:
    def test_four_hit_kick_detected(self) -> None:
        signal = _synthetic_kick_train(n_hits=4)
        onsets = detect_percussive_onsets(signal, sample_rate=SAMPLE_RATE)
        assert 3 <= onsets.size <= 5  # allow +/-1 slack for attack shape

    def test_silence_no_onsets(self) -> None:
        signal = np.zeros(int(0.5 * SAMPLE_RATE))
        onsets = detect_percussive_onsets(signal, sample_rate=SAMPLE_RATE)
        assert onsets.size == 0

    def test_sustained_sine_no_onsets(self) -> None:
        signal = _sine(440.0, duration=1.0)
        onsets = detect_percussive_onsets(signal, sample_rate=SAMPLE_RATE)
        # Sustained tone has no sharp attacks (maybe one at the onset itself).
        assert onsets.size <= 1

    def test_flat_noise_few_onsets(self) -> None:
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(int(1.0 * SAMPLE_RATE)) * 0.1
        onsets = detect_percussive_onsets(signal, sample_rate=SAMPLE_RATE)
        # Flat Gaussian noise has some prominence peaks but should stay small.
        assert onsets.size < 10


class TestPerHitMetrics:
    def test_metrics_shape(self) -> None:
        signal = _synthetic_kick_train(n_hits=4)
        summary = per_hit_transient_metrics(signal, sample_rate=SAMPLE_RATE)
        assert summary.hit_count >= 3
        assert 0.0 < summary.transient_peak_ratio_p50 <= 1.0

    def test_compressor_changes_transient_ratio(self) -> None:
        # A fast-attack compressor measurably changes the per-hit transient
        # ratio — depending on which part of the hit dominates, the ratio
        # can move up (click preserved while body gets shaved) or down
        # (click gets shaved while body catches up). Either way the
        # instrumentation must see a measurable change.
        signal = _synthetic_kick_train(n_hits=4)
        comp_result = synth.apply_compressor(
            signal,
            threshold_db=-30.0,
            ratio=8.0,
            attack_ms=0.5,
            release_ms=50.0,
            knee_db=0.0,
        )
        assert isinstance(comp_result, np.ndarray)
        comp_out: np.ndarray = comp_result
        dry_summary = per_hit_transient_metrics(signal, sample_rate=SAMPLE_RATE)
        wet_summary = per_hit_transient_metrics(comp_out, sample_rate=SAMPLE_RATE)
        assert dry_summary.hit_count == wet_summary.hit_count
        assert dry_summary.hit_count >= 3
        delta = abs(
            wet_summary.transient_peak_ratio_p50 - dry_summary.transient_peak_ratio_p50
        )
        assert delta > 0.05, f"expected measurable transient ratio shift, got {delta}"


# ---------------------------------------------------------------------------
# Per-effect IO delta: a_weighted + imd + transient_kill_db
# ---------------------------------------------------------------------------


class TestPerEffectIOMetrics:
    def test_a_weighted_metric_present(self) -> None:
        signal = _sine(880.0).astype(np.float64)
        driven = np.tanh(3.0 * signal)
        entry = synth._build_effect_analysis_entry(
            index=0,
            kind="drive",
            display_name="drive",
            input_signal=signal,
            output_signal=driven,
            sample_rate=SAMPLE_RATE,
        )
        assert "a_weighted_high_band_delta_db" in entry.metrics
        assert "imd_ratio_input" in entry.metrics
        assert "imd_ratio_output" in entry.metrics
        assert "imd_detection" in entry.metrics

    def test_percussive_flag_adds_transient_kill_metric(self) -> None:
        signal = _synthetic_kick_train(n_hits=4)
        comp_result = synth.apply_compressor(
            signal,
            threshold_db=-30.0,
            ratio=8.0,
            attack_ms=0.5,
            release_ms=50.0,
            knee_db=0.0,
        )
        assert isinstance(comp_result, np.ndarray)
        comp_out: np.ndarray = comp_result
        percussive_entry = synth._build_effect_analysis_entry(
            index=0,
            kind="compressor",
            display_name="compressor",
            input_signal=signal,
            output_signal=comp_out,
            sample_rate=SAMPLE_RATE,
            percussive=True,
        )
        non_percussive_entry = synth._build_effect_analysis_entry(
            index=0,
            kind="compressor",
            display_name="compressor",
            input_signal=signal,
            output_signal=comp_out,
            sample_rate=SAMPLE_RATE,
            percussive=False,
        )
        assert "transient_kill_db" in percussive_entry.metrics
        assert "hit_count" in percussive_entry.metrics
        assert int(percussive_entry.metrics["hit_count"]) >= 3
        assert "transient_kill_db" not in non_percussive_entry.metrics


# ---------------------------------------------------------------------------
# Voice-stem delta + new warning codes
# ---------------------------------------------------------------------------


class TestVoiceStemDelta:
    def test_build_voice_stem_delta_populates_metrics(self) -> None:
        signal = _synthetic_kick_train(n_hits=4)
        # Simulate a crude "crunchy" voice chain: soft-clip + brighten.
        wet = np.tanh(3.5 * signal)
        delta = synth.build_voice_stem_delta(
            voice_name="kick",
            dry_signal=signal,
            wet_signal=wet,
            sample_rate=SAMPLE_RATE,
            percussive=True,
        )
        assert delta["voice_name"] == "kick"
        assert delta["percussive"] is True
        metrics = delta["metrics"]
        assert "a_weighted_high_band_delta_db" in metrics
        assert "imd_ratio_output" in metrics
        assert "transient_kill_db" in metrics
        assert int(metrics["hit_count"]) >= 3

    def test_transient_killed_warning_fires(self) -> None:
        # Click-heavy kick train. The onset detector places onsets a few ms
        # into the hit (when the envelope has ramped up), so the 5 ms
        # transient window straddles the click's tail + body shoulder.
        # Attenuating the detected-onset region by ~15 dB kills the per-hit
        # transient measurably while leaving the sustain portion intact.
        signal = _synthetic_click_heavy_kick(n_hits=4)
        onsets = detect_percussive_onsets(signal, sample_rate=SAMPLE_RATE)
        assert onsets.size >= 3, f"expected >=3 onsets, got {onsets.size}"
        wet = signal.copy()
        attenuate_samples = int(0.010 * SAMPLE_RATE)
        for onset_idx in onsets:
            start = max(0, int(onset_idx))
            end = min(wet.size, start + attenuate_samples)
            wet[start:end] *= 0.15
        delta = synth.build_voice_stem_delta(
            voice_name="kick",
            dry_signal=signal,
            wet_signal=wet,
            sample_rate=SAMPLE_RATE,
            percussive=True,
        )
        codes = {w["code"] for w in delta["warnings"]}
        assert "percussive_transient_killed" in codes, (
            f"expected percussive_transient_killed, got "
            f"{codes} / metrics={delta['metrics']}"
        )

    def test_non_percussive_no_per_hit_warnings(self) -> None:
        signal = _sine(220.0, duration=1.0)
        wet = np.tanh(1.5 * signal)
        delta = synth.build_voice_stem_delta(
            voice_name="pad",
            dry_signal=signal,
            wet_signal=wet,
            sample_rate=SAMPLE_RATE,
            percussive=False,
        )
        codes = {w["code"] for w in delta["warnings"]}
        assert "percussive_transient_killed" not in codes
        assert "percussive_crunch_character" not in codes

    def test_perceptual_brightness_lift_warning_fires(self) -> None:
        signal = _sine(220.0, duration=1.0)
        # Strong saturation dumps energy into the 2-8 kHz band.
        wet = np.tanh(8.0 * signal)
        delta = synth.build_voice_stem_delta(
            voice_name="bass",
            dry_signal=signal,
            wet_signal=wet,
            sample_rate=SAMPLE_RATE,
            percussive=False,
        )
        codes = {w["code"] for w in delta["warnings"]}
        assert "perceptual_brightness_lift" in codes


# ---------------------------------------------------------------------------
# Percussive auto-detection heuristic
# ---------------------------------------------------------------------------


class TestPercussiveHeuristic:
    def test_drum_voice_engine_detected(self) -> None:
        voice = Voice(name="k", synth_defaults={"engine": "drum_voice"})
        assert voice.is_percussive() is True

    def test_kick_tom_engine_detected(self) -> None:
        voice = Voice(name="k", synth_defaults={"engine": "kick_tom"})
        assert voice.is_percussive() is True

    def test_choke_group_detected(self) -> None:
        voice = Voice(
            name="h", synth_defaults={"engine": "synth_voice"}, choke_group="hats"
        )
        assert voice.is_percussive() is True

    def test_peak_normalized_detected(self) -> None:
        voice = Voice(
            name="perc",
            synth_defaults={"engine": "synth_voice"},
            normalize_lufs=None,
            normalize_peak_db=-6.0,
        )
        assert voice.is_percussive() is True

    def test_synth_voice_not_percussive(self) -> None:
        voice = Voice(name="pad", synth_defaults={"engine": "synth_voice"})
        assert voice.is_percussive() is False

    def test_explicit_override_wins(self) -> None:
        voice = Voice(
            name="k",
            synth_defaults={"engine": "drum_voice"},
            percussive=False,
        )
        assert voice.is_percussive() is False

    def test_add_drum_voice_percussive_override(self) -> None:
        score = Score(f0_hz=55.0)
        add_drum_voice(
            score,
            "k",
            engine="drum_voice",
            preset="808_hiphop",
            percussive=False,
        )
        assert score.voices["k"].is_percussive() is False


# ---------------------------------------------------------------------------
# Chain summary: perceptual_brightness_lift rolls up a_weighted delta
# ---------------------------------------------------------------------------


class TestChainSummaryAWeighted:
    def test_perceptual_brightness_lift_warn(self) -> None:
        entries = [
            {
                "index": 0,
                "kind": "drive",
                "display_name": "drive",
                "metrics": {"a_weighted_high_band_delta_db": 3.0},
                "warnings": [],
            },
            {
                "index": 1,
                "kind": "clipper",
                "display_name": "clipper",
                "metrics": {"a_weighted_high_band_delta_db": 2.0},
                "warnings": [],
            },
        ]
        summary = synth.build_chain_summary_from_dicts(entries, chain_label="drum_bus")
        assert summary is not None
        codes = {w.code for w in summary.warnings}
        assert "perceptual_brightness_lift" in codes
        assert summary.metrics["total_a_weighted_high_band_lift_db"] == 5.0

    def test_perceptual_brightness_lift_suppressed_on_pure_linear_chain(
        self,
    ) -> None:
        """Pure linear time-based chain shouldn't trip perceptual_brightness_lift.

        Same reasoning as chain_papery: a feedback delay on a bright source
        accumulates a_weighted_high_band_delta without any stage introducing
        harmonic buildup. The warning is about stacked brittleness, not raw
        wet-signal replication.
        """
        entries = [
            {
                "index": 0,
                "kind": "eq",
                "display_name": "eq",
                "metrics": {"a_weighted_high_band_delta_db": 2.5},
                "warnings": [],
            },
            {
                "index": 1,
                "kind": "delay",
                "display_name": "delay",
                "metrics": {"a_weighted_high_band_delta_db": 3.5},
                "warnings": [],
            },
        ]
        summary = synth.build_chain_summary_from_dicts(
            entries, chain_label="bell_delay"
        )
        assert summary is not None
        assert summary.metrics["total_a_weighted_high_band_lift_db"] == 6.0
        codes = {w.code for w in summary.warnings}
        assert "perceptual_brightness_lift" not in codes


# ---------------------------------------------------------------------------
# End-to-end: voice_stem_deltas present in render_with_effect_analysis output
# ---------------------------------------------------------------------------


class TestRenderVoiceStemDeltas:
    def test_render_populates_voice_stem_deltas(self) -> None:
        score = Score(f0_hz=55.0)
        score.add_voice(
            "pad",
            synth_defaults={"engine": "synth_voice"},
            effects=[
                EffectSpec(
                    "transistor",
                    {"character": "soft_clip", "drive": 1.5, "mix": 0.5},
                ),
            ],
        )
        score.voices["pad"].notes.append(
            NoteEvent(start=0.0, duration=0.5, partial=1.0, amp=0.3)
        )
        _mix, _stems, _sends, effect_analysis = score.render_with_effect_analysis()
        assert "voice_stem_deltas" in effect_analysis
        assert "pad" in effect_analysis["voice_stem_deltas"]
        delta_entry = effect_analysis["voice_stem_deltas"]["pad"]
        assert "a_weighted_high_band_delta_db" in delta_entry["metrics"]
        assert delta_entry["percussive"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
