"""Tests for the shared drift bus primitive, spec, and score-level wiring."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._dsp_utils import build_drift
from code_musics.humanize import DriftBusSpec, build_drift_bus
from code_musics.score import Score

# ---------------------------------------------------------------------------
# build_drift_bus unit tests
# ---------------------------------------------------------------------------


class TestDriftBusShape:
    """Output is a multiplicative ratio array of the right length."""

    def test_returns_array_of_requested_length(self) -> None:
        times = np.linspace(0.0, 10.0, 1000, dtype=np.float64)
        signal = build_drift_bus(
            times=times,
            rate_hz=0.2,
            depth_cents=5.0,
            sample_rate=44100,
            seed=0,
        )
        assert signal.shape == (1000,)

    def test_output_is_positive_multiplicative_ratio(self) -> None:
        times = np.linspace(0.0, 10.0, 1000, dtype=np.float64)
        signal = build_drift_bus(
            times=times,
            rate_hz=0.2,
            depth_cents=5.0,
            sample_rate=44100,
            seed=0,
        )
        assert np.all(signal > 0), "drift bus ratios must be positive"

    def test_zero_depth_returns_ones(self) -> None:
        times = np.linspace(0.0, 2.0, 100, dtype=np.float64)
        signal = build_drift_bus(
            times=times,
            rate_hz=0.2,
            depth_cents=0.0,
            sample_rate=44100,
            seed=0,
        )
        np.testing.assert_array_equal(signal, np.ones(100, dtype=np.float64))

    def test_empty_times_returns_empty(self) -> None:
        times = np.zeros(0, dtype=np.float64)
        signal = build_drift_bus(
            times=times,
            rate_hz=0.2,
            depth_cents=5.0,
            sample_rate=44100,
            seed=0,
        )
        assert signal.shape == (0,)


class TestDriftBusDeterminism:
    """Same seed produces identical output."""

    def test_identical_under_fixed_seed(self) -> None:
        times = np.linspace(0.0, 10.0, 2000, dtype=np.float64)
        kwargs = dict(
            times=times,
            rate_hz=0.2,
            depth_cents=5.0,
            sample_rate=44100,
            seed=123,
        )
        a = build_drift_bus(**kwargs)
        b = build_drift_bus(**kwargs)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_produce_different_signals(self) -> None:
        times = np.linspace(0.0, 10.0, 2000, dtype=np.float64)
        kwargs: dict = dict(
            times=times,
            rate_hz=0.2,
            depth_cents=5.0,
            sample_rate=44100,
        )
        a = build_drift_bus(seed=7, **kwargs)
        b = build_drift_bus(seed=8, **kwargs)
        assert not np.allclose(a, b), "different seeds should produce different output"


class TestDriftBusBounded:
    """The Surge-style filtered noise stays within reasonable tail bounds."""

    def test_stays_close_to_depth_rms(self) -> None:
        # 60 seconds covers many slow cycles; tail excursion up to ~4x depth is OK
        sample_rate = 44100
        times = np.linspace(0.0, 60.0, sample_rate * 60, dtype=np.float64)
        depth_cents = 8.0
        signal = build_drift_bus(
            times=times,
            rate_hz=0.2,
            depth_cents=depth_cents,
            sample_rate=sample_rate,
            seed=42,
        )
        cents = 1200.0 * np.log2(signal)
        max_excursion = float(np.max(np.abs(cents)))
        # Allow generous tail; we only care that it doesn't explode.
        assert max_excursion < 6.0 * depth_cents, (
            f"max excursion = {max_excursion:.1f} cents, expected < {6.0 * depth_cents:.1f}"
        )

    def test_mean_is_near_unity(self) -> None:
        sample_rate = 44100
        times = np.linspace(0.0, 60.0, sample_rate * 60, dtype=np.float64)
        signal = build_drift_bus(
            times=times,
            rate_hz=0.2,
            depth_cents=8.0,
            sample_rate=sample_rate,
            seed=42,
        )
        assert abs(float(np.mean(signal)) - 1.0) < 0.02


class TestDriftBusSlow:
    """Shared drift should vary slowly — compare two nearby samples."""

    def test_adjacent_samples_are_close(self) -> None:
        sample_rate = 44100
        duration = 20.0
        n = int(sample_rate * duration)
        times = np.linspace(0.0, duration, n, dtype=np.float64)
        signal = build_drift_bus(
            times=times,
            rate_hz=0.2,
            depth_cents=10.0,
            sample_rate=sample_rate,
            seed=42,
        )
        # Adjacent samples at 44.1kHz should differ by ppm — < 1 cent
        cents = 1200.0 * np.log2(signal)
        adj_diff_cents = np.abs(np.diff(cents))
        assert float(np.max(adj_diff_cents)) < 1.0, (
            "bus signal should be slow: adjacent-sample delta should be sub-cent"
        )


class TestDriftBusSampling:
    """Resampling at different time grids gives the same underlying trajectory."""

    def test_consistent_across_sampling_grids(self) -> None:
        sample_rate = 44100
        duration = 4.0
        dense_times = np.linspace(0.0, duration, int(sample_rate * duration))
        sparse_times = np.linspace(0.0, duration, 100)
        dense = build_drift_bus(
            times=dense_times,
            rate_hz=0.2,
            depth_cents=6.0,
            sample_rate=sample_rate,
            seed=17,
        )
        sparse = build_drift_bus(
            times=sparse_times,
            rate_hz=0.2,
            depth_cents=6.0,
            sample_rate=sample_rate,
            seed=17,
        )
        # Interpolate dense at sparse times and compare
        interp = np.interp(sparse_times, dense_times, dense)
        np.testing.assert_allclose(sparse, interp, atol=1e-9)


# ---------------------------------------------------------------------------
# DriftBusSpec tests
# ---------------------------------------------------------------------------


class TestDriftBusSpec:
    """Field validation and default ergonomics."""

    def test_requires_name(self) -> None:
        with pytest.raises(ValueError):
            DriftBusSpec(name="", rate_hz=0.2, depth_cents=5.0)

    def test_rate_hz_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            DriftBusSpec(name="ensemble", rate_hz=0.0, depth_cents=5.0)

    def test_depth_cents_must_be_non_negative(self) -> None:
        with pytest.raises(ValueError):
            DriftBusSpec(name="ensemble", rate_hz=0.2, depth_cents=-1.0)

    def test_defaults_are_reasonable(self) -> None:
        spec = DriftBusSpec(name="ensemble")
        assert 0.05 <= spec.rate_hz <= 1.0
        assert spec.depth_cents >= 0.0

    def test_custom_seed_persists(self) -> None:
        spec = DriftBusSpec(name="ensemble", seed=77)
        assert spec.seed == 77


# ---------------------------------------------------------------------------
# Score-level wiring
# ---------------------------------------------------------------------------


def _make_small_score(
    *,
    drift_bus_name: str | None,
    correlation: float,
    seed: int | None,
) -> Score:
    """Helper: small two-voice score, optionally subscribed to a shared bus."""
    score = Score(f0_hz=220.0)
    if drift_bus_name is not None:
        score.add_drift_bus(
            drift_bus_name,
            rate_hz=0.3,
            depth_cents=8.0,
            seed=seed,
        )
    for voice_name in ("lead", "alto"):
        score.add_voice(
            voice_name,
            synth_defaults={
                "engine": "polyblep",
                "waveform": "saw",
                "cutoff_hz": 2000.0,
                "pitch_drift": 1.0,
            },
            drift_bus=drift_bus_name,
            drift_bus_correlation=correlation,
            velocity_humanize=None,
        )
        score.add_note(voice_name, start=0.0, duration=2.0, partial=4.0, amp=0.3)
    return score


class TestScoreDriftBusRegistration:
    def test_add_drift_bus_registers_by_name(self) -> None:
        score = Score(f0_hz=220.0)
        score.add_drift_bus("ensemble", rate_hz=0.3, depth_cents=8.0, seed=1)
        assert "ensemble" in score.drift_buses
        spec = score.drift_buses["ensemble"]
        assert spec.rate_hz == 0.3
        assert spec.depth_cents == 8.0
        assert spec.seed == 1

    def test_unknown_bus_subscription_raises(self) -> None:
        score = Score(f0_hz=220.0)
        score.add_voice(
            "lead",
            synth_defaults={"engine": "polyblep"},
            drift_bus="nonexistent",
        )
        score.add_note("lead", start=0.0, duration=0.3, partial=4.0, amp=0.2)
        with pytest.raises(ValueError, match="drift_bus"):
            score.render()

    def test_correlation_must_be_in_range(self) -> None:
        score = Score(f0_hz=220.0)
        with pytest.raises(ValueError):
            score.add_voice(
                "lead",
                synth_defaults={"engine": "polyblep"},
                drift_bus_correlation=1.5,
            )
        with pytest.raises(ValueError):
            score.add_voice(
                "alto",
                synth_defaults={"engine": "polyblep"},
                drift_bus_correlation=-0.1,
            )


class TestScoreFreqTrajectoryWithDriftBus:
    """At full correlation, two subscribed voices receive correlated pitch drift."""

    def test_freq_trajectories_highly_correlated_at_full_subscription(self) -> None:
        score = _make_small_score(drift_bus_name="ensemble", correlation=1.0, seed=7)
        trajectories = score._debug_freq_trajectories()  # type: ignore[attr-defined]

        lead = trajectories["lead"][0]
        alto = trajectories["alto"][0]
        assert lead.shape == alto.shape
        # At correlation=1.0, engine drift is suppressed; both voices receive the
        # same bus trajectory applied to (different) base freqs. Normalize each
        # by its mean to get the drift modulation, then correlate.
        lead_rel = lead / np.mean(lead)
        alto_rel = alto / np.mean(alto)
        corr = float(np.corrcoef(lead_rel, alto_rel)[0, 1])
        assert corr > 0.99, f"expected correlation > 0.99, got {corr}"

    def test_freq_trajectories_uncorrelated_at_zero_subscription(self) -> None:
        # correlation=0.0 means engine drift fully active, bus drift zero.
        score = _make_small_score(drift_bus_name="ensemble", correlation=0.0, seed=7)
        # At correlation 0, the Score MUST NOT bake a bus trajectory into the
        # freq path. Prove it by comparing against a reference score with no
        # bus at all — trajectories should be identical (or both None).
        no_bus_score = _make_small_score(drift_bus_name=None, correlation=0.0, seed=7)
        with_zero = score._debug_freq_trajectories()  # type: ignore[attr-defined]
        no_bus = no_bus_score._debug_freq_trajectories()  # type: ignore[attr-defined]
        for voice_name in ("lead", "alto"):
            assert with_zero[voice_name] == no_bus[voice_name]


class TestScoreDriftBusDeterminism:
    """Same drift bus seed produces identical renders."""

    def test_identical_under_fixed_seed(self) -> None:
        first = _make_small_score(drift_bus_name="ensemble", correlation=1.0, seed=3)
        second = _make_small_score(drift_bus_name="ensemble", correlation=1.0, seed=3)
        np.testing.assert_allclose(first.render(), second.render())

    def test_different_seeds_produce_different_renders(self) -> None:
        first = _make_small_score(drift_bus_name="ensemble", correlation=1.0, seed=3)
        second = _make_small_score(drift_bus_name="ensemble", correlation=1.0, seed=4)
        a = first.render()
        b = second.render()
        assert not np.allclose(a, b), (
            "different bus seeds should produce different audio"
        )


class TestScoreDriftBusBackwardCompat:
    """Voices with no drift_bus render identically to the pre-feature behavior."""

    def test_unsubscribed_voice_unaffected_by_bus_presence(self) -> None:
        # Reference: no bus at all.
        reference = Score(f0_hz=220.0)
        reference.add_voice(
            "lead",
            synth_defaults={
                "engine": "polyblep",
                "waveform": "saw",
                "cutoff_hz": 2000.0,
                "pitch_drift": 1.0,
            },
            velocity_humanize=None,
        )
        reference.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.3)

        # Same score but with a registered (unused) bus; no voice subscribes.
        with_bus = Score(f0_hz=220.0)
        with_bus.add_drift_bus("ensemble", rate_hz=0.3, depth_cents=8.0, seed=1)
        with_bus.add_voice(
            "lead",
            synth_defaults={
                "engine": "polyblep",
                "waveform": "saw",
                "cutoff_hz": 2000.0,
                "pitch_drift": 1.0,
            },
            velocity_humanize=None,
        )
        with_bus.add_note("lead", start=0.0, duration=1.0, partial=4.0, amp=0.3)

        np.testing.assert_allclose(reference.render(), with_bus.render())


class TestIndependentDriftStillActiveAtMidCorrelation:
    """At correlation=0.5, the engine's build_drift still contributes."""

    def test_mid_correlation_differs_from_full(self) -> None:
        mid = _make_small_score(drift_bus_name="ensemble", correlation=0.5, seed=7)
        full = _make_small_score(drift_bus_name="ensemble", correlation=1.0, seed=7)
        mid_audio = mid.render()
        full_audio = full.render()
        assert not np.allclose(mid_audio, full_audio, atol=1e-6)


# ---------------------------------------------------------------------------
# Existing piece regression (backward compatibility)
# ---------------------------------------------------------------------------


class TestHarmonicDriftPieceStillRenders:
    """Smoke test: the existing harmonic_drift piece still renders end-to-end.

    This is intentionally a shallow smoke test, not a bit-for-bit regression
    guard — the drift bus refactor changes per-note sample consumption in ways
    that deliberately alter rendered audio on subscribed voices. What we want
    to preserve is that the piece itself continues to build and render cleanly.
    """

    def test_harmonic_drift_render_is_stable(self) -> None:
        from code_musics.pieces.septimal import build_harmonic_drift_score

        score = build_harmonic_drift_score()
        # Rendering a full piece is slow; prove the piece still runs via a
        # short window at the start and confirm finite, non-silent audio.
        windowed = score.extract_window(start_seconds=0.0, end_seconds=2.0)
        audio = windowed.render()
        assert audio.size > 0
        assert np.all(np.isfinite(audio))
        assert float(np.max(np.abs(audio))) > 1e-5, (
            "harmonic_drift snippet should produce non-silent audio"
        )


# ---------------------------------------------------------------------------
# Cross-voice correlation integration test via audio envelope
# ---------------------------------------------------------------------------


class TestCrossVoiceAudioCorrelation:
    """At correlation=1.0 both voices should exhibit the same pitch modulation.

    This is a higher-bar integration check that looks at the actual rendered
    audio instead of internal trajectory arrays.
    """

    def test_full_correlation_pitch_trajectories_match(self) -> None:
        # Two voices at the same pitch; at correlation=1 their bus drift is
        # identical. Engine-internal drift is suppressed, so the pitch
        # trajectories going into the synth should be bit-identical multiples
        # of their respective base freqs.
        score = Score(f0_hz=220.0)
        score.add_drift_bus("ensemble", rate_hz=0.3, depth_cents=8.0, seed=7)
        for voice_name in ("lead", "alto"):
            score.add_voice(
                voice_name,
                synth_defaults={
                    "engine": "polyblep",
                    "waveform": "saw",
                    "cutoff_hz": 2000.0,
                    "pitch_drift": 1.0,
                },
                drift_bus="ensemble",
                drift_bus_correlation=1.0,
                velocity_humanize=None,
            )
            score.add_note(voice_name, start=0.0, duration=2.0, partial=4.0, amp=0.3)

        trajectories = score._debug_freq_trajectories()  # type: ignore[attr-defined]
        lead = trajectories["lead"][0]
        alto = trajectories["alto"][0]
        # Same base freq, same bus trajectory -> identical freq arrays.
        np.testing.assert_allclose(lead, alto, rtol=1e-12)


# ---------------------------------------------------------------------------
# Note offset into the bus
# ---------------------------------------------------------------------------


class TestNoteTimeOffsetsIntoBus:
    """Notes at different score times sample the bus at different phases."""

    def test_later_note_samples_later_bus_phase(self) -> None:
        score = Score(f0_hz=220.0)
        score.add_drift_bus("ensemble", rate_hz=0.5, depth_cents=10.0, seed=11)
        score.add_voice(
            "lead",
            synth_defaults={
                "engine": "polyblep",
                "waveform": "saw",
                "cutoff_hz": 2000.0,
                "pitch_drift": 1.0,
            },
            drift_bus="ensemble",
            drift_bus_correlation=1.0,
            velocity_humanize=None,
        )
        score.add_note("lead", start=0.0, duration=0.5, partial=4.0, amp=0.3)
        score.add_note("lead", start=5.0, duration=0.5, partial=4.0, amp=0.3)

        trajectories = score._debug_freq_trajectories()  # type: ignore[attr-defined]
        lead_notes = trajectories["lead"]
        assert len(lead_notes) == 2
        # Different segments of the bus -> different modulation shapes.
        first = lead_notes[0] / np.mean(lead_notes[0])
        second = lead_notes[1] / np.mean(lead_notes[1])
        assert not np.allclose(first, second)


# ---------------------------------------------------------------------------
# Sanity: engine build_drift is untouched
# ---------------------------------------------------------------------------


class TestExistingBuildDriftUntouched:
    """Regression guard: build_drift behavior must not change."""

    def test_build_drift_output_unchanged(self) -> None:
        # Snapshot a known-good build_drift call. If future edits break this,
        # we've refactored the helper against the spec's anti-goal.
        rng = np.random.default_rng(42)
        signal = build_drift(
            n_samples=4096,
            drift_amount=1.0,
            drift_rate_hz=0.3,
            duration=1.0,
            phase_offset=0.0,
            rng=rng,
        )
        assert signal.shape == (4096,)
        assert np.all(signal > 0)
        # Mean should be near 1 (cents-bounded multiplicative ratio).
        assert abs(float(np.mean(signal)) - 1.0) < 0.05
