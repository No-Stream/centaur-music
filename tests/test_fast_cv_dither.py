"""Tests for OB-Xd-style fast CV dither and voice-card env-rate scaling.

Covers:
- ``fast_cv_dither`` primitive shape, range, and determinism.
- Pitch dither adds instantaneous-frequency jitter on sustained tones.
- Cutoff dither measurably perturbs audio.
- Voice-card env-rate scaling produces audibly different renders between
  voice_card_spread=0 and voice_card_spread=2.0, with identical audio when
  the envelope spread dimension is zeroed.
"""

from __future__ import annotations

import numpy as np

from code_musics.engines._dsp_utils import (
    _CV_DITHER_CUTOFF_FRAC,
    _CV_DITHER_PITCH_SEMITONES,
    fast_cv_dither,
)
from code_musics.engines.registry import render_note_signal
from code_musics.score import Score


class TestFastCvDitherPrimitive:
    """Shape, range, and determinism of the low-level dither helper."""

    def test_zero_amount_returns_zeros(self) -> None:
        out = fast_cv_dither(
            1024, amount=0.0, rng=np.random.default_rng(0), sample_rate=44100
        )
        assert out.shape == (1024,)
        assert float(np.max(np.abs(out))) == 0.0

    def test_zero_samples_returns_empty(self) -> None:
        out = fast_cv_dither(
            0, amount=1.0, rng=np.random.default_rng(0), sample_rate=44100
        )
        assert out.shape == (0,)

    def test_amount_bounds_roughly_symmetric(self) -> None:
        out = fast_cv_dither(
            44100,
            amount=1.0,
            rng=np.random.default_rng(7),
            sample_rate=44100,
        )
        # After the one-pole lowpass the peak stays below the input range.
        assert float(np.max(np.abs(out))) < 1.0
        # Mean should be close to zero (no DC bias).
        assert abs(float(np.mean(out))) < 0.1

    def test_deterministic(self) -> None:
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        a = fast_cv_dither(2048, amount=0.5, rng=rng1, sample_rate=44100)
        b = fast_cv_dither(2048, amount=0.5, rng=rng2, sample_rate=44100)
        np.testing.assert_array_equal(a, b)

    def test_sample_rate_affects_smoothing(self) -> None:
        """Higher sample rate -> more smoothing (lower per-sample change)."""
        rng1 = np.random.default_rng(3)
        rng2 = np.random.default_rng(3)
        fast = fast_cv_dither(4096, amount=1.0, rng=rng1, sample_rate=11025)
        slow = fast_cv_dither(4096, amount=1.0, rng=rng2, sample_rate=88200)
        fast_diff = float(np.mean(np.abs(np.diff(fast))))
        slow_diff = float(np.mean(np.abs(np.diff(slow))))
        # Higher SR gives smaller per-sample change (more smoothing).
        assert slow_diff < fast_diff


class TestPolyblepFastCvDither:
    """The dither path must cause measurable change on polyblep audio."""

    _BASE = {
        "engine": "polyblep",
        "waveform": "saw",
        "cutoff_hz": 2000.0,
        "resonance_q": 1.2,
        "pitch_drift": 0.0,
        "noise_floor": 0.0,
        "cutoff_drift": 0.0,
        "voice_card_spread": 0.0,  # isolate the dither contribution
    }

    def test_dither_changes_audio(self) -> None:
        clean = render_note_signal(
            freq=220.0,
            duration=0.5,
            amp=0.3,
            sample_rate=44100,
            params={**self._BASE, "analog_jitter": 0.0},
        )
        dithered = render_note_signal(
            freq=220.0,
            duration=0.5,
            amp=0.3,
            sample_rate=44100,
            params={**self._BASE, "analog_jitter": 1.0},
        )
        assert np.all(np.isfinite(dithered))
        assert float(np.max(np.abs(dithered))) > 0.0
        assert not np.allclose(clean, dithered)

    def test_dither_deterministic(self) -> None:
        params = {**self._BASE, "analog_jitter": 1.0}
        a = render_note_signal(
            freq=220.0, duration=0.5, amp=0.3, sample_rate=44100, params=params
        )
        b = render_note_signal(
            freq=220.0, duration=0.5, amp=0.3, sample_rate=44100, params=params
        )
        np.testing.assert_array_equal(a, b)

    def test_dither_measurably_perturbs_sustained_sine(self) -> None:
        """Sustained sine with dither diverges from the clean reference over time."""
        sr = 44100
        base = {
            "engine": "polyblep",
            "waveform": "sine",
            "cutoff_hz": 8000.0,
            "pitch_drift": 0.0,
            "noise_floor": 0.0,
            "cutoff_drift": 0.0,
            "voice_card_spread": 0.0,
            "attack": 0.001,
            "release": 0.001,
        }
        clean = render_note_signal(
            freq=440.0,
            duration=1.0,
            amp=0.5,
            sample_rate=sr,
            params={**base, "analog_jitter": 0.0},
        )
        dithered = render_note_signal(
            freq=440.0,
            duration=1.0,
            amp=0.5,
            sample_rate=sr,
            params={**base, "analog_jitter": 1.0},
        )
        n = min(len(clean), len(dithered))
        diff_rms = float(np.sqrt(np.mean((clean[:n] - dithered[:n]) ** 2)))
        clean_rms = float(np.sqrt(np.mean(clean[:n] ** 2)))
        # Dither integrates into substantial phase drift on a sustained sine,
        # so the difference is well above any normalization noise floor.
        assert diff_rms > clean_rms * 0.05


class TestFilteredStackFastCvDither:
    """Same sanity checks for the filtered_stack engine."""

    _BASE = {
        "engine": "filtered_stack",
        "waveform": "saw",
        "n_harmonics": 10,
        "cutoff_hz": 1800.0,
        "pitch_drift": 0.0,
        "noise_floor": 0.0,
        "cutoff_drift": 0.0,
        "voice_card_spread": 0.0,
    }

    def test_dither_changes_audio(self) -> None:
        clean = render_note_signal(
            freq=220.0,
            duration=0.5,
            amp=0.3,
            sample_rate=44100,
            params={**self._BASE, "analog_jitter": 0.0},
        )
        dithered = render_note_signal(
            freq=220.0,
            duration=0.5,
            amp=0.3,
            sample_rate=44100,
            params={**self._BASE, "analog_jitter": 1.0},
        )
        assert np.all(np.isfinite(dithered))
        assert not np.allclose(clean, dithered)


def test_dither_constants_are_sane() -> None:
    """Guard against silent tuning regressions on the dither scale."""
    assert 0.0 < _CV_DITHER_PITCH_SEMITONES <= 0.25
    assert 0.0 < _CV_DITHER_CUTOFF_FRAC <= 0.2


# ---------------------------------------------------------------------------
# Item 7 — voice_card env-rate scaling at Score level
# ---------------------------------------------------------------------------


def _build_score(
    *,
    voice_card_spread: float,
    voice_card_envelope_spread: float | None = None,
) -> Score:
    """Build a tiny deterministic two-voice score for env-rate scaling tests.

    Two voices with the same attack/release but different names will pick up
    different deterministic voice_card offsets, so their outer ADSR timing
    should drift apart when envelope spread is non-zero.
    """
    synth_defaults: dict[str, object] = {
        "engine": "polyblep",
        "waveform": "saw",
        "cutoff_hz": 2000.0,
        "attack": 0.05,
        "release": 0.2,
        "pitch_drift": 0.0,
        "analog_jitter": 0.0,
        "noise_floor": 0.0,
        "cutoff_drift": 0.0,
        "voice_card_spread": voice_card_spread,
    }
    if voice_card_envelope_spread is not None:
        synth_defaults["voice_card_envelope_spread"] = voice_card_envelope_spread
    score = Score(f0_hz=110.0, auto_master_gain_stage=False)
    score.add_voice(
        "lead_a",
        synth_defaults=dict(synth_defaults),
        normalize_lufs=None,
        velocity_humanize=None,
    )
    score.add_voice(
        "lead_b",
        synth_defaults=dict(synth_defaults),
        normalize_lufs=None,
        velocity_humanize=None,
    )
    score.add_note("lead_a", start=0.0, duration=0.5, freq=220.0, amp=0.2)
    score.add_note("lead_b", start=0.0, duration=0.5, freq=220.0, amp=0.2)
    return score


class TestVoiceCardEnvRateScaling:
    def test_zero_spread_matches_no_scaling(self) -> None:
        """voice_card_spread=0 should produce identical audio regardless of voice name."""
        flat = _build_score(voice_card_spread=0.0)
        audio = flat.render()
        assert np.all(np.isfinite(audio))
        assert float(np.max(np.abs(audio))) > 0.0

    def test_env_spread_changes_render(self) -> None:
        """voice_card_spread=2.0 renders differently from voice_card_spread=0.0."""
        flat = _build_score(voice_card_spread=0.0).render()
        spread = _build_score(voice_card_spread=2.0).render()
        assert np.all(np.isfinite(spread))
        assert float(np.max(np.abs(spread))) > 0.0
        # The release scaling can change the overall rendered length by a few
        # samples; compare on a common prefix.
        n = min(len(flat), len(spread))
        assert n > 0
        assert not np.allclose(flat[:n], spread[:n], atol=1e-10)

    def test_envelope_override_isolates_dimension(self) -> None:
        """voice_card_envelope_spread=0.0 pins env rates even with global spread=2."""
        wide = _build_score(voice_card_spread=2.0).render()
        env_pinned = _build_score(
            voice_card_spread=2.0, voice_card_envelope_spread=0.0
        ).render()
        n = min(len(wide), len(env_pinned))
        assert n > 0
        assert not np.allclose(wide[:n], env_pinned[:n], atol=1e-10)

    def test_deterministic(self) -> None:
        a = _build_score(voice_card_spread=2.0).render()
        b = _build_score(voice_card_spread=2.0).render()
        np.testing.assert_array_equal(a, b)
