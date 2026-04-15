"""Tests for voice-level sympathetic resonance."""

from __future__ import annotations

from typing import Any

import numpy as np

from code_musics.score import Score

SAMPLE_RATE = 22_050


def _make_score(**voice_kwargs: Any) -> Score:  # noqa: ANN401
    score = Score(f0_hz=220.0, sample_rate=SAMPLE_RATE)
    score.add_voice("v", synth_defaults={"engine": "additive"}, **voice_kwargs)
    return score


class TestSympatheticDefaults:
    def test_off_by_default(self) -> None:
        score = _make_score()
        assert score.voices["v"].sympathetic_amount == 0.0

    def test_default_does_not_alter_output(self) -> None:
        score_a = _make_score()
        score_a.add_note("v", start=0.0, duration=0.5, partial=1, amp=0.5)
        audio_a = score_a.render()

        score_b = _make_score(sympathetic_amount=0.0)
        score_b.add_note("v", start=0.0, duration=0.5, partial=1, amp=0.5)
        audio_b = score_b.render()

        min_len = min(len(audio_a), len(audio_b))
        np.testing.assert_allclose(audio_a[:min_len], audio_b[:min_len], atol=1e-10)


class TestSympatheticBasic:
    def test_nondestructive_finite_output(self) -> None:
        score = _make_score(sympathetic_amount=0.3, normalize_lufs=None)
        score.add_note("v", start=0.0, duration=0.5, partial=1, amp=0.5)
        score.add_note("v", start=0.1, duration=0.5, partial=2, amp=0.4)
        audio = score.render()
        assert audio.size > 0
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) > 0.0

    def test_adds_energy(self) -> None:
        score_off = _make_score(sympathetic_amount=0.0, normalize_lufs=None)
        score_off.add_note("v", start=0.0, duration=0.5, partial=1, amp=0.5)
        score_off.add_note("v", start=0.1, duration=0.5, partial=2, amp=0.4)
        audio_off = score_off.render()

        score_on = _make_score(sympathetic_amount=0.3, normalize_lufs=None)
        score_on.add_note("v", start=0.0, duration=0.5, partial=1, amp=0.5)
        score_on.add_note("v", start=0.1, duration=0.5, partial=2, amp=0.4)
        audio_on = score_on.render()

        min_len = min(len(audio_off), len(audio_on))
        energy_off = float(np.sum(audio_off[:min_len] ** 2))
        energy_on = float(np.sum(audio_on[:min_len] ** 2))
        assert energy_on >= energy_off


class TestSympatheticHarmonicRelation:
    def test_consonant_interval_more_resonance_than_dissonant(self) -> None:
        def _resonance_energy(partial_a: float, partial_b: float) -> float:
            kwargs: dict = {
                "sympathetic_modes": 8,
                "normalize_lufs": None,
            }
            score_off = _make_score(sympathetic_amount=0.0, **kwargs)
            score_off.add_note("v", start=0.0, duration=0.5, partial=partial_a, amp=0.5)
            score_off.add_note("v", start=0.0, duration=0.5, partial=partial_b, amp=0.5)
            dry = score_off.render()

            score_on = _make_score(sympathetic_amount=0.5, **kwargs)
            score_on.add_note("v", start=0.0, duration=0.5, partial=partial_a, amp=0.5)
            score_on.add_note("v", start=0.0, duration=0.5, partial=partial_b, amp=0.5)
            wet = score_on.render()

            min_len = min(len(dry), len(wet))
            diff = wet[:min_len] - dry[:min_len]
            return float(np.sum(diff**2))

        consonant_energy = _resonance_energy(1.0, 1.5)
        dissonant_energy = _resonance_energy(1.0, 1.41)
        assert consonant_energy > dissonant_energy


class TestSympatheticIntegration:
    def test_harpsichord_with_sympathetic(self) -> None:
        score = Score(f0_hz=220.0, sample_rate=SAMPLE_RATE)
        score.add_voice(
            "hpsi",
            synth_defaults={"engine": "harpsichord"},
            sympathetic_amount=0.2,
            sympathetic_decay_s=1.5,
            sympathetic_modes=6,
            normalize_lufs=-24.0,
        )
        score.add_note("hpsi", start=0.0, duration=0.8, partial=1, amp=0.7)
        score.add_note("hpsi", start=0.2, duration=0.8, partial=1.5, amp=0.6)
        audio = score.render()
        assert audio.size > 0
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) > 0.0

    def test_piano_with_sympathetic(self) -> None:
        score = Score(f0_hz=220.0, sample_rate=SAMPLE_RATE)
        score.add_voice(
            "piano",
            synth_defaults={"engine": "piano"},
            sympathetic_amount=0.15,
            normalize_lufs=-24.0,
        )
        score.add_note("piano", start=0.0, duration=0.6, partial=1, amp=0.6)
        score.add_note("piano", start=0.0, duration=0.6, partial=2, amp=0.5)
        audio = score.render()
        assert audio.size > 0
        assert np.all(np.isfinite(audio))
