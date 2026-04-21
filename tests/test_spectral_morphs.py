"""Tests for Vital-style spectral morph operators on the additive engine.

Covers the five morph operators (``inharmonic_scale``, ``phase_disperse``,
``smear``, ``shepard``, ``random_amplitudes``) plus sigma-approximation
band-limiting.
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._spectral_morphs import (
    MORPH_TYPES,
    apply_inharmonic_scale,
    apply_phase_disperse,
    apply_random_amplitudes,
    apply_shepard,
    apply_sigma_approximation,
    apply_smear,
    apply_spectral_morph,
)
from code_musics.engines.registry import render_note_signal
from code_musics.score import Score
from code_musics.spectra import ratio_spectrum

# ---------------------------------------------------------------------------
# Pure-function morph tests
# ---------------------------------------------------------------------------


def _sample_partials() -> list[dict[str, float]]:
    return [
        {"ratio": 1.0, "amp": 1.0},
        {"ratio": 2.0, "amp": 0.5},
        {"ratio": 3.0, "amp": 0.33},
        {"ratio": 4.0, "amp": 0.25},
        {"ratio": 5.0, "amp": 0.20},
        {"ratio": 6.0, "amp": 0.167},
        {"ratio": 7.0, "amp": 0.143},
        {"ratio": 8.0, "amp": 0.125},
    ]


class TestInharmonicScale:
    def test_zero_amount_is_identity(self) -> None:
        partials = _sample_partials()
        result = apply_inharmonic_scale(partials, amount=0.0)

        assert len(result) == len(partials)
        for source, morphed in zip(partials, result, strict=True):
            assert morphed["ratio"] == pytest.approx(source["ratio"])
            assert morphed["amp"] == pytest.approx(source["amp"])

    def test_positive_amount_stretches_upper_partials(self) -> None:
        partials = _sample_partials()
        result = apply_inharmonic_scale(partials, amount=0.3)

        # Fundamental stays put.
        assert result[0]["ratio"] == pytest.approx(partials[0]["ratio"])
        # Upper partials shift upward (positive amount stretches).
        for source, morphed in zip(partials[1:], result[1:], strict=True):
            assert morphed["ratio"] > source["ratio"]

        # Higher partials shift more than lower partials.
        stretch_ratios = [
            morphed["ratio"] / source["ratio"]
            for source, morphed in zip(partials[1:], result[1:], strict=True)
        ]
        for i in range(1, len(stretch_ratios)):
            assert stretch_ratios[i] >= stretch_ratios[i - 1]

    def test_negative_amount_compresses_upper_partials(self) -> None:
        partials = _sample_partials()
        result = apply_inharmonic_scale(partials, amount=-0.2)

        for source, morphed in zip(partials[1:], result[1:], strict=True):
            assert morphed["ratio"] < source["ratio"]

    def test_preserves_amplitudes(self) -> None:
        partials = _sample_partials()
        result = apply_inharmonic_scale(partials, amount=0.4)

        for source, morphed in zip(partials, result, strict=True):
            assert morphed["amp"] == pytest.approx(source["amp"])

    def test_output_ratios_finite_and_positive(self) -> None:
        partials = _sample_partials()
        result = apply_inharmonic_scale(partials, amount=0.8)

        for entry in result:
            assert np.isfinite(entry["ratio"])
            assert entry["ratio"] > 0.0
            assert np.isfinite(entry["amp"])


class TestPhaseDisperse:
    def test_zero_amount_produces_zero_phase(self) -> None:
        partials = _sample_partials()
        result = apply_phase_disperse(partials, amount=0.0)

        for entry in result:
            assert entry.get("phase", 0.0) == pytest.approx(0.0)

    def test_positive_amount_produces_bounded_nonzero_phase(self) -> None:
        partials = _sample_partials()
        result = apply_phase_disperse(partials, amount=0.01, center_k=4)

        phases = np.asarray([entry["phase"] for entry in result])
        # Phases are bounded in [-2pi, 2pi] (sin output times 2pi).
        assert np.all(np.isfinite(phases))
        assert np.max(np.abs(phases)) <= 2.0 * np.pi + 1e-9
        # At least one partial gets a non-trivial phase.
        assert np.max(np.abs(phases)) > 1e-6

    def test_phase_at_center_is_zero(self) -> None:
        partials = _sample_partials()
        center_k = 3
        result = apply_phase_disperse(partials, amount=0.02, center_k=center_k)

        # ``center_k`` is the 1-indexed partial number, so the 0-indexed
        # position ``center_k - 1`` sits exactly at the quadratic's vertex.
        assert result[center_k - 1]["phase"] == pytest.approx(0.0)

    def test_amplitudes_and_ratios_untouched(self) -> None:
        partials = _sample_partials()
        result = apply_phase_disperse(partials, amount=0.02)

        for source, morphed in zip(partials, result, strict=True):
            assert morphed["ratio"] == pytest.approx(source["ratio"])
            assert morphed["amp"] == pytest.approx(source["amp"])


class TestSmear:
    def test_zero_amount_is_identity(self) -> None:
        partials = _sample_partials()
        result = apply_smear(partials, amount=0.0)

        for source, morphed in zip(partials, result, strict=True):
            assert morphed["amp"] == pytest.approx(source["amp"])
            assert morphed["ratio"] == pytest.approx(source["ratio"])

    def test_positive_amount_raises_upper_partial_amps(self) -> None:
        # With a natural rolloff, upper partials should be boosted by the spread.
        partials = _sample_partials()
        result = apply_smear(partials, amount=0.6)

        for source, morphed in zip(partials[1:], result[1:], strict=True):
            assert morphed["amp"] >= source["amp"] - 1e-12

    def test_full_amount_produces_finite_amps(self) -> None:
        partials = _sample_partials()
        result = apply_smear(partials, amount=1.0)

        amps = np.asarray([entry["amp"] for entry in result])
        assert np.all(np.isfinite(amps))
        assert np.all(amps >= 0.0)
        # The cumulative spread should raise the overall energy moderately.
        original = np.asarray([entry["amp"] for entry in partials])
        assert np.sum(amps) >= np.sum(original) * 0.8

    def test_preserves_ratios(self) -> None:
        partials = _sample_partials()
        result = apply_smear(partials, amount=0.5)

        for source, morphed in zip(partials, result, strict=True):
            assert morphed["ratio"] == pytest.approx(source["ratio"])


class TestShepard:
    def test_zero_amount_is_identity(self) -> None:
        partials = _sample_partials()
        result = apply_shepard(partials, amount=0.0, shift=1.0)

        for source, morphed in zip(partials, result, strict=True):
            assert morphed["amp"] == pytest.approx(source["amp"])

    def test_zero_shift_is_identity(self) -> None:
        partials = _sample_partials()
        result = apply_shepard(partials, amount=0.5, shift=0.0)

        for source, morphed in zip(partials, result, strict=True):
            assert morphed["amp"] == pytest.approx(source["amp"])

    def test_octave_shift_crossfades_amplitudes(self) -> None:
        # With shift=1.0 and amount=0.5 we blend partial k with partial 2k.
        partials = _sample_partials()
        result = apply_shepard(partials, amount=0.5, shift=1.0)

        # Partial 1 (ratio=1) should blend toward partial at ratio=2 (index 1).
        expected_blend = 0.5 * partials[0]["amp"] + 0.5 * partials[1]["amp"]
        assert result[0]["amp"] == pytest.approx(expected_blend, rel=0.05)

    def test_preserves_ratios_and_finite(self) -> None:
        partials = _sample_partials()
        result = apply_shepard(partials, amount=0.7, shift=1.0)

        for source, morphed in zip(partials, result, strict=True):
            assert morphed["ratio"] == pytest.approx(source["ratio"])
            assert np.isfinite(morphed["amp"])
            assert morphed["amp"] >= 0.0


class TestRandomAmplitudes:
    def test_zero_amount_is_identity(self) -> None:
        partials = _sample_partials()
        result = apply_random_amplitudes(partials, amount=0.0, shift=0.25, seed=42)

        for source, morphed in zip(partials, result, strict=True):
            assert morphed["amp"] == pytest.approx(source["amp"])

    def test_deterministic_under_fixed_seed(self) -> None:
        partials = _sample_partials()
        first = apply_random_amplitudes(partials, amount=0.6, shift=0.3, seed=7)
        second = apply_random_amplitudes(partials, amount=0.6, shift=0.3, seed=7)

        amps_first = [entry["amp"] for entry in first]
        amps_second = [entry["amp"] for entry in second]
        assert amps_first == pytest.approx(amps_second)

    def test_different_seeds_give_different_masks(self) -> None:
        partials = _sample_partials()
        first = apply_random_amplitudes(partials, amount=0.8, shift=0.4, seed=1)
        second = apply_random_amplitudes(partials, amount=0.8, shift=0.4, seed=2)

        amps_first = np.asarray([entry["amp"] for entry in first])
        amps_second = np.asarray([entry["amp"] for entry in second])
        assert not np.allclose(amps_first, amps_second)

    def test_moving_shift_changes_output(self) -> None:
        partials = _sample_partials()
        first = apply_random_amplitudes(partials, amount=0.8, shift=0.1, seed=7)
        second = apply_random_amplitudes(partials, amount=0.8, shift=0.9, seed=7)

        amps_first = np.asarray([entry["amp"] for entry in first])
        amps_second = np.asarray([entry["amp"] for entry in second])
        assert not np.allclose(amps_first, amps_second)

    def test_output_bounded_and_finite(self) -> None:
        partials = _sample_partials()
        result = apply_random_amplitudes(partials, amount=1.0, shift=0.5, seed=3)

        for source, morphed in zip(partials, result, strict=True):
            assert np.isfinite(morphed["amp"])
            assert morphed["amp"] >= 0.0
            # At amount=1.0 the random mask ranges in [0, 1] so output <= source.
            assert morphed["amp"] <= source["amp"] + 1e-9


class TestSigmaApproximation:
    def test_single_partial_applies_sinc_one_half(self) -> None:
        """Single partial receives the Lanczos sigma = sinc(1 / (max_k + 1)).

        For one partial, max_k = 1 and k = 1, so the factor is sinc(1 / 2).
        This pins down the actual numeric behavior instead of merely checking
        the output is finite and positive (which any passthrough would satisfy).
        """
        partials = [{"ratio": 1.0, "amp": 1.0}]
        result = apply_sigma_approximation(partials)

        assert len(result) == 1
        assert result[0]["ratio"] == pytest.approx(partials[0]["ratio"])
        expected_amp = float(partials[0]["amp"] * np.sinc(1.0 / 2.0))
        assert result[0]["amp"] == pytest.approx(expected_amp)

    def test_upper_partials_attenuated_more(self) -> None:
        partials = _sample_partials()
        result = apply_sigma_approximation(partials)

        # Lower partials should be attenuated less than higher ones.
        ratios_amp = [
            morphed["amp"] / source["amp"]
            for source, morphed in zip(partials, result, strict=True)
        ]
        # The attenuation factor should be monotonically decreasing.
        for i in range(1, len(ratios_amp)):
            assert ratios_amp[i] <= ratios_amp[i - 1] + 1e-9

    def test_preserves_ratios(self) -> None:
        partials = _sample_partials()
        result = apply_sigma_approximation(partials)

        for source, morphed in zip(partials, result, strict=True):
            assert morphed["ratio"] == pytest.approx(source["ratio"])


# ---------------------------------------------------------------------------
# Dispatcher tests
# ---------------------------------------------------------------------------


class TestDispatcher:
    def test_none_type_is_identity(self) -> None:
        partials = _sample_partials()
        result = apply_spectral_morph(
            partials,
            morph_type="none",
            amount=0.5,
        )

        for source, morphed in zip(partials, result, strict=True):
            assert morphed["amp"] == pytest.approx(source["amp"])
            assert morphed["ratio"] == pytest.approx(source["ratio"])

    def test_unknown_type_raises(self) -> None:
        partials = _sample_partials()
        with pytest.raises(ValueError, match="Unsupported spectral_morph_type"):
            apply_spectral_morph(partials, morph_type="nonsense", amount=0.5)

    def test_all_supported_types_succeed(self) -> None:
        partials = _sample_partials()
        for morph_type in MORPH_TYPES:
            if morph_type == "none":
                continue
            result = apply_spectral_morph(
                partials,
                morph_type=morph_type,
                amount=0.3,
                shift=0.5,
                center_k=24,
                seed=1,
            )
            assert len(result) == len(partials)
            for entry in result:
                assert np.isfinite(entry["ratio"])
                assert np.isfinite(entry["amp"])


# ---------------------------------------------------------------------------
# Engine integration tests
# ---------------------------------------------------------------------------


def _partials_for_integration() -> list[dict[str, float]]:
    return ratio_spectrum(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [1.0, 0.5, 0.33, 0.25, 0.2, 0.17, 0.14, 0.12],
    )


def test_engine_no_morph_is_backward_compatible() -> None:
    base_params = {
        "engine": "additive",
        "partials": _partials_for_integration(),
    }
    baseline = render_note_signal(
        freq=220.0, duration=0.5, amp=0.25, sample_rate=44100, params=base_params
    )
    with_off = render_note_signal(
        freq=220.0,
        duration=0.5,
        amp=0.25,
        sample_rate=44100,
        params={**base_params, "spectral_morph_type": "none"},
    )
    assert np.allclose(baseline, with_off)


def test_engine_zero_amount_is_backward_compatible() -> None:
    base_params = {
        "engine": "additive",
        "partials": _partials_for_integration(),
    }
    baseline = render_note_signal(
        freq=220.0, duration=0.5, amp=0.25, sample_rate=44100, params=base_params
    )
    with_zero = render_note_signal(
        freq=220.0,
        duration=0.5,
        amp=0.25,
        sample_rate=44100,
        params={
            **base_params,
            "spectral_morph_type": "inharmonic_scale",
            "spectral_morph_amount": 0.0,
        },
    )
    assert np.allclose(baseline, with_zero)


@pytest.mark.parametrize(
    "morph_type",
    ["inharmonic_scale", "phase_disperse", "smear", "shepard", "random_amplitudes"],
)
def test_engine_each_morph_renders_bounded_audio(morph_type: str) -> None:
    signal = render_note_signal(
        freq=220.0,
        duration=0.5,
        amp=0.25,
        sample_rate=44100,
        params={
            "engine": "additive",
            "partials": _partials_for_integration(),
            "spectral_morph_type": morph_type,
            "spectral_morph_amount": 0.4,
            "spectral_morph_shift": 0.5,
            "spectral_morph_seed": 17,
            "spectral_morph_center_k": 6,
        },
    )
    assert signal.size > 0
    assert np.all(np.isfinite(signal))
    # Audio is normalized to amp, so peak should be close to amp.
    peak = float(np.max(np.abs(signal)))
    assert peak <= 0.25 + 1e-6
    assert peak > 0.0


@pytest.mark.parametrize(
    "morph_type",
    ["inharmonic_scale", "phase_disperse", "smear", "shepard", "random_amplitudes"],
)
def test_engine_morph_changes_output_from_baseline(morph_type: str) -> None:
    base_params = {
        "engine": "additive",
        "partials": _partials_for_integration(),
    }
    baseline = render_note_signal(
        freq=220.0, duration=0.5, amp=0.25, sample_rate=44100, params=base_params
    )
    morphed = render_note_signal(
        freq=220.0,
        duration=0.5,
        amp=0.25,
        sample_rate=44100,
        params={
            **base_params,
            "spectral_morph_type": morph_type,
            "spectral_morph_amount": 0.45,
            "spectral_morph_shift": 0.6,
            "spectral_morph_seed": 23,
            "spectral_morph_center_k": 4,
        },
    )
    assert not np.allclose(baseline, morphed)


def test_engine_random_amplitudes_is_deterministic() -> None:
    params = {
        "engine": "additive",
        "partials": _partials_for_integration(),
        "spectral_morph_type": "random_amplitudes",
        "spectral_morph_amount": 0.8,
        "spectral_morph_shift": 0.5,
        "spectral_morph_seed": 9,
    }
    first = render_note_signal(
        freq=220.0, duration=0.5, amp=0.25, sample_rate=44100, params=params
    )
    second = render_note_signal(
        freq=220.0, duration=0.5, amp=0.25, sample_rate=44100, params=params
    )
    assert np.allclose(first, second)


def test_engine_sigma_approximation_softens_upper_partial_energy() -> None:
    base_params = {
        "engine": "additive",
        "partials": _partials_for_integration(),
    }
    without = render_note_signal(
        freq=220.0, duration=0.5, amp=0.25, sample_rate=44100, params=base_params
    )
    with_sigma = render_note_signal(
        freq=220.0,
        duration=0.5,
        amp=0.25,
        sample_rate=44100,
        params={**base_params, "sigma_approximation": True},
    )

    # Compute upper-band energy before/after — sigma should attenuate upper
    # partials more than the fundamental.
    def _upper_energy(signal: np.ndarray) -> float:
        spectrum = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(signal.size, d=1.0 / 44100)
        # 220 Hz fundamental; integrate energy above ratio 3 (660 Hz) up to 4 kHz.
        mask = (freqs >= 660.0) & (freqs <= 4000.0)
        return float(np.sum(spectrum[mask]))

    assert _upper_energy(with_sigma) < _upper_energy(without)


def test_engine_phase_disperse_changes_waveform_not_spectrum_much() -> None:
    base_params = {
        "engine": "additive",
        "partials": _partials_for_integration(),
    }
    baseline = render_note_signal(
        freq=220.0, duration=0.5, amp=0.25, sample_rate=44100, params=base_params
    )
    phase_morphed = render_note_signal(
        freq=220.0,
        duration=0.5,
        amp=0.25,
        sample_rate=44100,
        params={
            **base_params,
            "spectral_morph_type": "phase_disperse",
            "spectral_morph_amount": 0.03,
            "spectral_morph_center_k": 4,
        },
    )
    # Waveforms should differ clearly.
    assert not np.allclose(baseline, phase_morphed)

    # But the amplitude spectrum magnitudes should be similar at DC-adjacent
    # partial bins (phase shifts don't change magnitude spectrum).
    spec_baseline = np.abs(np.fft.rfft(baseline))
    spec_morphed = np.abs(np.fft.rfft(phase_morphed))
    # Compare total energy: phase shifting shouldn't dramatically change it
    # (though normalization to peak can).
    baseline_energy = float(np.sum(spec_baseline**2))
    morphed_energy = float(np.sum(spec_morphed**2))
    ratio = morphed_energy / max(baseline_energy, 1e-12)
    # Loose bound: within 2x in either direction.
    assert 0.4 < ratio < 2.5


# ---------------------------------------------------------------------------
# Score-level end-to-end smoke
# ---------------------------------------------------------------------------


def test_spectral_morph_through_score_render_produces_audio() -> None:
    """End-to-end: a Score voice with a spectral_morph_type renders clean audio.

    Ensures the additive-engine morph params flow through synth_defaults,
    per-note engine dispatch, normalization, and the Score render path without
    corrupting the signal (NaN/Inf) or collapsing to silence.
    """
    score = Score(f0_hz=220.0, auto_master_gain_stage=False)
    score.add_voice(
        "morph_pad",
        synth_defaults={
            "engine": "additive",
            "partials": [
                {"ratio": 1.0, "amp": 1.0},
                {"ratio": 2.0, "amp": 0.5},
                {"ratio": 3.0, "amp": 0.33},
                {"ratio": 4.0, "amp": 0.25},
                {"ratio": 5.0, "amp": 0.2},
            ],
            "spectral_morph_type": "inharmonic_scale",
            "spectral_morph_amount": 0.3,
            "sigma_approximation": True,
            "attack": 0.02,
            "release": 0.1,
        },
        velocity_humanize=None,
    )
    score.add_note("morph_pad", start=0.0, duration=0.5, partial=1.0, amp=0.3)
    audio = score.render()
    # Score.render() returns 1-D mono for a single un-panned voice; (2, N)
    # stereo when pan or stereo effects are present. Accept either.
    assert audio.ndim in (1, 2), (
        f"expected 1-D or (2, N) output, got shape {audio.shape}"
    )
    assert np.all(np.isfinite(audio))
    peak = float(np.max(np.abs(audio)))
    assert peak > 1e-3, f"rendered audio should not be silent; peak={peak}"
    assert peak < 1.5, f"rendered audio peak {peak:.3f} is too hot"
