"""Spectrum helper tests."""

from __future__ import annotations

import pytest

from code_musics.spectra import harmonic_spectrum, ratio_spectrum, stretched_spectrum


def test_ratio_spectrum_builds_explicit_partials() -> None:
    spectrum = ratio_spectrum([1.0, 7 / 4, 11 / 8], [1.0, 0.3, 0.1])

    assert spectrum == [
        {"ratio": 1.0, "amp": 1.0},
        {"ratio": pytest.approx(7 / 4), "amp": 0.3},
        {"ratio": pytest.approx(11 / 8), "amp": 0.1},
    ]


def test_ratio_spectrum_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="amps must match"):
        ratio_spectrum([1.0, 2.0], [1.0])


def test_harmonic_spectrum_matches_expected_fundamental_weighting() -> None:
    spectrum = harmonic_spectrum(
        n_partials=4,
        harmonic_rolloff=0.5,
        brightness_tilt=0.0,
        odd_even_balance=0.0,
    )

    assert spectrum == [
        {"ratio": 1.0, "amp": 1.0},
        {"ratio": 2.0, "amp": 0.5},
        {"ratio": 3.0, "amp": 0.25},
        {"ratio": 4.0, "amp": 0.125},
    ]


def test_stretched_spectrum_builds_non_harmonic_ratios() -> None:
    spectrum = stretched_spectrum(
        n_partials=4,
        stretch_exponent=1.1,
        harmonic_rolloff=0.5,
    )

    assert spectrum[0] == {"ratio": 1.0, "amp": 1.0}
    assert spectrum[1]["ratio"] > 2.0
    assert spectrum[2]["ratio"] > 3.0
    assert spectrum[3]["ratio"] > 4.0
