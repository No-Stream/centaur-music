"""Tuning helper tests."""

import math

from code_musics.tuning import (
    cents_to_ratio,
    edo_scale,
    harmonic_series,
    ji_chord,
    otonal,
    ratio_to_cents,
    utonal,
)


def test_harmonic_and_otonal_helpers() -> None:
    assert harmonic_series(55.0, 4) == [55.0, 110.0, 165.0, 220.0]
    assert otonal(55.0, [4, 5, 6, 7]) == [220.0, 275.0, 330.0, 385.0]


def test_utonal_helper() -> None:
    assert utonal(220.0, [1, 2, 4]) == [220.0, 110.0, 55.0]


def test_ji_chord_helper() -> None:
    assert ji_chord(110.0, [1.0, 5 / 4, 3 / 2]) == [110.0, 137.5, 165.0]


def test_cents_and_ratio_round_trip() -> None:
    ratio = cents_to_ratio(702.0)
    cents = ratio_to_cents(ratio)

    assert math.isclose(cents, 702.0, rel_tol=0.0, abs_tol=1e-9)


def test_edo_scale_preserves_octaves() -> None:
    scale = edo_scale(110.0, divisions=12, octaves=2)

    assert len(scale) == 25
    assert math.isclose(scale[12], 220.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(scale[-1], 440.0, rel_tol=0.0, abs_tol=1e-9)
