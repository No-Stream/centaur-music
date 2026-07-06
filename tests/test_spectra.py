"""Spectrum helper tests."""

from __future__ import annotations

import pytest

from code_musics.spectra import (
    harmonic_spectrum,
    ratio_spectrum,
    scale_fused_spectrum,
    stretched_spectrum,
)


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


class TestScaleFusedSpectrum:
    def test_skeleton_yields_integer_partials_without_fifth_harmonic(self) -> None:
        # 3/7-limit skeleton degrees across octaves -> {1,2,3,4,6,7,8,12,14}
        partials = scale_fused_spectrum([1.0, 3 / 2, 7 / 4], octaves=3)
        ratios = sorted(p["ratio"] for p in partials)
        assert ratios == pytest.approx(
            [1.0, 2.0, 3.0, 3.5, 4.0, 6.0, 7.0, 8.0, 12.0, 14.0]
        )
        assert not any(abs(r - 5.0) < 1e-9 or abs(r - 10.0) < 1e-9 for r in ratios)

    def test_color_degrees_yield_noninteger_partials(self) -> None:
        partials = scale_fused_spectrum([1.0, 19 / 16], octaves=2)
        ratios = sorted(p["ratio"] for p in partials)
        assert any(abs(r - 2.375) < 1e-9 for r in ratios)

    def test_rolloff_and_format(self) -> None:
        partials = scale_fused_spectrum([1.0, 3 / 2], octaves=2, rolloff_alpha=1.0)
        assert all(set(p) == {"ratio", "amp"} for p in partials)
        by_ratio = {p["ratio"]: p["amp"] for p in partials}
        assert by_ratio[1.0] == pytest.approx(1.0)
        assert by_ratio[3.0] == pytest.approx(1.0 / 3.0)

    def test_amp_floor_drops_weak_partials(self) -> None:
        partials = scale_fused_spectrum(
            [1.0], octaves=8, rolloff_alpha=2.0, amp_floor=0.02
        )
        assert all(p["amp"] >= 0.02 for p in partials)

    def test_rejects_empty_or_nonpositive(self) -> None:
        with pytest.raises(ValueError, match="degrees"):
            scale_fused_spectrum([])
        with pytest.raises(ValueError, match="positive"):
            scale_fused_spectrum([0.0])
