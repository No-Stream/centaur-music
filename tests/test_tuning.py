"""Tuning helper tests."""

import math

import pytest

from code_musics.tuning import (
    TuningTable,
    cents_to_ratio,
    cps,
    edo_scale,
    harmonic_series,
    hexany,
    hexany_triads,
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


# --- TuningTable tests ---


class TestTuningTableOctavePlacement:
    """Verify ratio_for handles octave displacement correctly."""

    def test_unison_at_root(self) -> None:
        table = TuningTable.five_limit_major()
        assert table.ratio_for(60, root_midi_note=60) == 1.0

    def test_octave_above(self) -> None:
        table = TuningTable.five_limit_major()
        assert table.ratio_for(72, root_midi_note=60) == 2.0

    def test_octave_below(self) -> None:
        table = TuningTable.five_limit_major()
        assert table.ratio_for(48, root_midi_note=60) == 0.5

    def test_note_below_root(self) -> None:
        """B3 (midi 59) is one semitone below root C4 (midi 60) -- should be 15/8 * 0.5."""
        table = TuningTable.five_limit_major()
        expected = (15 / 8) * 0.5
        assert math.isclose(
            table.ratio_for(59, root_midi_note=60), expected, rel_tol=1e-12
        )


class TestTuningTablePresetRatios:
    """Verify key intervals in each preset."""

    def test_five_limit_perfect_fifth(self) -> None:
        table = TuningTable.five_limit_major()
        assert math.isclose(table.ratios[7], 3 / 2, rel_tol=1e-12)

    def test_five_limit_major_third(self) -> None:
        table = TuningTable.five_limit_major()
        assert math.isclose(table.ratios[4], 5 / 4, rel_tol=1e-12)

    def test_five_limit_minor_third(self) -> None:
        table = TuningTable.five_limit_major()
        assert math.isclose(table.ratios[3], 6 / 5, rel_tol=1e-12)

    def test_seven_limit_minor_seventh(self) -> None:
        table = TuningTable.seven_limit()
        assert math.isclose(table.ratios[10], 7 / 4, rel_tol=1e-12)

    def test_seven_limit_tritone(self) -> None:
        table = TuningTable.seven_limit()
        assert math.isclose(table.ratios[6], 7 / 5, rel_tol=1e-12)

    def test_pythagorean_major_third(self) -> None:
        table = TuningTable.pythagorean()
        assert math.isclose(table.ratios[4], 81 / 64, rel_tol=1e-12)

    def test_pythagorean_perfect_fifth(self) -> None:
        table = TuningTable.pythagorean()
        assert math.isclose(table.ratios[7], 3 / 2, rel_tol=1e-12)


class TestTuningTableResolve:
    """Verify resolve() == f0 * ratio_for() for multiple notes."""

    def test_resolve_consistency(self) -> None:
        table = TuningTable.five_limit_major()
        f0 = 261.63
        for midi_note in [48, 55, 60, 64, 67, 72, 76, 84]:
            resolved = table.resolve(midi_note, f0, root_midi_note=60)
            expected = f0 * table.ratio_for(midi_note, root_midi_note=60)
            assert math.isclose(resolved, expected, rel_tol=1e-12), (
                f"resolve mismatch for midi_note={midi_note}"
            )


class TestTuningTableLabelFor:
    """Verify label_for returns correct label strings."""

    def test_unison_label(self) -> None:
        table = TuningTable.five_limit_major()
        assert table.label_for(60, root_midi_note=60) == "1/1"

    def test_fifth_label(self) -> None:
        table = TuningTable.five_limit_major()
        assert table.label_for(67, root_midi_note=60) == "3/2"

    def test_septimal_seventh_label(self) -> None:
        table = TuningTable.seven_limit()
        assert table.label_for(70, root_midi_note=60) == "7/4"


class TestTuningTableDescribe:
    """Verify describe() returns a readable multi-line mapping."""

    def test_contains_name(self) -> None:
        table = TuningTable.five_limit_major()
        desc = table.describe()
        assert "5-limit major" in desc

    def test_contains_ratio_labels(self) -> None:
        table = TuningTable.five_limit_major()
        desc = table.describe()
        assert "3/2" in desc
        assert "5/4" in desc
        assert "1/1" in desc

    def test_multiline(self) -> None:
        table = TuningTable.five_limit_major()
        desc = table.describe()
        lines = desc.strip().split("\n")
        assert len(lines) == 13  # header + 12 pitch classes


class TestTuningTableValidation:
    """Verify construction fails with wrong number of ratios or labels."""

    def test_wrong_number_of_ratios(self) -> None:
        with pytest.raises(ValueError, match="exactly 12 ratios"):
            TuningTable(
                ratios=(1.0, 2.0, 3.0),
                labels=("a",) * 12,
                name="bad",
            )

    def test_wrong_number_of_labels(self) -> None:
        with pytest.raises(ValueError, match="exactly 12 labels"):
            TuningTable(
                ratios=(1.0,) * 12,
                labels=("a", "b"),
                name="bad",
            )

    def test_zero_ratio_rejected(self) -> None:
        ratios = list(TuningTable.five_limit_major().ratios)
        ratios[3] = 0.0
        with pytest.raises(ValueError, match="positive"):
            TuningTable(ratios=tuple(ratios), labels=("x",) * 12, name="bad")

    def test_negative_ratio_rejected(self) -> None:
        ratios = list(TuningTable.five_limit_major().ratios)
        ratios[5] = -4 / 3
        with pytest.raises(ValueError, match="positive"):
            TuningTable(ratios=tuple(ratios), labels=("x",) * 12, name="bad")


class TestTuningTableDifferentRoot:
    """Verify behavior with a non-default root_midi_note."""

    def test_root_a3_unison(self) -> None:
        table = TuningTable.five_limit_major()
        f0 = 220.0
        root = 57  # A3
        freq = table.resolve(57, f0, root_midi_note=root)
        assert math.isclose(freq, 220.0, rel_tol=1e-12)

    def test_root_a3_fifth_above(self) -> None:
        """E4 (midi 64) is 7 semitones above A3 (midi 57) -- perfect fifth = 3/2."""
        table = TuningTable.five_limit_major()
        f0 = 220.0
        root = 57  # A3
        freq = table.resolve(64, f0, root_midi_note=root)
        assert math.isclose(freq, 330.0, rel_tol=1e-12)


# --- Combination Product Set (CPS) / hexany tests ---


class TestCps:
    """Verify cps() combination products, octave reduction, and validation."""

    def test_hexany_via_cps_matches_known_ratios(self) -> None:
        """2-out-of-4 CPS on (1, 3, 5, 7) normalized by 3 is the classic hexany."""
        result = cps((1, 3, 5, 7), 2, normalize=3.0)
        expected = sorted({1.0, 7 / 6, 5 / 4, 35 / 24, 5 / 3, 7 / 4})
        assert len(result) == len(expected)
        for actual, exp in zip(result, expected, strict=True):
            assert actual == pytest.approx(exp)

    def test_results_are_octave_reduced(self) -> None:
        result = cps((1, 3, 5, 7), 2, normalize=3.0)
        for ratio in result:
            assert 1.0 <= ratio < 2.0

    def test_results_sorted_ascending_and_deduplicated(self) -> None:
        result = cps((1, 3, 5, 7), 2, normalize=3.0)
        assert result == sorted(result)
        assert len(result) == len(set(result))

    def test_choose_one_returns_octave_reduced_factors(self) -> None:
        result = cps((1, 3, 5, 7), 1, normalize=1.0)
        expected = sorted({1.0, 3 / 2, 5 / 4, 7 / 4})
        assert len(result) == len(expected)
        for actual, exp in zip(result, expected, strict=True):
            assert actual == pytest.approx(exp)

    def test_choose_all_returns_single_product(self) -> None:
        result = cps((1, 3, 5), 3, normalize=1.0)
        assert len(result) == 1
        assert result[0] == pytest.approx(15 / 8)

    def test_empty_factors_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            cps((), 1)

    def test_non_positive_factor_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            cps((1, -3, 5), 2)

    def test_non_integer_factor_rejected(self) -> None:
        with pytest.raises(ValueError, match="int"):
            cps((1, 3.5, 5), 2)  # type: ignore[arg-type]

    def test_duplicate_factor_rejected(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            cps((1, 3, 3, 5), 2)

    def test_choose_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="choose"):
            cps((1, 3, 5, 7), 0)

    def test_choose_too_large_rejected(self) -> None:
        with pytest.raises(ValueError, match="choose"):
            cps((1, 3, 5, 7), 5)

    def test_non_positive_normalize_rejected(self) -> None:
        with pytest.raises(ValueError, match="normalize"):
            cps((1, 3, 5, 7), 2, normalize=0.0)


class TestHexany:
    """Verify hexany() default normalization and known ratio set."""

    def test_default_hexany_ratios(self) -> None:
        result = hexany()
        expected = sorted([1.0, 7 / 6, 5 / 4, 35 / 24, 5 / 3, 7 / 4])
        assert len(result) == 6
        for actual, exp in zip(result, expected, strict=True):
            assert actual == pytest.approx(exp)

    def test_hexany_requires_exactly_four_factors(self) -> None:
        with pytest.raises(ValueError, match="4 factors"):
            hexany((1, 3, 5))

    def test_hexany_custom_normalize(self) -> None:
        result_default = hexany((1, 3, 5, 7))
        result_explicit = hexany((1, 3, 5, 7), normalize=3.0)
        assert result_default == pytest.approx(result_explicit)


class TestHexanyTriads:
    """Verify hexany_triads() otonal/utonal triad structure."""

    def test_returns_four_otonal_and_four_utonal_triads(self) -> None:
        otonal_triads, utonal_triads = hexany_triads()
        assert len(otonal_triads) == 4
        assert len(utonal_triads) == 4

    def test_all_triad_members_are_hexany_notes(self) -> None:
        notes = set(hexany())
        otonal_triads, utonal_triads = hexany_triads()
        for triad in otonal_triads + utonal_triads:
            assert len(triad) == 3
            for member in triad:
                assert any(member == pytest.approx(note) for note in notes)

    def test_known_otonal_triad_fixing_factor_one(self) -> None:
        """Fixing x=1 over (1, 3, 5, 7): products {3, 5, 7}/3 octave-reduced and sorted."""
        otonal_triads, _ = hexany_triads()
        expected = (1.0, 7 / 6, 5 / 3)
        assert any(triad == pytest.approx(expected) for triad in otonal_triads), (
            otonal_triads
        )

    def test_triads_are_sorted_ascending(self) -> None:
        otonal_triads, utonal_triads = hexany_triads()
        for triad in otonal_triads + utonal_triads:
            assert list(triad) == sorted(triad)
