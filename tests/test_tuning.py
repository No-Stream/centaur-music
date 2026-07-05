"""Tuning helper tests."""

import math

import pytest

from code_musics.tuning import (
    TuningTable,
    cents_to_ratio,
    cps,
    dekany,
    dekany_chords,
    edo_scale,
    eikosany,
    eikosany_tetrads,
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


class TestDekany:
    """Verify dekany() default normalization and known ratio set."""

    def test_default_dekany_ratios(self) -> None:
        """2-of-5 CPS on (1, 3, 5, 7, 9) normalized by 3."""
        result = dekany()
        expected = sorted(
            [1.0, 9 / 8, 7 / 6, 5 / 4, 21 / 16, 35 / 24, 3 / 2, 5 / 3, 7 / 4, 15 / 8]
        )
        assert len(result) == 10
        for actual, exp in zip(result, expected, strict=True):
            assert actual == pytest.approx(exp)

    def test_dekany_contains_the_default_hexany(self) -> None:
        """Dropping the 9 leaves the classic 1-3-5-7 hexany embedded exactly."""
        dekany_notes = dekany()
        for hexany_note in hexany():
            assert any(hexany_note == pytest.approx(note) for note in dekany_notes), (
                hexany_note
            )

    def test_dekany_requires_exactly_five_factors(self) -> None:
        with pytest.raises(ValueError, match="5 factors"):
            dekany((1, 3, 5, 7))

    def test_dekany_custom_normalize(self) -> None:
        result_default = dekany((1, 3, 5, 7, 9))
        result_explicit = dekany((1, 3, 5, 7, 9), normalize=3.0)
        assert result_default == pytest.approx(result_explicit)


class TestDekanyChords:
    """Verify dekany_chords() otonal tetrad / utonal triad structure."""

    def test_returns_five_otonal_tetrads_and_ten_utonal_triads(self) -> None:
        otonal_tetrads, utonal_triads = dekany_chords()
        assert len(otonal_tetrads) == 5
        assert len(utonal_triads) == 10

    def test_all_chord_members_are_dekany_notes(self) -> None:
        notes = dekany()
        otonal_tetrads, utonal_triads = dekany_chords()
        for chord in list(otonal_tetrads) + list(utonal_triads):
            for member in chord:
                assert any(member == pytest.approx(note) for note in notes), member

    def test_known_otonal_tetrad_sharing_factor_nine(self) -> None:
        """Sharing x=9 over (1, 3, 5, 7, 9): the classic 1:3:5:7 harmonic tetrad.

        Products {9, 27, 45, 63} / 3 = {3, 9, 15, 21}, octave-reduced:
        9/8, 21/16, 3/2, 15/8 — proportional to 6:7:8:10, which is 1:3:5:7
        re-voiced into one octave.
        """
        otonal_tetrads, _ = dekany_chords()
        expected = (9 / 8, 21 / 16, 3 / 2, 15 / 8)
        assert any(tetrad == pytest.approx(expected) for tetrad in otonal_tetrads), (
            otonal_tetrads
        )

    def test_known_utonal_triad_over_three_five_seven(self) -> None:
        """Subset {3, 5, 7}: pair products {15, 21, 35} / 3 -> 5/4, 7/4, 35/24."""
        _, utonal_triads = dekany_chords()
        expected = (5 / 4, 35 / 24, 7 / 4)
        assert any(triad == pytest.approx(expected) for triad in utonal_triads), (
            utonal_triads
        )

    def test_chords_are_sorted_ascending(self) -> None:
        otonal_tetrads, utonal_triads = dekany_chords()
        for chord in list(otonal_tetrads) + list(utonal_triads):
            assert list(chord) == sorted(chord)

    def test_tetrad_and_triad_sizes(self) -> None:
        otonal_tetrads, utonal_triads = dekany_chords()
        for tetrad in otonal_tetrads:
            assert len(tetrad) == 4
        for triad in utonal_triads:
            assert len(triad) == 3


class TestEikosany:
    """Verify eikosany() default normalization and known ratio set."""

    def test_default_eikosany_ratios(self) -> None:
        """3-of-6 CPS on (1, 3, 5, 7, 9, 11) normalized by 1*3*5 = 15."""
        result = eikosany()
        expected = sorted(
            [
                1.0,
                33 / 32,
                21 / 20,
                11 / 10,
                9 / 8,
                7 / 6,
                99 / 80,
                77 / 60,
                21 / 16,
                11 / 8,
                7 / 5,
                231 / 160,
                3 / 2,
                63 / 40,
                77 / 48,
                33 / 20,
                7 / 4,
                9 / 5,
                11 / 6,
                77 / 40,
            ]
        )
        assert len(result) == 20
        for actual, exp in zip(result, expected, strict=True):
            assert actual == pytest.approx(exp)

    def test_eikosany_requires_exactly_six_factors(self) -> None:
        with pytest.raises(ValueError, match="6 factors"):
            eikosany((1, 3, 5, 7, 9))

    def test_eikosany_custom_normalize(self) -> None:
        result_default = eikosany((1, 3, 5, 7, 9, 11))
        result_explicit = eikosany((1, 3, 5, 7, 9, 11), normalize=15.0)
        assert result_default == pytest.approx(result_explicit)


class TestEikosanyTetrads:
    """Verify eikosany_tetrads() otonal/utonal tetrad structure."""

    def test_returns_fifteen_otonal_and_fifteen_utonal_tetrads(self) -> None:
        otonal_tetrads, utonal_tetrads = eikosany_tetrads()
        assert len(otonal_tetrads) == 15
        assert len(utonal_tetrads) == 15

    def test_all_tetrad_members_are_eikosany_notes(self) -> None:
        notes = eikosany()
        otonal_tetrads, utonal_tetrads = eikosany_tetrads()
        for tetrad in list(otonal_tetrads) + list(utonal_tetrads):
            assert len(tetrad) == 4
            for member in tetrad:
                assert any(member == pytest.approx(note) for note in notes), member

    def test_known_otonal_tetrad_for_pair_nine_eleven(self) -> None:
        """Pair {9, 11}: sounds as the otonal chord of {1, 3, 5, 7}."""
        otonal_tetrads, _ = eikosany_tetrads()
        expected = (33 / 32, 99 / 80, 231 / 160, 33 / 20)
        assert any(tetrad == pytest.approx(expected) for tetrad in otonal_tetrads), (
            otonal_tetrads
        )

    def test_known_otonal_tetrad_for_pair_one_three(self) -> None:
        """Pair {1, 3}: sounds as the otonal chord of {5, 7, 9, 11}."""
        otonal_tetrads, _ = eikosany_tetrads()
        expected = (1.0, 11 / 10, 7 / 5, 9 / 5)
        assert any(tetrad == pytest.approx(expected) for tetrad in otonal_tetrads), (
            otonal_tetrads
        )

    def test_known_utonal_tetrad_for_subset_one_three_five_nine(self) -> None:
        otonal_tetrads, utonal_tetrads = eikosany_tetrads()
        expected = (1.0, 9 / 8, 3 / 2, 9 / 5)
        assert any(tetrad == pytest.approx(expected) for tetrad in utonal_tetrads), (
            utonal_tetrads
        )

    def test_known_utonal_tetrad_for_subset_five_seven_nine_eleven(self) -> None:
        _, utonal_tetrads = eikosany_tetrads()
        expected = (33 / 32, 21 / 16, 231 / 160, 77 / 48)
        assert any(tetrad == pytest.approx(expected) for tetrad in utonal_tetrads), (
            utonal_tetrads
        )

    def test_tetrads_are_sorted_ascending(self) -> None:
        otonal_tetrads, utonal_tetrads = eikosany_tetrads()
        for tetrad in list(otonal_tetrads) + list(utonal_tetrads):
            assert list(tetrad) == sorted(tetrad)

    def test_eikosany_tetrads_requires_exactly_six_factors(self) -> None:
        with pytest.raises(ValueError, match="6 factors"):
            eikosany_tetrads((1, 3, 5, 7, 9))
