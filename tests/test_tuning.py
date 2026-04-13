"""Tuning helper tests."""

import math

import pytest

from code_musics.tuning import (
    TuningTable,
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
