"""Tests for physical model spectra builders and spectral convolution."""

from __future__ import annotations

import math

import pytest

from code_musics.spectra import (
    _BAR_MODES,
    _BOWL_MODES,
    _MEMBRANE_MODES,
    _STOPPED_PIPE_MODES,
    _cents_distance,
    _merge_partials,
    bar_spectrum,
    bowl_spectrum,
    formant_morph,
    formant_shape,
    fractal_spectrum,
    harmonic_spectrum,
    membrane_spectrum,
    plate_spectrum,
    spectral_convolve,
    tube_spectrum,
    vowel_formants,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ratios(spectrum: list[dict[str, float]]) -> list[float]:
    return [p["ratio"] for p in spectrum]


def _amps(spectrum: list[dict[str, float]]) -> list[float]:
    return [p["amp"] for p in spectrum]


# ---------------------------------------------------------------------------
# membrane_spectrum
# ---------------------------------------------------------------------------


class TestMembraneSpectrum:
    def test_returns_valid_partials(self) -> None:
        spectrum = membrane_spectrum()
        assert len(spectrum) == 12
        assert all("ratio" in p and "amp" in p for p in spectrum)

    def test_fundamental_is_one(self) -> None:
        assert membrane_spectrum()[0]["ratio"] == 1.0

    def test_ratios_sorted_ascending(self) -> None:
        ratios = _ratios(membrane_spectrum())
        assert ratios == sorted(ratios)

    def test_ratios_match_reference(self) -> None:
        spectrum = membrane_spectrum(n_modes=len(_MEMBRANE_MODES))
        for p, expected in zip(spectrum, _MEMBRANE_MODES, strict=True):
            assert p["ratio"] == pytest.approx(expected, abs=0.01)

    def test_n_modes_controls_length(self) -> None:
        assert len(membrane_spectrum(n_modes=1)) == 1
        assert len(membrane_spectrum(n_modes=5)) == 5
        assert len(membrane_spectrum(n_modes=16)) == 16

    def test_higher_damping_lower_high_mode_amps(self) -> None:
        low_damp = membrane_spectrum(n_modes=8, damping=0.1)
        high_damp = membrane_spectrum(n_modes=8, damping=1.0)
        assert low_damp[-1]["amp"] > high_damp[-1]["amp"]

    def test_zero_damping_all_amps_one(self) -> None:
        spectrum = membrane_spectrum(n_modes=5, damping=0.0)
        for p in spectrum:
            assert p["amp"] == pytest.approx(1.0)

    def test_rejects_zero_modes(self) -> None:
        with pytest.raises(ValueError, match="n_modes must be at least 1"):
            membrane_spectrum(n_modes=0)

    def test_rejects_negative_damping(self) -> None:
        with pytest.raises(ValueError, match="damping must be non-negative"):
            membrane_spectrum(damping=-0.1)

    def test_rejects_too_many_modes(self) -> None:
        with pytest.raises(ValueError, match="available membrane modes"):
            membrane_spectrum(n_modes=100)


# ---------------------------------------------------------------------------
# bar_spectrum
# ---------------------------------------------------------------------------


class TestBarSpectrum:
    def test_returns_valid_partials(self) -> None:
        spectrum = bar_spectrum()
        assert len(spectrum) == 8
        assert all("ratio" in p and "amp" in p for p in spectrum)

    def test_fundamental_is_one(self) -> None:
        assert bar_spectrum()[0]["ratio"] == 1.0

    def test_ratios_sorted_ascending(self) -> None:
        ratios = _ratios(bar_spectrum())
        assert ratios == sorted(ratios)

    def test_ratios_match_reference(self) -> None:
        spectrum = bar_spectrum(n_modes=len(_BAR_MODES))
        for p, expected in zip(spectrum, _BAR_MODES, strict=True):
            assert p["ratio"] == pytest.approx(expected, abs=0.01)

    def test_n_modes_controls_length(self) -> None:
        assert len(bar_spectrum(n_modes=1)) == 1
        assert len(bar_spectrum(n_modes=4)) == 4

    def test_materials_produce_different_rolloff(self) -> None:
        wood = bar_spectrum(material="wood")
        metal = bar_spectrum(material="metal")
        glass = bar_spectrum(material="glass")

        last_wood = wood[-1]["amp"]
        last_metal = metal[-1]["amp"]
        last_glass = glass[-1]["amp"]

        # metal sustains high modes most, wood least
        assert last_metal > last_glass > last_wood

    def test_rejects_zero_modes(self) -> None:
        with pytest.raises(ValueError, match="n_modes must be at least 1"):
            bar_spectrum(n_modes=0)

    def test_rejects_bad_material(self) -> None:
        with pytest.raises(ValueError, match="material must be one of"):
            bar_spectrum(material="rubber")

    def test_rejects_too_many_modes(self) -> None:
        with pytest.raises(ValueError, match="available bar modes"):
            bar_spectrum(n_modes=100)


# ---------------------------------------------------------------------------
# plate_spectrum
# ---------------------------------------------------------------------------


class TestPlateSpectrum:
    def test_returns_valid_partials(self) -> None:
        spectrum = plate_spectrum()
        assert len(spectrum) == 12
        assert all("ratio" in p and "amp" in p for p in spectrum)

    def test_fundamental_is_one(self) -> None:
        assert plate_spectrum()[0]["ratio"] == pytest.approx(1.0)

    def test_ratios_sorted_ascending(self) -> None:
        ratios = _ratios(plate_spectrum())
        assert ratios == sorted(ratios)

    def test_n_modes_controls_length(self) -> None:
        assert len(plate_spectrum(n_modes=1)) == 1
        assert len(plate_spectrum(n_modes=6)) == 6

    def test_square_vs_rectangular_different_modes(self) -> None:
        square = plate_spectrum(n_modes=8, aspect_ratio=1.0)
        rect = plate_spectrum(n_modes=8, aspect_ratio=2.0)

        square_ratios = _ratios(square)
        rect_ratios = _ratios(rect)

        # Different aspect ratios produce different mode structures
        assert square_ratios != rect_ratios

    def test_square_plate_has_degenerate_modes(self) -> None:
        # A square plate (aspect_ratio=1.0) has (m,n) and (n,m) at the same freq
        # so some ratios will coincide, giving fewer unique ratios
        square = plate_spectrum(n_modes=8, aspect_ratio=1.0)
        rect = plate_spectrum(n_modes=8, aspect_ratio=1.5)

        square_unique = len(set(round(r, 6) for r in _ratios(square)))
        rect_unique = len(set(round(r, 6) for r in _ratios(rect)))

        # Square plate has more degeneracy (fewer unique ratios)
        assert square_unique <= rect_unique

    def test_rejects_zero_modes(self) -> None:
        with pytest.raises(ValueError, match="n_modes must be at least 1"):
            plate_spectrum(n_modes=0)

    def test_rejects_non_positive_aspect_ratio(self) -> None:
        with pytest.raises(ValueError, match="aspect_ratio must be positive"):
            plate_spectrum(aspect_ratio=0.0)
        with pytest.raises(ValueError, match="aspect_ratio must be positive"):
            plate_spectrum(aspect_ratio=-1.0)


# ---------------------------------------------------------------------------
# tube_spectrum
# ---------------------------------------------------------------------------


class TestTubeSpectrum:
    def test_returns_valid_partials(self) -> None:
        spectrum = tube_spectrum()
        assert len(spectrum) == 8
        assert all("ratio" in p and "amp" in p for p in spectrum)

    def test_fundamental_is_one(self) -> None:
        for mode in ("both", "one", "neither"):
            assert tube_spectrum(open_ends=mode)[0]["ratio"] == pytest.approx(1.0)

    def test_ratios_sorted_ascending(self) -> None:
        for mode in ("both", "one", "neither"):
            ratios = _ratios(tube_spectrum(open_ends=mode))
            assert ratios == sorted(ratios)

    def test_both_ends_open_all_harmonics(self) -> None:
        spectrum = tube_spectrum(n_modes=5, open_ends="both")
        assert _ratios(spectrum) == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_one_end_open_odd_harmonics(self) -> None:
        spectrum = tube_spectrum(n_modes=5, open_ends="one")
        assert _ratios(spectrum) == [1.0, 3.0, 5.0, 7.0, 9.0]

    def test_neither_end_shifted_partials(self) -> None:
        spectrum = tube_spectrum(n_modes=4, open_ends="neither")
        expected = _STOPPED_PIPE_MODES[:4]
        for p, exp in zip(spectrum, expected, strict=True):
            assert p["ratio"] == pytest.approx(exp, abs=0.01)

    def test_n_modes_controls_length(self) -> None:
        assert len(tube_spectrum(n_modes=1)) == 1
        assert len(tube_spectrum(n_modes=4)) == 4

    def test_gentle_rolloff(self) -> None:
        spectrum = tube_spectrum(n_modes=4)
        amps = _amps(spectrum)
        # Monotonically decreasing
        for i in range(len(amps) - 1):
            assert amps[i] > amps[i + 1]

    def test_rejects_zero_modes(self) -> None:
        with pytest.raises(ValueError, match="n_modes must be at least 1"):
            tube_spectrum(n_modes=0)

    def test_rejects_bad_open_ends(self) -> None:
        with pytest.raises(ValueError, match="open_ends must be one of"):
            tube_spectrum(open_ends="half")


# ---------------------------------------------------------------------------
# bowl_spectrum
# ---------------------------------------------------------------------------


class TestBowlSpectrum:
    def test_returns_valid_partials(self) -> None:
        spectrum = bowl_spectrum()
        assert len(spectrum) == 8
        assert all("ratio" in p and "amp" in p for p in spectrum)

    def test_fundamental_is_one(self) -> None:
        assert bowl_spectrum()[0]["ratio"] == 1.0

    def test_ratios_sorted_ascending(self) -> None:
        ratios = _ratios(bowl_spectrum())
        assert ratios == sorted(ratios)

    def test_ratios_match_reference(self) -> None:
        spectrum = bowl_spectrum(n_modes=len(_BOWL_MODES))
        for p, expected in zip(spectrum, _BOWL_MODES, strict=True):
            assert p["ratio"] == pytest.approx(expected, abs=0.01)

    def test_n_modes_controls_length(self) -> None:
        assert len(bowl_spectrum(n_modes=1)) == 1
        assert len(bowl_spectrum(n_modes=5)) == 5

    def test_rejects_zero_modes(self) -> None:
        with pytest.raises(ValueError, match="n_modes must be at least 1"):
            bowl_spectrum(n_modes=0)

    def test_rejects_too_many_modes(self) -> None:
        with pytest.raises(ValueError, match="available bowl modes"):
            bowl_spectrum(n_modes=100)


# ---------------------------------------------------------------------------
# spectral_convolve
# ---------------------------------------------------------------------------


class TestSpectralConvolve:
    def test_single_partial_product(self) -> None:
        a = [{"ratio": 2.0, "amp": 1.0}]
        b = [{"ratio": 3.0, "amp": 1.0}]
        result = spectral_convolve(a, b)
        assert len(result) == 1
        assert result[0]["ratio"] == pytest.approx(6.0)
        assert result[0]["amp"] == pytest.approx(1.0)

    def test_identity_convolution(self) -> None:
        identity = [{"ratio": 1.0, "amp": 1.0}]
        spec = [
            {"ratio": 1.0, "amp": 1.0},
            {"ratio": 3 / 2, "amp": 0.5},
            {"ratio": 5 / 4, "amp": 0.3},
        ]
        result = spectral_convolve(identity, spec)
        assert len(result) == len(spec)
        for r, s in zip(result, sorted(spec, key=lambda p: p["ratio"]), strict=True):
            assert r["ratio"] == pytest.approx(s["ratio"], rel=1e-6)

    def test_ji_cross_products(self) -> None:
        a = [{"ratio": 1.0, "amp": 1.0}, {"ratio": 5 / 4, "amp": 0.8}]
        b = [{"ratio": 1.0, "amp": 1.0}, {"ratio": 3 / 2, "amp": 0.7}]
        result = spectral_convolve(a, b)

        result_ratios = [p["ratio"] for p in result]
        expected_products = {1.0, 3 / 2, 5 / 4, 15 / 8}
        for expected in expected_products:
            assert any(
                abs(1200 * math.log2(r / expected)) < 1.0 for r in result_ratios
            ), f"expected ratio {expected} not found in {result_ratios}"

    def test_merge_near_coincident(self) -> None:
        # Two partials very close together should merge
        a = [{"ratio": 1.0, "amp": 1.0}]
        b = [
            {"ratio": 3.0, "amp": 1.0},
            {"ratio": 3.0 * (2.0 ** (5 / 1200)), "amp": 1.0},  # 5 cents higher
        ]
        result = spectral_convolve(a, b, merge_tolerance_cents=10.0)
        # Should merge the two near-3.0 partials into one
        assert len(result) == 1

    def test_no_merge_when_far_apart(self) -> None:
        a = [{"ratio": 1.0, "amp": 1.0}]
        b = [
            {"ratio": 3.0, "amp": 1.0},
            {"ratio": 3.0 * (2.0 ** (50 / 1200)), "amp": 1.0},  # 50 cents higher
        ]
        result = spectral_convolve(a, b, merge_tolerance_cents=10.0)
        assert len(result) == 2

    def test_max_partials_cap(self) -> None:
        a = [{"ratio": float(i + 1), "amp": 1.0 / (i + 1)} for i in range(10)]
        b = [{"ratio": float(i + 1), "amp": 1.0 / (i + 1)} for i in range(10)]
        result = spectral_convolve(a, b, max_partials=5)
        assert len(result) <= 5

    def test_commutativity(self) -> None:
        a = [
            {"ratio": 1.0, "amp": 1.0},
            {"ratio": 5 / 4, "amp": 0.6},
            {"ratio": 3 / 2, "amp": 0.4},
        ]
        b = [
            {"ratio": 1.0, "amp": 1.0},
            {"ratio": 7 / 4, "amp": 0.5},
        ]
        result_ab = spectral_convolve(a, b)
        result_ba = spectral_convolve(b, a)

        assert len(result_ab) == len(result_ba)
        for p_ab, p_ba in zip(result_ab, result_ba, strict=True):
            assert p_ab["ratio"] == pytest.approx(p_ba["ratio"], rel=1e-6)
            assert p_ab["amp"] == pytest.approx(p_ba["amp"], rel=1e-6)

    def test_output_normalized(self) -> None:
        a = [{"ratio": 1.0, "amp": 0.5}, {"ratio": 2.0, "amp": 0.3}]
        b = [{"ratio": 1.0, "amp": 0.7}, {"ratio": 3.0, "amp": 0.2}]
        result = spectral_convolve(a, b)
        peak = max(p["amp"] for p in result)
        assert peak == pytest.approx(1.0)

    def test_pruning_removes_quiet_partials(self) -> None:
        a = [{"ratio": 1.0, "amp": 1.0}, {"ratio": 2.0, "amp": 0.0001}]
        b = [{"ratio": 1.0, "amp": 1.0}, {"ratio": 3.0, "amp": 0.0001}]
        # The cross product 2.0*3.0 has amp 0.0001*0.0001 = 1e-8, far below -60 dB
        result = spectral_convolve(a, b, min_amp_db=-60.0)
        result_ratios = [p["ratio"] for p in result]
        # The 6.0 product should be pruned
        assert not any(abs(r - 6.0) < 0.01 for r in result_ratios)

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="both input spectra must be non-empty"):
            spectral_convolve([], [{"ratio": 1.0, "amp": 1.0}])
        with pytest.raises(ValueError, match="both input spectra must be non-empty"):
            spectral_convolve([{"ratio": 1.0, "amp": 1.0}], [])

    def test_output_sorted_by_ratio(self) -> None:
        a = [{"ratio": 1.0, "amp": 1.0}, {"ratio": 2.0, "amp": 0.5}]
        b = [{"ratio": 1.0, "amp": 1.0}, {"ratio": 1.5, "amp": 0.5}]
        result = spectral_convolve(a, b)
        ratios = [p["ratio"] for p in result]
        assert ratios == sorted(ratios)


# ---------------------------------------------------------------------------
# Helpers (_cents_distance, _merge_partials)
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_cents_distance_octave(self) -> None:
        assert _cents_distance(2.0, 1.0) == pytest.approx(1200.0)

    def test_cents_distance_unison(self) -> None:
        assert _cents_distance(1.0, 1.0) == pytest.approx(0.0)

    def test_merge_partials_empty(self) -> None:
        assert _merge_partials([]) == []

    def test_merge_partials_combines_near(self) -> None:
        partials = [
            {"ratio": 1.0, "amp": 0.5},
            {"ratio": 1.003, "amp": 0.5},
        ]
        merged = _merge_partials(partials, cents_threshold=20.0)
        assert len(merged) == 1
        assert merged[0]["amp"] == pytest.approx(1.0)

    def test_merge_partials_keeps_distant(self) -> None:
        partials = [
            {"ratio": 1.0, "amp": 1.0},
            {"ratio": 2.0, "amp": 0.5},
        ]
        merged = _merge_partials(partials, cents_threshold=10.0)
        assert len(merged) == 2


# ---------------------------------------------------------------------------
# fractal_spectrum
# ---------------------------------------------------------------------------


class TestFractalSpectrum:
    def test_depth_zero_returns_seed(self) -> None:
        seed = [{"ratio": 1.0, "amp": 0.8}, {"ratio": 1.5, "amp": 0.4}]
        result = fractal_spectrum(seed, depth=0)
        assert len(result) == 2
        assert result[0]["amp"] == pytest.approx(1.0)
        assert result[0]["ratio"] == pytest.approx(1.0)

    def test_depth_one_produces_cross_products(self) -> None:
        seed = [{"ratio": 1.0, "amp": 1.0}, {"ratio": 1.5, "amp": 0.7}]
        result = fractal_spectrum(seed, depth=1, level_rolloff=0.5, max_partials=64)
        ratios = [p["ratio"] for p in result]
        assert any(r == pytest.approx(1.5 * 1.5) for r in ratios)

    def test_output_capped_at_max_partials(self) -> None:
        seed = [{"ratio": 1.0, "amp": 1.0}, {"ratio": 1.5, "amp": 0.7}]
        result = fractal_spectrum(seed, depth=3, max_partials=8)
        assert len(result) <= 8

    def test_deeper_depth_produces_more_partials(self) -> None:
        seed = [{"ratio": 1.0, "amp": 1.0}, {"ratio": 3 / 2, "amp": 0.7}]
        result_shallow = fractal_spectrum(
            seed, depth=1, level_rolloff=0.5, max_partials=64
        )
        result_deep = fractal_spectrum(
            seed, depth=3, level_rolloff=0.5, max_partials=64
        )
        assert len(result_deep) >= len(result_shallow)
        max_ratio_shallow = max(p["ratio"] for p in result_shallow)
        max_ratio_deep = max(p["ratio"] for p in result_deep)
        assert max_ratio_deep >= max_ratio_shallow

    def test_level_rolloff_controls_amplitude_rate(self) -> None:
        seed = [{"ratio": 1.0, "amp": 1.0}, {"ratio": 2.0, "amp": 0.5}]
        result_fast = fractal_spectrum(
            seed, depth=2, level_rolloff=0.3, max_partials=64
        )
        result_slow = fractal_spectrum(
            seed, depth=2, level_rolloff=0.8, max_partials=64
        )
        fast_min = min(p["amp"] for p in result_fast)
        slow_min = min(p["amp"] for p in result_slow)
        assert fast_min < slow_min

    def test_peak_normalized_to_one(self) -> None:
        seed = [{"ratio": 1.0, "amp": 1.0}, {"ratio": 1.5, "amp": 0.7}]
        result = fractal_spectrum(seed, depth=2)
        peak = max(p["amp"] for p in result)
        assert peak == pytest.approx(1.0)

    def test_invalid_params_raise(self) -> None:
        with pytest.raises(ValueError, match="seed must not be empty"):
            fractal_spectrum([])
        with pytest.raises(ValueError, match="depth must be non-negative"):
            fractal_spectrum([{"ratio": 1.0, "amp": 1.0}], depth=-1)
        with pytest.raises(ValueError, match="level_rolloff must be positive"):
            fractal_spectrum([{"ratio": 1.0, "amp": 1.0}], level_rolloff=0.0)
        with pytest.raises(ValueError, match="max_partials must be at least 1"):
            fractal_spectrum([{"ratio": 1.0, "amp": 1.0}], max_partials=0)


# ---------------------------------------------------------------------------
# vowel_formants
# ---------------------------------------------------------------------------


class TestVowelFormants:
    def test_all_five_vowels_return_valid_data(self) -> None:
        for name in ("a", "e", "i", "o", "u"):
            formants = vowel_formants(name)
            assert len(formants) >= 2
            for center, gain, bw in formants:
                assert center > 0
                assert gain > 0
                assert bw > 0

    def test_unknown_vowel_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown vowel"):
            vowel_formants("x")

    def test_case_insensitive(self) -> None:
        assert vowel_formants("A") == vowel_formants("a")


# ---------------------------------------------------------------------------
# formant_shape
# ---------------------------------------------------------------------------


class TestFormantShape:
    @pytest.fixture()
    def harmonic_partials(self) -> list[dict[str, float]]:
        return [{"ratio": float(i), "amp": 1.0} for i in range(1, 17)]

    def test_a_boosts_near_800hz(
        self, harmonic_partials: list[dict[str, float]]
    ) -> None:
        f0 = 100.0
        shaped = formant_shape(harmonic_partials, f0, "a")
        amp_at_8 = next(p["amp"] for p in shaped if p["ratio"] == 8.0)
        amp_at_1 = next(p["amp"] for p in shaped if p["ratio"] == 1.0)
        assert amp_at_8 > amp_at_1

    def test_different_vowels_produce_different_profiles(
        self, harmonic_partials: list[dict[str, float]]
    ) -> None:
        f0 = 200.0
        shaped_a = formant_shape(harmonic_partials, f0, "a")
        shaped_i = formant_shape(harmonic_partials, f0, "i")
        amps_a = [p["amp"] for p in shaped_a]
        amps_i = [p["amp"] for p in shaped_i]
        assert amps_a != amps_i

    def test_string_name_matches_explicit_formants(
        self, harmonic_partials: list[dict[str, float]]
    ) -> None:
        f0 = 200.0
        bw = 100.0
        shaped_str = formant_shape(harmonic_partials, f0, "a", bandwidth_hz=bw)
        explicit = [(c, g, bw) for c, g, _bw in vowel_formants("a")]
        shaped_explicit = formant_shape(harmonic_partials, f0, explicit)
        for ps, pe in zip(shaped_str, shaped_explicit, strict=True):
            assert ps["amp"] == pytest.approx(pe["amp"])

    def test_preserves_ratios(self, harmonic_partials: list[dict[str, float]]) -> None:
        shaped = formant_shape(harmonic_partials, 200.0, "e")
        for original, result in zip(harmonic_partials, shaped, strict=True):
            assert result["ratio"] == original["ratio"]


# ---------------------------------------------------------------------------
# formant_morph
# ---------------------------------------------------------------------------


class TestFormantMorph:
    @pytest.fixture()
    def simple_partials(self) -> list[dict[str, float]]:
        return [{"ratio": float(i), "amp": 1.0} for i in range(1, 9)]

    def test_single_vowel_weights_match_formant_shape(
        self, simple_partials: list[dict[str, float]]
    ) -> None:
        f0 = 200.0
        morphed = formant_morph(simple_partials, f0, ["a"])
        shaped = formant_shape(simple_partials, f0, vowel_formants("a"))
        for m, s in zip(morphed, shaped, strict=True):
            envelope = m["envelope"]
            assert len(envelope) == 1
            assert envelope[0]["value"] == pytest.approx(
                s["amp"] / m["amp"] if m["amp"] > 0 else 0.0
            )

    def test_generates_envelope_dicts(
        self, simple_partials: list[dict[str, float]]
    ) -> None:
        morphed = formant_morph(simple_partials, 200.0, ["a", "i"])
        for partial in morphed:
            assert "envelope" in partial
            envelope = partial["envelope"]
            assert isinstance(envelope, list)
            assert len(envelope) == 2
            for point in envelope:
                assert "time" in point
                assert "value" in point

    def test_even_time_distribution(
        self, simple_partials: list[dict[str, float]]
    ) -> None:
        morphed = formant_morph(simple_partials, 200.0, ["a", "e", "i"])
        envelope = morphed[0]["envelope"]
        times = [p["time"] for p in envelope]
        assert times == pytest.approx([0.0, 0.5, 1.0])

    def test_explicit_morph_times(
        self, simple_partials: list[dict[str, float]]
    ) -> None:
        custom_times = [0.0, 0.3, 1.0]
        morphed = formant_morph(
            simple_partials, 200.0, ["a", "e", "i"], morph_times=custom_times
        )
        envelope = morphed[0]["envelope"]
        times = [p["time"] for p in envelope]
        assert times == pytest.approx(custom_times)

    def test_preserves_original_amp(
        self, simple_partials: list[dict[str, float]]
    ) -> None:
        morphed = formant_morph(simple_partials, 200.0, ["a", "i"])
        for original, result in zip(simple_partials, morphed, strict=True):
            assert result["amp"] == original["amp"]

    def test_preserves_original_ratio(
        self, simple_partials: list[dict[str, float]]
    ) -> None:
        morphed = formant_morph(simple_partials, 200.0, ["a", "i"])
        for original, result in zip(simple_partials, morphed, strict=True):
            assert result["ratio"] == original["ratio"]

    def test_empty_vowel_sequence_raises(
        self, simple_partials: list[dict[str, float]]
    ) -> None:
        with pytest.raises(ValueError, match="vowel_sequence must not be empty"):
            formant_morph(simple_partials, 200.0, [])

    def test_mismatched_morph_times_raises(
        self, simple_partials: list[dict[str, float]]
    ) -> None:
        with pytest.raises(ValueError, match="morph_times length must match"):
            formant_morph(simple_partials, 200.0, ["a", "i"], morph_times=[0.0])

    def test_smoke_additive_engine_round_trip(self) -> None:
        """Verify morphed partials can be passed to the additive engine."""
        partials = harmonic_spectrum(n_partials=6, harmonic_rolloff=0.5)
        morphed = formant_morph(partials, 220.0, ["a", "o", "i"])

        for partial in morphed:
            assert "ratio" in partial
            assert "amp" in partial
            assert "envelope" in partial
            envelope = partial["envelope"]
            assert len(envelope) == 3
            assert all(isinstance(pt["time"], float) for pt in envelope)
            assert all(isinstance(pt["value"], float) for pt in envelope)
