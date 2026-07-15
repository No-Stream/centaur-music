"""Tests for sieved_harmonic_spectrum."""

from code_musics.spectra import harmonic_spectrum, sieved_harmonic_spectrum


class TestSievedHarmonicSpectrum:
    def test_omits_multiples_of_three(self) -> None:
        partials = sieved_harmonic_spectrum(n_partials=16)
        ratios = [p["ratio"] for p in partials]
        assert ratios == [1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0, 11.0, 13.0, 14.0, 16.0]

    def test_downweights_fives(self) -> None:
        plain = {
            p["ratio"]: p["amp"]
            for p in sieved_harmonic_spectrum(n_partials=16, harmonic_rolloff=0.8)
        }
        sieved = {
            p["ratio"]: p["amp"]
            for p in sieved_harmonic_spectrum(
                n_partials=16,
                harmonic_rolloff=0.8,
                downweight_factors={5: 0.35},
            )
        }
        assert sieved[5.0] == plain[5.0] * 0.35
        assert sieved[10.0] == plain[10.0] * 0.35
        assert sieved[7.0] == plain[7.0]

    def test_matches_harmonic_spectrum_when_nothing_sieved(self) -> None:
        assert sieved_harmonic_spectrum(
            n_partials=8, omit_factors=(), harmonic_rolloff=0.6
        ) == harmonic_spectrum(n_partials=8, harmonic_rolloff=0.6)
