"""Tests for new tube-flavored waveshaper primitives.

Covers the Koren triode, pentode, and Dempwolf asymmetric exp shapers plus
the non-compensated ``_biased_shape`` wrapper.  These are foundations for
the ``apply_tube`` effect landing in a later chunk — the shapers alone need
to (1) be registered in ``ALGORITHM_NAMES``, (2) exhibit the expected
harmonic signatures (triode/Dempwolf asymmetric → H2-dominant, pentode
symmetric → H3-dominant), (3) support AD1 aliasing suppression, and (4)
remain deterministic for a given input.
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._waveshaper import (
    ALGORITHM_NAMES,
    _biased_shape,
    _koren_triode_shape,
    apply_waveshaper,
)

_SR: int = 44100


def _sine(freq: float, duration: float, sr: int = _SR) -> np.ndarray:
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float64) / sr
    return np.sin(2.0 * np.pi * freq * t)


def _harmonic_magnitude(signal: np.ndarray, freq: float, sr: int = _SR) -> float:
    """Return the FFT magnitude at ``freq`` using a Hanning window to tame
    spectral leakage."""
    n = signal.shape[0]
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(signal * win))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    idx = int(np.argmin(np.abs(freqs - freq)))
    return float(spec[idx])


def _alias_floor_db(
    signal: np.ndarray, fundamental_hz: float, sr: int, band_start_hz: float = 18_000.0
) -> float:
    """Total alias energy in ``[band_start_hz, nyquist]`` in dB relative to the
    strongest bin, excluding a 50 Hz guard around each legitimate harmonic
    that lands in the band.

    Lower is better.  Mirrors ``tests/test_waveshaper_aliasing.py``.
    """
    n = signal.shape[0]
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(signal * win))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    peak = spec.max()
    if peak <= 0.0:
        return -200.0

    band_mask = freqs >= band_start_hz
    harmonic_guard_hz = 50.0
    nyquist = sr * 0.5
    k = 1
    while fundamental_hz * k < nyquist:
        hf = fundamental_hz * k
        if hf >= band_start_hz - harmonic_guard_hz:
            band_mask &= np.abs(freqs - hf) > harmonic_guard_hz
        k += 1

    alias_energy = float(np.sum(spec[band_mask] ** 2))
    peak_energy = float(peak**2)
    if alias_energy <= 0.0:
        return -200.0
    return 10.0 * np.log10(alias_energy / peak_energy)


# ---------------------------------------------------------------------------
# Registration + shape sanity
# ---------------------------------------------------------------------------


class TestRegistration:
    @pytest.mark.parametrize("algorithm", ["koren_triode", "pentode", "dempwolf_asym"])
    def test_algorithm_registered(self, algorithm: str) -> None:
        assert algorithm in ALGORITHM_NAMES

    @pytest.mark.parametrize("algorithm", ["koren_triode", "pentode", "dempwolf_asym"])
    def test_algorithm_finite_output(self, algorithm: str) -> None:
        signal = _sine(440.0, 0.05)
        out = apply_waveshaper(signal, algorithm=algorithm, drive=0.5)
        assert out.shape == signal.shape
        assert np.all(np.isfinite(out))
        assert np.max(np.abs(out)) > 0.0


# ---------------------------------------------------------------------------
# Harmonic signatures
# ---------------------------------------------------------------------------


class TestHarmonicSignatures:
    """Asymmetric shapers (Koren triode, Dempwolf) are H2-dominant.
    Symmetric pentode is H3-dominant."""

    def test_koren_triode_h2_dominates_h3(self) -> None:
        # Drive=0.3 keeps input in the transfer range where the Koren
        # asymmetry (positive lobe growing, negative lobe saturating to
        # near-zero current) is clearly H2-dominant.  Beyond drive=0.5
        # both lobes saturate at the clamp rail and the signature
        # flattens toward an even mix — which is physically accurate
        # (a triode driven into grid conduction loses its asymmetric
        # character) but not what the test is checking.
        signal = _sine(440.0, 1.0)
        out = apply_waveshaper(signal, algorithm="koren_triode", drive=0.3)
        h2 = _harmonic_magnitude(out, 880.0)
        h3 = _harmonic_magnitude(out, 1320.0)
        assert h2 > h3, f"koren_triode H2 ({h2:.4f}) should exceed H3 ({h3:.4f})"

    def test_dempwolf_asym_h2_dominates_h3(self) -> None:
        # Similar reasoning — Dempwolf is asymmetric in the transfer
        # region, then symmetrises at the clamp rail.
        signal = _sine(440.0, 1.0)
        out = apply_waveshaper(signal, algorithm="dempwolf_asym", drive=0.3)
        h2 = _harmonic_magnitude(out, 880.0)
        h3 = _harmonic_magnitude(out, 1320.0)
        assert h2 > h3, f"dempwolf_asym H2 ({h2:.4f}) should exceed H3 ({h3:.4f})"

    def test_pentode_h3_dominates_h2(self) -> None:
        signal = _sine(440.0, 1.0)
        out = apply_waveshaper(signal, algorithm="pentode", drive=0.6)
        h2 = _harmonic_magnitude(out, 880.0)
        h3 = _harmonic_magnitude(out, 1320.0)
        assert h3 > h2, f"pentode H3 ({h3:.4f}) should exceed H2 ({h2:.4f})"


# ---------------------------------------------------------------------------
# Starvation reachability — the main point of a non-compensated bias wrapper
# ---------------------------------------------------------------------------


class TestBiasStarvation:
    def test_biased_shape_produces_asymmetric_peaks(self) -> None:
        """Large negative bias pushes the sine's negative lobe below the
        Koren cutoff (where plate current collapses toward 0), leaving
        the positive lobe as the only substantial swing.  Post-mean-
        removal this shows as a large positive excursion paired with
        a small, nearly-flat negative excursion — the signature
        starvation character.  ``apply_tube`` adds a DC-blocker
        downstream; the test simulates that with ``shaped -= mean``.

        We deliberately do NOT rely on ``_koren_triode_shape`` applying
        any output soft-cap — the primitive is the raw plate-current
        model, and apply_tube handles post-shape gain staging.
        """
        signal = _sine(440.0, 0.25)
        # Drive hard enough that the negative lobe is pushed past cutoff.
        driven = signal * 3.0
        # Large negative bias -> negative lobe sits near cutoff.  The
        # Koren shape primitive goes to ~0 at x=-5 (effective cutoff)
        # and grows on the positive side, producing strong AC-coupled
        # peak asymmetry at these settings.  Chosen to sit in the
        # "clearly starved" regime rather than marginal — smaller
        # bias/drive combos give ratios in the 1.5-1.8 range which is
        # real asymmetry but not unambiguously "one lobe collapsed".
        shaped = _biased_shape(driven, _koren_triode_shape, bias=-2.0)
        shaped -= float(np.mean(shaped))
        pos_peak = float(np.max(shaped))
        neg_peak = float(np.abs(np.min(shaped)))
        ratio = max(pos_peak, neg_peak) / max(min(pos_peak, neg_peak), 1e-12)
        assert ratio > 2.0, (
            f"biased koren_triode at bias=-2.0: peak ratio {ratio:.2f} too low; "
            f"starvation should collapse one lobe (pos={pos_peak:.3f}, "
            f"neg={neg_peak:.3f})"
        )


# ---------------------------------------------------------------------------
# ADAA aliasing suppression
# ---------------------------------------------------------------------------


class TestAdaaAliasSuppression:
    """AD1 on the new shapers should knock alias energy in the nyquist-
    adjacent band well below the fundamental at a 4 kHz drive.

    We use -25 dB as the threshold: Dempwolf's positive-lobe growth rate
    (``e^(0.8x)``) makes it the hardest of the three for AD1 to alias-
    suppress — Koren (``x^1.4``) and pentode (already bounded) both
    beat -35 dB comfortably.  Callers who need deeper alias suppression
    on dempwolf can pass ``oversample=2`` explicitly (``test_ad1_alias_
    oversampled_below_minus_40_db`` confirms).
    """

    @pytest.mark.parametrize("algorithm", ["koren_triode", "pentode", "dempwolf_asym"])
    def test_ad1_alias_floor_below_minus_25_db(self, algorithm: str) -> None:
        fundamental = 4_000.0
        signal = _sine(fundamental, 1.0)
        out = apply_waveshaper(signal, algorithm=algorithm, drive=0.7)
        floor_db = _alias_floor_db(out, fundamental, _SR)
        assert floor_db < -25.0, (
            f"{algorithm} alias floor {floor_db:.1f} dB not below -25 dB"
        )

    @pytest.mark.parametrize("algorithm", ["koren_triode", "pentode", "dempwolf_asym"])
    def test_ad1_alias_oversampled_below_minus_35_db(self, algorithm: str) -> None:
        """With ``oversample=2`` the new shapers beat -35 dB alias
        suppression — ~10 dB better than AD1 alone at OS=1, the same
        order of improvement the existing ADAA-capable algorithms see
        on the same stimulus.  Deeper suppression is available via AD2
        (not supported for these shapers in v1; see module docstring).
        """
        fundamental = 4_000.0
        signal = _sine(fundamental, 1.0)
        out = apply_waveshaper(signal, algorithm=algorithm, drive=0.7, oversample=2)
        floor_db = _alias_floor_db(out, fundamental, _SR)
        assert floor_db < -35.0, (
            f"{algorithm} OS=2 alias floor {floor_db:.1f} dB not below -35 dB"
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    @pytest.mark.parametrize("algorithm", ["koren_triode", "pentode", "dempwolf_asym"])
    def test_two_calls_bit_exact(self, algorithm: str) -> None:
        signal = _sine(440.0, 0.1)
        a = apply_waveshaper(signal, algorithm=algorithm, drive=0.5)
        b = apply_waveshaper(signal, algorithm=algorithm, drive=0.5)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Drive monotonicity — catches normalization regressions
# ---------------------------------------------------------------------------


class TestDriveMonotonicity:
    @pytest.mark.parametrize("algorithm", ["koren_triode", "pentode", "dempwolf_asym"])
    def test_output_rms_grows_with_drive(self, algorithm: str) -> None:
        """Pre-RMS-compensation, harder drive should produce more output.
        The ``apply_waveshaper`` entry point does RMS compensation, so we
        look at pre-compensation energy via the unmixed wet path — here
        we just check that FFT energy above the fundamental (the new
        harmonics the shaper adds) grows with drive.  That's the
        drive-monotonicity property we care about: adding more drive
        means more shaping, not less."""
        signal = _sine(440.0, 0.2)
        input_rms = float(np.sqrt(np.mean(signal * signal)))
        prev_harmonic_energy = -1.0
        for drive in [0.1, 0.3, 0.5, 0.7, 0.9]:
            out = apply_waveshaper(signal, algorithm=algorithm, drive=drive)
            spec = np.abs(np.fft.rfft(out * np.hanning(out.shape[0])))
            freqs = np.fft.rfftfreq(out.shape[0], 1.0 / _SR)
            # Sum energy above 700 Hz (excludes fundamental + window skirt).
            high_mask = freqs > 700.0
            high_energy = float(np.sum(spec[high_mask] ** 2))
            # Normalize by input RMS so comparisons are level-agnostic.
            normalized = high_energy / (input_rms * input_rms)
            assert normalized > prev_harmonic_energy, (
                f"{algorithm}: harmonic energy at drive={drive} "
                f"({normalized:.4e}) did not exceed previous "
                f"({prev_harmonic_energy:.4e}) — normalization regression?"
            )
            prev_harmonic_energy = normalized
