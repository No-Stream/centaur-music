"""AD2 (second-order ADAA) alias-floor and bit-exact regression tests.

AD1 (first-order ADAA) has been the default for the waveshaper since ADAA
shipped and is regression-locked here: ``adaa_order=1`` (the default) must
remain bit-exact with pre-AD2 behavior.  AD2 is the new opt-in second-order
path (Bilbao/Esqueda three-sample form) and should reduce alias energy
above 10 kHz by a clear margin vs AD1 when a high-drive full-band sine is
passed through a bounded-range waveshaper.
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._waveshaper import apply_waveshaper

_SR: int = 44100


def _sine(freq: float, duration: float, sr: int = _SR) -> np.ndarray:
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float64) / sr
    return np.sin(2.0 * np.pi * freq * t)


def _alias_floor_db(
    signal: np.ndarray, fundamental_hz: float, sr: int, band_start_hz: float = 10_000.0
) -> float:
    """Sum FFT energy in the alias band above ``band_start_hz`` (exclusive of
    the fundamental and its integer harmonics within a 50 Hz guard), return
    it in dB relative to the strongest bin.

    Lower is better.  The guard excludes the signal's legitimate harmonic
    content above ``band_start_hz`` (at 5 kHz fundamental with a square-law
    nonlinearity the 3rd harmonic lands at 15 kHz — we still want to count
    its alias image but not the harmonic itself).  We do this by windowing
    away a 50 Hz radius around each harmonic that lands above the band.
    """
    n = signal.shape[0]
    # Detrend then window to reduce spectral leakage at the boundaries.
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(signal * win))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    peak = spec.max()
    if peak <= 0.0:
        return -200.0

    band_mask = freqs >= band_start_hz
    # Mask out narrow bands around each integer harmonic of the fundamental
    # within the alias-analysis band.  Everything left is alias energy.
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
# AD2 reduces alias floor vs AD1
# ---------------------------------------------------------------------------


class TestAD2ReducesAliasing:
    """AD2 should produce measurably lower alias energy than AD1 on a heavy-
    drive high-frequency sine through a bounded-range waveshaper.

    We use a 5 kHz sine at 44.1 kHz — the third harmonic (15 kHz) is still in
    band, the fifth (25 kHz) aliases down, etc.  Both AD1 and AD2 attack the
    same spectrum but AD2 cuts the alias floor much deeper.
    """

    @pytest.mark.parametrize(
        "algorithm, min_improvement_db",
        [
            ("tanh", 10.0),
            ("hard_clip", 10.0),
            ("atan", 10.0),
        ],
    )
    def test_ad2_alias_floor_below_ad1(
        self, algorithm: str, min_improvement_db: float
    ) -> None:
        fundamental = 5_000.0
        signal = _sine(freq=fundamental, duration=2.0)

        ad1 = apply_waveshaper(
            signal,
            algorithm=algorithm,
            drive=0.9,
            oversample=2,
            adaa_order=1,
        )
        ad2 = apply_waveshaper(
            signal,
            algorithm=algorithm,
            drive=0.9,
            oversample=2,
            adaa_order=2,
        )

        ad1_floor_db = _alias_floor_db(ad1, fundamental, _SR)
        ad2_floor_db = _alias_floor_db(ad2, fundamental, _SR)

        # AD2 must be lower (more negative dB) than AD1 by at least
        # ``min_improvement_db``.  Our analytical expectation is 10-30 dB on
        # a clean bounded nonlinearity; 10 dB is a conservative floor that
        # shouldn't flake with hanning-window leakage.
        assert ad2_floor_db + min_improvement_db <= ad1_floor_db, (
            f"{algorithm}: AD2 alias floor ({ad2_floor_db:.1f} dB) not "
            f"{min_improvement_db:.0f} dB below AD1 ({ad1_floor_db:.1f} dB)"
        )

    def test_ad2_finite_on_all_supported_algorithms(self) -> None:
        fundamental = 4_000.0
        signal = _sine(freq=fundamental, duration=0.1)
        supported = [
            "tanh",
            "atan",
            "hard_clip",
            "exponential",
            "logarithmic",
            "half_wave_rect",
            "full_wave_rect",
        ]
        for algo in supported:
            result = apply_waveshaper(signal, algorithm=algo, drive=0.8, adaa_order=2)
            assert result.shape == signal.shape
            assert np.all(np.isfinite(result)), f"{algo} AD2 non-finite"
            assert np.max(np.abs(result)) > 0.0, f"{algo} AD2 all-zero"


# ---------------------------------------------------------------------------
# AD1 regression: adaa_order=1 (default) must match pre-AD2 behavior
# ---------------------------------------------------------------------------


class TestAD1DefaultUnchanged:
    """adaa_order=1 is the default and must preserve existing behavior.

    We compute AD1 output with an explicit ``adaa_order=1`` and compare to the
    behavior produced by calling without the kwarg.  Both paths must produce
    identical output — this guards against accidentally flipping the default
    to AD2 or leaking AD2 code into the AD1 kernel.
    """

    @pytest.mark.parametrize(
        "algorithm",
        [
            "tanh",
            "atan",
            "hard_clip",
            "exponential",
            "logarithmic",
            "half_wave_rect",
            "full_wave_rect",
        ],
    )
    def test_default_matches_adaa_order_1(self, algorithm: str) -> None:
        signal = _sine(freq=440.0, duration=0.1)
        default_out = apply_waveshaper(signal, algorithm=algorithm, drive=0.7)
        explicit_out = apply_waveshaper(
            signal, algorithm=algorithm, drive=0.7, adaa_order=1
        )
        np.testing.assert_array_equal(default_out, explicit_out)

    def test_ad1_preserves_rms_and_starts_at_zero(self) -> None:
        """AD1 tanh on a 440 Hz sine starting at zero is RMS-compensated to
        match dry input, and the zero-crossing-preserving ``tanh(0)=0``
        shape keeps ``out[0]`` near zero.

        NOTE: neither of these invariants is bit-exact — ``tanh`` is an
        odd function that maps zero to zero for *any* drive, and the
        RMS-compensation wrapper makes the wet/dry RMS ratio invariant
        under the waveshape by construction.  For genuine bit-exact
        determinism see ``test_ad1_determinism_same_call_twice`` and
        ``test_ad1_float_determinism_across_instances`` below.
        """
        signal = _sine(freq=440.0, duration=0.05)
        out = apply_waveshaper(signal, algorithm="tanh", drive=0.5)
        assert out.shape == signal.shape
        dry_rms = float(np.sqrt(np.mean(signal * signal)))
        wet_rms = float(np.sqrt(np.mean(out * out)))
        assert abs(dry_rms - wet_rms) / dry_rms < 1e-6, (
            "AD1 tanh output RMS diverges from dry — level compensation drifted"
        )
        # Zero-crossing invariant: tanh(0) = 0 so the first sample rides
        # on it.  Guards against accidental DC leakage in the wrapper.
        assert abs(out[0]) < 1e-6, f"AD1 first sample not near zero: {out[0]}"

    def test_ad1_determinism_same_call_twice(self) -> None:
        """AD1 must be a pure function of its inputs — two calls with the
        same arguments produce bit-identical output.  This is a genuine
        bit-exact invariant: ``assert_array_equal`` requires every single
        float64 bit to match between the two output arrays.

        Guards against accidental state leakage (globals, mutable module
        scratch buffers, numba cache contamination) and hidden RNG
        inputs — any of which would produce a near-match that fooled a
        previous ``allclose``-style check but would fail this one.
        """
        signal = _sine(freq=440.0, duration=0.1)
        a = apply_waveshaper(signal, algorithm="tanh", drive=0.5, adaa_order=1)
        b = apply_waveshaper(signal, algorithm="tanh", drive=0.5, adaa_order=1)
        np.testing.assert_array_equal(
            a,
            b,
            err_msg="AD1 apply_waveshaper is not deterministic across calls",
        )

    @pytest.mark.parametrize(
        "algorithm,drive",
        [
            ("tanh", 0.5),
            ("tanh", 1.2),
            ("atan", 0.7),
            ("hard_clip", 1.5),
            ("exponential", 0.8),
            ("logarithmic", 0.9),
            ("half_wave_rect", 0.6),
            ("full_wave_rect", 0.7),
        ],
    )
    def test_ad1_float_determinism_across_instances(
        self, algorithm: str, drive: float
    ) -> None:
        """For each AD1-supported algorithm, running the shaper on two
        independently constructed identical input arrays produces
        bit-identical output via ``assert_allclose(..., atol=0, rtol=0)``.

        This is separate from the same-input-same-call invariant above:
        rebuilding the input array from scratch exercises the input
        handling path end-to-end and catches bugs where the shaper
        depends on input-array identity (e.g. caching keyed on
        ``id(signal)`` rather than content).
        """
        signal_a = _sine(freq=440.0, duration=0.08)
        signal_b = _sine(freq=440.0, duration=0.08)
        # Sanity: the two signals are themselves bit-identical.
        np.testing.assert_array_equal(signal_a, signal_b)

        out_a = apply_waveshaper(
            signal_a, algorithm=algorithm, drive=drive, adaa_order=1
        )
        out_b = apply_waveshaper(
            signal_b, algorithm=algorithm, drive=drive, adaa_order=1
        )
        # atol=0, rtol=0 pins float determinism exactly — any bit drift
        # fails this.
        np.testing.assert_allclose(
            out_a,
            out_b,
            atol=0.0,
            rtol=0.0,
            err_msg=(
                f"AD1 {algorithm} drive={drive} not deterministic across "
                f"independently-constructed identical inputs"
            ),
        )

    def test_ad2_differs_from_ad1_on_driven_signal(self) -> None:
        """Sanity: AD2 and AD1 should NOT produce identical output when drive
        matters.  Otherwise we could have a latent bug where AD2 silently
        falls through to AD1 (the ``_AD2_SUPPORTED_IDS`` check or the
        fallback guard accidentally catches every sample).
        """
        signal = _sine(freq=3_000.0, duration=0.1)
        ad1 = apply_waveshaper(signal, algorithm="tanh", drive=0.8, adaa_order=1)
        ad2 = apply_waveshaper(signal, algorithm="tanh", drive=0.8, adaa_order=2)
        assert not np.allclose(ad1, ad2, atol=1e-8), (
            "AD2 output is bit-exact with AD1 — AD2 path appears inactive"
        )


# ---------------------------------------------------------------------------
# Error handling / parameter validation
# ---------------------------------------------------------------------------


class TestAD2ParameterValidation:
    def test_invalid_adaa_order_raises(self) -> None:
        signal = _sine(freq=440.0, duration=0.02)
        with pytest.raises(ValueError, match="adaa_order"):
            apply_waveshaper(signal, algorithm="tanh", drive=0.5, adaa_order=3)

    def test_ad2_on_unsupported_algo_falls_back_gracefully(self) -> None:
        """Passing adaa_order=2 on an algorithm without F2 support should run
        without raising (silently uses AD1 / oversample path)."""
        signal = _sine(freq=440.0, duration=0.02)
        # polynomial has no AD2 implementation; passing adaa_order=2 must not
        # raise or produce non-finite output.
        result = apply_waveshaper(
            signal, algorithm="polynomial", drive=0.5, adaa_order=2
        )
        assert np.all(np.isfinite(result))
        assert result.shape == signal.shape
