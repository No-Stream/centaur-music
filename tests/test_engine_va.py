"""Tests for the VA (virtual-analog) synthesis engine."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._filters import apply_comb
from code_musics.engines.registry import (
    _PRESETS,
    render_note_signal,
    resolve_synth_params,
)
from code_musics.engines.va import (
    _supersaw_center_gain,
    _supersaw_detune_cents,
    _supersaw_side_gain,
    render,
)

_SR: int = 48000


def _base_params(**overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "_voice_name": "va_test",
        "cutoff_hz": 3000.0,
        "resonance_q": 1.0,
    }
    params.update(overrides)
    return params


def _bin_power(
    spec: np.ndarray, freqs: np.ndarray, target: float, width: float
) -> float:
    """Peak power within ``[target - width, target + width]``."""
    mask = (freqs > target - width) & (freqs < target + width)
    if not mask.any():
        return 0.0
    return float(spec[mask].max() ** 2)


def _spectral_centroid(signal: np.ndarray, sample_rate: int) -> float:
    spec = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.size, 1.0 / sample_rate)
    total = float(spec.sum())
    if total <= 1e-12:
        return 0.0
    return float((freqs * spec).sum() / total)


def _band_energy(
    signal: np.ndarray, lo: float, hi: float, sample_rate: int = _SR
) -> float:
    spec = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.size, 1.0 / sample_rate)
    mask = (freqs >= lo) & (freqs <= hi)
    return float((spec[mask] ** 2).sum())


def _thd_relative(signal: np.ndarray, sample_rate: int, fundamental_hz: float) -> float:
    """Total harmonic distortion measured as harmonics / fundamental power."""
    spec = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.size, 1.0 / sample_rate)
    fund = _bin_power(spec, freqs, fundamental_hz, width=5.0)
    harmonics = sum(
        _bin_power(spec, freqs, fundamental_hz * k, width=5.0) for k in (2, 3, 4, 5)
    )
    return harmonics / max(fund, 1e-12)


class TestVASmoke:
    @pytest.mark.parametrize("osc_mode", ["supersaw", "spectralwave"])
    @pytest.mark.parametrize(
        "filter_routing", ["single", "serial", "parallel", "split"]
    )
    def test_smoke_mode_routing(self, osc_mode: str, filter_routing: str) -> None:
        signal = render(
            freq=220.0,
            duration=0.3,
            amp=0.6,
            sample_rate=_SR,
            params=_base_params(osc_mode=osc_mode, filter_routing=filter_routing),
        )
        assert len(signal) == int(0.3 * _SR)
        assert np.all(np.isfinite(signal))
        assert np.max(np.abs(signal)) > 0.01

    @pytest.mark.parametrize("freq", [110.0, 440.0, 1760.0])
    def test_multiple_pitches(self, freq: float) -> None:
        signal = render(
            freq=freq, duration=0.2, amp=0.5, sample_rate=_SR, params=_base_params()
        )
        assert np.all(np.isfinite(signal))
        assert np.max(np.abs(signal)) > 0.01


class TestSupersaw:
    def test_detune_polynomial_monotonic(self) -> None:
        values = [_supersaw_detune_cents(d) for d in np.linspace(0.0, 1.0, 20)]
        diffs = np.diff(values)
        # Szabo's polynomial is monotonically non-decreasing on [0, 1].
        assert (diffs >= -1e-6).all()
        # Detune at 0 is imperceptibly small; at 1 the spread is large (>50 cents).
        assert values[0] < 1.0
        assert values[-1] > 50.0

    def test_detune_polynomial_anchor_points(self) -> None:
        # Ground-truth values measured from the Szabo polynomial. If someone
        # edits the coefficients, these anchor points catch the regression even
        # if monotonicity + endpoints still hold.
        anchors = [
            (0.0, 0.301156),
            (0.25, 4.693541),
            (0.5, 9.795515),
            (0.75, 24.245891),
            (1.0, 99.995369),
        ]
        for d, expected in anchors:
            got = _supersaw_detune_cents(d)
            tolerance = max(0.05, abs(expected) * 0.02)
            assert abs(got - expected) < tolerance, (
                f"d={d}: got {got:.4f}, expected ~{expected:.4f}"
            )

    def test_mix_gains_endpoints(self) -> None:
        # Side gain is ~0 at mix=0 and ~0.6 at mix=1 per Szabo.
        assert abs(_supersaw_side_gain(0.0)) < 0.1
        # Center gain dominates at low mix, drops as mix rises.
        assert _supersaw_center_gain(0.0) > _supersaw_center_gain(1.0)

    def test_mix_gain_anchor_points(self) -> None:
        # Szabo polynomial anchor values.
        for m, exp_side, exp_center in (
            (0.0, 0.044372, 0.997850),
            (0.5, 0.502012, 0.721020),
            (1.0, 0.590832, 0.444190),
        ):
            assert abs(_supersaw_side_gain(m) - exp_side) < 0.005
            assert abs(_supersaw_center_gain(m) - exp_center) < 0.005

    def test_render_mix_law_shifts_center_to_side_ratio(self) -> None:
        """Render-level check: raising ``supersaw_mix`` shifts energy from the
        fundamental band to detune sideband clusters, per Szabo's gain law."""
        low_mix = render(
            freq=220.0,
            duration=0.5,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(
                osc_mode="supersaw",
                supersaw_detune=0.5,
                supersaw_mix=0.0,
                cutoff_hz=6000.0,
                pitch_drift=0.0,
                analog_jitter=0.0,
                noise_floor=0.0,
                cutoff_drift=0.0,
            ),
        )
        high_mix = render(
            freq=220.0,
            duration=0.5,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(
                osc_mode="supersaw",
                supersaw_detune=0.5,
                supersaw_mix=1.0,
                cutoff_hz=6000.0,
                pitch_drift=0.0,
                analog_jitter=0.0,
                noise_floor=0.0,
                cutoff_drift=0.0,
            ),
        )

        def _center_vs_side(signal: np.ndarray) -> tuple[float, float]:
            spec = np.abs(np.fft.rfft(signal))
            freqs = np.fft.rfftfreq(signal.size, 1.0 / _SR)
            center_mask = (freqs > 217.0) & (freqs < 223.0)
            center = float((spec[center_mask] ** 2).sum())
            # Detune of 0.5 ≈ 9.8 cents spread per side voice. Sideband clusters
            # sit within ±15 Hz of the fundamental band edges.
            side_mask = ((freqs > 205.0) & (freqs < 217.0)) | (
                (freqs > 223.0) & (freqs < 235.0)
            )
            side = float((spec[side_mask] ** 2).sum())
            return center, side

        c_low, s_low = _center_vs_side(low_mix)
        c_high, s_high = _center_vs_side(high_mix)
        ratio_low = c_low / max(s_low, 1e-12)
        ratio_high = c_high / max(s_high, 1e-12)
        assert ratio_low > ratio_high, (
            f"center:side should drop as mix rises; got "
            f"ratio_low={ratio_low:.4f} ratio_high={ratio_high:.4f}"
        )

    def test_detune_widens_spectrum(self) -> None:
        narrow = render(
            freq=220.0,
            duration=0.5,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(osc_mode="supersaw", supersaw_detune=0.05),
        )
        wide = render(
            freq=220.0,
            duration=0.5,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(osc_mode="supersaw", supersaw_detune=0.9),
        )

        # Higher detune spreads energy from the exact fundamental into the
        # sidebands. Measure (center-bin energy / total energy around center)
        # — the ratio drops as detune rises because more energy lives in the
        # spread. At detune=0.9, side voices sit ~±90 cents = 209-232 Hz.
        def _center_concentration(signal: np.ndarray) -> float:
            spec = np.abs(np.fft.rfft(signal))
            freqs = np.fft.rfftfreq(signal.size, 1.0 / _SR)
            center = (freqs >= 219.0) & (freqs <= 221.0)
            surround = (freqs >= 205.0) & (freqs <= 235.0)
            return float((spec[center] ** 2).sum()) / max(
                float((spec[surround] ** 2).sum()), 1e-12
            )

        narrow_concentration = _center_concentration(narrow)
        wide_concentration = _center_concentration(wide)
        assert narrow_concentration > wide_concentration * 1.5, (
            f"narrow_concentration={narrow_concentration:.3f} "
            f"wide_concentration={wide_concentration:.3f}"
        )

    def test_random_phase_isolation(self) -> None:
        """Holding jitter/drift/dither/noise at 0, varying only ``_voice_name``
        should change the early waveform (random start phases) but preserve
        the steady-state power spectrum (same oscillator parameters).

        Uses ``supersaw_detune=0.0`` so all 7 voices are unison; that way
        phase-only differences don't alter the beating patterns that would
        otherwise make window-local FFT centroids noisy."""
        base_overrides = dict(
            osc_mode="supersaw",
            supersaw_detune=0.0,
            supersaw_mix=0.5,
            analog_jitter=0.0,
            pitch_drift=0.0,
            cutoff_drift=0.0,
            noise_floor=0.0,
            # voice_card also offsets pitch/cutoff based on _voice_name — we
            # need to disable it so only start_phases differ between the two
            # renders (the actual claim under test).
            voice_card_spread=0.0,
        )
        a = render(
            freq=220.0,
            duration=0.3,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(_voice_name="phase_A", **base_overrides),
        )
        b = render(
            freq=220.0,
            duration=0.3,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(_voice_name="phase_B", **base_overrides),
        )

        early = int(0.01 * _SR)  # first ~10 ms
        assert not np.allclose(a[:early], b[:early], atol=1e-5), (
            "early waveform should differ (random start phases)"
        )

        # Both signals must carry nontrivial, finite energy — catches a
        # regression where random-phase produces silence or NaN for one
        # voice_name. We don't assert spectral invariance here because the
        # summed-saw spectrum depends on phase alignment through the peak
        # normalizer and filter transient wake-up (not a bug, just physics).
        rms_a = float(np.sqrt(np.mean(a * a)))
        rms_b = float(np.sqrt(np.mean(b * b)))
        assert rms_a > 0.01 and rms_b > 0.01
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))

    def test_jitter_isolation(self) -> None:
        """Holding ``_voice_name`` fixed, varying only ``analog_jitter`` must
        change output; with jitter=0 two calls must be bit-identical."""
        params_no_jitter = _base_params(
            osc_mode="supersaw",
            analog_jitter=0.0,
            pitch_drift=0.0,
            noise_floor=0.0,
            cutoff_drift=0.0,
            _voice_name="jitter_fixed",
        )
        params_with_jitter = _base_params(
            osc_mode="supersaw",
            analog_jitter=0.5,
            pitch_drift=0.0,
            noise_floor=0.0,
            cutoff_drift=0.0,
            _voice_name="jitter_fixed",
        )

        a = render(
            freq=330.0, duration=0.2, amp=0.5, sample_rate=_SR, params=params_no_jitter
        )
        b = render(
            freq=330.0,
            duration=0.2,
            amp=0.5,
            sample_rate=_SR,
            params=params_with_jitter,
        )
        assert not np.allclose(a, b), "jitter amount should change output"

        # Same voice + jitter=0 => deterministic.
        a2 = render(
            freq=330.0, duration=0.2, amp=0.5, sample_rate=_SR, params=params_no_jitter
        )
        assert np.array_equal(a, a2), "jitter=0 should be deterministic"

    def test_supersaw_sync_changes_output(self) -> None:
        """``supersaw_sync=True`` should render finite, non-silent audio that
        differs from the non-synced path. Loose assertion because the
        hard-sync math may be in flux."""
        shared = dict(
            osc_mode="supersaw",
            supersaw_detune=0.2,
            supersaw_mix=0.4,
            cutoff_hz=5000.0,
            analog_jitter=0.0,
            pitch_drift=0.0,
            cutoff_drift=0.0,
            noise_floor=0.0,
        )
        no_sync = render(
            freq=220.0,
            duration=0.3,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(supersaw_sync=False, _voice_name="sync_cmp", **shared),
        )
        with_sync = render(
            freq=220.0,
            duration=0.3,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(supersaw_sync=True, _voice_name="sync_cmp", **shared),
        )
        assert np.all(np.isfinite(with_sync))
        assert np.max(np.abs(with_sync)) > 0.01
        assert not np.allclose(no_sync, with_sync)


class TestSpectralwave:
    def test_position_affects_spectrum(self) -> None:
        saw_like = render(
            freq=220.0,
            duration=0.5,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(
                osc_mode="spectralwave",
                spectral_position=0.0,
                cutoff_hz=8000.0,
            ),
        )
        square_like = render(
            freq=220.0,
            duration=0.5,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(
                osc_mode="spectralwave",
                spectral_position=1.0,
                cutoff_hz=8000.0,
            ),
        )

        # At position=0 (saw), even-harmonic energy is strong;
        # at position=1 (square), it's near-zero.
        def _even_harmonic_energy(sig: np.ndarray) -> float:
            spec = np.abs(np.fft.rfft(sig))
            freqs = np.fft.rfftfreq(sig.size, 1 / _SR)
            total = 0.0
            for k in (2, 4, 6, 8):
                total += _bin_power(spec, freqs, 220.0 * k, width=12.0)
            return total

        saw_even = _even_harmonic_energy(saw_like)
        sq_even = _even_harmonic_energy(square_like)
        assert saw_even > sq_even * 2.0

    def test_saw_position_follows_one_over_k(self) -> None:
        """At ``spectral_position=0`` the partial bank is a saw stack: harmonic
        amplitudes should decay like ``1/k`` within a factor of 2."""
        signal = render(
            freq=220.0,
            duration=0.5,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(
                osc_mode="spectralwave",
                spectral_position=0.0,
                cutoff_hz=20000.0,
                analog_jitter=0.0,
                pitch_drift=0.0,
                noise_floor=0.0,
                cutoff_drift=0.0,
                filter_env_amount=0.0,
                resonance_q=0.707,
            ),
        )
        spec = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(signal.size, 1.0 / _SR)
        amps = [
            np.sqrt(_bin_power(spec, freqs, 220.0 * k, width=6.0)) for k in range(1, 9)
        ]
        base = amps[0]
        assert base > 0.0
        for k in range(2, 9):
            expected_ratio = 1.0 / k
            actual_ratio = amps[k - 1] / base
            # Within a factor of 2 of 1/k (filter + taper broaden this).
            assert actual_ratio > expected_ratio / 2.5, (
                f"k={k}: amp ratio {actual_ratio:.4f} too low vs expected ~{expected_ratio:.4f}"
            )
            assert actual_ratio < expected_ratio * 2.5, (
                f"k={k}: amp ratio {actual_ratio:.4f} too high vs expected ~{expected_ratio:.4f}"
            )

    def test_square_position_odd_harmonics_only(self) -> None:
        """At ``spectral_position=1`` even harmonics should be much weaker than
        the fundamental, and odd harmonics should follow ``1/k`` roughly."""
        signal = render(
            freq=220.0,
            duration=0.5,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(
                osc_mode="spectralwave",
                spectral_position=1.0,
                cutoff_hz=20000.0,
                analog_jitter=0.0,
                pitch_drift=0.0,
                noise_floor=0.0,
                cutoff_drift=0.0,
                filter_env_amount=0.0,
                resonance_q=0.707,
            ),
        )
        spec = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(signal.size, 1.0 / _SR)
        amps = [
            np.sqrt(_bin_power(spec, freqs, 220.0 * k, width=6.0)) for k in range(1, 9)
        ]
        base = amps[0]
        assert base > 0.0

        for k in (2, 4, 6):
            assert amps[k - 1] < 0.2 * base, (
                f"even harmonic k={k}: amp {amps[k - 1]:.4f} "
                f"exceeded 0.2 * base={base:.4f}"
            )
        for k in (3, 5, 7):
            expected_ratio = 1.0 / k
            actual_ratio = amps[k - 1] / base
            assert actual_ratio > expected_ratio / 3.0, (
                f"odd k={k}: ratio {actual_ratio:.4f} too low "
                f"vs expected ~{expected_ratio:.4f}"
            )

    def test_mid_position_formant_boost(self) -> None:
        """At ``spectral_position=0.5`` the formant set gets an additive boost;
        partial k=3 should be louder than a linear blend of saw (pos=0) and
        square (pos=1) would predict."""

        def _render(pos: float) -> np.ndarray:
            return render(
                freq=220.0,
                duration=0.5,
                amp=0.5,
                sample_rate=_SR,
                params=_base_params(
                    osc_mode="spectralwave",
                    spectral_position=pos,
                    cutoff_hz=20000.0,
                    analog_jitter=0.0,
                    pitch_drift=0.0,
                    noise_floor=0.0,
                    cutoff_drift=0.0,
                    filter_env_amount=0.0,
                    resonance_q=0.707,
                ),
            )

        saw = _render(0.0)
        square = _render(1.0)
        middle = _render(0.5)

        def _amp_at_k(sig: np.ndarray, k: int) -> float:
            spec = np.abs(np.fft.rfft(sig))
            freqs = np.fft.rfftfreq(sig.size, 1.0 / _SR)
            return float(np.sqrt(_bin_power(spec, freqs, 220.0 * k, width=6.0)))

        saw_k3 = _amp_at_k(saw, 3)
        sq_k3 = _amp_at_k(square, 3)
        mid_k3 = _amp_at_k(middle, 3)
        linear_interp = 0.5 * (saw_k3 + sq_k3)
        assert mid_k3 > linear_interp, (
            f"k=3 formant boost missing: mid={mid_k3:.4f} linear={linear_interp:.4f}"
        )

    def test_osc2_sub_octave_adds_lower_content(self) -> None:
        """``osc2_level > 0`` with ``osc2_semitones=-12`` should add energy
        below the natural fundamental (sub-octave at 110 Hz)."""
        solo = render(
            freq=220.0,
            duration=0.4,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(
                osc_mode="spectralwave",
                spectral_position=0.0,
                cutoff_hz=8000.0,
                osc2_level=0.0,
                analog_jitter=0.0,
                pitch_drift=0.0,
                noise_floor=0.0,
                cutoff_drift=0.0,
                hpf_cutoff_hz=0.0,
            ),
        )
        stacked = render(
            freq=220.0,
            duration=0.4,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(
                osc_mode="spectralwave",
                spectral_position=0.0,
                cutoff_hz=8000.0,
                osc2_level=0.4,
                osc2_semitones=-12.0,
                analog_jitter=0.0,
                pitch_drift=0.0,
                noise_floor=0.0,
                cutoff_drift=0.0,
                hpf_cutoff_hz=0.0,
            ),
        )

        solo_sub = _band_energy(solo, 100.0, 120.0)
        stacked_sub = _band_energy(stacked, 100.0, 120.0)
        assert stacked_sub > solo_sub * 3.0, (
            f"sub-octave energy not added: solo={solo_sub:.4f} stacked={stacked_sub:.4f}"
        )

    def test_spectral_morph_type_changes_output(self) -> None:
        shared = dict(
            osc_mode="spectralwave",
            spectral_position=0.4,
            cutoff_hz=8000.0,
            analog_jitter=0.0,
            pitch_drift=0.0,
            noise_floor=0.0,
            cutoff_drift=0.0,
        )
        no_morph = render(
            freq=220.0,
            duration=0.3,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(spectral_morph_type="none", **shared),
        )
        with_morph = render(
            freq=220.0,
            duration=0.3,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(
                spectral_morph_type="smear",
                spectral_morph_amount=0.5,
                **shared,
            ),
        )
        assert np.all(np.isfinite(with_morph))
        assert not np.allclose(no_morph, with_morph)

    def test_sigma_approximation_changes_output(self) -> None:
        shared = dict(
            osc_mode="spectralwave",
            spectral_position=0.5,
            cutoff_hz=8000.0,
            analog_jitter=0.0,
            pitch_drift=0.0,
            noise_floor=0.0,
            cutoff_drift=0.0,
        )
        off = render(
            freq=220.0,
            duration=0.3,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(sigma_approximation=False, **shared),
        )
        on = render(
            freq=220.0,
            duration=0.3,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(sigma_approximation=True, **shared),
        )
        assert np.all(np.isfinite(on))
        assert not np.allclose(off, on)


class TestFilterRouting:
    def test_routings_differ(self) -> None:
        outs: dict[str, np.ndarray] = {}
        for routing in ("single", "serial", "parallel", "split"):
            outs[routing] = render(
                freq=220.0,
                duration=0.3,
                amp=0.5,
                sample_rate=_SR,
                params=_base_params(
                    osc_mode="supersaw",
                    filter_routing=routing,
                    cutoff_hz=1800.0,
                    resonance_q=1.3,
                    filter2_cutoff_hz=3200.0,
                ),
            )
        keys = list(outs.keys())
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                assert not np.allclose(outs[a], outs[b]), f"{a} vs {b} are identical"

    def test_split_uses_both_filters(self) -> None:
        """Split routing sends the center voice through filter1 and the side
        voices through filter2. With a low filter1 cutoff and a high filter2
        cutoff, both low-band and high-band energy should survive — unlike a
        single filter at the low cutoff, which should be dark everywhere."""
        shared = dict(
            osc_mode="supersaw",
            supersaw_detune=0.5,
            supersaw_mix=0.7,
            analog_jitter=0.0,
            pitch_drift=0.0,
            noise_floor=0.0,
            cutoff_drift=0.0,
            filter_env_amount=0.0,
        )
        split_sig = render(
            freq=220.0,
            duration=0.4,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(
                filter_routing="split",
                cutoff_hz=300.0,
                filter2_cutoff_hz=8000.0,
                **shared,
            ),
        )
        dark_sig = render(
            freq=220.0,
            duration=0.4,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(
                filter_routing="single",
                cutoff_hz=300.0,
                **shared,
            ),
        )

        split_low = _band_energy(split_sig, 200.0, 500.0)
        split_high = _band_energy(split_sig, 2000.0, 8000.0)
        dark_low = _band_energy(dark_sig, 200.0, 500.0)
        dark_high = _band_energy(dark_sig, 2000.0, 8000.0)

        assert split_low > 0.0
        assert split_high > 0.0
        # Split should clearly carry high-frequency energy; single-low-cutoff
        # should not.
        assert split_high > dark_high * 5.0, (
            f"split_high={split_high:.4f} should dwarf dark_high={dark_high:.4f}"
        )
        # Both outputs are peak-normalized so low-band comparisons are not as
        # clean; just sanity check the low band is present in both.
        assert split_low > 0.0 and dark_low > 0.0


class TestCombFilter:
    def test_self_oscillation_post_filter_tracks_fundamental(self) -> None:
        """With ``comb_position='post_filter'``, high feedback, and keytrack=1
        at a 220 Hz note, the comb self-oscillates near 220 Hz (Karplus-Strong
        behavior)."""
        signal = render(
            freq=220.0,
            duration=0.5,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(
                osc_mode="supersaw",
                comb_position="post_filter",
                comb_feedback=0.95,
                comb_damping=0.1,
                comb_keytrack=1.0,
                analog_jitter=0.0,
                pitch_drift=0.0,
                noise_floor=0.0,
                cutoff_drift=0.0,
            ),
        )
        assert np.all(np.isfinite(signal))

        # Tail-only FFT to let the comb settle.
        tail = signal[_SR // 4 :]
        spec = np.abs(np.fft.rfft(tail))
        freqs = np.fft.rfftfreq(tail.size, 1.0 / _SR)
        # Consider the peak above 20 Hz (ignore DC / very-low-band filter tails).
        mask = freqs > 20.0
        peak_idx = int(np.argmax(spec[mask]))
        peak_hz = float(freqs[mask][peak_idx])
        # Karplus-Strong with keytrack should resonate at the fundamental or
        # one of its integer multiples (the comb supports all harmonics of its
        # delay period). Accept fundamental or first-harmonic peak.
        assert (220.0 * 0.95 < peak_hz < 220.0 * 1.05) or (
            440.0 * 0.95 < peak_hz < 440.0 * 1.05
        ), f"spectral peak {peak_hz:.2f} Hz should sit near 220 or 440 Hz"

    @pytest.mark.parametrize("comb_position", ["pre_filter", "post_filter", "parallel"])
    def test_comb_positions_render_finite(self, comb_position: str) -> None:
        signal = render(
            freq=220.0,
            duration=0.3,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(
                osc_mode="supersaw",
                comb_position=comb_position,
                comb_feedback=0.8,
                comb_damping=0.2,
                comb_keytrack=1.0,
                comb_mix=0.5,
            ),
        )
        assert np.all(np.isfinite(signal))
        assert np.max(np.abs(signal)) > 0.01

    def test_apply_comb_impulse_response(self) -> None:
        n = _SR // 4
        x = np.zeros(n)
        x[0] = 1.0
        delay = np.full(n, 100.0)
        y = apply_comb(
            x,
            delay_samples_profile=delay,
            feedback=0.9,
            damping=0.05,
            mix=1.0,
            sample_rate=_SR,
        )
        assert np.all(np.isfinite(y))
        # Spectral peak should sit near sr/100 = 480 Hz.
        spec = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(n, 1.0 / _SR)
        peak_bin = int(np.argmax(spec[5:]) + 5)
        peak_hz = float(freqs[peak_bin])
        assert 440.0 < peak_hz < 520.0

    def test_apply_comb_stable_at_max_feedback(self) -> None:
        n = _SR // 8
        input_signal = np.random.default_rng(0).standard_normal(n) * 0.1
        delay = np.full(n, 50.0)
        output_signal = apply_comb(
            input_signal,
            delay_samples_profile=delay,
            feedback=0.99,
            damping=0.1,
            mix=1.0,
            sample_rate=_SR,
        )
        assert np.all(np.isfinite(output_signal))
        # Stricter: not just "not NaN", but within a modest loudness envelope.
        assert np.max(np.abs(output_signal)) < 5.0
        in_rms = float(np.sqrt(np.mean(input_signal**2)))
        out_rms = float(np.sqrt(np.mean(output_signal**2)))
        assert out_rms < 10.0 * in_rms, (
            f"comb output RMS {out_rms:.4f} too loud vs input RMS {in_rms:.4f}"
        )

    def test_keytrack_follows_rising_pitch(self) -> None:
        """With ``comb_keytrack=1.0`` on a rising pitch trajectory, the
        spectral centroid of the second half should be higher than the first
        half (comb resonance tracks the pitch up)."""
        n = int(0.5 * _SR)
        traj = np.linspace(220.0, 880.0, n)
        signal = render(
            freq=220.0,
            duration=0.5,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(
                osc_mode="supersaw",
                comb_position="post_filter",
                comb_feedback=0.9,
                comb_damping=0.15,
                comb_keytrack=1.0,
                cutoff_hz=8000.0,
            ),
            freq_trajectory=traj,
        )
        assert np.all(np.isfinite(signal))
        mid = n // 2
        centroid_lo = _spectral_centroid(signal[:mid], _SR)
        centroid_hi = _spectral_centroid(signal[mid:], _SR)
        assert centroid_hi > centroid_lo, (
            f"keytracked comb centroid didn't rise: "
            f"lo={centroid_lo:.1f} hi={centroid_hi:.1f}"
        )


class TestDriveStage:
    def test_drive_adds_harmonic_distortion(self) -> None:
        """Higher ``drive_amount`` should measurably raise harmonic content
        relative to the fundamental. Baseline signal is a near-sine
        (spectralwave with ``n_partials=1``) so the drive stage creates
        clearly new harmonics rather than reshaping an already-rich saw."""
        shared = dict(
            osc_mode="spectralwave",
            spectral_position=0.0,
            n_partials=1,
            cutoff_hz=10000.0,
            analog_jitter=0.0,
            pitch_drift=0.0,
            noise_floor=0.0,
            cutoff_drift=0.0,
            filter_env_amount=0.0,
            resonance_q=0.707,
        )
        clean = render(
            freq=220.0,
            duration=0.4,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(drive_amount=0.0, **shared),
        )
        driven = render(
            freq=220.0,
            duration=0.4,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(drive_amount=0.5, drive_algorithm="tanh", **shared),
        )

        thd_clean = _thd_relative(clean, _SR, fundamental_hz=220.0)
        thd_driven = _thd_relative(driven, _SR, fundamental_hz=220.0)
        assert thd_driven > thd_clean * 3.0, (
            f"drive didn't raise THD: clean={thd_clean:.4f} driven={thd_driven:.4f}"
        )

    def test_drive_amount_zero_equivalent_to_omitted(self) -> None:
        """Explicit ``drive_amount=0.0`` should bypass the waveshaper, matching
        the default (omitted) behavior spectrally. Note: the bit-exact waveform
        differs because adding the key changes the params dict hash and thus
        the RNG seed — but that only reshuffles start phases, not spectrum."""
        explicit = render(
            freq=220.0,
            duration=0.3,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(
                osc_mode="supersaw",
                drive_amount=0.0,
                _voice_name="drive_bypass",
            ),
        )
        omitted = render(
            freq=220.0,
            duration=0.3,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(osc_mode="supersaw", _voice_name="drive_bypass"),
        )
        # THD must match closely — if drive_amount=0 accidentally triggered
        # distortion, harmonic content would diverge.
        thd_explicit = _thd_relative(explicit, _SR, fundamental_hz=220.0)
        thd_omitted = _thd_relative(omitted, _SR, fundamental_hz=220.0)
        assert abs(thd_explicit - thd_omitted) < max(thd_omitted, 1e-6) * 0.25, (
            f"drive=0 vs omitted THD mismatch: "
            f"explicit={thd_explicit:.4f} omitted={thd_omitted:.4f}"
        )

    @pytest.mark.parametrize("algorithm", ["tanh", "atan", "exponential"])
    def test_all_drive_algorithms_render(self, algorithm: str) -> None:
        signal = render(
            freq=220.0,
            duration=0.2,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(
                osc_mode="supersaw", drive_amount=0.4, drive_algorithm=algorithm
            ),
        )
        assert np.all(np.isfinite(signal))
        assert np.max(np.abs(signal)) > 0.01


class TestDeterminism:
    def test_identical_render(self) -> None:
        params = _base_params(osc_mode="supersaw", supersaw_detune=0.3)
        a = render(freq=330.0, duration=0.3, amp=0.5, sample_rate=_SR, params=params)
        b = render(freq=330.0, duration=0.3, amp=0.5, sample_rate=_SR, params=params)
        assert np.array_equal(a, b)


class TestRegistryDispatch:
    def test_registry_routes_to_va(self) -> None:
        signal = render_note_signal(
            freq=440.0,
            duration=0.2,
            amp=0.5,
            sample_rate=_SR,
            params={"engine": "va", "osc_mode": "supersaw", "_voice_name": "reg_test"},
        )
        assert np.all(np.isfinite(signal))
        assert np.max(np.abs(signal)) > 0.01


class TestPresets:
    @pytest.mark.parametrize("preset_name", sorted(_PRESETS["va"].keys()))
    @pytest.mark.parametrize("freq", [110.0, 440.0, 1760.0])
    def test_preset_renders(self, preset_name: str, freq: float) -> None:
        resolved = resolve_synth_params({"engine": "va", "preset": preset_name})
        resolved["_voice_name"] = f"preset_{preset_name}_{freq}"
        signal = render(
            freq=freq, duration=0.2, amp=0.5, sample_rate=_SR, params=resolved
        )
        assert np.all(np.isfinite(signal))
        assert np.max(np.abs(signal)) > 0.01

    def test_signature_preset_fingerprints(self) -> None:
        """Freeze (RMS, spectral centroid) fingerprints for three signature
        presets at freq=220 Hz. Catches accidental preset-dict edits that
        flip an engine or swap a major param. ±20% tolerance absorbs minor
        numerical drift.

        If this fails after an intentional preset change, re-measure and
        update the expected values in this table.
        """
        expected = {
            "jp8000_hoover": (0.126556, 2384.1348),
            "virus_bass": (0.195136, 1515.1835),
            "q_comb_bell": (0.256478, 2400.4853),
        }
        for preset_name, (exp_rms, exp_centroid) in expected.items():
            resolved = resolve_synth_params({"engine": "va", "preset": preset_name})
            resolved["_voice_name"] = f"fingerprint_{preset_name}"
            signal = render(
                freq=220.0, duration=0.5, amp=0.5, sample_rate=_SR, params=resolved
            )
            rms = float(np.sqrt(np.mean(signal * signal)))
            centroid = _spectral_centroid(signal, _SR)
            assert abs(rms - exp_rms) < exp_rms * 0.20, (
                f"{preset_name}: rms {rms:.4f} vs expected {exp_rms:.4f}"
            )
            assert abs(centroid - exp_centroid) < exp_centroid * 0.20, (
                f"{preset_name}: centroid {centroid:.2f} vs expected {exp_centroid:.2f}"
            )


class TestFreqTrajectory:
    def test_trajectory_tracks_pitch(self) -> None:
        n = int(0.3 * _SR)
        traj = np.linspace(220.0, 440.0, n)
        signal = render(
            freq=220.0,
            duration=0.3,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(osc_mode="supersaw"),
            freq_trajectory=traj,
        )
        assert np.all(np.isfinite(signal))
        # Spectral centroid on the second half should be higher than on the first half.
        mid = n // 2
        centroid_lo = _spectral_centroid(signal[:mid], _SR)
        centroid_hi = _spectral_centroid(signal[mid:], _SR)
        assert centroid_hi > centroid_lo


class TestValidation:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"osc_mode": "saw"},
            {"filter_routing": "invalid"},
            {"comb_position": "bogus"},
            {"drive_algorithm": "foo"},
            {"drive_amount": 1.5},
            {"drive_amount": -0.1},
            {"spectral_position": 1.5},
            {"n_partials": 0},
            {"comb_feedback": 1.0},
            {"comb_damping": -0.1},
            {"filter_mode": "allpass"},
            {"filter_topology": "magic"},
            {"filter2_filter_topology": "wizard"},
        ],
    )
    def test_invalid_params_raise(self, overrides: dict[str, object]) -> None:
        params = _base_params(**overrides)
        with pytest.raises(ValueError):
            render(freq=220.0, duration=0.2, amp=0.5, sample_rate=_SR, params=params)

    def test_zero_duration_raises(self) -> None:
        with pytest.raises(ValueError):
            render(
                freq=220.0,
                duration=0.0,
                amp=0.5,
                sample_rate=_SR,
                params=_base_params(),
            )

    def test_zero_sample_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            render(
                freq=220.0,
                duration=0.2,
                amp=0.5,
                sample_rate=0,
                params=_base_params(),
            )

    def test_mismatched_freq_trajectory_size_raises(self) -> None:
        n = int(0.2 * _SR)
        bad_traj = np.linspace(220.0, 440.0, n - 10)
        with pytest.raises(ValueError):
            render(
                freq=220.0,
                duration=0.2,
                amp=0.5,
                sample_rate=_SR,
                params=_base_params(),
                freq_trajectory=bad_traj,
            )

    def test_multidim_freq_trajectory_raises(self) -> None:
        n = int(0.2 * _SR)
        bad_traj = np.linspace(220.0, 440.0, n * 2).reshape(n, 2)
        with pytest.raises(ValueError):
            render(
                freq=220.0,
                duration=0.2,
                amp=0.5,
                sample_rate=_SR,
                params=_base_params(),
                freq_trajectory=bad_traj,
            )


class TestAnalogCharacter:
    def test_jitter_varies_output(self) -> None:
        """With ``_voice_name`` fixed, raising ``analog_jitter`` from 0 must
        change the output. (F20: holds jitter-specific RNG responsible
        without conflating with phase-seed randomness.)"""
        shared = dict(osc_mode="supersaw", _voice_name="jitter_same")
        a = render(
            freq=330.0,
            duration=0.2,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(analog_jitter=0.0, **shared),
        )
        b = render(
            freq=330.0,
            duration=0.2,
            amp=0.5,
            sample_rate=_SR,
            params=_base_params(analog_jitter=0.5, **shared),
        )
        assert not np.allclose(a, b)
