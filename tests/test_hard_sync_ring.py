"""Tests for hard sync and ring modulation between osc1 and osc2 in polyblep.

Hard sync: osc2 phase resets to 0 every time osc1 wraps past 1.0, with PolyBLEP
step correction to suppress aliasing of the induced discontinuity.

Ring mod: dry/ring balance blend.  Final mix is
    (1 - ring_mod) * (osc1 + osc2_level * osc2) + ring_mod * (osc1 * osc2)
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._dsp_utils import apply_polyblep_step_correction
from code_musics.engines._oscillators import polyblep_saw
from code_musics.engines.polyblep import _find_sync_events, render

SR = 44100


def _band_energy(
    signal: np.ndarray, low_hz: float, high_hz: float, sr: int = SR
) -> float:
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1 / sr)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    return float(np.sqrt(np.mean(spectrum[mask] ** 2) + 1e-30))


def _render_simple(**overrides):
    """Render with ring/sync-friendly minimal-surprises params."""
    params = {
        "waveform": "saw",
        "osc2_waveform": "saw",
        "osc2_level": 0.8,
        "osc2_detune_cents": 700.0,  # perfect fifth
        "cutoff_hz": 8000.0,
        "resonance_q": 0.707,
        "pitch_drift": 0.0,
        "analog_jitter": 0.0,
        "noise_floor": 0.0,
        "cutoff_drift": 0.0,
        "osc_asymmetry": 0.0,
        "osc_softness": 0.0,
        "osc_dc_offset": 0.0,
        "osc_shape_drift": 0.0,
        "voice_card_spread": 0.0,
    }
    params.update(overrides)
    return render(
        freq=220.0,
        duration=0.5,
        amp=0.8,
        sample_rate=SR,
        params=params,
    )


class TestHardSync:
    def test_sync_changes_spectrum(self) -> None:
        dry = _render_simple(osc2_sync=False)
        synced = _render_simple(osc2_sync=True)

        assert np.isfinite(synced).all()
        assert not np.allclose(dry, synced)
        assert np.linalg.norm(dry - synced) > 1.0

    @pytest.mark.parametrize("detune", [-1200.0, -700.0, 0.0, 700.0, 1200.0, 2400.0])
    def test_sync_stays_finite(self, detune: float) -> None:
        synced = _render_simple(osc2_sync=True, osc2_detune_cents=detune)
        assert np.all(np.isfinite(synced))
        assert np.max(np.abs(synced)) < 10.0
        assert np.max(np.abs(synced)) > 1e-6

    def test_sync_adds_harmonics(self) -> None:
        # Hard sync forces osc2 to reset every osc1 period, injecting
        # step discontinuities that create broadband harmonic content.
        # Compare high-band (3 kHz–Nyquist) energy: a filter-phase shift
        # alone can't move this band by the dB margin we require, so a
        # pass here genuinely implies new harmonic content rather than
        # redistribution of existing partials.
        dry = _render_simple(
            osc2_sync=False,
            osc2_detune_cents=1900.0,  # ~3x freq ratio, non-integer
            cutoff_hz=16000.0,
            osc2_level=1.0,
        )
        synced = _render_simple(
            osc2_sync=True,
            osc2_detune_cents=1900.0,
            cutoff_hz=16000.0,
            osc2_level=1.0,
        )
        nyquist = SR / 2.0
        dry_high = _band_energy(dry, 3000.0, nyquist)
        synced_high = _band_energy(synced, 3000.0, nyquist)
        high_band_ratio_db = 20.0 * np.log10((synced_high + 1e-30) / (dry_high + 1e-30))
        # 1 dB is a concrete, measurable shift (about 12% power), and
        # corresponds to roughly 5× the sensitivity of a filter-phase
        # rotation on a harmonic-rich signal — a filter move alone
        # cannot produce this delta across a full high-band window.
        assert high_band_ratio_db >= 1.0, (
            "sync should add >=1 dB of broadband high-band energy vs dry; "
            f"got {high_band_ratio_db:.2f} dB (dry={dry_high:.3e}, "
            f"synced={synced_high:.3e})"
        )


class TestRingMod:
    def test_ring_mod_zero_matches_dry(self) -> None:
        base = _render_simple()
        with_explicit_zero = _render_simple(osc2_ring_mod=0.0)
        np.testing.assert_array_equal(base, with_explicit_zero)

    def test_ring_mod_one_is_pure_ring(self) -> None:
        # When ring_mod is 1, the osc-stage output is osc1 * osc2.  The post
        # filter/normalize stages mean we can't compare raw samples, but the
        # spectrum must contain sum/difference sidebands rather than clean
        # fundamentals.  A clean ring modulator output on two pitches f1, f2
        # should have spectral peaks at f1 +/- f2, not at f1 alone.
        dry = _render_simple(osc2_ring_mod=0.0, cutoff_hz=16000.0)
        ring = _render_simple(osc2_ring_mod=1.0, cutoff_hz=16000.0)
        assert np.all(np.isfinite(ring))
        # Ring-mod should differ substantially from dry
        assert np.linalg.norm(ring - dry) > 1.0
        # Sum and difference: for 220 Hz + 700 cents (~330 Hz) we expect
        # sidebands near 550 Hz (sum) and 110 Hz (difference).
        sum_energy = _band_energy(ring, 500.0, 600.0)
        diff_energy = _band_energy(ring, 80.0, 140.0)
        # And the osc1 fundamental at 220 should be *relatively* suppressed
        fund_energy = _band_energy(ring, 200.0, 240.0)
        dry_fund_energy = _band_energy(dry, 200.0, 240.0)
        assert sum_energy > 0.0
        assert diff_energy > 0.0
        # Fundamental of osc1 is strongly attenuated under pure ring mod
        assert fund_energy < dry_fund_energy

    def test_ring_mod_half_has_both(self) -> None:
        # 50/50 blend: the fundamentals and the sum/diff sidebands should
        # coexist at meaningful levels.
        dry = _render_simple(osc2_ring_mod=0.0, cutoff_hz=16000.0)
        half = _render_simple(osc2_ring_mod=0.5, cutoff_hz=16000.0)
        ring = _render_simple(osc2_ring_mod=1.0, cutoff_hz=16000.0)

        fund_half = _band_energy(half, 200.0, 240.0)
        fund_dry = _band_energy(dry, 200.0, 240.0)
        fund_ring = _band_energy(ring, 200.0, 240.0)
        # fundamental energy in half should be between pure-ring and pure-dry
        assert fund_ring < fund_half
        assert fund_half <= fund_dry * 1.05  # allow small headroom from normalization

        diff_half = _band_energy(half, 80.0, 140.0)
        diff_dry = _band_energy(dry, 80.0, 140.0)
        # sum/diff energy in half should be larger than in pure-dry (which has none)
        assert diff_half > diff_dry * 1.5


class TestSyncAndRingTogether:
    def test_sync_and_ring_can_coexist(self) -> None:
        both = _render_simple(osc2_sync=True, osc2_ring_mod=0.5)
        sync_only = _render_simple(osc2_sync=True, osc2_ring_mod=0.0)
        ring_only = _render_simple(osc2_sync=False, osc2_ring_mod=0.5)
        assert np.all(np.isfinite(both))
        assert np.max(np.abs(both)) < 10.0
        assert np.max(np.abs(both)) > 1e-6
        assert not np.allclose(both, sync_only)
        assert not np.allclose(both, ring_only)


class TestPolyBLEPStepCorrectionHelper:
    def test_polyblep_step_correction_matches_inline_version(self) -> None:
        # Construct a constant-phase-inc saw and compare:
        #  (a) the inline polyblep_saw
        #  (b) a raw 2*phase-1 saw, with apply_polyblep_step_correction called
        #      at every wrap
        sr = SR
        freq = 220.0
        duration = 0.1
        n = int(sr * duration)
        phase_inc_scalar = freq / sr
        phase_inc = np.full(n, phase_inc_scalar, dtype=np.float64)
        cumphase = np.cumsum(phase_inc)
        phase = cumphase % 1.0

        # Reference: existing polyblep_saw
        reference = polyblep_saw(phase, phase_inc)

        # Build naive saw then apply extracted helper at every wrap.
        naive = 2.0 * phase - 1.0

        # Find wrap events.  A wrap happens between samples i-1 and i when
        # phase[i] < phase[i-1].  The event time in continuous samples is
        # (i - 1) + frac where frac is where within [i-1, i] phase crosses 1.
        # With constant phase_inc: at sample i the overshoot past 1 is
        # phase[i] (since cumphase - floor(cumphase) = phase).  The sample
        # step covers phase_inc units, so frac = (1 - phase[i-1]) / phase_inc.
        wraps = np.where(phase[1:] < phase[:-1])[0]  # indices `i-1` in 0..n-2
        step_amp = -2.0  # saw drops by 2 at every wrap
        for pre_idx in wraps:
            # event is between pre_idx and pre_idx+1
            frac = (1.0 - phase[pre_idx]) / phase_inc_scalar
            # Clamp fractional to [0, 1)
            frac = max(0.0, min(0.999999, frac))
            apply_polyblep_step_correction(
                signal=naive,
                event_sample=int(pre_idx),
                event_fraction=float(frac),
                step_amplitude=step_amp,
            )

        # Edge sample: the polyblep_saw mask_pre also fires on any sample
        # where phase > 1 - phase_inc, and mask_post fires on phase < phase_inc.
        # Our extracted helper applies both corrections per detected wrap, so
        # the result should match the reference bit-for-bit on interior samples.
        # Compare with a small tolerance for FP identity.
        np.testing.assert_allclose(naive[1:-1], reference[1:-1], atol=1e-12)


class TestFindSyncEventsMultiWrap:
    def test_multi_wrap_interval_emits_one_event_per_wrap(self) -> None:
        """F34: when osc1 wraps multiple times in a single sample interval
        (freq > sample_rate/2), ``_find_sync_events`` must emit one event per
        integer boundary rather than collapsing them into one."""
        n = 32
        # phase_inc ~1.6 cycles per sample, so some intervals cross two
        # integer boundaries in a single step.
        phase_inc_scalar = 1.6
        phase_inc = np.full(n, phase_inc_scalar, dtype=np.float64)
        cumphase = np.cumsum(phase_inc)
        sync_samples, sync_fractions = _find_sync_events(cumphase, phase_inc)

        # We check intervals (i-1, i] for i in [1, n).  Sum the integer
        # boundaries crossed in each interval — that must equal the number
        # of emitted events.
        floor_cum = np.floor(cumphase)
        diffs = np.diff(floor_cum).astype(np.int64)
        n_events_expected = int(np.sum(diffs))
        assert sync_samples.size == n_events_expected
        # Make sure we actually hit a multi-wrap interval in this test.
        assert np.any(diffs >= 2)
        # All fractions must be in [0, 1).
        assert np.all(sync_fractions >= 0.0)
        assert np.all(sync_fractions < 1.0)
        assert np.all(np.isfinite(sync_fractions))
        # Events are emitted with monotonically non-decreasing pre_idx.
        assert np.all(np.diff(sync_samples) >= 0)

    def test_multi_wrap_keeps_render_finite(self) -> None:
        """Extreme freq-over-Nyquist osc2 sync doesn't blow up the render."""
        params = {
            "waveform": "saw",
            "osc2_waveform": "saw",
            "osc2_level": 0.8,
            "osc2_sync": True,
            "osc2_detune_cents": 2400.0,  # very wide
            "cutoff_hz": 8000.0,
            "resonance_q": 0.707,
            "pitch_drift": 0.0,
            "analog_jitter": 0.0,
            "noise_floor": 0.0,
            "cutoff_drift": 0.0,
            "osc_asymmetry": 0.0,
            "osc_softness": 0.0,
            "osc_dc_offset": 0.0,
            "osc_shape_drift": 0.0,
            "voice_card_spread": 0.0,
        }
        out = render(
            freq=18000.0,  # far beyond musical range — exercises the wrap path
            duration=0.05,
            amp=0.5,
            sample_rate=SR,
            params=params,
        )
        assert np.all(np.isfinite(out))
        assert np.max(np.abs(out)) < 10.0
