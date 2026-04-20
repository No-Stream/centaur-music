"""Per-note voice-distortion slot tests for the polyblep engine.

These tests verify that the ``voice_dist_*`` engine params wire the
:func:`code_musics.engines._voice_dist.apply_voice_dist` helper into the
per-note render path correctly, and — crucially — that distorting each
note independently before summing preserves chord clarity in a way
post-sum distortion cannot.  That pre-sum location is the whole point
of RePro-5-style polyphonic distortion.
"""

from __future__ import annotations

import numpy as np

from code_musics.engines._waveshaper import apply_waveshaper
from code_musics.engines.polyblep import render

SR = 44100


def _common_params(**overrides: object) -> dict[str, object]:
    """Disable all analog-noise surfaces so renders are purely deterministic."""
    base = {
        "waveform": "saw",
        "cutoff_hz": 3000.0,
        "resonance_q": 0.707,
        "filter_drive": 0.0,
        "analog_jitter": 0.0,
        "pitch_drift": 0.0,
        "cutoff_drift": 0.0,
        "noise_floor": 0.0,
        "osc_asymmetry": 0.0,
        "osc_softness": 0.0,
        "osc_dc_offset": 0.0,
        "osc_shape_drift": 0.0,
    }
    base.update(overrides)
    return base


class TestVoiceDistOffIsNoop:
    def test_off_mode_matches_absent_param(self) -> None:
        """`voice_dist_mode="off"` must be bit-identical to omitting the params."""
        baseline = render(
            freq=220.0,
            duration=0.3,
            amp=0.8,
            sample_rate=SR,
            params=_common_params(),
        )
        explicit_off = render(
            freq=220.0,
            duration=0.3,
            amp=0.8,
            sample_rate=SR,
            params=_common_params(
                voice_dist_mode="off",
                voice_dist_drive=0.5,
                voice_dist_mix=1.0,
                voice_dist_tone=0.0,
            ),
        )
        assert baseline.shape == explicit_off.shape
        np.testing.assert_array_equal(baseline, explicit_off)

    def test_drive_zero_is_noop(self) -> None:
        """Any mode with drive=0.0 must short-circuit to a clean passthrough."""
        baseline = render(
            freq=220.0,
            duration=0.3,
            amp=0.8,
            sample_rate=SR,
            params=_common_params(),
        )
        zero_drive = render(
            freq=220.0,
            duration=0.3,
            amp=0.8,
            sample_rate=SR,
            params=_common_params(
                voice_dist_mode="hard_clip",
                voice_dist_drive=0.0,
            ),
        )
        np.testing.assert_array_equal(baseline, zero_drive)


class TestVoiceDistPreSumBeatsPostSum:
    """The defining test for per-note pre-sum distortion.

    A triad distorted per-note-then-summed should exhibit less
    intermodulation (IMD: spectral energy at sum/difference frequencies
    of chord tones) than the same dry sum clipped post-mix.
    """

    def test_chord_imd_pre_sum_vs_post_sum(self) -> None:
        dur = 0.5
        sr = SR
        # A major triad: fundamentals at 220, 275, 330 Hz (JI 4:5:6).
        # These fundamentals make IMD bins easy to read in the FFT —
        # sum/diff products land on identifiable bins away from the
        # harmonic stack of any single tone.
        freqs = (220.0, 275.0, 330.0)

        pre_sum = np.zeros(int(dur * sr), dtype=np.float64)
        dry_sum = np.zeros(int(dur * sr), dtype=np.float64)
        for f in freqs:
            dry_note = render(
                freq=f,
                duration=dur,
                amp=0.6,
                sample_rate=sr,
                params=_common_params(cutoff_hz=4500.0),
            )
            dry_sum += dry_note
            distorted_note = render(
                freq=f,
                duration=dur,
                amp=0.6,
                sample_rate=sr,
                params=_common_params(
                    cutoff_hz=4500.0,
                    voice_dist_mode="hard_clip",
                    voice_dist_drive=1.5,
                ),
            )
            pre_sum += distorted_note

        # Post-sum reference: same dry sum passed through the canonical
        # hard-clip waveshaper with the same internal drive the voice_dist
        # helper would apply (drive=1.5 -> internal 0.75).  Calling
        # ``apply_waveshaper`` directly keeps this test honest if the
        # gain law in ``_waveshaper`` ever gets retuned.
        peak = np.max(np.abs(dry_sum))
        normalized = dry_sum / max(peak, 1e-9)
        post_sum = apply_waveshaper(normalized, algorithm="hard_clip", drive=0.75)

        # Focus on a set of IMD frequencies: pairwise |f_i +/- f_j|
        # excluding near-harmonic coincidences.  220+275=495, 220-275=55,
        # 275+330=605, 275-330=55 (same), 220+330=550, 330-220=110.
        # 110 and 55 collide with partials of 220 Hz harmonics so we use
        # the sum products 495, 550, 605 as cleanly-identifiable IMD.
        imd_freqs = (495.0, 550.0, 605.0)

        def energy_at(signal: np.ndarray, freq_hz: float, bw_hz: float = 6.0) -> float:
            spec = np.abs(np.fft.rfft(signal))
            hz_per_bin = sr / signal.size
            lo = max(0, int((freq_hz - bw_hz) / hz_per_bin))
            hi = min(spec.size, int((freq_hz + bw_hz) / hz_per_bin) + 1)
            return float(np.sum(spec[lo:hi] ** 2))

        pre_imd = sum(energy_at(pre_sum, f) for f in imd_freqs)
        post_imd = sum(energy_at(post_sum, f) for f in imd_freqs)

        # Post-sum clipping must create more IMD at those sum bins than
        # pre-sum distortion does.  A clear dB margin proves the point.
        eps = 1e-18
        ratio_db = 10.0 * np.log10((post_imd + eps) / (pre_imd + eps))
        assert ratio_db > 3.0, (
            f"expected post-sum IMD >= pre-sum by a clear margin, got "
            f"{ratio_db:.2f} dB (pre={pre_imd:.3e}, post={post_imd:.3e})"
        )


class TestVoiceDistModesSmoke:
    def test_each_mode_produces_finite_nonsilent_output(self) -> None:
        for mode in (
            "soft_clip",
            "hard_clip",
            "foldback",
            "corrode",
            "saturation",
            "preamp",
        ):
            out = render(
                freq=220.0,
                duration=0.2,
                amp=0.8,
                sample_rate=SR,
                params=_common_params(
                    voice_dist_mode=mode,
                    voice_dist_drive=0.8,
                    voice_dist_mix=1.0,
                ),
            )
            assert np.all(np.isfinite(out)), f"mode={mode} produced non-finite output"
            assert np.max(np.abs(out)) > 0.0, f"mode={mode} produced silence"
