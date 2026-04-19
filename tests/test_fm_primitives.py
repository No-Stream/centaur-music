"""Tests for fm_modulate_2op and phase_modulate_nop DSP primitives."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._dsp_utils import (
    fm_modulate,
    fm_modulate_2op,
    phase_modulate_nop,
)

SAMPLE_RATE = 44100


def _make_carrier_profile(freq: float, duration: float) -> np.ndarray:
    n_samples = int(duration * SAMPLE_RATE)
    return np.full(n_samples, freq, dtype=np.float64)


def _spectral_centroid(signal: np.ndarray, sr: int) -> float:
    spec = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.shape[0], d=1.0 / sr)
    total = spec.sum()
    if total <= 0:
        return 0.0
    return float((spec * freqs).sum() / total)


# ---------------------------------------------------------------------------
# fm_modulate_2op tests
# ---------------------------------------------------------------------------


class TestFmModulate2Op:
    def test_fm_modulate_2op_zero_mod_index_matches_sine(self) -> None:
        freq = 440.0
        duration = 0.5
        carrier = _make_carrier_profile(freq, duration)

        out = fm_modulate_2op(
            carrier,
            mod1_ratio=1.0,
            mod1_index=0.0,
            mod2_ratio=1.0,
            mod2_index=0.0,
            sample_rate=SAMPLE_RATE,
            mod1_feedback=0.0,
            mod2_feedback=0.0,
            carrier_feedback=0.0,
        )

        n = out.shape[0]
        t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
        expected = np.sin(2.0 * np.pi * freq * t)

        rms_err = float(np.sqrt(np.mean((out - expected) ** 2)))
        assert rms_err < 1e-6, f"RMS error {rms_err} not below 1e-6"

    def test_fm_modulate_2op_produces_sidebands(self) -> None:
        freq = 220.0
        duration = 1.0
        carrier = _make_carrier_profile(freq, duration)

        out = fm_modulate_2op(
            carrier,
            mod1_ratio=1.0,
            mod1_index=2.0,
            mod2_ratio=3.5,
            mod2_index=1.0,
            sample_rate=SAMPLE_RATE,
        )

        n = out.shape[0]
        t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
        pure_sine = np.sin(2.0 * np.pi * freq * t)

        spec_fm = np.abs(np.fft.rfft(out))
        spec_sine = np.abs(np.fft.rfft(pure_sine))

        # Total spectral energy outside the single carrier bin should be
        # much larger for the FM signal.
        carrier_bin = int(round(freq * n / SAMPLE_RATE))
        off_carrier_fm = spec_fm.sum() - spec_fm[carrier_bin]
        off_carrier_sine = spec_sine.sum() - spec_sine[carrier_bin]

        assert off_carrier_fm > 10 * off_carrier_sine, (
            f"FM off-carrier energy {off_carrier_fm} not substantially greater "
            f"than pure-sine off-carrier energy {off_carrier_sine}"
        )
        assert np.isfinite(out).all()

    def test_fm_modulate_2op_index_envelope_decays(self) -> None:
        freq = 200.0
        duration = 1.0
        carrier = _make_carrier_profile(freq, duration)
        n = carrier.shape[0]

        t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
        index_env = np.exp(-6.0 * t)  # fast exponential decay

        out = fm_modulate_2op(
            carrier,
            mod1_ratio=2.0,
            mod1_index=4.0,
            mod2_ratio=5.0,
            mod2_index=3.0,
            sample_rate=SAMPLE_RATE,
            index_envelope=index_env,
        )

        quarter = n // 4
        early = out[:quarter]
        late = out[-quarter:]

        early_centroid = _spectral_centroid(early, SAMPLE_RATE)
        late_centroid = _spectral_centroid(late, SAMPLE_RATE)

        assert late_centroid < early_centroid, (
            f"Late centroid {late_centroid} should be below early "
            f"centroid {early_centroid} as the tone de-brightens."
        )

    def test_fm_modulate_2op_feedback_adds_harmonics(self) -> None:
        freq = 300.0
        duration = 0.5
        carrier = _make_carrier_profile(freq, duration)

        no_fb = fm_modulate_2op(
            carrier,
            mod1_ratio=1.0,
            mod1_index=1.0,
            mod2_ratio=1.0,
            mod2_index=0.0,
            sample_rate=SAMPLE_RATE,
            mod1_feedback=0.0,
        )
        with_fb = fm_modulate_2op(
            carrier,
            mod1_ratio=1.0,
            mod1_index=1.0,
            mod2_ratio=1.0,
            mod2_index=0.0,
            sample_rate=SAMPLE_RATE,
            mod1_feedback=0.5,
        )

        centroid_no_fb = _spectral_centroid(no_fb, SAMPLE_RATE)
        centroid_fb = _spectral_centroid(with_fb, SAMPLE_RATE)

        assert centroid_fb > centroid_no_fb, (
            f"Feedback-on centroid {centroid_fb} not above feedback-off "
            f"centroid {centroid_no_fb}"
        )


# ---------------------------------------------------------------------------
# phase_modulate_nop tests
# ---------------------------------------------------------------------------


class TestPhaseModulateNOp:
    def test_phase_modulate_nop_4op_cymbal_non_silent(self) -> None:
        freq = 440.0
        duration = 1.0
        carrier = _make_carrier_profile(freq, duration)

        op_ratios = np.array([1.0, 1.42, 2.77, 5.4], dtype=np.float64)
        op_indices = np.array([3.0, 2.5, 2.0, 1.5], dtype=np.float64)
        op_feedbacks = np.zeros(4, dtype=np.float64)

        out = phase_modulate_nop(
            carrier,
            op_ratios=op_ratios,
            op_indices=op_indices,
            op_feedbacks=op_feedbacks,
            sample_rate=SAMPLE_RATE,
        )

        assert np.isfinite(out).all(), "Output contains NaN/Inf"
        peak = float(np.max(np.abs(out)))
        assert 0.0 < peak < 5.0, f"Peak {peak} out of expected range"
        rms = float(np.sqrt(np.mean(out**2)))
        assert rms > 1e-3, f"Output RMS {rms} too close to silence"

    def test_phase_modulate_nop_single_op_matches_fm_modulate(self) -> None:
        freq = 220.0
        duration = 0.5
        carrier = _make_carrier_profile(freq, duration)

        mod_ratio = 2.0
        mod_index = 1.5
        feedback = 0.25

        nop_out = phase_modulate_nop(
            carrier,
            op_ratios=np.array([mod_ratio], dtype=np.float64),
            op_indices=np.array([mod_index], dtype=np.float64),
            op_feedbacks=np.array([feedback], dtype=np.float64),
            sample_rate=SAMPLE_RATE,
        )

        ref_out = fm_modulate(
            carrier,
            mod_ratio=mod_ratio,
            mod_index=mod_index,
            sample_rate=SAMPLE_RATE,
            feedback=feedback,
        )

        rms_err = float(np.sqrt(np.mean((nop_out - ref_out) ** 2)))
        assert rms_err < 1e-4, (
            f"Single-op phase_modulate_nop diverged from fm_modulate: "
            f"RMS error {rms_err}"
        )

    def test_phase_modulate_nop_envelope_shaping(self) -> None:
        freq = 330.0
        duration = 1.0
        carrier = _make_carrier_profile(freq, duration)
        n = carrier.shape[0]

        op_ratios = np.array([1.0, 2.0], dtype=np.float64)
        op_indices = np.array([2.0, 2.0], dtype=np.float64)
        op_feedbacks = np.zeros(2, dtype=np.float64)

        env = np.ones((2, n), dtype=np.float64)
        t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
        env[0, :] = np.exp(-4.0 * t)

        out = phase_modulate_nop(
            carrier,
            op_ratios=op_ratios,
            op_indices=op_indices,
            op_feedbacks=op_feedbacks,
            sample_rate=SAMPLE_RATE,
            op_envelopes=env,
        )

        quarter = n // 4
        early_rms = float(np.sqrt(np.mean(out[:quarter] ** 2)))
        late_rms = float(np.sqrt(np.mean(out[-quarter:] ** 2)))

        assert late_rms < early_rms, (
            f"Late RMS {late_rms} should be below early RMS {early_rms} "
            "when op 0 envelope decays."
        )

    def test_phase_modulate_nop_mismatched_arrays_raises(self) -> None:
        freq = 440.0
        duration = 0.1
        carrier = _make_carrier_profile(freq, duration)

        with pytest.raises(ValueError, match="op_indices length"):
            phase_modulate_nop(
                carrier,
                op_ratios=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
                op_indices=np.array([1.0, 2.0, 3.0], dtype=np.float64),
                op_feedbacks=np.zeros(4, dtype=np.float64),
                sample_rate=SAMPLE_RATE,
            )
