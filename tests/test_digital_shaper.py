"""Tests for the digital-character waveshaper algorithms.

Covers the Machinedrum/SP-1200-flavoured ``bit_crush``, ``rate_reduce``, and
``digital_clip`` algorithms added to :mod:`code_musics.engines._waveshaper`.
"""

from __future__ import annotations

import numpy as np

from code_musics.engines._waveshaper import apply_waveshaper


def _sine(freq: float = 440.0, duration: float = 0.05, sr: int = 44100) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float64)
    return np.sin(2.0 * np.pi * freq * t)


def test_bit_crush_2_bits_produces_few_levels() -> None:
    rng = np.random.default_rng(seed=42)
    signal = rng.uniform(-1.0, 1.0, size=4096).astype(np.float64)
    # oversample=1 to inspect the raw quantization levels; the default
    # auto-upgrade to oversample=2 would lowpass the staircase into a
    # continuous range of values.
    result = apply_waveshaper(
        signal, algorithm="bit_crush", drive=0.5, bit_depth=2.0, oversample=1
    )
    # After RMS normalization the raw quantization levels get scaled, but the
    # number of distinct values should still stay small (<= 6 accounts for a
    # tiny bit of rounding slack around the nominal 4 levels).
    unique_values = np.unique(np.round(result, decimals=6))
    assert unique_values.size <= 6, (
        f"expected <=6 unique levels, got {unique_values.size}"
    )


def test_bit_crush_16_bits_near_transparent() -> None:
    signal = _sine()
    result = apply_waveshaper(signal, algorithm="bit_crush", drive=0.0, bit_depth=16.0)
    # With 16-bit quantization and unity drive, the shaper should be close to
    # transparent — RMS error should be well under the 2-bit case.
    rms_err = float(np.sqrt(np.mean((result - signal) ** 2)))
    assert rms_err < 1e-3, f"expected near-transparent RMS error, got {rms_err}"


def test_rate_reduce_staircase() -> None:
    signal = _sine(freq=440.0, duration=0.01)
    # oversample=1 to preserve the exact staircase blocks; the default
    # oversample=2 auto-upgrade would half the effective block width after
    # downsampling.
    result = apply_waveshaper(
        signal, algorithm="rate_reduce", drive=0.0, reduce_ratio=8.0, oversample=1
    )
    # Walk the signal in blocks of 8 and assert each block is flat.
    n_full_blocks = result.shape[0] // 8
    for block_idx in range(n_full_blocks):
        start = block_idx * 8
        block = result[start : start + 8]
        # Every sample in the block should equal the first (the held value).
        np.testing.assert_allclose(
            block,
            np.full_like(block, block[0]),
            atol=1e-9,
            err_msg=f"block {block_idx} is not flat: {block}",
        )


def test_digital_clip_asymmetric() -> None:
    ramp = np.linspace(-2.0, 2.0, 4096, dtype=np.float64)
    # oversample=1 so the exact clipped rails are preserved; the default
    # oversample=2 auto-upgrade introduces a tiny amount of resample_poly
    # ringing around the hard-clip transition.
    result = apply_waveshaper(
        ramp, algorithm="digital_clip", drive=0.0, mix=1.0, oversample=1
    )
    # drive=0.0 -> drive_gain=1.0 -> rails land at exactly +1.0 / -0.95, but
    # RMS normalization applies a scalar — compute that scalar and verify the
    # rails relative to it.  Simpler: skip normalization by inspecting the
    # ratio between min and max; the asymmetry (neg rail / pos rail) is
    # invariant under a positive scalar.
    assert np.all(np.isfinite(result))
    ratio = result.min() / result.max()
    np.testing.assert_allclose(ratio, -0.95, atol=1e-6)


def test_all_three_finite_on_random_input() -> None:
    rng = np.random.default_rng(seed=7)
    signal = rng.uniform(-1.0, 1.0, size=2048).astype(np.float64)
    for algo in ("bit_crush", "rate_reduce", "digital_clip"):
        result = apply_waveshaper(signal, algorithm=algo, drive=0.5)
        assert result.shape == signal.shape, f"{algo} shape mismatch"
        assert np.all(np.isfinite(result)), f"{algo} produced NaN/Inf"


def test_all_three_reachable_via_apply_waveshaper_by_name() -> None:
    signal = _sine()
    for algo in ("bit_crush", "rate_reduce", "digital_clip"):
        result = apply_waveshaper(signal, algorithm=algo, drive=0.5)
        assert np.all(np.isfinite(result)), f"{algo} produced non-finite output"
        assert not np.allclose(result, signal, atol=1e-6), (
            f"{algo} output is identical to input at drive=0.5"
        )


def test_oversample_2_runs_clean() -> None:
    signal = _sine()
    result = apply_waveshaper(signal, algorithm="bit_crush", drive=0.5, oversample=2)
    assert result.shape == signal.shape
    assert np.all(np.isfinite(result))
