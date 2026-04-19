"""Tests for the parallel modal resonator bank primitive."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._modal import render_modal_bank
from code_musics.spectra import get_mode_table

SAMPLE_RATE = 44100


def _impulse(n_samples: int) -> np.ndarray:
    exciter = np.zeros(n_samples, dtype=np.float64)
    exciter[0] = 1.0
    return exciter


def test_single_mode_produces_clean_tone() -> None:
    duration_s = 0.5
    n_samples = int(duration_s * SAMPLE_RATE)
    exciter = _impulse(n_samples)
    freq_hz = 440.0

    out = render_modal_bank(
        exciter,
        mode_ratios=[1.0],
        mode_amps=[1.0],
        mode_decays_s=[2.0],
        freq_hz=freq_hz,
        sample_rate=SAMPLE_RATE,
    )

    assert out.shape == (n_samples,)
    assert np.all(np.isfinite(out))

    spec = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / SAMPLE_RATE)
    peak_bin = int(np.argmax(spec))
    peak_freq = float(freqs[peak_bin])

    assert abs(peak_freq - freq_hz) / freq_hz < 0.02, (
        f"peak {peak_freq} Hz not within 2% of {freq_hz} Hz"
    )


def test_multi_mode_membrane() -> None:
    duration_s = 0.5
    n_samples = int(duration_s * SAMPLE_RATE)
    exciter = _impulse(n_samples)

    ratios = get_mode_table("membrane")[:4]
    amps = [1.0, 0.6, 0.4, 0.3]
    decays = [0.4, 0.3, 0.25, 0.2]

    out = render_modal_bank(
        exciter,
        mode_ratios=ratios,
        mode_amps=amps,
        mode_decays_s=decays,
        freq_hz=200.0,
        sample_rate=SAMPLE_RATE,
    )

    assert out.shape == (n_samples,)
    assert np.all(np.isfinite(out))
    assert float(np.max(np.abs(out))) > 0.0
    assert float(np.max(np.abs(out))) < 10.0


def test_decay_shortening() -> None:
    duration_s = 1.5
    n_samples = int(duration_s * SAMPLE_RATE)
    exciter = _impulse(n_samples)

    long_out = render_modal_bank(
        exciter,
        mode_ratios=[1.0],
        mode_amps=[1.0],
        mode_decays_s=[1.0],
        freq_hz=300.0,
        sample_rate=SAMPLE_RATE,
    )
    short_out = render_modal_bank(
        exciter,
        mode_ratios=[1.0],
        mode_amps=[1.0],
        mode_decays_s=[0.05],
        freq_hz=300.0,
        sample_rate=SAMPLE_RATE,
    )

    tail_start = int(0.9 * n_samples)
    long_rms = float(np.sqrt(np.mean(long_out[tail_start:] ** 2)))
    short_rms = float(np.sqrt(np.mean(short_out[tail_start:] ** 2)))

    assert long_rms > short_rms, (
        f"long-decay tail RMS {long_rms} should exceed short-decay tail RMS {short_rms}"
    )


def test_mode_above_nyquist_skipped() -> None:
    duration_s = 0.2
    n_samples = int(duration_s * SAMPLE_RATE)
    exciter = _impulse(n_samples)

    # freq_hz=1000 * ratio=30 = 30 kHz > Nyquist (22.05 kHz at 44.1k)
    out = render_modal_bank(
        exciter,
        mode_ratios=[1.0, 30.0],
        mode_amps=[1.0, 1.0],
        mode_decays_s=[0.3, 0.3],
        freq_hz=1000.0,
        sample_rate=SAMPLE_RATE,
    )

    assert out.shape == (n_samples,)
    assert np.all(np.isfinite(out))
    assert float(np.max(np.abs(out))) > 0.0


def test_get_mode_table_all_named_variants() -> None:
    names = [
        "membrane",
        "bar_wood",
        "bar_metal",
        "bar_glass",
        "plate",
        "bowl",
        "stopped_pipe",
    ]
    for name in names:
        ratios = get_mode_table(name)
        assert len(ratios) > 0, f"{name} returned empty table"
        assert abs(ratios[0] - 1.0) < 0.01, (
            f"{name} fundamental ratio {ratios[0]} not ~1.0"
        )


def test_get_mode_table_custom_raises() -> None:
    with pytest.raises(ValueError, match="custom"):
        get_mode_table("custom")


def test_mode_table_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown mode table"):
        get_mode_table("not_a_real_table")
