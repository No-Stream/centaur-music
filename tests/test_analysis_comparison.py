"""Tests for audio comparison tooling in analysis.py."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.analysis import (
    AudioComparison,
    compare_audio,
    compute_mfcc,
    plot_comparison,
)

SAMPLE_RATE = 44_100


def _sine(freq_hz: float, duration: float = 0.5, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * duration), dtype=np.float64) / SAMPLE_RATE
    return amp * np.sin(2.0 * np.pi * freq_hz * t)


def _noise(duration: float = 0.5, amp: float = 0.5, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return amp * rng.standard_normal(int(SAMPLE_RATE * duration))


# --- compute_mfcc -----------------------------------------------------------


def test_compute_mfcc_returns_correct_shape() -> None:
    signal = _sine(440.0)
    mfccs = compute_mfcc(signal, sample_rate=SAMPLE_RATE)
    assert mfccs.ndim == 2
    assert mfccs.shape[1] == 13  # default n_mfcc


def test_compute_mfcc_custom_n_mfcc() -> None:
    signal = _sine(440.0)
    mfccs = compute_mfcc(signal, sample_rate=SAMPLE_RATE, n_mfcc=7)
    assert mfccs.shape[1] == 7


def test_compute_mfcc_sine_vs_noise_differ() -> None:
    mfcc_sine = compute_mfcc(_sine(440.0), sample_rate=SAMPLE_RATE)
    mfcc_noise = compute_mfcc(_noise(), sample_rate=SAMPLE_RATE)
    mean_sine = np.mean(mfcc_sine, axis=0)
    mean_noise = np.mean(mfcc_noise, axis=0)
    # Cosine similarity should be noticeably less than 1.0
    cos_sim = float(
        np.dot(mean_sine, mean_noise)
        / (np.linalg.norm(mean_sine) * np.linalg.norm(mean_noise))
    )
    assert cos_sim < 0.95, f"sine vs noise MFCC cosine similarity too high: {cos_sim}"


# --- compare_audio ----------------------------------------------------------


def test_compare_identical_signals() -> None:
    signal = _sine(440.0)
    result = compare_audio(signal, signal, sample_rate=SAMPLE_RATE)

    assert isinstance(result, AudioComparison)
    assert result.envelope_correlation == pytest.approx(1.0, abs=0.01)
    assert result.mfcc_cosine_similarity == pytest.approx(1.0, abs=0.01)
    assert result.peak_diff_db == pytest.approx(0.0, abs=0.01)
    assert result.centroid_diff_hz == pytest.approx(0.0, abs=0.1)
    assert result.spectral_tilt_diff == pytest.approx(0.0, abs=0.01)
    for band_diff in result.band_energy_diff_db.values():
        assert band_diff == pytest.approx(0.0, abs=0.01)


def test_compare_very_different_signals() -> None:
    sine = _sine(100.0, duration=0.5)
    noise = _noise(duration=0.5)
    result = compare_audio(sine, noise, sample_rate=SAMPLE_RATE)

    # Envelope correlation should be low for uncorrelated signals
    assert result.envelope_correlation < 0.5
    # MFCC similarity should be well below 1
    assert result.mfcc_cosine_similarity < 0.95


def test_compare_handles_different_lengths() -> None:
    short = _sine(440.0, duration=0.2)
    long = _sine(440.0, duration=0.8)
    result = compare_audio(short, long, sample_rate=SAMPLE_RATE)

    # Should not crash; envelope correlation should still be high since both
    # are the same sine wave, just truncated to the shorter length.
    assert isinstance(result, AudioComparison)
    assert result.envelope_correlation > 0.9


def test_band_energy_diffs_in_reasonable_range() -> None:
    sig_a = _sine(100.0, duration=0.5, amp=0.8)
    sig_b = _sine(100.0, duration=0.5, amp=0.4)
    result = compare_audio(sig_a, sig_b, sample_rate=SAMPLE_RATE)

    for band_name, diff in result.band_energy_diff_db.items():
        # dB differences should be finite and within a plausible range
        assert np.isfinite(diff), f"band {band_name} diff is not finite"
        assert abs(diff) < 60.0, f"band {band_name} diff implausibly large: {diff}"


# --- plot_comparison --------------------------------------------------------


def test_plot_comparison_returns_figure() -> None:
    import matplotlib.pyplot as plt

    sig_a = _sine(440.0, duration=0.3)
    sig_b = _noise(duration=0.3)
    fig = plot_comparison(sig_a, sig_b, sample_rate=SAMPLE_RATE)

    try:
        from matplotlib.figure import Figure

        assert isinstance(fig, Figure)
        axes = fig.get_axes()
        # 2x2 grid = 4 axes + colorbars for the 2 spectrograms = 6 total
        assert len(axes) >= 4
    finally:
        plt.close(fig)


def test_plot_comparison_saves_to_file(tmp_path: pytest.TempPathFactory) -> None:
    import matplotlib.pyplot as plt

    sig_a = _sine(440.0, duration=0.2)
    sig_b = _sine(880.0, duration=0.2)
    out = tmp_path / "comparison.png"  # type: ignore[operator]
    fig = plot_comparison(
        sig_a,
        sig_b,
        sample_rate=SAMPLE_RATE,
        label_a="sine440",
        label_b="sine880",
        output_path=out,
    )
    try:
        assert out.exists()  # type: ignore[union-attr]
        assert out.stat().st_size > 0  # type: ignore[union-attr]
    finally:
        plt.close(fig)
