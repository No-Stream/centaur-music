"""Shared DSP helpers for tests.

Small, pure numerical utilities used by many filter/engine test modules.
Kept deliberately thin — no project imports, no test fixtures.  New test
files should prefer these over re-defining private ``_rms`` / ``_band_energy``
/ ``_noise`` helpers locally.
"""

from __future__ import annotations

import numpy as np


def rms(x: np.ndarray) -> float:
    """Root-mean-square of a signal."""
    return float(np.sqrt(np.mean(x * x)))


def band_energy(
    signal: np.ndarray,
    low_hz: float,
    high_hz: float,
    sr: int = 44100,
) -> float:
    """RMS of the rFFT magnitude within ``[low_hz, high_hz]``.

    Not a true Parseval-normalised energy — it is the RMS of spectrum bins in
    the band, which is what the existing test suite has been using as a
    relative indicator.  Sufficient for ratio-style assertions like
    "bandA_energy > bandB_energy".
    """
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    return float(np.sqrt(np.mean(spectrum[mask] ** 2)))


def noise(dur: float = 0.3, seed: int = 42, sr: int = 44100) -> np.ndarray:
    """Deterministic Gaussian noise of length ``int(sr * dur)`` samples."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(sr * dur))
