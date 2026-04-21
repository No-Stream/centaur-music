"""Tests for :mod:`code_musics.engines._voice_dist`.

The ``apply_voice_dist`` helper is the per-note distortion slot that
``polyblep`` / ``va`` / ``filtered_stack`` will call inside their note
loops.  These tests verify the fast-path semantics (``mode="off"`` and
``drive=0`` are bit-for-bit passthrough), monotonic THD with drive on
cheap shapers, the saturation-blend coefficient idiom at ``drive==0``
boundary, mix range behavior, tone-stage spectrum shift, and smoke
coverage for the expensive modes.
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._voice_dist import apply_voice_dist

SAMPLE_RATE: int = 44_100


def _sine(
    freq: float = 440.0,
    duration: float = 0.5,
    sr: int = SAMPLE_RATE,
    amp: float = 0.5,
) -> np.ndarray:
    t = np.arange(int(sr * duration), dtype=np.float64) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float64)


def _thd_percent(signal: np.ndarray, f0: float = 440.0, sr: int = SAMPLE_RATE) -> float:
    """FFT-based THD: sqrt(sum(h2..h10^2)) / h1 in percent."""
    n = signal.shape[0]
    spectrum = np.abs(np.fft.rfft(signal))
    bin_per_hz = n / sr
    h1_bin = int(round(f0 * bin_per_hz))
    h1 = spectrum[h1_bin]
    harmonics_sq = 0.0
    for h in range(2, 11):
        idx = int(round(h * f0 * bin_per_hz))
        if idx < spectrum.shape[0]:
            harmonics_sq += float(spectrum[idx]) ** 2
    return float(np.sqrt(harmonics_sq)) / (float(h1) + 1e-12) * 100.0


def test_off_mode_returns_input_unchanged() -> None:
    signal = _sine()
    out = apply_voice_dist(signal, mode="off", drive=1.5, mix=1.0, tone=0.5)
    assert out.shape == signal.shape
    # mode="off" must be an exact passthrough regardless of drive/mix/tone.
    np.testing.assert_array_equal(out, signal)


def test_drive_zero_returns_input_unchanged() -> None:
    signal = _sine()
    for mode in (
        "soft_clip",
        "hard_clip",
        "foldback",
        "corrode",
        "saturation",
        "preamp",
    ):
        out = apply_voice_dist(signal, mode=mode, drive=0.0, mix=1.0)
        np.testing.assert_allclose(
            out,
            signal,
            atol=1e-12,
            err_msg=f"{mode} drive=0 should be passthrough",
        )


def test_each_mode_runs_without_nan() -> None:
    signal = _sine()
    for mode in (
        "soft_clip",
        "hard_clip",
        "foldback",
        "corrode",
        "saturation",
        "preamp",
    ):
        out = apply_voice_dist(signal, mode=mode, drive=0.5, mix=1.0)
        assert out.shape == signal.shape, f"{mode} shape mismatch"
        assert np.all(np.isfinite(out)), f"{mode} produced NaN/Inf"


@pytest.mark.parametrize("mode", ["soft_clip", "hard_clip"])
def test_monotonic_thd_with_drive(mode: str) -> None:
    signal = _sine()
    thd_values: list[float] = []
    for drive in (0.3, 0.8, 1.5):
        out = apply_voice_dist(signal, mode=mode, drive=drive, mix=1.0)
        thd_values.append(_thd_percent(out))
    # THD should monotonically increase with drive on the clip-type
    # shapers.  (Foldback is deliberately excluded: wavefolders produce
    # non-monotonic harmonic content because the fold count resonates
    # with the input amplitude — higher drive does not necessarily mean
    # more THD for a fixed-amplitude sine.  They still produce rich
    # harmonic content at any non-zero drive, which is the point.)
    assert thd_values[0] < thd_values[1] <= thd_values[2], (
        f"{mode} non-monotonic THD across (0.3, 0.8, 1.5): {thd_values}"
    )
    assert thd_values[2] > thd_values[0] * 1.2, (
        f"{mode} max-drive THD={thd_values[2]:.2f}% not meaningfully above "
        f"min-drive THD={thd_values[0]:.2f}%"
    )


def test_foldback_mode_is_audibly_distorted() -> None:
    """Foldback at any non-zero drive produces substantial THD.

    Foldback THD is non-monotonic with drive (it's a wavefolder) so we
    don't assert monotonicity, but we do assert the mode actually
    distorts.
    """
    signal = _sine()
    clean_thd = _thd_percent(signal)
    out = apply_voice_dist(signal, mode="foldback", drive=0.5, mix=1.0)
    folded_thd = _thd_percent(out)
    assert folded_thd > max(clean_thd, 1.0) * 10.0, (
        f"foldback drive=0.5 produced only {folded_thd:.2f}% THD "
        f"(clean baseline {clean_thd:.4f}%)"
    )


def test_blend_zero_continuity() -> None:
    """Sweep drive across 0 and assert output RMS is continuous.

    The blend coefficient guarantees that automating drive through 0 has
    no step / click.  We measure RMS of each output at drive in the small
    neighborhood around zero and assert no large jumps.
    """
    signal = _sine()
    signal_rms = float(np.sqrt(np.mean(signal * signal)))
    drives = np.linspace(-0.001, 0.001, 20)
    rms_values: list[float] = []
    for drive in drives:
        out = apply_voice_dist(signal, mode="hard_clip", drive=float(drive), mix=1.0)
        rms_values.append(float(np.sqrt(np.mean(out * out))))
    rms_arr = np.asarray(rms_values, dtype=np.float64)
    max_adjacent_diff = float(np.max(np.abs(np.diff(rms_arr))))
    bound = 1e-3 * signal_rms
    assert max_adjacent_diff < bound, (
        f"blend-zero continuity broken: max adjacent RMS diff "
        f"{max_adjacent_diff:.3e} >= bound {bound:.3e}"
    )


def test_mix_range() -> None:
    """mix=0 is dry, mix=1 is wet, mix=0.5 is exactly the linear interpolation.

    RMS of the intermediate can dip below both endpoints when wet carries
    phase-coherent harmonics that partially cancel with the dry fundamental,
    so the strongest invariant is sample-wise linearity of the crossfade.
    """
    signal = _sine()
    wet_only = apply_voice_dist(signal, mode="hard_clip", drive=1.0, mix=1.0)
    dry_only = apply_voice_dist(signal, mode="hard_clip", drive=1.0, mix=0.0)
    half = apply_voice_dist(signal, mode="hard_clip", drive=1.0, mix=0.5)
    # mix=0 returns the input.
    np.testing.assert_allclose(dry_only, signal, atol=1e-12)
    # mix=0.5 is the sample-wise midpoint of dry and wet.
    expected_half = 0.5 * dry_only + 0.5 * wet_only
    np.testing.assert_allclose(half, expected_half, atol=1e-9)
    # And intermediate must actually differ from wet_only (wet is genuinely
    # distorted, not just a gain trim on the dry path).
    assert not np.allclose(half, wet_only, atol=1e-6)


def test_tone_shifts_spectrum() -> None:
    """tone>0 raises high-frequency energy; tone<0 reduces it."""
    signal = _sine(freq=200.0) + 0.3 * _sine(freq=5_000.0)
    # Use soft_clip with modest drive so the tone-stage spectral tilt is
    # visible in the output.
    neutral = apply_voice_dist(signal, mode="soft_clip", drive=0.5, mix=1.0, tone=0.0)
    bright = apply_voice_dist(signal, mode="soft_clip", drive=0.5, mix=1.0, tone=0.5)
    dark = apply_voice_dist(signal, mode="soft_clip", drive=0.5, mix=1.0, tone=-0.5)

    def high_energy(x: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / SAMPLE_RATE)
        mask = freqs >= 3_000.0
        return float(np.sum(spec[mask] ** 2))

    assert high_energy(bright) > high_energy(neutral), (
        "tone=+0.5 did not raise high-frequency energy"
    )
    assert high_energy(dark) < high_energy(neutral), (
        "tone=-0.5 did not reduce high-frequency energy"
    )


def test_unknown_mode_raises() -> None:
    signal = _sine(duration=0.01)
    with pytest.raises(ValueError, match="Unsupported voice_dist mode"):
        apply_voice_dist(signal, mode="not_a_mode", drive=0.5)


@pytest.mark.parametrize("mode", ["saturation", "preamp"])
def test_saturation_and_preamp_modes_smoke(mode: str) -> None:
    # Keep buffer tiny — these modes do heavy per-call processing.
    signal = _sine(duration=0.1, amp=0.3)
    out = apply_voice_dist(signal, mode=mode, drive=0.5, mix=1.0)
    assert out.shape == signal.shape
    assert np.all(np.isfinite(out))
    # Output must not be silent: the shaper should alter the signal.
    assert float(np.sqrt(np.mean(out * out))) > 1e-4
