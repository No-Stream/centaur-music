"""Tests for the per-note voice_dist slot wired into the ``va`` engine.

Verifies:

* ``voice_dist_mode="off"`` and ``voice_dist_drive=0`` are bit-for-bit
  no-ops vs omitting the params entirely (so existing pieces keep
  identical output).
* Pre-sum distortion produces lower chord-IMD sidebands than post-sum
  distortion of the clean chord — the RePro-5 polyphonic-distortion
  point expressed mechanically.
* Every non-off mode renders finite audio with audible energy.
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines.va import render

SR: int = 48000


def _base_params(**overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "_voice_name": "va_voice_dist_test",
        "osc_mode": "supersaw",
        "supersaw_detune": 0.3,
        "supersaw_mix": 0.5,
        "cutoff_hz": 6000.0,
        "resonance_q": 0.9,
        # Disable analog wobble so determinism tests are sharp.
        "analog_jitter": 0.0,
        "pitch_drift": 0.0,
        "noise_floor": 0.0,
        "cutoff_drift": 0.0,
    }
    params.update(overrides)
    return params


def _bin_power(
    spec: np.ndarray, freqs: np.ndarray, target: float, width: float
) -> float:
    mask = (freqs > target - width) & (freqs < target + width)
    if not mask.any():
        return 0.0
    return float(spec[mask].max() ** 2)


def test_voice_dist_off_is_noop() -> None:
    """``voice_dist_mode='off'`` with any drive/mix/tone is bit-identical to
    omitting the params entirely.  This is the existing-pieces guarantee."""
    params_default = _base_params()
    params_off = _base_params(
        voice_dist_mode="off",
        voice_dist_drive=1.5,
        voice_dist_mix=1.0,
        voice_dist_tone=0.3,
    )
    a = render(freq=220.0, duration=0.3, amp=0.5, sample_rate=SR, params=params_default)
    b = render(freq=220.0, duration=0.3, amp=0.5, sample_rate=SR, params=params_off)
    np.testing.assert_array_equal(a, b)


def test_voice_dist_drive_zero_is_noop() -> None:
    """Even with ``voice_dist_mode='hard_clip'``, drive=0 must be a fast-path
    passthrough — matches the ``apply_voice_dist`` helper contract."""
    params_default = _base_params()
    params_zero = _base_params(
        voice_dist_mode="hard_clip",
        voice_dist_drive=0.0,
        voice_dist_mix=1.0,
    )
    a = render(freq=220.0, duration=0.3, amp=0.5, sample_rate=SR, params=params_default)
    b = render(freq=220.0, duration=0.3, amp=0.5, sample_rate=SR, params=params_zero)
    np.testing.assert_array_equal(a, b)


def _render_triad_presum(drive: float) -> np.ndarray:
    """Render three notes with voice_dist applied inside each note and sum."""
    freqs = (220.0, 277.18, 329.63)  # A3, C#4, E4 — major triad
    out = np.zeros(int(0.4 * SR), dtype=np.float64)
    for freq in freqs:
        note = render(
            freq=freq,
            duration=0.4,
            amp=0.4,
            sample_rate=SR,
            params=_base_params(
                voice_dist_mode="hard_clip",
                voice_dist_drive=drive,
                voice_dist_mix=1.0,
            ),
        )
        out += note
    return out


def _render_triad_clean_then_postclip(drive: float) -> np.ndarray:
    """Render three clean notes, sum, and apply the same hard-clip shaper to
    the summed mix.  This is the post-sum "regular insert" that the RePro-5
    idiom is supposed to beat on chord clarity."""
    from code_musics.engines._waveshaper import apply_waveshaper

    freqs = (220.0, 277.18, 329.63)
    out = np.zeros(int(0.4 * SR), dtype=np.float64)
    for freq in freqs:
        note = render(
            freq=freq, duration=0.4, amp=0.4, sample_rate=SR, params=_base_params()
        )
        out += note
    internal_drive = float(np.clip(drive * 0.5, 0.0, 1.0))
    return apply_waveshaper(out, algorithm="hard_clip", drive=internal_drive, mix=1.0)


def test_chord_imd_pre_sum_vs_post_sum() -> None:
    """Pre-sum per-note distortion should show lower intermod sidebands at
    ``f2 +/- f1`` than equivalent post-sum hard-clip of the clean chord."""
    drive = 1.5
    pre_sum = _render_triad_presum(drive)
    post_sum = _render_triad_clean_then_postclip(drive)

    # f2 - f1 = 277.18 - 220 ~ 57.18 Hz (difference tone)
    # f1 + f2 = 497.18 Hz (sum tone)
    # f2 + f3 - f1 = 386.81 Hz (IMD product)
    targets = (57.18, 497.18)

    def _imd_power(signal: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(signal.size, 1.0 / SR)
        return sum(_bin_power(spec, freqs, t, width=4.0) for t in targets)

    imd_pre = _imd_power(pre_sum)
    imd_post = _imd_power(post_sum)
    assert imd_pre < imd_post, (
        f"pre-sum IMD ({imd_pre:.3e}) should be lower than post-sum IMD "
        f"({imd_post:.3e})"
    )


@pytest.mark.parametrize(
    "mode", ["soft_clip", "hard_clip", "foldback", "corrode", "saturation", "preamp"]
)
def test_each_mode_smoke(mode: str) -> None:
    """Every non-off mode renders finite, non-silent audio."""
    signal = render(
        freq=220.0,
        duration=0.2,
        amp=0.5,
        sample_rate=SR,
        params=_base_params(voice_dist_mode=mode, voice_dist_drive=0.5),
    )
    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(signal)) > 0.01
