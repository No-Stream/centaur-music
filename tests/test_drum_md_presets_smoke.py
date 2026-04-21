"""Smoke tests for the Machinedrum-inspired drum_voice presets.

Renders each new preset at a representative pitch and asserts the output is
finite, non-silent, and free of NaN/Inf.  Covers the EFM tone / EFM cymbal /
PI modal / digital-character shaper additions landed in this round.
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines.drum_voice import render as dv_render
from code_musics.engines.registry import resolve_synth_params

SAMPLE_RATE = 44_100
DURATION_S = 0.4
AMP = 0.8
RMS_FLOOR = 1e-5

# Sonic-role categories used to apply per-preset spectral-centroid bands.
# Bands are calibrated against the engine's actual output (post-normalization,
# shaping, and envelope): kicks land well under 2 kHz, bells/cymbals stay
# brightly above 4 kHz, and mid-role presets (cowbells, toms, hats, snares)
# sit in the middle.  Bands are deliberately loose so a slight DSP change
# doesn't flip them, but tight enough to catch "silent sine fallback" bugs.
LOW_CENTROID_PRESETS = {
    "efm_kick_deep",
    "efm_kick_punch",
    "kick_bitcrush",
}
HIGH_CENTROID_PRESETS = {
    "efm_cymbal_trash",
    "efm_cymbal_china",
    "pi_metal_bell",
    "pi_glass_ping",
    "pi_bowl_shimmer",
    "pi_kick_shell",
    "pi_wood_block",
    "efm_snare_bright",
    "snare_digital_fuzz",
}
MID_CENTROID_PRESETS = {
    "efm_cowbell",
    "pi_tom_membrane",
    "hat_rate_reduced",
}

LOW_CENTROID_MAX_HZ = 2000.0
HIGH_CENTROID_MIN_HZ = 4000.0
MID_CENTROID_MIN_HZ = 1000.0
MID_CENTROID_MAX_HZ = 8000.0


def _spectral_centroid(signal: np.ndarray, sample_rate: int) -> float:
    magnitudes = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.shape[0], d=1.0 / sample_rate)
    total = float(magnitudes.sum())
    if total < 1e-12:
        return 0.0
    return float((freqs * magnitudes).sum() / total)


# Each entry: (preset_name, freq_hz).  Frequencies chosen to match the preset
# role: low for kicks, mid for toms / snares / cowbells, high for bells /
# cymbals / hats.
MD_PRESETS: list[tuple[str, float]] = [
    # EFM tones
    ("efm_kick_deep", 60.0),
    ("efm_kick_punch", 60.0),
    ("efm_snare_bright", 200.0),
    ("efm_cowbell", 440.0),
    # EFM cymbals
    ("efm_cymbal_trash", 440.0),
    ("efm_cymbal_china", 440.0),
    # PI modal tones
    ("pi_tom_membrane", 200.0),
    ("pi_kick_shell", 60.0),
    ("pi_wood_block", 440.0),
    # PI modal banks (metallic)
    ("pi_metal_bell", 440.0),
    ("pi_glass_ping", 440.0),
    ("pi_bowl_shimmer", 440.0),
    # Digital-character shapers
    ("kick_bitcrush", 60.0),
    ("hat_rate_reduced", 440.0),
    ("snare_digital_fuzz", 200.0),
]


@pytest.mark.parametrize(("preset_name", "freq_hz"), MD_PRESETS)
def test_md_preset_renders_finite_nonsilent(preset_name: str, freq_hz: float) -> None:
    """Each Machinedrum-inspired preset must produce finite, audible audio
    that matches its sonic role."""
    resolved = resolve_synth_params({"engine": "drum_voice", "preset": preset_name})
    signal = dv_render(
        freq=freq_hz,
        duration=DURATION_S,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params=resolved,
    )

    assert isinstance(signal, np.ndarray), f"{preset_name}: expected ndarray"
    assert signal.size == int(SAMPLE_RATE * DURATION_S), (
        f"{preset_name}: unexpected signal length {signal.size}"
    )
    assert np.all(np.isfinite(signal)), f"{preset_name}: produced NaN / Inf samples"
    rms = float(np.sqrt(np.mean(signal * signal)))
    assert rms > RMS_FLOOR, (
        f"{preset_name}: silent output (rms={rms:.3e} <= {RMS_FLOOR:.3e})"
    )

    centroid_hz = _spectral_centroid(signal, SAMPLE_RATE)
    if preset_name in LOW_CENTROID_PRESETS:
        assert centroid_hz < LOW_CENTROID_MAX_HZ, (
            f"{preset_name}: expected low-centroid kick/tom, got "
            f"{centroid_hz:.1f} Hz (max {LOW_CENTROID_MAX_HZ:.1f} Hz)"
        )
    elif preset_name in HIGH_CENTROID_PRESETS:
        assert centroid_hz > HIGH_CENTROID_MIN_HZ, (
            f"{preset_name}: expected high-centroid bell/cymbal, got "
            f"{centroid_hz:.1f} Hz (min {HIGH_CENTROID_MIN_HZ:.1f} Hz)"
        )
    elif preset_name in MID_CENTROID_PRESETS:
        assert MID_CENTROID_MIN_HZ < centroid_hz < MID_CENTROID_MAX_HZ, (
            f"{preset_name}: expected mid-centroid cowbell/tom/hat, got "
            f"{centroid_hz:.1f} Hz (expected "
            f"{MID_CENTROID_MIN_HZ:.1f} < centroid < {MID_CENTROID_MAX_HZ:.1f})"
        )
