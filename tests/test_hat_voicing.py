"""Regression tests for hi-hat voicing noise density."""

from __future__ import annotations

import numpy as np

from code_musics.engines.drum_voice import render as drum_voice_render
from code_musics.engines.metallic_perc import render as metallic_perc_render
from code_musics.engines.registry import resolve_synth_params

SAMPLE_RATE = 44_100


def _render_preset(
    *,
    engine: str,
    preset: str,
    freq: float = 4_600.0,
    duration: float = 0.45,
) -> np.ndarray:
    params = resolve_synth_params({"engine": engine, "preset": preset})
    renderer = drum_voice_render if engine == "drum_voice" else metallic_perc_render
    return renderer(
        freq=freq,
        duration=duration,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params=params,
    )


def _high_band_flatness(audio: np.ndarray) -> float:
    spectrum = _high_band_spectrum(audio)
    return float(np.exp(np.mean(np.log(spectrum))) / np.mean(spectrum))


def _dominant_peak_ratio_db(audio: np.ndarray) -> float:
    spectrum = _high_band_spectrum(audio)
    return float(20.0 * np.log10(np.max(spectrum) / np.median(spectrum)))


def _high_band_spectrum(audio: np.ndarray) -> np.ndarray:
    window = np.hanning(audio.size)
    spectrum = np.abs(np.fft.rfft(audio * window)) + 1e-12
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / SAMPLE_RATE)
    band_mask = (freqs >= 3_000.0) & (freqs <= 14_000.0)
    return spectrum[band_mask]


class TestHiHatVoicing:
    def test_hat_presets_use_noise_forward_metallic_voicing(self) -> None:
        metallic_closed = resolve_synth_params(
            {"engine": "metallic_perc", "preset": "closed_hat"}
        )
        metallic_open = resolve_synth_params(
            {"engine": "metallic_perc", "preset": "open_hat"}
        )
        drum_closed = resolve_synth_params(
            {"engine": "drum_voice", "preset": "closed_hat"}
        )
        drum_open = resolve_synth_params({"engine": "drum_voice", "preset": "open_hat"})

        assert metallic_closed["voicing"] == "hat_noise"
        assert metallic_open["voicing"] == "hat_noise"
        assert drum_closed["metallic_type"] == "hat_noise"
        assert drum_open["metallic_type"] == "hat_noise"

    def test_legacy_metallic_hats_are_not_narrow_tonal_peaks(self) -> None:
        for preset, duration in [("closed_hat", 0.08), ("open_hat", 0.45)]:
            audio = _render_preset(
                engine="metallic_perc", preset=preset, duration=duration
            )

            assert np.isfinite(audio).all()
            assert _high_band_flatness(audio) >= 0.18
            assert _dominant_peak_ratio_db(audio) <= 34.0

    def test_drum_voice_open_hat_is_noisier_than_ride_bell(self) -> None:
        open_hat = _render_preset(engine="drum_voice", preset="open_hat")
        ride_bell = _render_preset(engine="drum_voice", preset="ride_bell")

        assert _high_band_flatness(open_hat) >= 0.30
        assert _dominant_peak_ratio_db(open_hat) <= 18.0
        assert _high_band_flatness(open_hat) > _high_band_flatness(ride_bell)
        assert _dominant_peak_ratio_db(open_hat) < _dominant_peak_ratio_db(ride_bell)

    def test_tonal_metallic_presets_keep_bell_like_peak_structure(self) -> None:
        cowbell = _render_preset(
            engine="metallic_perc", preset="cowbell", freq=800.0, duration=0.25
        )
        ride_bell = _render_preset(
            engine="drum_voice", preset="ride_bell", freq=4_600.0, duration=0.45
        )

        assert _dominant_peak_ratio_db(cowbell) >= 40.0
        assert _dominant_peak_ratio_db(ride_bell) >= 25.0
