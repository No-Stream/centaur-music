"""Synth utility and plugin-loading tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from code_musics import synth
from code_musics.synth import (
    ExternalPluginSpec,
    _float_to_int16_pcm,
    _load_external_plugin,
    _loaded_external_plugins,
    _shape_reverb_return,
    apply_bricasti,
)


def test_plugin_cache_key_includes_bundle_plugin_name() -> None:
    """Regression test: two plugins from the same bundle file (e.g. LSP compressor
    and limiter) must NOT share a cache slot.  The original bug used only
    (host, format, path) as the key, so loading the compressor first and then
    asking for the limiter returned the compressor plugin, causing AttributeError
    on parameters that only the limiter has (e.g. 'threshold_db').
    """
    bundle_path = Path("/fake/bundle.vst3")
    fake_compressor = MagicMock(name="CompressorPlugin")
    fake_limiter = MagicMock(name="LimiterPlugin")

    spec_compressor = ExternalPluginSpec(
        name="fake_compressor",
        path=bundle_path,
        format="vst3",
        bundle_plugin_name="Compressor Stereo",
    )
    spec_limiter = ExternalPluginSpec(
        name="fake_limiter",
        path=bundle_path,
        format="vst3",
        bundle_plugin_name="Limiter Stereo",
    )

    def fake_load_plugin(_path: str, plugin_name: str | None = None) -> MagicMock:
        return fake_compressor if plugin_name == "Compressor Stereo" else fake_limiter

    with (
        patch("code_musics.synth._get_external_plugin_spec") as mock_get_spec,
        patch("code_musics.synth.Path.exists", return_value=True),
        patch("pedalboard.load_plugin", side_effect=fake_load_plugin),
    ):
        _loaded_external_plugins.clear()

        mock_get_spec.return_value = spec_compressor
        got_compressor = _load_external_plugin(plugin_name="fake_compressor")

        mock_get_spec.return_value = spec_limiter
        got_limiter = _load_external_plugin(plugin_name="fake_limiter")

    # The two loads must return distinct plugin objects.
    assert got_compressor is not got_limiter, (
        "Plugin cache collision: compressor and limiter share the same cache slot "
        "because bundle_plugin_name was not part of the cache key."
    )
    assert got_compressor is fake_compressor
    assert got_limiter is fake_limiter


def test_shape_reverb_return_highpass_lowpass_and_tilt_change_spectrum() -> None:
    sample_rate = 44_100
    duration_seconds = 1.0
    time = np.linspace(
        0.0,
        duration_seconds,
        int(sample_rate * duration_seconds),
        endpoint=False,
    )
    low_component = np.sin(2.0 * np.pi * 80.0 * time)
    high_component = 0.3 * np.sin(2.0 * np.pi * 4_000.0 * time)
    wet_signal = low_component + high_component

    filtered = _shape_reverb_return(
        wet_signal,
        sample_rate=sample_rate,
        highpass_hz=250.0,
        lowpass_hz=6_000.0,
    )
    tilted = _shape_reverb_return(
        wet_signal,
        sample_rate=sample_rate,
        tilt_db=6.0,
    )

    def _component_rms(signal: np.ndarray, frequency_hz: float) -> float:
        reference = np.sin(2.0 * np.pi * frequency_hz * time)
        projection = 2.0 * np.mean(signal * reference)
        return float(abs(projection) / np.sqrt(2.0))

    low_before = _component_rms(wet_signal, 80.0)
    low_after = _component_rms(filtered, 80.0)
    high_before = _component_rms(wet_signal, 4_000.0)
    high_after = _component_rms(filtered, 4_000.0)

    assert low_after < low_before * 0.3
    assert high_after > high_before * 0.5

    tilted_low = _component_rms(tilted, 80.0)
    tilted_high = _component_rms(tilted, 4_000.0)
    assert (tilted_high / tilted_low) > (high_before / low_before)


def test_shape_reverb_return_rejects_invalid_band_edges() -> None:
    with pytest.raises(ValueError, match="highpass_hz must be lower than lowpass_hz"):
        _shape_reverb_return(
            np.ones(16, dtype=np.float64),
            sample_rate=44_100,
            highpass_hz=2_000.0,
            lowpass_hz=1_000.0,
        )


def test_write_wav_logs_peak_and_loudness_diagnostics(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    signal = np.array([0.0, 0.1, -0.1, 0.0], dtype=np.float64)

    with patch("code_musics.synth.wavfile.write"):
        caplog.set_level("INFO", logger="code_musics.synth")
        synth.write_wav(tmp_path / "quiet.wav", signal)

    assert "peak" in caplog.text
    assert "true peak" in caplog.text
    assert "integrated loudness" in caplog.text
    assert "unexpectedly low" in caplog.text


def test_apply_bricasti_blends_filtered_wet_only() -> None:
    input_signal = np.vstack(
        [
            np.full(32, 1.0, dtype=np.float64),
            np.full(32, -0.5, dtype=np.float64),
        ]
    )

    class _FakeBoard:
        def __call__(self, signal: np.ndarray, _sample_rate: int) -> np.ndarray:
            return np.asarray(signal, dtype=np.float64) * 2.0

    with (
        patch("code_musics.synth.BRICASTI_IR_DIR", Path("/fake/irs")),
        patch("pathlib.Path.exists", return_value=True),
        patch(
            "code_musics.synth._CONVOLUTION_CLS",
            side_effect=lambda path, mix: (path, mix),
        ),
        patch("code_musics.synth._PEDALBOARD_CLS", return_value=_FakeBoard()),
        patch(
            "code_musics.synth._shape_reverb_return",
            side_effect=lambda signal, **_: signal * 0.25,
        ) as mock_shape,
    ):
        output = apply_bricasti(
            input_signal,
            ir_name="fake_ir",
            wet=0.4,
            highpass_hz=300.0,
        )

    expected_wet_signal = input_signal * 2.0 * 0.25
    expected_output = ((1.0 - 0.4) * input_signal) + (0.4 * expected_wet_signal)

    np.testing.assert_allclose(output, expected_output)
    assert mock_shape.call_count == 1


def test_float_to_int16_pcm_clips_out_of_range_samples_without_wrapping() -> None:
    quantized = _float_to_int16_pcm(
        np.array([1.1, -1.1, 0.0], dtype=np.float64),
        rng=np.random.default_rng(0),
    )

    assert quantized.dtype == np.int16
    assert quantized[0] > 0
    assert quantized[1] < 0
    assert quantized.max() <= np.iinfo(np.int16).max
    assert quantized.min() >= np.iinfo(np.int16).min


def test_float_to_int16_pcm_applies_dither_to_quiet_signal() -> None:
    signal = np.full(4_096, 1e-6, dtype=np.float64)

    quantized = _float_to_int16_pcm(signal, rng=np.random.default_rng(0))

    assert quantized.dtype == np.int16
    assert np.unique(quantized).size > 1
