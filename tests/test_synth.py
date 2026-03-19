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


def _sine_wave(
    frequency_hz: float,
    *,
    sample_rate: int,
    duration_seconds: float = 1.0,
) -> np.ndarray:
    time = np.linspace(
        0.0,
        duration_seconds,
        int(sample_rate * duration_seconds),
        endpoint=False,
    )
    return np.sin(2.0 * np.pi * frequency_hz * time)


def _component_rms(
    signal: np.ndarray,
    *,
    frequency_hz: float,
    sample_rate: int,
) -> float:
    time = np.linspace(
        0.0,
        signal.shape[-1] / sample_rate,
        signal.shape[-1],
        endpoint=False,
    )
    reference = np.sin(2.0 * np.pi * frequency_hz * time)
    projection = 2.0 * np.mean(np.asarray(signal, dtype=np.float64) * reference)
    return float(abs(projection) / np.sqrt(2.0))


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


def test_apply_plugin_processor_resets_cached_plugin_state() -> None:
    fake_plugin = MagicMock()
    fake_plugin.return_value = np.zeros((2, 8), dtype=np.float32)

    with patch("code_musics.synth._load_external_plugin", return_value=fake_plugin):
        synth._apply_plugin_processor(
            np.zeros(8, dtype=np.float64),
            plugin_name="tal_chorus_lx",
            params={"dry_wet": 2.2},
        )

    fake_plugin.reset.assert_called_once_with()


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


def test_apply_eq_rejects_empty_bands() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        synth.apply_eq(np.ones(32, dtype=np.float64), bands=[])


def test_apply_eq_rejects_unknown_band_kind() -> None:
    with pytest.raises(ValueError, match="Unsupported EQ band kind"):
        synth.apply_eq(
            np.ones(32, dtype=np.float64),
            bands=[{"kind": "not_a_real_band", "freq_hz": 500.0}],
        )


def test_apply_eq_rejects_unsupported_slope() -> None:
    with pytest.raises(ValueError, match="slope_db_per_oct must be 12 or 24"):
        synth.apply_eq(
            np.ones(32, dtype=np.float64),
            bands=[{"kind": "highpass", "cutoff_hz": 100.0, "slope_db_per_oct": 18}],
        )


def test_apply_eq_rejects_invalid_frequency() -> None:
    with pytest.raises(ValueError, match="freq_hz must be between 0 and Nyquist"):
        synth.apply_eq(
            np.ones(32, dtype=np.float64),
            bands=[{"kind": "bell", "freq_hz": 0.0, "gain_db": 2.0, "q": 1.0}],
        )


def test_apply_eq_rejects_invalid_q() -> None:
    with pytest.raises(ValueError, match="q must be positive"):
        synth.apply_eq(
            np.ones(32, dtype=np.float64),
            bands=[{"kind": "high_shelf", "freq_hz": 2000.0, "gain_db": 2.0, "q": 0.0}],
        )


def test_apply_eq_highpass_24db_is_stronger_than_12db() -> None:
    sample_rate = synth.SAMPLE_RATE
    signal = _sine_wave(80.0, sample_rate=sample_rate) + (
        0.5 * _sine_wave(2_000.0, sample_rate=sample_rate)
    )

    highpass_12 = synth.apply_eq(
        signal,
        bands=[{"kind": "highpass", "cutoff_hz": 250.0, "slope_db_per_oct": 12}],
    )
    highpass_24 = synth.apply_eq(
        signal,
        bands=[{"kind": "highpass", "cutoff_hz": 250.0, "slope_db_per_oct": 24}],
    )

    low_after_12 = _component_rms(
        highpass_12, frequency_hz=80.0, sample_rate=sample_rate
    )
    low_after_24 = _component_rms(
        highpass_24, frequency_hz=80.0, sample_rate=sample_rate
    )
    high_after_12 = _component_rms(
        highpass_12, frequency_hz=2_000.0, sample_rate=sample_rate
    )

    assert low_after_12 < 0.5
    assert low_after_24 < low_after_12 * 0.6
    assert high_after_12 > 0.2


def test_apply_eq_lowpass_24db_is_stronger_than_12db() -> None:
    sample_rate = synth.SAMPLE_RATE
    signal = _sine_wave(200.0, sample_rate=sample_rate) + (
        0.5 * _sine_wave(6_000.0, sample_rate=sample_rate)
    )

    lowpass_12 = synth.apply_eq(
        signal,
        bands=[{"kind": "lowpass", "cutoff_hz": 1_500.0, "slope_db_per_oct": 12}],
    )
    lowpass_24 = synth.apply_eq(
        signal,
        bands=[{"kind": "lowpass", "cutoff_hz": 1_500.0, "slope_db_per_oct": 24}],
    )

    high_after_12 = _component_rms(
        lowpass_12, frequency_hz=6_000.0, sample_rate=sample_rate
    )
    high_after_24 = _component_rms(
        lowpass_24, frequency_hz=6_000.0, sample_rate=sample_rate
    )
    low_after_12 = _component_rms(
        lowpass_12, frequency_hz=200.0, sample_rate=sample_rate
    )

    assert high_after_12 < 0.1
    assert high_after_24 < high_after_12 * 0.6
    assert low_after_12 > 0.5


def test_apply_eq_bell_boost_and_cut_change_target_band_energy() -> None:
    sample_rate = synth.SAMPLE_RATE
    signal = _sine_wave(250.0, sample_rate=sample_rate) + (
        0.4 * _sine_wave(1_000.0, sample_rate=sample_rate)
    )

    boosted = synth.apply_eq(
        signal,
        bands=[{"kind": "bell", "freq_hz": 1_000.0, "gain_db": 6.0, "q": 1.0}],
    )
    cut = synth.apply_eq(
        signal,
        bands=[{"kind": "bell", "freq_hz": 1_000.0, "gain_db": -6.0, "q": 1.0}],
    )

    target_before = _component_rms(
        signal, frequency_hz=1_000.0, sample_rate=sample_rate
    )
    target_boosted = _component_rms(
        boosted, frequency_hz=1_000.0, sample_rate=sample_rate
    )
    target_cut = _component_rms(cut, frequency_hz=1_000.0, sample_rate=sample_rate)
    low_before = _component_rms(signal, frequency_hz=250.0, sample_rate=sample_rate)
    low_boosted = _component_rms(boosted, frequency_hz=250.0, sample_rate=sample_rate)

    assert target_boosted > target_before * 1.6
    assert target_cut < target_before * 0.7
    assert low_boosted > low_before * 0.8


def test_apply_eq_shelves_change_expected_spectral_regions() -> None:
    sample_rate = synth.SAMPLE_RATE
    signal = _sine_wave(120.0, sample_rate=sample_rate) + (
        0.5 * _sine_wave(5_000.0, sample_rate=sample_rate)
    )

    low_shelf = synth.apply_eq(
        signal,
        bands=[{"kind": "low_shelf", "freq_hz": 220.0, "gain_db": 6.0, "q": 0.707}],
    )
    high_shelf = synth.apply_eq(
        signal,
        bands=[{"kind": "high_shelf", "freq_hz": 2_500.0, "gain_db": -6.0, "q": 0.707}],
    )

    low_before = _component_rms(signal, frequency_hz=120.0, sample_rate=sample_rate)
    low_after = _component_rms(low_shelf, frequency_hz=120.0, sample_rate=sample_rate)
    high_before = _component_rms(signal, frequency_hz=5_000.0, sample_rate=sample_rate)
    high_after = _component_rms(
        high_shelf, frequency_hz=5_000.0, sample_rate=sample_rate
    )

    assert low_after > low_before * 1.5
    assert high_after < high_before * 0.7


def test_apply_eq_preserves_stereo_layout() -> None:
    left = _sine_wave(220.0, sample_rate=synth.SAMPLE_RATE)
    right = _sine_wave(880.0, sample_rate=synth.SAMPLE_RATE)
    stereo = np.stack([left, right])

    processed = synth.apply_eq(
        stereo,
        bands=[
            {"kind": "highpass", "cutoff_hz": 100.0, "slope_db_per_oct": 12},
            {"kind": "bell", "freq_hz": 1_000.0, "gain_db": 3.0, "q": 1.0},
        ],
    )

    assert processed.shape == stereo.shape
    assert np.isfinite(processed).all()


def test_apply_compressor_rejects_invalid_topology() -> None:
    with pytest.raises(ValueError, match="topology must be"):
        synth.apply_compressor(
            np.ones(64, dtype=np.float64),
            topology="not_real",
        )


def test_apply_compressor_rejects_negative_release_tail() -> None:
    with pytest.raises(ValueError, match="release_tail_ms must be positive"):
        synth.apply_compressor(
            np.ones(64, dtype=np.float64),
            release_ms=180.0,
            release_tail_ms=-1.0,
        )


def test_apply_compressor_rejects_release_tail_faster_than_primary_release() -> None:
    with pytest.raises(ValueError, match="greater than or equal to release_ms"):
        synth.apply_compressor(
            np.ones(64, dtype=np.float64),
            release_ms=180.0,
            release_tail_ms=90.0,
        )


def test_apply_compressor_reduces_hot_signal() -> None:
    signal = 1.1 * _sine_wave(
        220.0,
        sample_rate=synth.SAMPLE_RATE,
        duration_seconds=1.0,
    )

    processed = synth.apply_compressor(
        signal,
        threshold_db=-18.0,
        ratio=4.0,
        attack_ms=0.5,
        release_ms=120.0,
        knee_db=6.0,
    )

    settled_region = slice(synth.SAMPLE_RATE // 4, None)
    assert (
        np.max(np.abs(processed[settled_region]))
        < np.max(np.abs(signal[settled_region])) * 0.8
    )
    assert processed.shape == signal.shape
    assert np.isfinite(processed).all()


def test_apply_compressor_detector_eq_changes_control_behavior() -> None:
    sample_rate = synth.SAMPLE_RATE
    duration_seconds = 1.0
    time = np.linspace(
        0.0,
        duration_seconds,
        int(sample_rate * duration_seconds),
        endpoint=False,
    )
    low = 1.1 * np.sin(2.0 * np.pi * 70.0 * time)
    high = 0.25 * np.sin(2.0 * np.pi * 1_600.0 * time)
    signal = low + high

    without_detector_eq = synth.apply_compressor(
        signal,
        threshold_db=-22.0,
        ratio=3.5,
        attack_ms=8.0,
        release_ms=180.0,
        knee_db=4.0,
        detector_mode="rms",
    )
    with_detector_eq = synth.apply_compressor(
        signal,
        threshold_db=-22.0,
        ratio=3.5,
        attack_ms=8.0,
        release_ms=180.0,
        knee_db=4.0,
        detector_mode="rms",
        detector_bands=[
            {"kind": "highpass", "cutoff_hz": 180.0, "slope_db_per_oct": 24},
        ],
    )

    high_without_eq = _component_rms(
        without_detector_eq,
        frequency_hz=1_600.0,
        sample_rate=sample_rate,
    )
    high_with_eq = _component_rms(
        with_detector_eq,
        frequency_hz=1_600.0,
        sample_rate=sample_rate,
    )

    assert high_with_eq > high_without_eq * 1.2


def test_apply_compressor_feedforward_and_feedback_differ() -> None:
    sample_rate = synth.SAMPLE_RATE
    signal = np.zeros(sample_rate, dtype=np.float64)
    burst = _sine_wave(220.0, sample_rate=sample_rate, duration_seconds=0.08)
    signal[: burst.size] = 1.2 * burst

    feedforward = synth.apply_compressor(
        signal,
        threshold_db=-24.0,
        ratio=4.0,
        attack_ms=2.0,
        release_ms=220.0,
        topology="feedforward",
    )
    feedback = synth.apply_compressor(
        signal,
        threshold_db=-24.0,
        ratio=4.0,
        attack_ms=2.0,
        release_ms=220.0,
        topology="feedback",
    )

    assert np.max(np.abs(feedforward - feedback)) > 1e-3


def test_apply_compressor_preserves_stereo_layout() -> None:
    left = 1.1 * _sine_wave(220.0, sample_rate=synth.SAMPLE_RATE)
    right = 0.4 * _sine_wave(660.0, sample_rate=synth.SAMPLE_RATE)
    stereo = np.stack([left, right])

    processed = synth.apply_compressor(
        stereo,
        threshold_db=-20.0,
        ratio=3.0,
        attack_ms=6.0,
        release_ms=150.0,
        topology="feedforward",
    )

    assert processed.shape == stereo.shape
    assert np.isfinite(processed).all()


def test_apply_compressor_two_stage_release_recovers_fast_then_tails_out() -> None:
    sample_rate = synth.SAMPLE_RATE
    time = np.linspace(0.0, 1.0, sample_rate, endpoint=False)
    signal = 0.18 * np.sin(2.0 * np.pi * 220.0 * time)
    burst = 1.25 * np.sin(2.0 * np.pi * 220.0 * time[: int(0.10 * sample_rate)])
    signal[: burst.size] = burst

    processed = synth.apply_compressor(
        signal,
        threshold_db=-26.0,
        ratio=4.0,
        attack_ms=0.5,
        release_ms=25.0,
        release_tail_ms=350.0,
        knee_db=2.0,
    )

    early_window = slice(int(0.11 * sample_rate), int(0.16 * sample_rate))
    late_window = slice(int(0.40 * sample_rate), int(0.45 * sample_rate))
    early_recovery_ratio = float(
        np.sqrt(np.mean(np.square(processed[early_window])))
        / np.sqrt(np.mean(np.square(signal[early_window])))
    )
    late_recovery_ratio = float(
        np.sqrt(np.mean(np.square(processed[late_window])))
        / np.sqrt(np.mean(np.square(signal[late_window])))
    )

    assert early_recovery_ratio < late_recovery_ratio
