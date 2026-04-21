"""Unified drum_voice engine tests."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines.drum_voice import render

SAMPLE_RATE = 44_100


# ---------------------------------------------------------------------------
# 1. Basic rendering
# ---------------------------------------------------------------------------


def test_default_params_produce_finite_nonzero_correct_length() -> None:
    duration = 0.3
    audio = render(
        freq=50.0,
        duration=duration,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={},
    )
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert len(audio) == int(SAMPLE_RATE * duration)
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_very_short_duration_gives_empty_when_zero_samples() -> None:
    """Duration that rounds to zero samples returns an empty array."""
    audio = render(
        freq=50.0,
        duration=1e-6,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={},
    )
    assert len(audio) == 0


def test_deterministic_output_for_identical_params() -> None:
    kwargs: dict = {
        "freq": 55.0,
        "duration": 0.25,
        "amp": 0.7,
        "sample_rate": SAMPLE_RATE,
        "params": {
            "tone_type": "oscillator",
            "exciter_type": "click",
            "noise_type": "white",
            "noise_level": 0.1,
        },
    }
    first = render(**kwargs)
    second = render(**kwargs)
    assert np.allclose(first, second)


# ---------------------------------------------------------------------------
# 2. Exciter types
# ---------------------------------------------------------------------------


def test_exciter_click_no_tone() -> None:
    audio = render(
        freq=60.0,
        duration=0.15,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={"exciter_type": "click", "exciter_level": 1.0, "tone_type": None},
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_exciter_impulse_no_tone() -> None:
    audio = render(
        freq=60.0,
        duration=0.15,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={"exciter_type": "impulse", "exciter_level": 1.0, "tone_type": None},
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_exciter_multi_tap_no_tone() -> None:
    audio = render(
        freq=60.0,
        duration=0.15,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={"exciter_type": "multi_tap", "exciter_level": 1.0, "tone_type": None},
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


# ---------------------------------------------------------------------------
# 3. Tone types
# ---------------------------------------------------------------------------


def _render_tone_only(tone_type: str, extra_params: dict | None = None) -> np.ndarray:
    params: dict = {
        "tone_type": tone_type,
        "tone_level": 1.0,
        "exciter_type": None,
        "noise_type": None,
        "metallic_type": None,
    }
    if extra_params:
        params.update(extra_params)
    return render(
        freq=80.0,
        duration=0.2,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params=params,
    )


def test_tone_oscillator() -> None:
    audio = _render_tone_only("oscillator")
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_tone_resonator_differs_from_oscillator() -> None:
    osc = _render_tone_only("oscillator")
    res = _render_tone_only(
        "resonator", {"exciter_type": "click", "exciter_level": 1.0}
    )
    assert np.isfinite(res).all()
    assert np.max(np.abs(res)) > 0
    assert not np.allclose(osc, res)


def test_tone_fm_differs_from_oscillator() -> None:
    osc = _render_tone_only("oscillator")
    fm = _render_tone_only("fm", {"tone_fm_ratio": 1.41, "tone_fm_index": 3.0})
    assert np.isfinite(fm).all()
    assert np.max(np.abs(fm)) > 0
    assert not np.allclose(osc, fm)


def test_tone_additive_with_custom_partials() -> None:
    audio = _render_tone_only("additive", {"tone_partial_ratios": [1, 2, 3, 5]})
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


# ---------------------------------------------------------------------------
# 4. Noise types
# ---------------------------------------------------------------------------


def _render_noise_only(noise_type: str, extra_params: dict | None = None) -> np.ndarray:
    params: dict = {
        "tone_type": None,
        "exciter_type": None,
        "noise_type": noise_type,
        "noise_level": 1.0,
        "metallic_type": None,
    }
    if extra_params:
        params.update(extra_params)
    return render(
        freq=200.0,
        duration=0.15,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params=params,
    )


def test_noise_white() -> None:
    audio = _render_noise_only("white")
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_noise_colored_differs_from_white() -> None:
    white = _render_noise_only("white")
    colored = _render_noise_only("colored")
    assert np.isfinite(colored).all()
    # Spectral shape should differ even though both use the same RNG seed
    # (highpass filtering removes low frequencies).
    assert not np.allclose(white, colored)


def test_noise_bandpass() -> None:
    audio = _render_noise_only("bandpass")
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_noise_comb() -> None:
    audio = _render_noise_only("comb")
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


# ---------------------------------------------------------------------------
# 5. Metallic layer
# ---------------------------------------------------------------------------


def _render_metallic_only(extra_params: dict | None = None) -> np.ndarray:
    params: dict = {
        "tone_type": None,
        "exciter_type": None,
        "noise_type": None,
        "metallic_type": "partials",
        "metallic_level": 1.0,
    }
    if extra_params:
        params.update(extra_params)
    return render(
        freq=300.0,
        duration=0.15,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params=params,
    )


def test_metallic_partials() -> None:
    audio = _render_metallic_only()
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_metallic_square_mode_differs_from_sine() -> None:
    sine = _render_metallic_only({"metallic_oscillator_mode": "sine"})
    square = _render_metallic_only({"metallic_oscillator_mode": "square"})
    assert np.isfinite(square).all()
    assert not np.allclose(sine, square)


# ---------------------------------------------------------------------------
# 6. Creative combinations
# ---------------------------------------------------------------------------


def test_snare_like_oscillator_tone_plus_comb_noise() -> None:
    audio = render(
        freq=180.0,
        duration=0.25,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={
            "tone_type": "oscillator",
            "tone_level": 0.6,
            "noise_type": "comb",
            "noise_level": 0.4,
        },
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_hat_like_click_exciter_metallic_no_tone() -> None:
    audio = render(
        freq=400.0,
        duration=0.1,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={
            "tone_type": None,
            "exciter_type": "click",
            "exciter_level": 0.3,
            "metallic_type": "partials",
            "metallic_level": 1.0,
        },
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_808_kick_resonator_tone_click_exciter() -> None:
    audio = render(
        freq=50.0,
        duration=0.4,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={
            "tone_type": "resonator",
            "tone_level": 1.0,
            "exciter_type": "click",
            "exciter_level": 0.15,
        },
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


def test_full_hybrid_all_layers() -> None:
    audio = render(
        freq=120.0,
        duration=0.3,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={
            "tone_type": "fm",
            "tone_level": 0.5,
            "tone_fm_ratio": 1.41,
            "tone_fm_index": 3.0,
            "exciter_type": "click",
            "exciter_level": 0.2,
            "noise_type": "bandpass",
            "noise_level": 0.3,
            "metallic_type": "partials",
            "metallic_level": 0.4,
        },
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0


# ---------------------------------------------------------------------------
# 7. Features
# ---------------------------------------------------------------------------


def test_tone_shaper_changes_output() -> None:
    base_params: dict = {
        "tone_type": "oscillator",
        "tone_level": 1.0,
        "exciter_type": None,
        "noise_type": None,
        "metallic_type": None,
    }
    plain = render(
        freq=60.0, duration=0.2, amp=0.8, sample_rate=SAMPLE_RATE, params=base_params
    )
    shaped = render(
        freq=60.0,
        duration=0.2,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={**base_params, "tone_shaper": "tanh", "tone_shaper_drive": 0.8},
    )
    assert np.isfinite(shaped).all()
    assert not np.allclose(plain, shaped)


def test_voice_filter_changes_output() -> None:
    base_params: dict = {
        "tone_type": "oscillator",
        "tone_level": 1.0,
        "exciter_type": None,
        "noise_type": None,
        "metallic_type": None,
    }
    unfiltered = render(
        freq=60.0, duration=0.2, amp=0.8, sample_rate=SAMPLE_RATE, params=base_params
    )
    filtered = render(
        freq=60.0,
        duration=0.2,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={**base_params, "filter_mode": "lowpass", "filter_cutoff_hz": 500.0},
    )
    assert np.isfinite(filtered).all()
    assert not np.allclose(unfiltered, filtered)


def test_custom_tone_envelope_changes_output() -> None:
    base_params: dict = {
        "tone_type": "oscillator",
        "tone_level": 1.0,
        "exciter_type": None,
        "noise_type": None,
        "metallic_type": None,
    }
    default_env = render(
        freq=60.0, duration=0.3, amp=0.8, sample_rate=SAMPLE_RATE, params=base_params
    )
    custom_env = render(
        freq=60.0,
        duration=0.3,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={
            **base_params,
            "tone_envelope": [
                {"time": 0.0, "value": 1.0},
                {"time": 0.5, "value": 0.8, "curve": "linear"},
                {"time": 1.0, "value": 0.0, "curve": "exponential"},
            ],
        },
    )
    assert np.isfinite(custom_env).all()
    assert not np.allclose(default_env, custom_env)


def test_freq_trajectory_produces_different_output() -> None:
    duration = 0.2
    n_samples = int(SAMPLE_RATE * duration)
    base_params: dict = {
        "tone_type": "oscillator",
        "tone_level": 1.0,
        "exciter_type": None,
        "noise_type": None,
        "metallic_type": None,
    }

    static = render(
        freq=60.0,
        duration=duration,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params=base_params,
    )
    trajectory = np.linspace(60.0, 120.0, n_samples)
    swept = render(
        freq=60.0,
        duration=duration,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params=base_params,
        freq_trajectory=trajectory,
    )
    assert np.isfinite(swept).all()
    assert not np.allclose(static, swept)


def test_tone_wave_triangle_and_sine_clip() -> None:
    base_params: dict = {
        "tone_type": "oscillator",
        "tone_level": 1.0,
        "exciter_type": None,
        "noise_type": None,
        "metallic_type": None,
    }
    for wave in ("triangle", "sine_clip"):
        audio = render(
            freq=60.0,
            duration=0.15,
            amp=0.8,
            sample_rate=SAMPLE_RATE,
            params={**base_params, "tone_wave": wave},
        )
        assert np.isfinite(audio).all(), f"tone_wave={wave!r} produced non-finite"
        assert np.max(np.abs(audio)) > 0, f"tone_wave={wave!r} produced silence"


# ---------------------------------------------------------------------------
# 8. Validation
# ---------------------------------------------------------------------------


def test_invalid_tone_type_raises() -> None:
    with pytest.raises(ValueError, match="tone_type"):
        render(
            freq=50.0,
            duration=0.2,
            amp=0.8,
            sample_rate=SAMPLE_RATE,
            params={"tone_type": "bogus"},
        )


def test_invalid_exciter_type_raises() -> None:
    with pytest.raises(ValueError, match="exciter_type"):
        render(
            freq=50.0,
            duration=0.2,
            amp=0.8,
            sample_rate=SAMPLE_RATE,
            params={"exciter_type": "bogus"},
        )


def test_invalid_metallic_type_raises() -> None:
    with pytest.raises(ValueError, match="metallic_type"):
        render(
            freq=50.0,
            duration=0.2,
            amp=0.8,
            sample_rate=SAMPLE_RATE,
            params={"metallic_type": "bogus", "metallic_level": 1.0},
        )


# ---------------------------------------------------------------------------
# 9. Layer independence
# ---------------------------------------------------------------------------


def test_all_layers_disabled_produces_silence() -> None:
    audio = render(
        freq=60.0,
        duration=0.15,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={
            "tone_type": None,
            "exciter_type": None,
            "noise_type": None,
            "metallic_type": None,
        },
    )
    assert np.allclose(audio, 0.0)


# ---------------------------------------------------------------------------
# 10. Saturation / preamp shaper dispatch
# ---------------------------------------------------------------------------

_TONE_ONLY_PARAMS: dict = {
    "tone_type": "oscillator",
    "tone_level": 1.0,
    "exciter_type": None,
    "noise_type": None,
    "metallic_type": None,
}


class TestSaturationShaperDispatch:
    """Verify that shaper='saturation' dispatches to the real saturation effect."""

    def test_tone_shaper_saturation_renders_finite_nonzero(self) -> None:
        audio = render(
            freq=60.0,
            duration=0.2,
            amp=0.8,
            sample_rate=SAMPLE_RATE,
            params={**_TONE_ONLY_PARAMS, "tone_shaper": "saturation"},
        )
        assert np.isfinite(audio).all()
        assert np.max(np.abs(audio)) > 0

    def test_tone_shaper_saturation_differs_from_no_shaper(self) -> None:
        plain = render(
            freq=60.0,
            duration=0.2,
            amp=0.8,
            sample_rate=SAMPLE_RATE,
            params=_TONE_ONLY_PARAMS,
        )
        shaped = render(
            freq=60.0,
            duration=0.2,
            amp=0.8,
            sample_rate=SAMPLE_RATE,
            params={
                **_TONE_ONLY_PARAMS,
                "tone_shaper": "saturation",
                "tone_shaper_drive": 1.5,
            },
        )
        assert not np.allclose(plain, shaped)

    def test_tone_shaper_mode_tube_differs_from_iron(self) -> None:
        tube = render(
            freq=60.0,
            duration=0.2,
            amp=0.8,
            sample_rate=SAMPLE_RATE,
            params={
                **_TONE_ONLY_PARAMS,
                "tone_shaper": "saturation",
                "tone_shaper_drive": 1.5,
                "tone_shaper_mode": "tube",
            },
        )
        iron = render(
            freq=60.0,
            duration=0.2,
            amp=0.8,
            sample_rate=SAMPLE_RATE,
            params={
                **_TONE_ONLY_PARAMS,
                "tone_shaper": "saturation",
                "tone_shaper_drive": 1.5,
                "tone_shaper_mode": "iron",
            },
        )
        assert np.isfinite(tube).all()
        assert np.isfinite(iron).all()
        assert not np.allclose(tube, iron)

    def test_voice_shaper_saturation_renders_finite_nonzero(self) -> None:
        audio = render(
            freq=60.0,
            duration=0.2,
            amp=0.8,
            sample_rate=SAMPLE_RATE,
            params={**_TONE_ONLY_PARAMS, "shaper": "saturation", "shaper_drive": 1.5},
        )
        assert np.isfinite(audio).all()
        assert np.max(np.abs(audio)) > 0


class TestPreampShaperDispatch:
    """Verify that shaper='preamp' dispatches to the real preamp effect."""

    def test_tone_shaper_preamp_renders_finite_nonzero(self) -> None:
        audio = render(
            freq=60.0,
            duration=0.2,
            amp=0.8,
            sample_rate=SAMPLE_RATE,
            params={**_TONE_ONLY_PARAMS, "tone_shaper": "preamp"},
        )
        assert np.isfinite(audio).all()
        assert np.max(np.abs(audio)) > 0

    def test_tone_shaper_preamp_differs_from_no_shaper(self) -> None:
        plain = render(
            freq=60.0,
            duration=0.2,
            amp=0.8,
            sample_rate=SAMPLE_RATE,
            params=_TONE_ONLY_PARAMS,
        )
        shaped = render(
            freq=60.0,
            duration=0.2,
            amp=0.8,
            sample_rate=SAMPLE_RATE,
            params={
                **_TONE_ONLY_PARAMS,
                "tone_shaper": "preamp",
                "tone_shaper_drive": 0.8,
            },
        )
        assert not np.allclose(plain, shaped)

    def test_voice_shaper_preamp_renders_finite_nonzero(self) -> None:
        audio = render(
            freq=60.0,
            duration=0.2,
            amp=0.8,
            sample_rate=SAMPLE_RATE,
            params={**_TONE_ONLY_PARAMS, "shaper": "preamp", "shaper_drive": 0.8},
        )
        assert np.isfinite(audio).all()
        assert np.max(np.abs(audio)) > 0
