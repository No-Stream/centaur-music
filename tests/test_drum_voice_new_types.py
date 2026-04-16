"""Tests for new drum_voice synthesis types: fm_burst, noise_burst, ring_mod, fm_cluster."""

from __future__ import annotations

import numpy as np

from code_musics.engines.drum_voice import render

SAMPLE_RATE = 44_100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render(
    *,
    freq: float = 200.0,
    duration: float = 0.15,
    amp: float = 0.8,
    params: dict | None = None,
) -> np.ndarray:
    return render(
        freq=freq,
        duration=duration,
        amp=amp,
        sample_rate=SAMPLE_RATE,
        params=params or {},
    )


# ---------------------------------------------------------------------------
# 1. Exciter: fm_burst
# ---------------------------------------------------------------------------


class TestExciterFmBurst:
    def test_fm_burst_no_tone_produces_finite_nonzero(self) -> None:
        audio = _render(
            params={
                "exciter_type": "fm_burst",
                "exciter_level": 1.0,
                "tone_type": None,
            },
        )
        assert np.isfinite(audio).all()
        assert np.max(np.abs(audio)) > 0

    def test_fm_burst_custom_params_differ_from_defaults(self) -> None:
        default = _render(
            params={
                "exciter_type": "fm_burst",
                "exciter_level": 1.0,
                "tone_type": None,
            },
        )
        custom = _render(
            params={
                "exciter_type": "fm_burst",
                "exciter_level": 1.0,
                "tone_type": None,
                "exciter_fm_ratio": 3.0,
                "exciter_fm_index": 8.0,
            },
        )
        assert np.isfinite(custom).all()
        assert not np.allclose(default, custom)

    def test_fm_burst_deterministic(self) -> None:
        params: dict = {
            "exciter_type": "fm_burst",
            "exciter_level": 1.0,
            "tone_type": None,
        }
        first = _render(params=params)
        second = _render(params=params)
        assert np.allclose(first, second)


# ---------------------------------------------------------------------------
# 2. Exciter: noise_burst
# ---------------------------------------------------------------------------


class TestExciterNoiseBurst:
    def test_noise_burst_no_tone_produces_finite_nonzero(self) -> None:
        audio = _render(
            params={
                "exciter_type": "noise_burst",
                "exciter_level": 1.0,
                "tone_type": None,
            },
        )
        assert np.isfinite(audio).all()
        assert np.max(np.abs(audio)) > 0

    def test_noise_burst_with_filter_differs_from_unfiltered(self) -> None:
        unfiltered = _render(
            params={
                "exciter_type": "noise_burst",
                "exciter_level": 1.0,
                "tone_type": None,
            },
        )
        filtered = _render(
            params={
                "exciter_type": "noise_burst",
                "exciter_level": 1.0,
                "tone_type": None,
                "exciter_filter_cutoff_hz": 2000.0,
            },
        )
        assert np.isfinite(filtered).all()
        assert not np.allclose(unfiltered, filtered)

    def test_noise_burst_deterministic(self) -> None:
        params: dict = {
            "exciter_type": "noise_burst",
            "exciter_level": 1.0,
            "tone_type": None,
        }
        first = _render(params=params)
        second = _render(params=params)
        assert np.allclose(first, second)


# ---------------------------------------------------------------------------
# 3. Metallic: ring_mod
# ---------------------------------------------------------------------------


class TestMetallicRingMod:
    def test_ring_mod_no_tone_produces_finite_nonzero(self) -> None:
        audio = _render(
            freq=400.0,
            params={
                "tone_type": None,
                "exciter_type": None,
                "noise_type": None,
                "metallic_type": "ring_mod",
                "metallic_level": 1.0,
            },
        )
        assert np.isfinite(audio).all()
        assert np.max(np.abs(audio)) > 0

    def test_ring_mod_amount_zero_vs_one_differ(self) -> None:
        dry = _render(
            freq=400.0,
            params={
                "tone_type": None,
                "exciter_type": None,
                "noise_type": None,
                "metallic_type": "ring_mod",
                "metallic_level": 1.0,
                "metallic_ring_mod_amount": 0.0,
            },
        )
        wet = _render(
            freq=400.0,
            params={
                "tone_type": None,
                "exciter_type": None,
                "noise_type": None,
                "metallic_type": "ring_mod",
                "metallic_level": 1.0,
                "metallic_ring_mod_amount": 1.0,
            },
        )
        assert np.isfinite(dry).all()
        assert np.isfinite(wet).all()
        assert not np.allclose(dry, wet)

    def test_ring_mod_deterministic(self) -> None:
        params: dict = {
            "tone_type": None,
            "exciter_type": None,
            "noise_type": None,
            "metallic_type": "ring_mod",
            "metallic_level": 1.0,
        }
        first = _render(freq=400.0, params=params)
        second = _render(freq=400.0, params=params)
        assert np.allclose(first, second)


# ---------------------------------------------------------------------------
# 4. Metallic: fm_cluster
# ---------------------------------------------------------------------------


class TestMetallicFmCluster:
    def test_fm_cluster_no_tone_produces_finite_nonzero(self) -> None:
        audio = _render(
            freq=400.0,
            params={
                "tone_type": None,
                "exciter_type": None,
                "noise_type": None,
                "metallic_type": "fm_cluster",
                "metallic_level": 1.0,
            },
        )
        assert np.isfinite(audio).all()
        assert np.max(np.abs(audio)) > 0

    def test_fm_cluster_custom_ratios_differ_from_default(self) -> None:
        default = _render(
            freq=400.0,
            params={
                "tone_type": None,
                "exciter_type": None,
                "noise_type": None,
                "metallic_type": "fm_cluster",
                "metallic_level": 1.0,
            },
        )
        custom = _render(
            freq=400.0,
            params={
                "tone_type": None,
                "exciter_type": None,
                "noise_type": None,
                "metallic_type": "fm_cluster",
                "metallic_level": 1.0,
                "metallic_fm_ratios": [1.0, 1.73, 2.65, 3.14],
            },
        )
        assert np.isfinite(custom).all()
        assert not np.allclose(default, custom)

    def test_fm_cluster_deterministic(self) -> None:
        params: dict = {
            "tone_type": None,
            "exciter_type": None,
            "noise_type": None,
            "metallic_type": "fm_cluster",
            "metallic_level": 1.0,
        }
        first = _render(freq=400.0, params=params)
        second = _render(freq=400.0, params=params)
        assert np.allclose(first, second)


# ---------------------------------------------------------------------------
# 5. Creative combo
# ---------------------------------------------------------------------------


class TestCreativeCombos:
    def test_fm_burst_exciter_plus_fm_cluster_metallic(self) -> None:
        audio = _render(
            freq=300.0,
            params={
                "exciter_type": "fm_burst",
                "exciter_level": 0.3,
                "tone_type": None,
                "noise_type": None,
                "metallic_type": "fm_cluster",
                "metallic_level": 1.0,
            },
        )
        assert np.isfinite(audio).all()
        assert np.max(np.abs(audio)) > 0
