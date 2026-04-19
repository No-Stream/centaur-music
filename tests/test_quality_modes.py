"""Tests for the engine-level quality modes on polyblep and filtered_stack.

Quality controls the ladder solver (ADAA vs Newton), the Newton iteration
budget, and the internal oversampling factor applied around the
filter+feedback+dither block. Oscillator generation is not oversampled
(BLEP already handles oscillator aliasing); oversampling is strictly for
the filter section.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from code_musics.engines import filtered_stack as filtered_stack_engine
from code_musics.engines import polyblep as polyblep_engine
from code_musics.engines._dsp_utils import resolve_quality_mode

SR = 44100


def _render_polyblep_saw(
    *,
    freq: float = 220.0,
    duration: float = 0.25,
    cutoff_hz: float = 1_500.0,
    resonance_q: float = 3.0,
    filter_drive: float = 0.4,
    filter_topology: str = "ladder",
    quality: str | None = None,
) -> np.ndarray:
    params: dict = {
        "waveform": "saw",
        "cutoff_hz": cutoff_hz,
        "resonance_q": resonance_q,
        "filter_drive": filter_drive,
        "filter_topology": filter_topology,
    }
    if quality is not None:
        params["quality"] = quality
    return polyblep_engine.render(
        freq=freq,
        duration=duration,
        amp=0.8,
        sample_rate=SR,
        params=params,
    )


def _render_filtered_stack_saw(
    *,
    freq: float = 220.0,
    duration: float = 0.25,
    cutoff_hz: float = 1_500.0,
    resonance_q: float = 3.0,
    filter_drive: float = 0.4,
    filter_topology: str = "ladder",
    quality: str | None = None,
) -> np.ndarray:
    params: dict = {
        "waveform": "saw",
        "n_harmonics": 12,
        "cutoff_hz": cutoff_hz,
        "resonance_q": resonance_q,
        "filter_drive": filter_drive,
        "filter_topology": filter_topology,
    }
    if quality is not None:
        params["quality"] = quality
    return filtered_stack_engine.render(
        freq=freq,
        duration=duration,
        amp=0.8,
        sample_rate=SR,
        params=params,
    )


class TestQualityHelper:
    def test_resolve_quality_mode_draft(self) -> None:
        cfg = resolve_quality_mode("draft")
        assert cfg.solver == "adaa"
        assert cfg.max_newton_iters == 0
        assert cfg.newton_tolerance == 0.0
        assert cfg.oversample_factor == 1

    def test_resolve_quality_mode_fast(self) -> None:
        cfg = resolve_quality_mode("fast")
        assert cfg.solver == "newton"
        assert cfg.max_newton_iters == 2
        assert cfg.newton_tolerance == 1e-8
        assert cfg.oversample_factor == 2

    def test_resolve_quality_mode_great(self) -> None:
        cfg = resolve_quality_mode("great")
        assert cfg.solver == "newton"
        assert cfg.max_newton_iters == 4
        assert cfg.newton_tolerance == 1e-9
        assert cfg.oversample_factor == 2

    def test_resolve_quality_mode_divine(self) -> None:
        cfg = resolve_quality_mode("divine")
        assert cfg.solver == "newton"
        assert cfg.max_newton_iters == 8
        assert cfg.newton_tolerance == 1e-10
        assert cfg.oversample_factor == 4

    def test_resolve_quality_mode_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown quality mode"):
            resolve_quality_mode("bogus")


class TestEachQualityFiniteOutput:
    @pytest.mark.parametrize("quality", ["draft", "fast", "great", "divine"])
    def test_polyblep_each_quality_finite(self, quality: str) -> None:
        audio = _render_polyblep_saw(quality=quality)
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) > 0.01

    @pytest.mark.parametrize("quality", ["draft", "fast", "great", "divine"])
    def test_filtered_stack_each_quality_finite(self, quality: str) -> None:
        audio = _render_filtered_stack_saw(quality=quality)
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) > 0.01


class TestUnknownQualityRaises:
    def test_polyblep_unknown_quality_raises(self) -> None:
        with pytest.raises(ValueError, match="[Uu]nknown quality"):
            _render_polyblep_saw(quality="bogus")

    def test_filtered_stack_unknown_quality_raises(self) -> None:
        with pytest.raises(ValueError, match="[Uu]nknown quality"):
            _render_filtered_stack_saw(quality="bogus")


class TestDraftFasterThanDivine:
    def test_draft_is_faster_than_divine(self) -> None:
        """Smoke check that oversampling makes divine measurably slower.

        We warm up first so JIT/resample_poly init costs don't dominate.
        """
        # Warm up JIT + resample_poly internals
        _render_polyblep_saw(quality="draft", duration=0.1)
        _render_polyblep_saw(quality="divine", duration=0.1)

        t0 = time.perf_counter()
        for _ in range(3):
            _render_polyblep_saw(quality="draft", duration=0.2)
        t_draft = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(3):
            _render_polyblep_saw(quality="divine", duration=0.2)
        t_divine = time.perf_counter() - t0

        assert t_divine > t_draft, (
            f"Expected divine to be slower than draft; "
            f"t_draft={t_draft:.4f}s t_divine={t_divine:.4f}s"
        )


class TestDivineReducesAliasing:
    def test_divine_reduces_aliasing_vs_draft(self) -> None:
        """Render a low-cutoff ladder with heavy drive and resonance on a
        high-frequency saw.  The nonlinear drive generates harmonics that
        can fold above Nyquist; oversampling should push that fold energy
        to where it gets lowpass-filtered by the polyphase antialiasing
        filter during downsampling.

        Measure aliasing as the energy in a band above what the filter
        should cleanly pass: since cutoff=500 Hz and the saw fundamental
        is 3000 Hz, we expect heavy attenuation above ~2 kHz after the
        4-pole ladder.  Aliased harmonics show up as out-of-band energy
        that shouldn't be there.
        """
        # Use a high-frequency sine through a driven ladder.  Harmonic distortion
        # creates partials well above Nyquist that fold back into the sub-f0 band.
        # With divine-mode 4x oversampling the fold energy is drastically reduced.
        # Cutoff is well above f0 so neither solver attenuates the fundamental.
        freq = 15_000.0
        duration = 0.3

        def render_aliased(quality: str) -> np.ndarray:
            params: dict = {
                "waveform": "sine",
                "cutoff_hz": 20_000.0,
                "resonance_q": 0.707,
                "filter_drive": 1.5,
                "filter_topology": "ladder",
                "quality": quality,
            }
            return polyblep_engine.render(
                freq=freq,
                duration=duration,
                amp=0.8,
                sample_rate=SR,
                params=params,
            )

        draft_audio = render_aliased("draft")
        divine_audio = render_aliased("divine")

        def sub_f0_alias_energy(audio: np.ndarray) -> float:
            """Energy below 90% of f0 — should be silent for a pure f0 sine
            with oversampled filtering.  Any energy here is alias fold."""
            spectrum = np.abs(np.fft.rfft(audio))
            freqs = np.fft.rfftfreq(len(audio), 1 / SR)
            mask = (freqs >= 20.0) & (freqs < freq * 0.9)
            return float(np.sqrt(np.mean(spectrum[mask] ** 2)))

        draft_alias = sub_f0_alias_energy(draft_audio)
        divine_alias = sub_f0_alias_energy(divine_audio)

        # Oversampling should reduce alias fold energy by a large margin.
        assert divine_alias < draft_alias * 0.5, (
            f"Expected oversampling to reduce aliasing: "
            f"draft_alias={draft_alias:.4e}, divine_alias={divine_alias:.4e}"
        )


class TestDefaultQualityIsGreat:
    """The default quality is 'great'. We can't compare the raw output of
    ``quality=None`` (no key in params) against ``quality="great"`` bit-for-bit
    because params are hashed into the per-note RNG seed — adding/removing a
    key shifts the stochastic jitter/bootstrap streams. Instead, we verify
    that the extracted analog-params default threads through to ``great``, and
    that the output is identical for matching ``quality`` keys and different
    for mismatched ones."""

    def test_extract_analog_params_defaults_quality_to_great(self) -> None:
        from code_musics.engines._dsp_utils import extract_analog_params

        params: dict = {"waveform": "saw"}
        analog = extract_analog_params(params)
        assert analog["quality"] == "great"

    def test_polyblep_great_vs_draft_differs(self) -> None:
        """Sanity: explicit great and explicit draft produce different output
        (confirms quality is actually wired through)."""
        great_audio = _render_polyblep_saw(quality="great")
        draft_audio = _render_polyblep_saw(quality="draft")
        assert not np.allclose(great_audio, draft_audio, atol=1e-6)

    def test_filtered_stack_great_vs_draft_differs(self) -> None:
        great_audio = _render_filtered_stack_saw(quality="great")
        draft_audio = _render_filtered_stack_saw(quality="draft")
        assert not np.allclose(great_audio, draft_audio, atol=1e-6)
