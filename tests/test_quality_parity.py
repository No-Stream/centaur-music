"""Quality-mode parity tests for va / synth_voice / drum_voice.

The ``quality`` surface (``draft`` / ``fast`` / ``great`` / ``divine``) was
originally wired on ``polyblep`` and ``filtered_stack``.  These tests cover
the port to the remaining three tonal engines and lock in the
engine-specific defaults:

- ``va`` -> ``"fast"``   (era-accurate JP-8000 / Virus / Q character)
- ``synth_voice`` -> ``"great"``   (modern-clean default for the preferred tonal engine)
- ``drum_voice`` -> ``"great"``    (transient content benefits from OS)

Focus: default behavior, unknown-quality rejection, finite output across
tiers, and alias-floor reduction at ``divine`` vs ``draft`` under drive.
"""

from __future__ import annotations

import inspect
import re
from types import ModuleType

import numpy as np
import pytest

from code_musics.engines import drum_voice as drum_voice_engine
from code_musics.engines import synth_voice as synth_voice_engine
from code_musics.engines import va as va_engine

SR = 44100


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _render_va(
    *,
    freq: float = 220.0,
    duration: float = 0.2,
    filter_drive: float = 0.2,
    filter_topology: str = "ladder",
    cutoff_hz: float = 1500.0,
    resonance_q: float = 2.0,
    quality: str | None = None,
) -> np.ndarray:
    params: dict = {
        "_voice_name": "va_quality_test",
        "osc_mode": "supersaw",
        "cutoff_hz": cutoff_hz,
        "resonance_q": resonance_q,
        "filter_drive": filter_drive,
        "filter_topology": filter_topology,
    }
    if quality is not None:
        params["quality"] = quality
    return va_engine.render(
        freq=freq, duration=duration, amp=0.8, sample_rate=SR, params=params
    )


def _render_synth_voice(
    *,
    freq: float = 220.0,
    duration: float = 0.2,
    filter_drive: float = 0.4,
    filter_topology: str = "ladder",
    filter_cutoff_hz: float = 1500.0,
    resonance_q: float = 2.0,
    quality: str | None = None,
) -> np.ndarray:
    params: dict = {
        "osc_type": "polyblep",
        "osc_waveform": "saw",
        "filter_mode": "lowpass",
        "filter_topology": filter_topology,
        "filter_cutoff_hz": filter_cutoff_hz,
        "resonance_q": resonance_q,
        "filter_drive": filter_drive,
    }
    if quality is not None:
        params["quality"] = quality
    return synth_voice_engine.render(
        freq=freq, duration=duration, amp=0.8, sample_rate=SR, params=params
    )


def _render_drum_voice(
    *,
    freq: float = 120.0,
    duration: float = 0.25,
    filter_drive: float = 0.4,
    filter_topology: str = "ladder",
    filter_cutoff_hz: float = 1200.0,
    filter_q: float = 2.0,
    quality: str | None = None,
) -> np.ndarray:
    params: dict = {
        "tone_type": "oscillator",
        "filter_mode": "lowpass",
        "filter_topology": filter_topology,
        "filter_cutoff_hz": filter_cutoff_hz,
        "filter_q": filter_q,
        "filter_drive": filter_drive,
    }
    if quality is not None:
        params["quality"] = quality
    return drum_voice_engine.render(
        freq=freq, duration=duration, amp=0.8, sample_rate=SR, params=params
    )


# ---------------------------------------------------------------------------
# API surface: each engine accepts ``quality`` and rejects unknown values
# ---------------------------------------------------------------------------


class TestQualityAPISurface:
    @pytest.mark.parametrize("quality", ["draft", "fast", "great", "divine"])
    def test_va_accepts_quality(self, quality: str) -> None:
        audio = _render_va(quality=quality)
        assert audio.size > 0
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) > 0.01

    @pytest.mark.parametrize("quality", ["draft", "fast", "great", "divine"])
    def test_synth_voice_accepts_quality(self, quality: str) -> None:
        audio = _render_synth_voice(quality=quality)
        assert audio.size > 0
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) > 0.01

    @pytest.mark.parametrize("quality", ["draft", "fast", "great", "divine"])
    def test_drum_voice_accepts_quality(self, quality: str) -> None:
        audio = _render_drum_voice(quality=quality)
        assert audio.size > 0
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) > 0.01

    def test_va_rejects_unknown_quality(self) -> None:
        with pytest.raises(ValueError, match="[Uu]nknown quality"):
            _render_va(quality="bogus")

    def test_synth_voice_rejects_unknown_quality(self) -> None:
        with pytest.raises(ValueError, match="[Uu]nknown quality"):
            _render_synth_voice(quality="bogus")

    def test_drum_voice_rejects_unknown_quality(self) -> None:
        with pytest.raises(ValueError, match="[Uu]nknown quality"):
            _render_drum_voice(quality="bogus")


# ---------------------------------------------------------------------------
# Defaults: va -> "fast", synth_voice -> "great", drum_voice -> "great"
# ---------------------------------------------------------------------------


class TestQualityDefaults:
    """Verify that each engine's default quality tier routes to the documented
    tier.  Strategy: grep the engine module's source for the hardcoded
    default quality string (either the ``params.get("quality", <default>)``
    call for synth_voice/drum_voice, or the ``"quality": <default>`` entry
    in the engine defaults dict for va) and assert it matches the intended
    tier.

    This is strictly stronger than inferring the tier from spectral
    fingerprints: ``fast`` (Newton 2-iter) and ``great`` (Newton 4-iter)
    share OS factor and produce numerically indistinguishable output on
    signals that converge within 2 iterations — which is every bench
    signal we have.  Source inspection pins the tier uniquely regardless
    of whether the tier actually changes the audio, so a silent default
    flip from ``great`` to ``fast`` is caught at the authoring surface.
    """

    def test_va_default_is_fast(self) -> None:
        # ``va`` injects the default into a defaults dict before params
        # are resolved; look for that entry.
        default = _va_default_quality()
        assert default == "fast", (
            f"va default quality tier is {default!r}, expected 'fast'"
        )

    def test_synth_voice_default_is_great(self) -> None:
        default = _params_get_quality_default(synth_voice_engine)
        assert default == "great", (
            f"synth_voice default quality tier is {default!r}, expected 'great'"
        )

    def test_drum_voice_default_is_great(self) -> None:
        default = _params_get_quality_default(drum_voice_engine)
        assert default == "great", (
            f"drum_voice default quality tier is {default!r}, expected 'great'"
        )


_PARAMS_GET_QUALITY_RE = re.compile(
    r"""params\.get\(\s*["']quality["']\s*,\s*["']([a-z]+)["']\s*\)""",
    re.VERBOSE,
)

_VA_DEFAULTS_QUALITY_RE = re.compile(
    r"""["']quality["']\s*:\s*["']([a-z]+)["']""",
    re.VERBOSE,
)


def _params_get_quality_default(module: ModuleType) -> str:
    """Return the literal default string from the engine's
    ``params.get("quality", <default>)`` call — synth_voice and
    drum_voice both resolve the tier this way.

    Fails loudly if the engine source doesn't contain this exact pattern:
    the route through which the default tier is picked has changed and
    the test is no longer asserting what it thinks it is.
    """
    src = inspect.getsource(module)
    matches = _PARAMS_GET_QUALITY_RE.findall(src)
    assert matches, (
        f"engine {module.__name__!r} source does not contain a "
        f'`params.get("quality", <default>)` call — the default '
        f"quality routing has moved; update this test"
    )
    # All calls in one engine should agree on the default — guard against
    # a drift-by-copy-paste bug where different code paths set different
    # defaults.
    unique = set(matches)
    assert len(unique) == 1, (
        f"engine {module.__name__!r} has inconsistent default quality "
        f"strings across `params.get` calls: {sorted(unique)}"
    )
    return matches[0]


def _va_default_quality() -> str:
    """Return the ``"quality"`` default from the ``va`` engine's
    ``_apply_va_defaults`` function.  We grep the source of that
    function specifically so we don't accidentally match unrelated
    ``"quality": ...`` strings elsewhere in the module (e.g. presets).
    """
    apply_fn = getattr(va_engine, "_apply_va_defaults", None)
    assert apply_fn is not None, (
        "va module is missing _apply_va_defaults — default-tier routing "
        "has moved; update this test"
    )
    src = inspect.getsource(apply_fn)
    matches = _VA_DEFAULTS_QUALITY_RE.findall(src)
    assert matches, (
        'va._apply_va_defaults source does not contain a `"quality": '
        "<default>` entry — default-tier routing has moved; update this test"
    )
    unique = set(matches)
    assert len(unique) == 1, (
        f"va._apply_va_defaults has inconsistent default quality strings: "
        f"{sorted(unique)}"
    )
    return matches[0]


# ---------------------------------------------------------------------------
# Alias-floor reduction: divine oversampling pushes harmonic fold energy
# above Nyquist into the polyphase downsample filter's stopband.
# ---------------------------------------------------------------------------


def _sub_f0_alias_rms(audio: np.ndarray, f0_hz: float) -> float:
    """RMS energy below ``0.9 * f0`` — a band that should be near-silent
    for a pure-sine source through a clean filter.  Any energy here is
    alias fold from drive-generated harmonics above Nyquist bouncing back
    into the sub-f0 region.  Mirrors the measurement used by the polyblep
    quality-modes test suite (`tests/test_quality_modes.py`).
    """
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / SR)
    mask = (freqs >= 20.0) & (freqs < f0_hz * 0.9)
    if not mask.any():
        return 0.0
    return float(np.sqrt(np.mean(spectrum[mask] ** 2)))


# Alias rig: a pure high-frequency sine driven hard into a wide-open ladder
# generates harmonics well above Nyquist; 4x oversampling at ``divine``
# pushes the fold energy into the polyphase downsample filter's stopband
# where it is attenuated, while ``draft`` lets it bounce back into the
# sub-f0 silent region.
_ALIAS_F0_HZ = 15_000.0
_ALIAS_DURATION_S = 0.3
_ALIAS_CUTOFF_HZ = 20_000.0
_ALIAS_RESONANCE_Q = 0.707
_ALIAS_DRIVE = 0.7


def _alias_render_va(quality: str | None = None) -> np.ndarray:
    # Spectralwave mode lets us render a near-pure-sine partial (partial 1
    # with minimal spectral_position) for the cleanest alias measurement.
    # ``quality=None`` omits the key entirely so the engine's default tier
    # routes through its normal code path (no kwarg perturbation).
    params: dict = {
        "_voice_name": "va_alias_test",
        "osc_mode": "spectralwave",
        "spectral_position": 0.0,
        "n_partials": 1,
        "cutoff_hz": _ALIAS_CUTOFF_HZ,
        "resonance_q": _ALIAS_RESONANCE_Q,
        "filter_drive": _ALIAS_DRIVE,
        "filter_topology": "ladder",
    }
    if quality is not None:
        params["quality"] = quality
    return va_engine.render(
        freq=_ALIAS_F0_HZ,
        duration=_ALIAS_DURATION_S,
        amp=0.8,
        sample_rate=SR,
        params=params,
    )


def _alias_render_synth_voice(quality: str | None = None) -> np.ndarray:
    # Only the noise slot isn't a sine; we spin up the osc slot with a sine
    # waveform so the input to the filter is spectrally clean.
    params: dict = {
        "osc_type": "polyblep",
        "osc_waveform": "sine",
        "filter_mode": "lowpass",
        "filter_topology": "ladder",
        "filter_cutoff_hz": _ALIAS_CUTOFF_HZ,
        "resonance_q": _ALIAS_RESONANCE_Q,
        "filter_drive": _ALIAS_DRIVE,
    }
    if quality is not None:
        params["quality"] = quality
    return synth_voice_engine.render(
        freq=_ALIAS_F0_HZ,
        duration=_ALIAS_DURATION_S,
        amp=0.8,
        sample_rate=SR,
        params=params,
    )


def _alias_render_drum_voice(quality: str | None = None) -> np.ndarray:
    # Tone oscillator with a flat pitch sweep (ratio=1, decay ~inf) gives a
    # near-pure sine through the voice filter — matching the polyblep and
    # synth_voice rigs.
    params: dict = {
        "tone_type": "oscillator",
        "tone_waveform": "sine",
        "tone_sweep_ratio": 1.0,
        "tone_sweep_decay_s": 1000.0,
        "tone_decay_s": 1.0,
        "exciter_level": 0.0,
        "noise_level": 0.0,
        "filter_mode": "lowpass",
        "filter_topology": "ladder",
        "filter_cutoff_hz": _ALIAS_CUTOFF_HZ,
        "filter_q": _ALIAS_RESONANCE_Q,
        "filter_drive": _ALIAS_DRIVE,
    }
    if quality is not None:
        params["quality"] = quality
    return drum_voice_engine.render(
        freq=_ALIAS_F0_HZ,
        duration=_ALIAS_DURATION_S,
        amp=0.8,
        sample_rate=SR,
        params=params,
    )


class TestDivineReducesAliasing:
    """For each engine, the ``divine`` tier must measurably reduce alias
    energy versus ``draft``.  Measurement: sub-f0 energy on a high-frequency
    pure-sine source driven hard through a wide-open ladder.  With 4x OS
    the drive's upper-Nyquist harmonics are filtered out by the polyphase
    downsampler; without OS they fold back into the sub-f0 silent region.
    Tolerance is relative (0.5x) to absorb measurement noise.
    """

    def test_va_divine_lower_alias_floor(self) -> None:
        draft = _alias_render_va("draft")
        divine = _alias_render_va("divine")
        draft_alias = _sub_f0_alias_rms(draft, _ALIAS_F0_HZ)
        divine_alias = _sub_f0_alias_rms(divine, _ALIAS_F0_HZ)
        assert draft_alias > 0.0
        assert divine_alias < draft_alias * 0.5, (
            f"va: expected divine < 50% of draft sub-f0 alias energy; "
            f"draft={draft_alias:.4e} divine={divine_alias:.4e}"
        )

    def test_synth_voice_divine_lower_alias_floor(self) -> None:
        draft = _alias_render_synth_voice("draft")
        divine = _alias_render_synth_voice("divine")
        draft_alias = _sub_f0_alias_rms(draft, _ALIAS_F0_HZ)
        divine_alias = _sub_f0_alias_rms(divine, _ALIAS_F0_HZ)
        assert draft_alias > 0.0
        assert divine_alias < draft_alias * 0.5, (
            f"synth_voice: expected divine < 50% of draft sub-f0 alias energy; "
            f"draft={draft_alias:.4e} divine={divine_alias:.4e}"
        )

    def test_drum_voice_divine_lower_alias_floor(self) -> None:
        draft = _alias_render_drum_voice("draft")
        divine = _alias_render_drum_voice("divine")
        draft_alias = _sub_f0_alias_rms(draft, _ALIAS_F0_HZ)
        divine_alias = _sub_f0_alias_rms(divine, _ALIAS_F0_HZ)
        assert draft_alias > 0.0
        assert divine_alias < draft_alias * 0.5, (
            f"drum_voice: expected divine < 50% of draft sub-f0 alias energy; "
            f"draft={draft_alias:.4e} divine={divine_alias:.4e}"
        )
