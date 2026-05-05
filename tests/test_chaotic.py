"""Chaotic oscillator tests.

Covers:

- :mod:`code_musics.engines._chaotic` raw integrators (finite, bounded,
  deterministic, seed-sensitive, empty-buffer handling).
- ``osc_type="chaotic"`` dispatch through ``synth_voice``.
- The four curated presets wired through ``render_note_signal``.
- :class:`code_musics.modulation.ChaoticSource` sampling semantics and
  integration through a full score render via the modulation matrix.
- Off-state preservation: without ``osc_type="chaotic"``, nothing changes.
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.automation import AutomationTarget
from code_musics.engines._chaotic import (
    RATE_HZ_CEILINGS,
    SUPPORTED_SYSTEMS,
    render_chaotic,
)
from code_musics.engines.registry import render_note_signal
from code_musics.engines.synth_voice import render as synth_voice_render
from code_musics.modulation import ChaoticSource, ModConnection
from code_musics.score import NoteEvent, Phrase, Score

SAMPLE_RATE = 44_100


# ---------------------------------------------------------------------------
# Raw integrator — _chaotic.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("system", sorted(SUPPORTED_SYSTEMS))
def test_render_chaotic_finite_bounded_nonzero(system: str) -> None:
    n = SAMPLE_RATE // 4  # ~0.25 s
    audio = render_chaotic(
        system=system,
        rate_hz=12.0,
        amount=0.75,
        symmetry=0.0,
        n_samples=n,
        sample_rate=SAMPLE_RATE,
        seed=17,
    )
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float64
    assert audio.shape == (n,)
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) > 0.05, "chaotic output should not be silent"
    # Peak normalization caps at ~1.0; allow a small head-room margin.
    assert np.max(np.abs(audio)) <= 1.5


@pytest.mark.parametrize("system", sorted(SUPPORTED_SYSTEMS))
def test_render_chaotic_is_deterministic(system: str) -> None:
    n = SAMPLE_RATE // 10
    kwargs = dict(
        system=system,
        rate_hz=10.0,
        amount=0.6,
        symmetry=0.2,
        n_samples=n,
        sample_rate=SAMPLE_RATE,
        seed=42,
    )
    a = render_chaotic(**kwargs)
    b = render_chaotic(**kwargs)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("system", sorted(SUPPORTED_SYSTEMS))
def test_render_chaotic_seed_changes_output(system: str) -> None:
    n = SAMPLE_RATE // 10
    base = dict(
        system=system,
        rate_hz=10.0,
        amount=0.6,
        symmetry=0.1,
        n_samples=n,
        sample_rate=SAMPLE_RATE,
    )
    a = render_chaotic(**base, seed=1)
    b = render_chaotic(**base, seed=2)
    # Different seeds should produce different trajectories almost surely.
    assert not np.allclose(a, b)


def test_render_chaotic_zero_samples_returns_empty() -> None:
    audio = render_chaotic(
        system="lorenz",
        rate_hz=5.0,
        amount=0.5,
        symmetry=0.0,
        n_samples=0,
        sample_rate=SAMPLE_RATE,
        seed=0,
    )
    assert audio.shape == (0,)


def test_render_chaotic_invalid_system_raises() -> None:
    with pytest.raises(ValueError, match="chaotic system"):
        render_chaotic(
            system="mandelbrot",
            rate_hz=10.0,
            amount=0.5,
            symmetry=0.0,
            n_samples=128,
            sample_rate=SAMPLE_RATE,
            seed=0,
        )


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        dict(rate_hz=0.0),
        dict(rate_hz=-1.0),
        dict(amount=-0.1),
        dict(amount=1.1),
        dict(symmetry=-1.5),
        dict(symmetry=1.5),
    ],
)
def test_render_chaotic_out_of_range_raises(bad_kwargs: dict) -> None:
    base = dict(
        system="lorenz",
        rate_hz=10.0,
        amount=0.5,
        symmetry=0.0,
        n_samples=128,
        sample_rate=SAMPLE_RATE,
        seed=0,
    )
    base.update(bad_kwargs)
    with pytest.raises(ValueError):
        render_chaotic(**base)


@pytest.mark.parametrize("system", sorted(SUPPORTED_SYSTEMS))
def test_render_chaotic_long_run_stable(system: str) -> None:
    """10+ second render should stay finite."""
    n = 10 * SAMPLE_RATE
    audio = render_chaotic(
        system=system,
        rate_hz=20.0,
        amount=1.0,
        symmetry=0.0,
        n_samples=n,
        sample_rate=SAMPLE_RATE,
        seed=7,
    )
    assert np.all(np.isfinite(audio))
    assert audio.shape == (n,)


@pytest.mark.parametrize("system", sorted(SUPPORTED_SYSTEMS))
def test_render_chaotic_at_rate_hz_ceiling_is_stable(system: str) -> None:
    """At the documented ``RATE_HZ_CEILINGS`` value, the 10 s render stays finite.

    Boundary test: pins the documented ceiling so that later optimization /
    param changes don't silently narrow the usable range.
    """
    rate_hz = RATE_HZ_CEILINGS[system]
    n = 10 * SAMPLE_RATE
    audio = render_chaotic(
        system=system,
        rate_hz=rate_hz,
        amount=1.0,
        symmetry=0.0,
        n_samples=n,
        sample_rate=SAMPLE_RATE,
        seed=3,
    )
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) < 2.0


# ---------------------------------------------------------------------------
# synth_voice integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("system", sorted(SUPPORTED_SYSTEMS))
def test_synth_voice_chaotic_osc_produces_signal(system: str) -> None:
    audio = synth_voice_render(
        freq=220.0,
        duration=0.2,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params={
            "osc_type": "chaotic",
            "osc_chaos_system": system,
            "osc_chaos_rate_hz": 25.0,
            "osc_chaos_amount": 0.6,
        },
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.05


def test_synth_voice_unknown_chaos_system_raises() -> None:
    with pytest.raises(ValueError, match="osc_chaos_system"):
        synth_voice_render(
            freq=220.0,
            duration=0.1,
            amp=0.8,
            sample_rate=SAMPLE_RATE,
            params={
                "osc_type": "chaotic",
                "osc_chaos_system": "henon",
            },
        )


@pytest.mark.parametrize(
    "preset",
    ["lorenz_wobble", "rossler_smear", "duffing_bass", "chua_scatter"],
)
def test_synth_voice_chaotic_preset_renders(preset: str) -> None:
    audio = render_note_signal(
        freq=220.0,
        duration=0.3,
        amp=0.7,
        sample_rate=SAMPLE_RATE,
        params={"engine": "synth_voice", "preset": preset},
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.01


def test_chaos_macros_fill_in_params() -> None:
    """Macros ``chaos_amount`` / ``chaos_rate`` / ``chaos_system`` feed osc_chaos_*."""
    # Exercise the macro dispatch path explicitly.
    from code_musics.engines._synth_macros import resolve_macros

    params = {
        "osc_type": "chaotic",
        "chaos_amount": 0.8,
        "chaos_rate": 0.5,
        "chaos_system": "duffing",
    }
    resolve_macros(params)
    assert params["osc_chaos_amount"] == pytest.approx(0.8)
    # chaos_rate=0.5 -> exp-lerp between 0.5 and 200 Hz -> ~10 Hz.
    assert 5.0 < params["osc_chaos_rate_hz"] < 20.0
    assert params["osc_chaos_system"] == "duffing"
    # Macro keys are popped after resolution.
    assert "chaos_amount" not in params
    assert "chaos_rate" not in params
    assert "chaos_system" not in params


def test_chaos_macros_no_op_when_osc_type_not_chaotic() -> None:
    """Chaos macros should not pollute voices that don't use the chaotic slot."""
    from code_musics.engines._synth_macros import resolve_macros

    params = {
        "osc_type": "polyblep",
        "chaos_amount": 0.9,
        "chaos_rate": 0.5,
        "chaos_system": "chua",
    }
    resolve_macros(params)
    assert "osc_chaos_amount" not in params
    assert "osc_chaos_rate_hz" not in params
    assert "osc_chaos_system" not in params
    # Macro keys are popped regardless.
    assert "chaos_amount" not in params


# ---------------------------------------------------------------------------
# Off-state preservation: absence of osc_type="chaotic" should not change anything
# ---------------------------------------------------------------------------


def test_off_state_polyblep_identical_with_and_without_chaos_keys() -> None:
    """Stray chaos-related keys on non-chaotic voices do not change rendering."""
    base_params = {
        "osc_type": "polyblep",
        "osc_wave": "saw",
        "filter_mode": "lowpass",
        "filter_cutoff_hz": 2000.0,
    }
    baseline = synth_voice_render(
        freq=220.0,
        duration=0.05,
        amp=0.6,
        sample_rate=SAMPLE_RATE,
        params=dict(base_params),
    )
    # Same render but with chaos macros present — they should be stripped and
    # not influence the polyblep path.
    with_macros = synth_voice_render(
        freq=220.0,
        duration=0.05,
        amp=0.6,
        sample_rate=SAMPLE_RATE,
        params={**base_params, "chaos_amount": 0.9, "chaos_system": "lorenz"},
    )
    assert np.allclose(baseline, with_macros)


# ---------------------------------------------------------------------------
# ChaoticSource — modulation matrix integration
# ---------------------------------------------------------------------------


def test_chaotic_source_samples_finite_on_grid() -> None:
    from code_musics.modulation import SourceSamplingContext

    times = np.linspace(0.0, 2.0, 1024)
    ctx = SourceSamplingContext(sample_rate=SAMPLE_RATE, total_dur=2.0)
    source = ChaoticSource(
        system="lorenz", rate_hz=4.0, amount=0.7, symmetry=0.0, seed=5
    )
    curve = source.sample(times, ctx)
    assert curve.shape == times.shape
    assert np.all(np.isfinite(curve))
    assert np.max(np.abs(curve)) > 0.01


@pytest.mark.parametrize("system", sorted(SUPPORTED_SYSTEMS))
def test_chaotic_source_all_systems(system: str) -> None:
    from code_musics.modulation import SourceSamplingContext

    times = np.linspace(0.0, 1.0, 512)
    ctx = SourceSamplingContext(sample_rate=SAMPLE_RATE, total_dur=1.0)
    source = ChaoticSource(system=system, rate_hz=6.0, amount=0.5, seed=11)
    curve = source.sample(times, ctx)
    assert np.all(np.isfinite(curve))


def test_chaotic_source_invalid_system_raises() -> None:
    with pytest.raises(ValueError, match="system"):
        ChaoticSource(system="mandelbrot")  # type: ignore[arg-type]


def test_chaotic_source_invalid_rate_raises() -> None:
    with pytest.raises(ValueError, match="rate_hz"):
        ChaoticSource(rate_hz=0.0)


def test_chaotic_source_invalid_amount_raises() -> None:
    with pytest.raises(ValueError, match="amount"):
        ChaoticSource(amount=-0.1)


def test_chaotic_source_empty_times_returns_empty() -> None:
    from code_musics.modulation import SourceSamplingContext

    ctx = SourceSamplingContext(sample_rate=SAMPLE_RATE, total_dur=0.0)
    source = ChaoticSource()
    out = source.sample(np.zeros(0, dtype=np.float64), ctx)
    assert out.shape == (0,)


def test_chaotic_source_drives_cutoff_in_full_render() -> None:
    """End-to-end: ChaoticSource -> cutoff_hz through a Score render."""
    score = Score(f0_hz=220.0, sample_rate=SAMPLE_RATE)
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "saw",
            "cutoff_hz": 1200.0,
            "resonance_q": 0.9,
        },
        modulations=[
            ModConnection(
                source=ChaoticSource(
                    system="rossler",
                    rate_hz=3.0,
                    amount=0.6,
                    symmetry=0.0,
                    seed=31,
                ),
                target=AutomationTarget(kind="synth", name="cutoff_hz"),
                amount=900.0,
                bipolar=True,
                mode="add",
                name="chaos_cutoff",
            ),
        ],
    )
    phrase = Phrase(
        events=(
            NoteEvent(start=0.0, duration=0.4, partial=1.0, amp=0.6),
            NoteEvent(start=0.4, duration=0.4, partial=1.5, amp=0.6),
        )
    )
    score.add_phrase(voice_name="lead", phrase=phrase, start=0.0)
    audio = score.render()
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) > 0.001
