"""Tests for sustained ``drum_voice`` exciter types (bow / blow / rub) and
the ``contact_nonlinearity`` shaping of transient exciters.
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._drum_layers import render_exciter
from code_musics.engines._sustained_exciters import (
    render_blow,
    render_bow,
    render_rub,
)
from code_musics.engines.drum_voice import render
from code_musics.engines.registry import resolve_synth_params

SAMPLE_RATE = 44_100


# ---------------------------------------------------------------------------
# 1. Primitive renderers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("duration", [0.5, 2.0, 5.0])
def test_bow_primitive_renders_finite_correct_length(duration: float) -> None:
    n_samples = int(SAMPLE_RATE * duration)
    audio = render_bow(
        freq=220.0,
        duration=duration,
        sample_rate=SAMPLE_RATE,
        pressure=0.6,
        speed=0.5,
        position=0.3,
        noise_amount=0.2,
        seed=42,
    )
    assert audio.shape == (n_samples,)
    assert audio.dtype == np.float64
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.0


@pytest.mark.parametrize("duration", [0.5, 2.0, 5.0])
def test_blow_primitive_renders_finite_correct_length(duration: float) -> None:
    n_samples = int(SAMPLE_RATE * duration)
    audio = render_blow(
        freq=220.0,
        duration=duration,
        sample_rate=SAMPLE_RATE,
        pressure=0.7,
        embouchure=0.4,
        breath_noise=0.3,
        wobble_rate_hz=4.5,
        wobble_depth=0.15,
        seed=42,
    )
    assert audio.shape == (n_samples,)
    assert audio.dtype == np.float64
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.0


@pytest.mark.parametrize("duration", [0.5, 2.0, 5.0])
def test_rub_primitive_renders_finite_correct_length(duration: float) -> None:
    n_samples = int(SAMPLE_RATE * duration)
    audio = render_rub(
        freq=220.0,
        duration=duration,
        sample_rate=SAMPLE_RATE,
        pressure=0.5,
        speed=0.3,
        roughness=0.5,
        stiction=0.2,
        seed=42,
    )
    assert audio.shape == (n_samples,)
    assert audio.dtype == np.float64
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.0


def test_bow_deterministic_from_seed() -> None:
    kwargs: dict = dict(
        freq=220.0,
        duration=0.75,
        sample_rate=SAMPLE_RATE,
        pressure=0.5,
        speed=0.5,
        position=0.3,
        noise_amount=0.25,
        seed=99,
    )
    first = render_bow(**kwargs)
    second = render_bow(**kwargs)
    assert np.array_equal(first, second)


def test_blow_deterministic_from_seed() -> None:
    kwargs: dict = dict(
        freq=220.0,
        duration=0.75,
        sample_rate=SAMPLE_RATE,
        pressure=0.6,
        embouchure=0.4,
        breath_noise=0.25,
        wobble_rate_hz=4.0,
        wobble_depth=0.1,
        seed=17,
    )
    first = render_blow(**kwargs)
    second = render_blow(**kwargs)
    assert np.array_equal(first, second)


def test_rub_deterministic_from_seed() -> None:
    kwargs: dict = dict(
        freq=220.0,
        duration=0.75,
        sample_rate=SAMPLE_RATE,
        pressure=0.5,
        speed=0.3,
        roughness=0.5,
        stiction=0.3,
        seed=7,
    )
    first = render_rub(**kwargs)
    second = render_rub(**kwargs)
    assert np.array_equal(first, second)


def test_different_seeds_give_different_output() -> None:
    base: dict = dict(
        freq=220.0,
        duration=0.5,
        sample_rate=SAMPLE_RATE,
        pressure=0.5,
        speed=0.5,
        position=0.3,
        noise_amount=0.25,
    )
    a = render_bow(**base, seed=1)
    b = render_bow(**base, seed=2)
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# 2. drum_voice integration
# ---------------------------------------------------------------------------


def _render_through_drum_voice(
    exciter_type: str, extra_params: dict | None = None
) -> np.ndarray:
    params: dict = {
        "exciter_type": exciter_type,
        "exciter_level": 0.4,
        "tone_type": "modal",
        "tone_level": 1.0,
        "tone_sweep_ratio": 1.0,
        "tone_sweep_decay_s": 0.01,
        "modal_mode_table": "bar_metal",
        "modal_n_modes": 4,
        "modal_decay_s": 0.8,
        "modal_coupling": 0.15,
        "modal_dispersion": 0.2,
        "noise_level": 0.0,
        "metallic_level": 0.0,
    }
    if extra_params:
        params.update(extra_params)
    return render(
        freq=180.0,
        duration=0.6,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params=params,
    )


def _assert_late_energy_sustained(audio: np.ndarray) -> None:
    """Sustained exciters must keep late-window RMS within 0.25x early-window RMS.

    Matches the threshold used by ``test_sustained_exciter_holds_energy_throughout_note``
    — a regression to a transient click would produce a ratio near zero.
    """
    n = len(audio)
    early = audio[: n // 4]
    late = audio[3 * n // 4 :]
    early_rms = float(np.sqrt(np.mean(early**2)))
    late_rms = float(np.sqrt(np.mean(late**2)))
    assert late_rms >= 0.25 * early_rms, (
        f"late RMS {late_rms:.5f} < 0.25 * early RMS {early_rms:.5f}"
    )


def test_drum_voice_bow_renders_finite() -> None:
    audio = _render_through_drum_voice("bow")
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.0
    _assert_late_energy_sustained(audio)


def test_drum_voice_blow_renders_finite() -> None:
    audio = _render_through_drum_voice("blow")
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.0
    _assert_late_energy_sustained(audio)


def test_drum_voice_rub_renders_finite() -> None:
    audio = _render_through_drum_voice("rub")
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.0
    _assert_late_energy_sustained(audio)


def test_sustained_exciter_holds_energy_throughout_note() -> None:
    """Unlike transient exciters, a sustained exciter must deliver energy
    across the full note duration — not just at the attack."""
    audio = _render_through_drum_voice("bow")
    n = len(audio)
    early = audio[: n // 4]
    late = audio[3 * n // 4 :]
    early_rms = float(np.sqrt(np.mean(early**2)))
    late_rms = float(np.sqrt(np.mean(late**2)))
    # Late-third RMS should be at least 25% of the early-third RMS.  For a
    # transient kick the ratio would be essentially zero; sustained bow
    # should stay comparable.
    assert late_rms > 0.25 * early_rms


# ---------------------------------------------------------------------------
# 3. Presets
# ---------------------------------------------------------------------------


_SUSTAINED_PRESETS = [
    "bow_gentle",
    "bow_taut",
    "bow_aggressive",
    "blow_breath_pad",
    "blow_reed_sing",
    "blow_overblow",
    "rub_glass",
    "rub_skin",
    "rub_squeal",
]


@pytest.mark.parametrize("preset_name", _SUSTAINED_PRESETS)
def test_sustained_preset_renders(preset_name: str) -> None:
    resolved = resolve_synth_params({"engine": "drum_voice", "preset": preset_name})
    resolved.pop("engine", None)
    audio = render(
        freq=200.0,
        duration=0.8,
        amp=0.7,
        sample_rate=SAMPLE_RATE,
        params=resolved,
    )
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.0


# ---------------------------------------------------------------------------
# 4. contact_nonlinearity preservation
# ---------------------------------------------------------------------------


_TRANSIENT_EXCITERS: list[tuple[str, dict]] = [
    ("click", {"exciter_center_hz": 3000.0}),
    ("impulse", {}),
    ("multi_tap", {}),
    ("fm_burst", {"exciter_fm_ratio": 1.5, "exciter_fm_index": 3.0}),
    ("noise_burst", {}),
]


@pytest.mark.parametrize("exciter_type,extra_params", _TRANSIENT_EXCITERS)
def test_contact_nonlinearity_zero_preserves_output(
    exciter_type: str, extra_params: dict
) -> None:
    """contact_nonlinearity=0 (default) must leave transient exciters unchanged."""
    params_base = {**extra_params}
    params_with_zero = {**extra_params, "contact_nonlinearity": 0.0}

    def _run(params: dict) -> np.ndarray:
        rng = np.random.default_rng(1234)
        return render_exciter(
            n_samples=4410,
            freq=200.0,
            sample_rate=SAMPLE_RATE,
            rng=rng,
            exciter_type=exciter_type,
            params=params,
            velocity=0.5,
            duration=0.1,
        )

    base = _run(params_base)
    with_zero = _run(params_with_zero)
    assert np.array_equal(base, with_zero)


@pytest.mark.parametrize("exciter_type,extra_params", _TRANSIENT_EXCITERS)
def test_contact_nonlinearity_positive_changes_output(
    exciter_type: str, extra_params: dict
) -> None:
    """contact_nonlinearity>0 alters the transient exciter output."""
    params_base = {**extra_params}
    params_with_nl = {**extra_params, "contact_nonlinearity": 0.8}

    def _run(params: dict) -> np.ndarray:
        rng = np.random.default_rng(4321)
        return render_exciter(
            n_samples=4410,
            freq=200.0,
            sample_rate=SAMPLE_RATE,
            rng=rng,
            exciter_type=exciter_type,
            params=params,
            velocity=0.4,  # below 1.0 so alpha != 1
            duration=0.1,
        )

    base = _run(params_base)
    with_nl = _run(params_with_nl)
    assert np.isfinite(with_nl).all()
    # For the trivial 1-sample impulse the nonlinearity is a no-op (1**alpha == 1);
    # for everything else output must change.
    if exciter_type == "impulse":
        assert np.array_equal(base, with_nl)
    else:
        assert not np.array_equal(base, with_nl)


# ---------------------------------------------------------------------------
# 5. Pitch pairing: sustained exciter + modal bank carries fundamental energy
# ---------------------------------------------------------------------------


def test_bow_into_modal_bank_shows_fundamental_energy() -> None:
    """Pair a bow with a modal bank tuned to the note freq and confirm the
    output has non-trivial spectral energy near the fundamental.
    """
    freq = 220.0
    duration = 1.2
    params: dict = {
        "exciter_type": "bow",
        "exciter_level": 0.5,
        "exciter_bow_pressure": 0.65,
        "exciter_bow_speed": 0.5,
        "exciter_bow_position": 0.3,
        "exciter_bow_noise_amount": 0.2,
        "tone_type": "modal",
        "tone_level": 1.0,
        "tone_sweep_ratio": 1.0,
        "tone_sweep_decay_s": 0.01,
        "modal_ratios": [1.0, 2.0, 3.0, 4.0],
        "modal_amps": [1.0, 0.7, 0.5, 0.3],
        "modal_decays_s": [1.5, 1.3, 1.1, 0.9],
        "modal_coupling": 0.2,
        "modal_dispersion": 0.15,
        "noise_level": 0.0,
        "metallic_level": 0.0,
    }
    audio = render(
        freq=freq,
        duration=duration,
        amp=0.8,
        sample_rate=SAMPLE_RATE,
        params=params,
    )
    assert np.isfinite(audio).all()

    # Window the middle of the note (skip attack/release fades).
    n = audio.shape[0]
    window = audio[n // 4 : 3 * n // 4]
    spectrum = np.abs(np.fft.rfft(window))
    freqs = np.fft.rfftfreq(window.shape[0], d=1.0 / SAMPLE_RATE)

    # Find the bin closest to the fundamental.
    fund_idx = int(np.argmin(np.abs(freqs - freq)))

    # Compare fundamental-band energy (+/- 10 Hz) against the mean spectrum.
    # The bow excitation is broadband, so we don't expect huge peak-to-mean
    # ratios — just clear evidence that the modal bank is selectively
    # resonating at the fundamental rather than leaving the spectrum flat.
    bin_spacing = float(freqs[1] - freqs[0])
    half_bins = max(1, int(round(10.0 / bin_spacing)))
    lo = max(0, fund_idx - half_bins)
    hi = min(spectrum.shape[0], fund_idx + half_bins + 1)
    fundamental_energy = float(np.max(spectrum[lo:hi]))
    mean_energy = float(np.mean(spectrum))

    assert fundamental_energy > 2.0 * mean_energy, (
        f"Fundamental peak ({fundamental_energy:.3f}) is not sufficiently "
        f"stronger than the mean spectrum ({mean_energy:.3f})."
    )
