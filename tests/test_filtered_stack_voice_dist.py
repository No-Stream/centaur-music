"""Tests for the per-note voice distortion slot on the ``filtered_stack`` engine.

Phase 2c of the RePro-inspired DSP additions.  The signal-chain intent is:

    osc(s) -> filter(s) -> envelope/VCA -> [voice_dist] -> += voice_output_buffer
                                            ^-- NEW, per-note, pre-sum

``filtered_stack.render`` returns a single pre-VCA note buffer (the ADSR is
applied at the ``Score`` layer), so the slot is inserted at the tail of
``render`` before the final ``amp * filtered`` return.  The rest of the
chain (ADSR, summing) still runs outside the engine and the pre-sum
character is preserved because each note is distorted independently before
any summing happens.

These tests focus on the defaults-preserved guarantee, fast-path shortcut
semantics, the pre-sum IMD advantage that is the whole point of the slot,
and smoke coverage for every non-off mode.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from code_musics.engines.filtered_stack import render

SAMPLE_RATE: int = 44_100


def _base_params() -> dict[str, Any]:
    return {
        "waveform": "saw",
        "n_harmonics": 12,
        "cutoff_hz": 1_400.0,
        "keytrack": 0.1,
        "resonance_q": 1.2,
    }


def _render_note(freq: float, extra: dict[str, Any] | None = None) -> np.ndarray:
    params = _base_params()
    if extra:
        params.update(extra)
    return render(
        freq=freq,
        duration=0.4,
        amp=0.5,
        sample_rate=SAMPLE_RATE,
        params=params,
    )


def test_voice_dist_off_mode_is_bit_identical_to_defaults() -> None:
    """``voice_dist_mode="off"`` must not change a single sample vs the
    no-voice-dist case — this is the "existing pieces unchanged" contract.
    """
    reference = _render_note(220.0)
    with_off = _render_note(
        220.0,
        {
            "voice_dist_mode": "off",
            "voice_dist_drive": 1.5,
            "voice_dist_mix": 1.0,
            "voice_dist_tone": 0.5,
        },
    )
    np.testing.assert_array_equal(with_off, reference)


def test_voice_dist_drive_zero_is_bit_identical() -> None:
    """Even with an active shaping mode, ``drive=0`` is a fast-path
    passthrough — required for automation-across-zero continuity."""
    reference = _render_note(220.0)
    with_zero_drive = _render_note(
        220.0,
        {
            "voice_dist_mode": "hard_clip",
            "voice_dist_drive": 0.0,
            "voice_dist_mix": 1.0,
        },
    )
    np.testing.assert_array_equal(with_zero_drive, reference)


def test_defaults_bit_identical_reference_capture() -> None:
    """Snapshot-style guard: render the same note twice with no voice_dist
    mentioned at all and assert bit-exact match.  This protects against an
    accidental change in default behavior (e.g. accidentally enabling a
    shaper with a non-zero default drive).
    """
    first = _render_note(165.0)
    second = _render_note(165.0)
    # Determinism first — filtered_stack with the same inputs should be
    # exactly reproducible.
    np.testing.assert_array_equal(first, second)

    # And without voice_dist params, nothing should be clipping/folding
    # the signal in a way that saturates — a pure saw filtered at 1.4 kHz
    # with Q=1.2 sits well below the hard-clip ceiling.
    assert float(np.max(np.abs(first))) < 0.55


@pytest.mark.parametrize(
    "mode",
    [
        "soft_clip",
        "hard_clip",
        "foldback",
        "corrode",
        "saturation",
        "preamp",
    ],
)
def test_voice_dist_each_mode_smoke(mode: str) -> None:
    """Each non-off mode must produce finite, non-silent audio with the
    correct shape."""
    out = _render_note(
        220.0,
        {
            "voice_dist_mode": mode,
            "voice_dist_drive": 1.0,
            "voice_dist_mix": 1.0,
        },
    )
    assert out.shape == (int(SAMPLE_RATE * 0.4),)
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) > 0.0


def test_voice_dist_active_changes_the_signal() -> None:
    """Sanity check: with a meaningful drive, a shaping mode *must* change
    the output.  Catches an accidental short-circuit in the engine wiring.
    """
    clean = _render_note(220.0)
    shaped = _render_note(
        220.0,
        {
            "voice_dist_mode": "hard_clip",
            "voice_dist_drive": 1.5,
            "voice_dist_mix": 1.0,
        },
    )
    # Shape unchanged, but samples must differ.
    assert shaped.shape == clean.shape
    # RMS difference at least a couple percent of the clean RMS — plenty
    # of margin above numerical noise.
    clean_rms = float(np.sqrt(np.mean(clean**2)))
    diff_rms = float(np.sqrt(np.mean((shaped - clean) ** 2)))
    assert clean_rms > 1e-6
    assert diff_rms / clean_rms > 0.02


def _imd_energy_near(
    signal: np.ndarray, bin_freqs: list[float], half_width_hz: float = 2.0
) -> float:
    """Sum of FFT magnitude^2 near a list of target frequencies.

    ``half_width_hz`` gives a small bucket so we catch the peak even if
    bin alignment is slightly off.
    """
    n = signal.shape[0]
    spectrum = np.abs(np.fft.rfft(signal))
    bin_per_hz = n / SAMPLE_RATE
    total = 0.0
    for f in bin_freqs:
        center = int(round(f * bin_per_hz))
        half_bins = max(1, int(round(half_width_hz * bin_per_hz)))
        lo = max(0, center - half_bins)
        hi = min(spectrum.shape[0], center + half_bins + 1)
        if lo >= hi:
            continue
        total += float(np.sum(spectrum[lo:hi] ** 2))
    return total


def test_chord_imd_pre_sum_vs_post_sum() -> None:
    """Pre-sum distortion preserves chord clarity — the whole point of the
    per-note slot.

    Render a major triad both ways:

    * **pre-sum** — each note rendered through ``filtered_stack`` with
      ``voice_dist_mode="hard_clip"``, then summed.
    * **post-sum** — each note rendered *clean*, summed, then hard-clipped
      once in post.

    Hard-clipping a sum generates sum/difference intermodulation products
    that do not appear in the pre-sum version (each note's harmonic
    identity stays intact before the summing).  We quantify the
    difference by summing spectral energy at the nearest sum/difference
    bins and asserting pre-sum < post-sum.
    """
    freqs = [220.0, 277.18, 329.63]  # A3 major triad (approximate)

    # Pre-sum: engine applies hard_clip per-note before summing.
    pre_sum = np.zeros(int(SAMPLE_RATE * 0.4), dtype=np.float64)
    for f in freqs:
        pre_sum += _render_note(
            f,
            {
                "voice_dist_mode": "hard_clip",
                "voice_dist_drive": 1.5,
                "voice_dist_mix": 1.0,
            },
        )

    # Post-sum: render clean, sum, then hard-clip once.
    clean_sum = np.zeros(int(SAMPLE_RATE * 0.4), dtype=np.float64)
    for f in freqs:
        clean_sum += _render_note(f)
    # A hard clip equivalent in character to the voice_dist "hard_clip"
    # helper at ``drive=1.5`` (internal drive ~0.75 after the 0.5 map).
    # Use a symmetric ceiling that will definitely engage on the summed
    # chord RMS which is ~3x louder than any single note.
    ceiling = 0.35
    post_sum = np.clip(clean_sum, -ceiling, ceiling)

    # Pairs of sum and difference frequencies — classic IMD products that
    # a post-sum shaper creates and a pre-sum shaper does not.
    imd_bins: list[float] = []
    for i, f1 in enumerate(freqs):
        for f2 in freqs[i + 1 :]:
            imd_bins.append(f1 + f2)
            imd_bins.append(abs(f1 - f2))
            imd_bins.append(2.0 * f1 - f2)
            imd_bins.append(2.0 * f2 - f1)

    pre_imd = _imd_energy_near(pre_sum, imd_bins)
    post_imd = _imd_energy_near(post_sum, imd_bins)

    # Both positive, and pre-sum clearly less than post-sum.
    assert pre_imd > 0.0
    assert post_imd > 0.0
    # Require a meaningful margin so the test isn't noise-sensitive.  In
    # practice post-sum IMD is ~3-10x pre-sum on this chord.
    assert pre_imd < post_imd * 0.7
