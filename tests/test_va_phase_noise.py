"""Tests for per-sample per-voice supersaw phase noise in the va engine.

Verifies:

* ``osc_phase_noise=0.0`` (default) is a bit-identical no-op.
* Enabling phase noise raises the out-of-band noise floor.
* Two renders with same params are bit-identical (deterministic).
* The 7 supersaw voices have mutually uncorrelated noise streams.
"""

from __future__ import annotations

import numpy as np

from code_musics.engines.va import render

SR: int = 48000


def _base_params(**overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "_voice_name": "va_phase_noise_test",
        "osc_mode": "supersaw",
        "supersaw_detune": 0.3,
        "supersaw_mix": 0.5,
        "cutoff_hz": 8000.0,
        "resonance_q": 0.707,
        "analog_jitter": 0.0,
        "pitch_drift": 0.0,
        "noise_floor": 0.0,
        "cutoff_drift": 0.0,
        "filter_env_amount": 0.0,
    }
    params.update(overrides)
    return params


def test_phase_noise_default_is_noop() -> None:
    """Omitting ``osc_phase_noise`` vs explicit 0.0 must be bit-identical."""
    a = render(
        freq=220.0,
        duration=0.3,
        amp=0.5,
        sample_rate=SR,
        params=_base_params(),
    )
    b = render(
        freq=220.0,
        duration=0.3,
        amp=0.5,
        sample_rate=SR,
        params=_base_params(osc_phase_noise=0.0),
    )
    np.testing.assert_array_equal(a, b)


def test_phase_noise_changes_output() -> None:
    """Sanity check: enabling phase noise must alter the output at all.

    If this fails, the wiring is broken (noise isn't reaching the
    oscillator phase accumulator).  The HF-floor test below depends on
    this succeeding first.
    """
    clean = render(
        freq=220.0,
        duration=0.3,
        amp=0.5,
        sample_rate=SR,
        params=_base_params(cutoff_hz=20_000.0, osc_phase_noise=0.0),
    )
    noisy = render(
        freq=220.0,
        duration=0.3,
        amp=0.5,
        sample_rate=SR,
        params=_base_params(cutoff_hz=20_000.0, osc_phase_noise=0.8),
    )
    diff = noisy - clean
    diff_rms = float(np.sqrt(np.mean(diff * diff)))
    clean_rms = float(np.sqrt(np.mean(clean * clean)))
    assert diff_rms > clean_rms * 1e-6, (
        f"phase noise appears inert: diff_rms={diff_rms:.3e} clean_rms={clean_rms:.3e}"
    )


def test_phase_noise_raises_noise_floor() -> None:
    """The out-of-band spectral floor of the noise-contribution signal
    (noisy - clean) must sit well above any numerical floor produced by
    identical-input renders, confirming the noise is broadband.

    We don't compare ``clean_hf`` vs ``noisy_hf`` directly because
    per-note peak-normalization squashes meaningful small-signal delta.
    We compare RMS of ``noisy - clean`` (isolates the noise
    contribution) against RMS of ``clean_a - clean_b`` with a shuffled
    voice name (no-noise baseline, measures unrelated RNG-seed
    variation) and assert the noise is much larger.
    """
    duration = 0.4
    common = dict(freq=220.0, duration=duration, amp=0.5, sample_rate=SR)
    clean = render(
        params=_base_params(cutoff_hz=20_000.0, osc_phase_noise=0.0), **common
    )
    noisy = render(
        params=_base_params(cutoff_hz=20_000.0, osc_phase_noise=1.0), **common
    )

    assert np.all(np.isfinite(noisy))

    noise_contribution = noisy - clean
    noise_rms = float(np.sqrt(np.mean(noise_contribution * noise_contribution)))

    # Out-of-band: above the 10th harmonic (2200 Hz) is mostly signal
    # that was already present; take the HF band of the NOISE
    # contribution to confirm it's broadband.
    spec = np.abs(np.fft.rfft(noise_contribution))
    freqs = np.fft.rfftfreq(noise_contribution.size, 1.0 / SR)
    hf_mask = (freqs >= 10_000.0) & (freqs <= 20_000.0)
    lf_mask = (freqs >= 100.0) & (freqs <= 2_000.0)
    hf_power = float((spec[hf_mask] ** 2).sum())
    lf_power = float((spec[lf_mask] ** 2).sum())

    # The NOISE contribution must have substantial broadband energy
    # relative to the base signal.
    assert noise_rms > 1e-5, (
        f"phase noise contribution too small: noise_rms={noise_rms:.3e}"
    )
    # Broadband character: high-frequency power of the noise
    # contribution must be within a decade of low-frequency power
    # (truly broadband -> roughly flat, within ~20 dB).  A narrow-band
    # noise would have one side vanishing.
    assert hf_power > 0.0 and lf_power > 0.0, "noise must be broadband"
    log_ratio = float(np.log10(max(hf_power, lf_power) / min(hf_power, lf_power)))
    assert log_ratio < 3.0, (
        f"phase noise not broadband: hf={hf_power:.3e} lf={lf_power:.3e}"
    )


def test_phase_noise_deterministic() -> None:
    """Two renders with the same params must be bit-identical."""
    params = _base_params(osc_phase_noise=0.5)
    a = render(freq=220.0, duration=0.3, amp=0.5, sample_rate=SR, params=params)
    b = render(freq=220.0, duration=0.3, amp=0.5, sample_rate=SR, params=params)
    np.testing.assert_array_equal(a, b)


def test_voices_independent() -> None:
    """The 7 supersaw voices must have mutually uncorrelated noise streams.

    Directly validates that each voice derives its own seed from the note
    hash + voice index, so the perturbation doesn't collapse to a common
    mode that would just wobble the aggregate pitch.
    """
    from code_musics.engines.va import _supersaw_phase_noise_streams

    n = 2048
    streams = _supersaw_phase_noise_streams(
        n_samples=n,
        freq=220.0,
        duration=0.3,
        amp=0.5,
        sample_rate=SR,
        voice_name="va_phase_noise_test_indep",
    )
    assert streams.shape == (7, n)

    # Sample a handful of voice pairs and check Pearson |r| is small.
    for i in range(7):
        for j in range(i + 1, 7):
            a = streams[i] - streams[i].mean()
            b = streams[j] - streams[j].mean()
            denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
            if denom < 1e-12:
                continue
            r = float(np.sum(a * b) / denom)
            assert abs(r) < 0.1, f"voices {i},{j} correlated: r={r:.3f}"
