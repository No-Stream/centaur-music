"""Piano engine tests."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines.piano_additive import render
from code_musics.engines.registry import render_note_signal

SAMPLE_RATE = 44_100
DURATION = 0.3
FREQ = 220.0
AMP = 0.7


def _render_default(**overrides: object) -> np.ndarray:
    kwargs: dict = {
        "freq": FREQ,
        "duration": DURATION,
        "amp": AMP,
        "sample_rate": SAMPLE_RATE,
        "params": {},
    }
    kwargs.update(overrides)
    return render(**kwargs)


def test_piano_render_basic() -> None:
    signal = _render_default()

    n_expected = int(SAMPLE_RATE * DURATION)
    assert signal.shape == (n_expected,)
    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(signal)) > 0.0


def test_piano_render_deterministic() -> None:
    params = {
        "n_partials": 24,
        "inharmonicity": 0.0003,
        "hammer_hardness": 0.5,
        "brightness": 0.5,
        "drift": 0.1,
    }
    first = _render_default(params=params)
    second = _render_default(params=params)

    assert np.allclose(first, second)


def test_inharmonicity_stretches_partials() -> None:
    harmonic = _render_default(
        params={"inharmonicity": 0.0, "hammer_noise": 0.0, "drift": 0.0}
    )
    inharmonic = _render_default(
        params={"inharmonicity": 0.001, "hammer_noise": 0.0, "drift": 0.0}
    )

    assert not np.allclose(harmonic, inharmonic)


def test_hammer_hardness_affects_brightness() -> None:
    soft = _render_default(
        params={"hammer_hardness": 0.1, "drift": 0.0, "hammer_noise": 0.5}
    )
    hard = _render_default(
        params={"hammer_hardness": 0.9, "drift": 0.0, "hammer_noise": 0.5}
    )

    assert not np.allclose(soft, hard)


def test_hammer_adds_onset_energy() -> None:
    no_hammer = _render_default(params={"hammer_noise": 0.0, "drift": 0.0})
    with_hammer = _render_default(params={"hammer_noise": 0.5, "drift": 0.0})

    diff = np.abs(with_hammer - no_hammer)
    onset_samples = int(0.035 * SAMPLE_RATE)
    onset_diff = float(np.mean(diff[:onset_samples]))
    tail_diff = float(np.mean(diff[onset_samples:]))

    assert onset_diff > tail_diff


def test_unison_changes_signal() -> None:
    mono = _render_default(
        params={"unison_count": 1, "drift": 0.0, "hammer_noise": 0.0}
    )
    triple = _render_default(
        params={
            "unison_count": 3,
            "unison_detune": 3.0,
            "drift": 0.0,
            "hammer_noise": 0.0,
        }
    )

    assert not np.allclose(mono, triple)


def test_soundboard_changes_signal() -> None:
    no_board = _render_default(
        params={"soundboard_color": 0.0, "drift": 0.0, "hammer_noise": 0.0}
    )
    with_board = _render_default(
        params={"soundboard_color": 0.8, "drift": 0.0, "hammer_noise": 0.0}
    )

    assert not np.allclose(no_board, with_board)


def test_custom_partial_ratios() -> None:
    septimal_ratios = [
        {"ratio": 1.0, "amp": 1.0},
        {"ratio": 7 / 4, "amp": 0.6},
        {"ratio": 3 / 2, "amp": 0.5},
    ]
    custom = _render_default(
        params={"partial_ratios": septimal_ratios, "drift": 0.0, "hammer_noise": 0.0}
    )
    default = _render_default(params={"drift": 0.0, "hammer_noise": 0.0})

    assert np.all(np.isfinite(custom))
    assert np.max(np.abs(custom)) > 0.0
    assert not np.allclose(custom, default)


def test_freq_trajectory_support() -> None:
    n_samples = int(SAMPLE_RATE * DURATION)
    sweep = np.linspace(FREQ, FREQ * 1.5, n_samples)

    with_sweep = render(
        freq=FREQ,
        duration=DURATION,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={"drift": 0.0, "hammer_noise": 0.0},
        freq_trajectory=sweep,
    )
    static = _render_default(params={"drift": 0.0, "hammer_noise": 0.0})

    assert np.all(np.isfinite(with_sweep))
    assert not np.allclose(with_sweep, static)


def test_decay_tail_quieter_than_onset() -> None:
    signal = render(
        freq=FREQ,
        duration=2.0,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={"decay_base": 2.0},
    )

    n_total = signal.size
    q1_rms = float(np.sqrt(np.mean(signal[: n_total // 4] ** 2)))
    q4_rms = float(np.sqrt(np.mean(signal[3 * n_total // 4 :] ** 2)))

    assert q4_rms < q1_rms


def test_damper_thump_on_short_notes() -> None:
    no_damper = render(
        freq=FREQ,
        duration=0.15,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={"damper_noise": 0.0, "drift": 0.0},
    )
    with_damper = render(
        freq=FREQ,
        duration=0.15,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={"damper_noise": 0.3, "drift": 0.0},
    )

    tail_start = int(0.7 * no_damper.size)
    tail_diff = float(
        np.mean(np.abs(with_damper[tail_start:] - no_damper[tail_start:]))
    )

    assert tail_diff > 0.0


def test_negative_freq_raises() -> None:
    with pytest.raises(ValueError, match="freq must be positive"):
        render(
            freq=-1.0,
            duration=DURATION,
            amp=AMP,
            sample_rate=SAMPLE_RATE,
            params={},
        )


def test_invalid_inharmonicity_raises() -> None:
    with pytest.raises(ValueError, match="inharmonicity must be non-negative"):
        _render_default(params={"inharmonicity": -0.01})


def test_zero_duration_returns_empty() -> None:
    signal = render(
        freq=FREQ,
        duration=0.0,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={},
    )
    assert signal.shape == (0,)


def test_piano_via_registry() -> None:
    signal = render_note_signal(
        freq=FREQ,
        duration=DURATION,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={"engine": "piano_additive"},
    )

    n_expected = int(SAMPLE_RATE * DURATION)
    assert signal.shape == (n_expected,)
    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(signal)) > 0.0


def test_piano_preset_grand() -> None:
    signal = render_note_signal(
        freq=FREQ,
        duration=DURATION,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={"engine": "piano_additive", "preset": "grand"},
    )

    n_expected = int(SAMPLE_RATE * DURATION)
    assert signal.shape == (n_expected,)
    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(signal)) > 0.0


# --- New tests for piano realism overhaul ---


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal**2)))


def test_prompt_decay_drops_faster_than_no_prompt() -> None:
    """High decay_prompt should cause faster energy drop in the 50-300ms window."""
    base_params = {"drift": 0.0, "hammer_noise": 0.0, "unison_count": 1}

    with_prompt = render(
        freq=FREQ,
        duration=2.0,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={**base_params, "decay_prompt": 0.8},
    )
    no_prompt = render(
        freq=FREQ,
        duration=2.0,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={**base_params, "decay_prompt": 0.0},
    )

    start = int(0.05 * SAMPLE_RATE)
    end = int(0.3 * SAMPLE_RATE)
    rms_prompt = _rms(with_prompt[start:end])
    rms_no_prompt = _rms(no_prompt[start:end])

    late_start = int(1.0 * SAMPLE_RATE)
    late_end = int(1.5 * SAMPLE_RATE)
    late_rms_prompt = _rms(with_prompt[late_start:late_end])
    late_rms_no_prompt = _rms(no_prompt[late_start:late_end])

    # With prompt, the ratio of early-to-late energy should be higher
    # (energy drops faster from the prompt phase).
    ratio_prompt = rms_prompt / max(1e-12, late_rms_prompt)
    ratio_no_prompt = rms_no_prompt / max(1e-12, late_rms_no_prompt)
    assert ratio_prompt > ratio_no_prompt


def test_attack_contact_time_varies_with_hardness() -> None:
    """Soft hammer should have a slower attack rise than hard hammer."""
    base_params = {"drift": 0.0, "hammer_noise": 0.0, "unison_count": 1}

    soft = render(
        freq=FREQ,
        duration=0.5,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={**base_params, "hammer_hardness": 0.1},
    )
    hard = render(
        freq=FREQ,
        duration=0.5,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={**base_params, "hammer_hardness": 0.9},
    )

    # Measure time to reach 50% of peak via envelope follower.
    def time_to_half_peak(sig: np.ndarray) -> int:
        env = np.abs(sig)
        half = np.max(env) * 0.5
        indices = np.where(env >= half)[0]
        return int(indices[0]) if indices.size > 0 else sig.size

    soft_rise = time_to_half_peak(soft)
    hard_rise = time_to_half_peak(hard)
    assert soft_rise > hard_rise


def test_onset_noise_has_broadband_energy() -> None:
    """Onset noise should contribute energy in both low and high frequency bands."""
    no_noise = render(
        freq=FREQ,
        duration=0.2,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={"hammer_noise": 0.0, "drift": 0.0},
    )
    with_noise = render(
        freq=FREQ,
        duration=0.2,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={"hammer_noise": 0.8, "drift": 0.0},
    )

    onset_end = int(0.035 * SAMPLE_RATE)
    diff = with_noise[:onset_end] - no_noise[:onset_end]

    spectrum = np.abs(np.fft.rfft(diff))
    freqs = np.fft.rfftfreq(onset_end, 1.0 / SAMPLE_RATE)

    low_mask = (freqs >= 100) & (freqs <= 2000)
    high_mask = (freqs >= 2000) & (freqs <= 8000)

    low_energy = float(np.sum(spectrum[low_mask] ** 2))
    high_energy = float(np.sum(spectrum[high_mask] ** 2))

    assert low_energy > 0, "Felt thump should add low-band energy"
    assert high_energy > 0, "String chatter should add high-band energy"


def test_soundboard_adds_body_warmth() -> None:
    """The soundboard should add resonant energy in the body range (200-500 Hz)."""
    no_board = render(
        freq=FREQ,
        duration=0.5,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={"soundboard_color": 0.0, "drift": 0.0, "hammer_noise": 0.0},
    )
    with_board = render(
        freq=FREQ,
        duration=0.5,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={"soundboard_color": 0.8, "drift": 0.0, "hammer_noise": 0.0},
    )

    spectrum_no = np.abs(np.fft.rfft(no_board))
    spectrum_with = np.abs(np.fft.rfft(with_board))
    freqs = np.fft.rfftfreq(no_board.size, 1.0 / SAMPLE_RATE)

    body_mask = (freqs >= 200) & (freqs <= 500)
    body_no = float(np.sum(spectrum_no[body_mask] ** 2))
    body_with = float(np.sum(spectrum_with[body_mask] ** 2))
    assert body_with > body_no, "Soundboard should add body warmth in 200-500 Hz"


def test_backward_compat_decay_two_stage() -> None:
    """The old decay_two_stage param should still be accepted without error."""
    signal = render(
        freq=FREQ,
        duration=0.5,
        amp=AMP,
        sample_rate=SAMPLE_RATE,
        params={"decay_two_stage": 0.3},
    )
    assert np.all(np.isfinite(signal))
    assert np.max(np.abs(signal)) > 0.0
