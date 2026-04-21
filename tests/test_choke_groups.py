"""Choke group tests."""

from __future__ import annotations

import numpy as np

from code_musics.score import Score


def test_choke_group_cuts_other_voices() -> None:
    """A note in one choke-group voice should silence other voices in the group."""
    score = Score(f0_hz=440.0, sample_rate=44_100)
    score.add_voice(
        "open_hat",
        synth_defaults={"engine": "noise_perc", "preset": "chh"},
        normalize_peak_db=-6.0,
        choke_group="hats",
    )
    score.add_voice(
        "closed_hat",
        synth_defaults={"engine": "noise_perc", "preset": "chh"},
        normalize_peak_db=-6.0,
        choke_group="hats",
    )
    # Long open hat, then closed hat cuts it
    score.add_note("open_hat", start=0.0, duration=0.5, freq=9000.0, amp=0.5)
    score.add_note("closed_hat", start=0.1, duration=0.04, freq=9000.0, amp=0.5)

    audio = score.render()
    assert isinstance(audio, np.ndarray)
    assert np.isfinite(audio).all()
    # The open hat should be mostly silent after the closed hat onset (0.1s + 10ms fade)
    # Check that the tail of the mix (after 0.15s) is much quieter than the beginning
    sr = 44_100
    early_energy = np.mean(np.abs(audio[0 : int(0.08 * sr)]))
    late_energy = np.mean(np.abs(audio[int(0.15 * sr) : int(0.3 * sr)]))
    assert late_energy < early_energy * 0.3


def test_choke_group_none_does_not_affect_rendering() -> None:
    """Voices without choke_group should render normally."""
    score = Score(f0_hz=440.0, sample_rate=44_100)
    score.add_voice(
        "hat_a",
        synth_defaults={"engine": "noise_perc", "preset": "chh"},
        normalize_peak_db=-6.0,
    )
    score.add_voice(
        "hat_b",
        synth_defaults={"engine": "noise_perc", "preset": "chh"},
        normalize_peak_db=-6.0,
    )
    score.add_note("hat_a", start=0.0, duration=0.3, freq=9000.0, amp=0.5)
    score.add_note("hat_b", start=0.05, duration=0.3, freq=9000.0, amp=0.5)

    audio = score.render()
    assert isinstance(audio, np.ndarray)
    # Both voices should be audible throughout — no choking
    sr = 44_100
    late_energy = np.mean(np.abs(audio[int(0.1 * sr) : int(0.2 * sr)]))
    assert late_energy > 0.001


def test_choke_group_fade_is_smooth() -> None:
    """The choke fade should be gradual (10ms), not a hard cut."""
    score = Score(f0_hz=440.0, sample_rate=44_100)
    score.add_voice(
        "open_hat",
        synth_defaults={"engine": "additive"},
        normalize_peak_db=-6.0,
        choke_group="hats",
    )
    score.add_voice(
        "closed_hat",
        synth_defaults={"engine": "additive"},
        normalize_peak_db=-6.0,
        choke_group="hats",
    )
    score.add_note("open_hat", start=0.0, duration=1.0, freq=2000.0, amp=0.5)
    score.add_note("closed_hat", start=0.3, duration=0.1, freq=2000.0, amp=0.5)

    audio = score.render()
    sr = 44_100
    # Check that the sample right at the choke point is NOT zero (fade hasn't completed)
    choke_sample = int(0.3 * sr)
    # The signal should still be non-zero at the choke point
    assert np.max(np.abs(audio[choke_sample : choke_sample + 5])) > 0
    # But 20ms later it should be essentially silent
    after_fade = int(0.32 * sr)
    assert np.max(np.abs(audio[after_fade : after_fade + 100])) < np.max(
        np.abs(audio[choke_sample : choke_sample + 5])
    )
