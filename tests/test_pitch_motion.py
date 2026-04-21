"""Pitch-motion helper tests."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.pitch_motion import PitchMotionSpec, build_frequency_trajectory


def test_linear_bend_trajectory_hits_requested_target() -> None:
    motion = PitchMotionSpec.linear_bend(target_partial=3.0)

    trajectory = build_frequency_trajectory(
        base_freq=220.0,
        duration=0.01,
        sample_rate=1000,
        motion=motion,
        score_f0_hz=110.0,
    )

    assert len(trajectory) == 10
    assert np.isclose(trajectory[0], 220.0)
    assert np.isclose(trajectory[-1], 330.0)
    assert np.all(np.diff(trajectory) >= 0)


def test_ratio_glide_trajectory_is_geometric_and_positive() -> None:
    motion = PitchMotionSpec.ratio_glide(end_ratio=3 / 2)

    trajectory = build_frequency_trajectory(
        base_freq=220.0,
        duration=0.01,
        sample_rate=1000,
        motion=motion,
        score_f0_hz=110.0,
    )

    log_steps = np.diff(np.log(trajectory))

    assert np.isclose(trajectory[0], 220.0)
    assert np.isclose(trajectory[-1], 330.0)
    assert np.all(trajectory > 0)
    assert np.allclose(log_steps, log_steps[0])


def test_vibrato_trajectory_stays_positive_and_deterministic() -> None:
    motion = PitchMotionSpec.vibrato(depth_ratio=0.02, rate_hz=5.0, phase_rad=0.25)

    trajectory = build_frequency_trajectory(
        base_freq=220.0,
        duration=0.02,
        sample_rate=1000,
        motion=motion,
        score_f0_hz=110.0,
    )

    assert np.all(np.isfinite(trajectory))
    assert np.all(trajectory > 0)
    assert not np.allclose(trajectory, trajectory[0])


def test_pitch_motion_validation_rejects_invalid_ratio_glide() -> None:
    with pytest.raises(ValueError, match="end_ratio must be positive"):
        PitchMotionSpec.ratio_glide(end_ratio=0.0)


def test_linear_bend_requires_exactly_one_target_specification() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PitchMotionSpec.linear_bend(target_freq=330.0, target_partial=3.0)

    with pytest.raises(ValueError, match="exactly one"):
        PitchMotionSpec.linear_bend()
