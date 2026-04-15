"""Tests for the multi-point envelope system."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.engines._envelopes import (
    EnvelopePoint,
    exponential_decay,
    gate,
    parse_envelope_points,
    render_envelope,
)


def test_render_envelope_two_point_linear() -> None:
    pts = [
        EnvelopePoint(time=0.0, value=1.0),
        EnvelopePoint(time=1.0, value=0.0, curve="linear"),
    ]
    env = render_envelope(pts, 101)

    assert env[0] == pytest.approx(1.0)
    assert env[50] == pytest.approx(0.5, abs=0.02)
    assert env[-1] == pytest.approx(0.0)


def test_render_envelope_exponential_matches_shape() -> None:
    pts = [
        EnvelopePoint(time=0.0, value=1.0),
        EnvelopePoint(time=1.0, value=0.0, curve="exponential"),
    ]
    env = render_envelope(pts, 200)

    diffs = np.diff(env)
    assert np.all(diffs <= 1e-12), (
        "exponential decay should be monotonically decreasing"
    )

    mid = env[100]
    assert mid < 0.5, "exponential decay should be below linear midpoint at halfway"


def test_render_envelope_exponential_rise() -> None:
    pts = [
        EnvelopePoint(time=0.0, value=0.0),
        EnvelopePoint(time=1.0, value=1.0, curve="exponential"),
    ]
    env = render_envelope(pts, 200)

    diffs = np.diff(env)
    assert np.all(diffs >= -1e-12), (
        "exponential rise should be monotonically increasing"
    )


def test_render_envelope_bezier_stays_in_bounds() -> None:
    pts = [
        EnvelopePoint(time=0.0, value=0.0),
        EnvelopePoint(time=1.0, value=1.0, curve="bezier", cx=0.2, cy=0.8),
    ]
    env = render_envelope(pts, 500)

    assert np.all(env >= -0.2), "bezier should not wildly undershoot"
    assert np.all(env <= 1.2), "bezier should not wildly overshoot"
    assert env[-1] == pytest.approx(1.0)


def test_render_envelope_single_point_constant() -> None:
    pts = [EnvelopePoint(time=0.3, value=0.7)]
    env = render_envelope(pts, 100)

    assert len(env) == 100
    np.testing.assert_allclose(env, 0.7)


def test_render_envelope_empty_returns_default() -> None:
    env = render_envelope([], 50, default_value=-3.0)

    assert len(env) == 50
    np.testing.assert_allclose(env, -3.0)


def test_render_envelope_zero_samples() -> None:
    env = render_envelope([], 0)
    assert len(env) == 0


def test_parse_envelope_points_from_dicts() -> None:
    raw = [
        {"time": 0.5, "value": 1.0, "curve": "exponential"},
        {"time": 0.0, "value": 0.0},
    ]
    pts = parse_envelope_points(raw)

    assert len(pts) == 2
    assert all(isinstance(p, EnvelopePoint) for p in pts)
    assert pts[0].time == 0.0, "should be sorted by time"
    assert pts[1].time == 0.5
    assert pts[1].curve == "exponential"


def test_parse_envelope_points_none() -> None:
    assert parse_envelope_points(None) == []


def test_exponential_decay_factory() -> None:
    pts = exponential_decay(tau_ratio=0.2, start=1.0, end=0.0)
    env = render_envelope(pts, 200)

    assert env[0] == pytest.approx(1.0)
    assert env[-1] == pytest.approx(0.0)
    diffs = np.diff(env)
    assert np.all(diffs <= 1e-12), "should be monotonically decreasing"

    assert env[60] < 0.5, "should decay meaningfully by 30% of duration"


def test_gate_factory() -> None:
    pts = gate(hold_ratio=0.8, hold_level=1.0, release_ms_ratio=0.1)
    env = render_envelope(pts, 1000)

    hold_region = env[: int(0.8 * 999)]
    np.testing.assert_allclose(hold_region, 1.0, atol=0.01)

    assert env[-1] == pytest.approx(0.0, abs=0.01)


def test_render_envelope_holds_last_value() -> None:
    pts = [
        EnvelopePoint(time=0.0, value=0.0),
        EnvelopePoint(time=0.5, value=1.0, curve="linear"),
    ]
    env = render_envelope(pts, 1000)

    tail = env[600:]
    np.testing.assert_allclose(tail, 1.0)


def test_invalid_curve_raises() -> None:
    with pytest.raises(ValueError, match="curve must be"):
        EnvelopePoint(time=0.0, value=1.0, curve="cubic")


def test_invalid_time_raises() -> None:
    with pytest.raises(ValueError, match="time must be"):
        EnvelopePoint(time=1.5, value=1.0)
