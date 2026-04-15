"""Multi-point envelope system for drum synthesis engines.

Renders arbitrary piecewise envelopes with linear, exponential, and cubic Bezier
interpolation between points. Time is normalized 0-1 across the note duration;
values are in whatever unit the caller needs (amplitude, frequency ratio, Hz, etc.).

The ``curve`` field on an EnvelopePoint describes how we *arrive* at that point
from the previous one — the first point's curve is therefore ignored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)

_VALID_CURVES = frozenset({"linear", "exponential", "bezier"})


@dataclass(frozen=True, slots=True)
class EnvelopePoint:
    """A single point in a multi-segment envelope."""

    time: float
    value: float
    curve: str = "linear"
    cx: float = 0.0
    cy: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.time <= 1.0:
            raise ValueError(f"time must be in [0, 1], got {self.time}")
        if self.curve not in _VALID_CURVES:
            raise ValueError(
                f"curve must be one of {sorted(_VALID_CURVES)}, got {self.curve!r}"
            )


def parse_envelope_points(
    raw: list[EnvelopePoint | dict] | None,
) -> list[EnvelopePoint]:
    """Convert raw envelope data to a sorted list of EnvelopePoint.

    Accepts a list of EnvelopePoint objects, dicts with matching keys, or None.
    """
    if raw is None:
        return []
    out: list[EnvelopePoint] = []
    for item in raw:
        if isinstance(item, EnvelopePoint):
            out.append(item)
        elif isinstance(item, dict):
            out.append(EnvelopePoint(**item))
        else:
            raise TypeError(
                f"expected EnvelopePoint or dict, got {type(item).__name__}"
            )
    return sorted(out, key=lambda p: p.time)


def render_envelope(
    points: list[EnvelopePoint | dict],
    n_samples: int,
    *,
    default_value: float = 0.0,
) -> np.ndarray:
    """Render a multi-point envelope to a per-sample float64 array.

    *points* are sorted by time internally. Regions before the first point are
    filled with *default_value*; regions after the last point hold the last value.
    """
    if n_samples <= 0:
        return np.empty(0, dtype=np.float64)

    pts = parse_envelope_points(points)  # type: ignore[arg-type]

    if len(pts) == 0:
        return np.full(n_samples, default_value, dtype=np.float64)

    if len(pts) == 1:
        return np.full(n_samples, pts[0].value, dtype=np.float64)

    out = np.empty(n_samples, dtype=np.float64)

    first_sample = int(round(pts[0].time * (n_samples - 1)))
    if first_sample > 0:
        out[:first_sample] = default_value

    last_sample = int(round(pts[-1].time * (n_samples - 1)))
    if last_sample < n_samples - 1:
        out[last_sample + 1 :] = pts[-1].value

    for seg_idx in range(len(pts) - 1):
        p0 = pts[seg_idx]
        p1 = pts[seg_idx + 1]

        i_start = int(round(p0.time * (n_samples - 1)))
        i_end = int(round(p1.time * (n_samples - 1)))

        seg_len = i_end - i_start
        if seg_len <= 0:
            continue

        seg = _render_segment(p0.value, p1.value, seg_len, p1.curve, p1.cx, p1.cy)
        out[i_start : i_start + seg_len] = seg

    out[last_sample] = pts[-1].value
    return out


def _render_segment(
    v0: float,
    v1: float,
    n: int,
    curve: str,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Render a single envelope segment of *n* samples between values *v0* and *v1*."""
    if curve == "linear":
        return _seg_linear(v0, v1, n)
    if curve == "exponential":
        return _seg_exponential(v0, v1, n)
    if curve == "bezier":
        return _seg_bezier(v0, v1, n, cx, cy)
    raise ValueError(f"unknown curve type: {curve!r}")


def _seg_linear(v0: float, v1: float, n: int) -> np.ndarray:
    return np.linspace(v0, v1, n, endpoint=False, dtype=np.float64)


def _seg_exponential(v0: float, v1: float, n: int) -> np.ndarray:
    """Exponential interpolation shaped like an RC decay/rise.

    Uses a time constant chosen so ~95 % of the transition completes by the end of
    the segment (tau = n / 3). Falls back to linear when the value span is tiny.
    """
    if n <= 1:
        return np.array([v0], dtype=np.float64)

    span = v1 - v0
    if abs(span) < 1e-12:
        return np.full(n, v0, dtype=np.float64)

    tau = n / 3.0
    t = np.arange(n, dtype=np.float64)
    alpha = 1.0 - np.exp(-t / tau)
    return v0 + span * alpha


def _seg_bezier(v0: float, v1: float, n: int, cx: float, cy: float) -> np.ndarray:
    """Cubic Bezier easing between two values.

    Maps parametric *t* through a 1D cubic Bezier with two interior control
    values (*cx* and *cy*), then linearly maps the result to [v0, v1].

    When both cx and cy are in [0, 1] the output is guaranteed to stay within
    [v0, v1].  ``cx=cy=0.5`` gives approximately linear interpolation.
    ``cx=0, cy=1`` gives a fast-start/ease-out shape (like exponential decay).
    ``cx=1, cy=0`` gives a slow-start/ease-in shape.
    """
    if n <= 1:
        return np.array([v0], dtype=np.float64)

    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)

    p1 = float(np.clip(cx, 0.0, 1.0))
    p2 = float(np.clip(cy, 0.0, 1.0))

    omt = 1.0 - t
    alpha = 3.0 * omt**2 * t * p1 + 3.0 * omt * t**2 * p2 + t**3
    return v0 + (v1 - v0) * alpha


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------


def exponential_decay(
    tau_ratio: float = 0.2,
    start: float = 1.0,
    end: float = 0.0,
) -> list[EnvelopePoint]:
    """Single-segment exponential decay matching ``np.exp(-t/tau)`` behaviour.

    *tau_ratio* is the time constant expressed as a fraction of total note duration.
    """
    if tau_ratio <= 0:
        raise ValueError(f"tau_ratio must be positive, got {tau_ratio}")
    return [
        EnvelopePoint(time=0.0, value=start),
        EnvelopePoint(time=1.0, value=end, curve="exponential"),
    ]


def attack_decay(
    attack_ratio: float = 0.01,
    peak: float = 1.0,
    decay_tau: float = 0.2,
) -> list[EnvelopePoint]:
    """Quick linear attack to *peak*, then exponential decay to zero."""
    if not 0.0 < attack_ratio < 1.0:
        raise ValueError(f"attack_ratio must be in (0, 1), got {attack_ratio}")
    if decay_tau <= 0:
        raise ValueError(f"decay_tau must be positive, got {decay_tau}")
    return [
        EnvelopePoint(time=0.0, value=0.0),
        EnvelopePoint(time=attack_ratio, value=peak, curve="linear"),
        EnvelopePoint(time=1.0, value=0.0, curve="exponential"),
    ]


def attack_sustain_decay(
    attack_ratio: float = 0.01,
    sustain_ratio: float = 0.5,
    sustain_level: float = 0.8,
    peak: float = 1.0,
    end: float = 0.0,
) -> list[EnvelopePoint]:
    """Attack to *peak*, hold at *sustain_level*, then exponential decay to *end*."""
    if not 0.0 < attack_ratio < 1.0:
        raise ValueError(f"attack_ratio must be in (0, 1), got {attack_ratio}")
    sustain_end = attack_ratio + sustain_ratio
    if not sustain_end < 1.0:
        raise ValueError(
            f"attack_ratio + sustain_ratio must be < 1.0, got {sustain_end}"
        )
    return [
        EnvelopePoint(time=0.0, value=0.0),
        EnvelopePoint(time=attack_ratio, value=peak, curve="linear"),
        EnvelopePoint(time=sustain_end, value=sustain_level, curve="linear"),
        EnvelopePoint(time=1.0, value=end, curve="exponential"),
    ]


def gate(
    hold_ratio: float = 0.8,
    hold_level: float = 1.0,
    release_ms_ratio: float = 0.02,
) -> list[EnvelopePoint]:
    """Hard gate: hold at *hold_level* then quick linear drop to zero."""
    if not 0.0 < hold_ratio < 1.0:
        raise ValueError(f"hold_ratio must be in (0, 1), got {hold_ratio}")
    release_end = min(hold_ratio + release_ms_ratio, 1.0)
    return [
        EnvelopePoint(time=0.0, value=hold_level),
        EnvelopePoint(time=hold_ratio, value=hold_level, curve="linear"),
        EnvelopePoint(time=release_end, value=0.0, curve="linear"),
    ]
