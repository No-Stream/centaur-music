"""Ergonomic macro resolution for the drum_voice engine.

Three Jomox-style perceptual macros — ``punch``, ``decay_shape``, ``character`` —
map high-level knobs to multiple low-level synthesis params.  Macros only fill in
params that are not already present in the dict (i.e. not set by the user or a
preset).  The macro keys themselves are popped after processing so they never
reach the render function.
"""

from __future__ import annotations

from typing import Any

_MACRO_KEYS = frozenset({"punch", "decay_shape", "character"})


def resolve_macros(params: dict[str, Any]) -> dict[str, Any]:
    """Apply macro mappings, filling in params not already set.

    Operates on *params* in-place and returns it for convenience.  Macro keys
    are removed after processing.
    """
    _apply_punch(params)
    _apply_decay_shape(params)
    _apply_character(params)

    # Always pop macro keys, even when None / inactive.
    for key in _MACRO_KEYS:
        params.pop(key, None)

    return params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lerp(t: float, lo: float, hi: float) -> float:
    """Linear interpolation: t=0 -> lo, t=1 -> hi."""
    return lo + t * (hi - lo)


def _set_if_absent(params: dict[str, Any], key: str, value: Any) -> None:
    """Set *key* in *params* only if it is not already present."""
    if key not in params:
        params[key] = value


# ---------------------------------------------------------------------------
# Macro implementations
# ---------------------------------------------------------------------------


def _apply_punch(params: dict[str, Any]) -> None:
    """``punch`` (0.0=soft pillowy, 1.0=hard snappy)."""
    value = params.get("punch")
    if value is None:
        return
    t = float(value)

    _set_if_absent(params, "exciter_level", _lerp(t, 0.01, 0.25))
    _set_if_absent(params, "exciter_decay_s", _lerp(t, 0.012, 0.003))
    _set_if_absent(params, "exciter_center_hz", _lerp(t, 1500.0, 5000.0))
    _set_if_absent(params, "tone_punch", _lerp(t, 0.0, 0.35))


def _apply_decay_shape(params: dict[str, Any]) -> None:
    """``decay_shape`` (0.0=tight/gated, 1.0=long/boomy)."""
    value = params.get("decay_shape")
    if value is None:
        return
    t = float(value)

    _set_if_absent(params, "tone_decay_s", _lerp(t, 0.08, 0.9))
    _set_if_absent(params, "noise_decay_s", _lerp(t, 0.015, 0.3))
    _set_if_absent(params, "metallic_decay_s", _lerp(t, 0.03, 0.4))
    _set_if_absent(params, "tone_sweep_decay_s", _lerp(t, 0.02, 0.08))


def _apply_character(params: dict[str, Any]) -> None:
    """``character`` (0.0=clean/pure, 1.0=dirty/complex)."""
    value = params.get("character")
    if value is None:
        return
    t = float(value)

    # tone_shaper: None < 0.3, tanh 0.3-0.7, foldback >= 0.7
    if t < 0.3:
        shaper_value = None
    elif t < 0.7:
        shaper_value = "tanh"
    else:
        shaper_value = "foldback"

    if shaper_value is not None:
        _set_if_absent(params, "tone_shaper", shaper_value)

    _set_if_absent(params, "tone_shaper_drive", _lerp(t, 0.0, 0.6))
    _set_if_absent(params, "filter_drive", _lerp(t, 0.0, 0.3))

    # Boost existing noise_level rather than setting a new one.
    # Only applies if noise_level is already present (from preset or user).
    if "noise_level" in params:
        params["noise_level"] = params["noise_level"] * (1.0 + 0.3 * t)
