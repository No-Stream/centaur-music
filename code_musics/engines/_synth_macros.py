"""Ergonomic macro resolution for the synth_voice engine.

Four perceptual macros — ``brightness``, ``movement``, ``body``, ``dirt`` —
map high-level knobs to multiple low-level synthesis params.  Macros only fill
in params that are not already present in the dict (i.e. not set by the user
or a preset).  The macro keys themselves are popped after processing so they
never reach the render function.

Macro values follow the project knob convention: 0.2 subtle, 0.33
clear-but-subtle, 0.5 moderate, 0.66 strong, 0.8–1.0 intense but musical.
Resolution order: preset params -> macro fill-in -> user kwargs win.
"""

from __future__ import annotations

import math
from typing import Any

_MACRO_KEYS = frozenset({"brightness", "movement", "body", "dirt"})


def resolve_macros(params: dict[str, Any]) -> dict[str, Any]:
    """Apply macro mappings, filling in params not already set.

    Operates on *params* in-place and returns it for convenience.  Macro keys
    are removed after processing.
    """
    _apply_brightness(params)
    _apply_movement(params)
    _apply_body(params)
    _apply_dirt(params)

    for key in _MACRO_KEYS:
        params.pop(key, None)

    return params


def _lerp(t: float, lo: float, hi: float) -> float:
    return lo + t * (hi - lo)


def _exp_lerp(t: float, lo: float, hi: float) -> float:
    """Exponential interpolation: t=0 -> lo, t=1 -> hi (log-spaced).

    Used for frequency-domain params where perception is logarithmic.
    """
    if lo <= 0.0 or hi <= 0.0:
        return _lerp(t, lo, hi)
    return lo * math.exp(t * math.log(hi / lo))


def _set_if_absent(params: dict[str, Any], key: str, value: Any) -> None:
    if key not in params:
        params[key] = value


def _apply_brightness(params: dict[str, Any]) -> None:
    """``brightness`` (0=dark, 1=bright).

    Spectral tilt bias: pushes filter cutoff up (log-scaled), tilts partials
    brighter, nudges FM index and supersaw spread slightly at high values.
    """
    value = params.get("brightness")
    if value is None:
        return
    t = float(value)

    _set_if_absent(params, "filter_cutoff_hz", _exp_lerp(t, 400.0, 8000.0))
    _set_if_absent(params, "partials_brightness_tilt", _lerp(t, -0.3, 0.6))
    if t > 0.5:
        high = (t - 0.5) * 2.0
        _set_if_absent(params, "fm_index", _lerp(high, 1.0, 2.0))
        _set_if_absent(params, "osc_spread_cents", _lerp(high, 10.0, 22.0))


def _apply_movement(params: dict[str, Any]) -> None:
    """``movement`` (0=static, 1=alive).

    Timbral motion: filter envelope depth, default chorus mix, partials
    spectral decorrelation (smear, phase_disperse).
    """
    value = params.get("movement")
    if value is None:
        return
    t = float(value)

    _set_if_absent(params, "filter_env_amount", _lerp(t, 0.0, 0.6))
    _set_if_absent(params, "partials_smear", _lerp(t, 0.0, 0.4))
    _set_if_absent(params, "partials_phase_disperse", _lerp(t, 0.0, 0.5))
    _set_if_absent(params, "chorus_mix", _lerp(t, 0.0, 0.35))


def _apply_body(params: dict[str, Any]) -> None:
    """``body`` (0=thin, 1=heavy).

    Low-end weight: inverse HPF bias, gentle resonance bump, sub/low-voice
    level, partials lean toward even harmonics.
    """
    value = params.get("body")
    if value is None:
        return
    t = float(value)

    _set_if_absent(params, "hpf_cutoff_hz", _exp_lerp(1.0 - t, 20.0, 200.0))
    _set_if_absent(params, "resonance_q", _lerp(t, 0.707, 1.1))
    _set_if_absent(params, "osc2_sub_level", _lerp(t, 0.0, 0.5))
    _set_if_absent(params, "partials_odd_even_balance", _lerp(t, 0.0, 0.3))


def _apply_dirt(params: dict[str, Any]) -> None:
    """``dirt`` (0=clean, 1=crunchy).

    Saturation/distortion bias: voice shaper drive + mix, shaper mode
    progression (soft -> preamp -> hard), small feedback bump at extremes.
    """
    value = params.get("dirt")
    if value is None:
        return
    t = float(value)

    if t < 0.05:
        shaper_value: str | None = None
    elif t < 0.35:
        shaper_value = "saturation"
    elif t < 0.7:
        shaper_value = "preamp"
    else:
        shaper_value = "hard_clip"

    if shaper_value is not None:
        _set_if_absent(params, "shaper", shaper_value)

    _set_if_absent(params, "shaper_drive", _lerp(t, 0.0, 0.75))
    _set_if_absent(params, "shaper_mix", _lerp(t, 0.5, 1.0))

    if t > 0.8:
        high = (t - 0.8) * 5.0
        _set_if_absent(params, "feedback_amount", _lerp(high, 0.0, 0.15))
