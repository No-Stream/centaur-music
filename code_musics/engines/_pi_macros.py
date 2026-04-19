"""Physical-informed macro resolution for the drum_voice engine.

Higher-level perceptual knobs for the modal tone and metallic-modal layers:
``pi_hardness``, ``pi_tension``, ``pi_damping``, ``pi_damping_tilt``,
``pi_position``.  Each maps to a small set of concrete ``modal_*`` /
``metallic_*`` / exciter params and is non-destructive -- user-set values
always win.  Macro keys are popped after processing.
"""

from __future__ import annotations

from typing import Any

_PI_MACRO_KEYS = frozenset(
    {
        "pi_hardness",
        "pi_tension",
        "pi_damping",
        "pi_damping_tilt",
        "pi_position",
    }
)


def resolve_pi_macros(params: dict[str, Any]) -> dict[str, Any]:
    """Apply physical-informed macro mappings, filling in params not already set.

    Operates on *params* in-place and returns it.  Macro keys are popped
    after processing so they never reach the render function.
    """
    _apply_hardness(params)
    _apply_tension(params)
    _apply_damping(params)
    _apply_damping_tilt(params)
    _apply_position(params)

    for key in _PI_MACRO_KEYS:
        params.pop(key, None)

    return params


def _lerp(t: float, lo: float, hi: float) -> float:
    return lo + t * (hi - lo)


def _set_if_absent(params: dict[str, Any], key: str, value: Any) -> None:
    if key not in params:
        params[key] = value


def _apply_hardness(params: dict[str, Any]) -> None:
    """``pi_hardness`` (0..1): mallet/strike brightness.

    If no exciter is set, installs a short click whose bandpass center scales
    with hardness (500 Hz soft mallet -> 5500 Hz hard strike).  If an exciter
    is already set, biases its bandpass center toward the hardness value
    without overriding a user-supplied ``exciter_center_hz``.
    """
    value = params.get("pi_hardness")
    if value is None:
        return
    t = max(0.0, min(1.0, float(value)))

    center_hz = _lerp(t, 500.0, 5500.0)
    decay_s = _lerp(t, 0.012, 0.004)

    if params.get("exciter_type") is None:
        _set_if_absent(params, "exciter_type", "click")
        _set_if_absent(params, "exciter_level", 0.4)
        _set_if_absent(params, "exciter_center_hz", center_hz)
        _set_if_absent(params, "exciter_decay_s", decay_s)
    else:
        _set_if_absent(params, "exciter_center_hz", center_hz)


def _apply_tension(params: dict[str, Any]) -> None:
    """``pi_tension`` (-1..+1): fractional stretch of mode ratios.

    Positive tension stretches ratios (stiffer material / higher partials),
    negative relaxes them (closer to fundamental).  Stored as a scalar the
    modal renderers consume: ``ratios[i] = base_ratios[i] ** (1 + 0.3 * t)``.
    """
    value = params.get("pi_tension")
    if value is None:
        return
    t = max(-1.0, min(1.0, float(value)))
    _set_if_absent(params, "modal_tension", t)


def _apply_damping(params: dict[str, Any]) -> None:
    """``pi_damping`` (0..1): global decay multiplier for modal banks.

    0 = barely damped / long ring, 1 = heavily damped / short thud.
    Maps to a multiplicative scale on ``modal_decay_s`` / ``metallic_decay_s``
    base decays: ``mult = 2.5 - 2.3 * damping`` (soft 2.5x at damp=0,
    harsh 0.2x at damp=1).
    """
    value = params.get("pi_damping")
    if value is None:
        return
    t = max(0.0, min(1.0, float(value)))
    mult = _lerp(t, 2.5, 0.2)
    _set_if_absent(params, "modal_damping", mult)


def _apply_damping_tilt(params: dict[str, Any]) -> None:
    """``pi_damping_tilt`` (-1..+1): high- vs low-mode decay balance.

    Positive = high modes decay faster (wooden / mellow), negative = high
    modes ring longer (bell-like).  Stored for the modal renderers to apply
    as ``decays[i] *= exp(-tilt * i / n_modes)``.
    """
    value = params.get("pi_damping_tilt")
    if value is None:
        return
    t = max(-1.0, min(1.0, float(value)))
    _set_if_absent(params, "modal_damping_tilt", t)


def _apply_position(params: dict[str, Any]) -> None:
    """``pi_position`` (0..1): strike position window on the amp envelope.

    Modeled as a cosine amplitude window over mode index:
    ``amp[i] *= cos(pi * position * (i+1) / n_modes) ** 2``.  0 keeps all
    modes equal; higher values progressively null out high modes the way a
    struck-at-center membrane does.
    """
    value = params.get("pi_position")
    if value is None:
        return
    t = max(0.0, min(1.0, float(value)))
    _set_if_absent(params, "modal_position", t)
