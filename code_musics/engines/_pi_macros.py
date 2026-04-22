"""Physical-informed macro resolution for the drum_voice engine.

Higher-level perceptual knobs that fill in concrete ``modal_*`` / ``metallic_*``
/ exciter params.  Macros are non-destructive: user-set values always win.
Macro keys are popped after processing so they never reach the render function.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from code_musics.engines._modal import COUPLING_MAX

_PI_MACRO_KEYS = frozenset(
    {
        "pi_hardness",
        "pi_tension",
        "pi_damping",
        "pi_damping_tilt",
        "pi_position",
        "pi_coupling",
        "pi_dispersion",
    }
)


def _lerp(t: float, lo: float, hi: float) -> float:
    return lo + t * (hi - lo)


def _set_if_absent(params: dict[str, Any], key: str, value: Any) -> None:
    if key not in params:
        params[key] = value


@dataclass(frozen=True)
class _SimpleMacroSpec:
    """Declarative spec for the table-driven simple macros.

    ``macro_key`` is read from ``params``, clamped to ``[lo, hi]``, optionally
    transformed by ``xform``, and written to ``target_key`` via
    :func:`_set_if_absent`.
    """

    macro_key: str
    target_key: str
    lo: float
    hi: float
    xform: Callable[[float], float] | None = None


# Tension and damping_tilt clamp to [-1, 1]; everything else to [0, 1].
# ``pi_coupling`` scales the 0..1 macro into [0, COUPLING_MAX] so callers
# can't silently push the modal bank past its stability-safe ceiling.
_SIMPLE_MACROS: tuple[_SimpleMacroSpec, ...] = (
    _SimpleMacroSpec("pi_tension", "modal_tension", -1.0, 1.0),
    _SimpleMacroSpec(
        "pi_damping",
        "modal_damping",
        0.0,
        1.0,
        xform=lambda t: _lerp(t, 2.5, 0.2),
    ),
    _SimpleMacroSpec("pi_damping_tilt", "modal_damping_tilt", -1.0, 1.0),
    _SimpleMacroSpec("pi_position", "modal_position", 0.0, 1.0),
    _SimpleMacroSpec(
        "pi_coupling",
        "modal_coupling",
        0.0,
        1.0,
        xform=lambda t: t * COUPLING_MAX,
    ),
    _SimpleMacroSpec("pi_dispersion", "modal_dispersion", 0.0, 1.0),
)


def resolve_pi_macros(params: dict[str, Any]) -> dict[str, Any]:
    """Apply physical-informed macro mappings, filling in params not already set.

    Operates on *params* in-place and returns it.  Macro keys are popped after
    processing so they never reach the render function.
    """
    _apply_hardness(params)
    for spec in _SIMPLE_MACROS:
        _apply_simple_macro(params, spec)

    for key in _PI_MACRO_KEYS:
        params.pop(key, None)

    return params


def _apply_simple_macro(params: dict[str, Any], spec: _SimpleMacroSpec) -> None:
    """Table-driven macro application: clamp, transform, write-if-absent."""
    value = params.get(spec.macro_key)
    if value is None:
        return
    t = max(spec.lo, min(spec.hi, float(value)))
    out = spec.xform(t) if spec.xform is not None else t
    _set_if_absent(params, spec.target_key, out)


def _apply_hardness(params: dict[str, Any]) -> None:
    """``pi_hardness`` (0..1): mallet/strike brightness.

    Special-cased because it installs a click exciter when no exciter is set —
    that side effect doesn't fit the simple "clamp, transform, write one key"
    shape of the table-driven macros above.
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
