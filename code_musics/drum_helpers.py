"""Convenience helpers for drum voice setup and routing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from code_musics.score import EffectSpec, Score, VoiceSend

if TYPE_CHECKING:
    from code_musics.score import Voice


def setup_drum_bus(
    score: Score,
    *,
    bus_name: str = "drum_bus",
    effects: list[EffectSpec] | None = None,
    return_db: float = 0.0,
) -> str:
    """Create a shared drum sub-mix bus on *score*.

    Returns the bus name for use in :func:`add_drum_voice` or manual ``VoiceSend``.
    """
    bus_effects = effects if effects is not None else []
    score.add_send_bus(bus_name, effects=bus_effects, return_db=return_db)
    return bus_name


def add_drum_voice(
    score: Score,
    name: str,
    *,
    engine: str,
    preset: str | None = None,
    drum_bus: str | None = None,
    send_db: float = 0.0,
    choke_group: str | None = None,
    effects: list[EffectSpec] | None = None,
    mix_db: float = 0.0,
    normalize_peak_db: float = -6.0,
    synth_overrides: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Voice:
    """Add a percussion voice with standard drum defaults.

    This wraps ``score.add_voice()`` with sensible percussion defaults:
    ``normalize_peak_db=-6.0``, no velocity humanization, and optional
    automatic routing to a shared drum bus.
    """
    synth_defaults: dict[str, Any] = {"engine": engine}
    if preset is not None:
        synth_defaults["preset"] = preset
    if synth_overrides:
        synth_defaults.update(synth_overrides)

    sends: list[VoiceSend] = []
    if drum_bus is not None:
        sends.append(VoiceSend(target=drum_bus, send_db=send_db))

    # choke_group is passed via **kwargs — it will work once the choke-group
    # feature lands in Score.add_voice(); until then Python raises a clear
    # TypeError so both changes must land together.
    if choke_group is not None:
        kwargs["choke_group"] = choke_group

    return score.add_voice(
        name,
        synth_defaults=synth_defaults,
        effects=effects or [],
        normalize_peak_db=normalize_peak_db,
        mix_db=mix_db,
        velocity_humanize=None,
        sends=sends,
        **kwargs,
    )
