"""Convenience helpers for drum voice setup and routing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from code_musics.score import EffectSpec, Score, VoiceSend

if TYPE_CHECKING:
    from code_musics.score import Voice

# Sentinel to distinguish "user didn't pass effects" from "user passed effects=[]".
_NO_EFFECTS: object = object()

# ---------------------------------------------------------------------------
# Preset category sets — used to select default voice-level effects.
# ---------------------------------------------------------------------------

_KICK_PRESETS: frozenset[str] = frozenset(
    {
        "808_hiphop",
        "808_house",
        "808_tape",
        "909_techno",
        "909_house",
        "909_crunch",
        "distorted_hardkick",
        "zap_kick",
        "gated_808",
        "pitch_dive",
        "filtered_kick",
        "fm_body_kick",
        "foldback_kick",
        "808_resonant",
        "808_resonant_long",
    }
)

_TOM_PRESETS: frozenset[str] = frozenset(
    {
        "round_tom",
        "floor_tom",
        "electro_tom",
        "ring_tom",
        "melodic_resonator",
        "kick_bell",
        "resonant_tom",
    }
)

_SNARE_PRESETS: frozenset[str] = frozenset(
    {
        "909_tight",
        "909_fat",
        "rim_shot",
        "brush",
        "fm_tom",
        "fm_noise_burst",
        "gated_snare",
        "fm_snare",
        "driven_snare",
    }
)

_HAT_PRESETS: frozenset[str] = frozenset(
    {
        "closed_hat",
        "open_hat",
        "808_closed_hat",
        "808_open_hat",
        "808_cowbell_square",
        "harmonic_bell",
        "septimal_bell",
        "square_gamelan",
        "beating_hat_a",
        "beating_hat_b",
        "beating_hat_c",
        "swept_hat",
        "decaying_bell",
    }
)

_CLAP_PRESETS: frozenset[str] = frozenset(
    {
        "909_clap",
        "tight_clap",
        "big_clap",
        "finger_snap",
        "gated_clap",
        "909_clap_authentic",
        "scattered_clap",
        "granular_cascade",
        "micro_burst",
    }
)


def _default_effects_for_preset(preset: str | None) -> list[EffectSpec]:
    """Return recommended voice-level effects for a drum preset category."""
    if preset is None:
        return []
    if preset in _KICK_PRESETS:
        return [
            EffectSpec("compressor", {"preset": "kick_punch"}),
            EffectSpec("saturation", {"preset": "kick_weight"}),
        ]
    if preset in _TOM_PRESETS:
        return [EffectSpec("compressor", {"preset": "tom_control"})]
    if preset in _SNARE_PRESETS:
        return [EffectSpec("compressor", {"preset": "snare_punch"})]
    if preset in _HAT_PRESETS:
        return [EffectSpec("compressor", {"preset": "hat_control"})]
    # Clap presets and unknown/custom presets get no default effects.
    return []


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
    effects: list[EffectSpec] | None | object = _NO_EFFECTS,
    mix_db: float = 0.0,
    normalize_peak_db: float = -6.0,
    synth_overrides: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Voice:
    """Add a percussion voice with standard drum defaults.

    This wraps ``score.add_voice()`` with sensible percussion defaults:
    ``normalize_peak_db=-6.0``, no velocity humanization, and optional
    automatic routing to a shared drum bus.

    When *effects* is not provided, preset-aware defaults are applied
    (e.g. ``kick_punch`` compressor for kick presets).  Pass ``effects=[]``
    explicitly to suppress default effects.
    """
    resolved_effects: list[EffectSpec] = (
        _default_effects_for_preset(preset)
        if effects is _NO_EFFECTS
        else (effects or [])  # type: ignore[arg-type]
    )

    synth_defaults: dict[str, Any] = {"engine": engine}
    if preset is not None:
        synth_defaults["preset"] = preset
    if synth_overrides:
        synth_defaults.update(synth_overrides)

    sends: list[VoiceSend] = []
    if drum_bus is not None:
        sends.append(VoiceSend(target=drum_bus, send_db=send_db))

    if choke_group is not None:
        kwargs["choke_group"] = choke_group

    return score.add_voice(
        name,
        synth_defaults=synth_defaults,
        effects=resolved_effects,
        normalize_peak_db=normalize_peak_db,
        mix_db=mix_db,
        velocity_humanize=None,
        sends=sends,
        **kwargs,
    )
