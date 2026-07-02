"""Convenience helpers for drum voice setup and routing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from code_musics.score import EffectSpec, Score, VoiceSend

if TYPE_CHECKING:
    from code_musics.score import Voice

# Sentinel to distinguish "user didn't pass effects" from "user passed effects=[]".
_NO_EFFECTS: object = object()


# ---------------------------------------------------------------------------
# Drum bus style presets — default-on chains providing finished-kit glue.
#
# Order: compressor -> preamp/tube -> clipper (on heavier styles).
# The compressor glues dynamics first so transients are shaped by raw
# envelope; the middle stage adds harmonic color to the glued signal
# (apply_tube for triode/pentode character, apply_preamp for flux-domain
# transformer warmth with no papery treble buildup); the clipper shaves
# the remaining kick peaks with a polynomial soft-knee, leaving the
# master-bus limiter less absolute work to do.
#
# True-peak ceiling lives on the master bus (see DEFAULT_MASTER_EFFECTS
# in pieces/_shared.py). Bus-level limiters are unusual and redundant
# once the master has one.
#
# Clipper rewrite (April 2026):
# -----------------------------
# apply_clipper was disabled on these styles prior to the polynomial-knee
# rewrite. Its old default blended 85% hard-clip with 15% tanh, which
# (per scratch/clipper_bisect.py and output/clipper_forensics/bisect_v2.json)
# added +64% IMD on a two-tone stimulus and +6.5 dB of 2-8 kHz brightness
# on a kick at the same shave amount. The monotone cubic-Hermite poly-knee
# now reduces 2-8 kHz lift *below* naive hard-clip at the same shave,
# so the clipper is back on drum-bus duty where it belongs.
#
# Piece-aware calibration
# -----------------------
# Compressor uses ``target_avg_gr_db``, which binary-searches ``threshold_db``
# so that the average gain reduction on active samples (detector envelope
# within 40 dB of its p99 peak) matches the target. This means each style
# delivers the same *character* of gain reduction regardless of how hot the
# incoming mix is — "4 dB of comp" means "you'll measure 4 dB avg GR on the
# parts the compressor is working."
#
# Clipper uses ``max_shave_db`` so it targets a fixed peak-reduction depth
# regardless of input hotness.  This pairs cleanly with the
# target_avg_gr_db comp: each style delivers its intended glue *plus* its
# intended peak shave at a predictable total depth.  Knee width controls
# character: narrower (1-2 dB) for Berghain-wall edge, wider (3-4 dB) for
# forgiving kick-forward forward.
#
# Calibration assumes individual drum voices hit the bus with per-voice
# normalize_peak_db=-6.0 plus per-voice comp/sat from add_drum_voice's
# preset-aware defaults. Adjust via `style="..."` or override entirely with
# effects=[...].
# ---------------------------------------------------------------------------

_DRUM_BUS_STYLES: dict[str, list[EffectSpec]] = {
    "light": [
        # Four Tet / BoC territory: subtle glue, transients mostly intact.
        # target_avg_gr_db=2.5 — "light touch" design-intent band (2-3 dB).
        EffectSpec(
            "compressor",
            {
                "target_avg_gr_db": 2.5,
                "ratio": 1.8,
                "attack_ms": 20.0,
                "release_ms": 180.0,
                "knee_db": 6.0,
                "topology": "feedback",
                "detector_mode": "rms",
                "detector_bands": [
                    {"kind": "highpass", "cutoff_hz": 60.0, "slope_db_per_oct": 12}
                ],
                "mix": 0.7,
            },
        ),
        EffectSpec("tube", {"character": "triode", "drive": 1.4, "mix": 0.22}),
    ],
    "electronic": [
        # Default — modern electronic glue. target_avg_gr_db=5.0 — audible
        # dynamics control (4-6 dB design-intent band). Flux-domain preamp
        # warmth (no papery treble buildup), then a poly-knee clipper
        # shaving ~1.0 dB off the remaining p99 kick peaks with a mild
        # (2.5 dB) knee.  Real-world absolute peaks exceed p99 by ~1-1.5 dB
        # on 4/4 drum material, so measured shave typically lands near 2 dB
        # (within the calibrated "subtle peak shave" band).
        EffectSpec(
            "compressor",
            {
                "target_avg_gr_db": 5.0,
                "ratio": 2.4,
                "attack_ms": 10.0,
                "release_ms": 160.0,
                "knee_db": 5.0,
                "topology": "feedback",
                "detector_mode": "rms",
                "detector_bands": [
                    {"kind": "highpass", "cutoff_hz": 70.0, "slope_db_per_oct": 12}
                ],
            },
        ),
        EffectSpec("preamp", {"drive": 0.7, "mix": 0.35}),
        EffectSpec(
            "clipper",
            {"max_shave_db": 1.0, "knee_width_db": 2.5, "oversample_factor": 8},
        ),
    ],
    "weighty": [
        # Kick-forward, iron-heavy. target_avg_gr_db=6.0 — heavier glue
        # (5-7 dB design-intent band); slower comp for deeper glue on
        # isolated hits vs. electronic's faster attack. Flux-domain preamp
        # adds bass-emphasis saturation, clipper shaves ~1.5 dB off the p99
        # with a wide (3.5 dB) knee — kick-forward material runs hotter, so
        # measured shave typically lands 2.5-3 dB, the right "weighty but
        # musical" depth for this style.  Knee width forgives the
        # transient corner on isolated hits.
        EffectSpec(
            "compressor",
            {
                "target_avg_gr_db": 6.0,
                "ratio": 2.8,
                "attack_ms": 18.0,
                "release_ms": 220.0,
                "knee_db": 6.0,
                "topology": "feedback",
                "detector_mode": "rms",
                "detector_bands": [
                    {"kind": "highpass", "cutoff_hz": 55.0, "slope_db_per_oct": 12}
                ],
            },
        ),
        EffectSpec("preamp", {"preset": "iron_color", "drive": 0.42}),
        EffectSpec(
            "clipper",
            {"max_shave_db": 1.5, "knee_width_db": 3.5, "oversample_factor": 8},
        ),
    ],
    "berghain": [
        # Peak-hour techno wall — smashed, but still musical.
        # target_avg_gr_db=7.0 — smashed (6-8 dB design-intent band).
        # Flux-domain preamp in transformer-drive territory for heavy iron
        # color, then a narrow-knee (1.5 dB) clipper shaving up to 3 dB
        # for the "wall" character.  Narrower knee produces more 2nd/3rd
        # harmonic than wide, which is the right side of the knob for
        # peak-hour edge.
        #
        # UNTESTED on real material — no piece currently uses this style.
        # Values above inherit the same "target < measured by ~1-2 dB"
        # relationship as electronic/weighty, so measured shave is likely
        # 4-5 dB when adopted.  Retune when a piece exercises it.
        EffectSpec(
            "compressor",
            {
                "target_avg_gr_db": 7.0,
                "ratio": 4.0,
                "attack_ms": 6.0,
                "release_ms": 120.0,
                "knee_db": 4.0,
                "topology": "feedback",
                "detector_mode": "rms",
                "detector_bands": [
                    {"kind": "highpass", "cutoff_hz": 80.0, "slope_db_per_oct": 12}
                ],
            },
        ),
        EffectSpec("preamp", {"drive": 1.1, "mix": 0.45}),
        EffectSpec(
            "clipper",
            {"max_shave_db": 3.0, "knee_width_db": 1.5, "oversample_factor": 8},
        ),
    ],
}

DRUM_BUS_STYLES: frozenset[str] = frozenset(_DRUM_BUS_STYLES)

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
            # Flux-domain preamp (iron-core transformer saturation) is the
            # architecturally correct tool for "body without brightness" —
            # bass saturates more than highs in the flux model, while the
            # previous drive-based `kick_weight` added ~5 dB of 2-5 kHz
            # papery lift that kicked off `chain_papery` warnings.
            EffectSpec("preamp", {"preset": "kick_body"}),
        ]
    if preset in _TOM_PRESETS:
        return [
            EffectSpec("compressor", {"preset": "tom_control"}),
            EffectSpec("preamp", {"preset": "tom_body"}),
        ]
    if preset in _SNARE_PRESETS:
        return [EffectSpec("compressor", {"preset": "snare_punch"})]
    if preset in _HAT_PRESETS:
        return [EffectSpec("compressor", {"preset": "hat_control"})]
    # Clap presets and unknown/custom presets get no default effects.
    return []


def setup_drum_bus(
    score: Score,
    *,
    style: str = "electronic",
    bus_name: str = "drum_bus",
    effects: list[EffectSpec] | None = None,
    return_db: float = 0.0,
) -> str:
    """Create a shared drum sub-mix bus on *score*.

    Returns the bus name for use in :func:`add_drum_voice` or manual
    ``VoiceSend``.

    Parameters
    ----------
    style
        Named chain from :data:`DRUM_BUS_STYLES`. Default ``"electronic"``
        gives a finished modern-electronic kit with compressor glue and
        flux-domain preamp warmth (two-stage chain: comp -> sat). True-peak
        management lives on the master bus, not the drum bus. Options:
        ``"light"`` (Four Tet / BoC clean glue), ``"electronic"`` (default
        squash), ``"weighty"`` (iron-heavy, kick-forward), ``"berghain"``
        (peak-hour techno wall). Ignored when ``effects`` is provided.
    effects
        Explicit effect chain. When given, fully replaces the ``style``
        chain — pass ``effects=[]`` for a bare bus.
    """
    if effects is not None:
        bus_effects = effects
    else:
        if style not in _DRUM_BUS_STYLES:
            raise ValueError(
                f"Unknown drum bus style {style!r}. "
                f"Choose from: {sorted(DRUM_BUS_STYLES)}"
            )
        bus_effects = list(_DRUM_BUS_STYLES[style])
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
    percussive: bool | None = None,
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

    Parameters
    ----------
    percussive
        Override the automatic percussive-voice detection used by effect-
        chain instrumentation. ``None`` (default) runs auto-detection (see
        :meth:`Voice.is_percussive`). Set ``False`` to opt a drum voice out
        of per-hit transient diagnostics (unusual), or ``True`` to force
        them on (e.g., a non-drum engine used percussively).
    """
    resolved_effects: list[EffectSpec] = (  # ty: ignore[invalid-assignment]
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
        percussive=percussive,
        **kwargs,
    )
