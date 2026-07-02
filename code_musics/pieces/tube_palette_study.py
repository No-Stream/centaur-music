"""tube_palette_study — A/B tour of the four tube characters across two drive levels.

Plays the same 8-bar bed + lead through each of the four ``apply_tube``
characters at gentle and aggressive settings, back-to-back, so the listener
can hear the distinctive harmonic signature of each:

  1. triode @ gentle    — preset ``triode_glow``.  Subtle H2 amp-glow warmth.
  2. triode @ aggressive — preset ``triode_bloom``.  Bass-bloom, more colour.
  3. pentode @ gentle    — preset ``pentode_bite``.  Mid-focused H3 edge.
  4. pentode @ aggressive — custom.  Hard knee, fatter mix level.
  5. hg2 @ gentle        — preset ``hg2_enhancer``.  Black-Box-style sweetener.
  6. hg2 @ aggressive    — preset ``hg2_drive``.  Euphonic cascade pushed.
  7. culture @ warm      — preset ``culture_warm``.  Even-harmonic richness.
  8. culture @ starved   — preset ``culture_starve``.  One half-cycle cut —
                           the headline Culture-Vulture sound.

Each section is 14 seconds of the same musical material: a sustained
three-note JI chord bed (carrier of the tube colour) plus a simple moving
lead line (transient reference).  Only the bed has the tube effect; the
lead is untouched across all eight sections so it acts as a control.
One bed voice per section so each section can bake its own tube preset
into its own ``effects`` chain without rebuilding the chain at render time.

Key in F# minor (185 Hz f0), matching ``filter_palette_study`` and
``diva_study`` so A/B comparisons across study pieces are fair.
"""

from __future__ import annotations

from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score

SECTION_DUR = 14.0
N_SECTIONS = 8

# Tube characters in presentation order.  Each tuple is:
#   (section key, display label, tube-effect params dict).
# The params dict goes straight into ``EffectSpec("tube", params)`` for that
# section's bed voice.  Gentle/aggressive pairs are adjacent so the A/B is
# immediate.
_TUBE_SECTIONS: list[tuple[str, str, dict[str, object]]] = [
    (
        "triode_gentle",
        "Triode @ gentle (triode_glow)",
        {"preset": "triode_glow"},
    ),
    (
        "triode_aggressive",
        "Triode @ aggressive (triode_bloom)",
        {"preset": "triode_bloom"},
    ),
    (
        "pentode_gentle",
        "Pentode @ gentle (pentode_bite)",
        {"preset": "pentode_bite"},
    ),
    (
        "pentode_aggressive",
        "Pentode @ aggressive (hard knee)",
        {"character": "pentode", "drive": 0.7, "sharpness": 5.5, "mix": 0.6},
    ),
    (
        "hg2_gentle",
        "HG2 @ gentle (hg2_enhancer)",
        {"preset": "hg2_enhancer"},
    ),
    (
        "hg2_aggressive",
        "HG2 @ aggressive (hg2_drive)",
        # hg2_drive preset (pentode_drive=0.6, triode_drive=0.7, parallel=0.35)
        # drives substantial high-band lift into the reverb on rich JI chords.
        # Pull mix back so the per-section level matches its neighbors rather
        # than dominating and clipping the master bus.
        {"preset": "hg2_drive", "mix": 0.45},
    ),
    (
        "culture_warm",
        "Culture @ warm (culture_warm)",
        {"preset": "culture_warm"},
    ),
    (
        "culture_starved",
        "Culture @ starved (culture_starve)",
        {"preset": "culture_starve"},
    ),
]


# Rich-spectrum bed voice — additive slot on synth_voice gives a thick
# harmonic source for the tube shaper to chew on, which is where the H2/H3
# signature differences are most audible.  Neutral filter so the tube
# character is the only variable across sections.
_BED_BASE: dict[str, object] = {
    "engine": "synth_voice",
    "osc_type": "polyblep",
    "waveform": "saw",
    "partials_type": "additive",
    "partials_brightness": 0.45,
    "filter_topology": "svf",
    "filter_cutoff_hz": 1_800.0,
    "resonance_q": 0.9,
    "attack": 0.35,
    "decay": 0.6,
    "sustain_level": 0.85,
    "release": 1.6,
}

# Reference lead voice — same across all sections, no tube effect, just a
# touch of filter movement so the ear has a stable anchor while the bed
# colour shifts.
_LEAD_BASE: dict[str, object] = {
    "engine": "polyblep",
    "waveform": "triangle",
    "cutoff_hz": 2_200.0,
    "filter_env_amount": 1.2,
    "filter_env_decay": 0.4,
    "resonance_q": 1.4,
    "attack": 0.01,
    "decay": 0.18,
    "sustain_level": 0.7,
    "release": 0.35,
}


def _reverb() -> EffectSpec:
    return EffectSpec(
        "reverb",
        {"room_size": 0.6, "damping": 0.45, "wet_level": 0.2},
    )


def _delay() -> EffectSpec:
    return EffectSpec(
        "delay",
        {"delay_seconds": 0.375, "feedback": 0.25, "mix": 0.15},
    )


# 7-limit JI chord bed.  Three partials per change, with a ii-V-i-ish
# motion: tonic triad -> subdominant-ish -> dominant-ish -> resolve.
# Low-register triads (root < 2.0) load the bass band where tube bloom
# lives; the 7/4 and 5/4 members give H2/H3 interaction headroom.
_BED_VOICINGS: list[tuple[float, list[float]]] = [
    (0.0, [1.0, 5 / 4, 3 / 2]),  # Bars 1-2: i triad
    (3.5, [4 / 3, 5 / 3, 2.0]),  # Bars 3-4: iv-ish
    (7.0, [9 / 8, 3 / 2, 7 / 4]),  # Bars 5-6: V with septimal 7
    (10.5, [1.0, 6 / 5, 3 / 2]),  # Bars 7-8: minor i resolve
]
_BED_CHORD_HOLD = 3.5  # seconds held per chord


# Simple moving lead line — four phrases, one per chord change.  Entries
# are (t_in_section, duration, partial, amp_db).  Dense enough to keep the
# ear engaged; sparse enough that the tube-colored bed dominates.
_LEAD_PHRASE: list[tuple[float, float, float, float]] = [
    # Over chord 1
    (0.4, 0.8, 2.0, -12.0),
    (1.4, 0.6, 5 / 2, -13.0),
    (2.2, 1.1, 3.0, -11.0),
    # Over chord 2
    (3.8, 0.8, 8 / 3, -12.0),
    (4.8, 0.6, 10 / 3, -13.0),
    (5.6, 1.2, 4.0, -11.0),
    # Over chord 3
    (7.2, 0.7, 3.0, -12.0),
    (8.0, 0.6, 7 / 2, -13.0),
    (8.8, 1.4, 9 / 4, -10.0),
    # Over chord 4 — resolve down
    (10.8, 0.7, 2.0, -12.0),
    (11.6, 0.7, 9 / 5, -13.0),
    (12.4, 1.4, 3 / 2, -11.0),
]


def build_tube_palette_study() -> Score:
    f0 = 185.0  # F#3
    score = Score(f0_hz=f0, master_effects=DEFAULT_MASTER_EFFECTS)

    # One bed voice per section so the tube effect can be baked in at voice
    # construction time — matches the pattern used by ``filter_palette_study``
    # for per-section topology swaps.  Each bed voice only holds notes during
    # its 14-second window, so voices don't overlap across sections.
    for key, _label, tube_params in _TUBE_SECTIONS:
        score.add_voice(
            f"bed_{key}",
            synth_defaults=dict(_BED_BASE),
            effects=[EffectSpec("tube", dict(tube_params)), _reverb()],
            pan=-0.08,
            # -10 dB keeps the hot tube characters (hg2_drive, culture_starve)
            # from driving the reverb into intersample clipping while still
            # leaving the lead reference audible.
            mix_db=-10.0,
        )

    # Single reference lead voice, untouched tube-wise, present across every
    # section — a stable timbral anchor so the bed's tube colour is heard
    # against the same lead texture throughout.
    score.add_voice(
        "lead",
        synth_defaults=dict(_LEAD_BASE),
        effects=[_delay(), _reverb()],
        pan=0.12,
        mix_db=-12.0,
    )

    for section_idx, (key, _label, _tube_params) in enumerate(_TUBE_SECTIONS):
        section_start = section_idx * SECTION_DUR
        bed_voice = f"bed_{key}"

        # Bed: four chord changes, each held for ~3.5s, across the section.
        for chord_offset, voicing in _BED_VOICINGS:
            chord_start = section_start + chord_offset
            for partial in voicing:
                score.add_note(
                    bed_voice,
                    start=chord_start,
                    duration=_BED_CHORD_HOLD + 0.4,
                    partial=partial,
                    amp_db=-18.0,
                )

        # Lead: same phrase in every section for direct A/B comparison.
        for t, dur, partial, db in _LEAD_PHRASE:
            score.add_note(
                "lead",
                start=section_start + t,
                duration=dur,
                partial=partial,
                amp_db=db,
            )

    return score


PIECES: dict[str, PieceDefinition] = {
    "tube_palette_study": PieceDefinition(
        name="tube_palette_study",
        output_name="tube_palette_study",
        build_score=build_tube_palette_study,
        sections=tuple(
            PieceSection(
                label=label,
                start_seconds=i * SECTION_DUR,
                end_seconds=(i + 1) * SECTION_DUR,
            )
            for i, (_key, label, _params) in enumerate(_TUBE_SECTIONS)
        ),
        study=True,
    ),
}
