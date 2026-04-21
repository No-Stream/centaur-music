"""filter_palette_study — A/B tour of all 8 filter topologies.

Plays the same 8-bar phrase through each topology, back-to-back, so the
listener can hear the distinctive character of each voice:

  1. SVF (2-pole ZDF state-variable — neutral reference)
  2. Ladder (4-pole Moog Huovilainen — classic bass suck + growl)
  3. Sallen-Key (Diva-ish 12 dB/oct bite — CEM-3320-inspired)
  4. Cascade (Prophet-5 rev-2 / Juno — smooth 4-pole, no growl)
  5. SEM (Oberheim gentle 12 dB/oct — bass preserving, bloomy Q)
  6. Jupiter (Roland IR3109 4-pole OTA cascade — creamy, Juno-106 without HPF)
  7. K35 (Korg MS-20 — diode-clipped feedback, screaming snarl)
  8. Diode (TB-303 — 3-pole asymmetric, acid squelch)

Each section is 16 seconds of the same melodic + harmonic material:
a sustained chord bed with a moving lead that hits high Q at peak moments.
The filter envelope sweeps on each note so the resonance character reveals
itself through the filter's response curve, not just static tone.

Key in F# minor (185 Hz f0) — consistent with diva_study so A/B comparisons
across study pieces are fair.  No kick; this is a voice-character tour.
"""

from __future__ import annotations

from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score

SECTION_DUR = 16.0
N_SECTIONS = 8

# Topologies in presentation order.  Name, display label, base preset tuning
# dict (merged on top of a shared lead template).
_TOPOLOGY_SECTIONS: list[tuple[str, str, dict[str, object]]] = [
    (
        "svf",
        "SVF",
        {"filter_topology": "svf", "resonance_q": 3.5},
    ),
    (
        "ladder",
        "Ladder",
        {"filter_topology": "ladder", "resonance_q": 4.0, "bass_compensation": 0.4},
    ),
    (
        "sallen_key",
        "Sallen-Key",
        {"filter_topology": "sallen_key", "resonance_q": 4.5, "filter_drive": 0.35},
    ),
    (
        "cascade",
        "Cascade",
        {"filter_topology": "cascade", "resonance_q": 3.5, "filter_drive": 0.25},
    ),
    (
        "sem",
        "SEM",
        {"filter_topology": "sem", "resonance_q": 5.0, "filter_drive": 0.2},
    ),
    (
        "jupiter",
        "Jupiter",
        {
            "filter_topology": "jupiter",
            "resonance_q": 4.0,
            "hpf_cutoff_hz": 80.0,
            "filter_drive": 0.2,
        },
    ),
    (
        "k35",
        "K35/MS-20",
        {
            "filter_topology": "k35",
            "resonance_q": 6.0,
            "filter_drive": 0.55,
            "k35_feedback_asymmetry": 0.55,
        },
    ),
    (
        "diode",
        "Diode/TB-303",
        {
            "filter_topology": "diode",
            "filter_solver": "newton",
            "resonance_q": 7.0,
            "filter_drive": 0.45,
        },
    ),
]

# Shared lead params — everything except the topology-specific knobs.
_LEAD_BASE: dict[str, object] = {
    "engine": "polyblep",
    "waveform": "saw",
    "cutoff_hz": 900.0,
    "filter_env_amount": 2.4,
    "filter_env_decay": 0.35,
    "keytrack": 0.3,
    "osc_softness": 0.08,
    "attack": 0.006,
    "decay": 0.22,
    "sustain_level": 0.65,
    "release": 0.25,
}

# Sustained pad underneath every section so the filter character is heard
# against a consistent harmonic bed.  Keep the pad on a neutral topology so
# it doesn't compete with the lead's character.
_PAD_PRESET: dict[str, object] = {
    "engine": "polyblep",
    "waveform": "saw",
    "filter_topology": "svf",
    "cutoff_hz": 1_600.0,
    "resonance_q": 1.1,
    "osc2_level": 0.55,
    "osc2_detune_cents": 7.0,
    "attack": 0.25,
    "decay": 0.6,
    "sustain_level": 0.85,
    "release": 1.4,
}


def _reverb() -> EffectSpec:
    return EffectSpec(
        "reverb",
        {"room_size": 0.6, "damping": 0.4, "wet_level": 0.18},
    )


def _delay() -> EffectSpec:
    return EffectSpec(
        "delay",
        {"delay_seconds": 0.375, "feedback": 0.28, "mix": 0.2},
    )


# 8-bar lead phrase in 7-limit JI, expressed as (t_in_section, dur, partial, amp_db).
# ~2s per bar; dense enough to reveal filter character through multiple attacks.
_LEAD_PHRASE: list[tuple[float, float, float, float]] = [
    # Bars 1-2: rising motif through the fifth
    (0.0, 1.0, 1.0, -10.0),
    (1.0, 0.5, 5 / 4, -11.0),
    (1.5, 0.5, 3 / 2, -10.0),
    (2.0, 1.0, 7 / 4, -9.0),
    (3.0, 1.0, 2.0, -10.0),
    # Bars 3-4: descending response
    (4.0, 1.0, 3 / 2, -10.0),
    (5.0, 0.5, 5 / 4, -11.0),
    (5.5, 0.5, 4 / 3, -12.0),
    (6.0, 2.0, 1.0, -12.0),
    # Bars 5-6: climb and hold at the peak (where Q will scream)
    (8.0, 0.5, 3 / 2, -11.0),
    (8.5, 0.5, 7 / 4, -10.0),
    (9.0, 3.0, 9 / 4, -8.0),  # high held note — Q response is audible
    (12.0, 1.0, 2.0, -9.0),
    # Bars 7-8: resolve
    (13.0, 1.0, 7 / 4, -10.0),
    (14.0, 2.0, 3 / 2, -12.0),
]

# Chord voicings that hold across each section (one per 4 bars).
_PAD_VOICINGS: list[list[float]] = [
    [1.0, 5 / 4, 3 / 2, 7 / 4],
    [4 / 3, 5 / 3, 2.0, 9 / 4],
]


def build_filter_palette_study() -> Score:
    f0 = 185.0  # F#3
    score = Score(f0_hz=f0, master_effects=DEFAULT_MASTER_EFFECTS)

    # Shared pad voice (SVF, neutral).
    score.add_voice(
        "pad",
        synth_defaults=dict(_PAD_PRESET),
        effects=[_reverb()],
        pan=-0.1,
    )

    # One lead voice per topology; same synth_defaults template, each voice
    # overrides the topology-specific knobs.
    for topo_key, _label, overrides in _TOPOLOGY_SECTIONS:
        synth = dict(_LEAD_BASE)
        synth.update(overrides)
        score.add_voice(
            f"lead_{topo_key}",
            synth_defaults=synth,
            effects=[_delay(), _reverb()],
            pan=0.12,
        )

    for section_idx, (topo_key, _label, _ovr) in enumerate(_TOPOLOGY_SECTIONS):
        section_start = section_idx * SECTION_DUR

        # Pad bed: two 8-second chord changes per section.
        for chord_idx, voicing in enumerate(_PAD_VOICINGS):
            chord_start = section_start + chord_idx * (SECTION_DUR / 2)
            for partial in voicing:
                score.add_note(
                    "pad",
                    start=chord_start,
                    duration=SECTION_DUR / 2 + 0.5,
                    partial=partial,
                    amp_db=-22.0,
                )

        # Lead phrase on the topology-specific voice.
        voice_name = f"lead_{topo_key}"
        for t, dur, partial, db in _LEAD_PHRASE:
            score.add_note(
                voice_name,
                start=section_start + t,
                duration=dur,
                partial=partial,
                amp_db=db,
            )

    return score


PIECES: dict[str, PieceDefinition] = {
    "filter_palette_study": PieceDefinition(
        name="filter_palette_study",
        output_name="filter_palette_study",
        build_score=build_filter_palette_study,
        sections=tuple(
            PieceSection(
                label=label,
                start_seconds=i * SECTION_DUR,
                end_seconds=(i + 1) * SECTION_DUR,
            )
            for i, (_key, label, _ovr) in enumerate(_TOPOLOGY_SECTIONS)
        ),
        study=True,
    ),
}
