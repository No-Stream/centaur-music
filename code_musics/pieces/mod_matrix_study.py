"""Mod Matrix Study — demonstration of the per-connection modulation matrix.

A short 7-limit drone/lead sketch that exercises the Vital-style
modulation matrix across five flavors of connection:

* **Macro -> filter_morph** on the pad (rising from 0 -> 1 across the piece).
* **LFO -> cutoff_hz** on the lead at audio-rate via the per-sample
  ``param_profiles`` path, with a mildly concave power curve.
* **VelocitySource -> resonance_q** on the lead, unipolar so velocity
  only adds resonance (never subtracts) with a breakpoint curve for
  emphasis on the loudest notes.
* **ConstantSource stereo pan-split** on a doubled "twin" voice, one
  connection pulling pan negative, the other positive.
* **DriftAdapter(random_walk) -> osc2_detune_cents** on the pad for
  slow organic detune evolution without wiring a full humanize spec.

Key: F Lydian-ish drone (f0 = 174.614 Hz).  ~32 s.  Uses the default
master chain and a soft reverb send.
"""

from __future__ import annotations

from code_musics.automation import (
    AutomationSegment,
    AutomationSpec,
    AutomationTarget,
)
from code_musics.modulation import (
    ConstantSource,
    DriftAdapter,
    LFOSource,
    MacroSource,
    ModConnection,
    VelocitySource,
)
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS, SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import Score


def build_mod_matrix_study() -> Score:
    """Build the Mod Matrix Study score."""
    f0 = 174.614  # F3
    score = Score(f0_hz=f0, master_effects=DEFAULT_MASTER_EFFECTS)

    # Macro: filter_morph sweeps 0 -> 1 across the piece.
    score.add_macro(
        "brightness",
        default=0.0,
        automation=AutomationSpec(
            target=AutomationTarget(kind="control", name="mix"),
            segments=(
                AutomationSegment(
                    start=0.0,
                    end=30.0,
                    shape="linear",
                    start_value=0.0,
                    end_value=1.0,
                ),
            ),
            default_value=0.0,
        ),
    )

    # Pad: additive with DriftAdapter riding detune + macro modulating
    # filter_morph (on a polyblep layer via the "twin" voices below).
    # The pad itself is simple additive so the drift detune on osc2 is
    # modeled via the polyblep layer where osc2 exists.
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "saw",
            "cutoff_hz": 900.0,
            "resonance_q": 0.85,
            "filter_morph": 0.0,
            "osc2_level": 0.35,
            "osc2_detune_cents": 4.0,
            "attack": 0.9,
            "release": 1.2,
        },
        effects=[SOFT_REVERB_EFFECT],
        modulations=[
            ModConnection(
                source=MacroSource(name="brightness"),
                target=AutomationTarget(kind="synth", name="filter_morph"),
                amount=1.0,
                bipolar=False,
                mode="add",
                name="macro_filter_morph",
            ),
            ModConnection(
                source=DriftAdapter(
                    style="random_walk",
                    rate_hz=0.2,
                    smoothness=0.8,
                    seed=17,
                ),
                target=AutomationTarget(kind="synth", name="osc2_detune_cents"),
                amount=6.0,
                mode="add",
                name="drift_detune",
            ),
        ],
        pan=0.0,
    )

    # Lead: bright poly-blep with free-run LFO on cutoff (per-sample
    # param_profiles) and velocity -> resonance_q.
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "saw",
            "cutoff_hz": 1400.0,
            "resonance_q": 0.9,
            "osc2_level": 0.0,
            "attack": 0.02,
            "release": 0.6,
        },
        effects=[SOFT_REVERB_EFFECT],
        modulations=[
            ModConnection(
                source=LFOSource(rate_hz=0.7, waveshape="sine"),
                target=AutomationTarget(kind="synth", name="cutoff_hz"),
                amount=600.0,
                power=-4.0,  # mild convex: emphasis on the dips
                mode="add",
                name="lfo_cutoff",
            ),
            ModConnection(
                source=VelocitySource(velocity_scale=1.25),
                target=AutomationTarget(kind="synth", name="resonance_q"),
                amount=0.6,
                bipolar=False,
                breakpoints=((0.0, 0.0), (0.6, 0.15), (1.0, 1.0)),
                mode="add",
                name="velocity_resonance",
            ),
        ],
        pan=0.0,
    )

    # Twin voice + its stereo pan-split partner.  The two voices are
    # near-identical except for pan; each has a ConstantSource pan
    # connection with opposite sign so the matrix drives the split
    # deterministically.
    for voice_name, pan_amount in (("twin_left", -1.0), ("twin_right", 1.0)):
        score.add_voice(
            voice_name,
            synth_defaults={
                "engine": "polyblep",
                "waveform": "square",
                "cutoff_hz": 1100.0,
                "pulse_width": 0.42,
                "osc2_level": 0.0,
                "attack": 0.05,
                "release": 0.8,
            },
            effects=[SOFT_REVERB_EFFECT],
            modulations=[
                ModConnection(
                    source=ConstantSource(value=1.0),
                    target=AutomationTarget(kind="control", name="pan"),
                    amount=0.55 * pan_amount,
                    mode="add",
                    name=f"const_pan_{voice_name}",
                ),
            ],
        )

    # ---- Material: a slow F-lydian drone under a lead line. ----
    # Pad: sustained 1, 5/4, 3/2 triad that extends to 9/8 in section 2.
    for start, dur in [(0.0, 14.0), (14.0, 16.0)]:
        for partial, amp_db in [
            (0.5, -22.0),
            (1.0, -23.0),
            (5 / 4, -25.0),
            (3 / 2, -26.0),
        ]:
            score.add_note(
                "pad", start=start, duration=dur, partial=partial, amp_db=amp_db
            )
        if start >= 14.0:
            score.add_note(
                "pad", start=start, duration=dur, partial=9 / 8, amp_db=-27.0
            )

    # Lead line: slow motif + faster repetition in section 2.  Velocities
    # vary to exercise the VelocitySource -> resonance_q mapping.
    lead_motif = [
        (0.5, 2.5, 3 / 2, 1.05),
        (3.0, 1.5, 5 / 4, 0.9),
        (4.5, 2.0, 7 / 4, 1.15),
        (7.0, 2.5, 3 / 2, 0.95),
        (10.0, 1.5, 9 / 8, 1.08),
        (12.0, 1.8, 5 / 4, 1.0),
    ]
    for start, dur, partial, velocity in lead_motif:
        score.add_note(
            "lead",
            start=start,
            duration=dur,
            partial=partial,
            amp_db=-20.0,
            velocity=velocity,
        )

    # Section 2: same motif, denser, sitting under brighter pad.
    for start, dur, partial, velocity in [
        (14.5, 1.6, 3 / 2, 1.1),
        (16.5, 1.4, 7 / 4, 1.2),
        (18.2, 1.4, 2.0, 1.15),
        (20.0, 1.8, 5 / 4, 1.0),
        (22.4, 2.0, 3 / 2, 0.95),
        (25.0, 1.6, 9 / 8, 0.9),
        (27.0, 2.4, 5 / 4, 1.05),
    ]:
        score.add_note(
            "lead",
            start=start,
            duration=dur,
            partial=partial,
            amp_db=-19.0,
            velocity=velocity,
        )

    # Twin voices play a sparser, more rhythmic accompaniment to show
    # the stereo pan-split clearly.  Both voices hit the same notes at
    # the same time; the matrix alone determines L/R panning.
    twin_notes = [
        (2.0, 1.0, 1.0),
        (5.0, 1.0, 3 / 4),
        (8.0, 1.0, 5 / 4),
        (11.0, 1.0, 1.0),
        (15.0, 0.8, 3 / 2),
        (17.0, 0.8, 5 / 4),
        (19.0, 0.8, 1.0),
        (21.0, 0.8, 7 / 4),
        (24.0, 0.8, 5 / 4),
        (26.5, 0.8, 3 / 2),
        (28.5, 1.2, 1.0),
    ]
    for voice_name in ("twin_left", "twin_right"):
        for start, dur, partial in twin_notes:
            score.add_note(
                voice_name,
                start=start,
                duration=dur,
                partial=partial,
                amp_db=-26.0,
                velocity=1.0,
            )

    return score


PIECES: dict[str, PieceDefinition] = {
    "mod_matrix_study": PieceDefinition(
        name="mod_matrix_study",
        output_name="mod_matrix_study_01",
        build_score=build_mod_matrix_study,
        sections=(
            PieceSection(label="Awakening", start_seconds=0.0, end_seconds=14.0),
            PieceSection(label="Brightening", start_seconds=14.0, end_seconds=30.0),
        ),
        study=True,
    ),
}
