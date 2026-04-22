"""scanned_garden — slow unfolding Verplank-style mass-spring garden.

A three-section pad/lead meditation built around ``synth_voice``'s
Verplank-style scanned synthesis oscillator (``osc_type="scanned"``).  The
scanner reads displacement from a ring of coupled masses while the network
evolves mechanically at its own rate, producing a waveform that breathes
and morphs without external modulation.

Sections (target wall-clock times):

  1. Glass swarm alone     (0:00 - 0:25)  A crystalline ``scanned_glass_swarm``
     pad establishes the 7-limit JI frame at F.
  2. Singing loop enters   (0:25 - 0:55)  A ``scanned_singing_loop`` lead
     traces a slow melodic line with vibrato; pad continues.
  3. Taut wire accents     (0:55 - 1:20)  A ``scanned_taut_wire`` voice
     answers with brief percussive accents.  Lead & pad carry to close.

Key: F major, 7-limit JI (tonic 174.614 Hz = F3).  ~80 s.
"""

from __future__ import annotations

from code_musics.automation import (
    AutomationSegment,
    AutomationSpec,
    AutomationTarget,
)
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS, SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import Score, SendBusSpec, VoiceSend

F0_HZ = 174.614  # F3
TOTAL_DUR = 80.0

SECTION_1_END = 25.0
SECTION_2_END = 55.0


def build_scanned_garden() -> Score:
    """Build the Scanned Garden score."""
    score = Score(
        f0_hz=F0_HZ,
        master_effects=DEFAULT_MASTER_EFFECTS,
        send_buses=[
            SendBusSpec(
                name="hall",
                effects=[SOFT_REVERB_EFFECT],
                return_db=0.0,
            )
        ],
    )

    # ------------------------------------------------------------------
    # Pad: scanned_glass_swarm — crystalline, self-evolving wash.
    # Sits wide with a slow filter-morph ride to open the piece.
    # ------------------------------------------------------------------
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "scanned_glass_swarm",
            "filter_morph": 0.0,
            "attack": 2.0,
            "release": 3.5,
        },
        sends=[VoiceSend(target="hall", send_db=-6.0)],
        pan=-0.12,
        mix_db=-4.0,
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="filter_morph"),
                segments=(
                    AutomationSegment(
                        start=0.0,
                        end=TOTAL_DUR,
                        shape="linear",
                        start_value=0.0,
                        end_value=0.35,
                    ),
                ),
                default_value=0.0,
            ),
        ],
    )

    # ------------------------------------------------------------------
    # Lead: scanned_singing_loop — vocal, harmonic-rich, slow morph.
    # Vibrato on sustained notes keeps it alive.
    # ------------------------------------------------------------------
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "scanned_singing_loop",
            "attack": 0.3,
            "release": 1.5,
        },
        sends=[VoiceSend(target="hall", send_db=-4.0)],
        pan=0.18,
        mix_db=-6.0,
    )

    # ------------------------------------------------------------------
    # Accent: scanned_taut_wire — percussive, harmonic-rich snaps.
    # Short, bright, more centered in the stereo field.
    # ------------------------------------------------------------------
    score.add_voice(
        "wire",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "scanned_taut_wire",
            "attack": 0.003,
            "release": 0.6,
        },
        sends=[VoiceSend(target="hall", send_db=-3.0)],
        pan=0.0,
        mix_db=-10.0,
    )

    # ==================================================================
    # Section 1: pad alone (0 - 25s).
    # 7-limit chord: 1/1, 5/4, 3/2, 7/4 stacked and layered with gentle
    # octave for weight.
    # ==================================================================
    for partial, amp_db, offset, dur in [
        (0.5, -20.0, 0.0, 25.0),  # F2 sub root
        (1.0, -18.0, 0.0, 25.0),  # F3 root
        (5 / 4, -21.0, 1.5, 23.5),  # A3 (JI major 3rd, delayed entry)
        (3 / 2, -20.0, 3.0, 22.0),  # C4
        (7 / 4, -24.0, 6.0, 19.0),  # Eb4 (septimal 7th, last to arrive)
    ]:
        score.add_note(
            "pad", start=offset, duration=dur, partial=partial, amp_db=amp_db
        )

    # ==================================================================
    # Section 2: lead enters, slow melodic line across 25-55s.
    # Phrase traces 7-limit intervals over the pad.  Vibrato depth
    # ramps up slightly for later, more expressive notes.
    # ==================================================================
    lead_phrase = [
        # (start, duration, partial, amp_db, vibrato_depth)
        (26.0, 5.0, 3 / 2, -16.0, 0.003),  # open on the fifth
        (31.5, 3.0, 7 / 4, -15.0, 0.004),  # step up to septimal 7
        (35.0, 4.5, 2.0, -14.0, 0.0045),  # reach octave
        (40.0, 3.5, 5 / 4 * 2, -15.0, 0.005),  # JI major 3rd up octave
        (44.0, 5.0, 3 / 2, -15.0, 0.005),  # settle back to 3/2
        (49.5, 5.0, 1.0, -17.0, 0.004),  # rest on root
    ]
    for start, dur, partial, amp_db, vibrato_depth in lead_phrase:
        score.add_note(
            "lead",
            start=start,
            duration=dur,
            partial=partial,
            amp_db=amp_db,
            pitch_motion=PitchMotionSpec.vibrato(
                depth_ratio=vibrato_depth,
                rate_hz=4.5,
            ),
        )

    # Pad chord continues with a gentle reharmonization — drop to 4/3
    # root context for mid-section color.
    for partial, amp_db, offset, dur in [
        (0.5, -20.0, 25.0, 30.0),
        (1.0, -19.0, 25.0, 30.0),
        (4 / 3, -22.0, 28.0, 27.0),  # add 4/3 — opens voicing
        (5 / 4, -21.0, 25.0, 30.0),
        (3 / 2, -20.0, 25.0, 30.0),
        (7 / 4, -23.0, 32.0, 23.0),
    ]:
        score.add_note(
            "pad", start=offset, duration=dur, partial=partial, amp_db=amp_db
        )

    # ==================================================================
    # Section 3: taut wire accents answer the lead (55-80s).
    # Sparse percussive gestures — each one plucks the scanned network
    # and lets it decay naturally.
    # ==================================================================
    wire_accents = [
        # (start, duration, partial, amp_db)
        (55.5, 0.9, 2.0, -12.0),
        (56.8, 0.7, 5 / 2, -13.0),
        (58.5, 1.0, 3.0, -12.0),
        (61.0, 0.8, 7 / 4 * 2, -13.0),
        (63.5, 1.2, 3 / 2 * 2, -12.0),
        (66.5, 0.9, 2.0, -13.0),
        (68.8, 0.6, 5 / 2, -14.0),
        (70.5, 1.4, 3.0, -13.0),
    ]
    for start, dur, partial, amp_db in wire_accents:
        score.add_note(
            "wire", start=start, duration=dur, partial=partial, amp_db=amp_db
        )

    # Lead closes with two held tones, falling into the root.
    score.add_note(
        "lead",
        start=56.0,
        duration=8.0,
        partial=5 / 4,
        amp_db=-16.0,
        pitch_motion=PitchMotionSpec.vibrato(depth_ratio=0.004, rate_hz=4.0),
    )
    score.add_note(
        "lead",
        start=65.0,
        duration=7.5,
        partial=4 / 3,
        amp_db=-16.0,
        pitch_motion=PitchMotionSpec.vibrato(depth_ratio=0.0045, rate_hz=4.2),
    )
    score.add_note(
        "lead",
        start=73.0,
        duration=7.0,
        partial=1.0,
        amp_db=-18.0,
        pitch_motion=PitchMotionSpec.vibrato(depth_ratio=0.003, rate_hz=3.5),
    )

    # Pad tail — carry 1/1 + 3/2 through to the end for a settled close.
    for partial, amp_db in [(0.5, -22.0), (1.0, -20.0), (3 / 2, -22.0)]:
        score.add_note(
            "pad", start=55.0, duration=TOTAL_DUR - 55.0, partial=partial, amp_db=amp_db
        )

    return score


PIECES: dict[str, PieceDefinition] = {
    "scanned_garden": PieceDefinition(
        name="scanned_garden",
        output_name="scanned_garden_01",
        build_score=build_scanned_garden,
        sections=(
            PieceSection(
                label="Glass Swarm",
                start_seconds=0.0,
                end_seconds=SECTION_1_END,
            ),
            PieceSection(
                label="Singing Loop",
                start_seconds=SECTION_1_END,
                end_seconds=SECTION_2_END,
            ),
            PieceSection(
                label="Taut Wire",
                start_seconds=SECTION_2_END,
                end_seconds=TOTAL_DUR,
            ),
        ),
    ),
}
