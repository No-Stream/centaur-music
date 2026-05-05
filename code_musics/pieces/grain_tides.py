"""grain_tides — granular synthesis over a slow JI chord progression.

Three granular voices pulled across a slow harmonic motion.  Grain clouds
breathe at their own rate while a chord underneath shifts through 7-limit
JI changes — Fennesz territory, where source and process both evolve.

Voices:

  * ``cloud`` — ``grain_breathing_cloud`` main pad, dense hann grains over
    JI-quantized partials.  Fills the stereo middle.
  * ``shimmer`` — ``grain_shimmer_dust`` upper layer, tight grains pitched
    an octave up.  Pushed slightly right.
  * ``freeze`` — ``grain_frozen_time`` cameo, suspended time accent in
    section 2.  Pushed slightly left.

Section structure (target wall-clock):

  1. Cloud alone on Imaj      (0:00 - 0:20)  Establishes F tonic; shimmer
     fades in near the end.
  2. Chord shifts to IV        (0:20 - 0:45)  Freeze cameo enters; filter
     cutoff on the cloud rises into the section boundary.
  3. Chord pivots on V7        (0:45 - 1:10)  Full texture, densest grain
     density.
  4. Return to I, dissolve     (1:10 - 1:30)  Filter closes, shimmer fades,
     cloud lingers on tonic.

Key: F major, 7-limit JI (tonic 174.614 Hz).  ~90 s.
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
TOTAL_DUR = 90.0

S1_END = 20.0
S2_END = 45.0
S3_END = 70.0


def build_grain_tides() -> Score:
    """Build the Grain Tides score."""
    score = Score(
        f0_hz=F0_HZ,
        master_effects=DEFAULT_MASTER_EFFECTS,
        send_buses=[
            SendBusSpec(
                name="hall",
                effects=[SOFT_REVERB_EFFECT],
                return_db=-2.0,
            )
        ],
    )

    # ------------------------------------------------------------------
    # Cloud: dense grain cloud — main pad layer.
    # Filter morph rides from 0.0 -> 0.4 across the piece to open up
    # the texture through the middle sections.
    # ------------------------------------------------------------------
    score.add_voice(
        "cloud",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "grain_breathing_cloud",
            "filter_morph": 0.0,
            "attack": 1.2,
            "release": 2.2,
        },
        sends=[VoiceSend(target="hall", send_db=-4.0)],
        pan=0.0,
        mix_db=-4.0,
        automation=[
            # Open up the filter morph across the piece for progressive
            # brightening into section 3, then close back.
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="filter_morph"),
                segments=(
                    AutomationSegment(
                        start=0.0,
                        end=S3_END,
                        shape="linear",
                        start_value=0.0,
                        end_value=0.4,
                    ),
                    AutomationSegment(
                        start=S3_END,
                        end=TOTAL_DUR,
                        shape="linear",
                        start_value=0.4,
                        end_value=0.1,
                    ),
                ),
                default_value=0.0,
            ),
            # Cross-section mix_db ride — softer intro + outro, fuller middle.
            AutomationSpec(
                target=AutomationTarget(kind="control", name="mix_db"),
                segments=(
                    AutomationSegment(
                        start=0.0,
                        end=S1_END,
                        shape="linear",
                        start_value=-8.0,
                        end_value=-4.0,
                    ),
                    AutomationSegment(
                        start=S1_END,
                        end=S3_END,
                        shape="hold",
                        value=-4.0,
                    ),
                    AutomationSegment(
                        start=S3_END,
                        end=TOTAL_DUR,
                        shape="linear",
                        start_value=-4.0,
                        end_value=-9.0,
                    ),
                ),
                default_value=-4.0,
                mode="replace",
            ),
        ],
    )

    # ------------------------------------------------------------------
    # Shimmer: upper-register granular.  Fades in near end of S1 and
    # carries the high air through S2-S3.
    # ------------------------------------------------------------------
    score.add_voice(
        "shimmer",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "grain_shimmer_dust",
            "attack": 2.0,
            "release": 2.5,
        },
        sends=[VoiceSend(target="hall", send_db=-3.0)],
        pan=0.25,
        mix_db=-10.0,
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="control", name="mix_db"),
                segments=(
                    # Fade in slowly from -inf-ish to full by end of S1.
                    AutomationSegment(
                        start=0.0,
                        end=S1_END,
                        shape="linear",
                        start_value=-24.0,
                        end_value=-10.0,
                    ),
                    AutomationSegment(
                        start=S1_END,
                        end=S3_END,
                        shape="hold",
                        value=-8.0,
                    ),
                    # Fade down across S4.
                    AutomationSegment(
                        start=S3_END,
                        end=TOTAL_DUR,
                        shape="linear",
                        start_value=-8.0,
                        end_value=-20.0,
                    ),
                ),
                default_value=-10.0,
                mode="replace",
            ),
        ],
    )

    # ------------------------------------------------------------------
    # Freeze: time-suspended cameo — only active in S2.
    # Cameo voice pushed left to contrast with shimmer on the right.
    # ------------------------------------------------------------------
    score.add_voice(
        "freeze",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "grain_frozen_time",
            "attack": 3.0,
            "release": 4.0,
        },
        sends=[VoiceSend(target="hall", send_db=-2.0)],
        pan=-0.28,
        mix_db=-10.0,
    )

    # ==================================================================
    # Chord progression across the four sections.
    # Each chord is placed as sustained notes on all three grain voices
    # for the duration of the section it belongs to.
    # ==================================================================
    # S1: Imaj — 1/1, 5/4, 3/2  (F major triad, JI)
    s1_chord = [(1.0, -18.0), (5 / 4, -20.0), (3 / 2, -20.0)]
    # S2: IV — 4/3, 5/3, 2.0  (Bb major triad, JI)
    s2_chord = [(4 / 3, -18.0), (5 / 3, -20.0), (2.0, -20.0)]
    # S3: V7 — 3/2, 15/8, 9/4, 7/4*3/2  (C7, septimal seventh)
    s3_chord = [
        (3 / 2, -17.0),
        (15 / 8, -19.0),
        (9 / 4, -20.0),
        (7 / 4 * 3 / 2, -21.0),  # septimal 7 of V
    ]
    # S4: Imaj resolution, more sparse
    s4_chord = [(1.0, -20.0), (5 / 4, -22.0), (3 / 2, -23.0)]

    sections = [
        (0.0, S1_END, s1_chord),
        (S1_END, S2_END, s2_chord),
        (S2_END, S3_END, s3_chord),
        (S3_END, TOTAL_DUR, s4_chord),
    ]
    for start, end, chord in sections:
        dur = end - start
        for partial, amp_db in chord:
            # Cloud sustains the full chord.
            score.add_note(
                "cloud", start=start, duration=dur, partial=partial, amp_db=amp_db
            )
            # Shimmer plays the top two chord tones, up an octave.
        # Only the top two chord tones get the shimmer layer (up octave).
        for partial, amp_db in chord[-2:]:
            score.add_note(
                "shimmer",
                start=start,
                duration=dur,
                partial=partial * 2.0,
                amp_db=amp_db + 2.0,  # slightly softer
            )

    # ------------------------------------------------------------------
    # Freeze cameo: two long notes inside S2 only.  The frozen time
    # windowing creates a sustained texture even from these held pitches.
    # ------------------------------------------------------------------
    score.add_note(
        "freeze",
        start=S1_END + 2.0,
        duration=16.0,
        partial=4 / 3,
        amp_db=-16.0,
        pitch_motion=PitchMotionSpec.ratio_glide(
            start_ratio=1.0,
            end_ratio=3 / 2 / (4 / 3),  # glide up to 3/2 (a fifth) over the note
        ),
    )
    score.add_note(
        "freeze",
        start=S1_END + 8.0,
        duration=14.0,
        partial=5 / 3,
        amp_db=-17.0,
    )

    # ------------------------------------------------------------------
    # Low bass reinforcement on the cloud — sub presence on each chord
    # root an octave down.  Gives the piece a grounded low end.
    # ------------------------------------------------------------------
    sub_by_section = [
        (0.0, S1_END, 0.5),
        (S1_END, S2_END, 2 / 3),
        (S2_END, S3_END, 3 / 4),
        (S3_END, TOTAL_DUR, 0.5),
    ]
    for start, end, root_ratio in sub_by_section:
        score.add_note(
            "cloud",
            start=start,
            duration=end - start,
            partial=root_ratio,
            amp_db=-20.0,
        )

    return score


PIECES: dict[str, PieceDefinition] = {
    "grain_tides": PieceDefinition(
        name="grain_tides",
        output_name="grain_tides_01",
        build_score=build_grain_tides,
        sections=(
            PieceSection(label="Imaj", start_seconds=0.0, end_seconds=S1_END),
            PieceSection(label="IV", start_seconds=S1_END, end_seconds=S2_END),
            PieceSection(label="V7", start_seconds=S2_END, end_seconds=S3_END),
            PieceSection(label="Return", start_seconds=S3_END, end_seconds=TOTAL_DUR),
        ),
    ),
}
