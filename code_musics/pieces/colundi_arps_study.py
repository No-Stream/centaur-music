"""Colundi Arps Study — Four Tet-style Colundi JI song.

A ~90s song in the Four Tet / Colundi world.  Hook-based form built
around a hand-crafted 2-bar phrase: the "question" bar ascends from R3
through P5 / h7 to an octave-leap R4 (with s6 as a single passing
grace); the "answer" bar descends back to R3, with N2 as a narrow-
second suspension before the tonic lands.  Each hook bar has exactly
one framed spicy tone — placed where the phrase wants it, prepared
and resolved by consonances — so Colundi color is audible but never
chaotic.  Variations are hand-picked ornaments per section, not cell
permutations.

Tuning: Colundi 7-note JI (1/1, 11/10, 19/16, 4/3, 3/2, 49/30, 7/4, 2/1)
built on f0 = 55 Hz (A1), matching ``colundi_sequence.py``.  The missing
third gives the floating, unresolved character; 11/10 / 19/16 / 49/30 are
the "spicy" tones used as ornaments, 4/3 / 3/2 / 7/4 are the stable ones.

BPM = 96.  1 bar = 2.5 s.  36 bars ≈ 90 s.

Song form:
  bars  1– 8   Intro            pad drone (home chord)
  bars  9–14   Verse            hook iterates 3x, quiet; kick anchors at bar 13
  bar  15      Verse tail       hook bar 1 only (bar 2 silent — breath)
  bar  16      Pre-chorus drop  accelerating falling figure; bell pickup at 16.15
  bars 17–20   Chorus phrase 1  hook + brightness grace; bell phrases 1A/1B
                                (1A: soaring entry → P4 landing; 1B: twist,
                                 N2_4 narrow-2nd grace, octave drop to R3)
  bars 21–24   Chorus phrase 2  hook; bell phrases 2A/2B
                                (2A: variant answer → N2 held suspension → R3;
                                 2B: rising cadential tag, syncopated R3 at 24.2.5)
  bars 25–28   Bridge           sustained spicy drones (P4 / N2 / m3 / s6); no 16ths.
                                pad shifts focus to P4 (louder than chorus 1 P4)
  bars 29–32   Chorus return    hook + climax cluster; bell phrase 3 with
                                h7_4 peak at 30.1 (only note > R4 in piece)
  bars 33–35   Outro            hook skeleton thinning to a single R3 whisper
"""

from __future__ import annotations

from code_musics.automation import AutomationSegment, AutomationSpec, AutomationTarget
from code_musics.drum_helpers import add_drum_voice, setup_drum_bus
from code_musics.generative import euclidean_pattern
from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    VelocityHumanizeSpec,
)
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import EffectSpec, Score, SendBusSpec, VoiceSend
from code_musics.spectra import ratio_spectrum
from code_musics.synth import BRICASTI_IR_DIR

# ---------------------------------------------------------------------------
# Timing grid
# ---------------------------------------------------------------------------

BPM: float = 96.0
BEAT: float = 60.0 / BPM
BAR: float = 4.0 * BEAT
S16: float = BEAT / 4.0
DOTTED_EIGHTH: float = 3.0 * S16

F0: float = 55.0  # A1
TOTAL_BARS: int = 36


def _pos(bar: int, beat: int = 1, n16: int = 0) -> float:
    """Absolute time at bar:beat:sixteenth (1-indexed)."""
    return (bar - 1) * BAR + (beat - 1) * BEAT + n16 * S16


# Section boundaries in seconds (S1 end = start of bar 9, etc.)
S1_END: float = _pos(9)  # 20.0
S2_END: float = _pos(17)  # 40.0
S3_END: float = _pos(29)  # 70.0
TOTAL_DUR: float = _pos(TOTAL_BARS + 1)  # 90.0

# ---------------------------------------------------------------------------
# Colundi scale degrees as partials of f0 = 55 Hz.
#
# Scale:  R     N2      m3      P4    P5    s6      h7    8ve
# Ratio:  1/1   11/10   19/16   4/3   3/2   49/30   7/4   2/1
#
# Melody sits around octaves 3-4 (220-880 Hz).  For the arp we want
# A3-A4 roughly, so partial 4x (octave 3) is the main home.
# ---------------------------------------------------------------------------

# Octave 2 (110-220 Hz) — pad voicing anchors
R2: float = 2.0
P4_2: float = 8 / 3
P5_2: float = 3.0
h7_2: float = 7 / 2

# Octave 3 (220-440 Hz) — arp home + countermelody
R3: float = 4.0
N2_3: float = 22 / 5  # 11/10 * 4
m3_3: float = 19 / 4  # 19/16 * 4
P4_3: float = 16 / 3
P5_3: float = 6.0
s6_3: float = 98 / 15  # 49/30 * 4
h7_3: float = 7.0

# Octave 4 (440-880 Hz) — upper sparkle
R4: float = 8.0
N2_4: float = 44 / 5
P4_4: float = 32 / 3
P5_4: float = 12.0
h7_4: float = 14.0


# ---------------------------------------------------------------------------
# Colundi-aligned additive spectrum for the pad — overtone content mirrors
# the scale itself so the pad timbre resonates sympathetically with the
# harmonic language.  Ratios follow colundi_sequence._COLUNDI_PARTIALS.
# ---------------------------------------------------------------------------

_COLUNDI_PARTIALS = ratio_spectrum(
    ratios=[1.0, 11 / 10, 19 / 16, 4 / 3, 3 / 2, 49 / 30, 7 / 4, 2.0, 3.0, 7 / 2],
    amps=[1.0, 0.25, 0.18, 0.45, 0.55, 0.12, 0.35, 0.30, 0.15, 0.10],
)


# ---------------------------------------------------------------------------
# Hook onset tables
#
# Each row is (step, partial, velocity, amp_off_db, duration_sixteenths).
# Step is 0..15 within a bar (4 beats × 4 sixteenths).  amp_off is added
# to the section's amp_base, so the hook's internal dynamic contour stays
# consistent across verse / chorus / return while overall level rides the
# song arc.
#
# Position strengths on a 16th grid:
#   step 0 / 4 / 8 / 12  = strong (downbeat of each beat)
#   step 2 / 6 / 10 / 14 = medium (mid-beat)
#   everything else      = weak
#
# HOOK BAR 1 — "question", ascending.  R3 downbeat, climbs through P5 → h7
# with one passing spice (s6 at step 5, between P5 and h7 in pitch too), a
# held h7 beat, the SIGNATURE octave leap to R4 at the "a of 3", then a
# descending h7 / P5 pickup into bar 2.  7 onsets at [0, 3, 5, 6, 11, 13, 15].
# ---------------------------------------------------------------------------

_HOOK_BAR1: list[tuple[int, float, float, float, float]] = [
    (0, R3, 0.92, 0.0, 2.9),  # downbeat, long
    (3, P5_3, 0.62, -3.0, 1.9),  # syncopated
    (5, s6_3, 0.55, -5.5, 0.95),  # spice — passing P5 → h7
    (6, h7_3, 0.78, -1.5, 4.7),  # held, tension before leap
    (11, R4, 0.88, -0.5, 1.9),  # SIGNATURE octave leap peak
    (13, h7_3, 0.68, -2.5, 1.9),  # descent begins
    (15, P5_3, 0.56, -4.0, 1.0),  # pickup into bar 2
]

# HOOK BAR 2 — "answer", descending.  R4 downbeat mirrors bar 1's R3; drops
# through h7 / P5, bounces up to h7 briefly at the "a of 3" (rhythmic echo
# of bar 1's leap position), then descends P5 → N2 → R3.  N2 at step 14 is
# the narrow-second suspension — 165¢ above tonic — resolving to R3 on the
# final 16th.  7 onsets at [0, 3, 6, 11, 13, 14, 15].
_HOOK_BAR2: list[tuple[int, float, float, float, float]] = [
    (0, R4, 0.88, 0.0, 2.9),  # answer downbeat
    (3, h7_3, 0.62, -3.0, 2.9),  # descent
    (6, P5_3, 0.78, -1.5, 4.7),  # held, mirrors bar 1 step 6 h7
    (11, h7_3, 0.72, -2.0, 1.9),  # rhythmic echo of the leap position
    (13, P5_3, 0.62, -3.5, 0.95),
    (14, N2_3, 0.54, -5.5, 0.95),  # spice — narrow-2nd suspension
    (15, R3, 0.82, -1.0, 1.8),  # RESOLUTION landing
]

_VIBRATO_MIN_16THS: float = 2.5


def _vibrato_for_dur(dur_16ths: float) -> PitchMotionSpec | None:
    """Gentle vibrato on notes held long enough to sing."""
    if dur_16ths >= _VIBRATO_MIN_16THS:
        return PitchMotionSpec.vibrato(depth_ratio=0.0035, rate_hz=5.0)
    return None


def _step_to_beat_sub(step: int) -> tuple[int, int]:
    """Convert 0..15 step index to (beat, sub) for _pos()."""
    return (1 + step // 4, step % 4)


# ---------------------------------------------------------------------------
# Wood-block perc pitches drawn from the *strange* Colundi scale degrees —
# the pitched percussion broadcasts the xenharmonic fingerprint of the
# piece at transient-readable speed.
# ---------------------------------------------------------------------------

_PERC_PITCHES: list[float] = [
    N2_3 * F0,  # 11/10 area (~242 Hz)
    m3_3 * F0,  # 19/16 area (~261 Hz)
    s6_3 * F0,  # 49/30 area (~359 Hz)
    P4_3 * F0,  # stable anchor (~293 Hz)
]


def _make_hall_bus() -> SendBusSpec:
    """Soft dark reverb send bus."""
    if BRICASTI_IR_DIR.exists():
        effects = [
            EffectSpec(
                "bricasti",
                {
                    "ir_name": "1 Halls 07 Large & Dark",
                    "wet": 1.0,
                    "lowpass_hz": 5000.0,
                    "highpass_hz": 200.0,
                },
            ),
        ]
    else:
        effects = [
            EffectSpec(
                "reverb",
                {"room_size": 0.82, "damping": 0.55, "wet_level": 1.0},
            ),
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 200.0,
                            "slope_db_per_oct": 12,
                        },
                        {
                            "kind": "lowpass",
                            "cutoff_hz": 5000.0,
                            "slope_db_per_oct": 12,
                        },
                    ]
                },
            ),
        ]
    return SendBusSpec(name="hall", effects=effects, return_db=-2.0)


def _make_bell_delay_bus() -> SendBusSpec:
    """Dotted-8th delay into a reverb wash — for the bell only.

    Washy otherworldly tail.  HP roll-off keeps rumble out of the feedback
    path; no aggressive LP so the bell keeps its shimmering character.
    """
    return SendBusSpec(
        name="bell_delay",
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 220.0,
                            "slope_db_per_oct": 12,
                        },
                    ]
                },
            ),
            EffectSpec(
                "delay",
                {
                    "delay_seconds": DOTTED_EIGHTH,
                    "feedback": 0.5,
                    "mix": 1.0,
                },
            ),
            EffectSpec(
                "reverb",
                {"room_size": 0.88, "damping": 0.55, "wet_level": 0.5},
            ),
        ],
        return_db=-6.0,
    )


# ---------------------------------------------------------------------------
# build_score
# ---------------------------------------------------------------------------


def build_score() -> Score:
    """Build the Colundi Arps Study score."""
    score = Score(
        f0_hz=F0,
        master_effects=DEFAULT_MASTER_EFFECTS,
        send_buses=[_make_hall_bus(), _make_bell_delay_bus()],
        timing_humanize=TimingHumanizeSpec(preset="chamber"),
    )

    # Shared slow drift bus — tonal voices subscribe so the ensemble
    # breathes together.  Slow, subtle (5 cents peak, 0.12 Hz).
    score.add_drift_bus(
        "colundi_drift",
        rate_hz=0.12,
        depth_cents=5.0,
        seed=17,
    )

    drum_bus = setup_drum_bus(score, style="light", return_db=0.0)

    # ------------------------------------------------------------------
    # PAD — additive_pad_through_ladder overridden with Colundi partials
    # ------------------------------------------------------------------
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "additive_pad_through_ladder",
            "partials_partials": _COLUNDI_PARTIALS,
            "partials_n_harmonics": 10,  # ignored when partials_partials set
            "filter_cutoff_hz": 1300.0,
            "attack": 2.5,
            "release": 3.5,
            "brightness_tilt": -0.1,
        },
        sends=[VoiceSend(target="hall", send_db=-4.0)],
        pan=0.0,
        mix_db=-6.0,
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        drift_bus="colundi_drift",
        drift_bus_correlation=0.65,
        automation=[
            # Brightness tilt opens up toward the bloom, settles darker at end.
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="brightness_tilt"),
                segments=(
                    AutomationSegment(
                        start=0.0,
                        end=S2_END,
                        shape="linear",
                        start_value=-0.1,
                        end_value=0.0,
                    ),
                    AutomationSegment(
                        start=S2_END,
                        end=S3_END,
                        shape="linear",
                        start_value=0.0,
                        end_value=0.1,
                    ),
                    AutomationSegment(
                        start=S3_END,
                        end=TOTAL_DUR,
                        shape="linear",
                        start_value=0.1,
                        end_value=-0.05,
                    ),
                ),
                default_value=-0.1,
            ),
        ],
    )

    # ------------------------------------------------------------------
    # ARP — fm_bell_over_supersaw with dotted-8th delay.
    # Automation rides filter cutoff across the piece.
    # ------------------------------------------------------------------
    score.add_voice(
        "arp",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "fm_bell_over_supersaw",
            "filter_cutoff_hz": 1800.0,
            "attack": 0.015,
            "release": 0.45,
        },
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 180.0,
                            "slope_db_per_oct": 12,
                        },
                        {
                            "kind": "lowpass",
                            "cutoff_hz": 7500.0,
                            "slope_db_per_oct": 12,
                        },
                    ]
                },
            ),
            # Preamp at drive=0.1 measured as a no-op (effect_mostly_inactive);
            # rely on the shared master-bus preamp warmth instead.
            EffectSpec(
                "delay",
                {
                    "delay_seconds": DOTTED_EIGHTH,
                    "feedback": 0.32,
                    "mix": 0.22,
                },
            ),
        ],
        sends=[VoiceSend(target="hall", send_db=-6.0)],
        pan=-0.05,
        mix_db=-3.0,
        envelope_humanize=EnvelopeHumanizeSpec(preset="loose_pluck"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        drift_bus="colundi_drift",
        drift_bus_correlation=0.65,
        automation=[
            # Filter cutoff opens through S2-S3, closes slightly in S4.
            # Use exp shape for perceptually linear frequency motion.
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="cutoff_hz"),
                segments=(
                    AutomationSegment(
                        start=0.0,
                        end=S1_END,
                        shape="hold",
                        value=1400.0,
                    ),
                    AutomationSegment(
                        start=S1_END,
                        end=S3_END,
                        shape="exp",
                        start_value=1400.0,
                        end_value=2800.0,
                    ),
                    AutomationSegment(
                        start=S3_END,
                        end=TOTAL_DUR,
                        shape="exp",
                        start_value=2800.0,
                        end_value=1800.0,
                    ),
                ),
                default_value=1400.0,
            ),
            # Slow pan sway across the piece for Four Tet spatial breath.
            AutomationSpec(
                target=AutomationTarget(kind="control", name="pan"),
                segments=(
                    AutomationSegment(
                        start=0.0,
                        end=S2_END,
                        shape="linear",
                        start_value=-0.05,
                        end_value=0.18,
                    ),
                    AutomationSegment(
                        start=S2_END,
                        end=S3_END,
                        shape="linear",
                        start_value=0.18,
                        end_value=-0.15,
                    ),
                    AutomationSegment(
                        start=S3_END,
                        end=TOTAL_DUR,
                        shape="linear",
                        start_value=-0.15,
                        end_value=0.08,
                    ),
                ),
                default_value=-0.05,
                mode="replace",
            ),
        ],
    )

    # ------------------------------------------------------------------
    # BELL — sparse FM countermelody, only in S3, with washy delay send.
    # ------------------------------------------------------------------
    score.add_voice(
        "bell",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "two_op_bell",
            "fm_ratio": 1.75,  # 7/4 septimal — matches Colundi h7
            "fm_index": 1.8,
            "fm_index_decay": 1.6,
            "attack": 0.01,
            "release": 2.2,
        },
        sends=[
            VoiceSend(target="hall", send_db=-2.0),
            VoiceSend(target="bell_delay", send_db=-6.0),
        ],
        pan=0.18,
        mix_db=-8.0,
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        drift_bus="colundi_drift",
        drift_bus_correlation=0.65,
    )

    # ------------------------------------------------------------------
    # KICK — mellow 808_house with softened punch/exciter
    # ------------------------------------------------------------------
    add_drum_voice(
        score,
        "kick",
        engine="drum_voice",
        preset="808_house",
        drum_bus=drum_bus,
        send_db=-2.0,
        synth_overrides={
            "tone_punch": 0.15,
            "tone_second_harmonic": 0.06,
            "exciter_level": 0.05,
        },
        mix_db=-4.0,
    )

    # ------------------------------------------------------------------
    # PERC — wood block, pitched on Colundi strange-tones
    # ------------------------------------------------------------------
    add_drum_voice(
        score,
        "perc",
        engine="drum_voice",
        preset="pi_wood_block",
        drum_bus=drum_bus,
        send_db=-3.0,
        mix_db=-10.0,
        effects=[
            # Tame the wood-block's high-band emphasis — raw it's piercy
            # (spectral_centroid ~4.9 kHz, +3.7 dB high-band dominance).
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "lowpass",
                            "cutoff_hz": 4200.0,
                            "slope_db_per_oct": 12,
                        },
                    ]
                },
            ),
        ],
    )
    # Route perc lightly to hall for air.
    score.voices["perc"].sends.append(VoiceSend(target="hall", send_db=-10.0))

    # ------------------------------------------------------------------
    # SNAP — occasional finger-snap punctuation in S3
    # ------------------------------------------------------------------
    add_drum_voice(
        score,
        "snap",
        engine="drum_voice",
        preset="finger_snap",
        drum_bus=drum_bus,
        send_db=-3.0,
        mix_db=-10.0,
    )

    # ==================================================================
    # Note placement
    # ==================================================================

    _place_pad(score)
    _place_arp(score)
    _place_bell(score)
    _place_kick(score)
    _place_perc(score)
    _place_snap(score)

    return score


# ---------------------------------------------------------------------------
# Note placement helpers
# ---------------------------------------------------------------------------


def _place_pad(score: Score) -> None:
    """Pad drone + song-shaped chord motion.

    Core drone (R + P5 + h7 + octave sparkle) sustains the whole piece —
    this is the home chord, 4:6:7 otonal.

    Chord moves:
      - Chorus 1 (17-20): add P4_2 for the "lift" feel that supports the
        bell's ascending phrase.  Glides in from P5.
      - Bridge (25-28): add a LOUDER P4_2 (stronger than chorus 1 P4) so
        the harmonic center perceptually shifts toward the subdominant,
        plus a more-audible N2_2 shade to echo the arp's bridge spice.
      - Chorus return (29-31): brief P4 re-lift, fades before bar 32.
      - Outro: core only — voice fades via natural decay.
    """
    # Core drone (octave 2) — held across the full piece
    for partial, amp in ((R2, -8.0), (P5_2, -11.0), (h7_2, -11.0)):
        score.add_note(
            "pad",
            start=0.0,
            duration=TOTAL_DUR,
            partial=partial,
            amp_db=amp,
        )
    # Octave-up sparkle (R3 + P5_3) — held across the full piece
    for partial in (R3, P5_3):
        score.add_note(
            "pad",
            start=0.0,
            duration=TOTAL_DUR,
            partial=partial,
            amp_db=-14.0,
        )

    # Chorus 1 LIFT: P4_2 for 4 bars (17-20), glides in from P5
    score.add_note(
        "pad",
        start=_pos(17),
        duration=4 * BAR,
        partial=P4_2,
        amp_db=-11.0,
        pitch_motion=PitchMotionSpec.ratio_glide(
            start_ratio=(3 / 2) / (4 / 3),  # glide in from P5 to P4
            end_ratio=1.0,
        ),
    )

    # Bridge P4 FOCUS (25-28): louder than chorus 1 P4.  This is the move
    # that makes the bridge feel harmonically different rather than just
    # "same chord + a spicy sprinkle".  R2 still anchors the bass but P4
    # perceptually takes the lead.
    score.add_note(
        "pad",
        start=_pos(25),
        duration=4 * BAR,
        partial=P4_2,
        amp_db=-9.0,
    )
    # Bridge N2 shade — more audible than a pure decoration.  Paired with
    # the arp's sustained N2 / m3 / s6 drones above.
    score.add_note(
        "pad",
        start=_pos(25),
        duration=4 * BAR,
        partial=(11 / 10) * 2.0,  # N2 in octave 2
        amp_db=-14.0,
    )

    # Chorus return (29-31): brief P4 re-lift, glides in, fades before bar 32
    # so the final chorus resolves onto the pure home chord.
    score.add_note(
        "pad",
        start=_pos(29),
        duration=3 * BAR,
        partial=P4_2,
        amp_db=-12.0,
        pitch_motion=PitchMotionSpec.ratio_glide(
            start_ratio=(3 / 2) / (4 / 3),
            end_ratio=1.0,
        ),
    )


# ---------------------------------------------------------------------------
# Arp placement — hook dispatched per section
# ---------------------------------------------------------------------------


def _place_hook_bar1(score: Score, bar: int, amp_base: float) -> None:
    """Place the ascending 'question' bar of the hook."""
    for step, partial, vel, amp_off, dur_16ths in _HOOK_BAR1:
        beat, sub = _step_to_beat_sub(step)
        score.add_note(
            "arp",
            start=_pos(bar, beat, sub),
            duration=dur_16ths * S16,
            partial=partial,
            amp_db=amp_base + amp_off,
            velocity=vel,
            pitch_motion=_vibrato_for_dur(dur_16ths),
        )


def _place_hook_bar2(
    score: Score, bar: int, amp_base: float, *, extended_landing: bool = False
) -> None:
    """Place the descending 'answer' bar of the hook.

    ``extended_landing=True`` stretches the final R3 to 6 sixteenths so the
    resolution rings into the following bar — used at the end of chorus 2
    and chorus return for a satisfying arrival.
    """
    for step, partial, vel, amp_off, dur_16ths in _HOOK_BAR2:
        beat, sub = _step_to_beat_sub(step)
        dur = 6.0 if (extended_landing and step == 15) else dur_16ths
        score.add_note(
            "arp",
            start=_pos(bar, beat, sub),
            duration=dur * S16,
            partial=partial,
            amp_db=amp_base + amp_off,
            velocity=vel,
            pitch_motion=_vibrato_for_dur(dur),
        )


def _place_brightness_grace(score: Score, bar: int, amp_base: float) -> None:
    """Chorus ornament: h7_4 (upper-octave h7) grace at the 'e of 3'.

    Sits between the held h7_3 and the R4 leap; brightens the ascent
    without anticipating the R4 (so the leap still feels like a surprise).
    """
    beat, sub = _step_to_beat_sub(9)
    score.add_note(
        "arp",
        start=_pos(bar, beat, sub),
        duration=0.85 * S16,
        partial=h7_4,
        amp_db=amp_base - 5.0,
        velocity=0.52,
    )


def _place_climax_cluster(score: Score, bar: int, amp_base: float) -> None:
    """Chorus-return climax: denser ornament filling the first half.

    Graces at steps 2, 4, 7 — all within the ascent toward the octave
    leap.  The step-7 ornament is h7_4 (not R4) so the main R4 leap at
    step 11 remains the hook's peak.
    """
    graces = [
        (2, P5_3, 0.50, -5.5),
        (4, h7_3, 0.52, -5.0),
        (7, h7_4, 0.56, -4.5),
    ]
    for step, partial, vel, amp_off in graces:
        beat, sub = _step_to_beat_sub(step)
        score.add_note(
            "arp",
            start=_pos(bar, beat, sub),
            duration=0.85 * S16,
            partial=partial,
            amp_db=amp_base + amp_off,
            velocity=vel,
        )


def _place_prechorus_drop(score: Score, bar: int, amp_base: float) -> None:
    """Bar 16: accelerating descending figure, then ~1.25 beats of silence.

    Rhythm [0, 3, 5, 6, 7] — gaps of 3, 2, 1, 1 sixteenths, rolling toward
    the R3 landing.  m3 (19/16) at step 6 is the single framed spice — a
    passing tone on the descent from P5 → R.  R3 lands with a moderate
    duration so the second half of the bar reads as real silence (classic
    drop) rather than pedal tone.
    """
    notes: list[tuple[int, float, float, float, float]] = [
        (0, R4, 0.90, 0.0, 2.9),
        (3, h7_3, 0.75, -2.0, 1.9),
        (5, P5_3, 0.68, -2.5, 0.95),
        (6, m3_3, 0.60, -5.0, 0.95),  # spice — descending passing tone
        (7, R3, 0.82, -1.5, 3.0),  # landing, modest dur leaves a drop
    ]
    for step, partial, vel, amp_off, dur_16ths in notes:
        beat, sub = _step_to_beat_sub(step)
        score.add_note(
            "arp",
            start=_pos(bar, beat, sub),
            duration=dur_16ths * S16,
            partial=partial,
            amp_db=amp_base + amp_off,
            velocity=vel,
            pitch_motion=_vibrato_for_dur(dur_16ths),
        )


def _place_bridge_sustains(score: Score, amp_base: float = -11.0) -> None:
    """Bridge (25-28): sustained spicy JI drones, no 16th-note activity.

    All three spicy tones (N2 / m3 / s6) get time to breathe here, framed
    by P4 anchors at the beginning and end.  Wider vibrato (0.005 depth,
    4.5 Hz) than the hook — these are singing drones, not articulated
    notes.
    """
    # (bar, beat, partial, dur_beats, vel, amp_off)
    sustains: list[tuple[int, int, float, float, float, float]] = [
        (25, 1, P4_3, 4.0, 0.68, 0.0),  # P4 anchor — whole bar
        (26, 1, N2_3, 2.0, 0.60, -2.0),  # N2 first half
        (26, 3, m3_3, 2.0, 0.60, -2.0),  # m3 second half
        (27, 1, s6_3, 4.0, 0.65, -1.0),  # s6 spice peak — whole bar
        (28, 1, P4_3, 2.0, 0.62, -1.5),  # P4 return
        (28, 3, m3_3, 2.0, 0.52, -3.5),  # m3 fade into chorus return
    ]
    for bar, beat, partial, dur_beats, vel, amp_off in sustains:
        score.add_note(
            "arp",
            start=_pos(bar, beat, 0),
            duration=dur_beats * BEAT,
            partial=partial,
            amp_db=amp_base + amp_off,
            velocity=vel,
            pitch_motion=PitchMotionSpec.vibrato(depth_ratio=0.005, rate_hz=4.5),
        )


def _place_outro_skeleton(score: Score, amp_base: float = -12.0) -> None:
    """Outro (33-35): hook skeleton, thinning to a single R3 whisper.

    Bar 33 plays the bar-1 hook's skeleton (downbeat + held h7 + octave
    leap + pickup), bar 34 plays the bar-2 skeleton (descent to R3), bar
    35 is just a soft R3 ringing — the piece dissolves rather than cuts.
    """
    # Bar 33 — bar 1 skeleton
    bar33: list[tuple[int, float, float, float, float]] = [
        (0, R3, 0.80, 0.0, 2.9),
        (6, h7_3, 0.68, -2.5, 4.7),
        (11, R4, 0.72, -2.0, 1.9),  # signature leap, one last time
        (15, P5_3, 0.50, -5.0, 1.0),
    ]
    for step, partial, vel, amp_off, dur_16ths in bar33:
        beat, sub = _step_to_beat_sub(step)
        score.add_note(
            "arp",
            start=_pos(33, beat, sub),
            duration=dur_16ths * S16,
            partial=partial,
            amp_db=amp_base + amp_off,
            velocity=vel,
            pitch_motion=_vibrato_for_dur(dur_16ths),
        )
    # Bar 34 — bar 2 skeleton, R3 extended so resolution rings into bar 35
    bar34: list[tuple[int, float, float, float, float]] = [
        (0, R4, 0.72, -1.5, 2.9),
        (6, P5_3, 0.62, -3.0, 4.7),
        (15, R3, 0.70, -2.5, 5.0),
    ]
    for step, partial, vel, amp_off, dur_16ths in bar34:
        beat, sub = _step_to_beat_sub(step)
        score.add_note(
            "arp",
            start=_pos(34, beat, sub),
            duration=dur_16ths * S16,
            partial=partial,
            amp_db=amp_base + amp_off,
            velocity=vel,
            pitch_motion=_vibrato_for_dur(dur_16ths),
        )
    # Bar 35 — final R3 whisper
    score.add_note(
        "arp",
        start=_pos(35, 1, 0),
        duration=4.0 * BEAT,
        partial=R3,
        amp_db=amp_base - 5.0,
        velocity=0.45,
        pitch_motion=PitchMotionSpec.vibrato(depth_ratio=0.0035, rate_hz=5.0),
    )


def _place_arp(score: Score) -> None:
    """Place the arp section by section using the hand-crafted hook.

    The hook is a 2-bar question/answer phrase; variations per section
    are hand-picked ornaments (brightness grace, climax cluster) rather
    than cell permutations.  Amp base shapes the song arc: quiet verse,
    full chorus, intimate bridge, peak chorus return, dissolving outro.
    """
    # Verse (9-14): hook iterated 3x at soft level
    for hook_start in (9, 11, 13):
        _place_hook_bar1(score, hook_start, amp_base=-14.0)
        _place_hook_bar2(score, hook_start + 1, amp_base=-14.0)

    # Bar 15: verse tail — bar 1 only (bar 2 silent, breath before drop)
    _place_hook_bar1(score, 15, amp_base=-12.0)

    # Bar 16: pre-chorus drop — accelerating descent + silence
    _place_prechorus_drop(score, 16, amp_base=-13.0)

    # Chorus 1 (17-20): hook + brightness grace on each bar 1
    for hook_start in (17, 19):
        _place_hook_bar1(score, hook_start, amp_base=-9.0)
        _place_brightness_grace(score, hook_start, amp_base=-9.0)
        _place_hook_bar2(score, hook_start + 1, amp_base=-9.0)

    # Chorus 2 (21-24): hook, bar 24 extends the R3 landing into the bridge
    _place_hook_bar1(score, 21, amp_base=-9.0)
    _place_hook_bar2(score, 22, amp_base=-9.0)
    _place_hook_bar1(score, 23, amp_base=-9.0)
    _place_hook_bar2(score, 24, amp_base=-9.0, extended_landing=True)

    # Bridge (25-28): sustained spicy drones
    _place_bridge_sustains(score)

    # Chorus return (29-32): hook at peak level, bar 31 gets climax cluster,
    # bar 32 extends landing as the bridge → chorus return resolution rings
    # into the outro.
    _place_hook_bar1(score, 29, amp_base=-8.0)
    _place_hook_bar2(score, 30, amp_base=-8.0)
    _place_hook_bar1(score, 31, amp_base=-8.0)
    _place_climax_cluster(score, 31, amp_base=-8.0)
    _place_hook_bar2(score, 32, amp_base=-8.0, extended_landing=True)

    # Outro (33-35): skeleton thinning to a whisper
    _place_outro_skeleton(score)


def _bell_vibrato(dur_16ths: float) -> PitchMotionSpec | None:
    """Vibrato scaled to note length.

    Graces (<2 16ths) get none — too short to hear modulation.
    Medium notes get 0.0035 depth (gentle, conversational).
    Long notes (>=6 16ths) get 0.005 (singing).
    Very long landings (>=16 16ths) get 0.006 (expressive, but still
    within preferred max of ~0.005-0.007 — wider feels operatic).
    """
    if dur_16ths < 2.0:
        return None
    if dur_16ths >= 16.0:
        return PitchMotionSpec.vibrato(depth_ratio=0.006, rate_hz=4.0)
    if dur_16ths >= 6.0:
        return PitchMotionSpec.vibrato(depth_ratio=0.005, rate_hz=4.2)
    return PitchMotionSpec.vibrato(depth_ratio=0.0035, rate_hz=4.5)


# ---------------------------------------------------------------------------
# Bell melody — five hand-written phrases across choruses and return.
#
# Format per tuple: (bar, step, partial, velocity, amp_db, dur_16ths).
# ``step`` uses the 0..15 grid within a bar; negative values mean "pickup
# from the previous bar" (e.g., step -1 = last 16th of bar-1).  Duration
# is in sixteenths so the melodic rhythm reads clearly at the source.
#
# Design principles:
#   - Recurring rhythmic signature: last-16th pickup → downbeat held note
#     → motion → grace → landing.  Varies across phrases but the pickup
#     + held-downbeat anchor repeats so the ear internalizes it.
#   - One framed spice per phrase (except the final return, which is
#     clean and consonant — spice is for tension phrases, not resolution).
#   - Phrase 1A and 2A share the rhythmic signature but diverge in pitch
#     and spice; 1B introduces the octave-drop device; 2B is a faster
#     cadential tag with an unexpected syncopated landing; 3 is the
#     emotional peak with the piece's only note above R4.
# ---------------------------------------------------------------------------

# PHRASE 1A — "soaring entry" (bars 16.15 → 18).
# Pickup enters during the pre-chorus drop's silence.  Spice: s6 at 17.14
# (passing tone between h7 and P4 on the descent).  Lands on P4_3 held
# 12 sixteenths — deliberately overlaps the pad's P4 lift so melody and
# harmony share a subdominant moment.
_BELL_PHRASE_1A: list[tuple[int, int, float, float, float, float]] = [
    (16, 15, P5_3, 0.58, -11.0, 1.0),  # pickup
    (17, 0, R4, 0.88, -7.5, 5.0),  # held downbeat anchor
    (17, 6, h7_3, 0.78, -8.5, 2.0),
    (17, 10, R4, 0.70, -9.5, 2.0),
    (17, 14, s6_3, 0.52, -11.5, 1.0),  # spice — passing h7 → P4
    (17, 15, h7_3, 0.62, -10.5, 1.0),
    (18, 0, P4_3, 0.80, -8.5, 12.0),  # landing, sits over pad's P4 lift
]

# PHRASE 1B — "twisting, octave-drop" (bars 18.15 → 20.15).
# Pickup h7_3, new h7 entry, N2_4 grace at 19.6 (narrow-2nd upper
# neighbor above R4 — extremely Colundi), return to h7, R4 peak, m3
# grace at 20.6, then SIGNATURE OCTAVE DROP from R4 down to R3 at
# 20.11.  Different landing than 1A's P4 — descends all the way home.
_BELL_PHRASE_1B: list[tuple[int, int, float, float, float, float]] = [
    (18, 15, P4_3, 0.55, -11.0, 1.0),  # pickup
    (19, 0, h7_3, 0.86, -8.0, 4.0),  # held downbeat
    (19, 4, R4, 0.74, -9.0, 2.0),
    (19, 6, N2_4, 0.54, -11.5, 1.0),  # spice — narrow-2nd above R4
    (19, 7, R4, 0.68, -9.5, 3.0),
    (19, 12, h7_3, 0.72, -9.0, 3.0),
    (20, 0, R4, 0.82, -8.0, 5.0),  # re-peak
    (20, 6, m3_3, 0.52, -12.0, 1.0),  # spice — m3 as distant color
    (20, 7, P5_3, 0.70, -9.5, 4.0),
    (20, 11, R3, 0.78, -8.5, 5.0),  # OCTAVE DROP landing
]

# PHRASE 2A — "variant answer" (bars 20.15 → 22).
# Same rhythmic signature as 1A (pickup → downbeat → motion → grace →
# landing) but different pitches: P5 where 1A had h7, m3 grace where 1A
# had s6, lands on low R3.  Then an N2 HELD SUSPENSION for 4 sixteenths
# — a real tension (165¢ above R3), not just a grace — before resolving
# back to R3.  The suspension is the phrase's emotional core.
_BELL_PHRASE_2A: list[tuple[int, int, float, float, float, float]] = [
    (20, 15, h7_3, 0.58, -11.0, 1.0),  # pickup
    (21, 0, P5_3, 0.84, -8.0, 5.0),  # held downbeat (P5 not R4 — variant)
    (21, 6, P4_3, 0.74, -9.0, 2.0),
    (21, 10, P5_3, 0.66, -10.0, 2.0),
    (21, 14, m3_3, 0.52, -12.0, 1.0),  # spice — m3 variant of 1A's s6
    (21, 15, P4_3, 0.60, -11.0, 1.0),
    (22, 0, R3, 0.78, -8.5, 5.0),  # lands low
    (22, 8, N2_3, 0.68, -10.0, 4.0),  # HELD SUSPENSION — tension
    (22, 12, R3, 0.74, -9.0, 4.0),  # resolution
]

# PHRASE 2B — "cadential tag" (bars 22.15 → 24).
# Faster: a rising run P5 → h7 → R4 → P5 → P4, then m3 grace at 24.4,
# then lands on R3 at bar 24 BEAT 2.5 (syncopated, unexpected — the
# only landing on a weak beat in the melody).  Longer release rings
# through the rest of the bar.  This is the phrase that makes the
# melody feel conversational rather than schoolbook.
_BELL_PHRASE_2B: list[tuple[int, int, float, float, float, float]] = [
    (22, 15, P4_3, 0.58, -11.0, 1.0),  # pickup
    (23, 0, P5_3, 0.82, -8.5, 3.0),
    (23, 3, h7_3, 0.76, -9.0, 3.0),
    (23, 6, R4, 0.82, -8.5, 3.0),  # peak of the rise
    (23, 9, P5_3, 0.72, -9.5, 3.0),
    (23, 12, P4_3, 0.68, -10.0, 4.0),
    (24, 4, m3_3, 0.54, -12.0, 2.0),  # spice — pre-landing color
    (24, 6, R3, 0.82, -8.0, 10.0),  # SYNCOPATED LANDING at beat 2.5
]

# PHRASE 3 — "return and resolution" (bars 28.15 → 32).
# Longer entry: R4 held 8 sixteenths, h7_3 held 8.  Then the emotional
# peak: h7_4 at bar 30 beat 1 — the ONLY note above R4 in the piece,
# used exactly once.  Descent through R4 / P5 / h7_3 / P5 / P4, final
# held R3 spanning 24 sixteenths from bar 31.8 through bar 32.  NO
# SPICE — the return is consonant, definitive, resolved.
_BELL_PHRASE_3: list[tuple[int, int, float, float, float, float]] = [
    (28, 15, P5_3, 0.56, -11.0, 1.0),  # pickup
    (29, 0, R4, 0.90, -7.5, 8.0),  # long held entry
    (29, 8, h7_3, 0.82, -8.5, 8.0),
    (30, 0, h7_4, 0.94, -6.5, 5.0),  # EMOTIONAL PEAK — only note > R4
    (30, 5, R4, 0.80, -8.5, 3.0),
    (30, 8, P5_3, 0.74, -9.0, 4.0),
    (30, 12, h7_3, 0.72, -9.5, 4.0),
    (31, 0, P5_3, 0.70, -9.5, 4.0),
    (31, 4, P4_3, 0.66, -10.0, 4.0),
    (31, 8, R3, 0.80, -8.0, 24.0),  # final landing, rings into outro
]

_BELL_PHRASES: list[list[tuple[int, int, float, float, float, float]]] = [
    _BELL_PHRASE_1A,
    _BELL_PHRASE_1B,
    _BELL_PHRASE_2A,
    _BELL_PHRASE_2B,
    _BELL_PHRASE_3,
]


def _place_bell(score: Score) -> None:
    """Place the five bell phrases across chorus 1, chorus 2, and return.

    Bridge (bars 25-28) is silent — the arp's sustained spicy drones
    carry that section alone.  Graces and ornaments use quieter velocities
    and lower amp_db so they read as shape decoration, not equal-weight
    notes.  See phrase-level comments for design notes.
    """
    for phrase in _BELL_PHRASES:
        for bar, step, partial, vel, amp_db, dur_16ths in phrase:
            beat, sub = _step_to_beat_sub(step)
            score.add_note(
                "bell",
                start=_pos(bar, beat, sub),
                duration=dur_16ths * S16,
                partial=partial,
                amp_db=amp_db,
                velocity=vel,
                pitch_motion=_bell_vibrato(dur_16ths),
            )


def _place_kick(score: Score) -> None:
    """Mellow kick, song-shaped.

    Verse (13-15): beat 1 only, establishing pulse
    Pre-chorus (16): silent — the classic drop before the chorus arrives
    Chorus (17-24): beats 1 + 3, full head nod
    Bridge (25-28): beat 1 only, half-time — intimate, tense
    Chorus return (29-32): beats 1 + 3, full, slightly hotter
    Outro (33-34): beat 1 only, thinning
    (35-36): silent, let the tail ring out
    """
    # Verse — beat 1 only, sparse entry
    for bar in range(13, 16):
        score.add_note("kick", start=_pos(bar, 1), duration=0.6, freq=F0, amp_db=-9.0)
    # Pre-chorus bar 16 — silent (drop)

    # Chorus (17-24) — full head nod
    for bar in range(17, 25):
        score.add_note("kick", start=_pos(bar, 1), duration=0.6, freq=F0, amp_db=-6.0)
        score.add_note("kick", start=_pos(bar, 3), duration=0.6, freq=F0, amp_db=-7.0)

    # Bridge (25-28) — beat 1 only, half-time
    for bar in range(25, 29):
        score.add_note("kick", start=_pos(bar, 1), duration=0.6, freq=F0, amp_db=-8.5)

    # Chorus return (29-32) — full again, slightly hotter
    for bar in range(29, 33):
        score.add_note("kick", start=_pos(bar, 1), duration=0.6, freq=F0, amp_db=-5.5)
        score.add_note("kick", start=_pos(bar, 3), duration=0.6, freq=F0, amp_db=-6.5)

    # Outro (33-34) — thinning beat 1 only
    for bar in range(33, 35):
        score.add_note(
            "kick",
            start=_pos(bar, 1),
            duration=0.6,
            freq=F0,
            amp_db=-10.0 - (bar - 33) * 1.5,
        )


def _place_perc(score: Score) -> None:
    """Wood-block euclidean rhythm, bars 15-28.

    Euclidean 5-in-16 gives a syncopated sparse pattern.  Rotate by 2
    steps in the bloom section for slight variation.
    """
    pattern_a = euclidean_pattern(5, 16, rotation=0)
    pattern_b = euclidean_pattern(5, 16, rotation=2)
    for bar in range(15, 29):
        pattern = pattern_b if 21 <= bar <= 26 else pattern_a
        pitch_idx = (bar - 15) % len(_PERC_PITCHES)
        freq = _PERC_PITCHES[pitch_idx]
        base_amp = -14.0 if bar < 17 else -11.0
        if bar >= 27:
            base_amp -= 2.0  # thin out near section end
        for step in range(16):
            if not pattern[step]:
                continue
            beat = 1 + (step // 4)
            sub = step % 4
            # Velocity contour: downbeats a bit louder
            vel = 0.85 if step % 4 == 0 else 0.65
            score.add_note(
                "perc",
                start=_pos(bar, beat, sub),
                duration=0.08,
                freq=freq,
                amp_db=base_amp,
                velocity=vel,
            )


def _place_snap(score: Score) -> None:
    """Sparse finger-snap on beat 4, every other bar in S3 bloom."""
    for bar in (19, 21, 23, 25, 27):
        score.add_note(
            "snap",
            start=_pos(bar, 4),
            duration=0.08,
            freq=440.0,
            amp_db=-12.0,
            velocity=0.75,
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

PIECES: dict[str, PieceDefinition] = {
    "colundi_arps_study": PieceDefinition(
        name="colundi_arps_study",
        output_name="colundi_arps_study",
        build_score=build_score,
        sections=(
            PieceSection(label="S1 intro", start_seconds=0.0, end_seconds=S1_END),
            PieceSection(label="S2 verse", start_seconds=S1_END, end_seconds=S2_END),
            PieceSection(
                label="S3 chorus + bridge", start_seconds=S2_END, end_seconds=S3_END
            ),
            PieceSection(
                label="S4 return + outro",
                start_seconds=S3_END,
                end_seconds=TOTAL_DUR,
            ),
        ),
    ),
}
