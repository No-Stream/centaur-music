"""newton_bloom — patient bloom of the Diva-inspired subtractive overhaul.

A four-minute, post-Aphex-tender piece in 7-limit JI centered on F#3 that
treats the new Newton-solver self-oscillating ladder as a singing voice in
dialogue with a BBD-chorused analog pad.  The goal is to actually *use* the
new capabilities of the `avril-14` branch — Newton solver at `quality="divine"`,
`transient_mode="analog"` legato carryover, hard sync + ring mod with matrix
macro rides, audio-rate modulation via `OscillatorSource`, flow-exciter breath,
Vital-style spectral-morphed additive layers — as musical materials rather
than study checkboxes.

Sections (target wall-clock times):

  1. Pad bloom              (0:00 - 0:40)  Juno-chorused pad establishes F#
     tonic; breath layer fades in via flow exciter.
  2. Harmonic motion        (0:40 - 1:20)  Chord moves through 4/3, 9/8 roots.
     Still pad-only — setting the JI soil.
  3. Lead enters            (1:20 - 2:10)  Newton ladder lead sings; slow
     resonance_q rise toward the oscillation threshold.
  4. Dialogue               (2:10 - 2:50)  Sallen-Key sub joins with preamp
     voice distortion.  Lead counter-phrases; ring-mod macro rides in.
  5. Apex                   (2:50 - 3:30)  Audio-rate OscillatorSource shakes
     osc2_detune_cents on the lead (~180 Hz).  Pad thickens with a phase-
     dispersed additive arp.  Utonal resolution.
  6. Decay                  (3:30 - 4:00)  Lead descends through 5/4 to tonic.
     Pad + breath fade into hall-reverb tail.
"""

from __future__ import annotations

from code_musics.automation import (
    AutomationSegment,
    AutomationSpec,
    AutomationTarget,
)
from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    VelocityHumanizeSpec,
)
from code_musics.modulation import (
    LFOSource,
    MacroSource,
    ModConnection,
    OscillatorSource,
)
from code_musics.pieces._shared import (
    DEFAULT_MASTER_EFFECTS,
    SOFT_REVERB_EFFECT,
)
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import EffectSpec, Score, SendBusSpec, VoiceSend

BPM = 72.0
BEAT = 60.0 / BPM
BAR = 4.0 * BEAT  # ~3.333 s

S1_START = 0.0
S2_START = 8.0 * BAR  # ~26.7 s
S3_START = 16.0 * BAR  # ~53.3 s
S4_START = 26.0 * BAR  # ~86.7 s
S5_START = 34.0 * BAR  # ~113.3 s
S6_START = 42.0 * BAR  # ~140.0 s
TOTAL_DUR = 48.0 * BAR  # ~160.0 s
# ~2:40 total — tight but complete.  The plan optimistically said 4 min but
# `BAR ~= 3.33 s` means 48 bars is the right musical length for this slow
# breath-tempo material without padding with silence.

F0_HZ = 185.0  # F#3 — shared with diva_study.py


def _hall_bus() -> SendBusSpec:
    """Shared hall reverb return.  Leave return_db at 0; ride via voice send_db."""
    return SendBusSpec(
        name="hall",
        effects=[SOFT_REVERB_EFFECT],
        return_db=0.0,
    )


def _pad_effects() -> list[EffectSpec]:
    """BBD chorus + gentle EQ tilt — the Juno-I+II dream."""
    return [
        EffectSpec(
            "eq",
            {
                "bands": [
                    {"kind": "highpass", "cutoff_hz": 60.0, "slope_db_per_oct": 12},
                    {"kind": "high_shelf", "freq_hz": 6500.0, "gain_db": -1.2},
                ]
            },
        ),
        EffectSpec(
            "bbd_chorus",
            {"preset": "juno_i_plus_ii", "mix": 0.24},
            automation=[
                AutomationSpec(
                    target=AutomationTarget(kind="control", name="mix"),
                    segments=(
                        AutomationSegment(
                            start=0.0,
                            end=S5_START,
                            shape="hold",
                            value=0.24,
                        ),
                        AutomationSegment(
                            start=S5_START,
                            end=S6_START,
                            shape="linear",
                            start_value=0.24,
                            end_value=0.16,
                        ),
                        AutomationSegment(
                            start=S6_START,
                            end=TOTAL_DUR,
                            shape="linear",
                            start_value=0.16,
                            end_value=0.28,
                        ),
                    ),
                )
            ],
        ),
    ]


def _lead_effects() -> list[EffectSpec]:
    """Slapback delay tuned to the bar grid, then into a dark lowpass."""
    return [
        EffectSpec(
            "delay",
            {
                "delay_seconds": 3.0 * BEAT / 4.0,  # dotted-eighth at 72 BPM
                "feedback": 0.28,
                "mix": 0.18,
            },
        ),
    ]


def _sub_effects() -> list[EffectSpec]:
    return [
        EffectSpec(
            "eq",
            {
                "bands": [
                    {"kind": "highpass", "cutoff_hz": 32.0, "slope_db_per_oct": 12},
                    {"kind": "low_shelf", "freq_hz": 120.0, "gain_db": 1.5},
                ]
            },
        ),
    ]


def _breath_effects() -> list[EffectSpec]:
    return [
        EffectSpec(
            "eq",
            {
                "bands": [
                    {"kind": "highpass", "cutoff_hz": 200.0, "slope_db_per_oct": 12},
                    {"kind": "high_shelf", "freq_hz": 9000.0, "gain_db": -4.0},
                ]
            },
        ),
    ]


def _arp_effects() -> list[EffectSpec]:
    return [
        EffectSpec(
            "eq",
            {
                "bands": [
                    {"kind": "highpass", "cutoff_hz": 240.0, "slope_db_per_oct": 12},
                    {"kind": "high_shelf", "freq_hz": 7000.0, "gain_db": -2.0},
                ]
            },
        ),
    ]


_CHORD_PROGRESSION: list[tuple[tuple[float, ...], str]] = [
    # (partials, label)
    ((1.0, 5 / 4, 3 / 2, 7 / 4), "1"),  # F# 7-limit tonic with 7/4
    ((1.0, 5 / 4, 3 / 2, 7 / 4), "1"),  # sustain 2 bars
    ((9 / 8, 45 / 32, 27 / 16, 15 / 8), "9/8"),  # supertonic with 7th
    ((9 / 8, 45 / 32, 27 / 16, 15 / 8), "9/8"),
    ((4 / 3, 5 / 3, 2.0, 7 / 3), "4/3"),  # subdominant 7th
    ((4 / 3, 5 / 3, 2.0, 7 / 3), "4/3"),
    ((3 / 2, 15 / 8, 9 / 4, 21 / 8), "3/2"),  # dominant with septimal 7
    ((3 / 2, 15 / 8, 9 / 4, 21 / 8), "3/2"),
]
# Utonal closing voicing, reserved for section 5 apex.
_UTONAL_VOICING = (7 / 6, 35 / 24, 7 / 4, 49 / 24)


def _pad_chords(score: Score) -> None:
    """Two bars per chord, staggered onsets for voice leading."""
    stagger = (0.0, 0.015, 0.03, 0.045)
    for bar_index, (partials, _) in enumerate(_CHORD_PROGRESSION):
        start = bar_index * 2.0 * BAR
        duration = 2.0 * BAR + 0.5
        for partial, offset in zip(partials, stagger, strict=True):
            amp_db = -20.0 if bar_index < 8 else -18.0
            score.add_note(
                "pad",
                start=start + offset,
                duration=duration,
                partial=partial,
                amp_db=amp_db,
                velocity=0.95,
            )

    # Section 2 continuation — new root movement (cycles 9/8 -> 4/3 -> 9/8 -> 3/2)
    motion_progression = [
        (9 / 8, 45 / 32, 27 / 16, 15 / 8),
        (4 / 3, 5 / 3, 2.0, 7 / 3),
        (9 / 8, 45 / 32, 27 / 16, 63 / 32),  # 63/32 = 7-limit major 7
        (3 / 2, 15 / 8, 9 / 4, 21 / 8),
    ]
    for i, partials in enumerate(motion_progression):
        start = S2_START + i * 2.0 * BAR
        for partial, offset in zip(partials, stagger, strict=True):
            score.add_note(
                "pad",
                start=start + offset,
                duration=2.0 * BAR + 0.5,
                partial=partial,
                amp_db=-19.0,
                velocity=0.95,
            )

    # Section 3 — lead enters over continuing pad.  Longer holds, quieter to
    # leave room for the singing filter.
    s3_progression = [
        (1.0, 5 / 4, 3 / 2, 7 / 4),
        (4 / 3, 5 / 3, 2.0, 7 / 3),
        (1.0, 9 / 8, 5 / 4, 3 / 2),
        (9 / 8, 45 / 32, 27 / 16, 15 / 8),
        (3 / 2, 15 / 8, 9 / 4, 21 / 8),
    ]
    for i, partials in enumerate(s3_progression):
        start = S3_START + i * 2.0 * BAR
        for partial, offset in zip(partials, stagger, strict=True):
            score.add_note(
                "pad",
                start=start + offset,
                duration=2.0 * BAR + 0.6,
                partial=partial,
                amp_db=-22.0,
                velocity=0.9,
            )

    # Section 4 — dialogue.  Sparser chord rhythm so the sub has room.
    s4_progression = [
        (1.0, 5 / 4, 3 / 2, 7 / 4),
        (4 / 3, 5 / 3, 2.0, 5 / 2),  # 5/2 = major 10th above tonic
        (9 / 8, 45 / 32, 27 / 16, 15 / 8),
        (3 / 2, 15 / 8, 9 / 4, 21 / 8),
    ]
    for i, partials in enumerate(s4_progression):
        start = S4_START + i * 2.0 * BAR
        for partial, offset in zip(partials, stagger, strict=True):
            score.add_note(
                "pad",
                start=start + offset,
                duration=2.0 * BAR + 0.4,
                partial=partial,
                amp_db=-21.0,
                velocity=0.92,
            )

    # Section 5 — apex, utonal resolution for punctuation.
    s5_progression = [
        (1.0, 5 / 4, 3 / 2, 7 / 4),  # home
        _UTONAL_VOICING,  # septimal subdominant shadow
        (7 / 6, 5 / 3, 7 / 4, 49 / 24),  # utonal extension
        (1.0, 9 / 8, 5 / 4, 7 / 4),  # settle
    ]
    for i, partials in enumerate(s5_progression):
        start = S5_START + i * 2.0 * BAR
        for partial, offset in zip(partials, stagger, strict=True):
            score.add_note(
                "pad",
                start=start + offset,
                duration=2.0 * BAR + 0.6,
                partial=partial,
                amp_db=-19.0,
                velocity=0.98,
            )

    # Section 6 — decay.  Sustained tonic, lets reverb tail breathe.
    for partial, offset in zip((1.0, 5 / 4, 3 / 2, 9 / 4), stagger, strict=True):
        score.add_note(
            "pad",
            start=S6_START + offset,
            duration=(TOTAL_DUR - S6_START) - 0.5,
            partial=partial,
            amp_db=-22.0,
            velocity=0.85,
        )


_LEAD_MELODY_S3: list[tuple[float, float, float, float]] = [
    # (start_bar, duration_bars, partial, amp_db)
    (0.0, 2.0, 1.0, -15.0),
    (2.0, 1.5, 5 / 4, -14.0),
    (3.5, 1.5, 3 / 2, -13.0),
    (5.0, 2.0, 7 / 4, -12.0),
    (7.0, 1.0, 3 / 2, -13.0),
    (8.0, 1.5, 9 / 8, -14.0),
    (9.5, 0.5, 5 / 4, -14.0),
    # Long held phrase with vibrato
    (10.0, 3.0, 3 / 2, -13.0),
]


_LEAD_MELODY_S4: list[tuple[float, float, float, float]] = [
    (0.0, 2.0, 7 / 4, -12.0),
    (2.0, 1.0, 2.0, -11.0),
    (3.0, 1.0, 9 / 4, -11.0),
    (4.0, 1.5, 2.0, -12.0),
    (5.5, 2.5, 7 / 4, -12.0),
    # counter-phrase descending
    (8.0, 1.5, 3 / 2, -13.0),
    (9.5, 1.0, 5 / 4, -13.0),
    (10.5, 1.5, 4 / 3, -13.0),
]


_LEAD_MELODY_S5: list[tuple[float, float, float, float]] = [
    # Apex — reach for the high septimal.
    (0.0, 1.5, 2.0, -11.0),
    (1.5, 1.0, 9 / 4, -10.0),
    (2.5, 1.5, 5 / 2, -10.0),
    (4.0, 2.0, 21 / 8, -10.0),  # 21/8 = 3/2 * 7/4, "singing" interval
    (6.0, 1.5, 2.0, -12.0),
    (7.5, 0.5, 49 / 24, -12.0),  # septimal neighbor-tone
    # Descent
    (8.0, 2.0, 7 / 4, -13.0),
]


_LEAD_MELODY_S6: list[tuple[float, float, float, float]] = [
    # Decay back to tonic.
    (0.0, 2.0, 5 / 4, -15.0),
    (2.0, 3.0, 9 / 8, -16.0),
    (5.0, (TOTAL_DUR - S6_START) / BAR - 5.2, 1.0, -18.0),
]


def _add_lead_phrase(
    score: Score,
    *,
    section_start: float,
    melody: list[tuple[float, float, float, float]],
    vibrato_depth: float,
    with_vibrato_on_long_notes: bool = True,
) -> None:
    """Add the lead melody with vibrato on sustained notes (>=1.5 bars)."""
    for start_bar, duration_bars, partial, amp_db in melody:
        duration = duration_bars * BAR
        pitch_motion: PitchMotionSpec | None = None
        if with_vibrato_on_long_notes and duration_bars >= 1.5:
            pitch_motion = PitchMotionSpec.vibrato(
                depth_ratio=vibrato_depth,
                rate_hz=4.8,
            )
        score.add_note(
            "lead",
            start=section_start + start_bar * BAR,
            duration=duration,
            partial=partial,
            amp_db=amp_db,
            velocity=1.0,
            pitch_motion=pitch_motion,
        )


def _sub_line(score: Score) -> None:
    """Section 4-5 sub bass walks.  Sub-octave partials for weight."""
    # Section 4: 1 -> 7/4 -> 4/3 -> 1 (each 2 bars), sub-octave.
    s4_bass = [(0.5, -12.0), (7 / 8, -11.0), (2 / 3, -12.0), (0.5, -13.0)]
    for i, (partial, amp_db) in enumerate(s4_bass):
        score.add_note(
            "sub",
            start=S4_START + i * 2.0 * BAR,
            duration=2.0 * BAR + 0.3,
            partial=partial,
            amp_db=amp_db,
            velocity=0.98,
        )

    # Section 5: utonal territory in the sub — 7/12 = 7/6 one octave down,
    # then 35/48 (5/3 octave down * 7/10) — actually simpler: do 5/8, 7/12,
    # 1/2 as a low walking bass.
    s5_bass = [(0.5, -13.0), (7 / 12, -12.0), (7 / 12, -12.0), (0.5, -14.0)]
    for i, (partial, amp_db) in enumerate(s5_bass):
        score.add_note(
            "sub",
            start=S5_START + i * 2.0 * BAR,
            duration=2.0 * BAR + 0.3,
            partial=partial,
            amp_db=amp_db,
            velocity=0.95,
        )


def _breath_line(score: Score) -> None:
    """Flow-exciter breath layer — long sustained partials across the piece."""
    # One long sustain across the whole piece, chord following pad tonic.
    score.add_note(
        "breath",
        start=2.0,
        duration=TOTAL_DUR - 3.0,
        partial=1.0,
        amp_db=-24.0,
        velocity=1.0,
    )


def _arp_line(score: Score) -> None:
    """Section 5-only dispersed-additive arp counterpoint."""
    # Sparse two-note-per-bar gesture emphasizing the apex's septimal territory.
    arp_pattern = [
        (0.0, 0.75, 3.0),
        (1.0, 0.75, 7 / 2),  # = 7/4 * 2, "high shimmer"
        (2.0, 0.75, 5 / 2),
        (3.0, 0.75, 21 / 8),
        (4.0, 0.75, 3.0),
        (5.0, 0.75, 49 / 16),  # septimal ghost
        (6.0, 0.75, 2.0),
        (7.0, 1.5, 7 / 4),
    ]
    for start_beat, dur_beats, partial in arp_pattern:
        score.add_note(
            "arp",
            start=S5_START + start_beat * BEAT,
            duration=dur_beats * BEAT,
            partial=partial,
            amp_db=-18.0,
            velocity=0.85,
        )


def _build_lead_resonance_automation() -> AutomationSpec:
    """Slow resonance_q rise into apex, settle into decay.

    q curve: 8 (intro warmup, lead silent) -> 14 (enter) -> 28 (section 4
    dialogue) -> 36 (apex, self-oscillation threshold) -> 22 (decay).
    Uses exp shapes on the rising arcs to match frequency-domain perception.
    """
    return AutomationSpec(
        target=AutomationTarget(kind="synth", name="resonance_q"),
        segments=(
            AutomationSegment(
                start=0.0,
                end=S3_START,
                shape="hold",
                value=10.0,
            ),
            AutomationSegment(
                start=S3_START,
                end=S4_START,
                shape="exp",
                start_value=10.0,
                end_value=24.0,
            ),
            AutomationSegment(
                start=S4_START,
                end=S5_START,
                shape="exp",
                start_value=24.0,
                end_value=32.0,
            ),
            AutomationSegment(
                start=S5_START,
                end=S6_START,
                shape="linear",
                start_value=32.0,
                end_value=36.0,
            ),
            AutomationSegment(
                start=S6_START,
                end=TOTAL_DUR,
                shape="exp",
                start_value=36.0,
                end_value=14.0,
            ),
        ),
        clamp_min=2.0,
        clamp_max=40.0,
    )


def _build_lead_cutoff_automation() -> AutomationSpec:
    """Cutoff_hz rides opens the filter through the piece.  Exp shapes
    throughout since we're in frequency space."""
    return AutomationSpec(
        target=AutomationTarget(kind="synth", name="cutoff_hz"),
        segments=(
            AutomationSegment(
                start=0.0,
                end=S3_START,
                shape="hold",
                value=600.0,
            ),
            AutomationSegment(
                start=S3_START,
                end=S4_START,
                shape="exp",
                start_value=600.0,
                end_value=2400.0,
            ),
            AutomationSegment(
                start=S4_START,
                end=S5_START,
                shape="exp",
                start_value=2400.0,
                end_value=1800.0,  # settle while sub enters
            ),
            AutomationSegment(
                start=S5_START,
                end=S6_START,
                shape="exp",
                start_value=1800.0,
                end_value=3600.0,  # apex opens up
            ),
            AutomationSegment(
                start=S6_START,
                end=TOTAL_DUR,
                shape="exp",
                start_value=3600.0,
                end_value=900.0,
            ),
        ),
    )


def _build_pad_filter_morph_automation() -> AutomationSpec:
    """Slow breath on the pad's filter morph — cascade filter taps."""
    return AutomationSpec(
        target=AutomationTarget(kind="synth", name="filter_morph"),
        segments=(
            AutomationSegment(
                start=0.0,
                end=S3_START,
                shape="linear",
                start_value=0.0,
                end_value=0.25,
            ),
            AutomationSegment(
                start=S3_START,
                end=S5_START,
                shape="linear",
                start_value=0.25,
                end_value=0.45,
            ),
            AutomationSegment(
                start=S5_START,
                end=S6_START,
                shape="linear",
                start_value=0.45,
                end_value=0.60,
            ),
            AutomationSegment(
                start=S6_START,
                end=TOTAL_DUR,
                shape="linear",
                start_value=0.60,
                end_value=0.20,
            ),
        ),
    )


def _build_sub_cutoff_automation() -> AutomationSpec:
    """Sub bass filter opens in section 4, closes in section 6."""
    return AutomationSpec(
        target=AutomationTarget(kind="synth", name="cutoff_hz"),
        segments=(
            AutomationSegment(
                start=0.0,
                end=S4_START,
                shape="hold",
                value=300.0,
            ),
            AutomationSegment(
                start=S4_START,
                end=S5_START,
                shape="exp",
                start_value=300.0,
                end_value=1100.0,
            ),
            AutomationSegment(
                start=S5_START,
                end=S6_START,
                shape="exp",
                start_value=1100.0,
                end_value=1400.0,
            ),
            AutomationSegment(
                start=S6_START,
                end=TOTAL_DUR,
                shape="exp",
                start_value=1400.0,
                end_value=300.0,
            ),
        ),
    )


def _build_hall_send_automation(base_db: float = -18.0) -> AutomationSpec:
    """Lead hall send rides up into the apex and falls through the decay."""
    return AutomationSpec(
        target=AutomationTarget(kind="control", name="send_db"),
        segments=(
            AutomationSegment(
                start=0.0,
                end=S4_START,
                shape="hold",
                value=base_db,
            ),
            AutomationSegment(
                start=S4_START,
                end=S5_START,
                shape="linear",
                start_value=base_db,
                end_value=base_db + 5.0,
            ),
            AutomationSegment(
                start=S5_START,
                end=S6_START,
                shape="linear",
                start_value=base_db + 5.0,
                end_value=base_db + 8.0,
            ),
            AutomationSegment(
                start=S6_START,
                end=TOTAL_DUR,
                shape="linear",
                start_value=base_db + 8.0,
                end_value=base_db - 2.0,
            ),
        ),
    )


def _build_ring_macro_automation() -> AutomationSpec:
    """Ring-mod macro — 0 through most of the piece, rises in sections 4-5."""
    return AutomationSpec(
        target=AutomationTarget(kind="control", name="mix"),
        segments=(
            AutomationSegment(
                start=0.0,
                end=S4_START,
                shape="hold",
                value=0.0,
            ),
            AutomationSegment(
                start=S4_START,
                end=S5_START,
                shape="linear",
                start_value=0.0,
                end_value=0.22,
            ),
            AutomationSegment(
                start=S5_START,
                end=S6_START,
                shape="linear",
                start_value=0.22,
                end_value=0.38,
            ),
            AutomationSegment(
                start=S6_START,
                end=TOTAL_DUR,
                shape="linear",
                start_value=0.38,
                end_value=0.0,
            ),
        ),
        clamp_min=0.0,
        clamp_max=1.0,
    )


def _build_fm_macro_automation() -> AutomationSpec:
    """Audio-rate FM macro — active only in the apex."""
    fm_peak = S5_START + (S6_START - S5_START) * 0.5
    return AutomationSpec(
        target=AutomationTarget(kind="control", name="mix"),
        segments=(
            AutomationSegment(
                start=0.0,
                end=S5_START,
                shape="hold",
                value=0.0,
            ),
            AutomationSegment(
                start=S5_START,
                end=fm_peak,
                shape="linear",
                start_value=0.0,
                end_value=1.0,
            ),
            AutomationSegment(
                start=fm_peak,
                end=S6_START,
                shape="linear",
                start_value=1.0,
                end_value=0.0,
            ),
            AutomationSegment(
                start=S6_START,
                end=TOTAL_DUR,
                shape="hold",
                value=0.0,
            ),
        ),
        clamp_min=0.0,
        clamp_max=1.0,
    )


def build_score() -> Score:
    """Build the newton_bloom score."""
    score = Score(
        f0_hz=F0_HZ,
        master_effects=DEFAULT_MASTER_EFFECTS,
        timing_humanize=TimingHumanizeSpec(preset="chamber"),
        send_buses=[_hall_bus()],
    )

    # ---- Shared drift bus so pad + lead + sub breathe together ----
    score.add_drift_bus(
        "ensemble",
        rate_hz=0.07,
        depth_cents=2.5,
        seed=19,
    )

    # ---- Macros driving the modulation matrix ----
    score.add_macro(
        "ring_amount", default=0.0, automation=_build_ring_macro_automation()
    )
    score.add_macro("fm_amount", default=0.0, automation=_build_fm_macro_automation())

    # ---- Score-level modulations ----
    # Slow LFO rides the pad cutoff for a living-analog breath.
    score.modulations.extend(
        [
            ModConnection(
                name="pad_cutoff_lfo",
                source=LFOSource(rate_hz=0.05, waveshape="sine", seed=7),
                target=AutomationTarget(kind="synth", name="cutoff_hz"),
                amount=140.0,
                bipolar=True,
                mode="add",
            ),
        ]
    )

    # ---- Pad voice ----
    # prophet_pad preset through filter_morph automation; bus_chorus; breath
    # through the flow exciter.  transient_mode "analog" so phase carries
    # across the legato chord transitions.
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "polyblep",
            "preset": "prophet_pad",
            "filter_topology": "cascade",
            "quality": "great",
            "transient_mode": "analog",
            "voice_card_spread": 1.6,
            "analog_jitter": 0.6,
            "filter_morph": 0.0,
        },
        effects=_pad_effects(),
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(),
        velocity_group="ensemble",
        mix_db=-2.0,
        pan=-0.12,
        sends=[
            VoiceSend(target="hall", send_db=-14.0),
        ],
        automation=[
            _build_pad_filter_morph_automation(),
        ],
        drift_bus="ensemble",
        drift_bus_correlation=0.85,
    )

    # ---- Lead voice ----
    # Newton ladder at divine quality with transient-mode carryover for legato
    # singing.  voice_dist preamp for subtle tape warmth.  Mono with legato.
    lead_modulations = [
        # Ring macro rides osc2_level — a higher level pulls more of the
        # ring-modded signal into the mix (osc2_ring_mod stays constant so the
        # ring character is continuous, but its audibility tracks the macro).
        ModConnection(
            name="ring_macro_osc2_level",
            source=MacroSource(name="ring_amount"),
            target=AutomationTarget(kind="synth", name="osc2_level"),
            amount=0.6,
            bipolar=False,
            mode="add",
        ),
        # Audio-rate detune mod: OscillatorSource at 180 Hz shakes
        # osc2_detune_cents by ±8 cents throughout, rising to ±26 cents at
        # the apex via the fm_amount macro.  Two connections combine
        # additively — the fixed base plus the macro-scaled contribution.
        ModConnection(
            name="fm_detune_base",
            source=OscillatorSource(
                rate_hz=180.0,
                waveshape="sine",
            ),
            target=AutomationTarget(kind="synth", name="osc2_detune_cents"),
            amount=8.0,
            bipolar=True,
            mode="add",
        ),
        ModConnection(
            name="fm_apex_boost",
            source=MacroSource(name="fm_amount"),
            target=AutomationTarget(kind="synth", name="osc2_detune_cents"),
            amount=18.0,  # adds up to +18 cents of steady detune at apex
            bipolar=False,
            mode="add",
        ),
    ]
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "polyblep",
            "preset": "diva_bass_resonance",  # Newton ladder base
            "filter_topology": "ladder",
            "filter_solver": "newton",
            "quality": "divine",
            "transient_mode": "analog",
            "cutoff_hz": 600.0,
            "resonance_q": 10.0,
            "filter_drive": 0.6,
            "filter_env_amount": 1.2,
            "filter_env_decay": 0.9,
            "keytrack": 0.55,
            "bass_compensation": 0.35,
            # Give it melodic attack.
            "attack": 0.03,
            "decay": 0.45,
            "sustain_level": 0.65,
            "release": 0.6,
            # Osc2 on for sync/ring and audio-rate mod.
            "osc2_level": 0.35,
            "osc2_waveform": "saw",
            "osc2_sync": True,
            "osc2_detune_cents": 7.0,  # slight detune baseline; matrix rides this
            "osc2_ring_mod": 0.3,  # static ring character; macro rides osc2_level
            # Per-note voice distortion — subtle preamp warmth.
            "voice_dist_mode": "preamp",
            "voice_dist_drive": 0.22,
            "voice_dist_mix": 0.35,
            "voice_dist_tone": 0.4,
            # Analog character.
            "analog_jitter": 0.8,
            "voice_card_spread": 1.2,
            "osc_phase_noise": 0.12,
        },
        effects=_lead_effects(),
        envelope_humanize=None,
        velocity_humanize=None,
        velocity_group="ensemble",
        mix_db=-4.0,
        pan=0.15,
        sends=[
            VoiceSend(
                target="hall",
                send_db=-18.0,
                automation=[_build_hall_send_automation(base_db=-18.0)],
            ),
        ],
        automation=[
            _build_lead_resonance_automation(),
            _build_lead_cutoff_automation(),
        ],
        modulations=lead_modulations,
        drift_bus="ensemble",
        drift_bus_correlation=0.7,
    )

    # ---- Sub voice ----
    # Sallen-Key 2-pole, preamp voice_dist for tape warmth.
    score.add_voice(
        "sub",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "saw",
            "filter_topology": "sallen_key",
            "quality": "great",
            "transient_mode": "analog",
            "cutoff_hz": 300.0,
            "resonance_q": 1.2,
            "filter_drive": 0.35,
            "filter_env_amount": 1.0,
            "filter_env_decay": 0.4,
            "keytrack": 0.15,
            "attack": 0.03,
            "decay": 0.5,
            "sustain_level": 0.7,
            "release": 0.45,
            "osc2_level": 0.35,
            "osc2_waveform": "saw",
            "osc2_detune_cents": 6.0,
            "voice_dist_mode": "preamp",
            "voice_dist_drive": 0.3,
            "voice_dist_mix": 0.5,
            "voice_dist_tone": 0.5,
            "analog_jitter": 0.6,
            "voice_card_spread": 1.0,
        },
        effects=_sub_effects(),
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(),
        velocity_group="ensemble",
        mix_db=-5.0,
        pan=0.0,
        sends=[
            VoiceSend(target="hall", send_db=-22.0),
        ],
        automation=[
            _build_sub_cutoff_automation(),
        ],
        drift_bus="ensemble",
        drift_bus_correlation=0.9,
    )

    # ---- Breath voice ----
    # Additive engine with flow exciter.  Sparse partials plus organic breath.
    score.add_voice(
        "breath",
        synth_defaults={
            "engine": "additive",
            "partials": [
                {"ratio": 1.0, "amp": 0.45},
                {"ratio": 2.0, "amp": 0.28},
                {"ratio": 3.0, "amp": 0.12},
                {"ratio": 5.0, "amp": 0.08},
                {"ratio": 7.0, "amp": 0.05},
            ],
            "noise_amount": 0.55,
            "noise_mode": "flow",
            "flow_density": 0.08,
            "attack": 2.5,
            "decay": 0.5,
            "sustain_level": 0.9,
            "release": 3.0,
        },
        effects=_breath_effects(),
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(),
        mix_db=-10.0,
        pan=0.35,
        sends=[
            VoiceSend(target="hall", send_db=-10.0),
        ],
        automation=[
            # Slowly swell flow density — more organic in apex.
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="noise_amount"),
                segments=(
                    AutomationSegment(
                        start=0.0,
                        end=S3_START,
                        shape="linear",
                        start_value=0.35,
                        end_value=0.55,
                    ),
                    AutomationSegment(
                        start=S3_START,
                        end=S5_START,
                        shape="linear",
                        start_value=0.55,
                        end_value=0.65,
                    ),
                    AutomationSegment(
                        start=S5_START,
                        end=S6_START,
                        shape="linear",
                        start_value=0.65,
                        end_value=0.55,
                    ),
                    AutomationSegment(
                        start=S6_START,
                        end=TOTAL_DUR,
                        shape="linear",
                        start_value=0.55,
                        end_value=0.25,
                    ),
                ),
            ),
        ],
    )

    # ---- Arp voice (section 5 only) ----
    # Additive with phase_disperse spectral morph + sigma approximation.
    score.add_voice(
        "arp",
        synth_defaults={
            "engine": "additive",
            "partials": [
                {"ratio": 1.0, "amp": 0.6},
                {"ratio": 2.0, "amp": 0.4},
                {"ratio": 3.0, "amp": 0.3},
                {"ratio": 4.0, "amp": 0.22},
                {"ratio": 5.0, "amp": 0.16},
                {"ratio": 6.0, "amp": 0.12},
                {"ratio": 7.0, "amp": 0.09},
                {"ratio": 8.0, "amp": 0.07},
            ],
            "spectral_morph_type": "phase_disperse",
            "spectral_morph_amount": 0.55,
            "sigma_approximation": True,
            "noise_amount": 0.0,
            "attack": 0.08,
            "decay": 0.4,
            "sustain_level": 0.2,
            "release": 0.8,
        },
        effects=_arp_effects(),
        envelope_humanize=EnvelopeHumanizeSpec(preset="loose_pluck"),
        velocity_humanize=VelocityHumanizeSpec(),
        mix_db=-7.0,
        pan=-0.3,
        sends=[
            VoiceSend(target="hall", send_db=-8.0),
        ],
    )

    # ---- Populate the voices with notes ----
    _pad_chords(score)
    _breath_line(score)
    _sub_line(score)
    _arp_line(score)
    _add_lead_phrase(
        score,
        section_start=S3_START,
        melody=_LEAD_MELODY_S3,
        vibrato_depth=0.005,
    )
    _add_lead_phrase(
        score,
        section_start=S4_START,
        melody=_LEAD_MELODY_S4,
        vibrato_depth=0.006,
    )
    _add_lead_phrase(
        score,
        section_start=S5_START,
        melody=_LEAD_MELODY_S5,
        vibrato_depth=0.007,
    )
    _add_lead_phrase(
        score,
        section_start=S6_START,
        melody=_LEAD_MELODY_S6,
        vibrato_depth=0.004,
    )

    return score


PIECES: dict[str, PieceDefinition] = {
    "newton_bloom": PieceDefinition(
        name="newton_bloom",
        output_name="newton_bloom",
        build_score=build_score,
        sections=(
            PieceSection(
                label="Pad bloom", start_seconds=S1_START, end_seconds=S2_START
            ),
            PieceSection(
                label="Harmonic motion", start_seconds=S2_START, end_seconds=S3_START
            ),
            PieceSection(
                label="Lead enters", start_seconds=S3_START, end_seconds=S4_START
            ),
            PieceSection(
                label="Dialogue", start_seconds=S4_START, end_seconds=S5_START
            ),
            PieceSection(label="Apex", start_seconds=S5_START, end_seconds=S6_START),
            PieceSection(label="Decay", start_seconds=S6_START, end_seconds=TOTAL_DUR),
        ),
    ),
}
