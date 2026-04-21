"""Septimal Bloom — slow bloom in 7-limit JI for two Surge XT voices.

Form: A (otonal, warm) → B (utonal, elegiac) → A' (return, dissolve).
~2:30 duration.  The harmonic seventh (7/4) is the emotional thread.

Voice layout:
  - pad/pad2: Surge XT (Classic saw, unison detune, LP Vintage Ladder, mpe=False
              for loveless-style tremolo-bar chord glides)
  - melody:   Surge XT (Sine, 2-voice shimmer, brighter filter, per-note MPE)
  - drone:    Surge XT (Sine, pure, barely audible high partials in B)

Both pad layers and melody route to a shared "hall" send bus (bricasti or
native reverb + delay + saturation).  Master bus has Chow Tape for glue.

Harmonic language: 7-limit JI centred on A2 (110 Hz).

Section A: I → iv7 → vi → I7 (otonal, bittersweet)
Section B: I utonal → VII (tonic=7/4) → bVI → V7 (hollow, reaching)
Section A': restatement, melody lands on 7/6, dissolve
"""

from __future__ import annotations

from code_musics.automation import AutomationSegment, AutomationSpec, AutomationTarget
from code_musics.humanize import EnvelopeHumanizeSpec, TimingHumanizeSpec
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import EffectSpec, Score, SendBusSpec, VoiceSend
from code_musics.synth import BRICASTI_IR_DIR, has_external_plugin

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

F0_HZ: float = 110.0

# Tempo scaling — all note times are authored in "slow" time then compressed
# by helpers.  Change _TS to adjust overall pace without rewriting every time.
_TS: float = 0.85  # ~15% faster than authored time

# Section boundaries (in real/scaled seconds)
A_START: float = 0.0
B_START: float = 60.0 * _TS
A2_START: float = 110.0 * _TS
PIECE_END: float = 175.0 * _TS

# -- Surge XT patches -------------------------------------------------------

_PAD_PARAMS: dict[str, float] = {
    "a_osc_1_type": 0.0,  # Classic saw
    "a_osc_1_shape": 0.5,  # saw position
    "a_osc_1_unison_voices": 0.15,  # 3 voices
    "a_osc_1_unison_detune": 0.12,  # ~12 cents — the warble
    "a_filter_1_type": 0.295,  # LP Vintage Ladder
    "a_filter_1_cutoff": 0.52,  # ~500 Hz — dark and warm
    "a_filter_1_resonance": 0.12,  # 12%
    "a_amp_eg_attack": 0.50,  # ~350 ms
    "a_amp_eg_decay": 0.55,  # ~550 ms
    "a_amp_eg_sustain": 0.80,  # 80%
    "a_amp_eg_release": 0.58,  # ~700 ms
}

_MELODY_PARAMS: dict[str, float] = {
    "a_osc_1_type": 0.05,  # Sine
    "a_osc_1_shape": 0.54,  # slight warmth
    "a_osc_1_unison_voices": 0.05,  # 2 voices
    "a_osc_1_unison_detune": 0.04,  # ~4 cents — shimmer not warble
    "a_filter_1_type": 0.295,  # LP Vintage Ladder
    "a_filter_1_cutoff": 0.60,  # ~1 kHz — brighter
    "a_filter_1_resonance": 0.06,  # 6%
    "a_amp_eg_attack": 0.40,  # ~140 ms
    "a_amp_eg_decay": 0.50,  # ~350 ms
    "a_amp_eg_sustain": 0.85,  # 85%
    "a_amp_eg_release": 0.55,  # ~550 ms
}

_PAD2_PARAMS: dict[str, float] = {
    "a_osc_1_type": 0.0,  # Classic saw
    "a_osc_1_shape": 0.5,
    "a_osc_1_unison_voices": 0.15,  # 3 voices
    "a_osc_1_unison_detune": 0.08,  # 8 cents — different from pad1's 12
    "a_filter_1_type": 0.295,  # LP Vintage Ladder
    "a_filter_1_cutoff": 0.48,  # slightly darker than pad1
    "a_filter_1_resonance": 0.10,  # 10%
    "a_amp_eg_attack": 0.52,  # slightly slower attack
    "a_amp_eg_decay": 0.55,
    "a_amp_eg_sustain": 0.80,
    "a_amp_eg_release": 0.60,  # slightly longer release
}

_DRONE_PARAMS: dict[str, float] = {
    "a_osc_1_type": 0.05,  # Sine
    "a_osc_1_unison_voices": 0.0,  # 1 voice — pure
    "a_osc_1_unison_detune": 0.0,
    "a_filter_1_type": 0.295,  # LP Vintage Ladder
    "a_filter_1_cutoff": 0.55,  # ~855 Hz
    "a_filter_1_resonance": 0.04,
    "a_amp_eg_attack": 0.65,  # ~1.4s — very slow fade in
    "a_amp_eg_decay": 0.55,
    "a_amp_eg_sustain": 0.90,
    "a_amp_eg_release": 0.65,  # ~1.4s — slow fade out
}


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------


def _seg(start: float, end: float, sv: float, ev: float) -> AutomationSegment:
    """Linear automation segment with times scaled by _TS."""
    return AutomationSegment(
        start=start * _TS, end=end * _TS, shape="linear", start_value=sv, end_value=ev
    )


def _hall_reverb() -> EffectSpec:
    reverb_automation = AutomationSpec(
        target=AutomationTarget(kind="control", name="wet"),
        segments=(
            _seg(0, 50, 0.18, 0.25),
            _seg(50, 85, 0.25, 0.38),
            _seg(85, 130, 0.38, 0.28),
            _seg(130, 175, 0.28, 0.35),
        ),
    )
    if BRICASTI_IR_DIR.exists():
        return EffectSpec(
            "bricasti",
            {"ir_name": "1 Halls 07 Large & Dark", "wet": 0.22, "lowpass_hz": 6000},
            automation=[reverb_automation],
        )
    # Fallback — native reverb needs "wet_level" not "wet"
    fallback_auto = AutomationSpec(
        target=AutomationTarget(kind="control", name="wet_level"),
        segments=reverb_automation.segments,
    )
    return EffectSpec(
        "reverb",
        {"room_size": 0.82, "damping": 0.45, "wet_level": 0.22},
        automation=[fallback_auto],
    )


def _hall_delay() -> EffectSpec:
    return EffectSpec("delay", {"delay_seconds": 0.42, "feedback": 0.15, "mix": 0.14})


def _master_tape() -> EffectSpec:
    if has_external_plugin("chow_tape"):
        return EffectSpec(
            "chow_tape",
            {"drive": 0.58, "saturation": 0.52, "bias": 0.50, "mix": 62.0},
        )
    return EffectSpec("saturation", {"preset": "tube_warm", "mix": 0.30, "drive": 1.35})


# ---------------------------------------------------------------------------
# Score builder
# ---------------------------------------------------------------------------


def build_score() -> Score:
    score = Score(
        f0_hz=F0_HZ,
        timing_humanize=TimingHumanizeSpec(preset="chamber"),
        send_buses=[
            SendBusSpec(
                name="hall",
                effects=[
                    _hall_reverb(),
                    _hall_delay(),
                    EffectSpec(
                        "saturation",
                        {"preset": "tube_warm", "mix": 0.18, "drive": 1.08},
                    ),
                ],
            ),
        ],
        master_effects=[_master_tape()],
    )

    # -- Voices --------------------------------------------------------------

    pad_send = VoiceSend(target="hall", send_db=-6.0)
    melody_send = VoiceSend(
        target="hall",
        send_db=-4.0,
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="control", name="send_db"),
                segments=(
                    _seg(0, 60, -4, -4),
                    _seg(60, 85, -4, -1),
                    _seg(85, 110, -1, -3),
                ),
            ),
        ],
    )

    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "surge_xt",
            "surge_params": _PAD_PARAMS,
            "tail_seconds": 3.0,
            "mpe": False,  # global-glide: tremolo-bar chord slides
        },
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        normalize_lufs=-20.0,
        sends=[pad_send],
    )

    pad2_send = VoiceSend(target="hall", send_db=-8.0)
    score.add_voice(
        "pad2",
        synth_defaults={
            "engine": "surge_xt",
            "surge_params": _PAD2_PARAMS,
            "tail_seconds": 3.0,
            "mpe": False,  # same global-glide as pad
        },
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        normalize_lufs=-22.0,
        mix_db=-3.0,
        pan=-0.15,
        sends=[pad2_send],
    )

    score.add_voice(
        "melody",
        synth_defaults={
            "engine": "surge_xt",
            "surge_params": _MELODY_PARAMS,
            "tail_seconds": 2.5,
        },
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        normalize_lufs=-18.0,
        mix_db=-2.0,
        sends=[melody_send],
    )

    drone_send = VoiceSend(target="hall", send_db=-2.0)
    score.add_voice(
        "drone",
        synth_defaults={
            "engine": "surge_xt",
            "surge_params": _DRONE_PARAMS,
            "tail_seconds": 4.0,
        },
        normalize_lufs=-24.0,
        mix_db=-8.0,
        pan=0.1,
        sends=[drone_send],
    )

    _write_section_a(score)
    _write_section_b(score)
    _write_section_a_prime(score)

    return score


# ---------------------------------------------------------------------------
# Helpers — all times are in "authored" seconds, scaled by _TS internally
# ---------------------------------------------------------------------------

_MAX_GLIDE_S = 0.5


def _pc(
    score: Score,
    start: float,
    partials: list[float],
    dur: float,
    amp_db: float,
    vel: float = 0.7,
) -> None:
    """Pad chord (single layer). Times scaled by _TS."""
    for p in partials:
        score.add_note(
            "pad",
            partial=p,
            start=start * _TS,
            duration=dur * _TS,
            amp_db=amp_db,
            velocity=vel,
        )


def _bp(
    score: Score,
    start: float,
    partials: list[float],
    dur: float,
    amp_db: float,
    vel: float = 0.7,
) -> None:
    """Both pad layers. pad2 offset 0.08s for interference."""
    _pc(score, start, partials, dur, amp_db, vel)
    for p in partials:
        score.add_note(
            "pad2",
            partial=p,
            start=start * _TS + 0.08,
            duration=dur * _TS,
            amp_db=amp_db - 2,
            velocity=vel * 0.9,
        )


def _m(
    score: Score,
    start: float,
    partial: float,
    dur: float,
    amp_db: float = -7.0,
    vel: float = 0.8,
    glide_from: float | None = None,
    glide_time: float | None = None,
) -> None:
    """Melody note with optional pitch glide. Times scaled by _TS."""
    pm = None
    if glide_from is not None:
        gt = (
            glide_time
            if glide_time is not None
            else min(dur * _TS * 0.35, _MAX_GLIDE_S)
        )
        pm = PitchMotionSpec(
            kind="ratio_glide",
            params={
                "start_ratio": glide_from / partial,
                "end_ratio": 1.0,
                "glide_duration": gt,
            },
        )
    score.add_note(
        "melody",
        partial=partial,
        start=start * _TS,
        duration=dur * _TS,
        amp_db=amp_db,
        velocity=vel,
        pitch_motion=pm,
    )


# ---------------------------------------------------------------------------
# Section writers — all times in "authored" (unscaled) seconds
# ---------------------------------------------------------------------------


def _write_section_a(score: Score) -> None:
    # -- Pass 1: sparse bloom (0–34s authored) --------------------------------
    _pc(score, 0.0, [1], 8.0, amp_db=-10, vel=0.65)
    _pc(score, 4.0, [5 / 2, 3], 4.0, amp_db=-9, vel=0.7)
    _pc(score, 8.0, [4 / 3, 7 / 3, 8 / 3], 8.0, amp_db=-9)
    _pc(score, 16.0, [5 / 3, 2, 7 / 2, 5], 8.0, amp_db=-9)
    _pc(score, 24.0, [1 / 2, 1, 5 / 4, 3 / 2, 7 / 4], 10.0, amp_db=-9)

    # -- Pass 2: both pads, extended range (34–60s) --------------------------
    _bp(score, 34.0, [1, 5 / 4, 5 / 2, 3, 4, 6], 8.0, amp_db=-8, vel=0.75)
    _bp(score, 42.0, [4 / 3, 7 / 3, 8 / 3, 4, 14 / 3], 8.0, amp_db=-8, vel=0.75)
    _bp(score, 50.0, [5 / 3, 2, 7 / 2, 3, 5], 8.0, amp_db=-8, vel=0.75)
    _bp(score, 58.0, [1 / 2, 1, 5 / 4, 3 / 2, 7 / 4, 7 / 2], 8.0, amp_db=-8, vel=0.75)

    # -- Melody (enters ~10s) ------------------------------------------------
    # Pass 1 — tentative over iv7
    _m(score, 10.5, 8 / 3, 1.2, amp_db=-9, vel=0.7)
    _m(score, 12.0, 7 / 3, 0.8, amp_db=-10, vel=0.65)
    _m(score, 13.2, 3, 2.8, amp_db=-8, vel=0.75, glide_from=7 / 3)

    # over vi
    _m(score, 17.0, 7 / 2, 1.5, amp_db=-7, vel=0.8)
    _m(score, 18.8, 3, 0.6, amp_db=-9, vel=0.7)
    _m(score, 19.6, 10 / 3, 0.9, amp_db=-8, vel=0.75)
    _m(score, 20.8, 5 / 2, 2.5, amp_db=-8, vel=0.75, glide_from=10 / 3)
    _m(score, 23.5, 2, 0.7, amp_db=-10, vel=0.65)

    # over I7
    _m(score, 25.0, 7 / 4, 3.5, amp_db=-7, vel=0.8)
    _m(score, 29.0, 3 / 2, 1.0, amp_db=-9, vel=0.7, glide_from=7 / 4)
    _m(score, 30.5, 5 / 4, 0.7, amp_db=-10, vel=0.65)
    _m(score, 31.5, 3 / 2, 2.0, amp_db=-8, vel=0.75)

    # Pass 2 — confident, wandering
    _m(score, 35.5, 4, 0.8, amp_db=-6, vel=0.85)
    _m(score, 36.5, 9 / 2, 0.6, amp_db=-8, vel=0.7)
    _m(score, 37.2, 4, 1.5, amp_db=-6, vel=0.85, glide_from=9 / 2)
    _m(score, 39.0, 7 / 2, 0.7, amp_db=-7, vel=0.8)
    _m(score, 40.0, 3, 1.8, amp_db=-7, vel=0.8, glide_from=7 / 2)

    # over iv7 — dancing
    _m(score, 42.5, 10 / 3, 1.0, amp_db=-7, vel=0.78)
    _m(score, 43.8, 8 / 3, 0.8, amp_db=-8, vel=0.75)
    _m(score, 44.9, 3, 1.5, amp_db=-7, vel=0.8, glide_from=8 / 3)
    _m(score, 46.8, 7 / 3, 1.0, amp_db=-8, vel=0.75)
    _m(score, 48.2, 2, 1.6, amp_db=-7, vel=0.8, glide_from=7 / 3)

    # over vi — held tones
    _m(score, 50.5, 7 / 2, 2.5, amp_db=-6, vel=0.85)
    _m(score, 53.5, 3, 0.5, amp_db=-9, vel=0.7)
    _m(score, 54.3, 5 / 2, 1.8, amp_db=-7, vel=0.78)

    # bridge
    _m(score, 56.5, 7 / 4, 5.5, amp_db=-7, vel=0.8, glide_from=5 / 2)


def _write_section_b(score: Score) -> None:
    # -- Chords (60–110s authored) --------------------------------------------
    _bp(score, 60.0, [1, 8 / 5, 4 / 3, 2, 16 / 5], 12.0, amp_db=-9, vel=0.7)
    _bp(score, 72.0, [7 / 8, 7 / 4, 35 / 16, 21 / 8, 7], 12.0, amp_db=-8, vel=0.75)
    _bp(score, 84.0, [8 / 5, 2, 12 / 5, 16 / 5, 4], 12.0, amp_db=-9, vel=0.7)
    _bp(
        score, 96.0, [3 / 4, 3 / 2, 15 / 8, 9 / 4, 21 / 8, 3], 14.0, amp_db=-8, vel=0.75
    )

    # Drone (times scaled manually since they're direct add_note)
    score.add_note(
        "drone",
        partial=7 / 2,
        start=62.0 * _TS,
        duration=45.0 * _TS,
        amp_db=-10,
        velocity=0.5,
    )
    score.add_note(
        "drone",
        partial=7,
        start=70.0 * _TS,
        duration=35.0 * _TS,
        amp_db=-14,
        velocity=0.4,
    )

    # -- Melody: exposed, wandering -------------------------------------------
    # over I utonal
    _m(score, 62.0, 2, 0.6, amp_db=-8, vel=0.7)
    _m(score, 63.0, 8 / 3, 2.0, amp_db=-6, vel=0.85, glide_from=2)
    _m(score, 65.5, 8 / 5, 0.6, amp_db=-9, vel=0.65)
    _m(score, 66.5, 2, 1.5, amp_db=-7, vel=0.78)
    _m(score, 68.2, 4 / 3, 0.7, amp_db=-9, vel=0.7)
    _m(score, 69.2, 8 / 5, 2.0, amp_db=-7, vel=0.78, glide_from=4 / 3)

    # over VII — reaching, highest point of the piece
    _m(score, 72.8, 7 / 2, 1.0, amp_db=-5, vel=0.85)
    _m(score, 74.0, 35 / 16, 0.6, amp_db=-7, vel=0.75)
    _m(score, 74.8, 21 / 8, 1.5, amp_db=-6, vel=0.8, glide_from=35 / 16)
    _m(score, 76.5, 7 / 2, 0.5, amp_db=-7, vel=0.78)
    _m(score, 77.2, 7, 0.8, amp_db=-8, vel=0.7)  # shimmer — highest melody note
    _m(score, 78.2, 7 / 2, 2.0, amp_db=-5, vel=0.85, glide_from=7)
    _m(score, 80.5, 7 / 4, 0.6, amp_db=-8, vel=0.7)
    _m(score, 81.5, 21 / 8, 1.5, amp_db=-6, vel=0.8, glide_from=7 / 4)

    # over bVI — calmer
    _m(score, 84.8, 12 / 5, 2.0, amp_db=-6, vel=0.8)
    _m(score, 87.0, 2, 0.6, amp_db=-8, vel=0.7)
    _m(score, 88.0, 12 / 5, 0.6, amp_db=-9, vel=0.65)
    _m(score, 89.0, 8 / 5, 1.8, amp_db=-7, vel=0.78, glide_from=12 / 5)
    _m(score, 91.2, 3, 0.6, amp_db=-9, vel=0.7)
    _m(score, 92.0, 2, 2.0, amp_db=-7, vel=0.78, glide_from=3)

    # over V7 — pulling home
    _m(score, 96.5, 9 / 4, 1.2, amp_db=-6, vel=0.82)
    _m(score, 98.0, 3, 0.8, amp_db=-7, vel=0.78)
    _m(score, 99.0, 21 / 8, 1.0, amp_db=-7, vel=0.78)
    _m(score, 100.2, 2, 1.5, amp_db=-6, vel=0.82, glide_from=21 / 8)
    _m(score, 102.0, 9 / 4, 0.8, amp_db=-7, vel=0.78)
    _m(score, 103.0, 3, 5.0, amp_db=-6, vel=0.8, glide_from=9 / 4)


def _write_section_a_prime(score: Score) -> None:
    # -- Chords: full then thinning (110–175s authored) -----------------------
    _bp(score, 110.0, [1, 5 / 4, 5 / 2, 3, 4, 6], 8.0, amp_db=-8, vel=0.75)
    _bp(score, 118.0, [4 / 3, 7 / 3, 8 / 3, 4, 14 / 3], 8.0, amp_db=-9, vel=0.7)
    _pc(score, 126.0, [5 / 3, 2, 7 / 2, 3, 5], 8.0, amp_db=-9, vel=0.7)
    _pc(score, 134.0, [1 / 2, 1, 5 / 4, 3 / 2, 7 / 4], 8.0, amp_db=-10, vel=0.65)
    _pc(score, 142.0, [1, 3], 20.0, amp_db=-12, vel=0.55)

    # -- Melody: echoes A, reaches high, lands on 7/6 -------------------------
    _m(score, 111.5, 4, 0.7, amp_db=-6, vel=0.85)
    _m(score, 112.5, 9 / 2, 0.6, amp_db=-8, vel=0.7)
    _m(score, 113.3, 7 / 2, 1.5, amp_db=-6, vel=0.83, glide_from=9 / 2)
    _m(score, 115.0, 3, 0.5, amp_db=-8, vel=0.75)
    _m(score, 115.8, 5 / 2, 1.5, amp_db=-7, vel=0.78, glide_from=3)

    # over iv7
    _m(score, 118.5, 8 / 3, 1.0, amp_db=-7, vel=0.78)
    _m(score, 119.8, 7 / 3, 0.6, amp_db=-8, vel=0.75)
    _m(score, 120.6, 2, 1.5, amp_db=-7, vel=0.78, glide_from=7 / 3)
    _m(score, 122.5, 5 / 3, 0.8, amp_db=-9, vel=0.7)
    _m(score, 123.6, 4 / 3, 1.5, amp_db=-8, vel=0.75)

    # over vi — the climactic reach before descent
    _m(score, 126.0, 7 / 4, 1.5, amp_db=-7, vel=0.8)
    _m(
        score, 127.8, 5, 1.0, amp_db=-6, vel=0.85
    )  # highest melody note in A' — reaching
    _m(score, 129.0, 7 / 2, 1.5, amp_db=-6, vel=0.82, glide_from=5)  # glide back down
    _m(score, 130.8, 3 / 2, 1.0, amp_db=-8, vel=0.75, glide_from=7 / 2)

    # over I7 — the landing
    _m(score, 132.5, 7 / 4, 1.0, amp_db=-8, vel=0.75)
    _m(score, 133.8, 7 / 6, 4.0, amp_db=-7, vel=0.78, glide_from=7 / 4)  # 7/6 — askew

    # dissolve
    _m(score, 139.0, 3 / 2, 0.8, amp_db=-12, vel=0.55)
    _m(score, 140.2, 1, 8.0, amp_db=-14, vel=0.45, glide_from=3 / 2)
    _m(score, 150.0, 7 / 4, 6.0, amp_db=-18, vel=0.35)  # ghost of the harmonic 7th


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

PIECES: dict[str, PieceDefinition] = {
    "septimal_bloom": PieceDefinition(
        name="septimal_bloom",
        output_name="septimal_bloom",
        build_score=build_score,
        sections=(
            PieceSection(label="A", start_seconds=A_START, end_seconds=B_START),
            PieceSection(label="B", start_seconds=B_START, end_seconds=A2_START),
            PieceSection(
                label="A_prime", start_seconds=A2_START, end_seconds=PIECE_END
            ),
        ),
    ),
}
