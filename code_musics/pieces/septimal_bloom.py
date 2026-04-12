"""Septimal Bloom — slow bloom in 7-limit JI for two Surge XT voices.

Form: A (otonal, warm) → B (utonal, elegiac) → A' (return, dissolve).
~2:50 duration.  The harmonic seventh (7/4) is the emotional thread.

Voice layout:
  - pad:    Surge XT (Classic saw, 3-voice unison detune, LP Vintage Ladder)
  - melody: Surge XT (Sine, 2-voice shimmer, brighter filter, faster attack)

Both voices route to a shared "hall" send bus (bricasti or native reverb +
delay).  Master bus has gentle saturation for glue.

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

# Section boundaries (seconds)
A_START: float = 0.0
B_START: float = 60.0
A2_START: float = 110.0
PIECE_END: float = 175.0

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


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------


def _hall_reverb() -> EffectSpec:
    if BRICASTI_IR_DIR.exists():
        return EffectSpec(
            "bricasti",
            {"ir_name": "1 Halls 07 Large & Dark", "wet": 0.22, "lowpass_hz": 6000},
            automation=[
                AutomationSpec(
                    target=AutomationTarget(kind="control", name="wet"),
                    segments=(
                        AutomationSegment(
                            start=0.0,
                            end=50.0,
                            shape="linear",
                            start_value=0.18,
                            end_value=0.25,
                        ),
                        AutomationSegment(
                            start=50.0,
                            end=85.0,
                            shape="linear",
                            start_value=0.25,
                            end_value=0.38,
                        ),
                        AutomationSegment(
                            start=85.0,
                            end=130.0,
                            shape="linear",
                            start_value=0.38,
                            end_value=0.28,
                        ),
                        AutomationSegment(
                            start=130.0,
                            end=175.0,
                            shape="linear",
                            start_value=0.28,
                            end_value=0.35,
                        ),
                    ),
                ),
            ],
        )
    return EffectSpec(
        "reverb",
        {"room_size": 0.82, "damping": 0.45, "wet_level": 0.22},
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="control", name="wet_level"),
                segments=(
                    AutomationSegment(
                        start=0.0,
                        end=50.0,
                        shape="linear",
                        start_value=0.18,
                        end_value=0.25,
                    ),
                    AutomationSegment(
                        start=50.0,
                        end=85.0,
                        shape="linear",
                        start_value=0.25,
                        end_value=0.38,
                    ),
                    AutomationSegment(
                        start=85.0,
                        end=130.0,
                        shape="linear",
                        start_value=0.38,
                        end_value=0.28,
                    ),
                    AutomationSegment(
                        start=130.0,
                        end=175.0,
                        shape="linear",
                        start_value=0.28,
                        end_value=0.35,
                    ),
                ),
            ),
        ],
    )


def _hall_delay() -> EffectSpec:
    return EffectSpec("delay", {"delay_seconds": 0.42, "feedback": 0.15, "mix": 0.14})


def _master_tape() -> EffectSpec:
    """Chow Tape Model on the master bus for analog glue, or native saturation fallback."""
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
        f0=F0_HZ,
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
                    AutomationSegment(
                        start=0.0,
                        end=60.0,
                        shape="linear",
                        start_value=-4.0,
                        end_value=-4.0,
                    ),
                    AutomationSegment(
                        start=60.0,
                        end=85.0,
                        shape="linear",
                        start_value=-4.0,
                        end_value=-1.0,
                    ),
                    AutomationSegment(
                        start=85.0,
                        end=110.0,
                        shape="linear",
                        start_value=-1.0,
                        end_value=-3.0,
                    ),
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
            "mpe": False,  # global pitch bend — loveless-style glide between chords
        },
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        normalize_lufs=-20.0,
        sends=[pad_send],
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

    # -- Section A (0:00–1:00) — otonal, warm, bittersweet -------------------
    _write_section_a(score)

    # -- Section B (1:00–1:50) — utonal, elegiac, tonic drift ----------------
    _write_section_b(score)

    # -- Section A' (1:50–2:40) — return, dissolve ---------------------------
    _write_section_a_prime(score)

    return score


# ---------------------------------------------------------------------------
# Section writers
# ---------------------------------------------------------------------------


def _pad_chord(
    score: Score,
    start: float,
    partials: list[float],
    dur: float,
    amp_db: float,
    vel: float = 0.7,
) -> None:
    for partial in partials:
        score.add_note(
            "pad",
            partial=partial,
            start=start,
            duration=dur,
            amp_db=amp_db,
            velocity=vel,
        )


_MAX_GLIDE_S = 0.5  # cap bend duration — multi-second bends are distracting


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
    """Write a melody note, optionally with a pitch glide from a nearby partial."""
    pm = None
    if glide_from is not None:
        gt = glide_time if glide_time is not None else min(dur * 0.35, _MAX_GLIDE_S)
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
        start=start,
        duration=dur,
        amp_db=amp_db,
        velocity=vel,
        pitch_motion=pm,
    )


def _write_section_a(score: Score) -> None:
    # -- Pass 1: sparse bloom (0–34s) ----------------------------------------
    # I — bass alone first, upper voices join
    _pad_chord(score, 0.0, [1], 8.0, amp_db=-10, vel=0.65)
    _pad_chord(score, 4.0, [5 / 2, 3], 4.0, amp_db=-9, vel=0.7)

    # iv7
    _pad_chord(score, 8.0, [4 / 3, 7 / 3, 8 / 3], 8.0, amp_db=-9)

    # vi — add 5 (major 3rd two octaves up, 550 Hz) for first hint of shimmer
    _pad_chord(score, 16.0, [5 / 3, 2, 7 / 2, 5], 8.0, amp_db=-9)

    # I7 — add sub-octave for weight
    _pad_chord(score, 24.0, [1 / 2, 1, 5 / 4, 3 / 2, 7 / 4], 10.0, amp_db=-9)

    # -- Pass 2: fuller voicings with extended range (34–60s) ----------------
    # I — add 4 (A4) and 6 (E5, 660 Hz third harmonic) for sparkle
    _pad_chord(score, 34.0, [1, 5 / 4, 5 / 2, 3, 4, 6], 8.0, amp_db=-8, vel=0.75)
    # iv7 — add 14 / 3 (7th harmonic of 4/3 root, ~513 Hz)
    _pad_chord(score, 42.0, [4 / 3, 7 / 3, 8 / 3, 4, 14 / 3], 8.0, amp_db=-8, vel=0.75)
    # vi — add 5 again + 10 / 3 is now 3 to avoid the clash, but add higher 5
    _pad_chord(score, 50.0, [5 / 3, 2, 7 / 2, 3, 5], 8.0, amp_db=-8, vel=0.75)
    # I7 — sub + shimmer: 1/2 weight, 7/2 (sept 7th high)
    _pad_chord(
        score, 58.0, [1 / 2, 1, 5 / 4, 3 / 2, 7 / 4, 7 / 2], 8.0, amp_db=-8, vel=0.75
    )

    # -- Melody (enters ~10s) ------------------------------------------------
    # Palette per chord:
    #   I  -> harmonic series of 1: 2, 5/2, 3, 7/2, 4, 9/2
    #   iv7 -> overtones of 4/3: 4/3, 5/3, 2, 7/3, 8/3, 3, 10/3
    #   vi  -> 5/3, 2, 5/2, 3, 10/3, 7/2
    #   I7  -> rich — almost anything harmonic

    # Pass 1 — gentle entrance over iv7, a few tentative notes
    _m(score, 10.5, 8 / 3, 1.2, amp_db=-9, vel=0.7)  # D4, quiet
    _m(score, 12.0, 7 / 3, 0.8, amp_db=-10, vel=0.65)  # grace down
    _m(score, 13.2, 3, 2.8, amp_db=-8, vel=0.75, glide_from=7 / 3)  # settle on E4

    # over vi — slightly more present
    _m(score, 17.0, 7 / 2, 1.5, amp_db=-7, vel=0.8)  # sept 7th
    _m(score, 18.8, 3, 0.6, amp_db=-9, vel=0.7)  # flicker down
    _m(score, 19.6, 10 / 3, 0.9, amp_db=-8, vel=0.75)  # passing tone
    _m(score, 20.8, 5 / 2, 2.5, amp_db=-8, vel=0.75, glide_from=10 / 3)  # glide to C#4
    _m(score, 23.5, 2, 0.7, amp_db=-10, vel=0.65)  # quick A3

    # over I7 — the sept 7th arrives naturally
    _m(score, 25.0, 7 / 4, 3.5, amp_db=-7, vel=0.8)  # sept 7th, long hold
    _m(
        score, 29.0, 3 / 2, 1.0, amp_db=-9, vel=0.7, glide_from=7 / 4
    )  # glide down to E3
    _m(score, 30.5, 5 / 4, 0.7, amp_db=-10, vel=0.65)  # C#3
    _m(score, 31.5, 3 / 2, 2.0, amp_db=-8, vel=0.75)  # back to E3
    # silence — breathing space (32-35s)

    # Pass 2 — more confident, wider range, wandering
    # over I (fuller)
    _m(score, 35.5, 4, 0.8, amp_db=-6, vel=0.85)  # A4, reaching up
    _m(score, 36.5, 9 / 2, 0.6, amp_db=-8, vel=0.7)  # flash above
    _m(score, 37.2, 4, 1.5, amp_db=-6, vel=0.85, glide_from=9 / 2)  # settle back
    _m(score, 39.0, 7 / 2, 0.7, amp_db=-7, vel=0.8)  # sept 7th
    _m(score, 40.0, 3, 1.8, amp_db=-7, vel=0.8, glide_from=7 / 2)  # glide to E4

    # over iv7 (fuller) — melody dances in 4/3 overtones
    _m(score, 42.5, 10 / 3, 1.0, amp_db=-7, vel=0.78)
    _m(score, 43.8, 8 / 3, 0.8, amp_db=-8, vel=0.75)
    _m(score, 44.9, 3, 1.5, amp_db=-7, vel=0.8, glide_from=8 / 3)
    _m(score, 46.8, 7 / 3, 1.0, amp_db=-8, vel=0.75)
    _m(score, 48.2, 2, 1.6, amp_db=-7, vel=0.8, glide_from=7 / 3)  # glide to A3

    # over vi — longer held tones, more space
    _m(score, 50.5, 7 / 2, 2.5, amp_db=-6, vel=0.85)  # sept 7th, exposed
    _m(score, 53.5, 3, 0.5, amp_db=-9, vel=0.7)  # grace note
    _m(score, 54.3, 5 / 2, 1.8, amp_db=-7, vel=0.78)

    # bridge — sept 7th held, gliding slightly sharp then back
    _m(score, 56.5, 7 / 4, 5.5, amp_db=-7, vel=0.8, glide_from=5 / 2)


def _write_section_b(score: Score) -> None:
    # -- Chords: utonal + tonic drift (60–110s) ------------------------------

    # I utonal — hollow minor + high octave doubling for width
    _pad_chord(score, 60.0, [1, 8 / 5, 4 / 3, 2, 16 / 5], 12.0, amp_db=-9, vel=0.7)

    # VII — major chord on the harmonic 7th (tonic = 7/4)
    # Fullest chord in the piece: sub (7/8), root, third, fifth, shimmer (7)
    _pad_chord(
        score, 72.0, [7 / 8, 7 / 4, 35 / 16, 21 / 8, 7], 12.0, amp_db=-8, vel=0.75
    )

    # bVI — gentle rest + octave doublings above
    _pad_chord(score, 84.0, [8 / 5, 2, 12 / 5, 16 / 5, 4], 12.0, amp_db=-9, vel=0.7)

    # V7 — septimal dominant, add sub (3/4) + high 3 for pull
    _pad_chord(
        score, 96.0, [3 / 4, 3 / 2, 15 / 8, 9 / 4, 21 / 8, 3], 14.0, amp_db=-8, vel=0.75
    )

    # -- Melody: exposed, wandering, more breath between phrases ---------------
    # Palettes:
    #   I utonal (1, 8/5, 4/3): 1, 4/3, 8/5, 2, 8/3
    #   VII (7/4, 35/16, 21/8): 7/4, 7/2, 21/8, 35/16, 3
    #   bVI (8/5, 2, 12/5): 8/5, 2, 12/5, 3, 16/5
    #   V7 (3/2, 15/8, 9/4, 21/8): 3/2, 2, 9/4, 3, 15/8, 21/8

    # over I utonal — melody emerges from the bridge 7/4, drifts into hollow space
    _m(score, 62.0, 2, 0.6, amp_db=-8, vel=0.7)  # A3, tentative
    _m(
        score, 63.0, 8 / 3, 2.0, amp_db=-6, vel=0.85, glide_from=2
    )  # slow glide up to D4
    _m(score, 65.5, 8 / 5, 0.4, amp_db=-9, vel=0.65)  # flicker: Ab3
    _m(score, 66.2, 2, 1.5, amp_db=-7, vel=0.78)  # back to A3
    _m(score, 68.2, 4 / 3, 0.7, amp_db=-9, vel=0.7)  # D3, dip low
    _m(score, 69.2, 8 / 5, 2.2, amp_db=-7, vel=0.78, glide_from=4 / 3)  # glide up
    # silence (71.4-72.5)

    # over VII (tonic=7/4) — the reaching moment, higher register
    _m(score, 72.8, 7 / 2, 1.2, amp_db=-5, vel=0.85)  # sept 7th high, exposed
    _m(score, 74.2, 35 / 16, 0.5, amp_db=-7, vel=0.75)  # the compound interval
    _m(score, 75.0, 21 / 8, 1.8, amp_db=-6, vel=0.8, glide_from=35 / 16)  # glide
    _m(score, 77.2, 3, 0.4, amp_db=-8, vel=0.7)  # passing E4
    _m(score, 77.8, 7 / 2, 2.5, amp_db=-5, vel=0.85, glide_from=3)  # reach back up
    _m(score, 80.8, 7 / 4, 0.6, amp_db=-8, vel=0.7)  # drop an octave
    _m(score, 81.8, 21 / 8, 1.5, amp_db=-6, vel=0.8, glide_from=7 / 4)  # slow climb
    # silence (83.3-84.5)

    # over bVI — calmer, descending
    _m(score, 84.8, 12 / 5, 2.0, amp_db=-6, vel=0.8)  # Eb4-ish
    _m(score, 87.2, 2, 0.5, amp_db=-8, vel=0.7)  # A3
    _m(score, 88.2, 12 / 5, 0.6, amp_db=-9, vel=0.65)  # back up briefly
    _m(score, 89.0, 8 / 5, 1.8, amp_db=-7, vel=0.78, glide_from=12 / 5)  # sink
    _m(score, 91.2, 3, 0.6, amp_db=-9, vel=0.7)  # flicker high
    _m(score, 91.5, 2, 2.5, amp_db=-7, vel=0.78, glide_from=3)  # long settle on A3
    # silence (94.0-96.0)

    # over V7 — pulling home, melody stays in chord palette (avoid 7/4 vs 15/8 clash)
    _m(score, 96.5, 9 / 4, 1.2, amp_db=-6, vel=0.82)  # B3
    _m(score, 98.0, 3, 0.8, amp_db=-7, vel=0.78)  # E4
    _m(score, 99.0, 21 / 8, 1.2, amp_db=-7, vel=0.78)  # sept compound
    _m(score, 100.5, 2, 1.5, amp_db=-6, vel=0.82, glide_from=21 / 8)  # glide to A3
    _m(score, 102.5, 9 / 4, 0.8, amp_db=-7, vel=0.78)
    _m(score, 103.5, 3, 5.0, amp_db=-6, vel=0.8, glide_from=9 / 4)  # bridge: sept 7th


def _write_section_a_prime(score: Score) -> None:
    # -- Chords: A' starts full (matching A pass 2 extended), then thins ------
    # I — full, with shimmer from A pass 2
    _pad_chord(score, 110.0, [1, 5 / 4, 5 / 2, 3, 4, 6], 8.0, amp_db=-8, vel=0.75)
    # iv7 — shimmer but slightly less
    _pad_chord(score, 118.0, [4 / 3, 7 / 3, 8 / 3, 4, 14 / 3], 8.0, amp_db=-9, vel=0.7)
    # vi — thinning: drop the highest, keep 5
    _pad_chord(score, 126.0, [5 / 3, 2, 7 / 2, 3, 5], 8.0, amp_db=-9, vel=0.7)
    # I7 — no shimmer, just the core + sub for weight
    _pad_chord(score, 134.0, [1 / 2, 1, 5 / 4, 3 / 2, 7 / 4], 8.0, amp_db=-10, vel=0.65)
    # Dissolve — bare root + fifth, extended tail
    _pad_chord(score, 142.0, [1, 3], 20.0, amp_db=-12, vel=0.55)

    # -- Melody: echoes A but more fragmented, landing on 7/6 -----------------

    # over I — familiar opening, slightly varied
    _m(score, 111.5, 4, 0.7, amp_db=-6, vel=0.85)  # A4
    _m(score, 112.5, 9 / 2, 0.3, amp_db=-8, vel=0.7)  # flash above
    _m(score, 113.0, 7 / 2, 1.5, amp_db=-6, vel=0.83, glide_from=9 / 2)  # settle
    _m(score, 114.8, 3, 0.5, amp_db=-8, vel=0.75)
    _m(score, 115.6, 5 / 2, 1.8, amp_db=-7, vel=0.78, glide_from=3)  # glide down

    # over iv7 — shorter, pulling inward
    _m(score, 118.5, 8 / 3, 1.0, amp_db=-7, vel=0.78)
    _m(score, 119.8, 7 / 3, 0.6, amp_db=-8, vel=0.75)
    _m(score, 120.6, 2, 1.5, amp_db=-7, vel=0.78, glide_from=7 / 3)
    _m(score, 122.5, 5 / 3, 0.8, amp_db=-9, vel=0.7)
    _m(score, 123.6, 4 / 3, 1.5, amp_db=-8, vel=0.75)  # sinks to D3

    # over vi — the sept 7th lingers
    _m(score, 126.0, 7 / 4, 2.5, amp_db=-7, vel=0.8)  # sept 7th
    _m(score, 129.0, 5 / 3, 0.4, amp_db=-9, vel=0.7)  # grace
    _m(score, 129.6, 3 / 2, 1.2, amp_db=-8, vel=0.75, glide_from=5 / 3)
    # silence (130.8-131.5)

    # over I7 — the landing: 7/6, not 1
    _m(score, 131.8, 7 / 4, 1.0, amp_db=-8, vel=0.75)  # approach
    _m(
        score, 133.0, 7 / 6, 4.5, amp_db=-7, vel=0.78, glide_from=7 / 4
    )  # slow glide to 7/6 — askew
    # silence (137.5-139)

    # dissolve — ghosts of the melody, barely there
    _m(score, 139.5, 3 / 2, 0.8, amp_db=-12, vel=0.55)  # whisper: E3
    _m(
        score, 140.8, 1, 8.0, amp_db=-14, vel=0.45, glide_from=3 / 2
    )  # glide down to root, fading
    # final echo — the sept 7th one last time, almost subliminal
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
