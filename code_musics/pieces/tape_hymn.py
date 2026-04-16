"""Tape Hymn — slow meditative piece showcasing the new drum engine features.

72 BPM, 7-limit JI, ~90 seconds (27 bars).

Contrasts the resonator kick's warm bell-like ring with harsh 808 square-wave
metallic hats.  A four-part chorale pad enters gradually, breathing through
velocity-driven timbral arcs.

Features showcased: resonator kick body mode, square-wave oscillator hats,
FM snare with colored wire noise, velocity-to-timbre on all drums, sample
playback engine (cassette 808 kick layer), maracas preset, accelerating clap,
choke groups.

Section map:
  1-8    Awakening — sparse resonator kick, hats creep in, cassette layer joins
  9-20   Hymn — full drums, chorale enters bar 9 (low) and bar 13 (high),
         velocity crescendo bars 9-16, recede 17-20
  21-27  Decay — drums thin, chorale sustains and fades, long reverb tail
"""

from __future__ import annotations

from code_musics.automation import (
    AutomationSegment,
    AutomationSpec,
    AutomationTarget,
)
from code_musics.drum_helpers import add_drum_voice, setup_drum_bus
from code_musics.meter import Groove
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score, VoiceSend

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

BPM = 72.0
BEAT = 60.0 / BPM
BAR = 4.0 * BEAT
S16 = BEAT / 4.0
F0 = 55.0  # A1
TOTAL_BARS = 27

# Light eighth-note groove — enough to feel human, not enough to distort the hymn
GROOVE = Groove(
    subdivision="sixteenth",
    timing_offsets=(0.0, 0.08),
    velocity_weights=(1.0, 1.0),
)


def _pos(bar: int, beat: int = 1, n16: int = 0) -> float:
    """Bar/beat/16th -> seconds.  bar and beat are 1-based."""
    base = (bar - 1) * BAR + (beat - 1) * BEAT + n16 * S16
    return base + S16 * GROOVE.timing_offset_at(n16)


TOTAL_DUR = _pos(TOTAL_BARS + 1)

# ---------------------------------------------------------------------------
# Harmonic partials (7-limit JI from F0 = 55 Hz)
# ---------------------------------------------------------------------------

P1 = 1.0  # 55 Hz   A1
P2 = 2.0  # 110     A2
P3 = 3.0  # 165     ~E3
P4 = 4.0  # 220     A3
P5 = 5.0  # 275     ~C#4
P6 = 6.0  # 330     E4
P7 = 7.0  # 385     ~Bb4 (septimal 7th)
P8 = 8.0  # 440     A4
P9 = 9.0  # 495     ~D5 (9/8 above A4)
P10 = 10.0  # 550   ~C#5

# Chord voicings (as partial numbers)
# These are otonal — pure harmonic series stacks
CHORD_I = (P2, P3, P5, P7)  # A: root, 5th, M3, sept7
CHORD_IV = (P3, P4, P6, P9)  # ~D/A: rotated voicing
CHORD_V = (P3, P6, P8, P10)  # ~E/A: dominant feel
CHORD_I_HIGH = (P4, P6, P7, P8)  # A upper: 4th, 6th, sept7, octave

# Chord progression — 4-bar cycle
# Each entry: (partial_set, bars_duration)
PROGRESSION: list[tuple[tuple[float, ...], int]] = [
    (CHORD_I, 4),
    (CHORD_IV, 2),
    (CHORD_V, 2),
]

# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------


def _section(bar: int) -> str:
    if bar <= 8:
        return "awakening"
    if bar <= 20:
        return "hymn"
    return "decay"


def _velocity_arc(bar: int) -> float:
    """Dynamic arc: crescendo through hymn, recede toward decay.

    Returns a multiplier in [0.6, 1.3] for amp_db offset.
    """
    if bar <= 8:
        return 0.7 + 0.04 * (bar - 1)  # gentle rise 0.7 -> 0.98
    if bar <= 16:
        return 1.0 + 0.04 * (bar - 9)  # rise 1.0 -> 1.28
    if bar <= 20:
        return 1.28 - 0.07 * (bar - 16)  # fall 1.28 -> 1.0
    return 1.0 - 0.06 * (bar - 20)  # fall 1.0 -> 0.58


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_score() -> Score:
    score = Score(
        f0_hz=F0,
        sample_rate=44_100,
        master_effects=list(DEFAULT_MASTER_EFFECTS),
    )

    # -------------------------------------------------------------------
    # Send buses
    # -------------------------------------------------------------------

    drum_bus = setup_drum_bus(
        score,
        effects=[
            EffectSpec("compressor", {"preset": "kick_glue"}),
            EffectSpec(
                "saturation",
                {"drive": 0.15, "mix": 0.4, "mode": "tube"},
            ),
        ],
        return_db=0.0,
    )

    score.add_send_bus(
        "hall",
        effects=[
            EffectSpec(
                "reverb",
                {"room_size": 0.92, "damping": 0.45, "wet_level": 1.0},
            ),
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "lowpass",
                            "cutoff_hz": 4500.0,
                            "slope_db_per_oct": 12,
                        },
                        {
                            "kind": "highpass",
                            "cutoff_hz": 200.0,
                            "slope_db_per_oct": 12,
                        },
                    ],
                },
            ),
        ],
        return_db=-4.0,
    )

    # -------------------------------------------------------------------
    # Drum voices
    # -------------------------------------------------------------------

    # Resonator kick — the centerpiece
    add_drum_voice(
        score,
        "kick",
        engine="drum_voice",
        drum_bus=drum_bus,
        send_db=-2.0,
        effects=[EffectSpec("compressor", {"preset": "kick_punch"})],
        mix_db=3.0,
        synth_overrides={
            "tone_type": "resonator",
            "tone_decay_s": 0.400,
            "tone_sweep_ratio": 1.4,
            "tone_sweep_decay_s": 0.040,
            "tone_punch": 0.22,
            "tone_second_harmonic": 0.06,
            "exciter_level": 0.05,
            "noise_level": 0.02,
            "velocity_timbre_decay": 0.3,
            "velocity_timbre_brightness": 0.25,
        },
    )

    # Cassette 808 kick layer — lo-fi warmth underneath
    add_drum_voice(
        score,
        "cassette_kick",
        engine="sample",
        drum_bus=drum_bus,
        send_db=-6.0,
        mix_db=-3.0,
        synth_overrides={
            "sample_path": "../samples/Cassette808_SamplePack/Cassette808_Samples/Cassette808_BD01.wav",
            "root_freq": 55.0,
            "pitch_shift": True,
        },
    )

    # 808 closed hat — square wave, authentic ratios
    add_drum_voice(
        score,
        "closed_hat",
        engine="drum_voice",
        preset="808_closed_hat",
        drum_bus=drum_bus,
        send_db=-4.0,
        choke_group="hats",
        effects=[
            EffectSpec(
                "eq",
                {"bands": [{"kind": "high_shelf", "freq_hz": 6000.0, "gain_db": 2.0}]},
            ),
        ],
        mix_db=-8.0,
        synth_overrides={
            "noise_level": 0.20,
            "metallic_density": 0.7,
            "metallic_brightness": 0.65,
            "metallic_filter_q": 0.8,
            "velocity_timbre_brightness": 0.3,
            "velocity_timbre_decay": 0.2,
        },
    )

    # 808 open hat — choked by closed hat
    add_drum_voice(
        score,
        "open_hat",
        engine="drum_voice",
        preset="808_open_hat",
        drum_bus=drum_bus,
        send_db=-4.0,
        choke_group="hats",
        mix_db=-14.0,
        synth_overrides={
            "noise_level": 0.25,
            "metallic_density": 0.7,
            "metallic_brightness": 0.60,
            "metallic_filter_q": 0.8,
            "velocity_timbre_brightness": 0.3,
        },
    )
    score.voices["open_hat"].sends.append(VoiceSend(target="hall", send_db=-12.0))

    # FM snare — rich body with colored wire
    add_drum_voice(
        score,
        "snare",
        engine="drum_voice",
        drum_bus=drum_bus,
        send_db=-3.0,
        effects=[EffectSpec("compressor", {"preset": "snare_punch"})],
        mix_db=-4.0,
        synth_overrides={
            "tone_fm_ratio": 1.5,
            "tone_fm_index": 2.5,
            "tone_decay_s": 0.130,
            "noise_type": "comb",
            "noise_decay_s": 0.190,
            "noise_pre_noise_mode": "colored",
            "tone_level": 0.55,
            "noise_level": 0.45,
            "velocity_timbre_decay": 0.25,
            "velocity_timbre_brightness": 0.3,
            "velocity_timbre_harmonics": 0.2,
        },
    )
    score.voices["snare"].sends.append(VoiceSend(target="hall", send_db=-14.0))

    # Maracas — sixteenth-note texture
    add_drum_voice(
        score,
        "maracas",
        engine="drum_voice",
        preset="maracas",
        drum_bus=drum_bus,
        send_db=-6.0,
        mix_db=-14.0,
        synth_overrides={
            "velocity_timbre_brightness": 0.2,
        },
    )

    # Accelerating clap — accent on peak moments
    add_drum_voice(
        score,
        "clap",
        engine="drum_voice",
        preset="909_clap_authentic",
        drum_bus=drum_bus,
        send_db=-4.0,
        mix_db=-8.0,
    )
    score.voices["clap"].sends.append(VoiceSend(target="hall", send_db=-10.0))

    # -------------------------------------------------------------------
    # Tonal voices
    # -------------------------------------------------------------------

    # Chorale low — warm pad with slow filter sweep
    score.add_voice(
        "chorale_low",
        synth_defaults={
            "engine": "additive",
            "n_harmonics": 10,
            "harmonic_rolloff": 0.55,
            "brightness_tilt": -0.1,
            "attack": 0.8,
            "release": 2.0,
            "unison_voices": 2,
            "detune_cents": 4.0,
        },
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "lowpass",
                            "cutoff_hz": 2500.0,
                            "slope_db_per_oct": 12,
                        },
                        {"kind": "highpass", "cutoff_hz": 80.0, "slope_db_per_oct": 12},
                    ],
                },
            ),
        ],
        normalize_lufs=-24.0,
        mix_db=-6.0,
        sends=[VoiceSend(target="hall", send_db=-6.0)],
        automation=[
            bar_automation_lane(
                "cutoff_hz",
                [
                    (9, 800.0),
                    (14, 1800.0),
                    (18, 1400.0),
                    (22, 600.0),
                    (27, 400.0),
                ],
            ),
        ],
    )

    # Chorale high — thinner upper layer
    score.add_voice(
        "chorale_high",
        synth_defaults={
            "engine": "additive",
            "n_harmonics": 6,
            "harmonic_rolloff": 0.65,
            "brightness_tilt": 0.0,
            "attack": 1.2,
            "release": 2.5,
            "unison_voices": 2,
            "detune_cents": 5.0,
        },
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "lowpass",
                            "cutoff_hz": 3500.0,
                            "slope_db_per_oct": 12,
                        },
                        {
                            "kind": "highpass",
                            "cutoff_hz": 200.0,
                            "slope_db_per_oct": 12,
                        },
                    ],
                },
            ),
        ],
        normalize_lufs=-24.0,
        mix_db=-10.0,
        sends=[VoiceSend(target="hall", send_db=-4.0)],
    )

    # -------------------------------------------------------------------
    # Place notes
    # -------------------------------------------------------------------

    _place_kick(score)
    _place_cassette_kick(score)
    _place_closed_hat(score)
    _place_open_hat(score)
    _place_snare(score)
    _place_maracas(score)
    _place_clap(score)
    _place_chorale_low(score)
    _place_chorale_high(score)

    return score


# ---------------------------------------------------------------------------
# Automation helper
# ---------------------------------------------------------------------------


def bar_automation_lane(
    param: str,
    points: list[tuple[int, float]],
) -> AutomationSpec:
    """Build a voice-level automation lane from (bar, value) anchor pairs."""
    segments: list[AutomationSegment] = []
    for i in range(len(points) - 1):
        bar_a, val_a = points[i]
        bar_b, val_b = points[i + 1]
        t_a = _pos(bar_a)
        t_b = _pos(bar_b)
        segments.append(
            AutomationSegment(
                start=t_a, end=t_b, shape="linear", start_value=val_a, end_value=val_b
            )
        )
    first_val = points[0][1]
    return AutomationSpec(
        target=AutomationTarget(kind="synth", name=param),
        segments=tuple(segments),
        default_value=first_val,
    )


# ---------------------------------------------------------------------------
# Drum placement
# ---------------------------------------------------------------------------


def _place_kick(score: Score) -> None:
    for bar in range(1, TOTAL_BARS + 1):
        sec = _section(bar)
        vel = _velocity_arc(bar)

        if sec == "awakening":
            # Every 2 bars, beat 1 only
            if bar % 2 == 1:
                score.add_note(
                    "kick",
                    start=_pos(bar),
                    duration=0.6,
                    partial=0.5,
                    amp_db=-6.0 + 3.0 * (vel - 0.7),
                )
        elif sec == "hymn":
            # Beats 1 and 3
            for beat in (1, 3):
                db = -4.0 + 2.0 * (vel - 1.0)
                if beat == 3:
                    db -= 2.0  # beat 3 slightly softer
                score.add_note(
                    "kick", start=_pos(bar, beat), duration=0.5, partial=0.5, amp_db=db
                )
        else:  # decay
            if bar <= 24:
                # Beat 1 only, fading
                score.add_note(
                    "kick",
                    start=_pos(bar),
                    duration=0.6,
                    partial=0.5,
                    amp_db=-6.0 + 2.0 * (vel - 1.0),
                )
            elif bar <= 26 and bar % 2 == 1:
                score.add_note(
                    "kick", start=_pos(bar), duration=0.8, partial=0.5, amp_db=-8.0
                )


def _place_cassette_kick(score: Score) -> None:
    """Layer cassette 808 sample under the synth kick from bar 5 onward."""
    for bar in range(5, TOTAL_BARS + 1):
        sec = _section(bar)
        if sec == "awakening":
            if bar % 2 == 1:
                score.add_note(
                    "cassette_kick",
                    start=_pos(bar),
                    duration=0.5,
                    freq=55.0,
                    amp_db=-10.0,
                )
        elif sec == "hymn":
            for beat in (1, 3):
                score.add_note(
                    "cassette_kick",
                    start=_pos(bar, beat),
                    duration=0.4,
                    freq=55.0,
                    amp_db=-12.0,
                )
        elif bar <= 24:
            score.add_note(
                "cassette_kick", start=_pos(bar), duration=0.5, freq=55.0, amp_db=-14.0
            )


def _place_closed_hat(score: Score) -> None:
    """Eighth-note closed hat pulse with velocity accents."""
    for bar in range(3, TOTAL_BARS + 1):
        sec = _section(bar)
        vel = _velocity_arc(bar)

        if sec == "decay" and bar > 24:
            continue  # hats drop out late in decay

        for beat in range(1, 5):
            for eighth in (0, 2):  # 16th subdivisions 0 and 2 = eighth notes
                db_base = -14.0
                if eighth == 0:
                    db_base = -10.0  # downbeat accent

                # Section dynamics
                if sec == "awakening":
                    db_base -= 4.0
                elif sec == "decay":
                    db_base -= 3.0

                db = db_base + 2.0 * (vel - 1.0)

                score.add_note(
                    "closed_hat",
                    start=_pos(bar, beat, eighth),
                    duration=0.04,
                    freq=7000.0,
                    amp_db=db,
                )


def _place_open_hat(score: Score) -> None:
    """Open hat on select beats — choked by the closed hat."""
    for bar in range(9, 21):  # hymn section only
        vel = _velocity_arc(bar)
        # Beat 2 "and" (16th 2) and beat 4 "and" in fuller bars
        if bar >= 13:
            score.add_note(
                "open_hat",
                start=_pos(bar, 2, 2),
                duration=0.25,
                freq=7000.0,
                amp_db=-10.0 + 2.0 * (vel - 1.0),
            )
        if bar >= 17:
            score.add_note(
                "open_hat",
                start=_pos(bar, 4, 2),
                duration=0.25,
                freq=7000.0,
                amp_db=-12.0 + 2.0 * (vel - 1.0),
            )


def _place_snare(score: Score) -> None:
    """FM snare on beats 2 and 4 during hymn section."""
    for bar in range(9, 21):
        vel = _velocity_arc(bar)
        for beat in (2, 4):
            db = -6.0 + 2.0 * (vel - 1.0)
            if beat == 4:
                db -= 1.0  # ghost the 4 slightly
            score.add_note(
                "snare",
                start=_pos(bar, beat),
                duration=0.15,
                freq=200.0,
                amp_db=db,
            )


def _place_maracas(score: Score) -> None:
    """Sixteenth-note maracas texture during the heart of the hymn."""
    for bar in range(11, 19):
        vel = _velocity_arc(bar)
        for beat in range(1, 5):
            for n16 in range(4):
                # Skip some 16ths for breathing room
                if n16 == 1 and beat in (1, 3):
                    continue
                db = -18.0 + 1.5 * (vel - 1.0)
                # Subtle accent pattern: beat 1 louder, offbeats quieter
                if n16 == 0:
                    db += 2.0
                elif n16 == 2:
                    db += 0.5
                else:
                    db -= 1.0

                score.add_note(
                    "maracas",
                    start=_pos(bar, beat, n16),
                    duration=0.06,
                    freq=10000.0,
                    amp_db=db,
                )


def _place_clap(score: Score) -> None:
    """Accelerating 909 clap accents on peak moments."""
    # Clap hits at structural peaks
    peak_hits = [
        (12, 2, -6.0),  # building
        (14, 4, -8.0),  # accent
        (16, 2, -4.0),  # peak of the crescendo
        (16, 4, -6.0),
        (20, 2, -8.0),  # end of hymn
    ]
    for bar, beat, db in peak_hits:
        score.add_note(
            "clap",
            start=_pos(bar, beat),
            duration=0.12,
            freq=1200.0,
            amp_db=db,
        )


# ---------------------------------------------------------------------------
# Chorale placement
# ---------------------------------------------------------------------------


def _chord_at_bar(bar: int) -> tuple[float, ...]:
    """Return the chord partials for a given bar based on the progression cycle."""
    cycle_bar = (bar - 9) % 8  # 8-bar progression cycle starting at bar 9
    elapsed = 0
    for partials, dur_bars in PROGRESSION:
        if cycle_bar < elapsed + dur_bars:
            return partials
        elapsed += dur_bars
    return PROGRESSION[0][0]


def _place_chorale_low(score: Score) -> None:
    """Sustained chorale pad — enters bar 9, fades through decay."""
    for bar in range(9, TOTAL_BARS + 1):
        chord = _chord_at_bar(bar)
        vel = _velocity_arc(bar)
        sec = _section(bar)

        # Note duration = 1 bar (legato, overlapping with attack)
        dur = BAR + 0.2  # slight overlap for legato
        db_base = -16.0 if sec == "hymn" else -20.0
        db = db_base + 2.0 * (vel - 1.0)

        # Decay section: fade out progressively
        if sec == "decay":
            db -= 2.0 * (bar - 20)

        for partial in chord:
            score.add_note(
                "chorale_low",
                start=_pos(bar),
                duration=dur,
                partial=partial,
                amp_db=db - 2.0,  # spread voices slightly
            )


def _place_chorale_high(score: Score) -> None:
    """Upper chorale layer — enters bar 13, thinner voicing."""
    for bar in range(13, TOTAL_BARS + 1):
        sec = _section(bar)
        if sec == "decay" and bar > 23:
            continue  # drop out before end

        chord = _chord_at_bar(bar)
        vel = _velocity_arc(bar)
        dur = BAR + 0.3

        db = -20.0 + 2.0 * (vel - 1.0)
        if sec == "decay":
            db -= 3.0 * (bar - 20)

        # Use only top 2 notes of the chord, up an octave
        for partial in chord[-2:]:
            score.add_note(
                "chorale_high",
                start=_pos(bar),
                duration=dur,
                partial=partial * 2.0,  # up an octave
                amp_db=db,
            )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

PIECES: dict[str, PieceDefinition] = {
    "tape_hymn": PieceDefinition(
        name="tape_hymn",
        output_name="tape_hymn",
        build_score=build_score,
        sections=(
            PieceSection("awakening", _pos(1), _pos(9)),
            PieceSection("hymn", _pos(9), _pos(21)),
            PieceSection("decay", _pos(21), _pos(TOTAL_BARS + 1)),
        ),
    ),
}
