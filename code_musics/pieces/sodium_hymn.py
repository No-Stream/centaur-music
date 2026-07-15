"""Sodium Hymn — the 1-3-5-7-9-11 eikosany as a night-bus hymn.

Third and last panel of the CPS trilogy (hexany_garden -> ninth_wave ->
sodium_hymn).  The 3-of-6 Combination Product Set is the only CPS that
is its own mirror: fifteen full otonal tetrads and fifteen full utonal
tetrads, in exact duality.  The earlier panels lived in otonal light —
this one lives in the shadow half of the lattice and earns its light.

Structural gifts the piece leans on:

* An otonal tetrad O{x,y} and a utonal tetrad U{S} share two common
  tones exactly when {x,y} is inside S — the pivot mechanism.  Every
  chord change in this piece moves along those two-note hinges.
* O{9,11} sounds as 1:3:5:7 — hexany_garden's chord, note for note —
  but its pitches sit a 33/32 comma (~53 cents) off the tonic region.
  The old garden light appears here only as a detuned memory.
* The ten notes containing factor 11 are ninth_wave's 1-3-5-7-9 dekany
  transposed by 11; the otonal chords O{x,11} sound as pure 1-3-5-7-9
  harmony.  The middle of the piece walks that drowned dekany.
* Trilogy ending series: hexany_garden hung on 4:7; ninth_wave bloomed
  it into 4:5:6:7; sodium_hymn ends one rung further up the series, on
  5:7:9:11 (= O{1,3}, whose notes 1/1, 11/10, 7/5, 9/5 hold the tonic).

Euphony doctrine (the v2 rewrite):

* Octave-reduction is how you catalog a CPS, not how you voice it.
  Otonal stations are voiced as literal harmonic-series chords over a
  real bass root (1:3:5:7 spread wide, glowing); utonal stations are
  voiced open — no sustained adjacent seconds, fifths and sixths doing
  the work.  Hand-tuned voicings live in ``VOICINGS``.
* One tune.  The hymn theme (sigh / question / reach / settle, 4 bars)
  is stated by the ghost voice in S2 and restated in every section:
  on the comma-shifted light degrees in S4, augmented in the cathedral,
  fragmented as call-and-answer in S7, and complete one last time in
  S8 — where the settle lands on 11/10 instead of 1/1 and the reach
  note has risen 4.0 -> 4.125 -> 4.4 across the piece.
* Everything shares a clock.  A swung chord-tone arp is the coherence
  spine under the beat; bells land on the grid; only the voice floats.
* A quiet drone thread holds the station root at all times, so every
  chord is heard against ground.

Sound: Burial-inflected swung 2-step at 132 BPM with a Vangelis glow.
No four-on-floor.  Rain and vinyl crackle, tape haze, long dark hall.
The wordless ghost vocal (formant-morphing additive voice — strings
from one angle, a voice from another) is the centerpiece; bells ring
distant and reverb-drowned.  Bass is mid-harmonic warmth, not sub
pressure; F1 only at structural moments.

Tuning: eikosany over (1,3,5,7,9,11), normalized to 1*3*5, on
f0 = F2 ~ 87.31 Hz.  BPM = 132, 1 bar ~ 1.818 s, 214 bars ~ 6:29.

Form:
  bars   1- 16  S1 Rain         beatless; weather; drone ground; bells
                                hint the hymn's sigh
  bars  17- 40  S2 First voice  two full hymn statements over the home
                                shadow U{1,3,5,9}; at 33 the first
                                undecimal tint (U{1,3,9,11}) — the
                                voice slides 9/8 -> 11/10, a 40-cent
                                comma sigh; the arp ghosts in
  bars  41- 80  S3 Two-step     the beat materialises; arp spine, bass
                                roots; utonal walk with hymn fragments
  bars  81-104  S4 Light        O{9,11} voiced as a literal 1:3:5:7
                                series chord — the hexany-garden quote,
                                comma-shifted; the hymn on light degrees
  bars 105-136  S5 Cathedral    beat dissolves; deep shadow U{5,7,9,11};
                                the hymn augmented to double length,
                                bells shadowing a bar behind
  bars 137-138  S6 Blackness    two bars of near-silence
  bars 139-186  S7 Second wave  the 2-step returns evolved; dekany-walk
                                otonal series chords against utonal
                                answers; hymn fragments as call and
                                answer; full statement at the climax
  bars 187-214  S8 Dissolution  beat decays to ghosts; U{1,3,5,9} and
                                O{1,3} alternate around the pillars 1/1
                                and 9/5; the last hymn settles on 11/10
                                and the piece hangs on 5:7:9:11

Composed by Claude (Fable 5), July 2026.
"""

from __future__ import annotations

from itertools import combinations
from typing import cast

from code_musics.automation import (
    AutomationSegment,
    AutomationShape,
    AutomationSpec,
    AutomationTarget,
)
from code_musics.drum_helpers import add_drum_voice, setup_drum_bus
from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    VelocityHumanizeSpec,
)
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS, bricasti_or_reverb
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import EffectSpec, Score, VoiceSend
from code_musics.spectra import formant_morph, harmonic_spectrum
from code_musics.tuning import eikosany_tetrads

# ---------------------------------------------------------------------------
# Time and tuning
# ---------------------------------------------------------------------------

F0 = 87.3071  # F2
BPM = 132.0
BEAT = 60.0 / BPM
BAR = 4.0 * BEAT
SWING = 0.60  # position of the off-16th inside its eighth (0.5 = straight)

FACTORS: tuple[int, ...] = (1, 3, 5, 7, 9, 11)
_OTONAL_TETRADS, _UTONAL_TETRADS = eikosany_tetrads(FACTORS)
_PAIRS: list[tuple[int, int]] = list(combinations(FACTORS, 2))
_QUADS: list[tuple[int, int, int, int]] = list(combinations(FACTORS, 4))


def _otonal(x: int, y: int) -> tuple[float, float, float, float]:
    return _OTONAL_TETRADS[_PAIRS.index((x, y))]


def _utonal(a: int, b: int, c: int, d: int) -> tuple[float, float, float, float]:
    return _UTONAL_TETRADS[_QUADS.index((a, b, c, d))]


HOME = _utonal(1, 3, 5, 9)  # 1, 9/8, 3/2, 9/5 — the shadow home
TINT = _utonal(1, 3, 9, 11)  # 11/10, 99/80, 33/20, 9/5 — first 11
HEX_LIGHT = _otonal(9, 11)  # 33/32, 99/80, 231/160, 33/20 — sounds 1:3:5:7
DEEP = _utonal(5, 7, 9, 11)  # 33/32, 21/16, 231/160, 77/48 — cathedral
DARK_HOME = _utonal(1, 3, 5, 11)  # 1, 11/10, 11/8, 11/6 — home, 9 -> 11
FINAL_LIGHT = _otonal(1, 3)  # 1, 11/10, 7/5, 9/5 — sounds 5:7:9:11


def bar(n: float) -> float:
    """Seconds at the start of 1-indexed bar *n*."""
    return (n - 1.0) * BAR


def sw(bar_num: float, sixteenth: float) -> float:
    """Seconds at a swung sixteenth (0-15) inside 1-indexed *bar_num*.

    Even sixteenths sit on the eighth grid; odd sixteenths land at the
    SWING fraction of their eighth (0.5 would be straight time).
    """
    eighth_index = int(sixteenth) // 2
    is_off = int(sixteenth) % 2 == 1
    t = bar(bar_num) + eighth_index * (BEAT / 2.0)
    if is_off:
        t += SWING * (BEAT / 2.0)
    return t


S1_END = bar(17)
S2_END = bar(41)
S3_END = bar(81)
S4_END = bar(105)
S5_END = bar(137)
S6_END = bar(139)
S7_END = bar(187)
TOTAL_DUR = bar(215)

# ---------------------------------------------------------------------------
# Voicings — the euphony layer
# ---------------------------------------------------------------------------
#
# Each station gets a hand-tuned open voicing: (bass, pad tones, arp tones),
# all as partials of F0.  Otonal stations are literal series chords over
# their root (O{9,11} really is 1:3:5:7 over 33/32); utonal stations are
# spread so no sustained adjacent seconds survive.  The octave-reduced
# tetrads in HOME/TINT/... are the *material*; these are the *sounds*.

_R = 33.0 / 32.0  # the light root (and the comma)

# Arp voicings hold the pivot tone in the SAME register across adjacent
# stations (HOME->TINT keep 3.6, TINT->U(1,3,7,11) keep 2.2, ...), so
# chord changes are voice-led in the ear, not just on paper.
VOICINGS: dict[str, tuple[float, tuple[float, ...], tuple[float, ...]]] = {
    "HOME": (0.5, (1.0, 1.5, 2.25, 3.6), (1.5, 2.25, 3.0, 3.6)),
    "TINT": (0.45, (0.9, 1.1, 1.65, 2.475), (1.65, 2.2, 2.475, 3.6)),
    "U(1,3,7,11)": (0.7, (1.1, 1.925, 2.566667), (1.4, 1.925, 2.2, 2.566667)),
    "U(1,3,7,9)": (0.45, (0.9, 1.4, 2.1, 3.15), (1.4, 2.1, 2.8, 3.6)),
    # O{9,11}'s virtual fundamental is 33/40 (the {1,9,11} tone dropped
    # low): its notes are 0.825 x (1,3,5,7) in octave classes.  Voiced as
    # 4:5:6:7 over that root — ninth_wave's final chord, comma-shifted.
    # Bass sits an octave below the root (36 Hz felt, not heard).
    "HEX_LIGHT": (
        0.4125,
        (1.65, 2.0625, 2.475, 2.8875),  # 4:5:6:7 over 0.825
        (1.65, 2.0625, 2.475, 3.3),
    ),
    # O{5,11}'s fundamental is 11/12; notes are 0.9167 x (1,3,7,9).
    "O(5,11)": (
        0.458333,
        (0.916667, 1.833333, 2.75, 3.208333),  # 2:4:6:7 over the root
        (1.833333, 2.0625, 2.75, 3.208333),
    ),
    "DEEP": (0.721875, (1.03125, 1.3125, 1.604167, 2.0625), ()),
    "U(3,5,9,11)": (
        0.6875,
        (1.03125, 1.2375, 2.25, 2.75),
        (1.375, 2.0625, 2.475, 2.75),
    ),
    "DARK_HOME": (0.5, (1.0, 1.375, 1.833333, 2.2), ()),
    "O(1,11)": (0.55, (1.1, 1.833333, 2.566667, 3.3), (1.833333, 2.2, 2.566667, 3.3)),
    "O(3,11)": (0.55, (1.1, 1.375, 1.925, 2.475), (1.375, 1.925, 2.475, 2.75)),
    "O(7,11)": (
        0.641667,
        (1.283333, 1.925, 2.8875),  # 2:3:9 of the 77/120 series
        (1.283333, 1.925, 2.8875, 3.208333),
    ),
    "U(1,7,9,11)": (
        0.721875,
        (1.05, 1.283333, 1.65, 2.8875),
        (1.283333, 1.65, 2.1, 2.8875),
    ),
    "FINAL_LIGHT": (0.5, (1.4, 1.8, 2.2, 2.8), (1.4, 1.8, 2.2, 2.8)),
}

# Station root per bar (drives drone, bass, and tuned percussion).
# (start_bar, voicing_key); consulted with the latest entry <= bar.
ROOTS: list[tuple[float, str]] = [
    (1.0, "HOME"),
    (33.0, "TINT"),
    (41.0, "HOME"),
    (49.0, "TINT"),
    (57.0, "U(1,3,7,11)"),
    (65.0, "U(1,3,7,9)"),
    (73.0, "TINT"),
    (81.0, "HEX_LIGHT"),
    (89.0, "O(5,11)"),
    (97.0, "DEEP"),
    (105.0, "DEEP"),
    (117.0, "U(3,5,9,11)"),
    (125.0, "DARK_HOME"),
    (139.0, "O(1,11)"),
    (147.0, "O(3,11)"),
    (155.0, "O(5,11)"),
    (163.0, "O(7,11)"),
    (171.0, "HEX_LIGHT"),
    (179.0, "TINT"),
    (187.0, "HOME"),
    (191.0, "FINAL_LIGHT"),
    (195.0, "HOME"),
    (199.0, "FINAL_LIGHT"),
    (203.0, "HOME"),
    (207.0, "FINAL_LIGHT"),
]


def _voicing_at(bar_num: float) -> tuple[float, tuple[float, ...], tuple[float, ...]]:
    key = ROOTS[0][1]
    for start_b, k in ROOTS:
        if start_b <= bar_num:
            key = k
    return VOICINGS[key]


def _hold_ramp(
    target: AutomationTarget,
    points: list[tuple[float, float, float, AutomationShape]],
    default: float,
) -> AutomationSpec:
    """Replace-mode automation from (start, end, to_value, shape) rows,
    holding each reached value until the next ramp begins."""
    segments: list[AutomationSegment] = []
    prev_value = default
    prev_end = 0.0
    for start, end, to_value, shape in points:
        if start > prev_end and segments:
            segments.append(
                AutomationSegment(
                    start=prev_end, end=start, shape="hold", value=prev_value
                )
            )
        segments.append(
            AutomationSegment(
                start=start,
                end=end,
                shape=shape,
                start_value=prev_value,
                end_value=to_value,
            )
        )
        prev_value = to_value
        prev_end = end
    if prev_end < TOTAL_DUR:
        segments.append(
            AutomationSegment(
                start=prev_end, end=TOTAL_DUR, shape="hold", value=prev_value
            )
        )
    return AutomationSpec(target=target, segments=tuple(segments), mode="replace")


def _ghost_hall_ride() -> AutomationSpec:
    """The voice sits deep in the hall when alone, closer under the beat."""
    return _hold_ramp(
        AutomationTarget(kind="control", name="send_db"),
        [
            (S2_END - BAR, S2_END + 2 * BAR, -6.0, "linear"),
            (bar(101), S4_END, -2.5, "linear"),
            (S6_END, S6_END + 2 * BAR, -6.0, "linear"),
            (bar(183), S7_END, -2.5, "linear"),
        ],
        default=-3.5,
    )


def _pad_hall_ride() -> AutomationSpec:
    """The pad opens into the hall for the cathedral and the ending."""
    return _hold_ramp(
        AutomationTarget(kind="control", name="send_db"),
        [
            (S4_END - 2 * BAR, S4_END + 2 * BAR, -4.0, "linear"),
            (S6_END, S6_END + 2 * BAR, -8.0, "linear"),
            (bar(203), bar(209), -4.0, "linear"),
        ],
        default=-8.0,
    )


def _bass_cutoff_arc() -> AutomationSpec:
    """The bass opens across each wave and closes for the dissolution."""
    return _hold_ramp(
        AutomationTarget(kind="synth", name="cutoff_hz"),
        [
            (bar(45), bar(73), 900.0, "exp"),
            (bar(97), bar(105), 420.0, "exp"),
            (S6_END, bar(163), 1000.0, "exp"),
            (bar(187), bar(203), 450.0, "exp"),
        ],
        default=520.0,
    )


# ---------------------------------------------------------------------------
# Ghost vocal — formant-morphing additive voice
# ---------------------------------------------------------------------------

# Nearly flat source spectrum: the formant envelope does the sculpting
# (with a steep rolloff the partials carrying the 2-3 kHz singer formants
# are dead before the vowel weighting ever sees them).
_VOCAL_BASE = harmonic_spectrum(n_partials=26, harmonic_rolloff=0.78)


def _ghost_partials(
    freq_hz: float,
    vowels: list[str],
    morph_times: list[float] | None = None,
) -> list[dict]:
    """Formant-morph partials computed at the note's absolute frequency.

    Formants stay fixed in Hz while the pitch moves (how a real vocal
    tract works), so each note computes its own spectrum.  Envelope
    weights are normalized to the note's loudest formant peak, with a
    floor under the low partials — a real voice keeps its fundamental
    even when the formants sit far above it.
    """
    shaped = formant_morph(_VOCAL_BASE, freq_hz, list(vowels), morph_times)
    envelopes = [cast(list[dict[str, float]], p["envelope"]) for p in shaped]
    peak = max(point["value"] for envelope in envelopes for point in envelope)
    for p, envelope in zip(shaped, envelopes, strict=True):
        abs_freq = cast(float, p["ratio"]) * freq_hz
        if abs_freq < 260.0:
            floor = 0.80
        elif abs_freq < 480.0:
            floor = 0.35
        else:
            floor = 0.0
        for point in envelope:
            point["value"] = max(point["value"] / peak, floor)
    return shaped


def _sing(
    score: Score,
    *,
    start: float,
    duration: float,
    partial: float,
    vowels: list[str],
    morph_times: list[float] | None = None,
    amp_db: float = -13.0,
    velocity: float = 1.0,
    glide_from: float | None = None,
    vibrato: bool = True,
    breath: float = 0.11,
    attack: float = 0.30,
    release: float = 1.4,
    label: str | None = None,
) -> None:
    """One wordless sung note.  Glide notes approach from the previous
    pitch; sustained notes carry a late-blooming vibrato instead."""
    if glide_from is not None:
        motion: PitchMotionSpec | None = PitchMotionSpec.ratio_glide(
            start_ratio=glide_from / partial, end_ratio=1.0
        )
    elif vibrato:
        motion = PitchMotionSpec.vibrato(depth_ratio=0.005, rate_hz=5.1)
    else:
        motion = None
    score.add_note(
        "ghost",
        start=start,
        duration=duration,
        partial=partial,
        amp_db=amp_db,
        velocity=velocity,
        pitch_motion=motion,
        label=label,
        synth={
            "partials": _ghost_partials(F0 * partial, vowels, morph_times),
            "noise_amount": breath,
            "noise_mode": "flow",
            "flow_density": 0.25,
            "noise_bandwidth_hz": 160.0,
            "spectral_flicker": 0.1,
            "flicker_rate_hz": 1.5,
            "flicker_correlation": 0.7,
            "attack": attack,
            "decay": 0.5,
            "sustain_level": 0.85,
            "release": release,
        },
    )


# ---------------------------------------------------------------------------
# The hymn theme — sigh / question / reach / settle (16 beats)
# ---------------------------------------------------------------------------
#
# Each entry: (beat_offset, dur_beats, partial, vowels, glide_from, velocity).
# The reach note rises across the piece: 4.0 (home octave) -> 4.125 (light,
# 33/32 above) -> 4.4 (final, 11/10 above) — the series climbing.

HymnNote = tuple[float, float, float, list[str], float | None, float]

HYMN_HOME: list[HymnNote] = [
    (0.0, 1.5, 3.6, ["u", "o"], None, 0.9),
    (1.5, 0.5, 3.15, ["o"], 3.6, 0.72),
    (2.0, 2.0, 3.0, ["o", "a"], None, 0.95),
    (4.0, 1.0, 2.25, ["a"], None, 0.8),
    (5.0, 1.0, 2.333333, ["a", "e"], None, 0.85),
    (6.0, 2.0, 3.0, ["e", "o"], None, 0.9),
    (8.0, 1.5, 3.6, ["o", "a"], None, 0.95),
    (9.5, 2.5, 4.0, ["a", "e", "a"], 3.6, 1.0),
    (12.0, 1.0, 3.0, ["a", "o"], None, 0.85),
    (13.0, 1.0, 2.25, ["o"], None, 0.78),
    (14.0, 2.0, 2.0, ["o", "u"], None, 0.9),
]

HYMN_LIGHT: list[HymnNote] = [
    (0.0, 1.5, 3.2 * _R, ["u", "o"], None, 0.9),  # 3.3
    (1.5, 0.5, 3.15, ["o"], 3.2 * _R, 0.72),  # 63/40 passing, as at home
    (2.0, 2.0, 2.8 * _R, ["o", "a"], None, 0.95),  # 2.8875 = 7/2 of root /1.25
    (4.0, 1.0, 2.0 * _R, ["a"], None, 0.8),
    (5.0, 1.0, 2.545455 * _R, ["a", "e"], None, 0.85),  # 21/16 x2 inflection
    (6.0, 2.0, 2.8 * _R, ["e", "o"], None, 0.9),
    (8.0, 1.5, 3.2 * _R, ["o", "a"], None, 0.95),
    (9.5, 2.5, 4.0 * _R, ["a", "e", "a"], 3.2 * _R, 1.0),  # the reach: 4.125
    (12.0, 1.0, 2.8 * _R, ["a", "o"], None, 0.85),
    (13.0, 1.0, 2.4 * _R, ["o"], None, 0.78),  # 99/80 x2
    (14.0, 2.0, 2.0 * _R, ["o", "u"], None, 0.9),  # settle on the light root
]

HYMN_DEEP: list[HymnNote] = [
    (0.0, 1.5, 3.208333, ["u", "o"], None, 0.88),  # 77/48 x2
    (1.5, 0.5, 2.8875, ["o"], 3.208333, 0.7),
    (2.0, 2.0, 2.625, ["o", "u"], None, 0.92),  # 21/16 x2
    (4.0, 1.0, 2.0625, ["u"], None, 0.78),
    (5.0, 1.0, 2.75, ["u", "o"], None, 0.82),  # 11/8 x2 — undecimal color
    (6.0, 2.0, 2.625, ["o"], None, 0.88),
    (8.0, 1.5, 3.208333, ["o", "a"], None, 0.92),
    (9.5, 2.5, 4.125, ["a", "o", "u"], 3.208333, 0.95),
    (12.0, 1.0, 2.8875, ["o"], None, 0.82),
    (13.0, 1.0, 2.625, ["u"], None, 0.75),
    (14.0, 2.0, 2.0625, ["u", "o"], None, 0.88),
]

HYMN_FINAL: list[HymnNote] = [
    (0.0, 1.5, 3.6, ["u", "o"], None, 0.9),
    (1.5, 0.5, 3.15, ["o"], 3.6, 0.72),
    (2.0, 2.0, 2.8, ["o", "a"], None, 0.95),
    (4.0, 1.0, 2.25, ["a"], None, 0.8),
    (5.0, 1.0, 2.333333, ["a", "e"], None, 0.85),
    (6.0, 2.0, 2.8, ["e", "o"], None, 0.9),
    (8.0, 1.5, 3.6, ["o", "a"], None, 0.95),
    (9.5, 2.5, 4.4, ["a", "e", "a"], 3.6, 1.0),  # the reach becomes 11/10
    (12.0, 1.0, 2.8, ["a", "o"], None, 0.85),
    (13.0, 1.0, 2.25, ["o"], None, 0.78),
    (14.0, 3.0, 2.2, ["o", "u"], None, 0.92),  # settles on 11/10, not 1/1
]


def _hymn(
    score: Score,
    start_bar: float,
    notes: list[HymnNote],
    *,
    stretch: float = 1.0,
    amp_db: float = -13.5,
    breath: float = 0.16,
    release: float = 2.0,
) -> None:
    """State the hymn starting at *start_bar*, optionally time-stretched."""
    for beat_offset, dur_beats, partial, vowels, glide_from, velocity in notes:
        _sing(
            score,
            start=bar(start_bar) + beat_offset * stretch * BEAT,
            duration=max(0.4, dur_beats * stretch * BEAT * 0.96),
            partial=partial,
            vowels=vowels,
            glide_from=glide_from,
            amp_db=amp_db,
            velocity=velocity,
            attack=0.25 * stretch,
            release=release,
            breath=breath,
            label="vocal:hymn",
        )


def _hymn_fragment(
    score: Score,
    start_bar: float,
    tones: tuple[float, float, float],
    *,
    amp_db: float = -14.0,
    echo_bells: bool = True,
) -> None:
    """The sigh gesture (hi -> passing -> resolve) on station tones,
    optionally echoed by the bells two bars later, on the grid."""
    hi, passing, low = tones
    _sing(
        score,
        start=bar(start_bar),
        duration=1.4 * BEAT,
        partial=hi,
        vowels=["u", "o"],
        amp_db=amp_db,
        velocity=0.85,
        label="vocal:frag",
    )
    _sing(
        score,
        start=bar(start_bar) + 1.5 * BEAT,
        duration=0.45 * BEAT,
        partial=passing,
        vowels=["o"],
        glide_from=hi,
        amp_db=amp_db - 1.0,
        velocity=0.7,
        label="vocal:frag",
    )
    _sing(
        score,
        start=bar(start_bar) + 2.0 * BEAT,
        duration=2.5 * BEAT,
        partial=low,
        vowels=["o", "a"],
        amp_db=amp_db,
        velocity=0.9,
        release=2.2,
        label="vocal:frag",
    )
    if echo_bells:
        for i, tone in enumerate((hi, low)):
            score.add_note(
                "bells",
                start=sw(start_bar + 2.0, 0 if i == 0 else 4),
                duration=2.5,
                partial=tone,
                amp_db=-15.0,
                velocity=0.75 - 0.1 * i,
                label="bell:echo",
            )


# ---------------------------------------------------------------------------
# Groove spine: arp, bass, drone, kit
# ---------------------------------------------------------------------------

# Per-bar arp pattern: (sixteenth, tone index, velocity weight).  The
# figure rises through the voicing and falls back — locked to the swing.
_ARP_BAR_A = [(0, 0, 0.55), (3, 1, 0.7), (6, 2, 0.85), (10, 3, 0.75), (14, 1, 0.6)]
_ARP_BAR_B = [(2, 1, 0.6), (6, 3, 0.8), (8, 2, 0.7), (11, 0, 0.6), (14, 2, 0.7)]


def _arp_run(
    score: Score,
    start_bar: int,
    end_bar: int,
    *,
    base_vel: float = 1.0,
    amp_db: float = -10.0,
    rest_bars: set[int] | None = None,
) -> None:
    for b in range(start_bar, end_bar):
        if rest_bars and b in rest_bars:
            continue
        tones = _voicing_at(float(b))[2]
        if not tones:
            continue
        pattern = _ARP_BAR_A if b % 2 == 0 else _ARP_BAR_B
        for s, idx, weight in pattern:
            tone = tones[idx % len(tones)]
            wiggle = ((b * 7 + s * 3) % 5) * 0.015
            score.add_note(
                "arp",
                start=sw(b, s),
                duration=0.45 * BEAT,
                partial=tone,
                amp_db=amp_db,
                velocity=min(1.0, base_vel * (weight + wiggle)),
                label="arp",
            )


def _bass_run(
    score: Score,
    start_bar: int,
    end_bar: int,
    *,
    rest_bars: set[int] | None = None,
) -> None:
    """Sub-register bass: feel first, rumble, with some notes.

    Even bars hold a long sub root (40-65 Hz territory); odd bars answer
    with a short root plus an octave-up ghost note — the "some notes".
    Station changes get a slid approach tone.
    """
    change_bars = {int(b) for b, _ in ROOTS}
    for b in range(start_bar, end_bar):
        if rest_bars and b in rest_bars:
            continue
        root = _voicing_at(float(b))[0]
        if b % 2 == 0:
            score.add_note(
                "bass",
                start=sw(b, 2),
                duration=2.2 * BEAT,
                partial=root,
                amp_db=-6.0,
                velocity=0.85,
                label="bass:sub",
            )
        else:
            score.add_note(
                "bass",
                start=sw(b, 2),
                duration=0.7 * BEAT,
                partial=root,
                amp_db=-6.5,
                velocity=0.8,
                label="bass:root",
            )
            score.add_note(
                "bass",
                start=sw(b, 10),
                duration=0.5 * BEAT,
                partial=root * 2.0,
                amp_db=-11.0,
                velocity=0.6,
                label="bass:octave",
            )
        if (b + 1) in change_bars:
            next_root = _voicing_at(float(b + 1))[0]
            score.add_note(
                "bass",
                start=sw(b, 14),
                duration=0.45 * BEAT,
                partial=next_root,
                amp_db=-8.0,
                velocity=0.65,
                pitch_motion=PitchMotionSpec.ratio_glide(
                    start_ratio=root / next_root, end_ratio=1.0
                ),
                label="bass:approach",
            )


def _drone_thread(score: Score) -> None:
    """The ground: station roots as long crossfading tones, always there
    (except the blackness).  Every chord is heard against this."""
    spans: list[tuple[float, float, float]] = []
    for i, (start_b, key) in enumerate(ROOTS):
        end_b = ROOTS[i + 1][0] if i + 1 < len(ROOTS) else 215.0
        root = VOICINGS[key][0]
        spans.append((start_b, end_b, root))
    for start_b, end_b, root in spans:
        seg_start = max(bar(start_b), bar(5.0))  # the drone wakes at bar 5
        seg_end = min(bar(end_b) + 1.0, TOTAL_DUR - 2.0)
        if seg_end <= seg_start:
            continue
        # Silence through the blackness.
        if seg_start < S5_END < seg_end:
            seg_end = S5_END
        if S5_END <= seg_start < S6_END:
            continue
        score.add_note(
            "drone",
            start=seg_start,
            duration=seg_end - seg_start,
            partial=root * 2.0,
            amp_db=-19.0,
            label="drone:root",
        )


def _drum_hit(
    score: Score,
    voice: str,
    bar_num: float,
    sixteenth: float,
    velocity: float,
    *,
    duration: float = 0.25,
    partial: float = 1.0,
    label: str | None = None,
) -> None:
    score.add_note(
        voice,
        start=sw(bar_num, sixteenth),
        duration=duration,
        partial=partial,
        amp_db=0.0,
        velocity=velocity,
        label=label,
    )


def _beat(
    score: Score,
    start_bar: int,
    end_bar: int,
    *,
    intro: bool = False,
    evolved: bool = False,
    kick_out: set[int] | None = None,
    hats_only: set[int] | None = None,
    breath_bars: set[int] | None = None,
    all_out: set[int] | None = None,
) -> None:
    """The swung 2-step kit.  Fills land at 8-bar phrase ends; the rim is
    tuned to the station root so syncopation stays inside the harmony."""
    kick_out = kick_out or set()
    hats_only = hats_only or set()
    breath_bars = breath_bars or set()
    all_out = all_out or set()
    for b in range(start_bar, end_bar):
        if b in all_out:
            continue
        phase = b - start_bar
        breath = b in breath_bars
        only_hats = b in hats_only
        root = _voicing_at(float(b))[0]

        kick_on = (not intro or phase >= 2) and b not in kick_out and not only_hats
        snare_on = (not intro or phase >= 4) and not only_hats and not breath

        if kick_on:
            hits = [(0, 1.0), (10, 0.88)]
            if b % 4 == 3:
                hits = [(0, 1.0), (7, 0.7), (10, 0.88)]
            elif b % 8 == 6:
                hits = [(0, 1.0), (10, 0.86), (13, 0.62)]
            if evolved and b % 16 == 12:
                hits.append((15, 0.5))
            if breath:
                hits = hits[:1]
            for s, vel in hits:
                _drum_hit(score, "kick", b, s, vel, duration=0.35, partial=0.5)

        if snare_on:
            # The backbeat comes in waves, not as a constant 2+4 wall:
            # full for the phrase head, thinning to 4-only mid-phrase,
            # displaced at the sixth bar, ruff into every other phrase.
            p = b % 8
            if evolved:
                backbeat = [(4, 0.92), (12, 0.85)]
                if p == 3:
                    backbeat = [(12, 0.88)]
                elif p == 5:
                    backbeat = [(4, 0.9), (13, 0.55)]
            else:
                backbeat = [(4, 0.88), (12, 0.8)]
                if p in (2, 3):
                    backbeat = [(12, 0.85)]
                elif p == 5:
                    backbeat = [(12, 0.85), (13, 0.4)]
            for s, vel in backbeat:
                _drum_hit(score, "snare", b, s, vel, partial=2.0)
            if p == 1:
                _drum_hit(score, "snare", b, 6, 0.28, partial=2.0)
            if p == 7:
                if b % 16 == 15:
                    for s, vel in [(13, 0.35), (14, 0.45), (15, 0.6)]:
                        _drum_hit(score, "snare", b, s, vel, partial=2.0)
                else:
                    _drum_hit(score, "snare", b, 14, 0.5, partial=2.0)

        # Rim: tuned to the station root — syncopated color, not clash.
        if not breath and not only_hats:
            if evolved and 171 <= b < 179:
                # Climax rotor: every third sixteenth, drifting phase.
                for s in range((b * 16) % 3, 16, 3):
                    if s not in (0, 4, 12):
                        _drum_hit(score, "rim", b, s, 0.24, partial=root * 4.0)
            elif b % 8 in (2, 5) and (not intro or phase >= 6):
                _drum_hit(
                    score,
                    "rim",
                    b,
                    3 if b % 16 < 8 else 11,
                    0.42,
                    partial=root * 4.0,
                )

        hat_pattern = [(2, 0.5), (6, 0.66), (10, 0.5), (14, 0.7)]
        for s, vel in hat_pattern:
            if breath and s not in (6, 14):
                continue
            _drum_hit(score, "hat", b, s, vel, duration=0.12, partial=8.0)
        if not breath and (b % 2 == 0 or evolved):
            for s in (5, 9, 13):
                if not evolved and s == 9:
                    continue
                _drum_hit(score, "hat", b, s, 0.24, duration=0.1, partial=8.0)
        if b % 8 == 7:
            _drum_hit(score, "openhat", b, 14, 0.55, duration=0.5, partial=8.0)


def _pad_station(
    score: Score,
    start_b: float,
    end_b: float,
    key: str,
    *,
    amp_db: float = -16.0,
) -> None:
    tones = VOICINGS[key][1]
    for i, partial in enumerate(tones):
        score.add_note(
            "pad",
            start=bar(start_b),
            duration=bar(end_b) - bar(start_b) + 1.5,
            partial=partial,
            amp_db=amp_db - 0.8 * i,  # gentle tilt: lows carry, highs shimmer
            label=f"chord:{key}",
        )


# ---------------------------------------------------------------------------
# Score setup
# ---------------------------------------------------------------------------


def _setup(score: Score) -> None:
    score.add_send_bus(
        "hall",
        effects=[
            bricasti_or_reverb(
                "1 Halls 07 Large & Dark",
                1.0,
                room_size=0.88,
                damping=0.5,
                lowpass_hz=6800.0,
                tilt_db=-1.5,
            )
        ],
    )
    score.add_drift_bus("night", rate_hz=0.13, depth_cents=5.0, seed=1113)

    # Weather bed: rain wash + vinyl crackle, both sunk into the hall so
    # they read as a place, not as added noise.
    # Real rain: Poisson droplets over a breathing wash (rain_exciter).
    # Sections override density/wash per note — close in the intro,
    # washed-out under the beat, thinning to drizzle at the end.
    score.add_voice(
        "rain",
        synth_defaults={
            "engine": "synth_voice",
            "osc_type": None,
            "noise_type": "rain",
            # Dense, fine, dark: reads as weather, not as pitched squelch.
            "noise_rain_density": 0.7,
            "noise_rain_brightness": 0.3,
            "noise_rain_drop_size": 0.22,
            "noise_rain_wash": 0.65,
            "noise_level": 1.0,
            "cutoff_hz": 4200.0,
            "hpf_cutoff_hz": 220.0,
            "attack": 4.0,
            "release": 6.0,
        },
        # Light hall send only: the wash is already diffuse, and heavy
        # reverb on broadband noise reads as mud (and confuses the IMD
        # artifact probe, which counts diffuse energy as distortion).
        sends=[VoiceSend(target="hall", send_db=-14.0)],
        mix_db=-19.0,
        velocity_humanize=None,
        pan=0.0,
    )
    score.add_voice(
        "crackle",
        synth_defaults={
            "engine": "synth_voice",
            "osc_type": None,
            "noise_type": "flow",
            "noise_flow_density": 0.1,
            "noise_level": 1.0,
            "hpf_cutoff_hz": 1800.0,
            "cutoff_hz": 8200.0,
            "attack": 2.0,
            "release": 3.0,
        },
        sends=[VoiceSend(target="hall", send_db=-8.0)],
        mix_db=-13.0,
        velocity_humanize=None,
        pan=0.06,
    )

    # Ghost vocal — the centerpiece.
    score.add_voice(
        "ghost",
        synth_defaults={"engine": "additive"},
        sends=[VoiceSend(target="hall", send_db=-3.5, automation=[_ghost_hall_ride()])],
        mix_db=-1.5,
        drift_bus="night",
        # High correlation with the arp/pad drift: the ensemble breathes
        # together instead of beating against itself.
        drift_bus_correlation=0.7,
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        pan=-0.04,
    )

    # Pad: open-voiced station chords, dispersed, breathing.
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "additive",
            "n_harmonics": 8,
            "harmonic_rolloff": 0.45,
            "phase_disperse": 0.55,
            "spectral_flicker": 0.12,
            "flicker_rate_hz": 1.1,
            "flicker_correlation": 0.5,
            "attack": 2.8,
            "release": 5.0,
        },
        sends=[VoiceSend(target="hall", send_db=-8.0, automation=[_pad_hall_ride()])],
        mix_db=-7.5,
        drift_bus="night",
        drift_bus_correlation=0.6,
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        pan=0.05,
    )

    # Drone: the ground thread — almost-sine, felt more than heard.
    score.add_voice(
        "drone",
        synth_defaults={
            "engine": "additive",
            "n_harmonics": 3,
            "harmonic_rolloff": 0.4,
            "attack": 3.5,
            "release": 5.0,
        },
        sends=[VoiceSend(target="hall", send_db=-12.0)],
        mix_db=-9.0,
        drift_bus="night",
        drift_bus_correlation=0.8,
        pan=0.0,
    )

    # Arp: the coherence spine — soft pluck on the swing grid.
    score.add_voice(
        "arp",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "soft_pluck",
            "release": 0.5,
        },
        effects=[
            EffectSpec(
                "delay",
                {"delay_seconds": 0.75 * BEAT, "feedback": 0.3, "mix": 0.22},
            ),
        ],
        sends=[VoiceSend(target="hall", send_db=-10.0)],
        mix_db=-6.5,
        envelope_humanize=EnvelopeHumanizeSpec(preset="loose_pluck"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        drift_bus="night",
        drift_bus_correlation=0.5,
        pan=-0.14,
    )

    # Bells: distant FM, drowned in the hall, always on the grid.
    score.add_voice(
        "bells",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "two_op_bell",
            "release": 3.2,
        },
        effects=[
            EffectSpec(
                "delay",
                {"delay_seconds": 0.75 * BEAT, "feedback": 0.42, "mix": 0.3},
            ),
        ],
        sends=[VoiceSend(target="hall", send_db=-2.0)],
        mix_db=-9.5,
        envelope_humanize=EnvelopeHumanizeSpec(preset="loose_pluck"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        drift_bus="night",
        drift_bus_correlation=0.4,
        pan=0.18,
    )

    # Bass: mid-harmonic warmth, ducked under the kick.
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "soft_bass",
            "release": 0.4,
            # Brightness + dirt make the bass read through mids, not level.
            "brightness": 0.55,
            "dirt": 0.35,
        },
        effects=[
            EffectSpec(
                "compressor",
                {
                    "threshold_db": -21.0,
                    "ratio": 4.0,
                    "attack_ms": 2.0,
                    "release_ms": 140.0,
                    "lookahead_ms": 5.0,
                    "sidechain_source": "kick",
                    "detector_mode": "peak",
                },
            ),
        ],
        pan=0.0,
        mix_db=-7.5,
        max_polyphony=1,
        legato=True,
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        drift_bus="night",
        drift_bus_correlation=0.7,
        automation=[_bass_cutoff_arc()],
    )

    # 2-step kit: soft tape kick, brushed snare, tuned rim, swung hats.
    drum_bus = setup_drum_bus(score, style="light")
    add_drum_voice(
        score,
        "kick",
        engine="drum_voice",
        preset="808_tape",
        drum_bus=drum_bus,
        mix_db=-3.5,
    )
    add_drum_voice(
        score,
        "snare",
        engine="drum_voice",
        preset="brush",
        drum_bus=drum_bus,
        mix_db=-6.0,
        pan=0.05,
        # Crisper top than stock brush — reads through the rain haze.
        synth_overrides={"character": 0.6},
        # Soft 2-step hits barely tickle the preset compressors; the
        # drum-bus glue does the real work, so keep these voices bare.
        effects=[],
    )
    add_drum_voice(
        score,
        "rim",
        engine="drum_voice",
        preset="rim_shot",
        drum_bus=drum_bus,
        mix_db=-13.0,
        pan=-0.15,
        effects=[],
    )
    add_drum_voice(
        score,
        "hat",
        engine="drum_voice",
        preset="closed_hat",
        drum_bus=drum_bus,
        mix_db=-11.0,
        choke_group="hats",
        pan=0.12,
        effects=[],
    )
    add_drum_voice(
        score,
        "openhat",
        engine="drum_voice",
        preset="open_hat",
        drum_bus=drum_bus,
        mix_db=-13.0,
        choke_group="hats",
        pan=0.12,
        effects=[],
    )


# ---------------------------------------------------------------------------
# S1 Rain (bars 1-16): weather, ground, and the sigh foreshadowed
# ---------------------------------------------------------------------------


def _s1_rain(score: Score) -> None:
    # Close, detailed rain for the opening — you can hear the drops.
    score.add_note(
        "rain",
        start=0.0,
        duration=S2_END + 4.0,
        partial=1.0,
        amp_db=-12.0,
        synth={"noise_rain_density": 0.8, "noise_rain_wash": 0.55},
    )
    score.add_note(
        "crackle",
        start=bar(2),
        duration=S2_END - bar(2) + 3.0,
        partial=1.0,
        amp_db=-12.0,
    )

    # Bells on the grid, hinting the hymn: the sigh's three tones, slow.
    for start_b, s, partial, vel in [
        (5.0, 0, 3.0, 0.7),
        (7.0, 0, 3.6, 0.6),
        (9.0, 4, 3.0, 0.65),
        (11.0, 0, 2.25, 0.55),
        (13.0, 0, 3.6, 0.7),  # the sigh begins...
        (13.0, 6, 3.15, 0.5),
        (14.0, 0, 3.0, 0.65),  # ...and resolves
        (15.5, 0, 2.0, 0.5),
    ]:
        score.add_note(
            "bells",
            start=sw(start_b, s),
            duration=3.0,
            partial=partial,
            amp_db=-15.0,
            velocity=vel,
            label="bell:foreshadow",
        )


# ---------------------------------------------------------------------------
# S2 First voice (bars 17-40): the hymn, twice, then the comma sigh
# ---------------------------------------------------------------------------


def _s2_first_voice(score: Score) -> None:
    _pad_station(score, 17.0, 33.0, "HOME", amp_db=-15.0)
    _pad_station(score, 33.0, 41.0, "TINT", amp_db=-15.5)

    # First hymn statement — the tune this piece is about.
    _hymn(score, 18.0, HYMN_HOME, amp_db=-13.5)
    # Second statement, bells doubling the reach and the settle.
    _hymn(score, 25.0, HYMN_HOME, amp_db=-13.0)
    score.add_note(
        "bells",
        start=sw(27.0, 6),
        duration=3.0,
        partial=4.0,
        amp_db=-14.0,
        velocity=0.8,
        label="bell:reach",
    )
    score.add_note(
        "bells",
        start=sw(28.5, 0),
        duration=3.5,
        partial=3.0,
        amp_db=-15.0,
        velocity=0.7,
        label="bell:settle",
    )

    # The comma sigh (bars 32-40): the voice holds 9/8 as the chord
    # tints undecimal beneath it, then slides down the 40-cent comma to
    # 11/10 — the first 11 of the piece arrives *in the melody*.
    _sing(
        score,
        start=bar(32),
        duration=2.0 * BAR,
        partial=2.25,
        vowels=["a", "o"],
        amp_db=-13.5,
        velocity=0.85,
        release=1.5,
        label="vocal:comma-hold",
    )
    _sing(
        score,
        start=bar(34),
        duration=2.5 * BAR,
        partial=2.2,
        vowels=["o", "u"],
        glide_from=2.25,
        amp_db=-13.0,
        velocity=0.95,
        release=2.5,
        label="vocal:comma-sigh",
    )
    _sing(
        score,
        start=bar(37.5),
        duration=1.5 * BAR,
        partial=2.475,
        vowels=["u", "o"],
        amp_db=-14.0,
        velocity=0.8,
        label="vocal:C-lift",
    )
    _sing(
        score,
        start=bar(39),
        duration=2.0 * BAR,
        partial=3.3,
        vowels=["o", "a"],
        glide_from=2.475,
        amp_db=-13.5,
        velocity=0.9,
        release=2.5,
        label="vocal:C-crest",
    )

    # The arp ghosts in under the tint, preparing the beat's grid.
    _arp_run(score, 33, 41, base_vel=0.5, amp_db=-13.5)


# ---------------------------------------------------------------------------
# S3 Two-step (bars 41-80): the beat materialises; utonal walk
# ---------------------------------------------------------------------------


def _s3_two_step(score: Score) -> None:
    score.add_note(
        "rain",
        start=S2_END,
        duration=S3_END - S2_END + 3.0,
        partial=1.0,
        amp_db=-15.0,
    )
    score.add_note(
        "crackle",
        start=S2_END,
        duration=S3_END - S2_END + 3.0,
        partial=1.0,
        amp_db=-13.0,
    )

    for start_b, end_b, key in [
        (41.0, 49.0, "HOME"),
        (49.0, 57.0, "TINT"),
        (57.0, 65.0, "U(1,3,7,11)"),
        (65.0, 73.0, "U(1,3,7,9)"),
        (73.0, 81.0, "TINT"),
    ]:
        _pad_station(score, start_b, end_b, key, amp_db=-18.0)

    _beat(
        score,
        41,
        81,
        intro=True,
        kick_out={56, 80},
        hats_only={64},
        breath_bars={71, 72},
        all_out={80},
    )
    _arp_run(score, 41, 81, rest_bars={64, 71, 72, 80})
    _bass_run(score, 45, 80, rest_bars={56, 64, 71, 72})

    # Pivot bells: half a bar before each station change, a quiet bell
    # sounds the common tone the two chords share — the hinge made
    # audible, so no change arrives unearned.
    for b, pivot in [(48.5, 3.6), (56.5, 2.2), (64.5, 2.8), (72.5, 3.6)]:
        score.add_note(
            "bells",
            start=sw(b, 4),
            duration=2.5,
            partial=pivot,
            amp_db=-17.0,
            velocity=0.55,
            label="bell:pivot",
        )

    # Hymn fragments on the walk stations — the tune keeps surfacing.
    _hymn_fragment(score, 50.0, (3.3, 2.475, 2.2))  # TINT sigh
    _hymn_fragment(score, 60.0, (2.566667, 2.2, 1.925))  # septimal-undecimal
    _hymn_fragment(score, 66.0, (3.15, 2.8, 2.1))  # septimal shadow
    # The turn: the sigh lands on 33/20 — a tone the light already owns.
    _sing(
        score,
        start=bar(77),
        duration=1.0 * BAR,
        partial=3.6,
        vowels=["a"],
        amp_db=-13.5,
        velocity=0.9,
        label="vocal:turn",
    )
    _sing(
        score,
        start=bar(78),
        duration=1.0 * BAR,
        partial=3.5,
        vowels=["a", "o"],
        glide_from=3.6,
        amp_db=-13.5,
        velocity=0.85,
        label="vocal:turn",
    )
    _sing(
        score,
        start=bar(79),
        duration=2.0 * BAR,
        partial=3.3,
        vowels=["o", "u"],
        morph_times=[0.0, 0.75],
        glide_from=3.45,
        amp_db=-12.5,
        velocity=1.0,
        release=2.5,
        label="vocal:turn-pivot",
    )


# ---------------------------------------------------------------------------
# S4 Light (bars 81-104): the series chord — hexany garden through glass
# ---------------------------------------------------------------------------


def _s4_light(score: Score) -> None:
    score.add_note(
        "rain",
        start=S3_END,
        duration=bar(105) - S3_END + 3.0,
        partial=1.0,
        amp_db=-17.0,
    )
    score.add_note(
        "crackle",
        start=S3_END,
        duration=bar(105) - S3_END + 3.0,
        partial=1.0,
        amp_db=-14.0,
    )

    _pad_station(score, 81.0, 89.0, "HEX_LIGHT", amp_db=-15.5)
    _pad_station(score, 89.0, 97.0, "O(5,11)", amp_db=-16.0)
    _pad_station(score, 97.0, 105.0, "DEEP", amp_db=-15.5)

    _beat(score, 81, 101, breath_bars={96}, kick_out={100})
    _arp_run(score, 81, 101, base_vel=1.05, amp_db=-9.5)
    _bass_run(score, 81, 99)

    # Structural F1 touch on the downbeat of the light.
    score.add_note(
        "bass",
        start=bar(81),
        duration=1.5 * BEAT,
        partial=_R / 2.0,
        amp_db=-4.0,
        velocity=1.0,
        label="bass:F1-light",
    )

    # The hymn on the light degrees — same tune, a comma elsewhere.
    _hymn(score, 83.0, HYMN_LIGHT, amp_db=-13.0)

    # Bells: series arpeggios on the grid — the garden through glass.
    # Multiples of the light fundamental 0.825, so every tone is real.
    for b, s, mult, vel in [
        (81.0, 6, 2.0, 0.85),
        (82.0, 2, 3.0, 0.7),
        (82.0, 10, 5.0, 0.75),
        (83.0, 6, 7.0, 0.7),
        (85.0, 2, 5.0, 0.65),
        (87.0, 6, 3.0, 0.7),
        (89.0, 0, 2.0, 0.8),
        (91.0, 6, 4.0, 0.65),
        (93.0, 2, 6.0, 0.7),
        (95.0, 6, 3.0, 0.6),
    ]:
        score.add_note(
            "bells",
            start=sw(b, s),
            duration=2.8,
            partial=0.825 * mult,
            amp_db=-13.5,
            velocity=vel,
            label="bell:series",
        )

    # As the shadow slides in at 97 the voice descends into it.
    _sing(
        score,
        start=bar(98),
        duration=1.5 * BAR,
        partial=3.2 * _R,
        vowels=["o"],
        amp_db=-14.0,
        velocity=0.85,
        label="vocal:darken",
    )
    _sing(
        score,
        start=bar(99.5),
        duration=1.5 * BAR,
        partial=2.8875,
        vowels=["o", "u"],
        glide_from=3.2 * _R,
        amp_db=-14.0,
        velocity=0.85,
        label="vocal:darken",
    )
    _sing(
        score,
        start=bar(101),
        duration=3.0 * BAR,
        partial=2.625,
        vowels=["u", "o", "u"],
        morph_times=[0.0, 0.5, 1.0],
        glide_from=2.8875,
        amp_db=-13.5,
        velocity=0.9,
        release=3.5,
        label="vocal:into-shadow",
    )


# ---------------------------------------------------------------------------
# S5 Cathedral (bars 105-136) and S6 Blackness (137-138)
# ---------------------------------------------------------------------------


def _s5_cathedral(score: Score) -> None:
    score.add_note(
        "rain",
        start=bar(105),
        duration=S5_END - bar(105) + 2.0,
        partial=1.0,
        amp_db=-14.0,
    )
    score.add_note(
        "crackle",
        start=bar(105),
        duration=S5_END - bar(105),
        partial=1.0,
        amp_db=-16.0,
    )

    _pad_station(score, 105.0, 117.0, "DEEP", amp_db=-14.5)
    _pad_station(score, 117.0, 125.0, "U(3,5,9,11)", amp_db=-15.0)
    _pad_station(score, 125.0, 137.0, "DARK_HOME", amp_db=-14.5)

    # The hymn, augmented to double length — the piece's slow heart.
    _hymn(score, 106.0, HYMN_DEEP, stretch=2.0, amp_db=-12.5, breath=0.2, release=3.5)
    # Bells shadow the augmented theme a bar behind, an octave above.
    for beat_offset, _dur, partial, _vowels, _glide, vel in HYMN_DEEP[::2]:
        score.add_note(
            "bells",
            start=bar(107.0) + beat_offset * 2.0 * BEAT,
            duration=4.0,
            partial=partial * 2.0,
            amp_db=-16.5,
            velocity=vel * 0.75,
            label="bell:shadow",
        )

    # Over the darkened home: the sigh once more, slow, unaccompanied.
    _sing(
        score,
        start=bar(127),
        duration=2.0 * BAR,
        partial=2.75,
        vowels=["u", "o"],
        amp_db=-13.0,
        velocity=0.85,
        attack=0.6,
        release=3.0,
        breath=0.22,
        label="vocal:dark-sigh",
    )
    _sing(
        score,
        start=bar(129),
        duration=1.5 * BAR,
        partial=2.2,
        vowels=["o"],
        glide_from=2.75,
        amp_db=-13.5,
        velocity=0.8,
        release=3.0,
        breath=0.22,
        label="vocal:dark-sigh",
    )
    _sing(
        score,
        start=bar(131),
        duration=3.5 * BAR,
        partial=1.833333,
        vowels=["o", "u"],
        morph_times=[0.0, 0.7],
        amp_db=-13.0,
        velocity=0.85,
        attack=0.8,
        release=4.0,
        breath=0.24,
        label="vocal:dark-settle",
    )


def _s6_blackness(score: Score) -> None:
    score.add_note(
        "rain",
        start=S5_END,
        duration=S6_END - S5_END,
        partial=1.0,
        amp_db=-26.0,
    )


# ---------------------------------------------------------------------------
# S7 Second wave (bars 139-186): the drowned dekany, call and answer
# ---------------------------------------------------------------------------

# (station bar, otonal call key, utonal answer key)
S7_STATIONS: list[tuple[float, str, str]] = [
    (139.0, "O(1,11)", "TINT"),
    (147.0, "O(3,11)", "U(1,3,7,11)"),
    (155.0, "O(5,11)", "U(3,5,9,11)"),
    (163.0, "O(7,11)", "U(1,7,9,11)"),
    (171.0, "HEX_LIGHT", "DEEP"),
    (179.0, "TINT", "HOME"),
]

# The sigh fragment tones for each station's vocal call.
S7_FRAGMENTS: dict[str, tuple[float, float, float]] = {
    "O(1,11)": (3.3, 2.566667, 2.2),
    "O(3,11)": (2.75, 2.475, 1.925),
    "O(5,11)": (3.208333, 2.75, 1.833333),
    "O(7,11)": (3.208333, 2.8875, 1.925),
}


def _s7_second_wave(score: Score) -> None:
    score.add_note(
        "rain",
        start=S6_END,
        duration=S7_END - S6_END + 3.0,
        partial=1.0,
        amp_db=-16.0,
    )
    score.add_note(
        "crackle",
        start=S6_END,
        duration=S7_END - S6_END + 3.0,
        partial=1.0,
        amp_db=-13.0,
    )

    for station_bar, o_key, u_key in S7_STATIONS:
        _pad_station(score, station_bar, station_bar + 4.0, o_key, amp_db=-17.0)
        _pad_station(score, station_bar + 4.0, station_bar + 8.0, u_key, amp_db=-17.5)

    _beat(
        score,
        139,
        187,
        evolved=True,
        breath_bars={150, 166},
        kick_out={183, 184},
    )
    _arp_run(score, 139, 187, base_vel=1.1, amp_db=-9.0, rest_bars={183, 184})
    _bass_run(score, 140, 183)

    # Slam downbeat: F1 octave with the kick after the blackness.
    score.add_note(
        "bass",
        start=bar(139),
        duration=2.0 * BEAT,
        partial=0.55,
        amp_db=-3.5,
        velocity=1.0,
        label="bass:F1-slam",
    )

    # Call and answer: the voice states the sigh over each otonal call;
    # the bells answer it over the utonal mirror, on the grid.
    for station_bar, o_key, _u_key in S7_STATIONS[:4]:
        frag = S7_FRAGMENTS[o_key]
        _hymn_fragment(score, station_bar + 1.0, frag, echo_bells=True)

    # Climax (171-178): the full hymn over light/deep alternation.
    _hymn(score, 171.0, HYMN_LIGHT, amp_db=-12.5)
    # Wind-down station: the sigh on home tones, last time with drums.
    _hymn_fragment(score, 180.0, (3.6, 3.15, 3.0), echo_bells=True)


# ---------------------------------------------------------------------------
# S8 Dissolution (bars 187-214): the mirror pair, hanging on 5:7:9:11
# ---------------------------------------------------------------------------


def _s8_dissolution(score: Score) -> None:
    # The rain thins to a drizzle as the piece lets go.
    score.add_note(
        "rain",
        start=S7_END,
        duration=TOTAL_DUR - S7_END,
        partial=1.0,
        amp_db=-14.0,
        synth={
            "noise_rain_density": 0.4,
            "noise_rain_drop_size": 0.15,
            "noise_rain_wash": 0.6,
        },
    )
    score.add_note(
        "crackle",
        start=S7_END,
        duration=bar(207) - S7_END,
        partial=1.0,
        amp_db=-15.0,
    )

    # Shadow and light alternate around the pillars 1/1 and 9/5.
    for start_b, key in [
        (187.0, "HOME"),
        (191.0, "FINAL_LIGHT"),
        (195.0, "HOME"),
        (199.0, "FINAL_LIGHT"),
        (203.0, "HOME"),
    ]:
        _pad_station(score, start_b, start_b + 4.0, key, amp_db=-16.0)
    # The final hanging light — longer, quieter, undamped.
    tones = VOICINGS["FINAL_LIGHT"][1]
    for i, partial in enumerate(tones):
        score.add_note(
            "pad",
            start=bar(207),
            duration=bar(214) - bar(207),
            partial=partial,
            amp_db=-15.5 - 0.6 * i,
            label="chord:FINAL_LIGHT-hang",
        )

    # Beat decays: kick thins and is gone by 195; hats ghost to 202.
    for b in range(187, 195):
        if b % 2 == 1:
            _drum_hit(
                score,
                "kick",
                b,
                0,
                0.8 - 0.06 * (b - 187),
                duration=0.35,
                partial=0.5,
            )
        _drum_hit(score, "snare", b, 4, 0.7 - 0.05 * (b - 187), partial=2.0)
        if b < 191:
            _drum_hit(score, "snare", b, 12, 0.6, partial=2.0)
    for b in range(187, 203):
        fade = max(0.15, 0.5 - 0.025 * (b - 187))
        for s in (6, 14):
            _drum_hit(score, "hat", b, s, fade, duration=0.12, partial=8.0)
    # The arp answers only inside the light bars — a phrase, a silence,
    # a phrase — instead of running under everything.
    _arp_run(
        score,
        191,
        203,
        base_vel=0.65,
        amp_db=-12.0,
        rest_bars={195, 196, 197, 198},
    )

    # Bass: one sub pedal on the shared pillar 1/1 (both chords contain
    # it), breathing with the alternation instead of plucking against
    # it.  Gone after 202.
    for start_b in (187.0, 191.0, 195.0, 199.0):
        score.add_note(
            "bass",
            start=bar(start_b),
            duration=4.0 * BAR * 0.96,
            partial=0.5,
            amp_db=-9.0,
            velocity=0.62,
            label="bass:pedal",
        )

    # The last hymn: slow, and changed — the settle lands on 11/10 and
    # the reach has become 4.4.  The tune has learned the eleventh.  It
    # begins ON the first light chord, so the voice and the harmony are
    # having the same conversation.
    _hymn(score, 191.0, HYMN_FINAL, stretch=1.5, amp_db=-13.0, breath=0.2, release=3.5)

    # One last breath up to the eleventh, into the rain.
    _sing(
        score,
        start=bar(205),
        duration=4.0 * BAR,
        partial=2.2,
        vowels=["u", "o"],
        morph_times=[0.0, 0.8],
        amp_db=-14.5,
        velocity=0.75,
        attack=1.0,
        release=5.0,
        breath=0.24,
        label="vocal:last-rise",
    )

    # Final bells: the 5:7:9:11 series walked upward, one tone at a time.
    for b, partial, vel in [
        (207.0, 1.4, 0.7),
        (208.5, 1.8, 0.62),
        (210.0, 2.2, 0.55),
        (211.5, 2.8, 0.5),
        (213.0, 3.6, 0.45),
    ]:
        score.add_note(
            "bells",
            start=bar(b),
            duration=5.0,
            partial=partial,
            amp_db=-15.5,
            velocity=vel,
            label="bell:final-series",
        )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _swell(score: Score, end_bar: float, *, peak_db: float = -15.0) -> None:
    """A rain swell rising into a section turn — reverse-reverb-ish."""
    duration = 3.5 * BAR
    score.add_note(
        "rain",
        start=bar(end_bar) - duration,
        duration=duration,
        partial=1.0,
        amp_db=peak_db,
        synth={
            "attack": duration - 0.4,
            "release": 0.6,
            "noise_rain_density": 0.8,
            "noise_rain_wash": 0.75,
        },
        label="swell",
    )


def build_score() -> Score:
    score = Score(
        f0_hz=F0,
        # Tape haze in front of the default preamp/comp finishing chain.
        master_effects=[
            # Kept gentle: the sustained sub pedals intermodulate badly
            # through heavier tape drive (v3 tripped a severe IMD warning
            # at drive 0.45 / mix 55).
            EffectSpec(
                "chow_tape",
                {"drive": 0.28, "saturation": 0.38, "bias": 0.5, "mix": 32.0},
            ),
            *DEFAULT_MASTER_EFFECTS,
        ],
        timing_humanize=TimingHumanizeSpec(preset="loose_late_night"),
    )
    _setup(score)
    _swell(score, 81.0)
    _swell(score, 171.0, peak_db=-16.0)
    _drone_thread(score)
    _s1_rain(score)
    _s2_first_voice(score)
    _s3_two_step(score)
    _s4_light(score)
    _s5_cathedral(score)
    _s6_blackness(score)
    _s7_second_wave(score)
    _s8_dissolution(score)
    return score


PIECES: dict[str, PieceDefinition] = {
    "sodium_hymn": PieceDefinition(
        name="sodium_hymn",
        output_name="sodium_hymn",
        build_score=build_score,
        sections=(
            PieceSection(label="S1 rain", start_seconds=0.0, end_seconds=S1_END),
            PieceSection(
                label="S2 first voice", start_seconds=S1_END, end_seconds=S2_END
            ),
            PieceSection(label="S3 two-step", start_seconds=S2_END, end_seconds=S3_END),
            PieceSection(label="S4 light", start_seconds=S3_END, end_seconds=S4_END),
            PieceSection(
                label="S5 cathedral", start_seconds=S4_END, end_seconds=S5_END
            ),
            PieceSection(
                label="S6 blackness", start_seconds=S5_END, end_seconds=S6_END
            ),
            PieceSection(
                label="S7 second wave", start_seconds=S6_END, end_seconds=S7_END
            ),
            PieceSection(
                label="S8 dissolution", start_seconds=S7_END, end_seconds=TOTAL_DUR
            ),
        ),
    ),
}
