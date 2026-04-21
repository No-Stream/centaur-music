"""Night Lattice — deep house in F# dorian JI (7-limit).

152 bars at 124 BPM (~4:54).  Shuffled 16ths, 808 kick, pitched cowbell
on the fifth, septimal organ chords, composed bass lines walking through
JI intervals, and a cell-based polyblep arp with dotted-eighth delay.

Tuning: F# dorian JI from F#4 = 370.0 Hz (F0 = 46.25 Hz = F#1).
  1/1   F#   370.0
  9/8   G#   416.25
  6/5   A    444.0
  4/3   B    493.33
  3/2   C#   555.0
  5/3   D#   616.67
  7/4   E    647.5  (septimal 7th)
  2/1   F#'  740.0

Chord voicings (root region F#3 = 185 Hz):
  I   F#m7(sept)  — F#3 + A3 + C#4 + E4   (dark, minor, septimal 7th)
  II  F#sus4/6    — F#3 + B3 + C#4 + D#4  (suspended, bright, dorian)
  III F#m(add9)   — F#3 + G#3 + A3 + C#4  (crunchy cluster, climactic)
  IV  stacked 5ths — F#3 + C#4 + G#4      (open, spacious)

Section map:
      1-  8  intro: pad alone (IV), establishing F#
      9- 16  build_a: IV→I (+ kick bar 5, cowbell bar 5)
     17- 24  build_b: I (+ bass, hats)
     25- 32  groove_a: I↔II (+ arp teaser, clap)
     33- 56  groove_b: I↔II (arp established, bass active)
     57- 64  breakdown_1: IV (pad + arp fragments)
     65- 88  peak_1: III→I→II cycling (everything full)
     89- 96  breakdown_2: IV (bigger drop, just pad + reverb)
     97-112  peak_2: I↔II→III (rebuilt, full energy)
    113-120  settle: I→IV (elements thin)
    121-136  outro_a: IV (arp + pad, elements drop)
    137-152  outro_b: IV (pad fading alone)
"""

from __future__ import annotations

from code_musics.automation import (
    AutomationSegment,
    AutomationSpec,
    AutomationTarget,
)
from code_musics.drum_helpers import add_drum_voice, setup_drum_bus
from code_musics.humanize import TimingHumanizeSpec
from code_musics.meter import Groove
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score, VelocityParamMap, VoiceSend

# ---------------------------------------------------------------------------
# Timing grid
# ---------------------------------------------------------------------------

BPM = 124.0
BEAT = 60.0 / BPM
BAR = 4.0 * BEAT
S16 = BEAT / 4.0
GROOVE = Groove(
    subdivision="sixteenth",
    timing_offsets=(0.0, 0.16),
    velocity_weights=(1.0, 1.0),
)
F0 = 46.25  # F#1
TOTAL_BARS = 152

DOTTED_EIGHTH = 3.0 * S16  # ~0.3629 s at 124 BPM


def _pos(bar: int, beat: int = 1, n16: int = 0) -> float:
    """Bar/beat/16th -> seconds.  Groove-shifted on upbeat 16ths."""
    base = (bar - 1) * BAR + (beat - 1) * BEAT + n16 * S16
    return base + S16 * GROOVE.timing_offset_at(n16)


def _pos_straight(bar: int, beat: int = 1, n16: int = 0) -> float:
    return (bar - 1) * BAR + (beat - 1) * BEAT + n16 * S16


TOTAL_DUR = _pos_straight(TOTAL_BARS + 1)

# ---------------------------------------------------------------------------
# JI scale — F# dorian 7-limit (from F#4 = 370.0 Hz)
# ---------------------------------------------------------------------------

# Scale degrees as absolute freqs (octave 4)
_FS4 = 370.0  # 1/1
_GS4 = 370.0 * 9 / 8  # 416.25
_A4 = 370.0 * 6 / 5  # 444.0
_B4 = 370.0 * 4 / 3  # 493.33...
_CS5 = 370.0 * 3 / 2  # 555.0
_DS5 = 370.0 * 5 / 3  # 616.67
_E5 = 370.0 * 7 / 4  # 647.5 (septimal 7th)
_FS5 = 370.0 * 2  # 740.0

# Octave 3 (for chords and bass)
_FS3 = _FS4 / 2  # 185.0
_GS3 = _GS4 / 2  # 208.125
_A3 = _A4 / 2  # 222.0
_B3 = _B4 / 2  # 246.67
_CS4 = _CS5 / 2  # 277.5
_DS4 = _DS5 / 2  # 308.33
_E4 = _E5 / 2  # 323.75

# Octave 2 (for bass)
_FS2 = _FS3 / 2  # 92.5
_A2 = _A3 / 2  # 111.0
_B2 = _B3 / 2  # 123.33
_CS3 = _CS4 / 2  # 138.75

# ---------------------------------------------------------------------------
# Chord voicings (as lists of (freq, amp_db))
# ---------------------------------------------------------------------------

_CHORD_I: list[tuple[float, float]] = [
    (_FS3, -8.0),
    (_A3, -11.0),
    (_CS4, -12.0),
    (_E4, -14.0),
]
_CHORD_II: list[tuple[float, float]] = [
    (_FS3, -8.0),
    (_B3, -11.0),
    (_CS4, -12.0),
    (_DS4, -13.0),
]
_CHORD_III: list[tuple[float, float]] = [
    (_FS3, -8.0),
    (_GS3, -10.0),
    (_A3, -11.0),
    (_CS4, -12.0),
]
_CHORD_IV: list[tuple[float, float]] = [
    (_FS3, -8.0),
    (_CS4, -11.0),
    (_GS4, -14.0),
]


def _chord_for_bar(bar: int) -> list[tuple[float, float]]:
    """Return the chord voicing for a given bar, following the section map."""
    if bar <= 8:
        return _CHORD_IV  # intro
    if bar <= 16:
        # build_a: IV -> I across 8 bars
        return _CHORD_I if bar > 12 else _CHORD_IV
    if bar <= 24:
        return _CHORD_I  # build_b
    if bar <= 56:
        # groove_a + groove_b: alternate I / II every 4 bars
        return _CHORD_II if ((bar - 25) // 4) % 2 == 1 else _CHORD_I
    if bar <= 64:
        return _CHORD_IV  # breakdown_1
    if bar <= 88:
        # peak_1: III -> I -> II cycling every ~5 bars
        cycle_pos = ((bar - 65) // 5) % 3
        return [_CHORD_III, _CHORD_I, _CHORD_II][cycle_pos]
    if bar <= 96:
        return _CHORD_IV  # breakdown_2
    if bar <= 112:
        # peak_2: I <-> II with III at end
        cycle_pos = ((bar - 97) // 4) % 4
        return [_CHORD_I, _CHORD_II, _CHORD_I, _CHORD_III][cycle_pos]
    if bar <= 120:
        # settle: I -> IV
        return _CHORD_IV if bar > 116 else _CHORD_I
    # outro_a + outro_b: IV
    return _CHORD_IV


# ---------------------------------------------------------------------------
# Kick synth overrides
# ---------------------------------------------------------------------------

_KICK_808: dict = {
    "tone_decay_s": 0.65,
    "tone_envelope": [
        {"time": 0.0, "value": 1.0},
        {"time": 0.50, "value": 0.72, "curve": "linear"},
        {"time": 0.80, "value": 0.12, "curve": "exponential"},
        {"time": 1.0, "value": 0.0, "curve": "linear"},
    ],
    "tone_sweep_ratio": 2.5,
    "tone_sweep_decay_s": 0.038,
    "tone_wave": "sine",
    "tone_second_harmonic": 0.05,
    "tone_punch": 0.14,
    "filter_mode": "lowpass",
    "filter_cutoff_hz": 750.0,
    "filter_q": 0.7,
    "filter_envelope": [
        {"time": 0.0, "value": 2200.0},
        {"time": 0.08, "value": 750.0, "curve": "exponential"},
        {"time": 1.0, "value": 380.0, "curve": "linear"},
    ],
    "exciter_level": 0.025,
    "exciter_decay_s": 0.005,
    "exciter_center_hz": 1800.0,
    "noise_level": 0.006,
    "noise_decay_s": 0.020,
    "noise_center_ratio": 5.95,
}

# ---------------------------------------------------------------------------
# Hat synth — 909 noise-forward character
# ---------------------------------------------------------------------------

_HAT_909: dict = {
    "metallic_type": "partials",
    "metallic_n_partials": 3,
    "metallic_brightness": 0.22,
    "metallic_density": 0.65,
    "metallic_decay_s": 0.008,
    "metallic_level": 1.0,
    "metallic_filter_q": 0.6,
    "noise_level": 0.90,
    "noise_decay_s": 0.008,
    "exciter_level": 0.60,
    "exciter_decay_s": 0.0003,
    "tone_type": None,
}

_OHH_909: dict = {
    **_HAT_909,
    "metallic_decay_s": 0.130,
    "noise_decay_s": 0.130,
    "exciter_level": 0.35,
    "metallic_filter_q": 0.65,
}

# Hat pitch pool — harmonic-series of F0 = 46.25 Hz
HAT_FS = 7400.0  # H160 = 46.25 * 160 — root (F#)
HAT_CS = 8325.0  # H180 = 46.25 * 180 — fifth (C#)
HAT_SEPT = 9250.0  # H200 = 46.25 * 200 — high
HAT_A = 6660.0  # H144 = 46.25 * 144 — minor 3rd area

_HAT_PALETTES: list[tuple[int, tuple[float, float, float]]] = [
    (24, (HAT_FS, HAT_FS, HAT_FS)),  # build: neutral
    (56, (HAT_FS, HAT_CS, HAT_A)),  # groove: warming
    (64, (HAT_FS, HAT_FS, HAT_FS)),  # breakdown_1: withdrawn
    (88, (HAT_CS, HAT_A, HAT_SEPT)),  # peak_1: bright
    (96, (HAT_FS, HAT_FS, HAT_FS)),  # breakdown_2: withdrawn
    (112, (HAT_CS, HAT_A, HAT_SEPT)),  # peak_2: bright
    (120, (HAT_FS, HAT_CS, HAT_FS)),  # settle: narrowing
    (999, (HAT_FS, HAT_FS, HAT_FS)),  # outro: dark
]

_N16_TO_TIER: dict[int, int] = {0: 0, 1: 2, 2: 1, 3: 2}


def _hat_pitch(bar: int, n16: int) -> float:
    palette = (HAT_FS, HAT_FS, HAT_FS)
    for cutoff_bar, pal in _HAT_PALETTES:
        if bar <= cutoff_bar:
            palette = pal
            break
    tier = _N16_TO_TIER.get(n16, 0)
    return palette[tier]


_HAT_FILTERS: list[tuple[int, list[dict]]] = [
    (
        24,
        [
            {"time": 0.0, "value": 5500.0},
            {"time": 0.4, "value": 3800.0, "curve": "exponential"},
            {"time": 1.0, "value": 2800.0, "curve": "linear"},
        ],
    ),
    (
        56,
        [
            {"time": 0.0, "value": 8000.0},
            {"time": 0.35, "value": 5500.0, "curve": "exponential"},
            {"time": 1.0, "value": 4000.0, "curve": "linear"},
        ],
    ),
    (
        88,
        [
            {"time": 0.0, "value": 11000.0},
            {"time": 0.3, "value": 7500.0, "curve": "exponential"},
            {"time": 1.0, "value": 5000.0, "curve": "linear"},
        ],
    ),
    (
        112,
        [
            {"time": 0.0, "value": 11000.0},
            {"time": 0.3, "value": 7500.0, "curve": "exponential"},
            {"time": 1.0, "value": 5000.0, "curve": "linear"},
        ],
    ),
    (
        120,
        [
            {"time": 0.0, "value": 7000.0},
            {"time": 0.4, "value": 5000.0, "curve": "exponential"},
            {"time": 1.0, "value": 3500.0, "curve": "linear"},
        ],
    ),
    (
        999,
        [
            {"time": 0.0, "value": 4500.0},
            {"time": 0.5, "value": 3000.0, "curve": "exponential"},
            {"time": 1.0, "value": 2000.0, "curve": "linear"},
        ],
    ),
]


def _hat_synth(bar: int) -> dict:
    for cutoff_bar, filt in _HAT_FILTERS:
        if bar <= cutoff_bar:
            return {**_HAT_909, "filter_mode": "bandpass", "filter_envelope": filt}
    return _HAT_909


# ---------------------------------------------------------------------------
# Drum pattern helpers
# ---------------------------------------------------------------------------

_PAT_8THS: set[int] = {0, 2}
_PAT_16THS: set[int] = {0, 1, 2, 3}
_PAT_16THS_NO_GHOST: set[int] = {0, 2, 3}


def _chh_subdivs(bar: int) -> set[int]:
    if bar <= 24:
        return _PAT_8THS
    if bar <= 56:
        return _PAT_16THS if (bar - 1) % 4 < 3 else _PAT_16THS_NO_GHOST
    if bar <= 64:
        return set()  # breakdown_1: no hats
    if bar <= 88:
        # peak_1: full 16ths with variation
        phrase = ((bar - 65) // 4) % 3
        return [_PAT_16THS, _PAT_16THS_NO_GHOST, _PAT_16THS][phrase]
    if bar <= 96:
        return set()  # breakdown_2: no hats
    if bar <= 112:
        # peak_2: full 16ths
        phrase = ((bar - 97) // 4) % 3
        return [_PAT_16THS, _PAT_16THS_NO_GHOST, _PAT_16THS][phrase]
    if bar <= 120:
        return _PAT_8THS  # settle
    if bar <= 136:
        return _PAT_8THS if bar <= 128 else set()  # outro_a fade
    return set()  # outro_b: no hats


_ACCENTS_A: dict[int, float] = {0: -10.0, 1: -17.0, 2: -12.0, 3: -18.0}
_ACCENTS_B: dict[int, float] = {0: -11.0, 1: -16.0, 2: -10.0, 3: -17.0}


def _hat_accents(bar: int) -> dict[int, float]:
    return _ACCENTS_B if ((bar - 1) // 4) % 2 == 1 else _ACCENTS_A


def _has_ohh(bar: int) -> bool:
    return 25 <= bar <= 56 or 65 <= bar <= 88 or 97 <= bar <= 112


def _ohh_beats(bar: int) -> list[int]:
    phase = ((bar - 1) // 8) % 2
    return [2, 4] if phase == 0 else [4]


def _kick_pattern(bar: int, beat: int) -> bool:
    if bar <= 4 or bar > 136:
        return False  # no kick in first 4 bars or outro_b
    if 57 <= bar <= 64:
        return beat in {1, 3}  # breakdown_1: half time
    if 89 <= bar <= 96:
        return False  # breakdown_2: no kick
    if 121 <= bar <= 136:
        # outro_a: sparse kick, thinning
        return bar <= 128 and beat in {1, 3}
    # Occasional beat-3 drop every 8 bars for breathing
    return not (bar % 8 == 0 and 25 <= bar <= 88 and beat == 3)


def _has_clap(bar: int) -> bool:
    return 25 <= bar <= 56 or 65 <= bar <= 88 or 97 <= bar <= 112


def _clap_beats(bar: int) -> list[int]:
    return [2, 4]


# ---------------------------------------------------------------------------
# Cowbell pattern — pitched on C#5 (the fifth), distinctive element
# ---------------------------------------------------------------------------


def _has_cowbell(bar: int) -> bool:
    if bar <= 4:
        return False  # first 4 bars: pad alone
    if 57 <= bar <= 64:
        return False  # breakdown_1
    if 89 <= bar <= 96:
        return False  # breakdown_2
    return bar <= 136  # outro_b (137+): pad alone


def _cowbell_hits(bar: int) -> list[tuple[int, int, float]]:
    """Return (beat, n16, amp_db) for cowbell hits in this bar."""
    hits: list[tuple[int, int, float]] = []
    if bar % 2 == 1:
        hits.append((1, 0, -8.0))  # beat 1 every other bar
    if bar % 4 == 0:
        hits.append((3, 0, -12.0))  # beat 3 every 4th bar
    in_peak = 65 <= bar <= 88 or 97 <= bar <= 112
    if in_peak and bar % 2 == 0:
        hits.append((3, 2, -14.0))  # extra presence during peaks
    return hits


# ---------------------------------------------------------------------------
# Bass — composed per section, using absolute freq
# ---------------------------------------------------------------------------


def _place_bass_bar(score: Score, bar: int) -> None:
    if bar <= 24:
        # Simple: root on beat 1
        score.add_note(
            "bass",
            start=_pos_straight(bar, 1),
            duration=BEAT * 3.0,
            freq=_FS2,
            amp_db=-6.0,
        )
        if bar % 4 == 0 and bar > 20:
            # Pickup fifth
            score.add_note(
                "bass",
                start=_pos_straight(bar, 4, 2),
                duration=BEAT * 0.5,
                freq=_CS3,
                amp_db=-10.0,
            )

    elif bar <= 56:
        # Groove: root + fifth movement
        phase = ((bar - 25) // 4) % 3
        if phase == 0:
            score.add_note(
                "bass",
                start=_pos_straight(bar, 1),
                duration=BEAT * 2.5,
                freq=_FS2,
                amp_db=-6.0,
            )
            score.add_note(
                "bass",
                start=_pos_straight(bar, 3, 2),
                duration=BEAT * 0.8,
                freq=_CS3,
                amp_db=-9.0,
            )
        elif phase == 1:
            score.add_note(
                "bass",
                start=_pos_straight(bar, 1),
                duration=BEAT * 1.5,
                freq=_FS2,
                amp_db=-6.0,
            )
            score.add_note(
                "bass",
                start=_pos_straight(bar, 3),
                duration=BEAT * 1.0,
                freq=_CS3,
                amp_db=-9.0,
            )
            score.add_note(
                "bass",
                start=_pos_straight(bar, 4, 1),
                duration=BEAT * 0.5,
                freq=_A2,
                amp_db=-10.0,
            )
        else:
            score.add_note(
                "bass",
                start=_pos_straight(bar, 1),
                duration=BEAT * 3.5,
                freq=_FS2,
                amp_db=-6.0,
            )

    elif 57 <= bar <= 64:
        # Breakdown_1: sustained root, quiet
        score.add_note(
            "bass",
            start=_pos_straight(bar, 1),
            duration=BAR * 0.9,
            freq=_FS2,
            amp_db=-10.0,
        )

    elif bar <= 88:
        # Peak_1: most active — walking through F#, A, B, C#
        _place_bass_peak(score, bar, offset=65)

    elif 89 <= bar <= 96:
        # Breakdown_2: no bass
        pass

    elif bar <= 112:
        # Peak_2: same walking patterns, offset from bar 97
        _place_bass_peak(score, bar, offset=97)

    elif bar <= 120:
        # Settle: simpler
        score.add_note(
            "bass",
            start=_pos_straight(bar, 1),
            duration=BEAT * 2.5,
            freq=_FS2,
            amp_db=-7.0,
        )
        if bar % 2 == 0:
            score.add_note(
                "bass",
                start=_pos_straight(bar, 4),
                duration=BEAT * 0.5,
                freq=_CS3,
                amp_db=-11.0,
            )

    elif bar <= 136:
        # Outro_a: simple root, fading
        if bar <= 128:
            score.add_note(
                "bass",
                start=_pos_straight(bar, 1),
                duration=BEAT * 2.0,
                freq=_FS2,
                amp_db=-8.0,
            )


def _place_bass_peak(score: Score, bar: int, *, offset: int) -> None:
    """Shared peak bass pattern for peak_1 and peak_2."""
    phase = ((bar - offset) // 4) % 4
    if phase == 0:
        score.add_note(
            "bass",
            start=_pos_straight(bar, 1),
            duration=BEAT * 2.0,
            freq=_FS2,
            amp_db=-5.0,
        )
        score.add_note(
            "bass",
            start=_pos_straight(bar, 3),
            duration=BEAT * 0.75,
            freq=_A2,
            amp_db=-8.0,
        )
        score.add_note(
            "bass",
            start=_pos_straight(bar, 4),
            duration=BEAT * 0.75,
            freq=_FS2,
            amp_db=-7.0,
        )
    elif phase == 1:
        score.add_note(
            "bass",
            start=_pos_straight(bar, 1),
            duration=BEAT * 1.5,
            freq=_FS2,
            amp_db=-5.0,
        )
        score.add_note(
            "bass",
            start=_pos_straight(bar, 2, 2),
            duration=BEAT * 1.0,
            freq=_B2,
            amp_db=-8.0,
        )
        score.add_note(
            "bass",
            start=_pos_straight(bar, 4, 1),
            duration=BEAT * 0.5,
            freq=_CS3,
            amp_db=-9.0,
        )
    elif phase == 2:
        score.add_note(
            "bass",
            start=_pos_straight(bar, 1),
            duration=BEAT * 3.0,
            freq=_FS2,
            amp_db=-5.0,
        )
        score.add_note(
            "bass",
            start=_pos_straight(bar, 4, 2),
            duration=BEAT * 0.4,
            freq=_A2,
            amp_db=-9.0,
        )
    else:
        # Busiest phrase with septimal push
        score.add_note(
            "bass",
            start=_pos_straight(bar, 1),
            duration=BEAT * 1.5,
            freq=_FS2,
            amp_db=-5.0,
        )
        score.add_note(
            "bass",
            start=_pos_straight(bar, 2, 2),
            duration=BEAT * 0.75,
            freq=_A2,
            amp_db=-8.0,
        )
        score.add_note(
            "bass",
            start=_pos_straight(bar, 3, 2),
            duration=BEAT * 0.75,
            freq=_B2,
            amp_db=-9.0,
        )
        score.add_note(
            "bass",
            start=_pos_straight(bar, 4, 2),
            duration=BEAT * 0.4,
            freq=_FS2,
            amp_db=-7.0,
        )


# ---------------------------------------------------------------------------
# Arp cells — 3 contours through the F# dorian JI scale
#
# Each cell is a bar of 8 x 16th notes.  The cell picks scale tones
# from two pools: "chord I" tones (A-heavy, minor) and "chord II"
# tones (B-heavy, sus4).  For a given bar we choose the pool based
# on the current chord, then select a cell and optional variation.
# ---------------------------------------------------------------------------

# Tone pools per chord (arp octave 4-5)
_POOL_I: list[float] = [_FS4, _A4, _B4, _CS5, _FS5]  # chord I tones
_POOL_II: list[float] = [_FS4, _B4, _CS5, _DS5, _FS5]  # chord II tones

# Cell contours as indices into the 5-tone pool
_CELL_A: list[int] = [0, 2, 1, 3, 2, 3, 1, 0]  # F#->B->A->C#->B->C#->A->F# circular
_CELL_B: list[int] = [3, 1, 2, 0, 1, 4, 2, 1]  # C#->A->B->F#->A->F#'->B->A wider
_CELL_C: list[int] = [2, 3, 1, 2, 0, 4, 3, 2]  # B->C#->A->B->F#->F#'->C#->B ornamental

# Variations: octave drops, dyads, substitutions
_CELL_A_VAR: list[int] = [0, 2, 1, 3, 4, 3, 1, 0]  # octave F#' at pos 4
_CELL_B_VAR: list[int] = [3, 1, 4, 0, 1, 2, 0, 1]  # F#' at pos 2, extra root at 6
_CELL_C_VAR: list[int] = [2, 3, 1, 4, 0, 2, 3, 0]  # stretched, landing at root


# Velocity contours — give the arp life
_VEL_A: list[float] = [0.95, 0.80, 0.70, 0.90, 0.75, 0.85, 0.65, 0.70]
_VEL_B: list[float] = [0.90, 0.75, 0.85, 0.70, 0.80, 0.95, 0.70, 0.75]
_VEL_C: list[float] = [0.85, 0.90, 0.70, 0.80, 0.65, 0.90, 0.85, 0.75]

# Amplitude per-note offsets (relative to base amp_db)
_AMP_OFFSETS: list[float] = [0.0, -2.0, -3.0, -1.0, -2.5, -1.5, -4.0, -3.0]


def _arp_cell_for_bar(bar: int) -> tuple[list[int], list[float]]:
    """Select which cell pattern and velocity contour to use for a bar."""
    phrase_idx = ((bar - 1) // 4) % 6
    if phrase_idx in {0, 3}:
        # Use variation every other 4-bar phrase
        cell = _CELL_A_VAR if bar % 8 > 4 else _CELL_A
        return cell, _VEL_A
    if phrase_idx in {1, 4}:
        cell = _CELL_B_VAR if bar % 8 > 4 else _CELL_B
        return cell, _VEL_B
    cell = _CELL_C_VAR if bar % 8 > 4 else _CELL_C
    return cell, _VEL_C


def _arp_pool_for_bar(bar: int) -> list[float]:
    """Select the tone pool based on the current chord."""
    chord = _chord_for_bar(bar)
    if chord is _CHORD_II or chord is _CHORD_IV:
        return _POOL_II
    return _POOL_I


def _has_arp(bar: int) -> bool:
    """Whether the arp plays in this bar."""
    if 25 <= bar <= 32:
        # Teaser: only every other bar
        return bar % 2 == 1
    if 33 <= bar <= 56:
        return True  # groove_b: established
    if 57 <= bar <= 64:
        # Breakdown_1: sparse fragments (1 bar in 4)
        return (bar - 57) % 4 == 0
    if 65 <= bar <= 88:
        return True  # peak_1: full
    if 89 <= bar <= 96:
        # Breakdown_2: very sparse
        return (bar - 89) % 4 == 0
    if 97 <= bar <= 112:
        return True  # peak_2: full
    if 113 <= bar <= 120:
        # Settle: thinning
        return bar % 2 == 0
    if 121 <= bar <= 136:
        # Outro_a: ghost arp, sparse
        return bar % 4 == 1
    return False


def _arp_amp_for_bar(bar: int) -> float:
    """Base amplitude for the arp in this bar (dB)."""
    if 25 <= bar <= 32:
        return -18.0 + (bar - 25) * 0.75  # teaser: ramp from -18 to ~-12.75
    if 33 <= bar <= 56:
        return -10.0  # groove: present
    if 57 <= bar <= 64:
        return -16.0  # breakdown_1: fragments
    if 65 <= bar <= 88:
        return -8.0  # peak_1: prominent
    if 89 <= bar <= 96:
        return -18.0  # breakdown_2: very quiet fragments
    if 97 <= bar <= 112:
        return -8.0  # peak_2: prominent
    if 113 <= bar <= 120:
        return -12.0  # settle
    if 121 <= bar <= 136:
        return -18.0  # outro_a: ghost
    return -20.0  # outro_b: silence


# ---------------------------------------------------------------------------
# Build score
# ---------------------------------------------------------------------------


def build_score() -> Score:
    score = Score(
        f0_hz=F0,
        sample_rate=44_100,
        timing_humanize=TimingHumanizeSpec(
            preset="tight_ensemble",
            seed=42,
        ),
        master_effects=[
            EffectSpec("preamp", {"drive": 0.30, "mix": 0.65}),
            EffectSpec(
                "compressor",
                {
                    "threshold_db": -16.0,
                    "ratio": 2.0,
                    "attack_ms": 28.0,
                    "release_ms": 280.0,
                    "knee_db": 8.0,
                    "makeup_gain_db": 0.5,
                    "topology": "feedback",
                    "detector_mode": "rms",
                    "detector_bands": [
                        {"kind": "highpass", "cutoff_hz": 80.0, "slope_db_per_oct": 12},
                    ],
                },
            ),
        ],
    )

    # --- Send buses ---

    drum_bus = setup_drum_bus(
        score,
        effects=[
            EffectSpec(
                "compressor",
                {
                    "threshold_db": -18.0,
                    "ratio": 1.5,
                    "attack_ms": 22.0,
                    "release_ms": 180.0,
                    "knee_db": 8.0,
                    "topology": "feedback",
                    "detector_mode": "rms",
                    "detector_bands": [
                        {"kind": "highpass", "cutoff_hz": 60.0, "slope_db_per_oct": 12},
                    ],
                },
            ),
            EffectSpec("saturation", {"mode": "triode", "drive": 1.8, "mix": 0.28}),
        ],
        return_db=0.0,
    )

    score.add_send_bus(
        "hall",
        effects=[
            EffectSpec(
                "reverb",
                {"room_size": 0.88, "damping": 0.50, "wet_level": 1.0},
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
        return_db=-3.0,
    )

    # --- Voices ---

    # Kick
    add_drum_voice(
        score,
        "kick",
        engine="drum_voice",
        preset="808_hiphop",
        drum_bus=drum_bus,
        send_db=-2.0,
        effects=[EffectSpec("compressor", {"preset": "kick_glue"})],
        mix_db=0.0,
    )

    # Closed hat
    add_drum_voice(
        score,
        "closed_hat",
        engine="drum_voice",
        preset="closed_hat",
        drum_bus=drum_bus,
        send_db=-4.0,
        choke_group="hats",
        effects=[EffectSpec("compressor", {"preset": "hat_control"})],
        mix_db=-7.0,
    )
    score.voices["closed_hat"].sends.append(VoiceSend(target="hall", send_db=-16.0))

    # Open hat
    add_drum_voice(
        score,
        "open_hat",
        engine="drum_voice",
        preset="open_hat",
        drum_bus=drum_bus,
        send_db=-4.0,
        choke_group="hats",
        mix_db=-6.0,
    )
    score.voices["open_hat"].sends.append(VoiceSend(target="hall", send_db=-8.0))

    # Clap
    add_drum_voice(
        score,
        "clap",
        engine="drum_voice",
        preset="909_clap",
        drum_bus=drum_bus,
        send_db=-3.0,
        mix_db=-4.0,
    )
    score.voices["clap"].sends.append(VoiceSend(target="hall", send_db=-6.0))

    # Cowbell — pitched on the fifth (C#5 = 555.0 Hz)
    add_drum_voice(
        score,
        "cowbell",
        engine="drum_voice",
        preset="cowbell",
        drum_bus=drum_bus,
        send_db=-5.0,
        mix_db=-5.0,
    )
    score.voices["cowbell"].sends.append(VoiceSend(target="hall", send_db=-10.0))

    # Bass — polyblep sub with second osc for upper harmonics
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "polyblep",
            "preset": "sub_bass",
            "cutoff_hz": 220.0,
            "resonance_q": 1.2,
            "filter_env_amount": 1.5,
            "filter_env_decay": 0.10,
            "osc2_level": 0.25,
            "osc2_waveform": "square",
            "osc2_semitones": 12.0,
            "osc2_detune_cents": 5.0,
        },
        effects=[
            EffectSpec("preamp", {"drive": 0.18, "mix": 0.4}),
            EffectSpec(
                "compressor", {"preset": "kick_duck", "sidechain_source": "kick"}
            ),
        ],
        mix_db=1.0,
        velocity_humanize=None,
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="cutoff_hz"),
                segments=(
                    AutomationSegment(
                        start=_pos_straight(1),
                        end=_pos_straight(33),
                        shape="linear",
                        start_value=200.0,
                        end_value=260.0,
                    ),
                    AutomationSegment(
                        start=_pos_straight(33),
                        end=_pos_straight(57),
                        shape="linear",
                        start_value=260.0,
                        end_value=340.0,
                    ),
                    AutomationSegment(
                        start=_pos_straight(57),
                        end=_pos_straight(65),
                        shape="linear",
                        start_value=340.0,
                        end_value=220.0,
                    ),
                    AutomationSegment(
                        start=_pos_straight(65),
                        end=_pos_straight(89),
                        shape="linear",
                        start_value=220.0,
                        end_value=360.0,
                    ),
                    AutomationSegment(
                        start=_pos_straight(89),
                        end=_pos_straight(97),
                        shape="linear",
                        start_value=360.0,
                        end_value=200.0,
                    ),
                    AutomationSegment(
                        start=_pos_straight(97),
                        end=_pos_straight(113),
                        shape="linear",
                        start_value=200.0,
                        end_value=340.0,
                    ),
                    AutomationSegment(
                        start=_pos_straight(113),
                        end=_pos_straight(153),
                        shape="linear",
                        start_value=340.0,
                        end_value=200.0,
                    ),
                ),
            ),
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="resonance_q"),
                segments=(
                    AutomationSegment(
                        start=_pos_straight(1),
                        end=_pos_straight(65),
                        shape="linear",
                        start_value=1.0,
                        end_value=1.0,
                    ),
                    AutomationSegment(
                        start=_pos_straight(65),
                        end=_pos_straight(89),
                        shape="linear",
                        start_value=1.0,
                        end_value=1.8,
                    ),
                    AutomationSegment(
                        start=_pos_straight(89),
                        end=_pos_straight(97),
                        shape="linear",
                        start_value=1.8,
                        end_value=1.0,
                    ),
                    AutomationSegment(
                        start=_pos_straight(97),
                        end=_pos_straight(113),
                        shape="linear",
                        start_value=1.0,
                        end_value=1.8,
                    ),
                    AutomationSegment(
                        start=_pos_straight(113),
                        end=_pos_straight(153),
                        shape="linear",
                        start_value=1.8,
                        end_value=1.0,
                    ),
                ),
            ),
        ],
    )

    # Organ pad — dark registration, heavy vibrato
    score.add_voice(
        "keys",
        synth_defaults={
            "engine": "organ",
            "drawbars": [0, 8, 6, 4, 0, 0, 0, 0, 0],
            "click": 0.05,
            "click_brightness": 0.25,
            "vibrato_depth": 0.08,
            "vibrato_rate_hz": 5.2,
            "vibrato_chorus": 0.45,
            "drift": 0.18,
            "drift_rate_hz": 0.04,
            "leakage": 0.04,
            "tonewheel_shape": 0.0,
        },
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "lowpass",
                            "cutoff_hz": 2800.0,
                            "slope_db_per_oct": 12,
                        },
                        {
                            "kind": "highpass",
                            "cutoff_hz": 140.0,
                            "slope_db_per_oct": 12,
                        },
                    ]
                },
            ),
            EffectSpec("preamp", {"drive": 0.22, "mix": 0.5}),
            EffectSpec(
                "compressor", {"preset": "kick_duck_hard", "sidechain_source": "kick"}
            ),
        ],
        mix_db=-2.0,
        normalize_lufs=-22.0,
        velocity_humanize=None,
        sends=[VoiceSend(target="hall", send_db=-5.0)],
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="vibrato_depth"),
                segments=(
                    AutomationSegment(
                        start=_pos_straight(1),
                        end=_pos_straight(57),
                        shape="linear",
                        start_value=0.05,
                        end_value=0.12,
                    ),
                    AutomationSegment(
                        start=_pos_straight(57),
                        end=_pos_straight(65),
                        shape="linear",
                        start_value=0.12,
                        end_value=0.06,
                    ),
                    AutomationSegment(
                        start=_pos_straight(65),
                        end=_pos_straight(89),
                        shape="linear",
                        start_value=0.06,
                        end_value=0.15,
                    ),
                    AutomationSegment(
                        start=_pos_straight(89),
                        end=_pos_straight(97),
                        shape="linear",
                        start_value=0.15,
                        end_value=0.05,
                    ),
                    AutomationSegment(
                        start=_pos_straight(97),
                        end=_pos_straight(113),
                        shape="linear",
                        start_value=0.05,
                        end_value=0.15,
                    ),
                    AutomationSegment(
                        start=_pos_straight(113),
                        end=_pos_straight(153),
                        shape="linear",
                        start_value=0.15,
                        end_value=0.04,
                    ),
                ),
            ),
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="vibrato_chorus"),
                segments=(
                    AutomationSegment(
                        start=_pos_straight(1),
                        end=_pos_straight(57),
                        shape="linear",
                        start_value=0.30,
                        end_value=0.45,
                    ),
                    AutomationSegment(
                        start=_pos_straight(57),
                        end=_pos_straight(65),
                        shape="linear",
                        start_value=0.45,
                        end_value=0.30,
                    ),
                    AutomationSegment(
                        start=_pos_straight(65),
                        end=_pos_straight(89),
                        shape="linear",
                        start_value=0.30,
                        end_value=0.50,
                    ),
                    AutomationSegment(
                        start=_pos_straight(89),
                        end=_pos_straight(97),
                        shape="linear",
                        start_value=0.50,
                        end_value=0.30,
                    ),
                    AutomationSegment(
                        start=_pos_straight(97),
                        end=_pos_straight(113),
                        shape="linear",
                        start_value=0.30,
                        end_value=0.55,
                    ),
                    AutomationSegment(
                        start=_pos_straight(113),
                        end=_pos_straight(153),
                        shape="linear",
                        start_value=0.55,
                        end_value=0.30,
                    ),
                ),
            ),
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="drift"),
                segments=(
                    AutomationSegment(
                        start=_pos_straight(1),
                        end=_pos_straight(65),
                        shape="linear",
                        start_value=0.12,
                        end_value=0.22,
                    ),
                    AutomationSegment(
                        start=_pos_straight(65),
                        end=_pos_straight(97),
                        shape="linear",
                        start_value=0.22,
                        end_value=0.14,
                    ),
                    AutomationSegment(
                        start=_pos_straight(97),
                        end=_pos_straight(113),
                        shape="linear",
                        start_value=0.14,
                        end_value=0.22,
                    ),
                    AutomationSegment(
                        start=_pos_straight(113),
                        end=_pos_straight(153),
                        shape="linear",
                        start_value=0.22,
                        end_value=0.10,
                    ),
                ),
            ),
        ],
    )

    # Lead arp — polyblep pluck with dotted-eighth delay
    score.add_voice(
        "arp",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "saw",
            "osc2_level": 0.5,
            "osc2_waveform": "saw",
            "osc2_detune_cents": 7.0,
            "cutoff_hz": 1800.0,
            "filter_env_amount": 5.0,
            "filter_env_decay": 0.09,
            "resonance_q": 2.0,
            "attack_ms": 2.0,
            "decay_ms": 130.0,
            "sustain_ratio": 0.12,
            "release_ms": 180.0,
            "pitch_drift": 0.08,
            "analog_jitter": 0.7,
        },
        effects=[
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
                            "cutoff_hz": 9000.0,
                            "slope_db_per_oct": 12,
                        },
                    ]
                },
            ),
            EffectSpec("preamp", {"drive": 0.10, "mix": 0.30}),
            EffectSpec(
                "delay",
                {
                    "delay_seconds": DOTTED_EIGHTH,
                    "feedback": 0.38,
                    "mix": 0.22,
                },
            ),
        ],
        mix_db=-5.0,
        normalize_lufs=-22.0,
        pan=0.15,
        velocity_to_params={
            "filter_env_amount": VelocityParamMap(
                min_value=2.5, max_value=7.0, min_velocity=0.6, max_velocity=1.2
            ),
        },
        velocity_db_per_unit=10.0,
        sends=[VoiceSend(target="hall", send_db=-8.0)],
        automation=[
            # Pan drift: slow sway L↔R across the piece
            AutomationSpec(
                target=AutomationTarget(kind="control", name="pan"),
                segments=(
                    AutomationSegment(
                        start=_pos_straight(25),
                        end=_pos_straight(57),
                        shape="linear",
                        start_value=0.15,
                        end_value=-0.10,
                    ),
                    AutomationSegment(
                        start=_pos_straight(57),
                        end=_pos_straight(65),
                        shape="linear",
                        start_value=-0.10,
                        end_value=0.0,
                    ),
                    AutomationSegment(
                        start=_pos_straight(65),
                        end=_pos_straight(89),
                        shape="linear",
                        start_value=0.0,
                        end_value=0.20,
                    ),
                    AutomationSegment(
                        start=_pos_straight(89),
                        end=_pos_straight(97),
                        shape="linear",
                        start_value=0.20,
                        end_value=-0.05,
                    ),
                    AutomationSegment(
                        start=_pos_straight(97),
                        end=_pos_straight(113),
                        shape="linear",
                        start_value=-0.05,
                        end_value=0.15,
                    ),
                    AutomationSegment(
                        start=_pos_straight(113),
                        end=_pos_straight(153),
                        shape="linear",
                        start_value=0.15,
                        end_value=0.0,
                    ),
                ),
            ),
            # Filter cutoff: opens across peaks, closes during breakdowns
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="cutoff_hz"),
                segments=(
                    AutomationSegment(
                        start=_pos_straight(25),
                        end=_pos_straight(57),
                        shape="linear",
                        start_value=1400.0,
                        end_value=2200.0,
                    ),
                    AutomationSegment(
                        start=_pos_straight(57),
                        end=_pos_straight(65),
                        shape="linear",
                        start_value=2200.0,
                        end_value=1200.0,
                    ),
                    AutomationSegment(
                        start=_pos_straight(65),
                        end=_pos_straight(89),
                        shape="linear",
                        start_value=1200.0,
                        end_value=2600.0,
                    ),
                    AutomationSegment(
                        start=_pos_straight(89),
                        end=_pos_straight(97),
                        shape="linear",
                        start_value=2600.0,
                        end_value=1000.0,
                    ),
                    AutomationSegment(
                        start=_pos_straight(97),
                        end=_pos_straight(113),
                        shape="linear",
                        start_value=1000.0,
                        end_value=2400.0,
                    ),
                    AutomationSegment(
                        start=_pos_straight(113),
                        end=_pos_straight(153),
                        shape="linear",
                        start_value=2400.0,
                        end_value=800.0,
                    ),
                ),
            ),
            # Resonance: subtle rise during peaks for more bite
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="resonance_q"),
                segments=(
                    AutomationSegment(
                        start=_pos_straight(25),
                        end=_pos_straight(65),
                        shape="linear",
                        start_value=1.8,
                        end_value=2.0,
                    ),
                    AutomationSegment(
                        start=_pos_straight(65),
                        end=_pos_straight(89),
                        shape="linear",
                        start_value=2.0,
                        end_value=2.8,
                    ),
                    AutomationSegment(
                        start=_pos_straight(89),
                        end=_pos_straight(97),
                        shape="linear",
                        start_value=2.8,
                        end_value=1.6,
                    ),
                    AutomationSegment(
                        start=_pos_straight(97),
                        end=_pos_straight(113),
                        shape="linear",
                        start_value=1.6,
                        end_value=2.6,
                    ),
                    AutomationSegment(
                        start=_pos_straight(113),
                        end=_pos_straight(153),
                        shape="linear",
                        start_value=2.6,
                        end_value=1.4,
                    ),
                ),
            ),
        ],
    )

    # --- Place notes ---
    _place_kick(score)
    _place_closed_hat(score)
    _place_open_hat(score)
    _place_clap(score)
    _place_cowbell(score)
    _place_fills(score)
    _place_bass(score)
    _place_keys(score)
    _place_arp(score)
    return score


# ---------------------------------------------------------------------------
# Placement functions
# ---------------------------------------------------------------------------


def _place_kick(score: Score) -> None:
    for bar in range(1, TOTAL_BARS + 1):
        for beat in range(1, 5):
            if _kick_pattern(bar, beat):
                score.add_note(
                    "kick",
                    start=_pos_straight(bar, beat),
                    duration=1.1,
                    freq=F0,
                    amp_db=-4.0,
                    synth=_KICK_808,
                )


def _place_closed_hat(score: Score) -> None:
    for bar in range(17, TOTAL_BARS + 1):
        synth = _hat_synth(bar)
        subdivs = _chh_subdivs(bar)
        accents = _hat_accents(bar)
        for beat in range(1, 5):
            for n16 in range(4):
                if n16 not in subdivs:
                    continue
                # Skip where OHH plays
                if _has_ohh(bar) and beat in set(_ohh_beats(bar)) and n16 == 2:
                    continue
                score.add_note(
                    "closed_hat",
                    start=_pos(bar, beat, n16),
                    duration=0.025,
                    freq=_hat_pitch(bar, n16),
                    amp_db=accents.get(n16, -14.0),
                    synth=synth,
                )


def _place_open_hat(score: Score) -> None:
    for bar in range(1, TOTAL_BARS + 1):
        if not _has_ohh(bar):
            continue
        for beat in _ohh_beats(bar):
            score.add_note(
                "open_hat",
                start=_pos(bar, beat, 2),
                duration=0.4,
                freq=_hat_pitch(bar, 0),
                amp_db=-9.0,
                synth=_OHH_909,
            )


def _place_clap(score: Score) -> None:
    for bar in range(1, TOTAL_BARS + 1):
        if not _has_clap(bar):
            continue
        for beat in _clap_beats(bar):
            score.add_note(
                "clap",
                start=_pos_straight(bar, beat),
                duration=0.15,
                freq=2640.0,
                amp_db=-6.0,
            )


def _place_cowbell(score: Score) -> None:
    for bar in range(1, TOTAL_BARS + 1):
        if not _has_cowbell(bar):
            continue
        for beat, n16, amp_db in _cowbell_hits(bar):
            score.add_note(
                "cowbell",
                start=_pos(bar, beat, n16),
                duration=0.08,
                freq=_CS5,  # the fifth of F#, pitched cowbell
                amp_db=amp_db,
            )


def _place_fills(score: Score) -> None:
    """Drum fills at section transitions — hat rolls, clap rolls, cowbell flourishes."""
    for bar in range(1, TOTAL_BARS + 1):
        is_8bar_boundary = bar % 8 == 0
        is_16bar_boundary = bar % 16 == 0
        in_groove = 25 <= bar <= 56 or 65 <= bar <= 88 or 97 <= bar <= 112
        pre_breakdown = bar in {56, 88}
        pre_peak = bar in {64, 96}

        if not is_8bar_boundary:
            continue

        # Every 8-bar boundary: hat roll on beat 4
        for n16 in range(4):
            score.add_note(
                "closed_hat",
                start=_pos(bar, 4, n16),
                duration=0.02,
                freq=_hat_pitch(bar, n16),
                amp_db=-8.0 - n16 * 0.5,
                synth=_HAT_909,
            )

        # Every 16-bar boundary: add clap roll
        if is_16bar_boundary:
            for n16 in range(4):
                score.add_note(
                    "clap",
                    start=_pos(bar, 4, n16),
                    duration=0.08,
                    freq=2640.0,
                    amp_db=-8.0 + n16 * 1.0,
                )

        # Before breakdowns: bigger fills — clap roll + cowbell flourish
        if pre_breakdown:
            # Extra clap roll on beat 3
            for n16 in range(4):
                score.add_note(
                    "clap",
                    start=_pos(bar, 3, n16),
                    duration=0.08,
                    freq=2640.0,
                    amp_db=-10.0 + n16 * 0.8,
                )
            # Cowbell flourish: 16th-note triplet feel on beat 3
            for n16 in range(3):
                score.add_note(
                    "cowbell",
                    start=_pos(bar, 3, n16),
                    duration=0.06,
                    freq=_CS5,
                    amp_db=-10.0 + n16 * 1.5,
                )

        # Before peaks: building hat roll across beats 3-4
        if pre_peak:
            for beat in (3, 4):
                for n16 in range(4):
                    score.add_note(
                        "closed_hat",
                        start=_pos(bar, beat, n16),
                        duration=0.02,
                        freq=_hat_pitch(bar, n16),
                        amp_db=-10.0 + (beat - 3) * 2.0 + n16 * 0.5,
                        synth=_HAT_909,
                    )

        # Cowbell 16th-note triplet on beat 4 every 16 bars during grooves
        if is_16bar_boundary and in_groove:
            for n16 in range(3):
                score.add_note(
                    "cowbell",
                    start=_pos(bar, 4, n16),
                    duration=0.06,
                    freq=_CS5,
                    amp_db=-12.0 + n16 * 1.0,
                )


def _place_bass(score: Score) -> None:
    for bar in range(17, TOTAL_BARS + 1):
        _place_bass_bar(score, bar)


def _place_keys(score: Score) -> None:
    for bar in range(1, TOTAL_BARS + 1):
        in_breakdown = 57 <= bar <= 64 or 89 <= bar <= 96
        if in_breakdown:
            # Breakdowns: just root + fifth, quiet
            for freq, db in [(_FS3, -10.0), (_CS4, -14.0)]:
                score.add_note(
                    "keys",
                    start=_pos_straight(bar, 1),
                    duration=BAR * 0.95,
                    freq=freq,
                    amp_db=db,
                )
            continue

        if bar > 136:
            # Outro_b: fade the pad — only root, getting quieter
            fade_db = -10.0 - (bar - 136) * 0.8
            score.add_note(
                "keys",
                start=_pos_straight(bar, 1),
                duration=BAR * 0.95,
                freq=_FS3,
                amp_db=fade_db,
            )
            continue

        chord = _chord_for_bar(bar)
        # Intro: add second chord tone by bar 3 for faster complexity
        if bar <= 2:
            # First 2 bars: root only from chord IV
            score.add_note(
                "keys",
                start=_pos_straight(bar, 1),
                duration=BAR * 0.92,
                freq=_FS3,
                amp_db=-8.0,
            )
            continue

        for freq, db in chord:
            score.add_note(
                "keys",
                start=_pos_straight(bar, 1),
                duration=BAR * 0.92,
                freq=freq,
                amp_db=db,
            )

        # Upper octave shimmer during peaks
        in_peak = 65 <= bar <= 88 or 97 <= bar <= 112
        if in_peak and bar % 2 == 0:
            score.add_note(
                "keys",
                start=_pos_straight(bar, 1),
                duration=BAR * 0.85,
                freq=_FS4,
                amp_db=-20.0,
            )


def _place_arp(score: Score) -> None:
    """Place arp notes bar by bar using the cell system."""
    for bar in range(1, TOTAL_BARS + 1):
        if not _has_arp(bar):
            continue
        pool = _arp_pool_for_bar(bar)
        cell_indices, velocities = _arp_cell_for_bar(bar)
        base_amp = _arp_amp_for_bar(bar)

        for step in range(8):
            n16 = step  # 8 x 16th notes per bar = 2 beats
            beat = 1 + (n16 // 4)
            sub = n16 % 4

            freq = pool[cell_indices[step] % len(pool)]
            vel = velocities[step]
            amp_offset = _AMP_OFFSETS[step]

            score.add_note(
                "arp",
                start=_pos(bar, beat, sub),
                duration=S16 * 1.2,
                freq=freq,
                amp_db=base_amp + amp_offset,
                velocity=vel,
            )

        # Second half of bar: repeat cell with slight variation
        for step in range(8):
            n16 = step
            beat = 3 + (n16 // 4)
            sub = n16 % 4

            # Slight pitch variation: swap neighboring indices occasionally
            idx = cell_indices[step] % len(pool)
            if step == 3 and bar % 3 == 0:
                idx = (idx + 1) % len(pool)
            freq = pool[idx]
            vel = velocities[(step + 1) % 8]  # shifted velocity for variety
            amp_offset = _AMP_OFFSETS[(step + 2) % 8]

            score.add_note(
                "arp",
                start=_pos(bar, beat, sub),
                duration=S16 * 1.2,
                freq=freq,
                amp_db=base_amp + amp_offset - 1.0,  # slightly quieter second half
                velocity=vel,
            )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

PIECES: dict[str, PieceDefinition] = {
    "night_lattice": PieceDefinition(
        name="night_lattice",
        output_name="night_lattice",
        build_score=build_score,
        sections=(
            PieceSection("intro", _pos_straight(1), _pos_straight(9)),
            PieceSection("build_a", _pos_straight(9), _pos_straight(17)),
            PieceSection("build_b", _pos_straight(17), _pos_straight(25)),
            PieceSection("groove_a", _pos_straight(25), _pos_straight(33)),
            PieceSection("groove_b", _pos_straight(33), _pos_straight(57)),
            PieceSection("breakdown_1", _pos_straight(57), _pos_straight(65)),
            PieceSection("peak_1", _pos_straight(65), _pos_straight(89)),
            PieceSection("breakdown_2", _pos_straight(89), _pos_straight(97)),
            PieceSection("peak_2", _pos_straight(97), _pos_straight(113)),
            PieceSection("settle", _pos_straight(113), _pos_straight(121)),
            PieceSection("outro_a", _pos_straight(121), _pos_straight(137)),
            PieceSection("outro_b", _pos_straight(137), _pos_straight(153)),
        ),
    ),
}
