"""Amber Room — warm deep house inspired by Moodymann's dusty warmth.

150 bars at 120 BPM (~5 min).  Shuffled 16ths, long-tailed 808 kick with
body filter, septimal organ chords alternating dark/warm voicings, a simple
melodic hook on a brighter organ, composed bass lines, and constantly
shifting drum patterns.  Heavy automation throughout.

Section map:
      1-  8  intro: kick alone
      9- 16  +hats (sparse 8ths), building
     17- 24  +bass (simple), +organ chord I, +clap
     25- 32  +fuller hats (16ths), bass gets active, chord alternation begins
     33- 40  groove A: +OHH, +ghost snare, +shaker
     41- 48  groove A continues, patterns shift
     49- 56  breakdown 1: drums thin, organ sustains, melody teased
     57- 64  build B: everything returns, melodic hook enters
     65- 72  groove B: hook established, patterns shift
     73- 80  peak 1: hat filter widest, bass most active
     81- 88  peak 1 continues, patterns shift again
     89- 96  breakdown 2: bigger drop, pad + reverb only
     97-104  build C: elements return, hook varies
    105-112  peak 2: full energy, all elements
    113-120  peak 2 continues, everything cooking
    121-128  settle: elements thin
    129-136  outro: hat to 8ths, elements drop
    137-144  outro: kick + bass + sparse hat
    145-150  kick alone → end
"""

from __future__ import annotations

from code_musics.automation import (
    AutomationSegment,
    AutomationSpec,
    AutomationTarget,
)
from code_musics.drum_helpers import add_drum_voice, setup_drum_bus
from code_musics.meter import Groove
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score, VoiceSend

BPM = 120.0
BEAT = 60.0 / BPM
BAR = 4.0 * BEAT
S16 = BEAT / 4.0
GROOVE = Groove(
    subdivision="sixteenth",
    timing_offsets=(0.0, 0.18),
    velocity_weights=(1.0, 1.0),
)
F0 = 55.0  # A1
TOTAL_BARS = 150

P1 = 1.0  # 55 Hz   A1
P2 = 2.0  # 110 Hz  A2
P3 = 3.0  # 165 Hz  ~E3
P4 = 4.0  # 220 Hz  A3
P5 = 5.0  # 275 Hz  ~C#4
P6 = 6.0  # 330 Hz  E4
P7 = 7.0  # 385 Hz  ~Bb4 (septimal)
P8 = 8.0  # 440 Hz  A4
P9 = 9.0  # 495 Hz  ~B4


def _pos(bar: int, beat: int = 1, n16: int = 0) -> float:
    """Bar/beat/16th → seconds.  Groove-shifted on upbeat 16ths."""
    base = (bar - 1) * BAR + (beat - 1) * BEAT + n16 * S16
    return base + S16 * GROOVE.timing_offset_at(n16)


def _pos_straight(bar: int, beat: int = 1, n16: int = 0) -> float:
    return (bar - 1) * BAR + (beat - 1) * BEAT + n16 * S16


TOTAL_DUR = _pos_straight(TOTAL_BARS + 1)

# ---------------------------------------------------------------------------
# Kick synth overrides
# ---------------------------------------------------------------------------

_KICK_808: dict = {
    "tone_decay_s": 0.80,
    "tone_envelope": [
        {"time": 0.0, "value": 1.0},
        {"time": 0.55, "value": 0.75, "curve": "linear"},
        {"time": 0.85, "value": 0.15, "curve": "exponential"},
        {"time": 1.0, "value": 0.0, "curve": "linear"},
    ],
    "tone_sweep_ratio": 2.8,
    "tone_sweep_decay_s": 0.055,
    "tone_wave": "sine",
    "tone_second_harmonic": 0.06,
    "tone_punch": 0.10,
    "filter_mode": "lowpass",
    "filter_cutoff_hz": 800.0,
    "filter_q": 0.8,
    "filter_envelope": [
        {"time": 0.0, "value": 2400.0},
        {"time": 0.08, "value": 800.0, "curve": "exponential"},
        {"time": 1.0, "value": 400.0, "curve": "linear"},
    ],
    "exciter_level": 0.01,
    "exciter_decay_s": 0.008,
    "exciter_center_hz": 2000.0,
    "noise_level": 0.008,
    "noise_decay_s": 0.022,
    "noise_center_ratio": 10.91,
}

# Peak 2: barely-there warmth — house, not techno
_KICK_PEAK: dict = {
    **_KICK_808,
    "tone_shaper": "tanh",
    "tone_shaper_drive": 0.08,
    "tone_shaper_mix": 0.20,
}

# ---------------------------------------------------------------------------
# Hat synth — 909 noise-forward character
# ---------------------------------------------------------------------------

_HAT_909: dict = {
    "metallic_type": "partials",
    "metallic_n_partials": 3,
    "metallic_brightness": 0.25,
    "metallic_density": 0.7,
    "metallic_decay_s": 0.008,
    "metallic_level": 1.0,
    "metallic_filter_q": 0.65,
    "noise_level": 0.92,
    "noise_decay_s": 0.008,
    "exciter_level": 0.65,
    "exciter_decay_s": 0.0003,
    "tone_type": None,
}

_OHH_909: dict = {
    **_HAT_909,
    "metallic_decay_s": 0.140,
    "noise_decay_s": 0.140,
    "exciter_level": 0.40,
    "metallic_filter_q": 0.7,
}

# ---------------------------------------------------------------------------
# Hat pitch pool — harmonic-series frequencies of F0=55 Hz, in key
# ---------------------------------------------------------------------------

HAT_A = 7040.0  # H128 = 55 * 128 — root (A), dark
HAT_CS = 8800.0  # H160 = 55 * 160 — major 3rd (C#), medium
HAT_SEPT = 9240.0  # H168 = 55 * 168 — septimal (21/16), spicy
HAT_E = 10560.0  # H192 = 55 * 192 — fifth (E), bright

# (primary, secondary, tertiary) per section — primary on downbeats,
# secondary on "and" (pos 2), tertiary on ghost 16ths (pos 1, 3)
_HAT_PALETTES: list[tuple[int, tuple[float, float, float]]] = [
    (16, (HAT_CS, HAT_CS, HAT_CS)),  # intro: neutral
    (24, (HAT_CS, HAT_A, HAT_CS)),  # build: warm contrast enters
    (48, (HAT_CS, HAT_A, HAT_SEPT)),  # groove enter + A: septimal spice
    (56, (HAT_A, HAT_A, HAT_A)),  # breakdown 1: dark, withdrawn
    (72, (HAT_CS, HAT_SEPT, HAT_E)),  # build B + groove B: brightening
    (88, (HAT_CS, HAT_SEPT, HAT_E)),  # peak 1: handled specially below
    (96, (HAT_A, HAT_A, HAT_A)),  # breakdown 2: dark again
    (120, (HAT_CS, HAT_SEPT, HAT_E)),  # peak 2: handled specially below
    (128, (HAT_CS, HAT_A, HAT_CS)),  # settle: narrowing
    (999, (HAT_A, HAT_A, HAT_A)),  # outro: settling dark
]

# Peak rotation: cycle through 4 different palette assignments per 4-bar phrase
_PEAK_ROTATIONS: list[tuple[float, float, float]] = [
    (HAT_CS, HAT_A, HAT_SEPT),
    (HAT_E, HAT_CS, HAT_A),
    (HAT_SEPT, HAT_E, HAT_CS),
    (HAT_A, HAT_SEPT, HAT_E),
]

_N16_TO_TIER: dict[int, int] = {0: 0, 1: 2, 2: 1, 3: 2}  # pos → palette index


def _hat_pitch(bar: int, n16: int) -> float:
    """Return hat freq for a given bar and 16th position."""
    # Peak sections: rotate through full pool
    if 73 <= bar <= 88 or 105 <= bar <= 120:
        start = 73 if bar <= 88 else 105
        rot_idx = ((bar - start) // 4) % len(_PEAK_ROTATIONS)
        palette = _PEAK_ROTATIONS[rot_idx]
    else:
        palette = (HAT_CS, HAT_CS, HAT_CS)  # fallback
        for cutoff_bar, pal in _HAT_PALETTES:
            if bar <= cutoff_bar:
                palette = pal
                break
    tier = _N16_TO_TIER.get(n16, 0)
    return palette[tier]


# Filter envelopes evolve gradually (indexed by progress through the track)
_HAT_FILTERS: list[tuple[int, list[dict]]] = [
    (
        16,
        [
            {"time": 0.0, "value": 5000.0},
            {"time": 0.4, "value": 3500.0, "curve": "exponential"},
            {"time": 1.0, "value": 2500.0, "curve": "linear"},
        ],
    ),
    (
        32,
        [
            {"time": 0.0, "value": 7000.0},
            {"time": 0.35, "value": 5000.0, "curve": "exponential"},
            {"time": 1.0, "value": 3500.0, "curve": "linear"},
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
        120,
        [
            {"time": 0.0, "value": 9000.0},
            {"time": 0.35, "value": 6000.0, "curve": "exponential"},
            {"time": 1.0, "value": 4000.0, "curve": "linear"},
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
# Drum pattern helpers — vary every 4-8 bars
# ---------------------------------------------------------------------------

# Which 16th subdivisions the CHH plays per beat, by 4-bar phrase character
_PAT_8THS = {0, 2}  # just 8th notes
_PAT_16THS = {0, 1, 2, 3}  # full 16ths
_PAT_16THS_NO_GHOST = {0, 2, 3}  # skip "e", keep "a"
_PAT_SPARSE = {0}  # downbeat only


def _chh_subdivs(bar: int) -> set[int]:
    """Which 16th subdivisions the closed hat plays, varying per phrase."""
    if bar <= 16:
        return _PAT_8THS
    if bar <= 24:
        return _PAT_16THS_NO_GHOST if (bar - 1) % 8 < 4 else _PAT_8THS
    if bar <= 48:
        return _PAT_16THS if (bar - 1) % 4 < 3 else _PAT_16THS_NO_GHOST
    if bar <= 56:
        return _PAT_8THS  # breakdown
    if bar <= 88:
        # Main groove: shift pattern every 4 bars
        phrase = ((bar - 57) // 4) % 3
        return [_PAT_16THS, _PAT_16THS_NO_GHOST, _PAT_16THS][phrase]
    if bar <= 96:
        return _PAT_SPARSE  # breakdown 2
    if bar <= 120:
        phrase = ((bar - 97) // 4) % 4
        return [_PAT_16THS, _PAT_16THS_NO_GHOST, _PAT_16THS, _PAT_16THS_NO_GHOST][
            phrase
        ]
    if bar <= 136:
        return _PAT_8THS  # settle/outro
    return _PAT_SPARSE


# Ghost hat accent levels shift per 4-bar phrase
_ACCENTS_A: dict[int, float] = {0: -10.0, 1: -17.0, 2: -12.0, 3: -18.0}
_ACCENTS_B: dict[int, float] = {0: -11.0, 1: -16.0, 2: -10.0, 3: -17.0}  # "and" louder
_ACCENTS_C: dict[int, float] = {0: -10.0, 1: -15.0, 2: -13.0, 3: -15.0}  # flatter


def _hat_accents(bar: int) -> dict[int, float]:
    phase = ((bar - 1) // 4) % 3
    return [_ACCENTS_A, _ACCENTS_B, _ACCENTS_C][phase]


def _has_ohh(bar: int) -> bool:
    if 33 <= bar <= 48:
        return True
    if 57 <= bar <= 88:
        return True
    return 97 <= bar <= 120


def _ohh_beats(bar: int) -> list[int]:
    """Which beats get an open hat.  Varies per phrase."""
    phase = ((bar - 1) // 8) % 3
    if phase == 0:
        return [2, 4]
    if phase == 1:
        return [2]  # just beat 2 — breathes
    return [4]  # just beat 4 — pushes forward


def _kick_pattern(bar: int, beat: int) -> bool:
    """Whether the kick hits on this beat.  Usually 4otf, with variations."""
    if bar <= 8 or 137 <= bar <= 150:
        return beat == 1  # intro/outro: beat 1 only
    if 49 <= bar <= 56 or 89 <= bar <= 96:
        return beat in {1, 3}  # breakdowns: half time
    # Occasional beat-3 skip for breathing room (every 8th bar in groove)
    return not (bar % 8 == 0 and 33 <= bar <= 120 and beat == 3)


def _has_snare(bar: int) -> bool:
    return 33 <= bar <= 48 or 57 <= bar <= 88 or 97 <= bar <= 120


# Snare ghost pitch pool — chord-tone fundamentals
SNR_E = 165.0  # P3 — ~E3
SNR_A = 220.0  # P4 — A3 (original)
SNR_CS = 275.0  # P5 — ~C#4


def _snare_ghosts(bar: int) -> list[tuple[int, int, float, float]]:
    """Return (beat, n16, amp_db, freq) for ghost snare hits.

    Steady placement with gentle pitch drift — root on the main ghost,
    fifth or third on the secondary depending on 8-bar phrasing.
    """
    second_pitch = SNR_E if ((bar - 1) // 8) % 2 == 0 else SNR_CS
    return [
        (2, 1, -16.0, SNR_A),
        (3, 3, -18.0, second_pitch),
    ]


def _has_clap(bar: int) -> bool:
    return 17 <= bar <= 48 or 57 <= bar <= 120


def _clap_beats(bar: int) -> list[int]:
    phase = ((bar - 1) // 8) % 3
    if phase == 0:
        return [2, 4]
    if phase == 1:
        return [2]  # just 2, more relaxed
    return [2, 4]


def _has_shaker(bar: int) -> bool:
    if not (33 <= bar <= 48 or 57 <= bar <= 88 or 97 <= bar <= 120):
        return False
    # Intermittent: 2 bars on, 2 bars off within active sections
    return ((bar - 1) // 2) % 2 == 0


# ---------------------------------------------------------------------------
# Chord voicings — alternate every 4 bars once alternation starts (bar 25)
# ---------------------------------------------------------------------------

# Chord I: dark septimal — A + E + Bb(sept)
_CHORD_I: list[tuple[float, float]] = [(P4, -8.0), (P6, -12.0), (P7, -14.0)]
# Chord II: warm major-septimal — A + C# + Bb(sept)
_CHORD_II: list[tuple[float, float]] = [(P4, -8.0), (P5, -13.0), (P7, -14.0)]


def _chord_for_bar(bar: int) -> list[tuple[float, float]]:
    if bar < 25:
        return _CHORD_I
    # Alternate every 4 bars
    return _CHORD_II if ((bar - 25) // 4) % 2 == 1 else _CHORD_I


# ---------------------------------------------------------------------------
# Melodic fragments — hand-placed gestures, Moodymann-style found-sound feel.
# JI A dorian scale (from A4 = 440 Hz) for human-sounding runs.
# Each fragment: (offset_seconds, freq_hz, duration_seconds, amp_db)
# offset_seconds is relative to the fragment's anchor bar start.
# ---------------------------------------------------------------------------
# Bass phrases — composed per section, more active as track builds
# ---------------------------------------------------------------------------


def _place_bass_bar(score: Score, bar: int) -> None:
    """Place bass notes for a single bar.  Pattern varies by section."""
    if bar <= 24:
        # Simple: root on beat 1, long sustain
        score.add_note(
            "bass",
            start=_pos_straight(bar, 1),
            duration=BEAT * 3.0,
            partial=P2,
            amp_db=-6.0,
        )
        if bar % 4 == 0 and bar > 20:
            # Pickup fifth
            score.add_note(
                "bass",
                start=_pos_straight(bar, 4, 2),
                duration=BEAT * 0.5,
                partial=P3,
                amp_db=-10.0,
            )

    elif bar <= 48:
        # More active: root + fifth + octave movement
        phase = ((bar - 25) // 4) % 3
        if phase == 0:
            score.add_note(
                "bass",
                start=_pos_straight(bar, 1),
                duration=BEAT * 2.5,
                partial=P2,
                amp_db=-6.0,
            )
            score.add_note(
                "bass",
                start=_pos_straight(bar, 3, 2),
                duration=BEAT * 0.8,
                partial=P3,
                amp_db=-9.0,
            )
        elif phase == 1:
            score.add_note(
                "bass",
                start=_pos_straight(bar, 1),
                duration=BEAT * 1.5,
                partial=P2,
                amp_db=-6.0,
            )
            score.add_note(
                "bass",
                start=_pos_straight(bar, 3),
                duration=BEAT * 1.0,
                partial=P3,
                amp_db=-9.0,
            )
            score.add_note(
                "bass",
                start=_pos_straight(bar, 4, 1),
                duration=BEAT * 0.5,
                partial=P4,
                amp_db=-10.0,
            )
        else:
            score.add_note(
                "bass",
                start=_pos_straight(bar, 1),
                duration=BEAT * 3.5,
                partial=P2,
                amp_db=-6.0,
            )

    elif 49 <= bar <= 56 or 89 <= bar <= 96:
        # Breakdowns: sustained root, quiet
        score.add_note(
            "bass",
            start=_pos_straight(bar, 1),
            duration=BAR * 0.9,
            partial=P2,
            amp_db=-10.0,
        )

    elif bar <= 88:
        # Full groove / peak: most active bass
        phase = ((bar - 57) // 4) % 4
        if phase == 0:
            score.add_note(
                "bass",
                start=_pos_straight(bar, 1),
                duration=BEAT * 2.0,
                partial=P2,
                amp_db=-5.0,
            )
            score.add_note(
                "bass",
                start=_pos_straight(bar, 3),
                duration=BEAT * 0.75,
                partial=P3,
                amp_db=-8.0,
            )
            score.add_note(
                "bass",
                start=_pos_straight(bar, 4),
                duration=BEAT * 0.75,
                partial=P2,
                amp_db=-7.0,
            )
        elif phase == 1:
            score.add_note(
                "bass",
                start=_pos_straight(bar, 1),
                duration=BEAT * 1.5,
                partial=P2,
                amp_db=-5.0,
            )
            score.add_note(
                "bass",
                start=_pos_straight(bar, 2, 2),
                duration=BEAT * 1.0,
                partial=P3,
                amp_db=-8.0,
            )
            score.add_note(
                "bass",
                start=_pos_straight(bar, 4, 1),
                duration=BEAT * 0.5,
                partial=P4,
                amp_db=-9.0,
            )
        elif phase == 2:
            score.add_note(
                "bass",
                start=_pos_straight(bar, 1),
                duration=BEAT * 3.0,
                partial=P2,
                amp_db=-5.0,
            )
            score.add_note(
                "bass",
                start=_pos_straight(bar, 4, 2),
                duration=BEAT * 0.4,
                partial=P3,
                amp_db=-9.0,
            )
        else:
            # Busier phrase with septimal push
            score.add_note(
                "bass",
                start=_pos_straight(bar, 1),
                duration=BEAT * 1.5,
                partial=P2,
                amp_db=-5.0,
            )
            score.add_note(
                "bass",
                start=_pos_straight(bar, 2, 2),
                duration=BEAT * 0.75,
                partial=P3,
                amp_db=-8.0,
            )
            score.add_note(
                "bass",
                start=_pos_straight(bar, 3, 2),
                duration=BEAT * 0.75,
                partial=P4,
                amp_db=-9.0,
            )
            score.add_note(
                "bass",
                start=_pos_straight(bar, 4, 2),
                duration=BEAT * 0.4,
                partial=P2,
                amp_db=-7.0,
            )

    elif bar <= 120:
        # Peak 2: same active patterns
        phase = ((bar - 97) // 4) % 4
        if phase == 0:
            score.add_note(
                "bass",
                start=_pos_straight(bar, 1),
                duration=BEAT * 2.0,
                partial=P2,
                amp_db=-5.0,
            )
            score.add_note(
                "bass",
                start=_pos_straight(bar, 3, 2),
                duration=BEAT * 0.8,
                partial=P3,
                amp_db=-8.0,
            )
        elif phase == 1:
            score.add_note(
                "bass",
                start=_pos_straight(bar, 1),
                duration=BEAT * 1.5,
                partial=P2,
                amp_db=-5.0,
            )
            score.add_note(
                "bass",
                start=_pos_straight(bar, 3),
                duration=BEAT * 0.8,
                partial=P3,
                amp_db=-8.0,
            )
            score.add_note(
                "bass",
                start=_pos_straight(bar, 4, 1),
                duration=BEAT * 0.5,
                partial=P4,
                amp_db=-10.0,
            )
        else:
            score.add_note(
                "bass",
                start=_pos_straight(bar, 1),
                duration=BEAT * 3.0,
                partial=P2,
                amp_db=-5.0,
            )

    else:
        # Outro: simple root, fading
        score.add_note(
            "bass",
            start=_pos_straight(bar, 1),
            duration=BEAT * 2.0,
            partial=P2,
            amp_db=-8.0,
        )


# ---------------------------------------------------------------------------
# Build score
# ---------------------------------------------------------------------------


def build_score() -> Score:
    score = Score(
        f0_hz=F0,
        sample_rate=44_100,
        master_effects=[
            EffectSpec("preamp", {"drive": 0.35, "mix": 0.7}),
            EffectSpec(
                "compressor",
                {
                    "threshold_db": -16.0,
                    "ratio": 2.2,
                    "attack_ms": 30.0,
                    "release_ms": 300.0,
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
                    "ratio": 1.6,
                    "attack_ms": 25.0,
                    "release_ms": 200.0,
                    "knee_db": 8.0,
                    "topology": "feedback",
                    "detector_mode": "rms",
                    "detector_bands": [
                        {"kind": "highpass", "cutoff_hz": 60.0, "slope_db_per_oct": 12},
                    ],
                },
            ),
            EffectSpec("saturation", {"mode": "triode", "drive": 2.0, "mix": 0.30}),
        ],
        return_db=0.0,
    )

    score.add_send_bus(
        "hall",
        effects=[
            EffectSpec(
                "reverb",
                {"room_size": 0.92, "damping": 0.55, "wet_level": 1.0},
            ),
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "lowpass",
                            "cutoff_hz": 4000.0,
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

    add_drum_voice(
        score,
        "snare",
        engine="drum_voice",
        preset="brush",
        drum_bus=drum_bus,
        send_db=-5.0,
        mix_db=-6.0,
    )
    score.voices["snare"].sends.append(VoiceSend(target="hall", send_db=-10.0))

    add_drum_voice(
        score,
        "shaker",
        engine="drum_voice",
        preset="shaped_hit",
        drum_bus=drum_bus,
        send_db=-6.0,
        mix_db=-12.0,
        synth_overrides={"noise_level": 0.8, "noise_center_ratio": 2.5},
    )
    score.voices["shaker"].sends.append(VoiceSend(target="hall", send_db=-16.0))

    # Bass
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "polyblep",
            "preset": "sub_bass",
            "cutoff_hz": 200.0,
            "resonance_q": 1.0,
        },
        effects=[
            EffectSpec(
                "compressor", {"preset": "kick_duck", "sidechain_source": "kick"}
            ),
        ],
        mix_db=-1.0,
        velocity_humanize=None,
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="cutoff_hz"),
                segments=(
                    AutomationSegment(
                        start=_pos_straight(1),
                        end=_pos_straight(49),
                        shape="linear",
                        start_value=170.0,
                        end_value=240.0,
                    ),
                    AutomationSegment(
                        start=_pos_straight(49),
                        end=_pos_straight(73),
                        shape="linear",
                        start_value=200.0,
                        end_value=300.0,
                    ),
                    AutomationSegment(
                        start=_pos_straight(73),
                        end=_pos_straight(97),
                        shape="linear",
                        start_value=300.0,
                        end_value=220.0,
                    ),
                    AutomationSegment(
                        start=_pos_straight(97),
                        end=_pos_straight(121),
                        shape="linear",
                        start_value=220.0,
                        end_value=280.0,
                    ),
                    AutomationSegment(
                        start=_pos_straight(121),
                        end=_pos_straight(150),
                        shape="linear",
                        start_value=280.0,
                        end_value=160.0,
                    ),
                ),
            ),
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="resonance_q"),
                segments=(
                    AutomationSegment(
                        start=_pos_straight(1),
                        end=_pos_straight(73),
                        shape="linear",
                        start_value=1.0,
                        end_value=1.6,
                    ),
                    AutomationSegment(
                        start=_pos_straight(73),
                        end=_pos_straight(130),
                        shape="linear",
                        start_value=1.6,
                        end_value=1.0,
                    ),
                ),
            ),
        ],
    )

    # Organ pad — warm chords
    score.add_voice(
        "keys",
        synth_defaults={
            "engine": "organ",
            "drawbars": [0, 8, 4, 6, 0, 0, 0, 0, 0],
            "click": 0.06,
            "click_brightness": 0.3,
            "vibrato_depth": 0.08,
            "vibrato_rate_hz": 5.5,
            "vibrato_chorus": 0.4,
            "drift": 0.15,
            "drift_rate_hz": 0.05,
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
                            "cutoff_hz": 3000.0,
                            "slope_db_per_oct": 12,
                        },
                        {
                            "kind": "highpass",
                            "cutoff_hz": 150.0,
                            "slope_db_per_oct": 12,
                        },
                    ]
                },
            ),
            EffectSpec("preamp", {"drive": 0.20, "mix": 0.5}),
            EffectSpec(
                "compressor", {"preset": "kick_duck_hard", "sidechain_source": "kick"}
            ),
        ],
        mix_db=-2.5,
        normalize_lufs=-22.0,
        velocity_humanize=None,
        sends=[VoiceSend(target="hall", send_db=-6.0)],
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="vibrato_depth"),
                segments=(
                    AutomationSegment(
                        start=_pos_straight(1),
                        end=_pos_straight(49),
                        shape="linear",
                        start_value=0.04,
                        end_value=0.10,
                    ),
                    AutomationSegment(
                        start=_pos_straight(49),
                        end=_pos_straight(89),
                        shape="linear",
                        start_value=0.06,
                        end_value=0.18,
                    ),
                    AutomationSegment(
                        start=_pos_straight(89),
                        end=_pos_straight(97),
                        shape="linear",
                        start_value=0.18,
                        end_value=0.06,
                    ),
                    AutomationSegment(
                        start=_pos_straight(97),
                        end=_pos_straight(121),
                        shape="linear",
                        start_value=0.06,
                        end_value=0.16,
                    ),
                    AutomationSegment(
                        start=_pos_straight(121),
                        end=_pos_straight(145),
                        shape="linear",
                        start_value=0.16,
                        end_value=0.03,
                    ),
                ),
            ),
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="vibrato_chorus"),
                segments=(
                    AutomationSegment(
                        start=_pos_straight(1),
                        end=_pos_straight(73),
                        shape="linear",
                        start_value=0.3,
                        end_value=0.6,
                    ),
                    AutomationSegment(
                        start=_pos_straight(73),
                        end=_pos_straight(140),
                        shape="linear",
                        start_value=0.6,
                        end_value=0.25,
                    ),
                ),
            ),
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="drift"),
                segments=(
                    AutomationSegment(
                        start=_pos_straight(1),
                        end=_pos_straight(89),
                        shape="linear",
                        start_value=0.10,
                        end_value=0.22,
                    ),
                    AutomationSegment(
                        start=_pos_straight(89),
                        end=_pos_straight(145),
                        shape="linear",
                        start_value=0.22,
                        end_value=0.08,
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
    _place_snare(score)
    _place_shaker(score)
    _place_fills(score)
    _place_bass(score)
    _place_keys(score)
    return score


# ---------------------------------------------------------------------------
# Placement functions
# ---------------------------------------------------------------------------


def _place_kick(score: Score) -> None:
    for bar in range(1, TOTAL_BARS + 1):
        synth = _KICK_PEAK if 105 <= bar <= 120 else _KICK_808
        for beat in range(1, 5):
            if _kick_pattern(bar, beat):
                score.add_note(
                    "kick",
                    start=_pos_straight(bar, beat),
                    duration=1.2,
                    freq=55.0,
                    amp_db=-6.0,
                    synth=synth,
                )


def _place_closed_hat(score: Score) -> None:
    for bar in range(9, TOTAL_BARS + 1):
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


def _place_snare(score: Score) -> None:
    for bar in range(1, TOTAL_BARS + 1):
        if not _has_snare(bar):
            continue
        for beat, n16, db, freq in _snare_ghosts(bar):
            score.add_note(
                "snare",
                start=_pos(bar, beat, n16),
                duration=0.12,
                freq=freq,
                amp_db=db,
            )


def _place_shaker(score: Score) -> None:
    for bar in range(1, TOTAL_BARS + 1):
        if not _has_shaker(bar):
            continue
        for beat in range(1, 5):
            score.add_note(
                "shaker",
                start=_pos(bar, beat, 2),
                duration=0.03,
                freq=6160.0,
                amp_db=-14.0,
            )
            if beat % 2 == 0:
                score.add_note(
                    "shaker",
                    start=_pos(bar, beat, 3),
                    duration=0.025,
                    freq=6600.0,
                    amp_db=-18.0,
                )


def _place_fills(score: Score) -> None:
    """Drum fills before section transitions — hat rolls and snare flams."""
    # Bars before major transitions get a fill in beat 4
    fill_bars = {16, 24, 32, 48, 56, 72, 88, 96, 112, 120, 128}
    for bar in fill_bars:
        if bar > TOTAL_BARS:
            continue
        # 16th-note hat roll on beat 4
        for n16 in range(4):
            score.add_note(
                "closed_hat",
                start=_pos(bar, 4, n16),
                duration=0.02,
                freq=_hat_pitch(bar, n16),
                amp_db=-8.0 - n16 * 0.5,
                synth=_HAT_909,
            )
        # Snare flam on beat 4 "and" (if snare is active in this section)
        if _has_snare(bar):
            score.add_note(
                "snare",
                start=_pos(bar, 4, 2),
                duration=0.12,
                freq=220.0,
                amp_db=-10.0,
            )
            # Ghost flam slightly before
            score.add_note(
                "snare",
                start=_pos(bar, 4, 1),
                duration=0.08,
                freq=220.0,
                amp_db=-16.0,
            )

    # Bigger fills before breakdowns (bars 48, 88) — clap roll
    for bar in [48, 88]:
        if bar > TOTAL_BARS:
            continue
        for n16 in range(4):
            score.add_note(
                "clap",
                start=_pos(bar, 4, n16),
                duration=0.08,
                freq=2640.0,
                amp_db=-8.0 + n16 * 1.0,
            )


def _place_bass(score: Score) -> None:
    for bar in range(17, TOTAL_BARS + 1):
        if bar >= 145:
            continue
        _place_bass_bar(score, bar)


def _place_keys(score: Score) -> None:
    for bar in range(17, 137):
        if 49 <= bar <= 56:
            # Breakdown 1: just root + fifth, quiet
            for partial, db in [(P4, -10.0), (P6, -14.0)]:
                score.add_note(
                    "keys",
                    start=_pos_straight(bar, 1),
                    duration=BAR * 0.95,
                    partial=partial,
                    amp_db=db,
                )
            continue
        if 89 <= bar <= 96:
            # Breakdown 2: just root, quieter
            score.add_note(
                "keys",
                start=_pos_straight(bar, 1),
                duration=BAR * 0.95,
                partial=P4,
                amp_db=-10.0,
            )
            continue

        chord = _chord_for_bar(bar)
        for partial, db in chord:
            score.add_note(
                "keys",
                start=_pos_straight(bar, 1),
                duration=BAR * 0.92,
                partial=partial,
                amp_db=db,
            )

        # Upper octave shimmer in peaks
        if (73 <= bar <= 88 or 105 <= bar <= 120) and bar % 2 == 0:
            score.add_note(
                "keys",
                start=_pos_straight(bar, 1),
                duration=BAR * 0.85,
                partial=P8,
                amp_db=-20.0,
            )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

PIECES: dict[str, PieceDefinition] = {
    "amber_room": PieceDefinition(
        name="amber_room",
        output_name="amber_room",
        build_score=build_score,
        sections=(
            PieceSection("intro_kick", _pos_straight(1), _pos_straight(9)),
            PieceSection("intro_hats", _pos_straight(9), _pos_straight(17)),
            PieceSection("build_a", _pos_straight(17), _pos_straight(25)),
            PieceSection("groove_enter", _pos_straight(25), _pos_straight(33)),
            PieceSection("groove_a", _pos_straight(33), _pos_straight(49)),
            PieceSection("breakdown_1", _pos_straight(49), _pos_straight(57)),
            PieceSection("build_b_melody", _pos_straight(57), _pos_straight(65)),
            PieceSection("groove_b", _pos_straight(65), _pos_straight(73)),
            PieceSection("peak_1", _pos_straight(73), _pos_straight(89)),
            PieceSection("breakdown_2", _pos_straight(89), _pos_straight(97)),
            PieceSection("peak_2", _pos_straight(97), _pos_straight(121)),
            PieceSection("settle", _pos_straight(121), _pos_straight(129)),
            PieceSection("outro", _pos_straight(129), _pos_straight(145)),
            PieceSection("outro_end", _pos_straight(145), _pos_straight(151)),
        ),
    ),
}
