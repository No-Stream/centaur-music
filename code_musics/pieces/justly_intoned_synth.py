"""WTC-inspired song: 5-limit broken-chord study with 7-limit barbershop interlude.

Structure
---------
Exposition   — 2× 8-bar 5-limit A-major arpeggio
                 Pass 2 substitutes a 7-limit D in the V7 bar (one note, ~26¢ flat)
                 as a silent foreshadow of the barbershop section.
Pedal        — 4 bars with E bass frozen; closes on a 7-limit V7 hinge
Barbershop   — arpeggio stops; six 4:5:6:7 just-intonation chords bloom slowly.
               Stable layer (bass + triad) enters first; the 7th-harmonic colour
               voice drifts in ~1.5 s later so the listener hears the pure triad
               lock in before the alien note appears.
               Progression: I7 → IV7 → V7 → IV7 → V7 → I (pure, no 7th)
Recap        — arpeggio returns with cantus firmus (one long tone per bar)
Coda         — final held I chord
"""

from __future__ import annotations

import logging

from code_musics.composition import RhythmCell, line
from code_musics.pieces._shared import SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import Phrase, Score

logger = logging.getLogger(__name__)

# ─── arpeggio machinery ───────────────────────────────────────────────────────

_NOTE_DUR: float = 0.16
_N_PER_BAR: int = 8
_BAR_DUR: float = _NOTE_DUR * _N_PER_BAR   # 1.28 s

_ARP4: list[int] = [0, 1, 2, 3, 1, 2, 3, 2]


def _arp(freqs: list[float], amp_db: float = -10.0) -> Phrase:
    pattern = _ARP4 if len(freqs) == 4 else [0, 1, 2, 1, 0, 1, 2, 1]
    tones = [freqs[i] for i in pattern]
    return line(
        tones=tones,
        rhythm=RhythmCell(spans=tuple([_NOTE_DUR] * _N_PER_BAR)),
        pitch_kind="freq",
        amp_db=amp_db,
    )


def _place_bars(
    score: Score, voice: str, bars: list[list[float]], start: float, amp_db: float = -10.0
) -> float:
    t = start
    for b in bars:
        score.add_phrase(voice, _arp(b, amp_db=amp_db), start=t)
        t += _BAR_DUR
    return t


# ─── chord voicings ───────────────────────────────────────────────────────────

def _expo_bars(t: float) -> list[list[float]]:
    return [
        [t / 2,    t,        t * 5/4,   t * 3/2  ],   # I
        [t * 2/3,  t * 4/3,  t * 5/3,   t * 2    ],   # IV
        [t * 3/4,  t * 3/2,  t * 15/8,  t * 9/4  ],   # V
        [t / 2,    t,        t * 5/4,   t * 3/2  ],   # I
        [t * 5/6,  t * 5/3,  t * 2,     t * 5/2  ],   # vi
        [t * 9/16, t * 9/8,  t * 4/3,   t * 5/3  ],   # ii
        [t * 3/4,  t * 3/2,  t * 15/8,  t * 8/3  ],   # V7 (5-lim D)
        [t / 2,    t,        t * 5/4,   t * 3/2  ],   # I
    ]


def _expo_bars_pass2(t: float) -> list[list[float]]:
    """Same as pass 1 except bar 7 (V7) uses the 7-limit D (~26¢ flat).
    One note changed; the listener may not consciously notice, but the ear
    registers the slight difference — a seed for the barbershop section.
    """
    bars = _expo_bars(t)
    bars[6] = [t * 3/4, t * 3/2, t * 15/8, t * 21/8]   # 21/8 = 7-lim D
    return bars


def _pedal_bars(t: float) -> list[list[float]]:
    e = t * 3/4   # E2 = 165 Hz
    return [
        [e,  t * 3/2,  t * 15/8,  t * 9/4  ],   # V on pedal
        [e,  t,        t * 5/4,   t * 3/2  ],   # I6 (A chord over E bass)
        [e,  t * 3/2,  t * 15/8,  t * 9/4  ],   # V on pedal
        [e,  t * 3/2,  t * 15/8,  t * 21/8 ],   # V7 7-lim hinge → barbershop
    ]


# ─── barbershop chord helper ──────────────────────────────────────────────────

def _add_chorale_chord(
    score: Score,
    voice: str,
    t_start: float,
    stable_freqs: list[float],
    color_freq: float | None,
    chord_dur: float,
    stable_amp_db: float = -9.0,
    color_amp_db: float = -11.5,
    color_delay: float = 1.5,
) -> float:
    """Place a barbershop chord.

    Stable voices (bass + triad) enter at t_start.  The 7th-harmonic colour
    voice enters color_delay seconds later so the pure triad has time to ring
    before the alien note appears.  Returns the next chord start time.
    """
    for freq in stable_freqs:
        score.add_note(
            voice, start=t_start, duration=chord_dur,
            freq=freq, amp_db=stable_amp_db,
        )
    if color_freq is not None:
        color_dur = max(chord_dur - color_delay + 0.5, 1.0)
        score.add_note(
            voice, start=t_start + color_delay, duration=color_dur,
            freq=color_freq, amp_db=color_amp_db,
        )
    return t_start + chord_dur


# ─── build score ──────────────────────────────────────────────────────────────

def build_wtc_song_score() -> Score:
    tonic = 220.0   # A3

    score = Score(f0=tonic, master_effects=[SOFT_REVERB_EFFECT])

    score.add_voice(
        "arp",
        synth_defaults={
            "engine": "additive",
            "params": {"n_harmonics": 7, "harmonic_rolloff": 0.44},
            "env": {
                "attack_ms": 10.0,
                "decay_ms": 240.0,
                "sustain_ratio": 0.35,
                "release_ms": 380.0,
            },
        },
        mix_db=0.0,
        velocity_humanize=None,
    )

    # Chorale — slow bloom, long release so adjacent chords bleed into each other
    score.add_voice(
        "chorale",
        synth_defaults={
            "engine": "additive",
            "params": {"n_harmonics": 8, "harmonic_rolloff": 0.42},
            "env": {
                "attack_ms": 1500.0,
                "decay_ms": 700.0,
                "sustain_ratio": 0.84,
                "release_ms": 2400.0,
            },
        },
        mix_db=-1.0,
        velocity_humanize=None,
    )

    # Cantus — one long tone per bar during recap; remove by setting mix_db=-99
    score.add_voice(
        "cantus",
        synth_defaults={
            "engine": "additive",
            "params": {"n_harmonics": 5, "harmonic_rolloff": 0.50},
            "env": {
                "attack_ms": 180.0,
                "decay_ms": 550.0,
                "sustain_ratio": 0.52,
                "release_ms": 700.0,
            },
        },
        mix_db=-4.0,
        velocity_humanize=None,
    )

    # ── SECTION 1: Exposition ────────────────────────────────────────────────
    t = 0.0
    t = _place_bars(score, "arp", _expo_bars(tonic), start=t, amp_db=-9.0)
    t = _place_bars(score, "arp", _expo_bars_pass2(tonic), start=t, amp_db=-8.0)

    # ── SECTION 2: Dominant pedal ────────────────────────────────────────────
    t = _place_bars(score, "arp", _pedal_bars(tonic), start=t, amp_db=-8.5)

    # ── SECTION 3: Barbershop ────────────────────────────────────────────────
    #
    # All chords use 4:5:6:7 voicing (just major triad + septimal minor 7th).
    # Each stable layer enters first; the 7th-harmonic colour note follows ~1.5 s
    # later so you hear the triad lock in before the alien note appears.
    #
    # Progression  root      stable [bass, root, 3rd, 5th]        colour (7th harm)
    # ─────────────────────────────────────────────────────────────────────────
    # I7           A=55      [110, 220, 275, 330]                  385  (blue G)
    # IV7          D=73.33   [147, 293, 367, 440]                  513  (flat C)
    # V7           E=82.5    [165, 330, 412, 495]                  578  (flat D)
    # IV7          D         [147, 293, 367, 440]                  513
    # V7           E         [165, 330, 412, 495]                  578
    # I (pure)     A         [110, 220, 275, 330, 440]             —   (clean close)

    bs = t
    bs = _add_chorale_chord(
        score, "chorale", bs,
        stable_freqs=[tonic / 2, tonic, tonic * 5/4, tonic * 3/2],
        color_freq=tonic * 7/4,
        chord_dur=6.0,
    )
    bs = _add_chorale_chord(
        score, "chorale", bs,
        stable_freqs=[tonic * 2/3, tonic * 4/3, tonic * 5/3, tonic * 2],
        color_freq=tonic * 7/3,
        chord_dur=5.5,
    )
    bs = _add_chorale_chord(
        score, "chorale", bs,
        stable_freqs=[tonic * 3/4, tonic * 3/2, tonic * 15/8, tonic * 9/4],
        color_freq=tonic * 21/8,
        chord_dur=5.5,
    )
    bs = _add_chorale_chord(
        score, "chorale", bs,
        stable_freqs=[tonic * 2/3, tonic * 4/3, tonic * 5/3, tonic * 2],
        color_freq=tonic * 7/3,
        chord_dur=4.5,
        color_delay=1.2,   # slightly shorter colour delay on the return
    )
    bs = _add_chorale_chord(
        score, "chorale", bs,
        stable_freqs=[tonic * 3/4, tonic * 3/2, tonic * 15/8, tonic * 9/4],
        color_freq=tonic * 21/8,
        chord_dur=5.5,
    )
    # Final I: pure 5-limit, no 7th — the world snaps clean after the alien notes
    i_res_dur = 7.0
    _add_chorale_chord(
        score, "chorale", bs,
        stable_freqs=[tonic / 2, tonic, tonic * 5/4, tonic * 3/2, tonic * 2],
        color_freq=None,
        chord_dur=i_res_dur,
        stable_amp_db=-8.5,
    )

    # Arp re-enters 1.5 s before the I resolution chord ends
    t = bs + i_res_dur - 1.5

    # ── SECTION 4: Recap + cantus ────────────────────────────────────────────
    recap_start = t
    t = _place_bars(score, "arp", _expo_bars(tonic), start=recap_start, amp_db=-9.5)

    # Cantus: one long tone per bar, stepwise in the JI scale.
    # The jump to A4 on bar 5 is the moment of lift; the descent resolves.
    cantus_freqs = [
        tonic * 3/2,    # bar 1  I    → E4
        tonic * 4/3,    # bar 2  IV   → D4 (step down)
        tonic * 3/2,    # bar 3  V    → E4
        tonic * 3/2,    # bar 4  I    → E4 (stable)
        tonic * 2,      # bar 5  vi   → A4 (lift)
        tonic * 5/3,    # bar 6  ii   → F#4 (falling)
        tonic * 4/3,    # bar 7  V7   → D4
        tonic,          # bar 8  I    → A3 (resolution)
    ]
    for i, freq in enumerate(cantus_freqs):
        score.add_note(
            "cantus",
            start=recap_start + i * _BAR_DUR,
            duration=_BAR_DUR * 1.06,
            freq=freq,
            amp_db=-13.5,
        )

    # ── SECTION 5: Coda ──────────────────────────────────────────────────────
    for ratio in [0.5, 1.0, 5/4, 3/2, 2.0]:
        score.add_note("arp", start=t, duration=6.5, freq=tonic * ratio, amp_db=-11.0)

    return score


PIECES: dict[str, PieceDefinition] = {
    "justly_intoned_synth": PieceDefinition(
        name="justly_intoned_synth",
        output_name="28_justly_intoned_synth.wav",
        build_score=build_wtc_song_score,
    ),
}
