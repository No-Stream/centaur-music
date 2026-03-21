"""WTC-inspired JI arpeggio sketches.

Four pieces riffing on the Well-Tempered Clavier concept:
  wtc_ji_5limit  – broken-chord study in 5-limit JI  (Sketch A)
  wtc_ji_7limit  – same skeleton with septimal colour  (Sketch A)
  wtc_comma_pump – I-IV-ii-V loop that drifts +syntonic comma each cycle  (Sketch B)
  wtc_harmonic   – arp that climbs through harmonic-series windows  (Sketch C)
"""

from __future__ import annotations

import logging

from code_musics.composition import RhythmCell, line
from code_musics.pieces._shared import SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Phrase, Score

logger = logging.getLogger(__name__)

# ─── shared arpeggio machinery ────────────────────────────────────────────────

_NOTE_DUR: float = 0.16  # seconds per 16th-note step
_N_PER_BAR: int = 8
_BAR_DUR: float = _NOTE_DUR * _N_PER_BAR  # 1.28 s

# Bach-like pattern for 4-tone voicings [bass, low, mid, high]:
#   bass | low mid high low mid high mid
_ARP4: list[int] = [0, 1, 2, 3, 1, 2, 3, 2]
# Gentle rocking for 3-tone voicings [a, b, c]:
#   a b c b a b c b
_ARP3: list[int] = [0, 1, 2, 1, 0, 1, 2, 1]


def _arp(freqs: list[float], amp_db: float = -10.0) -> Phrase:
    """One bar of WTC-style arpeggio from a list of absolute frequencies."""
    pattern = _ARP4 if len(freqs) >= 4 else _ARP3
    tones = [freqs[i] for i in pattern]
    return line(
        tones=tones,
        rhythm=RhythmCell(spans=tuple([_NOTE_DUR] * _N_PER_BAR)),
        pitch_kind="freq",
        amp_db=amp_db,
    )


def _place_bars(
    score: Score,
    voice: str,
    bars: list[list[float]],
    start: float,
    amp_db: float = -10.0,
) -> float:
    """Place a sequence of arp bars; return end time."""
    t = start
    for bar_freqs in bars:
        score.add_phrase(voice, _arp(bar_freqs, amp_db=amp_db), start=t)
        t += _BAR_DUR
    return t


def _keyboard_synth(
    n_harmonics: int = 7,
    rolloff: float = 0.44,
    attack_ms: float = 10.0,
    decay_ms: float = 240.0,
    sustain_ratio: float = 0.35,
    release_ms: float = 380.0,
) -> dict:  # type: ignore[type-arg]
    """Additive synth tuned to feel like a clean keyboard/harpsichord."""
    return {
        "engine": "additive",
        "params": {
            "n_harmonics": n_harmonics,
            "harmonic_rolloff": rolloff,
        },
        "env": {
            "attack_ms": attack_ms,
            "decay_ms": decay_ms,
            "sustain_ratio": sustain_ratio,
            "release_ms": release_ms,
        },
    }


# ─── Sketch A: chord voicings ─────────────────────────────────────────────────
#
# Tonic = 220 Hz (A3).  Voicings span roughly A2–C#5 so the arp always starts
# on the bass note and rises through two-ish octaves — same register logic as
# the original BWV 846 C major prelude.


def _5limit_bars(t: float) -> list[list[float]]:
    """8-bar A-major-like progression in 5-limit JI from tonic *t*."""
    return [
        [t / 2, t, t * 5 / 4, t * 3 / 2],  # I   (A maj)
        [t * 2 / 3, t * 4 / 3, t * 5 / 3, t * 2],  # IV  (D maj)
        [t * 3 / 4, t * 3 / 2, t * 15 / 8, t * 9 / 4],  # V   (E maj)
        [t / 2, t, t * 5 / 4, t * 3 / 2],  # I
        [t * 5 / 6, t * 5 / 3, t * 2, t * 5 / 2],  # vi  (F# min)
        [t * 9 / 16, t * 9 / 8, t * 4 / 3, t * 5 / 3],  # ii  (B min)
        [t * 3 / 4, t * 3 / 2, t * 15 / 8, t * 8 / 3],  # V7  (E7, 5-lim D)
        [t / 2, t, t * 5 / 4, t * 3 / 2],  # I
    ]


def _7limit_bars(t: float) -> list[list[float]]:
    """Same skeleton with two septimal substitutions.

    Bar 5: tonic + septimal 7th (A-C#-G, 4:5:7 voicing) instead of F# minor.
    Bar 7: V7 with 7/4-based dominant 7th (21/8 ≈ 577 Hz) instead of 8/3 ≈ 587 Hz.
    """
    return [
        [t / 2, t, t * 5 / 4, t * 3 / 2],  # I
        [t * 2 / 3, t * 4 / 3, t * 5 / 3, t * 2],  # IV  (pure 5-limit)
        [t * 3 / 4, t * 3 / 2, t * 15 / 8, t * 9 / 4],  # V
        [t / 2, t, t * 5 / 4, t * 3 / 2],  # I
        # Septimal substitution: A dominant 7th using 7/4 (A-C#-G, no fifth)
        [t / 2, t, t * 5 / 4, t * 7 / 4],  # I+7 (A dom7 sept)
        [t * 9 / 16, t * 9 / 8, t * 4 / 3, t * 5 / 3],  # ii
        # 7-limit dominant 7th: 21/8 = (3/2)*(7/4), ~26¢ flatter than 8/3
        [t * 3 / 4, t * 3 / 2, t * 15 / 8, t * 21 / 8],  # V7  (7-lim D)
        [t / 2, t, t * 5 / 4, t * 3 / 2],  # I
    ]


# ─── Sketch A: 5-limit ────────────────────────────────────────────────────────


def build_wtc_ji_5limit_score() -> Score:
    """WTC-style broken-chord study in 5-limit just intonation (A major)."""
    tonic = 220.0
    score = Score(f0=tonic, master_effects=[SOFT_REVERB_EFFECT])
    score.add_voice(
        "arp",
        synth_defaults=_keyboard_synth(),
        mix_db=0.0,
        velocity_humanize=None,
    )

    bars = _5limit_bars(tonic)

    # Pass 1: plain arpeggio
    t = _place_bars(score, "arp", bars, start=0.0, amp_db=-9.0)
    # Pass 2: slightly louder — letting the harmony breathe
    t = _place_bars(score, "arp", bars, start=t, amp_db=-7.0)
    # Pass 3: first half only, fading
    t = _place_bars(score, "arp", bars[:4], start=t, amp_db=-11.5)
    # Final held I chord
    for ratio in [0.5, 1.0, 5 / 4, 3 / 2, 2.0]:
        score.add_note("arp", start=t, duration=5.0, freq=tonic * ratio, amp_db=-11.0)

    return score


# ─── Sketch A: 7-limit ────────────────────────────────────────────────────────


def build_wtc_ji_7limit_score() -> Score:
    """WTC-style broken-chord study with septimal (7-limit) colour.

    Two substitutions vs the 5-limit version:
    - Bar 5: A dom7 using the 7th harmonic (blue, resonant) instead of F# minor
    - Bar 7: 7-limit dominant 7th (the D is ~26¢ flatter, more 'in tune' but alien)
    """
    tonic = 220.0
    score = Score(f0=tonic, master_effects=[SOFT_REVERB_EFFECT])
    score.add_voice(
        "arp",
        synth_defaults=_keyboard_synth(),
        mix_db=0.0,
        velocity_humanize=None,
    )

    bars = _7limit_bars(tonic)

    t = _place_bars(score, "arp", bars, start=0.0, amp_db=-9.0)
    t = _place_bars(score, "arp", bars, start=t, amp_db=-7.0)
    t = _place_bars(score, "arp", bars[:4], start=t, amp_db=-11.5)
    # End on the open A+septimal-7th chord (no fifth — just 1, 5/4, 7/4)
    for ratio in [0.5, 1.0, 5 / 4, 7 / 4]:
        score.add_note("arp", start=t, duration=5.0, freq=tonic * ratio, amp_db=-11.0)

    return score


# ─── Sketch B: syntonic comma pump ────────────────────────────────────────────

_COMMA: float = 81 / 80  # +21.5 ¢ per cycle


def _comma_cycle_bars(t: float) -> list[list[float]]:
    """4-bar I-IV-ii-V voicings for one cycle from tonic *t*."""
    return [
        [t / 2, t, t * 5 / 4, t * 3 / 2],  # I
        [t * 2 / 3, t * 4 / 3, t * 5 / 3, t * 2],  # IV
        [t * 9 / 16, t * 9 / 8, t * 4 / 3, t * 5 / 3],  # ii
        [t * 3 / 4, t * 3 / 2, t * 15 / 8, t * 9 / 4],  # V
    ]


def build_wtc_comma_pump_score() -> Score:
    """Arpeggio loop that drifts upward +21.5¢ (syntonic comma) each cycle.

    6 cycles total; the tonic creeps ~129¢ sharp (just over a semitone).
    The final chord snaps back to the original A=220 Hz, making the drift
    audible in retrospect.
    """
    tonic = 220.0
    n_cycles = 6
    score = Score(
        f0=tonic,
        master_effects=[
            EffectSpec(
                "reverb", {"room_size": 0.62, "damping": 0.42, "wet_level": 0.22}
            ),
        ],
    )
    score.add_voice(
        "arp",
        synth_defaults=_keyboard_synth(n_harmonics=6, rolloff=0.42),
        mix_db=0.0,
        velocity_humanize=None,
    )

    t = 0.0
    for cycle_idx in range(n_cycles):
        cycle_tonic = tonic * (_COMMA**cycle_idx)
        logger.info(
            "comma pump cycle %d  tonic=%.2f Hz  (+%.1f ¢)",
            cycle_idx,
            cycle_tonic,
            1200 * (cycle_idx * (81 / 80 - 1) * 80),  # approx
        )
        bars = _comma_cycle_bars(cycle_tonic)
        # Tiny amplitude swell to make the drift more perceptible
        amp_db = -10.0 + cycle_idx * 0.4
        t = _place_bars(score, "arp", bars, start=t, amp_db=amp_db)

    # Snap back: land on original A major
    for ratio in [0.5, 1.0, 5 / 4, 3 / 2, 2.0]:
        score.add_note("arp", start=t, duration=5.5, freq=tonic * ratio, amp_db=-10.0)

    return score


# ─── Sketch C: harmonic series window ─────────────────────────────────────────


def build_wtc_harmonic_score() -> Score:
    """Broken-chord arp that climbs then descends through harmonic-series windows.

    Base = A1 = 55 Hz.  Each 'bar' arpegiates 4 consecutive partials.
    The window [4,5,6,7] = A3 C#4 E4 G4(7th-harm) — a just dominant 7th chord.
    The climb ascends to [9,10,11,12] = B4 C#5 ~D#5(11th) E5, then reverses.
    A slow drone at A2 = 110 Hz grounds everything.
    """
    base = 55.0  # A1; all arp notes are integer multiples of this
    bars_per_window = 4

    # Windows: (lowest_partial, n_bars_at_this_window)
    window_schedule: list[tuple[int, int]] = [
        (4, bars_per_window),
        (5, bars_per_window),
        (6, bars_per_window),
        (7, bars_per_window),
        (8, bars_per_window),
        (9, bars_per_window + 2),  # linger at the peak
        (8, bars_per_window),
        (7, bars_per_window),
        (6, bars_per_window),
        (5, bars_per_window),
        (4, bars_per_window),
    ]

    total_bars = sum(n for _, n in window_schedule)
    total_dur = total_bars * _BAR_DUR + 6.0  # + 6 s for final chord

    score = Score(
        f0=base,
        master_effects=[
            EffectSpec(
                "reverb", {"room_size": 0.70, "damping": 0.48, "wet_level": 0.28}
            ),
        ],
    )
    score.add_voice(
        "arp",
        synth_defaults=_keyboard_synth(n_harmonics=6, rolloff=0.40),
        mix_db=0.0,
        velocity_humanize=None,
    )
    score.add_voice(
        "drone",
        synth_defaults={
            "engine": "additive",
            "params": {"n_harmonics": 4, "harmonic_rolloff": 0.60},
            "env": {
                "attack_ms": 1800.0,
                "decay_ms": 300.0,
                "sustain_ratio": 0.88,
                "release_ms": 2500.0,
            },
        },
        mix_db=-10.0,
        velocity_humanize=None,
    )

    # Drone at A2 = 2 × base = 110 Hz, runs the whole piece
    score.add_note("drone", start=0.0, duration=total_dur, partial=2.0, amp_db=-12.0)

    t = 0.0
    for lowest_partial, n_bars in window_schedule:
        window_freqs = [base * (lowest_partial + i) for i in range(4)]
        logger.info(
            "harmonic window %d-%d  freqs=[%.1f, %.1f, %.1f, %.1f] Hz",
            lowest_partial,
            lowest_partial + 3,
            *window_freqs,
        )
        for _ in range(n_bars):
            score.add_phrase("arp", _arp(window_freqs, amp_db=-10.0), start=t)
            t += _BAR_DUR

    # Final chord: partials 4:5:6 (A C# E — pure major triad)
    for partial in [4, 5, 6]:
        score.add_note("arp", start=t, duration=6.0, freq=base * partial, amp_db=-12.0)

    return score


# ─── registration ─────────────────────────────────────────────────────────────

PIECES: dict[str, PieceDefinition] = {
    "wtc_ji_5limit": PieceDefinition(
        name="wtc_ji_5limit",
        output_name="24_wtc_ji_5limit.wav",
        build_score=build_wtc_ji_5limit_score,
    ),
    "wtc_ji_7limit": PieceDefinition(
        name="wtc_ji_7limit",
        output_name="25_wtc_ji_7limit.wav",
        build_score=build_wtc_ji_7limit_score,
    ),
    "wtc_comma_pump": PieceDefinition(
        name="wtc_comma_pump",
        output_name="26_wtc_comma_pump.wav",
        build_score=build_wtc_comma_pump_score,
    ),
    "wtc_harmonic": PieceDefinition(
        name="wtc_harmonic",
        output_name="27_wtc_harmonic.wav",
        build_score=build_wtc_harmonic_score,
    ),
}
