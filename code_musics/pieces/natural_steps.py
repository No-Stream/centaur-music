"""Harmonic-series arpeggio: stepwise progression through JI scale degrees.

Each chord is a 4:5:6:7 just-intonation voicing (pure major triad + septimal
minor 7th) built on a different scale-degree root.  The root ascends stepwise
through A→B→C#→D→E then descends back D→A, tracing a rising-and-falling arc.

Because every chord uses the same internal structure (4:5:6:7), each one rings
with its own harmonic-series resonance.  The alien quality comes from the
unusual relationships between chords — particularly the B and C# centres which
produce notes outside familiar JI territory — rather than from any single
chord sounding harsh.

The 7th partial of each root is always the top voice:
  A   → G4♭  (385 Hz,  the familiar blue G)
  B   → A4♭  (433 Hz,  ~27 ¢ flat of A4)
  C#  → B♭4? (481 Hz,  between A4 and B4 — most alien)
  D   → C5♭  (513 Hz,  septimal C)
  E   → D5♭  (578 Hz,  septimal D)

Bar counts vary (4-3-2-4-4-2-4 = 23 bars) so the ascent accelerates into the
peak and the descent broadens on the way home.  Amplitude swells gently on the
way up and eases on the way down.

A drone at A2 = 110 Hz grounds the whole piece.
"""

from __future__ import annotations

import logging

from code_musics.composition import RhythmCell, line
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Phrase, Score

logger = logging.getLogger(__name__)

_NOTE_DUR: float = 0.16
_N_PER_BAR: int = 8
_BAR_DUR: float = _NOTE_DUR * _N_PER_BAR

_ARP4: list[int] = [0, 1, 2, 3, 1, 2, 3, 2]


def _arp(freqs: list[float], amp_db: float = -10.0) -> Phrase:
    tones = [freqs[i] for i in _ARP4]
    return line(
        tones=tones,
        rhythm=RhythmCell(spans=tuple([_NOTE_DUR] * _N_PER_BAR)),
        pitch_kind="freq",
        amp_db=amp_db,
    )


def build_wtc_harmonic_dev_score() -> Score:
    """Stepwise 4:5:6:7 arpeggio through A→B→C#→D→E→D→A."""
    base = 55.0   # A1 — all partials are integer multiples of this

    score = Score(
        f0=base,
        master_effects=[
            EffectSpec("reverb", {"room_size": 0.68, "damping": 0.46, "wet_level": 0.26}),
        ],
    )

    score.add_voice(
        "arp",
        synth_defaults={
            "engine": "additive",
            "params": {"n_harmonics": 6, "harmonic_rolloff": 0.40},
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

    score.add_voice(
        "drone",
        synth_defaults={
            "engine": "additive",
            "params": {"n_harmonics": 4, "harmonic_rolloff": 0.60},
            "env": {
                "attack_ms": 2200.0,
                "decay_ms": 300.0,
                "sustain_ratio": 0.88,
                "release_ms": 3000.0,
            },
        },
        mix_db=-9.0,
        velocity_humanize=None,
    )

    # ── Progression ──────────────────────────────────────────────────────────
    #
    # root = base frequency of the 4:5:6:7 chord (= chord's "1st harmonic")
    # chord = [4r, 5r, 6r, 7r]
    #
    # JI scale roots (relative to A1=55 Hz):
    #   A  = 55      B  = 9/8 × 55 = 61.875
    #   C# = 5/4 × 55 = 68.75   D  = 4/3 × 55 = 73.33
    #   E  = 3/2 × 55 = 82.5

    progression: list[tuple[float, int, str]] = [
        # (chord root,  n_bars,  label)
        (base,              4, "A  (I7)"),
        (base * 9/8,        3, "B  (II7)"),
        (base * 5/4,        2, "C# (III7)"),
        (base * 4/3,        4, "D  (IV7)"),
        (base * 3/2,        4, "E  (V7)"),
        (base * 4/3,        2, "D  (IV7) ↓"),
        (base,              4, "A  (I7) home"),
    ]

    # Amplitude arc: swell up toward the peak (E), ease back down
    amp_arc: list[float] = [-11.0, -10.0, -9.0, -9.5, -9.0, -10.0, -11.0]

    total_bars = sum(n for _, n, _ in progression)
    total_dur = 2.0 + total_bars * _BAR_DUR + 5.0   # drone intro + arp + final chord tail

    # Drone at A2 = 110 Hz (partial 2 of base=55) throughout
    score.add_note("drone", start=0.0, duration=total_dur, partial=2.0, amp_db=-12.0)

    t = 2.0   # brief drone intro before arp enters

    for (root, n_bars, label), amp_db in zip(progression, amp_arc):
        chord = [root * 4, root * 5, root * 6, root * 7]
        logger.info(
            "%-18s  root=%.2f Hz  freqs=[%.1f, %.1f, %.1f, %.1f(7th)]",
            label, root, *chord,
        )
        for _ in range(n_bars):
            score.add_phrase("arp", _arp(chord, amp_db=amp_db), start=t)
            t += _BAR_DUR

    # Final held chord: land on the pure A major triad (partials 4:5:6)
    for partial in [4, 5, 6]:
        score.add_note("arp", start=t, duration=6.0, freq=base * partial, amp_db=-12.0)

    return score


PIECES: dict[str, PieceDefinition] = {
    "natural_steps": PieceDefinition(
        name="natural_steps",
        output_name="29_natural_steps.wav",
        build_score=build_wtc_harmonic_dev_score,
    ),
}
