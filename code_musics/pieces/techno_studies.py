"""Techno studies — harmonic-series material at 130 BPM.

Tuning: partial series of f0 = 55 Hz (A1).
Selected harmonic materials:
  partial  1   =  55.0 Hz  (A1)        bass root (sub)
  partial  1.5 =  82.5 Hz  (E2)        bass fifth
  partial  2   = 110.0 Hz  (A2)        bass octave
  partial  3.5 = 192.5 Hz              septimal 7th colour (7/4 × A2)
  partial  7   = 385.0 Hz  (G4 – 31¢) dark septimal 7th — lead shadow tone
  partial  8   = 440.0 Hz  (A4)        lead anchor
  partial  9   = 495.0 Hz  (B4 + 4¢)  near-whole-step above anchor
  partial 11   = 605.0 Hz              undecimal super-fourth (xenharmonic hook)

BPM = 130.  1 bar ≈ 1.846 s.  1 beat ≈ 0.4615 s.  1 sixteenth ≈ 0.1154 s.

Piece: spectral_kick
Structure:
  bars  1– 2   intro: kick alone
  bars  3– 4   kick + bass
  bars  5–12   kick + bass + lead phrase A (4 × 2-bar)
  bars 13–16   lead phrase B (truncated — sparse bar 2)
  bars 17–20   drop: kick + bass only
  bars 21–28   lead phrase A returns (4 × 2-bar)
  bars 29–32   lead phrase C: 3-note core gesture, one bar each
  bars 33–36   outro: kick + bass only
"""

from __future__ import annotations

from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Score

BPM: float = 130.0
BEAT: float = 60.0 / BPM  # quarter-note duration ≈ 0.4615 s
BAR: float = 4.0 * BEAT  # 4/4 bar ≈ 1.846 s
S16: float = BEAT / 4.0  # sixteenth-note ≈ 0.1154 s

F0: float = 55.0  # A1 — score root

# Harmonic partial constants (freq = F0 * partial)
P1: float = 1.0  # A1    55.0 Hz  bass root (sub)
P15: float = 1.5  # E2    82.5 Hz  bass fifth
P2: float = 2.0  # A2   110.0 Hz  bass octave
P35: float = 3.5  # —    192.5 Hz  septimal 7th colour
P7: float = 7.0  # G4   385.0 Hz  dark septimal 7th
P8: float = 8.0  # A4   440.0 Hz  lead anchor
P9: float = 9.0  # B4   495.0 Hz  near whole step
P11: float = 11.0  # —    605.0 Hz  undecimal super-fourth (the hook)

# note type: (bar_offset, beat, n16, partial, gate_sixteenths, amp_db)
_LeadNote = tuple[int, int, int, float, float, float]


def _pos(bar: int, beat: int = 1, n16: int = 0) -> float:
    """Absolute time in seconds at bar:beat:sixteenth (bar and beat are 1-indexed)."""
    return (bar - 1) * BAR + (beat - 1) * BEAT + n16 * S16


def build_spectral_kick() -> Score:
    """Harmonic-series techno sketch at 130 BPM.

    The 11th partial (605 Hz, undecimal super-fourth) is the xenharmonic signature —
    neither fourth nor tritone, just wrong in a compelling way. The 7th partial
    (385 Hz) gives the dark, heavy septimal colour. Together they outline a vocabulary
    that is clearly not 12-TET while remaining consonant within the harmonic series.

    Lead voice: short FM bell tones, Detroit-techno style.
    Bass: polyblep acid with JI-tuned pitches.
    Kick: 909-style, 4-on-the-floor.
    """
    score = Score(
        f0=F0,
        master_effects=[
            EffectSpec(
                "bricasti",
                {
                    "ir_name": "1 Halls 07 Large & Dark",
                    "wet": 0.16,
                    "lowpass_hz": 7000.0,
                },
            ),
            EffectSpec(
                "compressor",
                {
                    # Kick-driven pumping glue: mix peaks ~-10 dBFS post-reverb.
                    # Threshold at -16 dBFS → kick body drives 3-5 dB GR;
                    # 180 ms release recovers well before the next beat at 130 BPM.
                    # No sidechain HP — the kick is what we want to drive the pump.
                    "threshold_db": -16.0,
                    "ratio": 2.5,
                    "attack_ms": 10.0,
                    "release_ms": 180.0,
                    "knee_db": 5.0,
                    "makeup_gain_db": 1.5,
                    "detector_mode": "peak",
                },
            ),
        ],
    )

    total_bars = 36

    # ------------------------------------------------------------------
    # Kick: 909-style, 4-on-the-floor for the full piece
    # ------------------------------------------------------------------
    score.add_voice(
        "kick",
        synth_defaults={"engine": "kick_tom", "preset": "909_techno"},
        effects=[EffectSpec("compressor", {"preset": "kick_punch"})],
        normalize_peak_db=-6.0,
        mix_db=-2.0,
        velocity_humanize=None,
    )

    for bar in range(1, total_bars + 1):
        for beat in range(1, 5):
            score.add_note(
                "kick",
                start=_pos(bar, beat),
                duration=1.0,
                freq=62.0,
                amp_db=-6.0,
            )

    # ------------------------------------------------------------------
    # Bass: polyblep acid, JI-tuned partials
    # ------------------------------------------------------------------
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "polyblep",
            "preset": "acid_bass",
            "env": {
                "attack_ms": 3.0,
                "decay_ms": 130.0,
                "sustain_ratio": 0.50,
                "release_ms": 55.0,
            },
            "params": {
                "waveform": "triangle",
                "cutoff_hz": 500.0,
                "resonance_ratio": 0.22,
                "filter_env_depth_ratio": 0.55,
                "filter_env_decay_ms": 140.0,
                "filter_drive": 0.70,
            },
        },
        mix_db=-5.0,
        velocity_humanize=None,
    )

    # Pattern: (beat, n16, partial, gate_16ths, amp_db)
    # One bar long, ~82% gate ratio for staccato feel
    _bass_pattern: list[tuple[int, int, float, int, float]] = [
        (1, 0, P1, 3, -6.0),  # beat 1: root (A1 sub), dotted 8th
        (1, 3, P1, 1, -10.0),  # a-of-1: root ghost, 16th
        (2, 0, P15, 2, -7.0),  # beat 2: fifth (E2), 8th
        (2, 2, P1, 1, -9.0),  # +-of-2: root, 16th
        (3, 0, P1, 3, -6.0),  # beat 3: root, dotted 8th
        (3, 3, P35, 1, -8.0),  # a-of-3: septimal 7th colour, 16th
        (4, 0, P1, 2, -7.0),  # beat 4: root, 8th
        (4, 2, P15, 2, -9.0),  # &-of-4: fifth exit, 8th
    ]

    for bar in range(3, total_bars + 1):
        for beat, n16, partial, gate_16ths, amp_db in _bass_pattern:
            score.add_note(
                "bass",
                start=_pos(bar, beat, n16),
                duration=gate_16ths * S16 * 0.82,
                partial=partial,
                amp_db=amp_db,
            )

    # ------------------------------------------------------------------
    # Lead: FM bell tones, harmonic-series pitches
    # Short attack, fast decay, nearly no sustain — pointillist, not legato
    # ------------------------------------------------------------------
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "fm",
            "preset": "bell",
            "params": {
                # mod at 3× carrier: sidebands land on P7 (septimal) and P4 (octave) —
                # weaves the FM timbre into the harmonic-series vocabulary of the piece
                "mod_ratio": 3.0,
                "mod_index": 3.5,
            },
            "env": {
                "attack_ms": 4.0,
                "decay_ms": 260.0,
                "sustain_ratio": 0.12,
                "release_ms": 180.0,
            },
        },
        mix_db=-8.0,
        velocity_humanize=None,
    )

    def _place_lead(bar_start: int, phrase: list[_LeadNote]) -> None:
        for bar_off, beat, n16, partial, gate_16ths, amp_db in phrase:
            score.add_note(
                "lead",
                start=_pos(bar_start + bar_off, beat, n16),
                duration=gate_16ths * S16 * 0.78,
                partial=partial,
                amp_db=amp_db,
            )

    # Phrase A: full 2-bar statement
    # The 11th partial is the recurring hook; P7 provides the dark anchor.
    PHRASE_A: list[_LeadNote] = [
        # bar 1
        (0, 1, 0, P8, 1, -9.0),  # beat 1: anchor punch
        (0, 1, 1, P11, 3, -7.5),  # e-of-1: xenharmonic hook, dotted 8th
        (0, 2, 0, P7, 4, -8.5),  # beat 2: septimal 7th, quarter
        (0, 3, 0, P8, 2, -9.5),  # beat 3: anchor, 8th
        (0, 3, 2, P9, 2, -10.5),  # &-of-3: step up, 8th
        (0, 4, 1, P11, 1, -11.0),  # e-of-4: hook echo, 16th
        # bar 2
        (1, 1, 0, P11, 2, -7.5),  # beat 1: hook assertion, 8th
        (1, 1, 2, P7, 5, -8.5),  # &-of-1: septimal, dotted quarter
        (1, 3, 0, P8, 2, -9.5),  # beat 3: anchor return
        (1, 3, 3, P9, 1, -11.5),  # a-of-3: step, 16th
        (1, 4, 0, P7, 3, -10.0),  # beat 4: septimal tail, dotted 8th
    ]

    # Phrase B: same bar 1, sparse bar 2 — starts losing its second half
    PHRASE_B: list[_LeadNote] = [
        (0, 1, 0, P8, 1, -9.0),
        (0, 1, 1, P11, 3, -7.5),
        (0, 2, 0, P7, 4, -8.5),
        (0, 3, 0, P8, 2, -9.5),
        (0, 3, 2, P9, 2, -10.5),
        # bar 2: only two notes remain
        (1, 1, 2, P11, 2, -8.0),
        (1, 4, 0, P7, 3, -10.5),
    ]

    # Phrase C: 1-bar skeleton — just the 3-note xenharmonic core
    PHRASE_C: list[_LeadNote] = [
        (0, 1, 1, P11, 3, -8.0),  # hook
        (0, 2, 0, P7, 4, -9.0),  # shadow
        (0, 3, 0, P8, 2, -10.0),  # anchor
    ]

    # bars 5–12: phrase A × 4 (every 2 bars)
    for b in [5, 7, 9, 11]:
        _place_lead(b, PHRASE_A)
    # bars 13–16: phrase B × 2
    for b in [13, 15]:
        _place_lead(b, PHRASE_B)
    # bars 17–20: drop — no lead
    # bars 21–28: phrase A × 4
    for b in [21, 23, 25, 27]:
        _place_lead(b, PHRASE_A)
    # bars 29–32: phrase C × 4, one per bar (maximum fragmentation)
    for b in [29, 30, 31, 32]:
        _place_lead(b, PHRASE_C)
    # bars 33–36: outro — no lead

    return score


PIECES: dict[str, PieceDefinition] = {
    "spectral_kick": PieceDefinition(
        name="spectral_kick",
        output_name="spectral_kick",
        build_score=build_spectral_kick,
    ),
}
