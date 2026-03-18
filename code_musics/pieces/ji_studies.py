"""Just intonation scale studies — 5-limit diatonic explorations.

These pieces treat JI as a *scale* system rather than as harmonic-series
material.  Key phenomena:

- Pure major thirds (5/4 = 386 ¢ vs 12-edo 400 ¢) and their distinctive
  quality — slightly lower than expected, very stable.
- Two sizes of whole tone: 9/8 (204 ¢) between scale degrees 1–2, 4–5,
  6–7, and 10/9 (182 ¢) between 2–3 and 5–6.  The asymmetry gives JI
  melody its characteristic uneven feel.
- The syntonic comma (81/80 ≈ 21.5 ¢): tuning each chord purely in a
  I-IV-ii-V cycle shifts the implied tonic down by one comma per loop.
"""

from __future__ import annotations

import logging
import math

from code_musics.composition import line, with_synth_ramp
from code_musics.pieces.septimal import PieceDefinition
from code_musics.score import EffectSpec, Score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Piece 1 – ji_chorale
# ---------------------------------------------------------------------------


def build_ji_chorale_score() -> Score:
    """Extended chorale in 5-limit JI — ~2:30 in six sections.

    Structure:
      Prologue   (0–12 s):   Fs2+A3 sparse drone; lead pickup at t=8.
      A section  (12–54 s):  7-bar vi–iv alternation; counter enters bar 3.
      B section  (54–75 s):  I–V–I in A major; bright but grounded.
      Development(75–99 s):  F#m7 → Bm → V; exploratory, unsettled.
      Reprise    (99–123 s): vi–I–vi–I; bittersweet into tastefully major.
      Ending     (123–150 s):wide vi → Dm7 → Amaj7; unresolved, wide-voiced.
    """
    f0 = 110.0  # A2 = 110 Hz

    # Named pitches — all 5-limit JI from f0
    A2  = f0                  # 110.00
    B2  = f0 * 9 / 8          # 123.75
    D3  = f0 * 4 / 3          # 146.67  iv root
    E3  = f0 * 3 / 2          # 165.00
    F3  = D3 * 6 / 5          # 176.00  D-minor 3rd (outside A major)
    Fs2 = f0 * 5 / 6          # 91.67   sub-bass F#2
    Fs3 = f0 * 5 / 3          # 183.33  vi root
    Gs3 = E3 * 5 / 4          # 206.25  V-chord 3rd (G#3)
    A3  = f0 * 2              # 220.00
    B3  = f0 * 9 / 4          # 247.50
    Cs4 = f0 * 5 / 2          # 275.00
    C4  = D3 * 9 / 5          # 264.00  pure min-7 above D
    D4  = D3 * 2              # 293.33
    E4  = f0 * 3              # 330.00
    F4  = F3 * 2              # 352.00
    Fs4 = Fs3 * 2             # 366.67
    Gs4 = f0 * 15 / 4         # 412.50  A-maj7
    A4  = f0 * 4              # 440.00
    B4  = f0 * 9 / 2          # 495.00
    Cs5 = f0 * 5              # 550.00
    D5  = D4 * 2              # 586.67
    A5  = A4 * 2              # 880.00  — tonic, next octave

    score = Score(
        f0=f0,
        master_effects=[
            EffectSpec("delay",   {"delay_seconds": 0.28, "feedback": 0.16, "mix": 0.10}),
            # Bricasti "Large & Dark" hall — warm, long, dark character; outputs stereo
            EffectSpec("bricasti", {"ir_name": "1 Halls 07 Large & Dark", "wet": 0.30}),
        ],
    )

    # Bass: filtered_stack square — still warm, but a little more filtered-envelope
    # motion and less long tail so the harmony can breathe.
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "filtered_stack",
            "waveform": "square",
            "n_harmonics": 8,
            "cutoff_ratio": 4.2,
            "resonance": 0.10,
            "filter_env_amount": 0.55,
            "filter_env_decay": 0.70,
            "attack": 0.22,
            "decay": 0.18,
            "sustain_level": 0.60,
            "release": 0.90,
        },
    )
    # Tenor/alto: slightly less legato mass so the bed supports rather than blankets.
    chord_defaults: dict = {
        "harmonic_rolloff": 0.48,
        "n_harmonics": 6,
        "brightness_tilt": 0.02,
        "unison_voices": 2,
        "detune_cents": 2.0,
        "attack": 0.22,
        "decay": 0.18,
        "sustain_level": 0.56,
        "release": 0.70,
    }
    score.add_voice("tenor", synth_defaults=dict(chord_defaults))
    score.add_voice("alto",  synth_defaults=dict(chord_defaults))
    # Counter: filtered_stack reed — oboe-like
    score.add_voice(
        "counter",
        synth_defaults={
            "engine": "filtered_stack",
            "preset": "reed_lead",
            "cutoff_ratio": 8.0,
            "resonance": 0.10,
            "filter_env_amount": 0.95,
            "attack": 0.03,
            "decay": 0.12,
            "sustain_level": 0.52,
            "release": 0.30,
        },
    )
    # Lead: filtered saw-like voice with gentle filter motion; clearer and less
    # overtly synthetic than the brighter FM pass.
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "filtered_stack",
            "waveform": "saw",
            "n_harmonics": 11,
            "cutoff_ratio": 7.5,
            "resonance": 0.10,
            "filter_env_amount": 0.55,
            "filter_env_decay": 0.90,
            "attack": 0.085,
            "decay": 1.25,
            "sustain_level": 0.48,
            "release": 0.32,
        },
    )

    # ── Prologue (0–12 s): sparse Fs2+A3 drone ──────────────────────────────
    score.add_note("bass",  start=0.0, duration=12.0, freq=Fs2, amp=0.20)
    score.add_note("tenor", start=2.0, duration=10.0, freq=A3,  amp=0.15)

    # ── A section (12–54 s): 7-bar vi–iv alternation ────────────────────────
    a_note_dur = 6.9
    a_chords: list[tuple[float, float, float, float]] = [
        (12.0, Fs3, A3, Cs4),  # vi  F# minor  bar 1
        (18.0, D3,  F3, A3 ),  # iv  D minor   bar 2
        (24.0, Fs3, A3, Cs4),  # vi             bar 3  (counter enters)
        (30.0, D3,  F3, A3 ),  # iv             bar 4
        (36.0, Fs3, A3, Cs4),  # vi             bar 5
        (42.0, D3,  F3, A3 ),  # iv             bar 6
        (48.0, Fs3, A3, Cs4),  # vi             bar 7
    ]
    for start, b, t, a in a_chords:
        score.add_note("bass",  start=start, duration=a_note_dur, freq=b, amp=0.25)
        score.add_note("tenor", start=start, duration=a_note_dur, freq=t, amp=0.21)
        score.add_note("alto",  start=start, duration=a_note_dur, freq=a, amp=0.18)

    # ── B section (54–75 s): I–V–I in A major ───────────────────────────────
    b_note_dur = 7.2
    b_chords: list[tuple[float, float, float, float]] = [
        (54.0, A2,  E3,  Cs4),  # I   A major
        (61.0, E3,  Gs3, B3 ),  # V   E major (pure 5-limit)
        (68.0, A2,  E3,  Cs4),  # I   A major
    ]
    for start, b, t, a in b_chords:
        score.add_note("bass",  start=start, duration=b_note_dur, freq=b, amp=0.23)
        score.add_note("tenor", start=start, duration=b_note_dur, freq=t, amp=0.20)
        score.add_note("alto",  start=start, duration=b_note_dur, freq=a, amp=0.17)

    # ── Development (75–99 s): F#m7 → Bm → V ───────────────────────────────
    dev_note_dur = 8.1
    dev_chords: list[tuple[float, float, float, float]] = [
        (75.0, Fs2, A3,  Cs4),  # F#m7 (counter plays E4 = 7th)
        (83.0, B2,  D3,  Fs3),  # Bm
        (91.0, E3,  Gs3, B3 ),  # V — leads back to A
    ]
    for start, b, t, a in dev_chords:
        score.add_note("bass",  start=start, duration=dev_note_dur, freq=b, amp=0.24)
        score.add_note("tenor", start=start, duration=dev_note_dur, freq=t, amp=0.20)
        score.add_note("alto",  start=start, duration=dev_note_dur, freq=a, amp=0.18)

    # ── Reprise (99–123 s): vi–I–vi–I ───────────────────────────────────────
    rep_note_dur = 6.8
    rep_chords: list[tuple[float, float, float, float]] = [
        (99.0,  Fs3, A3,  Cs4),  # vi
        (105.0, A2,  E3,  Cs4),  # I
        (111.0, Fs3, A3,  Cs4),  # vi
        (117.0, A2,  E3,  Cs4),  # I
    ]
    for start, b, t, a in rep_chords:
        score.add_note("bass",  start=start, duration=rep_note_dur, freq=b, amp=0.24)
        score.add_note("tenor", start=start, duration=rep_note_dur, freq=t, amp=0.20)
        score.add_note("alto",  start=start, duration=rep_note_dur, freq=a, amp=0.18)

    # ── Ending (123–150 s): wide vi → Dm7 → Amaj7 ───────────────────────────
    # Wide vi — F#2 sub-bass, A3, C#5 across 3 octaves
    score.add_note("bass",  start=123.0, duration=8.4,  freq=Fs2, amp=0.27)
    score.add_note("tenor", start=123.0, duration=8.4,  freq=A3,  amp=0.20)
    score.add_note("alto",  start=123.0, duration=8.4,  freq=Cs5, amp=0.21)

    # Dm7 — D / F / A / C4 (264 Hz — pure minor 7th, the spice)
    score.add_note("bass",  start=131.0, duration=8.4,  freq=D3, amp=0.26)
    score.add_note("tenor", start=131.0, duration=8.4,  freq=F3, amp=0.21)
    score.add_note("alto",  start=131.0, duration=8.4,  freq=A3, amp=0.18)
    score.add_note("alto",  start=131.0, duration=8.4,  freq=C4, amp=0.16)

    # Amaj7 — A2 / E3 / C#4 / G#4 (pure 15/8 above A) — unresolved
    score.add_note("bass",  start=139.0, duration=10.0, freq=A2,  amp=0.23)
    score.add_note("tenor", start=139.0, duration=10.0, freq=E3,  amp=0.19)
    score.add_note("alto",  start=139.0, duration=10.0, freq=Cs4, amp=0.17)
    score.add_note("alto",  start=139.0, duration=10.0, freq=Gs4, amp=0.17)

    # ── Counter voice (enters A section bar 3 = t=24) ────────────────────────
    # Moves in contrary motion to the lead; fills the inner voice above alto.
    def _add_counter(t_start: float, notes: list[tuple[float, float]]) -> None:
        t = t_start
        for freq, dur in notes:
            score.add_note("counter", start=t, duration=dur * 1.02, freq=freq, amp=0.22)
            t += dur

    # A-section bars 3–7 (t=24–54)
    _add_counter(24.0, [(E4, 2.5), (D4, 2.0), (Cs4, 1.5)])   # bar 3 vi — descends as lead rises
    _add_counter(30.0, [(F4, 2.0), (E4, 2.0), (D4, 2.0)])    # bar 4 iv — D-minor colour
    _add_counter(36.0, [(E4, 1.5), (Cs4, 2.0), (E4, 2.5)])   # bar 5 vi — arching
    _add_counter(42.0, [(D4, 2.0), (E4, 2.5), (F4, 1.5)])    # bar 6 iv — rising as lead falls
    _add_counter(48.0, [(Cs4, 2.0), (E4, 2.5), (D4, 1.5)])   # bar 7 vi

    # Development (t=75–99)
    _add_counter(75.0, [(E4, 3.0), (Cs4, 2.5), (E4, 2.5)])   # F#m7 — E4 = the 7th
    _add_counter(83.0, [(Fs4, 3.0), (E4, 2.5), (D4, 2.5)])   # Bm
    _add_counter(91.0, [(Gs4, 3.0), (Fs4, 2.0), (E4, 3.0)])  # V

    # Reprise vi chords only (I chords rest — open space)
    _add_counter(99.0,  [(E4, 3.0), (Cs4, 3.0)])              # vi
    _add_counter(111.0, [(Cs4, 2.5), (E4, 3.5)])              # vi

    # Ending
    _add_counter(123.0, [(E4, 4.0), (Cs4, 4.0)])              # wide vi
    _add_counter(131.0, [(F4, 4.0), (E4, 4.0)])               # Dm7
    _add_counter(139.0, [(E4, 4.0), (Gs4, 4.0), (E4, 3.0)])  # Amaj7

    # ── Soprano lead — continuous from prologue pickup (t=8) ─────────────────
    def _add_lead_phrase(
        *,
        start: float,
        notes: list[tuple[float, float]],
        synth_start: dict[str, float],
        synth_end: dict[str, float],
        amp: float,
    ) -> None:
        phrase = line(
            tones=[freq for freq, _ in notes],
            rhythm=[dur for _, dur in notes],
            pitch_kind="freq",
            amp=amp,
            synth_defaults={"engine": "filtered_stack", "waveform": "saw"},
        )
        phrase = with_synth_ramp(phrase, start=synth_start, end=synth_end)
        score.add_phrase("lead", phrase, start=start)

    lead_prologue_and_a: list[tuple[float, float]] = [
        # Prologue pickup (8–12 s)
        (A4,  1.0), (Cs5, 1.0), (B4,  1.0), (A4,  1.0),

        # A-section bar 1 vi (12–18 s)
        (Cs5, 2.0), (B4,  1.0), (A4,  2.0), (Gs4, 1.0),
        # A-section bar 2 iv (18–24 s)
        (A4,  1.5), (F4,  2.0), (A4,  1.5), (Gs4, 0.5), (A4,  0.5),
        # A-section bar 3 vi (24–30 s) — counter enters
        (B4,  1.0), (Cs5, 2.0), (B4,  1.0), (A4,  1.5), (B4,  0.5),
        # A-section bar 4 iv (30–36 s)
        (A4,  1.0), (F4,  1.5), (E4,  1.0), (D4,  1.5), (E4,  1.0),
        # A-section bar 5 vi (36–42 s) — most ornate; reaches up to A5 (tonic, next octave)
        (A4,  0.5), (Cs5, 0.75), (B4, 0.5), (Cs5, 0.75),
        (B4,  0.5), (A4,  0.5),  (Gs4,0.5), (A4,  0.5), (Cs5, 0.75), (D5, 0.375), (A5, 0.375),
        # A-section bar 6 iv (42–48 s)
        (A4,  1.5), (F4,  1.0), (E4,  1.5), (D4,  1.5), (E4,  0.5),
        # A-section bar 7 vi (48–54 s)
        (Cs5, 2.5), (B4,  1.5), (A4,  1.5), (Gs4, 0.5),
    ]

    lead_b_and_development: list[tuple[float, float]] = [
        # B-section I (54–61 s)
        (E4,  1.5), (Cs5, 1.5), (A4,  2.0), (E4,  2.0),
        # B-section V (61–68 s)
        (B4,  2.0), (Gs4, 1.5), (Fs4, 1.5), (E4,  2.0),
        # B-section I (68–75 s)
        (Cs5, 2.5), (A4,  2.0), (E4,  2.5),

        # Development F#m7 (75–83 s) — A4→E4 echoes the B-section 2–1 cadence
        (A4,  1.5), (E4,  1.5), (Cs5, 1.5), (Fs4, 1.5), (A4,  2.0),
        # Development Bm (83–91 s)
        (D4,  1.0), (Fs4, 1.5), (A4,  2.0), (B4,  1.5), (A4,  2.0),
        # Development V (91–99 s)
        (Gs4, 2.0), (B4,  2.0), (Gs4, 1.5), (E4,  2.5),
    ]

    lead_reprise_and_ending: list[tuple[float, float]] = [
        # Reprise vi (99–105 s)
        (Cs5, 2.0), (B4,  1.5), (A4,  2.0), (Gs4, 0.5),
        # Reprise I (105–111 s) — tastefully major
        (A4,  1.5), (Cs5, 2.0), (E4,  2.5),
        # Reprise vi (111–117 s)
        (B4,  1.5), (Cs5, 1.5), (B4,  1.5), (A4,  1.5),
        # Reprise I (117–123 s)
        (Cs5, 2.5), (A4,  2.0), (E4,  1.5),

        # Ending wide vi (123–131 s)
        (Cs5, 2.5), (B4,  1.0), (A4,  1.0), (Gs4, 0.5), (A4,  1.0), (B4,  2.0),
        # Ending Dm7 (131–139 s) — descend to C4 = spicy minor 7th
        (A4,  1.5), (F4,  1.0), (C4,  2.0), (D4,  1.0), (F4,  1.5), (A4,  1.0),
        # Ending Amaj7 (139–150 s) — G#4 echoes maj7; arc to C#5 and settle
        (Gs4, 2.0), (A4,  1.5), (Cs5, 2.5), (B4,  1.5), (A4,  3.5),
    ]

    _add_lead_phrase(
        start=8.0,
        notes=lead_prologue_and_a,
        synth_start={"cutoff_ratio": 6.4, "filter_env_amount": 0.40, "release": 0.30},
        synth_end={"cutoff_ratio": 6.9, "filter_env_amount": 0.46, "release": 0.28},
        amp=0.29,
    )
    _add_lead_phrase(
        start=54.0,
        notes=lead_b_and_development,
        synth_start={"cutoff_ratio": 7.3, "filter_env_amount": 0.50, "release": 0.26},
        synth_end={"cutoff_ratio": 7.5, "filter_env_amount": 0.55, "release": 0.24},
        amp=0.30,
    )
    _add_lead_phrase(
        start=99.0,
        notes=lead_reprise_and_ending,
        synth_start={"cutoff_ratio": 7.0, "filter_env_amount": 0.50, "release": 0.26},
        synth_end={"cutoff_ratio": 6.2, "filter_env_amount": 0.34, "release": 0.34},
        amp=0.29,
    )

    # Silence buffer — gives reverb and delay tails room to fully decay
    score.add_note("bass", start=154.0, duration=0.5, freq=A2, amp=0.001)

    return score


# ---------------------------------------------------------------------------
# Piece 2 – ji_melody
# ---------------------------------------------------------------------------


def build_ji_melody_score() -> Score:
    """A lyrical melodic line in 5-limit JI A major over a bass pedal."""
    f0 = 220.0
    bass_f0 = 110.0

    A3  = f0 * 1
    B3  = f0 * 9 / 8
    Cs4 = f0 * 5 / 4
    D4  = f0 * 4 / 3
    E4  = f0 * 3 / 2
    Fs4 = f0 * 5 / 3
    Gs4 = f0 * 15 / 8
    A4  = f0 * 2
    B4  = f0 * 9 / 4
    Cs5 = f0 * 5 / 2

    score = Score(
        f0=bass_f0,
        master_effects=[
            EffectSpec("reverb", {"room_size": 0.60, "damping": 0.50, "wet_level": 0.24}),
            EffectSpec("delay",  {"delay_seconds": 0.34, "feedback": 0.16, "mix": 0.10}),
        ],
    )
    score.add_voice("bass",      synth_defaults={"harmonic_rolloff": 0.50, "n_harmonics": 8, "attack": 1.0,  "decay": 0.4, "sustain_level": 0.80, "release": 3.0})
    score.add_voice("bass_fifth",synth_defaults={"harmonic_rolloff": 0.42, "n_harmonics": 6, "attack": 1.2,  "decay": 0.3, "sustain_level": 0.72, "release": 3.0})
    score.add_voice("melody",    synth_defaults={"harmonic_rolloff": 0.28, "n_harmonics": 5, "attack": 0.04, "decay": 0.12,"sustain_level": 0.74, "release": 0.45})

    melody_notes: list[tuple[float, float]] = [
        (A4,  1.50), (Gs4, 0.75), (Fs4, 0.75),
        (E4,  1.125),(D4,  0.375),(Cs4, 0.75), (B3,  0.75),
        (A3,  1.50), (B3,  0.375),(Cs4, 0.375),(D4,  3.00),
        (E4,  0.75), (Fs4, 0.75), (Gs4, 0.75), (A4,  1.50),
        (B4,  0.75), (A4,  0.75), (Gs4, 1.50), (Fs4, 0.75), (E4,  3.00),
        (Cs5, 1.50), (B4,  0.375),(A4,  0.375),(Gs4, 0.75),
        (A4,  0.375),(Gs4, 0.375),(Fs4, 1.50), (E4,  0.75),
        (D4,  0.75), (Cs4, 0.75), (B3,  0.375),(A3,  4.50),
    ]

    t = 0.0
    for freq, dur in melody_notes:
        score.add_note("melody", start=t, duration=dur, freq=freq, amp=0.42)
        t += dur

    score.add_note("bass",       start=0.0, duration=t + 1.0, freq=bass_f0,       amp=0.32)
    score.add_note("bass_fifth", start=0.0, duration=t + 1.0, freq=bass_f0 * 3/2, amp=0.16)
    return score


# ---------------------------------------------------------------------------
# Piece 3 – ji_comma_drift
# ---------------------------------------------------------------------------

# Melody defined as (ratio_of_cycle_tonic, duration).
# Multiplied by f_c in _place_melody, so it drifts and snaps with the harmony.
# Each variant covers one full I-IV-ii-V cycle = 10 s (4 × 2.5 s chords).

# V1 — bold "call" statement: leap to C#5 on I, D-arpeggio on ii (avoids flat-B spotlight)
_MELODY_V1: list[tuple[float, float]] = [
    # Over I (2.5 s): A4 → C#5 leap, sigh to G#4
    (2.0,   0.50), (5/2,  0.75), (2.0,  0.50), (15/8, 0.75),
    # Over IV (2.5 s): F#4, hold A4, back to F#4
    (5/3,   0.50), (2.0,  1.00), (5/3,  1.00),
    # Over ii (2.5 s): D4 arpeggio up through the ii chord
    (4/3,   0.50), (5/3,  0.50), (2.0,  0.75), (5/3, 0.375), (4/3, 0.375),
    # Over V (2.5 s): E4 → G#4 → B4, back to G#4
    (3/2,   0.50), (15/8, 0.50), (9/4,  1.00), (15/8, 0.50),
]

# V2 — more ornate; now exposes the flat B3 (10/9) on the ii chord
_MELODY_V2: list[tuple[float, float]] = [
    # Over I (2.5 s): quick ascending run A4→B4→C#5 and back
    (2.0,  0.375), (9/4, 0.375), (5/2, 0.375), (9/4, 0.375),
    (2.0,  0.375), (15/8,0.375), (2.0, 0.25),
    # Over IV (2.5 s): higher arc through C#5
    (5/3,  0.75),  (2.0, 0.50),  (5/2, 0.50),  (2.0, 0.25), (5/3, 0.50),
    # Over ii (2.5 s): opens with flat B3 (10/9) — the comma moment is audible
    (10/9, 0.50),  (4/3, 0.375), (5/3, 0.375), (2.0, 0.50), (5/3, 0.375), (4/3, 0.375),
    # Over V (2.5 s): peak on C#5 then step down
    (9/4,  1.00),  (5/2, 0.75),  (9/4, 0.375), (15/8, 0.375),
]

# V3 — climax variant: starts at C#5, descends during V
_MELODY_V3: list[tuple[float, float]] = [
    # Over I (2.5 s): high start on C#5, descend
    (5/2,  1.00), (9/4,  0.50), (2.0,  0.50), (15/8, 0.50),
    # Over IV (2.5 s): F#4 with passing notes through A4
    (5/3,  0.50), (2.0,  0.50), (5/3,  0.375), (3/2, 0.375), (5/3, 0.75),
    # Over ii (2.5 s): D arpeggio climbing (avoid dwelling on flat B)
    (4/3,  0.75), (5/3,  0.50), (2.0,  0.75),  (5/3, 0.50),
    # Over V (2.5 s): settle through G#4 to E4 — arch coming down from climax
    (15/8, 0.50), (9/4,  0.75), (15/8, 0.50),  (3/2, 0.75),
]

# Snap-back melody — 3 s at pure drone_freq (landing)
_MELODY_SNAP: list[tuple[float, float]] = [
    (2.0, 1.00), (5/2, 0.75), (9/4, 0.50), (2.0, 0.75),
]

# Long snap-back — 5 s for the final block
_MELODY_SNAP_LONG: list[tuple[float, float]] = [
    (2.0, 1.25), (15/8, 0.50), (2.0, 0.75), (9/4, 0.75), (5/2, 0.75), (9/4, 0.50), (2.0, 0.50),
]


def _place_melody(
    score: Score,
    f_c: float,
    notes: list[tuple[float, float]],
    start: float,
    amp: float = 0.38,
) -> None:
    t = start
    for ratio, dur in notes:
        score.add_note("melody", start=t, duration=dur, freq=f_c * ratio, amp=amp)
        t += dur


def _place_chord_cycle(score: Score, f_c: float, t0: float, chord_dur: float) -> None:
    """One I–IV–ii–V cycle tuned purely from tonic f_c."""
    note_dur = chord_dur + 0.35
    bass_dur = chord_dur * 0.84
    arp_gap  = 0.08

    cycle: list[tuple[float, float, list[tuple[float, float]]]] = [
        (0,           1/2,    [(1.0,   0.20), (5/4,  0.16), (3/2,  0.13)]),  # I
        (chord_dur,   4/3/2,  [(4/3,   0.18), (5/3,  0.14), (2.0,  0.11)]),  # IV
        (2*chord_dur, 10/9/2, [(10/9,  0.17), (4/3,  0.13), (5/3,  0.11)]),  # ii
        (3*chord_dur, 3/2/2,  [(3/2,   0.20), (15/8, 0.15), (9/4,  0.13)]),  # V
    ]
    for chord_offset, bass_ratio, harmony_notes in cycle:
        t = t0 + chord_offset
        score.add_note("bass",  start=t, duration=bass_dur,  freq=f_c * bass_ratio, amp=0.26)
        for idx, (ratio, amp) in enumerate(harmony_notes):
            score.add_note("chord", start=t + idx * arp_gap, duration=note_dur, freq=f_c * ratio, amp=amp)


def _place_snap(score: Score, drone_freq: float, t: float, dur: float, melody: list[tuple[float, float]]) -> None:
    """Sustained I chord at drone_freq — the snap-back moment."""
    note_dur = dur - 0.4
    score.add_note("bass",  start=t, duration=note_dur, freq=drone_freq / 2,     amp=0.30)
    score.add_note("chord", start=t, duration=note_dur, freq=drone_freq,          amp=0.22)
    score.add_note("chord", start=t, duration=note_dur, freq=drone_freq * 5 / 4, amp=0.18)
    score.add_note("chord", start=t, duration=note_dur, freq=drone_freq * 3 / 2, amp=0.15)
    _place_melody(score, drone_freq, melody, t, amp=0.42)


def build_ji_comma_drift_score() -> Score:
    """Syntonic comma pump with an anthemic drifting melody.

    Structure: intro → [1 drift cycle + snap] × 3 → clean coda.

    One drift cycle = 21.5 ¢ flat.  The snap-back is a sustained I chord at
    the true A=220, giving the ear a clear correction before the next drift.
    Melody variants escalate in ornament across the three blocks before
    the coda restates V1 cleanly over a pure A=220 cycle.
    """
    drone_freq = 220.0
    chord_dur  = 2.5
    cycle_dur  = 4 * chord_dur  # 10 s per I-IV-ii-V loop
    comma      = 81 / 80

    intro_dur     = 4.0
    snap_dur      = 3.0
    snap_long_dur = 5.0
    n_blocks      = 3

    total_dur = (
        intro_dur
        + (n_blocks - 1) * (cycle_dur + snap_dur)
        + cycle_dur + snap_long_dur   # last block has long snap
        + cycle_dur                   # coda
        + 3.0                         # reverb tail
    )

    score = Score(
        f0=drone_freq,
        master_effects=[
            EffectSpec("delay",  {"delay_seconds": 0.26, "feedback": 0.14, "mix": 0.09}),
            EffectSpec("reverb", {"room_size": 0.68, "damping": 0.44, "wet_level": 0.24}),
        ],
    )

    score.add_voice("drone", synth_defaults={"harmonic_rolloff": 0.48, "n_harmonics": 8, "attack": 1.8, "decay": 0.4, "sustain_level": 0.82, "release": 3.5})
    score.add_voice("chord", synth_defaults={"harmonic_rolloff": 0.34, "n_harmonics": 6, "attack": 0.35, "decay": 0.20, "sustain_level": 0.70, "release": 0.90})
    score.add_voice("bass",  synth_defaults={"harmonic_rolloff": 0.50, "n_harmonics": 8, "attack": 0.14, "decay": 0.20, "sustain_level": 0.76, "release": 0.70})
    score.add_voice("melody",synth_defaults={"harmonic_rolloff": 0.24, "n_harmonics": 5, "attack": 0.05, "decay": 0.12, "sustain_level": 0.72, "release": 0.45})

    score.add_note("drone", start=0.0, duration=total_dur, freq=drone_freq, amp=0.22, label="A=220 drone")

    melody_variants = [_MELODY_V1, _MELODY_V2, _MELODY_V3]
    snap_melodies   = [_MELODY_SNAP, _MELODY_SNAP, _MELODY_SNAP_LONG]
    snap_durations  = [snap_dur, snap_dur, snap_long_dur]

    t = intro_dur
    for block_idx in range(n_blocks):
        # One drift cycle per block (21.5 ¢ flat)
        f_c = drone_freq * (1 / comma)
        logger.info(
            "block %d  tonic=%.2f Hz  (%.2f ¢ from drone)",
            block_idx, f_c, 1200 * math.log2(f_c / drone_freq),
        )
        _place_chord_cycle(score, f_c, t, chord_dur)
        _place_melody(score, f_c, melody_variants[block_idx], t)
        t += cycle_dur

        _place_snap(score, drone_freq, t, snap_durations[block_idx], snap_melodies[block_idx])
        t += snap_durations[block_idx]

    # Clean coda — pure A=220, V1 melody restated
    _place_chord_cycle(score, drone_freq, t, chord_dur)
    _place_melody(score, drone_freq, _MELODY_V1, t, amp=0.40)

    return score


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PIECES: dict[str, PieceDefinition] = {
    "ji_chorale": PieceDefinition(
        name="ji_chorale",
        output_name="17_ji_chorale.wav",
        build_score=build_ji_chorale_score,
    ),
    "ji_melody": PieceDefinition(
        name="ji_melody",
        output_name="18_ji_melody.wav",
        build_score=build_ji_melody_score,
    ),
    "ji_comma_drift": PieceDefinition(
        name="ji_comma_drift",
        output_name="19_ji_comma_drift.wav",
        build_score=build_ji_comma_drift_score,
    ),
}
