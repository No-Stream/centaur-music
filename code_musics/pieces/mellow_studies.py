"""Mellow studies — four short pieces exploring approachable xenharmony.

Aesthetic center: the mellow-emotive end of Aphex Twin (Aisatsana, Avril, Flim),
using JI and septimal harmony for either "more perfect" consonance or "subtly
alien" bittersweet colour.

All pieces use near-sine additive voices with a touch of reverb. The xenharmonic
materials are:
  1. septimal_lullaby   — 7-limit otonal tetrad 4:5:6:7 (the 7th partial is
                           the bittersweet heart)
  2. comma_rain         — a 3-note JI figure repeated, drifting up by one
                           syntonic comma (81/80 ≈ 22 cents) each pass
  3. harmonic_arpeggios — melody drawn from harmonic partials 8–16; partials
                           11 and 13 give the open, slightly-alien colour
  4. utonal_elegy       — utonal tetrad 1/1 : 7/6 : 7/5 : 7/4 (subharmonic
                           mirror of the otonal, more mournful)
"""

from __future__ import annotations

from code_musics.composition import RhythmCell, line
from code_musics.humanize import VelocityHumanizeSpec
from code_musics.pieces._shared import SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import EffectSpec, Score

# ---------------------------------------------------------------------------
# Shared synth voices
# ---------------------------------------------------------------------------

# Near-pure sine — 2 harmonics with heavy rolloff
_NEAR_SINE: dict = {
    "engine": "additive",
    "n_harmonics": 2,
    "harmonic_rolloff": 0.35,
}

# Slightly warmer sine — still simple but with a tiny bit of body
_WARM_SINE: dict = {
    "engine": "additive",
    "n_harmonics": 4,
    "harmonic_rolloff": 0.55,
    "brightness_tilt": -0.05,
}


# ---------------------------------------------------------------------------
# 1. Septimal Lullaby
# ---------------------------------------------------------------------------


def build_septimal_lullaby() -> Score:
    """Sparse melody over the 7-limit otonal tetrad 4:5:6:7.

    f0 = 110 Hz (A2). Harmonic partials used:
      partial 2.0  = A3  (220 Hz)   — root
      partial 2.5  = C#4 (275 Hz)   — pure major third (5/4 above A3)
      partial 3.0  = E4  (330 Hz)   — pure fifth (3/2 above A3)
      partial 3.5  = G♭7-4 (385 Hz)  — septimal minor 7th (7/4 above A3)
      partial 4.0  = A4  (440 Hz)   — octave

    The 7th partial is the bittersweet heart: sweeter than a minor 7th,
    darker than a major 6th. Avril 14th character — slow, sparse, lots of space.
    """
    score = Score(f0=110.0, master_effects=[SOFT_REVERB_EFFECT])

    score.add_voice(
        "bass",
        synth_defaults={**_NEAR_SINE, "attack": 1.5, "release": 5.0},
    )
    score.add_voice(
        "melody",
        synth_defaults={**_WARM_SINE, "attack": 0.06, "release": 2.8},
        velocity_humanize=VelocityHumanizeSpec(seed=1),
    )

    # Bass: A2 root throughout; E3 (fifth) enters quietly after a few seconds
    score.add_note("bass", start=0.0, duration=38.0, partial=1.0, amp_db=-24.0)
    score.add_note("bass", start=5.0, duration=32.0, partial=3 / 2, amp_db=-28.0)

    # Phrase A: descend through the 7th — E4 → G♭7 → E4 → C#4 → A3
    # The 3.5 partial (G♭7) is the dissonant beauty — linger on it
    phrase_a = line(
        tones=[3.0, 3.5, 3.0, 2.5, 2.0],
        rhythm=RhythmCell(
            spans=(1.2, 1.6, 0.9, 1.5, 2.5),
            gates=(0.80, 0.88, 0.72, 0.90, 1.00),
        ),
        amp_db=-13.0,
    )

    # Phrase B: ascend and linger on the 7th, then up to octave — A3 → C#4 → E4 → G♭7 → A4
    phrase_b = line(
        tones=[2.0, 2.5, 3.0, 3.5, 4.0],
        rhythm=RhythmCell(
            spans=(0.9, 1.1, 0.9, 1.8, 3.2),
            gates=(0.72, 0.78, 0.72, 0.95, 1.00),
        ),
        amp_db=-12.0,
    )

    # Phrase C: closing question — A4 → G♭7 → E4, unresolved
    phrase_c = line(
        tones=[4.0, 3.5, 3.0],
        rhythm=RhythmCell(spans=(0.9, 1.4, 3.5), gates=(0.70, 0.92, 1.00)),
        amp_db=-15.0,
    )

    score.add_phrase("melody", phrase_a, start=2.5)  # open with descent
    score.add_phrase("melody", phrase_b, start=12.0)  # ascend, linger on 7th
    score.add_phrase("melody", phrase_a, start=21.5, amp_scale=0.85)  # quieter reprise
    score.add_phrase("melody", phrase_c, start=31.5)  # unresolved close

    return score


# ---------------------------------------------------------------------------
# 2. Comma Rain
# ---------------------------------------------------------------------------


def build_comma_rain() -> Score:
    """A 5-note JI figure repeated 4 times, drifting up by one syntonic comma
    (81/80 ≈ 22 cents) each pass.

    f0 = 220 Hz (A3). The figure uses partials 1.0 / 5/4 / 3/2 — a pure
    major triad. Each repetition shifts all tones up by 81/80, so the
    harmony brightens almost imperceptibly. After four passes (≈ 88 cents
    total drift) a closing descent returns gracefully to the root.

    More Aisatsana than Avril — the piece is about time and memory.
    """
    score = Score(f0=220.0, master_effects=[SOFT_REVERB_EFFECT])

    score.add_voice(
        "melody",
        synth_defaults={**_WARM_SINE, "attack": 0.05, "release": 3.5},
        velocity_humanize=VelocityHumanizeSpec(seed=42),
    )
    score.add_voice(
        "bass",
        synth_defaults={**_NEAR_SINE, "attack": 1.2, "release": 4.0},
    )

    # Repeating figure: 1.0 → 5/4 → 3/2 → 5/4 → 1.0 (A3 → C#4 → E4 → C#4 → A3)
    base_tones = [1.0, 5 / 4, 3 / 2, 5 / 4, 1.0]
    phrase_rhythm = RhythmCell(
        spans=(1.5, 1.2, 1.0, 1.2, 2.8),
        gates=(0.72, 0.78, 0.72, 0.78, 1.00),
    )
    phrase_dur = 1.5 + 1.2 + 1.0 + 1.2 + 2.8  # 7.7 s
    cycle_dur = phrase_dur + 1.8  # 9.5 s gap between phrases

    comma = 81 / 80  # syntonic comma

    for i in range(4):
        drift = comma**i
        # Shift all partial values upward by accumulated comma drift
        shifted_tones = [t * drift for t in base_tones]
        phrase = line(
            tones=shifted_tones,
            rhythm=phrase_rhythm,
            amp_db=-13.0 + i * 0.4,  # very slightly louder/brighter as it drifts up
        )
        score.add_phrase("melody", phrase, start=2.0 + i * cycle_dur)

    # Closing descent: from near where we ended, back down to the root
    # The last drift was (81/80)^3, so base ~= 1.075; descend gracefully
    final_drift = comma**3
    close = line(
        tones=[3 / 2 * final_drift, 5 / 4 * final_drift, 1.0 * final_drift, 1.0],
        rhythm=RhythmCell(spans=(1.5, 1.5, 2.0, 4.5), gates=(0.78, 0.80, 0.90, 1.00)),
        amp_db=-16.0,
    )
    score.add_phrase("melody", close, start=2.0 + 4 * cycle_dur + 0.5)

    # Bass: A2 drone throughout (partial 0.5 of f0=220 = 110 Hz)
    total_dur = 2.0 + 4 * cycle_dur + 0.5 + phrase_dur + 4.0
    score.add_note("bass", start=0.0, duration=total_dur, partial=0.5, amp_db=-26.0)

    return score


# ---------------------------------------------------------------------------
# 3. Harmonic Series Arpeggios
# ---------------------------------------------------------------------------


def build_harmonic_arpeggios() -> Score:
    """Melody from harmonic partials 8–16 over bass on partials 1–3.

    f0 = 55 Hz (A1), so the melody lives in vocal range (440–880 Hz):
      partial  8 = 440 Hz  A4    — familiar
      partial  9 = 495 Hz  B4    — familiar
      partial 10 = 550 Hz  C#5   — familiar
      partial 11 = 605 Hz  ~D#5  — neutral/alien (≈ 551 cents above A4)
      partial 12 = 660 Hz  E5    — familiar
      partial 13 = 715 Hz  ~F#5  — slightly-flat sixth (≈ 840 cents)
      partial 14 = 770 Hz  G♭75  — septimal minor 7th
      partial 16 = 880 Hz  A5    — octave

    Partials 11 and 13 give the melody its open-sky alien quality; the
    others ground it in something close to pentatonic. Flim-ish gentle pulse.
    """
    score = Score(f0=55.0, master_effects=[SOFT_REVERB_EFFECT])

    score.add_voice(
        "bass",
        synth_defaults={**_NEAR_SINE, "attack": 1.0, "release": 3.5},
    )
    score.add_voice(
        "melody",
        synth_defaults={**_WARM_SINE, "attack": 0.04, "release": 1.8},
        velocity_humanize=VelocityHumanizeSpec(seed=7),
    )

    # Bass: A2 (partial 2) + E3 (partial 3) sustained; brief A1 (partial 1) pulse
    score.add_note("bass", start=0.0, duration=45.0, partial=2.0, amp_db=-20.0)
    score.add_note("bass", start=0.0, duration=45.0, partial=3.0, amp_db=-24.0)
    score.add_note("bass", start=6.0, duration=18.0, partial=1.0, amp_db=-22.0)

    # Phrase A: rise through the series, touching the 11th — 8→10→12→11→10→9→8
    phrase_a = line(
        tones=[8, 10, 12, 11, 10, 9, 8],
        rhythm=RhythmCell(
            spans=(0.5, 0.5, 0.6, 0.8, 0.5, 0.5, 1.8),
            gates=0.70,
        ),
        amp_db=-12.0,
    )

    # Phrase B: leap to septimal 14th, drift back down — 12→14→13→12→10→8
    phrase_b = line(
        tones=[12, 14, 13, 12, 10, 8],
        rhythm=RhythmCell(
            spans=(0.5, 0.9, 0.7, 0.5, 0.5, 2.2),
            gates=(0.65, 0.88, 0.72, 0.70, 0.65, 1.00),
        ),
        amp_db=-11.0,
    )

    # Phrase C: meander through the alien partials — 9→11→13→12→11→9→8
    phrase_c = line(
        tones=[9, 11, 13, 12, 11, 9, 8],
        rhythm=RhythmCell(
            spans=(0.6, 0.7, 0.8, 0.6, 0.7, 0.5, 2.8),
            gates=0.75,
        ),
        amp_db=-13.0,
    )

    score.add_phrase("melody", phrase_a, start=1.0)
    score.add_phrase("melody", phrase_b, start=7.5)
    score.add_phrase("melody", phrase_a, start=15.0, amp_scale=0.90)
    score.add_phrase("melody", phrase_c, start=22.0)
    score.add_phrase("melody", phrase_b, start=32.0, amp_scale=0.82)

    return score


# ---------------------------------------------------------------------------
# 4. Utonal Elegy
# ---------------------------------------------------------------------------


def build_utonal_elegy() -> Score:
    """Slow-moving utonal tetrad: 1/1 : 7/6 : 7/5 : 7/4.

    f0 = 220 Hz (A3). The utonal tetrad is the subharmonic mirror of the
    otonal 4:5:6:7 — built from undertones rather than overtones, and
    distinctly more mournful:
      partial 1.0       = A3  (220.0 Hz)   — root
      partial 7/6 ≈1.17 = ~B♭3 (256.7 Hz) — septimal minor third (~267 cents;
                                              flatter than a standard minor 3rd)
      partial 7/5 = 1.4 = ~D4 (308.0 Hz)  — septimal tritone (~583 cents)
      partial 7/4 = 1.75= G♭7-4 (385.0 Hz) — septimal minor 7th

    The 7/6 (septimal minor third) is the haunting interval: between a
    minor third and a major second, ancient-sounding and doleful.
    Notes enter one by one, like a slow exhale.
    """
    score = Score(f0=220.0, master_effects=[SOFT_REVERB_EFFECT])

    score.add_voice(
        "chord",
        synth_defaults={**_NEAR_SINE, "attack": 1.4, "release": 5.0},
    )
    score.add_voice(
        "melody",
        synth_defaults={**_WARM_SINE, "attack": 0.08, "release": 3.2},
        velocity_humanize=VelocityHumanizeSpec(seed=3),
    )
    score.add_voice(
        "bass",
        synth_defaults={**_NEAR_SINE, "attack": 1.8, "release": 6.0},
    )

    # Utonal chord tone partials (of f0=220 Hz)
    u_root = 1.0  # A3  = 220.0 Hz
    u_m3 = 7 / 6  # ~B♭3 = 256.7 Hz  (septimal minor third)
    u_tri = 7 / 5  # ~D4  = 308.0 Hz  (septimal tritone)
    u_m7 = 7 / 4  # G♭74 = 385.0 Hz  (septimal minor 7th)

    # Bass: A2 (half of f0) — quiet foundation
    score.add_note("bass", start=0.0, duration=50.0, partial=0.5, amp_db=-25.0)

    # Chord block 1: notes enter one by one, low to high
    score.add_note("chord", start=0.5, duration=14.0, partial=u_root, amp_db=-21.0)
    score.add_note("chord", start=2.5, duration=12.0, partial=u_m3, amp_db=-23.0)
    score.add_note("chord", start=4.5, duration=10.0, partial=u_tri, amp_db=-24.0)
    score.add_note("chord", start=6.5, duration=8.0, partial=u_m7, amp_db=-25.0)

    # Chord block 2: restate an octave higher, spaced differently
    score.add_note("chord", start=18.0, duration=12.0, partial=u_root * 2, amp_db=-22.0)
    score.add_note("chord", start=19.5, duration=11.0, partial=u_m3, amp_db=-23.0)
    score.add_note("chord", start=21.5, duration=10.0, partial=u_tri, amp_db=-24.0)

    # Melody phrase A: descend through the utonal space — G♭7 → tri → m3 → root → m3 → G♭7
    phrase_a = line(
        tones=[u_m7, u_tri, u_m3, u_root, u_m3, u_m7],
        rhythm=RhythmCell(
            spans=(1.5, 1.2, 1.0, 1.5, 1.2, 3.8),
            gates=(0.70, 0.80, 0.75, 0.85, 0.80, 1.00),
        ),
        amp_db=-13.0,
    )

    # Melody phrase B: descend from high octave — root×2 → G♭7 → tri → m3 → root
    phrase_b = line(
        tones=[u_root * 2, u_m7, u_tri, u_m3, u_root],
        rhythm=RhythmCell(
            spans=(1.0, 1.5, 1.2, 1.5, 4.5),
            gates=(0.75, 0.82, 0.80, 0.90, 1.00),
        ),
        amp_db=-14.0,
    )

    score.add_phrase("melody", phrase_a, start=2.0)
    score.add_phrase("melody", phrase_b, start=14.5)
    score.add_phrase("melody", phrase_a, start=28.0, amp_scale=0.85)
    score.add_phrase("melody", phrase_b, start=39.0, amp_scale=0.75)

    return score


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 5. Fond
# ---------------------------------------------------------------------------


def build_fond() -> Score:
    """'Fond' — a song built on the 7-limit otonal tetrad 4:5:6:7.

    f0 = 110 Hz (A2). One melodic voice over a barely-there bass pedal.
    The 7th partial (partial 3.5 = 385 Hz, G♭7) is the emotional center —
    the melody keeps returning to it, hovering, unable to fully resolve.

    Structure:
      Intro    ( 0–12 s): bass alone, almost subliminal
      Verse    (12–37 s): exhale phrase (descent A4→G♭7→E4→C#4→A3),
                          long silence, then ascending question landing on G♭7
      Bridge   (40–62 s): ascent through the tetrad to C#5 (climax),
                          descent back through the 7th to root
      Reprise  (65–78 s): opening exhale, quieter and more resigned
      Outro    (79–88 s): G♭7 alone, one last time, fading

    The bass never moves. The 7th partial is the piece's one unresolved question.
    """
    score = Score(f0=110.0, master_effects=[SOFT_REVERB_EFFECT])

    score.add_voice(
        "bass",
        synth_defaults={**_NEAR_SINE, "attack": 3.0, "release": 6.0},
    )
    score.add_voice(
        "melody",
        synth_defaults={**_WARM_SINE, "attack": 0.07, "release": 3.5},
        velocity_humanize=VelocityHumanizeSpec(seed=11),
        effects=[
            # Slow, rhythmically ambiguous delay — trails each note without
            # cluttering. Low feedback so it's atmospheric, not obviously echo-y.
            EffectSpec("delay", {"delay_seconds": 0.52, "feedback": 0.28, "mix": 0.20}),
        ],
    )

    # Bass: A2 root and E3 fifth held throughout — architectural, not melodic.
    # Long attacks so they barely register as attacks at all.
    score.add_note("bass", start=0.0, duration=88.0, partial=1.0, amp_db=-26.0)
    score.add_note("bass", start=6.0, duration=82.0, partial=3 / 2, amp_db=-30.0)

    # Partial shorthands (all relative to f0=110 Hz)
    A3 = 2.0    # 220 Hz
    Cs4 = 5 / 2  # 275 Hz  — pure major third above A3
    E4 = 3.0    # 330 Hz  — pure fifth above A3
    Gb7 = 7 / 2  # 385 Hz  — septimal minor seventh above A3; THE note
    A4 = 4.0    # 440 Hz
    Cs5 = 5.0   # 550 Hz  — pure major third, upper octave; climax

    # --- Verse ---------------------------------------------------------------
    # Exhale phrase: start high, fall through the 7th, land on root.
    # The G♭7 is the first strange thing heard — linger there.
    score.add_note("melody", start=12.0, duration=1.7, partial=A4, amp_db=-12.0)
    score.add_note("melody", start=14.0, duration=2.4, partial=Gb7, amp_db=-10.5)
    score.add_note("melody", start=16.7, duration=1.1, partial=E4, amp_db=-13.0)
    score.add_note("melody", start=18.1, duration=1.7, partial=Cs4, amp_db=-13.5)
    score.add_note("melody", start=20.2, duration=3.8, partial=A3, amp_db=-15.0)

    # Long rest (silence is compositional here).

    # Ascending question: A3 → C#4 → E4 → G♭7 [held].
    # Ends unresolved on the 7th — the piece's first unanswered question.
    score.add_note("melody", start=29.0, duration=1.1, partial=A3, amp_db=-13.0)
    score.add_note("melody", start=30.4, duration=1.1, partial=Cs4, amp_db=-12.5)
    score.add_note("melody", start=31.8, duration=0.9, partial=E4, amp_db=-12.0)
    score.add_note("melody", start=33.0, duration=4.5, partial=Gb7, amp_db=-11.0)

    # --- Bridge --------------------------------------------------------------
    # Ascent to climax: the melody rises through all four tetrad tones to C#5.
    # C#5 (partial 5.0) is the pure major third in the upper octave — bright
    # and open after all the 7th-partial hovering.
    score.add_note("melody", start=41.0, duration=1.0, partial=A3, amp_db=-12.5)
    score.add_note("melody", start=42.3, duration=0.9, partial=Cs4, amp_db=-12.0)
    score.add_note("melody", start=43.5, duration=0.8, partial=E4, amp_db=-11.5)
    score.add_note("melody", start=44.6, duration=0.9, partial=Gb7, amp_db=-11.0)
    score.add_note("melody", start=45.8, duration=1.3, partial=A4, amp_db=-10.5)
    score.add_note("melody", start=47.4, duration=3.2, partial=Cs5, amp_db=-9.5)

    # Descent from climax: C#5 → A4 → G♭7 [linger again] → E4 → C#4 → A3.
    # The 7th appears a second time on the way down — it's unavoidable.
    score.add_note("melody", start=51.8, duration=1.1, partial=A4, amp_db=-11.0)
    score.add_note("melody", start=53.2, duration=2.2, partial=Gb7, amp_db=-10.5)
    score.add_note("melody", start=55.7, duration=1.0, partial=E4, amp_db=-13.0)
    score.add_note("melody", start=57.0, duration=1.5, partial=Cs4, amp_db=-14.0)
    score.add_note("melody", start=58.8, duration=3.0, partial=A3, amp_db=-15.5)

    # --- Reprise -------------------------------------------------------------
    # Opening exhale returns, quieter. The shape is familiar now — resigned,
    # not searching. Stops before the ascending question this time.
    score.add_note("melody", start=65.0, duration=1.4, partial=A4, amp_db=-14.5)
    score.add_note("melody", start=66.7, duration=2.0, partial=Gb7, amp_db=-13.0)
    score.add_note("melody", start=69.0, duration=1.0, partial=E4, amp_db=-15.0)
    score.add_note("melody", start=70.3, duration=1.5, partial=Cs4, amp_db=-15.5)
    score.add_note("melody", start=72.2, duration=3.0, partial=A3, amp_db=-17.0)

    # --- Outro ---------------------------------------------------------------
    # The 7th partial alone, struck once, left to fade.
    # The question remains open.
    score.add_note("melody", start=79.0, duration=6.0, partial=Gb7, amp_db=-14.0)

    return score


# ---------------------------------------------------------------------------
# 6. Ether
# ---------------------------------------------------------------------------


def build_ether() -> Score:
    """'Ether' — harmonic series melody that falls from pure JI into alien territory.

    f0 = 55 Hz (A1). Four voices: pulse, bass, pad, melody.

    Partial map (melody range, all from A1):
      8  = A4  (440 Hz)  — familiar; pure octave
      9  = B4  (495 Hz)  — familiar; pure major second above A4
      10 = C#5 (550 Hz)  — familiar; pure major third above A4
      11 = ~605 Hz       — ALIEN; neutral fourth, ~551 cents above A4
      12 = E5  (660 Hz)  — familiar; pure fifth above A4
      13 = ~715 Hz       — ALIEN; slightly-flat minor sixth, ~840 cents
      14 = G♭75 (770 Hz) — alien but consonant; septimal minor seventh
      15 = G#5 (825 Hz)  — familiar; pure major seventh (leading tone)
      16 = A5  (880 Hz)  — familiar; octave

    Structure:
      Intro      (  0–10 s): bass + pulse alone; pure A1 root
      A section  ( 10–42 s): clean JI melody (8/9/10/12/15) over 8+12 pad;
                             the non-beating purity of JI is audible in the pad
      Transition ( 42–57 s): partial 11 appears once, almost unnoticed;
                             then again, longer — the dream begins
      B section  ( 57–76 s): alien territory; 11/13/14 in melody, 10+14 in pad
      Flicker    ( 76–84 s): alien at its most vivid; quick darting through
                             11/13/14; pad shifts to 11+12
      Climax     ( 85–95 s): ascent 9→10→12→14→15→16; reaches pure A5 octave
                             via alien partial 14; pad on 12+16
      Descent    ( 97–109 s): 16→15→14→12→11→10→9→8; full terrain, deliberate;
                             pad returns to 8+12
      Conclusion (103–118 s): bass partial 1 (A1, 55 Hz) heard for first time;
                             everything grounds to the fundamental
    """
    score = Score(f0=55.0, master_effects=[SOFT_REVERB_EFFECT])

    score.add_voice(
        "pulse",
        synth_defaults={**_NEAR_SINE, "attack": 0.015, "release": 0.45},
    )
    score.add_voice(
        "bass",
        synth_defaults={**_NEAR_SINE, "attack": 2.0, "release": 5.0},
    )
    score.add_voice(
        "pad",
        synth_defaults={**_NEAR_SINE, "attack": 1.8, "release": 5.5},
    )
    score.add_voice(
        "melody",
        synth_defaults={**_WARM_SINE, "attack": 0.06, "release": 2.5},
        velocity_humanize=VelocityHumanizeSpec(seed=17),
        effects=[
            EffectSpec("delay", {"delay_seconds": 0.48, "feedback": 0.22, "mix": 0.16}),
        ],
    )

    # Pulse: very quiet, felt rather than heard. A2 (partial 2), every 1.8 s.
    # Drops out during the climax (89–97 s) so partial 16 hangs in silence alone.
    n_pulses = int(118.0 / 1.8) + 1
    for i in range(n_pulses):
        pulse_t = i * 1.8
        if 89.0 <= pulse_t <= 97.0:
            continue
        score.add_note("pulse", start=pulse_t, duration=0.25, partial=2.0, amp_db=-28.0)

    # Bass: A2 throughout; E3 (fifth) enters quietly with the A section.
    # At the conclusion, A1 (partial 1, 55 Hz) drops in for the first time —
    # the true fundamental that the whole piece has been built on.
    score.add_note("bass", start=0.0, duration=118.0, partial=2.0, amp_db=-20.0)
    score.add_note("bass", start=10.0, duration=95.0, partial=3.0, amp_db=-25.0)
    score.add_note("bass", start=103.0, duration=15.0, partial=1.0, amp_db=-20.0,
                   synth={"attack": 5.0, "release": 7.0})

    # Pad chords — two simultaneous long tones each time.
    # A section: 8+12 = A4+E5, a pure fifth. Zero beating. This is the sound
    # of JI purity — the listener should hear the stillness of it.
    score.add_note("pad", start=8.0, duration=24.0, partial=8.0, amp_db=-19.0)
    score.add_note("pad", start=8.0, duration=24.0, partial=12.0, amp_db=-21.0)
    # Midway through A section, shift to 8+10 (pure major third — still clean).
    score.add_note("pad", start=30.0, duration=14.0, partial=8.0, amp_db=-20.0)
    score.add_note("pad", start=30.0, duration=14.0, partial=10.0, amp_db=-22.0)
    # Transition: 9+12 — still grounded but the familiar fifth is now off-root.
    score.add_note("pad", start=44.0, duration=14.0, partial=9.0, amp_db=-20.0)
    score.add_note("pad", start=44.0, duration=14.0, partial=12.0, amp_db=-22.0)
    # B section: 10+14 = C#5 + G♭75. The septimal minor 7th in the pad.
    # Alien-sounding but consonant — the pad makes it ambient, not harsh.
    score.add_note("pad", start=57.0, duration=20.0, partial=10.0, amp_db=-19.0)
    score.add_note("pad", start=57.0, duration=20.0, partial=14.0, amp_db=-23.0)
    # Alien flicker: 11+12 in the pad — the alien note alongside the familiar fifth.
    score.add_note("pad", start=76.0, duration=11.0, partial=11.0, amp_db=-21.0)
    score.add_note("pad", start=76.0, duration=11.0, partial=12.0, amp_db=-22.0)
    # Climax: 12+16 — pure fifth an octave up. Bright, open, clear.
    score.add_note("pad", start=86.0, duration=14.0, partial=12.0, amp_db=-19.0)
    score.add_note("pad", start=86.0, duration=12.0, partial=16.0, amp_db=-22.0)
    # Post-climax/return: back to 8+12 — the home sound, fading long.
    score.add_note("pad", start=98.0, duration=22.0, partial=8.0, amp_db=-19.0)
    score.add_note("pad", start=98.0, duration=20.0, partial=12.0, amp_db=-21.0)

    # ---------------------------------------------------------------------------
    # Melody
    # ---------------------------------------------------------------------------

    # A section — clean, singable, pure JI intervals only.
    # The goal is to establish the sound of "perfectly in tune" so clearly that
    # when the alien partials arrive, the contrast is visceral.

    # Phrase 1: ascending arpeggio — A4 → C#5 → E5 → C#5 → B4 → A4
    score.add_note("melody", start=10.0, duration=0.50, partial=8.0, amp_db=-13.5)
    score.add_note("melody", start=10.7, duration=0.52, partial=10.0, amp_db=-13.0)
    score.add_note("melody", start=11.4, duration=0.55, partial=12.0, amp_db=-12.5)
    score.add_note("melody", start=12.1, duration=0.62, partial=10.0, amp_db=-13.5)
    score.add_note("melody", start=12.9, duration=0.52, partial=9.0, amp_db=-14.0)
    score.add_note("melody", start=13.6, duration=1.90, partial=8.0, amp_db=-14.5)

    # Phrase 2: wider — climbs to G#5 (partial 15, leading tone), then descends
    score.add_note("melody", start=17.5, duration=0.48, partial=8.0, amp_db=-13.0)
    score.add_note("melody", start=18.1, duration=0.58, partial=9.0, amp_db=-12.5)
    score.add_note("melody", start=18.8, duration=0.62, partial=12.0, amp_db=-12.0)
    score.add_note("melody", start=19.6, duration=0.68, partial=15.0, amp_db=-11.5)  # G#5 peak
    score.add_note("melody", start=20.4, duration=0.58, partial=12.0, amp_db=-12.5)
    score.add_note("melody", start=21.1, duration=0.52, partial=10.0, amp_db=-13.0)
    score.add_note("melody", start=21.8, duration=2.20, partial=8.0, amp_db=-14.5)

    # Phrase 3: reprise of phrase 1, slightly quieter
    score.add_note("melody", start=26.0, duration=0.50, partial=8.0, amp_db=-14.0)
    score.add_note("melody", start=26.7, duration=0.52, partial=10.0, amp_db=-13.5)
    score.add_note("melody", start=27.4, duration=0.55, partial=12.0, amp_db=-13.0)
    score.add_note("melody", start=28.1, duration=0.62, partial=10.0, amp_db=-14.0)
    score.add_note("melody", start=28.9, duration=0.52, partial=9.0, amp_db=-14.5)
    score.add_note("melody", start=29.6, duration=2.40, partial=8.0, amp_db=-15.5)

    # Phrase 4: simpler — stays in the middle, gentle
    score.add_note("melody", start=34.0, duration=0.62, partial=9.0, amp_db=-14.5)
    score.add_note("melody", start=34.8, duration=0.72, partial=12.0, amp_db=-14.0)
    score.add_note("melody", start=35.7, duration=0.62, partial=10.0, amp_db=-14.5)
    score.add_note("melody", start=36.5, duration=0.58, partial=9.0, amp_db=-15.0)
    score.add_note("melody", start=37.2, duration=2.80, partial=8.0, amp_db=-16.0)

    # ---------------------------------------------------------------------------
    # Transition — partial 11 enters the dream, almost by accident
    # ---------------------------------------------------------------------------

    # Phrase 5: familiar ascent, but partial 11 appears in place of 12 — just once.
    # It's close enough to E5 that the listener might think they misheard.
    score.add_note("melody", start=42.0, duration=0.55, partial=8.0, amp_db=-13.0)
    score.add_note("melody", start=42.7, duration=0.62, partial=10.0, amp_db=-12.5)
    score.add_note("melody", start=43.5, duration=1.50, partial=11.0, amp_db=-13.0)  # ← alien, quiet
    score.add_note("melody", start=45.2, duration=0.58, partial=10.0, amp_db=-13.5)
    score.add_note("melody", start=46.0, duration=1.20, partial=8.0, amp_db=-16.5)
    # gap: ~46.2-50.5 s — pad (9+12) blooms alone into the silence

    # Phrase 6: the 11 again, now held longer — this time unmistakably strange
    score.add_note("melody", start=50.5, duration=0.52, partial=10.0, amp_db=-12.5)
    score.add_note("melody", start=51.2, duration=0.58, partial=12.0, amp_db=-12.0)
    score.add_note("melody", start=52.0, duration=1.90, partial=11.0, amp_db=-11.5)  # ← lingers
    score.add_note("melody", start=54.1, duration=0.60, partial=10.0, amp_db=-13.5)
    score.add_note("melody", start=55.0, duration=1.80, partial=9.0, amp_db=-15.0)

    # ---------------------------------------------------------------------------
    # B section — alien territory. The familiar notes are still here but
    # the strange ones are now equal citizens.
    # ---------------------------------------------------------------------------

    # Phrase 7: climbs through familiar into 13 — alien minor sixth.
    # Delayed to 61 s so the 10+14 pad (started at 57) gets 4 s to bloom alone —
    # that alien chord colour heard without melody is the moment of no return.
    score.add_note("melody", start=61.0, duration=0.48, partial=8.0, amp_db=-12.5)
    score.add_note("melody", start=61.6, duration=0.55, partial=10.0, amp_db=-12.0)
    score.add_note("melody", start=62.3, duration=0.55, partial=12.0, amp_db=-11.5)
    score.add_note("melody", start=63.0, duration=1.40, partial=13.0, amp_db=-11.0)  # alien peak
    score.add_note("melody", start=64.6, duration=0.58, partial=12.0, amp_db=-12.5)
    score.add_note("melody", start=65.3, duration=0.72, partial=11.0, amp_db=-13.0)  # alien descent
    score.add_note("melody", start=66.2, duration=2.00, partial=10.0, amp_db=-14.5)

    # Phrase 8: septimal descent — 14 as the high point, falling home
    # The G♭75 (partial 14) sounds alien but, by now, almost inevitable
    score.add_note("melody", start=69.5, duration=1.50, partial=14.0, amp_db=-11.0)  # G♭75
    score.add_note("melody", start=71.2, duration=0.60, partial=12.0, amp_db=-12.0)
    score.add_note("melody", start=72.0, duration=0.60, partial=10.0, amp_db=-12.5)
    score.add_note("melody", start=72.8, duration=0.58, partial=9.0, amp_db=-13.5)
    score.add_note("melody", start=73.6, duration=2.80, partial=8.0, amp_db=-15.0)

    # ---------------------------------------------------------------------------
    # Alien flicker — the dream at its most vivid
    # ---------------------------------------------------------------------------

    # Phrase 9: quick, darting motion through alien territory. Not chaotic —
    # the alien world revealing it has its own texture and logic.
    # 11→10→12→13→12→11→14 [held]
    score.add_note("melody", start=76.5, duration=0.42, partial=11.0, amp_db=-11.5)
    score.add_note("melody", start=77.1, duration=0.45, partial=10.0, amp_db=-12.0)
    score.add_note("melody", start=77.7, duration=0.48, partial=12.0, amp_db=-11.5)
    score.add_note("melody", start=78.3, duration=0.70, partial=13.0, amp_db=-11.0)  # alien peak
    score.add_note("melody", start=79.1, duration=0.48, partial=12.0, amp_db=-12.0)
    score.add_note("melody", start=79.7, duration=0.50, partial=11.0, amp_db=-12.5)
    score.add_note("melody", start=80.4, duration=3.20, partial=14.0, amp_db=-11.0)  # G♭75 held

    # ---------------------------------------------------------------------------
    # Climax — partial 16, reached through the alien
    # ---------------------------------------------------------------------------

    # Phrase 10: the ascent. Goes 9→10→12→14→15→16.
    # Partial 14 (alien) leads directly into partial 15 (leading tone) into
    # partial 16 (A5, pure octave) — the highest, clearest note in the piece.
    # You arrive at home by going through the strange.
    score.add_note("melody", start=85.5, duration=0.48, partial=9.0, amp_db=-12.0)
    score.add_note("melody", start=86.1, duration=0.55, partial=10.0, amp_db=-11.5)
    score.add_note("melody", start=86.8, duration=0.62, partial=12.0, amp_db=-11.0)
    score.add_note("melody", start=87.6, duration=0.88, partial=14.0, amp_db=-10.5)  # alien gateway
    score.add_note("melody", start=88.6, duration=0.80, partial=15.0, amp_db=-10.0)  # leading tone
    score.add_note("melody", start=89.6, duration=5.50, partial=16.0, amp_db=-9.5,   # PEAK — A5
                   pitch_motion=PitchMotionSpec.vibrato(depth_ratio=0.003, rate_hz=4.2))
    score.add_note("melody", start=89.6, duration=4.80, partial=14.0, amp_db=-13.5) # G♭75 — spice

    # ---------------------------------------------------------------------------
    # Post-climax descent — integrating the whole terrain
    # ---------------------------------------------------------------------------

    # Phrase 11: 16→15→14→12→11→10→9→8. Slow, deliberate. The alien notes
    # are just notes now — neither wrong nor special. The 11 gets extra space.
    score.add_note("melody", start=97.0, duration=0.72, partial=15.0, amp_db=-11.5)
    score.add_note("melody", start=97.9, duration=1.10, partial=14.0, amp_db=-11.0)
    score.add_note("melody", start=99.2, duration=0.65, partial=12.0, amp_db=-12.0)
    score.add_note("melody", start=100.1, duration=1.80, partial=11.0, amp_db=-12.5) # alien, accepted
    score.add_note("melody", start=102.1, duration=0.62, partial=10.0, amp_db=-13.5)
    score.add_note("melody", start=102.9, duration=0.58, partial=9.0, amp_db=-14.0)
    score.add_note("melody", start=103.7, duration=5.50, partial=8.0, amp_db=-14.5)  # home, long

    # The bass partial 1 (A1, 55 Hz) rises underneath the final note —
    # the true fundamental, heard for the first time, grounding everything.

    return score


PIECES: dict[str, PieceDefinition] = {
    "septimal_lullaby": PieceDefinition(
        name="septimal_lullaby",
        output_name="mellow_01_septimal_lullaby",
        build_score=build_septimal_lullaby,
    ),
    "comma_rain": PieceDefinition(
        name="comma_rain",
        output_name="mellow_02_comma_rain",
        build_score=build_comma_rain,
    ),
    "harmonic_arpeggios": PieceDefinition(
        name="harmonic_arpeggios",
        output_name="mellow_03_harmonic_arpeggios",
        build_score=build_harmonic_arpeggios,
    ),
    "utonal_elegy": PieceDefinition(
        name="utonal_elegy",
        output_name="mellow_04_utonal_elegy",
        build_score=build_utonal_elegy,
    ),
    "fond": PieceDefinition(
        name="fond",
        output_name="fond",
        build_score=build_fond,
    ),
    "ether": PieceDefinition(
        name="ether",
        output_name="ether",
        build_score=build_ether,
    ),
}
