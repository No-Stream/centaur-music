"""Harmonic-series arpeggio: arch-form composition through JI scale degrees.

Arch form A B C B′ A′ coda — each pass re-walks the same 4:5:6:7 harmonic
material at a different tempo, texture, and direction:

  Intro  — drone alone; the harmonic world opens before the arp enters
  A      — statement: single arp voice, ascending A→B→C#→D→E  (×1.2, slow)
  B      — phase canon: two arp voices whose delay drifts from half-bar echo
             toward unison by bar 12, then arp2 pulls ahead as leader;
             + sustained 7th-partial melody thread; full round-trip A→…→E→…→A
             (×1.0, medium)
  C      — peak: both voices fast on the E chord  (×0.75), fixed half-bar
             offset; then sudden drop to drone alone — the emotional hinge
  B′     — descending canon with mirror arp pattern, E→D→C#→B→A  (×1.1)
  A′     — augmented return: single voice, ascending A→E very slowly  (×1.8)
  Coda   — sustained A major triad  (4:5:6 — the septimal 7th is gone),
             drone below; resolve and fade

Phase arc in section B: the canon delay starts at half a bar (echo) and
decreases by a fixed amount each bar.  Around bar 12 the two voices reach
unison — maximum density, the patterns lock together — then arp2 slips ahead
and becomes the leader while arp becomes the echo.  This is the piano-phase
gesture embedded in the middle of the piece.

Drone unfolds harmonically across the piece:
  partial 2 (A2 = 110 Hz) — from the intro, throughout
  partial 3 (E2 = 165 Hz) — enters at section B, enriching the bass fifth

The 7th partial of each chord root traces a slow high melody in sections B/C/B′:
  G♭4 (A root) → A♭4 (B) → ~B♭4 (C#) → C♭5 (D) → D♭5 (E)

JI roots relative to base = A1 = 55 Hz:
  A  = 55 Hz        B  = 9/8 × 55 = 61.875 Hz
  C# = 5/4 × 55 = 68.75 Hz    D  = 4/3 × 55 ≈ 73.33 Hz
  E  = 3/2 × 55 = 82.5 Hz
"""

from __future__ import annotations

import logging

from code_musics.composition import RhythmCell, line
from code_musics.pieces.registry import PieceDefinition
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import EffectSpec, Phrase, Score

logger = logging.getLogger(__name__)

# ── Canonical timing ──────────────────────────────────────────────────────────
_NOTE_DUR: float = 0.16        # canonical note duration; time_scale stretches this
_N_PER_BAR: int = 8
_BAR_DUR: float = _NOTE_DUR * _N_PER_BAR   # 1.28 s per bar at ×1.0

# ── Arp patterns — indices into the 4:5:6:7 chord [4r, 5r, 6r, 7r] ──────────
_ARP_ASC: list[int] = [0, 1, 2, 3, 1, 2, 3, 2]   # rising, returns through middle
_ARP_DESC: list[int] = [3, 2, 1, 0, 2, 1, 0, 1]  # descending mirror
_ARP_PEAK: list[int] = [0, 3, 2, 3, 1, 3, 2, 1]  # 7th-heavy, agitated


def _arp(
    freqs: list[float],
    amp_db: float = -10.0,
    pattern: list[int] | None = None,
) -> Phrase:
    _pat = pattern if pattern is not None else _ARP_ASC
    return line(
        tones=[freqs[i] for i in _pat],
        rhythm=RhythmCell(spans=tuple([_NOTE_DUR] * _N_PER_BAR)),
        pitch_kind="freq",
        amp_db=amp_db,
    )


def _pulse_drone(
    score: Score,
    partial_val: float,
    start: float,
    end: float,
    periods: tuple[float, ...],
    note_dur: float,
    amp_db: float,
    vibrato_rate_hz: float = 0.0,
    vibrato_depth: float = 0.0,
) -> None:
    """Place overlapping drone pulses from start to end.

    Consecutive notes overlap heavily so the amplitude breathes rather than
    ticking.  Optional very-slow vibrato gives the layer a live, slightly
    unstable pitch texture; when notes overlap the slight phase offset between
    their vibrato cycles produces gentle beating — a slow chorus on the bass.
    """
    t = start
    i = 0
    while t < end:
        pm: PitchMotionSpec | None = None
        if vibrato_rate_hz > 0 and vibrato_depth > 0:
            # Stagger phase across pulses so overlapping notes breathe independently
            pm = PitchMotionSpec.vibrato(
                depth_ratio=vibrato_depth,
                rate_hz=vibrato_rate_hz,
                phase=i * 0.7,   # irrational-ish step avoids synchronisation
            )
        score.add_note(
            "drone",
            start=t,
            duration=note_dur,
            partial=partial_val,
            amp_db=amp_db,
            pitch_motion=pm,
        )
        t += periods[i % len(periods)]
        i += 1


def _place_section(
    score: Score,
    progression: list[tuple[float, int, str]],
    amp_arc: list[float],
    start: float,
    time_scale: float,
    arp_voice: str = "arp",
    canon_voice: str | None = None,
    canon_delay: float = 0.0,
    canon_drift: float = 0.0,
    melody_voice: str | None = None,
    arp_pattern: list[int] | None = None,
) -> float:
    """Place one arch-form section bar by bar. Returns updated time cursor.

    canon_drift: seconds to subtract from canon_delay each bar.  A positive
    value causes the follower to catch up to the leader, cross into unison,
    and then slip ahead.  This is the piano-phase mechanism.
    """
    bar_dur = _BAR_DUR * time_scale
    t = start
    current_delay = canon_delay
    for (root, n_bars, label), amp_db in zip(progression, amp_arc):
        chord = [root * 4, root * 5, root * 6, root * 7]
        phrase = _arp(chord, amp_db=amp_db, pattern=arp_pattern)
        chord_dur = n_bars * bar_dur
        logger.info("%-24s root=%.2f Hz  t=%.1f s", label, root, t)

        for bar in range(n_bars):
            bar_start = t + bar * bar_dur
            score.add_phrase(arp_voice, phrase, start=bar_start, time_scale=time_scale)
            if canon_voice is not None:
                score.add_phrase(
                    canon_voice,
                    phrase,
                    start=bar_start + current_delay,
                    time_scale=time_scale,
                    amp_scale=0.70,
                )
                current_delay -= canon_drift

        # Melody: one sustained note per chord — the 7th partial as a slow thread
        if melody_voice:
            score.add_note(
                melody_voice,
                start=t,
                duration=chord_dur,
                freq=root * 7,
                amp_db=amp_db - 7,
            )

        t += chord_dur
    return t


def build_natural_steps_score() -> Score:
    """Arch-form 4:5:6:7 JI arpeggio: A B C B′ A′ coda."""
    base = 55.0   # A1 — all partials are integer multiples of this

    score = Score(
        f0=base,
        master_effects=[
            EffectSpec("reverb", {"room_size": 0.72, "damping": 0.44, "wet_level": 0.28}),
        ],
    )

    # ── Voices ────────────────────────────────────────────────────────────────

    # Main arp — slightly right of centre
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
        pan=0.18,
        velocity_humanize=None,
    )

    # Phase follower — opposite side, softer, slightly slower attack to
    # distinguish its timbre from arp as they drift in and out of phase
    score.add_voice(
        "arp2",
        synth_defaults={
            "engine": "additive",
            "params": {"n_harmonics": 5, "harmonic_rolloff": 0.46},
            "env": {
                "attack_ms": 22.0,
                "decay_ms": 260.0,
                "sustain_ratio": 0.28,
                "release_ms": 440.0,
            },
        },
        mix_db=-3.0,
        pan=-0.28,
        velocity_humanize=None,
    )

    # Melody — slow pad tracing the 7th-partial thread
    score.add_voice(
        "melody",
        synth_defaults={
            "engine": "additive",
            "params": {"n_harmonics": 3, "harmonic_rolloff": 0.68},
            "env": {
                "attack_ms": 1400.0,
                "decay_ms": 700.0,
                "sustain_ratio": 0.80,
                "release_ms": 2400.0,
            },
        },
        mix_db=-5.0,
        pan=0.08,
        velocity_humanize=None,
    )

    # Drone — unfolds from A2 alone to A2+E2 at section B
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

    # ── JI roots ──────────────────────────────────────────────────────────────
    A  = base
    B  = base * 9 / 8
    Cs = base * 5 / 4
    D  = base * 4 / 3
    E  = base * 3 / 2

    # ── Time scales (×canonical bar) ──────────────────────────────────────────
    TS_A  = 1.20   # statement: slow, deliberate
    TS_B  = 1.00   # canon: medium
    TS_C  = 0.75   # peak: driven, fast
    TS_BD = 1.10   # descending canon: medium-slow
    TS_AA = 1.80   # augmented return: very slow

    # Canon delays: half a bar in each section's own time
    CANON_B  = _BAR_DUR * TS_B  / 2   # 0.64 s
    CANON_C  = _BAR_DUR * TS_C  / 2   # 0.48 s
    CANON_BD = _BAR_DUR * TS_BD / 2   # 0.70 s

    # Phase drift in section B: delay decreases by this amount each bar.
    # Section B has 18 bars; starting at CANON_B=0.64 s, the delay crosses
    # zero around bar 12 → unison → arp2 leads for bars 13-18.
    _B_BARS = 18
    _B_UNISON_BAR = 12
    PHASE_DRIFT_B = CANON_B / _B_UNISON_BAR   # ≈ 0.053 s / bar

    # ── Progressions ──────────────────────────────────────────────────────────

    # A — ascending statement (single voice, no canon yet)
    prog_A = [
        (A,  4, "A  [statement]"),
        (B,  3, "B  [statement]"),
        (Cs, 2, "C# [statement]"),
        (D,  3, "D  [statement]"),
        (E,  3, "E  [statement]"),
    ]
    amp_A = [-13.0, -12.5, -12.0, -11.5, -10.5]

    # B — full round-trip with drifting phase canon + melody thread
    prog_B = [
        (A,  3, "A  [canon]"),
        (B,  2, "B  [canon]"),
        (Cs, 2, "C# [canon]"),
        (D,  3, "D  [canon]"),
        (E,  3, "E  [canon]"),
        (D,  2, "D  [canon ↓]"),
        (A,  3, "A  [canon ↓]"),
    ]
    amp_B = [-11.0, -10.5, -10.0, -9.5, -9.0, -10.0, -11.0]
    assert sum(n for _, n, _ in prog_B) == _B_BARS

    # C — peak: E chord only, both voices at full speed
    prog_C = [(E, 4, "E  [peak]")]
    amp_C  = [-8.0]

    # B′ — descent, descending arp pattern, melody thread reappears
    prog_BD = [
        (E,  2, "E  [descent]"),
        (D,  2, "D  [descent]"),
        (Cs, 2, "C# [descent]"),
        (B,  2, "B  [descent]"),
        (A,  3, "A  [descent]"),
    ]
    amp_BD = [-10.5, -11.0, -11.5, -12.0, -12.5]

    # A′ — augmented return (single voice, much slower)
    prog_AA = [
        (A,  2, "A  [augmented]"),
        (B,  2, "B  [augmented]"),
        (Cs, 1, "C# [augmented]"),
        (D,  2, "D  [augmented]"),
        (E,  2, "E  [augmented]"),
    ]
    amp_AA = [-13.0, -12.5, -12.0, -11.5, -11.0]

    # ── Compose ───────────────────────────────────────────────────────────────

    INTRO_DUR   = 3.5
    SILENCE_DUR = 3.5   # drone-alone hinge between C and B′
    CODA_DUR    = 9.0

    t = INTRO_DUR

    # A — statement
    t = _place_section(
        score, prog_A, amp_A, t, TS_A,
        arp_voice="arp",
    )

    t_B = t  # section B start — drone E2 fifth enters here

    # B — drifting phase canon + melody
    t = _place_section(
        score, prog_B, amp_B, t, TS_B,
        arp_voice="arp",
        canon_voice="arp2",
        canon_delay=CANON_B,
        canon_drift=PHASE_DRIFT_B,
        melody_voice="melody",
    )

    # C — peak (fixed canon, no drift — energy over subtlety)
    t_peak = t
    t = _place_section(
        score, prog_C, amp_C, t, TS_C,
        arp_voice="arp",
        canon_voice="arp2",
        canon_delay=CANON_C,
        arp_pattern=_ARP_PEAK,
    )
    # Melody: hold E's 7th through the peak only; let silence breathe alone
    score.add_note(
        "melody",
        start=t_peak,
        duration=4 * _BAR_DUR * TS_C,
        freq=E * 7,
        amp_db=-8.0,
    )

    t_silence = t          # silence begins here
    t += SILENCE_DUR

    t_BD = t  # section B′ start — A3 octave enters here

    # B′ — descending canon + melody
    t = _place_section(
        score, prog_BD, amp_BD, t, TS_BD,
        arp_voice="arp",
        canon_voice="arp2",
        canon_delay=CANON_BD,
        melody_voice="melody",
        arp_pattern=_ARP_DESC,
    )

    t_AA = t  # section A′ start — drone thins back out

    # A′ — augmented return, single voice
    t = _place_section(
        score, prog_AA, amp_AA, t, TS_AA,
        arp_voice="arp",
    )

    t_coda = t

    # Coda — A major triad (4:5:6); septimal 7th absent for the first time
    for mult in [4, 5, 6]:
        score.add_note("arp", start=t_coda, duration=CODA_DUR, freq=base * mult, amp_db=-12.0)

    total_dur = t_coda + CODA_DUR + 2.0

    # ── Drone — layered, breathing, harmonically unfolding ────────────────────
    #
    # The drone has an arc that mirrors the piece's structural shape:
    #
    #   Intro / A  : A2 alone, gently pulsing
    #   B          : A2 + E2 fifth enters (bass opens up)
    #   Silence    : A2 + E2 + one long glide from A2→E2 (drone's featured moment)
    #   B′         : A2 + E2 + A3 octave + E3 high fifth (fullest harmonic stack)
    #   A′         : A2 + E2 (thins back, mirrors B)
    #   Coda       : A2 + E2 + sub A1 deepens the resolution
    #
    # Each layer uses overlapping pulses so amplitude breathes rather than
    # sustaining flat.  The A2 and E2 layers carry subtle slow vibrato whose
    # phase is staggered across pulses, producing gentle beating between
    # simultaneously sounding notes — a slow drift-chorus on the bass.

    # Layer 1 — A2 (partial 2 = 110 Hz), throughout
    _pulse_drone(
        score, 2.0,
        start=0.0, end=total_dur,
        periods=(7.5, 9.5, 8.0, 10.0),
        note_dur=18.0, amp_db=-12.0,
        vibrato_rate_hz=0.08, vibrato_depth=0.003,
    )

    # Layer 2 — E2 (partial 3 = 165 Hz), enters at section B
    _pulse_drone(
        score, 3.0,
        start=t_B, end=total_dur,
        periods=(11.5, 13.0, 12.0),
        note_dur=24.0, amp_db=-16.0,
        vibrato_rate_hz=0.06, vibrato_depth=0.0025,
    )

    # Featured glide during the silence: A2 rises to E2 as the drone speaks alone
    score.add_note(
        "drone",
        start=t_silence,
        duration=SILENCE_DUR + 3.0,   # fades into the opening of B′
        partial=2.0,
        amp_db=-10.5,
        pitch_motion=PitchMotionSpec.ratio_glide(start_ratio=1.0, end_ratio=3 / 2),
    )

    # Layer 3 — A3 (partial 4 = 220 Hz), enters at peak, stops at A′
    _pulse_drone(
        score, 4.0,
        start=t_peak, end=t_AA,
        periods=(9.0, 11.5),
        note_dur=18.0, amp_db=-21.0,
    )

    # Layer 4 — E3 (partial 6 = 330 Hz), enters at B′ descent, stops at A′
    _pulse_drone(
        score, 6.0,
        start=t_BD, end=t_AA,
        periods=(13.5, 16.0),
        note_dur=26.0, amp_db=-25.0,
    )

    # Sub-bass A1 (partial 1 = 55 Hz), enters at coda for extra depth on resolution
    score.add_note(
        "drone",
        start=t_coda,
        duration=CODA_DUR + 1.0,
        partial=1.0,
        amp_db=-18.0,
    )

    return score


PIECES: dict[str, PieceDefinition] = {
    "natural_steps": PieceDefinition(
        name="natural_steps",
        output_name="29_natural_steps.wav",
        build_score=build_natural_steps_score,
    ),
}
