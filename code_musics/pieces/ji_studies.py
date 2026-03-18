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
from dataclasses import replace

from code_musics.composition import (
    ContextSection,
    ContextSectionSpec,
    HarmonicContext,
    build_context_sections,
    line,
    place_ratio_chord,
    place_ratio_line,
    with_synth_ramp,
)
from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    VelocityHumanizeSpec,
)
from code_musics.pieces.septimal import PieceDefinition
from code_musics.score import EffectSpec, Score, VelocityParamMap

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
    A2 = f0  # 110.00
    B2 = f0 * 9 / 8  # 123.75
    D3 = f0 * 4 / 3  # 146.67  iv root
    E3 = f0 * 3 / 2  # 165.00
    F3 = D3 * 6 / 5  # 176.00  D-minor 3rd (outside A major)
    Fs2 = f0 * 5 / 6  # 91.67   sub-bass F#2
    Fs3 = f0 * 5 / 3  # 183.33  vi root
    Gs3 = E3 * 5 / 4  # 206.25  V-chord 3rd (G#3)
    A3 = f0 * 2  # 220.00
    B3 = f0 * 9 / 4  # 247.50
    Cs4 = f0 * 5 / 2  # 275.00
    C4 = D3 * 9 / 5  # 264.00  pure min-7 above D
    D4 = D3 * 2  # 293.33
    E4 = f0 * 3  # 330.00
    F4 = F3 * 2  # 352.00
    Fs4 = Fs3 * 2  # 366.67
    Gs4 = f0 * 15 / 4  # 412.50  A-maj7
    A4 = f0 * 4  # 440.00
    B4 = f0 * 9 / 2  # 495.00
    Cs5 = f0 * 5  # 550.00
    D5 = D4 * 2  # 586.67
    A5 = A4 * 2  # 880.00  — tonic, next octave

    score = Score(
        f0=f0,
        # Chamber ensemble looseness: ~18 ms shared drift, 16 ms chord spread so
        # block chords don't fire as a single MIDI click.
        timing_humanize=TimingHumanizeSpec(preset="chamber", chord_spread_ms=16.0),
        master_effects=[
            # Neve-style mix glue before the spatial effects; drive bumped slightly
            # so it's actually doing something measurable at this level.
            EffectSpec(
                "saturation", {"preset": "neve_gentle", "mix": 0.22, "drive": 1.16}
            ),
            # Subtle tape warmth — glues the mix with light nonlinearity.
            EffectSpec(
                "chow_tape",
                {"drive": 0.15, "saturation": 0.18, "bias": 0.5, "mix": 50.0},
            ),
            EffectSpec("delay", {"delay_seconds": 0.28, "feedback": 0.16, "mix": 0.10}),
            # Bricasti "Large & Dark" hall — warm, long, dark character; outputs stereo
            EffectSpec("bricasti", {"ir_name": "1 Halls 07 Large & Dark", "wet": 0.30}),
        ],
    )

    # Bass: filtered_stack square — warm dark foundation; velocity_group "harmony"
    # so bass/tenor/alto breathe together as one ensemble unit.
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "filtered_stack",
            "waveform": "square",
            "n_harmonics": 8,
            "cutoff_hz": 460.0,
            "keytrack": 0.1,
            "resonance": 0.10,
            "filter_env_amount": 0.55,
            "filter_env_decay": 0.70,
            "attack": 0.22,
            "decay": 0.18,
            "sustain_level": 0.60,
            "release": 0.90,
        },
        pan=-0.08,
        velocity_group="harmony",
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble"),
        velocity_db_per_unit=8.0,
        # Louder chord arrivals open the filter slightly — more energy, more body.
        velocity_to_params={
            "cutoff_hz": VelocityParamMap(min_value=340.0, max_value=580.0)
        },
    )
    # Tenor/alto: additive pad — the harmonic bed; same group as bass.
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
    score.add_voice(
        "tenor",
        synth_defaults=dict(chord_defaults),
        pan=-0.16,
        velocity_group="harmony",
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble"),
        velocity_db_per_unit=8.0,
        # Additive engine: louder arrivals get slightly brighter harmonic tilt.
        velocity_to_params={
            "brightness_tilt": VelocityParamMap(min_value=-0.02, max_value=0.06)
        },
    )
    score.add_voice(
        "alto",
        synth_defaults=dict(chord_defaults),
        pan=0.14,
        velocity_group="harmony",
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble"),
        velocity_db_per_unit=8.0,
        velocity_to_params={
            "brightness_tilt": VelocityParamMap(min_value=-0.02, max_value=0.06)
        },
    )
    # Counter: filtered_stack reed — oboe-like inner voice; "melody" group with lead.
    score.add_voice(
        "counter",
        synth_defaults={
            "engine": "filtered_stack",
            "preset": "reed_lead",
            "cutoff_hz": 2_400.0,
            "keytrack": 0.1,
            "resonance": 0.10,
            "filter_env_amount": 0.95,
            "attack": 0.03,
            "decay": 0.12,
            "sustain_level": 0.52,
            "release": 0.30,
        },
        pan=0.08,
        velocity_group="melody",
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        velocity_db_per_unit=10.0,
        # Louder notes open the filter and increase envelope sweep — oboe-like expressivity.
        velocity_to_params={
            "cutoff_hz": VelocityParamMap(min_value=1_800.0, max_value=3_000.0),
            "filter_env_amount": VelocityParamMap(min_value=0.72, max_value=1.18),
        },
    )
    # Lead: filtered saw — the expressive soprano line; same "melody" group as counter.
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "filtered_stack",
            "waveform": "saw",
            "n_harmonics": 18,
            "cutoff_hz": 3_000.0,
            "keytrack": 0.05,
            "resonance": 0.10,
            "filter_env_amount": 0.55,
            "filter_env_decay": 0.90,
            "attack": 0.085,
            "decay": 1.25,
            "sustain_level": 0.48,
            "release": 0.32,
        },
        effects=[
            # TAL-Chorus-LX: authentic Juno-60 BBD chorus; mode I for subtle width.
            EffectSpec(
                "tal_chorus_lx", {"mix": 0.22, "chorus_1": True, "chorus_2": False}
            ),
            # Dragonfly plate: intimate pre-reverb before the master Bricasti hall —
            # gives the soprano line a "singer in a room" quality.
            EffectSpec(
                "dragonfly",
                {
                    "variant": "plate",
                    "wet_level": 16.0,
                    "decay_s": 0.55,
                    "low_cut_hz": 180.0,
                    "high_cut_hz": 14000.0,
                    "predelay_ms": 8.0,
                },
            ),
        ],
        pan=0.20,
        velocity_group="melody",
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        # Accented notes open the filter and add resonance — louder = brighter = more edge.
        # filter_env_amount replaces the ramp's value at note level; ramp only handles cutoff_hz.
        velocity_to_params={
            "filter_env_amount": VelocityParamMap(min_value=0.30, max_value=0.75),
            "resonance": VelocityParamMap(min_value=0.04, max_value=0.12),
        },
    )

    # ── Prologue (0–12 s): sparse Fs2+A3 drone with rhythmic pickup ─────────
    # Bass: three articulations build into the A section — sub-bass pulses in.
    score.add_note("bass", start=0.0, duration=3.2, freq=Fs2, amp=0.17, velocity=0.72)
    score.add_note("bass", start=3.8, duration=2.6, freq=Fs2, amp=0.20, velocity=0.78)
    score.add_note("bass", start=6.8, duration=5.2, freq=Fs2, amp=0.22, velocity=0.82)
    # Tenor: "la  laaa  la  la  la  laaaa" figure — simple F#m pickup before bar 1.
    score.add_note("tenor", start=2.0, duration=0.9, freq=A3, amp=0.12, velocity=0.72)
    score.add_note("tenor", start=3.5, duration=1.4, freq=Cs4, amp=0.13, velocity=0.78)
    score.add_note("tenor", start=5.5, duration=0.7, freq=B3, amp=0.12, velocity=0.73)
    score.add_note("tenor", start=6.5, duration=0.6, freq=A3, amp=0.11, velocity=0.70)
    score.add_note("tenor", start=7.4, duration=0.5, freq=Cs4, amp=0.11, velocity=0.72)
    score.add_note("tenor", start=8.2, duration=3.8, freq=A3, amp=0.14, velocity=0.80)

    # ── A section (12–54 s): 7-bar vi–iv alternation ────────────────────────
    # Velocity builds through bar 5 (the A5 climax) then eases back.
    a_note_dur = 6.9
    a_chords: list[tuple[float, float, float, float]] = [
        (12.0, Fs3, A3, Cs4),  # vi  F# minor  bar 1
        (18.0, D3, F3, A3),  # iv  D minor   bar 2
        (24.0, Fs3, A3, Cs4),  # vi             bar 3  (counter enters)
        (30.0, D3, F3, A3),  # iv             bar 4
        (36.0, Fs3, A3, Cs4),  # vi             bar 5  — lead climax here
        (42.0, D3, F3, A3),  # iv             bar 6
        (48.0, Fs3, A3, Cs4),  # vi             bar 7
    ]
    a_velocities = [0.90, 0.95, 1.02, 1.05, 1.12, 0.98, 0.90]
    for (start, b, t, a), vel in zip(a_chords, a_velocities, strict=True):
        score.add_note(
            "bass", start=start, duration=a_note_dur, freq=b, amp=0.25, velocity=vel
        )
        score.add_note(
            "tenor", start=start, duration=a_note_dur, freq=t, amp=0.21, velocity=vel
        )
        score.add_note(
            "alto", start=start, duration=a_note_dur, freq=a, amp=0.18, velocity=vel
        )

    # ── B section (54–75 s): I–V–I in A major ───────────────────────────────
    b_note_dur = 7.2
    b_chords: list[tuple[float, float, float, float]] = [
        (54.0, A2, E3, Cs4),  # I   A major
        (61.0, E3, Gs3, B3),  # V   E major (pure 5-limit)
        (68.0, A2, E3, Cs4),  # I   A major
    ]
    b_velocities = [1.05, 1.10, 0.95]
    for (start, b, t, a), vel in zip(b_chords, b_velocities, strict=True):
        score.add_note(
            "bass", start=start, duration=b_note_dur, freq=b, amp=0.23, velocity=vel
        )
        score.add_note(
            "tenor", start=start, duration=b_note_dur, freq=t, amp=0.20, velocity=vel
        )
        score.add_note(
            "alto", start=start, duration=b_note_dur, freq=a, amp=0.17, velocity=vel
        )

    # ── Development (75–99 s): F#m7 → Bm → V ───────────────────────────────
    # Velocity builds through the section — the most harmonically tense passage.
    dev_note_dur = 8.1
    dev_chords: list[tuple[float, float, float, float]] = [
        (75.0, Fs2, A3, Cs4),  # F#m7 (counter plays E4 = 7th)
        (83.0, B2, D3, Fs3),  # Bm
        (91.0, E3, Gs3, B3),  # V — leads back to A
    ]
    dev_velocities = [1.02, 1.08, 1.14]
    for (start, b, t, a), vel in zip(dev_chords, dev_velocities, strict=True):
        score.add_note(
            "bass", start=start, duration=dev_note_dur, freq=b, amp=0.24, velocity=vel
        )
        score.add_note(
            "tenor", start=start, duration=dev_note_dur, freq=t, amp=0.20, velocity=vel
        )
        score.add_note(
            "alto", start=start, duration=dev_note_dur, freq=a, amp=0.18, velocity=vel
        )

    # ── Reprise (99–123 s): vi–I–vi–I ───────────────────────────────────────
    rep_note_dur = 6.8
    rep_chords: list[tuple[float, float, float, float]] = [
        (99.0, Fs3, A3, Cs4),  # vi
        (105.0, A2, E3, Cs4),  # I
        (111.0, Fs3, A3, Cs4),  # vi
        (117.0, A2, E3, Cs4),  # I
    ]
    rep_velocities = [1.00, 0.95, 1.00, 0.88]  # gentle descent into ending
    for (start, b, t, a), vel in zip(rep_chords, rep_velocities, strict=True):
        score.add_note(
            "bass", start=start, duration=rep_note_dur, freq=b, amp=0.24, velocity=vel
        )
        score.add_note(
            "tenor", start=start, duration=rep_note_dur, freq=t, amp=0.20, velocity=vel
        )
        score.add_note(
            "alto", start=start, duration=rep_note_dur, freq=a, amp=0.18, velocity=vel
        )

    # ── Ending (123–150 s): wide vi → Dm7 → Amaj7 ───────────────────────────
    # Wide vi — F#2 sub-bass, A3, C#5 across 3 octaves
    score.add_note("bass", start=123.0, duration=8.4, freq=Fs2, amp=0.27, velocity=1.05)
    score.add_note("tenor", start=123.0, duration=8.4, freq=A3, amp=0.20, velocity=1.05)
    score.add_note("alto", start=123.0, duration=8.4, freq=Cs5, amp=0.21, velocity=1.05)

    # Dm7 — D / F / A / C4 (264 Hz — pure minor 7th, the spice)
    score.add_note("bass", start=131.0, duration=8.4, freq=D3, amp=0.26, velocity=1.00)
    score.add_note("tenor", start=131.0, duration=8.4, freq=F3, amp=0.21, velocity=1.00)
    score.add_note("alto", start=131.0, duration=8.4, freq=A3, amp=0.18, velocity=1.00)
    score.add_note("alto", start=131.0, duration=8.4, freq=C4, amp=0.16, velocity=1.00)

    # Amaj7 — A2 / E3 / C#4 / G#4 (pure 15/8 above A) — unresolved, fading
    score.add_note("bass", start=139.0, duration=10.0, freq=A2, amp=0.23, velocity=0.88)
    score.add_note(
        "tenor", start=139.0, duration=10.0, freq=E3, amp=0.19, velocity=0.88
    )
    score.add_note(
        "alto", start=139.0, duration=10.0, freq=Cs4, amp=0.17, velocity=0.88
    )
    score.add_note(
        "alto", start=139.0, duration=10.0, freq=Gs4, amp=0.17, velocity=0.88
    )

    # ── Counter voice (enters A section bar 3 = t=24) ────────────────────────
    # Moves in contrary motion to the lead; fills the inner voice above alto.
    # notes: (freq, dur, velocity)
    def _add_counter(t_start: float, notes: list[tuple[float, float, float]]) -> None:
        t = t_start
        for freq, dur, vel in notes:
            score.add_note(
                "counter",
                start=t,
                duration=dur * 1.02,
                freq=freq,
                amp=0.22,
                velocity=vel,
            )
            t += dur

    # A-section bars 3–7 (t=24–54)
    _add_counter(24.0, [(E4, 2.5, 1.00), (D4, 2.0, 0.93), (Cs4, 1.5, 0.85)])  # bar 3
    _add_counter(
        30.0, [(F4, 2.0, 1.05), (E4, 2.0, 0.97), (D4, 2.0, 0.88)]
    )  # bar 4 D-minor
    _add_counter(
        36.0, [(E4, 1.5, 1.02), (Cs4, 2.0, 0.92), (E4, 2.5, 1.00)]
    )  # bar 5 arching
    _add_counter(
        42.0, [(D4, 2.0, 0.92), (E4, 2.5, 1.00), (F4, 1.5, 1.08)]
    )  # bar 6 rising
    _add_counter(48.0, [(Cs4, 2.0, 1.02), (E4, 2.5, 1.05), (D4, 1.5, 0.90)])  # bar 7

    # Development (t=75–99)
    _add_counter(
        75.0, [(E4, 3.0, 1.05), (Cs4, 2.5, 0.92), (E4, 2.5, 1.00)]
    )  # F#m7 — E4 = 7th
    _add_counter(83.0, [(Fs4, 3.0, 1.10), (E4, 2.5, 1.00), (D4, 2.5, 0.90)])  # Bm
    _add_counter(
        91.0, [(Gs4, 3.0, 1.18), (Fs4, 2.0, 1.02), (E4, 3.0, 0.88)]
    )  # V — Gs4 is peak

    # Reprise vi chords only (I chords rest — open space)
    _add_counter(99.0, [(E4, 3.0, 1.08), (Cs4, 3.0, 0.90)])
    _add_counter(111.0, [(Cs4, 2.5, 0.95), (E4, 3.5, 1.05)])

    # Ending
    _add_counter(123.0, [(E4, 4.0, 1.00), (Cs4, 4.0, 0.90)])  # wide vi
    _add_counter(131.0, [(F4, 4.0, 1.08), (E4, 4.0, 0.95)])  # Dm7 — F4 color
    _add_counter(139.0, [(E4, 4.0, 1.00), (Gs4, 4.0, 1.10), (E4, 3.0, 0.85)])  # Amaj7

    # ── Soprano lead — continuous from prologue pickup (t=8) ─────────────────
    # Per-note velocity shapes the melodic arc: phrase peaks accented, tails tapered,
    # the A5 climax (bar 5) is the loudest single moment in the piece.
    def _add_lead_phrase(
        *,
        start: float,
        notes: list[tuple[float, float]],
        synth_start: dict[str, float],
        synth_end: dict[str, float],
        amp_db: float,
        velocities: list[float],
    ) -> None:
        phrase = line(
            tones=[freq for freq, _ in notes],
            rhythm=[dur for _, dur in notes],
            pitch_kind="freq",
            amp_db=amp_db,
            synth_defaults={"engine": "filtered_stack", "waveform": "saw"},
        )
        phrase = with_synth_ramp(phrase, start=synth_start, end=synth_end)
        # Stamp per-note velocities onto the frozen NoteEvent instances.
        phrase = replace(
            phrase,
            events=tuple(
                replace(evt, velocity=vel)
                for evt, vel in zip(phrase.events, velocities, strict=True)
            ),
        )
        score.add_phrase("lead", phrase, start=start)

    lead_prologue_and_a: list[tuple[float, float]] = [
        # Prologue pickup (8–12 s): tentative entry, builds to Cs5
        (A4, 1.0),
        (Cs5, 1.0),
        (B4, 1.0),
        (A4, 1.0),
        # A bar 1 vi (12–18 s): Cs5 is the phrase peak
        (Cs5, 2.0),
        (B4, 1.0),
        (A4, 2.0),
        (Gs4, 1.0),
        # A bar 2 iv (18–24 s): F4 is the dark D-minor color note
        (A4, 1.5),
        (F4, 2.0),
        (A4, 1.5),
        (Gs4, 0.5),
        (A4, 0.5),
        # A bar 3 vi (24–30 s): counter enters; Cs5 again peaks
        (B4, 1.0),
        (Cs5, 2.0),
        (B4, 1.0),
        (A4, 1.5),
        (B4, 0.5),
        # A bar 4 iv (30–36 s): F4 recurs, D4 is the iv root
        (A4, 1.0),
        (F4, 1.5),
        (E4, 1.0),
        (D4, 1.5),
        (E4, 1.0),
        # A bar 5 vi (36–42 s): most ornate; A5 is the climactic peak of the whole section
        (A4, 0.5),
        (Cs5, 0.75),
        (B4, 0.5),
        (Cs5, 0.75),
        (B4, 0.5),
        (A4, 0.5),
        (Gs4, 0.5),
        (A4, 0.5),
        (Cs5, 0.75),
        (D5, 0.375),
        (A5, 0.375),
        # A bar 6 iv (42–48 s): descent from the A5 peak
        (A4, 1.5),
        (F4, 1.0),
        (E4, 1.5),
        (D4, 1.5),
        (E4, 0.5),
        # A bar 7 vi (48–54 s): winding down to cadence
        (Cs5, 2.5),
        (B4, 1.5),
        (A4, 1.5),
        (Gs4, 0.5),
    ]
    prologue_a_velocities: list[float] = [
        # Prologue pickup — soft, tentative
        0.85,
        1.12,
        1.00,
        0.88,
        # A bar 1 — Cs5 peaks
        1.22,
        1.02,
        0.95,
        0.82,
        # A bar 2 — F4 is the dark color note
        1.05,
        1.15,
        1.02,
        0.88,
        0.85,
        # A bar 3 — Cs5 peaks again, counter entering
        1.02,
        1.25,
        1.05,
        0.95,
        0.85,
        # A bar 4 — F4 color recurs, D4 is the iv root
        1.05,
        1.12,
        1.00,
        0.90,
        0.85,
        # A bar 5 — climactic run to A5; each step builds
        1.00,
        1.10,
        1.02,
        1.15,
        1.05,
        1.00,
        0.95,
        1.05,
        1.20,
        1.32,
        1.42,
        # A bar 6 — coming down from the A5 peak
        1.08,
        0.95,
        0.90,
        0.85,
        0.80,
        # A bar 7 — winding down, phrase tail
        1.02,
        0.92,
        0.85,
        0.75,
    ]

    lead_b_and_development: list[tuple[float, float]] = [
        # B section I (54–61 s): bright new section; E4 low entry, Cs5 peaks
        (E4, 1.5),
        (Cs5, 1.5),
        (A4, 2.0),
        (E4, 2.0),
        # B section V (61–68 s): harmonic tension; B4 is the leading-tone peak
        (B4, 2.0),
        (Gs4, 1.5),
        (Fs4, 1.5),
        (E4, 2.0),
        # B section I (68–75 s): resolution; settle quietly
        (Cs5, 2.5),
        (A4, 2.0),
        (E4, 2.5),
        # Development F#m7 (75–83 s): exploratory; Cs5 glows amid the ambiguity
        (A4, 1.5),
        (E4, 1.5),
        (Cs5, 1.5),
        (Fs4, 1.5),
        (A4, 2.0),
        # Development Bm (83–91 s): climbing; B4 is the section peak
        (D4, 1.0),
        (Fs4, 1.5),
        (A4, 2.0),
        (B4, 1.5),
        (A4, 2.0),
        # Development V (91–99 s): leading-tone push; B4 peaks, falls to E4
        (Gs4, 2.0),
        (B4, 2.0),
        (Gs4, 1.5),
        (E4, 2.5),
    ]
    b_dev_velocities: list[float] = [
        # B I — E4 modest entry, Cs5 peaks
        0.98,
        1.25,
        1.02,
        0.88,
        # B V — B4 leading-tone tension
        1.15,
        1.05,
        0.95,
        0.85,
        # B I return — settled, soft close
        1.12,
        0.97,
        0.82,
        # Dev F#m7 — Cs5 glows through the ambiguity
        1.00,
        0.92,
        1.12,
        0.92,
        0.98,
        # Dev Bm — climbing to B4
        0.88,
        1.00,
        1.10,
        1.22,
        1.02,
        # Dev V — B4 peaks, resolve falls to E4
        1.12,
        1.20,
        1.02,
        0.85,
    ]

    lead_reprise_and_ending: list[tuple[float, float]] = [
        # Reprise vi (99–105 s): echo of A section
        (Cs5, 2.0),
        (B4, 1.5),
        (A4, 2.0),
        (Gs4, 0.5),
        # Reprise I (105–111 s): tastefully major, clear
        (A4, 1.5),
        (Cs5, 2.0),
        (E4, 2.5),
        # Reprise vi (111–117 s): Cs5 variant
        (B4, 1.5),
        (Cs5, 1.5),
        (B4, 1.5),
        (A4, 1.5),
        # Reprise I (117–123 s): bridge to ending
        (Cs5, 2.5),
        (A4, 2.0),
        (E4, 1.5),
        # Ending wide vi (123–131 s): spacious, high Cs5 opens the final space
        (Cs5, 2.5),
        (B4, 1.0),
        (A4, 1.0),
        (Gs4, 0.5),
        (A4, 1.0),
        (B4, 2.0),
        # Ending Dm7 (131–139 s): C4 = spicy pure minor 7th — the emotional peak here
        (A4, 1.5),
        (F4, 1.0),
        (C4, 2.0),
        (D4, 1.0),
        (F4, 1.5),
        (A4, 1.0),
        # Ending Amaj7 (139–150 s): arc to Cs5, settle on a very quiet final A4
        (Gs4, 2.0),
        (A4, 1.5),
        (Cs5, 2.5),
        (B4, 1.5),
        (A4, 3.5),
    ]
    reprise_ending_velocities: list[float] = [
        # Reprise vi — echo of A section, slightly softer
        1.15,
        1.00,
        0.90,
        0.75,
        # Reprise I — clear, bright, modest
        0.95,
        1.10,
        0.85,
        # Reprise vi variant
        1.05,
        1.18,
        1.05,
        0.90,
        # Reprise I — bridge, heading down
        1.12,
        0.98,
        0.85,
        # Ending wide vi — spacious, Cs5 prominent
        1.22,
        1.02,
        0.92,
        0.80,
        0.85,
        0.98,
        # Ending Dm7 — C4 is the spicy moment, F4 colors it
        1.05,
        1.12,
        1.22,
        1.05,
        1.00,
        0.90,
        # Ending Amaj7 — arc to Cs5, very quiet final note
        1.00,
        0.95,
        1.08,
        0.90,
        0.78,
    ]

    _add_lead_phrase(
        start=8.0,
        notes=lead_prologue_and_a,
        synth_start={"cutoff_hz": 2_800.0, "release": 0.30},
        synth_end={"cutoff_hz": 3_100.0, "release": 0.28},
        amp_db=-15.0,
        velocities=prologue_a_velocities,
    )
    _add_lead_phrase(
        start=54.0,
        notes=lead_b_and_development,
        synth_start={"cutoff_hz": 3_150.0, "release": 0.26},
        synth_end={"cutoff_hz": 3_300.0, "release": 0.24},
        amp_db=-14.5,
        velocities=b_dev_velocities,
    )
    _add_lead_phrase(
        start=99.0,
        notes=lead_reprise_and_ending,
        synth_start={"cutoff_hz": 3_000.0, "release": 0.26},
        synth_end={"cutoff_hz": 2_650.0, "release": 0.34},
        amp_db=-15.0,
        velocities=reprise_ending_velocities,
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

    A3 = f0 * 1
    B3 = f0 * 9 / 8
    Cs4 = f0 * 5 / 4
    D4 = f0 * 4 / 3
    E4 = f0 * 3 / 2
    Fs4 = f0 * 5 / 3
    Gs4 = f0 * 15 / 8
    A4 = f0 * 2
    B4 = f0 * 9 / 4
    Cs5 = f0 * 5 / 2

    score = Score(
        f0=bass_f0,
        master_effects=[
            EffectSpec(
                "reverb", {"room_size": 0.60, "damping": 0.50, "wet_level": 0.24}
            ),
            EffectSpec("delay", {"delay_seconds": 0.34, "feedback": 0.16, "mix": 0.10}),
        ],
    )
    score.add_voice(
        "bass",
        synth_defaults={
            "harmonic_rolloff": 0.50,
            "n_harmonics": 8,
            "attack": 1.0,
            "decay": 0.4,
            "sustain_level": 0.80,
            "release": 3.0,
        },
    )
    score.add_voice(
        "bass_fifth",
        synth_defaults={
            "harmonic_rolloff": 0.42,
            "n_harmonics": 6,
            "attack": 1.2,
            "decay": 0.3,
            "sustain_level": 0.72,
            "release": 3.0,
        },
    )
    score.add_voice(
        "melody",
        synth_defaults={
            "harmonic_rolloff": 0.28,
            "n_harmonics": 5,
            "attack": 0.04,
            "decay": 0.12,
            "sustain_level": 0.74,
            "release": 0.45,
        },
    )

    melody_notes: list[tuple[float, float]] = [
        (A4, 1.50),
        (Gs4, 0.75),
        (Fs4, 0.75),
        (E4, 1.125),
        (D4, 0.375),
        (Cs4, 0.75),
        (B3, 0.75),
        (A3, 1.50),
        (B3, 0.375),
        (Cs4, 0.375),
        (D4, 3.00),
        (E4, 0.75),
        (Fs4, 0.75),
        (Gs4, 0.75),
        (A4, 1.50),
        (B4, 0.75),
        (A4, 0.75),
        (Gs4, 1.50),
        (Fs4, 0.75),
        (E4, 3.00),
        (Cs5, 1.50),
        (B4, 0.375),
        (A4, 0.375),
        (Gs4, 0.75),
        (A4, 0.375),
        (Gs4, 0.375),
        (Fs4, 1.50),
        (E4, 0.75),
        (D4, 0.75),
        (Cs4, 0.75),
        (B3, 0.375),
        (A3, 4.50),
    ]

    t = 0.0
    for freq, dur in melody_notes:
        score.add_note("melody", start=t, duration=dur, freq=freq, amp=0.42)
        t += dur

    score.add_note("bass", start=0.0, duration=t + 1.0, freq=bass_f0, amp=0.32)
    score.add_note(
        "bass_fifth", start=0.0, duration=t + 1.0, freq=bass_f0 * 3 / 2, amp=0.16
    )
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
    (2.0, 0.50),
    (5 / 2, 0.75),
    (2.0, 0.50),
    (15 / 8, 0.75),
    # Over IV (2.5 s): F#4, hold A4, back to F#4
    (5 / 3, 0.50),
    (2.0, 1.00),
    (5 / 3, 1.00),
    # Over ii (2.5 s): D4 arpeggio up through the ii chord
    (4 / 3, 0.50),
    (5 / 3, 0.50),
    (2.0, 0.75),
    (5 / 3, 0.375),
    (4 / 3, 0.375),
    # Over V (2.5 s): E4 → G#4 → B4, back to G#4
    (3 / 2, 0.50),
    (15 / 8, 0.50),
    (9 / 4, 1.00),
    (15 / 8, 0.50),
]

# V2 — more ornate; now exposes the flat B3 (10/9) on the ii chord
_MELODY_V2: list[tuple[float, float]] = [
    # Over I (2.5 s): quick ascending run A4→B4→C#5 and back
    (2.0, 0.375),
    (9 / 4, 0.375),
    (5 / 2, 0.375),
    (9 / 4, 0.375),
    (2.0, 0.375),
    (15 / 8, 0.375),
    (2.0, 0.25),
    # Over IV (2.5 s): higher arc through C#5
    (5 / 3, 0.75),
    (2.0, 0.50),
    (5 / 2, 0.50),
    (2.0, 0.25),
    (5 / 3, 0.50),
    # Over ii (2.5 s): opens with flat B3 (10/9) — the comma moment is audible
    (10 / 9, 0.50),
    (4 / 3, 0.375),
    (5 / 3, 0.375),
    (2.0, 0.50),
    (5 / 3, 0.375),
    (4 / 3, 0.375),
    # Over V (2.5 s): peak on C#5 then step down
    (9 / 4, 1.00),
    (5 / 2, 0.75),
    (9 / 4, 0.375),
    (15 / 8, 0.375),
]

# V3 — climax variant: starts at C#5, descends during V
_MELODY_V3: list[tuple[float, float]] = [
    # Over I (2.5 s): high start on C#5, descend
    (5 / 2, 1.00),
    (9 / 4, 0.50),
    (2.0, 0.50),
    (15 / 8, 0.50),
    # Over IV (2.5 s): F#4 with passing notes through A4
    (5 / 3, 0.50),
    (2.0, 0.50),
    (5 / 3, 0.375),
    (3 / 2, 0.375),
    (5 / 3, 0.75),
    # Over ii (2.5 s): D arpeggio climbing (avoid dwelling on flat B)
    (4 / 3, 0.75),
    (5 / 3, 0.50),
    (2.0, 0.75),
    (5 / 3, 0.50),
    # Over V (2.5 s): settle through G#4 to E4 — arch coming down from climax
    (15 / 8, 0.50),
    (9 / 4, 0.75),
    (15 / 8, 0.50),
    (3 / 2, 0.75),
]

# Snap-back melody — 3 s at pure drone_freq (landing)
_MELODY_SNAP: list[tuple[float, float]] = [
    (2.0, 1.00),
    (5 / 2, 0.75),
    (9 / 4, 0.50),
    (2.0, 0.75),
]

# Long snap-back — 5 s for the final block
_MELODY_SNAP_LONG: list[tuple[float, float]] = [
    (2.0, 1.25),
    (15 / 8, 0.50),
    (2.0, 0.75),
    (9 / 4, 0.75),
    (5 / 2, 0.75),
    (9 / 4, 0.50),
    (2.0, 0.50),
]


def _place_melody(
    score: Score,
    section: ContextSection,
    notes: list[tuple[float, float]],
    amp: float = 0.38,
) -> None:
    place_ratio_line(
        score,
        "melody",
        section=section,
        tones=[ratio for ratio, _ in notes],
        rhythm=[dur for _, dur in notes],
        amp=amp,
    )


def _place_chord_cycle(
    score: Score,
    sections: tuple[ContextSection, ...],
    chord_dur: float,
) -> None:
    """One I–IV–ii–V cycle tuned purely from tonic f_c."""
    note_dur = chord_dur + 0.35
    bass_dur = chord_dur * 0.84
    arp_gap = 0.08

    cycle: list[tuple[float, list[tuple[float, float]]]] = [
        (1 / 2, [(1.0, 0.20), (5 / 4, 0.16), (3 / 2, 0.13)]),  # I
        (4 / 3 / 2, [(4 / 3, 0.18), (5 / 3, 0.14), (2.0, 0.11)]),  # IV
        (10 / 9 / 2, [(10 / 9, 0.17), (4 / 3, 0.13), (5 / 3, 0.11)]),  # ii
        (3 / 2 / 2, [(3 / 2, 0.20), (15 / 8, 0.15), (9 / 4, 0.13)]),  # V
    ]
    for section, (bass_ratio, harmony_notes) in zip(sections, cycle, strict=True):
        place_ratio_chord(
            score,
            "bass",
            section=section,
            ratios=[bass_ratio],
            duration=bass_dur,
            amp=0.26,
        )
        place_ratio_chord(
            score,
            "chord",
            section=section,
            ratios=[ratio for ratio, _ in harmony_notes],
            duration=note_dur,
            amp=[amp for _, amp in harmony_notes],
            gap=arp_gap,
        )


def _place_snap(
    score: Score,
    section: ContextSection,
    melody: list[tuple[float, float]],
) -> None:
    """Sustained I chord at drone_freq — the snap-back moment."""
    note_dur = section.duration - 0.4
    place_ratio_chord(
        score,
        "bass",
        section=section,
        ratios=[1 / 2],
        duration=note_dur,
        amp=0.30,
    )
    place_ratio_chord(
        score,
        "chord",
        section=section,
        ratios=[1.0, 5 / 4, 3 / 2],
        duration=note_dur,
        amp=[0.22, 0.18, 0.15],
    )
    _place_melody(score, section, melody, amp=0.42)


def build_ji_comma_drift_score() -> Score:
    """Syntonic comma pump with an anthemic drifting melody.

    Structure: intro → [1 drift cycle + snap] × 3 → clean coda.

    One drift cycle = 21.5 ¢ flat.  The snap-back is a sustained I chord at
    the true A=220, giving the ear a clear correction before the next drift.
    Melody variants escalate in ornament across the three blocks before
    the coda restates V1 cleanly over a pure A=220 cycle.
    """
    drone_freq = 220.0
    chord_dur = 2.5
    cycle_dur = 4 * chord_dur  # 10 s per I-IV-ii-V loop
    comma = 81 / 80

    intro_dur = 4.0
    snap_dur = 3.0
    snap_long_dur = 5.0
    n_blocks = 3

    total_dur = (
        intro_dur
        + (n_blocks - 1) * (cycle_dur + snap_dur)
        + cycle_dur
        + snap_long_dur  # last block has long snap
        + cycle_dur  # coda
        + 3.0  # reverb tail
    )

    score = Score(
        f0=drone_freq,
        master_effects=[
            EffectSpec("delay", {"delay_seconds": 0.26, "feedback": 0.14, "mix": 0.09}),
            EffectSpec(
                "reverb", {"room_size": 0.68, "damping": 0.44, "wet_level": 0.24}
            ),
        ],
    )

    score.add_voice(
        "drone",
        synth_defaults={
            "harmonic_rolloff": 0.48,
            "n_harmonics": 8,
            "attack": 1.8,
            "decay": 0.4,
            "sustain_level": 0.82,
            "release": 3.5,
        },
    )
    score.add_voice(
        "chord",
        synth_defaults={
            "harmonic_rolloff": 0.34,
            "n_harmonics": 6,
            "attack": 0.35,
            "decay": 0.20,
            "sustain_level": 0.70,
            "release": 0.90,
        },
    )
    score.add_voice(
        "bass",
        synth_defaults={
            "harmonic_rolloff": 0.50,
            "n_harmonics": 8,
            "attack": 0.14,
            "decay": 0.20,
            "sustain_level": 0.76,
            "release": 0.70,
        },
    )
    score.add_voice(
        "melody",
        synth_defaults={
            "harmonic_rolloff": 0.24,
            "n_harmonics": 5,
            "attack": 0.05,
            "decay": 0.12,
            "sustain_level": 0.72,
            "release": 0.45,
        },
    )

    score.add_note(
        "drone",
        start=0.0,
        duration=total_dur,
        freq=drone_freq,
        amp=0.22,
        label="A=220 drone",
    )

    melody_variants = [_MELODY_V1, _MELODY_V2, _MELODY_V3]
    snap_melodies = [_MELODY_SNAP, _MELODY_SNAP, _MELODY_SNAP_LONG]
    snap_durations = [snap_dur, snap_dur, snap_long_dur]

    t = intro_dur
    for block_idx in range(n_blocks):
        # One drift cycle per block (21.5 ¢ flat)
        f_c = drone_freq * (1 / comma)
        logger.info(
            "block %d  tonic=%.2f Hz  (%.2f ¢ from drone)",
            block_idx,
            f_c,
            1200 * math.log2(f_c / drone_freq),
        )
        drift_sections = build_context_sections(
            base_tonic=drone_freq,
            start=t,
            specs=(
                ContextSectionSpec(name="I", duration=chord_dur, tonic_ratio=1 / comma),
                ContextSectionSpec(
                    name="IV", duration=chord_dur, tonic_ratio=1 / comma
                ),
                ContextSectionSpec(
                    name="ii", duration=chord_dur, tonic_ratio=1 / comma
                ),
                ContextSectionSpec(name="V", duration=chord_dur, tonic_ratio=1 / comma),
            ),
        )
        _place_chord_cycle(score, drift_sections, chord_dur)
        _place_melody(
            score,
            ContextSection(
                start=t,
                duration=cycle_dur,
                context=HarmonicContext(tonic=f_c, name=f"block_{block_idx}_drift"),
            ),
            melody_variants[block_idx],
        )
        t += cycle_dur

        _place_snap(
            score,
            ContextSection(
                start=t,
                duration=snap_durations[block_idx],
                context=HarmonicContext(
                    tonic=drone_freq, name=f"block_{block_idx}_snap"
                ),
            ),
            snap_melodies[block_idx],
        )
        t += snap_durations[block_idx]

    # Clean coda — pure A=220, V1 melody restated
    coda_sections = build_context_sections(
        base_tonic=drone_freq,
        start=t,
        specs=(
            ContextSectionSpec(name="I", duration=chord_dur),
            ContextSectionSpec(name="IV", duration=chord_dur),
            ContextSectionSpec(name="ii", duration=chord_dur),
            ContextSectionSpec(name="V", duration=chord_dur),
        ),
    )
    _place_chord_cycle(score, coda_sections, chord_dur)
    _place_melody(
        score,
        ContextSection(
            start=t,
            duration=cycle_dur,
            context=HarmonicContext(tonic=drone_freq, name="coda"),
        ),
        _MELODY_V1,
        amp=0.40,
    )

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
