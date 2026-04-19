"""diva_study — showcase for the Diva-inspired subtractive overhaul.

Seven short sections, demonstrating different facets of the 2026
Diva-inspired subtractive overhaul:

  1. Cascade pad — `filter_topology="cascade"`, smooth Prophet-5-style
     4-pole with peaking bandpass resonance (no global tanh growl).
  2. Sallen-Key lead — `filter_topology="sallen_key"`, CEM-3320-ish bite
     with pre-filter asymmetric soft-clip.
  3. Newton-ladder acid bass — `filter_topology="ladder"`,
     `filter_solver="newton"`, implicit per-sample solve of the delay-free
     feedback loop at high resonance.
  4. Divine-quality self-oscillation — `quality="divine"` on a ladder with
     q=40, 4x internal oversampling and 8 Newton iterations; clean sine-like
     self-oscillating lead tone.
  5. Quality A/B — the same preset rendered back-to-back at `quality="draft"`
     then `quality="great"`, so the listener can hear the solver difference.
  6. Hard sync screamer — `osc2_sync=True` with a fifth-up osc2 detune,
     producing the classic biting "sync lead" formant-chase timbre.
  7. Ring modulation bell-lead — `osc2_ring_mod=0.6` with a major-third
     osc2 detune, producing metallic sum/difference sidebands over a
     cascade filter.

Key in F# minor (370.00 Hz tonic → 185.00 Hz f0) to stay in the kick-friendly
range per AGENTS.md, but no kick on this piece — it's a subtractive voice
showcase, not a beats piece.
"""

from __future__ import annotations

from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score

SECTION_DUR = 40.0


def _reverb() -> EffectSpec:
    return EffectSpec(
        "reverb",
        {"room_size": 0.65, "damping": 0.35, "wet_level": 0.2},
    )


def _delay(
    delay_seconds: float = 0.38, feedback: float = 0.28, mix: float = 0.22
) -> EffectSpec:
    return EffectSpec(
        "delay",
        {"delay_seconds": delay_seconds, "feedback": feedback, "mix": mix},
    )


def build_diva_study() -> Score:
    f0 = 185.00  # F#3

    score = Score(f0_hz=f0, master_effects=DEFAULT_MASTER_EFFECTS)

    # ---- Voices ------------------------------------------------------------
    score.add_voice(
        "cascade_pad",
        synth_defaults={
            "engine": "polyblep",
            "preset": "prophet_pad",
        },
        effects=[_reverb()],
        pan=-0.15,
    )

    score.add_voice(
        "sk_lead",
        synth_defaults={
            "engine": "polyblep",
            "preset": "sk_bite_lead",
        },
        effects=[_delay(0.42, 0.32, 0.25), _reverb()],
        pan=0.18,
    )

    score.add_voice(
        "newton_bass",
        synth_defaults={
            "engine": "polyblep",
            "preset": "moog_acid_newton",
        },
        effects=[_reverb()],
        pan=0.0,
        normalize_peak_db=-6.0,
    )

    score.add_voice(
        "divine_osc",
        synth_defaults={
            "engine": "polyblep",
            "preset": "diva_bass_resonance",
            "quality": "divine",
            "resonance_q": 40.0,  # push into self-oscillation
            "filter_drive": 0.2,
            "cutoff_hz": 320.0,
            "filter_env_amount": 0.3,
            "attack": 0.02,
            "release": 0.4,
        },
        effects=[_delay(0.25, 0.4, 0.3), _reverb()],
        pan=0.0,
    )

    score.add_voice(
        "ab_draft",
        synth_defaults={
            "engine": "polyblep",
            "preset": "moog_lead",
            "quality": "draft",
        },
        effects=[_delay(0.33, 0.3, 0.2), _reverb()],
        pan=-0.25,
    )

    score.add_voice(
        "ab_great",
        synth_defaults={
            "engine": "polyblep",
            "preset": "moog_lead",
            "quality": "great",
        },
        effects=[_delay(0.33, 0.3, 0.2), _reverb()],
        pan=0.25,
    )

    score.add_voice(
        "sync_lead_voice",
        synth_defaults={
            "engine": "polyblep",
            "preset": "sync_screamer",
        },
        effects=[_delay(0.36, 0.3, 0.22), _reverb()],
        pan=0.1,
    )

    score.add_voice(
        "ring_lead_voice",
        synth_defaults={
            "engine": "polyblep",
            "preset": "ring_mod_lead",
        },
        effects=[_delay(0.42, 0.35, 0.25), _reverb()],
        pan=-0.1,
    )

    # ---- Section 1: Cascade pad (0 – 40s) ----------------------------------
    # Spread-voiced 7-limit chord, root moving between 1, 4/3, 5/4, 3/2
    section_1 = 0.0
    chord_voicings = [
        [(1.0, 0.0), (5 / 4, 0.01), (3 / 2, 0.02), (7 / 4, 0.03)],
        [(4 / 3, 0.0), (5 / 3, 0.01), (2.0, 0.02), (7 / 3, 0.03)],
        [(9 / 8, 0.0), (45 / 32, 0.01), (27 / 16, 0.02), (15 / 8, 0.03)],
        [(3 / 2, 0.0), (15 / 8, 0.01), (9 / 4, 0.02), (21 / 8, 0.03)],
    ]
    for i, voicing in enumerate(chord_voicings):
        start = section_1 + i * 10.0
        for partial, stagger in voicing:
            score.add_note(
                "cascade_pad",
                start=start + stagger,
                duration=10.5,
                partial=partial,
                amp_db=-18.0,
            )

    # ---- Section 2: Sallen-Key lead over sparse pad (40 – 80s) -------------
    section_2 = SECTION_DUR
    # Continue pad at lower level
    for i, voicing in enumerate(chord_voicings):
        start = section_2 + i * 10.0
        for partial, stagger in voicing:
            score.add_note(
                "cascade_pad",
                start=start + stagger,
                duration=10.5,
                partial=partial,
                amp_db=-24.0,
            )

    # Lead melody in 7-limit JI
    lead_melody = [
        (0.0, 1.5, 3 / 2, -14.0),
        (1.5, 1.5, 7 / 4, -13.0),
        (3.0, 1.0, 2.0, -12.0),
        (4.0, 2.0, 7 / 4, -13.0),
        (6.0, 1.0, 3 / 2, -14.0),
        (7.0, 1.0, 5 / 4, -15.0),
        (8.0, 2.0, 4 / 3, -14.0),
        (10.0, 1.0, 5 / 4, -15.0),
        (11.0, 1.5, 3 / 2, -13.0),
        (12.5, 1.5, 7 / 4, -12.0),
        (14.0, 2.0, 9 / 4, -11.0),
        (16.0, 3.0, 2.0, -13.0),
        (20.0, 1.5, 7 / 4, -14.0),
        (21.5, 1.5, 3 / 2, -15.0),
        (23.0, 2.0, 5 / 4, -14.0),
        (25.0, 1.0, 3 / 2, -13.0),
        (26.0, 1.0, 7 / 4, -12.0),
        (27.0, 3.0, 2.0, -11.0),
        (30.0, 2.0, 9 / 4, -10.0),
        (32.0, 4.0, 2.0, -12.0),
    ]
    for t, dur, partial, db in lead_melody:
        score.add_note(
            "sk_lead",
            start=section_2 + t,
            duration=dur,
            partial=partial,
            amp_db=db,
        )

    # ---- Section 3: Newton-solved acid bass (80 – 120s) --------------------
    section_3 = 2 * SECTION_DUR
    # Tight 16th-note acid line, filter env sweeping high → low each bar
    # 8 beats/sec at 120bpm = 16ths every 0.125s
    bass_notes = [
        1.0,
        1.0,
        2.0,
        1.0,
        3 / 2,
        1.0,
        7 / 4,
        1.0,
        5 / 4,
        1.0,
        4 / 3,
        2.0,
        1.0,
        3 / 2,
        1.0,
        7 / 4,
    ]
    step = 0.125  # 16th at 120 BPM
    for bar in range(10):  # 10 bars ~= 40s
        for i, partial in enumerate(bass_notes):
            # Accent every 4th note
            db = -6.0 if i % 4 == 0 else -10.0
            score.add_note(
                "newton_bass",
                start=section_3 + bar * (len(bass_notes) * step) + i * step,
                duration=step * 0.9,
                partial=partial * 0.5,  # sub octave
                amp_db=db,
            )

    # ---- Section 4: Divine-quality self-oscillating lead (120 – 160s) ------
    section_4 = 3 * SECTION_DUR
    # Slow melodic sweep; the resonance itself sings.
    sweep_notes = [
        (0.0, 5.0, 1.0, -14.0),
        (5.0, 5.0, 3 / 2, -13.0),
        (10.0, 4.0, 2.0, -12.0),
        (14.0, 3.0, 7 / 4, -13.0),
        (17.0, 6.0, 5 / 4, -14.0),
        (23.0, 4.0, 3 / 2, -13.0),
        (27.0, 5.0, 9 / 8, -12.0),
        (32.0, 8.0, 1.0, -14.0),
    ]
    for t, dur, partial, db in sweep_notes:
        score.add_note(
            "divine_osc",
            start=section_4 + t,
            duration=dur,
            partial=partial,
            amp_db=db,
        )

    # ---- Section 5: Quality A/B (160 – 200s) -------------------------------
    # Same melody on draft (L) and great (R) — listener compares
    section_5 = 4 * SECTION_DUR
    ab_melody = [
        (0.0, 2.0, 1.0, -12.0),
        (2.0, 1.0, 5 / 4, -12.0),
        (3.0, 1.0, 3 / 2, -11.0),
        (4.0, 2.0, 7 / 4, -10.0),
        (6.0, 2.0, 2.0, -11.0),
        (8.0, 2.0, 3 / 2, -12.0),
        (10.0, 1.5, 5 / 4, -13.0),
        (11.5, 1.5, 4 / 3, -12.0),
        (13.0, 3.0, 3 / 2, -11.0),
        (16.0, 1.0, 7 / 4, -10.0),
        (17.0, 3.0, 2.0, -10.0),
        (20.0, 2.0, 9 / 4, -9.0),
        (22.0, 2.0, 5 / 2, -10.0),
        (24.0, 2.0, 2.0, -11.0),
        (26.0, 2.0, 3 / 2, -12.0),
        (28.0, 4.0, 1.0, -13.0),
    ]
    for t, dur, partial, db in ab_melody:
        # Stagger draft slightly before great so the difference is audible
        for voice in ("ab_draft", "ab_great"):
            score.add_note(
                voice,
                start=section_5 + t,
                duration=dur,
                partial=partial,
                amp_db=db,
            )

    # ---- Section 6: Hard sync screamer (200 – 224s) ------------------------
    # osc2_sync=True with an aggressive detune; classic chirpy sync lead.
    # 24s = 8 bars at 120 BPM.
    section_6 = 5 * SECTION_DUR
    # Lower pad bed for context at low level
    for i, voicing in enumerate(chord_voicings):
        start = section_6 + i * 6.0
        for partial, stagger in voicing:
            score.add_note(
                "cascade_pad",
                start=start + stagger,
                duration=6.5,
                partial=partial,
                amp_db=-30.0,
            )
    # Sync lead phrase — strong opening, escalating, with held notes that
    # let the sync formant chew through the filter envelope.
    sync_melody = [
        (0.0, 1.5, 3 / 2, -10.0),
        (1.5, 0.5, 7 / 4, -11.0),
        (2.0, 2.0, 2.0, -9.0),
        (4.0, 1.0, 7 / 4, -10.0),
        (5.0, 1.0, 3 / 2, -11.0),
        (6.0, 2.0, 5 / 4, -10.0),
        (8.0, 1.5, 4 / 3, -11.0),
        (9.5, 0.5, 3 / 2, -12.0),
        (10.0, 2.0, 7 / 4, -9.0),
        (12.0, 1.0, 2.0, -8.0),
        (13.0, 1.0, 9 / 4, -8.0),
        (14.0, 3.0, 5 / 2, -9.0),
        (17.0, 1.5, 7 / 4, -10.0),
        (18.5, 0.5, 2.0, -11.0),
        (19.0, 4.0, 3 / 2, -11.0),
    ]
    for t, dur, partial, db in sync_melody:
        score.add_note(
            "sync_lead_voice",
            start=section_6 + t,
            duration=dur,
            partial=partial,
            amp_db=db,
        )

    # ---- Section 7: Ring modulation bell-lead (224 – 248s) -----------------
    # Ring mod at 0.6 with major-third detune creates metallic bell-lead.
    # 24s of sparse, bell-like phrasing.
    SECTION_6_DUR = 24.0
    section_7 = section_6 + SECTION_6_DUR
    # Quiet pad underneath for harmonic reference.
    for i, voicing in enumerate(chord_voicings):
        start = section_7 + i * 6.0
        for partial, stagger in voicing:
            score.add_note(
                "cascade_pad",
                start=start + stagger,
                duration=6.5,
                partial=partial,
                amp_db=-28.0,
            )
    # Ring-mod bell lead — sparse, let each note ring and reveal its
    # sum/difference sidebands.
    ring_melody = [
        (0.0, 3.0, 1.0, -10.0),
        (3.0, 2.0, 5 / 4, -10.0),
        (5.0, 3.0, 3 / 2, -9.0),
        (8.0, 2.0, 7 / 4, -10.0),
        (10.0, 4.0, 2.0, -9.0),
        (14.0, 2.0, 5 / 4, -11.0),
        (16.0, 3.0, 4 / 3, -10.0),
        (19.0, 5.0, 1.0, -11.0),
    ]
    for t, dur, partial, db in ring_melody:
        score.add_note(
            "ring_lead_voice",
            start=section_7 + t,
            duration=dur,
            partial=partial,
            amp_db=db,
        )

    return score


PIECES: dict[str, PieceDefinition] = {
    "diva_study": PieceDefinition(
        name="diva_study",
        output_name="diva_01_study",
        build_score=build_diva_study,
        sections=(
            PieceSection(label="Cascade Pad", start_seconds=0.0, end_seconds=40.0),
            PieceSection(label="Sallen-Key Lead", start_seconds=40.0, end_seconds=80.0),
            PieceSection(label="Newton Acid", start_seconds=80.0, end_seconds=120.0),
            PieceSection(
                label="Divine Self-Oscillation", start_seconds=120.0, end_seconds=160.0
            ),
            PieceSection(
                label="Draft vs Great", start_seconds=160.0, end_seconds=200.0
            ),
            PieceSection(label="Sync Screamer", start_seconds=200.0, end_seconds=224.0),
            PieceSection(
                label="Ring Mod Bell Lead", start_seconds=224.0, end_seconds=248.0
            ),
        ),
        study=True,
    ),
}
