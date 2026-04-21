"""Beating Light — interference patterns as compositional material.

Two additive pad voices play the same JI chord (1, 5/4, 3/2, 7/4) but voice B
is microtonally offset by a few cents on each interval. The beating between
matched partials creates slow polyrhythmic amplitude envelopes. Over the course
of the piece the detuning amounts evolve, shifting the beat texture from simple
to complex and back toward stillness.

A very quiet melody traces through the beating texture in the upper harmonics.
"""

from __future__ import annotations

from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score
from code_musics.tuning import cents_to_ratio


def build_beating_light() -> Score:
    """Build the Beating Light score."""
    # ── constants ──────────────────────────────────────────────────────
    f0 = 110.0

    # JI chord intervals (partials relative to f0)
    chord_partials = [1.0, 5 / 4, 3 / 2, 7 / 4]

    # Voice B cent offsets per chord tone — chosen so beat rates form an
    # interesting polyrhythm.  At f0_hz=110 Hz, a 2-cent offset on the root
    # gives ~0.127 Hz beating (~7.9s period).  Higher partials beat faster
    # because the absolute Hz offset scales with frequency.
    #
    # Phase 1 (0-18s):  gentle, simple beating
    # Phase 2 (18-38s): wider offsets, complex polyrhythmic texture
    # Phase 3 (38-55s): offsets narrow back toward near-unison

    # We'll create three overlapping chord layers per voice to achieve
    # the evolving detuning: each layer covers ~20s with crossfade via
    # attack/release overlap.

    score = Score(
        f0_hz=f0,
        master_effects=[
            EffectSpec("delay", {"delay_seconds": 0.75, "feedback": 0.18, "mix": 0.12}),
            EffectSpec(
                "saturation", {"preset": "tube_warm", "mix": 0.15, "drive": 1.08}
            ),
            EffectSpec(
                "reverb", {"room_size": 0.82, "damping": 0.45, "wet_level": 0.28}
            ),
        ],
    )

    # ── shared additive pad synth ──────────────────────────────────────
    pad_synth = {
        "engine": "additive",
        "harmonic_rolloff": 0.42,
        "n_harmonics": 6,
        "attack": 4.0,
        "decay": 0.3,
        "sustain_level": 0.85,
        "release": 6.0,
    }

    # Voice A — exact JI, panned slightly left
    score.add_voice(
        "pad_ji",
        synth_defaults=pad_synth,
        pan=-0.2,
        velocity_humanize=None,
    )
    # Voice B — detuned, panned slightly right
    score.add_voice(
        "pad_detuned",
        synth_defaults=pad_synth,
        pan=0.2,
        velocity_humanize=None,
    )

    # ── detuning schedule ──────────────────────────────────────────────
    # Each phase: (start, duration, cents_offsets_per_chord_tone)
    phases: list[tuple[float, float, list[float]]] = [
        # Phase 1: gentle beating, simple ratios
        (0.0, 22.0, [2.0, 3.0, 1.5, 4.0]),
        # Phase 2: wider, more complex — beat rates multiply
        (16.0, 24.0, [3.5, 5.5, 2.5, 7.0]),
        # Phase 3: narrowing back — approaching unison
        (34.0, 21.0, [1.0, 1.5, 0.8, 2.0]),
    ]

    # ── lay down chord tones ──────────────────────────────────────────
    for phase_start, phase_dur, cent_offsets in phases:
        for partial, cents in zip(chord_partials, cent_offsets, strict=True):
            # Voice A: exact JI partial
            score.add_note(
                "pad_ji",
                start=phase_start,
                duration=phase_dur,
                partial=partial,
                amp_db=-6.0,
            )
            # Voice B: detuned partial
            detuned_partial = partial * cents_to_ratio(cents)
            score.add_note(
                "pad_detuned",
                start=phase_start,
                duration=phase_dur,
                partial=detuned_partial,
                amp_db=-6.0,
            )

    # ── quiet melody voice ─────────────────────────────────────────────
    melody_synth = {
        "engine": "additive",
        "harmonic_rolloff": 0.25,
        "n_harmonics": 4,
        "attack": 0.8,
        "decay": 0.4,
        "sustain_level": 0.5,
        "release": 2.5,
    }
    score.add_voice(
        "melody",
        synth_defaults=melody_synth,
        pan=0.0,
        mix_db=-6.0,
        effects=[
            EffectSpec("delay", {"delay_seconds": 1.1, "feedback": 0.25, "mix": 0.2}),
        ],
    )

    # Melody notes — sparse, haunting, drawn from the harmonic series above
    # the chord.  Each note lives in a gap between beat peaks, so the melody
    # emerges and recedes with the interference texture.
    melody_events: list[tuple[float, float, float, float]] = [
        # (start, dur, partial, amp_db)
        (5.0, 4.0, 3.0, -10.0),  # octave + fifth
        (12.0, 3.5, 7 / 2, -12.0),  # septimal region
        (18.0, 5.0, 2.0, -9.0),  # octave
        (26.0, 4.0, 5 / 2, -11.0),  # major tenth
        (33.0, 3.0, 3.0, -10.0),  # fifth above octave
        (39.0, 5.5, 7 / 4, -8.0),  # harmonic seventh — settles into chord
        (47.0, 6.0, 2.0, -9.0),  # final octave, long fade
    ]
    for start, dur, partial, amp_db in melody_events:
        score.add_note(
            "melody",
            start=start,
            duration=dur,
            partial=partial,
            amp_db=amp_db,
        )

    return score


PIECES: dict[str, PieceDefinition] = {
    "beating_light": PieceDefinition(
        name="beating_light",
        output_name="beating_light.wav",
        build_score=build_beating_light,
        sections=(
            PieceSection("gentle_beating", 0.0, 18.0),
            PieceSection("complex_polyrhythm", 18.0, 38.0),
            PieceSection("resolution", 38.0, 55.0),
        ),
    ),
}
