"""Breath Study — demonstration of the Brush/Flow exciter noise source.

A slow, airy three-part study built around the "Flow" rare-event S&H noise
primitive (ported from Mutable Instruments Elements).  Three voices share
the noise source at different densities to sketch the range it can cover:

* a soft breath-pad (``brush_breath``) holds a sustained 7-limit JI chord;
* a very sparse brush pulse (custom ``noise_mode="flow"`` at low density)
  answers from the other side of the stereo field;
* a denser brush shimmer (``brush_cymbal``) colors section 3 with a
  metallic / brushed-cymbal air.

Key in F major (tonal baseline 174.61 Hz) to suit ambient material.
Approximately 45 s.  Uses `SOFT_REVERB_EFFECT` and the default master chain.
"""

from __future__ import annotations

from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS, SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score


def build_breath_study() -> Score:
    """Build the Breath Study score."""
    f0 = 174.614  # F3

    score = Score(f0_hz=f0, master_effects=DEFAULT_MASTER_EFFECTS)

    # Soft breath pad — brush_breath preset.
    score.add_voice(
        "breath",
        synth_defaults={
            "engine": "additive",
            "preset": "brush_breath",
            "env": {
                "attack_ms": 2500.0,
                "decay_ms": 600.0,
                "sustain_ratio": 0.75,
                "release_ms": 3500.0,
            },
        },
        effects=[SOFT_REVERB_EFFECT],
        pan=-0.15,
    )

    # Answering exhale — direct flow mode control, very sparse events.
    score.add_voice(
        "exhale",
        synth_defaults={
            "engine": "additive",
            "partials": [
                {"ratio": 1.0, "amp": 0.22, "noise": 0.9},
                {"ratio": 3 / 2, "amp": 0.14, "noise": 0.7},
                {"ratio": 7 / 4, "amp": 0.08, "noise": 0.8},
            ],
            "noise_amount": 0.85,
            "noise_bandwidth_hz": 320.0,
            "noise_mode": "flow",
            "flow_density": 0.08,  # very sparse, exhale-like
            "env": {
                "attack_ms": 1200.0,
                "decay_ms": 800.0,
                "sustain_ratio": 0.6,
                "release_ms": 2500.0,
            },
        },
        effects=[
            EffectSpec(
                "delay",
                {"delay_seconds": 0.63, "feedback": 0.22, "mix": 0.18},
            ),
            SOFT_REVERB_EFFECT,
        ],
        pan=0.22,
    )

    # Brushed shimmer — metallic bar partials with dense flow noise.
    score.add_voice(
        "shimmer",
        synth_defaults={
            "engine": "additive",
            "preset": "brush_cymbal",
            "env": {
                "attack_ms": 800.0,
                "decay_ms": 1200.0,
                "sustain_ratio": 0.5,
                "release_ms": 1800.0,
            },
        },
        effects=[SOFT_REVERB_EFFECT],
        pan=0.05,
    )

    # ---- Section 1: breath alone (0-15s) ----
    # Sustained 1, 5/4, 3/2 chord — gentle 7-limit support.
    for partial, amp_db in [(1.0, -20.0), (5 / 4, -22.0), (3 / 2, -23.0)]:
        score.add_note(
            "breath", start=0.0, duration=15.0, partial=partial, amp_db=amp_db
        )

    # ---- Section 2: add sparse exhale answers (12-30s) ----
    score.add_note("breath", start=13.0, duration=16.0, partial=1.0, amp_db=-21.0)
    score.add_note("breath", start=13.0, duration=16.0, partial=7 / 4, amp_db=-24.0)
    score.add_note("breath", start=13.0, duration=16.0, partial=9 / 8, amp_db=-25.0)

    # Exhale gestures — each a long, sparse breath through the flow source.
    score.add_note("exhale", start=15.0, duration=6.0, partial=1.0, amp_db=-20.0)
    score.add_note("exhale", start=22.0, duration=5.0, partial=3 / 2, amp_db=-21.0)

    # ---- Section 3: brushed shimmer arrives (28-45s) ----
    score.add_note("breath", start=28.0, duration=16.0, partial=1.0, amp_db=-22.0)
    score.add_note("breath", start=28.0, duration=16.0, partial=5 / 4, amp_db=-24.0)

    # Exhale tracks under the shimmer.
    score.add_note("exhale", start=30.0, duration=8.0, partial=7 / 4, amp_db=-22.0)
    score.add_note("exhale", start=36.0, duration=7.0, partial=5 / 4, amp_db=-23.0)

    # Shimmer entries — the brush crescendo.
    score.add_note("shimmer", start=29.0, duration=10.0, partial=1.0, amp_db=-24.0)
    score.add_note("shimmer", start=33.0, duration=9.0, partial=3 / 2, amp_db=-25.0)

    return score


PIECES: dict[str, PieceDefinition] = {
    "breath_study": PieceDefinition(
        name="breath_study",
        output_name="brush_01_breath_study",
        build_score=build_breath_study,
        sections=(
            PieceSection(label="Breath", start_seconds=0.0, end_seconds=15.0),
            PieceSection(label="Answers", start_seconds=15.0, end_seconds=28.0),
            PieceSection(label="Shimmer", start_seconds=28.0, end_seconds=45.0),
        ),
        study=True,
    ),
}
