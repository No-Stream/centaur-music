"""Slow Glass — a Colundi-scale étude.

The Colundi scale (1/1, 11/10, 19/16, 4/3, 3/2, 49/30, 7/4, 2/1) is a
seven-note JI tuning with familiar anchors (perfect fourth, perfect fifth)
seen through unfamiliar intervals (undecimal neutral second, septimal sixth).

The name comes from Bob Shaw's sci-fi concept of glass so thick that light
takes years to pass through — you see the past through it.

The harmonic anchor is the 4:6:7 otonal chord (R + P5 + h7) — a septimal
triad whose beating-free purity is both warm and alien.

The melody is built on two motifs:
  α (call):  R → P5 → h7 → P5    — a leap to the fifth, step to the seventh
  β (answer): P4 → m3 → N2 → R   — descending through the alien intervals

Structure:
  A  (  0–32 s): melody alone, motifs α and β
  B  ( 32–66 s): bass + harmony enter, ghost voice interlocks, 4:6:7 arrives
  C  ( 66–96 s): full texture, 4:6:7:8 chord, voices in dialogue
  B' ( 96–116 s): thinning, motif α returns quietly
  A' (116–135 s): coda — root, seventh, root over fading 4:6:7
"""

from __future__ import annotations

from code_musics.automation import AutomationSegment, AutomationSpec, AutomationTarget
from code_musics.composition import ArticulationSpec, RhythmCell, line
from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    VelocityHumanizeSpec,
)
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score, VelocityParamMap

# ---------------------------------------------------------------------------
# Colundi scale as partials of f0 = 110 Hz
# ---------------------------------------------------------------------------

# Base octave (110–220 Hz) — used for harmony and bass
_R1 = 1.0  # root         110.0 Hz
_m3_1 = 19 / 16  # narrow m3     130.6 Hz
_P4_1 = 4 / 3  # perfect 4th   146.7 Hz
_P5_1 = 3 / 2  # perfect 5th   165.0 Hz
_h7_1 = 7 / 4  # harmonic 7th  192.5 Hz

# Melody octave (220–440 Hz)
_R = 2.0  # A3   220.0 Hz
_N2 = 11 / 5  # ~B3  242.0 Hz
_m3 = 19 / 8  # ~C4  261.3 Hz
_P4 = 8 / 3  # ~D4  293.3 Hz
_P5 = 3.0  # E4   330.0 Hz
_s6 = 49 / 15  # ~F4  359.3 Hz
_h7 = 7 / 2  # ~G4  385.0 Hz
_R8 = 4.0  # A4   440.0 Hz

# High octave — texture sparkle (varied Colundi degrees)
_R_HI = 4.0  # A4   440.0 Hz
_N2_HI = 22 / 5  # ~B4  484.0 Hz
_m3_HI = 19 / 4  # ~C5  522.5 Hz
_P5_HI = 6.0  # E5   660.0 Hz
_s6_HI = 98 / 15  # ~F5  718.7 Hz
_h7_HI = 7.0  # ~G5  770.0 Hz


def build_score() -> Score:
    """Build the Slow Glass score."""
    score = Score(
        f0_hz=110.0,
        timing_humanize=TimingHumanizeSpec(preset="chamber", seed=42),
        master_effects=[
            # Tonal shaping: lift the high end, tame sub rumble
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 35.0, "slope_db_per_oct": 12},
                        {"kind": "high_shelf", "freq_hz": 3000.0, "gain_db": 3.5},
                        {"kind": "bell", "freq_hz": 800.0, "gain_db": -1.5, "q": 0.8},
                    ]
                },
            ),
            EffectSpec(
                "reverb",
                {
                    "room_size": 0.72,
                    "damping": 0.50,
                    "wet_level": 0.22,
                },
                automation=[
                    AutomationSpec(
                        target=AutomationTarget(kind="control", name="wet_level"),
                        segments=(
                            # Swell into the climax
                            AutomationSegment(
                                start=0.0,
                                end=32.0,
                                shape="linear",
                                start_value=0.22,
                                end_value=0.22,
                            ),
                            AutomationSegment(
                                start=32.0,
                                end=66.0,
                                shape="linear",
                                start_value=0.22,
                                end_value=0.28,
                            ),
                            # Peak reverb at climax
                            AutomationSegment(
                                start=66.0,
                                end=84.0,
                                shape="linear",
                                start_value=0.28,
                                end_value=0.34,
                            ),
                            # Retreat
                            AutomationSegment(
                                start=84.0,
                                end=116.0,
                                shape="linear",
                                start_value=0.34,
                                end_value=0.25,
                            ),
                            # Coda: spacious fade
                            AutomationSegment(
                                start=116.0,
                                end=135.0,
                                shape="linear",
                                start_value=0.25,
                                end_value=0.30,
                            ),
                        ),
                    ),
                ],
            ),
            *DEFAULT_MASTER_EFFECTS,
        ],
    )

    # -- Voices ---------------------------------------------------------------

    score.add_voice(
        "melody",
        synth_defaults={
            "engine": "additive",
            "n_harmonics": 6,
            "harmonic_rolloff": 0.55,
            "brightness_tilt": 0.02,
            "attack": 0.06,
            "release": 2.2,
        },
        max_polyphony=1,
        legato=True,
        velocity_humanize=VelocityHumanizeSpec(seed=7),
        velocity_group="ensemble",
        velocity_to_params={
            "brightness_tilt": VelocityParamMap(
                min_value=-0.04, max_value=0.08, min_velocity=0.8, max_velocity=1.2
            ),
        },
        effects=[
            EffectSpec(
                "delay",
                {"delay_seconds": 0.54, "feedback": 0.22, "mix": 0.16},
                automation=[
                    AutomationSpec(
                        target=AutomationTarget(kind="control", name="mix"),
                        segments=(
                            # A: dry, intimate
                            AutomationSegment(
                                start=0.0,
                                end=32.0,
                                shape="linear",
                                start_value=0.10,
                                end_value=0.14,
                            ),
                            # B: delay opens
                            AutomationSegment(
                                start=32.0,
                                end=66.0,
                                shape="linear",
                                start_value=0.14,
                                end_value=0.22,
                            ),
                            # C: fullest delay
                            AutomationSegment(
                                start=66.0,
                                end=84.0,
                                shape="linear",
                                start_value=0.22,
                                end_value=0.28,
                            ),
                            # Retreat
                            AutomationSegment(
                                start=84.0,
                                end=116.0,
                                shape="linear",
                                start_value=0.28,
                                end_value=0.12,
                            ),
                            # Coda: intimate again
                            AutomationSegment(
                                start=116.0,
                                end=135.0,
                                shape="linear",
                                start_value=0.12,
                                end_value=0.08,
                            ),
                        ),
                    ),
                ],
            ),
            EffectSpec(
                "reverb", {"room_size": 0.55, "damping": 0.45, "wet_level": 0.18}
            ),
        ],
        automation=[
            # Brightness arc across the piece
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="brightness_tilt"),
                segments=(
                    AutomationSegment(
                        start=0.0,
                        end=32.0,
                        shape="linear",
                        start_value=0.0,
                        end_value=0.02,
                    ),
                    AutomationSegment(
                        start=32.0,
                        end=66.0,
                        shape="linear",
                        start_value=0.02,
                        end_value=0.05,
                    ),
                    AutomationSegment(
                        start=66.0,
                        end=84.0,
                        shape="linear",
                        start_value=0.05,
                        end_value=0.08,
                    ),
                    AutomationSegment(
                        start=84.0,
                        end=116.0,
                        shape="linear",
                        start_value=0.08,
                        end_value=0.0,
                    ),
                    AutomationSegment(
                        start=116.0,
                        end=135.0,
                        shape="linear",
                        start_value=0.0,
                        end_value=-0.03,
                    ),
                ),
            ),
        ],
        mix_db=-4.0,
        pan=0.0,
    )

    # Harmony voice with brightness automation across the arc
    score.add_voice(
        "harmony",
        synth_defaults={
            "engine": "additive",
            "n_harmonics": 8,
            "harmonic_rolloff": 0.50,
            "brightness_tilt": -0.08,
            "attack": 1.5,
            "release": 3.0,
        },
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad", seed=13),
        effects=[
            EffectSpec(
                "chorus",
                {"preset": "juno_subtle"},
                automation=[
                    AutomationSpec(
                        target=AutomationTarget(kind="control", name="mix"),
                        segments=(
                            AutomationSegment(
                                start=34.0,
                                end=66.0,
                                shape="linear",
                                start_value=0.18,
                                end_value=0.25,
                            ),
                            AutomationSegment(
                                start=66.0,
                                end=84.0,
                                shape="linear",
                                start_value=0.25,
                                end_value=0.30,
                            ),
                            AutomationSegment(
                                start=84.0,
                                end=130.0,
                                shape="linear",
                                start_value=0.30,
                                end_value=0.15,
                            ),
                        ),
                    ),
                ],
            ),
        ],
        pan=0.0,
        mix_db=-3.0,
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="brightness_tilt"),
                segments=(
                    # B section: dark, slowly opening
                    AutomationSegment(
                        start=34.0,
                        end=66.0,
                        shape="linear",
                        start_value=-0.08,
                        end_value=0.0,
                    ),
                    # C section: brightest at the climax peak
                    AutomationSegment(
                        start=66.0,
                        end=84.0,
                        shape="linear",
                        start_value=0.0,
                        end_value=0.04,
                    ),
                    # Retreat after climax
                    AutomationSegment(
                        start=84.0,
                        end=96.0,
                        shape="linear",
                        start_value=0.04,
                        end_value=-0.04,
                    ),
                    # B' and coda: darker than we started
                    AutomationSegment(
                        start=96.0,
                        end=130.0,
                        shape="linear",
                        start_value=-0.04,
                        end_value=-0.12,
                    ),
                ),
            ),
        ],
    )

    score.add_voice(
        "bass",
        synth_defaults={"engine": "polyblep", "preset": "sub_bass"},
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 40.0, "slope_db_per_oct": 12}
                    ]
                },
            )
        ],
        pan=0.0,
        mix_db=-3.0,
    )

    score.add_voice(
        "texture",
        synth_defaults={
            "engine": "additive",
            "n_harmonics": 5,
            "harmonic_rolloff": 0.50,
            "brightness_tilt": 0.06,
            "attack": 0.02,
            "release": 1.5,
        },
        effects=[
            EffectSpec(
                "reverb", {"room_size": 0.60, "damping": 0.35, "wet_level": 0.25}
            ),
        ],
        pan=0.0,
        mix_db=-2.0,
    )

    score.add_voice(
        "ghost",
        synth_defaults={
            "engine": "additive",
            "n_harmonics": 5,
            "harmonic_rolloff": 0.55,
            "attack": 0.12,
            "release": 2.5,
        },
        velocity_group="ensemble",
        effects=[
            EffectSpec(
                "reverb", {"room_size": 0.65, "damping": 0.50, "wet_level": 0.22}
            ),
        ],
        pan=0.22,
        mix_db=-5.0,
        automation=[
            # Pan drifts gently across the stereo field
            AutomationSpec(
                target=AutomationTarget(kind="control", name="pan"),
                segments=(
                    AutomationSegment(
                        start=0.0,
                        end=32.0,
                        shape="linear",
                        start_value=0.22,
                        end_value=0.22,
                    ),
                    AutomationSegment(
                        start=32.0,
                        end=66.0,
                        shape="linear",
                        start_value=0.22,
                        end_value=-0.15,
                    ),
                    AutomationSegment(
                        start=66.0,
                        end=96.0,
                        shape="linear",
                        start_value=-0.15,
                        end_value=0.30,
                    ),
                    AutomationSegment(
                        start=96.0,
                        end=135.0,
                        shape="linear",
                        start_value=0.30,
                        end_value=0.10,
                    ),
                ),
            ),
        ],
    )

    # ===================================================================
    # Section A (0:00–0:32): Melody alone — motif α and β
    # ===================================================================

    # α: R → P5 → h7 → P5 (accent the h7 — the emotional peak)
    phrase_alpha = line(
        tones=[_R, _P5, _h7, _P5],
        rhythm=RhythmCell(
            spans=(1.2, 1.4, 2.0, 2.5),
            gates=(0.80, 0.85, 0.90, 1.0),
        ),
        amp_db=-13.0,
        velocity=(0.90, 1.0, 1.15, 0.88),
        articulation=ArticulationSpec(accent_pattern=(0.88, 1.0, 1.15, 0.90)),
    )
    score.add_phrase("melody", phrase_alpha, start=2.0)

    # β: P4 → m3 → N2 → R (diminuendo — the answer trails away)
    phrase_beta = line(
        tones=[_P4, _m3, _N2, _R],
        rhythm=RhythmCell(
            spans=(1.0, 1.2, 1.0, 3.0),
            gates=(0.78, 0.80, 0.75, 1.0),
        ),
        amp_db=-14.0,
        velocity=(1.05, 0.95, 0.88, 0.80),
        articulation=ArticulationSpec(accent_pattern=(1.0, 0.92, 0.85, 0.78)),
    )
    score.add_phrase("melody", phrase_beta, start=11.0)

    # α extended: R → P5 → h7 → R8 → h7 → P5 (peak on R8)
    phrase_alpha_ext = line(
        tones=[_R, _P5, _h7, _R8, _h7, _P5],
        rhythm=RhythmCell(
            spans=(1.0, 1.0, 1.4, 1.6, 1.0, 2.5),
            gates=(0.78, 0.82, 0.88, 0.90, 0.82, 1.0),
        ),
        amp_db=-12.5,
        velocity=(0.88, 0.95, 1.08, 1.18, 1.0, 0.85),
        articulation=ArticulationSpec(
            accent_pattern=(0.85, 0.95, 1.08, 1.15, 1.0, 0.85)
        ),
    )
    score.add_phrase("melody", phrase_alpha_ext, start=18.0)

    # β fragment trailing off: P4 → N2 → R
    phrase_beta_frag = line(
        tones=[_P4, _N2, _R],
        rhythm=RhythmCell(
            spans=(1.2, 1.5, 3.5),
            gates=(0.78, 0.80, 1.0),
        ),
        amp_db=-15.0,
        velocity=(0.95, 0.85, 0.75),
        articulation=ArticulationSpec(accent_pattern=(0.95, 0.85, 0.75)),
    )
    score.add_phrase("melody", phrase_beta_frag, start=27.0)

    # ===================================================================
    # Section B (0:32–1:06): Bass + harmony, ghost enters
    # ===================================================================

    # Bass: sustained root, fifth joins
    score.add_note("bass", start=32.0, duration=36.0, partial=1.0, amp_db=-20.0)
    score.add_note("bass", start=35.0, duration=32.0, partial=3 / 2, amp_db=-24.0)

    # Harmony chord 1: R + P4 + P5 (open fourth — suspended, spacious)
    score.add_note("harmony", start=34.0, duration=12.0, partial=_R1, amp_db=-19.0)
    score.add_note("harmony", start=34.0, duration=12.0, partial=_P4_1, amp_db=-20.0)
    score.add_note("harmony", start=34.0, duration=12.0, partial=_P5_1, amp_db=-21.0)

    # Melody: α recontextualized (accent the h7 leap)
    phrase_b1 = line(
        tones=[_m3, _h7, _P5, _m3],
        rhythm=RhythmCell(
            spans=(0.9, 1.6, 1.2, 2.2),
            gates=(0.78, 0.88, 0.82, 1.0),
        ),
        amp_db=-12.0,
        velocity=(0.92, 1.18, 0.95, 0.85),
        articulation=ArticulationSpec(accent_pattern=(0.90, 1.15, 0.95, 0.82)),
    )
    score.add_phrase("melody", phrase_b1, start=37.0)

    # Ghost enters with β
    score.add_phrase("ghost", phrase_beta, start=43.0, amp_scale=0.85)

    # Harmony chord 2: R + P5 + h7 — the 4:6:7 chord, first appearance
    score.add_note("harmony", start=46.0, duration=12.0, partial=_R1, amp_db=-19.0)
    score.add_note("harmony", start=46.0, duration=12.0, partial=_P5_1, amp_db=-20.0)
    score.add_note("harmony", start=46.0, duration=12.0, partial=_h7_1, amp_db=-21.0)

    # Melody: reaching through alien territory (accent the h7 and s6)
    phrase_b2 = line(
        tones=[_R, _h7, _s6, _P5, _R],
        rhythm=RhythmCell(
            spans=(0.8, 1.8, 1.0, 0.9, 2.8),
            gates=(0.75, 0.90, 0.82, 0.78, 1.0),
        ),
        amp_db=-11.5,
        velocity=(0.90, 1.15, 1.08, 0.92, 0.82),
        articulation=ArticulationSpec(accent_pattern=(0.88, 1.15, 1.05, 0.90, 0.80)),
    )
    score.add_phrase("melody", phrase_b2, start=49.0)

    # Ghost: ascending answer
    phrase_ghost_rise = line(
        tones=[_P5, _h7, _R8],
        rhythm=RhythmCell(
            spans=(1.2, 1.5, 2.5),
            gates=(0.82, 0.88, 1.0),
        ),
        amp_db=-14.0,
        velocity=(0.88, 1.05, 1.12),
        articulation=ArticulationSpec(accent_pattern=(0.85, 1.0, 1.10)),
    )
    score.add_phrase("ghost", phrase_ghost_rise, start=55.0)

    # Harmony chord 3: P4 root — P4 + s6 + R' (new root, septimal color)
    score.add_note("harmony", start=58.0, duration=10.0, partial=_P4_1, amp_db=-19.0)
    score.add_note("harmony", start=58.0, duration=10.0, partial=_s6 / 2, amp_db=-20.0)
    score.add_note("harmony", start=58.0, duration=10.0, partial=_R, amp_db=-21.0)

    # Melody bridge (building crescendo)
    phrase_bridge = line(
        tones=[_P5, _h7, _P5, _R8],
        rhythm=RhythmCell(
            spans=(0.9, 1.5, 1.2, 2.8),
            gates=(0.80, 0.88, 0.82, 1.0),
        ),
        amp_db=-11.0,
        velocity=(0.88, 1.05, 0.92, 1.18),
        articulation=ArticulationSpec(accent_pattern=(0.85, 1.0, 0.90, 1.15)),
    )
    score.add_phrase("melody", phrase_bridge, start=61.0)

    # Texture through B: mixed durations and Colundi degrees
    for tick_t, tick_partial, tick_dur, tick_amp, tick_vel in [
        (40.0, _P5_HI, 0.3, -19.0, 1.05),
        (47.0, _h7_HI, 0.8, -20.0, 0.92),
        (50.5, _m3_HI, 0.2, -21.0, 1.10),
        (53.0, _R_HI, 0.3, -18.0, 1.0),
        (57.0, _s6_HI, 1.2, -20.0, 0.95),
        (59.0, _P5_HI, 0.3, -19.0, 1.08),
        (62.0, _N2_HI, 0.5, -21.0, 0.88),
        (64.0, _h7_HI, 0.3, -18.0, 1.05),
    ]:
        score.add_note(
            "texture",
            start=tick_t,
            duration=tick_dur,
            partial=tick_partial,
            amp_db=tick_amp,
            velocity=tick_vel,
        )

    # ===================================================================
    # Section C (1:06–1:36): Climax — 4:6:7:8 chord, voices in dialogue
    # ===================================================================

    # Harmony: R + P5 + h7 + R' (the full 4:6:7:8)
    score.add_note("harmony", start=66.0, duration=18.0, partial=_R1, amp_db=-18.0)
    score.add_note("harmony", start=66.0, duration=18.0, partial=_P5_1, amp_db=-19.0)
    score.add_note("harmony", start=66.0, duration=18.0, partial=_h7_1, amp_db=-20.0)
    score.add_note("harmony", start=66.0, duration=18.0, partial=_R, amp_db=-21.0)

    # Melody: leaping (peak on R8)
    phrase_c1 = line(
        tones=[_R, _P5, _h7, _R8, _P5],
        rhythm=RhythmCell(
            spans=(0.8, 1.0, 1.2, 1.8, 2.0),
            gates=(0.78, 0.82, 0.88, 0.92, 1.0),
        ),
        amp_db=-10.5,
        velocity=(0.88, 0.98, 1.10, 1.20, 0.90),
        articulation=ArticulationSpec(accent_pattern=(0.85, 0.95, 1.08, 1.18, 0.88)),
    )
    score.add_phrase("melody", phrase_c1, start=70.0)

    # Ghost interlock: contrasting descending motion
    phrase_c1_ghost = line(
        tones=[_h7, _P4, _R, _m3, _P5],
        rhythm=RhythmCell(
            spans=(0.9, 1.0, 1.2, 1.0, 2.2),
            gates=(0.85, 0.78, 0.82, 0.80, 1.0),
        ),
        amp_db=-13.0,
        velocity=(1.12, 0.92, 0.85, 0.90, 1.05),
        articulation=ArticulationSpec(accent_pattern=(1.10, 0.90, 0.82, 0.88, 1.0)),
    )
    score.add_phrase("ghost", phrase_c1_ghost, start=72.0)

    # Melody: descent through alien intervals (peak start, then diminuendo)
    phrase_c2 = line(
        tones=[_R8, _s6, _P5, _h7, _P5, _R],
        rhythm=RhythmCell(
            spans=(1.0, 1.2, 0.8, 1.6, 1.0, 2.8),
            gates=(0.85, 0.82, 0.78, 0.90, 0.80, 1.0),
        ),
        amp_db=-11.0,
        velocity=(1.15, 1.08, 0.90, 1.10, 0.88, 0.78),
        articulation=ArticulationSpec(
            accent_pattern=(1.12, 1.05, 0.88, 1.08, 0.85, 0.78),
        ),
    )
    score.add_phrase("melody", phrase_c2, start=78.0)

    # Ghost: ascending mirror
    phrase_c2_ghost = line(
        tones=[_m3, _P5, _h7, _R8],
        rhythm=RhythmCell(
            spans=(1.0, 1.2, 1.5, 2.5),
            gates=(0.80, 0.85, 0.88, 1.0),
        ),
        amp_db=-13.5,
        velocity=(0.85, 0.95, 1.08, 1.15),
        articulation=ArticulationSpec(accent_pattern=(0.82, 0.92, 1.05, 1.12)),
    )
    score.add_phrase("ghost", phrase_c2_ghost, start=82.0)

    # Second C harmony: P5-rooted voicing (P5 + h7 + R') — harmonic shift
    score.add_note("harmony", start=84.0, duration=12.0, partial=_P5_1, amp_db=-18.5)
    score.add_note("harmony", start=84.0, duration=12.0, partial=_h7_1, amp_db=-19.5)
    score.add_note("harmony", start=84.0, duration=12.0, partial=_R, amp_db=-20.5)

    # Texture: varied durations and all Colundi high-octave degrees
    for tick_t, tick_partial, tick_dur, tick_amp, tick_vel in [
        (68.0, _R_HI, 0.3, -17.0, 1.05),
        (71.0, _h7_HI, 1.0, -18.0, 0.95),
        (74.0, _s6_HI, 0.4, -19.0, 1.10),
        (77.5, _P5_HI, 0.3, -17.0, 1.08),
        (79.0, _m3_HI, 0.8, -19.0, 0.90),
        (81.0, _h7_HI, 0.25, -18.0, 1.05),
        (84.0, _R_HI, 1.5, -17.0, 1.0),
        (87.0, _N2_HI, 0.5, -19.0, 0.88),
        (89.0, _P5_HI, 0.3, -18.0, 1.12),
        (92.0, _h7_HI, 0.8, -19.0, 0.95),
    ]:
        score.add_note(
            "texture",
            start=tick_t,
            duration=tick_dur,
            partial=tick_partial,
            amp_db=tick_amp,
            velocity=tick_vel,
        )

    # ===================================================================
    # Section B' (1:36–1:56): Thinning
    # ===================================================================

    # Harmony: N2-rooted voicing (N2 + P4 + h7) — new color on return
    score.add_note("harmony", start=96.0, duration=14.0, partial=_N2 / 2, amp_db=-20.0)
    score.add_note("harmony", start=96.0, duration=14.0, partial=_P4_1, amp_db=-21.0)
    score.add_note("harmony", start=96.0, duration=14.0, partial=_h7_1, amp_db=-22.0)

    # Bass returns, quieter
    score.add_note("bass", start=96.0, duration=16.0, partial=1.0, amp_db=-22.0)

    # Melody: α reprise — transposed up a step (P5-based), quieter, more wistful
    phrase_alpha_high = line(
        tones=[_P5, _h7, _R8, _h7],
        rhythm=RhythmCell(
            spans=(1.2, 1.4, 2.0, 2.5),
            gates=(0.80, 0.85, 0.90, 1.0),
        ),
        amp_db=-15.0,
        velocity=(0.85, 0.95, 1.05, 0.82),
        articulation=ArticulationSpec(accent_pattern=(0.88, 1.0, 1.10, 0.85)),
    )
    score.add_phrase("melody", phrase_alpha_high, start=100.0)

    # Ghost: β fragment — the last answer
    score.add_phrase("ghost", phrase_beta_frag, start=108.0, amp_scale=0.55)

    # Texture: sparse, fading
    score.add_note(
        "texture", start=102.0, duration=0.8, partial=_P5_HI, amp_db=-21.0, velocity=0.9
    )
    score.add_note(
        "texture",
        start=107.0,
        duration=1.2,
        partial=_h7_HI,
        amp_db=-22.0,
        velocity=0.85,
    )
    score.add_note(
        "texture",
        start=112.0,
        duration=0.4,
        partial=_m3_HI,
        amp_db=-23.0,
        velocity=0.80,
    )

    # ===================================================================
    # Section A' (1:56–2:15): Coda
    #
    # The 4:6:7 chord fades underneath the final melody statement,
    # providing a harmonic bed for the R → h7 → R to resolve into.
    # ===================================================================

    # Fading bass
    score.add_note("bass", start=116.0, duration=14.0, partial=1.0, amp_db=-24.0)

    # 4:6:7 chord as a quiet bed under the final phrase
    score.add_note("harmony", start=116.0, duration=16.0, partial=_R1, amp_db=-22.0)
    score.add_note("harmony", start=116.0, duration=16.0, partial=_P5_1, amp_db=-23.0)
    score.add_note("harmony", start=116.0, duration=16.0, partial=_h7_1, amp_db=-24.0)

    # Final phrase: R → h7 → R (gentle peak on h7)
    phrase_final = line(
        tones=[_R, _h7, _R],
        rhythm=RhythmCell(
            spans=(1.8, 2.5, 5.0),
            gates=(0.82, 0.90, 1.0),
        ),
        amp_db=-14.0,
        velocity=(0.92, 1.12, 0.82),
        articulation=ArticulationSpec(accent_pattern=(0.90, 1.10, 0.85)),
    )
    score.add_phrase("melody", phrase_final, start=118.0)

    # One last texture note — a long, fading h7 shimmer
    score.add_note(
        "texture",
        start=120.0,
        duration=2.0,
        partial=_h7_HI,
        amp_db=-22.0,
        velocity=0.80,
    )

    return score


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PIECES: dict[str, PieceDefinition] = {
    "slow_glass_v2": PieceDefinition(
        name="slow_glass_v2",
        output_name="slow_glass_v2",
        build_score=build_score,
        sections=(
            PieceSection("A: motifs", 0.0, 32.0),
            PieceSection("B: entering", 32.0, 66.0),
            PieceSection("C: climax", 66.0, 96.0),
            PieceSection("B': thinning", 96.0, 116.0),
            PieceSection("A': coda", 116.0, 135.0),
        ),
    ),
}
