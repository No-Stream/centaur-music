"""Emergence — reverse bloom / deconstruction piece.

Starts at maximum density (thick septimal JI wall, heavy effects) and gradually
peels away layers until only a melody and a single 7/4 interval remain.  The
listener discovers the melody was there all along, buried in the wall.
"""

from __future__ import annotations

from code_musics.automation import AutomationSegment, AutomationSpec, AutomationTarget
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score, VoiceSend

# ---------------------------------------------------------------------------
# Timing constants (seconds)
# ---------------------------------------------------------------------------
TOTAL_DUR = 55.0
# Phase boundaries — layers peel at these approximate points
PHASE_1_END = 12.0  # high doublings start fading
PHASE_2_END = 24.0  # pad_mid fades
PHASE_3_END = 36.0  # bass simplifies to root only
PHASE_4_END = 48.0  # pad_thin → root+fifth then fades
# 48-55: just melody + bare 7/4

F0 = 110.0

# ---------------------------------------------------------------------------
# Septimal JI ratios (relative to f0)
# ---------------------------------------------------------------------------
# Full chord spanning ~3 octaves:
#   sub-octave root, root, major third, fifth, harmonic seventh,
#   octave, octave+third, octave+fifth, octave+seventh
CHORD_RATIOS = [1 / 2, 1, 5 / 4, 3 / 2, 7 / 4, 2, 5 / 2, 3, 7 / 2]

# Sub-groups for different voices
HIGH_RATIOS = [5 / 2, 3, 7 / 2]  # top octave doublings
MID_RATIOS = [1, 5 / 4, 3 / 2, 7 / 4, 2]  # core chord
BASS_RATIOS_FULL = [1 / 2, 1]  # bass: root + octave
BASS_RATIO_SIMPLE = [1 / 2]  # bass: root only
THIN_RATIOS = [1, 3 / 2]  # root + fifth for the thinning phase
FINAL_INTERVAL = [1, 7 / 4]  # bare septimal seventh


# ---------------------------------------------------------------------------
# Melody — septimal, gentle, scalar motion
# ---------------------------------------------------------------------------
# Each tuple: (start_time, duration, ratio, amp_db, velocity)
# The melody is present from t=0 but buried; velocities increase over time.
MELODY_NOTES: list[tuple[float, float, float, float, float]] = [
    # -- buried in the wall (quiet, slow, mid-register) --
    (0.5, 3.5, 5 / 4, -19.0, 0.45),
    (4.5, 3.0, 3 / 2, -19.0, 0.45),
    (8.0, 3.5, 7 / 4, -18.0, 0.48),
    # -- starting to peek through as highs fade --
    (12.0, 3.0, 2.0, -17.0, 0.55),
    (15.5, 3.5, 7 / 4, -16.0, 0.58),
    (19.5, 3.0, 3 / 2, -15.0, 0.62),
    # -- clearly audible as pad_mid fades --
    (23.0, 3.5, 5 / 4, -14.0, 0.68),
    (27.0, 4.0, 7 / 4, -13.0, 0.72),
    (31.5, 3.0, 3 / 2, -12.0, 0.78),
    # -- exposed, melodic line carrying the piece --
    (35.0, 4.0, 5 / 4, -11.0, 0.82),
    (39.5, 3.5, 7 / 4, -10.0, 0.88),
    (43.5, 4.0, 3 / 2, -10.0, 0.90),
    # -- final phrase: descend to the bare seventh --
    (48.0, 6.5, 7 / 4, -12.0, 0.85),
]


def _mix_fade(start: float, end: float, from_db: float, to_db: float) -> AutomationSpec:
    """Build a voice mix_db automation ramp."""
    return AutomationSpec(
        target=AutomationTarget(kind="control", name="mix_db"),
        segments=(
            AutomationSegment(
                start=start,
                end=end,
                shape="linear",
                start_value=from_db,
                end_value=to_db,
            ),
        ),
    )


def _send_fade(
    start: float, end: float, from_db: float, to_db: float
) -> AutomationSpec:
    """Build a voice send_db automation ramp."""
    return AutomationSpec(
        target=AutomationTarget(kind="control", name="send_db"),
        segments=(
            AutomationSegment(
                start=start,
                end=end,
                shape="linear",
                start_value=from_db,
                end_value=to_db,
            ),
        ),
    )


def build_score() -> Score:
    """Build the Emergence reverse-bloom score."""
    score = Score(
        f0=F0,
        master_effects=[
            # Master saturation — warm glue that thins over time
            EffectSpec(
                "saturation",
                {"preset": "tube_warm", "drive": 1.2, "mix": 0.30},
                automation=[
                    AutomationSpec(
                        target=AutomationTarget(kind="control", name="mix"),
                        segments=(
                            AutomationSegment(
                                start=0.1,
                                end=PHASE_3_END,
                                shape="linear",
                                start_value=0.30,
                                end_value=0.06,
                            ),
                            AutomationSegment(
                                start=PHASE_3_END,
                                end=TOTAL_DUR,
                                shape="hold",
                                value=0.06,
                            ),
                        ),
                    ),
                ],
            ),
            # Master reverb — stays present throughout for continuity
            EffectSpec(
                "reverb",
                {"room_size": 0.78, "damping": 0.35, "wet_level": 0.28},
            ),
        ],
    )

    # -----------------------------------------------------------------------
    # Shared send bus: room reverb for spatial depth
    # -----------------------------------------------------------------------
    score.add_send_bus(
        "room",
        effects=[
            EffectSpec(
                "reverb", {"room_size": 0.82, "damping": 0.30, "wet_level": 0.80}
            ),
        ],
        return_db=0.0,
    )

    # -----------------------------------------------------------------------
    # Voice: pad_high — octave doublings, first to fade
    # -----------------------------------------------------------------------
    score.add_voice(
        "pad_high",
        synth_defaults={
            "engine": "additive",
            "preset": "soft_pad",
            "n_harmonics": 6,
            "harmonic_rolloff": 0.35,
            "detune_cents": 8.0,
            "unison_voices": 3,
            "env": {
                "attack_ms": 1800.0,
                "decay_ms": 400.0,
                "sustain_ratio": 0.80,
                "release_ms": 3000.0,
            },
        },
        effects=[
            EffectSpec("chorus", {"preset": "juno_subtle", "mix": 0.30}),
        ],
        sends=[
            VoiceSend(
                "room",
                send_db=-4.0,
                automation=[
                    _send_fade(0.1, PHASE_1_END, -4.0, -12.0),
                ],
            ),
        ],
        mix_db=-3.0,
        pan=0.15,
        automation=[
            _mix_fade(0.1, PHASE_1_END, -3.0, -40.0),
        ],
        velocity_humanize=None,
    )

    # -----------------------------------------------------------------------
    # Voice: pad_mid — core septimal chord, second to fade
    # -----------------------------------------------------------------------
    score.add_voice(
        "pad_mid",
        synth_defaults={
            "engine": "additive",
            "preset": "soft_pad",
            "n_harmonics": 5,
            "harmonic_rolloff": 0.40,
            "detune_cents": 5.0,
            "unison_voices": 2,
            "env": {
                "attack_ms": 2200.0,
                "decay_ms": 500.0,
                "sustain_ratio": 0.82,
                "release_ms": 3500.0,
            },
        },
        effects=[
            EffectSpec("chorus", {"preset": "ensemble_soft", "mix": 0.25}),
        ],
        sends=[
            VoiceSend(
                "room",
                send_db=-3.0,
                automation=[
                    _send_fade(PHASE_1_END, PHASE_2_END, -3.0, -14.0),
                ],
            ),
        ],
        mix_db=-2.0,
        pan=-0.10,
        automation=[
            _mix_fade(PHASE_1_END, PHASE_2_END, -2.0, -40.0),
        ],
        velocity_humanize=None,
    )

    # -----------------------------------------------------------------------
    # Voice: bass — sub-octave foundation, simplifies then fades
    # -----------------------------------------------------------------------
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "additive",
            "preset": "drone",
            "n_harmonics": 3,
            "harmonic_rolloff": 0.55,
            "env": {
                "attack_ms": 2500.0,
                "decay_ms": 300.0,
                "sustain_ratio": 0.88,
                "release_ms": 4000.0,
            },
        },
        sends=[
            VoiceSend("room", send_db=-8.0),
        ],
        mix_db=-1.0,
        pan=0.0,
        automation=[
            _mix_fade(PHASE_2_END, PHASE_3_END, -1.0, -40.0),
        ],
        velocity_humanize=None,
    )

    # -----------------------------------------------------------------------
    # Voice: pad_thin — root+fifth, bridges into the final exposure
    # -----------------------------------------------------------------------
    score.add_voice(
        "pad_thin",
        synth_defaults={
            "engine": "additive",
            "preset": "soft_pad",
            "n_harmonics": 4,
            "harmonic_rolloff": 0.45,
            "env": {
                "attack_ms": 2000.0,
                "decay_ms": 400.0,
                "sustain_ratio": 0.78,
                "release_ms": 3500.0,
            },
        },
        sends=[
            VoiceSend(
                "room",
                send_db=-5.0,
                automation=[
                    _send_fade(PHASE_3_END, PHASE_4_END, -5.0, -18.0),
                ],
            ),
        ],
        mix_db=-4.0,
        pan=0.08,
        automation=[
            _mix_fade(PHASE_3_END, PHASE_4_END, -4.0, -40.0),
        ],
        velocity_humanize=None,
    )

    # -----------------------------------------------------------------------
    # Voice: melody — polyblep, present from the start but buried
    # -----------------------------------------------------------------------
    score.add_voice(
        "melody",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "triangle",
            "cutoff_hz": 2400.0,
            "resonance": 0.06,
            "env": {
                "attack_ms": 120.0,
                "decay_ms": 300.0,
                "sustain_ratio": 0.60,
                "release_ms": 1400.0,
            },
        },
        sends=[
            VoiceSend("room", send_db=-3.0),
        ],
        mix_db=-6.0,
        pan=-0.05,
        velocity_db_per_unit=10.0,
        velocity_humanize=None,
    )

    # -----------------------------------------------------------------------
    # Voice: final_drone — bare 7/4 interval at the end
    # -----------------------------------------------------------------------
    score.add_voice(
        "final_drone",
        synth_defaults={
            "engine": "additive",
            "n_harmonics": 3,
            "harmonic_rolloff": 0.50,
            "env": {
                "attack_ms": 3000.0,
                "decay_ms": 200.0,
                "sustain_ratio": 0.85,
                "release_ms": 4500.0,
            },
        },
        sends=[
            VoiceSend("room", send_db=-2.0),
        ],
        mix_db=-8.0,
        pan=0.0,
        velocity_humanize=None,
    )

    # ===================================================================
    # NOTES — populate the score
    # ===================================================================

    # --- pad_high: octave doublings, held for the opening wall ---
    for ratio in HIGH_RATIOS:
        score.add_note(
            "pad_high",
            start=0.0,
            duration=PHASE_1_END + 4.0,  # overlap into fade
            freq=F0 * ratio,
            amp_db=-18.0,
        )

    # --- pad_mid: core septimal chord ---
    for ratio in MID_RATIOS:
        score.add_note(
            "pad_mid",
            start=0.0,
            duration=PHASE_2_END + 4.0,
            freq=F0 * ratio,
            amp_db=-16.0,
        )

    # --- bass: full then simplified ---
    # Full bass (root + octave) for the opening
    for ratio in BASS_RATIOS_FULL:
        score.add_note(
            "bass",
            start=0.0,
            duration=PHASE_3_END + 4.0,
            freq=F0 * ratio,
            amp_db=-14.0,
        )

    # --- pad_thin: root + fifth bridge ---
    for ratio in THIN_RATIOS:
        score.add_note(
            "pad_thin",
            start=0.0,
            duration=PHASE_4_END + 4.0,
            freq=F0 * ratio,
            amp_db=-17.0,
        )

    # --- melody: present throughout, velocity rises as layers peel ---
    for start, duration, ratio, amp_db, velocity in MELODY_NOTES:
        score.add_note(
            "melody",
            start=start,
            duration=duration,
            freq=F0 * ratio,
            amp_db=amp_db,
            velocity=velocity,
        )

    # --- final_drone: bare root + 7/4 emerging at the very end ---
    for ratio in FINAL_INTERVAL:
        score.add_note(
            "final_drone",
            start=PHASE_4_END - 4.0,  # overlap slightly with pad_thin fade
            duration=TOTAL_DUR - (PHASE_4_END - 4.0) + 2.0,  # ring past total_dur
            freq=F0 * ratio,
            amp_db=-16.0,
        )

    return score


PIECES: dict[str, PieceDefinition] = {
    "emergence_reverse": PieceDefinition(
        name="emergence_reverse",
        output_name="emergence_reverse.wav",
        build_score=build_score,
        sections=(
            PieceSection(label="wall", start_seconds=0.0, end_seconds=PHASE_1_END),
            PieceSection(
                label="highs_fade", start_seconds=PHASE_1_END, end_seconds=PHASE_2_END
            ),
            PieceSection(
                label="pad_fade", start_seconds=PHASE_2_END, end_seconds=PHASE_3_END
            ),
            PieceSection(
                label="bass_fade", start_seconds=PHASE_3_END, end_seconds=PHASE_4_END
            ),
            PieceSection(
                label="exposed", start_seconds=PHASE_4_END, end_seconds=TOTAL_DUR
            ),
        ),
    ),
}
