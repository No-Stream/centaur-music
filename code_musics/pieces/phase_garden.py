"""Phase Garden -- polyrhythmic phase drift in 7-limit JI.

Three pad voices play the same four-chord progression at tempos in ratio 4:5:7.
They start aligned on HOME and immediately drift apart, creating constantly
shifting harmonic combinations as different chord transitions overlap.  A bass
drone anchors the harmony.

Voice A: 16s cycle (4s/chord)
Voice B: 20s cycle (5s/chord) -- ratio 4:5
Voice C: 28s cycle (7s/chord) -- ratio 4:7

The piece runs ~60s, long enough for the phase relationships to develop and
for a near-realignment to occur.

Harmonic language: 7-limit JI centred on A2 (110 Hz).
"""

from __future__ import annotations

from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    VelocityHumanizeSpec,
)
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import (
    EffectSpec,
    NoteEvent,
    Phrase,
    Score,
    SendBusSpec,
    VoiceSend,
)
from code_musics.smear import smear_progression, strum
from code_musics.synth import BRICASTI_IR_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

F0_HZ: float = 110.0  # A2

# Chord vocabulary (ratios relative to f0)
HOME = [1.0, 5 / 4, 3 / 2, 7 / 4]
DARK = [1.0, 7 / 6, 4 / 3, 8 / 5]
SUSPENDED = [1.0, 8 / 7, 3 / 2, 7 / 4]

PROGRESSION = [HOME, DARK, SUSPENDED, HOME]

# Cycle durations per voice (seconds per chord)
VOICE_A_CHORD_DUR: float = 4.0  # 16s total cycle
VOICE_B_CHORD_DUR: float = 5.0  # 20s total cycle
VOICE_C_CHORD_DUR: float = 7.0  # 28s total cycle

# Piece length -- ~60s, enough for rich phasing.
# LCM of 16, 20, 28 is 140s (full realignment), but ~60s gives us
# interesting near-alignments and partial convergences.
PIECE_DUR: float = 62.0

# How many full cycles each voice completes in PIECE_DUR:
# A: 62/16 = 3.875 cycles -- nearly 4 full cycles
# B: 62/20 = 3.1 cycles
# C: 62/28 = 2.21 cycles


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------


def _hall_reverb() -> EffectSpec:
    """Large reverb to blend drifting voices."""
    if BRICASTI_IR_DIR.exists():
        return EffectSpec(
            "bricasti",
            {"ir_name": "1 Halls 07 Large & Dark", "wet": 0.28, "lowpass_hz": 5000},
        )
    return EffectSpec(
        "reverb",
        {"room_size": 0.82, "damping": 0.45, "wet_level": 0.28},
    )


def _master_saturation() -> EffectSpec:
    """Gentle master glue."""
    return EffectSpec("saturation", {"preset": "tube_warm", "mix": 0.18, "drive": 1.1})


# ---------------------------------------------------------------------------
# Voice helpers
# ---------------------------------------------------------------------------


def _additive_pad_defaults(
    detune_cents: float,
    attack: float,
    release: float,
) -> dict:
    """Shared additive pad config -- 2 unison, 6 harmonics max."""
    return {
        "engine": "additive",
        "unison_voices": 2,
        "detune_cents": detune_cents,
        "n_harmonics": 6,
        "harmonic_rolloff": 0.50,
        "upper_partial_drift_cents": 3.0,
        "attack": attack,
        "decay": 0.8,
        "sustain_level": 0.85,
        "release": release,
    }


def _write_smeared_voice(
    score: Score,
    voice_name: str,
    chord_dur: float,
) -> None:
    """Write repeating smeared progression cycles for a voice until PIECE_DUR."""
    durations = [chord_dur] * len(PROGRESSION)
    cycle_dur = chord_dur * len(PROGRESSION)

    smeared = smear_progression(
        PROGRESSION,
        durations,
        overlap=0.22,
        voice_behavior=["glide", "glide", "glide", "glide"],
    )

    cursor = 0.0
    while cursor < PIECE_DUR:
        for voice_phrase in smeared:
            for event in voice_phrase.events:
                note_start = event.start + cursor
                # Trim notes that extend past the piece end
                if note_start >= PIECE_DUR:
                    continue
                remaining = PIECE_DUR - note_start
                note_dur = min(event.duration, remaining)
                if note_dur <= 0.1:
                    continue

                score.add_note(
                    voice_name,
                    partial=event.partial,
                    start=note_start,
                    duration=note_dur,
                    amp_db=-10.0,
                    velocity=0.68,
                    pitch_motion=event.pitch_motion,
                )
        cursor += cycle_dur


def _write_strummed_entries(
    score: Score,
    voice_name: str,
    chord_dur: float,
    strum_spread_ms: float,
) -> None:
    """Add strummed chord attacks at each chord boundary for a voice."""
    cycle_dur = chord_dur * len(PROGRESSION)
    cursor = 0.0
    while cursor < PIECE_DUR:
        for chord_idx, chord in enumerate(PROGRESSION):
            chord_time = cursor + chord_idx * chord_dur
            if chord_time >= PIECE_DUR:
                break
            remaining = PIECE_DUR - chord_time
            note_dur = min(chord_dur * 0.6, remaining)
            if note_dur <= 0.1:
                continue

            chord_phrase = Phrase(
                events=tuple(
                    NoteEvent(
                        start=0.0,
                        duration=note_dur,
                        partial=p,
                        amp_db=-14.0,
                        velocity=0.55,
                    )
                    for p in chord
                )
            )
            strummed = strum(chord_phrase, spread_ms=strum_spread_ms, direction="down")
            for event in strummed.events:
                abs_start = chord_time + event.start
                if abs_start >= PIECE_DUR:
                    continue
                score.add_note(
                    voice_name,
                    partial=event.partial,
                    start=abs_start,
                    duration=event.duration,
                    amp_db=-14.0,
                    velocity=0.55,
                )
        cursor += cycle_dur


# ---------------------------------------------------------------------------
# Score builder
# ---------------------------------------------------------------------------


def build_score() -> Score:
    score = Score(
        f0=F0_HZ,
        timing_humanize=TimingHumanizeSpec(preset="loose_late_night"),
        send_buses=[
            SendBusSpec(
                name="hall",
                effects=[_hall_reverb()],
            ),
        ],
        master_effects=[_master_saturation()],
    )

    hall_send = VoiceSend(target="hall", send_db=-4.0)
    hall_send_wet = VoiceSend(target="hall", send_db=-1.0)

    # -- Voice A: brightest, fastest (16s cycle) --------------------------------
    score.add_voice(
        "phase_a",
        synth_defaults=_additive_pad_defaults(
            detune_cents=8.0,
            attack=1.2,
            release=2.0,
        ),
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble"),
        velocity_group="phase",
        normalize_lufs=-23.0,
        mix_db=0.0,
        pan=0.0,
        sends=[hall_send],
    )

    # -- Voice B: warmer detune, moderate speed (20s cycle) ---------------------
    score.add_voice(
        "phase_b",
        synth_defaults=_additive_pad_defaults(
            detune_cents=12.0,
            attack=1.6,
            release=2.5,
        ),
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble"),
        velocity_group="phase",
        normalize_lufs=-23.0,
        mix_db=-1.0,
        pan=-0.2,
        sends=[hall_send],
    )

    # -- Voice C: widest detune, slowest (28s cycle) ----------------------------
    score.add_voice(
        "phase_c",
        synth_defaults=_additive_pad_defaults(
            detune_cents=16.0,
            attack=2.2,
            release=3.0,
        ),
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble"),
        velocity_group="phase",
        normalize_lufs=-23.0,
        mix_db=-1.0,
        pan=0.2,
        sends=[hall_send],
    )

    # -- Bass drone: 1/1 anchor -------------------------------------------------
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "additive",
            "n_harmonics": 4,
            "harmonic_rolloff": 0.65,
            "attack": 2.0,
            "decay": 1.0,
            "sustain_level": 0.90,
            "release": 3.0,
        },
        normalize_lufs=-24.0,
        mix_db=-3.0,
        sends=[hall_send_wet],
    )

    # -----------------------------------------------------------------------
    # Write the three phasing voices
    # -----------------------------------------------------------------------
    _write_smeared_voice(score, "phase_a", VOICE_A_CHORD_DUR)
    _write_smeared_voice(score, "phase_b", VOICE_B_CHORD_DUR)
    _write_smeared_voice(score, "phase_c", VOICE_C_CHORD_DUR)

    # Strummed chord onsets layered on top for organic articulation
    _write_strummed_entries(score, "phase_a", VOICE_A_CHORD_DUR, strum_spread_ms=35.0)
    _write_strummed_entries(score, "phase_b", VOICE_B_CHORD_DUR, strum_spread_ms=45.0)
    _write_strummed_entries(score, "phase_c", VOICE_C_CHORD_DUR, strum_spread_ms=55.0)

    # -----------------------------------------------------------------------
    # Bass drone -- a single long note on the fundamental
    # -----------------------------------------------------------------------
    score.add_note(
        "bass",
        partial=1 / 2,
        start=0.0,
        duration=PIECE_DUR,
        amp_db=-10.0,
        velocity=0.55,
    )

    # A second bass entry an octave up, quieter, enters at 8s for warmth
    score.add_note(
        "bass",
        partial=1.0,
        start=8.0,
        duration=PIECE_DUR - 12.0,
        amp_db=-16.0,
        velocity=0.42,
    )

    return score


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

PIECES: dict[str, PieceDefinition] = {
    "phase_garden": PieceDefinition(
        name="phase_garden",
        output_name="phase_garden",
        build_score=build_score,
        sections=(
            PieceSection(label="Aligned", start_seconds=0.0, end_seconds=16.0),
            PieceSection(label="Drifting", start_seconds=16.0, end_seconds=48.0),
            PieceSection(
                label="Convergence", start_seconds=48.0, end_seconds=PIECE_DUR
            ),
        ),
    ),
}
