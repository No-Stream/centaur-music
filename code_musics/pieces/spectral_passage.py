"""Spectral Passage — timbre traversal through JI space.

~55 seconds.  The timbre itself is the harmonic journey.

A simple melody plays three times.  Each time, the additive voice that
carries it has a different partial set:

  1. Harmonic-series partials  [1, 2, 3, 4, 5, 6]       — warm, familiar
  2. Mixed partials            [1, 2, 7/4, 3, 7/2, 5]    — uncanny middle
  3. Septimal partials         [1, 7/4, 7/2, 7, 21/4, 14] — alien consonance

Voices overlap: as each new timbre enters, the previous one sustains at
lower amplitude, so the reverb tail carries ghosts of earlier spectra.

A low drone uses attack_partials → sustain partials with spectral_morph_time
to continuously traverse the same harmonic-to-septimal arc across the whole
piece.

All voices feed a shared hall reverb.  Master tape saturation for glue.

Voice layout:
  - timbre_1:  additive, harmonic-series partials (first melody pass)
  - timbre_2:  additive, mixed partials (second pass)
  - timbre_3:  additive, septimal partials (third pass)
  - drone:     additive, spectral morph from harmonic → septimal
  - ghost:     additive, very quiet spectral echoes, heavy reverb
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
from code_musics.synth import BRICASTI_IR_DIR, has_external_plugin

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

F0_HZ: float = 110.0  # A2

# Section boundaries (seconds)
HARMONIC_START: float = 0.0
MIXED_START: float = 17.0
SEPTIMAL_START: float = 34.0
PIECE_END: float = 56.0

# ---------------------------------------------------------------------------
# Partial sets — the three timbral identities
# ---------------------------------------------------------------------------

# Standard harmonic series — warm, resolved, the sound of "normal"
HARMONIC_PARTIALS = [
    {"ratio": 1.0, "amp": 1.0},
    {"ratio": 2.0, "amp": 0.6},
    {"ratio": 3.0, "amp": 0.35},
    {"ratio": 4.0, "amp": 0.20},
    {"ratio": 5.0, "amp": 0.12},
    {"ratio": 6.0, "amp": 0.07},
]

# Mixed — some standard harmonics, some septimal infiltrators
MIXED_PARTIALS = [
    {"ratio": 1.0, "amp": 1.0},
    {"ratio": 2.0, "amp": 0.50},
    {"ratio": 7 / 4, "amp": 0.40},  # septimal 7th sneaking in
    {"ratio": 3.0, "amp": 0.28},
    {"ratio": 7 / 2, "amp": 0.22},  # septimal colour
    {"ratio": 5.0, "amp": 0.10},
]

# Fully septimal — alien but internally consonant
SEPTIMAL_PARTIALS = [
    {"ratio": 1.0, "amp": 1.0},
    {"ratio": 7 / 4, "amp": 0.65},
    {"ratio": 7 / 2, "amp": 0.40},
    {"ratio": 7.0, "amp": 0.22},
    {"ratio": 21 / 4, "amp": 0.14},
    {"ratio": 14.0, "amp": 0.06},
]

# Drone: morphs from harmonic attack to septimal sustain
DRONE_ATTACK_PARTIALS = [
    {"ratio": 1.0, "amp": 1.0},
    {"ratio": 2.0, "amp": 0.55},
    {"ratio": 3.0, "amp": 0.30},
    {"ratio": 4.0, "amp": 0.18},
    {"ratio": 5.0, "amp": 0.10},
]

DRONE_SUSTAIN_PARTIALS = [
    {"ratio": 1.0, "amp": 1.0},
    {"ratio": 7 / 4, "amp": 0.55},
    {"ratio": 7 / 2, "amp": 0.30},
    {"ratio": 7.0, "amp": 0.18},
    {"ratio": 21 / 4, "amp": 0.10},
]

# ---------------------------------------------------------------------------
# The melody — seven notes, simple and memorable, septimal intervals
# ---------------------------------------------------------------------------

MELODY: list[tuple[float, float, float]] = [
    # (partial, duration, velocity)
    (2.0, 2.0, 0.78),  # A3 — home
    (7 / 4, 1.5, 0.72),  # septimal 7th — the signature interval
    (3 / 2, 1.0, 0.68),  # E3 — stable fifth
    (2.0, 1.8, 0.75),  # A3 — return
    (5 / 2, 2.2, 0.80),  # C#4 — bright reach
    (7 / 4, 1.5, 0.72),  # back to the 7th
    (1.0, 2.5, 0.70),  # A2 — settle low
]


def _melody_phrase() -> Phrase:
    """Build the core melody as a Phrase."""
    events: list[NoteEvent] = []
    t = 0.0
    for partial, dur, vel in MELODY:
        events.append(
            NoteEvent(
                start=t,
                duration=dur,
                partial=partial,
                amp_db=-7.0,
                velocity=vel,
            )
        )
        t += dur + 0.3  # small gap between notes
    return Phrase(events=tuple(events))


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------


def _hall_reverb() -> EffectSpec:
    """Large, darkish reverb — bricasti if available, native fallback."""
    if BRICASTI_IR_DIR.exists():
        return EffectSpec(
            "bricasti",
            {"ir_name": "1 Halls 07 Large & Dark", "wet": 0.28, "lowpass_hz": 5000},
        )
    return EffectSpec(
        "reverb",
        {"room_size": 0.82, "damping": 0.45, "wet_level": 0.28},
    )


def _hall_saturation() -> EffectSpec:
    """Warm saturation on the send bus — glues the reverb tail together."""
    return EffectSpec("saturation", {"preset": "tube_warm", "mix": 0.18, "drive": 1.1})


def _master_tape() -> EffectSpec:
    """Master bus tape glue."""
    if has_external_plugin("chow_tape"):
        return EffectSpec(
            "chow_tape",
            {"drive": 0.50, "saturation": 0.42, "bias": 0.50, "mix": 55.0},
        )
    return EffectSpec("saturation", {"preset": "tube_warm", "mix": 0.20, "drive": 1.15})


# ---------------------------------------------------------------------------
# Score builder
# ---------------------------------------------------------------------------


def build_score() -> Score:
    score = Score(
        f0_hz=F0_HZ,
        timing_humanize=TimingHumanizeSpec(preset="loose_late_night"),
        send_buses=[
            SendBusSpec(
                name="hall",
                effects=[_hall_reverb(), _hall_saturation()],
            ),
        ],
        master_effects=[_master_tape()],
    )

    _setup_voices(score)
    _write_drone(score)
    _write_melody_passes(score)
    _write_ghosts(score)

    return score


# ---------------------------------------------------------------------------
# Voice setup
# ---------------------------------------------------------------------------


def _setup_voices(score: Score) -> None:
    hall_send = VoiceSend(target="hall", send_db=-6.0)
    hall_send_wet = VoiceSend(target="hall", send_db=-3.0)
    hall_send_drowned = VoiceSend(target="hall", send_db=0.0)

    # -- timbre_1: harmonic series partials (warm, familiar) -------------------
    score.add_voice(
        "timbre_1",
        synth_defaults={
            "engine": "additive",
            "partials": HARMONIC_PARTIALS,
            "unison_voices": 2,
            "detune_cents": 6.0,
            "upper_partial_drift_cents": 2.5,
            "attack": 0.25,
            "decay": 0.4,
            "sustain_level": 0.80,
            "release": 1.8,
        },
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble"),
        normalize_lufs=-21.0,
        mix_db=0.0,
        pan=0.05,
        sends=[hall_send],
    )

    # -- timbre_2: mixed partials (uncanny transition) -------------------------
    score.add_voice(
        "timbre_2",
        synth_defaults={
            "engine": "additive",
            "partials": MIXED_PARTIALS,
            "unison_voices": 2,
            "detune_cents": 8.0,
            "upper_partial_drift_cents": 4.0,
            "attack": 0.30,
            "decay": 0.5,
            "sustain_level": 0.78,
            "release": 2.2,
        },
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble"),
        normalize_lufs=-21.0,
        mix_db=-1.0,
        pan=-0.08,
        sends=[hall_send],
    )

    # -- timbre_3: septimal partials (alien consonance) ------------------------
    score.add_voice(
        "timbre_3",
        synth_defaults={
            "engine": "additive",
            "partials": SEPTIMAL_PARTIALS,
            "unison_voices": 2,
            "detune_cents": 10.0,
            "upper_partial_drift_cents": 5.0,
            "attack": 0.35,
            "decay": 0.6,
            "sustain_level": 0.75,
            "release": 2.8,
        },
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble"),
        normalize_lufs=-21.0,
        mix_db=-1.5,
        pan=0.10,
        sends=[hall_send_wet],
    )

    # -- drone: spectral morph from harmonic → septimal over full duration -----
    score.add_voice(
        "drone",
        synth_defaults={
            "engine": "additive",
            "attack_partials": DRONE_ATTACK_PARTIALS,
            "partials": DRONE_SUSTAIN_PARTIALS,
            "spectral_morph_time": 0.92,  # morph over ~92% of note duration
            "unison_voices": 2,
            "detune_cents": 5.0,
            "upper_partial_drift_cents": 3.0,
            "attack": 3.0,
            "decay": 2.0,
            "sustain_level": 0.85,
            "release": 4.0,
        },
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        normalize_lufs=-24.0,
        mix_db=-3.0,
        pan=0.0,
        sends=[hall_send],
    )

    # -- ghost: very quiet, very wet spectral shadow ---------------------------
    score.add_voice(
        "ghost",
        synth_defaults={
            "engine": "additive",
            "unison_voices": 2,
            "detune_cents": 14.0,
            "n_harmonics": 6,
            "harmonic_rolloff": 0.40,
            "upper_partial_drift_cents": 7.0,
            "attack": 2.5,
            "decay": 1.5,
            "sustain_level": 0.70,
            "release": 4.0,
        },
        normalize_lufs=-27.0,
        mix_db=-8.0,
        pan=0.20,
        sends=[hall_send_drowned],
    )


# ---------------------------------------------------------------------------
# Drone — one long note, spectral morph carries the timbral arc
# ---------------------------------------------------------------------------


def _write_drone(score: Score) -> None:
    # The drone enters just before the melody and sustains through the whole piece.
    # The attack_partials → partials morph means it starts harmonic and ends septimal,
    # mirroring the melody's timbral journey but as a continuous process.
    score.add_note(
        "drone",
        partial=1 / 2,  # A1 — sub-bass foundation
        start=1.0,
        duration=52.0,
        amp_db=-10.0,
        velocity=0.55,
    )

    # A second drone voice an octave up, entering later for warmth
    score.add_note(
        "drone",
        partial=1.0,  # A2
        start=8.0,
        duration=44.0,
        amp_db=-14.0,
        velocity=0.45,
    )


# ---------------------------------------------------------------------------
# Melody — three passes, each with a different timbre
# ---------------------------------------------------------------------------


def _write_melody_passes(score: Score) -> None:
    melody = _melody_phrase()

    # -- Pass 1 (0-15s): harmonic timbre, clear and present --------------------
    for event in melody.events:
        score.add_note(
            "timbre_1",
            partial=event.partial,
            start=event.start + 2.0,  # offset from piece start
            duration=event.duration,
            amp_db=-6.0,
            velocity=event.velocity,
        )

    # -- Pass 2 (17-32s): mixed timbre enters, timbre_1 sustains a quiet chord -
    # timbre_1 holds a soft root+fifth underneath as it fades
    score.add_note(
        "timbre_1",
        partial=1.0,
        start=MIXED_START,
        duration=14.0,
        amp_db=-16.0,
        velocity=0.40,
    )
    score.add_note(
        "timbre_1",
        partial=3 / 2,
        start=MIXED_START + 0.5,
        duration=13.0,
        amp_db=-17.0,
        velocity=0.38,
    )

    # timbre_2 plays the melody
    for event in melody.events:
        score.add_note(
            "timbre_2",
            partial=event.partial,
            start=event.start + MIXED_START + 1.0,
            duration=event.duration,
            amp_db=-6.0,
            velocity=event.velocity,
        )

    # -- Pass 3 (34-50s): septimal timbre, timbre_1 and timbre_2 hold ghosts ---
    # timbre_1: faint root drone (the original timbre is now just a memory)
    score.add_note(
        "timbre_1",
        partial=1.0,
        start=SEPTIMAL_START,
        duration=16.0,
        amp_db=-20.0,
        velocity=0.32,
    )

    # timbre_2: holds the 7/4 — the interval that bridged the two worlds
    score.add_note(
        "timbre_2",
        partial=7 / 4,
        start=SEPTIMAL_START,
        duration=14.0,
        amp_db=-18.0,
        velocity=0.35,
    )
    score.add_note(
        "timbre_2",
        partial=1.0,
        start=SEPTIMAL_START + 2.0,
        duration=12.0,
        amp_db=-19.0,
        velocity=0.33,
    )

    # timbre_3 plays the melody
    for event in melody.events:
        score.add_note(
            "timbre_3",
            partial=event.partial,
            start=event.start + SEPTIMAL_START + 1.0,
            duration=event.duration * 1.1,  # slightly slower, more spacious
            amp_db=-6.0,
            velocity=event.velocity,
        )

    # -- Coda: all three timbres hold the root together, layered ---------------
    coda_start = 50.0
    score.add_note(
        "timbre_1",
        partial=1.0,
        start=coda_start,
        duration=6.0,
        amp_db=-14.0,
        velocity=0.42,
    )
    score.add_note(
        "timbre_2",
        partial=1.0,
        start=coda_start + 0.3,
        duration=5.5,
        amp_db=-15.0,
        velocity=0.40,
    )
    score.add_note(
        "timbre_3",
        partial=1.0,
        start=coda_start + 0.6,
        duration=5.0,
        amp_db=-16.0,
        velocity=0.38,
    )


# ---------------------------------------------------------------------------
# Ghost notes — spectral shadows in the reverb
# ---------------------------------------------------------------------------


def _write_ghosts(score: Score) -> None:
    # Sparse, quiet notes that sit deep in the reverb, carrying harmonic memory.
    # They use the default n_harmonics partial set (no custom partials), so they're
    # a neutral spectral presence — the reverb tail colours them.

    # Early: harmonic echoes
    score.add_note(
        "ghost", partial=2.0, start=6.0, duration=10.0, amp_db=-18.0, velocity=0.35
    )
    score.add_note(
        "ghost", partial=3 / 2, start=12.0, duration=8.0, amp_db=-19.0, velocity=0.32
    )

    # Middle: the 7/4 appears in the ghost layer too
    score.add_note(
        "ghost", partial=7 / 4, start=22.0, duration=12.0, amp_db=-17.0, velocity=0.36
    )
    score.add_note(
        "ghost", partial=5 / 2, start=28.0, duration=10.0, amp_db=-18.0, velocity=0.34
    )

    # Late: septimal echoes dominate
    score.add_note(
        "ghost", partial=7 / 2, start=38.0, duration=12.0, amp_db=-17.0, velocity=0.36
    )
    score.add_note(
        "ghost", partial=7 / 4, start=44.0, duration=10.0, amp_db=-18.0, velocity=0.34
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

PIECES: dict[str, PieceDefinition] = {
    "spectral_passage": PieceDefinition(
        name="spectral_passage",
        output_name="spectral_passage",
        build_score=build_score,
        sections=(
            PieceSection(
                label="Harmonic",
                start_seconds=HARMONIC_START,
                end_seconds=MIXED_START,
            ),
            PieceSection(
                label="Mixed", start_seconds=MIXED_START, end_seconds=SEPTIMAL_START
            ),
            PieceSection(
                label="Septimal",
                start_seconds=SEPTIMAL_START,
                end_seconds=PIECE_END,
            ),
        ),
    ),
}
