"""Velvet Wall — Loveless-inspired JI smearing in 7-limit space.

Form: Emerge → Wall → Dissolve.  ~4:50 duration.

The piece explores pitch smearing as a compositional technique in Just
Intonation, where — unlike 12-TET — the smearing navigates a rich landscape
of consonance.  Voices linger near pure intervals and move quickly through
rough zones, so the pitch motion has harmonic colour rather than just blur.

Harmonic language: 7-limit JI centred on A2 (110 Hz).

Section I  — Emerge (0:00–1:30): thin melody, gradual thickening, wobble grows
Section II — Wall   (1:30–3:30): smeared progression, comma drift, max density
Section III— Dissolve (3:30–4:50): layers peel, melody returns, septimal fade

Voice layout:
  - pad_a:   additive, 4 unison voices, 10ct detune, warm JI partials
  - pad_b:   additive, 3 unison voices, 14ct detune, septimal partials, chorus
  - melody:  polyblep, dual osc saw, mod_delay
  - bass:    additive, subharmonic, simple
  - ghost:   additive, very quiet, long attack, heavy reverb

All voices feed a shared "hall" send bus (bricasti reverb + mod_delay +
saturation).  Master bus has tape saturation for glue.
"""

from __future__ import annotations

from code_musics.automation import AutomationSegment, AutomationSpec, AutomationTarget
from code_musics.harmonic_drift import harmonic_drift
from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    VelocityHumanizeSpec,
)
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import (
    EffectSpec,
    NoteEvent,
    Phrase,
    Score,
    SendBusSpec,
    VoiceSend,
)
from code_musics.smear import smear_progression, strum, thicken
from code_musics.synth import BRICASTI_IR_DIR, has_external_plugin

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

F0_HZ: float = 110.0  # A2

# Section boundaries (seconds)
EMERGE_START: float = 0.0
WALL_START: float = 90.0
DISSOLVE_START: float = 210.0
PIECE_END: float = 290.0

# -- Chord vocabulary (ratios relative to f0) --------------------------------

# Otonal tetrad — bright, warm, the "home" sound
HOME = [1.0, 5 / 4, 3 / 2, 7 / 4]

# Comma-shifted — same roles but through a different harmonic lens
# 5/4 → 9/7 (septimal supermajor third), 7/4 → 12/7 (septimal major sixth)
COMMA = [1.0, 9 / 7, 3 / 2, 12 / 7]

# Utonal / subharmonic — hollow, dark
DARK = [1.0, 7 / 6, 4 / 3, 8 / 5]

# Suspended — unresolved, yearning
SUSPENDED = [1.0, 8 / 7, 3 / 2, 7 / 4]


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------


def _hall_reverb() -> EffectSpec:
    """Large dark reverb — bricasti IR if available, native fallback."""
    wet_auto = AutomationSpec(
        target=AutomationTarget(kind="control", name="wet"),
        segments=(
            # Emerge: very dry
            AutomationSegment(
                start=0, end=60, shape="linear", start_value=0.10, end_value=0.14
            ),
            # Late emerge: opening
            AutomationSegment(
                start=60, end=90, shape="linear", start_value=0.14, end_value=0.22
            ),
            # Wall: wet, immersive
            AutomationSegment(
                start=90, end=150, shape="linear", start_value=0.22, end_value=0.36
            ),
            # Wall peak
            AutomationSegment(
                start=150, end=195, shape="linear", start_value=0.36, end_value=0.40
            ),
            # Dissolve: stays wet even as voices thin (the space persists)
            AutomationSegment(
                start=195, end=260, shape="linear", start_value=0.40, end_value=0.35
            ),
            # Final: stay wet — the space persists after the notes thin
            AutomationSegment(
                start=260, end=290, shape="linear", start_value=0.35, end_value=0.33
            ),
        ),
    )
    if BRICASTI_IR_DIR.exists():
        return EffectSpec(
            "bricasti",
            {"ir_name": "1 Halls 07 Large & Dark", "wet": 0.10, "lowpass_hz": 5500},
            automation=[wet_auto],
        )
    return EffectSpec(
        "reverb",
        {"room_size": 0.85, "damping": 0.40, "wet_level": 0.10},
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="control", name="wet_level"),
                segments=wet_auto.segments,
            ),
        ],
    )


def _hall_mod_delay() -> EffectSpec:
    """Modulated delay on the send bus — smeared, slowly drifting echoes."""
    return EffectSpec("mod_delay", {"preset": "dream_echo", "mix": 0.20})


def _hall_saturation() -> EffectSpec:
    """Warm saturation on the send bus return — glues the reverb tail."""
    return EffectSpec("saturation", {"preset": "tube_warm", "mix": 0.22, "drive": 1.12})


def _master_tape() -> EffectSpec:
    """Master bus tape glue — Chow Tape if available, native fallback."""
    if has_external_plugin("chow_tape"):
        return EffectSpec(
            "chow_tape",
            {"drive": 0.55, "saturation": 0.48, "bias": 0.50, "mix": 58.0},
        )
    return EffectSpec("saturation", {"preset": "tube_warm", "mix": 0.25, "drive": 1.25})


def _melody_insert_delay() -> EffectSpec:
    """Mod delay on the melody voice — pitch-wandering echoes."""
    return EffectSpec(
        "mod_delay",
        {
            "delay_ms": 340.0,
            "mod_rate_hz": 0.08,
            "mod_depth_ms": 10.0,
            "feedback": 0.40,
            "feedback_lpf_hz": 2400.0,
            "stereo_offset_deg": 120.0,
            "mix": 0.22,
        },
    )


def _pad_b_phaser() -> EffectSpec:
    """Phaser on pad_b — metallic shimmer, Loveless style."""
    if has_external_plugin("chow_phaser_stereo"):
        return EffectSpec("phaser", {"preset": "metallic_shimmer"})
    # Fallback: use chorus for width if phaser unavailable
    return EffectSpec("chorus", {"preset": "juno_wide"})


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
                effects=[_hall_reverb(), _hall_mod_delay(), _hall_saturation()],
            ),
        ],
        master_effects=[_master_tape()],
    )

    _setup_voices(score)
    _write_emerge(score)
    _write_wall(score)
    _write_dissolve(score)

    return score


# ---------------------------------------------------------------------------
# Voice setup
# ---------------------------------------------------------------------------


def _setup_voices(score: Score) -> None:
    hall_send = VoiceSend(target="hall", send_db=-6.0)
    hall_send_wet = VoiceSend(target="hall", send_db=-3.0)
    hall_send_drowned = VoiceSend(target="hall", send_db=0.0)

    # -- pad_a: warm, 5-limit aligned partials, generous unison detuning -------
    score.add_voice(
        "pad_a",
        synth_defaults={
            "engine": "additive",
            "unison_voices": 2,
            "detune_cents": 10.0,
            "n_harmonics": 6,
            "harmonic_rolloff": 0.55,
            "upper_partial_drift_cents": 3.5,
            "attack": 1.8,
            "decay": 1.0,
            "sustain_level": 0.85,
            "release": 2.5,
        },
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble"),
        normalize_lufs=-22.0,
        mix_db=1.0,
        pan=0.08,
        sends=[hall_send],
    )

    # -- pad_b: septimal character, phaser, different detune for interference ---
    score.add_voice(
        "pad_b",
        synth_defaults={
            "engine": "additive",
            "unison_voices": 2,
            "detune_cents": 14.0,
            "partials": [
                {"ratio": 1.0, "amp": 1.0},
                {"ratio": 2.0, "amp": 0.6},
                {"ratio": 3.0, "amp": 0.35},
                {"ratio": 3.5, "amp": 0.25},  # 7/2 relative — septimal colour
                {"ratio": 4.0, "amp": 0.18},
                {"ratio": 5.0, "amp": 0.10},
                {"ratio": 6.0, "amp": 0.06},
                {"ratio": 7.0, "amp": 0.04},
            ],
            "upper_partial_drift_cents": 5.0,
            "attack": 2.2,
            "decay": 1.2,
            "sustain_level": 0.80,
            "release": 3.0,
        },
        effects=[_pad_b_phaser()],
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble"),
        velocity_group="pads",
        normalize_lufs=-23.0,
        mix_db=-1.0,
        pan=-0.12,
        sends=[hall_send],
    )

    # -- melody: polyblep dual-osc saw, mod_delay for smeared echoes -----------
    score.add_voice(
        "melody",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "saw",
            "osc2_waveform": "saw",
            "osc2_detune_cents": 6.0,
            "osc2_level": 0.7,
            "cutoff_hz": 2200.0,
            "resonance_q": 0.08,
            "attack": 0.12,
            "decay": 0.4,
            "sustain_level": 0.75,
            "release": 1.2,
        },
        effects=[_melody_insert_delay()],
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        normalize_lufs=-19.0,
        mix_db=-3.0,
        sends=[hall_send_wet],
    )

    # -- bass: additive, simple, anchoring -------------------------------------
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "additive",
            "n_harmonics": 5,
            "harmonic_rolloff": 0.70,
            "attack": 0.8,
            "decay": 0.6,
            "sustain_level": 0.90,
            "release": 1.5,
        },
        normalize_lufs=-22.0,
        mix_db=-3.0,
        sends=[hall_send],
    )

    # -- ghost: very quiet, very wet, long attack — spectral shadow ------------
    score.add_voice(
        "ghost",
        synth_defaults={
            "engine": "additive",
            "unison_voices": 2,
            "detune_cents": 18.0,
            "n_harmonics": 6,
            "harmonic_rolloff": 0.40,
            "upper_partial_drift_cents": 8.0,
            "attack": 3.5,
            "decay": 2.0,
            "sustain_level": 0.70,
            "release": 5.0,
        },
        normalize_lufs=-26.0,
        mix_db=-8.0,
        pan=0.25,
        sends=[hall_send_drowned],
    )


# ---------------------------------------------------------------------------
# Melody helper
# ---------------------------------------------------------------------------


def _m(
    score: Score,
    start: float,
    partial: float,
    dur: float,
    amp_db: float = -7.0,
    vel: float = 0.8,
    glide_from: float | None = None,
) -> None:
    """Write a melody note, optionally with a ratio glide from a nearby partial."""
    pm = None
    if glide_from is not None:
        gt = min(dur * 0.35, 0.5)
        pm = PitchMotionSpec(
            kind="ratio_glide",
            params={
                "start_ratio": glide_from / partial,
                "end_ratio": 1.0,
                "glide_duration": gt,
            },
        )
    score.add_note(
        "melody",
        partial=partial,
        start=start,
        duration=dur,
        amp_db=amp_db,
        velocity=vel,
        pitch_motion=pm,
    )


# ---------------------------------------------------------------------------
# Section I — Emerge (0:00–1:30)
# ---------------------------------------------------------------------------


def _write_emerge(score: Score) -> None:
    # -- Melody: sparse, septimal intervals, gradually thickening ---------------

    # Opening: root area, tentative
    _m(score, 1.0, 2.0, 3.5, amp_db=-9, vel=0.65)  # A3, long, quiet
    _m(score, 5.5, 7 / 4, 2.0, amp_db=-10, vel=0.60)  # septimal 7th — the signature
    _m(score, 8.5, 3 / 2, 1.2, amp_db=-9, vel=0.65)  # E3
    _m(score, 10.5, 2.0, 4.0, amp_db=-8, vel=0.70, glide_from=3 / 2)  # glide back to A3

    # Second phrase: reaching higher
    _m(score, 16.0, 5 / 2, 2.5, amp_db=-8, vel=0.72)  # C#4
    _m(score, 19.0, 3.0, 1.5, amp_db=-7, vel=0.75)  # E4
    _m(score, 21.0, 7 / 2, 3.0, amp_db=-7, vel=0.78, glide_from=3.0)  # sept 7th high
    _m(score, 25.0, 5 / 2, 1.0, amp_db=-9, vel=0.68)  # step back
    _m(score, 26.5, 2.0, 3.5, amp_db=-8, vel=0.72, glide_from=5 / 2)  # settle on A3

    # Third phrase: more rhythmic, the 7/4 insistent
    _m(score, 31.5, 7 / 4, 1.5, amp_db=-7, vel=0.78)
    _m(score, 33.5, 5 / 4, 0.8, amp_db=-8, vel=0.72)  # drop to C#3
    _m(score, 34.8, 3 / 2, 0.6, amp_db=-9, vel=0.70)
    _m(
        score, 35.8, 7 / 4, 2.5, amp_db=-6, vel=0.82, glide_from=3 / 2
    )  # glide up to 7/4
    _m(score, 39.0, 2.0, 1.0, amp_db=-8, vel=0.72)
    _m(score, 40.5, 7 / 4, 4.0, amp_db=-7, vel=0.78)  # long hold

    # -- Fourth phrase (45-65s): wider leaps, more presence ---------------------
    _m(score, 46.0, 3.0, 2.0, amp_db=-6, vel=0.82)  # E4
    _m(score, 48.5, 4.0, 1.0, amp_db=-7, vel=0.78)  # A4, reaching
    _m(score, 50.0, 7 / 2, 2.0, amp_db=-6, vel=0.82, glide_from=4.0)  # glide to 7/2
    _m(score, 53.0, 5 / 2, 0.8, amp_db=-8, vel=0.72)
    _m(score, 54.2, 2.0, 1.5, amp_db=-7, vel=0.75)
    _m(score, 56.5, 7 / 4, 3.5, amp_db=-6, vel=0.80)  # long sept 7th

    # -- Bass pedal enters at 25s — very quiet, just grounding ------------------
    score.add_note(
        "bass", partial=1 / 2, start=25.0, duration=65.0, amp_db=-12, velocity=0.48
    )

    # -- Ghost voice: echoes throughout Emerge, spectral shadow ------------------
    score.add_note(
        "ghost", partial=2.0, start=15.0, duration=12.0, amp_db=-18, velocity=0.35
    )
    score.add_note(
        "ghost", partial=7 / 4, start=30.0, duration=12.0, amp_db=-16, velocity=0.40
    )
    score.add_note(
        "ghost", partial=5 / 2, start=45.0, duration=15.0, amp_db=-16, velocity=0.38
    )
    score.add_note(
        "ghost", partial=3 / 2, start=55.0, duration=15.0, amp_db=-17, velocity=0.36
    )
    score.add_note(
        "ghost", partial=7 / 2, start=70.0, duration=15.0, amp_db=-16, velocity=0.38
    )

    # -- Pad_a: soft root+fifth dyad at 30s, then fuller chords -----------------
    # 30s: just root + fifth, barely there
    score.add_note(
        "pad_a", partial=1.0, start=30.0, duration=12.0, amp_db=-14, velocity=0.50
    )
    score.add_note(
        "pad_a", partial=3 / 2, start=30.0, duration=12.0, amp_db=-14, velocity=0.50
    )

    # 42s: add the third — now it's a triad
    for p in [1.0, 5 / 4, 3 / 2]:
        score.add_note(
            "pad_a", partial=p, start=42.0, duration=14.0, amp_db=-12, velocity=0.55
        )

    # -- Pad_b enters at 40s — sustained septimal 7th underneath ----------------
    score.add_note(
        "pad_b", partial=7 / 4, start=40.0, duration=15.0, amp_db=-14, velocity=0.48
    )

    # -- Fuller HOME chord at 55s, strummed (pad_a) ----------------------------
    chord_55 = Phrase(
        events=tuple(
            NoteEvent(start=0.0, duration=15.0, partial=p, amp_db=-11, velocity=0.58)
            for p in [1.0, 5 / 4, 3 / 2, 7 / 4]
        )
    )
    strummed_55 = strum(chord_55, spread_ms=50.0, direction="down")
    for event in strummed_55.events:
        score.add_note(
            "pad_a",
            partial=event.partial,
            start=event.start + 55.0,
            duration=event.duration,
            amp_db=-11,
            velocity=0.58,
        )

    # Pad_b joins the HOME chord at 60s
    for p in [1.0, 5 / 4, 3 / 2, 7 / 4]:
        score.add_note(
            "pad_b", partial=p, start=60.0, duration=14.0, amp_db=-13, velocity=0.52
        )

    # -- Thickened melody phrase (65-85s): copies layered ON TOP of pads --------
    motif = Phrase(
        events=(
            NoteEvent(start=0.0, duration=2.0, partial=7 / 4, amp_db=-7, velocity=0.78),
            NoteEvent(start=2.5, duration=1.5, partial=3 / 2, amp_db=-8, velocity=0.72),
            NoteEvent(start=4.5, duration=3.0, partial=2.0, amp_db=-7, velocity=0.78),
        )
    )
    copies = thicken(
        motif, n=3, detune_cents=6.0, spread_ms=15.0, stereo_width=0.5, seed=7
    )
    for offset in [65.0, 76.0]:
        for i, copy in enumerate(copies):
            voice_name = "melody" if i == 1 else "pad_a"
            for event in copy.phrase.events:
                score.add_note(
                    voice_name,
                    partial=event.partial,
                    start=event.start + offset,
                    duration=event.duration,
                    amp_db=(event.amp_db or -7.0) + copy.amp_offset_db,
                    velocity=event.velocity,
                )

    # -- Pad chord at 75s leading into the Wall --------------------------------
    for p in [1.0, 5 / 4, 3 / 2, 7 / 4, 2.0]:
        score.add_note(
            "pad_a", partial=p, start=75.0, duration=18.0, amp_db=-10, velocity=0.60
        )
    for p in [1.0, 5 / 4, 3 / 2, 7 / 4]:
        score.add_note(
            "pad_b", partial=p, start=77.0, duration=15.0, amp_db=-12, velocity=0.55
        )


# ---------------------------------------------------------------------------
# Section II — Wall (1:30–3:30)
# ---------------------------------------------------------------------------


def _write_wall(score: Score) -> None:
    # Pad wobble omitted here: smear_progression notes carry pitch_motion glides,
    # and voice-level pitch_ratio automation cannot coexist with per-note
    # pitch_motion.  The pads get their organic pitch movement from unison
    # detuning (10-14ct), upper_partial_drift, and the phaser/chorus effects.

    # -- Smeared chord progression (90–155s) ------------------------------------
    # HOME → COMMA → DARK → SUSPENDED → HOME
    progression_chords = [HOME, COMMA, DARK, SUSPENDED, HOME]
    progression_durs = [15.0, 18.0, 16.0, 14.0, 12.0]  # total 75s

    smeared_phrases = smear_progression(
        progression_chords,
        progression_durs,
        overlap=0.25,
        voice_behavior=["glide", "glide", "glide", "glide"],
    )

    # Place the smeared progression on pad_a, voiced across two octaves
    prog_start = WALL_START
    for phrase in smeared_phrases:
        for event in phrase.events:
            if event.partial is None:
                continue
            # Voice across two octaves: lower and upper
            for octave_mult in [1.0, 2.0]:
                score.add_note(
                    "pad_a",
                    partial=event.partial * octave_mult,
                    start=event.start + prog_start,
                    duration=event.duration,
                    amp_db=-9 if octave_mult == 1.0 else -12,
                    velocity=0.72 if octave_mult == 1.0 else 0.60,
                    pitch_motion=event.pitch_motion,
                )

    # pad_b enters staggered, plays the same progression offset slightly
    for phrase in smeared_phrases:
        for event in phrase.events:
            score.add_note(
                "pad_b",
                partial=event.partial,
                start=event.start + prog_start + 0.15,  # slight offset for interference
                duration=event.duration,
                amp_db=-11,
                velocity=0.65,
                pitch_motion=event.pitch_motion,
            )

    # -- Harmonic drift: the comma shift (105–130s) is the centrepiece ----------
    # During chords 1→2 (HOME→COMMA), the harmonic_drift provides the
    # JI-aware trajectory.  We apply it as additional voice-level automation
    # on a held bass chord.
    drift_lanes = harmonic_drift(
        start_chord=[1 / 2, 5 / 8, 3 / 4, 7 / 8],  # HOME voiced low
        end_chord=[1 / 2, 9 / 14, 3 / 4, 6 / 7],  # COMMA voiced low
        duration=25.0,
        attraction=0.80,
        prime_limit=7,
        wander=0.0,
        smoothness=0.75,
        resolution_ms=250.0,  # coarser than default — 25s drift needs ~100 points, not 500
        seed=13,
    )

    # Write held bass notes and apply drift automation
    drift_start = 105.0
    bass_partials = [1 / 2, 5 / 8, 3 / 4, 7 / 8]
    for partial in bass_partials:
        score.add_note(
            "bass",
            partial=partial,
            start=drift_start,
            duration=25.0,
            amp_db=-8,
            velocity=0.65,
        )
    # Apply drift automation to bass voice (the lanes control pitch_ratio)
    for lane in drift_lanes:
        # Offset the automation times to start at drift_start
        shifted_segments = tuple(
            AutomationSegment(
                start=seg.start + drift_start,
                end=seg.end + drift_start,
                shape=seg.shape,
                start_value=seg.start_value,
                end_value=seg.end_value,
            )
            for seg in lane.segments
        )
        shifted_lane = AutomationSpec(
            target=lane.target,
            segments=shifted_segments,
            default_value=lane.default_value,
            mode=lane.mode,
        )
        score.voices["bass"].automation.append(shifted_lane)

    # -- Bass pedal through the rest of the wall --------------------------------
    score.add_note(
        "bass",
        partial=1 / 2,
        start=WALL_START,
        duration=15.0,
        amp_db=-10,
        velocity=0.55,
    )
    score.add_note(
        "bass", partial=1 / 2, start=130.0, duration=80.0, amp_db=-9, velocity=0.58
    )

    # -- Melody continues through the wall, increasingly buried -----------------
    # Over HOME chord
    _m(score, 92.0, 7 / 4, 2.5, amp_db=-6, vel=0.82)
    _m(score, 95.0, 5 / 2, 1.5, amp_db=-7, vel=0.78)
    _m(score, 97.0, 3.0, 3.0, amp_db=-6, vel=0.82, glide_from=5 / 2)
    _m(score, 101.0, 7 / 2, 2.0, amp_db=-7, vel=0.78)
    _m(score, 103.5, 2.0, 1.5, amp_db=-8, vel=0.72)

    # Over COMMA (the drift moment) — melody recedes, let pads dominate
    _m(score, 108.0, 9 / 7, 2.0, amp_db=-10, vel=0.62)  # the comma-3rd, quieter
    _m(score, 111.0, 3 / 2, 1.0, amp_db=-11, vel=0.58)
    _m(score, 113.0, 12 / 7, 3.0, amp_db=-9, vel=0.65, glide_from=3 / 2)  # comma-7th
    _m(score, 117.0, 2.0, 2.0, amp_db=-10, vel=0.62)
    _m(score, 120.0, 9 / 7, 1.5, amp_db=-11, vel=0.58)
    _m(score, 122.0, 1.0, 3.0, amp_db=-10, vel=0.65, glide_from=9 / 7)  # sink to root

    # Over DARK — darker melody, lower register
    _m(score, 126.0, 7 / 6, 2.5, amp_db=-7, vel=0.75)  # septimal minor 3rd
    _m(score, 129.0, 4 / 3, 1.5, amp_db=-8, vel=0.72)
    _m(score, 131.0, 8 / 5, 2.0, amp_db=-7, vel=0.75, glide_from=4 / 3)
    _m(score, 134.0, 7 / 6, 1.0, amp_db=-9, vel=0.68)
    _m(score, 135.5, 1.0, 2.5, amp_db=-8, vel=0.72)

    # Over SUSPENDED — restless, unresolved
    _m(score, 139.0, 8 / 7, 1.5, amp_db=-7, vel=0.78)  # septimal major 2nd
    _m(score, 141.0, 3 / 2, 2.0, amp_db=-6, vel=0.82, glide_from=8 / 7)
    _m(score, 143.5, 7 / 4, 3.0, amp_db=-6, vel=0.82)

    # Over return HOME — melody buried in the wall, almost lost
    _m(score, 147.5, 5 / 2, 2.0, amp_db=-9, vel=0.65)
    _m(score, 150.0, 7 / 2, 4.0, amp_db=-8, vel=0.70, glide_from=5 / 2)
    _m(score, 155.0, 3.0, 2.0, amp_db=-9, vel=0.65)

    # -- Climax: second pass through progression, denser (155–195s) -------------
    climax_chords = [HOME, DARK, SUSPENDED, HOME]
    climax_durs = [10.0, 12.0, 10.0, 10.0]
    climax_phrases = smear_progression(
        climax_chords,
        climax_durs,
        overlap=0.3,
    )

    climax_start = 155.0
    for phrase in climax_phrases:
        for event in phrase.events:
            if event.partial is None:
                continue
            for octave in [
                1.0,
                2.0,
            ]:  # two octaves — density from overlap, not stacking
                amp = -8 if octave == 1.0 else -11
                vel = 0.75 if octave == 1.0 else 0.62
                score.add_note(
                    "pad_a",
                    partial=event.partial * octave,
                    start=event.start + climax_start,
                    duration=event.duration,
                    amp_db=amp,
                    velocity=vel,
                    pitch_motion=event.pitch_motion,
                )

    # pad_b doubles the climax at 1x octave (safe with vectorized automation)
    for phrase in climax_phrases:
        for event in phrase.events:
            if event.partial is None:
                continue
            score.add_note(
                "pad_b",
                partial=event.partial,
                start=event.start + climax_start + 0.12,
                duration=event.duration,
                amp_db=-12,
                velocity=0.58,
                pitch_motion=event.pitch_motion,
            )

    # Ghost voice: long sustained notes through the wall — more activity
    score.add_note(
        "ghost", partial=7 / 4, start=95.0, duration=25.0, amp_db=-15, velocity=0.40
    )
    score.add_note(
        "ghost", partial=2.0, start=110.0, duration=20.0, amp_db=-16, velocity=0.38
    )
    score.add_note(
        "ghost", partial=5 / 4, start=125.0, duration=25.0, amp_db=-16, velocity=0.38
    )
    score.add_note(
        "ghost", partial=7 / 6, start=140.0, duration=18.0, amp_db=-16, velocity=0.36
    )
    score.add_note(
        "ghost", partial=3 / 2, start=155.0, duration=30.0, amp_db=-14, velocity=0.42
    )
    score.add_note(
        "ghost", partial=7 / 2, start=175.0, duration=20.0, amp_db=-16, velocity=0.38
    )
    score.add_note(
        "ghost", partial=5 / 2, start=190.0, duration=18.0, amp_db=-15, velocity=0.40
    )

    # -- Transition to dissolve: melody reaches one last time -------------------
    _m(score, 195.0, 4.0, 1.5, amp_db=-6, vel=0.85)  # A4, highest point
    _m(score, 197.0, 7 / 2, 2.0, amp_db=-6, vel=0.82, glide_from=4.0)
    _m(score, 200.0, 3.0, 1.5, amp_db=-7, vel=0.78)
    _m(score, 202.0, 5 / 2, 2.0, amp_db=-8, vel=0.72)
    _m(score, 205.0, 7 / 4, 5.0, amp_db=-7, vel=0.78)  # last long sept 7th


# ---------------------------------------------------------------------------
# Section III — Dissolve (3:30–4:50)
# ---------------------------------------------------------------------------


def _write_dissolve(score: Score) -> None:
    # -- Pads thin out but sustain longer than before ----------------------------
    # pad_b drops first — a fading HOME chord
    for p in [1.0, 5 / 4, 3 / 2, 7 / 4]:
        score.add_note(
            "pad_b",
            partial=p,
            start=DISSOLVE_START,
            duration=15.0,
            amp_db=-13,
            velocity=0.50,
        )

    # -- Brief SUSPENDED chord (220–232s): one last tension before resolution ---
    for p in [1.0, 8 / 7, 3 / 2, 7 / 4]:
        score.add_note(
            "pad_a",
            partial=p,
            start=220.0,
            duration=12.0,
            amp_db=-12,
            velocity=0.52,
        )

    # pad_a continues with thinning voicings
    for p in [1.0, 3 / 2, 7 / 4]:
        score.add_note(
            "pad_a",
            partial=p,
            start=DISSOLVE_START,
            duration=20.0,
            amp_db=-11,
            velocity=0.55,
        )
    # Root + fifth
    for p in [1.0, 3 / 2]:
        score.add_note(
            "pad_a",
            partial=p,
            start=232.0,
            duration=22.0,
            amp_db=-13,
            velocity=0.48,
        )
    # Root alone — extends to 285s (was 275)
    score.add_note(
        "pad_a", partial=1.0, start=252.0, duration=33.0, amp_db=-13, velocity=0.45
    )

    # pad_b sustained 7/4 in the final stretch — septimal shimmer persists
    score.add_note(
        "pad_b", partial=7 / 4, start=260.0, duration=22.0, amp_db=-14, velocity=0.45
    )

    # -- Bass holds longer -----------------------------------------------------
    score.add_note(
        "bass",
        partial=1 / 2,
        start=DISSOLVE_START,
        duration=40.0,
        amp_db=-10,
        velocity=0.55,
    )
    # Extends to 280 (was 270)
    score.add_note(
        "bass", partial=1 / 2, start=250.0, duration=30.0, amp_db=-12, velocity=0.45
    )

    # -- Melody returns, clearer now, echoing the opening -----------------------
    _m(score, 212.0, 2.0, 3.0, amp_db=-7, vel=0.78)  # A3 — familiar
    _m(score, 216.0, 7 / 4, 2.5, amp_db=-7, vel=0.75)  # sept 7th
    _m(score, 219.5, 3 / 2, 1.5, amp_db=-8, vel=0.72)  # E3
    _m(score, 221.5, 2.0, 3.5, amp_db=-7, vel=0.75, glide_from=3 / 2)  # glide home

    # Second phrase: inflected by the journey, over the SUSPENDED chord
    _m(score, 226.0, 5 / 2, 2.0, amp_db=-8, vel=0.72)
    _m(score, 228.5, 7 / 4, 1.5, amp_db=-7, vel=0.75)
    _m(
        score, 230.5, 7 / 6, 3.0, amp_db=-7, vel=0.75, glide_from=7 / 4
    )  # sept minor 3rd

    # Fragmenting, but not as quiet as before
    _m(score, 235.0, 3 / 2, 1.5, amp_db=-8, vel=0.68)
    _m(score, 237.0, 7 / 4, 4.0, amp_db=-7, vel=0.72)  # one more sept 7th
    _m(score, 242.0, 2.0, 2.0, amp_db=-9, vel=0.62)
    _m(score, 245.0, 5 / 4, 1.5, amp_db=-10, vel=0.58)

    # -- Ghost melody: the memory — denser than before -------------------------
    score.add_note(
        "ghost", partial=7 / 4, start=225.0, duration=15.0, amp_db=-15, velocity=0.40
    )
    score.add_note(
        "ghost", partial=3 / 2, start=238.0, duration=18.0, amp_db=-16, velocity=0.38
    )
    score.add_note(
        "ghost", partial=2.0, start=248.0, duration=22.0, amp_db=-15, velocity=0.38
    )

    # -- Final notes: louder than before, the piece ends warm not vanishing -----
    _m(score, 250.0, 2.0, 3.0, amp_db=-10, vel=0.55, glide_from=5 / 4)

    # The last sound: 7/4, present enough to hear, dissolving into reverb tail
    _m(score, 260.0, 7 / 4, 18.0, amp_db=-12, vel=0.50)

    # Ghost doubles it an octave up — the shimmer persists
    score.add_note(
        "ghost", partial=7 / 2, start=262.0, duration=22.0, amp_db=-16, velocity=0.38
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

PIECES: dict[str, PieceDefinition] = {
    "velvet_wall": PieceDefinition(
        name="velvet_wall",
        output_name="velvet_wall",
        build_score=build_score,
        sections=(
            PieceSection(
                label="Emerge", start_seconds=EMERGE_START, end_seconds=WALL_START
            ),
            PieceSection(
                label="Wall", start_seconds=WALL_START, end_seconds=DISSOLVE_START
            ),
            PieceSection(
                label="Dissolve", start_seconds=DISSOLVE_START, end_seconds=PIECE_END
            ),
        ),
    ),
}
