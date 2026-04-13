"""Velvet Wall — Loveless-inspired JI smearing in 7-limit space.

Form: Emerge → Wall → Dissolve.  ~4:50 duration.

The piece explores pitch smearing as a compositional technique in Just
Intonation, where — unlike 12-TET — the smearing navigates a rich landscape
of consonance.  Voices linger near pure intervals and move quickly through
rough zones, so the pitch motion has harmonic colour rather than just blur.

Harmonic language: 7-limit JI centred on A2 (110 Hz).

Section I  — Emerge (0:00–1:30): thin melody, gradual thickening, wobble grows
Section II — Wall   (1:30–3:30): smeared progression, climax, max density
Section III— Dissolve (3:30–4:50): layers peel, melody returns, open-fifth resolve

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
from code_musics.generative.cloud import stochastic_cloud
from code_musics.generative.tone_pool import TonePool
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

# Wide voicings — root dropped an octave, upper voices spread for spaciousness
HOME_WIDE = [1 / 2, 5 / 4, 3 / 2, 7 / 2]
COMMA_WIDE = [1 / 2, 9 / 7, 3 / 2, 12 / 7 * 2]
DARK_WIDE = [1 / 2, 7 / 6, 4 / 3, 8 / 5 * 2]
SUSPENDED_WIDE = [1 / 2, 8 / 7, 3 / 2, 7 / 2]


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
    # Filter and detune are automated across the piece for a timbral arc:
    #   Emerge: warm/muted → Wall: opens up → Climax: brightest → Dissolve: closes
    score.add_voice(
        "melody",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "saw",
            "osc2_waveform": "saw",
            "osc2_detune_cents": 6.0,
            "osc2_level": 0.7,
            "cutoff_hz": 1400.0,
            "resonance_q": 0.08,
            "attack": 0.12,
            "decay": 0.4,
            "sustain_level": 0.75,
            "release": 1.2,
        },
        effects=[_melody_insert_delay()],
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        normalize_lufs=-19.0,
        mix_db=-3.0,
        sends=[hall_send_wet],
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="cutoff_hz"),
                segments=(
                    AutomationSegment(
                        start=0,
                        end=90,
                        shape="linear",
                        start_value=1400,
                        end_value=1800,
                    ),
                    AutomationSegment(
                        start=90,
                        end=155,
                        shape="linear",
                        start_value=1800,
                        end_value=3200,
                    ),
                    AutomationSegment(
                        start=155,
                        end=175,
                        shape="linear",
                        start_value=3200,
                        end_value=4000,
                    ),
                    AutomationSegment(
                        start=175,
                        end=210,
                        shape="linear",
                        start_value=4000,
                        end_value=2800,
                    ),
                    AutomationSegment(
                        start=210,
                        end=290,
                        shape="linear",
                        start_value=2800,
                        end_value=1200,
                    ),
                ),
            ),
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="osc2_detune_cents"),
                segments=(
                    AutomationSegment(start=0, end=90, shape="hold", value=6.0),
                    AutomationSegment(
                        start=90,
                        end=195,
                        shape="linear",
                        start_value=6.0,
                        end_value=14.0,
                    ),
                    AutomationSegment(
                        start=195,
                        end=290,
                        shape="linear",
                        start_value=14.0,
                        end_value=5.0,
                    ),
                ),
            ),
        ],
    )

    # -- tendril: FM engine counter-melody, metallic/bell-like contrast --------
    # mod_index automated: moderate → metallic at climax → gentle in dissolve
    score.add_voice(
        "tendril",
        synth_defaults={
            "engine": "fm",
            "carrier_ratio": 1.0,
            "mod_ratio": 3.5,  # 7/2 — septimal modulator for inharmonic shimmer
            "mod_index": 1.8,
            "index_decay": 0.6,
            "feedback": 0.08,
            "attack": 0.08,
            "decay": 0.5,
            "sustain_level": 0.60,
            "release": 1.8,
        },
        effects=[
            EffectSpec(
                "mod_delay",
                {
                    "delay_ms": 220.0,
                    "mod_rate_hz": 0.12,
                    "mod_depth_ms": 6.0,
                    "feedback": 0.30,
                    "feedback_lpf_hz": 3200.0,
                    "stereo_offset_deg": 95.0,
                    "mix": 0.18,
                },
            ),
        ],
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        normalize_lufs=-20.0,
        mix_db=-4.0,
        pan=-0.20,
        sends=[hall_send_wet],
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="mod_index"),
                segments=(
                    AutomationSegment(start=0, end=155, shape="hold", value=1.8),
                    AutomationSegment(
                        start=155,
                        end=185,
                        shape="linear",
                        start_value=1.8,
                        end_value=3.0,
                    ),
                    AutomationSegment(
                        start=185,
                        end=290,
                        shape="linear",
                        start_value=3.0,
                        end_value=0.8,
                    ),
                ),
            ),
        ],
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
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble"),
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

    # -- keys: piano punctuation in the dissolve — mortal against eternal ------
    score.add_voice(
        "keys",
        synth_defaults={
            "engine": "piano",
            "preset": "septimal",
            "decay_base": 4.5,
            "soundboard_color": 0.50,
            "soundboard_brightness": 0.45,
            "attack": 0.003,
            "sustain_level": 1.0,
            "release": 0.3,
        },
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        normalize_lufs=-22.0,
        mix_db=-4.0,
        pan=0.10,
        sends=[hall_send],
    )

    # -- shimmer: stochastic high-register sparkle, dissolves into reverb ------
    score.add_voice(
        "shimmer",
        synth_defaults={
            "engine": "additive",
            "n_harmonics": 4,
            "harmonic_rolloff": 0.30,
            "attack": 0.3,
            "decay": 0.5,
            "sustain_level": 0.60,
            "release": 4.0,
        },
        normalize_lufs=-28.0,
        mix_db=-10.0,
        pan=0.0,
        sends=[VoiceSend(target="hall", send_db=-4.0)],
    )


# ---------------------------------------------------------------------------
# Melody helpers
# ---------------------------------------------------------------------------


def _note(
    score: Score,
    voice: str,
    start: float,
    partial: float,
    dur: float,
    amp_db: float = -7.0,
    vel: float = 0.8,
    glide_from: float | None = None,
) -> None:
    """Write a note to any voice, optionally with a ratio glide."""
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
        voice,
        partial=partial,
        start=start,
        duration=dur,
        amp_db=amp_db,
        velocity=vel,
        pitch_motion=pm,
    )


def _m(
    score: Score,
    start: float,
    partial: float,
    dur: float,
    amp_db: float = -7.0,
    vel: float = 0.8,
    glide_from: float | None = None,
) -> None:
    """Write a melody note."""
    _note(score, "melody", start, partial, dur, amp_db, vel, glide_from)


def _t(
    score: Score,
    start: float,
    partial: float,
    dur: float,
    amp_db: float = -8.0,
    vel: float = 0.75,
    glide_from: float | None = None,
) -> None:
    """Write a tendril (counter-melody) note."""
    _note(score, "tendril", start, partial, dur, amp_db, vel, glide_from)


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

    # -- Smeared chord progression (90–141s), wide voicings ----------------------
    # HOME → COMMA → SUSPENDED → HOME (DARK saved for climax)
    progression_chords = [HOME_WIDE, COMMA_WIDE, SUSPENDED_WIDE, HOME_WIDE]
    progression_durs = [15.0, 14.0, 12.0, 10.0]  # total 51s

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

    # -- Bass pedal through the wall — anchored on root, no drift ----------------
    # The pads carry all the harmonic motion; the bass stays grounded.
    score.add_note(
        "bass",
        partial=1 / 2,
        start=WALL_START,
        duration=55.0,
        amp_db=-9,
        velocity=0.58,
    )
    score.add_note(
        "bass", partial=1 / 2, start=145.0, duration=65.0, amp_db=-9, velocity=0.58
    )

    # -- Melody continues through the wall, increasingly buried -----------------
    # Over HOME chord
    _m(score, 92.0, 7 / 4, 2.5, amp_db=-6, vel=0.82)
    _m(score, 95.0, 5 / 2, 1.5, amp_db=-7, vel=0.78)
    _m(score, 97.0, 3.0, 3.0, amp_db=-6, vel=0.82, glide_from=5 / 2)
    _m(score, 101.0, 7 / 2, 2.0, amp_db=-7, vel=0.78)
    _m(score, 103.5, 2.0, 1.5, amp_db=-8, vel=0.72)

    # Over COMMA (105–119s) — melody recedes, let pads carry the harmonic shift
    _m(score, 108.0, 9 / 7, 2.0, amp_db=-10, vel=0.62)  # comma-3rd, quieter
    _m(score, 111.0, 3 / 2, 1.0, amp_db=-11, vel=0.58)
    _m(score, 113.0, 12 / 7, 3.0, amp_db=-9, vel=0.65, glide_from=3 / 2)  # comma-7th
    _m(score, 117.0, 2.0, 2.0, amp_db=-10, vel=0.62)

    # Over SUSPENDED (119–131s) — restless, SUSPENDED-consonant intervals
    _m(score, 119.5, 8 / 7, 2.0, amp_db=-7, vel=0.75)  # septimal major 2nd
    _m(score, 122.0, 3 / 2, 2.0, amp_db=-6, vel=0.78, glide_from=8 / 7)
    _m(score, 124.5, 7 / 4, 2.5, amp_db=-7, vel=0.75)
    _m(score, 127.5, 3 / 2, 1.5, amp_db=-8, vel=0.72)
    _m(score, 129.5, 1.0, 2.0, amp_db=-9, vel=0.68, glide_from=3 / 2)  # sink to root

    # Over return HOME (131–141s) — melody surfaces briefly, warm
    _m(score, 131.5, 5 / 4, 2.0, amp_db=-8, vel=0.72)
    _m(score, 134.0, 3 / 2, 2.5, amp_db=-7, vel=0.75, glide_from=5 / 4)
    _m(score, 137.0, 7 / 4, 3.0, amp_db=-8, vel=0.70)  # sept 7th, fading into wall

    # -- Tendril counter-melody: interlocked with melody, contrary motion -------
    # The tendril enters during the HOME chord, moves opposite to the melody.
    # FM bell-like timbre contrasts with the saw lead.

    # Over HOME — tendril answers melody's opening phrases in the lower register
    _t(score, 93.5, 5 / 4, 1.5, amp_db=-9, vel=0.70)  # while melody is on 7/4
    _t(score, 96.0, 1.0, 1.0, amp_db=-10, vel=0.65)  # melody goes up, tendril goes down
    _t(score, 97.5, 7 / 6, 0.6, amp_db=-10, vel=0.62)  # quick passing tone
    _t(score, 98.3, 5 / 4, 2.0, amp_db=-8, vel=0.72, glide_from=7 / 6)
    _t(score, 101.0, 3 / 2, 1.0, amp_db=-9, vel=0.68)
    _t(score, 102.5, 7 / 4, 2.0, amp_db=-8, vel=0.72, glide_from=3 / 2)

    # Over COMMA (109–118s) — tendril follows COMMA intervals, no comma clash
    _t(score, 109.0, 9 / 7, 1.5, amp_db=-10, vel=0.62)  # COMMA third (matches pads)
    _t(score, 111.5, 12 / 7, 2.0, amp_db=-9, vel=0.65)  # COMMA seventh
    _t(score, 114.0, 3 / 2, 0.5, amp_db=-11, vel=0.58)  # quick
    _t(score, 114.8, 9 / 7, 0.4, amp_db=-11, vel=0.55)  # figuration
    _t(score, 115.5, 1.0, 2.5, amp_db=-10, vel=0.62, glide_from=9 / 7)

    # Over SUSPENDED (120–130s) — tendril in higher register, contrary to melody
    _t(score, 120.0, 5 / 2, 1.5, amp_db=-8, vel=0.72)
    _t(score, 122.0, 7 / 2, 1.5, amp_db=-8, vel=0.72, glide_from=5 / 2)
    _t(score, 124.0, 3.0, 0.5, amp_db=-10, vel=0.62)
    _t(score, 124.8, 5 / 2, 0.4, amp_db=-10, vel=0.60)
    _t(score, 125.5, 2.0, 2.0, amp_db=-9, vel=0.65, glide_from=5 / 2)

    # Over return HOME (132–140s) — tendril and melody converge, fading
    _t(score, 132.0, 3 / 2, 0.8, amp_db=-8, vel=0.75)
    _t(score, 133.0, 7 / 4, 0.5, amp_db=-9, vel=0.70)
    _t(score, 133.7, 2.0, 0.4, amp_db=-9, vel=0.68)
    _t(score, 134.3, 5 / 2, 0.5, amp_db=-8, vel=0.72)
    _t(score, 135.0, 3.0, 0.4, amp_db=-9, vel=0.68)
    _t(score, 135.7, 5 / 2, 2.5, amp_db=-9, vel=0.65, glide_from=3.0)  # settle

    # -- HOME arrival: clear consonant anchor before the climax (141–155s) -------
    for p in HOME:
        score.add_note(
            "pad_a", partial=p, start=141.0, duration=14.0, amp_db=-10, velocity=0.62
        )
    for p in [1.0, 5 / 4, 3 / 2]:
        score.add_note(
            "pad_b", partial=p, start=143.0, duration=12.0, amp_db=-12, velocity=0.55
        )
    # Brief melody motif recalling the opening — "we're home"
    _m(score, 143.0, 2.0, 2.0, amp_db=-8, vel=0.72)
    _m(score, 145.5, 7 / 4, 2.5, amp_db=-7, vel=0.75)
    _m(score, 148.5, 3 / 2, 4.0, amp_db=-8, vel=0.70, glide_from=7 / 4)  # settle on 5th

    # -- Climax: faster melody figuration (160-175s) ----------------------------
    # A rapid 7-limit arpeggio that dissolves into the reverb wash
    _m(score, 160.0, 5 / 2, 0.5, amp_db=-7, vel=0.85)
    _m(score, 160.7, 3.0, 0.4, amp_db=-8, vel=0.80)
    _m(score, 161.3, 7 / 2, 0.5, amp_db=-7, vel=0.85)
    _m(score, 162.0, 4.0, 0.4, amp_db=-8, vel=0.78)
    _m(score, 162.6, 7 / 2, 0.5, amp_db=-7, vel=0.82)
    _m(score, 163.3, 3.0, 0.6, amp_db=-8, vel=0.78)
    _m(score, 164.2, 5 / 2, 3.0, amp_db=-7, vel=0.80, glide_from=3.0)  # settle

    # Tendril answers with its own rapid descent
    _t(score, 161.0, 7 / 2, 0.4, amp_db=-8, vel=0.78)
    _t(score, 161.6, 3.0, 0.4, amp_db=-9, vel=0.75)
    _t(score, 162.2, 5 / 2, 0.5, amp_db=-8, vel=0.78)
    _t(score, 162.9, 2.0, 0.4, amp_db=-9, vel=0.75)
    _t(score, 163.5, 7 / 4, 0.5, amp_db=-8, vel=0.78)
    _t(score, 164.2, 5 / 4, 0.6, amp_db=-9, vel=0.72)
    _t(score, 165.0, 1.0, 3.0, amp_db=-8, vel=0.75, glide_from=5 / 4)

    # Melody and tendril interlock at the peak (170-180s)
    _m(score, 170.0, 7 / 4, 1.5, amp_db=-7, vel=0.82)
    _t(score, 170.8, 5 / 4, 1.0, amp_db=-8, vel=0.75)
    _m(score, 172.0, 5 / 2, 0.8, amp_db=-7, vel=0.80)
    _t(score, 172.5, 7 / 4, 0.8, amp_db=-8, vel=0.72)
    _m(score, 173.2, 3.0, 2.0, amp_db=-6, vel=0.85, glide_from=5 / 2)
    _t(score, 174.0, 2.0, 1.5, amp_db=-8, vel=0.72, glide_from=7 / 4)
    _m(score, 176.0, 7 / 2, 3.0, amp_db=-6, vel=0.82)  # melody's climax note
    _t(
        score, 177.0, 3 / 2, 2.5, amp_db=-9, vel=0.68, glide_from=2.0
    )  # tendril descends

    # Post-climax: both voices slow down and thin
    _m(score, 180.0, 3.0, 2.0, amp_db=-7, vel=0.78)
    _t(score, 181.0, 5 / 4, 2.0, amp_db=-9, vel=0.65)
    _m(score, 183.0, 5 / 2, 2.5, amp_db=-8, vel=0.72)
    _t(score, 184.0, 7 / 4, 2.0, amp_db=-10, vel=0.60)
    _m(score, 186.0, 2.0, 3.0, amp_db=-8, vel=0.72)
    _t(score, 187.0, 1.0, 2.5, amp_db=-10, vel=0.58)  # tendril sinks to root

    # -- Climax pads: second pass through progression, denser (155–195s) --------
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
        "ghost", partial=7 / 4, start=140.0, duration=18.0, amp_db=-16, velocity=0.36
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

    # -- Stochastic shimmer: high-register sparkle through the wall --------------
    shimmer_pool = TonePool.weighted(
        {
            2.0: 5.0,
            4.0: 4.0,  # octaves — safest
            3.0: 3.5,
            6.0: 2.5,  # fifths
            5 / 2: 2.0,
            5.0: 1.5,  # major 3rds
            7 / 2: 1.0,
            7.0: 0.5,  # septimal — rare color
        }
    )

    wall_shimmer = stochastic_cloud(
        tones=shimmer_pool,
        duration=36.0,
        density=[(0.0, 0.2), (0.5, 0.35), (1.0, 0.25)],
        amp_db_range=(-18.0, -14.0),
        note_dur_range=(2.0, 5.0),
        pitch_kind="partial",
        seed=77,
    )
    score.add_phrase("shimmer", wall_shimmer, start=105.0)

    climax_shimmer = stochastic_cloud(
        tones=shimmer_pool,
        duration=35.0,
        density=[(0.0, 0.4), (0.4, 0.8), (0.7, 0.7), (1.0, 0.3)],
        amp_db_range=(-16.0, -12.0),
        note_dur_range=(1.5, 4.0),
        pitch_kind="partial",
        seed=78,
    )
    score.add_phrase("shimmer", climax_shimmer, start=155.0)

    dissolve_shimmer = stochastic_cloud(
        tones=shimmer_pool,
        duration=50.0,
        density=[(0.0, 0.3), (0.3, 0.2), (0.7, 0.1), (1.0, 0.05)],
        amp_db_range=(-18.0, -14.0),
        note_dur_range=(3.0, 6.0),
        pitch_kind="partial",
        seed=79,
    )
    score.add_phrase("shimmer", dissolve_shimmer, start=210.0)

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

    # pad_b adds 3/2 in the final stretch — reinforces the open-fifth resolution
    score.add_note(
        "pad_b", partial=3 / 2, start=268.0, duration=14.0, amp_db=-14, velocity=0.42
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

    # -- Melody and tendril return, interlocked, echoing the opening -------------
    _m(score, 212.0, 2.0, 3.0, amp_db=-7, vel=0.78)  # A3 — familiar
    _t(score, 213.0, 5 / 4, 2.0, amp_db=-10, vel=0.62)  # tendril answers softly
    _m(score, 216.0, 7 / 4, 2.5, amp_db=-7, vel=0.75)  # sept 7th
    _t(score, 217.0, 3 / 2, 1.5, amp_db=-10, vel=0.60)  # tendril mirrors
    _m(score, 219.5, 3 / 2, 1.5, amp_db=-8, vel=0.72)  # E3
    _m(score, 221.5, 2.0, 3.5, amp_db=-7, vel=0.75, glide_from=3 / 2)  # glide home
    _t(
        score, 222.0, 7 / 4, 2.5, amp_db=-10, vel=0.58, glide_from=3 / 2
    )  # parallel glide

    # Second phrase: melody and tendril over the SUSPENDED chord
    _m(score, 226.0, 5 / 2, 2.0, amp_db=-8, vel=0.72)
    _t(score, 226.5, 7 / 4, 1.5, amp_db=-10, vel=0.60)
    _m(score, 228.5, 7 / 4, 1.5, amp_db=-7, vel=0.75)
    _t(score, 229.0, 5 / 4, 1.5, amp_db=-11, vel=0.55)
    _m(
        score, 230.5, 7 / 6, 3.0, amp_db=-7, vel=0.75, glide_from=7 / 4
    )  # sept minor 3rd
    _t(score, 231.0, 1.0, 2.5, amp_db=-11, vel=0.52, glide_from=5 / 4)  # tendril sinks

    # Fragmenting — tendril fades before melody, gravitating toward HOME
    _m(score, 235.0, 3 / 2, 1.5, amp_db=-8, vel=0.68)
    _t(score, 235.5, 7 / 4, 1.5, amp_db=-12, vel=0.48)  # tendril barely there
    _m(score, 237.0, 7 / 4, 2.5, amp_db=-7, vel=0.72)  # shorter sept 7th
    _t(score, 238.0, 5 / 4, 2.0, amp_db=-13, vel=0.42)  # tendril's last breath
    _m(score, 240.0, 3 / 2, 2.0, amp_db=-8, vel=0.68)  # pull toward fifth
    _m(score, 243.0, 2.0, 2.0, amp_db=-9, vel=0.62)
    _m(score, 246.0, 5 / 4, 1.5, amp_db=-10, vel=0.58)

    # -- Piano punctuation: sparse, decaying notes against the pad wash ---------
    # Physical, mortal sounds in the eternal wash. Not a melody — moments.
    score.add_note(
        "keys", partial=5 / 4, start=214.0, duration=5.0, amp_db=-8, velocity=0.68
    )
    score.add_note(
        "keys", partial=3 / 2, start=219.0, duration=4.0, amp_db=-9, velocity=0.62
    )
    score.add_note(
        "keys", partial=2.0, start=224.0, duration=4.5, amp_db=-8, velocity=0.65
    )
    score.add_note(
        "keys",
        partial=3 / 2,
        start=229.0,
        duration=5.0,
        amp_db=-9,
        velocity=0.60,
        pitch_motion=PitchMotionSpec.ratio_glide(start_ratio=8 / 7, end_ratio=3 / 2),
    )
    score.add_note(
        "keys", partial=5 / 4, start=238.0, duration=4.0, amp_db=-10, velocity=0.55
    )
    score.add_note(
        "keys", partial=7 / 4, start=246.0, duration=5.0, amp_db=-10, velocity=0.52
    )
    score.add_note(
        "keys", partial=3 / 2, start=256.0, duration=6.0, amp_db=-10, velocity=0.50
    )

    # -- Ghost melody: the memory — denser than before -------------------------
    score.add_note(
        "ghost", partial=7 / 4, start=225.0, duration=15.0, amp_db=-15, velocity=0.40
    )
    score.add_note(
        "ghost", partial=3 / 2, start=238.0, duration=18.0, amp_db=-16, velocity=0.38
    )
    score.add_note(
        "ghost", partial=2.0, start=248.0, duration=14.0, amp_db=-15, velocity=0.38
    )

    # -- Final passage: 7/4 yearns, resolves through 5/4 warmth to open fifth --
    _m(score, 250.0, 2.0, 3.0, amp_db=-10, vel=0.55, glide_from=5 / 4)

    # 7/4 hold — the yearning, but it resolves this time
    _m(score, 255.0, 7 / 4, 10.0, amp_db=-11, vel=0.50)
    # Resolve: 7/4 glides down to 3/2 (the fifth)
    _m(score, 265.0, 3 / 2, 10.0, amp_db=-12, vel=0.48, glide_from=7 / 4)

    # pad_b: brief 5/4 warmth during the 7/4 hold, fades before resolution
    score.add_note(
        "pad_b", partial=5 / 4, start=258.0, duration=10.0, amp_db=-14, velocity=0.42
    )

    # Ghost: high fifth reinforces the final open-fifth sonority
    score.add_note(
        "ghost", partial=3.0, start=262.0, duration=20.0, amp_db=-16, velocity=0.38
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
