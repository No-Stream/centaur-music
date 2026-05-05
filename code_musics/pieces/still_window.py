"""still_window -- a slow xenharmonic solo-piano meditation.

Aisatsana-adjacent: one foreground piano, long silences, a single memorable
motif turned patiently three times.  7-limit JI in F# major, with septimal
intervals (7/4, 7/6, 9/7) as normal vocabulary rather than study-piece
accents.  The bias is toward pleasantness -- the piece should read as a song,
not a study.

Materials:

- motif ALPHA: a five-note stepwise descent that lands on tonic, drops to
  the submediant, and climbs back.  Stated in the Intro, developed in A,
  and restated an octave up (with a septimal cadential ornament) in A'.
- pedal BETA: sparse low F#1 / C#2 quarter-ish pulses introduced in A,
  returning under the B-section climax.  Very soft -- grounding, not bass.
- ornament GAMMA: two-note septimal grace figures (7/4 -> 2/1 cadentially,
  9/7 -> 3/2 at the B-section "opening up").  Used 3-4 times total.

Structure (~2:50):

- Intro (0:00 - 0:25)   ALPHA bare, tonic register, long tail.
- A     (0:25 - 1:10)   ALPHA varied, BETA enters, 7/6 passing tone.
- B     (1:10 - 2:00)   9/7 modulatory shift, quiet climax on 4:5:6:7,
                        tiny linear_bend on the sustained top voice.
- A'    (2:00 - 2:35)   ALPHA octave-up with 7/4 cadential ornament.
- Tail  (2:35 - 2:55)   Held tonic decays into silence.

Single `piano` voice, `felt` preset, `partial_ratios` overridden with a
7-limit set so modal overtones stay JI-compatible with the harmony.
Sympathetic resonance is on to let held notes ring into each other.  A
slow shared drift bus (0.06 Hz, 2 cents) gives the piano a room-breathing
quality rather than vibrato.  Short dark Bricasti reverb send, small
delay send that opens only at cadence points.
"""

from __future__ import annotations

from code_musics.automation import AutomationSegment, AutomationSpec, AutomationTarget
from code_musics.humanize import TimingHumanizeSpec, VelocityHumanizeSpec
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS
from code_musics.pieces.registry import PieceDefinition
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import EffectSpec, Score, SendBusSpec, VoiceSend
from code_musics.synth import BRICASTI_IR_DIR

# -- Tuning -------------------------------------------------------------------
# 7-limit JI diatonic major from F#3 tonic, plus septimal colors.  These are
# partial ratios against score.f0_hz (= F#3).  Down-octave variants are
# explicit so section authoring reads like pitch names instead of math.

F0_HZ = 185.0  # F#3

# Diatonic degrees (1/1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2/1).
P_F_S = 1.0  # F#3 (tonic)
P_G_S = 9 / 8  # G#3
P_A_S = 5 / 4  # A#3
P_B = 4 / 3  # B3
P_C_S = 3 / 2  # C#4
P_D_S = 5 / 3  # D#4
P_E_S = 15 / 8  # E#4 (leading tone)
P_F_S_OCT = 2.0  # F#4

# Septimal colors.
P_E = 7 / 4  # E4 -- septimal minor 7th (cadential flat-7)
P_A = 7 / 6  # A3 -- septimal subminor 3rd (shadow color)
P_C_X = 9 / 7  # C##3 -- supermajor 3rd (9/7 ~= 435 cents) for B-section

# Down-octave variants for the pedal and motif's 5/3(8vb).
P_F_S_DOWN = 0.5  # F#2
P_C_S_DOWN = 3 / 4  # C#3
P_D_S_DOWN = 5 / 6  # D#3 (= 5/3 one octave down)

# Up-octave variants for A' restatement.
P_A_S_UP = 5 / 2  # A#4
P_G_S_UP = 9 / 4  # G#4
P_E_UP = 7 / 2  # E5

# Pedal: a full octave lower than down-octave (F#1).
P_F_S_PEDAL = 0.25  # F#1


# -- Voice config -------------------------------------------------------------

# 7-limit partial ratio set for the piano's modal overtones.  Overrides the
# felt preset's natural mode ratios so the piano's harmonics align with the
# JI intervals we use melodically and harmonically.  inharmonicity=0.0 keeps
# these ratios pure.
_SEVEN_LIMIT_PARTIALS = [
    {"ratio": 1.0, "amp": 1.00},
    {"ratio": 2.0, "amp": 0.55},  # octave
    {"ratio": 3.0, "amp": 0.30},  # twelfth
    {"ratio": 4.0, "amp": 0.22},  # two octaves
    {"ratio": 5.0, "amp": 0.18},  # major 3rd up two octaves
    {"ratio": 6.0, "amp": 0.14},
    {"ratio": 7.0, "amp": 0.12},  # septimal partial -- JI-consistent
    {"ratio": 8.0, "amp": 0.09},
    {"ratio": 9.0, "amp": 0.08},
    {"ratio": 10.0, "amp": 0.07},
    {"ratio": 12.0, "amp": 0.05},
    {"ratio": 14.0, "amp": 0.04},
]


def _piano_synth_defaults() -> dict[str, object]:
    """felt piano base, partials overridden to a 7-limit set.

    felt's default hammer_stiffness=2e7 and decay_base=2.5 keep the tone
    intimate and soft; the 7-limit partial override replaces the usual
    near-harmonic modal ladder with ratios that alias cleanly against the
    piece's JI intervals.  inharmonicity=0 because the partial set is
    already doing the spectral-shape work.
    """
    return {
        "engine": "piano",
        "preset": "felt",
        "partial_ratios": _SEVEN_LIMIT_PARTIALS,
        "inharmonicity": 0.0,
        # A slightly softer hammer than felt's default -- the motif needs
        # to sit very quiet at the intro and coda.
        "hammer_stiffness": 1.5e7,
        # Slightly longer decay than felt's 2.5 so held notes actually hang
        # in the air long enough for sympathetic resonance to read.
        "decay_base": 3.4,
    }


def _piano_effects() -> list[EffectSpec]:
    """EQ (mild cleanup) -> Juno-I chorus for stereo width from mono."""
    return [
        EffectSpec(
            "eq",
            {
                "bands": [
                    {"kind": "highpass", "cutoff_hz": 60.0, "slope_db_per_oct": 12},
                    {"kind": "high_shelf", "freq_hz": 8000.0, "gain_db": -1.5},
                ]
            },
        ),
        EffectSpec(
            "bbd_chorus",
            {"preset": "juno_i", "mix": 0.12},
        ),
    ]


# -- Send buses ---------------------------------------------------------------


def _reverb_bus() -> SendBusSpec:
    """Dark, long Bricasti hall with pre-delay and warm tone shaping.

    Falls back to native reverb if the IR directory isn't available.
    """
    if BRICASTI_IR_DIR.exists():
        reverb = EffectSpec(
            "bricasti",
            {
                "ir_name": "1 Halls 07 Large & Dark",
                "wet": 0.55,
                "highpass_hz": 200.0,
                "lowpass_hz": 6000.0,
                "tilt_db": -1.0,
            },
        )
    else:
        reverb = EffectSpec(
            "reverb", {"room_size": 0.88, "damping": 0.65, "wet_level": 0.55}
        )
    return SendBusSpec(name="hall", effects=[reverb], return_db=0.0)


def _delay_bus() -> SendBusSpec:
    """Short quarter-note-ish delay, mostly wet.  Rides on cadential gestures."""
    return SendBusSpec(
        name="echo",
        effects=[
            EffectSpec(
                "delay",
                {"delay_seconds": 0.42, "feedback": 0.28, "mix": 1.0},
            )
        ],
        return_db=-6.0,
    )


# -- Motif alpha --------------------------------------------------------------

# Motif alpha as a sequence of (partial, start_offset, duration, amp_db, velocity)
# tuples.  We don't use Phrase/ratio_line here because the rhythm is
# deliberately uneven and several notes carry per-note pitch_motion or amp_db
# overrides that read clearer inline.
#
# Shape (5 notes):
#   1. 5/4 (A#3)       dotted-half   -- the "opening" note
#   2. 9/8 (G#3)       half          -- stepwise down
#   3. 1/1 (F#3)       quarter       -- tonic arrival
#   4. 5/6 (D#3, 8vb)  quarter       -- drop to the submediant below
#   5. 1/1 (F#3)       whole+fermata -- home, long hold
#
# Rough timings (in seconds, relative to phrase start):
#   t=0.00  n1  d=2.4  amp_db=-12 vel=0.82
#   t=2.60  n2  d=1.7  amp_db=-13 vel=0.78
#   t=4.45  n3  d=0.95 amp_db=-14 vel=0.72
#   t=5.55  n4  d=0.95 amp_db=-13 vel=0.76
#   t=6.70  n5  d=5.5  amp_db=-13 vel=0.74
#
# The last note bleeds through whatever comes next via the reverb + sympathetic
# resonance; score authors should leave 2-3 s of silence before the next
# phrase.

# Motif alpha, with real velocity shaping within the 5 notes:
#   - opening 5/4 lands with a clear mp accent (0.82), the "statement"
#   - 9/8 softer, step down (0.62)
#   - 1/1 tonic arrival sits just a touch up (0.70) -- a small relief
#   - 5/6 (D#3 octave down) rises into accent as the "reach" (0.86)
#   - closing 1/1 is the quiet settle (0.52)
#
# Durations vary substantially: dotted-half sustained, half, quick
# quarter, quarter+tie, long-held-with-fermata.  Piano decay carries
# the long holds; short notes sound articulate despite the pedal.
_ALPHA_NOTES: tuple[tuple[float, float, float, float, float], ...] = (
    (P_A_S, 0.00, 2.00, -12.0, 0.82),  # accent: statement note
    (P_G_S, 2.15, 1.40, -13.5, 0.62),  # relaxed step down
    (P_F_S, 3.65, 0.55, -14.0, 0.70),  # articulate tonic arrival
    (P_D_S_DOWN, 4.30, 0.75, -12.5, 0.86),  # accent: downward reach
    (P_F_S, 5.20, 5.00, -14.5, 0.52),  # quiet settle, long hold
)

ALPHA_DURATION = 10.2  # seconds including the final note's tail


def _place_alpha(
    score: Score,
    voice: str,
    start: float,
    *,
    partial_scale: float = 1.0,
    amp_db_offset: float = 0.0,
    velocity_scale: float = 1.0,
    last_note_pitch_motion: PitchMotionSpec | None = None,
) -> None:
    """Place motif alpha on the voice, with optional transforms.

    partial_scale multiplies every partial (use 2.0 for the A' octave-up
    restatement).  amp_db_offset shifts the whole phrase's dynamics.
    last_note_pitch_motion attaches a bend to the sustained final note.
    """
    for idx, (partial, t_rel, dur, amp_db, vel) in enumerate(_ALPHA_NOTES):
        pitch_motion = last_note_pitch_motion if idx == len(_ALPHA_NOTES) - 1 else None
        score.add_note(
            voice,
            start=start + t_rel,
            duration=dur,
            partial=partial * partial_scale,
            amp_db=amp_db + amp_db_offset,
            velocity=vel * velocity_scale,
            pitch_motion=pitch_motion,
        )


# -- Section timings ----------------------------------------------------------

# Section starts scaled ~0.83x from the original timing.  Alpha is now
# ~10s instead of 12s; the rest of the piece matches that pace.
INTRO_START = 0.0
A_START = 20.0
B_START = 57.0
APRIME_START = 98.0
TAIL_START = 128.0
TOTAL_DUR = 145.0  # ~2:25


# -- Score build --------------------------------------------------------------


def build_score() -> Score:
    """Build the still_window score."""
    score = Score(
        f0_hz=F0_HZ,
        master_effects=DEFAULT_MASTER_EFFECTS,
        send_buses=[_reverb_bus(), _delay_bus()],
        timing_humanize=TimingHumanizeSpec(
            preset="chamber",
            ensemble_amount_ms=30.0,
            micro_jitter_ms=0.8,
            seed=19830104,
        ),
    )

    # Shared drift bus -- one voice, but we wire it through so the piece
    # reads as "played in a room" rather than "rendered on a grid".
    score.add_drift_bus(
        "ensemble",
        rate_hz=0.06,
        depth_cents=2.0,
        seed=19830104,
    )

    # ---- Send-level automation on the delay bus -----------------------
    # Sits at -inf for most of the piece (send_db = -60 is effectively off).
    # Opens to musically audible levels only at cadential moments:
    # end of A, climax of B, A' cadence, and briefly into the tail.
    delay_send_automation = AutomationSpec(
        target=AutomationTarget(kind="control", name="send_db"),
        segments=(
            AutomationSegment(
                start=0.0,
                end=A_START + 33.0,
                shape="hold",
                value=-60.0,
            ),
            # End-of-A cadence: open briefly under the 7/6 passing tone
            # and its resolution.
            AutomationSegment(
                start=A_START + 33.0,
                end=A_START + 35.5,
                shape="linear",
                start_value=-60.0,
                end_value=-16.0,
            ),
            AutomationSegment(
                start=A_START + 35.5,
                end=B_START,
                shape="linear",
                start_value=-16.0,
                end_value=-60.0,
            ),
            # B-section climax: the 4:5:6:7 chord rings with gentle echo.
            AutomationSegment(
                start=B_START + 23.0,
                end=B_START + 26.5,
                shape="linear",
                start_value=-60.0,
                end_value=-14.0,
            ),
            AutomationSegment(
                start=B_START + 26.5,
                end=APRIME_START,
                shape="linear",
                start_value=-14.0,
                end_value=-60.0,
            ),
            # A' cadential 7/4 ornament opens the echo again.
            AutomationSegment(
                start=APRIME_START + 16.5,
                end=APRIME_START + 20.0,
                shape="linear",
                start_value=-60.0,
                end_value=-14.0,
            ),
            AutomationSegment(
                start=APRIME_START + 20.0,
                end=TAIL_START,
                shape="linear",
                start_value=-14.0,
                end_value=-60.0,
            ),
            # Tail: whisper of echo on the very last tonic.
            AutomationSegment(
                start=TAIL_START,
                end=TAIL_START + 2.0,
                shape="linear",
                start_value=-60.0,
                end_value=-20.0,
            ),
            AutomationSegment(
                start=TAIL_START + 2.0,
                end=TOTAL_DUR,
                shape="linear",
                start_value=-20.0,
                end_value=-60.0,
            ),
        ),
    )

    # ---- The piano voice --------------------------------------------
    score.add_voice(
        "piano",
        synth_defaults=_piano_synth_defaults(),
        effects=_piano_effects(),
        mix_db=0.0,
        # Sparse solo piano has a huge crest factor (spiky attacks, soft
        # decays, long silences).  LUFS normalization gains the quiet
        # body up and clips the attacks; peak normalization preserves
        # the authored dynamic relationships directly.
        normalize_peak_db=-6.0,
        percussive=False,
        velocity_humanize=VelocityHumanizeSpec(
            preset="subtle_living",
            note_jitter=0.025,
            seed=19830105,
        ),
        sympathetic_amount=0.18,
        sympathetic_decay_s=3.0,
        sympathetic_modes=12,
        drift_bus="ensemble",
        drift_bus_correlation=1.0,
        sends=[
            VoiceSend(target="hall", send_db=-9.0),
            VoiceSend(
                target="echo",
                send_db=-60.0,
                automation=[delay_send_automation],
            ),
        ],
    )

    _compose_intro(score)
    _compose_a(score)
    _compose_b(score)
    _compose_aprime(score)
    _compose_tail(score)

    return score


# -- Section composers --------------------------------------------------------


def _compose_intro(score: Score) -> None:
    """Intro: ALPHA stated, a soft gathering response, then a small turn.

    Dynamics: p -- the softest statement of the piece.  Sets the expectation
    that this music breathes and doesn't push.  The response phrase after
    alpha keeps the melodic line alive.
    """
    # ALPHA -- bare statement.
    _place_alpha(
        score,
        "piano",
        start=INTRO_START,
        amp_db_offset=-3.0,
        velocity_scale=0.92,
    )

    # Response phrase at 0:10 -- a rising reply with arcing dynamics:
    # starts soft, peaks at C#4, settles quietly on G#3.
    #
    # Duration variety: short quarter, dotted-quarter, half, whole.
    # Velocity arc: 0.56 -> 0.74 -> 0.84 -> 0.46 (crescendo-decrescendo).
    response_t = INTRO_START + 10.8
    response_notes = (
        (P_F_S, 0.00, 0.45, -17.0, 0.56),  # soft pickup
        (P_A_S, 0.60, 0.90, -15.5, 0.74),  # warmer
        (P_C_S, 1.65, 2.10, -14.0, 0.84),  # peak: reaching
        (P_G_S, 4.00, 3.20, -17.5, 0.46),  # quiet settle
    )
    for partial, t_rel, dur, amp_db, vel in response_notes:
        score.add_note(
            "piano",
            start=response_t + t_rel,
            duration=dur,
            partial=partial,
            amp_db=amp_db,
            velocity=vel,
        )

    # Small inner-voice turn at 0:17 -- two notes under the G#3 tail
    # with a gentle rise in dynamic to signal A's approach.
    # Short grace + longer held note, velocity rising.
    score.add_note(
        "piano",
        start=INTRO_START + 17.4,
        duration=0.40,
        partial=P_D_S_DOWN,  # D#3, staccato grace
        amp_db=-19.5,
        velocity=0.58,
    )
    score.add_note(
        "piano",
        start=INTRO_START + 18.0,
        duration=1.85,
        partial=P_F_S,  # F#3 -- returns us to tonic for A's entry
        amp_db=-17.0,
        velocity=0.72,
    )


def _compose_a(score: Score) -> None:
    """A: ALPHA varied, pedal BETA enters, 7/6 passing tone.

    Densified: between each of alpha's notes and the septimal fragment,
    inner-voice ornaments and a short arpeggiated response keep motion
    alive.  Pedal BETA enters irregularly -- not quarter-notes, more like
    a slow heartbeat -- and continues under the septimal fragment.

    Ends with a small cadential ornament (7/4 -> 2/1) that also lines up
    with the end-of-A delay send opening.
    """
    # ALPHA restatement, slightly fuller dynamic.  Alpha carries its
    # own velocity authoring (see _ALPHA_NOTES).
    _place_alpha(
        score,
        "piano",
        start=A_START,
        amp_db_offset=-1.0,
        velocity_scale=1.0,
    )

    # Inner-voice counter-line during alpha's second half.  Short grace
    # then a longer held 4/3 (B3).  Velocity rises between them to keep
    # the inner line alive as alpha decays.
    score.add_note(
        "piano",
        start=A_START + 3.6,
        duration=0.35,  # staccato grace note
        partial=P_C_S_DOWN,  # C#3
        amp_db=-21.0,
        velocity=0.52,
    )
    score.add_note(
        "piano",
        start=A_START + 4.3,
        duration=2.40,  # sustained under alpha's descent
        partial=P_B,  # B3 (4/3)
        amp_db=-18.5,
        velocity=0.68,
    )

    # Pedal BETA: 4 low pulses, irregular spacing + varied durations.
    # Slow heartbeat, velocities sag then rise as the texture develops.
    pedal_notes = (
        (6.3, 3.2, -26.5, 0.48),  # short soft heartbeat
        (8.8, 2.1, -26.0, 0.52),
        (12.5, 3.8, -25.0, 0.62),  # longer + louder: texture blooms
        (14.7, 1.6, -26.5, 0.50),  # quick pickup
    )
    for offset, dur, amp_db, vel in pedal_notes:
        score.add_note(
            "piano",
            start=A_START + offset,
            duration=dur,
            partial=P_F_S_PEDAL,
            amp_db=amp_db,
            velocity=vel,
        )

    # Connective fragment between alpha's tail and the septimal moment.
    # Uses the diatonic sixth (D#4) to set up the contrast with the 7/6
    # subminor coming next.  Arc: soft -> brighter -> peak -> settle.
    # Mixed durations: eighth, sixteenth-grace, dotted-quarter, half.
    bridge_notes = (
        (P_G_S, 11.2, 0.75, -18.5, 0.60),  # gentle start
        (P_B, 12.1, 0.35, -18.0, 0.72),  # grace note: sudden life
        (P_D_S, 12.6, 1.90, -15.5, 0.82),  # peak: diatonic 6th
        (P_C_S, 14.7, 1.35, -18.0, 0.58),  # decline
    )
    for partial, t_rel, dur, amp_db, vel in bridge_notes:
        score.add_note(
            "piano",
            start=A_START + t_rel,
            duration=dur,
            partial=partial,
            amp_db=amp_db,
            velocity=vel,
        )

    # Septimal fragment: 7/6 passing tone between 5/4 and 9/8.  This is
    # the first septimal moment of the piece.  Velocity dips on the 7/6
    # (the "blue" note) so the ear registers it as color, not weight.
    #   A#3 (5/4) -> A3 (7/6) -> G#3 (9/8) -> F#3 (1/1)
    fragment_start = A_START + 16.4
    fragment_notes = (
        (P_A_S, 0.0, 1.20, -13.0, 0.84),  # accent
        (P_A, 1.30, 1.85, -15.5, 0.58),  # soft septimal shadow
        (P_G_S, 3.25, 0.50, -16.0, 0.66),  # short step
        (P_F_S, 3.85, 2.85, -13.5, 0.74),  # tonic arrival
    )
    for partial, t_rel, dur, amp_db, vel in fragment_notes:
        score.add_note(
            "piano",
            start=fragment_start + t_rel,
            duration=dur,
            partial=partial,
            amp_db=amp_db,
            velocity=vel,
        )

    # C#2 pedal under the septimal fragment -- grounds the new color.
    score.add_note(
        "piano",
        start=A_START + 17.0,
        duration=3.8,
        partial=P_C_S_DOWN,
        amp_db=-23.5,
        velocity=0.56,
    )

    # Arpeggiated response -- rising broken-chord figure through the
    # 7-limit tonic triad + septimal 7th, then returning.  Big velocity
    # arc: soft start, crescendo to the 7/4 apex, decay back down.
    # Durations vary from sixteenth-grace to dotted-quarter.
    #   F#3 -> A#3 -> C#4 -> E4 (apex 7/4) -> C#4 -> A#3 -> G#3
    arp_start = A_START + 21.8
    arp_notes = (
        (P_F_S, 0.00, 0.55, -18.0, 0.58),  # quiet entry
        (P_A_S, 0.70, 0.85, -17.0, 0.66),
        (P_C_S, 1.70, 0.70, -15.5, 0.78),  # building
        (P_E, 2.55, 1.55, -13.5, 0.90),  # septimal apex, held
        (P_C_S, 4.30, 0.40, -17.0, 0.72),  # quick turn
        (P_A_S, 4.80, 0.95, -18.5, 0.60),  # declining
        (P_G_S, 5.90, 2.20, -20.0, 0.44),  # quiet resolution
    )
    for partial, t_rel, dur, amp_db, vel in arp_notes:
        score.add_note(
            "piano",
            start=arp_start + t_rel,
            duration=dur,
            partial=partial,
            amp_db=amp_db,
            velocity=vel,
        )

    # Small inner phrase between arp and cadence -- descending steps.
    # Staccato-ish durations, velocities rise toward the cadence to
    # build anticipation for the 7/4 grace.
    settle_notes = (
        (P_B, 29.4, 0.60, -19.0, 0.56),
        (P_A_S, 30.2, 0.45, -18.5, 0.64),
        (P_G_S, 30.8, 1.30, -17.5, 0.74),
    )
    for partial, t_rel, dur, amp_db, vel in settle_notes:
        score.add_note(
            "piano",
            start=A_START + t_rel,
            duration=dur,
            partial=partial,
            amp_db=amp_db,
            velocity=vel,
        )

    # End-of-A cadential ornament: 7/4 grace -> 2/1 tonic resolution.
    # Lines up with the delay-send open at A_START + 33.0.
    cadence_t = A_START + 33.0
    score.add_note(
        "piano",
        start=cadence_t,
        duration=0.35,  # fast grace
        partial=P_E,  # 7/4 grace note
        amp_db=-15.0,
        velocity=0.78,  # accented
    )
    score.add_note(
        "piano",
        start=cadence_t + 0.40,
        duration=3.20,
        partial=P_F_S_OCT,  # 2/1 octave tonic
        amp_db=-12.0,
        velocity=0.86,  # strong resolution
    )

    # Brief inner response during the cadence tail -- three notes
    # descending back toward F#3 with decrescendo into silence.
    coda_notes = (
        (P_E_S, 34.8, 0.55, -19.0, 0.62),
        (P_D_S, 35.5, 0.80, -20.0, 0.54),
        (P_C_S, 36.5, 1.90, -20.5, 0.42),  # quiet settle into B
    )
    for partial, t_rel, dur, amp_db, vel in coda_notes:
        score.add_note(
            "piano",
            start=A_START + t_rel,
            duration=dur,
            partial=partial,
            amp_db=amp_db,
            velocity=vel,
        )


def _compose_b(score: Score) -> None:
    """B: the quiet climax.  9/7 shift -> 4:5:6:7 chord.

    The section opens with a 9/7 (supermajor 3rd) gesture that pulls the
    tonal center briefly toward 3/2 (C#).  This is the piece's one truly
    unusual interval and it should feel like a door opening.  Then we
    build a voiced 4:5:6:7 chord (F#, A#, C#, E) held over ~4 seconds,
    with a tiny linear_bend on the sustained top note (7/4 E) toward
    exact septimal alignment.

    Dynamics peak at mf here -- roughly 3-4 dB louder than A.
    """
    # Opening gesture: a climbing line through 9/7 toward C#4, with
    # passing tones so the climb feels like walking up a staircase
    # instead of a sudden lift.  Strong crescendo toward the 9/7
    # "opening" moment, then a held 3/2.  Durations span eighth-to-
    # dotted-half.
    #
    #   F#3 (1/1) -> G#3 (9/8) -> A#3 (5/4) -> B3 (4/3)
    #   -> C##3 (9/7) -> C#4 (3/2)
    gesture_notes = (
        (P_F_S, 0.00, 1.20, -14.0, 0.62),  # soft entry
        (P_G_S, 1.30, 0.55, -13.5, 0.70),  # step
        (P_A_S, 2.00, 0.90, -13.0, 0.78),  # building
        (P_B, 3.00, 0.45, -12.5, 0.82),  # accelerating
        (P_C_X, 3.60, 1.95, -11.5, 0.92),  # 9/7 -- "door opens", strong
        (P_C_S, 5.70, 3.00, -12.0, 0.80),  # settles on 3/2, held long
    )
    for partial, t_rel, dur, amp_db, vel in gesture_notes:
        score.add_note(
            "piano",
            start=B_START + t_rel,
            duration=dur,
            partial=partial,
            amp_db=amp_db,
            velocity=vel,
        )

    # Inner-voice arpeggio in the "other side" key -- uses C# as the new
    # local tonic.  Undulating line with velocity breathing: swells on
    # each direction change, softer on passing tones.  Mixed durations.
    reflection_start = B_START + 8.8
    reflection_notes = (
        (P_E, 0.00, 0.70, -14.5, 0.80),  # septimal color, bright
        (P_D_S, 0.85, 1.15, -16.0, 0.58),  # soft passing
        (P_C_S, 2.15, 0.55, -14.5, 0.72),  # quick return
        (P_B, 2.80, 1.45, -16.5, 0.50),  # soft held low
        (P_C_S, 4.35, 0.40, -14.0, 0.78),  # accent
        (P_E_S, 4.85, 1.70, -13.5, 0.86),  # reach up: 15/8
        (P_D_S, 6.65, 0.55, -16.0, 0.60),  # step down
        (P_C_S, 7.30, 2.40, -14.0, 0.72),  # settles on C#4
    )
    for partial, t_rel, dur, amp_db, vel in reflection_notes:
        score.add_note(
            "piano",
            start=reflection_start + t_rel,
            duration=dur,
            partial=partial,
            amp_db=amp_db,
            velocity=vel,
        )

    # Approach to the climax -- descending four-note figure building
    # intensity.  The 7/6 septimal is briefly soft ("one more blue tone
    # before the big arrival") then resolves assertively onto tonic.
    #
    #   A#3 (5/4) -> G#3 (9/8) -> A3 (7/6) -> F#3 (1/1)
    approach_start = B_START + 18.5
    approach_notes = (
        (P_A_S, 0.00, 0.60, -14.0, 0.78),  # accent
        (P_G_S, 0.75, 0.50, -15.0, 0.66),  # step down
        (P_A, 1.40, 1.40, -16.0, 0.56),  # 7/6 septimal: soft
        (P_F_S, 2.90, 1.30, -12.5, 0.88),  # strong tonic arrival
    )
    for partial, t_rel, dur, amp_db, vel in approach_notes:
        score.add_note(
            "piano",
            start=approach_start + t_rel,
            duration=dur,
            partial=partial,
            amp_db=amp_db,
            velocity=vel,
        )

    # The climax: voiced 4:5:6:7 chord, F#-A#-C#-E.  Played as a gentle
    # roll (not simultaneous) so each note has its own attack, and all
    # four ring together for ~4.5 seconds.  Velocity climbs clearly from
    # root to top (7/4) so the septimal sings over the chord.
    chord_start = B_START + 23.0
    chord_notes = (
        (P_F_S, 0.000, 4.80, -12.5, 0.78),  # F#3 (1/1): foundation
        (P_A_S, 0.075, 4.75, -12.5, 0.82),  # A#3 (5/4)
        (P_C_S, 0.150, 4.70, -12.0, 0.88),  # C#4 (3/2)
        (P_E, 0.225, 4.60, -10.5, 0.94),  # E4 (7/4): top, peak
    )
    for idx, (partial, t_rel, dur, amp_db, vel) in enumerate(chord_notes):
        if idx == len(chord_notes) - 1:
            # Top note gets a tiny bend up to +1.5 cents above pure 7/4,
            # audible as a small drift on the sustained top voice.
            bend_target = P_E * 1.000867
            pitch_motion = PitchMotionSpec.linear_bend(target_partial=bend_target)
        else:
            pitch_motion = None
        score.add_note(
            "piano",
            start=chord_start + t_rel,
            duration=dur,
            partial=partial,
            amp_db=amp_db,
            velocity=vel,
            pitch_motion=pitch_motion,
        )

    # Pedal BETA returns under the climax -- low F#1 reinforces the root
    # at a quiet dynamic so it's felt, not heard.
    score.add_note(
        "piano",
        start=B_START + 23.0,
        duration=5.4,
        partial=P_F_S_PEDAL,
        amp_db=-21.0,
        velocity=0.62,
    )

    # Release: chord fades into a soft descending line that walks the
    # ear from C#4 back down toward F#3, preparing A's return.  Mixed
    # durations + clear decrescendo: the piece is exhaling.
    release_notes = (
        (P_C_S, 31.0, 2.60, -17.0, 0.68),  # held C#4 (the 3/2 lingers)
        (P_B, 32.8, 0.85, -19.0, 0.58),  # stepping down
        (P_A_S, 33.8, 1.05, -19.5, 0.52),
        (P_G_S, 34.9, 1.25, -20.5, 0.46),
        (P_F_S, 36.3, 3.40, -19.0, 0.52),  # soft tonic settle
    )
    for partial, t_rel, dur, amp_db, vel in release_notes:
        score.add_note(
            "piano",
            start=B_START + t_rel,
            duration=dur,
            partial=partial,
            amp_db=amp_db,
            velocity=vel,
        )


def _compose_aprime(score: Score) -> None:
    """A': ALPHA restated an octave higher with inner-line accompaniment.

    The octave lift is what makes A' feel like a return rather than a
    repeat -- same shape, new altitude.  Underneath the top line, a
    gentle inner-voice counter-line keeps the section alive.  The 7/4
    cadential ornament before the final tonic is the piece's most
    explicit septimal statement.
    """
    # ALPHA, one octave up (partial_scale=2.0).  Softer than A -- this
    # is the "coming home quietly" restatement.
    _place_alpha(
        score,
        "piano",
        start=APRIME_START,
        partial_scale=2.0,
        amp_db_offset=-4.0,
        velocity_scale=0.88,
    )

    # Inner-voice accompaniment during alpha' -- broken-chord figure in
    # the mid-register, rocking between the tonic triad tones.  Mixed
    # durations (short/long/short) + breathing velocity swells so it
    # feels played, not looped.
    inner_notes = (
        (P_F_S, 0.40, 0.55, -21.0, 0.56),
        (P_A_S, 1.10, 1.40, -19.5, 0.68),  # reach + hold
        (P_C_S, 2.60, 0.45, -21.0, 0.54),  # quick bounce
        (P_A_S, 3.20, 1.05, -20.0, 0.62),
        (P_F_S, 4.45, 1.80, -19.0, 0.72),  # warm hold
        (P_C_S_DOWN, 6.40, 0.50, -22.0, 0.50),
        (P_F_S, 7.10, 0.85, -21.0, 0.58),
        (P_B, 8.20, 1.15, -20.5, 0.60),
        (P_A_S, 9.55, 0.45, -21.5, 0.56),
        (P_G_S, 10.20, 1.30, -20.5, 0.64),
        (P_F_S, 11.80, 1.40, -20.0, 0.70),
    )
    for partial, t_rel, dur, amp_db, vel in inner_notes:
        score.add_note(
            "piano",
            start=APRIME_START + t_rel,
            duration=dur,
            partial=partial,
            amp_db=amp_db,
            velocity=vel,
        )

    # Pre-ornament lift: two notes climbing to the 7/4 grace, so the
    # cadence arrives with direction instead of appearing out of air.
    # Crescendo into the leading tone to set up the cadence.
    pre_orn = (
        (P_D_S, 13.6, 0.70, -18.5, 0.62),
        (P_E_S, 14.5, 1.80, -16.0, 0.80),  # 15/8 leading tone, strong
    )
    for partial, t_rel, dur, amp_db, vel in pre_orn:
        score.add_note(
            "piano",
            start=APRIME_START + t_rel,
            duration=dur,
            partial=partial,
            amp_db=amp_db,
            velocity=vel,
        )

    # 7/4 cadential ornament GAMMA: E5 (7/2) grace -> F#4 (2/1) tonic.
    # Lines up with the delay-send open at APRIME_START + 16.5.
    orn_t = APRIME_START + 16.5
    score.add_note(
        "piano",
        start=orn_t,
        duration=0.40,  # crisp grace
        partial=P_E_UP,  # 7/2 = E5
        amp_db=-15.0,
        velocity=0.80,
    )
    score.add_note(
        "piano",
        start=orn_t + 0.55,
        duration=4.60,
        partial=P_F_S_OCT,  # 2/1 = F#4
        amp_db=-12.5,
        velocity=0.90,  # strongest statement of the final cadence
    )

    # Very faint pedal BETA underneath -- the low F#1 one more time.
    score.add_note(
        "piano",
        start=APRIME_START + 18.0,
        duration=3.5,
        partial=P_F_S_PEDAL,
        amp_db=-27.5,
        velocity=0.54,
    )

    # Descending "farewell" line after the cadence tonic -- walks down
    # from C#4 toward F#3, preparing the tail.  Decrescendo with
    # mixed durations -- the piece exhaling into silence.
    farewell_notes = (
        (P_C_S, 22.3, 1.30, -19.5, 0.60),
        (P_B, 23.9, 0.60, -20.5, 0.48),
        (P_A_S, 24.7, 1.00, -21.0, 0.52),
        (P_G_S, 26.0, 2.70, -21.5, 0.40),  # soft settle into tail
    )
    for partial, t_rel, dur, amp_db, vel in farewell_notes:
        score.add_note(
            "piano",
            start=APRIME_START + t_rel,
            duration=dur,
            partial=partial,
            amp_db=amp_db,
            velocity=vel,
        )


def _compose_tail(score: Score) -> None:
    """Tail: held tonic F#3 with sympathetic + reverb tail carrying it out.

    Dynamics: pp.  The note is mostly a vehicle for the reverb tail; the
    piece ends when the last sample decays to silence.
    """
    score.add_note(
        "piano",
        start=TAIL_START,
        duration=min(TOTAL_DUR - TAIL_START, 18.0),
        partial=P_F_S,
        amp_db=-18.0,
        velocity=0.60,
    )
    # One very quiet echo of pedal F# underneath.
    score.add_note(
        "piano",
        start=TAIL_START + 1.2,
        duration=12.0,
        partial=P_F_S_DOWN,  # F#2
        amp_db=-28.0,
        velocity=0.52,
    )


PIECES: dict[str, PieceDefinition] = {
    "still_window": PieceDefinition(
        name="still_window",
        output_name="60_still_window",
        build_score=build_score,
        # Intimate sparse piano piece -- trying to pull its integrated
        # LUFS up to the project-wide -18 default triggers aggressive
        # limiting that damages the dynamic identity of the piece.
        # -26 LUFS lands naturally even with the denser inner-voice
        # accompaniment, with minimal export makeup gain and no
        # audible limiter pumping.
        export_target_lufs=-26.0,
    ),
}
