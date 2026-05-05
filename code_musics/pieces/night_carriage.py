"""Night Carriage — Burial-side hauntological JI.

Sibling to ``velvet_wall``. Where velvet_wall takes MBV's *Loveless* as its
cue — guitar walls, 7-limit chord smear, distortion-adjacent — this piece
takes Burial's *Untrue* side: urban-hauntological melancholy, slower
motion, 2-step skeleton, vocal-ish pitched ghost smears, tremolo-wobbled
tremolo piano, faint vinyl crackle. Same smear + harmonic_drift
infrastructure; different aesthetic target.

Form: Arrival → Carriage → Fade. ~3:20 duration.

Harmonic language: 11-limit JI centered on F#2 (92.5 Hz). The chord cycle
mixes subminor (6:7:9), neutral-third (11:9), and septimal-seventh (7:4)
colors to reach the Burial melancholy tell without staying in one mode:

  I    — subminor triad + 9th  : [1/1, 7/6, 3/2, 9/4]
  ♭III — utonal inversion       : [6/5, 27/20, 42/25, 9/4]
  VI   — neutral-3rd + sept-7th : [4/3, 44/27, 2/1, 7/3]
  I'   — wide voicing return    : [1/2, 7/6, 3/2, 9/4]

(Second voicing of each chord adjusts to keep voice counts consistent.)

Section I  — Arrival  (0:00–1:05): pads + smear bed, piano motif with
                                   voice-level pitch_wobble, bell chime.
                                   No drums.
Section II — Carriage (1:05–2:35): 2-step kick + ghost snare on 2.5/4,
                                   pitched hats (tracking harmonic 11 of
                                   f0), thickened vocal-ish ghost smear,
                                   vinyl-dust flow-exciter bed.
Section III— Fade     (2:35–3:20): drums drop out over 6s; final I→♭III
                                   glide via progression_drift_lanes;
                                   bell descent; long I tail with
                                   pitch_wobble still breathing.

Voice layout:
  - pad_a:   additive dispersed_pad, primary chord bed, to hall
  - pad_b:   additive smear_drone, sub-layer bed at lower octave, to hall
  - piano:   modal piano (ethereal), voice-level pitch_wobble, dry-ish
  - bell:    FM (septimal mod_ratio), chime stabs, to hall
  - ghost_a/b/c: synth_voice flow_exciter_pad thicken copies
  - dust:    additive with flow_density bandpass bed (vinyl crackle)
  - kick:    drum_voice 808_hiphop (enters in Carriage)
  - snare:   drum_voice brush (ghost strikes on 2.5 and 4)
  - hat:     drum_voice closed_hat tuned via freq override (pitched
             sparkle at harmonic 11 of f0)

Master bus: DEFAULT_MASTER_EFFECTS (preamp → bus glue). Drum bus uses
style="light" — Burial isn't hard glue, it's clean dusty Four-Tet-ish
color on the kit, with the pads carrying all the weight.
"""

from __future__ import annotations

from code_musics.automation import AutomationSegment, AutomationSpec, AutomationTarget
from code_musics.drum_helpers import add_drum_voice, setup_drum_bus
from code_musics.harmonic_drift import (
    ChordVoicingFn,
    drifted_chord_events,
    progression_drift_lanes,
)
from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    VelocityHumanizeSpec,
)
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import (
    EffectSpec,
    NoteEvent,
    Phrase,
    Score,
    SendBusSpec,
    VoiceSend,
)
from code_musics.smear import pitch_wobble, smear_progression, strum, thicken
from code_musics.synth import BRICASTI_IR_DIR

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

F0_HZ: float = 92.5  # F#2 — dark, sits under the voice

# Section boundaries (seconds)
ARRIVAL_START: float = 0.0
CARRIAGE_START: float = 65.0
FADE_START: float = 155.0
PIECE_END: float = 200.0

# 2-step grid (70 BPM)
BPM: float = 70.0
BEAT: float = 60.0 / BPM  # ~0.857s
BAR: float = 4.0 * BEAT  # ~3.43s
S16: float = BEAT / 4.0  # sixteenth

# Chord vocabulary (partials relative to f0).  Voiced as four tones each for
# progression_drift_lanes.
HOME = [1.0, 7 / 6, 3 / 2, 9 / 4]  # subminor triad + 9th
DIM = [6 / 5, 27 / 20, 42 / 25, 9 / 4]  # utonal inversion
NEUTRAL = [4 / 3, 44 / 27, 2.0, 7 / 3]  # neutral-3rd + septimal-7th
HOME_WIDE = [1 / 2, 7 / 6, 3 / 2, 9 / 4]  # root dropped an octave

# Piano motif ratios — sparse, Burial-ish simple descending figure on partials
# that sit inside HOME's outline: 9/4 → 2/1 → 7/4 → 3/2.
PIANO_MOTIF: tuple[tuple[float, float], ...] = (
    (9 / 4, 4.0),
    (2.0, 3.5),
    (7 / 4, 4.0),
    (3 / 2, 6.0),
)


# ---------------------------------------------------------------------------
# Effect factories
# ---------------------------------------------------------------------------


def _hall_reverb() -> EffectSpec:
    """Dark, very wet reverb for the hauntological body."""
    wet_auto = AutomationSpec(
        target=AutomationTarget(kind="control", name="wet"),
        segments=(
            AutomationSegment(
                start=0.0,
                end=CARRIAGE_START,
                shape="linear",
                start_value=0.26,
                end_value=0.32,
            ),
            AutomationSegment(
                start=CARRIAGE_START,
                end=FADE_START,
                shape="linear",
                start_value=0.32,
                end_value=0.30,
            ),
            AutomationSegment(
                start=FADE_START,
                end=PIECE_END,
                shape="linear",
                start_value=0.30,
                end_value=0.36,  # opens up as drums leave
            ),
        ),
    )
    if BRICASTI_IR_DIR.exists():
        return EffectSpec(
            "bricasti",
            {"ir_name": "1 Halls 07 Large & Dark", "wet": 0.26, "lowpass_hz": 4800},
            automation=[wet_auto],
        )
    return EffectSpec(
        "reverb",
        {"room_size": 0.90, "damping": 0.55, "wet_level": 0.26},
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="control", name="wet_level"),
                segments=wet_auto.segments,
            ),
        ],
    )


def _hall_tape_delay() -> EffectSpec:
    """Tape-wandering delay on the hall return — smeared, pitch-drifting."""
    return EffectSpec("mod_delay", {"preset": "tape_wander", "mix": 0.18})


def _hall_warmth() -> EffectSpec:
    """Subtle warmth on the wet return — tube_warm preset at gentle mix.

    An earlier revision dropped this to near-zero because the analysis
    manifest showed a 12%+ "THD delta" from this stage. That was a
    measurement artifact: on the dense Bricasti + mod_delay input, THD
    percentage conflates legitimate harmonic energy (JI partials, reverb
    tail content) with actual distortion. The analysis layer now uses
    IMD-ratio growth instead, which correctly sees this stage as subtle
    warmth rather than heavy distortion.
    """
    return EffectSpec("drive", {"preset": "tube_warm", "drive": 0.25, "mix": 0.20})


def _piano_chorus() -> EffectSpec:
    """Juno-I chorus on piano — Burial-style wobble over the pitch_wobble."""
    return EffectSpec("bbd_chorus", {"preset": "juno_i", "mix": 0.30})


def _hat_top_tame() -> EffectSpec:
    """Gentle top shelf — pitched hat needs less sparkle than a standard hat."""
    return EffectSpec(
        "eq",
        {"bands": [{"kind": "high_shelf", "freq_hz": 8000.0, "gain_db": -2.5}]},
    )


def _master_chain() -> list[EffectSpec]:
    """Master chain: ``DEFAULT_MASTER_EFFECTS`` + a gentle tube_warm drive.

    The tube_warm tail adds warm, lush character on top of the preamp +
    bus-comp pair. An earlier revision removed it after the analysis
    manifest reported post-master "THD" jumping to 180% — that was a
    measurement artifact (the old detector labelled all integer-harmonic
    energy as distortion, which breaks on dense JI content like the
    ghost voices here). The analysis layer now uses IMD-ratio relative
    growth, which correctly reports this chain as warm rather than
    broken, and the audible result is the warmer / more finished tone
    the first revision was going for.
    """
    return [
        *DEFAULT_MASTER_EFFECTS,
        EffectSpec("drive", {"preset": "tube_warm", "mix": 0.20}),
    ]


# ---------------------------------------------------------------------------
# Voice setup
# ---------------------------------------------------------------------------


def _setup_voices(score: Score) -> None:
    hall = VoiceSend(target="hall", send_db=-6.0)
    hall_wet = VoiceSend(target="hall", send_db=-3.0)
    hall_drowned = VoiceSend(target="hall", send_db=0.0)

    # -- pad_a: primary chord bed ---------------------------------------------
    score.add_voice(
        "pad_a",
        synth_defaults={
            "engine": "additive",
            "preset": "dispersed_pad",
            "unison_voices": 3,
            "detune_cents": 14.0,
            "upper_partial_drift_cents": 5.0,
            "attack": 2.5,
            "decay": 1.5,
            "sustain_level": 0.82,
            "release": 3.5,
        },
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble"),
        velocity_group="pads",
        normalize_lufs=-22.0,
        mix_db=0.0,
        pan=0.08,
        sends=[hall],
    )

    # -- pad_b: smear drone beneath, lower octave ------------------------------
    score.add_voice(
        "pad_b",
        synth_defaults={
            "engine": "additive",
            "preset": "smear_drone",
            "unison_voices": 2,
            "detune_cents": 18.0,
            "upper_partial_drift_cents": 6.0,
            "attack": 3.2,
            "decay": 2.0,
            "sustain_level": 0.80,
            "release": 4.5,
        },
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble"),
        velocity_group="pads",
        normalize_lufs=-24.0,
        mix_db=-2.0,
        pan=-0.12,
        sends=[hall],
    )

    # -- piano: modal, with voice-level pitch_wobble ---------------------------
    # The pitch_wobble precludes per-note pitch_motion on this voice; piano
    # notes stay as straight holds and the wobble provides the "out of tune
    # on purpose" Burial character across every held note.
    wobble = pitch_wobble(
        duration=PIECE_END,
        rate_hz=0.12,
        depth_cents=14.0,
        style="smooth",
        seed=42,
    )
    score.add_voice(
        "piano",
        synth_defaults={
            "engine": "piano",
            "preset": "felt",  # soft hammers, darker soundboard — Burial territory
            "decay_base": 5.0,
            "soundboard_color": 0.55,
            "soundboard_brightness": 0.30,
            "attack": 0.003,
            "sustain_level": 1.0,
            "release": 0.4,
        },
        effects=[_piano_chorus()],
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        normalize_lufs=-22.0,
        mix_db=-3.0,
        pan=0.05,
        sends=[VoiceSend(target="hall", send_db=-9.0)],  # stay dry-ish
        automation=[wobble],
    )

    # -- bell: FM chime, septimal modulator ------------------------------------
    score.add_voice(
        "bell",
        synth_defaults={
            "engine": "fm",
            "carrier_ratio": 1.0,
            "mod_ratio": 3.5,  # 7/2 — septimal
            "mod_index": 1.4,
            "index_decay": 0.55,
            "feedback": 0.05,
            "attack": 0.008,
            "decay": 1.2,
            "sustain_level": 0.15,
            "release": 2.0,
        },
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        normalize_lufs=-22.0,
        mix_db=-6.0,
        pan=-0.18,
        sends=[hall_wet],
    )

    # -- ghost_a/b/c: vocal-ish flow-exciter smear, three thicken copies -------
    # Each is a separate voice so thicken copies can pan independently.
    for idx, (name, pan) in enumerate(
        [("ghost_a", -0.35), ("ghost_b", 0.02), ("ghost_c", 0.35)]
    ):
        score.add_voice(
            name,
            synth_defaults={
                "engine": "synth_voice",
                "preset": "flow_exciter_pad",
                "attack": 2.0,
                "decay": 1.5,
                "sustain_level": 0.70,
                "release": 4.0,
            },
            envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
            velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble"),
            velocity_group="ghosts",
            normalize_lufs=-27.0,
            mix_db=-8.0 - idx * 0.5,  # outer copies slightly quieter
            pan=pan,
            sends=[hall_drowned],
        )

    # -- dust: additive bandpass noise bed, flow_density vinyl crackle -------
    # Very low amp, persistent throughout Carriage, not a musical voice.
    score.add_voice(
        "dust",
        synth_defaults={
            "engine": "additive",
            "n_harmonics": 1,  # just the noise band, really
            "noise_mode": "flow",
            "flow_density": 0.22,
            "noise_band_low_hz": 600.0,
            "noise_band_high_hz": 3800.0,
            "noise_to_partial_db": 6.0,  # noise louder than the token partial
            "attack": 0.5,
            "decay": 1.0,
            "sustain_level": 0.85,
            "release": 1.5,
        },
        envelope_humanize=None,  # keep the texture stable
        velocity_humanize=None,
        normalize_lufs=-30.0,
        mix_db=-12.0,
        pan=0.0,
        sends=[VoiceSend(target="hall", send_db=-14.0)],
    )


# ---------------------------------------------------------------------------
# Section I — Arrival (0:00–1:05)
# ---------------------------------------------------------------------------


def _write_arrival(score: Score) -> None:
    # -- Smear_progression across HOME → DIM → NEUTRAL → HOME_WIDE, low octave
    # Four chords, total ~55s, so the progression covers most of Arrival.
    chord_seq = [HOME, DIM, NEUTRAL, HOME_WIDE]
    durs = [14.0, 13.0, 13.0, 15.0]

    smeared = smear_progression(
        chord_seq,
        durs,
        overlap=0.40,
        voice_behavior=["glide", "glide", "reattack", "glide"],
    )

    prog_start = 4.0  # small lead-in silence at the very start
    for phrase in smeared:
        for event in phrase.events:
            if event.partial is None:
                continue
            score.add_note(
                "pad_a",
                partial=event.partial,
                start=event.start + prog_start,
                duration=event.duration,
                amp_db=-11,
                velocity=0.58,
                pitch_motion=event.pitch_motion,
            )

    # pad_b doubles the lowest voice of each chord, one octave down, no smear.
    # This gives a subtle drone without competing with the smear on pad_a.
    pad_b_cursor = prog_start
    for chord, dur in zip(chord_seq, durs, strict=True):
        root = chord[0]
        score.add_note(
            "pad_b",
            partial=root * 0.5,
            start=pad_b_cursor,
            duration=dur + 0.8,  # slight overlap into next
            amp_db=-13,
            velocity=0.52,
            # No pitch_motion — the drone stays put while pad_a smears above.
        )
        pad_b_cursor += dur

    # -- Piano motif: one slow descent across Arrival, sparse ------------------
    # Starts around 10s (lets the pads establish), four notes spanning ~50s.
    # With voice-level pitch_wobble, each held note breathes by ±14¢.
    piano_starts = [10.0, 20.0, 30.0, 44.0]
    for (partial, dur), start_t in zip(PIANO_MOTIF, piano_starts, strict=True):
        score.add_note(
            "piano",
            partial=partial,
            start=start_t,
            duration=dur,
            amp_db=-10,
            velocity=0.62,
        )

    # -- Bell: a single chime stab halfway through Arrival, septimal -----------
    score.add_note(
        "bell",
        partial=7 / 2,  # septimal harmonic
        start=32.0,
        duration=3.0,
        amp_db=-9,
        velocity=0.65,
    )
    score.add_note(
        "bell",
        partial=5 / 2,
        start=48.0,
        duration=2.8,
        amp_db=-10,
        velocity=0.60,
    )

    # -- Strummed chord entry right before Carriage, on pad_a only -------------
    # Soft HOME voicing strummed across 90 ms, overlaps the beat entry at 1:05.
    pre_beat_chord = Phrase(
        events=tuple(
            NoteEvent(start=0.0, duration=8.0, partial=p, amp_db=-11, velocity=0.60)
            for p in HOME
        )
    )
    strummed = strum(pre_beat_chord, spread_ms=90.0, direction="down")
    for event in strummed.events:
        score.add_note(
            "pad_a",
            partial=event.partial,
            start=event.start + 58.0,
            duration=event.duration,
            amp_db=event.amp_db or -11,
            velocity=event.velocity,
        )


# ---------------------------------------------------------------------------
# Section II — Carriage (1:05–2:35)
# ---------------------------------------------------------------------------


def _write_carriage(score: Score) -> None:
    carriage_bars = int((FADE_START - CARRIAGE_START) // BAR)  # ~26 bars

    # -- 2-step drums: kick on 1 and 3, ghost snare on 2.5 and 4 ---------------
    # Skipping kick on every beat would make it too sparse; we go kick-2.5-kick-4
    # to land the 2-step feel. Velocity alternates so the pattern doesn't sound
    # mechanical.
    for bar in range(carriage_bars):
        bar_start = CARRIAGE_START + bar * BAR
        # Kick on 1 and 3
        score.add_note(
            "kick",
            partial=1.0,
            start=bar_start,
            duration=0.5,
            amp_db=-4.0,
            velocity=0.90 if bar % 2 == 0 else 0.85,
        )
        score.add_note(
            "kick",
            partial=1.0,
            start=bar_start + 2.0 * BEAT,
            duration=0.5,
            amp_db=-4.0,
            velocity=0.82,
        )
        # Snare ghost on 2.5 (between beats 2 and 3) and 4
        score.add_note(
            "snare",
            partial=1.0,
            start=bar_start + 1.5 * BEAT,
            duration=0.3,
            amp_db=-12.0,
            velocity=0.42,
        )
        score.add_note(
            "snare",
            partial=1.0,
            start=bar_start + 3.0 * BEAT,
            duration=0.3,
            amp_db=-9.0,
            velocity=0.72,
        )

        # Pitched hat: sparse 8ths, with the "covert pitched shimmer" at
        # harmonic 11 of f0 (~1017 Hz for F#2=92.5).  We get this by setting
        # `freq` explicitly on each hat event.  Three hits per bar — not 8ths
        # or 16ths, deliberately spare for the 2-step feel.
        for beat_idx in [0, 1.5, 2.5, 3.5]:
            score.add_note(
                "hat",
                freq=F0_HZ * 11.0,
                start=bar_start + beat_idx * BEAT,
                duration=0.12,
                amp_db=-14.0,
                velocity=0.48 + (0.15 if beat_idx in (1.5, 3.5) else 0.0),
            )

    # -- Pad bed continues through Carriage, same progression restated ---------
    # Second pass: higher voicing, slightly more movement.
    chord_seq_carriage = [HOME, NEUTRAL, DIM, HOME]  # different order
    durs_carriage = [22.0, 22.0, 22.0, 24.0]  # total 90s

    smeared_carriage = smear_progression(
        chord_seq_carriage,
        durs_carriage,
        overlap=0.35,
        voice_behavior=["glide", "glide", "glide", "reattack"],
    )
    for phrase in smeared_carriage:
        for event in phrase.events:
            if event.partial is None:
                continue
            score.add_note(
                "pad_a",
                partial=event.partial,
                start=event.start + CARRIAGE_START,
                duration=event.duration,
                amp_db=-10,
                velocity=0.60,
                pitch_motion=event.pitch_motion,
            )

    # pad_b restated at low octave, slight offset for interference
    cursor = CARRIAGE_START + 0.4
    for chord, dur in zip(chord_seq_carriage, durs_carriage, strict=True):
        score.add_note(
            "pad_b",
            partial=chord[0] * 0.5,
            start=cursor,
            duration=dur + 1.0,
            amp_db=-12,
            velocity=0.50,
        )
        cursor += dur

    # -- Piano motif restated, slightly faster, transposed up 7/6 --------------
    piano_start_carriage = CARRIAGE_START + 12.0
    for i, (partial, dur) in enumerate(PIANO_MOTIF):
        score.add_note(
            "piano",
            partial=partial * (7 / 6),  # transpose up a subminor third
            start=piano_start_carriage + i * 10.0,
            duration=dur * 0.85,
            amp_db=-11,
            velocity=0.60,
        )

    # Second piano phrase near end of Carriage — back to original pitches
    piano_start_2 = CARRIAGE_START + 56.0
    for i, (partial, dur) in enumerate(PIANO_MOTIF):
        score.add_note(
            "piano",
            partial=partial,
            start=piano_start_2 + i * 7.5,
            duration=dur * 0.75,
            amp_db=-12,
            velocity=0.56,
        )

    # -- Ghost smear via thicken: a simple vocal-ish 3-note phrase -------------
    # The phrase holds three chord tones from HOME (3/2, 7/4, 2/1 high) and
    # we thicken n=3 across ghost_a/b/c with moderate detune + spread.
    ghost_phrase = Phrase(
        events=(
            NoteEvent(start=0.0, duration=8.0, partial=3 / 2, amp_db=-8, velocity=0.58),
            NoteEvent(
                start=6.0, duration=10.0, partial=7 / 4, amp_db=-7, velocity=0.62
            ),
            NoteEvent(start=13.0, duration=12.0, partial=2.0, amp_db=-8, velocity=0.60),
        )
    )
    ghost_copies = thicken(
        ghost_phrase,
        n=3,
        detune_cents=6.0,
        spread_ms=18.0,
        stereo_width=0.8,
        seed=11,
    )
    # Dispatch each copy to its own ghost voice.  Ghost_b gets the center
    # (unison-ish), _a and _c get the detune offsets.
    ghost_voice_names = ["ghost_a", "ghost_b", "ghost_c"]
    ghost_entry_offsets = [CARRIAGE_START + 16.0, CARRIAGE_START + 48.0]
    for offset in ghost_entry_offsets:
        for voice_name, copy in zip(ghost_voice_names, ghost_copies, strict=True):
            for event in copy.phrase.events:
                score.add_note(
                    voice_name,
                    partial=event.partial,
                    start=event.start + offset,
                    duration=event.duration,
                    amp_db=(event.amp_db or -8.0) + copy.amp_offset_db,
                    velocity=event.velocity,
                )

    # -- Vinyl dust bed: continuous through Carriage ---------------------------
    # One long note on the dust voice — the envelope shape is delivered by
    # flow_exciter itself, not by gating.
    score.add_note(
        "dust",
        partial=1.0,  # nominal; the voice is really noise-based
        start=CARRIAGE_START,
        duration=FADE_START - CARRIAGE_START,
        amp_db=-18,
        velocity=0.55,
    )


# ---------------------------------------------------------------------------
# Section III — Fade (2:35–3:20)
# ---------------------------------------------------------------------------


def _write_fade(score: Score) -> None:
    # -- Drums taper out over the first ~6 seconds of Fade ---------------------
    # Three more kick+snare hits, each quieter than the last, then silence.
    for i, (offset, kick_db, snare_db) in enumerate(
        [(0.0, -5.0, -11.0), (2.0, -7.0, -13.0), (4.0, -11.0, -16.0)]
    ):
        score.add_note(
            "kick",
            partial=1.0,
            start=FADE_START + offset,
            duration=0.5,
            amp_db=kick_db,
            velocity=0.75 - i * 0.15,
        )
        score.add_note(
            "snare",
            partial=1.0,
            start=FADE_START + offset + BEAT,
            duration=0.3,
            amp_db=snare_db,
            velocity=0.55 - i * 0.12,
        )
        # One last hat on each step
        score.add_note(
            "hat",
            freq=F0_HZ * 11.0,
            start=FADE_START + offset + 0.5 * BEAT,
            duration=0.10,
            amp_db=-16.0 - i * 2.0,
            velocity=0.45,
        )

    # -- Final pad progression with the actual drift: I → DIM glide ------------
    # Two chords; drift lanes on the transition only.  Built via
    # drifted_chord_events so the chord tones carry per-note automation lanes.

    fade_chord_dur = 18.0
    progression_callables: list[ChordVoicingFn] = [
        lambda: [(p, -10.0) for p in HOME],
        lambda: [(p, -11.0) for p in DIM],
    ]
    drift_lanes = progression_drift_lanes(
        progression_callables,
        chord_dur=fade_chord_dur,
        attraction=0.6,
        wander=0.25,
        smoothness=0.88,
        target_time=fade_chord_dur * 0.85,
        glide_transitions={0},  # only the I → DIM transition glides
        seed_base=7,
    )

    fade_chord_start = FADE_START + 6.0
    cursor = fade_chord_start
    for chord_idx, (chord_fn, lanes) in enumerate(
        zip(progression_callables, drift_lanes, strict=True)
    ):
        events = drifted_chord_events(
            chord_fn(),
            duration=fade_chord_dur if chord_idx == 0 else PIECE_END - cursor,
            drift_lanes=lanes,
        )
        for ev in events:
            score.add_note(
                "pad_a",
                partial=ev.partial,
                start=cursor,
                duration=ev.duration,
                amp_db=ev.amp_db,
                velocity=0.55,
                automation=list(ev.automation) if ev.automation else None,
            )
        cursor += fade_chord_dur

    # pad_b: sustain a low root through Fade, with a final octave drop
    score.add_note(
        "pad_b",
        partial=0.5,
        start=fade_chord_start,
        duration=PIECE_END - fade_chord_start,
        amp_db=-13,
        velocity=0.48,
    )

    # -- Bell: final three-note descent ----------------------------------------
    for i, (partial, start_offset) in enumerate(
        [(7 / 2, 4.0), (3.0, 12.0), (5 / 2, 22.0)]
    ):
        score.add_note(
            "bell",
            partial=partial,
            start=FADE_START + start_offset,
            duration=3.5 - i * 0.5,
            amp_db=-9 - i * 1.5,
            velocity=0.62 - i * 0.08,
        )

    # -- Piano: one final note held into the tail ------------------------------
    score.add_note(
        "piano",
        partial=3 / 2,
        start=FADE_START + 18.0,
        duration=PIECE_END - (FADE_START + 18.0) - 2.0,
        amp_db=-12,
        velocity=0.55,
    )

    # -- Ghost smear: one final phrase, very drowned in reverb -----------------
    final_ghost = Phrase(
        events=(
            NoteEvent(
                start=0.0, duration=14.0, partial=7 / 4, amp_db=-10, velocity=0.45
            ),
            NoteEvent(start=10.0, duration=16.0, partial=2.0, amp_db=-9, velocity=0.50),
        )
    )
    final_copies = thicken(
        final_ghost,
        n=3,
        detune_cents=5.0,
        spread_ms=22.0,
        stereo_width=0.85,
        seed=99,
    )
    for voice_name, copy in zip(
        ["ghost_a", "ghost_b", "ghost_c"], final_copies, strict=True
    ):
        for event in copy.phrase.events:
            score.add_note(
                voice_name,
                partial=event.partial,
                start=event.start + FADE_START + 8.0,
                duration=event.duration,
                amp_db=(event.amp_db or -9.0) + copy.amp_offset_db,
                velocity=event.velocity,
            )


# ---------------------------------------------------------------------------
# Drum voice setup
# ---------------------------------------------------------------------------


def _setup_drums(score: Score) -> None:
    """Register drum bus + drum voices.  Called from ``build_score()`` after
    the tonal voices are up."""
    drum_bus = setup_drum_bus(score, style="light")

    add_drum_voice(
        score,
        "kick",
        engine="drum_voice",
        preset="808_hiphop",
        drum_bus=drum_bus,
        send_db=-3.0,
        mix_db=-1.0,
        # Skip the preset-default kick preamp — it adds crunchy 2-5 kHz lift on
        # this piece (2.24 dB high-band delta with IMD ratio ~23 at full mix).
        # The drum bus's `light` compressor + gentle drive already glues the
        # kick; we want clean body here, not "weighted" character.
        effects=[EffectSpec("compressor", {"preset": "kick_glue"})],
    )
    # Tiny wash into hall — barely audible but ties the kick to the space.
    score.voices["kick"].sends.append(VoiceSend(target="hall", send_db=-18.0))

    add_drum_voice(
        score,
        "snare",
        engine="drum_voice",
        preset="brush",
        drum_bus=drum_bus,
        send_db=-5.0,
        mix_db=-4.0,
        pan=0.08,
    )
    score.voices["snare"].sends.append(VoiceSend(target="hall", send_db=-8.0))

    add_drum_voice(
        score,
        "hat",
        engine="drum_voice",
        preset="closed_hat",
        drum_bus=drum_bus,
        send_db=-6.0,
        choke_group="hats",
        effects=[
            EffectSpec("compressor", {"preset": "hat_control"}),
            _hat_top_tame(),
        ],
        mix_db=-9.0,
        pan=-0.22,
    )
    score.voices["hat"].sends.append(VoiceSend(target="hall", send_db=-10.0))


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
                effects=[_hall_reverb(), _hall_tape_delay(), _hall_warmth()],
            ),
        ],
        master_effects=_master_chain(),
    )

    _setup_voices(score)
    _setup_drums(score)
    _write_arrival(score)
    _write_carriage(score)
    _write_fade(score)

    return score


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

PIECES: dict[str, PieceDefinition] = {
    "night_carriage": PieceDefinition(
        name="night_carriage",
        output_name="night_carriage",
        build_score=build_score,
        sections=(
            PieceSection(
                label="Arrival",
                start_seconds=ARRIVAL_START,
                end_seconds=CARRIAGE_START,
            ),
            PieceSection(
                label="Carriage",
                start_seconds=CARRIAGE_START,
                end_seconds=FADE_START,
            ),
            PieceSection(
                label="Fade",
                start_seconds=FADE_START,
                end_seconds=PIECE_END,
            ),
        ),
    ),
}
