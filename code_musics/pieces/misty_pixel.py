"""misty_pixel — Tuss-dense polymeter IDM in 7-limit JI at F#3.

Aesthetic anchors: Girl/Boy Song (strings + impossible drums), Yellow Calx
(ticky glitch + murky warmth), Mt Saint Michel (delicacy vs. speed), The Tuss
(acid + complex breakbeat + hook).  The piece keeps kick + snare + bell
locked to the 16-beat bar while three satellite layers phase at 11 / 9 / 7
beats, and an EP counter-line phases at 20 beats against the bell's 16.
Nothing stays random: bell is a real melody, pad is a real progression,
drums feel programmed and busy rather than noisy.

Sections (BPM 130, BEAT ≈ 0.462 s, BAR ≈ 1.846 s):

  A Intro       (bars 0-8,   ~0:00-0:15)  sparse glitch + pad swell
  B Groove in   (bars 8-24,  ~0:15-0:44)  drums lock, bass, bell statement
  C Counter     (bars 24-40, ~0:44-1:14)  EP enters (20-beat phasing)
  D Break       (bars 40-48, ~1:14-1:29)  drums thin, pad + bell breathe
  E Peak        (bars 48-72, ~1:29-2:13)  full arrangement, acid bass, max polymeter
  F Resolution  (bars 72-84, ~2:13-2:35)  strip, bell final statement
  G Tail        (bars 84-88, ~2:35-2:43)  pad only, hall bloom
"""

from __future__ import annotations

from collections.abc import Sequence

from code_musics.automation import (
    AutomationSegment,
    AutomationSpec,
    AutomationTarget,
)
from code_musics.composition import (
    polymeter_alignment,
    polymeter_layer,
)
from code_musics.drum_helpers import add_drum_voice, setup_drum_bus
from code_musics.generative.ca_rhythm import ca_rhythm
from code_musics.generative.euclidean import euclidean_pattern
from code_musics.generative.turing import TuringMachine
from code_musics.modulation import MacroSource, ModConnection
from code_musics.pieces._shared import SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import EffectSpec, NoteEvent, Phrase, Score, VoiceSend
from code_musics.synth import amp_to_db

# ---------------------------------------------------------------------------
# Time grid
# ---------------------------------------------------------------------------

BPM = 130.0
BEAT = 60.0 / BPM
BAR = 4.0 * BEAT
S16 = BEAT / 4.0

S_A = 0  # intro
S_B = 8  # groove enters
S_C = 24  # counter enters
S_D = 40  # break
S_E = 48  # peak
S_F = 72  # resolution
S_G = 84  # tail
TOTAL_BARS = 88

T_A = S_A * BAR
T_B = S_B * BAR
T_C = S_C * BAR
T_D = S_D * BAR
T_E = S_E * BAR
T_F = S_F * BAR
T_G = S_G * BAR
TOTAL_DUR = TOTAL_BARS * BAR

F0_HZ = 185.0  # F#3 — shared tonic with clock_of_7 / newton_bloom
KICK_PARTIAL = 0.25  # F#1 sub

# Polymeter reality check — compile-time assertions so the form commits to
# non-trivial phase relationships.  The satellites never realign with the
# 16-beat grid inside the piece; that is the point.
assert polymeter_alignment([16, 11, 9, 7]) == 11088.0
assert polymeter_alignment([16, 20]) == 80.0  # EP vs bell realigns every 5 bars


# ---------------------------------------------------------------------------
# Harmonic material — 7-limit JI progression in F# minor flavor
# ---------------------------------------------------------------------------

# 4-chord progression over 8 bars (2 bars per chord), repeated through the piece.
# Rooted on 1, 7/6 (septimal m3), 3/2, 4/3 — bittersweet septimal minor with a
# subdominant resolution.  Each chord is a four-voice stack (root, 3rd, 5th, 7th)
# anchored above F#3.
_PAD_CHORDS: tuple[tuple[float, ...], ...] = (
    (1.0, 7 / 6, 3 / 2, 7 / 4),  # i7 (septimal minor 7)
    (7 / 6, 7 / 5, 5 / 3, 2.0),  # bIII-ish
    (3 / 2, 7 / 4, 9 / 4, 21 / 8),  # v7
    (4 / 3, 5 / 3, 2.0, 7 / 3),  # iv
)

# Bell melody — a 16-beat phrase (one bar) that phrases against the 16-beat
# kick bar.  Notes are in 16th positions keyed to the pad chord.  The phrase
# is hummable: a 4-note rising kernel, a hold, a turn, a descending tail.
# Positions are (step_in_bar_0_15, partial, duration_16ths, velocity).
_BELL_PHRASE: tuple[tuple[int, float, int, float], ...] = (
    (0, 2.0, 2, 1.0),  # strong downbeat — octave above F#
    (2, 7 / 4, 2, 0.85),  # septimal 7
    (4, 9 / 4, 3, 1.0),  # M9 above root — bittersweet high
    (7, 2.0, 1, 0.8),
    (10, 7 / 4, 2, 0.85),
    (12, 5 / 3, 2, 0.9),  # 5/3 — major 6
    (14, 3 / 2, 2, 0.75),  # descending into pad
)

# Variation of bell phrase for C/E sections — same rhythm, reharmonizes higher.
_BELL_PHRASE_HIGH: tuple[tuple[int, float, int, float], ...] = (
    (0, 8 / 3, 2, 1.0),  # 8/3 — major 10th
    (2, 7 / 3, 2, 0.85),
    (4, 3.0, 3, 1.0),  # 3x fundamental — octave+5th
    (7, 8 / 3, 1, 0.8),
    (10, 7 / 3, 2, 0.85),
    (12, 2.0, 2, 0.9),
    (14, 7 / 4, 2, 0.75),
)

# EP counter-line — 20-beat cycle (5 bars of 4/4).  Lower than bell, voiced
# as a stepwise JI descent to give harmonic motion inside the phasing.
# Positions are (step_in_20_beat_cycle_quarters_0_19, partial, duration_beats, vel).
# Kept deliberately simple to survive the phase clash with bell.
_EP_PHRASE: tuple[tuple[float, float, float, float], ...] = (
    (0.0, 3 / 2, 4.0, 0.7),
    (4.0, 4 / 3, 4.0, 0.7),
    (8.0, 7 / 6, 4.0, 0.7),
    (12.0, 9 / 8, 4.0, 0.7),
    (16.0, 1.0, 4.0, 0.7),
)

# Bass cycles — three variations.  All are 8 beats (32 sixteenths). Partials
# are sub-octave (below F0).  None = rest.  Accented beats get higher velocity
# and ratio_glide on beat 1 of the cycle.

# V1 — "simple": B section.  Root-walking, sparse, accent on 1.
_BASS_V1_SIMPLE: tuple[float | None, ...] = (
    0.5,
    None,
    None,
    0.5,  # beat 1 (root F#), pickup to beat 2
    None,
    0.5,
    None,
    None,  # beat 2
    7 / 12,
    None,
    None,
    0.5,  # beat 3 (septimal m3), back to root
    None,
    2 / 3,
    None,
    None,  # beat 4 (5th)
    0.5,
    None,
    None,
    0.5,
    None,
    0.5,
    None,
    None,
    7 / 12,
    None,
    None,
    0.5,
    None,
    2 / 3,
    None,
    None,
)

# V2 — "walking": C section.  More movement; chromatic-style walk that resolves
# back.  Every 16th has a shot at an onset so the line moves.
_BASS_V2_WALK: tuple[float | None, ...] = (
    0.5,
    None,
    0.5,
    None,  # 1, 1.5
    7 / 12,
    None,
    0.5,
    None,  # 2 (m3), 2.5
    2 / 3,
    None,
    7 / 12,
    None,  # 3 (5th), 3.5 (m3)
    0.5,
    0.5,
    None,
    None,  # 4, 4.5
    0.5,
    None,
    0.5,
    None,
    7 / 12,
    None,
    2 / 3,
    None,
    3 / 4,
    None,
    2 / 3,
    None,  # 7 (M3) resolving to 5
    0.5,
    None,
    0.5,
    None,
)

# V3 — "peak": E section.  16th grid fully active, busiest pattern, accents
# every 4 steps so the kick still rides.  Includes a 7/8 stab (lower-octave
# septimal 7th) for that Tuss-y harmonic spice.
_BASS_V3_PEAK: tuple[float | None, ...] = (
    0.5,
    0.5,
    None,
    0.5,
    7 / 12,
    None,
    0.5,
    7 / 12,
    2 / 3,
    None,
    7 / 12,
    0.5,
    7 / 16,
    None,
    0.5,
    None,  # 7/16 = lower-oct septimal 7
    0.5,
    0.5,
    None,
    0.5,
    7 / 12,
    None,
    0.5,
    None,
    3 / 4,
    None,
    2 / 3,
    7 / 12,
    0.5,
    7 / 12,
    0.5,
    None,
)

# Closed-hat pattern — euclidean(7, 11) onsets across 11 sixteenths.
_HAT_11_PATTERN = euclidean_pattern(7, 11, rotation=0)

# Glitch perc — Turing machine over 9 16th-steps with small flip probability.
# Amp pool = 2 levels; pitch pool = 2 metallic partial ratios for flavor.
_GLITCH_TURING = TuringMachine(
    length=9,
    flip_probability=0.04,
    tones=[1.0, 2.0],  # only two distinct states; used as ratio pool for perc
    seed=17,
)

# Tick CA — rule 90 over 7 steps for a sparse, interestingly chattery pattern.
_TICK_CA = ca_rhythm(rule=90, steps=7, span=S16, seed=3)


# ---------------------------------------------------------------------------
# Polymeter phrase builders (single-cycle source phrases, then tiled via
# polymeter_layer over the active window of each section)
# ---------------------------------------------------------------------------


def _one_cycle_hat_phrase() -> Phrase:
    """Build a 1-cycle phrase for the closed_hat at 11 sixteenths."""
    events: list[NoteEvent] = []
    for step, on in enumerate(_HAT_11_PATTERN):
        if not on:
            continue
        events.append(
            NoteEvent(
                start=step * S16,
                duration=S16 * 0.6,
                partial=1.0,
                amp_db=-10.0,
                velocity=0.85 if step == 0 else 0.7,
            )
        )
    return Phrase(events=tuple(events))


def _one_cycle_glitch_phrase() -> Phrase:
    """Build a 1-cycle phrase for glitch_perc: 9 steps, half of them sounding.

    The Turing machine supplies a deterministic per-step value pattern; we
    translate odd/even outputs into hit/rest, and use pattern variation
    to pick the voice character (pitched up vs down).
    """
    values = _GLITCH_TURING.generate(9)
    events: list[NoteEvent] = []
    for step, value in enumerate(values):
        # Use value > 1.5 as the hit predicate — roughly half-density.
        if value < 1.5:
            continue
        # Slight pitch offset on alternating hits for a "two-voice" feel.
        partial = 1.0 if step % 2 == 0 else 0.5
        events.append(
            NoteEvent(
                start=step * S16,
                duration=S16 * 0.45,
                partial=partial,
                amp_db=-12.0 if step == 0 else -14.0,
                velocity=0.9 if step == 0 else 0.75,
            )
        )
    # Safety: if the Turing register by bad luck produced all rests for this
    # seed, force a single hit on step 0 so the layer is not silent.
    if not events:
        events.append(
            NoteEvent(start=0.0, duration=S16 * 0.45, partial=1.0, amp_db=-14.0)
        )
    return Phrase(events=tuple(events))


def _one_cycle_tick_phrase() -> Phrase:
    """Build a 1-cycle phrase for ticks: 7-step CA pattern as onsets."""
    # _TICK_CA.spans already has the rest-absorbed hits.  Convert each span
    # into an onset (starts accumulate), dropping the trailing last span
    # duration since it's "how long the final hit rings", not onset spacing.
    spans = _TICK_CA.spans
    events: list[NoteEvent] = []
    cursor = 0.0
    for index, span in enumerate(spans):
        events.append(
            NoteEvent(
                start=cursor,
                duration=min(S16 * 0.5, span * 0.7),
                partial=1.0,
                amp_db=-15.0 if index > 0 else -12.0,
                velocity=0.85 if index == 0 else 0.65,
            )
        )
        cursor += span
    # Ensure phrase fits inside one 7-beat cycle of 7 * S16 = 7/16 beat?  No —
    # tick cycle is "7 beats" in the plan's phasing, not 7 sixteenths.  We
    # want the ticks to phase at a cycle of 7 BEATS against the 16-beat bar.
    # So we scale up: CA has 7 steps at S16 spacing = 7/4 beats = 0.4375 bars.
    # That's too fast — we instead use 7 beats as cycle length, and repeat
    # the 7-step CA 4x inside it (28 ticks).  Done at phrase-build time: we
    # return the spans as-is and let the caller set cycle=7*BEAT (roughly 4
    # inner repetitions).  But polymeter_layer tiles the phrase, so the
    # inner spacing of 7*S16 stays.  We instead extend the phrase here to
    # cover a full 7-beat cycle by concatenation.
    # Simpler: return as-is with span = 7 * S16, and set tick cycle_seconds
    # = 7 * S16 in the caller (that's 7 sixteenths ≈ 0.202 s).  This gives
    # a dense, chattery tick — classic Aphex foreground perc.  The 7 vs 16
    # phasing is then 7/16 vs 16/16 at the SAME grid, which is still a
    # satisfying phase relationship, just faster than 7 beats.
    return Phrase(events=tuple(events))


def _one_cycle_bell_phrase(high: bool = False) -> Phrase:
    """Build the 16-beat bell phrase from the step/partial/dur/vel table."""
    source = _BELL_PHRASE_HIGH if high else _BELL_PHRASE
    events: list[NoteEvent] = []
    for step, partial, dur16, vel in source:
        events.append(
            NoteEvent(
                start=step * S16,
                duration=dur16 * S16 * 0.95,
                partial=partial,
                amp_db=-6.0,
                velocity=vel,
            )
        )
    return Phrase(events=tuple(events))


def _one_cycle_ep_phrase() -> Phrase:
    """Build the 20-beat EP counter-line (5 bars of 4/4, one note per bar)."""
    events: list[NoteEvent] = []
    for step_beats, partial, dur_beats, vel in _EP_PHRASE:
        events.append(
            NoteEvent(
                start=step_beats * BEAT,
                duration=dur_beats * BEAT * 0.9,
                partial=partial,
                amp_db=-12.0,
                velocity=vel,
            )
        )
    return Phrase(events=tuple(events))


def _one_cycle_bass_phrase(
    pattern: tuple[float | None, ...] = _BASS_V1_SIMPLE,
) -> Phrase:
    """Build an 8-beat bass phrase (one note per sounding 16th) from a pattern.

    Pattern is 32 16ths (8 beats, which is 2 bars of 4/4).  Accents fall on
    beats 1, 3, 5, 7 (sixteenth positions 0, 8, 16, 24).
    """
    events: list[NoteEvent] = []
    accent_positions = {0, 8, 16, 24}
    for step, partial in enumerate(pattern):
        if partial is None:
            continue
        motion = None
        # Ratio-glide into the top-of-cycle accent from the last-sounding note.
        if step == 0:
            motion = PitchMotionSpec.ratio_glide(
                start_ratio=(2 / 3) / 0.5, end_ratio=1.0
            )
        is_accent = step in accent_positions
        if step == 0:
            amp_db, velocity = -4.0, 1.08
        elif is_accent:
            amp_db, velocity = -6.5, 1.0
        else:
            amp_db, velocity = -9.5, 0.88
        events.append(
            NoteEvent(
                start=step * S16,
                duration=S16 * 0.85,
                partial=partial,
                amp_db=amp_db,
                velocity=velocity,
                pitch_motion=motion,
            )
        )
    return Phrase(events=tuple(events))


# ---------------------------------------------------------------------------
# Drum note writers (lock-to-bar voices: kick, snare, clap)
# ---------------------------------------------------------------------------


def _add_kick(score: Score) -> None:
    """Steady kick with section-sensitive density + a Tuss-style flam every 8 bars."""
    for bar in range(TOTAL_BARS):
        # A: no kick yet.  B onward: 4-on-floor.  D: drop first two bars.
        if bar < S_B:
            continue
        if S_D <= bar < S_D + 4:
            # Break: drop kick entirely for 4 bars, then 2 solo downbeats
            continue
        if bar >= S_G:
            continue  # tail: no kick
        # F (resolution): thin out
        if S_F <= bar < S_G:
            if bar % 2 == 0:
                score.add_note(
                    "kick",
                    start=bar * BAR,
                    duration=0.25,
                    partial=KICK_PARTIAL,
                    amp_db=-4.0,
                    velocity=0.95,
                )
            continue
        # Standard 4-on-floor
        for beat in range(4):
            score.add_note(
                "kick",
                start=bar * BAR + beat * BEAT,
                duration=0.25,
                partial=KICK_PARTIAL,
                amp_db=-3.0 if beat == 0 else -5.0,
                velocity=1.0 if beat == 0 else 0.9,
            )
        # Tuss-style flam every 8 bars just before the "1" — insert 2 grace
        # hits at 15th + 15.5th 16th positions of the bar.
        if bar > 0 and (bar % 8) == 7 and S_B <= bar < S_D:
            for grace_16th, grace_db in ((15.0, -9.0), (15.5, -7.0)):
                score.add_note(
                    "kick",
                    start=bar * BAR + grace_16th * S16,
                    duration=0.08,
                    partial=KICK_PARTIAL,
                    amp_db=grace_db,
                    velocity=0.85,
                )
    # Break ending downbeats
    for bar in (S_D + 4, S_D + 6):
        score.add_note(
            "kick",
            start=bar * BAR,
            duration=0.25,
            partial=KICK_PARTIAL,
            amp_db=-5.0,
            velocity=0.95,
        )


def _snare_rhythm_for_bar(bar: int) -> Sequence[tuple[float, float, float]]:
    """Return (step_16th, amp_db, velocity) tuples for snare hits in one bar.

    Backbeat on 4 and 12.  Ghost notes filled in by mutation pass at density
    that grows with section intensity.  The ghost set is a deterministic
    function of bar index, so the piece is seed-stable across renders.
    """
    hits: list[tuple[float, float, float]] = []
    # Backbeat
    hits.append((4.0, -5.0, 1.0))
    hits.append((12.0, -5.0, 1.0))

    # Section-sensitive ghost density
    if bar < S_B:
        return hits
    if S_D <= bar < S_D + 6:
        return hits  # break: backbeat only
    # Ghost positions cycle through a small library, picked by bar index.
    ghost_libraries: tuple[tuple[float, ...], ...] = (
        (6.5, 14.5),
        (2.0, 10.5, 15.0),
        (6.5, 11.0, 13.5),
        (3.5, 6.75, 14.75),
    )
    library = ghost_libraries[bar % len(ghost_libraries)]
    # E section: push into Tuss-dense territory — double up ghosts.
    extra: tuple[float, ...] = ()
    if S_E <= bar < S_F:
        extra = (10.25, 13.75)
    for step in tuple(library) + extra:
        hits.append((step, -20.0, 0.5))
    return hits


def _add_snare(score: Score) -> None:
    """Snare backbeat + mutating ghost swarms."""
    for bar in range(TOTAL_BARS):
        if bar < S_B:
            continue
        if bar >= S_G:
            continue
        for step_16th, amp_db, vel in _snare_rhythm_for_bar(bar):
            score.add_note(
                "snare",
                start=bar * BAR + step_16th * S16,
                duration=S16 * 0.6,
                partial=1.0,
                amp_db=amp_db,
                velocity=vel,
            )


def _add_clap(score: Score) -> None:
    """Clap on beat 3, skipping every 5th bar — Aphex 'wrong clock' breathe."""
    for bar in range(TOTAL_BARS):
        if bar < S_B + 4:  # clap enters a bit after drums lock
            continue
        if S_D <= bar < S_D + 6:
            continue  # break
        if bar >= S_G:
            continue
        if S_F <= bar < S_G and bar % 2 == 1:
            continue  # thin during F
        if bar % 5 == 4:
            continue  # the 5-bar skip
        score.add_note(
            "clap",
            start=bar * BAR + 2 * BEAT,
            duration=0.18,
            partial=1.0,
            amp_db=-6.0,
            velocity=1.0,
        )


def _add_crash(score: Score) -> None:
    """Section-entry crashes on C, E, and a long tail hit near G.

    Crashes mark form boundaries — C (counter enters), E (peak), and a
    final decay hit at the start of G that rings through the fade.
    """
    # Bar, velocity, amp_db — pre-entry hits placed on the "and" of beat 4
    # of the last bar of the preceding section for the most typical
    # drum-machine swell feel.
    hits: tuple[tuple[float, float, float], ...] = (
        (S_C - 1 + 3.5 / 4, 1.0, -10.0),  # pickup into C
        (S_E - 1 + 3.5 / 4, 1.05, -7.0),  # pickup into E (big one)
        (S_E + 8, 0.85, -14.0),  # mid-E accent
        (S_E + 16, 0.85, -14.0),
        (S_G, 0.7, -16.0),  # long ring at tail start
    )
    for bar_pos, vel, amp_db in hits:
        if bar_pos >= TOTAL_BARS:
            continue
        score.add_note(
            "crash",
            start=bar_pos * BAR,
            duration=3.0 * BAR,  # long, natural decay; voice-level peak norm tames
            partial=1.0,
            amp_db=amp_db,
            velocity=vel,
        )


# ---------------------------------------------------------------------------
# Polymeter layer writers — closed_hat (11-beat), glitch_perc (9-beat), ticks
# ---------------------------------------------------------------------------


def _add_closed_hat(score: Score) -> None:
    """Tile the 11-sixteenth hat pattern across active sections.

    Cycle length in the phase-relation is 11 SIXTEENTHS = 11/16 of a bar,
    so against the 16-sixteenth bar the pattern phases every 11 bars (11*16
    = 176 sixteenths LCM).  We tile it over the whole active span; the
    phase motion is audible and coherent rather than random.
    """
    base = _one_cycle_hat_phrase()
    cycle_seconds = 11 * S16

    # Section-by-section tiling, each with its own start/end + amp_db offset.
    sections: tuple[tuple[float, float, float], ...] = (
        (T_B + 4 * BAR, T_C, -2.0),  # enters partway through B, ramps to C
        (T_C, T_D, 0.0),  # full in C
        (T_D + 4 * BAR, T_E, -6.0),  # brief return at end of break
        (T_E, T_F, 2.0),  # peak, hotter
        (T_F, T_G, -4.0),  # resolution thinning
    )
    for start, end, amp_off in sections:
        tiled = polymeter_layer(base, cycle=cycle_seconds, total=end - start, start=0.0)
        for event in tiled.events:
            score.add_note(
                "closed_hat",
                start=start + event.start,
                duration=event.duration,
                partial=event.partial or 1.0,
                amp_db=amp_to_db(float(event.amp or 1.0)) + amp_off,
                velocity=event.velocity,
            )


def _add_glitch_perc(score: Score) -> None:
    """Tile the 9-sixteenth Turing glitch layer across C through F."""
    base = _one_cycle_glitch_phrase()
    cycle_seconds = 9 * S16
    sections: tuple[tuple[float, float, float], ...] = (
        (T_C, T_D, -2.0),
        (T_D + 4 * BAR, T_E, -8.0),  # tease in late-break
        (T_E, T_F, 2.0),
        (T_F, T_G - 2 * BAR, -6.0),
    )
    for start, end, amp_off in sections:
        if end <= start:
            continue
        tiled = polymeter_layer(base, cycle=cycle_seconds, total=end - start, start=0.0)
        for event in tiled.events:
            score.add_note(
                "glitch_perc",
                start=start + event.start,
                duration=event.duration,
                partial=event.partial or 1.0,
                amp_db=amp_to_db(float(event.amp or 1.0)) + amp_off,
                velocity=event.velocity,
            )


def _add_ticks(score: Score) -> None:
    """Tile the 7-step CA tick pattern across A through F.

    Ticks are the one drum-family voice that starts in the intro — a
    Girl/Boy-Song-style foreground chatter that establishes the weird
    before the drums land.
    """
    base = _one_cycle_tick_phrase()
    # One cycle of the CA at S16 spacing is 7 * S16 seconds.
    cycle_seconds = 7 * S16
    sections: tuple[tuple[float, float, float], ...] = (
        (T_A + 2 * BAR, T_B, -10.0),  # intro entry — very quiet, just a whisper
        (T_B, T_C, -6.0),  # groove: tucked behind kit
        (T_C, T_D, -3.0),  # counter
        (T_D, T_E, 0.0),  # break: featured
        (T_E, T_F, -2.0),  # peak (kit loud)
        (T_F, T_G, -4.0),  # resolution
    )
    for start, end, amp_off in sections:
        tiled = polymeter_layer(base, cycle=cycle_seconds, total=end - start, start=0.0)
        for event in tiled.events:
            score.add_note(
                "ticks",
                start=start + event.start,
                duration=event.duration,
                partial=event.partial or 1.0,
                amp_db=amp_to_db(float(event.amp or 1.0)) + amp_off,
                velocity=event.velocity,
            )


# ---------------------------------------------------------------------------
# Tonal voice writers
# ---------------------------------------------------------------------------


def _add_pad(score: Score) -> None:
    """Stagger-voice the 4-chord JI progression across the whole piece.

    Each chord is 2 bars long.  Progression cycles; utonal-flavored variant
    used during the break (D) for contrast.
    """
    stagger = (0.0, 0.018, 0.036, 0.054)
    chord_span = 2.0 * BAR
    num_chords = int((TOTAL_DUR + 0.01) // chord_span)
    for chord_index in range(num_chords):
        bar_start = chord_index * 2
        absolute_start = chord_index * chord_span
        if absolute_start >= TOTAL_DUR:
            break
        # Break section: shift to a more utonal voicing for harmonic contrast.
        if S_D <= bar_start < S_E:
            chord = (1.0, 7 / 6, 7 / 5, 7 / 4)  # stacked septimals
        else:
            chord = _PAD_CHORDS[chord_index % len(_PAD_CHORDS)]
        # Last 2 bars — resolve on tonic.
        if bar_start >= S_G - 2:
            chord = (1.0, 3 / 2, 2.0, 3.0)
        duration = min(chord_span - 0.08, TOTAL_DUR - absolute_start - 0.02)
        if duration <= 0:
            continue
        for partial, offset in zip(chord, stagger, strict=True):
            score.add_note(
                "pad",
                start=absolute_start + offset,
                duration=duration,
                partial=partial,
                amp_db=-16.0,
                velocity=0.9,
            )


def _add_bell(score: Score) -> None:
    """Bell carries the melody — locked 16-beat cycle, stays singable."""
    base_low = _one_cycle_bell_phrase(high=False)
    base_high = _one_cycle_bell_phrase(high=True)

    # Bell statements per section (each one bar long).
    # Entries: (bar, which_phrase, amp_off_db, velocity_scale)
    statements: tuple[tuple[int, Phrase, float, float], ...] = (
        (S_A + 6, base_low, -6.0, 0.6),  # intro hint (quiet)
        (S_B, base_low, -2.0, 0.85),
        (S_B + 4, base_low, 0.0, 0.95),
        (S_B + 8, base_low, 0.0, 0.95),
        (S_B + 12, base_low, -1.0, 0.9),
        (S_C, base_high, 0.0, 1.0),
        (S_C + 4, base_high, 0.0, 1.0),
        (S_C + 8, base_low, 0.0, 0.95),
        (S_C + 12, base_low, -1.0, 0.9),
        (S_D + 2, base_low, -4.0, 0.8),  # held note in break (below)
        # Break features the bell — sparse, long tones
        (S_E, base_high, 2.0, 1.0),
        (S_E + 4, base_high, 2.0, 1.0),
        (S_E + 8, base_low, 1.0, 0.95),
        (S_E + 12, base_high, 2.0, 1.0),
        (S_E + 16, base_low, 1.0, 0.95),
        (S_E + 20, base_low, 0.0, 0.9),
        (S_F, base_low, -2.0, 0.9),
        (S_F + 4, base_low, -3.0, 0.85),
        (S_F + 8, base_low, -5.0, 0.75),
    )
    for bar, phrase, amp_off, vel_scale in statements:
        if bar >= TOTAL_BARS:
            continue
        bar_start = bar * BAR
        for event in phrase.events:
            # Skip any note that would run past the tail.
            if bar_start + event.start >= TOTAL_DUR:
                continue
            # Add gentle vibrato on long notes during E (peak) and F.
            pitch_motion = None
            if event.duration > S16 * 2.0 and bar_start >= S_E * BAR:
                pitch_motion = PitchMotionSpec.vibrato(
                    depth_ratio=0.005,
                    rate_hz=5.5,
                )
            score.add_note(
                "bell",
                start=bar_start + event.start,
                duration=event.duration,
                partial=event.partial or 1.0,
                amp_db=amp_to_db(float(event.amp or 1.0)) + amp_off,
                velocity=max(0.55, min(1.1, event.velocity * vel_scale)),
                pitch_motion=pitch_motion,
            )


def _add_ep(score: Score) -> None:
    """EP counter-melody — 20-beat cycle (5 bars), phasing against bell's 16.

    Enters at section C and runs through E.  Kept simple (5 long notes per
    cycle) so the 20-vs-16 phase relation is audible motion rather than harmonic
    conflict.
    """
    base = _one_cycle_ep_phrase()
    cycle_seconds = 20.0 * BEAT
    sections: tuple[tuple[float, float, float], ...] = (
        (T_C, T_D, -2.0),
        (T_E, T_F, 0.0),
    )
    for start, end, amp_off in sections:
        tiled = polymeter_layer(base, cycle=cycle_seconds, total=end - start, start=0.0)
        for event in tiled.events:
            score.add_note(
                "ep",
                start=start + event.start,
                duration=event.duration,
                partial=event.partial or 1.0,
                amp_db=amp_to_db(float(event.amp or 1.0)) + amp_off,
                velocity=event.velocity,
            )


def _add_bass(score: Score) -> None:
    """Acid bass — different 8-beat pattern per section for forward motion.

    B (simple root walk), C (chromatic walk with more motion),
    early-E (simple again to re-seat), late-E + into F start (peak pattern),
    F resolution (simple one last time).  Sits out D (break).
    """
    cycle_seconds = 8.0 * BEAT  # 2 bars
    simple = _one_cycle_bass_phrase(_BASS_V1_SIMPLE)
    walk = _one_cycle_bass_phrase(_BASS_V2_WALK)
    peak = _one_cycle_bass_phrase(_BASS_V3_PEAK)

    # (pattern, start, end, amp_off_db)
    sections: tuple[tuple[Phrase, float, float, float], ...] = (
        (simple, T_B + 4 * BAR, T_C, -1.0),  # simple enters mid-B
        (walk, T_C, T_D, 0.0),  # walking through C
        (simple, T_E, T_E + 8 * BAR, 0.5),  # re-seat at peak start
        (peak, T_E + 8 * BAR, T_F, 2.0),  # busiest pattern late in peak
        (simple, T_F, T_F + 8 * BAR, -2.0),  # resolution tail-off
    )
    for base, start, end, amp_off in sections:
        if end <= start:
            continue
        tiled = polymeter_layer(base, cycle=cycle_seconds, total=end - start, start=0.0)
        for event in tiled.events:
            pitch_motion = event.pitch_motion  # ratio_glide preserved on beat-1
            score.add_note(
                "bass",
                start=start + event.start,
                duration=event.duration,
                partial=event.partial or 1.0,
                amp_db=amp_to_db(float(event.amp or 1.0)) + amp_off,
                velocity=event.velocity,
                pitch_motion=pitch_motion,
            )


# ---------------------------------------------------------------------------
# Automation lanes
# ---------------------------------------------------------------------------


def _intensity_automation() -> AutomationSpec:
    """Piece-wide intensity arc: 0 → 0.5 (B) → 0.7 (C) → 0.4 (D) → 1.0 (E) → 0."""
    return AutomationSpec(
        target=AutomationTarget(kind="control", name="mix"),
        segments=(
            AutomationSegment(
                start=0.0, end=T_B, shape="linear", start_value=0.0, end_value=0.5
            ),
            AutomationSegment(
                start=T_B, end=T_C, shape="linear", start_value=0.5, end_value=0.65
            ),
            AutomationSegment(
                start=T_C, end=T_D, shape="linear", start_value=0.65, end_value=0.75
            ),
            AutomationSegment(
                start=T_D, end=T_E, shape="linear", start_value=0.75, end_value=0.45
            ),
            AutomationSegment(
                start=T_E, end=T_F, shape="linear", start_value=0.45, end_value=1.0
            ),
            AutomationSegment(
                start=T_F, end=T_G, shape="linear", start_value=1.0, end_value=0.3
            ),
            AutomationSegment(
                start=T_G,
                end=TOTAL_DUR,
                shape="linear",
                start_value=0.3,
                end_value=0.0,
            ),
        ),
        clamp_min=0.0,
        clamp_max=1.0,
    )


def _pad_cutoff_automation() -> AutomationSpec:
    """Pad filter cutoff rides the form — exponential for natural frequency motion."""
    return AutomationSpec(
        target=AutomationTarget(kind="synth", name="cutoff_hz"),
        segments=(
            AutomationSegment(
                start=0.0, end=T_B, shape="exp", start_value=600.0, end_value=1200.0
            ),
            AutomationSegment(
                start=T_B, end=T_D, shape="exp", start_value=1200.0, end_value=2400.0
            ),
            AutomationSegment(
                start=T_D, end=T_E, shape="exp", start_value=2400.0, end_value=1500.0
            ),
            AutomationSegment(
                start=T_E, end=T_F, shape="exp", start_value=1500.0, end_value=3800.0
            ),
            AutomationSegment(
                start=T_F,
                end=TOTAL_DUR,
                shape="exp",
                start_value=3800.0,
                end_value=800.0,
            ),
        ),
    )


def _pad_mix_automation() -> AutomationSpec:
    """Pad mix fader rides the form, stays audible throughout."""
    return AutomationSpec(
        target=AutomationTarget(kind="control", name="mix_db"),
        segments=(
            AutomationSegment(
                start=0.0, end=T_B, shape="linear", start_value=-18.0, end_value=-10.0
            ),
            AutomationSegment(
                start=T_B, end=T_D, shape="linear", start_value=-10.0, end_value=-8.0
            ),
            AutomationSegment(
                start=T_D, end=T_E, shape="linear", start_value=-8.0, end_value=-4.0
            ),
            AutomationSegment(
                start=T_E, end=T_F, shape="linear", start_value=-4.0, end_value=-6.0
            ),
            AutomationSegment(
                start=T_F,
                end=TOTAL_DUR,
                shape="linear",
                start_value=-6.0,
                end_value=-22.0,
            ),
        ),
    )


def _bass_cutoff_automation() -> AutomationSpec:
    """Bass cutoff stays modest in B/C, squelches open in E."""
    return AutomationSpec(
        target=AutomationTarget(kind="synth", name="cutoff_hz"),
        segments=(
            AutomationSegment(start=0.0, end=T_C, shape="hold", value=320.0),
            AutomationSegment(
                start=T_C, end=T_E, shape="exp", start_value=320.0, end_value=450.0
            ),
            AutomationSegment(
                start=T_E, end=T_F, shape="exp", start_value=450.0, end_value=850.0
            ),
            AutomationSegment(start=T_F, end=TOTAL_DUR, shape="hold", value=320.0),
        ),
    )


# ---------------------------------------------------------------------------
# Master and send buses
# ---------------------------------------------------------------------------


def _bell_hall_send_automation() -> AutomationSpec:
    """Bell hall-send rides wetter through the break, snaps back for E.

    During D (break) the arrangement thins out; pushing the bell hall send
    from -10 dB up to -3 dB turns it into a spacious featured element.
    Snap back to -10 dB for E where the dense arrangement wants the bell
    dry and present.
    """
    pre_d = T_D - BAR
    pre_e = T_E - BAR
    return AutomationSpec(
        target=AutomationTarget(kind="control", name="send_db"),
        segments=(
            AutomationSegment(start=0.0, end=pre_d, shape="hold", value=-10.0),
            AutomationSegment(
                start=pre_d, end=T_D, shape="linear", start_value=-10.0, end_value=-3.0
            ),
            AutomationSegment(start=T_D, end=pre_e, shape="hold", value=-3.0),
            AutomationSegment(
                start=pre_e, end=T_E, shape="linear", start_value=-3.0, end_value=-10.0
            ),
            AutomationSegment(start=T_E, end=T_F, shape="hold", value=-10.0),
            AutomationSegment(
                start=T_F,
                end=TOTAL_DUR,
                shape="linear",
                start_value=-10.0,
                end_value=-6.0,
            ),
        ),
    )


def _master_effects() -> list[EffectSpec]:
    """Override DEFAULT_MASTER_EFFECTS to add tape + tone shaping.

    High-shelf roll-off above 7 kHz tames aggressive perc top; low-shelf
    lift seats the kick + bass in the mix.
    """
    return [
        EffectSpec(
            "eq",
            {
                "bands": [
                    {"kind": "low_shelf", "freq_hz": 120.0, "gain_db": 1.0},
                    {"kind": "high_shelf", "freq_hz": 7500.0, "gain_db": -2.5},
                ]
            },
        ),
        EffectSpec("preamp", {"preset": "neve_warmth"}),
        EffectSpec(
            "compressor",
            {"preset": "master_glue", "threshold_db": -20.0, "ratio": 1.6},
        ),
        EffectSpec(
            "chow_tape",
            {"drive": 0.3, "saturation": 0.4, "bias": 0.5, "mix": 45.0},
        ),
    ]


# ---------------------------------------------------------------------------
# Score builder
# ---------------------------------------------------------------------------


def build_score() -> Score:
    score = Score(f0_hz=F0_HZ, master_effects=_master_effects())

    # Macros
    score.add_macro("intensity", default=0.0, automation=_intensity_automation())

    # Drum sidechain duck via a macro routed to a few voice mix faders — a
    # gentle glue effect, not a hard pump.  Implemented as a slow-attack
    # ModConnection rather than a proper sidechain compressor.
    score.modulations.extend(
        [
            ModConnection(
                name="intensity_to_pad_cutoff",
                source=MacroSource(name="intensity"),
                target=AutomationTarget(kind="synth", name="cutoff_hz"),
                amount=400.0,
                bipolar=False,
                mode="add",
            ),
        ]
    )

    # Send buses
    drum_bus = setup_drum_bus(
        score,
        effects=[
            EffectSpec(
                "compressor",
                {
                    "threshold_db": -16.0,
                    "ratio": 2.2,
                    "attack_ms": 10.0,
                    "release_ms": 140.0,
                    "knee_db": 6.0,
                    "topology": "feedback",
                    "detector_mode": "rms",
                },
            ),
        ],
        return_db=0.0,
    )
    score.add_send_bus(
        "hall",
        effects=[
            SOFT_REVERB_EFFECT,
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 220.0,
                            "slope_db_per_oct": 12,
                        },
                        {
                            "kind": "lowpass",
                            "cutoff_hz": 5200.0,
                            "slope_db_per_oct": 12,
                        },
                    ]
                },
            ),
        ],
        return_db=-4.0,
    )
    score.add_send_bus(
        "delay",
        effects=[
            EffectSpec(
                "delay",
                {
                    "delay_seconds": 3.0 * BEAT / 4.0,  # dotted-8th at 130 BPM
                    "feedback": 0.32,
                    "mix": 1.0,  # bus return handles overall level
                },
            ),
        ],
        return_db=-10.0,
    )

    # ---- Drum voices ----
    # Mix philosophy: kick/bass own the low end (loudest); tonal elements
    # (pad/bell/ep) sit at conversational levels; percussion is supporting,
    # not featured.  Tuss-dense does NOT mean trebly-dense.
    add_drum_voice(
        score,
        "kick",
        engine="drum_voice",
        preset="808_house",
        drum_bus=drum_bus,
        send_db=-3.0,
        synth_overrides={
            # Add a bit of click for presence — layer the exciter on top of
            # the 808_house body without changing the body itself.
            "exciter_type": "click",
            "exciter_level": 0.18,
            "exciter_decay_s": 0.004,
        },
        effects=[
            EffectSpec("compressor", {"preset": "kick_punch"}),
            # Iron-oxide-ish weight for the low end.
            EffectSpec("saturation", {"preset": "kick_weight"}),
        ],
        mix_db=1.0,
    )
    add_drum_voice(
        score,
        "snare",
        engine="drum_voice",
        preset="909_tight",
        drum_bus=drum_bus,
        send_db=-5.0,
        synth_overrides={
            # More pitched body, longer body tail — this is the thwack.
            # Wire is slightly tamed so body carries the weight.
            "tone_level": 1.1,
            "tone_decay_s": 0.09,
            "noise_level": 0.8,
            "exciter_level": 0.9,
        },
        effects=[
            # Faster snare_punch attack for a sharper transient slap.
            EffectSpec(
                "compressor",
                {"preset": "snare_punch", "attack_ms": 3.0, "ratio": 3.5},
            ),
            # Low-mid push for body; tiny 6 kHz cut so ghosts don't get crispy.
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "bell", "freq_hz": 220.0, "gain_db": 2.0, "q": 1.1},
                        {"kind": "high_shelf", "freq_hz": 6000.0, "gain_db": -1.0},
                    ]
                },
            ),
        ],
        mix_db=-8.0,  # -3 dB from previous -5.0
    )
    add_drum_voice(
        score,
        "clap",
        engine="drum_voice",
        preset="909_clap",
        drum_bus=drum_bus,
        send_db=-5.0,
        effects=[EffectSpec("compressor", {"preset": "snare_punch"})],
        mix_db=-8.0,
        pan=0.15,
    )
    score.voices["clap"].sends.append(VoiceSend(target="hall", send_db=-10.0))

    # Crash — for section-entry swells (C, E) and a final decay tail.
    add_drum_voice(
        score,
        "crash",
        engine="drum_voice",
        preset="crash",
        drum_bus=drum_bus,
        send_db=-6.0,
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 200.0,
                            "slope_db_per_oct": 12,
                        },
                        {"kind": "high_shelf", "freq_hz": 8000.0, "gain_db": -2.0},
                    ]
                },
            ),
        ],
        mix_db=-10.0,
        pan=0.05,
    )
    score.voices["crash"].sends.append(VoiceSend(target="hall", send_db=-4.0))

    add_drum_voice(
        score,
        "closed_hat",
        engine="drum_voice",
        preset="beating_hat_a",
        drum_bus=drum_bus,
        send_db=-7.0,
        effects=[
            EffectSpec("compressor", {"preset": "hat_control"}),
            # Gentle top tame — preset is bright enough, we don't need more.
            EffectSpec(
                "eq",
                {"bands": [{"kind": "high_shelf", "freq_hz": 9000.0, "gain_db": -2.0}]},
            ),
        ],
        mix_db=-13.0,
        pan=-0.25,
    )
    add_drum_voice(
        score,
        "glitch_perc",
        engine="drum_voice",
        preset="hat_rate_reduced",
        drum_bus=drum_bus,
        send_db=-8.0,
        effects=[EffectSpec("compressor", {"preset": "hat_control"})],
        mix_db=-16.0,
        pan=0.3,
    )
    add_drum_voice(
        score,
        "ticks",
        engine="metallic_perc",
        preset="clave",
        drum_bus=drum_bus,
        send_db=-12.0,
        normalize_peak_db=-14.0,
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "lowpass", "cutoff_hz": 6500.0, "slope_db_per_oct": 12}
                    ]
                },
            ),
        ],
        mix_db=-18.0,
        pan=-0.35,
    )

    # ---- Tonal voices ----
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "additive_pad_through_ladder",
            "attack": 0.6,
            "release": 2.2,
        },
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 70.0, "slope_db_per_oct": 12},
                        {"kind": "high_shelf", "freq_hz": 6500.0, "gain_db": -1.5},
                    ]
                },
            ),
            # Transformer warmth before the chorus — valve sheen, not drive.
            EffectSpec("preamp", {"preset": "neve_warmth"}),
            EffectSpec("bbd_chorus", {"preset": "juno_i_plus_ii", "mix": 0.28}),
        ],
        mix_db=-10.0,
        pan=-0.08,
        velocity_humanize=None,
        sends=[VoiceSend(target="hall", send_db=-6.0)],
        automation=[_pad_cutoff_automation(), _pad_mix_automation()],
    )

    score.add_voice(
        "bell",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "fm_bell_over_supersaw",
            "release": 1.1,
        },
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 150.0,
                            "slope_db_per_oct": 12,
                        },
                    ]
                },
            ),
            # Light transformer color — rounds the FM harmonics, keeps it sweet.
            EffectSpec("preamp", {"preset": "neve_warmth"}),
        ],
        mix_db=-4.0,
        pan=0.1,
        velocity_humanize=None,
        sends=[
            VoiceSend(
                target="hall",
                send_db=-10.0,
                automation=[_bell_hall_send_automation()],
            ),
            VoiceSend(target="delay", send_db=-14.0),
        ],
    )

    score.add_voice(
        "ep",
        synth_defaults={
            "engine": "fm",
            "preset": "chorused_ep",
        },
        effects=[
            EffectSpec("bbd_chorus", {"preset": "juno_i", "mix": 0.22}),
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 110.0,
                            "slope_db_per_oct": 12,
                        },
                        {"kind": "high_shelf", "freq_hz": 5500.0, "gain_db": -1.0},
                    ]
                },
            ),
        ],
        mix_db=-8.0,
        pan=-0.18,
        velocity_humanize=None,
        sends=[VoiceSend(target="hall", send_db=-12.0)],
    )

    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "polyblep",
            "preset": "tb303_bass",
            "filter_solver": "newton",
            "quality": "great",
            # Drive the diode ladder harder for squelch
            "filter_drive": 0.7,
            "resonance_q": 7.5,
            # Per-note corrode — the "rubbery" character, applied pre-mix so
            # chord tones would distort independently (moot for mono bass but
            # cleaner character than a post-mix shaper).
            "voice_dist_mode": "corrode",
            "voice_dist_drive": 0.3,
            "voice_dist_mix": 0.35,
            "voice_dist_tone": 0.55,
            "analog_jitter": 0.55,
            "voice_card_spread": 1.0,
            "osc_phase_noise": 0.08,
        },
        effects=[
            EffectSpec(
                "compressor", {"preset": "kick_duck", "sidechain_source": "kick"}
            ),
            # Tube warmth downstream of the diode ladder — harmonics, not distortion.
            EffectSpec("saturation", {"preset": "tube_warm"}),
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 40.0, "slope_db_per_oct": 12},
                        {"kind": "low_shelf", "freq_hz": 110.0, "gain_db": 1.5},
                    ]
                },
            ),
        ],
        mix_db=-4.0,
        pan=0.0,
        velocity_humanize=None,
        automation=[_bass_cutoff_automation()],
    )

    # ---- Populate notes ----
    _add_kick(score)
    _add_snare(score)
    _add_clap(score)
    _add_crash(score)
    _add_closed_hat(score)
    _add_glitch_perc(score)
    _add_ticks(score)
    _add_pad(score)
    _add_bell(score)
    _add_ep(score)
    _add_bass(score)

    return score


PIECES: dict[str, PieceDefinition] = {
    "misty_pixel": PieceDefinition(
        name="misty_pixel",
        output_name="misty_pixel",
        build_score=build_score,
        sections=(
            PieceSection(label="A Intro", start_seconds=T_A, end_seconds=T_B),
            PieceSection(label="B Groove in", start_seconds=T_B, end_seconds=T_C),
            PieceSection(label="C Counter", start_seconds=T_C, end_seconds=T_D),
            PieceSection(label="D Break", start_seconds=T_D, end_seconds=T_E),
            PieceSection(label="E Peak", start_seconds=T_E, end_seconds=T_F),
            PieceSection(label="F Resolution", start_seconds=T_F, end_seconds=T_G),
            PieceSection(label="G Tail", start_seconds=T_G, end_seconds=TOTAL_DUR),
        ),
    ),
}
