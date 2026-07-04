"""The Ninth Wave — Erv Wilson's 1-3-5-7-9 dekany as a tidal journey.

The 2-of-5 Combination Product Set on {1, 3, 5, 7, 9} gives ten notes —
1/1, 9/8, 7/6, 5/4, 21/16, 35/24, 3/2, 5/3, 7/4, 15/8 — and two gifts
the hexany never had.  First: drop any one factor and the remaining six
notes form a complete hexany, so the dekany contains five overlapping
hexanies — and the omit-9 one is *exactly* the scale of hexany_garden.
The old garden turns out to be one facet of a larger crystal.  Second:
the 9 brings the 3/2.  The perfect fifth the garden never had arrives
with the new factor, like a tide reaching a walled garden.

Harmony: each factor's four product-partners form an otonal tetrad
(share-9 is 1:3:5:7 itself, revoiced 6:7:8:10), and every pair of
tetrads shares exactly one note — a pivot tone that holds while three
voices move.  The 2-of-5 set has no true utonal tetrads; its shadows
are the ten utonal triads (one per 3-subset).  The piece works in
triads for most of its length — each island section can only see
tetrad fragments — and the full four-voice sonorities are saved for
the final wave, when all ten notes are finally on the table.

Form is a tide crossing the five embedded hexanies (Carl Craig "At
Les" / M83 register: journey techno, string-machine stabs, one big
undertow).  It ends by answering hexany_garden's ending: that piece
closed on a bare, unresolved 4:7 dyad; this one takes the same dyad
and blooms it into 4:5:6:7 — a chord that needs the 3/2, i.e. a chord
the hexany could never play.

Tuning: 1-3-5-7-9 dekany (``code_musics.tuning.dekany``) on
f0 = G2 = 98 Hz; the bass rides an octave below at G1 = 49 Hz where a
techno kick wants company, kept out of the kick's way by offbeat
patterns plus hard kick-keyed ducking.

BPM = 122.  1 bar ≈ 1.967 s.  224 bars ≈ 7:21.

Form:
  bars   1– 24  S1 Ebb          beatless; dekany-spectrum drone; the
                                old garden motif quoted in the omit-9
                                hexany, then the wave motif brings the
                                first 9/8 and 3/2 the piece has heard
  bars  25– 64  S2 First wave   omit-7 island (pure 5-limit sunrise);
                                kick walks in on 1 & 3, four-on-floor
                                from 41; the string-stab riff states
                                itself; bass finds the root
  bars  65–112  S3 Open water   omit-5 island (septimal drive), reached
                                over a two-bar bridge (the lead walks
                                down onto 7/6 while the kick cuts);
                                full kit, rolling offbeat bass, tide
                                arp; four-bar kick drop at 97, slam at
                                101
  bars 113–144  S4 Undertow     omit-1 island: no 1/1, no 3/2 — the
                                floor gone.  Drums out, utonal triads,
                                the motif inverted; the seafloor pedal
                                on 15/8; from 137 the long climb back
                                (golden section ≈ bar 139 mid-climb),
                                then two near-black bars at 143–144 —
                                only the bass pulse — before the break
  bars 145–200  S5 Ninth wave   the full dekany at last: all five
                                otonal tetrads cycling on pivot tones,
                                first four-voice harmony of the piece;
                                harmonic rhythm doubles at 185; two
                                held-breath bars at 199
  bars 201–224  S6 Slack water  drums fall away; the old garden motif
                                once more; the bare 4:7 dyad from the
                                end of hexany_garden — then 5/4 and
                                3/2 join it: 4:5:6:7, the answer

Composed by Claude (Fable 5), July 2026.
"""

from __future__ import annotations

import random

from code_musics.automation import (
    AutomationSegment,
    AutomationShape,
    AutomationSpec,
    AutomationTarget,
)
from code_musics.drum_helpers import add_drum_voice, setup_drum_bus
from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    VelocityHumanizeSpec,
)
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS, bricasti_or_reverb
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import EffectSpec, Score, SendBusSpec, VoiceSend
from code_musics.spectra import ratio_spectrum
from code_musics.tuning import dekany

# ---------------------------------------------------------------------------
# Timing grid
# ---------------------------------------------------------------------------

BPM: float = 122.0
BEAT: float = 60.0 / BPM
BAR: float = 4.0 * BEAT
S16: float = BEAT / 4.0
S8: float = BEAT / 2.0
DOTTED_EIGHTH: float = 3.0 * S16

F0: float = 98.0  # G2; bass voices ride octave -1 → G1 = 49 Hz
TOTAL_BARS: int = 224

HAT_SWING_S: float = 0.10 * S16
ARP_SWING_S: float = 0.05 * S16


def _pos(bar: int, beat: float = 1.0, n16: int = 0) -> float:
    """Absolute time at bar:beat:sixteenth (1-indexed bar/beat)."""
    return (bar - 1) * BAR + (beat - 1.0) * BEAT + n16 * S16


# Section boundaries (bar numbers; S1 starts at bar 1)
S2_BAR: int = 25
S3_BAR: int = 65
S4_BAR: int = 113
S5_BAR: int = 145
S6_BAR: int = 201
END_BAR: int = TOTAL_BARS + 1

S1_END: float = _pos(S2_BAR)
S2_END: float = _pos(S3_BAR)
S3_END: float = _pos(S4_BAR)
S4_END: float = _pos(S5_BAR)
S5_END: float = _pos(S6_BAR)
TOTAL_DUR: float = _pos(END_BAR)

# Bars where the drums hold their breath.
_DROP_BARS: frozenset[int] = frozenset(range(97, 101)) | frozenset({199, 200})

# ---------------------------------------------------------------------------
# The dekany.  Degree indices order the scale ascending within an octave.
# Each degree is a product of two of {1, 3, 5, 7, 9}; the four degrees
# sharing a factor sound as an otonal tetrad over that factor, and every
# pair of tetrads shares exactly one degree — the pivot tone.
# ---------------------------------------------------------------------------

_DEGREE_RATIOS: tuple[float, ...] = tuple(dekany())
_N_DEGREES: int = len(_DEGREE_RATIOS)

_DEGREE_FACTORS: tuple[frozenset[int], ...] = (
    frozenset({1, 3}),  # 0: 1/1    = 1·3
    frozenset({3, 9}),  # 1: 9/8    = 3·9
    frozenset({1, 7}),  # 2: 7/6    = 1·7
    frozenset({3, 5}),  # 3: 5/4    = 3·5
    frozenset({7, 9}),  # 4: 21/16  = 7·9
    frozenset({5, 7}),  # 5: 35/24  = 5·7
    frozenset({1, 9}),  # 6: 3/2    = 1·9
    frozenset({1, 5}),  # 7: 5/3    = 1·5
    frozenset({3, 7}),  # 8: 7/4    = 3·7
    frozenset({5, 9}),  # 9: 15/8   = 5·9
)


def _check_tetrad_factors() -> None:
    """Each otonal tetrad's four degrees must actually share its factor."""
    for shared_factor, tetrad in (
        (9, _T9),
        (3, _T3),
        (7, _T7),
        (5, _T5),
        (1, _T1),
    ):
        if not all(shared_factor in _DEGREE_FACTORS[d] for d in tetrad):
            raise AssertionError(
                f"tetrad {tetrad} does not share factor {shared_factor}"
            )


# The five otonal tetrads (degrees sharing one factor), the piece's full
# four-voice sonorities — only playable once every island has been visited.
_T9: tuple[int, ...] = (1, 4, 6, 9)  # share 9: 6:7:8:10 = 1:3:5:7 — the wave
_T3: tuple[int, ...] = (0, 1, 3, 8)  # share 3: 8:9:10:14 on 1/1 — home
_T7: tuple[int, ...] = (2, 4, 5, 8)  # share 7: 8:9:10:12 on 7/6 — lift
_T5: tuple[int, ...] = (3, 5, 7, 9)  # share 5: 6:7:8:9 — glow
_T1: tuple[int, ...] = (0, 2, 6, 7)  # share 1: 6:7:9:10 — tide

_check_tetrad_factors()

# Island triads: each island (omit one factor) truncates the tetrads to
# three notes.  S2's omit-7 island is pure 5-limit; S3's omit-5 island is
# septimal; S4's omit-1 island has no 1/1 and no 3/2 — the floor gone.
_H3: tuple[int, ...] = (0, 1, 3)  # S2 home: 1/1 9/8 5/4
_H9: tuple[int, ...] = (1, 6, 9)  # S2 wave-lite: 9/8 3/2 15/8
_H1: tuple[int, ...] = (0, 6, 7)  # S2 open: 1/1 3/2 5/3
_H5: tuple[int, ...] = (3, 7, 9)  # S2 high glow: 5/4 5/3 15/8
_P1: tuple[int, ...] = (0, 2, 6)  # S3 subminor: 1/1 7/6 3/2
_P9: tuple[int, ...] = (1, 4, 6)  # S3 crest: 9/8 21/16 3/2 = 6:7:8
_P3: tuple[int, ...] = (0, 1, 8)  # S3 undertone: 1/1 9/8 7/4
_P7: tuple[int, ...] = (2, 4, 8)  # S3 septimal lift: 7/6 21/16 7/4
# S4 utonal triads (3-subsets avoiding factor 1).
_U359: tuple[int, ...] = (1, 3, 9)  # 9/8 5/4 15/8   ~ 1/9 : 1/5 : 1/3
_U379: tuple[int, ...] = (1, 4, 8)  # 9/8 21/16 7/4  ~ 1/9 : 1/7 : 1/3
_U579: tuple[int, ...] = (4, 5, 9)  # 21/16 35/24 15/8 ~ 1/9 : 1/7 : 1/5
_U357: tuple[int, ...] = (3, 5, 8)  # 5/4 35/24 7/4  ~ 1/7 : 1/5 : 1/3
# The ending: hexany_garden's bare 4:7 dyad, then its answer.
_DYAD47: tuple[int, ...] = (0, 8)  # 1/1 7/4
_ANSWER: tuple[int, ...] = (0, 3, 6, 8)  # 1/1 5/4 3/2 7/4 = 4:5:6:7

# Harmonic plan: (start_bar, chord degrees, root degree, label).
# Each slot runs to the next start.  S1 is drone-only (implied 1/1).
_CHORD_SLOTS: tuple[tuple[int, tuple[int, ...], int, str], ...] = (
    (1, _H3, 0, "drone"),
    # S2 — first wave, omit-7 island (5-limit sunrise)
    (25, _H3, 0, "home"),
    (29, _H1, 0, "open"),
    (33, _H3, 0, "home"),
    (37, _H9, 1, "wave_lite"),
    (41, _H3, 0, "home"),
    (45, _H5, 3, "glow_lite"),
    (49, _H9, 1, "wave_lite"),
    (53, _H1, 0, "open"),
    (57, _H3, 0, "home"),
    # Pivot into the septimal island: _H1 shares both 1/1 and 3/2 with _P1,
    # so only one pad lane moves at the S3 seam (5/3 slides down to 7/6's
    # world) instead of the whole chord lurching.
    (61, _H1, 0, "open"),
    # S3 — open water, omit-5 island (septimal drive)
    (65, _P1, 0, "subminor"),
    (69, _P9, 1, "crest"),
    (73, _P1, 0, "subminor"),
    (77, _P3, 0, "undertone"),
    (81, _P7, 2, "sept_lift"),
    (85, _P9, 1, "crest"),
    (89, _P1, 0, "subminor"),
    (93, _P3, 0, "undertone"),
    (97, _P9, 1, "crest"),  # the drop
    (101, _P1, 0, "subminor"),  # the slam
    (105, _P7, 2, "sept_lift"),
    (109, _P9, 1, "crest"),
    # S4 — undertow, omit-1 island (rootless, utonal)
    (113, _U359, 9, "u_glass"),
    (117, _U379, 9, "u_iron"),
    (121, _U579, 9, "u_deep"),
    (125, _U357, 9, "u_veil"),
    (129, _U379, 9, "u_iron"),
    (133, _U359, 9, "u_glass"),
    (137, _U579, 1, "u_climb"),  # the climb back begins
    (141, _U379, 1, "u_climb"),
    # S5 — the ninth wave: full tetrads, pivot-tone cycle
    (145, _T9, 1, "wave"),
    (149, _T3, 0, "home"),
    (153, _T7, 2, "lift"),
    (157, _T5, 3, "glow"),
    (161, _T1, 0, "tide"),
    (165, _T9, 1, "wave"),
    (169, _T3, 0, "home"),
    (173, _T7, 2, "lift"),
    (177, _T5, 3, "glow"),
    (181, _T1, 0, "tide"),
    (185, _T9, 1, "wave"),  # harmonic rhythm doubles
    (187, _T3, 0, "home"),
    (189, _T7, 2, "lift"),
    (191, _T5, 3, "glow"),
    (193, _T1, 0, "tide"),
    (195, _T9, 1, "wave"),
    (197, _T3, 0, "home"),
    (199, _T9, 1, "wave"),  # held breath
    # S6 — slack water and the answer
    (201, _T3, 0, "home"),
    (205, _T1, 0, "tide"),
    (209, _T3, 0, "home"),
    (213, _DYAD47, 0, "dyad47"),
    (217, _ANSWER, 0, "answer"),
)


def _slot_at_bar(bar: int) -> tuple[int, tuple[int, ...], int, str]:
    slot = _CHORD_SLOTS[0]
    for candidate in _CHORD_SLOTS:
        if bar >= candidate[0]:
            slot = candidate
    return slot


def _chord_at_bar(bar: int) -> tuple[int, ...]:
    return _slot_at_bar(bar)[1]


def _root_at_bar(bar: int) -> int:
    return _slot_at_bar(bar)[2]


def _partial(degree: int, octave: int) -> float:
    """Partial of f0 for a scale degree in an octave (octave 0 = f0..2*f0)."""
    return _DEGREE_RATIOS[degree] * (2.0**octave)


def _pi(degree: int, octave: int) -> int:
    return octave * _N_DEGREES + degree


# ---------------------------------------------------------------------------
# Motifs.
#
# The wave motif is the piece's own seed: every note contains the factor 9
# (9/8 → 3/2 → 21/16 → 15/8 → 3/2) — a rising-falling wave drawn entirely
# in the tones the hexany never had.
#
# The garden motif is hexany_garden's seed motif quoted literally into
# dekany degrees (1/1, 5/4, 7/4, 5/3, 5/4) — the memory the piece opens
# with and returns to at the end.
# ---------------------------------------------------------------------------

_WAVE_MOTIF: tuple[tuple[int, float], ...] = (
    (1, 1.0),
    (6, 0.5),
    (4, 1.5),
    (9, 1.0),
    (6, 2.0),
)

_GARDEN_MOTIF: tuple[tuple[int, float], ...] = (
    (0, 1.0),
    (3, 0.5),
    (8, 1.5),
    (7, 1.0),
    (3, 2.0),
)

# The undertow inversion: the wave motif's contour reflected through its
# opening pitch (pitch-index reflection), mapped to omit-1-island degrees.
_UNDERTOW_MOTIF: tuple[tuple[int, float], ...] = (
    (9, 1.0),
    (5, 0.5),
    (8, 1.5),
    (3, 1.0),
    (5, 2.0),
)


def _place_motif(
    score: Score,
    voice: str,
    start: float,
    motif: tuple[tuple[int, float], ...],
    *,
    octave: int,
    time_scale: float = 1.0,
    amp_db: float = -10.0,
    vel: float = 0.72,
    ring: float = 1.5,
    vibrato: bool = True,
    stmt: str = "seam",
) -> None:
    t = start
    for idx, (degree, beats) in enumerate(motif):
        dur = beats * BEAT * time_scale
        score.add_note(
            voice,
            start=t,
            duration=dur * ring,
            partial=_partial(degree, octave),
            amp_db=amp_db,
            velocity=vel,
            pitch_motion=(
                PitchMotionSpec.vibrato(depth_ratio=0.007, rate_hz=5.2)
                if vibrato and beats >= 1.0
                else None
            ),
            label=f"motif;deg={degree};oct={octave};idx={idx};stmt={stmt}",
        )
        t += dur


# ---------------------------------------------------------------------------
# Pad: dekany-spectrum chords with nearest-tone voice leading.
# ---------------------------------------------------------------------------

# The pad's spectrum is the dekany itself spread over three octaves —
# every partial interval is a product of {1,3,5,7,9}.  Timbre-harmony fusion.
_DEKANY_SPECTRUM = ratio_spectrum(
    [1.0, 2.0, 9.0 / 4.0, 5.0 / 2.0, 3.0, 7.0 / 2.0, 9.0 / 2.0, 5.0, 6.0, 7.0],
    amps=[1.0, 0.60, 0.34, 0.38, 0.30, 0.24, 0.14, 0.16, 0.10, 0.09],
)


def _place_pad(score: Score) -> None:
    # S1: the ebb — a bare 1/1 pedal in two octaves; the 9-tones do not
    # exist yet.  Their absence is the tension.
    score.add_note(
        "pad",
        start=_pos(1),
        duration=S1_END - _pos(1) + 2.0,
        partial=1.0,
        amp_db=-10.0,
        velocity=0.6,
        label="pad;deg=0;oct=0;role=drone",
    )
    score.add_note(
        "pad",
        start=_pos(7),
        duration=S1_END - _pos(7) + 2.0,
        partial=2.0,
        amp_db=-15.0,
        velocity=0.5,
        label="pad;deg=0;oct=1;role=drone",
    )
    # Bar 17: the pad quietly gains the first 9-tone (9/8) under the wave
    # motif's first statement — the tide starting to reach the garden.
    score.add_note(
        "pad",
        start=_pos(17),
        duration=S1_END - _pos(17) + 2.0,
        partial=_partial(1, 1),
        amp_db=-18.0,
        velocity=0.45,
        label="pad;deg=1;oct=1;role=first_nine",
    )

    # From S2 on: slot chords with actual voice-leading.  At each change
    # every lane moves to the nearest available chord tone in pitch-index
    # space; common tones hold; moving voices arrive a beat late.
    slots = [s for s in _CHORD_SLOTS if s[0] >= S2_BAR]
    glide = PitchMotionSpec.ratio_glide(start_ratio=0.996, end_ratio=1.0)
    prev_voicing: list[tuple[int, int]] | None = None  # (degree, octave)
    prev_size = 0
    for i, (start_bar, chord, root, name) in enumerate(slots):
        end_bar = slots[i + 1][0] if i + 1 < len(slots) else END_BAR
        start = _pos(start_bar)
        dur = _pos(end_bar) - start + 1.5
        ranked = sorted(chord)
        if prev_voicing is None or len(chord) != prev_size:
            # Fresh voicing on size changes (triads → tetrads → dyad):
            # close position from octave 1.
            voicing = [(d, 1) for d in ranked]
            if len(voicing) == 2:  # the closing dyad gets a top octave
                voicing.append((ranked[0], 2))
        else:
            available = list(chord)
            voicing = []
            for deg_prev, oct_prev in prev_voicing:
                pi_prev = _pi(deg_prev, oct_prev)
                best = min(
                    (abs(_pi(d, o) - pi_prev), d, o)
                    for d in set(available)
                    for o in (1, 2)
                )
                _, deg_new, oct_new = best
                available.remove(deg_new)
                voicing.append((deg_new, oct_new))
        for lane, (degree, octave) in enumerate(voicing):
            held = (
                prev_voicing is not None
                and lane < len(prev_voicing)
                and prev_voicing[lane] == (degree, octave)
            )
            lane_start = start if held or prev_voicing is None else start + BEAT
            is_root = degree == root
            score.add_note(
                "pad",
                start=lane_start,
                duration=dur - (lane_start - start),
                partial=_partial(degree, octave),
                amp_db=-12.0 if is_root else -15.0,
                velocity=0.6 if is_root else 0.5,
                pitch_motion=glide,
                label=(
                    f"pad;deg={degree};oct={octave};lane={lane}"
                    f";held={int(held)};slot_bar={start_bar};chord={name}"
                ),
            )
        prev_voicing = voicing
        prev_size = len(chord)
        # S5/S6 add the root an octave up for glow.
        if start_bar >= S5_BAR:
            score.add_note(
                "pad",
                start=start,
                duration=dur,
                partial=_partial(root, 2),
                amp_db=-20.0,
                velocity=0.45,
                label=f"pad;deg={root};oct=2;role=glow;slot_bar={start_bar}",
            )
    # The answer chord swells: the 5/4 and 3/2 lanes above enter at 217 via
    # the slot itself; here the low 1/1 returns to ground the 4:5:6:7.
    score.add_note(
        "pad",
        start=_pos(217),
        duration=TOTAL_DUR - _pos(217) + 2.0,
        partial=1.0,
        amp_db=-13.0,
        velocity=0.55,
        label="pad;deg=0;oct=0;role=answer_ground",
    )


# ---------------------------------------------------------------------------
# Stab: the string-machine riff (the At Les DNA).
# ---------------------------------------------------------------------------

# Two one-bar rhythm cells in sixteenths as (step, held_16ths, accent).
_RIFF_A: tuple[tuple[int, int, float], ...] = (
    (0, 3, 0.14),
    (3, 3, 0.02),
    (6, 4, 0.08),
    (10, 2, 0.0),
    (13, 3, 0.05),
)
_RIFF_B: tuple[tuple[int, int, float], ...] = (
    (0, 3, 0.14),
    (3, 3, 0.02),
    (6, 2, 0.06),
    (10, 2, 0.10),
    (12, 2, 0.0),
    (14, 2, 0.04),
)


def _stab_voicing(chord: tuple[int, ...], root: int) -> list[tuple[int, int]]:
    """Close voicing: root at octave 1, upper tones stacked ascending."""
    ranked = sorted(chord, key=lambda d: (d != root, _DEGREE_RATIOS[d]))
    voicing: list[tuple[int, int]] = [(ranked[0], 1)]
    prev_pi = _pi(ranked[0], 1)
    for degree in sorted((d for d in ranked[1:]), key=lambda d: _DEGREE_RATIOS[d]):
        octave = 1
        while _pi(degree, octave) <= prev_pi:
            octave += 1
        voicing.append((degree, octave))
        prev_pi = _pi(degree, octave)
    return voicing


def _place_stab(score: Score) -> None:
    rng = random.Random(13579)  # the CPS factors, of course
    for bar in range(33, S6_BAR):
        if S4_BAR <= bar < 137:
            continue  # the undertow silences the riff; it returns for the climb
        if bar in (143, 144, 199, 200):
            continue  # blackness before the slam; held breath at the end
        in_s5 = bar >= S5_BAR
        climb = 137 <= bar < 143
        _, chord, root, name = _slot_at_bar(bar)
        voicing = _stab_voicing(chord, root)
        riff = _RIFF_B if bar % 4 == 3 else _RIFF_A
        # S2 entry keeps the riff thinned; the climb re-enters sparse too,
        # and the bridge bars thin out under the lead's descent.
        keep_prob = 0.6 if bar < 41 else (0.7 if climb else 1.0)
        if bar in (63, 64):
            keep_prob = 0.45
        base_vel = 0.62 if in_s5 else (0.5 if climb else 0.56)
        base_amp = -10.5 if in_s5 else (-13.5 if climb else -11.5)
        for step, held, accent in riff:
            if rng.random() > keep_prob:
                continue
            start = _pos(bar) + step * S16
            dur = held * S16 * 0.85
            vel = min(1.0, base_vel + accent + rng.uniform(-0.03, 0.03))
            for k, (degree, octave) in enumerate(voicing):
                score.add_note(
                    "stab",
                    start=start,
                    duration=dur,
                    partial=_partial(degree, octave),
                    amp_db=base_amp - (0.0 if k == 0 else 2.0),
                    velocity=vel if k == 0 else max(0.3, vel - 0.08),
                    label=(
                        f"stab;deg={degree};oct={octave};step={step}"
                        f";chord={name};lane={k}"
                    ),
                )

    # Ghost pre-echo in S1: two lowpassed, distant stab hits foreshadow the
    # riff before the beat exists.
    for bar, beat in ((21, 1.0), (23, 3.0)):
        for k, (degree, octave) in enumerate(_stab_voicing(_H3, 0)):
            score.add_note(
                "stab",
                start=_pos(bar, beat),
                duration=BEAT * 1.5,
                partial=_partial(degree, octave),
                amp_db=-22.0 - 2.0 * k,
                velocity=0.4,
                label=f"stab;deg={degree};oct={octave};role=pre_echo",
            )


# ---------------------------------------------------------------------------
# Bass: G1 roots.  Offbeat eighths keep it out of the kick's pocket.
# ---------------------------------------------------------------------------


def _fifth_like(chord: tuple[int, ...], root: int) -> int:
    """The chord tone whose octave-folded interval above the root is
    closest to 3/2 — the bassline's counterweight degree."""
    root_ratio = _DEGREE_RATIOS[root]

    def fifth_distance(degree: int) -> float:
        interval = _DEGREE_RATIOS[degree] / root_ratio
        while interval < 1.0:
            interval *= 2.0
        while interval >= 2.0:
            interval /= 2.0
        return abs(interval - 1.5)

    candidates = [d for d in chord if d != root]
    return min(candidates, key=fifth_distance) if candidates else root


def _place_bass(score: Score) -> None:
    rng = random.Random(35)

    def note(
        bar: int,
        beat: float,
        dur: float,
        degree: int,
        octave: int,
        *,
        amp_db: float = -8.0,
        vel: float = 0.72,
        role: str = "root",
    ) -> None:
        score.add_note(
            "bass",
            start=_pos(bar, beat),
            duration=dur,
            partial=_partial(degree, octave),
            amp_db=amp_db,
            velocity=vel,
            label=f"bass;deg={degree};oct={octave};role={role}",
        )

    # S2: half-note roots finding their feet, then offbeat 8ths from 49.
    for bar in range(S2_BAR, 49):
        root = _root_at_bar(bar)
        note(bar, 1.0, 2.0 * BEAT * 0.92, root, -1, vel=0.7)
        note(bar, 3.0, 2.0 * BEAT * 0.85, root, -1, amp_db=-9.0, vel=0.62)
    # Rolling offbeat eighths: S2 tail, S3, and S5.
    rolling = list(range(49, 97)) + list(range(101, S4_BAR)) + list(range(S5_BAR, 199))
    for bar in rolling:
        _, chord, root, _ = _slot_at_bar(bar)
        fifth = _fifth_like(chord, root)
        in_s5 = bar >= S5_BAR
        for step in (2, 6, 10, 14):  # the offbeat 8ths
            octave = -1
            degree = root
            # The mid-bar offbeat leans on the fifth-like chord tone every
            # other bar — a two-bar root/fifth cell instead of a drone.
            if step == 6 and bar % 2 == 1:
                degree = fifth
            # Slot-final bar: walk the last two hits toward the next root.
            if step >= 10 and _slot_at_bar(bar + 1)[0] == bar + 1:
                next_root = _root_at_bar(bar + 1)
                degree = next_root if step == 14 else root
            # S5 occasionally pops the octave on the last offbeat.
            if in_s5 and step == 14 and rng.random() < 0.3:
                octave = 0
            # 16th pickup into the next downbeat at 8-bar phrase ends.
            if step == 14 and bar % 8 == 0 and _slot_at_bar(bar + 1)[0] != bar + 1:
                note(
                    bar,
                    4.75,
                    S16 * 0.9,
                    root,
                    octave,
                    amp_db=-9.5,
                    vel=0.6,
                    role="pickup16",
                )
            note(
                bar,
                1.0 + step / 4.0,
                S8 * 0.88,
                degree,
                octave,
                amp_db=-8.0 if step in (2, 10) else -8.8,
                vel=0.74 if step in (2, 10) else 0.66,
                role="roll",
            )
    # S4: the seafloor pedal — 15/8 two octaves down (~46 Hz), the deepest
    # and least-rooted note the piece has.  Re-articulated every 4 bars.
    for bar in range(S4_BAR, 137, 4):
        note(
            bar,
            1.0,
            4 * BAR * 0.97,
            9,
            -2,
            amp_db=-9.0,
            vel=0.6,
            role="seafloor",
        )
    # The climb: quarter-note pulse on 9/8, rising.
    for bar in range(137, S5_BAR):
        for beat in (1.0, 2.0, 3.0, 4.0):
            rise = (bar - 137 + (beat - 1.0) / 4.0) / 8.0
            note(
                bar,
                beat,
                BEAT * 0.8,
                1,
                -1,
                amp_db=-11.0 + 3.0 * rise,
                vel=0.55 + 0.2 * rise,
                role="climb",
            )
    # S6: long roots, then the answer's low 1/1.
    for bar in (201, 205, 209):
        note(bar, 1.0, 4 * BAR * 0.95, 0, -1, amp_db=-9.0, vel=0.6, role="slack")
    note(217, 1.0, TOTAL_DUR - _pos(217), 0, -1, amp_db=-10.5, vel=0.55, role="answer")


# ---------------------------------------------------------------------------
# Lead: the voice that sings the motifs.
# ---------------------------------------------------------------------------


def _place_lead(score: Score) -> None:
    # S1: memory first — the old garden motif, slow, in the omit-9 hexany.
    _place_motif(
        score,
        "lead",
        _pos(9),
        _GARDEN_MOTIF,
        octave=2,
        time_scale=2.0,
        amp_db=-11.0,
        vel=0.66,
        ring=1.9,
        stmt="garden_memory",
    )
    # Then the wave motif brings the first 9-tones the piece has heard.
    _place_motif(
        score,
        "lead",
        _pos(17),
        _WAVE_MOTIF,
        octave=2,
        time_scale=1.5,
        amp_db=-10.0,
        vel=0.7,
        ring=1.8,
        stmt="first_wave",
    )
    # S2: the wave motif owns the sunrise.
    _place_motif(score, "lead", _pos(41), _WAVE_MOTIF, octave=2, stmt="sunrise")
    _place_motif(
        score, "lead", _pos(57), _WAVE_MOTIF, octave=3, amp_db=-12.0, stmt="sunrise_hi"
    )
    # The bridge into open water: a descending half-note line that walks the
    # sunrise island down onto 7/6 — the septimal color arrives as melody two
    # beats before the harmony turns, so the S3 shift is led, not sprung.
    for k, (degree, bar, beat) in enumerate(
        ((7, 63, 1.0), (6, 63, 3.0), (3, 64, 1.0), (2, 64, 3.0))
    ):
        score.add_note(
            "lead",
            start=_pos(bar, beat),
            duration=2.0 * BEAT * (2.2 if degree == 2 else 1.3),
            partial=_partial(degree, 2),
            amp_db=-11.0 - 0.4 * k,
            velocity=0.64 - 0.02 * k,
            pitch_motion=PitchMotionSpec.vibrato(depth_ratio=0.005, rate_hz=4.8),
            label=f"motif;deg={degree};oct=2;idx={k};stmt=bridge_descent",
        )
    # S3: sparse fragments answering the riff (first three notes only).
    for bar, octave in ((77, 3), (93, 3)):
        t = _pos(bar)
        for idx, (degree, beats) in enumerate(_WAVE_MOTIF[:3]):
            dur = beats * BEAT
            score.add_note(
                "lead",
                start=t,
                duration=dur * 1.6,
                partial=_partial(degree, octave),
                amp_db=-12.5,
                velocity=0.6,
                pitch_motion=PitchMotionSpec.vibrato(depth_ratio=0.006, rate_hz=5.0),
                label=f"motif;deg={degree};oct={octave};idx={idx};stmt=fragment",
            )
            t += dur
    _place_motif(
        score, "lead", _pos(105), _WAVE_MOTIF, octave=3, amp_db=-11.5, stmt="open_water"
    )
    # S4: the undertow sings the inversion, slow — the emotional low point.
    _place_motif(
        score,
        "lead",
        _pos(117),
        _UNDERTOW_MOTIF,
        octave=2,
        time_scale=2.0,
        amp_db=-10.0,
        vel=0.64,
        ring=1.8,
        stmt="undertow",
    )
    _place_motif(
        score,
        "lead",
        _pos(129),
        _UNDERTOW_MOTIF,
        octave=2,
        time_scale=2.0,
        amp_db=-10.5,
        vel=0.6,
        ring=1.8,
        stmt="undertow",
    )
    # The golden-section climb (≈ bar 139): the wave motif augmented,
    # exposed over the rising pulse, crescendo written into velocity.
    t = _pos(137)
    for idx, (degree, beats) in enumerate(_WAVE_MOTIF):
        dur = beats * BEAT * 2.4
        score.add_note(
            "lead",
            start=t,
            duration=dur * 1.9,
            partial=_partial(degree, 2),
            amp_db=-9.0 + idx * 0.5,
            velocity=0.6 + 0.07 * idx,
            pitch_motion=PitchMotionSpec.vibrato(
                depth_ratio=0.004 + 0.0006 * idx, rate_hz=5.0
            ),
            label=f"motif;deg={degree};oct=2;idx={idx};stmt=golden_climb",
        )
        t += dur
    # S5: statements riding the wave, then doubled-time as the web spins.
    _place_motif(score, "lead", _pos(149), _WAVE_MOTIF, octave=3, stmt="ninth_wave")
    _place_motif(
        score, "lead", _pos(165), _WAVE_MOTIF, octave=3, amp_db=-10.5, stmt="canon_lead"
    )
    _place_motif(
        score, "lead", _pos(181), _WAVE_MOTIF, octave=2, amp_db=-10.0, stmt="canon_lead"
    )
    _place_motif(
        score,
        "lead",
        _pos(195),
        _WAVE_MOTIF,
        octave=3,
        time_scale=0.5,
        amp_db=-11.0,
        stmt="spin",
    )
    # S6: the garden motif once more, then the held 7/4 of the final dyad.
    _place_motif(
        score,
        "lead",
        _pos(205),
        _GARDEN_MOTIF,
        octave=2,
        time_scale=2.0,
        amp_db=-12.0,
        vel=0.6,
        ring=1.9,
        stmt="garden_return",
    )
    score.add_note(
        "lead",
        start=_pos(213),
        duration=TOTAL_DUR - _pos(213),
        partial=_partial(8, 2),
        amp_db=-13.0,
        velocity=0.55,
        pitch_motion=PitchMotionSpec.vibrato(depth_ratio=0.004, rate_hz=4.6),
        label="motif;deg=8;oct=2;stmt=final_seven",
    )


# ---------------------------------------------------------------------------
# Tide arp: a machine current under the open water and the ninth wave.
# Deterministic up-down over chord tones with density gating — a colder,
# more mechanical cousin of hexany_garden's organic walker.
# ---------------------------------------------------------------------------


def _place_arp(score: Score) -> None:
    rng = random.Random(97)
    active = list(range(73, S4_BAR)) + list(range(S5_BAR, 199))
    for bar in active:
        _, chord, _, name = _slot_at_bar(bar)
        in_s5 = bar >= S5_BAR
        density = 0.78 if in_s5 else 0.7
        if S5_BAR <= bar < S5_BAR + 4:
            density = 1.0  # the bell-arp bursts out of the blackness at full spin
        ranked = sorted(chord, key=lambda d: _DEGREE_RATIOS[d])
        # Up-down cycle over the chord across two octaves; S5 reaches one
        # octave higher every other bar.
        top_oct = 3 if (in_s5 and bar % 2 == 0) else 2
        cycle = [(d, 2) for d in ranked] + [(d, top_oct) for d in ranked]
        cycle = cycle + cycle[-2:0:-1]  # palindrome
        phase = (bar - active[0]) * 3  # rotate the pattern each bar
        for step in range(16):
            if rng.random() > density:
                continue
            degree, octave = cycle[(step + phase) % len(cycle)]
            accent = 0.12 if step % 4 == 0 else (0.05 if step % 2 == 0 else 0.0)
            swing = ARP_SWING_S if step % 2 else 0.0
            score.add_note(
                "arp",
                start=_pos(bar) + step * S16 + swing,
                duration=S16 * 0.85,
                partial=_partial(degree, octave),
                amp_db=-11.5 if in_s5 else -14.0,
                velocity=min(1.0, 0.5 + accent + rng.uniform(-0.04, 0.04)),
                label=f"arp;deg={degree};oct={octave};step={step};chord={name}",
            )


# ---------------------------------------------------------------------------
# Glimmer: high delayed answers — the spray off the wave.
# ---------------------------------------------------------------------------


def _place_glimmer(score: Score) -> None:
    rng = random.Random(59)
    # S2/S3: sparse high answers to the lead's motif statements — the bell
    # world is present from the first wave on, so its S4 blossoming is a
    # return, not an arrival.
    for slot_start, n_notes in ((43, 2), (59, 3), (79, 2), (95, 3), (107, 2)):
        chord = _chord_at_bar(slot_start)
        beat = rng.choice((2.5, 3.0, 3.5))
        for k in range(n_notes):
            degree = chord[rng.randrange(len(chord))]
            score.add_note(
                "glimmer",
                start=_pos(slot_start, beat) + k * DOTTED_EIGHTH,
                duration=S8,
                partial=_partial(degree, 4),
                amp_db=-18.0,
                velocity=0.4 + 0.1 * rng.random(),
                label=f"glimmer;deg={degree};oct=4;role=early_answer",
            )
    # S4: one fragile answer per slot, high above the rootless water.
    for slot_start in (114, 118, 122, 126, 130, 134):
        chord = _chord_at_bar(slot_start)
        n_notes = rng.choice((2, 3))
        beat = rng.choice((2.0, 2.5, 3.0))
        for k in range(n_notes):
            degree = chord[rng.randrange(len(chord))]
            score.add_note(
                "glimmer",
                start=_pos(slot_start, beat) + k * DOTTED_EIGHTH,
                duration=S8,
                partial=_partial(degree, 4),
                amp_db=-17.0,
                velocity=0.42 + 0.12 * rng.random(),
                label=f"glimmer;deg={degree};oct=4",
            )
    # S5: canon answers — the wave motif one bar behind the lead, above it.
    for lead_bar in (165, 181):
        _place_motif(
            score,
            "glimmer",
            _pos(lead_bar + 1),
            _WAVE_MOTIF,
            octave=4,
            amp_db=-13.5,
            vel=0.55,
            ring=1.3,
            vibrato=False,
            stmt="canon_answer",
        )
    # S6: the answer chord's 5/4 arrives as a glimmer bloom at 217.
    for k, degree in enumerate((3, 6)):
        score.add_note(
            "glimmer",
            start=_pos(217) + k * BEAT,
            duration=8 * BAR,
            partial=_partial(degree, 3),
            amp_db=-18.0,
            velocity=0.45,
            label=f"glimmer;deg={degree};oct=3;role=answer_bloom",
        )


# ---------------------------------------------------------------------------
# Haze: dekany-quantized grain spray — S1, the undertow, and slack water.
# ---------------------------------------------------------------------------


def _place_haze(score: Score) -> None:
    for start_bar, dur_bars, degree, octave, amp in (
        (1, 24, 0, 1, -24.5),  # distant in the ebb; a memory, not a presence
        # The undertow's cloud wanders the lattice instead of freezing:
        (S4_BAR, 8, 9, 1, -21.0),
        (121, 8, 4, 1, -21.5),
        (129, 8, 5, 2, -22.5),
        (137, 6, 1, 2, -20.0),  # the climb's rising spray, cut before the slam
        (S6_BAR, 23, 0, 1, -21.0),
    ):
        score.add_note(
            "haze",
            start=_pos(start_bar),
            duration=dur_bars * BAR,
            partial=_partial(degree, octave),
            amp_db=amp,
            velocity=0.5,
            label=f"haze;deg={degree};oct={octave}",
        )


# ---------------------------------------------------------------------------
# Drums
# ---------------------------------------------------------------------------


def _place_kick(score: Score) -> None:
    def kick(bar: int, beat: float, amp: float, vel: float) -> None:
        score.add_note(
            "kick",
            start=_pos(bar, beat),
            duration=0.3,
            freq=F0 * 0.5,
            amp_db=amp,
            velocity=vel,
        )

    for bar in range(S2_BAR, 33):  # walking in: 1 and 3
        kick(bar, 1.0, -8.5, 0.8)
        kick(bar, 3.0, -9.5, 0.72)
    for bar in range(33, 41):
        kick(bar, 1.0, -7.5, 0.85)
        kick(bar, 3.0, -8.0, 0.8)
    for bar in range(41, S4_BAR):  # four on the floor
        if bar in _DROP_BARS:
            continue
        for beat in (1.0, 2.0, 3.0, 4.0):
            # One-beat breath before each 16-bar phrase head, and a hard
            # cut on the last two beats of the bridge into open water.
            if beat == 4.0 and bar % 16 == 15:
                continue
            if bar == 64 and beat >= 3.0:
                continue
            kick(bar, beat, -6.5 if beat == 1.0 else -7.0, 0.9 if beat == 1.0 else 0.85)
        if (bar - 41) % 8 == 7:
            kick(bar, 4.75, -11.5, 0.55)  # phrase-end push
    # S4: silence, then the heartbeat gathering from 129 — gone again for
    # the two black bars before the slam.
    for bar in range(129, 143):
        kick(bar, 1.0, -9.5, 0.7)
        if bar >= 137:
            kick(bar, 3.0, -10.0, 0.66)
    for bar in range(S5_BAR, 199):  # the ninth wave: full floor
        for beat in (1.0, 2.0, 3.0, 4.0):
            if beat == 4.0 and bar % 16 == 15:
                continue  # phrase-head breath
            kick(
                bar, beat, -6.0 if beat == 1.0 else -6.5, 0.92 if beat == 1.0 else 0.86
            )
        if (bar - S5_BAR) % 8 == 7:
            kick(bar, 4.75, -11.0, 0.55)
    for k, bar in enumerate(range(S6_BAR, 209)):  # fading heartbeat
        kick(bar, 1.0, -9.0 - k, 0.7)


def _place_clap(score: Score) -> None:
    active = list(range(49, 97)) + list(range(101, S4_BAR)) + list(range(S5_BAR, 199))
    for bar in active:
        for beat in (2.0, 4.0):
            score.add_note(
                "clap",
                start=_pos(bar, beat),
                duration=0.3,
                freq=1200.0,
                amp_db=-11.0 if bar >= S5_BAR else -12.5,
                velocity=0.78 if beat == 2.0 else 0.72,
            )
        # Ghost clap pushing into phrase heads.
        if bar % 8 == 0 and bar >= 65:
            score.add_note(
                "clap",
                start=_pos(bar, 4.75),
                duration=0.2,
                freq=1200.0,
                amp_db=-17.0,
                velocity=0.4,
            )
        # S5 shuffle ghost on the and-of-2, every fourth bar.
        if bar >= S5_BAR and bar % 4 == 2:
            score.add_note(
                "clap",
                start=_pos(bar, 2.75),
                duration=0.15,
                freq=1200.0,
                amp_db=-18.5,
                velocity=0.35,
            )


def _place_hats(score: Score) -> None:
    def hat(
        voice: str, bar: int, step: int, vel: float, amp: float, dur: float = 0.05
    ) -> None:
        score.add_note(
            voice,
            start=_pos(bar) + step * S16 + (HAT_SWING_S if step % 2 else 0.0),
            duration=dur,
            freq=4600.0,
            amp_db=amp,
            velocity=vel,
        )

    # Offbeat 8ths through the driving sections.
    for bar in range(33, 199):
        if bar in _DROP_BARS or S4_BAR <= bar < 137 or bar in (143, 144):
            continue
        if bar in (88, 176):
            continue  # one-bar hat dropout mid-section — the floor breathes
        in_s5 = bar >= S5_BAR
        climb = 137 <= bar < 143
        base = -10.5 if in_s5 else (-14.0 if climb else -12.0)
        for step in (2, 6, 10, 14):
            hat("hat_c", bar, step, 0.66 if step in (6, 14) else 0.56, base)
        # S3/S5 add quiet 16th ticks between the offbeats — in four-bar
        # waves rather than wall-to-wall, so the top end keeps moving.
        ticks_s3 = 73 <= bar < S4_BAR and (bar // 4) % 2 == 1
        ticks_s5 = in_s5 and bar >= 161 and (bar - 161) % 8 != 7
        if ticks_s3 or ticks_s5:
            for step in (1, 3, 5, 7, 9, 11, 13, 15):
                hat("hat_c", bar, step, 0.3, base - 6.5)
    # Open hat accents at phrase ends; every 2 bars in S5.
    for bar in range(36, 199, 4):
        if bar in _DROP_BARS or S4_BAR <= bar < 145:
            continue
        hat("hat_o", bar, 14, 0.62, -13.0, dur=0.45)
    for bar in range(S5_BAR + 1, 199, 2):
        hat("hat_o", bar, 6, 0.5, -15.0, dur=0.35)


def _place_shaker(score: Score) -> None:
    # 16th bed in the driving stretches.
    for bar in list(range(73, 97)) + list(range(105, S4_BAR)) + list(range(153, 199)):
        for step in range(16):
            vel = 0.55 if step % 4 == 0 else (0.42 if step % 2 == 0 else 0.3)
            score.add_note(
                "shaker",
                start=_pos(bar) + step * S16 + (HAT_SWING_S if step % 2 else 0.0),
                duration=0.1,
                freq=950.0,
                amp_db=-16.0,
                velocity=vel,
            )
    # Rising rolls into the section slams.  (No roll into S5 — the wave
    # arrives out of two bars of blackness instead.)
    for roll_bar, n_bars in ((63, 2), (111, 2)):
        total = n_bars * 16
        for k in range(total):
            rise = k / (total - 1)
            score.add_note(
                "shaker",
                start=_pos(roll_bar) + k * S16,
                duration=0.09,
                freq=950.0,
                amp_db=-17.0 + 6.0 * rise,
                velocity=0.3 + 0.55 * rise,
            )


def _place_rim(score: Score) -> None:
    """A dry clave tick cycling every three 16ths — a 3-against-4 rotor
    that gives the late driving sections their inner clockwork."""
    active = list(range(89, 97)) + list(range(105, S4_BAR)) + list(range(161, 199))
    origin = active[0]
    for bar in active:
        for step in range(16):
            if ((bar - origin) * 16 + step) % 3 != 0:
                continue
            score.add_note(
                "rim",
                start=_pos(bar) + step * S16,
                duration=0.06,
                freq=F0 * 8.0,
                amp_db=-19.0,
                velocity=0.5 if step % 4 == 0 else 0.42,
            )


# ---------------------------------------------------------------------------
# Buses and automation
# ---------------------------------------------------------------------------


def _make_hall_bus() -> SendBusSpec:
    return SendBusSpec(
        name="hall",
        effects=[
            bricasti_or_reverb(
                "1 Halls 07 Large & Dark",
                1.0,
                room_size=0.82,
                damping=0.5,
                lowpass_hz=5500.0,
                highpass_hz=220.0,
            )
        ],
        return_db=0.0,
    )


def _ramp(
    target: AutomationTarget,
    points: list[tuple[float, float, float, AutomationShape]],
    default: float,
) -> AutomationSpec:
    """Replace-mode automation from (start, end, to_value, shape) rows."""
    segments: list[AutomationSegment] = []
    prev_value = default
    prev_end = 0.0
    for start, end, to_value, shape in points:
        if start > prev_end and segments:
            segments.append(
                AutomationSegment(
                    start=prev_end, end=start, shape="hold", value=prev_value
                )
            )
        segments.append(
            AutomationSegment(
                start=start,
                end=end,
                shape=shape,
                start_value=prev_value,
                end_value=to_value,
            )
        )
        prev_value = to_value
        prev_end = end
    if prev_end < TOTAL_DUR:
        segments.append(
            AutomationSegment(
                start=prev_end, end=TOTAL_DUR, shape="hold", value=prev_value
            )
        )
    return AutomationSpec(
        target=target,
        segments=tuple(segments),
        default_value=default,
        mode="replace",
    )


def _stab_cutoff_arc() -> AutomationSpec:
    """The riff's filter is the tide line: it opens as each wave builds."""
    return _ramp(
        AutomationTarget(kind="synth", name="cutoff_hz"),
        [
            (_pos(33), _pos(49), 2000.0, "exp"),  # first wave opens
            (_pos(57), _pos(65), 2600.0, "exp"),  # into open water
            (_pos(89), _pos(97), 1500.0, "exp"),  # closing toward the drop
            (_pos(101), _pos(112), 3000.0, "exp"),  # the slam shines
            (_pos(113), _pos(120), 900.0, "exp"),  # undertow (riff silent)
            (_pos(137), _pos(148), 3200.0, "exp"),  # the climb bursts open
            (_pos(181), _pos(195), 3800.0, "exp"),  # spinning peak
            (_pos(199), _pos(213), 1200.0, "exp"),  # slack water closes down
        ],
        default=1300.0,
    )


def _stab_delay_throw() -> AutomationSpec:
    """Delay blooms at the seams: the riff's tail washes forward."""
    return _ramp(
        AutomationTarget(kind="control", name="mix"),
        [
            (_pos(63), _pos(66), 0.38, "linear"),
            (_pos(69), _pos(73), 0.18, "linear"),
            (_pos(95), _pos(97), 0.45, "linear"),  # runaway into the drop
            (_pos(101), _pos(103), 0.18, "linear"),
            (_pos(111), _pos(113), 0.42, "linear"),
            (_pos(143), _pos(145), 0.40, "linear"),
            (_pos(147), _pos(151), 0.20, "linear"),
            (_pos(197), _pos(201), 0.46, "linear"),
            (_pos(205), _pos(209), 0.22, "linear"),
        ],
        default=0.16,
    )


def _stab_hall_ride() -> AutomationSpec:
    return _ramp(
        AutomationTarget(kind="control", name="send_db"),
        [
            (_pos(95), _pos(101), -5.0, "linear"),
            (_pos(103), _pos(109), -8.0, "linear"),
            (_pos(197), _pos(205), -4.0, "linear"),
        ],
        default=-8.0,
    )


def _pad_brightness_arc() -> AutomationSpec:
    return _ramp(
        AutomationTarget(kind="synth", name="brightness_tilt"),
        [
            (0.0, S1_END, -0.05, "linear"),
            (S1_END, S3_END, 0.04, "linear"),
            (S3_END, _pos(137), -0.14, "linear"),  # undertow darkens
            (_pos(137), _pos(161), 0.08, "linear"),  # the wave glows
            (S5_END, TOTAL_DUR, -0.10, "linear"),
        ],
        default=-0.10,
    )


def _pad_hall_ride() -> AutomationSpec:
    """The undertow and the slack water are wetter; the wave pulls close."""
    return _ramp(
        AutomationTarget(kind="control", name="send_db"),
        [
            (S3_END - 2 * BAR, S3_END + 2 * BAR, -4.0, "linear"),
            (S4_END - 2 * BAR, S4_END + 2 * BAR, -6.5, "linear"),
            (S5_END, S5_END + 4 * BAR, -4.0, "linear"),
        ],
        default=-6.0,
    )


def _bass_cutoff_arc() -> AutomationSpec:
    """The bass opens up as the tide rises."""
    return _ramp(
        AutomationTarget(kind="synth", name="cutoff_hz"),
        [
            (_pos(49), _pos(65), 620.0, "exp"),
            (_pos(113), _pos(121), 320.0, "exp"),  # seafloor: subs only
            (_pos(137), _pos(149), 750.0, "exp"),
            (_pos(199), _pos(213), 380.0, "exp"),
        ],
        default=480.0,
    )


def _arp_cutoff_arc() -> AutomationSpec:
    """The arp brightens with each wave and rings out fully at the peak."""
    return _ramp(
        AutomationTarget(kind="synth", name="cutoff_hz"),
        [
            (_pos(73), _pos(97), 1900.0, "exp"),
            (_pos(101), S3_END, 2200.0, "exp"),
            (S4_END, _pos(161), 2600.0, "exp"),
            (_pos(185), S5_END, 3200.0, "exp"),
            (S5_END, S5_END + 4 * BAR, 1200.0, "exp"),
        ],
        default=1500.0,
    )


def _arp_delay_throw() -> AutomationSpec:
    """The bell-arp's tail washes wide as it bursts out of the blackness."""
    return _ramp(
        AutomationTarget(kind="control", name="mix"),
        [
            (S4_END, S4_END + 2 * BAR, 0.34, "linear"),
            (S4_END + 4 * BAR, S4_END + 8 * BAR, 0.16, "linear"),
        ],
        default=0.16,
    )


def _arp_pan_sway() -> AutomationSpec:
    return _ramp(
        AutomationTarget(kind="control", name="pan"),
        [
            (S2_END, S3_END, 0.22, "linear"),
            (S4_END, _pos(177), -0.18, "linear"),
            (_pos(177), S5_END, 0.15, "linear"),
        ],
        default=-0.12,
    )


def _haze_mix_ride() -> AutomationSpec:
    return _ramp(
        AutomationTarget(kind="control", name="mix_db"),
        [
            (S3_END - 2 * BAR, S3_END + 2 * BAR, -14.0, "linear"),
            (_pos(137), _pos(142), -13.5, "linear"),  # the climb's spray swells
            (_pos(142), _pos(143), -26.0, "linear"),  # cut into the blackness
            (S5_END, S5_END + 4 * BAR, -15.0, "linear"),
        ],
        default=-17.0,
    )


# ---------------------------------------------------------------------------
# build_score
# ---------------------------------------------------------------------------


# The default master chain plus a gentle air shelf: the piece is
# deliberately sub-forward (49 Hz kick + bass), so the top end gets lift
# rather than the bottom getting cut.
_MASTER_EFFECTS = [
    EffectSpec(
        "eq",
        {
            "bands": [
                {"kind": "high_shelf", "freq_hz": 6800.0, "gain_db": 2.0},
            ]
        },
    ),
    *DEFAULT_MASTER_EFFECTS,
]


def build_score() -> Score:
    """Build The Ninth Wave score."""
    score = Score(
        f0_hz=F0,
        master_effects=_MASTER_EFFECTS,
        # Pad + stabs sustain heavily from S2 on; trim keeps the master
        # vari-mu breathing instead of pinned.
        master_input_gain_db=-2.5,
        send_buses=[_make_hall_bus()],
        timing_humanize=TimingHumanizeSpec(preset="tight_ensemble"),
    )
    score.add_drift_bus("tide_drift", rate_hz=0.07, depth_cents=4.0, seed=13579)
    drum_bus = setup_drum_bus(score, style="electronic", return_db=0.0)

    def kick_duck(threshold_db: float, ratio: float) -> EffectSpec:
        return EffectSpec(
            "compressor",
            {
                "threshold_db": threshold_db,
                "ratio": ratio,
                "attack_ms": 2.0,
                "release_ms": 140.0,
                "lookahead_ms": 5.0,
                "sidechain_source": "kick",
                "detector_mode": "peak",
            },
        )

    # ---- Stab: the string-machine riff ----
    score.add_voice(
        "stab",
        synth_defaults={
            "engine": "va",
            "preset": "supersaw_pad",
            "attack": 0.008,
            "decay": 0.18,
            "sustain": 0.75,
            "release": 0.28,
            "supersaw_detune": 0.42,
            "supersaw_mix": 0.6,
            "cutoff_hz": 1300.0,
            "resonance_q": 1.1,
            "hpf_cutoff_hz": 160.0,
            "drive_amount": 0.18,  # Virus-style pre-filter warmth
        },
        effects=[
            EffectSpec("bbd_chorus", {"preset": "juno_ii", "mix": 0.5}),
            EffectSpec(
                "delay",
                {"delay_seconds": DOTTED_EIGHTH, "feedback": 0.34, "mix": 0.16},
                automation=[_stab_delay_throw()],
            ),
            kick_duck(-24.0, 2.5),
        ],
        sends=[VoiceSend(target="hall", send_db=-8.0, automation=[_stab_hall_ride()])],
        pan=-0.08,
        mix_db=-3.0,
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        drift_bus="tide_drift",
        drift_bus_correlation=0.7,
        automation=[_stab_cutoff_arc()],
    )

    # ---- Bass: G1 roots ----
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "soft_bass",
            "release": 0.35,
            # Brightness + dirt buy audible mid harmonics, so the bass
            # reads on small speakers without adding more sub energy.
            "brightness": 0.6,
            "dirt": 0.3,
        },
        effects=[kick_duck(-21.0, 4.0)],
        pan=0.0,
        mix_db=-6.5,
        max_polyphony=1,
        legato=True,
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        drift_bus="tide_drift",
        drift_bus_correlation=0.9,
        automation=[_bass_cutoff_arc()],
    )

    # ---- Pad: dekany-spectrum chords ----
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "additive_pad_through_ladder",
            "partials_partials": _DEKANY_SPECTRUM,
            "filter_cutoff_hz": 1050.0,
            "attack": 3.0,
            "release": 4.5,
            "brightness_tilt": -0.10,
        },
        effects=[
            EffectSpec(
                "compressor",
                {
                    "threshold_db": -30.0,
                    "ratio": 2.0,
                    "attack_ms": 4.0,
                    "release_ms": 230.0,
                    "lookahead_ms": 5.0,
                    "sidechain_source": "kick",
                    "detector_mode": "peak",
                    "detector_bands": [
                        {"kind": "lowpass", "cutoff_hz": 150.0, "slope_db_per_oct": 12}
                    ],
                },
            ),
        ],
        sends=[VoiceSend(target="hall", send_db=-6.0, automation=[_pad_hall_ride()])],
        pan=0.0,
        mix_db=-10.5,
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        drift_bus="tide_drift",
        drift_bus_correlation=0.8,
        automation=[_pad_brightness_arc()],
    )

    # ---- Lead: the vowel voice ----
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "formant_vowel_lead",
            "attack": 0.06,
            "release": 1.6,
        },
        sends=[VoiceSend(target="hall", send_db=-4.5)],
        pan=0.12,
        mix_db=-8.5,
        max_polyphony=2,
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        drift_bus="tide_drift",
        drift_bus_correlation=0.6,
    )

    # ---- Tide arp ----
    score.add_voice(
        "arp",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "acid_pluck",
            "release": 0.25,
            "filter_cutoff_hz": 1500.0,
        },
        effects=[
            EffectSpec(
                "delay",
                {"delay_seconds": 3.0 * S16, "feedback": 0.28, "mix": 0.16},
                automation=[_arp_delay_throw()],
            ),
            kick_duck(-28.0, 2.0),
        ],
        sends=[VoiceSend(target="hall", send_db=-10.0)],
        pan=-0.12,
        mix_db=-6.5,
        envelope_humanize=EnvelopeHumanizeSpec(preset="loose_pluck"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        drift_bus="tide_drift",
        drift_bus_correlation=0.5,
        automation=[_arp_pan_sway(), _arp_cutoff_arc()],
    )

    # ---- Glimmer: the spray ----
    score.add_voice(
        "glimmer",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "two_op_bell",
            "release": 1.8,
        },
        effects=[
            EffectSpec(
                "delay",
                {"delay_seconds": 5.0 * S16, "feedback": 0.42, "mix": 0.35},
            ),
        ],
        sends=[VoiceSend(target="hall", send_db=-3.5)],
        pan=0.38,
        mix_db=-9.0,
        envelope_humanize=EnvelopeHumanizeSpec(preset="loose_pluck"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        drift_bus="tide_drift",
        drift_bus_correlation=0.4,
    )

    # ---- Haze: dekany-quantized grain spray ----
    score.add_voice(
        "haze",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "grain_breathing_cloud",
            "grain_ji_lattice": list(_DEGREE_RATIOS),
        },
        sends=[VoiceSend(target="hall", send_db=-1.5)],
        pan=0.0,
        mix_db=-17.0,
        automation=[_haze_mix_ride()],
    )

    # ---- Drums ----
    add_drum_voice(
        score,
        "kick",
        engine="drum_voice",
        preset="909_techno",
        drum_bus=drum_bus,
        send_db=-3.0,
        # Snappier than the stock preset: shorter body, harder transient.
        synth_overrides={
            "tone_decay_s": 0.22,
            "tone_punch": 0.4,
            "exciter_level": 0.24,
        },
        mix_db=-5.0,
    )
    add_drum_voice(
        score,
        "clap",
        engine="drum_voice",
        preset="909_clap",
        drum_bus=drum_bus,
        send_db=-4.0,
        mix_db=-11.5,
        pan=-0.08,
    )
    hat_darkening = {
        "metallic_brightness": 0.55,
        "metallic_hat_noise_bp_hz": 6400.0,
    }
    add_drum_voice(
        score,
        "hat_c",
        engine="drum_voice",
        preset="closed_hat",
        drum_bus=drum_bus,
        send_db=-8.0,
        synth_overrides=hat_darkening,
        choke_group="hats",
        mix_db=-15.5,
        pan=0.16,
    )
    add_drum_voice(
        score,
        "hat_o",
        engine="drum_voice",
        preset="open_hat",
        drum_bus=drum_bus,
        send_db=-7.0,
        synth_overrides=hat_darkening,
        choke_group="hats",
        mix_db=-16.5,
        pan=0.16,
    )
    add_drum_voice(
        score,
        "shaker",
        engine="drum_voice",
        preset="brush",
        drum_bus=drum_bus,
        send_db=-5.0,
        mix_db=-15.0,
        pan=-0.24,
    )
    add_drum_voice(
        score,
        "rim",
        engine="drum_voice",
        preset="clave",
        drum_bus=drum_bus,
        send_db=-9.0,
        mix_db=-16.0,
        pan=0.3,
    )

    # ==================================================================
    # Notes
    # ==================================================================
    _place_pad(score)
    _place_stab(score)
    _place_bass(score)
    _place_lead(score)
    _place_arp(score)
    _place_glimmer(score)
    _place_haze(score)
    _place_kick(score)
    _place_clap(score)
    _place_hats(score)
    _place_shaker(score)
    _place_rim(score)

    return score


PIECES: dict[str, PieceDefinition] = {
    "ninth_wave": PieceDefinition(
        name="ninth_wave",
        output_name="ninth_wave",
        build_score=build_score,
        sections=(
            PieceSection(label="S1 ebb", start_seconds=0.0, end_seconds=S1_END),
            PieceSection(
                label="S2 first wave", start_seconds=S1_END, end_seconds=S2_END
            ),
            PieceSection(
                label="S3 open water", start_seconds=S2_END, end_seconds=S3_END
            ),
            PieceSection(label="S4 undertow", start_seconds=S3_END, end_seconds=S4_END),
            PieceSection(
                label="S5 ninth wave", start_seconds=S4_END, end_seconds=S5_END
            ),
            PieceSection(
                label="S6 slack water", start_seconds=S5_END, end_seconds=TOTAL_DUR
            ),
        ),
    ),
}
