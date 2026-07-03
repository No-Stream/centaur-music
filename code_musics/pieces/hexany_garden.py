"""Hexany Garden — Erv Wilson's 1-3-5-7 hexany as a wandering garden.

The 2-of-4 Combination Product Set on {1, 3, 5, 7} gives six notes —
1/1, 7/6, 5/4, 35/24, 5/3, 7/4 — with *no tonic and no perfect fifth*.
Every note is a product of two primes, and the scale contains exactly
eight triads: four otonal (each fixes a shared factor) and four utonal
(each excludes one).  Harmony here is not a progression away from and
back to a home key; it is drift across a web where every chord shares
two of its three notes with several neighbors.  The otonal side of the
web is the garden in sunlight; the utonal side is the same garden after
dark.

The melodic spine is a seeded walker over the hexany graph: notes that
share a factor are adjacent and easy to move between; the three "polar"
pairs (1·3 vs 5·7, 1·5 vs 3·7, 1·7 vs 3·5) share nothing and are only
crossed in the darker middle sections, as leaps.  A five-note seed
motif — stated slowly by the bell before the beat exists — is the one
fixed object; everything melodic that is not the walker is a transform
of it.

Tuning: 1-3-5-7 hexany (via ``code_musics.tuning.hexany``) on
f0 = F#2 = 92.5 Hz — kick-friendly per house rules, and a key no other
piece in the repo sits in.  The pad's additive spectrum is built from
the same product set, so timbre and harmony come from one ratio world.

BPM = 92.  1 bar ≈ 2.609 s.  132 bars ≈ 5:44.

Form:
  bars   1– 16  S1 Dew           hexany-spectrum drone; grain dust; the
                                 bell states the seed motif, augmented
  bars  17– 48  S2 First bloom   walker arp + soft beat; otonal regions
                                 only (4:5:7 home, 4:5:6 lift, 3:5:7
                                 bright); bass anchors the region roots
  bars  49– 76  S3 The turn      bar 49 drops to bass + the utonally
                                 mirrored motif — the first silence
                                 since the beat.  The walker re-enters
                                 low and half-time; a 7-eighth thumb
                                 line starts phasing against 4/4; polar
                                 leaps unlock
  bars  77–108  S4 Full garden   otonal and utonal regions interlock;
                                 motif in canon (bell + thread); glint
                                 voice answers from above; the 6:7:8
                                 cluster blooms at the golden section;
                                 harmonic rhythm doubles to 2-bar slots
                                 from 101; two held-breath bars at 99
                                 before the slam
  bars 109–132  S5 Seed          drums fall away; walker slows to
                                 quarter-notes; ends on the bare 4:7
                                 dyad (1/1 + 7/4) — neither major nor
                                 minor, the question the piece asked

Composed by Claude (Fable 5), July 2026.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

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
from code_musics.tuning import hexany

# ---------------------------------------------------------------------------
# Timing grid
# ---------------------------------------------------------------------------

BPM: float = 92.0
BEAT: float = 60.0 / BPM
BAR: float = 4.0 * BEAT
S16: float = BEAT / 4.0
S8: float = BEAT / 2.0
DOTTED_EIGHTH: float = 3.0 * S16

F0: float = 92.5  # F#2
TOTAL_BARS: int = 132

ARP_SWING_S: float = 0.07 * S16
HAT_SWING_S: float = 0.13 * S16


def _pos(bar: int, beat: float = 1.0, n16: int = 0) -> float:
    """Absolute time at bar:beat:sixteenth (1-indexed bar/beat)."""
    return (bar - 1) * BAR + (beat - 1.0) * BEAT + n16 * S16


# Section boundaries (bar numbers; S1 starts at bar 1)
S2_BAR: int = 17
S3_BAR: int = 49
S4_BAR: int = 77
S5_BAR: int = 109
END_BAR: int = TOTAL_BARS + 1

S1_END: float = _pos(S2_BAR)
S2_END: float = _pos(S3_BAR)
S3_END: float = _pos(S4_BAR)
S4_END: float = _pos(S5_BAR)
TOTAL_DUR: float = _pos(END_BAR)

# ---------------------------------------------------------------------------
# The hexany.  Degree indices order the scale ascending within an octave;
# a "pitch index" (octave * 6 + degree) gives a total ordering for the
# walker's contour logic.  Each degree is a product of two of {1,3,5,7};
# degrees that share a factor are graph-adjacent, and the three polar
# pairs share nothing.
# ---------------------------------------------------------------------------

_DEGREE_RATIOS: tuple[float, ...] = tuple(hexany())  # (1, 7/6, 5/4, 35/24, 5/3, 7/4)
_N_DEGREES: int = len(_DEGREE_RATIOS)

_DEGREE_FACTORS: tuple[frozenset[int], ...] = (
    frozenset({1, 3}),  # 1/1    = 1·3
    frozenset({1, 7}),  # 7/6    = 1·7
    frozenset({3, 5}),  # 5/4    = 3·5
    frozenset({5, 7}),  # 35/24  = 5·7
    frozenset({1, 5}),  # 5/3    = 1·5
    frozenset({3, 7}),  # 7/4    = 3·7
)

# The eight triad regions, as degree triples.  Otonal regions fix a
# common factor; utonal regions exclude one.
_HOME: tuple[int, ...] = (0, 2, 5)  # fix 3: 1/1 5/4 7/4      = 4:5:7
_LIFT: tuple[int, ...] = (1, 3, 5)  # fix 7: 7/6 35/24 7/4    = 4:5:6 on 7/6
_BRIGHT: tuple[int, ...] = (0, 1, 4)  # fix 1: 1/1 7/6 5/3    = 6:7:10
_CLUSTER: tuple[int, ...] = (2, 3, 4)  # fix 5: 5/4 35/24 5/3  = 6:7:8
_U_MINOR: tuple[int, ...] = (0, 2, 4)  # excl 7: 1/1 5/4 5/3   (just minor)
_U_SUB: tuple[int, ...] = (0, 1, 5)  # excl 5: 1/1 7/6 7/4     (subminor 7)
_U_SHADE: tuple[int, ...] = (1, 3, 4)  # excl 3
_U_VEIL: tuple[int, ...] = (2, 3, 5)  # excl 1

# Harmonic plan: (start_bar, region).  Each slot runs to the next start.
_CHORD_SLOTS: tuple[tuple[int, tuple[int, ...]], ...] = (
    (1, _HOME),  # S1: implied — drone only
    (17, _HOME),
    (21, _LIFT),
    (25, _HOME),
    (29, _BRIGHT),
    (33, _HOME),
    (37, _LIFT),
    (41, _BRIGHT),
    (45, _HOME),
    (49, _U_MINOR),  # S3: the turn
    (53, _U_SUB),
    (57, _U_MINOR),
    (61, _U_SHADE),
    (65, _U_SUB),
    (69, _U_VEIL),
    (73, _U_SUB),
    (77, _HOME),  # S4: interlock
    (81, _U_MINOR),
    (85, _LIFT),
    (89, _U_SUB),
    (93, _CLUSTER),  # the one 6:7:8 bloom, near the golden section
    (97, _U_VEIL),
    (101, _BRIGHT),  # from here the web spins faster: 2-bar slots
    (103, _U_MINOR),
    (105, _HOME),
    (107, _LIFT),
    (109, _HOME),  # S5: seed
    (113, _U_SUB),
    (117, _HOME),
    (121, (0, 5)),  # bare 4:7 dyad to the end
)


def _region_at_bar(bar: int) -> tuple[int, ...]:
    region = _CHORD_SLOTS[0][1]
    for start, slot_region in _CHORD_SLOTS:
        if bar >= start:
            region = slot_region
    return region


def _partial(degree: int, octave: int) -> float:
    """Partial of f0 for a scale degree in an octave (octave 0 = f0..2*f0)."""
    return _DEGREE_RATIOS[degree] * (2.0**octave)


def _shares_factor(deg_a: int, deg_b: int) -> bool:
    return bool(_DEGREE_FACTORS[deg_a] & _DEGREE_FACTORS[deg_b])


# ---------------------------------------------------------------------------
# The seed motif.  Stated by the bell in S1 (augmented), restated at
# section seams, put in canon in S4, and fragmented in S5.
# (degree, duration_beats) at a base octave supplied per statement.
# ---------------------------------------------------------------------------

_MOTIF: tuple[tuple[int, float], ...] = (
    (0, 1.0),
    (2, 0.5),
    (5, 1.5),
    (4, 1.0),
    (2, 2.0),
)


def _place_motif(
    score: Score,
    voice: str,
    start: float,
    *,
    octave: int,
    time_scale: float = 1.0,
    amp_db: float = -10.0,
    vel: float = 0.75,
    ring: float = 1.6,
    vibrato: bool = False,
) -> None:
    t = start
    for degree, beats in _MOTIF:
        dur = beats * BEAT * time_scale
        score.add_note(
            voice,
            start=t,
            duration=dur * ring,
            partial=_partial(degree, octave),
            amp_db=amp_db,
            velocity=vel,
            pitch_motion=(
                PitchMotionSpec.vibrato(depth_ratio=0.006, rate_hz=5.0)
                if vibrato and beats >= 1.0
                else None
            ),
        )
        t += dur


# ---------------------------------------------------------------------------
# The walker arp.  A seeded random walk over (degree, octave) states.
# Movement prefers small steps, chord tones of the active region, and
# factor-adjacent degrees; polar crossings are rare and only unlocked
# in S3/S4.
#
# Structure over drift: the walk is organized into 8-bar riff phrases.
# The first two bars of each phrase are generated fresh (opening with a
# transposed quote of the seed motif at the S2 anchors) and *recorded*;
# the next four replay that cell with one mutation per pass; the last
# two walk free.  Repetition you can latch onto, produced by the same
# walk so nothing sounds pasted in.  The pluck timbre also ages across
# the form: wool in the dew, wire at the peak.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _WalkParams:
    density: float  # probability a grid step sounds
    burst_prob: float  # probability an eighth splits into two 16ths
    chord_pull: float  # weight multiplier for active-region tones
    polar_weight: float  # weight for polar (no shared factor) moves
    pi_min: int  # lowest pitch index (octave * 6 + degree)
    pi_max: int  # highest pitch index
    vel_base: float
    amp_db: float
    pluck_hardness: float  # excitation brightness: wool -> wire
    pluck_damping: float  # KS loop darkness
    pluck_position: float  # pick-position comb


def _walk_params_for_bar(bar: int) -> _WalkParams:
    if bar < S2_BAR:  # S1: sparse condensation out of the drone
        return _WalkParams(0.30, 0.0, 5.0, 0.0, 14, 22, 0.52, -13.0, 0.35, 0.45, 0.35)
    if bar < 45:  # S2: the bloom
        return _WalkParams(0.72, 0.10, 4.0, 0.0, 12, 24, 0.62, -10.5, 0.45, 0.35, 0.30)
    if bar < S3_BAR:  # S2 tail: the garden holds its breath
        return _WalkParams(0.60, 0.06, 4.0, 0.0, 12, 22, 0.58, -11.0, 0.45, 0.40, 0.32)
    # S3: bars 49-52 are silent (the event) — handled in _place_walker.
    if bar < 65:  # S3a: re-entry, low and half-time
        return _WalkParams(0.45, 0.0, 3.0, 0.30, 6, 16, 0.54, -12.0, 0.40, 0.52, 0.40)
    if bar < S4_BAR:  # S3b: recovering its stride, still shadowed
        return _WalkParams(0.62, 0.06, 3.0, 0.30, 8, 20, 0.58, -11.5, 0.50, 0.38, 0.32)
    if bar < 85:  # S4: full garden, register climbing section-long
        return _WalkParams(0.80, 0.14, 3.5, 0.18, 12, 24, 0.64, -10.2, 0.55, 0.30, 0.25)
    if bar < 93:
        return _WalkParams(0.82, 0.16, 3.5, 0.18, 12, 26, 0.66, -10.0, 0.60, 0.26, 0.22)
    if bar < S5_BAR:  # the cluster bloom and the fast-spinning tail
        return _WalkParams(0.85, 0.18, 3.5, 0.18, 14, 28, 0.68, -9.8, 0.65, 0.22, 0.20)
    return _WalkParams(0.38, 0.0, 5.0, 0.0, 12, 21, 0.50, -13.5, 0.35, 0.50, 0.40)


def _pi(degree: int, octave: int) -> int:
    return octave * _N_DEGREES + degree


def _from_pi(pi: int) -> tuple[int, int]:
    return pi % _N_DEGREES, pi // _N_DEGREES


def _walker_step(
    rng: random.Random, cur_pi: int, region: tuple[int, ...], p: _WalkParams
) -> int:
    cur_degree, _ = _from_pi(cur_pi)
    candidates: list[int] = []
    weights: list[float] = []
    for pi in range(p.pi_min, p.pi_max + 1):
        delta = abs(pi - cur_pi)
        if delta > 7:  # cap leaps around an octave
            continue
        degree, _ = _from_pi(pi)
        w = 1.0 / (1.0 + 0.35 * delta)
        if degree in region:
            w *= p.chord_pull
        if not _shares_factor(cur_degree, degree) and degree != cur_degree:
            w *= p.polar_weight  # polar crossing: rare or forbidden
        if pi == cur_pi:
            w *= 0.30  # discourage immediate repetition
        if w <= 0.0:
            continue
        candidates.append(pi)
        weights.append(w)
    return rng.choices(candidates, weights=weights, k=1)[0]


# S4 slot-head bars where the walker adds a dyad hit on the downbeat.
_DYAD_BARS: frozenset[int] = frozenset(
    start for start, _ in _CHORD_SLOTS if S4_BAR <= start < S5_BAR
)

# 8-bar riff phrases: record 2 bars, replay-with-mutation 4, walk free 2.
_RIFF_PHRASE_STARTS: tuple[int, ...] = (17, 25, 33, 41, 77, 85, 93, 101)
# Phrases whose cell opens with the seed motif quoted into the region.
_QUOTE_BARS: frozenset[int] = frozenset({17, 25, 33, 41})
# Bars where the lead leaves 4/4 and joins the thumb's 7-eighth cycle.
_SEVEN_GRID_BARS: frozenset[int] = frozenset({90, 91, 106, 107})
# S3b recovery grid: tresillo (3+3+2) in sixteenths as (offset, held).
_TRESILLO_STEPS: tuple[tuple[int, int], ...] = (
    (0, 3),
    (3, 3),
    (6, 2),
    (8, 3),
    (11, 3),
    (14, 2),
)


@dataclass(frozen=True)
class _ArpNote:
    offset_s: float  # relative to its bar start (swing baked in)
    duration: float
    degree: int
    octave: int
    accent: float


def _quote_notes(region: tuple[int, ...]) -> tuple[list[_ArpNote], int]:
    """The seed motif transposed into a region: 6 beats of a riff cell.

    Motif ranks map onto the region's ranked tones; the motif's one
    non-chord tone becomes a factor-adjacent passing tone.
    """
    ranked = sorted(region)
    passing = next(
        d for d in range(_N_DEGREES) if d not in region and _shares_factor(d, ranked[1])
    )
    degrees = [ranked[0], ranked[1], ranked[2], passing, ranked[1]]
    notes: list[_ArpNote] = []
    t_beats = 0.0
    for (_, beats), degree in zip(_MOTIF, degrees, strict=True):
        accent = 0.12 if t_beats == 0.0 else 0.05
        notes.append(_ArpNote(t_beats * BEAT, beats * BEAT * 0.9, degree, 3, accent))
        t_beats += beats
    return notes, _pi(degrees[-1], 3)


def _gen_cell(
    rng: random.Random,
    first_bar: int,
    n_bars: int,
    cur_pi: int,
    *,
    quote: bool = False,
) -> tuple[list[tuple[int, _ArpNote]], int]:
    """Generate walker material as (bar_offset, note) rows.

    Handles all four grids: straight eighths (with 16th bursts), the
    half-time stretches, the S3b tresillo, and the S4 seven-grid joins.
    """
    rows: list[tuple[int, _ArpNote]] = []
    start_eighth = 0
    if quote:
        quote_notes, cur_pi = _quote_notes(_region_at_bar(first_bar))
        for note in quote_notes:
            bar_off = int(note.offset_s // BAR)
            rows.append(
                (bar_off, replace(note, offset_s=note.offset_s - bar_off * BAR))
            )
        start_eighth = 12  # the quote spans six beats of the cell

    for bar_off in range(n_bars):
        bar = first_bar + bar_off
        p = _walk_params_for_bar(bar)
        region = _region_at_bar(bar)
        in_s4 = S4_BAR <= bar < S5_BAR
        slow = bar >= S5_BAR or 53 <= bar < 65
        swing_s = ARP_SWING_S * (1.35 if in_s4 else 1.0)

        def emit(offset_s: float, duration: float, accent: float) -> None:
            nonlocal cur_pi
            cur_pi = _walker_step(rng, cur_pi, region, p)  # noqa: B023
            degree, octave = _from_pi(cur_pi)
            rows.append(
                (bar_off, _ArpNote(offset_s, duration, degree, octave, accent))  # noqa: B023
            )

        if bar in _SEVEN_GRID_BARS:
            # Join the thumb's 7-cycle for a two-bar phasing stretch.
            pair_start = 90 if bar in (90, 91) else 106
            for k in range(8):
                g = k + 8 * (bar - pair_start)
                if g % 7 not in (0, 2, 5):
                    continue
                if rng.random() > p.density + 0.1:
                    continue
                swing = swing_s if k % 2 else 0.0
                emit(k * S8 + swing, S8 * 0.88, 0.10 if g % 7 == 0 else 0.02)
        elif 65 <= bar < S4_BAR:
            # Tresillo (3+3+2): the walker recovers its stride sideways.
            for offset16, held in _TRESILLO_STEPS:
                if rng.random() > p.density + 0.15:
                    continue
                accent = 0.12 if offset16 in (0, 8) else 0.03
                emit(offset16 * S16, held * S16 * 0.88, accent)
        else:
            steps = range(0, 8, 2) if slow else range(8)
            for step8 in steps:
                if bar_off * 8 + step8 < start_eighth:
                    continue  # the quote owns these beats
                if rng.random() > p.density:
                    continue
                swing = swing_s if step8 % 2 else 0.0
                offset = step8 * S8 + swing
                accent = 0.14 if step8 in (0, 4) else (0.06 if step8 in (2, 6) else 0.0)
                if rng.random() < p.burst_prob:
                    emit(offset, S16 * 0.9, accent)
                    emit(offset + S16, S16 * 0.9, accent - 0.12)
                else:
                    emit(offset, (BEAT if slow else S8) * 0.88, accent)
    return rows, cur_pi


def _mutate_cell(
    rng: random.Random, cell: list[tuple[int, _ArpNote]], region: tuple[int, ...]
) -> None:
    """Nudge one riff note toward the current region, in place."""
    if not cell:
        return
    idx = rng.randrange(len(cell))
    bar_off, note = cell[idx]
    cell[idx] = (bar_off, replace(note, degree=rng.choice(region)))


def _place_walker(score: Score) -> dict[int, int]:
    """The arp spine.  Returns each bar's final walker degree so the
    bass can choose its approach tones by listening to the lead."""
    rng = random.Random(1357)  # the CPS factors, of course
    cur_pi = _pi(0, 3)  # start on 1/1, octave 3
    cell: list[tuple[int, _ArpNote]] = []
    cell_end_pi = cur_pi
    last_degree: dict[int, int] = {}

    def place(bar: int, notes: list[_ArpNote]) -> None:
        p = _walk_params_for_bar(bar)
        region = _region_at_bar(bar)
        in_s4 = S4_BAR <= bar < S5_BAR
        phrase_lift = 0.05 * ((bar - 1) % 4) / 4.0 if in_s4 else 0.0
        pluck = {
            "osc_pluck_hardness": p.pluck_hardness,
            "osc_pluck_damping": p.pluck_damping,
            "osc_pluck_position": p.pluck_position,
        }
        for note in sorted(notes, key=lambda n: n.offset_s):
            start = _pos(bar) + note.offset_s
            vel = min(
                1.0,
                max(
                    0.25,
                    p.vel_base + note.accent + phrase_lift + rng.uniform(-0.04, 0.04),
                ),
            )
            if in_s4 and bar in _DYAD_BARS and note.offset_s < S16:
                # Dyad hit on the slot head: a region-mate under the lead.
                mate = (
                    region[(region.index(note.degree) + 1) % len(region)]
                    if note.degree in region
                    else region[0]
                )
                score.add_note(
                    "arp",
                    start=start,
                    duration=note.duration,
                    partial=_partial(mate, max(1, note.octave - 1)),
                    amp_db=p.amp_db - 3.0,
                    velocity=max(0.3, vel - 0.1),
                    synth=pluck,
                )
            score.add_note(
                "arp",
                start=start,
                duration=note.duration,
                partial=_partial(note.degree, note.octave),
                amp_db=p.amp_db,
                velocity=vel,
                synth=pluck,
                pitch_motion=(
                    PitchMotionSpec.vibrato(depth_ratio=0.004, rate_hz=4.6)
                    if note.duration >= BEAT * 0.8
                    else None
                ),
            )
            last_degree[bar] = note.degree

    bar = 1
    while bar <= TOTAL_BARS:
        if bar < 9 or (S3_BAR <= bar < 53) or bar >= 129:
            bar += 1  # pre-dew silence, the turn, and the final dyad
            continue
        phrase_pos = next(
            (bar - s for s in _RIFF_PHRASE_STARTS if s <= bar < s + 8), None
        )
        if phrase_pos == 0:
            # Record a fresh 2-bar cell (motif quote at the S2 anchors).
            cell, cell_end_pi = _gen_cell(rng, bar, 2, cur_pi, quote=bar in _QUOTE_BARS)
            cur_pi = cell_end_pi
            for off in (0, 1):
                place(bar + off, [n for o, n in cell if o == off])
            bar += 2
        elif phrase_pos in (2, 4) and cell:
            # Replay the riff, one nudge per pass.
            _mutate_cell(rng, cell, _region_at_bar(bar))
            for off in (0, 1):
                place(bar + off, [n for o, n in cell if o == off])
            cur_pi = cell_end_pi
            bar += 2
        else:
            rows, cur_pi = _gen_cell(rng, bar, 1, cur_pi)
            place(bar, [n for _, n in rows])
            bar += 1
    return last_degree


# ---------------------------------------------------------------------------
# Thumb line: a 7-eighth-note cycle phasing against the 4/4 grid,
# S3 through S4.  Three notes per cycle on chord tones — a kalimba-ish
# figure whose downbeat rotates through the bar.
# ---------------------------------------------------------------------------


def _place_thumb(score: Score) -> None:
    cycle_hits: tuple[int, ...] = (0, 2, 5)  # eighth positions inside the 7-cycle
    start_bar, end_bar = 57, 105
    origin = _pos(start_bar)
    total_eighths = (end_bar - start_bar) * 8
    rng = random.Random(753)

    for eighth in range(total_eighths):
        if eighth % 7 not in cycle_hits:
            continue
        t = origin + eighth * S8
        bar = start_bar + eighth // 8
        region = _region_at_bar(bar)
        hit_rank = cycle_hits.index(eighth % 7)
        degree = region[hit_rank % len(region)]
        octave = 2 if hit_rank else 1
        in_s4 = bar >= S4_BAR
        if bar in (99, 100):
            continue  # held breath before the slam
        if not in_s4 and rng.random() < 0.25:
            continue  # S3: the cycle is still finding itself
        score.add_note(
            "thumb",
            start=t + (ARP_SWING_S if eighth % 2 else 0.0),
            duration=S8 * 1.6,
            partial=_partial(degree, octave),
            amp_db=-14.0 if in_s4 else -16.5,
            velocity=0.62 if hit_rank == 0 else 0.5,
        )


# ---------------------------------------------------------------------------
# Pad, bass, bell, haze
# ---------------------------------------------------------------------------

# The pad's spectrum is the hexany itself, spread across three octaves —
# every partial interval is a product of {1,3,5,7}.  Timbre-harmony fusion.
_HEXANY_SPECTRUM = ratio_spectrum(
    [1.0, 2.0, 5.0 / 2.0, 10.0 / 3.0, 7.0 / 2.0, 5.0, 35.0 / 6.0, 7.0],
    amps=[1.0, 0.55, 0.40, 0.30, 0.28, 0.18, 0.12, 0.10],
)


def _place_pad(score: Score) -> None:
    # S1: the dew — a bare 1/1 pedal, two octaves, nothing else.
    score.add_note(
        "pad",
        start=_pos(1),
        duration=_pos(S2_BAR) - _pos(1) + 2.0,
        partial=1.0,
        amp_db=-10.0,
        velocity=0.6,
    )
    score.add_note(
        "pad",
        start=_pos(5),
        duration=_pos(S2_BAR) - _pos(5) + 2.0,
        partial=2.0,
        amp_db=-15.0,
        velocity=0.5,
    )

    # From S2 on: the region as slow chords with actual voice-leading.
    # Three lanes; at each slot change every lane moves to the nearest
    # available region tone in pitch-index space, common tones simply
    # hold, and moving voices enter one beat late (a suspension into
    # its resolution).  The pad *does* harmony instead of stating it.
    slots = [(s, r) for s, r in _CHORD_SLOTS if s >= S2_BAR]
    glide = PitchMotionSpec.ratio_glide(start_ratio=0.996, end_ratio=1.0)
    prev_voicing: list[tuple[int, int]] | None = None  # (degree, octave)
    for i, (start_bar, region) in enumerate(slots):
        end_bar = slots[i + 1][0] if i + 1 < len(slots) else END_BAR
        start = _pos(start_bar)
        dur = _pos(end_bar) - start + 1.5
        ranked = sorted(region)
        if prev_voicing is None:
            voicing = [(d, 1) for d in ranked]
            if len(voicing) == 2:  # the closing dyad
                voicing.append((ranked[0], 2))
        else:
            # Greedy nearest-tone assignment, each region degree used once
            # (repeats allowed only for the 2-degree closing dyad).
            available = list(region) * (2 if len(region) == 2 else 1)
            voicing = []
            for deg_prev, oct_prev in prev_voicing:
                pi_prev = _pi(deg_prev, oct_prev)
                best = min(
                    (
                        (abs(_pi(d, o) - pi_prev), d, o)
                        for d in set(available)
                        for o in (1, 2)  # octave 0 belongs to the bass
                    ),
                )
                _, deg_new, oct_new = best
                available.remove(deg_new)
                voicing.append((deg_new, oct_new))
        for lane, (degree, octave) in enumerate(voicing):
            held = prev_voicing is not None and prev_voicing[lane] == (
                degree,
                octave,
            )
            # Suspension: moving voices arrive a beat after the slot.
            lane_start = start if held or prev_voicing is None else start + BEAT
            is_root = degree == ranked[0]
            score.add_note(
                "pad",
                start=lane_start,
                duration=dur - (lane_start - start),
                partial=_partial(degree, octave),
                amp_db=-12.0 if is_root else -15.0,
                velocity=0.6 if is_root else 0.5,
                pitch_motion=glide,
            )
        prev_voicing = voicing
        # S4/S5 add the root an octave up for glow
        if start_bar >= S4_BAR:
            score.add_note(
                "pad",
                start=start,
                duration=dur,
                partial=_partial(ranked[0], 2),
                amp_db=-18.0,
                velocity=0.45,
            )


def _place_bass(score: Score, lead_degrees: dict[int, int]) -> None:
    slots = [(s, r) for s, r in _CHORD_SLOTS if S2_BAR <= s < 121]
    rng = random.Random(35)
    for i, (start_bar, region) in enumerate(slots):
        end_bar = slots[i + 1][0] if i + 1 < len(slots) else 121
        root = min(region)
        in_s4 = S4_BAR <= start_bar < S5_BAR
        third = sorted(region)[1]
        for bar in range(start_bar, end_bar):
            if bar >= S5_BAR and bar >= 117:
                break
            # Root held over the bar, re-articulated; a pickup push on
            # the and-of-4 into each new slot; octave pops in S4.
            # Bass lives in octave 0 (93–160 Hz) — the 46 Hz kick owns
            # the sub octave, and sharing it doubled master-limiter IMD.
            bar_in_slot = bar - start_bar
            slot_len = end_bar - start_bar
            motif_bar = in_s4 and (
                bar_in_slot == 2 if slot_len >= 4 else bar_in_slot == 1
            )
            if motif_bar and bar not in (99, 100):
                # The seed motif's rhythm on root↔third — the ground
                # floor finally speaks the theme.
                t = _pos(bar)
                for k, (_, beats) in enumerate(_MOTIF[:4]):
                    pitch = root if k % 2 == 0 else third
                    score.add_note(
                        "bass",
                        start=t,
                        duration=beats * BEAT * 0.9,
                        partial=_partial(pitch, 0),
                        amp_db=-8.5,
                        velocity=0.68 if k % 2 == 0 else 0.6,
                    )
                    t += beats * BEAT
                continue
            score.add_note(
                "bass",
                start=_pos(bar),
                duration=BAR * 0.92,
                partial=_partial(root, 0),
                amp_db=-8.0,
                velocity=0.72,
            )
            if bar == end_bar - 1 and end_bar < 121:
                # Approach tone chosen by listening to the lead: prefer
                # a next-region tone factor-adjacent to the walker's
                # last note this bar — counterpoint, not just a pickup.
                next_region = _region_at_bar(end_bar)
                lead_last = lead_degrees.get(bar, root)
                linked = [d for d in next_region if _shares_factor(d, lead_last)]
                pickup = (
                    min(linked, key=lambda d: _DEGREE_RATIOS[d])
                    if linked
                    else min(next_region)
                )
                score.add_note(
                    "bass",
                    start=_pos(bar, 4.5),
                    duration=S8 * 0.9,
                    partial=_partial(pickup, 0),
                    amp_db=-9.5,
                    velocity=0.6,
                )
            if in_s4 and rng.random() < 0.35:
                score.add_note(
                    "bass",
                    start=_pos(bar, 3.5),
                    duration=S8 * 0.9,
                    partial=_partial(root, 1),
                    amp_db=-10.5,
                    velocity=0.58,
                )


def _place_bells(score: Score) -> None:
    # S1: the seed motif, augmented 2x, alone over the drone.
    _place_motif(
        score,
        "bell",
        _pos(11),
        octave=2,
        time_scale=2.0,
        amp_db=-9.0,
        vel=0.7,
        ring=2.2,
    )
    # Seam statements at S2 and mid-S2, tempo primo.
    _place_motif(score, "bell", _pos(17), octave=2, amp_db=-11.0, vel=0.72)
    _place_motif(score, "bell", _pos(33), octave=3, amp_db=-12.5, vel=0.65)
    # Bar 49 IS the turn: over bass alone, the motif returns *utonally
    # shadowed* — same contour, degrees mapped through the polar
    # complement (0↔3, 1↔2, 4↔5).  First real silence since the beat.
    polar = {0: 3, 1: 2, 2: 1, 3: 0, 4: 5, 5: 4}
    t = _pos(49)
    for degree, beats in _MOTIF:
        dur = beats * BEAT * 2.0
        score.add_note(
            "bell",
            start=t,
            duration=dur * 2.0,
            partial=_partial(polar[degree], 2),
            amp_db=-10.0,
            velocity=0.66,
        )
        t += dur
    # S4: canon — bell leads, thread follows a bar later at the fifth-less
    # hexany "answer" (up a 7/6).
    for lead_bar in (85, 101):
        _place_motif(score, "bell", _pos(lead_bar), octave=3, amp_db=-11.0, vel=0.7)
        _place_motif(
            score,
            "thread",
            _pos(lead_bar + 1),
            octave=2,
            amp_db=-12.0,
            vel=0.62,
            ring=1.9,
            vibrato=True,
        )
    # S5: the fragment — first two notes, then silence.
    for degree, beats, when in ((0, 2.0, 121), (2, 4.0, 123)):
        score.add_note(
            "bell",
            start=_pos(when),
            duration=beats * BEAT * 2.5,
            partial=_partial(degree, 2),
            amp_db=-13.0,
            velocity=0.55,
        )


def _place_glint(score: Score) -> None:
    """S4's own color: high, sparse answers to the walker's phrases.

    Two-to-three-note fragments in octave 4, landing mid-bar on slot
    heads and ricocheting through a 5/16 delay — a sound no other
    section owns.
    """
    rng = random.Random(93)
    for slot_start in (78, 82, 86, 90, 94, 98, 102, 106):
        region = _region_at_bar(slot_start)
        n_notes = rng.choice((2, 3))
        beat = rng.choice((2.5, 3.0, 3.5))
        for k in range(n_notes):
            degree = region[rng.randrange(len(region))]
            score.add_note(
                "glint",
                start=_pos(slot_start, beat) + k * DOTTED_EIGHTH,
                duration=S8,
                partial=_partial(degree, 4),
                amp_db=-16.0,
                velocity=0.45 + 0.12 * rng.random(),
            )


def _place_haze(score: Score) -> None:
    """Grain cloud pinned to the hexany lattice; S3 onward.

    Each stretch sits on the *current region's root* so the cloud
    harmonizes with the harmony instead of hovering nearby.
    """
    for start_bar, dur_bars, octave, amp in (
        (45, 8, 1, -22.0),  # fades in under the turn
        (53, 12, 2, -20.0),
        (65, 12, 2, -20.0),  # octave 2: the bow owns octave 1 here
        (77, 16, 2, -19.0),
        (93, 16, 1, -19.0),
        (109, 16, 1, -20.0),
        (125, 7, 1, -22.0),
    ):
        root = min(_region_at_bar(start_bar))
        score.add_note(
            "haze",
            start=_pos(start_bar),
            duration=dur_bars * BAR,
            partial=_partial(root, octave),
            amp_db=amp,
            velocity=0.5,
        )


def _place_bow(score: Score) -> None:
    """A bowed body under the shadow garden: S3 only, region roots."""
    for slot_start in (53, 57, 61, 65, 69, 73):
        root = min(_region_at_bar(slot_start))
        score.add_note(
            "bow",
            start=_pos(slot_start),
            duration=4 * BAR * 0.96,
            partial=_partial(root, 1),
            amp_db=-14.0,
            velocity=0.55,
        )


# ---------------------------------------------------------------------------
# Drums
# ---------------------------------------------------------------------------


def _place_kick(score: Score) -> None:
    def kick(bar: int, beat: float, amp: float, vel: float) -> None:
        score.add_note(
            "kick",
            start=_pos(bar, beat),
            duration=0.6,
            freq=F0 * 0.5,
            amp_db=amp,
            velocity=vel,
        )

    for bar in range(S2_BAR, 21):  # sparse entry: beat 1 only
        kick(bar, 1.0, -8.5, 0.8)
    for bar in range(21, 45):  # head nod
        kick(bar, 1.0, -6.5, 0.9)
        kick(bar, 3.0, -7.0, 0.85)
    # 45–52: kick out — the breath, then the turn's bare event
    for bar in range(53, S4_BAR):  # S3: heartbeat only
        kick(bar, 1.0, -8.0, 0.82)
        if (bar - 53) % 4 == 2:
            kick(bar, 3.75, -12.0, 0.5)  # ghost push
    for bar in range(S4_BAR, S5_BAR):  # S4: full nod + ghosts
        if bar in (99, 100):
            continue  # two bars of held breath before the slam at 101
        kick(bar, 1.0, -6.0, 0.92)
        kick(bar, 3.0, -6.5, 0.86)
        if (bar - S4_BAR) % 4 == 3:
            kick(bar, 4.75, -11.0, 0.55)
    for bar in (S5_BAR, S5_BAR + 1, S5_BAR + 2, S5_BAR + 3):  # fading heartbeat
        kick(bar, 1.0, -9.0 - (bar - S5_BAR), 0.7)


def _place_hats(score: Score) -> None:
    def hat(
        voice: str, bar: int, step: int, vel: float, amp: float, dur: float = 0.05
    ) -> None:
        score.add_note(
            voice,
            start=_pos(bar) + step * S16 + (HAT_SWING_S if step % 2 else 0.0),
            duration=dur,
            freq=4400.0,
            amp_db=amp,
            velocity=vel,
        )

    # Offbeat-8th bed, S2 (from 21) through S4.
    for bar in range(21, S5_BAR):
        if 45 <= bar < 53:
            continue  # the breath and the turn's bare event
        if bar in (99, 100):
            continue  # held breath before the slam
        in_s4 = bar >= S4_BAR
        base = -10.0 if in_s4 else -12.0
        for step in (2, 6, 10, 14):
            hat("hat_c", bar, step, 0.68 if step in (6, 14) else 0.58, base)
        if in_s4 and bar % 2 == 0:
            for step in (7, 15):  # 16th graces
                hat("hat_c", bar, step, 0.38, base - 4.0)

    # The 7-cycle accent: every 7th sixteenth from bar 57 (with the
    # thumb line), rotating against the grid — hats phase with the thumb.
    origin_bar = 57
    total_16ths = (S5_BAR - origin_bar) * 16
    for k in range(0, total_16ths, 7):
        bar = origin_bar + k // 16
        step = k % 16
        if bar in (99, 100):
            continue
        in_s4 = bar >= S4_BAR
        hat("hat_c", bar, step, 0.78, -9.0 if in_s4 else -11.5)

    # Open hat at 4-bar phrase ends, choked by the next closed hat.
    for bar in range(24, S5_BAR, 4):
        if 45 <= bar < 56 or bar == 100:
            continue
        hat("hat_o", bar, 14, 0.66, -12.5, dur=0.5)


def _place_shaker(score: Score) -> None:
    """Soft brush backbeat in the bloom sections, plus the cluster riser."""
    for bar in range(25, S5_BAR):
        if 45 <= bar < 61 or bar in (99, 100):
            continue
        for step, vel in ((4, 0.55), (12, 0.66)):
            score.add_note(
                "shaker",
                start=_pos(bar) + step * S16,
                duration=0.15,
                freq=900.0,
                amp_db=-13.0,
                velocity=vel,
            )
    # 16th-note roll rising through bar 92 into the 6:7:8 cluster bloom.
    for k in range(8, 16):
        rise = (k - 8) / 7.0
        score.add_note(
            "shaker",
            start=_pos(92) + k * S16,
            duration=0.12,
            freq=900.0,
            amp_db=-15.0 + 4.0 * rise,
            velocity=0.35 + 0.45 * rise,
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
                room_size=0.85,
                damping=0.5,
                lowpass_hz=5000.0,
                highpass_hz=200.0,
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


def _arp_cutoff_arc() -> AutomationSpec:
    return _ramp(
        AutomationTarget(kind="synth", name="cutoff_hz"),
        [
            (S1_END, S1_END + 8 * BAR, 2100.0, "exp"),  # bloom opens
            (S2_END - 4 * BAR, S2_END + 4 * BAR, 1250.0, "exp"),  # turn darkens
            (_pos(65), S3_END, 1900.0, "exp"),  # stride recovers
            (S3_END, _pos(93), 2800.0, "exp"),  # garden opens
            (_pos(101), S4_END, 3300.0, "exp"),  # fast-spinning peak
            (S4_END, S4_END + 8 * BAR, 1300.0, "exp"),  # seed closes down
        ],
        default=1600.0,
    )


def _arp_pan_sway() -> AutomationSpec:
    return _ramp(
        AutomationTarget(kind="control", name="pan"),
        [
            (0.0, S2_END, 0.12, "linear"),
            (S2_END, S3_END, -0.22, "linear"),
            (S3_END, S4_END, 0.08, "linear"),
            (S4_END, TOTAL_DUR, -0.10, "linear"),
        ],
        default=-0.10,
    )


def _arp_hall_ride() -> AutomationSpec:
    """Hall opens into the turn and again as the seed scatters."""
    return _ramp(
        AutomationTarget(kind="control", name="send_db"),
        [
            (S2_END - 2 * BAR, S2_END + 2 * BAR, -5.0, "linear"),
            (S3_END - 2 * BAR, S3_END + 2 * BAR, -8.5, "linear"),
            (S4_END - 2 * BAR, S4_END + 4 * BAR, -4.0, "linear"),
        ],
        default=-8.5,
    )


def _arp_delay_throw() -> AutomationSpec:
    """Delay blooms at the turn, again at the S4 seam, and as S5 opens."""
    return _ramp(
        AutomationTarget(kind="control", name="mix"),
        [
            (S2_END - 2 * BAR, S2_END + 2 * BAR, 0.40, "linear"),
            (S2_END + 6 * BAR, _pos(73), 0.20, "linear"),
            (_pos(75), S3_END, 0.44, "linear"),  # runaway into the garden
            (S3_END + 2 * BAR, S3_END + 4 * BAR, 0.22, "linear"),
            (S4_END - 4 * BAR, S4_END, 0.46, "linear"),
            (S4_END + 4 * BAR, S4_END + 8 * BAR, 0.24, "linear"),
        ],
        default=0.20,
    )


def _pad_brightness_arc() -> AutomationSpec:
    return _ramp(
        AutomationTarget(kind="synth", name="brightness_tilt"),
        [
            (0.0, S1_END, -0.06, "linear"),
            (S1_END, S2_END, 0.04, "linear"),
            (S2_END, S3_END, -0.12, "linear"),  # after dark
            (S3_END, S4_END, 0.08, "linear"),  # full glow
            (S4_END, TOTAL_DUR, -0.14, "linear"),
        ],
        default=-0.12,
    )


def _haze_mix_ride() -> AutomationSpec:
    return _ramp(
        AutomationTarget(kind="control", name="mix_db"),
        [
            (S2_END - 4 * BAR, S2_END + 4 * BAR, -15.0, "linear"),
            (S3_END - 4 * BAR, S3_END, -12.5, "linear"),  # riser into S4
            (S3_END, S3_END + 2 * BAR, -18.0, "linear"),
            (_pos(91), _pos(93), -11.5, "linear"),  # swell into the cluster
            (_pos(93), _pos(95), -18.5, "linear"),  # cut on the bloom
            (S4_END, TOTAL_DUR - 4 * BAR, -13.0, "linear"),  # haze inherits
        ],
        default=-18.0,
    )


def _pad_hall_ride() -> AutomationSpec:
    """The shadow garden is wetter; the full garden pulls closer."""
    return _ramp(
        AutomationTarget(kind="control", name="send_db"),
        [
            (S2_END, S2_END + 4 * BAR, -4.5, "linear"),
            (S3_END - 2 * BAR, S3_END + 2 * BAR, -6.5, "linear"),
            (S4_END, S4_END + 4 * BAR, -4.5, "linear"),
        ],
        default=-6.0,
    )


# ---------------------------------------------------------------------------
# build_score
# ---------------------------------------------------------------------------


def build_score() -> Score:
    """Build the Hexany Garden score."""
    score = Score(
        f0_hz=F0,
        master_effects=DEFAULT_MASTER_EFFECTS,
        # Pad + haze sustain continuously from S2 on; without a trim the
        # master vari-mu never relaxes below 1 dB of gain reduction.
        master_input_gain_db=-2.5,
        send_buses=[_make_hall_bus()],
        timing_humanize=TimingHumanizeSpec(preset="chamber"),
    )
    score.add_drift_bus("garden_drift", rate_hz=0.09, depth_cents=4.5, seed=1357)
    drum_bus = setup_drum_bus(score, style="electronic", return_db=0.0)

    # ---- Walker arp: the spine ----
    score.add_voice(
        "arp",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "soft_pluck",
            "release": 0.7,
            "filter_mode": "lowpass",
            "filter_topology": "svf",
            "filter_cutoff_hz": 1600.0,
            "resonance_q": 0.8,
        },
        effects=[
            EffectSpec(
                "delay",
                {"delay_seconds": DOTTED_EIGHTH, "feedback": 0.32, "mix": 0.20},
                automation=[_arp_delay_throw()],
            ),
        ],
        sends=[VoiceSend(target="hall", send_db=-8.5, automation=[_arp_hall_ride()])],
        pan=-0.10,
        mix_db=-2.0,
        envelope_humanize=EnvelopeHumanizeSpec(preset="loose_pluck"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        drift_bus="garden_drift",
        drift_bus_correlation=0.6,
        automation=[_arp_cutoff_arc(), _arp_pan_sway()],
    )

    # ---- Thumb line: the 7-cycle, other side of the stereo field ----
    score.add_voice(
        "thumb",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "harp_pluck",
            "release": 1.1,
            "filter_mode": "lowpass",
            "filter_cutoff_hz": 1250.0,
        },
        sends=[VoiceSend(target="hall", send_db=-7.0)],
        pan=0.30,
        mix_db=-6.0,
        envelope_humanize=EnvelopeHumanizeSpec(preset="loose_pluck"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        drift_bus="garden_drift",
        drift_bus_correlation=0.6,
    )

    # ---- Pad: hexany-spectrum drone ----
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "additive_pad_through_ladder",
            "partials_partials": _HEXANY_SPECTRUM,
            "filter_cutoff_hz": 1100.0,
            "attack": 3.5,
            "release": 5.0,
            "brightness_tilt": -0.12,
        },
        effects=[
            # Subtle kick-keyed breathing: only the kick's low thump
            # triggers (detector lowpass), and mix keeps it gentle.
            EffectSpec(
                "compressor",
                {
                    "threshold_db": -30.0,
                    "ratio": 2.0,
                    "attack_ms": 4.0,
                    "release_ms": 220.0,
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
        mix_db=-10.0,
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        drift_bus="garden_drift",
        drift_bus_correlation=0.8,
        automation=[_pad_brightness_arc()],
    )

    # ---- Bass: mono legato roots ----
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "soft_bass",
            "release": 0.5,
        },
        effects=[
            # Kick-keyed ducking: gives the kick its pocket and keeps
            # bass + kick from stacking into the master limiter at once.
            EffectSpec(
                "compressor",
                {
                    "threshold_db": -24.0,
                    "ratio": 3.0,
                    "attack_ms": 1.5,
                    "release_ms": 180.0,
                    "lookahead_ms": 5.0,
                    "sidechain_source": "kick",
                    "detector_mode": "peak",
                },
            ),
        ],
        pan=0.0,
        mix_db=-6.5,
        max_polyphony=1,
        legato=True,
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        drift_bus="garden_drift",
        drift_bus_correlation=0.85,
    )

    # ---- Bell: the seed motif's voice ----
    score.add_voice(
        "bell",
        synth_defaults={
            "engine": "drum_voice",
            "preset": "septimal_bell",
        },
        normalize_peak_db=-6.0,
        sends=[VoiceSend(target="hall", send_db=-2.0)],
        pan=0.24,
        mix_db=-15.0,
        velocity_humanize=None,
    )

    # ---- Thread: the canon answer ----
    score.add_voice(
        "thread",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "fm_bell_over_supersaw",
            "release": 2.0,
        },
        sends=[VoiceSend(target="hall", send_db=-4.0)],
        pan=-0.28,
        mix_db=-12.0,
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        drift_bus="garden_drift",
        drift_bus_correlation=0.5,
    )

    # ---- Bow: a live body under the shadow garden (S3 only) ----
    score.add_voice(
        "bow",
        synth_defaults={
            "engine": "drum_voice",
            "preset": "bow_gentle",
        },
        normalize_peak_db=-6.0,
        sends=[VoiceSend(target="hall", send_db=-4.0)],
        pan=-0.15,
        mix_db=-16.0,
        velocity_humanize=None,
        drift_bus="garden_drift",
        drift_bus_correlation=0.7,
    )

    # ---- Glint: S4's own color — high delayed answers ----
    score.add_voice(
        "glint",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "soft_pluck",
            "release": 1.4,
            "filter_mode": "lowpass",
            "filter_cutoff_hz": 3800.0,
        },
        effects=[
            EffectSpec(
                "delay",
                {"delay_seconds": 5.0 * S16, "feedback": 0.45, "mix": 0.38},
            ),
        ],
        sends=[VoiceSend(target="hall", send_db=-3.0)],
        pan=0.42,
        mix_db=-9.0,
        envelope_humanize=EnvelopeHumanizeSpec(preset="loose_pluck"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        drift_bus="garden_drift",
        drift_bus_correlation=0.4,
    )

    # ---- Haze: hexany-quantized grain dust ----
    score.add_voice(
        "haze",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "grain_breathing_cloud",
            "grain_ji_lattice": list(_DEGREE_RATIOS),
        },
        sends=[VoiceSend(target="hall", send_db=-3.0)],
        pan=0.0,
        mix_db=-18.0,
        automation=[_haze_mix_ride()],
    )

    # ---- Drums ----
    add_drum_voice(
        score,
        "kick",
        engine="drum_voice",
        preset="808_house",
        drum_bus=drum_bus,
        send_db=-3.0,
        synth_overrides={"tone_punch": 0.15, "exciter_level": 0.04},
        mix_db=-5.0,
    )
    # Slightly darkened hats: the noise-forward voicing is bright by
    # design, and the bus clipper adds its own high-band lift when driven.
    hat_darkening = {
        "metallic_brightness": 0.45,
        "metallic_hat_noise_bp_hz": 6200.0,
    }
    add_drum_voice(
        score,
        "hat_c",
        engine="drum_voice",
        preset="closed_hat",
        drum_bus=drum_bus,
        send_db=-7.0,
        synth_overrides=hat_darkening,
        choke_group="hats",
        mix_db=-17.0,
        pan=0.14,
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
        mix_db=-18.0,
        pan=0.14,
    )
    add_drum_voice(
        score,
        "shaker",
        engine="drum_voice",
        preset="brush",
        drum_bus=drum_bus,
        send_db=-5.0,
        mix_db=-16.0,
        pan=-0.26,
    )

    # ==================================================================
    # Notes
    # ==================================================================
    _place_pad(score)
    lead_degrees = _place_walker(score)
    _place_thumb(score)
    _place_bass(score, lead_degrees)
    _place_bells(score)
    _place_glint(score)
    _place_haze(score)
    _place_bow(score)
    _place_kick(score)
    _place_hats(score)
    _place_shaker(score)

    return score


PIECES: dict[str, PieceDefinition] = {
    "hexany_garden": PieceDefinition(
        name="hexany_garden",
        output_name="hexany_garden",
        build_score=build_score,
        sections=(
            PieceSection(label="S1 dew", start_seconds=0.0, end_seconds=S1_END),
            PieceSection(
                label="S2 first bloom", start_seconds=S1_END, end_seconds=S2_END
            ),
            PieceSection(label="S3 the turn", start_seconds=S2_END, end_seconds=S3_END),
            PieceSection(
                label="S4 full garden", start_seconds=S3_END, end_seconds=S4_END
            ),
            PieceSection(label="S5 seed", start_seconds=S4_END, end_seconds=TOTAL_DUR),
        ),
    ),
}
