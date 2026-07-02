"""Weaving Room — Four Tet-style wandering arps in Colundi JI.

Where ``colundi_arps_study`` is a *song* (fixed 2-bar hook, chorus form),
this piece is a *weave*: a 2-bar arp cell that never repeats exactly.
Every 2-bar window applies seeded, curated mutations to the current cell
— degree substitution, octave lifts, note additions/drops, duration and
accent nudges — so the arp reads as one continuous evolving thought.
A divergence tether keeps the cell recognizably itself: when it drifts
more than a few slots from the base cell, the oldest drift reverts.
Repetition-with-drift is the Four Tet trick: loops that breathe, not
melodies that develop.

Tuning: Colundi 7-note JI (1/1, 11/10, 19/16, 4/3, 3/2, 49/30, 7/4) on
f0 = G1 = 49 Hz — a kick-friendly key, and deliberately distinct from
the two existing Colundi pieces (both A1 = 55 Hz).  Stable tones (R, 4/3,
3/2, 7/4) are the home vocabulary; 11/10, 19/16, 49/30 are spice whose
mutation weight is itself automated across the form.

BPM = 104.  1 bar ≈ 2.308 s.  96 bars ≈ 3:42.

Form (density arc, not verse/chorus):
  bars  1– 8   S1 Loom          pad + tape-dust grain; arp fades in at 5
  bars  9–32   S2 First weave   beat enters (sparse → full at 13), bass
                                anchors R–P4–P5, low mutation rate
  bars 33–48   S3 Opening       mutation + spice rise, hats fill, filter
                                opens, canon arp (P5 up, one window behind)
  bars 49–60   S4 Breath        drums thin then drop; arp augments to 2x
                                note lengths; pad leans subdominant + 49/30
                                — the strangest moment, framed
  bars 61–84   S5 Second weave  full beat, both arps, brightest register
  bars 85–96   S6 Unravel       elements peel off; arp thins to a whisper

The mutation engine is deterministic (seeded ``random.Random``) so every
render is identical; mutation intensity, spice probability, add-note
bias, and octave-lift probability are per-section compositional
parameters (see ``_params_for_bar``).
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
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import EffectSpec, Score, SendBusSpec, VoiceSend
from code_musics.spectra import ratio_spectrum
from code_musics.synth import BRICASTI_IR_DIR

# ---------------------------------------------------------------------------
# Timing grid
# ---------------------------------------------------------------------------

BPM: float = 104.0
BEAT: float = 60.0 / BPM
BAR: float = 4.0 * BEAT
S16: float = BEAT / 4.0
DOTTED_EIGHTH: float = 3.0 * S16

F0: float = 49.0  # G1
TOTAL_BARS: int = 96

# Deterministic swing: odd 16th steps land late.  Arp swings lightly,
# hats swing harder (~56% MPC feel).
ARP_SWING_S: float = 0.08 * S16
HAT_SWING_S: float = 0.14 * S16


def _pos(bar: int, beat: int = 1, n16: int = 0) -> float:
    """Absolute time at bar:beat:sixteenth (1-indexed bar/beat)."""
    return (bar - 1) * BAR + (beat - 1) * BEAT + n16 * S16


# Section boundaries (bar numbers; S1 starts at bar 1)
S2_BAR: int = 9
S3_BAR: int = 33
S4_BAR: int = 49
S5_BAR: int = 61
S6_BAR: int = 85
END_BAR: int = TOTAL_BARS + 1

S1_END: float = _pos(S2_BAR)
S2_END: float = _pos(S3_BAR)
S3_END: float = _pos(S4_BAR)
S4_END: float = _pos(S5_BAR)
S5_END: float = _pos(S6_BAR)
TOTAL_DUR: float = _pos(END_BAR)

# ---------------------------------------------------------------------------
# Colundi scale.  Degree indices order the scale ascending within an
# octave; a "pitch index" (octave * 7 + degree) gives a total ordering
# used for contour-aware substitution.
# ---------------------------------------------------------------------------

_DEGREE_RATIOS: tuple[float, ...] = (
    1.0,  # R
    11 / 10,  # N2   narrow second (~165c)
    19 / 16,  # m3   (~298c)
    4 / 3,  # P4
    3 / 2,  # P5
    49 / 30,  # s6   (~849c)
    7 / 4,  # h7   harmonic seventh
)
_N_DEGREES: int = len(_DEGREE_RATIOS)
_HOME_DEGREES: frozenset[int] = frozenset({0, 3, 4, 6})
_SPICE_DEGREES: frozenset[int] = frozenset({1, 2, 5})

# Arp pitch range: P4 in octave 2 (partial ~5.3, ≈261 Hz) up to R in
# octave 4 (partial 16, ≈784 Hz).
_PI_MIN: int = 2 * _N_DEGREES + 3
_PI_MAX: int = 4 * _N_DEGREES + 0


def _partial(degree: int, octave: int) -> float:
    """Partial of f0 for a scale degree in a partial-space octave.

    Octave 0 spans partials 1–2 (49–98 Hz); octave 3 spans 8–16
    (392–784 Hz), the arp's home register.
    """
    return _DEGREE_RATIOS[degree] * (2.0**octave)


def _pitch_index(degree: int, octave: int) -> int:
    return octave * _N_DEGREES + degree


def _from_pitch_index(pi: int) -> tuple[int, int]:
    return pi % _N_DEGREES, pi // _N_DEGREES


# ---------------------------------------------------------------------------
# The base cell — 2 bars on a 32-step 16th grid.  Bar A rises R → P5 →
# h7 with a low-octave undertow dip; bar B answers from the held h7 and
# falls home through P4, dipping under to h7 an octave down before the
# resolution.  Steps 0 and 16 are anchors: mutations never re-pitch or
# drop them, so the loop keeps its identity no matter how far it drifts.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellNote:
    step: int  # 0..31 within the 2-bar cell
    degree: int
    octave: int
    velocity: float
    dur_16ths: float


_ANCHOR_STEPS: frozenset[int] = frozenset({0, 16})

_BASE_CELL: tuple[CellNote, ...] = (
    # Bar A — rising question
    CellNote(0, 0, 3, 0.86, 2.8),  # R    392 Hz anchor
    CellNote(3, 4, 2, 0.58, 1.9),  # P5   294 — syncopated undertow dip
    CellNote(6, 3, 3, 0.74, 1.9),  # P4   523
    CellNote(8, 4, 3, 0.70, 2.8),  # P5   588
    CellNote(11, 6, 3, 0.78, 1.9),  # h7   686 peak
    CellNote(14, 4, 3, 0.56, 1.9),  # P5   pickup
    # Bar B — falling answer
    CellNote(16, 6, 3, 0.80, 3.8),  # h7 held anchor
    CellNote(20, 4, 3, 0.62, 1.9),  # P5
    CellNote(22, 3, 3, 0.70, 2.8),  # P4
    CellNote(25, 0, 3, 0.60, 1.9),  # R
    CellNote(27, 6, 2, 0.55, 1.9),  # h7 octave-down under-dip
    CellNote(29, 0, 3, 0.74, 2.8),  # R resolve before the loop restarts
)


# ---------------------------------------------------------------------------
# Weave parameters — the compositional knobs of the mutation engine.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeaveParams:
    intensity: float  # 0..1 → mutations per 2-bar window (1 + 4*intensity)
    spice: float  # probability a substitution lands on a spice degree
    add_bias: float  # 0..1 extra weight on add-note (drives density)
    lift: float  # probability octave shifts go up rather than down
    amp_base: float  # arp level for the window (dB)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _params_for_bar(bar: int) -> WeaveParams:
    """Per-window weave parameters, following the piece's density arc."""
    if bar < S2_BAR:  # S1 Loom — arp barely moving, fading in
        t = (bar - 5) / max(S2_BAR - 5, 1)
        return WeaveParams(0.12, 0.05, 0.10, 0.05, _lerp(-16.0, -13.0, t))
    if bar < S3_BAR:  # S2 First weave — settled, slowly loosening
        t = (bar - S2_BAR) / (S3_BAR - S2_BAR)
        return WeaveParams(
            _lerp(0.20, 0.30, t),
            _lerp(0.08, 0.14, t),
            _lerp(0.25, 0.35, t),
            0.12,
            -12.0,
        )
    if bar < S4_BAR:  # S3 Opening — mutation and spice bloom
        t = (bar - S3_BAR) / (S4_BAR - S3_BAR)
        return WeaveParams(
            _lerp(0.35, 0.55, t),
            _lerp(0.18, 0.35, t),
            _lerp(0.45, 0.60, t),
            0.25,
            -10.0,
        )
    if bar < S5_BAR:  # S4 Breath — sparse but strange
        return WeaveParams(0.30, 0.50, 0.10, 0.10, -11.5)
    if bar < S6_BAR:  # S5 Second weave — brightest, densest
        t = (bar - S5_BAR) / (S6_BAR - S5_BAR)
        return WeaveParams(0.52, _lerp(0.28, 0.20, t), 0.60, 0.35, -9.0)
    # S6 Unravel — mutation and level wind down together
    t = (bar - S6_BAR) / (END_BAR - S6_BAR)
    return WeaveParams(_lerp(0.35, 0.10, t), 0.08, 0.05, 0.05, _lerp(-11.0, -16.0, t))


# ---------------------------------------------------------------------------
# Mutation engine
# ---------------------------------------------------------------------------


def _clamp_pitch(pi: int) -> int:
    return max(_PI_MIN, min(_PI_MAX, pi))


def _nearest_in_set(degree: int, allowed: frozenset[int]) -> int:
    """Nearest degree (in scale order) drawn from ``allowed``."""
    return min(allowed, key=lambda d: (abs(d - degree), d))


def _substitute(cell: list[CellNote], rng: random.Random, p: WeaveParams) -> None:
    """Re-pitch one non-anchor note by a step or two of scale contour."""
    candidates = [i for i, n in enumerate(cell) if n.step not in _ANCHOR_STEPS]
    if not candidates:
        return
    i = rng.choice(candidates)
    note = cell[i]
    move = rng.choice((-2, -1, 1, 2))
    degree, octave = _from_pitch_index(
        _clamp_pitch(_pitch_index(note.degree, note.octave) + move)
    )
    # Spice gate: land on a spice degree only when the dice allow it.
    if degree in _SPICE_DEGREES and rng.random() > p.spice:
        degree = _nearest_in_set(degree, _HOME_DEGREES)
    cell[i] = replace(note, degree=degree, octave=octave)


def _spice_swap(cell: list[CellNote], rng: random.Random, _p: WeaveParams) -> None:
    """Deliberately recolor one home tone to its nearest spice neighbor.

    The contour moves in ``_substitute`` rarely land on spice degrees, so
    this mutation is the piece's main source of Colundi color; its weight
    scales directly with the window's spice parameter.
    """
    candidates = [
        i
        for i, n in enumerate(cell)
        if n.step not in _ANCHOR_STEPS and n.degree in _HOME_DEGREES
    ]
    if not candidates:
        return
    i = rng.choice(candidates)
    note = cell[i]
    spice_degree = _nearest_in_set(note.degree, _SPICE_DEGREES)
    cell[i] = replace(note, degree=spice_degree)


def _octave_shift(cell: list[CellNote], rng: random.Random, p: WeaveParams) -> None:
    """Throw one non-anchor note up (or down) an octave — the sparkle move."""
    candidates = [i for i, n in enumerate(cell) if n.step not in _ANCHOR_STEPS]
    if not candidates:
        return
    i = rng.choice(candidates)
    note = cell[i]
    direction = 1 if rng.random() < p.lift else -1
    pi = _pitch_index(note.degree, note.octave) + direction * _N_DEGREES
    if not (_PI_MIN <= pi <= _PI_MAX):
        return
    degree, octave = _from_pitch_index(pi)
    cell[i] = replace(note, degree=degree, octave=octave)


def _add_note(cell: list[CellNote], rng: random.Random, p: WeaveParams) -> None:
    """Fill an empty weak 16th with a quiet echo of its neighbor's pitch."""
    occupied = {n.step for n in cell}
    empty = [s for s in range(32) if s not in occupied]
    if not empty:
        return
    weak = [s for s in empty if s % 2 == 1] or empty
    step = rng.choice(weak)
    neighbor = min(cell, key=lambda n: abs(n.step - step))
    move = rng.choice((-1, 1))
    degree, octave = _from_pitch_index(
        _clamp_pitch(_pitch_index(neighbor.degree, neighbor.octave) + move)
    )
    if degree in _SPICE_DEGREES and rng.random() > p.spice:
        degree = _nearest_in_set(degree, _HOME_DEGREES)
    cell.append(CellNote(step, degree, octave, rng.uniform(0.42, 0.55), 1.0))
    cell.sort(key=lambda n: n.step)


def _drop_note(cell: list[CellNote], rng: random.Random, _p: WeaveParams) -> None:
    """Silence one quiet, non-anchor note — the loop inhales."""
    candidates = [
        i
        for i, n in enumerate(cell)
        if n.step not in _ANCHOR_STEPS and n.velocity < 0.72
    ]
    if not candidates or len(cell) <= 6:
        return
    del cell[rng.choice(candidates)]


def _nudge_duration(cell: list[CellNote], rng: random.Random, _p: WeaveParams) -> None:
    i = rng.randrange(len(cell))
    note = cell[i]
    factor = rng.choice((0.7, 1.5))
    cell[i] = replace(note, dur_16ths=max(0.8, min(6.0, note.dur_16ths * factor)))


def _reaccent(cell: list[CellNote], rng: random.Random, _p: WeaveParams) -> None:
    i = rng.randrange(len(cell))
    note = cell[i]
    vel = max(0.40, min(0.95, note.velocity + rng.uniform(-0.12, 0.12)))
    cell[i] = replace(note, velocity=vel)


def _divergent_steps(cell: list[CellNote]) -> list[int]:
    """Steps where the cell differs in pitch from the base (or is extra)."""
    base_by_step = {n.step: (n.degree, n.octave) for n in _BASE_CELL}
    return [n.step for n in cell if base_by_step.get(n.step) != (n.degree, n.octave)]


def _revert_toward_base(cell: list[CellNote], rng: random.Random) -> None:
    """Pull one drifted slot back to the base cell — the tether."""
    divergent = _divergent_steps(cell)
    missing = {n.step for n in _BASE_CELL} - {n.step for n in cell}
    pool: list[tuple[str, int]] = [("drifted", s) for s in divergent]
    pool.extend(("missing", s) for s in missing)
    if not pool:
        return
    kind, step = rng.choice(pool)
    base_by_step = {n.step: n for n in _BASE_CELL}
    if kind == "missing":
        cell.append(base_by_step[step])
        cell.sort(key=lambda n: n.step)
        return
    i = next(idx for idx, n in enumerate(cell) if n.step == step)
    base_note = base_by_step.get(step)
    if base_note is None:
        del cell[i]  # an added extra — remove it
    else:
        cell[i] = replace(
            base_note, velocity=cell[i].velocity, dur_16ths=cell[i].dur_16ths
        )


def _weave_window(
    cell: list[CellNote], rng: random.Random, p: WeaveParams
) -> list[CellNote]:
    """Evolve the cell by one 2-bar window: tether first, then mutate."""
    new_cell = list(cell)
    max_divergence = 3 + round(p.intensity * 4)
    while len(_divergent_steps(new_cell)) > max_divergence:
        _revert_toward_base(new_cell, rng)

    mutations = [
        (_substitute, 2.0),
        (_spice_swap, 2.0 * p.spice),
        (_add_note, 0.5 + 1.5 * p.add_bias),
        (_drop_note, max(0.25, 0.8 - 0.6 * p.add_bias)),
        (_octave_shift, 0.7),
        (_nudge_duration, 0.5),
        (_reaccent, 0.5),
    ]
    fns = [m[0] for m in mutations]
    weights = [m[1] for m in mutations]
    n_mutations = 1 + round(p.intensity * 4)
    for fn in rng.choices(fns, weights=weights, k=n_mutations):
        fn(new_cell, rng, p)
    return new_cell


# ---------------------------------------------------------------------------
# Arp placement
# ---------------------------------------------------------------------------


def _vibrato_for_dur(dur_16ths: float) -> PitchMotionSpec | None:
    if dur_16ths >= 2.5:
        return PitchMotionSpec.vibrato(depth_ratio=0.0035, rate_hz=5.0)
    return None


def _place_cell(
    score: Score,
    voice: str,
    bar: int,
    cell: list[CellNote],
    amp_base: float,
    *,
    transpose: float = 1.0,
    time_scale: float = 1.0,
    min_velocity: float = 0.0,
) -> None:
    """Place one cell instance starting at ``bar``.

    ``time_scale=2.0`` augments the cell across 4 bars (the Breath).
    ``min_velocity`` thins the cell (the Unravel).
    """
    for note in cell:
        if note.velocity < min_velocity:
            continue
        swing = ARP_SWING_S if note.step % 2 == 1 else 0.0
        dur_16 = note.dur_16ths * time_scale
        score.add_note(
            voice,
            start=_pos(bar) + note.step * S16 * time_scale + swing,
            duration=dur_16 * S16,
            partial=_partial(note.degree, note.octave) * transpose,
            amp_db=amp_base,
            velocity=note.velocity,
            pitch_motion=_vibrato_for_dur(dur_16),
        )


def _weave_history() -> dict[int, list[CellNote]]:
    """Evolve the cell across every 2-bar window of the piece.

    Returns {window_start_bar: cell}.  Deterministic: single seeded RNG
    consumed in bar order.
    """
    rng = random.Random(20260702)
    history: dict[int, list[CellNote]] = {}
    cell = list(_BASE_CELL)
    for bar in range(5, TOTAL_BARS + 1, 2):
        cell = _weave_window(cell, rng, _params_for_bar(bar))
        history[bar] = cell
    return history


def _place_arps(score: Score, history: dict[int, list[CellNote]]) -> None:
    """Place arp 1 (the spine) and arp 2 (the canon, one window behind)."""
    for bar, cell in history.items():
        p = _params_for_bar(bar)
        if S4_BAR <= bar < S5_BAR:
            # Breath: augmented placement every other window; each spans
            # 4 bars.  Only the strong notes survive — long, singing.
            if (bar - S4_BAR) % 4 == 0:
                strong = [n for n in cell if n.velocity >= 0.68 or n.dur_16ths >= 2.8]
                _place_cell(score, "arp", bar, strong, p.amp_base, time_scale=2.0)
            continue
        min_vel = 0.0
        if bar >= S6_BAR + 4:  # unravel thinning
            min_vel = 0.60 if bar < S6_BAR + 8 else 0.72
        _place_cell(score, "arp", bar, cell, p.amp_base, min_velocity=min_vel)

    # Canon voice: previous window's cell a P5 up — an echo of memory.
    # Half-lit in S3, brighter in S5.
    for bar in range(S3_BAR, S4_BAR, 2):
        prev = history[bar - 2]
        _place_cell(
            score, "arp2", bar, prev, _params_for_bar(bar).amp_base - 3.5, transpose=1.5
        )
    for bar in range(S5_BAR, S6_BAR, 2):
        prev = history[bar - 2]
        _place_cell(
            score, "arp2", bar, prev, _params_for_bar(bar).amp_base - 1.5, transpose=1.5
        )


# ---------------------------------------------------------------------------
# Bass — mono legato undertow, R · R · P4 · P5 in octave 1 (98–172 Hz)
# ---------------------------------------------------------------------------

_BASS_R: float = 2.0
_BASS_P4: float = 8 / 3
_BASS_P5: float = 3.0
_BASS_H7: float = 7 / 2


def _place_bass(score: Score) -> None:
    """4-bar bass loop with an h7 turnaround pickup into each repeat.

    Active bars 9–92; drops out with the drums in the Breath (53–58).
    """
    for loop_start in range(S2_BAR, TOTAL_BARS - 3, 4):
        if 53 <= loop_start <= 58:
            continue
        quiet = -2.5 if loop_start >= S6_BAR else 0.0
        a, b, c, d = loop_start, loop_start + 1, loop_start + 2, loop_start + 3
        if 53 <= d <= 58:
            continue
        notes: list[tuple[int, int, float, float, float]] = [
            (a, 1, _BASS_R, 4.0, 0.80),
            (b, 1, _BASS_R, 3.0, 0.72),
            (b, 4, _BASS_P5 / 2, 1.0, 0.58),  # P5 an octave under (73.5 Hz dip)
            (c, 1, _BASS_P4, 4.0, 0.76),
            (d, 1, _BASS_P5, 3.0, 0.74),
            (d, 4, _BASS_H7, 1.0, 0.62),  # h7 turnaround pulls back to R
        ]
        for bar, beat, partial, dur_beats, vel in notes:
            score.add_note(
                "bass",
                start=_pos(bar, beat),
                duration=dur_beats * BEAT,
                partial=partial,
                amp_db=-8.0 + quiet,
                velocity=vel,
            )


# ---------------------------------------------------------------------------
# Pad — Colundi-spectrum additive drone with section chord moves
# ---------------------------------------------------------------------------

_COLUNDI_PARTIALS = ratio_spectrum(
    ratios=[1.0, 11 / 10, 19 / 16, 4 / 3, 3 / 2, 49 / 30, 7 / 4, 2.0, 3.0, 7 / 2],
    amps=[1.0, 0.22, 0.16, 0.45, 0.55, 0.12, 0.35, 0.30, 0.15, 0.10],
)


# Pad chord vocabulary: (partial, amp_db) voicings over the constant R2
# anchor.  A = home 4:6:7, B = subdominant lift, C = narrow-2nd color.
_PAD_CHORDS: dict[str, tuple[tuple[float, float], ...]] = {
    "A": ((3.0, -11.0), (3.5, -11.0), (6.0, -15.0)),
    "B": ((8 / 3, -10.5), (3.5, -12.0), (16 / 3, -15.5)),
    "C": ((3.0, -11.0), (22 / 5, -13.0), (3.5, -12.0), (7.0, -15.5)),
}

# 8-bar harmonic rhythm: (start_bar, n_bars, chord).  S4 (the Breath)
# keeps its own bespoke subdominant + s6 voicing below.
_PAD_PROGRESSION: tuple[tuple[int, int, str], ...] = (
    (1, 8, "A"),  # S1 loom
    (9, 8, "A"),  # S2
    (17, 8, "B"),
    (25, 8, "A"),
    (33, 8, "B"),  # S3
    (41, 8, "C"),
    (61, 8, "A"),  # S5
    (69, 8, "B"),
    (77, 8, "C"),
    (85, 12, "A"),  # S6 unravel — home to the end
)


def _place_pad(score: Score) -> None:
    """Constant R2 anchor + an 8-bar harmonic rhythm through A/B/C chords.

    The pad's slow attack/release crossfades adjacent chord blocks, so
    the harmony breathes on a longer cycle than the arp's 2-bar weave.
    The Breath (S4) gets its own bespoke voicing: subdominant focus with
    an s6 shade — the harmonic strange-place of the piece.
    """
    score.add_note("pad", start=0.0, duration=TOTAL_DUR, partial=2.0, amp_db=-10.0)

    for start_bar, n_bars, chord_name in _PAD_PROGRESSION:
        for partial, amp_db in _PAD_CHORDS[chord_name]:
            score.add_note(
                "pad",
                start=_pos(start_bar),
                duration=n_bars * BAR,
                partial=partial,
                amp_db=amp_db,
            )

    # S4 Breath: subdominant focus + s6 shade (the strangest chord),
    # gliding in from the P5 of the preceding C chord.
    score.add_note(
        "pad",
        start=_pos(S4_BAR),
        duration=(S5_BAR - S4_BAR) * BAR,
        partial=_BASS_P4,
        amp_db=-9.0,
        pitch_motion=PitchMotionSpec.ratio_glide(
            start_ratio=(3 / 2) / (4 / 3), end_ratio=1.0
        ),
    )
    score.add_note(
        "pad",
        start=_pos(S4_BAR),
        duration=(S5_BAR - S4_BAR) * BAR,
        partial=(49 / 30) * 2.0,  # s6 shade ≈ 160 Hz
        amp_db=-13.5,
    )
    score.add_note(
        "pad",
        start=_pos(S4_BAR),
        duration=(S5_BAR - S4_BAR) * BAR,
        partial=3.5,
        amp_db=-12.5,
    )


# ---------------------------------------------------------------------------
# Thread — a sparse vocal-ish lead that sings over the weave.
#
# Three appearances only: understated in S3's C-chord block, fuller at the
# start of S5, and a peak variant over S5's final C block.  Long tones,
# few notes — a voice humming over the loom, not a busy melody.  Each
# phrase saves its one spice tone for a framed moment (N2 shadow in A,
# the held s6 in B, the N2-above-the-octave peak in C).
#
# Rows: (bar, beat, degree, octave, dur_beats, velocity, amp_off_db).
# ---------------------------------------------------------------------------

_THREAD_PHRASES: tuple[
    tuple[tuple[int, int, int, int, float, float, float], ...], ...
] = (
    # Phrase A — S3, bars 41-46: enters with the C chord, low-key
    (
        (41, 2, 4, 3, 3.0, 0.60, -2.0),  # P5 entry
        (42, 1, 6, 3, 2.0, 0.65, -1.5),  # h7
        (42, 3, 3, 3, 2.0, 0.55, -2.5),  # P4
        (43, 1, 4, 3, 5.0, 0.70, -1.0),  # P5 held, sings
        (45, 2, 1, 3, 2.0, 0.50, -3.0),  # N2 shadow — framed spice
        (45, 4, 0, 3, 5.0, 0.60, -2.0),  # R settle
    ),
    # Phrase B — S5, bars 65-71: fuller, with the held s6 as its core
    (
        (65, 1, 0, 4, 3.0, 0.75, 0.0),  # R octave-up entry
        (65, 4, 6, 3, 1.0, 0.60, -2.0),
        (66, 1, 4, 3, 6.0, 0.70, -1.0),  # P5 held
        (68, 1, 5, 3, 3.0, 0.65, -1.5),  # s6 held — the strange singing tone
        (68, 4, 4, 3, 1.0, 0.55, -2.5),
        (69, 1, 3, 3, 4.0, 0.70, -1.0),  # P4 lands with the B chord
        (70, 2, 4, 3, 6.0, 0.60, -2.0),  # P5 sus over the lift, fades
    ),
    # Phrase C — S5 peak, bars 77-82: the highest moment of the piece
    (
        (77, 1, 6, 3, 2.0, 0.70, -1.0),
        (77, 3, 0, 4, 4.0, 0.80, 0.0),  # R4 held
        (79, 1, 1, 4, 2.0, 0.70, -1.0),  # N2 above the octave — peak spice
        (79, 3, 0, 4, 6.0, 0.75, -0.5),  # R4 resolution rings
        (81, 2, 4, 3, 6.0, 0.60, -2.0),  # P5 settles under the weave
    ),
)

_THREAD_AMP_BASE: float = -10.0


def _place_thread(score: Score) -> None:
    for phrase in _THREAD_PHRASES:
        for bar, beat, degree, octave, dur_beats, vel, amp_off in phrase:
            score.add_note(
                "thread",
                start=_pos(bar, beat),
                duration=dur_beats * BEAT,
                partial=_partial(degree, octave),
                amp_db=_THREAD_AMP_BASE + amp_off,
                velocity=vel,
                pitch_motion=PitchMotionSpec.vibrato(depth_ratio=0.006, rate_hz=4.2)
                if dur_beats >= 2.0
                else None,
            )


# ---------------------------------------------------------------------------
# Dust — tape-grain air, breathing under everything
# ---------------------------------------------------------------------------


def _place_dust(score: Score) -> None:
    """Overlapping 8-bar grain notes alternating R / P5, whole piece."""
    for i, bar in enumerate(range(1, TOTAL_BARS, 8)):
        partial = 4.0 if i % 2 == 0 else 6.0
        score.add_note(
            "dust",
            start=_pos(bar),
            duration=9.0 * BAR,  # 1-bar overlap into the next note
            partial=partial,
            amp_db=-6.0,
            velocity=0.6,
        )


# ---------------------------------------------------------------------------
# Drums
# ---------------------------------------------------------------------------


def _kick_bars() -> list[tuple[int, tuple[float, ...], float]]:
    """(bar, beats, amp_db) rows for the kick pattern per section."""
    rows: list[tuple[int, tuple[float, ...], float]] = []
    for bar in range(S2_BAR, 13):  # sparse entry: beat 1 only
        rows.append((bar, (1.0,), -8.0))
    for bar in range(13, S3_BAR):  # full head nod
        rows.append((bar, (1.0, 3.0), -6.5))
    for bar in range(S3_BAR, S4_BAR):
        beats: tuple[float, ...] = (1.0, 3.0)
        if (bar - S3_BAR) % 4 == 3:  # ghost on the a-of-4 every 4th bar
            beats = (1.0, 3.0, 4.75)
        rows.append((bar, beats, -6.0))
    for bar in range(S4_BAR, 53):  # breath approach: beat 1 only
        rows.append((bar, (1.0,), -8.5))
    # 53–58 silent
    for bar in (59, 60):  # return heartbeat
        rows.append((bar, (1.0,), -8.0))
    for bar in range(S5_BAR, S6_BAR):
        beats = (1.0, 3.0)
        if (bar - S5_BAR) % 4 == 3:
            beats = (1.0, 3.0, 4.75)
        rows.append((bar, beats, -5.5))
    for bar in range(S6_BAR, 91):  # unravel: thinning
        rows.append((bar, (1.0,), -8.0))
    for bar in (91, 92):
        rows.append((bar, (1.0,), -10.0))
    return rows


def _place_kick(score: Score) -> None:
    for bar, beats, amp in _kick_bars():
        for beat in beats:
            ghost = beat != int(beat)
            score.add_note(
                "kick",
                start=_pos(bar) + (beat - 1.0) * BEAT,
                duration=0.6,
                freq=F0,
                amp_db=amp - (4.0 if ghost else 0.0),
                velocity=0.55 if ghost else 0.9,
            )


def _place_hats(score: Score) -> None:
    """Closed hats on the offbeat 8ths with swing; 16th graces when open.

    Open hat marks the end of each 4-bar phrase (choke pair).
    """

    def _hat_note(bar: int, step: int, vel: float, amp: float) -> None:
        score.add_note(
            "hat_c",
            start=_pos(bar) + step * S16 + (HAT_SWING_S if step % 2 else 0.0),
            duration=0.05,
            freq=7200.0,
            amp_db=amp,
            velocity=vel,
        )

    for bar in range(17, TOTAL_BARS - 5):
        in_s3 = S3_BAR <= bar < S4_BAR
        in_s5 = S5_BAR <= bar < S6_BAR
        if S4_BAR <= bar < S5_BAR:
            continue  # breath: no hats
        if bar >= 89:
            continue  # unravel: hats gone by 89
        base_amp = -9.0 if (in_s3 or in_s5) else -11.0
        for step in (2, 6, 10, 14):
            _hat_note(bar, step, 0.72 if step in (6, 14) else 0.62, base_amp)
        # 16th graces on alternate bars once the weave opens
        if (in_s3 or in_s5) and bar % 2 == 1:
            for step in (7, 15):
                _hat_note(bar, step, 0.40, base_amp - 4.0)

    # Open hat at phrase ends (step 14, ringing over the barline)
    open_bars = [
        bar
        for bar in range(20, TOTAL_BARS - 8, 4)
        if not (S4_BAR <= bar < S5_BAR) and bar < S6_BAR
    ]
    for bar in open_bars:
        score.add_note(
            "hat_o",
            start=_pos(bar) + 14 * S16 + HAT_SWING_S,
            duration=0.4,
            freq=7200.0,
            amp_db=-11.0,
            velocity=0.7,
        )


def _place_brush(score: Score) -> None:
    """Soft brush backbeats (beats 2 & 4) in the open sections only."""
    for bar in range(S3_BAR, S6_BAR):
        if S4_BAR <= bar < S5_BAR:
            continue
        for step, vel in ((4, 0.6), (12, 0.7)):
            score.add_note(
                "brush",
                start=_pos(bar) + step * S16,
                duration=0.15,
                freq=800.0,
                amp_db=-11.0,
                velocity=vel,
            )


# ---------------------------------------------------------------------------
# Buses and automation
# ---------------------------------------------------------------------------


def _make_hall_bus() -> SendBusSpec:
    """Large dark hall; voice send_db does the balancing (return at 0)."""
    if BRICASTI_IR_DIR.exists():
        effects = [
            EffectSpec(
                "bricasti",
                {
                    "ir_name": "1 Halls 07 Large & Dark",
                    "wet": 1.0,
                    "lowpass_hz": 5200.0,
                    "highpass_hz": 190.0,
                },
            ),
        ]
    else:
        effects = [
            EffectSpec("reverb", {"room_size": 0.84, "damping": 0.5, "wet_level": 1.0}),
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 190.0,
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
        ]
    return SendBusSpec(name="hall", effects=effects, return_db=0.0)


def _ramp(
    target: AutomationTarget,
    points: list[tuple[float, float, float, AutomationShape]],
    default: float,
) -> AutomationSpec:
    """Build a replace-mode automation from (start, end, to_value, shape) rows.

    Each row ramps from the previous row's end value; gaps between rows
    are bridged with hold segments so the value persists (replace mode
    falls back to the base value outside segments otherwise).
    """
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


def _arp_cutoff_automation() -> AutomationSpec:
    return _ramp(
        AutomationTarget(kind="synth", name="cutoff_hz"),
        [
            (S1_END, S2_END, 1900.0, "exp"),
            (S2_END, S3_END, 2800.0, "exp"),  # opening
            (S3_END, S3_END + 4 * BAR, 1700.0, "exp"),  # breath darkens
            (S4_END, S4_END + 8 * BAR, 3200.0, "exp"),  # second weave peak
            (S5_END, TOTAL_DUR, 1400.0, "exp"),  # unravel closes down
        ],
        default=1500.0,
    )


def _arp_pan_sway() -> AutomationSpec:
    return _ramp(
        AutomationTarget(kind="control", name="pan"),
        [
            (0.0, S2_END, 0.10, "linear"),
            (S2_END, S3_END, -0.20, "linear"),
            (S3_END, S4_END, 0.05, "linear"),
            (S4_END, S5_END, -0.15, "linear"),
            (S5_END, TOTAL_DUR, -0.05, "linear"),
        ],
        default=-0.12,
    )


def _arp_hall_ride() -> AutomationSpec:
    """Hall send opens wide during the Breath, settles after."""
    return _ramp(
        AutomationTarget(kind="control", name="send_db"),
        [
            (S3_END - 2 * BAR, S3_END + 2 * BAR, -4.0, "linear"),
            (S4_END - 2 * BAR, S4_END + 2 * BAR, -8.0, "linear"),
        ],
        default=-8.0,
    )


def _pad_brightness_arc() -> AutomationSpec:
    return _ramp(
        AutomationTarget(kind="synth", name="brightness_tilt"),
        [
            (0.0, S2_END, -0.05, "linear"),
            (S2_END, S3_END, 0.05, "linear"),
            (S3_END, S4_END, -0.10, "linear"),  # breath: darker, closer
            (S4_END, S5_END, 0.10, "linear"),  # second weave glows
            (S5_END, TOTAL_DUR, -0.12, "linear"),
        ],
        default=-0.10,
    )


def _dust_mix_ride() -> AutomationSpec:
    """Dust rises into the Breath's near-silence, recedes after."""
    return _ramp(
        AutomationTarget(kind="control", name="mix_db"),
        [
            (S3_END - 2 * BAR, S3_END + 4 * BAR, -12.0, "linear"),
            (S4_END - 2 * BAR, S4_END + 2 * BAR, -17.0, "linear"),
        ],
        default=-17.0,
    )


# ---------------------------------------------------------------------------
# build_score
# ---------------------------------------------------------------------------


def build_score() -> Score:
    """Build the Weaving Room score."""
    score = Score(
        f0_hz=F0,
        master_effects=DEFAULT_MASTER_EFFECTS,
        send_buses=[_make_hall_bus()],
        timing_humanize=TimingHumanizeSpec(preset="chamber"),
    )
    score.add_drift_bus("weave_drift", rate_hz=0.10, depth_cents=4.0, seed=41)
    # "electronic" style includes the clipper — needed to keep the subby
    # 49 Hz kick's peaks from driving the export limiter into IMD.
    drum_bus = setup_drum_bus(score, style="electronic", return_db=0.0)

    # ---- Arp 1: the spine — warm KS pluck through a gentle SVF ----
    score.add_voice(
        "arp",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "soft_pluck",
            "release": 0.8,
            "filter_mode": "lowpass",
            "filter_topology": "svf",
            "filter_cutoff_hz": 1500.0,
            "resonance_q": 0.8,
        },
        effects=[
            EffectSpec(
                "delay",
                {"delay_seconds": DOTTED_EIGHTH, "feedback": 0.34, "mix": 0.22},
            ),
        ],
        sends=[VoiceSend(target="hall", send_db=-8.0, automation=[_arp_hall_ride()])],
        pan=-0.12,
        mix_db=-1.5,
        envelope_humanize=EnvelopeHumanizeSpec(preset="loose_pluck"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        drift_bus="weave_drift",
        drift_bus_correlation=0.6,
        automation=[_arp_cutoff_automation(), _arp_pan_sway()],
    )

    # ---- Arp 2: the canon — same cloth, darker filter, other side ----
    score.add_voice(
        "arp2",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "soft_pluck",
            "release": 1.0,
            "filter_mode": "lowpass",
            "filter_topology": "svf",
            "filter_cutoff_hz": 1300.0,
            "resonance_q": 0.7,
        },
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 220.0,
                            "slope_db_per_oct": 12,
                        },
                    ]
                },
            ),
            EffectSpec(
                "delay",
                {"delay_seconds": DOTTED_EIGHTH, "feedback": 0.28, "mix": 0.18},
            ),
        ],
        sends=[VoiceSend(target="hall", send_db=-7.0)],
        pan=0.28,
        mix_db=-5.0,
        envelope_humanize=EnvelopeHumanizeSpec(preset="loose_pluck"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        drift_bus="weave_drift",
        drift_bus_correlation=0.6,
    )

    # ---- Pad: Colundi-spectrum additive drone ----
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "additive_pad_through_ladder",
            "partials_partials": _COLUNDI_PARTIALS,
            "filter_cutoff_hz": 1200.0,
            "attack": 3.0,
            "release": 4.0,
            "brightness_tilt": -0.10,
        },
        sends=[VoiceSend(target="hall", send_db=-6.0)],
        pan=0.0,
        mix_db=-10.5,
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        drift_bus="weave_drift",
        drift_bus_correlation=0.75,
        automation=[_pad_brightness_arc()],
    )

    # ---- Thread: sparse vocal-ish lead over the weave ----
    score.add_voice(
        "thread",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "formant_vowel_lead",
            "attack": 0.25,
            "release": 1.6,
        },
        sends=[VoiceSend(target="hall", send_db=-4.0)],
        pan=0.08,
        mix_db=-6.0,
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        drift_bus="weave_drift",
        drift_bus_correlation=0.5,
    )

    # ---- Bass: mono legato undertow ----
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "soft_bass",
            "release": 0.5,
        },
        pan=0.0,
        mix_db=-6.0,
        max_polyphony=1,
        legato=True,
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        drift_bus="weave_drift",
        drift_bus_correlation=0.8,
    )

    # ---- Dust: tape-grain air ----
    score.add_voice(
        "dust",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "grain_tape_dust",
        },
        sends=[VoiceSend(target="hall", send_db=-6.0)],
        pan=0.0,
        mix_db=-17.0,
        automation=[_dust_mix_ride()],
    )

    # ---- Drums ----
    add_drum_voice(
        score,
        "kick",
        engine="drum_voice",
        preset="808_house",
        drum_bus=drum_bus,
        send_db=-2.0,
        synth_overrides={
            "tone_punch": 0.18,
            "tone_second_harmonic": 0.06,
            "exciter_level": 0.05,
        },
        mix_db=-4.0,
    )
    add_drum_voice(
        score,
        "hat_c",
        engine="drum_voice",
        preset="closed_hat",
        drum_bus=drum_bus,
        send_db=-6.0,
        choke_group="hats",
        mix_db=-15.0,
        pan=0.15,
    )
    add_drum_voice(
        score,
        "hat_o",
        engine="drum_voice",
        preset="open_hat",
        drum_bus=drum_bus,
        send_db=-6.0,
        choke_group="hats",
        mix_db=-16.0,
        pan=0.15,
    )
    add_drum_voice(
        score,
        "brush",
        engine="drum_voice",
        preset="brush",
        drum_bus=drum_bus,
        send_db=-4.0,
        mix_db=-15.0,
        pan=0.3,
    )

    # ==================================================================
    # Notes
    # ==================================================================
    history = _weave_history()
    _place_arps(score, history)
    _place_bass(score)
    _place_pad(score)
    _place_thread(score)
    _place_dust(score)
    _place_kick(score)
    _place_hats(score)
    _place_brush(score)

    return score


PIECES: dict[str, PieceDefinition] = {
    "weaving_room": PieceDefinition(
        name="weaving_room",
        output_name="weaving_room",
        build_score=build_score,
        sections=(
            PieceSection(label="S1 loom", start_seconds=0.0, end_seconds=S1_END),
            PieceSection(
                label="S2 first weave", start_seconds=S1_END, end_seconds=S2_END
            ),
            PieceSection(label="S3 opening", start_seconds=S2_END, end_seconds=S3_END),
            PieceSection(label="S4 breath", start_seconds=S3_END, end_seconds=S4_END),
            PieceSection(
                label="S5 second weave", start_seconds=S4_END, end_seconds=S5_END
            ),
            PieceSection(
                label="S6 unravel", start_seconds=S5_END, end_seconds=TOTAL_DUR
            ),
        ),
    ),
}
