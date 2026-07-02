"""VA Trance Anthem — uplifting-euphoric trance in 5-limit JI F minor.

Showcases the VA engine's JP-8000 / Virus / Waldorf-Q presets in their
natural habitat: 4-on-the-floor drums, sidechain pump, filter-sweep builds,
and a classic anthem chord progression (i – ♭VI – ♭III – ♭VII).

Tuning:  5-limit JI, F0 = 43.654 Hz (F1).  Optional 7/4 color on bass in
DROP-2 only; everything else stays 5-limit.

Voice map:
    kick    — drum_voice / 909_house   (4-on-the-floor)
    clap    — drum_voice / 909_clap    (beats 2 & 4 in drops)
    hat     — drum_voice / chh         (16ths, density-modulated)
    open_hat— drum_voice / open_hat    (off-beat sparkles in drops)
    bass    — va / virus_bass          (off-beat 8ths, kick-ducked)
    pad     — va / supersaw_pad        (sustained chords, filter sweeps)
    hoover  — va / jp8000_hoover       (stabs on chord changes)
    lead    — va / virus_lead          (sync-lead topline)
    bells   — va / q_comb_bell         (breakdown melody + counterline)

Form (128 bars, ~3:42 at 138 BPM):
    bars   1-8    INTRO        kick + hat + pad + atmospheric wash + bell foreshadow
    bars   9-24   BREAKDOWN-A  bells state theme over pad, no drums
    bars  25-32   BUILD-1      drums return, long filter sweep, risers
    bars  33-56   DROP-1       full arrangement with topline lead
    bars  57-72   BREAKDOWN-B  emotional peak: pad + bells + lead, no drums
    bars  73-88   BUILD-2      extended: hoover escalation, reverse cymbal, 4-bar riser
    bars  89-96   DROP-2a      full arrangement, Drop-2 variations (walking bass,
                               lead harmony, kick break on 96, shaker)
    bars  97-104  CUTAWAY      mid-drop breakdown on Ab — lead/bells conversation,
                               no kick, pad holds single chord, reverse swell at 103
    bars 105-120  DROP-2b      slam-back, full arrangement, climactic lift
    bars 121-128  OUTRO        elements peel away to tonic + reverb washout
"""

from __future__ import annotations

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
from code_musics.pieces._shared import (
    DEFAULT_MASTER_EFFECTS,
    SOFT_REVERB_EFFECT,
)
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import EffectSpec, Score, VoiceSend

# ── Tempo and grid ────────────────────────────────────────────────────
BPM: float = 138.0
BEAT: float = 60.0 / BPM
BAR: float = 4.0 * BEAT
S16: float = BEAT / 4.0
F0: float = 43.654  # F1 — stays in the F–G# electronic-kick sweet spot
TOTAL_BARS: int = 128

# ── Section boundaries (1-indexed bars, exclusive upper bound) ────────
INTRO_END = 9  # bars 1-8
BREAKDOWN_A_END = 25  # bars 9-24
BUILD_1_END = 33  # bars 25-32
DROP_1_END = 57  # bars 33-56
BREAKDOWN_B_END = 73  # bars 57-72
BUILD_2_END = 89  # bars 73-88   (16-bar build, escalatory)
CUTAWAY_START = 97  # bars 97-104 — mid-Drop-2 breakdown on Ab
CUTAWAY_END = 105
DROP_2_END = 121  # bars 89-120  (32-bar drop with cutaway in the middle)
OUTRO_END = TOTAL_BARS + 1  # bars 121-128

# ── Chord progression: Fm – Db – Ab – Eb ──────────────────────────────
# Ratios relative to F (f0 = F1).  Each chord lasts 4 bars.
# Root, minor third, fifth — then octave-doubled for pad richness.
#   Fm:  1, 6/5, 3/2           (F, Ab, C)
#   Db:  8/5, 2, 12/5          (Db, F, Ab)
#   Ab:  6/5, 3/2, 9/5         (Ab, C, Eb)
#   Eb:  9/5, 9/4, 27/10       (Eb, G, Bb)   — major triad on ♭VII
PAD_VOICINGS: list[list[float]] = [
    # Voice leading: each chord change moves as few voices as possible, and
    # only by step, sharing held tones across boundaries.  Register Eb3-F4.
    # Bass voice handles the real roots; middle chords (Db, Ab) sit in
    # inversion, which is what lets the upper voices hold still.
    # Motion audit (per cycle):
    #   Fm → Db: 1 voice moves  (C4 → Db4)
    #   Db → Ab: 3 voices move  (F3 → Eb3, Db4 → C4, F4 → Eb4), Ab3 held
    #   Ab → Eb: 2 voices move  (Ab3 → G3, C4 → Bb3), Eb3 & Eb4 held
    #   Eb → Fm: 4 voices move  (the cycle "lift" — all by step)
    [4.0, 24 / 5, 6.0, 8.0],  # Fm: F3, Ab3, C4, F4
    [4.0, 24 / 5, 32 / 5, 8.0],  # Db/F: F3, Ab3, Db4, F4     (1st inversion)
    [18 / 5, 24 / 5, 6.0, 36 / 5],  # Ab/Eb: Eb3, Ab3, C4, Eb4  (2nd inversion)
    [18 / 5, 9 / 2, 27 / 5, 36 / 5],  # Eb: Eb3, G3, Bb3, Eb4   (root position)
]

# Bass roots for the off-beat 8th bounce (one octave below pad roots).
# Fm=2/1, Db=8/5, Ab=12/5, Eb=9/5  (partial of F1 in octave 2)
BASS_ROOTS: list[float] = [2.0, 8 / 5, 12 / 5, 9 / 5]

# 7-limit color for DROP-2: optional harmonic-seventh on Eb chord bar
# (bar 4 of each loop).  Replaces the 9/5 root with 7/4 only in the bass.
BASS_ROOTS_7LIM: list[float] = [2.0, 8 / 5, 12 / 5, 7 / 4]

# Hoover-stab chord voicings (lower and tighter than the pad — stabs).
HOOVER_VOICINGS: list[list[float]] = [
    [2.0, 12 / 5, 3.0],  # Fm: F2, Ab2, C3
    [8 / 5, 2.0, 12 / 5],  # Db: Db2, F2, Ab2
    [12 / 5, 3.0, 18 / 5],  # Ab: Ab2, C3, Eb3
    [9 / 5, 9 / 4, 27 / 10],  # Eb: Eb2, G2, Bb2
]


# ── Lead melody — memorable topline for the drops ────────────────────
# The lead sits an octave above the pad (partials 8-16 range).
# Each bar gets a short phrase; each 4-bar block follows one chord.
# Shape: descending motion for sadness/yearning, rising for lift.
# Uses ratios from the chord tone set plus 4/3 for passing motion.
_C5 = 6.0  # C5 over F1 (5th of Fm)
_Eb5 = 36 / 5  # Eb5
_F5 = 8.0  # F5 (tonic octave)
_G5 = 9.0  # G5 (note: major-2, used over Eb chord)
_Ab5 = 48 / 5  # Ab5 (minor third)
_Bb5 = 54 / 5  # Bb5
_C6 = 12.0  # C6 (5th of Fm octave)
_Db6 = 64 / 5  # Db6
_Eb6 = 72 / 5  # Eb6
_F6 = 16.0  # F6

# Main lead phrase per chord in the DROP — (beat, n16, partial, gate_in_16ths, velocity)
# Chord 1 (Fm): yearning descent F5 -> C5, pause, reaching back up
_LEAD_FM_DROP: list[tuple[int, int, float, int, float]] = [
    (1, 0, _F5, 4, 0.85),
    (2, 0, _Eb5, 2, 0.70),
    (2, 2, _C5, 2, 0.65),
    (3, 0, _F5, 3, 0.78),
    (3, 3, _Ab5, 5, 0.88),
]
# Chord 2 (Db): reaching upward, Ab -> Db6 -> F6 is the lift
_LEAD_DB_DROP: list[tuple[int, int, float, int, float]] = [
    (1, 0, _Ab5, 4, 0.82),
    (2, 0, _Db6, 3, 0.80),
    (2, 3, _F6, 4, 0.92),
    (3, 3, _Db6, 2, 0.75),
    (4, 1, _Ab5, 3, 0.70),
]
# Chord 3 (Ab): sighing descent, chord tones of Ab major
_LEAD_AB_DROP: list[tuple[int, int, float, int, float]] = [
    (1, 0, _C6, 4, 0.88),
    (2, 0, _Bb5, 2, 0.72),
    (2, 2, _Ab5, 2, 0.72),
    (3, 0, _Eb5, 4, 0.68),
    (3, 3, _C5, 5, 0.75),
]
# Chord 4 (Eb): resolution lift, setting up return to Fm
_LEAD_EB_DROP: list[tuple[int, int, float, int, float]] = [
    (1, 0, _Bb5, 3, 0.85),
    (1, 3, _G5, 2, 0.70),
    (2, 1, _Eb5, 3, 0.72),
    (3, 0, _G5, 2, 0.78),
    (3, 2, _Bb5, 2, 0.82),
    (4, 0, _Eb6, 4, 0.92),
]
LEAD_DROP: list[list[tuple[int, int, float, int, float]]] = [
    _LEAD_FM_DROP,
    _LEAD_DB_DROP,
    _LEAD_AB_DROP,
    _LEAD_EB_DROP,
]

# Breakdown-B lead: more lyrical, longer sustained notes, vibrato-friendly
_LEAD_FM_BD: list[tuple[int, int, float, int, float]] = [
    (1, 0, _F5, 6, 0.72),
    (2, 2, _Eb5, 2, 0.62),  # passing tone into Ab5 — mirrors the drop's descent
    (3, 0, _Ab5, 8, 0.78),
]
_LEAD_DB_BD: list[tuple[int, int, float, int, float]] = [
    (1, 0, _F6, 10, 0.82),
    (3, 2, _Db6, 6, 0.76),
]
_LEAD_AB_BD: list[tuple[int, int, float, int, float]] = [
    (1, 0, _C6, 6, 0.78),
    (2, 2, _Bb5, 4, 0.72),
    (3, 2, _Ab5, 6, 0.70),
]
_LEAD_EB_BD: list[tuple[int, int, float, int, float]] = [
    (1, 0, _Bb5, 6, 0.74),
    (3, 0, _Eb6, 10, 0.85),
]
LEAD_BD: list[list[tuple[int, int, float, int, float]]] = [
    _LEAD_FM_BD,
    _LEAD_DB_BD,
    _LEAD_AB_BD,
    _LEAD_EB_BD,
]

# ── Bells melody for BREAKDOWN-A ──────────────────────────────────────
# Simpler, hymn-like statement of the theme — lets the pad breathe.
# Sits one register lower than lead (partials 4-12 range).
_BELL_FM_A: list[tuple[int, int, float, int, float]] = [
    (1, 0, _F5, 4, 0.78),
    (3, 0, _Ab5, 4, 0.72),
]
_BELL_DB_A: list[tuple[int, int, float, int, float]] = [
    (1, 0, _Ab5, 4, 0.75),
    (3, 0, _Db6, 4, 0.80),
]
_BELL_AB_A: list[tuple[int, int, float, int, float]] = [
    (1, 0, _C6, 4, 0.82),
    (3, 0, _Bb5, 4, 0.72),
]
_BELL_EB_A: list[tuple[int, int, float, int, float]] = [
    (1, 0, _Bb5, 4, 0.78),
    (3, 0, _G5, 2, 0.65),
    (3, 2, _Bb5, 2, 0.70),
]
BELL_A: list[list[tuple[int, int, float, int, float]]] = [
    _BELL_FM_A,
    _BELL_DB_A,
    _BELL_AB_A,
    _BELL_EB_A,
]

# Bells counterline for BREAKDOWN-B — higher, more arpeggiated, answers the lead
_BELL_FM_B: list[tuple[int, int, float, int, float]] = [
    (2, 0, _C6, 2, 0.65),
    (2, 2, _F6, 2, 0.72),
    (4, 0, _Ab5, 2, 0.60),
    (4, 2, _C6, 2, 0.65),
]
_BELL_DB_B: list[tuple[int, int, float, int, float]] = [
    (2, 0, _Db6, 2, 0.68),
    (2, 2, _F6, 2, 0.75),
    (4, 0, _Ab5, 2, 0.62),
    (4, 2, _Db6, 2, 0.70),
]
_BELL_AB_B: list[tuple[int, int, float, int, float]] = [
    (2, 0, _Eb6, 2, 0.72),
    (2, 2, _C6, 2, 0.68),
    (4, 0, _Bb5, 2, 0.60),
    (4, 2, _Eb6, 2, 0.72),
]
_BELL_EB_B: list[tuple[int, int, float, int, float]] = [
    (2, 0, _Eb6, 2, 0.75),
    (2, 2, _Bb5, 2, 0.62),
    (4, 0, _G5, 2, 0.60),
    (4, 2, _Bb5, 2, 0.70),
]
# ── Cutaway lead + bells (bars 97-104, 8-bar phrase on Ab) ────────────
# Lead and bells in three-voice conversation.  Bar 101 is the trough
# (silence — pad filter is closing), then the climb back begins.
# All partials come from the Ab-major chord tones: Ab (24/5), C (6),
# Eb (36/5), with 4/3 (Bb, partial 27/5) as a reaching color tone,
# and F (8) as the leading-tone back into Fm at bar 105.
# Format: list of 8 bar-motifs, each a list of (beat, n16, partial, gate_16ths, velocity).
LEAD_CUTAWAY: list[list[tuple[int, int, float, int, float]]] = [
    # Bar 97 — lead call A: Ab5 → C6 (chord-tone reach)
    [(1, 0, _Ab5, 6, 0.70), (3, 0, _C6, 6, 0.75)],
    # Bar 98 — lead rests; bells carry (populated below in BELL_CUTAWAY)
    [],
    # Bar 99 — lead call B: C6 → Eb6 (reach higher, the hopeful move)
    [(1, 0, _C6, 4, 0.72), (2, 2, _Eb6, 8, 0.80)],
    # Bar 100 — lead rests; bells echo
    [],
    # Bar 101 — trough, total silence on lead
    [],
    # Bar 102 — single sustained Bb5 (the 9th of Ab — yearning, unresolved)
    [(1, 0, _Bb5, 12, 0.68)],
    # Bar 103 — lead lift: C6 sustained into the arpeggio below
    [(1, 0, _C6, 8, 0.78), (3, 0, _Eb6, 4, 0.82)],
    # Bar 104 — the climb home: F5 → C6 → F6 into the Fm slam at bar 105
    [
        (1, 0, _F5, 2, 0.80),
        (1, 2, _C6, 2, 0.84),
        (2, 0, _F6, 8, 0.90),
        (4, 0, _C6, 2, 0.75),
        (4, 2, _F6, 2, 0.88),
    ],
]

# Cutaway bells — answer the lead's calls, arpeggiate on rest bars
BELL_CUTAWAY: list[list[tuple[int, int, float, int, float]]] = [
    # Bar 97 — bells rest (lead leading)
    [],
    # Bar 98 — answer call A with Ab-chord arpeggio (Ab-C-Eb ascending)
    [
        (1, 0, _Ab5, 2, 0.72),
        (1, 2, _C6, 2, 0.70),
        (2, 0, _Eb6, 4, 0.78),
        (3, 2, _C6, 2, 0.66),
    ],
    # Bar 99 — bells rest
    [],
    # Bar 100 — answer call B with higher arpeggio (C-Eb-F-Ab back down)
    [
        (1, 0, _F6, 2, 0.72),
        (1, 2, _Eb6, 2, 0.70),
        (2, 0, _C6, 4, 0.68),
        (3, 2, _Ab5, 2, 0.62),
    ],
    # Bar 101 — trough, nothing
    [],
    # Bar 102 — single bell toll on beat 3 (distant, lonely)
    [(3, 0, _Eb6, 8, 0.55)],
    # Bar 103 — arpeggio climbing with the lead
    [
        (1, 0, _Ab5, 2, 0.68),
        (1, 2, _C6, 2, 0.70),
        (2, 0, _Eb6, 2, 0.72),
        (2, 2, _F6, 2, 0.74),
        (4, 0, _Eb6, 2, 0.70),
    ],
    # Bar 104 — shimmer under the lead's final climb
    [
        (1, 0, _C6, 2, 0.70),
        (1, 2, _Eb6, 2, 0.72),
        (3, 0, _F6, 2, 0.78),
        (3, 2, _Ab5, 2, 0.70),
    ],
]


BELL_B: list[list[tuple[int, int, float, int, float]]] = [
    _BELL_FM_B,
    _BELL_DB_B,
    _BELL_AB_B,
    _BELL_EB_B,
]


# ── Helpers ───────────────────────────────────────────────────────────
def _pos(bar: int, beat: int = 1, n16: int = 0) -> float:
    """Bar/beat/16th to absolute seconds (1-indexed bar and beat)."""
    return (bar - 1) * BAR + (beat - 1) * BEAT + n16 * S16


def _ci(bar: int) -> int:
    """Chord index 0..3 — cycles every 4 bars."""
    return (bar - 1) % 4


def _is_drop_1(bar: int) -> bool:
    return BUILD_1_END <= bar < DROP_1_END


def _is_drop_2(bar: int) -> bool:
    """True anywhere inside Drop-2 (bars 89-120), *including* the cutaway.

    Use this for continuous Drop-2 state (automation, voice enable flags that
    don't need to drop for the cutaway).  Use ``_is_drop_2_active`` instead
    when a voice should stop playing during the cutaway.
    """
    return BUILD_2_END <= bar < DROP_2_END


def _is_cutaway(bar: int) -> bool:
    """True inside the 8-bar mid-Drop-2 cutaway (bars 97-104)."""
    return CUTAWAY_START <= bar < CUTAWAY_END


def _is_drop_2_active(bar: int) -> bool:
    """Drop-2 with the cutaway excluded — the 'arrangement is playing' flag."""
    return _is_drop_2(bar) and not _is_cutaway(bar)


def _is_drop(bar: int) -> bool:
    return _is_drop_1(bar) or _is_drop_2(bar)


def _is_drop_active(bar: int) -> bool:
    """Either drop's full arrangement — excludes the cutaway in Drop-2."""
    return _is_drop_1(bar) or _is_drop_2_active(bar)


def _is_breakdown_a(bar: int) -> bool:
    return INTRO_END <= bar < BREAKDOWN_A_END


def _is_breakdown_b(bar: int) -> bool:
    return DROP_1_END <= bar < BREAKDOWN_B_END


def _is_build_1(bar: int) -> bool:
    return BREAKDOWN_A_END <= bar < BUILD_1_END


def _is_build_2(bar: int) -> bool:
    return BREAKDOWN_B_END <= bar < BUILD_2_END


def _is_build(bar: int) -> bool:
    return _is_build_1(bar) or _is_build_2(bar)


def _is_outro(bar: int) -> bool:
    return DROP_2_END <= bar < OUTRO_END


# Section boundary times (seconds)
_T_START = _pos(1)
_T_BDA = _pos(INTRO_END)
_T_B1 = _pos(BREAKDOWN_A_END)
_T_D1 = _pos(BUILD_1_END)
_T_BDB = _pos(DROP_1_END)
_T_B2 = _pos(BREAKDOWN_B_END)
_T_D2 = _pos(BUILD_2_END)
_T_CUTAWAY = _pos(CUTAWAY_START)
_T_CUTAWAY_MID = _pos(CUTAWAY_START + 4)  # bar 101 — filter trough + reopen
_T_CUTAWAY_END = _pos(CUTAWAY_END)
_T_OUT = _pos(DROP_2_END)
_T_END = _pos(TOTAL_BARS + 1)


def _seg(
    start: float,
    end: float,
    start_v: float,
    end_v: float,
    shape: AutomationShape = "linear",
) -> AutomationSegment:
    return AutomationSegment(
        start=start,
        end=end,
        shape=shape,
        start_value=start_v,
        end_value=end_v,
    )


def _hold(start: float, end: float, value: float) -> AutomationSegment:
    return AutomationSegment(start=start, end=end, shape="hold", value=value)


def _synth(name: str) -> AutomationTarget:
    return AutomationTarget(kind="synth", name=name)


def _ctrl(name: str) -> AutomationTarget:
    return AutomationTarget(kind="control", name=name)


def _duck(tier: str = "standard") -> EffectSpec:
    """Kick-sidechain ducking at one of three strengths.

    - ``pad``: kick_duck_hard (4:1, 300 ms release). Audible swell-back; use on
      sustained chord beds so the pump is part of the groove.
    - ``light``: 2:1, 80 ms release, ~3 dB GR. Tight hat/perc duck — moves with
      the kick without choking the hiss.
    - ``standard``: kick_duck (3:1, 100 ms release). The default everywhere
      else (bass, lead, bells, hoover, etc).
    """
    if tier == "pad":
        return EffectSpec(
            "compressor",
            {"preset": "kick_duck_hard", "sidechain_source": "kick"},
        )
    if tier == "light":
        return EffectSpec(
            "compressor",
            {
                "sidechain_source": "kick",
                "threshold_db": -12.0,
                "ratio": 2.0,
                "attack_ms": 1.0,
                "release_ms": 80.0,
                "knee_db": 2.0,
                "makeup_gain_db": 0.0,
                "mix": 1.0,
                "topology": "feedforward",
                "detector_mode": "peak",
                "lookahead_ms": 1.0,
            },
        )
    return EffectSpec(
        "compressor",
        {"preset": "kick_duck", "sidechain_source": "kick"},
    )


# ── Piece builder ─────────────────────────────────────────────────────
def build_score() -> Score:
    """Uplifting-euphoric trance anthem built around the VA engine."""
    score = Score(
        f0_hz=F0,
        master_effects=list(DEFAULT_MASTER_EFFECTS),
        timing_humanize=TimingHumanizeSpec(preset="tight_ensemble", seed=11),
    )

    # ── Send buses ────────────────────────────────────────────────────
    score.add_send_bus(
        "plate",
        effects=[SOFT_REVERB_EFFECT],
        return_db=-2.0,
    )
    score.add_send_bus(
        "delay",
        effects=[
            EffectSpec(
                "delay",
                {
                    "delay_seconds": BEAT,  # quarter-note delay
                    "feedback": 0.38,
                    "mix": 1.0,
                },
            ),
        ],
        return_db=-4.0,
    )
    score.add_send_bus(
        "pad_chorus",
        effects=[
            EffectSpec("bbd_chorus", {"preset": "juno_i_plus_ii", "mix": 0.55}),
        ],
        return_db=0.0,
    )

    # Drum sub-mix bus (glue on the kit as a unit)
    drum_bus = setup_drum_bus(
        score,
        bus_name="drum_bus",
        style="electronic",
        return_db=0.0,
    )

    # ── Kick ──────────────────────────────────────────────────────────
    add_drum_voice(
        score,
        "kick",
        engine="drum_voice",
        preset="909_house",
        drum_bus=drum_bus,
        send_db=0.0,
        mix_db=-4.0,
        synth_overrides={"params": {"tone_decay_s": 0.15, "tone_punch": 0.10}},
    )
    for bar in range(1, TOTAL_BARS + 1):
        # Kick present in intro (thin), both drops (minus cutaway), both builds.
        # Silent in breakdowns, cutaway, and outro tail.
        if _is_breakdown_a(bar) or _is_breakdown_b(bar):
            continue
        if _is_cutaway(bar):
            continue
        if bar >= DROP_2_END + 6:  # last 2 bars of outro
            continue
        # Drop kick for the final bar of each build so the drop hits feel bigger.
        if bar == BUILD_1_END - 1 or bar == BUILD_2_END - 1:
            continue
        # Kick breaks: bars 48 (mid-Drop-1 phrase) and the last bar before
        # the cutaway (bar 96) — beats 3 & 4 drop so the air lands first.
        # Bar 96 also serves as the cutaway cue.
        kick_break_bars = {BUILD_1_END + 15, CUTAWAY_START - 1}  # 48, 96
        for beat in range(1, 5):
            if bar in kick_break_bars and beat in (3, 4):
                continue
            amp = -8.0 if bar < INTRO_END else -4.0
            score.add_note(
                "kick",
                start=_pos(bar, beat),
                duration=0.6,
                freq=55.0,
                amp_db=amp,
            )

    # ── Backbeat: layered clap + snare on 2 & 4 (drops only) ──────────
    # The 909_clap preset is pure bandpass noise with no pitched body, which
    # reads as "hissy hands" on its own. Layering a body-forward `snare`
    # engine voice under it gives the classic 909 snare+clap image — the
    # clap provides the transient snap, the snare provides the 150-250 Hz
    # punch that anchors the backbeat.
    add_drum_voice(
        score,
        "clap",
        engine="drum_voice",
        preset="909_clap",
        drum_bus=drum_bus,
        send_db=-2.0,
        mix_db=-10.0,
        pan=0.12,
        effects=[],  # keep it dry-ish inside the drum bus
    )
    score.voices["clap"].sends.append(VoiceSend(target="plate", send_db=-12.0))

    add_drum_voice(
        score,
        "snare",
        engine="snare",
        preset="909_fat",
        drum_bus=drum_bus,
        send_db=-4.0,
        mix_db=-10.5,
        pan=-0.10,
        synth_overrides={
            "body_mix": 0.62,
            "wire_mix": 0.38,
            "body_decay": 0.18,
            "click_amount": 0.20,
            "comb_amount": 0.55,
        },
        effects=[
            EffectSpec(
                "compressor",
                {"preset": "snare_punch", "attack_ms": 3.0, "ratio": 3.5},
            ),
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "bell", "freq_hz": 200.0, "gain_db": 2.5, "q": 1.1},
                        {"kind": "bell", "freq_hz": 3800.0, "gain_db": 1.5, "q": 1.4},
                        {"kind": "high_shelf", "freq_hz": 7000.0, "gain_db": -1.5},
                    ]
                },
            ),
            EffectSpec("transistor", {"preset": "snare_bite"}),
        ],
    )
    score.voices["snare"].sends.append(VoiceSend(target="plate", send_db=-14.0))

    for bar in range(1, TOTAL_BARS + 1):
        if not _is_drop_active(bar):
            continue
        for beat in [2, 4]:
            score.add_note(
                "clap",
                start=_pos(bar, beat),
                duration=0.12,
                freq=3000.0,
                amp_db=-3.0,
            )
            score.add_note(
                "snare",
                start=_pos(bar, beat),
                duration=0.18,
                freq=200.0,
                amp_db=-4.0,
            )

    # ── Hats (16ths with density automation) ──────────────────────────
    add_drum_voice(
        score,
        "hat",
        engine="drum_voice",
        preset="chh",
        drum_bus=drum_bus,
        send_db=-6.0,
        mix_db=-14.0,
        effects=[
            EffectSpec(
                "eq",
                {"bands": [{"kind": "high_shelf", "freq_hz": 8500.0, "gain_db": 2.0}]},
            ),
            EffectSpec("compressor", {"preset": "hat_control"}),
            _duck("light"),
        ],
    )
    for bar in range(1, TOTAL_BARS + 1):
        # Hat gating: absent in breakdowns, 8ths in intro/outro, 16ths in drops.
        if _is_breakdown_a(bar) or _is_breakdown_b(bar):
            continue
        if bar >= DROP_2_END + 6:
            continue
        # During the cutaway, hat stays on offbeats only — keeps pulse barely
        # alive so the slam-back feels like something returning, not starting.
        if _is_cutaway(bar):
            for beat in range(1, 5):
                score.add_note(
                    "hat",
                    start=_pos(bar, beat, 2),
                    duration=0.04,
                    freq=11500.0,
                    amp_db=-18.0,
                )
            continue
        if bar < INTRO_END:
            density = "8ths"
        elif _is_build(bar):
            # Density ramps up over the build.  Build-1 is 8 bars; Build-2 is
            # 16 bars (the escalatory pre-Drop-2 preparation), so the half-way
            # cutoff scales with the build length.
            if _is_build_1(bar):
                build_start, build_len = BREAKDOWN_A_END, 8.0
            else:
                build_start, build_len = BREAKDOWN_B_END, 16.0
            progress = (bar - build_start) / build_len
            density = "8ths" if progress < 0.5 else "soft_16ths"
        elif _is_drop(bar):
            density = "full_16ths"
        elif _is_outro(bar):
            density = "soft_16ths"
        else:
            density = "8ths"
        for beat in range(1, 5):
            for n16 in range(4):
                if density == "8ths" and n16 in (1, 3):
                    continue
                if density == "soft_16ths" and n16 in (1, 3):
                    amp = -18.0
                else:
                    amp = -10.0 if n16 == 0 else -13.5 if n16 == 2 else -16.0
                score.add_note(
                    "hat",
                    start=_pos(bar, beat, n16),
                    duration=0.04,
                    freq=11500.0,
                    amp_db=amp,
                )

    # ── Open hat (sparkle on the off-beat 8ths in drops) ──────────────
    add_drum_voice(
        score,
        "open_hat",
        engine="drum_voice",
        preset="open_hat",
        drum_bus=drum_bus,
        send_db=-4.0,
        mix_db=-16.0,
        choke_group="hats",
        effects=[
            EffectSpec("compressor", {"preset": "hat_control"}),
            _duck("light"),
        ],
    )
    for bar in range(1, TOTAL_BARS + 1):
        if not _is_drop_active(bar):
            continue
        # Every 8th bar, skip the open-hat to break the loop (last bar of
        # each 8-bar phrase — the ear lands on silence, then the next phrase
        # starts fresh).
        drop_start = BUILD_1_END if _is_drop_1(bar) else BUILD_2_END
        if (bar - drop_start) % 8 == 7:
            continue
        # Off-beat 8ths (on the "and" of 1, 2, 3, 4 → beat+n16=2)
        for beat in range(1, 5):
            score.add_note(
                "open_hat",
                start=_pos(bar, beat, 2),
                duration=0.18,
                freq=9000.0,
                amp_db=-10.0,
            )

    # ── Crash on each drop downbeat (trance staple) ───────────────────
    # Derived from the open_hat preset but with the ring extended ~4× so the
    # cymbal rings through the impact rather than choking with the hats.
    # No choke_group, so it survives the hat retriggers during the drop.
    add_drum_voice(
        score,
        "crash",
        engine="drum_voice",
        preset="open_hat",
        drum_bus=drum_bus,
        send_db=-3.0,
        mix_db=-13.0,
        synth_overrides={
            "metallic_decay_s": 1.6,
            "noise_decay_s": 1.2,
            "metallic_brightness": 0.95,
            "metallic_filter_cutoff_hz": 9500.0,
        },
        effects=[EffectSpec("compressor", {"preset": "hat_control"})],
    )
    score.voices["crash"].sends.append(VoiceSend(target="plate", send_db=-4.0))
    for _drop_bar in (BUILD_1_END, BUILD_2_END):
        score.add_note(
            "crash",
            start=_pos(_drop_bar, 1),
            duration=2.5,
            freq=9500.0,
            amp_db=-4.0,
        )

    # ── Shaker (Drop-2 only — adds density without pulling focus) ─────
    # Derived from the closed-hat preset with the metallic partials dialed
    # down (so it reads as pure noise texture, not a second hi-hat) and a
    # tighter decay.  Placed on 16ths with velocity accents on the "&" and
    # "a" of each beat so it moves in its own rhythmic layer under the hat.
    add_drum_voice(
        score,
        "shaker",
        engine="drum_voice",
        preset="chh",
        drum_bus=drum_bus,
        send_db=-8.0,
        mix_db=-18.0,
        synth_overrides={
            "metallic_level": 0.25,
            "noise_level": 0.85,
            "noise_decay_s": 0.04,
            "metallic_decay_s": 0.03,
            "metallic_filter_cutoff_hz": 7200.0,
            "noise_center_ratio": 80.0,
        },
        effects=[
            EffectSpec("compressor", {"preset": "hat_control"}),
            _duck("light"),
        ],
    )
    for bar in range(1, TOTAL_BARS + 1):
        if not _is_drop_2(bar) or _is_cutaway(bar):
            continue
        for beat in range(1, 5):
            for n16 in range(4):
                # Accent offbeats lightly: n16=2 ("&") = +1 dB, n16=3 ("a") = -1
                accent = {0: -3.0, 1: 0.0, 2: 1.0, 3: -1.0}[n16]
                score.add_note(
                    "shaker",
                    start=_pos(bar, beat, n16),
                    duration=0.03,
                    freq=9000.0,
                    amp_db=-19.0 + accent,
                )

    # ── Bass (virus_bass off-beat 8ths) ───────────────────────────────
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "va",
            "preset": "virus_bass",
            "attack": 0.004,
            "decay": 0.14,
            "sustain_level": 0.6,
            "release": 0.10,
        },
        mix_db=-7.0,
        max_polyphony=1,
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living", seed=23),
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog", seed=24),
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 45.0, "slope_db_per_oct": 12},
                        {"kind": "low_shelf", "freq_hz": 120.0, "gain_db": 1.5},
                    ],
                },
            ),
            EffectSpec("preamp", {"preset": "neve_warmth"}),
            _duck(),
        ],
        automation=[
            # Cutoff opens gradually through each drop — keeps the bass fresh
            AutomationSpec(
                target=_synth("cutoff_hz"),
                segments=(
                    _hold(_T_START, _T_D1, 900.0),
                    _seg(_T_D1, _T_BDB, 1500.0, 3000.0, shape="exp"),
                    _hold(_T_BDB, _T_D2, 1200.0),
                    _seg(_T_D2, _T_OUT, 1800.0, 3400.0, shape="exp"),
                    _seg(_T_OUT, _T_END, 3400.0, 900.0, shape="exp"),
                ),
            ),
            AutomationSpec(
                target=_synth("resonance_q"),
                segments=(
                    _seg(_T_D1, _T_BDB, 1.5, 1.9),
                    _seg(_T_D2, _T_OUT, 1.5, 2.1),
                ),
            ),
            # Section-level mix_db arc — lifts the Drop-2b slam-back by
            # ~1.5 dB over Drop-1 so it feels like the biggest-bass moment.
            # Cutaway drone sits slightly under so the drone doesn't
            # overwhelm the foreground lead/bells conversation.
            AutomationSpec(
                target=_ctrl("mix_db"),
                segments=(
                    _hold(_T_D1, _T_BDB, -7.0),
                    _hold(_T_D2, _T_CUTAWAY, -7.0),
                    _hold(_T_CUTAWAY, _T_CUTAWAY_END, -8.0),  # drone ducks
                    _hold(_T_CUTAWAY_END, _T_OUT, -5.5),  # slam-back lift
                ),
            ),
        ],
    )
    for bar in range(1, TOTAL_BARS + 1):
        # Bass drops in late in BUILD-1 (last 2 bars), full in drops,
        # silent in breakdowns and outro tail.  Cutaway bass is sustained
        # and placed separately below.
        if bar < BUILD_1_END - 2:
            continue
        if _is_breakdown_b(bar):
            continue
        if _is_cutaway(bar):
            continue
        if bar >= DROP_2_END + 4:
            continue
        roots = BASS_ROOTS_7LIM if _is_drop_2(bar) else BASS_ROOTS
        # Walking-bass fill on each Eb bar in Drop-2 (chord index 3) — one
        # bar of stepwise descent instead of the off-beat 8ths, leading the
        # ear into the next Fm downbeat.  Drop-1 keeps the straight pattern
        # so the Drop-2 walking figure lands as a new-to-Drop-2 variation.
        if _is_drop_2(bar) and _ci(bar) == 3:
            # Eb3 → Bb2 → C3 → F3 (leading tone into Fm's F2 root).
            # Partials relative to F1: Eb3=18/5, Bb2=27/10, C3=3, F3=4.
            walking = [(1, 18 / 5), (2, 27 / 10), (3, 3.0), (4, 4.0)]
            for beat, partial in walking:
                amp = -6.0 if beat in (1, 3) else -7.5
                score.add_note(
                    "bass",
                    start=_pos(bar, beat, 2),
                    duration=S16 * 3.2,  # longer sustain — this is a line, not stabs
                    partial=partial,
                    amp_db=amp,
                )
            continue
        root = roots[_ci(bar)]
        # Off-beat 8th pattern: beats 1+, 2+, 3+, 4+ with root / octave up
        # First 3 sixteenths after the beat are silent (kick space), hits on
        # the "&" of each beat — the classic trance off-beat bass.
        for beat in range(1, 5):
            partial = root if beat % 2 == 1 else root * 2.0
            amp = -6.0 if beat in (1, 3) else -8.0
            score.add_note(
                "bass",
                start=_pos(bar, beat, 2),
                duration=S16 * 1.8,
                partial=partial,
                amp_db=amp,
            )

    # ── Cutaway bass: sustained Ab drone through bars 97-104 ──────────
    # Bass is mono (max_polyphony=1) so the 5th above sits in the pad instead.
    # Holds Ab2 as a root drone — one long note per 2-bar phrase with amp
    # variance so it breathes.  Ab2 = 12/5 partial of F1.
    for _sub_bar, _amp in (
        (CUTAWAY_START, -6.0),
        (CUTAWAY_START + 2, -5.0),
        (CUTAWAY_START + 4, -5.0),
        (CUTAWAY_START + 6, -7.0),
    ):
        score.add_note(
            "bass",
            start=_pos(_sub_bar),
            duration=BAR * 1.95,
            partial=12 / 5,  # Ab2 root
            amp_db=_amp,
        )

    # ── Acid bass (octave-up bright doubling, Felix/Ladyhawke move) ───
    # Mirrors the off-beat 8th rhythm an octave up, run through a resonant
    # k35 filter for acid bite. Kick-ducked hard so it clears the kick.
    # Absent in BUILD-1 (tease), light in DROP-1, louder and more open in
    # DROP-2 — the "oh, there's a second layer" reveal.
    score.add_voice(
        "acid_bass",
        synth_defaults={
            "engine": "va",
            "preset": "virus_bass",
            "filter_topology": "k35",
            "k35_feedback_asymmetry": 0.05,
            "drive_amount": 0.0,
            "attack": 0.003,
            "decay": 0.10,
            "sustain_level": 0.45,
            "release": 0.06,
            "filter_env_amount": 0.6,
        },
        mix_db=-14.0,
        max_polyphony=1,
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living", seed=73),
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog", seed=74),
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 180.0,
                            "slope_db_per_oct": 12,
                        },
                        {"kind": "high_shelf", "freq_hz": 4000.0, "gain_db": -2.0},
                    ],
                },
            ),
            _duck(),
        ],
        sends=[VoiceSend(target="delay", send_db=-20.0)],
        automation=[
            # Resonant sweep opening across each drop — the signature acid move
            AutomationSpec(
                target=_synth("cutoff_hz"),
                segments=(
                    _seg(_T_D1, _T_BDB, 1400.0, 2600.0, shape="exp"),
                    _seg(_T_D2, _T_OUT, 1800.0, 3400.0, shape="exp"),
                ),
            ),
            AutomationSpec(
                target=_synth("resonance_q"),
                segments=(
                    _seg(_T_D1, _T_BDB, 1.1, 1.4),
                    _seg(_T_D2, _T_OUT, 1.3, 1.6),
                ),
            ),
        ],
    )
    for bar in range(1, TOTAL_BARS + 1):
        # Joins properly in DROP-1, absent in breakdowns and cutaway,
        # louder in DROP-2.
        if not _is_drop_active(bar):
            continue
        roots = BASS_ROOTS_7LIM if _is_drop_2(bar) else BASS_ROOTS
        root = roots[_ci(bar)]
        drop_2 = _is_drop_2(bar)
        for beat in range(1, 5):
            # Octave above the main bass — partial × 2, preserve octave alternation
            base_partial = root if beat % 2 == 1 else root * 2.0
            partial = base_partial * 2.0
            # DROP-2 is the payoff: louder + slightly tighter gate
            amp = (
                (-9.0 if beat in (1, 3) else -11.0)
                if not drop_2
                else (-6.0 if beat in (1, 3) else -8.0)
            )
            score.add_note(
                "acid_bass",
                start=_pos(bar, beat, 2),
                duration=S16 * 1.4,
                partial=partial,
                amp_db=amp,
            )

    # ── Riser + impact into DROP-1 (and a shorter version into DROP-2) ─
    # The signature trance "is-this-going-to-happen-OH-YES-IT'S-HAPPENING"
    # element. 4-bar bandpass-noise sweep rising in amplitude and cutoff,
    # released into a single reverb-washed impact on the drop's downbeat.
    score.add_voice(
        "riser",
        synth_defaults={
            "engine": "synth_voice",
            "osc_type": None,
            "partials_type": None,
            "fm_type": None,
            "noise_type": "bandpass",
            "noise_level": 1.0,
            "filter_mode": "bandpass",
            "filter_cutoff_hz": 1000.0,
            "filter_q": 2.2,
            "attack": 0.01,
            "decay": 0.0,
            "sustain_level": 1.0,
            "release": 0.05,
        },
        mix_db=-14.0,
        velocity_humanize=None,
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 400.0,
                            "slope_db_per_oct": 12,
                        },
                    ],
                },
            ),
        ],
        sends=[
            VoiceSend(target="plate", send_db=-6.0),
            VoiceSend(target="delay", send_db=-14.0),
        ],
    )

    # Impact voice — one reverb-bombed hit per drop entry.
    score.add_voice(
        "impact",
        synth_defaults={
            "engine": "drum_voice",
            "preset": "909_fat",
        },
        normalize_peak_db=-6.0,
        mix_db=-4.0,
        velocity_humanize=None,
        effects=[
            EffectSpec("compressor", {"preset": "snare_punch", "attack_ms": 2.0}),
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "bell", "freq_hz": 160.0, "gain_db": 3.0, "q": 0.9},
                        {"kind": "high_shelf", "freq_hz": 6000.0, "gain_db": -2.0},
                    ]
                },
            ),
        ],
        sends=[
            VoiceSend(target="plate", send_db=-2.0),
            VoiceSend(target="delay", send_db=-10.0),
        ],
    )

    def _place_riser(start_bar: int, bars: int, peak_amp_db: float = -6.0) -> None:
        """Place a *bars*-bar noise sweep terminating at *start_bar + bars*."""
        start_t = _pos(start_bar)
        duration = bars * BAR
        # Amp swell: ease-in from silent to peak over the whole span
        amp_env = [
            {"time": 0.0, "value": 0.02},
            {"time": 0.55, "value": 0.35, "curve": "exponential"},
            {"time": 0.90, "value": 0.95, "curve": "exponential"},
            {"time": 1.0, "value": 1.0, "curve": "linear"},
        ]
        # Cutoff sweep: 800 Hz → 9 kHz, exponential
        filt_env = [
            {"time": 0.0, "value": 800.0},
            {"time": 1.0, "value": 9000.0, "curve": "exponential"},
        ]
        score.add_note(
            "riser",
            start=start_t,
            duration=duration,
            freq=1000.0,
            amp_db=peak_amp_db,
            synth={"noise_envelope": amp_env, "filter_envelope": filt_env},
        )

    def _place_reverse_swell(
        start_bar: int, bars: int, peak_amp_db: float = -8.0
    ) -> None:
        """Reverse sweep: peaks at the start, decays to silence at *start_bar+bars*.

        Cutoff drops from 9 kHz → 800 Hz so the "air" closes in as the sweep
        ends.  Designed to glue a drop → breakdown transition: the sweep
        crescendos backward into the downbeat of the new (sparse) section,
        then falls away, handing the ear off to the bells or pad.
        """
        start_t = _pos(start_bar)
        duration = bars * BAR
        amp_env = [
            {"time": 0.0, "value": 1.0},
            {"time": 0.10, "value": 0.95, "curve": "linear"},
            {"time": 0.45, "value": 0.40, "curve": "exponential"},
            {"time": 1.0, "value": 0.02, "curve": "exponential"},
        ]
        filt_env = [
            {"time": 0.0, "value": 9000.0},
            {"time": 1.0, "value": 800.0, "curve": "exponential"},
        ]
        score.add_note(
            "riser",
            start=start_t,
            duration=duration,
            freq=1000.0,
            amp_db=peak_amp_db,
            synth={"noise_envelope": amp_env, "filter_envelope": filt_env},
        )

    def _place_phrase_sweep(
        start_bar: int, bars: int, peak_amp_db: float = -14.0
    ) -> None:
        """Short 1-bar HPF-style sweep to punctuate phrase boundaries in drops.

        Swells up in the first half, tucks back in the second — sits *under*
        the drop groove rather than crowding it.  Use at phrase boundaries
        inside long drops where chord-cycle churn alone isn't enough motion.
        """
        start_t = _pos(start_bar)
        duration = bars * BAR
        amp_env = [
            {"time": 0.0, "value": 0.02},
            {"time": 0.45, "value": 0.85, "curve": "exponential"},
            {"time": 0.65, "value": 1.0, "curve": "linear"},
            {"time": 1.0, "value": 0.05, "curve": "exponential"},
        ]
        filt_env = [
            {"time": 0.0, "value": 400.0},
            {"time": 1.0, "value": 6000.0, "curve": "exponential"},
        ]
        score.add_note(
            "riser",
            start=start_t,
            duration=duration,
            freq=1000.0,
            amp_db=peak_amp_db,
            synth={"noise_envelope": amp_env, "filter_envelope": filt_env},
        )

    # Full 4-bar riser into DROP-1 (first-reveal — bigger gesture)
    _place_riser(BUILD_1_END - 4, 4, peak_amp_db=-5.0)
    score.add_note(
        "impact",
        start=_pos(BUILD_1_END),
        duration=0.8,
        freq=180.0,
        amp_db=-2.0,
    )

    # 4-bar riser into DROP-2 (extended build-2 escalation).
    # Peaks into the bar-89 impact.
    _place_riser(BUILD_2_END - 4, 4, peak_amp_db=-6.0)
    score.add_note(
        "impact",
        start=_pos(BUILD_2_END),
        duration=0.8,
        freq=180.0,
        amp_db=-3.0,
    )

    # Reverse cymbal sweep at bar 82 — mid-Build-2 punctuation.  Similar to
    # _place_reverse_swell but with a brighter spectrum (highpass start
    # at 2500 Hz, sweeping down to 400 Hz) that reads as a cymbal reverse.
    def _place_reverse_cymbal(start_bar: int, bars: int, peak_amp_db: float) -> None:
        start_t = _pos(start_bar)
        duration = bars * BAR
        amp_env = [
            {"time": 0.0, "value": 1.0},
            {"time": 0.25, "value": 0.7, "curve": "linear"},
            {"time": 0.7, "value": 0.25, "curve": "exponential"},
            {"time": 1.0, "value": 0.02, "curve": "exponential"},
        ]
        filt_env = [
            {"time": 0.0, "value": 7500.0},
            {"time": 1.0, "value": 400.0, "curve": "exponential"},
        ]
        score.add_note(
            "riser",
            start=start_t,
            duration=duration,
            freq=3000.0,
            amp_db=peak_amp_db,
            synth={"noise_envelope": amp_env, "filter_envelope": filt_env},
        )

    _place_reverse_cymbal(BREAKDOWN_B_END + 8, 2, peak_amp_db=-10.0)  # bars 81-82
    # Also one reverse-cymbal gesture bridging the cutaway trough → climb —
    # at bar 101 (the silent trough) to foreshadow the reopen.
    _place_reverse_cymbal(CUTAWAY_START + 4, 2, peak_amp_db=-14.0)  # bars 101-102

    # Reverse swell into BREAKDOWN-B — the drums drop out at bar 57, and
    # this sweep crescendos backward into that silence, then falls away.
    # Sits under the clap roll in bars 55-56.
    _place_reverse_swell(DROP_1_END - 2, 2, peak_amp_db=-8.0)

    # Reverse swell + impact into the CUTAWAY → slam-back transition
    # (bars 103-104 point at the bar-105 downbeat).  Gives the "return"
    # the same dramatic weight as a fresh drop entry.
    _place_reverse_swell(CUTAWAY_END - 2, 2, peak_amp_db=-7.0)
    score.add_note(
        "impact",
        start=_pos(CUTAWAY_END),
        duration=0.8,
        freq=180.0,
        amp_db=-2.0,
    )

    # Phrase-boundary sweeps inside each drop — short 1-bar HPF punctuations
    # every 8 bars so a long drop doesn't just churn on chord loops.
    for _sweep_bar in (
        BUILD_1_END + 7,  # bar 40 — end of first 8-bar phrase of DROP-1
        BUILD_1_END + 15,  # bar 48 — end of second 8-bar phrase of DROP-1
        BUILD_2_END + 7,  # bar 96 — last bar before cutaway (doubles as cue)
        CUTAWAY_END + 7,  # bar 112 — mid Drop-2b
    ):
        _place_phrase_sweep(_sweep_bar, 1, peak_amp_db=-14.0)

    # ── Atmospheric noise wash (intro only) ───────────────────────────
    # Filtered pink-noise drone, plate-drenched, sets the "entering a
    # cathedral" vibe under the sparse intro.  Fades out as Breakdown-A's
    # bells take over the foreground.
    score.add_voice(
        "atmosphere",
        synth_defaults={
            "engine": "synth_voice",
            "osc_type": None,
            "partials_type": None,
            "fm_type": None,
            "noise_type": "pink",
            "noise_level": 1.0,
            "filter_mode": "lowpass",
            "filter_cutoff_hz": 2800.0,
            "filter_q": 0.5,
            "attack": 2.5,
            "decay": 0.0,
            "sustain_level": 1.0,
            "release": 6.0,  # long tail — bleeds into Breakdown-A
        },
        mix_db=-22.0,
        velocity_humanize=None,
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 120.0,
                            "slope_db_per_oct": 12,
                        },
                        {"kind": "low_shelf", "freq_hz": 250.0, "gain_db": 3.0},
                        {"kind": "high_shelf", "freq_hz": 4000.0, "gain_db": -4.0},
                    ],
                },
            ),
        ],
        sends=[
            VoiceSend(target="plate", send_db=-2.0),
            VoiceSend(target="delay", send_db=-10.0),
        ],
        automation=[
            # Gentle swell-in + ease-back so the wash doesn't peak right
            # before the bells start stating the theme — it crests mid-intro
            # (bar 4-ish) and eases under by bar 8.
            AutomationSpec(
                target=_ctrl("mix_db"),
                segments=(
                    _seg(_T_START, _pos(4), -24.0, -20.0),
                    _hold(_pos(4), _pos(7), -20.0),
                    _seg(_pos(7), _T_BDA, -20.0, -26.0),
                ),
            ),
        ],
    )
    # One long note spanning the full intro, fading by Breakdown-A
    score.add_note(
        "atmosphere",
        start=_pos(1),
        duration=(INTRO_END - 1) * BAR + BAR * 0.5,
        freq=800.0,
        amp_db=-14.0,  # -6 dB vs prior -8 dB
    )

    # ── Intro bell arpeggio (foreshadows Breakdown-A theme) ───────────
    # Sparse bell tones on bars 5-8 outlining the F-minor triad so the
    # ear subconsciously recognizes the theme when Breakdown-A states it.
    # Note added via score.add_note against the 'bells' voice defined
    # later in this function — forward reference is fine since all notes
    # resolve at render time.

    # ── Pad (supersaw_pad sustained chords) ───────────────────────────
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "va",
            "preset": "supersaw_pad",
            "attack": 1.8,
            "decay": 0.8,
            "sustain_level": 0.85,
            "release": 2.8,
        },
        mix_db=-10.0,
        velocity_group="trance_ensemble",
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad", seed=30),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living", seed=31),
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
                    ],
                },
            ),
            _duck("pad"),
        ],
        sends=[
            VoiceSend(target="pad_chorus", send_db=-4.0),
            VoiceSend(
                target="plate",
                send_db=-10.0,
                automation=[
                    AutomationSpec(
                        target=_ctrl("send_db"),
                        segments=(
                            _seg(_T_START, _T_BDA, -10.0, -6.0),
                            _seg(_T_BDA, _T_B1, -6.0, -5.0),
                            _seg(_T_B1, _T_D1, -8.0, -11.0),
                            _hold(_T_D1, _T_BDB, -12.0),
                            _seg(_T_BDB, _T_B2, -5.0, -4.0),
                            _seg(_T_B2, _T_D2, -8.0, -11.0),
                            _hold(_T_D2, _T_OUT, -12.0),
                            # Outro washout: wet rides UP as the pad itself
                            # fades, producing a reverb-tail cloud rather
                            # than a clean peel.
                            _seg(_T_OUT, _T_END, -8.0, 0.0),
                        ),
                    ),
                ],
            ),
        ],
        automation=[
            # The signature trance filter sweep on the pad
            AutomationSpec(
                target=_synth("cutoff_hz"),
                segments=(
                    _seg(_T_START, _T_BDA, 300.0, 500.0, shape="exp"),
                    _seg(_T_BDA, _T_B1, 500.0, 1200.0, shape="exp"),
                    _seg(_T_B1, _T_D1, 1200.0, 5500.0, shape="exp"),  # build sweep
                    _seg(_T_D1, _T_BDB, 3500.0, 5000.0, shape="exp"),
                    _seg(_T_BDB, _T_B2, 1800.0, 1500.0, shape="exp"),
                    _seg(_T_B2, _T_D2, 1500.0, 6000.0, shape="exp"),  # build 2
                    # Drop-2a: open continuously up to the cutaway
                    _seg(_T_D2, _T_CUTAWAY, 4000.0, 5500.0, shape="exp"),
                    # Cutaway: close across the first 4 bars, reopen
                    # across the last 4 — the ear follows a breathing arc,
                    # not a static chord.
                    _seg(_T_CUTAWAY, _T_CUTAWAY_MID, 5500.0, 1400.0, shape="exp"),
                    _seg(_T_CUTAWAY_MID, _T_CUTAWAY_END, 1400.0, 6500.0, shape="exp"),
                    # Drop-2b: holds open for the slam-back
                    _seg(_T_CUTAWAY_END, _T_OUT, 6500.0, 5500.0, shape="exp"),
                    _seg(_T_OUT, _T_END, 5000.0, 400.0, shape="exp"),
                ),
            ),
            AutomationSpec(
                target=_synth("resonance_q"),
                segments=(
                    _hold(_T_START, _T_BDA, 0.9),
                    _seg(_T_BDA, _T_B1, 0.9, 1.2),
                    _seg(_T_B1, _T_D1, 1.2, 1.9),  # zing during build
                    _hold(_T_D1, _T_BDB, 0.9),
                    _seg(_T_BDB, _T_B2, 1.0, 1.2),
                    _seg(_T_B2, _T_D2, 1.2, 2.0),
                    _hold(_T_D2, _T_CUTAWAY, 0.9),
                    # Resonance spikes at the cutaway trough for extra zing
                    # as the filter closes, then eases back for the reopen.
                    _seg(_T_CUTAWAY, _T_CUTAWAY_MID, 0.9, 1.6),
                    _seg(_T_CUTAWAY_MID, _T_CUTAWAY_END, 1.6, 0.9),
                    _hold(_T_CUTAWAY_END, _T_OUT, 0.9),
                    _seg(_T_OUT, _T_END, 0.9, 0.6),
                ),
            ),
            # Slow stereo drift — supersaw pads can feel locked dead-centre
            # without a little motion. ~16 s period, ±0.06 depth.
            AutomationSpec(
                target=_ctrl("pan"),
                segments=(
                    AutomationSegment(
                        start=_T_BDA,
                        end=_T_END,
                        shape="sine_lfo",
                        freq_hz=1.0 / 16.0,
                        depth=0.06,
                    ),
                ),
            ),
            # Tempo-gated 1/8-note tremolo across BREAKDOWN-B — the campy
            # Cascada-waiting-for-the-drop pulse shelters under the lead.
            # 4.6 Hz = eighth notes at 138 BPM. Peaks land on beats (phase_rad
            # = π/2), trough on the "&" — the pad pulses with the beat.
            AutomationSpec(
                target=_ctrl("mix_db"),
                segments=(
                    AutomationSegment(
                        start=_T_BDB,
                        end=_T_B2,
                        shape="sine_lfo",
                        freq_hz=BPM / 60.0 * 2.0,
                        depth=5.0,
                        offset=-13.0,
                        phase_rad=1.5707963267948966,  # π/2 — start at peak
                    ),
                ),
            ),
            # Section-level loudness arc (outside BREAKDOWN-B where the
            # tremolo takes over).  Keeps the pad from feeling flat across
            # the 3:42 runtime — subtle swells into drops, gentle ride down
            # through the outro washout.
            AutomationSpec(
                target=_ctrl("mix_db"),
                segments=(
                    _seg(_T_START, _T_BDA, -14.0, -11.0),  # intro swell in
                    _hold(_T_BDA, _T_B1, -11.0),  # BDA — bells take focus
                    _seg(_T_B1, _T_D1, -11.0, -9.0),  # build-1 crescendo
                    _hold(_T_D1, _T_BDB, -10.0),  # drop-1
                    # _T_BDB → _T_B2 handled by tremolo above
                    _seg(_T_B2, _T_D2, -11.0, -9.0),  # build-2 crescendo
                    _hold(_T_D2, _T_CUTAWAY, -10.0),  # drop-2a
                    _hold(_T_CUTAWAY, _T_CUTAWAY_END, -9.0),  # cutaway slight lift
                    _hold(_T_CUTAWAY_END, _T_OUT, -10.0),  # drop-2b slam-back
                    _seg(_T_OUT, _T_END, -11.0, -18.0),  # outro taper
                ),
            ),
        ],
    )
    # Dedicated Ab voicing for the cutaway — a single sustained chord across
    # the 8 bars, stacked with the 5th added back since bass drops the Eb.
    # Register still in the Eb3-Eb4 tight band.  Matches the current Ab
    # voice-leading voicing and adds the 4th partial as a 9th (Bb4) for
    # extra shimmer.
    cutaway_ab_voicing: list[float] = [18 / 5, 24 / 5, 6.0, 36 / 5, 27 / 5]
    # Eb3, Ab3, C4, Eb4, Bb3 (9th — color)

    for bar in range(1, TOTAL_BARS + 1):
        # Pad plays continuously from the end of INTRO through OUTRO (minus the
        # last 2 bars where everything fades).
        if bar < 5:  # pad sneaks in bars 5-8
            continue
        if bar >= TOTAL_BARS - 1:
            continue
        if _is_cutaway(bar):
            # Cutaway: one sustained Ab chord re-struck every 2 bars so the
            # breathing pad-chorus envelope restarts and the chord feels
            # alive, not droney.
            if (bar - CUTAWAY_START) % 2 != 0:
                continue
            for i, partial in enumerate(cutaway_ab_voicing):
                score.add_note(
                    "pad",
                    start=_pos(bar),
                    duration=BAR * 1.97,
                    partial=partial,
                    amp_db=-5.0 - i * 0.4,
                )
            continue
        chord = PAD_VOICINGS[_ci(bar)]
        amp = -7.0 if bar < INTRO_END else -6.0
        for i, partial in enumerate(chord):
            score.add_note(
                "pad",
                start=_pos(bar),
                duration=BAR * 0.98,
                partial=partial,
                amp_db=amp - i * 0.5,
            )

    # ── Bells (q_comb_bell) ───────────────────────────────────────────
    score.add_voice(
        "bells",
        synth_defaults={
            "engine": "va",
            "preset": "q_comb_bell",
            "attack": 0.004,
            "decay": 0.5,
            "sustain_level": 0.3,
            "release": 1.2,
        },
        mix_db=-12.0,
        velocity_group="trance_ensemble",
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble", seed=40),
        envelope_humanize=EnvelopeHumanizeSpec(preset="loose_pluck", seed=41),
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 400.0,
                            "slope_db_per_oct": 12,
                        },
                        {"kind": "bell", "freq_hz": 2200.0, "gain_db": 1.5, "q": 1.0},
                    ],
                },
            ),
            _duck(),
        ],
        sends=[
            VoiceSend(
                target="delay",
                send_db=-12.0,
                automation=[
                    AutomationSpec(
                        target=_ctrl("send_db"),
                        segments=(
                            _seg(_T_BDA, _T_B1, -8.0, -10.0),
                            _hold(_T_B1, _T_BDB, -16.0),
                            _seg(_T_BDB, _T_B2, -8.0, -7.0),
                            _hold(_T_B2, _T_D2, -14.0),
                            _hold(_T_D2, _T_OUT, -18.0),
                        ),
                    ),
                ],
            ),
            VoiceSend(
                target="plate",
                send_db=-10.0,
                automation=[
                    AutomationSpec(
                        target=_ctrl("send_db"),
                        segments=(
                            _hold(_T_BDA, _T_B1, -6.0),
                            _seg(_T_B1, _T_D1, -8.0, -14.0),
                            _hold(_T_BDB, _T_B2, -5.0),
                            _seg(_T_B2, _T_D2, -8.0, -14.0),
                        ),
                    ),
                ],
            ),
        ],
        automation=[
            # Slow comb feedback swell during BREAKDOWN-B for the emotional peak
            AutomationSpec(
                target=_synth("comb_feedback"),
                segments=(
                    _hold(_T_START, _T_BDB, 0.92),
                    _seg(_T_BDB, _T_B2, 0.88, 0.94),
                    _hold(_T_B2, _T_END, 0.92),
                ),
            ),
            # Section-level mix_db arc.
            #   Intro: -16 (the quiet foreshadow bells)
            #   Breakdown-A: crescendo -14 → -10 as the theme establishes
            #   Build/drops: -12 (sparse accents sit back)
            #   Breakdown-B: peak at -10 (bars 65-68) then ease back
            #   Cutaway: ride with the lead, -10
            #   Outro: taper to -18 as everything fades
            AutomationSpec(
                target=_ctrl("mix_db"),
                segments=(
                    _hold(_T_START, _T_BDA, -16.0),
                    _seg(_T_BDA, _T_B1, -14.0, -10.0),
                    _hold(_T_B1, _T_BDB, -12.0),
                    _seg(_T_BDB, _pos(DROP_1_END + 8), -11.0, -10.0),
                    _hold(_pos(DROP_1_END + 8), _pos(DROP_1_END + 12), -10.0),
                    _seg(_pos(DROP_1_END + 12), _T_B2, -11.0, -12.0),
                    _hold(_T_B2, _T_D2, -12.0),
                    _hold(_T_D2, _T_CUTAWAY, -12.0),
                    _hold(_T_CUTAWAY, _T_CUTAWAY_END, -10.0),
                    _hold(_T_CUTAWAY_END, _T_OUT, -12.0),
                    _seg(_T_OUT, _T_END, -14.0, -18.0),
                ),
            ),
        ],
    )
    # Intro bell arpeggio (foreshadows the Breakdown-A theme).
    # Bars 5-8: sparse Fm triad tones — F5, Ab5, C6 — one per bar, with
    # the theme's melodic shape previewed on bar 8.  Very quiet, plate-
    # drenched, so it reads as a distant recall before the full statement.
    _intro_foreshadow: list[tuple[int, int, int, float, int, float]] = [
        # (bar, beat, n16, partial, gate, velocity)
        (5, 1, 0, _F5, 8, 0.48),  # F5 held, establishes home
        (6, 3, 0, _Ab5, 8, 0.52),  # Ab5 — minor 3rd, the "minor" color
        (7, 1, 0, _C6, 8, 0.50),  # C5 — the 5th
        (8, 1, 0, _F5, 2, 0.55),  # mini theme statement
        (8, 2, 2, _Ab5, 2, 0.50),
    ]
    for _bar, _beat, _n16, _partial, _gate, _vel in _intro_foreshadow:
        score.add_note(
            "bells",
            start=_pos(_bar, _beat, _n16),
            duration=_gate * S16 * 0.85,
            partial=_partial,
            amp_db=-14.0,
            velocity=_vel,
        )

    # Bells state the theme in BREAKDOWN-A
    for bar in range(INTRO_END, BREAKDOWN_A_END):
        motif = BELL_A[_ci(bar)]
        for beat, n16, partial, gate, vel in motif:
            score.add_note(
                "bells",
                start=_pos(bar, beat, n16),
                duration=gate * S16 * 0.85,
                partial=partial,
                amp_db=-7.0,
                velocity=vel,
            )
    # Bells counterline in BREAKDOWN-B (answers the lead on beats 2 & 4)
    for bar in range(DROP_1_END, BREAKDOWN_B_END):
        motif = BELL_B[_ci(bar)]
        for beat, n16, partial, gate, vel in motif:
            score.add_note(
                "bells",
                start=_pos(bar, beat, n16),
                duration=gate * S16 * 0.85,
                partial=partial,
                amp_db=-9.0,
                velocity=vel,
            )
    # Sparse bell accents in DROP-2 (answer the lead once per 4-bar phrase).
    # Cutaway handled separately below.
    for bar in range(BUILD_2_END, DROP_2_END):
        if _is_cutaway(bar):
            continue
        if (bar - BUILD_2_END) % 4 != 3:  # only the 4th bar of each chord
            continue
        motif = BELL_B[_ci(bar)]
        for beat, n16, partial, gate, vel in motif[:2]:
            score.add_note(
                "bells",
                start=_pos(bar, beat, n16),
                duration=gate * S16 * 0.85,
                partial=partial,
                amp_db=-12.0,
                velocity=vel * 0.8,
            )
    # Cutaway bells — call-and-response with the lead (see LEAD_CUTAWAY
    # and BELL_CUTAWAY for the 8-bar phrase shape).
    for _i_bar in range(CUTAWAY_END - CUTAWAY_START):
        bar = CUTAWAY_START + _i_bar
        motif = BELL_CUTAWAY[_i_bar]
        for beat, n16, partial, gate, vel in motif:
            score.add_note(
                "bells",
                start=_pos(bar, beat, n16),
                duration=gate * S16 * 0.85,
                partial=partial,
                amp_db=-10.0,
                velocity=vel,
            )

    # ── Hoover (jp8000_hoover stabs) ──────────────────────────────────
    score.add_voice(
        "hoover",
        synth_defaults={
            "engine": "va",
            "preset": "jp8000_hoover",
            "attack": 0.015,
            "decay": 0.25,
            "sustain_level": 0.55,
            "release": 0.30,
        },
        mix_db=-11.0,
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living", seed=50),
        effects=[
            EffectSpec("preamp", {"preset": "neve_warmth"}),
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 180.0,
                            "slope_db_per_oct": 12,
                        },
                    ],
                },
            ),
            _duck(),
        ],
        sends=[VoiceSend(target="plate", send_db=-10.0)],
    )
    # Hoover teases escalate across BUILD-2 (16 bars).  Four phases,
    # 4 bars each: silence → sparse → medium → dense.
    #   bars 73-76: silent (Breakdown-B continuation)
    #   bars 77-80: single stab on bar 1 of each 4-bar chord block (sparse)
    #   bars 81-84: stabs on beat 1 of every bar (medium)
    #   bars 85-88: stabs on beats 1 and 3 of every bar (dense, driving)
    for bar in range(BREAKDOWN_B_END, BUILD_2_END):
        phase = (bar - BREAKDOWN_B_END) // 4  # 0..3
        if phase == 0:
            continue  # silent phase
        chord = HOOVER_VOICINGS[_ci(bar)]
        if phase == 1:
            # sparse: only first bar of each 4-bar chord block gets a stab
            if (bar - BREAKDOWN_B_END) % 4 != 0:
                continue
            hit_beats = [1]
            amp, vel, dur = -10.0, 0.60, BEAT * 1.5
        elif phase == 2:
            hit_beats = [1]
            amp, vel, dur = -8.0, 0.72, BEAT * 1.5
        else:  # phase 3 — driving
            hit_beats = [1, 3]
            amp, vel, dur = -6.0, 0.82, BEAT * 1.2
        for beat in hit_beats:
            for partial in chord:
                score.add_note(
                    "hoover",
                    start=_pos(bar, beat),
                    duration=dur,
                    partial=partial,
                    amp_db=amp,
                    velocity=vel,
                )
    # Hoover stabs on chord changes in both drops
    for bar in range(BUILD_1_END, DROP_1_END):
        if (bar - BUILD_1_END) % 4 != 0:  # bar 1 of each chord only
            continue
        chord = HOOVER_VOICINGS[_ci(bar)]
        for partial in chord:
            score.add_note(
                "hoover",
                start=_pos(bar, 1),
                duration=BEAT * 2.0,
                partial=partial,
                amp_db=-6.0,
                velocity=0.82,
            )
    for bar in range(BUILD_2_END, DROP_2_END):
        if _is_cutaway(bar):
            continue
        # DROP-2: stabs on bars 1 and 3 of each chord for extra urgency
        offset = (bar - BUILD_2_END) % 4
        if offset not in (0, 2):
            continue
        chord = HOOVER_VOICINGS[_ci(bar)]
        vel = 0.85 if offset == 0 else 0.72
        amp = -6.0 if offset == 0 else -9.0
        for partial in chord:
            score.add_note(
                "hoover",
                start=_pos(bar, 1),
                duration=BEAT * 1.5,
                partial=partial,
                amp_db=amp,
                velocity=vel,
            )
    # Distant hoover stab at bar 103 — a single quiet Ab-chord stab
    # signaling the return.  Plate-drenched, low velocity.
    for partial in HOOVER_VOICINGS[2]:  # Ab voicing
        score.add_note(
            "hoover",
            start=_pos(CUTAWAY_END - 2, 1),
            duration=BEAT * 1.5,
            partial=partial,
            amp_db=-12.0,
            velocity=0.55,
        )

    # ── Lead (virus_lead topline) ─────────────────────────────────────
    dotted_eighth = 3.0 * S16
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "va",
            "preset": "virus_lead",
            "attack": 0.02,
            "decay": 0.18,
            "sustain_level": 0.7,
            "release": 0.22,
        },
        mix_db=-10.0,
        velocity_group="trance_ensemble",
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble", seed=60),
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog", seed=61),
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 180.0,
                            "slope_db_per_oct": 12,
                        },
                    ],
                },
            ),
            EffectSpec(
                "delay",
                # Higher feedback than a typical slap — the last 2 beats of
                # each drop get a mix spike that lets the echoes bloom into
                # the sparse section, simulating a feedback runaway.
                {"delay_seconds": dotted_eighth, "feedback": 0.42, "mix": 0.12},
                automation=[
                    AutomationSpec(
                        target=_ctrl("mix"),
                        segments=(
                            _hold(_T_D1, _pos(DROP_1_END - 1, 3), 0.10),
                            _seg(_pos(DROP_1_END - 1, 3), _T_BDB, 0.10, 0.45),
                            _seg(_T_BDB, _T_B2, 0.18, 0.24),
                            _seg(_T_B2, _T_D2, 0.22, 0.12),
                            _hold(_T_D2, _pos(DROP_2_END - 1, 3), 0.10),
                            _seg(_pos(DROP_2_END - 1, 3), _T_OUT, 0.10, 0.45),
                            _seg(_T_OUT, _T_END, 0.14, 0.28),
                        ),
                    ),
                ],
            ),
            _duck(),
        ],
        sends=[
            VoiceSend(
                target="plate",
                send_db=-14.0,
                automation=[
                    AutomationSpec(
                        target=_ctrl("send_db"),
                        segments=(
                            _hold(_T_D1, _T_BDB, -16.0),
                            _seg(_T_BDB, _T_B2, -6.0, -5.0),
                            _hold(_T_B2, _T_D2, -10.0),
                            _hold(_T_D2, _T_OUT, -12.0),
                            # Outro washout: ride the wet up as dry fades.
                            _seg(_T_OUT, _T_END, -10.0, -2.0),
                        ),
                    ),
                ],
            ),
            VoiceSend(target="delay", send_db=-16.0),
        ],
        automation=[
            AutomationSpec(
                target=_synth("cutoff_hz"),
                clamp_min=800.0,
                segments=(
                    _hold(_T_D1, _T_BDB, 2800.0),
                    _seg(_T_BDB, _T_B2, 1800.0, 2400.0, shape="exp"),
                    _seg(_T_B2, _T_D2, 2400.0, 4500.0, shape="exp"),
                    _hold(_T_D2, _T_OUT, 3500.0),
                ),
            ),
            # Section-level loudness arc.  Drop-1 baseline, Breakdown-B peaks
            # (+2 dB, the emotional moment), Drop-2a sits slightly under
            # Drop-1, cutaway rides up from quiet to loud across its 8 bars,
            # Drop-2b is the slam-back peak (+2 dB over Drop-1).
            AutomationSpec(
                target=_ctrl("mix_db"),
                segments=(
                    _hold(_T_D1, _T_BDB, -10.0),
                    # Breakdown-B rises into the peak-4-bars (bars 65-68)
                    # then eases back toward Build-2
                    _seg(_T_BDB, _pos(DROP_1_END + 8), -11.0, -8.0),
                    _hold(_pos(DROP_1_END + 8), _pos(DROP_1_END + 12), -8.0),
                    _seg(_pos(DROP_1_END + 12), _T_B2, -9.0, -11.0),
                    # Build-2: lead silent but keep the bed ready
                    _hold(_T_B2, _T_D2, -11.0),
                    # Drop-2a: slightly under Drop-1 so the slam-back lifts
                    _hold(_T_D2, _T_CUTAWAY, -11.0),
                    # Cutaway: starts quiet, climbs as the phrase reaches up
                    _seg(_T_CUTAWAY, _T_CUTAWAY_MID, -9.0, -9.0),
                    _seg(_T_CUTAWAY_MID, _T_CUTAWAY_END, -9.0, -6.0),
                    # Drop-2b slam-back: peak loudness
                    _hold(_T_CUTAWAY_END, _T_OUT, -8.0),
                ),
            ),
        ],
    )
    # Lead plays the topline in both drops and the emotional peak (BREAKDOWN-B)
    for bar in range(BUILD_1_END, DROP_1_END):
        motif = LEAD_DROP[_ci(bar)]
        for beat, n16, partial, gate, vel in motif:
            score.add_note(
                "lead",
                start=_pos(bar, beat, n16),
                duration=gate * S16 * 0.85,
                partial=partial,
                amp_db=-6.0,
                velocity=vel,
            )
    # BREAKDOWN-B: lyrical held notes with vibrato
    # Second half (bars 65-68) gets an octave-down unison doubling — fattens
    # the money moment without changing the melody. Only the peak four bars
    # so the lift from "single-voice introduction" to "doubled climax" is felt.
    peak_start = DROP_1_END + 8  # bar 65
    peak_end = DROP_1_END + 12  # bar 69 (exclusive)
    for bar in range(DROP_1_END, BREAKDOWN_B_END):
        motif = LEAD_BD[_ci(bar)]
        in_peak = peak_start <= bar < peak_end
        for beat, n16, partial, gate, vel in motif:
            score.add_note(
                "lead",
                start=_pos(bar, beat, n16),
                duration=gate * S16 * 0.85,
                partial=partial,
                amp_db=-7.0,
                velocity=vel,
                pitch_motion=PitchMotionSpec.vibrato(
                    depth_ratio=0.005,
                    rate_hz=5.2,
                ),
            )
            if in_peak:
                # Octave-down unison — quieter and less vibrato depth so the
                # top note remains the focus.
                score.add_note(
                    "lead",
                    start=_pos(bar, beat, n16),
                    duration=gate * S16 * 0.85,
                    partial=partial * 0.5,
                    amp_db=-11.0,
                    velocity=vel * 0.85,
                    pitch_motion=PitchMotionSpec.vibrato(
                        depth_ratio=0.003,
                        rate_hz=5.2,
                    ),
                )
    # DROP-2: same topline, more expressive with light vibrato on longer notes.
    # Cutaway is handled separately below.  Climactic Eb6 on the last beat of
    # each Eb bar gets a G6 harmony (major 3rd above) — the "money chord" lift.
    for bar in range(BUILD_2_END, DROP_2_END):
        if _is_cutaway(bar):
            continue
        motif = LEAD_DROP[_ci(bar)]
        is_eb_bar = _ci(bar) == 3  # chord index 3 = Eb
        for beat, n16, partial, gate, vel in motif:
            pm = None
            if gate >= 4:
                pm = PitchMotionSpec.vibrato(depth_ratio=0.004, rate_hz=5.8)
            score.add_note(
                "lead",
                start=_pos(bar, beat, n16),
                duration=gate * S16 * 0.85,
                partial=partial,
                amp_db=-5.0,
                velocity=vel * 1.02,
                pitch_motion=pm,
            )
            # Drop-2 harmonization: double the climactic Eb6 on beat 4 of
            # each Eb bar with a 3rd above (G6, partial 18 = 9/2 × 4).
            # G6 is 9/2 × 4 = 18.  Quieter than the top note so Eb6 stays
            # on top but the 3rd fattens the lift into the next Fm.
            if is_eb_bar and beat == 4 and n16 == 0 and partial == _Eb6:
                score.add_note(
                    "lead",
                    start=_pos(bar, beat, n16),
                    duration=gate * S16 * 0.85,
                    partial=18.0,  # G6 — major 3rd above Eb6
                    amp_db=-9.0,
                    velocity=vel * 0.85,
                    pitch_motion=pm,
                )

    # Cutaway lead: 8-bar phrase on Ab (bars 97-104), call-and-response with
    # bells.  Vibrato on any note ≥ 6 sixteenths so the long sustains breathe.
    for _i_bar in range(CUTAWAY_END - CUTAWAY_START):
        bar = CUTAWAY_START + _i_bar
        motif = LEAD_CUTAWAY[_i_bar]
        for beat, n16, partial, gate, vel in motif:
            pm = None
            if gate >= 6:
                pm = PitchMotionSpec.vibrato(depth_ratio=0.005, rate_hz=5.4)
            score.add_note(
                "lead",
                start=_pos(bar, beat, n16),
                duration=gate * S16 * 0.85,
                partial=partial,
                amp_db=-6.0,
                velocity=vel,
                pitch_motion=pm,
            )

    # ── Fills on beat 4 of the last bar before each major transition ──
    # Snare flams telegraph incoming drops; clap rolls crescendo into
    # breakdowns. Keeps section edges from feeling abrupt.
    drop_entry_fills = [BUILD_1_END - 1, BUILD_2_END - 1]  # bars 32, 80
    for fill_bar in drop_entry_fills:
        # Ghost 16th on beat 4 "e"
        score.add_note(
            "snare",
            start=_pos(fill_bar, 4, 1),
            duration=0.12,
            freq=200.0,
            amp_db=-14.0,
        )
        # Main flam on beat 4 "and"
        score.add_note(
            "snare",
            start=_pos(fill_bar, 4, 2),
            duration=0.18,
            freq=200.0,
            amp_db=-6.0,
        )
        # Accent on beat 4 "a" — final push before the drop hangs
        score.add_note(
            "snare",
            start=_pos(fill_bar, 4, 3),
            duration=0.16,
            freq=220.0,
            amp_db=-4.0,
        )

    breakdown_entry_fills = [DROP_1_END - 1, DROP_2_END - 1]  # bars 56, 120
    for fill_bar in breakdown_entry_fills:
        for n16 in range(4):
            score.add_note(
                "clap",
                start=_pos(fill_bar, 4, n16),
                duration=0.10,
                freq=3000.0,
                amp_db=-10.0 + n16 * 1.5,
            )

    # Ghost snare build in the last 4 bars of BUILD-2 (bars 85-88).
    # One hit per bar on beat 4 "e", velocity climbing, except bar 88
    # which already has the full fill above.
    for _i, _bar in enumerate(range(BUILD_2_END - 4, BUILD_2_END - 1)):
        score.add_note(
            "snare",
            start=_pos(_bar, 4, 1),
            duration=0.10,
            freq=200.0,
            amp_db=-18.0 + _i * 2.5,  # -18 → -15.5 → -13
            velocity=0.45 + _i * 0.08,
        )

    # Gentle snare roll into the cutaway (bar 96, under the kick break).
    # Softer than the breakdown roll — a suggestion, not a hit.
    for n16 in range(4):
        score.add_note(
            "snare",
            start=_pos(CUTAWAY_START - 1, 4, n16),
            duration=0.10,
            freq=200.0,
            amp_db=-16.0 + n16 * 1.0,
            velocity=0.40 + n16 * 0.05,
        )

    return score


PIECES: dict[str, PieceDefinition] = {
    "va_trance": PieceDefinition(
        name="va_trance",
        output_name="va_trance",
        build_score=build_score,
        sections=(
            PieceSection("intro", _T_START, _T_BDA),
            PieceSection("breakdown_a", _T_BDA, _T_B1),
            PieceSection("build_1", _T_B1, _T_D1),
            PieceSection("drop_1", _T_D1, _T_BDB),
            PieceSection("breakdown_b", _T_BDB, _T_B2),
            PieceSection("build_2", _T_B2, _T_D2),
            PieceSection("drop_2a", _T_D2, _T_CUTAWAY),
            PieceSection("cutaway", _T_CUTAWAY, _T_CUTAWAY_END),
            PieceSection("drop_2b", _T_CUTAWAY_END, _T_OUT),
            PieceSection("outro", _T_OUT, _T_END),
        ),
    ),
}
