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

Form (112 bars, ~3:15 at 138 BPM):
    bars   1-8    INTRO        kick + filtered hat + pad bed
    bars   9-24   BREAKDOWN-A  bells state theme over pad, no drums
    bars  25-32   BUILD-1      drums return, long filter sweep, risers
    bars  33-56   DROP-1       full arrangement with topline lead
    bars  57-72   BREAKDOWN-B  emotional peak: pad + bells + lead, no drums
    bars  73-80   BUILD-2      hoover stabs tease the return
    bars  81-104  DROP-2       full arrangement, expressive lead, counterline
    bars 105-112  OUTRO        elements peel away to tonic
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
TOTAL_BARS: int = 112

# ── Section boundaries (1-indexed bars, exclusive upper bound) ────────
INTRO_END = 9  # bars 1-8
BREAKDOWN_A_END = 25  # bars 9-24
BUILD_1_END = 33  # bars 25-32
DROP_1_END = 57  # bars 33-56
BREAKDOWN_B_END = 73  # bars 57-72
BUILD_2_END = 81  # bars 73-80
DROP_2_END = 105  # bars 81-104
OUTRO_END = TOTAL_BARS + 1  # bars 105-112

# ── Chord progression: Fm – Db – Ab – Eb ──────────────────────────────
# Ratios relative to F (f0 = F1).  Each chord lasts 4 bars.
# Root, minor third, fifth — then octave-doubled for pad richness.
#   Fm:  1, 6/5, 3/2           (F, Ab, C)
#   Db:  8/5, 2, 12/5          (Db, F, Ab)
#   Ab:  6/5, 3/2, 9/5         (Ab, C, Eb)
#   Eb:  9/5, 9/4, 27/10       (Eb, G, Bb)   — major triad on ♭VII
PAD_VOICINGS: list[list[float]] = [
    # Pad plays in partial range ~4-13; open voicings centered on F4-F5.
    [4.0, 24 / 5, 6.0, 12.0],  # Fm: F3, Ab3, C4, F5
    [32 / 5, 8.0, 48 / 5, 64 / 5],  # Db: Db4, F4, Ab4, Db5
    [24 / 5, 6.0, 36 / 5, 48 / 5],  # Ab: Ab3, C4, Eb4, Ab4
    [36 / 5, 9.0, 54 / 5, 72 / 5],  # Eb: Eb4, G4, Bb4, Eb5
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
    (1, 0, _F5, 8, 0.72),
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
    return BUILD_2_END <= bar < DROP_2_END


def _is_drop(bar: int) -> bool:
    return _is_drop_1(bar) or _is_drop_2(bar)


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


def _duck(strength: str = "kick_duck") -> EffectSpec:
    return EffectSpec("compressor", {"preset": strength, "sidechain_source": "kick"})


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
        effects=[
            EffectSpec(
                "compressor",
                {
                    "threshold_db": -20.0,
                    "ratio": 2.5,
                    "attack_ms": 8.0,
                    "release_ms": 100.0,
                    "knee_db": 4.0,
                    "makeup_gain_db": 1.5,
                },
            ),
        ],
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
        # Kick present in intro (thin), both drops, both builds.  Silent in
        # breakdowns and outro tail.
        if _is_breakdown_a(bar) or _is_breakdown_b(bar):
            continue
        if bar >= DROP_2_END + 6:  # last 2 bars of outro
            continue
        for beat in range(1, 5):
            amp = -8.0 if bar < INTRO_END else -4.0
            score.add_note(
                "kick",
                start=_pos(bar, beat),
                duration=0.6,
                freq=55.0,
                amp_db=amp,
            )

    # ── Clap (beats 2 & 4, drops only) ────────────────────────────────
    add_drum_voice(
        score,
        "clap",
        engine="drum_voice",
        preset="909_clap",
        drum_bus=drum_bus,
        send_db=-2.0,
        mix_db=-7.0,
        effects=[],  # keep it dry-ish inside the drum bus
    )
    # Plate send for the clap directly (not via drum bus)
    score.voices["clap"].sends.append(VoiceSend(target="plate", send_db=-12.0))
    for bar in range(1, TOTAL_BARS + 1):
        if not _is_drop(bar):
            continue
        for beat in [2, 4]:
            score.add_note(
                "clap",
                start=_pos(bar, beat),
                duration=0.12,
                freq=3000.0,
                amp_db=-3.0,
            )

    # ── Hats (16ths with density automation) ──────────────────────────
    add_drum_voice(
        score,
        "hat",
        engine="drum_voice",
        preset="chh",
        drum_bus=drum_bus,
        send_db=-6.0,
        mix_db=-11.0,
        effects=[
            EffectSpec(
                "eq",
                {"bands": [{"kind": "high_shelf", "freq_hz": 8500.0, "gain_db": 2.0}]},
            ),
            EffectSpec("compressor", {"preset": "hat_control"}),
            _duck("kick_duck"),
        ],
    )
    for bar in range(1, TOTAL_BARS + 1):
        # Hat gating: absent in breakdowns, 8ths in intro/outro, 16ths in drops.
        if _is_breakdown_a(bar) or _is_breakdown_b(bar):
            continue
        if bar >= DROP_2_END + 6:
            continue
        if bar < INTRO_END:
            density = "8ths"
        elif _is_build(bar):
            # density ramps up over the build
            build_start = BREAKDOWN_A_END if _is_build_1(bar) else BREAKDOWN_B_END
            progress = (bar - build_start) / 8.0
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
        mix_db=-14.0,
        choke_group="hats",
        effects=[
            EffectSpec("compressor", {"preset": "hat_control"}),
            _duck("kick_duck"),
        ],
    )
    for bar in range(1, TOTAL_BARS + 1):
        if not _is_drop(bar):
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
            _duck("kick_duck"),
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
        ],
    )
    for bar in range(1, TOTAL_BARS + 1):
        # Bass drops in late in BUILD-1 (last 2 bars), full in drops,
        # silent in breakdowns and outro tail.
        if bar < BUILD_1_END - 2:
            continue
        if _is_breakdown_b(bar):
            continue
        if bar >= DROP_2_END + 4:
            continue
        roots = BASS_ROOTS_7LIM if _is_drop_2(bar) else BASS_ROOTS
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
        mix_db=-12.0,
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
            _duck("kick_duck"),
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
                            _seg(_T_START, _T_BDA, -14.0, -10.0),
                            _seg(_T_BDA, _T_B1, -6.0, -5.0),
                            _seg(_T_B1, _T_D1, -8.0, -11.0),
                            _hold(_T_D1, _T_BDB, -12.0),
                            _seg(_T_BDB, _T_B2, -5.0, -4.0),
                            _seg(_T_B2, _T_D2, -8.0, -11.0),
                            _hold(_T_D2, _T_OUT, -12.0),
                            _seg(_T_OUT, _T_END, -10.0, -18.0),
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
                    _seg(_T_D2, _T_OUT, 4000.0, 5500.0, shape="exp"),
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
                    _hold(_T_D2, _T_OUT, 0.9),
                    _seg(_T_OUT, _T_END, 0.9, 0.6),
                ),
            ),
        ],
    )
    for bar in range(1, TOTAL_BARS + 1):
        # Pad plays continuously from the end of INTRO through OUTRO (minus the
        # last 2 bars where everything fades).
        if bar < 5:  # pad sneaks in bars 5-8
            continue
        if bar >= TOTAL_BARS - 1:
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
        mix_db=-10.0,
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
            _duck("kick_duck"),
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
        ],
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
                amp_db=-5.0,
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
                amp_db=-7.0,
                velocity=vel,
            )
    # Sparse bell accents in DROP-2 (answer the lead once per 4-bar phrase)
    for bar in range(BUILD_2_END, DROP_2_END):
        if (bar - BUILD_2_END) % 4 != 3:  # only the 4th bar of each chord
            continue
        motif = BELL_B[_ci(bar)]
        for beat, n16, partial, gate, vel in motif[:2]:
            score.add_note(
                "bells",
                start=_pos(bar, beat, n16),
                duration=gate * S16 * 0.85,
                partial=partial,
                amp_db=-10.0,
                velocity=vel * 0.8,
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
            _duck("kick_duck"),
        ],
        sends=[VoiceSend(target="plate", send_db=-10.0)],
    )
    # Hoover teases in BUILD-2 (not BUILD-1 — save the reveal for the second drop)
    for bar in range(BUILD_2_END - 4, BUILD_2_END):
        # Tease: hoover stab on beat 1 only, building urgency
        chord = HOOVER_VOICINGS[_ci(bar)]
        for partial in chord:
            score.add_note(
                "hoover",
                start=_pos(bar, 1),
                duration=BEAT * 1.5,
                partial=partial,
                amp_db=-8.0,
                velocity=0.70,
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
                {"delay_seconds": dotted_eighth, "feedback": 0.30, "mix": 0.12},
            ),
            _duck("kick_duck"),
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
    for bar in range(DROP_1_END, BREAKDOWN_B_END):
        motif = LEAD_BD[_ci(bar)]
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
    # DROP-2: same topline, more expressive with light vibrato on longer notes
    for bar in range(BUILD_2_END, DROP_2_END):
        motif = LEAD_DROP[_ci(bar)]
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
            PieceSection("drop_2", _T_D2, _T_OUT),
            PieceSection("outro", _T_OUT, _T_END),
        ),
    ),
}
