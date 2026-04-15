"""Trance studies — 5-limit JI progressive trance at 128 BPM.

Tuning: 5-limit just intonation from A (f0 = 55 Hz).

Main progression:      Am(7/4) → Fmaj7 → D → E
Breakdown progression: Am(7/4) → Em(7/4) → C → Fmaj7

Structure (136 bars, ~4:15):
  bars   1-16   INTRO        Pad + filtered lead, kick enters bar 9
  bars  17-48   DROP 1       Full groove, bass as 8th bounce, building
  bars  49-72   BREAKDOWN    Stripped: pad + lyrical lead, breakdown chords
  bars  73-84   BUILD 2      Elements return, dramatic filter sweep
  bars  85-120  DROP 2       Full intensity, countermelody bass, baroque lead
                bars 109-112: Em(7/4) callback — breakdown chord returns briefly
  bars 121-136  OUTRO        Elements peel away
"""

from __future__ import annotations

from code_musics.automation import AutomationSegment, AutomationSpec, AutomationTarget
from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    VelocityHumanizeSpec,
)
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Score, VoiceSend

# ── Tempo and grid ──────────────────────────────────────────────────────
BPM: float = 128.0
BEAT: float = 60.0 / BPM
BAR: float = 4.0 * BEAT
S16: float = BEAT / 4.0
F0: float = 55.0
TOTAL_BARS: int = 136

# ── Chord voicings (pad register) ────────────────────────────────────
PAD_MAIN: list[list[float]] = [
    [4.0, 24 / 5, 6.0, 7.0],
    [4.0, 24 / 5, 6.0, 32 / 5],
    [4.0, 16 / 3, 20 / 3, 8.0],
    [9 / 2, 6.0, 15 / 2, 9.0],
]
PAD_BD: list[list[float]] = [
    [4.0, 24 / 5, 6.0, 7.0],
    [9 / 2, 21 / 4, 6.0, 36 / 5],
    [24 / 5, 6.0, 36 / 5, 48 / 5],
    [4.0, 24 / 5, 6.0, 32 / 5],
]

# Bass root + octave per main chord (for simple 8th bounce)
BASS_RO: list[tuple[float, float]] = [
    (2.0, 4.0),
    (8 / 5, 16 / 5),
    (4 / 3, 8 / 3),
    (3 / 2, 3.0),
]

# ── Lead pitch constants ──────────────────────────────────────────────
_A4 = 8.0
_B4 = 9.0
_C5 = 48 / 5
_D5 = 32 / 3
_E5 = 12.0
_F5 = 64 / 5
_Fs5 = 40 / 3
_Gs5 = 15.0
_Gsept5 = 14.0
_A5 = 16.0
_B5 = 18.0
_G5_NAT = 72 / 5
_Dsept5 = 21 / 2
_N = tuple[int, int, float, int, float]

# ── Main lead motifs (4 levels × 4 chords) — compact ─────────────────
_AM = [
    [
        (1, 0, _A4, 2, 0.90),
        (1, 3, _E5, 2, 0.60),
        (2, 1, _Gsept5, 3, 0.78),
        (3, 0, _C5, 2, 0.82),
        (3, 3, _A4, 2, 0.50),
        (4, 2, _E5, 2, 0.58),
    ],
    [
        (1, 0, _A4, 2, 0.88),
        (1, 2, _B4, 1, 0.55),
        (1, 3, _C5, 2, 0.72),
        (2, 1, _E5, 2, 0.68),
        (2, 3, _Gsept5, 3, 0.80),
        (3, 2, _E5, 1, 0.55),
        (3, 3, _C5, 2, 0.70),
        (4, 1, _A4, 2, 0.62),
    ],
    [
        (1, 0, _A4, 1, 0.90),
        (1, 1, _B4, 1, 0.62),
        (1, 2, _C5, 1, 0.70),
        (1, 3, _E5, 2, 0.80),
        (2, 1, _Gsept5, 2, 0.85),
        (2, 3, _E5, 1, 0.58),
        (3, 0, _C5, 1, 0.78),
        (3, 1, _B4, 1, 0.60),
        (3, 2, _A4, 2, 0.72),
        (4, 0, _E5, 1, 0.65),
        (4, 2, _Gsept5, 2, 0.75),
    ],
    [
        (1, 0, _A4, 1, 0.95),
        (1, 1, _C5, 1, 0.72),
        (1, 2, _E5, 1, 0.82),
        (1, 3, _Gsept5, 2, 0.92),
        (2, 1, _E5, 1, 0.68),
        (2, 2, _C5, 1, 0.75),
        (2, 3, _B4, 1, 0.58),
        (3, 0, _A4, 1, 0.88),
        (3, 1, _C5, 1, 0.70),
        (3, 2, _D5, 1, 0.78),
        (3, 3, _E5, 1, 0.82),
        (4, 0, _Gsept5, 2, 0.93),
        (4, 2, _E5, 1, 0.65),
        (4, 3, _A5, 1, 0.90),
    ],
]
_FM = [
    [
        (1, 0, _A4, 2, 0.85),
        (1, 2, _F5, 3, 0.72),
        (2, 1, _E5, 2, 0.65),
        (3, 0, _C5, 2, 0.80),
        (3, 2, _A4, 2, 0.48),
        (4, 1, _E5, 3, 0.62),
        (4, 3, _F5, 1, 0.55),
    ],
    [
        (1, 0, _F5, 3, 0.82),
        (1, 3, _E5, 1, 0.60),
        (2, 0, _C5, 2, 0.72),
        (2, 2, _B4, 1, 0.52),
        (2, 3, _A4, 2, 0.68),
        (3, 1, _C5, 1, 0.65),
        (3, 2, _E5, 2, 0.78),
        (4, 1, _F5, 2, 0.70),
    ],
    [
        (1, 0, _A4, 1, 0.85),
        (1, 1, _C5, 1, 0.68),
        (1, 2, _E5, 1, 0.75),
        (1, 3, _F5, 2, 0.88),
        (2, 1, _E5, 1, 0.62),
        (2, 2, _C5, 1, 0.70),
        (2, 3, _B4, 1, 0.52),
        (3, 0, _A4, 1, 0.78),
        (3, 1, _C5, 1, 0.65),
        (3, 2, _E5, 2, 0.80),
        (4, 0, _F5, 1, 0.72),
        (4, 2, _E5, 2, 0.60),
    ],
    [
        (1, 0, _F5, 1, 0.92),
        (1, 1, _E5, 1, 0.75),
        (1, 2, _C5, 1, 0.68),
        (1, 3, _A4, 1, 0.78),
        (2, 0, _B4, 1, 0.60),
        (2, 1, _C5, 1, 0.72),
        (2, 2, _E5, 1, 0.82),
        (2, 3, _F5, 2, 0.93),
        (3, 1, _E5, 1, 0.70),
        (3, 2, _C5, 1, 0.65),
        (3, 3, _A4, 1, 0.58),
        (4, 0, _C5, 1, 0.80),
        (4, 1, _E5, 1, 0.85),
        (4, 2, _F5, 2, 0.90),
    ],
]
_DD = [
    [
        (1, 0, _D5, 2, 0.92),
        (1, 2, _Fs5, 2, 0.70),
        (2, 0, _A5, 3, 0.88),
        (2, 3, _Fs5, 2, 0.55),
        (3, 1, _D5, 2, 0.75),
        (4, 0, _A4, 3, 0.68),
    ],
    [
        (1, 0, _D5, 2, 0.90),
        (1, 3, _A5, 2, 0.72),
        (2, 1, _Fs5, 3, 0.82),
        (3, 0, _D5, 1, 0.78),
        (3, 1, _E5, 1, 0.60),
        (3, 2, _Fs5, 2, 0.72),
        (4, 1, _A4, 2, 0.65),
    ],
    [
        (1, 0, _A4, 1, 0.88),
        (1, 1, _D5, 1, 0.75),
        (1, 2, _E5, 1, 0.68),
        (1, 3, _Fs5, 2, 0.85),
        (2, 1, _A5, 2, 0.90),
        (2, 3, _Fs5, 1, 0.62),
        (3, 0, _E5, 1, 0.72),
        (3, 1, _D5, 1, 0.65),
        (3, 2, _A4, 2, 0.78),
        (4, 0, _D5, 1, 0.70),
        (4, 2, _Fs5, 2, 0.80),
    ],
    [
        (1, 0, _D5, 1, 0.95),
        (1, 1, _E5, 1, 0.72),
        (1, 2, _Fs5, 1, 0.82),
        (1, 3, _A5, 2, 0.93),
        (2, 1, _Fs5, 1, 0.68),
        (2, 2, _E5, 1, 0.60),
        (2, 3, _D5, 1, 0.72),
        (3, 0, _A4, 1, 0.80),
        (3, 2, _Fs5, 1, 0.85),
        (3, 3, _A5, 2, 0.95),
        (4, 1, _Fs5, 1, 0.70),
        (4, 2, _D5, 1, 0.62),
        (4, 3, _E5, 1, 0.75),
    ],
]
_EE = [
    [
        (1, 0, _E5, 2, 0.88),
        (1, 3, _Gs5, 2, 0.68),
        (2, 0, _B5, 3, 0.90),
        (3, 0, _Gs5, 2, 0.80),
        (3, 2, _E5, 2, 0.60),
        (4, 1, _B4, 3, 0.70),
    ],
    [
        (1, 0, _E5, 3, 0.85),
        (2, 0, _Gs5, 2, 0.82),
        (2, 2, _B5, 2, 0.88),
        (3, 0, _E5, 2, 0.72),
        (3, 3, _B4, 2, 0.58),
        (4, 0, _Gs5, 3, 0.75),
    ],
    [
        (1, 0, _E5, 1, 0.88),
        (1, 1, _Gs5, 1, 0.72),
        (1, 2, _B4, 1, 0.60),
        (1, 3, _E5, 1, 0.78),
        (2, 0, _Gs5, 1, 0.82),
        (2, 1, _B5, 2, 0.90),
        (2, 3, _Gs5, 1, 0.65),
        (3, 0, _E5, 1, 0.75),
        (3, 1, _B4, 1, 0.58),
        (3, 3, _Gs5, 2, 0.80),
        (4, 1, _E5, 2, 0.70),
        (4, 3, _B4, 1, 0.62),
    ],
    [
        (1, 0, _B4, 1, 0.85),
        (1, 1, _E5, 1, 0.78),
        (1, 2, _Gs5, 1, 0.88),
        (1, 3, _B5, 2, 0.95),
        (2, 1, _Gs5, 1, 0.72),
        (2, 2, _E5, 1, 0.68),
        (2, 3, _B4, 1, 0.60),
        (3, 0, _E5, 1, 0.82),
        (3, 1, _Gs5, 1, 0.88),
        (3, 2, _B5, 1, 0.92),
        (3, 3, _Gs5, 1, 0.70),
        (4, 0, _E5, 2, 0.80),
        (4, 2, _B4, 1, 0.65),
        (4, 3, _Gs5, 2, 0.90),
    ],
]

LEAD_MAIN: list[list[list[_N]]] = [[_AM[i], _FM[i], _DD[i], _EE[i]] for i in range(4)]

# ── Breakdown lead: lyrical held notes ────────────────────────────────
LEAD_BD: list[list[list[_N]]] = [
    [
        [(1, 0, _A4, 6, 0.85), (2, 2, _Gsept5, 5, 0.80), (3, 3, _E5, 5, 0.78)],
        [(1, 0, _E5, 6, 0.85), (2, 2, _G5_NAT, 5, 0.78), (3, 3, _B4, 5, 0.80)],
        [(1, 0, _C5, 6, 0.82), (2, 2, _E5, 5, 0.78), (3, 3, _G5_NAT, 5, 0.80)],
        [(1, 0, _F5, 8, 0.85), (3, 0, _E5, 4, 0.78), (4, 0, _C5, 4, 0.72)],
    ],
    [
        [
            (1, 0, _E5, 4, 0.82),
            (2, 0, _Gsept5, 6, 0.88),
            (3, 2, _C5, 4, 0.72),
            (4, 2, _A4, 4, 0.68),
        ],
        [
            (1, 0, _B4, 4, 0.80),
            (2, 0, _E5, 6, 0.85),
            (3, 2, _G5_NAT, 5, 0.78),
            (4, 3, _Dsept5, 3, 0.72),
        ],
        [
            (1, 0, _E5, 4, 0.80),
            (1, 3, _G5_NAT, 4, 0.75),
            (2, 3, _C5, 6, 0.82),
            (4, 0, _E5, 4, 0.70),
        ],
        [
            (1, 0, _A4, 4, 0.78),
            (2, 0, _C5, 4, 0.75),
            (3, 0, _E5, 4, 0.80),
            (4, 0, _F5, 4, 0.85),
        ],
    ],
]

# ── Bass countermelody motifs (2 variants × 4 chords) ────────────────
BASS_CM: list[list[list[_N]]] = [
    [
        [
            (1, 0, 2.0, 3, -5),
            (1, 3, 12 / 5, 2, -7),
            (2, 1, 3.0, 3, -6),
            (3, 0, 4.0, 2, -5),
            (3, 2, 3.0, 2, -8),
            (4, 0, 12 / 5, 2, -7),
            (4, 2, 2.0, 2, -6),
        ],
        [
            (1, 0, 8 / 5, 3, -5),
            (1, 3, 2.0, 2, -7),
            (2, 1, 12 / 5, 3, -6),
            (3, 0, 3.0, 3, -5),
            (3, 3, 12 / 5, 2, -8),
            (4, 1, 2.0, 2, -7),
            (4, 3, 8 / 5, 1, -9),
        ],
        [
            (1, 0, 4 / 3, 3, -5),
            (1, 3, 5 / 3, 2, -7),
            (2, 1, 2.0, 3, -6),
            (3, 0, 8 / 3, 2, -5),
            (3, 2, 2.0, 2, -8),
            (4, 0, 5 / 3, 3, -7),
            (4, 3, 4 / 3, 1, -9),
        ],
        [
            (1, 0, 3 / 2, 3, -5),
            (1, 3, 15 / 8, 2, -7),
            (2, 1, 9 / 4, 3, -6),
            (3, 0, 3.0, 2, -5),
            (3, 2, 9 / 4, 2, -8),
            (4, 0, 15 / 8, 3, -7),
        ],
    ],
    [
        [
            (1, 0, 2.0, 2, -5),
            (1, 2, 3.0, 2, -7),
            (2, 0, 4.0, 3, -5),
            (2, 3, 3.0, 1, -9),
            (3, 1, 12 / 5, 3, -6),
            (4, 0, 2.0, 2, -5),
            (4, 3, 3.0, 1, -8),
        ],
        [
            (1, 0, 8 / 5, 2, -5),
            (1, 2, 12 / 5, 2, -7),
            (2, 0, 3.0, 3, -5),
            (2, 3, 2.0, 2, -8),
            (3, 1, 8 / 5, 3, -6),
            (4, 0, 12 / 5, 2, -7),
            (4, 2, 2.0, 2, -8),
        ],
        [
            (1, 0, 4 / 3, 2, -5),
            (1, 2, 2.0, 2, -7),
            (2, 0, 8 / 3, 3, -5),
            (2, 3, 5 / 3, 2, -8),
            (3, 1, 4 / 3, 3, -6),
            (4, 0, 2.0, 2, -7),
            (4, 2, 5 / 3, 2, -8),
        ],
        [
            (1, 0, 3 / 2, 2, -5),
            (1, 2, 9 / 4, 2, -7),
            (2, 0, 3.0, 3, -5),
            (2, 3, 15 / 8, 2, -8),
            (3, 1, 3 / 2, 3, -6),
            (4, 0, 9 / 4, 2, -7),
            (4, 2, 15 / 8, 2, -8),
        ],
    ],
]

# ── Simple bass: 8th-note root-octave bounce ──────────────────────────
_BASS_8TH: list[tuple[int, int, int, int, float]] = [
    (1, 0, 0, 2, -5.0),
    (1, 2, 1, 2, -8.0),
    (2, 0, 0, 2, -6.0),
    (2, 2, 1, 2, -8.0),
    (3, 0, 0, 2, -5.0),
    (3, 2, 1, 2, -8.0),
    (4, 0, 0, 2, -6.0),
    (4, 2, 1, 2, -9.0),
]


# ── Helpers ───────────────────────────────────────────────────────────
def _pos(bar: int, beat: int = 1, n16: int = 0) -> float:
    return (bar - 1) * BAR + (beat - 1) * BEAT + n16 * S16


def _ci(bar: int) -> int:
    return (bar - 1) % 4


def _is_breakdown(bar: int) -> bool:
    return 49 <= bar <= 72


def _is_coda_callback(bar: int) -> bool:
    """Bars 109-112: Em(7/4) chord returns as a surprise callback."""
    return 109 <= bar <= 112


def _lead_level(bar: int) -> int:
    if bar <= 16:
        return 0
    if bar <= 48:
        return min((bar - 17) // 8, 3)
    if bar <= 72:
        return -1
    if bar <= 84:
        return 2 + min((bar - 73) // 4, 1)
    if bar <= 120:
        return 3
    return max(0, 2 - (bar - 121) // 4)


# ── Section boundary times ────────────────────────────────────────────
_T = _pos(1)
_T1 = _pos(17)
_T_BD = _pos(49)
_T_B2 = _pos(73)
_T_D2 = _pos(85)
_T_OUT = _pos(121)
_T_END = _pos(TOTAL_BARS + 1)


def _s(s: float, e: float, sv: float, ev: float) -> AutomationSegment:
    return AutomationSegment(
        start=s, end=e, shape="linear", start_value=sv, end_value=ev
    )


def _h(s: float, e: float, v: float) -> AutomationSegment:
    return AutomationSegment(start=s, end=e, shape="hold", value=v)


def _lfo(
    s: float, e: float, hz: float, depth: float, offset: float
) -> AutomationSegment:
    return AutomationSegment(
        start=s,
        end=e,
        shape="sine_lfo",
        freq_hz=hz,
        depth=depth,
        offset=offset,
    )


def _synth(name: str) -> AutomationTarget:
    return AutomationTarget(kind="synth", name=name)


def _ctrl(name: str) -> AutomationTarget:
    return AutomationTarget(kind="control", name=name)


def build_pure_states() -> Score:
    """5-limit JI progressive trance — full arrangement with deep automation."""
    score = Score(
        f0_hz=F0,
        timing_humanize=TimingHumanizeSpec(preset="tight_ensemble", seed=7),
        master_effects=[
            EffectSpec(
                "compressor",
                {
                    "threshold_db": -22.0,
                    "ratio": 3.0,
                    "attack_ms": 15.0,
                    "release_ms": 200.0,
                    "knee_db": 5.0,
                    "makeup_gain_db": 3.0,
                    "topology": "feedback",
                    "detector_mode": "rms",
                    "detector_bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 120.0,
                            "slope_db_per_oct": 12,
                        },
                    ],
                },
            ),
        ],
    )

    # ── Send bus ──────────────────────────────────────────────────────
    score.add_send_bus(
        "hall",
        effects=[
            EffectSpec(
                "bricasti",
                {
                    "ir_name": "1 Halls 07 Large & Dark",
                    "wet": 1.0,
                    "lowpass_hz": 6000.0,
                    "highpass_hz": 200.0,
                },
            ),
        ],
    )

    # ── Kick ──────────────────────────────────────────────────────────
    # Punchier, quicker decay — timekeeping not boomy.
    score.add_voice(
        "kick",
        synth_defaults={
            "engine": "kick_tom",
            "preset": "909_house",
            "params": {"body_decay_ms": 160.0, "body_punch_ratio": 0.12},
        },
        normalize_peak_db=-6.0,
        mix_db=-5.0,
        velocity_humanize=None,
    )
    for bar in range(1, TOTAL_BARS + 1):
        if bar <= 8 or 49 <= bar <= 72 or bar >= 133:
            continue
        for beat in range(1, 5):
            score.add_note(
                "kick", start=_pos(bar, beat), duration=0.6, freq=55.0, amp_db=-6.0
            )

    # ── Bass ──────────────────────────────────────────────────────────
    # Drop 1: simple 8th root-octave bounce.
    # Build 2: transition (root-octave still).
    # Drop 2: full countermelody — the payoff.
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "polyblep",
            "preset": "moog_bass",
            "params": {
                "cutoff_hz": 160.0,
                "filter_env_amount": 1.2,
                "filter_env_decay": 0.10,
                "resonance_q": 2.0,
                "filter_drive": 0.55,
            },
        },
        mix_db=-8.0,
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living", seed=30),
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog", seed=31),
        max_polyphony=1,
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "low_shelf", "freq_hz": 120.0, "gain_db": 2.5},
                        {"kind": "bell", "freq_hz": 500.0, "gain_db": -3.0, "q": 1.2},
                        {"kind": "lowpass", "cutoff_hz": 800.0, "slope_db_per_oct": 12},
                    ]
                },
            ),
            EffectSpec("saturation", {"drive": 0.35, "mix": 0.4, "mode": "tube"}),
            EffectSpec(
                "compressor", {"preset": "kick_duck", "sidechain_source": "kick"}
            ),
        ],
        automation=[
            AutomationSpec(
                target=_synth("cutoff_hz"),
                segments=(
                    _s(_T1, _T_BD, 160, 200),
                    _s(_T_B2, _T_D2, 160, 200),
                    _s(_T_D2, _T_OUT, 200, 260),
                    _s(_T_OUT, _T_END, 260, 160),
                ),
            ),
            AutomationSpec(
                target=_synth("filter_env_amount"),
                segments=(
                    _s(_T1, _T_BD, 1.2, 1.4),
                    _s(_T_B2, _T_D2, 1.2, 1.5),
                    _s(_T_D2, _T_OUT, 1.5, 1.7),
                    _s(_T_OUT, _T_END, 1.7, 1.2),
                ),
            ),
            AutomationSpec(
                target=_synth("filter_drive"),
                segments=(
                    _h(_T1, _T_BD, 0.45),
                    _h(_T_B2, _T_D2, 0.50),
                    _s(_T_D2, _T_OUT, 0.50, 0.65),
                    _s(_T_OUT, _T_END, 0.65, 0.45),
                ),
            ),
            AutomationSpec(
                target=_synth("resonance_q"),
                segments=(
                    _s(_T1, _T_BD, 1.8, 2.2),
                    _s(_T_B2, _T_D2, 1.8, 2.4),
                    _s(_T_D2, _T_OUT, 2.0, 2.6),
                    _s(_T_OUT, _T_END, 2.6, 1.8),
                ),
            ),
            AutomationSpec(
                target=_synth("filter_env_decay"),
                segments=(
                    _h(_T1, _T_BD, 0.10),
                    _h(_T_B2, _T_D2, 0.09),
                    _s(_T_D2, _T_OUT, 0.09, 0.07),
                    _s(_T_OUT, _T_END, 0.07, 0.10),
                ),
            ),
        ],
    )
    for bar in range(1, TOTAL_BARS + 1):
        if bar <= 16 or 49 <= bar <= 72 or _is_coda_callback(bar) or bar >= 133:
            continue
        ci = _ci(bar)
        if 85 <= bar <= 120:
            # Drop 2: countermelody
            motif = BASS_CM[bar % 2][ci]
            for beat, n16, partial, g, amp in motif:
                score.add_note(
                    "bass",
                    start=_pos(bar, beat, n16),
                    duration=g * S16 * 0.80,
                    partial=partial,
                    amp_db=amp,
                )
        else:
            # Drop 1, Build 2, Outro: simple 8th bounce
            root, octave = BASS_RO[ci]
            for beat, n16, tone_idx, g, amp in _BASS_8TH:
                p = root if tone_idx == 0 else octave
                score.add_note(
                    "bass",
                    start=_pos(bar, beat, n16),
                    duration=g * S16 * 0.80,
                    partial=p,
                    amp_db=amp,
                )

    # ── Pad ──────────────────────────────────────────────────────────
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "filtered_stack",
            "preset": "string_pad",
            "params": {
                "cutoff_hz": 500.0,
                "resonance_q": 0.8,
                "filter_env_amount": 0.10,
                "filter_env_decay": 1.5,
            },
            "env": {
                "attack_ms": 600.0,
                "decay_ms": 400.0,
                "sustain_ratio": 0.85,
                "release_ms": 1800.0,
            },
        },
        mix_db=-9.0,
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad", seed=10),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living", seed=11),
        effects=[
            EffectSpec(
                "tal_chorus_lx", {"mix": 0.5, "chorus_1": True, "chorus_2": False}
            ),
            EffectSpec(
                "compressor", {"preset": "kick_duck", "sidechain_source": "kick"}
            ),
        ],
        sends=[
            VoiceSend(
                target="hall",
                send_db=-10.0,
                automation=[
                    AutomationSpec(
                        target=_ctrl("send_db"),
                        segments=(
                            _s(_T, _T1, -12.0, -10.0),
                            _s(_T1, _T_BD, -10.0, -8.0),
                            _s(_T_BD, _T_B2, -5.0, -4.0),  # lush in breakdown
                            _s(_T_B2, _T_D2, -8.0, -7.0),
                            _h(_T_D2, _T_OUT, -7.0),
                            _s(_T_OUT, _T_END, -7.0, -12.0),
                        ),
                    ),
                ],
            ),
        ],
        automation=[
            AutomationSpec(
                target=_synth("cutoff_hz"),
                segments=(
                    _s(_T, _T1, 500, 700),
                    _s(_T1, _T_BD, 700, 1000),
                    _s(_T_BD, _T_B2, 1000, 1200),  # open for breakdown chords
                    _s(_T_B2, _T_D2, 800, 1000),
                    _s(_T_D2, _T_OUT, 1000, 1200),
                    _s(_T_OUT, _T_END, 1200, 500),
                ),
            ),
            AutomationSpec(
                target=_synth("filter_env_amount"),
                segments=(
                    _h(_T, _T1, 0.10),
                    _s(_T1, _T_BD, 0.15, 0.30),
                    _h(_T_BD, _T_B2, 0.12),
                    _s(_T_B2, _T_D2, 0.20, 0.35),
                    _s(_T_D2, _T_OUT, 0.35, 0.40),
                    _s(_T_OUT, _T_END, 0.40, 0.10),
                ),
            ),
            AutomationSpec(
                target=_synth("resonance_q"),
                segments=(
                    _h(_T, _T1, 0.8),
                    _s(_T1, _T_BD, 0.8, 1.1),
                    _s(_T_BD, _T_B2, 1.2, 1.4),  # resonance bloom in breakdown
                    _s(_T_B2, _T_D2, 0.9, 1.0),
                    _s(_T_D2, _T_OUT, 1.0, 1.2),
                    _s(_T_OUT, _T_END, 1.2, 0.8),
                ),
            ),
            AutomationSpec(
                target=_ctrl("pan"),
                segments=(
                    _lfo(_T, _T_END, 0.03, 0.06, 0.0),  # slow stereo drift
                ),
            ),
        ],
    )
    for bar in range(1, TOTAL_BARS + 1):
        use_bd = _is_breakdown(bar) or _is_coda_callback(bar)
        chords = PAD_BD if use_bd else PAD_MAIN
        chord = chords[_ci(bar)]
        for i, partial in enumerate(chord):
            score.add_note(
                "pad",
                start=_pos(bar),
                duration=BAR * 0.98,
                partial=partial,
                amp_db=-6.0 - i * 0.5,
            )

    # ── Lead ──────────────────────────────────────────────────────────
    dotted_eighth = 3.0 * S16
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "polyblep",
            "preset": "warm_lead",
            "params": {
                "cutoff_hz": 600.0,
                "filter_env_amount": 0.6,
                "filter_env_decay": 0.06,
                "resonance_q": 1.8,
                "filter_drive": 0.05,
            },
            "env": {
                "attack_ms": 3.0,
                "decay_ms": 100.0,
                "sustain_ratio": 0.12,
                "release_ms": 70.0,
            },
        },
        mix_db=-10.0,
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble", seed=42),
        envelope_humanize=EnvelopeHumanizeSpec(preset="loose_pluck", seed=43),
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 300.0,
                            "slope_db_per_oct": 12,
                        },
                        {"kind": "bell", "freq_hz": 1500.0, "gain_db": 2.0, "q": 1.0},
                    ]
                },
            ),
            EffectSpec(
                "chow_tape",
                {"drive": 0.25, "saturation": 0.3, "bias": 0.5, "mix": 35.0},
            ),
            EffectSpec(
                "delay", {"delay_seconds": dotted_eighth, "feedback": 0.45, "mix": 0.38}
            ),
            EffectSpec(
                "compressor", {"preset": "kick_duck", "sidechain_source": "kick"}
            ),
        ],
        sends=[
            VoiceSend(
                target="hall",
                send_db=-14.0,
                automation=[
                    AutomationSpec(
                        target=_ctrl("send_db"),
                        segments=(
                            _s(_T, _T1, -16.0, -12.0),
                            _s(_T1, _T_BD, -12.0, -8.0),
                            _h(_T_BD, _T_B2, -4.0),  # lush breakdown
                            _s(_T_B2, _T_D2, -8.0, -6.0),
                            _h(_T_D2, _T_OUT, -6.0),
                            _s(_T_OUT, _T_END, -6.0, -16.0),
                        ),
                    ),
                ],
            ),
        ],
        automation=[
            AutomationSpec(
                target=_synth("cutoff_hz"),
                clamp_min=800.0,
                segments=(
                    _s(_T, _T1, 600, 1000),
                    _s(_T1, _T_BD, 1000, 2200),
                    _s(_T_BD, _T_B2, 1400, 1000),
                    _s(_T_B2, _T_D2, 800, 2800),  # dramatic sweep
                    _h(_T_D2, _T_OUT, 2600),
                    _s(_T_OUT, _T_END, 2600, 800),
                ),
            ),
            AutomationSpec(
                target=_synth("filter_env_amount"),
                segments=(
                    _h(_T, _T1, 0.6),
                    _s(_T1, _T_BD, 0.8, 1.6),
                    _h(_T_BD, _T_B2, 0.5),
                    _s(_T_B2, _T_D2, 1.0, 1.8),
                    _h(_T_D2, _T_OUT, 1.8),
                    _s(_T_OUT, _T_END, 1.8, 0.6),
                ),
            ),
            AutomationSpec(
                target=_synth("filter_drive"),
                segments=(
                    _h(_T, _T1, 0.05),
                    _s(_T1, _T_BD, 0.08, 0.18),
                    _h(_T_BD, _T_B2, 0.05),
                    _s(_T_B2, _T_D2, 0.10, 0.25),
                    _h(_T_D2, _T_OUT, 0.22),
                    _s(_T_OUT, _T_END, 0.22, 0.05),
                ),
            ),
            AutomationSpec(
                target=_synth("decay"),
                segments=(
                    _h(_T, _T_BD, 0.100),
                    _h(_T_BD, _T_B2, 0.140),  # longer in breakdown
                    _s(_T_B2, _T_D2, 0.100, 0.080),  # tighter in climax
                    _h(_T_D2, _T_OUT, 0.075),
                    _s(_T_OUT, _T_END, 0.075, 0.100),
                ),
            ),
            AutomationSpec(
                target=_synth("release"),
                segments=(
                    _h(_T, _T_BD, 0.070),
                    _h(_T_BD, _T_B2, 0.120),
                    _s(_T_B2, _T_D2, 0.070, 0.050),
                    _h(_T_D2, _T_OUT, 0.050),
                    _s(_T_OUT, _T_END, 0.050, 0.070),
                ),
            ),
            AutomationSpec(
                target=_ctrl("pan"),
                segments=(
                    _lfo(_T, _T_END, 0.05, 0.08, 0.0),  # gentle stereo drift
                ),
            ),
        ],
    )
    for bar in range(1, TOTAL_BARS + 1):
        ci = _ci(bar)
        level = _lead_level(bar)
        motif = LEAD_BD[bar % 2][ci] if level == -1 else LEAD_MAIN[level][ci]
        for beat, n16, partial, gate, vel in motif:
            score.add_note(
                "lead",
                start=_pos(bar, beat, n16),
                duration=gate * S16 * 0.85,
                partial=partial,
                amp_db=-6.0,
                velocity=vel,
            )

    # ── Hats ──────────────────────────────────────────────────────────
    score.add_voice(
        "hat",
        synth_defaults={"engine": "noise_perc", "preset": "chh"},
        mix_db=-12.0,
        normalize_peak_db=-6.0,
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living", seed=20),
        effects=[
            EffectSpec(
                "eq",
                {"bands": [{"kind": "high_shelf", "freq_hz": 8000.0, "gain_db": 2.5}]},
            ),
            EffectSpec(
                "compressor", {"preset": "kick_duck", "sidechain_source": "kick"}
            ),
        ],
    )
    for bar in range(1, TOTAL_BARS + 1):
        if bar <= 8 or 49 <= bar <= 72 or bar >= 133:
            continue
        if bar <= 16 or 73 <= bar <= 78:
            density, freq = "8ths", 10000.0
        elif bar <= 32 or 79 <= bar <= 84:
            density, freq = "soft_16ths", 11500.0
        elif bar <= 48 or bar <= 120:
            density, freq = "full_16ths", 13500.0
        else:
            density, freq = "8ths", 11000.0
        for beat in range(1, 5):
            for n16 in range(4):
                if density == "8ths" and n16 in (1, 3):
                    continue
                amp = (
                    -17.0
                    if (density == "soft_16ths" and n16 in (1, 3))
                    else (-10.0 if n16 == 0 else -12.5 if n16 == 2 else -15.0)
                )
                score.add_note(
                    "hat",
                    start=_pos(bar, beat, n16),
                    duration=0.04,
                    freq=freq,
                    amp_db=amp,
                )

    # ── Clap ──────────────────────────────────────────────────────────
    score.add_voice(
        "clap",
        synth_defaults={"engine": "noise_perc", "preset": "clap"},
        mix_db=-9.5,
        normalize_peak_db=-6.0,
        velocity_humanize=None,
        effects=[EffectSpec("chorus", {"preset": "juno_subtle", "mix": 0.12})],
        sends=[VoiceSend(target="hall", send_db=-14.0)],
    )
    for bar in range(1, TOTAL_BARS + 1):
        if bar <= 16 or 49 <= bar <= 72 or bar >= 133:
            continue
        for beat in [2, 4]:
            score.add_note(
                "clap", start=_pos(bar, beat), duration=0.12, freq=3000.0, amp_db=-4.0
            )
        if 85 <= bar <= 120 and bar % 4 == 0:
            for n16 in [1, 2, 3]:
                score.add_note(
                    "clap",
                    start=_pos(bar, 4, n16),
                    duration=0.06,
                    freq=3500.0,
                    amp_db=-22.0 + n16 * 0.8,
                )

    return score


PIECES: dict[str, PieceDefinition] = {
    "pure_states": PieceDefinition(
        name="pure_states",
        output_name="pure_states",
        build_score=build_pure_states,
    ),
}
