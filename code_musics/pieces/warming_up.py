"""warming_up -- an analog polysynth waking from cold.

Starts precise and digital, warms up, overshoots into unstable searching,
then locks in and builds to euphoria.

Five sections: Cold → Warming → Searching → Arriving → Euphoria.
~5 min, 82 bars at 66 BPM, 7-limit JI in F#.

Showcases ladder filter, oscillator imperfections, voice card spread,
filter morphing, feedback path, serial HPF, and VCA nonlinearity.
"""

from __future__ import annotations

from code_musics.automation import AutomationSegment, AutomationSpec, AutomationTarget
from code_musics.composition import augment, diminish
from code_musics.generative.cloud import stochastic_cloud
from code_musics.generative.tone_pool import TonePool
from code_musics.harmonic_drift import (
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
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import (
    EffectSpec,
    NoteEvent,
    Phrase,
    Score,
    VelocityParamMap,
    VoiceSend,
)
from code_musics.smear import strum, thicken
from code_musics.synth import db_to_amp

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

BPM = 66.0
BEAT = 60.0 / BPM
BAR = 4.0 * BEAT
TOTAL_BARS = 82
F0 = 46.25  # F#1

# Section boundaries (bar numbers, 1-indexed)
COLD_BAR = 1  # 8 bars
WARM_BAR = 9  # 12 bars
SEARCH_BAR = 21  # 14 bars — unstable, utonal, drift overshoot
ARRIVE_BAR = 35  # 16 bars — locks in, lead enters
EUPH_BAR = 51  # 32 bars — full warmth, 5-limit ending
END_BAR = 83


def _pos(bar: int, beat: float = 1.0) -> float:
    return (bar - 1) * BAR + (beat - 1) * BEAT


# Section timestamps
T_COLD = _pos(COLD_BAR)
T_WARM = _pos(WARM_BAR)
T_SEARCH = _pos(SEARCH_BAR)
T_ARRIVE = _pos(ARRIVE_BAR)
T_EUPH = _pos(EUPH_BAR)
T_END = _pos(END_BAR)

# Post-end release tails per voice family. Notes/drones may extend past T_END
# so their envelopes breathe out naturally instead of cutting at the bar line.
RELEASE_TAIL_S = 2.0  # lead + thickened lead copies
TEXTURE_RELEASE_TAIL_S = (
    6.0  # sustained drone/texture layers — extended to let hall reverb ring out
)
GHOST_RELEASE_TAIL_S = 6.0  # long reverb-soaked ghost echoes

# ---------------------------------------------------------------------------
# Tuning — 7-limit JI partial ratios relative to F#
# ---------------------------------------------------------------------------

P1 = 1.0
P_9_8 = 9 / 8  # G#
P_7_6 = 7 / 6  # septimal minor third
P_6_5 = 6 / 5  # minor third (5-limit)
P_5_4 = 5 / 4  # A# (major third)
P_4_3 = 4 / 3  # B (perfect fourth)
P_3_2 = 3 / 2  # C# (perfect fifth)
P_8_5 = 8 / 5  # minor sixth (5-limit)
P_5_3 = 5 / 3  # D# (major sixth)
P_7_4 = 7 / 4  # E (septimal seventh)
P_15_8 = 15 / 8  # E#/F (major seventh)
P2 = 2.0  # F#'


# ---------------------------------------------------------------------------
# Chord voicings
# ---------------------------------------------------------------------------

# --- Major / warm voicings ---


def _chord_i_simple() -> list[tuple[float, float]]:
    """F#maj7(7) — root position, simple."""
    return [
        (P1 * 4, -6.0),
        (P_5_4 * 4, -8.0),
        (P_3_2 * 4, -8.0),
        (P_7_4 * 4, -9.0),
    ]


def _chord_i_open() -> list[tuple[float, float]]:
    """F#maj7(7) — open voicing."""
    return [
        (P1 * 2, -7.0),
        (P_5_4 * 4, -8.0),
        (P_3_2 * 4, -8.5),
        (P_7_4 * 4, -9.0),
        (P1 * 8, -10.0),
    ]


def _chord_i_rich() -> list[tuple[float, float]]:
    """F#maj7(7) — rich, added 9th."""
    return [
        (P1 * 4, -6.0),
        (P_9_8 * 4, -10.0),
        (P_5_4 * 4, -8.0),
        (P_3_2 * 4, -8.0),
        (P_7_4 * 4, -8.5),
    ]


def _chord_iv() -> list[tuple[float, float]]:
    """B major."""
    return [
        (P_4_3 * 4, -6.0),
        (P_5_3 * 4, -8.0),
        (P2 * 4, -8.0),
        (P_5_4 * 8, -9.0),
    ]


def _chord_iv_rich() -> list[tuple[float, float]]:
    """B major — wide."""
    return [
        (P_4_3 * 2, -7.0),
        (P_5_3 * 4, -8.0),
        (P2 * 4, -8.0),
        (P_5_4 * 8, -9.0),
    ]


def _chord_v() -> list[tuple[float, float]]:
    """C# dominant."""
    return [
        (P_3_2 * 4, -6.0),
        (P_15_8 * 4, -8.0),
        (P_9_8 * 8, -8.0),
    ]


def _chord_v_rich() -> list[tuple[float, float]]:
    """C# — open."""
    return [
        (P_3_2 * 2, -7.0),
        (P_15_8 * 4, -8.0),
        (P_9_8 * 8, -8.5),
        (P_3_2 * 8, -10.0),
    ]


def _chord_vi() -> list[tuple[float, float]]:
    """D# minor."""
    return [
        (P_5_3 * 4, -6.0),
        (P2 * 4, -8.0),
        (P_5_4 * 8, -9.0),
        (P_3_2 * 8, -10.0),
    ]


def _chord_vi_rich() -> list[tuple[float, float]]:
    """D# minor — wide."""
    return [
        (P_5_3 * 2, -7.0),
        (P2 * 4, -8.0),
        (P_5_4 * 8, -9.0),
        (P_3_2 * 8, -10.0),
        (P_5_3 * 8, -10.5),
    ]


# --- Searching voicings: utonal / open fifths, harmonically spare ---


def _chord_open_fifth_i() -> list[tuple[float, float]]:
    """F# power chord — no third, exposed."""
    return [(P1 * 4, -6.0), (P_3_2 * 4, -7.0), (P1 * 8, -9.0)]


def _chord_open_fifth_iv() -> list[tuple[float, float]]:
    """B power chord."""
    return [(P_4_3 * 4, -6.0), (P2 * 4, -7.0), (P_4_3 * 8, -9.0)]


def _chord_utonal_i() -> list[tuple[float, float]]:
    """F# utonal — minor-flavoured, dark."""
    return [
        (P1 * 4, -6.0),
        (P_6_5 * 4, -8.0),
        (P_3_2 * 4, -8.0),
        (P_8_5 * 4, -9.5),
    ]


def _chord_utonal_iv() -> list[tuple[float, float]]:
    """B utonal — dark fourth."""
    return [
        (P_4_3 * 4, -6.0),
        (P_4_3 * P_6_5 * 4, -8.0),
        (P2 * 4, -8.0),
    ]


# --- Euphoric voicings: pure 5-limit, no septimal ---


def _chord_i_euphoric() -> list[tuple[float, float]]:
    """F# major — pure 5-limit, juicy."""
    return [
        (P1 * 2, -7.0),
        (P1 * 4, -6.0),
        (P_5_4 * 4, -7.0),
        (P_3_2 * 4, -7.5),
        (P1 * 8, -8.0),
        (P_5_4 * 8, -9.5),
    ]


def _chord_iv_euphoric() -> list[tuple[float, float]]:
    """B major — pure 5-limit."""
    return [
        (P_4_3 * 2, -7.0),
        (P_4_3 * 4, -6.0),
        (P_5_3 * 4, -7.0),
        (P2 * 4, -7.5),
        (P_5_4 * 8, -8.5),
    ]


def _chord_v_euphoric() -> list[tuple[float, float]]:
    """C# major — pure 5-limit."""
    return [
        (P_3_2 * 2, -7.0),
        (P_3_2 * 4, -6.0),
        (P_15_8 * 4, -7.5),
        (P_9_8 * 8, -8.0),
        (P_3_2 * 8, -9.0),
    ]


# ---------------------------------------------------------------------------
# Progressions per section
# ---------------------------------------------------------------------------

PROG_COLD = [_chord_i_simple, _chord_vi, _chord_iv, _chord_i_simple]
PROG_WARM_1 = [_chord_i_open, _chord_iv, _chord_v, _chord_vi]
PROG_WARM_2 = [_chord_i_open, _chord_vi]
PROG_SEARCH = [
    _chord_open_fifth_i,
    _chord_utonal_iv,
    _chord_utonal_i,
    _chord_open_fifth_iv,
    _chord_utonal_i,
    _chord_open_fifth_i,
    _chord_open_fifth_iv,
]
PROG_ARRIVE = [_chord_i_rich, _chord_iv_rich, _chord_v_rich, _chord_vi_rich]
PROG_EUPH_1 = [_chord_iv_rich, _chord_v_rich, _chord_i_rich, _chord_vi_rich]
PROG_EUPH_2 = [
    _chord_i_euphoric,
    _chord_iv_euphoric,
    _chord_v_euphoric,
    _chord_i_euphoric,
]
PROG_EUPH_END = [_chord_iv_euphoric, _chord_v_euphoric, _chord_i_euphoric]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synth_auto(name: str, segs: tuple[AutomationSegment, ...]) -> AutomationSpec:
    return AutomationSpec(
        target=AutomationTarget(kind="synth", name=name), segments=segs
    )


def _ctrl(name: str, segs: tuple[AutomationSegment, ...]) -> AutomationSpec:
    return AutomationSpec(
        target=AutomationTarget(kind="control", name=name), segments=segs
    )


def _lin(t0: float, t1: float, start: float, end: float) -> AutomationSegment:
    """Linear-ramp automation segment. Prefer kw args for readability at call sites."""
    return AutomationSegment(
        start=t0, end=t1, shape="linear", start_value=start, end_value=end
    )


def _exp(t0: float, t1: float, start: float, end: float) -> AutomationSegment:
    """Exponential-ramp automation segment. Prefer kw args for readability."""
    return AutomationSegment(
        start=t0, end=t1, shape="exp", start_value=start, end_value=end
    )


def _hold(t0: float, t1: float, value: float) -> AutomationSegment:
    """Hold-value automation segment."""
    return AutomationSegment(start=t0, end=t1, shape="hold", value=value)


def _lfo(
    t0: float, t1: float, hz: float, depth: float, offset: float = 0.0
) -> AutomationSegment:
    """Sine-LFO automation segment. Prefer kw args for hz/depth/offset."""
    return AutomationSegment(
        start=t0, end=t1, shape="sine_lfo", freq_hz=hz, depth=depth, offset=offset
    )


def _add_bounded_note(
    score: Score,
    voice: str,
    bar: int,
    beat: float,
    partial: float,
    dur_beats: float,
    *,
    amp_db: float,
    velocity: float,
    time_limit: float,
    pitch_motion: PitchMotionSpec | None = None,
) -> None:
    """Add a note to ``voice`` if its start time is before ``time_limit``.

    Consolidates the bass/lead bounded-add closures so every voice uses the
    same guard against notes spilling past the piece end.
    """
    t = _pos(bar, beat)
    if t < time_limit:
        score.add_note(
            voice,
            start=t,
            duration=dur_beats * BEAT,
            partial=partial,
            amp_db=amp_db,
            velocity=velocity,
            pitch_motion=pitch_motion,
        )


# ---------------------------------------------------------------------------
# build_score
# ---------------------------------------------------------------------------


def build_score() -> Score:
    score = Score(
        f0_hz=F0,
        sample_rate=44_100,
        master_effects=DEFAULT_MASTER_EFFECTS,
        timing_humanize=TimingHumanizeSpec(preset="chamber"),
    )

    score.add_send_bus(
        "hall",
        effects=[
            EffectSpec(
                "reverb", {"room_size": 0.88, "damping": 0.45, "wet_level": 0.60}
            ),
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 180.0,
                            "slope_db_per_oct": 12,
                        },
                        {
                            "kind": "high_shelf",
                            "freq_hz": 6000.0,
                            "gain_db": -2.5,
                            "q": 0.7,
                        },
                    ]
                },
            ),
        ],
        return_db=0.0,
    )

    score.add_send_bus(
        "delay",
        effects=[
            EffectSpec("mod_delay", {"preset": "tape_wander"}),
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 250.0,
                            "slope_db_per_oct": 12,
                        }
                    ]
                },
            ),
        ],
        return_db=0.0,
    )

    # --- PAD ---
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "saw",
            "cutoff_hz": 2800.0,
            "resonance_q": 1.2,
            "filter_topology": "svf",
            "filter_drive": 0.0,
            "filter_even_harmonics": 0.0,
            "filter_morph": 0.0,
            "osc_softness": 0.0,
            "osc_asymmetry": 0.0,
            "osc_shape_drift": 0.0,
            "voice_card_spread": 0.0,
            "voice_card_pitch_spread": 0.3,
            "pitch_drift": 0.0,
            "analog_jitter": 0.0,
            "noise_floor": 0.0,
            "cutoff_drift": 0.0,
            "attack": 0.3,
            "decay": 0.5,
            "sustain_level": 0.7,
            "release": 1.2,
        },
        effects=[
            EffectSpec(
                "chorus",
                {"rate_hz": 0.4, "depth_ms": 2.0, "mix": 0.2},
                automation=[
                    _ctrl(
                        "mix",
                        (
                            _hold(T_COLD, _pos(6), 0.05),  # dry, just waking
                            _lin(_pos(6), T_WARM, 0.05, 0.2),  # subtle creep in
                            _hold(T_WARM, T_END, 0.2),
                        ),
                    )
                ],
            ),
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 120.0, "slope_db_per_oct": 12}
                    ]
                },
            ),
        ],
        normalize_lufs=-22.0,
        mix_db=-3.0,
        sends=[VoiceSend(target="hall", send_db=-6.0)],
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_group="warmth",
        automation=[
            # Analog character: 0 → overshoot in Search → settle in Arrive/Euph
            _synth_auto(
                "pitch_drift",
                (
                    _hold(T_COLD, T_WARM, 0.0),
                    _lin(T_WARM, T_SEARCH, 0.0, 0.1),
                    _lin(T_SEARCH, T_ARRIVE, 0.1, 0.25),  # overshoot
                    _lin(T_ARRIVE, T_EUPH, 0.25, 0.15),
                    _hold(T_EUPH, T_END, 0.15),
                ),
            ),
            _synth_auto(
                "analog_jitter",
                (
                    _hold(T_COLD, _pos(5), 0.0),
                    _lin(_pos(5), T_WARM, 0.0, 0.05),  # subtle awakening
                    _lin(T_WARM, T_SEARCH, 0.05, 0.5),
                    _lin(T_SEARCH, T_ARRIVE, 0.5, 1.5),  # overshoot
                    _lin(T_ARRIVE, T_EUPH, 1.5, 1.0),
                    _hold(T_EUPH, T_END, 1.0),
                ),
            ),
            _synth_auto(
                "noise_floor",
                (
                    _hold(T_COLD, T_WARM, 0.0),
                    _lin(T_WARM, T_ARRIVE, 0.0, 0.001),
                    _hold(T_ARRIVE, T_END, 0.001),
                ),
            ),
            _synth_auto(
                "cutoff_drift",
                (
                    _hold(T_COLD, T_WARM, 0.0),
                    _lin(T_WARM, T_SEARCH, 0.0, 0.3),
                    _lin(T_SEARCH, T_ARRIVE, 0.3, 0.8),  # overshoot
                    _lin(T_ARRIVE, T_EUPH, 0.8, 0.5),
                    _hold(T_EUPH, T_END, 0.5),
                ),
            ),
            # Osc character
            _synth_auto(
                "osc_softness",
                (
                    _hold(T_COLD, T_WARM, 0.0),
                    _lin(T_WARM, T_SEARCH, 0.0, 0.1),
                    _lin(T_SEARCH, T_ARRIVE, 0.1, 0.18),
                    _hold(T_ARRIVE, T_END, 0.18),
                ),
            ),
            _synth_auto(
                "osc_shape_drift",
                (
                    _hold(T_COLD, _pos(6), 0.0),
                    _lin(_pos(6), T_WARM, 0.0, 0.02),  # barely perceptible creep
                    _lin(T_WARM, T_SEARCH, 0.02, 0.15),
                    _lin(T_SEARCH, T_ARRIVE, 0.15, 0.45),  # overshoot
                    _lin(T_ARRIVE, T_EUPH, 0.45, 0.3),
                    _hold(T_EUPH, T_END, 0.3),
                ),
            ),
            _synth_auto(
                "osc_asymmetry",
                (
                    _hold(T_COLD, T_WARM, 0.0),
                    _lin(T_WARM, T_ARRIVE, 0.0, 0.1),
                    _hold(T_ARRIVE, T_END, 0.1),
                ),
            ),
            # Resonance bloom at the Warming→Searching boundary —
            # transient spike as the filter destabilizes before Searching.
            _synth_auto(
                "resonance_q",
                (
                    _hold(T_COLD, _pos(20), 1.2),
                    _lin(_pos(20), _pos(20, 3), 1.2, 2.5),  # bloom up
                    _lin(_pos(20, 3), T_SEARCH, 2.5, 1.4),  # fall back
                    _hold(T_SEARCH, T_END, 1.4),
                ),
            ),
            # Voice card spread: opens, overshoots in Search, settles
            _synth_auto(
                "voice_card_spread",
                (
                    _hold(T_COLD, T_WARM, 0.0),
                    _lin(T_WARM, T_SEARCH, 0.0, 1.0),
                    _lin(T_SEARCH, T_ARRIVE, 1.0, 2.8),  # overshoot
                    _lin(T_ARRIVE, T_EUPH, 2.8, 2.0),
                    _hold(T_EUPH, T_END, 2.0),
                ),
            ),
            _synth_auto(
                "voice_card_filter_spread",
                (
                    _hold(T_COLD, T_WARM, 0.0),
                    _lin(T_WARM, T_SEARCH, 0.0, 1.0),
                    _lin(T_SEARCH, T_ARRIVE, 1.0, 3.0),  # wide in search
                    _lin(T_ARRIVE, T_EUPH, 3.0, 2.5),
                    _hold(T_EUPH, T_END, 2.5),
                ),
            ),
            # Filter: morph sweeps further in Search, settles
            _synth_auto(
                "filter_morph",
                (
                    _hold(T_COLD, T_WARM, 0.0),
                    _lin(T_WARM, T_SEARCH, 0.0, 0.2),
                    _lin(T_SEARCH, T_ARRIVE, 0.2, 0.7),  # past BP territory
                    _lin(T_ARRIVE, T_EUPH, 0.7, 0.4),
                    _hold(T_EUPH, T_END, 0.4),
                ),
            ),
            _synth_auto(
                "filter_drive",
                (
                    _hold(T_COLD, T_WARM, 0.0),
                    _lin(T_WARM, T_SEARCH, 0.0, 0.06),
                    _lin(T_SEARCH, T_ARRIVE, 0.06, 0.15),
                    _hold(T_ARRIVE, T_END, 0.15),
                ),
            ),
            _synth_auto(
                "filter_even_harmonics",
                (
                    _hold(T_COLD, T_WARM, 0.0),
                    _lin(T_WARM, T_EUPH, 0.0, 0.15),
                    _hold(T_EUPH, T_END, 0.15),
                ),
            ),
            _synth_auto(
                "cutoff_hz",
                (
                    _hold(T_COLD, _pos(5), 2800.0),
                    _exp(_pos(5), T_WARM, 2800.0, 2900.0),  # tiny opening late Cold
                    _exp(T_WARM, _pos(19), 2900.0, 3200.0),
                    # Warming→Searching destabilization: brief dip/close at bar 19-20
                    _exp(_pos(19), _pos(20, 3), 3200.0, 2400.0),  # close on overshoot
                    _exp(_pos(20, 3), T_SEARCH, 2400.0, 3200.0),  # bounce back
                    # Searching: sweep, then pre-arrival pull-back
                    _exp(T_SEARCH, _pos(34), 3200.0, 2200.0),
                    _exp(_pos(34), _pos(34, 3), 2200.0, 1500.0),  # dip before arrival
                    _exp(_pos(34, 3), T_ARRIVE, 1500.0, 2200.0),  # recover into Arrive
                    _exp(T_ARRIVE, T_EUPH, 2200.0, 3500.0),
                    _hold(T_EUPH, T_END, 3500.0),
                ),
            ),
            # Osc2 thickness
            _synth_auto(
                "osc2_level",
                (
                    _hold(T_COLD, T_SEARCH, 0.0),
                    _lin(T_SEARCH, T_EUPH, 0.0, 0.35),
                    _hold(T_EUPH, T_END, 0.35),
                ),
            ),
            _synth_auto(
                "osc2_detune_cents",
                (
                    _hold(T_COLD, T_ARRIVE, 0.0),
                    _lin(T_ARRIVE, T_EUPH, 0.0, 8.0),
                    _hold(T_EUPH, T_END, 8.0),
                ),
            ),
            # Mix / send rides
            _ctrl("pan", (_lfo(T_WARM, T_END, 0.04, 0.12),)),
            _ctrl(
                "send_db",
                (
                    _hold(T_COLD, T_WARM, -6.0),
                    _lin(T_WARM, T_SEARCH, -6.0, -4.0),
                    _lin(T_SEARCH, T_ARRIVE, -4.0, -2.5),  # wetter in search
                    _hold(T_ARRIVE, T_END, -3.0),
                ),
            ),
            _ctrl(
                "mix_db",
                (
                    _hold(T_COLD, T_SEARCH, -3.0),
                    # Searching: drift toward -4.5 then pull back harder before Arrive
                    _lin(T_SEARCH, _pos(33), -3.0, -4.5),
                    _lin(_pos(33), _pos(34, 3), -4.5, -5.5),  # pre-arrival pullback
                    _lin(_pos(34, 3), T_ARRIVE, -5.5, -4.5),  # recover
                    # Arriving: push forward, with a last-bar swell before Euphoria
                    _lin(T_ARRIVE, _pos(49), -4.5, -2.0),
                    _lin(_pos(49), _pos(50), -2.0, -1.0),  # pre-Euphoria swell
                    _lin(_pos(50), T_EUPH, -1.0, -1.5),
                    # Euphoria: hold, mid-section lull, final breath
                    _hold(T_EUPH, _pos(63), -1.5),
                    _lin(_pos(63), _pos(64), -1.5, -2.5),  # dip into lull
                    _hold(_pos(64), _pos(65), -2.5),
                    _lin(_pos(65), _pos(66), -2.5, -1.5),  # return
                    _hold(_pos(66), _pos(70, 3), -1.5),
                    _lin(_pos(70, 3), _pos(72), -1.5, -3.0),  # breath before final
                    _lin(_pos(72), _pos(73), -3.0, -1.5),  # recover for final
                    _hold(_pos(73), _pos(78), -1.5),
                    _lin(_pos(78), _pos(80), -1.5, -2.5),  # final breath
                    _lin(_pos(80), T_END, -2.5, -1.5),  # rise into final held chord
                ),
            ),
        ],
    )

    # --- BASS ---
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "saw",
            "filter_topology": "ladder",
            "cutoff_hz": 500.0,
            "resonance_q": 2.0,
            "bass_compensation": 0.7,
            "filter_drive": 0.1,
            "feedback_amount": 0.0,
            "feedback_saturation": 0.4,
            "osc_softness": 0.1,
            "voice_card_spread": 0.5,
            "voice_card_pitch_spread": 0.2,
            "attack": 0.02,
            "decay": 0.3,
            "sustain_level": 0.6,
            "release": 0.4,
        },
        effects=[
            EffectSpec("saturation", {"preset": "tube_warm", "mix": 0.15}),
        ],
        normalize_lufs=-22.0,
        mix_db=-1.0,
        sends=[VoiceSend(target="hall", send_db=-14.0)],
        envelope_humanize=EnvelopeHumanizeSpec(preset="loose_pluck"),
        velocity_group="warmth",
        automation=[
            _synth_auto(
                "feedback_amount",
                (
                    _hold(T_COLD, T_WARM, 0.0),
                    _lin(T_WARM, T_SEARCH, 0.0, 0.15),
                    _lin(T_SEARCH, T_ARRIVE, 0.15, 0.35),  # growly in search
                    _lin(T_ARRIVE, T_EUPH, 0.35, 0.25),
                    _hold(T_EUPH, T_END, 0.25),
                ),
            ),
            _synth_auto(
                "cutoff_hz",
                (
                    _hold(T_COLD, T_WARM, 450.0),
                    _exp(T_WARM, T_SEARCH, 450.0, 650.0),
                    _exp(T_SEARCH, T_ARRIVE, 650.0, 500.0),  # darker in search
                    _exp(T_ARRIVE, T_EUPH, 500.0, 900.0),
                    _hold(T_EUPH, T_END, 900.0),
                ),
            ),
            _synth_auto(
                "filter_drive",
                (
                    _hold(T_COLD, T_WARM, 0.1),
                    _lin(T_WARM, T_SEARCH, 0.1, 0.2),
                    _lin(T_SEARCH, T_ARRIVE, 0.2, 0.3),
                    _hold(T_ARRIVE, T_END, 0.3),
                ),
            ),
            _synth_auto(
                "resonance_q",
                (
                    _hold(T_COLD, T_ARRIVE, 2.0),
                    _lin(T_ARRIVE, T_EUPH, 2.0, 2.8),
                    _hold(T_EUPH, T_END, 2.5),
                ),
            ),
            _ctrl(
                "send_db",
                (
                    _hold(T_COLD, T_ARRIVE, -14.0),
                    _lin(T_ARRIVE, T_EUPH, -14.0, -10.0),
                    _hold(T_EUPH, T_END, -10.0),
                ),
            ),
        ],
    )

    # --- SUB ---
    score.add_voice(
        "sub",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "sine",
            "filter_topology": "ladder",
            "cutoff_hz": 180.0,
            "resonance_q": 0.8,
            "filter_drive": 0.08,
            "voice_card_spread": 0.0,
            "attack": 0.05,
            "decay": 0.2,
            "sustain_level": 0.85,
            "release": 0.5,
        },
        effects=[
            EffectSpec("saturation", {"preset": "tube_warm", "mix": 0.2}),
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "lowpass", "cutoff_hz": 250.0, "slope_db_per_oct": 24}
                    ]
                },
            ),
        ],
        normalize_peak_db=-6.0,
        velocity_humanize=None,
        mix_db=-6.0,
        automation=[
            _synth_auto(
                "cutoff_hz",
                (
                    _hold(T_COLD, T_WARM, 150.0),
                    _exp(T_WARM, T_EUPH, 150.0, 220.0),
                    _hold(T_EUPH, T_END, 220.0),
                ),
            ),
            _ctrl(
                "mix_db",
                (
                    _hold(T_COLD, _pos(WARM_BAR + 4), -20.0),
                    _lin(_pos(WARM_BAR + 4), T_SEARCH, -20.0, -10.0),
                    _hold(T_SEARCH, T_ARRIVE, -10.0),
                    _lin(T_ARRIVE, T_EUPH, -10.0, -4.0),
                    _hold(T_EUPH, T_END, -4.0),
                ),
            ),
        ],
    )

    # --- LEAD ---
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "saw",
            "cutoff_hz": 3200.0,
            "resonance_q": 1.3,
            "filter_env_amount": 1.8,
            "filter_env_decay": 0.35,
            "hpf_cutoff_hz": 200.0,
            "hpf_resonance_q": 1.5,
            "keytrack": 0.5,
            "osc_asymmetry": 0.1,
            "osc_softness": 0.08,
            "voice_card_spread": 1.5,
            "voice_card_pitch_spread": 0.4,
            "vca_nonlinearity": 0.25,
            "attack": 0.03,
            "decay": 0.4,
            "sustain_level": 0.55,
            "release": 0.5,
        },
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "bell", "freq_hz": 2500.0, "gain_db": 2.0, "q": 1.2}
                    ]
                },
            ),
        ],
        normalize_lufs=-22.0,
        mix_db=-4.0,
        sends=[
            VoiceSend(target="hall", send_db=-8.0),
            VoiceSend(
                target="delay",
                send_db=-12.0,
                automation=[
                    AutomationSpec(
                        target=AutomationTarget(kind="control", name="send_db"),
                        segments=(
                            _hold(T_COLD, T_ARRIVE, -30.0),
                            _lin(T_ARRIVE, T_EUPH, -30.0, -8.0),
                            _hold(T_EUPH, T_END, -8.0),
                        ),
                    ),
                ],
            ),
        ],
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        velocity_to_params={
            # Narrower range so the "spacious descent" (bars 65-68) doesn't
            # swing brightness too wildly across its 0.55-0.95 velocity span.
            "cutoff_hz": VelocityParamMap(min_value=3000.0, max_value=3700.0),
            "filter_env_amount": VelocityParamMap(min_value=1.5, max_value=2.0),
        },
        automation=[
            _synth_auto(
                "vca_nonlinearity",
                (
                    _lin(T_ARRIVE, T_EUPH, 0.2, 0.35),
                    _hold(T_EUPH, T_END, 0.35),
                ),
            ),
            _synth_auto(
                "cutoff_hz",
                (
                    _hold(T_ARRIVE, T_EUPH, 3200.0),
                    _exp(T_EUPH, T_END, 3200.0, 4200.0),
                ),
            ),
            _synth_auto(
                "filter_env_amount",
                (
                    _hold(T_ARRIVE, T_EUPH, 1.8),
                    _lin(T_EUPH, T_END, 1.8, 2.5),
                ),
            ),
            _synth_auto(
                "resonance_q",
                (
                    _hold(T_ARRIVE, T_EUPH, 1.3),
                    _lin(T_EUPH, _pos(EUPH_BAR + 16), 1.3, 2.0),
                    _lin(_pos(EUPH_BAR + 16), T_END, 2.0, 1.5),
                ),
            ),
            _synth_auto(
                "hpf_cutoff_hz",
                (
                    _hold(T_ARRIVE, T_EUPH, 200.0),
                    _exp(T_EUPH, T_END, 200.0, 140.0),
                ),
            ),
            _ctrl(
                "pan",
                (
                    _hold(T_ARRIVE, T_EUPH, 0.0),
                    _lfo(T_EUPH, T_END, 0.08, 0.15),
                ),
            ),
            _ctrl(
                "send_db",
                (
                    _hold(T_ARRIVE, T_EUPH, -8.0),
                    _lin(T_EUPH, _pos(63), -8.0, -6.0),
                    _lin(_pos(63), _pos(64), -6.0, -7.5),  # dip into mid-Euph lull
                    _hold(_pos(64), _pos(65), -7.5),
                    _lin(_pos(65), _pos(EUPH_BAR + 16), -7.5, -5.0),  # recover & climax
                    _lin(_pos(EUPH_BAR + 16), T_END, -5.0, -7.0),
                ),
            ),
        ],
    )

    # --- TEXTURE ---
    score.add_voice(
        "texture",
        synth_defaults={
            "engine": "filtered_stack",
            "waveform": "saw",
            "n_harmonics": 32,
            "cutoff_hz": 1200.0,
            "resonance_q": 0.8,
            "filter_morph": 0.6,
            "voice_card_spread": 1.8,
            "voice_card_pitch_spread": 0.5,
            "osc_shape_drift": 0.4,
            "attack": 2.0,
            "decay": 1.0,
            "sustain_level": 0.6,
            "release": 3.0,
        },
        effects=[
            EffectSpec("chorus", {"rate_hz": 0.25, "depth_ms": 3.0, "mix": 0.35}),
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 300.0,
                            "slope_db_per_oct": 12,
                        },
                        {
                            "kind": "lowpass",
                            "cutoff_hz": 4000.0,
                            "slope_db_per_oct": 12,
                        },
                    ]
                },
            ),
        ],
        normalize_lufs=-26.0,
        mix_db=-8.0,
        sends=[VoiceSend(target="hall", send_db=-4.0)],
        velocity_humanize=None,
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        automation=[
            _synth_auto(
                "cutoff_hz",
                (
                    _hold(T_COLD, T_WARM, 1000.0),
                    _exp(T_WARM, T_SEARCH, 1000.0, 1300.0),
                    _exp(T_SEARCH, T_ARRIVE, 1300.0, 900.0),  # darker in search
                    _exp(T_ARRIVE, T_EUPH, 900.0, 1600.0),
                    _hold(T_EUPH, T_END, 1600.0),
                ),
            ),
            _ctrl(
                "mix_db",
                (
                    # Cold awakening: silent at first, barely audible crawl in
                    _hold(T_COLD, _pos(5), -20.0),
                    _lin(_pos(5), T_WARM, -20.0, -10.0),
                    _lin(T_WARM, T_SEARCH, -10.0, -7.0),
                    _hold(T_SEARCH, T_ARRIVE, -7.0),
                    _lin(T_ARRIVE, T_EUPH, -7.0, -5.0),
                    # Mid-Euphoria lull and final breath
                    _hold(T_EUPH, _pos(63), -5.0),
                    _lin(_pos(63), _pos(64), -5.0, -6.5),  # dip into lull
                    _hold(_pos(64), _pos(65), -6.5),
                    _lin(_pos(65), _pos(66), -6.5, -5.0),  # return
                    _hold(_pos(66), _pos(78), -5.0),
                    _lin(_pos(78), _pos(80), -5.0, -6.0),  # final breath
                    _lin(_pos(80), T_END, -6.0, -5.0),
                ),
            ),
            _ctrl("pan", (_lfo(T_COLD, T_END, 0.05, 0.2),)),
            _ctrl(
                "send_db",
                (
                    _hold(T_COLD, T_ARRIVE, -4.0),
                    _lin(T_ARRIVE, T_EUPH, -4.0, -2.0),
                    _hold(T_EUPH, T_END, -2.0),
                ),
            ),
        ],
    )

    # --- GHOST (counter-melody echo) ---
    score.add_voice(
        "ghost",
        synth_defaults={
            "engine": "filtered_stack",
            "waveform": "saw",
            "n_harmonics": 24,
            "cutoff_hz": 1800.0,
            "resonance_q": 1.0,
            "filter_morph": 0.3,
            "voice_card_spread": 2.0,
            "voice_card_pitch_spread": 0.6,
            "attack": 0.8,
            "decay": 1.5,
            "sustain_level": 0.4,
            "release": 2.5,
        },
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
                        {
                            "kind": "lowpass",
                            "cutoff_hz": 3000.0,
                            "slope_db_per_oct": 12,
                        },
                    ]
                },
            ),
        ],
        normalize_lufs=-26.0,
        mix_db=-10.0,
        sends=[VoiceSend(target="hall", send_db=-2.0)],
        velocity_humanize=None,
        automation=[
            _ctrl(
                "mix_db",
                (
                    _hold(T_COLD, T_ARRIVE, -30.0),
                    _lin(T_ARRIVE, _pos(ARRIVE_BAR + 4), -30.0, -10.0),
                    _hold(_pos(ARRIVE_BAR + 4), T_EUPH, -10.0),
                    _lin(T_EUPH, _pos(EUPH_BAR + 16), -10.0, -7.0),
                    _lin(_pos(EUPH_BAR + 16), T_END, -7.0, -12.0),
                ),
            ),
        ],
    )

    _place_pad(score)
    _place_bass(score)
    _place_sub(score)
    _place_lead(score)
    _place_lead_motif_echoes(score)
    _place_lead_thicken(score)
    _place_ghost(score)
    _place_texture(score)
    return score


# ---------------------------------------------------------------------------
# Note placement
# ---------------------------------------------------------------------------


def _place_pad(score: Score) -> None:
    # Cold (1-8): clean SVF, 2-bar chords — no drift (digital/precise narrative)
    for i, ch in enumerate(PROG_COLD):
        t = _pos(COLD_BAR + i * 2)
        for p, db in ch():
            score.add_note("pad", start=t, duration=2 * BAR - 0.1, partial=p, amp_db=db)

    # Warming (9-20): ladder, 2-bar chords with overlap, strum, and gentle drift
    syn_w = {"filter_topology": "ladder", "bass_compensation": 0.5}
    overlap_w = 0.3 * BAR
    warm_dur = 2 * BAR + overlap_w - 0.1
    warm_prog = PROG_WARM_1 + list(PROG_WARM_2)
    warm_drift = progression_drift_lanes(
        warm_prog,
        chord_dur=warm_dur,
        attraction=0.6,
        wander=0.1,
        smoothness=0.85,
        seed_base=100,
        target_time=2 * BAR,  # glide reaches end_ratio at the next chord boundary
        # Sparse accents: iv→V in the first cycle, vi→I_open at start of second
        glide_transitions={1, 3},
    )
    bar = WARM_BAR
    for i, ch in enumerate(warm_prog):
        t = _pos(bar)
        events = drifted_chord_events(ch(), warm_dur, warm_drift[i])
        strummed = strum(Phrase(events=events), spread_ms=20, direction="down")
        score.add_phrase("pad", strummed, start=t, synth=syn_w)
        bar += 2

    # Searching (21-34): utonal/open fifths, 2-bar chords, drifty strum,
    # unstable harmonically-unsettled drift
    syn_s = {"filter_topology": "ladder", "bass_compensation": 0.3}
    overlap_s = 0.5 * BAR
    search_dur = 2 * BAR + overlap_s - 0.1
    search_drift = progression_drift_lanes(
        PROG_SEARCH,
        chord_dur=search_dur,
        # Lower attraction + higher wander so the glides feel genuinely
        # unsettled — these are now placed accents rather than continuous,
        # so each one can carry more character.
        attraction=0.15,
        wander=0.55,
        smoothness=0.65,
        seed_base=200,
        target_time=2 * BAR,
        # Two accents: utonal_iv→utonal_i (dark-descending) and
        # utonal_i→open_fifth_i (exposed/unsettled).
        glide_transitions={1, 4},
    )
    bar = SEARCH_BAR
    for i, ch in enumerate(PROG_SEARCH):
        t = _pos(bar)
        events = drifted_chord_events(ch(), search_dur, search_drift[i])
        # "out" strum keeps the attacks consistent bar-to-bar; "random"
        # produced a stutter-like feel because every chord had a different
        # pattern and the Searching overlap=0.5*BAR let them pile up.
        strummed = strum(Phrase(events=events), spread_ms=25, direction="out")
        score.add_phrase("pad", strummed, start=t, synth=syn_s)
        bar += 2

    # Arriving (35-50): rich voicings, 4-bar chords with overlap,
    # settling toward purity
    syn_a = {
        "filter_topology": "ladder",
        "bass_compensation": 0.6,
        "osc_dc_offset": 0.15,
    }
    overlap_a = 0.5 * BAR
    arrive_dur = 4 * BAR + overlap_a - 0.1
    arrive_drift = progression_drift_lanes(
        PROG_ARRIVE,
        chord_dur=arrive_dur,
        attraction=0.7,
        wander=0.1,
        smoothness=0.85,
        seed_base=300,
        target_time=4 * BAR,
        # Single accent on I→IV at the start — an early "locking in" gesture
        glide_transitions={0},
    )
    for i, ch in enumerate(PROG_ARRIVE):
        t = _pos(ARRIVE_BAR + i * 4)
        events = drifted_chord_events(ch(), arrive_dur, arrive_drift[i])
        strummed = strum(Phrase(events=events), spread_ms=20, direction="down")
        score.add_phrase("pad", strummed, start=t, synth=syn_a)

    # Euphoria (51-82): 3-bar chords → euphoric ending, outward strum,
    # locking into pure JI
    syn_e = {
        "filter_topology": "ladder",
        "bass_compensation": 0.65,
        "osc_dc_offset": 0.2,
        "osc_shape_drift": 0.35,
    }
    overlap_e = 0.4 * BAR
    euph1_dur = 3.5 * BAR + overlap_e - 0.1
    euph1_drift = progression_drift_lanes(
        PROG_EUPH_1,
        chord_dur=euph1_dur,
        attraction=0.8,
        wander=0.05,
        smoothness=0.9,
        seed_base=400,
        target_time=3 * BAR,  # next chord attacks 3 bars in; overlap runs 0.5 more
        # No glides — the "locked in" section breathes cleanly between chords
        glide_transitions=set(),
    )
    bar = EUPH_BAR
    for i, ch in enumerate(PROG_EUPH_1):
        events = drifted_chord_events(
            ch(), euph1_dur, euph1_drift[i], amp_db_offset=1.0
        )
        strummed = strum(Phrase(events=events), spread_ms=25, direction="out")
        score.add_phrase("pad", strummed, start=_pos(bar), synth=syn_e)
        bar += 3

    euph2_dur = 3.5 * BAR + overlap_e - 0.1
    euph2_drift = progression_drift_lanes(
        PROG_EUPH_2,
        chord_dur=euph2_dur,
        attraction=0.8,
        wander=0.05,
        smoothness=0.9,
        seed_base=500,
        target_time=3 * BAR,
        # Single accent on V→I toward the end — one color gesture
        glide_transitions={2},
    )
    for i, ch in enumerate(PROG_EUPH_2):
        events = drifted_chord_events(
            ch(), euph2_dur, euph2_drift[i], amp_db_offset=1.0
        )
        strummed = strum(Phrase(events=events), spread_ms=25, direction="out")
        score.add_phrase("pad", strummed, start=_pos(bar), synth=syn_e)
        bar += 3

    # Final chords — compute drift for EUPH_END; last chord gets None
    euph_end_durs = [
        4 * BAR + overlap_e - 0.1,
        4 * BAR + overlap_e - 0.1,
        5 * BAR + overlap_e - 0.1,
    ]
    # Use first chord's duration for drift computation (close enough for the
    # penultimate transition; the last chord has no drift anyway)
    euph_end_drift = progression_drift_lanes(
        PROG_EUPH_END,
        chord_dur=euph_end_durs[0],
        attraction=0.8,
        wander=0.05,
        smoothness=0.9,
        seed_base=600,
        target_time=4 * BAR,  # chord-to-chord boundary in final sequence
        # No glides — the final resolution should be clean, no blur
        glide_transitions=set(),
    )
    for i, ch in enumerate(PROG_EUPH_END):
        dur = euph_end_durs[i]
        events = drifted_chord_events(ch(), dur, euph_end_drift[i], amp_db_offset=1.0)
        strummed = strum(Phrase(events=events), spread_ms=25, direction="out")
        score.add_phrase("pad", strummed, start=_pos(bar), synth=syn_e)
        bar += 4 if ch != PROG_EUPH_END[-1] else 5


def _place_bass(score: Score) -> None:
    """Bass — chord tones only, *2 minimum (~92 Hz floor)."""

    def _bn(
        bar: int,
        beat: float,
        partial: float,
        dur: float,
        db: float = -6.0,
        vel: float = 0.85,
    ) -> None:
        _add_bounded_note(
            score,
            "bass",
            bar,
            beat,
            partial,
            dur,
            amp_db=db,
            velocity=vel,
            time_limit=T_END,
        )

    # Cold: sustained roots
    for i, root in enumerate([P1 * 2, P_5_3, P_4_3, P1 * 2]):
        _bn(COLD_BAR + i * 2, 1.0, root, 7.5, -8.0)

    # Warming: root + fifth per chord
    warm_roots = [
        (P1, P_3_2),
        (P_4_3, P1 * 2),
        (P_3_2, P1 * 2),
        (P_5_3, P_3_2 * 2),
        (P1 * 2, P_5_4 * 2),
        (P_5_3, P_3_2 * 2),
    ]
    bar = WARM_BAR
    for root, fifth in warm_roots:
        _bn(bar, 1.0, root, 3.5, -6.0, 0.85)
        _bn(bar, 4.5, fifth, 2.0, -8.0, 0.7)
        bar += 2

    # Searching: sparse open fifths, exposed
    search_roots = [P1 * 2, P_4_3, P1 * 2, P_4_3, P1 * 2, P1 * 2, P_4_3]
    bar = SEARCH_BAR
    for root in search_roots:
        _bn(bar, 1.0, root, 6.0, -7.0, 0.75)
        bar += 2

    # Arriving: chord tones, 2-4 per 4-bar group, varied rhythm
    arrive_notes = [
        (35, [(1, P1 * 2, 4, 0.88), (3, P_5_4 * 2, 2, 0.72), (5, P1 * 2, 4, 0.82)]),
        (
            39,
            [
                (1, P_4_3, 4, 0.88),
                (3.5, P_5_3 * 2, 1.5, 0.7),
                (5, P_4_3 * 2, 3, 0.82),
                (7.5, P_5_3, 2, 0.68),
            ],
        ),
        (43, [(1, P_3_2, 4, 0.88), (3, P_15_8 * 2, 2, 0.7), (5, P_3_2 * 2, 4, 0.82)]),
        (
            47,
            [
                (1, P_5_3, 3, 0.85),
                (2.5, P1 * 2, 2, 0.72),
                (4.5, P_5_3, 2, 0.7),
                (6, P_5_3 * 2, 3, 0.78),
            ],
        ),
    ]
    for bar, notes in arrive_notes:
        for beat, partial, dur, vel in notes:
            _bn(bar, beat, partial, dur, -5.5, vel)

    # Euphoria: chord tones, varied rhythm — avoids repeating the same shape
    euph_notes = [
        (51, [(1, P_4_3 * 2, 5, 0.9), (5, P_5_3 * 2, 3, 0.75)]),
        (
            54,
            [
                (1, P_3_2 * 2, 4, 0.88),
                (3.5, P_15_8 * 2, 2, 0.72),
                (6, P_3_2 * 2, 2.5, 0.7),
            ],
        ),
        (57, [(1, P1 * 2, 5, 0.9), (3, P_5_4 * 2, 2, 0.72), (5, P_3_2 * 2, 3, 0.78)]),
        (60, [(1.5, P_5_3 * 2, 4, 0.82), (5, P1 * 2, 3, 0.75)]),
        (
            63,
            [
                (1, P1 * 2, 3, 0.92),
                (2.5, P_3_2 * 2, 2, 0.75),
                (4, P_5_4 * 2, 2, 0.7),
                (6, P1 * 2, 3, 0.78),
            ],
        ),
        (66, [(1, P_4_3 * 2, 6, 0.9), (4.5, P1 * 2, 2, 0.72)]),
        (
            69,
            [
                (1, P_3_2 * 2, 3, 0.88),
                (3, P_9_8 * 2, 2, 0.72),
                (5.5, P_3_2 * 2, 3, 0.75),
            ],
        ),
        (72, [(1, P1 * 2, 7, 0.92), (4, P_5_4 * 2, 3, 0.75)]),
    ]
    for bar, notes in euph_notes:
        for beat, partial, dur, vel in notes:
            _bn(bar, beat, partial, dur, -5.0, vel)

    # Final sustained root
    _bn(END_BAR - 5, 1.0, P1 * 2, 20.0, -5.0)


def _place_sub(score: Score) -> None:
    """Sine sub — enters mid-Warming, drops to drone in Searching, blooms in Euphoria."""
    roots_warm = [P1, P_4_3 / 2, P_3_2 / 2, P_5_3 / 2]
    bar = WARM_BAR + 4
    for root in roots_warm:
        if bar >= SEARCH_BAR:
            break
        score.add_note(
            "sub", start=_pos(bar), duration=4 * BAR - 0.1, partial=root, amp_db=-8.0
        )
        bar += 4

    # Searching: just the root drone
    score.add_note(
        "sub",
        start=T_SEARCH,
        duration=(T_ARRIVE - T_SEARCH) - 0.1,
        partial=P1,
        amp_db=-7.0,
    )

    # Arriving
    for i, root in enumerate([P1, P_4_3 / 2, P_3_2 / 2, P_5_3 / 2]):
        score.add_note(
            "sub",
            start=_pos(ARRIVE_BAR + i * 4),
            duration=4 * BAR - 0.1,
            partial=root,
            amp_db=-6.0,
        )

    # Euphoria
    euph_roots = [P_4_3 / 2, P_3_2 / 2, P1, P_5_3 / 2, P1, P_4_3 / 2, P_3_2 / 2, P1]
    bar = EUPH_BAR
    for root in euph_roots:
        if bar >= END_BAR - 3:
            break
        score.add_note(
            "sub", start=_pos(bar), duration=3 * BAR - 0.1, partial=root, amp_db=-5.0
        )
        bar += 3

    # Final drone
    score.add_note(
        "sub", start=_pos(END_BAR - 5), duration=5 * BAR + 2.0, partial=P1, amp_db=-5.0
    )


def _place_lead(score: Score) -> None:
    """CS80 brass lead — tentative preview in Searching, full entry at Arriving."""

    def _n(
        base_bar: int,
        bar_off: int,
        beat: float,
        partial: float,
        dur: float,
        db: float,
        vel: float,
        pm: PitchMotionSpec | None = None,
    ) -> None:
        _add_bounded_note(
            score,
            "lead",
            base_bar + bar_off,
            beat,
            partial,
            dur,
            amp_db=db,
            velocity=vel,
            time_limit=T_END + RELEASE_TAIL_S,
            pitch_motion=pm,
        )

    # Searching: tentative fragment, solo, lower register, sparse
    _glide_flat = PitchMotionSpec.ratio_glide(start_ratio=0.996)
    _vib_tentative = PitchMotionSpec.vibrato(depth_ratio=0.003, rate_hz=4.0)
    _n(SEARCH_BAR, 4, 2.0, P_3_2 * 4, 3.0, -8.0, 0.6, _glide_flat)
    _n(SEARCH_BAR, 5, 2.0, P_5_4 * 4, 2.0, -9.0, 0.55)
    _n(SEARCH_BAR, 6, 1.0, P_7_4 * 4, 4.0, -7.5, 0.65, _vib_tentative)
    _n(SEARCH_BAR, 8, 2.0, P1 * 4, 3.0, -8.5, 0.58, _glide_flat)
    _n(SEARCH_BAR, 9, 2.0, P_5_4 * 4, 2.5, -9.0, 0.55)
    _n(SEARCH_BAR, 10, 1.0, P_3_2 * 4, 5.0, -7.0, 0.62, _vib_tentative)

    # Arriving melody (bar_off from ARRIVE_BAR)
    arrive = [
        (0, 1.0, P_7_4 * 8, 2.5, -6.0, 0.85),
        (0, 3.5, P_5_3 * 8, 0.5, -9.0, 0.6),
        (0, 4.0, P_3_2 * 8, 2.5, -7.0, 0.78),
        (1, 2.5, P_5_4 * 8, 1.5, -8.0, 0.68),
        (1, 4.0, P_4_3 * 8, 2.5, -6.5, 0.82),
        (2, 3.0, P_5_4 * 8, 1.0, -8.5, 0.62),
        (2, 4.0, P_3_2 * 8, 3.0, -6.0, 0.85),
        (3, 3.0, P_5_3 * 8, 0.8, -8.0, 0.65),
        (3, 4.0, P_7_4 * 8, 3.5, -5.5, 0.92),
        (5, 1.5, P_5_4 * 8, 1.5, -7.5, 0.75),
        (5, 3.0, P_4_3 * 8, 0.5, -9.0, 0.58),
        (5, 3.5, P_3_2 * 8, 2.0, -6.5, 0.85),
        (6, 1.5, P_5_3 * 8, 2.5, -6.0, 0.88),
        (6, 4.0, P_3_2 * 8, 0.5, -8.5, 0.6),
        (6, 4.5, P_7_4 * 8, 3.0, -5.5, 0.92),
        (8, 1.0, P_5_3 * 8, 1.5, -7.0, 0.75),
        (8, 2.5, P_3_2 * 8, 0.5, -9.0, 0.58),
        (8, 3.0, P_5_4 * 8, 3.0, -6.5, 0.82),
        (9, 2.0, P_4_3 * 8, 1.0, -8.0, 0.65),
        (9, 3.0, P_5_4 * 8, 0.5, -9.0, 0.55),
        (9, 3.5, P1 * 8, 5.0, -6.0, 0.88),
        (11, 3.0, P_5_4 * 8, 0.5, -9.0, 0.55),
        (11, 3.5, P_3_2 * 8, 1.5, -7.5, 0.72),
    ]
    for bo, bt, p, dur, db, vel in arrive:
        pm: PitchMotionSpec | None = None
        if bo == 9 and dur >= 5.0:
            pm = PitchMotionSpec.vibrato(depth_ratio=0.005, rate_hz=5.0)
        elif dur >= 3.0:
            pm = PitchMotionSpec.vibrato(depth_ratio=0.004, rate_hz=5.0)
        elif dur >= 2.5:
            pm = PitchMotionSpec.vibrato(depth_ratio=0.004, rate_hz=4.8)
        _n(ARRIVE_BAR, bo, bt, p, dur, db, vel, pm)

    # Euphoria melody (bar_off from EUPH_BAR)
    euphoria = [
        (0, 1.0, P_5_4 * 8, 1.0, -7.0, 0.78),
        (0, 2.0, P_4_3 * 8, 0.5, -9.0, 0.6),
        (0, 2.5, P_3_2 * 8, 1.5, -6.5, 0.85),
        (0, 4.0, P_5_3 * 8, 0.5, -8.0, 0.65),
        (0, 4.5, P_7_4 * 8, 2.5, -5.0, 0.95),
        (1, 3.5, P_5_3 * 8, 0.5, -8.5, 0.6),
        (1, 4.0, P2 * 8, 2.5, -5.5, 0.92),
        (3, 1.0, P_7_4 * 8, 1.0, -6.0, 0.85),
        (3, 2.0, P_5_3 * 8, 0.5, -8.0, 0.65),
        (3, 2.5, P_3_2 * 8, 1.5, -6.5, 0.82),
        (3, 4.0, P_5_4 * 8, 0.8, -7.5, 0.7),
        (3, 4.8, P_4_3 * 8, 2.0, -6.0, 0.88),
        (4, 2.8, P_5_4 * 8, 0.5, -9.0, 0.55),
        (4, 3.3, P_3_2 * 8, 0.5, -8.0, 0.62),
        (4, 3.8, P_5_4 * 8, 2.5, -6.5, 0.82),
        (6, 1.0, P2 * 8, 1.5, -5.5, 0.92),
        (6, 2.5, P_7_4 * 8, 0.5, -8.0, 0.62),
        (6, 3.0, P_3_2 * 4, 2.0, -6.5, 0.8),
        (7, 1.0, P_5_3 * 8, 0.5, -8.5, 0.6),
        (7, 1.5, P_7_4 * 8, 2.5, -5.0, 0.95),
        (7, 4.0, P_5_3 * 8, 0.5, -8.0, 0.65),
        (7, 4.5, P2 * 8, 3.0, -5.5, 0.9),
        (10, 1.0, P_5_4 * 8, 0.8, -6.0, 0.88),
        (10, 1.8, P_5_4 * 8, 0.3, -10.0, 0.5),
        (10, 2.2, P_3_2 * 8, 1.5, -5.5, 0.92),
        (10, 3.7, P_5_3 * 8, 0.5, -7.5, 0.68),
        (10, 4.2, P_7_4 * 8, 1.5, -5.0, 0.95),
        (11, 1.7, P_7_4 * 8, 0.3, -9.0, 0.55),
        (11, 2.0, P_5_3 * 8, 0.5, -8.0, 0.62),
        (11, 2.5, P2 * 8, 2.5, -5.0, 0.92),
        # Spacious descent
        (14, 2.0, P_7_4 * 8, 2.5, -5.5, 0.9),
        (15, 1.0, P_5_3 * 8, 2.0, -6.0, 0.85),
        (15, 3.0, P_3_2 * 8, 0.5, -8.5, 0.6),
        (15, 3.5, P_5_3 * 8, 1.0, -7.0, 0.72),
        (16, 1.0, P_3_2 * 8, 3.0, -5.5, 0.88),
        (17, 1.0, P_5_4 * 8, 2.5, -6.0, 0.82),
        (17, 3.5, P_4_3 * 8, 0.5, -8.5, 0.58),
        (17, 4.0, P_5_4 * 8, 1.0, -7.0, 0.7),
        (18, 1.0, P1 * 8, 4.5, -5.0, 0.92),
        # Final
        (21, 1.0, P_3_2 * 8, 0.5, -8.5, 0.6),
        (21, 1.5, P_5_4 * 8, 0.5, -8.0, 0.65),
        (21, 2.0, P1 * 8, 2.0, -6.0, 0.85),
        (21, 4.0, P_5_4 * 8, 0.5, -9.0, 0.55),
        (21, 4.5, P2 * 8, 6.0, -6.0, 0.82),
    ]
    for bo, bt, p, dur, db, vel in euphoria:
        pm = None
        if bo == 21 and dur >= 6.0:
            # Final held note — slow, wide vibrato
            pm = PitchMotionSpec.vibrato(depth_ratio=0.007, rate_hz=4.5)
        elif p == P2 * 8 and dur >= 2.5:
            # Climactic high notes
            pm = PitchMotionSpec.vibrato(depth_ratio=0.006, rate_hz=5.5)
        elif dur >= 2.5:
            # Sustained notes — confident vibrato
            pm = PitchMotionSpec.vibrato(depth_ratio=0.005, rate_hz=5.5)
        elif dur >= 1.5:
            # Medium notes — lighter vibrato
            pm = PitchMotionSpec.vibrato(depth_ratio=0.003, rate_hz=5.0)
        _n(EUPH_BAR, bo, bt, p, dur, db, vel, pm)


def _place_lead_motif_echoes(score: Score) -> None:
    """Transformed restatements of the Arrive opening motif in Euphoria gaps.

    The core gesture is the descending septimal line from the first two bars
    of Arriving: 7/4 → 5/3(grace) → 3/2 → 5/4 → 4/3.  Three transformed
    versions are placed into quiet moments of the Euphoria melody so the
    lead has thematic continuity without sounding looped.
    """

    # Core motif: the Arrive opening gesture (relative times, octave 8).
    # Uses amp (not amp_db) so augment/diminish can use dataclasses.replace
    # without triggering NoteEvent's amp/amp_db mutual-exclusion check.
    core = Phrase(
        events=(
            NoteEvent(
                start=0.0,
                duration=2.5 * BEAT,
                partial=P_7_4 * 8,
                amp=db_to_amp(-7.5),
                velocity=0.78,
            ),
            NoteEvent(
                start=2.5 * BEAT,
                duration=0.5 * BEAT,
                partial=P_5_3 * 8,
                amp=db_to_amp(-10.0),
                velocity=0.55,
            ),
            NoteEvent(
                start=3.0 * BEAT,
                duration=2.5 * BEAT,
                partial=P_3_2 * 8,
                amp=db_to_amp(-8.0),
                velocity=0.72,
            ),
            NoteEvent(
                start=5.5 * BEAT,
                duration=1.5 * BEAT,
                partial=P_5_4 * 8,
                amp=db_to_amp(-9.0),
                velocity=0.62,
            ),
            NoteEvent(
                start=7.0 * BEAT,
                duration=2.5 * BEAT,
                partial=P_4_3 * 8,
                amp=db_to_amp(-7.5),
                velocity=0.75,
            ),
        )
    )

    # Echo 1 (bar_off ~2, beat 2): augmented 1.3x, sits in the gap before bar 3
    echo1 = augment(core, 1.3)
    score.add_phrase("lead", echo1, start=_pos(EUPH_BAR + 2, 2.0))

    # Echo 2 (bar_off ~9, beat 1): diminished 0.8x, compressed into the gap at bar 9
    echo2 = diminish(core, 1.25)
    score.add_phrase("lead", echo2, start=_pos(EUPH_BAR + 9, 1.0))

    # Echo 3 (bar_off ~19, beat 2): original tempo, fills the quiet before the final
    score.add_phrase("lead", core, start=_pos(EUPH_BAR + 19, 2.0))


def _place_lead_thicken(score: Score) -> None:
    """Micro-detuned lead copies in late Euphoria for unison thickness."""
    # Sustained euphoria notes from bar_off >= 6 with dur >= 1.5
    euph_thick: list[tuple[int, float, float, float, float]] = [
        # (bar_off, beat, partial, dur_beats, amp_db)
        (6, 1.0, P2 * 8, 1.5, -5.5),
        (6, 3.0, P_3_2 * 4, 2.0, -6.5),
        (7, 1.5, P_7_4 * 8, 2.5, -5.0),
        (7, 4.5, P2 * 8, 3.0, -5.5),
        (10, 2.2, P_3_2 * 8, 1.5, -5.5),
        (10, 4.2, P_7_4 * 8, 1.5, -5.0),
        (11, 2.5, P2 * 8, 2.5, -5.0),
        (14, 2.0, P_7_4 * 8, 2.5, -5.5),
        (15, 1.0, P_5_3 * 8, 2.0, -6.0),
        (16, 1.0, P_3_2 * 8, 3.0, -5.5),
        (17, 1.0, P_5_4 * 8, 2.5, -6.0),
        (18, 1.0, P1 * 8, 4.5, -5.0),
        (21, 2.0, P1 * 8, 2.0, -6.0),
        (21, 4.5, P2 * 8, 6.0, -6.0),
    ]
    # Build phrase with times relative to the thickening start point (bar_off 6)
    base_time = _pos(EUPH_BAR + 6)
    events = tuple(
        NoteEvent(
            start=_pos(EUPH_BAR + bo, bt) - base_time,
            duration=dur * BEAT,
            partial=p,
            amp_db=db,
        )
        for bo, bt, p, dur, db in euph_thick
    )
    phrase = Phrase(events=events)
    copies = thicken(
        phrase,
        n=3,
        detune_cents=5.0,
        spread_ms=6.0,  # tighter — was 12.0, which read as "loose" at bar 57
        stereo_width=0.4,
        amp_taper_db=-3.0,
        seed=42,
    )
    for copy in copies:
        for ev in copy.phrase.events:
            t = base_time + ev.start
            if ev.partial and t < T_END + RELEASE_TAIL_S:
                score.add_note(
                    "lead",
                    start=t,
                    duration=ev.duration,
                    partial=ev.partial,
                    amp_db=-8.0 + copy.amp_offset_db,
                    synth=ev.synth,
                )


def _place_ghost(score: Score) -> None:
    """Ghostly echoes of the lead — octave lower, delayed, sparse."""
    # Selected lead moments echoed as ghost tones (~2 beats delayed, octave down)
    # (bar_base, bar_off, beat, partial_octave_below, dur_beats, amp_db)
    ghost_notes: list[tuple[int, int, float, float, float, float]] = [
        # Arriving echoes — sparse, most memorable lead moments
        (ARRIVE_BAR, 0, 3.0, P_7_4 * 4, 4.0, -8.0),
        (ARRIVE_BAR, 3, 2.0, P_7_4 * 4, 5.0, -7.0),
        (ARRIVE_BAR, 6, 2.5, P_7_4 * 4, 4.0, -7.5),
        (ARRIVE_BAR, 9, 1.5, P1 * 4, 6.0, -7.0),
        # Euphoria echoes — more present
        (EUPH_BAR, 0, 2.5, P_7_4 * 4, 4.0, -7.0),
        (EUPH_BAR, 3, 2.8, P_4_3 * 4, 3.5, -7.5),
        (EUPH_BAR, 7, 3.5, P_7_4 * 4, 4.0, -6.5),
        (EUPH_BAR, 11, 4.5, P2 * 4, 4.0, -7.0),
        (EUPH_BAR, 14, 4.0, P_7_4 * 4, 4.0, -7.0),
        (EUPH_BAR, 18, 3.0, P1 * 4, 6.0, -6.5),
        (EUPH_BAR, 21, 4.0, P1 * 4, 8.0, -7.0),
    ]
    for base, bo, bt, p, dur, db in ghost_notes:
        t = _pos(base + bo, bt)
        if t < T_END + GHOST_RELEASE_TAIL_S:
            score.add_note(
                "ghost",
                start=t,
                duration=dur * BEAT,
                partial=p,
                amp_db=db,
            )


def _place_texture(score: Score) -> None:
    """Drone atmosphere — darkens in Searching, brightens in Arriving."""
    layers = [
        (T_COLD, T_SEARCH, P1 * 2, -13.0),
        (T_COLD, T_SEARCH, P_3_2 * 2, -15.0),
        (T_WARM, T_ARRIVE, P_5_4 * 2, -15.0),
        # Searching: drop the third, add utonal colour
        (T_SEARCH, T_ARRIVE, P1 * 2, -11.0),
        (T_SEARCH, T_ARRIVE, P_3_2 * 2, -13.0),
        (T_SEARCH, T_ARRIVE, P_8_5 * 2, -14.0),  # minor sixth — dark
        # Arriving onward
        (T_ARRIVE, T_END, P_7_4 * 2, -14.0),
        # Euphoria: full bloom
        (T_EUPH, T_END + TEXTURE_RELEASE_TAIL_S, P1 * 2, -10.0),
        (T_EUPH, T_END + TEXTURE_RELEASE_TAIL_S, P_5_4 * 2, -12.0),
        (T_EUPH, T_END + TEXTURE_RELEASE_TAIL_S, P_3_2 * 2, -12.0),
        (T_EUPH, T_END + TEXTURE_RELEASE_TAIL_S, P_7_4 * 2, -11.0),
        # High shimmer
        (_pos(EUPH_BAR + 8), T_END + TEXTURE_RELEASE_TAIL_S, P1 * 8, -16.0),
        (_pos(EUPH_BAR + 8), T_END + TEXTURE_RELEASE_TAIL_S, P_5_4 * 8, -17.0),
    ]
    for start, end, p, db in layers:
        score.add_note(
            "texture", start=start, duration=end - start, partial=p, amp_db=db
        )

    # Warming→Searching overshoot gesture: brief high filtered sweep emerging
    # from bar 20 beat 3 through the Searching boundary. Subtle, just a hint of
    # the instability to come.
    score.add_note(
        "texture",
        start=_pos(20, 3),
        duration=2 * BAR,
        partial=P_3_2 * 8,
        amp_db=-14.0,
    )

    # Searching→Arriving riser: high-register drone rising in amplitude from
    # bar 33 beat 1 through bar 34 beat 4. This is the "locking in" riser.
    score.add_note(
        "texture",
        start=_pos(33),
        duration=2 * BAR - 0.1,
        partial=P1 * 16,
        amp_db=-14.0,
    )

    # Arriving→Euphoria swell: texture push in last 2 bars of Arriving
    score.add_note(
        "texture",
        start=_pos(49),
        duration=2 * BAR - 0.1,
        partial=P_5_4 * 8,
        amp_db=-14.0,
    )

    # Euphoria 16-bar boundary shimmer (bar 66.5-67.5) — brief gesture
    score.add_note(
        "texture",
        start=_pos(66, 3),
        duration=BAR,
        partial=P_3_2 * 16,
        amp_db=-17.0,
    )

    # Euphoria shimmer — sparse high-register stochastic sparkle
    shimmer_start = _pos(EUPH_BAR + 8)
    shimmer_pool = TonePool.weighted(
        {
            P1 * 16: 3.0,
            P_5_4 * 16: 2.5,
            P_3_2 * 16: 2.5,
            P_7_4 * 16: 1.5,
            P1 * 8: 2.0,
            P_5_4 * 8: 1.5,
            P_3_2 * 8: 1.5,
        }
    )
    shimmer = stochastic_cloud(
        tones=shimmer_pool,
        duration=T_END - shimmer_start + TEXTURE_RELEASE_TAIL_S,
        density=[(0.0, 0.15), (0.4, 0.3), (0.8, 0.25), (1.0, 0.1)],
        amp_db_range=(-20.0, -14.0),
        note_dur_range=(1.5, 4.0),
        seed=17,
    )
    score.add_phrase("texture", shimmer, start=shimmer_start)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

PIECES: dict[str, PieceDefinition] = {
    "warming_up": PieceDefinition(
        name="warming_up",
        output_name="warming_up",
        build_score=build_score,
        sections=(
            PieceSection("cold", T_COLD, T_WARM),
            PieceSection("warming", T_WARM, T_SEARCH),
            PieceSection("searching", T_SEARCH, T_ARRIVE),
            PieceSection("arriving", T_ARRIVE, T_EUPH),
            PieceSection("euphoria", T_EUPH, T_END),
        ),
    ),
}
