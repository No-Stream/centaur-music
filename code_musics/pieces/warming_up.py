"""warming_up -- an analog polysynth waking from cold.

Starts precise and digital, gradually acquiring drift, character, and life
as the voices warm up.  A long build toward euphoria.

Showcases ladder filter, oscillator imperfections, voice card spread with
per-group pitch override, filter morphing, feedback path, serial HPF, and
VCA nonlinearity.

~4:20, 72 bars at 66 BPM, 7-limit JI in F#.
"""

from __future__ import annotations

from code_musics.automation import AutomationSegment, AutomationSpec, AutomationTarget
from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    VelocityHumanizeSpec,
)
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score, VoiceSend

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

BPM = 66.0
BEAT = 60.0 / BPM
BAR = 4.0 * BEAT
TOTAL_BARS = 72
F0 = 46.25  # F#1

# Section boundaries (bar numbers, 1-indexed)
A_START_BAR = 1
A_END_BAR = 9  # 8 bars: cold
B_START_BAR = 9
B_END_BAR = 29  # 20 bars: warming
C_START_BAR = 29
C_END_BAR = 45  # 16 bars: arriving
D_START_BAR = 45
D_END_BAR = 73  # 28 bars: euphoria


def _pos(bar: int, beat: float = 1.0) -> float:
    return (bar - 1) * BAR + (beat - 1) * BEAT


SEC_A = _pos(A_START_BAR)
SEC_B = _pos(B_START_BAR)
SEC_C = _pos(C_START_BAR)
SEC_D = _pos(D_START_BAR)
SEC_END = _pos(D_END_BAR)

# ---------------------------------------------------------------------------
# Tuning -- 7-limit JI partial ratios relative to F#
# ---------------------------------------------------------------------------

P1 = 1.0
P_9_8 = 9 / 8  # G#
P_9_7 = 9 / 7  # septimal major third (rich)
P_7_6 = 7 / 6  # septimal minor third
P_5_4 = 5 / 4  # A# (major third)
P_11_8 = 11 / 8  # undecimal tritone (colour)
P_4_3 = 4 / 3  # B (perfect fourth)
P_3_2 = 3 / 2  # C# (perfect fifth)
P_5_3 = 5 / 3  # D# (major sixth)
P_7_4 = 7 / 4  # E (septimal seventh)
P_15_8 = 15 / 8  # E#/F (major seventh)
P2 = 2.0  # F#'


# ---------------------------------------------------------------------------
# Chord voicings
# ---------------------------------------------------------------------------


def _chord_i_simple() -> list[tuple[float, float]]:
    """F#maj7(7) -- root position, simple spacing."""
    return [
        (P1 * 4, -6.0),
        (P_5_4 * 4, -8.0),
        (P_3_2 * 4, -8.0),
        (P_7_4 * 4, -9.0),
    ]


def _chord_i_open() -> list[tuple[float, float]]:
    """F#maj7(7) -- open voicing, wider register."""
    return [
        (P1 * 2, -7.0),  # F#2
        (P_5_4 * 4, -8.0),  # A#3
        (P_3_2 * 4, -8.5),  # C#4
        (P_7_4 * 4, -9.0),  # E4
        (P1 * 8, -10.0),  # F#5 (octave doubling)
    ]


def _chord_i_rich() -> list[tuple[float, float]]:
    """F#maj7(7) -- rich voicing with added 9th and 11th colour."""
    return [
        (P1 * 4, -6.0),
        (P_9_8 * 4, -10.0),  # added 9th
        (P_5_4 * 4, -8.0),
        (P_3_2 * 4, -8.0),
        (P_7_4 * 4, -8.5),
        (P_11_8 * 8, -12.0),  # undecimal colour, high and quiet
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
    """B major -- wide voicing with 7th."""
    return [
        (P_4_3 * 2, -7.0),  # B2
        (P_5_3 * 4, -8.0),  # D#4
        (P2 * 4, -8.0),  # F#4
        (P_5_4 * 8, -9.0),  # A#4
        (P_7_4 * 4 * 4 / 3, -11.0),  # septimal colour
    ]


def _chord_v() -> list[tuple[float, float]]:
    """C# (dominant)."""
    return [
        (P_3_2 * 4, -6.0),
        (P_15_8 * 4, -8.0),
        (P_9_8 * 8, -8.0),
    ]


def _chord_v_rich() -> list[tuple[float, float]]:
    """C# -- rich voicing with septimal flavour."""
    return [
        (P_3_2 * 2, -7.0),  # C#3
        (P_15_8 * 4, -8.0),  # E#4
        (P_9_8 * 8, -8.5),  # G#4
        (P_9_7 * 8, -10.0),  # septimal major 3rd above, colour
        (P_3_2 * 8, -10.5),  # C#5 octave doubling
    ]


def _chord_vi() -> list[tuple[float, float]]:
    """D# minor (approx)."""
    return [
        (P_5_3 * 4, -6.0),
        (P2 * 4, -8.0),
        (P_5_4 * 8, -9.0),
        (P_3_2 * 8, -10.0),
    ]


def _chord_vi_rich() -> list[tuple[float, float]]:
    """D# minor -- wide voicing with septimal colour."""
    return [
        (P_5_3 * 2, -7.0),  # D#3
        (P_7_6 * 8, -9.5),  # septimal minor 3rd, colour
        (P2 * 4, -8.0),  # F#4
        (P_5_4 * 8, -9.0),  # A#4
        (P_3_2 * 8, -10.0),  # C#5
    ]


# ---------------------------------------------------------------------------
# Progressions
# ---------------------------------------------------------------------------

PROG_A = [_chord_i_simple, _chord_vi, _chord_iv, _chord_i_simple]
PROG_B_1 = [_chord_i_open, _chord_iv, _chord_v, _chord_vi]
PROG_B_2 = [_chord_i_open, _chord_vi, _chord_iv, _chord_v]
PROG_C = [_chord_i_rich, _chord_iv_rich, _chord_v_rich, _chord_vi_rich]
PROG_D_1 = [_chord_iv_rich, _chord_v_rich, _chord_i_rich, _chord_vi_rich]
PROG_D_2 = [_chord_i_rich, _chord_iv_rich, _chord_vi_rich, _chord_v_rich]
PROG_D_END = [_chord_iv_rich, _chord_v_rich, _chord_i_rich]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synth_auto(name: str, segments: tuple[AutomationSegment, ...]) -> AutomationSpec:
    return AutomationSpec(
        target=AutomationTarget(kind="synth", name=name),
        segments=segments,
    )


def _control_auto(name: str, segments: tuple[AutomationSegment, ...]) -> AutomationSpec:
    return AutomationSpec(
        target=AutomationTarget(kind="control", name=name),
        segments=segments,
    )


def _lin(t0: float, t1: float, v0: float, v1: float) -> AutomationSegment:
    return AutomationSegment(
        start=t0, end=t1, shape="linear", start_value=v0, end_value=v1
    )


def _hold(t0: float, t1: float, v: float) -> AutomationSegment:
    return AutomationSegment(start=t0, end=t1, shape="hold", value=v)


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

    # --- Send buses ---
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

    # -----------------------------------------------------------------------
    # Voice: PAD
    # -----------------------------------------------------------------------

    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "saw",
            "cutoff_hz": 2800.0,
            "resonance_q": 1.2,
            "filter_topology": "svf",
            "filter_drive": 0.0,
            "filter_morph": 0.0,
            "osc_softness": 0.0,
            "osc_asymmetry": 0.0,
            "osc_shape_drift": 0.0,
            "voice_card_spread": 0.0,
            "voice_card_pitch_spread": 0.3,
            "attack": 0.3,
            "decay": 0.5,
            "sustain_level": 0.7,
            "release": 1.2,
        },
        effects=[
            EffectSpec("chorus", {"rate_hz": 0.4, "depth_ms": 2.0, "mix": 0.2}),
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 120.0,
                            "slope_db_per_oct": 12,
                        },
                    ]
                },
            ),
        ],
        normalize_lufs=-22.0,
        mix_db=-3.0,
        sends=[VoiceSend(target="hall", send_db=-6.0)],
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        automation=[
            _synth_auto(
                "osc_softness",
                (
                    _hold(SEC_A, SEC_B, 0.0),
                    _lin(SEC_B, SEC_C, 0.0, 0.15),
                    _lin(SEC_C, SEC_D, 0.15, 0.2),
                    _hold(SEC_D, SEC_END, 0.2),
                ),
            ),
            _synth_auto(
                "osc_shape_drift",
                (
                    _hold(SEC_A, SEC_B, 0.0),
                    _lin(SEC_B, SEC_D, 0.0, 0.35),
                    _hold(SEC_D, SEC_END, 0.35),
                ),
            ),
            _synth_auto(
                "voice_card_spread",
                (
                    _hold(SEC_A, SEC_B, 0.0),
                    _lin(SEC_B, SEC_D, 0.0, 2.2),
                    _hold(SEC_D, SEC_END, 2.2),
                ),
            ),
            _synth_auto(
                "voice_card_filter_spread",
                (
                    _hold(SEC_A, SEC_B, 0.0),
                    _lin(SEC_B, SEC_C, 0.0, 1.5),
                    _lin(SEC_C, SEC_END, 1.5, 2.8),
                ),
            ),
            _synth_auto(
                "filter_morph",
                (
                    _hold(SEC_A, SEC_B, 0.0),
                    _lin(SEC_B, SEC_D, 0.0, 0.45),
                    _hold(SEC_D, SEC_END, 0.45),
                ),
            ),
            _synth_auto(
                "filter_drive",
                (
                    _hold(SEC_A, SEC_B, 0.0),
                    _lin(SEC_B, SEC_END, 0.0, 0.15),
                ),
            ),
            _synth_auto(
                "cutoff_hz",
                (
                    _hold(SEC_A, SEC_B, 2800.0),
                    _lin(SEC_B, SEC_D, 2800.0, 3500.0),
                    _hold(SEC_D, SEC_END, 3500.0),
                ),
            ),
            _synth_auto(
                "osc_asymmetry",
                (
                    _hold(SEC_A, SEC_B, 0.0),
                    _lin(SEC_B, SEC_C, 0.0, 0.08),
                    _lin(SEC_C, SEC_END, 0.08, 0.14),
                ),
            ),
        ],
    )

    # -----------------------------------------------------------------------
    # Voice: BASS -- ladder filter with feedback
    # -----------------------------------------------------------------------

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
        automation=[
            _synth_auto(
                "feedback_amount",
                (
                    _hold(SEC_A, SEC_B, 0.0),
                    _lin(SEC_B, SEC_C, 0.02, 0.2),
                    _lin(SEC_C, SEC_D, 0.2, 0.35),
                    _hold(SEC_D, SEC_END, 0.35),
                ),
            ),
            _synth_auto(
                "cutoff_hz",
                (
                    _hold(SEC_A, SEC_B, 450.0),
                    _lin(SEC_B, SEC_C, 450.0, 700.0),
                    _lin(SEC_C, SEC_D, 700.0, 900.0),
                    _hold(SEC_D, SEC_END, 900.0),
                ),
            ),
            _synth_auto(
                "filter_drive",
                (
                    _hold(SEC_A, SEC_B, 0.1),
                    _lin(SEC_B, SEC_D, 0.1, 0.3),
                    _hold(SEC_D, SEC_END, 0.3),
                ),
            ),
        ],
    )

    # -----------------------------------------------------------------------
    # Voice: SUB -- sine sub bass for low-end reinforcement
    # -----------------------------------------------------------------------

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
                        {"kind": "lowpass", "cutoff_hz": 250.0, "slope_db_per_oct": 24},
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
                    _hold(SEC_A, SEC_B, 150.0),
                    _lin(SEC_B, SEC_D, 150.0, 220.0),
                    _hold(SEC_D, SEC_END, 220.0),
                ),
            ),
            _control_auto(
                "mix_db",
                (
                    _hold(SEC_A, _pos(B_START_BAR + 4), -20.0),
                    _lin(_pos(B_START_BAR + 4), SEC_C, -20.0, -8.0),
                    _lin(SEC_C, SEC_D, -8.0, -4.0),
                    _hold(SEC_D, SEC_END, -4.0),
                ),
            ),
        ],
    )

    # -----------------------------------------------------------------------
    # Voice: LEAD -- CS80-style brass, enters at section C
    # -----------------------------------------------------------------------

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
                        {"kind": "bell", "freq_hz": 2500.0, "gain_db": 2.0, "q": 1.2},
                    ]
                },
            ),
        ],
        normalize_lufs=-22.0,
        mix_db=-4.0,
        sends=[VoiceSend(target="hall", send_db=-8.0)],
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        automation=[
            _synth_auto(
                "vca_nonlinearity",
                (
                    _lin(SEC_C, SEC_D, 0.2, 0.35),
                    _hold(SEC_D, SEC_END, 0.35),
                ),
            ),
            _synth_auto(
                "cutoff_hz",
                (
                    _hold(SEC_C, SEC_D, 3200.0),
                    _lin(SEC_D, SEC_END, 3200.0, 3800.0),
                ),
            ),
        ],
    )

    # -----------------------------------------------------------------------
    # Voice: TEXTURE -- drone atmosphere
    # -----------------------------------------------------------------------

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
    )

    # -----------------------------------------------------------------------
    # Place notes
    # -----------------------------------------------------------------------

    _place_pad(score)
    _place_bass(score)
    _place_sub(score)
    _place_lead(score)
    _place_texture(score)

    return score


# ---------------------------------------------------------------------------
# Note placement
# ---------------------------------------------------------------------------


def _place_pad(score: Score) -> None:
    chord_dur_a = 2 * BAR

    # Section A (bars 1-8): cold, clean SVF, 2-bar chords
    for i, chord_fn in enumerate(PROG_A):
        t = _pos(A_START_BAR + i * 2)
        for partial, amp_db in chord_fn():
            score.add_note(
                "pad",
                start=t,
                duration=chord_dur_a - 0.1,
                partial=partial,
                amp_db=amp_db,
            )

    # Section B (bars 9-28): switch to ladder, 2.5-bar chords, two cycles
    synth_b = {"filter_topology": "ladder", "bass_compensation": 0.5}
    bar = B_START_BAR
    for prog in [PROG_B_1, PROG_B_2]:
        for chord_fn in prog:
            dur_bars = 2.5
            t = _pos(bar)
            for partial, amp_db in chord_fn():
                score.add_note(
                    "pad",
                    start=t,
                    duration=dur_bars * BAR - 0.1,
                    partial=partial,
                    amp_db=amp_db,
                    synth=synth_b,
                )
            bar += 2  # slight overlap from 2.5-bar sustain over 2-bar grid
        bar += 2  # gap between cycles

    # Section C (bars 29-44): richer voicings, 4-bar chords
    synth_c = {
        "filter_topology": "ladder",
        "bass_compensation": 0.6,
        "osc_dc_offset": 0.15,
    }
    for i, chord_fn in enumerate(PROG_C):
        t = _pos(C_START_BAR + i * 4)
        for partial, amp_db in chord_fn():
            score.add_note(
                "pad",
                start=t,
                duration=4 * BAR - 0.1,
                partial=partial,
                amp_db=amp_db,
                synth=synth_c,
            )

    # Section D (bars 45-72): euphoria, full voicings, 3.5-bar chords
    synth_d = {
        "filter_topology": "ladder",
        "bass_compensation": 0.65,
        "osc_dc_offset": 0.2,
        "osc_shape_drift": 0.4,
    }
    bar = D_START_BAR
    for prog in [PROG_D_1, PROG_D_2]:
        for chord_fn in prog:
            t = _pos(bar)
            for partial, amp_db in chord_fn():
                score.add_note(
                    "pad",
                    start=t,
                    duration=3.5 * BAR - 0.1,
                    partial=partial,
                    amp_db=amp_db + 1.0,
                    synth=synth_d,
                )
            bar += 3
    # Final sustained chord
    for chord_fn in PROG_D_END:
        t = _pos(bar)
        dur = 4 if chord_fn != PROG_D_END[-1] else 5
        for partial, amp_db in chord_fn():
            score.add_note(
                "pad",
                start=t,
                duration=dur * BAR - 0.1,
                partial=partial,
                amp_db=amp_db + 1.0,
                synth=synth_d,
            )
        bar += dur


def _place_bass(score: Score) -> None:
    # Section A: simple whole-note roots
    roots_a = [P1, P_5_3 / 2, P_4_3 / 2, P1]
    for i, root in enumerate(roots_a):
        score.add_note(
            "bass",
            start=_pos(A_START_BAR + i * 2),
            duration=2 * BAR - 0.2,
            partial=root * 2,
            amp_db=-8.0,
        )

    # Section B: two hits per chord, more rhythmic
    roots_b = [P1, P_4_3 / 2, P_3_2 / 2, P_5_3 / 2, P1, P_5_3 / 2, P_4_3 / 2, P_3_2 / 2]
    bar = B_START_BAR
    for root in roots_b:
        t1 = _pos(bar)
        t2 = _pos(bar, 3.0)
        score.add_note(
            "bass", start=t1, duration=2 * BEAT - 0.05, partial=root * 2, amp_db=-6.0
        )
        score.add_note(
            "bass", start=t2, duration=BEAT * 4 - 0.05, partial=root * 2, amp_db=-8.0
        )
        bar += 2
        if bar >= B_END_BAR:
            break

    # Section C: more active, octave jumps
    roots_c = [P1, P_4_3 / 2, P_3_2 / 2, P_5_3 / 2]
    bar = C_START_BAR
    for root in roots_c:
        for beat_in_group in range(8):
            if beat_in_group % 3 == 0:
                t = _pos(bar, 1.0 + beat_in_group)
                octave = 2 if beat_in_group < 4 else 4
                score.add_note(
                    "bass",
                    start=t,
                    duration=BEAT * 2.5,
                    partial=root * octave,
                    amp_db=-5.5,
                )
        bar += 4

    # Section D: most active, syncopated with octave play
    roots_d = [P_4_3 / 2, P_3_2 / 2, P1, P_5_3 / 2, P1, P_4_3 / 2, P_5_3 / 2, P_3_2 / 2]
    bar = D_START_BAR
    for root in roots_d:
        for beat_in_group in range(12):
            if beat_in_group in (0, 3, 5, 8, 10):
                t = _pos(bar, 1.0 + beat_in_group)
                if t >= SEC_END:
                    break
                octave = 2 if beat_in_group in (0, 8) else 4
                vel = 0.9 if beat_in_group == 0 else 0.7
                score.add_note(
                    "bass",
                    start=t,
                    duration=BEAT * 2.0,
                    partial=root * octave,
                    amp_db=-4.5 if octave == 2 else -6.5,
                    velocity=vel,
                )
        bar += 3
        if bar >= D_END_BAR - 3:
            break
    # Final long bass note
    score.add_note(
        "bass", start=_pos(D_END_BAR - 5), duration=5 * BAR, partial=P1 * 2, amp_db=-5.0
    )


def _place_sub(score: Score) -> None:
    """Sine sub enters mid-B, blooms in D."""
    # Follows bass roots one octave below
    roots_b = [P1, P_4_3 / 2, P_3_2 / 2, P_5_3 / 2]
    bar = B_START_BAR + 4  # enters 4 bars into warming
    for root in roots_b:
        if bar >= B_END_BAR:
            break
        score.add_note(
            "sub", start=_pos(bar), duration=4 * BAR - 0.1, partial=root, amp_db=-8.0
        )
        bar += 4

    roots_c = [P1, P_4_3 / 2, P_3_2 / 2, P_5_3 / 2]
    bar = C_START_BAR
    for root in roots_c:
        score.add_note(
            "sub", start=_pos(bar), duration=4 * BAR - 0.1, partial=root, amp_db=-6.0
        )
        bar += 4

    roots_d = [
        P_4_3 / 2,
        P_3_2 / 2,
        P1,
        P_5_3 / 2,
        P1,
        P_4_3 / 2,
        P_5_3 / 2,
        P_3_2 / 2,
        P_4_3 / 2,
        P_3_2 / 2,
    ]
    bar = D_START_BAR
    for root in roots_d:
        if bar >= D_END_BAR - 3:
            break
        score.add_note(
            "sub", start=_pos(bar), duration=3 * BAR - 0.1, partial=root, amp_db=-5.0
        )
        bar += 3
    # Final sustained sub
    score.add_note(
        "sub",
        start=_pos(D_END_BAR - 5),
        duration=5 * BAR + 2.0,
        partial=P1,
        amp_db=-5.0,
    )


def _place_lead(score: Score) -> None:
    """CS80 brass lead: enters at C, develops through D."""

    # -- Section C: initial melody, moderate pace --
    melody_c: list[tuple[int, float, float, float, float, float]] = [
        # (bar_offset, beat, partial, dur_beats, amp_db, velocity)
        (0, 1.0, P_7_4 * 8, 3.0, -6.0, 0.85),
        (0, 4.5, P_5_3 * 8, 2.0, -8.0, 0.72),
        (1, 3.0, P_3_2 * 8, 2.5, -7.0, 0.8),
        (2, 1.5, P_5_4 * 8, 1.5, -8.5, 0.68),
        (2, 3.5, P_4_3 * 8, 3.0, -6.5, 0.82),
        (3, 3.0, P_5_4 * 8, 2.5, -7.5, 0.75),
        # Ascending phrase
        (5, 1.0, P_5_4 * 8, 2.0, -7.0, 0.78),
        (5, 3.0, P_3_2 * 8, 2.0, -6.5, 0.85),
        (6, 1.0, P_5_3 * 8, 3.0, -6.0, 0.88),
        (7, 1.0, P_7_4 * 8, 4.0, -5.5, 0.92),  # sustained septimal 7th
        # Resolving
        (8, 2.0, P_3_2 * 8, 2.0, -7.0, 0.75),
        (8, 4.5, P_5_4 * 8, 3.5, -6.5, 0.8),
        (10, 1.0, P1 * 8, 5.0, -6.0, 0.85),  # root, long sustain
    ]

    for bar_off, beat, partial, dur_beats, amp_db, vel in melody_c:
        t = _pos(C_START_BAR + bar_off, beat)
        if t < SEC_D:
            score.add_note(
                "lead",
                start=t,
                duration=dur_beats * BEAT,
                partial=partial,
                amp_db=amp_db,
                velocity=vel,
            )

    # -- Section D: developed melody, faster, wider intervals, more dynamic --
    melody_d: list[tuple[int, float, float, float, float, float]] = [
        # Phrase 1: quick ascending run
        (0, 1.0, P_5_4 * 8, 1.5, -7.0, 0.78),
        (0, 2.5, P_4_3 * 8, 1.0, -8.0, 0.7),
        (0, 3.5, P_3_2 * 8, 1.5, -6.5, 0.85),
        (1, 1.0, P_5_3 * 8, 2.0, -6.0, 0.88),
        (1, 3.5, P_7_4 * 8, 3.0, -5.0, 0.95),  # septimal peak
        # Phrase 2: descending, ornamented
        (3, 1.0, P_7_4 * 8, 1.0, -6.0, 0.85),
        (3, 2.0, P_5_3 * 8, 0.8, -7.5, 0.72),
        (3, 3.0, P_3_2 * 8, 1.5, -6.5, 0.82),
        (3, 4.5, P_5_4 * 8, 1.0, -8.0, 0.68),
        (4, 1.5, P_4_3 * 8, 2.5, -6.0, 0.88),
        (4, 4.0, P_5_4 * 8, 2.0, -7.0, 0.78),
        # Phrase 3: call and response — high then low
        (6, 1.0, P2 * 8, 2.0, -5.5, 0.92),  # high octave
        (6, 3.0, P_3_2 * 4, 2.0, -6.5, 0.8),  # drop an octave
        (7, 1.0, P_7_4 * 8, 2.5, -5.0, 0.95),  # back up to septimal 7th
        (7, 3.5, P_5_3 * 8, 1.5, -6.0, 0.85),
        (8, 1.0, P2 * 8, 3.0, -5.5, 0.9),
        # Phrase 4: rhythmic, punchy
        (10, 1.0, P_5_4 * 8, 1.0, -6.0, 0.88),
        (10, 2.0, P_5_4 * 8, 0.5, -8.0, 0.65),  # ghost note
        (10, 2.5, P_3_2 * 8, 1.5, -5.5, 0.92),
        (10, 4.0, P_5_3 * 8, 1.0, -6.5, 0.8),
        (11, 1.0, P_7_4 * 8, 1.5, -5.0, 0.95),
        (11, 2.5, P_7_4 * 8, 0.5, -7.0, 0.7),  # echo
        (11, 3.5, P2 * 8, 2.5, -5.0, 0.92),
        # Phrase 5: final arc — slow descent to root
        (14, 1.0, P_7_4 * 8, 3.0, -5.0, 0.92),
        (15, 1.0, P_5_3 * 8, 2.5, -5.5, 0.88),
        (16, 1.0, P_3_2 * 8, 3.0, -5.5, 0.85),
        (17, 1.0, P_5_4 * 8, 3.0, -6.0, 0.82),
        (18, 1.0, P1 * 8, 4.5, -5.0, 0.9),
        # Very last note — high root, fading into reverb
        (20, 1.0, P2 * 8, 6.0, -6.0, 0.85),
    ]

    for bar_off, beat, partial, dur_beats, amp_db, vel in melody_d:
        t = _pos(D_START_BAR + bar_off, beat)
        if t < SEC_END + 2.0:
            score.add_note(
                "lead",
                start=t,
                duration=dur_beats * BEAT,
                partial=partial,
                amp_db=amp_db,
                velocity=vel,
            )


def _place_texture(score: Score) -> None:
    """Drone atmosphere throughout, thickening with each section."""
    layers: list[tuple[float, float, float, float]] = [
        # Section A-B: root + fifth, quiet
        (SEC_A, SEC_C, P1 * 2, -13.0),
        (SEC_A, SEC_C, P_3_2 * 2, -15.0),
        # Section B-C: add major third
        (SEC_B, SEC_D, P_5_4 * 2, -15.0),
        # Section C-D: septimal colour enters
        (SEC_C, SEC_END, P_7_4 * 2, -14.0),
        # Section D: full drone — root, third, fifth, seventh
        (SEC_D, SEC_END + 3.0, P1 * 2, -10.0),
        (SEC_D, SEC_END + 3.0, P_5_4 * 2, -12.0),
        (SEC_D, SEC_END + 3.0, P_3_2 * 2, -12.0),
        (SEC_D, SEC_END + 3.0, P_7_4 * 2, -11.0),
        # High shimmer in euphoria
        (SEC_D + 8 * BAR, SEC_END + 3.0, P1 * 8, -16.0),
        (SEC_D + 8 * BAR, SEC_END + 3.0, P_5_4 * 8, -17.0),
    ]
    for start, end, partial, amp_db in layers:
        score.add_note(
            "texture", start=start, duration=end - start, partial=partial, amp_db=amp_db
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

PIECES: dict[str, PieceDefinition] = {
    "warming_up": PieceDefinition(
        name="warming_up",
        output_name="warming_up",
        build_score=build_score,
        sections=(
            PieceSection("cold", SEC_A, SEC_B),
            PieceSection("warming", SEC_B, SEC_C),
            PieceSection("arriving", SEC_C, SEC_D),
            PieceSection("euphoria", SEC_D, SEC_END),
        ),
    ),
}
