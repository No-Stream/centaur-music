"""Colundi Sequence — melodic techno in Colundi tuning.

Tuning: Colundi JI scale as partials of f0 = 55 Hz (A1).
The Colundi scale (1/1, 11/10, 19/16, 4/3, 3/2, 49/30, 7/4, 2/1) omits the
conventional third, giving harmony a floating, non-resolving quality. The
4:6:7 otonal chord is the natural resting place — warm but ambiguous.

BPM = 122.  1 bar ~= 1.967 s.  1 beat ~= 0.492 s.  1 sixteenth ~= 0.123 s.

Piece: colundi_sequence
Structure (~52 bars / ~102 s):
  bars  1– 4   kick alone
  bars  5– 8   + bass (R-P4-P5 motion)
  bars  9–16   + hats + perc + clap (full rhythm section)
  bars 17–20   + pad (4:6:7 home chord, fading in)
  bars 21–24   pad moves to IV chord
  bars 25–32   + melody (primary motif: P5-h7-R-h7 arch)
  bars 33–40   melody develops (secondary motif, alien colour tones)
  bars 41–44   peel back: hats thin, perc drops
  bars 45–48   bass drops, melody fragments
  bars 49–52   pad + delay tails, fade
"""

from __future__ import annotations

from code_musics.automation import AutomationSegment, AutomationSpec, AutomationTarget
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score, VoiceSend
from code_musics.spectra import ratio_spectrum
from code_musics.synth import BRICASTI_IR_DIR

BPM: float = 122.0
BEAT: float = 60.0 / BPM
BAR: float = 4.0 * BEAT
S16: float = BEAT / 4.0

F0: float = 55.0

# ---------------------------------------------------------------------------
# Colundi scale as partials of f0 = 55 Hz
# Degree:  R     N2      m3      P4    P5    s6      h7    8ve
# Ratio:   1/1   11/10   19/16   4/3   3/2   49/30   7/4   2/1
# ---------------------------------------------------------------------------

# Octave 1: sub/bass (55-110 Hz)
R1: float = 1.0
N2_1: float = 11 / 10
m3_1: float = 19 / 16
P4_1: float = 4 / 3
P5_1: float = 3 / 2
s6_1: float = 49 / 30
h7_1: float = 7 / 4

# Octave 2: low-mid (110-220 Hz)
R2: float = 2.0
N2_2: float = 11 / 5
m3_2: float = 19 / 8
P4_2: float = 8 / 3
P5_2: float = 3.0
s6_2: float = 49 / 15
h7_2: float = 7 / 2

# Octave 3: melody (220-440 Hz)
R3: float = 4.0
N2_3: float = 22 / 5
m3_3: float = 19 / 4
P4_3: float = 16 / 3
P5_3: float = 6.0
s6_3: float = 98 / 15
h7_3: float = 7.0

# Octave 4: high sparkle (440-880 Hz)
R4: float = 8.0
N2_4: float = 44 / 5
P4_4: float = 32 / 3
P5_4: float = 12.0
h7_4: float = 14.0


def _pos(bar: int, beat: int = 1, n16: int = 0) -> float:
    """Absolute time at bar:beat:sixteenth (1-indexed)."""
    return (bar - 1) * BAR + (beat - 1) * BEAT + n16 * S16


def _make_reverb_bus() -> list[EffectSpec]:
    """Dark, spacious reverb for the send bus."""
    if BRICASTI_IR_DIR.exists():
        return [
            EffectSpec(
                "bricasti",
                {
                    "ir_name": "1 Halls 07 Large & Dark",
                    "wet": 1.0,
                    "lowpass_hz": 5500.0,
                    "highpass_hz": 180.0,
                },
            )
        ]
    return [
        EffectSpec("reverb", {"room_size": 0.82, "damping": 0.55, "wet_level": 1.0}),
        EffectSpec(
            "eq",
            {
                "bands": [
                    {"kind": "highpass", "cutoff_hz": 180.0, "slope_db_per_oct": 12},
                    {"kind": "lowpass", "cutoff_hz": 5500.0, "slope_db_per_oct": 12},
                ]
            },
        ),
    ]


# Colundi-aligned additive spectrum: overtone content mirrors the scale itself
# so the timbre resonates sympathetically with the harmonic language.
_COLUNDI_PARTIALS = ratio_spectrum(
    ratios=[1.0, 11 / 10, 19 / 16, 4 / 3, 3 / 2, 49 / 30, 7 / 4, 2.0, 3.0, 7 / 2],
    amps=[1.0, 0.25, 0.18, 0.45, 0.55, 0.12, 0.35, 0.30, 0.15, 0.10],
)


TOTAL_BARS: int = 52


def build_score() -> Score:
    """Build the Colundi Sequence score."""
    score = Score(
        f0=F0,
        master_effects=[
            EffectSpec(
                "compressor",
                {
                    "threshold_db": -18.0,
                    "ratio": 2.0,
                    "attack_ms": 30.0,
                    "release_ms": 300.0,
                    "knee_db": 8.0,
                    "makeup_gain_db": 0.5,
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
            EffectSpec(
                "saturation", {"preset": "tube_warm", "mix": 0.12, "drive": 0.7}
            ),
        ],
    )

    # ------------------------------------------------------------------
    # Send bus: dark, spacious reverb
    # ------------------------------------------------------------------
    score.add_send_bus("space", effects=_make_reverb_bus())

    # ------------------------------------------------------------------
    # Kick: 808_house, four-on-the-floor
    # ------------------------------------------------------------------
    score.add_voice(
        "kick",
        synth_defaults={
            "engine": "kick_tom",
            "preset": "808_house",
            "params": {
                "body_punch_ratio": 0.32,
                "overtone_amount": 0.09,
                "click_amount": 0.10,
            },
        },
        effects=[
            EffectSpec("compressor", {"preset": "kick_punch"}),
            EffectSpec("saturation", {"preset": "kick_weight"}),
        ],
        normalize_peak_db=-6.0,
        mix_db=-4.0,
        velocity_humanize=None,
    )

    for bar in range(1, TOTAL_BARS + 1):
        for beat in range(1, 5):
            score.add_note(
                "kick",
                start=_pos(bar, beat),
                duration=0.8,
                freq=F0,
                amp_db=-6.0,
            )

    # ------------------------------------------------------------------
    # Bass: deep sine-ish sub with drive, root-anchored, rhythmic groove
    # ------------------------------------------------------------------
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "polyblep",
            "preset": "sub_bass",
            "params": {
                "waveform": "sine",
                "cutoff_hz": 95.0,
                "filter_env_amount": 0.3,
                "filter_env_decay": 0.08,
                "resonance_q": 0.5,
                "filter_drive": 0.3,
            },
        },
        mix_db=-6.5,
        max_polyphony=1,
        velocity_humanize=None,
        effects=[
            EffectSpec(
                "saturation", {"preset": "kick_weight", "mix": 0.18, "drive": 0.9}
            ),
            EffectSpec(
                "compressor", {"preset": "kick_duck", "sidechain_source": "kick"}
            ),
        ],
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="cutoff_hz"),
                segments=(
                    AutomationSegment(
                        start=_pos(5),
                        end=_pos(33),
                        shape="linear",
                        start_value=110.0,
                        end_value=150.0,
                    ),
                    AutomationSegment(
                        start=_pos(33),
                        end=_pos(45),
                        shape="linear",
                        start_value=150.0,
                        end_value=110.0,
                    ),
                ),
            ),
        ],
    )

    _build_bass_notes(score)

    # ------------------------------------------------------------------
    # Hats: 16th notes with velocity shaping
    # ------------------------------------------------------------------
    score.add_voice(
        "hat",
        synth_defaults={"engine": "noise_perc", "preset": "chh"},
        mix_db=-11.0,
        velocity_humanize=None,
        effects=[
            EffectSpec(
                "eq",
                {"bands": [{"kind": "high_shelf", "freq_hz": 8000.0, "gain_db": 3.0}]},
            ),
            EffectSpec("saturation", {"drive": 0.35}),
            EffectSpec("delay", {"delay_seconds": S16, "feedback": 0.28, "mix": 0.28}),
            EffectSpec(
                "compressor", {"preset": "kick_duck", "sidechain_source": "kick"}
            ),
        ],
        sends=[VoiceSend(target="space", send_db=-18.0)],
    )

    _build_hat_notes(score)

    # ------------------------------------------------------------------
    # Perc: syncopated tick/rim
    # ------------------------------------------------------------------
    score.add_voice(
        "perc",
        synth_defaults={"engine": "noise_perc", "preset": "tick"},
        mix_db=-10.0,
        normalize_peak_db=-6.0,
        velocity_humanize=None,
        effects=[
            EffectSpec(
                "compressor", {"preset": "kick_duck", "sidechain_source": "kick"}
            ),
        ],
        sends=[VoiceSend(target="space", send_db=-16.0)],
    )

    _build_perc_notes(score)

    # ------------------------------------------------------------------
    # Clap: beats 2 and 4
    # ------------------------------------------------------------------
    score.add_voice(
        "clap",
        synth_defaults={"engine": "noise_perc", "preset": "clap"},
        mix_db=-14.5,
        normalize_peak_db=-6.0,
        velocity_humanize=None,
        effects=[
            EffectSpec("chorus", {"preset": "juno_subtle", "mix": 0.06}),
        ],
        sends=[VoiceSend(target="space", send_db=-16.0)],
    )

    _build_clap_notes(score)

    # ------------------------------------------------------------------
    # Pad: additive with Colundi-aligned spectrum, 4:6:7 chord
    # ------------------------------------------------------------------
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "additive",
            "params": {
                "partials": _COLUNDI_PARTIALS,
                "unison_voices": 3,
                "unison_detune_cents": 6.0,
            },
            "env": {
                "attack_ms": 800.0,
                "decay_ms": 600.0,
                "sustain_ratio": 0.7,
                "release_ms": 1200.0,
            },
        },
        mix_db=-7.5,
        effects=[
            EffectSpec("chorus", {"preset": "juno_subtle", "mix": 0.25}),
            EffectSpec(
                "compressor", {"preset": "kick_duck", "sidechain_source": "kick"}
            ),
        ],
        sends=[VoiceSend(target="space", send_db=-8.0)],
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="brightness_tilt"),
                segments=(
                    AutomationSegment(
                        start=_pos(13),
                        end=_pos(29),
                        shape="linear",
                        start_value=-0.3,
                        end_value=0.15,
                    ),
                    AutomationSegment(
                        start=_pos(29),
                        end=_pos(49),
                        shape="linear",
                        start_value=0.15,
                        end_value=-0.2,
                    ),
                ),
            ),
        ],
    )

    _build_pad_notes(score)

    # ------------------------------------------------------------------
    # Melody: FM bell, slow deliberate phrases
    # ------------------------------------------------------------------
    score.add_voice(
        "melody",
        synth_defaults={
            "engine": "fm",
            "preset": "bell",
            "params": {
                "mod_ratio": 1.0,
                "mod_index": 1.2,
            },
            "env": {
                "attack_ms": 20.0,
                "decay_ms": 500.0,
                "sustain_ratio": 0.25,
                "release_ms": 600.0,
            },
        },
        mix_db=-9.5,
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 350.0, "slope_db_per_oct": 24}
                    ]
                },
            ),
            EffectSpec(
                "saturation", {"preset": "tube_warm", "mix": 0.18, "drive": 0.9}
            ),
            EffectSpec(
                "delay",
                {"delay_seconds": 3.0 * S16, "feedback": 0.45, "mix": 0.35},
            ),
            EffectSpec(
                "compressor", {"preset": "kick_duck", "sidechain_source": "kick"}
            ),
        ],
        sends=[
            VoiceSend(
                target="space",
                send_db=-10.0,
                automation=[
                    AutomationSpec(
                        target=AutomationTarget(kind="control", name="send_db"),
                        segments=(
                            AutomationSegment(
                                start=_pos(19),
                                end=_pos(40),
                                shape="linear",
                                start_value=-10.0,
                                end_value=-5.0,
                            ),
                        ),
                    ),
                ],
            )
        ],
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="mod_index"),
                segments=(
                    AutomationSegment(
                        start=_pos(19),
                        end=_pos(38),
                        shape="linear",
                        start_value=1.2,
                        end_value=2.0,
                    ),
                    AutomationSegment(
                        start=_pos(38),
                        end=_pos(49),
                        shape="linear",
                        start_value=2.0,
                        end_value=1.0,
                    ),
                ),
            ),
            AutomationSpec(
                target=AutomationTarget(kind="control", name="pan"),
                segments=(
                    AutomationSegment(
                        start=_pos(19),
                        end=_pos(35),
                        shape="linear",
                        start_value=-0.06,
                        end_value=0.06,
                    ),
                    AutomationSegment(
                        start=_pos(35),
                        end=_pos(49),
                        shape="linear",
                        start_value=0.06,
                        end_value=-0.04,
                    ),
                ),
            ),
        ],
    )

    _build_melody_notes(score)

    return score


# ======================================================================
# Note-writing helpers
# ======================================================================


def _build_bass_notes(score: Score) -> None:
    """Bass: deep root-anchored sub with syncopated 16th groove.

    Stays on the root ~90% of the time. Groove comes from rhythm, not pitch.
    Four 1-bar patterns rotate for variety — all root-anchored but with
    different syncopation, ghost notes, and gate lengths.
    P4 appears only as rare punctuation (once per 8 bars).
    """
    # (beat, n16, partial, gate_16ths, amp_db)
    # gate_16ths controls note length — short for staccato pulse, long for legato sub

    # Pattern A: classic house — beat 1 anchor, off-beat ghost, &-of-3 syncopation
    _pat_a = [
        (1, 0, R1, 3, -5.0),  # beat 1: root, held
        (2, 2, R1, 1, -11.0),  # &-of-2: ghost
        (3, 2, R1, 2, -6.5),  # &-of-3: syncopated accent
        (4, 2, R1, 1, -12.0),  # &-of-4: ghost
    ]

    # Pattern B: pushed — anticipation before beat 1, off-beat pulse
    _pat_b = [
        (1, 0, R1, 2, -5.5),  # beat 1: root
        (1, 3, R1, 1, -11.5),  # a-of-1: ghost push
        (3, 0, R1, 3, -5.0),  # beat 3: root, held
        (4, 1, R1, 1, -10.5),  # e-of-4: syncopated ghost
    ]

    # Pattern C: sparse and deep — fewer notes, longer gates
    _pat_c = [
        (1, 0, R1, 4, -4.5),  # beat 1: root, long hold
        (3, 2, R1, 3, -6.0),  # &-of-3: syncopated, held
    ]

    # Pattern D: busier — 16th-note energy, creates forward motion
    _pat_d = [
        (1, 0, R1, 2, -5.0),  # beat 1
        (1, 2, R1, 1, -10.0),  # &-of-1: ghost
        (2, 3, R1, 2, -7.0),  # a-of-2: anticipation
        (3, 2, R1, 2, -6.0),  # &-of-3: accent
        (4, 1, R1, 1, -11.0),  # e-of-4: ghost
        (4, 3, R1, 1, -9.0),  # a-of-4: push into next bar
    ]

    _patterns = [_pat_a, _pat_b, _pat_c, _pat_d]

    for bar in range(5, 45):
        pat_idx = (bar - 5) % 4
        pattern = _patterns[pat_idx]
        for beat, n16, partial, gate_16ths, amp_db in pattern:
            score.add_note(
                "bass",
                start=_pos(bar, beat, n16),
                duration=gate_16ths * S16 * 0.85,
                partial=partial,
                amp_db=amp_db,
            )

    # P4 punctuation — rare, once per ~8 bars, always on an off-beat
    _p4_hits = [
        (12, 3, 2, 2, -7.0),  # &-of-3
        (20, 4, 2, 2, -7.5),  # &-of-4
        (28, 3, 2, 2, -6.5),  # &-of-3
        (36, 4, 2, 2, -7.0),  # &-of-4
    ]
    for bar, beat, n16, gate_16ths, amp_db in _p4_hits:
        score.add_note(
            "bass",
            start=_pos(bar, beat, n16),
            duration=gate_16ths * S16 * 0.85,
            partial=P4_1,
            amp_db=amp_db,
        )


def _build_hat_notes(score: Score) -> None:
    """Hats: 16th-note pattern with per-subdivision velocity."""
    _hat_amps = {0: -11.0, 1: -16.0, 2: -13.5, 3: -16.5}

    _hat_section_offset: dict[int, float] = {
        **{b: -3.0 for b in range(9, 13)},
        **{b: 0.0 for b in range(13, 33)},
        **{b: -1.5 for b in range(33, 41)},
        **{b: -4.0 for b in range(41, 45)},
    }

    _hat_section_freq: dict[int, float] = {
        **{b: 10500.0 for b in range(9, 13)},
        **{b: 13000.0 for b in range(13, 25)},
        **{b: 14000.0 for b in range(25, 41)},
        **{b: 11000.0 for b in range(41, 45)},
    }

    for bar in range(9, 45):
        offset = _hat_section_offset.get(bar, 0.0)
        hat_freq = _hat_section_freq.get(bar, 13000.0)
        for beat in range(1, 5):
            for n16 in range(4):
                score.add_note(
                    "hat",
                    start=_pos(bar, beat, n16),
                    duration=0.04,
                    freq=hat_freq,
                    amp_db=_hat_amps[n16] + offset,
                )


def _build_perc_notes(score: Score) -> None:
    """Perc: syncopated tick pattern interlocking with hats."""
    # Hits on &-of-2, e-of-4, and occasionally &-of-1 for variation
    _perc_pattern_a = [(2, 2, -6.0), (4, 1, -7.5)]
    _perc_pattern_b = [(1, 2, -8.0), (2, 2, -6.5), (4, 1, -7.0)]

    for bar in range(9, 41):
        pattern = _perc_pattern_a if (bar % 4 in (1, 2)) else _perc_pattern_b
        for beat, n16, amp_db in pattern:
            score.add_note(
                "perc",
                start=_pos(bar, beat, n16),
                duration=0.06,
                freq=6000.0,
                amp_db=amp_db,
            )


def _build_clap_notes(score: Score) -> None:
    """Clap: beats 2 and 4, active in full arrangement sections."""
    _clap_offset: dict[int, float] = {
        **{b: -2.0 for b in range(9, 13)},
        **{b: 0.0 for b in range(13, 41)},
        **{b: -2.0 for b in range(41, 45)},
    }

    for bar in range(9, 45):
        offset = _clap_offset.get(bar, 0.0)
        for beat in [2, 4]:
            score.add_note(
                "clap",
                start=_pos(bar, beat),
                duration=0.12,
                freq=1800.0,
                amp_db=-4.0 + offset,
            )


def _build_pad_notes(score: Score) -> None:
    """Pad: 4:6:7 home chord and IV transposition."""
    # Home chord (I): R + P5 + h7 in octave 2 = partials 2.0, 3.0, 3.5
    # IV chord: P4 + R(8ve) + h7 = partials 8/3, 4.0, 7/2
    # (4/3 × {1, 3/2, 7/4} = {4/3, 2, 7/3} but we voice it as {8/3, 4, 7/2}
    #  to keep the chord in a similar register)

    _home_chord = [R2, P5_2, h7_2]  # [2.0, 3.0, 3.5]
    _iv_chord = [P4_2, R3, h7_2]  # [8/3, 4.0, 3.5]
    _color_chord = [m3_2, P5_2, s6_2]  # [19/8, 3.0, 49/15] — tense

    _pad_sections: list[tuple[int, int, list[float]]] = [
        (13, 16, _home_chord),
        (17, 20, _iv_chord),
        (21, 24, _home_chord),
        (25, 28, _iv_chord),
        (29, 32, _color_chord),
        (33, 36, _home_chord),
        (37, 40, _iv_chord),
        (41, 48, _home_chord),
    ]

    for start_bar, end_bar, chord in _pad_sections:
        # Hold each chord for its full section duration
        chord_start = _pos(start_bar)
        chord_dur = _pos(end_bar + 1) - chord_start - 0.1
        amp = -8.0 if start_bar < 41 else -12.0
        for partial in chord:
            score.add_note(
                "pad",
                start=chord_start,
                duration=chord_dur,
                partial=partial,
                amp_db=amp,
            )

    # Final root drone for the tail (bars 49-52)
    score.add_note(
        "pad",
        start=_pos(49),
        duration=_pos(TOTAL_BARS + 1) - _pos(49) - 0.1,
        partial=R2,
        amp_db=-14.0,
    )


def _build_melody_notes(score: Score) -> None:
    """Melody: one simple motif, repeated. Octave 4 (440-880 Hz).

    Core motif (2 bars): P5 -> h7 -> R(high) -> h7
    Four notes in an arch. The P5-h7 step (septimal minor third) is bittersweet.
    The reach to R is hope. The fall to h7 is not-quite-resolving.

    The motif repeats ~11 times from bar 19 to bar 40. On a couple of
    repetitions one note changes — those small departures are the only
    development. The dotted-eighth delay fills in the rest.
    """
    R5 = 16.0  # 880 Hz

    # Core motif: 4 notes across 2 bars, addressed as 16th offsets from bar start.
    # (offset_in_16ths, partial, dur_beats, amp_db)
    _CORE = [
        (0, P5_4, 2.5, -6.0),  # P5 — the opening
        (10, h7_4, 1.5, -6.5),  # h7 — bittersweet step up
        (16, R5, 2.5, -5.5),  # R(high) — the peak
        (26, h7_4, 2.0, -7.0),  # h7 — fall back, unresolved
    ]

    # Variation: last note resolves down to P5 — an exhale
    _RESOLVE = [
        (0, P5_4, 2.5, -6.0),
        (10, h7_4, 1.5, -6.5),
        (16, R5, 2.5, -5.5),
        (26, P5_4, 2.0, -7.5),
    ]

    # Variation: peak on R4 (lower octave) — more intimate
    _LOW = [
        (0, P5_4, 2.5, -6.0),
        (10, h7_4, 1.5, -6.5),
        (16, R4, 2.5, -6.0),
        (26, h7_4, 2.0, -7.0),
    ]

    def _place_motif(
        bar: int,
        motif: list[tuple[int, float, float, float]],
        amp_offset: float = 0.0,
    ) -> None:
        motif_start = _pos(bar)
        for sixteenths_offset, partial, dur_beats, amp_db in motif:
            score.add_note(
                "melody",
                start=motif_start + sixteenths_offset * S16,
                duration=dur_beats * BEAT * 0.9,
                partial=partial,
                amp_db=amp_db + amp_offset,
            )

    # Bars 19-40: motif every 2 bars. Mostly _CORE with rare departures.
    _sequence: list[tuple[int, list[tuple[int, float, float, float]], float]] = [
        (19, _CORE, 0.0),
        (21, _CORE, 0.0),
        (23, _CORE, 0.0),
        (25, _RESOLVE, 0.0),
        (27, _CORE, 0.0),
        (29, _CORE, 0.0),
        (31, _LOW, 0.0),
        (33, _CORE, 0.0),
        (35, _RESOLVE, 0.0),
        (37, _CORE, -1.0),
        (39, _CORE, -2.5),
    ]

    for bar, motif, amp_offset in _sequence:
        _place_motif(bar, motif, amp_offset)

    # Tail: single long notes, fragments of the motif
    score.add_note(
        "melody",
        start=_pos(45),
        duration=4.0 * BEAT * 0.9,
        partial=P5_4,
        amp_db=-10.0,
    )
    score.add_note(
        "melody",
        start=_pos(47),
        duration=4.0 * BEAT * 0.9,
        partial=h7_4,
        amp_db=-11.0,
    )


PIECES: dict[str, PieceDefinition] = {
    "colundi_sequence": PieceDefinition(
        name="colundi_sequence",
        output_name="colundi_sequence",
        build_score=build_score,
        sections=(
            PieceSection("kick_intro", _pos(1), _pos(5)),
            PieceSection("bass_enters", _pos(5), _pos(9)),
            PieceSection("full_rhythm", _pos(9), _pos(13)),
            PieceSection("pad_enters", _pos(13), _pos(19)),
            PieceSection("melody_hook", _pos(19), _pos(33)),
            PieceSection("melody_fades", _pos(33), _pos(41)),
            PieceSection("peel_back", _pos(41), _pos(49)),
            PieceSection("tail", _pos(49), _pos(TOTAL_BARS + 1)),
        ),
    ),
}
