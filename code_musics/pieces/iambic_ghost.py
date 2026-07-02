"""iambic_ghost — weird IDM-y breakbeat over lush septimal JI harmony.

A fast drum-and-bass-adjacent piece at 172 BPM, 7-limit JI, ~2:47. Built on
Squarepusher/Venetian Snares/Drukqs energy: the drums are brutal and
constantly-mutating (Amen base phrase per-section mutated via
``mutate_rhythm``, non-four-on-floor), the glitch layer is dense and alive
(bit-crushed modal ghost perc, chaotic-attractor lead with audio-rate FM
on osc detune), but the underlying harmony is *euphonic* — otonal septimal
chord stacks (4:5:6:7, 6:7:9:11) with deliberate voice-leading. The pad is
recessed in level but harmonically prominent. The goal is "sublime chaos":
fast and aggressive but unmistakably musical.

Section map (120 bars at 172 BPM, ~2:47):
    1-  8   Intro. Pad + filtered ghost perc. Drum bus K35 @ ~600 Hz.
    9- 16   Break emerges. Kick + hats, bus K35 opens to ~4 kHz.
   17- 36   Groove A. Full break + bass + sub. Mutation seed A.
   37- 48   Glitch breakdown. Kick drops, glitch_lead + ghost ride.
   49- 72   Groove B. Second drop. Per-bar kick timbre morphing.
   73-104   Final push. Peak density. Retrigger fills every 8 bars.
  105-120   Outro. Elements peel away; reverb tail.
"""

from __future__ import annotations

from code_musics.automation import (
    AutomationSegment,
    AutomationSpec,
    AutomationTarget,
)
from code_musics.composition import diminish
from code_musics.drum_helpers import add_drum_voice, setup_drum_bus
from code_musics.generative import TuringMachine, euclidean_pattern, mutate_rhythm
from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    VelocityHumanizeSpec,
)
from code_musics.modulation import (
    ChaoticSource,
    MacroSource,
    ModConnection,
    OscillatorSource,
)
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, NoteEvent, Phrase, Score, VoiceSend
from code_musics.synth import db_to_amp

# ---------------------------------------------------------------------------
# Tempo, grid, tuning constants
# ---------------------------------------------------------------------------

BPM = 172.0
BEAT = 60.0 / BPM
BAR = 4.0 * BEAT
S8 = BEAT / 2.0
S16 = BEAT / 4.0
S32 = BEAT / 8.0

F0_HZ = 98.0  # G2; sub at partial=0.5 => 49 Hz

# 7-limit JI ratios used by the bass line (chord tables use numeric literals)
_R7_6 = 7.0 / 6.0
_R7_4 = 7.0 / 4.0

TOTAL_BARS = 120


def _pos(bar: int, beat: int = 1, n16: int = 0, n32: int = 0) -> float:
    """Return absolute seconds for a 1-indexed bar/beat/n16/n32 grid."""
    return (bar - 1) * BAR + (beat - 1) * BEAT + n16 * S16 + n32 * S32


TOTAL_DUR = _pos(TOTAL_BARS + 1)

# Section boundaries (bar indices; end-exclusive)
S_INTRO = (1, 9)
S_BREAK_IN = (9, 17)
S_GROOVE_A = (17, 37)
S_BREAKDOWN = (37, 49)
S_GROOVE_B = (49, 73)
S_FINAL_PUSH = (73, 105)
S_OUTRO = (105, 121)

# Section boundaries in seconds (for automation)
T_INTRO = _pos(S_INTRO[0])
T_BREAK_IN = _pos(S_BREAK_IN[0])
T_GROOVE_A = _pos(S_GROOVE_A[0])
T_BREAKDOWN = _pos(S_BREAKDOWN[0])
T_GROOVE_B = _pos(S_GROOVE_B[0])
T_FINAL_PUSH = _pos(S_FINAL_PUSH[0])
T_OUTRO = _pos(S_OUTRO[0])

# Mutation seeds — pinned per (section, voice) so renders are deterministic
# and the seed becomes part of the composition, not a tuning knob.
_SEEDS: dict[str, dict[str, int]] = {
    "break_in": {"kick": 11, "snare": 13},
    "groove_a": {"kick": 17, "snare": 23},
    "breakdown": {"kick": 29, "snare": 31},
    "groove_b": {"kick": 37, "snare": 41},
    "final_push": {"kick": 53, "snare": 59},
    "outro": {"kick": 71, "snare": 73},
}

# ---------------------------------------------------------------------------
# Pad chord progression — 2 septimal otonal stacks cycling every 8 bars.
# Kept simple after initial 4-chord version didn't mesh (septimal bass 7/8
# clashed with the sub, and chord 4 minor-tint muddied the whole texture).
# Both chords are pure 4:5:6:7 otonal stacks for maximum JI sheen and
# minimum voice-leading tension — the piece's harmonic weight is steady
# while the drums + lead carry all the motion.
# ---------------------------------------------------------------------------

# (bass_root_partial, pad_partials) — pad partials sound above the bass root.
#   Chord I  — root 1.0 (F0), otonal 4:5:6:7 stack
#   Chord IV — root 4/3, otonal 4:5:6:7 stack transposed
_PAD_PROGRESSION: list[tuple[float, tuple[float, ...]]] = [
    # Chord I — 4:5:6:7 above the root, plus +9/4 for shimmer
    (1.0, (2.0, 2.5, 3.0, 3.5, 4.5)),
    # Chord IV — same 4:5:6:7 shape shifted up a perfect fourth (4/3)
    (4.0 / 3.0, (2.0 * 4.0 / 3.0, 2.5 * 4.0 / 3.0, 3.0 * 4.0 / 3.0, 3.5 * 4.0 / 3.0)),
]

# ---------------------------------------------------------------------------
# Base Amen 2-bar phrase — hand-authored from reference material.
# Kick pattern: 1, 1.75, 3, 3.5 (bar 1) | 1.25, 2.75, 3 (bar 2)
# Snare pattern: beat 2, 4 (bar 1) | beat 2, 4 + ghost 3.5 (bar 2)
# Pitches are dummy sentinels — drums use preset-baked pitch envelopes.
# ---------------------------------------------------------------------------

_KICK_FREQ = 55.0  # dummy; drum_voice internally ignores for its sweep env
_SNARE_FREQ = 200.0


def _build_base_kick_phrase() -> Phrase:
    # NoteEvents used as bases for composition helpers (diminish, augment, etc.)
    # must be authored with linear amp=, not amp_db=, because dataclasses.replace
    # inside those helpers fails if both amp and amp_db are set. The amp_db
    # comments annotate the design intent.
    pattern_s16 = [
        # (step_in_32, amp_db)  — step measured in 16ths within a 32-step 2-bar loop
        (0, -3.0),  # bar 1, beat 1
        (3, -8.0),  # bar 1, beat 1 + 3/16 (the "ee" of 1.75)
        (8, -4.0),  # bar 1, beat 3
        (10, -6.0),  # bar 1, beat 3.5
        (17, -7.0),  # bar 2, beat 1 + 1/16
        (22, -5.0),  # bar 2, beat 2.5
        (24, -4.0),  # bar 2, beat 3
    ]
    events = tuple(
        NoteEvent(
            start=step * S16,
            duration=S16 * 0.9,
            freq=_KICK_FREQ,
            amp=db_to_amp(amp_db),
            velocity=1.0,
        )
        for step, amp_db in pattern_s16
    )
    return Phrase(events=events)


def _build_base_snare_phrase() -> Phrase:
    # See _build_base_kick_phrase for why amp= (not amp_db=) is required here.
    # Amen-style: beat 2 and 4 both bars, plus ghost at 1-and, ghost at 3.5-ee bar 2
    pattern_s16 = [
        (4, -5.0, 1.0),  # bar 1, beat 2 (accent)
        (12, -4.0, 1.1),  # bar 1, beat 4 (strong accent)
        (15, -15.0, 0.6),  # bar 1, ghost before bar 2
        (20, -5.0, 1.0),  # bar 2, beat 2
        (27, -13.0, 0.7),  # bar 2, ghost at 3.75
        (28, -4.0, 1.1),  # bar 2, beat 4
        (30, -16.0, 0.55),  # bar 2, ghost at 4.5
    ]
    events = tuple(
        NoteEvent(
            start=step * S16,
            duration=S16 * 0.7,
            freq=_SNARE_FREQ,
            amp=db_to_amp(amp_db),
            velocity=velocity,
        )
        for step, amp_db, velocity in pattern_s16
    )
    return Phrase(events=events)


_BASE_KICK = _build_base_kick_phrase()
_BASE_SNARE = _build_base_snare_phrase()

# ---------------------------------------------------------------------------
# Per-bar kick timbre overrides (Groove B + Final Push).
# Each variant is a synth-override dict layered on the base preset.
# ---------------------------------------------------------------------------

_KICK_DIRTY: dict = {}  # baseline `distorted_hardkick`, no override

# The kick variants preserve the preset's shaper="preamp" (quality flux-domain
# distortion). Only tone / envelope params are morphed. Tiny drive bumps via
# `shaper_drive` are OK when a variant wants extra push.
_KICK_FM: dict = {
    "tone_type": "fm",
    "tone_fm_ratio": 1.41,
    "tone_fm_index": 4.0,
    "tone_fm_index_envelope": [
        {"time": 0.0, "value": 1.0},
        {"time": 0.08, "value": 0.12, "curve": "exponential"},
        {"time": 1.0, "value": 0.0, "curve": "linear"},
    ],
    "shaper_drive": 0.6,
}

_KICK_FOLDBACK: dict = {
    "tone_shaper": "foldback",
    "tone_shaper_drive": 0.6,
    "tone_shaper_mix": 0.65,
    "shaper_drive": 0.7,
}

_KICK_DIVE: dict = {
    "tone_pitch_envelope": [
        {"time": 0.0, "value": 6.0},
        {"time": 0.03, "value": 2.0, "curve": "bezier", "cx": 0.1, "cy": 0.9},
        {"time": 0.14, "value": 1.0, "curve": "exponential"},
    ],
    "tone_decay_s": 0.36,
    "shaper_drive": 0.65,
}


def _kick_variant_for_bar(bar: int) -> dict:
    """Return kick synth overrides for ``bar`` (1-indexed).

    Groove A / early bars stay clean-distorted. Groove B morphs every 4 bars.
    Final push cycles through all four in 2-bar blocks for restless energy.
    """
    if bar < S_GROOVE_B[0]:
        return _KICK_DIRTY
    if bar < S_FINAL_PUSH[0]:
        phase = (bar - S_GROOVE_B[0]) // 4
        return [
            _KICK_DIRTY,
            _KICK_FM,
            _KICK_FOLDBACK,
            _KICK_FM,
            _KICK_DIVE,
            _KICK_FOLDBACK,
        ][phase % 6]
    if bar < S_OUTRO[0]:
        phase = (bar - S_FINAL_PUSH[0]) // 2
        return [
            _KICK_FOLDBACK,
            _KICK_FM,
            _KICK_DIVE,
            _KICK_FOLDBACK,
            _KICK_DIRTY,
            _KICK_FM,
        ][phase % 6]
    return _KICK_DIVE


# ---------------------------------------------------------------------------
# Hats: euclidean(11, 16) scaffold with open-hat accents and choke group.
# Open-hat steps are flagged; other steps play closed hat via synth overrides.
# ---------------------------------------------------------------------------

_HAT_STEPS: tuple[bool, ...] = euclidean_pattern(11, 16, rotation=0)
# Open-hat accents fall on "a" of beat 2 + beat 4 — classic offbeat openness.
_OPEN_HAT_STEPS: frozenset[int] = frozenset({7, 15})
_HAT_SUBDIV_DB: dict[int, float] = {0: -8.0, 1: -14.0, 2: -11.0, 3: -15.0}

_CLOSED_HAT_SYNTH: dict = {
    "exciter_type": "click",
    "exciter_level": 0.08,
    "exciter_decay_s": 0.005,
    "exciter_center_hz": 8500.0,
    "metallic_type": "partials",
    "metallic_level": 1.0,
    "metallic_decay_s": 0.045,
    "metallic_n_partials": 6,
    "metallic_brightness": 0.9,
    "metallic_filter_mode": "bandpass",
    "metallic_filter_q": 1.6,
    "metallic_filter_cutoff_hz": 9500.0,
    "noise_type": "bandpass",
    "noise_level": 0.12,
    "noise_decay_s": 0.045,
    "shaper": "rate_reduce",
    "shaper_drive": 0.3,
    "reduce_ratio": 3.0,
    "shaper_mix": 0.8,
}

_OPEN_HAT_SYNTH: dict = {
    **_CLOSED_HAT_SYNTH,
    "metallic_decay_s": 0.22,
    "noise_decay_s": 0.22,
    "metallic_filter_cutoff_hz": 8500.0,
    "shaper_mix": 0.7,
}

# ---------------------------------------------------------------------------
# Ghost perc: drum_voice with modal resonator tuned to chord tones.
# Turing machine drives the step pattern on a 32nd-note grid.
# ---------------------------------------------------------------------------

_GHOST_TURING = TuringMachine(
    length=13,
    flip_probability=0.05,
    # Tones here are just for the TuringMachine API; we use the indices, not ratios.
    tones=(1.0, 9.0 / 8.0, 7.0 / 6.0, 5.0 / 4.0, 4.0 / 3.0, 3.0 / 2.0, 7.0 / 4.0),
    seed=19,
)


def _ghost_pattern_for_section(
    n_steps: int, density: float, phase_seed: int
) -> list[float]:
    """Turing-derived ratio sequence, gated by density threshold."""
    ratios = _GHOST_TURING.generate(n_steps)
    # Use seed-offset noise to gate per step (deterministic from phase_seed).
    import random

    rng = random.Random(phase_seed * 7919 + 131)
    return [r if rng.random() < density else 0.0 for r in ratios]


# ---------------------------------------------------------------------------
# Automation builders
# ---------------------------------------------------------------------------


def _drum_bus_cutoff_automation() -> AutomationSpec:
    """K35 drum-bus cutoff: gate-opening intro, wobble Grooves, open in push."""
    return AutomationSpec(
        target=AutomationTarget(kind="control", name="cutoff_hz"),
        segments=(
            AutomationSegment(
                start=0.0,
                end=T_BREAK_IN,
                shape="hold",
                value=650.0,
            ),
            AutomationSegment(
                start=T_BREAK_IN,
                end=T_GROOVE_A,
                shape="exp",
                start_value=650.0,
                end_value=4500.0,
            ),
            # Groove A: slow wobble around 3 kHz (euphonic — never fully closed)
            AutomationSegment(
                start=T_GROOVE_A,
                end=T_BREAKDOWN,
                shape="sine_lfo",
                freq_hz=0.28,
                depth=800.0,
                offset=3000.0,
            ),
            # Breakdown: rise to 12 kHz so glitch layer rings
            AutomationSegment(
                start=T_BREAKDOWN,
                end=T_GROOVE_B,
                shape="exp",
                start_value=3000.0,
                end_value=12000.0,
            ),
            # Groove B: held bright
            AutomationSegment(
                start=T_GROOVE_B,
                end=T_FINAL_PUSH,
                shape="hold",
                value=9000.0,
            ),
            # Final push: wobble back in, faster
            AutomationSegment(
                start=T_FINAL_PUSH,
                end=T_OUTRO,
                shape="sine_lfo",
                freq_hz=0.55,
                depth=2000.0,
                offset=6500.0,
            ),
            # Outro: close down
            AutomationSegment(
                start=T_OUTRO,
                end=TOTAL_DUR,
                shape="exp",
                start_value=6500.0,
                end_value=900.0,
            ),
        ),
        default_value=4500.0,
        mode="replace",
    )


def _bass_cutoff_automation() -> AutomationSpec:
    """Bass opens across the piece: 420 Hz -> 4 kHz, modulating with sections."""
    return AutomationSpec(
        target=AutomationTarget(kind="synth", name="cutoff_hz"),
        segments=(
            AutomationSegment(
                start=0.0,
                end=T_GROOVE_A,
                shape="hold",
                value=600.0,
            ),
            AutomationSegment(
                start=T_GROOVE_A,
                end=T_BREAKDOWN,
                shape="exp",
                start_value=600.0,
                end_value=1600.0,
            ),
            AutomationSegment(
                start=T_BREAKDOWN,
                end=T_GROOVE_B,
                shape="exp",
                start_value=1600.0,
                end_value=1000.0,
            ),
            AutomationSegment(
                start=T_GROOVE_B,
                end=T_FINAL_PUSH,
                shape="exp",
                start_value=1000.0,
                end_value=3000.0,
            ),
            AutomationSegment(
                start=T_FINAL_PUSH,
                end=T_OUTRO,
                shape="exp",
                start_value=3000.0,
                end_value=4500.0,
            ),
            AutomationSegment(
                start=T_OUTRO,
                end=TOTAL_DUR,
                shape="exp",
                start_value=4500.0,
                end_value=500.0,
            ),
        ),
        default_value=680.0,
        mode="replace",
    )


def _pad_mix_db_automation() -> AutomationSpec:
    """Pad level ride: all values shifted ~6 dB down for backgrounded mix.

    Pad lives primarily as reverb tail (hall send +3 dB); the voice-level
    curve is secondary texture. Breakdown still gets the relative lift
    (6 dB louder than grooves) but absolute values sit in the -10 to -16
    dB band rather than -4 to -10 so the lead and drums carry the piece.
    """
    return AutomationSpec(
        target=AutomationTarget(kind="control", name="mix_db"),
        segments=(
            AutomationSegment(
                start=0.0,
                end=T_BREAK_IN,
                shape="linear",
                start_value=-15.0,
                end_value=-13.0,
            ),
            AutomationSegment(
                start=T_BREAK_IN,
                end=T_GROOVE_A,
                shape="linear",
                start_value=-13.0,
                end_value=-14.0,
            ),
            AutomationSegment(
                start=T_GROOVE_A,
                end=T_BREAKDOWN,
                shape="linear",
                start_value=-14.0,
                end_value=-16.0,
            ),
            AutomationSegment(
                start=T_BREAKDOWN,
                end=T_GROOVE_B,
                shape="linear",
                start_value=-16.0,
                end_value=-10.0,
            ),
            AutomationSegment(
                start=T_GROOVE_B,
                end=T_FINAL_PUSH,
                shape="linear",
                start_value=-10.0,
                end_value=-14.0,
            ),
            AutomationSegment(
                start=T_FINAL_PUSH,
                end=T_OUTRO,
                shape="linear",
                start_value=-14.0,
                end_value=-15.0,
            ),
            AutomationSegment(
                start=T_OUTRO,
                end=TOTAL_DUR,
                shape="linear",
                start_value=-15.0,
                end_value=-10.0,
            ),
        ),
        default_value=-8.0,
        mode="replace",
    )


def _intensity_macro_automation() -> AutomationSpec:
    """Macro ramp: 0 intro -> 0.5 groove -> 0.3 breakdown -> 1.0 push -> 0 outro."""
    return AutomationSpec(
        target=AutomationTarget(kind="control", name="mix"),
        segments=(
            AutomationSegment(
                start=0.0,
                end=T_GROOVE_A,
                shape="linear",
                start_value=0.1,
                end_value=0.45,
            ),
            AutomationSegment(
                start=T_GROOVE_A,
                end=T_BREAKDOWN,
                shape="linear",
                start_value=0.45,
                end_value=0.6,
            ),
            AutomationSegment(
                start=T_BREAKDOWN,
                end=T_GROOVE_B,
                shape="linear",
                start_value=0.6,
                end_value=0.35,
            ),
            AutomationSegment(
                start=T_GROOVE_B,
                end=T_FINAL_PUSH,
                shape="linear",
                start_value=0.35,
                end_value=0.75,
            ),
            AutomationSegment(
                start=T_FINAL_PUSH,
                end=T_OUTRO,
                shape="linear",
                start_value=0.75,
                end_value=1.0,
            ),
            AutomationSegment(
                start=T_OUTRO,
                end=TOTAL_DUR,
                shape="linear",
                start_value=1.0,
                end_value=0.0,
            ),
        ),
        clamp_min=0.0,
        clamp_max=1.0,
    )


# ---------------------------------------------------------------------------
# Score build
# ---------------------------------------------------------------------------


def build_score() -> Score:
    score = Score(
        f0_hz=F0_HZ,
        master_effects=DEFAULT_MASTER_EFFECTS,
        timing_humanize=TimingHumanizeSpec(
            preset="tight_ensemble",
            ensemble_amount_ms=6.5,
        ),
    )

    # ---- Macros ----
    score.add_macro("intensity", default=0.1, automation=_intensity_macro_automation())

    # ---- Send buses ----
    drum_bus = setup_drum_bus(score, style="weighty")
    # Append a K35 lowpass to the drum bus for piece-wide filter wobble.
    # The bus was just created with the "weighty" chain; we mutate its effects.
    drum_bus_spec = next(b for b in score.send_buses if b.name == drum_bus)
    drum_bus_spec.effects.append(
        EffectSpec(
            "analog_filter",
            {
                "filter_topology": "k35",
                "mode": "lp",
                "resonance_q": 0.7,
                # Drive 0.45 -> 0.15: the "weighty" bus already handles
                # saturation; the K35 on top was eating transients (7.4 dB
                # crest loss on the chain). Resonance trimmed too; we want
                # wobble, not a narrow peak.
                "filter_drive": 0.15,
                "cutoff_hz": 4500.0,
                "mix": 0.8,
            },
            automation=[_drum_bus_cutoff_automation()],
        )
    )

    # Hall: "Large & Deep" IR for ethereal, long-tail reverb character.
    # Wet return is shaped via bricasti's built-in tone controls (highpass
    # to clear sub mud, lowpass for dark velvet character, negative tilt
    # to emphasise the low/body bands). Return pushed +3 dB so pad sends
    # really bloom; the eq block below is a safety lowpass on top.
    score.add_send_bus(
        "hall",
        effects=[
            EffectSpec(
                "bricasti",
                {
                    "ir_name": "1 Halls 08 Large & Deep",
                    "wet": 1.0,
                    "highpass_hz": 200.0,
                    "lowpass_hz": 5000.0,
                    "tilt_db": -2.5,
                },
            ),
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
                            # High shelf (gentle) rolls off further brightness.
                            "kind": "high_shelf",
                            "freq_hz": 3500.0,
                            "gain_db": -3.0,
                        },
                    ],
                },
            ),
        ],
        return_db=-1.0,
    )

    score.add_send_bus(
        "delay",
        effects=[
            EffectSpec(
                "mod_delay",
                {
                    "delay_ms": BEAT * 750.0,  # 3/4 beat in ms
                    "feedback": 0.35,
                    "mix": 1.0,
                    "mod_depth_ms": 1.8,
                    "mod_rate_hz": 0.28,
                },
            ),
        ],
        return_db=-8.0,
    )

    score.add_send_bus(
        "glitch_fx",
        effects=[
            EffectSpec(
                "analog_filter",
                {
                    "filter_topology": "k35",
                    "mode": "lp",
                    "resonance_q": 0.9,
                    # Drive dialed 0.55 -> 0.25: analysis flagged 7.5 dB
                    # crest loss on this send chain. Glitch character lives
                    # in the phaser + dragonfly downstream; the K35 just
                    # needs resonance shape, not compression.
                    "filter_drive": 0.25,
                    "cutoff_hz": 3500.0,
                    "mix": 0.85,
                },
            ),
            EffectSpec(
                "phaser",
                {"rate_hz": 0.22, "depth": 0.5, "feedback": 0.3, "mix": 0.4},
            ),
            EffectSpec(
                "dragonfly",
                {
                    "variant": "hall",
                    "wet_level": 70.0,
                    "dry_level": 0.0,
                    "decay_s": 2.2,
                    "size_m": 22.0,
                },
            ),
        ],
        return_db=-9.0,
    )

    # ---- Voices ----
    _add_kick(score, drum_bus)
    _add_snare(score, drum_bus)
    _add_hats(score, drum_bus)
    _add_ghost_perc(score, drum_bus)
    _add_bass(score)
    _add_sub(score)
    _add_reese(score)
    _add_pad(score)
    _add_pad_lead(score)
    _add_glitch_lead(score)
    _add_riser(score)

    # ---- Note placement ----
    _place_kick(score)
    _place_snare(score)
    _place_hats(score)
    _place_ghost_perc(score)
    _place_bass(score)
    _place_sub(score)
    _place_reese(score)
    _place_pad(score)
    _place_pad_lead(score)
    _place_glitch_lead(score)
    _place_fills(score)
    _place_riser(score)
    _place_section_markers(score)

    return score


# ---------------------------------------------------------------------------
# Voice definitions
# ---------------------------------------------------------------------------


def _add_kick(score: Score, drum_bus: str) -> None:
    # The `distorted_hardkick` preset carries its distortion via
    # shaper="preamp" (flux-domain transformer saturation, quality dist).
    # No voice-level drive stacking needed — that was producing papery
    # 2-8 kHz buildup. Voice chain is compression + bass-body preamp.
    add_drum_voice(
        score,
        "kick",
        engine="drum_voice",
        preset="distorted_hardkick",
        drum_bus=drum_bus,
        send_db=-1.0,
        effects=[
            EffectSpec("compressor", {"preset": "kick_punch"}),
            EffectSpec("preamp", {"preset": "kick_body"}),
        ],
        # Kick preset revision (preamp shaper, cleaner) runs about 2 dB quieter
        # than the old tanh+0.6 version. Compressor was idle — bumped so
        # kick_punch can do its job and keep the backbeat solid.
        mix_db=2.0,
    )


def _add_snare(score: Score, drum_bus: str) -> None:
    add_drum_voice(
        score,
        "snare",
        engine="drum_voice",
        preset="snare_digital_fuzz",
        drum_bus=drum_bus,
        send_db=-3.0,
        effects=[EffectSpec("compressor", {"preset": "snare_punch"})],
        # Snare preset revision (noise-forward) runs quieter than the old
        # FM-heavy version; bumped from -3.0 -> +1.0 dB to keep the backbeat
        # present in the mix against the new lead.
        mix_db=1.0,
    )
    score.voices["snare"].sends.append(VoiceSend(target="hall", send_db=-10.0))
    score.voices["snare"].sends.append(VoiceSend(target="delay", send_db=-14.0))


def _add_hats(score: Score, drum_bus: str) -> None:
    add_drum_voice(
        score,
        "hats",
        engine="drum_voice",
        preset="hat_rate_reduced",
        drum_bus=drum_bus,
        send_db=-5.0,
        choke_group="hats",
        effects=[
            EffectSpec("compressor", {"preset": "hat_control"}),
            EffectSpec(
                "eq",
                {"bands": [{"kind": "high_shelf", "freq_hz": 8000.0, "gain_db": 2.0}]},
            ),
        ],
        mix_db=-6.0,
    )


def _add_ghost_perc(score: Score, drum_bus: str) -> None:
    # Modal resonator tuned to chord tones; bit-crushed at bit_depth=8 for
    # sparkling digital character rather than abrasive crush.
    add_drum_voice(
        score,
        "ghost_perc",
        engine="drum_voice",
        synth_overrides={
            "exciter_type": "click",
            "exciter_level": 0.09,
            "exciter_decay_s": 0.003,
            "exciter_center_hz": 5000.0,
            "tone_type": "modal",
            "tone_level": 1.0,
            "tone_decay_s": 0.18,
            "modal_n_modes": 5,
            "modal_mode_ratios": [1.0, 7.0 / 4.0, 5.0 / 2.0, 7.0 / 2.0, 9.0 / 2.0],
            "modal_mode_amps": [1.0, 0.6, 0.45, 0.3, 0.2],
            "modal_damping": 0.35,
            "modal_coupling": 0.2,
            "modal_dispersion": 0.25,
            "noise_type": "bandpass",
            "noise_level": 0.04,
            "noise_decay_s": 0.035,
            "noise_center_ratio": 28.0,
            "shaper": "bit_crush",
            "bit_depth": 8.0,
            "shaper_drive": 0.35,
            "shaper_mix": 0.75,
        },
        drum_bus=drum_bus,
        send_db=-6.0,
        effects=[EffectSpec("compressor", {"preset": "hat_control"})],
        mix_db=-8.0,
    )
    score.voices["ghost_perc"].sends.append(VoiceSend(target="glitch_fx", send_db=-6.0))
    score.voices["ghost_perc"].sends.append(VoiceSend(target="hall", send_db=-13.0))


def _add_bass(score: Score) -> None:
    # acid_bass (diode Q=10.87) through native tube-mode drive + kick_duck
    # with fast 55 ms release override for 172 BPM sidechain sanity.
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "polyblep",
            "preset": "acid_bass",
            "cutoff_hz": 680.0,
        },
        effects=[
            EffectSpec(
                "tube",
                {"character": "triode", "drive": 0.55, "tone": 0.5, "mix": 0.7},
            ),
            EffectSpec(
                "compressor",
                {
                    "preset": "kick_duck",
                    "sidechain_source": "kick",
                    "release_ms": 55.0,  # 172 BPM override
                },
            ),
        ],
        # Dialed -4.0 -> -9.0 dB: acid_bass resonant-ladder stabs were
        # producing a "bouncing ball" character in Groove B around 1:20,
        # especially as the cutoff automation rose into the 2-3 kHz band
        # where Q=10.87 peaks became intrusive.
        mix_db=-9.0,
        normalize_peak_db=-6.0,
        velocity_humanize=None,
        automation=[_bass_cutoff_automation()],
    )


def _add_sub(score: Score) -> None:
    # Pure sine sub at partial=0.5 (49 Hz) for DnB weight.
    score.add_voice(
        "sub",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "sine",
            "cutoff_hz": 260.0,
            "resonance_q": 0.7,
            "filter_drive": 0.0,
            "attack": 0.004,
            "decay": 0.08,
            "sustain_level": 0.9,
            "release": 0.18,
        },
        effects=[
            EffectSpec(
                "compressor",
                {
                    "preset": "kick_duck",
                    "sidechain_source": "kick",
                    "release_ms": 55.0,
                },
            ),
        ],
        mix_db=-5.0,
        normalize_peak_db=-8.0,
        velocity_humanize=None,
    )


def _add_reese(score: Score) -> None:
    """Reese-style sub-bass layer — two detuned saws through diode LP.

    Classic DnB Reese character: slow LFO on cutoff for the breathing
    wobble, heavy sidechain from kick so it ducks under each hit.  Only
    plays during Groove B + Final Push drops (see `_place_reese`), so it
    arrives as a "new texture" in the second half of the piece.
    """
    score.add_voice(
        "reese",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "saw",
            "pulse_width": 0.5,
            # Osc 2: detuned partner saw for the classic Reese beating
            "osc2_level": 0.9,
            "osc2_waveform": "saw",
            "osc2_detune_cents": 7.0,
            "osc2_semitones": 0.0,
            # Diode ladder LP for analog warmth; filter envelope off so
            # character comes from the LFO sweep on cutoff, not per-note env
            "filter_mode": "lowpass",
            "filter_topology": "diode",
            "cutoff_hz": 420.0,
            "resonance_q": 1.6,
            "filter_drive": 0.25,
            "filter_env_amount": 0.0,
            "attack": 0.025,
            "decay": 0.1,
            "sustain_level": 0.95,
            "release": 0.35,
        },
        effects=[
            EffectSpec(
                "compressor",
                {
                    "preset": "kick_duck",
                    "sidechain_source": "kick",
                    "release_ms": 55.0,
                },
            ),
        ],
        mix_db=-11.0,
        normalize_peak_db=-8.0,
        velocity_humanize=None,
        modulations=[
            # Slow cutoff LFO — the Reese-breathing signature.  Low rate
            # (0.18 Hz = cycle every ~5.5 s), wide depth centered on the
            # base cutoff of 420 Hz; the LFO adds swing, doesn't replace.
            ModConnection(
                source=OscillatorSource(
                    rate_hz=0.18,
                    waveshape="sine",
                    stereo=False,
                ),
                target=AutomationTarget(kind="synth", name="cutoff_hz"),
                amount=280.0,
                bipolar=True,
                mode="add",
                name="reese_wobble",
            ),
        ],
    )


def _add_pad(score: Score) -> None:
    # Additive partials with septimal voicings; long release for harmonic bloom.
    # kick_duck_hard @ 300 ms release pumps slowly with the kick — feature.
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "additive",
            "preset": "soft_pad",
            # Ethereal revision: slow attack (2.5s) so notes bloom in rather
            # than arrive; long release (5s) for cross-bar tail overlap;
            # darker brightness (0.12) + stronger tilt (-0.5) pulls the
            # upper partials down so the pad breathes as low harmonic bed.
            "brightness": 0.12,
            "brightness_tilt": -0.5,
            "attack": 2.5,
            "release": 5.0,
        },
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            # Tighter HP removes rumble that the long
                            # reverb tails would muddy.
                            "kind": "highpass",
                            "cutoff_hz": 200.0,
                            "slope_db_per_oct": 12,
                        },
                        {
                            # Darker lowpass (2800 -> 2200 Hz) for the
                            # "backgrounded" feel. Additive partials above
                            # this would fight with the lead motif.
                            "kind": "lowpass",
                            "cutoff_hz": 2200.0,
                            "slope_db_per_oct": 12,
                        },
                    ],
                },
            ),
            EffectSpec(
                "compressor",
                {"preset": "kick_duck", "sidechain_source": "kick"},
            ),
        ],
        # Pad dialed -8 -> -14 dB; hall send lifted -2 -> +3 so the voice
        # lives primarily as reverb tail (reverb-dominant character is the
        # whole "ethereal" shape).
        mix_db=-14.0,
        sends=[
            VoiceSend(target="hall", send_db=3.0),
            VoiceSend(target="delay", send_db=-6.0),
        ],
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        automation=[_pad_mix_db_automation()],
    )


def _add_pad_lead(score: Score) -> None:
    """Articulate pad voice that takes the melody in the outro.

    Distinct from `pad` (harmonic bed): shorter envelope, clearer partials,
    heavier hall send, no sidechain duck — it's the "song dissolving into
    itself" voice. Uses a modal synth_voice preset for bell-ish clarity that
    still blends with the pad harmonic bed.
    """
    score.add_voice(
        "pad_lead",
        synth_defaults={
            "engine": "synth_voice",
            "osc_type": "polyblep",
            "waveform": "triangle",
            "partials_type": "additive",
            "partials_brightness": 0.35,
            "filter_topology": "ladder",
            "filter_cutoff_hz": 1600.0,
            "resonance_q": 0.9,
            "attack": 0.12,
            "release": 1.8,
        },
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 280.0,
                            "slope_db_per_oct": 12,
                        },
                        {
                            "kind": "lowpass",
                            "cutoff_hz": 4500.0,
                            "slope_db_per_oct": 12,
                        },
                    ],
                },
            ),
        ],
        pan=-0.1,
        mix_db=-7.0,
        sends=[
            VoiceSend(target="hall", send_db=0.0),
            VoiceSend(target="delay", send_db=-8.0),
        ],
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
    )


def _add_riser(score: Score) -> None:
    """Filter-sweep riser voice for transitions into drops.

    Single long white-noise note per riser, gated by its own envelope.
    Cutoff is swept via note-level automation (linear in cents feel, exp
    in Hz) so it reads as a proper pre-drop announcement. Sent heavy to
    glitch_fx so the K35 / phaser / dragonfly chain gives it the "pre-drop
    whoosh" character.
    """
    score.add_voice(
        "riser",
        synth_defaults={
            "engine": "synth_voice",
            "osc_type": None,
            "partials_type": None,
            "noise_type": "white",
            "noise_level": 1.0,
            "filter_topology": "k35",
            "filter_mode": "lowpass",
            "filter_cutoff_hz": 400.0,
            "resonance_q": 2.2,
            "filter_drive": 0.35,
            "attack": 0.02,
            "release": 0.4,
        },
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 350.0,
                            "slope_db_per_oct": 12,
                        },
                    ],
                },
            ),
        ],
        mix_db=-3.0,
        normalize_peak_db=-10.0,
        velocity_humanize=None,
        sends=[
            VoiceSend(target="glitch_fx", send_db=-2.0),
            VoiceSend(target="hall", send_db=-8.0),
        ],
    )


def _add_glitch_lead(score: Score) -> None:
    # chua_scatter preset (diode ladder + bit_crush 7). Pitches locked to
    # septimal ratios. ChaoticSource on cutoff + audio-rate OscillatorSource
    # (stereo=False!) on osc2_detune_cents for restless timbre.
    score.add_voice(
        "glitch_lead",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "chua_scatter",
            "osc_chaos_rate_hz": 18.0,
            "osc_chaos_amount": 0.9,
            "osc_chaos_symmetry": 0.1,
            "attack": 0.004,
            "release": 0.45,
            "bit_depth": 8.0,  # sparkle, not rubble
            "shaper_mix": 0.55,
        },
        effects=[],
        pan=0.15,
        mix_db=-11.0,
        normalize_peak_db=-10.0,
        sends=[
            VoiceSend(target="glitch_fx", send_db=-3.0),
            VoiceSend(target="delay", send_db=-10.0),
            VoiceSend(target="hall", send_db=-9.0),
        ],
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble"),
        modulations=[
            ModConnection(
                source=ChaoticSource(
                    system="lorenz",
                    rate_hz=6.0,
                    amount=0.7,
                    symmetry=0.0,
                    seed=83,
                ),
                target=AutomationTarget(kind="synth", name="cutoff_hz"),
                amount=1800.0,
                bipolar=True,
                mode="add",
                name="glitch_chaos_cutoff",
            ),
            ModConnection(
                source=OscillatorSource(
                    rate_hz=180.0,
                    waveshape="sine",
                    stereo=False,  # per-sample synth target requires mono
                ),
                target=AutomationTarget(kind="synth", name="osc2_detune_cents"),
                amount=35.0,
                bipolar=True,
                mode="add",
                name="glitch_audio_fm",
            ),
            # Intensity macro rides filter_drive — as the piece intensifies
            # the glitch_lead pushes harder into diode-ladder saturation.
            ModConnection(
                source=MacroSource(name="intensity"),
                target=AutomationTarget(kind="synth", name="filter_drive"),
                amount=0.35,
                bipolar=False,
                mode="add",
                name="intensity_filter_drive",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Note placement
# ---------------------------------------------------------------------------


def _place_kick(score: Score) -> None:
    """Place the mutated Amen kick across all active sections."""
    _place_drum_phrase(
        score,
        voice_name="kick",
        base=_BASE_KICK,
        section_key="break_in",
        start_bar=S_BREAK_IN[0],
        end_bar=S_BREAK_IN[1],
        subdivide_prob=0.06,
        shift_amount=S16 * 0.05,
        drop_prob=0.0,
        accent_drift=0.05,
        use_kick_variants=False,
    )
    _place_drum_phrase(
        score,
        voice_name="kick",
        base=_BASE_KICK,
        section_key="groove_a",
        start_bar=S_GROOVE_A[0],
        end_bar=S_GROOVE_A[1],
        subdivide_prob=0.08,
        shift_amount=S16 * 0.07,
        drop_prob=0.02,
        accent_drift=0.08,
        use_kick_variants=False,
    )
    # Breakdown — kick drops. Only downbeats survive, very sparse.
    for bar in range(S_BREAKDOWN[0], S_BREAKDOWN[1]):
        if (bar - S_BREAKDOWN[0]) % 4 == 0:
            score.add_note(
                "kick",
                start=_pos(bar, 1),
                duration=S16 * 0.9,
                freq=_KICK_FREQ,
                amp_db=-4.0,
                velocity=1.0,
            )
    _place_drum_phrase(
        score,
        voice_name="kick",
        base=_BASE_KICK,
        section_key="groove_b",
        start_bar=S_GROOVE_B[0],
        end_bar=S_GROOVE_B[1],
        subdivide_prob=0.08,
        shift_amount=S16 * 0.07,
        drop_prob=0.02,
        accent_drift=0.08,
        use_kick_variants=True,
    )
    _place_drum_phrase(
        score,
        voice_name="kick",
        base=_BASE_KICK,
        section_key="final_push",
        start_bar=S_FINAL_PUSH[0],
        end_bar=S_FINAL_PUSH[1],
        subdivide_prob=0.08,
        shift_amount=S16 * 0.07,
        drop_prob=0.03,
        accent_drift=0.10,
        use_kick_variants=True,
    )
    _place_drum_phrase(
        score,
        voice_name="kick",
        base=_BASE_KICK,
        section_key="outro",
        start_bar=S_OUTRO[0],
        end_bar=S_OUTRO[1] - 4,  # leave 4 bars bare for reverb tail
        subdivide_prob=0.04,
        shift_amount=S16 * 0.04,
        drop_prob=0.15,  # progressively drop out
        accent_drift=0.10,
        use_kick_variants=True,
    )


def _place_snare(score: Score) -> None:
    _place_drum_phrase(
        score,
        voice_name="snare",
        base=_BASE_SNARE,
        section_key="break_in",
        start_bar=S_BREAK_IN[0] + 2,  # snare enters 2 bars after kick
        end_bar=S_BREAK_IN[1],
        subdivide_prob=0.10,
        shift_amount=S16 * 0.06,
        drop_prob=0.05,
        accent_drift=0.12,
        use_kick_variants=False,
    )
    _place_drum_phrase(
        score,
        voice_name="snare",
        base=_BASE_SNARE,
        section_key="groove_a",
        start_bar=S_GROOVE_A[0],
        end_bar=S_GROOVE_A[1],
        subdivide_prob=0.15,
        shift_amount=S16 * 0.07,
        drop_prob=0.04,
        accent_drift=0.14,
        use_kick_variants=False,
    )
    _place_drum_phrase(
        score,
        voice_name="snare",
        base=_BASE_SNARE,
        section_key="breakdown",
        start_bar=S_BREAKDOWN[0],
        end_bar=S_BREAKDOWN[1],
        subdivide_prob=0.18,  # more ghost proliferation during breakdown
        shift_amount=S16 * 0.08,
        drop_prob=0.08,
        accent_drift=0.16,
        use_kick_variants=False,
    )
    _place_drum_phrase(
        score,
        voice_name="snare",
        base=_BASE_SNARE,
        section_key="groove_b",
        start_bar=S_GROOVE_B[0],
        end_bar=S_GROOVE_B[1],
        subdivide_prob=0.17,
        shift_amount=S16 * 0.07,
        drop_prob=0.04,
        accent_drift=0.15,
        use_kick_variants=False,
    )
    _place_drum_phrase(
        score,
        voice_name="snare",
        base=_BASE_SNARE,
        section_key="final_push",
        start_bar=S_FINAL_PUSH[0],
        end_bar=S_FINAL_PUSH[1],
        subdivide_prob=0.18,
        shift_amount=S16 * 0.08,
        drop_prob=0.03,
        accent_drift=0.17,
        use_kick_variants=False,
    )
    _place_drum_phrase(
        score,
        voice_name="snare",
        base=_BASE_SNARE,
        section_key="outro",
        start_bar=S_OUTRO[0],
        end_bar=S_OUTRO[1] - 4,
        subdivide_prob=0.08,
        shift_amount=S16 * 0.06,
        drop_prob=0.18,
        accent_drift=0.10,
        use_kick_variants=False,
    )


def _place_drum_phrase(
    score: Score,
    *,
    voice_name: str,
    base: Phrase,
    section_key: str,
    start_bar: int,
    end_bar: int,
    subdivide_prob: float,
    shift_amount: float,
    drop_prob: float,
    accent_drift: float,
    use_kick_variants: bool,
) -> None:
    """Tile the base 2-bar phrase across ``[start_bar, end_bar)`` with per-section mutation.

    Each 2-bar tile gets its own mutation seed derived from the section seed
    + bar index so patterns drift within a section, not just between sections.
    """
    section_seed = _SEEDS[section_key][voice_name]
    for tile_start in range(start_bar, end_bar, 2):
        tile_end = min(tile_start + 2, end_bar)
        if tile_end <= tile_start:
            break
        tile_seed = section_seed * 101 + (tile_start - start_bar) * 7
        mutated = mutate_rhythm(
            base,
            subdivide_prob=subdivide_prob,
            shift_amount=shift_amount,
            drop_prob=drop_prob,
            accent_drift=accent_drift,
            seed=tile_seed,
        )
        if use_kick_variants and voice_name == "kick":
            # Apply per-bar timbre morph on the kick variant.
            synth_override = _kick_variant_for_bar(tile_start)
        else:
            synth_override = None
        score.add_phrase(
            voice_name,
            mutated,
            start=_pos(tile_start),
            synth=synth_override,
        )


def _place_hats(score: Score) -> None:
    """Place euclidean(11,16) hats across bars where active.

    Closed/open distinction via synth override per step (no separate voice).
    Hats enter at S_BREAK_IN and run through to outro with density variation.
    """
    for bar in range(S_BREAK_IN[0], S_OUTRO[1] - 2):
        # Breakdown thins the hats
        if S_BREAKDOWN[0] <= bar < S_BREAKDOWN[1]:
            active_mask = [i % 2 == 0 for i in range(16)]
        # Intro of break_in is sparse
        elif bar < S_BREAK_IN[0] + 4:
            active_mask = [
                bool(_HAT_STEPS[i]) and i in {0, 4, 8, 12} for i in range(16)
            ]
        else:
            active_mask = list(_HAT_STEPS)

        for step, active in enumerate(active_mask):
            if not active:
                continue
            beat = step // 4 + 1
            n16 = step % 4
            is_open = step in _OPEN_HAT_STEPS
            synth = _OPEN_HAT_SYNTH if is_open else _CLOSED_HAT_SYNTH
            subdiv_db = _HAT_SUBDIV_DB[n16]
            # Open hats get a boost; closed hats stay at subdiv_db
            amp_db = subdiv_db + (4.0 if is_open else 0.0)
            duration = 0.18 if is_open else 0.04
            score.add_note(
                "hats",
                start=_pos(bar, beat, n16),
                duration=duration,
                freq=9000.0,
                amp_db=amp_db,
                velocity=1.0,
                synth=synth,
            )


def _place_ghost_perc(score: Score) -> None:
    """Ghost perc on 32nd grid, Turing-driven, per-section density.

    Freq mapping: pick chord-tone partials from the active chord for modal
    ghost_perc placement. Each note's freq is locked to a chord-tone ratio
    above F0 so the metallic perc sings with the harmony.
    """
    section_configs = [
        ("break_in", S_BREAK_IN, 0.25, 37),
        ("groove_a", S_GROOVE_A, 0.4, 41),
        ("breakdown", S_BREAKDOWN, 0.65, 43),  # dense in breakdown
        ("groove_b", S_GROOVE_B, 0.45, 47),
        ("final_push", S_FINAL_PUSH, 0.55, 53),
        ("outro", (S_OUTRO[0], S_OUTRO[1] - 4), 0.2, 59),
    ]
    for _label, (start_bar, end_bar), density, phase_seed in section_configs:
        n_steps = (end_bar - start_bar) * 32
        pattern = _ghost_pattern_for_section(n_steps, density, phase_seed)
        for step_index, ratio_value in enumerate(pattern):
            if ratio_value == 0.0:
                continue
            # Absolute 32nd-step position
            bar_offset = step_index // 32
            step_in_bar = step_index % 32
            beat = step_in_bar // 8 + 1
            n32 = step_in_bar % 8
            # Chord for the current bar (cycles every 8 bars)
            chord_idx = ((start_bar + bar_offset - 1) // 4) % len(_PAD_PROGRESSION)
            _bass_root, pad_partials = _PAD_PROGRESSION[chord_idx]
            # Pick a partial from the pad voicing, biased by ratio index
            partial_choice = pad_partials[step_index % len(pad_partials)]
            # Ghost perc one octave up for presence
            score.add_note(
                "ghost_perc",
                start=_pos(start_bar + bar_offset, beat, n32 // 2, n32 % 2),
                duration=S32 * 0.85,
                partial=partial_choice * 2.0,
                amp_db=-12.0 + (-4.0 if step_in_bar % 8 == 0 else 0.0),
                velocity=0.6 + 0.4 * ((step_index * 31 + 17) % 100) / 100.0,
            )


def _place_bass(score: Score) -> None:
    """Bass follows the 8-bar chord cycle; root notes on beat 1 + octave stabs on 2.5/4.5."""
    for bar in range(S_GROOVE_A[0], S_OUTRO[1] - 4):
        if S_BREAKDOWN[0] <= bar < S_BREAKDOWN[1]:
            continue  # breakdown drops the bass
        chord_idx = ((bar - 1) // 4) % len(_PAD_PROGRESSION)
        bass_root, _pad = _PAD_PROGRESSION[chord_idx]
        # Root on the one
        score.add_note(
            "bass",
            start=_pos(bar, 1),
            duration=BEAT * 1.8,
            partial=bass_root,
            amp_db=-5.0,
            velocity=1.0,
        )
        # Mid-bar stab on 2 "and"
        score.add_note(
            "bass",
            start=_pos(bar, 2, 2),
            duration=BEAT * 0.4,
            partial=bass_root,
            amp_db=-9.0,
            velocity=0.85,
        )
        # Off-beat stab on bar 2/4 of each chord cycle
        if bar % 2 == 0:
            score.add_note(
                "bass",
                start=_pos(bar, 3, 2),
                duration=BEAT * 0.6,
                partial=bass_root * _R7_6,
                amp_db=-11.0,
                velocity=0.75,
            )
        # Septimal 7 push in Final Push every 4th bar
        if bar >= S_FINAL_PUSH[0] and bar % 4 == 0:
            score.add_note(
                "bass",
                start=_pos(bar, 4, 2),
                duration=BEAT * 0.35,
                partial=bass_root * _R7_4,
                amp_db=-12.0,
                velocity=0.8,
            )


def _place_sub(score: Score) -> None:
    """Sub doubles the bass root at partial=0.5 of root (one octave down)."""
    for bar in range(S_GROOVE_A[0], S_OUTRO[1] - 4):
        if S_BREAKDOWN[0] <= bar < S_BREAKDOWN[1]:
            continue
        chord_idx = ((bar - 1) // 4) % len(_PAD_PROGRESSION)
        bass_root, _pad = _PAD_PROGRESSION[chord_idx]
        score.add_note(
            "sub",
            start=_pos(bar, 1),
            duration=BEAT * 3.8,
            partial=bass_root * 0.5,
            amp_db=-4.0,
            velocity=1.0,
        )


def _place_reese(score: Score) -> None:
    """Reese bass held across Groove B + Final Push drops only.

    Notes sustain 4 bars to let the filter LFO do its work; skip Breakdown
    (kick drops) and the intro/break-in sections (intro is pad-only).
    """
    active_ranges = (S_GROOVE_B, S_FINAL_PUSH)
    for start_bar, end_bar in active_ranges:
        for bar in range(start_bar, end_bar, 4):
            if bar + 4 > end_bar:
                break
            chord_idx = ((bar - 1) // 4) % len(_PAD_PROGRESSION)
            bass_root, _pad = _PAD_PROGRESSION[chord_idx]
            score.add_note(
                "reese",
                start=_pos(bar, 1),
                duration=BAR * 3.95,
                partial=bass_root,
                amp_db=-3.0,
                velocity=1.0,
            )


def _place_pad(score: Score) -> None:
    """Place septimal pad chords every 4 bars (each chord lasts 4 bars)."""
    for bar in range(S_INTRO[0], S_OUTRO[1] - 4, 4):
        chord_idx = ((bar - 1) // 4) % len(_PAD_PROGRESSION)
        _bass_root, pad_partials = _PAD_PROGRESSION[chord_idx]
        for partial in pad_partials:
            score.add_note(
                "pad",
                start=_pos(bar, 1),
                duration=BAR * 3.95,  # full 4-bar hold, slight overlap for bloom
                partial=partial,
                amp_db=-10.0,
                velocity=0.85,
            )


# Through-composed 2-bar lead motif. Pitches are JI ratios relative to
# chord root, voiced an octave up (×2) for presence. Written to be
# recognizable: lands on the otonal 5/4 and 7/4 ("hook" septimal seventh),
# with a faster 32nd run at the end of bar 2 to propel forward.
#
# Motif events: (start_in_bars, duration_in_beats, partial_ratio_over_root, velocity)
_LEAD_MOTIF_A: tuple[tuple[float, float, float, float], ...] = (
    # Bar 1 — question phrase
    (0.0, 0.45, 2.5, 0.95),  # beat 1:   5/4 (major third, ×2 octave = 2.5)
    (0.125, 0.35, 3.0, 0.85),  # beat 1.5: 3/2 (fifth)
    (0.375, 0.60, 3.5, 1.00),  # beat 2.5: 7/4 (septimal 7 — the hook)
    (0.625, 0.35, 3.0, 0.80),  # beat 3.5: 3/2 (step down)
    (0.875, 0.20, 2.5, 0.75),  # beat 4.5: 5/4 (return)
    # Bar 2 — answer phrase, climbs then descends
    (1.0, 0.45, 2.5, 0.95),  # beat 1 bar2: 5/4
    (1.25, 0.40, 4.0, 1.05),  # beat 2:      2.0 (octave) — peak
    (1.5, 0.30, 3.5, 0.95),  # beat 3:      7/4
    (1.75, 0.18, 3.0, 0.85),  # beat 4:      3/2
    (1.875, 0.15, 2.5 * 9 / 8, 0.80),  # beat 4.5:   9/8 of 5/4 (passing)
    (1.9375, 0.12, 3.5, 0.90),  # 32nd run:   7/4
)

# Variant B — same skeleton but syncopated and adds octave leap at end,
# used in Groove B + Final Push to refresh the motif without abandoning it.
_LEAD_MOTIF_B: tuple[tuple[float, float, float, float], ...] = (
    # Bar 1 — delayed entry, rhythmically syncopated
    (0.125, 0.30, 2.5, 0.80),  # start on "1-and"
    (0.375, 0.55, 3.5, 1.00),
    (0.5625, 0.25, 3.0, 0.85),
    (0.75, 0.30, 2.5 * 9 / 8, 0.90),  # 9/8 passing — tension note
    (0.9375, 0.20, 3.5, 0.95),
    # Bar 2 — big octave leap up
    (1.0, 0.40, 3.0, 0.95),
    (1.25, 0.35, 4.0, 1.00),
    (1.5, 0.25, 5.0, 1.10),  # octave + major third — climax
    (1.6875, 0.18, 3.5, 0.90),
    (1.8125, 0.18, 3.0, 0.85),
    (1.9375, 0.15, 2.5, 0.80),
)


def _place_lead_motif(
    score: Score,
    *,
    start_bar: int,
    motif: tuple[tuple[float, float, float, float], ...],
    gain_db_offset: float = 0.0,
) -> None:
    """Place one 2-bar motif instance starting at ``start_bar``.

    Partials in the motif are multiplied by the current chord's bass root
    so the melody transposes with the progression.
    """
    chord_idx = ((start_bar - 1) // 4) % len(_PAD_PROGRESSION)
    bass_root, _pad = _PAD_PROGRESSION[chord_idx]
    for offset_bars, duration_beats, ratio, velocity in motif:
        start = _pos(start_bar) + offset_bars * BAR
        score.add_note(
            "glitch_lead",
            start=start,
            duration=duration_beats * BEAT,
            partial=ratio * bass_root,
            amp_db=-8.0 + gain_db_offset,
            velocity=velocity,
        )


def _place_glitch_lead(score: Score) -> None:
    """Through-composed 2-bar lead motif restated across the piece.

    Motif A rides Groove A (bars 17-36) every 2 bars, restated 10 times.
    Breakdown (bars 37-48) drops to a sparser rearticulation every 4 bars.
    Motif B rides Groove B + Final Push, every 2 bars, for extra lift.
    Outro hints at the motif once before fading.
    """
    # Groove A — motif A every 2 bars, with one bar of rest before each
    # repeat ending so the 2-bar phrase breathes. That's a motif every 2 bars.
    for start_bar in range(S_GROOVE_A[0], S_GROOVE_A[1] - 1, 2):
        _place_lead_motif(score, start_bar=start_bar, motif=_LEAD_MOTIF_A)

    # Breakdown — sparser: restate just the hook notes (7/4 landings) at
    # bar starts, let the glitch_fx send carry the texture.
    for bar in range(S_BREAKDOWN[0], S_BREAKDOWN[1]):
        chord_idx = ((bar - 1) // 4) % len(_PAD_PROGRESSION)
        bass_root, _pad = _PAD_PROGRESSION[chord_idx]
        score.add_note(
            "glitch_lead",
            start=_pos(bar, 2, 2),  # beat 2-and
            duration=BEAT * 0.6,
            partial=3.5 * bass_root,  # the septimal 7 hook
            amp_db=-7.0,
            velocity=1.0,
        )

    # Groove B — motif B every 2 bars, with 4-dB louder in the push for lift
    for start_bar in range(S_GROOVE_B[0], S_GROOVE_B[1] - 1, 2):
        _place_lead_motif(score, start_bar=start_bar, motif=_LEAD_MOTIF_B)

    # Final Push — alternate motif A and B every 4 bars for restless energy
    for start_bar in range(S_FINAL_PUSH[0], S_FINAL_PUSH[1] - 1, 2):
        which = (
            _LEAD_MOTIF_A
            if ((start_bar - S_FINAL_PUSH[0]) // 2) % 2 == 0
            else _LEAD_MOTIF_B
        )
        _place_lead_motif(score, start_bar=start_bar, motif=which, gain_db_offset=1.5)

    # Outro — the pad_lead voice takes the motif now (see _place_pad_lead),
    # so the glitch_lead falls silent and the song "dissolves" onto the
    # smoother pad melody as the piece fades.


def _place_fills(score: Score) -> None:
    """Retrigger fills at bars 31, 55, 79, 103: diminish(snare, 2.0) + echo ghost."""
    fill_bars = [31, 55, 79, 103]
    for bar in fill_bars:
        if bar >= TOTAL_BARS:
            break
        # Diminished snare — fast 32nd roll across the last beat of the bar
        snare_roll = diminish(_BASE_SNARE, 2.0)
        score.add_phrase(
            "snare",
            snare_roll,
            start=_pos(bar, 4),
            time_scale=0.5,
            amp_scale=0.8,
        )
        # Ghost perc echo on the last two 16ths of the bar
        ghost_chord_idx = ((bar - 1) // 4) % len(_PAD_PROGRESSION)
        _r, pad_p = _PAD_PROGRESSION[ghost_chord_idx]
        for n in range(4):
            score.add_note(
                "ghost_perc",
                start=_pos(bar, 4, 2) + n * S32,
                duration=S32 * 0.8,
                partial=pad_p[n % len(pad_p)] * 2.0,
                amp_db=-8.0 + n * -1.5,
                velocity=0.9,
            )


def _place_pad_lead(score: Score) -> None:
    """Pad_lead takes Motif A through the outro as the piece dissolves.

    Two motif statements at the start of the outro, then a final slow
    arpeggio of chord tones to close. Notes voiced in the motif's original
    octave (partial ×2 via the motif table), which sits nicely against
    the pad harmonic bed one octave below.
    """
    first_bar = S_OUTRO[0] + 1
    second_bar = S_OUTRO[0] + 5
    for start_bar in (first_bar, second_bar):
        if start_bar + 2 >= S_OUTRO[1] - 2:
            break
        _place_pad_lead_motif(score, start_bar=start_bar)

    # Final slow arpeggio — four chord tones across 4 bars, dissolving away
    final_bar = S_OUTRO[0] + 9
    if final_bar < S_OUTRO[1] - 2:
        chord_idx = ((final_bar - 1) // 4) % len(_PAD_PROGRESSION)
        bass_root, pad_partials = _PAD_PROGRESSION[chord_idx]
        for beat_offset, partial_idx, amp_db in (
            (0.0, 0, -8.0),
            (2.0, 1, -10.0),
            (4.0, 2, -12.0),
            (6.0, 3, -14.0),
        ):
            score.add_note(
                "pad_lead",
                start=_pos(final_bar) + beat_offset * BEAT,
                duration=BEAT * 2.2,
                partial=pad_partials[partial_idx % len(pad_partials)] * bass_root,
                amp_db=amp_db,
                velocity=0.75,
            )


def _place_pad_lead_motif(score: Score, *, start_bar: int) -> None:
    """Place Motif A on the pad_lead voice transposed by current chord root."""
    chord_idx = ((start_bar - 1) // 4) % len(_PAD_PROGRESSION)
    bass_root, _pad = _PAD_PROGRESSION[chord_idx]
    for offset_bars, duration_beats, ratio, velocity in _LEAD_MOTIF_A:
        start = _pos(start_bar) + offset_bars * BAR
        score.add_note(
            "pad_lead",
            start=start,
            duration=duration_beats * BEAT,
            partial=ratio * bass_root,
            amp_db=-9.0,
            velocity=velocity * 0.9,
        )


def _place_riser(score: Score) -> None:
    """Filter-sweep risers into the two main drops.

    Big riser: bars 71-72 (last 2 bars of Groove B, into Final Push at 73).
    Small riser: bars 47-48 (last 2 bars of Breakdown, into Groove B at 49).

    Each is one long note with cutoff_hz automation that sweeps exp from
    ~400 Hz to ~8 kHz over the full duration, producing the classic DnB
    pre-drop "whoosh" character through the K35 filter + glitch_fx send.
    """
    # Small riser — 2 bars before Groove B drop. Boosted +10 dB relative to
    # first pass: first render had risers at -34 dBFS peak (inaudible).
    _place_single_riser(
        score,
        start_bar=S_GROOVE_B[0] - 2,
        duration_bars=2.0,
        peak_amp_db=3.0,
    )
    # Big riser — 2 bars before Final Push drop
    _place_single_riser(
        score,
        start_bar=S_FINAL_PUSH[0] - 2,
        duration_bars=2.0,
        peak_amp_db=5.0,
    )


def _place_single_riser(
    score: Score,
    *,
    start_bar: int,
    duration_bars: float,
    peak_amp_db: float,
) -> None:
    """Place a single riser note with a note-level cutoff exp sweep."""
    start = _pos(start_bar)
    duration = duration_bars * BAR
    sweep_automation = AutomationSpec(
        target=AutomationTarget(kind="synth", name="cutoff_hz"),
        segments=(
            AutomationSegment(
                start=0.0,
                end=duration,
                shape="exp",
                start_value=400.0,
                end_value=8000.0,
            ),
        ),
        default_value=400.0,
        mode="replace",
    )
    score.add_note(
        "riser",
        start=start,
        duration=duration,
        freq=3000.0,  # freq is mostly irrelevant for noise source
        amp_db=peak_amp_db,
        velocity=1.0,
        automation=[sweep_automation],
    )


def _place_section_markers(score: Score) -> None:
    """Single bright accent ghost hits on the downbeats of major sections.

    Structural markers — a high bright modal hit that breaks the pattern
    momentarily, cuing the listener to the transition. Bars 17, 37, 49, 73
    (Groove A, Breakdown, Groove B, Final Push starts).
    """
    marker_bars = (
        S_GROOVE_A[0],
        S_BREAKDOWN[0],
        S_GROOVE_B[0],
        S_FINAL_PUSH[0],
    )
    marker_synth_override: dict = {
        # Brighter modal partials with longer decay — sticks out against
        # the normal ghost_perc voice. Same engine so it reuses the voice's
        # chain (compressor, glitch_fx send).
        "modal_mode_ratios": [1.0, 2.0, 3.0, 5.0, 7.0],
        "modal_mode_amps": [1.0, 0.8, 0.6, 0.4, 0.3],
        "modal_damping": 0.15,
        "modal_coupling": 0.3,
        "modal_dispersion": 0.35,
        "tone_decay_s": 0.6,
        "metallic_level": 0.0,
        "noise_level": 0.0,
        "shaper_mix": 0.5,
    }
    for bar in marker_bars:
        chord_idx = ((bar - 1) // 4) % len(_PAD_PROGRESSION)
        bass_root, _pad = _PAD_PROGRESSION[chord_idx]
        score.add_note(
            "ghost_perc",
            start=_pos(bar, 1),
            duration=BEAT * 1.5,
            partial=bass_root * 4.0,  # bright — two octaves up
            amp_db=-6.0,
            velocity=1.0,
            synth=marker_synth_override,
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

PIECES: dict[str, PieceDefinition] = {
    "iambic_ghost": PieceDefinition(
        name="iambic_ghost",
        output_name="iambic_ghost",
        build_score=build_score,
        sections=(
            PieceSection("intro", _pos(S_INTRO[0]), _pos(S_INTRO[1])),
            PieceSection("break_in", _pos(S_BREAK_IN[0]), _pos(S_BREAK_IN[1])),
            PieceSection("groove_a", _pos(S_GROOVE_A[0]), _pos(S_GROOVE_A[1])),
            PieceSection("breakdown", _pos(S_BREAKDOWN[0]), _pos(S_BREAKDOWN[1])),
            PieceSection("groove_b", _pos(S_GROOVE_B[0]), _pos(S_GROOVE_B[1])),
            PieceSection("final_push", _pos(S_FINAL_PUSH[0]), _pos(S_FINAL_PUSH[1])),
            PieceSection("outro", _pos(S_OUTRO[0]), _pos(S_OUTRO[1])),
        ),
    ),
}
