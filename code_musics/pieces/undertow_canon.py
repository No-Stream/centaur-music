"""Undertow Canon — otonal light folded through a utonal shadow.

A canon-song in G-centered 7/11-limit JI.  The piece starts with a warm
organ subject, lets additive arps and a quiet pulse animate it, then turns the
same motif through a darker utonal reharmonization before resolving around
shared tones.
"""

from __future__ import annotations

from code_musics.automation import AutomationSegment, AutomationSpec, AutomationTarget
from code_musics.drum_helpers import add_drum_voice, setup_drum_bus
from code_musics.humanize import EnvelopeHumanizeSpec, TimingHumanizeSpec
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS, bricasti_or_reverb
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import EffectSpec, Score, SendBusSpec, VoiceSend

F0_HZ: float = 98.0
BPM: float = 92.0
BEAT: float = 60.0 / BPM
BAR: float = BEAT * 4.0
TOTAL_BARS: int = 80
TOTAL_DUR: float = TOTAL_BARS * BAR

GROUND_START_BAR: int = 1
CANON_START_BAR: int = 17
UNDERTOW_START_BAR: int = 33
COLLISION_START_BAR: int = 49
RETURN_START_BAR: int = 65

OTONAL_HOME: tuple[float, ...] = (1 / 2, 1, 5 / 4, 3 / 2, 7 / 4)
OTONAL_COLOR: tuple[float, ...] = (2 / 3, 1, 4 / 3, 5 / 3, 11 / 6)
UTONAL_SHADOW: tuple[float, ...] = (1 / 2, 7 / 6, 4 / 3, 8 / 5, 7 / 4)
SUSPENDED_BRIDGE: tuple[float, ...] = (3 / 4, 1, 9 / 8, 3 / 2, 7 / 4)
FINAL_RESOLVE: tuple[float, ...] = (1 / 2, 1, 5 / 4, 3 / 2, 2)

SUBJECT: tuple[float, ...] = (2, 9 / 4, 5 / 2, 3, 7 / 2, 3, 5 / 2, 2)
UTONAL_SUBJECT: tuple[float, ...] = (
    2,
    7 / 3,
    8 / 3,
    16 / 5,
    7 / 2,
    16 / 5,
    8 / 3,
    2,
)

GROUND_ROOTS: tuple[float, ...] = (1 / 2, 2 / 3, 3 / 4, 1 / 2)
MAIN_PROGRESSION: tuple[tuple[float, ...], ...] = (
    OTONAL_HOME,
    OTONAL_COLOR,
    SUSPENDED_BRIDGE,
    OTONAL_HOME,
)
SHADOW_PROGRESSION: tuple[tuple[float, ...], ...] = (
    UTONAL_SHADOW,
    SUSPENDED_BRIDGE,
    OTONAL_COLOR,
    UTONAL_SHADOW,
)


def _bar(bar: int, beat: float = 1.0) -> float:
    """Return seconds for 1-indexed bar and beat positions."""
    return (bar - 1) * BAR + (beat - 1.0) * BEAT


def _section_start(bar: int) -> float:
    return _bar(bar)


def _mix_automation(
    start_db: float,
    middle_db: float,
    end_db: float,
) -> AutomationSpec:
    return AutomationSpec(
        target=AutomationTarget(kind="control", name="mix_db"),
        segments=(
            AutomationSegment(
                start=0.0,
                end=_section_start(UNDERTOW_START_BAR),
                shape="linear",
                start_value=start_db,
                end_value=middle_db,
            ),
            AutomationSegment(
                start=_section_start(UNDERTOW_START_BAR),
                end=TOTAL_DUR,
                shape="linear",
                start_value=middle_db,
                end_value=end_db,
            ),
        ),
        default_value=middle_db,
        mode="replace",
    )


def _make_hall_bus() -> SendBusSpec:
    return SendBusSpec(
        name="hall",
        effects=[
            bricasti_or_reverb(
                "1 Halls 07 Large & Dark",
                0.30,
                room_size=0.84,
                damping=0.58,
                lowpass_hz=6200.0,
                highpass_hz=120.0,
            ),
            EffectSpec("tube", {"preset": "triode_glow", "drive": 0.28, "mix": 0.16}),
        ],
        return_db=-2.0,
    )


def _make_bell_delay_bus() -> SendBusSpec:
    return SendBusSpec(
        name="bell_delay",
        effects=[
            EffectSpec(
                "delay",
                {
                    "delay_seconds": BEAT * 0.75,
                    "feedback": 0.26,
                    "mix": 0.24,
                },
            ),
            bricasti_or_reverb(
                "1 Halls 07 Large & Dark",
                0.22,
                room_size=0.80,
                damping=0.62,
                lowpass_hz=7800.0,
                highpass_hz=240.0,
            ),
        ],
        return_db=-4.0,
    )


def build_score() -> Score:
    """Build the Undertow Canon score."""
    score = Score(
        f0_hz=F0_HZ,
        timing_humanize=TimingHumanizeSpec(
            ensemble_amount_ms=7.0,
            follow_strength=0.78,
        ),
        master_input_gain_db=-4.0,
        send_buses=[_make_hall_bus(), _make_bell_delay_bus()],
        master_effects=list(DEFAULT_MASTER_EFFECTS),
    )
    drum_bus = setup_drum_bus(score, style="light", bus_name="pulse_bus")

    _add_voices(score, drum_bus)
    _write_ground(score)
    _write_canon_bloom(score)
    _write_undertow(score)
    _write_collision(score)
    _write_return(score)
    _write_pulse(score)

    return score


def _add_voices(score: Score, drum_bus: str) -> None:
    hall = VoiceSend(target="hall", send_db=-8.0)
    wet_hall = VoiceSend(target="hall", send_db=-5.0)

    score.add_voice(
        "organ_bass",
        synth_defaults={"engine": "organ", "preset": "cathedral"},
        sends=[VoiceSend(target="hall", send_db=-11.0)],
        mix_db=-1.0,
        pan=-0.08,
    )
    score.add_voice(
        "organ_subject",
        synth_defaults={"engine": "organ", "preset": "baroque"},
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        sends=[hall],
        pan=0.12,
    )
    score.add_voice(
        "organ_answer",
        synth_defaults={"engine": "organ", "preset": "warm"},
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        sends=[hall],
        mix_db=-2.0,
        pan=-0.16,
    )
    score.add_voice(
        "organ_shadow",
        synth_defaults={"engine": "organ", "preset": "jazz"},
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        sends=[wet_hall],
        mix_db=-4.0,
        pan=0.04,
    )
    score.add_voice(
        "bell_arp",
        synth_defaults={
            "engine": "additive",
            "preset": "plucked_ji",
            "env": {
                "attack_ms": 8.0,
                "decay_ms": 260.0,
                "sustain_ratio": 0.35,
                "release_ms": 700.0,
            },
        },
        sends=[
            VoiceSend(target="hall", send_db=-8.0),
            VoiceSend(target="bell_delay", send_db=-5.0),
        ],
        mix_db=-6.0,
        pan=0.18,
        automation=[_mix_automation(-18.0, -5.5, -13.0)],
    )
    score.add_voice(
        "grain_shimmer",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "grain_shimmer_dust",
            "attack": 2.6,
            "release": 4.2,
            "grain_ji_lattice": [
                1.0,
                9 / 8,
                7 / 6,
                5 / 4,
                4 / 3,
                3 / 2,
                8 / 5,
                7 / 4,
                2.0,
            ],
        },
        sends=[VoiceSend(target="hall", send_db=-3.0)],
        mix_db=-18.0,
        pan=-0.22,
        automation=[_mix_automation(-24.0, -10.0, -18.0)],
    )

    add_drum_voice(
        score,
        "kick",
        engine="drum_voice",
        preset="808_resonant",
        drum_bus=drum_bus,
        send_db=-9.0,
        effects=[],
        mix_db=-9.5,
        synth_overrides={"tone_decay_s": 0.8, "click_level": 0.16},
    )
    add_drum_voice(
        score,
        "wood",
        engine="drum_voice",
        preset="clave",
        drum_bus=drum_bus,
        send_db=-7.0,
        mix_db=-14.0,
    )
    add_drum_voice(
        score,
        "hat",
        engine="drum_voice",
        preset="closed_hat",
        drum_bus=drum_bus,
        send_db=-12.0,
        effects=[],
        mix_db=-18.0,
        choke_group="hat",
        synth_overrides={"metallic_decay_s": 0.11, "noise_decay_s": 0.06},
    )


def _write_ground(score: Score) -> None:
    _add_ground_roots(score, GROUND_START_BAR, GROUND_ROOTS, velocity=0.62)
    _add_chord(score, "organ_subject", OTONAL_HOME, bar=1, bars=8, amp_db=-17.0)
    _add_subject(score, "organ_subject", SUBJECT, start_bar=5, amp_db=-8.5)
    _add_subject(
        score,
        "organ_answer",
        SUBJECT,
        start_bar=11,
        amp_db=-11.0,
        transpose=3 / 4,
        velocity=0.68,
    )


def _write_canon_bloom(score: Score) -> None:
    _write_chord_cycle(score, "organ_answer", MAIN_PROGRESSION, CANON_START_BAR)
    _add_ground_roots(score, CANON_START_BAR, GROUND_ROOTS, velocity=0.70)
    _add_subject(score, "organ_subject", SUBJECT, start_bar=17, amp_db=-7.5)
    _add_subject(
        score,
        "organ_answer",
        SUBJECT,
        start_bar=19,
        amp_db=-8.5,
        transpose=3 / 2,
        velocity=0.76,
    )
    _add_subject(
        score,
        "organ_shadow",
        SUBJECT,
        start_bar=23,
        amp_db=-11.0,
        transpose=3 / 4,
        velocity=0.66,
    )
    for bar in range(CANON_START_BAR, UNDERTOW_START_BAR):
        _add_bell_bar(score, bar, MAIN_PROGRESSION[(bar - CANON_START_BAR) % 4])


def _write_undertow(score: Score) -> None:
    _write_chord_cycle(score, "organ_shadow", SHADOW_PROGRESSION, UNDERTOW_START_BAR)
    _add_ground_roots(
        score,
        UNDERTOW_START_BAR,
        (1 / 2, 7 / 12, 2 / 3, 1 / 2),
        velocity=0.66,
    )
    _add_subject(
        score,
        "organ_shadow",
        UTONAL_SUBJECT,
        start_bar=33,
        amp_db=-8.0,
        velocity=0.76,
    )
    _add_subject(
        score,
        "organ_answer",
        UTONAL_SUBJECT,
        start_bar=37,
        amp_db=-10.0,
        transpose=3 / 4,
        velocity=0.66,
    )
    _add_grain_chord(score, UTONAL_SHADOW, bar=33, bars=16, amp_db=-22.0)
    for bar in range(UNDERTOW_START_BAR, COLLISION_START_BAR):
        chord = SHADOW_PROGRESSION[(bar - UNDERTOW_START_BAR) % 4]
        _add_bell_bar(score, bar, chord, amp_db=-16.5)


def _write_collision(score: Score) -> None:
    _write_chord_cycle(score, "organ_subject", MAIN_PROGRESSION, COLLISION_START_BAR)
    _write_chord_cycle(score, "organ_shadow", SHADOW_PROGRESSION, COLLISION_START_BAR)
    _add_ground_roots(score, COLLISION_START_BAR, GROUND_ROOTS, velocity=0.74)
    _add_subject(score, "organ_subject", SUBJECT, start_bar=49, amp_db=-7.0)
    _add_subject(
        score,
        "organ_shadow",
        UTONAL_SUBJECT,
        start_bar=51,
        amp_db=-8.8,
        velocity=0.74,
    )
    _add_subject(
        score,
        "organ_answer",
        SUBJECT,
        start_bar=57,
        amp_db=-8.5,
        transpose=3 / 2,
        velocity=0.76,
    )
    _add_grain_chord(score, OTONAL_HOME + UTONAL_SHADOW[1:], bar=49, bars=16)
    for bar in range(COLLISION_START_BAR, RETURN_START_BAR):
        _add_bell_bar(score, bar, MAIN_PROGRESSION[(bar - COLLISION_START_BAR) % 4])


def _write_return(score: Score) -> None:
    _add_ground_roots(
        score,
        RETURN_START_BAR,
        (1 / 2, 3 / 4, 2 / 3, 1 / 2),
        velocity=0.58,
    )
    _add_chord(score, "organ_subject", FINAL_RESOLVE, bar=65, bars=8, amp_db=-15.0)
    _add_subject(
        score,
        "organ_subject",
        SUBJECT,
        start_bar=65,
        amp_db=-10.5,
        velocity=0.62,
    )
    _add_subject(
        score,
        "organ_answer",
        tuple(reversed(SUBJECT)),
        start_bar=69,
        amp_db=-12.5,
        transpose=3 / 4,
        velocity=0.58,
    )
    _add_grain_chord(score, FINAL_RESOLVE, bar=65, bars=13, amp_db=-24.0)
    _add_chord(score, "organ_bass", (1 / 2, 1), bar=77, bars=4, amp_db=-14.0)
    _add_chord(score, "organ_subject", FINAL_RESOLVE, bar=77, bars=4, amp_db=-18.0)
    _add_bell_bar(score, 73, FINAL_RESOLVE, amp_db=-19.0)
    _add_bell_bar(score, 77, FINAL_RESOLVE, amp_db=-22.0)


def _write_pulse(score: Score) -> None:
    for bar in range(17, 65):
        score.add_note(
            "kick",
            start=_bar(bar),
            duration=0.42,
            partial=0.5,
            amp_db=-7.5 if bar >= COLLISION_START_BAR else -9.0,
            velocity=0.80,
        )
        if bar >= COLLISION_START_BAR:
            score.add_note(
                "kick",
                start=_bar(bar, 3.0),
                duration=0.36,
                partial=0.5,
                amp_db=-10.5,
                velocity=0.64,
            )
        if bar % 2 == 0:
            score.add_note(
                "wood",
                start=_bar(bar, 2.5),
                duration=0.18,
                partial=2.0,
                amp_db=-13.0,
                velocity=0.64,
            )
        _write_hat_bar(score, bar)

    for bar in range(65, 73, 2):
        score.add_note(
            "wood",
            start=_bar(bar, 4.5),
            duration=0.16,
            partial=7 / 4,
            amp_db=-18.0,
            velocity=0.48,
        )


def _add_ground_roots(
    score: Score,
    start_bar: int,
    roots: tuple[float, ...],
    *,
    velocity: float,
) -> None:
    for index, root in enumerate(roots):
        bar = start_bar + index * 4
        score.add_note(
            "organ_bass",
            start=_bar(bar),
            duration=4.25 * BAR,
            partial=root,
            amp_db=-8.0,
            velocity=velocity,
            pitch_motion=PitchMotionSpec.vibrato(depth_ratio=0.0012, rate_hz=4.2),
        )


def _write_chord_cycle(
    score: Score,
    voice: str,
    progression: tuple[tuple[float, ...], ...],
    start_bar: int,
) -> None:
    for index, chord in enumerate(progression):
        _add_chord(
            score,
            voice,
            chord,
            bar=start_bar + index * 4,
            bars=4,
            amp_db=-16.0 if voice != "organ_shadow" else -18.0,
            stagger=0.09,
        )


def _add_chord(
    score: Score,
    voice: str,
    partials: tuple[float, ...],
    *,
    bar: int,
    bars: float,
    amp_db: float,
    stagger: float = 0.12,
) -> None:
    for index, partial in enumerate(partials):
        score.add_note(
            voice,
            start=_bar(bar) + index * stagger,
            duration=bars * BAR - index * stagger + BEAT,
            partial=partial,
            amp_db=amp_db - index * 0.9,
            velocity=0.58 + index * 0.035,
        )


def _add_subject(
    score: Score,
    voice: str,
    partials: tuple[float, ...],
    *,
    start_bar: int,
    amp_db: float,
    transpose: float = 1.0,
    velocity: float = 0.74,
) -> None:
    durations = (0.9, 0.85, 0.85, 1.35, 0.8, 0.8, 1.1, 1.8)
    amp_contour = (0.0, -1.4, -0.8, 0.6, 1.2, -0.9, -1.4, -0.2)
    for index, partial in enumerate(partials):
        note_duration = durations[index] * BEAT
        pitch_motion = None
        if note_duration >= BEAT * 1.2:
            pitch_motion = PitchMotionSpec.vibrato(
                depth_ratio=0.0024,
                rate_hz=4.8 + index * 0.08,
            )
        score.add_note(
            voice,
            start=_bar(start_bar) + index * BEAT,
            duration=note_duration,
            partial=partial * transpose,
            amp_db=amp_db + amp_contour[index],
            velocity=velocity + (0.04 if index in {0, 3, 4, 7} else -0.02),
            pitch_motion=pitch_motion,
        )


def _add_bell_bar(
    score: Score,
    bar: int,
    chord: tuple[float, ...],
    *,
    amp_db: float = -15.0,
) -> None:
    pattern = (0, 2, 4, 1, 3, 2)
    offsets = (1.0, 1.75, 2.5, 3.25, 3.75, 4.5)
    for step, beat in zip(pattern, offsets, strict=True):
        partial = chord[step % len(chord)] * 4.0
        score.add_note(
            "bell_arp",
            start=_bar(bar, beat),
            duration=0.42,
            partial=partial,
            amp_db=amp_db - step * 0.45,
            velocity=0.58 + step * 0.035,
        )


def _add_grain_chord(
    score: Score,
    partials: tuple[float, ...],
    *,
    bar: int,
    bars: float,
    amp_db: float = -23.0,
) -> None:
    for index, partial in enumerate(partials[-4:]):
        score.add_note(
            "grain_shimmer",
            start=_bar(bar) + index * 0.55,
            duration=bars * BAR - index * 0.4,
            partial=partial * 2.0,
            amp_db=amp_db - index,
            velocity=0.50,
        )


def _write_hat_bar(score: Score, bar: int) -> None:
    if bar < 25:
        beats = (2.0, 4.0)
    elif bar < UNDERTOW_START_BAR:
        beats = (1.5, 2.0, 3.5, 4.0)
    elif bar < COLLISION_START_BAR:
        beats = (2.0, 3.5, 4.0)
    else:
        beats = (1.5, 2.0, 2.75, 3.5, 4.0)
    for index, beat in enumerate(beats):
        score.add_note(
            "hat",
            start=_bar(bar, beat),
            duration=0.08,
            partial=8.0,
            amp_db=-20.0 + (1.2 if index % 2 == 0 else 0.0),
            velocity=0.42,
        )


PIECES: dict[str, PieceDefinition] = {
    "undertow_canon": PieceDefinition(
        name="undertow_canon",
        output_name="undertow_canon",
        build_score=build_score,
        sections=(
            PieceSection(
                label="ground",
                start_seconds=_section_start(GROUND_START_BAR),
                end_seconds=_section_start(CANON_START_BAR),
            ),
            PieceSection(
                label="canon_bloom",
                start_seconds=_section_start(CANON_START_BAR),
                end_seconds=_section_start(UNDERTOW_START_BAR),
            ),
            PieceSection(
                label="undertow",
                start_seconds=_section_start(UNDERTOW_START_BAR),
                end_seconds=_section_start(COLLISION_START_BAR),
            ),
            PieceSection(
                label="collision",
                start_seconds=_section_start(COLLISION_START_BAR),
                end_seconds=_section_start(RETURN_START_BAR),
            ),
            PieceSection(
                label="return",
                start_seconds=_section_start(RETURN_START_BAR),
                end_seconds=TOTAL_DUR,
            ),
        ),
    ),
}
