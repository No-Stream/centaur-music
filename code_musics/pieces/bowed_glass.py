"""Bowed Glass — sustained exciters through coupled modal bodies.

A slow, crystalline five-section piece built entirely around sustained
excitation (bow / blow / rub) driving modal resonator banks.  The
premise: take `drum_voice`'s modal tone layer, which traditionally rings
out after a transient exciter, and drive it continuously instead — so
each "note" is a living sustained body that breathes against its coupled
modes rather than a struck object that decays to silence.

Section map:
  1   (0-15s)   Breath onset — slow rub_glass over bowl modal bank
  2   (15-32s)  Bowing rises — bow_gentle/taut duet on bar_metal +
                bowl, coupled modes produce rolling beats
  3   (32-52s)  Blown chorale — blow_breath_pad chord voiced in JI over
                the ongoing bowing texture
  4   (52-70s)  Friction climb — rub_squeal accents punctuate the pad,
                dispersion increases across section
  5   (70-90s)  Dissolve — everything thins, rub_glass returns, long
                reverb-dependent tail

Key: E-ish.  f0 = 165 Hz (E3), 7-limit JI.  ~90 seconds total.
"""

from __future__ import annotations

from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score, SendBusSpec, VoiceSend

# ---------------------------------------------------------------------------
# Tuning + timing constants
# ---------------------------------------------------------------------------

F0 = 165.0  # E3
# 7-limit JI ratios relative to F0.
R_1 = 1.0
R_9_8 = 9.0 / 8.0
R_5_4 = 5.0 / 4.0
R_3_2 = 3.0 / 2.0
R_5_3 = 5.0 / 3.0
R_7_4 = 7.0 / 4.0
R_2 = 2.0

TOTAL_DUR = 90.0

SECTIONS: tuple[PieceSection, ...] = (
    PieceSection(label="breath_onset", start_seconds=0.0, end_seconds=15.0),
    PieceSection(label="bowing_rises", start_seconds=15.0, end_seconds=32.0),
    PieceSection(label="blown_chorale", start_seconds=32.0, end_seconds=52.0),
    PieceSection(label="friction_climb", start_seconds=52.0, end_seconds=70.0),
    PieceSection(label="dissolve", start_seconds=70.0, end_seconds=TOTAL_DUR),
)


# ---------------------------------------------------------------------------
# Score construction
# ---------------------------------------------------------------------------


def build_score() -> Score:
    """Build the Bowed Glass score."""
    reverb_bus = SendBusSpec(
        name="hall",
        effects=[
            EffectSpec(
                "reverb",
                {
                    "room_size": 0.88,
                    "damping": 0.45,
                    "wet_level": 1.0,
                },
            ),
        ],
        return_db=0.0,
    )

    score = Score(
        f0_hz=F0,
        master_effects=DEFAULT_MASTER_EFFECTS,
        send_buses=[reverb_bus],
    )

    _add_rub_voice(score)
    _add_bow_low_voice(score)
    _add_bow_high_voice(score)
    _add_blow_chord_voices(score)
    _add_squeal_voice(score)

    return score


def _add_rub_voice(score: Score) -> None:
    """Long rubbed-glass pedal, thread through the whole piece."""
    score.add_voice(
        "rub_pedal",
        synth_defaults={
            "engine": "drum_voice",
            "preset": "rub_glass",
            "modal_dispersion": 0.55,
            "modal_coupling": 0.25,
            "modal_coupling_topology": "ring",
            "pi_damping": 0.45,
        },
        pan=-0.15,
        mix_db=-8.0,
        sends=[VoiceSend(target="hall", send_db=-6.0)],
        normalize_peak_db=-6.0,
        velocity_humanize=None,
    )
    score.add_note(
        "rub_pedal",
        start=0.0,
        duration=16.0,
        freq=F0 * R_3_2,
        amp_db=-6.0,
    )
    score.add_note(
        "rub_pedal",
        start=60.0,
        duration=8.0,
        freq=F0 * R_5_3,
        amp_db=-9.0,
    )
    score.add_note(
        "rub_pedal",
        start=71.0,
        duration=18.0,
        freq=F0 * R_3_2,
        amp_db=-5.0,
    )


def _add_bow_low_voice(score: Score) -> None:
    """Low bowed voice — bar_metal modal, slow arrivals in section 2+."""
    score.add_voice(
        "bow_low",
        synth_defaults={
            "engine": "drum_voice",
            "preset": "bow_gentle",
            "modal_coupling": 0.3,
            "modal_coupling_topology": "chain",
            "modal_dispersion": 0.35,
            "modal_decay_s": 3.0,
        },
        pan=-0.35,
        mix_db=-5.0,
        sends=[VoiceSend(target="hall", send_db=-3.0)],
        normalize_peak_db=-6.0,
        velocity_humanize=None,
    )

    stroke_times = [15.5, 22.0, 34.0, 43.0]
    stroke_freqs = [F0 * R_1, F0 * R_5_4, F0 * R_9_8, F0 * R_5_4 * 2.0]
    stroke_durs = [6.0, 9.0, 8.0, 8.0]
    stroke_amps = [-8.0, -5.0, -6.0, -4.0]
    for t, f, d, a in zip(
        stroke_times, stroke_freqs, stroke_durs, stroke_amps, strict=True
    ):
        score.add_note("bow_low", start=t, duration=d, freq=f, amp_db=a)


def _add_bow_high_voice(score: Score) -> None:
    """High bowed voice — bowl modal with ring topology."""
    score.add_voice(
        "bow_high",
        synth_defaults={
            "engine": "drum_voice",
            "preset": "bow_taut",
            "modal_coupling": 0.32,
            "modal_coupling_topology": "ring",
            "modal_dispersion": 0.4,
            "modal_decay_s": 2.6,
        },
        pan=0.35,
        mix_db=-7.0,
        sends=[VoiceSend(target="hall", send_db=-4.0)],
        normalize_peak_db=-6.0,
        velocity_humanize=None,
    )
    stroke_times = [18.0, 25.5, 37.0, 46.0]
    stroke_freqs = [F0 * R_3_2, F0 * R_7_4, F0 * R_5_3 * 2.0, F0 * R_2]
    stroke_durs = [5.5, 7.0, 7.0, 6.0]
    stroke_amps = [-10.0, -7.0, -7.0, -6.0]
    for t, f, d, a in zip(
        stroke_times, stroke_freqs, stroke_durs, stroke_amps, strict=True
    ):
        score.add_note("bow_high", start=t, duration=d, freq=f, amp_db=a)


def _add_blow_chord_voices(score: Score) -> None:
    """Four voices forming a blown chorale — sustained reed-table timbre."""
    pan_positions = [-0.45, -0.15, 0.2, 0.45]
    chord_freqs = [F0 * R_1, F0 * R_3_2, F0 * R_7_4, F0 * R_2 * R_3_2]
    chord_amps = [-6.0, -8.0, -9.0, -10.0]

    chord_start = 32.0
    chord_dur = 20.0

    for idx, (pan, freq, amp_db) in enumerate(
        zip(pan_positions, chord_freqs, chord_amps, strict=True)
    ):
        name = f"blow_{idx}"
        score.add_voice(
            name,
            synth_defaults={
                "engine": "drum_voice",
                "preset": "blow_breath_pad",
                "modal_coupling": 0.2 + 0.03 * idx,
                "modal_coupling_topology": "chain",
                "modal_dispersion": 0.3,
                "modal_decay_s": 2.0 + 0.2 * idx,
                "exciter_blow_wobble_rate_hz": 2.5 + 0.7 * idx,
            },
            pan=pan,
            mix_db=-7.0,
            sends=[VoiceSend(target="hall", send_db=-4.0)],
            normalize_peak_db=-6.0,
            velocity_humanize=None,
        )
        entry_stagger = idx * 0.6
        score.add_note(
            name,
            start=chord_start + entry_stagger,
            duration=chord_dur - entry_stagger,
            freq=freq,
            amp_db=amp_db,
        )


def _add_squeal_voice(score: Score) -> None:
    """Accent voice — rub_squeal bursts in section 4.

    Per-note ``modal_dispersion`` ramps up across the section via the
    note-level ``synth`` override, producing progressively more warped
    squeals as the section climbs.  This is simpler than an automation
    spec (``modal_dispersion`` isn't a registered synth automation
    target) and gives the same audible effect since each squeal is
    only ~2 s long.
    """
    score.add_voice(
        "squeal",
        synth_defaults={
            "engine": "drum_voice",
            "preset": "rub_squeal",
            "modal_coupling": 0.28,
            "modal_coupling_topology": "all",
        },
        pan=0.1,
        mix_db=-10.0,
        sends=[VoiceSend(target="hall", send_db=-2.0)],
        normalize_peak_db=-6.0,
        velocity_humanize=None,
    )

    squeal_times = [53.5, 55.2, 58.0, 60.5, 63.2, 66.0, 68.5]
    squeal_freqs = [
        F0 * R_7_4 * 2.0,
        F0 * R_5_3 * 2.0,
        F0 * R_2 * R_5_4,
        F0 * R_7_4 * 2.0,
        F0 * R_2 * R_3_2,
        F0 * R_5_4 * 2.0,
        F0 * R_7_4 * 2.0,
    ]
    squeal_amps = [-14.0, -12.0, -11.0, -9.0, -8.0, -10.0, -12.0]
    section_start, section_end = 52.0, 70.0
    dispersion_start, dispersion_end = 0.3, 0.85
    for t, f, a in zip(squeal_times, squeal_freqs, squeal_amps, strict=True):
        progress = (t - section_start) / (section_end - section_start)
        dispersion = dispersion_start + progress * (dispersion_end - dispersion_start)
        score.add_note(
            "squeal",
            start=t,
            duration=1.8,
            freq=f,
            amp_db=a,
            synth={"modal_dispersion": float(dispersion)},
        )


PIECES: dict[str, PieceDefinition] = {
    "bowed_glass": PieceDefinition(
        name="bowed_glass",
        output_name="bowed_glass",
        build_score=build_score,
        sections=SECTIONS,
    ),
}
