"""VA Showcase — 90s/00s virtual-analog character study.

Four ~20s sections demonstrating the ``va`` engine's osc_modes, filter
routings, drive, and comb. Aesthetic brief follows AGENTS.md electronic-key
guidance (F-G# range) plus the writer's JP/Virus/Q inspirations.

Section map (90 s total):
    A ( 0-22)  JP-8000 hoover arp with filter sweep, bbd_chorus send
    B (22-45)  Virus spectral pad + driven sub-bass, spectral_position motion
    C (45-67)  Waldorf Q comb bell arpeggio over phase-dispersed pad
    D (67-90)  Layered supersaw pad + sync lead with filter automation
"""

from __future__ import annotations

from code_musics.automation import (
    AutomationSegment,
    AutomationSpec,
    AutomationTarget,
)
from code_musics.composition import line
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS, SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score, SendBusSpec, VoiceSend

BPM = 124.0
BEAT = 60.0 / BPM
BAR = 4.0 * BEAT
S16 = BEAT / 4.0
F0 = 49.0  # G1 — in the F-G# sweet spot for electronic impact

SECTION_A_END = 22.0
SECTION_B_END = 45.0
SECTION_C_END = 67.0
SECTION_D_END = 90.0


def build_score() -> Score:
    bbd_chorus_bus = SendBusSpec(
        name="bbd_chorus",
        effects=[EffectSpec("bbd_chorus", {"preset": "juno_i_plus_ii", "mix": 0.45})],
        return_db=0.0,
    )
    plate_bus = SendBusSpec(
        name="plate_tail",
        effects=[SOFT_REVERB_EFFECT],
        return_db=-2.0,
    )

    score = Score(
        f0_hz=F0,
        master_effects=list(DEFAULT_MASTER_EFFECTS),
        send_buses=[bbd_chorus_bus, plate_bus],
    )

    # --------------------------------------------------------------
    # Section A: JP-8000 hoover arpeggio
    # --------------------------------------------------------------
    score.add_voice(
        "hoover",
        synth_defaults={
            "engine": "va",
            "preset": "jp8000_hoover",
            "attack": 0.02,
            "decay": 0.18,
            "sustain_level": 0.65,
            "release": 0.22,
        },
        pan=0.0,
        sends=[VoiceSend(target="bbd_chorus", send_db=-6.0)],
        mix_db=-10.0,
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="cutoff_hz"),
                segments=(
                    AutomationSegment(
                        start=0.0,
                        end=SECTION_A_END,
                        shape="exp",
                        start_value=1200.0,
                        end_value=4200.0,
                    ),
                ),
            ),
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="resonance_q"),
                segments=(
                    AutomationSegment(
                        start=0.0,
                        end=SECTION_A_END,
                        shape="linear",
                        start_value=2.5,
                        end_value=3.4,
                    ),
                ),
            ),
        ],
    )
    # G minor pentatonic-ish arpeggio (partials over F0=49 Hz G1): G, Bb, D, F, G (oct)
    # Partial numbers picked for approx minor pentatonic above G.
    hoover_tones = [4.0, 4.8, 6.0, 7.2, 8.0, 7.2, 6.0, 4.8]  # G2 Bb2 D3 F3 G3 F3 D3 Bb2
    n_arp_reps = int(SECTION_A_END // (BEAT * 2.0))
    arp_rhythm = (S16, S16, S16, S16, S16, S16, S16, S16)
    for rep in range(n_arp_reps):
        start = rep * BEAT * 2.0
        if start >= SECTION_A_END:
            break
        score.add_phrase(
            "hoover",
            line(tones=hoover_tones, rhythm=arp_rhythm, amp_db=-6.0),
            start=start,
        )

    # --------------------------------------------------------------
    # Section B: Virus spectral pad + driven sub-bass
    # --------------------------------------------------------------
    score.add_voice(
        "virus_pad",
        synth_defaults={
            "engine": "va",
            "preset": "virus_pad",
            "attack": 1.4,
            "decay": 0.6,
            "sustain_level": 0.8,
            "release": 2.5,
        },
        pan=-0.15,
        sends=[VoiceSend(target="plate_tail", send_db=-4.0)],
        mix_db=-9.0,
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="spectral_position"),
                segments=(
                    AutomationSegment(
                        start=SECTION_A_END,
                        end=SECTION_B_END,
                        shape="linear",
                        start_value=0.25,
                        end_value=0.7,
                    ),
                ),
            ),
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="cutoff_hz"),
                segments=(
                    AutomationSegment(
                        start=SECTION_A_END,
                        end=SECTION_B_END,
                        shape="exp",
                        start_value=800.0,
                        end_value=3200.0,
                    ),
                ),
            ),
        ],
    )
    score.add_voice(
        "virus_bass",
        synth_defaults={
            "engine": "va",
            "preset": "virus_bass",
            "attack": 0.005,
            "decay": 0.16,
            "sustain_level": 0.7,
            "release": 0.14,
        },
        pan=0.1,
        mix_db=-8.0,
    )
    # Pad chord stack over section B: Gm (G-Bb-D) then Eb (Eb-G-Bb) then F (F-A-C)
    pad_partials_per_chord = [
        (2.0, 2.4, 3.0),  # G2 Bb2 D3
        (1.66667, 2.0, 2.4),  # Eb2 G2 Bb2
        (1.86667, 2.33, 2.8),  # F2 A2 C3
    ]
    chord_duration = (SECTION_B_END - SECTION_A_END) / 3.0
    for i, chord in enumerate(pad_partials_per_chord):
        chord_start = SECTION_A_END + i * chord_duration
        for partial in chord:
            score.add_note(
                "virus_pad",
                start=chord_start,
                duration=chord_duration + 0.5,
                partial=partial,
                amp_db=-14.0,
            )

    # Sub-bass riff: root movement G → Eb → F tracking chord changes.
    bass_tones = [1.0, 1.0, 1.0, 1.0, 0.8333, 0.8333, 0.9333, 0.9333]
    bass_rhythm = tuple([BEAT] * 8)
    n_bass_reps = int((SECTION_B_END - SECTION_A_END) // (8 * BEAT)) + 1
    for rep in range(n_bass_reps):
        start = SECTION_A_END + rep * 8 * BEAT
        if start >= SECTION_B_END:
            break
        score.add_phrase(
            "virus_bass",
            line(tones=bass_tones, rhythm=bass_rhythm, amp_db=-4.0),
            start=start,
        )

    # --------------------------------------------------------------
    # Section C: Waldorf Q comb bell arpeggio + phase-dispersed pad
    # --------------------------------------------------------------
    score.add_voice(
        "q_pad",
        synth_defaults={
            "engine": "va",
            "preset": "q_comb_pad",
            "attack": 2.5,
            "decay": 0.8,
            "sustain_level": 0.75,
            "release": 3.0,
        },
        pan=-0.1,
        sends=[VoiceSend(target="plate_tail", send_db=-4.0)],
        mix_db=-11.0,
    )
    score.add_voice(
        "q_bell",
        synth_defaults={
            "engine": "va",
            "preset": "q_comb_bell",
            "attack": 0.003,
            "decay": 0.3,
            "sustain_level": 0.25,
            "release": 0.8,
        },
        pan=0.2,
        sends=[VoiceSend(target="plate_tail", send_db=-2.0)],
        mix_db=-10.0,
    )
    # Q pad: sustained chord under the bell motif. C minor-ish (C-Eb-G-Bb)
    q_pad_partials = [1.33333, 1.6, 2.0, 2.4]  # C2 Eb2 G2 Bb2
    for partial in q_pad_partials:
        score.add_note(
            "q_pad",
            start=SECTION_B_END,
            duration=SECTION_C_END - SECTION_B_END,
            partial=partial,
            amp_db=-14.0,
        )

    # Bell motif: septimal-colored pentatonic wandering
    bell_motif = [4.0, 6.0, 5.0, 7.0, 6.0, 5.0, 4.0, 3.5, 7.0, 6.0, 5.0, 4.0]
    bell_rhythm = (
        S16 * 3,
        S16,
        S16 * 2,
        S16 * 2,
        S16,
        S16,
        S16 * 3,
        S16 * 2,
        S16 * 2,
        S16,
        S16,
        S16 * 4,
    )
    n_bell_reps = int((SECTION_C_END - SECTION_B_END) // (sum(bell_rhythm))) + 1
    for rep in range(n_bell_reps):
        start = SECTION_B_END + rep * sum(bell_rhythm)
        if start >= SECTION_C_END:
            break
        score.add_phrase(
            "q_bell",
            line(tones=bell_motif, rhythm=bell_rhythm, amp_db=-5.0),
            start=start,
        )

    # --------------------------------------------------------------
    # Section D: Supersaw pad + sync lead
    # --------------------------------------------------------------
    score.add_voice(
        "supersaw_pad",
        synth_defaults={
            "engine": "va",
            "preset": "supersaw_pad",
            "attack": 1.8,
            "decay": 0.6,
            "sustain_level": 0.85,
            "release": 3.5,
        },
        pan=-0.2,
        sends=[VoiceSend(target="bbd_chorus", send_db=-8.0)],
        mix_db=-10.0,
    )
    score.add_voice(
        "sync_lead",
        synth_defaults={
            "engine": "va",
            "preset": "virus_lead",
            "attack": 0.02,
            "decay": 0.3,
            "sustain_level": 0.6,
            "release": 0.5,
        },
        pan=0.18,
        sends=[VoiceSend(target="plate_tail", send_db=-6.0)],
        mix_db=-8.0,
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="cutoff_hz"),
                segments=(
                    AutomationSegment(
                        start=SECTION_C_END,
                        end=SECTION_D_END,
                        shape="exp",
                        start_value=1800.0,
                        end_value=5500.0,
                    ),
                ),
            ),
        ],
    )

    # Pad chord: G major 7-ish (G-B-D-F#) for a brighter, more euphoric mood.
    supersaw_chord = [2.0, 2.5, 3.0, 3.75]
    for partial in supersaw_chord:
        score.add_note(
            "supersaw_pad",
            start=SECTION_C_END,
            duration=SECTION_D_END - SECTION_C_END,
            partial=partial,
            amp_db=-13.0,
        )

    lead_tones = [6.0, 7.5, 8.0, 9.0, 8.0, 7.5, 6.0, 5.0, 6.0, 7.5, 9.0, 10.0]
    lead_rhythm = (
        BEAT,
        BEAT * 0.5,
        BEAT * 0.5,
        BEAT,
        BEAT * 0.5,
        BEAT * 0.5,
        BEAT,
        BEAT,
        BEAT * 0.5,
        BEAT * 0.5,
        BEAT,
        BEAT,
    )
    n_lead_reps = int((SECTION_D_END - SECTION_C_END) // sum(lead_rhythm)) + 1
    for rep in range(n_lead_reps):
        start = SECTION_C_END + rep * sum(lead_rhythm)
        if start >= SECTION_D_END:
            break
        score.add_phrase(
            "sync_lead",
            line(tones=lead_tones, rhythm=lead_rhythm, amp_db=-6.0),
            start=start,
        )

    return score


PIECES: dict[str, PieceDefinition] = {
    "va_showcase": PieceDefinition(
        name="va_showcase",
        output_name="va_showcase",
        build_score=build_score,
        sections=(
            PieceSection("jp8000_hoover", 0.0, SECTION_A_END),
            PieceSection("virus_pad_bass", SECTION_A_END, SECTION_B_END),
            PieceSection("q_comb_bells", SECTION_B_END, SECTION_C_END),
            PieceSection("supersaw_sync_lead", SECTION_C_END, SECTION_D_END),
        ),
    ),
}
