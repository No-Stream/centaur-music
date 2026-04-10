"""`ji_chorale` piece builder."""

from __future__ import annotations

import logging
from dataclasses import replace

from code_musics import synth
from code_musics.composition import line, with_synth_ramp
from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    VelocityHumanizeSpec,
)
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score, VelocityParamMap

logger = logging.getLogger(__name__)


def build_ji_chorale_score() -> Score:
    """Extended chorale in 5-limit JI — ~2:30 in six sections.

    Structure:
      Prologue   (0–12 s):   Fs2+A3 sparse drone; lead pickup at t=8.
      A section  (12–54 s):  8-bar vi–iv alternation; counter enters bar 3.
      B section  (54–75 s):  I–V–I in A major; bright but grounded.
      Development(75–99 s):  F#m7 → Bm → V; exploratory, unsettled.
      Reprise    (99–120 s): vi–I–vi–I; bittersweet into tastefully major.
      Ending     (120–149 s):wide vi → Dm7 → Amaj7; unresolved, wide-voiced.
    """
    f0 = 110.0  # A2 = 110 Hz

    # Named pitches — all 5-limit JI from f0
    A2 = f0  # 110.00
    B2 = f0 * 9 / 8  # 123.75
    D3 = f0 * 4 / 3  # 146.67  iv root
    E3 = f0 * 3 / 2  # 165.00
    F3 = D3 * 6 / 5  # 176.00  D-minor 3rd (outside A major)
    Fs2 = f0 * 5 / 6  # 91.67   sub-bass F#2
    Fs3 = f0 * 5 / 3  # 183.33  vi root
    Gs3 = E3 * 5 / 4  # 206.25  V-chord 3rd (G#3)
    A3 = f0 * 2  # 220.00
    B3 = f0 * 9 / 4  # 247.50
    Cs4 = f0 * 5 / 2  # 275.00
    C4 = D3 * 9 / 5  # 264.00  pure min-7 above D
    D4 = D3 * 2  # 293.33
    E4 = f0 * 3  # 330.00
    F4 = F3 * 2  # 352.00
    Fs4 = Fs3 * 2  # 366.67
    Gs4 = f0 * 15 / 4  # 412.50  A-maj7
    A4 = f0 * 4  # 440.00
    B4 = f0 * 9 / 2  # 495.00
    Cs5 = f0 * 5  # 550.00
    D5 = D4 * 2  # 586.67
    A5 = A4 * 2  # 880.00  — tonic, next octave

    master_effects: list[EffectSpec] = [
        EffectSpec("saturation", {"preset": "neve_gentle", "mix": 0.18, "drive": 0.85}),
    ]
    if synth.has_external_plugin("lsp_compressor_stereo"):
        master_effects.append(
            EffectSpec(
                "plugin",
                {
                    "plugin_name": "lsp_compressor_stereo",
                    "params": {
                        "ratio": 2.2,
                        "attack_threshold_db": -18.0,
                        "attack_time_ms": 28.0,
                        "release_time_ms": 220.0,
                        "knee_db": -10.0,
                        "makeup_gain_db": 1.2,
                        "sidechain_mode": "RMS",
                        "dry_wet_balance": 100.0,
                    },
                },
            )
        )
    else:
        logger.warning(
            "Skipping optional ji_chorale glue compressor: LSP VST3 bundle/runtime not available."
        )
    master_effects.extend(
        [
            EffectSpec(
                "chow_tape",
                {"drive": 0.15, "saturation": 0.18, "bias": 0.5, "mix": 50.0},
            ),
            EffectSpec(
                "bricasti",
                {
                    "ir_name": "1 Halls 07 Large & Dark",
                    "wet": 0.15,
                    "highpass_hz": 150.0,
                },
            ),
        ]
    )

    score = Score(
        f0=f0,
        timing_humanize=TimingHumanizeSpec(preset="chamber", chord_spread_ms=7.0),
        master_effects=master_effects,
    )

    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "filtered_stack",
            "waveform": "square",
            "n_harmonics": 12,
            "cutoff_hz": 900.0,
            "keytrack": 0.1,
            "resonance": 0.0,
            "filter_env_amount": 0.55,
            "filter_env_decay": 0.70,
            "attack": 0.22,
            "decay": 0.18,
            "sustain_level": 0.60,
            "release": 0.90,
        },
        mix_db=-2.0,
        pan=-0.08,
        velocity_group="harmony",
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble"),
        velocity_db_per_unit=8.0,
        velocity_to_params={
            "cutoff_hz": VelocityParamMap(min_value=680.0, max_value=1100.0)
        },
    )
    chord_defaults: dict = {
        "harmonic_rolloff": 0.38,
        "n_harmonics": 8,
        "brightness_tilt": 0.06,
        "unison_voices": 2,
        "detune_cents": 3,
        "attack": 0.22,
        "decay": 0.18,
        "sustain_level": 0.56,
        "release": 0.70,
    }
    score.add_voice(
        "tenor",
        synth_defaults={
            "engine": "filtered_stack",
            "waveform": "saw",
            "n_harmonics": 12,
            "cutoff_hz": 1_000.0,
            "keytrack": 0.1,
            "resonance": 0.08,
            "filter_env_amount": 0.18,
            "filter_env_decay": 0.5,
            "attack": 0.05,
            "decay": 0.25,
            "sustain_level": 0.56,
            "release": 0.70,
        },
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 100.0, "slope_db_per_oct": 24}
                    ]
                },
            ),
        ],
        mix_db=-11.0,
        pan=-0.16,
        velocity_group="harmony",
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble"),
        velocity_db_per_unit=8.0,
        velocity_to_params={
            "cutoff_hz": VelocityParamMap(min_value=750.0, max_value=1_300.0)
        },
    )
    score.add_voice(
        "alto",
        synth_defaults=dict(chord_defaults),
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 120.0, "slope_db_per_oct": 24}
                    ]
                },
            ),
        ],
        mix_db=-9.5,
        pan=0.14,
        velocity_group="harmony",
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble"),
        velocity_db_per_unit=8.0,
        velocity_to_params={
            "brightness_tilt": VelocityParamMap(min_value=-0.02, max_value=0.06)
        },
    )
    score.add_voice(
        "counter",
        synth_defaults={
            "engine": "filtered_stack",
            "preset": "reed_lead",
            "cutoff_hz": 1_400.0,
            "keytrack": 0.1,
            "resonance": 0.10,
            "filter_env_amount": 0.95,
            "attack": 0.03,
            "decay": 0.12,
            "sustain_level": 0.52,
            "release": 0.30,
        },
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 180.0, "slope_db_per_oct": 24}
                    ]
                },
            ),
        ],
        mix_db=-10.0,
        pan=0.08,
        velocity_group="melody",
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        velocity_db_per_unit=10.0,
        velocity_to_params={
            "cutoff_hz": VelocityParamMap(min_value=1_800.0, max_value=3_000.0),
            "filter_env_amount": VelocityParamMap(min_value=0.72, max_value=1.18),
        },
    )
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "triangle",
            "cutoff_hz": 2_200.0,
            "keytrack": 0.1,
            "resonance": 0.05,
            "filter_env_amount": 0.13,
            "filter_env_decay": 1.0,
            "filter_drive": 0.05,
            "attack": 0.085,
            "decay": 1.25,
            "sustain_level": 0.48,
            "release": 0.32,
        },
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 200.0, "slope_db_per_oct": 24}
                    ]
                },
            ),
            EffectSpec(
                "chorus",
                {
                    "preset": "juno_subtle",
                    "mix": 0.16,
                    "depth_ms": 1.8,
                    "feedback": 0.02,
                    "wet_lowpass_hz": 4_800.0,
                },
            ),
            EffectSpec(
                "bricasti",
                {
                    "ir_name": "2 Plates 06 Vocal Plate",
                    "wet": 0.16,
                    "highpass_hz": 320.0,
                    "lowpass_hz": 8_500.0,
                    "tilt_db": -1.5,
                },
            ),
        ],
        mix_db=-4.0,
        pan=0.20,
        velocity_group="melody",
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        velocity_to_params={
            "filter_env_amount": VelocityParamMap(min_value=0.30, max_value=0.75),
            "resonance": VelocityParamMap(min_value=0.04, max_value=0.12),
        },
    )

    score.add_note("bass", start=0.0, duration=3.2, freq=Fs2, amp=0.17, velocity=0.72)
    score.add_note("bass", start=3.8, duration=2.6, freq=Fs2, amp=0.20, velocity=0.78)
    score.add_note("bass", start=6.8, duration=5.2, freq=Fs2, amp=0.22, velocity=0.82)
    score.add_note("tenor", start=2.0, duration=0.9, freq=A3, amp=0.12, velocity=0.72)
    score.add_note("tenor", start=3.5, duration=1.4, freq=Cs4, amp=0.13, velocity=0.78)
    score.add_note("tenor", start=5.5, duration=0.7, freq=B3, amp=0.12, velocity=0.73)
    score.add_note("tenor", start=6.5, duration=0.6, freq=A3, amp=0.11, velocity=0.70)
    score.add_note("tenor", start=7.4, duration=0.5, freq=Cs4, amp=0.11, velocity=0.72)
    score.add_note("tenor", start=8.2, duration=3.8, freq=A3, amp=0.14, velocity=0.80)

    a_note_dur = 5.5
    a_chords: list[tuple[float, float, float, float]] = [
        (12.00, Fs3, A3, Cs4),
        (17.25, D3, F3, A3),
        (22.50, Fs3, A3, Cs4),
        (27.75, D3, F3, A3),
        (33.00, Fs3, A3, Cs4),
        (38.25, D3, F3, A3),
        (43.50, Fs3, A3, Cs4),
        (48.75, D3, F3, A3),
    ]
    a_velocities = [0.90, 0.95, 1.02, 1.05, 1.12, 0.98, 0.90, 0.85]
    for (start, bass_freq, tenor_freq, alto_freq), velocity in zip(
        a_chords,
        a_velocities,
        strict=True,
    ):
        score.add_note(
            "bass",
            start=start,
            duration=a_note_dur,
            freq=bass_freq,
            amp=0.25,
            velocity=velocity,
        )
        score.add_note(
            "tenor",
            start=start,
            duration=a_note_dur,
            freq=tenor_freq,
            amp=0.21,
            velocity=velocity,
        )
        score.add_note(
            "alto",
            start=start,
            duration=a_note_dur,
            freq=alto_freq,
            amp=0.18,
            velocity=velocity,
        )

    b_note_dur = 5.8
    b_chords: list[tuple[float, float, float, float]] = [
        (54.0, A2, E3, Cs4),
        (59.5, E3, Gs3, B3),
        (65.0, A2, E3, Cs4),
    ]
    b_velocities = [1.05, 1.10, 0.95]
    for (start, bass_freq, tenor_freq, alto_freq), velocity in zip(
        b_chords,
        b_velocities,
        strict=True,
    ):
        score.add_note(
            "bass",
            start=start,
            duration=b_note_dur,
            freq=bass_freq,
            amp=0.23,
            velocity=velocity,
        )
        score.add_note(
            "tenor",
            start=start,
            duration=b_note_dur,
            freq=tenor_freq,
            amp=0.20,
            velocity=velocity,
        )
        score.add_note(
            "alto",
            start=start,
            duration=b_note_dur,
            freq=alto_freq,
            amp=0.17,
            velocity=velocity,
        )

    # Last dev chord extended to 99s to fill gap before reprise (dominant preparation).
    dev_note_durs = [6.3, 6.3, 12.0]
    dev_chords: list[tuple[float, float, float, float]] = [
        (75.0, Fs2, A3, Cs4),
        (81.0, B2, D3, Fs3),
        (87.0, E3, Gs3, B3),
    ]
    dev_velocities = [1.02, 1.08, 1.14]
    for (start, bass_freq, tenor_freq, alto_freq), velocity, dev_note_dur in zip(
        dev_chords,
        dev_velocities,
        dev_note_durs,
        strict=True,
    ):
        score.add_note(
            "bass",
            start=start,
            duration=dev_note_dur,
            freq=bass_freq,
            amp=0.24,
            velocity=velocity,
        )
        score.add_note(
            "tenor",
            start=start,
            duration=dev_note_dur,
            freq=tenor_freq,
            amp=0.20,
            velocity=velocity,
        )
        score.add_note(
            "alto",
            start=start,
            duration=dev_note_dur,
            freq=alto_freq,
            amp=0.18,
            velocity=velocity,
        )

    rep_note_dur = 5.3
    rep_chords: list[tuple[float, float, float, float]] = [
        (99.0, Fs3, A3, Cs4),
        (104.0, A2, E3, Cs4),
        (109.0, Fs3, A3, Cs4),
        (114.0, A2, E3, Cs4),
    ]
    rep_velocities = [1.00, 0.95, 1.00, 0.88]
    for (start, bass_freq, tenor_freq, alto_freq), velocity in zip(
        rep_chords,
        rep_velocities,
        strict=True,
    ):
        score.add_note(
            "bass",
            start=start,
            duration=rep_note_dur,
            freq=bass_freq,
            amp=0.24,
            velocity=velocity,
        )
        score.add_note(
            "tenor",
            start=start,
            duration=rep_note_dur,
            freq=tenor_freq,
            amp=0.20,
            velocity=velocity,
        )
        score.add_note(
            "alto",
            start=start,
            duration=rep_note_dur,
            freq=alto_freq,
            amp=0.18,
            velocity=velocity,
        )

    # Reprise: Gs4 in alto holds for the full I chord — Amaj7 texture (counter added below).
    score.add_note("alto", start=114.0, duration=5.3, freq=Gs4, amp=0.09, velocity=0.72)

    score.add_note("bass", start=120.0, duration=8.0, freq=Fs2, amp=0.27, velocity=1.05)
    score.add_note("tenor", start=120.0, duration=8.0, freq=A3, amp=0.20, velocity=1.05)
    score.add_note("alto", start=120.0, duration=8.0, freq=Cs5, amp=0.21, velocity=1.05)

    score.add_note("bass", start=128.0, duration=8.0, freq=D3, amp=0.26, velocity=1.00)
    score.add_note("tenor", start=128.0, duration=8.0, freq=F3, amp=0.21, velocity=1.00)
    score.add_note("alto", start=128.0, duration=8.0, freq=A3, amp=0.18, velocity=1.00)
    score.add_note("alto", start=128.0, duration=8.0, freq=C4, amp=0.16, velocity=1.00)

    score.add_note("bass", start=136.0, duration=10.0, freq=A2, amp=0.23, velocity=0.88)
    score.add_note(
        "tenor", start=136.0, duration=10.0, freq=E3, amp=0.19, velocity=0.88
    )
    score.add_note(
        "alto", start=136.0, duration=10.0, freq=Cs4, amp=0.17, velocity=0.88
    )
    score.add_note(
        "alto", start=136.0, duration=10.0, freq=Gs4, amp=0.17, velocity=0.88
    )

    def _add_counter(t_start: float, notes: list[tuple[float, float, float]]) -> None:
        t = t_start
        for freq, duration, velocity in notes:
            score.add_note(
                "counter",
                start=t,
                duration=duration * 1.02,
                freq=freq,
                amp=0.22,
                velocity=velocity,
            )
            t += duration

    _add_counter(22.5, [(E4, 2.25, 1.00), (D4, 1.75, 0.93), (Cs4, 1.25, 0.85)])
    _add_counter(27.75, [(F4, 1.75, 1.05), (E4, 1.75, 0.97), (D4, 1.75, 0.88)])
    _add_counter(33.0, [(E4, 1.25, 1.02), (Cs4, 1.75, 0.92), (E4, 2.25, 1.00)])
    # D4→F4→A4: chord tones of D minor; avoids E4 which clashes with F3 bass (maj-7).
    _add_counter(38.25, [(D4, 1.75, 0.92), (F4, 2.25, 1.00), (A4, 1.25, 1.05)])
    _add_counter(43.5, [(Cs4, 1.75, 1.02), (E4, 2.25, 1.05), (D4, 1.25, 0.90)])

    # B section: embryonic E→G# hint — same Gs4 that blooms fully at the Ending
    _add_counter(59.5, [(E4, 2.0, 0.90), (Gs4, 3.5, 0.85)])

    _add_counter(75.0, [(E4, 2.25, 1.05), (Cs4, 1.75, 0.92), (E4, 2.0, 1.00)])
    _add_counter(81.0, [(Fs4, 2.25, 1.10), (E4, 2.0, 1.00), (D4, 2.0, 0.90)])
    _add_counter(87.0, [(Gs4, 2.25, 1.18), (Fs4, 1.5, 1.02), (E4, 2.25, 0.88)])

    _add_counter(99.0, [(E4, 2.5, 1.08), (Cs4, 2.5, 0.90)])
    _add_counter(109.0, [(Cs4, 2.25, 0.95), (E4, 2.75, 1.05)])
    # Amaj7 moment at 114s: counter traces maj7→9 over held Amaj7 chord.
    _add_counter(114.0, [(Gs4, 2.75, 1.05), (B4, 2.25, 0.95)])

    _add_counter(120.0, [(E4, 4.0, 1.00), (Cs4, 4.0, 0.90)])
    _add_counter(128.0, [(F4, 4.0, 1.08), (E4, 4.0, 0.95)])
    _add_counter(136.0, [(E4, 4.0, 1.00), (Gs4, 4.0, 1.10), (E4, 3.0, 0.85)])

    def _add_lead_phrase(
        *,
        start: float,
        notes: list[tuple[float, float]],
        synth_start: dict[str, float],
        synth_end: dict[str, float],
        amp_db: float,
        velocities: list[float],
    ) -> None:
        phrase = line(
            tones=[freq for freq, _ in notes],
            rhythm=[duration for _, duration in notes],
            pitch_kind="freq",
            amp_db=amp_db,
        )
        phrase = with_synth_ramp(phrase, start=synth_start, end=synth_end)
        phrase = replace(
            phrase,
            events=tuple(
                replace(event, velocity=velocity)
                for event, velocity in zip(phrase.events, velocities, strict=True)
            ),
        )
        score.add_phrase("lead", phrase, start=start)

    lead_prologue_and_a: list[tuple[float, float]] = [
        (A4, 1.0),
        (Cs5, 1.0),
        (B4, 1.0),
        (A4, 1.0),
        (Cs5, 2.0),
        (B4, 1.0),
        (A4, 2.0),
        (Gs4, 1.0),
        (A4, 1.5),
        (F4, 2.0),
        (A4, 1.5),
        (Gs4, 0.5),
        (A4, 0.5),
        (B4, 1.0),
        (Cs5, 2.0),
        (B4, 1.0),
        (A4, 1.5),
        (B4, 0.5),
        (A4, 1.0),
        (F4, 1.5),
        (E4, 1.0),
        (D4, 1.5),
        (E4, 1.0),
        (A4, 0.5),
        (Cs5, 0.75),
        (B4, 0.5),
        (Cs5, 0.75),
        (B4, 0.5),
        (A4, 0.5),
        (Gs4, 0.5),
        (A4, 0.5),
        (Cs5, 0.75),
        (D5, 0.375),
        (A5, 0.375),
        (A4, 1.5),
        (F4, 1.0),
        (E4, 1.5),
        (D4, 1.5),
        (E4, 0.5),
        (
            Cs5,
            0.75,
        ),  # ends at D-minor onset (48.75s); Cs5 over Amaj avoids tritone clash
        (A4, 1.75),  # chord tone of D minor
        (B4, 1.5),
        (A4, 1.5),
        (Gs4, 0.5),
    ]
    prologue_a_velocities: list[float] = [
        0.85,
        1.12,
        1.00,
        0.88,
        1.22,
        1.02,
        0.95,
        0.82,
        1.05,
        1.15,
        1.02,
        0.88,
        0.85,
        1.02,
        1.25,
        1.05,
        0.95,
        0.85,
        1.05,
        1.12,
        1.00,
        0.90,
        0.85,
        1.00,
        1.10,
        1.02,
        1.15,
        1.05,
        1.00,
        0.95,
        1.05,
        1.20,
        1.32,
        1.42,
        1.08,
        0.95,
        0.90,
        0.85,
        0.80,
        1.02,
        0.90,  # new A4 note
        0.92,
        0.85,
        0.75,
    ]

    lead_b_and_development: list[tuple[float, float]] = [
        (E4, 1.5),
        (Cs5, 1.5),
        (A4, 2.0),
        (E4, 2.0),
        (B4, 2.0),
        (Gs4, 1.5),
        (Fs4, 1.5),
        (E4, 2.0),
        (Cs5, 2.5),
        (A4, 2.0),
        (E4, 2.5),
        (A4, 1.5),
        (E4, 1.5),
        (Cs5, 1.5),
        (Fs4, 1.5),
        (A4, 2.0),
        (D4, 1.0),
        (Fs4, 1.5),
        (A4, 2.0),
        (B4, 1.5),
        (A4, 2.0),
        (Gs4, 2.0),
        (B4, 2.0),
        (Gs4, 1.5),
        (E4, 2.5),
    ]
    b_dev_velocities: list[float] = [
        0.98,
        1.25,
        1.02,
        0.88,
        1.15,
        1.05,
        0.95,
        0.85,
        1.12,
        0.97,
        0.82,
        1.00,
        0.92,
        1.12,
        0.92,
        0.98,
        0.88,
        1.00,
        1.10,
        1.22,
        1.02,
        1.12,
        1.20,
        1.02,
        0.85,
    ]

    lead_reprise_and_ending: list[tuple[float, float]] = [
        (Cs5, 2.0),
        (B4, 1.5),
        (A4, 2.0),
        (Gs4, 0.5),
        (A4, 1.5),
        (Cs5, 2.0),
        (E4, 2.5),
        (B4, 1.5),
        (Cs5, 1.5),
        (B4, 1.5),
        (A4, 1.5),
        (Cs5, 2.5),
        (A4, 2.0),
        (E4, 1.5),
        (Cs5, 2.5),
        (B4, 1.0),
        (A4, 1.0),
        (Gs4, 0.5),
        (A4, 1.0),
        (B4, 2.0),
        (A4, 1.5),
        (F4, 1.0),
        (C4, 2.0),
        (D4, 1.0),
        (F4, 1.5),
        (A4, 1.0),
        (Gs4, 2.0),
        (A4, 1.5),
        (Cs5, 2.5),
        (B4, 1.5),
        (A4, 3.5),
    ]
    reprise_ending_velocities: list[float] = [
        1.15,
        1.00,
        0.90,
        0.75,
        0.95,
        1.10,
        0.85,
        1.05,
        1.18,
        1.05,
        0.90,
        1.12,
        0.98,
        0.85,
        1.22,
        1.02,
        0.92,
        0.80,
        0.85,
        0.98,
        1.05,
        1.12,
        1.22,
        1.05,
        1.00,
        0.90,
        1.00,
        0.95,
        1.08,
        0.90,
        0.78,
    ]

    _add_lead_phrase(
        start=8.0,
        notes=lead_prologue_and_a,
        synth_start={"cutoff_hz": 1_900.0, "release": 0.30},
        synth_end={"cutoff_hz": 2_200.0, "release": 0.28},
        amp_db=-18.0,
        velocities=prologue_a_velocities,
    )
    _add_lead_phrase(
        start=54.0,
        notes=lead_b_and_development,
        synth_start={"cutoff_hz": 2_200.0, "release": 0.26},
        synth_end={"cutoff_hz": 2_400.0, "release": 0.24},
        amp_db=-17.5,
        velocities=b_dev_velocities,
    )
    _add_lead_phrase(
        start=99.0,
        notes=lead_reprise_and_ending,
        synth_start={"cutoff_hz": 2_100.0, "release": 0.26},
        synth_end={"cutoff_hz": 1_750.0, "release": 0.34},
        amp_db=-18.0,
        velocities=reprise_ending_velocities,
    )

    score.add_note("bass", start=149.0, duration=0.5, freq=A2, amp=0.001)
    return score


PIECES: dict[str, PieceDefinition] = {
    "ji_chorale": PieceDefinition(
        name="ji_chorale",
        output_name="17_ji_chorale",
        build_score=build_ji_chorale_score,
        sections=(
            PieceSection(label="Prologue", start_seconds=0.0, end_seconds=12.0),
            PieceSection(label="A", start_seconds=12.0, end_seconds=54.0),
            PieceSection(label="B", start_seconds=54.0, end_seconds=75.0),
            PieceSection(label="Development", start_seconds=75.0, end_seconds=99.0),
            PieceSection(label="Reprise", start_seconds=99.0, end_seconds=120.0),
            PieceSection(label="Ending", start_seconds=120.0, end_seconds=149.5),
        ),
    )
}
