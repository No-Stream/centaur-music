"""ji_chorale_v2 — 5-limit JI chorale, rebuilt on the Timeline / meter API.

88 BPM · 4/4 · six sections · ~2:27.

Changes from ji_chorale
-----------------------
- All timing is bar/beat-relative via a shared Timeline.  Absolute seconds
  are derived at call time so the structure is readable and editable in
  musical terms without touching any individual number.
- Chord durations vary within each section.  The A section alternates 2-bar
  and 1.5-bar blocks so the harmony has a subtle rhythmic wave rather than
  eight identical slabs.  Development uses uneven 3/2.5/3.5-bar blocks.
  The Ending expands to 3/4/4-bar pads.
- Voices stagger within each section:
    A / Reprise  — bass on beat, alto +½ beat, tenor +1 beat (bloom).
    B            — all three enter together (tight, bright).
    Development  — bass anticipates the bar line by ½ beat (unsettled);
                   tenor follows ½ beat behind alto.
    Ending       — wide stagger (bass / alto +1 / tenor +1.5 beats).
- Bass sustains ~¼ s into the next chord; tenor releases ~¼ s early.
  This creates a natural overlap that is different from the humanizer spread.
- Counter voice uses bar/beat positioning and dotted rhythms that
  intentionally cross bar lines rather than landing on them.  It enters
  one beat after the chord change in most sections.
- Lead melody is rewritten in beat values (Q/H/dotH/E etc.) with the same
  pitch content and D5/A5 climax preserved as an eighth-note run at bar 16.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from code_musics import synth
from code_musics.composition import bar_automation, grid_line, with_synth_ramp
from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    VelocityHumanizeSpec,
)
from code_musics.meter import E, H, Q, S, Timeline, W, dotted
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score, VelocityParamMap, VoiceSend

logger = logging.getLogger(__name__)

# ── timeline ──────────────────────────────────────────────────────────────────
TL = Timeline(bpm=88, meter=(4, 4))

# Bar positions referenced throughout:
#   bar  1  =    0.00 s  (prologue start)
#   bar  5  =   10.91 s  (A section)
#   bar 19  =   49.09 s  (B section)
#   bar 27  =   70.91 s  (Development)
#   bar 36  =   95.45 s  (Reprise)
#   bar 44  =  117.27 s  (Ending)
#   bar 55  =  147.27 s  (tail / end)


def build_ji_chorale_v2_score() -> Score:
    """Return the ji_chorale_v2 Score."""
    f0 = 110.0  # A2 = 110 Hz

    # ── pitches — 5-limit JI from f0 ──────────────────────────────────────
    A2 = f0  # 110.00
    B2 = f0 * 9 / 8  # 123.75
    D3 = f0 * 4 / 3  # 146.67  iv root
    E3 = f0 * 3 / 2  # 165.00
    F3 = D3 * 6 / 5  # 176.00  D-minor 3rd
    Fs2 = f0 * 5 / 6  #  91.67  sub-bass F#2
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
    A5 = A4 * 2  # 880.00  tonic, next octave

    # ── master effects ────────────────────────────────────────────────────
    master_effects: list[EffectSpec] = [
        EffectSpec("preamp", {"preset": "iron_color"}),
    ]
    if synth.has_external_plugin("lsp_compressor_stereo"):
        master_effects.append(
            EffectSpec(
                "plugin",
                {
                    "plugin_name": "lsp_compressor_stereo",
                    "params": {
                        "ratio": 2.4,
                        "attack_threshold_db": -24.0,
                        "attack_time_ms": 28.0,
                        "release_time_ms": 220.0,
                        "knee_db": -12.0,
                        "makeup_gain_db": 2.0,
                        "sidechain_mode": "RMS",
                        "dry_wet_balance": 100.0,
                    },
                },
            )
        )
    else:
        logger.warning(
            "Skipping optional ji_chorale_v2 glue compressor: LSP VST3 not available."
        )
    master_effects.extend(
        [
            EffectSpec(
                "chow_tape",
                {"drive": 0.35, "saturation": 0.35, "bias": 0.5, "mix": 35.0},
            ),
        ]
    )

    # ── score ─────────────────────────────────────────────────────────────
    score = Score(
        f0=f0,
        timing_humanize=TimingHumanizeSpec(preset="chamber", chord_spread_ms=7.0),
        master_effects=master_effects,
    )
    score.add_send_bus(
        "dark_hall",
        effects=[
            EffectSpec(
                "bricasti",
                {
                    "ir_name": "1 Halls 07 Large & Dark",
                    "wet": 1.0,
                    "highpass_hz": 300.0,
                    "lowpass_hz": 6_200.0,
                    "tilt_db": -3.0,
                    "tilt_pivot_hz": 1_800.0,
                },
            )
        ],
    )

    bass_automation = [
        bar_automation(
            target="cutoff_hz",
            timeline=TL,
            points=(
                (1, 0.0, 580.0),
                (5, 0.0, 760.0),
                (19, 0.0, 1_180.0),
                (27, 0.0, 860.0),
                (33, 0.0, 1_280.0),
                (36, 0.0, 1_080.0),
                (44, 0.0, 920.0),
                (51, 0.0, 1_120.0),
                (55, 0.0, 520.0),
                (59, 0.0, 520.0),
            ),
            clamp_min=180.0,
        ),
        bar_automation(
            target="filter_env_amount",
            timeline=TL,
            points=(
                (1, 0.0, 0.28),
                (5, 0.0, 0.36),
                (19, 0.0, 0.66),
                (27, 0.0, 0.44),
                (33, 0.0, 0.84),
                (36, 0.0, 0.62),
                (44, 0.0, 0.48),
                (51, 0.0, 0.68),
                (55, 0.0, 0.22),
                (59, 0.0, 0.22),
            ),
            clamp_min=0.0,
        ),
    ]
    tenor_automation = [
        bar_automation(
            target="cutoff_hz",
            timeline=TL,
            points=(
                (1, 0.0, 720.0),
                (5, 0.0, 980.0),
                (19, 0.0, 1_750.0),
                (27, 0.0, 1_150.0),
                (33, 0.0, 1_900.0),
                (36, 0.0, 1_550.0),
                (44, 0.0, 1_250.0),
                (51, 0.0, 1_750.0),
                (55, 0.0, 700.0),
                (59, 0.0, 700.0),
            ),
            clamp_min=220.0,
        ),
        bar_automation(
            target="resonance_q",
            timeline=TL,
            points=(
                (1, 0.0, 1.16),
                (19, 0.0, 1.84),
                (27, 0.0, 1.38),
                (33, 0.0, 2.06),
                (44, 0.0, 1.50),
                (55, 0.0, 1.05),
                (59, 0.0, 1.05),
            ),
            clamp_min=0.5,
        ),
    ]
    alto_automation = [
        bar_automation(
            target="cutoff_hz",
            timeline=TL,
            points=(
                (1, 0.0, 850.0),
                (5, 0.0, 1_050.0),
                (19, 0.0, 2_200.0),
                (27, 0.0, 1_350.0),
                (33, 0.0, 2_450.0),
                (36, 0.0, 1_900.0),
                (44, 0.0, 1_500.0),
                (51, 0.0, 2_300.0),
                (55, 0.0, 820.0),
                (59, 0.0, 820.0),
            ),
            clamp_min=220.0,
        ),
        bar_automation(
            target="brightness_tilt",
            timeline=TL,
            points=(
                (1, 0.0, -0.18),
                (5, 0.0, -0.08),
                (19, 0.0, 0.14),
                (27, 0.0, -0.02),
                (33, 0.0, 0.18),
                (36, 0.0, 0.08),
                (44, 0.0, 0.00),
                (51, 0.0, 0.12),
                (55, 0.0, -0.22),
                (59, 0.0, -0.22),
            ),
        ),
    ]
    counter_automation = [
        bar_automation(
            target="cutoff_hz",
            timeline=TL,
            points=(
                (1, 0.0, 1_250.0),
                (9, 0.0, 1_700.0),
                (19, 0.0, 2_600.0),
                (27, 0.0, 1_800.0),
                (33, 0.0, 3_200.0),
                (36, 0.0, 2_200.0),
                (44, 0.0, 2_000.0),
                (47, 0.0, 2_600.0),
                (51, 0.0, 3_000.0),
                (55, 0.0, 900.0),
                (59, 0.0, 900.0),
            ),
            clamp_min=240.0,
        ),
        bar_automation(
            target="filter_env_amount",
            timeline=TL,
            points=(
                (1, 0.0, 0.72),
                (9, 0.0, 0.92),
                (19, 0.0, 1.02),
                (27, 0.0, 0.84),
                (33, 0.0, 1.18),
                (36, 0.0, 0.96),
                (44, 0.0, 0.88),
                (51, 0.0, 1.08),
                (55, 0.0, 0.50),
                (59, 0.0, 0.50),
            ),
            clamp_min=0.0,
        ),
    ]

    # ── voices ────────────────────────────────────────────────────────────
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "filtered_stack",
            "waveform": "square",
            "n_harmonics": 12,
            "cutoff_hz": 900.0,
            "keytrack": 0.1,
            "resonance_q": 0.707,
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
        automation=bass_automation,
        sends=[VoiceSend("dark_hall", send_db=-19.0)],
    )
    chord_defaults: dict = {
        "engine": "filtered_stack",
        "waveform": "saw",
        "harmonic_rolloff": 0.38,
        "n_harmonics": 8,
        "cutoff_hz": 1_350.0,
        "keytrack": 0.12,
        "resonance_q": 1.27,
        "filter_env_amount": 0.16,
        "filter_env_decay": 0.75,
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
            "resonance_q": 1.61,
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
        automation=tenor_automation,
        sends=[VoiceSend("dark_hall", send_db=-11.5)],
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
        automation=alto_automation,
        sends=[VoiceSend("dark_hall", send_db=-9.0)],
    )
    score.add_voice(
        "counter",
        synth_defaults={
            "engine": "filtered_stack",
            "preset": "reed_lead",
            "cutoff_hz": 1_400.0,
            "keytrack": 0.1,
            "resonance_q": 1.84,
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
        automation=counter_automation,
        sends=[VoiceSend("dark_hall", send_db=-10.5)],
    )
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "triangle",
            "cutoff_hz": 2_800.0,
            "keytrack": 0.12,
            "resonance_q": 2.40,
            "filter_env_amount": 0.13,
            "filter_env_decay": 1.0,
            "filter_drive": 0.10,
            "attack": 0.085,
            "decay": 1.25,
            "sustain_level": 0.60,
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
        ],
        mix_db=-4.0,
        pan=0.20,
        velocity_group="melody",
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        velocity_to_params={
            "filter_env_amount": VelocityParamMap(min_value=0.30, max_value=0.75),
            "resonance_q": VelocityParamMap(min_value=1.16, max_value=2.06),
        },
        sends=[VoiceSend("dark_hall", send_db=-8.5)],
    )

    # ── convenience values ────────────────────────────────────────────────
    spb = TL.seconds_per_beat  # seconds per quarter-note beat
    q = TL.duration(Q)  # quarter note in seconds
    h = TL.duration(H)  # half note

    # Overlap/trim applied to every harmony chord:
    BASS_EXTEND = 0.30  # bass hangs into the next chord a little
    TENOR_TRIM = 0.25  # tenor releases slightly before chord boundary

    # ── helpers ───────────────────────────────────────────────────────────

    def _add_chord(
        *,
        bar: int,
        beat: float,
        dur_bars: float,
        bass_f: float,
        tenor_f: float,
        alto_f: float,
        velocity: float,
        bass_stagger: float = 0.0,  # beats
        alto_stagger: float = 0.0,  # beats
        tenor_stagger: float = 0.0,  # beats
        extra_alto_f: float | None = None,  # second alto note (for maj7 doublings)
        extra_alto_amp: float = 0.09,
        skip_alto: bool = False,  # omit the main alto note (for manual arpeggiation)
    ) -> None:
        dur_secs = TL.measures(dur_bars)
        t_bass = TL.at(bar=bar, beat=beat) + bass_stagger * spb
        t_alto = TL.at(bar=bar, beat=beat) + alto_stagger * spb
        t_tenor = TL.at(bar=bar, beat=beat) + tenor_stagger * spb
        score.add_note(
            "bass",
            start=t_bass,
            duration=dur_secs + BASS_EXTEND,
            freq=bass_f,
            amp=0.25,
            velocity=velocity,
        )
        score.add_note(
            "tenor",
            start=t_tenor,
            duration=max(0.5, dur_secs - tenor_stagger * spb - TENOR_TRIM),
            freq=tenor_f,
            amp=0.21,
            velocity=velocity,
        )
        if not skip_alto:
            score.add_note(
                "alto",
                start=t_alto,
                duration=max(0.5, dur_secs - alto_stagger * spb),
                freq=alto_f,
                amp=0.18,
                velocity=velocity,
            )
        if extra_alto_f is not None:
            score.add_note(
                "alto",
                start=t_alto,
                duration=max(0.5, dur_secs - alto_stagger * spb),
                freq=extra_alto_f,
                amp=extra_alto_amp,
                velocity=velocity,
            )

    def _add_counter(
        bar: int, beat: float, notes: list[tuple[float, float, float]]
    ) -> None:
        """Place counter notes from bar/beat; notes are (freq, dur_beats, velocity)."""
        t = TL.at(bar=bar, beat=beat)
        for freq, dur_beats, velocity in notes:
            dur_secs = dur_beats * spb
            score.add_note(
                "counter",
                start=t,
                duration=dur_secs,
                freq=freq,
                amp=0.22,
                velocity=velocity,
            )
            t += dur_secs

    def _add_lead_phrase(
        *,
        bar: int,
        notes: list[tuple[float, Any]],
        synth_start: dict,
        synth_end: dict,
        amp_db: float,
        velocities: list[float],
    ) -> None:
        freqs = [f for f, _ in notes]
        durs = [d for _, d in notes]
        phrase = grid_line(freqs, durs, timeline=TL, pitch_kind="freq", amp_db=amp_db)
        phrase = with_synth_ramp(phrase, start=synth_start, end=synth_end)
        phrase = replace(
            phrase,
            events=tuple(
                replace(ev, velocity=v)
                for ev, v in zip(phrase.events, velocities, strict=True)
            ),
        )
        score.add_phrase("lead", phrase, start=TL.at(bar=bar))

    # ══════════════════════════════════════════════════════════════════════
    # PROLOGUE  (bars 1–4, 0–10.9 s)
    # ══════════════════════════════════════════════════════════════════════
    # Bass: two drone hits on F#2, long fades.
    score.add_note(
        "bass",
        start=TL.at(bar=1),
        duration=TL.measures(2.5),
        freq=Fs2,
        amp=0.17,
        velocity=0.72,
    )
    score.add_note(
        "bass",
        start=TL.at(bar=3, beat=2),
        duration=TL.measures(1.5),
        freq=Fs2,
        amp=0.20,
        velocity=0.78,
    )

    # Tenor: sparse arpeggiation over bars 2–4, building a quiet A-major
    # triad before the A section arrives.
    score.add_note(
        "tenor", start=TL.at(bar=2), duration=q, freq=A3, amp=0.10, velocity=0.70
    )
    score.add_note(
        "tenor",
        start=TL.at(bar=2, beat=2),
        duration=h,
        freq=Cs4,
        amp=0.11,
        velocity=0.73,
    )
    score.add_note(
        "tenor", start=TL.at(bar=3), duration=q * 1.5, freq=B3, amp=0.10, velocity=0.70
    )
    score.add_note(
        "tenor",
        start=TL.at(bar=3, beat=2),
        duration=h,
        freq=A3,
        amp=0.11,
        velocity=0.72,
    )
    score.add_note(
        "tenor",
        start=TL.at(bar=4),
        duration=TL.measures(1.0),
        freq=Cs4,
        amp=0.12,
        velocity=0.75,
    )

    # ══════════════════════════════════════════════════════════════════════
    # A SECTION  (bars 5–18, ~10.9–49.1 s)
    # 8 chords alternating vi (Fs3) and iv (D3).
    # Durations: [2, 2, 1.5, 1.5, 2, 1.5, 2, 1.5] bars (total 14 bars).
    # Stagger: bass on beat, alto +½ beat, tenor +1 beat.
    # ══════════════════════════════════════════════════════════════════════

    # (bar, beat, dur_bars, bass_f, tenor_f, alto_f, sus_alto_freq)
    # sus_alto_freq: old alto pitch held 1 beat over the new chord = 7-6 suspension.
    # Cs4 over D3 = major-7th dissonance; resolves when alto moves to A3 one beat later.
    a_chords = [
        (5, 0.0, 2.0, Fs3, A3, Cs4, None),
        (7, 0.0, 2.0, D3, F3, A3, Cs4),  # 7-6 suspension
        (9, 0.0, 1.5, Fs3, A3, Cs4, None),
        (10, 2.0, 1.5, D3, F3, A3, Cs4),  # 7-6 suspension
        (12, 0.0, 2.0, Fs3, A3, Cs4, None),
        (14, 0.0, 1.5, D3, F3, A3, Cs4),  # 7-6 suspension
        (15, 2.0, 2.0, Fs3, A3, Cs4, None),
        (17, 2.0, 1.5, D3, F3, A3, None),  # final iv: no suspension, let it breathe
    ]
    a_velocities = [0.90, 0.95, 1.02, 1.05, 1.12, 0.98, 0.90, 0.85]

    for (cbar, cbeat, cdur, bf, tf, af, sus_alto), vel in zip(
        a_chords, a_velocities, strict=True
    ):
        # Where there is a suspension, the new alto (A3) enters 1 beat late.
        alto_stag = 1.0 if sus_alto is not None else 0.5
        if sus_alto is not None:
            # Old Cs4 rings 1 beat + a little decay over the new D3 bass.
            score.add_note(
                "alto",
                start=TL.at(bar=cbar, beat=cbeat),
                duration=q + 0.15,
                freq=sus_alto,
                amp=0.14,
                velocity=vel * 0.78,
            )
        _add_chord(
            bar=cbar,
            beat=cbeat,
            dur_bars=cdur,
            bass_f=bf,
            tenor_f=tf,
            alto_f=af,
            velocity=vel,
            alto_stagger=alto_stag,
            tenor_stagger=1.0,
        )

    # ── A section counter (enters from chord 3, bar 9) ───────────────────
    # Enters 1 beat after each chord change; dotted rhythms cross bar lines.

    # Chord 3 vi (bar 9, 6 beats):
    _add_counter(9, 1.0, [(E4, 3.0, 1.00), (D4, 1.5, 0.93), (Cs4, 1.5, 0.85)])
    # Chord 4 iv (bar 10 beat 2, ½ beat into chord → bar 10 beat 2.5, 5.5 beats):
    _add_counter(10, 2.5, [(F4, 2.5, 1.05), (E4, 2.0, 0.97), (D4, 1.0, 0.88)])
    # Chord 5 vi (bar 12, 8 beats):
    _add_counter(12, 0.0, [(E4, 1.5, 1.02), (Cs4, 3.5, 0.92), (E4, 3.0, 1.00)])
    # Chord 6 iv (bar 14, 6 beats):
    _add_counter(14, 0.0, [(D4, 2.0, 0.92), (F4, 2.5, 1.00), (A4, 1.5, 1.05)])
    # Chord 7 vi (bar 15 beat 2, 8 beats):
    _add_counter(15, 2.0, [(Cs4, 2.0, 0.95), (E4, 4.0, 1.05), (D4, 2.0, 0.88)])
    # Chord 8 iv (bar 17 beat 2, 6 beats):
    _add_counter(17, 2.0, [(D4, 2.0, 0.90), (F4, 2.5, 1.00), (A4, 1.5, 1.05)])

    # ══════════════════════════════════════════════════════════════════════
    # B SECTION  (bars 19–26, ~49.1–70.9 s)
    # I – V – I in A major.  All voices enter together (bright, tight).
    # Chord durations: [3, 2.5, 2.5] bars (total 8 bars).
    # ══════════════════════════════════════════════════════════════════════

    # I chord (bar 19): bass + tenor enter cleanly; alto rolls up A3→Cs4 in 16ths.
    _s = TL.duration(S)
    _add_chord(
        bar=19,
        beat=0.0,
        dur_bars=3.0,
        bass_f=A2,
        tenor_f=E3,
        alto_f=Cs4,
        velocity=1.05,
        skip_alto=True,
    )
    score.add_note(
        "alto", start=TL.at(bar=19), duration=_s, freq=A3, amp=0.11, velocity=0.88
    )
    score.add_note(
        "alto",
        start=TL.at(bar=19) + _s,
        duration=TL.measures(3.0) - _s,
        freq=Cs4,
        amp=0.18,
        velocity=1.05,
    )

    # V and I-return chords — normal tight entries.
    for (cbar, cbeat, cdur, bf, tf, af), vel in zip(
        [(22, 0.0, 2.5, E3, Gs3, B3), (24, 2.0, 2.5, A2, E3, Cs4)],
        [1.10, 0.95],
        strict=True,
    ):
        _add_chord(
            bar=cbar,
            beat=cbeat,
            dur_bars=cdur,
            bass_f=bf,
            tenor_f=tf,
            alto_f=af,
            velocity=vel,
        )

    # B section counter: imitative canon 1 bar behind the lead, transposed a 4th below.
    # Lead bar 19: E4(H) Cs5(H) — rising 6th.
    # Counter bar 20: B3(H) Gs4(H) — same contour a 4th below (tonal answer, dominant pitch-space).
    # Then continues imitating for 2 more bars before diverging.
    _add_counter(20, 0.0, [(B3, 2.0, 0.88), (Gs4, 2.0, 0.85)])  # imitates bar 19
    _add_counter(
        21, 0.0, [(E4, 1.0, 0.90), (B3, 1.0, 0.85), (Fs4, 2.0, 0.88)]
    )  # imitates bar 20
    _add_counter(
        22, 0.0, [(Fs4, 1.0, 0.92), (Cs4, 1.0, 0.88), (B3, 2.0, 0.85)]
    )  # imitates bar 21
    _add_counter(23, 0.0, [(Gs4, 2.0, 0.88), (E4, 2.0, 0.82)])  # V
    _add_counter(25, 0.0, [(Cs4, 2.0, 0.80), (A3, 2.0, 0.75)])  # I-return, winding down

    # ══════════════════════════════════════════════════════════════════════
    # DEVELOPMENT  (bars 27–35, ~70.9–95.5 s)
    # F#m7 → Bm → E dominant.
    # Bass anticipates each chord by ½ beat; tenor follows ½ beat behind alto.
    # Chord durations: [3, 2.5, 3.5] bars (total 9 bars).
    # ══════════════════════════════════════════════════════════════════════

    # F#m7 — bass anticipates by ½ beat (forward energy into Development).
    _add_chord(
        bar=27,
        beat=0.0,
        dur_bars=3.0,
        bass_f=Fs2,
        tenor_f=A3,
        alto_f=Cs4,
        velocity=1.02,
        bass_stagger=-0.5,
        alto_stagger=0.0,
        tenor_stagger=0.5,
    )

    # Bm — bass tritone suspension: Fs2 rings 1 beat before B2 enters.
    # Fs2 against B2 = tritone (diabolus in musica): maximum tension, baroque effect.
    score.add_note(
        "bass", start=TL.at(bar=30), duration=q + 0.2, freq=Fs2, amp=0.22, velocity=1.05
    )
    _add_chord(
        bar=30,
        beat=0.0,
        dur_bars=2.5,
        bass_f=B2,
        tenor_f=D3,
        alto_f=Fs3,
        velocity=1.08,
        bass_stagger=1.0,
        alto_stagger=0.0,
        tenor_stagger=0.5,
    )

    # V dominant — bass anticipates again for urgency into the cadence.
    _add_chord(
        bar=32,
        beat=2.0,
        dur_bars=3.5,
        bass_f=E3,
        tenor_f=Gs3,
        alto_f=B3,
        velocity=1.14,
        bass_stagger=-0.5,
        alto_stagger=0.0,
        tenor_stagger=0.5,
    )

    # Development counter: more exploratory, wider intervals.
    _add_counter(
        27, 0.0, [(E4, 2.0, 1.05), (Cs4, 2.0, 0.92), (E4, 4.0, 1.00), (D4, 4.0, 0.88)]
    )
    _add_counter(
        30, 0.0, [(Fs4, 3.0, 1.10), (E4, 2.5, 1.00), (D4, 2.5, 0.90), (Cs4, 2.0, 0.82)]
    )
    _add_counter(
        32, 2.0, [(Gs4, 3.5, 1.18), (Fs4, 2.5, 1.02), (E4, 4.5, 0.90), (D4, 3.5, 0.80)]
    )

    # ══════════════════════════════════════════════════════════════════════
    # REPRISE  (bars 36–43, ~95.5–117.3 s)
    # vi – I – vi – I.  Stagger: bass on beat, alto +¼, tenor +½.
    # ══════════════════════════════════════════════════════════════════════

    # vi (bar 36) — 3-note baroque arpeggio in alto: Fs3→A3→Cs4.
    # The arpeggio rolls through all three chord tones before holding the 3rd.
    _add_chord(
        bar=36,
        beat=0.0,
        dur_bars=2.0,
        bass_f=Fs3,
        tenor_f=A3,
        alto_f=Cs4,
        velocity=1.00,
        alto_stagger=0.5,
        tenor_stagger=0.5,
        skip_alto=True,
    )
    score.add_note(
        "alto", start=TL.at(bar=36), duration=_s, freq=Fs3, amp=0.10, velocity=0.82
    )
    score.add_note(
        "alto", start=TL.at(bar=36) + _s, duration=_s, freq=A3, amp=0.13, velocity=0.90
    )
    score.add_note(
        "alto",
        start=TL.at(bar=36) + 2 * _s,
        duration=TL.measures(2.0) - 2 * _s,
        freq=Cs4,
        amp=0.18,
        velocity=1.00,
    )

    # Remaining reprise chords — normal bloom stagger.
    rep_chords_tail = [
        (38, 0.0, 2.0, A2, E3, Cs4, None),  # I
        (40, 0.0, 2.0, Fs3, A3, Cs4, None),  # vi
        (42, 0.0, 2.0, A2, E3, Cs4, Gs4),  # Amaj7 — alto doubles maj7
    ]
    rep_velocities_tail = [0.95, 1.00, 0.88]

    for (cbar, cbeat, cdur, bf, tf, af, af2), vel in zip(
        rep_chords_tail, rep_velocities_tail, strict=True
    ):
        _add_chord(
            bar=cbar,
            beat=cbeat,
            dur_bars=cdur,
            bass_f=bf,
            tenor_f=tf,
            alto_f=af,
            velocity=vel,
            alto_stagger=0.25,
            tenor_stagger=0.5,
            extra_alto_f=af2,
        )

    # Reprise counter: settling into simpler contours.
    _add_counter(36, 0.0, [(E4, 3.0, 1.08), (Cs4, 3.0, 0.90), (B4, 2.0, 0.82)])
    _add_counter(38, 1.0, [(Cs4, 3.0, 0.95), (E4, 4.0, 1.05)])
    _add_counter(40, 0.0, [(Cs4, 3.0, 0.95), (E4, 4.0, 1.05), (D4, 1.0, 0.88)])
    _add_counter(42, 0.0, [(Gs4, 3.5, 1.05), (B4, 4.5, 0.95)])

    # ══════════════════════════════════════════════════════════════════════
    # ENDING  (bars 44–54, ~117.3–147.3 s)
    # Wide, slowly dissolving.  Stagger: bass on beat, alto +1, tenor +1.5.
    # Dm7 has a doubled alto (A3 + C4); Amaj7 doubles Gs4.
    # ══════════════════════════════════════════════════════════════════════

    # Wide vi
    _add_chord(
        bar=44,
        beat=0.0,
        dur_bars=3.0,
        bass_f=Fs2,
        tenor_f=A3,
        alto_f=Cs5,
        velocity=1.05,
        alto_stagger=1.0,
        tenor_stagger=1.5,
    )

    # Dm7 — two notes in alto voiced wide
    _add_chord(
        bar=47,
        beat=0.0,
        dur_bars=4.0,
        bass_f=D3,
        tenor_f=F3,
        alto_f=A3,
        velocity=1.00,
        alto_stagger=1.0,
        tenor_stagger=1.5,
    )
    # C4 in alto enters with the same stagger
    score.add_note(
        "alto",
        start=TL.at(bar=47, beat=1.0),
        duration=TL.measures(4.0) - q,
        freq=C4,
        amp=0.16,
        velocity=1.00,
    )

    # Amaj7
    _add_chord(
        bar=51,
        beat=0.0,
        dur_bars=4.0,
        bass_f=A2,
        tenor_f=E3,
        alto_f=Cs4,
        velocity=0.88,
        alto_stagger=1.0,
        tenor_stagger=1.5,
        extra_alto_f=Gs4,
        extra_alto_amp=0.14,
    )

    # Ending counter: long, slowly fading melodic phrases.
    _add_counter(44, 0.0, [(E4, 4.0, 1.00), (Cs4, 4.0, 0.90), (E4, 4.0, 0.82)])
    _add_counter(
        47, 0.0, [(F4, 4.0, 1.08), (E4, 4.0, 0.95), (D4, 4.0, 0.85), (C4, 4.0, 0.75)]
    )
    _add_counter(51, 0.0, [(E4, 4.0, 0.95), (Gs4, 4.0, 1.05), (E4, 4.0, 0.82)])
    # Bar 54: quiet callback of the Development dun-dun motif (Fs4→E4→Cs4),
    # the top 3 notes of the bar-30 line, compressed into the 4-beat gap before
    # the tail.  Soft and diminuendo — a memory, not a statement.
    _add_counter(54, 0.0, [(Fs4, 1.5, 0.78), (E4, 1.5, 0.70), (Cs4, 1.0, 0.62)])

    # ══════════════════════════════════════════════════════════════════════
    # LEAD MELODY
    # Three add_phrase calls matching the three-movement phrase structure.
    # All durations are beat values compiled by grid_line through the TL.
    # ══════════════════════════════════════════════════════════════════════

    # Phrase 1 — prologue pickup + A section (bars 4–18, 60 beats, 45 notes).
    lead_a: list[tuple[float, Any]] = [
        # bar 4 — pickup
        (A4, Q),
        (Cs5, Q),
        (B4, H),
        # bar 5
        (A4, Q),
        (Cs5, H),
        (B4, Q),
        # bar 6
        (A4, H),
        (Gs4, Q),
        (A4, Q),
        # bar 7 — long Cs5, step down
        (Cs5, dotted(H)),
        (B4, Q),
        # bar 8 — dark F4 moment
        (A4, H),
        (F4, H),
        # bar 9 — ornamental run into B4
        (A4, Q),
        (Gs4, E),
        (A4, E),
        (B4, H),
        # bar 10 — high Cs5, then stepdown over chord change (beat 2)
        (Cs5, H),
        (B4, Q),
        (A4, Q),
        # bar 11 — over iv: F4 descent
        (F4, H),
        (E4, Q),
        (D4, Q),
        # bar 12 — rise through vi territory (hold back: Cs5 not yet)
        (E4, Q),
        (A4, Q),
        (Cs5, H),
        # bar 13 — circle below, no Cs5 preview; save the top for bar 16
        (B4, Q),
        (A4, Q),
        (B4, H),
        # bar 14 — ascending iv chord tones
        (D4, Q),
        (F4, Q),
        (A4, H),
        # bar 15 — Gs4 ornament leads to Cs5 over vi
        (Gs4, Q),
        (A4, Q),
        (Cs5, H),
        # bar 16 — climax: D5→A5 apex, then quick descent
        (D5, E),
        (A5, E),
        (Cs5, Q),
        (B4, E),
        (A4, E),
        (A4, Q),
        # bar 17
        (A4, H),
        (B4, Q),
        (A4, Q),
        # bar 18 — close A section
        (F4, Q),
        (A4, H),
        (Gs4, Q),
    ]
    lead_a_velocities: list[float] = [
        0.82,
        1.00,
        0.92,  # bar 4
        0.88,
        1.10,
        0.95,  # bar 5
        0.95,
        0.82,
        0.90,  # bar 6
        1.15,
        0.90,  # bar 7
        0.88,
        1.05,  # bar 8
        0.92,
        0.80,
        0.85,
        1.00,  # bar 9
        1.18,
        0.98,
        0.85,  # bar 10
        1.05,
        0.90,
        0.82,  # bar 11
        0.85,
        0.95,
        1.08,  # bar 12 — held back; Cs5 less emphatic
        0.88,
        0.78,
        0.85,  # bar 13 — quieter circle (no Cs5)
        0.88,
        1.00,
        1.10,  # bar 14 — starting to build
        0.92,
        0.98,
        1.25,  # bar 15 — Cs5 now emphatic; last step before leap
        1.25,
        1.42,
        1.15,
        1.00,
        0.88,
        0.95,  # bar 16 — D5→A5 apex, descent
        1.05,
        0.92,
        0.85,  # bar 17
        0.95,
        1.05,
        0.80,  # bar 18
    ]

    # Phrase 2 — B + Development (bars 19–35, 68 beats, 42 notes).
    lead_b_dev: list[tuple[float, Any]] = [
        # ─ B section ─
        # bar 19
        (E4, H),
        (Cs5, H),
        # bar 20
        (A4, Q),
        (E4, Q),
        (B4, H),
        # bar 21 — G# color
        (B4, Q),
        (Gs4, Q),
        (Fs4, H),
        # bar 22 — V chord; long B4
        (E4, Q),
        (B4, dotted(H)),
        # bar 23
        (Gs4, Q),
        (Fs4, Q),
        (E4, H),
        # bar 24 — I returns beat 2; Cs5 bridges
        (Cs5, H),
        (A4, Q),
        (E4, Q),
        # bar 25
        (A4, dotted(H)),
        (Cs5, Q),
        # bar 26
        (B4, H),
        (A4, H),
        # ─ Development ─
        # bar 27
        (E4, Q),
        (Cs4, Q),
        (E4, H),
        # bar 28
        (Fs4, Q),
        (E4, Q),
        (Cs4, H),
        # bar 29 — hovering, ambiguous
        (E4, H),
        (Cs4, H),
        # bar 30 — Bm territory
        (Fs4, Q),
        (E4, Q),
        (D4, H),
        # bar 31
        (D4, dotted(H)),
        (Cs4, Q),
        # bar 32 — V prep begins beat 2
        (E4, H),
        (Fs4, H),
        # bar 33 — V chord bright arrival
        (Gs4, dotted(H)),
        (Fs4, Q),
        # bar 34 — building
        (E4, Q),
        (Gs4, Q),
        (B4, H),
        # bar 35 — settling for reprise
        (Gs4, H),
        (E4, H),
    ]
    lead_b_dev_velocities: list[float] = [
        1.00,
        1.20,  # bar 19
        0.95,
        0.82,
        1.05,  # bar 20
        1.05,
        0.90,
        0.88,  # bar 21
        0.85,
        1.10,  # bar 22
        1.05,
        0.90,
        0.85,  # bar 23
        1.15,
        0.95,
        0.80,  # bar 24
        0.92,
        1.00,  # bar 25
        1.05,
        0.88,  # bar 26
        0.90,
        0.82,
        1.00,  # bar 27
        1.05,
        0.92,
        0.85,  # bar 28
        0.80,
        0.85,  # bar 29
        1.00,
        0.88,
        0.95,  # bar 30
        0.95,
        0.80,  # bar 31
        0.88,
        1.05,  # bar 32
        1.25,
        1.05,  # bar 33 — Gs4 arrival
        0.92,
        1.10,
        1.20,  # bar 34
        1.10,
        0.90,  # bar 35
    ]

    # Phrase 3 — Reprise + Ending (bars 36–54, 76 beats, 39 notes).
    lead_rep_end: list[tuple[float, Any]] = [
        # ─ Reprise ─
        # bar 36
        (Cs5, H),
        (B4, H),
        # bar 37 — ornamental turn
        (A4, Q),
        (Gs4, E),
        (A4, E),
        (Cs5, H),
        # bar 38 — I chord, bright fifth
        (E4, H),
        (B4, H),
        # bar 39
        (Cs5, Q),
        (B4, Q),
        (A4, H),
        # bar 40
        (Cs5, H),
        (A4, H),
        # bar 41 — lower register, settling
        (E4, dotted(H)),
        (Cs4, Q),
        # bar 42 — Amaj7 moment
        (Gs4, H),
        (B4, H),
        # bar 43 — long held tonic
        (A4, W),
        # ─ Ending ─
        # bar 44
        (E4, H),
        (Cs5, H),
        # bar 45
        (A4, dotted(H)),
        (Gs4, Q),
        # bar 46
        (A4, W),
        # bar 47 — Dm7
        (F4, H),
        (A4, H),
        # bar 48
        (C4, Q),
        (D4, Q),
        (F4, H),
        # bar 49 — Gs4 over Dm7: bittersweet
        (A4, H),
        (Gs4, H),
        # bar 50
        (A4, W),
        # bar 51 — Amaj7
        (Cs5, H),
        (E4, H),
        # bar 52
        (Gs4, Q),
        (A4, Q),
        (Cs5, H),
        # bar 53
        (B4, dotted(H)),
        (A4, Q),
        # bar 54 — final fade
        (A4, W),
    ]
    lead_rep_end_velocities: list[float] = [
        1.10,
        0.95,  # bar 36
        0.88,
        0.78,
        0.82,
        1.05,  # bar 37
        0.85,
        1.05,  # bar 38
        1.15,
        1.00,
        0.90,  # bar 39
        1.05,
        0.92,  # bar 40
        0.82,
        0.75,  # bar 41
        1.00,
        1.10,  # bar 42
        0.95,  # bar 43
        0.92,
        1.05,  # bar 44
        0.98,
        0.80,  # bar 45
        0.85,  # bar 46
        1.00,
        1.08,  # bar 47
        0.90,
        0.95,
        1.05,  # bar 48
        1.10,
        1.00,  # bar 49
        0.90,  # bar 50
        0.95,
        0.82,  # bar 51
        0.88,
        0.92,
        1.00,  # bar 52
        0.95,
        0.80,  # bar 53
        0.72,  # bar 54 — final whisper
    ]

    _add_lead_phrase(
        bar=4,
        notes=lead_a,
        synth_start={"cutoff_hz": 1_900.0, "release": 0.30},
        synth_end={"cutoff_hz": 2_200.0, "release": 0.28},
        amp_db=-18.0,
        velocities=lead_a_velocities,
    )
    _add_lead_phrase(
        bar=19,
        notes=lead_b_dev,
        synth_start={"cutoff_hz": 2_200.0, "release": 0.26},
        synth_end={"cutoff_hz": 2_400.0, "release": 0.24},
        amp_db=-17.5,
        velocities=lead_b_dev_velocities,
    )
    _add_lead_phrase(
        bar=36,
        notes=lead_rep_end,
        synth_start={"cutoff_hz": 2_100.0, "release": 0.26},
        synth_end={"cutoff_hz": 1_750.0, "release": 0.34},
        amp_db=-18.0,
        velocities=lead_rep_end_velocities,
    )

    # ══════════════════════════════════════════════════════════════════════
    # CODA  (bars 55–59, ~147.3–160.9 s)
    # Dissolution: the piece fragments into isolated gestures with long
    # silences between.  Only lead, counter, and a soft bass tonic remain.
    # ══════════════════════════════════════════════════════════════════════

    # Bass: one soft tonic sustain, then silence.
    score.add_note(
        "bass",
        start=TL.at(bar=55),
        duration=TL.measures(3.0),
        freq=A2,
        amp=0.08,
        velocity=0.52,
    )

    # Counter: a single falling phrase into the silence.
    _add_counter(
        55, 0.0, [(E4, 2.0, 0.65), (D4, 1.0, 0.58), (Cs4, 1.5, 0.50), (A3, 1.5, 0.42)]
    )

    # Lead: three isolated notes with widening silences; the last is barely audible.
    score.add_note(
        "lead",
        start=TL.at(bar=55),
        duration=TL.duration(H),
        freq=Cs5,
        amp_db=-24.0,
        velocity=0.62,
    )
    score.add_note(
        "lead",
        start=TL.at(bar=56),
        duration=TL.duration(dotted(H)),
        freq=A4,
        amp_db=-25.5,
        velocity=0.55,
    )
    score.add_note(
        "lead",
        start=TL.at(bar=57, beat=2),
        duration=TL.duration(W),
        freq=E4,
        amp_db=-27.0,
        velocity=0.48,
    )
    score.add_note(
        "lead",
        start=TL.at(bar=59),
        duration=TL.duration(W),
        freq=A4,
        amp_db=-29.0,
        velocity=0.40,
    )

    # Tail: silent note to hold the render window open past the last reverb tail.
    score.add_note("bass", start=TL.at(bar=60), duration=0.5, freq=A2, amp=0.001)

    return score


PIECES: dict[str, PieceDefinition] = {
    "ji_chorale_v2": PieceDefinition(
        name="ji_chorale_v2",
        output_name="17b_ji_chorale_v2",
        build_score=build_ji_chorale_v2_score,
        sections=(
            PieceSection(
                label="Prologue", start_seconds=TL.at(bar=1), end_seconds=TL.at(bar=5)
            ),
            PieceSection(
                label="A", start_seconds=TL.at(bar=5), end_seconds=TL.at(bar=19)
            ),
            PieceSection(
                label="B", start_seconds=TL.at(bar=19), end_seconds=TL.at(bar=27)
            ),
            PieceSection(
                label="Development",
                start_seconds=TL.at(bar=27),
                end_seconds=TL.at(bar=36),
            ),
            PieceSection(
                label="Reprise", start_seconds=TL.at(bar=36), end_seconds=TL.at(bar=44)
            ),
            PieceSection(
                label="Ending", start_seconds=TL.at(bar=44), end_seconds=TL.at(bar=55)
            ),
            PieceSection(
                label="Coda",
                start_seconds=TL.at(bar=55),
                end_seconds=TL.at(bar=60) + 2.0,
            ),
        ),
    )
}
