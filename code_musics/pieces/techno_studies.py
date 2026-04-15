"""Techno studies — harmonic-series material at 130 BPM.

Tuning: partial series of f0 = 55 Hz (A1).
Selected harmonic materials:
  partial  1   =  55.0 Hz  (A1)        bass root (sub)
  partial  1.5 =  82.5 Hz  (E2)        bass fifth
  partial  2   = 110.0 Hz  (A2)        bass octave
  partial  3.5 = 192.5 Hz              septimal 7th colour (7/4 × A2)
  partial  4   = 220.0 Hz  (A3)        lead low anchor (octave below P8)
  partial  5   = 275.0 Hz  (C#4 +14¢) harmonic major third — available, unused
  partial  6   = 330.0 Hz  (E4 +  2¢) harmonic perfect fifth — lead open shadow
  partial  7   = 385.0 Hz  (G4 – 31¢) septimal 7th — spice only
  partial  8   = 440.0 Hz  (A4)        lead anchor (home)
  partial  9   = 495.0 Hz  (B4 +  4¢) whole step above anchor — spice only
  partial 11   = 605.0 Hz              undecimal super-fourth — spice only

BPM = 130.  1 bar ≈ 1.846 s.  1 beat ≈ 0.4615 s.  1 sixteenth ≈ 0.1154 s.

Piece: spectral_kick
Structure:
  bars  1– 2   intro: kick alone
  bars  3– 4   kick + bass
  bars  5–12   kick + bass + lead phrase A (2 × 4-bar)
  bars 13–16   lead phrase B (pre-drop — dissolving, unresolved)
  bars 17–20   drop: kick + bass only
  bars 21–28   lead phrase A returns (2 × 4-bar)
  bars 29–32   lead phrase C: 3-note skeleton, one bar each
  bars 33–36   outro: kick + bass only
"""

from __future__ import annotations

from code_musics.automation import AutomationSegment, AutomationSpec, AutomationTarget
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Score, VoiceSend

BPM: float = 130.0
BEAT: float = 60.0 / BPM  # quarter-note duration ≈ 0.4615 s
BAR: float = 4.0 * BEAT  # 4/4 bar ≈ 1.846 s
S16: float = BEAT / 4.0  # sixteenth-note ≈ 0.1154 s

F0: float = 55.0  # A1 — score root

# Harmonic partial constants (freq = F0 * partial)
P1: float = 1.0  # A1    55.0 Hz  bass root (sub)
P15: float = 1.5  # E2    82.5 Hz  bass fifth
P2: float = 2.0  # A2   110.0 Hz  bass octave
P3: float = 3.0  # E3   165.0 Hz  harmonic fifth — lead lower voice
P35: float = 3.5  # —    192.5 Hz  septimal 7th colour (bass accent)
P4: float = 4.0  # A3   220.0 Hz  lead home
P5: float = 5.0  # C#4  275.0 Hz  harmonic major third (available)
P6: float = 6.0  # E4   330.0 Hz  harmonic perfect fifth — open shadow
P7: float = 7.0  # G4   385.0 Hz  septimal 7th — spice only
P8: float = 8.0  # A4   440.0 Hz  lead home
P9: float = 9.0  # B4   495.0 Hz  whole step above home — spice only
P11: float = 11.0  # —    605.0 Hz  undecimal super-fourth — spice only

# note type: (bar_offset, beat, n16, partial, gate_sixteenths, amp_db)
_LeadNote = tuple[int, int, int, float, float, float]


def _pos(bar: int, beat: int = 1, n16: int = 0) -> float:
    """Absolute time in seconds at bar:beat:sixteenth (bar and beat are 1-indexed)."""
    return (bar - 1) * BAR + (beat - 1) * BEAT + n16 * S16


def build_spectral_kick() -> Score:
    """Harmonic-series techno sketch at 130 BPM.

    The 11th partial (605 Hz, undecimal super-fourth) is the xenharmonic signature —
    neither fourth nor tritone, just wrong in a compelling way. The 7th partial
    (385 Hz) gives the dark, heavy septimal colour. Together they outline a vocabulary
    that is clearly not 12-TET while remaining consonant within the harmonic series.

    Lead voice: short FM bell tones, Detroit-techno style.
    Three pitches only: A3 (P4), A4 (P8), E4 (P6 — JI fifth).
    FM mod at 1:1 ratio keeps all sidebands on integer harmonics of the carrier.
    Bass: polyblep acid with JI-tuned pitches.
    Kick: 909-style, 4-on-the-floor.
    """
    score = Score(
        f0_hz=F0,
        master_effects=[
            EffectSpec(
                "compressor",
                {
                    # Glue compressor: gentle ratio, slow attack, wide knee, RMS/feedback.
                    # Slow 25ms attack lets kick transients through; 300ms release.
                    # Detector HP at 120 Hz (standard master-comp SC trim) prevents
                    # kick/bass low-end from driving excessive GR.
                    "threshold_db": -20.0,
                    "ratio": 2.5,
                    "attack_ms": 25.0,
                    "release_ms": 250.0,
                    "knee_db": 10.0,
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
        ],
    )

    total_bars = 36

    # ------------------------------------------------------------------
    # Kick: 909-style, 4-on-the-floor for the full piece
    # ------------------------------------------------------------------
    score.add_voice(
        "kick",
        synth_defaults={"engine": "kick_tom", "preset": "909_techno"},
        effects=[EffectSpec("compressor", {"preset": "kick_punch"})],
        normalize_peak_db=-6.0,
        mix_db=-4.0,
        velocity_humanize=None,
    )

    for bar in range(1, total_bars + 1):
        for beat in range(1, 5):
            score.add_note(
                "kick",
                start=_pos(bar, beat),
                duration=1.0,
                freq=62.0,
                amp_db=-6.0,
            )

    # ------------------------------------------------------------------
    # Shared send bus: Bricasti room reverb — lead, hats, clap only.
    # 100 % wet (send return); voices balance via send_db.
    # ------------------------------------------------------------------
    score.add_send_bus(
        "room",
        effects=[
            EffectSpec(
                "bricasti",
                {
                    "ir_name": "1 Halls 07 Large & Dark",
                    "wet": 1.0,
                    "lowpass_hz": 7000.0,
                    "highpass_hz": 200.0,
                },
            )
        ],
    )

    # ------------------------------------------------------------------
    # Bass: polyblep acid, JI-tuned partials
    # ------------------------------------------------------------------
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "polyblep",
            "preset": "moog_bass",
            "params": {
                # Saw for even+odd harmonics (110, 220, 330 Hz content) —
                # spectral bridge between the sub bass and the P6/P7/P9 chord.
                # cutoff_hz=220 is dark/sub-focused but not buried; the filter
                # envelope opens it to 440 Hz (P8, the lead root) at each attack
                # for a real Moog-style pluck punch. resonance_q=0.707 (Butterworth)
                # keeps the filter flat at the base cutoff without suckout.
                "waveform": "saw",
                "cutoff_hz": 220.0,
                "filter_env_amount": 1.0,
                "filter_env_decay": 0.10,
                "filter_drive": 0.50,
                "resonance_q": 0.707,
            },
        },
        mix_db=-7.0,
        velocity_humanize=None,
        max_polyphony=1,
        effects=[
            EffectSpec(
                "compressor", {"preset": "kick_duck", "sidechain_source": "kick"}
            ),
        ],
        automation=[
            # Cutoff opens from 220 Hz through the build (attack peak 440 Hz =
            # P8), collapses for the drop, then fully blooms in the return
            # (attack peak reaches ~860 Hz by bar 29).
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="cutoff_hz"),
                segments=(
                    AutomationSegment(
                        start=_pos(3),
                        end=_pos(17),
                        shape="linear",
                        start_value=220.0,
                        end_value=300.0,
                    ),
                    AutomationSegment(
                        start=_pos(17),
                        end=_pos(21),
                        shape="linear",
                        start_value=185.0,
                        end_value=220.0,
                    ),
                    AutomationSegment(
                        start=_pos(21),
                        end=_pos(29),
                        shape="linear",
                        start_value=220.0,
                        end_value=360.0,
                    ),
                ),
            ),
            # Filter env amount opens alongside cutoff — more pronounced "talking"
            # pluck as the piece builds; climax hits 1.6 (= moog_bass preset max).
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="filter_env_amount"),
                segments=(
                    AutomationSegment(
                        start=_pos(3),
                        end=_pos(17),
                        shape="linear",
                        start_value=1.0,
                        end_value=1.35,
                    ),
                    AutomationSegment(
                        start=_pos(17),
                        end=_pos(21),
                        shape="linear",
                        start_value=0.80,
                        end_value=1.0,
                    ),
                    AutomationSegment(
                        start=_pos(21),
                        end=_pos(29),
                        shape="linear",
                        start_value=1.0,
                        end_value=1.60,
                    ),
                ),
            ),
        ],
    )

    # Two-bar alternating bass pattern. Beat 2 and 4 stay open (kick+clap).
    # Pattern A: root-focused, steady drive.
    # Pattern B: more syncopated — push notes anticipate beats 2 and 1.
    _bass_pattern_a: list[tuple[int, int, float, int, float]] = [
        (1, 0, P1, 1, -6.0),  # beat 1: root
        (1, 2, P1, 1, -9.5),  # &-of-1: echo
        (2, 2, P1, 1, -8.0),  # &-of-2: mid-bar pull
        (3, 0, P1, 1, -6.5),  # beat 3: root
        (4, 2, P1, 1, -9.5),  # &-of-4: tail
    ]
    _bass_pattern_b: list[tuple[int, int, float, int, float]] = [
        (1, 0, P1, 1, -6.0),  # beat 1: root
        (1, 3, P1, 1, -10.5),  # a-of-1: push before 2 (anticipation)
        (2, 2, P1, 1, -8.5),  # &-of-2: continuation
        (3, 0, P1, 1, -6.5),  # beat 3: root
        (4, 3, P1, 1, -10.5),  # a-of-4: push before next bar's 1
    ]

    for bar in range(3, total_bars + 1):
        pattern = _bass_pattern_a if (bar % 2 == 1) else _bass_pattern_b
        for beat, n16, partial, gate_16ths, amp_db in pattern:
            score.add_note(
                "bass",
                start=_pos(bar, beat, n16),
                duration=gate_16ths * S16 * 0.75,
                partial=partial,
                amp_db=amp_db,
            )

    # Fifth (P15 = E2, 82.5 Hz): sparse and irregular — not every bar,
    # never the same beat position twice in a row.
    _fifth_hits: list[tuple[int, int, int]] = [
        (7, 4, 2),  # bar 7,  &-of-4
        (11, 2, 2),  # bar 11, &-of-2
        (15, 3, 3),  # bar 15, a-of-3
        (23, 4, 2),  # bar 23, &-of-4
        (27, 2, 2),  # bar 27, &-of-2
        (31, 1, 2),  # bar 31, &-of-1
    ]
    for bar, beat, n16 in _fifth_hits:
        score.add_note(
            "bass",
            start=_pos(bar, beat, n16),
            duration=S16 * 0.82,
            partial=P15,
            amp_db=-8.5,
        )

    # ------------------------------------------------------------------
    # Lead: FM bell tones, harmonic-series pitches
    # Short attack, fast decay, nearly no sustain — pointillist, not legato
    # ------------------------------------------------------------------
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "fm",
            "preset": "bell",
            "params": {
                # mod at 1× carrier: sidebands land on 2fc, 3fc, 4fc... —
                # all integer harmonics of the carrier, maximally consonant.
                # Starting dark (1.0) and ramping via automation; by bar 28
                # it has opened up to a brighter, more harmonically dense texture.
                "mod_ratio": 1.0,
                "mod_index": 1.0,
            },
            "env": {
                # Slower attack for a chord bloom rather than a bell strike.
                # Long decay so the three-note chord sustains through the bar.
                "attack_ms": 14.0,
                "decay_ms": 340.0,
                "sustain_ratio": 0.04,
                "release_ms": 260.0,
            },
        },
        mix_db=-8.0,
        velocity_humanize=None,
        effects=[
            # HP at 280 Hz: chord root is P6=330 Hz; cut below to keep bass space clean.
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 280.0, "slope_db_per_oct": 24}
                    ]
                },
            ),
            # Chorus before delay: promotes to stereo and adds gentle width.
            # juno_subtle at low mix — spread only, no obvious wobble.
            EffectSpec("chorus", {"preset": "juno_subtle", "mix": 0.20}),
            # Dotted-8th delay — high feedback and mix for a dense, hypnotic wash.
            # At 130 BPM the tail rings through 6+ repetitions, filling the space
            # between chord hits.
            EffectSpec(
                "delay",
                {
                    "delay_seconds": 3.0 * S16,
                    "feedback": 0.54,
                    "mix": 0.48,
                },
            ),
            EffectSpec(
                "compressor", {"preset": "kick_duck", "sidechain_source": "kick"}
            ),
        ],
        automation=[
            # mod_index ramps 1.0 → 2.4 from bar 5 through bar 28:
            # the chord opens from dark and fundamental to harmonically complex.
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="mod_index"),
                segments=(
                    AutomationSegment(
                        start=_pos(5),
                        end=_pos(29),
                        shape="linear",
                        start_value=1.0,
                        end_value=2.4,
                    ),
                ),
            ),
            # decay ramps 0.340 → 0.550 s alongside mod_index — as the timbre
            # opens up harmonically, the chord bloom also sustains longer,
            # creating a cumulative sense of expansion toward the peak.
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="decay"),
                segments=(
                    AutomationSegment(
                        start=_pos(5),
                        end=_pos(29),
                        shape="linear",
                        start_value=0.340,
                        end_value=0.550,
                    ),
                ),
            ),
            # release ramps 0.260 → 0.400 s — longer trails as the piece opens,
            # creating progressively longer reverberant tails on chord hits.
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="release"),
                segments=(
                    AutomationSegment(
                        start=_pos(5),
                        end=_pos(29),
                        shape="linear",
                        start_value=0.260,
                        end_value=0.400,
                    ),
                ),
            ),
            # Pan: slow drift -0.08 → 0.0 (bars 5→17) → +0.06 (bars 17→28).
            # Below the threshold of conscious notice; just gives the chord a
            # sense of spatial breathing rather than sitting rigidly in the centre.
            AutomationSpec(
                target=AutomationTarget(kind="control", name="pan"),
                segments=(
                    AutomationSegment(
                        start=_pos(5),
                        end=_pos(17),
                        shape="linear",
                        start_value=-0.08,
                        end_value=0.0,
                    ),
                    AutomationSegment(
                        start=_pos(17),
                        end=_pos(29),
                        shape="linear",
                        start_value=0.0,
                        end_value=0.06,
                    ),
                ),
            ),
        ],
        sends=[
            VoiceSend(
                target="room",
                send_db=-9.0,
                automation=[
                    # Room reverb opens up as the piece builds:
                    # starts relatively dry, blooms through the return, peaks
                    # in the skeleton section for a ghostly, dissolving close.
                    AutomationSpec(
                        target=AutomationTarget(kind="control", name="send_db"),
                        segments=(
                            AutomationSegment(
                                start=_pos(5),
                                end=_pos(17),
                                shape="linear",
                                start_value=-9.0,
                                end_value=-5.0,
                            ),
                            AutomationSegment(
                                start=_pos(21),
                                end=_pos(29),
                                shape="linear",
                                start_value=-6.0,
                                end_value=-3.0,
                            ),
                            AutomationSegment(
                                start=_pos(29),
                                end=_pos(33),
                                shape="hold",
                                value=-1.0,
                            ),
                        ),
                    ),
                ],
            )
        ],
    )

    def _place_lead(bar_start: int, phrase: list[_LeadNote]) -> None:
        for bar_off, beat, n16, partial, gate_16ths, amp_db in phrase:
            score.add_note(
                "lead",
                start=_pos(bar_start + bar_off, beat, n16),
                duration=gate_16ths * S16 * 0.88,
                partial=partial,
                amp_db=amp_db,
            )

    # ---------------------------------------------------------------
    # Chord: P6–P7–P9 (E4 / G4−31¢ / B4) — septimal E minor-ish.
    # Relative to the A bass, this spells: fifth / flat-seventh / ninth.
    # Very dark, slightly alien, consonant within the harmonic series.
    # ---------------------------------------------------------------
    # Phrase A: 4-bar — two chord hits with lots of space between.
    # First hit on the 'a' of beat 2 (off-beat), second in bar 3.
    # The dotted-8th delay + room reverb fill the two silent bars.
    # ---------------------------------------------------------------
    def _chord(
        bar_off: int,
        beat: int,
        n16: int,
        gate_16ths: float,
        amp_db: float,
    ) -> list[_LeadNote]:
        """Return three simultaneous notes for the E-minor-septimal chord."""
        return [
            (bar_off, beat, n16, P6, gate_16ths, amp_db),  # E4
            (bar_off, beat, n16, P7, gate_16ths, amp_db - 0.5),  # G4 −31¢
            (bar_off, beat, n16, P9, gate_16ths, amp_db - 1.0),  # B4
        ]

    PHRASE_A: list[_LeadNote] = [
        *_chord(0, 2, 3, 10, -8.0),  # bar 1, 'a'-of-2: chord, sustained
        *_chord(2, 3, 1, 8, -9.5),  # bar 3, &-of-3: echo, softer
    ]

    # ---------------------------------------------------------------
    # Phrase B: pre-drop dissolve — single chord hit, late and quiet.
    # ---------------------------------------------------------------
    PHRASE_B: list[_LeadNote] = [
        *_chord(1, 4, 2, 6, -11.0),  # bar 2, 'e'-of-4: chord fading out
    ]

    # ---------------------------------------------------------------
    # Phrase C: 1-bar skeleton (bars 29, 31).
    # One chord hit per bar, very quiet — nearly inaudible, just air.
    # ---------------------------------------------------------------
    PHRASE_C: list[_LeadNote] = [
        *_chord(0, 1, 2, 6, -11.5),  # 'e' of beat 1
    ]

    # bars 5–12: phrase A × 2
    for b in [5, 9]:
        _place_lead(b, PHRASE_A)
    # bars 13–16: phrase B × 1 (pre-drop dissolve)
    _place_lead(13, PHRASE_B)
    # bars 17–20: drop — no lead
    # bars 21–28: phrase A × 2
    for b in [21, 25]:
        _place_lead(b, PHRASE_A)
    # bars 29–32: phrase C on bars 29 and 31 only — silence on 30 and 32
    for b in [29, 31]:
        _place_lead(b, PHRASE_C)
    # bars 33–36: outro — no lead

    # ------------------------------------------------------------------
    # Hats: CHH preset, 16th notes, bars 3–32
    # Beat loudest, & medium, e/a soft ghosts; very short gate
    # ------------------------------------------------------------------
    score.add_voice(
        "hat",
        synth_defaults={"engine": "noise_perc", "preset": "chh"},
        mix_db=-11.0,
        velocity_humanize=None,
        effects=[
            # High shelf for added air and crispness
            EffectSpec(
                "eq",
                {"bands": [{"kind": "high_shelf", "freq_hz": 8000.0, "gain_db": 3.5}]},
            ),
            # Light saturation — subtle harmonic grit for a crisp, slightly dirty chh
            EffectSpec("saturation", {"drive": 0.40}),
            # 16th-note echo — adds shimmer and rhythmic motion without smearing
            EffectSpec("delay", {"delay_seconds": S16, "feedback": 0.33, "mix": 0.33}),
            EffectSpec(
                "compressor", {"preset": "kick_duck", "sidechain_source": "kick"}
            ),
        ],
        sends=[VoiceSend(target="room", send_db=-18.0)],
    )

    # 16ths: beat loudest, & medium, e and a soft ghosts.
    # Base amps per 16th subdivision (0=beat, 1=&, 2=e, 3=a):
    _hat_amps = {0: -11.0, 1: -16.0, 2: -13.5, 3: -16.5}
    # Section-based overall offset (dB) — intro quiet, builds through arrangement,
    # drops out for the drop section, swells back, then fades in the outro skeleton.
    _hat_section_offset: dict[int, float] = {
        **{b: -3.0 for b in range(3, 5)},  # bars 3–4: intro, pulling in
        **{b: 0.0 for b in range(5, 13)},  # bars 5–12: phrase A × 2, full
        **{b: -2.0 for b in range(13, 17)},  # bars 13–16: pre-drop, thinning out
        **{b: -1.5 for b in range(17, 21)},  # bars 17–20: drop, hats stay but quieter
        **{b: 0.0 for b in range(21, 29)},  # bars 21–28: phrase A returns, full
        **{b: -1.5 for b in range(29, 33)},  # bars 29–32: skeleton, settling
    }
    # Section-based frequency shapes hat brightness — lower = darker/more body,
    # higher = brighter/more aggressive. Bandpass center = freq × 1.0 (CHH preset).
    _hat_section_freq: dict[int, float] = {
        **{b: 10000.0 for b in range(3, 5)},  # intro: restrained, dark
        **{b: 13000.0 for b in range(5, 13)},  # phrase A × 2: standard bright
        **{b: 11500.0 for b in range(13, 17)},  # pre-drop: pulling back
        **{b: 9000.0 for b in range(17, 21)},  # drop: darkest, subdued
        **{b: 13500.0 for b in range(21, 29)},  # return: peak brightness
        **{b: 11000.0 for b in range(29, 33)},  # skeleton: settling
    }
    for bar in range(3, 33):
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

    # ------------------------------------------------------------------
    # Clap: pure noise, beats 2 and 4
    # Active in full arrangement sections; absent during intro, drop, and outro
    # ------------------------------------------------------------------
    score.add_voice(
        "clap",
        synth_defaults={"engine": "noise_perc", "preset": "clap"},
        mix_db=-6.5,
        normalize_peak_db=-6.0,  # percussive — LUFS normalization is unreliable here
        velocity_humanize=None,
        effects=[
            # Gate: peak at -6 dBFS (normalize_peak_db); noise RMS at onset
            # ≈ -9 to -12 dBFS (broadband noise crest factor ~4-6 dB).
            # Threshold -18 dBFS opens comfortably at hit; with noise_decay=40ms
            # the gate closes at ~40-45 ms, hold=30ms + release=12ms → fully
            # closed by ~85 ms — tight and punchy against a 120 ms note.
            # No kick_duck sidechain: kick hits every beat (1-2-3-4) which
            # would pump the clap reverb tail on off-beats, creating ghost rhythms.
            EffectSpec(
                "gate",
                {
                    "threshold_db": -18.0,
                    "attack_ms": 0.3,
                    "hold_ms": 30.0,
                    "release_ms": 12.0,
                },
            ),
            # Very subtle chorus just for stereo spread on the noise burst.
            EffectSpec("chorus", {"preset": "juno_subtle", "mix": 0.15}),
        ],
        sends=[VoiceSend(target="room", send_db=-15.0)],
    )

    # Section-based clap amp offset — louder in the full sections, quieter around
    # the drop to let the drop feel like a release of tension.
    _clap_section_offset: dict[int, float] = {
        **{b: -3.0 for b in range(3, 5)},  # bars 3–4: intro, quiet entry
        **{b: 0.0 for b in range(5, 13)},  # bars 5–12: full
        **{b: -2.5 for b in range(13, 17)},  # bars 13–16: pre-drop, fading
        **{b: 1.0 for b in range(21, 29)},  # bars 21–28: return, slightly louder
        **{b: -1.5 for b in range(29, 33)},  # bars 29–32: skeleton, quieter
        **{b: -3.0 for b in range(33, 37)},  # bars 33–36: outro, fading out
    }
    clap_bars = list(range(3, 17)) + list(range(21, 37))
    for bar in clap_bars:
        offset = _clap_section_offset.get(bar, 0.0)
        for beat in [2, 4]:
            # freq=3000 Hz → bandpass center at 3000 × 0.8 = 2400 Hz (1500–3300 Hz
            # range) — proper clap body range with the updated preset bandpass_ratio=0.8.
            score.add_note(
                "clap",
                start=_pos(bar, beat),
                duration=0.12,
                freq=3000.0,
                amp_db=-4.0 + offset,
            )

    return score


PIECES: dict[str, PieceDefinition] = {
    "spectral_kick": PieceDefinition(
        name="spectral_kick",
        output_name="spectral_kick",
        build_score=build_spectral_kick,
    ),
}
