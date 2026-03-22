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

from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Score

BPM: float = 130.0
BEAT: float = 60.0 / BPM  # quarter-note duration ≈ 0.4615 s
BAR: float = 4.0 * BEAT  # 4/4 bar ≈ 1.846 s
S16: float = BEAT / 4.0  # sixteenth-note ≈ 0.1154 s

F0: float = 55.0  # A1 — score root

# Harmonic partial constants (freq = F0 * partial)
P1: float = 1.0    # A1    55.0 Hz  bass root (sub)
P15: float = 1.5   # E2    82.5 Hz  bass fifth
P2: float = 2.0    # A2   110.0 Hz  bass octave
P35: float = 3.5   # —    192.5 Hz  septimal 7th colour (bass accent)
P4: float = 4.0    # A3   220.0 Hz  lead low anchor
P5: float = 5.0    # C#4  275.0 Hz  harmonic major third (available)
P6: float = 6.0    # E4   330.0 Hz  harmonic perfect fifth — open shadow
P7: float = 7.0    # G4   385.0 Hz  septimal 7th — spice only
P8: float = 8.0    # A4   440.0 Hz  lead home
P9: float = 9.0    # B4   495.0 Hz  whole step above home — spice only
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
        f0=F0,
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
                        {"kind": "highpass", "cutoff_hz": 120.0, "slope_db_per_oct": 12},
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
        mix_db=-2.0,
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
    # Bass: polyblep acid, JI-tuned partials
    # ------------------------------------------------------------------
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "polyblep",
            "preset": "moog_bass",
            "params": {
                # Square for odd-harmonic grit on top of the sub body;
                # cutoff kept low for subbiness, drive adds the distorted edge.
                "waveform": "square",
                "cutoff_hz": 210.0,
                "filter_env_amount": 0.6,
                "filter_drive": 0.45,
                "resonance": 0.10,
            },
        },
        mix_db=-5.0,
        velocity_humanize=None,
    )

    # Repeating pattern: all root. Beats 2 and 4 open for kick+snare.
    _bass_pattern: list[tuple[int, int, float, int, float]] = [
        (1, 0, P1, 1, -6.0),   # beat 1: root
        (1, 2, P1, 1, -9.0),   # &-of-1: root
        (1, 3, P1, 1, -11.0),  # a-of-1: ghost push
        # beat 2 open — kick+snare
        (2, 2, P1, 1, -8.0),   # &-of-2: root
        (3, 0, P1, 1, -6.5),   # beat 3: root
        (3, 2, P1, 1, -9.0),   # &-of-3: root
        (3, 3, P1, 1, -11.0),  # a-of-3: ghost push
        # beat 4 open — kick+snare
        (4, 2, P1, 1, -9.0),   # &-of-4: root
    ]

    for bar in range(3, total_bars + 1):
        for beat, n16, partial, gate_16ths, amp_db in _bass_pattern:
            score.add_note(
                "bass",
                start=_pos(bar, beat, n16),
                duration=gate_16ths * S16 * 0.82,
                partial=partial,
                amp_db=amp_db,
            )

    # Fifth (P15 = E2, 82.5 Hz): sparse and irregular — not every bar,
    # never the same beat position twice in a row.
    _fifth_hits: list[tuple[int, int, int]] = [
        (7,  4, 2),   # bar 7,  &-of-4
        (11, 2, 2),   # bar 11, &-of-2
        (15, 3, 3),   # bar 15, a-of-3
        (23, 4, 2),   # bar 23, &-of-4
        (27, 2, 2),   # bar 27, &-of-2
        (31, 1, 2),   # bar 31, &-of-1
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
                # every note's FM overtones reinforce its own harmonic series.
                "mod_ratio": 1.0,
                "mod_index": 2.0,
            },
            "env": {
                "attack_ms": 4.0,
                "decay_ms": 220.0,
                "sustain_ratio": 0.05,
                "release_ms": 180.0,
            },
        },
        mix_db=-13.0,
        velocity_humanize=None,
        effects=[
            # highpass for cleanliness — lowest lead note is P4 = 220 Hz
            EffectSpec("eq", {"bands": [{"kind": "highpass", "cutoff_hz": 160.0, "slope_db_per_oct": 12}]}),
            # dotted-8th delay (3 × S16 ≈ 0.346s) — bounces rhythmically against kick
            EffectSpec(
                "delay",
                {
                    "delay_seconds": 3.0 * S16,
                    "feedback": 0.39,
                    "mix": 0.36,
                },
            ),
            # light algorithmic reverb for air
            EffectSpec(
                "reverb",
                {
                    "room_size": 0.38,
                    "damping": 0.60,
                    "wet_level": 0.30,
                },
            ),
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
    # Phrase A: 4-bar arc — low anchor → octave leap → open fifth.
    # Three notes. The delay does the rest.
    # ---------------------------------------------------------------
    PHRASE_A: list[_LeadNote] = [
        # bar 1 — deep low bell; hold and let FM + delay fill
        (0, 1, 0, P4, 8, -7.5),   # beat 1: low anchor (A3), half note
        # bar 2 — silence, then octave leap
        (1, 3, 0, P8, 4, -8.0),   # beat 3: home (A4), quarter
        # bars 3–4 — open fifth, held long into the silence
        (2, 1, 0, P6, 12, -8.5),  # beat 1: fifth (E4), dotted half — breathe out
    ]

    # ---------------------------------------------------------------
    # Phrase B: 4-bar pre-drop dissolve.
    # Same pitches, everything shifted late and shorter — hollowing out.
    # ---------------------------------------------------------------
    PHRASE_B: list[_LeadNote] = [
        (0, 1, 0, P4, 6, -7.5),   # bar 1 beat 1: low anchor, shorter
        (1, 4, 0, P8, 2, -8.5),   # bar 2 beat 4: late, hesitant 8th
        (3, 2, 0, P6, 8, -9.5),   # bar 4 beat 2: fifth arrives late — hollow
    ]

    # ---------------------------------------------------------------
    # Phrase C: 1-bar skeleton (bars 29–32).
    # Compressed: octave touch → fifth falls away.
    # ---------------------------------------------------------------
    PHRASE_C: list[_LeadNote] = [
        (0, 1, 0, P8, 2, -9.0),   # beat 1: home (A4), 8th
        (0, 4, 0, P6, 4, -10.5),  # beat 4: fifth, falling away
    ]

    # bars 5–12: phrase A × 2 (4-bar phrase, every 4 bars)
    for b in [5, 9]:
        _place_lead(b, PHRASE_A)
    # bars 13–16: phrase B × 1 (pre-drop dissolve)
    _place_lead(13, PHRASE_B)
    # bars 17–20: drop — no lead
    # bars 21–28: phrase A × 2
    for b in [21, 25]:
        _place_lead(b, PHRASE_A)
    # bars 29–32: phrase C × 4, one per bar (skeleton)
    for b in [29, 30, 31, 32]:
        _place_lead(b, PHRASE_C)
    # bars 33–36: outro — no lead

    # ------------------------------------------------------------------
    # Hats: CHH preset, 16th notes, bars 3–32
    # Beat loudest, & medium, e/a soft ghosts; very short gate
    # ------------------------------------------------------------------
    score.add_voice(
        "hat",
        synth_defaults={"engine": "noise_perc", "preset": "chh"},
        mix_db=-14.0,
        velocity_humanize=None,
        effects=[
            # 16th-note echo — adds shimmer and rhythmic motion without smearing
            EffectSpec("delay", {"delay_seconds": S16, "feedback": 0.12, "mix": 0.15}),
        ],
    )

    # 16ths: beat loudest, & medium, e and a soft ghosts
    _hat_amps = {0: -11.0, 1: -16.0, 2: -13.5, 3: -16.5}
    for bar in range(3, 33):
        for beat in range(1, 5):
            for n16 in range(4):
                score.add_note("hat", start=_pos(bar, beat, n16), duration=0.04, freq=13000.0, amp_db=_hat_amps[n16])

    # ------------------------------------------------------------------
    # Snare/clap: snareish preset, beats 2 and 4
    # Active in full arrangement sections; absent during intro, drop, and outro
    # ------------------------------------------------------------------
    score.add_voice(
        "snare",
        synth_defaults={"engine": "noise_perc", "preset": "snareish"},
        mix_db=-6.0,
        velocity_humanize=None,
    )

    snare_bars = list(range(5, 17)) + list(range(21, 33))
    for bar in snare_bars:
        for beat in [2, 4]:
            # freq=800 → bandpass center at 1280 Hz, well above the bass register
            score.add_note("snare", start=_pos(bar, beat), duration=0.18, freq=200.0, amp_db=-4.0)

    return score


PIECES: dict[str, PieceDefinition] = {
    "spectral_kick": PieceDefinition(
        name="spectral_kick",
        output_name="spectral_kick",
        build_score=build_spectral_kick,
    ),
}
