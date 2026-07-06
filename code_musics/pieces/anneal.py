"""anneal — Colundi stretch-drift piece and its audition studies.

Sethares co-design: every tonal voice's partials are drawn from the scale's
own degrees, so the scale's intervals are consonant by construction and the
Act II pseudo-octave stretch warps scale and spectrum together.

Design spec: docs/plans/2026-07-05-anneal-design.md
Plan:        docs/plans/2026-07-05-anneal-plan.md

`anneal_fusion_sketch` is audition study 1: a chord ladder proving
(a) skeleton-spectrum fusion, (b) chord-role-aware color-partial fusion, and
(c) that a matched stretch at P=2.07 reads as "warped world", not "out of
tune". Four chords at home tuning, then a continuous stretch ramp on the
tonic (the piece's Act II gesture in miniature) with a slow anneal home.
"""

from __future__ import annotations

from code_musics.automation import (
    AutomationSegment,
    AutomationSpec,
    AutomationTarget,
)
from code_musics.drum_helpers import add_drum_voice, setup_drum_bus
from code_musics.humanize import EnvelopeHumanizeSpec, TimingHumanizeSpec
from code_musics.meter import Groove
from code_musics.pieces._shared import (
    DEFAULT_MASTER_EFFECTS,
    SOFT_REVERB_EFFECT,
    bricasti_or_reverb,
)
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import EffectSpec, Score, SendBusSpec, VoiceSend
from code_musics.spectra import scale_fused_spectrum
from code_musics.tuning import stretch_ratio

F0 = 98.0  # G2; kick fundamental (later tasks) at 49 Hz

HOME_PSEUDO_OCTAVE = 2.0
PEAK_PSEUDO_OCTAVE = 2.07

# 3/7-limit skeleton degrees -> integer partials {1,2,3,3.5,4,6,7,8,12,14}.
# Deliberately NO 5th/10th harmonic: the scale's third is 19/16, and a 5/4
# partial would beat against every third in the piece.
SKELETON_DEGREES = [1.0, 3 / 2, 7 / 4]
SKELETON_PARTIALS = scale_fused_spectrum(SKELETON_DEGREES, octaves=3)

# Color partials are CHORD-ROLE-AWARE: a note carries a color partial only
# when it is an exact octave-multiple of a chord-internal interval, so it
# lands on another chord tone's partial. Giving every note the same color
# list creates comma collisions instead of fusion (audition 1 finding: the
# fifth's 19/8 partial at 3/2*19/8 = 3.5625 beat against the root's 7/4
# partial at 3.5 — a 57/56 clash, ~31 cents / ~6 Hz of roughness).
COLOR_19_RATIOS = [19 / 8, 19 / 4]  # on the ROOT of 16:19:24 -> third's partials
COLOR_49_RATIOS = [49 / 40 * 2, 49 / 40 * 4]  # 2.45, 4.9: on the lower two
# notes of the neutral triad -> the 49/30-note's partials (plus a 0.7-cent
# slow shimmer against the octave note).

# Chords as (degree, color_ratios) pairs; 2.0 degrees stretch to the
# pseudo-octave like everything else.
TONIC_4_6_7: list[tuple[float, list[float]]] = [
    (1.0, []),
    (3 / 2, []),
    (7 / 4, []),
]
MINOR_16_19_24: list[tuple[float, list[float]]] = [
    (1.0, COLOR_19_RATIOS),
    (19 / 16, []),
    (3 / 2, []),
]
NEUTRAL_SUBDOMINANT: list[tuple[float, list[float]]] = [
    (4 / 3, COLOR_49_RATIOS),
    (49 / 30, COLOR_49_RATIOS),
    (2.0, []),
]


def _with_color(
    base: list[dict[str, float]], color_ratios: list[float], weight: float = 0.4
) -> list[dict[str, float]]:
    """Blend color-degree partials into a fused spectrum at reduced weight."""
    if not color_ratios:
        return base
    extra = [{"ratio": ratio, "amp": weight / ratio} for ratio in color_ratios]
    return sorted(base + extra, key=lambda partial: partial["ratio"])


def _smoothstep(x: float) -> float:
    clamped = min(max(x, 0.0), 1.0)
    return 3 * clamped**2 - 2 * clamped**3


def _stretched_partials(
    partials: list[dict[str, float]], pseudo_octave: float
) -> list[dict[str, float]]:
    """Map a partial list through the stretch law (identity at 2.0)."""
    return [
        {"ratio": stretch_ratio(partial["ratio"], pseudo_octave), "amp": partial["amp"]}
        for partial in partials
    ]


def _fusion_sketch_score() -> Score:
    score = Score(
        f0_hz=F0,
        master_effects=DEFAULT_MASTER_EFFECTS,
    )
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "additive",
            "attack": 0.8,
            "release": 1.6,
            "decay_power": 2.0,
            "spectral_morph_type": "phase_disperse",
            "spectral_morph_amount": 0.35,
        },
        effects=[SOFT_REVERB_EFFECT],
        pan=0.0,
    )

    def add_chord(
        chord: list[tuple[float, list[float]]],
        start: float,
        duration: float,
        pseudo_octave: float,
    ) -> None:
        for note_index, (degree, color_ratios) in enumerate(chord):
            partials = _stretched_partials(
                _with_color(SKELETON_PARTIALS, color_ratios), pseudo_octave
            )
            score.add_note(
                "pad",
                start=start,
                duration=duration,
                partial=stretch_ratio(degree, pseudo_octave),
                amp_db=-15.0 if note_index == 0 else -18.0,
                synth={"partials": partials},
            )

    # Chord ladder at home tuning (role-aware color partials).
    add_chord(TONIC_4_6_7, 0.0, 5.0, HOME_PSEUDO_OCTAVE)
    add_chord(MINOR_16_19_24, 6.0, 5.0, HOME_PSEUDO_OCTAVE)
    add_chord(NEUTRAL_SUBDOMINANT, 12.0, 5.0, HOME_PSEUDO_OCTAVE)
    add_chord(TONIC_4_6_7, 18.0, 5.0, HOME_PSEUDO_OCTAVE)

    # Continuous stretch ramp — the piece's Act II gesture in miniature.
    # Tonic re-struck every 3 s while S climbs 2.00 -> 2.07 (smoothstep).
    stretch_span = PEAK_PSEUDO_OCTAVE - HOME_PSEUDO_OCTAVE
    for strike in range(6):
        onset = 24.0 + strike * 3.0
        pseudo_octave = HOME_PSEUDO_OCTAVE + stretch_span * _smoothstep(
            (onset - 24.0) / 18.0
        )
        add_chord(TONIC_4_6_7, onset, 2.8, pseudo_octave)

    # Hold at full stretch.
    add_chord(TONIC_4_6_7, 42.0, 5.0, PEAK_PSEUDO_OCTAVE)

    # Anneal home (ease-out, slower feel than the climb).
    for strike in range(3):
        onset = 48.0 + strike * 3.0
        pseudo_octave = HOME_PSEUDO_OCTAVE + stretch_span * (
            1.0 - _smoothstep((onset - 48.0) / 12.0)
        )
        add_chord(TONIC_4_6_7, onset, 2.8, pseudo_octave)
    add_chord(TONIC_4_6_7, 57.0, 6.0, HOME_PSEUDO_OCTAVE)
    return score


# ---------------------------------------------------------------------------
# Palette sketch (audition study 2): all voice roles at home tuning over a
# simple I -> IV(neutral) -> V(6:7:8) -> I pass, ~52 s at 110 BPM.
# The voice-construction helpers below are reused verbatim by the full piece.
# ---------------------------------------------------------------------------

BPM = 110.0
BEAT = 60.0 / BPM
BAR = 4.0 * BEAT
S16 = BEAT / 4.0

GROOVE = Groove.sixteenths_swing()

# V chord: 6:7:8 septimal subminor on the fifth (all scale members).
FIFTH_6_7_8: list[tuple[float, list[float]]] = [
    (3 / 2, []),
    (7 / 4, []),
    (2.0, []),
]

# Workhorse IV: open fifth on 4/3 — every tone pure. The neutral triad is
# color, not a resting consonance (audition 2 finding), so it appears only
# as a brief pad-only moment with the arp staying on open tones above it.
FOURTH_OPEN: list[tuple[float, list[float]]] = [
    (4 / 3, []),
    (2.0, []),
    (8 / 3, []),
]

# (start_bar, n_bars, pad_chord, arp_chord) — bars 0-1 are a pad/atmosphere
# intro; arp_chord lets the pad take a color voicing while the arp stays pure.
_PALETTE_PROGRESSION: list[
    tuple[int, int, list[tuple[float, list[float]]], list[tuple[float, list[float]]]]
] = [
    (0, 2, TONIC_4_6_7, TONIC_4_6_7),
    (2, 6, TONIC_4_6_7, TONIC_4_6_7),
    (8, 4, FOURTH_OPEN, FOURTH_OPEN),
    (12, 2, FIFTH_6_7_8, FIFTH_6_7_8),
    (14, 4, TONIC_4_6_7, TONIC_4_6_7),
    (18, 2, NEUTRAL_SUBDOMINANT, FOURTH_OPEN),
    (20, 2, FIFTH_6_7_8, FIFTH_6_7_8),
    (22, 2, TONIC_4_6_7, TONIC_4_6_7),
]
_PALETTE_BARS = 24


def _bell_partials(
    pseudo_octave: float = HOME_PSEUDO_OCTAVE,
) -> list[dict[str, object]]:
    """Fused skeleton with mallet physics: higher partials decay faster."""
    bell: list[dict[str, object]] = []
    for partial in SKELETON_PARTIALS:
        ratio = stretch_ratio(partial["ratio"], pseudo_octave)
        # Envelope times are normalized to note duration; higher partials
        # reach their floor sooner = mallet/bell physics.
        decay_frac = min(1.0, 0.9 / ratio**0.5)
        bell.append(
            {
                "ratio": ratio,
                "amp": partial["amp"],
                "envelope": [
                    {"time": 0.0, "value": 1.0},
                    {"time": decay_frac, "value": 0.08},
                ],
            }
        )
    return bell


def _hall_bus() -> SendBusSpec:
    return SendBusSpec(
        name="hall",
        effects=[
            bricasti_or_reverb(
                "1 Halls 07 Large & Dark",
                1.0,
                room_size=0.78,
                damping=0.55,
            )
        ],
        return_db=0.0,
    )


def _kick_duck(threshold_db: float, ratio: float) -> EffectSpec:
    return EffectSpec(
        "compressor",
        {
            "threshold_db": threshold_db,
            "ratio": ratio,
            "attack_ms": 3.0,
            "release_ms": 160.0,
            "lookahead_ms": 5.0,
            "sidechain_source": "kick",
            "detector_mode": "peak",
        },
    )


def _add_atmosphere(score: Score, total_dur: float) -> None:
    """The unstretched found-sound room: empty building + distant city."""
    for name, preset, mix_db in [
        ("room_tone", "found_empty_room", -16.0),
        ("city", "found_city_at_night", -20.0),
    ]:
        score.add_voice(
            name,
            synth_defaults={"engine": "synth_voice", "preset": preset},
            pan=0.0 if name == "room_tone" else -0.2,
            mix_db=mix_db,
            velocity_humanize=None,
        )
        score.add_note(name, start=0.0, duration=total_dur, partial=1.0, amp_db=-6.0)


def _add_bass(score: Score) -> None:
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "synth_voice",
            "partials_type": "additive",
            "partials_partials": [
                {"ratio": 1.0, "amp": 1.0},
                {"ratio": 2.0, "amp": 0.45},
                {"ratio": 3.0, "amp": 0.22},
                {"ratio": 7.0, "amp": 0.05},
            ],
            "filter_topology": "ladder",
            "cutoff_hz": 420.0,
            "resonance_q": 0.9,
            "attack": 0.012,
            "release": 0.3,
        },
        effects=[_kick_duck(-21.0, 4.0)],
        pan=0.0,
        mix_db=-6.5,
        max_polyphony=1,
        legato=True,
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        drift_bus="kiln_drift",
        drift_bus_correlation=0.9,
    )


_BASS_PARTIAL_RATIOS = [(1.0, 1.0), (2.0, 0.45), (3.0, 0.22), (7.0, 0.05)]


def _bass_partials(pseudo_octave: float) -> list[dict[str, float]]:
    return [
        {"ratio": stretch_ratio(ratio, pseudo_octave), "amp": amp}
        for ratio, amp in _BASS_PARTIAL_RATIOS
    ]


def _place_bass(
    score: Score,
    start_bar: int,
    n_bars: int,
    root_degree: float,
    pseudo_octave: float = HOME_PSEUDO_OCTAVE,
    fifth_pickup: bool = False,
) -> None:
    octave_down = 1.0 / stretch_ratio(2.0, pseudo_octave)
    synth = {"partials_partials": _bass_partials(pseudo_octave)}
    for bar in range(start_bar, start_bar + n_bars):
        bar_t = bar * BAR
        low_root = stretch_ratio(root_degree, pseudo_octave) * octave_down
        score.add_note(
            "bass",
            start=bar_t,
            duration=1.5 * BEAT,
            partial=low_root,
            velocity=0.95,
            synth=synth,
        )
        score.add_note(
            "bass",
            start=bar_t + 2.5 * BEAT,
            duration=0.6 * BEAT,
            partial=low_root,
            velocity=0.7,
            synth=synth,
        )
        if fifth_pickup and bar % 4 == 3:
            score.add_note(
                "bass",
                start=bar_t + 3.5 * BEAT,
                duration=0.45 * BEAT,
                partial=stretch_ratio(root_degree * 3 / 2, pseudo_octave) * octave_down,
                velocity=0.6,
                synth=synth,
            )


def _add_pad(score: Score) -> None:
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "additive",
            "attack": 0.9,
            "release": 1.8,
            "decay_power": 2.0,
            "spectral_morph_type": "phase_disperse",
            "spectral_morph_amount": 0.3,
        },
        effects=[_kick_duck(-26.0, 1.8)],
        sends=[VoiceSend(target="hall", send_db=-10.0)],
        pan=0.0,
        mix_db=-4.0,
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        drift_bus="kiln_drift",
        drift_bus_correlation=0.85,
    )


def _place_pad(
    score: Score,
    start_bar: int,
    n_bars: int,
    chord: list[tuple[float, list[float]]],
    pseudo_octave: float = HOME_PSEUDO_OCTAVE,
) -> None:
    start = start_bar * BAR
    duration = n_bars * BAR
    for note_index, (degree, color_ratios) in enumerate(chord):
        partials = _stretched_partials(
            _with_color(SKELETON_PARTIALS, color_ratios), pseudo_octave
        )
        score.add_note(
            "pad",
            start=start,
            duration=duration,
            partial=stretch_ratio(degree, pseudo_octave),
            amp_db=-15.0 if note_index == 0 else -18.0,
            synth={"partials": partials},
        )


def _add_bell_arp(score: Score) -> None:
    score.add_voice(
        "arp",
        synth_defaults={
            "engine": "additive",
            "partials": _bell_partials(),
            "attack": 0.004,
            "release": 0.28,
            "decay_power": 2.0,
        },
        sends=[VoiceSend(target="hall", send_db=-12.0)],
        pan=0.12,
        mix_db=-8.0,
        drift_bus="kiln_drift",
        drift_bus_correlation=0.7,
    )


# 2-bar arp cycle: (16th step, chord-tone index, velocity). Tone indices walk
# chord tones one and two octaves up; step 29 is a deliberate syncopation.
_ARP_CYCLE: list[tuple[int, int, float]] = [
    (0, 0, 1.0),
    (2, 1, 0.7),
    (4, 2, 0.8),
    (6, 3, 0.65),
    (8, 2, 0.75),
    (10, 1, 0.6),
    (12, 0, 0.85),
    (14, 2, 0.6),
    (16, 3, 0.9),
    (18, 4, 0.7),
    (20, 2, 0.75),
    (22, 1, 0.6),
    (24, 0, 0.8),
    (26, 2, 0.65),
    (29, 4, 0.7),
    (30, 3, 0.6),
]


def _arp_tones(
    chord: list[tuple[float, list[float]]], pseudo_octave: float
) -> list[float]:
    """Chord tones one and two (pseudo-)octaves up, plus the reserved 11/10
    inflection (index 5, used only by the Act II climax pattern)."""
    degrees = [degree for degree, _ in chord]
    octave_up = stretch_ratio(2.0, pseudo_octave)
    return [
        stretch_ratio(degrees[0], pseudo_octave) * octave_up,
        stretch_ratio(degrees[1], pseudo_octave) * octave_up,
        stretch_ratio(degrees[2], pseudo_octave) * octave_up,
        stretch_ratio(degrees[0], pseudo_octave) * octave_up**2,
        stretch_ratio(degrees[1], pseudo_octave) * octave_up**2,
        stretch_ratio(degrees[0] * 11 / 10, pseudo_octave) * octave_up,
    ]


def _place_arp(
    score: Score,
    start_bar: int,
    n_bars: int,
    chord: list[tuple[float, list[float]]],
    pseudo_octave: float = HOME_PSEUDO_OCTAVE,
    pattern: list[tuple[int, int, float]]
    | list[tuple[int, int, float, float]]
    | None = None,
    duration_default: float = 0.11,
    velocity_scale: float = 1.0,
    vibrato: bool = False,
) -> None:
    cycle = pattern if pattern is not None else _ARP_CYCLE
    tones = _arp_tones(chord, pseudo_octave)
    steps_per_cycle = 32
    total_steps = n_bars * 16
    for cycle_start in range(0, total_steps, steps_per_cycle):
        for entry in cycle:
            step, tone_index, velocity = entry[0], entry[1], entry[2]
            duration = entry[3] if len(entry) > 3 else duration_default
            absolute_step = cycle_start + step
            if absolute_step >= total_steps:
                break
            onset = (
                start_bar * BAR
                + absolute_step * S16
                + S16 * GROOVE.timing_offset_at(absolute_step)
            )
            pitch_motion = (
                PitchMotionSpec.vibrato(depth_ratio=0.006, rate_hz=5.2)
                if vibrato and duration > 0.3
                else None
            )
            score.add_note(
                "arp",
                start=onset,
                duration=duration,
                partial=tones[tone_index],
                velocity=min(1.0, velocity * velocity_scale),
                pitch_motion=pitch_motion,
                synth={"partials": _bell_partials(pseudo_octave)},
            )


def _add_drums(score: Score, drum_bus: str) -> None:
    add_drum_voice(
        score,
        "kick",
        engine="drum_voice",
        preset="909_techno",
        drum_bus=drum_bus,
        send_db=-6.0,
        synth_overrides={
            "tone_decay_s": 0.22,
            # Skin, not snare (auditions 2-3): the preset's bandpassed noise
            # layer is zeroed outright — a ~20 ms midband noise burst is the
            # acoustic signature of a snare — and the beater knock is short
            # and hard with a steep pitch sweep for the punch.
            "noise_level": 0.0,
            "exciter_level": 0.14,
            "exciter_beater_chirp_s": 0.003,
            "tone_punch": 0.75,
            "tone_sweep_ratio": 3.1,
            "tone_sweep_decay_s": 0.03,
        },
        mix_db=-2.0,
    )
    add_drum_voice(
        score,
        "hat_closed",
        engine="drum_voice",
        preset="closed_hat",
        drum_bus=drum_bus,
        send_db=-16.0,
        choke_group="hats",
        mix_db=-13.0,
        pan=0.18,
    )
    add_drum_voice(
        score,
        "hat_open",
        engine="drum_voice",
        preset="open_hat",
        drum_bus=drum_bus,
        send_db=-14.0,
        choke_group="hats",
        mix_db=-16.0,
        pan=0.18,
    )
    add_drum_voice(
        score,
        "tom",
        engine="drum_voice",
        drum_bus=drum_bus,
        send_db=-10.0,
        mix_db=-12.0,
        pan=-0.25,
        synth_overrides={
            "exciter_type": "click",
            "exciter_level": 0.08,
            "tone_type": "modal",
            "tone_level": 1.0,
            # Pure skeleton modes only. Color-family modes ring color
            # intervals over whatever chord is playing — the "out of tune
            # bleep" of audition 2. Role-awareness applies to drums too.
            "modal_ratios": [1.0, 2.0, 3.0, 3.5],
            "modal_decays_s": [0.5, 0.34, 0.26, 0.2],
        },
    )


_KICK_PATTERNS: dict[str, tuple[list[tuple[int, float]], list[tuple[int, float]]]] = {
    # (even bar, odd bar) 16th-step patterns.
    "main": ([(0, 1.0), (7, 0.85), (10, 0.9)], [(0, 1.0), (7, 0.85), (13, 0.75)]),
    "dense": (
        [(0, 1.0), (4, 0.7), (7, 0.85), (10, 0.9), (13, 0.6)],
        [(0, 1.0), (7, 0.85), (10, 0.9), (14, 0.7)],
    ),
    "sparse": ([(0, 1.0), (10, 0.8)], [(0, 0.95), (7, 0.7)]),
}


def _place_drums(
    score: Score,
    start_bar: int,
    n_bars: int,
    root_degree: float = 1.0,
    pseudo_octave: float = HOME_PSEUDO_OCTAVE,
    kick_pattern: str = "main",
    kick_on: bool = True,
    hats_on: bool = True,
    open_hat_on: bool = True,
    toms_on: bool = True,
    fill_bars: bool = False,
) -> None:
    kick_even, kick_odd = _KICK_PATTERNS[kick_pattern]
    tom_synth = {
        "modal_ratios": [
            stretch_ratio(ratio, pseudo_octave) for ratio in [1.0, 2.0, 3.0, 3.5]
        ]
    }
    octave_down = 1.0 / stretch_ratio(2.0, pseudo_octave)
    for bar in range(start_bar, start_bar + n_bars):
        bar_t = bar * BAR
        kick_steps = kick_even if bar % 2 == 0 else kick_odd
        if not kick_on:
            kick_steps = []
        # The kick stays anchored on the tonic (49 Hz at home tuning) and only
        # moves with the world's stretch, never with the chord root.
        kick_freq = F0 * octave_down
        for step, velocity in kick_steps:
            score.add_note(
                "kick",
                start=bar_t + step * S16,
                duration=0.3,
                freq=kick_freq,
                velocity=velocity,
            )
        if hats_on:
            for step in range(0, 16, 2):
                if bar % 4 == 2 and step == 8:
                    continue  # breath every fourth bar
                onset = bar_t + step * S16 + S16 * GROOVE.timing_offset_at(step)
                score.add_note(
                    "hat_closed",
                    start=onset,
                    duration=0.09,
                    freq=784.0,
                    velocity=0.85 if step % 4 == 0 else 0.5,
                )
        if open_hat_on and bar % 2 == 1:
            onset = bar_t + 14 * S16 + S16 * GROOVE.timing_offset_at(14)
            score.add_note(
                "hat_open", start=onset, duration=0.4, freq=784.0, velocity=0.7
            )
        tom_freq = F0 * stretch_ratio(root_degree, pseudo_octave)
        if fill_bars and bar % 8 == 7:
            for step, velocity, ratio in [
                (8, 0.55, 1.0),
                (10, 0.65, 3 / 4),
                (11, 0.75, 1.0),
                (14, 0.9, 3 / 2),
            ]:
                score.add_note(
                    "tom",
                    start=bar_t + step * S16,
                    duration=0.4,
                    freq=tom_freq * stretch_ratio(ratio, pseudo_octave),
                    velocity=velocity,
                    synth=tom_synth,
                )
        elif toms_on and bar % 4 == 3:
            score.add_note(
                "tom",
                start=bar_t + 11 * S16,
                duration=0.5,
                freq=tom_freq,
                velocity=0.7,
                synth=tom_synth,
            )
            score.add_note(
                "tom",
                start=bar_t + 14 * S16,
                duration=0.5,
                freq=tom_freq * 3 / 4,
                velocity=0.5,
                synth=tom_synth,
            )


def _palette_sketch_score() -> Score:
    score = Score(
        f0_hz=F0,
        master_effects=DEFAULT_MASTER_EFFECTS,
        send_buses=[_hall_bus()],
        timing_humanize=TimingHumanizeSpec(preset="tight_ensemble"),
    )
    score.add_drift_bus("kiln_drift", rate_hz=0.06, depth_cents=3.5, seed=20260705)
    drum_bus = setup_drum_bus(score, style="light")

    total_dur = _PALETTE_BARS * BAR + 3.0
    _add_atmosphere(score, total_dur)
    _add_pad(score)
    _add_bass(score)
    _add_bell_arp(score)
    _add_drums(score, drum_bus)

    for start_bar, n_bars, pad_chord, arp_chord in _PALETTE_PROGRESSION:
        _place_pad(score, start_bar, n_bars, pad_chord)
        if start_bar >= 2:
            root_degree = arp_chord[0][0]
            _place_bass(score, start_bar, n_bars, root_degree)
            _place_arp(score, start_bar, n_bars, arp_chord)
            _place_drums(score, start_bar, n_bars, root_degree)
    return score


# ---------------------------------------------------------------------------
# The full piece: 176 bars (~6:24 + tail) in three acts.
#   Act I  "cool"   bars   0-55   S = 2.000
#   Act II "heat"   bars  56-123  S ramps to 2.07 (peak from bar 114)
#   Act III "anneal" bars 124-176 S eases home slower than it rose
# ---------------------------------------------------------------------------

_PIECE_BARS = 176
_RAMP_START_BAR = 56.0
_PEAK_BAR = 114.0
_COOL_START_BAR = 124.0
_HOME_AGAIN_BAR = 160.0
_DRUM_STRETCH_LEAD_BARS = 4.0  # percussion warps ~8 s early


def _pseudo_octave_for_bar(bar: float) -> float:
    """The master S(t) curve, in bar time."""
    span = PEAK_PSEUDO_OCTAVE - HOME_PSEUDO_OCTAVE
    if bar < _RAMP_START_BAR:
        return HOME_PSEUDO_OCTAVE
    if bar < _PEAK_BAR:
        return HOME_PSEUDO_OCTAVE + span * _smoothstep(
            (bar - _RAMP_START_BAR) / (_PEAK_BAR - _RAMP_START_BAR)
        )
    if bar < _COOL_START_BAR:
        return PEAK_PSEUDO_OCTAVE
    if bar < _HOME_AGAIN_BAR:
        # Ease-in cooling: starts slow (annealing), reaches home by bar 160.
        x = (bar - _COOL_START_BAR) / (_HOME_AGAIN_BAR - _COOL_START_BAR)
        return PEAK_PSEUDO_OCTAVE - span * x**2
    return HOME_PSEUDO_OCTAVE


def _drum_pseudo_octave_for_bar(bar: float) -> float:
    return _pseudo_octave_for_bar(bar + _DRUM_STRETCH_LEAD_BARS)


# Sparse legato arp for the dark Act II turn and the Act III cooling; long
# notes carry vibrato. Entries: (step, tone index, velocity, duration s).
_ARP_CYCLE_B: list[tuple[int, int, float, float]] = [
    (0, 0, 0.9, 0.5),
    (4, 2, 0.7, 0.4),
    (8, 1, 0.75, 0.5),
    (14, 3, 0.6, 0.3),
    (16, 2, 0.85, 0.5),
    (22, 4, 0.65, 0.4),
    (24, 1, 0.7, 0.5),
    (28, 2, 0.6, 0.35),
]

# Dense climax pattern; tone index 5 is the reserved 11/10 melodic inflection.
_ARP_CYCLE_C: list[tuple[int, int, float]] = [
    (0, 0, 1.0),
    (2, 1, 0.75),
    (3, 5, 0.55),
    (4, 2, 0.85),
    (6, 3, 0.7),
    (8, 2, 0.8),
    (9, 5, 0.5),
    (10, 1, 0.7),
    (12, 4, 0.9),
    (14, 3, 0.65),
    (16, 0, 0.95),
    (18, 2, 0.7),
    (20, 5, 0.6),
    (21, 4, 0.75),
    (22, 3, 0.7),
    (24, 2, 0.85),
    (26, 1, 0.65),
    (28, 0, 0.8),
    (29, 2, 0.6),
    (30, 4, 0.7),
]

# The Act II climax spice chord: the piece's single structural 11/10 sonority.
# Role-aware fusion: the root carries the 2.2/4.4 partials that land exactly
# on the 11/10 note's partial ladder.
SPICE_CHORD: list[tuple[float, list[float]]] = [
    (1.0, [11 / 5, 22 / 5]),
    (11 / 10, []),
    (3 / 2, []),
    (7 / 4, []),
]

# 16:19:24 as the Act II darkened tonic keeps its role-aware 19-color.
_ELEVEN_CARRIER_PARTIALS = [
    (1.0, 0.9),
    (11 / 5, 0.55),
    (2.0, 0.3),
    (22 / 5, 0.3),
    (3.0, 0.15),
    (44 / 5, 0.12),
]


def _eleven_partials(pseudo_octave: float) -> list[dict[str, float]]:
    return [
        {"ratio": stretch_ratio(ratio, pseudo_octave), "amp": amp}
        for ratio, amp in _ELEVEN_CARRIER_PARTIALS
    ]


def _echo_bus() -> SendBusSpec:
    """Dotted-eighth delay for the arp (wet-only return)."""
    return SendBusSpec(
        name="echo",
        effects=[
            EffectSpec(
                "delay",
                {"delay_seconds": 3.0 * S16, "feedback": 0.38, "mix": 1.0},
            )
        ],
        return_db=0.0,
    )


def _bar_pos(bar: float) -> float:
    return bar * BAR


def _ramp_spec(
    target: AutomationTarget,
    points: list[tuple[float, float, float, str]],
    default: float,
) -> AutomationSpec:
    """Replace-mode automation from (start_bar, end_bar, to_value, shape)."""
    segments: list[AutomationSegment] = []
    prev_value = default
    prev_end = 0.0
    for start_bar, end_bar, to_value, shape in points:
        start, end = _bar_pos(start_bar), _bar_pos(end_bar)
        if start > prev_end and segments:
            segments.append(
                AutomationSegment(
                    start=prev_end, end=start, shape="hold", value=prev_value
                )
            )
        segments.append(
            AutomationSegment(
                start=start,
                end=end,
                shape=shape,  # type: ignore[arg-type]
                start_value=prev_value,
                end_value=to_value,
            )
        )
        prev_value = to_value
        prev_end = end
    return AutomationSpec(
        target=target, segments=tuple(segments), default_value=default
    )


def _arp_echo_ride() -> AutomationSpec:
    """The arp's delay send: absent, blooming through Act II, gone by the end."""
    return _ramp_spec(
        AutomationTarget(kind="control", name="send_db"),
        [
            (40.0, 56.0, -16.0, "linear"),
            (88.0, 112.0, -10.0, "linear"),
            (124.0, 152.0, -20.0, "linear"),
            (152.0, 164.0, -34.0, "linear"),
        ],
        default=-34.0,
    )


def _bass_cutoff_arc() -> AutomationSpec:
    """Dark start, open through the build, subs-only in the dark turn,
    widest at the climax, settling home for Act III."""
    return _ramp_spec(
        AutomationTarget(kind="synth", name="cutoff_hz"),
        [
            (24.0, 48.0, 640.0, "exp"),
            (56.0, 64.0, 340.0, "exp"),
            (72.0, 112.0, 780.0, "exp"),
            (124.0, 148.0, 430.0, "exp"),
        ],
        default=420.0,
    )


def _pad_hall_ride() -> AutomationSpec:
    """More reverb as the world melts, drier again once home."""
    return _ramp_spec(
        AutomationTarget(kind="control", name="send_db"),
        [
            (56.0, 100.0, -6.0, "linear"),
            (124.0, 160.0, -10.0, "linear"),
        ],
        default=-10.0,
    )


def _add_eleven_carrier(score: Score) -> None:
    """The Act II counter-voice — the only voice possessing 11-family partials."""
    score.add_voice(
        "eleven",
        synth_defaults={
            "engine": "additive",
            "attack": 1.6,
            "release": 2.4,
            "decay_power": 2.0,
            "spectral_morph_type": "phase_disperse",
            "spectral_morph_amount": 0.5,
        },
        sends=[VoiceSend(target="hall", send_db=-5.0)],
        pan=-0.3,
        mix_db=-11.0,
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        drift_bus="kiln_drift",
        drift_bus_correlation=0.6,
    )


def _place_eleven_carrier(score: Score, start_bar: int, end_bar: int) -> None:
    """One long gliding note per 4 bars, alternating root <-> 11/10, swelling
    toward the climax and dissolving after it."""
    octave = 1  # sings above the pad
    for note_index, bar in enumerate(range(start_bar, end_bar, 4)):
        pseudo_octave = _pseudo_octave_for_bar(float(bar))
        octave_up = stretch_ratio(2.0, pseudo_octave) ** octave
        glide_up = note_index % 2 == 0
        motion = PitchMotionSpec.ratio_glide(
            start_ratio=1.0,
            end_ratio=stretch_ratio(11 / 10, pseudo_octave)
            if glide_up
            else 1.0 / stretch_ratio(11 / 10, pseudo_octave),
        )
        base_degree = 1.0 if glide_up else 11 / 10
        closeness = 1.0 - abs(bar - _PEAK_BAR) / (_PEAK_BAR - start_bar)
        score.add_note(
            "eleven",
            start=bar * BAR,
            duration=4 * BAR - 0.5,
            partial=stretch_ratio(base_degree, pseudo_octave) * octave_up,
            amp_db=-18.0 + 6.0 * max(0.0, closeness),
            velocity=0.6 + 0.35 * max(0.0, closeness),
            pitch_motion=motion,
            synth={"partials": _eleven_partials(pseudo_octave)},
        )


def _anneal_score() -> Score:
    score = Score(
        f0_hz=F0,
        master_effects=DEFAULT_MASTER_EFFECTS,
        send_buses=[_hall_bus(), _echo_bus()],
        timing_humanize=TimingHumanizeSpec(preset="tight_ensemble"),
    )
    score.add_drift_bus("kiln_drift", rate_hz=0.06, depth_cents=3.5, seed=20260705)
    drum_bus = setup_drum_bus(score, style="light")

    total_dur = _PIECE_BARS * BAR + 4.0
    _add_atmosphere(score, total_dur)
    _add_pad(score)
    _add_bass(score)
    _add_bell_arp(score)
    _add_drums(score, drum_bus)
    _add_eleven_carrier(score)

    # Automation rides (attached post-hoc to the palette voices).
    score.voices["arp"].sends.append(
        VoiceSend(target="echo", send_db=-34.0, automation=[_arp_echo_ride()])
    )
    score.voices["bass"].automation.append(_bass_cutoff_arc())
    score.voices["pad"].sends[0].automation.append(_pad_hall_ride())

    def pad(bar: int, n: int, chord: list[tuple[float, list[float]]]) -> None:
        _place_pad(score, bar, n, chord, _pseudo_octave_for_bar(float(bar)))

    def arp(
        bar: int,
        n: int,
        chord: list[tuple[float, list[float]]],
        pattern: object = None,
        velocity_scale: float = 1.0,
        vibrato: bool = False,
    ) -> None:
        # Sample S per 2 bars so the ramp is smooth (~2 cents per step).
        for cell in range(bar, bar + n, 2):
            cell_bars = min(2, bar + n - cell)
            _place_arp(
                score,
                cell,
                cell_bars,
                chord,
                _pseudo_octave_for_bar(float(cell)),
                pattern=pattern,  # type: ignore[arg-type]
                velocity_scale=velocity_scale,
                vibrato=vibrato,
            )

    def bass(bar: int, n: int, root: float, fifth_pickup: bool = False) -> None:
        for cell in range(bar, bar + n, 2):
            cell_bars = min(2, bar + n - cell)
            _place_bass(
                score,
                cell,
                cell_bars,
                root,
                _pseudo_octave_for_bar(float(cell)),
                fifth_pickup=fifth_pickup,
            )

    def drums(bar: int, n: int, root: float, **kwargs: object) -> None:
        for cell in range(bar, bar + n, 2):
            cell_bars = min(2, bar + n - cell)
            _place_drums(
                score,
                cell,
                cell_bars,
                root,
                _drum_pseudo_octave_for_bar(float(cell)),
                **kwargs,  # type: ignore[arg-type]
            )

    # ---- Act I: cool (bars 0-55) --------------------------------------
    pad(0, 8, TONIC_4_6_7)
    pad(8, 4, TONIC_4_6_7)
    pad(12, 4, FOURTH_OPEN)
    arp(8, 4, TONIC_4_6_7, velocity_scale=0.8)
    arp(12, 4, FOURTH_OPEN, velocity_scale=0.85)

    pad(16, 4, TONIC_4_6_7)
    pad(20, 2, FIFTH_6_7_8)
    pad(22, 2, TONIC_4_6_7)
    arp(16, 4, TONIC_4_6_7)
    arp(20, 2, FIFTH_6_7_8)
    arp(22, 2, TONIC_4_6_7)
    bass(16, 4, 1.0)
    bass(20, 2, 3 / 2)
    bass(22, 2, 1.0)
    drums(16, 8, 1.0, kick_on=False, open_hat_on=False, toms_on=False)

    pad(24, 4, TONIC_4_6_7)
    pad(28, 4, FOURTH_OPEN)
    arp(24, 4, TONIC_4_6_7)
    arp(28, 4, FOURTH_OPEN)
    bass(24, 4, 1.0)
    bass(28, 4, 4 / 3)
    drums(24, 8, 1.0, open_hat_on=False, toms_on=False)

    pad(32, 2, TONIC_4_6_7)
    pad(34, 2, FOURTH_OPEN)
    pad(36, 2, FIFTH_6_7_8)
    pad(38, 2, TONIC_4_6_7)
    arp(32, 2, TONIC_4_6_7)
    arp(34, 2, FOURTH_OPEN)
    arp(36, 2, FIFTH_6_7_8)
    arp(38, 2, TONIC_4_6_7)
    bass(32, 2, 1.0)
    bass(34, 2, 4 / 3)
    bass(36, 2, 3 / 2)
    bass(38, 2, 1.0)
    drums(32, 8, 1.0)

    # Theme A: the motif statement the ending will answer.
    for offset, chord, root, n in [
        (40, TONIC_4_6_7, 1.0, 4),
        (44, FOURTH_OPEN, 4 / 3, 4),
        (48, FIFTH_6_7_8, 3 / 2, 2),
        (50, TONIC_4_6_7, 1.0, 2),
        (52, FOURTH_OPEN, 4 / 3, 2),
        (54, FIFTH_6_7_8, 3 / 2, 2),
    ]:
        pad(offset, n, chord)
        arp(offset, n, chord)
        bass(offset, n, root, fifth_pickup=True)
    drums(40, 15, 1.0, fill_bars=True)
    # bar 55: percussion breath before the heat.

    # ---- Act II: heat (bars 56-123) ------------------------------------
    pad(56, 4, MINOR_16_19_24)
    pad(60, 4, NEUTRAL_SUBDOMINANT)
    arp(56, 8, MINOR_16_19_24, pattern=_ARP_CYCLE_B, vibrato=True)
    bass(56, 4, 1.0)
    bass(60, 4, 4 / 3)
    drums(56, 8, 1.0, kick_pattern="sparse", open_hat_on=False)

    pad(64, 4, MINOR_16_19_24)
    pad(68, 4, FIFTH_6_7_8)
    arp(64, 4, MINOR_16_19_24, pattern=_ARP_CYCLE_B, vibrato=True)
    arp(68, 4, FIFTH_6_7_8, pattern=_ARP_CYCLE_B, vibrato=True)
    bass(64, 4, 1.0)
    bass(68, 4, 3 / 2)
    drums(64, 8, 1.0)

    for offset, chord, root in [
        (72, MINOR_16_19_24, 1.0),
        (76, FOURTH_OPEN, 4 / 3),
        (80, NEUTRAL_SUBDOMINANT, 4 / 3),
        (84, FIFTH_6_7_8, 3 / 2),
    ]:
        pad(offset, 4, chord)
        arp(offset, 4, chord)
        bass(offset, 4, root, fifth_pickup=True)
    drums(72, 16, 1.0, fill_bars=True)

    for offset, chord, root in [
        (88, MINOR_16_19_24, 1.0),
        (92, NEUTRAL_SUBDOMINANT, 4 / 3),
        (96, FIFTH_6_7_8, 3 / 2),
        (100, MINOR_16_19_24, 1.0),
    ]:
        pad(offset, 4, chord)
        arp(offset, 4, chord, pattern=_ARP_CYCLE_C, velocity_scale=1.05)
        bass(offset, 4, root, fifth_pickup=True)
    drums(88, 16, 1.0, kick_pattern="dense", fill_bars=True)

    pad(104, 4, FIFTH_6_7_8)
    pad(108, 4, FIFTH_6_7_8)
    arp(104, 8, FIFTH_6_7_8, pattern=_ARP_CYCLE_C, velocity_scale=1.1)
    bass(104, 7, 3 / 2, fifth_pickup=True)
    drums(104, 7, 1.0, kick_pattern="dense")
    # bar 111: full percussion dropout — one bar of held breath.

    pad(112, 4, SPICE_CHORD)
    arp(112, 4, SPICE_CHORD, pattern=_ARP_CYCLE_C, velocity_scale=1.1)
    bass(112, 4, 1.0)
    drums(113, 3, 1.0, kick_pattern="dense", fill_bars=True)

    pad(116, 4, TONIC_4_6_7)
    pad(120, 4, TONIC_4_6_7)
    arp(116, 8, TONIC_4_6_7)
    bass(116, 7, 1.0)
    drums(116, 7, 1.0)
    _place_eleven_carrier(score, 72, 124)
    # bar 123: percussion breath before the cooling.

    # ---- Act III: anneal (bars 124-176) --------------------------------
    for offset, chord, root in [
        (124, TONIC_4_6_7, 1.0),
        (128, FOURTH_OPEN, 4 / 3),
        (132, TONIC_4_6_7, 1.0),
        (136, FOURTH_OPEN, 4 / 3),
    ]:
        pad(offset, 4, chord)
        arp(offset, 4, chord, pattern=_ARP_CYCLE_B, vibrato=True)
        bass(offset, 4, root)
    drums(124, 16, 1.0, open_hat_on=False)

    for offset, chord, root in [
        (140, TONIC_4_6_7, 1.0),
        (144, FIFTH_6_7_8, 3 / 2),
        (148, TONIC_4_6_7, 1.0),
    ]:
        pad(offset, 4, chord)
        arp(offset, 4, chord, pattern=_ARP_CYCLE_B, vibrato=True, velocity_scale=0.9)
        bass(offset, 4, root)
    drums(140, 12, 1.0, kick_pattern="sparse", open_hat_on=False, toms_on=False)

    # Theme A returns as the world settles home.
    for offset, chord, root, n in [
        (152, TONIC_4_6_7, 1.0, 4),
        (156, FOURTH_OPEN, 4 / 3, 4),
        (160, FIFTH_6_7_8, 3 / 2, 2),
        (162, TONIC_4_6_7, 1.0, 2),
        (164, FOURTH_OPEN, 4 / 3, 2),
        (166, FIFTH_6_7_8, 3 / 2, 2),
    ]:
        pad(offset, n, chord)
        arp(offset, n, chord, velocity_scale=0.9)
        bass(offset, n, root)
    drums(152, 12, 1.0, kick_pattern="sparse", open_hat_on=False, toms_on=False)

    # Home: the final maximum-fusion tonic, room only underneath.
    pad(168, 8, TONIC_4_6_7)
    arp(168, 4, TONIC_4_6_7, velocity_scale=0.75)
    bass(168, 4, 1.0)
    score.add_note(
        "pad",
        start=168 * BAR,
        duration=8 * BAR + 2.0,
        partial=stretch_ratio(7 / 4, HOME_PSEUDO_OCTAVE) * 2.0,
        amp_db=-22.0,
        velocity=0.6,
        pitch_motion=PitchMotionSpec.vibrato(depth_ratio=0.004, rate_hz=4.6),
        synth={"partials": SKELETON_PARTIALS},
    )
    return score


_ANNEAL_SECTIONS = (
    PieceSection("Act I: cool — room and pad", 0.0, 8 * BAR),
    PieceSection("Act I: arp motif enters", 8 * BAR, 16 * BAR),
    PieceSection("Act I: beat assembles", 16 * BAR, 40 * BAR),
    PieceSection("Act I: theme A", 40 * BAR, 56 * BAR),
    PieceSection("Act II: dark turn (16:19:24)", 56 * BAR, 72 * BAR),
    PieceSection("Act II: 11-carrier, stretch deepens", 72 * BAR, 88 * BAR),
    PieceSection("Act II: energy peak", 88 * BAR, 111 * BAR),
    PieceSection("Act II: dropout + spice-chord climax", 111 * BAR, 116 * BAR),
    PieceSection("Act II: afterglow at full stretch", 116 * BAR, 124 * BAR),
    PieceSection("Act III: cooling", 124 * BAR, 152 * BAR),
    PieceSection("Act III: theme A returns", 152 * BAR, 168 * BAR),
    PieceSection("Act III: home", 168 * BAR, 176 * BAR + 4.0),
)


_PALETTE_SECTIONS = (
    PieceSection("intro (pad + room)", 0.0, 2 * BAR),
    PieceSection("tonic groove", 2 * BAR, 8 * BAR),
    PieceSection("neutral subdominant", 8 * BAR, 12 * BAR),
    PieceSection("septimal V (6:7:8)", 12 * BAR, 14 * BAR),
    PieceSection("tonic return", 14 * BAR, 18 * BAR),
    PieceSection("IV - V - I close", 18 * BAR, 24 * BAR),
)

_FUSION_SECTIONS = (
    PieceSection("tonic 4:6:7 (home)", 0.0, 6.0),
    PieceSection("16:19:24, root 19-color (home)", 6.0, 12.0),
    PieceSection("neutral triad, 49-fusion (home)", 12.0, 18.0),
    PieceSection("tonic again (home)", 18.0, 24.0),
    PieceSection("stretch ramp 2.00 -> 2.07", 24.0, 42.0),
    PieceSection("held at 2.07", 42.0, 48.0),
    PieceSection("anneal home", 48.0, 63.0),
)

PIECES = {
    "anneal_fusion_sketch": PieceDefinition(
        name="anneal_fusion_sketch",
        output_name="anneal_fusion_sketch",
        build_score=_fusion_sketch_score,
        sections=_FUSION_SECTIONS,
        study=True,
    ),
    "anneal_palette_sketch": PieceDefinition(
        name="anneal_palette_sketch",
        output_name="anneal_palette_sketch",
        build_score=_palette_sketch_score,
        sections=_PALETTE_SECTIONS,
        study=True,
    ),
    "anneal": PieceDefinition(
        name="anneal",
        output_name="anneal",
        build_score=_anneal_score,
        sections=_ANNEAL_SECTIONS,
    ),
}
