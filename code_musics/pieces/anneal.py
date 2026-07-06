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

from code_musics.drum_helpers import add_drum_voice, setup_drum_bus
from code_musics.humanize import EnvelopeHumanizeSpec, TimingHumanizeSpec
from code_musics.meter import Groove
from code_musics.pieces._shared import (
    DEFAULT_MASTER_EFFECTS,
    SOFT_REVERB_EFFECT,
    bricasti_or_reverb,
)
from code_musics.pieces.registry import PieceDefinition, PieceSection
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


def _place_bass(score: Score, start_bar: int, n_bars: int, root_degree: float) -> None:
    for bar in range(start_bar, start_bar + n_bars):
        bar_t = bar * BAR
        low_root = stretch_ratio(root_degree, HOME_PSEUDO_OCTAVE) * 0.5
        score.add_note(
            "bass", start=bar_t, duration=1.5 * BEAT, partial=low_root, velocity=0.95
        )
        score.add_note(
            "bass",
            start=bar_t + 2.5 * BEAT,
            duration=0.6 * BEAT,
            partial=low_root,
            velocity=0.7,
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


def _place_arp(
    score: Score,
    start_bar: int,
    n_bars: int,
    chord: list[tuple[float, list[float]]],
    pseudo_octave: float = HOME_PSEUDO_OCTAVE,
) -> None:
    degrees = [degree for degree, _ in chord]
    tones = [
        degrees[0] * 2.0,
        degrees[1] * 2.0,
        degrees[2] * 2.0,
        degrees[0] * 4.0,
        degrees[1] * 4.0,
    ]
    steps_per_cycle = 32
    total_steps = n_bars * 16
    for cycle_start in range(0, total_steps, steps_per_cycle):
        for step, tone_index, velocity in _ARP_CYCLE:
            absolute_step = cycle_start + step
            if absolute_step >= total_steps:
                break
            onset = (
                start_bar * BAR
                + absolute_step * S16
                + S16 * GROOVE.timing_offset_at(absolute_step)
            )
            score.add_note(
                "arp",
                start=onset,
                duration=0.11,
                partial=stretch_ratio(tones[tone_index], pseudo_octave),
                velocity=velocity,
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
            # Punchy low thump, minimal click: in this sparse mix a loud
            # exciter reads as a separate midrange "clap" (audition 2).
            "tone_punch": 0.6,
            "exciter_level": 0.08,
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


def _place_drums(
    score: Score, start_bar: int, n_bars: int, root_degree: float = 1.0
) -> None:
    kick_steps_a = [(0, 1.0), (7, 0.85), (10, 0.9)]
    kick_steps_b = [(0, 1.0), (7, 0.85), (13, 0.75)]
    for bar in range(start_bar, start_bar + n_bars):
        bar_t = bar * BAR
        kick_steps = kick_steps_a if bar % 2 == 0 else kick_steps_b
        for step, velocity in kick_steps:
            score.add_note(
                "kick",
                start=bar_t + step * S16,
                duration=0.3,
                freq=49.0,
                velocity=velocity,
            )
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
        if bar % 2 == 1:
            onset = bar_t + 14 * S16 + S16 * GROOVE.timing_offset_at(14)
            score.add_note(
                "hat_open", start=onset, duration=0.4, freq=784.0, velocity=0.7
            )
        if bar % 4 == 3:
            tom_freq = F0 * root_degree
            score.add_note(
                "tom",
                start=bar_t + 11 * S16,
                duration=0.5,
                freq=tom_freq,
                velocity=0.7,
            )
            score.add_note(
                "tom",
                start=bar_t + 14 * S16,
                duration=0.5,
                freq=tom_freq * 3 / 4,
                velocity=0.5,
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
}
