"""clock_of_7 — polymetric Machinedrum-inspired acid workout in 7-limit JI.

Sibling to `newton_bloom.py` in the same tonal soil (F#3 = 185 Hz, 7-limit JI).
Where newton_bloom is patient and subtractive, this piece is rhythmic and
alien: a 7/8 acid-lead phrase phasing against a 4/4 drum grid, a 303-style
diode-filter acid bass with `voice_dist_mode="corrode"`, a Newton-solver
acid lead at `quality="divine"`, euclidean(5,8) hats, and a spectralwave
VA pad ducked by the kick.

Structural arc (BPM 118, BEAT ≈ 0.508s, BAR ≈ 2.034s):

  1. Intro       (0-8   bars, ~0:00-0:16)  — kick + euclidean hats
  2. Bass in     (8-24  bars, ~0:16-0:49)  — acid bass + pad wash
  3. Main groove (24-48 bars, ~0:49-1:38)  — full ensemble, 7/8 lead phasing
  4. Breakdown   (48-64 bars, ~1:38-2:10)  — drums thin, lead sings alone
  5. Apex        (64-80 bars, ~2:10-2:43)  — full + audio-rate bass FM
  6. Outro       (80-90 bars, ~2:43-3:03)  — elements drop, hall tail
"""

from __future__ import annotations

from code_musics.automation import (
    AutomationSegment,
    AutomationSpec,
    AutomationTarget,
)
from code_musics.drum_helpers import add_drum_voice, setup_drum_bus
from code_musics.generative.euclidean import euclidean_pattern
from code_musics.modulation import (
    LFOSource,
    MacroSource,
    ModConnection,
    OscillatorSource,
)
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS, SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import EffectSpec, Score, VoiceSend

BPM = 118.0
BEAT = 60.0 / BPM
BAR = 4.0 * BEAT
S16 = BEAT / 4.0

# Section bar boundaries
S1_BAR = 0  # intro (kick + hats)
S2_BAR = 8  # bass enters
S3_BAR = 24  # main groove
S4_BAR = 48  # breakdown
S5_BAR = 64  # apex
S6_BAR = 80  # outro
TOTAL_BARS = 90

S1_START = S1_BAR * BAR
S2_START = S2_BAR * BAR
S3_START = S3_BAR * BAR
S4_START = S4_BAR * BAR
S5_START = S5_BAR * BAR
S6_START = S6_BAR * BAR
TOTAL_DUR = TOTAL_BARS * BAR

F0_HZ = 185.0  # F#3 — shared with newton_bloom + diva_study
KICK_PARTIAL = 0.25  # F#1 (46.25 Hz) — on-root sub-kick, two octaves below F0


# ---------------------------------------------------------------------------
# Rhythm & melody materials
# ---------------------------------------------------------------------------

# 7-note phrase in 7/8 that phases against the 4/4 grid.  Each phrase note
# is one 16th, so one cycle = 7 * S16; the bar is 16 * S16, creating a
# 7 vs 16 polymeter that re-aligns every 7 bars.
_LEAD_7_PHRASE: tuple[float, ...] = (
    1.0,  # F# tonic
    7 / 6,  # septimal m3
    3 / 2,  # 5th
    7 / 4,  # septimal 7th
    9 / 8,  # M2
    5 / 3,  # M6
    7 / 5,  # septimal tritone
)

# Acid bass line — 16 steps at sub-octave partials.  Accents every 4th step.
_BASS_16_PATTERN: tuple[float | None, ...] = (
    0.5,
    None,
    0.5,
    None,
    7 / 12,
    None,
    0.5,
    7 / 16,
    2 / 3,
    None,
    0.5,
    None,
    7 / 12,
    0.5,
    7 / 16,
    None,
)
_BASS_ACCENT_STEPS: frozenset[int] = frozenset({0, 4, 8, 12})


def _bass_step_amp_db(step_index: int) -> float:
    return -6.0 if step_index in _BASS_ACCENT_STEPS else -11.0


# Euclidean(5, 8) hat pattern — five onsets spread across 8 sixteenth-notes.
_HAT_EUCLIDEAN: tuple[bool, ...] = euclidean_pattern(5, 8, rotation=0)
# The open-hat slots are those where the pattern falls on the off-beat
# halves (steps 2, 5, 7 ish — derived from pattern below).
_HAT_OPEN_STEPS: frozenset[int] = frozenset({2, 5})


# Clap fires on beat 3 of every bar (offset-quarter snare-sub), plus a
# septimal pass-through: once per 7-bar cycle we skip the clap to make
# the polyrhythm breathe.
def _clap_hits(start_bar: int, end_bar: int) -> list[float]:
    hits: list[float] = []
    for bar in range(start_bar, end_bar):
        if bar % 7 == 6:
            # Skip every 7th bar — punctuates the 7-cycle
            continue
        hits.append(bar * BAR + 2 * BEAT)
    return hits


# Pad chord progression — two bars per chord, moving through 7-limit JI roots.
# Utonal chords reserved for the breakdown to mark the alien section.
_PAD_PROGRESSION: list[tuple[float, ...]] = [
    (1.0, 5 / 4, 3 / 2, 7 / 4),  # bar 0-2
    (1.0, 5 / 4, 3 / 2, 7 / 4),
    (4 / 3, 5 / 3, 2.0, 7 / 3),  # bar 4
    (4 / 3, 5 / 3, 2.0, 7 / 3),
    (9 / 8, 45 / 32, 27 / 16, 63 / 32),
    (9 / 8, 45 / 32, 27 / 16, 63 / 32),
    (3 / 2, 15 / 8, 9 / 4, 21 / 8),
    (3 / 2, 15 / 8, 9 / 4, 21 / 8),
]
_PAD_BREAKDOWN_PROGRESSION: list[tuple[float, ...]] = [
    (7 / 6, 35 / 24, 7 / 4, 49 / 24),  # utonal shadow
    (7 / 6, 35 / 24, 7 / 4, 49 / 24),
    (1.0, 7 / 6, 7 / 5, 7 / 4),  # septimal cloud
    (1.0, 7 / 6, 7 / 5, 7 / 4),
    (4 / 3, 49 / 30, 7 / 4, 49 / 24),
    (4 / 3, 49 / 30, 7 / 4, 49 / 24),
    (3 / 2, 15 / 8, 7 / 4, 21 / 8),  # pivot back toward groove
    (3 / 2, 15 / 8, 7 / 4, 21 / 8),
]


def _add_kick(score: Score) -> None:
    """Every-beat kick starting bar 0; rests through breakdown; returns apex+outro."""
    # Intro + bass-in + main groove: continuous four-on-the-floor.
    for bar in range(S3_BAR + (S4_BAR - S3_BAR)):
        # Break: drop the kick on every 8th beat in bars 22-23 for a
        # build-up fill into main groove? Skip — keep it simple and steady.
        if bar >= S4_BAR:
            break
        for beat in range(4):
            score.add_note(
                "kick",
                start=bar * BAR + beat * BEAT,
                duration=0.25,
                partial=KICK_PARTIAL,
                amp_db=-3.0 if beat == 0 else -4.5,
                velocity=1.0,
            )
    # Breakdown: kick drops out entirely except for bar-boundary markers
    # at the start of each 4-bar subsection.
    for bar in range(S4_BAR, S5_BAR):
        if bar % 4 == 0:
            score.add_note(
                "kick",
                start=bar * BAR,
                duration=0.25,
                partial=KICK_PARTIAL,
                amp_db=-6.0,
                velocity=0.95,
            )
    # Apex + outro tail: full kick.
    for bar in range(S5_BAR, TOTAL_BARS - 2):
        for beat in range(4):
            score.add_note(
                "kick",
                start=bar * BAR + beat * BEAT,
                duration=0.25,
                partial=KICK_PARTIAL,
                amp_db=-3.0 if beat == 0 else -4.5,
                velocity=1.0,
            )
    # Final two bars: kick only on downbeat, letting reverb breathe.
    for bar in (TOTAL_BARS - 2, TOTAL_BARS - 1):
        score.add_note(
            "kick",
            start=bar * BAR,
            duration=0.25,
            partial=KICK_PARTIAL,
            amp_db=-5.0,
            velocity=0.9,
        )


def _add_hats(score: Score) -> None:
    """Euclidean(5,8) 16th-note hats across 8-step windows, panned wide."""
    steps_per_bar = 8
    step_dur = BEAT / 2.0  # 8th notes
    for bar in range(TOTAL_BARS):
        # Thin out during breakdown.
        hat_attenuation_db = -8.0 if S4_BAR <= bar < S5_BAR else 0.0
        # Drop hats for the first 4 bars of the breakdown entirely.
        if S4_BAR <= bar < S4_BAR + 4:
            continue
        for step in range(steps_per_bar):
            if not _HAT_EUCLIDEAN[step]:
                continue
            start = bar * BAR + step * step_dur
            voice_name = "open_hat" if step in _HAT_OPEN_STEPS else "closed_hat"
            score.add_note(
                voice_name,
                start=start,
                duration=step_dur * 0.8,
                partial=1.0,
                amp_db=-10.0 + hat_attenuation_db,
                velocity=0.9 if step in _HAT_OPEN_STEPS else 0.75,
            )


def _add_clap(score: Score) -> None:
    """Clap on beat 3 of every bar, with a 7-bar skip for polyrhythmic lilt."""
    # Claps start in bar 4 (after the intro kick has locked in) and stop
    # at the breakdown, then return for the apex.
    for start_time in _clap_hits(start_bar=4, end_bar=S4_BAR):
        score.add_note(
            "clap",
            start=start_time,
            duration=0.2,
            partial=1.0,
            amp_db=-7.0,
            velocity=1.0,
        )
    for start_time in _clap_hits(start_bar=S5_BAR, end_bar=S6_BAR):
        score.add_note(
            "clap",
            start=start_time,
            duration=0.2,
            partial=1.0,
            amp_db=-6.0,
            velocity=1.05,
        )


def _add_bass(score: Score) -> None:
    """16th-note acid bass with accents and ratio_glide on accented notes."""
    step_dur = S16
    start_bar = S2_BAR
    end_bar = S6_BAR  # bass plays through apex, drops in outro
    for bar in range(start_bar, end_bar):
        # Breakdown: bass thins to just downbeats.
        if S4_BAR <= bar < S5_BAR:
            partial = _BASS_16_PATTERN[0]
            if partial is not None:
                score.add_note(
                    "bass",
                    start=bar * BAR,
                    duration=step_dur * 14,  # long hold
                    partial=partial,
                    amp_db=-8.0,
                    velocity=0.9,
                )
            continue
        for step_index, partial in enumerate(_BASS_16_PATTERN):
            if partial is None:
                continue
            amp_db = _bass_step_amp_db(step_index)
            pitch_motion = None
            if step_index in _BASS_ACCENT_STEPS and step_index > 0:
                # Ratio glide from the previous pitch into this one.
                prev_partial = None
                for look_back in range(1, 5):
                    candidate = _BASS_16_PATTERN[(step_index - look_back) % 16]
                    if candidate is not None:
                        prev_partial = candidate
                        break
                if prev_partial is not None and prev_partial != partial:
                    pitch_motion = PitchMotionSpec.ratio_glide(
                        start_ratio=prev_partial / partial,
                        end_ratio=1.0,
                    )
            score.add_note(
                "bass",
                start=bar * BAR + step_index * step_dur,
                duration=step_dur * 0.9,
                partial=partial,
                amp_db=amp_db,
                velocity=1.05 if step_index in _BASS_ACCENT_STEPS else 0.9,
                pitch_motion=pitch_motion,
            )


def _add_lead(score: Score) -> None:
    """7-step 16th phrase looping against 4/4 bar — polymetric 7 vs 16."""
    step_dur = S16
    # Lead enters at main groove, active through breakdown (where it's featured),
    # apex, and into early outro.
    start_bar = S3_BAR
    end_bar = S6_BAR + 4  # tail into outro for fade
    start_time = start_bar * BAR
    end_time = end_bar * BAR
    phrase_len = len(_LEAD_7_PHRASE)
    step_index = 0
    t = start_time
    while t < end_time:
        partial = _LEAD_7_PHRASE[step_index % phrase_len]
        # Breakdown: longer durations, held legato feel.
        if S4_START <= t < S5_START:
            dur = step_dur * 2.5
            amp_db = -8.0
            velocity = 0.95
        else:
            dur = step_dur * 0.85
            amp_db = -6.0
            velocity = 1.0
        # Outro: fade out velocity.
        if t >= S6_START:
            amp_db -= 4.0
            velocity = 0.85
        score.add_note(
            "lead",
            start=t,
            duration=dur,
            partial=partial,
            amp_db=amp_db,
            velocity=velocity,
        )
        # Advance time.  Use longer steps in breakdown for melodic phrasing.
        if S4_START <= t < S5_START:
            t += step_dur * 3.0
        else:
            t += step_dur
        step_index += 1


def _add_pad(score: Score) -> None:
    """Spectralwave pad chords across sections, with breakdown utonal shift."""
    stagger = (0.0, 0.025, 0.05, 0.075)
    # Main groove uses the standard progression; breakdown uses utonal progression.
    for bar in range(S3_BAR, S4_BAR):
        progression_index = (bar - S3_BAR) // 2
        partials = _PAD_PROGRESSION[progression_index % len(_PAD_PROGRESSION)]
        for partial, offset in zip(partials, stagger, strict=True):
            score.add_note(
                "pad",
                start=bar * BAR + offset,
                duration=2.0 * BAR - 0.1,
                partial=partial,
                amp_db=-16.0,
                velocity=0.9,
            )
    # Breakdown: utonal shadows, held longer.
    for bar in range(S4_BAR, S5_BAR):
        progression_index = (bar - S4_BAR) // 2
        partials = _PAD_BREAKDOWN_PROGRESSION[
            progression_index % len(_PAD_BREAKDOWN_PROGRESSION)
        ]
        for partial, offset in zip(partials, stagger, strict=True):
            score.add_note(
                "pad",
                start=bar * BAR + offset,
                duration=2.0 * BAR - 0.1,
                partial=partial,
                amp_db=-12.0,  # pad featured in breakdown
                velocity=0.95,
            )
    # Apex: back to main progression, hotter.
    for bar in range(S5_BAR, S6_BAR):
        progression_index = (bar - S5_BAR) // 2
        partials = _PAD_PROGRESSION[progression_index % len(_PAD_PROGRESSION)]
        for partial, offset in zip(partials, stagger, strict=True):
            score.add_note(
                "pad",
                start=bar * BAR + offset,
                duration=2.0 * BAR - 0.1,
                partial=partial,
                amp_db=-14.0,
                velocity=0.95,
            )
    # Outro: long sustained tonic, fades via mix_db automation.
    score.add_note(
        "pad",
        start=S6_START,
        duration=TOTAL_DUR - S6_START - 0.1,
        partial=1.0,
        amp_db=-16.0,
        velocity=0.85,
    )
    score.add_note(
        "pad",
        start=S6_START + 0.025,
        duration=TOTAL_DUR - S6_START - 0.1,
        partial=3 / 2,
        amp_db=-17.0,
        velocity=0.85,
    )


# ---------------------------------------------------------------------------
# Automation lanes
# ---------------------------------------------------------------------------


def _build_intensity_macro_automation() -> AutomationSpec:
    """Intensity: 0 (intro) → 0.5 (main) → 0.3 (breakdown) → 1.0 (apex) → 0 (outro)."""
    return AutomationSpec(
        target=AutomationTarget(kind="control", name="mix"),
        segments=(
            AutomationSegment(
                start=0.0,
                end=S3_START,
                shape="linear",
                start_value=0.0,
                end_value=0.45,
            ),
            AutomationSegment(
                start=S3_START,
                end=S4_START,
                shape="linear",
                start_value=0.45,
                end_value=0.55,
            ),
            AutomationSegment(
                start=S4_START,
                end=S5_START,
                shape="linear",
                start_value=0.55,
                end_value=0.35,
            ),
            AutomationSegment(
                start=S5_START,
                end=S6_START,
                shape="linear",
                start_value=0.35,
                end_value=1.0,
            ),
            AutomationSegment(
                start=S6_START,
                end=TOTAL_DUR,
                shape="linear",
                start_value=1.0,
                end_value=0.0,
            ),
        ),
        clamp_min=0.0,
        clamp_max=1.0,
    )


def _build_fm_macro_automation() -> AutomationSpec:
    """Audio-rate bass FM: active only in the apex."""
    apex_mid = S5_START + (S6_START - S5_START) * 0.5
    return AutomationSpec(
        target=AutomationTarget(kind="control", name="mix"),
        segments=(
            AutomationSegment(
                start=0.0,
                end=S5_START,
                shape="hold",
                value=0.0,
            ),
            AutomationSegment(
                start=S5_START,
                end=apex_mid,
                shape="linear",
                start_value=0.0,
                end_value=1.0,
            ),
            AutomationSegment(
                start=apex_mid,
                end=S6_START,
                shape="linear",
                start_value=1.0,
                end_value=0.0,
            ),
            AutomationSegment(
                start=S6_START,
                end=TOTAL_DUR,
                shape="hold",
                value=0.0,
            ),
        ),
        clamp_min=0.0,
        clamp_max=1.0,
    )


def _build_lead_cutoff_automation() -> AutomationSpec:
    """Lead cutoff rides with the intensity arc.  Exp for frequency space."""
    return AutomationSpec(
        target=AutomationTarget(kind="synth", name="cutoff_hz"),
        segments=(
            AutomationSegment(
                start=0.0,
                end=S3_START,
                shape="hold",
                value=800.0,
            ),
            AutomationSegment(
                start=S3_START,
                end=S4_START,
                shape="exp",
                start_value=800.0,
                end_value=2400.0,
            ),
            AutomationSegment(
                start=S4_START,
                end=S5_START,
                shape="exp",
                start_value=2400.0,
                end_value=3600.0,
            ),
            AutomationSegment(
                start=S5_START,
                end=S6_START,
                shape="exp",
                start_value=3600.0,
                end_value=4800.0,
            ),
            AutomationSegment(
                start=S6_START,
                end=TOTAL_DUR,
                shape="exp",
                start_value=4800.0,
                end_value=600.0,
            ),
        ),
    )


def _build_lead_filter_morph_automation() -> AutomationSpec:
    """Filter morph rides across the breakdown — dramatic slope changes."""
    return AutomationSpec(
        target=AutomationTarget(kind="synth", name="filter_morph"),
        segments=(
            AutomationSegment(
                start=0.0,
                end=S3_START,
                shape="hold",
                value=0.0,
            ),
            AutomationSegment(
                start=S3_START,
                end=S4_START,
                shape="linear",
                start_value=0.0,
                end_value=0.2,
            ),
            AutomationSegment(
                start=S4_START,
                end=S4_START + 8.0 * BAR,
                shape="linear",
                start_value=0.2,
                end_value=0.85,
            ),
            AutomationSegment(
                start=S4_START + 8.0 * BAR,
                end=S5_START,
                shape="linear",
                start_value=0.85,
                end_value=0.3,
            ),
            AutomationSegment(
                start=S5_START,
                end=S6_START,
                shape="linear",
                start_value=0.3,
                end_value=0.15,
            ),
            AutomationSegment(
                start=S6_START,
                end=TOTAL_DUR,
                shape="linear",
                start_value=0.15,
                end_value=0.0,
            ),
        ),
    )


def _build_pad_spectral_position_automation() -> AutomationSpec:
    """Pad spectral_position slowly morphs — saw→spectral→square continuum."""
    return AutomationSpec(
        target=AutomationTarget(kind="synth", name="spectral_position"),
        segments=(
            AutomationSegment(
                start=0.0,
                end=S4_START,
                shape="linear",
                start_value=0.25,
                end_value=0.45,
            ),
            AutomationSegment(
                start=S4_START,
                end=S5_START,
                shape="linear",
                start_value=0.45,
                end_value=0.75,
            ),
            AutomationSegment(
                start=S5_START,
                end=S6_START,
                shape="linear",
                start_value=0.75,
                end_value=0.55,
            ),
            AutomationSegment(
                start=S6_START,
                end=TOTAL_DUR,
                shape="linear",
                start_value=0.55,
                end_value=0.25,
            ),
        ),
        clamp_min=0.0,
        clamp_max=1.0,
    )


def _build_pad_mix_automation() -> AutomationSpec:
    """Pad fades in through intro, peaks in apex, fades out in outro."""
    return AutomationSpec(
        target=AutomationTarget(kind="control", name="mix_db"),
        segments=(
            AutomationSegment(
                start=0.0,
                end=S2_START,
                shape="linear",
                start_value=-18.0,
                end_value=-10.0,
            ),
            AutomationSegment(
                start=S2_START,
                end=S4_START,
                shape="linear",
                start_value=-10.0,
                end_value=-6.0,
            ),
            AutomationSegment(
                start=S4_START,
                end=S5_START,
                shape="linear",
                start_value=-6.0,
                end_value=-3.0,  # featured in breakdown
            ),
            AutomationSegment(
                start=S5_START,
                end=S6_START,
                shape="linear",
                start_value=-3.0,
                end_value=-5.0,
            ),
            AutomationSegment(
                start=S6_START,
                end=TOTAL_DUR,
                shape="linear",
                start_value=-5.0,
                end_value=-24.0,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Score builder
# ---------------------------------------------------------------------------


def build_score() -> Score:
    score = Score(
        f0_hz=F0_HZ,
        master_effects=DEFAULT_MASTER_EFFECTS,
    )

    # ---- Macros ----
    score.add_macro(
        "intensity", default=0.0, automation=_build_intensity_macro_automation()
    )
    score.add_macro("fm_apex", default=0.0, automation=_build_fm_macro_automation())

    # ---- Score-level modulations ----
    score.modulations.extend(
        [
            # Slow LFO breathes the pad cutoff.
            ModConnection(
                name="pad_lfo_cutoff",
                source=LFOSource(rate_hz=0.09, waveshape="sine", seed=11),
                target=AutomationTarget(kind="synth", name="filter1_cutoff_hz"),
                amount=220.0,
                bipolar=True,
                mode="add",
            ),
            # Intensity macro rides the lead's resonance Q.
            ModConnection(
                name="intensity_macro_resonance",
                source=MacroSource(name="intensity"),
                target=AutomationTarget(kind="synth", name="resonance_q"),
                amount=14.0,  # up to +14 on the base 22 = 36 peak
                bipolar=False,
                mode="add",
            ),
            # Audio-rate FM on bass detune during apex.
            ModConnection(
                name="bass_fm_base",
                source=OscillatorSource(rate_hz=210.0, waveshape="sine"),
                target=AutomationTarget(kind="synth", name="osc2_detune_cents"),
                amount=4.0,  # subtle base
                bipolar=True,
                mode="add",
            ),
            ModConnection(
                name="bass_fm_apex_boost",
                source=MacroSource(name="fm_apex"),
                target=AutomationTarget(kind="synth", name="osc2_detune_cents"),
                amount=14.0,
                bipolar=False,
                mode="add",
            ),
        ]
    )

    # ---- Send buses ----
    drum_bus = setup_drum_bus(
        score,
        effects=[
            EffectSpec(
                "compressor",
                {
                    "threshold_db": -18.0,
                    "ratio": 2.0,
                    "attack_ms": 12.0,
                    "release_ms": 150.0,
                    "knee_db": 6.0,
                    "topology": "feedback",
                    "detector_mode": "rms",
                },
            ),
        ],
        return_db=0.0,
    )
    score.add_send_bus(
        "hall",
        effects=[
            SOFT_REVERB_EFFECT,
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
                            "kind": "lowpass",
                            "cutoff_hz": 5500.0,
                            "slope_db_per_oct": 12,
                        },
                    ]
                },
            ),
        ],
        return_db=-6.0,
    )

    # ---- Drum voices ----
    add_drum_voice(
        score,
        "kick",
        engine="drum_voice",
        preset="efm_kick_punch",
        drum_bus=drum_bus,
        send_db=-2.0,
        effects=[EffectSpec("compressor", {"preset": "kick_punch"})],
        mix_db=0.0,
    )
    add_drum_voice(
        score,
        "closed_hat",
        engine="drum_voice",
        preset="closed_hat",
        drum_bus=drum_bus,
        send_db=-5.0,
        choke_group="hats",
        effects=[
            EffectSpec("compressor", {"preset": "hat_control"}),
            EffectSpec(
                "eq",
                {"bands": [{"kind": "high_shelf", "freq_hz": 8000.0, "gain_db": 2.5}]},
            ),
        ],
        mix_db=-7.0,
        pan=-0.25,
    )
    add_drum_voice(
        score,
        "open_hat",
        engine="drum_voice",
        preset="open_hat",
        drum_bus=drum_bus,
        send_db=-4.0,
        choke_group="hats",
        effects=[
            EffectSpec(
                "eq",
                {"bands": [{"kind": "high_shelf", "freq_hz": 7000.0, "gain_db": 2.0}]},
            ),
        ],
        mix_db=-6.0,
        pan=0.25,
    )
    add_drum_voice(
        score,
        "clap",
        engine="drum_voice",
        preset="tight_clap",
        drum_bus=drum_bus,
        send_db=-3.0,
        effects=[
            EffectSpec("compressor", {"preset": "snare_punch"}),
        ],
        mix_db=-4.0,
        pan=0.0,
    )
    score.voices["clap"].sends.append(VoiceSend(target="hall", send_db=-10.0))

    # ---- Acid bass ----
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "saw",
            "filter_topology": "diode",
            "filter_solver": "newton",
            "quality": "great",
            "transient_mode": "osc_reset",
            "cutoff_hz": 320.0,
            "resonance_q": 8.0,
            "filter_drive": 0.8,
            "filter_env_amount": 3.2,
            "filter_env_decay": 0.13,
            "keytrack": 0.2,
            "bass_compensation": 0.5,
            "attack": 0.003,
            "decay": 0.2,
            "sustain_level": 0.1,
            "release": 0.08,
            "osc2_level": 0.3,
            "osc2_waveform": "saw",
            "osc2_detune_cents": 0.0,  # driven by matrix
            "voice_dist_mode": "corrode",
            "voice_dist_drive": 0.35,
            "voice_dist_mix": 0.4,
            "voice_dist_tone": 0.55,
            "analog_jitter": 0.5,
            "voice_card_spread": 0.8,
            "osc_phase_noise": 0.1,
        },
        effects=[
            EffectSpec(
                "compressor",
                {
                    "preset": "kick_duck",
                    "sidechain_source": "kick",
                },
            ),
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 40.0, "slope_db_per_oct": 12},
                        {"kind": "low_shelf", "freq_hz": 120.0, "gain_db": 1.5},
                    ]
                },
            ),
        ],
        mix_db=-3.0,
        pan=-0.05,
        velocity_humanize=None,
    )

    # ---- Newton-ladder acid lead ----
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "saw",
            "filter_topology": "ladder",
            "filter_solver": "newton",
            "quality": "divine",
            "transient_mode": "osc_reset",
            "cutoff_hz": 800.0,
            "resonance_q": 22.0,  # matrix adds up to +14 at apex
            "filter_drive": 0.5,
            "filter_env_amount": 2.4,
            "filter_env_decay": 0.2,
            "keytrack": 0.35,
            "bass_compensation": 0.3,
            "attack": 0.004,
            "decay": 0.35,
            "sustain_level": 0.25,
            "release": 0.18,
            "osc2_level": 0.4,
            "osc2_waveform": "saw",
            "osc2_sync": True,
            "osc2_detune_cents": 7.0,
            "osc2_ring_mod": 0.22,
            "voice_dist_mode": "soft_clip",
            "voice_dist_drive": 0.2,
            "voice_dist_mix": 0.35,
            "voice_dist_tone": 0.4,
            "analog_jitter": 0.7,
            "voice_card_spread": 1.1,
            "osc_phase_noise": 0.1,
            "filter_morph": 0.0,
        },
        effects=[
            EffectSpec(
                "delay",
                {
                    "delay_seconds": 3.0 * BEAT / 4.0,  # dotted-eighth
                    "feedback": 0.35,
                    "mix": 0.22,
                },
            ),
        ],
        mix_db=-5.0,
        pan=0.18,
        velocity_humanize=None,
        sends=[VoiceSend(target="hall", send_db=-14.0)],
        automation=[
            _build_lead_cutoff_automation(),
            _build_lead_filter_morph_automation(),
        ],
    )

    # ---- VA spectralwave pad ----
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "va",
            "osc_mode": "spectralwave",
            "spectral_position": 0.35,
            "filter_topology": "k35",
            "filter1_cutoff_hz": 1600.0,
            "filter1_resonance_q": 1.4,
            "drive_amount": 0.2,
            "attack": 0.4,
            "decay": 0.8,
            "sustain_level": 0.85,
            "release": 1.2,
            "voice_card_spread": 1.8,
        },
        effects=[
            EffectSpec(
                "bbd_chorus",
                {"preset": "juno_i_plus_ii", "mix": 0.3},
            ),
            EffectSpec(
                "compressor",
                {"preset": "kick_duck", "sidechain_source": "kick"},
            ),
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 80.0, "slope_db_per_oct": 12},
                        {"kind": "high_shelf", "freq_hz": 6000.0, "gain_db": -1.5},
                    ]
                },
            ),
        ],
        mix_db=-10.0,
        pan=-0.1,
        velocity_humanize=None,
        sends=[VoiceSend(target="hall", send_db=-8.0)],
        automation=[
            _build_pad_spectral_position_automation(),
            _build_pad_mix_automation(),
        ],
    )

    # ---- Populate notes ----
    _add_kick(score)
    _add_hats(score)
    _add_clap(score)
    _add_bass(score)
    _add_lead(score)
    _add_pad(score)

    return score


PIECES: dict[str, PieceDefinition] = {
    "clock_of_7": PieceDefinition(
        name="clock_of_7",
        output_name="clock_of_7",
        build_score=build_score,
        sections=(
            PieceSection(label="Intro", start_seconds=S1_START, end_seconds=S2_START),
            PieceSection(
                label="Bass enters", start_seconds=S2_START, end_seconds=S3_START
            ),
            PieceSection(
                label="Main groove", start_seconds=S3_START, end_seconds=S4_START
            ),
            PieceSection(
                label="Breakdown", start_seconds=S4_START, end_seconds=S5_START
            ),
            PieceSection(label="Apex", start_seconds=S5_START, end_seconds=S6_START),
            PieceSection(label="Outro", start_seconds=S6_START, end_seconds=TOTAL_DUR),
        ),
    ),
}
