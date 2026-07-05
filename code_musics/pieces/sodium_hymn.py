"""Sodium Hymn — the 1-3-5-7-9-11 eikosany as a night-bus hymn.

Third and last panel of the CPS trilogy (hexany_garden -> ninth_wave ->
sodium_hymn).  The 3-of-6 Combination Product Set is the only CPS that
is its own mirror: fifteen full otonal tetrads and fifteen full utonal
tetrads, in exact duality.  The earlier panels lived in otonal light —
this one lives in the shadow half of the lattice and earns its light.

Structural gifts the piece leans on:

* An otonal tetrad O{x,y} and a utonal tetrad U{S} share two common
  tones exactly when {x,y} is inside S — the pivot mechanism.  Every
  chord change in this piece moves along those two-note hinges.
* O{9,11} sounds as 1:3:5:7 — hexany_garden's chord, note for note —
  but its pitches sit a 33/32 comma (~53 cents) off the tonic region.
  The old garden light appears here only as a detuned memory.
* The ten notes containing factor 11 are ninth_wave's 1-3-5-7-9 dekany
  transposed by 11; the otonal chords O{x,11} sound as pure 1-3-5-7-9
  harmony.  The middle of the piece walks that drowned dekany.
* Trilogy ending series: hexany_garden hung on 4:7; ninth_wave bloomed
  it into 4:5:6:7; sodium_hymn ends one rung further up the series, on
  5:7:9:11 (= O{1,3}, whose notes 1/1, 11/10, 7/5, 9/5 hold the tonic).

Home is U{1,3,5,9} — 1/1, 9/8, 3/2, 9/5, a graspable minor-seventh
shadow with no 11 in it.  The undecimal color arrives deliberately, one
pivot at a time, deepening across the arc until the fully undecimal
U{5,7,9,11} cathedral at the center.

Sound: Burial-inflected swung 2-step at 132 BPM.  No four-on-floor.
Vinyl crackle and rain, tape haze, long dark hall.  A wordless ghost
vocal (formant-morphing additive voice, gliding between eikosany tones)
is the centerpiece; bells — the trilogy's connective tissue — ring
distant and reverb-drowned, Perälä heard through a wall.  Bass is
mid-harmonic warmth, not sub pressure; F1 only at structural moments.

Tuning: eikosany over (1,3,5,7,9,11), normalized to 1*3*5, on
f0 = F2 ~ 87.31 Hz.  BPM = 132, 1 bar ~ 1.818 s, 214 bars ~ 6:29.

Form:
  bars   1- 16  S1 Rain         beatless; vinyl/rain bed; distant bells
                                outline the home shadow tetrad
  bars  17- 40  S2 First voice  the ghost vocal enters over a smeared
                                pad; home U{1,3,5,9}; at 33 the first
                                undecimal tint (U{1,3,9,11}) — the
                                voice slides 9/8 -> 11/10, a 40-cent
                                comma sigh
  bars  41- 80  S3 Two-step     the beat materialises; swung kit, bass
                                enters; utonal walk
  bars  81-104  S4 Light        pivot into O{9,11} = 1:3:5:7 — the
                                hexany-garden quote, comma-shifted;
                                brightest bells
  bars 105-136  S5 Cathedral    beat dissolves; deep shadow U{5,7,9,11},
                                fully undecimal; vocal and bells alone
                                in the long hall
  bars 137-138  S6 Blackness    two bars of near-silence
  bars 139-186  S7 Second wave  the 2-step returns evolved; dekany-walk
                                otonal chords against utonal answers;
                                melody leads every turn; F1 touches
  bars 187-214  S8 Dissolution  beat decays to ghosts; U{1,3,5,9} and
                                O{1,3} alternate around the fixed
                                pillars 1/1 and 9/5; the piece hangs
                                on 5:7:9:11, and the rain outlasts it

Composed by Claude (Fable 5), July 2026.
"""

from __future__ import annotations

from itertools import combinations
from typing import cast

from code_musics.drum_helpers import add_drum_voice, setup_drum_bus
from code_musics.humanize import EnvelopeHumanizeSpec, VelocityHumanizeSpec
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS, bricasti_or_reverb
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import EffectSpec, Score, VoiceSend
from code_musics.spectra import formant_morph, harmonic_spectrum
from code_musics.tuning import eikosany_tetrads

# ---------------------------------------------------------------------------
# Time and tuning
# ---------------------------------------------------------------------------

F0 = 87.3071  # F2
BPM = 132.0
BEAT = 60.0 / BPM
BAR = 4.0 * BEAT
SWING = 0.60  # position of the off-16th inside its eighth (0.5 = straight)

FACTORS: tuple[int, ...] = (1, 3, 5, 7, 9, 11)
_OTONAL_TETRADS, _UTONAL_TETRADS = eikosany_tetrads(FACTORS)
_PAIRS: list[tuple[int, int]] = list(combinations(FACTORS, 2))
_QUADS: list[tuple[int, int, int, int]] = list(combinations(FACTORS, 4))


def _otonal(x: int, y: int) -> tuple[float, float, float, float]:
    return _OTONAL_TETRADS[_PAIRS.index((x, y))]


def _utonal(a: int, b: int, c: int, d: int) -> tuple[float, float, float, float]:
    return _UTONAL_TETRADS[_QUADS.index((a, b, c, d))]


HOME = _utonal(1, 3, 5, 9)  # 1, 9/8, 3/2, 9/5 — the shadow home
TINT = _utonal(1, 3, 9, 11)  # 11/10, 99/80, 33/20, 9/5 — first 11
HEX_LIGHT = _otonal(9, 11)  # 33/32, 99/80, 231/160, 33/20 — sounds 1:3:5:7
DEEP = _utonal(5, 7, 9, 11)  # 33/32, 21/16, 231/160, 77/48 — cathedral
DARK_HOME = _utonal(1, 3, 5, 11)  # 1, 11/10, 11/8, 11/6 — home, 9 -> 11
FINAL_LIGHT = _otonal(1, 3)  # 1, 11/10, 7/5, 9/5 — sounds 5:7:9:11


def bar(n: float) -> float:
    """Seconds at the start of 1-indexed bar *n*."""
    return (n - 1.0) * BAR


def sw(bar_num: float, sixteenth: float) -> float:
    """Seconds at a swung sixteenth (0-15) inside 1-indexed *bar_num*.

    Even sixteenths sit on the eighth grid; odd sixteenths land at the
    SWING fraction of their eighth (0.5 would be straight time).
    """
    eighth_index = int(sixteenth) // 2
    is_off = int(sixteenth) % 2 == 1
    t = bar(bar_num) + eighth_index * (BEAT / 2.0)
    if is_off:
        t += SWING * (BEAT / 2.0)
    return t


S1_END = bar(17)
S2_END = bar(41)
S3_END = bar(81)
S4_END = bar(105)
S5_END = bar(137)
S6_END = bar(139)
S7_END = bar(187)
TOTAL_DUR = bar(215)

# ---------------------------------------------------------------------------
# Ghost vocal — formant-morphing additive voice
# ---------------------------------------------------------------------------

# Nearly flat source spectrum: the formant envelope does the sculpting
# (with a steep rolloff the partials carrying the 2-3 kHz singer formants
# are dead before the vowel weighting ever sees them).
_VOCAL_BASE = harmonic_spectrum(n_partials=26, harmonic_rolloff=0.80)


def _ghost_partials(
    freq_hz: float,
    vowels: list[str],
    morph_times: list[float] | None = None,
) -> list[dict]:
    """Formant-morph partials computed at the note's absolute frequency.

    Formants stay fixed in Hz while the pitch moves (how a real vocal
    tract works), so each note computes its own spectrum.  Envelope
    weights are normalized to the note's loudest formant peak, with a
    floor under the low partials — a real voice keeps its fundamental
    even when the formants sit far above it.
    """
    shaped = formant_morph(_VOCAL_BASE, freq_hz, list(vowels), morph_times)
    envelopes = [cast(list[dict[str, float]], p["envelope"]) for p in shaped]
    peak = max(point["value"] for envelope in envelopes for point in envelope)
    for p, envelope in zip(shaped, envelopes, strict=True):
        abs_freq = cast(float, p["ratio"]) * freq_hz
        if abs_freq < 260.0:
            floor = 0.80
        elif abs_freq < 480.0:
            floor = 0.35
        else:
            floor = 0.0
        for point in envelope:
            point["value"] = max(point["value"] / peak, floor)
    return shaped


def _sing(
    score: Score,
    *,
    start: float,
    duration: float,
    partial: float,
    vowels: list[str],
    morph_times: list[float] | None = None,
    amp_db: float = -13.0,
    velocity: float = 1.0,
    glide_from: float | None = None,
    vibrato: bool = True,
    breath: float = 0.16,
    attack: float = 0.30,
    release: float = 1.4,
    label: str | None = None,
) -> None:
    """One wordless sung note.  Glide notes approach from the previous
    pitch; sustained notes carry a late-blooming vibrato instead."""
    if glide_from is not None:
        motion: PitchMotionSpec | None = PitchMotionSpec.ratio_glide(
            start_ratio=glide_from / partial, end_ratio=1.0
        )
    elif vibrato:
        motion = PitchMotionSpec.vibrato(depth_ratio=0.0075, rate_hz=5.1)
    else:
        motion = None
    score.add_note(
        "ghost",
        start=start,
        duration=duration,
        partial=partial,
        amp_db=amp_db,
        velocity=velocity,
        pitch_motion=motion,
        label=label,
        synth={
            "partials": _ghost_partials(F0 * partial, vowels, morph_times),
            "noise_amount": breath,
            "noise_mode": "flow",
            "flow_density": 0.25,
            "noise_bandwidth_hz": 160.0,
            "spectral_flicker": 0.16,
            "flicker_rate_hz": 1.5,
            "flicker_correlation": 0.7,
            "attack": attack,
            "decay": 0.5,
            "sustain_level": 0.85,
            "release": release,
        },
    )


# ---------------------------------------------------------------------------
# Score setup
# ---------------------------------------------------------------------------


def _setup(score: Score) -> None:
    score.add_send_bus(
        "hall",
        effects=[
            bricasti_or_reverb(
                "1 Halls 07 Large & Dark",
                1.0,
                room_size=0.88,
                damping=0.5,
                lowpass_hz=6800.0,
                tilt_db=-1.5,
            )
        ],
    )
    score.add_drift_bus("night", rate_hz=0.13, depth_cents=5.0, seed=1113)

    # Weather bed: rain (pink) + vinyl crackle (sparse flow events).
    score.add_voice(
        "rain",
        synth_defaults={
            "engine": "synth_voice",
            "osc_type": None,
            "noise_type": "pink",
            "noise_level": 1.0,
            "cutoff_hz": 3400.0,
            "hpf_cutoff_hz": 180.0,
            "attack": 4.0,
            "release": 6.0,
        },
        mix_db=-14.0,
        velocity_humanize=None,
        pan=0.0,
    )
    score.add_voice(
        "crackle",
        synth_defaults={
            "engine": "synth_voice",
            "osc_type": None,
            "noise_type": "flow",
            "noise_flow_density": 0.12,
            "noise_level": 1.0,
            "hpf_cutoff_hz": 1400.0,
            "cutoff_hz": 9500.0,
            "attack": 2.0,
            "release": 3.0,
        },
        mix_db=-16.0,
        velocity_humanize=None,
        pan=0.06,
    )

    # Ghost vocal — the centerpiece.
    score.add_voice(
        "ghost",
        synth_defaults={"engine": "additive"},
        sends=[VoiceSend(target="hall", send_db=-4.0)],
        mix_db=-1.5,
        drift_bus="night",
        drift_bus_correlation=0.35,
        envelope_humanize=EnvelopeHumanizeSpec(preset="breathing_pad"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        pan=-0.04,
    )

    # Smeared shadow pad holding the tetrads.
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "additive",
            "n_harmonics": 10,
            "harmonic_rolloff": 0.5,
            "phase_disperse": 0.55,
            "spectral_flicker": 0.12,
            "flicker_rate_hz": 1.1,
            "flicker_correlation": 0.5,
            "attack": 2.8,
            "release": 5.0,
        },
        sends=[VoiceSend(target="hall", send_db=-8.0)],
        mix_db=-7.0,
        drift_bus="night",
        drift_bus_correlation=0.6,
        pan=0.05,
    )

    # Bells: distant FM, drowned in the hall.
    score.add_voice(
        "bells",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "two_op_bell",
            "release": 3.2,
        },
        sends=[VoiceSend(target="hall", send_db=-2.0)],
        mix_db=-9.0,
        envelope_humanize=EnvelopeHumanizeSpec(preset="loose_pluck"),
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        drift_bus="night",
        drift_bus_correlation=0.4,
        pan=0.18,
    )

    # Bass: mid-harmonic warmth, ducked under the kick.
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "soft_bass",
            "release": 0.4,
            # Brightness + dirt make the bass read through mids, not level.
            "brightness": 0.55,
            "dirt": 0.35,
        },
        effects=[
            EffectSpec(
                "compressor",
                {
                    "threshold_db": -21.0,
                    "ratio": 4.0,
                    "attack_ms": 2.0,
                    "release_ms": 140.0,
                    "lookahead_ms": 5.0,
                    "sidechain_source": "kick",
                    "detector_mode": "peak",
                },
            ),
        ],
        pan=0.0,
        mix_db=-7.5,
        max_polyphony=1,
        legato=True,
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        drift_bus="night",
        drift_bus_correlation=0.7,
    )

    # 2-step kit: soft tape kick, brushed snare, rim accents, swung hats.
    drum_bus = setup_drum_bus(score, style="light")
    add_drum_voice(
        score, "kick", engine="drum_voice", preset="808_tape",
        drum_bus=drum_bus, mix_db=-3.5,
    )
    add_drum_voice(
        score, "snare", engine="drum_voice", preset="brush",
        drum_bus=drum_bus, mix_db=-6.0, pan=0.05,
    )
    add_drum_voice(
        score, "rim", engine="drum_voice", preset="rim_shot",
        drum_bus=drum_bus, mix_db=-10.0, pan=-0.15,
    )
    add_drum_voice(
        score, "hat", engine="drum_voice", preset="closed_hat",
        drum_bus=drum_bus, mix_db=-11.0, choke_group="hats",
        pan=0.12,
    )
    add_drum_voice(
        score, "openhat", engine="drum_voice", preset="open_hat",
        drum_bus=drum_bus, mix_db=-13.0, choke_group="hats",
        pan=0.12,
    )


# ---------------------------------------------------------------------------
# S1 Rain (bars 1-16): weather fades in, bells outline the home shadow
# ---------------------------------------------------------------------------


def _s1_rain(score: Score) -> None:
    # Continuous weather across S1-S2 (later sections re-seed their own).
    score.add_note(
        "rain", start=0.0, duration=S2_END + 4.0, partial=1.0, amp_db=-6.0
    )
    score.add_note(
        "crackle", start=bar(2), duration=S2_END - bar(2) + 3.0, partial=1.0,
        amp_db=-6.0,
    )

    # Distant bells: home tetrad tones, high and sparse.
    for start_bar, partial, level_db in [
        (4.0, 3.0, -16.0),  # 3/2
        (7.5, 3.6, -18.0),  # 9/5
        (10.0, 4.5, -17.0),  # 9/8, octave up
        (13.0, 3.0, -19.0),
        (15.0, 7.2, -22.0),  # 9/5 two octaves up, barely there
    ]:
        score.add_note(
            "bells",
            start=bar(start_bar),
            duration=3.5,
            partial=partial,
            amp_db=level_db,
            label="bell:distant",
        )


# ---------------------------------------------------------------------------
# S2 First voice (bars 17-40): ghost vocal over the shadow pad
# ---------------------------------------------------------------------------


def _s2_first_voice(score: Score) -> None:
    # Pad: home tetrad, then the first undecimal tint at bar 33.
    for partial in HOME:
        score.add_note(
            "pad", start=bar(17), duration=bar(33) - bar(17) + 1.0,
            partial=partial, amp_db=-16.0, label="chord:U(1,3,5,9)",
        )
    for partial in TINT:
        score.add_note(
            "pad", start=bar(33), duration=S2_END - bar(33) + 2.0,
            partial=partial, amp_db=-17.0, label="chord:U(1,3,9,11)",
        )

    # Phrase A — the sigh: 9/5 held, falling through 63/40 to 3/2.
    _sing(
        score, start=bar(18), duration=2.5 * BAR, partial=3.6,
        vowels=["u", "o"], morph_times=[0.0, 0.65], amp_db=-14.0,
        velocity=0.85, attack=0.7, label="vocal:A1",
    )
    _sing(
        score, start=bar(20.5), duration=0.75 * BAR, partial=3.15,
        vowels=["o"], glide_from=3.6, amp_db=-15.0, velocity=0.8,
        label="vocal:A2",
    )
    _sing(
        score, start=bar(21.25), duration=2.25 * BAR, partial=3.0,
        vowels=["o", "a", "o"], morph_times=[0.0, 0.45, 1.0], amp_db=-13.5,
        velocity=0.95, release=2.0, label="vocal:A3",
    )

    # Phrase B — the question: rising 7/6, falling to a long 1/1.
    _sing(
        score, start=bar(25), duration=1.0 * BAR, partial=3.0,
        vowels=["u", "e"], morph_times=[0.0, 0.8], amp_db=-14.5,
        velocity=0.8, label="vocal:B1",
    )
    _sing(
        score, start=bar(26), duration=1.5 * BAR, partial=2.333333,
        vowels=["e", "a"], glide_from=3.0, amp_db=-13.5, velocity=0.9,
        label="vocal:B2",
    )
    _sing(
        score, start=bar(27.5), duration=1.0 * BAR, partial=2.25,
        vowels=["a", "o"], amp_db=-14.0, velocity=0.85, label="vocal:B3",
    )
    _sing(
        score, start=bar(28.5), duration=3.0 * BAR, partial=2.0,
        vowels=["o", "u"], morph_times=[0.0, 0.7], amp_db=-13.0,
        velocity=1.0, release=2.5, label="vocal:B4",
    )

    # Phrase C — the comma sigh: 9/8 slides down to 11/10 as the chord
    # tints undecimal; crest on 33/20 and hang on 99/80.
    _sing(
        score, start=bar(32), duration=1.0 * BAR, partial=2.25,
        vowels=["a"], amp_db=-14.0, velocity=0.85, label="vocal:C1",
    )
    _sing(
        score, start=bar(33), duration=2.0 * BAR, partial=2.2,
        vowels=["a", "o"], glide_from=2.25, amp_db=-13.0, velocity=0.95,
        release=2.0, label="vocal:C2-comma",
    )
    _sing(
        score, start=bar(35.5), duration=1.0 * BAR, partial=2.475,
        vowels=["o", "e"], amp_db=-14.0, velocity=0.85, label="vocal:C3",
    )
    _sing(
        score, start=bar(36.5), duration=1.5 * BAR, partial=3.3,
        vowels=["e", "a"], glide_from=2.475, amp_db=-13.0, velocity=1.0,
        label="vocal:C4",
    )
    _sing(
        score, start=bar(38), duration=2.5 * BAR, partial=2.475,
        vowels=["a", "o", "u"], morph_times=[0.0, 0.4, 1.0],
        glide_from=3.3, amp_db=-13.5, velocity=0.9, release=3.0,
        label="vocal:C5-hang",
    )


# ---------------------------------------------------------------------------
# S3 Two-step (bars 41-80): the beat materialises; utonal walk
# ---------------------------------------------------------------------------

# Eight-bar harmonic stations for the walk.  Every adjacent pair shares at
# least one note; the last station is TINT so its two common tones with
# O{9,11} hinge open the light at bar 81.
S3_STATIONS: list[tuple[float, tuple[float, float, float, float], float, str]] = [
    (41.0, HOME, 1.0, "U(1,3,5,9)"),
    (49.0, TINT, 1.1, "U(1,3,9,11)"),
    (57.0, _utonal(1, 3, 7, 11), 1.1, "U(1,3,7,11)"),
    (65.0, _utonal(1, 3, 7, 9), 0.9, "U(1,3,7,9)"),
    (73.0, TINT, 1.1, "U(1,3,9,11)"),
]


def _drum_hit(
    score: Score,
    voice: str,
    bar_num: float,
    sixteenth: float,
    velocity: float,
    *,
    duration: float = 0.25,
    partial: float = 1.0,
    label: str | None = None,
) -> None:
    score.add_note(
        voice,
        start=sw(bar_num, sixteenth),
        duration=duration,
        partial=partial,
        amp_db=0.0,
        velocity=velocity,
        label=label,
    )


def _s3_beat(score: Score, start_bar: int, end_bar: int, *, intro: bool) -> None:
    """The swung 2-step kit over [start_bar, end_bar) with evolution."""
    for b in range(start_bar, end_bar):
        phase = b - start_bar
        # Dropout choreography: phrase-head breaths and Objekt-style waves.
        kick_on = not intro or phase >= 2
        snare_on = not intro or phase >= 4
        if b in (56, 80):
            kick_on = False
        hats_only = b == 64
        breath = b in (71, 72)

        # Kick: 1 and the off of 3 (classic 2-step), with variants.
        if kick_on and not hats_only:
            hits = [(0, 1.0), (10, 0.88)]
            if b % 4 == 3:
                hits = [(0, 1.0), (7, 0.7), (10, 0.88)]
            elif b % 8 == 6:
                hits = [(0, 1.0), (10, 0.85), (13, 0.62)]
            if breath:
                hits = hits[:1]
            for s, vel in hits:
                _drum_hit(score, "kick", b, s, vel, duration=0.35, partial=0.5)

        # Snare: 2 and 4, brushed; ghosts breathe around the backbeat.
        if snare_on and not hats_only and not breath:
            _drum_hit(score, "snare", b, 4, 0.9, partial=2.0)
            _drum_hit(score, "snare", b, 12, 0.82, partial=2.0)
            if b % 2 == 1:
                _drum_hit(score, "snare", b, 6, 0.32, partial=2.0)
            if b % 4 == 0:
                _drum_hit(score, "snare", b, 15, 0.28, partial=2.0)

        # Rim: sporadic syncopation, more often late in the section.
        if not breath and b % 8 in (2, 5) and (not intro or phase >= 6):
            _drum_hit(score, "rim", b, 3 if b % 16 < 8 else 11, 0.5, partial=4.0)

        # Hats: swung offbeats with ghosts; open hat closes 8-bar phrases.
        hat_pattern = [(2, 0.5), (6, 0.66), (10, 0.5), (14, 0.7)]
        for s, vel in hat_pattern:
            if breath and s not in (6, 14):
                continue
            _drum_hit(score, "hat", b, s, vel, duration=0.12, partial=8.0)
        if b % 2 == 0 and not breath:
            for s in (5, 9, 13):
                _drum_hit(score, "hat", b, s, 0.24, duration=0.1, partial=8.0)
        if b % 8 == 7:
            _drum_hit(score, "openhat", b, 14, 0.55, duration=0.5, partial=8.0)


def _station_at(bar_num: float) -> tuple[tuple[float, float, float, float], float, str]:
    current = S3_STATIONS[0]
    for station in S3_STATIONS:
        if station[0] <= bar_num:
            current = station
    return current[1], current[2], current[3]


def _s3_bass(score: Score, start_bar: int, end_bar: int) -> None:
    """Sparse swung bass: root after the kick, a colour tone at the off
    of 3, resting every fourth bar — dub space, not techno drive."""
    for b in range(start_bar, end_bar):
        if b % 4 == 0 or b in (64, 71, 72, 80):
            continue
        chord, root, _label = _station_at(float(b))
        color = chord[2] / 2.0 if chord[2] / 2.0 > 0.62 else chord[2]
        score.add_note(
            "bass", start=sw(b, 2), duration=0.6 * BEAT, partial=root,
            amp_db=-6.0, velocity=0.85, label="bass:root",
        )
        score.add_note(
            "bass", start=sw(b, 11), duration=0.5 * BEAT, partial=color,
            amp_db=-8.0, velocity=0.7, label="bass:color",
        )


def _s3_two_step(score: Score) -> None:
    # Weather continues, a shade quieter under the beat.
    score.add_note(
        "rain", start=S2_END, duration=S3_END - S2_END + 3.0, partial=1.0,
        amp_db=-9.0,
    )
    score.add_note(
        "crackle", start=S2_END, duration=S3_END - S2_END + 3.0, partial=1.0,
        amp_db=-7.0,
    )

    # Pad: the walk stations.
    for i, (station_bar, chord, _root, label) in enumerate(S3_STATIONS):
        end = S3_STATIONS[i + 1][0] if i + 1 < len(S3_STATIONS) else 81.0
        for partial in chord:
            score.add_note(
                "pad", start=bar(station_bar), duration=bar(end) - bar(station_bar) + 1.5,
                partial=partial, amp_db=-19.0, label=f"chord:{label}",
            )

    _s3_beat(score, 41, 81, intro=True)
    _s3_bass(score, 45, 81)

    # Phrase D (bars 49-53, over the tint): a short call, bells answering.
    _sing(
        score, start=bar(49), duration=1.5 * BAR, partial=2.475,
        vowels=["u", "o"], amp_db=-15.0, velocity=0.8, label="vocal:D1",
    )
    _sing(
        score, start=bar(50.5), duration=2.0 * BAR, partial=2.2,
        vowels=["o", "a", "u"], morph_times=[0.0, 0.35, 1.0],
        glide_from=2.475, amp_db=-14.5, velocity=0.85, release=2.2,
        label="vocal:D2",
    )
    score.add_note(
        "bells", start=bar(53), duration=2.5, partial=3.3, amp_db=-15.0,
        velocity=0.8, label="bell:answer",
    )
    score.add_note(
        "bells", start=bar(54.5), duration=2.5, partial=2.475, amp_db=-17.0,
        velocity=0.7, label="bell:answer",
    )

    # Phrase E (bars 65-71, septimal shadow): the 11 recedes for a breath.
    _sing(
        score, start=bar(65), duration=1.5 * BAR, partial=2.8,
        vowels=["o", "e"], amp_db=-14.0, velocity=0.85, label="vocal:E1",
    )
    _sing(
        score, start=bar(66.5), duration=1.0 * BAR, partial=3.15,
        vowels=["e", "a"], amp_db=-13.5, velocity=0.95, label="vocal:E2",
    )
    _sing(
        score, start=bar(67.5), duration=1.5 * BAR, partial=2.8,
        vowels=["a", "o"], glide_from=3.15, amp_db=-14.0, velocity=0.85,
        label="vocal:E3",
    )
    _sing(
        score, start=bar(69), duration=2.5 * BAR, partial=2.1,
        vowels=["o", "u"], morph_times=[0.0, 0.6], glide_from=2.8,
        amp_db=-13.5, velocity=0.9, release=2.5, label="vocal:E4",
    )

    # The turn (bars 78-81): the voice leads the descent onto 33/20 — a
    # tone that already belongs to the coming light — while the kit thins.
    _sing(
        score, start=bar(78), duration=1.0 * BAR, partial=3.6,
        vowels=["a"], amp_db=-13.5, velocity=0.9, label="vocal:turn1",
    )
    _sing(
        score, start=bar(79), duration=1.0 * BAR, partial=3.3,
        vowels=["a", "o"], glide_from=3.6, amp_db=-13.0, velocity=0.95,
        label="vocal:turn2",
    )
    _sing(
        score, start=bar(80), duration=2.0 * BAR, partial=3.3,
        vowels=["o", "u"], morph_times=[0.0, 0.75], amp_db=-12.5,
        velocity=1.0, release=2.5, label="vocal:turn3-pivot",
    )


# ---------------------------------------------------------------------------
# S4 Light (bars 81-104): O{9,11} = 1:3:5:7 — the hexany-garden memory
# ---------------------------------------------------------------------------


def _s4_light(score: Score) -> None:
    score.add_note(
        "rain", start=S3_END, duration=bar(105) - S3_END + 3.0, partial=1.0,
        amp_db=-11.0,
    )
    score.add_note(
        "crackle", start=S3_END, duration=bar(105) - S3_END + 3.0, partial=1.0,
        amp_db=-8.0,
    )

    # Harmony: the light (81), a second dekany-flavoured light (89), then
    # the deep shadow slides underneath (97) as the beat dissolves.
    o_511 = _otonal(5, 11)  # sounds 1:3:7:9
    for chord, start_b, end_b, label in [
        (HEX_LIGHT, 81.0, 89.0, "O(9,11)=1:3:5:7"),
        (o_511, 89.0, 97.0, "O(5,11)=1:3:7:9"),
        (DEEP, 97.0, 105.0, "U(5,7,9,11)"),
    ]:
        for partial in chord:
            score.add_note(
                "pad", start=bar(start_b), duration=bar(end_b) - bar(start_b) + 1.5,
                partial=partial, amp_db=-17.5, label=f"chord:{label}",
            )

    # The beat keeps rolling through the light, thinning from 101.
    _s3_beat(score, 81, 101, intro=False)
    _s3_bass_light(score, 81, 101)

    # Structural F1 touch on the downbeat of the light.
    score.add_note(
        "bass", start=bar(81), duration=1.5 * BEAT, partial=33.0 / 64.0,
        amp_db=-4.0, velocity=1.0, label="bass:F1-light",
    )

    # Bells: the brightest passage — 1:3:5:7 arpeggio figures, the garden
    # heard through glass.  Ratios of O{9,11}, octave-lifted.
    light_tones = [p * 2.0 for p in HEX_LIGHT] + [HEX_LIGHT[0] * 4.0]
    for b, tone_idx, vel in [
        (81.5, 0, 0.9), (82.25, 1, 0.7), (83.0, 2, 0.8), (84.5, 3, 0.75),
        (85.5, 4, 0.85), (87.0, 2, 0.7), (89.5, 1, 0.8), (91.0, 3, 0.7),
        (93.0, 0, 0.75), (95.0, 2, 0.65),
    ]:
        score.add_note(
            "bells", start=bar(b), duration=3.0, partial=light_tones[tone_idx],
            amp_db=-13.0, velocity=vel, label="bell:light",
        )

    # Vocal floats up into the light — brighter vowels, rising figures.
    _sing(
        score, start=bar(83), duration=2.0 * BAR, partial=2.475,
        vowels=["o", "e"], morph_times=[0.0, 0.7], amp_db=-14.0,
        velocity=0.85, label="vocal:L1",
    )
    _sing(
        score, start=bar(85.5), duration=1.5 * BAR, partial=2.8875,
        vowels=["e", "i"], glide_from=2.475, amp_db=-13.5, velocity=0.9,
        label="vocal:L2",
    )
    _sing(
        score, start=bar(87), duration=3.0 * BAR, partial=3.3,
        vowels=["i", "e", "a"], morph_times=[0.0, 0.4, 1.0], amp_db=-13.0,
        velocity=1.0, release=2.5, label="vocal:L3",
    )
    _sing(
        score, start=bar(91), duration=1.5 * BAR, partial=4.125,
        vowels=["i", "e"], amp_db=-14.5, velocity=0.8, label="vocal:L4-crest",
    )
    _sing(
        score, start=bar(92.5), duration=2.5 * BAR, partial=3.3,
        vowels=["e", "o", "u"], morph_times=[0.0, 0.5, 1.0],
        glide_from=4.125, amp_db=-13.5, velocity=0.9, release=3.0,
        label="vocal:L5",
    )
    # As the shadow slides in at 97 the voice descends into it: the
    # melody leads the darkening, landing on 231/160 — a DEEP tone.
    _sing(
        score, start=bar(97), duration=1.5 * BAR, partial=3.3,
        vowels=["o"], amp_db=-14.0, velocity=0.85, label="vocal:L6",
    )
    _sing(
        score, start=bar(98.5), duration=1.5 * BAR, partial=2.8875,
        vowels=["o", "u"], glide_from=3.3, amp_db=-14.0, velocity=0.85,
        label="vocal:L7",
    )
    _sing(
        score, start=bar(100), duration=3.0 * BAR, partial=2.625,
        vowels=["u", "o", "u"], morph_times=[0.0, 0.5, 1.0],
        glide_from=2.8875, amp_db=-13.5, velocity=0.9, release=3.5,
        label="vocal:L8-into-shadow",
    )


def _s3_bass_light(score: Score, start_bar: int, end_bar: int) -> None:
    """Bass under the light: 33/32-rooted, same swung placement."""
    for b in range(start_bar, end_bar):
        if b % 4 == 0 or b >= 99:
            continue
        root = 33.0 / 32.0 if b < 97 else 33.0 / 32.0
        color = 231.0 / 320.0 if b % 2 == 1 else 33.0 / 40.0
        score.add_note(
            "bass", start=sw(b, 2), duration=0.6 * BEAT, partial=root,
            amp_db=-6.5, velocity=0.82, label="bass:root",
        )
        score.add_note(
            "bass", start=sw(b, 11), duration=0.5 * BEAT, partial=color,
            amp_db=-8.5, velocity=0.68, label="bass:color",
        )


# ---------------------------------------------------------------------------
# S5 Cathedral (bars 105-136) and S6 Blackness (137-138)
# ---------------------------------------------------------------------------


def _s5_cathedral(score: Score) -> None:
    # The hall opens: only rain, faint crackle, voice, bells, pad.
    score.add_note(
        "rain", start=bar(105), duration=S5_END - bar(105) + 2.0, partial=1.0,
        amp_db=-8.0,
    )
    score.add_note(
        "crackle", start=bar(105), duration=S5_END - bar(105), partial=1.0,
        amp_db=-10.0,
    )

    u_35911 = _utonal(3, 5, 9, 11)  # 33/32, 9/8, 99/80, 11/8
    for chord, start_b, end_b, label in [
        (DEEP, 105.0, 117.0, "U(5,7,9,11)"),
        (u_35911, 117.0, 125.0, "U(3,5,9,11)"),
        (DARK_HOME, 125.0, 137.0, "U(1,3,5,11)"),
    ]:
        for partial in chord:
            score.add_note(
                "pad", start=bar(start_b),
                duration=bar(end_b) - bar(start_b) + 2.0,
                partial=partial, amp_db=-15.5, label=f"chord:{label}",
            )
    # Pedal: 33/32 under the deep shadow; the true 1/1 returns only with
    # the darkened home at 125 — ground rediscovered, changed.
    score.add_note(
        "bass", start=bar(105), duration=bar(117) - bar(105), partial=33.0 / 64.0,
        amp_db=-10.0, velocity=0.6, label="pedal:33/64",
    )
    score.add_note(
        "bass", start=bar(125), duration=bar(137) - bar(125), partial=0.5,
        amp_db=-9.0, velocity=0.65, label="pedal:1/2",
    )

    # Vocal and bells in duet — the longest, most undecimal lines.
    duet = [
        # (start_bar, dur_bars, vocal_partial, vowels, bell_partial, bell_delay_bars)
        (106.0, 3.0, 2.0625, ["u", "o"], 4.125, 1.5),
        (110.0, 2.5, 2.625, ["o", "a"], 5.25, 1.0),
        (113.0, 3.0, 2.8875, ["a", "o", "u"], 5.775, 1.5),
        (118.0, 3.0, 2.25, ["u", "o"], 4.95, 1.5),
        (122.0, 2.5, 2.75, ["o", "e"], 4.5, 1.0),
        (126.0, 3.5, 2.2, ["u", "o", "a"], 5.5, 2.0),
        (130.5, 2.5, 2.75, ["a", "o"], 4.4, 1.0),
        (133.0, 3.5, 1.833333, ["o", "u"], 3.666667, 1.5),
    ]
    for start_b, dur_b, v_partial, vowels, b_partial, b_delay in duet:
        _sing(
            score, start=bar(start_b), duration=dur_b * BAR, partial=v_partial,
            vowels=vowels, amp_db=-13.0, velocity=0.9, attack=0.5,
            release=3.0, breath=0.2, label="vocal:cathedral",
        )
        score.add_note(
            "bells", start=bar(start_b + b_delay), duration=4.0,
            partial=b_partial, amp_db=-15.0, velocity=0.7,
            label="bell:cathedral",
        )


def _s6_blackness(score: Score) -> None:
    # Two bars of near-silence: the rain alone, barely.
    score.add_note(
        "rain", start=S5_END, duration=S6_END - S5_END, partial=1.0,
        amp_db=-20.0,
    )


# ---------------------------------------------------------------------------
# S7 Second wave (bars 139-186): the drowned dekany, o/u call and answer
# ---------------------------------------------------------------------------

S7_STATIONS: list[
    tuple[float, tuple[float, float, float, float], tuple[float, float, float, float], float, str]
] = [
    # (bar, otonal call, utonal answer, bass root, label)
    (139.0, _otonal(1, 11), TINT, 1.1, "O(1,11)|U(1,3,9,11)"),
    (147.0, _otonal(3, 11), _utonal(1, 3, 7, 11), 1.1, "O(3,11)|U(1,3,7,11)"),
    (155.0, _otonal(5, 11), _utonal(3, 5, 9, 11), 33.0 / 32.0, "O(5,11)|U(3,5,9,11)"),
    (163.0, _otonal(7, 11), _utonal(1, 7, 9, 11), 77.0 / 120.0 * 2.0, "O(7,11)|U(1,7,9,11)"),
    (171.0, HEX_LIGHT, DEEP, 33.0 / 32.0, "O(9,11)|U(5,7,9,11)"),
    (179.0, TINT, HOME, 1.0, "U(1,3,9,11)|U(1,3,5,9)"),
]


def _s7_second_wave(score: Score) -> None:
    score.add_note(
        "rain", start=S6_END, duration=S7_END - S6_END + 3.0, partial=1.0,
        amp_db=-10.0,
    )
    score.add_note(
        "crackle", start=S6_END, duration=S7_END - S6_END + 3.0, partial=1.0,
        amp_db=-7.0,
    )

    # o/u call-and-answer: otonal chord for the first half of each
    # 8-bar station, utonal mirror for the second half.
    for station_bar, o_chord, u_chord, _root, label in S7_STATIONS:
        for partial in o_chord:
            score.add_note(
                "pad", start=bar(station_bar), duration=4.0 * BAR + 1.0,
                partial=partial, amp_db=-18.0, label=f"chord:{label.split('|')[0]}",
            )
        for partial in u_chord:
            score.add_note(
                "pad", start=bar(station_bar + 4.0), duration=4.0 * BAR + 1.0,
                partial=partial, amp_db=-18.0, label=f"chord:{label.split('|')[1]}",
            )

    # Kit: evolved — denser hats, rim rotor, more kick variants.
    _s7_beat(score, 139, 187)
    _s7_bass(score, 139, 187)

    # Slam downbeat: F1 octave with the kick after the blackness.
    score.add_note(
        "bass", start=bar(139), duration=2.0 * BEAT, partial=0.55,
        amp_db=-3.5, velocity=1.0, label="bass:F1-slam",
    )

    # Vocal: one phrase per station, each leading the harmonic turn a
    # bar early — the voice pulls the chords, never chases them.
    phrases = [
        # (bars relative to station, partials with glides)
        (138.0, [(0.0, 1.5, 2.2, ["a", "o"], None), (1.5, 2.0, 2.475, ["o", "u"], 2.2)]),
        (146.0, [(0.0, 1.5, 2.75, ["o", "a"], None), (1.5, 2.5, 2.2, ["a", "u"], 2.75)]),
        (154.0, [(0.0, 1.5, 2.475, ["u", "o"], None), (1.5, 2.0, 2.0625, ["o", "u"], 2.475)]),
        (162.0, [(0.0, 1.5, 2.566667, ["o", "e"], None), (1.5, 2.5, 2.8875, ["e", "a"], 2.566667)]),
        (170.0, [(0.0, 2.0, 3.3, ["a", "e"], None), (2.0, 2.5, 2.8875, ["e", "o"], 3.3)]),
        (178.0, [(0.0, 1.5, 2.475, ["o", "u"], None), (1.5, 3.0, 2.25, ["u", "o", "u"], 2.475)]),
    ]
    for station_start, notes in phrases:
        for offset_b, dur_b, partial, vowels, glide_from in notes:
            _sing(
                score, start=bar(station_start + offset_b), duration=dur_b * BAR,
                partial=partial, vowels=vowels, glide_from=glide_from,
                amp_db=-13.5, velocity=0.9, release=2.0, label="vocal:wave",
            )

    # Bells thread the climax stations.
    for b, partial, vel in [
        (171.5, 4.125, 0.85), (173.0, 4.95, 0.75), (175.0, 5.775, 0.8),
        (177.0, 4.125, 0.7), (181.0, 3.6, 0.75), (184.0, 3.0, 0.7),
    ]:
        score.add_note(
            "bells", start=bar(b), duration=3.0, partial=partial,
            amp_db=-13.5, velocity=vel, label="bell:wave",
        )


def _s7_beat(score: Score, start_bar: int, end_bar: int) -> None:
    for b in range(start_bar, end_bar):
        phase = b - start_bar
        breath = b % 16 == 10  # phrase-head breaths, off the obvious grid
        drop_wave = 183 <= b < 185  # pre-dissolution thinning

        if not breath and not drop_wave:
            hits = [(0, 1.0), (10, 0.9)]
            if b % 4 == 3:
                hits = [(0, 1.0), (7, 0.72), (10, 0.9)]
            if b % 8 == 6:
                hits = [(0, 1.0), (10, 0.88), (13, 0.66)]
            if phase < 2:
                hits = [(0, 1.0)]  # let the slam ring
            for s, vel in hits:
                _drum_hit(score, "kick", b, s, vel, duration=0.35, partial=0.5)

        if not breath:
            _drum_hit(score, "snare", b, 4, 0.92, partial=2.0)
            _drum_hit(score, "snare", b, 12, 0.85, partial=2.0)
            if b % 2 == 0:
                _drum_hit(score, "snare", b, 14, 0.3, partial=2.0)

        # Rim rotor: 3-against-4 — every third sixteenth, drifting
        # across the bar line (Objekt's trick).
        rotor_phase = (b - start_bar) * 16 % 3
        if b % 8 >= 4 and not drop_wave:
            for s in range(int(rotor_phase), 16, 3):
                if s not in (0, 4, 12):
                    _drum_hit(score, "rim", b, s, 0.34, partial=4.0)

        hat_pattern = [(2, 0.55), (6, 0.7), (10, 0.55), (14, 0.72)]
        for s, vel in hat_pattern:
            _drum_hit(score, "hat", b, s, vel, duration=0.12, partial=8.0)
        for s in (1, 5, 9, 13):
            if (b + s) % 3 != 0:
                _drum_hit(score, "hat", b, s, 0.26, duration=0.1, partial=8.0)
        if b % 8 == 7:
            _drum_hit(score, "openhat", b, 14, 0.6, duration=0.5, partial=8.0)


def _s7_bass(score: Score, start_bar: int, end_bar: int) -> None:
    for b in range(start_bar, end_bar):
        if b % 8 == 7 or b in (139, 183, 184):
            continue
        station = S7_STATIONS[0]
        for s in S7_STATIONS:
            if s[0] <= b:
                station = s
        _, o_chord, u_chord, root, _label = station
        in_answer = (b - station[0]) % 8 >= 4
        chord = u_chord if in_answer else o_chord
        color = chord[1] / 2.0 if chord[1] / 2.0 > 0.62 else chord[1]
        score.add_note(
            "bass", start=sw(b, 2), duration=0.6 * BEAT, partial=root,
            amp_db=-6.0, velocity=0.85, label="bass:root",
        )
        score.add_note(
            "bass", start=sw(b, 11), duration=0.5 * BEAT, partial=color,
            amp_db=-8.0, velocity=0.7, label="bass:color",
        )
        if b % 8 == 3:
            score.add_note(
                "bass", start=sw(b, 14), duration=0.4 * BEAT, partial=root,
                amp_db=-9.0, velocity=0.6, label="bass:push",
            )


# ---------------------------------------------------------------------------
# S8 Dissolution (bars 187-214): the mirror pair, hanging on 5:7:9:11
# ---------------------------------------------------------------------------


def _s8_dissolution(score: Score) -> None:
    score.add_note(
        "rain", start=S7_END, duration=TOTAL_DUR - S7_END, partial=1.0,
        amp_db=-8.0,
    )
    score.add_note(
        "crackle", start=S7_END, duration=bar(207) - S7_END, partial=1.0,
        amp_db=-9.0,
    )

    # The alternation: HOME and FINAL_LIGHT swing around the fixed
    # pillars 1/1 and 9/5 — shadow, light, shadow, light... and the
    # last chord is the light, left hanging.
    alternation = [
        (187.0, HOME, "U(1,3,5,9)"),
        (191.0, FINAL_LIGHT, "O(1,3)=5:7:9:11"),
        (195.0, HOME, "U(1,3,5,9)"),
        (199.0, FINAL_LIGHT, "O(1,3)=5:7:9:11"),
        (203.0, HOME, "U(1,3,5,9)"),
    ]
    for start_b, chord, label in alternation:
        for partial in chord:
            score.add_note(
                "pad", start=bar(start_b), duration=4.0 * BAR + 1.5,
                partial=partial, amp_db=-16.5, label=f"chord:{label}",
            )
    # The final hanging light: longer, quieter, undamped.
    for i, partial in enumerate(FINAL_LIGHT):
        score.add_note(
            "pad", start=bar(207), duration=bar(214) - bar(207),
            partial=partial, amp_db=-16.0 - 0.5 * i,
            label="chord:O(1,3)=5:7:9:11-final",
        )

    # Beat decays: kick thins out and is gone by 195; hats ghost to 202.
    for b in range(187, 195):
        if b % 2 == 1:
            _drum_hit(score, "kick", b, 0, 0.8 - 0.06 * (b - 187), duration=0.35, partial=0.5)
        _drum_hit(score, "snare", b, 4, 0.7 - 0.05 * (b - 187), partial=2.0)
        if b < 191:
            _drum_hit(score, "snare", b, 12, 0.6, partial=2.0)
    for b in range(187, 203):
        fade = max(0.15, 0.5 - 0.025 * (b - 187))
        for s in (6, 14):
            _drum_hit(score, "hat", b, s, fade, duration=0.12, partial=8.0)

    # Bass: pillar tones only, then silence after 202.
    for b in range(187, 203, 2):
        root = 1.0 if (b - 187) // 4 % 2 == 0 else 0.9
        score.add_note(
            "bass", start=sw(b, 2), duration=1.2 * BEAT, partial=root,
            amp_db=-8.0, velocity=0.7, label="bass:pillar",
        )

    # The last vocal lines: descending onto the pillars, then one final
    # rise onto 11/10 — the undecimal tone the piece taught us to hear.
    _sing(
        score, start=bar(188), duration=2.5 * BAR, partial=3.6,
        vowels=["o", "u"], amp_db=-13.5, velocity=0.85, release=3.0,
        label="vocal:F1",
    )
    _sing(
        score, start=bar(192), duration=2.0 * BAR, partial=2.8,
        vowels=["u", "o"], glide_from=3.6, amp_db=-14.0, velocity=0.8,
        release=3.0, label="vocal:F2",
    )
    _sing(
        score, start=bar(196), duration=3.0 * BAR, partial=2.25,
        vowels=["o", "u"], amp_db=-14.0, velocity=0.8, release=3.5,
        label="vocal:F3",
    )
    _sing(
        score, start=bar(200), duration=3.0 * BAR, partial=2.0,
        vowels=["u", "o", "u"], morph_times=[0.0, 0.5, 1.0], amp_db=-13.5,
        velocity=0.85, release=4.0, label="vocal:F4-pillar",
    )
    _sing(
        score, start=bar(205), duration=4.0 * BAR, partial=2.2,
        vowels=["u", "o"], morph_times=[0.0, 0.8], glide_from=2.0,
        amp_db=-14.5, velocity=0.75, attack=1.0, release=5.0, breath=0.24,
        label="vocal:F5-last-rise",
    )

    # Final bells: the 5:7:9:11 chord tones, one by one, into the rain.
    for b, partial, vel in [
        (207.0, 2.0, 0.7), (208.5, 2.2, 0.6), (210.0, 2.8, 0.55),
        (211.5, 3.6, 0.5),
    ]:
        score.add_note(
            "bells", start=bar(b), duration=5.0, partial=partial,
            amp_db=-16.0, velocity=vel, label="bell:final",
        )


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_score() -> Score:
    score = Score(f0_hz=F0, master_effects=list(DEFAULT_MASTER_EFFECTS))
    _setup(score)
    _s1_rain(score)
    _s2_first_voice(score)
    _s3_two_step(score)
    _s4_light(score)
    _s5_cathedral(score)
    _s6_blackness(score)
    _s7_second_wave(score)
    _s8_dissolution(score)
    return score


PIECES: dict[str, PieceDefinition] = {
    "sodium_hymn": PieceDefinition(
        name="sodium_hymn",
        output_name="sodium_hymn",
        build_score=build_score,
        sections=(
            PieceSection(label="S1 rain", start_seconds=0.0, end_seconds=S1_END),
            PieceSection(
                label="S2 first voice", start_seconds=S1_END, end_seconds=S2_END
            ),
            PieceSection(
                label="S3 two-step", start_seconds=S2_END, end_seconds=S3_END
            ),
            PieceSection(label="S4 light", start_seconds=S3_END, end_seconds=S4_END),
            PieceSection(
                label="S5 cathedral", start_seconds=S4_END, end_seconds=S5_END
            ),
            PieceSection(
                label="S6 blackness", start_seconds=S5_END, end_seconds=S6_END
            ),
            PieceSection(
                label="S7 second wave", start_seconds=S6_END, end_seconds=S7_END
            ),
            PieceSection(
                label="S8 dissolution", start_seconds=S7_END, end_seconds=TOTAL_DUR
            ),
        ),
    ),
}
