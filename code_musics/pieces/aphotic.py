"""aphotic -- vast dark crystalline cave in no-threes JI.

Design: docs/plans/2026-07-07-aphotic-design.md. Subgroup 2.7.11.13 with
prime 3 absent everywhere (no fifths, no fourths -- floating, nondirectional)
and prime 5 rationed to three "light" events: a blink-and-miss pre-echo in
II, the main illumination bloom in III, and a fading memory in IV. Every
pitched voice uses tri-free spectra (no partial divisible by 3), so no sound
in the piece contains an acoustic fifth.

Form (~5:50 + tail): I. Dark adaptation -> II. Skeleton -> III. Illumination
-> IV. Recession. The kick and sub are one object (Autechre-style floor);
the raindrop arp is the de-literalized water drip -- stochastic onsets with
occasional burst clusters, mostly reverb. Space is a parallel two-depth pair
of FDN sends (close wet rock / vast unreachable dark).
"""

from __future__ import annotations

import random

from code_musics.drum_helpers import add_drum_voice, setup_drum_bus
from code_musics.humanize import TimingHumanizeSpec
from code_musics.meter import TempoMap, TempoPoint, Timeline
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS
from code_musics.pieces.aphotic_audition import (
    APHOTIC_DEGREES,
    APHOTIC_F0_HZ,
    APHOTIC_LABELS,
)
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import EffectSpec, Score, VoiceSend
from code_musics.spectra import sieved_harmonic_spectrum

_DEGREE: dict[str, float] = dict(zip(APHOTIC_LABELS, APHOTIC_DEGREES, strict=True))

# Section boundaries in bars (1-indexed, inclusive start).
_BARS_I = (1, 24)
_BARS_II = (25, 60)
_BARS_III = (61, 82)
_BARS_IV = (83, 110)
_FINAL_BAR = 111  # end of form


def _timeline() -> Timeline:
    """~76 BPM with slow breathing: II lifts slightly, III relaxes, IV sinks."""
    return Timeline.from_tempo_map(
        TempoMap(
            points=(
                TempoPoint(beat=0.0, bpm=75.0),
                TempoPoint(beat=(_BARS_II[0] - 1) * 4.0, bpm=77.0, curve="linear"),
                TempoPoint(beat=(_BARS_III[0] - 1) * 4.0, bpm=74.0, curve="linear"),
                TempoPoint(beat=(_BARS_IV[0] - 1) * 4.0, bpm=76.0, curve="linear"),
                TempoPoint(beat=(_FINAL_BAR - 1) * 4.0, bpm=72.0, curve="linear"),
            )
        ),
        meter=(4, 4),
    )


def _fdn(decay_s: float, **overrides: float) -> EffectSpec:
    params: dict[str, object] = {
        "decay_s": decay_s,
        "size": 0.95,
        "predelay_ms": 55.0,
        "damping_hz": 4_200.0,
        "low_decay_mult": 1.35,
        "modulation_depth": 0.3,
        "modulation_rate_hz": 0.11,
        "diffusion": 0.75,
        "mix": 1.0,
        "highpass_hz": 60.0,
        "lowpass_hz": 9_000.0,
    }
    params.update(overrides)
    return EffectSpec("fdn_reverb", params)


def _crystal_defaults() -> dict[str, object]:
    """Tri-free additive strike patch (validated in aphotic_audition)."""
    return {
        "engine": "additive",
        "partials": sieved_harmonic_spectrum(
            n_partials=16,
            downweight_factors={5: 0.35},
            harmonic_rolloff=0.72,
        ),
        "attack": 0.006,
        "decay": 0.4,
        "sustain": 0.5,
        "release": 2.8,
        "decay_power": 2.0,
        "release_power": 2.0,
        "upper_partial_drift_cents": 2.5,
        "upper_partial_drift_min_ratio": 2.0,
        "spectral_flicker": 0.1,
        "flicker_rate_hz": 1.2,
        "flicker_correlation": 0.6,
        "sigma_approximation": True,
        "max_partial_hz": 9_000.0,
        "partial_rolloff_start_hz": 3_600.0,
        "partial_rolloff_db_per_octave": 5.0,
        "partial_spatial": {
            "mode": "ratio_gradient",
            "low_pan": -0.15,
            "high_pan": 0.3,
            "width": 0.2,
            "bass_mono_below_ratio": 1.5,
        },
    }


def _rain_defaults() -> dict[str, object]:
    """Small bright drip ping: sparser sieve, fast decay, mostly send."""
    defaults = _crystal_defaults()
    defaults["partials"] = sieved_harmonic_spectrum(
        n_partials=11,
        downweight_factors={5: 0.35},
        harmonic_rolloff=0.6,
    )
    defaults["attack"] = 0.002
    defaults["decay"] = 0.12
    defaults["sustain"] = 0.25
    defaults["release"] = 0.9
    return defaults


def _swell_defaults() -> dict[str, object]:
    """Bowed-crystal swell: slow attack, dark cap, for Section III."""
    defaults = _crystal_defaults()
    defaults["attack"] = 2.6
    defaults["release"] = 4.5
    defaults["max_partial_hz"] = 7_000.0
    return defaults


def _underbed_defaults() -> dict[str, object]:
    """Dark utonal bed under the floor: few partials, low cap."""
    defaults = _crystal_defaults()
    defaults["partials"] = sieved_harmonic_spectrum(
        n_partials=8,
        downweight_factors={5: 0.35},
        harmonic_rolloff=0.55,
    )
    defaults["attack"] = 6.0
    defaults["release"] = 8.0
    defaults["max_partial_hz"] = 2_400.0
    return defaults


def _lattice_defaults() -> dict[str, object]:
    """Bell-ish melodic voice: brighter sieve, plucked envelope, 5 rationed
    harder so its melody never leaks major-third color."""
    defaults = _crystal_defaults()
    defaults["partials"] = sieved_harmonic_spectrum(
        n_partials=14,
        downweight_factors={5: 0.2},
        harmonic_rolloff=0.66,
    )
    defaults["attack"] = 0.004
    defaults["decay"] = 0.8
    defaults["sustain"] = 0.18
    defaults["release"] = 1.9
    return defaults


def _glow_defaults() -> dict[str, object]:
    """Soft harmonic pad above the floor: mid register, dark cap."""
    defaults = _crystal_defaults()
    defaults["partials"] = sieved_harmonic_spectrum(
        n_partials=10,
        downweight_factors={5: 0.35},
        harmonic_rolloff=0.6,
    )
    defaults["attack"] = 1.8
    defaults["release"] = 3.5
    defaults["max_partial_hz"] = 5_200.0
    return defaults


def _slow_vibrato(depth_ratio: float = 0.0023, rate_hz: float = 3.0) -> PitchMotionSpec:
    return PitchMotionSpec.vibrato(depth_ratio=depth_ratio, rate_hz=rate_hz)


def _add_voices(score: Score) -> None:
    drum_bus = setup_drum_bus(score, style="light")
    score.add_send_bus("close", effects=[_fdn(3.5, size=0.5, predelay_ms=18.0)])
    score.add_send_bus("vast", effects=[_fdn(45.0, predelay_ms=110.0)])
    score.add_drift_bus("breath", rate_hz=0.05, depth_cents=2.6, seed=46)

    add_drum_voice(
        score,
        "floor",
        engine="drum_voice",
        preset="808_house",
        drum_bus=drum_bus,
        send_db=-2.0,
        mix_db=-4.0,
    )
    add_drum_voice(
        score,
        "ticks",
        engine="drum_voice",
        preset="closed_hat",
        drum_bus=drum_bus,
        send_db=-6.0,
        mix_db=-16.0,
        pan=0.22,
    )
    score.add_voice(
        "underbed",
        synth_defaults=_underbed_defaults(),
        mix_db=-14.0,
        sends=[VoiceSend("close", send_db=-10.0), VoiceSend("vast", send_db=-12.0)],
        velocity_group="bed",
        normalize_lufs=-24.0,
        drift_bus="breath",
        drift_bus_correlation=0.85,
    )
    score.add_voice(
        "rain",
        synth_defaults=_rain_defaults(),
        mix_db=-13.0,
        sends=[VoiceSend("close", send_db=-8.0), VoiceSend("vast", send_db=-4.0)],
        velocity_group="drips",
        normalize_lufs=-24.0,
        drift_bus="breath",
        drift_bus_correlation=0.3,
    )
    score.add_voice(
        "crystal",
        synth_defaults=_crystal_defaults(),
        mix_db=-9.0,
        sends=[VoiceSend("close", send_db=-9.0), VoiceSend("vast", send_db=-7.0)],
        velocity_group="drips",
        normalize_lufs=-24.0,
        drift_bus="breath",
        drift_bus_correlation=0.55,
    )
    score.add_voice(
        "lattice",
        synth_defaults=_lattice_defaults(),
        mix_db=-11.0,
        pan=-0.12,
        sends=[VoiceSend("close", send_db=-9.0), VoiceSend("vast", send_db=-7.0)],
        velocity_group="drips",
        normalize_lufs=-24.0,
        drift_bus="breath",
        drift_bus_correlation=0.5,
        max_polyphony=3,
    )
    score.add_voice(
        "glow",
        synth_defaults=_glow_defaults(),
        mix_db=-15.0,
        pan=0.08,
        sends=[VoiceSend("close", send_db=-11.0), VoiceSend("vast", send_db=-9.0)],
        velocity_group="bed",
        normalize_lufs=-24.0,
        drift_bus="breath",
        drift_bus_correlation=0.85,
    )
    score.add_voice(
        "bow",
        synth_defaults=_swell_defaults(),
        mix_db=-10.0,
        sends=[VoiceSend("close", send_db=-10.0), VoiceSend("vast", send_db=-6.0)],
        velocity_group="bed",
        normalize_lufs=-24.0,
        drift_bus="breath",
        drift_bus_correlation=0.85,
    )
    score.add_voice(
        "air",
        synth_defaults={"engine": "synth_voice", "preset": "found_empty_room"},
        mix_db=-20.0,
        sends=[VoiceSend("vast", send_db=-16.0)],
        velocity_humanize=None,
        normalize_lufs=-24.0,
    )


def _rain_onsets(
    timeline: Timeline,
    *,
    start_bar: int,
    end_bar: int,
    density_start: float,
    density_end: float,
    seed: int,
) -> list[float]:
    """Stochastic drip onsets: exponential-ish gaps thinned by a density ramp,
    with occasional 2-3 drop burst clusters so the water has gesture.

    Density 0..1 maps to a mean inter-onset gap from ~6.5 s down to ~0.5 s.
    """
    rng = random.Random(seed)
    start_s = timeline.at(bar=start_bar)
    end_s = timeline.at(bar=end_bar + 1)
    onsets: list[float] = []
    cursor = start_s + rng.uniform(0.5, 2.0)
    while cursor < end_s:
        progress = (cursor - start_s) / max(end_s - start_s, 1e-9)
        density = density_start + (density_end - density_start) * progress
        mean_gap = max(0.5, 6.5 - 6.8 * max(0.0, min(density, 1.0)))
        onsets.append(cursor)
        if density > 0.25 and rng.random() < 0.28:
            burst_count = rng.choice((1, 2))
            burst_cursor = cursor
            for _ in range(burst_count):
                burst_cursor += rng.uniform(0.12, 0.32)
                if burst_cursor < end_s:
                    onsets.append(burst_cursor)
            cursor = burst_cursor
        cursor += max(0.3, rng.expovariate(1.0 / mean_gap))
    return onsets


_RAIN_POOL_RATIOS: tuple[float, ...] = (
    1.0,
    _DEGREE["seventh"],
    _DEGREE["floater"],
    _DEGREE["wide_second"],
    _DEGREE["dusk_sixth"],
    _DEGREE["dark_third"],
)
_RAIN_POOL_WEIGHTS: tuple[float, ...] = (0.28, 0.24, 0.18, 0.12, 0.1, 0.08)


def _rain_pitch(rng: random.Random) -> float:
    """Draw a drip pitch: consonance-weighted degree in a high register."""
    ratio = rng.choices(_RAIN_POOL_RATIOS, weights=_RAIN_POOL_WEIGHTS, k=1)[0]
    octave = rng.choices((8.0, 16.0, 32.0), weights=(0.35, 0.45, 0.2), k=1)[0]
    return ratio * octave


def _add_rain(
    score: Score,
    timeline: Timeline,
    *,
    start_bar: int,
    end_bar: int,
    density_start: float,
    density_end: float,
    onset_seed: int,
    pitch_seed: int,
    amp_db: float,
) -> None:
    rng = random.Random(pitch_seed)
    for onset in _rain_onsets(
        timeline,
        start_bar=start_bar,
        end_bar=end_bar,
        density_start=density_start,
        density_end=density_end,
        seed=onset_seed,
    ):
        score.add_note(
            "rain",
            start=onset,
            duration=rng.uniform(0.25, 0.6),
            partial=_rain_pitch(rng),
            amp_db=amp_db + rng.uniform(-3.0, 1.5),
            velocity=rng.uniform(0.45, 0.85),
        )


def _add_air(
    score: Score,
    timeline: Timeline,
    *,
    start_bar: int,
    end_bar: int,
    amp_db: float,
) -> None:
    cursor = timeline.at(bar=start_bar)
    section_end = timeline.at(bar=end_bar)
    breath_index = 0
    while cursor < section_end:
        breath_dur = 20.0 + 5.0 * (breath_index % 2)
        score.add_note(
            "air",
            start=cursor,
            duration=breath_dur,
            partial=4.0,
            amp_db=amp_db - (2.0 if breath_index % 2 else 0.0),
            velocity=0.55,
        )
        cursor += breath_dur * 0.7
        breath_index += 1


# The crystal motif: a falling three-note gesture (seventh -> dusk_sixth ->
# floater) that recurs through II and IV and rings as a chord in III (those
# ratios are the octave-reduced voices of partials 7, 13, 11).
_CRYSTAL_MOTIF: tuple[str, ...] = ("seventh", "dusk_sixth", "floater")


def _add_crystal_motif(
    score: Score,
    timeline: Timeline,
    *,
    bar: int,
    beat: float,
    octave: float = 8.0,
    amp_db: float = -16.5,
    velocity: float = 0.8,
    stretch: float = 1.0,
    invert: bool = False,
) -> None:
    labels = tuple(reversed(_CRYSTAL_MOTIF)) if invert else _CRYSTAL_MOTIF
    for step_index, label in enumerate(labels):
        score.add_note(
            "crystal",
            start=timeline.at(bar=bar, beat=beat + step_index * 1.1 * stretch),
            duration=2.0 * stretch,
            partial=_DEGREE[label] * octave,
            amp_db=amp_db - 1.2 * step_index,
            velocity=velocity - 0.05 * step_index,
            pitch_motion=_slow_vibrato(),
        )


# Lattice phrase: a lilting 4-bar line built from the working scale. Beats
# are relative to the phrase's start bar; durations in beats.
_LATTICE_PHRASE: tuple[tuple[float, str, float, float], ...] = (
    # (beat, label, octave_mult, duration_beats)
    (0.0, "root", 16.0, 1.6),
    (2.0, "seventh", 8.0, 1.2),
    (3.5, "floater", 16.0, 2.0),
    (6.0, "wide_second", 16.0, 1.4),
    (8.0, "seventh", 16.0, 1.8),
    (10.5, "dusk_sixth", 8.0, 1.2),
    (12.0, "root", 32.0, 3.2),
)


def _add_lattice_phrase(
    score: Score,
    timeline: Timeline,
    *,
    bar: int,
    amp_db: float = -15.0,
    velocity: float = 0.78,
    stretch: float = 1.0,
    skip_steps: frozenset[int] = frozenset(),
) -> None:
    for step_index, (beat, label, octave_mult, dur_beats) in enumerate(_LATTICE_PHRASE):
        if step_index in skip_steps:
            continue
        start = timeline.at(bar=bar, beat=beat * stretch)
        duration = timeline.at(bar=bar, beat=(beat + dur_beats) * stretch) - start
        score.add_note(
            "lattice",
            start=start,
            duration=duration,
            partial=_DEGREE[label] * octave_mult,
            amp_db=amp_db - 0.6 * step_index,
            velocity=velocity - 0.03 * step_index,
            pitch_motion=_slow_vibrato(depth_ratio=0.002, rate_hz=3.3),
        )


def _compose_section_i(score: Score, timeline: Timeline) -> None:
    """Dark adaptation: air, drips already gathering, the floor fades in."""
    _add_air(score, timeline, start_bar=_BARS_I[0], end_bar=_BARS_II[0], amp_db=-24.0)
    # The arp is present almost immediately and clearly building by the end.
    _add_rain(
        score,
        timeline,
        start_bar=_BARS_I[0],
        end_bar=_BARS_I[1],
        density_start=0.08,
        density_end=0.45,
        onset_seed=131,
        pitch_seed=4611,
        amp_db=-20.0,
    )

    # The floor arrives as if it were always there: sparse tonic kicks from
    # bar 13, one per two bars, drifting placement inside the bar.
    floor_rng = random.Random(4647)
    for bar in range(13, _BARS_I[1] + 1, 2):
        score.add_note(
            "floor",
            start=timeline.at(bar=bar, beat=floor_rng.uniform(0.0, 0.35)),
            duration=1.4,
            partial=1.0,
            amp_db=-10.0 - max(0.0, (19 - bar)) * 1.2,
            velocity=0.62 + 0.015 * max(0, bar - 15),
        )

    # Utonal bed ghosts in beneath: /7 then /11 under the guide partial 4.
    score.add_note(
        "underbed",
        start=timeline.at(bar=8),
        duration=timeline.at(bar=16) - timeline.at(bar=8),
        partial=16.0 / 7.0,
        amp_db=-26.0,
        velocity=0.55,
        pitch_motion=_slow_vibrato(depth_ratio=0.0016, rate_hz=2.2),
    )
    score.add_note(
        "underbed",
        start=timeline.at(bar=14),
        duration=timeline.at(bar=_BARS_II[0] + 1) - timeline.at(bar=14),
        partial=32.0 / 11.0,
        amp_db=-27.0,
        velocity=0.55,
        pitch_motion=_slow_vibrato(depth_ratio=0.0016, rate_hz=2.6),
    )

    # Crystal answers arriving earlier and more often: the cave notices.
    score.add_note(
        "crystal",
        start=timeline.at(bar=16, beat=1.4),
        duration=2.2,
        partial=_DEGREE["seventh"] * 8.0,
        amp_db=-16.0,
        velocity=0.82,
        pitch_motion=_slow_vibrato(),
    )
    score.add_note(
        "crystal",
        start=timeline.at(bar=19, beat=2.6),
        duration=2.2,
        partial=_DEGREE["floater"] * 8.0,
        amp_db=-17.5,
        velocity=0.74,
        pitch_motion=_slow_vibrato(),
    )
    _add_crystal_motif(score, timeline, bar=22, beat=1.0, amp_db=-17.0, velocity=0.76)


def _compose_section_ii(score: Score, timeline: Timeline) -> None:
    """Skeleton: the kick coheres out of the drips; lattice and glow enter;
    the cave answers with the motif and one blink of prime-5 light."""
    kick_rng = random.Random(4711)

    # Bars 25-32: the kick is still a drip -- placement wandering, then
    # tightening until it has settled into a pulse by bar 33.
    for bar in range(_BARS_II[0], 33):
        settle = (bar - _BARS_II[0]) / (33 - _BARS_II[0])
        wander = 0.45 * (1.0 - settle) + 0.06 * settle
        score.add_note(
            "floor",
            start=timeline.at(bar=bar, beat=kick_rng.uniform(0.0, wander)),
            duration=1.4,
            partial=1.0,
            amp_db=-8.5,
            velocity=0.66 + 0.06 * settle,
        )

    # Bars 33-60: the settled skeleton. Beat 1 every bar; a soft push on the
    # "and of 3" roughly every other bar; whole-bar dropouts keep it breathing.
    dropout_bars = {41, 49, 57}
    for bar in range(33, _BARS_II[1] + 1):
        if bar in dropout_bars:
            continue
        score.add_note(
            "floor",
            start=timeline.at(bar=bar) + kick_rng.uniform(-0.018, 0.018),
            duration=1.4,
            partial=1.0,
            amp_db=-8.0,
            velocity=kick_rng.uniform(0.72, 0.8),
        )
        if bar % 2 == 1 and kick_rng.random() < 0.7:
            score.add_note(
                "floor",
                start=timeline.at(bar=bar, beat=2.5 + kick_rng.uniform(-0.03, 0.03)),
                duration=1.1,
                partial=1.0,
                amp_db=-11.5,
                velocity=kick_rng.uniform(0.5, 0.6),
            )

    # Ticks: near-subliminal metallic offbeats from bar 33.
    tick_rng = random.Random(4723)
    for bar in range(33, _BARS_II[1] + 1):
        for eighth in range(8):
            if eighth % 2 == 0:
                continue
            if tick_rng.random() > 0.55:
                continue
            score.add_note(
                "ticks",
                start=timeline.at(bar=bar, beat=eighth * 0.5),
                duration=0.1,
                partial=64.0,
                amp_db=-26.0 + tick_rng.uniform(-2.0, 1.0),
                velocity=tick_rng.uniform(0.35, 0.55),
            )

    # Rain reaches full arp intensity quickly and holds it.
    _add_rain(
        score,
        timeline,
        start_bar=_BARS_II[0],
        end_bar=_BARS_II[1],
        density_start=0.5,
        density_end=0.72,
        onset_seed=137,
        pitch_seed=4733,
        amp_db=-18.5,
    )
    _add_air(score, timeline, start_bar=_BARS_II[0], end_bar=_BARS_III[0], amp_db=-27.0)

    # Underbed: the utonal pair as one dark chord, then a septimal shift that
    # leans the harmony toward the illumination to come.
    for partial, amp_db in ((16.0 / 7.0, -25.0), (32.0 / 11.0, -27.0)):
        score.add_note(
            "underbed",
            start=timeline.at(bar=27),
            duration=timeline.at(bar=46) - timeline.at(bar=27),
            partial=partial,
            amp_db=amp_db,
            velocity=0.57,
            pitch_motion=_slow_vibrato(depth_ratio=0.0016, rate_hz=2.4),
        )
    score.add_note(
        "underbed",
        start=timeline.at(bar=47),
        duration=timeline.at(bar=_BARS_III[0]) - timeline.at(bar=47),
        partial=7.0 / 4.0,
        amp_db=-25.0,
        velocity=0.6,
        pitch_motion=PitchMotionSpec.ratio_glide(start_ratio=0.997, end_ratio=1.0),
    )

    # Glow pad: a soft otonal bed above the floor, changing once mid-section.
    for chord, start_bar, end_bar in (
        ((7.0, 8.0, 11.0), 33, 46),
        ((7.0, 11.0, 13.0), 47, 60),
    ):
        for note_index, partial in enumerate(chord):
            score.add_note(
                "glow",
                start=timeline.at(bar=start_bar) + 0.4 * note_index,
                duration=timeline.at(bar=end_bar + 1) - timeline.at(bar=start_bar),
                partial=partial,
                amp_db=-24.0 - 1.5 * note_index,
                velocity=0.6,
                pitch_motion=_slow_vibrato(depth_ratio=0.0015, rate_hz=2.0),
            )

    # Lattice: the melodic line enters once the skeleton has settled,
    # restating with small variations.
    _add_lattice_phrase(score, timeline, bar=37)
    _add_lattice_phrase(
        score, timeline, bar=45, amp_db=-16.0, skip_steps=frozenset({1})
    )
    _add_lattice_phrase(score, timeline, bar=53, velocity=0.82)

    # The crystal motif recurs, answering the lattice from the dark.
    _add_crystal_motif(score, timeline, bar=30, beat=1.5, amp_db=-17.5, velocity=0.74)
    _add_crystal_motif(score, timeline, bar=42, beat=0.5)
    _add_crystal_motif(
        score, timeline, bar=50, beat=2.0, octave=16.0, amp_db=-19.0, velocity=0.7
    )
    _add_crystal_motif(score, timeline, bar=56, beat=0.5, invert=True)

    # One blink of prime-5 light, high and gone -- a pre-echo of III.
    score.add_note(
        "crystal",
        start=timeline.at(bar=51, beat=2.2),
        duration=1.6,
        partial=20.0,
        amp_db=-22.0,
        velocity=0.6,
        pitch_motion=_slow_vibrato(),
    )

    # Bow foreshadow as the beat prepares to leave.
    score.add_note(
        "bow",
        start=timeline.at(bar=58),
        duration=timeline.at(bar=_BARS_III[0] + 1) - timeline.at(bar=58),
        partial=7.0,
        amp_db=-24.0,
        velocity=0.6,
        pitch_motion=_slow_vibrato(depth_ratio=0.0018, rate_hz=2.4),
    )


def _compose_section_iii(score: Score, timeline: Timeline) -> None:
    """Illumination: the beat dissolves, the crystals ring together, and the
    5 blooms inside the chord and recedes -- the motif frozen into a sonority."""
    kick_rng = random.Random(4787)

    # The beat lets go: bars 61-64 thin out, then silence from 65.
    for bar, keep_chance in ((61, 1.0), (62, 0.7), (63, 0.45), (64, 0.3)):
        if kick_rng.random() > keep_chance:
            continue
        score.add_note(
            "floor",
            start=timeline.at(bar=bar) + kick_rng.uniform(-0.02, 0.05),
            duration=1.4,
            partial=1.0,
            amp_db=-9.5 - (bar - 61) * 1.4,
            velocity=0.7 - 0.06 * (bar - 61),
        )

    # Rain almost stops: the cave holds its breath.
    _add_rain(
        score,
        timeline,
        start_bar=_BARS_III[0],
        end_bar=_BARS_III[1],
        density_start=0.45,
        density_end=0.08,
        onset_seed=139,
        pitch_seed=4789,
        amp_db=-21.0,
    )

    # Staggered entries assemble the standing chord: 4, then 7, 11, 13.
    chord_end = timeline.at(bar=79)
    for entry_bar, partial, amp_db, vibrato_rate in (
        (62, 4.0, -20.0, 2.0),
        (64, 7.0, -21.5, 2.3),
        (66, 11.0, -23.0, 2.7),
        (68, 13.0, -25.5, 3.1),
    ):
        entry_start = timeline.at(bar=entry_bar)
        score.add_note(
            "bow",
            start=entry_start,
            duration=chord_end - entry_start,
            partial=partial,
            amp_db=amp_db,
            velocity=0.66,
            pitch_motion=_slow_vibrato(depth_ratio=0.0019, rate_hz=vibrato_rate),
        )

    # The illumination: partial 5 -- a major third of light inside the chord,
    # swelling in and gone again. Quiet arrival, not a climax.
    bloom_start = timeline.at(bar=70)
    score.add_note(
        "bow",
        start=bloom_start,
        duration=timeline.at(bar=77) - bloom_start,
        partial=5.0,
        amp_db=-19.0,
        velocity=0.74,
        synth={"attack": 4.5, "release": 6.5},
        pitch_motion=_slow_vibrato(depth_ratio=0.0026, rate_hz=3.4),
    )
    # A lattice echo of the light: 5/4-colored dyad answering, once.
    score.add_note(
        "lattice",
        start=timeline.at(bar=74, beat=2.0),
        duration=3.0,
        partial=10.0,
        amp_db=-20.0,
        velocity=0.62,
        pitch_motion=_slow_vibrato(depth_ratio=0.002, rate_hz=3.0),
    )
    score.add_note(
        "lattice",
        start=timeline.at(bar=75, beat=1.0),
        duration=3.0,
        partial=_DEGREE["seventh"] * 8.0,
        amp_db=-19.0,
        velocity=0.6,
        pitch_motion=_slow_vibrato(depth_ratio=0.002, rate_hz=3.0),
    )

    _add_air(score, timeline, start_bar=_BARS_III[0], end_bar=_BARS_IV[0], amp_db=-28.0)


def _compose_section_iv(score: Score, timeline: Timeline) -> None:
    """Recession: the skeleton returns thinner with the glow's memory of the
    light, then everything unravels until the dark floor remains."""
    kick_rng = random.Random(4801)

    # The beat returns sparser -- every other bar -- then loses coherence in
    # reverse (mirroring its arrival) and is gone by bar 105.
    for bar in range(83, 95, 2):
        score.add_note(
            "floor",
            start=timeline.at(bar=bar) + kick_rng.uniform(-0.02, 0.02),
            duration=1.4,
            partial=1.0,
            amp_db=-9.0,
            velocity=kick_rng.uniform(0.68, 0.76),
        )
    for bar in range(95, 105, 2):
        unravel = (bar - 95) / (105 - 95)
        if kick_rng.random() < 0.3 * unravel:
            continue
        score.add_note(
            "floor",
            start=timeline.at(bar=bar, beat=kick_rng.uniform(0.0, 0.5 * unravel)),
            duration=1.4,
            partial=1.0,
            amp_db=-9.5 - 4.0 * unravel,
            velocity=0.7 - 0.15 * unravel,
        )

    # Ticks make a brief, thinner return while the kick is steady.
    tick_rng = random.Random(4813)
    for bar in range(84, 97):
        for eighth in (1, 3, 5, 7):
            if tick_rng.random() > 0.3:
                continue
            score.add_note(
                "ticks",
                start=timeline.at(bar=bar, beat=eighth * 0.5),
                duration=0.1,
                partial=64.0,
                amp_db=-28.0 + tick_rng.uniform(-2.0, 1.0),
                velocity=tick_rng.uniform(0.3, 0.45),
            )

    # Rain thins toward single distant drops.
    _add_rain(
        score,
        timeline,
        start_bar=_BARS_IV[0],
        end_bar=_BARS_IV[1] - 1,
        density_start=0.45,
        density_end=0.06,
        onset_seed=149,
        pitch_seed=4817,
        amp_db=-20.5,
    )

    # The utonal bed returns for the middle of the recession, then lets go.
    score.add_note(
        "underbed",
        start=timeline.at(bar=85),
        duration=timeline.at(bar=100) - timeline.at(bar=85),
        partial=16.0 / 7.0,
        amp_db=-26.0,
        velocity=0.55,
        pitch_motion=_slow_vibrato(depth_ratio=0.0016, rate_hz=2.2),
    )
    score.add_note(
        "underbed",
        start=timeline.at(bar=88),
        duration=timeline.at(bar=100) - timeline.at(bar=88),
        partial=32.0 / 11.0,
        amp_db=-28.0,
        velocity=0.52,
        pitch_motion=_slow_vibrato(depth_ratio=0.0016, rate_hz=2.6),
    )

    # Glow: the memory of the light -- the II chord returns carrying a soft
    # partial 10 (the 5, two octaves up) that fades before the bed does.
    for partial, amp_db, end_bar in (
        (7.0, -24.0, 100),
        (11.0, -25.5, 100),
        (10.0, -28.0, 93),
    ):
        score.add_note(
            "glow",
            start=timeline.at(bar=86),
            duration=timeline.at(bar=end_bar) - timeline.at(bar=86),
            partial=partial,
            amp_db=amp_db,
            velocity=0.55,
            pitch_motion=_slow_vibrato(depth_ratio=0.0015, rate_hz=2.0),
        )

    # The motif remembers itself, each return further away; the lattice
    # answers once, slower and thinner.
    _add_crystal_motif(score, timeline, bar=87, beat=1.0, amp_db=-17.5, velocity=0.74)
    _add_lattice_phrase(
        score,
        timeline,
        bar=91,
        amp_db=-18.0,
        velocity=0.66,
        stretch=2.0,
        skip_steps=frozenset({1, 3, 5}),
    )
    _add_crystal_motif(
        score, timeline, bar=98, beat=0.5, stretch=1.6, amp_db=-20.0, velocity=0.64
    )
    score.add_note(
        "crystal",
        start=timeline.at(bar=104, beat=1.2),
        duration=3.2,
        partial=_DEGREE["seventh"] * 8.0,
        amp_db=-20.5,
        velocity=0.6,
        pitch_motion=_slow_vibrato(),
    )

    # Air breathes to the end; one last drip and one last far crystal, then
    # only the vast tail.
    _add_air(score, timeline, start_bar=_BARS_IV[0], end_bar=_FINAL_BAR, amp_db=-27.0)
    score.add_note(
        "crystal",
        start=timeline.at(bar=107, beat=2.0),
        duration=3.5,
        partial=_DEGREE["seventh"] * 16.0,
        amp_db=-24.0,
        velocity=0.5,
        pitch_motion=_slow_vibrato(),
    )
    score.add_note(
        "rain",
        start=timeline.at(bar=109, beat=1.0),
        duration=0.6,
        partial=_DEGREE["floater"] * 16.0,
        amp_db=-24.0,
        velocity=0.45,
    )


def _section_boundaries(timeline: Timeline) -> tuple[PieceSection, ...]:
    return (
        PieceSection(
            "I. Dark adaptation",
            timeline.at(bar=_BARS_I[0]),
            timeline.at(bar=_BARS_II[0]),
        ),
        PieceSection(
            "II. Skeleton",
            timeline.at(bar=_BARS_II[0]),
            timeline.at(bar=_BARS_III[0]),
        ),
        PieceSection(
            "III. Illumination",
            timeline.at(bar=_BARS_III[0]),
            timeline.at(bar=_BARS_IV[0]),
        ),
        PieceSection(
            "IV. Recession",
            timeline.at(bar=_BARS_IV[0]),
            timeline.at(bar=_FINAL_BAR),
        ),
    )


def build_aphotic() -> Score:
    """Build the aphotic score."""
    timeline = _timeline()
    score = Score(
        f0_hz=APHOTIC_F0_HZ,
        master_effects=DEFAULT_MASTER_EFFECTS,
        timing_humanize=TimingHumanizeSpec(preset="chamber", seed=46),
        effect_tail_seconds=50.0,
        timeline=timeline,
    )
    _add_voices(score)
    _compose_section_i(score, timeline)
    _compose_section_ii(score, timeline)
    _compose_section_iii(score, timeline)
    _compose_section_iv(score, timeline)
    return score


PIECES: dict[str, PieceDefinition] = {
    "aphotic": PieceDefinition(
        name="aphotic",
        output_name="aphotic",
        build_score=build_aphotic,
        sections=_section_boundaries(_timeline()),
    ),
}
