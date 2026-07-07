"""aphotic -- vast dark crystalline cave in no-threes JI.

Design: docs/plans/2026-07-07-aphotic-design.md. Subgroup 2.7.11.13 with
prime 3 absent everywhere (no fifths, no fourths -- floating, nondirectional)
and prime 5 reserved for a single "illumination" bloom in Section III. Every
pitched voice uses tri-free spectra (no partial divisible by 3), so no sound
in the piece contains an acoustic fifth.

Form (~8 min): I. Dark adaptation -> II. Skeleton -> III. Illumination ->
IV. Recession. The kick and sub are one object (Autechre-style floor); the
raindrop arp is the de-literalized water drip -- stochastic onsets, mostly
reverb. Space is a parallel two-depth pair of FDN sends (close wet rock /
vast unreachable dark).
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
_BARS_I = (1, 38)
_BARS_II = (39, 77)
_BARS_III = (78, 105)
_BARS_IV = (106, 152)
_FINAL_BAR = 153  # end of form


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
    """Stochastic drip onsets: exponential-ish gaps thinned by a density ramp.

    Density 0..1 maps to a mean inter-onset gap from ~7 s down to ~0.9 s,
    so the arp reads as unpredictable water, not a grid.
    """
    rng = random.Random(seed)
    start_s = timeline.at(bar=start_bar)
    end_s = timeline.at(bar=end_bar + 1)
    onsets: list[float] = []
    cursor = start_s + rng.uniform(0.5, 2.5)
    while cursor < end_s:
        progress = (cursor - start_s) / max(end_s - start_s, 1e-9)
        density = density_start + (density_end - density_start) * progress
        mean_gap = 7.0 - 6.1 * max(0.0, min(density, 1.0))
        onsets.append(cursor)
        cursor += max(0.35, rng.expovariate(1.0 / mean_gap))
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


def _compose_section_i(score: Score, timeline: Timeline) -> None:
    """Dark adaptation: air, first drips, the floor fades in unnoticed."""
    section_start = timeline.at(bar=_BARS_I[0])
    section_end = timeline.at(bar=_BARS_II[0])

    # Air floor: overlapping long breaths across the whole section.
    air_cursor = section_start
    breath_index = 0
    while air_cursor < section_end:
        breath_dur = 22.0 + 4.0 * (breath_index % 3)
        score.add_note(
            "air",
            start=air_cursor,
            duration=breath_dur,
            partial=4.0,
            amp_db=-24.0 - (2.0 if breath_index % 2 else 0.0),
            velocity=0.6,
        )
        air_cursor += breath_dur * 0.7
        breath_index += 1

    # Raindrop arp: barely-there, thickening toward the section's end.
    rain_rng = random.Random(4611)
    for onset in _rain_onsets(
        timeline,
        start_bar=_BARS_I[0],
        end_bar=_BARS_I[1],
        density_start=0.04,
        density_end=0.3,
        seed=131,
    ):
        score.add_note(
            "rain",
            start=onset,
            duration=rain_rng.uniform(0.25, 0.6),
            partial=_rain_pitch(rain_rng),
            amp_db=-20.0 + rain_rng.uniform(-3.0, 1.5),
            velocity=rain_rng.uniform(0.45, 0.8),
        )

    # The floor arrives as if it were always there: sparse tonic kicks from
    # bar 20, one per two bars, drifting placement inside the bar.
    floor_rng = random.Random(4647)
    for bar in range(20, _BARS_I[1] + 1, 2):
        score.add_note(
            "floor",
            start=timeline.at(bar=bar, beat=floor_rng.uniform(0.0, 0.35)),
            duration=1.4,
            partial=1.0,
            amp_db=-10.0 - max(0.0, (28 - bar)) * 0.9,
            velocity=0.62 + 0.01 * max(0, bar - 24),
        )

    # Utonal bed ghosts in beneath: /7 then /11 under the guide partial 4.
    score.add_note(
        "underbed",
        start=timeline.at(bar=12),
        duration=timeline.at(bar=24) - timeline.at(bar=12),
        partial=16.0 / 7.0,
        amp_db=-26.0,
        velocity=0.55,
        pitch_motion=_slow_vibrato(depth_ratio=0.0016, rate_hz=2.2),
    )
    score.add_note(
        "underbed",
        start=timeline.at(bar=22),
        duration=timeline.at(bar=_BARS_II[0]) - timeline.at(bar=22),
        partial=32.0 / 11.0,
        amp_db=-27.0,
        velocity=0.55,
        pitch_motion=_slow_vibrato(depth_ratio=0.0016, rate_hz=2.6),
    )

    # A single crystal answer near the section's close: the cave notices.
    score.add_note(
        "crystal",
        start=timeline.at(bar=31, beat=1.4),
        duration=2.2,
        partial=_DEGREE["seventh"] * 8.0,
        amp_db=-16.0,
        velocity=0.82,
        pitch_motion=_slow_vibrato(),
    )
    score.add_note(
        "crystal",
        start=timeline.at(bar=35, beat=2.1),
        duration=2.4,
        partial=_DEGREE["floater"] * 8.0,
        amp_db=-17.5,
        velocity=0.74,
        pitch_motion=_slow_vibrato(),
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
    """Build the aphotic score (Sections II-IV land in follow-up passes)."""
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
    return score


PIECES: dict[str, PieceDefinition] = {
    "aphotic": PieceDefinition(
        name="aphotic",
        output_name="aphotic",
        build_score=build_aphotic,
        sections=_section_boundaries(_timeline()),
    ),
}
