"""aphotic_audition -- no-threes JI scale, chord, and tri-free patch audition.

Audition study for the *aphotic* piece (docs/plans/2026-07-07-aphotic-design.md).
Locks four decisions before composition: whether 13 stays in the working set,
whether the tri-free spectral rule audibly earns its keep, which otonal
voicings lock best, and whether the lone 5/4 "illumination" bloom lands.
"""

from __future__ import annotations

from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS, bricasti_or_reverb
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import EffectSpec, Score, VoiceSend
from code_musics.spectra import harmonic_spectrum, sieved_harmonic_spectrum

APHOTIC_F0_HZ = 46.249  # F#1: sub tonic of the eventual piece
APHOTIC_DEGREES: tuple[float, ...] = (
    1.0,
    8 / 7,
    13 / 11,
    14 / 11,
    11 / 8,
    13 / 8,
    7 / 4,
)
APHOTIC_LABELS: tuple[str, ...] = (
    "root",
    "wide_second",
    "dark_third",
    "glass_third",
    "floater",
    "dusk_sixth",
    "seventh",
)

_DEGREE_BY_LABEL: dict[str, float] = dict(
    zip(APHOTIC_LABELS, APHOTIC_DEGREES, strict=True)
)

# Audition register: two octaves above the sub tonic (F#3 region).
_AUDITION_OCTAVE = 4.0

_SECTIONS: tuple[tuple[str, float], ...] = (
    ("Scale walk", 26.0),
    ("Otonal chords", 27.0),
    ("Utonal mirror", 16.0),
    ("13 probation A/B", 24.0),
    ("Tri-free A/B", 20.0),
    ("Illumination preview", 18.0),
)


def _degree(label: str, octave: float = _AUDITION_OCTAVE) -> float:
    return _DEGREE_BY_LABEL[label] * octave


def _crystal_defaults() -> dict[str, object]:
    """Tri-free additive patch: the working crystal timbre candidate."""
    return {
        "engine": "additive",
        "partials": sieved_harmonic_spectrum(
            n_partials=16,
            downweight_factors={5: 0.35},
            harmonic_rolloff=0.72,
        ),
        "attack": 0.9,
        "release": 2.4,
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


def _drone_defaults() -> dict[str, object]:
    """Darker, slower tri-free patch for the root drone."""
    defaults = _crystal_defaults()
    defaults["partials"] = sieved_harmonic_spectrum(
        n_partials=12,
        downweight_factors={5: 0.35},
        harmonic_rolloff=0.58,
    )
    defaults["attack"] = 2.2
    defaults["release"] = 3.5
    defaults["max_partial_hz"] = 4_500.0
    return defaults


def _full_harmonic_defaults() -> dict[str, object]:
    """A/B twin of the crystal patch with an unsieved harmonic ladder."""
    defaults = _crystal_defaults()
    defaults["partials"] = harmonic_spectrum(n_partials=12, harmonic_rolloff=0.72)
    return defaults


def _slow_vibrato() -> PitchMotionSpec:
    # ~4 cents of slow shimmer: enough to feel alive, no audible wobble.
    return PitchMotionSpec.vibrato(depth_ratio=0.0023, rate_hz=3.1)


def build_aphotic_audition() -> Score:
    """Build the aphotic tuning/timbre audition score."""
    score = Score(f0_hz=APHOTIC_F0_HZ, master_effects=DEFAULT_MASTER_EFFECTS)
    score.add_send_bus(
        "cave",
        effects=[bricasti_or_reverb("1 Halls 07 Large & Dark", 1.0)],
    )
    score.add_drift_bus("breath", rate_hz=0.06, depth_cents=2.4, seed=11)

    common_voice = {
        "normalize_lufs": -24.0,
        "drift_bus": "breath",
        "drift_bus_correlation": 0.7,
    }
    score.add_voice(
        "crystal",
        synth_defaults=_crystal_defaults(),
        mix_db=-6.0,
        sends=[VoiceSend("cave", send_db=-8.0)],
        velocity_group="ensemble",
        **common_voice,
    )
    score.add_voice(
        "drone",
        synth_defaults=_drone_defaults(),
        mix_db=-9.0,
        sends=[VoiceSend("cave", send_db=-12.0)],
        velocity_group="ensemble",
        **common_voice,
    )
    score.add_voice(
        "harmonic_full",
        synth_defaults=_full_harmonic_defaults(),
        mix_db=-6.0,
        sends=[VoiceSend("cave", send_db=-8.0)],
        velocity_group="ensemble",
        **common_voice,
    )
    score.add_voice(
        "bloom",
        synth_defaults=_crystal_defaults(),
        mix_db=-10.0,
        sends=[VoiceSend("cave", send_db=-6.0)],
        velocity_group="ensemble",
        **common_voice,
    )

    section_start = {label: 0.0 for label, _ in _SECTIONS}
    cursor = 0.0
    for label, duration in _SECTIONS:
        section_start[label] = cursor
        cursor += duration

    # --- 1. Scale walk: each degree held over the root drone. -------------
    t = section_start["Scale walk"]
    score.add_note(
        "drone", start=t, duration=25.0, partial=_degree("root", 2.0), amp_db=-22.0
    )
    for index, label in enumerate(APHOTIC_LABELS):
        score.add_note(
            "crystal",
            start=t + 1.0 + index * 3.2,
            duration=2.9,
            partial=_degree(label),
            amp_db=-18.0,
            velocity=0.82 - 0.02 * index,
            pitch_motion=_slow_vibrato(),
        )
    score.add_note(
        "crystal",
        start=t + 1.0 + 7 * 3.2,
        duration=2.6,
        partial=_degree("root") * 2.0,
        amp_db=-18.5,
        velocity=0.74,
        pitch_motion=_slow_vibrato(),
    )

    # --- 2. Otonal core chords: 4:7:11, 4:7:11:13, 7:8:11:13. -------------
    t = section_start["Otonal chords"]
    chords: tuple[tuple[float, ...], ...] = (
        (4.0, 7.0, 11.0),
        (4.0, 7.0, 11.0, 13.0),
        (7.0, 8.0, 11.0, 13.0),
    )
    for chord_index, chord in enumerate(chords):
        chord_start = t + chord_index * 9.0
        for note_index, partial in enumerate(chord):
            score.add_note(
                "crystal",
                start=chord_start + 0.18 * note_index,
                duration=8.0 - 0.18 * note_index,
                partial=partial,
                amp_db=-20.0 - 1.5 * note_index,
                velocity=0.8 - 0.04 * note_index,
                pitch_motion=_slow_vibrato(),
            )

    # --- 3. Utonal mirror: /7 and /11 dyads under a guide tone. -----------
    t = section_start["Utonal mirror"]
    utonal_chords: tuple[tuple[float, ...], ...] = (
        (32.0 / 7.0, 8.0),
        (64.0 / 11.0, 8.0),
        (32.0 / 7.0, 64.0 / 11.0, 8.0),
    )
    for chord_index, chord in enumerate(utonal_chords):
        chord_start = t + chord_index * 5.0
        for note_index, partial in enumerate(chord):
            score.add_note(
                "drone",
                start=chord_start + 0.22 * note_index,
                duration=4.6,
                partial=partial,
                amp_db=-20.0 - note_index,
                velocity=0.76,
            )

    # --- 4. 13 probation A/B: same gesture with and without 13. -----------
    t = section_start["13 probation A/B"]
    pad_with_13 = (4.0, 7.0, 11.0, 13.0)
    pad_without_13 = (4.0, 7.0, 11.0)
    walk_with_13 = ("root", "dark_third", "floater", "dusk_sixth", "seventh")
    walk_without_13 = ("root", "wide_second", "floater", "seventh", "seventh")
    for variant_index, (pad, walk) in enumerate(
        ((pad_with_13, walk_with_13), (pad_without_13, walk_without_13))
    ):
        variant_start = t + variant_index * 12.0
        for note_index, partial in enumerate(pad):
            score.add_note(
                "drone",
                start=variant_start + 0.2 * note_index,
                duration=10.5,
                partial=partial,
                amp_db=-23.0 - note_index,
            )
        for step_index, label in enumerate(walk):
            octave = _AUDITION_OCTAVE * (2.0 if label == walk[-1] else 1.0)
            score.add_note(
                "crystal",
                start=variant_start + 1.0 + step_index * 1.9,
                duration=1.7,
                partial=_degree(
                    label, octave if step_index == len(walk) - 1 else _AUDITION_OCTAVE
                ),
                amp_db=-18.0,
                velocity=0.84 - 0.03 * step_index,
                pitch_motion=_slow_vibrato(),
            )

    # --- 5. Tri-free A/B: identical chord, sieved vs plain harmonic. ------
    t = section_start["Tri-free A/B"]
    ab_chord = (4.0, 7.0, 11.0)
    for variant_index, voice_name in enumerate(("harmonic_full", "crystal")):
        variant_start = t + variant_index * 10.0
        for note_index, partial in enumerate(ab_chord):
            score.add_note(
                voice_name,
                start=variant_start + 0.15 * note_index,
                duration=8.5,
                partial=partial,
                amp_db=-20.0 - 1.5 * note_index,
                velocity=0.8,
                pitch_motion=_slow_vibrato(),
            )

    # --- 6. Illumination preview: 4:7:11 held, 5/4 blooms in and away. ----
    t = section_start["Illumination preview"]
    for note_index, partial in enumerate((4.0, 7.0, 11.0)):
        score.add_note(
            "crystal",
            start=t + 0.2 * note_index,
            duration=15.5,
            partial=partial,
            amp_db=-20.0 - 1.5 * note_index,
            velocity=0.78,
            pitch_motion=_slow_vibrato(),
        )
    score.add_note(
        "bloom",
        start=t + 5.0,
        duration=8.0,
        partial=5.0,
        amp_db=-22.0,
        velocity=0.7,
        synth={"attack": 3.0, "release": 4.0},
        pitch_motion=_slow_vibrato(),
    )

    return score


def _section_boundaries() -> tuple[PieceSection, ...]:
    sections: list[PieceSection] = []
    cursor = 0.0
    for label, duration in _SECTIONS:
        sections.append(PieceSection(label, cursor, cursor + duration))
        cursor += duration
    return tuple(sections)


# --------------------------------------------------------------------------
# Space audition: the same material through four cave-space candidates.
# --------------------------------------------------------------------------

_SPACE_VARIANT_DUR = 24.0

_SPACE_VARIANTS: tuple[tuple[str, str], ...] = (
    ("fdn_20s", "FDN alone, 20 s decay"),
    ("fdn_45s", "FDN alone, 45 s decay"),
    ("serial", "close dark reverb into 30 s FDN"),
    ("parallel", "two-depth parallel sends (close + vast)"),
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


def _space_material(score: Score, voice: str, start: float) -> None:
    """~14 s of shared audition material: chord swell, strikes, drip pings."""
    for note_index, partial in enumerate((4.0, 7.0, 11.0)):
        score.add_note(
            voice,
            start=start + 0.2 * note_index,
            duration=7.5,
            partial=partial,
            amp_db=-21.0 - 1.5 * note_index,
            velocity=0.78,
            pitch_motion=_slow_vibrato(),
        )
    for strike_index, (offset, label, octave) in enumerate(
        (
            (8.6, "seventh", _AUDITION_OCTAVE),
            (10.2, "floater", _AUDITION_OCTAVE * 2.0),
            (12.0, "root", _AUDITION_OCTAVE * 2.0),
        )
    ):
        score.add_note(
            voice,
            start=start + offset,
            duration=0.9,
            partial=_degree(label, octave),
            amp_db=-17.0,
            velocity=0.86 - 0.04 * strike_index,
            synth={"attack": 0.004, "release": 1.2},
        )
    # The rest of the variant window is tail listening.


def build_aphotic_space_audition() -> Score:
    """Audition four cave-space candidates on identical tri-free material."""
    score = Score(f0_hz=APHOTIC_F0_HZ, master_effects=DEFAULT_MASTER_EFFECTS)

    close_reverb = bricasti_or_reverb(
        "1 Halls 07 Large & Dark", 1.0, room_size=0.5, damping=0.75
    )
    score.add_send_bus("fdn_20s", effects=[_fdn(20.0)])
    score.add_send_bus("fdn_45s", effects=[_fdn(45.0)])
    score.add_send_bus("serial", effects=[close_reverb, _fdn(30.0)])
    # Parallel two-depth: the variant voice feeds both of these at once.
    score.add_send_bus("close", effects=[close_reverb])
    score.add_send_bus("vast", effects=[_fdn(45.0, predelay_ms=110.0)])

    variant_sends: dict[str, list[VoiceSend]] = {
        "fdn_20s": [VoiceSend("fdn_20s", send_db=-6.0)],
        "fdn_45s": [VoiceSend("fdn_45s", send_db=-6.0)],
        "serial": [VoiceSend("serial", send_db=-6.0)],
        "parallel": [
            VoiceSend("close", send_db=-9.0),
            VoiceSend("vast", send_db=-7.0),
        ],
    }
    for variant_index, (variant_name, _description) in enumerate(_SPACE_VARIANTS):
        score.add_voice(
            f"crystal_{variant_name}",
            synth_defaults=_crystal_defaults(),
            mix_db=-7.0,
            sends=variant_sends[variant_name],
            normalize_lufs=-24.0,
        )
        _space_material(
            score,
            f"crystal_{variant_name}",
            variant_index * _SPACE_VARIANT_DUR,
        )
    return score


def _space_section_boundaries() -> tuple[PieceSection, ...]:
    return tuple(
        PieceSection(
            description,
            index * _SPACE_VARIANT_DUR,
            (index + 1) * _SPACE_VARIANT_DUR,
        )
        for index, (_name, description) in enumerate(_SPACE_VARIANTS)
    )


PIECES: dict[str, PieceDefinition] = {
    "aphotic_audition": PieceDefinition(
        name="aphotic_audition",
        output_name="aphotic_audition",
        build_score=build_aphotic_audition,
        sections=_section_boundaries(),
        study=True,
    ),
    "aphotic_space_audition": PieceDefinition(
        name="aphotic_space_audition",
        output_name="aphotic_space_audition",
        build_score=build_aphotic_space_audition,
        sections=_space_section_boundaries(),
        study=True,
    ),
}
