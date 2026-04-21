"""Additive expansion studies.

Two pieces exploring tuning-timbre interplay through the new additive
synthesis features: per-partial envelopes, noise hybrid, physical model
spectra, spectral convolution, fractal spectra, formant shaping, spectral
gravity, and stochastic flickering.

* **vowel_cathedral** — Voices morph through vowel shapes while spectral
  gravity pulls partials toward JI.  The timbre and tuning progressively
  fuse over three sections (/a/ → /o/ → /i/).

* **struck_light** — An arc from inharmonic physical-model strikes to
  harmonically fused fractal drones.  Tuning-timbre tension resolves as
  spectral gravity mediates the transition.
"""

from __future__ import annotations

from code_musics.composition import RhythmCell, line
from code_musics.humanize import VelocityHumanizeSpec
from code_musics.pieces._shared import SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score
from code_musics.spectra import (
    bar_spectrum,
    bowl_spectrum,
    formant_shape,
    fractal_spectrum,
    harmonic_spectrum,
    membrane_spectrum,
    spectral_convolve,
)

# ---------------------------------------------------------------------------
# Shared spectral materials
# ---------------------------------------------------------------------------

_CHOIR_BASE = harmonic_spectrum(n_partials=16, harmonic_rolloff=0.52)
_CHOIR_A = formant_shape(_CHOIR_BASE, 110.0, "a")
_CHOIR_O = formant_shape(_CHOIR_BASE, 110.0, "o")
_CHOIR_I = formant_shape(_CHOIR_BASE, 110.0, "i")

_MELODY_PARTIALS = formant_shape(
    harmonic_spectrum(n_partials=12, harmonic_rolloff=0.48), 110.0, "a"
)

_BOWL_PARTIALS = bowl_spectrum(n_modes=6)

_BAR_METAL = bar_spectrum(n_modes=6, material="metal")
_MEMBRANE = membrane_spectrum(n_modes=8, damping=0.3)
_BAR_BOWL_BRIDGE = spectral_convolve(
    bar_spectrum(n_modes=5, material="metal"),
    bowl_spectrum(n_modes=4),
    max_partials=16,
)
_FRACTAL_FIFTH = fractal_spectrum(
    [{"ratio": 1.0, "amp": 1.0}, {"ratio": 3 / 2, "amp": 0.7}],
    depth=3,
    max_partials=20,
)


# ===================================================================
# Study 1: Vowel Cathedral
# ===================================================================


def build_vowel_cathedral() -> Score:
    """Three vowel movements with increasing spectral gravity."""
    score = Score(
        f0_hz=110.0,
        master_effects=[
            SOFT_REVERB_EFFECT,
            EffectSpec("saturation", {"drive": 0.08, "mix": 0.2}),
        ],
    )

    # --- Bass drone ---
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "additive",
            "n_harmonics": 8,
            "harmonic_rolloff": 0.55,
            "spectral_flicker": 0.2,
            "flicker_rate_hz": 1.8,
            "flicker_correlation": 0.6,
            "attack": 2.0,
            "release": 4.0,
        },
        effects=[],
        pan=0.0,
    )

    # --- Choir (formant-shaped, gravity-pulled) ---
    score.add_voice(
        "choir",
        synth_defaults={
            "engine": "additive",
            "partials": _CHOIR_A,
            "spectral_flicker": 0.18,
            "flicker_rate_hz": 2.2,
            "flicker_correlation": 0.5,
            "attack": 1.5,
            "release": 3.0,
        },
        effects=[
            EffectSpec("chorus", {"rate_hz": 0.4, "depth_ms": 2.0, "mix": 0.2}),
        ],
        pan=-0.08,
    )

    # --- Melody ---
    score.add_voice(
        "melody",
        synth_defaults={
            "engine": "additive",
            "partials": _MELODY_PARTIALS,
            "attack": 0.08,
            "release": 2.5,
        },
        effects=[
            EffectSpec("delay", {"delay_seconds": 0.42, "feedback": 0.2, "mix": 0.15}),
        ],
        velocity_humanize=VelocityHumanizeSpec(seed=42),
        pan=0.1,
    )

    # --- Bowl bells ---
    score.add_voice(
        "bowl",
        synth_defaults={
            "engine": "additive",
            "partials": _BOWL_PARTIALS,
            "upper_partial_drift_cents": 8.0,
            "attack": 0.01,
            "decay": 0.3,
            "sustain_level": 0.15,
            "release": 2.5,
        },
        effects=[],
        pan=0.15,
    )

    # ---- Section 1: /a/ — Open (0–30s) ----

    # Bass: sustained 1/1
    score.add_note("bass", start=0.0, duration=30.0, partial=1, amp_db=-24.0)

    # Choir: sustained chord, /a/ formant, low gravity
    for partial, amp_offset in [(1, 0), (5 / 4, -2), (3 / 2, -3), (7 / 4, -5)]:
        score.add_note(
            "choir",
            start=1.0,
            duration=28.0,
            partial=partial,
            amp_db=-20.0 + amp_offset,
            synth={"partials": _CHOIR_A, "spectral_gravity": 0.2},
        )

    # Melody: ascending 7-limit phrase
    melody_1 = line(
        tones=[1, 9 / 8, 5 / 4, 3 / 2, 5 / 4, 1],
        rhythm=RhythmCell(spans=(3.5, 3.0, 3.5, 4.0, 3.5, 4.0)),
        amp_db=-14.0,
    )
    score.add_phrase("melody", melody_1, start=4.0)

    # ---- Section 2: /o/ — Round (30–60s) ----

    # Bowl bell at transition
    score.add_note("bowl", start=29.0, duration=4.0, partial=3, amp_db=-18.0)

    # Bass: shift to 3/2
    score.add_note("bass", start=30.0, duration=30.0, partial=3 / 2, amp_db=-24.0)

    # Choir: /o/ formant, medium gravity
    for partial, amp_offset in [(1, 0), (7 / 6, -2), (4 / 3, -3), (3 / 2, -4)]:
        score.add_note(
            "choir",
            start=31.0,
            duration=28.0,
            partial=partial,
            amp_db=-20.0 + amp_offset,
            synth={"partials": _CHOIR_O, "spectral_gravity": 0.4},
        )

    # Melody: septimal exploration
    melody_2 = line(
        tones=[1, 7 / 6, 4 / 3, 7 / 4, 3 / 2, 7 / 6],
        rhythm=RhythmCell(spans=(3.0, 3.5, 4.0, 3.5, 3.0, 4.5)),
        amp_db=-14.0,
    )
    score.add_phrase("melody", melody_2, start=34.0)

    # ---- Section 3: /i/ — Bright, resolved (60–92s) ----

    # Bowl bells at transition
    score.add_note("bowl", start=58.0, duration=5.0, partial=4, amp_db=-17.0)
    score.add_note("bowl", start=60.0, duration=4.0, partial=6, amp_db=-20.0)

    # Bass: back to 1/1
    score.add_note("bass", start=60.0, duration=32.0, partial=1, amp_db=-23.0)

    # Choir: /i/ formant, high gravity — tuning and timbre fuse
    for partial, amp_offset in [(1, 0), (5 / 4, -2), (3 / 2, -3), (2, -5)]:
        score.add_note(
            "choir",
            start=61.0,
            duration=28.0,
            partial=partial,
            amp_db=-19.0 + amp_offset,
            synth={
                "partials": _CHOIR_I,
                "spectral_gravity": 0.6,
                "spectral_flicker": 0.25,
            },
        )

    # Melody: resolving phrase
    melody_3 = line(
        tones=[5 / 4, 3 / 2, 7 / 4, 2, 5 / 4, 1],
        rhythm=RhythmCell(spans=(3.5, 3.0, 4.0, 4.5, 3.5, 5.0)),
        amp_db=-13.0,
    )
    score.add_phrase("melody", melody_3, start=64.0)

    # Final bowl
    score.add_note("bowl", start=85.0, duration=6.0, partial=2, amp_db=-16.0)

    return score


# ===================================================================
# Study 2: Struck Light
# ===================================================================


def build_struck_light() -> Score:
    """Arc from inharmonic strikes to harmonically fused fractal drone."""
    score = Score(
        f0_hz=220.0,
        master_effects=[SOFT_REVERB_EFFECT],
    )

    # --- Membrane hits ---
    score.add_voice(
        "membrane",
        synth_defaults={
            "engine": "additive",
            "partials": _MEMBRANE,
            "partial_decay_tilt": 1.5,
            "attack": 0.002,
            "decay": 0.15,
            "sustain_level": 0.0,
            "release": 0.1,
        },
        effects=[],
        normalize_peak_db=-6.0,
        normalize_lufs=None,
        pan=0.1,
    )

    # --- Bar melody ---
    score.add_voice(
        "bar",
        synth_defaults={
            "engine": "additive",
            "partials": _BAR_METAL,
            "spectral_gravity": 0.25,
            "gravity_rate": 1.5,
            "attack": 0.005,
            "decay": 0.3,
            "sustain_level": 0.1,
            "release": 0.6,
        },
        effects=[
            EffectSpec("delay", {"delay_seconds": 0.36, "feedback": 0.18, "mix": 0.12}),
        ],
        velocity_humanize=VelocityHumanizeSpec(seed=77),
        pan=-0.08,
    )

    # --- Bowl sustain ---
    score.add_voice(
        "bowl_sustain",
        synth_defaults={
            "engine": "additive",
            "partials": _BOWL_PARTIALS,
            "noise_amount": 0.1,
            "noise_bandwidth_hz": 35.0,
            "upper_partial_drift_cents": 6.0,
            "attack": 0.4,
            "release": 2.0,
        },
        effects=[],
        pan=-0.12,
    )

    # --- Convolved bridge ---
    score.add_voice(
        "bridge",
        synth_defaults={
            "engine": "additive",
            "partials": _BAR_BOWL_BRIDGE,
            "spectral_gravity": 0.3,
            "attack": 0.15,
            "release": 1.5,
        },
        effects=[],
        pan=0.0,
    )

    # --- Fractal drone ---
    score.add_voice(
        "fractal",
        synth_defaults={
            "engine": "additive",
            "partials": _FRACTAL_FIFTH,
            "spectral_flicker": 0.3,
            "flicker_rate_hz": 2.0,
            "flicker_correlation": 0.35,
            "unison_voices": 2,
            "detune_cents": 3.0,
            "attack": 3.0,
            "release": 5.0,
        },
        effects=[
            EffectSpec("chorus", {"rate_hz": 0.3, "depth_ms": 1.8, "mix": 0.15}),
        ],
        pan=0.0,
    )

    # ---- Section 1: Strike (0–25s) ----

    # Membrane pulse — sparse, establishing
    membrane_times = [0.5, 3.0, 5.5, 9.0, 12.0, 16.0, 20.0]
    membrane_partials = [1, 1, 3 / 2, 1, 5 / 4, 3 / 2, 1]
    for t, p in zip(membrane_times, membrane_partials, strict=True):
        score.add_note("membrane", start=t, duration=0.4, partial=p, amp_db=-12.0)

    # Bar melody — short struck notes, JI intervals on inharmonic timbre
    bar_phrase_1 = line(
        tones=[1, 3 / 2, 5 / 4, 7 / 4, 3 / 2],
        rhythm=RhythmCell(spans=(2.5, 2.0, 2.5, 3.0, 2.5), gates=0.6),
        amp_db=-14.0,
    )
    score.add_phrase("bar", bar_phrase_1, start=2.0)

    # Bowl tones between strikes
    score.add_note("bowl_sustain", start=4.0, duration=6.0, partial=1, amp_db=-22.0)
    score.add_note(
        "bowl_sustain", start=12.0, duration=6.0, partial=3 / 2, amp_db=-23.0
    )
    score.add_note(
        "bowl_sustain", start=19.0, duration=5.0, partial=5 / 4, amp_db=-22.0
    )

    # ---- Section 2: Soften (25–50s) ----

    # Bar melody — longer notes, gravity starting to pull
    bar_phrase_2 = line(
        tones=[5 / 4, 3 / 2, 7 / 4, 2, 3 / 2, 5 / 4],
        rhythm=RhythmCell(spans=(3.0, 3.5, 3.0, 3.5, 3.0, 4.0), gates=0.85),
        amp_db=-15.0,
    )
    score.add_phrase("bar", bar_phrase_2, start=26.0)

    # Convolved bridge enters
    score.add_note("bridge", start=28.0, duration=8.0, partial=1, amp_db=-21.0)
    score.add_note("bridge", start=37.0, duration=8.0, partial=3 / 2, amp_db=-22.0)
    score.add_note("bridge", start=46.0, duration=6.0, partial=5 / 4, amp_db=-21.0)

    # Bowl sustain continues
    score.add_note(
        "bowl_sustain", start=27.0, duration=8.0, partial=7 / 4, amp_db=-23.0
    )
    score.add_note("bowl_sustain", start=38.0, duration=7.0, partial=1, amp_db=-22.0)

    # Fractal drone begins to swell
    score.add_note("fractal", start=35.0, duration=15.0, partial=1, amp_db=-28.0)

    # Sparse membrane
    for t in [26.0, 33.0, 41.0]:
        score.add_note("membrane", start=t, duration=0.3, partial=1, amp_db=-15.0)

    # ---- Section 3: Fuse (50–82s) ----

    # Fractal drone swells to prominence — timbre IS the harmony
    score.add_note("fractal", start=50.0, duration=30.0, partial=1, amp_db=-18.0)
    score.add_note("fractal", start=55.0, duration=25.0, partial=3 / 2, amp_db=-22.0)

    # Bar: final phrase with strong gravity — inharmonic partials pulled toward JI
    bar_phrase_3 = line(
        tones=[3 / 2, 7 / 4, 2, 5 / 4, 1],
        rhythm=RhythmCell(spans=(4.0, 4.0, 3.5, 4.0, 5.0), gates=0.9),
        amp_db=-16.0,
    )
    score.add_phrase("bar", bar_phrase_3, start=52.0)

    # Bowl bell marks resolution
    score.add_note("bowl_sustain", start=68.0, duration=8.0, partial=2, amp_db=-19.0)

    # Bridge: one last sustained hybrid tone
    score.add_note(
        "bridge",
        start=58.0,
        duration=12.0,
        partial=1,
        amp_db=-20.0,
        synth={"spectral_gravity": 0.45},
    )

    return score


# ===================================================================
# Registration
# ===================================================================

PIECES: dict[str, PieceDefinition] = {
    "vowel_cathedral": PieceDefinition(
        name="vowel_cathedral",
        output_name="additive_01_vowel_cathedral",
        build_score=build_vowel_cathedral,
        sections=(
            PieceSection(label="/a/ — Open", start_seconds=0.0, end_seconds=30.0),
            PieceSection(label="/o/ — Round", start_seconds=30.0, end_seconds=60.0),
            PieceSection(label="/i/ — Bright", start_seconds=60.0, end_seconds=92.0),
        ),
    ),
    "struck_light": PieceDefinition(
        name="struck_light",
        output_name="additive_02_struck_light",
        build_score=build_struck_light,
        sections=(
            PieceSection(label="Strike", start_seconds=0.0, end_seconds=25.0),
            PieceSection(label="Soften", start_seconds=25.0, end_seconds=50.0),
            PieceSection(label="Fuse", start_seconds=50.0, end_seconds=82.0),
        ),
    ),
}
