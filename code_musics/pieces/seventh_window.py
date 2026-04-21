"""Seventh Window — 7-limit JI contemplation for three Vital wavetable voices.

Form: Open → Bloom → Fade.  ~2:00 duration.

A contemplative piece in 7-limit Just Intonation that showcases Vital's
wavetable engine through three distinct timbres: a shimmering unison pad,
a glass-bell melodic voice, and a warm filtered bass.  Each voice uses
different oscillator, filter, envelope, and effect configurations so the
three instruments sound genuinely different despite sharing the same
underlying wavetable engine.

Harmonic language: 7-limit JI centred on A2 (110 Hz), with otonal,
subharmonic, and septimal-suspended shapes.  Melody is drawn from
harmonic-series intervals and developed through transposition, extension,
and rhythmic transformation.

Section I  — Open  (0:00–0:30): bass and pad emerge, spacious, harmonic world
Section II — Bloom (0:30–1:20): bell melody enters, progression develops,
                                voices interact, texture thickens
Section III— Fade  (1:20–2:00): melody simplifies, texture thins, gentle ending

Voice layout:
  - pad:   Vital, 4 unison voices, chorus + reverb, slow envelopes
  - bell:  Vital, clean osc, filtered, short envelopes, touch of delay
  - bass:  Vital, simple waveform, distortion warmth, filtered
"""

from __future__ import annotations

from code_musics.composition import HarmonicContext, ratio_line
from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    VelocityHumanizeSpec,
)
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

F0: float = 110.0  # A2

# Section boundaries (seconds)
OPEN_START: float = 0.0
BLOOM_START: float = 30.0
FADE_START: float = 80.0
PIECE_END: float = 120.0

# ---------------------------------------------------------------------------
# Chord vocabulary (ratios relative to f0)
# ---------------------------------------------------------------------------

HOME = [1, 5 / 4, 3 / 2, 7 / 4]  # otonal tetrad — bright, warm
COMMA = [1, 9 / 7, 3 / 2, 12 / 7]  # septimal recoloring of the same roles
DARK = [1, 7 / 6, 4 / 3, 8 / 5]  # utonal — hollow, introspective
SUSPENDED = [1, 8 / 7, 3 / 2, 7 / 4]  # septimal sus — yearning
RESOLVE = [1, 5 / 4, 3 / 2, 2]  # open octave — resolution

# Wide voicings (root dropped an octave for spaciousness)
HOME_WIDE = [1 / 2, 5 / 4, 3 / 2, 7 / 4]
COMMA_WIDE = [1 / 2, 9 / 7, 3 / 2, 12 / 7]
DARK_WIDE = [1 / 2, 7 / 6, 4 / 3, 8 / 5]
SUSPENDED_WIDE = [1 / 2, 8 / 7, 3 / 2, 7 / 4]

# ---------------------------------------------------------------------------
# Vital voice configurations
# ---------------------------------------------------------------------------

PAD_VITAL_PARAMS: dict[str, float] = {
    # Osc 1: wavetable pad with unison shimmer
    "oscillator_1_switch": 1.0,
    "oscillator_1_wave_frame": 0.25,
    "oscillator_1_unison_voices": 0.2,  # ~4 unison voices
    "oscillator_1_unison_detune": 0.3,
    "oscillator_1_stereo_spread": 0.85,
    "oscillator_1_level": 0.65,
    # Amp envelope: slow bloom, long release
    "envelope_1_attack": 0.35,
    "envelope_1_decay": 0.5,
    "envelope_1_sustain": 0.85,
    "envelope_1_release": 0.45,
    # Effects: chorus for motion, reverb for space
    "chorus_switch": 1.0,
    "chorus_mix": 0.25,
    "chorus_voices": 1.0,
    "chorus_frequency": 0.25,
    "chorus_mod_depth": 0.4,
    "reverb_switch": 1.0,
    "reverb_mix": 0.3,
    "reverb_decay_time": 0.65,
    "reverb_size": 0.7,
    # Polyphony for chords
    "polyphony": 0.35,  # ~11 voices
}

BELL_VITAL_PARAMS: dict[str, float] = {
    # Osc 1: brighter wavetable position, clean
    "oscillator_1_switch": 1.0,
    "oscillator_1_wave_frame": 0.55,
    "oscillator_1_unison_voices": 0.0,  # single voice, clean
    "oscillator_1_level": 0.6,
    # Filter 1: gentle lowpass for warmth
    "filter_1_switch": 1.0,
    "filter_1_cutoff": 0.55,
    "filter_1_resonance": 0.35,
    "filter_1_model": 0.0,
    "filter_1_style": 0.0,
    # Amp envelope: plucky attack, moderate decay
    "envelope_1_attack": 0.01,
    "envelope_1_decay": 0.38,
    "envelope_1_sustain": 0.45,
    "envelope_1_release": 0.3,
    # Delay for spatial depth
    "delay_switch": 1.0,
    "delay_mix": 0.15,
    "delay_feedback": 0.35,
    "delay_tempo": 0.55,
    # Touch of reverb
    "reverb_switch": 1.0,
    "reverb_mix": 0.2,
    "reverb_decay_time": 0.5,
    "reverb_size": 0.55,
    # Polyphony
    "polyphony": 0.25,
}

BASS_VITAL_PARAMS: dict[str, float] = {
    # Osc 1: simple waveform, low wavetable position
    "oscillator_1_switch": 1.0,
    "oscillator_1_wave_frame": 0.08,
    "oscillator_1_unison_voices": 0.0,
    "oscillator_1_level": 0.7,
    # Filter 1: lowpass for warmth
    "filter_1_switch": 1.0,
    "filter_1_cutoff": 0.35,
    "filter_1_resonance": 0.2,
    "filter_1_model": 0.0,
    "filter_1_style": 0.0,
    # Distortion for analog warmth
    "distortion_switch": 1.0,
    "distortion_drive": 0.35,
    "distortion_mix": 0.5,
    # Amp envelope: smooth onset, clean release
    "envelope_1_attack": 0.08,
    "envelope_1_decay": 0.35,
    "envelope_1_sustain": 0.8,
    "envelope_1_release": 0.25,
    # Minimal reverb
    "reverb_switch": 1.0,
    "reverb_mix": 0.1,
    "reverb_decay_time": 0.35,
    "reverb_size": 0.4,
    # Polyphony
    "polyphony": 0.13,  # ~4 voices
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chord_notes(
    voice: str,
    score: Score,
    ratios: list[float],
    start: float,
    duration: float,
    amp_db: float = -6.0,
    velocities: list[float] | None = None,
) -> None:
    """Add a chord (list of ratios) to a voice."""
    for i, ratio in enumerate(ratios):
        vel = velocities[i] if velocities else 0.8
        score.add_note(
            voice,
            partial=ratio,
            start=start,
            duration=duration,
            amp_db=amp_db,
            velocity=vel,
        )


def _melody_context(root_ratio: float = 1.0) -> HarmonicContext:
    """Create a harmonic context at a given root ratio relative to f0."""
    return HarmonicContext(tonic=F0 * root_ratio)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_open(score: Score) -> None:
    """Section I — Open (0:00–0:30): bass and pad emerge."""
    # Bass enters alone on the root, long sustained note
    score.add_note(
        "bass", partial=1 / 2, start=0.0, duration=12.0, amp_db=-8.0, velocity=0.7
    )

    # Pad fades in with the home chord (staggered entry for organic feel)
    for i, ratio in enumerate(HOME_WIDE):
        score.add_note(
            "pad",
            partial=ratio,
            start=2.0 + i * 0.4,
            duration=13.0 - i * 0.3,
            amp_db=-10.0,
            velocity=0.65 + i * 0.04,
        )

    # Bass moves to 3/2 briefly
    score.add_note(
        "bass", partial=3 / 4, start=13.0, duration=4.0, amp_db=-9.0, velocity=0.65
    )

    # Pad shifts to SUSPENDED — the yearning quality hints at what's coming
    for i, ratio in enumerate(SUSPENDED_WIDE):
        score.add_note(
            "pad",
            partial=ratio,
            start=16.0 + i * 0.3,
            duration=10.0,
            amp_db=-10.0,
            velocity=0.6 + i * 0.03,
        )

    # Bass returns to root
    score.add_note(
        "bass", partial=1 / 2, start=18.0, duration=8.0, amp_db=-8.0, velocity=0.7
    )

    # Pad resolves to HOME as we approach the bloom
    for i, ratio in enumerate(HOME_WIDE):
        score.add_note(
            "pad",
            partial=ratio,
            start=26.5 + i * 0.2,
            duration=6.0,
            amp_db=-9.0,
            velocity=0.68,
        )


def _build_bloom(score: Score) -> None:
    """Section II — Bloom (0:30–1:20): melody enters, progression develops."""
    t = BLOOM_START

    # ---- Melodic motifs ----
    # Primary motif: ascending through the harmonic series
    motif_a = ratio_line(
        tones=[1, 9 / 8, 5 / 4, 3 / 2],
        rhythm=[1.2, 0.8, 1.0, 1.5],
        context=_melody_context(),
        amp_db=-5.0,
    )

    # Motif B: reaching higher with septimal color
    motif_b = ratio_line(
        tones=[5 / 4, 11 / 8, 3 / 2, 7 / 4],
        rhythm=[0.8, 1.0, 0.6, 2.0],
        context=_melody_context(),
        amp_db=-5.0,
    )

    # Motif C: descending, more introspective
    motif_c = ratio_line(
        tones=[7 / 4, 3 / 2, 9 / 7, 1],
        rhythm=[1.0, 0.8, 1.2, 1.8],
        context=_melody_context(),
        amp_db=-6.0,
    )

    # Extended motif: motif_a developed with added tones
    motif_a_ext = ratio_line(
        tones=[1, 9 / 8, 5 / 4, 11 / 8, 3 / 2, 7 / 4, 2],
        rhythm=[0.8, 0.6, 0.7, 0.5, 0.8, 1.0, 1.8],
        context=_melody_context(),
        amp_db=-5.0,
    )

    # Motif D: high, sparse, bell-like
    motif_d = ratio_line(
        tones=[2, 7 / 4, 3 / 2, 5 / 4],
        rhythm=[1.5, 1.2, 0.8, 2.5],
        context=_melody_context(),
        amp_db=-7.0,
    )

    # ---- Phrase 1: HOME → COMMA (t+0 to t+9) ----
    # Bell enters with motif A
    score.add_phrase("bell", motif_a, start=t + 0.5)
    # Pad holds HOME then transitions
    _chord_notes("pad", score, HOME_WIDE, start=t, duration=8.5, amp_db=-10.0)
    score.add_note(
        "bass", partial=1 / 2, start=t, duration=8.0, amp_db=-8.0, velocity=0.7
    )

    # ---- Phrase 2: COMMA (t+9 to t+17) ----
    score.add_phrase("bell", motif_b, start=t + 9.5)
    _chord_notes("pad", score, COMMA_WIDE, start=t + 9.0, duration=7.5, amp_db=-10.0)
    # Bass walks to the comma root anticipation
    score.add_note(
        "bass", partial=9 / 14, start=t + 8.0, duration=2.0, amp_db=-10.0, velocity=0.55
    )
    score.add_note(
        "bass", partial=1 / 2, start=t + 10.0, duration=6.5, amp_db=-8.0, velocity=0.68
    )

    # ---- Phrase 3: DARK (t+17 to t+25) ----
    score.add_phrase("bell", motif_c, start=t + 17.5)
    _chord_notes("pad", score, DARK_WIDE, start=t + 17.0, duration=7.5, amp_db=-10.0)
    # Bass descends to 7/6 region
    score.add_note(
        "bass", partial=7 / 12, start=t + 17.0, duration=7.0, amp_db=-8.5, velocity=0.65
    )

    # ---- Phrase 4: SUSPENDED → HOME (t+25 to t+35) ----
    # Extended melody — the developmental climax
    score.add_phrase("bell", motif_a_ext, start=t + 25.5)
    _chord_notes(
        "pad", score, SUSPENDED_WIDE, start=t + 25.0, duration=5.0, amp_db=-9.5
    )
    _chord_notes("pad", score, HOME_WIDE, start=t + 30.5, duration=5.0, amp_db=-9.0)
    score.add_note(
        "bass", partial=4 / 7, start=t + 25.0, duration=5.0, amp_db=-8.5, velocity=0.65
    )
    score.add_note(
        "bass", partial=1 / 2, start=t + 30.5, duration=5.5, amp_db=-8.0, velocity=0.72
    )

    # ---- Phrase 5: Second bloom — COMMA → DARK (t+36 to t+44) ----
    # Bell echoes motif A at a higher register
    motif_a_high = ratio_line(
        tones=[2, 9 / 4, 5 / 2, 3],
        rhythm=[1.0, 0.7, 0.9, 1.8],
        context=_melody_context(),
        amp_db=-6.0,
    )
    score.add_phrase("bell", motif_a_high, start=t + 36.5)
    _chord_notes("pad", score, COMMA_WIDE, start=t + 36.0, duration=4.0, amp_db=-9.5)
    _chord_notes("pad", score, DARK_WIDE, start=t + 40.5, duration=4.5, amp_db=-10.0)
    score.add_note(
        "bass", partial=1 / 2, start=t + 36.0, duration=4.0, amp_db=-8.5, velocity=0.65
    )
    score.add_note(
        "bass", partial=7 / 12, start=t + 40.0, duration=4.5, amp_db=-9.0, velocity=0.6
    )

    # ---- Phrase 6: settling down (t+45 to t+50) ----
    score.add_phrase("bell", motif_d, start=t + 45.0)
    _chord_notes("pad", score, HOME_WIDE, start=t + 45.0, duration=6.0, amp_db=-10.0)
    score.add_note(
        "bass", partial=1 / 2, start=t + 45.0, duration=6.0, amp_db=-8.0, velocity=0.68
    )


def _build_fade(score: Score) -> None:
    """Section III — Fade (1:20–2:00): texture simplifies, gentle ending."""
    t = FADE_START

    # Pad holds a warm suspended chord, then resolves
    _chord_notes(
        "pad",
        score,
        SUSPENDED_WIDE,
        start=t,
        duration=10.0,
        amp_db=-11.0,
        velocities=[0.55, 0.5, 0.52, 0.55],
    )

    # Bell: sparse, widely-spaced long tones
    score.add_note(
        "bell", partial=7 / 4, start=t + 2.0, duration=4.0, amp_db=-8.0, velocity=0.55
    )
    score.add_note(
        "bell", partial=3 / 2, start=t + 8.0, duration=3.5, amp_db=-9.0, velocity=0.5
    )

    # Bass simplifies — just the root, fading
    score.add_note(
        "bass", partial=1 / 2, start=t, duration=10.0, amp_db=-10.0, velocity=0.55
    )

    # Pad resolves to HOME
    _chord_notes(
        "pad",
        score,
        HOME_WIDE,
        start=t + 11.0,
        duration=12.0,
        amp_db=-12.0,
        velocities=[0.5, 0.48, 0.5, 0.52],
    )

    # Bell: final ascending figure — the motif one last time, gentle
    final_melody = ratio_line(
        tones=[1, 5 / 4, 3 / 2, 7 / 4],
        rhythm=[2.0, 1.5, 1.5, 4.0],
        context=_melody_context(),
        amp_db=-9.0,
    )
    score.add_phrase("bell", final_melody, start=t + 14.0)

    # Bass fades out early
    score.add_note(
        "bass", partial=1 / 2, start=t + 12.0, duration=8.0, amp_db=-14.0, velocity=0.4
    )

    # Final pad chord — RESOLVE (open octave), very soft
    for i, ratio in enumerate([1 / 2, 5 / 4, 3 / 2, 2]):
        score.add_note(
            "pad",
            partial=ratio,
            start=t + 24.0 + i * 0.5,
            duration=14.0,
            amp_db=-14.0,
            velocity=0.4 + i * 0.02,
        )

    # Bell: one final high note
    score.add_note(
        "bell", partial=2, start=t + 30.0, duration=6.0, amp_db=-12.0, velocity=0.35
    )


# ---------------------------------------------------------------------------
# Score assembly
# ---------------------------------------------------------------------------


def build_score() -> Score:
    """Build the complete Seventh Window score."""
    score = Score(
        f0_hz=F0,
        sample_rate=44100,
        master_effects=[
            EffectSpec(
                kind="eq",
                params={
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 30, "slope_db_per_oct": 12},
                        {"kind": "lowpass", "cutoff_hz": 16000, "slope_db_per_oct": 12},
                    ]
                },
            ),
            EffectSpec(kind="saturation", params={"drive": 0.08, "mix": 0.3}),
        ],
        timing_humanize=TimingHumanizeSpec(preset="chamber"),
    )

    # ---- Pad voice ----
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "vital",
            "vital_params": PAD_VITAL_PARAMS,
            "tail_seconds": 5.0,
        },
        normalize_lufs=-22.0,
        pan=-0.1,
        effects=[
            EffectSpec(
                kind="eq",
                params={
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 80, "slope_db_per_oct": 12},
                        {"kind": "lowpass", "cutoff_hz": 12000, "slope_db_per_oct": 12},
                    ]
                },
            ),
        ],
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        mix_db=-2.0,
    )

    # ---- Bell voice ----
    score.add_voice(
        "bell",
        synth_defaults={
            "engine": "vital",
            "vital_params": BELL_VITAL_PARAMS,
            "tail_seconds": 3.0,
        },
        normalize_lufs=-20.0,
        pan=0.15,
        effects=[
            EffectSpec(
                kind="eq",
                params={
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 150, "slope_db_per_oct": 12},
                        {"kind": "lowpass", "cutoff_hz": 14000, "slope_db_per_oct": 12},
                    ]
                },
            ),
        ],
        velocity_humanize=VelocityHumanizeSpec(preset="breathing_ensemble"),
        envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
        mix_db=-1.0,
    )

    # ---- Bass voice ----
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "vital",
            "vital_params": BASS_VITAL_PARAMS,
            "tail_seconds": 2.0,
        },
        normalize_lufs=-22.0,
        pan=0.0,
        effects=[
            EffectSpec(
                kind="eq",
                params={
                    "bands": [
                        {"kind": "highpass", "cutoff_hz": 30, "slope_db_per_oct": 12},
                        {"kind": "lowpass", "cutoff_hz": 800, "slope_db_per_oct": 12},
                    ]
                },
            ),
            EffectSpec(
                kind="compressor",
                params={
                    "threshold_db": -18.0,
                    "ratio": 3.0,
                    "attack_ms": 20.0,
                    "release_ms": 150.0,
                },
            ),
        ],
        velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
        mix_db=0.0,
    )

    # ---- Build sections ----
    _build_open(score)
    _build_bloom(score)
    _build_fade(score)

    return score


# ---------------------------------------------------------------------------
# Piece registration
# ---------------------------------------------------------------------------

PIECES: dict[str, PieceDefinition] = {
    "seventh_window": PieceDefinition(
        name="seventh_window",
        output_name="seventh_window",
        build_score=build_score,
        sections=(
            PieceSection(
                label="Open", start_seconds=OPEN_START, end_seconds=BLOOM_START
            ),
            PieceSection(
                label="Bloom", start_seconds=BLOOM_START, end_seconds=FADE_START
            ),
            PieceSection(label="Fade", start_seconds=FADE_START, end_seconds=PIECE_END),
        ),
    ),
}
