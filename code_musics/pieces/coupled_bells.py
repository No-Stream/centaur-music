"""coupled_bells — ensemble of struck bodies with mode-coupling + dispersion.

A bell garden of four struck voices driven by ``drum_voice`` modal banks.
Each voice runs a non-zero ``modal_coupling`` (0.2-0.3) and moderate
``modal_dispersion`` (0.35-0.55), so the modes within each body exchange
energy and smear their attacks — producing rolling beats and living decays
rather than clean, static rings.

Four struck bodies covering the spectrum:

  * ``membrane`` (low bowl, ~F3) — deep struck drum-bell.
  * ``bar_metal`` (mid bell, ~C4) — bright ping with long decay.
  * ``bowl`` (high bowl, ~F4) — shimmering upper-register resonator.
  * ``plate`` (glassy accent, ~A4) — chimes over the texture.

Rhythm: a 2-voice polyrhythm (3-against-4) layered with a euclidean
pattern on the high plate — irregular but patterned, letting the modal
couplings sing between strikes.

Key: F 7-limit JI (tonic 174.614 Hz).  ~60 s.
"""

from __future__ import annotations

from code_musics.drum_helpers import setup_drum_bus
from code_musics.generative.euclidean import euclidean_pattern
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS, SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score, VoiceSend

F0_HZ = 174.614  # F3
TOTAL_DUR = 60.0
BPM = 66.0
BEAT = 60.0 / BPM  # ~0.909 s per beat

# 7-limit ratios used for bell tunings.
R_ROOT = 1.0
R_FIFTH = 3 / 2
R_SEPTIMAL_SEVENTH = 7 / 4
R_MAJOR_THIRD = 5 / 4
R_SECOND = 9 / 8
R_OCTAVE = 2.0


def _chord_at(bar_index: int) -> tuple[float, ...]:
    """Rotate through a small JI chord progression across the piece."""
    progression = [
        (R_ROOT, R_MAJOR_THIRD, R_FIFTH, R_SEPTIMAL_SEVENTH),
        (R_ROOT, R_FIFTH, R_SEPTIMAL_SEVENTH, R_MAJOR_THIRD * R_OCTAVE),
        (R_ROOT, R_SECOND, R_MAJOR_THIRD, R_FIFTH),
        (
            R_ROOT * 4 / 3,
            R_ROOT * 4 / 3 * R_MAJOR_THIRD,
            R_ROOT * 4 / 3 * R_FIFTH,
            R_SEPTIMAL_SEVENTH,
        ),
    ]
    return progression[bar_index % len(progression)]


def build_coupled_bells() -> Score:
    """Build the Coupled Bells score."""
    score = Score(f0_hz=F0_HZ, master_effects=DEFAULT_MASTER_EFFECTS)

    bus_name = setup_drum_bus(
        score,
        bus_name="bell_hall",
        effects=[SOFT_REVERB_EFFECT],
        return_db=-3.0,
    )

    # ------------------------------------------------------------------
    # Low membrane bell — F3 region, deep struck resonator.
    # Chain coupling + medium dispersion smears attacks into a bloom.
    # ------------------------------------------------------------------
    score.add_voice(
        "membrane",
        synth_defaults={
            "engine": "drum_voice",
            "exciter_type": "click",
            "exciter_level": 0.45,
            "exciter_decay_s": 0.008,
            "tone_type": "modal",
            "tone_level": 1.0,
            "tone_decay_s": 1.8,
            "modal_mode_table": "membrane",
            "modal_n_modes": 6,
            "modal_coupling": 0.28,
            "modal_coupling_topology": "chain",
            "modal_dispersion": 0.45,
            "modal_dispersion_n_stages": 4,
            "pi_hardness": 0.55,
            "pi_tension": -0.1,
            "pi_damping": 0.3,
            "pi_position": 0.35,
        },
        effects=[EffectSpec("compressor", {"preset": "tom_control"})],
        normalize_peak_db=-6.0,
        mix_db=-8.0,
        velocity_humanize=None,
        pan=-0.25,
        sends=[VoiceSend(target=bus_name, send_db=-4.0)],
    )

    # ------------------------------------------------------------------
    # Mid metal bar bell — C4 region, bright ping, long decay.
    # Parallel coupling + higher dispersion gives a rolling shimmer.
    # ------------------------------------------------------------------
    score.add_voice(
        "metal",
        synth_defaults={
            "engine": "drum_voice",
            "exciter_type": "click",
            "exciter_level": 0.3,
            "exciter_decay_s": 0.004,
            "tone_type": None,
            "metallic_type": "modal_bank",
            "metallic_level": 1.0,
            "metallic_mode_table": "bar_metal",
            "metallic_n_modes": 6,
            "metallic_decay_s": 1.6,
            "metallic_coupling": 0.22,
            "metallic_coupling_topology": "ring",
            "metallic_dispersion": 0.4,
            "metallic_dispersion_n_stages": 4,
            "metallic_tension": 0.15,
            "metallic_damping": 0.28,
            "metallic_position": 0.4,
        },
        effects=[EffectSpec("compressor", {"preset": "hat_control"})],
        normalize_peak_db=-6.0,
        mix_db=-12.0,
        velocity_humanize=None,
        pan=0.2,
        sends=[VoiceSend(target=bus_name, send_db=-2.0)],
    )

    # ------------------------------------------------------------------
    # High bowl bell — F4 region, glass-like sustain.
    # High coupling (0.32) + moderate dispersion for living decay.
    # ------------------------------------------------------------------
    score.add_voice(
        "bowl",
        synth_defaults={
            "engine": "drum_voice",
            "exciter_type": "click",
            "exciter_level": 0.22,
            "exciter_decay_s": 0.015,
            "tone_type": None,
            "metallic_type": "modal_bank",
            "metallic_level": 0.95,
            "metallic_mode_table": "bowl",
            "metallic_n_modes": 6,
            "metallic_decay_s": 2.2,
            "metallic_coupling": 0.28,
            "metallic_coupling_topology": "chain",
            "metallic_dispersion": 0.55,
            "metallic_dispersion_n_stages": 5,
            "metallic_tension": 0.05,
            "metallic_damping": 0.2,
            "metallic_position": 0.3,
        },
        normalize_peak_db=-6.0,
        mix_db=-13.0,
        velocity_humanize=None,
        pan=-0.1,
        sends=[VoiceSend(target=bus_name, send_db=-2.0)],
    )

    # ------------------------------------------------------------------
    # Plate chime — A4 region, glassy sparse accents.
    # Lower coupling (0.2) keeps strikes brighter but still wobbly.
    # ------------------------------------------------------------------
    score.add_voice(
        "plate",
        synth_defaults={
            "engine": "drum_voice",
            "exciter_type": "click",
            "exciter_level": 0.18,
            "exciter_decay_s": 0.003,
            "tone_type": None,
            "metallic_type": "modal_bank",
            "metallic_level": 0.85,
            "metallic_mode_table": "plate",
            "metallic_n_modes": 5,
            "metallic_decay_s": 2.0,
            "metallic_coupling": 0.2,
            "metallic_coupling_topology": "chain",
            "metallic_dispersion": 0.35,
            "metallic_dispersion_n_stages": 3,
            "metallic_tension": 0.25,
            "metallic_damping": 0.35,
            "metallic_position": 0.5,
        },
        normalize_peak_db=-6.0,
        mix_db=-15.0,
        velocity_humanize=None,
        pan=0.3,
        sends=[VoiceSend(target=bus_name, send_db=-4.0)],
    )

    # ------------------------------------------------------------------
    # Sustained drone — harmonic additive pad underneath the bell texture.
    # Grounds the JI root with pure harmonics so the coupled-modal bells
    # can sing over a stable tonal bed rather than fighting an inharmonic
    # wash (which drove the master limiter into -30+ dB gain reduction).
    # ------------------------------------------------------------------
    score.add_voice(
        "drone",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "additive_pad_through_ladder",
            "attack": 3.0,
            "release": 4.0,
        },
        sends=[VoiceSend(target=bus_name, send_db=-10.0)],
        pan=0.0,
        mix_db=-9.0,
    )
    for partial, amp_db in [(0.5, -22.0), (1.0, -21.0), (3 / 2, -23.0)]:
        score.add_note(
            "drone", start=0.0, duration=TOTAL_DUR, partial=partial, amp_db=amp_db
        )

    # ==================================================================
    # Rhythm material:
    # - membrane: every 3 beats (slower pulse).
    # - metal: every 2 beats, offset.
    # - bowl: every 4 beats, slightly anticipated.
    # - plate: euclidean 5-in-8 at half-note resolution for irregular
    #   accents.
    # ==================================================================
    total_beats = int(TOTAL_DUR / BEAT)  # ~66 beats

    # Membrane: on every beat with alternating accent for more continuous pulse.
    membrane_partials = [R_ROOT, R_FIFTH / 2, R_ROOT * 4 / 3]
    for step, beat_idx in enumerate(range(0, total_beats, 2)):
        bar_index = beat_idx // 8  # switch "chord" every 8 beats
        chord = _chord_at(bar_index)
        # pick a root from chord
        partial = membrane_partials[step % len(membrane_partials)]
        # hand-rolled velocity sway for motion
        amp_db = -14.0 if step % 4 == 0 else -17.0
        score.add_note(
            "membrane",
            start=beat_idx * BEAT,
            duration=1.2,
            partial=partial * chord[0] / R_ROOT,
            amp_db=amp_db,
        )

    # Metal: cross-rhythm every other beat with a half-beat offset — sparser
    # than a hit-per-beat so the coupled/dispersed decays don't pile up and
    # overdrive the master limiter.
    for step, beat_idx in enumerate(range(1, total_beats, 2)):
        bar_index = beat_idx // 8
        chord = _chord_at(bar_index)
        partial = chord[1] * 2  # use chord's 3rd/5th octave up
        amp_db = -13.0 if step % 3 == 0 else -16.0
        score.add_note(
            "metal",
            start=beat_idx * BEAT,
            duration=1.0,
            partial=partial,
            amp_db=amp_db,
        )

    # Bowl: every 3 beats, slightly anticipated — continuous rolling texture.
    for beat_idx in range(0, total_beats, 3):
        bar_index = beat_idx // 8
        chord = _chord_at(bar_index)
        # Rotate through chord tones
        partial_idx = (beat_idx // 4) % len(chord)
        partial = chord[partial_idx] * 2  # up an octave
        amp_db = -14.0 if partial_idx == 0 else -17.0
        score.add_note(
            "bowl",
            start=max(0.0, beat_idx * BEAT - 0.05),
            duration=1.5,
            partial=partial,
            amp_db=amp_db,
        )

    # Plate: euclidean 5-in-8, half-note resolution (2 beats per step).
    pattern = euclidean_pattern(hits=5, steps=8, rotation=2)
    half_step = 2.0 * BEAT
    for cycle_start in range(0, int(TOTAL_DUR / (8 * half_step)) + 1):
        for step, hit in enumerate(pattern):
            if not hit:
                continue
            start = (cycle_start * 8 + step) * half_step
            if start >= TOTAL_DUR:
                break
            bar_index = int(start // (8 * BEAT))
            chord = _chord_at(bar_index)
            # Plate picks the highest chord tone * 2
            partial = chord[-1] * 2
            amp_db = -15.0 if step in (0, 3) else -18.0
            score.add_note(
                "plate",
                start=start,
                duration=1.2,
                partial=partial,
                amp_db=amp_db,
            )

    # Final sustained bowl swell — a slimmer JI triad so the coupled tails
    # sing without piling peaks into the limiter.
    final_partials = [R_ROOT, R_FIFTH, R_ROOT * R_OCTAVE]
    for partial in final_partials:
        score.add_note(
            "bowl",
            start=TOTAL_DUR - 6.0,
            duration=6.0,
            partial=partial,
            amp_db=-19.0,
        )

    return score


PIECES: dict[str, PieceDefinition] = {
    "coupled_bells": PieceDefinition(
        name="coupled_bells",
        output_name="coupled_bells_01",
        build_score=build_coupled_bells,
        sections=(
            PieceSection(
                label="Coupled Bells", start_seconds=0.0, end_seconds=TOTAL_DUR
            ),
        ),
    ),
}
