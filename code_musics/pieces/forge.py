"""Forge — a study in composable percussion.

Demonstrates the unified drum_voice engine: hybrid timbres that cross the
boundaries between kick, hat, snare, and melodic percussion.  Three sections
evolve from clean electronic drums into increasingly alien territory.

    Section A  (bars 1-12):   Clean — recognizable electronic drums, 909-ish
    Section B  (bars 13-24):  Hybrid — metallic overtones on kicks, FM hats,
                              ring-mod bells, saturation-shaped bodies
    Section C  (bars 25-36):  Alien — full composable percussion, fm_burst
                              exciters, fm_cluster metallics, resonator+partial
                              hybrids

120 BPM, 7-limit JI rooted on F (~87.3 Hz), ~72 seconds.
"""

from __future__ import annotations

import math
from typing import Any

from code_musics.drum_helpers import add_drum_voice, setup_drum_bus
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score, VoiceSend

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

BPM = 120.0
BEAT = 60.0 / BPM  # 0.5 s
BAR = 4 * BEAT  # 2.0 s
S16 = BEAT / 4  # 0.125 s

# 7-limit JI intervals
F0 = 87.307  # F2
P5 = 3 / 2  # perfect fifth
P7 = 7 / 4  # septimal seventh
P3 = 5 / 4  # major third
P11 = 11 / 8  # undecimal fourth
OCTAVE = 2

N_BARS = 36
TOTAL_DUR = N_BARS * BAR + 2.0  # pad for tail


def _pos(bar: int, beat: int = 1, n16: int = 0) -> float:
    """Bar-beat-16th position in seconds (1-indexed bars and beats)."""
    return (bar - 1) * BAR + (beat - 1) * BEAT + n16 * S16


# ---------------------------------------------------------------------------
# Section boundaries
# ---------------------------------------------------------------------------

SEC_A = (1, 12)  # clean
SEC_B = (13, 24)  # hybrid
SEC_C = (25, 36)  # alien


def _in_section(bar: int, section: tuple[int, int]) -> bool:
    return section[0] <= bar <= section[1]


# ---------------------------------------------------------------------------
# Drum voice synth overrides per section
# ---------------------------------------------------------------------------

# --- Kick ---
_KICK_CLEAN: dict[str, Any] = {}  # use preset defaults

_KICK_HYBRID: dict[str, Any] = {
    "metallic_type": "partials",
    "metallic_level": 0.06,
    "metallic_partial_ratios": [P7, P11 * 2],
    "metallic_n_partials": 2,
    "metallic_brightness": 0.5,
    "metallic_decay_s": 0.12,
    "tone_shaper": "saturation",
    "tone_shaper_drive": 0.3,
    "tone_shaper_mix": 0.5,
    "tone_shaper_mode": "triode",
}

_KICK_ALIEN: dict[str, Any] = {
    "tone_type": "resonator",
    "exciter_type": "fm_burst",
    "exciter_fm_ratio": P7,
    "exciter_fm_index": 5.0,
    "exciter_level": 0.15,
    "metallic_type": "fm_cluster",
    "metallic_level": 0.08,
    "metallic_n_operators": 3,
    "metallic_brightness": 0.4,
    "metallic_decay_s": 0.15,
    "tone_shaper": "preamp",
    "tone_shaper_drive": 0.4,
    "tone_shaper_mix": 0.6,
}


def _kick_synth(bar: int) -> dict[str, Any]:
    if _in_section(bar, SEC_C):
        return _KICK_ALIEN
    if _in_section(bar, SEC_B):
        return _KICK_HYBRID
    return _KICK_CLEAN


# --- Hat ---
_HAT_CLEAN: dict[str, Any] = {}

_HAT_FM: dict[str, Any] = {
    "metallic_type": "fm_cluster",
    "metallic_n_operators": 4,
    "metallic_fm_index": 4.0,
    "metallic_brightness": 0.6,
    "metallic_density": 0.3,
}

_HAT_ALIEN: dict[str, Any] = {
    "metallic_type": "fm_cluster",
    "metallic_n_operators": 6,
    "metallic_fm_index": 6.0,
    "metallic_brightness": 0.5,
    "metallic_density": 0.5,
    "exciter_type": "fm_burst",
    "exciter_fm_ratio": math.sqrt(2),
    "exciter_fm_index": 3.0,
    "exciter_level": 0.12,
    "metallic_filter_mode": "highpass",
    "metallic_filter_cutoff_hz": 6000.0,
}


def _hat_synth(bar: int) -> dict[str, Any]:
    if _in_section(bar, SEC_C):
        return _HAT_ALIEN
    if _in_section(bar, SEC_B):
        return _HAT_FM
    return _HAT_CLEAN


# --- Bell (ring-mod metallic, enters in section B) ---
_BELL_BASE: dict[str, Any] = {
    "metallic_type": "ring_mod",
    "metallic_ring_mod_freq_ratio": P7,
    "metallic_ring_mod_amount": 0.6,
    "metallic_n_partials": 5,
    "metallic_brightness": 0.8,
    "metallic_decay_s": 0.4,
    "metallic_filter_mode": "bandpass",
    "metallic_filter_cutoff_hz": 4000.0,
    "metallic_filter_q": 2.0,
}


# ---------------------------------------------------------------------------
# Pattern helpers
# ---------------------------------------------------------------------------


def _four_on_floor(score: Score, bar: int) -> None:
    """Standard four-on-the-floor kick."""
    synth = _kick_synth(bar)
    for beat in range(1, 5):
        score.add_note(
            "kick",
            start=_pos(bar, beat),
            duration=0.4,
            freq=F0,
            amp_db=-6.0,
            synth=synth if synth else None,
        )


def _hat_pattern(score: Score, bar: int) -> None:
    """Hats: 8ths with ghost 16ths evolving across sections."""
    synth = _hat_synth(bar)
    hat_freq = F0 * P5 * OCTAVE * OCTAVE * OCTAVE * OCTAVE  # ~6283 Hz
    for n16 in range(16):
        on_eighth = n16 % 2 == 0
        if on_eighth:
            db = -10.0 if n16 % 4 == 0 else -14.0
        else:
            # ghost 16ths: louder in later sections
            if _in_section(bar, SEC_A):
                if n16 % 4 != 1:
                    continue
                db = -22.0
            elif _in_section(bar, SEC_B):
                db = -20.0
            else:
                db = -18.0

        # pitch micro-variation: different 16th positions get different freqs
        pitch_mult = 1.0 + 0.02 * math.sin(n16 * 0.7)
        score.add_note(
            "hat",
            start=_pos(bar, 1, n16),
            duration=0.06,
            freq=hat_freq * pitch_mult,
            amp_db=db,
            synth=synth if synth else None,
        )


def _open_hat(score: Score, bar: int) -> None:
    """Open hat on the 'and' of beat 2 and 4."""
    synth = _hat_synth(bar)
    for beat in [2, 4]:
        score.add_note(
            "open_hat",
            start=_pos(bar, beat, 2),
            duration=0.15,
            freq=F0 * P5 * OCTAVE**4,
            amp_db=-14.0,
            synth=synth if synth else None,
        )


def _snare_hits(score: Score, bar: int) -> None:
    """Snare on 2 and 4."""
    freq = F0 * OCTAVE * OCTAVE  # ~349 Hz
    score.add_note(
        "snare",
        start=_pos(bar, 2),
        duration=0.2,
        freq=freq,
        amp_db=-8.0,
    )
    score.add_note(
        "snare",
        start=_pos(bar, 4),
        duration=0.2,
        freq=freq,
        amp_db=-8.0,
    )
    # ghost on the 'a' of 3 in later sections
    if not _in_section(bar, SEC_A):
        score.add_note(
            "snare",
            start=_pos(bar, 3, 3),
            duration=0.12,
            freq=freq * P5,
            amp_db=-20.0,
        )


def _bell_accents(score: Score, bar: int) -> None:
    """Ring-mod bell accents — enter in section B."""
    if _in_section(bar, SEC_A):
        return
    intervals = [1.0, P5, P7, P3, OCTAVE]
    bar_in_section = (bar - SEC_B[0]) if _in_section(bar, SEC_B) else (bar - SEC_C[0])
    idx = bar_in_section % len(intervals)
    bell_freq = F0 * OCTAVE * OCTAVE * intervals[idx]

    # one accent per bar, position varies
    beat_pos = [1, 3, 2, 4, 1][idx]
    n16_pos = [0, 2, 1, 3, 0][idx]
    score.add_note(
        "bell",
        start=_pos(bar, beat_pos, n16_pos),
        duration=0.3,
        freq=bell_freq,
        amp_db=-12.0,
    )
    # second bell in alien section, a fifth above
    if _in_section(bar, SEC_C) and bar % 2 == 0:
        score.add_note(
            "bell",
            start=_pos(bar, 3),
            duration=0.25,
            freq=bell_freq * P5,
            amp_db=-16.0,
        )


def _alien_perc(score: Score, bar: int) -> None:
    """Extra percussion layer in alien section: pitched resonator hits."""
    if not _in_section(bar, SEC_C):
        return
    intervals = [1.0, P5, P7, P3, P11]
    for n16 in [1, 5, 9, 13]:
        idx = (bar + n16) % len(intervals)
        perc_freq = F0 * OCTAVE * intervals[idx]
        score.add_note(
            "alien_perc",
            start=_pos(bar, 1, n16),
            duration=0.15,
            freq=perc_freq,
            amp_db=-16.0,
        )


# ---------------------------------------------------------------------------
# Score builder
# ---------------------------------------------------------------------------


def build_score() -> Score:
    """Build the Forge study score."""
    score = Score(
        f0_hz=F0,
        time_reference_total_dur=TOTAL_DUR,
        sample_rate=44100,
        master_effects=DEFAULT_MASTER_EFFECTS,
    )

    # --- Reverb send bus ---
    score.add_send_bus(
        "hall",
        effects=[
            EffectSpec("bricasti", {"ir_name": "1 Halls 07 Large & Dark", "wet": 1.0})
        ],
        return_db=-6.0,
    )

    # --- Drum bus ---
    drum_bus = setup_drum_bus(
        score,
        effects=[
            EffectSpec(
                "compressor",
                {
                    "topology": "feedback",
                    "threshold_db": -18.0,
                    "ratio": 1.8,
                    "attack_ms": 20.0,
                    "release_ms": 180.0,
                    "detector_mode": "rms",
                    "detector_bands": [
                        {"kind": "highpass", "cutoff_hz": 55.0, "slope_db_per_oct": 12}
                    ],
                },
            ),
            EffectSpec("saturation", {"mode": "triode", "drive": 1.6, "mix": 0.25}),
        ],
    )

    # --- Drum voices ---

    # Kick: starts as 808, evolves into hybrid then alien
    add_drum_voice(
        score,
        "kick",
        engine="drum_voice",
        preset="808_hiphop",
        drum_bus=drum_bus,
        mix_db=-2.0,
    )

    # Closed hat: starts as standard partials, evolves into FM cluster
    hat = add_drum_voice(
        score,
        "hat",
        engine="drum_voice",
        preset="closed_hat",
        drum_bus=drum_bus,
        mix_db=-4.0,
    )
    hat.sends.append(VoiceSend(target="hall", send_db=-18.0))

    # Open hat
    oh = add_drum_voice(
        score,
        "open_hat",
        engine="drum_voice",
        preset="open_hat",
        drum_bus=drum_bus,
        mix_db=-6.0,
    )
    oh.sends.append(VoiceSend(target="hall", send_db=-12.0))

    # Snare: 909 tight
    snare = add_drum_voice(
        score,
        "snare",
        engine="drum_voice",
        preset="909_tight",
        drum_bus=drum_bus,
        mix_db=-3.0,
    )
    snare.sends.append(VoiceSend(target="hall", send_db=-14.0))

    # Bell: ring-mod metallic percussion (enters section B)
    bell = add_drum_voice(
        score,
        "bell",
        engine="drum_voice",
        preset=None,
        drum_bus=drum_bus,
        mix_db=-5.0,
        effects=[],  # no default effects — this is a custom voice
        synth_overrides=_BELL_BASE,
    )
    bell.sends.append(VoiceSend(target="hall", send_db=-8.0))

    # Alien perc: resonator + FM burst exciter + metallic (enters section C)
    alien_perc = add_drum_voice(
        score,
        "alien_perc",
        engine="drum_voice",
        preset=None,
        drum_bus=drum_bus,
        mix_db=-6.0,
        effects=[],
        synth_overrides={
            "tone_type": "resonator",
            "exciter_type": "fm_burst",
            "exciter_fm_ratio": P11,
            "exciter_fm_index": 3.0,
            "exciter_level": 0.2,
            "exciter_decay_s": 0.01,
            "tone_decay_s": 0.3,
            "metallic_type": "partials",
            "metallic_level": 0.1,
            "metallic_partial_ratios": [P7, P11, P5 * 2],
            "metallic_brightness": 0.6,
            "metallic_decay_s": 0.2,
            "metallic_filter_mode": "bandpass",
            "metallic_filter_cutoff_hz": 3000.0,
        },
    )
    alien_perc.sends.append(VoiceSend(target="hall", send_db=-10.0))

    # --- Place notes ---
    for bar in range(1, N_BARS + 1):
        _four_on_floor(score, bar)
        _hat_pattern(score, bar)
        _snare_hits(score, bar)
        _open_hat(score, bar)
        _bell_accents(score, bar)
        _alien_perc(score, bar)

    return score


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

PIECES: dict[str, PieceDefinition] = {
    "forge": PieceDefinition(
        name="forge",
        output_name="forge",
        build_score=build_score,
        study=True,
        sections=(
            PieceSection("clean", _pos(1), _pos(13)),
            PieceSection("hybrid", _pos(13), _pos(25)),
            PieceSection("alien", _pos(25), _pos(37)),
        ),
    ),
}
