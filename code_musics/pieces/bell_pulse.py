"""Bell Pulse — drums as harmony.

80 BPM, 7-limit JI, ~100 seconds (33 bars, plus tail).

The resonator kick walks a bass line through harmonic series intervals.
JI-tuned metallic bells play arpeggiated figures consonant with the kick.
The FM snare morphs from a pitched tom to a noise burst across the piece.
Three slightly detuned hats create beating interference patterns.

Section map:
  1-8    Bells — kick bass line + JI bells only, spacious
  9-16   Pulse — beating hats enter, snare (as tom), granular texture accents
  17-24  Arc — full texture, snare FM index climbs, dynamics peak at bar 20
  25-33  Dissolve — snare index falls, hats thin, kick sustains long, bells fade
"""

from __future__ import annotations

from code_musics.drum_helpers import add_drum_voice, setup_drum_bus
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score, VoiceSend

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

BPM = 80.0
BEAT = 60.0 / BPM
BAR = 4.0 * BEAT
S16 = BEAT / 4.0
F0 = 55.0  # A1
TOTAL_BARS = 33


def _pos(bar: int, beat: int = 1, n16: int = 0) -> float:
    """Bar/beat/16th -> seconds.  bar and beat are 1-based."""
    return (bar - 1) * BAR + (beat - 1) * BEAT + n16 * S16


TOTAL_DUR = _pos(TOTAL_BARS + 1) + 2.0  # extra tail for reverb

# ---------------------------------------------------------------------------
# Harmonic series partials
# ---------------------------------------------------------------------------

P1 = 1.0  # 55 Hz   A1
P32 = 3 / 2  # 82.5   ~E2
P74 = 7 / 4  # 96.25  ~G2 (septimal)
P2 = 2.0  # 110    A2
P3 = 3.0  # 165    ~E3
P4 = 4.0  # 220    A3
P5 = 5.0  # 275    ~C#4
P6 = 6.0  # 330    E4
P7 = 7.0  # 385    ~Bb4
P8 = 8.0  # 440    A4

# Kick bass line — 4-bar cycle through harmonic series
KICK_BASS_LINE: list[tuple[float, ...]] = [
    # bar 1: root, bar 2: 5th, bar 3: septimal 7th, bar 4: octave
    # Each tuple is (beat1_partial, beat3_partial) — kick on 1 and 3
    (P1, P1),
    (P32, P32),
    (P74, P2),
    (P2, P1),
]

# Bell arpeggio intervals (relative to kick's current partial)
# These partials stack consonant intervals above the kick
BELL_INTERVALS = [1.0, 3 / 2, 2.0, 5 / 2, 3.0]

# ---------------------------------------------------------------------------
# FM snare morph curve: bar -> fm_index
# ---------------------------------------------------------------------------


def _snare_fm_index(bar: int) -> float:
    """FM index arc: tom-like at edges, noise burst at peak."""
    if bar <= 8:
        return 0.5
    if bar <= 16:
        return 0.5 + 0.5 * (bar - 8) / 8  # 0.5 -> 1.0
    if bar <= 20:
        return 1.0 + 5.0 * (bar - 16) / 4  # 1.0 -> 6.0
    if bar <= 24:
        return 6.0 - 3.0 * (bar - 20) / 4  # 6.0 -> 3.0
    return 3.0 - 2.0 * (bar - 24) / 9  # 3.0 -> 1.0


def _section(bar: int) -> str:
    if bar <= 8:
        return "bells"
    if bar <= 16:
        return "pulse"
    if bar <= 24:
        return "arc"
    return "dissolve"


def _dynamics(bar: int) -> float:
    """Dynamic arc in dB offset."""
    if bar <= 8:
        return -2.0 + 0.25 * bar  # -2 -> 0
    if bar <= 20:
        return 0.0 + 0.25 * (bar - 8)  # 0 -> 3
    if bar <= 24:
        return 3.0 - 0.5 * (bar - 20)  # 3 -> 1
    return 1.0 - 0.3 * (bar - 24)  # 1 -> -1.7


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_score() -> Score:
    score = Score(
        f0_hz=F0,
        sample_rate=44_100,
        master_effects=list(DEFAULT_MASTER_EFFECTS),
    )

    # -------------------------------------------------------------------
    # Send buses
    # -------------------------------------------------------------------

    drum_bus = setup_drum_bus(
        score,
        effects=[
            EffectSpec("compressor", {"preset": "kick_glue"}),
        ],
        return_db=0.0,
    )

    score.add_send_bus(
        "hall",
        effects=[
            EffectSpec(
                "reverb",
                {"room_size": 0.88, "damping": 0.50, "wet_level": 1.0},
            ),
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "lowpass",
                            "cutoff_hz": 5000.0,
                            "slope_db_per_oct": 12,
                        },
                        {
                            "kind": "highpass",
                            "cutoff_hz": 150.0,
                            "slope_db_per_oct": 12,
                        },
                    ],
                },
            ),
        ],
        return_db=-3.0,
    )

    # -------------------------------------------------------------------
    # Drum voices
    # -------------------------------------------------------------------

    # Melodic resonator kick — long ring, minimal sweep
    add_drum_voice(
        score,
        "kick",
        engine="drum_voice",
        drum_bus=drum_bus,
        send_db=-2.0,
        effects=[EffectSpec("compressor", {"preset": "kick_punch"})],
        mix_db=2.0,
        synth_overrides={
            "tone_type": "resonator",
            "tone_decay_s": 0.5,
            "tone_sweep_ratio": 1.1,
            "tone_sweep_decay_s": 0.020,
            "tone_punch": 0.15,
            "tone_second_harmonic": 0.12,
            "exciter_level": 0.03,
            "noise_level": 0.01,
            "velocity_timbre_decay": 0.3,
            "velocity_timbre_brightness": 0.2,
        },
    )
    score.voices["kick"].sends.append(VoiceSend(target="hall", send_db=-14.0))

    # JI-tuned bells — harmonic partial ratios
    add_drum_voice(
        score,
        "bell",
        engine="drum_voice",
        drum_bus=drum_bus,
        send_db=-6.0,
        mix_db=-6.0,
        synth_overrides={
            "metallic_type": "partials",
            "metallic_n_partials": 5,
            "metallic_partial_ratios": [1.0, 1.5, 2.0, 2.5, 3.0],
            "metallic_oscillator_mode": "sine",
            "metallic_brightness": 0.80,
            "metallic_decay_s": 0.400,
            "metallic_level": 1.0,
            "metallic_filter_mode": "bandpass",
            "metallic_filter_cutoff_hz": 4000.0,
            "metallic_filter_q": 0.9,
            "metallic_density": 0.1,
            "exciter_level": 0.06,
            "noise_level": 0.02,
            "tone_type": None,
            "velocity_timbre_brightness": 0.3,
            "velocity_timbre_decay": 0.2,
        },
    )
    score.voices["bell"].sends.append(VoiceSend(target="hall", send_db=-6.0))

    # Beating hat trio — three slightly detuned square-wave hats
    for hat_idx, (ratios, dens) in enumerate(
        [
            ([1.0, 1.3348, 1.4755, 1.6818, 1.9307, 2.5452], 0.60),
            ([1.005, 1.340, 1.480, 1.688, 1.937, 2.552], 0.65),
            ([0.995, 1.330, 1.471, 1.676, 1.925, 2.538], 0.55),
        ]
    ):
        name = f"hat_{hat_idx}"
        add_drum_voice(
            score,
            name,
            engine="drum_voice",
            drum_bus=drum_bus,
            send_db=-6.0,
            mix_db=-10.0,
            synth_overrides={
                "metallic_type": "partials",
                "metallic_n_partials": 6,
                "metallic_partial_ratios": ratios,
                "metallic_oscillator_mode": "square",
                "metallic_brightness": 0.65,
                "metallic_decay_s": 0.045,
                "metallic_level": 1.0,
                "metallic_density": dens,
                "metallic_filter_q": 0.8,
                "noise_level": 0.15,
                "tone_type": None,
                "velocity_timbre_brightness": 0.25,
            },
        )

    # FM snare — morph from tom to noise via per-note index override
    add_drum_voice(
        score,
        "snare",
        engine="drum_voice",
        drum_bus=drum_bus,
        send_db=-3.0,
        effects=[EffectSpec("compressor", {"preset": "snare_body"})],
        mix_db=-4.0,
        synth_overrides={
            "tone_fm_ratio": 1.5,
            "tone_fm_index": 0.5,  # default low -- overridden per note
            "tone_decay_s": 0.140,
            "noise_type": "comb",
            "noise_decay_s": 0.160,
            "noise_pre_noise_mode": "colored",
            "tone_level": 0.7,
            "noise_level": 0.3,
            "velocity_timbre_decay": 0.2,
            "velocity_timbre_harmonics": 0.3,
        },
    )
    score.voices["snare"].sends.append(VoiceSend(target="hall", send_db=-12.0))

    # Granular clap texture
    add_drum_voice(
        score,
        "texture",
        engine="drum_voice",
        drum_bus=drum_bus,
        send_db=-4.0,
        mix_db=-8.0,
        synth_overrides={
            "exciter_type": "multi_tap",
            "exciter_n_taps": 8,
            "exciter_tap_spacing_s": 0.004,
            "exciter_tap_acceleration": 0.7,
            "exciter_tap_decay_s": 0.005,
            "exciter_tap_crescendo": 0.6,
            "exciter_tap_freq_spread": 0.5,
            "exciter_level": 0.03,
            "noise_decay_s": 0.120,
            "noise_center_ratio": 1.2,
            "tone_type": None,
        },
    )
    score.voices["texture"].sends.append(VoiceSend(target="hall", send_db=-8.0))

    # -------------------------------------------------------------------
    # Tonal voice — harmonic glue pad
    # -------------------------------------------------------------------

    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "additive",
            "n_harmonics": 8,
            "harmonic_rolloff": 0.55,
            "brightness_tilt": -0.15,
            "attack": 1.5,
            "release": 3.0,
            "unison_voices": 2,
            "detune_cents": 3.0,
        },
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "lowpass",
                            "cutoff_hz": 2000.0,
                            "slope_db_per_oct": 12,
                        },
                        {
                            "kind": "highpass",
                            "cutoff_hz": 100.0,
                            "slope_db_per_oct": 12,
                        },
                    ],
                },
            ),
        ],
        normalize_lufs=-24.0,
        mix_db=-10.0,
        sends=[VoiceSend(target="hall", send_db=-4.0)],
    )

    # -------------------------------------------------------------------
    # Place notes
    # -------------------------------------------------------------------

    _place_kick(score)
    _place_bells(score)
    _place_hats(score)
    _place_snare(score)
    _place_texture(score)
    _place_pad(score)

    return score


# ---------------------------------------------------------------------------
# Placement functions
# ---------------------------------------------------------------------------


def _kick_partial_at(bar: int, beat: int) -> float:
    """Which harmonic the kick rings at for this bar/beat."""
    cycle = KICK_BASS_LINE[(bar - 1) % len(KICK_BASS_LINE)]
    return cycle[0] if beat <= 2 else cycle[1]


def _place_kick(score: Score) -> None:
    for bar in range(1, TOTAL_BARS + 1):
        sec = _section(bar)
        dyn = _dynamics(bar)

        if sec == "dissolve" and bar > 30:
            # Final bars: single long kick
            if bar == 31:
                score.add_note(
                    "kick",
                    start=_pos(bar),
                    duration=BAR * 3,
                    partial=P1,
                    amp_db=-4.0,
                )
            continue

        for beat in (1, 3):
            partial = _kick_partial_at(bar, beat)
            dur = 0.6 if sec != "dissolve" else 1.2
            db = -4.0 + dyn
            if beat == 3:
                db -= 1.5

            score.add_note(
                "kick",
                start=_pos(bar, beat),
                duration=dur,
                partial=partial,
                amp_db=db,
            )


def _place_bells(score: Score) -> None:
    """JI bell arpeggios consonant with the kick's current pitch."""
    for bar in range(1, TOTAL_BARS + 1):
        sec = _section(bar)
        dyn = _dynamics(bar)

        if sec == "dissolve" and bar > 28:
            continue  # bells fade out

        # Arpeggio on 16th notes starting on beat 2
        kick_root = _kick_partial_at(bar, 1)

        # Choose 3-4 intervals from the bell set
        n_notes = 3 if sec in ("bells", "dissolve") else 4
        for i in range(n_notes):
            interval = BELL_INTERVALS[i % len(BELL_INTERVALS)]
            bell_partial = kick_root * interval * 4  # up 2 octaves into bell register
            n16_offset = i * 2  # every other 16th
            beat = 2 + n16_offset // 4
            n16 = n16_offset % 4

            if beat > 4:
                continue

            db = -10.0 + dyn * 0.7
            # Accent pattern: first note louder
            if i == 0:
                db += 2.0
            elif i == n_notes - 1:
                db -= 2.0

            # In dissolve, fade
            if sec == "dissolve":
                db -= 1.5 * (bar - 24)

            score.add_note(
                "bell",
                start=_pos(bar, beat, n16),
                duration=0.3,
                partial=bell_partial,
                amp_db=db,
            )

    # Second half: mirror arpeggio on beat 4 for call-and-response
    for bar in range(9, 25):
        kick_root = _kick_partial_at(bar, 3)
        dyn = _dynamics(bar)
        for i in range(2):
            interval = BELL_INTERVALS[(i + 2) % len(BELL_INTERVALS)]
            bell_partial = kick_root * interval * 4
            db = -12.0 + dyn * 0.5
            score.add_note(
                "bell",
                start=_pos(bar, 4, i * 2),
                duration=0.25,
                partial=bell_partial,
                amp_db=db,
            )


def _place_hats(score: Score) -> None:
    """Beating hat trio — enters in pulse section."""
    for bar in range(9, TOTAL_BARS + 1):
        sec = _section(bar)
        dyn = _dynamics(bar)

        if sec == "dissolve" and bar > 28:
            continue

        for beat in range(1, 5):
            for eighth in (0, 2):  # eighth-note pulse
                db_base = -14.0 if eighth == 0 else -18.0

                if sec == "dissolve":
                    db_base -= 2.0 * (bar - 24)

                db = db_base + dyn * 0.5

                # Distribute across the three hat voices
                for hat_idx in range(3):
                    # Stagger: not all three play every hit
                    # hat_0 plays all, hat_1 skips some, hat_2 skips more
                    if hat_idx == 1 and (beat + eighth) % 3 == 0:
                        continue
                    if hat_idx == 2 and (beat + eighth) % 2 == 0:
                        continue

                    # Each hat slightly different level for shimmer
                    hat_db = db - hat_idx * 1.5

                    score.add_note(
                        f"hat_{hat_idx}",
                        start=_pos(bar, beat, eighth),
                        duration=0.04,
                        freq=7000.0 + hat_idx * 35.0,
                        amp_db=hat_db,
                    )


def _place_snare(score: Score) -> None:
    """FM snare with per-note index morphing."""
    for bar in range(9, TOTAL_BARS + 1):
        sec = _section(bar)
        dyn = _dynamics(bar)
        fm_idx = _snare_fm_index(bar)

        if sec == "dissolve" and bar > 30:
            continue

        for beat in (2, 4):
            db = -6.0 + dyn
            if beat == 4:
                db -= 1.0

            # Ghost notes in the arc section
            ghost = sec == "arc" and beat == 4
            if ghost:
                db -= 3.0

            # Noise mix increases with FM index
            noise_mix = 0.3 + 0.05 * fm_idx

            score.add_note(
                "snare",
                start=_pos(bar, beat),
                duration=0.15,
                freq=180.0,
                amp_db=db,
                synth={
                    "tone_fm_index": fm_idx,
                    "noise_level": min(0.7, noise_mix),
                    "tone_level": max(0.3, 1.0 - noise_mix),
                },
            )


def _place_texture(score: Score) -> None:
    """Granular clap textures at structural peaks."""
    accent_hits = [
        (12, 2, -8.0),
        (16, 4, -6.0),
        (18, 2, -6.0),
        (20, 2, -4.0),  # peak
        (20, 4, -5.0),
        (22, 2, -7.0),
        (24, 4, -8.0),
    ]
    for bar, beat, db in accent_hits:
        dyn = _dynamics(bar)
        score.add_note(
            "texture",
            start=_pos(bar, beat),
            duration=0.2,
            freq=2000.0,
            amp_db=db + dyn * 0.3,
        )


def _place_pad(score: Score) -> None:
    """Sustained harmonic pad — enters bar 13, supports the kick's harmony."""
    for bar in range(13, TOTAL_BARS + 1):
        sec = _section(bar)
        dyn = _dynamics(bar)

        kick_root = _kick_partial_at(bar, 1)
        dur = BAR + 0.3

        db = -18.0 + dyn * 0.5
        if sec == "dissolve":
            db -= 1.5 * (bar - 24)

        # Root + 5th above the kick, in a mid register
        for interval in (1.0, 3 / 2):
            score.add_note(
                "pad",
                start=_pos(bar),
                duration=dur,
                partial=kick_root * interval * 4,
                amp_db=db,
            )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

PIECES: dict[str, PieceDefinition] = {
    "bell_pulse": PieceDefinition(
        name="bell_pulse",
        output_name="bell_pulse",
        build_score=build_score,
        sections=(
            PieceSection("bells", _pos(1), _pos(9)),
            PieceSection("pulse", _pos(9), _pos(17)),
            PieceSection("arc", _pos(17), _pos(25)),
            PieceSection("dissolve", _pos(25), _pos(TOTAL_BARS + 1)),
        ),
    ),
}
