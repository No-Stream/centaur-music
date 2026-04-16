"""Dark minimal techno loop showcasing the full percussion engine palette.

32 bars at 128 BPM.  Every drum voice uses the unified drum_voice engine:
closed hat, open hat, ride bell, snare (909_tight), clap (909_clap).
Uses choke groups, drum bus routing, and the new effect presets.
"""

from __future__ import annotations

from code_musics.automation import (
    AutomationSegment,
    AutomationSpec,
    AutomationTarget,
)
from code_musics.drum_helpers import add_drum_voice, setup_drum_bus
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score, VoiceSend

# ---------------------------------------------------------------------------
# Timing grid
# ---------------------------------------------------------------------------

BPM = 128.0
BEAT = 60.0 / BPM
BAR = 4.0 * BEAT
S16 = BEAT / 4.0
F0 = 55.0  # A1
TOTAL_BARS = 32


def _pos(bar: int, beat: int = 1, n16: int = 0) -> float:
    """Bar/beat/16th → seconds.  bar and beat are 1-based."""
    return (bar - 1) * BAR + (beat - 1) * BEAT + n16 * S16


TOTAL_DUR = _pos(TOTAL_BARS + 1)

# ---------------------------------------------------------------------------
# Harmonic partials (harmonic series of F0 = 55 Hz)
# ---------------------------------------------------------------------------

P1 = 1.0  # 55 Hz  A1
P2 = 2.0  # 110     A2
P3 = 3.0  # 165     ~E3
P4 = 4.0  # 220     A3
P5 = 5.0  # 275     ~C#4
P6 = 6.0  # 330     E4
P7 = 7.0  # 385     ~Bb4 (septimal)
P8 = 8.0  # 440     A4

# ---------------------------------------------------------------------------
# Section definitions
# ---------------------------------------------------------------------------

#   1- 4   kick alone
#   5- 8   + ch hat, + bass
#   9-16   + oh hat (choked), + snare on 2&4, + clap layered
#  17-24   full groove: + ride accents, hat opens up
#  25-28   thin out: snare drops, ride stays
#  29-32   outro: kick + bass + hat fade


def _in_bars(bar: int, start: int, end: int) -> bool:
    return start <= bar <= end


# ---------------------------------------------------------------------------
# Hat pattern helpers
# ---------------------------------------------------------------------------

# Per-16th amplitude offsets: downbeat loudest, upbeat ghosts quieter
_HAT_SUBDIV_DB: dict[int, float] = {0: -10.0, 1: -16.0, 2: -13.0, 3: -17.0}

# Section-based hat amplitude offsets
_HAT_SECTION_OFFSET: dict[tuple[int, int], float] = {
    (5, 8): -3.0,  # entering
    (9, 16): -1.0,  # building
    (17, 24): 0.0,  # full
    (25, 28): -2.0,  # thinning
    (29, 32): -4.0,  # fading
}

# Section-based hat frequencies
_HAT_SECTION_FREQ: dict[tuple[int, int], float] = {
    (5, 8): 8000.0,
    (9, 16): 9000.0,
    (17, 24): 10000.0,
    (25, 28): 8500.0,
    (29, 32): 7000.0,
}


def _hat_offset(bar: int) -> float:
    for (s, e), offset in _HAT_SECTION_OFFSET.items():
        if s <= bar <= e:
            return offset
    return -6.0


def _hat_freq(bar: int) -> float:
    for (s, e), freq in _HAT_SECTION_FREQ.items():
        if s <= bar <= e:
            return freq
    return 9000.0


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_score() -> Score:
    score = Score(
        f0_hz=F0,
        sample_rate=44_100,
        master_effects=[
            EffectSpec(
                "compressor",
                {
                    "threshold_db": -20.0,
                    "ratio": 2.5,
                    "attack_ms": 25.0,
                    "release_ms": 250.0,
                    "knee_db": 8.0,
                    "makeup_gain_db": 0.5,
                    "topology": "feedback",
                    "detector_mode": "rms",
                    "detector_bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 120.0,
                            "slope_db_per_oct": 12,
                        },
                    ],
                },
            ),
        ],
    )

    # -----------------------------------------------------------------------
    # Send buses
    # -----------------------------------------------------------------------

    # Drum sub-mix bus with gentle group compression
    drum_bus = setup_drum_bus(
        score,
        effects=[
            EffectSpec(
                "compressor",
                {
                    "threshold_db": -22.0,
                    "ratio": 1.8,
                    "attack_ms": 20.0,
                    "release_ms": 180.0,
                    "knee_db": 8.0,
                    "makeup_gain_db": 0.0,
                    "topology": "feedback",
                    "detector_mode": "rms",
                    "detector_bands": [
                        {"kind": "highpass", "cutoff_hz": 80.0, "slope_db_per_oct": 12},
                    ],
                },
            ),
        ],
        return_db=0.0,
    )

    # Reverb send — dark hall
    score.add_send_bus(
        "hall",
        effects=[
            EffectSpec(
                "reverb",
                {"room_size": 0.85, "damping": 0.7, "wet_level": 1.0},
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
                            "cutoff_hz": 250.0,
                            "slope_db_per_oct": 12,
                        },
                    ],
                },
            ),
        ],
        return_db=-6.0,
    )

    # -----------------------------------------------------------------------
    # Voices
    # -----------------------------------------------------------------------

    # --- Kick ---
    add_drum_voice(
        score,
        "kick",
        engine="drum_voice",
        preset="909_techno",
        drum_bus=drum_bus,
        send_db=-2.0,
        effects=[EffectSpec("compressor", {"preset": "kick_punch"})],
        mix_db=0.0,
    )

    # --- Closed hat ---
    add_drum_voice(
        score,
        "closed_hat",
        engine="drum_voice",
        preset="closed_hat",
        drum_bus=drum_bus,
        send_db=-4.0,
        choke_group="hats",
        effects=[
            EffectSpec("compressor", {"preset": "hat_control"}),
            EffectSpec(
                "eq",
                {"bands": [{"kind": "high_shelf", "freq_hz": 8000.0, "gain_db": 2.5}]},
            ),
        ],
        mix_db=-6.0,
    )

    # --- Open hat (choked by closed hat) ---
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
        mix_db=-5.0,
    )

    # --- Snare ---
    add_drum_voice(
        score,
        "snare",
        engine="drum_voice",
        preset="909_tight",
        drum_bus=drum_bus,
        send_db=-3.0,
        effects=[
            EffectSpec("compressor", {"preset": "snare_punch"}),
            EffectSpec("saturation", {"preset": "snare_bite"}),
        ],
        mix_db=-2.0,
    )
    # Snare also sends to reverb
    score.voices["snare"].sends.append(VoiceSend(target="hall", send_db=-12.0))

    # --- Clap ---
    add_drum_voice(
        score,
        "clap",
        engine="drum_voice",
        preset="909_clap",
        drum_bus=drum_bus,
        send_db=-3.0,
        mix_db=-3.0,
    )
    # Clap sends to reverb
    score.voices["clap"].sends.append(VoiceSend(target="hall", send_db=-10.0))

    # --- Ride bell ---
    add_drum_voice(
        score,
        "ride",
        engine="drum_voice",
        preset="ride_bell",
        drum_bus=drum_bus,
        send_db=-4.0,
        effects=[
            EffectSpec(
                "delay",
                {
                    "delay_seconds": BEAT * 0.75,  # dotted-eighth
                    "feedback": 0.35,
                    "mix": 0.30,
                },
            ),
        ],
        mix_db=-7.0,
    )
    # Ride sends to reverb
    score.voices["ride"].sends.append(VoiceSend(target="hall", send_db=-8.0))

    # --- Bass ---
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "polyblep",
            "preset": "sub_bass",
            "cutoff_hz": 350.0,
            "resonance_q": 1.2,
        },
        effects=[
            EffectSpec(
                "compressor",
                {"preset": "kick_duck", "sidechain_source": "kick"},
            ),
        ],
        mix_db=-3.0,
        velocity_humanize=None,
    )

    # --- Pad (dark wash, very quiet) ---
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "additive",
            "preset": "soft_pad",
            "brightness": 0.25,
            "brightness_tilt": -0.3,
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
            EffectSpec(
                "compressor",
                {"preset": "kick_duck", "sidechain_source": "kick"},
            ),
        ],
        mix_db=-9.0,
        velocity_humanize=None,
        sends=[VoiceSend(target="hall", send_db=-6.0)],
        automation=[
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="brightness"),
                segments=(
                    AutomationSegment(
                        start=_pos(1),
                        end=_pos(17),
                        shape="linear",
                        start_value=0.15,
                        end_value=0.35,
                    ),
                    AutomationSegment(
                        start=_pos(17),
                        end=_pos(32),
                        shape="linear",
                        start_value=0.35,
                        end_value=0.15,
                    ),
                ),
            ),
        ],
    )

    # -----------------------------------------------------------------------
    # Note placement
    # -----------------------------------------------------------------------

    _place_kick(score)
    _place_closed_hat(score)
    _place_open_hat(score)
    _place_snare(score)
    _place_clap(score)
    _place_ride(score)
    _place_bass(score)
    _place_pad(score)

    return score


# ---------------------------------------------------------------------------
# Pattern placement functions
# ---------------------------------------------------------------------------


def _place_kick(score: Score) -> None:
    """Four-on-the-floor for all 32 bars."""
    for bar in range(1, TOTAL_BARS + 1):
        for beat in range(1, 5):
            score.add_note(
                "kick",
                start=_pos(bar, beat),
                duration=1.0,
                freq=60.0,
                amp_db=-6.0,
            )


def _place_closed_hat(score: Score) -> None:
    """16th notes, bars 5-32.  Velocity-shaped groove."""
    # Open hat plays on "and" of beats 2 and 4 (n16=2) in bars 9-24.
    # Skip closed hat there so the choke group doesn't kill the open hat
    # at the same sample.
    for bar in range(5, TOTAL_BARS + 1):
        section_offset = _hat_offset(bar)
        freq = _hat_freq(bar)
        for beat in range(1, 5):
            for n16 in range(4):
                subdiv_db = _HAT_SUBDIV_DB[n16]
                # Skip where open hat plays (choke group needs the open hat
                # to ring before a LATER closed hat chokes it)
                if 9 <= bar <= 24 and beat in {2, 4} and n16 == 2:
                    continue
                # Skip some 16ths in bars 25-28 for sparser feel
                if 25 <= bar <= 28 and n16 in {1, 3} and beat in {2, 4}:
                    continue
                # Skip 16ths in final bars for fade feel
                if 29 <= bar <= 32 and n16 in {1, 3}:
                    continue
                score.add_note(
                    "closed_hat",
                    start=_pos(bar, beat, n16),
                    duration=0.04,
                    freq=freq,
                    amp_db=subdiv_db + section_offset,
                )


def _place_open_hat(score: Score) -> None:
    """Off-beat open hats, bars 9-24.  Choked by closed hat."""
    for bar in range(9, 25):
        # Open hat on the "and" of beats 2 and 4
        for beat in [2, 4]:
            score.add_note(
                "open_hat",
                start=_pos(bar, beat, 2),  # the "and"
                duration=0.35,
                freq=9000.0,
                amp_db=-9.0,
            )


def _place_snare(score: Score) -> None:
    """Beats 2 & 4, bars 9-24.  Some ghost notes."""
    for bar in range(9, 25):
        for beat in [2, 4]:
            score.add_note(
                "snare",
                start=_pos(bar, beat),
                duration=0.25,
                freq=200.0,
                amp_db=-5.0,
            )
        # Ghost note on the "e" of beat 3 every other bar
        if bar % 2 == 0:
            score.add_note(
                "snare",
                start=_pos(bar, 3, 1),
                duration=0.15,
                freq=200.0,
                amp_db=-14.0,
            )


def _place_clap(score: Score) -> None:
    """Clap layered on snare every other bar, bars 9-24."""
    for bar in range(9, 25):
        if bar % 2 == 1:
            for beat in [2, 4]:
                score.add_note(
                    "clap",
                    start=_pos(bar, beat),
                    duration=0.12,
                    freq=2800.0,
                    amp_db=-5.0,
                )


def _place_ride(score: Score) -> None:
    """Sparse ride bell accents, bars 17-28."""
    # Quarter notes on beats 1 and 3, plus a syncopated hit
    for bar in range(17, 29):
        score.add_note(
            "ride",
            start=_pos(bar, 1),
            duration=0.4,
            freq=4500.0,
            amp_db=-8.0,
        )
        score.add_note(
            "ride",
            start=_pos(bar, 3),
            duration=0.4,
            freq=4500.0,
            amp_db=-10.0,
        )
        # Syncopated accent on "and" of beat 4 every 4th bar
        if bar % 4 == 0:
            score.add_note(
                "ride",
                start=_pos(bar, 4, 2),
                duration=0.3,
                freq=5000.0,
                amp_db=-7.0,
            )


def _place_bass(score: Score) -> None:
    """Minimal sub bass, bars 5-32.  Root + occasional fifth."""
    for bar in range(5, TOTAL_BARS + 1):
        # Root on beat 1
        score.add_note(
            "bass",
            start=_pos(bar, 1),
            duration=BEAT * 3.5,
            partial=P2,
            amp_db=-8.0,
        )
        # Fifth (P3) on the "and" of beat 3, every other bar
        if bar % 2 == 0 and _in_bars(bar, 9, 28):
            score.add_note(
                "bass",
                start=_pos(bar, 3, 2),
                duration=BEAT * 1.0,
                partial=P3,
                amp_db=-12.0,
            )
        # Octave push on beat 4 "e" for momentum, every 4th bar
        if bar % 4 == 3 and _in_bars(bar, 9, 24):
            score.add_note(
                "bass",
                start=_pos(bar, 4, 1),
                duration=BEAT * 0.5,
                partial=P4,
                amp_db=-11.0,
            )


def _place_pad(score: Score) -> None:
    """Dark ambient wash, bars 9-28.  Simple two-note voicing."""
    for bar in range(9, 29):
        # Dark open fifth: P4 + P6 (A3 + E4)
        for partial, db in [(P4, -10.0), (P6, -14.0)]:
            score.add_note(
                "pad",
                start=_pos(bar, 1),
                duration=BAR * 0.95,
                partial=partial,
                amp_db=db,
            )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

PIECES: dict[str, PieceDefinition] = {
    "iron_pulse": PieceDefinition(
        name="iron_pulse",
        output_name="iron_pulse",
        build_score=build_score,
        sections=(
            PieceSection("kick_intro", _pos(1), _pos(5)),
            PieceSection("groove_builds", _pos(5), _pos(9)),
            PieceSection("full_rhythm", _pos(9), _pos(17)),
            PieceSection("peak_groove", _pos(17), _pos(25)),
            PieceSection("thin_out", _pos(25), _pos(29)),
            PieceSection("outro", _pos(29), _pos(33)),
        ),
    ),
}
