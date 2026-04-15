"""Iron Pulse v2 — dark minimal techno showcasing the upgraded drum palette.

32 bars at 128 BPM.  Builds on iron_pulse with multi-point envelopes, FM
body synthesis, per-oscillator waveshaping, filter sweeps on hats, gated
snare/clap, and a noise_perc texture layer.

Section map:
    1- 4   FM kick alone (harmonically rich attack)
    5- 8   + hats (filter opening), + bass
    9-12   + gated snare, + gated clap, hats open
   13-16   + open hats (choked), + noise_perc texture, kick gains filter
   17-20   peak A: foldback kick, + decaying_bell ride, hat filter widest
   21-24   peak B: kick pitch envelope evolves, perc fills intensify
   25-28   breakdown: clean FM kick, hats thin, snare drops, ride stays
   29-32   outro: pitch_dive kick, sparse hats closing
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

BPM = 128.0
BEAT = 60.0 / BPM
BAR = 4.0 * BEAT
S16 = BEAT / 4.0
F0 = 55.0  # A1
TOTAL_BARS = 32

P2 = 2.0  # 110 Hz  A2
P3 = 3.0  # 165 Hz  ~E3
P4 = 4.0  # 220 Hz  A3
P6 = 6.0  # 330 Hz  E4
P7 = 7.0  # 385 Hz  ~Bb4 (septimal)


def _pos(bar: int, beat: int = 1, n16: int = 0) -> float:
    return (bar - 1) * BAR + (beat - 1) * BEAT + n16 * S16


TOTAL_DUR = _pos(TOTAL_BARS + 1)

# ---------------------------------------------------------------------------
# Per-section kick synth overrides
# ---------------------------------------------------------------------------

# Bars 1-8: FM kick — harmonically rich metallic attack thinning to clean sub
_KICK_FM: dict = {
    "body_fm_ratio": 1.41,
    "body_fm_index": 3.5,
    "body_fm_index_envelope": [
        {"time": 0.0, "value": 1.0},
        {"time": 0.06, "value": 0.08, "curve": "exponential"},
        {"time": 1.0, "value": 0.0, "curve": "linear"},
    ],
}

# Bars 9-16: FM + lowpass filter sweep (darker body, opens then closes)
_KICK_FILTERED: dict = {
    **_KICK_FM,
    "body_filter_mode": "lowpass",
    "body_filter_cutoff_hz": 1400.0,
    "body_filter_q": 1.1,
    "body_filter_envelope": [
        {"time": 0.0, "value": 3500.0},
        {"time": 0.10, "value": 1400.0, "curve": "exponential"},
        {"time": 1.0, "value": 600.0, "curve": "linear"},
    ],
}

# Bars 17-24: Foldback distortion for aggression at the peak
_KICK_FOLDBACK: dict = {
    **_KICK_FM,
    "body_distortion": "foldback",
    "body_distortion_drive": 0.40,
    "body_distortion_mix": 0.55,
    "body_filter_mode": "lowpass",
    "body_filter_cutoff_hz": 1800.0,
    "body_filter_q": 0.9,
    "body_filter_envelope": [
        {"time": 0.0, "value": 5000.0},
        {"time": 0.08, "value": 1800.0, "curve": "exponential"},
        {"time": 1.0, "value": 800.0, "curve": "linear"},
    ],
}

# Bars 29-32: Pitch dive — alien outro
_KICK_DIVE: dict = {
    "pitch_envelope": [
        {"time": 0.0, "value": 5.0},
        {"time": 0.025, "value": 2.0, "curve": "bezier", "cx": 0.1, "cy": 0.9},
        {"time": 0.12, "value": 1.0, "curve": "exponential"},
    ],
    "body_amp_envelope": [
        {"time": 0.0, "value": 1.0},
        {"time": 0.7, "value": 0.6, "curve": "linear"},
        {"time": 0.8, "value": 0.0, "curve": "exponential"},
    ],
}

# ---------------------------------------------------------------------------
# Hat filter profiles — the hat "opens" and "closes" across sections
# ---------------------------------------------------------------------------

# Closed tight (bars 5-8)
_HAT_FILTER_TIGHT: list[dict] = [
    {"time": 0.0, "value": 5000.0},
    {"time": 0.5, "value": 3000.0, "curve": "exponential"},
    {"time": 1.0, "value": 2000.0, "curve": "linear"},
]

# Opening (bars 9-16)
_HAT_FILTER_MID: list[dict] = [
    {"time": 0.0, "value": 7000.0},
    {"time": 0.4, "value": 5000.0, "curve": "exponential"},
    {"time": 1.0, "value": 3500.0, "curve": "linear"},
]

# Wide open (bars 17-24)
_HAT_FILTER_OPEN: list[dict] = [
    {"time": 0.0, "value": 12000.0},
    {"time": 0.3, "value": 8000.0, "curve": "exponential"},
    {"time": 1.0, "value": 5000.0, "curve": "linear"},
]

# Closing (bars 25-28)
_HAT_FILTER_CLOSING: list[dict] = [
    {"time": 0.0, "value": 6000.0},
    {"time": 0.5, "value": 3500.0, "curve": "exponential"},
    {"time": 1.0, "value": 2000.0, "curve": "linear"},
]

# Minimal (bars 29-32)
_HAT_FILTER_MINIMAL: list[dict] = [
    {"time": 0.0, "value": 4000.0},
    {"time": 0.4, "value": 2000.0, "curve": "exponential"},
    {"time": 1.0, "value": 1200.0, "curve": "linear"},
]

_HAT_SUBDIV_DB: dict[int, float] = {0: -10.0, 1: -16.0, 2: -13.0, 3: -17.0}


def _hat_filter_for_bar(bar: int) -> list[dict]:
    if bar <= 8:
        return _HAT_FILTER_TIGHT
    if bar <= 16:
        return _HAT_FILTER_MID
    if bar <= 24:
        return _HAT_FILTER_OPEN
    if bar <= 28:
        return _HAT_FILTER_CLOSING
    return _HAT_FILTER_MINIMAL


def _kick_synth_for_bar(bar: int) -> dict:
    if bar <= 8:
        return _KICK_FM
    if bar <= 16:
        return _KICK_FILTERED
    if bar <= 24:
        return _KICK_FOLDBACK
    if bar <= 28:
        return _KICK_FM
    return _KICK_DIVE


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
                    "threshold_db": -18.0,
                    "ratio": 2.8,
                    "attack_ms": 20.0,
                    "release_ms": 200.0,
                    "knee_db": 6.0,
                    "makeup_gain_db": 1.0,
                    "topology": "feedback",
                    "detector_mode": "rms",
                    "detector_bands": [
                        {
                            "kind": "highpass",
                            "cutoff_hz": 100.0,
                            "slope_db_per_oct": 12,
                        },
                    ],
                },
            ),
        ],
    )

    # --- Send buses ---

    drum_bus = setup_drum_bus(
        score,
        effects=[
            EffectSpec(
                "compressor",
                {
                    "threshold_db": -20.0,
                    "ratio": 2.0,
                    "attack_ms": 15.0,
                    "release_ms": 160.0,
                    "knee_db": 6.0,
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

    score.add_send_bus(
        "hall",
        effects=[
            EffectSpec(
                "reverb",
                {"room_size": 0.88, "damping": 0.65, "wet_level": 1.0},
            ),
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "lowpass",
                            "cutoff_hz": 4500.0,
                            "slope_db_per_oct": 12,
                        },
                        {
                            "kind": "highpass",
                            "cutoff_hz": 300.0,
                            "slope_db_per_oct": 12,
                        },
                    ],
                },
            ),
        ],
        return_db=-6.0,
    )

    # --- Voices ---

    # Kick: base preset is 909_techno, FM/filter/distortion via per-note overrides
    add_drum_voice(
        score,
        "kick",
        engine="kick_tom",
        preset="909_techno",
        drum_bus=drum_bus,
        send_db=-2.0,
        effects=[EffectSpec("compressor", {"preset": "kick_punch"})],
        mix_db=0.0,
    )

    # Closed hat: metallic_perc with per-note filter_envelope
    add_drum_voice(
        score,
        "closed_hat",
        engine="metallic_perc",
        preset="closed_hat",
        drum_bus=drum_bus,
        send_db=-4.0,
        choke_group="hats",
        effects=[
            EffectSpec("compressor", {"preset": "hat_control"}),
            EffectSpec(
                "eq",
                {"bands": [{"kind": "high_shelf", "freq_hz": 8000.0, "gain_db": 2.0}]},
            ),
        ],
        mix_db=-6.0,
    )

    # Open hat
    add_drum_voice(
        score,
        "open_hat",
        engine="metallic_perc",
        preset="open_hat",
        drum_bus=drum_bus,
        send_db=-4.0,
        choke_group="hats",
        effects=[
            EffectSpec(
                "eq",
                {"bands": [{"kind": "high_shelf", "freq_hz": 7000.0, "gain_db": 1.5}]},
            ),
        ],
        mix_db=-5.0,
    )

    # Snare: gated for tighter modern feel
    add_drum_voice(
        score,
        "snare",
        engine="snare",
        preset="gated_snare",
        drum_bus=drum_bus,
        send_db=-3.0,
        effects=[
            EffectSpec("compressor", {"preset": "snare_punch"}),
            EffectSpec("saturation", {"preset": "snare_bite"}),
        ],
        mix_db=-2.0,
    )
    score.voices["snare"].sends.append(VoiceSend(target="hall", send_db=-14.0))

    # Clap: gated
    add_drum_voice(
        score,
        "clap",
        engine="clap",
        preset="gated_clap",
        drum_bus=drum_bus,
        send_db=-3.0,
        mix_db=-3.0,
    )
    score.voices["clap"].sends.append(VoiceSend(target="hall", send_db=-10.0))

    # Ride: decaying_bell for more character than plain ride_bell
    add_drum_voice(
        score,
        "ride",
        engine="metallic_perc",
        preset="decaying_bell",
        drum_bus=drum_bus,
        send_db=-4.0,
        effects=[
            EffectSpec(
                "delay",
                {"delay_seconds": BEAT * 0.75, "feedback": 0.30, "mix": 0.25},
            ),
        ],
        mix_db=-7.0,
    )
    score.voices["ride"].sends.append(VoiceSend(target="hall", send_db=-8.0))

    # Perc texture: noise_perc shaped_hit for fills and ghost accents
    add_drum_voice(
        score,
        "perc",
        engine="noise_perc",
        preset="shaped_hit",
        drum_bus=drum_bus,
        send_db=-5.0,
        mix_db=-8.0,
    )
    score.voices["perc"].sends.append(VoiceSend(target="hall", send_db=-12.0))

    # Bass
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "polyblep",
            "preset": "sub_bass",
            "cutoff_hz": 320.0,
            "resonance_q": 1.3,
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

    # Pad (dark wash)
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "additive",
            "preset": "soft_pad",
            "brightness": 0.20,
            "brightness_tilt": -0.35,
        },
        effects=[
            EffectSpec(
                "eq",
                {
                    "bands": [
                        {
                            "kind": "lowpass",
                            "cutoff_hz": 1800.0,
                            "slope_db_per_oct": 12,
                        },
                        {
                            "kind": "highpass",
                            "cutoff_hz": 120.0,
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
        mix_db=-10.0,
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
                        start_value=0.12,
                        end_value=0.30,
                    ),
                    AutomationSegment(
                        start=_pos(17),
                        end=_pos(32),
                        shape="linear",
                        start_value=0.30,
                        end_value=0.10,
                    ),
                ),
            ),
        ],
    )

    # --- Note placement ---
    _place_kick(score)
    _place_closed_hat(score)
    _place_open_hat(score)
    _place_snare(score)
    _place_clap(score)
    _place_ride(score)
    _place_perc(score)
    _place_bass(score)
    _place_pad(score)

    return score


# ---------------------------------------------------------------------------
# Pattern placement
# ---------------------------------------------------------------------------


def _place_kick(score: Score) -> None:
    for bar in range(1, TOTAL_BARS + 1):
        synth_overrides = _kick_synth_for_bar(bar)
        for beat in range(1, 5):
            score.add_note(
                "kick",
                start=_pos(bar, beat),
                duration=1.0,
                freq=60.0,
                amp_db=-6.0,
                synth=synth_overrides,
            )


def _place_closed_hat(score: Score) -> None:
    for bar in range(5, TOTAL_BARS + 1):
        hat_filter = _hat_filter_for_bar(bar)
        for beat in range(1, 5):
            for n16 in range(4):
                # Skip where open hat plays
                if 13 <= bar <= 24 and beat in {2, 4} and n16 == 2:
                    continue
                # Thinner in bars 25-28
                if 25 <= bar <= 28 and n16 in {1, 3} and beat in {2, 4}:
                    continue
                # Sparse in outro
                if 29 <= bar <= 32 and n16 in {1, 3}:
                    continue

                subdiv_db = _HAT_SUBDIV_DB[n16]
                score.add_note(
                    "closed_hat",
                    start=_pos(bar, beat, n16),
                    duration=0.04,
                    freq=9000.0,
                    amp_db=subdiv_db,
                    synth={"filter_envelope": hat_filter},
                )


def _place_open_hat(score: Score) -> None:
    for bar in range(13, 25):
        for beat in [2, 4]:
            score.add_note(
                "open_hat",
                start=_pos(bar, beat, 2),
                duration=0.35,
                freq=9000.0,
                amp_db=-9.0,
            )


def _place_snare(score: Score) -> None:
    for bar in range(9, 25):
        for beat in [2, 4]:
            score.add_note(
                "snare",
                start=_pos(bar, beat),
                duration=0.25,
                freq=200.0,
                amp_db=-5.0,
            )
        # Ghost notes
        if bar % 2 == 0:
            score.add_note(
                "snare",
                start=_pos(bar, 3, 1),
                duration=0.15,
                freq=200.0,
                amp_db=-14.0,
            )
        # Extra ghost on "a" of beat 1 every 4th bar for momentum
        if bar % 4 == 0:
            score.add_note(
                "snare",
                start=_pos(bar, 1, 3),
                duration=0.12,
                freq=200.0,
                amp_db=-16.0,
            )


def _place_clap(score: Score) -> None:
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
    for bar in range(17, 29):
        score.add_note(
            "ride",
            start=_pos(bar, 1),
            duration=0.5,
            freq=4500.0,
            amp_db=-8.0,
        )
        score.add_note(
            "ride",
            start=_pos(bar, 3),
            duration=0.5,
            freq=4500.0,
            amp_db=-10.0,
        )
        if bar % 4 == 0:
            score.add_note(
                "ride",
                start=_pos(bar, 4, 2),
                duration=0.35,
                freq=5000.0,
                amp_db=-7.0,
            )


def _place_perc(score: Score) -> None:
    """Noise_perc texture hits: offbeat accents and fills, bars 13-24."""
    for bar in range(13, 25):
        # Offbeat 16th ghost hits
        score.add_note(
            "perc",
            start=_pos(bar, 2, 3),
            duration=0.08,
            freq=350.0,
            amp_db=-12.0,
        )
        score.add_note(
            "perc",
            start=_pos(bar, 4, 1),
            duration=0.08,
            freq=400.0,
            amp_db=-13.0,
        )
        # Extra fill every 4th bar
        if bar % 4 == 0:
            for n16 in [1, 2, 3]:
                score.add_note(
                    "perc",
                    start=_pos(bar, 4, n16),
                    duration=0.06,
                    freq=300.0 + n16 * 80.0,
                    amp_db=-11.0 - n16 * 1.5,
                )


def _place_bass(score: Score) -> None:
    for bar in range(5, TOTAL_BARS + 1):
        score.add_note(
            "bass",
            start=_pos(bar, 1),
            duration=BEAT * 3.5,
            partial=P2,
            amp_db=-8.0,
        )
        if bar % 2 == 0 and 9 <= bar <= 28:
            score.add_note(
                "bass",
                start=_pos(bar, 3, 2),
                duration=BEAT,
                partial=P3,
                amp_db=-12.0,
            )
        if bar % 4 == 3 and 9 <= bar <= 24:
            score.add_note(
                "bass",
                start=_pos(bar, 4, 1),
                duration=BEAT * 0.5,
                partial=P4,
                amp_db=-11.0,
            )
        # Septimal 7th push in peak section every 8th bar
        if bar % 8 == 4 and 17 <= bar <= 24:
            score.add_note(
                "bass",
                start=_pos(bar, 4, 3),
                duration=BEAT * 0.4,
                partial=P7,
                amp_db=-14.0,
            )


def _place_pad(score: Score) -> None:
    for bar in range(9, 29):
        for partial, db in [(P4, -10.0), (P6, -14.0)]:
            score.add_note(
                "pad",
                start=_pos(bar, 1),
                duration=BAR * 0.95,
                partial=partial,
                amp_db=db,
            )
        # Add septimal 7th to pad voicing in peak section
        if 17 <= bar <= 24:
            score.add_note(
                "pad",
                start=_pos(bar, 1),
                duration=BAR * 0.95,
                partial=P7,
                amp_db=-18.0,
            )


PIECES: dict[str, PieceDefinition] = {
    "iron_pulse_v2": PieceDefinition(
        name="iron_pulse_v2",
        output_name="iron_pulse_v2",
        build_score=build_score,
        sections=(
            PieceSection("fm_kick_intro", _pos(1), _pos(5)),
            PieceSection("hats_enter", _pos(5), _pos(9)),
            PieceSection("full_rhythm", _pos(9), _pos(13)),
            PieceSection("texture_builds", _pos(13), _pos(17)),
            PieceSection("peak_foldback", _pos(17), _pos(21)),
            PieceSection("peak_evolve", _pos(21), _pos(25)),
            PieceSection("breakdown", _pos(25), _pos(29)),
            PieceSection("outro_dive", _pos(29), _pos(33)),
        ),
    ),
}
