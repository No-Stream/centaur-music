"""Bach-style JI organ passacaglia — ground bass with variations."""

from __future__ import annotations

from code_musics.humanize import TimingHumanizeSpec
from code_musics.pieces._shared import WARM_SATURATION_EFFECT
from code_musics.pieces.registry import PieceDefinition
from code_musics.score import EffectSpec, Score, SendBusSpec, VoiceSend
from code_musics.synth import BRICASTI_IR_DIR

# -- Timing ---------------------------------------------------------------

STEP = 1.3  # seconds per chord (half-note at ~46 BPM)
DUR = 1.6  # note duration (legato overlap)
PASSING_OFFSET = 0.65  # where passing tones fall within the step
PASSING_DUR = 0.7  # duration of a passing tone

# -- Ground bass & SATB voicings -----------------------------------------
# f0 = 110 Hz (A2).  All partials relative to f0.
# Scale: A=1, B=9/8, C#=5/4, D=4/3, E=3/2, F#=5/3, G#=15/8
# Octave doublings: A3=2, C#4=5/2, E4=3, A4=4, C#5=5, E5=6, A5=8
#
# The septimal seventh (7/4 = harmonic 7th, ~969 cents) appears at chord 5
# as a structural color chord — the 7th partial of A, resolving to IV.

CHORDS: list[dict[str, float]] = [
    {"soprano": 5, "alto": 3, "tenor": 5 / 2, "bass": 2},  # I  — A major
    {"soprano": 6, "alto": 15 / 4, "tenor": 3, "bass": 3 / 2},  # V  — E major
    {"soprano": 5, "alto": 10 / 3, "tenor": 5 / 2, "bass": 5 / 3},  # vi — F# minor
    {"soprano": 16 / 3, "alto": 10 / 3, "tenor": 8 / 3, "bass": 4 / 3},  # IV — D major
    {"soprano": 5, "alto": 7 / 2, "tenor": 3, "bass": 2},  # I7♭ — septimal seventh
    {
        "soprano": 16 / 3,
        "alto": 10 / 3,
        "tenor": 8 / 3,
        "bass": 4 / 3,
    },  # IV — resolution
    {"soprano": 9 / 2, "alto": 10 / 3, "tenor": 8 / 3, "bass": 9 / 8},  # ii — B minor
    {"soprano": 6, "alto": 15 / 4, "tenor": 3, "bass": 3 / 2},  # V  — E major
]

# Passing tones between adjacent chords (soprano and tenor only).
# None means no passing tone for that transition.
SOPRANO_PASSING: list[float | None] = [
    16 / 3,  # C#→E: pass through D
    None,  # E→C#: direct leap (characteristic)
    None,  # C#→D: step, no passing needed
    5,  # D→C#: could ornament (back to C#)
    16 / 3,  # C#→D: pass through D
    5,  # D→B: pass through C#
    None,  # B→E: rising fourth (characteristic leap)
    None,  # E→C# (back to top): direct
]

TENOR_PASSING: list[float | None] = [
    None,  # C#→E: direct
    None,  # E→C#: direct
    None,  # C#→D: step
    3,  # D→E (into sept chord): pass through E
    8 / 3,  # E→D: pass back
    None,  # D→D: holds
    None,  # D→E: step
    None,  # E→C#: direct (back to top)
]


def _hall_reverb() -> EffectSpec:
    if BRICASTI_IR_DIR.exists():
        return EffectSpec(
            "bricasti",
            {"ir_name": "1 Halls 07 Large & Dark", "wet": 0.60, "lowpass_hz": 5500},
        )
    return EffectSpec("reverb", {"room_size": 0.90, "damping": 0.42, "wet_level": 0.55})


def _place_chord(
    score: Score,
    start: float,
    chord: dict[str, float],
    voices: list[str],
    velocity: float,
    note_dur: float = DUR,
) -> None:
    voice_vel_scale = {"soprano": 1.0, "alto": 0.88, "tenor": 0.84, "bass": 0.90}
    for voice_name in voices:
        if voice_name not in chord:
            continue
        vel = velocity * voice_vel_scale.get(voice_name, 1.0)
        score.add_note(
            voice_name,
            start=start,
            duration=note_dur,
            partial=chord[voice_name],
            velocity=vel,
        )


def _place_passing_tone(
    score: Score,
    voice: str,
    start: float,
    partial: float,
    velocity: float,
) -> None:
    score.add_note(
        voice,
        start=start,
        duration=PASSING_DUR,
        partial=partial,
        velocity=velocity * 0.72,
    )


def build_score() -> Score:
    """Passacaglia in A major (JI) for organ.

    A descending ground bass with a septimal seventh color chord, presented
    through six variations that build from a lone cathedral bass to a full
    four-voice climax, then thin to a quiet close.
    """
    score = Score(
        f0=110.0,
        timing_humanize=TimingHumanizeSpec(
            ensemble_amount_ms=8.0, follow_strength=0.80
        ),
        send_buses=[SendBusSpec(name="hall", effects=[_hall_reverb()])],
        master_effects=[WARM_SATURATION_EFFECT],
    )

    hall_send = VoiceSend(target="hall", send_db=-7.0)
    hall_send_deep = VoiceSend(target="hall", send_db=-10.0)

    score.add_voice(
        "soprano",
        synth_defaults={"engine": "organ", "preset": "baroque"},
        sends=[hall_send],
        pan=0.18,
    )
    score.add_voice(
        "alto",
        synth_defaults={"engine": "organ", "preset": "warm"},
        sends=[hall_send],
        pan=-0.12,
        mix_db=-1.5,
    )
    score.add_voice(
        "tenor",
        synth_defaults={"engine": "organ", "preset": "jazz"},
        sends=[hall_send],
        pan=0.06,
        mix_db=-2.0,
    )
    score.add_voice(
        "bass",
        synth_defaults={"engine": "organ", "preset": "cathedral"},
        sends=[hall_send_deep],
        pan=-0.06,
        mix_db=-0.5,
    )
    # Climax doublings
    score.add_voice(
        "soprano_high",
        synth_defaults={"engine": "organ", "preset": "warm"},
        sends=[hall_send],
        pan=0.22,
        mix_db=-5.0,
    )
    score.add_voice(
        "bass_deep",
        synth_defaults={"engine": "organ", "preset": "full"},
        sends=[hall_send_deep],
        pan=-0.08,
        mix_db=-4.0,
    )

    n_chords = len(CHORDS)
    var_dur = n_chords * STEP

    # -- Variation 1: Bass alone (cathedral) --------------------------------
    var_start = 0.0
    for i in range(n_chords):
        _place_chord(
            score,
            start=var_start + i * STEP,
            chord=CHORDS[i],
            voices=["bass"],
            velocity=0.62,
        )

    # -- Variation 2: Bass + Soprano ----------------------------------------
    var_start = var_dur
    for i in range(n_chords):
        _place_chord(
            score,
            start=var_start + i * STEP,
            chord=CHORDS[i],
            voices=["bass", "soprano"],
            velocity=0.70,
        )

    # -- Variation 3: Bass + Soprano + Alto ---------------------------------
    var_start = 2 * var_dur
    for i in range(n_chords):
        _place_chord(
            score,
            start=var_start + i * STEP,
            chord=CHORDS[i],
            voices=["bass", "soprano", "alto"],
            velocity=0.76,
        )

    # -- Variation 4: Full SATB, homophonic ---------------------------------
    var_start = 3 * var_dur
    for i in range(n_chords):
        _place_chord(
            score,
            start=var_start + i * STEP,
            chord=CHORDS[i],
            voices=["soprano", "alto", "tenor", "bass"],
            velocity=0.82,
        )

    # -- Variation 5: Full SATB + passing tones (building) ------------------
    var_start = 4 * var_dur
    for i in range(n_chords):
        chord_start = var_start + i * STEP
        _place_chord(
            score,
            start=chord_start,
            chord=CHORDS[i],
            voices=["soprano", "alto", "tenor", "bass"],
            velocity=0.88,
        )
        passing_partial = SOPRANO_PASSING[i]
        if passing_partial is not None:
            _place_passing_tone(
                score, "soprano", chord_start + PASSING_OFFSET, passing_partial, 0.88
            )
        tenor_passing = TENOR_PASSING[i]
        if tenor_passing is not None:
            _place_passing_tone(
                score, "tenor", chord_start + PASSING_OFFSET, tenor_passing, 0.88
            )

    # -- Variation 6: Climax — full voices + octave doublings ---------------
    var_start = 5 * var_dur
    for i in range(n_chords):
        chord_start = var_start + i * STEP
        _place_chord(
            score,
            start=chord_start,
            chord=CHORDS[i],
            voices=["soprano", "alto", "tenor", "bass"],
            velocity=0.95,
            note_dur=DUR * 1.1,
        )
        # Soprano octave doubling (one octave up)
        score.add_note(
            "soprano_high",
            start=chord_start,
            duration=DUR,
            partial=CHORDS[i]["soprano"] * 2,
            velocity=0.65,
        )
        # Bass sub-octave doubling
        score.add_note(
            "bass_deep",
            start=chord_start,
            duration=DUR * 1.1,
            partial=CHORDS[i]["bass"] / 2,
            velocity=0.78,
        )
        # Passing tones in the climax too
        passing_partial = SOPRANO_PASSING[i]
        if passing_partial is not None:
            _place_passing_tone(
                score, "soprano", chord_start + PASSING_OFFSET, passing_partial, 0.92
            )

    # -- Coda: thin to bass + soprano, then final cadence -------------------
    coda_start = 6 * var_dur

    # Chords 1-4: just bass and soprano, quieter
    for i in range(4):
        _place_chord(
            score,
            start=coda_start + i * STEP,
            chord=CHORDS[i],
            voices=["bass", "soprano"],
            velocity=0.65,
            note_dur=DUR * 1.2,
        )

    # Chords 5-8 (including the septimal chord): all four voices return
    # for a gentle final statement
    for i in range(4, n_chords):
        _place_chord(
            score,
            start=coda_start + i * STEP,
            chord=CHORDS[i],
            voices=["soprano", "alto", "tenor", "bass"],
            velocity=0.72,
            note_dur=DUR * 1.2,
        )

    # Final resolution: I chord, held long
    final_start = coda_start + n_chords * STEP
    final_chord = {"soprano": 4, "alto": 5 / 2, "tenor": 2, "bass": 1}
    for voice_name, partial in final_chord.items():
        vel_scale = {"soprano": 1.0, "alto": 0.88, "tenor": 0.84, "bass": 0.90}
        score.add_note(
            voice_name,
            start=final_start,
            duration=5.0,
            partial=partial,
            velocity=0.78 * vel_scale[voice_name],
        )

    return score


PIECES: dict[str, PieceDefinition] = {
    "organ_passacaglia": PieceDefinition(
        name="organ_passacaglia",
        output_name="organ_passacaglia",
        build_score=build_score,
    ),
}
