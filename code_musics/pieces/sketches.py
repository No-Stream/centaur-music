"""Experimental sketches exploring six compositional ideas."""

from __future__ import annotations

import logging

from code_musics.composition import ArticulationSpec, RhythmCell, line
from code_musics.pieces.septimal import PieceDefinition
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import EffectSpec, Phrase, Score
from code_musics.tuning import utonal

logger = logging.getLogger(__name__)

_REVERB = EffectSpec("bricasti", {"ir_name": "1 Halls 07 Large & Dark", "wet": 0.32})
_SOFT_REVERB = EffectSpec("bricasti", {"ir_name": "1 Halls 07 Large & Dark", "wet": 0.25})
_DELAY = EffectSpec("delay", {"delay_seconds": 0.32, "feedback": 0.22, "mix": 0.16})


def build_passacaglia_sketch() -> Score:
    """Descending bass ostinato (8-7-6-5-4) with five accumulating variations."""
    score = Score(
        f0=55.0,
        master_effects=[_DELAY, _REVERB],
    )

    score.add_voice(
        "bass",
        synth_defaults={
            "harmonic_rolloff": 0.52,
            "attack": 0.04,
            "decay": 0.12,
            "sustain_level": 0.85,
            "release": 0.75,
        },
    )
    score.add_voice(
        "upper",
        synth_defaults={
            "harmonic_rolloff": 0.28,
            "attack": 0.07,
            "decay": 0.14,
            "sustain_level": 0.68,
            "release": 0.95,
        },
    )
    score.add_voice(
        "mid",
        synth_defaults={
            "harmonic_rolloff": 0.36,
            "attack": 0.80,
            "decay": 0.20,
            "sustain_level": 0.75,
            "release": 2.5,
        },
    )
    score.add_voice(
        "alto",
        synth_defaults={
            "harmonic_rolloff": 0.34,
            "attack": 0.05,
            "decay": 0.10,
            "sustain_level": 0.73,
            "release": 0.60,
        },
    )

    ground = Phrase.from_partials(
        [8, 7, 6, 5, 4],
        note_dur=2.2,
        step=2.0,
        amp=0.44,
        synth_defaults={
            "harmonic_rolloff": 0.52,
            "attack": 0.04,
            "decay": 0.12,
            "sustain_level": 0.85,
            "release": 0.75,
        },
    )
    ground_dur = ground.duration  # ~10.2s

    # Ground repeats five times
    for rep in range(5):
        score.add_phrase("bass", ground, start=rep * ground_dur)

    # Var 2: upper melody enters
    upper_phrase = Phrase.from_partials(
        [12, 14, 13, 12, 11, 12],
        note_dur=1.6,
        step=1.45,
        amp=0.28,
        synth_defaults={
            "harmonic_rolloff": 0.26,
            "attack": 0.07,
            "decay": 0.14,
            "sustain_level": 0.66,
            "release": 0.95,
        },
    )
    score.add_phrase("upper", upper_phrase, start=ground_dur)
    score.add_phrase("upper", upper_phrase, start=ground_dur * 2, partial_shift=2.0)

    # Var 3: mid sustained tones swell in
    for partial, offset in [(5, 0.0), (6, 1.0), (7, 2.2)]:
        score.add_note("mid", start=ground_dur * 2 + offset, duration=8.5, partial=partial, amp=0.20)

    # Var 4: alto countermelody
    alto_phrase = Phrase.from_partials(
        [9, 10, 9, 8, 10, 9],
        note_dur=1.4,
        step=1.2,
        amp=0.32,
        synth_defaults={
            "harmonic_rolloff": 0.34,
            "attack": 0.05,
            "decay": 0.10,
            "sustain_level": 0.74,
            "release": 0.58,
        },
    )
    score.add_phrase("alto", alto_phrase, start=ground_dur * 3)
    score.add_phrase("alto", alto_phrase, start=ground_dur * 3 + 5.0, partial_shift=-1.0, amp_scale=0.80)

    # Var 5: climax — all voices converge, bloom of sustained tones
    score.add_phrase("upper", upper_phrase, start=ground_dur * 4, partial_shift=4.0, amp_scale=0.68)
    score.add_phrase("alto", alto_phrase, start=ground_dur * 4, amp_scale=1.05)
    for partial, offset in [(8, 0.0), (10, 0.7), (12, 1.4), (7, 0.4)]:
        score.add_note("mid", start=ground_dur * 4 + offset, duration=10.5, partial=partial, amp=0.14)

    return score


def build_invention_sketch() -> Score:
    """Two-voice imitative counterpoint on a six-note JI subject."""
    score = Score(
        f0=110.0,
        master_effects=[_DELAY, _REVERB],
    )

    score.add_voice(
        "voice_a",
        synth_defaults={
            "harmonic_rolloff": 0.34,
            "attack": 0.05,
            "decay": 0.12,
            "sustain_level": 0.72,
            "release": 0.65,
        },
    )
    score.add_voice(
        "voice_b",
        synth_defaults={
            "harmonic_rolloff": 0.30,
            "attack": 0.06,
            "decay": 0.14,
            "sustain_level": 0.68,
            "release": 0.70,
        },
    )
    score.add_voice(
        "pedal",
        synth_defaults={
            "harmonic_rolloff": 0.48,
            "attack": 0.90,
            "decay": 0.20,
            "sustain_level": 0.80,
            "release": 3.0,
        },
    )

    subject = Phrase.from_partials(
        [6, 7, 8, 9, 8, 7],
        note_dur=1.3,
        step=1.0,
        amp=0.38,
        synth_defaults={
            "harmonic_rolloff": 0.34,
            "attack": 0.05,
            "decay": 0.12,
            "sustain_level": 0.72,
            "release": 0.65,
        },
    )
    subject_dur = subject.duration  # ~6.3s

    # Subject stated alone in voice A
    score.add_phrase("voice_a", subject, start=0.0)

    # Answer: voice B enters mid-way, partial_shift=+4 (a fourth up in harmonic space)
    answer_start = 4.0
    score.add_phrase("voice_b", subject, start=answer_start, partial_shift=4)

    # Development: two fragments traded between voices
    head = Phrase.from_partials(
        [6, 7, 8, 9],
        note_dur=1.1,
        step=0.85,
        amp=0.35,
        synth_defaults={
            "harmonic_rolloff": 0.34,
            "attack": 0.05,
            "decay": 0.10,
            "sustain_level": 0.70,
            "release": 0.55,
        },
    )
    tail = Phrase.from_partials(
        [9, 8, 7, 6],
        note_dur=1.1,
        step=0.85,
        amp=0.33,
        synth_defaults={
            "harmonic_rolloff": 0.34,
            "attack": 0.05,
            "decay": 0.10,
            "sustain_level": 0.70,
            "release": 0.55,
        },
    )

    dev_start = answer_start + subject_dur + 1.0
    score.add_phrase("voice_a", head, start=dev_start)
    score.add_phrase("voice_b", tail, start=dev_start + 0.6, partial_shift=3)
    score.add_phrase("voice_b", head, start=dev_start + head.duration + 0.8, partial_shift=4)
    score.add_phrase("voice_a", tail, start=dev_start + head.duration + 1.4, partial_shift=-1)

    # Stretto: subject in both voices, now only 2s apart instead of 4s
    stretto_start = dev_start + head.duration + tail.duration + 2.0
    score.add_phrase("voice_a", subject, start=stretto_start)
    score.add_phrase("voice_b", subject, start=stretto_start + 2.0, partial_shift=4)

    # Pedal tones ground the stretto
    score.add_note("pedal", start=stretto_start - 1.0, duration=subject_dur + 3.5, partial=4, amp=0.24)
    score.add_note("pedal", start=stretto_start - 1.0, duration=subject_dur + 3.5, partial=6, amp=0.16)

    return score


def build_arpeggios_sketch() -> Score:
    """Sparse high-partial melody drifting downward — simple and tender."""
    score = Score(
        f0=55.0,
        master_effects=[_SOFT_REVERB],
    )

    score.add_voice(
        "drone",
        synth_defaults={
            "harmonic_rolloff": 0.60,
            "n_harmonics": 3,
            "attack": 2.5,
            "decay": 0.30,
            "sustain_level": 0.65,
            "release": 6.0,
        },
    )
    score.add_voice(
        "solo",
        synth_defaults={
            "harmonic_rolloff": 0.18,
            "n_harmonics": 4,
            "attack": 0.12,
            "decay": 0.40,
            "sustain_level": 0.30,
            "release": 3.5,
        },
    )

    # Very soft root drone beneath everything
    score.add_note("drone", start=0.0, duration=72.0, partial=1.0, amp=0.15, label="root")
    score.add_note("drone", start=8.0, duration=57.0, partial=2.0, amp=0.08, label="octave")

    # Sparse melody: descending arc with oscillation over ~70s
    # f0=55 → partial 14=770Hz, 12=660Hz, 10=550Hz, 8=440Hz, 6=330Hz
    melody_events: list[tuple[float, float, float]] = [
        (0.0,  14, 0.28),
        (5.5,  12, 0.30),
        (10.5, 14, 0.22),
        (15.5, 12, 0.28),
        (20.0, 10, 0.32),
        (24.5,  9, 0.30),
        (29.5, 12, 0.20),
        (34.0, 10, 0.26),
        (38.5,  9, 0.28),
        (43.0,  8, 0.32),
        (47.5, 10, 0.24),
        (52.0,  9, 0.28),
        (56.5,  8, 0.30),
        (61.0,  7, 0.34),
        (65.5,  8, 0.26),
        (69.5,  6, 0.36),
    ]
    for start, partial, amp in melody_events:
        score.add_note("solo", start=start, duration=5.0, partial=partial, amp=amp)

    return score


def build_variations_sketch() -> Score:
    """One JI theme heard through five transform lenses."""
    f0 = 110.0
    score = Score(
        f0=f0,
        master_effects=[_DELAY, _REVERB],
    )

    score.add_voice(
        "melody",
        synth_defaults={
            "harmonic_rolloff": 0.32,
            "attack": 0.06,
            "decay": 0.12,
            "sustain_level": 0.72,
            "release": 0.75,
        },
    )
    score.add_voice(
        "harmony",
        synth_defaults={
            "harmonic_rolloff": 0.40,
            "attack": 0.60,
            "decay": 0.20,
            "sustain_level": 0.70,
            "release": 2.5,
        },
    )

    # Asymmetric arch: 4-5-6-7-8-7-6 (rises to 8, descends one note short of origin)
    theme = Phrase.from_partials(
        [4, 5, 6, 7, 8, 7, 6],
        note_dur=1.1,
        step=0.9,
        amp=0.38,
        synth_defaults={
            "harmonic_rolloff": 0.32,
            "attack": 0.06,
            "decay": 0.12,
            "sustain_level": 0.72,
            "release": 0.75,
        },
    )
    theme_dur = theme.duration  # ~6.5s
    gap = 2.0
    cursor = 0.0

    # Theme as stated
    score.add_phrase("melody", theme, start=cursor)
    cursor += theme_dur + gap

    # Var 1: augmented (twice as slow)
    score.add_phrase("melody", theme, start=cursor, time_scale=2.0)
    cursor += theme_dur * 2.0 + gap

    # Var 2: retrograde
    score.add_phrase("melody", theme, start=cursor, reverse=True)
    cursor += theme_dur + gap

    # Var 3: transposed up through harmonics
    score.add_phrase("melody", theme, start=cursor, partial_shift=4.0)
    cursor += theme_dur + gap

    # Var 4: original theme with utonal harmony sounding below
    score.add_phrase("melody", theme, start=cursor)
    for freq in utonal(f0 * 8.0, [4, 5, 6, 7]):
        score.add_note("harmony", start=cursor, duration=theme_dur + 1.5, freq=freq, amp=0.16)
    cursor += theme_dur + gap

    # Var 5: diminution (twice as fast) in two registers at once
    score.add_phrase("melody", theme, start=cursor, time_scale=0.5)
    score.add_phrase("melody", theme, start=cursor, time_scale=0.5, partial_shift=4.0, amp_scale=0.65)

    return score


def build_spiral_sketch() -> Score:
    """Same melodic arch at four fundamentals rising by JI fifths (3/2)."""
    score = Score(
        f0=55.0,  # reference only; all melody uses freq= directly
        master_effects=[_DELAY, _REVERB],
    )

    score.add_voice(
        "melody",
        synth_defaults={
            "harmonic_rolloff": 0.30,
            "attack": 0.06,
            "decay": 0.14,
            "sustain_level": 0.70,
            "release": 0.85,
        },
    )
    score.add_voice(
        "drone",
        synth_defaults={
            "harmonic_rolloff": 0.48,
            "attack": 1.2,
            "decay": 0.25,
            "sustain_level": 0.80,
            "release": 3.5,
        },
    )

    # Arch contour as ratios (4:5:6:7:6:5:4 of the local fundamental)
    shape_ratios = [1.0, 1.25, 1.5, 1.75, 1.5, 1.25, 1.0]
    note_dur = 1.4
    step = 1.1
    section_gap = 5.0
    melody_dur = (len(shape_ratios) - 1) * step + note_dur  # ~8.0s
    section_dur = melody_dur + section_gap

    # Each section raises the fundamental by a JI fifth (3/2)
    fundamentals = [55.0]
    for _ in range(3):
        fundamentals.append(fundamentals[-1] * 3 / 2)
    # 55.0, 82.5, 123.75, 185.625 Hz

    for section_idx, f0_section in enumerate(fundamentals):
        section_start = section_idx * section_dur
        # Drone one octave above the local fundamental
        score.add_note(
            "drone",
            start=section_start,
            duration=melody_dur + 2.0,
            freq=f0_section * 2.0,
            amp=0.28,
        )
        # Melody at 4× the local fundamental (keeps things in the mid register)
        for note_idx, ratio in enumerate(shape_ratios):
            score.add_note(
                "melody",
                start=section_start + note_idx * step,
                duration=note_dur,
                freq=f0_section * 4.0 * ratio,
                amp=0.36,
            )

    return score


def build_interference_sketch() -> Score:
    """Two harmonic series 0.5 Hz apart, creating layered beating patterns."""
    f0_a = 110.0
    f0_b = 110.5  # 0.5 Hz apart → partial k beats at 0.5k Hz

    score = Score(
        f0=f0_a,
        master_effects=[_SOFT_REVERB],
    )

    score.add_voice(
        "series_a",
        synth_defaults={
            "harmonic_rolloff": 0.40,
            "n_harmonics": 3,
            "attack": 1.8,
            "decay": 0.25,
            "sustain_level": 0.84,
            "release": 5.0,
        },
    )
    score.add_voice(
        "series_b",
        synth_defaults={
            "harmonic_rolloff": 0.38,
            "n_harmonics": 3,
            "attack": 2.2,
            "decay": 0.25,
            "sustain_level": 0.80,
            "release": 5.0,
        },
    )
    score.add_voice(
        "solo",
        synth_defaults={
            "harmonic_rolloff": 0.22,
            "n_harmonics": 4,
            "attack": 0.10,
            "decay": 0.30,
            "sustain_level": 0.50,
            "release": 1.8,
        },
    )

    # Both series hold partials 2-8 for 40s; series_b enters slightly later
    held_dur = 40.0
    for k in range(2, 9):
        amp_a = max(0.04, 0.16 - k * 0.01)
        amp_b = max(0.03, 0.14 - k * 0.01)
        score.add_note("series_a", start=0.0, duration=held_dur, freq=f0_a * k, amp=amp_a)
        score.add_note("series_b", start=0.6, duration=held_dur - 1.0, freq=f0_b * k, amp=amp_b)

    # Solo voice picks out specific partials above the texture
    solo_events: list[tuple[float, float, float]] = [
        (6.0,  f0_a * 6, 0.26),
        (14.0, f0_a * 8, 0.22),
        (22.0, f0_a * 7, 0.28),
        (30.0, f0_a * 5, 0.30),
        (37.0, f0_a * 4, 0.32),
    ]
    for start, freq, amp in solo_events:
        score.add_note("solo", start=start, duration=4.0, freq=freq, amp=amp)

    return score


def build_arpeggios_cross_sketch() -> Score:
    """Two voices in contrary motion: one descends, one ascends, weaving JI chords."""
    score = Score(
        f0=55.0,
        master_effects=[_SOFT_REVERB],
    )

    shared_synth: dict = {
        "harmonic_rolloff": 0.18,
        "n_harmonics": 4,
        "attack": 0.10,
        "decay": 0.35,
        "sustain_level": 0.28,
        "release": 3.5,
    }
    score.add_voice(
        "drone",
        synth_defaults={
            "harmonic_rolloff": 0.60,
            "n_harmonics": 3,
            "attack": 2.5,
            "decay": 0.30,
            "sustain_level": 0.65,
            "release": 6.0,
        },
    )
    score.add_voice("voice_a", synth_defaults=shared_synth)
    score.add_voice("voice_b", synth_defaults={**shared_synth, "harmonic_rolloff": 0.22})

    score.add_note("drone", start=0.0, duration=44.0, partial=1.0, amp=0.12)
    score.add_note("drone", start=4.0, duration=38.0, partial=2.0, amp=0.06)

    note_dur = 4.5

    # Voice A descends from 14 to 6 over ~40s
    voice_a_events: list[tuple[float, float, float]] = [
        (0.0,  14, 0.26), (2.5,  12, 0.28), (5.0,  14, 0.22), (7.5,  11, 0.26),
        (10.0, 10, 0.28), (12.5,  9, 0.30), (15.0, 11, 0.22), (17.5, 10, 0.26),
        (20.0,  9, 0.26), (22.5,  8, 0.30), (25.0, 10, 0.22), (27.5,  9, 0.26),
        (30.0,  8, 0.28), (32.5,  7, 0.32), (35.0,  8, 0.24), (38.0,  6, 0.34),
    ]
    # Voice B ascends from 6 to 14, offset by 1.2s so they interleave
    voice_b_events: list[tuple[float, float, float]] = [
        (1.2,   6, 0.28), (3.7,   7, 0.26), (6.2,   6, 0.24), (8.7,   8, 0.28),
        (11.2,  7, 0.28), (13.7,  9, 0.26), (16.2,  8, 0.24), (18.7, 10, 0.26),
        (21.2,  9, 0.26), (23.7, 10, 0.26), (26.2,  9, 0.24), (28.7, 11, 0.26),
        (31.2, 10, 0.24), (33.7, 12, 0.28), (36.2, 11, 0.24), (39.2, 14, 0.30),
    ]

    for start, partial, amp in voice_a_events:
        score.add_note("voice_a", start=start, duration=note_dur, partial=partial, amp=amp)
    for start, partial, amp in voice_b_events:
        score.add_note("voice_b", start=start, duration=note_dur, partial=partial, amp=amp)

    return score


def build_spiral_arch_sketch() -> Score:
    """Spiral up 3 JI fifths then back down — arch shape inverts on the return."""
    score = Score(
        f0=55.0,
        master_effects=[_DELAY, _REVERB],
    )

    score.add_voice(
        "melody",
        synth_defaults={
            "harmonic_rolloff": 0.30,
            "attack": 0.06,
            "decay": 0.14,
            "sustain_level": 0.70,
            "release": 0.85,
        },
    )
    score.add_voice(
        "drone",
        synth_defaults={
            "harmonic_rolloff": 0.48,
            "attack": 1.0,
            "decay": 0.25,
            "sustain_level": 0.80,
            "release": 3.5,
        },
    )

    # Going up: arch peaks at center; going down: arch valleys at center
    ascending_shape = [1.0, 1.25, 1.5, 1.75, 1.5, 1.25, 1.0]
    descending_shape = [1.75, 1.5, 1.25, 1.0, 1.25, 1.5, 1.75]

    # Rhythmic pattern: quickens into the peak, broadens on arrival.
    # 7 notes → 6 inter-onset intervals (IOIs).
    # Ascending: short-short-LONG at the peak (index 3), then quick release
    ascending_ioi  = [0.90, 0.65, 0.65, 1.50, 0.65, 0.85]
    ascending_durs = [1.10, 0.80, 0.80, 2.00, 0.80, 1.00, 1.20]
    ascending_amps = [0.32, 0.28, 0.30, 0.42, 0.30, 0.28, 0.32]
    # Descending: mirror — broad at the start (the peak is now first), quickens away
    descending_ioi  = [0.85, 0.65, 1.50, 0.65, 0.65, 0.90]
    descending_durs = [1.20, 1.00, 0.80, 2.00, 0.80, 0.80, 1.10]
    descending_amps = [0.42, 0.32, 0.28, 0.30, 0.30, 0.28, 0.32]

    def ioi_to_onsets(ioi: list[float]) -> list[float]:
        onsets = [0.0]
        for interval in ioi:
            onsets.append(onsets[-1] + interval)
        return onsets

    ascending_onsets = ioi_to_onsets(ascending_ioi)
    descending_onsets = ioi_to_onsets(descending_ioi)

    section_gap = 2.0
    # Phrase duration = last onset + last note duration
    ascending_phrase_dur = ascending_onsets[-1] + ascending_durs[-1]    # ~6.4s
    descending_phrase_dur = descending_onsets[-1] + descending_durs[-1]  # ~6.3s

    # Up: 55 → 82.5 → 123.75 → 185.625, then back: → 123.75 → 82.5 → 55
    ascending_fundamentals = [55.0]
    for _ in range(3):
        ascending_fundamentals.append(ascending_fundamentals[-1] * 3 / 2)
    sections = ascending_fundamentals + ascending_fundamentals[-2::-1]
    shapes = [ascending_shape] * 4 + [descending_shape] * 3
    onsets_per_section = [ascending_onsets] * 4 + [descending_onsets] * 3
    durs_per_section = [ascending_durs] * 4 + [descending_durs] * 3
    amps_per_section = [ascending_amps] * 4 + [descending_amps] * 3
    phrase_durs = [ascending_phrase_dur] * 4 + [descending_phrase_dur] * 3

    cursor = 0.0
    for f0_section, shape, onsets, durs, amps, phrase_dur in zip(
        sections, shapes, onsets_per_section, durs_per_section, amps_per_section, phrase_durs
    ):
        score.add_note(
            "drone",
            start=cursor,
            duration=phrase_dur + 1.5,
            freq=f0_section * 2.0,
            amp=0.26,
        )
        for ratio, onset, dur, amp in zip(shape, onsets, durs, amps):
            score.add_note(
                "melody",
                start=cursor + onset,
                duration=dur,
                freq=f0_section * 4.0 * ratio,
                amp=amp,
            )
        cursor += phrase_dur + section_gap

    return score


def build_interference_v2_sketch() -> Score:
    """Beating texture that shifts gears: slow beating phase, then fast."""
    f0_a = 110.0
    f0_b_slow = 110.5   # 0.5 Hz apart → beats 0.5k Hz per partial k
    f0_b_fast = 113.0   # 3.0 Hz apart → beats 3.0k Hz per partial k

    score = Score(
        f0=f0_a,
        master_effects=[_SOFT_REVERB],
    )

    score.add_voice(
        "series_a",
        synth_defaults={
            "harmonic_rolloff": 0.40,
            "n_harmonics": 3,
            "attack": 1.8,
            "decay": 0.25,
            "sustain_level": 0.84,
            "release": 5.0,
        },
    )
    score.add_voice(
        "series_b",
        synth_defaults={
            "harmonic_rolloff": 0.38,
            "n_harmonics": 3,
            "attack": 2.0,
            "decay": 0.25,
            "sustain_level": 0.80,
            "release": 5.0,
        },
    )
    score.add_voice(
        "melody",
        synth_defaults={
            "harmonic_rolloff": 0.22,
            "n_harmonics": 4,
            "attack": 0.10,
            "decay": 0.30,
            "sustain_level": 0.50,
            "release": 1.8,
        },
    )

    # series_a: anchor, held throughout
    for k in range(2, 9):
        amp_a = max(0.04, 0.15 - k * 0.01)
        score.add_note("series_a", start=0.0, duration=55.0, freq=f0_a * k, amp=amp_a)

    # series_b phase 1: slow beating (0-35s, long release carries it further)
    for k in range(2, 9):
        amp_b = max(0.03, 0.12 - k * 0.01)
        score.add_note("series_b", start=0.5, duration=32.0, freq=f0_b_slow * k, amp=amp_b)

    # series_b phase 2: fast beating enters at t=25, overlaps with phase 1
    for k in range(2, 9):
        amp_b = max(0.03, 0.12 - k * 0.01)
        score.add_note("series_b", start=25.0, duration=30.0, freq=f0_b_fast * k, amp=amp_b)

    # Melody: active arc descending over the full texture
    melody_events: list[tuple[float, float, float]] = [
        (3.0,  f0_a * 7, 0.28),
        (7.0,  f0_a * 8, 0.24),
        (11.0, f0_a * 9, 0.22),
        (15.0, f0_a * 8, 0.26),
        (19.0, f0_a * 7, 0.28),
        (23.0, f0_a * 6, 0.30),
        (27.0, f0_a * 8, 0.22),
        (31.0, f0_a * 7, 0.26),
        (35.0, f0_a * 6, 0.28),
        (39.0, f0_a * 5, 0.30),
        (44.0, f0_a * 4, 0.34),
    ]
    for start, freq, amp in melody_events:
        score.add_note("melody", start=start, duration=3.5, freq=freq, amp=amp)

    return score


def build_interference_ji_sketch() -> Score:
    """Three JI-related drone series (root, fifth, harmonic seventh) entering in sequence."""
    f0_a = 110.0           # root — 2nd partial of 55
    f0_b = 110.0 * 3 / 2  # 165.0 — JI fifth, 3rd partial of 55
    f0_c = 110.0 * 7 / 4  # 192.5 — harmonic seventh, 7th/2 partial of 55

    score = Score(
        f0=f0_a,
        master_effects=[_SOFT_REVERB],
    )

    score.add_voice(
        "series_a",
        synth_defaults={
            "harmonic_rolloff": 0.45,
            "n_harmonics": 3,
            "attack": 2.0,
            "decay": 0.20,
            "sustain_level": 0.85,
            "release": 5.0,
        },
    )
    score.add_voice(
        "series_b",
        synth_defaults={
            "harmonic_rolloff": 0.40,
            "n_harmonics": 3,
            "attack": 2.5,
            "decay": 0.20,
            "sustain_level": 0.82,
            "release": 5.0,
        },
    )
    score.add_voice(
        "series_c",
        synth_defaults={
            "harmonic_rolloff": 0.35,
            "n_harmonics": 3,
            "attack": 3.0,
            "decay": 0.20,
            "sustain_level": 0.78,
            "release": 5.0,
        },
    )
    score.add_voice(
        "melody",
        synth_defaults={
            "harmonic_rolloff": 0.22,
            "n_harmonics": 4,
            "attack": 0.10,
            "decay": 0.30,
            "sustain_level": 0.50,
            "release": 1.8,
        },
    )

    held_dur = 55.0

    # series_a: root + harmonics, enters first
    for k in range(1, 7):
        amp = max(0.04, 0.18 - k * 0.02)
        score.add_note("series_a", start=0.0, duration=held_dur, freq=f0_a * k, amp=amp)

    # series_b: JI fifth + harmonics, enters at t=5
    for k in range(1, 5):
        amp = max(0.04, 0.16 - k * 0.02)
        score.add_note("series_b", start=5.0, duration=held_dur - 5.0, freq=f0_b * k, amp=amp)

    # series_c: harmonic seventh + harmonics, enters at t=18
    for k in range(1, 4):
        amp = max(0.03, 0.12 - k * 0.02)
        score.add_note("series_c", start=18.0, duration=held_dur - 18.0, freq=f0_c * k, amp=amp)

    # Melody: explores the combined harmonic space, mostly in the 330-880 Hz range
    melody_events: list[tuple[float, float, float]] = [
        (2.0,  f0_a * 6, 0.28),  # 660 Hz
        (6.0,  f0_a * 8, 0.24),  # 880 Hz
        (10.0, f0_a * 7, 0.28),  # 770 Hz — septimal partial
        (14.0, f0_b * 4, 0.26),  # 660 Hz via fifth (=6×f0_a, reinforces)
        (18.0, f0_c * 2, 0.26),  # 385 Hz — 7th partial of f0_a
        (22.0, f0_a * 8, 0.22),
        (26.0, f0_b * 3, 0.26),  # 495 Hz
        (30.0, f0_a * 6, 0.28),  # 660 Hz
        (34.0, f0_c * 3, 0.24),  # 577.5 Hz — septimal colour
        (38.0, f0_a * 5, 0.30),  # 550 Hz
        (42.0, f0_b * 2, 0.28),  # 330 Hz
        (47.0, f0_a * 4, 0.34),  # 440 Hz — landing on the octave
    ]
    for start, freq, amp in melody_events:
        score.add_note("melody", start=start, duration=4.0, freq=freq, amp=amp)

    return score


def build_articulation_study_sketch() -> Score:
    """Short study contrasting clipped JI rhythm with ratio glides."""
    score = Score(
        f0=110.0,
        master_effects=[_SOFT_REVERB],
    )
    score.add_voice(
        "drone",
        synth_defaults={
            "engine": "additive",
            "preset": "drone",
            "attack": 0.6,
            "release": 2.0,
        },
    )
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "fm",
            "preset": "glass_lead",
            "attack": 0.03,
            "release": 0.35,
        },
    )

    score.add_note("drone", start=0.0, duration=16.0, partial=1.0, amp=0.18)
    score.add_note("drone", start=4.0, duration=10.0, partial=3 / 2, amp=0.12)

    phrase = line(
        tones=[6.0, 7.0, 8.0, 9.0, 8.0, 7.0],
        rhythm=RhythmCell(spans=(0.75, 0.75, 1.1, 0.65, 0.9, 1.4)),
        amp=0.30,
        articulation=ArticulationSpec(
            gate=(0.55, 0.55, 0.92, 0.5, 0.72, 1.08),
            accent_pattern=(1.0, 0.9, 1.18, 0.92, 1.05, 1.2),
            tail_breath=0.12,
        ),
        pitch_motion=(
            None,
            PitchMotionSpec.linear_bend(target_partial=8.0),
            None,
            PitchMotionSpec.ratio_glide(start_ratio=1.0, end_ratio=7 / 6),
            None,
            PitchMotionSpec.vibrato(depth_ratio=0.015, rate_hz=6.2),
        ),
    )
    score.add_phrase("lead", phrase, start=1.5)
    score.add_phrase("lead", phrase, start=8.0, partial_shift=2.0, amp_scale=0.92)

    return score


PIECES: dict[str, PieceDefinition] = {
    "sketch_passacaglia": PieceDefinition(
        name="sketch_passacaglia",
        output_name="07_sketch_passacaglia.wav",
        build_score=build_passacaglia_sketch,
    ),
    "sketch_invention": PieceDefinition(
        name="sketch_invention",
        output_name="08_sketch_invention.wav",
        build_score=build_invention_sketch,
    ),
    "sketch_arpeggios": PieceDefinition(
        name="sketch_arpeggios",
        output_name="09_sketch_arpeggios.wav",
        build_score=build_arpeggios_sketch,
    ),
    "sketch_variations": PieceDefinition(
        name="sketch_variations",
        output_name="10_sketch_variations.wav",
        build_score=build_variations_sketch,
    ),
    "sketch_spiral": PieceDefinition(
        name="sketch_spiral",
        output_name="11_sketch_spiral.wav",
        build_score=build_spiral_sketch,
    ),
    "sketch_interference": PieceDefinition(
        name="sketch_interference",
        output_name="12_sketch_interference.wav",
        build_score=build_interference_sketch,
    ),
    "sketch_arpeggios_cross": PieceDefinition(
        name="sketch_arpeggios_cross",
        output_name="13_sketch_arpeggios_cross.wav",
        build_score=build_arpeggios_cross_sketch,
    ),
    "sketch_spiral_arch": PieceDefinition(
        name="sketch_spiral_arch",
        output_name="14_sketch_spiral_arch.wav",
        build_score=build_spiral_arch_sketch,
    ),
    "sketch_interference_v2": PieceDefinition(
        name="sketch_interference_v2",
        output_name="15_sketch_interference_v2.wav",
        build_score=build_interference_v2_sketch,
    ),
    "sketch_interference_ji": PieceDefinition(
        name="sketch_interference_ji",
        output_name="16_sketch_interference_ji.wav",
        build_score=build_interference_ji_sketch,
    ),
    "sketch_articulation_study": PieceDefinition(
        name="sketch_articulation_study",
        output_name="20_sketch_articulation_study.wav",
        build_score=build_articulation_study_sketch,
    ),
}
