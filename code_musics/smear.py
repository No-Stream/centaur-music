"""Loveless-inspired compositional tools for pitch smearing, textural thickening, and orchestration."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

from code_musics.automation import AutomationSegment, AutomationSpec, AutomationTarget
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import NoteEvent, Phrase, Score
from code_musics.tuning import cents_to_ratio

_PITCH_RATIO_TARGET = AutomationTarget(kind="pitch_ratio", name="pitch_ratio")


def _note_sort_key(event: NoteEvent) -> float:
    """Return a sortable pitch value for a NoteEvent."""
    if event.partial is not None:
        return event.partial
    if event.freq is not None:
        return event.freq
    raise ValueError("NoteEvent must have partial or freq")


@dataclass(frozen=True)
class ThickenedCopy:
    """One micro-detuned, panned copy produced by thicken()."""

    phrase: Phrase
    pan: float
    amp_offset_db: float


def strum(
    phrase: Phrase,
    spread_ms: float = 40.0,
    direction: str = "down",
    seed: int = 0,
) -> Phrase:
    """Stagger simultaneous chord notes across a time spread.

    Takes a Phrase containing simultaneous notes (a chord) and returns a new
    Phrase where note start times are staggered to simulate a strum.

    Args:
        phrase: Input phrase (typically a chord with all notes at start=0).
        spread_ms: Total time spread in milliseconds across all notes.
        direction: "down" (low-to-high), "up" (high-to-low), "out" (center
            outward), or "random" (seeded deterministic).
        seed: Random seed for "random" direction.
    """
    if spread_ms < 0:
        raise ValueError("spread_ms must be non-negative")
    if direction not in {"down", "up", "out", "random"}:
        raise ValueError("direction must be 'down', 'up', 'out', or 'random'")

    events = list(phrase.events)
    if len(events) <= 1:
        return phrase

    spread_seconds = spread_ms / 1000.0

    sorted_indices = sorted(range(len(events)), key=lambda i: _note_sort_key(events[i]))

    if direction == "down":
        order = sorted_indices
    elif direction == "up":
        order = list(reversed(sorted_indices))
    elif direction == "out":
        order = _center_outward_order(sorted_indices)
    else:
        rng = random.Random(seed)
        order = list(sorted_indices)
        rng.shuffle(order)

    step = spread_seconds / max(len(events) - 1, 1)
    offsets = {idx: rank * step for rank, idx in enumerate(order)}

    strummed_events = []
    for i, event in enumerate(events):
        offset = offsets[i]
        strummed_events.append(
            replace(
                event,
                start=event.start + offset,
                amp_db=None,
                synth=dict(event.synth) if event.synth is not None else None,
            )
        )

    return Phrase(events=tuple(strummed_events))


def _center_outward_order(sorted_indices: list[int]) -> list[int]:
    """Reorder indices so the center comes first, expanding outward."""
    n = len(sorted_indices)
    center = n // 2
    order = [sorted_indices[center]]
    left = center - 1
    right = center + 1
    while len(order) < n:
        if left >= 0:
            order.append(sorted_indices[left])
            left -= 1
        if right < n and len(order) < n:
            order.append(sorted_indices[right])
            right += 1
    return order


def thicken(
    phrase: Phrase,
    n: int = 5,
    detune_cents: float = 8.0,
    spread_ms: float = 20.0,
    stereo_width: float = 0.7,
    amp_taper_db: float = -2.0,
    seed: int = 0,
) -> list[ThickenedCopy]:
    """Create micro-detuned, time-staggered, pan-spread copies of a phrase.

    Returns a list of ThickenedCopy, each containing a detuned phrase, a pan
    position, and an amplitude offset in dB. The caller is responsible for
    placing these on separate voices (or the same voice with pan overrides).

    Args:
        phrase: Input phrase to thicken.
        n: Number of copies to produce.
        detune_cents: Total detune spread in cents (distributed across +/- half).
        spread_ms: Total time stagger in ms (each copy offset within +/- half).
        stereo_width: Pan spread (copies distributed across +/- stereo_width).
        amp_taper_db: Amplitude reduction applied to the outermost copies.
        seed: Deterministic seed for time offsets.
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    rng = random.Random(seed)
    spread_seconds = spread_ms / 1000.0

    copies: list[ThickenedCopy] = []
    for copy_index in range(n):
        fraction = 0.5 if n == 1 else copy_index / (n - 1)

        detune_offset_cents = -detune_cents / 2.0 + fraction * detune_cents
        freq_scale = cents_to_ratio(detune_offset_cents)

        time_offset = rng.uniform(-spread_seconds / 2.0, spread_seconds / 2.0)

        pan = 0.0 if n == 1 else -stereo_width + fraction * 2.0 * stereo_width

        distance_from_center = abs(fraction - 0.5) * 2.0
        amp_offset = amp_taper_db * distance_from_center

        detuned_events = []
        for event in phrase.events:
            if event.freq is not None:
                new_event = replace(
                    event,
                    start=max(0.0, event.start + time_offset),
                    freq=event.freq * freq_scale,
                    amp_db=None,
                    synth=dict(event.synth) if event.synth is not None else None,
                )
            else:
                note_synth = dict(event.synth) if event.synth is not None else {}
                note_synth["freq_scale"] = freq_scale
                new_event = replace(
                    event,
                    start=max(0.0, event.start + time_offset),
                    amp_db=None,
                    synth=note_synth,
                )
            detuned_events.append(new_event)

        copies.append(
            ThickenedCopy(
                phrase=Phrase(events=tuple(detuned_events)),
                pan=pan,
                amp_offset_db=amp_offset,
            )
        )

    return copies


def pitch_wobble(
    duration: float,
    rate_hz: float = 0.15,
    depth_cents: float = 12.0,
    style: str = "smooth",
    start_time: float = 0.0,
    seed: int = 0,
    depth_curve: list[tuple[float, float]] | None = None,
    segment_interval: float = 0.05,
) -> AutomationSpec:
    """Generate a continuous pitch modulation automation lane.

    Returns an AutomationSpec with target "pitch_ratio" and mode "multiply",
    producing gentle pitch wobble simulating tremolo bar or tape flutter.

    Args:
        duration: Length of the wobble in seconds.
        rate_hz: Modulation rate (LFO frequency or noise spectral center).
        depth_cents: Modulation depth in cents (+/- half for LFO, RMS for smooth/drunk).
        style: "lfo" (sine), "smooth" (filtered Brownian), or "drunk" (random walk
            with momentum).
        start_time: Absolute start time for the automation segments.
        seed: Deterministic random seed.
        depth_curve: Optional list of (time, depth_cents) pairs for time-varying depth.
            Times are relative to start_time.
        segment_interval: Approximate time between automation segments in seconds.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")
    if depth_cents < 0:
        raise ValueError("depth_cents must be non-negative")
    if style not in {"lfo", "smooth", "drunk"}:
        raise ValueError("style must be 'lfo', 'smooth', or 'drunk'")

    n_points = max(int(duration / segment_interval) + 1, 3)
    times = np.linspace(0.0, duration, n_points)

    if style == "lfo":
        values_cents = (depth_cents / 2.0) * np.sin(2.0 * np.pi * rate_hz * times)
    elif style == "smooth":
        values_cents = _brownian_filtered(times, rate_hz, depth_cents, seed)
    else:
        values_cents = _drunk_walk(times, rate_hz, depth_cents, seed)

    if depth_curve is not None:
        depth_envelope = _interpolate_depth_curve(times, depth_curve, depth_cents)
        values_cents = values_cents * depth_envelope

    ratios = np.array([cents_to_ratio(c) for c in values_cents])

    segments = _build_linear_segments(times + start_time, ratios)

    return AutomationSpec(
        target=_PITCH_RATIO_TARGET,
        segments=tuple(segments),
        default_value=1.0,
        mode="multiply",
    )


def _brownian_filtered(
    times: np.ndarray, rate_hz: float, depth_cents: float, seed: int
) -> np.ndarray:
    """Generate filtered Brownian motion with approximate spectral center at rate_hz."""
    rng = np.random.default_rng(seed)
    n = len(times)
    dt = float(times[1] - times[0]) if n > 1 else 0.05
    sample_rate = 1.0 / dt

    raw_noise = rng.standard_normal(n)
    walk = np.cumsum(raw_noise)

    nyquist = sample_rate / 2.0
    cutoff = min(rate_hz * 2.0, nyquist * 0.9)
    if cutoff > 0 and nyquist > 0:
        sos = butter(2, cutoff / nyquist, btype="low", output="sos")
        walk = sosfiltfilt(sos, walk)

    walk = walk - np.mean(walk)
    rms = float(np.sqrt(np.mean(walk**2)))
    if rms > 0:
        walk = walk * (depth_cents / rms)

    return walk


def _drunk_walk(
    times: np.ndarray, rate_hz: float, depth_cents: float, seed: int
) -> np.ndarray:
    """Random walk with momentum -- more unpredictable than smooth but still continuous."""
    rng = np.random.default_rng(seed)
    n = len(times)
    dt = float(times[1] - times[0]) if n > 1 else 0.05

    damping = rate_hz * 2.0 * np.pi * dt
    noise_scale = depth_cents * np.sqrt(dt) * 0.5

    position = 0.0
    velocity = 0.0
    values = np.zeros(n)

    for i in range(n):
        velocity += rng.standard_normal() * noise_scale
        velocity *= max(0.0, 1.0 - damping)
        position += velocity * dt
        position *= max(0.0, 1.0 - damping * 0.1)
        values[i] = position

    values = values - np.mean(values)
    rms = float(np.sqrt(np.mean(values**2)))
    if rms > 0:
        values = values * (depth_cents / rms)

    return values


def _interpolate_depth_curve(
    times: np.ndarray, depth_curve: list[tuple[float, float]], base_depth_cents: float
) -> np.ndarray:
    """Linearly interpolate a depth envelope from (time, depth_cents) pairs."""
    curve_times = np.array([p[0] for p in depth_curve])
    curve_depths = np.array([p[1] for p in depth_curve])
    interpolated = np.interp(times, curve_times, curve_depths)
    if base_depth_cents > 0:
        return interpolated / base_depth_cents
    return np.ones_like(times)


def _build_linear_segments(
    times: np.ndarray, values: np.ndarray
) -> list[AutomationSegment]:
    """Build consecutive linear automation segments from time/value arrays."""
    segments: list[AutomationSegment] = []
    for i in range(len(times) - 1):
        t_start = float(times[i])
        t_end = float(times[i + 1])
        if t_end <= t_start:
            continue
        segments.append(
            AutomationSegment(
                start=t_start,
                end=t_end,
                shape="linear",
                start_value=float(values[i]),
                end_value=float(values[i + 1]),
            )
        )
    return segments


def smear_progression(
    chords: Sequence[Sequence[float]],
    durations: Sequence[float],
    overlap: float = 0.5,
    voice_behavior: Sequence[str] | None = None,
) -> list[Phrase]:
    """Build gliding voice phrases from a chord progression.

    Each chord is a list of partial ratios. Returns one Phrase per voice index
    (i.e., per chord tone position across all chords). Notes use `partial` values
    so they are relative to Score.f0_hz.

    The ratio_glide spans the full note duration. The overlap parameter controls
    how much notes extend into the next chord's time, which determines the glide
    duration.

    Args:
        chords: List of ratio lists, e.g. [[1, 5/4, 3/2], [1, 6/5, 3/2]].
        durations: Duration in seconds for each chord.
        overlap: Fraction of the next chord's duration during which the previous
            chord still sounds (0=gap, 0.5=halfway overlap, 1=full legato).
        voice_behavior: Optional per-voice-index behavior list: "glide" (default)
            or "reattack".
    """
    if not chords:
        raise ValueError("chords must not be empty")
    if len(chords) != len(durations):
        raise ValueError("chords and durations must have the same length")
    if any(d <= 0 for d in durations):
        raise ValueError("durations must be positive")

    voice_count = len(chords[0])
    if any(len(chord) != voice_count for chord in chords):
        raise ValueError("all chords must have the same number of voices")

    if voice_behavior is None:
        behaviors = ["glide"] * voice_count
    else:
        behaviors = list(voice_behavior)
        if len(behaviors) != voice_count:
            raise ValueError("voice_behavior length must match chord voice count")

    voice_phrases: list[list[NoteEvent]] = [[] for _ in range(voice_count)]

    cursor = 0.0
    for chord_index, (chord, dur) in enumerate(zip(chords, durations, strict=True)):
        is_last = chord_index == len(chords) - 1
        next_chord = chords[chord_index + 1] if not is_last else None

        for voice_index in range(voice_count):
            partial = float(chord[voice_index])
            behavior = behaviors[voice_index]

            if is_last or behavior == "reattack" or next_chord is None:
                voice_phrases[voice_index].append(
                    NoteEvent(
                        start=cursor,
                        duration=dur,
                        partial=partial,
                    )
                )
            else:
                next_partial = float(next_chord[voice_index])
                overlap_time = overlap * durations[chord_index + 1]
                note_duration = dur + overlap_time

                glide_ratio = next_partial / partial if partial > 0 else 1.0
                motion = PitchMotionSpec.ratio_glide(
                    start_ratio=1.0,
                    end_ratio=glide_ratio,
                )

                voice_phrases[voice_index].append(
                    NoteEvent(
                        start=cursor,
                        duration=note_duration,
                        partial=partial,
                        pitch_motion=motion,
                    )
                )

        cursor += dur

    return [Phrase(events=tuple(events)) for events in voice_phrases]


def bloom(
    score: Score,
    voice_specs: Sequence[dict[str, Any]],
    center_time: float,
    grow_dur: float = 4.0,
    peak_dur: float = 8.0,
    fade_dur: float = 4.0,
) -> Score:
    """Orchestration helper for gradual layer introduction and dissolution.

    Staggers voice entries across grow_dur, sustains all voices during peak_dur,
    then staggers exits across fade_dur. Each voice gets amplitude automation
    for smooth fades.

    Args:
        score: The Score to add voices to.
        voice_specs: List of dicts with keys: "name" (str), "synth_defaults" (dict),
            "phrase" (Phrase), and optionally "pan" (float) plus other Voice kwargs.
        center_time: The midpoint of the peak section.
        grow_dur: Duration over which voices stagger their entries.
        peak_dur: Duration during which all voices sound.
        fade_dur: Duration over which voices stagger their exits.
    """
    if not voice_specs:
        raise ValueError("voice_specs must not be empty")
    if grow_dur < 0:
        raise ValueError("grow_dur must be non-negative")
    if peak_dur < 0:
        raise ValueError("peak_dur must be non-negative")
    if fade_dur < 0:
        raise ValueError("fade_dur must be non-negative")

    n_voices = len(voice_specs)
    peak_start = center_time - peak_dur / 2.0
    grow_start = peak_start - grow_dur
    fade_end = center_time + peak_dur / 2.0 + fade_dur

    fade_in_dur = min(2.0, grow_dur / max(n_voices, 1))
    fade_out_dur = min(2.0, fade_dur / max(n_voices, 1))

    for voice_index, spec in enumerate(voice_specs):
        name = spec["name"]
        synth_defaults = spec.get("synth_defaults", {})
        phrase = spec["phrase"]
        pan = spec.get("pan", 0.0)

        entry_fraction = 0.0 if n_voices == 1 else voice_index / (n_voices - 1)

        entry_time = grow_start + entry_fraction * grow_dur
        exit_time = fade_end - (1.0 - entry_fraction) * fade_dur

        voice_duration = exit_time - entry_time
        if voice_duration <= 0:
            continue

        extra_kwargs: dict[str, Any] = {}
        for key in spec:
            if key not in {"name", "synth_defaults", "phrase", "pan"}:
                extra_kwargs[key] = spec[key]

        score.add_voice(
            name,
            synth_defaults=synth_defaults,
            pan=pan,
            **extra_kwargs,
        )

        phrase_duration = phrase.duration
        if phrase_duration <= 0:
            continue

        repeats_needed = int(np.ceil(voice_duration / phrase_duration))
        cursor = entry_time
        for _ in range(repeats_needed):
            remaining = exit_time - cursor
            if remaining <= 0:
                break

            for event in phrase.events:
                note_start = cursor + event.start
                note_dur = min(event.duration, remaining - event.start)
                if note_dur <= 0 or note_start >= exit_time:
                    continue

                amp_envelope = _bloom_amp_at_time(
                    note_start,
                    entry_time,
                    entry_time + fade_in_dur,
                    exit_time - fade_out_dur,
                    exit_time,
                )
                if amp_envelope < 1e-6:
                    continue

                score.add_note(
                    name,
                    start=note_start,
                    duration=note_dur,
                    partial=event.partial,
                    freq=event.freq,
                    amp=float(event.amp or 1.0) * amp_envelope,
                    velocity=event.velocity,
                    synth=dict(event.synth) if event.synth is not None else None,
                )

            cursor += phrase_duration

    return score


def _bloom_amp_at_time(
    t: float,
    fade_in_start: float,
    fade_in_end: float,
    fade_out_start: float,
    fade_out_end: float,
) -> float:
    """Compute amplitude envelope value for a bloom voice at time t."""
    if t <= fade_in_start:
        return 0.0
    if t >= fade_out_end:
        return 0.0

    amp = 1.0
    if t < fade_in_end and fade_in_end > fade_in_start:
        amp *= (t - fade_in_start) / (fade_in_end - fade_in_start)
    if t > fade_out_start and fade_out_end > fade_out_start:
        amp *= (fade_out_end - t) / (fade_out_end - fade_out_start)

    return max(0.0, min(1.0, amp))
