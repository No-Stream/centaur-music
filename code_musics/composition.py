"""Higher-level composition helpers that compile to score events."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import cycle, islice
from typing import TYPE_CHECKING, Any

from code_musics import synth
from code_musics.automation import (
    AutomationMode,
    AutomationSegment,
    AutomationSpec,
    AutomationTarget,
)
from code_musics.meter import (
    B,
    BeatSpan,
    BeatValue,
    DurationLike,
    MeasurePosition,
    Timeline,
    TimePointLike,
)
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import BeatTiming, NoteEvent, Phrase

if TYPE_CHECKING:
    from code_musics.score import Score

__all__ = [
    "ArticulationSpec",
    "ContextSection",
    "ContextSectionSpec",
    "HarmonicContext",
    "MeteredSectionSpec",
    "PitchMotionSpec",
    "RhythmCell",
    "build_context_sections",
    "bar_automation",
    "canon",
    "concat",
    "echo",
    "grid_canon",
    "grid_line",
    "grid_ratio_line",
    "grid_sequence",
    "legato",
    "line",
    "metered_sections",
    "overlay",
    "place_ratio_chord",
    "place_ratio_line",
    "progression",
    "recontextualize_phrase",
    "ratio_line",
    "resolve_ratios",
    "sequence",
    "staccato",
    "voiced_ratio_chord",
    "with_accent_pattern",
    "with_gate",
    "with_synth_ramp",
    "with_tail_breath",
]


@dataclass(frozen=True)
class RhythmCell:
    """Reusable onset spans with optional gate values."""

    spans: tuple[float, ...]
    gates: float | tuple[float, ...] = 1.0

    def __post_init__(self) -> None:
        if not self.spans:
            raise ValueError("spans must not be empty")
        if any(span <= 0 for span in self.spans):
            raise ValueError("spans must be positive")
        _expand_positive_values(self.gates, len(self.spans), "gates")


@dataclass(frozen=True)
class ArticulationSpec:
    """Phrase-level articulation controls."""

    gate: float | tuple[float, ...] = 1.0
    attack_scale: float = 1.0
    release_scale: float = 1.0
    accent_pattern: float | tuple[float, ...] = 1.0
    tail_breath: float = 0.0

    def __post_init__(self) -> None:
        _validate_positive_scalar_or_sequence(self.gate, "gate")
        if self.attack_scale <= 0:
            raise ValueError("attack_scale must be positive")
        if self.release_scale <= 0:
            raise ValueError("release_scale must be positive")
        _validate_non_negative_scalar_or_sequence(self.accent_pattern, "accent_pattern")
        if self.tail_breath < 0:
            raise ValueError("tail_breath must be non-negative")


@dataclass(frozen=True)
class HarmonicContext:
    """A local tuning frame for resolving ratio material into concrete frequencies."""

    tonic: float
    name: str | None = None

    def __post_init__(self) -> None:
        if self.tonic <= 0:
            raise ValueError("tonic must be positive")

    def resolve_ratio(self, ratio: float) -> float:
        """Resolve a ratio against the local tonic."""
        ratio_value = float(ratio)
        if ratio_value <= 0:
            raise ValueError("ratio must be positive")
        return self.tonic * ratio_value

    def drifted(self, *, by_ratio: float, name: str | None = None) -> HarmonicContext:
        """Return a new context whose tonic is multiplied by the given ratio."""
        ratio_value = float(by_ratio)
        if ratio_value <= 0:
            raise ValueError("by_ratio must be positive")
        return HarmonicContext(
            tonic=self.tonic * ratio_value,
            name=self.name if name is None else name,
        )


@dataclass(frozen=True)
class ContextSectionSpec:
    """Specification for a time window resolved from a base tonic."""

    duration: float
    tonic_ratio: float = 1.0
    name: str | None = None

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.tonic_ratio <= 0:
            raise ValueError("tonic_ratio must be positive")


@dataclass(frozen=True)
class MeteredSectionSpec:
    """Specification for a harmonic section measured in bars."""

    bars: float
    tonic_ratio: float = 1.0
    name: str | None = None

    def __post_init__(self) -> None:
        if self.bars <= 0:
            raise ValueError("bars must be positive")
        if self.tonic_ratio <= 0:
            raise ValueError("tonic_ratio must be positive")


@dataclass(frozen=True)
class ContextSection:
    """A placed harmonic context over an absolute time span."""

    start: float
    duration: float
    context: HarmonicContext

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.duration <= 0:
            raise ValueError("duration must be positive")

    @property
    def end(self) -> float:
        """Return the section endpoint."""
        return self.start + self.duration


def line(
    tones: Sequence[float],
    rhythm: RhythmCell | Sequence[float],
    *,
    pitch_kind: str = "partial",
    amp: float | None = None,
    amp_db: float | None = None,
    synth_defaults: dict[str, Any] | None = None,
    articulation: ArticulationSpec | None = None,
    motions: PitchMotionSpec | Sequence[PitchMotionSpec | None] | None = None,
    pitch_motion: PitchMotionSpec | Sequence[PitchMotionSpec | None] | None = None,
    labels: Sequence[str] | None = None,
) -> Phrase:
    """Build a phrase from tones, onset spans, articulation, and motion."""
    tones = tuple(float(tone) for tone in tones)
    if not tones:
        raise ValueError("tones must not be empty")
    if pitch_kind not in {"partial", "freq"}:
        raise ValueError("pitch_kind must be 'partial' or 'freq'")
    if amp is not None and amp_db is not None:
        raise ValueError("provide amp or amp_db, not both")

    if amp_db is not None:
        base_amp = synth.db_to_amp(amp_db)
    elif amp is None:
        base_amp = 1.0
    else:
        if amp <= 0:
            raise ValueError("amp must be positive")
        base_amp = amp

    rhythm_cell = (
        rhythm if isinstance(rhythm, RhythmCell) else RhythmCell(spans=tuple(rhythm))
    )
    rhythm_spans, rhythm_gates = _resolve_rhythm_values(rhythm_cell, len(tones))
    note_labels = tuple(labels) if labels is not None else (None,) * len(tones)
    if len(note_labels) != len(tones):
        raise ValueError("labels must have the same length as tones")

    motion_input = _resolve_motion_input(motions, pitch_motion)
    note_motions = _expand_motions(motion_input, len(tones))

    articulation = articulation or ArticulationSpec()
    articulation_gates = _expand_positive_values(articulation.gate, len(tones), "gate")
    accents = _expand_non_negative_values(
        articulation.accent_pattern, len(tones), "accent_pattern"
    )

    events: list[NoteEvent] = []
    cursor = 0.0
    for index, (
        tone,
        span,
        rhythm_gate,
        articulation_gate,
        accent,
        motion,
        label,
    ) in enumerate(
        zip(
            tones,
            rhythm_spans,
            rhythm_gates,
            articulation_gates,
            accents,
            note_motions,
            note_labels,
            strict=True,
        )
    ):
        duration = span * rhythm_gate * articulation_gate
        if index == len(tones) - 1 and articulation.tail_breath:
            duration -= articulation.tail_breath
        if duration <= 0:
            raise ValueError("articulation produced a non-positive duration")

        note_synth = _build_note_synth(
            synth_defaults=synth_defaults,
            attack_scale=articulation.attack_scale,
            release_scale=articulation.release_scale,
        )
        note_kwargs: dict[str, Any] = {
            "start": cursor,
            "duration": duration,
            "amp": base_amp * accent,
            "synth": dict(note_synth) if note_synth is not None else None,
            "pitch_motion": motion,
            "label": label,
        }
        if pitch_kind == "partial":
            note_kwargs["partial"] = tone
        else:
            note_kwargs["freq"] = tone

        events.append(NoteEvent(**note_kwargs))
        cursor += span

    return Phrase(events=tuple(events))


def grid_line(
    tones: Sequence[float],
    durations: Sequence[DurationLike],
    *,
    timeline: Timeline,
    pitch_kind: str = "partial",
    amp: float | None = None,
    amp_db: float | None = None,
    synth_defaults: dict[str, Any] | None = None,
    articulation: ArticulationSpec | None = None,
    motions: PitchMotionSpec | Sequence[PitchMotionSpec | None] | None = None,
    pitch_motion: PitchMotionSpec | Sequence[PitchMotionSpec | None] | None = None,
    labels: Sequence[str] | None = None,
) -> Phrase:
    """Build a phrase from beat-relative durations on a musical timeline."""
    beat_durations = _resolve_duration_beats_sequence(durations)
    phrase = line(
        tones=tones,
        rhythm=_resolve_grid_second_spans(beat_durations, timeline=timeline),
        pitch_kind=pitch_kind,
        amp=amp,
        amp_db=amp_db,
        synth_defaults=synth_defaults,
        articulation=articulation,
        motions=motions,
        pitch_motion=pitch_motion,
        labels=labels,
    )
    return Phrase(
        events=phrase.events,
        beat_timings=_build_beat_timings(beat_durations),
    )


def build_context_sections(
    *,
    base_tonic: float,
    specs: Sequence[ContextSectionSpec],
    start: float = 0.0,
) -> tuple[ContextSection, ...]:
    """Build absolute-time context windows from a base tonic and section specs."""
    if base_tonic <= 0:
        raise ValueError("base_tonic must be positive")
    if start < 0:
        raise ValueError("start must be non-negative")
    if not specs:
        raise ValueError("specs must not be empty")

    sections: list[ContextSection] = []
    cursor = start
    for spec in specs:
        section_context = HarmonicContext(
            tonic=base_tonic * spec.tonic_ratio,
            name=spec.name,
        )
        sections.append(
            ContextSection(
                start=cursor,
                duration=spec.duration,
                context=section_context,
            )
        )
        cursor += spec.duration
    return tuple(sections)


def metered_sections(
    *,
    timeline: Timeline,
    base_tonic: float,
    specs: Sequence[MeteredSectionSpec],
    start: TimePointLike | None = None,
) -> tuple[ContextSection, ...]:
    """Build harmonic sections measured in bars on a musical timeline."""
    if not specs:
        raise ValueError("specs must not be empty")

    return build_context_sections(
        base_tonic=base_tonic,
        specs=tuple(
            ContextSectionSpec(
                duration=timeline.measures(spec.bars),
                tonic_ratio=spec.tonic_ratio,
                name=spec.name,
            )
            for spec in specs
        ),
        start=_resolve_time_point(
            MeasurePosition(1.0) if start is None else start,
            timeline=timeline,
        ),
    )


def resolve_ratios(
    context: HarmonicContext,
    ratios: Sequence[float],
) -> list[float]:
    """Resolve a sequence of ratios against a harmonic context."""
    resolved = [context.resolve_ratio(ratio) for ratio in ratios]
    if not resolved:
        raise ValueError("ratios must not be empty")
    return resolved


def ratio_line(
    tones: Sequence[float],
    rhythm: RhythmCell | Sequence[float],
    *,
    context: HarmonicContext,
    amp: float | None = None,
    amp_db: float | None = None,
    synth_defaults: dict[str, Any] | None = None,
    articulation: ArticulationSpec | None = None,
    motions: PitchMotionSpec | Sequence[PitchMotionSpec | None] | None = None,
    pitch_motion: PitchMotionSpec | Sequence[PitchMotionSpec | None] | None = None,
    labels: Sequence[str] | None = None,
) -> Phrase:
    """Build a frequency-resolved phrase from ratio material in a local context."""
    return line(
        tones=resolve_ratios(context, tones),
        rhythm=rhythm,
        pitch_kind="freq",
        amp=amp,
        amp_db=amp_db,
        synth_defaults=synth_defaults,
        articulation=articulation,
        motions=motions,
        pitch_motion=pitch_motion,
        labels=labels,
    )


def grid_ratio_line(
    tones: Sequence[float],
    durations: Sequence[DurationLike],
    *,
    context: HarmonicContext,
    timeline: Timeline,
    amp: float | None = None,
    amp_db: float | None = None,
    synth_defaults: dict[str, Any] | None = None,
    articulation: ArticulationSpec | None = None,
    motions: PitchMotionSpec | Sequence[PitchMotionSpec | None] | None = None,
    pitch_motion: PitchMotionSpec | Sequence[PitchMotionSpec | None] | None = None,
    labels: Sequence[str] | None = None,
) -> Phrase:
    """Build a ratio-authored phrase from beat-relative durations."""
    beat_durations = _resolve_duration_beats_sequence(durations)
    phrase = ratio_line(
        tones=tones,
        rhythm=_resolve_grid_second_spans(beat_durations, timeline=timeline),
        context=context,
        amp=amp,
        amp_db=amp_db,
        synth_defaults=synth_defaults,
        articulation=articulation,
        motions=motions,
        pitch_motion=pitch_motion,
        labels=labels,
    )
    return Phrase(
        events=phrase.events,
        beat_timings=_build_beat_timings(beat_durations),
    )


def place_ratio_line(
    score: Score,
    voice_name: str,
    *,
    section: ContextSection,
    tones: Sequence[float],
    rhythm: RhythmCell | Sequence[float],
    offset: float = 0.0,
    amp: float | None = None,
    amp_db: float | None = None,
    synth_defaults: dict[str, Any] | None = None,
    articulation: ArticulationSpec | None = None,
    motions: PitchMotionSpec | Sequence[PitchMotionSpec | None] | None = None,
    pitch_motion: PitchMotionSpec | Sequence[PitchMotionSpec | None] | None = None,
    labels: Sequence[str] | None = None,
) -> list[NoteEvent]:
    """Resolve ratio material in a section and place it on a score."""
    if offset < 0:
        raise ValueError("offset must be non-negative")
    phrase = ratio_line(
        tones=tones,
        rhythm=rhythm,
        context=section.context,
        amp=amp,
        amp_db=amp_db,
        synth_defaults=synth_defaults,
        articulation=articulation,
        motions=motions,
        pitch_motion=pitch_motion,
        labels=labels,
    )
    return score.add_phrase(voice_name, phrase, start=section.start + offset)


def place_ratio_chord(
    score: Score,
    voice_name: str,
    *,
    section: ContextSection,
    ratios: Sequence[float],
    duration: float,
    start: float = 0.0,
    gap: float = 0.0,
    amp: float | Sequence[float] = 1.0,
    synth: dict[str, Any] | None = None,
    labels: Sequence[str] | None = None,
) -> list[NoteEvent]:
    """Place simultaneous or slightly staggered ratio-based notes into a section."""
    if duration <= 0:
        raise ValueError("duration must be positive")
    if start < 0:
        raise ValueError("start must be non-negative")
    if gap < 0:
        raise ValueError("gap must be non-negative")

    resolved_freqs = resolve_ratios(section.context, ratios)
    amps = _expand_positive_values(amp, len(resolved_freqs), "amp")
    note_labels = tuple(labels) if labels is not None else (None,) * len(resolved_freqs)
    if len(note_labels) != len(resolved_freqs):
        raise ValueError("labels must have the same length as ratios")

    notes: list[NoteEvent] = []
    for index, (freq, amp_value, label) in enumerate(
        zip(resolved_freqs, amps, note_labels, strict=True)
    ):
        notes.append(
            score.add_note(
                voice_name,
                start=section.start + start + (index * gap),
                duration=duration,
                freq=freq,
                amp=amp_value,
                synth=dict(synth) if synth is not None else None,
                label=label,
            )
        )
    return notes


def with_gate(phrase: Phrase, gate: float | Sequence[float]) -> Phrase:
    """Return a phrase with scaled note durations and unchanged onsets."""
    gates = _expand_positive_values(gate, len(phrase.events), "gate")
    events = []
    for event, gate_value in zip(phrase.events, gates, strict=True):
        duration = event.duration * gate_value
        if duration <= 0:
            raise ValueError("gate produced a non-positive duration")
        events.append(
            replace(
                event,
                duration=duration,
                synth=dict(event.synth) if event.synth is not None else None,
            )
        )
    return Phrase(events=tuple(events))


def with_accent_pattern(
    phrase: Phrase, accent_pattern: float | Sequence[float]
) -> Phrase:
    """Return a phrase with amplitude accents applied per event."""
    accents = _expand_non_negative_values(
        accent_pattern, len(phrase.events), "accent_pattern"
    )
    events: list[NoteEvent] = []
    for event, accent in zip(phrase.events, accents, strict=True):
        resolved_amp = _require_resolved_amp(event)
        events.append(
            replace(
                event,
                amp=resolved_amp * accent,
                synth=dict(event.synth) if event.synth is not None else None,
            )
        )
    return Phrase(events=tuple(events))


def with_tail_breath(phrase: Phrase, tail_breath: float) -> Phrase:
    """Shorten the last event slightly to create phrase breathing room."""
    if tail_breath < 0:
        raise ValueError("tail_breath must be non-negative")
    if not phrase.events or tail_breath == 0:
        return phrase

    last_event = phrase.events[-1]
    duration = last_event.duration - tail_breath
    if duration <= 0:
        raise ValueError("articulation shortens a note to a non-positive duration")
    events = list(phrase.events[:-1])
    events.append(
        replace(
            last_event,
            duration=duration,
            synth=dict(last_event.synth) if last_event.synth is not None else None,
        )
    )
    return Phrase(events=tuple(events))


def with_synth_ramp(
    phrase: Phrase,
    *,
    start: dict[str, float],
    end: dict[str, float],
) -> Phrase:
    """Interpolate synth parameters across successive phrase events."""
    if not phrase.events:
        return phrase
    if not start or not end:
        raise ValueError("start and end synth ramps must not be empty")
    if set(start) != set(end):
        raise ValueError("start and end synth ramps must use the same parameter keys")

    if len(phrase.events) == 1:
        fractions = (0.0,)
    else:
        fractions = tuple(
            index / (len(phrase.events) - 1) for index in range(len(phrase.events))
        )

    events: list[NoteEvent] = []
    for event, fraction in zip(phrase.events, fractions, strict=True):
        note_synth = dict(event.synth or {})
        for key in start:
            start_value = float(start[key])
            end_value = float(end[key])
            note_synth[key] = start_value + ((end_value - start_value) * fraction)
        events.append(replace(event, synth=note_synth))
    return Phrase(events=tuple(events))


def bar_automation(
    *,
    target: str,
    timeline: Timeline,
    points: Sequence[tuple[int, float, float]],
    mode: AutomationMode = "replace",
    clamp_min: float | None = None,
    clamp_max: float | None = None,
) -> AutomationSpec:
    """Build a linear synth automation lane from bar/beat anchor points.

    The returned lane uses the first anchor value as its default value before the
    first segment starts. To keep the last value alive through a piece tail,
    include a final anchor at the desired endpoint with the same value.
    """
    if len(points) < 2:
        raise ValueError(
            "bar_automation requires at least two (bar, beat, value) points"
        )

    previous_time: float | None = None
    previous_value: float | None = None
    segments: list[AutomationSegment] = []

    for bar, beat, value in points:
        if bar < 1:
            raise ValueError("bar_automation bar numbers must be >= 1")
        time_seconds = timeline.at(bar=bar, beat=beat)
        if previous_time is not None and time_seconds <= previous_time:
            raise ValueError(
                "bar_automation points must be strictly increasing in time"
            )
        if previous_time is not None and previous_value is not None:
            segments.append(
                AutomationSegment(
                    start=previous_time,
                    end=time_seconds,
                    shape="linear",
                    start_value=previous_value,
                    end_value=float(value),
                )
            )
        previous_time = time_seconds
        previous_value = float(value)

    return AutomationSpec(
        target=AutomationTarget(kind="synth", name=target),
        segments=tuple(segments),
        default_value=float(points[0][2]),
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        mode=mode,
    )


def staccato(phrase: Phrase, gate: float = 0.45) -> Phrase:
    """Convenience wrapper for clipped phrasing."""
    return with_gate(phrase, gate)


def legato(phrase: Phrase, gate: float = 1.1) -> Phrase:
    """Convenience wrapper for overlapping phrasing."""
    return with_gate(phrase, gate)


def echo(
    phrase: Phrase,
    *,
    delay: float,
    amp_scale: float = 0.7,
    partial_shift: float = 0.0,
    freq_scale: float = 1.0,
) -> Phrase:
    """Return a delayed, quieter copy of a phrase."""
    if delay < 0:
        raise ValueError("delay must be non-negative")
    if amp_scale <= 0:
        raise ValueError("amp_scale must be positive")
    if freq_scale <= 0:
        raise ValueError("freq_scale must be positive")

    has_partial_events = any(event.partial is not None for event in phrase.events)
    has_freq_events = any(event.freq is not None for event in phrase.events)
    if (
        has_partial_events
        and has_freq_events
        and partial_shift != 0.0
        and freq_scale != 1.0
    ):
        raise ValueError(
            "partial_shift and freq_scale are mutually exclusive for mixed-pitch phrases"
        )
    if partial_shift != 0.0 and has_freq_events:
        raise ValueError(
            "partial_shift has no effect on freq-pitched events; use freq_scale instead"
        )

    echoed_events: list[NoteEvent] = []
    for event in phrase.events:
        new_partial = event.partial
        if new_partial is not None:
            new_partial += partial_shift
        new_freq = event.freq
        if new_freq is not None:
            new_freq *= freq_scale
        resolved_amp = _require_resolved_amp(event)
        echoed_events.append(
            replace(
                event,
                start=event.start + delay,
                amp=resolved_amp * amp_scale,
                partial=new_partial,
                freq=new_freq,
                synth=dict(event.synth) if event.synth is not None else None,
            )
        )
    return Phrase(events=tuple(echoed_events))


def concat(*phrases: Phrase) -> Phrase:
    """Chain phrases end-to-end."""
    if not phrases:
        raise ValueError("phrases must not be empty")

    events: list[NoteEvent] = []
    cursor = 0.0
    for phrase in phrases:
        for event in phrase.events:
            events.append(
                replace(
                    event,
                    start=event.start + cursor,
                    synth=dict(event.synth) if event.synth is not None else None,
                )
            )
        cursor += phrase.duration
    return Phrase(events=tuple(sorted(events, key=lambda event: event.start)))


def overlay(*phrases: Phrase, offset: float = 0.0) -> Phrase:
    """Superimpose phrases, optionally staggering each successive phrase."""
    if not phrases:
        raise ValueError("phrases must not be empty")
    if offset < 0:
        raise ValueError("offset must be non-negative")

    events: list[NoteEvent] = []
    for index, phrase in enumerate(phrases):
        phrase_offset = index * offset
        for event in phrase.events:
            events.append(
                replace(
                    event,
                    start=event.start + phrase_offset,
                    synth=dict(event.synth) if event.synth is not None else None,
                )
            )
    return Phrase(events=tuple(sorted(events, key=lambda event: event.start)))


def recontextualize_phrase(
    phrase: Phrase,
    *,
    target_context: HarmonicContext,
    source_tonic: float = 1.0,
) -> Phrase:
    """Resolve a phrase against a new harmonic context as concrete frequencies."""
    if source_tonic <= 0:
        raise ValueError("source_tonic must be positive")

    recontextualized_events: list[NoteEvent] = []
    for event in phrase.events:
        source_pitch = event.freq if event.freq is not None else event.partial
        if source_pitch is None:
            raise ValueError("event must define freq or partial")
        ratio = float(source_pitch) / source_tonic
        recontextualized_events.append(
            replace(
                event,
                freq=target_context.resolve_ratio(ratio),
                partial=None,
                synth=dict(event.synth) if event.synth is not None else None,
            )
        )
    return Phrase(events=tuple(recontextualized_events))


def sequence(
    score: Score,
    voice_name: str,
    phrase: Phrase,
    *,
    starts: Sequence[float],
    time_scales: float | Sequence[float] = 1.0,
    partial_shifts: float | Sequence[float] = 0.0,
    amp_scales: float | Sequence[float] = 1.0,
    reverses: bool | Sequence[bool] = False,
    sections: Sequence[ContextSection | None] | None = None,
    source_tonic: float = 1.0,
) -> list[list[NoteEvent]]:
    """Place repeated phrase entries with optional per-entry transforms."""
    if not starts:
        raise ValueError("starts must not be empty")

    entry_count = len(starts)
    expanded_starts = tuple(float(start) for start in starts)
    if any(start < 0 for start in expanded_starts):
        raise ValueError("starts must be non-negative")

    expanded_time_scales = _expand_positive_values(
        time_scales, entry_count, "time_scales"
    )
    expanded_partial_shifts = _expand_scalar_values(
        partial_shifts,
        entry_count,
        "partial_shifts",
    )
    expanded_amp_scales = _expand_positive_values(amp_scales, entry_count, "amp_scales")
    expanded_reverses = _expand_bool_values(reverses, entry_count, "reverses")

    if sections is None:
        expanded_sections: tuple[ContextSection | None, ...] = (None,) * entry_count
    else:
        expanded_sections = tuple(sections)
        if len(expanded_sections) != entry_count:
            raise ValueError("sections length must match starts")

    placed_entries: list[list[NoteEvent]] = []
    for start, time_scale, partial_shift, amp_scale, reverse, section in zip(
        expanded_starts,
        expanded_time_scales,
        expanded_partial_shifts,
        expanded_amp_scales,
        expanded_reverses,
        expanded_sections,
        strict=True,
    ):
        phrase_to_place = phrase
        absolute_start = start
        if section is not None:
            if partial_shift != 0.0 and any(
                event.partial is not None for event in phrase_to_place.events
            ):
                phrase_to_place = Phrase(
                    events=tuple(
                        phrase_to_place.transformed(partial_shift=partial_shift)
                    )
                )
                partial_shift = 0.0
            phrase_to_place = recontextualize_phrase(
                phrase_to_place,
                target_context=section.context,
                source_tonic=source_tonic,
            )
            absolute_start += section.start

        placed_entries.append(
            score.add_phrase(
                voice_name,
                phrase_to_place,
                start=absolute_start,
                time_scale=time_scale,
                partial_shift=partial_shift,
                amp_scale=amp_scale,
                reverse=reverse,
            )
        )
    return placed_entries


def grid_sequence(
    score: Score,
    voice_name: str,
    phrase: Phrase,
    *,
    timeline: Timeline,
    at: Sequence[TimePointLike],
    time_scales: float | Sequence[float] = 1.0,
    partial_shifts: float | Sequence[float] = 0.0,
    amp_scales: float | Sequence[float] = 1.0,
    reverses: bool | Sequence[bool] = False,
    sections: Sequence[ContextSection | None] | None = None,
    source_tonic: float = 1.0,
) -> list[list[NoteEvent]]:
    """Place repeated phrase entries using beat/bar references."""
    start_beats = tuple(
        _resolve_time_point_beats(value, timeline=timeline) for value in at
    )
    return _place_grid_sequence_entries(
        score,
        voice_name,
        phrase,
        timeline=timeline,
        start_beats=start_beats,
        time_scales=time_scales,
        partial_shifts=partial_shifts,
        amp_scales=amp_scales,
        reverses=reverses,
        sections=sections,
        source_tonic=source_tonic,
    )


def canon(
    score: Score,
    *,
    voice_names: Sequence[str],
    phrase: Phrase,
    start: float,
    delays: float | Sequence[float],
    repeats: int = 1,
    repeat_gap: float = 0.0,
    amp_scales: float | Sequence[float] = 1.0,
    partial_shifts: float | Sequence[float] = 0.0,
    time_scales: float | Sequence[float] = 1.0,
    reverses: bool | Sequence[bool] = False,
    sections: Sequence[ContextSection | None] | None = None,
    source_tonic: float = 1.0,
) -> dict[str, list[list[NoteEvent]]]:
    """Place delayed imitative entries of a phrase across voices."""
    if start < 0:
        raise ValueError("start must be non-negative")
    if not voice_names:
        raise ValueError("voice_names must not be empty")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if repeat_gap < 0:
        raise ValueError("repeat_gap must be non-negative")

    entry_count = len(voice_names)
    if isinstance(delays, (int, float)):
        delay_values = tuple(float(delays) for _ in range(max(entry_count - 1, 0)))
    else:
        delay_values = tuple(float(delay) for delay in delays)
        if len(delay_values) != max(entry_count - 1, 0):
            raise ValueError("delays must have length len(voice_names) - 1")
    if any(delay < 0 for delay in delay_values):
        raise ValueError("delays must be non-negative")

    starts = [start]
    cursor = start
    for delay in delay_values:
        cursor += delay
        starts.append(cursor)

    expanded_amp_scales = _expand_positive_values(amp_scales, entry_count, "amp_scales")
    expanded_partial_shifts = _expand_scalar_values(
        partial_shifts,
        entry_count,
        "partial_shifts",
    )
    expanded_time_scales = _expand_positive_values(
        time_scales, entry_count, "time_scales"
    )
    expanded_reverses = _expand_bool_values(reverses, entry_count, "reverses")
    if sections is None:
        expanded_sections: tuple[ContextSection | None, ...] = (None,) * entry_count
    else:
        expanded_sections = tuple(sections)
        if len(expanded_sections) != entry_count:
            raise ValueError("sections length must match voice_names")

    placed: dict[str, list[list[NoteEvent]]] = {}
    for (
        voice_name,
        entry_start,
        amp_scale,
        partial_shift,
        time_scale,
        reverse,
        section,
    ) in zip(
        voice_names,
        starts,
        expanded_amp_scales,
        expanded_partial_shifts,
        expanded_time_scales,
        expanded_reverses,
        expanded_sections,
        strict=True,
    ):
        repeat_starts = tuple(
            entry_start + (repeat_index * ((phrase.duration * time_scale) + repeat_gap))
            for repeat_index in range(repeats)
        )
        placed[voice_name] = sequence(
            score,
            voice_name,
            phrase,
            starts=repeat_starts,
            amp_scales=(amp_scale,) * repeats,
            partial_shifts=(partial_shift,) * repeats,
            time_scales=(time_scale,) * repeats,
            reverses=(reverse,) * repeats,
            sections=(section,) * repeats,
            source_tonic=source_tonic,
        )
    return placed


def grid_canon(
    score: Score,
    *,
    voice_names: Sequence[str],
    phrase: Phrase,
    timeline: Timeline,
    start: TimePointLike,
    delays: DurationLike | Sequence[DurationLike],
    repeats: int = 1,
    repeat_gap: float | DurationLike = 0.0,
    amp_scales: float | Sequence[float] = 1.0,
    partial_shifts: float | Sequence[float] = 0.0,
    time_scales: float | Sequence[float] = 1.0,
    reverses: bool | Sequence[bool] = False,
    sections: Sequence[ContextSection | None] | None = None,
    source_tonic: float = 1.0,
) -> dict[str, list[list[NoteEvent]]]:
    """Place delayed imitative entries using beat/bar timing references."""
    start_beats = _resolve_time_point_beats(start, timeline=timeline)
    if phrase.beat_timings is not None:
        return _place_grid_canon_entries(
            score,
            voice_names=voice_names,
            phrase=phrase,
            timeline=timeline,
            start_beats=start_beats,
            delays=delays,
            repeats=repeats,
            repeat_gap=repeat_gap,
            amp_scales=amp_scales,
            partial_shifts=partial_shifts,
            time_scales=time_scales,
            reverses=reverses,
            sections=sections,
            source_tonic=source_tonic,
        )

    resolved_delays = (
        _resolve_duration_value(delays, timeline=timeline, allow_zero=True)
        if isinstance(delays, (int, float, BeatSpan, BeatValue))
        else tuple(
            _resolve_duration_value(delay, timeline=timeline, allow_zero=True)
            for delay in delays
        )
    )

    return canon(
        score,
        voice_names=voice_names,
        phrase=phrase,
        start=timeline.position(B(start_beats)),
        delays=resolved_delays,
        repeats=repeats,
        repeat_gap=_resolve_duration_value(
            repeat_gap,
            timeline=timeline,
            allow_zero=True,
        ),
        amp_scales=amp_scales,
        partial_shifts=partial_shifts,
        time_scales=time_scales,
        reverses=reverses,
        sections=sections,
        source_tonic=source_tonic,
    )


def voiced_ratio_chord(
    ratios: Sequence[float],
    *,
    context: HarmonicContext,
    voicing: str = "close",
    inversion: int = 0,
    low_hz: float | None = None,
    high_hz: float | None = None,
) -> list[float]:
    """Resolve ratio material into a voiced chord in a target register."""
    if not ratios:
        raise ValueError("ratios must not be empty")
    if voicing not in {"close", "open", "spread", "drop2", "drop3"}:
        raise ValueError(
            "voicing must be 'close', 'open', 'spread', 'drop2', or 'drop3'"
        )

    voiced_freqs = sorted(resolve_ratios(context, ratios))
    inversion_count = int(inversion)
    if inversion_count < 0:
        raise ValueError("inversion must be non-negative")
    for _ in range(inversion_count):
        voiced_freqs[0] *= 2.0
        voiced_freqs.sort()

    if voicing == "open":
        for index in range(1, len(voiced_freqs), 2):
            voiced_freqs[index] *= 2.0
        voiced_freqs.sort()
    elif voicing == "spread":
        for index in range(1, len(voiced_freqs)):
            voiced_freqs[index] *= 2.0**index
        voiced_freqs.sort()
    elif voicing == "drop2":
        _drop_chord_tone(voiced_freqs, drop_number=2)
    elif voicing == "drop3":
        _drop_chord_tone(voiced_freqs, drop_number=3)

    if low_hz is not None and low_hz <= 0:
        raise ValueError("low_hz must be positive")
    if high_hz is not None and high_hz <= 0:
        raise ValueError("high_hz must be positive")
    if low_hz is not None and high_hz is not None and low_hz >= high_hz:
        raise ValueError("low_hz must be less than high_hz")

    if low_hz is not None or high_hz is not None:
        voiced_freqs = _fit_chord_to_register(
            voiced_freqs,
            low_hz=low_hz,
            high_hz=high_hz,
        )
    return voiced_freqs


def progression(
    score: Score,
    voice_name: str,
    *,
    sections: Sequence[ContextSection],
    chords: Sequence[Sequence[float]],
    pattern: str = "block",
    amp: float | Sequence[float] = 1.0,
    duration_scale: float = 1.0,
    voicing: str = "close",
    inversion: int | Sequence[int] = 0,
    arpeggio_order: Sequence[int] | str = "ascending",
    low_hz: float | None = None,
    high_hz: float | None = None,
) -> list[NoteEvent]:
    """Place a harmonic progression using simple accompaniment patterns."""
    if not sections:
        raise ValueError("sections must not be empty")
    if len(sections) != len(chords):
        raise ValueError("sections and chords must have the same length")
    if duration_scale <= 0:
        raise ValueError("duration_scale must be positive")
    if pattern not in {"block", "arpeggio", "pedal_upper"}:
        raise ValueError("pattern must be 'block', 'arpeggio', or 'pedal_upper'")

    expanded_amps = _expand_positive_values(amp, len(sections), "amp")
    expanded_inversions = _expand_int_values(inversion, len(sections), "inversion")

    placed_notes: list[NoteEvent] = []
    for section, chord, amp_value, inversion_value in zip(
        sections,
        chords,
        expanded_amps,
        expanded_inversions,
        strict=True,
    ):
        freqs = voiced_ratio_chord(
            chord,
            context=section.context,
            voicing=voicing,
            inversion=inversion_value,
            low_hz=low_hz,
            high_hz=high_hz,
        )
        note_duration = section.duration * duration_scale

        if pattern == "block":
            per_note_amp = amp_value / len(freqs)
            for freq in freqs:
                placed_notes.append(
                    score.add_note(
                        voice_name,
                        start=section.start,
                        duration=note_duration,
                        freq=freq,
                        amp=per_note_amp,
                    )
                )
            continue

        if pattern == "arpeggio":
            ordered_freqs = _ordered_arpeggio_freqs(freqs, arpeggio_order)
            step = section.duration / len(ordered_freqs)
            per_note_amp = amp_value
            for index, freq in enumerate(ordered_freqs):
                placed_notes.append(
                    score.add_note(
                        voice_name,
                        start=section.start + (index * step),
                        duration=step * duration_scale,
                        freq=freq,
                        amp=per_note_amp,
                    )
                )
            continue

        pedal_freq = freqs[0]
        upper_freqs = freqs[1:]
        placed_notes.append(
            score.add_note(
                voice_name,
                start=section.start,
                duration=note_duration,
                freq=pedal_freq,
                amp=amp_value * 0.65,
            )
        )
        if upper_freqs:
            upper_amp = (amp_value * 0.75) / len(upper_freqs)
            for freq in upper_freqs:
                placed_notes.append(
                    score.add_note(
                        voice_name,
                        start=section.start,
                        duration=note_duration * 0.82,
                        freq=freq,
                        amp=upper_amp,
                    )
                )
    return placed_notes


def _require_resolved_amp(event: NoteEvent) -> float:
    """Return a concrete amplitude for an event that should already be resolved."""
    if event.amp is None:
        raise ValueError("event amp unexpectedly missing")
    return float(event.amp)


def _resolve_rhythm_values(
    rhythm_cell: RhythmCell,
    tone_count: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(rhythm_cell.spans) > tone_count:
        raise ValueError(
            "tones and spans must have the same length or a shorter rhythm"
        )

    if len(rhythm_cell.spans) == tone_count:
        spans = rhythm_cell.spans
        gates = _expand_positive_values(rhythm_cell.gates, tone_count, "gates")
        return spans, gates

    base_gates = _expand_positive_values(
        rhythm_cell.gates,
        len(rhythm_cell.spans),
        "gates",
    )
    spans = tuple(islice(cycle(rhythm_cell.spans), tone_count))
    gates = tuple(islice(cycle(base_gates), tone_count))
    return spans, gates


def _drop_chord_tone(voiced_freqs: list[float], *, drop_number: int) -> None:
    if len(voiced_freqs) < drop_number:
        raise ValueError(f"{drop_number=} requires at least {drop_number} chord tones")
    drop_index = len(voiced_freqs) - drop_number
    voiced_freqs[drop_index] *= 0.5
    voiced_freqs.sort()


def _ordered_arpeggio_freqs(
    freqs: Sequence[float],
    arpeggio_order: Sequence[int] | str,
) -> list[float]:
    if isinstance(arpeggio_order, str):
        if arpeggio_order == "ascending":
            order = list(range(len(freqs)))
        elif arpeggio_order == "descending":
            order = list(range(len(freqs) - 1, -1, -1))
        elif arpeggio_order == "inside_out":
            order = _inside_out_order(len(freqs))
        else:
            raise ValueError(
                "arpeggio_order must be 'ascending', 'descending', 'inside_out', or a valid index sequence"
            )
    else:
        order = [int(index) for index in arpeggio_order]
        if len(order) != len(freqs):
            raise ValueError("custom arpeggio_order length must match the chord size")
        if len(set(order)) != len(freqs):
            raise ValueError("custom arpeggio_order must not repeat indices")
        if any(index < 0 or index >= len(freqs) for index in order):
            raise ValueError(
                "custom arpeggio_order indices must be within chord bounds"
            )

    return [freqs[index] for index in order]


def _inside_out_order(count: int) -> list[int]:
    if count <= 0:
        return []

    order = [count // 2]
    left_index = (count // 2) - 1
    right_index = (count // 2) + 1
    while len(order) < count:
        if left_index >= 0:
            order.append(left_index)
            left_index -= 1
        if right_index < count:
            order.append(right_index)
            right_index += 1
    return order


def _resolve_motion_input(
    motions: PitchMotionSpec | Sequence[PitchMotionSpec | None] | None,
    pitch_motion: PitchMotionSpec | Sequence[PitchMotionSpec | None] | None,
) -> PitchMotionSpec | Sequence[PitchMotionSpec | None] | None:
    if motions is not None and pitch_motion is not None and motions != pitch_motion:
        raise ValueError("use only one of motions or pitch_motion")
    return pitch_motion if pitch_motion is not None else motions


def _expand_motions(
    motions: PitchMotionSpec | Sequence[PitchMotionSpec | None] | None,
    count: int,
) -> tuple[PitchMotionSpec | None, ...]:
    if motions is None:
        return (None,) * count
    if isinstance(motions, PitchMotionSpec):
        return (motions,) * count

    expanded = tuple(motions)
    if len(expanded) != count:
        raise ValueError(f"motions must have length {count}")
    return expanded


def _expand_positive_values(
    value: float | Sequence[float],
    count: int,
    field_name: str,
) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        expanded = (float(value),) * count
    else:
        expanded = tuple(float(item) for item in value)
        if len(expanded) != count:
            raise ValueError(f"{field_name} length must match the phrase length")
    if any(item <= 0 for item in expanded):
        raise ValueError(f"{field_name} must be positive")
    return expanded


def _expand_scalar_values(
    value: float | Sequence[float],
    count: int,
    field_name: str,
) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        return (float(value),) * count
    expanded = tuple(float(item) for item in value)
    if len(expanded) != count:
        raise ValueError(f"{field_name} length must match the phrase length")
    return expanded


def _expand_bool_values(
    value: bool | Sequence[bool],
    count: int,
    field_name: str,
) -> tuple[bool, ...]:
    if isinstance(value, bool):
        return (value,) * count
    expanded = tuple(bool(item) for item in value)
    if len(expanded) != count:
        raise ValueError(f"{field_name} length must match the phrase length")
    return expanded


def _expand_int_values(
    value: int | Sequence[int],
    count: int,
    field_name: str,
) -> tuple[int, ...]:
    if isinstance(value, int):
        expanded = (value,) * count
    else:
        expanded = tuple(int(item) for item in value)
        if len(expanded) != count:
            raise ValueError(f"{field_name} length must match the phrase length")
    return expanded


def _expand_non_negative_values(
    value: float | Sequence[float],
    count: int,
    field_name: str,
) -> tuple[float, ...]:
    if isinstance(value, (int, float)):
        expanded = (float(value),) * count
    else:
        expanded = tuple(float(item) for item in value)
        if len(expanded) != count:
            raise ValueError(f"{field_name} length must match the phrase length")
    if any(item < 0 for item in expanded):
        raise ValueError(f"{field_name} must be non-negative")
    return expanded


def _validate_positive_scalar_or_sequence(
    value: float | Sequence[float], field_name: str
) -> None:
    if isinstance(value, (int, float)):
        if float(value) <= 0:
            raise ValueError(f"{field_name} must be positive")
        return
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    if any(float(item) <= 0 for item in value):
        raise ValueError(f"{field_name} values must be positive")


def _validate_non_negative_scalar_or_sequence(
    value: float | Sequence[float],
    field_name: str,
) -> None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError(f"{field_name} must not be empty")
        if any(float(item) < 0 for item in value):
            raise ValueError(f"{field_name} values must be non-negative")
        return
    if float(value) < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _build_note_synth(
    *,
    synth_defaults: dict[str, Any] | None,
    attack_scale: float,
    release_scale: float,
) -> dict[str, Any] | None:
    if synth_defaults is None and attack_scale == 1.0 and release_scale == 1.0:
        return None

    note_synth = dict(synth_defaults or {})
    if attack_scale != 1.0:
        note_synth["attack_scale"] = attack_scale
    if release_scale != 1.0:
        note_synth["release_scale"] = release_scale
    return note_synth


def _fit_chord_to_register(
    freqs: Sequence[float],
    *,
    low_hz: float | None,
    high_hz: float | None,
) -> list[float]:
    fitted = [float(freq) for freq in freqs]
    for index, freq in enumerate(fitted):
        while low_hz is not None and freq < low_hz:
            freq *= 2.0
        while high_hz is not None and freq >= high_hz:
            freq /= 2.0
        fitted[index] = freq
    return sorted(fitted)


def _resolve_duration_sequence(
    durations: Sequence[DurationLike],
    *,
    timeline: Timeline,
) -> tuple[float, ...]:
    if not durations:
        raise ValueError("durations must not be empty")
    return tuple(
        _resolve_duration_value(duration, timeline=timeline, allow_zero=False)
        for duration in durations
    )


def _resolve_duration_beats_sequence(
    durations: Sequence[DurationLike],
) -> tuple[float, ...]:
    if not durations:
        raise ValueError("durations must not be empty")
    return tuple(_coerce_duration_beats_value(duration) for duration in durations)


def _resolve_duration_value(
    value: float | DurationLike,
    *,
    timeline: Timeline,
    allow_zero: bool,
) -> float:
    if allow_zero and isinstance(value, (int, float)) and float(value) == 0.0:
        return 0.0
    if allow_zero and isinstance(value, (BeatSpan, BeatValue)) and value.beats == 0.0:
        return 0.0
    return timeline.duration(value)


def _resolve_time_point(
    value: TimePointLike,
    *,
    timeline: Timeline,
) -> float:
    return timeline.position(value)


def _resolve_time_point_beats(
    value: TimePointLike,
    *,
    timeline: Timeline,
) -> float:
    return timeline.absolute_beats(value)


def _resolve_grid_second_spans(
    beat_durations: Sequence[float],
    *,
    timeline: Timeline,
) -> tuple[float, ...]:
    cursor_beats = 0.0
    starts = [0.0]
    for duration_beats in beat_durations:
        cursor_beats += duration_beats
        starts.append(cursor_beats)
    resolved_seconds = tuple(timeline.position(B(start)) for start in starts)
    return tuple(
        end - start
        for start, end in zip(
            resolved_seconds,
            resolved_seconds[1:],
            strict=False,
        )
    )


def _build_beat_timings(beat_durations: Sequence[float]) -> tuple[BeatTiming, ...]:
    cursor_beats = 0.0
    timings: list[BeatTiming] = []
    for duration_beats in beat_durations:
        timings.append(
            BeatTiming(start_beats=cursor_beats, duration_beats=duration_beats)
        )
        cursor_beats += duration_beats
    return tuple(timings)


def _coerce_duration_beats_value(value: DurationLike) -> float:
    beats = value.beats if isinstance(value, (BeatSpan, BeatValue)) else float(value)
    if beats <= 0:
        raise ValueError("duration beats must be positive")
    return beats


def _place_grid_sequence_entries(
    score: Score,
    voice_name: str,
    phrase: Phrase,
    *,
    timeline: Timeline,
    start_beats: Sequence[float],
    time_scales: float | Sequence[float],
    partial_shifts: float | Sequence[float],
    amp_scales: float | Sequence[float],
    reverses: bool | Sequence[bool],
    sections: Sequence[ContextSection | None] | None,
    source_tonic: float,
) -> list[list[NoteEvent]]:
    if phrase.beat_timings is None:
        return sequence(
            score,
            voice_name,
            phrase,
            starts=tuple(timeline.position(B(start)) for start in start_beats),
            time_scales=time_scales,
            partial_shifts=partial_shifts,
            amp_scales=amp_scales,
            reverses=reverses,
            sections=sections,
            source_tonic=source_tonic,
        )

    if not start_beats:
        raise ValueError("starts must not be empty")
    if any(start < 0 for start in start_beats):
        raise ValueError("starts must be non-negative")

    entry_count = len(start_beats)
    expanded_time_scales = _expand_positive_values(
        time_scales, entry_count, "time_scales"
    )
    expanded_partial_shifts = _expand_scalar_values(
        partial_shifts,
        entry_count,
        "partial_shifts",
    )
    expanded_amp_scales = _expand_positive_values(amp_scales, entry_count, "amp_scales")
    expanded_reverses = _expand_bool_values(reverses, entry_count, "reverses")

    if sections is None:
        expanded_sections: tuple[ContextSection | None, ...] = (None,) * entry_count
    else:
        expanded_sections = tuple(sections)
        if len(expanded_sections) != entry_count:
            raise ValueError("sections length must match starts")

    phrase_duration_beats = max(
        timing.start_beats + timing.duration_beats for timing in phrase.beat_timings
    )
    placed_entries: list[list[NoteEvent]] = []

    for start_beat, time_scale, partial_shift, amp_scale, reverse, section in zip(
        start_beats,
        expanded_time_scales,
        expanded_partial_shifts,
        expanded_amp_scales,
        expanded_reverses,
        expanded_sections,
        strict=True,
    ):
        placed_notes: list[NoteEvent] = []
        for event, beat_timing in zip(
            phrase.events,
            phrase.beat_timings,
            strict=True,
        ):
            if reverse:
                relative_start_beats = (
                    phrase_duration_beats
                    - beat_timing.start_beats
                    - beat_timing.duration_beats
                ) * time_scale
            else:
                relative_start_beats = beat_timing.start_beats * time_scale

            absolute_start_beats = start_beat + relative_start_beats
            scaled_duration_beats = beat_timing.duration_beats * time_scale
            absolute_end_beats = absolute_start_beats + scaled_duration_beats

            placed_event = _resolve_grid_event_pitch(
                event=event,
                partial_shift=partial_shift,
                amp_scale=amp_scale,
                section=section,
                source_tonic=source_tonic,
            )
            placed_notes.append(
                score.add_note(
                    voice_name,
                    start=timeline.position(B(absolute_start_beats)),
                    duration=timeline.position(B(absolute_end_beats))
                    - timeline.position(B(absolute_start_beats)),
                    partial=placed_event.partial,
                    freq=placed_event.freq,
                    amp=placed_event.amp,
                    velocity=placed_event.velocity,
                    pitch_motion=placed_event.pitch_motion,
                    synth=dict(placed_event.synth)
                    if placed_event.synth is not None
                    else None,
                    label=placed_event.label,
                    automation=list(placed_event.automation)
                    if placed_event.automation is not None
                    else None,
                )
            )
        placed_entries.append(placed_notes)
    return placed_entries


def _resolve_grid_event_pitch(
    *,
    event: NoteEvent,
    partial_shift: float,
    amp_scale: float,
    section: ContextSection | None,
    source_tonic: float,
) -> NoteEvent:
    resolved_amp = _require_resolved_amp(event) * amp_scale

    if section is None:
        new_partial = None if event.partial is None else event.partial + partial_shift
        return replace(
            event,
            amp=resolved_amp,
            partial=new_partial,
        )

    source_pitch = event.freq
    shifted_partial = event.partial
    if shifted_partial is not None:
        shifted_partial += partial_shift
        source_pitch = shifted_partial

    if source_pitch is None:
        raise ValueError("event must define freq or partial")
    ratio = float(source_pitch) / source_tonic
    return replace(
        event,
        amp=resolved_amp,
        freq=section.context.resolve_ratio(ratio),
        partial=None,
    )


def _place_grid_canon_entries(
    score: Score,
    *,
    voice_names: Sequence[str],
    phrase: Phrase,
    timeline: Timeline,
    start_beats: float,
    delays: DurationLike | Sequence[DurationLike],
    repeats: int,
    repeat_gap: float | DurationLike,
    amp_scales: float | Sequence[float],
    partial_shifts: float | Sequence[float],
    time_scales: float | Sequence[float],
    reverses: bool | Sequence[bool],
    sections: Sequence[ContextSection | None] | None,
    source_tonic: float,
) -> dict[str, list[list[NoteEvent]]]:
    if not voice_names:
        raise ValueError("voice_names must not be empty")
    if start_beats < 0:
        raise ValueError("start must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    entry_count = len(voice_names)
    if isinstance(delays, (int, float, BeatSpan, BeatValue)):
        delay_values = tuple(
            _resolve_duration_beats_value(delays, allow_zero=True)
            for _ in range(max(entry_count - 1, 0))
        )
    else:
        delay_values = tuple(
            _resolve_duration_beats_value(delay, allow_zero=True) for delay in delays
        )
        if len(delay_values) != max(entry_count - 1, 0):
            raise ValueError("delays must have length len(voice_names) - 1")
    if any(delay < 0 for delay in delay_values):
        raise ValueError("delays must be non-negative")

    repeat_gap_beats = _resolve_duration_beats_value(repeat_gap, allow_zero=True)
    if repeat_gap_beats < 0:
        raise ValueError("repeat_gap must be non-negative")

    starts = [start_beats]
    cursor = start_beats
    for delay in delay_values:
        cursor += delay
        starts.append(cursor)

    expanded_amp_scales = _expand_positive_values(amp_scales, entry_count, "amp_scales")
    expanded_partial_shifts = _expand_scalar_values(
        partial_shifts,
        entry_count,
        "partial_shifts",
    )
    expanded_time_scales = _expand_positive_values(
        time_scales, entry_count, "time_scales"
    )
    expanded_reverses = _expand_bool_values(reverses, entry_count, "reverses")
    if sections is None:
        expanded_sections: tuple[ContextSection | None, ...] = (None,) * entry_count
    else:
        expanded_sections = tuple(sections)
        if len(expanded_sections) != entry_count:
            raise ValueError("sections length must match voice_names")

    phrase_duration_beats = max(
        timing.start_beats + timing.duration_beats
        for timing in phrase.beat_timings or ()
    )
    placed: dict[str, list[list[NoteEvent]]] = {}
    for (
        voice_name,
        entry_start_beats,
        amp_scale,
        partial_shift,
        time_scale,
        reverse,
        section,
    ) in zip(
        voice_names,
        starts,
        expanded_amp_scales,
        expanded_partial_shifts,
        expanded_time_scales,
        expanded_reverses,
        expanded_sections,
        strict=True,
    ):
        repeat_starts = tuple(
            entry_start_beats
            + (repeat_index * ((phrase_duration_beats * time_scale) + repeat_gap_beats))
            for repeat_index in range(repeats)
        )
        placed[voice_name] = _place_grid_sequence_entries(
            score,
            voice_name,
            phrase,
            timeline=timeline,
            start_beats=repeat_starts,
            amp_scales=(amp_scale,) * repeats,
            partial_shifts=(partial_shift,) * repeats,
            time_scales=(time_scale,) * repeats,
            reverses=(reverse,) * repeats,
            sections=(section,) * repeats,
            source_tonic=source_tonic,
        )
    return placed


def _resolve_duration_beats_value(
    value: float | DurationLike,
    *,
    allow_zero: bool,
) -> float:
    if allow_zero and isinstance(value, (int, float)) and float(value) == 0.0:
        return 0.0
    if allow_zero and isinstance(value, (BeatSpan, BeatValue)) and value.beats == 0.0:
        return 0.0
    return _coerce_duration_beats_value(value)
