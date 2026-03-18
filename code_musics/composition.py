"""Higher-level composition helpers that compile to score events."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from code_musics import synth
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import NoteEvent, Phrase

if TYPE_CHECKING:
    from code_musics.score import Score

__all__ = [
    "ArticulationSpec",
    "ContextSection",
    "ContextSectionSpec",
    "HarmonicContext",
    "PitchMotionSpec",
    "RhythmCell",
    "build_context_sections",
    "echo",
    "legato",
    "line",
    "place_ratio_chord",
    "place_ratio_line",
    "ratio_line",
    "resolve_ratios",
    "staccato",
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
    if len(rhythm_cell.spans) != len(tones):
        raise ValueError("tones and spans must have the same length")
    note_labels = tuple(labels) if labels is not None else (None,) * len(tones)
    if len(note_labels) != len(tones):
        raise ValueError("labels must have the same length as tones")

    motion_input = _resolve_motion_input(motions, pitch_motion)
    note_motions = _expand_motions(motion_input, len(tones))

    articulation = articulation or ArticulationSpec()
    rhythm_gates = _expand_positive_values(rhythm_cell.gates, len(tones), "gates")
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
            rhythm_cell.spans,
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
) -> Phrase:
    """Return a delayed, quieter copy of a phrase."""
    if delay < 0:
        raise ValueError("delay must be non-negative")
    if amp_scale <= 0:
        raise ValueError("amp_scale must be positive")

    echoed_events: list[NoteEvent] = []
    for event in phrase.events:
        new_partial = event.partial
        if new_partial is not None:
            new_partial += partial_shift
        resolved_amp = _require_resolved_amp(event)
        echoed_events.append(
            replace(
                event,
                start=event.start + delay,
                amp=resolved_amp * amp_scale,
                partial=new_partial,
                synth=dict(event.synth) if event.synth is not None else None,
            )
        )
    return Phrase(events=tuple(echoed_events))


def _require_resolved_amp(event: NoteEvent) -> float:
    """Return a concrete amplitude for an event that should already be resolved."""
    if event.amp is None:
        raise ValueError("event amp unexpectedly missing")
    return float(event.amp)


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
