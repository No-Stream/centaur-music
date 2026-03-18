"""Score-domain abstractions for composing pieces."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from code_musics.engines import render_note_signal, resolve_synth_params
from code_musics import synth


@dataclass(frozen=True)
class EffectSpec:
    """Declarative effect-chain item."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NoteEvent:
    """Atomic score event, represented in relative or absolute time."""

    start: float
    duration: float
    amp: float = 1.0
    partial: float | None = None
    freq: float | None = None
    synth: dict[str, Any] | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if (self.partial is None) == (self.freq is None):
            raise ValueError("exactly one of partial or freq must be provided")


@dataclass(frozen=True)
class Phrase:
    """Reusable collection of relative-time note events."""

    events: tuple[NoteEvent, ...]

    @classmethod
    def from_partials(
        cls,
        partials: list[float],
        note_dur: float,
        step: float,
        amp: float = 1.0,
        synth_defaults: dict[str, Any] | None = None,
    ) -> Phrase:
        """Create a phrase from equally spaced harmonic partials."""
        events = tuple(
            NoteEvent(
                start=index * step,
                duration=note_dur,
                amp=amp,
                partial=partial,
                synth=dict(synth_defaults) if synth_defaults is not None else None,
            )
            for index, partial in enumerate(partials)
        )
        return cls(events=events)

    @property
    def duration(self) -> float:
        """Return the phrase duration from note endpoints."""
        if not self.events:
            return 0.0
        return max(event.start + event.duration for event in self.events)

    def transformed(
        self,
        *,
        start: float = 0.0,
        time_scale: float = 1.0,
        partial_shift: float = 0.0,
        amp_scale: float = 1.0,
        reverse: bool = False,
    ) -> list[NoteEvent]:
        """Return transformed note events ready for placement in a score."""
        if time_scale <= 0:
            raise ValueError("time_scale must be positive")
        if amp_scale <= 0:
            raise ValueError("amp_scale must be positive")

        transformed_events: list[NoteEvent] = []
        phrase_duration = self.duration

        for event in self.events:
            scaled_start = event.start * time_scale
            scaled_duration = event.duration * time_scale
            if reverse:
                placed_start = start + ((phrase_duration - event.start - event.duration) * time_scale)
            else:
                placed_start = start + scaled_start

            new_partial = None if event.partial is None else event.partial + partial_shift
            transformed_events.append(
                replace(
                    event,
                    start=placed_start,
                    duration=scaled_duration,
                    amp=event.amp * amp_scale,
                    partial=new_partial,
                )
            )

        return transformed_events


@dataclass
class Voice:
    """Named collection of note events with shared synth/effect defaults."""

    name: str
    synth_defaults: dict[str, Any] = field(default_factory=dict)
    effects: list[EffectSpec] = field(default_factory=list)
    notes: list[NoteEvent] = field(default_factory=list)


@dataclass
class Score:
    """Top-level composition model and renderer."""

    f0: float
    sample_rate: int = synth.SAMPLE_RATE
    master_effects: list[EffectSpec] = field(default_factory=list)
    voices: dict[str, Voice] = field(default_factory=dict)

    def add_voice(
        self,
        name: str,
        *,
        synth_defaults: dict[str, Any] | None = None,
        effects: list[EffectSpec] | None = None,
    ) -> Voice:
        """Add or replace a named voice definition."""
        voice = Voice(
            name=name,
            synth_defaults=dict(synth_defaults or {}),
            effects=list(effects or []),
        )
        self.voices[name] = voice
        return voice

    def get_voice(self, name: str) -> Voice:
        """Get or create a voice with no defaults."""
        if name not in self.voices:
            self.voices[name] = Voice(name=name)
        return self.voices[name]

    def add_note(
        self,
        voice_name: str,
        *,
        start: float,
        duration: float,
        partial: float | None = None,
        freq: float | None = None,
        amp: float = 1.0,
        synth: dict[str, Any] | None = None,
        label: str | None = None,
    ) -> NoteEvent:
        """Add a single note event to a voice."""
        note = NoteEvent(
            start=start,
            duration=duration,
            partial=partial,
            freq=freq,
            amp=amp,
            synth=dict(synth) if synth is not None else None,
            label=label,
        )
        self.get_voice(voice_name).notes.append(note)
        return note

    def add_phrase(
        self,
        voice_name: str,
        phrase: Phrase,
        *,
        start: float,
        time_scale: float = 1.0,
        partial_shift: float = 0.0,
        amp_scale: float = 1.0,
        reverse: bool = False,
    ) -> list[NoteEvent]:
        """Place a phrase on a voice with optional transforms."""
        placed_notes = phrase.transformed(
            start=start,
            time_scale=time_scale,
            partial_shift=partial_shift,
            amp_scale=amp_scale,
            reverse=reverse,
        )
        voice = self.get_voice(voice_name)
        voice.notes.extend(placed_notes)
        return placed_notes

    @property
    def total_dur(self) -> float:
        """Compute total score duration from note endpoints."""
        endpoints = [
            note.start + note.duration
            for voice in self.voices.values()
            for note in voice.notes
        ]
        return max(endpoints, default=0.0)

    def render(self) -> np.ndarray:
        """Render the score to mono or stereo audio."""
        rendered_voices: list[np.ndarray] = []
        for voice in self.voices.values():
            rendered_voice = self._render_voice(voice)
            if rendered_voice.size > 0:
                rendered_voices.append(rendered_voice)

        if not rendered_voices:
            return np.zeros(0)

        mix = self._stack_signals(rendered_voices)
        if self.master_effects:
            if mix.ndim != 1:
                raise ValueError("master effect chains currently expect mono input")
            mix = synth.apply_effect_chain(mix, self.master_effects)
        return mix

    def plot_piano_roll(self, path: str | Path | None = None) -> tuple[Any, Any]:
        """Plot score events as a piano-roll style visualization."""
        figure, axis = plt.subplots(figsize=(12, 5))

        voice_names = list(self.voices)
        for row_index, voice_name in enumerate(voice_names):
            voice = self.voices[voice_name]
            base_y = row_index * 24
            for note in sorted(voice.notes, key=lambda item: item.start):
                pitch_value = note.partial if note.partial is not None else note.freq / self.f0
                axis.broken_barh(
                    [(note.start, note.duration)],
                    (base_y + pitch_value, 0.8),
                    facecolors=f"C{row_index % 10}",
                    alpha=0.8,
                )

        axis.set_xlabel("Time (seconds)")
        axis.set_ylabel("Voice / partial")
        axis.set_title("Score Piano Roll")
        axis.set_yticks([index * 24 + 7 for index in range(len(voice_names))])
        axis.set_yticklabels(voice_names)
        axis.grid(True, axis="x", alpha=0.3)

        if path is not None:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path, bbox_inches="tight")

        return figure, axis

    def _render_voice(self, voice: Voice) -> np.ndarray:
        voice_signals: list[np.ndarray] = []
        for note in voice.notes:
            synth_params = dict(voice.synth_defaults)
            if note.synth is not None:
                synth_params.update(note.synth)
            synth_params = resolve_synth_params(synth_params)

            note_signal = render_note_signal(
                freq=self._resolve_freq(note),
                duration=note.duration,
                amp=note.amp,
                sample_rate=self.sample_rate,
                params=synth_params,
            )
            note_signal = synth.adsr(
                note_signal,
                attack=synth_params.get("attack", 0.04),
                decay=synth_params.get("decay", 0.1),
                sustain_level=synth_params.get("sustain_level", 0.75),
                release=synth_params.get("release", 0.3),
                sample_rate=self.sample_rate,
            )
            voice_signals.append(synth.at_sample_rate(note_signal, note.start, self.sample_rate))

        if not voice_signals:
            return np.zeros(0)

        voice_mix = self._stack_signals(voice_signals)
        if voice.effects:
            if voice_mix.ndim != 1:
                raise ValueError("voice effect chains currently expect mono input")
            voice_mix = synth.apply_effect_chain(voice_mix, voice.effects)
        return voice_mix

    def _resolve_freq(self, note: NoteEvent) -> float:
        if note.freq is not None:
            return note.freq
        if note.partial is None:
            raise ValueError("note must provide partial or freq")
        return self.f0 * note.partial

    @staticmethod
    def _stack_signals(signals: list[np.ndarray]) -> np.ndarray:
        max_len = max(signal.shape[-1] for signal in signals)
        first_signal = signals[0]
        if first_signal.ndim == 1:
            output = np.zeros(max_len)
            for signal in signals:
                output[: signal.shape[-1]] += signal
            return output

        channels = first_signal.shape[0]
        output = np.zeros((channels, max_len))
        for signal in signals:
            output[:, : signal.shape[-1]] += signal
        return output
