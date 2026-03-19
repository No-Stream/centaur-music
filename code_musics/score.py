"""Score-domain abstractions for composing pieces."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from code_musics import synth
from code_musics.automation import (
    AutomationSpec,
    apply_synth_automation,
    build_pitch_ratio_trajectory,
    has_pitch_ratio_automation,
)
from code_musics.engines import render_note_signal, resolve_synth_params
from code_musics.humanize import (
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    TimingTarget,
    VelocityHumanizeSpec,
    VelocityTarget,
    build_timing_offsets,
    build_velocity_multipliers,
    resolve_envelope_params,
)
from code_musics.pitch_motion import PitchMotionSpec, build_frequency_trajectory


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
    amp: float | None = None
    amp_db: float | None = None
    velocity: float = 1.0
    partial: float | None = None
    freq: float | None = None
    synth: dict[str, Any] | None = None
    label: str | None = None
    pitch_motion: PitchMotionSpec | None = None
    automation: list[AutomationSpec] | None = None

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if (self.partial is None) == (self.freq is None):
            raise ValueError("exactly one of partial or freq must be provided")
        if self.amp is not None and self.amp_db is not None:
            raise ValueError("provide amp or amp_db, not both")
        if self.velocity <= 0 or self.velocity > 2.0:
            raise ValueError("velocity must be in the range (0, 2]")

        resolved_amp = self.amp
        if self.amp_db is not None:
            resolved_amp = synth.db_to_amp(self.amp_db)
        elif resolved_amp is None:
            resolved_amp = 1.0

        if resolved_amp <= 0:
            raise ValueError("amp must be positive")

        object.__setattr__(self, "amp", resolved_amp)


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
        amp: float | None = None,
        amp_db: float | None = None,
        velocity: float = 1.0,
        synth_defaults: dict[str, Any] | None = None,
    ) -> Phrase:
        """Create a phrase from equally spaced harmonic partials."""
        events = tuple(
            NoteEvent(
                start=index * step,
                duration=note_dur,
                amp=amp,
                amp_db=amp_db,
                velocity=velocity,
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
                placed_start = start + (
                    (phrase_duration - event.start - event.duration) * time_scale
                )
            else:
                placed_start = start + scaled_start

            resolved_amp = _require_resolved_amp(event)
            new_partial = (
                None if event.partial is None else event.partial + partial_shift
            )
            transformed_events.append(
                replace(
                    event,
                    start=placed_start,
                    duration=scaled_duration,
                    amp=resolved_amp * amp_scale,
                    partial=new_partial,
                )
            )

        return transformed_events


@dataclass(frozen=True)
class VelocityParamMap:
    """Linear velocity-to-parameter mapping."""

    min_value: float
    max_value: float
    min_velocity: float = 0.75
    max_velocity: float = 1.25

    def __post_init__(self) -> None:
        if self.min_velocity <= 0:
            raise ValueError("min_velocity must be positive")
        if self.max_velocity <= 0:
            raise ValueError("max_velocity must be positive")
        if self.min_velocity >= self.max_velocity:
            raise ValueError("min_velocity must be < max_velocity")

    def resolve(self, velocity: float) -> float:
        bounded_velocity = float(
            np.clip(velocity, self.min_velocity, self.max_velocity)
        )
        return float(
            np.interp(
                bounded_velocity,
                [self.min_velocity, self.max_velocity],
                [self.min_value, self.max_value],
            )
        )


@dataclass
class Voice:
    """Named collection of note events with shared synth/effect defaults."""

    name: str
    synth_defaults: dict[str, Any] = field(default_factory=dict)
    effects: list[EffectSpec] = field(default_factory=list)
    envelope_humanize: EnvelopeHumanizeSpec | None = None
    velocity_humanize: VelocityHumanizeSpec | None = field(
        default_factory=VelocityHumanizeSpec
    )
    velocity_group: str | None = None
    velocity_to_params: dict[str, VelocityParamMap] = field(default_factory=dict)
    velocity_db_per_unit: float = 12.0
    normalize_lufs: float | None = -24.0
    pan: float = 0.0
    automation: list[AutomationSpec] = field(default_factory=list)
    notes: list[NoteEvent] = field(default_factory=list)


@dataclass(frozen=True)
class ResolvedTimingNote:
    """Note timing snapshot after score-level timing humanization."""

    key: tuple[str, int]
    voice_name: str
    note_index: int
    authored_start: float
    resolved_start: float
    timing_offset_seconds: float
    duration: float
    resolved_end: float
    freq_hz: float
    partial: float | None
    label: str | None


@dataclass
class Score:
    """Top-level composition model and renderer."""

    f0: float
    sample_rate: int = synth.SAMPLE_RATE
    timing_humanize: TimingHumanizeSpec | None = None
    master_effects: list[EffectSpec] = field(default_factory=list)
    voices: dict[str, Voice] = field(default_factory=dict)

    def add_voice(
        self,
        name: str,
        *,
        synth_defaults: dict[str, Any] | None = None,
        effects: list[EffectSpec] | None = None,
        envelope_humanize: EnvelopeHumanizeSpec | None = None,
        velocity_humanize: VelocityHumanizeSpec | None = None,
        velocity_group: str | None = None,
        velocity_to_params: dict[str, VelocityParamMap] | None = None,
        velocity_db_per_unit: float = 12.0,
        normalize_lufs: float | None = -24.0,
        pan: float = 0.0,
        automation: list[AutomationSpec] | None = None,
    ) -> Voice:
        """Add or replace a named voice definition."""
        if not -1.0 <= pan <= 1.0:
            raise ValueError("pan must be between -1 and 1")
        if velocity_db_per_unit < 0:
            raise ValueError("velocity_db_per_unit must be non-negative")
        if normalize_lufs is not None and not np.isfinite(normalize_lufs):
            raise ValueError("normalize_lufs must be finite when provided")
        voice = Voice(
            name=name,
            synth_defaults=dict(synth_defaults or {}),
            effects=list(effects or []),
            envelope_humanize=envelope_humanize,
            velocity_humanize=(
                VelocityHumanizeSpec()
                if velocity_humanize is None
                else velocity_humanize
            ),
            velocity_group=velocity_group,
            velocity_to_params=dict(velocity_to_params or {}),
            velocity_db_per_unit=velocity_db_per_unit,
            normalize_lufs=normalize_lufs,
            pan=pan,
            automation=list(automation or []),
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
        amp: float | None = None,
        amp_db: float | None = None,
        velocity: float = 1.0,
        pitch_motion: PitchMotionSpec | None = None,
        synth: dict[str, Any] | None = None,
        label: str | None = None,
        automation: list[AutomationSpec] | None = None,
    ) -> NoteEvent:
        """Add a single note event to a voice."""
        note = NoteEvent(
            start=start,
            duration=duration,
            partial=partial,
            freq=freq,
            amp=amp,
            amp_db=amp_db,
            velocity=velocity,
            synth=dict(synth) if synth is not None else None,
            label=label,
            pitch_motion=pitch_motion,
            automation=list(automation) if automation is not None else None,
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
        rendered_voices = list(self.render_stems().values())

        if not rendered_voices:
            return np.zeros(0)

        mix = self._stack_signals(rendered_voices)
        if self.master_effects:
            mix = synth.apply_effect_chain(mix, self.master_effects)

        # Transparent output ceiling: only reduces gain if the signal exceeds
        # -0.5 dBFS (~0.944), so normal mixes are untouched but accidental
        # overloads cannot hard-clip the final output.
        ceiling = 0.944
        peak = float(np.max(np.abs(mix))) if mix.size > 0 else 0.0
        if peak > ceiling:
            mix = mix * (ceiling / peak)

        return mix

    def render_stems(self) -> dict[str, np.ndarray]:
        """Render each voice independently before master-bus effects."""
        rendered_stems: dict[str, np.ndarray] = {}
        timing_offsets = self.resolve_timing_offsets()
        velocity_multipliers = self._build_velocity_multiplier_map()
        for voice_name, voice in self.voices.items():
            rendered_voice = self._render_voice(
                voice_name=voice_name,
                voice=voice,
                timing_offsets=timing_offsets,
                velocity_multipliers=velocity_multipliers,
            )
            if rendered_voice.size > 0:
                rendered_stems[voice_name] = rendered_voice
        return rendered_stems

    def resolve_timing_offsets(self) -> dict[tuple[str, int], float]:
        """Return the deterministic render-time timing offset for each note."""
        return build_timing_offsets(
            targets=self._timing_targets(),
            humanize=self.timing_humanize,
            total_dur=self.total_dur,
        )

    def resolved_timing_notes(self) -> list[ResolvedTimingNote]:
        """Return note timing data after score-level timing humanization."""
        timing_offsets = self.resolve_timing_offsets()
        resolved_notes: list[ResolvedTimingNote] = []
        for voice_name, voice in self.voices.items():
            for note_index, note in enumerate(voice.notes):
                resolved_start = max(
                    0.0,
                    note.start + timing_offsets.get((voice_name, note_index), 0.0),
                )
                resolved_notes.append(
                    ResolvedTimingNote(
                        key=(voice_name, note_index),
                        voice_name=voice_name,
                        note_index=note_index,
                        authored_start=note.start,
                        resolved_start=resolved_start,
                        timing_offset_seconds=resolved_start - note.start,
                        duration=note.duration,
                        resolved_end=resolved_start + note.duration,
                        freq_hz=self._resolve_freq(note),
                        partial=note.partial,
                        label=note.label,
                    )
                )
        return resolved_notes

    def plot_piano_roll(self, path: str | Path | None = None) -> tuple[Any, Any]:
        """Plot score events as a piano-roll style visualization."""
        figure, axis = plt.subplots(figsize=(12, 5))

        voice_names = list(self.voices)
        for row_index, voice_name in enumerate(voice_names):
            voice = self.voices[voice_name]
            base_y = row_index * 24
            for note in sorted(voice.notes, key=lambda item: item.start):
                if note.partial is not None:
                    pitch_value = float(note.partial)
                else:
                    if note.freq is None:
                        raise ValueError("note must define partial or freq")
                    pitch_value = float(note.freq / self.f0)
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

    def _render_voice(
        self,
        *,
        voice_name: str,
        voice: Voice,
        timing_offsets: dict[tuple[str, int], float],
        velocity_multipliers: dict[tuple[str, int], float],
    ) -> np.ndarray:
        voice_signals: list[np.ndarray] = []
        for note_index, note in enumerate(voice.notes):
            synth_params = dict(voice.synth_defaults)
            if note.synth is not None:
                synth_params.update(note.synth)
            synth_params = resolve_synth_params(synth_params)
            resolved_velocity = float(
                np.clip(
                    note.velocity
                    * velocity_multipliers.get((voice_name, note_index), 1.0),
                    0.05,
                    2.0,
                )
            )
            synth_params.update(
                {
                    param_name: velocity_map.resolve(resolved_velocity)
                    for param_name, velocity_map in voice.velocity_to_params.items()
                }
            )
            note_automation = list(note.automation or [])
            synth_params = apply_synth_automation(
                params=synth_params,
                voice_automation=voice.automation,
                note_automation=note_automation,
                note_start=note.start,
            )
            attack_scale = float(synth_params.pop("attack_scale", 1.0))
            release_scale = float(synth_params.pop("release_scale", 1.0))
            note_freq = self._resolve_freq(note)
            humanized_start = max(
                0.0,
                note.start + timing_offsets.get((voice_name, note_index), 0.0),
            )
            attack, decay, sustain_level, release = resolve_envelope_params(
                base_attack=float(synth_params.get("attack", 0.04)) * attack_scale,
                base_decay=float(synth_params.get("decay", 0.1)),
                base_sustain_level=float(synth_params.get("sustain_level", 0.75)),
                base_release=float(synth_params.get("release", 0.3)) * release_scale,
                note_start=note.start,
                humanize=voice.envelope_humanize,
                total_dur=self.total_dur,
                voice_name=voice_name,
            )
            held_samples = int(note.duration * self.sample_rate)
            total_samples = int((note.duration + release) * self.sample_rate)
            freq_trajectory = None
            if note.pitch_motion is not None and (
                has_pitch_ratio_automation(voice.automation)
                or has_pitch_ratio_automation(note_automation)
            ):
                raise ValueError(
                    "pitch_ratio automation cannot be combined with pitch_motion on the same note"
                )
            if note.pitch_motion is not None:
                held_trajectory = build_frequency_trajectory(
                    base_freq=note_freq,
                    duration=note.duration,
                    sample_rate=self.sample_rate,
                    motion=note.pitch_motion,
                    score_f0=self.f0,
                )
                release_samples = max(0, total_samples - held_samples)
                if release_samples > 0:
                    release_tail = np.full(release_samples, held_trajectory[-1])
                    freq_trajectory = np.concatenate([held_trajectory, release_tail])
                else:
                    freq_trajectory = held_trajectory
            else:
                held_trajectory = build_pitch_ratio_trajectory(
                    base_freq=note_freq,
                    duration=note.duration,
                    sample_rate=self.sample_rate,
                    voice_automation=voice.automation,
                    note_automation=note_automation,
                    note_start=note.start,
                )
                if held_trajectory is not None:
                    release_samples = max(0, total_samples - held_samples)
                    if release_samples > 0:
                        release_tail = np.full(release_samples, held_trajectory[-1])
                        freq_trajectory = np.concatenate(
                            [held_trajectory, release_tail]
                        )
                    else:
                        freq_trajectory = held_trajectory

            note_signal = render_note_signal(
                freq=note_freq,
                duration=note.duration + release,
                amp=self._resolve_note_amp(
                    note=note,
                    resolved_velocity=resolved_velocity,
                    velocity_db_per_unit=voice.velocity_db_per_unit,
                ),
                sample_rate=self.sample_rate,
                params=synth_params,
                freq_trajectory=freq_trajectory,
            )
            note_signal = synth.adsr(
                note_signal,
                attack=attack,
                decay=decay,
                sustain_level=sustain_level,
                release=release,
                sample_rate=self.sample_rate,
                hold_duration=note.duration,
            )
            voice_signals.append(
                synth.at_sample_rate(note_signal, humanized_start, self.sample_rate)
            )

        if not voice_signals:
            return np.zeros(0)

        voice_mix = self._stack_signals(voice_signals)
        if voice.normalize_lufs is not None:
            voice_mix = synth.normalize_to_lufs(
                voice_mix,
                sample_rate=self.sample_rate,
                target_lufs=voice.normalize_lufs,
            )
        if voice.pan != 0.0:
            voice_mix = synth.apply_pan(voice_mix, pan=voice.pan)
        if voice.effects:
            voice_mix = synth.apply_effect_chain(voice_mix, voice.effects)
        return voice_mix

    def _timing_targets(self) -> list[TimingTarget]:
        targets: list[TimingTarget] = []
        for voice_name, voice in self.voices.items():
            for note_index, note in enumerate(voice.notes):
                targets.append(
                    TimingTarget(
                        key=(voice_name, note_index),
                        voice_name=voice_name,
                        start=note.start,
                    )
                )
        return targets

    def _velocity_targets(
        self,
    ) -> dict[VelocityHumanizeSpec | None, list[VelocityTarget]]:
        targets: dict[VelocityHumanizeSpec | None, list[VelocityTarget]] = {}
        for voice_name, voice in self.voices.items():
            group_name = voice.velocity_group or voice_name
            for note_index, note in enumerate(voice.notes):
                targets.setdefault(voice.velocity_humanize, []).append(
                    VelocityTarget(
                        key=(voice_name, note_index),
                        voice_name=voice_name,
                        group_name=group_name,
                        start=note.start,
                    )
                )
        return targets

    def _resolve_freq(self, note: NoteEvent) -> float:
        if note.freq is not None:
            return note.freq
        if note.partial is None:
            raise ValueError("note must provide partial or freq")
        return self.f0 * note.partial

    def _build_velocity_multiplier_map(self) -> dict[tuple[str, int], float]:
        velocity_multipliers: dict[tuple[str, int], float] = {}
        for humanize, targets in self._velocity_targets().items():
            velocity_multipliers.update(
                build_velocity_multipliers(
                    targets=targets,
                    humanize=humanize,
                    total_dur=self.total_dur,
                )
            )
        return velocity_multipliers

    @staticmethod
    def _resolve_note_amp(
        *,
        note: NoteEvent,
        resolved_velocity: float,
        velocity_db_per_unit: float,
    ) -> float:
        base_amp_db = (
            note.amp_db
            if note.amp_db is not None
            else synth.amp_to_db(_require_resolved_amp(note))
        )
        velocity_db_offset = (resolved_velocity - 1.0) * velocity_db_per_unit
        return synth.db_to_amp(base_amp_db + velocity_db_offset)

    @staticmethod
    def _stack_signals(signals: list[np.ndarray]) -> np.ndarray:
        max_len = max(signal.shape[-1] for signal in signals)
        if all(signal.ndim == 1 for signal in signals):
            output = np.zeros(max_len)
            for signal in signals:
                output[: signal.shape[-1]] += signal
            return output

        channels = 2
        output = np.zeros((channels, max_len))
        for signal in signals:
            if signal.ndim == 1:
                output[0, : signal.shape[-1]] += signal
                output[1, : signal.shape[-1]] += signal
            else:
                output[:, : signal.shape[-1]] += signal
        return output


def _require_resolved_amp(event: NoteEvent) -> float:
    """Return a concrete amplitude for an event after NoteEvent validation."""
    if event.amp is None:
        raise ValueError("event amp unexpectedly missing")
    return float(event.amp)
