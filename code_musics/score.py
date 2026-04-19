"""Score-domain abstractions for composing pieces."""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np

from code_musics import synth
from code_musics.automation import (
    AutomationSpec,
    apply_control_automation,
    apply_synth_automation,
    build_pitch_ratio_trajectory,
    has_pitch_ratio_automation,
)
from code_musics.engines import (
    is_instrument_engine,
    normalize_synth_spec,
    render_note_signal,
    resolve_synth_params,
)
from code_musics.engines._dsp_utils import (
    extract_analog_params,
    voice_card_offsets,
)
from code_musics.humanize import (
    DriftBusSpec,
    EnvelopeHumanizeSpec,
    TimingHumanizeSpec,
    TimingTarget,
    VelocityHumanizeSpec,
    VelocityTarget,
    build_drift_bus,
    build_timing_offsets,
    build_velocity_multipliers,
    resolve_envelope_params,
)
from code_musics.modulation import (
    MacroDefinition,
    ModConnection,
    SourceSamplingContext,
    build_macro_lookup,
    combine_connections_on_curve,
    combine_connections_scalar,
    is_per_sample_synth_destination,
    iter_connections_for_target,
)
from code_musics.pitch_motion import PitchMotionSpec, build_frequency_trajectory

logger: logging.Logger = logging.getLogger(__name__)

_OUTPUT_CEILING_DBFS: float = -0.5


def _apply_voice_card_env_rate_scaling(
    *,
    attack: float,
    release: float,
    synth_params: dict[str, Any],
    voice_name: str,
) -> tuple[float, float]:
    """Apply OB-Xd-style multiplicative per-voice envelope-rate scaling.

    Pulls ``voice_card_spread`` (with an optional
    ``voice_card_envelope_spread`` override) out of ``synth_params`` and
    uses the deterministic per-voice ``attack_scale`` / ``release_scale``
    offsets to nudge the outer ADSR times.  This is the "ensemble of slightly
    different voice cards" flavour of analog warmth — every voice has its own
    permanent env-rate bias.

    Opt-in: only fires when ``voice_card_spread``, ``voice_card``, or the
    envelope-specific override ``voice_card_envelope_spread`` is explicitly
    present in ``synth_params``.  Voices that never touch these knobs keep
    their exact attack/release times (preserving fixed-length render semantics
    for score tests and older pieces).
    """
    if not voice_name:
        return attack, release
    opted_in = (
        "voice_card_spread" in synth_params
        or "voice_card" in synth_params
        or "voice_card_envelope_spread" in synth_params
    )
    if not opted_in:
        return attack, release
    analog = extract_analog_params(synth_params)
    env_spread = analog["voice_card_envelope_spread"]
    if env_spread <= 0.0:
        return attack, release
    vc = voice_card_offsets(voice_name)
    attack_factor = 1.0 + (vc["attack_scale"] - 1.0) * env_spread
    release_factor = 1.0 + (vc["release_scale"] - 1.0) * env_spread
    return attack * attack_factor, release * release_factor


EffectKind = Literal[
    "gate",
    "delay",
    "reverb",
    "chow_tape",
    "bricasti",
    "brit_pre",
    "chorus",
    "bbd_chorus",
    "mod_delay",
    "saturation",
    "preamp",
    "eq",
    "compressor",
    "phaser",
    "tal_chorus_lx",
    "tal_reverb2",
    "dragonfly",
    "mjuc_jr",
    "plugin",
]

_UNSET: object = object()  # sentinel distinguishing "not passed" from explicit None


@dataclass(frozen=True)
class EffectSpec:
    """Declarative effect-chain item."""

    kind: EffectKind
    params: dict[str, Any] = field(default_factory=dict)
    automation: list[AutomationSpec] = field(default_factory=list)


@dataclass(frozen=True)
class VoiceSend:
    """Post-fader routing from a voice into a named shared send bus."""

    target: str
    send_db: float = 0.0
    automation: list[AutomationSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError("voice send target must be non-empty")
        if not np.isfinite(self.send_db):
            raise ValueError("voice send_db must be finite")


@dataclass(frozen=True)
class SendBusSpec:
    """Shared aux return bus fed by one or more voice sends."""

    name: str
    effects: list[EffectSpec] = field(default_factory=list)
    return_db: float = 0.0
    pan: float = 0.0
    automation: list[AutomationSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("send bus name must be non-empty")
        if not np.isfinite(self.return_db):
            raise ValueError("send bus return_db must be finite")
        if not -1.0 <= self.pan <= 1.0:
            raise ValueError("send bus pan must be between -1 and 1")


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
    automation: list[AutomationSpec] = field(default_factory=list)

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
class BeatTiming:
    """Beat-domain timing metadata for grid-authored phrases."""

    start_beats: float
    duration_beats: float

    def __post_init__(self) -> None:
        if self.start_beats < 0:
            raise ValueError("start_beats must be non-negative")
        if self.duration_beats <= 0:
            raise ValueError("duration_beats must be positive")


@dataclass(frozen=True)
class Phrase:
    """Reusable collection of relative-time note events."""

    events: tuple[NoteEvent, ...]
    beat_timings: tuple[BeatTiming, ...] | None = None

    def __post_init__(self) -> None:
        if self.beat_timings is not None and len(self.beat_timings) != len(self.events):
            raise ValueError("beat_timings length must match events length")

    @classmethod
    def from_partials(
        cls,
        partials: list[float],
        duration: float | None = None,
        onset_interval: float | None = None,
        *,
        amp: float | None = None,
        amp_db: float | None = None,
        velocity: float = 1.0,
        synth_defaults: dict[str, Any] | None = None,
        # deprecated aliases
        note_dur: float | None = None,
        step: float | None = None,
    ) -> Phrase:
        """Create a phrase from equally spaced harmonic partials.

        Parameters
        ----------
        duration : float
            Duration of each note event in seconds.
        onset_interval : float
            Time between successive note onsets in seconds.
        note_dur : float, optional
            **Deprecated** — use *duration* instead.
        step : float, optional
            **Deprecated** — use *onset_interval* instead.
        """
        # Handle deprecated aliases ------------------------------------------------
        if note_dur is not None:
            if duration is not None:
                raise ValueError("Cannot specify both 'note_dur' and 'duration'")
            warnings.warn(
                "Phrase.from_partials: 'note_dur' is deprecated, use 'duration'",
                DeprecationWarning,
                stacklevel=2,
            )
            duration = note_dur
        if step is not None:
            if onset_interval is not None:
                raise ValueError("Cannot specify both 'step' and 'onset_interval'")
            warnings.warn(
                "Phrase.from_partials: 'step' is deprecated, use 'onset_interval'",
                DeprecationWarning,
                stacklevel=2,
            )
            onset_interval = step
        if duration is None:
            raise TypeError("from_partials() requires 'duration'")
        if onset_interval is None:
            raise TypeError("from_partials() requires 'onset_interval'")
        # --------------------------------------------------------------------------
        events = tuple(
            NoteEvent(
                start=index * onset_interval,
                duration=duration,
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
        freq_scale: float = 1.0,
        amp_scale: float = 1.0,
        reverse: bool = False,
    ) -> list[NoteEvent]:
        """Return transformed note events ready for placement in a score."""
        if time_scale <= 0:
            raise ValueError("time_scale must be positive")
        if amp_scale <= 0:
            raise ValueError("amp_scale must be positive")
        if freq_scale <= 0:
            raise ValueError("freq_scale must be positive")

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
            new_freq = event.freq
            if new_freq is not None and freq_scale != 1.0:
                new_freq *= freq_scale
            transformed_events.append(
                replace(
                    event,
                    start=placed_start,
                    duration=scaled_duration,
                    amp=resolved_amp * amp_scale,
                    amp_db=None,
                    partial=new_partial,
                    freq=new_freq,
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
    pre_fx_gain_db: float = 0.0
    mix_db: float = 0.0
    sends: list[VoiceSend] = field(default_factory=list)
    normalize_lufs: float | None = -24.0
    normalize_peak_db: float | None = None
    max_polyphony: int | None = None
    legato: bool = False
    choke_group: str | None = None
    pan: float = 0.0
    sympathetic_amount: float = 0.0
    sympathetic_decay_s: float = 2.0
    sympathetic_modes: int = 8
    drift_bus: str | None = None
    drift_bus_correlation: float = 1.0
    automation: list[AutomationSpec] = field(default_factory=list)
    modulations: list[ModConnection] = field(default_factory=list)
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
class PreparedVoiceNote:
    """Resolved note render state before voice-level mixing."""

    note_index: int
    note: NoteEvent
    synth_params: dict[str, Any]
    resolved_velocity: float
    note_freq: float
    humanized_start: float
    attack: float
    decay: float
    sustain_level: float
    release: float
    freq_trajectory: np.ndarray | None
    effective_hold_duration: float
    effective_attack: float
    effective_release: float
    vca_nonlinearity: float
    attack_power: float = 1.0
    decay_power: float = 1.0
    release_power: float = 1.0
    attack_target: float = 1.0
    param_profiles: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Score:
    """Top-level composition model and renderer."""

    f0_hz: float
    sample_rate: int = synth.SAMPLE_RATE
    timing_humanize: TimingHumanizeSpec | None = None
    auto_master_gain_stage: bool = True
    master_bus_target_lufs: float = -24.0
    master_bus_max_true_peak_dbfs: float = -6.0
    master_input_gain_db: float = 0.0
    master_effects: list[EffectSpec] = field(default_factory=list)
    send_buses: list[SendBusSpec] = field(default_factory=list)
    drift_buses: dict[str, DriftBusSpec] = field(default_factory=dict)
    macros: dict[str, MacroDefinition] = field(default_factory=dict)
    modulations: list[ModConnection] = field(default_factory=list)
    voices: dict[str, Voice] = field(default_factory=dict)
    time_origin_seconds: float = 0.0
    time_reference_total_dur: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.master_bus_target_lufs):
            raise ValueError("master_bus_target_lufs must be finite")
        if not np.isfinite(self.master_bus_max_true_peak_dbfs):
            raise ValueError("master_bus_max_true_peak_dbfs must be finite")
        if not np.isfinite(self.master_input_gain_db):
            raise ValueError("master_input_gain_db must be finite")
        self._validate_send_buses()
        for voice in self.voices.values():
            self._validate_voice_sends(voice.sends)
        # Lazy per-render cache of full-score drift bus trajectories, keyed by
        # (name, seed, rate_hz, total_dur).  Same inputs -> same array, so it
        # is safe to keep across renders.
        self._drift_bus_cache: dict[
            tuple[str, int | None, float, float], tuple[np.ndarray, np.ndarray]
        ] = {}
        # Memoized macro lookup tables, keyed by (n_samples, first_time).
        # Cleared implicitly whenever a render pass uses a new time grid.
        self._macro_lookup_cache: dict[tuple[int, float], dict[str, np.ndarray]] = {}

    def add_voice(
        self,
        name: str,
        *,
        synth_defaults: dict[str, Any] | None = None,
        effects: list[EffectSpec] | None = None,
        envelope_humanize: EnvelopeHumanizeSpec | None = None,
        velocity_humanize: VelocityHumanizeSpec | None = _UNSET,  # type: ignore[assignment]
        velocity_group: str | None = None,
        velocity_to_params: dict[str, VelocityParamMap] | None = None,
        velocity_db_per_unit: float = 12.0,
        pre_fx_gain_db: float = 0.0,
        mix_db: float = 0.0,
        sends: list[VoiceSend] | None = None,
        normalize_lufs: float | None = _UNSET,  # type: ignore[assignment]
        normalize_peak_db: float | None = None,
        max_polyphony: int | None = None,
        legato: bool = False,
        choke_group: str | None = None,
        pan: float = 0.0,
        sympathetic_amount: float = 0.0,
        sympathetic_decay_s: float = 2.0,
        sympathetic_modes: int = 8,
        drift_bus: str | None = None,
        drift_bus_correlation: float = 1.0,
        automation: list[AutomationSpec] | None = None,
        modulations: list[ModConnection] | None = None,
    ) -> Voice:
        """Add or replace a named voice definition."""
        # Resolve normalize_lufs default: -24.0 unless normalize_peak_db is given,
        # in which case default to None (caller doesn't need to explicitly clear it).
        if normalize_lufs is _UNSET:
            normalize_lufs = None if normalize_peak_db is not None else -24.0
        if not -1.0 <= pan <= 1.0:
            raise ValueError("pan must be between -1 and 1")
        if velocity_db_per_unit < 0:
            raise ValueError("velocity_db_per_unit must be non-negative")
        if not np.isfinite(pre_fx_gain_db):
            raise ValueError("pre_fx_gain_db must be finite")
        if not np.isfinite(mix_db):
            raise ValueError("mix_db must be finite")
        if normalize_lufs is not None and normalize_peak_db is not None:
            raise ValueError(
                "normalize_lufs and normalize_peak_db cannot both be set; "
                "choose LUFS normalization for tonal voices or peak normalization "
                "for percussive voices"
            )
        if normalize_lufs is not None and not np.isfinite(normalize_lufs):
            raise ValueError("normalize_lufs must be finite when provided")
        if normalize_peak_db is not None and not np.isfinite(normalize_peak_db):
            raise ValueError("normalize_peak_db must be finite when provided")
        if max_polyphony is not None and max_polyphony < 1:
            raise ValueError("max_polyphony must be >= 1 when provided")
        if not 0.0 <= drift_bus_correlation <= 1.0:
            raise ValueError("drift_bus_correlation must be between 0 and 1")
        if velocity_humanize is _UNSET:
            velocity_humanize = VelocityHumanizeSpec()
        resolved_sends = list(sends or [])
        self._validate_voice_sends(resolved_sends)
        voice = Voice(
            name=name,
            synth_defaults=dict(synth_defaults or {}),
            effects=list(effects or []),
            envelope_humanize=envelope_humanize,
            velocity_humanize=velocity_humanize,
            velocity_group=velocity_group,
            velocity_to_params=dict(velocity_to_params or {}),
            velocity_db_per_unit=velocity_db_per_unit,
            pre_fx_gain_db=pre_fx_gain_db,
            mix_db=mix_db,
            sends=resolved_sends,
            normalize_lufs=normalize_lufs,
            normalize_peak_db=normalize_peak_db,
            max_polyphony=max_polyphony,
            legato=legato,
            choke_group=choke_group,
            pan=pan,
            sympathetic_amount=sympathetic_amount,
            sympathetic_decay_s=sympathetic_decay_s,
            sympathetic_modes=sympathetic_modes,
            drift_bus=drift_bus,
            drift_bus_correlation=drift_bus_correlation,
            automation=list(automation or []),
            modulations=list(modulations or []),
        )
        self.voices[name] = voice
        return voice

    def add_send_bus(
        self,
        name: str,
        *,
        effects: list[EffectSpec] | None = None,
        return_db: float = 0.0,
        pan: float = 0.0,
        automation: list[AutomationSpec] | None = None,
    ) -> SendBusSpec:
        """Add or replace a named shared send bus definition."""
        send_bus = SendBusSpec(
            name=name,
            effects=list(effects or []),
            return_db=return_db,
            pan=pan,
            automation=list(automation or []),
        )
        existing_index = next(
            (index for index, bus in enumerate(self.send_buses) if bus.name == name),
            None,
        )
        if existing_index is None:
            self.send_buses.append(send_bus)
        else:
            self.send_buses[existing_index] = send_bus
        return send_bus

    def add_macro(
        self,
        name: str,
        *,
        default: float = 0.0,
        automation: AutomationSpec | None = None,
    ) -> MacroDefinition:
        """Register a named macro for use with :class:`MacroSource`.

        Macros are shared scalar values in ``[0, 1]`` that ``MacroSource``
        connections read via their name.  A macro can carry an
        :class:`AutomationSpec` (with ``target.kind == "control"``) so
        it rides across the piece timeline; without one the macro
        stays at ``default``.
        """
        macro = MacroDefinition(name=name, default=default, automation=automation)
        self.macros[name] = macro
        return macro

    def add_drift_bus(
        self,
        name: str,
        *,
        rate_hz: float = 0.2,
        depth_cents: float = 5.0,
        seed: int | None = None,
    ) -> DriftBusSpec:
        """Add or replace a named shared drift bus definition.

        Voices that set ``drift_bus=name`` receive a correlated slow pitch-drift
        signal on top of their independent per-voice drift.  The amount of
        correlation is controlled per voice via ``drift_bus_correlation``.
        """
        spec = DriftBusSpec(
            name=name,
            rate_hz=rate_hz,
            depth_cents=depth_cents,
            seed=seed,
        )
        self.drift_buses[name] = spec
        return spec

    def get_or_create_voice(self, name: str) -> Voice:
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
            automation=list(automation) if automation else [],
        )
        self.get_or_create_voice(voice_name).notes.append(note)
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
        synth: dict[str, Any] | None = None,
    ) -> list[NoteEvent]:
        """Place a phrase on a voice with optional transforms.

        When *synth* is provided, its entries are merged into each placed note's
        synth overrides as a base layer — note-level overrides win on conflict.
        """
        placed_notes = phrase.transformed(
            start=start,
            time_scale=time_scale,
            partial_shift=partial_shift,
            amp_scale=amp_scale,
            reverse=reverse,
        )
        if synth is not None:
            merged_notes: list[NoteEvent] = []
            for note in placed_notes:
                if note.synth is not None:
                    merged = {**synth, **note.synth}
                else:
                    merged = dict(synth)
                merged_notes.append(replace(note, synth=merged))
            placed_notes = merged_notes
        voice = self.get_or_create_voice(voice_name)
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
        stems, send_returns, _, _, _ = self._render_mix_components_internal(
            collect_effect_analysis=False
        )
        mix_inputs = [*stems.values(), *send_returns.values()]
        if not mix_inputs:
            return np.zeros(0)
        mix = self._stack_signals(mix_inputs)
        pre_peak = float(np.max(np.abs(mix))) if mix.size > 0 else 0.0
        pre_peak_db = 20.0 * np.log10(max(pre_peak, 1e-12))
        logger.info(f"Master bus pre-processing: peak {pre_peak_db:.2f} dBFS")
        return self._apply_master_bus_processing(mix)

    def render_with_effect_analysis(
        self,
        *,
        collect_effect_analysis: bool = True,
    ) -> tuple[
        np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]
    ]:
        """Render the score plus per-effect diagnostics for agents and analysis."""
        stems, send_returns, voice_effects, send_effects, _ = (
            self._render_mix_components_internal(
                collect_effect_analysis=collect_effect_analysis
            )
        )
        mix_inputs = [*stems.values(), *send_returns.values()]
        if not mix_inputs:
            return (
                np.zeros(0),
                {},
                {},
                {"mix_effects": [], "voice_effects": {}, "send_effects": {}},
            )

        mix = self._stack_signals(mix_inputs)
        mix_effects: list[synth.EffectAnalysisEntry] = []
        mix = self._apply_master_bus_processing(
            mix,
            collect_effect_analysis=collect_effect_analysis,
            mix_effects=mix_effects,
        )

        return (
            mix,
            stems,
            send_returns,
            {
                "mix_effects": [entry.to_dict() for entry in mix_effects],
                "voice_effects": {
                    voice_name: [entry.to_dict() for entry in entries]
                    for voice_name, entries in voice_effects.items()
                },
                "send_effects": {
                    bus_name: [entry.to_dict() for entry in entries]
                    for bus_name, entries in send_effects.items()
                },
            },
        )

    def extract_window(
        self,
        *,
        start_seconds: float,
        end_seconds: float,
    ) -> Score:
        """Return a score containing only notes audible within a time window."""
        if start_seconds < 0:
            raise ValueError("start_seconds must be non-negative")
        if end_seconds <= start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")

        shifted_voices: dict[str, Voice] = {}
        for voice_name, voice in self.voices.items():
            kept_notes = [
                replace(note, start=max(0.0, note.start - start_seconds), amp_db=None)
                for note in voice.notes
                if (note.start + note.duration) > start_seconds
                and note.start < end_seconds
            ]
            if not kept_notes:
                continue
            shifted_voices[voice_name] = replace(voice, notes=kept_notes)

        reference_total_dur = (
            self.time_reference_total_dur
            if self.time_reference_total_dur is not None
            else self.total_dur
        )
        return Score(
            f0_hz=self.f0_hz,
            sample_rate=self.sample_rate,
            timing_humanize=self.timing_humanize,
            auto_master_gain_stage=self.auto_master_gain_stage,
            master_bus_target_lufs=self.master_bus_target_lufs,
            master_bus_max_true_peak_dbfs=self.master_bus_max_true_peak_dbfs,
            master_input_gain_db=self.master_input_gain_db,
            master_effects=list(self.master_effects),
            send_buses=list(self.send_buses),
            voices=shifted_voices,
            time_origin_seconds=self.time_origin_seconds + start_seconds,
            time_reference_total_dur=reference_total_dur,
        )

    def render_stems(self) -> dict[str, np.ndarray]:
        """Render each voice independently before master-bus effects."""
        rendered_stems, _, _, _, _ = self._render_mix_components_internal(
            collect_effect_analysis=False
        )
        return rendered_stems

    def render_stems_with_effect_analysis(
        self,
    ) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, Any]]]]:
        """Render stems plus voice-level effect diagnostics."""
        rendered_stems, _, voice_effects, _, _ = self._render_mix_components_internal(
            collect_effect_analysis=True
        )
        return rendered_stems, {
            voice_name: [entry.to_dict() for entry in entries]
            for voice_name, entries in voice_effects.items()
        }

    def render_for_stem_export(
        self, *, dry: bool = False
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
        """Render all components needed for audio stem WAV export.

        Returns (voice_stems, send_returns, mix_audio) where:
        - voice_stems: wet (post-effects/pan/fader) or dry (post-normalization only)
        - send_returns: mixed bus returns (empty dict if dry=True)
        - mix_audio: always the full wet mix with master bus processing
        """
        wet_stems, send_returns, _, _, dry_stems = self._render_mix_components_internal(
            collect_effect_analysis=False
        )
        # Build the full wet mix for reference
        mix_inputs = [*wet_stems.values(), *send_returns.values()]
        if not mix_inputs:
            mix = np.zeros(0)
        else:
            mix = self._stack_signals(mix_inputs)
            mix = self._apply_master_bus_processing(mix)

        if dry:
            return dry_stems, {}, mix
        return wet_stems, send_returns, mix

    def _apply_choke_groups(
        self,
        rendered_voice_bases: dict[str, np.ndarray],
        timing_offsets: dict[tuple[str, int], float],
    ) -> None:
        """Apply choke-group fade-outs across voices that share a group.

        For each choke group, when a note starts in one voice, all other
        voices in the same group are faded to silence over 10 ms from that
        onset.  Modifies *rendered_voice_bases* in place.
        """
        choke_fade_ms = 10.0
        groups: dict[str, list[str]] = {}
        for voice_name, voice in self.voices.items():
            if voice.choke_group is not None:
                groups.setdefault(voice.choke_group, []).append(voice_name)

        for group_voices in groups.values():
            if len(group_voices) < 2:
                continue

            onsets: list[tuple[float, str]] = []
            for vn in group_voices:
                for i, note in enumerate(self.voices[vn].notes):
                    offset = timing_offsets.get((vn, i), 0.0)
                    onsets.append((max(0.0, note.start + offset), vn))
            onsets.sort(key=lambda x: x[0])

            fade_samples = max(1, int(choke_fade_ms / 1000.0 * self.sample_rate))

            # Pre-compute per-voice onset sample sets for finding the "silence
            # until next own-voice note" boundary after the fade.
            voice_onset_samples: dict[str, list[int]] = {}
            for vn in group_voices:
                sorted_samples = sorted(
                    int(
                        (note.start + timing_offsets.get((vn, i), 0.0))
                        * self.sample_rate
                    )
                    for i, note in enumerate(self.voices[vn].notes)
                )
                voice_onset_samples[vn] = sorted_samples

            for onset_time, onset_voice in onsets:
                onset_sample = int(onset_time * self.sample_rate)
                for vn in group_voices:
                    if vn == onset_voice:
                        continue
                    buf = rendered_voice_bases[vn]
                    buf_len = buf.shape[-1] if buf.ndim > 1 else buf.shape[0]
                    if onset_sample >= buf_len:
                        continue
                    end = min(onset_sample + fade_samples, buf_len)
                    fade_len = end - onset_sample
                    fade = np.linspace(1.0, 0.0, fade_len)

                    # Find next onset of the choked voice after the fade to
                    # limit the zero region — don't erase future notes.
                    zero_end = buf_len
                    for own_onset in voice_onset_samples[vn]:
                        if own_onset > onset_sample:
                            zero_end = own_onset
                            break

                    if buf.ndim == 1:
                        buf[onset_sample:end] *= fade
                        buf[end:zero_end] = 0.0
                    else:
                        buf[:, onset_sample:end] *= fade[np.newaxis, :]
                        buf[:, end:zero_end] = 0.0

    def _render_mix_components_internal(
        self,
        *,
        collect_effect_analysis: bool,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, list[synth.EffectAnalysisEntry]],
        dict[str, list[synth.EffectAnalysisEntry]],
        dict[str, np.ndarray],
    ]:
        """Render dry stems, send returns, and optional effect diagnostics."""
        voice_effects: dict[str, list[synth.EffectAnalysisEntry]] = {}
        send_inputs: dict[str, list[np.ndarray]] = {
            send_bus.name: [] for send_bus in self.send_buses
        }
        timing_offsets = self.resolve_timing_offsets()
        velocity_multipliers = self._build_velocity_multiplier_map()
        rendered_voice_bases: dict[str, np.ndarray] = {}
        for voice_name, voice in self.voices.items():
            self._validate_voice_sends(voice.sends)
            rendered_voice_bases[voice_name] = self._render_voice_base(
                voice_name=voice_name,
                voice=voice,
                timing_offsets=timing_offsets,
                velocity_multipliers=velocity_multipliers,
            )

        self._apply_choke_groups(rendered_voice_bases, timing_offsets)

        processed_voice_outputs: dict[str, np.ndarray] = {}
        for voice_name in self._voice_processing_order():
            voice = self.voices[voice_name]
            rendered_voice, rendered_send_inputs, rendered_effect_analysis = (
                self._finalize_voice_output(
                    voice_name=voice_name,
                    voice=voice,
                    voice_mix=rendered_voice_bases[voice_name],
                    processed_voice_outputs=processed_voice_outputs,
                    collect_effect_analysis=collect_effect_analysis,
                )
            )
            processed_voice_outputs[voice_name] = rendered_voice
            for bus_name, bus_signal in rendered_send_inputs.items():
                send_inputs.setdefault(bus_name, []).append(bus_signal)
            if rendered_effect_analysis:
                voice_effects[voice_name] = rendered_effect_analysis

        rendered_stems = {
            voice_name: processed_voice_outputs[voice_name]
            for voice_name in self.voices
            if voice_name in processed_voice_outputs
            and processed_voice_outputs[voice_name].size > 0
        }
        send_returns, send_effects = self._render_send_returns(
            send_inputs=send_inputs,
            collect_effect_analysis=collect_effect_analysis,
        )
        # dry_stems: post-normalization, post-choke, pre-effects/pan/fader
        dry_stems = {
            name: rendered_voice_bases[name]
            for name in self.voices
            if name in rendered_voice_bases and rendered_voice_bases[name].size > 0
        }
        return rendered_stems, send_returns, voice_effects, send_effects, dry_stems

    def resolve_timing_offsets(self) -> dict[tuple[str, int], float]:
        """Return the deterministic render-time timing offset for each note."""
        return build_timing_offsets(
            targets=self._timing_targets(),
            humanize=self.timing_humanize,
            total_dur=self._time_reference_total_dur(),
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
                        authored_start=self._absolute_note_start(note.start),
                        resolved_start=self._absolute_note_start(resolved_start),
                        timing_offset_seconds=resolved_start - note.start,
                        duration=note.duration,
                        resolved_end=self._absolute_note_start(
                            resolved_start + note.duration
                        ),
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
                    pitch_value = float(note.freq / self.f0_hz)
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

    def _render_voice_base(
        self,
        *,
        voice_name: str,
        voice: Voice,
        timing_offsets: dict[tuple[str, int], float],
        velocity_multipliers: dict[tuple[str, int], float],
    ) -> np.ndarray:
        # External instrument engines (e.g. Surge XT, Vital) render the whole
        # voice at once instead of note-by-note, so dispatch early before the
        # per-note loop.
        synth_defaults = normalize_synth_spec(voice.synth_defaults)
        resolved_defaults = resolve_synth_params(synth_defaults)
        engine_name = str(resolved_defaults.get("engine", "additive"))

        if is_instrument_engine(engine_name):
            return self._render_voice_via_instrument(
                voice_name=voice_name,
                voice=voice,
                timing_offsets=timing_offsets,
                velocity_multipliers=velocity_multipliers,
                engine_params=resolved_defaults,
                engine_name=engine_name,
            )

        voice_signals: list[np.ndarray] = []
        prepared_notes = self._prepare_voice_notes(
            voice_name=voice_name,
            voice=voice,
            timing_offsets=timing_offsets,
            velocity_multipliers=velocity_multipliers,
        )
        prepared_notes = self._apply_voice_polyphony(
            prepared_notes=prepared_notes,
            max_polyphony=voice.max_polyphony,
            legato=voice.legato,
        )
        for prepared_note in prepared_notes:
            if (
                prepared_note.effective_hold_duration + prepared_note.effective_release
                <= 0
            ):
                continue
            note = prepared_note.note
            note_signal = render_note_signal(
                freq=prepared_note.note_freq,
                duration=prepared_note.effective_hold_duration
                + prepared_note.effective_release,
                amp=self._resolve_note_amp(
                    note=note,
                    resolved_velocity=prepared_note.resolved_velocity,
                    velocity_db_per_unit=voice.velocity_db_per_unit,
                ),
                sample_rate=self.sample_rate,
                params=prepared_note.synth_params,
                freq_trajectory=prepared_note.freq_trajectory,
                param_profiles=prepared_note.param_profiles or None,
            )
            note_signal = synth.adsr(
                note_signal,
                attack=prepared_note.effective_attack,
                decay=prepared_note.decay,
                sustain_level=prepared_note.sustain_level,
                release=prepared_note.effective_release,
                sample_rate=self.sample_rate,
                hold_duration=prepared_note.effective_hold_duration,
                vca_nonlinearity=prepared_note.vca_nonlinearity,
                attack_power=prepared_note.attack_power,
                decay_power=prepared_note.decay_power,
                release_power=prepared_note.release_power,
                attack_target=prepared_note.attack_target,
            )
            voice_signals.append(
                synth.at_sample_rate(
                    note_signal, prepared_note.humanized_start, self.sample_rate
                )
            )

        if not voice_signals:
            return np.zeros(0)

        voice_mix = self._stack_signals(voice_signals)
        if voice.sympathetic_amount > 0:
            voice_mix = self._apply_sympathetic_resonance(
                voice_mix, voice, prepared_notes
            )
        return self._normalize_voice_signal(voice_mix, voice)

    def _apply_sympathetic_resonance(
        self,
        voice_mix: np.ndarray,
        voice: Voice,
        prepared_notes: list[PreparedVoiceNote],
    ) -> np.ndarray:
        """Add sympathetic resonance by exciting resonant modes from the voice signal.

        For each note, computes harmonic mode frequencies, measures how much
        energy the voice signal contains at each mode via windowed correlation,
        then synthesizes decaying sinusoids at those frequencies scaled by the
        measured excitation.  This is unconditionally numerically stable (no
        recursive filters).
        """
        if voice.sympathetic_amount <= 0 or not prepared_notes:
            return voice_mix

        nyquist = self.sample_rate * 0.45
        max_resonators = 64

        mode_entries: list[tuple[float, float, float]] = []
        for pn in prepared_notes:
            note_dur = pn.effective_hold_duration + pn.effective_release
            for k in range(1, voice.sympathetic_modes + 1):
                mode_freq = pn.note_freq * k
                if mode_freq >= nyquist:
                    break
                mode_entries.append((mode_freq, pn.humanized_start, note_dur))

        if not mode_entries:
            return voice_mix

        mode_entries.sort(key=lambda x: x[0])
        deduped: list[tuple[float, float, float]] = []
        for freq, start, dur in mode_entries:
            if deduped and abs(freq - deduped[-1][0]) / deduped[-1][0] < 0.01:
                existing_freq, existing_start, existing_dur = deduped[-1]
                deduped[-1] = (
                    existing_freq,
                    min(existing_start, start),
                    max(existing_dur, dur),
                )
            else:
                deduped.append((freq, start, dur))

        if len(deduped) > max_resonators:
            deduped = deduped[:max_resonators]

        n_samples = len(voice_mix)
        sr = self.sample_rate
        voice_mix_f64 = (
            voice_mix.astype(np.float64) if voice_mix.dtype != np.float64 else voice_mix
        )
        t_all = np.arange(n_samples, dtype=np.float64) / sr
        decay_tau = max(voice.sympathetic_decay_s, 0.01)

        # Phase 1: measure excitation per mode (loop over segments, not full arrays)
        excited: list[tuple[float, int, float]] = []
        for mode_freq, note_start, note_dur in deduped:
            start_sample = max(0, int(note_start * sr))
            end_sample = min(n_samples, int((note_start + note_dur) * sr))
            if end_sample <= start_sample:
                continue
            segment = voice_mix_f64[start_sample:end_sample]
            ref_sin = np.sin(2.0 * np.pi * mode_freq * t_all[start_sample:end_sample])
            excitation = abs(np.dot(segment, ref_sin) / len(segment)) * 2.0

            if excitation < 1e-12:
                continue
            excited.append((mode_freq, start_sample, excitation))

        if not excited:
            return voice_mix

        # Phase 2: batch resonator synthesis
        freqs_arr = np.array([e[0] for e in excited], dtype=np.float64)
        starts_arr = np.array([e[1] for e in excited], dtype=np.float64)
        excitations = np.array([e[2] for e in excited], dtype=np.float64)

        r_count = len(freqs_arr)
        chunk_threshold = 50_000_000

        if r_count * n_samples <= chunk_threshold:
            t_2d = t_all[np.newaxis, :]
            phase = 2.0 * np.pi * freqs_arr[:, np.newaxis] * t_2d
            start_times = starts_arr[:, np.newaxis] / sr
            offsets = t_2d - start_times
            ring_env = np.where(offsets >= 0, np.exp(-offsets / decay_tau), 0.0)
            resonance_sum = np.sum(
                excitations[:, np.newaxis] * np.sin(phase) * ring_env, axis=0
            )
        else:
            resonance_sum = np.zeros(n_samples, dtype=np.float64)
            chunk_size = max(1, chunk_threshold // r_count)
            for chunk_start in range(0, n_samples, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_samples)
                t_chunk = t_all[np.newaxis, chunk_start:chunk_end]
                phase_chunk = 2.0 * np.pi * freqs_arr[:, np.newaxis] * t_chunk
                start_times = starts_arr[:, np.newaxis] / sr
                offsets_chunk = t_chunk - start_times
                ring_env_chunk = np.where(
                    offsets_chunk >= 0, np.exp(-offsets_chunk / decay_tau), 0.0
                )
                resonance_sum[chunk_start:chunk_end] = np.sum(
                    excitations[:, np.newaxis] * np.sin(phase_chunk) * ring_env_chunk,
                    axis=0,
                )

        sum_peak = float(np.max(np.abs(resonance_sum)))
        input_peak = float(np.max(np.abs(voice_mix)))
        if sum_peak > 0 and input_peak > 0:
            resonance_sum = resonance_sum * (input_peak / sum_peak)
        return voice_mix + resonance_sum * voice.sympathetic_amount

    def _render_voice_via_instrument(
        self,
        *,
        voice_name: str,
        voice: Voice,
        timing_offsets: dict[tuple[str, int], float],
        velocity_multipliers: dict[tuple[str, int], float],
        engine_params: dict[str, Any],
        engine_name: str,
    ) -> np.ndarray:
        """Render a voice through an external instrument plugin (e.g. Surge XT, Vital).

        Instead of iterating notes and calling ``render_note_signal`` per note,
        this builds a list of note dicts and delegates to the engine's
        ``render_voice`` entry point which drives the plugin with MIDI.
        """
        import importlib  # noqa: PLC0415

        if voice.sympathetic_amount > 0:
            logger.warning(
                "sympathetic_amount=%.2f on instrument-engine voice %r ignored "
                "(sympathetic resonance only applies to native per-note engines)",
                voice.sympathetic_amount,
                voice_name,
            )

        engine_module = importlib.import_module(f"code_musics.engines.{engine_name}")

        note_dicts: list[dict[str, Any]] = []
        for note_index, note in enumerate(voice.notes):
            note_freq = self._resolve_freq(note)
            humanized_start = max(
                0.0,
                note.start + timing_offsets.get((voice_name, note_index), 0.0),
            )
            resolved_velocity = float(
                np.clip(
                    note.velocity
                    * velocity_multipliers.get((voice_name, note_index), 1.0),
                    0.05,
                    2.0,
                )
            )
            note_amp = self._resolve_note_amp(
                note=note,
                resolved_velocity=resolved_velocity,
                velocity_db_per_unit=voice.velocity_db_per_unit,
            )
            note_dict: dict[str, Any] = {
                "freq": note_freq,
                "start": humanized_start,
                "duration": note.duration,
                "velocity": resolved_velocity,
                "amp": note_amp,
            }

            # Translate score-level PitchMotionSpec into engine-level glide params
            if note.pitch_motion is not None:
                motion = note.pitch_motion
                if motion.kind == "linear_bend":
                    target_freq = motion.target_frequency(score_f0_hz=self.f0_hz)
                    # linear_bend: note starts at note_freq, bends toward target.
                    # Engine glide: note starts at glide_from, bends toward freq.
                    note_dict["freq"] = target_freq
                    note_dict["glide_from"] = note_freq
                    note_dict["glide_time"] = note.duration
                elif motion.kind == "ratio_glide":
                    start_ratio = float(motion.params.get("start_ratio", 1.0))
                    end_ratio = float(motion.params.get("end_ratio", 1.0))
                    note_dict["freq"] = note_freq * end_ratio
                    note_dict["glide_from"] = note_freq * start_ratio
                    note_dict["glide_time"] = float(
                        motion.params.get("glide_duration", note.duration)
                    )

            note_dicts.append(note_dict)

        if not note_dicts:
            return np.zeros(0)

        max_end = max(n["start"] + n["duration"] for n in note_dicts)
        voice_mix = engine_module.render_voice(
            notes=note_dicts,
            total_duration=max_end,
            sample_rate=self.sample_rate,
            params=engine_params,
        )

        return self._normalize_voice_signal(voice_mix, voice)

    def _normalize_voice_signal(
        self, voice_mix: np.ndarray, voice: Voice
    ) -> np.ndarray:
        """Apply LUFS or peak normalization to a rendered voice signal."""
        if voice.normalize_lufs is not None:
            voice_mix = synth.normalize_to_lufs(
                voice_mix,
                sample_rate=self.sample_rate,
                target_lufs=voice.normalize_lufs,
            )
        elif voice.normalize_peak_db is not None:
            peak = float(np.max(np.abs(voice_mix)))
            if peak > 0.0:
                voice_mix = voice_mix * (
                    synth.db_to_amp(voice.normalize_peak_db) / peak
                )
        return voice_mix

    def _macro_lookup_for_times(self, times: np.ndarray) -> dict[str, np.ndarray]:
        """Sample every registered macro against a time grid.

        Per-render memoization via ``_macro_lookup_cache`` keyed by the
        times array's id/shape keeps repeated lookups cheap across
        voice evaluations in the same render pass.
        """
        if not self.macros:
            return {}
        cache_key = (times.shape[0], float(times[0]) if times.size else 0.0)
        cached = self._macro_lookup_cache.get(cache_key)
        if cached is not None:
            return cached
        lookup = build_macro_lookup(self.macros, times=times)
        self._macro_lookup_cache[cache_key] = lookup
        return lookup

    def _matrix_connections_for_voice(self, voice: Voice) -> list[ModConnection]:
        """Return score-level + voice-level modulation connections."""
        if not self.modulations and not voice.modulations:
            return []
        return [*self.modulations, *voice.modulations]

    def _build_source_context(
        self,
        *,
        times: np.ndarray,
        note_velocity: float | None = None,
        note_start: float | None = None,
        note_duration: float | None = None,
    ) -> SourceSamplingContext:
        """Build a ``SourceSamplingContext`` prefilled with macro values."""
        macro_lookup = self._macro_lookup_for_times(times) if times.size else {}
        return SourceSamplingContext(
            sample_rate=self.sample_rate,
            total_dur=self._time_reference_total_dur(),
            macro_lookup=macro_lookup,
            note_velocity=note_velocity,
            note_start=note_start,
            note_duration=note_duration,
        )

    def _apply_matrix_to_synth_params(
        self,
        *,
        params: dict[str, Any],
        connections: list[ModConnection],
        resolved_velocity: float,
        note_start: float,
        note_duration: float,
    ) -> dict[str, Any]:
        """Fold matrix contributions into synth params at note onset.

        Per-sample-capable synth destinations are skipped here because
        they are threaded through the engine via ``param_profiles``;
        everything else is sampled as a scalar at ``note_start`` and
        written back into ``params`` under the target name.
        """
        if not connections:
            return params
        synth_connections = [
            connection
            for connection in connections
            if connection.target.kind == "synth"
        ]
        if not synth_connections:
            return params
        # Eagerly group to skip per-sample targets — those are handled
        # separately via build_param_profiles below.
        context = self._build_source_context(
            times=np.asarray([note_start], dtype=np.float64),
            note_velocity=resolved_velocity,
            note_start=note_start,
            note_duration=note_duration,
        )
        updated = dict(params)
        grouped: dict[str, list[ModConnection]] = {}
        for connection in synth_connections:
            if is_per_sample_synth_destination(connection.target.name):
                continue
            grouped.setdefault(connection.target.name, []).append(connection)
        for param_name, group in grouped.items():
            base = float(updated.get(param_name, 0.0))
            combined = combine_connections_scalar(
                base=base,
                connections=group,
                context=context,
            )
            updated[param_name] = combined
        return updated

    def _build_param_profiles(
        self,
        *,
        connections: list[ModConnection],
        synth_params: dict[str, Any],
        resolved_velocity: float,
        note_start: float,
        note_duration: float,
        total_samples: int,
    ) -> dict[str, np.ndarray]:
        """Build per-sample profiles for synth destinations in the whitelist.

        Only destinations with ``is_per_sample_synth_destination(name)``
        true AND at least one connection get a profile.  The profile
        carries the sampled scalar base (from ``synth_params``) plus
        the matrix contribution across the note's time grid.
        """
        if total_samples <= 0 or not connections:
            return {}
        grouped: dict[str, list[ModConnection]] = {}
        for connection in connections:
            if connection.target.kind != "synth":
                continue
            if not is_per_sample_synth_destination(connection.target.name):
                continue
            grouped.setdefault(connection.target.name, []).append(connection)
        if not grouped:
            return {}
        times = note_start + np.arange(total_samples, dtype=np.float64) / float(
            self.sample_rate
        )
        context = self._build_source_context(
            times=times,
            note_velocity=resolved_velocity,
            note_start=note_start,
            note_duration=note_duration,
        )
        profiles: dict[str, np.ndarray] = {}
        for param_name, group in grouped.items():
            base_value = float(synth_params.get(param_name, 0.0))
            base_curve = np.full(times.shape, base_value, dtype=np.float64)
            profiles[param_name] = combine_connections_on_curve(
                base=base_curve,
                connections=group,
                times=times,
                context=context,
            )
        return profiles

    def describe_modulations(self) -> list[dict[str, Any]]:
        """Return a flat summary of score + voice modulation connections.

        Inspection helper: one dict per connection with source type,
        destination, and key shaping fields.  Does not touch audio.
        """
        rows: list[dict[str, Any]] = []
        for connection in self.modulations:
            rows.append(self._describe_connection(connection, scope="score"))
        for voice_name, voice in self.voices.items():
            for connection in voice.modulations:
                rows.append(
                    self._describe_connection(connection, scope=f"voice:{voice_name}")
                )
        return rows

    @staticmethod
    def _describe_connection(
        connection: ModConnection, *, scope: str
    ) -> dict[str, Any]:
        return {
            "scope": scope,
            "name": connection.name,
            "source": type(connection.source).__name__,
            "target_kind": connection.target.kind,
            "target_name": connection.target.name,
            "amount": connection.amount,
            "bipolar": connection.bipolar,
            "stereo": connection.stereo,
            "power": connection.power,
            "mode": connection.mode,
        }

    def _drift_bus_trajectory(self, bus: DriftBusSpec) -> tuple[np.ndarray, np.ndarray]:
        """Lazily build and cache the full-score bus trajectory.

        Memoizing on ``Score`` means every subscribed voice/note samples the
        SAME ratio array via ``np.interp``, giving the promised cross-voice
        correlation at identical wall-clock times.
        """
        total_dur = self._time_reference_total_dur()
        cache_key = (bus.name, bus.seed, round(bus.rate_hz, 9), round(total_dur, 6))
        cached = self._drift_bus_cache.get(cache_key)
        if cached is not None:
            return cached
        # Pad so np.interp never needs to extrapolate past the final sample.
        padded_dur = max(total_dur, 0.0) + 1.0
        internal_rate = 1000.0
        n_internal = max(2, int(math.ceil(padded_dur * internal_rate)) + 2)
        dense_times = np.arange(n_internal, dtype=np.float64) / internal_rate
        dense_ratio = build_drift_bus(
            times=dense_times,
            rate_hz=bus.rate_hz,
            depth_cents=bus.depth_cents,
            sample_rate=self.sample_rate,
            seed=bus.seed,
        )
        self._drift_bus_cache[cache_key] = (dense_times, dense_ratio)
        return dense_times, dense_ratio

    def _apply_drift_bus_to_trajectory(
        self,
        *,
        voice: Voice,
        note_start: float,
        duration: float,
        release: float,
        base_freq: float,
        freq_trajectory: np.ndarray | None,
    ) -> tuple[np.ndarray | None, float]:
        """Return ``(freq_trajectory, pitch_drift_scale)`` for a subscribed note.

        Samples the cached shared drift bus via ``np.interp`` so every voice at
        the same wall-clock time sees the same ratio.  Returns a scale for the
        engine's independent ``pitch_drift`` so the caller can apply it
        explicitly (no hidden mutation of synth params).  See
        ``docs/score_api.md`` for the log-space blend semantics.
        """
        bus_name = voice.drift_bus
        if bus_name is None:
            return freq_trajectory, 1.0
        if bus_name not in self.drift_buses:
            raise ValueError(
                f"Voice {voice.name!r} subscribes to drift_bus {bus_name!r} "
                "which is not defined on the score"
            )
        correlation = float(voice.drift_bus_correlation)
        pitch_drift_scale = 1.0 - correlation if correlation > 0.0 else 1.0
        if correlation <= 0.0:
            return freq_trajectory, pitch_drift_scale

        total_samples = max(0, int((duration + release) * self.sample_rate))
        if total_samples == 0:
            return freq_trajectory, pitch_drift_scale

        dense_times, dense_ratio = self._drift_bus_trajectory(
            self.drift_buses[bus_name]
        )
        sample_times = note_start + np.arange(total_samples, dtype=np.float64) / float(
            self.sample_rate
        )
        bus_ratio = np.interp(sample_times, dense_times, dense_ratio)
        scaled_bus = np.power(bus_ratio, correlation)

        if freq_trajectory is None:
            return base_freq * scaled_bus, pitch_drift_scale
        return freq_trajectory * scaled_bus, pitch_drift_scale

    def _debug_freq_trajectories(
        self,
    ) -> dict[str, list[np.ndarray | None]]:
        """Return per-voice, per-note freq trajectories as seen by the engine.

        Intended for tests and diagnostics — prepares voices through the
        normal pipeline (including drift-bus wiring) without rendering audio.
        """
        trajectories: dict[str, list[np.ndarray | None]] = {}
        for voice_name, voice in self.voices.items():
            prepared = self._prepare_voice_notes(
                voice_name=voice_name,
                voice=voice,
                timing_offsets={},
                velocity_multipliers={
                    (voice_name, idx): 1.0 for idx in range(len(voice.notes))
                },
            )
            trajectories[voice_name] = [n.freq_trajectory for n in prepared]
        return trajectories

    def _prepare_voice_notes(
        self,
        *,
        voice_name: str,
        voice: Voice,
        timing_offsets: dict[tuple[str, int], float],
        velocity_multipliers: dict[tuple[str, int], float],
    ) -> list[PreparedVoiceNote]:
        prepared_notes: list[PreparedVoiceNote] = []
        for note_index, note in enumerate(voice.notes):
            synth_params = normalize_synth_spec(voice.synth_defaults)
            if note.synth is not None:
                synth_params.update(normalize_synth_spec(note.synth))
            synth_params = resolve_synth_params(synth_params)
            synth_params["_voice_name"] = voice_name
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
            note_automation = list(note.automation)
            absolute_note_start = self._absolute_note_start(note.start)
            synth_params = apply_synth_automation(
                params=synth_params,
                voice_automation=voice.automation,
                note_automation=note_automation,
                note_start=absolute_note_start,
            )
            matrix_connections = self._matrix_connections_for_voice(voice)
            synth_params = self._apply_matrix_to_synth_params(
                params=synth_params,
                connections=matrix_connections,
                resolved_velocity=resolved_velocity,
                note_start=absolute_note_start,
                note_duration=note.duration,
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
                note_start=absolute_note_start,
                humanize=voice.envelope_humanize,
                total_dur=self._time_reference_total_dur(),
                voice_name=voice_name,
            )
            attack, release = _apply_voice_card_env_rate_scaling(
                attack=attack,
                release=release,
                synth_params=synth_params,
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
                    score_f0_hz=self.f0_hz,
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
                    note_start=absolute_note_start,
                )
                pitch_ratio_connections = [
                    connection
                    for connection in matrix_connections
                    if connection.target.kind == "pitch_ratio"
                ]
                if pitch_ratio_connections and held_samples > 0:
                    held_times = absolute_note_start + np.arange(
                        held_samples, dtype=np.float64
                    ) / float(self.sample_rate)
                    # Matrix rides the freq ratio, so fold into the
                    # trajectory (or build from scratch if automation
                    # didn't produce one).  Start from an all-ones
                    # ratio so 'add'/'multiply'/'replace' combine the
                    # same way the base automation would at 1.0.
                    if held_trajectory is None:
                        base_ratio = np.ones(held_samples, dtype=np.float64)
                    else:
                        base_ratio = held_trajectory / note_freq
                    context = self._build_source_context(
                        times=held_times,
                        note_velocity=resolved_velocity,
                        note_start=absolute_note_start,
                        note_duration=note.duration,
                    )
                    modulated_ratio = combine_connections_on_curve(
                        base=base_ratio,
                        connections=pitch_ratio_connections,
                        times=held_times,
                        context=context,
                    )
                    if np.any(modulated_ratio <= 0):
                        raise ValueError(
                            "pitch_ratio modulation must produce strictly positive ratios"
                        )
                    held_trajectory = note_freq * modulated_ratio
                if held_trajectory is not None:
                    release_samples = max(0, total_samples - held_samples)
                    if release_samples > 0:
                        release_tail = np.full(release_samples, held_trajectory[-1])
                        freq_trajectory = np.concatenate(
                            [held_trajectory, release_tail]
                        )
                    else:
                        freq_trajectory = held_trajectory

            # Blend shared drift bus into freq trajectory; scale engine drift
            # so the two sources don't double-count.  See docs/score_api.md.
            freq_trajectory, pitch_drift_scale = self._apply_drift_bus_to_trajectory(
                voice=voice,
                note_start=absolute_note_start,
                duration=note.duration,
                release=release,
                base_freq=note_freq,
                freq_trajectory=freq_trajectory,
            )
            if pitch_drift_scale != 1.0 and "pitch_drift" in synth_params:
                synth_params["pitch_drift"] = (
                    float(synth_params["pitch_drift"]) * pitch_drift_scale
                )

            vca_nonlinearity = float(synth_params.get("vca_nonlinearity", 0.0))
            attack_power = float(synth_params.get("attack_power", 1.0))
            decay_power = float(synth_params.get("decay_power", 1.0))
            release_power = float(synth_params.get("release_power", 1.0))
            attack_target = float(synth_params.get("attack_target", 1.0))
            param_profiles = self._build_param_profiles(
                connections=matrix_connections,
                synth_params=synth_params,
                resolved_velocity=resolved_velocity,
                note_start=absolute_note_start,
                note_duration=note.duration,
                total_samples=total_samples,
            )
            prepared_notes.append(
                PreparedVoiceNote(
                    note_index=note_index,
                    note=note,
                    synth_params=synth_params,
                    resolved_velocity=resolved_velocity,
                    note_freq=note_freq,
                    humanized_start=humanized_start,
                    attack=attack,
                    decay=decay,
                    sustain_level=sustain_level,
                    release=release,
                    freq_trajectory=freq_trajectory,
                    effective_hold_duration=note.duration,
                    effective_attack=attack,
                    effective_release=release,
                    vca_nonlinearity=vca_nonlinearity,
                    attack_power=attack_power,
                    decay_power=decay_power,
                    release_power=release_power,
                    attack_target=attack_target,
                    param_profiles=param_profiles,
                )
            )
        return prepared_notes

    def _apply_voice_polyphony(
        self,
        *,
        prepared_notes: list[PreparedVoiceNote],
        max_polyphony: int | None,
        legato: bool,
    ) -> list[PreparedVoiceNote]:
        if max_polyphony is None:
            return prepared_notes

        sorted_notes = sorted(
            prepared_notes,
            key=lambda prepared_note: (
                prepared_note.humanized_start,
                prepared_note.note.start,
                prepared_note.note_index,
            ),
        )
        active_notes: list[PreparedVoiceNote] = []
        for prepared_note in sorted_notes:
            prepared_note.effective_hold_duration = prepared_note.note.duration
            prepared_note.effective_attack = prepared_note.attack
            prepared_note.effective_release = prepared_note.release
            current_start = prepared_note.humanized_start
            active_notes = [
                active_note
                for active_note in active_notes
                if (
                    active_note.humanized_start
                    + active_note.effective_hold_duration
                    + active_note.effective_release
                )
                > current_start
            ]
            while len(active_notes) >= max_polyphony:
                stolen_note = min(
                    active_notes,
                    key=lambda active_note: (
                        active_note.humanized_start,
                        active_note.note_index,
                    ),
                )
                overlap = (
                    stolen_note.humanized_start
                    + stolen_note.effective_hold_duration
                    + stolen_note.effective_release
                ) > current_start
                _STEAL_RELEASE_S = 0.005  # 5 ms micro-release prevents click
                stolen_note.effective_hold_duration = max(
                    0.0, current_start - stolen_note.humanized_start
                )
                stolen_note.effective_release = _STEAL_RELEASE_S
                active_notes.remove(stolen_note)
                if legato and max_polyphony == 1 and overlap:
                    prepared_note.effective_attack = 0.0
            active_notes.append(prepared_note)
        return prepared_notes

    def _finalize_voice_output(
        self,
        *,
        voice_name: str,
        voice: Voice,
        voice_mix: np.ndarray,
        processed_voice_outputs: dict[str, np.ndarray],
        collect_effect_analysis: bool,
    ) -> tuple[
        np.ndarray,
        dict[str, np.ndarray],
        list[synth.EffectAnalysisEntry],
    ]:
        effect_analysis: list[synth.EffectAnalysisEntry] = []
        processed_voice_mix = np.asarray(voice_mix, dtype=np.float64)
        signal_times = self._signal_times(processed_voice_mix.shape[-1])
        voice_matrix = self._matrix_connections_for_voice(voice)
        processed_voice_mix = self._apply_db_control(
            processed_voice_mix,
            base_db=voice.pre_fx_gain_db,
            automation_specs=voice.automation,
            target_name="pre_fx_gain_db",
            signal_times=signal_times,
            modulations=voice_matrix,
        )
        processed_voice_mix = self._apply_pan_control(
            processed_voice_mix,
            base_pan=voice.pan,
            automation_specs=voice.automation,
            signal_times=signal_times,
            modulations=voice_matrix,
        )
        if voice.effects and processed_voice_mix.size > 0:
            if collect_effect_analysis:
                rendered_voice_mix, rendered_effect_analysis = cast(
                    tuple[np.ndarray, list[synth.EffectAnalysisEntry]],
                    synth.apply_effect_chain(
                        processed_voice_mix,
                        voice.effects,
                        sidechain_signals=processed_voice_outputs,
                        signal_name=voice_name,
                        start_time_seconds=self.time_origin_seconds,
                        return_analysis=True,
                    ),
                )
                processed_voice_mix = rendered_voice_mix
                effect_analysis = rendered_effect_analysis
            else:
                processed_voice_mix = cast(
                    np.ndarray,
                    synth.apply_effect_chain(
                        processed_voice_mix,
                        voice.effects,
                        sidechain_signals=processed_voice_outputs,
                        signal_name=voice_name,
                        start_time_seconds=self.time_origin_seconds,
                    ),
                )
        processed_voice_mix = self._apply_db_control(
            processed_voice_mix,
            base_db=voice.mix_db,
            automation_specs=voice.automation,
            target_name="mix_db",
            signal_times=signal_times,
            modulations=voice_matrix,
        )
        send_signals = {
            send.target: self._apply_db_control(
                processed_voice_mix,
                base_db=send.send_db,
                automation_specs=send.automation,
                target_name="send_db",
                signal_times=signal_times,
                modulations=voice_matrix,
            )
            for send in voice.sends
        }
        return processed_voice_mix, send_signals, effect_analysis

    def _render_send_returns(
        self,
        *,
        send_inputs: dict[str, list[np.ndarray]],
        collect_effect_analysis: bool,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, list[synth.EffectAnalysisEntry]],
    ]:
        """Render summed shared send returns after collecting all voice feeds."""
        send_returns: dict[str, np.ndarray] = {}
        send_effects: dict[str, list[synth.EffectAnalysisEntry]] = {}
        for send_bus in self.send_buses:
            bus_inputs = send_inputs.get(send_bus.name, [])
            if not bus_inputs:
                continue
            bus_mix = self._stack_signals(bus_inputs)
            signal_times = self._signal_times(bus_mix.shape[-1])
            effect_analysis: list[synth.EffectAnalysisEntry] = []
            if send_bus.effects:
                if collect_effect_analysis:
                    processed_bus_mix, rendered_effect_analysis = cast(
                        tuple[np.ndarray, list[synth.EffectAnalysisEntry]],
                        synth.apply_effect_chain(
                            bus_mix,
                            send_bus.effects,
                            start_time_seconds=self.time_origin_seconds,
                            return_analysis=True,
                        ),
                    )
                    bus_mix = processed_bus_mix
                    effect_analysis = rendered_effect_analysis
                else:
                    bus_mix = cast(
                        np.ndarray,
                        synth.apply_effect_chain(
                            bus_mix,
                            send_bus.effects,
                            start_time_seconds=self.time_origin_seconds,
                        ),
                    )
            bus_mix = self._apply_db_control(
                bus_mix,
                base_db=send_bus.return_db,
                automation_specs=send_bus.automation,
                target_name="return_db",
                signal_times=signal_times,
            )
            bus_mix = self._apply_pan_control(
                bus_mix,
                base_pan=send_bus.pan,
                automation_specs=send_bus.automation,
                target_name="pan",
                signal_times=signal_times,
            )
            if bus_mix.size > 0:
                send_returns[send_bus.name] = bus_mix
            if effect_analysis:
                send_effects[send_bus.name] = effect_analysis
        return send_returns, send_effects

    def _timing_targets(self) -> list[TimingTarget]:
        targets: list[TimingTarget] = []
        for voice_name, voice in self.voices.items():
            for note_index, note in enumerate(voice.notes):
                targets.append(
                    TimingTarget(
                        key=(voice_name, note_index),
                        voice_name=voice_name,
                        start=self._absolute_note_start(note.start),
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
                        start=self._absolute_note_start(note.start),
                    )
                )
        return targets

    def _absolute_note_start(self, note_start: float) -> float:
        return self.time_origin_seconds + note_start

    def _time_reference_total_dur(self) -> float:
        if self.time_reference_total_dur is not None:
            return self.time_reference_total_dur
        return self.time_origin_seconds + self.total_dur

    def _validate_send_buses(self) -> None:
        send_bus_names = [send_bus.name for send_bus in self.send_buses]
        if len(send_bus_names) != len(set(send_bus_names)):
            raise ValueError("send bus names must be unique")

    def _validate_voice_sends(self, sends: list[VoiceSend]) -> None:
        send_bus_names = {send_bus.name for send_bus in self.send_buses}
        seen_targets: set[str] = set()
        for send in sends:
            if send.target in seen_targets:
                raise ValueError(
                    f"voice send targets must be unique per voice: {send.target}"
                )
            seen_targets.add(send.target)
            if send.target not in send_bus_names:
                raise ValueError(
                    f"voice send target does not exist on score: {send.target}"
                )

    def _voice_sidechain_sources(self, *, voice_name: str, voice: Voice) -> list[str]:
        sources: list[str] = []
        for effect in voice.effects:
            if effect.kind != "compressor":
                continue
            sidechain_source = effect.params.get("sidechain_source")
            if sidechain_source is None:
                continue
            if not isinstance(sidechain_source, str) or not sidechain_source.strip():
                raise ValueError("sidechain_source must be a non-empty string")
            normalized_source = sidechain_source.strip()
            if normalized_source == voice_name:
                continue
            if normalized_source not in self.voices:
                raise ValueError(f"Unknown sidechain_source: {normalized_source!r}")
            sources.append(normalized_source)
        return sources

    def _voice_processing_order(self) -> list[str]:
        ordered_voice_names: list[str] = []
        visiting: list[str] = []
        visit_state: dict[str, bool] = {}

        def visit(voice_name: str) -> None:
            if visit_state.get(voice_name) is True:
                return
            if voice_name in visiting:
                cycle_start = visiting.index(voice_name)
                cycle = visiting[cycle_start:] + [voice_name]
                raise ValueError(
                    "Voice sidechain routing contains a cycle: " + " -> ".join(cycle)
                )

            visiting.append(voice_name)
            for source_voice_name in self._voice_sidechain_sources(
                voice_name=voice_name,
                voice=self.voices[voice_name],
            ):
                visit(source_voice_name)
            visiting.pop()
            visit_state[voice_name] = True
            ordered_voice_names.append(voice_name)

        for voice_name in self.voices:
            visit(voice_name)
        return ordered_voice_names

    def _apply_master_bus_processing(
        self,
        mix: np.ndarray,
        *,
        collect_effect_analysis: bool = False,
        mix_effects: list[synth.EffectAnalysisEntry] | None = None,
    ) -> np.ndarray:
        processed_mix = np.asarray(mix, dtype=np.float64)
        if self.auto_master_gain_stage:
            processed_mix = synth.gain_stage_for_master_bus(
                processed_mix,
                sample_rate=self.sample_rate,
                target_lufs=self.master_bus_target_lufs,
                max_true_peak_dbfs=self.master_bus_max_true_peak_dbfs,
            )
            post_stage_peak = (
                float(np.max(np.abs(processed_mix))) if processed_mix.size > 0 else 0.0
            )
            logger.info(
                f"Master auto gain stage: peak {20.0 * np.log10(max(post_stage_peak, 1e-12)):.2f} dBFS"
            )
        if self.master_input_gain_db != 0.0:
            processed_mix = processed_mix * synth.db_to_amp(self.master_input_gain_db)
        if self.master_effects:
            if collect_effect_analysis:
                mastered_mix, effect_entries = cast(
                    tuple[np.ndarray, list[synth.EffectAnalysisEntry]],
                    synth.apply_effect_chain(
                        processed_mix,
                        self.master_effects,
                        start_time_seconds=self.time_origin_seconds,
                        return_analysis=True,
                    ),
                )
                processed_mix = mastered_mix
                if mix_effects is not None:
                    mix_effects.extend(effect_entries)
            else:
                processed_mix = cast(
                    np.ndarray,
                    synth.apply_effect_chain(
                        processed_mix,
                        self.master_effects,
                        start_time_seconds=self.time_origin_seconds,
                    ),
                )

        # Transparent output ceiling: only reduces gain if the signal exceeds
        # -0.5 dBFS (~0.944), so normal mixes are untouched but accidental
        # overloads cannot hard-clip the final output.
        ceiling = synth.db_to_amp(_OUTPUT_CEILING_DBFS)
        peak = float(np.max(np.abs(processed_mix))) if processed_mix.size > 0 else 0.0
        if peak > ceiling:
            attenuation_db = 20.0 * np.log10(ceiling / peak)
            logger.warning(
                f"Master output ceiling activated: peak was "
                f"{20.0 * np.log10(peak):.2f} dBFS, attenuated by "
                f"{attenuation_db:.2f} dB to {20.0 * np.log10(ceiling):.2f} dBFS"
            )
            processed_mix = processed_mix * (ceiling / peak)
        return processed_mix

    def _resolve_freq(self, note: NoteEvent) -> float:
        if note.freq is not None:
            return note.freq
        assert (
            note.partial is not None
        )  # NoteEvent invariant: exactly one of partial/freq
        return self.f0_hz * note.partial

    def _signal_times(self, sample_count: int) -> np.ndarray:
        return self.time_origin_seconds + (
            np.arange(sample_count, dtype=np.float64) / self.sample_rate
        )

    def _apply_db_control(
        self,
        signal: np.ndarray,
        *,
        base_db: float,
        automation_specs: list[AutomationSpec],
        target_name: str,
        signal_times: np.ndarray,
        modulations: list[ModConnection] | None = None,
    ) -> np.ndarray:
        if signal.size == 0:
            return np.asarray(signal, dtype=np.float64)
        db_curve = apply_control_automation(
            base_value=base_db,
            specs=automation_specs,
            target_name=target_name,
            times=signal_times,
        )
        if modulations:
            matrix_connections = iter_connections_for_target(
                modulations, kind="control", name=target_name
            )
            if matrix_connections:
                context = self._build_source_context(times=signal_times)
                db_curve = combine_connections_on_curve(
                    base=db_curve,
                    connections=matrix_connections,
                    times=signal_times,
                    context=context,
                )
        gain_curve = np.power(10.0, db_curve / 20.0)
        if signal.ndim == 1:
            return np.asarray(signal * gain_curve, dtype=np.float64)
        return np.asarray(signal * gain_curve[np.newaxis, :], dtype=np.float64)

    def _apply_pan_control(
        self,
        signal: np.ndarray,
        *,
        base_pan: float,
        automation_specs: list[AutomationSpec],
        target_name: str = "pan",
        signal_times: np.ndarray,
        modulations: list[ModConnection] | None = None,
    ) -> np.ndarray:
        if signal.size == 0:
            return np.asarray(signal, dtype=np.float64)
        has_pan_automation = any(
            spec.target.kind == "control" and spec.target.name == target_name
            for spec in automation_specs
        )
        pan_connections = (
            iter_connections_for_target(modulations, kind="control", name=target_name)
            if modulations
            else []
        )
        if base_pan == 0.0 and not has_pan_automation and not pan_connections:
            return np.asarray(signal, dtype=np.float64)
        pan_curve = apply_control_automation(
            base_value=base_pan,
            specs=automation_specs,
            target_name=target_name,
            times=signal_times,
        )
        if pan_connections:
            context = self._build_source_context(times=signal_times)
            pan_curve = combine_connections_on_curve(
                base=pan_curve,
                connections=pan_connections,
                times=signal_times,
                context=context,
            )
        return synth.apply_pan_automation(signal, pan_curve=pan_curve)

    def _build_velocity_multiplier_map(self) -> dict[tuple[str, int], float]:
        velocity_multipliers: dict[tuple[str, int], float] = {}
        for humanize, targets in self._velocity_targets().items():
            velocity_multipliers.update(
                build_velocity_multipliers(
                    targets=targets,
                    humanize=humanize,
                    total_dur=self._time_reference_total_dur(),
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
