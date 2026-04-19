"""Per-connection modulation matrix.

Vital-style modulation wiring where every routing is a first-class
``ModConnection`` object with ``amount``, ``bipolar``, ``stereo``,
``power``, and an optional drawable ``breakpoints`` curve.  Complements
``AutomationSpec`` (segment-based timeline curves) by making every
``source -> destination`` wiring explicit and reusable.

The matrix is evaluated alongside existing automation: per-sample
destinations (``pitch_ratio``, control targets) receive a combined
curve; per-note synth destinations receive a scalar folded into the
params dict at note onset.  See ``docs/score_api.md`` for the combine
order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from code_musics.automation import (
    AutomationMode,
    AutomationSpec,
    AutomationTarget,
    apply_mode_vectorized,
)
from code_musics.humanize import (
    DriftSpec,
    _sample_drift_curve,
    hann_crossfade_anchors,
    seed_or_default,
    stable_seed,
)

ModSourceDomain = Literal["bipolar", "unipolar"]
LFOWaveshape = Literal[
    "sine", "triangle", "saw_up", "saw_down", "square", "smoothed_random"
]
OscillatorWaveshape = Literal["sine", "saw", "triangle"]

# Synth destinations that support per-sample sampling via engine
# ``param_profiles``.  Engines not listed here accept only the scalar
# note-onset value from ``apply_synth_automation`` plus the matrix
# scalar fold.  Destinations in this set get a per-sample ndarray
# threaded to the engine through ``render_note_signal``.
_PER_SAMPLE_SYNTH_DESTINATIONS: frozenset[str] = frozenset(
    {
        "cutoff_hz",
        "pulse_width",
        "osc2_detune_cents",
        "osc2_freq_ratio",
        "osc_spread_cents",
    }
)


def is_per_sample_synth_destination(name: str) -> bool:
    """Return whether a synth-kind destination supports per-sample profiles."""
    return name in _PER_SAMPLE_SYNTH_DESTINATIONS


@dataclass(frozen=True)
class SourceSamplingContext:
    """Information a source needs to render into a time grid.

    Passed to ``ModSource.sample(...)`` so sources can look up macro
    values, derive per-note seeds, or snap to the absolute time base.
    Constructed by the render loop; sources treat it as read-only.
    """

    sample_rate: int
    total_dur: float
    # Pre-resolved macro values keyed by macro name.  Macros with
    # timelines have already been sampled against the requested times;
    # macros without have their constant default broadcast.
    macro_lookup: dict[str, np.ndarray] = field(default_factory=dict)
    # Per-note state.  Populated for per-note evaluation; left as None
    # for score-time per-sample evaluation where no active note
    # dominates the context (e.g., send-bus pan rides).
    note_velocity: float | None = None
    note_start: float | None = None
    note_duration: float | None = None


@dataclass(frozen=True)
class ModSource:
    """Abstract base for modulation sources.

    Subclasses implement :meth:`sample` to produce a mono curve on the
    requested time grid.  Sources advertise ``output_domain`` so
    ``ModConnection`` shaping can rectify or scale appropriately.
    Sources are frozen dataclasses so they hash cleanly and can be
    shared across connections.
    """

    output_domain: ModSourceDomain = "bipolar"

    def sample(self, _times: np.ndarray, _context: SourceSamplingContext) -> np.ndarray:
        raise NotImplementedError

    def is_per_note_scalar(self) -> bool:
        """Return whether this source emits exactly one scalar per note.

        Per-note scalars feed synth destinations at note onset.  False
        (default) means the source can be sampled at arbitrary time
        grids.
        """
        return False


@dataclass(frozen=True)
class LFOSource(ModSource):
    """Free-run or retriggered LFO with multiple waveshapes."""

    rate_hz: float = 1.0
    waveshape: LFOWaveshape = "sine"
    phase_rad: float = 0.0
    retrigger: bool = False
    seed: int | None = None
    output_domain: ModSourceDomain = "bipolar"

    def __post_init__(self) -> None:
        if self.rate_hz <= 0:
            raise ValueError("LFOSource.rate_hz must be positive")
        if self.waveshape == "smoothed_random" and self.rate_hz > 200:
            raise ValueError("smoothed_random LFO rate is capped below audio rate")

    def sample(self, times: np.ndarray, context: SourceSamplingContext) -> np.ndarray:
        if times.size == 0:
            return np.zeros(0, dtype=np.float64)
        if self.retrigger and context.note_start is not None:
            local_times = times - context.note_start
        else:
            local_times = times
        phase = 2.0 * math.pi * self.rate_hz * local_times + self.phase_rad

        if self.waveshape == "sine":
            return np.sin(phase)
        if self.waveshape == "triangle":
            frac = (self.rate_hz * local_times + self.phase_rad / (2.0 * math.pi)) % 1.0
            return np.asarray(2.0 * np.abs(2.0 * frac - 1.0) - 1.0, dtype=np.float64)
        if self.waveshape == "saw_up":
            frac = (self.rate_hz * local_times + self.phase_rad / (2.0 * math.pi)) % 1.0
            return np.asarray(2.0 * frac - 1.0, dtype=np.float64)
        if self.waveshape == "saw_down":
            frac = (self.rate_hz * local_times + self.phase_rad / (2.0 * math.pi)) % 1.0
            return np.asarray(1.0 - 2.0 * frac, dtype=np.float64)
        if self.waveshape == "square":
            return np.where(np.sin(phase) >= 0.0, 1.0, -1.0).astype(np.float64)
        if self.waveshape == "smoothed_random":
            return _sample_smoothed_random_lfo(
                times=local_times,
                rate_hz=self.rate_hz,
                seed=seed_or_default(self.seed, "lfo_smoothed_random"),
            )
        raise ValueError(f"Unsupported LFO waveshape: {self.waveshape!r}")


@dataclass(frozen=True)
class OscillatorSource(ModSource):
    """Arbitrary-rate oscillator modulation source (LFO-to-audio-rate).

    Sibling to :class:`LFOSource` that removes the sub-audio rate cap,
    intended for audio-rate modulation destinations (PWM, osc2 detune
    FM, supersaw spread modulation).  Running at rates near or above
    ``sample_rate / 2`` produces undefined (naively-aliased) output —
    callers are expected to keep ``rate_hz`` comfortably below Nyquist
    and rely on the destination engine's oversampling for higher
    quality.

    Parameters mirror :class:`LFOSource` for consistency.  Shapes are
    limited to ``sine``, ``saw``, and ``triangle`` for the MVP; square
    is intentionally omitted pending anti-aliased generation.

    Parameters
    ----------
    rate_hz
        Oscillator frequency in Hz, must be strictly positive.  There
        is no upper cap, but values above ``sample_rate / 2`` alias.
    waveshape
        One of ``"sine"``, ``"saw"``, ``"triangle"``.
    phase
        Starting phase normalized to ``[0, 1)`` cycles (0.25 cycles
        == 90 degrees).
    stereo
        When True, :meth:`sample` returns a ``(2, n)`` array whose L
        and R rows differ by ``stereo_phase_offset`` cycles so a
        stereo-aware consumer can decorrelate L/R without a second
        source.  When False (default), :meth:`sample` returns a 1D
        mono array and behaves identically to :class:`LFOSource`'s
        matrix contract.
    stereo_phase_offset
        Phase shift applied to the right channel in cycles.  Only
        consulted when ``stereo=True``.
    """

    rate_hz: float = 1.0
    waveshape: OscillatorWaveshape = "sine"
    phase: float = 0.0
    stereo: bool = False
    stereo_phase_offset: float = 0.25
    output_domain: ModSourceDomain = "bipolar"

    def __post_init__(self) -> None:
        if self.rate_hz <= 0:
            raise ValueError("OscillatorSource.rate_hz must be positive")
        if self.waveshape not in ("sine", "saw", "triangle"):
            raise ValueError(
                "OscillatorSource.waveshape must be one of 'sine', 'saw', 'triangle'"
            )

    def sample(self, times: np.ndarray, context: SourceSamplingContext) -> np.ndarray:
        if times.size == 0:
            if self.stereo:
                return np.zeros((2, 0), dtype=np.float64)
            return np.zeros(0, dtype=np.float64)
        local_times = np.asarray(times, dtype=np.float64)
        if not self.stereo:
            return _render_oscillator_wave(
                times=local_times,
                rate_hz=self.rate_hz,
                phase_cycles=self.phase,
                waveshape=self.waveshape,
            )
        left = _render_oscillator_wave(
            times=local_times,
            rate_hz=self.rate_hz,
            phase_cycles=self.phase,
            waveshape=self.waveshape,
        )
        right = _render_oscillator_wave(
            times=local_times,
            rate_hz=self.rate_hz,
            phase_cycles=self.phase + self.stereo_phase_offset,
            waveshape=self.waveshape,
        )
        return np.stack([left, right], axis=0)


def _render_oscillator_wave(
    *,
    times: np.ndarray,
    rate_hz: float,
    phase_cycles: float,
    waveshape: OscillatorWaveshape,
) -> np.ndarray:
    """Render a mono oscillator waveform over ``times`` (seconds)."""
    if waveshape == "sine":
        phase_rad = 2.0 * math.pi * (rate_hz * times + phase_cycles)
        return np.sin(phase_rad).astype(np.float64)
    frac = (rate_hz * times + phase_cycles) % 1.0
    if waveshape == "saw":
        return np.asarray(2.0 * frac - 1.0, dtype=np.float64)
    if waveshape == "triangle":
        return np.asarray(2.0 * np.abs(2.0 * frac - 1.0) - 1.0, dtype=np.float64)
    raise ValueError(f"Unsupported OscillatorSource waveshape: {waveshape!r}")


@dataclass(frozen=True)
class EnvelopeSource(ModSource):
    """Per-note ADSR envelope source.

    Triggered at note onset; returns ``0`` before the note starts and
    the release tail after the held duration.  Curves use the same
    power convention as synth ADSR so a ``decay_power=2`` envelope here
    matches the one in the synth engines.
    """

    attack: float = 0.01
    hold: float = 0.0
    decay: float = 0.1
    sustain: float = 1.0
    release: float = 0.2
    attack_power: float = 1.0
    decay_power: float = 1.0
    release_power: float = 1.0
    output_domain: ModSourceDomain = "unipolar"

    def __post_init__(self) -> None:
        for stage_name, stage_value in (
            ("attack", self.attack),
            ("hold", self.hold),
            ("decay", self.decay),
            ("release", self.release),
        ):
            if stage_value < 0:
                raise ValueError(f"EnvelopeSource.{stage_name} must be non-negative")
        if not 0.0 <= self.sustain <= 1.0:
            raise ValueError("EnvelopeSource.sustain must be in [0, 1]")
        for power_name, power_value in (
            ("attack_power", self.attack_power),
            ("decay_power", self.decay_power),
            ("release_power", self.release_power),
        ):
            if power_value <= 0:
                raise ValueError(f"EnvelopeSource.{power_name} must be positive")

    def sample(self, times: np.ndarray, context: SourceSamplingContext) -> np.ndarray:
        if times.size == 0:
            return np.zeros(0, dtype=np.float64)
        if context.note_start is None or context.note_duration is None:
            raise ValueError(
                "EnvelopeSource requires note_start and note_duration in context"
            )
        t = times - context.note_start
        out = np.zeros(times.shape, dtype=np.float64)
        attack_end = self.attack
        hold_end = attack_end + self.hold
        decay_end = hold_end + self.decay
        hold_duration = max(context.note_duration, 0.0)
        release_start = hold_duration
        release_end = release_start + self.release

        # Attack ramp (0 -> 1)
        if self.attack > 0:
            mask = (t >= 0.0) & (t < attack_end)
            progress = np.clip(t[mask] / self.attack, 0.0, 1.0)
            out[mask] = np.power(progress, self.attack_power)
        else:
            out[(t >= 0.0) & (t < hold_end)] = 1.0

        # Hold plateau (1)
        if self.hold > 0:
            mask = (t >= attack_end) & (t < hold_end)
            out[mask] = 1.0

        # Decay (1 -> sustain)
        if self.decay > 0:
            mask = (t >= hold_end) & (t < decay_end)
            progress = np.clip((t[mask] - hold_end) / self.decay, 0.0, 1.0)
            shaped = np.power(progress, self.decay_power)
            out[mask] = 1.0 - (1.0 - self.sustain) * shaped
        # Sustain
        sustain_mask = (t >= decay_end) & (t < release_start)
        out[sustain_mask] = self.sustain

        # Release (sustain -> 0)
        if self.release > 0:
            mask = (t >= release_start) & (t < release_end)
            progress = np.clip((t[mask] - release_start) / self.release, 0.0, 1.0)
            shaped = np.power(progress, self.release_power)
            out[mask] = self.sustain * (1.0 - shaped)

        return out


@dataclass(frozen=True)
class MacroSource(ModSource):
    """Named macro scalar.

    Resolved via ``context.macro_lookup[name]``.  Macros are registered
    on the ``Score`` with ``Score.add_macro(...)`` and may carry an
    ``AutomationSpec`` so the macro value rides the piece timeline.
    """

    name: str = ""
    output_domain: ModSourceDomain = "unipolar"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MacroSource.name must be non-empty")

    def sample(self, times: np.ndarray, context: SourceSamplingContext) -> np.ndarray:
        if times.size == 0:
            return np.zeros(0, dtype=np.float64)
        curve = context.macro_lookup.get(self.name)
        if curve is None:
            raise ValueError(
                f"MacroSource {self.name!r} has no registered value"
                " — call Score.add_macro(name, ...) first"
            )
        if curve.shape != times.shape:
            raise ValueError(
                f"macro {self.name!r} has shape {curve.shape} but expected {times.shape};"
                " macro lookup was sampled on a different time grid"
            )
        return curve


@dataclass(frozen=True)
class VelocitySource(ModSource):
    """Per-note velocity in [0, 1].

    Emits ``note_velocity / velocity_scale`` clipped to ``[0, 1]``.
    The default ``velocity_scale=1.25`` matches the upper bound of the
    common ``[0.75, 1.25]`` velocity humanization range, so 1.25+ maps
    to 1.0.
    """

    velocity_scale: float = 1.25
    output_domain: ModSourceDomain = "unipolar"

    def __post_init__(self) -> None:
        if self.velocity_scale <= 0:
            raise ValueError("VelocitySource.velocity_scale must be positive")

    def is_per_note_scalar(self) -> bool:
        return True

    def sample(self, times: np.ndarray, context: SourceSamplingContext) -> np.ndarray:
        if context.note_velocity is None:
            raise ValueError("VelocitySource requires note_velocity in context")
        value = float(np.clip(context.note_velocity / self.velocity_scale, 0.0, 1.0))
        if times.size == 0:
            return np.zeros(0, dtype=np.float64)
        return np.full(times.shape, value, dtype=np.float64)


@dataclass(frozen=True)
class RandomSource(ModSource):
    """Seeded sample-and-hold source.

    Piecewise-constant between hold intervals.  When ``retrigger`` is
    set, a fresh hold sample is drawn at each note onset; otherwise
    the source is a free-running S&H at ``rate_hz``.
    """

    rate_hz: float = 2.0
    retrigger: bool = False
    seed: int | None = None
    output_domain: ModSourceDomain = "bipolar"

    def __post_init__(self) -> None:
        if self.rate_hz <= 0:
            raise ValueError("RandomSource.rate_hz must be positive")

    def sample(self, times: np.ndarray, context: SourceSamplingContext) -> np.ndarray:
        if times.size == 0:
            return np.zeros(0, dtype=np.float64)
        hold_period = 1.0 / self.rate_hz
        if self.retrigger and context.note_start is not None:
            local_times = times - context.note_start
            local_times = np.maximum(local_times, 0.0)
        else:
            local_times = times
        bucket = np.floor(local_times / hold_period).astype(np.int64)
        unique_buckets, inverse = np.unique(bucket, return_inverse=True)
        base_seed = seed_or_default(self.seed, "random_sh")
        values = np.empty(unique_buckets.size, dtype=np.float64)
        for index, bucket_value in enumerate(unique_buckets):
            rng = np.random.default_rng(stable_seed(base_seed, int(bucket_value)))
            if self.output_domain == "unipolar":
                values[index] = float(rng.random())
            else:
                values[index] = float(rng.uniform(-1.0, 1.0))
        return values[inverse]


@dataclass(frozen=True)
class ConstantSource(ModSource):
    """Constant mono value.

    For stereo pan-split effects on mono destinations, use two voices
    with opposite-signed ``amount`` on their respective
    ``ModConnection`` (see ``code_musics/pieces/mod_matrix_study.py``
    for a demo).  A native stereo source for mono destinations is
    deferred future work.
    """

    value: float = 1.0
    output_domain: ModSourceDomain = "bipolar"

    def sample(self, times: np.ndarray, context: SourceSamplingContext) -> np.ndarray:
        if times.size == 0:
            return np.zeros(0, dtype=np.float64)
        return np.full(times.shape, float(self.value), dtype=np.float64)


@dataclass(frozen=True)
class DriftAdapter(ModSource):
    """Adapter that exposes a :class:`humanize.DriftSpec` as a mod source.

    The underlying generator is the exact drift curve used by
    ``TimingHumanizeSpec`` / ``EnvelopeHumanizeSpec`` /
    ``VelocityHumanizeSpec``, so the musical character matches.  The
    curve is normalized to ``max|curve| = 1`` before being handed to
    the connection, keeping it in bipolar-native domain.
    """

    style: Literal[
        "random_walk", "smooth_noise", "lfo", "sample_hold", "smoothed_random"
    ] = "random_walk"
    rate_hz: float = 0.3
    smoothness: float = 0.8
    seed: int | None = None
    output_domain: ModSourceDomain = "bipolar"

    def __post_init__(self) -> None:
        if self.rate_hz <= 0:
            raise ValueError("DriftAdapter.rate_hz must be positive")
        if not 0.0 <= self.smoothness <= 1.0:
            raise ValueError("DriftAdapter.smoothness must be in [0, 1]")

    def sample(self, times: np.ndarray, context: SourceSamplingContext) -> np.ndarray:
        spec = DriftSpec(
            style=self.style,
            rate_hz=self.rate_hz,
            smoothness=self.smoothness,
            seed=self.seed,
        )
        return _sample_drift_curve(
            spec,
            times=np.asarray(times, dtype=np.float64),
            total_dur=max(context.total_dur, 1e-6),
            seed=seed_or_default(self.seed, "drift_adapter"),
        )


@dataclass(frozen=True)
class ModConnection:
    """One source -> destination routing.

    Fields follow Vital's ``ModulationConnectionProcessor`` surface.
    The combine order applied by the sampler is:

        raw = source.sample(times)
        if not bipolar: raw = clip(raw, 0, +inf) if domain=='bipolar'
                              else raw  # unipolar stays in [0,1]
        shaped = power_curve(raw, power)
        shaped = breakpoint_curve(shaped, breakpoints)
        signal = amount * shaped
        (stereo handled separately by destination consumers)
    """

    source: ModSource
    target: AutomationTarget
    amount: float = 1.0
    bipolar: bool = True
    stereo: bool = False
    power: float = 0.0
    breakpoints: tuple[tuple[float, float], ...] | None = None
    mode: AutomationMode = "add"
    name: str | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.amount):
            raise ValueError("ModConnection.amount must be finite")
        if not -20.0 <= self.power <= 20.0:
            raise ValueError("ModConnection.power must be in [-20, 20]")
        if self.breakpoints is not None:
            if len(self.breakpoints) < 2:
                raise ValueError("breakpoints must have at least two points")
            previous_x = -math.inf
            for x, y in self.breakpoints:
                if not 0.0 <= x <= 1.0:
                    raise ValueError("breakpoint x must be in [0, 1]")
                if x <= previous_x:
                    raise ValueError("breakpoints must be strictly increasing in x")
                if not np.isfinite(y):
                    raise ValueError("breakpoint y must be finite")
                previous_x = x
        if self.mode not in {"replace", "add", "multiply"}:
            raise ValueError(f"Unsupported ModConnection.mode: {self.mode!r}")

    def shape(self, raw: np.ndarray) -> np.ndarray:
        """Apply bipolar/power/breakpoint shaping and ``amount`` scaling."""
        if raw.size == 0:
            return raw
        signal = np.asarray(raw, dtype=np.float64)
        if not self.bipolar and self.source.output_domain == "bipolar":
            # Unipolar sources are already non-negative; bipolar sources
            # get rectified when the connection is marked unipolar.
            signal = np.maximum(signal, 0.0)
        if self.power != 0.0:
            signal = _apply_power_curve(signal, self.power, self.source.output_domain)
        if self.breakpoints is not None:
            signal = _apply_breakpoints(
                signal, self.breakpoints, self.source.output_domain
            )
        return signal * float(self.amount)


def _apply_power_curve(
    signal: np.ndarray, power: float, domain: ModSourceDomain
) -> np.ndarray:
    """Apply Vital's sign-magnitude power curve.

    In unipolar space the mapping is ``y = x ** exponent`` where
    ``exponent = exp(-power/4)``.  Positive ``power`` bends concave
    (loud near 1), negative bends convex.  Bipolar signals are folded
    to unipolar magnitude, shaped, then re-signed.
    """
    exponent = math.exp(-power / 4.0)
    if domain == "unipolar":
        clipped = np.clip(signal, 0.0, 1.0)
        return np.power(clipped, exponent)
    magnitude = np.clip(np.abs(signal), 0.0, 1.0)
    shaped = np.power(magnitude, exponent)
    return np.sign(signal) * shaped


def _apply_breakpoints(
    signal: np.ndarray,
    breakpoints: tuple[tuple[float, float], ...],
    domain: ModSourceDomain,
) -> np.ndarray:
    """Apply a piecewise-linear breakpoint curve mapping [0, 1] -> y."""
    xs = np.asarray([bp[0] for bp in breakpoints], dtype=np.float64)
    ys = np.asarray([bp[1] for bp in breakpoints], dtype=np.float64)
    if domain == "unipolar":
        clipped = np.clip(signal, 0.0, 1.0)
        return np.interp(clipped, xs, ys)
    magnitude = np.clip(np.abs(signal), 0.0, 1.0)
    shaped = np.interp(magnitude, xs, ys)
    return np.sign(signal) * shaped


def _sample_smoothed_random_lfo(
    *,
    times: np.ndarray,
    rate_hz: float,
    seed: int,
) -> np.ndarray:
    """Helm-style smoothed-random LFO: random anchors with Hann crossfade."""
    if times.size == 0:
        return np.zeros(0, dtype=np.float64)
    period = 1.0 / rate_hz
    t0 = float(times[0])
    t1 = float(times[-1])
    span = max(t1 - t0, period)
    anchor_count = int(math.ceil(span / period)) + 2
    anchor_times = t0 + np.arange(anchor_count, dtype=np.float64) * period
    rng = np.random.default_rng(seed)
    anchor_values = rng.uniform(-1.0, 1.0, size=anchor_count).astype(np.float64)
    return hann_crossfade_anchors(times, anchor_times, anchor_values)


# --- High-level evaluation helpers ------------------------------------------


def build_macro_lookup(
    macros: dict[str, MacroDefinition],
    *,
    times: np.ndarray,
) -> dict[str, np.ndarray]:
    """Sample every registered macro against ``times``.

    Each macro with an attached :class:`AutomationSpec` is sampled on
    ``times``; macros without a spec fall back to their ``default``
    broadcast over the grid.
    """
    lookup: dict[str, np.ndarray] = {}
    if times.size == 0:
        return lookup
    for name, macro in macros.items():
        if macro.automation is None:
            lookup[name] = np.full(times.shape, float(macro.default), dtype=np.float64)
            continue
        base = np.full(times.shape, float(macro.default), dtype=np.float64)
        sampled = macro.automation.sample_many_raw(times)
        combined = np.where(np.isnan(sampled), base, sampled)
        lookup[name] = np.clip(combined, 0.0, 1.0)
    return lookup


@dataclass(frozen=True)
class MacroDefinition:
    """Registered macro: name, default value, and optional timeline."""

    name: str
    default: float = 0.0
    automation: AutomationSpec | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MacroDefinition.name must be non-empty")
        if not 0.0 <= self.default <= 1.0:
            raise ValueError("MacroDefinition.default must be in [0, 1]")
        if self.automation is not None and self.automation.target.kind != "control":
            raise ValueError("MacroDefinition.automation target.kind must be 'control'")


def iter_connections_for_target(
    connections: list[ModConnection],
    *,
    kind: str,
    name: str,
) -> list[ModConnection]:
    """Return connections routed to a specific destination."""
    return [
        connection
        for connection in connections
        if connection.target.kind == kind and connection.target.name == name
    ]


def combine_connections_on_curve(
    *,
    base: np.ndarray,
    connections: list[ModConnection],
    times: np.ndarray,
    context: SourceSamplingContext,
    source_cache: dict[Any, np.ndarray] | None = None,
) -> np.ndarray:
    """Evaluate and combine connections that share a destination curve.

    Combine order follows the plan: ``replace`` (last wins) -> then
    ``multiply`` -> then ``add``.  Matrix contributions are summed
    against ``base`` after base automation has already been applied.
    """
    if not connections:
        return base
    replace_connections = [c for c in connections if c.mode == "replace"]
    multiply_connections = [c for c in connections if c.mode == "multiply"]
    add_connections = [c for c in connections if c.mode == "add"]
    curve = base
    for connection in replace_connections:
        shaped = _sample_and_shape(connection, times, context, source_cache)
        curve = apply_mode_vectorized("replace", curve, shaped)
    for connection in multiply_connections:
        shaped = _sample_and_shape(connection, times, context, source_cache)
        curve = apply_mode_vectorized("multiply", curve, shaped)
    for connection in add_connections:
        shaped = _sample_and_shape(connection, times, context, source_cache)
        curve = apply_mode_vectorized("add", curve, shaped)
    return curve


def combine_connections_scalar(
    *,
    base: float,
    connections: list[ModConnection],
    context: SourceSamplingContext,
    source_cache: dict[Any, np.ndarray] | None = None,
) -> float:
    """Evaluate connections into a single scalar at ``context.note_start``.

    Used for synth destinations that are folded into the params dict
    at note onset.  Connections whose source is per-sample are sampled
    at the note's start time and the first element is used.
    """
    if not connections:
        return base
    if context.note_start is None:
        raise ValueError("scalar combine requires context.note_start")
    eval_times = np.asarray([context.note_start], dtype=np.float64)
    scalar_base = np.asarray([float(base)], dtype=np.float64)
    combined = combine_connections_on_curve(
        base=scalar_base,
        connections=connections,
        times=eval_times,
        context=context,
        source_cache=source_cache,
    )
    return float(combined[0])


def _sample_and_shape(
    connection: ModConnection,
    times: np.ndarray,
    context: SourceSamplingContext,
    source_cache: dict[Any, np.ndarray] | None,
) -> np.ndarray:
    cache_key: Any = (id(connection.source), times.ctypes.data, times.size)
    if source_cache is not None and cache_key in source_cache:
        raw = source_cache[cache_key]
    else:
        raw = connection.source.sample(times, context)
        if source_cache is not None:
            source_cache[cache_key] = raw
    return connection.shape(raw)
