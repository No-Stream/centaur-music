# Composition API Reference

This document describes the higher-level composition helpers that sit above the
core `Score` / `Phrase` / `NoteEvent` model.

For the concrete score-domain reference covering `Score`, `Voice`, `NoteEvent`,
`Phrase`, and render-time expression semantics, see
[docs/score_api.md](docs/score_api.md).

The composition layer is phrase-first and xen-friendly:

- it builds ordinary `Phrase` objects
- it keeps timing in seconds rather than MIDI/grid assumptions
- it supports ratio-aware pitch motion for JI and harmonic-series writing

There is now also an optional high-level musical-time layer for authoring in
beats and bars. It compiles back down to the same `Phrase` / `Score` surfaces,
so the underlying render model is still seconds-based.

## Where This Is Used

- [code_musics/composition.py](code_musics/composition.py)
- [code_musics/pitch_motion.py](code_musics/pitch_motion.py)
- [code_musics/score.py](code_musics/score.py)

## Core Types

This doc touches a few nearby score-expression controls for context, but the
detailed score-domain reference now lives in
[docs/score_api.md](docs/score_api.md).

Nearby topics that matter when using the composition helpers:

- note-level `amp_db` and `velocity`
- score-level timing humanization
- voice-level envelope humanization
- voice-level velocity humanization and `velocity_group`
- velocity-driven synth parameter mapping
- bar-aware automation helpers for voice timbre arcs

### `RhythmCell`

Stores onset spans and optional gate values.

- `spans: tuple[float, ...]`
- `gates: float | tuple[float, ...] = 1.0`

`spans` define when each note starts relative to the next onset. `gates` define
how much of each span is actually sounded.

If `spans` is shorter than the tone list passed to `line(...)` or
`ratio_line(...)`, the rhythm cell is cycled to fit. If it is longer than the
tone list, that is treated as an error.

**Important:** `len(rhythm.spans)` must be `<=` `len(tones)`. The rhythm cycles
over tones (a shorter rhythm repeats), but tones do NOT cycle over a longer
rhythm -- that raises a `ValueError`.

- `gates < 1.0` gives clipped/staccato phrasing
- `gates = 1.0` fills the full span
- `gates > 1.0` creates overlap and legato smear

## Musical-Time Layer

Use `code_musics.meter` when you want musical time rather than raw seconds.

Core APIs:

- `Timeline(bpm=..., meter=(num, den), groove=...)`
- rhythmic values: `W`, `H`, `Q`, `E`, `S`
- groove templates: `Groove.eighths_swing(...)`, `Groove.sixteenths_swing(...)`,
  and named presets like `Groove.dilla_lazy()`
- helpers: `B(...)`, `M(...)`, `dotted(...)`, `triplet(...)`, `tuplet(...)`

Example:

```python
from code_musics.composition import grid_line, grid_sequence
from code_musics.meter import Groove, M, Q, Timeline

timeline = Timeline(
    bpm=96,
    meter=(4, 4),
    groove=Groove.eighths_swing(0.62),
)

motif = grid_line(
    tones=[1.0, 5 / 4, 3 / 2, 5 / 4],
    durations=[Q, Q, Q, Q],
    timeline=timeline,
    amp_db=-14.0,
)

grid_sequence(
    score,
    "lead",
    motif,
    timeline=timeline,
    at=[M(1), M(2), M(3)],
)
```

Important conventions:

- the low-level score is still authored and rendered in seconds
- the new helpers compile beats/bars into seconds before creating notes
- plain numeric durations in the grid helpers are interpreted as beats
- `B(...)` is a beat span or absolute beat offset from time zero
- `M(...)` is a bar-position reference where `M(1)` is the first bar start
- `Groove` replaces the earlier `SwingSpec` with a richer per-step model

### `Groove`

`Groove` is the rhythmic feel specification. It stores per-step timing offsets
and velocity weights, providing a richer model than the earlier `SwingSpec`
(which only controlled offbeat position).

Fields:

- `subdivision: Literal["eighth", "sixteenth"]`
- `timing_offsets: tuple[float, ...]` -- per-step timing shift, cycled across
  the subdivision grid
- `velocity_weights: tuple[float, ...]` -- per-step velocity multiplier, cycled
  similarly
- `name: str` -- optional label for display/debugging

Factory methods:

- `Groove.eighths_swing(amount)` -- eighth-note swing; `amount` is the offbeat
  position in `[0.5, 1.0)`, default `2/3` (triplet feel)
- `Groove.sixteenths_swing(amount)` -- sixteenth-note swing, same semantics

Named presets:

- `Groove.mpc_tight()` -- sixteenth-note. Tight MPC-style
  groove with subtle push on beat 2, slight pull on beat 4.
- `Groove.dilla_lazy()` -- sixteenth-note. J Dilla-style lazy
  feel with heavy offbeat drag, quiet ghost notes.
- `Groove.motown_pocket()` -- eighth-note. Classic Motown
  pocket with gently pushed offbeats.
- `Groove.bossa()` -- eighth-note. Bossa nova with anticipated
  (early) offbeats.
- `Groove.tr808_swing()` -- sixteenth-note. TR-808-style swing
  with pushed second sixteenth.

```python
from code_musics.meter import Groove, Timeline

# Factory swing
timeline = Timeline(bpm=96, groove=Groove.eighths_swing(0.62))

# Named preset
timeline = Timeline(bpm=88, groove=Groove.dilla_lazy())

# Custom groove from scratch
groove = Groove(
    subdivision="sixteenth",
    timing_offsets=(0.0, 0.15, 0.0, 0.10),
    velocity_weights=(1.0, 0.6, 0.8, 0.5),
)
timeline = Timeline(bpm=120, groove=groove)
```

### `tuplet(...)`

General tuplet duration helper for any n-in-the-space-of-m subdivision.

```python
from code_musics.meter import Q, E, tuplet

# Quintuplet quarter: 5 notes in the space of 4 quarters
dur = tuplet(5, 4, Q)

# Septuplet eighth: 7 notes in the space of 4 eighths
dur = tuplet(7, 4, E)
```

`tuplet(n, in_space_of, value)` returns a `BeatSpan` equal to
`(in_space_of / n) * value`. Use alongside the existing `triplet(value)` and
`dotted(value)` helpers.

### `Timeline`

`Timeline` is the conversion layer between musical time and seconds.

Useful methods:

- `timeline.duration(Q)`
- `timeline.measures(2.0)`
- `timeline.at(bar=3, beat=1.5)`
- `timeline.position(M(2.5))`
- `timeline.locate(seconds)`

`meter` affects bar length. Beat values are expressed in quarter-note beats, so
`Q` is always one beat and `meter=(6, 8)` gives a 3-beat bar.

If `groove` is set, `Timeline.position(...)`, `Timeline.at(...)`, and
`Timeline.locate(...)` use the grooved grid. Standalone scalar durations like
`timeline.duration(Q)` remain straight-time conversions; groove-aware note spans
are resolved sequence-positionally inside `grid_line(...)` and
`grid_ratio_line(...)`.

### Grid Helpers

These helpers mirror the existing composition helpers but accept musical-time
inputs:

- `grid_line(...)`
- `grid_ratio_line(...)`
- `grid_sequence(...)`
- `grid_canon(...)`
- `metered_sections(...)`

They are additive APIs, not replacements. Use the original seconds-based
helpers when direct control is clearer.

### `MeteredSectionSpec`

`MeteredSectionSpec` is the bar-measured counterpart to `ContextSectionSpec`.
`metered_sections(...)` takes a sequence of these plus a `Timeline` and a
`base_tonic` and resolves them into the same `ContextSection` tuple that the
seconds-based `build_context_sections(...)` produces.

Fields:

- `bars: float` -- section length in bars on the given timeline; must be
  positive. Fractional bar counts are allowed.
- `tonic_ratio: float = 1.0` -- multiplicative offset applied to `base_tonic`
  for this section, matching `ContextSectionSpec.tonic_ratio`.
- `name: str | None = None` -- optional section label.

Use this when section boundaries should snap to the bar grid rather than being
expressed in raw seconds.

### `bar_automation(...)`

Builds a voice-automation lane from musical bar/beat anchor points.

Parameters:

- `target`
- `timeline`
- `points`
- `mode="replace"`
- `clamp_min=None`
- `clamp_max=None`

`points` is a sequence of `(bar, beat, value)` tuples. Consecutive anchors are
connected with linear automation segments in score time.

Example:

```python
from code_musics.composition import bar_automation
from code_musics.meter import Timeline

timeline = Timeline(bpm=88, meter=(4, 4))

cutoff_lane = bar_automation(
    target="cutoff_hz",
    timeline=timeline,
    points=(
        (1, 0.0, 700.0),
        (19, 0.0, 1600.0),
        (27, 0.0, 1100.0),
        (36, 0.0, 1900.0),
    ),
)
```

Practical notes:

- the first point's value becomes the lane's `default_value`
- to keep the last value alive through a tail, add a final anchor with the same
  value at the desired endpoint
- this is a good fit for section-level opening/closing arcs on `Voice.automation`

### `ArticulationSpec`

Phrase-level articulation controls:

- `gate`
- `attack_scale`
- `release_scale`
- `accent_pattern`
- `tail_breath`

These values are applied when building the phrase. Timing remains the primary
articulation mechanism; envelope scaling is a secondary shaping tool.

### `PitchMotionSpec`

Note-level pitch motion attached to a `NoteEvent`. Lives in
`code_musics.pitch_motion`. A `PitchMotionSpec` is frozen and deterministic:
given a base frequency, duration, and sample rate, it produces a per-sample
frequency trajectory that the renderer integrates into oscillator phase.

Fields:

- `kind: Literal["linear_bend", "ratio_glide", "vibrato"]`
- `params: dict[str, Any]`

Prefer the classmethod constructors over building specs directly; each
constructor validates its own parameter set at construction time.

#### `PitchMotionSpec.linear_bend(...)`

Linear bend from the note's base frequency to a single target pitch.
Exactly one of the following keyword arguments must be supplied:

- `target_freq: float | None = None` -- absolute target frequency in Hz.
- `target_partial: float | None = None` -- target resolved as
  `Score.f0_hz * target_partial`.
- `target_ratio: float | None = None` -- target resolved as
  `Score.f0_hz * target_ratio` (same underlying math as `target_partial`;
  use the name that matches how the rest of the phrase is authored --
  partial-space vs. ratio-space).

Returns a `PitchMotionSpec` with `kind="linear_bend"`. The bend is linear
in frequency (not pitch), so large-interval bends will feel faster at the
bottom and slower at the top. Use `ratio_glide(...)` when you want motion
that stays clearly grounded in ratio space.

#### `PitchMotionSpec.ratio_glide(...)`

Logarithmic glide that interpolates evenly in ratio (pitch) space.

- `start_ratio: float = 1.0` -- starting multiplier on the note's base
  frequency.
- `end_ratio: float = 1.0` -- ending multiplier on the note's base
  frequency.

Returns a `PitchMotionSpec` with `kind="ratio_glide"`. Both ratios must be
positive. Prefer this for JI-aware voice leading: an octave glide here is
the same perceptual rate as a semitone glide.

#### `PitchMotionSpec.vibrato(...)`

Small deterministic sinusoidal vibrato around the base frequency.

- `depth_ratio: float = 0.01` -- peak deviation as a multiplicative fraction
  of the base frequency. Must be in `(0, 0.25)`. Typical musical values are
  `0.003`-`0.015`; values above `~0.02` start to feel like modulation rather
  than vibrato.
- `rate_hz: float = 5.5` -- vibrato rate in Hz. Must be positive.
- `phase_rad: float = 0.0` -- starting phase in radians. Use a per-note value
  if you want multiple voices to desynchronize.

Returns a `PitchMotionSpec` with `kind="vibrato"`.

#### `PitchMotionSpec.target_frequency(score_f0_hz)`

Resolve the absolute target frequency for a `linear_bend` spec against the
score root. Raises `ValueError` when called on non-`linear_bend` motions.

#### `build_frequency_trajectory(...)`

Low-level helper that turns a `PitchMotionSpec` into a dense per-sample
frequency trajectory. Pieces usually do not call this directly -- the
renderer does. It is exposed for DSP utilities, analysis, and custom engines.

```python
def build_frequency_trajectory(
    *,
    base_freq: float,
    duration: float,
    sample_rate: int,
    motion: PitchMotionSpec,
    score_f0_hz: float,
) -> np.ndarray
```

- `base_freq` -- the note's resolved base frequency in Hz (must be positive).
- `duration` -- note duration in seconds (must be positive).
- `sample_rate` -- sample rate in Hz (must be positive).
- `motion` -- the `PitchMotionSpec` to realize.
- `score_f0_hz` -- score root, used to resolve partial/ratio bend targets.

Returns a 1-D `np.ndarray` of strictly positive, finite frequencies of
length `int(duration * sample_rate)`. Returns an empty array when the
integer sample count is zero.

#### `phase_from_frequency_trajectory(...)`

Integrate a frequency trajectory into an oscillator phase trajectory.

```python
def phase_from_frequency_trajectory(
    freq_trajectory: np.ndarray,
    *,
    sample_rate: int,
) -> np.ndarray
```

- `freq_trajectory` -- 1-D frequency samples (Hz).
- `sample_rate` -- sample rate in Hz.

Returns a 1-D `np.ndarray` of cumulative phase values in radians, starting
at `0.0`. Useful for feeding any oscillator that consumes phase directly.

### `NoteEvent`

`NoteEvent` is still the atomic score event, but its expressive surface is
broader than the earlier "start/duration/partial/amp" model.

Key fields worth remembering:

- `amp_db: float | None`
- `velocity: float = 1.0`
- `partial: float | None`
- `freq: float | None`
- `synth: dict[str, Any] | None`
- `pitch_motion: PitchMotionSpec | None`

Important behavior:

- exactly one of `partial` or `freq` must be provided
- `amp` and `amp_db` are mutually exclusive
- `velocity` must stay in `(0, 2]`
- `amp_db` is resolved to linear amplitude at construction time

`velocity` is now part of normal authoring, not just a downstream render tweak.
Use it for accents, emphasis, and phrase shape.

### `Voice`

`Voice` is where most render-time expression defaults now live.

Important voice-level fields:

- `synth_defaults`
- `effects`
- `pan`
- `envelope_humanize`
- `velocity_humanize`
- `velocity_group`
- `velocity_to_params`
- `velocity_db_per_unit`

Practical mental model:

- `timing_humanize` belongs to the whole `Score`
- `envelope_humanize` belongs to a `Voice`
- `velocity_humanize` belongs to a `Voice`
- explicit note `velocity` belongs to each `NoteEvent`

### `VelocityParamMap`

`VelocityParamMap` linearly maps resolved velocity to a synth parameter range.

Fields:

- `min_value`
- `max_value`
- `min_velocity = 0.75`
- `max_velocity = 1.25`

Use it when louder notes should also be brighter, noisier, sharper, or
otherwise more animated, rather than merely louder.

Example:

```python
from code_musics.score import VelocityParamMap

score.add_voice(
    "lead",
    synth_defaults={
        "engine": "filtered_stack",
        "preset": "round_bass",
        "env": {"attack_ms": 10.0, "release_ms": 300.0},
    },
    velocity_to_params={
        "cutoff_hz": VelocityParamMap(
            min_value=250.0,
            max_value=1600.0,
            min_velocity=0.8,
            max_velocity=1.2,
        )
    },
)
```

## Humanization Model

The newer humanization APIs are render-time expression layers. They should make
the result feel less rigid without forcing you to abandon explicit score timing
or composition structure.

The main rule of thumb:

- timing humanization changes note start times
- envelope humanization changes ADSR parameters
- velocity humanization changes resolved per-note velocity multipliers

All of them are deterministic for a given seed.

### `DriftSpec`

`DriftSpec` is the reusable drift generator shared by the humanization specs
below (`ensemble_drift` on timing, `drift` on envelope and velocity).

Fields:

- `style: DriftStyle = "random_walk"`
- `rate_hz: float = 0.035`
- `smoothness: float = 0.84`
- `seed: int | None = None`

Accepted `style` values:

- `random_walk` -- cumulative Gaussian increments; slow wandering drift without
  a central tendency.
- `smooth_noise` -- smoothed random anchors linearly interpolated; pink-ish
  organic variation.
- `lfo` -- deterministic sinusoidal oscillation at `rate_hz` with a seeded
  starting phase.
- `sample_hold` -- steppy piecewise-constant values drawn at each anchor; holds
  until the next crossing.
- `smoothed_random` -- Helm-style: random anchors at `rate_hz` crossfaded with
  a raised-cosine (Hann) window. Distinct from the steppy `sample_hold` and
  from `smooth_noise`'s linearly interpolated smoothed anchors; produces
  organic wobble that neither neighbor captures.

### `TimingHumanizeSpec`

High-level ensemble timing drift attached to `Score(timing_humanize=...)`.

Fields:

- `preset`
- `ensemble_drift`
- `ensemble_amount_ms`
- `follow_strength`
- `voice_spread_ms`
- `micro_jitter_ms`
- `chord_spread_ms`
- `seed`

Current presets:

- `tight_ensemble`
- `chamber`
- `relaxed_band`
- `loose_late_night`

How to think about the parameters:

- `ensemble_amount_ms` is the size of the shared temporal breathing
- `follow_strength` controls how tightly voices move together
- `voice_spread_ms` adds per-voice separation
- `micro_jitter_ms` adds small note-level randomness
- `chord_spread_ms` offsets notes that start together so simultaneities
  are not perfectly vertical

Use this when the whole score should feel performed rather than grid-perfect.
Do not use it as a substitute for writing actual rubato or changing the
written rhythm.

Example:

```python
from code_musics.humanize import TimingHumanizeSpec
from code_musics.score import Score

score = Score(
    f0_hz=110.0,
    timing_humanize=TimingHumanizeSpec(
        preset="chamber",
        seed=17,
    ),
)
```

### `EnvelopeHumanizeSpec`

Voice-level ADSR drift attached with `Score.add_voice(..., envelope_humanize=...)`.

Fields:

- `preset`
- `drift`
- `attack_amount_frac`
- `decay_amount_frac`
- `sustain_amount_frac`
- `release_amount_frac`
- `seed`

Current presets:

- `subtle_analog`
- `breathing_pad`
- `loose_pluck`

This is the current API surface for "env slop": smooth, seeded variation in
ADSR timing and sustain amount over score time.

Use it for:

- pads that should breathe
- plucks that should not feel machine-cloned
- analog-ish instability that stays musically bounded

### `VelocityHumanizeSpec`

Voice-level render-time velocity drift attached with
`Score.add_voice(..., velocity_humanize=...)`.

Fields:

- `preset`
- `drift`
- `group_amount`
- `follow_strength`
- `voice_spread`
- `note_jitter`
- `chord_spread`
- `min_multiplier`
- `max_multiplier`
- `seed`

Current presets:

- `subtle_living`
- `breathing_ensemble`

How to think about the parameters:

- `group_amount` controls the shared curve strength
- `follow_strength` controls cross-voice correlation
- `voice_spread` adds per-voice deviation around the shared curve
- `note_jitter` adds independent note-level randomness
- `chord_spread` varies simultaneous notes inside a chord or onset cluster
- `min_multiplier` / `max_multiplier` clamp the final result

`velocity_humanize` defaults to a subtle preset when you call `add_voice(...)`.
If you want fully fixed note dynamics, set `velocity_humanize=None`.

### `velocity_group`

`velocity_group` is a simple but important piece of the model.

Voices with the same `velocity_group` share the same higher-level velocity drift
curve, so they breathe together instead of each voice wobbling independently.

Use it for:

- multiple parts of one ensemble gesture
- doubled lines
- pad stacks that should swell together

Leave it unset when each voice should have its own dynamic life.

Example:

```python
from code_musics.humanize import VelocityHumanizeSpec

shared_velocity = VelocityHumanizeSpec(
    preset="breathing_ensemble",
    seed=5,
)

for voice_name in ("lead", "alto"):
    score.add_voice(
        voice_name,
        velocity_humanize=shared_velocity,
        velocity_group="ensemble",
    )
```

### `HarmonicContext` and context drift helpers

These helpers support sectional/context drift without introducing persistent
pitch identities.

- `HarmonicContext(tonic=..., name=...)`
- `context.drifted(by_ratio=...)`
- `build_context_sections(base_tonic=..., specs=...)`
- `ratio_line(..., context=...)`
- `place_ratio_line(..., section=...)`

They are designed for cases where the local tuning frame changes by section,
but the rendered output should still be ordinary absolute frequencies.

New section-building helpers sit on top of this same model rather than creating
a separate sequencing system:

- `recontextualize_phrase(..., target_context=..., source_tonic=...)`
- `sequence(..., starts=..., sections=...)`
- `canon(..., voice_names=..., delays=...)`
- `voiced_ratio_chord(..., voicing=..., inversion=...)`
- `progression(..., sections=..., chords=..., pattern=...)`

Use them when you already have a good phrase or harmonic idea and want to
develop it into a section without manually re-placing every note.

## `line(...)`

`line(...)` is the main phrase builder.

It accepts:

- a tone sequence
- a `RhythmCell` or onset spans
- `pitch_kind="partial"` or `pitch_kind="freq"`
- `amp_db` for perceptual level-setting, or legacy linear `amp`
- optional `ArticulationSpec`
- optional pitch motion, either scalar or per-note

When you pass a shorter rhythm cell than the tone list, the onset spans and
rhythm-cell gates repeat automatically. This makes it easy to write a repeating
rhythmic cell against a longer melodic contour.

It returns an ordinary `Phrase`, so existing `Score.add_phrase(...)`,
`Phrase.transformed(...)`, and piece code continue to work.

## Examples

### Articulated partial-space line

```python
from code_musics.composition import ArticulationSpec, RhythmCell, line

phrase = line(
    tones=[6.0, 7.0, 8.0, 9.0],
    rhythm=RhythmCell(spans=(0.7, 0.7, 1.1, 1.4)),
    articulation=ArticulationSpec(
        gate=(0.55, 0.55, 0.9, 1.05),
        accent_pattern=(1.0, 0.9, 1.15, 1.2),
        tail_breath=0.1,
    ),
    amp_db=-14.0,
)
```

### Overlapping/legato line

```python
from code_musics.composition import RhythmCell, line

phrase = line(
    tones=[4.0, 5.0, 6.0],
    rhythm=RhythmCell(spans=(1.0, 1.0, 1.0), gates=1.1),
    amp_db=-18.0,
)
```

### Cycling a shorter rhythm cell

```python
phrase = line(
    tones=[1.0, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 15 / 8],
    rhythm=RhythmCell(spans=(0.5, 0.5, 0.75, 0.25)),
    amp_db=-18.0,
)
```

### Ratio glide in JI space

```python
from code_musics.composition import line
from code_musics.pitch_motion import PitchMotionSpec

phrase = line(
    tones=[6.0, 7.0, 8.0],
    rhythm=(0.8, 0.8, 1.2),
    amp_db=-16.0,
    pitch_motion=(
        None,
        PitchMotionSpec.ratio_glide(start_ratio=1.0, end_ratio=7 / 6),
        PitchMotionSpec.vibrato(depth_ratio=0.015, rate_hz=6.0),
    ),
)
```

### Context drift across sections

```python
from code_musics.composition import (
    ContextSectionSpec,
    build_context_sections,
    place_ratio_line,
)

sections = build_context_sections(
    base_tonic=220.0,
    start=4.0,
    specs=(
        ContextSectionSpec(name="stable", duration=2.5),
        ContextSectionSpec(name="drifted", duration=2.5, tonic_ratio=80 / 81),
    ),
)

place_ratio_line(
    score,
    "melody",
    section=sections[0],
    tones=[1.0, 5 / 4, 3 / 2],
    rhythm=(0.5, 0.5, 1.0),
    amp_db=-16.0,
)
```

## Level Authoring

Prefer `amp_db` over linear `amp` when writing pieces. Decibels track perception
and mix decisions more naturally; small linear changes are often not intuitive.

`amp` is still supported as a raw multiplier, but `amp_db` should be the default
authoring choice for new work.

For voice-level balancing, prefer `Score.add_voice(..., mix_db=...)` as the main
"mixer fader" control, and use `pre_fx_gain_db` when you specifically want to
change how hard a voice hits its own effect chain. `normalize_lufs` is a more
specialized stem-standardization control and is usually not the right first tool
for ordinary mix moves.

Rough starting ranges for note-level balances:

- sub bass or kick-like parts: around `-10 dB` to `-6 dB`
- bass lines and pedals: around `-16 dB` to `-10 dB`
- leads: around `-20 dB` to `-12 dB`
- inner voices and pads: around `-24 dB` to `-16 dB`
- bright percussion or punctuation: often quieter in sustain, but
  transient material may still need `-14 dB` to `-8 dB`

These are only starting points. Apparent loudness depends strongly on spectrum,
envelope, register, and density. In practice, low sustained parts often need more
energy than upper voices, while bright leads may need less level than expected to
sit correctly.

## Velocity Authoring

Use note-level `velocity` for local expression and accents, and use
`velocity_humanize` for gentle render-time variation on top.

Recommended default mental model:

- leave most notes around `velocity=1.0`
- use roughly `0.85` to `0.95` for softer notes
- use roughly `1.05` to `1.2` for accents
- avoid treating velocity like a substitute for all gain staging;
  use `amp_db` for the larger balance decisions

By default, resolved velocity affects loudness through `velocity_db_per_unit`
on the voice. It can also affect timbre when `velocity_to_params` is configured.

That means two equally loud notated lines can still differ musically:

- by note-level `amp_db`
- by note-level `velocity`
- by velocity-driven timbral mapping
- by velocity humanization

## Helper Transforms

These all return new phrases and do not mutate the source:

- `with_gate(...)`
- `with_accent_pattern(...)`
- `with_tail_breath(...)`
- `with_synth_ramp(...)`
- `staccato(...)`
- `legato(...)`
- `echo(...)`
- `concat(...)`
- `overlay(...)`

**`with_synth_ramp(phrase, *, start_params, end_params)`** -- interpolate synth
parameters linearly across successive events. `start_params` and `end_params`
are dicts of matching parameter keys (e.g. `{"cutoff_hz": 400.0}` and
`{"cutoff_hz": 1800.0}`); each event receives a synth override with every key
evenly interpolated between its `start` and `end` value. The first event gets
the start values, the last event gets the end values, and a single-event
phrase gets the start values. Useful for timbral sweeps (filter opens,
brightness rises, detune widens) without hand-authoring every note.

Use them when a phrase is already built and you want a fast re-articulation pass
without rewriting the note list.

`echo(...)` now supports both pitch surfaces:

- use `partial_shift` for partial-authored phrases
- use `freq_scale` for frequency-authored phrases

If you try to use `partial_shift` on a frequency-authored phrase, it raises
instead of silently doing nothing.

### Rhythmic Phrase Transforms

These transforms operate on a phrase's timing and rhythm structure while
preserving pitch content. All return new phrases without mutating the source.

**`augment(phrase, factor)`** -- stretch all durations and inter-onset times
by `factor`. Classical augmentation: `augment(p, 2.0)` doubles all note
lengths.

**`diminish(phrase, factor)`** -- compress all durations by `factor`.
`diminish(p, 2.0)` is equivalent to `augment(p, 0.5)`.

**`rhythmic_retrograde(phrase)`** -- reverse the duration/timing order while
preserving pitch order. If the original is `[Q, E, E, H]` with pitches
`[A, B, C, D]`, the result is `[H, E, E, Q]` with pitches `[A, B, C, D]`.
Different from `reverse=True` on `Phrase.transformed()` which reverses both
pitch and timing.

**`displace(phrase, offset)`** -- shift all note onsets by `offset` seconds.
Positive values push later, negative values push earlier. Use for creating
syncopation or off-beat placements.

**`rotate(phrase, steps)`** -- rotate events cyclically. `rotate(p, 1)` moves
the first event to the end, shifting all others earlier. Preserves total
span. Negative steps rotate in the opposite direction.

```python
from code_musics.composition import (
    augment,
    diminish,
    displace,
    rhythmic_retrograde,
    rotate,
)

slow = augment(motif, 2.0)      # half speed
fast = diminish(motif, 2.0)     # double speed
flipped = rhythmic_retrograde(motif)  # same pitches, reversed rhythm
pushed = displace(motif, 0.125)       # syncopated
spun = rotate(motif, 2)              # cyclic rotation
```

## Polyrhythm and Cross-Rhythm

These builders create interlocking rhythmic layers from division ratios.

### `polyrhythm(a, b, span)`

Returns two `RhythmCell` objects that divide the same timespan into `a` and
`b` equal parts. Use them to build two phrases that interlock polyrhythmically.

```python
from code_musics.composition import polyrhythm

r3, r4 = polyrhythm(3, 4, span=2.0)
# r3 has 3 equal divisions of 2.0 s
# r4 has 4 equal divisions of 2.0 s
```

### `cross_rhythm(layers, span)`

Builds aligned phrases from multiple division layers. Each layer is a tuple of
`(divisions, tones)` where `divisions` is the number of equal subdivisions and
`tones` is a sequence of pitches to cycle through. Returns a list of `Phrase`
objects, one per layer, all spanning the same duration.

```python
from code_musics.composition import cross_rhythm

phrases = cross_rhythm(
    layers=[
        (3, [1.0, 5 / 4, 3 / 2]),   # 3 in the space
        (4, [2.0, 7 / 4]),           # 4 in the space
        (5, [1.0]),                   # 5 in the space
    ],
    span=4.0,
    amp_db=-16.0,
)
# phrases[0] has 3 events, phrases[1] has 4, phrases[2] has 5
```

## Section Helpers

These helpers are for moving from a single phrase or chord idea to a short
section.

- `recontextualize_phrase(...)` resolves an existing phrase into a new local
  tonic as concrete frequencies.
- `sequence(...)` places one phrase multiple times with per-entry transforms.
- `canon(...)` places delayed imitative entries across voices and can repeat the
  subject inside each voice with `repeats` and `repeat_gap`.
- `voiced_ratio_chord(...)` resolves ratios into a register-aware voicing,
  including `drop2` and `drop3` alongside the earlier `close`, `open`, and
  `spread` modes.
- `place_ratio_chord(...)` resolves a list of ratios against a `ContextSection`
  and places them as simultaneous or slightly staggered notes on a voice.
  Parameters: `score`, `voice_name`, `section`, `ratios`, `duration`,
  `offset=0.0`, `gap=0.0` (stagger between successive notes in seconds),
  `amp` (scalar or per-note sequence), `amp_db`, `velocity=1.0`, `synth`,
  `labels`. Returns the list of placed `NoteEvent`s. Use `gap > 0` for
  arpeggiated/strummed chords, `gap = 0` for block chords.
- `progression(...)` places simple harmonic accompaniment patterns from a list
  of sections and ratio chords.

Current `progression(...)` pattern modes:

- `block`
- `arpeggio`
- `pedal_upper`

For `pattern="arpeggio"`, `arpeggio_order` can be:

- `"ascending"`
- `"descending"`
- `"inside_out"`
- an explicit index order into the voiced/sorted chord

## Harmonic Drift

`code_musics.harmonic_drift` provides JI-aware pitch drift automation shaped by
consonance. It generates slow, smooth `pitch_ratio` automation lanes that
glide between chords while lingering near pure JI intervals and moving quickly
through rough zones -- useful for pad voice leading that stays audibly tuned
rather than sliding generically between targets.

Two layers:

- `harmonic_drift(...)` is the low-level engine: given two chords, it returns
  one `AutomationSpec` per voice (chord tone).
- `progression_drift_lanes(...)` sits on top for multi-chord progressions,
  returning per-chord lane lists.
- `drifted_chord_events(...)` builds `NoteEvent` tuples for a chord and
  attaches drift lanes per voice in matching sorted order.

All three cooperate through one convention: **chord tones are sorted low-to-high
by partial before lanes are generated or attached**, so lane indices line up
with the sorted order used by `drifted_chord_events`.

### `harmonic_drift(...)`

```python
def harmonic_drift(
    start_chord: list[float],
    end_chord: list[float],
    duration: float,
    attraction: float = 0.5,
    prime_limit: int = 7,
    wander: float = 0.0,
    smoothness: float = 0.8,
    resolution_ms: float = 50.0,
    seed: int = 0,
    glide_ms: float | None = 250.0,
    max_interval_cents: float = 700.0,
    target_time: float | None = None,
) -> list[AutomationSpec]
```

Generate `pitch_ratio` automation lanes that drift between two JI chords.

- `start_chord` -- list of starting partial ratios, one per voice.
- `end_chord` -- list of ending partial ratios, same length as `start_chord`.
- `duration` -- total automation duration in seconds. Must be positive.
- `attraction` -- strength of consonance pull, in `[0, 1]`. `0.0` gives a
  straight geometric interpolation; higher values warp the trajectory to
  linger near JI waypoints of bounded Tenney height.
- `prime_limit` -- JI prime limit for the waypoint search (default `7`).
- `wander` -- optional smoothed Brownian deviation biased toward nearby JI
  ratios, in `[0, 1]`. `0.0` is a direct glide.
- `smoothness` -- exponential smoothing factor on the final trajectory, in
  `[0, 1]`. Higher values produce slower, glassier motion.
- `resolution_ms` -- internal trajectory time step in milliseconds.
- `seed` -- deterministic seed for the wander noise.
- `glide_ms` -- glide window in ms before the glide target time. The voice
  holds at `start_ratio` for most of the note and only glides in the final
  `glide_ms` before the target. Pass `None` to glide across the entire
  duration (legacy behavior). Defaults to `250.0` ms, which is short enough
  to read as portamento rather than a riser.
- `max_interval_cents` -- voices whose pitch change exceeds this magnitude
  skip pitch drift and emit a flat unity lane. Default `700.0` (roughly a
  perfect fifth). Prevents multi-octave slides when chord voicings pair
  voices from very different registers.
- `target_time` -- time (in seconds, from the note's start) at which the
  glide must reach `end_ratio`. After this time the lane holds at
  `end_ratio` for the rest of the note. Defaults to `duration`. Use the
  boundary time of the next chord for overlapping transitions so the glide
  settles before the next chord attacks and the two voices don't beat.

Returns one `AutomationSpec` per voice, each targeting `pitch_ratio` in
`"multiply"` mode.

Raises `ValueError` when `start_chord` and `end_chord` differ in length,
when `duration <= 0`, or when any of `attraction`, `wander`, `smoothness`
falls outside `[0, 1]`, or when `glide_ms <= 0`, `max_interval_cents < 0`,
or `target_time` falls outside `(0, duration]`.

### `progression_drift_lanes(...)`

```python
def progression_drift_lanes(
    progression: list[ChordVoicingFn],
    chord_dur: float,
    attraction: float,
    wander: float,
    smoothness: float = 0.85,
    seed_base: int = 0,
    glide_ms: float | None = 250.0,
    max_interval_cents: float = 700.0,
    target_time: float | None = None,
    glide_transitions: set[int] | None = None,
) -> list[list[AutomationSpec] | None]
```

Compute drift lanes for consecutive chords in a progression.

- `progression` -- list of chord voicing callables. Each callable takes no
  arguments and returns `list[tuple[float, float]]`, i.e. a list of
  `(partial_ratio, amp_db)` tuples for that chord.
- `chord_dur` -- duration of each chord (shared across the progression).
- `attraction`, `wander`, `smoothness`, `glide_ms`, `max_interval_cents`,
  `target_time` -- forwarded to `harmonic_drift(...)`; see above.
- `seed_base` -- base seed; chord `i` uses `seed_base + i`.
- `glide_transitions` -- optional set of chord indices whose outgoing
  transition should glide. If `None`, every transition glides (default). Use
  this to place drift as an accent on specific transitions rather than a
  continuous effect across the section.

Returns one entry per chord. Each entry is either a list of `AutomationSpec`
(one per voice in sorted order, describing drift toward the next chord) or
`None` for the last chord and for chords whose transition is not selected by
`glide_transitions`.

### `drifted_chord_events(...)`

```python
def drifted_chord_events(
    chord_partials_db: list[tuple[float, float]],
    duration: float,
    drift_lanes: list[AutomationSpec] | None,
    amp_db_offset: float = 0.0,
) -> tuple[NoteEvent, ...]
```

Build `NoteEvent`s for a chord, optionally attaching per-note pitch drift.

- `chord_partials_db` -- `(partial_ratio, amp_db)` pairs for the chord. Tones
  are sorted low-to-high by partial so the sort order matches the lanes
  produced by `progression_drift_lanes`.
- `duration` -- per-note duration in seconds.
- `drift_lanes` -- list of automation specs (one per voice in sorted order),
  or `None` to skip drift for this chord.
- `amp_db_offset` -- additive dB offset applied uniformly to every chord
  tone (e.g. for a per-chord swell).

Returns a tuple of `NoteEvent`s all starting at `0.0`, each with its
matching drift lane attached as note-local automation.

### Type alias

- `ChordVoicingFn = Callable[[], list[tuple[float, float]]]` -- a zero-arg
  callable returning `(partial_ratio, amp_db)` pairs for one chord.

## Smearing and Textural Thickening

`code_musics.smear` provides loveless-inspired tools for pitch smearing,
textural thickening, and slow orchestration moves. It is the companion to
`harmonic_drift` for shoegaze / dream-pop writing: where `harmonic_drift`
shapes voice-leading trajectories in pitch space, `smear` shapes the
grain and density of the sound by stacking micro-detuned copies and
pushing notes through pitch wobble and chord-strum gestures.

All functions are deterministic for a given seed.

### `ThickenedCopy`

```python
@dataclass(frozen=True)
class ThickenedCopy:
    phrase: Phrase
    pan: float
    amp_offset_db: float
```

One micro-detuned, panned copy produced by `thicken(...)`.

- `phrase` -- the detuned, time-offset `Phrase` for this copy.
- `pan` -- pan position in `[-1, 1]` suggested for the copy.
- `amp_offset_db` -- suggested amplitude offset in dB (typically negative
  at the outer copies to taper the stack).

The caller decides how to route copies -- typically one voice per copy with
`pan` and a `mix_db` trim applied from `amp_offset_db`.

### `strum(...)`

```python
def strum(
    phrase: Phrase,
    spread_ms: float = 40.0,
    direction: str = "down",
    seed: int = 0,
) -> Phrase
```

Stagger simultaneous chord notes across a time spread.

- `phrase` -- input phrase (typically a chord with all notes starting at
  `0.0`).
- `spread_ms` -- total time spread in milliseconds across all notes. Must
  be non-negative.
- `direction` -- `"down"` (low-to-high), `"up"` (high-to-low), `"out"`
  (center outward), or `"random"` (seeded deterministic).
- `seed` -- seed for `"random"` direction.

Returns a new `Phrase` whose note start times are staggered to simulate a
strum. Single-note phrases are returned unchanged.

Raises `ValueError` for negative `spread_ms` or unrecognized `direction`.

### `thicken(...)`

```python
def thicken(
    phrase: Phrase,
    n: int = 5,
    detune_cents: float = 8.0,
    spread_ms: float = 20.0,
    stereo_width: float = 0.7,
    amp_taper_db: float = -2.0,
    seed: int = 0,
) -> list[ThickenedCopy]
```

Create micro-detuned, time-staggered, pan-spread copies of a phrase.

- `phrase` -- the source phrase to thicken.
- `n` -- number of copies to produce. Must be at least `1`.
- `detune_cents` -- total detune spread in cents (distributed across
  `+/- detune_cents / 2`).
- `spread_ms` -- total time stagger in ms (each copy's offset is a seeded
  uniform draw in `+/- spread_ms / 2`).
- `stereo_width` -- pan spread; copies are distributed evenly across
  `[-stereo_width, +stereo_width]`.
- `amp_taper_db` -- amplitude reduction in dB applied to the outermost
  copies; the center copy is unattenuated.
- `seed` -- deterministic seed for time offsets.

Returns a list of `ThickenedCopy`. The caller is responsible for placing
each copy on a voice (or on the same voice with pan overrides). Copies
with `partial`-authored notes have their detune applied through an injected
`freq_scale` synth override; copies with `freq`-authored notes have the
scaling baked into `freq`.

### `pitch_wobble(...)`

```python
def pitch_wobble(
    duration: float,
    rate_hz: float = 0.15,
    depth_cents: float = 12.0,
    style: str = "smooth",
    start_time: float = 0.0,
    seed: int = 0,
    depth_curve: list[tuple[float, float]] | None = None,
    segment_interval: float = 0.05,
) -> AutomationSpec
```

Generate a continuous pitch-modulation automation lane targeting
`pitch_ratio` in `"multiply"` mode. Good for gentle tremolo-bar motion,
tape-flutter feel, or slow whole-voice drift.

- `duration` -- length of the wobble in seconds. Must be positive.
- `rate_hz` -- modulation rate. For `"lfo"` it is the sine frequency; for
  `"smooth"` it is the approximate spectral center of the filtered noise;
  for `"drunk"` it sets the damping on the random-walk dynamics.
- `depth_cents` -- modulation depth in cents. For `"lfo"` it is the peak
  deviation (`+/- depth_cents / 2`); for `"smooth"` / `"drunk"` it is the
  RMS deviation.
- `style` -- `"lfo"` (sine), `"smooth"` (filtered Brownian motion), or
  `"drunk"` (random walk with momentum).
- `start_time` -- absolute start time for the automation segments.
- `seed` -- deterministic seed for `"smooth"` / `"drunk"` styles.
- `depth_curve` -- optional list of `(time, depth_cents)` pairs for
  time-varying depth. Times are measured relative to `start_time`; the
  curve is linearly interpolated.
- `segment_interval` -- approximate time between automation segments in
  seconds. Smaller values give smoother motion at the cost of more
  segments.

Returns an `AutomationSpec` built from a sequence of linear segments.

Raises `ValueError` for non-positive `duration` or `rate_hz`, negative
`depth_cents`, or an unrecognized `style`.

### `smear_progression(...)`

```python
def smear_progression(
    chords: Sequence[Sequence[float]],
    durations: Sequence[float],
    overlap: float = 0.5,
    voice_behavior: Sequence[str] | None = None,
) -> list[Phrase]
```

Build gliding voice phrases from a chord progression. Returns one `Phrase`
per voice index (i.e. per chord tone position across all chords). Notes use
`partial` values so they resolve against `Score.f0_hz`.

- `chords` -- list of chord ratio lists, e.g.
  `[[1.0, 5/4, 3/2], [1.0, 6/5, 3/2]]`. Every chord must have the same
  length.
- `durations` -- per-chord duration in seconds. Must match `chords` in
  length; all values must be positive.
- `overlap` -- fraction of the next chord's duration that the previous
  chord continues to sound for. `0.0` gives a gap between chords, `0.5`
  gives half-chord overlap, `1.0` gives full legato carry-through.
- `voice_behavior` -- optional per-voice-index behavior list. Each entry is
  either `"glide"` (default; attach a `ratio_glide` `PitchMotionSpec` that
  lands on the next chord tone during the overlap) or `"reattack"` (cut
  clean at the chord boundary and re-articulate). If provided, its length
  must match the chord voice count.

Returns a list of `Phrase`, one per voice index. The final chord always
reattacks since there is no next chord to glide into.

Raises `ValueError` for empty `chords`, mismatched `chords` / `durations`
lengths, non-positive durations, non-uniform chord sizes, or a
`voice_behavior` list whose length does not match the chord voice count.

### `bloom(...)`

```python
def bloom(
    score: Score,
    voice_specs: Sequence[dict[str, Any]],
    center_time: float,
    grow_dur: float = 4.0,
    peak_dur: float = 8.0,
    fade_dur: float = 4.0,
) -> Score
```

Orchestration helper for gradual layer introduction and dissolution. Staggers
voice entries across `grow_dur`, sustains all voices during `peak_dur`, then
staggers exits across `fade_dur`. Each voice gets amplitude-envelope shaping
so layers fade in and out smoothly instead of popping.

- `score` -- the `Score` to add voices to.
- `voice_specs` -- list of voice specification dicts. Required keys:
  `"name"` (str), `"phrase"` (`Phrase`). Optional keys: `"synth_defaults"`
  (dict, default `{}`), `"pan"` (float, default `0.0`). Any other keys are
  forwarded verbatim as `Score.add_voice(...)` kwargs.
- `center_time` -- midpoint of the peak section in seconds.
- `grow_dur` -- duration over which voices stagger their entries. Must be
  non-negative.
- `peak_dur` -- duration during which all voices sound. Must be
  non-negative.
- `fade_dur` -- duration over which voices stagger their exits. Must be
  non-negative.

Returns the mutated `Score`. The helper registers each voice via
`Score.add_voice(...)` and then inserts amplitude-shaped `add_note(...)`
calls across the bloom window, repeating the source phrase as needed to
fill the voice's active span.

Raises `ValueError` for empty `voice_specs` or any negative duration.

## Recipes

### Make a line more clipped

Use shorter gates either in `RhythmCell` or in `ArticulationSpec`:

```python
phrase = line(
    tones=[6.0, 7.0, 8.0],
    rhythm=RhythmCell(spans=(0.8, 0.8, 1.2), gates=0.55),
)
```

### Intentionally create overlap

Use gates above `1.0` when you want notes to smear into one another:

```python
phrase = line(
    tones=[4.0, 5.0, 6.0],
    rhythm=RhythmCell(spans=(1.0, 1.0, 1.0), gates=(1.1, 1.05, 1.2)),
)
```

### Show a JI arrival bend

Use `linear_bend` or `ratio_glide` to make the tuning itself audible as motion:

```python
phrase = line(
    tones=[6.0, 7.0],
    rhythm=(0.7, 1.2),
    pitch_motion=(
        None,
        PitchMotionSpec.linear_bend(target_partial=8.0),
    ),
)
```

## Generative Composition Helpers

The `code_musics.generative` package provides algorithmic and stochastic
composition tools. All generators work in ratio space, are deterministic when
seeded, and produce standard `Phrase`, `RhythmCell`, or raw ratio lists that
plug directly into the existing composition and score surfaces.

### `TonePool`

Weighted pitch pool for stochastic pitch selection. Ratios are frequency ratios
(e.g. `1.0`, `1.25`, `1.5` for a 4:5:6 triad); weights are normalized to sum
to 1.0.

Constructors:

- `TonePool.uniform(ratios)` -- equal weight for every ratio
- `TonePool.weighted({ratio: weight, ...})` -- auto-normalized from a mapping
- `TonePool.from_harmonics([4, 5, 6, 7])` -- harmonic partial numbers, uniform
  weights, ratios derived as `partial / min(partials)`

Drawing:

- `pool.draw(n, seed=0, replace=True)` -- draw `n` ratios according to weights
- `pool.draw_one(rng=rng)` -- draw a single ratio using an external
  `random.Random` instance (used internally by other generators)

```python
from code_musics.generative import TonePool

pool = TonePool.from_harmonics([4, 5, 6, 7])
ratios = pool.draw(16, seed=42)

# weighted toward the root and fifth
pool = TonePool.weighted({1.0: 3, 5/4: 1, 3/2: 2, 7/4: 1})
```

### Euclidean Rhythm

Bjorklund's algorithm distributes `hits` onsets as evenly as possible across
`steps`. Three entry points at different levels of abstraction:

**`euclidean_pattern(hits, steps, rotation=0)`**

Returns a `tuple[bool, ...]` onset mask.

**`euclidean_rhythm(hits, steps, span=0.25, rotation=0)`**

Converts the pattern into a `RhythmCell`. Silent steps are absorbed into the
preceding sounding step's span. Returns `None` when `hits` is 0.

**`euclidean_line(tones, hits, steps, span=0.25, rotation=0, ...)`**

Builds a complete `Phrase` by cycling `tones` through the sounding positions of
a euclidean rhythm. Accepts `pitch_kind`, `amp`/`amp_db`, `gate`, `synth`, and
an optional `HarmonicContext` for ratio resolution.

```python
from code_musics.generative import euclidean_line, euclidean_rhythm

rhythm = euclidean_rhythm(5, 8, span=0.25)

phrase = euclidean_line(
    tones=[1.0, 5/4, 3/2],
    hits=5,
    steps=8,
    span=0.25,
    amp_db=-14.0,
)
```

### `prob_gate`

Probabilistically filters notes from an existing phrase, preserving original
timing. Notes that survive the gate keep their position; removed notes leave
silence.

```python
prob_gate(
    phrase,
    density=0.7,        # base survival probability [0, 1]
    accent_bias=0.0,    # bias toward keeping louder notes [0, 1]
    position_weights=None,  # per-step weight cycle (e.g. [1, 0.5, 0.8, 0.5])
    seed=0,
)
```

- `density` is the base probability that any note survives.
- `accent_bias > 0` makes louder notes more likely to survive.
- `position_weights` applies a cyclic per-position multiplier to the survival
  probability, useful for emphasizing downbeats or specific metric positions.

```python
from code_musics.generative import prob_gate

sparse = prob_gate(phrase, density=0.5, accent_bias=0.3, seed=7)
```

### `RatioMarkov`

Markov chain over JI ratios with configurable order (memory depth).

Constructors:

- `RatioMarkov.from_transitions({src: {dst: weight, ...}, ...})` -- order-1
  chain from a simple source-to-target mapping; weights are auto-normalized
- `RatioMarkov.from_table({(s1, s2): {dst: weight}, ...}, order=2)` --
  higher-order chain from explicit state tuples
- `RatioMarkov.from_phrase(phrase, order=1, context=None)` -- learn transition
  probabilities from an existing phrase's pitch content

Generation:

- `chain.generate(n, start=None, seed=0)` -- produce `n` ratios; `start` can
  be a single ratio or a tuple matching the chain order
- `chain.to_phrase(n, rhythm, seed=0, start=None, context=None, **line_kwargs)`
  -- generate ratios and build a `Phrase` directly

When the chain reaches a state with no defined transitions, it picks a random
known state and continues.

```python
from code_musics.generative import RatioMarkov

chain = RatioMarkov.from_transitions({
    1.0:  {5/4: 2, 3/2: 1},
    5/4:  {3/2: 1, 1.0: 1},
    3/2:  {1.0: 2, 7/4: 1},
    7/4:  {1.0: 1},
})

ratios = chain.generate(32, seed=11)
phrase = chain.to_phrase(16, rhythm=(0.25,), seed=11, amp_db=-16.0)
```

### `TuringMachine`

Shift-register sequencer inspired by the Music Thing Turing Machine module.
A binary register of configurable length loops through tone selections;
`flip_probability` controls how much the register mutates each step.

- `flip_probability=0.0` produces a fixed loop with period `length`
- Small values introduce gradual mutations
- `flip_probability=1.0` is fully random

```python
TuringMachine(
    length=8,              # register length in bits
    flip_probability=0.0,  # mutation rate [0, 1]
    tones=...,             # TonePool or Sequence[float]
    seed=0,
)
```

- `tm.generate(n)` -- produce `n` ratios
- `tm.to_phrase(n, rhythm, context=None, **line_kwargs)` -- generate and build
  a `Phrase`

```python
from code_musics.generative import TonePool, TuringMachine

pool = TonePool.from_harmonics([4, 5, 6, 7, 9, 11])
tm = TuringMachine(length=6, flip_probability=0.05, tones=pool, seed=3)
phrase = tm.to_phrase(32, rhythm=(0.25,), amp_db=-16.0)
```

### `LatticeWalker`

Random walk on the JI prime-factor lattice. Each step moves one unit along a
randomly chosen prime axis (default: 3, 5, 7), with optional gravitational pull
toward the origin (1/1). Ratios are octave-reduced by default.

```python
LatticeWalker(
    axes=(3, 5, 7),       # prime axes to walk
    step_weights=None,     # {prime: weight} bias toward certain axes
    gravity=0.0,           # pull toward 1/1 [0, 1]
    max_distance=3,        # max exponent magnitude per axis
    octave_reduce=True,    # keep ratios in [1, 2)
    seed=0,
)
```

- `walker.walk(n, start=None)` -- produce `n` ratios; `start` is an optional
  `{prime: exponent}` dict
- `walker.to_phrase(n, rhythm, context=None, start=None, **line_kwargs)` --
  walk and build a `Phrase`

With `gravity=0` the walk is unbiased and can drift far from the origin. Higher
gravity values make the walk orbit closer to 1/1, which tends to produce more
consonant sequences.

```python
from code_musics.generative import LatticeWalker

walker = LatticeWalker(
    axes=(3, 5, 7),
    gravity=0.3,
    max_distance=2,
    seed=5,
)
phrase = walker.to_phrase(24, rhythm=(0.5, 0.25, 0.25), amp_db=-18.0)
```

### `stochastic_cloud`

Generates a cloud of stochastic notes as a `Phrase`. Note start times, pitches,
durations, and amplitudes are all drawn randomly within specified ranges.

```python
stochastic_cloud(
    tones=...,                          # TonePool or Sequence[float]
    duration=10.0,                      # total cloud duration in seconds
    density=5.0,                        # notes per second (float), or
                                        # density breakpoints (see below)
    amp_db_range=(-18.0, -6.0),         # uniform random amp_db range
    note_dur_range=(0.1, 0.5),          # uniform random note duration range
    pitch_kind="partial",               # "partial" or "freq"
    context=None,                       # optional HarmonicContext
    seed=0,
    synth=None,                         # optional per-note synth overrides
)
```

When `density` is a float, it means notes per second (total count =
`density * duration`). When `density` is a sequence of `(time_fraction, rate)`
breakpoints, the density varies over time via piecewise-linear interpolation.
Breakpoints must start at fraction `0.0` and end at `1.0`.

```python
from code_musics.generative import TonePool, stochastic_cloud

pool = TonePool.from_harmonics([4, 5, 6, 7, 9])

# constant density
cloud = stochastic_cloud(tones=pool, duration=8.0, density=4.0, seed=1)

# time-varying density: sparse start, dense middle, sparse end
cloud = stochastic_cloud(
    tones=pool,
    duration=12.0,
    density=[(0.0, 1.0), (0.4, 8.0), (0.7, 8.0), (1.0, 1.0)],
    seed=2,
    amp_db_range=(-20.0, -10.0),
)
```

### Generative Rhythm Tools

These generators produce rhythmic material (RhythmCells and Phrases) from
algorithmic processes. Like the pitch generators above, all are deterministic
when seeded.

#### `prob_rhythm(steps, ...)`

Probabilistic rhythm generation with cycling metric weights.

```python
prob_rhythm(
    steps,                  # number of grid positions
    *,
    onset_weights=0.7,      # per-step onset probability, cycled
    accent_weights=1.0,     # per-step gate/accent, cycled
    span=0.25,              # duration per step (seconds)
    seed=0,
)
```

`onset_weights` cycle if shorter than `steps` -- a 4-element list like
`[1.0, 0.3, 0.5, 0.3]` naturally emphasizes downbeats in a sixteenth grid.
`accent_weights` set the gate (articulation) of surviving onsets. Returns a
`RhythmCell` with at least one onset (if the random draw produces zero
onsets, the first step is forced on).

```python
from code_musics.generative import prob_rhythm

# Metric-weighted sixteenth pattern
rhythm = prob_rhythm(
    16,
    onset_weights=[1.0, 0.3, 0.6, 0.3],
    accent_weights=[1.0, 0.7, 0.85, 0.7],
    span=0.125,
    seed=42,
)
```

#### `AksakPattern`

Additive meter patterns from unequal pulse groups (Balkan, Turkish, etc.).
Each group produces one span equal to `group_size * pulse`.

Fields:

- `grouping: tuple[int, ...]` -- pulse group sizes, e.g. `(3, 3, 2)`
- `pulse: float` -- duration of one pulse unit in seconds

Constructors:

- `AksakPattern(grouping=..., pulse=...)` -- direct
- `AksakPattern.from_timeline(grouping, timeline)` -- derive
  pulse from a Timeline's sixteenth-note duration

Named presets (all take `pulse` as argument):

- `AksakPattern.balkan_7(pulse)` -- 7/8 as 2+2+3
- `AksakPattern.turkish_9(pulse)` -- 9/8 as 2+2+2+3 (zeybek)
- `AksakPattern.take_five(pulse)` -- 5/4 as 3+2 (Brubeck)

Conversion:

- `pattern.to_rhythm()` -- returns a `RhythmCell` with one span
  per group (each span = group_size * pulse).
- `pattern.to_pulses(accent_first=True)` -- expands all pulses as
  individual equal steps. When `accent_first=True`, the first
  pulse of each group gets `gate=1.0` and inner pulses get
  `gate=0.7`. When `False`, all pulses get uniform `gate=1.0`.

```python
from code_musics.generative import AksakPattern

aksak = AksakPattern.balkan_7(pulse=0.15)
rhythm = aksak.to_rhythm()

# From a timeline
from code_musics.meter import Timeline

tl = Timeline(bpm=140, meter=(7, 8))
aksak = AksakPattern.from_timeline((2, 2, 3), tl)
```

#### `ca_rhythm(rule, steps, ...)` and `ca_rhythm_layers(...)`

1D elementary cellular automata as rhythm generators. A CA evolves a row of
cells according to a Wolfram rule number (0--255), and one generation's live
cells become onsets.

**`ca_rhythm(...)`** -- single-layer rhythm from one CA row.

```python
ca_rhythm(
    rule,           # Wolfram rule number (0-255)
    steps,          # width (number of cells / time steps)
    *,
    init=None,      # initial state as bit pattern (None = center cell)
    span=0.25,      # duration per step
    row=-1,         # which generation (-1 = last)
    seed=0,         # used when init=None for random init
)
```

**`ca_rhythm_layers(...)`** -- multiple rows from the same CA evolution as
layered rhythm patterns, picking evenly-spaced rows from the history.

```python
ca_rhythm_layers(
    rule, steps,
    *,
    layers=3,       # number of rhythm layers to extract
    init=None,
    span=0.25,
    seed=0,
)
```

```python
from code_musics.generative import ca_rhythm, ca_rhythm_layers

# Rule 110 -- complex, aperiodic
rhythm = ca_rhythm(110, 16, span=0.125)

# Multi-layer: 3 related patterns for kick / snare / hat
layers = ca_rhythm_layers(30, 16, layers=3, span=0.125)
```

#### `mutate_rhythm(phrase, ...)`

Stochastic variation of an existing phrase's rhythm. Each mutation type
is independent and controlled by its own probability or amount parameter.

```python
mutate_rhythm(
    phrase,
    *,
    add_prob=0.0,        # insert ghost note between events
    drop_prob=0.0,       # remove an event
    shift_amount=0.0,    # max onset shift in seconds
    subdivide_prob=0.0,  # split a note into two at midpoint
    merge_prob=0.0,      # merge with next note (first pitch, combined dur)
    accent_drift=0.0,    # max velocity change per note
    seed=0,
)
```

Apply repeatedly with different seeds for evolving grooves across sections.

```python
from code_musics.generative import mutate_rhythm

# Subtle variation
v1 = mutate_rhythm(groove, shift_amount=0.02, accent_drift=0.1, seed=1)

# More aggressive mutation
v2 = mutate_rhythm(
    groove,
    drop_prob=0.1,
    subdivide_prob=0.15,
    shift_amount=0.03,
    seed=2,
)
```

### Common Patterns

All generators are deterministic for a given `seed`. Changing the seed produces
a different but reproducible result. This makes generative output stable across
renders while still allowing exploration by trying different seeds.

Most generators accept a `HarmonicContext` for ratio resolution. When a context
is provided, ratios are resolved against the context tonic as absolute
frequencies. Without a context, ratios are treated as partials relative to
`Score.f0_hz`.

Generators that produce `Phrase` objects work with the full existing composition
surface: `Score.add_phrase(...)`, `concat(...)`, `overlay(...)`,
`Phrase.transformed(...)`, and all placement transforms.
