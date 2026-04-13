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

- `gates < 1.0` gives clipped/staccato phrasing
- `gates = 1.0` fills the full span
- `gates > 1.0` creates overlap and legato smear

## Musical-Time Layer

Use `code_musics.meter` when you want musical time rather than raw seconds.

Core APIs:

- `Timeline(bpm=..., meter=(num, den), swing=...)`
- rhythmic values: `W`, `H`, `Q`, `E`, `S`
- swing helper: `SwingSpec.eighths(...)`, `SwingSpec.sixteenths(...)`
- helpers: `B(...)`, `M(...)`, `dotted(...)`, `triplet(...)`

Example:

```python
from code_musics.composition import grid_line, grid_sequence
from code_musics.meter import M, Q, SwingSpec, Timeline

timeline = Timeline(
    bpm=96,
    meter=(4, 4),
    swing=SwingSpec.eighths(0.62),
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
- `SwingSpec(..., offbeat_position=0.5)` is accepted as explicit straight feel,
  though `swing=None` is still the cleaner default

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

If `swing` is set, `Timeline.position(...)`, `Timeline.at(...)`, and
`Timeline.locate(...)` use the swung grid. Standalone scalar durations like
`timeline.duration(Q)` remain straight-time conversions; swing-aware note spans
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

Note-level pitch motion attached to a `NoteEvent`.

Available constructors:

- `PitchMotionSpec.linear_bend(...)`
- `PitchMotionSpec.ratio_glide(...)`
- `PitchMotionSpec.vibrato(...)`

Prefer `ratio_glide(...)` when you want motion that stays clearly grounded in
ratio space.

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
    f0=110.0,
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
- `attack_amount_pct`
- `decay_amount_pct`
- `sustain_amount_pct`
- `release_amount_pct`
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
- `staccato(...)`
- `legato(...)`
- `echo(...)`
- `concat(...)`
- `overlay(...)`

Use them when a phrase is already built and you want a fast re-articulation pass
without rewriting the note list.

`echo(...)` now supports both pitch surfaces:

- use `partial_shift` for partial-authored phrases
- use `freq_scale` for frequency-authored phrases

If you try to use `partial_shift` on a frequency-authored phrase, it raises
instead of silently doing nothing.

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

### Common Patterns

All generators are deterministic for a given `seed`. Changing the seed produces
a different but reproducible result. This makes generative output stable across
renders while still allowing exploration by trying different seeds.

Most generators accept a `HarmonicContext` for ratio resolution. When a context
is provided, ratios are resolved against the context tonic as absolute
frequencies. Without a context, ratios are treated as partials relative to
`Score.f0`.

Generators that produce `Phrase` objects work with the full existing composition
surface: `Score.add_phrase(...)`, `concat(...)`, `overlay(...)`,
`Phrase.transformed(...)`, and all placement transforms.
