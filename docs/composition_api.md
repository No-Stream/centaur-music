# Composition API Reference

This document describes the higher-level composition helpers that sit above the
core `Score` / `Phrase` / `NoteEvent` model.

For the concrete score-domain reference covering `Score`, `Voice`, `NoteEvent`,
`Phrase`, and render-time expression semantics, see
[docs/score_api.md](/home/jan/workspace/code-musics/docs/score_api.md).

The composition layer is phrase-first and xen-friendly:

- it builds ordinary `Phrase` objects
- it keeps timing in seconds rather than MIDI/grid assumptions
- it supports ratio-aware pitch motion for JI and harmonic-series writing

There is now also an optional high-level musical-time layer for authoring in
beats and bars. It compiles back down to the same `Phrase` / `Score` surfaces,
so the underlying render model is still seconds-based.

## Where This Is Used

- [code_musics/composition.py](/home/jan/workspace/code-musics/code_musics/composition.py)
- [code_musics/pitch_motion.py](/home/jan/workspace/code-musics/code_musics/pitch_motion.py)
- [code_musics/score.py](/home/jan/workspace/code-musics/code_musics/score.py)

## Core Types

This doc touches a few nearby score-expression controls for context, but the
detailed score-domain reference now lives in
[docs/score_api.md](/home/jan/workspace/code-musics/docs/score_api.md).

Nearby topics that matter when using the composition helpers:

- note-level `amp_db` and `velocity`
- score-level timing humanization
- voice-level envelope humanization
- voice-level velocity humanization and `velocity_group`
- velocity-driven synth parameter mapping

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
    synth_defaults={"engine": "filtered_stack", "preset": "round_bass"},
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
- `chord_spread_ms` offsets notes that start together so simultaneities are not perfectly vertical

Use this when the whole score should feel performed rather than grid-perfect.
Do not use it as a substitute for writing actual rubato or changing the written rhythm.

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
- bright percussion or punctuation: often quieter in sustain, but transient material may still need `-14 dB` to `-8 dB`

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
- avoid treating velocity like a substitute for all gain staging; use `amp_db` for the larger balance decisions

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
