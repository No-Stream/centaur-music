# Composition API Reference

This document describes the higher-level composition helpers that sit above the
core `Score` / `Phrase` / `NoteEvent` model.

The composition layer is phrase-first and xen-friendly:

- it builds ordinary `Phrase` objects
- it keeps timing in seconds rather than MIDI/grid assumptions
- it supports ratio-aware pitch motion for JI and harmonic-series writing

## Where This Is Used

- [code_musics/composition.py](/home/jan/workspace/code-musics/code_musics/composition.py)
- [code_musics/pitch_motion.py](/home/jan/workspace/code-musics/code_musics/pitch_motion.py)
- [code_musics/score.py](/home/jan/workspace/code-musics/code_musics/score.py)

## Core Types

### `RhythmCell`

Stores onset spans and optional gate values.

- `spans: tuple[float, ...]`
- `gates: float | tuple[float, ...] = 1.0`

`spans` define when each note starts relative to the next onset. `gates` define
how much of each span is actually sounded.

- `gates < 1.0` gives clipped/staccato phrasing
- `gates = 1.0` fills the full span
- `gates > 1.0` creates overlap and legato smear

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

## `line(...)`

`line(...)` is the main phrase builder.

It accepts:

- a tone sequence
- a `RhythmCell` or onset spans
- `pitch_kind="partial"` or `pitch_kind="freq"`
- `amp_db` for perceptual level-setting, or legacy linear `amp`
- optional `ArticulationSpec`
- optional pitch motion, either scalar or per-note

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

## Helper Transforms

These all return new phrases and do not mutate the source:

- `with_gate(...)`
- `with_accent_pattern(...)`
- `with_tail_breath(...)`
- `staccato(...)`
- `legato(...)`
- `echo(...)`

Use them when a phrase is already built and you want a fast re-articulation pass
without rewriting the note list.

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
