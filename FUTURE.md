# Future Work

This file tracks what still feels highest-value from the current state of the
project. Core infrastructure such as the `Score` / `Voice` / `Phrase` model,
named-piece rendering, piano-roll plotting, multiple synth engines, presets,
articulation helpers, and note-level pitch motion already exist.

## High priority

### More complete pieces

The main artistic question is still the same: can we make xenharmonic music
that feels like full composition rather than only study, sketch, or demo?

Most valuable directions:

- stronger large-scale form
- more memorable motifs and thematic return
- clearer contrast, release, and pacing across sections
- better balance between "pleasant" and "strange"
- more deliberate orchestration so pieces feel arranged, not merely layered

### Better piece-generation tools

The score abstraction is in good shape, but composition could still become much
faster through a richer helper layer.

Most valuable next helpers:

- phrase transformation helpers beyond current placement options
- idiomatic generators for canons, pedals, blooms, echoes, and harmonic-series
  gestures
- utility functions for voice-leading inside otonal and utonal spaces
- gesture builders such as `converge`, `diverge`, `spiral`, `comma_drift`,
  `harmonic_shadow`, and `undertow`
- sectional/context drift helpers that re-realize ratio material against local
  tonics without introducing fixed pitch-class identities
- overlap and beating summaries so we can spot where a piece has become too
  static, too drony, or too crowded before full render

_todo: update, think we added some of this_

### Sound design and synthesis direction

The current engine palette is a solid base, but there is room to broaden it
without losing the tuning-first workflow.

Likely useful directions:

- richer additive voices with more role-specific presets
- more FM presets and parameter idioms that interact well with JI materials
- plucked or struck voices for clearer articulation
- noise-plus-tone and hybrid percussion voices
- more explicit role presets for bed, lead, counterpoint, bass, and accent layers

## Medium priority

### Timbre and synth automation

This is now a particularly attractive next step. The engine/preset layer exists,
but most timbral behavior is still static over the life of a note or phrase.

Useful directions:

- time-varying synth parameters at note, phrase, or voice scope
- extend the new automation surface beyond synth params into pan, gain, plugin
  params, effect wetness, and other mix controls
- phrase-level timbre gestures so sounds can evolve musically without requiring
  low-level parameter plumbing in every piece
- automation-aware analysis so agents can see whether a sound actually opens,
  darkens, widens, or settles the way intended

The main goal is not maximal flexibility for its own sake. It is to make pieces
feel more alive and shaped over time.

### Utonal and subharmonic writing

The helper functions exist, but there is still a lot of compositional territory
to explore:

- darker subharmonic passages
- overtone / undertone contrasts inside a single form
- comma-drift and undertow gestures that change local harmonic interpretation
  without forcing conventional pitch-class identity
- pitch-motion idioms that make harmonic gravity and arrival bends more audible

### JI drift harmony and recontextualization

The current context helpers cover the basics, but there is still room for more
musically direct tooling around ratio-space reinterpretation.

Useful directions:

- stronger phrase-level recontextualization helpers
- comma-drift movement that is easier to author as normal compositional material
- voice-leading helpers that stay elegant while local tonics shift
- sectional harmonic reinterpretation without falling back to fixed pitch-class
  identity

### Better effect integration

The declarative effect model is in place and already supports a useful core:
delay, algorithmic reverb, Bricasti IR convolution, native saturation, Chow
Tape, chorus, stereo-aware effect chains, and per-voice pan. The next step is
less "invent effects from scratch" and more "round out the palette so mixes can
be shaped deliberately inside the existing render path."

Most promising directions:

- EQ and compression as first-class mix tools; these are now the most obvious
  gaps. (think we have these now but not used via LSP linux studio plugs?)
- richer modulation and shaping such as tremolo, autopan, filtering, and
  transient shaping
- role-based effect presets such as `glass_pad`, `sub_drone`, `reed_lead`, or
  `dark_hall`
- better dry/wet and send-style routing so ambience can be shaped more
  deliberately
- plugin-backed EQ / glue / color inside the current `pedalboard`-based
  pipeline, with a bias toward Linux-native VST3 or LV2 before chasing
  Windows-only favorites through bridges

#### Plugin shortlist

These are the most attractive near-term plugin candidates to widen the sound
without opening a large tuning or host-integration project:

- **LSP Plugins**: highest-value utility bundle; broad Linux-native coverage for
  parametric EQ, compression, multiband control, and convolution
- **x42 EQ / x42 Compressor**: lean, Linux-native workhorse pair if a smaller
  set of mix staples is preferable to a huge suite
- **Airwindows Consolidated**: very attractive for subtle console-ish color,
  tape-ish sweetening, and low-friction analog glue
- **Bandbreite**: another saturation color to complement Chow Tape rather than
  replace it
- **TAL-Chorus-LX**: classic analog-style widening/polish that could be useful
  even with the current native chorus available
- **Dragonfly Reverb** and/or a TAL reverb: broader ambience palette beyond the
  existing built-in reverb and Bricasti IR path

Suggested install / evaluation order:

- EQ + compressor first: LSP or x42
- color / glue next: Airwindows, then Bandbreite
- extra modulation / space after that: TAL-Chorus-LX, Dragonfly, TAL reverb

WSL2 / Linux plugin-hosting notes:

- Prefer Linux-native VST3 or LV2 first; that is the least annoying path
- **yabridge** is still the best "unlock the Windows plugin folder later"
  option if native Linux choices prove insufficient but sadly no WSL2 support
- a Windows-side render helper remains possible, but should be treated as a
  fallback rather than the default architecture

_todo: some of these are added, need to update this section_

#### Synth plugins

Backlog rather than near-term priority.

Interesting Linux-friendly synths may exist, but the xenharmonic workflow makes
them a separate project because proper tuning support would likely require
pitch-bend batching, MPE, MTS-ESP, or some other tuning-aware host strategy.
Effects give much higher payoff right now with less integration risk.

### Rhythm and mixed voice roles

Pieces would benefit from stronger rhythmic identity and clearer timbral role
separation.

Useful directions:

- drum-ish and hybrid percussion layers
- clearer distinctions between harmonic bed, lead, bass/pedal, counterpoint,
  and accent layers
- more traditional song-like structures that still retain meaningful
  xenharmonic character

### Analysis and feedback tooling

We now have a usable render pipeline and the core analysis/artifact path is in
good shape, so this is no longer urgent. It is still worth improving when the
composition loop starts to feel bottlenecked again.

Useful additions:

- richer render-to-render comparison views
- clearer summaries for orchestration, density, and spectral balance
- analysis manifests that are even easier for agents to consume
- selective improvements to plots or summary diagnostics when they materially
  improve iteration speed

### Ladder filter for subtractive voices

Useful later once the current analog-style filter path has settled:

- add a ladder-style low-pass character filter for `polyblep` or a related
  subtractive engine
- bias toward musical resonance and tone color rather than maximal circuit
  realism
- treat this as a distinct flavor on top of the current ZDF/TPT filter, not a
  replacement

## Lower priority

### Swing, Humanization
- voices drift together a la Group Humanizer
- simple imperfection in timing
- swing

### Slop and Osc/Env/Etc Drift

- pitch drift at the osc, synth, voice level - not always a great idea in xenharmonic systems, but still useful. (not to be confused with comma drift)
- cutoff freq etc should offer drift

### Sound Quality improvements
- aliasing, rendering at higher res and downsampling, etc. - esp for subtractive  
- improved warmth via plugins or otherwise
- paid Linux-capable effects already owned, such as `u-he Satin` and
  `u-he Presswerk`, are still attractive but should stay low-priority until a
  low-friction activation path under WSL2/Linux is in place

### Parametric piece generation

Generate families of pieces from parameters like root, prime limit, or chosen
partials, mainly as a composition aid rather than a replacement for hand-written
work.

### Microtonal MIDI export

Export pieces to DAW-friendly MIDI with pitch bend so ideas can move more easily
into other production environments.

### Identity-preserving pitch drift

Low priority, and likely optional if it ever exists.

- exploratory system for notes that retain some tuning memory across sections
- should not require fixed note names, scale degrees, or conservative
  music-theory identities
- contextual drift helpers are the preferred near-term direction; this would be
  for stranger long-memory tuning behavior later, if it proves musically useful

### Granular synthesis

A granular layer could open up a nice middle ground between harmonic writing and
texture design, especially for transitions or atmospheric sections.
