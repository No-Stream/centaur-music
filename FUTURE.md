# Future Work

This file is for ideas that are still ahead of us. Core infrastructure like the
`Score` / `Voice` / `Phrase` model, piano-roll plotting, named-piece rendering, and
basic tuning helpers are already implemented.

## High priority

### More complete pieces

The big musical question is still the same: can we make xenharmonic music that feels
like full composition rather than only demonstration or experiment?

Areas to push:

- stronger large-scale form
- more memorable motifs and thematic return
- clearer contrast and release across sections
- better balance between "pleasant" and "strange"

### Better piece-generation tools

We now have a usable score abstraction, but we could still make composition faster by
adding more reusable building blocks:

- phrase transformation helpers beyond the current placement options
- idiomatic generators for canons, pedals, blooms, and harmonic-series gestures
- utility functions for voice-leading inside otonal and utonal spaces
- lightweight analysis helpers for inspecting phrase density, registral spread, and
  timing

### Stronger visual analysis

The piano roll exists today, but richer analysis views would help both composition and
agentic iteration:

- FFT plots for rendered excerpts
- spectrograms over time
- interval and partial-distribution summaries
- render-to-render comparison views when revising a piece

## Medium priority

### Sound design and synthesis direction

The current additive synth is a good base, but it is too narrow to carry the whole
project. We should broaden the palette without losing the tuning-first workflow.

Preferred direction:

- keep the `Score` / `Voice` model as the composition layer
- support multiple synth engines behind a consistent note / voice interface
- prefer small, dependable building blocks over a giant custom DSP system
- use libraries where they save real time and complexity
- stay open to freeware VSTs or existing plugins when they genuinely expand the sound
  world, but avoid making fragile plugin hosting the core of the project

Likely useful voice / engine types:

- richer additive voices with detune, partial weighting, brightness tilt, and stereo spread
- FM voices, especially ratio-based operator setups that interact well with JI material
- plucked or struck voices for clearer attacks and more articulated lines
- noise-plus-tone voices for drum-ish and hybrid percussive sounds
- simple pad / lead / bass role presets so pieces can differentiate musical function

FM synthesis still looks especially promising here. Rational operator ratios could mesh
well with JI materials, while more inharmonic ratios could support the stranger
sections without abandoning the tuning focus.

### Utonal and subharmonic writing

We already have helper functions, but there is more compositional territory to explore:

- darker subharmonic passages
- otonal / utonal contrasts inside a single form
- more deliberate harmonic motion between overtone and undertone worlds

### Better effect integration

The declarative effect model is in place, but there is room to improve the palette:

- richer built-in ambience and modulation options
- cleaner support for external tools or VST-style workflows where practical
- more repeatable effect presets tied to musical roles

Specific directions worth exploring:

- chorus, ensemble, tremolo, autopan, filtering, saturation, and transient shaping
- effect presets tied to musical jobs such as `glass_pad`, `sub_drone`, `reed_lead`,
  `bell_fm`, `soft_room`, or `dark_hall`
- better dry/wet and send-style routing so ambience can be shaped more deliberately
- support for render-time comparison when changing effect chains

### Libraries and plugin strategy

We should avoid reinventing the wheel when existing tools can give us more interesting
results quickly.

Practical default:

- first prefer Python-accessible libraries and stable native dependencies
- keep the built-in synth and effect path simple and deterministic
- treat plugin hosting as an expansion path, not the first dependency

Reasons:

- native Python / library-based rendering is easier to test and less brittle in WSL2
- plugin discovery, licensing, GUI requirements, and platform quirks can make VST
  workflows unreliable
- paid plugins may not load cleanly in this environment, so we should not depend on
  them for core rendering

Still, VSTs are worth exploring where they unlock a lot:

- freeware synths or effects that bring strong timbral value
- Windows-installed plugins that can be rendered offline if the bridge is reliable
- external rendering paths that complement the Python toolchain instead of replacing it

Possible implementation path:

- phase 1: expand the native synth / effect palette inside Python
- phase 2: add a thin abstraction for external or plugin-backed renderers
- phase 3: experiment with a curated set of freeware VSTs that are likely to work well
  under the current environment

Selection criteria for any external tool:

- renders offline and repeatably
- works without a fragile interactive GUI dependency
- is scriptable enough to fit the current workflow
- adds genuinely new musical value rather than duplicating what a simple built-in tool
  can already do

### Rhythm

Generate drum-ish sounds and add some rhythm

### Mixed Voice Types

Possibly via libraries or plugins, we can have e.g. chords/keys/pads/leads/basses etc.
Not necessarily tied to traditional instrument roles, but perhaps some differentiation.

A useful framing here is not strict instrument emulation, but clearer timbral roles:

- sustained harmonic bed
- lyrical lead
- articulated counterpoint voice
- bass / pedal anchor
- percussive or noisy accent layer

That should make pieces sound more intentionally orchestrated even when the materials
stay obviously synthetic and xenharmonic.

### More traditional songs

Working with chords, chord progressions, leads, basses, and rhythms, but with 
nontrivial xenharmonic character.

## Lower priority

### Parametric piece generation

Generate families of pieces from parameters like root, prime limit, or chosen
partials, mainly as a composition aid rather than a replacement for hand-written work.

### Microtonal MIDI export

Export pieces to DAW-friendly MIDI with pitch bend so ideas can move more easily into
other production environments.

### Granular synthesis

A granular layer could open up a nice middle ground between harmonic writing and
texture design, especially for transitions or atmospheric sections.

---

## WiP / To Continue

- sketch spiral arch  
- 
