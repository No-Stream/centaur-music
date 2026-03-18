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

### Analysis and feedback tooling

We now have a usable render pipeline, but we still need better feedback loops.
Audio plus a piano roll is not enough when iterating on orchestration, density,
and spectral balance, especially for agent-driven workflows.

Useful additions:

- rendered-audio FFT / averaged spectrum views
- spectrograms over time
- coarse band-energy summaries and spectral-tilt diagnostics
- score-derived density, overlap, and registral-spread summaries
- render-to-render comparison views when revising a piece
- analysis manifests that make artifact paths and summary stats easy for agents
  to consume without relying purely on ears

_^^mostly done?_

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

## Medium priority

### Timbre and synth automation

This is now a particularly attractive next step. The engine/preset layer exists,
but most timbral behavior is still static over the life of a note or phrase.

Useful directions:

- time-varying synth parameters at note, phrase, or voice scope
- automation curves for brightness, filter cutoff, resonance, FM index, detune,
  stereo spread, and effect wetness
- phrase-level timbre gestures so sounds can evolve musically without requiring
  low-level parameter plumbing in every piece
- automation-aware analysis so agents can see whether a sound actually opens,
  darkens, widens, or settles the way intended

The main goal is not maximal flexibility for its own sake. It is to make pieces
feel more alive and shaped over time.

### Swing, Humanization
- voices drift together a la Group Humanizer
- simple imperfection in timing
- swing

### Slop and Osc/Env/Etc Drift

- pitch drift at the osc, synth, voice level - not always a great idea in xenharmonic systems, but still useful. (not to be confused with comma drift)  
- cutoff freq etc should offer drift

### Sound design and synthesis direction

The current engine palette is a solid base, but there is room to broaden it
without losing the tuning-first workflow.

Likely useful directions:

- richer additive voices with more role-specific presets
- more FM presets and parameter idioms that interact well with JI materials
- plucked or struck voices for clearer articulation
- noise-plus-tone and hybrid percussion voices
- more explicit role presets for bed, lead, counterpoint, bass, and accent layers

### Utonal and subharmonic writing

The helper functions exist, but there is still a lot of compositional territory
to explore:

- darker subharmonic passages
- overtone / undertone contrasts inside a single form
- comma-drift and undertow gestures that change local harmonic interpretation
  without forcing conventional pitch-class identity
- pitch-motion idioms that make harmonic gravity and arrival bends more audible

### Better effect integration

The declarative effect model is in place, but the palette and routing are still
fairly simple.

Most promising directions:

- chorus, tremolo, autopan, filtering, saturation, and transient shaping - chorus and saturation in particular would do a lot to make sounds lusher and more analog
- role-based effect presets such as `glass_pad`, `sub_drone`, `reed_lead`, or
  `dark_hall`
- better dry/wet and send-style routing so ambience can be shaped more
  deliberately
- **stereo rendering and per-voice panning**: voices currently render mono and
  are summed before any stereo effect (e.g. Bricasti) is applied; per-voice pan
  position would make counterpoint and layered textures much more legible
- **warmth and saturation** (high interest): several strong candidates already
  owned — SPL TwinTube, Black Box HG-2, Arturia True Iron / Pre 1973 / Tape
  MELLO-FI, and free options like Softube Saturation Knob and iZotope Vinyl;
  macOS builds are also available for many of these but are equally unusable
  under Linux; the VST loading problem in WSL2 has several realistic paths:
  - **yabridge**: bridges Windows VST2/VST3 → Linux via Wine; well-maintained,
    widely used in Linux DAW setups; would unlock the full existing Windows
    plugin library from within the normal render pipeline — highest payoff if
    setup succeeds
  - **Chow Tape Model**: free, Linux-native VST3 (and LV2), excellent tape
    saturation character; best "no new infrastructure" option for warmth
  - **WSL2 Windows interop**: WSL2 can call Windows executables directly; a
    thin `render_vst.py` running under Windows Python + pedalboard could apply
    a plugin and write audio back; fragile but zero new dependencies
  - **Python-native waveshaping**: implement tanh soft-clipping and a harmonic
    exciter (2nd/3rd partial boost) directly in `synth.py` as a new `EffectSpec`
    kind; no plugin dependency, deterministic, fits the existing model cleanly;
    less character than hardware emulations but immediately usable

### Rhythm and mixed voice roles

Pieces would benefit from stronger rhythmic identity and clearer timbral role
separation.

Useful directions:

- drum-ish and hybrid percussion layers
- clearer distinctions between harmonic bed, lead, bass/pedal, counterpoint,
  and accent layers
- more traditional song-like structures that still retain meaningful
  xenharmonic character

## Lower priority

### Sound Quality improvements
- aliasing, rendering at higher res and downsampling, etc. - esp for subtractive  
- improved warmth via plugins or otherwise

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
