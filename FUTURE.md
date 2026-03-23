# Future Work

This file tracks what still feels highest-value from the current state of the
project, and distinguishes that from capabilities that already exist.

The repo is no longer at the "basic scaffolding" stage. We already have:

- a solid `Score` / `Voice` / `Phrase` / `NoteEvent` composition model
- phrase-first composition helpers such as `line`, `ratio_line`, `concat`,
  `overlay`, `echo`, `sequence`, `canon`, `voiced_ratio_chord`, and
  `progression`
- multiple synth engines with presets: additive, FM, filtered-stack, polyBLEP,
  and noise/percussion
- render-time expression surfaces including note `velocity`, note
  `pitch_motion`, score-level timing humanization, voice-level envelope and
  velocity humanization, velocity-to-parameter mapping, and score/note
  automation
- a declarative effect chain with native EQ, native compressor, saturation,
  delay, algorithmic reverb, Bricasti convolution, stereo-aware routing, and a
  working Linux plugin path through `pedalboard`
- snippet rendering, exact-window rendering, timestamp inspection, timeline JSON
  artifacts, analysis manifests, and drift/artifact-risk diagnostics
- named pieces and studies substantial enough that the bottleneck is now more
  often musical direction than missing infrastructure

So the main question is no longer "can this system render xenharmonic music at
all?" It can. The question is how to turn the current toolkit into stronger
music, faster iteration, and a broader but still coherent sonic language.

## High priority

### More complete pieces

This is still the top priority.

The central artistic question remains: can we make xenharmonic music that feels
like full composition rather than only study, sketch, or demo?

Most valuable directions:

- stronger large-scale form
- more memorable motifs and thematic return
- clearer contrast, release, pacing, and arrival across sections
- better balance between "pleasant" and "strange"
- more deliberate orchestration so pieces feel arranged, not merely layered
- more works that feel finished enough to revisit, compare, and refine rather
  than discard after one experiment

### Better piece-generation and phrase-writing tools

The composition layer is now real and useful, but it can still become much
faster and more musical.

Most valuable next helpers:

- richer phrase transformation helpers beyond the current set
- more idiomatic generators for pedals, blooms, suspensions, staggered entries,
  harmonic-series gestures, and contrapuntal restatement
- voice-leading helpers for otonal and utonal spaces that stay readable rather
  than turning into tuning math soup
- higher-level section builders that make it easier to restate material with new
  orchestration, local tonic reinterpretation, or registral shift
- overlap, beating, density, and register summaries that better flag when a
  texture has become too static, too crowded, or too continuously drony

### Creative composition helpers
- riemann
- random notes in a scale
- turing machines
- probability gates + add proba into composition helpers as a first-class citizen
- euclidean rytms + synths (not just rhythms)
- probability dists (well PMFs) over notes in a scale
- markov processes / markov chains - with and without memory

### Additive Synthesis Specifically
  Additive lets us not assume traditional harmonic structure. 
  e.g. we can have harmonics of just intervals, not e.g. simple sawtooth harmonics
  this should unlock some interesting creative opportunities.
  for example - 
  1. Bell / mallet / metal / glass / struck-material sounds
  These are already naturally inharmonic or quasi-inharmonic, so they pair beautifully with unusual tunings. Xenharmonic music often feels more convincing when the timbre does not keep insisting on a standard harmonic ladder.
  2. Drones and spectral harmony
  If you want harmony to emerge from partial relationships rather than chord symbols, additive is almost ideal. You can sculpt consonance directly in the spectrum.
  3. Pads with “non-Western” or ambiguous tonal centers
  A subtractive pad often still sounds like “a normal synth pad but retuned.” An additive pad can sound like it came from a different musical physics.
  4. Timbre-harmony fusion
  You can make a chord whose note frequencies and internal overtone structures are both drawn from the same ratio world.
  We should consider how to do this with a clean, musical interface that encourages sane defaults and easy programming, and modularity.
  Remaining work after the new explicit spectral-partial additive voice:
  - richer stereo-from-spectrum tools so different partial groups can occupy subtly different widths/positions
  - automatic register-aware darkening / brightness limiting so high notes do not get brittle and low notes do not get muddy
  - controlled inharmonic stretch and physical-object style detuning beyond the current gentle upper-partial drift
  - deeper programmable per-partial envelopes and modulation, once the simple onset/sustain morph proves musically useful
  - a broader role-oriented preset family for spectral additive voices, beyond the first JI / septimal / 11-limit / utonal set
  _implemented some of this, should revise this section_

### MIDI Export

- see email note to self. plan around mostly scl/tun + Oddsound w/ polyphonic bend for extra synth compatibility

### Sound design and synthesis direction

The engine palette is strong enough to compose with now; the next step is
making it broader and more role-aware rather than merely adding more knobs.

Most promising directions:

- richer additive presets with clearer arrangement roles
- more FM presets and FM parameter idioms that interact musically with JI
  material. _subtle ones_ that work with harmonies and don't shout
- stronger plucked, struck, and hybrid percussive voices
- more explicit role presets for bed, lead, counterpoint, bass, and accent
  layers
- more preset curation around "musical default sounds" rather than only
  technical engine coverage

---

## Medium priority

### Slop, swing, and drift extensions

  Some of this is implemented already through timing, envelope, and velocity
  humanization; what remains is the more specialized layer.
  We recently added notes (rather than just time-based) arranging.
  And we have humanization + imperfection. But we should eventually add swing.

### Polyrhythm/polymeter

  To some extent we already support this implicitly. Worth expanding?

### Autoresearch, for music

  Automatically generate pieces and critique them, prune, develop.  
  Challenging given current-gen agents can't hear, but that's a challenge for this entire project.

### Timbre and mix automation

  Automation now exists, but it is still an intentionally limited v1 surface.

  Useful next steps:

  - extend automation beyond the current synth-param and `pitch_ratio` targets
    into pan, gain, effect wetness, and plugin parameters
  - phrase-level timbre gestures so sounds can evolve musically without low-level
    automation plumbing in every piece
  - more reusable automation idioms for opening, darkening, widening, blooming,
    and settling
  - stronger analysis feedback so we can verify whether a sound actually opens,
    softens, or narrows the way intended

### Utonal, subharmonic, and drift-based harmony

  The helper layer exists, but there is still a lot of compositional ground left
  to cover.

  Most interesting directions:

  - darker subharmonic passages that feel structurally intentional, not just novel
  - stronger overtone / undertone contrasts inside a single form
  - phrase-level recontextualization helpers that make comma drift and local tonic
    reinterpretation easier to write as normal music
  - voice-leading idioms that stay elegant while harmonic context shifts

### Trance, progressive house
  fun genres to explore alt tunings in, since they're so reliant on harmony
  let's get slightly cheesy

### Colundi-ish Scale
  ```
  ! colundi_ji_core.scl
  !
  Approximate Colundi-inspired 7-note JI scale
  7
  !
  11/10
  19/16
  4/3
  3/2
  49/30
  7/4
  2/1
  ```

### Organs!
  Organs sound great, are fundamentally additive ish, should be alt-tuning friendly are huge in Bach.
  Let's try making some organ models and playing them!

### Pianoteq
  I have a Pianoteq license and the Linux package downloaded. We can try setting it up headless. 
  (Pianoteq is famous for microtonal support but we'll need to hack around a bit to hopefully get it running without a DAW.)

### Co-design scale + synth
  - Additive synthesis + scale. works great with non-octave scales, e.g. Colundi (since we can omit octave harmonics)
  - FM, similar story
  - Physical modeling: we can use physical models with _harmonics that are compatible with our scale_. e.g. Aleksi Perala does very cool xen bells
  - Granular (not necessarily _great_ for xenharmonic but doesn't have the octave bias of subtractive synths)

### Combination Product Set - Harmonic Lattice (Erv Wilson)
  - cool idea. let's try.

### Non-octave-privileged distortion
  Traditional distortion/saturation relies on octave-based products. 
  Could we do something interesting with _other_ multiples?! Aligned with our scale.

### Better effect integration and routing

The effect story is much better than it used to be, so this is no longer about
basic availability. It is about making the mix path more deliberate and easier
to shape.

Most promising directions:

- richer modulation and shaping such as tremolo, autopan, filtering, and
  transient emphasis
- role-based effect presets such as `glass_pad`, `sub_drone`, `reed_lead`, or
  `dark_hall`
- better dry/wet and send-style routing so ambience can be shaped more
  intentionally
- continued plugin-backed EQ / glue / color exploration with a bias toward
  stable Linux-native VST3 or LV2 paths

#### Filter drive and saturation

I suspect our saturation/warmth native plugin could be better. Currently sounds fuzzy and probably aliased.

#### Plugin reliability follow-up

The recent plugin work surfaced issues around cached state and repeatable
render behavior. Some of that has been fixed, but reliability is still a real
follow-up area.

Most valuable next steps:

- reproduce plugin-specific artifacts in minimal fixtures when they show up in
  real pieces
- define a tighter test strategy for repeated renders, reset behavior, and
  on/off toggles
- sharpen the debugging split between host bugs, plugin bugs, and our parameter
  mapping bugs
- decide more explicitly when native effects should be preferred over external
  plugins for stability

### Analysis and feedback tooling

Analysis is now useful and part of the normal workflow. Future work here should
stay focused on iteration speed, not on building an in-repo DAW.

Useful additions:

- richer render-to-render comparison views
- clearer summaries for orchestration, density, register use, and spectral
  balance
- analysis outputs that are even easier for agents to consume directly
- selective visual improvements when they materially speed up listening and
  revision loops

### Repo structure follow-up

The repo is in better shape than before, but a few architectural cleanups still
look worthwhile:

- split `code_musics/render.py` further into render orchestration vs artifact
  metadata helpers if it keeps growing
- tighten package API boundaries so internal modules rely less on compatibility
  surfaces
- decide whether the repo-root `synth.py` compatibility wrapper should stay,
  move, or go away
- consider whether generated `output/` artifacts should keep living in the repo
  root by default or move behind a clearer workspace/output boundary

### Wavetable engine
- allows unique timbres, somewhat complicated, needs to be done right (aliasing etc)

---

## Lower priority

### Rhythm and clearer voice roles

Still worthwhile, but not blocked on infrastructure:

- stronger rhythmic identity in more pieces
- more drum-ish and hybrid percussion layers
- clearer distinctions among harmonic bed, lead, bass/pedal, counterpoint, and
  accent layers
- more song-like or suite-like structures that still retain meaningful
  xenharmonic character

### Additional subtractive color

Useful later once the current filter palette has settled:

- a ladder-style low-pass flavor for `polyblep` or a related subtractive engine
- potential other analog-inspired filters like SEM
- more emphasis on musicality (analog influence can be useful but not required)
- treat this as an additional flavor, not a replacement for the current filter
  path

Possible later additions:

- swing-oriented helpers where it actually serves the music
- more correlated ensemble behavior across voices for specific groove feels
- selective synth-parameter drift such as cutoff drift or mild oscillator drift
  where it helps rather than muddies tuning clarity

### Sound-quality refinement

The project already sounds good enough to make real musical decisions with.
These items matter, but they should stay behind composition and tooling wins.

Likely later directions:

- further anti-aliasing and oversampling/downsampling improvements for brighter
  subtractive material
- more "warmth" and glue through either better native processing or carefully
  chosen plugins
- eventual use of paid Linux-capable effects already owned, such as `u-he
  Satin` and `u-he Presswerk`, once the activation path is low-friction enough
