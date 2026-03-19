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

### Sound design and synthesis direction

The engine palette is strong enough to compose with now; the next step is
making it broader and more role-aware rather than merely adding more knobs.

Most promising directions:

- richer additive presets with clearer arrangement roles
- more FM presets and FM parameter idioms that interact musically with JI
  material
- stronger plucked, struck, and hybrid percussive voices
- more explicit role presets for bed, lead, counterpoint, bass, and accent
  layers
- more preset curation around "musical default sounds" rather than only
  technical engine coverage

## Medium priority

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

#### Plugin reliability follow-up

The recent plugin work surfaced real issues around cached state and repeatable
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
- more emphasis on musical resonance and color than on maximal analog modeling
- treat this as an additional flavor, not a replacement for the current filter
  path

### Slop, swing, and drift extensions

Some of this is implemented already through timing, envelope, and velocity
humanization; what remains is the more specialized or riskier layer.

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
