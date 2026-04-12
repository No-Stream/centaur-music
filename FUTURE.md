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

### Chord smearing and loveless-adjacent ideas

  - on loveless, shields uses repitching to create smearing, ambiguity, reaching, yearning. let's capture some of this.
  - so on a guitar, fragmented chords + global pitch bend (tremolo)
  - interfaces between multiple voices, layering, collaboration
  - relatively simple progressions
  - seconds, sus2/sus4 unresolving, add9, etc.
  - octave duplication of notes for size
  - stereo, chorus, etc., but as bonuses, the core is the score
  - possibly: out of sync, polyrhythmic, weird drums
  - stacked saturation and warmth



### Creative composition helpers

- riemann

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

### Pianos

  **Modal piano implemented.** The `piano` engine now uses modal synthesis with
  physical hammer-string interaction: a nonlinear contact model
  (`F = K * max(delta, 0)^p`) excites a bank of second-order resonators, so
  velocity naturally shapes timbre through the hammer physics. Two-phase
  rendering (Numba JIT contact + vectorized NumPy decay) keeps it tractable.
  Supports unison strings with drift, soundboard coloring, body saturation,
  damper noise, and custom partial ratios for xenharmonic timbre-harmony fusion.
  Eight presets including a septimal variant.

  The legacy additive piano is still available as `piano_additive`.

  Architecture note: the contact simulation's audio output is NOT used directly
  (its coherent peak creates extreme crest factors).  Instead, the simulation
  initializes mode states and the entire audio output comes from the free decay
  with an 8ms cosine fade-in.  Reintroducing contact audio properly would
  require bridge/soundboard absorption modeling to tame the coherent peak.

  Known calibration gaps:

- Upper harmonic content is still low (~1.5% energy above 2 kHz vs 5-15% for a
  real piano).  The 1/k output rolloff + felt filtering + natural 1/ω resonator
  response triple-dips on high-frequency attenuation.  Fixing this properly
  requires compensating for the resonator's frequency-dependent response.
- Unison beating creates ~25 dB AM at ~5 Hz on individual voices.  Real pianos
  have bridge-mediated coupling that dampens beating over time (double decay).
  Adding bridge coupling would help.
- Velocity-dependent timbral variation is subtle — the normalized waveforms
  for soft vs loud hits are very similar.  The hammer contact does change the
  mode excitation, but peak normalization partially masks the effect.

  Follow-up ideas for the modal `piano` engine:

- Bridge/soundboard absorption during contact (enables using contact audio
  in the output for a more realistic attack transient)
- Sympathetic string resonance (cross-note coupling; requires voice-level
  awareness so undamped strings can ring in sympathy with active notes)
- Double decay via bridge coupling (fast initial decay from energy transfer
  to soundboard, slow secondary decay from residual string energy)
- Sustain/una corda pedal modeling
- Feedback delay network (FDN) body model replacing the resonator bank for
  richer, more realistic soundboard response
- Bass longitudinal mode enhancement (phantom partials from nonlinear
  string-bridge coupling in the low register)
- Register-dependent hammer mass/position (bass hammers are heavier and strike
  further from the bridge than treble hammers)
- Frequency compensation for resonator response rolloff (the modal resonators
  naturally attenuate high-frequency modes by ~1/ω; compensating for this
  in the input weights would let the felt filtering alone shape the spectrum)
- Higher-order hammer contact models (wave-digital formulation for improved
  numerical stability at high stiffness)
- Soundboard PDE / measured IR convolution for more realistic body resonance
- Duplex scaling (upper bridge treble resonance adding shimmer to high notes)
- Multi-axis velocity response (velocity modulates decay, damping,
  inharmonicity, hammer position simultaneously -- not just loudness and
  hammer stiffness)
- Prepared piano extensions (muting, objects on strings -- mute_position,
  mute_amount, extra inharmonic partial layers from bolts/screws)

  Follow-up ideas for `piano_additive` (legacy engine):

- Register-dependent spectral templates (different partial profiles for
  bass/mid/treble)
- Pitch-scaled attack duration (higher notes get proportionally faster attacks)
- Per-note soundboard coupling

### Pianoteq

  I have a Pianoteq license and the Linux package downloaded. We can try setting it up headless.
  (Pianoteq is famous for microtonal support but we'll need to hack around a bit to hopefully get it running without a DAW.)
  (Surge XT instrument hosting is now working via pedalboard — the VSTi
  hosting path is proven. Pianoteq would use the same mechanism.)

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

I suspect our saturation/warmth native plugin could be better. Currently sounds fuzzy and probably aliased. (not sure if this is current.)

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

### Harpsichord — Implemented

**The `harpsichord` engine is now implemented** with pluck-excitation + modal
resonator synthesis, four-register blending (front 8', back 8', 4', lute),
per-note spectral morphing, velocity expression, and custom `partial_ratios`
for xenharmonic tuning. Seven presets including `septimal` and `glass`.
Voice-level sympathetic resonance is also available (engine-agnostic).

Follow-up ideas:

- Karplus-Strong / waveguide alternative synthesis path (different timbral
  character, more natural pluck sustain, but harder to integrate custom
  partial ratios — would need a hybrid approach)
- Additive synthesis with pluck envelopes (lighter computation, full spectral
  control, but less physically grounded than the modal approach)
- Coupling / sympathetic resonance between registers (currently registers are
  independent; bridge coupling would let them interact)
- Extended register palette: 16' sub-octave, nasalized/reed-like fantasy stops
- Buff stop physical modeling (currently approximated with decay/brightness
  scaling)

### DawDreamer

DawDreamer is a Python DAW framework with full MIDI + instrument hosting,
Faust DSP compilation, RubberBand time-stretch, and complex routing graphs.
It is GPLv3 — same license family as pedalboard (already a dep) — but the
project prefers MIT for its own code, so evaluate carefully before adopting.
Main value-adds beyond current pedalboard usage: Faust DSP compilation for
rapid synth/effect prototyping, and CLAP plugin hosting. Consider if/when
we need capabilities pedalboard cannot provide.

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

### Deferred generative ideas

Ideas discussed but deferred during the generative toolkit build:

- Combination Product Sets (Erv Wilson) — hexanies, dekanies, eikosanies as
  harmonic vocabularies feeding TonePool and other generators
- Comma pump generator — auto-generate chord progressions that drift by a
  specified comma per cycle
- Phase process helpers — parameterized Steve Reich-style gradual phase shifting
- L-systems — Lindenmayer systems mapped to pitch/rhythm for fractal structures
- Process pipelines — composable generator chaining for combining generators
- Cellular automata — 1D CA rules mapped to musical parameters

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
