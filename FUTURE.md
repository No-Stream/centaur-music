# Future Work

This file tracks what still feels highest-value from the current state of the
project, and distinguishes that from capabilities that already exist.

The repo is no longer at the "basic scaffolding" stage. We already have:

- a solid `Score` / `Voice` / `Phrase` / `NoteEvent` composition model
- phrase-first composition helpers such as `line`, `ratio_line`, `concat`,
  `overlay`, `echo`, `sequence`, `canon`, `voiced_ratio_chord`, and
  `progression`
- multiple synth engines with presets: additive, FM, filtered-stack, polyBLEP,
  noise/percussion, organ, piano, harpsichord, kick_tom, and surge_xt
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
  Explicit spectral partial sets, onset-to-sustain spectral morphing, and the
  first JI/septimal/11-limit/utonal role-oriented presets are all implemented.
  Remaining work:

- richer stereo-from-spectrum tools so different partial groups can occupy subtly different widths/positions
- automatic register-aware darkening / brightness limiting so high notes do not get brittle and low notes do not get muddy
- controlled inharmonic stretch and physical-object style detuning beyond the current gentle upper-partial drift
- deeper programmable per-partial envelopes and modulation beyond the current onset/sustain morph
- a broader role-oriented preset family expanding beyond the current JI / septimal / 11-limit / utonal set
- **Vital-style spectral morphs on the existing partial bank** — frequency-domain
  transforms applied to the additive engine's partials at render time: inharmonic
  stretch (`pow(stretch, log2(partial_index))`), phase dispersion (quadratic
  phase offset centered at harmonic 24), amplitude smear (running average across
  harmonics), Shepard tone wrapping (octave-wrapped harmonic crossfade). These
  operate on the existing partial data without requiring a wavetable — they're
  orthogonal to and simpler than the wavetable engine's spectral morphs. Source:
  Vital WavetableOscillator (the transforms are separable from the wavetable
  frame machinery).
- **Sigma-approximation (Lánczos σ-damping) for band-limited additive
  tables** — multiply each Fourier coefficient by `sinc(K/(MaxK+1))`
  before summing instead of hard-truncating. Removes Gibbs ringing at
  near-zero CPU cost. Strictly better than our current hard truncation.
  Source: MZ2SYNTH wavetable build (`SOURCE/wvecmp.f90`).
- **Brush/Flow exciter as an additive-or-organ "breath" source** — rare-
  event stochastic sample-and-hold. `threshold = 0.0001 + 0.125·param^4`;
  flip state when `rand < threshold`; output = `state + (rand - 0.5 -
  state)·param^4`. Ten lines. Produces organic/breathy character that
  uniform noise and plain S&H can't. Pairs well as a note-onset exciter
  for pad/breath voices. Source: Mutable Instruments `elements/dsp/
  exciter.cc::ProcessFlow`.

### MIDI Export — Implemented

MIDI export is implemented with per-voice stems and shared tuning files
(Scala/TUN/KBM). See `docs/midi_export.md` for the full surface.

Remaining follow-up: `pitch_motion` and `pitch_ratio` automation are not yet
exported to MIDI.

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

### A few concrete references to draw inspiration from in future pieces

- "at les"
- "aisatsana"
- four tet-y arps + colundi scale - think these could blend really well. with some more organic additive, fm, or other textures (we can also add engines)

### Slop, swing, and drift extensions — Groove Implemented

Timing, envelope, and velocity humanization were already in place.
**Groove templates are now implemented** in `meter.py`: the `Groove`
class replaces `SwingSpec` with per-step timing offsets + velocity
weights, factory swing methods, and named presets (`mpc_tight`,
`dilla_lazy`, `motown_pocket`, `bossa`, `tr808_swing`). `Timeline`
accepts `groove=` for grid-aware feel. General `tuplet(n, m, value)`
is also implemented for quintuplets, septuplets, etc.

Remaining follow-up:

- tempo maps and tempo automation (accelerando, ritardando)
- metric modulation helpers (pivot between related tempi)
- groove extraction from audio references (analyze a WAV, produce a
  `Groove` template)
- per-voice groove offsets (e.g. drums slightly ahead, bass slightly
  behind)

### Polyrhythm/polymeter — Partially Implemented

`polyrhythm(a, b, span)` and `cross_rhythm(layers, span)` are
implemented in `composition.py` for building interlocking rhythmic
layers. Rhythmic phrase transforms (`augment`, `diminish`,
`rhythmic_retrograde`, `displace`, `rotate`) are also implemented.
Generative rhythm tools (`prob_rhythm`, `AksakPattern`, `ca_rhythm`,
`mutate_rhythm`) extend this further.

Remaining follow-up:

- polymeter helpers where voices run in different time signatures
  simultaneously (current tools handle polyrhythm within a shared
  span, but not independent meters)
- metric phasing helpers (Steve Reich-style gradual phase drift
  between voices)

### Autoresearch, for music — Evaluation Implemented

  **Standalone evaluation is implemented** (`make evaluate PIECE=...`).
  Four LLM judges (Opus 4.6, Sonnet 4.6, Opus 4.5, Sonnet 4.5) run in
  parallel via Claude Code headless, scoring each piece across five broad
  dimensions (Musical Substance 25%, Structure & Form 20%, Texture &
  Expression 15%, Completeness 10%, Open Subjective 30%).  Judges receive a
  rich multimodal packet (score snapshot, timeline, analysis manifest,
  piano-roll/spectrogram/density PNGs).  Scores are aggregated by median with
  inter-judge agreement tracking.  Results are saved as `eval.json` per piece
  and appended to `output/eval_log.jsonl` for longitudinal tracking.

  Anti-reward-hacking design: generators receive only an aggregate score plus
  a brief qualitative synthesis — never the rubric, per-dimension scores, or
  individual judge responses.  See `code_musics/evaluate.py` and
  `code_musics/eval_rubric.py`.

  Remaining work toward full autonomous loop:

- **Closed-loop orchestrator**: a composer agent modifies `build_score()`,
    renders, evaluates, decides keep/discard (git-as-state-machine, inspired
    by [karpathy/autoresearch](https://github.com/karpathy/autoresearch)),
    and iterates with only the synthesized feedback.
- **API judge backends**: Bedrock, OpenRouter, or direct Anthropic API
    calls for faster invocation and access to non-Anthropic models.
- **LLM-based feedback synthesis**: replace the current template-based
    concatenation with a dedicated synthesis pass for more natural prose.
- **Rubric calibration**: run evaluations on several pieces and tune
    dimension descriptions, scale anchors, and weights against human taste.
- **Ranking/comparison mode**: generate N variations of a section, judge
    all, and rank them.
- **Audio-capable judges**: feed actual audio (or spectrograms as images)
    to omnimodal models (Gemini, Gemma4) for judges that can hear.

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

### Modulation architecture

Ideas for the modulation wiring layer itself (not new sources — see below
for those).

- **Per-connection modulation remap (Vital-style)** — our automation is
  rich at the segment level but per-target-per-voice wiring is ad hoc.
  Vital's pattern: every mod connection is a first-class object with
  `amount`, `bipolar: bool`, `stereo: bool`, `power ∈ [-20, 20]`, and
  an optional drawable `curve`. A single system-wide matrix (~32-64
  slots) with these fields replaces scattered one-off mod wiring.
  Stereo "constant" sources (`(1, 0)` poly_float) become a mod source
  you can pan-split anything with. Integrates naturally with our
  existing `AutomationSpec` and humanization system.
  Source: Vital `ModulationConnectionProcessor`.
- **Diva-style global `accuracy` dial** — one user-facing quality
  parameter with named tiers (`draft`/`fast`/`great`/`divine`) that
  simultaneously controls oversampling factor, iterative solver
  convergence count, and feedback-path precision. Auto-escalates for
  offline render. Users happily accept CPU cost when the metaphor is
  clear. Source: Diva manual (Main panel, Accuracy setting).

### Modulation sources and aliveness

Ideas for richer, more organic modulation beyond the current humanization and
automation surfaces.

- **Lorenz attractor as modulation source** — 3 coupled ODEs (sigma=10, rho=28,
  beta=8/3), Euler or RK4 integration at control rate (~100-500 Hz). Produces
  chaotic but smooth 3-output modulation. Non-repeating, organic. Three outputs
  can modulate different targets with natural correlation (e.g. filter cutoff,
  pan, send level). Source: Vital's random LFO mode.
- **Enhanced sample-and-hold with slew limiter** — Buchla 266 "Source of
  Uncertainty" style. Clock-triggered random values with adjustable one-pole
  slew for everything from stepped staircase to smooth organic curves.
- **Envelope follower as general modulation source** — already exists inside
  saturation and compressor effects but not exposed as a reusable modulation
  routing target. Would enable sidechain-style cross-voice modulation of
  filter, pan, send, etc.
- **Ornstein-Uhlenbeck process as general modulation** — mean-reverting random
  walk (`dx = theta*(mu-x)*dt + sigma*dW`) that naturally returns to center
  without hard clamping. Better character than clamped random walk for filter
  cutoff, pan, etc. (Note: our current `build_cutoff_drift` is sine-based, not
  truly O-U, despite being documented as such.)
- **Per-sample oscillator phase noise** — tiny random perturbation to phase
  accumulator (distinct from pitch drift which is coherent). Simulates real
  oscillator zero-crossing jitter. Subtler and higher-frequency than existing
  drift.
- **Helm-style smoothed-random LFO** — at each LFO period boundary, draw a new
  uniform random `[-1, 1]`; between boundaries crossfade via
  `t = (1 - cos(π·phase))/2`. Five lines. Sits alongside our existing
  `random_walk`/`smooth_noise`/`lfo`/`sample_hold` styles in `DriftSpec` but
  has a distinctive organic-wobble character that the others miss — neither
  woolly like filtered noise nor blocky like S&H. Source: Helm `helm_lfo.cpp`.
- **Shared drift bus with correlation knob** — our drift is per-voice
  independent. A single slow (0.05–0.5 Hz) random-walk generator mixed into
  every voice at configurable depth, with a `correlation ∈ [0, 1]` knob that
  blends between "fully independent" and "fully shared," replicates the
  modular-rack-patched-to-one-S&H feel. Complements `follow_strength` in
  humanization (which correlates timing/velocity but not pitch/cutoff drift).
  Sources: VCV Eurorack idiom + Surge DriftLFO.
- **OB-Xd dual-layer voice variance** — we have stable per-voice card offsets
  (slow) and per-note jitter (fresh per note). Missing: the OB-Xd fast-layer
  per-sample CV dither (`pitch += dirt*noise` with `dirt≈0.05 semitones` on
  pitch and `±3%` on cutoff), applied continuously on top of the stable
  seed. Gives "the CV rail isn't clean" character — held chords breathe
  subtly without the slow drift being cranked up. Source: OB-Xd
  `ObxdOscillatorB.h::ProcessSample` and `ObxdVoice.h::ProcessSample`.

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

See the Colundi scale definition in `AGENTS.md`. Interesting exploration
territory for a piece — the 11-limit and septimal intervals (11/10, 49/30,
7/4) pair well with the additive engine's xenharmonic presets.

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
- **Rings-style modal position via cosine-amplitude weighting** — if/when
  we revisit the modal engine. Rather than post-filtering to simulate
  pickup position (which flangers when modulated), encode position as
  per-mode amplitude weighting: `amp[k] *= cos(2π·k·position)`. Moving
  position is just re-weighting the mode sum — no delay, no flanger
  artifact. Bonus: free odd/even stereo split (Out = sum of even modes,
  Aux = sum of odd). Also adopt RT60 damping parameterization
  (`rt60 = 0.07 * 2^(damping*8)` seconds) instead of raw feedback
  coefficients — much more musical to reason about. Source: Mutable
  Instruments `rings/dsp/resonator.cc`.

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

### Some JI intervals I haven't used much, to try

- 6:7:9 (septimal minor / subminor triad)
- 11/9 neutral third triads
- 9/7 (supermajor third)
- Utonal tetrads (1/4:1/5:1/6:1/7) (we have already explored utonal a bit)
As always, _musically_ not just throwing weird intervals out there randomly.

### Tuning-aware effects

- **Tuning-aware chorus** — delay times relative to note period rather than
  fixed ms, avoiding 12-TET comb filtering artifacts on pure JI intervals.
  Conventional chorus at fixed delay times can smear or cancel partials
  that should be clean in JI; scaling delay to the note's period preserves
  interval purity.
- **Consonance-shaped unison** — detune spread weighted toward harmonically
  related intervals rather than symmetric cents. Instead of +/- N cents,
  detune voices toward nearby ratios (e.g. 3/2, 5/4) so the beating
  reinforces the harmonic series rather than fighting it.

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

#### BBD ensemble chorus

6 parallel delay lines with 120-degree LFO phase offsets (the Juno/Dimension D
secret). Dual-LFO modulation at ~0.18 Hz and ~5.52 Hz. Anti-alias filter cutoff
tracks clock rate. This is the classic thick-but-clear chorus character that
plugin chorus approximates but rarely nails. Source: Surge BBDEnsembleEffect.

Our current `apply_chorus` is a digital LFO chorus styled "Juno-inspired,"
not a BBD model. A Juno-faithful rebuild (stereo quadrature LFOs + cross
feedback + pre/post bandlimiting + optional soft compander per channel) is
the biggest single effect gap. Mode defaults based on Juno service manual:

- Mode I: base 3.2 ms, depth ±1.5 ms, rate 0.51 Hz, cross-fb 0.08
- Mode II: base 4.4 ms, depth ±2.8 ms, rate 0.83 Hz, cross-fb 0.20

Critical implementation details:

- sum dry + wet (don't crossfade)
- quadrature LFOs (π/2 offset between L and R delay times)
- cross-feedback (L→R and R→L, not self-feedback) for airy stereo width
- fractional-delay interpolation, ideally 3-point Lagrange
- pre/post 6 kHz LPF + 120 Hz HPF for BBD bandlimiting
- optional gentle per-channel soft compander or tanh for BBD character

Sources: Juno-106 emulation (`stevengoldberg/juno106`) + general BBD
knowledge + Surge `sst-effects/BBDEnsembleEffect.h`.

#### Wavefolders

Linear fold and sine fold as effect-chain waveshapers. `linear_fold:
|mod(x*drive*0.25+0.75, 1)*-4+2| - 1`, `sine_fold: sin(x*drive*pi)`.
Different character from saturation — adds harmonics by folding the waveform
back on itself rather than clipping. Useful for aggressive timbral shaping
and west-coast-style processing. Source: Vital.

#### Filter drive and saturation

The native saturation effect now uses a two-stage analog-style path with
optional clean low/high-band preservation and higher-fidelity processing.
The old fuzzy/aliased behavior is gone. Remaining area to explore: whether
the current saturation character is musically ideal across all use cases
(e.g. gentle mix warmth vs aggressive drive vs bass-specific grit), or
whether additional saturation modes/curves would help.

Research finding: Vital uses 7 distinct saturation functions by role —
`algebraicSat` (barely-there state limiting), `quickTanh` (softer knee),
`bumpSat` (linear longer, sharper knee), `hardTanh` (clamp with soft
overflow), etc. Exposing selectable saturation curves/modes rather than a
single tanh-family shape would let the same effect cover gentle mix warmth,
musical drive, and aggressive clipping without separate effect types.

#### Analog modeling

Ideas for more convincing analog character across the signal path:

- **Thermal noise injection at specific signal path points** — pre-filter
  (shapes through filter character), in filter feedback (prevents periodic
  ringing), in release tail (circuit noise audible as signal fades). Each
  injection point has different sonic character.
- **Cross-voice oscillator bleed/crosstalk** — tiny fraction of neighboring
  voices mixed in, simulating PCB trace coupling in analog polysynths.
  Subtle but contributes to perceived warmth and ensemble cohesion.
- **Per-sample oscillator phase noise** — see "Modulation sources and
  aliveness" above. Distinct from pitch drift; simulates zero-crossing
  jitter.
- **Bootstrap 1e-6 noise on feedback paths** — a specific case of thermal
  noise that's ubiquitous in quality analog models. Without it, a pure
  digital ladder/feedback path at max resonance won't oscillate on
  silence. One line at the input of our ladder filter and post-filter
  feedback summation: `input += 1e-6 * (2*rng.uniform() - 1)`. Source:
  VCV Fundamental `VCF.cpp`.
- **Envelope curve shaping** — our `adsr()` uses pure linear segments. Known
  gap. Minimum viable upgrade: add `attack_power`, `decay_power`, and
  `release_power` exponents (defaults 1.0 for backward-compat) applied via
  `y = pow(position, power)` per stage. Also consider VCV's overshoot
  target trick (attack ramps toward 1.2, clamps at 1.0 — keeps the curve
  curvy at the top instead of flattening). Sources: Vital DAHDSR, OB-Xd
  exponential coefficient ADSR, VCV Fundamental `ADSR.cpp:ATT_TARGET=1.2`.
- **Saturation-blend coefficient idiom** — instead of a boolean "driven"
  flag on filters/stages, always compute both the clean and driven paths
  and blend via a 0-1 coefficient. Zero modulation stepping when drive
  modulates across zero. Surge's K35 does this with three coefficients
  (`saturation`, `saturation_blend`, `saturation_blend_inv`). Source:
  Surge `sst-filters/K35Filter.h`.

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

### Surge XT Parameter Automation

The `surge_xt` engine is implemented and working for basic rendering (see
`docs/synth_api.md`). This section describes the follow-up work for
automating Surge XT's internal parameters over time.

We explored three approaches for automating Surge XT's internal parameters
(filter cutoff, resonance, etc.) over time during a render:

1. **MIDI CC automation** (`cc_curves` in engine params) — The infrastructure
   works and sends CC messages during render, but Surge XT's init patch has no
   CC-to-parameter modulation routing configured. The CCs arrive and are
   silently ignored. This would work if the loaded patch had modulation routing
   set up (e.g. CC74 → filter cutoff).

2. **Chunked rendering** (`param_curves` in engine params) — Breaks the render
   into short segments, updates plugin parameters between chunks, concatenates
   with crossfade. **Fundamentally broken**: creates clicking/popping artifacts
   at chunk boundaries because the plugin's internal DSP state (IIR filter
   feedback, oscillator phase) cannot smoothly transition when parameters change
   as step functions. Tried crossfade overlap (128 samples) and aggressive chunk
   reduction (0.5 s down to 0.05 s) — neither fixes it. The artifacts are
   generated inside the plugin before the output, so output-level crossfading
   cannot help.

3. **Native post-processing filter** (current workaround) — Render Surge XT
   with a fixed bright filter, then apply a native lowpass EQ as a voice insert
   effect with score-time automation. Sample-accurate, zero artifacts. Works for
   overall brightness control but cannot access the synth's internal filter
   character (resonance, self-oscillation, nonlinear feedback, etc.).

**Path forward — configure Surge XT's modulation matrix via `raw_state`:**

- Save a preset with CC→cutoff (and other parameter) modulation routing
  configured, load it via `raw_state`, then use `cc_curves` to drive the
  parameters via MIDI CC. This keeps the modulation inside the synth where it
  belongs and gives access to the full filter character.
- Alternatively, configure a very slow internal LFO (period = piece duration)
  routed to filter cutoff via the mod matrix. Same challenge of getting the
  routing into the state blob.
- Both approaches require either figuring out Surge XT's state format
  programmatically or using the GUI to configure routing and capturing the
  resulting state.
- MTS-ESP (already mentioned elsewhere in this file) would also complement
  this for dynamic tuning changes without pitch bend.

**Global-glide chord mode** (`mpe=False`):

The `_build_global_bend_messages()` implementation provides correct
tremolo-bar-style chord glides where the whole harmonic structure slides as a
unit. The bass note gets perfect pitch; upper voices have up to ~31 cent error
from MIDI note quantization (musically desirable for a loveless/shoegaze
aesthetic). Chord-to-chord transitions glide smoothly over a configurable
duration (default 0.4 s).

### Harpsichord follow-up

Follow-up ideas for the implemented `harpsichord` engine (see `docs/synth_api.md`
for the current surface):

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

Allows unique timbres — somewhat complicated, needs to be done right
(aliasing etc).

Research findings from Vital's `WavetableOscillator`:

- **FFT-domain antialiasing**: store wavetable frames as FFT; at render time,
  zero bins above Nyquist/f0 before IFFT. This is the clean way to avoid
  aliasing without oversampling.
- **Catmull-Rom cubic interpolation** between samples within a frame (better
  than linear, cheaper than sinc).
- **Linear crossfade between frames** during morphing (simple, artifact-free).
- **Spectral morphs operating on frequency-domain frames**: inharmonic stretch
  (`pow(stretch, log2(partial_index))`), phase dispersion (quadratic phase
  offset centered at harmonic 24), amplitude smear (running average across
  harmonics), Shepard tone (octave-wrapped harmonic crossfade). These
  transforms also apply to the existing additive engine's partial bank — see
  "Spectral/additive extensions" below.

Source: Vital WavetableOscillator.

### Phase distortion synthesis

Warp the phase trajectory of a sine rather than modulating frequency (like FM).
Asymmetric triangle modulator shapes phase; distortion depth scales with
`timbre^2` and inversely with harmonic ratio (self-limiting aliasing). Simpler
than FM, less aliasing, different character. Good candidate for a lightweight
engine with strong timbral range.

Source: Mutable Instruments Plaits.

### Waveguide string engine

Recirculating delay line (length = SR/f0) with Hermite cubic fractional delay.
Two-stage loop filter (FIR brightness + IIR SVF lowpass) for
frequency-dependent damping. Allpass in series for dispersion (stretches upper
partials like real stiff strings). Dual detuned strings for beating.

Complementary to the existing modal piano/harpsichord — fundamentally different
sound (plucked guitar, dulcimer, bowed textures). Waveguides are cheap to run
and naturally produce rich, evolving sustain that modal synthesis approximates
with many modes.

Source: Surge StringOscillator + Mutable Instruments Rings.

### Particle/dust engine

Stochastic impulse trains filtered through a resonant bandpass. Between "noise"
and "pitched." Per-sample: random float vs density threshold triggers impulse
through resonant BPF. Frequency randomized per block. Good for textural layers,
transitional material, and percussion-adjacent voices that don't fit conventional
drum or noise engine models.

Source: Mutable Instruments Plaits particle.h.

#### Utonal pieces

We've focused primarily on otonal composition and JI, utonal seems worth exploring.

### Creative drum voices

Weird, unique, strange, creative drum voices. Think utonic VST or Elektron's OG machinedrum.

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
- **K35 (Korg35) filter** — two first-order TPT filters in a resonant feedback
  loop. Grittier, more aggressive resonance than SVF. LP mode:
  `LPF1 → sum → LPF2 + HPF1(feedback)`. Resonance stabilized by
  `alpha = 1/(1 - mk*G + mk*G²)`. Source: sst-filters / Surge.
- **Diode filter** — 4-pole ladder with per-stage feedback injection and 2-pole
  HP pre-filter. Distinct from Moog-style ladder (asymmetric clipping per
  stage). Primary tanh at input, drive amplified by resonance. Source: Vital.
- **Formant filter** — 4x 12dB SVF bandpass in series, 2D bilinear vowel
  interpolation (4 corner vowels on X/Y axes). Elegant and directly portable
  to existing SVF infrastructure. Source: Vital.

Possible later additions:

- ~~swing-oriented helpers where it actually serves the music~~ DONE
  — `Groove` templates with named presets are implemented
- more correlated ensemble behavior across voices for specific
  groove feels
- selective synth-parameter drift such as cutoff drift or mild
  oscillator drift where it helps rather than muddies tuning
  clarity

### Deferred generative ideas

Ideas discussed but deferred during the generative toolkit build:

- Combination Product Sets (Erv Wilson) — hexanies, dekanies, eikosanies as
  harmonic vocabularies feeding TonePool and other generators
- Comma pump generator — auto-generate chord progressions that drift by a
  specified comma per cycle
- Phase process helpers — parameterized Steve Reich-style gradual phase shifting
- L-systems — Lindenmayer systems mapped to pitch/rhythm for
  fractal structures
- Process pipelines — composable generator chaining for combining
  generators
- ~~Cellular automata — 1D CA rules mapped to musical parameters~~
  DONE — `ca_rhythm` and `ca_rhythm_layers` are implemented

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
