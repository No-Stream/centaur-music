# Future Work

This file is a compact roadmap, not an implementation journal. Keep it focused
on useful next work: musical direction, missing capabilities, testing gaps, and
design bets that should not be forgotten. Completed feature details belong in
`docs/score_api.md`, `docs/composition_api.md`, `docs/synth_api.md`, or a dated
plan under `docs/plans/`.

## Baseline

The repo is past basic scaffolding. It already has:

- a real `Score` / `Voice` / `Phrase` / `NoteEvent` composition model
- phrase-first helpers for lines, ratios, overlays, canon, progressions, grid
  timing, grooves, tuplets, polyrhythm, and generative rhythm
- additive, FM, filtered-stack, polyBLEP, VA, piano, harpsichord, organ,
  drum/percussion, sample, and plugin-backed voices
- expression surfaces for velocity, pitch motion, timing / envelope / velocity
  humanization, score-time automation, and modulation matrix routing
- native EQ, compression, limiting, clipping, preamp / tube / transistor color,
  delay, chorus, reverb, Bricasti fallback routing, analog filters, send buses,
  and plugin hosting through `pedalboard`
- snippet / window rendering, timestamp inspection, piano-roll and analysis
  artifacts, LLM evaluation, MIDI import/export, and stem export

The main question is no longer whether the system can render xenharmonic music.
It can. The highest-value work is making stronger pieces, reducing iteration
friction, and broadening the sonic language without turning the library into a
pile of special cases.

## Highest Priority

### More Complete Pieces

The central artistic question remains: can we make xenharmonic music that feels
like full composition rather than a study, sketch, or demo?

Most valuable directions:

- stronger large-scale form, pacing, contrast, release, and arrival
- memorable motifs with thematic return and transformation
- clearer voice roles: bass/pedal, harmonic bed, lead, counterpoint, accent,
  percussion, and atmosphere
- more deliberate orchestration so pieces feel arranged rather than layered
- a better balance between pleasant, strange, tender, chaotic, and alien
- more pieces worth revisiting, comparing, evaluating, and refining

Concrete piece prompts:

- **Generative Music that Sounds like Music**: not modular slop. real music. 
  maybe using creative ideas or generative approaches that create more memorable patterns,
  locking in patterns, mixing generative and traditional composition, etc.
- **Loveless-adjacent smearing:** fragmented chords, global glide/tremolo-bar
  bends, sus2/sus4/add9 ambiguity, octave duplication, chorus, saturation, and
  polyrhythmic drift.
- **Focusing on Sine, Additive, (and modal/physical/karplus)**: Aleksi Perala
  is an example of a producer who blends synthesis with scale. E.g. can construct
  additive sounds that omit the octave with a scale that omits the octave, can reduce
  the clash of sounds "wanting the octave" or not adapting well to alt tunings.
  (Note: octave just an example here, relevant w/ other intervals.)
- **Utonal / subharmonic form:** darker undertone passages that contrast with
  otonal material inside a coherent structure.
- **Four Tet-ish Colundi arps:** warm, organic arpeggios using the Colundi-ish
  11-limit / septimal world, with additive, FM bell, grain, and smear textures.
- **Trance / progressive house:** slightly cheesy harmonic movement in alt
  tunings, with real rhythm and mix confidence.
- **Fake found sound:** sample-free synthetic field-recording / Burial-adjacent
  atmospheres that can sit inside pieces.

### Better Composition Tools

The composition layer is useful, but writing finished music should be faster.

Useful next helpers:

- phrase transformations beyond the current core set
- idiomatic builders for pedals, blooms, suspensions, staggered entries,
  harmonic-series gestures, and contrapuntal restatement
- readable otonal / utonal voice-leading helpers that do not become tuning math
- section builders for restating material with new orchestration, local-tonic
  reinterpretation, or registral shift
- summaries for overlap, beating, density, register, and static/crowded textures
- Riemannian / neo-Riemannian helpers where they fit the tuning world

### VST / DAW Export

Expose the strongest native voices and effects as Ableton-usable VST3/AU tools
without rewriting everything in C++.

Likely shape:

- `CentaurVoice`: instrument plugin for `synth_voice`, `drum_voice`, and core
  engines
- `CentaurFX`: effect plugin for analog filters, color, compression, clipping,
  limiting, chorus, reverb, and related native processing
- thin JUCE shell plus an off-thread Python render worker, using shared-memory
  buffers and intentional lookahead rather than embedded CPython on the audio
  thread
- first milestone: refactor engines toward `render_block(state, ..., note_off)`,
  remove whole-buffer assumptions, and add typed parameter schemas

Full design: `docs/plans/2026-04-26-vst-export-design.md`.

## Medium Priority

### Additive / Spectral Composition

Additive synthesis is the best place to make tuning and timbre feel like one
system instead of "normal synth, retuned."

Open directions:

- richer stereo-from-spectrum tools, such as partial-group width and panning
- register-aware darkening / brightness limiting
- controlled inharmonic stretch and physical-object detuning
- deeper per-partial envelopes and modulation
- broader role-oriented presets beyond the current JI / septimal / 11-limit /
  utonal family
- brush / flow exciters for organic breath and onset texture
- more co-designed scales + spectra, especially non-octave and Colundi-ish
  materials

### Voice Engines

Useful engine additions or unifications:

- modal / physical resonator slot inside `synth_voice`
- hard-sync and per-slot cross-modulation in `synth_voice`
- selective 4-op / 6-op FM algorithm support if a piece needs it
- wavetable engine with FFT-domain antialiasing and frame morphing
- phase-distortion engine for a lightweight FM-adjacent color
- waveguide string engine for plucked, dulcimer, guitar, and bowed textures
- particle / dust engine for resonant stochastic textures
- sample player follow-up for foley and atmospheric bits
- Pianoteq headless/plugin path if the Linux setup is worth the friction

Lower-value unless a piece demands it:

- full modular audio-rate patch graph
- multiple same-type slots in one voice
- organ parity beyond current drawbar coverage

### Rhythm, Groove, and Tempo

Implemented foundations: tempo maps, groove templates, tuplets, polyrhythm,
cross-rhythm, rhythmic transforms, probability rhythms, aksak patterns, CA
rhythms, and rhythm mutation.

Still useful:

- metric modulation helpers
- independent polymeter helpers where voices run their own cycles
- Reich-style phasing helpers
- groove extraction from audio
- per-voice groove offsets, such as drums ahead and bass behind
- stronger drum and hybrid-percussion identities in actual pieces

### Modulation and Aliveness

The modulation matrix now has broad audio-rate coverage for core filter,
drive, oscillator, VA filter-slot, and phrase-gesture use cases. Future work
should focus on sources, stereo destinations, and deeper integration rather
than repeating the basic filter/drive plumbing. Current feature details live
in `docs/score_api.md`, `docs/synth_api.md`, and `docs/composition_api.md`.

Worth doing:

- **Filter gap: dynamic Newton parity for non-SVF topologies.** The current
  modulation work gives all eight filter topologies per-sample cutoff plus
  dynamic `resonance_q`, `filter_drive`, `filter_morph`, and
  `feedback_amount`, but non-SVF topologies switch to the profiled stateful
  kernel for those dynamic main controls only when the caller explicitly sets
  `filter_solver="adaa"`. That preserves musical motion and is useful, but it
  is not the same as the scalar Newton-closed feedback behavior for ladder,
  sallen-key, cascade, SEM, Jupiter, K35, and diode. Future work should close
  the dynamic feedback loops per sample for those topologies, keep
  scalar/constant-profile renders bit-identical, stress-test high-Q /
  high-feedback stability, and remove the fail-closed guard only once the
  profiled and scalar quality contracts genuinely match.
- stereo modulation for currently mono synth destinations
- `VelocityParamMap` lowering into matrix connections
- envelope follower as a reusable modulation source
- Lorenz, Ornstein-Uhlenbeck, enhanced sample-and-hold, and shared drift-bus
  sources
- richer phrase-level timbre gestures that compose multiple lanes, such as
  opening while blooming into a send or settling while narrowing

Keep current humanization ergonomics unless a migration clearly improves the
authoring surface.

### Effects and Mixing

The effect story is no longer about basic availability. The next wins are better
character, clearer routing, and more trustworthy analysis.

Open ideas:

- role-based effect presets such as `glass_pad`, `sub_drone`, `reed_lead`,
  `dark_hall`, and drum-bus variants
- a more faithful Juno / Dimension-style BBD ensemble chorus
- wavefolder effects: linear fold and sine fold
- tuning-aware chorus and consonance-shaped unison
- non-octave-privileged distortion products
- selectable saturation / clip curves for gentle warmth, musical drive, and
  aggressive clipping
- formant filter and Buchla-style low-pass gate
- plugin reliability fixtures for repeated renders, reset behavior, on/off
  toggles, and host-vs-plugin-vs-mapping failures
- paid Linux-capable plugins (`u-he Satin`, `u-he Presswerk`) once activation is
  low-friction enough

### Arrangement / Arc Analysis Tooling

Hand-rolled scratch scripts during aphotic (per-bucket RMS "loudness arc"
with ASCII bars) proved immediately useful for verifying form — worth
promoting into the standard analysis pass:

- loudness-arc timeline in the analysis manifest: RMS/LUFS per N-second
  bucket for the mix and per voice, plus an ASCII/plot rendering, so
  "does III stay quieter than II" is answerable without a scratch script
- section-aware aggregates keyed to `PieceSection` boundaries: note count,
  onset density, active-voice count, mean velocity, spectral centroid per
  section — an "arrangement map" that shows orchestration at a glance
- tuning-aware pitch histogram over time resolved to scale-degree labels
  (chromagram is close but bins, not ratios); would directly audit things
  like aphotic's no-3 constraint or a lone prime-5 event
- per-voice activity/density curves for spotting dead zones and
  over-stacked moments
- render-to-render arc diff: same buckets, two renders, signed delta —
  the fastest possible answer to "what did this revision change, where"

### Evaluation and Iteration

Standalone LLM evaluation exists. The next step is making it useful in a
closed-loop composition workflow without reward hacking.

Useful additions:

- composer-agent loop: modify score, render, evaluate, keep/discard, iterate
  with git as the state machine
- API-backed judge calls for faster runs and model diversity
- LLM synthesis pass for more natural feedback
- calibration against human taste across representative pieces
- ranking mode for comparing N variations of a section
- audio-capable judges when omnimodal models are useful enough
- render-to-render comparison views and more agent-readable analysis summaries

### Testing and Robustness

Highest-value test gaps:

- snippet / render-window equivalence against full renders
- wet stem sum plus send returns matching the pre-master mix under controlled
  conditions
- shared send-bus gain math with several voices and non-zero `return_db`
- mixed LUFS-normalized tonal voices and peak-normalized percussion in one score
- `DEFAULT_MASTER_EFFECTS` on a real score, including plugin fallback behavior
- choke groups combined with `max_polyphony` and `legato`
- MIDI import -> Score -> MIDI export round-trip
- full `OscillatorSource` -> `ModConnection` -> `Voice` -> render integration
- cross-engine coverage for documented engine-agnostic features

### Repo Structure

Worth revisiting when nearby:

- split `code_musics/render.py` into orchestration vs artifact metadata if it
  keeps growing
- tighten package API boundaries and compatibility surfaces
- decide the future of the repo-root `synth.py` compatibility wrapper
- consider moving generated `output/` artifacts behind a clearer workspace
  boundary
- fix the `ModSource.sample(times, context)` base-signature ty ignore

## Lower Priority / Research Parking Lot

These are worth remembering but should not outrank composition, iteration speed,
or obvious quality gaps.

### Tuning and Harmony

- Combination Product Sets / Erv Wilson harmonic lattice materials — three
  pieces exist (hexany_garden, ninth_wave, sodium_hymn), so CPS is deliberately
  resting for now; definitely worth returning to when a piece calls for it
- comma-pump progression generator
- 6:7:9 subminor triads, 11/9 neutral-third triads, 9/7 supermajor color,
  utonal tetrads
- stronger overtone / undertone contrasts inside single forms
- local-tonic reinterpretation and comma drift as normal phrase operations

### Physical Modeling

- modal piano improvements: sympathetic resonance, double decay, sustain/una
  corda, FDN body, register-dependent hammer behavior, duplex scaling, prepared
  piano, and better high-frequency compensation
- harpsichord follow-up: waveguide alternative, coupling, 16' / fantasy stops,
  and buff-stop modeling
- waveguide / comb upgrades with fractional delay, loop damping, dispersion, and
  TPT-family feedback combs

### Analog / VA DSP

- WDF block for a flagship pedal or preamp
- small MNA/state-space experiment for a tone stack or diode clipper
- PolyBLAMP for slope-discontinuity waveforms
- BLIT alternative for extreme sync or audio-rate FM
- ADAA refinements for fold-family waveshapers and Newton filter paths
- power-supply sag, component-tolerance drift, unison voice-card variance, VCA
  asymmetry, CV feedthrough, and summing-bus bleed
- cross-voice oscillator crosstalk and persistent DC-thump character
- oscillator sync character variants

### Drum and Bus Diagnostics

- TRX-style drum/percussion kernel
- per-step parameter-lock macros for drum voices
- kick-specific bus diagnostics: sub preservation, transient preservation, and
  click-band lift
- per-effect warning thresholds for saturation, clipping, compressor/limiter
  interaction, and too many nonlinear stages
- `preserve_lows_hz` / `preserve_highs_hz` deprecation cleanup in favor of the
  modern multiband crossover controls
- clipper quality refinement for clean 1-3 dB mastering-style shave

### External Tools

- Mutable Instruments ideas worth mining: Rings, Clouds, Elements flow exciter,
  Plaits phase distortion / particles
- DawDreamer only if pedalboard cannot cover a needed routing or CLAP/Faust use
  case; note GPLv3 implications before adopting
- Surge XT internal parameter automation through preset state / CC routing if
  native post-filter automation is insufficient; see
  `docs/plans/surge_xt_parameter_automation_notes.md`
