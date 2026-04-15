# Rhythm Studies Design

Three study pieces showcasing the new rhythm and groove expansion features.
All studies are 90-120 seconds, substantially longer than existing 18-30s studies,
with real compositional arc.

## study_groove -- "Same Beat, Different Feel"

~100s at 92 BPM (~34 bars). One beat, one chord loop, five groove lenses.

### Section map

| Bars  | Section | Groove                   | What happens                                      |
| ----- | ------- | ------------------------ | ------------------------------------------------- |
| 1-6   | Intro   | Straight (no groove)     | Kick + hat establish the beat. Clinical, quantized |
| 7-12  | MPC     | `Groove.mpc_tight()`     | Same beat, now pocket. Hat lifts off the grid      |
| 13-20 | Dilla   | `Groove.dilla_lazy()`    | Behind-the-beat, drunken. Pad + bass enter         |
| 21-26 | Bossa   | `Groove.bossa()`         | Bossa feel. Anticipated offbeats. Light, airy      |
| 27-34 | 808     | `Groove.tr808_swing()`   | Full energy, classic shuffle. Build to peak, fade  |

### Voices

- **Kick** (`kick_tom`, 808 preset): `prob_rhythm` pattern, four-on-the-floor with ghosts
- **Hi-hat** (`metallic_perc`, closed): Dense `prob_rhythm`, evolving via `mutate_rhythm` per section
- **Chord pad** (`organ`, septimal): 7-limit JI two-chord vamp (otonal/utonal)
- **Bass** (`polyblep`, sub): Root motion, enters in Dilla section
- **Clap** (`clap`, 909): Backbeat, enters in 808 section

### Key showcase

Groove velocity weighting changes how the hi-hat sounds in each section. Timing
displacement is subtle; velocity shaping is the dramatic character change.
Also: `prob_rhythm` for drum pattern generation, `mutate_rhythm` for evolving hats.

### Technical notes

- Each section builds a new `Timeline(bpm=92, groove=...)` and re-places the same
  patterns through the groove-aware grid helpers. The compositional material is
  identical; only the groove changes.
- Humanization: `tight_ensemble` -- clean enough to hear the groove clearly.
- Shared reverb send bus for depth.
- Send bus for pad/bass; dry drums.

---

## study_aksak -- "Unequal Pulses"

~105s at 140 BPM (pulse) (~35 aksak bars in 7/8). Balkan asymmetric meter
meets septimal JI.

### Section map

| Bars  | Section    | What happens                                                          |
| ----- | ---------- | --------------------------------------------------------------------- |
| 1-4   | Pulse      | Sparse kick on group downbeats (2+2+3). Establishing the lopsided groove |
| 5-8   | Build      | Metallic perc on all pulses, bass enters on group roots               |
| 9-16  | Groove     | Full texture. Melody on `fm`. `cross_rhythm([(7, melody), (3, counter)])` tension |
| 17-24 | Mutation   | `mutate_rhythm` evolves drum patterns (ghosts, subdivisions)          |
| 25-30 | Poly peak  | `polyrhythm(3, 7)` creates 3-against-7 -- maximum rhythmic tension   |
| 31-35 | Release    | Elements thin. Kick + pad sustain. Long reverb tail                   |

### Voices

- **Kick** (`kick_tom`, 909_house): Aksak group downbeats via `aksak.to_rhythm()`
- **Pulse perc** (`metallic_perc`, closed_hat): Every pulse via `aksak.to_pulses()`
- **Bass** (`polyblep`, moog_bass): Root-fifth motion following aksak grouping
- **Melody** (`fm`, bell): Septimal JI intervals, enters section 3
- **Counter** (`additive`, soft_pad): Cross-rhythm layer, enters section 3
- **Pad** (`filtered_stack`, warm_pad): Slow harmonic motion, 7-limit chords

### Pitch material

7-limit JI from F0=110 Hz. Key partials: 1, 5/4, 3/2, 7/4, 7/6.

### Key showcase

`AksakPattern.balkan_7()` (both `to_rhythm` and `to_pulses`), `cross_rhythm`,
`mutate_rhythm` for groove evolution, `polyrhythm(3, 7)`.

### Technical notes

- The aksak pattern defines the bar structure. Kick plays on group boundaries,
  metallic perc plays every pulse.
- `mutate_rhythm` seeds change per section for evolving groove.
- `cross_rhythm` layers create tension by dividing the same aksak bar differently.
- Humanization: `chamber` -- the asymmetric meter needs a little looseness.

---

## study_rhythm_transforms -- "Motif Development"

~110s at 108 BPM (~40 bars). A single motif developed through every rhythmic
transform, building polyphonic texture like a canonic fugue.

### The motif

A short (2-bar) 7-limit JI phrase with rhythmic character: mixed durations,
syncopation, ~8 notes. Something like `[Q, E, E, Q, E, E, Q, Q]` with
harmonic-series pitches.

### Section map

| Bars  | Section        | Transform                             | Voices active     |
| ----- | -------------- | ------------------------------------- | ----------------- |
| 1-8   | Statement      | Original motif, repeated 4x           | V1 + CA percussion |
| 9-16  | Augmentation   | V2: `augment(motif, 2.0)` (half speed) | V1 + V2           |
| 17-22 | Diminution     | V3: `diminish(motif, 2.0)` (2x speed) | V1 + V2 + V3      |
| 23-30 | Retro+Displace | V4: `rhythmic_retrograde(motif)`, V5: `displace(motif, E)` | Peak: V1-V5 |
| 31-38 | Rotation       | Thin to 2 voices. `rotate(motif, 1)`, `rotate(motif, 2)`, etc. | 2 voices |
| 39-40 | Coda           | Original statement alone, one final time | V1 only           |

### Voices

- **V1** (original): `harpsichord`, baroque -- clean, articulate
- **V2** (augmented): `organ`, warm -- sustained pad at half speed
- **V3** (diminished): `fm`, glass_lead -- bright, percussive at double speed
- **V4** (retrograde): `piano`, warm -- different color, same register
- **V5** (displaced): `additive`, soft pad -- offbeat shadow
- **Percussion**: `metallic_perc` + `noise_perc`, 3 layers from
  `ca_rhythm_layers(rule=30, steps=16, layers=3)`

### Key showcase

`augment`, `diminish`, `rhythmic_retrograde`, `displace`, `rotate`,
`ca_rhythm_layers`. Each transform's effect is audible as a separate voice.

### Technical notes

- The motif is built once with `grid_line` and transformed with the new functions.
- Each transformed voice enters at a clear section boundary so the listener can
  hear what's changing.
- Rotation section shows shifting accent patterns over the same pitches.
- Humanization: `chamber` -- polyphonic layering needs breathing room.
- CA percussion provides structured-but-unpredictable rhythmic texture throughout.

---

## Shared conventions

- All three register as `study=True` in the piece registry.
- All use JI pitch material (7-limit).
- All use shared reverb send buses.
- All use appropriate humanization presets.
- All use `normalize_peak_db=-6.0` for percussion, `normalize_lufs=-24.0` for tonal.
- Master effects: `DEFAULT_MASTER_EFFECTS` from `_shared.py`.

## Verification

After implementation:

1. `make all` must pass
2. `make render PIECE=study_groove` -- verify all 5 groove sections sound distinct
3. `make render PIECE=study_aksak` -- verify asymmetric feel, cross-rhythm tension
4. `make render PIECE=study_rhythm_transforms` -- verify each transform is audible
5. Check analysis artifacts for artifact warnings
