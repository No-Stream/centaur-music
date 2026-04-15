# Additive Expansion Studies — Design

Two studies showcasing the 8 new additive synthesis features, with
tuning-timbre interplay as the compositional subject.

## Study 1: "Vowel Cathedral" (~90s)

**Concept:** Voices morph through vowel shapes while spectral gravity pulls
partials toward JI intervals. The timbre and tuning progressively fuse.

**Basics:** f0=110 Hz (A2), 7-limit JI, slow/contemplative, generous reverb.

### Voices

1. **Bass drone** — harmonic_spectrum(8), spectral_flicker=0.25. Pan center.
   Sustained pedal on 1/1, occasionally 3/2 or 4/3.

2. **Choir** — partials = JI scale degrees [1, 9/8, 5/4, 4/3, 3/2, 5/3, 7/4, 2].
   Formant-shaped: /a/ in section 1, /o/ in section 2, /i/ in section 3.
   spectral_gravity increases: 0.2 → 0.4 → 0.6 across sections.
   spectral_flicker=0.2, flicker_correlation=0.5.
   Pan slightly left.

3. **Melody** — formant_shape(harmonic_spectrum(12), f0, "a").
   Simple ascending/descending 7-limit phrases.
   Per-partial envelopes: gentle staggered entry.
   Pan slightly right.

4. **Bowl bells** — bowl_spectrum(6), upper_partial_drift_cents=8.
   Sparse punctuation at section transitions.
   Pan wider (±0.15).

### Structure

- **Section 1 (0-30s): /a/ — Open**
  Bass on 1/1. Choir enters shaped by /a/. Melody: 1/1, 9/8, 5/4, 3/2.
  Low gravity (0.2).

- **Section 2 (30-60s): /o/ — Round**
  Bass shifts to 3/2. Choir morphs to /o/. Bowl bell marks transition.
  Melody explores 7/4, 7/6, septimal territory. Gravity 0.4.

- **Section 3 (60-90s): /i/ — Bright, resolved**
  Bass returns to 1/1. Choir morphs to /i/. Bowl bells ring.
  Melody resolves: 5/4 → 1/1. Gravity 0.6. Flicker increases.

### Effects

- Voices: gentle chorus on choir, delay on melody
- Master: SOFT_REVERB_EFFECT + subtle saturation

---

## Study 2: "Struck Light" (~80s)

**Concept:** The arc from inharmonic to harmonic is a journey from timbral
dissonance to tuning-timbre fusion. Physical model spectra (inherently
inharmonic) give way to fractal-JI spectra (inherently harmonic).

**Basics:** f0=220 Hz (A3), 7-limit JI, rhythmic→sustained arc.

### Voices

1. **Membrane** — membrane_spectrum(8, damping=0.3).
   Short attacks, sparse rhythmic pattern. Per-partial envelopes (fast decay).
   Pan slightly right.

2. **Bar melody** — bar_spectrum(6, "metal").
   Plays JI intervals (3/2, 5/4, 7/4) — sweet intervals, inharmonic timbre.
   Per-partial envelopes: each mode decays at its own rate.
   spectral_gravity: 0.0 in section 1, increases to 0.5.
   Pan center-left.

3. **Bowl** — bowl_spectrum(6).
   Sustained tones between strikes. noise_amount=0.1 (metallic shimmer).
   upper_partial_drift_cents=6.
   Pan slightly left.

4. **Convolved bridge** — spectral_convolve(bar(5, "metal"), bowl(4)).
   Enters in section 2. Hybrid inharmonic/harmonic.
   spectral_gravity=0.3.
   Pan center.

5. **Fractal drone** — fractal_spectrum([1, 3/2], depth=3).
   Swells in section 2-3. Timbre IS the harmonic language.
   spectral_flicker=0.3. unison_voices=2, detune_cents=3.
   Pan wide.

### Structure

- **Section 1 (0-25s): Strike**
  Membrane hits establish pulse. Bar melody: short struck notes.
  Bowl tones sustain between. No gravity — pure inharmonic beauty.

- **Section 2 (25-50s): Soften**
  Bar notes lengthen. Convolved bridge enters.
  Gravity starts pulling. Fractal drone begins to swell.
  Membrane thins out.

- **Section 3 (50-80s): Fuse**
  Fractal drone dominant. Bar plays final phrase with strong gravity.
  Inharmonic partials have drifted close to JI.
  Bowl bell marks resolution. Slow fractal fade.

### Effects

- Bar/bowl: subtle delay
- Fractal drone: chorus for width
- Master: SOFT_REVERB_EFFECT

---

## Features Coverage

| Feature | Vowel Cathedral | Struck Light |
|---------|----------------|--------------|
| Per-partial envelopes | melody stagger | bar/membrane decay |
| Noise hybrid | — | bowl shimmer |
| Physical model spectra | bowl bells | membrane, bar, bowl |
| Spectral convolution | — | bar×bowl bridge |
| Fractal spectra | — | fractal drone |
| Formant shaping | choir + melody | — |
| Spectral gravity | increasing (0.2→0.6) | increasing (0→0.5) |
| Stochastic flickering | choir + bass | fractal drone |

## Implementation

Both pieces go in `code_musics/pieces/additive_studies.py` as a multi-study
module, registered in `__init__.py`. Each gets a `build_*() -> Score` function.
