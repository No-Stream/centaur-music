# Velvet Wall — Development Design

Three additions to develop the piece further: a timbral arc via automation,
a generative shimmer layer, and a physical instrument in the Dissolve.

## 1. Timbral Arc (Automation)

Add parameter automation to existing voices so the piece has a timbral journey,
not just a harmonic/dynamic one.

### Melody `cutoff_hz`

The biggest single change. The melody's polyblep saw filter opens and closes
across the piece:

- Emerge (0-90s): 1400 -> 1800 Hz. Warm, slightly muted.
- Wall (90-155s): 1800 -> 3200 Hz. Opens through the progression.
- Climax (155-195s): 3200 -> 4000 Hz peak, then 4000 -> 2800.
- Dissolve (210-290s): 2800 -> 1200 Hz. Closes back, intimate.

### Melody `osc2_detune_cents`

The dual-osc saw pair spreads wider during the Wall (more shoegaze blur) and
narrows in the Dissolve (more intimate):

- Emerge: 6ct (tight)
- Wall: 6 -> 14ct
- Dissolve: 14 -> 5ct

### Tendril `mod_index`

FM brightness follows the arc:

- Emerge/early Wall: 1.8 (default)
- Climax: 1.8 -> 3.0 (more metallic sidebands)
- Dissolve: 3.0 -> 0.8 (nearly sine-like, gentle)

### Melody velocity-to-cutoff

Voice-level `VelocityParamMap` mapping velocity to `cutoff_hz` with ~400 Hz
range. Louder notes brighter, softer notes darker. Phrasing shapes timbre.

## 2. Stochastic Shimmer Layer

A new voice: **shimmer**. Additive engine, high register, bell-like tones that
dissolve into the reverb wash. Conservative: quiet, safe pitches, ambient.

### Engine and timbre

Additive, bright partials (rolloff ~0.3), fast attack (0.3s), long release
(4.0s). A few harmonics, no unison (clean, precise). Heavy hall send.

### Pitch source

`TonePool` weighted toward the safest intervals, all in high register
(partials 2.0-8.0):

- 2.0, 4.0 (octaves) — highest weight
- 3.0, 6.0 (twelfths/fifths) — high weight
- 5/2, 5.0 (major 10ths/3rds) — moderate weight
- 7/2, 7.0 (septimal 7ths) — low weight, occasional color

### Density via `stochastic_cloud`

- Silent during Emerge.
- Wall entry (105-141s): very sparse, ~1 note per 3-4s, -16 dB, vel 0.35.
- Climax (155-190s): denser, ~1 note per 1-2s, -14 dB, vel 0.42.
- Post-climax (190-210s): thinning.
- Dissolve (210-260s): sparse, fading. Silent by 260.

### Mix

Very low. Heavy hall send (send_db ~ -4). Should feel like light refracting
through reverb, not a foreground voice.

## 3. Piano in the Dissolve

A new voice: **keys**. Modal piano with the `septimal` preset for
timbre-harmony fusion. Sparse, decaying notes as punctuation.

### Engine and timbre

`piano` engine, `septimal` preset (7-limit partial ratios). Sympathetic
resonance enabled (amount=0.3, decay=2.0). Harmonically related notes
ring into each other.

### Writing

6-8 hand-placed notes between 225-260s. Slow, deliberate, with space.
Not a melody — punctuation. Individual moments of clarity against the
pad wash.

Suggested pitches following the Dissolve harmonic trajectory:

- Over HOME (210-220s): 5/4, 3/2
- Over SUSPENDED (220-232s): 2.0, 8/7 gliding to 3/2
- Fragmenting (235-250s): 5/4, 7/4
- Final passage (250-265s): 5/4, 3/2 (resolving with the piece)

### Mix

Moderate level (-8 to -10 dB), moderate hall send. Clearly audible but in
the same reverberant space. The decay and sympathetic resonance trails blend
it into the wash.

## Implementation Order

1. Timbral arc (automation) — quickest, highest impact
2. Piano in the Dissolve — straightforward, small scope
3. Stochastic shimmer — needs generative tools, slightly more involved
