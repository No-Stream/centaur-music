# Unified Composable Drum Voice Engine — Design & Implementation Plan

## Context

The current drum synthesis system has 5 separate engines (kick_tom, snare, clap,
metallic_perc, noise_perc) with 63 presets across them. Each engine is a siloed
architecture — kick_tom has FM + waveshaping + resonator mode + ZDF filter, but
metallic_perc has none of those. You can't put FM metallic partials on a kick, or
additive inharmonic overtones on a snare, without building it from scratch.

This "separate playgrounds" model locks agents into traditional timbres and
discourages creative hybrid sounds. The research on classic drum machines (808,
909, Jomox, Machinedrum, Rytm) shows they all decompose percussion into
composable layers — exciter + resonant body + noise/metallic component +
nonlinearity + envelope choreography. We should do the same.

**Goal**: Replace 5 engines with one `drum_voice` engine where any synthesis mode
is combinable with any other, presets just work, and the interface is clean enough
for agents to explore creatively.

---

## 1. Architecture: Four Composable Layers

Every percussive sound decomposes into four perceptual dimensions. Each is an
independent, mixable layer with its own synthesis type selector:

### Exciter Layer — transient/attack energy

| Type | Description | Key params |
|------|-------------|------------|
| `click` | Bandpass-filtered noise burst (current kick_tom/metallic click) | `center_hz`, `emphasis` |
| `noise_burst` | Wider-band noise with optional tail filter | `bandwidth_ratio`, `filter_cutoff_hz` |
| `fm_burst` | Short FM oscillator burst via existing `fm_modulate` | `fm_ratio`, `fm_index`, `fm_feedback` |
| `multi_tap` | Multiple rapid micro-bursts (current clap architecture) | `n_taps`, `tap_spacing_s`, `tap_crescendo`, `tap_acceleration`, `tap_freq_spread` |
| `impulse` | Sub-ms click for exciting resonators | `width_samples` |

### Tone Layer — pitched periodic content (the "body")

| Type | Description | Key params |
|------|-------------|------------|
| `oscillator` | Waveform + pitch sweep (current kick_tom oscillator body) | `wave` (sine/tri/sine_clip), `sweep_ratio`, `sweep_decay_s`, `punch`, `second_harmonic` |
| `resonator` | Time-varying biquad bandpass driven by exciter output | `q_scale` |
| `fm` | FM synthesis body (current kick_tom/snare FM path) | `fm_ratio`, `fm_index`, `fm_feedback`, `fm_index_decay_s` |
| `additive` | Small partial set at configurable ratios | `partial_ratios`, `n_partials` |

### Noise Layer — aperiodic texture

| Type | Description | Key params |
|------|-------------|------------|
| `white` | Raw white noise | (none beyond shared) |
| `colored` | White + pre-highpass (current snare colored wire mode) | `pre_hp_hz` |
| `bandpass` | FFT-domain bandpass noise | `center_ratio`, `width_ratio` |
| `comb` | Noise through comb filter at note freq (snare wire buzz) | `comb_feedback`, `pre_noise_mode` |

### Metallic Layer — inharmonic periodic content

| Type | Description | Key params |
|------|-------------|------------|
| `partials` | Additive inharmonic partials (current metallic_perc core) | `partial_ratios`, `n_partials`, `oscillator_mode` (sine/square), `brightness`, `density` |
| `ring_mod` | Ring modulator on partial sum | `ring_mod_freq_ratio`, `ring_mod_amount` |
| `fm_cluster` | Multiple FM operators at inharmonic ratios (**new**) | `n_operators`, `ratios`, `fm_index`, `fm_feedback` |

Each layer has: type selector (`None` = disabled), `level`, `decay_s`/`decay_ms`,
independent multi-point envelope, and type-specific params.

---

## 2. Signal Flow

```
                   ┌─────────────┐
                   │ Pitch Sweep │  (shared freq_profile for tone + metallic)
                   └──────┬──────┘
                          │
Exciter ─→ [shaper] ──┐  Tone ─→ [shaper] ──┐
                       │                      │
                    ×level×env             ×level×env
                       │                      │
                       └──────────┬───────────┘
                                  │
Noise ─→ [ZDF filter] ──┐        │
                      ×level×env  │
                         │        │
Metallic ─→ [ZDF filter]─┤        │
          ×level×env ─────┘        │
                                  │
                       ───────────┘
                          │
                       Sum mix
                          │
                    [Voice filter]  (optional ZDF SVF)
                          │
                    [Voice shaper]  (optional)
                          │
                    Peak normalize → ×amp
```

### Key routing details

- **Exciter → resonator**: When `tone_type="resonator"`, the exciter feeds the
  resonator as excitation. `exciter_level` controls how much raw exciter also
  appears in the mix (0 = only resonated result, >0 = blend).
- **Pitch envelope is shared**: Tone and metallic both receive `freq_profile`.
  Metallic partial ratios are relative to this, so sweep affects all pitched
  content coherently.
- **Shapers are pre-envelope**: Distortion on the raw oscillator before decay
  shapes it — sonically correct and matches current kick_tom/snare behavior.
- **Noise + metallic get per-layer ZDF filters**: These produce broadband content
  that almost always needs filtering. Envelope-modulated cutoff supported.
- **Voice filter + shaper are post-mix**: Glue the layers, then final shaping.

---

## 3. Shaper Architecture — Unified Nonlinearity Dispatch

The shaper slots (`exciter_shaper`, `tone_shaper`, `shaper`) are a dispatch
interface to three tiers of nonlinearity, all existing code:

| Shaper value | Routes to | Quality | When to use |
|---|---|---|---|
| `"tanh"`, `"foldback"`, `"atan"`, etc. (11 algorithms) | `_waveshaper.py` algorithms, **upgraded with ADAA** | Good (anti-aliased) | Fast, simple tone shaping |
| `"saturation"` | `synth.py` modern saturation (tube/triode/iron) | Excellent (oversampled, two-stage, sag, band preservation) | Rich analog character |
| `"preamp"` | `synth.py` preamp (flux-domain transformer) | Excellent (frequency-dependent, Chebyshev harmonics) | Warm transformer color |

Additional shaper params (mode, fidelity, preserve_lows_hz, etc.) are forwarded
to the underlying implementation. No code duplication — the drum voice calls into
existing effect functions.

### Waveshaper ADAA upgrade

The 11 existing waveshaper algorithms in `_waveshaper.py` currently have zero
anti-aliasing. The upgrade brings first-order ADAA (already proven in `_filters.py`)
to all algorithms. For each algorithm `f(x)`, we need the antiderivative `F(x)` and
compute `(F(x_n) - F(x_{n-1})) / (x_n - x_{n-1})` instead of `f(x_n)`.

Antiderivatives for the 11 algorithms:

- `tanh`: `F(x) = ln(cosh(x))` (already in `_filters.py`)
- `atan`: `F(x) = x*atan(x) - 0.5*ln(1+x^2)`
- `hard_clip`: piecewise `x^2/2` / `x - 0.5`
- `exponential`, `polynomial`, `logarithmic`: analytical antiderivatives exist
- `foldback`, `linear_fold`, `sine_fold`: piecewise antiderivatives
- `half_wave_rect`, `full_wave_rect`: piecewise

The `oversample` parameter (default 1) additionally wraps shaper calls in 2x
polyphase resampling when set to 2, for the heaviest use cases.

---

## 4. Parameter Interface

Flat namespace with layer prefixes. Not nested dicts — this preserves override
ergonomics, consistency with other engines, and automation compatibility.

### Layer params (one set per layer, shown for tone)

```
tone_type          = "oscillator"     # type selector (or None)
tone_level         = 1.0              # mix level
tone_decay_s       = 0.26             # decay time
tone_envelope      = None             # multi-point amp envelope override
tone_pitch_envelope = None            # multi-point pitch envelope override
tone_wave          = "sine"           # (oscillator-specific)
tone_sweep_ratio   = 2.5              # (oscillator-specific)
tone_sweep_decay_s = 0.042            # (oscillator-specific)
tone_punch         = 0.20             # (oscillator-specific)
tone_fm_ratio      = 1.41             # (fm-specific)
tone_fm_index      = 3.0              # (fm-specific)
... etc
```

Same pattern for `exciter_*`, `noise_*`, `metallic_*`.

### Per-layer shaper params

```
tone_shaper          = None           # algorithm name, "saturation", "preamp", or None
tone_shaper_drive    = 0.5            # drive amount
tone_shaper_mix      = 1.0            # wet/dry
tone_shaper_mode     = "triode"       # (when shaper="saturation")
tone_shaper_drive_envelope = None     # multi-point drive modulation
```

### Voice-wide params

```
filter_mode          = None           # LP/BP/HP/None
filter_cutoff_hz     = 2000.0
filter_q             = 0.707
filter_drive         = 0.0
filter_envelope      = None
shaper               = None           # voice-level shaper (same dispatch as per-layer)
shaper_drive         = 0.5
shaper_mix           = 1.0
oversample           = 1              # 1 or 2 (applies to all shaper + driven filter stages)
```

### Ergonomic macros

```
character            = None           # 0.0=clean, 1.0=dirty (maps to shaper, drive, ring_mod, noise)
punch                = None           # 0.0=soft, 1.0=hard (maps to exciter, tone_punch)
decay_shape          = None           # 0.0=tight, 1.0=boomy (maps to decay times, sweep times)
```

Macros are `None` by default (inactive). When set, they fill in params that
neither the user nor the preset has explicitly set. They never override explicit
values. Resolution order: preset defaults → macro fills → user overrides win all.

### Velocity-to-timbre (existing system, carried forward)

```
velocity_timbre_decay      = 0.0
velocity_timbre_brightness = 0.0
velocity_timbre_harmonics  = 0.0
velocity_timbre_noise      = 0.0
```

---

## 5. Preset Model

### Presets are flat param dicts

```python
"drum_voice": {
    "909_kick": {
        "exciter_type": "click",
        "exciter_level": 0.08,
        "exciter_decay_s": 0.007,
        "exciter_center_hz": 3200.0,
        "tone_type": "oscillator",
        "tone_level": 1.0,
        "tone_decay_s": 0.26,
        "tone_wave": "sine",
        "tone_sweep_ratio": 2.5,
        "tone_sweep_decay_s": 0.042,
        "tone_punch": 0.20,
        "tone_second_harmonic": 0.16,
        "noise_type": "bandpass",
        "noise_level": 0.02,
        "noise_decay_s": 0.028,
    },
}
```

Override semantics: user params merge on top of preset params (existing behavior).

```python
# Start with 909_kick, make decay longer
{"engine": "drum_voice", "preset": "909_kick", "tone_decay_s": 0.5}

# Start with 909_kick, add metallic shimmer (not in original)
{"engine": "drum_voice", "preset": "909_kick",
 "metallic_type": "partials", "metallic_level": 0.15, "metallic_decay_ms": 120}
```

### Minimal valid voices

```python
{"engine": "drum_voice", "preset": "909_kick"}                    # preset — just works
{"engine": "drum_voice", "tone_type": "oscillator"}               # ~50Hz sine with sweep
{"engine": "drum_voice", "metallic_type": "partials"}             # metallic partials only
{"engine": "drum_voice", "noise_type": "comb", "noise_level": 1.0, "tone_type": None}  # noise only
```

When no preset, defaults: `tone_type="oscillator"` enabled, all other layers
`None`. `tone_type` is the only layer that defaults to something other than off.

### Preset-aware effect recommendations via drum_helpers

`add_drum_voice` should automatically apply recommended voice-level EffectSpecs
based on the preset category unless the user provides explicit effects:

- Kick presets → `kick_punch` compressor + `kick_weight` saturation
- Snare presets → `snare_punch` compressor
- Hat presets → `hat_control` compressor
- Custom / no preset → no default effects

This is a `drum_helpers` enhancement, not part of the engine itself.

---

## 6. Ergonomic Macro Details

### `punch` (0.0 = soft pillowy, 1.0 = hard snappy)

Maps to (when not explicitly set):

- `exciter_level`: 0.01 → 0.25
- `exciter_decay_s`: 0.012 → 0.003
- `exciter_center_hz`: 1500 → 5000
- `tone_punch`: 0.0 → 0.35

### `decay_shape` (0.0 = tight/gated, 1.0 = long/boomy)

Maps to:

- `tone_decay_s`: 0.08 → 0.9
- `noise_decay_s`: proportional
- `metallic_decay_ms`: proportional
- `tone_sweep_decay_s`: shorter sweeps at low values (tight), longer at high

### `character` (0.0 = clean/pure, 1.0 = dirty/complex)

Maps to:

- `tone_shaper`: None at 0.0, `"tanh"` at 0.3+, `"foldback"` at 0.7+
- `tone_shaper_drive`: 0.0 → 0.6
- `metallic_ring_mod_amount`: scaled up (if metallic active)
- `filter_drive`: 0.0 → 0.3
- `noise_level`: slight boost at higher values

### Resolution rule

Macros compute target values via interpolation curves. For each target param,
they check whether the param was set by user OR preset. If yes, macro does not
touch it. If no, macro fills it in. This means:

```python
# Macro fills gaps
{"preset": "909_kick", "punch": 0.8}
# punch boosts exciter_level — but tone_decay_s stays at preset's 0.26

# Explicit always wins
{"punch": 0.9, "exciter_level": 0.01}
# exciter_level stays at 0.01 despite high punch
```

---

## 7. Migration

### Phase 1: New engine alongside old ones

- Implement `drum_voice` as a new engine in `engines/`
- Port all 63 presets to `drum_voice` format
- Integration tests verify each preset produces output within tolerance of the
  original engine's render

### Phase 2: Alias old engine names

- `kick_tom`, `snare`, `clap`, `metallic_perc`, `noise_perc` resolve to
  `drum_voice` in `resolve_synth_params`
- Parameter translation (`body_decay` → `tone_decay_s`, etc.) with deprecation
  warnings
- Full test suite passes; existing pieces render identically

### Phase 3: Port pieces, remove aliases

- Update all pieces to use `drum_voice` directly
- Remove alias/translation code
- Delete old engine files

### Parameter translation map (key examples)

```
kick_tom:
  body_decay         → tone_decay_s
  pitch_sweep_amount_ratio → tone_sweep_ratio
  pitch_sweep_decay  → tone_sweep_decay_s
  body_punch_ratio   → tone_punch
  click_amount       → exciter_level
  click_decay        → exciter_decay_s
  click_tone_hz      → exciter_center_hz
  noise_amount       → noise_level
  body_fm_ratio      → tone_fm_ratio
  body_distortion    → tone_shaper
  body_filter_mode   → filter_mode
  overtone_amount    → (secondary tone partial or metallic)

snare:
  body_decay         → tone_decay_s
  wire_decay         → noise_decay_s
  wire_center_ratio  → noise_center_ratio
  comb_amount        → noise_comb_feedback
  body_mix           → tone_level
  wire_mix           → noise_level
  wire_noise_mode    → noise_pre_noise_mode ("colored" → noise_type="comb" w/ colored)

clap:
  n_taps             → exciter_n_taps
  tap_spacing        → exciter_tap_spacing_s
  body_decay         → noise_decay_s (the tail)
  filter_center_ratio → noise_center_ratio

metallic_perc:
  partial_ratios     → metallic_partial_ratios
  brightness         → metallic_brightness
  ring_mod_amount    → metallic_ring_mod_amount
  filter_mode        → metallic_filter_mode
```

---

## 8. File Structure

### New files

| File | Purpose | ~Lines |
|------|---------|--------|
| `code_musics/engines/drum_voice.py` | Main `render()` entry point, signal flow, mixing, normalization | 350-450 |
| `code_musics/engines/_drum_layers.py` | Four layer generator functions migrated from existing engines | 500-600 |
| `code_musics/engines/_drum_macros.py` | Macro resolution logic | 80-120 |

### Modified files

| File | Changes |
|------|---------|
| `code_musics/engines/registry.py` | Register `drum_voice`, add 63+ migrated presets, alias layer |
| `code_musics/engines/_waveshaper.py` | Add ADAA antiderivatives for all 11 algorithms |
| `code_musics/drum_helpers.py` | Preset-aware default effect recommendations |
| `docs/synth_api.md` | Document `drum_voice` engine, close doc gaps for existing features |
| `docs/score_api.md` | Minor updates for drum_voice integration |

### Unchanged (reused as-is)

- `_envelopes.py`, `_filters.py`, `_drum_utils.py`, `_dsp_utils.py` — shared infra
- `synth.py` — saturation, preamp, compressor called by drum_voice but not modified

---

## 9. Implementation Plan

### Step 1: Waveshaper ADAA upgrade

Upgrade `_waveshaper.py` with first-order ADAA for all 11 algorithms. This is a
prerequisite for the drum voice and also immediately improves existing kick_tom
and snare quality.

- Add antiderivative functions for each algorithm
- Modify `apply_waveshaper` to use ADAA path by default
- Add `oversample` parameter (1 or 2) with polyphase resampling wrapper
- Test: verify existing kick_tom/snare renders still pass, check spectral
  improvement

### Step 2: Core drum_voice engine

Implement `drum_voice.py` with the four-layer architecture and signal flow.
Start with the layer types that directly port from existing engines:

- Exciter: `click`, `impulse`, `multi_tap`
- Tone: `oscillator`, `resonator`, `fm`
- Noise: `white`, `colored`, `bandpass`, `comb`
- Metallic: `partials`

Defer to step 4: `fm_burst` (exciter), `noise_burst` (exciter), `additive`
(tone), `ring_mod` (metallic), `fm_cluster` (metallic).

### Step 3: Macro system

Implement `_drum_macros.py` with `punch`, `decay_shape`, `character` macros.
Wire into parameter resolution in `drum_voice.py`.

### Step 4: New synthesis types

Implement the types that are genuinely new (not ported from existing engines):

- `fm_burst` exciter — short FM burst using existing `fm_modulate`
- `noise_burst` exciter — wider noise with tail filter
- `additive` tone — small partial set (ports metallic additive to tone context)
- `ring_mod` metallic — standalone ring mod
- `fm_cluster` metallic — multiple FM operators at inharmonic ratios

### Step 5: Preset migration

Port all 63 presets to `drum_voice` format. Write A/B integration tests
comparing old engine output vs. new engine output for each preset.

### Step 6: Shaper dispatch to saturation/preamp

Wire `tone_shaper="saturation"` and `tone_shaper="preamp"` to call the existing
effect implementations in `synth.py`. Refactor the effect functions if needed to
accept raw numpy arrays (they currently expect EffectSpec context).

### Step 7: Alias layer + piece migration

Add backward-compatible aliases for old engine names with param translation.
Verify all pieces render correctly. Then port pieces to `drum_voice` directly
and remove aliases.

### Step 8: drum_helpers enhancement

Add preset-aware default effect chains to `add_drum_voice`. Update documentation.

### Step 9: Documentation

Update `docs/synth_api.md` with full `drum_voice` documentation. Also close
existing doc gaps (resonator mode, colored wire noise, FM on snare, oscillator
mode on metallic, etc.).

---

## 10. Verification

- `make all` passes after each step
- A/B tests: each migrated preset renders within tolerance of original engine
- New synthesis combinations work: FM metallic + resonator body, etc.
- Macros produce musically reasonable results across their range
- Shaper dispatch to saturation/preamp produces correct output
- Existing pieces render identically through alias layer
- Ported pieces render identically with `drum_voice` directly
- Spectral tests verify ADAA reduces aliasing in waveshaper output

### Delegation breakdown

- **Main context**: orchestration, design decisions, user communication, creative
  preset tuning, macro curve tuning
- **Subagents**: implementation of each step (ADAA upgrade, layer generators,
  macro system, preset migration, tests, documentation)
- **Model**: inherit for implementation steps (non-trivial DSP + design judgment),
  sonnet for mechanical preset translation and doc updates
