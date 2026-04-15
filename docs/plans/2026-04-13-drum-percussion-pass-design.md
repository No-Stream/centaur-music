# Drum & Percussion Comprehensive Pass — Design

## Context

The library has two percussion engines: `kick_tom` (808/909 kicks and toms, 12
presets) and `noise_perc` (noise+tone hybrid, 5 presets for snare/hat/clap/tick).
Kicks and toms sound fine. Hats don't sound metallic enough (they're just
bandpassed noise, not inharmonic partials). Snares sound bad (just noise+tone, no
wire resonance). Claps are okay but limited to single noise bursts.

The effects chain is solid (oversampled saturation, full compressor with sidechain
and drum presets, EQ, preamp, send buses). The gap is in synthesis engines and some
infrastructure.

**Goal**: Make percussion sound polished and finished. Electronic-first aesthetic
(808/909 lineage), with flexible engines that can also go further.

## Decisions Made

- **Electronic-first** aesthetic — FM/additive metallic synthesis, not physical modeling
- **Flexible additive** engine for metallic percussion (not fixed 808 circuit clone),
  with presets that nail classic tones. Preset quality is critical.
- **Remove kick_tom internal drive** — it's a naive `tanh` at 1x sample rate, strictly
  worse than our external saturation (oversampled, analog-modeled) and preamp
  (flux-domain). Engine focuses on synthesis; effect chain handles coloring.
- **Narrow choke groups** — simple string on Voice, voices in same group cut each
  other on note onset with 10ms fade. Percussion-only concept.
- **Drum bus as helper function**, not new Score primitive — wraps existing
  SendBusSpec/VoiceSend with drum-friendly defaults.

---

## Track 1: Polish Existing DSP

### 1A. kick_tom — Remove internal drive and lowpass

**Files**: `code_musics/engines/kick_tom.py`

Remove `_apply_drive()` (line 226) and `_one_pole_lowpass()` (line 232). These are:

- Naive `tanh` waveshaper at 1x sample rate (no oversampling → aliasing)
- 6 dB/oct one-pole lowpass in a Python for-loop (slow, weak rejection)

Both are strictly inferior to our external effects:

- `EffectSpec("saturation", {"preset": "kick_weight"})` — oversampled, analog-modeled
- `EffectSpec("preamp", {"preset": "neve_warmth"})` — flux-domain transformer model

**Changes**:

- Remove `_apply_drive()` and `_one_pole_lowpass()` functions
- Remove `drive_ratio` and `post_lowpass_hz` from the render path
- Deprecate the params: log warning if passed, ignore them
- Remove `drive_ratio` and `post_lowpass_hz` from all 12 presets
- Update any pieces that relied on internal drive to use effect chain instead

### 1B. noise_perc — Click envelope attack ramp

**File**: `code_musics/engines/noise_perc.py`

Add 0.2ms linear attack ramp to `_click_envelope()` to prevent hard-onset pop.
Add `click_attack_ms` param (default 0.2). Subtle change, shouldn't alter preset
character.

### 1C. noise_perc — Document missing params

`bandpass_width_ratio` is used in the `clap` preset but undocumented. `chh` and
`clap` presets exist in the registry but aren't listed in `synth_api.md`.

---

## Track 2: New Engines

All new engines follow the existing contract:

- `render(*, freq, duration, amp, sample_rate, params, ...) -> np.ndarray`
- Deterministic seeded RNG via `_rng_for_note()` pattern
- Use `_filters.py` ZDF SVF for all filtering (Numba-JIT, quality)
- No internal distortion/drive — effect chain handles coloring
- Peak-normalize then multiply by amp

### 2A. `metallic_perc` Engine — Hihats, Cymbals, Cowbell, Clave

**File**: `code_musics/engines/metallic_perc.py` (new, ~250-300 lines)

**Signal path**:

1. Generate N sine oscillators at `freq * ratio[i]` with proper phase accumulation
   (`np.cumsum(2*pi*freq/sr)`)
2. Weight each partial by brightness rolloff curve (higher partials fade)
3. Optional ring modulator: `signal * sin(ring_mod_freq * t)` mixed in by
   `ring_mod_amount`
4. Sum → ZDF SVF bandpass from `_filters.py` for tone shaping
5. Exponential decay envelope with 0.1ms attack ramp to prevent clicks
6. Optional transient click (short noise burst, same pattern as kick_tom)
7. Peak-normalize → multiply by amp

**Parameters**:

| Parameter | Default | Description |
|---|---|---|
| `n_partials` | 6 | Number of metallic partials |
| `partial_ratios` | sqrt series | Override default ratio set |
| `brightness` | 0.7 | Upper partial amplitude rolloff (0=dark, 1=bright) |
| `decay_ms` | 80.0 | Main envelope decay |
| `ring_mod_amount` | 0.0 | Ring modulation depth (0-1) |
| `ring_mod_freq_ratio` | 1.48 | Ring mod frequency as ratio of `freq` |
| `filter_center_ratio` | 1.0 | Bandpass center as ratio of `freq` |
| `filter_q` | 1.2 | Bandpass resonance |
| `click_amount` | 0.05 | Transient noise burst level |
| `click_decay_ms` | 3.0 | Transient decay |
| `density` | 0.5 | Partial spacing jitter (per-note RNG) |

**Presets** (8):

| Preset | Character | Key params |
|---|---|---|
| `closed_hat` | Tight, bright, 808-ish | decay_ms=45, brightness=0.8, filter_q=1.5, click=0.12 |
| `open_hat` | Sustained shimmer | decay_ms=450, brightness=0.65, filter_q=0.9, click=0.08 |
| `pedal_hat` | Medium, darker | decay_ms=120, brightness=0.6, filter_q=1.3 |
| `ride_bell` | Pingy, resonant | decay_ms=800, n_partials=4, ring_mod=0.3, filter_q=2.0 |
| `ride_bow` | Washy sustain | decay_ms=1200, n_partials=8, ring_mod=0.15, filter_q=0.8 |
| `crash` | Long, dense | decay_ms=2000, n_partials=8, density=0.8, filter_q=0.7 |
| `cowbell` | Two-partial, resonant | n_partials=2, ratios=[1.0, 1.504], filter_q=2.5 |
| `clave` | Short, pitched | decay_ms=25, n_partials=3, ratios=[1,2,3], filter_q=3.0 |

**Preset tuning**: Use `make render` + spectral analysis to compare against
reference 808/909 hihat spectra. The sqrt-series ratios (1.0, 1.414, 1.732, 2.0,
2.236, 2.449) approximate the 808's 6-oscillator metallic circuit. Adjust
brightness curves and filter settings until closed_hat and open_hat pass the ear
test.

### 2B. `snare` Engine — 909-inspired with wire resonance

**File**: `code_musics/engines/snare.py` (new, ~220-270 lines)

**Signal path** (three components):

1. **Body**: Two sine oscillators at `freq` and `freq * body_overtone_ratio`
   (default 1.6). Pitch sweep from `body_sweep_ratio * freq` down to `freq`
   over `body_sweep_decay_ms`. Uses `_integrated_phase()` for proper phase
   accumulation (same as kick_tom). Exponential decay with `body_decay_ms`.

2. **Wire**: White noise → **comb filter** → bandpass (ZDF SVF).
   - Comb filter: `y[n] = x[n] + comb_amount * y[n - delay]` where
     `delay = sample_rate / freq`. This creates pitched resonance at the snare
     body frequency — the critical character that makes wires sound like wires.
   - Numba-JIT the comb filter loop for performance.
   - Bandpass via ZDF SVF at `wire_center_ratio * freq`, Q from `wire_q`.
   - Exponential decay with `wire_decay_ms`.

3. **Click**: Short broadband noise burst (3-8ms), same transient pattern as
   kick_tom.

4. **Mix**: `body_mix * body + wire_mix * wire + click_amount * click` →
   peak-normalize → multiply by amp.

**Parameters**:

| Parameter | Default | Description |
|---|---|---|
| `body_decay_ms` | 120.0 | Body tone decay |
| `body_overtone_ratio` | 1.6 | Second body partial ratio |
| `body_sweep_ratio` | 1.8 | Initial/final pitch ratio for sweep |
| `body_sweep_decay_ms` | 15.0 | Pitch sweep decay |
| `wire_decay_ms` | 180.0 | Wire noise decay |
| `wire_center_ratio` | 3.0 | Wire bandpass center (ratio of freq) |
| `wire_q` | 0.8 | Wire bandpass resonance |
| `comb_amount` | 0.45 | Comb filter feedback (0-1, wire resonance) |
| `body_mix` | 0.5 | Body component level |
| `wire_mix` | 0.5 | Wire component level |
| `click_amount` | 0.15 | Transient click level |
| `click_decay_ms` | 5.0 | Click decay |

**Presets** (6):

| Preset | Character | Key params |
|---|---|---|
| `909_tight` | Snappy, punchy | body=90ms, wire=140ms, comb=0.5 |
| `909_fat` | Bigger body, more sweep | body=150ms, wire=200ms, sweep=2.2 |
| `808_snare` | Longer, more body | body=200ms, wire=250ms, comb=0.3 |
| `rim_shot` | Click-dominant | body=40ms, click=0.5, click_decay=3ms |
| `brush` | Wire-dominant, soft | body=30ms, wire=400ms, body_mix=0.1 |
| `cross_stick` | Very short, clicky | body=20ms, click=0.65 |

### 2C. `clap` Engine — Multi-tap noise burst

**File**: `code_musics/engines/clap.py` (new, ~180-220 lines)

**Signal path**:

1. Generate `n_taps` individual noise bursts (3-6 taps, 2-8ms apart):
   - Each tap: white noise (deterministic RNG) → bandpass (ZDF SVF) → short
     exponential decay envelope (`tap_decay_ms`)
   - 0.1ms attack ramp per tap to prevent clicks
   - Each successive tap slightly louder (`tap_crescendo` controls the ramp)

2. Place taps sequentially with `tap_spacing_ms` gaps into a full-length buffer.

3. **Body tail**: After the final tap, a longer noise tail with `body_decay_ms`
   decay. Bandpass-filtered. Starts at the onset of the last tap.

4. Sum all components → peak-normalize → multiply by amp.

**Implementation**: Build full-length signal buffer, overlay each tap at its
computed offset. Vectorized, no per-sample loops.

**Parameters**:

| Parameter | Default | Description |
|---|---|---|
| `n_taps` | 4 | Number of micro-bursts (2-8) |
| `tap_spacing_ms` | 5.0 | Time between tap onsets |
| `tap_decay_ms` | 3.0 | Per-tap envelope decay |
| `tap_crescendo` | 0.3 | Amplitude increase per tap (0=equal, 1=2x) |
| `body_decay_ms` | 60.0 | Main body tail decay |
| `filter_center_ratio` | 1.0 | Bandpass center as ratio of freq |
| `filter_q` | 0.8 | Bandpass Q |
| `click_amount` | 0.08 | Initial transient |

**Presets** (5):

| Preset | Character | Key params |
|---|---|---|
| `909_clap` | Classic 909 | 4 taps, 5ms spacing, 65ms body |
| `tight_clap` | Snappy, short | 3 taps, 3ms spacing, 35ms body |
| `big_clap` | Roomy, sustained | 6 taps, 7ms spacing, 120ms body |
| `finger_snap` | Quick, high | 2 taps, 2ms spacing, 20ms body, high Q |
| `hand_clap` | Natural-ish | 5 taps, 4.5ms spacing, 80ms body |

---

## Track 3: Infrastructure

### 3A. Choke Groups

**File**: `code_musics/score.py`

Add `choke_group: str | None = None` field to `Voice` dataclass and
`add_voice()`. Post-render operation: after all voice bases are rendered, apply
choke cuts across voices sharing a group.

**Algorithm**:

1. Group voices by `choke_group` string (skip None)
2. Collect all note onset times across all voices in each group
3. Sort chronologically
4. At each onset, apply 10ms linear fade-out to all OTHER voices in the group
   from that sample onward (zero after fade)

This is clean because it doesn't touch per-note rendering. Tiny CPU cost (just
array multiplication on already-rendered buffers).

```python
# Usage:
score.add_voice("open_hat", choke_group="hats", ...)
score.add_voice("closed_hat", choke_group="hats", ...)
# closed_hat note onset fades out any ringing open_hat
```

### 3B. Drum Bus Helper

**File**: `code_musics/drum_helpers.py` (new, ~60-80 lines)

Two convenience functions wrapping existing SendBusSpec/VoiceSend:

```python
def setup_drum_bus(score, *, bus_name="drum_bus", effects=None, return_db=0.0) -> str
def add_drum_voice(score, name, *, engine, preset, drum_bus=None,
                   send_db=0.0, choke_group=None, effects=None,
                   mix_db=0.0, normalize_peak_db=-6.0, **kwargs) -> Voice
```

Drum-friendly defaults: `normalize_peak_db=-6.0`, no `velocity_humanize`,
automatic send routing to drum bus.

### 3C. New Effect Presets

**File**: `code_musics/synth.py`

Compressor presets:

- `snare_punch`: 4ms attack, 120ms release, 3.0 ratio, -12 dB threshold, peak
  detector, SC HP at 100 Hz
- `snare_body`: 18ms attack, 200ms release, 2.0 ratio — lets transient through
- `hat_control`: 2ms attack, 60ms release, 2.5 ratio, -16 dB threshold

Saturation preset:

- `snare_bite`: triode mode, drive 1.8, mix 0.35, preserve_lows_hz=150

### 3D. Velocity-to-Timbre Examples

No code changes. Document recommended `VelocityParamMap` configurations for each
engine in docstrings and `synth_api.md`. The infrastructure already exists on Voice.

---

## Tests

### New test files

| File | Tests |
|---|---|
| `tests/test_engine_metallic_perc.py` | Render sanity, brightness/decay effect, ring mod adds content, closed < open decay, determinism |
| `tests/test_engine_snare.py` | Render sanity, body/wire mix spectral change, comb adds resonance, preset decay ordering, determinism |
| `tests/test_engine_clap.py` | Render sanity, tap count changes sound, multi-tap peaks visible in waveform, determinism |
| `tests/test_choke_groups.py` | Choke cuts other voices, None doesn't affect rendering, fade is smooth not hard cut |

### Existing tests to update

- `tests/test_engine_kick_tom.py`: Update for removed drive/lowpass params
- `tests/test_score.py`: Any tests referencing `drive_ratio` in kick_tom presets

---

## Implementation Order

### Phase 1: Polish existing (parallel)

- 1A: Remove kick_tom internal drive + lowpass
- 1B: noise_perc click envelope fix

### Phase 2: New engines (parallel, after Phase 1)

- 2A: metallic_perc engine + presets + registration + tests
- 2B: snare engine + presets + registration + tests
- 2C: clap engine + presets + registration + tests

### Phase 3: Infrastructure (after Phase 2)

- 3A: Choke groups on Voice + render pipeline
- 3B: Drum bus helper module
- 3C: New compressor/saturation presets
- 3D: Velocity-to-timbre documentation

### Phase 4: Integration & verification

- Preset tuning via spectral analysis (especially metallic_perc)
- Update existing pieces to use new engines where appropriate
- Update docs (synth_api.md, score_api.md, CLAUDE.md)
- `make all` passes

## Delegation Plan

- **Main context**: Orchestration, creative decisions, preset tuning review
- **Subagents (inherit model)**: Engine implementation (one per engine), choke group
  implementation, infrastructure/helpers, test writing
- **Subagents (sonnet)**: Doc updates, preset registration in registry.py,
  mechanical refactors (removing drive params from presets)

## Critical Files

- `code_musics/engines/kick_tom.py` — remove drive/lowpass
- `code_musics/engines/noise_perc.py` — click fix
- `code_musics/engines/metallic_perc.py` — new
- `code_musics/engines/snare.py` — new
- `code_musics/engines/clap.py` — new
- `code_musics/engines/registry.py` — register engines + all presets
- `code_musics/engines/_filters.py` — existing ZDF SVF (read-only, reuse)
- `code_musics/score.py` — choke_group field + render logic
- `code_musics/synth.py` — new compressor/saturation presets
- `code_musics/drum_helpers.py` — new convenience module
- `docs/synth_api.md` — engine + preset docs
- `docs/score_api.md` — choke group docs

## Verification

1. `make all` — format, lint, typecheck, full test suite passes
2. `make render PIECE=techno_studies` (or a new test piece) — verify drums render
   cleanly with new engines
3. Spectral analysis of metallic_perc presets — verify inharmonic partials,
   compare closed_hat/open_hat character
4. Listen tests via rendered WAVs for all new presets
