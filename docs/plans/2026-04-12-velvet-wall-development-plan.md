# Velvet Wall Development — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add timbral arc automation, a stochastic shimmer layer, and a piano voice in the Dissolve to velvet_wall.

**Architecture:** All changes are in one file (`code_musics/pieces/velvet_wall.py`). Task 1 adds automation to existing voices. Task 2 adds a new "shimmer" voice using generative tools. Task 3 adds a new "keys" voice with hand-written notes. Each task is independently useful.

**Tech Stack:** `AutomationSpec`/`AutomationTarget` for timbral arcs, `VelocityParamMap` for velocity-to-cutoff, `TonePool` + `stochastic_cloud` for shimmer, `piano` engine for keys.

**Key file:** `code_musics/pieces/velvet_wall.py`

**Design doc:** `docs/plans/2026-04-12-velvet-wall-development-design.md`

**Section constants for reference:**

- `EMERGE_START = 0.0`, `WALL_START = 90.0`, `DISSOLVE_START = 210.0`, `PIECE_END = 290.0`
- HOME arrival: 141-155s, Climax: 155-195s

---

## Task 1: Timbral Arc — Automation on Melody and Tendril

Add four automation lanes to existing voices so the timbral character evolves across the piece.

### Files

- Modify: `code_musics/pieces/velvet_wall.py`
  - Imports (~line 29): add `VelocityParamMap` to the score import
  - `_setup_voices()` (~line 273): add automation and velocity_to_params to melody voice, add automation to tendril voice

### Step 1: Add `VelocityParamMap` to imports

In the score import block (~line 37-44), add `VelocityParamMap`:

```python
from code_musics.score import (
    EffectSpec,
    NoteEvent,
    Phrase,
    Score,
    SendBusSpec,
    VelocityParamMap,
    VoiceSend,
)
```

### Step 2: Add automation to the melody voice

Replace the `score.add_voice("melody", ...)` call (~lines 273-294) to add three automation lanes and a velocity-to-params mapping. The `cutoff_hz` default in `synth_defaults` should change from 2200 to 1400 (the automation will drive it from there).

The new melody voice setup:

```python
score.add_voice(
    "melody",
    synth_defaults={
        "engine": "polyblep",
        "waveform": "saw",
        "osc2_waveform": "saw",
        "osc2_detune_cents": 6.0,
        "osc2_level": 0.7,
        "cutoff_hz": 1400.0,  # lower default; automation drives it
        "resonance_q": 0.08,
        "attack": 0.12,
        "decay": 0.4,
        "sustain_level": 0.75,
        "release": 1.2,
    },
    effects=[_melody_insert_delay()],
    envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
    velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
    velocity_to_params={
        "cutoff_hz": VelocityParamMap(min_value=-400.0, max_value=400.0),
    },
    normalize_lufs=-19.0,
    mix_db=-3.0,
    sends=[hall_send_wet],
    automation=[
        # -- Melody filter sweep: warm -> bright -> warm ---------------------
        AutomationSpec(
            target=AutomationTarget(kind="synth", name="cutoff_hz"),
            segments=(
                AutomationSegment(
                    start=0, end=90, shape="linear",
                    start_value=1400, end_value=1800,
                ),
                AutomationSegment(
                    start=90, end=155, shape="linear",
                    start_value=1800, end_value=3200,
                ),
                AutomationSegment(
                    start=155, end=175, shape="linear",
                    start_value=3200, end_value=4000,
                ),
                AutomationSegment(
                    start=175, end=210, shape="linear",
                    start_value=4000, end_value=2800,
                ),
                AutomationSegment(
                    start=210, end=290, shape="linear",
                    start_value=2800, end_value=1200,
                ),
            ),
        ),
        # -- Melody osc2 detune: tight -> wide -> tight ----------------------
        AutomationSpec(
            target=AutomationTarget(kind="synth", name="osc2_detune_cents"),
            segments=(
                AutomationSegment(
                    start=0, end=90, shape="hold", value=6.0,
                ),
                AutomationSegment(
                    start=90, end=195, shape="linear",
                    start_value=6.0, end_value=14.0,
                ),
                AutomationSegment(
                    start=195, end=290, shape="linear",
                    start_value=14.0, end_value=5.0,
                ),
            ),
        ),
    ],
)
```

**Note on VelocityParamMap mode:** The velocity-to-cutoff mapping uses `mode="add"` behavior. Check whether `VelocityParamMap` values are absolute or additive. From the API exploration: `VelocityParamMap` resolves to an absolute value via `np.interp`, and that value **overwrites** the synth param. Since we want it additive (relative to the automation baseline), we need to check: does automation or velocity_to_params win? Per the code (`_prepare_voice_notes` at line 1134-1139), velocity_to_params is applied **after** automation, overwriting it. So velocity_to_params should produce absolute values, not offsets.

**Revised approach:** Don't use `velocity_to_params` for `cutoff_hz` since it would fight the automation. Instead, rely on the automation alone for the filter arc. The velocity shaping is a nice-to-have but conflicts with per-note automation resolution. Skip it for now — the filter sweep is the main win.

Simplified: remove the `velocity_to_params` line from the voice setup. The four automation segments are the change.

### Step 3: Add automation to the tendril voice

Replace the `score.add_voice("tendril", ...)` call (~lines 297-331) to add a `mod_index` automation lane:

```python
score.add_voice(
    "tendril",
    synth_defaults={
        "engine": "fm",
        "carrier_ratio": 1.0,
        "mod_ratio": 3.5,
        "mod_index": 1.8,
        "index_decay": 0.6,
        "feedback": 0.08,
        "attack": 0.08,
        "decay": 0.5,
        "sustain_level": 0.60,
        "release": 1.8,
    },
    effects=[
        EffectSpec(
            "mod_delay",
            {
                "delay_ms": 220.0,
                "mod_rate_hz": 0.12,
                "mod_depth_ms": 6.0,
                "feedback": 0.30,
                "feedback_lpf_hz": 3200.0,
                "stereo_offset_deg": 95.0,
                "mix": 0.18,
            },
        ),
    ],
    envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
    velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
    normalize_lufs=-20.0,
    mix_db=-4.0,
    pan=-0.20,
    sends=[hall_send_wet],
    automation=[
        # -- Tendril FM brightness: moderate -> metallic -> gentle -----------
        AutomationSpec(
            target=AutomationTarget(kind="synth", name="mod_index"),
            segments=(
                AutomationSegment(
                    start=0, end=155, shape="hold", value=1.8,
                ),
                AutomationSegment(
                    start=155, end=185, shape="linear",
                    start_value=1.8, end_value=3.0,
                ),
                AutomationSegment(
                    start=185, end=290, shape="linear",
                    start_value=3.0, end_value=0.8,
                ),
            ),
        ),
    ],
)
```

### Step 4: Run tests

```
make test
```

Expected: all tests pass (especially the velvet_wall smoke test in `test_pieces_smoke.py`).

### Step 5: Render a snippet to verify

```
make snippet PIECE=velvet_wall AT=2:30 WINDOW=30 ANALYSIS=0
```

This renders the climax area where the filter should be near peak brightness and the osc2 detune widest. Compare with the previous render — the melody should sound brighter and wider here.

Also:

```
make snippet PIECE=velvet_wall AT=4:10 WINDOW=20 ANALYSIS=0
```

The Dissolve — melody should sound muted and intimate, tendril nearly sine-like.

---

## Task 2: Piano Voice in the Dissolve

Add a new "keys" voice using the `piano` engine with the `septimal` preset. Hand-place 7 sparse notes in the Dissolve section.

### Files

- Modify: `code_musics/pieces/velvet_wall.py`
  - `_setup_voices()` (~after line 371): add the keys voice
  - `_write_dissolve()`: add piano note-writing

### Step 1: Add the keys voice in `_setup_voices`

After the ghost voice setup (~line 371), add:

```python
# -- keys: piano punctuation in the dissolve — mortal against eternal ----
score.add_voice(
    "keys",
    synth_defaults={
        "engine": "piano",
        "preset": "septimal",
        "decay_base": 4.5,
        "soundboard_color": 0.50,
        "soundboard_brightness": 0.45,
        "attack": 0.003,
        "sustain_level": 1.0,
        "release": 0.3,
    },
    envelope_humanize=EnvelopeHumanizeSpec(preset="subtle_analog"),
    velocity_humanize=VelocityHumanizeSpec(preset="subtle_living"),
    normalize_lufs=-22.0,
    mix_db=-4.0,
    pan=0.10,
    sends=[hall_send],
)
```

Notes: no sympathetic resonance params — the piano engine's physical model
provides its own resonance. The `septimal` preset gives 7-limit partial
ratios for timbre-harmony fusion. `decay_base=4.5` gives moderately long
tails. Moderate hall send for blending.

### Step 2: Add piano notes in `_write_dissolve`

In `_write_dissolve()`, after the melody/tendril interlocking section
(~after line 927, before the "Fragmenting" comment), add a section for
the piano. Use a simple helper or direct `score.add_note` calls:

```python
# -- Piano punctuation: sparse, decaying notes against the pad wash ------
# These are moments of clarity — physical, mortal sounds in the eternal wash.
# Over HOME (210-220s)
score.add_note(
    "keys", partial=5 / 4, start=214.0, duration=5.0, amp_db=-8, velocity=0.68
)
score.add_note(
    "keys", partial=3 / 2, start=219.0, duration=4.0, amp_db=-9, velocity=0.62
)
# Over SUSPENDED (220-232s)
score.add_note(
    "keys", partial=2.0, start=224.0, duration=4.5, amp_db=-8, velocity=0.65
)
score.add_note(
    "keys",
    partial=3 / 2,
    start=229.0,
    duration=5.0,
    amp_db=-9,
    velocity=0.60,
    pitch_motion=PitchMotionSpec.ratio_glide(8 / 7, 3 / 2),
)
# Fragmenting (235-255s) — sparser, quieter
score.add_note(
    "keys", partial=5 / 4, start=238.0, duration=4.0, amp_db=-10, velocity=0.55
)
score.add_note(
    "keys", partial=7 / 4, start=246.0, duration=5.0, amp_db=-10, velocity=0.52
)
# Final passage — resolving with the piece
score.add_note(
    "keys", partial=3 / 2, start=256.0, duration=6.0, amp_db=-10, velocity=0.50
)
```

### Step 3: Run tests

```
make test
```

### Step 4: Render a snippet of the Dissolve

```
make snippet PIECE=velvet_wall AT=3:50 WINDOW=40 ANALYSIS=0
```

Check that the piano notes are audible against the pads, decaying naturally,
and blending into the reverb space.

---

## Task 3: Stochastic Shimmer Layer

Add a new "shimmer" voice using `TonePool` + `stochastic_cloud` for
conservative, ambient, high-register sparkle during the Wall and early Dissolve.

### Files

- Modify: `code_musics/pieces/velvet_wall.py`
  - Imports (~line 29): add `TonePool`, `stochastic_cloud`
  - `_setup_voices()`: add the shimmer voice
  - `_write_wall()`: add stochastic cloud phrases

### Step 1: Add imports

```python
from code_musics.generative.cloud import stochastic_cloud
from code_musics.generative.tone_pool import TonePool
```

### Step 2: Add the shimmer voice in `_setup_voices`

After the keys voice, add:

```python
# -- shimmer: stochastic high-register sparkle, dissolves into reverb ----
score.add_voice(
    "shimmer",
    synth_defaults={
        "engine": "additive",
        "n_harmonics": 4,
        "harmonic_rolloff": 0.30,
        "attack": 0.3,
        "decay": 0.5,
        "sustain_level": 0.60,
        "release": 4.0,
    },
    normalize_lufs=-28.0,
    mix_db=-10.0,
    pan=0.0,
    sends=[VoiceSend(target="hall", send_db=-4.0)],
)
```

Very quiet (mix_db=-10, normalize_lufs=-28), heavy hall send (send_db=-4).
The shimmer is meant to be subliminal.

### Step 3: Define the TonePool and generate cloud phrases

In `_write_wall()`, after the ghost voice notes and before the transition
melody (~after line 825, before the "Transition to dissolve" comment), add:

```python
# -- Stochastic shimmer: high-register sparkle through the wall ----------
shimmer_pool = TonePool.weighted({
    2.0: 5.0,   # octave — safest
    4.0: 4.0,   # double octave
    3.0: 3.5,   # twelfth
    6.0: 2.5,   # high fifth
    5 / 2: 2.0, # major 10th
    5.0: 1.5,   # high major 3rd
    7 / 2: 1.0, # septimal 7th — rare color
    7.0: 0.5,   # high septimal — very rare
})

# Wall entry: very sparse shimmer (105-141s)
wall_shimmer = stochastic_cloud(
    tones=shimmer_pool,
    duration=36.0,
    density=[(0.0, 0.2), (0.5, 0.35), (1.0, 0.25)],
    amp_db_range=(-18.0, -14.0),
    note_dur_range=(2.0, 5.0),
    pitch_kind="partial",
    seed=77,
)
score.add_phrase("shimmer", wall_shimmer, start=105.0)

# Climax: denser shimmer (155-190s)
climax_shimmer = stochastic_cloud(
    tones=shimmer_pool,
    duration=35.0,
    density=[(0.0, 0.4), (0.4, 0.8), (0.7, 0.7), (1.0, 0.3)],
    amp_db_range=(-16.0, -12.0),
    note_dur_range=(1.5, 4.0),
    pitch_kind="partial",
    seed=78,
)
score.add_phrase("shimmer", climax_shimmer, start=155.0)

# Dissolve: sparse fading shimmer (210-260s)
dissolve_shimmer = stochastic_cloud(
    tones=shimmer_pool,
    duration=50.0,
    density=[(0.0, 0.3), (0.3, 0.2), (0.7, 0.1), (1.0, 0.05)],
    amp_db_range=(-18.0, -14.0),
    note_dur_range=(3.0, 6.0),
    pitch_kind="partial",
    seed=79,
)
score.add_phrase("shimmer", dissolve_shimmer, start=210.0)
```

Note: the density values are notes-per-second. 0.25 = one note every 4s,
0.8 = roughly one note every 1.25s. These are conservative — the shimmer
should add texture without cluttering.

### Step 4: Run tests

```
make test
```

### Step 5: Render the full piece

```
make render PIECE=velvet_wall
```

Check:

- The shimmer should be barely noticeable on first listen — it adds
  "life" to the Wall without being identifiable as a separate voice.
- At the climax (~2:35-3:10), the shimmer should be most active.
- In the Dissolve, it should thin out and fade.

---

## Verification (after all tasks)

1. `make all` — full quality gate
2. `make render PIECE=velvet_wall` — full render with plot and analysis
3. Listen/inspect the key moments:
   - 1:00 — melody should sound warm and muted (filter at ~1600 Hz)
   - 2:00 — melody brighter (filter ~2800 Hz), osc2 detune wider
   - 2:50 — climax peak: melody brightest, tendril most metallic, shimmer densest
   - 3:40 — Dissolve: melody closing down, tendril softening, piano enters
   - 4:20 — ending: melody very muted, tendril nearly sine, piano fading, shimmer gone
