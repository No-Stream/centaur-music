# Synth API Reference

This document describes the synth-facing API used by `Voice.synth_defaults` and
per-note `synth={...}` overrides.

The rendering path is frequency-first:

- `NoteEvent.partial` resolves against `Score.f0`
- `NoteEvent.freq` uses an absolute frequency directly
- optional `NoteEvent.pitch_motion` expands that base pitch into a per-sample trajectory
- synth engines receive a concrete `freq` in Hz, not a MIDI note number

## Where This Is Used

- [code_musics/score.py](/home/jan/workspace/code-musics/code_musics/score.py) merges voice-level and note-level synth params, resolves presets, and dispatches to the requested engine
- [code_musics/engines/registry.py](/home/jan/workspace/code-musics/code_musics/engines/registry.py) defines the engine registry and preset map
- [code_musics/engines/additive.py](/home/jan/workspace/code-musics/code_musics/engines/additive.py)
- [code_musics/engines/fm.py](/home/jan/workspace/code-musics/code_musics/engines/fm.py)
- [code_musics/engines/filtered_stack.py](/home/jan/workspace/code-musics/code_musics/engines/filtered_stack.py)
- [code_musics/engines/noise_perc.py](/home/jan/workspace/code-musics/code_musics/engines/noise_perc.py)

## Parameter Resolution

Synth params are resolved in this order:

1. preset values, if `preset` is set
2. voice-level `synth_defaults`
3. note-level `synth` overrides

Explicit params always override preset values.

If `engine` is omitted, it defaults to `additive`.

## Shared Parameters

These are consumed by the score renderer after the engine returns a raw mono signal:

- `amp_db: float`
- `attack: float`
- `decay: float`
- `sustain_level: float`
- `release: float`

`amp_db` is the recommended authoring control for note loudness; it is converted
to the renderer's linear `amp` internally. Linear `amp` is still supported, but
it is less intuitive for balancing voices.

These control the ADSR envelope applied in [code_musics/synth.py](/home/jan/workspace/code-musics/code_musics/synth.py).

The composition helper layer may also attach note-level `attack_scale` and
`release_scale` values inside `NoteEvent.synth`; the score renderer applies them
after merging voice defaults and note overrides.

## Velocity and Expression Resolution

Velocity is now a core part of the render path.

At render time, note loudness and timbre are resolved in roughly this order:

1. note `amp` / `amp_db` becomes the base amplitude
2. note `velocity` is combined with any `velocity_humanize` multiplier
3. the resolved velocity is converted to a dB offset via `velocity_db_per_unit`
4. any `velocity_to_params` mappings write synth params from the resolved velocity
5. the engine renders with those params
6. ADSR is applied, including any `envelope_humanize`

Important consequences:

- `amp_db` is still the main mix-level control
- `velocity` is the main note-expression control
- `velocity_to_params` is how velocity becomes timbral, not just louder/quieter
- `envelope_humanize` happens after synth params are merged, so it can gently vary
  the final attack/release behavior even when base ADSR values come from presets

### `velocity_db_per_unit`

Each voice has a `velocity_db_per_unit` setting that controls how much a
resolved velocity above or below `1.0` changes loudness.

Default:

- `12.0 dB` per velocity unit

So, very roughly:

- `velocity=1.0` means no extra dB offset
- `velocity=1.1` adds a modest positive dB bump
- `velocity=0.9` subtracts a modest dB bump

Set it lower if you want velocity to behave more like timbral emphasis than a
large loudness swing. Set it to `0.0` if you want velocity to affect only mapped
parameters and not level.

### `velocity_to_params`

`velocity_to_params` maps the resolved velocity into synth parameters before the
engine renders.

This is the preferred mechanism for things like:

- brighter accents
- stronger filter opening on louder notes
- more aggressive FM index on accented hits
- noisier or sharper transients on higher-velocity percussion

Example:

```python
from code_musics.score import VelocityParamMap

score.add_voice(
    "bass",
    synth_defaults={"engine": "filtered_stack", "preset": "round_bass"},
    velocity_humanize=None,
    velocity_to_params={
        "cutoff_hz": VelocityParamMap(
            min_value=250.0,
            max_value=1600.0,
            min_velocity=0.8,
            max_velocity=1.2,
        )
    },
)
```

## Envelope Variation

`EnvelopeHumanizeSpec` is not an engine-specific preset system; it is a
renderer-level variation layer applied after the base synth params are resolved.

That means:

- presets still define the baseline ADSR shape
- note-level synth overrides can still set explicit ADSR values
- envelope humanization then adds smooth, bounded variation around those values

This is the right place for subtle analog inconsistency or "env slop" rather
than inventing separate per-engine randomness knobs.

## Pitch Motion

Pitch motion is score-level note metadata, not an engine preset parameter.

The current motion helpers live in
[code_musics/pitch_motion.py](/home/jan/workspace/code-musics/code_musics/pitch_motion.py)
and attach to `NoteEvent.pitch_motion`.

Supported motion kinds:

- `linear_bend`
- `ratio_glide`
- `vibrato`

Current engine support:

- `additive`
- `fm`
- `filtered_stack`

Unsupported in v1:

- `noise_perc`

If pitch motion is attached to a note rendered through `noise_perc`, rendering
raises `ValueError` rather than silently ignoring the motion.

## Engine Selection

Set the engine with:

```python
synth_defaults = {"engine": "fm"}
```

Available engines:

- `additive`
- `fm`
- `filtered_stack`
- `noise_perc`

## Presets

Set a preset with:

```python
synth_defaults = {"engine": "filtered_stack", "preset": "warm_pad"}
```

Available presets:

- `additive`: `soft_pad`, `drone`, `bright_pluck`
- `fm`: `bell`, `glass_lead`, `metal_bass`
- `filtered_stack`: `warm_pad`, `reed_lead`, `round_bass`
- `noise_perc`: `kickish`, `snareish`, `tick`

## Effects

Effects are attached with `EffectSpec(kind, params)` on either `Voice.effects`
or `Score.master_effects`.

The effect path is now stereo-aware:

- mono effects can be chained before or after stereo effects
- once an effect returns stereo, later effects keep working in stereo
- `Score.render()` may therefore return either mono or stereo depending on the effect chain

Effects that support presets resolve parameters in this order:

1. effect preset values, if `preset` is set
2. explicit `EffectSpec.params` overrides

### `chorus`

Implementation: [code_musics/synth.py](/home/jan/workspace/code-musics/code_musics/synth.py)

Warm stereo chorus inspired by classic analog/BBD units and intended for subtle
depth rather than obvious wobble.

Parameters:

- `preset: str`
  Supported presets: `juno_subtle`, `juno_wide`, `ensemble_soft`
- `mix: float`
  Dry/wet blend from `0` to `1`. Typical musical use is around `0.2` to `0.33`.
- `rate_hz: float`
  Base LFO rate in Hertz.
- `depth_ms: float`
  Modulation depth in milliseconds.
- `center_delay_ms: float`
  Base chorus delay time in milliseconds.
- `stereo_phase_deg: float`
  Phase offset between left and right modulation.
- `feedback: float`
  Very light recirculation on the wet path.
- `wet_lowpass_hz: float`
  Darkens the wet path to keep the effect smooth.
- `wet_highpass_hz: float`
  Removes low-end smear from the wet path.
- `drift_amount: float`
  Adds a slower secondary modulation for analog drift.
- `wet_saturation: float`
  Adds slight nonlinearity on the wet path so the chorus feels less sterile.

Notes:

- chorus promotes mono input to stereo
- `juno_subtle` is the safest general-purpose default

Example:

```python
score.add_voice(
    "pad",
    synth_defaults={"engine": "filtered_stack", "preset": "warm_pad"},
    effects=[EffectSpec("chorus", {"preset": "juno_subtle", "mix": 0.28})],
)
```

### `saturation`

Implementation: [code_musics/synth.py](/home/jan/workspace/code-musics/code_musics/synth.py)

Subtle analog-style warming stage intended for tube/iron/preamp color rather
than obvious distortion.

Parameters:

- `preset: str`
  Supported presets: `tube_warm`, `iron_soft`, `neve_gentle`
- `drive: float`
  Amount of nonlinearity. Keep this conservative for bus sweetening.
- `mix: float`
  Dry/wet blend from `0` to `1`.
- `bias: float`
  Asymmetry control that nudges the transfer toward even-order warmth.
- `even_harmonics: float`
  Blend between a more symmetric and more asymmetric saturation curve.
- `oversample_factor: int`
  Oversampling factor used around the nonlinear stage.
- `highpass_hz: float`
  Removes excessive sub/DC before saturation.
- `tone_tilt: float`
  Tilts more low-mid or upper-mid energy into the nonlinear stage.
- `output_lowpass_hz: float`
  Smooths the post-saturation output.
- `compensation: bool`
  Applies output gain compensation so the result does not simply get louder.

Notes:

- default tuning is intentionally subtle enough for “always on” use
- `tube_warm` is the safest default

Example:

```python
score = Score(
    f0=110.0,
    master_effects=[
        EffectSpec("saturation", {"preset": "tube_warm", "mix": 0.24}),
        EffectSpec("reverb", {"room_size": 0.65, "damping": 0.45, "wet_level": 0.22}),
    ],
)
```

### `chow_tape`

Implementation: [code_musics/synth.py](/home/jan/workspace/code-musics/code_musics/synth.py)

Wrapper around the Chow Tape Model VST3 for tape-style saturation/color.

Parameters:

- `drive: float`
  Tape drive amount from `0` to `1`.
- `saturation: float`
  Tape saturation density from `0` to `1`.
- `bias: float`
  Tape bias from `0` to `1`, affecting harmonic balance.
- `mix: float`
  Dry/wet blend in percent from `0` to `100`.

Notes:

- expects `~/.vst3/CHOWTapeModel.vst3` to be installed
- the wrapper currently disables wow/flutter and tape-loss switches so this is a
  clean tape-saturation comparison rather than a lo-fi motion effect

Example:

```python
score = Score(
    f0=110.0,
    master_effects=[
        EffectSpec(
            "chow_tape",
            {"drive": 0.62, "saturation": 0.58, "bias": 0.52, "mix": 72.0},
        )
    ],
)
```

## `additive`

Implementation: [code_musics/engines/additive.py](/home/jan/workspace/code-musics/code_musics/engines/additive.py)

Parameters:

- `n_harmonics: int`
  Number of harmonic partials to include before Nyquist limiting.
- `harmonic_rolloff: float`
  Geometric amplitude falloff across harmonics. Lower values suppress upper partials more strongly.
- `brightness_tilt: float`
  Additional harmonic weighting by partial index. Positive values emphasize higher harmonics; negative values bias toward lower ones.
- `odd_even_balance: float`
  Bias between odd and even partials. Positive values strengthen odd harmonics and reduce even harmonics; negative values do the reverse.
- `detune_cents: float`
  Total detune spread used when `unison_voices > 1`.
- `unison_voices: int`
  Number of detuned additive copies to average together.

Notes:

- Omitting the new parameters preserves the old additive behavior closely.
- `odd_even_balance` is clamped internally to avoid zeroing the spectrum too aggressively.

Example:

```python
score.add_voice(
    "pad",
    synth_defaults={
        "engine": "additive",
        "n_harmonics": 10,
        "harmonic_rolloff": 0.55,
        "brightness_tilt": -0.1,
        "unison_voices": 3,
        "detune_cents": 5.0,
        "attack": 0.4,
        "release": 1.2,
    },
)
```

## `fm`

Implementation: [code_musics/engines/fm.py](/home/jan/workspace/code-musics/code_musics/engines/fm.py)

Parameters:

- `carrier_ratio: float`
  Multiplier applied to the resolved note frequency for the carrier oscillator.
- `mod_ratio: float`
  Multiplier applied to the resolved note frequency for the modulator oscillator.
- `mod_index: float`
  Base modulation index. Larger values produce broader sideband spread.
- `feedback: float`
  Feedback amount applied to the modulator phase.
- `index_decay: float`
  Duration in seconds over which the modulation index decays from full strength toward `index_sustain`.
- `index_sustain: float`
  Post-decay multiplier applied to `mod_index`.

Validation:

- `carrier_ratio > 0`
- `mod_ratio > 0`
- `mod_index >= 0`
- `index_decay >= 0`

Example:

```python
score.add_voice(
    "lead",
    synth_defaults={
        "engine": "fm",
        "carrier_ratio": 1.0,
        "mod_ratio": 7 / 4,
        "mod_index": 2.8,
        "index_decay": 0.1,
        "index_sustain": 0.45,
        "attack": 0.02,
        "release": 0.35,
    },
)
```

## `filtered_stack`

Implementation: [code_musics/engines/filtered_stack.py](/home/jan/workspace/code-musics/code_musics/engines/filtered_stack.py)

Parameters:

- `waveform: str`
  Source harmonic weighting. Supported values: `saw`, `square`, `pulse`, `triangle`.
- `n_harmonics: int`
  Number of source harmonics to generate before Nyquist limiting.
- `cutoff_hz: float`
  Base low-pass cutoff in Hertz.
- `keytrack: float`
  Exponent controlling how strongly the cutoff follows note pitch relative to `reference_freq_hz`.
- `reference_freq_hz: float`
  Reference pitch for key tracking. When the note frequency equals this value, the effective cutoff is `cutoff_hz` before envelope modulation.
- `resonance: float`
  Extra weighting around the moving cutoff region.
- `filter_env_amount: float`
  Multiplier controlling how much the cutoff starts above the base `cutoff_hz`.
- `filter_env_decay: float`
  Time constant in seconds for the cutoff envelope to decay back toward the base cutoff.
- `pulse_width: float`
  Pulse width used when `waveform="pulse"`.

Validation:

- `n_harmonics >= 1`
- `cutoff_hz > 0`
- `reference_freq_hz > 0`
- `filter_env_decay > 0`
- `0 < pulse_width < 1`

Example:

```python
score.add_voice(
    "bass",
    synth_defaults={
        "engine": "filtered_stack",
        "waveform": "square",
        "n_harmonics": 12,
        "cutoff_hz": 550.0,
        "keytrack": 0.15,
        "resonance": 0.15,
        "filter_env_amount": 0.8,
        "filter_env_decay": 0.18,
        "attack": 0.01,
        "release": 0.3,
    },
)
```

## `noise_perc`

Implementation: [code_musics/engines/noise_perc.py](/home/jan/workspace/code-musics/code_musics/engines/noise_perc.py)

Parameters:

- `noise_mix: float`
  Blend between pitched tone and noise layer. `0` is fully pitched; `1` is fully noise-driven.
- `pitch_decay: float`
  Decay time in seconds for the noise-focused transient envelope.
- `tone_decay: float`
  Decay time in seconds for the pitched tone body.
- `bandpass_ratio: float`
  Ratio applied to the resolved note frequency to choose the center of the noise shaping band.
- `click_amount: float`
  Level of the short transient click layer.

Validation:

- `0 <= noise_mix <= 1`
- `pitch_decay > 0`
- `tone_decay > 0`
- `bandpass_ratio > 0`
- `click_amount >= 0`

Notes:

- This engine is deterministic for identical inputs, which helps tests and repeatable rendering.

Example:

```python
score.add_voice(
    "perc",
    synth_defaults={
        "engine": "noise_perc",
        "noise_mix": 0.7,
        "pitch_decay": 0.04,
        "tone_decay": 0.12,
        "bandpass_ratio": 1.5,
        "click_amount": 0.1,
    },
)
```
