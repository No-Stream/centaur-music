# Synth API Reference

This document describes the synth-facing API used by `Voice.synth_defaults` and
per-note `synth={...}` overrides.

The rendering path is frequency-first:

- `NoteEvent.partial` resolves against `Score.f0`
- `NoteEvent.freq` uses an absolute frequency directly
- optional `NoteEvent.pitch_motion` expands that base pitch into a per-sample trajectory
- optional automation can also generate a per-sample pitch-ratio trajectory or
  note-start synth-param changes
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
5. voice and note automation can adjust supported synth params
6. the engine renders with those params
7. ADSR is applied, including any `envelope_humanize`

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
- `polyblep`

Unsupported in v1:

- `noise_perc`

If pitch motion is attached to a note rendered through `noise_perc`, rendering
raises `ValueError` rather than silently ignoring the motion.

## Automation

Automation is now a first-class score surface alongside velocity and humanization.

Current v1 targets:

- `pitch_ratio` for per-sample pitch motion
- supported synth params for note-start modulation:
  `cutoff_hz`, `resonance`, `brightness_tilt`, `filter_env_amount`,
  `mod_index`, `attack`, `decay`, `sustain_level`, and `release`

Attachment points:

- `Voice.automation` for score-time lanes
- `NoteEvent.automation` for note-local lanes

Important v1 limits:

- `pitch_ratio` automation is per-sample
- synth-param automation is sampled at note start
- `pitch_motion` and `pitch_ratio` automation cannot be combined on the same note

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
- `polyblep`

## Presets

Set a preset with:

```python
synth_defaults = {"engine": "filtered_stack", "preset": "warm_pad"}
```

Available presets:

- `additive`: `soft_pad`, `drone`, `bright_pluck`, `organ`
- `fm`: `bell`, `glass_lead`, `metal_bass`, `dx_piano`, `lately_bass`, `fm_clav`, `fm_mallet`, `chorused_ep`
- `filtered_stack`: `warm_pad`, `reed_lead`, `round_bass`, `saw_pad`, `string_pad`
- `noise_perc`: `kickish`, `snareish`, `tick`
- `polyblep`: `warm_lead`, `synth_pluck`, `analog_brass`, `square_lead`, `hoover`, `moog_bass`, `sync_lead`, `acid_bass`, `sub_bass`, `resonant_sweep`, `soft_square_pad`

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

### `plugin`

Implementation: [code_musics/synth.py](/home/jan/workspace/code-musics/code_musics/synth.py)

Generic external-plugin effect. Use this when you want to host a plugin directly
instead of adding a dedicated wrapper kind.

Parameters:

- `plugin_name: str | None`
  Registered plugin id. Current built-in ids are `chow_tape`, `tal_chorus_lx`,
  `tal_reverb2`, `dragonfly_plate`, `dragonfly_room`, `dragonfly_hall`,
  `dragonfly_early`, `lsp_compressor_stereo`, `lsp_limiter_stereo`, and
  `lsp_compressor_stereo_vst2`.
- `plugin_path: str | None`
  Explicit plugin path. Use this for ad hoc plugins that are not in the registry.
- `plugin_format: str`
  Plugin format identifier. Default `vst3`.
- `host: str`
  Plugin host backend. Default `pedalboard`.
- `params: dict[str, Any]`
  Raw plugin parameter map. Keys are applied as plugin attributes.

Notes:

- the current backend supports VST3 plugins through `pedalboard`
- `lsp_compressor_stereo` targets the multi-plugin LSP VST3 bundle and preloads a
  small local Cairo runtime shim before loading `Compressor Stereo`
- `lsp_limiter_stereo` targets the same LSP VST3 bundle and loads `Limiter Stereo`
- `lsp_compressor_stereo_vst2` remains registered as the direct Linux VST2 `.so`
  target, but it still needs a future VST2-capable backend before it can run here
- Linux `.so` / VST2 / LV2 are not hosted yet; the abstraction is in place so a
  future backend can be added without redesigning `EffectSpec`
- mono input is promoted to stereo before plugin processing, then matched back to
  the original layout when possible

Example:

```python
score = Score(
    f0=110.0,
    master_effects=[
        EffectSpec(
            "plugin",
            {
                "plugin_path": "~/.vst3/MyBusComp.vst3",
                "params": {"threshold_db": -18.0, "ratio": 2.0},
            },
        )
    ],
)
```

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

Presets:

- `soft_pad` - mellow harmonic pad with light additive detune.
- `drone` - slower, darker sustained additive bed.
- `bright_pluck` - compact harmonic pluck with a brighter attack.
- `organ` - steady drawbar-ish additive organ with fast attack and minimal decay.

Presets:

- `bell` - bright struck FM bell with a decaying modulation index.
- `glass_lead` - glassy sustained lead with moderate bite.
- `metal_bass` - metallic low-register FM bass.
- `dx_piano` - classic DX-style electric piano gesture with a bright attack and softer sustain.
- `lately_bass` - punchy FM bass inspired by late-80s / early-90s digital bass presets.
- `fm_clav` - short, bright FM clavinet-like attack with a dry percussive body.

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
- implemented through the shared external-plugin backend
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

### `tal_chorus_lx`

Implementation: [code_musics/synth.py](/home/jan/workspace/code-musics/code_musics/synth.py)

Wrapper around the TAL-Chorus-LX VST3, a Roland Juno-60 BBD chorus emulation.

Parameters:

- `mix: float`
  Dry/wet blend from `0` to `1`.
- `chorus_1: bool`
  Enable chorus mode I (subtle, slower LFO). Default `True`.
- `chorus_2: bool`
  Enable chorus mode II (wider, faster LFO). Default `False`. Both modes can be
  enabled simultaneously.
- `stereo: float`
  Stereo width from `0` to `1`. Default `1.0`.

Notes:

- expects `~/.vst3/TAL-Chorus-LX.vst3` to be installed
- implemented through the shared external-plugin backend
- promotes mono to stereo
- mode I alone is the closest match to a classic Juno-60 Chorus I sound; enabling
  both gives the Juno Chorus I+II character (wider, slightly faster)

Example:

```python
score.add_voice(
    "pad",
    effects=[EffectSpec("tal_chorus_lx", {"mix": 0.25, "chorus_1": True})],
)
```

### `bricasti`

Implementation: [code_musics/synth.py](/home/jan/workspace/code-musics/code_musics/synth.py)

Stereo convolution reverb wrapper around the local Bricasti impulse responses.
This is the repo's main IR reverb path for more realistic rooms/halls than the
built-in algorithmic reverb.

Parameters:

- `ir_name: str`
  Bricasti impulse-response name. The wrapper looks for matching `44K L/R.wav`
  files under the configured IR directory.
- `wet: float`
  Dry/wet blend from `0` to `1`. Default `0.35`.
- `highpass_hz: float`
  Optional post-convolution high-pass on the wet return. Useful for clearing mud
  in the low mids and subs. Default `0.0` (disabled).
- `lowpass_hz: float`
  Optional post-convolution low-pass on the wet return. Useful for darker, less
  splashy tails. Default `0.0` (disabled).
- `tilt_db: float`
  Optional tilt-EQ amount on the wet return. Positive values brighten the tail;
  negative values darken it. Default `0.0`.
- `tilt_pivot_hz: float`
  Pivot frequency for the tilt EQ. Default `1500.0`.

Notes:

- uses a fully wet convolution internally, then applies the dry/wet blend in the
  wrapper so the tone controls only affect the reverb tail
- `highpass_hz` and `lowpass_hz` can be combined, but `highpass_hz` must stay
  below `lowpass_hz`
- a common starting point for cleaner halls is `highpass_hz=180` to `350`

Example:

```python
score.add_voice(
    "pad",
    effects=[
        EffectSpec(
            "bricasti",
            {
                "ir_name": "1 Halls 07 Large & Dark",
                "wet": 0.30,
                "highpass_hz": 220.0,
                "lowpass_hz": 8_500.0,
                "tilt_db": -1.5,
            },
        )
    ],
)
```

Presets:

- `bell` - bright struck FM bell with a decaying modulation index.
- `glass_lead` - glassy sustained lead with moderate bite.
- `metal_bass` - metallic low-register FM bass.
- `dx_piano` - classic DX-style electric piano gesture with a bright attack and softer sustain.
- `lately_bass` - punchy FM bass inspired by late-80s / early-90s digital bass presets.
- `fm_clav` - short, bright FM clavinet-like attack with a dry percussive body.
- `fm_mallet` - brighter struck mallet voice with a compact metallic bloom.
- `chorused_ep` - softer electric-piano core intended to pair well with chorus effects.

Presets:

- `warm_pad` - soft subtractive pad with a gentle opening filter sweep.
- `reed_lead` - square-based lead with a nasal midrange focus.
- `round_bass` - triangle-leaning low bass with restrained brightness.
- `saw_pad` - bread-and-butter saw pad with a slower opening filter and wider harmonic bed.
- `string_pad` - slower, more orchestral synth-string pad with a gentle top and long release.

### `tal_reverb2`

Implementation: [code_musics/synth.py](/home/jan/workspace/code-musics/code_musics/synth.py)

Wrapper around TAL-Reverb-2, a vintage-flavored algorithmic reverb with a warm
plate character.

Parameters:

- `wet: float`
  Wet level from `0` to `1`. Default `0.3`.
- `room_size: float`
  Room size from `0` to `1`. Default `0.75`.
- `pre_delay: float`
  Pre-delay from `0` to `1` (plugin's normalized range). Default `0.13`.
- `stereo: float`
  Stereo width from `0` to `1`. Default `1.0`.

Notes:

- expects `~/.vst3/TAL-Reverb-2.vst3` to be installed
- implemented through the shared external-plugin backend
- always sets dry to `1.0`; wet is an additive level, not a crossfade
- promotes mono to stereo

Example:

```python
score.add_voice(
    "lead",
    effects=[EffectSpec("tal_reverb2", {"wet": 0.22, "room_size": 0.60})],
)
```

Presets:

- `warm_pad` - soft subtractive pad with a gentle opening filter sweep.
- `reed_lead` - square-based lead with a nasal midrange focus.
- `round_bass` - triangle-leaning low bass with restrained brightness.
- `saw_pad` - bread-and-butter saw pad with a slower opening filter and wider harmonic bed.
- `string_pad` - slower, more orchestral synth-string pad with a gentle top and long release.

### `dragonfly`

Implementation: [code_musics/synth.py](/home/jan/workspace/code-musics/code_musics/synth.py)

Unified wrapper for the Dragonfly Reverb VST3 suite. Selects a plugin variant
via the `variant` parameter.

Parameters:

- `variant: str`
  Selects the plugin: `"plate"`, `"room"`, `"hall"`, or `"early"`.
- `wet_level: float`
  Wet level from `0` to `100` (percent). Default `20.0`.
- `decay_s: float`
  Reverb decay in seconds. Default `0.4`.
- `width: float`
  Stereo width. Plate range: `50`–`150`; room/hall range: `0`–`100`. Default `100.0`.
- `predelay_ms: float`
  Pre-delay in milliseconds. Default `0.0`.
- `low_cut_hz: float`
  Low-cut frequency in Hz. Supported by plate (`0`–`200`), room, hall.
- `high_cut_hz: float`
  High-cut frequency in Hz. Supported by plate, room, hall.
- `dampen_hz: float`
  Damping cutoff in Hz. Plate only. Default `13000.0`.
- `size_m: float`
  Room/hall size in metres. Room and hall only. Default `12.0`.
- `diffuse: float`
  Diffusion `0`–`100`. Room and hall only. Default `70.0`.

Notes:

- expects the appropriate `~/.vst3/DragonflY*.vst3` to be installed
- implemented through the shared external-plugin backend
- `dry_level` is always set to `100`; `wet_level` is an additive level
- promotes mono to stereo
- plate is the cleanest choice for voice-level pre-reverb; room and hall suit
  wider spatial treatments

Example:

```python
score.add_voice(
    "lead",
    effects=[
        EffectSpec("dragonfly", {
            "variant": "plate",
            "wet_level": 16.0,
            "decay_s": 0.55,
            "low_cut_hz": 350.0,
            "predelay_ms": 8.0,
        })
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
- `filter_mode: str`
  Filter response mode. Supported values: `lowpass`, `bandpass`, `highpass`, `notch`.
  Default `lowpass`.
- `filter_drive: float`
  Analog-inspired drive amount inside the ZDF filter path. `0.0` is clean; higher values
  thicken and soften the response. Default `0.0`.

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

## `polyblep`

Implementation: [code_musics/engines/polyblep.py](/home/jan/workspace/code-musics/code_musics/engines/polyblep.py)

Time-domain bandlimited oscillator using polynomial BLEPs (Bandlimited Step
functions). Generates waveforms directly rather than summing sine harmonics, so
there is no Gibbs phenomenon at discontinuities. The resulting sound has a smooth
analog character with a correct `1/n` harmonic spectrum, then passes through an
internal zero-delay-feedback / topology-preserving state-variable filter.

Parameters:

- `waveform: str`
  Oscillator waveform. Supported values: `saw`, `square`, `triangle`. Triangle is
  generated by integrating the bandlimited square (BLAMP approach) and is alias-free;
  `pulse_width` is ignored for triangle.
- `pulse_width: float`
  Pulse width used when `waveform="square"`. `0.5` is a symmetric square wave.
- `cutoff_hz: float`
  Base low-pass cutoff in Hertz. Default `3000.0`.
- `filter_mode: str`
  Filter response mode. Supported values: `lowpass`, `bandpass`, `highpass`,
  `notch`. Default `lowpass`.
- `keytrack: float`
  Exponent controlling how strongly the cutoff follows note pitch relative to
  `reference_freq_hz`. Default `0.0` (no tracking).
- `reference_freq_hz: float`
  Reference pitch for key tracking. When the note frequency equals this value, the
  effective cutoff is `cutoff_hz` before envelope modulation. Default `220.0`.
- `resonance: float`
  Resonance emphasis for the internal state-variable filter. Higher values sharpen
  the cutoff region and make the filter feel more analog/reactive. Default `0.0`.
- `filter_drive: float`
  Analog-inspired drive amount inside the filter path. `0` is clean; higher values
  thicken and soften the response. Default `0.0`.
- `filter_env_amount: float`
  Multiplier controlling how much the cutoff starts above the base `cutoff_hz` at
  note onset. Default `0.0`.
- `filter_env_decay: float`
  Time constant in seconds for the cutoff envelope to decay back toward the base
  cutoff. Default `0.18`.

Validation:

- `cutoff_hz > 0`
- `reference_freq_hz > 0`
- `filter_env_decay > 0`
- `filter_drive >= 0`
- `0 < pulse_width < 1`
- `waveform in {"saw", "square", "triangle"}`
- `filter_mode in {"lowpass", "bandpass", "highpass", "notch"}`

Notes:

- Unlike `filtered_stack`, the spectrum is generated in the time domain, so there
  is no `n_harmonics` cap and no Gibbs ringing at waveform discontinuities.
- The filter sweep is handled sample by sample through a topology-preserving
  state-variable filter, so modulation is smoother and more analog-like than the
  previous segmented approximation.
- The output is peak-normalized before the final amplitude scale, matching the
  behavior of the `fm` engine.
- Supports `freq_trajectory` for pitch motion.

Presets:

- `warm_lead` — saw wave with a gentle filter envelope, light resonance, and a
  stronger touch of filter drive, useful as a drop-in analog lead.
- `synth_pluck` - square-based subtractive pluck with a snappy filter envelope and more built-in drive.
- `analog_brass` - saw-based brass stab with a stronger filter push and noticeable drive.
- `square_lead` - classic square lead with a compact filter sweep and moderate analog grit.
- `hoover` - bright aggressive band-passed square voice meant for rave-ish sustain textures.
- `moog_bass` - driven lowpass saw bass with a punchy envelope and heavy filter drive.
- `sync_lead` - bright cutting lead with a more aggressive filtered edge.
- `acid_bass` - resonant lowpass bass with a sharp envelope and built-in squelch.
- `sub_bass` - simple low square bass with restrained brightness and a little saturation.
- `resonant_sweep` - animated resonant sweep patch for rising figures and timbral motion.
- `soft_square_pad` - mellow square-based pad with a slow envelope and gentle filter warmth.

Example:

```python
score.add_voice(
    "lead",
    synth_defaults={
        "engine": "polyblep",
        "waveform": "saw",
        "cutoff_hz": 2500.0,
        "filter_mode": "lowpass",
        "keytrack": 0.08,
        "resonance": 0.12,
        "filter_drive": 0.18,
        "filter_env_amount": 0.6,
        "filter_env_decay": 0.5,
        "attack": 0.01,
        "release": 0.3,
    },
)
```
