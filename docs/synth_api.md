# Synth API Reference

This document describes the synth-facing API used by `Voice.synth_defaults` and
per-note `synth={...}` overrides.

The recommended authoring style is now a structured synth spec with explicit
units in parameter names. The legacy flat dict interface still works and remains
fully supported for backward compatibility.

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
- [code_musics/engines/kick_tom.py](/home/jan/workspace/code-musics/code_musics/engines/kick_tom.py)
- [code_musics/engines/noise_perc.py](/home/jan/workspace/code-musics/code_musics/engines/noise_perc.py)

## Canonical Authoring Shape

Preferred shape:

```python
synth_defaults = {
    "engine": "polyblep",
    "preset": "warm_lead",
    "env": {
        "attack_ms": 30.0,
        "decay_ms": 180.0,
        "sustain_ratio": 0.62,
        "release_ms": 280.0,
    },
    "params": {
        "cutoff_hz": 2200.0,
        "resonance_ratio": 0.12,
        "filter_env_depth_ratio": 0.7,
        "filter_env_decay_ms": 240.0,
        "filter_drive_ratio": 0.18,
    },
}
```

Compatibility notes:

- flat legacy keys like `attack`, `release`, `sustain_level`, and
  `filter_env_decay` still work
- flat unit-bearing aliases like `attack_ms` and `release_ms` also work
- if both legacy and new names are present, the explicit unit-bearing / new names win
- nested `env={...}` and `params={...}` are normalized before voice defaults and
  note overrides are merged, so note-level partial overrides compose cleanly

## Parameter Resolution

Synth params are resolved in this order:

1. preset values, if `preset` is set
2. voice-level `synth_defaults`
3. note-level `synth` overrides

Explicit params always override preset values.

If `engine` is omitted, it defaults to `additive`.

## Shared Envelope Parameters

These are consumed by the score renderer after the engine returns a raw mono signal:

- `amp_db: float`
- `env.attack_ms: float`
- `env.decay_ms: float`
- `env.sustain_ratio: float`
- `env.release_ms: float`

`amp_db` is the recommended authoring control for note loudness; it is converted
to the renderer's linear `amp` internally. Linear `amp` is still supported, but
it is less intuitive for balancing voices.

These control the ADSR envelope applied in [code_musics/synth.py](/home/jan/workspace/code-musics/code_musics/synth.py).

Canonical meanings:

- `attack_ms`, `decay_ms`, and `release_ms` are durations in milliseconds
- `sustain_ratio` is a normalized sustain level in the range `0..1`

The composition helper layer may also attach note-level `attack_scale` and
`release_scale` values inside `NoteEvent.synth`; the score renderer applies them
after merging voice defaults and note overrides.

Legacy compatibility:

- `attack`, `decay`, and `release` are still accepted in seconds
- `sustain_level` is still accepted as the legacy name for `sustain_ratio`
- top-level flat aliases like `attack_ms=30.0` are accepted if you prefer a flat style

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
    synth_defaults={
        "engine": "filtered_stack",
        "preset": "round_bass",
    },
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
- `kick_tom`
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
- `kick_tom`: `808_hiphop`, `808_house`, `808_tape`, `909_techno`, `909_house`, `909_crunch`, `distorted_hardkick`, `zap_kick`, `round_tom`, `floor_tom`, `electro_tom`, `ring_tom`
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

### `eq`

Implementation: [code_musics/synth.py](/home/jan/workspace/code-musics/code_musics/synth.py)

Native minimum-phase EQ for routine voice and bus tone shaping.

Parameters:

- `bands: list[dict[str, Any]]`
  Ordered EQ band definitions. Bands are applied in list order.

Supported band kinds:

- `highpass`
  Parameters: `cutoff_hz`, `slope_db_per_oct`
- `lowpass`
  Parameters: `cutoff_hz`, `slope_db_per_oct`
- `bell`
  Parameters: `freq_hz`, `gain_db`, `q`
- `low_shelf`
  Parameters: `freq_hz`, `gain_db`, `q` (default `0.707`)
- `high_shelf`
  Parameters: `freq_hz`, `gain_db`, `q` (default `0.707`)

Notes:

- this EQ is causal / minimum-phase and uses native IIR filters, not linear-phase processing
- `highpass` and `lowpass` support only `12` or `24 dB/oct`
- `q` must be positive
- all frequencies must be above `0` and below Nyquist
- v1 intentionally has no presets, tilt mode, or makeup gain

Example:

```python
score = Score(
    f0=110.0,
    master_effects=[
        EffectSpec(
            "eq",
            {
                "bands": [
                    {"kind": "highpass", "cutoff_hz": 35.0, "slope_db_per_oct": 24},
                    {"kind": "low_shelf", "freq_hz": 140.0, "gain_db": 1.5},
                    {"kind": "bell", "freq_hz": 2200.0, "gain_db": -2.0, "q": 1.1},
                    {"kind": "high_shelf", "freq_hz": 8000.0, "gain_db": -1.0},
                ]
            },
        )
    ],
)
```

### `compressor`

Implementation: [code_musics/synth.py](/home/jan/workspace/code-musics/code_musics/synth.py)

Native stereo-linked compressor for glue, stem control, and master-bus shaping.
It supports both feedforward and feedback topologies and can EQ the
detector/listener path without EQing the audible signal itself. On score voices,
it also supports external sidechaining from another named voice.

Render analysis records native compressor metering in the analysis manifest.
Current effect-analysis metrics include:

- `avg_gain_reduction_db`
- `max_gain_reduction_db`
- `p95_gain_reduction_db`
- `active_gain_reduction_fraction`
- `avg_gain_reduction_when_active_db`
- `below_1db_fraction`
- `longest_run_above_1db_seconds`

Those diagnostics are used to warn when a compressor appears mostly inactive or
unusually aggressive.

Parameters:

- `preset: str`
  Supported presets: `kick_glue`, `kick_punch`, `tom_control`
- `threshold_db: float`
  Compression threshold in dBFS. Default `-20.0`.
- `ratio: float`
  Compression ratio. Must be at least `1.0`. Default `3.0`.
- `attack_ms: float`
  Attack time in milliseconds. Default `15.0`. Intended musical range includes
  roughly `0.5` to `50`.
- `release_ms: float`
  Primary release time in milliseconds. Default `180.0`. Intended musical range
  includes roughly `50` to `1000+`.
- `release_tail_ms: float | None`
  Optional slower second-stage release tail in milliseconds. Leave as `None` for
  a single-stage release using only `release_ms`.
- `knee_db: float`
  Soft-knee width in dB. Default `6.0`.
- `makeup_gain_db: float`
  Linear post-compression makeup gain in dB. Default `0.0`.
- `mix: float`
  Dry/wet blend from `0` to `1`. Default `1.0`.
- `topology: str`
  `feedforward` or `feedback`. Default `feedforward`.
- `detector_mode: str`
  `peak` or `rms`. Default `rms`.
- `detector_bands: list[dict[str, Any]] | None`
  Optional EQ bands applied only to the detector path. Band syntax is identical
  to the native `eq` effect.
- `sidechain_source: str | None`
  Optional external detector source for score voice effects. Names another voice
  and uses that voice's post-everything output as the detector input.
- `lookahead_ms: float`
  Detector lookahead in milliseconds. Default `0.0`. Lets the compressor react a
  little early, which is especially useful for kick/bass ducking and peak control.

Notes:

- channels are stereo-linked so left/right image does not wander under compression
- `kick_punch` is the most obvious starting point for finished kick voices, while
  `kick_glue` is gentler and `tom_control` is the safer tom preset
- all three kick/tom presets are calibrated for use in rhythmic electronic contexts
  (120–150 BPM): thresholds are set so the compressor naturally exits between hits
  as the kick tail decays, delivering 5–8 dB GR at the peak rather than continuous
  limiting — expect roughly 3 dB for `kick_glue`, 5–7 dB for `kick_punch`
- detector EQ is the native control-path tone-shaping surface; use it for things
  like bass-insensitive bus compression via a detector highpass
- `sidechain_source` is currently a score-voice routing feature rather than a
  general-purpose raw-audio API on every effect-chain call site
- `release_ms` is the main “easy” release knob; add `release_tail_ms` only when
  you want a faster bounce-back followed by a slower, gentler tail
- the gain release is two-stage when `release_tail_ms` is set, which tends to feel
  more musical than a single fixed release all the way back to zero reduction
- `lookahead_ms` uses offline lookahead so the rendered output stays sample-aligned
  while the detector can react ahead of fast transients

Example:

```python
score.add_voice("kick", normalize_peak_db=-6.0)
score.add_voice(
    "bass",
    effects=[
        EffectSpec(
            "compressor",
            {
                "threshold_db": -28.0,
                "ratio": 4.0,
                "attack_ms": 1.0,
                "release_ms": 120.0,
                "lookahead_ms": 5.0,
                "sidechain_source": "kick",
                "detector_mode": "peak",
                "detector_bands": [
                    {"kind": "lowpass", "cutoff_hz": 180.0, "slope_db_per_oct": 12}
                ],
            },
        )
    ],
)
```

### `plugin`

Implementation: [code_musics/synth.py](/home/jan/workspace/code-musics/code_musics/synth.py)

Generic external-plugin effect. Use this when you want to host a plugin directly
instead of adding a dedicated wrapper kind.

Render analysis cannot usually read a hosted plugin's internal meter state
directly, so plugin stages currently use before/after diagnostics in the
analysis manifest. Current metrics include peak and true-peak deltas,
crest-factor change, clipping and near-full-scale occupancy deltas, and
spectral-centroid movement. That is the current agent-facing legibility layer
for plugin-driven compression, distortion, warmth, and related dynamic/color
stages.

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
    synth_defaults={
        "engine": "filtered_stack",
        "preset": "warm_pad",
        "env": {"attack_ms": 400.0, "release_ms": 1400.0},
    },
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

High-fidelity native saturation for tube/iron/preamp-style warmth rather than
obvious guitar-style distortion. The default engine is a modern two-stage
saturator with DC-safe asymmetry, bounded loudness compensation, and optional
clean low/high-band reintegration so warmth does not have to come from blanket
top-end darkening. The previous one-stage soft-clip path remains available via
`algorithm="legacy"`.

Render analysis records saturation-stage diagnostics in the analysis manifest.
Metrics include the shared before/after density/clipping proxies plus:

- `algorithm`: `"modern"` or `"legacy"`.
- `mode`: qualitative voicing for the modern engine.
- `shaper_hot_fraction`: fraction of samples where the internal drive signal exceeds
  ±1.0 (clearly nonlinear region).
- `dc_offset`: final mono-reference DC offset after processing.
- `thd_pct`: characteristic THD% computed from a 440 Hz reference sine at −12 dBFS
  processed through the same shaper curve. Reflects the shaper character, not the
  actual audio content.
- `thd_character`: qualitative label derived from `thd_pct` —
  `"clean"` (< 0.5%), `"subtle_warmth"` (0.5–2%), `"warmth"` (2–5%),
  `"saturation"` (5–15%), `"distortion"` (15–40%), `"fuzz"` (> 40%).
- `compensation_mode_used`: final makeup mode after resolving `auto` or honoring an
  explicit strict mode.
- `compensation_gain_db`: bounded output gain applied after measurement matching.

Parameters:

- `preset: str`
  Supported presets: `tube_warm`, `iron_soft`, `neve_gentle`, `kick_weight`,
  `kick_crunch`, `tom_thicken`
- `algorithm: str`
  `modern` (default) or `legacy`.
- `mode: str`
  Modern-engine voicing: `tube`, `triode`, or `iron`.
- `drive: float`
  Amount of nonlinearity. Keep this conservative for bus sweetening.
- `mix: float`
  Dry/wet blend from `0` to `1`.
- `tone: float`
  Broad tonal push before/through the nonlinear stages. Positive values add a
  little more bite and presence; negative values lean softer/thicker.
- `fidelity: float`
  `0`–`1` control for how much low/high clean-band preservation is allowed back
  into the saturated result. Higher values keep the effect more open and hi-fi.
- `bias: float`
  Advanced asymmetry trim. Still supported, but the modern engine is voiced
  primarily through `mode`, `tone`, and `fidelity`.
- `even_harmonics: float`
  Advanced blend between symmetric and asymmetric curves.
- `oversample_factor: int`
  Oversampling factor used around the nonlinear stage.
- `highpass_hz: float`
  Removes excessive sub/DC before saturation.
- `tone_tilt: float`
  Legacy-compatible extra tilt into the nonlinear path.
- `output_lowpass_hz: float`
  Optional extra post-saturation smoothing. Default modern behavior does not rely
  on a fixed lowpass to create warmth.
- `preserve_lows_hz: float`
  Low-band crossover below which some clean signal may be reintegrated.
- `preserve_highs_hz: float`
  High-band crossover above which some clean signal may be reintegrated.
- `compensation_mode: str`
  `none`, `auto`, `rms`, or `lufs`. Default is `auto`: prefer LUFS for sustained
  material and switch to RMS for short/sparse/transient material. If you explicitly
  request `lufs` or `rms`, that mode is used strictly with no hidden fallback.
- `output_trim_db: float`
  Final manual trim after automatic compensation.

Notes:

- default tuning is intentionally subtle enough for “always on” use
- the modern engine is now the default for existing `EffectSpec("saturation", ...)`
  call sites
- `tube_warm` is the safest bus-sweetening default
- `iron_soft` is the most transformer-like subtle thickener
- `kick_weight` is the recommended saturation companion for kick presets
- use `algorithm="legacy"` only when you explicitly want the older one-stage
  soft-clip behavior

Example:

```python
score = Score(
    f0=110.0,
    master_effects=[
        EffectSpec(
            "saturation",
            {"preset": "tube_warm", "mix": 0.24, "preserve_highs_hz": 7000.0},
        ),
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

- `partials: list[dict[str, float]]`
  Optional explicit spectral profile. Each entry must define positive `ratio`
  and non-negative `amp`. When present, this becomes the authoritative
  spectrum for the note and the old harmonic-stack generator params are ignored.
- `attack_partials: list[dict[str, float]]`
  Optional onset spectrum in the same format as `partials`. The engine builds
  the union of `attack_partials` and `partials`, treats missing entries as
  silent, and morphs from the attack amplitudes into the sustain amplitudes.
- `spectral_morph_time: float`
  Seconds spent interpolating from `attack_partials` into `partials`.
- `partial_decay_tilt: float`
  Makes higher-ratio partials settle faster over note time while lower partials
  remain more stable.
- `upper_partial_drift_cents: float`
  Gentle deterministic drift applied only to upper partials.
- `upper_partial_drift_min_ratio: float`
  Ratios below this stay stable; ratios above it receive progressively more of
  `upper_partial_drift_cents`.

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
- If `partials` is omitted, the engine still uses the legacy harmonic ladder
  generated from `n_harmonics`, `harmonic_rolloff`, `brightness_tilt`, and
  `odd_even_balance`.
- `odd_even_balance` is clamped internally to avoid zeroing the spectrum too aggressively.
- `attack_partials` only does anything when paired with `spectral_morph_time > 0`.
- Explicit spectral ratios are relative to the resolved note frequency, not to
  `Score.f0`.

Helper builders:

- `code_musics.spectra.ratio_spectrum(...)`
  Build an explicit `partials` list from ratio and amplitude inputs.
- `code_musics.spectra.harmonic_spectrum(...)`
  Build a harmonic-series `partials` list matching the additive engine's
  legacy weightings.
- `code_musics.spectra.stretched_spectrum(...)`
  Build a stretched or compressed overtone family for non-harmonic ladders.

Recommended additive presets:

- `soft_pad`
- `drone`
- `bright_pluck`
- `organ`
- `ji_fusion_pad`
- `septimal_reed`
- `eleven_limit_glass`
- `utonal_drone`

Example:

```python
from code_musics.spectra import ratio_spectrum

score.add_voice(
    "pad",
    synth_defaults={
        "engine": "additive",
        "env": {"attack_ms": 400.0, "release_ms": 1200.0},
        "params": {
            "partials": ratio_spectrum(
                [1.0, 5 / 4, 3 / 2, 7 / 4],
                [1.0, 0.4, 0.28, 0.16],
            ),
            "attack_partials": ratio_spectrum(
                [1.0, 5 / 4, 3 / 2, 7 / 4, 11 / 8],
                [1.0, 0.48, 0.34, 0.2, 0.12],
            ),
            "spectral_morph_time": 0.18,
            "partial_decay_tilt": 0.25,
            "unison_voices": 3,
            "detune_cents": 5.0,
        },
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

Recommended authoring aliases:

- `params.index_decay_ms`
- `params.index_sustain_ratio`

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
        "env": {"attack_ms": 20.0, "release_ms": 350.0},
        "params": {
            "carrier_ratio": 1.0,
            "mod_ratio": 7 / 4,
            "mod_index": 2.8,
            "index_decay_ms": 100.0,
            "index_sustain_ratio": 0.45,
        },
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
  thicken and soften the response. The response is intentionally eased so that
  very small settings stay subtle instead of jumping straight into obvious
  distortion. Default `0.0`.

Recommended authoring aliases:

- `params.resonance_ratio`
- `params.filter_env_depth_ratio`
- `params.filter_env_decay_ms`
- `params.filter_drive_ratio`

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
        "env": {"attack_ms": 10.0, "release_ms": 300.0},
        "params": {
            "waveform": "square",
            "n_harmonics": 12,
            "cutoff_hz": 550.0,
            "keytrack": 0.15,
            "resonance_ratio": 0.15,
            "filter_env_depth_ratio": 0.8,
            "filter_env_decay_ms": 180.0,
        },
    },
)
```

## `noise_perc`

Implementation: [code_musics/engines/noise_perc.py](/home/jan/workspace/code-musics/code_musics/engines/noise_perc.py)

Parameters:

- `noise_mix: float`
  Blend between pitched tone and noise layer. `0` is fully pitched; `1` is fully noise-driven.
- `pitch_decay: float`
  Decay time in seconds for the tone component's attack transient. Also used as the
  default for `noise_decay` when `noise_decay` is not specified (backward-compatible).
- `noise_decay: float`
  Decay time in seconds for the noise body envelope, independent of `pitch_decay`.
  Set this longer than `pitch_decay` to get a full noise body after the pitched
  transient dies — the key parameter for clap and snare body. Defaults to
  `pitch_decay` when omitted.
- `tone_decay: float`
  Decay time in seconds for the pitched tone body.
- `bandpass_ratio: float`
  Ratio applied to the resolved note frequency to choose the center of the noise
  shaping band. Bandpass width is `max(80, center × 0.75)` Hz.
- `click_amount: float`
  Level of the short transient click layer.

Recommended authoring aliases:

- `params.noise_mix_ratio`
- `params.pitch_decay_ms`
- `params.noise_decay_ms`
- `params.tone_decay_ms`

Validation:

- `0 <= noise_mix <= 1`
- `pitch_decay > 0`
- `noise_decay > 0`
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
        "params": {
            "noise_mix_ratio": 0.7,
            "pitch_decay_ms": 40.0,
            "tone_decay_ms": 120.0,
            "bandpass_ratio": 1.5,
            "click_amount": 0.1,
        },
    },
)
```

## `kick_tom`

Implementation: [code_musics/engines/kick_tom.py](/home/jan/workspace/code-musics/code_musics/engines/kick_tom.py)

Compact electro-drum voice aimed at 808/909-style kicks, toms, and adjacent
experimental low-end percussion. The engine owns its internal pitch sweep, so
standard kick behavior does not require score-level `pitch_motion`.

Parameters:

- `body_decay_ms: float`
  Main body decay time in milliseconds.
- `pitch_sweep_amount_ratio: float`
  Starting pitch multiplier for the internal sweep. Values above `1` start
  higher and fall toward the resolved note frequency.
- `pitch_sweep_decay_ms: float`
  Decay time for the internal sweep.
- `body_wave: str`
  `sine`, `triangle`, or `sine_clip`.
- `body_tone_ratio: float`
  Blend toward a brighter second-harmonic body component.
- `body_punch_ratio: float`
  Extra short-lived body emphasis for punchier attacks.
- `overtone_amount: float`
  Level of the ring/overtone component.
- `overtone_ratio: float`
  Frequency ratio of the overtone component relative to the swept body pitch.
- `overtone_decay_ms: float`
  Decay time for the overtone layer.
- `click_amount: float`
  Level of the short filtered transient click.
- `click_decay_ms: float`
  Click decay time in milliseconds.
- `click_tone_hz: float`
  Center frequency for the click-brightness shaping.
- `noise_amount: float`
  Level of the filtered noise burst.
- `noise_decay_ms: float`
  Decay time for the noise burst.
- `noise_bandpass_hz: float`
  Center frequency of the noise burst shaping band.
- `drive_ratio: float`
  Internal nonlinear shaping amount from `0` to `1`.
- `post_lowpass_hz: float`
  Final smoothing lowpass after the internal drive stage.

Recommended authoring aliases:

- `params.body_decay_ms`
- `params.sweep_amount_ratio`
- `params.sweep_decay_ms`
- `params.body_tone`
- `params.body_punch`
- `params.drive`

Notes:

- use `freq` as the final drum pitch; the internal sweep rises above that pitch
  and then decays back toward it
- kick presets bias toward more sweep and sub weight
- tom presets bias toward more overtone/ring and less sub dominance
- this engine is intended to be paired with native effect presets such as
  `EffectSpec("compressor", {"preset": "kick_punch"})` and
  `EffectSpec("saturation", {"preset": "kick_weight"})`

Example:

```python
score.add_voice(
    "kick",
    synth_defaults={"engine": "kick_tom", "preset": "909_techno"},
    effects=[
        EffectSpec("compressor", {"preset": "kick_punch"}),
        EffectSpec("saturation", {"preset": "kick_weight"}),
    ],
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
  Resonance on a 0–1 internal scale. Maps to Q via `Q = 0.707 + 11.293 × resonance`
  (so `resonance=0` → Q=0.707 Butterworth flat, `resonance=0.1` → Q≈1.84). Prefer
  `resonance_q` when an explicit Q is known.
- `resonance_q: float`
  Filter Q as a direct value (≥ 0.5). Takes precedence over `resonance` when both
  are provided. Q=0.707 is Butterworth flat (no resonance peak). Q=1 is a gentle
  peak; Q=3–4 is clearly resonant; Q=8+ approaches self-oscillation. For bass
  voices, keep Q near 0.707 to avoid filter suckout; occasional bumps to Q=1.5–2.5
  add flavor without muddying the fundamental.
- `filter_drive: float`
  Analog-inspired drive amount inside the filter path. `0` is clean; higher values
  thicken and soften the response. The response is intentionally eased so that
  very small settings stay subtle instead of jumping straight into obvious
  distortion. Default `0.0`.
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

Artifact-risk guidance:

- `cutoff_hz`
  Safe: roughly low-mid hundreds up through the low-to-mid 2000s for most
  mellow voices.
  Risky: sustained settings above ~3200 Hz when the patch is already hot or
  resonance-heavy.
- `filter_env_amount`
  Safe: subtle to moderate motion.
  Risky: values near or above ~0.8 when combined with broad cutoff movement.
- `filter_drive`
  Safe: restrained saturation, often below ~0.08 for normal authoring.
  Risky: stronger drive layered with higher resonance and aggressive filter
  envelopes.
- `resonance`
  Safe: low to moderate emphasis.
  Risky: higher values when cutoff automation, drive, and bright source waves
  are all active at once.

Common interaction trap:

- `high cutoff sweep + strong filter_env_amount + resonance + velocity-driven
  cutoff` is the easiest way to author harsh, divebomb-like, or unstable
  behavior by accident. If you want that sound, push it deliberately and expect
  artifact-risk warnings in the render analysis.

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
        "env": {"attack_ms": 10.0, "release_ms": 300.0},
        "params": {
            "waveform": "saw",
            "cutoff_hz": 2500.0,
            "filter_mode": "lowpass",
            "keytrack": 0.08,
            "resonance_ratio": 0.12,
            "filter_drive_ratio": 0.18,
            "filter_env_depth_ratio": 0.6,
            "filter_env_decay_ms": 500.0,
        },
    },
)
```
