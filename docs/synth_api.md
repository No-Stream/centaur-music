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

- [code_musics/score.py](code_musics/score.py) merges voice-level and note-level synth params, resolves presets, and dispatches to the requested engine
- [code_musics/engines/registry.py](code_musics/engines/registry.py) defines the engine registry and preset map
- [code_musics/engines/additive.py](code_musics/engines/additive.py)
- [code_musics/engines/fm.py](code_musics/engines/fm.py)
- [code_musics/engines/filtered_stack.py](code_musics/engines/filtered_stack.py)
- [code_musics/engines/harpsichord.py](code_musics/engines/harpsichord.py)
- [code_musics/engines/clap.py](code_musics/engines/clap.py)
- [code_musics/engines/kick_tom.py](code_musics/engines/kick_tom.py)
- [code_musics/engines/metallic_perc.py](code_musics/engines/metallic_perc.py)
- [code_musics/engines/noise_perc.py](code_musics/engines/noise_perc.py)
- [code_musics/engines/organ.py](code_musics/engines/organ.py)
- [code_musics/engines/piano.py](code_musics/engines/piano.py)
- [code_musics/engines/piano_additive.py](code_musics/engines/piano_additive.py)
- [code_musics/engines/snare.py](code_musics/engines/snare.py)
- [code_musics/engines/surge_xt.py](code_musics/engines/surge_xt.py)
- [code_musics/engines/vital.py](code_musics/engines/vital.py)

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

These control the ADSR envelope applied in [code_musics/synth.py](code_musics/synth.py).

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
[code_musics/pitch_motion.py](code_musics/pitch_motion.py)
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
- control-surface lanes for score-time mixing and spatial motion:
  `pan`, `mix_db`, `pre_fx_gain_db`, `send_db`, `return_db`, `mix`, `wet`,
  and `wet_level`

Attachment points:

- `Voice.automation` for score-time lanes
- `NoteEvent.automation` for note-local lanes
- `VoiceSend.automation` for send level rides
- `SendBusSpec.automation` for aux return level and pan rides
- `EffectSpec.automation` for score-time wet/mix rides on insert, send, or
  master effects

Important v1 limits:

- `pitch_ratio` automation is per-sample
- synth-param automation is sampled at note start
- control-surface automation is score-time and applied after note rendering at
  the relevant voice/effect/send stage
- effect automation currently targets wetness controls (`mix`, `wet`,
  `wet_level`), not arbitrary effect internals
- `pitch_motion` and `pitch_ratio` automation cannot be combined on the same note

## Engine Selection

Set the engine with:

```python
synth_defaults = {"engine": "fm"}
```

Available engines:

- `additive`
- `clap`
- `fm`
- `filtered_stack`
- `harpsichord`
- `kick_tom`
- `metallic_perc`
- `noise_perc`
- `organ`
- `piano`
- `piano_additive`
- `polyblep`
- `snare`
- `surge_xt`
- `vital`

## Presets

Set a preset with:

```python
synth_defaults = {"engine": "filtered_stack", "preset": "warm_pad"}
```

Available presets:

- `additive`: `soft_pad`, `drone`, `bright_pluck`, `organ`, `ji_fusion_pad`, `septimal_reed`, `eleven_limit_glass`, `utonal_drone`, `plucked_ji`, `breathy_flute`, `ancient_bell`, `whispered_chord`, `struck_membrane`, `singing_bowl`, `marimba_bar`, `convolved_bell`, `thick_drone`, `fractal_fifth`, `fractal_septimal`, `vowel_a_pad`, `singing_glass`, `gravity_cloud`, `drifting_to_just`, `living_drone`, `candle_light`
- `fm`: `bell`, `glass_lead`, `metal_bass`, `dx_piano`, `lately_bass`, `fm_clav`, `fm_mallet`, `chorused_ep`
- `filtered_stack`: `warm_pad`, `reed_lead`, `round_bass`, `saw_pad`, `string_pad`
- `kick_tom`: `808_hiphop`, `808_house`, `808_tape`, `909_techno`, `909_house`, `909_crunch`, `distorted_hardkick`, `zap_kick`, `round_tom`, `floor_tom`, `electro_tom`, `ring_tom`, `gated_808`, `pitch_dive`, `filtered_kick`, `fm_body_kick`, `foldback_kick`
- `metallic_perc`: `closed_hat`, `open_hat`, `pedal_hat`, `ride_bell`, `ride_bow`, `crash`, `cowbell`, `clave`, `swept_hat`, `decaying_bell`
- `noise_perc`: `kickish`, `snareish`, `tick`, `chh`, `clap`, `shaped_hit`
- `snare`: `909_tight`, `909_fat`, `808_snare`, `rim_shot`, `brush`, `cross_stick`, `gated_snare`
- `clap`: `909_clap`, `tight_clap`, `big_clap`, `finger_snap`, `hand_clap`, `gated_clap`
- `organ`: `warm`, `full`, `jazz`, `gospel`, `cathedral`, `baroque`, `septimal`, `glass_organ`
- `harpsichord`: `baroque`, `concert`, `bright`, `warm`, `ethereal`, `glass`, `septimal`
- `piano`: `grand`, `bright`, `warm`, `felt`, `honky_tonk`, `tack`, `glass`, `septimal`
- `piano_additive`: `grand`, `bright`, `warm`, `felt`, `honky_tonk`, `tack`, `glass`, `septimal`
- `polyblep`: `warm_lead`, `synth_pluck`, `analog_brass`, `square_lead`, `hoover`, `moog_bass`, `sync_lead`, `acid_bass`, `sub_bass`, `resonant_sweep`, `soft_square_pad`, `juno_pad`, `analog_bass`, `prophet_lead`, `glass_pad`

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

Implementation: [code_musics/synth.py](code_musics/synth.py)

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

Implementation: [code_musics/synth.py](code_musics/synth.py)

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
  Supported presets: `kick_glue`, `kick_punch`, `tom_control`, `snare_punch`,
  `snare_body`, `hat_control`, `master_glue`
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
- `snare_punch` is a fast-attack transient controller (4 ms attack, 120 ms release,
  3:1 ratio, peak detector with SC HP at 100 Hz) — lets the initial crack through
  then clamps; ~5 dB GR at peak for a snare at −6 dBFS
- `snare_body` uses a slower attack (18 ms) with RMS detection and feedback topology
  to let the transient fully through while smoothing the body/tail sustain
- `hat_control` is a very fast attack (2 ms) peak limiter for taming hi-hat spikes;
  short 60 ms release avoids pumping on rapid 16th-note patterns
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
- `master_glue` is a gentle vari-mu inspired bus glue preset for the default master
  chain. Feedback topology, RMS detector, 2:1 ratio, 30 ms attack, 200 ms release,
  6 dB soft knee, HP sidechain at 80 Hz. Calibrated for ~-24 LUFS input (the auto
  gain staging target). Average GR ~2–3 dB on typical material.

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

Implementation: [code_musics/synth.py](code_musics/synth.py)

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

Implementation: [code_musics/synth.py](code_musics/synth.py)

Warm stereo chorus inspired by classic analog/BBD units and intended for subtle
depth rather than obvious wobble.

Parameters:

- `preset: str`
  Supported presets: `juno_subtle`, `juno_wide`, `ensemble_soft`
- `mix: float`
  Dry/wet blend from `0` to `1`. Default `0.28`. Typical musical use is
  around `0.2` to `0.33`.
- `rate_hz: float`
  Base LFO rate in Hertz. Default `0.32`.
- `depth_ms: float`
  Modulation depth in milliseconds. Default `2.4`. Note: the parameter name
  is `depth_ms`, not `depth`.
- `center_delay_ms: float`
  Base chorus delay time in milliseconds. Default `13.5`.
- `stereo_phase_deg: float`
  Phase offset between left and right modulation. Default `115.0`.
- `feedback: float`
  Very light recirculation on the wet path. Default `0.04`.
- `wet_lowpass_hz: float`
  Darkens the wet path to keep the effect smooth. Default `6000.0`.
- `wet_highpass_hz: float`
  Removes low-end smear from the wet path. Default `160.0`.
- `drift_amount: float`
  Adds a slower secondary modulation for analog drift. Default `0.12`.
- `wet_saturation: float`
  Adds slight nonlinearity on the wet path so the chorus feels less sterile.
  Default `0.06`.

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

### `stereo_width`

Implementation: [code_musics/synth.py](code_musics/synth.py)

Mid/side stereo width control for narrowing or exaggerating the stereo image.
Mono signals pass through unchanged.

Parameters:

- `width: float`
  Stereo width multiplier. Default `1.0`. `0.0` collapses to mono, `1.0` leaves
  the signal unchanged, `2.0` exaggerates the stereo field. Values above `1.0`
  boost the side channel relative to mid, so very high settings can thin out
  centered content.

Notes:

- only affects stereo signals; mono input passes through unmodified
- useful as a mix tool on voices, buses, or the master chain
- narrowing (`width < 1.0`) can help tighten bass or center a vocal-like voice;
  widening (`width > 1.0`) can push pads and ambient textures outward

Example:

```python
# Narrow a bass voice to mono center
score.add_voice(
    "bass",
    effects=[EffectSpec("stereo_width", {"width": 0.0})],
)

# Widen a pad's stereo image
score.add_voice(
    "pad",
    effects=[EffectSpec("stereo_width", {"width": 1.5})],
)
```

### `saturation`

Implementation: [code_musics/synth.py](code_musics/synth.py)

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
  `kick_crunch`, `tom_thicken`, `snare_bite`
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
- `snare_bite` is a triode-mode saturation preset (drive 1.8, mix 0.35) for adding
  grit and harmonic edge to snare voices with clean low-band preservation at 150 Hz
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

### `preamp`

Implementation: [code_musics/synth.py](code_musics/synth.py)

Flux-domain transformer saturation modeling real iron-core physics. Unlike
memoryless waveshaping, this operates in the magnetic flux domain where low
frequencies naturally saturate more than highs (Faraday's law: V = N·dΦ/dt).
The result is frequency-dependent harmonic generation with warm, analog
character and minimal intermodulation on complex material.

**Parameters:**

- `drive` (0.0–2.0+): How hard the transformer core is driven.
  0.25 = barely there, 0.5 = gentle warmth, 1.0 = rich, 1.5+ = crunchy.
- `mix` (0.0–1.0): Wet/dry blend. Default 0.30 for subtle bus color.
- `warmth` (0.0–1.0): Pre-emphasis bass shelf. Higher = more bass enrichment.
- `brightness` (-1.0–1.0): Post tilt EQ. 0 = neutral.
- `even_odd` (0.0–1.0): Even vs odd harmonic balance. 0 = odd-dominant,
  1.0 = even-dominant (transformer-like). Default 0.7.
- `flux_cutoff_hz`: Leaky integrator corner (default 12 Hz).
- `harmonic_injection` (0.0–1.0): Parallel Chebyshev harmonic injection amount.

**Presets:**

- `neve_warmth`: Subtle Neve-like transformer color (drive=0.5, mix=0.30).
  Default for master bus warmth.
- `iron_color`: Assertive transformer saturation (drive=0.6, mix=0.35).
- `tube_glow`: Tube-flavored warmth with more odd harmonics (drive=0.5, mix=0.30).
- `transformer_drive`: Driven transformer, approaching distortion (drive=1.2, mix=0.50).

Example:

```python
score = Score(
    f0=110.0,
    master_effects=[
        EffectSpec("preamp", {"preset": "neve_warmth"}),
        EffectSpec("reverb", {"room_size": 0.65, "damping": 0.45, "wet_level": 0.22}),
    ],
)
```

### `chow_tape`

Implementation: [code_musics/synth.py](code_musics/synth.py)

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

### `airwindows`

Implementation: [code_musics/synth.py](code_musics/synth.py)

Wrapper around Airwindows Consolidated, giving access to hundreds of Chris
Johnson's audio algorithms through a single plugin. Algorithm selection happens
via internal preset patching — the plugin's parameter names change dynamically
to match each algorithm.

Parameters:

- `algorithm: str`
  Algorithm name. Examples: `"Density"`, `"IronOxide5"`, `"ToTape6"`,
  `"Tube"`, `"Drive"`, `"Coils"`, `"Channel9"`.
- `input_level: float`
  Input trim in dB. Default `0.0`.
- `output_level: float`
  Output trim in dB. Default `0.0`.
- `**algo_params`
  Algorithm-specific parameters passed through as plugin attributes after the
  algorithm switch.

Presets:

- `density_glue` — Density algorithm, moderate settings for bus glue
- `iron_warmth` — IronOxide5 tape head saturation for warmth
- `tape_subtle` — ToTape6 subtle tape emulation
- `tube_warmth` — Tube algorithm, gentle tube color
- `coils_xformer` — Coils transformer saturation
- `channel_ssl` — Channel9 SSL console color
- `drive_gentle` — Drive algorithm at low settings

Verified algorithms and their parameters:

| Algorithm | Parameters |
|-----------|-----------|
| Density | density, highpass, out_level, dry_wet |
| Drive | drive, highpass, out_level, dry_wet |
| Tube | tube |
| Mojo | input |
| Distortion | input, mode, output, dry_wet |
| ToTape6 | input, soften, head_b, flutter, output, dry_wet |
| IronOxide5 | input_trim, tape_high, tape_low, flutter, noise, output_trim, inv_dry_wet |
| Tape | slam, bump |
| Coils | saturat, core_dc, dry_wet |
| Coils2 | saturate, cheapness, dry_wet |
| Channel9 | console_type, drive, output |
| Channel8 | console_type, drive, output |
| Mackity | in_trim, out_pad |
| MackEQ | trim, hi, lo, gain, dry_wet |
| Precious | hardns, persnlty, drive, output |
| UnBox | input, unbox, output |
| PurestSquish | squish, bassblm, output, dry_wet |
| ADClip7 | boost, soften, enhance, mode |
| Capacitor2 | lowpass, highpass, nonlin, dry_wet |
| Weight | freq, weight |
| Point | input_trim, point, reaction_speed |

Notes:

- expects `~/.vst3/Airwindows Consolidated.vst3` to be installed
- algorithm switching works by patching the plugin's VST3 preset XML
- parameter names are dynamic and change per algorithm
- the plugin instance is shared (cached), so algorithm switching happens on each call

Example:

```python
# Direct parameters
EffectSpec("airwindows", {"algorithm": "Density", "density": 1.3, "dry_wet": 0.7})

# Named preset
EffectSpec("airwindows", {"preset": "iron_warmth"})
```

### `byod`

Implementation: [code_musics/synth.py](code_musics/synth.py)

Wrapper around BYOD (Build Your Own Distortion), a modular distortion/overdrive
pedal from ChowDSP with 40 built-in presets emulating classic pedals and amp
tones.

Parameters:

- `program: str`
  Preset program name. Default `"Tube Screamer"`.
- `in_gain: float`
  Input gain in dB. Default `0.0`.
- `out_gain: float`
  Output gain in dB. Default `0.0`.
- `dry_wet: float`
  Dry/wet blend from `0` to `100` (percent). Default `100.0`.
- `mode: str`
  Processing mode — `"Stereo"` or `"Mono"`. Default `"Stereo"`.
- `**program_params`
  Additional program-specific parameters set as plugin attributes after the
  program switch. Parameter names change dynamically per program.

Presets:

- `tube_screamer` — Tube Screamer emulation
- `centaur` — Klon Centaur emulation
- `american` — American Sound (Fender-like amp tone)
- `zen_drive` — ZenDrive overdrive
- `king_of_tone` — King Of Tone overdrive

Available programs: Default, Bass Face, Instant Metal, Modern Hi-Gain,
Chopped Flange, Laser Cave, Mixed In Modulation, Seasick Phase, American Sound,
Big Muff (and variants), Centaur, Hot Cakes, Hot Fuzz, King Of Tone,
MXR Distortion, OctaVerb, RAT, Tube Screamer, Wah Pedal, ZenDrive,
plus artist-inspired presets (Clapton, Hendrix, Neil Young, Nirvana, etc.)
and utility presets (Gainful Clipper, Violet Mist).

Notes:

- expects `~/.vst3/BYOD.vst3` to be installed
- program selection exposes different parameters per preset
- the internal processing chain is configured by the program, not individually controllable

Example:

```python
EffectSpec("byod", {"program": "American Sound", "dry_wet": 80.0})
EffectSpec("byod", {"preset": "tube_screamer"})
```

### `chow_centaur`

Implementation: [code_musics/synth.py](code_musics/synth.py)

Wrapper around ChowCentaur, a neural-network-modeled Klon Centaur overdrive.
At low gain settings this works well as a subtle warmth/color tool.

Parameters:

- `gain: float`
  Drive amount from `0` to `1`. Default `0.3`.
- `treble: float`
  Treble/tone control from `0` to `1`. Default `0.5`.
- `level: float`
  Output level from `0` to `1`. Default `0.7`.
- `mode: str`
  Model mode — `"Neural"` (ML-modeled) or `"Traditional"` (DSP approximation).
  Default `"Neural"`.

Presets:

- `subtle_warmth` — very low gain, Neural mode, gentle color
- `light_edge` — moderate gain, slight grit
- `traditional_clean` — low gain, Traditional mode

Notes:

- expects `~/.vst3/ChowCentaur.vst3` to be installed
- the Neural mode uses a trained neural network for more accurate Klon emulation
- at gain below 0.3, this is primarily a color/warmth effect rather than distortion

Example:

```python
EffectSpec("chow_centaur", {"gain": 0.25, "treble": 0.5, "level": 0.7})
EffectSpec("chow_centaur", {"preset": "subtle_warmth"})
```

### `tal_chorus_lx`

Implementation: [code_musics/synth.py](code_musics/synth.py)

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

Implementation: [code_musics/synth.py](code_musics/synth.py)

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

### `tal_reverb2`

Implementation: [code_musics/synth.py](code_musics/synth.py)

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

### `dragonfly`

Implementation: [code_musics/synth.py](code_musics/synth.py)

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

## Shared Drum DSP Infrastructure

All drum engines share common DSP primitives:

### Multi-point envelopes (`_envelopes.py`)

Any drum engine parameter ending in `_envelope` accepts a list of envelope
point dicts. Each point has:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `time` | float | — | Position in [0, 1] normalized to note duration |
| `value` | float | — | Value at this point (meaning depends on context) |
| `curve` | str | `"linear"` | Interpolation to reach this point: `"linear"`, `"exponential"`, `"bezier"` |
| `cx` | float | 0.0 | Bezier shape control (0-1) |
| `cy` | float | 0.0 | Bezier shape control (0-1) |

The `curve` describes how we *arrive* at a point from the previous one. The
first point's curve is ignored.

### Waveshaper algorithms (`_waveshaper.py`)

Available for per-oscillator distortion via `body_distortion` (kick_tom):

| Algorithm | Character |
|-----------|-----------|
| `hard_clip` | Digital hard clipping |
| `tanh` | Smooth tube-like saturation |
| `atan` | Softer than tanh, more gradual |
| `exponential` | Asymptotic compression |
| `polynomial` | Cubic soft clip |
| `logarithmic` | Log compression, gentle |
| `foldback` | Wavefolder — rich harmonic generation |
| `linear_fold` | Linear wavefolder — folds the waveform back on itself via modular wrap. From Vital. |
| `sine_fold` | Sine wavefolder — soft musical folding via sine transfer function. |
| `half_wave_rect` | Octave-up character |
| `full_wave_rect` | Full rectification |

Drive follows the standard knob convention: 0-0.25 subtle, 0.33-0.66
moderate, 0.66-1.0 strong.

## `additive`

Implementation: [code_musics/engines/additive.py](code_musics/engines/additive.py)

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
- Per-partial `envelope`: optional key on partial dicts. List of
  `{"time": 0-1, "value": 0-1, "curve": "linear"|"exponential"|"bezier"}` points.
  Modulates the partial's amplitude over note duration. Composes multiplicatively
  with morph and decay_tilt.
- `noise_amount: float` (0-1, default 0.0)
  Global noise level mixed with each partial's sine. Creates breathy/organic character.
- `noise_bandwidth_hz: float` (default 60.0)
  Width of the noise band centered on each partial.
- Per-partial `noise` and `noise_bw`: override global noise params per partial.
- `spectral_gravity: float` (0-1, default 0.0)
  Strength of attraction toward nearby just intervals. Partials slowly drift toward
  the nearest JI attractor, weighted by ratio simplicity (Tenney height).
- `gravity_targets: list[float]` (default: [1, 9/8, 5/4, 4/3, 3/2, 5/3, 7/4, 2])
  JI ratios to attract toward. Octave equivalents are searched automatically.
- `gravity_rate: float` (default 1.0)
  Speed of gravitational drift.
- `spectral_flicker: float` (0-1, default 0.0)
  Per-partial random amplitude modulation depth for organic, living timbres.
- `flicker_rate_hz: float` (default 3.0)
  Smoothness of the flicker modulation (lowpass filter cutoff).
- `flicker_correlation: float` (0-1, default 0.3)
  How correlated adjacent partials' flickering is. 0=independent, 1=all in sync.

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

Helper builders (`code_musics.spectra`):

- `ratio_spectrum(ratios, amps)` — explicit partials from ratio/amplitude lists
- `harmonic_spectrum(n_partials, ...)` — harmonic-series partials matching legacy weightings
- `stretched_spectrum(n_partials, stretch_exponent, ...)` — stretched/compressed overtone families
- `membrane_spectrum(n_modes, damping)` — circular drumhead modes (Bessel zeros)
- `bar_spectrum(n_modes, material)` — vibrating bar modes (marimba/xylophone)
- `plate_spectrum(n_modes, aspect_ratio)` — rectangular plate modes
- `tube_spectrum(n_modes, open_ends)` — cylindrical tube modes (flute/clarinet/stopped)
- `bowl_spectrum(n_modes)` — singing bowl modes
- `spectral_convolve(spec_a, spec_b, ...)` — cross-product of two spectra (combination tones)
- `fractal_spectrum(seed, depth, level_rolloff, ...)` — self-similar spectra via iterated self-convolution
- `formant_shape(partials, f0, formants)` — apply vowel formant resonance peaks to partials
- `vowel_formants(name)` — named vowel formant data ("a", "e", "i", "o", "u")
- `formant_morph(partials, f0, vowel_sequence, morph_times)` — generate per-partial envelopes for time-varying vowel morphs

Formant usage notes:

- `formant_shape()` computes weights at a specific `f0`. When used in presets
  (e.g., `vowel_a_pad` in `registry.py`), the `f0` is baked at registration
  time. Playing those presets at different pitches shifts the formant character
  — this is physically accurate (real vocal formants are fixed-frequency
  resonances), but it means a preset built at 220 Hz will sound different at
  440 Hz. For pitch-accurate formant shaping, call `formant_shape()` at
  composition time with the actual note `f0`.
- `formant_morph()` generates per-partial envelopes normalized 0--1 across
  *note* duration, not piece duration. For piece-level vowel evolution, use
  different `formant_shape()` calls per section and pass distinct `partials`
  via per-note `synth={}` overrides or per-section voice reconfiguration.

Additive presets:

- `soft_pad`, `drone`, `bright_pluck`, `organ` — legacy harmonic presets
- `ji_fusion_pad`, `septimal_reed`, `eleven_limit_glass`, `utonal_drone` — JI spectral presets
- `plucked_ji` — JI partials with per-partial pluck envelopes (higher partials decay faster)
- `breathy_flute` — harmonic partials + breath noise bands
- `ancient_bell` — stretched inharmonic partials + metallic noise shimmer
- `whispered_chord` — JI chord + high noise for airy consonance
- `struck_membrane` — drumhead modes with fast decay tilt
- `singing_bowl` — bowl modes with slow attack and upper-partial drift
- `marimba_bar` — bar modes with fast decay
- `convolved_bell` — bar x bowl spectral hybrid via convolution
- `thick_drone` — JI chord convolved with harmonic series
- `fractal_fifth` — iterated-fifth self-similar spectrum
- `fractal_septimal` — septimal-seed fractal spectrum
- `vowel_a_pad` — harmonic spectrum shaped by /a/ formant resonances
- `singing_glass` — bowl spectrum + /o/ formant = eerie vocal resonance
- `gravity_cloud` — detuned partials coalescing toward JI intervals
- `drifting_to_just` — stretched spectrum gravitating toward harmonic ratios
- `living_drone` — subtle organic flicker on harmonic partials
- `candle_light` — strong independent per-partial amplitude wavering

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

## `clap`

Implementation: [code_musics/engines/clap.py](code_musics/engines/clap.py)

Multi-tap noise burst engine for clap and snap sounds. The characteristic clap
texture comes from several rapid micro-bursts (taps) before a longer noise body
tail. Each tap is a short bandpass-filtered noise burst with exponential decay.

Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_taps` | `int` | `4` | Number of micro-burst taps (1–8) |
| `tap_spacing_ms` | `float` | `5.0` | Time between consecutive taps in ms |
| `tap_decay_ms` | `float` | `3.0` | Decay time for each individual tap |
| `tap_crescendo` | `float` | `0.3` | Amplitude ramp across taps (0–1); later taps are louder |
| `body_decay_ms` | `float` | `60.0` | Decay time for the noise body tail after the last tap |
| `filter_center_ratio` | `float` | `1.0` | Bandpass center as a ratio of the note frequency |
| `filter_width_ratio` | `float` | `2.0` | Bandpass width relative to center frequency |
| `click_amount` | `float` | `0.08` | Level of an initial high-frequency click transient |
| `click_decay_ms` | `float` | `2.0` | Decay time for the click transient |

**Multi-point envelope params (optional):**

- `body_amp_envelope: list[dict]` — replaces body tail exponential decay
- `overall_amp_envelope: list[dict]` — applied to final mixed signal (useful for gated claps)

Notes:

- does not support pitch motion (`freq_trajectory`)
- deterministic for identical inputs (SHA-256 seeded RNG)
- use `freq` to set the center frequency of the bandpass shaping; typical clap
  frequencies are around 1000–3000 Hz
- `filter_width_ratio` controls spectral breadth; higher values give a wider,
  more full-spectrum sound

Presets:

| Preset | Character |
|--------|-----------|
| `909_clap` | Classic 909-style 4-tap clap |
| `tight_clap` | Short, snappy 3-tap clap |
| `big_clap` | Wider 6-tap clap with longer body |
| `finger_snap` | Quick 2-tap snap with narrow bandpass |
| `hand_clap` | Natural 5-tap hand clap with crescendo |

Example:

```python
score.add_voice(
    "clap",
    synth_defaults={"engine": "clap", "preset": "909_clap"},
    normalize_peak_db=-6.0,
)
```

## `fm`

Implementation: [code_musics/engines/fm.py](code_musics/engines/fm.py)

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

### Analog Character (fm)

Same analog character parameters as `polyblep`, minus the filter-specific ones
(`cutoff_drift`, `filter_even_harmonics`) since the FM engine has no internal
filter.

- `pitch_drift: float`
  1/f multi-rate pitch drift. Default `0.12` (~1.5 cents peak). `0` disables.
  Range `[0, 1]`.
- `analog_jitter: float`
  Per-note parameter variation: attack ±12%, amp ±0.3 dB, random phase. Default
  `1.0`. `0` disables. Range `[0, 2]`.
- `noise_floor: float`
  Pink noise floor level, envelope-following, bandlimited 100–8000 Hz. Default
  `0.001` (−60 dBFS). `0` disables. Range `[0, 0.01]`.
- `drift_rate_hz: float`
  Base frequency of the drift oscillator. Default `0.07`. Range `[0.01, 1.0]`.

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

Implementation: [code_musics/engines/filtered_stack.py](code_musics/engines/filtered_stack.py)

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

### Analog Character (filtered_stack)

Same analog character parameters as `polyblep`. These add per-note analog
imperfection and default to subtle, conservative values.

- `pitch_drift: float`
  1/f multi-rate pitch drift. Default `0.12` (~1.5 cents peak). `0` disables.
  Range `[0, 1]`.
- `analog_jitter: float`
  Per-note parameter variation: cutoff ±3%, attack ±12%, amp ±0.3 dB, random
  phase. Default `1.0`. `0` disables. Range `[0, 2]`.
- `noise_floor: float`
  Pink noise floor level, envelope-following, bandlimited 100–8000 Hz. Default
  `0.001` (−60 dBFS). `0` disables. Range `[0, 0.01]`.
- `drift_rate_hz: float`
  Base frequency of the drift oscillator. Default `0.07`. Range `[0.01, 1.0]`.
- `cutoff_drift: float`
  Ornstein-Uhlenbeck mean-reverting drift on filter cutoff. Default `0.5`. `0`
  gives a static cutoff. Range `[0, 2]`.
- `filter_even_harmonics: float`
  Asymmetric bias in the driven filter for even-harmonic warmth. Default `0.0`.
  Range `[0, 0.5]`.

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

## `harpsichord`

Implementation: [code_musics/engines/harpsichord.py](code_musics/engines/harpsichord.py)

Pluck-excitation + modal-resonator harpsichord engine. Captures the crisp,
immediate attack and bright decay of a plucked string while going beyond
physical constraints: velocity expression, per-note spectral morphing,
continuous register blending, and custom partial ratios for xenharmonic tuning.

The synthesis chain:

1. Shaped pluck impulse determines initial mode amplitudes
2. Modal resonator bank (decaying sinusoids) produces the string tone
3. Multiple registers (8', 4', lute) are blended with independent pluck
   character and decay
4. Per-note spectral morphing fades from bright attack to warmer sustain
5. Post-processing: drift, soundboard, saturation, release noise

Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pluck_position` | `float` | `0.15` | Pluck point as fraction of string length; affects which modes are excited via position comb filtering |
| `pluck_hardness` | `float` | `0.6` | Pluck hardness (0-1); harder plucks excite more high partials |
| `pluck_noise` | `float` | `0.3` | Level of the short noise burst mixed into the pluck attack |
| `velocity_tilt` | `float` | `0.4` | How much velocity shifts pluck hardness; higher values make louder notes brighter |
| `n_modes` | `int` | `40` | Number of string modes in the resonator bank |
| `inharmonicity` | `float` | `0.00005` | Stretch coefficient B in f_n = n *freq* sqrt(1 + B * n^2). Real harpsichords have very low inharmonicity. Set to `0.0` for pure harmonic modes (JI use). |
| `decay_base` | `float` | `1.5` | Base decay time constant in seconds |
| `decay_tilt` | `float` | `1.2` | How much faster upper modes decay relative to lower ones |
| `attack_brightness` | `float` | `1.5` | Spectral morph onset boost; values above 1.0 make the attack brighter than the sustain |
| `morph_time` | `float` | `0.3` | Seconds spent morphing from bright attack spectrum to steady sustain spectrum |
| `drift` | `float` | `0.06` | Slow sinusoidal pitch drift amount (0-1) |
| `drift_rate_hz` | `float` | `0.04` | Drift wander speed in Hz |
| `body_saturation` | `float` | `0.10` | Subtle body saturation for warmth (0=clean) |
| `soundboard_color` | `float` | `0.25` | Soundboard resonance coloring amount (0=bypass, 1=full wet) |
| `soundboard_brightness` | `float` | `0.65` | Soundboard filter cutoff position (0=dark, 1=bright) |
| `release_noise` | `float` | `0.06` | Level of the short bright noise burst near note end representing the plectrum's return past the string |
| `partial_ratios` | `list[dict] \| list[float]` | `None` | Custom partial set. Each entry is either a bare ratio or `{"ratio": float, "amp": float}`. Overrides `n_modes` and `inharmonicity` when present. |

### Register System

The harpsichord engine supports multiple registers that are blended together,
each with independent pluck character, pitch multiplier, and decay scaling.

Default registers:

| Register | `pitch_mult` | `pluck_position` | `pluck_hardness` | `brightness_tilt` | `decay_scale` | Default `blend` |
|----------|-------------|-------------------|-------------------|---------------------|----------------|-----------------|
| `front_8` | `1.0` | `0.15` | `0.6` | `0.0` | `1.0` | `1.0` |
| `back_8` | `1.0` | `0.22` | `0.55` | `-0.1` | `1.05` | `0.0` |
| `four_foot` | `2.0` | `0.12` | `0.7` | `0.15` | `0.7` | `0.0` |
| `lute` | `1.0` | `0.18` | `0.4` | `-0.3` | `0.6` | `0.0` |

Convenience blend parameters (override default register blend levels):

- `front_8_blend: float`
- `back_8_blend: float`
- `four_foot_blend: float`
- `lute_blend: float`

For fully custom register sets, pass `registers` as a list of dicts with keys:
`name`, `pitch_mult`, `pluck_position`, `pluck_hardness`, `pluck_noise`,
`brightness_tilt`, `decay_scale`, `partial_ratios`, `blend`.

Presets:

| Preset | Character |
|--------|-----------|
| `baroque` | Single front 8' register, moderate hardness, clear and articulate |
| `concert` | Front 8' + back 8' coupled, fuller body, standard concert sound |
| `bright` | Front 8' + 4' register, harder pluck, sparkling highs |
| `warm` | Front 8' + lute stop, softer pluck, darker soundboard, gentler character |
| `ethereal` | Front 8' + light back 8', soft pluck, long morph and decay, shimmering sustain |
| `glass` | Custom 11-limit partial ratios, hard pluck, crystalline and sparse |
| `septimal` | Custom 7-limit partial ratios for xenharmonic timbre-harmony fusion |

Xenharmonic usage note: for JI and xenharmonic work, set `inharmonicity=0.0` so
modes remain purely harmonic, or use `partial_ratios` to specify an explicit
set of JI ratios (e.g., septimal intervals like 7/4, 3/2, 7/2). The `septimal`
and `glass` presets demonstrate this approach. When `partial_ratios` is provided,
`n_modes` and `inharmonicity` are ignored.

Example:

```python
score.add_voice(
    "harpsichord",
    synth_defaults={
        "engine": "harpsichord",
        "preset": "baroque",
        "env": {"attack_ms": 3.0, "release_ms": 200.0},
    },
)
```

## `metallic_perc`

Implementation: [code_musics/engines/metallic_perc.py](code_musics/engines/metallic_perc.py)

Additive/FM metallic percussion engine for hihats, cymbals, cowbell, and clave.
Generates inharmonic partials at non-integer frequency ratios, with optional ring
modulation, bandpass filtering, and a transient click layer.

DSP chain: N sine partials at non-integer frequency ratios -> optional ring
modulation -> ZDF bandpass filter -> exponential decay envelope -> transient
click layer -> mix and peak-normalize.

Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_partials` | `int` | `6` | Number of additive partials |
| `partial_ratios` | `list[float]` | `None` | Custom partial frequency ratios; defaults to `sqrt(1), sqrt(2), ...` when omitted |
| `brightness` | `float` | `0.7` | Upper-partial weighting (0–1); higher values keep upper partials louder |
| `decay_ms` | `float` | `80.0` | Exponential decay time in ms |
| `ring_mod_amount` | `float` | `0.0` | Ring modulation depth (0–1) |
| `ring_mod_freq_ratio` | `float` | `1.48` | Ring modulator frequency as a ratio of the note frequency |
| `filter_center_ratio` | `float` | `1.0` | Bandpass filter center as a ratio of the note frequency |
| `filter_q` | `float` | `1.2` | Bandpass filter Q (>= 0.5) |
| `click_amount` | `float` | `0.05` | Level of the transient click layer |
| `click_decay_ms` | `float` | `3.0` | Click decay time in ms |
| `density` | `float` | `0.5` | Partial frequency jitter (0–1); adds randomness to partial tuning for thicker, less tonal results |

**Multi-point envelope params (optional):**

- `amp_envelope: list[dict]` — replaces exponential decay on partials
- `filter_envelope: list[dict]` — modulates bandpass filter cutoff (values = Hz)
- `filter_mode: str` — configurable filter mode (default `"bandpass"`)

Notes:

- does not support pitch motion (`freq_trajectory`)
- deterministic for identical inputs (SHA-256 seeded RNG)
- `freq` sets the base frequency; partial ratios are relative to it. Typical
  hi-hat frequencies are 8000–12000 Hz; cowbell around 540–800 Hz; clave around
  2000–3000 Hz.
- default partial ratios are `sqrt(n)` which gives the characteristic metallic
  inharmonicity of real cymbals
- `ring_mod_amount > 0` adds sidebands for a more complex, splashy character
  (useful for rides and crashes)

Presets:

| Preset | Character |
|--------|-----------|
| `closed_hat` | Tight, bright closed hi-hat (45 ms decay) |
| `open_hat` | Longer open hi-hat (450 ms decay) |
| `pedal_hat` | Medium pedal hi-hat between closed and open |
| `ride_bell` | Focused ride bell with ring modulation |
| `ride_bow` | Washy ride bow with many partials |
| `crash` | Long crash cymbal (2 s decay) with high density |
| `cowbell` | Two-partial cowbell with fixed ratios [1.0, 1.504] |
| `clave` | Sharp, short clave click with harmonic partials |

Example:

```python
score.add_voice(
    "hihat",
    synth_defaults={"engine": "metallic_perc", "preset": "closed_hat"},
    normalize_peak_db=-6.0,
)
```

## `noise_perc`

Implementation: [code_musics/engines/noise_perc.py](code_musics/engines/noise_perc.py)

Parameters:

- `noise_mix: float`
  Blend between pitched tone and noise layer. `0` is fully pitched; `1` is fully noise-driven.
- `pitch_decay_ms: float`
  Decay time in milliseconds for the tone component's attack transient. Also used as
  the default for `noise_decay_ms` when `noise_decay_ms` is not specified.
- `noise_decay_ms: float`
  Decay time in milliseconds for the noise body envelope, independent of
  `pitch_decay_ms`. Set this longer than `pitch_decay_ms` to get a full noise body
  after the pitched transient dies — the key parameter for clap and snare body.
  Defaults to `pitch_decay_ms` when omitted.
- `tone_decay_ms: float`
  Decay time in milliseconds for the pitched tone body.
- `bandpass_ratio: float`
  Ratio applied to the resolved note frequency to choose the center of the noise
  shaping band.
- `bandpass_width_ratio: float`
  Controls the width of the noise bandpass relative to the center frequency.
  Default `0.75`. Width is `max(80, center × bandpass_width_ratio)` Hz. Higher
  values give broader, more full-spectrum noise; the `clap` preset uses `2.5`
  for wide coverage.
- `click_amount: float`
  Level of the short transient click layer.

Recommended authoring aliases:

- `params.noise_mix_ratio`

Validation:

- `0 <= noise_mix <= 1`
- `pitch_decay_ms > 0`
- `noise_decay_ms > 0`
- `tone_decay_ms > 0`
- `bandpass_ratio > 0`
- `bandpass_width_ratio > 0`
- `click_amount >= 0`

**Multi-point envelope params (optional):**

- `tone_amp_envelope: list[dict]` — replaces tonal component exponential decay
- `noise_amp_envelope: list[dict]` — replaces noise component exponential decay
- `overall_amp_envelope: list[dict]` — applied to final mixed signal

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

Implementation: [code_musics/engines/kick_tom.py](code_musics/engines/kick_tom.py)

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
  **Deprecated.** Use `EffectSpec("saturation", ...)` on the voice instead.
  Logs a warning if present; ignored by the engine.
- `post_lowpass_hz: float`
  **Deprecated.** Use `EffectSpec("eq", ...)` on the voice instead.
  Logs a warning if present; ignored by the engine.

**Multi-point envelope params (optional — when absent, existing behavior is preserved):**

- `body_amp_envelope: list[dict]`
  Multi-point envelope for the body amplitude shape. Replaces the simple
  exponential `body_decay_ms` decay. Each dict has `time` (0-1), `value` (0-1),
  `curve` ("linear" / "exponential" / "bezier"), and optional `cx`/`cy` for
  bezier control.
- `pitch_envelope: list[dict]`
  Multi-point envelope for the frequency multiplier over time. Replaces the
  `pitch_sweep_amount_ratio` + `pitch_sweep_decay_ms` exponential sweep.
  Values are frequency ratios (e.g. 3.5 at start, 1.0 at end).
- `overtone_amp_envelope: list[dict]`
  Multi-point envelope for the overtone amplitude. Replaces the exponential
  `overtone_decay_ms` decay.

**Body filter params (optional — filter is bypassed when `body_filter_mode` is absent):**

- `body_filter_mode: str`
  `"lowpass"`, `"bandpass"`, or `"highpass"`. Applies a ZDF SVF filter to the
  body signal after assembly but before mixing with overtone/click/noise.
- `body_filter_cutoff_hz: float` (default 2000.0)
  Base cutoff frequency.
- `body_filter_q: float` (default 0.707)
  Filter resonance (>= 0.5). 0.707 = Butterworth.
- `body_filter_drive: float` (default 0.0)
  Filter drive amount. 0.0 = fully linear.
- `body_filter_envelope: list[dict]`
  Multi-point envelope for cutoff modulation. Value axis = cutoff in Hz.

**FM body params (optional — standard oscillator used when `body_fm_ratio` is absent):**

- `body_fm_ratio: float`
  Modulator frequency as ratio of body frequency. Enables FM synthesis on
  the body oscillator, replacing the standard sine/triangle/sine_clip.
- `body_fm_index: float` (default 2.0)
  Peak modulation index (controls harmonic richness).
- `body_fm_feedback: float` (default 0.0)
  Modulator self-feedback (adds noise/complexity).
- `body_fm_index_envelope: list[dict]`
  Multi-point envelope for index modulation. Value axis = 0-1 multiplier.
  Default when absent: `exp(-t/0.05)` for percussive FM.

**Body distortion params (optional — bypassed when `body_distortion` is absent):**

- `body_distortion: str`
  Waveshaping algorithm: `"hard_clip"`, `"tanh"`, `"atan"`, `"exponential"`,
  `"polynomial"`, `"logarithmic"`, `"foldback"`, `"half_wave_rect"`,
  `"full_wave_rect"`. Applied after body assembly, before the filter.
- `body_distortion_drive: float` (default 0.5)
  Drive amount (0-1). 0-0.25 subtle, 0.33-0.66 moderate, 0.66-1.0 strong.
- `body_distortion_mix: float` (default 1.0)
  Dry/wet blend (0 = fully dry, 1 = fully wet).
- `body_distortion_drive_envelope: list[dict]`
  Multi-point envelope for drive modulation over time.

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

## `organ`

Implementation: [code_musics/engines/organ.py](code_musics/engines/organ.py)

Drawbar organ engine covering both Hammond-style tonewheel and pipe organ
character. Additive synthesis at its core, with drawbar-level mixing, tonewheel
drift, key click, scanner vibrato/chorus, leakage, and per-drawbar harmonic
shaping.

Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `drawbars` | `list[int]` | `[0,8,8,8,0,0,0,0,0]` | 9 drawbar levels, each 0-8 (Hammond convention) |
| `drawbar_ratios` | `list[float]` | `[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]` | frequency ratios for each drawbar; override for xenharmonic timbre |
| `click` | `float` | `0.15` | key click / pipe chiff amount (0-1) |
| `click_brightness` | `float` | `0.5` | spectral center of click (0-1; lower=breathier chiff, higher=sharper click) |
| `vibrato_depth` | `float` | `0.0` | scanner vibrato depth (0-1; ~0.1=V1, ~0.3=V2, ~0.6=V3) |
| `vibrato_rate_hz` | `float` | `6.8` | scanner vibrato rate |
| `vibrato_chorus` | `float` | `0.0` | blend dry+modulated for chorus effect (0=pure vibrato, 1=full chorus) |
| `drift` | `float` | `0.12` | organic tonewheel detuning (0-1, maps to 0-4 cents) |
| `drift_rate_hz` | `float` | `0.07` | drift wander speed |
| `leakage` | `float` | `0.08` | crosstalk from adjacent drawbar tonewheels (0-1) |
| `tonewheel_shape` | `float` | `0.0` | per-drawbar harmonic richness: 0=pure sine (Hammond), 0.3-0.5=warm flue pipe, 0.6+=brighter principal/reed |

Hammond drawbar convention: the 9 drawbars correspond to pipe footages: 16'
(sub-octave), 8' (fundamental), 5-1/3' (3rd harmonic), 4' (2nd harmonic),
2-2/3' (6th), 2' (8th), 1-3/5' (10th), 1-1/3' (12th), 1' (16th). Levels 0-8
follow the real Hammond convention. For xenharmonic use, override
`drawbar_ratios` with any set of frequency ratios.

Presets:

| Preset | Character |
|--------|-----------|
| `warm` | Classic warm Hammond, fundamental-heavy, gentle chorus |
| `full` | Full registration, rock/soul wall of sound |
| `jazz` | Clean jazz combo, light registration |
| `gospel` | Bright, gritty, strong click and vibrato |
| `cathedral` | Pipe organ, no vibrato, pipe harmonic color (shape=0.4), more drift |
| `baroque` | Clear, articulate Bach-friendly, warm pipe color (shape=0.3) |
| `septimal` | 7-limit xenharmonic drawbar ratios for timbre-harmony fusion |
| `glass_organ` | 11-limit ethereal shimmer with custom drawbar ratios |

Xenharmonic usage note: the default Hammond drawbar ratios (0.5, 1, 1.5, 2, 3,
4, 5, 6, 8) are already harmonic-series based and naturally JI-compatible. For
deeper timbre-harmony fusion, override `drawbar_ratios` with JI intervals (e.g.,
septimal ratios like 7/4, 7/2, 7/6) so the organ's internal harmonics reinforce
the scale's intervallic structure.

Example:

```python
score.add_voice(
    "organ",
    synth_defaults={
        "engine": "organ",
        "preset": "warm",
        "env": {"attack_ms": 8.0, "release_ms": 80.0},
    },
)
```

## `piano`

Implementation: [code_musics/engines/piano.py](code_musics/engines/piano.py)

Modal piano synthesis with physical hammer-string interaction. Each note is
rendered as a bank of second-order resonators (one per string mode) excited by a
nonlinear hammer contact model (`F = K * max(delta, 0)^p`). Velocity naturally
shapes timbre through the hammer physics rather than through a separate
brightness knob. The engine uses two-phase rendering: Numba JIT for the contact
phase (~2-8 ms) and vectorized NumPy for the free decay, giving both physical
realism and reasonable render speed. Supports unison strings with drift and
detune, soundboard coloring via a resonator bank, body saturation, and damper
noise. The engine peak-normalizes internally and is deterministic for identical
inputs.

Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_modes` | `int` | `32` | Number of string modes in the resonator bank |
| `inharmonicity` | `float` | `0.0005` | Stretch coefficient B in f_n = n *freq* sqrt(1 + B * n^2). Higher values widen upper modes (real pianos ~0.0001-0.001). Set to `0.0` for pure harmonic modes (JI use). |
| `partial_ratios` | `list[dict] \| list[float]` | `None` | Custom partial set. Each entry is either a bare ratio or `{"ratio": float, "amp": float}`. Overrides `n_modes` and `inharmonicity` when present. |
| `decay_base` | `float` | `3.5` | Base decay time constant in seconds |
| `decay_tilt` | `float` | `0.5` | How much faster upper modes decay relative to lower ones (0-1) |
| `hammer_mass` | `float` | `0.01` | Normalized hammer mass affecting contact duration and energy transfer |
| `hammer_stiffness` | `float` | `1e8` | K in the contact force law; higher values give harder, brighter attacks |
| `hammer_exponent` | `float` | `2.5` | Felt nonlinearity exponent p; higher values increase velocity-dependent brightness |
| `hammer_position` | `float` | `0.12` | Strike point as fraction of string length; affects which modes are excited |
| `bridge_position` | `float` | `0.95` | Output pickup position as fraction of string length |
| `max_hammer_velocity` | `float` | `4.0` | Maximum hammer velocity in m/s for velocity scaling |
| `drift` | `float` | `0.08` | Slow sinusoidal pitch drift amount for all strings (0-1, maps to 0-4 cents) |
| `drift_rate_hz` | `float` | `0.05` | Drift wander speed in Hz |
| `unison_count` | `int \| None` | `None` | Number of unison strings. `None` = automatic: 1 below 200 Hz, 2 below 400 Hz, 3 above. |
| `unison_detune` | `float` | `3.0` | Detune spread between unison strings in cents |
| `unison_drift` | `float` | `0.15` | Additional drift amount for non-primary unison strings (0-1) |
| `body_saturation` | `float` | `0.15` | Subtle body saturation for warmth (0=clean, higher=warmer) |
| `soundboard_color` | `float` | `0.4` | Soundboard resonance coloring amount (0=bypass, 1=full wet) |
| `soundboard_brightness` | `float` | `0.5` | Soundboard filter cutoff position (0=dark ~200 Hz, 1=bright ~4200 Hz) |
| `damper_noise` | `float` | `0.08` | Level of the short noise burst near note end representing damper contact (0=off) |

Backward compatibility: `n_partials` is accepted as an alias for `n_modes`, and
`decay_partial_tilt` is accepted as an alias for `decay_tilt`. Legacy additive
parameters like `hammer_hardness`, `brightness`, and `hammer_noise` are silently
ignored (physical defaults are used instead).

Presets:

| Preset | Character |
|--------|-----------|
| `grand` | Full concert grand with rich modes, moderate inharmonicity, and warm soundboard |
| `bright` | Stiffer hammer with more upper-mode presence |
| `warm` | Lower stiffness, darker soundboard, gentler overall character |
| `felt` | Very soft felt-damped piano; low stiffness, muted attack, shorter decay |
| `honky_tonk` | Wide unison detune and extra drift for a detuned barroom character |
| `tack` | High stiffness and exponent for a percussive, bright attack — tack piano / prepared character |
| `glass` | Fewer modes, more inharmonicity, long decay, minimal soundboard — crystalline and sparse |
| `septimal` | Custom 7-limit partial ratios for xenharmonic timbre-harmony fusion |

Xenharmonic usage note: for JI and xenharmonic work, set `inharmonicity=0.0` so
modes remain purely harmonic, or use `partial_ratios` to specify an explicit
set of JI ratios (e.g., septimal intervals like 7/4, 3/2, 7/2). The `septimal`
preset demonstrates this approach with a 7-limit partial set. When
`partial_ratios` is provided, `n_modes` and `inharmonicity` are ignored.

Example:

```python
score.add_voice(
    "piano",
    synth_defaults={
        "engine": "piano",
        "preset": "grand",
        "env": {"attack_ms": 5.0, "release_ms": 400.0},
    },
)
```

## `piano_additive`

Implementation: [code_musics/engines/piano_additive.py](code_musics/engines/piano_additive.py)

Legacy additive piano synthesis with physical modeling envelopes. Each note
renders a set of sinusoidal partials with stretched tuning (inharmonicity),
per-partial two-stage exponential decay, velocity-dependent hammer excitation
(filtered noise burst), optional unison strings with drift and detune,
soundboard coloring via ZDF SVF, and a short damper thump near the note release.
This is the original piano engine, now available as `piano_additive` after the
modal engine took over the `piano` name. Same presets and parameter surface as
before.

Use `engine="piano_additive"` in `synth_defaults` to access the legacy engine.

## `polyblep`

Implementation: [code_musics/engines/polyblep.py](code_musics/engines/polyblep.py)

Time-domain bandlimited oscillator using polynomial BLEPs (Bandlimited Step
functions). Generates waveforms directly rather than summing sine harmonics, so
there is no Gibbs phenomenon at discontinuities. The resulting sound has a smooth
analog character with a correct `1/n` harmonic spectrum, then passes through an
internal zero-delay-feedback / topology-preserving state-variable filter.

Parameters:

- `waveform: str`
  Oscillator waveform. Supported values: `saw`, `square`, `triangle`, `sine`.
  Triangle is generated by integrating the bandlimited square (BLAMP approach) and
  is alias-free; `pulse_width` is ignored for triangle and sine. Sine is a pure
  fundamental with no harmonics — useful for deep sub-bass.
- `pulse_width: float`
  Pulse width used when `waveform="square"`. `0.5` is a symmetric square wave.
- `osc2_level: float`
  Level of the optional second oscillator. `0.0` disables it; higher values blend
  in a second PolyBLEP oscillator before the filter, normalized against oscillator 1.
- `osc2_waveform: str`
  Optional waveform for oscillator 2. Supported values: `saw`, `square`,
  `triangle`, `sine`. Defaults to the main `waveform`.
- `osc2_pulse_width: float`
  Pulse width for oscillator 2 when `osc2_waveform="square"`. Defaults to the main
  `pulse_width`.
- `osc2_semitones: float`
  Coarse oscillator-2 tuning offset in semitones. Default `0.0`.
- `osc2_detune_cents: float`
  Fine oscillator-2 tuning offset in cents. Default `0.0`.
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
- `osc2_spread_power: float`
  Power-scaled unison detune for osc2. `1.0` (default) is linear detune, matching
  prior behavior. Values above `1.0` cluster the detune near center for a warmer,
  more focused sound. Values below `1.0` spread detune more evenly for a wider,
  more diffuse image. Range `[0.5, 3.0]`.

### Analog Character (polyblep)

These parameters add per-note analog imperfection. They are shared with the
`filtered_stack` and `fm` engines (with filter-specific params only on engines
that have filters). All default to subtle, conservative values and can be zeroed
individually.

- `pitch_drift: float`
  1/f multi-rate pitch drift. Default `0.12` (~1.5 cents peak). `0` disables.
  Conservative for JI work. Range `[0, 1]`.
- `analog_jitter: float`
  Per-note parameter variation: cutoff ±3%, attack ±12%, amp ±0.3 dB, random
  phase. Default `1.0`. `0` disables all jitter. Range `[0, 2]`.
- `noise_floor: float`
  Pink noise floor level, envelope-following, bandlimited 100–8000 Hz. Default
  `0.001` (−60 dBFS). `0` disables. Range `[0, 0.01]`.
- `drift_rate_hz: float`
  Base frequency of the drift oscillator. Higher values produce faster wander.
  Default `0.07`. Range `[0.01, 1.0]`.
- `cutoff_drift: float`
  Ornstein-Uhlenbeck mean-reverting drift on filter cutoff for organic filter
  movement. Default `0.5`. `0` gives a static cutoff. Range `[0, 2]`.
- `filter_even_harmonics: float`
  Asymmetric bias in the driven filter for even-harmonic warmth. `0` (default)
  gives symmetric (odd-only) saturation. Non-zero values add 2nd/4th harmonic
  content via an envelope-tracked DC bias in the filter. Range `[0, 0.5]`.

Validation:

- `cutoff_hz > 0`
- `reference_freq_hz > 0`
- `filter_env_decay > 0`
- `filter_drive >= 0`
- `osc2_level >= 0`
- `0 < pulse_width < 1`
- `0 < osc2_pulse_width < 1`
- `waveform in {"saw", "square", "triangle", "sine"}`
- `osc2_waveform in {"saw", "square", "triangle", "sine"}`
- `filter_mode in {"lowpass", "bandpass", "highpass", "notch"}`

Notes:

- Unlike `filtered_stack`, the spectrum is generated in the time domain, so there
  is no `n_harmonics` cap and no Gibbs ringing at waveform discontinuities.
- The filter sweep is handled sample by sample through a topology-preserving
  state-variable filter, so modulation is smoother and more analog-like than the
  previous segmented approximation.
- The output is peak-normalized before the final amplitude scale, matching the
  behavior of the `fm` engine.
- For stacked subtractive sounds, use `osc2_level` plus small `osc2_detune_cents`
  values for width, or `osc2_semitones=-12` for a built-in sub layer.
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

Filter drive implementation note:

The driven SVF uses topology-aware saturation rather than naive waveshaping:

- Single ADAA (antiderivative anti-aliasing) tanh at the feedback summation point
  eliminates aliasing from the nonlinearity.
- `algebraicSat` for integrator state limiting keeps the filter transparent at low
  levels while preventing runaway at high drive/resonance.
- Bidirectional drive/resonance interaction: higher drive naturally compresses the
  resonance peak, so the filter does not scream at moderate drive + resonance
  combos the way a naive driven SVF would.
- The drive range 0.05–0.4 now has a much wider "warm" sweet spot than before,
  making it practical for always-on voice coloring rather than only aggressive
  distortion.

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
- `juno_pad` - warm saw pad with moderate filter, chorus-ready, osc2 detuned 5 cents.
- `analog_bass` - deep saw + square sub bass with filter envelope.
- `prophet_lead` - bright saw lead with resonant filter sweep.
- `glass_pad` - gentle sine pad with subtle triangle osc2.

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

## `snare`

Implementation: [code_musics/engines/snare.py](code_musics/engines/snare.py)

909-inspired snare drum engine with a three-layer architecture: pitched body with
sweep, comb-filtered wire buzz, and broadband transient click.

The body layer uses a pitched tone with an overtone and an initial pitch sweep
(similar to `kick_tom`). The wire layer passes white noise through a comb filter
tuned to the body pitch, then bandpass-filters the result, creating the
characteristic snare buzz that's harmonically related to the drum's tuning.

Parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `body_decay_ms` | `float` | `120.0` | Body tone decay time in ms |
| `body_overtone_ratio` | `float` | `1.6` | Frequency ratio of the body overtone |
| `body_sweep_ratio` | `float` | `1.8` | Starting pitch multiplier for the body sweep |
| `body_sweep_decay_ms` | `float` | `15.0` | Decay time for the pitch sweep |
| `wire_decay_ms` | `float` | `180.0` | Wire/snare buzz decay time in ms |
| `wire_center_ratio` | `float` | `3.0` | Bandpass center for wire noise as a ratio of the note freq |
| `wire_q` | `float` | `0.8` | Bandpass Q for the wire filter (>= 0.5) |
| `comb_amount` | `float` | `0.45` | Comb filter feedback (0–1); higher values add more pitched resonance to the wire |
| `body_mix` | `float` | `0.5` | Level of the body layer in the final mix |
| `wire_mix` | `float` | `0.5` | Level of the wire layer in the final mix |
| `click_amount` | `float` | `0.15` | Level of the broadband click transient |
| `click_decay_ms` | `float` | `5.0` | Click decay time in ms |

**Multi-point envelope params (optional):**

- `body_amp_envelope: list[dict]` — replaces body exponential decay
- `wire_amp_envelope: list[dict]` — replaces wire exponential decay
- `body_pitch_envelope: list[dict]` — replaces body pitch sweep (values = freq multiplier)
- `wire_filter_envelope: list[dict]` — modulates wire bandpass filter cutoff (values = Hz)

Notes:

- does not support pitch motion (`freq_trajectory`)
- deterministic for identical inputs (SHA-256 seeded RNG)
- `freq` sets the snare body pitch; typical snare frequencies are 150–250 Hz
- the comb filter tunes the wire resonance to the body pitch, so wire and body
  are harmonically related
- pair with `EffectSpec("compressor", {"preset": "snare_punch"})` or
  `EffectSpec("saturation", {"preset": "snare_bite"})` for finished snare sounds
- uses Numba JIT for the comb filter inner loop

Presets:

| Preset | Character |
|--------|-----------|
| `909_tight` | Tight 909-style snare with balanced body/wire |
| `909_fat` | Fatter 909 with longer decay and more body emphasis |
| `808_snare` | Longer 808-style snare with softer wire |
| `rim_shot` | Short, sharp rim shot dominated by click transient |
| `brush` | Soft brush sweep with long wire decay and minimal body |
| `cross_stick` | Very short, sharp cross-stick click |

Example:

```python
score.add_voice(
    "snare",
    synth_defaults={"engine": "snare", "preset": "909_tight"},
    normalize_peak_db=-6.0,
    effects=[
        EffectSpec("compressor", {"preset": "snare_punch"}),
        EffectSpec("saturation", {"preset": "snare_bite"}),
    ],
)
```

## `surge_xt`

Implementation: [code_musics/engines/surge_xt.py](code_musics/engines/surge_xt.py)

External instrument engine that renders voices through Surge XT via pedalboard's
VSTi hosting. Unlike the native per-note engines, it renders the whole voice at
once by building a MIDI stream and feeding it to the plugin. Uses MPE-style
per-note pitch bend with a 48-semitone range (24 semitones up/down) for sub-cent
microtonal accuracy.

The engine is registered separately from the per-note engine registry because it
operates at the voice level rather than the note level. The score renderer
detects `engine="surge_xt"` and delegates the entire voice to
`surge_xt.render_voice()`.

Parameters:

- `preset_path: str`
  Path to a `.vstpreset` or `.fxp` file to load before rendering.
- `raw_state: bytes`
  Serialised plugin state. Use this to restore a previously captured Surge XT
  patch state programmatically. Takes effect only when `preset_path` is not set.
- `surge_params: dict[str, float]`
  Direct Surge XT parameter overrides. Keys are Surge XT parameter names (e.g.,
  `a_osc1_type`, `a_filter1_cutoff`); values are raw floats in the plugin's
  internal range. Applied after preset/state loading. Unknown parameter names
  are logged as warnings and skipped.
- `mpe: bool`
  When `True` (default), sends MCM (MPE Configuration Message) to enable MPE
  Lower Zone so pitch bend is per-note with sub-cent accuracy. When `False`,
  uses global-bend chord mode: notes are grouped into chords by start time,
  each chord shares a single pitch bend derived from the bass note, and
  consecutive chords glide smoothly between reference pitches (like a tremolo
  bar). Non-bass notes have up to ~50 cent rounding error from MIDI note
  quantisation; the bass is always perfectly tuned.
- `global_glide_time: float`
  Seconds for the pitch-bend glide between consecutive chords in global-bend
  mode (default `0.4`). Only used when `mpe=False`.
- `cc_curves: list[dict[str, Any]]`
  MIDI CC automation curves. Each entry has `cc` (0--127), optional `channel`
  (default 0), and `points` (list of `(time_seconds, value_0_to_1)`
  breakpoints). Breakpoints are linearly interpolated at 10 ms resolution and
  sent as MIDI CC messages. Use this for expression, mod wheel, or any CC-mapped
  Surge XT parameter.
- `tail_seconds: float`
  Extra render time after last note-off (default `2.0`). Captures release tails
  and reverb decay from the plugin's internal effects.
- `release_padding: float`
  Seconds to keep an MPE channel reserved after note-off so pitch bend from new
  notes does not bleed into release tails (default `1.0`).
- `buffer_size: int`
  Pedalboard processing block size in samples (default `256`). MIDI events are
  delivered at block boundaries, so smaller blocks give finer-grained pitch bend
  and CC timing. The pedalboard default of 8192 (~186 ms) produces audible
  staircase artifacts on pitch glides; 256 (~5.8 ms) matches typical DAW host
  granularity.
- `param_curves: list[dict[str, Any]]`
  **Experimental / broken.** Direct Surge XT parameter automation via chunked
  rendering. Produces audible clicking and popping at chunk boundaries because
  the plugin's internal DSP state cannot smoothly transition across parameter
  steps. Prefer native post-processing effects with score-time automation, or
  use `cc_curves` for smoother MIDI-rate automation. Retained for
  experimentation only.

Per-note fields (passed in the notes list, not in params):

- `glide_from: float`
  Starting frequency in Hz for a per-note pitch sweep toward the note's `freq`.
  The glide is linear in pitch-bend space at ~200 Hz update rate (5 ms steps).
  If the glide span exceeds the 24-semitone bend range, a warning is logged and
  the glide is skipped.
- `glide_time: float`
  Duration of the glide in seconds (defaults to the note's full `duration`).

Notes:

- expects `~/.vst3/Surge XT.vst3` to be installed
- if Surge XT is not found, the voice renders as silence with a warning
- the engine handles up to 15 simultaneous MPE notes (MIDI channels 1--15);
  channel collisions are logged as warnings
- the output is always stereo (2 channels)
- silent tail after the last note-off is automatically trimmed
- no presets are defined in the engine registry; sound design is done through
  Surge XT patches (`preset_path` / `raw_state`) and `surge_params` overrides
- real-time parameter automation is limited: `cc_curves` provides MIDI-rate
  control, but continuous parameter sweeps require pre-configured modulation
  matrix routing inside the Surge XT patch itself

Example:

```python
score.add_voice(
    "pad",
    synth_defaults={
        "engine": "surge_xt",
        "params": {
            "preset_path": "patches/warm_pad.fxp",
            "tail_seconds": 3.0,
        },
    },
)
```

Global-bend chord mode example (tremolo-bar-style chord slides):

```python
score.add_voice(
    "chords",
    synth_defaults={
        "engine": "surge_xt",
        "params": {
            "preset_path": "patches/strings.fxp",
            "mpe": False,
            "global_glide_time": 0.3,
        },
    },
)
```

## `vital`

Implementation: [code_musics/engines/vital.py](code_musics/engines/vital.py)

External instrument engine that renders voices through Vital (wavetable synth)
via pedalboard's VSTi hosting. Same MPE per-note pitch bend approach as
`surge_xt` for sub-cent microtonal accuracy: each note gets its own MIDI channel
with independent pitch bend. Uses a 24-semitone bend range (12 semitones
up/down). Vital ignores standard MPE RPN messages for bend range configuration,
so `mpe_enabled` and `pitch_bend_range` are set via the parameter API instead.

The engine is registered separately from the per-note engine registry because it
operates at the voice level rather than the note level. The score renderer
detects `engine="vital"` and delegates the entire voice to
`vital.render_voice()`.

Parameters:

- `preset_path: str`
  Path to a `.vital` preset file to load before rendering.
- `raw_state: bytes`
  Serialised plugin state. Use this to restore a previously captured Vital patch
  state programmatically. Takes effect only when `preset_path` is not set.
- `vital_params: dict[str, float]`
  Direct Vital parameter overrides. Keys are Vital parameter names; values are
  raw floats in the plugin's internal range. Applied after preset/state loading.
  Unknown parameter names are logged as warnings and skipped.
- `mpe: bool`
  When `True` (default), enables MPE per-note pitch bend with sub-cent accuracy.
  When `False`, uses global-bend chord mode: notes are grouped into chords by
  start time, each chord shares a single pitch bend derived from the bass note,
  and consecutive chords glide smoothly between reference pitches. Non-bass
  notes have up to ~50 cent rounding error from MIDI note quantisation; the bass
  is always perfectly tuned.
- `global_glide_time: float`
  Seconds for the pitch-bend glide between consecutive chords in global-bend
  mode (default `0.4`). Only used when `mpe=False`.
- `cc_curves: list[dict[str, Any]]`
  MIDI CC automation curves. Each entry has `cc` (0--127), optional `channel`
  (default 0), and `points` (list of `(time_seconds, value_0_to_1)`
  breakpoints). Breakpoints are linearly interpolated at 10 ms resolution and
  sent as MIDI CC messages.
- `tail_seconds: float`
  Extra render time after last note-off (default `2.0`). Captures release tails
  and reverb decay from the plugin's internal effects.
- `release_padding: float`
  Seconds to keep an MPE channel reserved after note-off so pitch bend from new
  notes does not bleed into release tails (default `1.0`).
- `buffer_size: int`
  Pedalboard processing block size in samples (default `256`). MIDI events are
  delivered at block boundaries, so smaller blocks give finer-grained pitch bend
  and CC timing.

Per-note fields (passed in the notes list, not in params):

- `glide_from: float`
  Starting frequency in Hz for a per-note pitch sweep toward the note's `freq`.
- `glide_time: float`
  Duration of the glide in seconds (defaults to the note's full `duration`).

Notes:

- expects `~/.vst3/Vital.vst3` to be installed (symlink to system install)
- if Vital is not found, the voice renders as silence with a warning
- the engine handles up to 15 simultaneous MPE notes (MIDI channels 1--15)
- the output is always stereo (2 channels)
- silent tail after the last note-off is automatically trimmed
- no presets are defined in the engine registry; sound design is done through
  Vital patches (`preset_path` / `raw_state`) and `vital_params` overrides
- Vital's MPE bend range is configured via `mpe_enabled` and `pitch_bend_range`
  parameters on the plugin instance, not via MIDI RPN messages

Example:

```python
score.add_voice(
    "pad",
    synth_defaults={
        "engine": "vital",
        "params": {
            "preset_path": "patches/warm_wavetable.vital",
            "tail_seconds": 3.0,
        },
    },
)
```

Global-bend chord mode example:

```python
score.add_voice(
    "chords",
    synth_defaults={
        "engine": "vital",
        "params": {
            "preset_path": "patches/keys.vital",
            "mpe": False,
            "global_glide_time": 0.3,
        },
    },
)
```
