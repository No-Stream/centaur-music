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
- [code_musics/engines/drum_voice.py](code_musics/engines/drum_voice.py)
- [code_musics/engines/kick_tom.py](code_musics/engines/kick_tom.py)
- [code_musics/engines/metallic_perc.py](code_musics/engines/metallic_perc.py)
- [code_musics/engines/noise_perc.py](code_musics/engines/noise_perc.py)
- [code_musics/engines/organ.py](code_musics/engines/organ.py)
- [code_musics/engines/piano.py](code_musics/engines/piano.py)
- [code_musics/engines/piano_additive.py](code_musics/engines/piano_additive.py)
- [code_musics/engines/polyblep.py](code_musics/engines/polyblep.py)
- [code_musics/engines/sample.py](code_musics/engines/sample.py)
- [code_musics/engines/snare.py](code_musics/engines/snare.py)
- [code_musics/engines/surge_xt.py](code_musics/engines/surge_xt.py)
- [code_musics/engines/va.py](code_musics/engines/va.py)
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

### Per-stage curve powers

Each ADSR stage can reshape its linear ramp through a per-stage exponent. The
shaped position is `s = p**power` where `p` is the linear ramp `[0, 1)` across
the stage; the stage then interpolates `start + (end - start) * s`. With
`power == 1.0` the stage is linear and the output is identical to the legacy
ADSR.

- `attack_power: float` — default `1.0`. Values `< 1` give a concave
  (fast-start) attack; values `> 1` give a convex (slow-start) attack.
  Clamped to `[0.1, 8.0]`.
- `decay_power: float` — default `1.0`. Values `> 1` make the decay linger
  near the top before falling to `sustain_level`; values `< 1` drop quickly
  then ease into sustain. Clamped to `[0.1, 8.0]`.
- `release_power: float` — default `1.0`. Values `< 1` give a natural
  exponential-looking release (concave, sounds like energy dissipation);
  values `> 1` hold near the sustain level longer before dropping to zero.
  Clamped to `[0.1, 8.0]`.
- `attack_target: float` — default `1.0`. When `> 1.0` (typical `1.2`), the
  attack ramp is scaled toward `attack_target` and clamped at `1.0`. The
  shaped ramp therefore reaches `1.0` with a non-zero slope instead of
  asymptoting, producing a "pokey" analog-feeling attack top (VCV
  Fundamental `ATT_TARGET` idiom). Clamped to `[1.0, 1.5]`.

Recommended starting points (not applied automatically):

- For acoustic-flavored voices, `decay_power=2.0` and `release_power=2.0`
  give natural exponential-looking decay and release.
- For "pokey" analog attacks on pads or plucks, try `attack_target=1.2`
  alone, or combine with `attack_power=1.5..2.5` for more character.

These params are also registered as automation targets, so they can be swept
over time like any other synth param. They are not drifted by
`EnvelopeHumanizeSpec`; the curve shape is treated as authored intent rather
than ensemble variation.

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

### Per-Sample Engine Param Profiles

The modulation matrix (`code_musics/modulation.py`) can ride selected
synth params at audio rate via the engine-level `param_profiles`
kwarg on `render_note_signal`.  Engines opt in by registering via
`register_param_profile_support(engine_name)`.  When a profile is
supplied for a supported param, it replaces the scalar base before
the engine's internal envelope/drift/jitter stack.  Engines not in
the whitelist silently ignore the kwarg.

Current support:

| Engine | Per-sample params |
|---|---|
| `polyblep` | `cutoff_hz`, `pulse_width`, `osc2_detune_cents`, `osc2_freq_ratio` |
| `va` | `cutoff_hz`, `osc_spread_cents` |

All other synth destinations still resolve to a scalar at note
onset.  See `FUTURE.md` for the deferred per-sample destinations
(`filter_morph`, `resonance_q`, `hpf_cutoff_hz`, comb/filter1/2
family, `vibrato_depth`, etc.).  The `ModConnection` surface is
unchanged across the two modes — only the plumbing into the engine
differs.

For audio-rate modulation (above `LFOSource`'s 200 Hz cap), pair
these destinations with `OscillatorSource` in
`code_musics/modulation.py` — a sibling to `LFOSource` with
`waveshape` in `{"sine", "saw", "triangle"}` and no upper rate
limit.  Typical musical uses: PWM on `pulse_width` (polyblep),
cross-osc FM on `osc2_freq_ratio`, and audio-rate detune
modulation on `osc_spread_cents` (va supersaw).

### Per-note voice distortion (`voice_dist_*`)

`polyblep`, `va`, and `filtered_stack` expose a per-note distortion
slot placed **inside the engine's note loop, after the VCA but
before per-note buffers are summed into the voice output**.  This
is the RePro-5 idiom: chord notes distort independently, so
chord-tone intermodulation is preserved cleanly instead of
collapsing into IMD mud when you sum first and distort after.

Params (same surface across the three engines):

- `voice_dist_mode: str = "off"` — one of `"off"`, `"soft_clip"`,
  `"hard_clip"`, `"foldback"`, `"corrode"` (bitcrush + rate
  reduction), `"saturation"` (dispatches through the modern
  `apply_saturation`), `"preamp"` (flux-domain transformer via
  `apply_preamp`).
- `voice_dist_drive: float = 0.5` — 0.0–2.0.  `drive <= 0` is a
  fast-path passthrough.  Modulation through zero uses a blend
  coefficient so there is no step at the threshold.
- `voice_dist_mix: float = 1.0` — wet/dry inside the slot.  Default
  is fully wet because this is the voice distortion, not a bus
  effect; dial drive down or use a voice-level `EffectSpec`
  insert for blended flavors.
- `voice_dist_tone: float = 0.0` — -1..1 pre-stage tilt.  Positive
  brightens (high-shelf blend), negative darkens.

`voice_dist_mode="off"` is the default.  With the default params
the slot is a true no-op and existing pieces render bit-for-bit
identically.

### Per-sample oscillator phase noise (`osc_phase_noise`)

`polyblep` and `va` accept `osc_phase_noise: float = 0.0`
(0.0–1.0).  At `1.0` the oscillator phase accumulator picks up ~1
cent of jitter per sample — audible as organic zero-crossing
texture, not pitch wobble.  Distinct from `analog_jitter` (per-note
onset jitter plus the OB-Xd-style 4 kHz-smoothed CV dither) and
from `drift_bus` (0.05–0.5 Hz correlated bus drift): phase noise
is broadband, per-sample, per-voice, and deterministic from the
note hash.  Each oscillator (osc1/osc2 in polyblep; the 7 supersaw
voices in va) gets an independent RNG stream.

## Engine Selection

Set the engine with:

```python
synth_defaults = {"engine": "fm"}
```

Available engines:

- `additive`
- `clap`
- `drum_voice` — unified composable percussion engine (four mixable layers:
  exciter, tone, noise, metallic). Covers all kick/snare/clap/hat/metallic
  territory plus Machinedrum-inspired EFM, PI modal, and digital-character
  kernels. Recommended successor to the five separate drum engines.
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
- `sample` — WAV-playback engine with pitch tracking, optional filter slot,
  retrigger/flam, and Machinedrum-E12-style bend/ring/bit-crush macros.
  Useful for one-shot samples outside the `drum_voice` layer architecture.
- `snare`
- `surge_xt`
- `va` — 90s/00s Virtual Analog (JP-8000 supersaw and Waldorf Q-style
  spectralwave oscillator modes). Dual filter slots, pre-filter drive,
  optional resonant comb. Presets for Roland / Access / Waldorf flavors.
- `vital`

## Presets

Set a preset with:

```python
synth_defaults = {"engine": "filtered_stack", "preset": "warm_pad"}
```

Available presets:

- `additive`: `soft_pad`, `drone`, `bright_pluck`, `organ`, `ji_fusion_pad`, `septimal_reed`, `eleven_limit_glass`, `utonal_drone`, `plucked_ji`, `breathy_flute`, `ancient_bell`, `whispered_chord`, `struck_membrane`, `singing_bowl`, `marimba_bar`, `convolved_bell`, `thick_drone`, `fractal_fifth`, `fractal_septimal`, `vowel_a_pad`, `singing_glass`, `gravity_cloud`, `drifting_to_just`, `living_drone`, `candle_light`, `brush_breath`, `brush_cymbal`, `stiff_piano`, `dispersed_pad`, `smear_drone`, `shepard_bells`, `chaos_cloud`
- `fm`: `bell`, `glass_lead`, `metal_bass`, `dx_piano`, `lately_bass`, `fm_clav`, `fm_mallet`, `chorused_ep`
- `filtered_stack`: `warm_pad`, `reed_lead`, `round_bass`, `saw_pad`, `string_pad`, `analog_strings`
- `kick_tom`: `808_hiphop`, `808_house`, `808_tape`, `909_techno`, `909_house`, `909_crunch`, `distorted_hardkick`, `zap_kick`, `round_tom`, `floor_tom`, `electro_tom`, `ring_tom`, `gated_808`, `pitch_dive`, `filtered_kick`, `fm_body_kick`, `foldback_kick`, `808_resonant`, `808_resonant_long`, `resonant_tom`, `melodic_resonator`, `kick_bell`
- `metallic_perc`: `closed_hat`, `open_hat`, `pedal_hat`, `ride_bell`, `ride_bow`, `crash`, `cowbell`, `clave`, `swept_hat`, `decaying_bell`, `harmonic_bell`, `septimal_bell`, `square_gamelan`, `beating_hat_a`, `beating_hat_b`, `beating_hat_c`, `808_closed_hat`, `808_open_hat`, `808_cowbell_square`
- `noise_perc`: `kickish`, `snareish`, `tick`, `chh`, `clap`, `shaped_hit`
- `snare`: `909_tight`, `909_fat`, `808_snare`, `rim_shot`, `brush`, `cross_stick`, `gated_snare`, `fm_snare`, `driven_snare`, `fm_tom`, `fm_noise_burst`
- `clap`: `909_clap`, `tight_clap`, `big_clap`, `finger_snap`, `hand_clap`, `gated_clap`, `909_clap_authentic`, `scattered_clap`, `granular_cascade`, `micro_burst`
- `drum_voice`: All presets from the five drum engines above are available, plus hybrid presets. See the `drum_voice` engine section for the full list.
- `organ`: `warm`, `full`, `jazz`, `gospel`, `cathedral`, `baroque`, `septimal`, `glass_organ`
- `harpsichord`: `baroque`, `concert`, `bright`, `warm`, `ethereal`, `glass`, `septimal`
- `piano`: `grand`, `bright`, `warm`, `felt`, `honky_tonk`, `tack`, `glass`, `septimal`
- `piano_additive`: `grand`, `bright`, `warm`, `felt`, `honky_tonk`, `tack`, `glass`, `septimal`
- `polyblep`: `warm_lead`, `synth_pluck`, `analog_brass`, `square_lead`, `hoover`, `moog_bass`, `sync_lead`, `acid_bass`, `sub_bass`, `resonant_sweep`, `soft_square_pad`, `juno_pad`, `analog_bass`, `prophet_lead`, `glass_pad`, `moog_lead`, `moog_bass_ladder`, `cs80_brass`, `oberheim_pad`, `jupiter_saw`, `acid_ladder`, `diva_bass_resonance`, `cs80_attack`, `prophet_pad`, `moog_acid_newton`, `sk_bite_lead`, `cascade_bass`, `sync_screamer`, `ring_mod_lead`

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

### `bbd_chorus`

Implementation: [code_musics/synth.py](code_musics/synth.py)

Juno-faithful native bucket-brigade-style stereo chorus. The signature move
is **quadrature LFOs** (L at 0, R at 90 degrees) that produce genuine stereo
decorrelation from a mono input, **cross-feedback** (L -> R and R -> L, never
self-feedback) that keeps the wet field airy instead of metallic, and pre/post
bandlimiting that gives the wet path the soft, rolled-off BBD flavor. Unlike
the `chorus` effect, the wet image is **summed** with the dry signal (not
crossfaded) — `mix` scales the wet contribution, so the dry signal is always
fully present.

Prefer this over `chorus` when you want the recognizable Juno-106 / Dimension-D
character. The older `chorus` effect is a digital LFO chorus and is kept for
the pieces that already rely on it.

Parameters:

- `preset: str`
  Supported presets: `juno_i`, `juno_ii`, `juno_i_plus_ii`, `dimension_wide`.
- `mix: float`
  Wet level added to dry (0 - 1). Typical musical range is `0.25 - 0.45`.
  Values above ~0.5 get obvious.
- `rate_hz: float`
  LFO rate in Hz. Juno I is 0.51 Hz, Juno II is 0.83 Hz,
  Dimension-D is ~0.3 Hz.
- `depth_ms: float`
  Peak modulation depth around the base delay.
- `center_delay_ms: float`
  Base delay time. Juno is ~3 - 5 ms; Dimension-D territory is ~10 ms. Must be
  greater than `depth_ms` so the delay stays positive at the LFO trough.
- `cross_feedback: float`
  L -> R and R -> L recirculation (0 - 0.5). Higher values lock the stereo
  field; self-feedback is deliberately avoided.
- `compander_amount: float`
  Gentle wet-path soft-limiting (0 - 1) that mimics BBD I/O companding without
  a full expander pair. 0 is bypass; ~0.2 is the Juno default.
- `pre_lowpass_hz: float`
  Pre-delay input bandlimit (BBD input filter).
- `wet_lowpass_hz: float`
  Post-delay wet lowpass (BBD output filter).
- `wet_highpass_hz: float`
  Removes low-end smear from the wet path.
- `stack_count: int`
  1 or 2. `2` stacks Juno I and II sections with staggered LFO phases for
  denser motion.

Notes:

- Promotes mono input to stereo via the quadrature LFOs. Stereo input is
  preserved per-channel.
- `juno_i` is the safest subtle default; `juno_ii` is wider and faster;
  `juno_i_plus_ii` stacks both; `dimension_wide` is for Dimension-D-style
  longer delays and deeper modulation.
- Reference: Juno-106 service manual (BBD clock rates, chorus parameters)
  plus standard BBD chorus topology. Implemented from algorithmic description;
  no proprietary code was copied.

Example:

```python
score.add_voice(
    "pad",
    synth_defaults={
        "engine": "filtered_stack",
        "preset": "warm_pad",
        "env": {"attack_ms": 400.0, "release_ms": 1400.0},
    },
    effects=[EffectSpec("bbd_chorus", {"preset": "juno_i"})],
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

### `delay`

Implementation: [code_musics/synth.py](code_musics/synth.py)

Simple feedback delay, backed by pedalboard's built-in `Delay`. Useful when
all you need is a short slap or a longer echo with no modulation. For
modulated / filtered repeats prefer `mod_delay`; for stereo cross-feedback
and BBD character prefer `bbd_chorus`.

Parameters:

- `delay_seconds: float`
  Delay time in seconds. Default `0.35`.
- `feedback: float`
  Feedback amount `[0, ~1]`. Default `0.35`. Values near or above `0.95`
  can self-oscillate.
- `mix: float`
  Dry/wet blend `[0, 1]`. Default `0.30`.

Example:

```python
score.add_voice(
    "lead",
    effects=[EffectSpec("delay", {"delay_seconds": 0.42, "feedback": 0.4, "mix": 0.25})],
)
```

### `reverb`

Implementation: [code_musics/synth.py](code_musics/synth.py)

Simple Freeverb-style algorithmic reverb via pedalboard's built-in `Reverb`.
For more realistic tails prefer `bricasti` (convolution) or `dragonfly` /
`tal_reverb2` plugin wrappers.

Parameters:

- `room_size: float`
  Reverb size `[0, 1]`. Default `0.75`.
- `damping: float`
  High-frequency damping `[0, 1]`. Higher values darken the tail. Default
  `0.4`.
- `wet_level: float`
  Wet amount `[0, 1]`. Default `0.25`. Dry level is set internally to
  `1 - wet_level`.

Example:

```python
score = Score(
    f0=110.0,
    master_effects=[EffectSpec("reverb", {"room_size": 0.6, "damping": 0.5, "wet_level": 0.18})],
)
```

### `mod_delay`

Implementation: [code_musics/synth.py](code_musics/synth.py)

Native modulated delay with LFO-modulated read position, lowpass-filtered
feedback, and stereo phase offset — a chorus/delay hybrid that sits in the
longer-delay territory that `bbd_chorus` does not cover. Uses cubic
Hermite interpolation on the fractional delay read to avoid the
high-frequency roll-off naïve linear interpolation gives on continuously
moving taps.

Parameters:

- `preset: str`
  Supported presets: `dream_echo`, `shimmer_slap`, `tape_wander`.
- `delay_ms: float`
  Base delay time (50–500 ms typical musical range). Default `200.0`.
- `mod_rate_hz: float`
  LFO rate. Default `0.2`. Musical range `0.03`–`3.0`.
- `mod_depth_ms: float`
  LFO depth around the base delay. Default `5.0`. Musical range
  `0.5`–`30`.
- `feedback: float`
  Feedback amount `[0, 0.92]` (hard-clipped internally for stability).
  Default `0.35`.
- `feedback_lpf_hz: float`
  Lowpass cutoff inside the feedback loop. Lower values darken the
  repeats further and tame runaway. Default `4000.0`.
- `stereo_offset_deg: float`
  LFO phase offset between left and right channels, producing stereo
  spread from mono input. Default `90.0`.
- `mix: float`
  Dry/wet blend `[0, 1]`. Default `0.30`.

Notes:

- Promotes mono to stereo via the channel phase offset.
- `dream_echo` is the safest longer-delay ambient default;
  `shimmer_slap` is shorter and slap-like; `tape_wander` pushes into
  tape-style wobble.

Example:

```python
score.add_voice(
    "pad",
    effects=[EffectSpec("mod_delay", {"preset": "dream_echo", "mix": 0.25})],
)
```

### `phaser`

Implementation: [code_musics/synth.py](code_musics/synth.py)

Wrapper around ChowDSP's ChowPhaser VST3 stereo phaser. Returns the signal
unchanged (with a warning) when ChowPhaser is not installed.

Parameters:

- `preset: str`
  Optional named preset from the phaser preset table.
- `rate_hz: float`
  LFO rate in Hz. Internally clamped to `[0, 16]`. Default `0.3`.
- `depth: float`
  Modulation depth `[0, 1]`. Default `0.5`. Maps to both the plugin's LFO
  depth (scaled to the plugin's max of 0.95) and its modulation control.
- `feedback: float`
  Feedback amount `[0, 1]`, scaled to the plugin's max of 0.95. Default
  `0.4`.
- `mix: float`
  Dry/wet blend `[0, 1]`. Default `0.35`.

Notes:

- Requires `~/.vst3/ChowPhaser.vst3` (or the stereo variant) to be
  installed; missing plugins fall back to a passthrough with a logged
  warning rather than a crash.

### `gate`

Implementation: [code_musics/synth.py](code_musics/synth.py)

Single-channel-aware noise gate with an attack/hold/release gain envelope
and a 2 ms RMS key-signal smoother. Useful for tightening percussion tails
and silencing sustained noise floors between phrases.

Parameters:

- `threshold_db: float`
  Gate opens when the smoothed level exceeds this. Default `-40.0`.
- `attack_ms: float`
  Ramp time from floor to full gain once the gate opens. Default `0.5`.
  Must be positive.
- `hold_ms: float`
  Minimum time the gate stays open after the signal drops below
  threshold. Default `40.0`. Must be non-negative.
- `release_ms: float`
  Ramp time from full gain back to floor once hold expires. Default
  `20.0`. Must be positive.
- `floor_db: float`
  Attenuation when the gate is fully closed. Default `-80.0` (effectively
  silent).

Notes:

- Stereo signals are gated per channel. For strict L/R link use a
  pre-summing bus or route through a compressor with a peak detector
  instead.
- Render analysis records threshold/hold/release/floor when
  `return_analysis=True` is requested internally.

Example:

```python
score.add_voice(
    "snare",
    effects=[EffectSpec("gate", {"threshold_db": -32.0, "hold_ms": 25.0})],
)
```

### `pan`

Implementation: [code_musics/synth.py](code_musics/synth.py)

Equal-power static pan. Promotes mono input to stereo and places it in
the L/R field.

Parameters:

- `pan: float`
  Pan position `[-1, 1]`. `-1` is full left, `0` is center, `1` is full
  right. Default `0.0`.

Notes:

- Unlike `Voice.pan` (which is a voice-level placement value), `pan` as
  an `EffectSpec` is a real insert effect and can be automated through
  the automation surface. Prefer `Voice.pan` for simple static placement
  and the `pan` effect when you need automation or want to insert pan
  between other effects.
- For per-sample moving pan curves the effect chain uses the internal
  `apply_pan_automation` path — the static `pan` parameter is replaced
  by the automation curve.

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

## Plugin Toolbox

Dedicated effect-kind wrappers for plugins that are not covered by a
named section above. Each wrapper is a thin passthrough to an underlying
VST3, so parameter semantics match the upstream plugin — the entries
below document the Python-side surface exposed via `EffectSpec`. All of
them obey the plugin-missing fallback behavior: if the plugin isn't
installed, the effect is skipped with a loud warning rather than
crashing.

### Valhalla

- `valhalla_supermassive` — Valhalla Supermassive hybrid reverb/delay.
  Good for lush, "supermassive" ambient tails that start to overlap
  with delay territory.
  Parameters: `mix` (0–100, percent), `delay_ms`, `feedback` (0–100),
  `density` (0–100), `width` (0–100), `low_cut`, `high_cut`, `mod_rate`
  (Hz), `mod_depth` (0–100).
- `valhalla_freq_echo` — Valhalla FreqEcho frequency-shifting delay.
  Signature "spiraling" echoes from a frequency shifter in the feedback
  path.
  Parameters: `mix`, `shift`, `delay`, `feedback`, `low_cut`, `high_cut`.
- `valhalla_space_mod` — Valhalla SpaceModulator flanging / modulation.
  Parameters: `mix` (0–100), `rate` (Hz), `depth` (0–100),
  `feedback` (0–100).

### Compressors / leveling

- `tdr_kotelnikov` — TDR Kotelnikov transparent mastering compressor.
  Parameters: `threshold_db`, `ratio`, `attack_ms`, `release_rms_ms`,
  `makeup_db`, `soft_knee_db`, `peak_crest_db`, `dry_wet` (0 = fully
  compressed wet).
- `mjuc_jr` — MJUCjr vari-mu compressor (Klanghelm).
  Parameters: `compress` (0–48 dB), `makeup` (0–48 dB),
  `timing` (`"slow"` / `"fast"`).
- `fetish` — FETish 1176-style FET compressor.
  Parameters: `input_db`, `output_db`, `attack_us` (microseconds!),
  `release_ms`, `ratio`, `mix` (0–100), `hpf_hz` (sidechain).
- `lala` — LALA LA-2A-style optical compressor.
  Parameters: `gain` (0–100), `peak_reduction` (0–100), `hf` (0–100),
  `mode` (1.0 = compress, other values = limit).
- `kolin` — Kolin SSL-style bus compressor.
  Parameters: `input_db`, `output_db`, `attack_ms` (1–50),
  `release_ms` (100–3000), `mix` (0–100), `hpf_hz` (sidechain).
- `laea` — LAEA LA-3A-style leveling amplifier.
  Parameters: `gain` (0–100), `reduction` (0–100), `mix` (0–100),
  `limit` (bool — True for limit mode, False for compress).
- `britpressor` — Britpressor Neve 2254-style compressor/limiter.
  Parameters: `compressor_threshold` (−20 to 10 dB),
  `ratio` (discrete string, e.g. `"3:1"`), `gain` (0–20),
  `mix` (23-step string like `"0/100"`), `compressor_recovery_time`
  (discrete string, `"100ms"` … `"Auto-2"`),
  `limit_level` (4–15), `level_recovery_time` (discrete string),
  `high` (−6..6, 3 dB steps), `mid` (−6..6, 3 dB steps),
  `high_pass_filter` (discrete string, `"OFF"` … `"360Hz"`).

### EQ / channel

- `brit_channel` — BritChannel Neve 1073-style channel strip.
  Parameters: `preamp_gain_db` (−24…24), `output_trim_db` (−24…24),
  `highpass` / `low_freq` / `mid_freq` (discrete strings),
  `low_gain_db` / `mid_gain_db` / `high_gain_db` (−15…15).
- `brit_pre` — BritPre Neve-style preamp.
  Parameters: `gain` (−20…40, 5 dB steps), `output_db` (−24…24),
  `highpass_filter` / `lowpass_filter` (discrete strings).
- `merica` — MERICA American-style 3-band EQ.
  Parameters: `low_gain_db` / `mid_gain_db` / `high_gain_db` (−12…12),
  `low_freq` / `mid_freq` / `high_freq` (discrete Hz values),
  `input_db`, `output_db`.
- `rare_se` — RareSE Pultec-style passive EQ (L/M section).
  Parameters: `low_boost` (0–10), `low_atten` (0–10),
  `high_boost` (0–10), `high_atten` (0–10),
  `high_bandwidth` (1–10), `output_db`, `low_frequency` (20–100 Hz),
  `high_frequency` (3000–16000 Hz), `high_atten_frequency` (up to
  20000 Hz).

### Drive / distortion / coloring

- `ivgi` — IVGI saturation/distortion (Klanghelm).
  Parameters: `drive` (0–10), `trim` (0–10), `output` (0–10),
  `asymmetry` (0–10), `freq_response` (0–10).
- `distox` — Distox multi-mode distortion.
  Parameters: `input_db` (−30…30), `output_db` (−30…30),
  `mix` (0–100), `hpf_hz` (5–2000), `lpf_khz` (10–20),
  `mode` (discrete string, e.g. `"Op-Amp 1"`, `"Tube 3"`).
- `fet_drive` — FetDrive FET-style saturation.
  Parameters: `drive_db` (0–50), `tone` (0–100),
  `output_db` (−15…15), `mix` (0–100).
- `prebox` — PreBOX preamp saturation.
  Parameters: `input_db` (−24…24), `output_db` (−24…24),
  `model` (0–10, 11 discrete models), `hpf` (0/1/2/3),
  `lpf` (0/1), `agc` (`"AGC Off"` / `"AGC On"`).
- `tuba` — TUBA tube amplifier.
  Parameters: `level` (1–20), `output_db` (−96…12),
  `gain` (`"Low Gain"` / `"High Gain"`), `high_gain`
  (discrete −6/0/3/6), `low_gain` (discrete −6/0/6).

Notes:

- These wrappers live alongside the named-section plugins (`chow_tape`,
  `byod`, `chow_centaur`, `airwindows`, `tal_chorus_lx`, `tal_reverb2`,
  `bricasti`, `dragonfly`) in the same `_SIMPLE_EFFECT_DISPATCH` table.
  Pick whichever is installed.
- Plugin availability is a per-machine question — see the
  Linux/macOS palette notes in `AGENTS.md`. On Linux the Analog
  Obsession / Acustica-style plugins are the commonly-installed set;
  on macOS the ChowDSP / TAL / Dragonfly stack is broader.
- Each wrapper passes parameters straight through to the plugin via
  `_apply_plugin_processor` — the Python-side param names above match
  the VST3 parameter names (with underscoring / lowercasing applied by
  pedalboard).

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
| `bit_crush` | Digital quantization to `bit_depth` bits (1.0-16.0). Machinedrum-flavored crunch. |
| `rate_reduce` | Integer sample-and-hold at `reduce_ratio` (>= 1.0). Lo-fi digital aliasing. |
| `digital_clip` | Asymmetric hard clip with small positive bias. Harsh DX-style clipping. |

Drive follows the standard knob convention: 0-0.25 subtle, 0.33-0.66
moderate, 0.66-1.0 strong.

Most algorithms include first-order ADAA (Anti-Derivative Anti-Aliasing)
for reduced aliasing at no significant CPU cost. For algorithms where a
closed-form antiderivative is not available (`foldback`, `linear_fold`,
`sine_fold`, `half_wave_rect`), optional 2x oversampling is available via
the `oversample=2` parameter. The `apply_waveshaper` function accepts an
optional `drive_envelope` (per-sample array in [0, 1]) for time-varying
drive modulation. The three digital-character algorithms (`bit_crush`,
`rate_reduce`, `digital_clip`) use 2x oversampling rather than ADAA since
their transfer functions are piecewise-constant / discontinuous; they read
the extra scalar params `bit_depth` and `reduce_ratio` from
`apply_waveshaper`'s keyword arguments.

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
- `noise_mode: str` (default `"white"`)
  Base noise source for the partial noise bands.  Choices:
  - `"white"`: Gaussian white noise (smooth, woolly).
  - `"flow"`: Mutable Instruments Elements-style "Flow" rare-event
    sample-and-hold exciter.  Produces breath- or brush-like texture
    (organic, granular character that white noise and plain S&H cannot
    match).  RMS-normalized so switching modes preserves apparent
    loudness.
- `flow_density: float` (0-1, default 0.5)
  Only used when `noise_mode="flow"`.  Controls the density of flow
  events: low values (~0.05-0.2) yield very sparse, exhale-like gestures;
  mid values (~0.4-0.6) give an audible brush; high values (>0.8) approach
  uniform granular noise.
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

Vital-style spectral morphs (re-implemented from algorithmic description; no
verbatim GPL-3 code). All morphs operate on the per-partial `(ratio, amp)`
array *before* resynthesis. Defaults leave the spectrum unchanged, so existing
presets render bit-identically without explicit opt-in.

- `spectral_morph_type: str` (default `"none"`)
  One of `"none"`, `"inharmonic_scale"`, `"phase_disperse"`, `"smear"`,
  `"shepard"`, `"random_amplitudes"`. Unknown values raise `ValueError` at
  render time (fail-fast).
- `spectral_morph_amount: float` (default `0.0`)
  Strength of the morph. Reduces to identity at `0.0`. Clamped to `[0, 1]`
  for `smear`, `shepard`, and `random_amplitudes`; unclamped for
  `inharmonic_scale` (negative values compress, positive stretch) and
  `phase_disperse` (typical range 0-0.05).
- `spectral_morph_shift: float` (default `0.0`)
  Used by `shepard` as an octave shift for the ghost copy and by
  `random_amplitudes` as a 0-1 position scrolling through a 16-stage
  interpolated random mask. Ignored by the other morph types.
- `spectral_morph_center_k: int` (default `24`)
  Used by `phase_disperse` to select the partial index where the quadratic
  phase offset is zero. Vital's default is 24; small harmonic banks usually
  want values in the 2-8 range.
- `spectral_morph_seed: int` (default `0`)
  Used by `random_amplitudes` to select a stable random mask sequence.
  Deterministic under fixed seed.
- `sigma_approximation: bool` (default `False`)
  When `True`, multiplies each partial's amplitude by `sinc(k / (K + 1))`
  (Lanczos sigma factors) before resynthesis. Reduces Gibbs ringing from
  hard band-limiting. Strictly better than hard truncation; cheap to enable.

Morph semantics:

- **`inharmonic_scale`** (piano-stiffness / inharmonic drift)
  `new_ratio[k] = ratio[k] * (1 + amount * log2(k) / log2(k_max))`.
  The fundamental is unaffected; higher partials shift progressively more.
  Small positive amounts (0.05-0.2) give piano-plate-like stretch. Larger
  amounts drift toward bell/gong inharmonicity.
- **`phase_disperse`** (Vital pad width without chorus)
  `phase[k] += sin((k - center_k)^2 * amount) * 2*pi`. Quadratic phase
  offsets across partials create a "spread" waveform with unchanged
  magnitude spectrum. Typical amounts 0.005-0.03.
- **`smear`** (pink-shifted amplitude spread)
  `amp[k+1] = (1-amount)*amp[k+1] + amount*amp[k]*(1 + 0.25/k)`. A
  first-order running mixer that leaks amplitude into upper partials;
  creates a softly pink drift of overtones while preserving ratios.
- **`shepard`** (octave-ghost crossfade)
  Each partial blends toward the partner at `ratio * 2^shift` (amplitude
  only in v1). A log-distance gate (~0.25 octaves) prevents wild jumps
  when no close partner exists. Useful for gently pushing a timbre toward
  or away from its octave twin.
- **`random_amplitudes`** (Vital-style stable random mask)
  Generates 16 seeded random amplitude vectors under `seed` and
  interpolates between two adjacent stages from `shift in [0, 1]`
  (wrapped circularly). Deterministic under fixed `seed`; sweep `shift`
  via automation for evolving timbres.

Notes:

- Omitting the new parameters preserves the old additive behavior closely.
- If `partials` is omitted, the engine still uses the legacy harmonic ladder
  generated from `n_harmonics`, `harmonic_rolloff`, `brightness_tilt`, and
  `odd_even_balance`.
- `odd_even_balance` is clamped internally to avoid zeroing the spectrum too aggressively.
- `attack_partials` only does anything when paired with `spectral_morph_time > 0`.
- Explicit spectral ratios are relative to the resolved note frequency, not to
  `Score.f0`.
- When a spectral morph is set and `attack_partials` is also provided, the
  morph is applied to *both* the sustain and attack partial sets using the
  same parameters before morph-time crossfading. Ratio-changing morphs
  (`inharmonic_scale`, `shepard`) therefore preserve the onset/sustain
  relationship but shift both spectra together.

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
- `brush_breath` — 7-limit JI support + sparse "Flow" S&H exciter, exhale-like
- `brush_cymbal` — inharmonic bar partials + dense "Flow" exciter, brushed-cymbal shimmer
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

Spectral-morph demo presets (showcase each morph type at musical settings):

- `stiff_piano` — `inharmonic_scale` at 0.08 for piano-like string stiffness
- `dispersed_pad` — `phase_disperse` at 0.02 center_k=4 for Vital-style pad width
- `smear_drone` — `smear` at 0.55 pushing harmonic energy into overtones
- `shepard_bells` — `shepard` at 0.4 shift=1.0 for octave-ghost crossfade
- `chaos_cloud` — `random_amplitudes` at 0.7 with sigma-approximated spectrum

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

## Spectral Builders API

Implementation: [code_musics/spectra.py](code_musics/spectra.py)

`code_musics.spectra` exposes helpers for building the `partials` /
`attack_partials` lists consumed by the `additive` engine and the `modal`
mode tables consumed by `drum_voice`. Every builder returns a standard
shape: a list of `{"ratio": float > 0, "amp": float >= 0}` dicts, sorted
by ratio. Mode-table helpers return raw `list[float]` ratio tables.

See also: the additive engine's `partials` / `attack_partials` /
`spectral_morph_type` params, and the `drum_voice` engine's
`tone_type="modal"` / `metallic_type="modal_bank"` paths which consume
these builders.

### Mode tables

- `get_mode_table(name) -> list[float]`
  Returns a copy of a named physical-model ratio table. Valid names:
  `"membrane"`, `"bar_wood"`, `"bar_metal"`, `"bar_glass"`, `"plate"`,
  `"bowl"`, `"stopped_pipe"`. The three `bar_*` variants share the same
  Euler-Bernoulli ratio table — the material name is documentation for
  the caller (damping / decay is a property of the consumer, not the
  table). Passing `"custom"` is not allowed — supply ratios directly.

### Core builders

- `ratio_spectrum(ratios, amps=None) -> list[dict]`
  Build an explicit additive spectrum from ratio and optional amplitude
  lists. Amplitudes default to 1.0. Raises if any ratio is ≤ 0 or any
  amp < 0, or if the lengths do not match.
- `harmonic_spectrum(*, n_partials, harmonic_rolloff=0.5, brightness_tilt=0.0, odd_even_balance=0.0) -> list[dict]`
  Harmonic-series spectrum matching the additive engine's legacy
  `n_harmonics` / `harmonic_rolloff` / `brightness_tilt` /
  `odd_even_balance` semantics. Use this when you want the default
  harmonic stack but wrapped as an explicit `partials` list so you can
  post-process it (morphs, gravity, formant shaping).
  `n_partials >= 1`; `odd_even_balance` clamped internally to
  `[-0.95, 0.95]`.
- `stretched_spectrum(*, n_partials, stretch_exponent, harmonic_rolloff=0.5, brightness_tilt=0.0) -> list[dict]`
  Stretched / compressed overtone family: ratios follow
  `partial_index ** stretch_exponent`. `stretch_exponent > 1` stretches
  (piano-like), `< 1` compresses. Raises if `stretch_exponent <= 0`.

### Physical-model spectra

- `membrane_spectrum(*, n_modes=12, damping=0.3) -> list[dict]`
  Circular drumhead modes from Bessel zeros (normalized to the
  fundamental). `damping` drives exponential amplitude rolloff
  `exp(-damping * i)`. `n_modes` capped at 16 (the table's length).
- `bar_spectrum(*, n_modes=8, material="wood") -> list[dict]`
  Euler-Bernoulli free-free bar modes (marimba/xylophone). `material`
  chooses damping: `"wood"` (fastest decay), `"metal"` (slowest),
  `"glass"` (moderate). `n_modes` capped at 8.
- `plate_spectrum(*, n_modes=12, aspect_ratio=1.0) -> list[dict]`
  Rectangular plate modes from `f_{m,n} = C * (m^2/a^2 + n^2/b^2)`.
  `aspect_ratio=1.0` (square) is maximally degenerate; other values
  break degeneracy for richer spectra. Amplitudes are `1/(m*n)`.
- `tube_spectrum(*, n_modes=8, open_ends="both") -> list[dict]`
  Cylindrical tube modes. `open_ends="both"` (flute-like) gives
  `[1, 2, 3, ...]`; `"one"` (clarinet-like) gives odd-only
  `[1, 3, 5, ...]`; `"neither"` (stopped pipe) uses the slightly shifted
  tabulated ratios. Amplitude rolloff `1 / (1 + 0.3 * i)`.
- `bowl_spectrum(*, n_modes=8) -> list[dict]`
  Singing bowl / Tibetan bowl modes from published acoustics
  measurements. Amplitude rolloff `1 / (1 + 0.5 * i)`. `n_modes` capped
  at 8.

### Combinators

- `spectral_convolve(spec_a, spec_b, *, max_partials=32, merge_tolerance_cents=10.0, min_amp_db=-60.0) -> list[dict]`
  Cross-product of two partial lists — every pair `(a, b)` produces a
  product partial at `a.ratio * b.ratio` with amp `a.amp * b.amp`.
  Near-coincident partials (within `merge_tolerance_cents`) are merged
  via amplitude-weighted geometric-mean ratio and summed amplitudes.
  Result is pruned below `min_amp_db` relative to peak, capped at
  `max_partials`, and peak-normalized to 1.0. Musically useful for JI
  spectra because cross-products of JI ratios stay in the ratio family.
- `fractal_spectrum(seed, *, depth=2, level_rolloff=0.5, max_partials=32) -> list[dict]`
  Self-similar spectrum via iterated self-convolution. Each level
  convolves `seed` with the previous level and scales amplitudes by
  `level_rolloff ** level`. All levels merge into a single
  peak-normalized spectrum capped at `max_partials`. `depth=0` returns
  a normalized copy of the seed.

### Formant shaping

- `vowel_formants(name) -> list[(center_hz, gain, bandwidth_hz)]`
  Return named vowel formant data. Valid names: `"a"`, `"e"`, `"i"`,
  `"o"`, `"u"`. The returned list is suitable for `formant_shape` and
  `formant_morph`.
- `formant_weight(abs_freq, formants) -> float`
  Sum of Gaussian resonance peaks at `abs_freq` for the given formant
  list. Useful when computing per-partial amplitudes manually or for
  custom formant envelopes.
- `formant_shape(partials, f0, formants, *, bandwidth_hz=100.0) -> list[dict]`
  Shape an existing partial spectrum through formant resonance peaks at
  absolute frequencies (via `partial.ratio * f0`). `formants` is either
  a vowel name string (looked up via `vowel_formants`) or an explicit
  list of `(center_hz, gain, bandwidth_hz)` tuples. When a string is
  passed, per-formant bandwidths are overridden by `bandwidth_hz`.
- `formant_morph(partials, f0, vowel_sequence, morph_times=None) -> list[dict]`
  Generate per-partial envelopes for time-varying vowel morphs. Each
  returned partial carries an `"envelope"` key with `{time, value}`
  keyframes suitable for the additive engine's per-partial envelope
  feature. `morph_times` defaults to evenly spaced `[0, 1/(N-1), …, 1]`
  values. Note that partial amplitudes are preserved — the envelopes
  modulate them multiplicatively.

## Filter primitives

Reusable DSP primitives from `code_musics.engines._filters` that
engine authors can call directly. Keeping them listed here so future
engines can reuse proven primitives instead of reimplementing.

### `apply_comb`

Resonant comb filter with per-sample delay and damped feedback. Used by
the `va` engine's comb slot; available for any future engine that wants
karplus-strong-ish resonance, flange-like motion, or tuned resonator
coloring.

Signature:

```python
apply_comb(
    signal: np.ndarray,               # 1-D input
    *,
    delay_samples_profile: np.ndarray,  # per-sample delay length in samples
    feedback: float,                    # [0, 0.99]; soft-clipped internally
    damping: float,                     # [0, 1]; feedback-path 1-pole LPF
    mix: float,                         # [0, 1]; dry/wet balance
    sample_rate: int,
) -> np.ndarray
```

Parameter semantics:

- `delay_samples_profile` — 1-D array, must match `signal` length.
  Values `< 1.0` clamp to 1.0; values above one second's worth of
  samples clamp to `sample_rate`. A frequency-tracked delay
  (`sample_rate / freq`) produces karplus-strong-ish bell tones.
- `feedback` — `[0, 0.99]`. Above ~0.95 with low damping the loop
  self-oscillates. Soft-clipped internally so runaway values fail
  gracefully rather than exploding.
- `damping` — `[0, 1]`. 0 is bright sustained resonance; higher values
  progressively darken and shorten the decay. Internally clamped below
  1.0 so the feedback-path 1-pole always receives a nonzero fraction
  of the delayed signal — damping=1.0 approaches fully muted feedback
  without freezing state.
- `mix` — `[0, 1]`. 0 returns the input unchanged; 1 returns pure wet.
- `sample_rate` — Samples per second. Sets the hard cap on delay
  length (one second).

Intended use:

- Pair with `delay_samples_profile = sample_rate / freq_profile` for a
  pitched comb resonator (bell / karplus-strong / string-in-pipe
  character).
- Use a static `delay_samples_profile` for a fixed-frequency comb
  coloration.

Raises `ValueError` when `feedback`, `damping`, or `mix` fall outside
their ranges, when `delay_samples_profile` length does not match
`signal`, or when `signal` is not 1-D.

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
| `tap_acceleration` | `float` | `0.0` | Tap spacing compression (0–1); higher values make later taps closer together, creating a "roll" effect |
| `tap_freq_spread` | `float` | `0.0` | Per-tap frequency jitter (0–1); randomizes each tap's bandpass center for a more natural, scattered sound |
| `tail_filter_cutoff_hz` | `float` | `None` | Optional lowpass cutoff applied to the noise body tail; useful for darker, warmer clap tails |
| `tail_filter_q` | `float` | `1.5` | Q for the tail lowpass filter (>= 0.5) |

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
| `hand_clap` | Natural 4-tap hand clap with wider spacing, softer click, and tap frequency spread |
| `909_clap_authentic` | 909-accurate timing with tap acceleration and frequency spread |
| `scattered_clap` | Wide 6-tap clap with heavy acceleration and spread for organic character |
| `granular_cascade` | Dense 8-tap granular burst with high acceleration and spread |
| `micro_burst` | Tight 6-tap burst with subtle spread, very short body |

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

The oscillator imperfections (`osc_asymmetry`, `osc_softness`, `osc_dc_offset`,
`osc_shape_drift`), voice card spread (`voice_card_spread`), filter topology
(`filter_topology`, `bass_compensation`), filter mode morphing (`filter_morph`),
serial HPF (`hpf_cutoff_hz`, `hpf_resonance_q`), feedback path
(`feedback_amount`, `feedback_saturation`), and VCA nonlinearity
(`vca_nonlinearity`) documented under Analog Character (polyblep) are also
available on this engine. See that section for full parameter details.

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
| `oscillator_mode` | `str` | `"sine"` | Partial oscillator mode: `"sine"` for standard additive, `"square"` for bandlimited square partials (808-style digital metallic character) |
| `noise_amount` | `float` | `0.0` | Level of a broadband noise layer mixed into the output (0–1); adds sizzle and air to the metallic sound |

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
- `oscillator_mode="square"` uses bandlimited PolyBLEP square waves instead of
  sines, giving a harsher, more digital character similar to the 808 hi-hat circuit

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
| `harmonic_bell` | Harmonic integer-ratio partials for tuned bell character |
| `septimal_bell` | 7-limit JI partial ratios for xenharmonic bell |
| `square_gamelan` | Square-wave partials with gamelan-inspired inharmonic ratios |
| `beating_hat_a` / `b` / `c` | Near-unison partial pairs for beating interference patterns |
| `swept_hat` | Hi-hat with filter sweep via filter envelope |
| `decaying_bell` | Long bell with multi-point amplitude envelope |
| `808_closed_hat` | 808-style square-wave closed hat with fixed partial ratios |
| `808_open_hat` | 808-style square-wave open hat |
| `808_cowbell_square` | 808-style square-wave cowbell |

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

- `body_mode: str`
  Body synthesis mode. `"oscillator"` (default) uses a waveform oscillator with
  optional FM; `"resonator"` drives a time-varying biquad bandpass resonator from
  the click transient (or a short noise impulse when the click is absent).
  Resonator mode ignores FM body params.
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
- resonator mode (`body_mode="resonator"`) is driven by the click transient;
  the exciter energy passes through a time-varying biquad tuned to the body
  pitch, producing a more naturally resonant, less synthetic body character

Presets:

| Preset | Character |
|--------|-----------|
| `808_hiphop` | Long 808-style sub kick with deep sweep |
| `808_house` | Medium 808 with moderate sweep and punch |
| `808_tape` | 808 with sine_clip body for tape-saturated warmth |
| `909_techno` | Hard 909-style kick with triangle body and strong click |
| `909_house` | Clean 909 house kick with sharp attack |
| `909_crunch` | 909 with sine_clip body and extra click emphasis |
| `distorted_hardkick` | Aggressive hard kick with heavy click and sine_clip |
| `zap_kick` | Wide pitch sweep (5x) for zappy electronic kicks |
| `round_tom` | Gentle tom with moderate sweep and triangle body |
| `floor_tom` | Deep floor tom with longer decay |
| `electro_tom` | Punchy electronic tom |
| `ring_tom` | Tom with prominent ring/overtone character |
| `gated_808` | 808 with multi-point body envelope for gated decay |
| `pitch_dive` | Extreme pitch sweep (6x) for dramatic dive effect |
| `filtered_kick` | Kick with envelope-modulated lowpass voice filter |
| `fm_body_kick` | FM body synthesis for harmonically rich attack |
| `foldback_kick` | Foldback waveshaper on the body for harmonic density |
| `808_resonant` | Resonator body mode driven by click excitation |
| `808_resonant_long` | Long-decay resonator with wider pitch sweep |
| `resonant_tom` | Resonator body for naturally resonant tom character |
| `melodic_resonator` | Long resonator body tuned for melodic percussion |
| `kick_bell` | Kick with metallic partials layer for bell overtones |

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
- `osc2_sync: bool`
  Hard sync: when `True`, osc2's phase is reset to 0 every time osc1 wraps past
  1.0, with PolyBLEP step corrections injected at each sync event to suppress
  aliasing. Produces the classic chirpy "sawtooth-with-formant-resonance" timbre
  whose perceived pitch follows osc1 while the formant moves with osc2's
  detune. Only supported when `osc2_waveform="saw"`; raises `ValueError`
  otherwise. Default `False`.
- `osc2_ring_mod: float`
  Dry/ring blend between the normal osc1+osc2 mix and `osc1 * osc2`. `0.0`
  (default) preserves the original dry sum
  `(osc1 + osc2_level*osc2) / (1 + osc2_level)` bit-for-bit. `1.0` outputs pure
  ring modulation `osc1 * osc2` — osc1's fundamental is suppressed and the
  spectrum contains sum/difference sidebands of the two oscillator frequencies.
  Intermediate values mix both paths. Aliasing is bounded for musical detune
  ratios; at extreme settings use `quality="fast"` or higher to bring in
  oversampling. Range `[0.0, 1.0]`.

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

#### Oscillator Imperfections

- `osc_asymmetry: float`
  Saw wave reset slope softening. Blends toward a softer reset shape. At 0:
  perfect digital saw. At 0.3: subtle analog warmth. For square waves,
  introduces slight duty cycle asymmetry. Default `0.0`. Range `[0, 1]`.
- `osc_softness: float`
  Bandwidth limiting that models capacitor-limited VCO slew rate. Reduces higher
  harmonics with a frequency-tracking one-pole lowpass. At 0: full harmonics.
  At 0.3: gentle warmth. At 1.0: almost sinusoidal. Default `0.0`. Range `[0, 1]`.
- `osc_dc_offset: float`
  Per-oscillator DC bias. Deterministic sign per voice card. When hitting a
  driven filter, produces even harmonics (the same mechanism as in real VCOs).
  Removed downstream by DC blocking. Default `0.0`. Range `[0, 1]`.
- `osc_shape_drift: float`
  Slow waveform shape modulation via Ornstein-Uhlenbeck process. Models thermal
  drift in VCO capacitors. For square waves, modulates effective pulse width;
  for saws, modulates the asymmetry blend. Default `0.0`. Range `[0, 1]`.

#### Voice Card Spread

- `voice_card_spread: float`
  Controls inter-voice calibration variation. Replaces the old `voice_card`
  parameter (backward compatible). `0.0` = identical voices, `1.0` =
  conservative (JI-safe), `2.0` = Roland/Yamaha level, `3.0` = Oberheim Four
  Voice level. Scales all per-voice offsets: pitch, cutoff, amplitude, attack
  time, release time, pulse width, resonance, softness, and drift rate. Default
  `1.0`. Range `[0, 3]`.

  When `voice_card_spread` (or `voice_card_envelope_spread`) is explicitly
  present on a voice, the outer Score-level ADSR attack and release times are
  also scaled per-voice (±5% at spread=1.0, linearly wider at higher spread).
  Voices that never set the knob keep their exact authored attack/release —
  this preserves fixed-length render semantics for older pieces and length-
  sensitive tests; opt in by setting `voice_card_spread` to see per-voice env
  drift in addition to the per-sample parameter offsets.

#### OB-Xd-Style Fast CV Dither

  The `analog_jitter` knob also drives an audio-rate OB-Xd-style dither layer
  on top of the stable `voice_card` offsets (polyblep + filtered_stack
  engines). At `analog_jitter=1.0` the dither is ±0.05 semitone on pitch and
  ±3% on filter cutoff, smoothed by a 4 kHz one-pole lowpass so it reads as
  warm rustle rather than pure hiss. FM engine skips the cutoff path (no
  filter) but is otherwise unaffected — `analog_jitter=0` disables dither
  entirely, preserving prior behaviour.

#### Filter Topology

- `filter_topology: str`
  Selects filter architecture. Eight options:
  - `"svf"` — 2-pole (12 dB/oct) ZDF state-variable filter. Default.
  - `"ladder"` — 4-pole (24 dB/oct) Moog-style ladder with per-stage saturation.
  - `"sallen_key"` — 2-pole Sallen-Key-flavored ZDF SVF with narrower resonance
    peak and pre-filter asymmetric soft-clip under drive, producing the biting
    CEM-3320 character of a Diva-style "Bite" filter. The underlying DSP is
    the same TPT 2-pole SVF math as `"svf"`, with a tighter Q-to-damping
    mapping and an asymmetric-tanh input stage that adds even-harmonic bias
    when `filter_drive > 0`. Literal-SK cross-sample positive-feedback
    topologies have state-eigenvalue stability issues at moderate Q even
    when the instantaneous ZDF solve is well-conditioned; this approach
    preserves the full ZDF/TPT benefits without that risk.
  - `"cascade"` — 4-pole (24 dB/oct) cascade of four independent ZDF 1-poles
    plus a separate peaking bandpass at cutoff. No global feedback loop, so
    the character is smoother and less growly than the Moog ladder — closer
    to a Prophet-5 rev-2 or Juno VCF. Supports the same 4→3→2→1-pole morph
    as the ladder via `filter_morph`.
  - `"sem"` — 2-pole (12 dB/oct) Oberheim SEM-flavored ZDF SVF. Wider
    Q-to-damping curve than `"svf"` so resonance is gentler and the peak
    "blooms" rather than spikes; per-integrator `_algebraic_sat` cap
    models OTA saturation without ladder-style growl. Bass is preserved
    through high Q (no sag). Supports a 3-stage `filter_morph ∈ [0, 2]`
    that sweeps continuous LP→Notch→HP — the signature SEM "one knob"
    morph (BP is available via `filter_mode="bandpass"` but sits outside
    the continuous sweep, matching the real hardware).
  - `"jupiter"` — 4-pole (24 dB/oct) OTA cascade modeling the Roland
    IR3109. A single global tanh on the feedback summation (NOT per-stage
    as in Moog ladder) plus a softer Q→k mapping (saturates at ≈2.6 vs
    ladder's ~4.0) yield the creamy Jupiter-8 character — minimal bass
    suction at high Q, cleaner self-oscillation than Moog, and a less
    peaky resonance shape. Pair with `hpf_cutoff_hz > 0` for the
    Jupiter-8 dual-filter architecture; leave the HPF at 0 for a
    Juno-106. Supports `filter_morph ∈ [0, 3]` (24→18→12→6 dB/oct
    pole-tap blend) and both `"adaa"` and `"newton"` solvers.
  - `"k35"` — Korg MS-20 (Korg35) Sallen-Key with diode-clipped
    feedback. Two TPT 1-poles inside a positive-feedback resonance
    loop resolved with the alpha-compensation closed form (no Newton
    needed). Feedback path is shaped by `_diode_shape(y_prev,
    k35_feedback_asymmetry)` — a Shockley-like asymmetric shaper that
    produces the defining MS-20 "snarl" of even-harmonic bias rising
    with Q. `filter_drive > 0` engages an additional asymmetric
    input-stage soft-clip for the classic MS-20 crunch (the real unit
    overloads easily). LP and HP modes; BP/notch coerce to LP. No
    `filter_morph` support (no pole taps to blend between).
  - `"diode"` — 3-pole (18 dB/oct native) diode ladder modeling the
    TB-303. Feedback tap between stages 2 and 3 (from state `s2`, not
    from the output) is the defining topological quirk versus Moog —
    it produces the bass-suck + squelch as Q rises, not a Moog-style
    growl. Feedback shaped by `_diode_shape(s2, asym)` with asymmetry
    scaling with drive, so cranking `filter_drive` amplifies the
    acid-bark character rather than just compressing. LP only (other
    modes coerce). Supports `filter_morph ∈ [0, 2]` (18→12→6 dB/oct
    pole-tap blend) and both `"adaa"` and `"newton"` solvers — use
    `"newton"` for most authentic high-Q squelch.
- `bass_compensation: float`
  Ladder only. Restores low-frequency energy lost to resonance feedback. At 0:
  classic Moog behavior (bass loss at high resonance). At 1.0: full bass
  restoration. Based on Rossum's approach in the Subsequent 37. Default `0.5`
  (moderate bass preservation — most modern Moog-style usage wants this; drop
  to `0.0` for authentic vintage bass-suck, e.g. acid bass patches). Range
  `[0, 1]`. Only affects the `ladder` topology; ignored elsewhere.
- `k35_feedback_asymmetry: float`
  K35 only. Controls the even-harmonic bias of the diode feedback path.
  At `0.0`: symmetric diode (cleaner K35). At `0.5`: classic MS-20 snarl.
  At `1.0`: deranged. Default `0.0`. Range `[0, 1]`. Ignored by non-k35
  topologies.

#### Filter Solver

- `filter_solver: str`
  Direct control over which solver the topology uses for its delay-free
  feedback loop. Accepted values are `"newton"` (default) and `"adaa"`.
  Most users should leave this at the default and pick a `quality` preset
  instead — `quality` sets `filter_solver` plus Newton iteration counts,
  tolerance, and oversampling together. Set `filter_solver` explicitly
  when you want to force the legacy one-step-delay ADAA character on a
  single voice without dropping the rest of the quality stack.
  Applies to `"ladder"`, `"jupiter"`, and `"diode"` topologies. Ignored
  by topologies that do not have a per-sample implicit feedback solve
  (`"svf"`, `"sallen_key"`, `"cascade"`, `"sem"`, `"k35"` — the K35
  uses a closed-form alpha-compensation path and does not need Newton).

#### Quality Modes

- `quality: str`
  Engine-level quality control for the filter solver and internal oversampling.
  Default `"great"`. Raising quality gives more accurate feedback behavior
  and less aliasing under drive, at linearly proportional CPU cost. Acts as
  a single-knob preset over `filter_solver`, `max_newton_iters`,
  `newton_tolerance`, and the per-engine oversampling factor.

  | mode | solver | Newton iters | tol | oversample |
  |-|-|-|-|-|
  | `"draft"` | ADAA (one-step-delay tanh, as pre-2026-overhaul) | — | — | 1x |
  | `"fast"` | Newton | 2 | 1e-8 | 2x |
  | `"great"` | Newton | 4 | 1e-9 | 2x |
  | `"divine"` | Newton | 8 | 1e-10 | 4x |

  The Newton solver resolves the filter's delay-free feedback loop implicitly
  at the current sample (Diva/Zavalishin-style), rather than using the prior
  sample's state as the feedback input. This is audible at high resonance
  (cleaner self-oscillation onset), high drive (less intermodulation
  between drive nonlinearity and resonance), and fast cutoff modulation.
  Topologies without an implicit solve path (see `filter_solver`) ignore
  the solver selection; oversampling still applies.

  Newton mode also closes the **external feedback loop** (`feedback_amount`)
  implicitly on six topologies: `svf`, `cascade`, `sem`, `sallen_key`,
  `ladder`, `jupiter`. The scalar residual uses the shared
  `_solve_ext_feedback_newton` helper (linear-body topologies) or a
  combined internal-plus-outer-tanh residual (`ladder`, `jupiter`).
  Without this, the `feedback_amount` path used a one-sample delay that
  damped high-resonance behaviour and smeared fast transients. Remaining
  unit-delay hold-outs (see `FUTURE.md`): `k35`, `diode`, and the driven
  SVF path (`filter_drive > 0`).

  The ladder k-mapping differs per solver so both reach self-oscillation at
  appropriate user-facing Q values: `k_adaa = min(3.98, 4(1 - 1/(2q)))`,
  `k_newton = min(4.25, 4.2(1 - 1/(2q)))`.

#### Filter Mode Morphing

- `filter_morph: float`
  Continuous blend between filter modes. Semantics depend on topology:
  - `"svf"`: cycles LP → BP → HP → Notch → LP over `[0, 3]`.
  - `"ladder"`, `"cascade"`, `"jupiter"`: pole-tap blend (4→3→2→1 pole)
    over `[0, 3]`, giving continuous slope control 24→18→12→6 dB/oct.
  - `"diode"`: pole-tap blend (3→2→1 pole) over `[0, 2]`, giving
    18→12→6 dB/oct. Clamped at 2.
  - `"sem"`: 3-stage LP → Notch → HP sweep over `[0, 2]` — the SEM's
    signature single-knob morph (BP is available via `filter_mode` but
    sits outside the continuous sweep).
  - `"sallen_key"`, `"k35"`: no morph support (no pole taps or morph
    path to blend between). `filter_morph` is silently ignored.
  Automatable. Default `0.0`.

#### Serial Highpass Filter

- `hpf_cutoff_hz: float`
  When > 0, adds a 2-pole ZDF SVF highpass before the main filter. Models
  CS80/Jupiter-8 dual-filter architecture. Default `0.0`.
- `hpf_resonance_q: float`
  Resonance for the serial HPF. Default `0.707`.

#### Feedback Path

- `feedback_amount: float`
  Minimoog-style post-filter -> pre-filter feedback. At 0.3: subtle thickening.
  At 0.7: aggressive growl. Works with every filter topology. Default
  `0.0`. Range `[0, 1]`.
- `feedback_saturation: float`
  Saturation in the feedback path (tanh). Tames feedback and adds harmonics.
  Default `0.3`. Range `[0, 1]`.

All feedback summations inject a deterministic ~1e-6 (120 dB below unity)
bootstrap noise seed — derived from a hash of the input signal, so identical
inputs produce identical noise. Without it, a silent input into a high-Q
self-oscillating filter would stay silent; with it, the filter can wake up
from exact silence. Inaudible on normal material.

The external feedback loop is solved implicitly under `filter_solver="newton"`
(the default) on `svf`, `cascade`, `sem`, `sallen_key`, `ladder`, `jupiter` —
eliminating the one-sample delay that otherwise damps high-resonance behaviour.
`k35` and `diode` still use unit-delay external feedback (see `FUTURE.md`).
The `"adaa"` solver stays on unit-delay everywhere for bit-identical legacy
behaviour.

#### VCA Nonlinearity

- `vca_nonlinearity: float`
  Gain-dependent envelope saturation modeling OTA-based VCAs. Drive scales with
  envelope level — attack peaks get maximum saturation, release tails stay
  clean. At 0.1-0.3: subtle warmth. At 0.5-0.8: audible push on loud notes.
  Default `0.0`. Range `[0, 1]`.

#### Transient / Reset Modes

- `transient_mode: str`
  Controls what oscillator state (phase and DC-offset sign) is carried from
  one note to the next within a voice.  Supported modes:

  | mode | oscillator phase | DC-offset sign | character |
  |-|-|-|-|
  | `"analog"` (default) | carried forward | carried forward | most analog-feeling — each note inherits the previous note's exact phase + DC sign, eliminating zero-crossing clicks on rapid restrikes |
  | `"dc_reset"` | carried forward | redrawn per note | clean attack with a fresh DC decision but no phase-reset glitch — useful when DC drift would otherwise accumulate obvious asymmetric character |
  | `"osc_reset"` | deterministic per-note seed | redrawn per note | fully reset oscillator state, approximating the pre-carry-era behavior where every note started from a fresh random phase — picks up a faint "new note" click that can be musical on staccato patches |

  The default `"analog"` is usually the right choice.  Reach for `"dc_reset"`
  on bass patches where you want a fresh attack feel but smooth phase
  continuity; reach for `"osc_reset"` only when you specifically want the
  legacy percussive onset.

  State carryover relies on a per-voice mutable `voice_state` dict that the
  score-level render loop threads through automatically — you do not need
  to manage it yourself.  When `voice_state` is `None` (for example, in
  isolated `render_note_signal(...)` calls from a unit test), every note
  starts fresh regardless of `transient_mode`, matching the pre-carry era.

  Example:

  ```yaml
  synth_defaults:
    engine: polyblep
    preset: moog_bass_ladder
    transient_mode: dc_reset
  ```

  Applies to both the `polyblep` and `filtered_stack` engines.

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
- `moog_lead` - ladder-filtered saw lead with bass compensation and feedback.
- `moog_bass_ladder` - 4-pole ladder bass with resonance and feedback growl.
- `cs80_brass` - dual-filter brass stab using serial HPF and filter morph.
- `oberheim_pad` - wide voice card spread pad with oscillator imperfections and VCA warmth.
- `jupiter_saw` - bright saw stack with ladder filter and subtle oscillator softness.
- `acid_ladder` - resonant ladder bass with filter morph automation and feedback saturation.

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

## `va`

Implementation: [code_musics/engines/va.py](code_musics/engines/va.py)

Voiced 90s/00s Virtual Analog engine. The character does not come from one
feature but from the engine as a whole: oscillator voicing, pre-filter drive,
dual-filter routing, optional resonant comb, and the standard analog-character
surface (voice-card, jitter, drift, CV dither). Presets cover Roland JP-8000,
Access Virus, and Waldorf Q flavors.

Two oscillator modes:

- `osc_mode="supersaw"` — 7-voice PolyBLEP saw bank with Szabo-accurate
  nonlinear detune and mix laws (JP-8000 lineage), random starting phase per
  note, and optional hard-sync (Virus HyperSaw-adjacent).
- `osc_mode="spectralwave"` — partial-bank oscillator with continuous
  `spectral_position` sweep from saw → spectral → square, plus optional
  spectral-morph layer (smear, inharmonic_scale, phase_disperse, shepard,
  random_amplitudes).

Signal flow:

```text
osc_mix → drive (waveshaper) → [comb pre] → filter chain → [comb post] → post-proc → amp
                                                                ↑
                                             single | serial | parallel | split
```

Parameters:

### Oscillator selection

- `osc_mode: str` — `"supersaw"` or `"spectralwave"`. Default `"supersaw"`.

### Supersaw params (`osc_mode="supersaw"`)

- `supersaw_detune: float` — Perceptual detune amount in [0, 1]. Maps via
  Szabo's reverse-engineered polynomial to per-voice cents spread. Nonlinear:
  low values stay subtle, high values widen rapidly. Default `0.3`.
- `supersaw_mix: float` — Mix amount in [0, 1]. Drives both side-voice and
  center-voice gain through Szabo's nonlinear laws; the center level changes
  with mix (critical to the JP sound). Default `0.5`.
- `supersaw_sync: bool` — Hard-sync the 6 side voices to the center voice's
  phase wrap. Off by default; enabling produces a more aggressive
  Virus-HyperSaw character. Default `False`.

### Spectralwave params (`osc_mode="spectralwave"`)

- `spectral_position: float` — Morph axis in [0, 1]. `0` is saw-like (all
  harmonics, `1/n` roll-off); `1` is square-like (odd-only, `1/n`); mid values
  emphasize formant-like odd harmonics (3, 5, 9, 13, 21) for Virus-classic
  "spectral wave" character. Default `0.0`.
- `n_partials: int` — Number of partials in the bank, up to 64. Default `64`.
- `spectral_morph_type: str` — Optional morph layer: `"none"`,
  `"inharmonic_scale"`, `"phase_disperse"`, `"smear"`, `"shepard"`,
  `"random_amplitudes"`. Applied to the partial bank before rendering.
- `spectral_morph_amount: float` — Strength of the morph layer. Default `0.0`.
- `spectral_morph_shift: float`, `spectral_morph_center_k: int`,
  `spectral_morph_seed: int` — Per-morph-type parameters; see
  `_spectral_morphs.py` for semantics.
- `sigma_approximation: bool` — Apply Lanczos sigma factors against Gibbs
  ringing. Default `False`.
- `osc2_level: float` — Optional simple osc2 stack level in `[0, 1]`.
  Default `0.0` (disabled). Mixed as `(osc1 + osc2_level * osc2) /
  (1 + osc2_level)` so enabling it does not raise overall loudness.
- `osc2_semitones: float` — Coarse tuning offset for osc2 in semitones.
  Default `0.0`. `-12` / `+12` give sub and octave-up stacks; negative
  values work fine. No clamped range.
- `osc2_detune_cents: float` — Fine tuning offset for osc2 in cents.
  Default `0.0`. Musical range roughly `±30`; larger values turn the
  stack into a detuned unison.

### Pre-filter drive stage

- `drive_amount: float` — Virus-style pre-filter saturation amount in [0, 1].
  Default `0.0` (bypass). Uses the waveshaper stage with ADAA anti-aliasing.
- `drive_algorithm: str` — `"tanh"` (default; warm analog), `"atan"` (softer
  shoulder), or `"exponential"` (brighter, more aggressive upper-mid grit).

### Dual-filter routing

- `filter_routing: str` — `"single"` (F1 only, default), `"serial"` (F1→F2),
  `"parallel"` (mean of F1 and F2), `"split"` (osc split: center/osc1 → F1,
  satellites/osc2 → F2).

Each filter slot (`filter1_*`, `filter2_*`) accepts the full `apply_filter`
surface. The top-level `cutoff_hz`, `resonance_q`, etc. map to `filter1_*` for
polyblep-style shorthand. Filter 2 defaults to inheriting filter 1 (so
`"serial"` with no f2 params cascades the same filter twice for a steeper
slope).

Key per-slot params: `cutoff_hz`, `resonance_q`, `keytrack`, `reference_freq_hz`,
`filter_env_amount`, `filter_env_decay`, `filter_mode`, `filter_drive`,
`filter_topology`, `bass_compensation`, `filter_morph`, `hpf_cutoff_hz`,
`hpf_resonance_q`, `feedback_amount`, `feedback_saturation`,
`k35_feedback_asymmetry` (K35-topology-only; see Filter Topology). All
eight topologies from the shared Filter Topology surface are accepted
on each slot, and the top-level shorthand routes into filter 1 for
polyblep-style authoring.

### Comb filter slot

- `comb_position: str` — `"off"` (default), `"pre_filter"`, `"post_filter"`,
  or `"parallel"` (summed with filter-chain output).
- `comb_delay_ms: float` — Base delay in ms. Default `8.0`.
- `comb_feedback: float` — Feedback gain in [0, 0.99]. Above ~0.95 with low
  damping produces sustained self-oscillation. Soft-clipped internally.
- `comb_damping: float` — Feedback-path 1-pole lowpass amount in [0, 1].
  `0` is bright resonance, `1` is dark short decay. Default `0.2`.
- `comb_keytrack: float` — Frequency tracking in [0, 1]. At `1.0` the comb
  resonates exactly at the note's fundamental (karplus-strong-ish bell).
  Default `0.0`.
- `comb_mix: float` — Parallel-position dry/wet balance. Default `0.5`.

### Analog Character (va)

The `va` engine consumes the full analog-character surface: `pitch_drift`,
`analog_jitter`, `noise_floor`, `drift_rate_hz`, `cutoff_drift`,
`voice_card_spread` (and per-group overrides). See the polyblep section for
individual parameter semantics. Defaults are tuned slightly looser than
polyblep since these synths are meant to feel "alive but produced."

Validation:

- `duration > 0`
- `sample_rate > 0`
- `osc_mode ∈ {"supersaw", "spectralwave"}`
- `filter_routing ∈ {"single", "serial", "parallel", "split"}`
- `comb_position ∈ {"off", "pre_filter", "post_filter", "parallel"}`
- `drive_algorithm ∈ {"tanh", "atan", "exponential"}`
- `filter{1,2}_cutoff_hz > 0`, `filter{1,2}_filter_env_decay > 0`
- `drive_amount ∈ [0, 1]`, `spectral_position ∈ [0, 1]`, `n_partials >= 1`
- `comb_delay_ms > 0`, `comb_feedback ∈ [0, 0.99]`,
  `comb_damping ∈ [0, 1]`, `comb_keytrack ∈ [0, 1]`, `comb_mix ∈ [0, 1]`

Notes:

- Returns mono `float64`. Stereo comes from `Voice.pan`, `SmearVoice`, or
  effects.
- Peak-normalizes before multiplying by `amp`, so the per-note level stays
  predictable regardless of drive/filter/comb combinations.
- `freq_trajectory` is honored — pitch motion works on both osc modes.
- ADSR is applied outside the engine by the score layer.

Artifact-risk guidance:

- `supersaw_detune > 0.9` — verges on chorus-ish warble. Musical at `0.3-0.7`.
- `drive_amount > 0.7` — moves from Virus-style saturation into obvious
  distortion. For a clean warmth, stay at `0.2-0.4`.
- `comb_feedback > 0.95` with `comb_damping < 0.15` — approaches
  self-oscillation. Intentional for comb-bell presets; otherwise keep
  `feedback < 0.9`.
- High `resonance_q` × low `cutoff_hz` × `ladder` topology with
  `feedback_amount > 0` can self-oscillate below the note fundamental.
  Bootstrap noise keeps this bounded, but it's usually a signal that the
  filter is over-driven.

Common interaction trap:

The `drive_amount` stage and the per-filter `filter1_drive` / `filter2_drive`
stages are **independent**. Running both at high amounts compounds saturation
quickly. For analog-modeling gain staging, prefer raising only one at a time.
`drive_amount` is the more Virus-flavored surface (pre-filter gain into a
clean filter); `filter_drive` is the more Ladder-flavored surface (drive
inside the filter feedback path).

Presets:

- `jp8000_hoover` — Classic trance hoover with ladder filter and light drive.
- `jp8000_lead` — Tighter supersaw with moderate filter_env for melodic leads.
- `supersaw_pad` — Wide stacked pad, serial filter cascade for a gentler roll-off.
- `virus_pad` — Spectralwave with smear morph, long release, serial filter.
- `virus_bass` — Low spectral_position (saw-leaning) with pre-filter drive and
  HPF cleanup; fast filter_env for punch.
- `virus_lead` — Supersaw with hard-sync, parallel BPF+LPF, mid drive.
- `q_comb_pad` — Spectralwave with phase-disperse morph and keytracked comb
  post-filter.
- `q_comb_bell` — Self-oscillating comb (`feedback=0.92`, `keytrack=1.0`) over
  a spectralwave with light inharmonic morph — karplus-strong-adjacent bell.
- `q_spectral_lead` — Random-amplitude spectral morph through a
  slope-morphed ladder filter.

Example (structured spec form):

```python
voice = score.add_voice(
    "lead",
    synth_defaults={
        "engine": "va",
        "preset": "virus_lead",
        "env": {"attack_ms": 20.0, "release_ms": 500.0},
        "params": {
            "supersaw_detune_ratio": 0.18,
            "supersaw_mix_ratio": 0.45,
            "drive_amount_ratio": 0.35,
            "filter1_resonance_ratio": 2.2,
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

**Wire noise mode:**

- `wire_noise_mode: str`
  Noise source for the wire layer. `"white"` (default) uses raw white noise;
  `"colored"` applies a 500 Hz highpass before the comb filter for a brighter,
  thinner wire character.

**FM body params (optional — standard oscillator used when `body_fm_ratio` is absent):**

- `body_fm_ratio: float`
  Modulator frequency as ratio of body frequency. Enables FM synthesis on the body,
  replacing the standard sine oscillator with an FM pair.
- `body_fm_index: float` (default 2.0)
  Peak modulation index (controls harmonic richness).
- `body_fm_feedback: float` (default 0.0)
  Modulator self-feedback (adds noise/complexity).
- `body_fm_index_envelope: list[dict]`
  Multi-point envelope for index modulation. Value axis = 0-1 multiplier.
  Default when absent: `exp(-t/0.05)` for percussive FM.

**Body distortion params (optional — bypassed when `body_distortion` is absent):**

- `body_distortion: str`
  Waveshaping algorithm applied to the body after assembly. Same algorithms as
  kick_tom: `"tanh"`, `"foldback"`, etc. See Shared Drum DSP Infrastructure.
- `body_distortion_drive: float` (default 0.5)
  Drive amount (0-1).
- `body_distortion_mix: float` (default 1.0)
  Dry/wet blend (0 = fully dry, 1 = fully wet).
- `body_distortion_drive_envelope: list[dict]`
  Multi-point envelope for drive modulation over time.

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
| `fm_snare` | FM body for harmonically rich snare attack |
| `driven_snare` | Tanh waveshaping on the body for gritty character |
| `fm_tom` | FM body with lighter wire, tom-like tuning |
| `fm_noise_burst` | Heavy FM + noise, minimal body — noise-forward burst |
| `gated_snare` | Multi-point body envelope with sharp gate cutoff |

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

## `drum_voice`

Implementation: [code_musics/engines/drum_voice.py](code_musics/engines/drum_voice.py)

Composable percussion synthesizer with four independent, mixable layers --
exciter, tone, noise, and metallic. Any synthesis mode can be combined with any
other, enabling creative hybrid timbres that the separate drum engines cannot
produce (e.g., FM metallic partials on a kick, resonator body on a snare).
Replaces the five separate drum engines (`kick_tom`, `snare`, `clap`,
`metallic_perc`, `noise_perc`) with a unified architecture. All original presets
are available under the `drum_voice` engine name.

### Machinedrum-inspired kernels

Four additional kernel families extend the base architecture with
Elektron-Machinedrum-style synthesis primitives:

- **EFM (two-op FM)** — via `tone_type="efm"`. Aggressive DX-style FM with a
  primary modulator plus optional second modulator, feedback, and a decaying
  index envelope. Great for punchy kicks with harmonic bite, metallic snare
  cracks, chirpy cowbells. Reads `efm_ratio` / `efm_index_peak` /
  `efm_feedback` / `efm_ratio_2` / `efm_index_2` / `efm_feedback_2` /
  `efm_index_decay_s` / `efm_carrier_feedback` / `efm_index_envelope`.
- **PI modal resonator banks** — via `tone_type="modal"` (body) or
  `metallic_type="modal_bank"` (bell / shimmer layer). Real
  physical-informed modal synthesis sourced from named mode tables
  (`membrane`, `bar_wood`, `bar_metal`, `bar_glass`, `plate`, `bowl`,
  `stopped_pipe`) in `code_musics/spectra.py`, driven by the exciter
  layer. Ergonomic `pi_hardness` / `pi_tension` / `pi_damping` /
  `pi_damping_tilt` / `pi_position` macros shape strike character,
  stretch, decay balance, and strike position.
- **E12-style sample exciters** — via `exciter_type="sample"`. Loads a
  WAV and plays it back as the transient layer, with pitch-tracked
  playback, micro start-offset jitter, optional ring-mod layer, reverse,
  and bend envelopes. Pair with a PI modal body for convincing
  sample-plus-resonator drums.
- **Digital-character shapers** — via `shaper="bit_crush"` /
  `"rate_reduce"` / `"digital_clip"` on the voice shaper slot. Crunch,
  lo-fi aliasing, and asymmetric DX-style clipping. These read
  `bit_depth` (bit_crush) and `reduce_ratio` (rate_reduce) at the voice
  level.

### Architecture

```text
                   +---------------+
                   |  Pitch Sweep  |  (shared freq_profile for tone + metallic)
                   +-------+-------+
                           |
Exciter -> [shaper] --+    Tone -> [shaper] --+
                      |                       |
                   x level x env           x level x env
                      |                       |
                      +----------+------------+
                                 |
Noise -> [ZDF filter] --+       |
                     x level x env
                        |       |
Metallic -> [ZDF filter]+       |
          x level x env         |
                                |
                     -----------+
                         |
                      Sum mix
                         |
                   [Voice filter]  (optional ZDF SVF)
                         |
                   [Voice shaper]  (optional)
                         |
                   Peak normalize -> x amp
```

Key routing:

- **Exciter feeds resonator**: When `tone_type="resonator"`, the exciter output
  drives the resonator as excitation. `exciter_level` controls how much raw
  exciter also appears in the mix (0 = only resonated result).
- **Pitch sweep is shared**: Tone and metallic both receive the swept
  `freq_profile`. Metallic partial ratios are relative to this, so the sweep
  affects all pitched content coherently.
- **Shapers are pre-envelope**: Distortion on the raw oscillator before decay
  shapes it.
- **Noise + metallic get per-layer ZDF filters**: These layers produce broadband
  content that almost always needs filtering. Envelope-modulated cutoff supported.
- **Voice filter + shaper are post-mix**: Glue the layers, then final shaping.

### Layer types

**Exciter layer** (transient/attack energy):

| Type | Description | Key params |
|------|-------------|------------|
| `click` | Bandpass-filtered noise burst | `exciter_center_hz`, `exciter_emphasis` |
| `impulse` | Sub-ms pulse for exciting resonators | `exciter_width_samples` |
| `multi_tap` | Multiple rapid micro-bursts (clap architecture) | `exciter_n_taps`, `exciter_tap_spacing_s`, `exciter_tap_decay_s`, `exciter_tap_crescendo`, `exciter_tap_acceleration`, `exciter_tap_freq_spread`, `exciter_tap_bandwidth_ratio` |
| `fm_burst` | Short FM oscillator burst | `exciter_fm_ratio`, `exciter_fm_index`, `exciter_fm_feedback` |
| `noise_burst` | Wider-band noise with optional tail filter | `exciter_bandwidth_ratio`, `exciter_filter_cutoff_hz`, `exciter_filter_q` |
| `sample` | WAV playback as transient layer | `exciter_sample_path`, `exciter_sample_pitch_shift`, `exciter_sample_start_jitter_ms`, `exciter_sample_ring_freq_hz`, `exciter_sample_reverse` |

**Tone layer** (pitched periodic content / body):

| Type | Description | Key params |
|------|-------------|------------|
| `oscillator` | Waveform + pitch sweep + punch | `tone_wave` (sine/tri/sine_clip), `tone_punch`, `tone_second_harmonic` |
| `resonator` | Time-varying biquad bandpass driven by exciter | `tone_punch` (affects fallback impulse if no exciter) |
| `fm` | FM synthesis body | `tone_fm_ratio`, `tone_fm_index`, `tone_fm_feedback`, `tone_fm_index_decay_s` |
| `efm` | Two-op DX-style FM (Machinedrum EFM) | `efm_ratio`, `efm_index_peak`, `efm_feedback`, `efm_ratio_2`, `efm_index_2` |
| `modal` | Physical-informed modal resonator bank | `modal_mode_table`, `modal_n_modes`, `pi_hardness`, `pi_tension`, `pi_damping`, `pi_position` |
| `additive` | Small partial set at configurable ratios | `tone_partial_ratios`, `tone_n_partials`, `tone_brightness` |

**Noise layer** (aperiodic texture):

| Type | Description | Key params |
|------|-------------|------------|
| `white` | Raw white noise | (none beyond shared) |
| `colored` | White + highpass filter | `noise_pre_hp_hz` |
| `bandpass` | FFT-domain bandpass noise | `noise_center_ratio`, `noise_width_ratio` |
| `comb` | Noise through comb filter at note freq (snare wire buzz) | `noise_comb_feedback`, `noise_pre_noise_mode` (white/colored), `noise_pre_hp_hz` |

**Metallic layer** (inharmonic periodic content):

| Type | Description | Key params |
|------|-------------|------------|
| `partials` | Additive inharmonic partials | `metallic_partial_ratios`, `metallic_n_partials`, `metallic_oscillator_mode` (sine/square), `metallic_brightness`, `metallic_density` |
| `ring_mod` | Ring modulator on harmonic partial sum | `metallic_ring_mod_freq_ratio`, `metallic_ring_mod_amount`, `metallic_n_partials`, `metallic_brightness`, `metallic_density` |
| `fm_cluster` | Multiple FM operators at inharmonic ratios | `metallic_n_operators`, `metallic_fm_ratios`, `metallic_fm_index`, `metallic_fm_feedback`, `metallic_brightness`, `metallic_density` |
| `efm_cymbal` | N-op PM cymbal (Machinedrum EFM cymbal) | `cymbal_op_count`, `cymbal_ratio_set` (tr808 / tr909 / bar / plate), `cymbal_index`, `cymbal_feedback` |
| `modal_bank` | Modal resonator bank (metallic variant) | `metallic_mode_table`, `metallic_n_modes`, `pi_hardness`, `pi_tension`, `pi_damping`, `pi_position` |

### Parameters

**Layer type selectors** (set to `None` to disable a layer):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tone_type` | `str \| None` | `"oscillator"` | Tone layer synthesis type |
| `exciter_type` | `str \| None` | `None` | Exciter layer synthesis type |
| `noise_type` | `str \| None` | `None` | Noise layer synthesis type |
| `metallic_type` | `str \| None` | `None` | Metallic layer synthesis type |

**Layer levels:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tone_level` | `float` | `1.0` | Tone layer mix level |
| `exciter_level` | `float` | `0.08` | Exciter layer mix level |
| `noise_level` | `float` | `0.02` | Noise layer mix level |
| `metallic_level` | `float` | `0.0` | Metallic layer mix level |

**Decay times:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tone_decay_s` | `float` | `0.26` | Tone layer decay time in seconds |
| `exciter_decay_s` | `float` | `0.007` | Exciter layer decay time in seconds |
| `noise_decay_s` | `float` | `0.028` | Noise layer decay time in seconds |
| `metallic_decay_s` | `float` | `0.08` | Metallic layer decay time in seconds |

**Pitch sweep:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tone_sweep_ratio` | `float` | `2.5` | Starting pitch multiplier for the sweep (decays toward 1.0) |
| `tone_sweep_decay_s` | `float` | `0.042` | Sweep decay time in seconds |

**Per-layer shaper params** (same pattern for `exciter_shaper_*`, `tone_shaper_*`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tone_shaper` | `str \| None` | `None` | Shaper algorithm, `"saturation"`, `"preamp"`, or `None` |
| `tone_shaper_drive` | `float` | `0.5` | Drive amount (0-1) |
| `tone_shaper_mix` | `float` | `1.0` | Dry/wet blend |
| `tone_shaper_mode` | `str` | `"triode"` | Saturation mode (when shaper=`"saturation"`) |
| `tone_shaper_fidelity` | `float` | `0.5` | Saturation fidelity (when shaper=`"saturation"`) |

**Voice filter** (post-mix, optional):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filter_mode` | `str \| None` | `None` | `"lowpass"`, `"bandpass"`, `"highpass"`, or `None` |
| `filter_cutoff_hz` | `float` | `2000.0` | Base cutoff frequency |
| `filter_q` | `float` | `0.707` | Filter resonance (>= 0.5) |
| `filter_drive` | `float` | `0.0` | Filter drive amount |
| `filter_envelope` | `list[dict]` | `None` | Multi-point envelope for cutoff modulation (values = Hz) |
| `filter_topology` | `str` | `"svf"` | Any of the eight shared topologies — `"svf"`, `"ladder"`, `"sallen_key"`, `"cascade"`, `"sem"`, `"jupiter"`, `"k35"`, `"diode"`. Same semantics as the polyblep/va filter topology surface. |
| `filter_morph` | `float` | `0.0` | Continuous pole-tap / mode blend. Range and behavior are topology-dependent; see the Filter Mode Morphing section under polyblep. |
| `k35_feedback_asymmetry` | `float` | `0.0` | K35-only feedback-diode asymmetry `[0, 1]`. Ignored by other topologies. |

**Voice shaper** (post-mix, optional):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shaper` | `str \| None` | `None` | Voice-level shaper (same dispatch as per-layer) |
| `shaper_drive` | `float` | `0.5` | Drive amount |
| `shaper_mix` | `float` | `1.0` | Dry/wet blend |
| `shaper_mode` | `str` | `"triode"` | Saturation mode |
| `shaper_fidelity` | `float` | `0.5` | Saturation fidelity |

**Per-layer noise/metallic filters** (optional):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `noise_filter_mode` | `str \| None` | `None` | ZDF SVF filter on noise layer |
| `noise_filter_cutoff_hz` | `float` | `2000.0` | Noise filter cutoff |
| `noise_filter_q` | `float` | `0.707` | Noise filter Q |
| `metallic_filter_mode` | `str \| None` | `None` | ZDF SVF filter on metallic layer |
| `metallic_filter_cutoff_hz` | `float` | `2000.0` | Metallic filter cutoff |
| `metallic_filter_q` | `float` | `1.2` | Metallic filter Q |

**Multi-point envelope overrides** (optional):

- `tone_envelope: list[dict]` -- replaces tone exponential decay
- `exciter_envelope: list[dict]` -- replaces exciter exponential decay
- `noise_envelope: list[dict]` -- replaces noise exponential decay
- `metallic_envelope: list[dict]` -- replaces metallic exponential decay
- `tone_pitch_envelope: list[dict]` -- replaces pitch sweep (values = freq multiplier)

**Velocity-to-timbre:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `velocity_timbre_decay` | `float` | `0.0` | Velocity sensitivity for decay time scaling |
| `velocity_timbre_brightness` | `float` | `0.0` | Velocity sensitivity for brightness scaling |
| `velocity_timbre_harmonics` | `float` | `0.0` | Velocity sensitivity for harmonic content (FM index) |
| `velocity_timbre_noise` | `float` | `0.0` | Velocity sensitivity for noise balance |

### Shaper dispatch

The shaper slots (`exciter_shaper`, `tone_shaper`, `shaper`) dispatch to three
tiers of nonlinearity, all reusing existing code:

| Shaper value | Routes to | Character |
|---|---|---|
| `"tanh"`, `"foldback"`, `"atan"`, etc. (11 algorithms) | `_waveshaper.py` with ADAA | Fast, simple tone shaping |
| `"saturation"` | `synth.py` modern saturation (tube/triode/iron) | Rich analog character with optional oversampling |
| `"preamp"` | `synth.py` preamp (flux-domain transformer) | Warm transformer color, frequency-dependent |

Additional params (`mode`, `fidelity`) are forwarded to the underlying
implementation. The drum voice calls into the existing effect functions with no
code duplication.

### Ergonomic macros

Three high-level perceptual knobs that fill in params not already set by the
user or a preset. Set to `None` (default) to leave inactive. Macros never
override explicit values.

**`punch`** (0.0 = soft pillowy, 1.0 = hard snappy):

| Target param | At 0.0 | At 1.0 |
|---|---|---|
| `exciter_level` | 0.01 | 0.25 |
| `exciter_decay_s` | 0.012 | 0.003 |
| `exciter_center_hz` | 1500 | 5000 |
| `tone_punch` | 0.0 | 0.35 |

**`decay_shape`** (0.0 = tight/gated, 1.0 = long/boomy):

| Target param | At 0.0 | At 1.0 |
|---|---|---|
| `tone_decay_s` | 0.08 | 0.9 |
| `noise_decay_s` | 0.015 | 0.3 |
| `metallic_decay_s` | 0.03 | 0.4 |
| `tone_sweep_decay_s` | 0.02 | 0.08 |

**`character`** (0.0 = clean/pure, 1.0 = dirty/complex):

| Target param | At 0.0 | At 1.0 |
|---|---|---|
| `tone_shaper` | None | `"tanh"` (0.3+), `"foldback"` (0.7+) |
| `tone_shaper_drive` | 0.0 | 0.6 |
| `filter_drive` | 0.0 | 0.3 |
| `noise_level` | (unchanged) | +30% boost if already present |

### Machinedrum kernel parameters

**EFM tone params** (consumed when `tone_type="efm"`):

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `efm_ratio` | `float` | > 0 | `1.5` | Primary modulator carrier-ratio |
| `efm_index_peak` | `float` | >= 0 | `3.0` | Peak modulation index for the primary modulator |
| `efm_feedback` | `float` | 0.0-1.0 | `0.0` | Primary modulator self-feedback |
| `efm_ratio_2` | `float` | >= 0 | `0.0` | Second modulator ratio (0 disables) |
| `efm_index_2` | `float` | >= 0 | `0.0` | Peak index for the second modulator |
| `efm_feedback_2` | `float` | 0.0-1.0 | `0.0` | Second modulator self-feedback |
| `efm_index_decay_s` | `float` | > 0 | `0.05` | Exp-decay time constant for both modulator indices (when no explicit envelope) |
| `efm_carrier_feedback` | `float` | 0.0-1.0 | `0.0` | Carrier self-feedback for added growl |
| `efm_index_envelope` | `list[dict]` | -- | `None` | Optional multi-point index envelope; overrides `efm_index_decay_s` |

**EFM cymbal params** (consumed when `metallic_type="efm_cymbal"`):

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `cymbal_op_count` | `int` | >= 1 | `6` | Number of PM operators modulating a sine carrier |
| `cymbal_ratio_set` | `str` | -- | `"tr808"` | Named ratio set: `"tr808"`, `"tr909"`, `"bar"`, `"plate"` |
| `cymbal_index` | `float` | >= 0 | `2.5` | Peak PM index, broadcast to all operators |
| `cymbal_feedback` | `float` | 0.0-1.0 | `0.0` | Per-operator self-feedback |
| `cymbal_index_envelope` | `list[dict]` | -- | `None` | Optional multi-point index envelope applied identically to all operators |

**PI modal params** (tone uses `modal_` prefix when `tone_type="modal"`; metallic uses `metallic_` prefix when `metallic_type="modal_bank"`):

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `{prefix}_mode_table` | `str` | -- | `"membrane"` (tone) / `"bar_metal"` (metallic) | Named mode table from `spectra.py`: `membrane`, `bar_wood`, `bar_metal`, `bar_glass`, `plate`, `bowl`, `stopped_pipe` |
| `{prefix}_n_modes` | `int` | >= 1 | (table length) | Cap on number of modes used from the table |
| `{prefix}_ratios` | `list[float]` | -- | `None` | Explicit custom mode ratios (overrides the table) |
| `{prefix}_amps` | `list[float]` | -- | `None` | Explicit per-mode amplitudes (must match `ratios` length) |
| `{prefix}_decays_s` | `list[float]` | -- | `None` | Explicit per-mode decay times (must match `ratios` length) |
| `{prefix}_decay_s` | `float` | > 0 | `0.6` (tone) / `0.4` (metallic) | Global decay multiplier |
| `{prefix}_tension` | `float` | -1.0 to 1.0 | `0.0` | Fractional stretch of mode ratios (also accepted as `pi_tension`) |
| `{prefix}_damping` | `float` | 0.0-1.0 | `1.0` | Global decay multiplier (also accepted as `pi_damping`) |
| `{prefix}_damping_tilt` | `float` | -1.0 to 1.0 | `0.0` | High- vs low-mode decay balance (also accepted as `pi_damping_tilt`) |
| `{prefix}_position` | `float` | 0.0-1.0 | `0.0` | Strike position window on mode amplitudes (also accepted as `pi_position`) |

**PI macros** (map onto the modal params above for quick perceptual control):

| Macro | Range | Description |
|-------|-------|-------------|
| `pi_hardness` | 0.0-1.0 | Mallet / strike brightness (drives exciter brightness and index) |
| `pi_tension` | -1.0 to 1.0 | Mode-ratio stretch factor (negative = compressed, positive = stretched) |
| `pi_damping` | 0.0-1.0 | Global modal decay multiplier |
| `pi_damping_tilt` | -1.0 to 1.0 | Decay balance: positive dampens high modes faster, negative the opposite |
| `pi_position` | 0.0-1.0 | Strike-position window on the modal amplitudes |

**Sample exciter params** (consumed when `exciter_type="sample"`):

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `exciter_sample_path` | `str` | -- | **required** | Path to the WAV file |
| `exciter_sample_root_freq` | `float` | > 0 | `freq` | Source pitch of the sample (for pitch-tracked playback) |
| `exciter_sample_pitch_shift` | `bool` | -- | `True` | Whether to pitch-shift to the note freq |
| `exciter_sample_pitch_shift_semitones` | `float` | -- | `0.0` | Constant additional semitone offset |
| `exciter_sample_start_offset_ms` | `float` | >= 0 | `0.0` | Skip this many ms from the start of the sample |
| `exciter_sample_start_jitter_ms` | `float` | >= 0 | `0.0` | Random start-offset jitter range in ms |
| `exciter_sample_bend_envelope` | `list[dict]` | -- | `None` | Optional multi-point pitch-bend envelope (cents) |
| `exciter_sample_ring_freq_hz` | `float` | >= 0 | `0.0` | Ring-mod frequency (0 disables) |
| `exciter_sample_ring_depth` | `float` | 0.0-1.0 | `0.0` | Ring-mod depth |
| `exciter_sample_reverse` | `bool` | -- | `False` | Play the sample backwards |

**Digital-character voice shaper params** (consumed by the voice `shaper` slot when set to a digital-character algorithm):

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `bit_depth` | `float` | 1.0-16.0 | `8.0` | Effective bit-depth for `shaper="bit_crush"` |
| `reduce_ratio` | `float` | >= 1.0 | `2.0` | Sample-and-hold factor for `shaper="rate_reduce"` |

### Presets (drum_voice)

All 62 migrated presets organized by original engine category, plus 15
Machinedrum-inspired presets at the end (EFM tones / EFM cymbals / PI modal /
digital character):

**Kick/tom** (from kick_tom):

| Preset | Character |
|--------|-----------|
| `808_hiphop` | Long 808-style sub kick with deep sweep |
| `808_house` | Medium 808 with moderate sweep and punch |
| `808_tape` | 808 with sine_clip body for tape-saturated warmth |
| `909_techno` | Hard 909-style kick with triangle body and strong click |
| `909_house` | Clean 909 house kick with sharp attack |
| `909_crunch` | 909 with sine_clip body and extra click emphasis |
| `distorted_hardkick` | Aggressive hard kick with heavy click and sine_clip |
| `zap_kick` | Wide pitch sweep (5x) for zappy electronic kicks |
| `round_tom` | Gentle tom with moderate sweep and triangle body |
| `floor_tom` | Deep floor tom with longer decay |
| `electro_tom` | Punchy electronic tom |
| `ring_tom` | Tom with prominent ring/overtone character |
| `melodic_resonator` | Long resonator body tuned for melodic percussion |
| `kick_bell` | Kick with metallic partials layer for bell overtones |
| `gated_808` | 808 with multi-point body envelope for gated decay |
| `pitch_dive` | Extreme pitch sweep (6x) for dramatic dive effect |
| `filtered_kick` | Kick with envelope-modulated lowpass voice filter |
| `fm_body_kick` | FM body synthesis for harmonically rich attack |
| `foldback_kick` | Foldback waveshaper on the body for harmonic density |
| `808_resonant` | Resonator body mode driven by click excitation |
| `808_resonant_long` | Long-decay resonator with wider pitch sweep |
| `resonant_tom` | Resonator body for naturally resonant tom character |

**Snare** (from snare):

| Preset | Character |
|--------|-----------|
| `909_tight` | Tight 909-style snare with balanced body/wire |
| `909_fat` | Fatter 909 with longer decay and more body emphasis |
| `rim_shot` | Short, sharp rim shot dominated by click transient |
| `brush` | Soft brush sweep with long wire decay and minimal body |
| `fm_tom` | FM body with lighter wire, tom-like tuning |
| `fm_noise_burst` | Heavy FM + noise, minimal body -- noise-forward burst |
| `gated_snare` | Multi-point body envelope with sharp gate cutoff |
| `fm_snare` | FM body for harmonically rich snare attack |
| `driven_snare` | Tanh waveshaping on the body for gritty character |

**Clap** (from clap):

| Preset | Character |
|--------|-----------|
| `909_clap` | Classic 909-style 4-tap clap |
| `tight_clap` | Short, snappy 3-tap clap |
| `big_clap` | Wider 6-tap clap with longer body |
| `finger_snap` | Quick 2-tap snap with narrow bandpass |
| `hand_clap` | Natural 4-tap hand clap with wider spacing and softer click |
| `gated_clap` | 909-style clap with gated tail via noise envelope |
| `909_clap_authentic` | 909-accurate timing with tap acceleration and frequency spread |
| `scattered_clap` | Wide 6-tap clap with heavy acceleration and spread |
| `granular_cascade` | Dense 8-tap granular burst with high acceleration and spread |
| `micro_burst` | Tight 6-tap burst with subtle spread, very short body |

**Metallic** (from metallic_perc):

| Preset | Character |
|--------|-----------|
| `closed_hat` | Tight, bright closed hi-hat |
| `open_hat` | Longer open hi-hat |
| `pedal_hat` | Medium pedal hi-hat between closed and open |
| `ride_bell` | Focused ride bell |
| `ride_bow` | Washy ride bow with long decay and extra partial |
| `crash` | Long crash cymbal (1.8 s decay) with high partial density |
| `cowbell` | Two-partial cowbell with fixed ratios |
| `clave` | Sharp, short clave click |
| `harmonic_bell` | Harmonic integer-ratio partials for tuned bell |
| `septimal_bell` | 7-limit JI partial ratios for xenharmonic bell |
| `square_gamelan` | Square-wave partials with gamelan-inspired ratios |
| `beating_hat_a` / `b` / `c` | Near-unison partial pairs for beating interference |
| `swept_hat` | Hi-hat with lowered filter start for sweep character |
| `decaying_bell` | Long bell with multi-point amplitude envelope |
| `808_closed_hat` | 808-style square-wave closed hat |
| `808_open_hat` | 808-style square-wave open hat |
| `808_cowbell_square` | 808-style square-wave cowbell |

**Noise perc** (from noise_perc):

| Preset | Character |
|--------|-----------|
| `kickish` | Noise-tone hybrid with kick-like low body |
| `snareish` | Noise-heavy snare-like hit |
| `tick` | Sharp high-frequency tick |
| `chh` | Short noise burst for closed hi-hat |
| `clap_noise` | Wide bandpass noise clap |
| `shaped_hit` | Balanced tone+noise with multi-point noise envelope |

**Machinedrum-inspired** (EFM / PI modal / digital character):

| Preset | Character |
|--------|-----------|
| `efm_kick_deep` | Deep EFM kick with slow decay and punchy sweep |
| `efm_kick_punch` | Brighter 909-ish EFM kick with tight attack |
| `efm_snare_bright` | EFM metallic crack snare with white noise wire |
| `efm_cowbell` | Clangy, chirpy EFM cowbell with dual modulators |
| `efm_cymbal_trash` | Chaotic 6-op EFM crash, TR-909 ratio set |
| `efm_cymbal_china` | Splashy 4-op EFM china with bar-mode ratios |
| `pi_tom_membrane` | Physically-modelled membrane tom via modal bank |
| `pi_kick_shell` | Physical kick-drum shell with compressed modes |
| `pi_wood_block` | Wooden block strike via bar-wood mode table |
| `pi_metal_bell` | Metallic bell hit via bar-metal modal bank |
| `pi_glass_ping` | Crystalline glass ping via bar-glass mode table |
| `pi_bowl_shimmer` | Long-decay singing bowl shimmer |
| `kick_bitcrush` | 808 hip-hop kick through bit-crush voice shaper |
| `hat_rate_reduced` | Lo-fi sample-and-held closed-hat via rate-reduce |
| `snare_digital_fuzz` | EFM snare body into asymmetric digital clip |

Notes:

- does not support pitch motion (`freq_trajectory`) at the score level;
  the engine owns its internal pitch sweep
- deterministic for identical inputs (SHA-256 seeded RNG)
- `freq` sets the base pitch; pitch sweep, partial ratios, and noise bandpass
  centers are all relative to this frequency
- when no preset is used and no layer types are specified, `tone_type` defaults
  to `"oscillator"` and all other layers default to `None`
- the engine peak-normalizes internally before applying `amp`

Examples:

Simple preset usage:

```python
score.add_voice(
    "kick",
    synth_defaults={"engine": "drum_voice", "preset": "909_techno"},
    normalize_peak_db=-6.0,
)
```

Hybrid -- start from a preset, add a metallic shimmer layer:

```python
score.add_voice(
    "kick_bell",
    synth_defaults={
        "engine": "drum_voice",
        "preset": "808_house",
        "metallic_type": "partials",
        "metallic_level": 0.15,
        "metallic_decay_s": 0.12,
    },
    normalize_peak_db=-6.0,
)
```

From scratch -- FM body + comb noise + FM cluster metallic:

```python
score.add_voice(
    "hybrid_perc",
    synth_defaults={
        "engine": "drum_voice",
        "exciter_type": "click",
        "exciter_level": 0.12,
        "tone_type": "fm",
        "tone_level": 0.6,
        "tone_fm_ratio": 1.41,
        "tone_fm_index": 3.0,
        "noise_type": "comb",
        "noise_level": 0.3,
        "noise_comb_feedback": 0.4,
        "metallic_type": "fm_cluster",
        "metallic_level": 0.2,
        "metallic_decay_s": 0.1,
    },
    normalize_peak_db=-6.0,
)
```

Using macros for quick timbral control:

```python
score.add_voice(
    "snare",
    synth_defaults={
        "engine": "drum_voice",
        "preset": "909_tight",
        "punch": 0.8,
        "character": 0.4,
    },
    normalize_peak_db=-6.0,
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

## `synth_voice`

Implementation: [code_musics/engines/synth_voice.py](code_musics/engines/synth_voice.py)

Composable tonal synthesizer with four independent, mixable source slots —
`osc`, `partials`, `fm`, `noise` — summed into a shared post-chain (HPF →
dual filter → VCA → voice shaper). Designed for cross-pollination: stack a
supersaw under additive partials with a 2-op FM bell on top, run the result
through a Moog ladder, and get a voice no prior single engine could
produce.

**Prefer `synth_voice` for new tonal composition.** The older engines
(`polyblep`, `va`, `additive`, `fm`, `filtered_stack`, `organ`, `piano`,
`piano_additive`, `harpsichord`) remain registered — existing pieces keep
working — but `synth_voice` is the recommended default for new voices that
don't need the physically-modeled hammer/pluck topology of piano/harpsichord.

### synth_voice architecture

```text
   osc ─→[shaper]─┐
                   │
   partials ─→[sh]─┤
                   ├─×level×env─→ SUM ─→ HPF ─→ FILTER ─→ VCA ─→ VOICE_SHAPER ─→ OUT
   fm ─→[shaper]──┤
                   │
   noise ─→[sh]───┘
```

Any `{slot}_type=None` (or omitted) disables the slot. Slot renderers
compose existing engine primitives rather than reimplementing DSP.

### Slot types

**`osc`** — time-domain bandlimited oscillators

- `polyblep` — single PolyBLEP saw/square/triangle/sine, optional osc2
  layer (`osc2_level`, `osc2_wave`, `osc2_detune_cents`, `osc2_freq_ratio`)
- `supersaw` — Szabo-law 7-voice PolyBLEP bank (`osc_spread_cents` 0–100,
  `osc_mix` 0–1)
- `pulse` — PolyBLEP square with `osc_pulse_width` (0.05–0.95)

**`partials`** — frequency-domain partial banks

- `additive` — harmonic series via `partials_n_harmonics`,
  `partials_harmonic_rolloff`, `partials_brightness_tilt`,
  `partials_odd_even_balance`, or an explicit `partials_partials` list of
  `{"ratio", "amp", "phase"}` dicts
- `spectralwave` — VA saw→formant→square morph via
  `partials_spectral_position` (0..1)
- `drawbars` — Hammond-style 9-stop additive; override ratios via
  `partials_drawbar_ratios` and amps via `partials_drawbar_amps`

**`fm`** — 2-op FM

- `two_op` — carrier + modulator with feedback and index envelope.
  Params: `fm_carrier_ratio`, `fm_ratio` (modulator), `fm_index`,
  `fm_feedback`, `fm_index_decay`, `fm_index_sustain`

**`noise`** — aperiodic texture

- `white` — gaussian white
- `pink` — 1/f-shaped via rfft weighting
- `bandpass` — FFT-bandpassed around `noise_center_hz` (defaults to note
  frequency), `noise_bandwidth_ratio`
- `flow` — Mutable Elements rare-event S&H exciter; `noise_flow_density`
  0..1

### Per-slot common params

Every slot uniformly exposes:

- `{slot}_type` — type selector, `None` disables
- `{slot}_level` — mix level (default 1.0 for tonal slots, 0.1 for noise)
- `{slot}_envelope` — optional multi-point envelope override (falls back
  to the voice ADSR shaping)
- `{slot}_shaper` — optional per-slot nonlinearity (any waveshaper algo,
  `saturation`, or `preamp`); plus `{slot}_shaper_drive` / `_mix`

### Voice post-chain

- **HPF**: `hpf_cutoff_hz` (0 = disabled; CS80/Jupiter-style 2-pole ZDF)
- **Main filter**: `filter_mode` (`lowpass`/`highpass`/`bandpass`/...),
  `filter_topology` (any of the 8 from `_filters.py` —
  svf/ladder/sallen_key/cascade/sem/jupiter/k35/diode), `filter_cutoff_hz`,
  `resonance_q`, `filter_drive`, `filter_morph`, `k35_feedback_asymmetry`,
  `filter_envelope` (multi-point cutoff curve)
- **Voice shaper**: `shaper` (any waveshaper algo, `saturation`, or
  `preamp`), `shaper_drive`, `shaper_mix`, `shaper_mode`, `bit_depth`,
  `reduce_ratio`
- **Voice ADSR**: `attack`, `release` apply simple fades to suppress edge
  clicks; richer amp shaping is owned by the Score layer

### Perceptual macros

Four macros, all default to `None` (inactive). Each fans out to underlying
params via `_set_if_absent` — explicit preset or user values always win.
Resolution order: **preset params → macro fill-in → user kwargs win.**
Macro keys are popped before render-time extraction. Value range
convention: 0.2 subtle, 0.33 clear-but-subtle, 0.5 moderate, 0.66 strong,
0.8–1.0 intense-but-musical.

| Macro | Fans out to |
|---|---|
| `brightness` | `filter_cutoff_hz` (exp-scaled 400→8000 Hz), `partials_brightness_tilt`, `fm_index` and `osc_spread_cents` bumps above 0.5 |
| `movement` | `filter_env_amount`, `partials_smear`, `partials_phase_disperse`, `chorus_mix` |
| `body` | `hpf_cutoff_hz` (inverse — high body pushes HPF down), `resonance_q` bump, `osc2_sub_level`, `partials_odd_even_balance` toward even |
| `dirt` | `shaper` mode (saturation → preamp → hard_clip at thresholds), `shaper_drive`, `shaper_mix`, `feedback_amount` bump above 0.8 |

### synth_voice presets

15 curated presets ship with the engine, split between 5 basic starters
and 10 cross-pollination specialties. Use by name:

```python
synth_defaults={"engine": "synth_voice", "preset": "fm_bell_over_supersaw"}
```

**Basic starters**: `bright_saw_lead`, `warm_pad`, `soft_bass`, `glass_pad`,
`two_op_bell`.

**Cross-pollination specialties**: `fm_bell_over_supersaw`,
`additive_pad_through_ladder`, `drawbar_diode_acid`, `stiff_piano_sub`,
`flow_exciter_pad`, `spectralwave_jupiter`, `chaos_cloud_texture`,
`virus_hybrid_pad`, `formant_vowel_lead`, `tonewheel_drive`.

### Usage example

```python
voice = Voice(
    name="hybrid_pad",
    synth_defaults={
        "engine": "synth_voice",
        # Stack supersaw + additive partials + FM bell + pink noise
        "osc_type": "supersaw",
        "osc_level": 0.5,
        "osc_spread_cents": 20.0,
        "partials_type": "additive",
        "partials_level": 0.4,
        "partials_harmonic_rolloff": 0.75,
        "partials_brightness_tilt": 0.1,
        "fm_type": "two_op",
        "fm_level": 0.3,
        "fm_ratio": 3.0,
        "fm_index": 1.5,
        "fm_index_decay": 0.7,
        "noise_type": "pink",
        "noise_level": 0.08,
        # Post-chain
        "filter_mode": "lowpass",
        "filter_topology": "ladder",
        "filter_cutoff_hz": 1900.0,
        "resonance_q": 0.85,
        "hpf_cutoff_hz": 50.0,
        # Ergonomic macros fill any unset params
        "movement": 0.35,
        "body": 0.45,
    },
)
```

### What does `synth_voice` *not* cover?

Piano, harpsichord, and organ stay as dedicated engines because their
physical topology (hammer-contact / pluck → modal-resonator bank, organ
tonewheel crosstalk) doesn't fit the source→filter→VCA model. Drawbar
*additive spectra* are covered via `partials_type="drawbars"`, but
full Hammond character (key-click, crosstalk, scanner vibrato) remains
in the `organ` engine.

Future extensions tracked in `FUTURE.md`: modal/physical resonator
"operator" slots, 4-op/6-op FM algorithm matrices, a routing matrix for
cross-slot modulation, and "two-of-a-kind" slots for e.g. two supersaws
stacked in one voice.
