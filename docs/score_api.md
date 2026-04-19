# Score API Reference

This document describes the concrete score-domain API in
[code_musics/score.py](code_musics/score.py).

If `docs/composition_api.md` is the higher-level phrase/composition layer and
`docs/synth_api.md` is the engine-facing layer, this file is the middle: the
actual note, phrase, voice, and score objects that most pieces interact with.

## Where This Is Used

- [code_musics/score.py](code_musics/score.py)
- [code_musics/composition.py](code_musics/composition.py)
- [code_musics/render.py](code_musics/render.py)
- [code_musics/humanize.py](code_musics/humanize.py)

## Mental Model

The score model has four main layers:

- `NoteEvent`: one musical event
- `Phrase`: reusable relative-time note material
- `Voice`: one lane of notes plus shared defaults
- `Score`: the full timeline and renderer

The current design is intentionally simple:

- notes are authored in seconds, not MIDI ticks
- pitch is either relative to `f0` via `partial` or absolute via `freq`
- phrases are reusable note collections, not a separate sequencing system
- render-time expression lives mostly on `Score` and `Voice`, not hidden inside engines

## `EffectSpec`

`EffectSpec` is the declarative effect-chain item used by voices, shared send
buses, and the master bus.

Fields:

- `kind: str`
- `params: dict[str, Any] = {}`
- `automation: list[AutomationSpec] = []`

Use it in:

- `Voice.effects`
- `SendBusSpec.effects`
- `Score.master_effects`

Practical note:

- phase-1 effect automation is for score-time wetness control on `mix`, `wet`,
  or `wet_level`; it does not yet automate arbitrary effect internals such as
  delay feedback or compressor threshold

Example:

```python
from code_musics.score import EffectSpec, Score

score = Score(
    f0_hz=110.0,
    master_effects=[
        EffectSpec("saturation", {"preset": "tube_warm", "mix": 0.2}),
    ],
)
```

For supported effect kinds and parameters, see
[docs/synth_api.md](docs/synth_api.md).

## `NoteEvent`

`NoteEvent` is the atomic score event.

Fields:

- `start: float`
- `duration: float`
- `amp: float | None = None`
- `amp_db: float | None = None`
- `velocity: float = 1.0`
- `partial: float | None = None`
- `freq: float | None = None`
- `synth: dict[str, Any] | None = None`
- `label: str | None = None`
- `pitch_motion: PitchMotionSpec | None = None`
- `automation: list[AutomationSpec] | None = None`

Validation and behavior:

- `duration` must be positive
- `start` must be non-negative
- exactly one of `partial` or `freq` must be provided
- `amp` and `amp_db` are mutually exclusive
- `velocity` must be in `(0, 2]`
- if `amp_db` is provided, it is converted to linear `amp` immediately
- if neither `amp` nor `amp_db` is provided, `amp` defaults to `1.0`

Authoring guidance:

- prefer `amp_db` for mix-level choices
- use `velocity` for note-level accents and phrasing
- use `partial` when the note should track `Score.f0_hz`
- use `freq` when the note should stay absolute
- use `synth` for note-local engine overrides or articulation tweaks
- use `automation` for note-local pitch gestures and explicit param motion

Example:

```python
from code_musics.score import NoteEvent
from code_musics.automation import AutomationSegment, AutomationSpec, AutomationTarget

note = NoteEvent(
    start=0.0,
    duration=0.8,
    partial=5 / 4,
    amp_db=-16.0,
    velocity=1.1,
    synth={"attack_scale": 0.8},
    automation=[
        AutomationSpec(
            target=AutomationTarget(kind="pitch_ratio", name="pitch_ratio"),
            segments=(
                AutomationSegment(
                    start=0.0,
                    end=0.8,
                    shape="linear",
                    start_value=1.0,
                    end_value=6 / 5,
                ),
            ),
        )
    ],
)
```

## `Phrase`

`Phrase` is a reusable collection of relative-time `NoteEvent`s.

Field:

- `events: tuple[NoteEvent, ...]`

Key API:

- `Phrase.from_partials(...)`
- `phrase.duration`
- `phrase.transformed(...)`

### `Phrase.from_partials(...)`

Convenience constructor for equally spaced partial-based phrases.

Parameters:

- `partials`
- `note_dur`
- `step`
- `amp`
- `amp_db`
- `velocity`
- `synth_defaults`

This is a quick sketching helper, not the only way to build phrases.

### `phrase.duration`

Returns the duration implied by the latest note endpoint inside the phrase.

### `phrase.transformed(...)`

Returns a new list of placed `NoteEvent`s suitable for insertion into a score.

Parameters:

- `start=0.0`
- `time_scale=1.0`
- `partial_shift=0.0`
- `amp_scale=1.0`
- `reverse=False`

Behavior:

- does not mutate the source phrase
- scales both onset times and durations
- shifts partial-space material by addition
- multiplies resolved linear amplitude
- preserves other note metadata such as `velocity` and `pitch_motion`
- if `reverse=True`, mirrors phrase event placement in time before applying `start`

Use `Phrase` when you want reusable motifs. Use direct `Score.add_note(...)`
when the event is one-off and not worth abstracting.

## `VelocityParamMap`

`VelocityParamMap` linearly maps resolved velocity to a synth parameter range.

Fields:

- `min_value`
- `max_value`
- `min_velocity=0.75`
- `max_velocity=1.25`

Behavior:

- the incoming velocity is clipped into `[min_velocity, max_velocity]`
- the output is linearly interpolated between `min_value` and `max_value`

This is the main score-level bridge from expressive dynamics to timbre.

Example:

```python
from code_musics.score import VelocityParamMap

brightness_map = VelocityParamMap(
    min_value=300.0,
    max_value=2000.0,
    min_velocity=0.8,
    max_velocity=1.2,
)
```

## `Voice`

`Voice` stores note events plus shared defaults and expression settings.

Fields:

- `name`
- `synth_defaults`
- `effects`
- `envelope_humanize`
- `velocity_humanize`
- `velocity_group`
- `velocity_to_params`
- `velocity_db_per_unit`
- `pre_fx_gain_db`
- `mix_db`
- `sends`
- `normalize_lufs`
- `normalize_peak_db`
- `max_polyphony`
- `legato`
- `choke_group`
- `pan`
- `sympathetic_amount`
- `sympathetic_decay_s`
- `sympathetic_modes`
- `drift_bus`
- `drift_bus_correlation`
- `automation`
- `notes`

Important behavior:

- `velocity_humanize` defaults to `VelocityHumanizeSpec()` in `Score.add_voice(...)`
- `pan` must be between `-1.0` and `1.0`
- `velocity_db_per_unit` must be non-negative
- `pre_fx_gain_db` and `mix_db` must be finite when provided
- `normalize_lufs` defaults to `-24.0`; set to `None` only as a last resort (prefer `normalize_peak_db` for percussive voices)
- `normalize_peak_db` defaults to `None`; mutually exclusive with `normalize_lufs`
- `max_polyphony` defaults to `None` (no cap); when provided it must be `>= 1`
- `choke_group` defaults to `None`; when set, all voices sharing the same string tag form a choke group

Practical interpretation:

- `synth_defaults` is the baseline sound
- `effects` is the per-voice processing chain
- `envelope_humanize` is the ADSR variation layer
- `velocity_humanize` is the render-time dynamic variation layer
- `velocity_group` links multiple voices into a shared velocity-drift family
- `velocity_to_params` makes louder/softer notes timbrally different
- `pre_fx_gain_db` is the voice input trim before voice effects; use it when you want to hit chorus, saturation, compression, or reverb harder or softer without changing the note writing
- `mix_db` is a **post-fader channel level** (like a mixing console fader), not a wet/dry mix ratio — it controls how loud this voice is in the final stereo bus, in dB, after normalization and voice effects have been applied; defaults to `0.0` (unity gain); use it **only for mix balance**, not for gain staging; for wet/dry control on effects, use effect-level `mix` or `wet` parameters on `EffectSpec` instead
- `sends` routes the post-fader voice signal into one or more shared send buses
- `normalize_lufs` applies an integrated-LUFS stem gain trim before `pre_fx_gain_db`, pan, voice effects, and `mix_db`; the default `-24.0` is the right choice for all tonal, melodic, and sustained voices
- `normalize_peak_db` is the alternative for percussive/transient voices (kicks, toms, noise hits): it normalizes the voice to a target peak level before effects, making compressor thresholds and effect drive predictable regardless of BPM or individual note `amp_db` values — use `-6.0` as the standard target when pairing with the `kick_punch` or `kick_glue` compressor presets
- `normalize_lufs` and `normalize_peak_db` are mutually exclusive; raise if both are set
- `max_polyphony=1` gives strict mono note allocation for the voice by truncating any currently sounding note when a new note claims the last slot
- `legato=True` only matters when `max_polyphony=1`: overlapping note transitions skip the new note's attack retrigger, giving a simple mono-legato glide behavior without continuous oscillator state carryover
- `pan` places the rendered voice in stereo
- `sympathetic_amount` controls the level of sympathetic resonance added to the voice; `0.0` (default) disables it
- `sympathetic_decay_s` sets the decay time in seconds for sympathetic resonator ringing; default `2.0`
- `sympathetic_modes` sets how many harmonic modes per note are used as resonator frequencies; default `8`
- `drift_bus` is the name of a shared `DriftBusSpec` on the score that this voice subscribes to; default `None` (no bus). When set, the bus name must be registered via `Score.add_drift_bus(...)`
- `drift_bus_correlation` controls how much of the voice's slow pitch drift comes from the shared bus vs. the engine's independent drift; `1.0` (default) is fully shared, `0.0` is fully independent; must be in `[0, 1]`
- `automation` adds explicit score-time parameter lanes beyond humanization
- in phase 1, `Voice.automation` can target synth params, `pitch_ratio`, and
  control surfaces such as `pan`, `pre_fx_gain_db`, and `mix_db`

### Sympathetic Resonance

`sympathetic_amount`, `sympathetic_decay_s`, and `sympathetic_modes` control an
optional resonator bank that adds sympathetic ringing to a voice.

Parameters:

- `sympathetic_amount: float = 0.0`
  Level of sympathetic resonance mixed into the voice. `0.0` disables the
  feature entirely.
- `sympathetic_decay_s: float = 2.0`
  Decay time in seconds for each resonator mode.
- `sympathetic_modes: int = 8`
  Number of harmonic modes per note used as resonator frequencies.

How it works:

- for each note in the voice, harmonic mode frequencies (`note_freq * k` for
  `k = 1..sympathetic_modes`) are collected, deduplicated (modes within 1% are
  merged), and capped at 64 total resonators
- the voice's mixed signal is analyzed at each mode frequency via windowed
  correlation to measure excitation energy
- decaying sinusoids are synthesized at the excited mode frequencies and summed
- the resonance sum is peak-normalized relative to the input signal and mixed in
  at `sympathetic_amount`

Pipeline position: sympathetic resonance is applied after note mixing but before
voice normalization (`normalize_lufs` / `normalize_peak_db`), so the resonance
tail is gain-staged alongside the dry voice signal.

Sympathetic resonance is only applied to native per-note engines (additive, fm,
harpsichord, piano, etc.). Instrument-engine voices (e.g., `surge_xt`) return
pre-mixed audio from a plugin, so the resonator bank cannot operate on individual
notes. Setting `sympathetic_amount > 0` on an instrument-engine voice emits a
warning and is ignored.

The feature is most effective with harmonically related notes where the resonator
modes reinforce each other naturally. Good candidates include harpsichord, piano,
and other plucked or struck string voices.

Example:

```python
score.add_voice(
    "harpsichord",
    synth_defaults={"engine": "harpsichord", "preset": "baroque"},
    sympathetic_amount=0.15,
    sympathetic_decay_s=2.5,
    sympathetic_modes=6,
)
```

## `VoiceSend`

`VoiceSend` routes a voice into a named shared send bus.

Fields:

- `target: str`
- `send_db: float = 0.0`
- `automation: list[AutomationSpec] = []`

Behavior:

- `target` must be a non-empty send-bus name defined on the score
- `send_db` must be finite
- sends are post-fader in v1, so they follow the voice's `mix_db`
- sends tap the voice after normalization, `pre_fx_gain_db`, pan, and `effects`
- in normal mixing, treat `send_db` plus the voice's post-fader level as the main reverb/delay balance controls
- `automation` can ride `send_db` over score time

## `SendBusSpec`

`SendBusSpec` defines a shared aux return bus on the score.

Fields:

- `name: str`
- `effects: list[EffectSpec] = []`
- `return_db: float = 0.0`
- `pan: float = 0.0`
- `automation: list[AutomationSpec] = []`

Behavior:

- send-bus names must be unique within a score
- `return_db` must be finite
- `pan` must be in `[-1.0, 1.0]`
- buses are meant for shared return-style processing such as reverb, delay, or chorus
- prefer leaving `return_db` at `0.0` in normal use so the wet level is determined by the post-fader source and each voice's `send_db`
- treat `return_db` as an uncommon escape hatch for global aux-return trim, not the primary way to balance how audible a send effect is
- if you heavily attenuate both `send_db` and `return_db`, the shared return can become effectively inaudible and harder to reason about
- `automation` can ride `return_db` and `pan` over score time

## `DriftBusSpec`

`DriftBusSpec` defines a shared slow pitch-drift bus on the score. Voices set
`drift_bus=<name>` and `drift_bus_correlation=<x>` to subscribe; the bus output
is mixed into their per-note frequency trajectories in log-cents space.

Fields:

- `name: str`
- `rate_hz: float = 0.2` — characteristic drift rate; 0.05-0.5 Hz is the musically useful range
- `depth_cents: float = 5.0` — RMS excursion in cents; 2-12 is the subtle-to-breathing range
- `seed: int | None = None` — deterministic seed

Behavior:

- bus names must be unique within a score (enforced by `Score.drift_buses` being a dict)
- `rate_hz` must be positive, `depth_cents` must be non-negative
- the bus runs independently of audio sample rate; it produces a single slow random-walk trajectory that all subscribed voices sample at their respective absolute note times
- at `Voice.drift_bus_correlation=1.0` the bus replaces the engine's independent pitch drift entirely; at `0.0` it is ignored; intermediate values blend (engine `pitch_drift` is scaled by `(1 - correlation)` and the bus is applied as `bus_ratio ** correlation` to the frequency trajectory)
- implementation follows Surge XT's published `DriftLFO` algorithm (single-pole filtered uniform noise with RMS-preserving gain compensation); re-written from the description, not copied

## `Score`

`Score` is the top-level composition container and renderer.

Fields:

- `f0: float`
- `sample_rate: int = synth.SAMPLE_RATE`
- `timing_humanize: TimingHumanizeSpec | None = None`
- `auto_master_gain_stage: bool = True`
- `master_bus_target_lufs: float = -24.0`
- `master_bus_max_true_peak_dbfs: float = -6.0`
- `master_input_gain_db: float = 0.0`
- `master_effects: list[EffectSpec] = []`
- `send_buses: list[SendBusSpec] = []`
- `drift_buses: dict[str, DriftBusSpec] = {}`
- `voices: dict[str, Voice] = {}`

### `Score.add_voice(...)`

Adds or replaces a named voice definition.

Parameters:

- `name`
- `synth_defaults`
- `effects`
- `envelope_humanize`
- `velocity_humanize`
- `velocity_group`
- `velocity_to_params`
- `velocity_db_per_unit`
- `pre_fx_gain_db`
- `mix_db`
- `sends`
- `normalize_lufs`
- `normalize_peak_db`
- `max_polyphony`
- `legato`
- `choke_group` — optional string tag; voices sharing a choke group cut each other on note onset (see below)
- `pan`
- `sympathetic_amount` — strength of sympathetic resonance; `0.0` (default) disables it
- `sympathetic_decay_s` — decay time in seconds for sympathetic resonator ringing; default `2.0`
- `sympathetic_modes` — number of harmonic modes per note used as resonator frequencies; default `8`
- `drift_bus` — optional name of a shared `DriftBusSpec` registered on the score via `add_drift_bus(...)`; default `None` (no shared drift)
- `drift_bus_correlation` — fraction of the voice's slow pitch drift coming from the shared bus vs. the engine's independent drift; `1.0` (default) is fully shared, `0.0` is fully independent; must be in `[0, 1]`
- `automation` — voice-level score-time automation specs; default `None`

Important behavior:

- calling `add_voice(...)` with an existing name replaces that voice definition
- `velocity_humanize=None` in the method call currently means "use the default subtle humanizer", not "disable velocity humanization"
- if you want to disable velocity humanization after a voice exists, set `voice.velocity_humanize = None`
- `pre_fx_gain_db` defaults to `0.0` and acts like a pre-insert trim
- `mix_db` defaults to `0.0` (unity gain) and acts like a post-insert channel fader in dB — it sets how loud the voice is in the final stereo bus; use it for mix balance only, not gain staging; despite the name, it is **not** a wet/dry mix ratio — for effect wet/dry control, use `mix` or `wet` params on `EffectSpec`
- `sends` defaults to `[]` and routes the post-fader voice into named shared aux buses
- `normalize_lufs=-24.0` (default) handles gain staging for all tonal voices — leave it at the default and use `mix_db` to balance
- use `normalize_peak_db=-6.0` for percussive voices (kicks, toms, noise hits) instead of `normalize_lufs`; this gives effects a predictable input level regardless of BPM or note-level `amp_db` variation
- `normalize_lufs` and `normalize_peak_db` are mutually exclusive
- `max_polyphony=1` is the strict-mono setting for basses, leads, and other voices where overlap smear is unwanted
- with `max_polyphony=1`, `legato=True` suppresses the attack retrigger on overlapped note changes
- `choke_group` assigns the voice to a named choke group; when any voice in the
  group plays a note, all other voices in the same group are faded out with a
  10 ms linear ramp at that onset time — the classic use case is open/closed
  hi-hat pairs where a closed hit silences a ringing open hit
- `choke_group=None` (default) means the voice is not in any choke group

That second-to-last point is easy to miss and worth being explicit about.

Example:

```python
from code_musics.humanize import EnvelopeHumanizeSpec, VelocityHumanizeSpec
from code_musics.score import Score, VelocityParamMap

score.add_voice(
    "lead",
    synth_defaults={
        "engine": "filtered_stack",
        "preset": "reed_lead",
        "env": {"attack_ms": 30.0, "release_ms": 280.0},
    },
    envelope_humanize=EnvelopeHumanizeSpec(preset="loose_pluck", seed=9),
    velocity_humanize=VelocityHumanizeSpec(preset="subtle_living", seed=9),
    velocity_to_params={
        "cutoff_hz": VelocityParamMap(
            min_value=800.0,
            max_value=2600.0,
            min_velocity=0.85,
            max_velocity=1.2,
        )
    },
    pre_fx_gain_db=2.0,
    mix_db=-3.0,
    max_polyphony=1,
    legato=True,
    pan=-0.15,
)
```

### `Score.add_send_bus(...)`

Adds or replaces a named shared send bus definition.

Parameters:

- `name`
- `effects`
- `return_db`
- `pan`

Use shared send buses when several voices should feed the same reverb, delay,
or modulation return instead of duplicating similar insert chains per voice.
In normal authoring, prefer balancing the effect with voice `mix_db` and per-voice
`send_db`, leaving `return_db=0.0` unless you explicitly want a global return trim.

### `Score.add_drift_bus(...)`

Adds or replaces a named shared pitch-drift bus definition. Voices that set
`drift_bus=<name>` receive a correlated slow pitch-drift signal; the amount of
correlation is controlled per voice via `drift_bus_correlation`.

Parameters:

- `name: str` — bus name referenced by `Voice.drift_bus`
- `rate_hz: float = 0.2` — characteristic drift rate; musically useful range is 0.05-0.5 Hz
- `depth_cents: float = 5.0` — RMS excursion in cents; 2-12 is the subtle-to-breathing range
- `seed: int | None = None` — deterministic seed; omitting it still produces the same output across identical scores, but varying `seed` picks a different random-walk trajectory

Behavior:

- the bus produces a single shared slow random-walk signal; every subscribing voice samples it at their note's absolute score time
- per-voice `drift_bus_correlation=1.0` means the engine's independent pitch drift is fully replaced by the bus; `0.0` means the bus is entirely ignored and the voice keeps its original independent drift; intermediate values blend linearly in cents space (equivalent to `bus_ratio ** correlation` multiplied into the freq trajectory, with the engine's internal `pitch_drift` parameter scaled by `(1 - correlation)`)
- the bus is MIT-licensed and re-implemented from Surge XT's published `DriftLFO` algorithm (single-pole filtered uniform noise with gain compensation)

Example:

```python
score = Score(f0_hz=220.0)
score.add_drift_bus("ensemble", rate_hz=0.2, depth_cents=6.0, seed=17)
for voice_name in ("lead", "alto", "bass"):
    score.add_voice(
        voice_name,
        synth_defaults={"engine": "polyblep", "waveform": "saw"},
        drift_bus="ensemble",
        drift_bus_correlation=0.7,  # mostly shared, with a hint of individual wobble
    )
```

### `Score.get_voice(name)`

Returns an existing voice or creates a blank `Voice(name=name)` if missing.

This is the escape hatch when you want to mutate a voice directly:

```python
voice = score.get_voice("lead")
voice.velocity_humanize = None
```

### `Score.add_note(...)`

Adds one note to a named voice and returns the created `NoteEvent`.

Parameters:

- `voice_name`
- `start`
- `duration`
- `partial` or `freq`
- `amp` or `amp_db`
- `velocity`
- `pitch_motion`
- `synth`
- `label`

If the voice does not exist yet, it is created with default settings.

### `Score.add_phrase(...)`

Places a `Phrase` onto a voice using the same transform surface as
`Phrase.transformed(...)`.

Parameters:

- `voice_name`
- `phrase`
- `start`
- `time_scale`
- `partial_shift`
- `amp_scale`
- `reverse`

Use this for phrase-first composition. It returns the placed `NoteEvent`s.

### `Score.total_dur`

Derived property returning the latest note endpoint across all voices.

This value drives:

- overall render duration
- time normalization for humanization drift sampling

### `Score.render()`

Renders the full score, including voice effects, shared send returns, and master effects.

Behavior:

- returns an empty mono array if the score has no rendered voices
- may return mono or stereo depending on pan and effects
- voice insert effects are resolved after each voice's dry/base render is built, so
  native compressor sidechains can read other voices' final post-everything outputs
- when `auto_master_gain_stage=True`, raises or lowers the summed post-fader mix
  toward `master_bus_target_lufs` before the master bus while keeping premaster
  true peak under `master_bus_max_true_peak_dbfs`
- applies `master_input_gain_db` to the summed mix before `master_effects`
- mixes shared send returns into the premaster mix before the master bus
- applies `master_effects` after the voices are mixed
- does not perform export mastering itself; the named-piece render workflow applies
  final LUFS/true-peak mastering when writing the final WAV
- export mastering first drives the mix toward the render LUFS target with the
  limiter, then uses any remaining true-peak headroom up to the export ceiling

Practical interpretation:

- leave `auto_master_gain_stage=True` in normal work so voice faders behave like
  balance controls rather than ad hoc premaster loudness controls
- use `master_bus_target_lufs` and `master_bus_max_true_peak_dbfs` only when you
  intentionally want a different premaster operating level for the score
- leave `master_input_gain_db` at `0.0` in normal work; the default gain staging
  should usually be musically reasonable without touching it
- use `master_input_gain_db` only when you intentionally want to hit the master
  bus saturation, compression, tape, or reverb a bit harder or softer
- do not treat `master_input_gain_db` as the normal delivery loudness control;
  export LUFS targeting handles the final file level later

### `Score.extract_window(...)`

Builds a score containing only notes that can sound within a requested time
window.

Parameters:

- `start_seconds`
- `end_seconds`

Behavior:

- keeps notes whose authored time range overlaps the requested window
- shifts kept notes so the extracted window starts at local time `0`, clamping
  any earlier overlap to local time zero
- preserves the original global time context for timing humanization, envelope
  humanization, and voice automation
- is the score-domain helper used by snippet rendering

### `Score.render_stems()`

Renders each voice independently before shared send returns and master-bus effects.

Returns:

- `dict[str, np.ndarray]`

Useful for:

- debugging arrangement balance
- inspecting per-voice rendering
- feeding into `export_stem_bundle()` for per-voice WAV export

### `Score.render_for_stem_export(dry=False)`

Renders all components needed for audio stem WAV export in a single pass.

Returns `(voice_stems, send_returns, mix_audio)`:

- `voice_stems: dict[str, np.ndarray]` — wet (post-effects/pan/fader) or dry
  (post-normalization, pre-effects/pan, mono) depending on the `dry` flag
- `send_returns: dict[str, np.ndarray]` — mixed bus returns; empty dict if `dry=True`
- `mix_audio: np.ndarray` — always the full wet mix with master-bus processing

In wet mode, `sum(voice_stems) + sum(send_returns) ≈ pre-master mix`.

Used by `export_stem_bundle()` in `code_musics/stem_export.py`. See
`make stems PIECE=...` for the CLI workflow.

### `Score.resolve_timing_offsets()`

Returns the deterministic render-time timing offset for each note key after
applying `Score.timing_humanize`.

Returns:

- `dict[tuple[str, int], float]`

Use this when you need to inspect how far notes drifted from authored score
time without rendering audio.

### `Score.resolved_timing_notes()`

Returns resolved timing snapshots for all score notes after score-level timing
humanization.

Each snapshot includes:

- voice name and note index
- authored start time
- resolved start and end time
- resolved timing offset
- duration
- resolved frequency and optional partial
- optional label

Use this for timestamp inspection, timeline export, and timing-drift analysis.

### `Score.plot_piano_roll(...)`

Plots score events as a piano-roll style view.

Parameters:

- `path: str | Path | None = None`

Behavior:

- uses partial-space vertical placement when notes have `partial`
- uses `freq / f0` when notes have absolute frequency
- saves the figure if `path` is provided
- returns `(figure, axis)`

## Modulation Matrix

`code_musics/modulation.py` implements a Vital-style per-connection
modulation matrix.  Every routing is a first-class `ModConnection`
object with `amount`, `bipolar`, `stereo`, `power`, optional
`breakpoints`, and a combine `mode`.  Connections attach at voice
scope (`Voice.modulations`) or score scope (`Score.modulations`).

### Sources

All sources expose a common `sample(times, context) -> np.ndarray`
interface; each subclass advertises its natural output domain
(bipolar `[-1, 1]` or unipolar `[0, 1]`).

| Source | Output | Per-note | Notes |
|---|---|---|---|
| `LFOSource(rate_hz, waveshape, phase_rad, retrigger, seed)` | bipolar | optional | waveshapes: `sine`, `triangle`, `saw_up`, `saw_down`, `square`, `smoothed_random` |
| `EnvelopeSource(attack, hold, decay, sustain, release, *_power)` | unipolar | always | triggered at note onset; mirrors synth ADSR curve powers |
| `MacroSource(name)` | unipolar | shared | resolved via `Score.add_macro(name, default, automation)` |
| `VelocitySource(velocity_scale)` | unipolar | always | wraps the resolved (humanized) note velocity |
| `RandomSource(rate_hz, retrigger, seed)` | bipolar | optional | seeded sample-and-hold |
| `ConstantSource(value)` | bipolar | no | used for stereo pan-split tricks (`value=1.0`, `amount=+0.55` on one voice, `-0.55` on another) |
| `DriftAdapter(style, rate_hz, smoothness, seed)` | bipolar | no | reuses `humanize.DriftSpec` curves so drift sources are routable |

### `ModConnection`

```python
@dataclass(frozen=True)
class ModConnection:
    source: ModSource
    target: AutomationTarget       # same (kind, name) as AutomationSpec
    amount: float = 1.0
    bipolar: bool = True           # False rectifies bipolar sources to >= 0
    stereo: bool = False           # reserved for stereo-aware destinations
    power: float = 0.0             # Vital sign-magnitude curve, [-20, 20]
    breakpoints: tuple[tuple[float, float], ...] | None = None
    mode: AutomationMode = "add"   # "replace" | "add" | "multiply"
    name: str | None = None        # inspection label
```

Shaping pipeline applied per connection:
`raw = source.sample(...)` -> optional bipolar rectification ->
`power` curve -> `breakpoints` -> `amount` scaling.

### Combine order at a destination

When multiple connections target the same destination, evaluation
proceeds:

1. existing `AutomationSpec` lanes apply to the base value first
   (unchanged from the pre-matrix behavior);
2. matrix contributions are then combined in order: all `replace`
   connections (last wins), then `multiply`, then `add`.

This matches Vital's precedence: a hard override fully replaces, then
envelope/LFO scaling, then additive LFO/macro bias.

### Destination namespace

Targets reuse `AutomationTarget`:

- `kind="synth"` — any name in
  `_SUPPORTED_SYNTH_AUTOMATION_PARAMS`;
- `kind="control"` — `mix_db`, `pan`, `send_db`, `return_db`,
  `pre_fx_gain_db`, `wet`, `mix`, `wet_level`;
- `kind="pitch_ratio"` — the dedicated `pitch_ratio` target.

### Time resolution

Destinations decide the time resolution:

- **Per-sample** — all control targets, `pitch_ratio`, and the
  per-sample synth whitelist below.
- **Per-note scalar** — every other synth target is sampled at note
  onset and folded into the synth params dict alongside
  `apply_synth_automation`.

The **per-sample synth whitelist** for MVP is a single destination:
`cutoff_hz` on the `polyblep` engine, via the engine's
`param_profiles` kwarg.  Engines opt in explicitly via
`register_param_profile_support(engine_name)`; engines that don't opt
in silently ignore profiles and use the scalar value.

See `FUTURE.md` under "Modulation architecture" for the deferred
per-sample destinations.

### Macros

```python
score.add_macro("brightness", default=0.5, automation=<AutomationSpec>)
```

Macros are shared `[0, 1]` scalars resolved via `MacroSource(name)`.
The optional `automation` must target `kind="control"` and can ride
the timeline the same way voice/score automations do.

### `Score.describe_modulations()`

Returns a flat list of dicts summarizing every registered connection
(scope, source type, destination, shaping fields).  Useful for
inspection / analysis without having to walk the voice graph.

### Example

```python
score = Score(f0_hz=174.614, master_effects=DEFAULT_MASTER_EFFECTS)
score.add_macro("brightness", default=0.0, automation=AutomationSpec(
    target=AutomationTarget(kind="control", name="mix"),
    segments=(AutomationSegment(
        start=0.0, end=30.0, shape="linear",
        start_value=0.0, end_value=1.0,
    ),),
))
score.add_voice(
    "lead",
    synth_defaults={"engine": "polyblep", "cutoff_hz": 1400.0},
    modulations=[
        ModConnection(
            source=LFOSource(rate_hz=0.7, waveshape="sine"),
            target=AutomationTarget(kind="synth", name="cutoff_hz"),
            amount=600.0, power=-4.0, mode="add",
        ),
        ModConnection(
            source=VelocitySource(),
            target=AutomationTarget(kind="synth", name="resonance_q"),
            amount=0.6, bipolar=False,
            breakpoints=((0.0, 0.0), (0.6, 0.15), (1.0, 1.0)),
            mode="add",
        ),
    ],
)
```

See `code_musics/pieces/mod_matrix_study.py` for a full end-to-end
demo piece covering LFO, Macro, Velocity, DriftAdapter, and
stereo-split `ConstantSource` use cases.

## Render-Time Resolution Order

The practical render path for one note is:

1. merge `voice.synth_defaults` with note-level `synth`
2. resolve presets and engine params
3. build resolved velocity from note velocity and any velocity humanization
4. map resolved velocity into synth params via `velocity_to_params`
5. resolve base frequency from `partial` or `freq`
6. build pitch trajectory if `pitch_motion` is present
7. render the raw engine signal
8. convert velocity into a dB offset using `velocity_db_per_unit`
9. resolve ADSR with `envelope_humanize`
10. place the note in time using `timing_humanize`
11. if `max_polyphony` is set, apply voice-level note allocation; in strict mono this can truncate older notes, and `legato=True` suppresses attack retriggers on overlapped note changes
12. mix notes into the dry voice stem
13. if `sympathetic_amount > 0`, apply sympathetic resonance to the mixed voice signal
14. if `normalize_lufs` is set, apply a uniform gain trim toward that target integrated loudness; if `normalize_peak_db` is set instead, normalize to that peak level — these are mutually exclusive
15. apply `pre_fx_gain_db`
16. apply pan to produce the dry/base voice render
17. resolve voice effects, including any named voice sidechains for native compressors
18. apply `mix_db`
19. derive any post-fader voice sends
20. sum dry voices and separately sum/process shared send returns
21. mix dry voices and send returns together
22. if enabled, auto-stage the premaster mix for the master bus
23. apply `master_input_gain_db`
24. apply master effects

That ordering matters because it explains why:

- timing humanization changes placement but not note duration
- velocity can affect both loudness and timbre
- render analysis now also emits artifact-risk warnings for suspicious rendered
  outcomes and risky parameter surfaces, so bright/unstable settings are easier
  to catch before they become mystery listening bugs

## Artifact-Risk Guidance

The render workflow now emits warning-only artifact-risk diagnostics into the
analysis manifest and render logs. Treat these as guardrails, not hard errors.

High-value risky surfaces:

- `velocity_to_params["cutoff_hz"]`
  Safe: compact expressive spans, often a few hundred Hz to around 800 Hz.
  Risky: large spans above ~1000 Hz, especially on already-bright leads.
- `velocity_to_params["filter_env_amount"]`
  Safe: subtle variation around an already-moderate envelope.
  Risky: very wide spans that make accents sound like different presets.
- authored note `amp_db`
  Safe: conservative note levels when also sweeping cutoff upward.
  Risky: combining hotter note levels with bright cutoff ramps and aggressive
  filter motion.
- note-level `synth` overrides that sweep `cutoff_hz`
  Safe: gradual, moderate movement.
  Risky: large section-long sweeps stacked with velocity-driven cutoff motion.

Common failure mode:

- large cutoff sweep + strong `filter_env_amount` + wide velocity-driven cutoff
  mapping + elevated drive/resonance can produce divebomb, wah, tremolo-like,
  or brittle artifacts even when each parameter looks individually plausible
- envelope humanization operates on the post-merge ADSR values
- master effects do not affect the individual stem renders

## Recommended Usage

Prefer this split when composing:

- `Phrase` and `line(...)` for reusable musical material
- `Score.add_phrase(...)` for most placement
- `Score.add_note(...)` for accents, pedals, transitions, and exceptions
- `amp_db` for balance
- `velocity` for local expression
- `timing_humanize` for ensemble feel
- `envelope_humanize` for subtle ADSR life
- `velocity_group` when multiple voices should breathe together

## Related Docs

- [docs/composition_api.md](docs/composition_api.md)
- [docs/synth_api.md](docs/synth_api.md)
