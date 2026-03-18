# Score API Reference

This document describes the concrete score-domain API in
[code_musics/score.py](/home/jan/workspace/code-musics/code_musics/score.py).

If `docs/composition_api.md` is the higher-level phrase/composition layer and
`docs/synth_api.md` is the engine-facing layer, this file is the middle: the
actual note, phrase, voice, and score objects that most pieces interact with.

## Where This Is Used

- [code_musics/score.py](/home/jan/workspace/code-musics/code_musics/score.py)
- [code_musics/composition.py](/home/jan/workspace/code-musics/code_musics/composition.py)
- [code_musics/render.py](/home/jan/workspace/code-musics/code_musics/render.py)
- [code_musics/humanize.py](/home/jan/workspace/code-musics/code_musics/humanize.py)

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

`EffectSpec` is the declarative effect-chain item used by voices and the master
bus.

Fields:

- `kind: str`
- `params: dict[str, Any] = {}`

Use it in:

- `Voice.effects`
- `Score.master_effects`

Example:

```python
from code_musics.score import EffectSpec, Score

score = Score(
    f0=110.0,
    master_effects=[
        EffectSpec("saturation", {"preset": "tube_warm", "mix": 0.2}),
    ],
)
```

For supported effect kinds and parameters, see
[docs/synth_api.md](/home/jan/workspace/code-musics/docs/synth_api.md).

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
- use `partial` when the note should track `Score.f0`
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
- `normalize_lufs`
- `pan`
- `automation`
- `notes`

Important behavior:

- `velocity_humanize` defaults to `VelocityHumanizeSpec()` in `Score.add_voice(...)`
- `pan` must be between `-1.0` and `1.0`
- `velocity_db_per_unit` must be non-negative
- `normalize_lufs` defaults to `-24.0` and can be set to `None` to disable stem auto-normalization

Practical interpretation:

- `synth_defaults` is the baseline sound
- `effects` is the per-voice processing chain
- `envelope_humanize` is the ADSR variation layer
- `velocity_humanize` is the render-time dynamic variation layer
- `velocity_group` links multiple voices into a shared velocity-drift family
- `velocity_to_params` makes louder/softer notes timbrally different
- `normalize_lufs` applies an integrated-LUFS stem gain trim before pan, voice effects, and the final mix
- `pan` places the rendered voice in stereo
- `automation` adds explicit score-time parameter lanes beyond humanization

## `Score`

`Score` is the top-level composition container and renderer.

Fields:

- `f0: float`
- `sample_rate: int = synth.SAMPLE_RATE`
- `timing_humanize: TimingHumanizeSpec | None = None`
- `master_effects: list[EffectSpec] = []`
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
- `normalize_lufs`
- `pan`

Important behavior:

- calling `add_voice(...)` with an existing name replaces that voice definition
- `velocity_humanize=None` in the method call currently means "use the default subtle humanizer", not "disable velocity humanization"
- if you want to disable velocity humanization after a voice exists, set `voice.velocity_humanize = None`
- `normalize_lufs=None` disables the default per-voice auto-normalization if you want raw stem gain instead

That second point is easy to miss and worth being explicit about.

Example:

```python
from code_musics.humanize import EnvelopeHumanizeSpec, VelocityHumanizeSpec
from code_musics.score import Score, VelocityParamMap

score.add_voice(
    "lead",
    synth_defaults={"engine": "filtered_stack", "preset": "reed_lead"},
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
    pan=-0.15,
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

Renders the full score, including voice effects and master effects.

Behavior:

- returns an empty mono array if the score has no rendered voices
- may return mono or stereo depending on pan and effects
- applies `master_effects` after the voices are mixed
- does not perform export mastering itself; the named-piece render workflow applies
  final LUFS/true-peak mastering when writing the final WAV

### `Score.render_stems()`

Renders each voice independently before master-bus effects.

Returns:

- `dict[str, np.ndarray]`

Useful for:

- debugging arrangement balance
- inspecting per-voice rendering
- exporting stems later if that becomes a workflow

### `Score.plot_piano_roll(...)`

Plots score events as a piano-roll style view.

Parameters:

- `path: str | Path | None = None`

Behavior:

- uses partial-space vertical placement when notes have `partial`
- uses `freq / f0` when notes have absolute frequency
- saves the figure if `path` is provided
- returns `(figure, axis)`

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
11. mix notes into the dry voice stem
12. if `normalize_lufs` is set, apply a uniform gain trim toward that target integrated loudness
13. apply pan and voice effects
14. mix voices together, then apply master effects

That ordering matters because it explains why:

- timing humanization changes placement but not note duration
- velocity can affect both loudness and timbre
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

- [docs/composition_api.md](/home/jan/workspace/code-musics/docs/composition_api.md)
- [docs/synth_api.md](/home/jan/workspace/code-musics/docs/synth_api.md)
