# Repository Guide

## Overview

- `code_musics/` is the main package.
- `docs/synth_api.md` documents synth engines, presets, and engine-specific params.
- `docs/score_api.md` documents the concrete `NoteEvent` / `Phrase` / `Voice` /
  `Score` API, render-time expression controls, and the render path.
- `docs/composition_api.md` documents the higher-level composition helpers plus
  phrase-building helpers that sit above the core score model.
- `code_musics/synth.py` contains low-level DSP, rendering, and effect helpers.
- `code_musics/score.py` contains the composition abstractions: `NoteEvent`,
  `Phrase`, `Voice`, `Score`, `EffectSpec`, and `VelocityParamMap`.
- `code_musics/humanize.py` contains timing, envelope, and velocity humanization
  specs plus the drift helpers they build on.
- `code_musics/engines/` contains the synth engine registry and per-engine renderers.
- `code_musics/tuning.py` contains small just-intonation, harmonic-series, utonal,
  and EDO helper functions.
- `code_musics/pieces/` contains named musical works that can be rendered by the
  registry, including smaller themed study modules plus JI subpackages.
- `code_musics/render.py` is the named-piece orchestration layer.
- `code_musics/midi_export.py` exports score-backed pieces as shared tuning files plus per-voice MIDI stems.
- `code_musics/meter.py` contains the optional high-level musical-time layer:
  `Timeline`, beat/bar helpers, rhythmic values, and bar-aware location math.
- `main.py` is the main entrypoint for listing and rendering pieces.

## Composition Model

- `NoteEvent` is the atomic musical event. It stores timing, amplitude, and either a
  `partial` relative to `Score.f0` or an absolute `freq`.
- `NoteEvent` also supports per-note `velocity`, `amp_db`, optional
  `pitch_motion`, and optional note-local automation.
- `Phrase` is a reusable collection of relative-time `NoteEvent`s. Use it for motifs,
  sequences, and transformed restatement.
- `Score.add_note(...)` is the escape hatch for one-off pedals, accents, blooms, and
  transitions.
- `Score.add_phrase(...)` is the main composing API. It supports placement transforms
  like `time_scale`, `partial_shift`, `amp_scale`, and `reverse`.
- `Voice` stores note events plus synth defaults, voice-level effects, pan,
  humanization settings, optional velocity-to-parameter mappings, and optional
  voice-time automation.
- `EffectSpec`, `VoiceSend`, and `SendBusSpec` also support score-time automation
  for wet/mix, send level, return level, and pan control surfaces.
- `Score` also supports shared named send buses so multiple voices can feed the
  same reverb, delay, or other return-style effect chain without duplicating
  insert effects per voice.
- When using shared send buses, usually leave the bus `return_db` at `0.0` and
  balance audibility with the voice fader plus per-voice `send_db`; treat
  `return_db` as a rare global-return trim rather than the main wet-level knob.
- `Score` owns the timeline, derives `total_dur`, renders audio, and can save a
  piano-roll plot.
- `Score.timing_humanize` applies render-time ensemble timing drift across the whole
  score, while voice-level humanizers shape envelope and velocity variation.
- `Voice.synth_defaults` and note-level `synth={...}` overrides accept an `engine`
  name, optional `preset`, and engine-specific params documented in `docs/synth_api.md`.
- Voices can now enforce `max_polyphony` (including strict mono with `1`) and an
  optional simple `legato` mode for overlap-driven non-retrigger behavior.
- `code_musics.composition` includes phrase-first helpers for melodic writing and
  section building. High-level examples: `line(...)` / `ratio_line(...)` for phrase
  creation, `concat(...)` / `overlay(...)` / `echo(...)` for phrase transforms,
  `sequence(...)` / `canon(...)` for repeated placement, and
  `voiced_ratio_chord(...)` / `progression(...)` for harmonic writing. The optional
  high-level timing layer adds `Timeline`, rhythmic values like `Q` / `E`, optional
  `SwingSpec` feel control for eighth- or sixteenth-note swing, and grid-style
  helpers such as `grid_line(...)`, `grid_sequence(...)`, `grid_canon(...)`,
  `metered_sections(...)`, and `bar_automation(...)` for bar-aware voice timbre
  arcs that compile back down to the existing seconds-based score model. Full API
  details live in
  `docs/composition_api.md`.
- For detailed score-surface semantics, parameter meanings, and render-order
  behavior, read `docs/score_api.md`.

## Expression Model

- Prefer `amp_db` over raw `amp` for authoring levels. Linear `amp` still works, but
  dB is usually easier to reason about in mixes.
- Voices are LUFS-normalized by default (`normalize_lufs=-24.0`). This is the
  correct gain-staging approach for all tonal, melodic, and sustained voices — it
  ensures effects always see a consistent input level. Do not treat a synth engine's
  raw output level as the mix decision.
- **`mix_db` is for mix balance only, not gain staging.** Normalization handles gain
  staging; `mix_db` is the fader for balancing voices against each other. Similarly,
  `pre_fx_gain_db` is an intentional trim to drive effects harder or softer.
- **For percussive voices (kick, tom, noise hits), use `normalize_peak_db=-6.0`
  instead of `normalize_lufs`.** LUFS normalization is unreliable for transient
  content because the integrated LUFS varies with BPM and silence duration.
  `normalize_peak_db` normalizes to a target peak level, giving compressors and
  other effects a predictable input regardless of BPM or per-note `amp_db` values.
  The `kick_punch` and `kick_glue` compressor presets are calibrated for a
  `-6.0 dBFS` input peak.
- `normalize_lufs` and `normalize_peak_db` are mutually exclusive. Avoid
  `normalize_lufs=None` — prefer `normalize_peak_db` for any voice where LUFS
  normalization is inappropriate.
- `master_input_gain_db` trims the summed mix into `Score.master_effects`. Leave it
  at `0.0` by default; reach for it only when you intentionally want to change
  how hard the master bus glue/tone chain is being driven.
- `Score` now auto-stages the summed post-fader mix into the master bus by
  default, so voice `mix_db` is mainly for balance rather than manual premaster
  loudness management.
- Use note-level `velocity` for accents and phrasing. By default it affects loudness
  through `velocity_db_per_unit`, and it can also drive synth params through
  `VelocityParamMap`.
- `velocity_humanize` is voice-level and on by default when adding voices. Set it to
  `None` when you want a fully fixed/programmed result.
- `velocity_group` lets multiple voices share correlated velocity drift, which is the
  right tool for ensemble breathing rather than independent per-voice wobble.
- `envelope_humanize` is for subtle ADSR variation over score time. This is the
  current "env slop" surface.
- `timing_humanize` is score-level. Use it for ensemble looseness and shared drift,
  not for rewriting rhythmic structure.
- automation is the explicit parameter-motion surface. Use it for deliberate
  sweeps, bends, timbral motion, and wet/send/pan rides; use humanization for
  subtle living variation.
- When documenting or changing these APIs, keep `AGENTS.md` high-level and put the
  parameter-by-parameter details in `docs/score_api.md`,
  `docs/composition_api.md`, and
  `docs/synth_api.md`.

## Running Commands — IMPORTANT

**Never run bare `python` commands in this repo.**  The project uses `uv` for
environment management; a bare `python` or `pip` invocation will use the wrong
interpreter and will fail with import errors.

**Always use one of:**

```
make all                           # default quality gate: format-check, lint, compile, typecheck, full tests
make check                         # alias for make all
make list                          # list registered pieces
make render PIECE=harmonic_drift   # render a piece (adds --plot by default)
make render PIECE=harmonic_drift PLOT=0  # render without plot
make inspect PIECE=ji_chorale AT=2:10    # inspect score context around a timestamp
make snippet PIECE=ji_chorale AT=2:10 WINDOW=12   # render a centered snippet
make render-window PIECE=ji_chorale START=130 DUR=12  # render an exact snippet window
make render-all                    # render every piece
make midi PIECE=ji_chorale          # export a full MIDI bundle
make midi-snippet PIECE=ji_chorale AT=2:10 WINDOW=12  # export a centered MIDI snippet bundle
make midi-window PIECE=ji_chorale START=130 DUR=12    # export an exact MIDI snippet bundle
make test                          # run the full test suite
make test-selected TESTS=tests/test_score.py  # run a focused subset while iterating
make typecheck                     # basedpyright
make compile                       # syntax / bytecode compilation check
make lint                          # ruff check with bug-finding rules
make format-check                  # verify formatting without modifying files
make format                        # ruff format
```

If you need to run a one-off Python command, prefix it with `uv run` and set
`PYTHONPATH=.`:

```
PYTHONPATH=. uv run python main.py --list
PYTHONPATH=. uv run pytest tests/
```

The Makefile's `UV_RUN = PYTHONPATH=. uv run` variable handles this for all
standard targets.

`make all` is the default target and should be the normal final verification step
after meaningful code changes. It runs formatting verification, Ruff, Python
compilation, basedpyright, and the full test suite. Use narrower commands while
iterating if helpful, but do not stop there.
Make commands are preferred to `uv run` since they don't require human sign-off,
making for a better experience for both agent and human.

## Rendering Workflow

- `make list` — see all registered pieces.
- `make render PIECE=<name>` — render a named piece and save a piano-roll PNG.
  Default render path. Always capture the full output — do **not** pipe through
  `tail` or truncate in any way. The render emits loudness/peak stats, artifact-risk
  warnings (amplitude modulation, compression, brightness), and analysis manifest
  paths that are all important for diagnosing mix and DSP issues.
- `make render PIECE=<name> PLOT=0` — render without the plot.
- `make inspect PIECE=<name> AT=<timestamp>` — inspect a score-backed piece around
  a timestamp like `130` or `2:10`.
- `make snippet PIECE=<name> AT=<timestamp> WINDOW=<seconds>` — render a short,
  centered score-backed snippet for faster iteration.
- `make render-window PIECE=<name> START=<timestamp> DUR=<seconds>` — render an
  exact score-backed snippet window when you want repeatable boundaries.
- Pieces should expose either a `build_score()` function or a direct `render_audio()`
  function if they do not fit the score abstraction cleanly.
- **Rendering is slow** — a full piece takes multiple DSP passes and typically runs
  longer than real-time. When calling render commands via Bash, set a generous
  timeout (at minimum 3–5× the piece duration in wall-clock seconds; 180 s is a
  safe floor for most pieces). Prefer `make snippet` or `make render-window` when
  iterating on a local moment rather than re-rendering the whole piece.

## Musical Direction

The project is no longer just about proving that xenharmonic ideas work. Prefer music
that feels like music: shaped, intentional, and complete, even when it is strange.

Current aesthetic center:

- just intonation
- harmonic-series materials, including otonal and utonal writing
- a bias toward pleasant, listenable results without becoming timid or conventional
- willingness to move between gentle clarity and more chaotic or alien sections

Named inspirations already captured in repo notes:

- Bach
- Aphex Twin's simple and tender side
- Aphex Twin's chaotic and unstable side

If a task is about aesthetic direction, piece planning, or "what should this project
sound like?", also read `inspirations_and_ideas.md` before making recommendations.

When adding or revising pieces, it is usually better to ask "how do we make this more
musical?" than "how do we make this weirder?" Weirdness is easy; satisfying form is
harder and more valuable here.

## Future-Facing Ideas

See `FUTURE.md` for way more ideas.

## Implementation Notes

- Prefer phrase-first composition, but keep direct note insertion available.
- Treat velocity, timing humanization, and envelope humanization as part of the
  normal composition surface, not as obscure implementation details.
- Keep low-level synthesis simple unless a task explicitly calls for DSP changes.
- Treat effect chains declaratively with `EffectSpec` on voices or the master bus.
- The additive engine now supports explicit spectral partial sets plus optional
  onset-to-sustain spectral morphing; use that when tuning and timbre should be
  co-designed instead of assuming a plain harmonic ladder.
- The `polyblep` engine now supports an optional second oscillator via `osc2_*`
  parameters for detuned stacks and sub layers.
- The native effect chain now includes a minimum-phase multi-band `eq` effect with
  ordered highpass, lowpass, bell, and shelf bands for routine tone shaping.
- The native effect chain also includes a stereo-linked `compressor` effect with
  feedforward/feedback modes, detector-path EQ bands, and voice-to-voice
  sidechaining plus lookahead for ducking/glue workflows.
- The native `saturation` effect now defaults to a higher-fidelity two-stage
  analog-style path with optional clean low/high-band preservation; see
  `docs/synth_api.md` for the modern vs legacy behavior and parameter surface.
- There is now a dedicated `kick_tom` synth engine for 808/909-style kicks and
  toms; the intended happy path is pairing it with the native drum-oriented
  compressor/saturation presets documented in `docs/synth_api.md`.
- The `bricasti` convolution wrapper now supports basic wet-return tone shaping
  (`highpass_hz`, `lowpass_hz`, `tilt_db`) for cleaner, darker, or brighter tails.
- The local Linux environment now has a small plugin palette installed for
  experimentation: `LSP` utilities in `~/.vst` and `~/.lv2`, `Dragonfly Reverb`
  in `~/.vst3` and `~/.lv2`, `TAL-Chorus-LX` and `TAL-Reverb-2` in `~/.vst3`,
  and legacy Linux `Airwindows` VSTs in `~/.vst`.
- `Chow Tape Model`, `TAL-Chorus-LX`, `TAL-Reverb-2`, and Dragonfly VST3s have
  already been verified to load through `pedalboard`; prefer those before
  introducing more bridge-heavy or activation-heavy plugin paths.
- The macOS environment has a broader plugin palette: all ChowDSP plugins
  (CHOWTapeModel, BYOD, ChowMatrix, ChowCentaur, ChowKick, ChowMultiTool,
  ChowPhaser), TAL (Chorus-LX, Reverb-2, Reverb-3), Dragonfly (Plate,
  Room, Hall, Early Reflections), Airwindows Consolidated, and Surge XT.
  LSP Plugins are Linux-only and unavailable on macOS.
- Render analysis now also generates librosa-based visual analysis:
  mel spectrogram, tuning-aware chromagram (36 bins/octave for JI
  visibility), spectral contrast, onset envelope, and harmonic-percussive
  separation balance — all registered in the analysis manifest. These are
  the primary feedback channel for AI-assisted composition since LLMs
  cannot hear audio.
- There is now a `surge_xt` instrument engine that renders voices through
  Surge XT via pedalboard's VSTi hosting. It uses MPE-style per-note
  pitch bend (48-semitone range) for microtonal accuracy. Unlike the
  native per-note engines, it renders the whole voice at once.
  Use `engine="surge_xt"` in `synth_defaults`.
- Keep plugin notes here high-level. Detailed parameter semantics and any new
  `EffectSpec` integration still belong in `docs/synth_api.md`.
- If you change score/expression parameters or presets, update the docs in the same
  pass. `AGENTS.md` should mention the feature exists; the detailed semantics belong
  in the docs, especially `docs/score_api.md` for score-surface changes.
- When you add new functionality that future agents are likely to use, add a brief
  high-level callout to `AGENTS.md` in the same pass so the capability persists
  across sessions. Keep the mention short and intuitive here, and put the full API
  semantics in `docs/composition_api.md`, `docs/score_api.md`, or
  `docs/synth_api.md` as appropriate.
- Timestamp inspection is part of the normal workflow now. Prefer the timeline
  artifacts and `make inspect` when responding to comments like "2:10 in
  ji_chorale" instead of manually hunting through score code.
- Snippet rendering is part of the normal workflow for score-backed pieces now.
  Prefer `make snippet` or `make render-window` when iterating on a local moment
  instead of re-rendering the whole piece.
- MIDI export now mirrors that workflow for score-backed pieces via shared tuning files plus per-voice stems; prefer `make midi-snippet` or `make midi-window` when iterating on a local passage for DAW work.
- Score analysis now includes timing-drift diagnostics. Overall drift is fine, but
  inter-voice spread should stay musically plausible across the piece; keep the
  warning thresholds and artifact details documented under `docs/`.
- Render analysis now also records effect-chain diagnostics for compressors,
  saturation, and plugin stages so agents can see gain reduction, clipping-like
  density, and "mostly inactive" vs aggressive behavior in the analysis manifest.
- WAV export logging now reports peak, true-peak, and integrated LUFS at write
  time, and warns when an exported master lands suspiciously far below the
  expected ceiling.
- WAV export defaults to 24-bit PCM via `soundfile`. 16-bit (TPDF dithered)
  and 32-bit float are available via the `bit_depth` parameter on `write_wav()`.
- Render analysis now also emits artifact-risk warnings for suspicious
  brightness, modulation, compression/clipping, and risky filter-motion
  parameter combos; when touching those surfaces, update docs and tests in the
  same pass.
- When using or extending synth engines, read `docs/synth_api.md` first for the
  current engine names, presets, and parameter surface.
- Prefer absolute imports and typed, readable Python.
- Add tests for score timing, phrase transforms, tuning math, and piece integration
  when changing behavior.
- If a doc in this repo describes something as future work, verify it against the code
  before repeating it. Several foundational ideas are already implemented.
- Think carefully about interface design. Imagine you were interacting with devices with
  practically no context. What would be intuitive and easy to use?
  Prefer absolute units when possible (like msec for a compressor). For arbitrary knobs,
  use typical, usable ranges: 0-0.25 should be quite subtle, 0.25-0.33 should be clearly audible but
  still somewhat subtle, 0.33-0.66 should be moderate, and 0.66-0.8 should be strong. It's fine
  to have plausibly musical but not broken very strong effects from 0.8-1.0.
  For example, a saturation effect might offer gentle mix warmth at 0.2, harmony-compatible warmth at 0.33,
  musical saturation at 0.5, strong but still musical saturation at 0.66, and distortion at 0.8.
- Effects and voices should be designed considering musicality, not textbook designs.
  Don't cut corners to save time.
  For example, if a distortion plugin would benefit from antialiasing,
  don't just implement the most naive algo.

## Agent Delegation Policy

Core composition and piece writing should happen in the main agent context for
creative continuity and direct authorial control. Mechanical work (new engines,
features, infrastructure, repo improvements, portability fixes) follows the normal
delegation-to-subagents pattern.

## Test Philosophy

- Test early and often. Ideally write tests _first_ then code (TDD).  
- Focus on realistic, e2e tests (smoke, integration).  
- No need for trivial tests or testing each unexpected edge case. First and foremost, tests should validate that code runs properly, end to end, without major bugs; and they should prevent regressions.
- Backward compatibility is not always required or expected; this is a local, creative library not a business one.
- No fallbacks. Fail fast!

---

## Loose Inspirations & Ideas to Explore

- ji
- septimal harmony
- harmonic series, otonal and utonal
- aphex twin, both the simple and pleasant side (avril, aisatsana) and the chaotic side
- bach, the GOAT
- making xenharmonic ideas sound weird is pretty easy, making them sound pleasant is harder. let's bias in the pleasant direction, but not be constrained to narrow ideas of what pleasant is
- why are most xenharmonic pieces more like experimentations or sketches and not complete pieces? let's err on the side of making music that sounds musical. which shouldn't wall off experimentation and riffing, but can we make full compositions?
- commas and quirks of tuning systems (but musically; think Giant Steps)
- bittersweet, haunting. (think burial - untrue or MBV loveless)
- spacious, reverb, lofi (BoC, m83's dead cities)
- spicy but euphonic chords, creative voicing, voice leading, sevenths and septimal harmony
- it's fine to explore and fail. we can try a bunch of ideas, and none of them have to work
- think of music creation like a GAN: we come up with ideas, we see how we like them, and we iterate
(for more ideas, use `make inspire`)
