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
  registry.
- `code_musics/render.py` is the named-piece orchestration layer.
- `main.py` is the main entrypoint for listing and rendering pieces.
- `sketches/01_septimal.py` is a compatibility wrapper that renders the septimal
  reference pieces through the registry.

## Composition Model

- `NoteEvent` is the atomic musical event. It stores timing, amplitude, and either a
  `partial` relative to `Score.f0` or an absolute `freq`.
- `NoteEvent` also supports per-note `velocity`, `amp_db`, and optional
  `pitch_motion`. Velocity is a first-class expressive control now, not just an
  engine detail.
- `Phrase` is a reusable collection of relative-time `NoteEvent`s. Use it for motifs,
  sequences, and transformed restatement.
- `Score.add_note(...)` is the escape hatch for one-off pedals, accents, blooms, and
  transitions.
- `Score.add_phrase(...)` is the main composing API. It supports placement transforms
  like `time_scale`, `partial_shift`, `amp_scale`, and `reverse`.
- `Voice` stores note events plus synth defaults, voice-level effects, pan, envelope
  humanization, velocity humanization, and optional velocity-to-parameter mappings.
- `Score` owns the timeline, derives `total_dur`, renders audio, and can save a
  piano-roll plot.
- `Score.timing_humanize` applies render-time ensemble timing drift across the whole
  score, while voice-level humanizers shape envelope and velocity variation.
- `Voice.synth_defaults` and note-level `synth={...}` overrides accept an `engine`
  name, optional `preset`, and engine-specific params documented in `docs/synth_api.md`.
- For detailed score-surface semantics, parameter meanings, and render-order
  behavior, read `docs/score_api.md`.

## Expression Model

- Prefer `amp_db` over raw `amp` for authoring levels. Linear `amp` still works, but
  dB is usually easier to reason about in mixes.
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
make render-all                    # render every piece
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

## Rendering Workflow

- `make list` — see all registered pieces.
- `make render PIECE=<name>` — render a named piece and save a piano-roll PNG.
- `make render PIECE=<name> PLOT=0` — render without the plot.
- Pieces should expose either a `build_score()` function or a direct `render_audio()`
  function if they do not fit the score abstraction cleanly.

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

These are good directions to keep in mind while working, even if they are not all
implemented yet:

- expand the library of phrase-level compositional helpers
- explore richer visual analysis beyond the current piano roll
- push further into utonal and subharmonic writing as a structural device
- explore FM and other timbral approaches that interact well with JI ratios
- support fuller composition, not just studies, demos, or sketches
See `FUTURE.md` for way more ideas.

## Implementation Notes

- Prefer phrase-first composition, but keep direct note insertion available.
- Treat velocity, timing humanization, and envelope humanization as part of the
  normal composition surface, not as obscure implementation details.
- Keep low-level synthesis simple unless a task explicitly calls for DSP changes.
- Treat effect chains declaratively with `EffectSpec` on voices or the master bus.
- If you change score/expression parameters or presets, update the docs in the same
  pass. `AGENTS.md` should mention the feature exists; the detailed semantics belong
  in the docs, especially `docs/score_api.md` for score-surface changes.
- When using or extending synth engines, read `docs/synth_api.md` first for the
  current engine names, presets, and parameter surface.
- Prefer absolute imports and typed, readable Python.
- Add tests for score timing, phrase transforms, tuning math, and piece integration
  when changing behavior.
- If a doc in this repo describes something as future work, verify it against the code
  before repeating it. Several foundational ideas are already implemented.

## Test Philosophy
- Test early and often. Ideally write tests _first_ then code (TDD).  
- Focus on realistic, e2e tests (smoke, integration).  
- No need for trivial tests or testing each unexpected edge case. First and foremost, tests should validate that code runs properly, end to end, without major bugs; and they should prevent regressions.
- Backward compatibility is not always required or expected; this is a local, creative library not a business one.

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

