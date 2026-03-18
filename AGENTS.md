# Repository Guide

## Overview

- `code_musics/` is the main package.
- `docs/synth_api.md` documents synth engines, presets, and engine-specific params.
- `code_musics/synth.py` contains low-level DSP, rendering, and effect helpers.
- `code_musics/score.py` contains the composition abstractions: `NoteEvent`,
  `Phrase`, `Voice`, `Score`, and `EffectSpec`.
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
- `Phrase` is a reusable collection of relative-time `NoteEvent`s. Use it for motifs,
  sequences, and transformed restatement.
- `Score.add_note(...)` is the escape hatch for one-off pedals, accents, blooms, and
  transitions.
- `Score.add_phrase(...)` is the main composing API. It supports placement transforms
  like `time_scale`, `partial_shift`, `amp_scale`, and `reverse`.
- `Voice` stores note events plus synth defaults and voice-level effects.
- `Score` owns the timeline, derives `total_dur`, renders audio, and can save a
  piano-roll plot.
- `Voice.synth_defaults` and note-level `synth={...}` overrides accept an `engine`
  name, optional `preset`, and engine-specific params documented in `docs/synth_api.md`.

## Running Commands — IMPORTANT

**Never run bare `python` commands in this repo.**  The project uses `uv` for
environment management; a bare `python` or `pip` invocation will use the wrong
interpreter and will fail with import errors.

**Always use one of:**

```
make list                          # list registered pieces
make render PIECE=harmonic_drift   # render a piece (adds --plot by default)
make render PIECE=harmonic_drift PLOT=0  # render without plot
make render-all                    # render every piece
make test                          # run the test suite
make lint                          # ruff check
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

## Implementation Notes

- Prefer phrase-first composition, but keep direct note insertion available.
- Keep low-level synthesis simple unless a task explicitly calls for DSP changes.
- Treat effect chains declaratively with `EffectSpec` on voices or the master bus.
- When using or extending synth engines, read `docs/synth_api.md` first for the
  current engine names, presets, and parameter surface.
- Prefer absolute imports and typed, readable Python.
- Add tests for score timing, phrase transforms, tuning math, and piece integration
  when changing behavior.
- If a doc in this repo describes something as future work, verify it against the code
  before repeating it. Several foundational ideas are already implemented.
