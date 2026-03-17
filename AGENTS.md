# Repository Guide

## Overview
- `code_musics/` is the main package.
- `code_musics/synth.py` contains low-level DSP and effect helpers.
- `code_musics/score.py` contains the composition abstractions: `NoteEvent`, `Phrase`, `Voice`, `Score`, and `EffectSpec`.
- `code_musics/tuning.py` contains small just-intonation and EDO helper functions.
- `code_musics/pieces/` contains named musical works that can be rendered by the registry.
- `main.py` is the main entrypoint for listing and rendering named pieces.
- `sketches/01_septimal.py` is now a compatibility wrapper that renders the septimal reference pieces through the registry.

## Composition Model
- `NoteEvent` is the atomic musical event. It stores timing, amplitude, and either a `partial` relative to `Score.f0` or an absolute `freq`.
- `Phrase` is a reusable collection of relative-time `NoteEvent`s. Use it for motifs and transformations.
- `Score.add_note(...)` is the escape hatch for one-off pedals, accents, and transitions.
- `Score.add_phrase(...)` is the main composing API. It supports placement transforms like `time_scale`, `partial_shift`, `amp_scale`, and `reverse`.
- `Voice` stores note events plus synth defaults and voice-level effects.
- `Score` owns the timeline, derives `total_dur`, renders audio, and can save a piano-roll plot.

## Rendering Workflow
- Use `python main.py --list` to see the registered pieces.
- Use `python main.py harmonic_drift --plot` to render a piece and save a piano-roll image when available.
- `code_musics/render.py` is the main orchestration layer for named-piece rendering.
- Pieces should either expose a `build_score()` function or a direct `render_audio()` function if they do not fit the score abstraction cleanly.

## Implementation Notes
- Prefer phrase-first composition, but keep direct note insertion available.
- Keep low-level synthesis simple unless a task explicitly calls for DSP changes.
- Treat effect chains declaratively with `EffectSpec` on voices or the master bus.
- Prefer absolute imports and typed, readable Python.
- Add tests for score timing, phrase transforms, tuning math, and piece integration when changing behavior.
