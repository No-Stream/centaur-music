# centaur musical workstation

Exploring musical collaboration with agentic AI.

This project focuses on alternative tuning systems, especially just intonation and
other xenharmonic ideas that are awkward to explore in traditional tools. What can we make together? Can agentic collaboration create new, weird, unique, fresh ideas and let us work in alternate tunings more easily?

Naturally, the synths are custom, to support alternate tunings. The effects are a mix of internal and vst-based (supports Linux VST3s).

This repo contains both the tooling and the pieces; not pruned at all, so most of them will sound aggressively unpleasant and unfinished!

## Current shape

- `code_musics/score.py` provides the composition model: `Score`, `Voice`,
  `Phrase`, `NoteEvent`, and `EffectSpec`.
- `code_musics/tuning.py` provides small helpers for harmonic-series, JI, utonal,
  and EDO work.
- `code_musics/pieces/` contains registered named pieces.
- `code_musics/render.py` and `main.py` handle listing and rendering pieces.
- `code_musics/synth.py` contains the lower-level synthesis and effect utilities.

## Project direction

The current musical center of gravity is:

- just intonation
- harmonic-series writing, including otonal and utonal materials
- xenharmonic music that can still sound pleasant, lyrical, and structurally like
  music rather than only like an experiment
- a wide expressive range, from simple and gentle to strange and chaotic


## Usage

List available pieces:

```bash
make list
```

Render a piece:

```bash
make render PIECE=harmonic_drift
```

Render a piece and save a piano-roll plot when available:

```bash
make render PIECE=harmonic_drift
```

Render a piece without the piano roll:

```bash
make render PIECE=harmonic_drift PLOT=0
```

By default, renders now also emit analysis artifacts and a JSON manifest into
`output/`, alongside the WAV file. Use the CLI directly if you want to disable
analysis for a render:

```bash
PYTHONPATH=. uv run python main.py harmonic_drift --no-analysis
```
