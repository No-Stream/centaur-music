# code-musics

Exploring musical collaboration with generative AI.

This project focuses on alternative tuning systems, especially just intonation and
other xenharmonic ideas that are awkward to explore in traditional tools. The goal
is not just to make sketches or tuning demos, but to make complete pieces that feel
musical.

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

Some touchstones for the aesthetic are Bach, plus both the tender and the unruly
sides of Aphex Twin.

## Usage

List available pieces:

```bash
python main.py --list
```

Render a piece:

```bash
python main.py harmonic_drift
```

Render a piece and save a piano-roll plot when available:

```bash
python main.py harmonic_drift --plot
```
