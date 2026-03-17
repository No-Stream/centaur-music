# Future Work

## High priority

### Score abstraction
The current approach tangles "what notes to play" with "how to render them" — cursor arithmetic is error-prone and hard to edit after the fact. A lightweight `Score` / `Voice` / `Phrase` model would fix this:

```python
score = Score(f0=55.0)
score.add_phrase("canon_a", partials=[4,5,6,7,6,5,4], start=46.0, note_dur=1.6, step=0.78)
score.add("pedal", start=54.0, partial=4, dur=18.0)
audio = score.render(effects=[...])
score.plot()  # piano roll — see the whole piece
```

Key properties wanted:
- Notes as data (`start`, `partial`, `duration`, `amp`, optionally timbre params)
- `total_dur` derived automatically from note endpoints (not manually set)
- Piano roll visualization via matplotlib so both human and Claude can inspect structure
- Effect chains applied cleanly post-render per-voice or on the full mix
- Motif / phrase reuse: define once, place multiple times with transformations (transpose, invert, retrograde, speed up)

### tuning.py
Thin wrapper around our idioms for JI/EDO work:

```python
harmonic_series(f0, n_partials)     # [f0*1, f0*2, ..., f0*n]
ji_chord(f0, ratios)                # e.g. ji_chord(110, [1, 5/4, 3/2, 7/4])
edo_scale(f0, divisions, octaves)   # 12-EDO, 31-EDO, 53-EDO etc.
otonal(f0, partials)                # select partials from harmonic series
utonal(f0, partials)                # subharmonic series (1/n)
cents_to_ratio(cents)
ratio_to_cents(ratio)
```

## Medium priority

### FM synthesis
Frequency modulation synthesis — operator ratios interact with JI in interesting ways.
Rational FM ratios (e.g. modulator = 7/4 × carrier) stay in-harmonic and reinforce the
septimal character. Irrational ratios produce metallic inharmonic timbres good for the
alien sections. All implementable with numpy, no new deps needed.

### Spectral / piano roll visualization
- FFT plot of a rendered segment: see harmonic relationships, verify JI tuning visually
- Spectrogram over time: watch how the texture evolves
- Piano roll of a Score: the canonical "what does this piece look like" view
- Useful for both the composer and for Claude to reason about the music without needing
  to hear it (Claude cannot currently receive audio)

### Utonal / subharmonic exploration
Subharmonic series (1/n) has a darker, more minor quality than the overtone series.
Otonal for ascending/light passages, utonal for shadow/descent. Could be powerful in
combination — the alien section in 03 is a natural candidate.

### Valhalla Supermassive (VST3, free)
Free Linux VST3 build available. Shimmer/lush long-tail modes would suit this music well.
Load via pedalboard once installed.

## Lower priority

### Parametric piece generation
Functions that take `(f0, limit_prime, scale_partials)` and generate a full piece —
good for exploring "what does this sound like in 11-limit vs 13-limit?"

### Microtonal MIDI export
pretty_midi with pitch bend to export pieces in DAW-compatible format.
14-bit pitch bend gives ~1¢ resolution, enough for JI.

### Granular synthesis
numpy-based granular engine — scatter short grains from the harmonic series with
randomised position, pitch, and envelope. Interesting textural middle ground between
synthesis and sampling.
