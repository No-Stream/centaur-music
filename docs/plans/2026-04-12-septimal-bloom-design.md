# Septimal Bloom — Design

## Concept

A ~2.5-minute slow bloom in 7-limit JI for two Surge XT voices. Starts
sparse, layers build, effects deepen, peaks at a utonal B section, then
dissolves. The harmonic seventh (7/4) is the emotional thread.

Aesthetic: elegiac, bittersweet, loveless-adjacent. Warbly detuned pads,
a melody that feels just out of reach.

## Key & Tuning

f0 = 110 Hz (A2). All partials relative to this. 7-limit JI — intervals
drawn from primes 2, 3, 5, 7.

## Structure

### A (~60s) — warm, otonal, bittersweet

Slow bloom from sparse to full. Chords enter gradually.

Progression (each chord ~7-8s, paced for breathing room):

| Chord | Partials | Character |
|-------|----------|-----------|
| I | 1, 5/2, 3 | Open JI major |
| iv7 | 4/3, 7/3, 8/3 | Septimal fourth — 7/4 above D |
| vi | 5/3, 2, 7/2 | Harmonic 7th on top |
| I7 | 1, 5/4, 3/2, 7/4 | Full septimal dominant — bittersweet |

Melody enters ~15s in. Sparse, descending, narrow range. Uses 8/7, 7/6,
9/8 steps. Gravitates toward 7/4 as a recurring passing tone.

Effects: reverb starts ~0.2 wet, delay subtle, saturation minimal.

### B (~50s) — elegiac, hollow, utonal + tonic drift

The I7 chord's 7/4 holds as a bridge into B.

| Chord | Partials | Character |
|-------|----------|-----------|
| I utonal | 1, 8/5, 4/3 | Hollow minor — undertonal triad |
| VII (tonic = 7/4) | 7/4, 35/16, 21/8 | Major chord on the harmonic 7th — the "reaching" moment |
| bVI | 8/5, 2, 12/5 | Gentle resting point |
| V7 | 3/2, 15/8, 9/4, 21/8 | Septimal dominant pulling home |

Melody more exposed, higher register, slower. Lingers on 7/4 and 7/6.
Fewer notes, longer durations. The "just out of reach" quality.

Effects at peak: reverb ~0.35 wet, delay feedback up, saturation warming.

### A' (~50s) — return, transformed

Same progression as A but voiced fuller. Melody restates A theme but
lands on 7/6 over I instead of resolving to 1 — home, but seen from
a different angle. Notes thin out gradually. Reverb tail is the last
thing you hear.

## Voices

### Pad (Surge XT)

The existing study patch — the "body" of the piece.

- Classic saw, 3-voice unison, ~12 cent detune (the warble)
- LP Vintage Ladder ~500 Hz, resonance ~12%
- Slow envelope: attack ~350ms, release ~700ms
- Handles all chords across all sections

### Melody (Surge XT)

A cleaner, more focused voice that sits above the pad.

- Sine or Modern oscillator — less harmonically dense
- 2-voice unison with ~4 cent detune (slight shimmer, not warble)
- LP Vintage Ladder brighter (~1 kHz), low resonance
- Faster attack (~100ms) so notes speak more clearly
- Handles the melody line only

## Effects

All native (project effect chain), not Surge XT internal FX:

- **Reverb**: Bricasti "Large & Dark" on master bus. Automate wet from
  ~0.2 (A) to ~0.35 (B peak) back to ~0.25 (A').
- **Delay**: ~400ms, feedback ~0.15, on melody voice or shared send.
  Creates rhythmic echo without clutter.
- **Saturation**: tube_warm, subtle (mix ~0.15). Glue for the dense
  B section.

## Melody Design

- Narrow range: roughly a fifth (3/2 span), mostly in the partial 5/2 to
  7/2 region (~275-385 Hz)
- Small intervals: 8/7 (septimal whole tone, ~231 cents), 7/6 (septimal
  minor third, ~267 cents), 9/8 (whole tone, ~204 cents)
- Mostly descending contour with occasional upward reaches
- 3-5 notes per chord change, long durations
- 7/4 as gravitational center: keeps passing through it, lingers more
  in B, ends on it or 7/6 in A'
- Phrasing: slow, breathy, like someone singing with eyes closed

## Duration Budget

| Section | Duration | Cumulative |
|---------|----------|------------|
| A | ~60s | 0:00-1:00 |
| B | ~50s | 1:00-1:50 |
| A' | ~50s | 1:50-2:40 |
| Tail | ~10s | 2:40-2:50 |
| **Total** | **~170s** | **~2:50** |

## Implementation Notes

- Single file: `code_musics/pieces/septimal_bloom.py`
- Registered as a full piece (not a study) in `pieces/__init__.py`
- Two Surge XT render passes (one per voice) — since the plugin is
  cached and shared, the melody voice params will overwrite the pad
  params. This is fine because voices render sequentially.
- Use `score.add_note()` directly for hand-composed chords and melody
  (same approach as the study, scaled up)
- Score-time automation on master reverb wet for the bloom arc
- HarmonicContext + drifted() for the B section tonic shift to 7/4
