# sodium_hymn — eikosany trilogy-closer (design spec)

**Date:** 2026-07-04
**Piece:** `sodium_hymn` — third and final panel of the CPS trilogy
(`hexany_garden` → `ninth_wave` → `sodium_hymn`).

## Identity

Burial-inflected swung 2-step at **~132 BPM** over the **1-3-5-7-9-11 eikosany**
(3-of-6 CPS, 20 notes). Where the earlier panels default to otonal light, this
piece lives in **utonal shadow** and earns its otonal moments as shafts of
light. Bittersweet, haunted, weathered: rain on a night-bus window. Bells
remain the trilogy's connective tissue, but distant and reverb-drowned.
Centerpiece is a **wordless ghost vocal** — formant-morphing additive voice,
pitch-gliding between eikosany tones, smeared and drowned.

Distinctness contract vs. siblings:

- **No four-on-floor.** Shuffled 2-step: kick on the swing grid, snare/clap on
  2 and 4, ghost-velocity hats, heavy dropout bars, phrase-head breaths.
- **Lofi/weathered** where they are clean: vinyl crackle + rain noise bed,
  tape on the bus, gentle high roll-off, long dark shared reverb,
  reverse-reverb swells into section turns.
- **Mid-register warmth**, not sub pressure: bass audible via mid harmonics
  (brightness/dirt), F1 touches only at structural drops.

## Tuning

- Scale: eikosany over factors `(1, 3, 5, 7, 9, 11)`, normalized by
  `1*3*5 = 15` so the note {1,3,5} is 1/1.
- **f0 = F2 ≈ 87.31 Hz** (distinct from ninth_wave's G, inside the F–G#
  impact zone for the kick).
- New helpers in `code_musics/tuning.py` (mirroring `dekany`/`dekany_chords`),
  with tests:
  - `eikosany(factors=(1,3,5,7,9,11), *, normalize=None)` → 20 sorted ratios;
    default normalize `factors[0]*factors[1]*factors[2]`.
  - `eikosany_tetrads(...)` → `(otonal_tetrads, utonal_tetrads)`: 15 each.
    Otonal tetrad per factor pair {x,y} (notes xy·z, sounding as the chord of
    the complementary four factors); utonal tetrad per 4-subset S (notes
    prod(S)/a, sounding 1/a:1/b:1/c:1/d). Sorted ascending, octave-reduced,
    same normalization as `eikosany(...)`.

### Structural facts the piece leans on

- The 3-of-6 CPS is the unique **self-dual** CPS: 15 full otonal AND 15 full
  utonal tetrads. An otonal tetrad O{x,y} and utonal tetrad U{S} share **two
  common tones** iff {x,y} ⊂ S — the piece's pivot mechanism.
- **O{9,11} sounds as 1:3:5:7** — the hexany_garden chord — but its notes
  (33/32, 99/80, 231/160, 33/20) sit a 33/32 comma (~53¢) off the tonic
  region: the light is a detuned memory.
- Notes containing factor 11 form the 1-3-5-7-9 dekany (ninth_wave's scale)
  transposed by 11; otonal chords O{x,11} sound as pure 1-3-5-7-9 harmony
  (the dekany quote region).
- Trilogy ending series: hexany_garden hung on 4:7 → ninth_wave resolved
  4:5:6:7 → sodium_hymn ends on **5:7:9:11** (= O{1,3}, whose notes 1/1,
  11/10, 7/5, 9/5 contain the tonic) — the next odd harmonics, resolution
  and openness at once.

## Harmonic spine

Home is deliberately **non-undecimal shadow**; the 11 flavor deepens
intentionally across the piece, one pivot at a time.

| Station | Chord | Sounds as | Notes | Role |
|---|---|---|---|---|
| Home | U{1,3,5,9} | 1/1:1/3:1/5:1/9 | 1, 9/8, 3/2, 9/5 | graspable minor-7 shadow, no 11 |
| First tint | U{1,3,9,11} | 1/1:1/3:1/9:1/11 | 11/10, 99/80, 33/20, 9/5 | 11 creeps in; pivot 9/5 with home |
| Hexany light | O{9,11} | 1:3:5:7 | 33/32, 99/80, 231/160, 33/20 | hexany_garden quote, comma-shifted; 2 pivots (99/80, 33/20) with U{1,3,9,11} |
| Dekany walk | O{5,11}, O{1,11}, … | 1-3-5-7-9 subsets | (11-bearing notes) | ninth_wave world quoted; chained by single common tones |
| Deep shadow | U{5,7,9,11} | 1/5:1/7:1/9:1/11 | 33/32, 21/16, 231/160, 77/48 | fully undecimal utonal — cathedral core; 2 pivots (33/32, 231/160) with O{9,11} |
| Darkening home | U{1,3,5,11} | 1/1:1/3:1/5:1/11 | 1, 11/10, 11/8, 11/6 | home shape with 9→11 swap; pivots on held 1/1 |
| Final light | O{1,3} | 5:7:9:11 | 1, 11/10, 7/5, 9/5 | ending sonority over the tonic |

Finale alternates U{1,3,5,9} ↔ O{1,3} around the fixed pillars 1/1 and 9/5
(9/8, 3/2 ↔ 11/10, 7/5), and the piece hangs on O{1,3}.

Exact bar-level progressions are a composition-time decision; the spine above
is the contract. Melody must lead every section turn (bridge descents), per
the ninth_wave feedback.

## Form (~6.5 min at 132 BPM, 4/4 swung)

1. **Rain intro** (beatless): vinyl/rain bed, distant bells outlining home
   U{1,3,5,9} tones.
2. **Ghost vocal enters** (beatless): siren lines over smeared pad; home →
   first tint.
3. **First 2-step section**: beat materializes; utonal walk; bass enters.
4. **Otonal light passage**: pivot into O{9,11} hexany quote; brightest bells.
5. **Cathedral** (beat dissolves): deep shadow U{5,7,9,11}; vocal + bell duet;
   longest reverbs.
6. **Blackness**: ~2 bars near-silence.
7. **Second drop** (fullest): 2-step returns evolved; dekany-walk otonal
   chords against utonal answers; melody-led turns; F1 bass touches.
8. **Dissolution outro**: beat decays to dropout ghosts; U{1,3,5,9} ↔ O{1,3}
   alternation; final hanging 5:7:9:11 into rain.

## Voices

| Voice | Engine/approach | Notes |
|---|---|---|
| Ghost vocal | additive w/ formant morphing (`vowel_cathedral` machinery), vowel slides mid-phrase, `PitchMotionSpec` glides, `smear.py` treatment, dark reverb send | centerpiece; iterate until it convinces |
| Bells | FM (`synth_voice` fm slot) and/or modal, heavy reverb send | Perälä-through-a-wall; thread whole piece |
| Smeared pad | `SmearVoice` loveless treatment | holds the tetrads |
| Bass | `synth_voice`, mid-harmonic-forward (dirt/brightness) | swung-8th gaps; F1 only at drops |
| 2-step kit | `drum_voice` via `add_drum_voice`; soft round kick, snare/clap 2+4, shuffled ghost hats, wood/rim accents | Objekt-style pattern evolution; dropout bars |
| Weather bed | noise/flow exciter + vinyl crackle | constant, automated level |

Mix: LUFS-normalized tonal voices, `normalize_peak_db=-6` percussion,
`DEFAULT_MASTER_EFFECTS`-style chain plus tape stage and lofi high-shelf
tilt; shared dark reverb send bus (`bricasti_or_reverb` or Dragonfly hall).

## Process

1. **Mechanical prelude (Sonnet subagent):** `eikosany` + `eikosany_tetrads`
   in `tuning.py`, tests in `tests/test_tuning.py` (TDD), docs touch-up.
2. **Composition (main context):** build `code_musics/pieces/sodium_hymn.py`
   incrementally with `make snippet` / `make render-window`; full renders +
   analysis plots as the feedback channel.
3. **Ghost-vocal iteration loop:** isolated snippet renders of the vocal
   patch until the formant motion reads as voice, before full arrangement.
4. **Finish:** full render, artifact-warning triage, full eval panel
   (read spread, not single-judge deltas).

The piece file gets a signed header comment (composer's signature welcome).

## Success criteria

- Renders clean via `make render PIECE=sodium_hymn` with no severe artifact
  warnings that aren't understood and justified.
- Recognizably distinct from both siblings in groove, texture, and mood.
- Undecimal color audibly deepens over the arc; ending lands on 5:7:9:11.
- Ghost vocal reads as a voice, not a vowel-ish pad.
- `make all` green; eikosany helpers tested.
