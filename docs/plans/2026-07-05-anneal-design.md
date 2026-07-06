# anneal — Colundi stretch-drift (design spec)

**Date:** 2026-07-05
**Piece:** `anneal` — co-designed scale + spectra study grown into a full piece.
The first repo piece where tuning and timbre are one system *and where that
system itself is the dramatic arc*: the whole world (scale and partials
together) is heated until it stretches, held, then slow-cooled back home.
Annealing: heat, hold, slow cool.

## Identity

Four Tet-organic broken beat at **~110 BPM** in a warm Colundi-flavored
11-limit JI world on **G**. Wandering arps, glowing fused pads, tuned modal
percussion, and a synthetic found-sound "room" underneath. Act II slowly
stretches the pseudo-octave to ~2.07 — spectrum and tuning warp *together*, so
the music stays self-consonant while becoming unmistakably alien (gamelan/bell
physics) — then Act III relaxes home slower than it rose. Euphonic first,
strange by construction rather than by dissonance.

Core principle (Sethares): consonance is a property of the scale × spectrum
pair. Every tonal voice's partials are drawn from the scale's own degrees, so
the scale's intervals are maximally smooth by construction and chords fuse
into single composite timbres.

## Tuning and spectrum system

- **Scale:** Colundi core set — `1/1, 11/10, 19/16, 4/3, 3/2, 49/30, 7/4, 2/1`
  (0, 165, 297.5, 498, 702, 849, 969, 1200 cents).
- **f0 = G2 = 98.0 Hz**; kick fundamental at G1 = 49 Hz, reusing the proven
  `ninth_wave` 49 Hz kick+bass mix recipe. Distinct root from sodium_hymn (F)
  and inside the F–G# kick-impact zone.
- **Stretch machinery:** one master curve **S(t)** — pseudo-octave over score
  time, 2.000 → ~2.07 → 2.000. A ratio r maps through
  `stretch(r, P) = r ** log2(P)` (pure exponent scaling in log-pitch space:
  step geometry preserved, everything widens; at P=2.07 the octave is +60 c,
  the fifth ~737 c). Helper `stretch_ratio(r, pseudo_octave)` lands in
  `code_musics/tuning.py` with unit tests.
- **Single source of truth:** the same S(t) curve feeds
  1. note frequencies — sampled at each note's onset when phrases are built;
  2. per-note spectral stretch — each tonal voice's partial set is stretched
     by the matching value (via `inharmonic_scale` where the engine supports
     it, or by computing explicit stretched partial ratios per note on the
     additive path).
  Stretch moves over minutes, so per-note-onset sampling is seamless; scale
  and spectrum cannot drift apart because both derive from one function.
- **Fused spectra (all explicit partial lists):** scale degrees transposed by
  octaves. The 3/7-limit skeleton yields the integer set
  **{1, 2, 3, 4, 6, 7, 8, 12, 14, 16}** with rolloff — critically **no 5th
  harmonic** (or 10/15/20): the scale has no 5/4 (19/16 is 89 c away), so a
  5th partial would beat against every third in the piece. The color degrees
  yield inharmonic partials that carry their harmony timbrally:
  19/16 → {2.375, 4.75}; 49/30 → {3.27, 6.53}; 11/10 → {2.2, 4.4}.
- Because every tonal voice uses explicit partial lists, the stretch is
  applied by mapping each list through `stretch_ratio(p, S(t))` per note —
  exact and engine-agnostic; `inharmonic_scale` is not needed.
- **Distortion restraint:** saturation regenerates all integer harmonics
  including the forbidden 5th. Tonal voices get warmth from multiband preamp
  (lows/highs bypass) or very subtle drive only; no `voice_dist` on fused
  voices. Filters are safe (subtractive only).
- Engine modifications/extensions are in scope where the piece needs them
  (delegate granular engine work to subagents; composition stays in the
  main context).

## Harmony (intentional, restrained)

- **Skeleton:** real root motion `1/1 → 4/3 → 3/2 → 1/1` with a `7/4`
  turnaround. Tonic sonority is the glowing **4:6:7**.
- **Colors, each with one job:**
  - `19/16` — the dark minor third (16:19:24 triads) for Act II's turn;
  - `49/40` — the neutral third that lives only on the subdominant
    (`4/3–49/30–2/1`);
  - `11/10` — **reserved**: melodic appoggiaturas plus exactly one structural
    spice chord at the Act II climax. No 11s in the pads.
- Voice-led progressions, not drones; suspensions and stepwise inner motion
  using the scale's small steps (`15/14`-sized 7/4→49/30 descent, etc.).

## Form (~6.5 min, ~110 BPM)

- **Act I — cool (0:00–2:00), S = 2.000:** found-sound room fades in; fused
  4:6:7 pad breathes; main wandering arp motif enters; beat assembles piece by
  piece (hats → kick → full kit). Pure JI.
- **Act II — heat (2:00–4:30), S ramps to ~2.07:** harmony darkens through
  19/16 territory and the neutral subdominant; full kit, energy peak; the
  11/10 spice chord lands at the climax at maximum stretch. Modal percussion
  mode tables warp with the tuning.
- **Act III — anneal (4:30–6:30), S relaxes home slower than it rose:** beat
  thins; Act I motif returns in home tuning; final arrival on
  maximum-fusion 4:6:7; found-sound outro.
- **Deliberate contrast:** the found-sound bed does **not** stretch — the room
  stays real while the music inside it melts. Grounds Act II and makes the
  warp legible.

## Orchestration

| Role | Voice |
|---|---|
| Bass | `synth_voice`: sine/sub osc (49 Hz foundation) + partials layer {1, 2, 3, whisper of 7} through a ladder filter |
| Lead arp | **additive bell, not FM** (FM sidebands land off-scale): skeleton spectrum with exponential per-partial decays (faster high-partial decay = mallet physics), touch of `phase_disperse`, velocity → brightness, groove template, vibrato |
| Pad | `additive`, skeleton spectrum + section-dependent color partials (+19-family in Act II dark passages, +49-family on the neutral subdominant — timbre foreshadows harmony), `phase_disperse`, per-partial slow envelopes, mild spectral gravity, `ratio_glide` |
| Act II counter-voice | the **11-carrier**: additive spectrum {1, 2.2, 4.4, 8.8} over a light skeleton, optionally through the `grain` slot (shimmer halo). The only voice possessing 11-family partials — the restraint rule enforced timbrally |
| Drums | 49 Hz kick recipe + `drum_voice` modal hats/toms with mode tables from scale-degree ratios; the mode tables receive S(t) too, and drums stretch slightly *ahead* of tonal voices in Act II to introduce the warp timbrally |
| Atmosphere | `found_empty_room` + `found_city_at_night` presets, unstretched, low in the mix |

Standard finish: `DEFAULT_MASTER_EFFECTS`, shared reverb send bus
(`bricasti_or_reverb`), full automation passes (filter rides, send rides,
dropout bars, transition swells), humanization + drift bus on, pitch motion on
all melodic/sustained voices.

## Implementation shape

- **Audition-first milestones.** Before composing the full piece, produce
  short sketch renders for user sign-off:
  1. *Timbre/fusion sketch:* skeleton pad playing 4:6:7 and 16:19:24 with and
     without color partials, at S=2.000 and S=2.07 — confirm fusion is
     audible/visible in the chromagram and that explicit non-integer partials
     render cleanly.
  2. *Palette sketch:* ~30–60 s with bass, bell arp, pad, drums, and the
     found-sound bed at home tuning — sanity-check the sound world and mix
     before any form-building.
  User auditions each before the piece proceeds.
- `code_musics/pieces/anneal.py` (composed in main context).
- `stretch_ratio(...)` helper in `code_musics/tuning.py` + unit tests.
- Piece integration/render smoke test following existing piece-test
  conventions; snippet renders (`make render-window`) for iteration.
- The fake found-sound toolkit (`code_musics/found_sound.py`, presets, tests,
  docs) is being built by a delegated subagent as independent infrastructure.
- Docs: this spec; any engine surface changes documented in
  `docs/synth_api.md` in the same pass per repo policy.

## Risks

- **Stretch legibility:** if Act II reads as "out of tune" rather than "the
  world warped," lean on the fused spectra (they carry the consonance) and on
  the unstretched found-sound anchor; consider stretching the drums' modal
  banks earlier than the tonal voices so the warp is introduced timbrally.
- **11-limit creep:** the restraint rule is structural — audit the final score
  for 11/10 and 49/30 usage against their designated jobs.
- **Per-note stretch cost:** per-note partial sets defeat some engine caching;
  quantize S(t) to per-bar values if render time balloons.
