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
- **Fused spectra:** partial sets built from scale degrees across octaves,
  dominated by the 3/7-limit skeleton (1, 3/2, 7/4 and their compounds).
  11-flavored partials appear only in the designated spice voice/moments.
- Verify during implementation that `inharmonic_scale` semantics on the
  chosen engines match `n ** log2(P)` per partial; if any engine's stretch
  law differs, use explicit partial ratio lists instead. Engine
  modifications/extensions are in scope where the piece needs them
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
| Bass | `synth_voice`, fused scale-partials, follows the root motion |
| Lead arp | additive / FM bell with fused spectrum, groove template, velocity phrasing, vibrato |
| Pad | fused additive chords, `ratio_glide` between changes, spectral gravity |
| Act II counter-voice | grain or FM bell carrying the 11-limit color |
| Drums | 49 Hz kick recipe + `drum_voice` modal hats/toms with mode tables tuned to scale degrees, electronic drum bus |
| Atmosphere | `found_sound` toolkit presets, unstretched, low in the mix |

Standard finish: `DEFAULT_MASTER_EFFECTS`, shared reverb send bus
(`bricasti_or_reverb`), full automation passes (filter rides, send rides,
dropout bars, transition swells), humanization + drift bus on, pitch motion on
all melodic/sustained voices.

## Implementation shape

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
