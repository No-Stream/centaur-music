# aphotic — design spec

*2026-07-07. Status: approved design, pre-implementation.*

A piece inspired by a meditative image: a vast dark underground cave,
crystalline stalactites and stalagmites, dripping water, serene, with a sacred
"Awareness" quality — private, secluded, unreachable; the crisp air of a dark
camping night; austere and beautiful. The name refers to the aphotic zone: the
depth at which light no longer reaches.

## Aesthetic contract

- **Crisp, clean, airy, hifi.** No lo-fi dust, no tape wobble aesthetics. High
  dynamic range; silence has a floor but no grit.
- **Austere.** ~6 voices. Space between events is part of the material.
- **Alive, not mechanical.** Varied tempo, varied velocity, automation arcs,
  humanization, and pitch motion are designed in from the start, not deferred
  polish (see Aliveness section).
- **Euphony first, spice intentional.** Dissonant/spicy intervals (bare 11/8,
  13-limit clusters) are placed deliberately — at transitions and tension
  points, resolving into otonal locks — never scattered as default color.
  Anything that auditions as "wrong note" rather than "alien light" gets cut.

## Tuning world: no-threes JI

Subgroup **2.7.11.13**, prime 5 rationed (see the illumination conceit). Prime
3 is absent entirely: no 3/2, no 4/3, no 9/8 — the strongest directional
forces in Western harmony do not exist in this piece. Octaves are kept.

Core sonority: **4:7:11**, extended to **4:7:11:13** for shimmer. 13 is on
probation — auditioned early; dropped from the working set if it reads as
sour rather than dusky.

Working scale (all 3-free; 5-free except where noted):

| ratio | cents | character |
|-------|-------|-----------|
| 1/1   | 0     | anchor |
| 8/7   | 231   | wide second, bright |
| 13/11 | 289   | dark neutral-third color |
| 14/11 | 418   | supermajor third, glassy |
| 11/8  | 551   | floating not-fourth-not-tritone |
| 13/8  | 841   | neutral sixth, dusky |
| 7/4   | 969   | septimal seventh, workhorse |
| 2/1   | 1200  | octave |

Harmony discipline: chords are mostly otonal over the drone (7:8:11:13-family
voicings sharing a fundamental, so they lock). Undecimal/tridecimal intervals
appear inside chords rather than as bare melodic leaps. Utonal material (/7,
/11 under the tonic) supplies the dark floor in still sections.

**The illumination conceit:** prime 5 appears exactly once. The whole cave is
5-free; at the central arrival a 5/4 blooms inside the crystal chord — the
most human interval touching the structure for the first time — then recedes.
The third is an event, not a default.

Tonic ≈ **F#1 (~46 Hz)** (F–G# kick-key rule; G/49 Hz already owned by
ninth_wave).

## Spectral co-design: the tri-free rule

Every additive/FM voice **omits all partials divisible by 3** (no 3rd, 6th,
9th, 12th, …) and **down-weights partials divisible by 5**. Crystal spectra
use partials {1, 2, 4, 7, 8, 11, 13, 14, 16, 22, 26, …} — timbres made of the
same numbers as the scale. No sound in the piece contains an acoustic fifth.

- This is a subtractive contract, chosen for auditability after anneal showed
  co-design (warp-rule style) is hard to verify by ear alone. Verification:
  A/B audition renders (full-harmonic vs tri-free versions of each patch) plus
  spectrum inspection via the analysis artifacts.
- FM voices can regenerate 3·f via sidebands; FM ratios get audited against
  the spectrogram, falling back to additive where FM misbehaves.
- Engine focus: **additive and FM primary**; subtractive engines only in
  non-pitched supporting roles (noise, air) where the rule is moot.

## Voices (~6)

1. **Floor** — kick and sub as one object (Autechre-style): `drum_voice` kick
   tuned to the tonic with a long sub tail that *is* the drone. Sparse utonal
   dyads ghost beneath in still sections.
2. **Raindrop arp** — the de-literalized drip: `TonePool` weighted toward
   consonant degrees, stochastic/unpredictable onset timing (probability
   gates; no grid feel), FM-bell or additive pings, mostly reverb.
3. **Crystal strikes** — struck modal/additive voices with `modal_coupling` +
   `modal_dispersion`, tri-free spectra; sparse punctuation that accumulates.
4. **Bowed crystal** — sustained bow/rub-family or additive swell voice,
   reserved mainly for the illumination passage.
5. **Air** — `found_empty_room`-style floor, barely audible, clean.
6. **Ticks** — faintest metallic percussion, beat sections only, near
   subliminal.

## Space: stacked reverbs

Shared sends at two-to-three depths: a *close* darker reverb (wet rock
nearby) and a *vast* one (the unreachable dark), possibly chained serially so
tails re-reverberate. Target: enormous, unworldly, but clean.

This is a **delegated DSP workstream** (Opus subagent and/or Codex): candidates
are (a) a proper FDN reverb with very long, modulated, dark tails as a new
native effect, or (b) a curated serial/parallel stack of existing
Bricasti/Dragonfly/native stages with tone-shaped returns. Outputs are
auditioned A/B before the piece commits to either.

## Beat

Chill-Autechre register: clean subby kick, unhurried, subtly irregular
pattern, never busy — a skeleton that materializes and dissolves rather than
drives. Comes in and out across the form.

## Form (~8 minutes)

- **I. Dark adaptation** (0–2:00): near-silence, air, first raindrops; the sub
  drone fades up until you realize it was always there.
- **II. Skeleton** (2:00–4:00): the kick coheres out of the drips; arp
  establishes; crystals begin answering.
- **III. Illumination** (4:00–5:30): beat drops away; bowed crystals ring
  together; the lone 5/4 blooms and fades. Quiet arrival, not a climax.
- **IV. Recession** (5:30–8:00): beat returns sparser, then thins; drips slow;
  the dark floor is the last thing left.

## Aliveness (designed in, not bolted on)

- **Tempo**: a `TempoMap` with slow section-level breathing; rubato feel in
  the still sections (I, III); the beat sections hold a steady-ish pulse that
  still drifts a few BPM over minutes.
- **Timing**: score-level `timing_humanize` drift; the raindrop arp's timing
  is stochastic by construction.
- **Velocity**: per-note velocity shaping everywhere; `velocity_humanize` with
  shared `velocity_group`s so ensemble breathing is correlated, not
  per-voice wobble.
- **Automation arcs**: no held tone sits still — drone swell arcs, brightness
  rides on crystals, reverb-send rides across sections, arp density evolution,
  air level breathing. Exponential shape for all `_hz` targets.
- **Pitch motion**: `ratio_glide` between drone chord changes; slow shallow
  vibrato blooming on sustained bowed notes; drift bus subscription so held
  material breathes as one correlated body.
- **Envelope**: `envelope_humanize` on sustained voices; curved ADSR powers
  for acoustic-feeling decays.

## Verification / iteration methodology

- Early audition pass before full composition: (1) scale/chord audition
  sketches (is 13 in or out? which voicings lock?), (2) tri-free vs
  full-harmonic patch A/Bs, (3) reverb-stack candidates A/B.
- Snippet renders (`make snippet` / `make render-window`) for local iteration;
  full renders checked against analysis artifacts (chromagram should show the
  no-3 world; spectrogram audits the tri-free rule).
- LLM evaluation pass once the piece stands.

## Decisions log

- CPS deliberately rested for this piece (FUTURE.md updated; 3 CPS pieces
  exist).
- Colundi and stretched-octave co-design rejected for repetition with anneal.
- Fifths omitted via subgroup choice (no prime 3) rather than by scale
  curation — the constraint lives in the tuning system itself.
- Starker bass option chosen: kick/sub as one object rather than a separate
  drone voice above a dry kick.
- ~8 minute target; prime-5-at-illumination conceit confirmed.
