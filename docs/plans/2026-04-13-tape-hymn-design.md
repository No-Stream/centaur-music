# Tape Hymn — Design

## Concept

A slow, meditative 90-second piece at 72 BPM in 7-limit JI. Built on the
contrast between the resonator kick's warm bell-like ring and harsh 808
square-wave metallic hats. A four-part chorale pad enters gradually, breathing
through velocity-driven timbral arcs.

F0 = 55 Hz (low A). Three sections: awakening (sparse), hymn (full texture),
decay (dissolving).

## Voices

### Drums (5)

1. **Resonator kick** — `kick_tom`, `body_mode="resonator"`, velocity-timbre
   active (`velocity_timbre_decay: 0.3, velocity_timbre_brightness: 0.25`).
   Sparse half-note pattern, crescendo through hymn section.

2. **808 closed hat** — `metallic_perc`, `oscillator_mode="square"`,
   `partial_ratios=[1.0, 1.3348, 1.4755, 1.6818, 1.9307, 2.5452]`, ~8 kHz.
   Steady eighth-note pulse with velocity accents. Choke group "hats".

3. **808 open hat** — same engine/ratios, longer decay. Select beats only.
   Choke group "hats".

4. **FM snare** — `snare`, `body_fm_ratio=1.5`, `body_fm_index=2.5`,
   `wire_noise_mode="colored"`, velocity-timbre active. Backbeat in hymn.

5. **Maracas** — `noise_perc` preset "maracas", ~10 kHz. Sixteenth-note
   texture in hymn section only.

### Tonal (3)

1. **Chorale low** — `additive` engine, sustained JI chords. Warm pad with
   slow filter automation. Root, 3/2, 7/4, 9/8 voicings.

2. **Chorale high** — same engine, upper octave, thinner. Enters bar 13.

3. **Cassette kick layer** — `sample` engine, `../samples/Cassette808_SamplePack/
   Cassette808_Samples/Cassette808_BD01.wav`. Mixed underneath resonator kick
   at lower level for lo-fi warmth.

## Structure (72 BPM, 27 bars, ~90 s)

| Section | Bars | Content |
|---------|------|---------|
| Awakening | 1-8 | Kick every 2 bars (bar 1, 3, 5, 7). Closed hat enters bar 3. Cassette layer joins bar 5. Spacious. |
| Hymn | 9-20 | Full drum pattern. Chorale low enters bar 9. Snare backbeat. Chorale high enters bar 13. Maracas fill. Velocity crescendo bars 9-16, recede 17-20. Open hat accents. |
| Decay | 21-27 | Drums thin (kick + hat → kick alone bar 25). Chorale sustains and fades. Long reverb tail. |

## Effects / Routing

- Drum bus: `kick_glue` compressor + subtle saturation
- Hall reverb send: for chorale voices and open hat splash
- Master: `DEFAULT_MASTER_EFFECTS`
- `bar_automation` on chorale filter cutoff (low→high through hymn, back down)

## Features Showcased

- Resonator kick body mode
- Square-wave oscillator metallic_perc
- FM snare body + colored wire noise
- Velocity-to-timbre on all drums
- Sample playback engine (cassette kick layer)
- Maracas preset
- Accelerating clap (accent on peak moments)
- Choke groups (open/closed hat)

## Implementation

Single file: `code_musics/pieces/tape_hymn.py`
Register in `code_musics/pieces/__init__.py`
Use low-level position approach (like amber_room) for direct control.
