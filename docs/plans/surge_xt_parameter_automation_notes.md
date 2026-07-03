# Surge XT Parameter Automation Notes

**Status:** research note. No implementation currently planned.

## Context

The `surge_xt` engine can render basic notes through the plugin host. The
remaining question is how to automate Surge XT's internal parameters, such as
filter cutoff and resonance, over score time without artifacts.

This note preserves the negative results from earlier exploration so future
work does not rediscover the same dead ends.

## Approaches Tried

### MIDI CC Automation

Engine params can send MIDI CC curves during render. The infrastructure works:
CC messages are emitted at the intended times.

The blocker is Surge XT patch state. The init patch has no CC-to-parameter
modulation routing, so the CC messages arrive and are ignored. This approach
should work if a loaded patch maps a CC, for example CC74, to the desired
internal target.

### Chunked Parameter Rendering

This approach split the render into short chunks, updated plugin parameters
between chunks, concatenated the output, and tried crossfading the boundaries.

It is not a viable path. It clicks and pops because the plugin's internal DSP
state changes discontinuously: filters, oscillator phase, feedback paths, and
other stateful blocks do not transition smoothly when host parameters step at
chunk boundaries. Shorter chunks and output-level crossfades do not solve this
because the artifacts are generated inside the plugin before audio reaches the
crossfade.

Do not retry this as the main automation strategy.

### Native Post-Processing Filter

The practical workaround is to render Surge XT with a fixed bright patch, then
apply a native score-time automated filter or EQ as a voice insert.

This is sample-accurate and artifact-free. It works for broad brightness motion,
but it cannot access Surge XT's internal filter character, resonance,
self-oscillation, or nonlinear feedback.

## Best Path Forward

Configure Surge XT's modulation matrix in the patch state, then drive that
routing with MIDI CC curves.

Likely implementation options:

- create a preset in the Surge XT GUI with CC-to-parameter routing, save it, and
  load the resulting state through `raw_state`
- programmatically generate or edit the plugin state if the state format becomes
  clear enough
- use a very slow internal LFO in the patch state for score-length sweeps when
  external CC is not needed

This keeps modulation inside Surge XT, where the filter and other stateful DSP
can respond continuously.

## Related Notes

MTS-ESP would complement this work for dynamic tuning changes without relying on
pitch bend, but it does not solve internal parameter automation by itself.

The existing global-bend chord mode with `mpe=False` is useful for
tremolo-bar-style harmonic glides: the bass note is exact, while upper voices
can have small quantization offsets that may be musically useful for shoegaze
textures.
