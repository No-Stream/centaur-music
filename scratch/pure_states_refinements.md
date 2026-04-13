# Pure States — Refinement Pass

## User Feedback (post full-arrangement render)

### Concrete fixes

1. **Clap**: -2.5 dB (too prominent)
2. **Bass**: start simple (8th octave bounce), bring countermelody in later (Drop 2?)
3. **Lead inaudible ~2:35**: automation closing filter too far in outro/late drop2.
   Also: consider guardrails against broken automation values.
4. **Kick**: more punchy, less sustained. Quicker decay, less boom.

### More automation everywhere

- Chords (pad) need MUCH more audible automation — cutoff, resonance, env amt,
  chorus depth, send levels. Should feel alive.
- Bass: same story. More filter motion, drive variation.
- Stereo: pan automation on lead, pad width changes
- Sends: delay wet/feedback evolving, reverb send levels per section
- Distortion: lead tape drive, bass saturation evolving
- Decay/release on lead and bass envelopes
- Rule: **multiple automations always in motion at any moment**

### More humanization

- Already have velocity + envelope humanize on lead/pad
- Could add more: timing jitter, velocity on bass, envelope on bass

### Structural addition

- Coda/bridge/variation toward the end of Drop 2 or before outro
- Could be: chord substitution, lead variation, new element, tempo feel change

## Plan

### Step 1: Quick fixes (clap, kick, lead visibility)

- Clap mix_db: -7 → -9.5
- Kick: reduce body_decay_ms via params override. 240 → 160ms.
  Or use a tighter preset. Could also reduce body_punch_ratio.
- Lead automation: ensure cutoff never drops below ~800 Hz. Add clamp_min
  to the cutoff AutomationSpec.

### Step 2: Bass evolution

- Bars 17-48 (Drop 1): simple 8th-note root-octave bounce
- Bars 73-84 (Build 2): transition — maybe root-fifth pattern
- Bars 85-120 (Drop 2): full countermelody
- This makes the countermelody feel special when it arrives

### Step 3: Deep automation pass

For every voice, ensure multiple params are in motion at all times:

**Pad:**

- cutoff_hz: already there, verify audible
- resonance_q: subtle sweep 0.8 → 1.3 across drops, peak in breakdown
- filter_env_decay: slow → faster across the build (more attack character)
- chorus mix (via send or effect amount?): varies per section
- pan: subtle drift ±0.06
- hall send_db: grows into breakdown, pulls back for drops

**Bass:**

- cutoff_hz: already there
- filter_env_amount: already there  
- filter_env_decay: short → shorter (punchier over time)
- filter_drive: evolving — more squelch in later sections
- resonance_q: subtle motion

**Lead:**

- cutoff_hz: already there
- filter_env_amount: already there
- filter_drive: already there
- decay (amp env): shorter in climax (tighter notes)
- release (amp env): shorter in climax
- pan: slow drift
- delay feedback: grows slightly
- hall send_db: already there

**Hats:**

- Could automate brightness (freq param) — already section-based
- Could automate mix_db per section for dynamics

### Step 4: Coda/variation

Options:

- Bars ~109-116 (late Drop 2): introduce a chord substitution or new voicing
- Or: a "fake breakdown" — 4 bars where the lead drops to lyrical briefly
  before the climax cascade returns
- Or: bring back the breakdown Em(7/4) chord for one bar in Drop 2 as a callback
