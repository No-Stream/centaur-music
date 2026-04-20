# Unified Composable Synth Voice Engine — Design Plan

## Context

The drum-voice unification (commit `0445d12`, design doc
`docs/plans/2026-04-15-unified-drum-voice-design.md`) replaced five siloed
percussion engines with one `drum_voice` composed of four parallel layers
(exciter / tone / noise / metallic) plus a shared post-chain. The win was
**cross-pollination** — you can now stack an FM body with an inharmonic
modal-bank metallic and a flow-noise exciter in a single voice, which no
prior engine exposed. FUTURE.md line 246 carries the one-line prompt:

> *"As with our drum voices, let's be able to share pieces b/w our synth voices."*

The tonal engine surface today sits in two de facto tiers:

- **Already partly unified** (share `_filters.py`, `_dsp_utils.py` analog
  glue, `_voice_dist.py`): `polyblep`, `va`, `filtered_stack`, `fm`.
- **Siloed** (do not use shared filter or envelope surfaces): `additive`,
  `organ`, `piano`, `piano_additive`, `harpsichord`.

You can't today put additive partials through a Moog ladder, or stack a
Szabo supersaw under a 2-op FM bell, or add a flow-noise exciter to a
drawbar bank — each engine is its own playground. This locks agents into
one-engine-per-voice writing and hides the most interesting hybrid
timbres behind "build a new engine."

**Goal (v1):** ship a new `synth_voice` engine that internally composes
four parallel source slots (osc / partials / fm / noise) into a shared
serial post-chain (HPF → dual filter → VCA → voice shaper), with
flat-namespace params, preset merge, and three-to-four perceptual macros.
Old engines stay registered and alive (drum_voice precedent); new pieces
are steered toward `synth_voice`.

**Explicit non-goals (v1):** no routing matrix / modular patching (use
the existing `ModConnection` matrix if needed); no modal or physical
"operator" slots (deferred to FUTURE.md, medium priority); no
deprecation of old engines.

---

## 1. Architecture: Four Parallel Source Slots + Shared Post-Chain

Every tonal voice decomposes into at most four parallel source layers,
summed pre-filter. Each slot is string-dispatched on `{slot}_type`; set
`{slot}_type=None` to disable the slot. This mirrors `drum_voice`
exactly.

### `osc` slot — time-domain bandlimited oscillators

| Type | Description | Key params |
|------|-------------|------------|
| `polyblep` | Single PolyBLEP saw/square/triangle/sine/pulse, built-in `osc2_*` for detune/sub | `wave`, `pulse_width`, `osc2_*`, `hard_sync` |
| `supersaw` | Szabo-law 7-voice PolyBLEP bank (from `va`) | `osc_spread_cents`, `osc_mix` |
| `pulse` | Variable-width pulse with PWM target | `pulse_width` |

### `partials` slot — frequency-domain partial banks

| Type | Description | Key params |
|------|-------------|------------|
| `additive` | Full additive engine: per-partial amps/envs, rolloff, noise hybrid, flow-exciter, formant morph, Tenney gravity | `partials`, `brightness_tilt`, `odd_even_balance`, `formant_*`, `flow_density` |
| `spectralwave` | VA partial-bank with continuous saw→spectral→square morph, `_spectral_morphs` layering | `spectral_position`, `spectral_morphs` |
| `drawbars` | Fixed-ratio sine bank with drawbar amps (Hammond-style), optional per-drawbar shape | `drawbar_ratios`, `drawbar_amps`, `drawbar_shape` |

### `fm` slot — 2-op FM

| Type | Description | Key params |
|------|-------------|------------|
| `two_op` | Carrier + modulator (reuses `fm.py` primitives) with feedback, index envelope | `fm_ratio`, `fm_index`, `fm_index_decay`, `fm_index_sustain`, `fm_feedback` |

### `noise` slot — aperiodic texture

| Type | Description | Key params |
|------|-------------|------------|
| `white` | Raw white | (none) |
| `pink` | 1/f filtered | (none) |
| `bandpass` | Center/width bandpassed | `center_hz`, `bandwidth_ratio` |
| `flow` | Mutable Elements flow-exciter (rare events) | `flow_density` |

Each slot uniformly exposes: `{slot}_type`, `{slot}_level`,
`{slot}_envelope` (optional multi-point, else inherits voice ADSR),
`{slot}_shaper` (optional per-slot nonlinearity, dispatches to
waveshaper / saturation / preamp like drum_voice), plus type-specific
params.

---

## 2. Signal Flow

```text
   osc ─→[shaper]─┐
                   │
   partials ─→[sh]─┤
                   ├─×level×env─→ SUM ─→ HPF ─→ FILTER1 ─→ FILTER2? ─→ VCA ─→ VOICE_SHAPER ─→ OUT
   fm ─→[shaper]──┤                        (serial/parallel/split, VA-style dual filter)
                   │
   noise ─→[sh]───┘
```

- Parallel sum of enabled slots, identical to drum_voice.
- Voice-wide post-chain reuses the full shared `_filters.py` surface
  (all 8 topologies, ZDF/Newton solvers, feedback, morph), dual-filter
  routing from `va`, and the 3-way voice-level shaper dispatch
  (waveshaper algos with ADAA / saturation / preamp).
- HPF is the existing `hpf_cutoff_hz` 2-pole ZDF (CS80/Jupiter-style).
- VCA honors per-voice ADSR plus `attack_power` / `decay_power` /
  `release_power` / `attack_target`, same as today's engines.

---

## 3. Parameter Interface

**Flat namespace with slot prefix**, drum_voice-style. No nested dicts.

```python
{
    # sources
    "osc_type": "supersaw", "osc_level": 0.6, "osc_spread_cents": 18,
    "partials_type": "additive", "partials_level": 0.5,
    "partials_partials": [1, 2, 3, 5, 7], "partials_brightness_tilt": 0.4,
    "fm_type": "two_op", "fm_level": 0.3, "fm_ratio": 2.0, "fm_index": 1.5,
    "noise_type": "pink", "noise_level": 0.1,

    # post-chain
    "filter_topology": "ladder", "filter_cutoff_hz": 1200, "resonance_q": 0.7,
    "hpf_cutoff_hz": 40,
    "shaper": "preamp", "shaper_drive": 0.3,

    # ADSR + analog glue (unchanged from existing engines)
    "attack": 0.02, "decay": 0.3, "sustain_level": 0.7, "release": 0.4,
    "analog_jitter": 0.5, "voice_card_spread": 1.2,
}
```

Rules:

- Type-specific params are read only when the relevant `{slot}_type` is
  active; unused params are ignored (no errors on stale keys).
- `{slot}_type=None` or omitted → slot disabled, zero cost.
- A minimally valid voice is one active slot + an `osc_type` or
  `partials_type`; everything else has defaults.

---

## 4. Perceptual Macros

Four macros, all default to `None` (inactive). Each fans out to
underlying params via `_set_if_absent` so explicit user values always
win. Resolution order: **preset params → macro fill-in → user kwargs
win.** Macros are popped before `render()` reads params. Macro values
follow the CLAUDE.md convention (0.2 subtle, 0.33 clear-but-subtle, 0.5
moderate, 0.66 strong, 0.8–1.0 intense-but-musical).

| Macro | Fans out to |
|---|---|
| `brightness` | `filter_cutoff_hz` (exp-scaled), `partials_brightness_tilt`, small `osc_spread_cents` bump at high values, small `fm_index` increase |
| `movement` | filter env depth, a default LFO→cutoff mod (auto-registered to `Voice.modulations` if none present), chorus mix, `partials_phase_disperse`, `partials_smear` |
| `body` | `hpf_cutoff_hz` (inverse — low values push HPF down), `resonance_q` bump, `osc2_sub_level` / supersaw low-voice gain, partials odd/even balance toward even |
| `dirt` | voice `shaper_mix`, `shaper_drive`, per-slot shaper mode selection by threshold (low → `soft_clip`/`saturation`, mid → `preamp`, high → `hard_clip`/`foldback`), small `feedback_amount` bump at extremes |

Rationale for 4 not 3: drums only needed 3 because transient-vs-body
covers most perceptual variance. Tonal voices have a clearer
"spectral tilt vs. motion vs. weight vs. drive" four-way split, and
`dirt` is a real separate axis from `brightness` in practice.

Individual-module control is always available. Macros are an
ergonomics layer on top of full-surface control, not a wall in front.

---

## 5. Preset Model

Flat-dict presets merge on top of defaults, user kwargs merge on top of
presets, macros fill unset params last. Identical to drum_voice.

v1 ships **10–15 curated presets** that specifically demonstrate
cross-pollination — things that were impossible or ugly before. Not a
port of every polyblep/va/additive preset. Examples (final list
selected during implementation):

- `fm_bell_over_supersaw` — 2-op bell in `fm` slot + detuned supersaw
  in `osc` slot, ladder filter, slow pad envelope.
- `additive_pad_through_ladder` — additive partials (inharmonic scale
  0.02, formant-morphed) into a 4-pole Moog ladder with moderate drive.
- `drawbar_diode_acid` — drawbar partials into the TB-303 diode
  ladder, short envelope, `dirt=0.5`.
- `stiff_piano_sub` — additive stiff-piano partials + polyblep sub
  under, SEM morph filter, `movement=0.3`.
- `flow_exciter_pad` — flow-noise exciter + additive pad, ZDF SVF in
  notch morph, long `movement`.
- `shepard_lead` — partials shepard + polyblep hard-sync osc2,
  Jupiter 4-pole, modest `dirt`.
- `chaos_cloud_texture` — additive `random_amplitudes` +
  bandpass noise, diode filter, high `movement`.
- `virus_hybrid_pad` — supersaw + FM 2-op at 3:1 ratio, dual-filter
  serial routing (LP→HP), `body=0.6`.
- `formant_vowel_lead` — additive vowel morphs (ae→o), SVF bandpass,
  narrow vibrato via `PitchMotionSpec`.
- `tonewheel_drive` — drawbars + pink noise key-click approximation,
  K35 filter with resonance, `dirt=0.4`.

Plus 2–5 "basic" presets (`bright_saw_lead`, `warm_pad`, `soft_bass`)
so composers have obvious starting points.

---

## 6. Migration Strategy: Keep-Alive + Opportunistic

**Strategy ii from brainstorming.** No deprecation, no aliasing.

- `synth_voice` is registered alongside all existing tonal engines in
  `registry.py:55-72`. No existing engine is removed or renamed.
- `synth_voice.py` imports primitives from existing engines (e.g.
  `from .additive import _render_partial_bank`) instead of
  duplicating — same pattern drum_voice uses (`from kick_tom import
  _resonator_body`). Keeps DSP debt flat.
- `docs/synth_api.md` gets a new `synth_voice` section positioned as
  the **preferred engine for new composition**. Old-engine sections
  remain but get a one-line "for new voices, consider `synth_voice`"
  pointer at the top of each.
- `CLAUDE.md` gets a one-liner under "Implementation Notes" flagging
  `synth_voice` as the composable default.
- Pieces are ported opportunistically — when we revisit a piece and
  want cross-pollination we didn't have, port to `synth_voice`.
  No mass port.

**Organ handling (decision C from brainstorming):** `organ.py` stays
as-is. `synth_voice` supports drawbar-style additive via
`partials_type="drawbars"` + a preset with Hammond ratios. Key-click
and crosstalk stay in `organ.py` as its reason to exist. Piano and
harpsichord similarly stay as dedicated engines (hammer-contact /
pluck → modal-resonator topology doesn't fit the
source→filter→VCA model).

---

## 7. File Structure

New:

- `code_musics/engines/synth_voice.py` (~400 lines) — orchestrator,
  signal flow, slot dispatch, post-chain, mirrors `drum_voice.py`.
- `code_musics/engines/_synth_slots.py` (~600-800 lines) — per-slot
  renderers; each one thin, delegating to existing engine primitives
  (`additive._render_partial_bank`, `va._render_supersaw_bank`,
  `fm._render_two_op`, etc.).
- `code_musics/engines/_synth_macros.py` (~150 lines) — macro
  resolution, fan-out tables for `brightness` / `movement` / `body` /
  `dirt`.
- `tests/test_engine_synth_voice.py` — integration tests (one per
  slot type, one per macro, cross-pollination smoke tests).
- `tests/test_synth_voice_presets.py` — render each curated preset,
  verify non-silent + non-clipping + spectrally-plausible.
- `tests/test_synth_macros.py` — `_set_if_absent` semantics,
  resolution order, per-macro fan-out.
- `docs/plans/<today>-unified-synth-voice-design.md` — final design
  doc (this file, copied out of the plan workspace).

Modified:

- `code_musics/engines/registry.py` — add `"synth_voice": synth_voice.render`
  to `_ENGINES` (line ~55-72); add preset table; optionally opt into
  `_PARAM_PROFILE_AWARE_ENGINES` and `_VOICE_STATE_AWARE_ENGINES`.
- `docs/synth_api.md` — add `synth_voice` section mirroring the
  `drum_voice` section's structure; add "consider synth_voice" pointer
  at the top of `polyblep`/`va`/`filtered_stack`/`additive`/`fm` sections.
- `CLAUDE.md` — one-line implementation note under "Implementation Notes".
- `FUTURE.md` — add entries for modal/physical operator slots
  (medium priority), 4-op/6-op FM (low-medium), full modular
  patching matrix (low / opportunistic), and "two-of-a-kind slots"
  (two supersaws stacked) as low-priority extensibility.

Unchanged (no migration):

- `polyblep.py`, `va.py`, `filtered_stack.py`, `additive.py`, `fm.py`,
  `organ.py`, `piano.py`, `piano_additive.py`, `harpsichord.py`,
  `surge_xt.py`, `vital.py`, `sample.py`, `drum_voice.py` and
  all drum-related files.
- All existing pieces.
- All existing tests.

---

## 8. Implementation Plan

Implementation-phase work (not part of planning):

1. **Stub + dispatch skeleton.** Create `synth_voice.py` with the
   parallel-sum + post-chain skeleton, all slots returning zeros.
   Register in `registry.py`. Add one smoke test (empty voice renders
   silence without crashing). ~1 hour.

2. **`osc` slot.** Wire `polyblep`, `supersaw`, `pulse` to existing
   `_oscillators.py` + `va._render_supersaw_bank`. Tests for each
   type rendering plausibly. ~2 hours.

3. **`noise` slot.** Wire `white` / `pink` / `bandpass` / `flow`
   to `_dsp_utils.bandpass_noise` and `flow_exciter`. ~1 hour.

4. **`fm` slot.** Wire `two_op` to `fm.py` primitives. ~1 hour.

5. **`partials` slot.** Wire `additive` to
   `additive._render_partial_bank`, `spectralwave` to
   `va._render_spectralwave_bank`, `drawbars` to a thin
   drawbar-on-additive wrapper. ~3 hours (hardest slot — partial
   envelope + formant + flow-exciter integration).

6. **Post-chain.** HPF + dual-filter routing from `va` + VCA +
   voice-level shaper dispatch from `drum_voice`. Most of this is
   wiring existing code. ~2 hours.

7. **Macros.** `_synth_macros.py` with resolve function and fan-out
   tables. Test resolution order. ~2 hours.

8. **Presets.** Author + audition 10–15 curated presets. Iterate
   with `make render-window` until each sounds good. This is the
   creative-density work. ~4–6 hours + per-preset listening.

9. **Docs.** `synth_api.md` section, `CLAUDE.md` note, `FUTURE.md`
   additions, copy the final design doc from plan workspace to
   `docs/plans/`. ~1 hour.

10. **`make all` green.** Full suite passes including the new
    preset render tests. Fix anything discovered.

**Delegation plan:** steps 2–6 (mechanical wiring) are great subagent
work — each is a well-scoped "implement slot X by composing existing
primitive Y" task. Steps 7 (macros) and 8 (presets) want main-context
taste — macros because the fan-out mapping is a judgment call, presets
because they require creative density and iterative listening. Step 9
docs can be split: reference tables → subagent, prose/pointers → main.

---

## 9. Verification

- `make test-selected TESTS=tests/test_engine_synth_voice.py` — all
  slot-type smoke tests pass.
- `make test-selected TESTS=tests/test_synth_voice_presets.py` —
  every preset renders non-silent, non-clipping, with
  spectrally-plausible output (librosa-based spectral analysis in
  the test, mirroring the drum_voice preset-audit approach).
- `make test-selected TESTS=tests/test_synth_macros.py` — macro
  resolution order correct, `_set_if_absent` respects user
  kwargs, macro fan-out covers the documented targets.
- `make render PIECE=<a piece that uses synth_voice>` — author a
  small showcase piece (or extend an existing one) that uses the
  v1 presets, render end-to-end, listen, verify analysis manifest
  shows no artifact-risk warnings.
- `make all` — full suite green including format, lint, typecheck,
  all tests.

Manual listening verification on the curated presets is the real
acceptance criterion beyond automated tests — `dirt=0.8` should
sound "intense but musical," `movement=0.5` should sound clearly
animated, etc. This maps back to the CLAUDE.md knob-range
convention.

---

## 10. Critical Files to Modify

- `code_musics/engines/synth_voice.py` (new)
- `code_musics/engines/_synth_slots.py` (new)
- `code_musics/engines/_synth_macros.py` (new)
- `code_musics/engines/registry.py` (register engine + presets)
- `docs/synth_api.md` (new section + pointers)
- `CLAUDE.md` (one-line implementation note)
- `FUTURE.md` (operator slots, 4/6-op FM, modular matrix, two-of-a-kind)
- Test files listed in §7

Primary existing files to read/import from (not modify):

- `code_musics/engines/drum_voice.py` — architectural template
- `code_musics/engines/_drum_macros.py` — macro-resolution template
- `code_musics/engines/additive.py` — partial-bank primitives
- `code_musics/engines/va.py` — supersaw bank + spectralwave +
  dual-filter routing
- `code_musics/engines/polyblep.py` — osc primitives
- `code_musics/engines/fm.py` — 2-op FM primitives
- `code_musics/engines/_filters.py` — full filter surface
- `code_musics/engines/_dsp_utils.py` — analog glue
- `code_musics/engines/_waveshaper.py` — voice-level shaper dispatch
