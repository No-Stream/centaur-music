# Repository Guide

## Overview

- `code_musics/` is the main package.
- `docs/synth_api.md` documents synth engines, presets, and engine-specific params.
- `docs/score_api.md` documents the concrete `NoteEvent` / `Phrase` / `Voice` /
  `Score` API, render-time expression controls, and the render path.
- `docs/composition_api.md` documents the higher-level composition helpers plus
  phrase-building helpers that sit above the core score model.
- `code_musics/synth.py` contains low-level DSP, rendering, and effect helpers.
- `code_musics/score.py` contains the composition abstractions: `NoteEvent`,
  `Phrase`, `Voice`, `Score`, `EffectSpec`, and `VelocityParamMap`.
- `code_musics/humanize.py` contains timing, envelope, and velocity humanization
  specs plus the drift helpers they build on.
- `code_musics/engines/` contains the synth engine registry and per-engine renderers.
  `_mpe_utils.py` contains shared MPE/MIDI utility functions for instrument engines.
- `code_musics/tuning.py` contains small just-intonation, harmonic-series, utonal,
  and EDO helper functions.
- `code_musics/generative/` contains algorithmic and stochastic composition
  tools: TonePool (weighted pitch pools), euclidean rhythms, probability gates,
  Markov chains, Turing machine sequencers, harmonic lattice walkers, and
  stochastic cloud generators. All are seeded/deterministic, work in ratio
  space, and produce standard Phrase/RhythmCell types.
- `code_musics/pieces/` contains named musical works that can be rendered by the
  registry, including smaller themed study modules plus JI subpackages.
- `code_musics/composition.py` contains phrase-first composition helpers (`line`,
  `ratio_line`, `canon`, `progression`, etc.) and the metered-time grid layer.
- `code_musics/automation.py` contains `AutomationSegment`, `AutomationSpec`, and
  `AutomationTarget` for score-time parameter motion.
- `code_musics/pitch_motion.py` contains per-note pitch motion specs (glide,
  vibrato, bend).
- `code_musics/smear.py` contains loveless-inspired pitch smearing and textural
  thickening tools for shoegaze/dream-pop aesthetics.
- `code_musics/harmonic_drift.py` contains JI-aware consonance-shaped pitch drift
  trajectories for slow timbral evolution.
- `code_musics/drum_helpers.py` contains `setup_drum_bus()` and `add_drum_voice()`
  convenience helpers for percussion voice setup and routing.
- `code_musics/render.py` is the named-piece orchestration layer.
- `code_musics/analysis.py` contains render analysis, artifact-risk warnings, and
  librosa-based visual analysis (mel spectrogram, chromagram, spectral contrast).
- `code_musics/inspection.py` contains the timestamp inspector for score context.
- `code_musics/spectra.py` contains spectral profile helpers for the additive engine.
- `code_musics/midi_export.py` exports score-backed pieces as shared tuning files
  plus per-voice MIDI stems.
- `code_musics/stem_export.py` exports per-voice audio stem WAVs with send bus
  returns and optional mastered reference mix.
- `code_musics/midi_import.py` imports MIDI files into the score model.
- `code_musics/meter.py` contains the optional high-level musical-time layer:
  `Timeline`, beat/bar helpers, rhythmic values, and bar-aware location math.
- `code_musics/evaluate.py` is the LLM-based piece evaluation system.  Four
  judges (Opus 4.6, Sonnet 4.6, Opus 4.5, Sonnet 4.5) score pieces across
  five dimensions via Claude Code headless.  `code_musics/eval_rubric.py`
  defines the rubric, dimensions, and prompt templates.
- `main.py` is the main entrypoint for listing and rendering pieces.

## Composition Model

- `NoteEvent` is the atomic musical event. It stores timing, amplitude, and either a
  `partial` relative to `Score.f0_hz` or an absolute `freq`.
- `NoteEvent` also supports per-note `velocity`, `amp_db`, optional
  `pitch_motion`, and optional note-local automation.
- `Phrase` is a reusable collection of relative-time `NoteEvent`s. Use it for motifs,
  sequences, and transformed restatement.
- `Score.add_note(...)` is the escape hatch for one-off pedals, accents, blooms, and
  transitions.
- `Score.add_phrase(...)` is the main composing API. It supports placement transforms
  like `time_scale`, `partial_shift`, `amp_scale`, and `reverse`.
- `Voice` stores note events plus synth defaults, voice-level effects, pan,
  humanization settings, optional velocity-to-parameter mappings, and optional
  voice-time automation.
- `EffectSpec`, `VoiceSend`, and `SendBusSpec` also support score-time automation
  for wet/mix, send level, return level, and pan control surfaces.
- `Score` also supports shared named send buses so multiple voices can feed the
  same reverb, delay, or other return-style effect chain without duplicating
  insert effects per voice.
- When using shared send buses, usually leave the bus `return_db` at `0.0` and
  balance audibility with the voice fader plus per-voice `send_db`; treat
  `return_db` as a rare global-return trim rather than the main wet-level knob.
- `Score` owns the timeline, derives `total_dur`, renders audio, and can save a
  piano-roll plot.
- `Score.timing_humanize` applies render-time ensemble timing drift across the whole
  score, while voice-level humanizers shape envelope and velocity variation.
- `Voice.synth_defaults` and note-level `synth={...}` overrides accept an `engine`
  name, optional `preset`, and engine-specific params documented in `docs/synth_api.md`.
- Voices can enforce `max_polyphony` (including strict mono with `1`) and an
  optional simple `legato` mode for overlap-driven non-retrigger behavior.
- `code_musics.composition` includes phrase-first helpers for melodic
  writing and section building. Phrase creation: `line(...)` /
  `ratio_line(...)`. Articulation transforms: `concat(...)` /
  `overlay(...)` / `echo(...)`. Rhythmic transforms: `augment(...)` /
  `diminish(...)` / `rhythmic_retrograde(...)` / `displace(...)` /
  `rotate(...)`. Polyrhythm builders: `polyrhythm(...)` /
  `cross_rhythm(...)`. Placement: `sequence(...)` / `canon(...)`.
  Harmonic: `voiced_ratio_chord(...)` / `progression(...)`. The
  optional high-level timing layer adds `Timeline`, rhythmic values
  like `Q` / `E`, `Groove` templates with named presets and per-step
  velocity weighting, `tuplet(n, m, value)` for general tuplet
  durations, and grid-style helpers (`grid_line`, `grid_sequence`,
  `grid_canon`, `metered_sections`, `bar_automation`) that compile
  back down to the existing seconds-based score model. Full API
  details live in `docs/composition_api.md`.
- `code_musics.generative` provides algorithmic composition helpers
  including weighted pitch pools, euclidean rhythms, probability
  gates, Markov chains, Turing machine sequencers, lattice walkers,
  stochastic clouds, and generative rhythm tools: `prob_rhythm`
  (weighted metric probability), `AksakPattern` (additive meter
  with Balkan/Turkish presets), `ca_rhythm` / `ca_rhythm_layers`
  (cellular automata rhythms), and `mutate_rhythm` (stochastic
  groove variation). All are seeded/deterministic and produce
  standard `Phrase` or `RhythmCell` objects. Full API details live
  in the "Generative Composition Helpers" section of
  `docs/composition_api.md`.
- For detailed score-surface semantics, parameter meanings, and render-order
  behavior, read `docs/score_api.md`.
- Prefer musical notation compostion (bars, notes) over raw time.  

## Expression Model

- Prefer `amp_db` over raw `amp` for authoring levels. Linear `amp` still works, but
  dB is usually easier to reason about in mixes.
- Voices are LUFS-normalized by default (`normalize_lufs=-24.0`). This is the
  correct gain-staging approach for all tonal, melodic, and sustained voices — it
  ensures effects always see a consistent input level. Do not treat a synth engine's
  raw output level as the mix decision.
- **`mix_db` is for mix balance only, not gain staging.** Normalization handles gain
  staging; `mix_db` is the fader for balancing voices against each other. Similarly,
  `pre_fx_gain_db` is an intentional trim to drive effects harder or softer.
- **For percussive voices (kick, tom, noise hits), use `normalize_peak_db=-6.0`
  instead of `normalize_lufs`.** LUFS normalization is unreliable for transient
  content because the integrated LUFS varies with BPM and silence duration.
  `normalize_peak_db` normalizes to a target peak level, giving compressors and
  other effects a predictable input regardless of BPM or per-note `amp_db` values.
  The `kick_punch` and `kick_glue` compressor presets are calibrated for a
  `-6.0 dBFS` input peak.
- `normalize_lufs` and `normalize_peak_db` are mutually exclusive. Avoid
  `normalize_lufs=None` — prefer `normalize_peak_db` for any voice where LUFS
  normalization is inappropriate.
- `master_input_gain_db` trims the summed mix into `Score.master_effects`. Leave it
  at `0.0` by default; reach for it only when you intentionally want to change
  how hard the master bus glue/tone chain is being driven.
- `Score` auto-stages the summed post-fader mix into the master bus by default,
  so voice `mix_db` is mainly for balance rather than manual premaster loudness
  management.
- `code_musics/pieces/_shared.py` exports `DEFAULT_MASTER_EFFECTS` — a
  plugin-preferred default master chain (BritPre preamp → MJUC Jr vari-mu
  compression, with native saturation + compressor fallbacks). New pieces
  can use `master_effects=DEFAULT_MASTER_EFFECTS` for a "sounds finished"
  baseline. Pieces that define their own `master_effects` fully replace the
  default — no layering.
- Use note-level `velocity` for accents and phrasing. By default it affects loudness
  through `velocity_db_per_unit`, and it can also drive synth params through
  `VelocityParamMap`.
- `velocity_humanize` is voice-level and on by default when adding voices. Set it to
  `None` when you want a fully fixed/programmed result.
- `velocity_group` lets multiple voices share correlated velocity drift, which is the
  right tool for ensemble breathing rather than independent per-voice wobble.
- `envelope_humanize` is for subtle ADSR variation over score time. This is the
  current "env slop" surface.
- Per-stage ADSR curve powers (`attack_power`, `decay_power`, `release_power`)
  and the VCV-style `attack_target` overshoot are accepted as synth params (in
  `synth_defaults` / per-note `synth={...}`) and as automation targets. Defaults
  are `1.0` (linear, preserves legacy behavior). For acoustic-feeling voices
  try `decay_power=2.0` / `release_power=2.0`; for "pokey" analog attack tops
  try `attack_target=1.2`. See `docs/synth_api.md` for the full surface.
- `timing_humanize` is score-level. Use it for ensemble looseness and shared drift,
  not for rewriting rhythmic structure.
- `DriftSpec.style` supports `random_walk`, `smooth_noise`, `lfo`, `sample_hold`,
  and `smoothed_random` (Helm-style: random anchors at `rate_hz` crossfaded with
  a raised-cosine window — organic wobble distinct from the steppy `sample_hold`
  and the pink-ish `smooth_noise`).
- automation is the explicit parameter-motion surface. Use it for deliberate
  sweeps, bends, timbral motion, and wet/send/pan rides; use humanization for
  subtle living variation.
- `code_musics/modulation.py` adds a Vital-style per-connection modulation
  matrix. Every routing is a `ModConnection` (source -> destination with
  `amount`, `bipolar`, `stereo`, `power`, optional `breakpoints`, combine
  `mode`). Sources: `LFOSource`, `EnvelopeSource`, `MacroSource`,
  `VelocitySource`, `RandomSource`, `ConstantSource` (stereo pan-split),
  `DriftAdapter`. Attach via `Voice.modulations` or `Score.modulations`;
  register shared scalars via `Score.add_macro(name, default, automation)`.
  Complements `AutomationSpec` (timeline curves) rather than replacing it —
  matrix contributions combine after base automation per destination. MVP
  per-sample synth coverage is `cutoff_hz` on `polyblep` via engine
  `param_profiles`; other synth targets are sampled per-note at onset. See
  `docs/score_api.md` and `FUTURE.md` for full details and deferred work, and
  `code_musics/pieces/mod_matrix_study.py` for a worked example.
- **Pitch motion is a standard part of the composition surface**, not an optional
  extra. Melodic and sustained voices should almost always use `PitchMotionSpec`:
  lead voices get vibrato on sustained notes (increasing depth/rate with
  intensity), pad voices benefit from `ratio_glide` between chord changes, and
  `linear_bend` serves deliberate pitch gestures. A melody with no pitch motion
  sounds static and lifeless.
- **Use exponential automation (`shape="exp"`) for frequency-domain parameters**
  (cutoff_hz, hpf_cutoff_hz, any `_hz` param). Frequency perception is
  logarithmic, so linear sweeps sound unnatural — they rush through the bottom
  and crawl at the top. Linear (`shape="linear"`) is correct for dB, pan, drive,
  morph, and other perceptually linear parameters.
- When documenting or changing these APIs, keep `AGENTS.md` high-level and put the
  parameter-by-parameter details in `docs/score_api.md`,
  `docs/composition_api.md`, and
  `docs/synth_api.md`.

## Running Commands — IMPORTANT

**Never run bare `python` commands in this repo.**  The project uses `uv` for
environment management; a bare `python` or `pip` invocation will use the wrong
interpreter and will fail with import errors.

**Always use one of:**

```bash
make all                           # default quality gate: format-check, lint, compile, typecheck, full tests
make check                         # alias for make all
make list                          # list registered pieces
make render PIECE=harmonic_drift   # render a piece (adds --plot by default)
make render PIECE=harmonic_drift PLOT=0  # render without plot
make render PIECE=harmonic_drift ANALYSIS=0  # render without analysis (faster)
make render PIECE=harmonic_drift PLOT=0 ANALYSIS=0  # fastest iteration render
make inspect PIECE=ji_chorale AT=2:10    # inspect score context around a timestamp
make snippet PIECE=ji_chorale AT=2:10 WINDOW=12   # render a centered snippet
make render-window PIECE=ji_chorale START=130 DUR=12  # render an exact snippet window
make render-all                    # render every piece
make midi PIECE=ji_chorale          # export a full MIDI bundle
make midi-snippet PIECE=ji_chorale AT=2:10 WINDOW=12  # export a centered MIDI snippet bundle
make midi-window PIECE=ji_chorale START=130 DUR=12    # export an exact MIDI snippet bundle
make stems PIECE=ji_chorale         # export per-voice audio stem WAVs
make stems PIECE=ji_chorale DRY=1   # export dry (pre-effects) stems
make stems-snippet PIECE=ji_chorale AT=2:10 WINDOW=12  # export stem snippet
make stems-window PIECE=ji_chorale START=130 DUR=12    # export exact stem window
make test                          # run the full test suite
make test-selected TESTS=tests/test_score.py  # run a focused subset while iterating
make test-selected TESTS="tests/test_a.py tests/test_b.py"  # multiple files
make scratch SCRIPT=scratch/smoke.py  # run a scratch script (no prompt; scratch/ only)
make typecheck                     # basedpyright
make compile                       # syntax / bytecode compilation check
make lint                          # ruff check with bug-finding rules
make format-check                  # verify formatting without modifying files
make format                        # ruff format
make evaluate PIECE=slow_glass     # evaluate a rendered piece with LLM judges
make evaluate PIECE=slow_glass MODELS=opus  # single model (faster iteration)
make evaluate-all                  # evaluate all rendered pieces
make inspire                       # oblique strategy / musical inspiration prompts
```

For **read-only smoke-test scripts with zero side effects**, use `make scratch
SCRIPT=scratch/foo.py`.  The `make *` pattern is allowlisted so `make scratch`
runs without a permission prompt; raw `uv run python ...` invocations prompt
every time.  The target refuses any path outside `scratch/` (which is
gitignored).

**Strict scope.** `make scratch` is only for read-only inspection, engine
smoke tests, and character measurements — things you'd be happy running
twice by accident.  Do NOT use it for:

- writing files to the repo (including logs, audio, plots, caches)
- piece renders, MIDI export, stem export, evaluation (use `make render`,
  `make midi`, `make stems`, `make evaluate` — those have proper plumbing)
- network calls, subprocess spawning, env mutation
- anything you'd hesitate to run without reading carefully

If a script outgrows that scope, promote it to a proper test in `tests/`
or a named make target instead.

```bash
# Write the script first, then run via make:
cat > scratch/smoke_filters.py <<'EOF'
import numpy as np
from code_musics.engines._filters import apply_filter
y = apply_filter(np.random.randn(4410), cutoff_profile=np.full(4410, 1000.0),
                 sample_rate=44100, filter_topology="k35", resonance_q=4.0)
print(f"k35: peak={np.max(np.abs(y)):.3f}, finite={np.all(np.isfinite(y))}")
EOF
make scratch SCRIPT=scratch/smoke_filters.py
```

For edge cases where `make` targets don't fit, you can still fall back to:

```bash
PYTHONPATH=. uv run python main.py --list
PYTHONPATH=. uv run pytest tests/
```

The Makefile's `UV_RUN = PYTHONPATH=. uv run` variable handles this for all
standard targets.

`make all` is the default target and should be the normal final verification step
after meaningful code changes. It runs formatting verification, Ruff, Python
compilation, basedpyright, and the full test suite. Use narrower commands while
iterating if helpful, but do not stop there.
Make commands are preferred to `uv run` since they don't require human sign-off,
making for a better experience for both agent and human.

## Rendering Workflow

- `make list` — see all registered pieces.
- `make render PIECE=<name>` — render a named piece and save a piano-roll PNG.
  Default render path. Always capture the full output — do **not** pipe through
  `tail` or truncate in any way. The render emits loudness/peak stats, artifact-risk
  warnings (amplitude modulation, compression, brightness), and analysis manifest
  paths that are all important for diagnosing mix and DSP issues.
- `make render PIECE=<name> PLOT=0` — render without the plot.
- `make inspect PIECE=<name> AT=<timestamp>` — inspect a score-backed piece around
  a timestamp like `130` or `2:10`.
- `make snippet PIECE=<name> AT=<timestamp> WINDOW=<seconds>` — render a short,
  centered score-backed snippet for faster iteration.
- `make render-window PIECE=<name> START=<timestamp> DUR=<seconds>` — render an
  exact score-backed snippet window when you want repeatable boundaries.
- Pieces should expose either a `build_score()` function or a direct `render_audio()`
  function if they do not fit the score abstraction cleanly.
- **Rendering is slow** — a full piece takes multiple DSP passes and typically runs
  longer than real-time. When calling render commands via Bash, set a generous
  timeout (at minimum 3–5× the piece duration in wall-clock seconds; 180 s is a
  safe floor for most pieces). Prefer `make snippet` or `make render-window` when
  iterating on a local moment rather than re-rendering the whole piece.

## Musical Direction

The project is no longer just about proving that xenharmonic ideas work. Prefer music
that feels like music: shaped, intentional, and complete, even when it is strange.
Be ambitious! Use velocity, humanization, frequent and thorough parameter automation, effects, and so on to create rich, complex, alive pieces. Creativity is welcome and mistakes are cheap. Be not afraid!

**API feedback welcome:** If you encounter API surfaces in this library that feel
unintuitive, backwards, or have surprising defaults, flag them to the user rather
than silently working around them. The goal is a library that's delightful for
agents to compose with — bad ergonomics should be fixed, not tolerated.

Current aesthetic center:

- just intonation
- harmonic-series materials, including otonal and utonal writing
- a bias toward pleasant, listenable results without becoming timid or conventional
- willingness to move between gentle clarity and more chaotic or alien sections

Named inspirations already captured in repo notes:

- Bach
- Aphex Twin's simple and tender side
- Aphex Twin's chaotic and unstable side

If a task is about aesthetic direction, piece planning, or "what should this project
sound like?", also read the "Loose Inspirations" section at the bottom of this file
and `FUTURE.md` before making recommendations.

When adding or revising pieces, it is usually better to ask "how do we make this more
musical?" than "how do we make this weirder?" Weirdness is easy; satisfying form is
harder and more valuable here.

## Future-Facing Ideas

See `FUTURE.md` for way more ideas.

## Implementation Notes

- Prefer phrase-first composition, but keep direct note insertion available.
- Treat velocity, timing humanization, and envelope humanization as part of the
  normal composition surface, not as obscure implementation details.
- Keep low-level synthesis simple unless a task explicitly calls for DSP changes.
- Treat effect chains declaratively with `EffectSpec` on voices or the master bus.
- The additive engine supports explicit spectral partial sets plus optional
  onset-to-sustain spectral morphing; use that when tuning and timbre should be
  co-designed instead of assuming a plain harmonic ladder.
- The additive engine also supports Vital-style spectral morphs on the partial
  bank: `inharmonic_scale` (piano-stiffness / inharmonic drift),
  `phase_disperse` (pad-style spread without chorus), `smear` (pink amplitude
  leak into overtones), `shepard` (octave-ghost crossfade), and
  `random_amplitudes` (seeded stable 16-stage random mask). Optional
  `sigma_approximation=True` applies Lanczos sigma factors to reduce Gibbs
  ringing. See `docs/synth_api.md` and the `stiff_piano` / `dispersed_pad` /
  `smear_drone` / `shepard_bells` / `chaos_cloud` demo presets for usage.
- The `polyblep` engine supports an optional second oscillator via `osc2_*`
  parameters for detuned stacks and sub layers.
- The `va` engine provides 90s/00s Virtual Analog-flavored synthesis with two
  oscillator modes: `supersaw` (Szabo-accurate JP-8000 detune/mix law + 7-voice
  PolyBLEP bank with random phase and optional hard-sync) and `spectralwave`
  (partial-bank with continuous saw→spectral→square `spectral_position` plus
  optional `_spectral_morphs` layering). Supports dual-filter routing
  (single/serial/parallel/split), pre-filter waveshaper drive, and a resonant
  comb filter slot with keytracking for karplus-strong-ish bell character.
  Presets cover JP-8000 (`jp8000_hoover`, `jp8000_lead`, `supersaw_pad`),
  Access Virus (`virus_pad`, `virus_bass`, `virus_lead`), and Waldorf Q
  (`q_comb_pad`, `q_comb_bell`, `q_spectral_lead`) flavors. The new
  `apply_comb(...)` primitive in `_filters.py` is available to future engines.
  See `docs/synth_api.md` and `code_musics/pieces/va_showcase.py`.
- The `polyblep` and `filtered_stack` engines support eight `filter_topology`
  options: `"svf"` (2-pole ZDF state-variable, default), `"ladder"` (4-pole
  Moog-style with per-stage saturation + `bass_compensation`), `"sallen_key"`
  (Diva-style biting 2-pole with pre-filter asymmetric soft-clip under drive),
  `"cascade"` (4-pole Prophet-5-style cascade of independent 1-poles + peaking
  bandpass — no global tanh growl), `"sem"` (Oberheim SEM-flavored 2-pole with
  continuous LP→Notch→HP morph and bass-preserving gentle resonance),
  `"jupiter"` (Roland IR3109-flavored 4-pole OTA cascade with single global
  tanh feedback for creamy, less-peaky character; pair with `hpf_cutoff_hz`
  for Jupiter-8 dual-filter or leave at 0 for Juno-106), `"k35"` (Korg MS-20
  Sallen-Key with diode-clipped feedback via `k35_feedback_asymmetry` knob —
  the screamer), and `"diode"` (TB-303 3-pole diode ladder with feedback
  tap between stages 2 and 3 for acid squelch, ADAA + Newton solvers). The
  drum_voice engine also honors `filter_topology` on its post-mix voice
  filter so percussion voices can borrow any analog character (defaults to
  `"svf"` — existing drum patches unchanged). See `docs/synth_api.md` for
  the full per-topology parameter surface, and
  `code_musics/pieces/filter_palette_study.py` for an 8-topology A/B tour.
- The ladder filter supports a Diva-style **Newton-iterated ZDF solver**
  (`filter_solver="newton"`) that resolves the delay-free feedback loop
  implicitly per-sample. Engine-level `quality` param (`"draft"` / `"fast"`
  / `"great"` / `"divine"`, default `"great"`) picks the solver + internal
  oversampling factor. Audibly cleaner self-oscillation and drive behavior
  vs. the prior one-step-delay ADAA path. See `docs/synth_api.md` Quality
  Modes for the full mapping and `diva_bass_resonance`, `cs80_attack`,
  `prophet_pad`, `moog_acid_newton`, `sk_bite_lead`, `cascade_bass` presets
  for showcase patches.
- Oscillator imperfection params (`osc_asymmetry`, `osc_softness`,
  `osc_dc_offset`, `osc_shape_drift`) model analog VCO behavior in the
  `polyblep` and `filtered_stack` engines.
- `voice_card_spread` (0-3) replaces the old `voice_card` param for controlling
  inter-voice calibration variation, with named tiers from JI-conservative (1.0)
  through Oberheim-level (3.0). When set explicitly on a voice it also drives
  multiplicative per-voice attack/release scaling at the Score level (OB-Xd-
  style); `voice_card_envelope_spread` overrides that dimension independently.
- `analog_jitter` now also drives an OB-Xd-style per-sample CV dither layer
  (pitch ±0.05 semitone, cutoff ±3% at amount=1.0, 4 kHz one-pole smoothed)
  on polyblep + filtered_stack, stacked on top of the stable voice_card
  offsets; `analog_jitter=0` disables it.
- `voice_dist_mode` on `polyblep`, `va`, and `filtered_stack` adds a
  RePro-5-style per-note distortion slot applied *inside the engine's note
  loop, after the VCA and before the per-note buffers sum into the voice
  output*. Modes: `soft_clip` / `hard_clip` / `foldback` / `corrode` /
  `saturation` (reuses `apply_saturation`) / `preamp` (reuses
  `apply_preamp`). Chord tones distort independently, preserving harmonic
  identity instead of collapsing into the IMD mud that a post-mix shaper
  produces. Default `off`; paired params `voice_dist_drive`,
  `voice_dist_mix`, `voice_dist_tone`. See `docs/synth_api.md`.
- Audio-rate modulation coverage: `polyblep` accepts per-sample
  `pulse_width`, `osc2_detune_cents`, `osc2_freq_ratio` (in addition to
  `cutoff_hz`), and `va` accepts per-sample `osc_spread_cents`. Drive
  these from the new `OscillatorSource` in `code_musics/modulation.py` —
  a sibling to `LFOSource` with no 200 Hz cap, usable at audio rate for
  PWM, cross-osc FM, and detune modulation.
- `osc_phase_noise` (0.0–1.0) on `polyblep` and `va` adds per-sample
  phase-accumulator jitter — broadband zero-crossing texture distinct
  from `analog_jitter` (CV-rail dither) and `drift_bus` (slow correlated
  drift). Deterministic from note hash; per-oscillator independent
  streams.
- `filter_morph` enables continuous blending between filter modes (SVF:
  LP/BP/HP/Notch cycle; ladder: pole-tap blending for 24 -> 6 dB/oct slope
  control). Automatable.
- `feedback_amount` and `feedback_saturation` model Minimoog-style
  post-filter -> pre-filter feedback for thickening and growl. Ladder and
  SVF feedback summations inject deterministic 1e-6 bootstrap noise (seeded
  from the signal) so high-Q self-oscillation can wake from silence.
- `hpf_cutoff_hz` adds a serial 2-pole ZDF highpass before the main filter,
  modeling CS80/Jupiter-8 dual-filter architecture.
- `vca_nonlinearity` adds gain-dependent envelope saturation for OTA-based VCA
  character on attack peaks.
- The native effect chain includes a minimum-phase multi-band `eq` effect with
  ordered highpass, lowpass, bell, and shelf bands for routine tone shaping.
  Band params: highpass/lowpass use `cutoff_hz`/`slope_db_per_oct`; bell/shelf
  use `freq_hz`/`gain_db`/`q`. See `docs/synth_api.md` for full details.
- The native effect chain includes a stereo-linked `compressor` effect with
  feedforward/feedback modes, detector-path EQ bands, and voice-to-voice
  sidechaining plus lookahead for ducking/glue workflows.
- The native `saturation` effect defaults to a higher-fidelity two-stage
  analog-style path with optional clean low/high-band preservation; see
  `docs/synth_api.md` for the modern vs legacy behavior and parameter surface.
- The `preamp` effect provides flux-domain transformer saturation for
  analog-style warmth. Unlike the `saturation` effect (which uses waveshaping),
  `preamp` operates in the magnetic flux domain where bass naturally saturates
  more than treble, producing minimal intermodulation on harmonically rich
  material. Use `preamp` for gentle warmth/coloring (master bus, subtle voice
  color); use `saturation` for intentional distortion/drive effects.
- The native `bbd_chorus` effect is a Juno-faithful BBD-style stereo chorus
  with quadrature LFOs (true L/R decorrelation from mono input), cross-feedback,
  BBD-style pre/post bandlimiting, and an optional gentle compander. Presets:
  `juno_i`, `juno_ii`, `juno_i_plus_ii`, `dimension_wide`. Wet is summed with
  dry (not crossfaded) — `mix` controls wet level. Prefer this over the older
  native `chorus` when you want the recognizable Juno-106 / Dimension-D vibe.
- The `kick_tom` synth engine provides 808/909-style kicks and toms with
  optional multi-point envelopes (body amp, pitch, overtone), per-body ZDF
  SVF filter with envelope-modulated cutoff, FM body synthesis for
  harmonically rich attacks, and per-oscillator waveshaping (9 algorithms).
  The intended happy path is pairing it with the native drum-oriented
  compressor/saturation presets documented in `docs/synth_api.md`. Note:
  `drive_ratio` and `post_lowpass_hz` are deprecated; use the effect chain.
- The `metallic_perc` engine provides additive/FM metallic percussion (hihats,
  cymbals, cowbell, clave) with inharmonic partials and optional ring modulation.
  Supports optional multi-point envelopes for amplitude and filter cutoff, and
  configurable filter mode. Presets: `closed_hat`, `open_hat`, `ride_bell`,
  `cowbell`, `clave`. Use `engine="metallic_perc"` in `synth_defaults`.
- The `snare` engine provides 909-inspired snare synthesis with pitched body,
  comb-filtered wire buzz, and click transient. Supports optional multi-point
  envelopes for body amp, wire amp, body pitch sweep, and wire filter cutoff.
  Pair with `snare_punch`/`snare_body` compressor presets and `snare_bite`
  saturation preset. Presets: `909_tight`, `909_fat`, `rim_shot`, `brush`.
  Use `engine="snare"` in `synth_defaults`.
- The `clap` engine provides multi-tap noise burst synthesis for clap and snap
  sounds. Supports optional multi-point envelopes for body tail decay and
  overall amplitude shaping (gated claps). Presets: `909_clap`, `tight_clap`,
  `big_clap`, `finger_snap`. Use `engine="clap"` in `synth_defaults`.
- The `drum_voice` engine is a unified composable percussion synthesizer with four
  independent layers (exciter, tone, noise, metallic). It replaces the five separate
  drum engines with a single architecture where any synthesis mode can be combined
  with any other. Presets from all original engines are available. Shaper slots can
  dispatch to waveshaper algorithms, the modern saturation effect, or the preamp
  transformer model. Three ergonomic macros (punch, decay_shape, character) provide
  high-level perceptual control. `drum_voice` now also covers Machinedrum-inspired
  kernels: EFM 2-op FM bodies (`tone_type="efm"`), PI modal resonator banks driven
  by `spectra.py` mode tables (`tone_type="modal"` / `metallic_type="modal_bank"`),
  EFM cymbals (`metallic_type="efm_cymbal"`), E12-style sample exciters
  (`exciter_type="sample"`), and digital-character voice shapers (`shaper="bit_crush"`
  / `"rate_reduce"` / `"digital_clip"`), with `pi_hardness` / `pi_tension` /
  `pi_damping` / `pi_damping_tilt` / `pi_position` macros for quick perceptual
  shaping. See `docs/synth_api.md` for the full parameter surface.
- The `sample` engine plays back WAV one-shots with pitch adjustment (map
  note `freq` against `root_freq`), optional `decay_ms` / `amp_envelope`
  shaping, a filter slot, and Machinedrum-E12-style macros (retrigger flams,
  pitch bend envelope, ring modulation, rate reduction, bit-depth grit).
  Samples resolve relative to the project root; loaded buffers are cached.
  Use `engine="sample"` with a `sample_path` param in `synth_defaults`.
- The waveshaper module (`_waveshaper.py`) now includes first-order ADAA
  anti-aliasing for 7 of 11 algorithms and optional 2x oversampling for the
  remaining fold-type algorithms.
- All drum engines share multi-point envelope support via `_envelopes.py`
  (linear, exponential, bezier interpolation). Any param ending in `_envelope`
  accepts a list of `{time, value, curve}` dicts. Shared utilities live in
  `_drum_utils.py` (RNG, bandpass noise, phase integration) and
  `_waveshaper.py` (11 distortion algorithms with ADAA for per-oscillator use).
- `Voice.choke_group` lets voices in the same named group cut each other on note
  onset (e.g., open/closed hi-hat pairs). See `docs/score_api.md`.
- `code_musics/drum_helpers.py` provides `setup_drum_bus()` and `add_drum_voice()`
  convenience helpers for percussion voice setup with sensible defaults
  (`normalize_peak_db=-6.0`, no velocity humanization, optional bus routing).
- The `bricasti` convolution wrapper supports basic wet-return tone shaping
  (`highpass_hz`, `lowpass_hz`, `tilt_db`) for cleaner, darker, or brighter tails.
- The local Linux environment has a small plugin palette installed for
  experimentation: `LSP` utilities in `~/.vst` and `~/.lv2`, `Dragonfly Reverb`
  in `~/.vst3` and `~/.lv2`, `TAL-Chorus-LX` and `TAL-Reverb-2` in `~/.vst3`,
  and legacy Linux `Airwindows` VSTs in `~/.vst`.
- `Chow Tape Model`, `TAL-Chorus-LX`, `TAL-Reverb-2`, Dragonfly VST3s,
  `Airwindows Consolidated`, `BYOD`, and `ChowCentaur` have all been verified
  to load through `pedalboard`.
- `Airwindows Consolidated` is wrapped via `apply_airwindows()` with algorithm
  switching (Density, IronOxide5, ToTape6, Tube, Drive, Coils, Channel9, etc.)
  and named presets. Algorithm selection works by patching the plugin's VST3
  preset XML. See `docs/synth_api.md` for the full algorithm/parameter table.
- `BYOD` is wrapped via `apply_byod()` with 40 built-in program presets
  (Tube Screamer, Centaur, American Sound, Big Muff, RAT, etc.) selectable
  via the `program` parameter. Parameters change dynamically per program.
- `ChowCentaur` is wrapped via `apply_chow_centaur()` with gain/treble/level
  controls and Neural/Traditional modes. At low gain settings it works well
  as a subtle warmth/color tool.
- The macOS environment has a broader plugin palette: all ChowDSP plugins
  (CHOWTapeModel, BYOD, ChowMatrix, ChowCentaur, ChowKick, ChowMultiTool,
  ChowPhaser), TAL (Chorus-LX, Reverb-2, Reverb-3), Dragonfly (Plate,
  Room, Hall, Early Reflections), Airwindows Consolidated, and Surge XT.
  LSP Plugins are Linux-only and unavailable on macOS.
- Render analysis generates librosa-based visual analysis:
  mel spectrogram, tuning-aware chromagram (36 bins/octave for JI
  visibility), spectral contrast, onset envelope, and harmonic-percussive
  separation balance — all registered in the analysis manifest. These are
  the primary feedback channel for AI-assisted composition since LLMs
  cannot hear audio.
- The `organ` synth engine provides drawbar organ synthesis with tonewheel drift,
  key click, scanner vibrato/chorus, crosstalk, and per-drawbar harmonic shaping
  (`tonewheel_shape`). It supports custom drawbar ratio sets for xenharmonic
  timbre-harmony fusion. Use `engine="organ"` in `synth_defaults`.
- The `piano` synth engine uses modal synthesis with physical hammer-string
  interaction (nonlinear contact model, second-order resonator bank). Velocity
  naturally shapes timbre through the hammer physics. It supports unison strings
  with drift, soundboard coloring, body saturation, damper noise, and custom
  partial ratio sets for xenharmonic timbre-harmony fusion. Use
  `engine="piano"` in `synth_defaults`. The legacy additive piano is available
  as `engine="piano_additive"`.
- The `harpsichord` synth engine uses pluck excitation + modal resonator
  synthesis with multi-register blending (front 8', back 8', 4', lute),
  per-note spectral morphing, and velocity-driven brightness. It supports
  custom partial ratio sets for xenharmonic timbre-harmony fusion. Use
  `engine="harpsichord"` in `synth_defaults`.
- `Voice` supports engine-agnostic sympathetic resonance via
  `sympathetic_amount`, `sympathetic_decay_s`, and `sympathetic_modes`. This
  adds a resonator bank tuned to active note harmonics, applied after note
  mixing and before normalization. Works best with harpsichord and piano
  voices where harmonically related notes reinforce each other.
- `Score.add_drift_bus(name, rate_hz, depth_cents, seed)` registers a shared
  slow pitch-drift bus; voices subscribe via `drift_bus=<name>` and
  `drift_bus_correlation` (0.0 = fully independent, 1.0 = fully shared). Held
  chords on subscribed voices breathe as a correlated unit, replacing the
  per-voice-independent analog drift with modular-rack-style shared CV.
  Defaults: no bus and no subscription, so existing pieces are unchanged.
- The `surge_xt` instrument engine renders voices through Surge XT via
  pedalboard's VSTi hosting. It uses MPE-style per-note pitch bend
  (48-semitone range) for microtonal accuracy. Unlike the native per-note
  engines, it renders the whole voice at once. Use `engine="surge_xt"` in
  `synth_defaults`.
- The `vital` instrument engine renders voices through Vital via
  pedalboard's VSTi hosting. Same MPE per-note pitch bend approach as
  surge_xt with sub-cent microtonal accuracy. Vital ignores standard MPE
  RPN for bend range, so `mpe_enabled` and `pitch_bend_range` are set
  via the parameter API. Use `engine="vital"` in `synth_defaults`.
- `code_musics/smear.py` provides loveless-inspired pitch smearing tools:
  `SmearVoice` for per-voice micro-detuned layering, `SmearChord` for
  whole-chord tremolo-bar drift, and helpers for building shoegaze/dream-pop
  textures from score phrases.
- `code_musics/harmonic_drift.py` provides JI-aware pitch drift automation
  shaped by consonance — slow, smooth trajectories that weight their path
  toward harmonically related intervals rather than drifting randomly.
- The additive engine supports advanced timbral features beyond basic
  harmonic stacks: per-partial envelopes, noise hybrid bands, spectral
  gravity (attraction toward JI intervals), and stochastic flickering as
  engine params. `code_musics/spectra.py` provides builder functions for
  physical model spectra (membrane, bar, plate, tube, bowl), spectral
  convolution, fractal spectra, and formant shaping/morphing. See
  `docs/synth_api.md` for the full parameter surface. Two study pieces
  demonstrate these features: `vowel_cathedral` (vowel formant morphing
  with spectral gravity) and `struck_light` (inharmonic strikes resolving
  to fractal drones) in `code_musics/pieces/additive_studies.py`.
- The additive engine's partial noise bands support `noise_mode="flow"`
  plus `flow_density` (0-1) for a Mutable Instruments Elements-style
  rare-event sample-and-hold exciter.  Produces breath/brush-like
  organic texture that plain white noise and S&H cannot.  `brush_breath`
  and `brush_cymbal` presets demonstrate the range.  The `breath_study`
  piece in `code_musics/pieces/breath_study.py` shows it in context.
  Underlying primitive is `flow_exciter()` in
  `code_musics/engines/_dsp_utils.py`.
- Keep plugin notes here high-level. Detailed parameter semantics and any new
  `EffectSpec` integration still belong in `docs/synth_api.md`.
- **Docs are part of implementation, not cleanup.** When you add, change, or
  remove a public surface (engine, effect, score/composition API, generative
  helper, export path, etc.), update the relevant docs in the same pass.
  `AGENTS.md` gets a short high-level callout; the full parameter/signature
  reference lives in `docs/synth_api.md`, `docs/score_api.md`,
  `docs/composition_api.md`, or a dedicated `docs/<topic>.md` file. Do not
  defer doc writes to a later pass — stale or missing docs cause future agent
  work to stall. A code change is not complete until the docs match.
- Timestamp inspection is part of the normal workflow. Prefer the timeline
  artifacts and `make inspect` when responding to comments like "2:10 in
  ji_chorale" instead of manually hunting through score code.
- Snippet rendering is part of the normal workflow for score-backed pieces.
  Prefer `make snippet` or `make render-window` when iterating on a local moment
  instead of re-rendering the whole piece.
- MIDI export mirrors that workflow for score-backed pieces via shared tuning
  files plus per-voice stems; prefer `make midi-snippet` or `make midi-window`
  when iterating on a local passage for DAW work.
- Audio stem export writes per-voice WAVs (wet or dry), send bus returns, and an
  optional mastered reference mix to a bundle directory with a JSON manifest.
  Wet stems + send returns sum to the pre-master mix. Dry stems are
  post-normalization, pre-effects/pan (mono). Use `make stems PIECE=...` or
  `make stems-window` for windowed export.
- Score analysis includes timing-drift diagnostics. Overall drift is fine, but
  inter-voice spread should stay musically plausible across the piece; keep the
  warning thresholds and artifact details documented under `docs/`.
- Render analysis records effect-chain diagnostics for compressors,
  saturation, and plugin stages so agents can see gain reduction, clipping-like
  density, and "mostly inactive" vs aggressive behavior in the analysis manifest.
- WAV export logging reports peak, true-peak, and integrated LUFS at write
  time, and warns when an exported master lands suspiciously far below the
  expected ceiling.
- WAV export defaults to 24-bit PCM via `soundfile`. 16-bit (TPDF dithered)
  and 32-bit float are available via the `bit_depth` parameter on `write_wav()`.
- Render analysis emits artifact-risk warnings for suspicious
  brightness, modulation, compression/clipping, and risky filter-motion
  parameter combos; when touching those surfaces, update docs and tests in the
  same pass.
- When using or extending synth engines, read `docs/synth_api.md` first for the
  current engine names, presets, and parameter surface.
- Prefer absolute imports and typed, readable Python.
- Add tests for score timing, phrase transforms, tuning math, and piece integration
  when changing behavior.
- If a doc in this repo describes something as future work, verify it against the code
  before repeating it. Several foundational ideas are already implemented.
- Think carefully about interface design. Imagine you were interacting with devices with
  practically no context. What would be intuitive and easy to use?
  Prefer absolute units when possible (like msec for a compressor). For arbitrary knobs,
  use typical, usable ranges: 0-0.25 should be quite subtle, 0.25-0.33 should be clearly audible but
  still somewhat subtle, 0.33-0.66 should be moderate, and 0.66-0.8 should be strong. It's fine
  to have plausibly musical but not broken very strong effects from 0.8-1.0.
  For example, a saturation effect might offer gentle mix warmth at 0.2, harmony-compatible warmth at 0.33,
  musical saturation at 0.5, strong but still musical saturation at 0.66, and distortion at 0.8.
- Effects and voices should be designed considering musicality, not textbook designs.
  Don't cut corners to save time.
  For example, if a distortion plugin would benefit from antialiasing,
  don't just implement the most naive algo.

## Agent Delegation Policy

Core composition and piece writing should happen in the main agent context for
creative continuity and direct authorial control. Mechanical work (new engines,
features, infrastructure, repo improvements, portability fixes) follows the normal
delegation-to-subagents pattern.

## Test Philosophy

- Test early and often. Ideally write tests *first* then code (TDD).  
- Focus on realistic, e2e tests (smoke, integration).  
- No need for trivial tests or testing each unexpected edge case. First and foremost, tests should validate that code runs properly, end to end, without major bugs; and they should prevent regressions.
- Backward compatibility is not always required or expected; this is a local, creative library not a business one.
- No fallbacks. Fail fast!

---

## Loose Inspirations & Ideas to Explore

- ji
- septimal harmony
- harmonic series, otonal and utonal
- aphex twin, both the simple and pleasant side (avril, aisatsana) and the chaotic side
- bach, the GOAT, and a very algorithmic composer for his time; useful inspiration both in barqoue-adjacent styles and broadly
- making xenharmonic ideas sound weird is pretty easy, making them sound pleasant is harder. let's bias in the pleasant direction, but not be constrained to narrow ideas of what pleasant is
- why are most xenharmonic pieces more like experimentations or sketches and not complete pieces? let's err on the side of making music that sounds musical. which shouldn't wall off experimentation and riffing, but can we make full compositions?
- commas and quirks of tuning systems (but musically; think Giant Steps, not pure studies)
- bittersweet, haunting. (think burial - untrue or MBV loveless)
- spacious, reverb, lofi (BoC, m83's dead cities)
- spicy but euphonic chords, creative voicing, voice leading, sevenths and septimal harmony
- it's fine to explore and fail. we can try a bunch of ideas, and none of them have to work
- think of music creation like a GAN: we come up with ideas, we see how we like them, and we iterate
- four tet (pretty and euphonic but not cheesy, wandering arps, cool organic sounds)
- autechre (creative, glitchy, weird, but still poppy at its core)
- emotional journey side of techno, think carl craig's "at les"
(for more ideas, use `make inspire`; this is an idea generator inspired by Eno's *Oblique Strategies*)

some tuning schemes -

- ji 5-limit
- ji 7-limit
- harmonic series
- colundi
- otonal
- meantone and other historical tunings (constraints are good)

```text
! 7-limit JI.scl
!
7-limit Just Intonation scale. Simple 5-limit JI with septimal tritone (7/5) and septimal seventh (7/4).
 12
!
16/15
9/8
6/5
5/4
4/3
7/5
3/2
8/5
5/3
7/4
15/8
2/1
```

```text
! colundi_ji_core.scl
!
Approximate Colundi-inspired 7-note JI scale
7
!
11/10
19/16
4/3
3/2
49/30
7/4
2/1
```

- EDOs/TETs
- Meantone, well temperaments, and baroque tunings
- Bohlen Pierce?
- Other?

Note: typically stick in the keys of F through G# for songs with an
electronic kick for maximum impact. For more ambient/chill pieces, there are
no restrictions, but don't default to Cmaj/Amin — consider randomizing the
key or picking one intentionally.

### Automation Ideas

Some classic automation targets to consider:

1. Filter cutoff (LP/HP)
2. Volume rides (especially pads, textures, background elements)
3. Reverb send amount
4. Delay send amount
5. EQ high shelf / low-end roll-off
6. Element dropout (kick, hats, snare out for 1-8 bars)
7. Delay feedback (runaway into transitions)
8. Sidechain depth
9. Drive/saturation amount
10. Reverb decay time
11. Stereo width (dry/wet on widener, or mid-side balance)
12. White noise / riser sweeps at transitions (or genre appropriate analogues)
13. Filter resonance (independent from cutoff)
14. Note length / gate time (staccato ↔ legato between sections)
15. Reverse reverb tails before key hits
16. Pan / auto-pan rate on hats and percussion
17. Swing/groove and humanization (imperfection, drift)
Not limited to the above. Creativity is encouraged. Automation can happen on multiple timescales (e.g. 4 bar and piece level).
