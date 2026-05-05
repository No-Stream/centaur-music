# Centaur VSTs — Plan (Path 2: JUCE shell + off-thread Python)

**Status:** planning doc only. No implementation to start.

**Date:** 2026-04-26

## Context

`centaur-music` has two flagship engines (`synth_voice`, `drum_voice`) and a
large native-effect palette that currently only run inside the project's
offline Python render pipeline. The user wants to access them in Ableton Live
as VST plugins (one instrument, one effect) without rewriting the DSP in C++
— that would throw away the work and re-open a months-long DSP correctness
problem. A slight performance hit is acceptable; parity with commercial
plugins is not required.

Three paths were evaluated:

1. **Render-to-sampler** — ships fastest but loses every stochastic /
   drifting / coupled element the project is actually about.
2. **JUCE shell + off-thread Python render daemon** — preserves live Python
   DSP. User chose this path.
3. **Full C++ rewrite** — what `megamegaseq` chose; explicitly out of scope.

This doc plans path 2 in detail. Two plugins are in scope:

- **`CentaurVoice`** — instrument VST3/AU hosting `synth_voice` + `drum_voice`.
- **`CentaurFX`** — effect VST3/AU exposing the native effect palette:
  the 8 analog filter topologies, drive + preamp, compressor + clipper +
  limiter, BBD chorus + Bricasti convolution + analog_filter effect.

## Architectural reality check

Three findings shape every decision below.

### 1. Engines are offline per-note whole-buffer renderers

`synth_voice.render()` (`code_musics/engines/synth_voice.py:53`) and
`drum_voice.render()` (`code_musics/engines/drum_voice.py:42`) share this
contract:

- Input: scalar `freq`, scalar `duration`, scalar `amp`, `sample_rate`,
  `params: dict`, optional `freq_trajectory: np.ndarray` of exactly
  `n_samples` length.
- Output: 1-D mono `np.ndarray` of length `n_samples = int(sr * duration)`.
- Internals assume the whole buffer is available: peak-normalize with
  `signal / np.max(np.abs(signal))` at the end, bake attack/release fades
  at the tail.
- No cross-note state in these two engines (`_VOICE_STATE_AWARE_ENGINES`
  in `code_musics/engines/registry.py:5418` lists only `polyblep` /
  `filtered_stack`).
- Pure NumPy + SciPy + Numba. No subprocess. No plugin hosting.

**Implication:** we cannot drop `render()` into a VST `process(buffer)`
callback. The plugin has to *schedule notes* (not stream samples), dispatch
each full note to the Python side, and play back the returned buffer with
latency equal to the render time plus a lookahead budget.

### 2. Embedded CPython on the audio thread is not viable

GIL contention + non-deterministic cyclic-GC pauses break the real-time
audio deadline. Sub-interpreters (PEP 684) only partially help and have
rough edges for NumPy. Nobody ships production plugins this way.

**Implication:** Python must run **off** the audio thread, in a worker
process, communicating via shared memory + lock-free queues. The audio
thread only ever does sample I/O from pre-rendered ring buffers.

### 3. `megamegaseq` chose pure-C++ and offers no Python-hosting pattern

What's still worth copying (`/Users/flatljan/personal/megamegaseq/`):

- Engine / bridge / UI directory separation.
- CHOC `WebView` + `view.bind()` JS↔C++ surface (`standalone/Main.cpp:186`).
- Three-mode bridge pattern (`choc` / `ws` / `mock`) in `ui/src/bridge.js`
  for testability.
- Build-time UI embedding (`tools/embed_ui.py` → header blob).
- Mock + WebSocket + Playwright test layers.

## Chosen approach

### Plugin process topology

```text
┌──────────────────────── DAW process ───────────────────────────┐
│  JUCE plugin (C++)                                             │
│  ┌──────────────────┐   ┌──────────────────────────────────┐  │
│  │  Audio thread    │   │  UI thread (CHOC WebView + JS)   │  │
│  │  - MIDI in       │   │  - preset browser                │  │
│  │  - note scheduler│──▶│  - param editor                  │  │
│  │  - ring buffers  │   │  - scope / meters                │  │
│  └────────┬─────────┘   └──────────────────────────────────┘  │
│           │                                                    │
│   shmem + lock-free queues (note requests / rendered buffers) │
│           │                                                    │
└───────────┼────────────────────────────────────────────────────┘
            ▼
┌──────── Python worker process (uv-managed) ────────────────────┐
│  - imports code_musics.engines                                 │
│  - consumes NoteRequest, calls synth_voice.render / drum_voice │
│  - writes rendered buffer to shmem, posts NoteDone             │
│  - warm-imports numba kernels on startup                       │
└────────────────────────────────────────────────────────────────┘
```

### Latency model

Fundamental tradeoff: notes must be rendered before their onset. Two
modes:

- **"Clip-playback" mode (primary)** — Ableton transport running, MIDI
  notes known ahead via look-ahead (JUCE's `getPlayHead()` reports
  transport + future-note metadata inside the current block). Plugin
  introduces a fixed lookahead (configurable: 100 / 250 / 500 ms);
  notes entering the lookahead window are dispatched to Python; the
  rendered buffer plays at the correct sample offset. This is the
  mode that actually works for the project's material.
- **"Live-play" mode (best-effort)** — no transport, MIDI keyboard
  input. Latency = render time. Acceptable for drums (short notes,
  fast render), painful for long pads. Document it as a known
  limitation.

### Note lifecycle (clip-playback)

1. Audio thread sees a MIDI note-on at sample `s_on` with velocity `v`,
   duration `d` (from look-ahead or MIDI note-off inference).
2. Plugin writes a `NoteRequest { id, freq, duration, amp, params_blob, sr }`
   to a lock-free SPSC queue in shared memory.
3. Python worker pops, calls `render()`, writes the resulting
   `float32` buffer into a shmem arena, posts
   `NoteDone { id, shmem_offset, n_samples }`.
4. Audio thread pops `NoteDone`, registers the buffer with the mixer
   at `s_on + lookahead_samples`, then on subsequent blocks mixes it
   into the output.
5. When the note's play cursor reaches `n_samples`, the buffer is
   returned to the free list.

### Parameter handling

`synth_voice` has ~60 params, `drum_voice` ~50. No typed registry
exists today. We need:

- A new module `code_musics/engines/_param_schema.py` that enumerates
  each engine's params with `{ name, type, range, default, unit,
  automatable }`. This is the same schema the user has already
  wanted for auto-calibrated effects (see `FUTURE.md:30`).
- VST3 param IDs = stable hashes of param names, frozen per release.
- Preset = flat dict of `{ param_name: value }`, same shape as
  `synth_defaults`. The 15 curated `synth_voice` + drum presets in
  `registry.py` `_PRESETS` become the factory bank.

Four-macro control surface (`brightness` / `movement` / `body` / `dirt`
on `synth_voice`; `punch` / `decay_shape` / `character` on `drum_voice`)
become first-class top-level knobs in the UI. These are already the
"performable" surface of the engines.

### Process boundary protocol

Binary, not JSON (audio-buffer throughput).

- Control channel (UI ↔ audio-thread): `choc::value` JSON over
  `WebView.bind()` — same pattern as megamegaseq.
- Audio channel (audio-thread ↔ Python): shared-memory ring with
  16-byte-aligned records. POSIX `shm_open` on macOS/Linux,
  `CreateFileMapping` on Windows. Lock-free SPSC via atomics.
- Control channel (plugin ↔ Python): small UDS / named-pipe for
  lifecycle (`boot`, `shutdown`, `preset-change`, `heartbeat`).
- Worker supervises itself with a heartbeat watchdog; if Python
  crashes the plugin substitutes silence + a red "engine offline"
  banner and respawns.

### CentaurFX specifics

All four effect categories are pure-numpy and work on full-buffer
inputs today. That's actually *easier* than the instrument case:
effects don't have the "know duration up front" problem — they can
run on arbitrary chunks.

Constraint to verify: several effects have internal IIR state
(filters, compressor detector, BBD chorus modulation, Bricasti
partitioned convolution). Running them on independent chunks
produces clicks at chunk boundaries. Two choices:

- **Chunk-with-overlap-and-fade** — simplest; ~2–5 ms audible
  transient suppression. Fine for pad / sustained material,
  audible on percussion.
- **Stateful chunking** — expose per-effect state-carry dicts (the
  same pattern `polyblep` / `filtered_stack` already use for
  `voice_state`). Harder, correct. This is probably the right
  long-term fix and enables streaming effects project-wide.

Plan: start with chunk-with-overlap-and-fade at 2048-sample chunks
(~46 ms at 44.1 kHz — matches a generous DAW buffer size), add
stateful chunking as a second-pass improvement.

Bricasti is the only effect with hard external deps (Fusion-IR
files at `/Users/flatljan/personal/LiquidSonics Bricasti M7 Fusion-IR Sources/`).
Ship a bundle of IRs with the plugin or make the IR directory
configurable.

## Phased scope

### Phase 0 — param schema + block-mode `render()` (pure Python, no C++)

Unblocks everything else. Can and should happen first, inside this repo.

1. `code_musics/engines/_param_schema.py` with typed entries for every
   `synth_voice` + `drum_voice` param.
2. `code_musics/engines/synth_voice.py` and `drum_voice.py` grow a
   `render_block(state, freq, amp, params, n_samples, sample_rate, note_off: bool)`
   entrypoint that returns `(audio_chunk, new_state, done)`. The
   existing `render()` becomes a thin wrapper that calls
   `render_block` repeatedly.
3. Strip peak-normalization out of the block path; normalize at the
   voice mix stage instead (already present in `Score`).
4. Strip attack/release tail bake; only apply release when `note_off`
   is true.
5. Tests proving `render_block` (called in chunks) is bit-identical to
   `render()` on full notes for the non-stochastic presets and
   numerically close for stochastic ones.

This is ~2 weeks of careful work and **delivers value to the existing
Python-only project regardless of whether phases 1+ happen**: streaming
renders, lower peak memory, and stateful effect chunking all drop out
of it.

### Phase 1 — Python worker process + IPC protocol

Python-side only. Still no C++.

1. `code_musics/vst_worker/` module: worker entrypoint, shmem ring
   reader/writer, request/response dataclasses, graceful-shutdown.
2. A Python-side "fake host" harness that drives the worker with
   synthetic MIDI clips; proves the worker renders at expected
   throughput (~5–20× realtime for `synth_voice` post-numba-warmup
   per existing unofficial benchmarks).
3. Decide shmem layout, queue sizes, record format. Freeze v1.

### Phase 2 — CentaurVoice JUCE plugin skeleton

C++ starts here. Target VST3 + AU on macOS, VST3 on Windows.

1. Standard JUCE plugin scaffold (`CMakeLists.txt`, `Source/PluginProcessor.*`,
   `Source/PluginEditor.*`) using JUCE from git submodule.
2. Worker-process spawn on plugin construction, connection via the
   Phase 1 protocol.
3. Audio thread: MIDI input → note scheduler → shmem request → ring
   buffer mixer → output.
4. Minimal UI (no WebView yet): preset dropdown, 8 macro knobs, CPU
   meter, engine-status banner.
5. Passes `pluginval` strictness level 5.

### Phase 3 — CentaurFX JUCE plugin skeleton

Parallel to phase 2 but simpler — no note scheduling, just buffered
audio through.

1. Plugin scaffold.
2. Worker process reused (same Python runtime, different entrypoint).
3. Chunk-with-overlap-and-fade dispatch for the four effect families.
4. Param surface: one "effect slot" dropdown + per-effect params.
   Multi-effect chaining deferred to later.

### Phase 4 — CHOC WebView UI for both plugins

1. Ported megamegaseq's bridge/UI pattern: `ui/src/` + `tools/embed_ui.py`.
2. Preset browser (same JSON format as Phase 0).
3. Macro knobs, deep-param reveal, visual feedback (scope on output,
   GR meter on compressor, etc.).
4. Mock + WS + real bridge modes; Playwright test layer.

### Phase 5 — packaging

- macOS: codesigning + notarization.
- Windows: installer.
- Bundle the Python runtime (uv-managed, pinned). Plugin ships with
  its own isolated interpreter and numpy/numba/scipy wheels.
- Install-time warmup script that pre-compiles numba kernels so the
  first-use latency spike doesn't hit the user in Ableton.
- IR bundle for Bricasti.

## Key risks and mitigations

1. **Numba cold-start is seconds.** Mitigation: install-time warmup
   that runs a synthetic preset through every engine once, caches
   to `~/Library/Application Support/CentaurVoice/numba_cache/`.
2. **Python worker crash = silent plugin.** Mitigation: heartbeat
   watchdog + auto-respawn + red "engine offline" banner. Plugin
   never blocks the audio thread waiting for the worker.
3. **Live-play latency on long notes.** Document as a known
   limitation. Drums are fine (short notes). Pads need transport +
   lookahead. No workaround without rewriting the engine.
4. **Bundle size.** A Python runtime + numpy + numba + scipy is
   ~150–300 MB. Accept; it's the cost of not rewriting in C++.
5. **MIDI note-off timing in live-play mode.** Engines don't
   support "extend note" mid-render. Phase 0's `render_block` with
   a `note_off` flag fixes this.
6. **Ableton-specific quirks.** Ableton's MIDI clip engine and
   PDC (plugin delay compensation) behavior is worth proving out
   early in Phase 2 before building more.

## Critical files to modify (when implementation starts)

- `code_musics/engines/synth_voice.py:53` — `render()` → `render_block()` refactor.
- `code_musics/engines/drum_voice.py:42` — same.
- `code_musics/engines/registry.py` — parameter schema hook.
- `code_musics/engines/_param_schema.py` — new.
- `code_musics/vst_worker/` — new package.
- `code_musics/synth.py` — effect functions grow optional `state` kwarg for
  stateful-chunking mode.
- `CMakeLists.txt`, `Source/`, `ui/src/` — new top-level C++ / JS tree
  (separate directory or sibling repo; probably sibling to keep Python and
  C++ build systems clean).

## Verification plan

Phase-by-phase:

- **Phase 0**: `make all` clean; new tests prove `render_block` == `render`
  bit-exactly for deterministic presets, numerically close for stochastic.
  Render 3 existing pieces through the block API, compare outputs (peak /
  LUFS / null-test RMS).
- **Phase 1**: Python-only harness renders a 60-second synthetic MIDI
  clip through the worker at >5× realtime. Heartbeat kill-and-respawn
  tested.
- **Phase 2**: `pluginval` level 5 passes. Manual test in Ableton Live
  12+: load plugin, play a clip, verify no xruns at 256-sample buffer
  with 10 voices. CPU meter within 30% of expected.
- **Phase 3**: A/B null-test each exposed effect against its direct
  Python call on a static buffer — differences should only be at
  chunk boundaries and < -60 dBFS.
- **Phase 4**: Playwright UI tests pass against mock + WS + real
  bridges. Preset save/recall round-trips bit-exact.
- **Phase 5**: Fresh macOS install in a clean VM loads and runs
  the plugin without user intervention.

## Time estimate

Rough, assuming part-time work:

- Phase 0: 2 weeks
- Phase 1: 1 week
- Phase 2: 3–4 weeks
- Phase 3: 1–2 weeks
- Phase 4: 2 weeks
- Phase 5: 1–2 weeks

Total: **10–13 weeks** to a shippable v1. Phase 0 alone is independently
valuable and should happen regardless.

## Alternatives considered and rejected

- **Render-to-sampler (SFZ/DecentSampler).** Ships in 1–2 weeks but
  loses drift buses, smoothed_random, coupled modal banks, stochastic
  clouds, chaotic oscillators — every reason this project isn't just
  another sampler. Rejected because the loss is the thing that makes
  it interesting.
- **Full C++ rewrite (megamegaseq's path).** Months, doubles
  maintenance, opens DSP correctness regression risk. Rejected per
  user.
- **Neutone SDK.** Torch-only. Would require rewriting numpy DSP as
  torch modules. Rejected.
- **Embedded CPython on the audio thread.** GIL + GC kills
  real-time. Rejected.
