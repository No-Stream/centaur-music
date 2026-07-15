# aphotic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> (inline, main context) for all composition tasks per the repo Agent
> Delegation Policy; the reverb DSP workstream (Task 3) is the only task
> dispatched to subagents. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build *aphotic* — an ~8-minute no-threes-JI cave piece per
`docs/plans/2026-07-07-aphotic-design.md` — plus the two pieces of
infrastructure it needs: a sieved (tri-free) harmonic spectrum builder and an
enormous dark reverb.

**Architecture:** Three phases. Phase A auditions the tuning world and
tri-free patches cheaply before composition (learning from anneal: co-design
must be verified by ear + spectrogram early). Phase B is the delegated reverb
DSP workstream, run in parallel. Phase C composes the piece in the main
context, section by section, with snippet renders and user listening
checkpoints.

**Tech Stack:** existing `Score`/`Voice`/additive-engine stack,
`code_musics.spectra` builders, `code_musics.generative` (TonePool,
prob gates), `drum_voice` kick, native effects + new FDN reverb.

## Global Constraints

- Tuning subgroup **2.7.11.13**; prime 3 forbidden everywhere; prime 5
  reserved for the single illumination event (Section III).
- Working scale: 1/1, 8/7, 13/11, 14/11, 11/8, 13/8, 7/4, 2/1. 13 is on
  probation pending Task 2 audition.
- **Tri-free spectral rule:** every additive/FM voice omits partials with a
  factor of 3; partials with a factor of 5 down-weighted (default ×0.35).
  Verified by A/B render + spectrogram, not assumed.
- Tonic **F#1 ≈ 46.249 Hz** (`f0_hz=46.249`).
- Engines: additive (and FM where it passes the sideband audit) for pitched
  material; subtractive only for non-pitched support (noise/air).
- Aesthetic: crisp/clean/hifi (no lo-fi artifacts), austere (~6 voices),
  alive (tempo map, humanization, automation arcs, pitch motion designed in).
- All verification via `make` targets (never bare `python`); `make all` before
  any commit that touches library code; render commands get ≥180 s timeouts.
- No AI attribution in commit messages.

---

## Phase A — foundations and auditions

### Task 1: `sieved_harmonic_spectrum` in `code_musics/spectra.py`

**Files:**
- Modify: `code_musics/spectra.py` (add function near `harmonic_spectrum`)
- Test: `tests/test_spectra_sieved.py` (new)
- Docs: `docs/synth_api.md` (spectral builders section, one entry)

**Interfaces:**
- Produces: `sieved_harmonic_spectrum(*, n_partials: int, omit_factors:
  tuple[int, ...] = (3,), downweight_factors: dict[int, float] | None = None,
  harmonic_rolloff: float = 0.5, brightness_tilt: float = 0.0) ->
  list[dict[str, float]]` — a `harmonic_spectrum` variant that drops any
  partial index divisible by an `omit_factors` entry and multiplies amps of
  partials divisible by a `downweight_factors` key by its value. Consumed by
  Tasks 2 and 4+.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for sieved_harmonic_spectrum."""

from code_musics.spectra import harmonic_spectrum, sieved_harmonic_spectrum


class TestSievedHarmonicSpectrum:
    def test_omits_multiples_of_three(self) -> None:
        partials = sieved_harmonic_spectrum(n_partials=16)
        ratios = [p["ratio"] for p in partials]
        assert ratios == [1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0, 11.0, 13.0, 14.0, 16.0]

    def test_downweights_fives(self) -> None:
        plain = {
            p["ratio"]: p["amp"]
            for p in sieved_harmonic_spectrum(n_partials=16, harmonic_rolloff=0.8)
        }
        sieved = {
            p["ratio"]: p["amp"]
            for p in sieved_harmonic_spectrum(
                n_partials=16,
                harmonic_rolloff=0.8,
                downweight_factors={5: 0.35},
            )
        }
        assert sieved[5.0] == plain[5.0] * 0.35
        assert sieved[10.0] == plain[10.0] * 0.35
        assert sieved[7.0] == plain[7.0]

    def test_matches_harmonic_spectrum_when_nothing_sieved(self) -> None:
        assert sieved_harmonic_spectrum(
            n_partials=8, omit_factors=(), harmonic_rolloff=0.6
        ) == harmonic_spectrum(n_partials=8, harmonic_rolloff=0.6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `make test-selected TESTS=tests/test_spectra_sieved.py`
Expected: FAIL with ImportError (`sieved_harmonic_spectrum` not defined).

- [ ] **Step 3: Implement**

```python
def sieved_harmonic_spectrum(
    *,
    n_partials: int,
    omit_factors: tuple[int, ...] = (3,),
    downweight_factors: dict[int, float] | None = None,
    harmonic_rolloff: float = 0.5,
    brightness_tilt: float = 0.0,
) -> list[dict[str, float]]:
    """Harmonic spectrum with sieved (omitted) and down-weighted partial indices.

    Built for scale/timbre co-design in restricted JI subgroups: e.g. a
    no-threes (2.5.7.11) piece uses ``omit_factors=(3,)`` so no voice carries
    an acoustic twelfth/fifth, and ``downweight_factors={5: 0.35}`` to ration
    major-third color.
    """
    for factor in omit_factors:
        if factor < 2:
            raise ValueError("omit_factors entries must be >= 2")
    resolved_downweights = downweight_factors or {}
    for factor, weight in resolved_downweights.items():
        if factor < 2 or not 0.0 <= weight <= 1.0:
            raise ValueError("downweight_factors must map factor>=2 to weight in [0,1]")

    spectrum: list[dict[str, float]] = []
    for partial in harmonic_spectrum(
        n_partials=n_partials,
        harmonic_rolloff=harmonic_rolloff,
        brightness_tilt=brightness_tilt,
    ):
        index = int(partial["ratio"])
        if any(index % factor == 0 for factor in omit_factors):
            continue
        amp = partial["amp"]
        for factor, weight in resolved_downweights.items():
            if index % factor == 0:
                amp *= weight
        spectrum.append({"ratio": partial["ratio"], "amp": amp})
    return spectrum
```

- [ ] **Step 4: Run tests, typecheck, docs**

Run: `make test-selected TESTS=tests/test_spectra_sieved.py` → PASS.
Add a short entry to `docs/synth_api.md` next to `harmonic_spectrum`
documenting the signature and the no-threes co-design use case.
Run `make all COV=0` → green.

- [ ] **Step 5: Commit**

```bash
git add code_musics/spectra.py tests/test_spectra_sieved.py docs/synth_api.md
git commit -m "feat(spectra): sieved_harmonic_spectrum for tri-free co-design"
```

### Task 2: `aphotic_audition` study piece + user listening checkpoint

**Files:**
- Create: `code_musics/pieces/aphotic_audition.py`
- Modify: `code_musics/pieces/registry.py` (register, following the pattern of
  other study pieces), `code_musics/pieces/__init__.py` if imports are listed
  there
- Test: covered automatically by `tests/test_pieces_smoke.py`
  (`test_score_backed_piece_builds_valid_score` parametrizes over the registry)

**Interfaces:**
- Consumes: `sieved_harmonic_spectrum` (Task 1).
- Produces: the piece module exports `APHOTIC_F0_HZ = 46.249`,
  `APHOTIC_DEGREES` and `APHOTIC_LABELS` constants that Task 4 imports:

```python
APHOTIC_F0_HZ = 46.249
APHOTIC_DEGREES: tuple[float, ...] = (
    1.0, 8 / 7, 13 / 11, 14 / 11, 11 / 8, 13 / 8, 7 / 4,
)
APHOTIC_LABELS: tuple[str, ...] = (
    "root", "wide_second", "dark_third", "glass_third",
    "floater", "dusk_sixth", "seventh",
)
```

- [ ] **Step 1: Build the audition score.** ~2.5–3 minutes, clearly
  sectioned with labeled phrases (viz-style structured labels optional), at a
  comfortable audition register (roots around F#2–F#3, not the sub tonic):
  1. Scale walk — each degree held over a root drone, tri-free additive patch.
  2. Otonal core chords — 4:7:11, then 4:7:11:13, then 7:8:11:13 voicing,
     each held ~8 s.
  3. Utonal mirror — /7 and /11 dyads under a guide tone.
  4. **13 probation A/B** — same passage twice: with 13-degrees (13/11,
     13/8, :13 chord extensions) and with them removed.
  5. **Tri-free A/B** — the same short chord passage on (a) plain
     `harmonic_spectrum(n_partials=12)` and (b)
     `sieved_harmonic_spectrum(n_partials=16, downweight_factors={5: 0.35})`.
  6. **Illumination preview** — the 4:7:11 chord, then the same chord with
     5/4 blooming in (a separate voice fading up and away).
  Use one shared `bricasti_or_reverb` send bus so chords are heard in space.
- [ ] **Step 2: Smoke-verify and render.**
  Run: `make all COV=0` → green (registry smoke test picks the piece up).
  Run: `make render PIECE=aphotic_audition` (timeout ≥ 600 s; capture full
  output including analysis warnings). Check the chromagram artifact: energy
  should sit on scale degrees only; the tri-free A/B segments should visibly
  differ at 3·f partials in the spectrogram.
- [ ] **Step 3: Commit**

```bash
git add code_musics/pieces/aphotic_audition.py code_musics/pieces/registry.py code_musics/pieces/__init__.py
git commit -m "feat(pieces): aphotic_audition — no-threes scale/chord/patch audition study"
```

- [ ] **Step 4: USER LISTENING CHECKPOINT.** Decisions to lock before
  Phase C: (a) is 13 in or out, (b) does the tri-free rule audibly earn its
  keep, (c) which chord voicings lock best, (d) does the 5/4 bloom land.
  Iterate the audition piece with snippet renders until the user signs off.

## Phase B — reverb DSP workstream (delegated, runs parallel to Phase A/C)

### Task 3: `fdn_reverb` native effect + stacked-space audition

**Files:**
- Create: `code_musics/engines/_fdn_reverb.py` (or a location the subagent
  argues better; DSP core)
- Modify: `code_musics/synth.py` (apply function + `EFFECT_APPLIERS`
  registration as `"fdn_reverb"`)
- Test: `tests/test_fdn_reverb.py`
- Docs: `docs/synth_api.md` (full parameter surface), `AGENTS.md` (one-line
  callout)

**Interfaces:**
- Produces: `EffectSpec("fdn_reverb", {...})` usable on voices, send buses,
  and master. Parameter contract (subagent may extend, not shrink):
  - `decay_s` (float, RT60 target, musical up to ≥ 45 s)
  - `size` (0–1 perceptual room scale), `predelay_ms`
  - `damping_hz` (HF decay corner), `low_decay_mult` (bass tail multiplier)
  - `modulation_depth` (0–1), `modulation_rate_hz` (slow, chorus-free tails)
  - `diffusion` (0–1), `mix` (wet level, automatable), `highpass_hz` /
    `lowpass_hz` wet-return tone shaping
  - deterministic (seeded) — repeated renders bit-identical
- Consumes: nothing from other tasks; independent.

- [ ] **Step 1: Dispatch.** Send to an Opus subagent AND a Codex rescue pass
  (two independent designs or design+review, dispatcher's call) with: the
  parameter contract above, the aesthetic target ("enormous, unworldly, dark,
  clean — a cave beyond architectural scale; tails must stay smooth and
  chorus-free at 30–60 s decays"), repo conventions (make targets, fail-fast,
  TDD, docs-in-same-pass), and the test contract below.
- [ ] **Step 2: Test contract (subagent implements via TDD).**
  - impulse response RT60 within ±25% of `decay_s` at 1 kHz
  - output finite and bounded for 60 s renders at max settings
  - L/R decorrelation: broadband interchannel correlation of the late tail
    < 0.4
  - no DC buildup (tail mean → 0), no denormal stalls
  - determinism: two renders of the same input identical
  - `make all` green including docs formatting
- [ ] **Step 3: Space audition render.** Extend `aphotic_audition` (or a
  scratch window of it) with three space candidates on the same material:
  (a) `fdn_reverb` alone at ~20 s and ~45 s decay, (b) serial stack
  close-dark `bricasti_or_reverb` → `fdn_reverb`, (c) parallel two-depth
  sends (close dark + vast). Render and compare artifacts.
- [ ] **Step 4: USER LISTENING CHECKPOINT.** Pick the space architecture for
  the piece. Commit the effect work per the subagent's TDD cycle.

## Phase C — the piece (main context, creative iteration)

Composition tasks are deliberately iterative: each one is "compose → snippet
render → inspect analysis → adjust → user checkpoint", not one-shot code.
Exact note content is authored in-session; the steps below fix structure,
voices, and verification gates.

### Task 4: `aphotic` skeleton — score, voices, buses, registry

**Files:**
- Create: `code_musics/pieces/aphotic.py`
- Modify: `code_musics/pieces/registry.py`, `code_musics/pieces/__init__.py`
- Test: registry smoke test (automatic); add `"aphotic"` to
  `_RENDER_SMOKE_PIECE_NAMES` in `tests/test_pieces_smoke.py` with a
  `_PIECE_WINDOW_OVERRIDES` entry past the silent intro (e.g. `(120.0, 122.0)`).

**Interfaces:**
- Consumes: `APHOTIC_F0_HZ` / `APHOTIC_DEGREES` / `APHOTIC_LABELS` from
  `aphotic_audition` (import, don't duplicate), `sieved_harmonic_spectrum`,
  the Task-3 space architecture.
- Produces: `build_score() -> Score` with all six voices registered (even if
  sections are sparse initially), a `TempoMap` with section-level breathing,
  the shared send buses, `master_effects=DEFAULT_MASTER_EFFECTS` (or a
  piece-specific dark master chain if auditioning demands), and
  `timing_humanize` configured.

- [ ] **Step 1: Voice plan in code.** Six voices per the spec: floor
  (`drum_voice` kick, tonic-tuned, long sub tail, `normalize_peak_db=-6.0`,
  drum bus per `setup_drum_bus(style="light")` or bespoke); raindrop arp
  (additive/FM pings, TonePool + prob-gated timing, seeded); crystal strikes
  (additive, `sieved_harmonic_spectrum` partials, modal-flavored envelopes);
  bowed crystal (sustained additive swell or bow-family, mostly Section III);
  air (`synth_voice` noise slot `found_empty_room`-style, LUFS-normalized very
  low); ticks (`metallic_perc` or `drum_voice` metallic layer, near
  subliminal). All tonal voices subscribe to a shared drift bus
  (`Score.add_drift_bus`); velocity groups shared across crystal + arp.
- [ ] **Step 2: Verify skeleton.** `make all COV=0` green;
  `make render-window PIECE=aphotic START=0 DUR=30` renders without warnings
  that indicate broken routing.
- [ ] **Step 3: Commit** (`feat(pieces): aphotic skeleton — voices, buses, tempo map`).

### Task 5: Section I — Dark adaptation (0:00–2:00)

- [ ] Compose: air floor from t=0; first raindrops sparse (Poisson-feel via
  probability gates, density automation ramping ~0.05→0.3); sub drone fades
  up so late the listener realizes it was always there (long exp fade from
  -inf landing ~1:10); one or two utonal dyad ghosts.
- [ ] Aliveness gate: rubato-feel tempo (no grid audible), automation on air
  level + arp send, no statically held parameter.
- [ ] Verify: `make render-window PIECE=aphotic START=0 DUR=125` + artifacts;
  user checkpoint; commit.

### Task 6: Section II — Skeleton (2:00–4:00)

- [ ] Compose: kick coheres out of the drips (first kicks placed *as if* they
  were drips, then settling toward — but never onto — a rigid grid); subtle
  BPM drift over the section; arp establishes with TonePool weights favoring
  root/7/4/11-8; crystal strikes begin answering arp events; ticks enter near
  subliminal.
- [ ] Consonance gate: undecimal intervals appear inside otonal chords only;
  any bare exposed 11/8 or 13-limit leap must be intentional (transition
  moments), else revoiced.
- [ ] Verify: snippet renders of 2:00–2:40 and 3:20–4:10 (transition into
  III); check drum-bus and kick sub via analysis; user checkpoint; commit.

### Task 7: Section III — Illumination (4:00–5:30)

- [ ] Compose: beat dissolves (kick thins, then stops — element-dropout
  automation); bowed crystal chord assembles by staggered entries into the
  4:7:11(:13) sonority; the single 5/4 blooms in a dedicated voice or note
  (slow attack, vibrato blooming, long release into the vast send) and
  recedes; drips nearly stop.
- [ ] This is the emotional center: quiet arrival, not climax — check RMS
  envelope does *not* peak here relative to Section II.
- [ ] Verify: `make render-window PIECE=aphotic START=230 DUR=100`; user
  checkpoint; commit.

### Task 8: Section IV — Recession + full-form pass (5:30–8:00)

- [ ] Compose: beat returns sparser (fewer kicks per bar than II), then
  thins; arp density automation ramps back down; final ~45 s is floor + air +
  last drips, ending on the dark, not on a cadence.
- [ ] Full-form pass: render the whole piece (`make render PIECE=aphotic`,
  timeout ≥ 900 s, full output captured); check pacing, section transitions,
  loudness arc, artifact warnings, timing-drift diagnostics.
- [ ] User checkpoint on the full render; commit.

### Task 9: Aliveness + mix pass, then evaluation

- [ ] Sweep every voice against the aliveness checklist from the design spec
  (tempo breathing, velocity shaping + shared groups, envelope humanize,
  pitch motion on every sustained/melodic voice, automation arcs on sends and
  brightness, exp shape on all `_hz` automation).
- [ ] Mix: LUFS-normalized tonal voices balanced via `mix_db`; kick/sub via
  `normalize_peak_db`; verify master true-peak/LUFS from the WAV export log;
  no unexplained artifact-risk warnings (fix or follow the IMD-probe
  fix-or-fix-probe standard).
- [ ] `make all` (full, with coverage) green.
- [ ] `make evaluate PIECE=aphotic MODELS=opus` for a first judge pass; read
  feedback, iterate if it flags real issues; then full `make evaluate
  PIECE=aphotic`.
- [ ] Final commit; update `FUTURE.md` piece-prompt list if aphotic satisfies
  one of its entries (fake-found-sound and utonal/subharmonic form both get
  partial credit — note, don't delete).

---

## Self-review notes

- Spec coverage: tuning world (T2 locks decisions), tri-free rule (T1+T2),
  voices (T4), space (T3), beat+form (T5–T8), aliveness (designed into T4–T8
  gates + T9 sweep), verification methodology (every task has render gates).
- 13-probation and space-architecture decisions are explicit user
  checkpoints before dependent work.
- Types/names consistent: `sieved_harmonic_spectrum`, `APHOTIC_*` constants,
  `EffectSpec("fdn_reverb", ...)` used identically across tasks.
