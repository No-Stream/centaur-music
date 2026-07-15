# anneal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Delegation policy (overrides the default):** Tasks 1–2 are mechanical and delegable to subagents. Tasks 3–5 are *composition* and MUST be executed in the main agent context per repo policy (`AGENTS.md` Agent Delegation Policy). Tasks 3 and 4 end in **user audition gates** — do not proceed past them without explicit user sign-off.

**Goal:** Build `anneal` — a ~6.5 min Four Tet-organic piece on a Colundi scale over G where every tonal voice's spectrum is built from the scale's own degrees, and the middle act stretches scale + spectra together to a pseudo-octave of ~2.07 before annealing home.

**Architecture:** Two small tuning/spectra helpers (pure functions, TDD), then two short *audition sketch* pieces (fusion proof, then palette proof) registered as studies, then the full piece composed iteratively in the main context against the design spec `docs/plans/2026-07-05-anneal-design.md`.

**Tech Stack:** existing `Score`/`Voice` model, `additive` + `synth_voice` + `drum_voice` engines, `found_sound` presets, `make` targets only.

## Global Constraints

- Never run bare `python`; use `make` targets (`make test-selected`, `make render`, `make all`, `make scratch`).
- `f0_hz = 98.0` (G2); kick fundamental 49 Hz.
- Colundi core scale degrees: `1/1, 11/10, 19/16, 4/3, 3/2, 49/30, 7/4` (octave-equivalent).
- Stretch law: `stretch(r, P) = r ** log2(P)`; peak `P ≈ 2.07`.
- Fused skeleton spectrum: integer partials `{1, 2, 3, 4, 6, 7, 8, 12, 14, 16}` — **never a 5th/10th/15th/20th harmonic** in any tonal voice's partial list.
- Color partials: 19-family `{2.375, 4.75}`, 49-family `{49/15≈3.267, 98/15≈6.533}`, 11-family `{2.2, 4.4, 8.8}` (11-family lives in exactly one voice).
- No `voice_dist` / heavy saturation on fused tonal voices (integer-harmonic regeneration would reintroduce the 5th); warmth via multiband preamp or subtle drive only.
- Additive partial format: `list[dict]` with keys `"ratio"`, `"amp"` (optional `"envelope"`, `"phase"`); `synth_voice` explicit partials param is `partials_partials`; `drum_voice` custom modes via `modal_ratios` (+ optional `modal_amps`, `modal_decays_s`).
- Docs updated in the same pass as any public-surface change.
- `make all` green before every commit that touches code.

---

### Task 1: Tuning helpers — `stretch_ratio` + `colundi_core` (delegable)

**Files:**
- Modify: `code_musics/tuning.py` (append after `eikosany_tetrads`)
- Test: `tests/test_tuning.py` (append new test classes)

**Interfaces:**
- Produces: `stretch_ratio(ratio: float, pseudo_octave: float) -> float` and `colundi_core() -> list[float]` in `code_musics.tuning`, used by Tasks 2–5.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_tuning.py`, matching its pytest style; import the two new names at the top of the file):

```python
class TestStretchRatio:
    def test_identity_at_true_octave(self) -> None:
        assert stretch_ratio(3 / 2, 2.0) == pytest.approx(1.5)
        assert stretch_ratio(7 / 4, 2.0) == pytest.approx(1.75)

    def test_octave_maps_to_pseudo_octave(self) -> None:
        assert stretch_ratio(2.0, 2.07) == pytest.approx(2.07)

    def test_fifth_widens_at_2_07(self) -> None:
        cents = ratio_to_cents(stretch_ratio(3 / 2, 2.07))
        assert cents == pytest.approx(701.955 * math.log2(2.07), abs=0.01)
        assert 735.0 < cents < 739.0

    def test_preserves_multiplicative_geometry(self) -> None:
        p = 2.07
        assert stretch_ratio(3 / 2 * 7 / 4, p) == pytest.approx(
            stretch_ratio(3 / 2, p) * stretch_ratio(7 / 4, p)
        )

    def test_rejects_bad_args(self) -> None:
        with pytest.raises(ValueError, match="ratio"):
            stretch_ratio(0.0, 2.07)
        with pytest.raises(ValueError, match="pseudo_octave"):
            stretch_ratio(1.5, 1.0)


class TestColundiCore:
    def test_degrees(self) -> None:
        expected = [1.0, 11 / 10, 19 / 16, 4 / 3, 3 / 2, 49 / 30, 7 / 4]
        result = colundi_core()
        assert len(result) == 7
        for actual, exp in zip(result, expected, strict=True):
            assert actual == pytest.approx(exp)

    def test_sorted_and_within_octave(self) -> None:
        result = colundi_core()
        assert result == sorted(result)
        assert all(1.0 <= r < 2.0 for r in result)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `make test-selected TESTS=tests/test_tuning.py`
Expected: FAIL — `ImportError: cannot import name 'stretch_ratio'`.

- [ ] **Step 3: Implement** (append to `code_musics/tuning.py`; module already imports `math`):

```python
def stretch_ratio(ratio: float, pseudo_octave: float) -> float:
    """Map a JI ratio into a stretched pitch space.

    Pure exponent scaling in log-pitch space: ``r ** log2(P)``. The true
    octave 2/1 maps to the pseudo-octave ``P`` and all interval *geometry*
    (products of intervals) is preserved, so a scale and a spectrum
    stretched by the same ``P`` remain mutually consonant (Sethares).
    """
    if ratio <= 0.0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    if pseudo_octave <= 1.0:
        raise ValueError(f"pseudo_octave must be > 1.0, got {pseudo_octave}")
    return float(ratio ** math.log2(pseudo_octave))


def colundi_core() -> list[float]:
    """Approximate Colundi-inspired 7-note 11-limit JI scale (octave-equivalent)."""
    return sorted([1.0, 11 / 10, 19 / 16, 4 / 3, 3 / 2, 49 / 30, 7 / 4])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `make test-selected TESTS=tests/test_tuning.py` — Expected: PASS.

- [ ] **Step 5: Docs + gate + commit**

Add both helpers to the tuning bullet in `AGENTS.md` (one line) — they are public surface. Run `make all`.

```bash
git add code_musics/tuning.py tests/test_tuning.py AGENTS.md
git commit -m "feat(tuning): stretch_ratio + colundi_core helpers"
```

---

### Task 2: `scale_fused_spectrum` builder in spectra.py (delegable)

**Files:**
- Modify: `code_musics/spectra.py` (append near `ratio_spectrum`)
- Test: `tests/test_spectra.py` (append)
- Modify: `docs/synth_api.md` (spectra builders section, one entry)

**Interfaces:**
- Consumes: nothing from Task 1 (degrees passed in already-stretched by callers).
- Produces: `scale_fused_spectrum(degrees: Sequence[float], *, octaves: int = 3, rolloff_alpha: float = 1.0, amp_floor: float = 0.02) -> list[dict[str, float]]` — same `{"ratio","amp"}` dict-list format as every other builder, directly usable as additive `partials` / synth_voice `partials_partials`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_spectra.py`, matching existing builder-test style):

```python
class TestScaleFusedSpectrum:
    def test_skeleton_yields_integer_partials_without_fifth_harmonic(self) -> None:
        # 3/7-limit skeleton degrees across octaves -> {1,2,3,4,6,7,8,12,14}
        partials = scale_fused_spectrum([1.0, 3 / 2, 7 / 4], octaves=3)
        ratios = sorted(p["ratio"] for p in partials)
        assert ratios == pytest.approx([1.0, 2.0, 3.0, 3.5, 4.0, 6.0, 7.0, 8.0, 12.0, 14.0])
        assert not any(abs(r - 5.0) < 1e-9 or abs(r - 10.0) < 1e-9 for r in ratios)

    def test_color_degrees_yield_noninteger_partials(self) -> None:
        partials = scale_fused_spectrum([1.0, 19 / 16], octaves=2)
        ratios = sorted(p["ratio"] for p in partials)
        assert any(abs(r - 2.375) < 1e-9 for r in ratios)

    def test_rolloff_and_format(self) -> None:
        partials = scale_fused_spectrum([1.0, 3 / 2], octaves=2, rolloff_alpha=1.0)
        assert all(set(p) == {"ratio", "amp"} for p in partials)
        by_ratio = {p["ratio"]: p["amp"] for p in partials}
        assert by_ratio[1.0] == pytest.approx(1.0)
        assert by_ratio[3.0] == pytest.approx(1.0 / 3.0)

    def test_amp_floor_drops_weak_partials(self) -> None:
        partials = scale_fused_spectrum([1.0], octaves=8, rolloff_alpha=2.0, amp_floor=0.02)
        assert all(p["amp"] >= 0.02 for p in partials)

    def test_rejects_empty_or_nonpositive(self) -> None:
        with pytest.raises(ValueError, match="degrees"):
            scale_fused_spectrum([])
        with pytest.raises(ValueError, match="positive"):
            scale_fused_spectrum([0.0])
```

- [ ] **Step 2: Run to verify failure** — `make test-selected TESTS=tests/test_spectra.py` → ImportError.

- [ ] **Step 3: Implement** (append to `code_musics/spectra.py`):

```python
def scale_fused_spectrum(
    degrees: Sequence[float],
    *,
    octaves: int = 3,
    rolloff_alpha: float = 1.0,
    amp_floor: float = 0.02,
) -> list[dict[str, float]]:
    """Build a partial set from a scale's own degrees transposed by octaves.

    The Sethares co-design move: a spectrum whose partials are scale members
    makes the scale's intervals maximally consonant by construction. Partial
    ratios are ``degree * 2**k`` for ``k`` in ``0..octaves``, deduplicated
    and sorted; amps roll off as ``ratio ** -rolloff_alpha`` and partials
    below ``amp_floor`` are dropped. Same ``{"ratio", "amp"}`` format as the
    other builders (additive ``partials`` / synth_voice ``partials_partials``).
    """
    if not degrees:
        raise ValueError("degrees must be non-empty")
    if any(d <= 0.0 for d in degrees):
        raise ValueError("degrees must be strictly positive")
    ratios: set[float] = set()
    for degree in degrees:
        # Only the root degree contributes its k=0 value; other degrees enter
        # from the octave up (no sub-octave partials smearing the fundamental
        # region — the canonical fused skeleton is {1, 2, 3, 4, 6, 7, ...}).
        k_start = 0 if math.isclose(float(degree), 1.0) else 1
        for k in range(k_start, octaves + 1):
            ratios.add(round(float(degree) * (2.0**k), 9))
    partials = [
        {"ratio": ratio, "amp": ratio**-rolloff_alpha}
        for ratio in sorted(ratios)
    ]
    return [p for p in partials if p["amp"] >= amp_floor]
```

- [ ] **Step 4: Verify pass** — `make test-selected TESTS=tests/test_spectra.py`.

- [ ] **Step 5: Docs + gate + commit** — add one builder entry to the spectra section of `docs/synth_api.md` (usage: pass skeleton degrees `[1, 3/2, 7/4, 4/3]` for the workhorse fused spectrum; append color-degree partials for section-dependent flavor). Run `make all`.

```bash
git add code_musics/spectra.py tests/test_spectra.py docs/synth_api.md
git commit -m "feat(spectra): scale_fused_spectrum co-design builder"
```

---

### Task 3: Fusion sketch — `anneal_fusion_sketch` (MAIN CONTEXT + AUDITION GATE)

**Files:**
- Create: `code_musics/pieces/anneal.py`
- Modify: `code_musics/pieces/__init__.py` (import alias + `merge_piece_maps` entry)

**Interfaces:**
- Consumes: `stretch_ratio`, `colundi_core` (Task 1); `scale_fused_spectrum` (Task 2).
- Produces: module-level `PIECES` dict in `code_musics/pieces/anneal.py`; module constants `F0 = 98.0`, `SKELETON_DEGREES`, chord-spelling constants reused by Tasks 4–5.

**Content:** a ~48 s study (registered `study=True`) with one additive pad voice playing a fixed A/B ladder — each chord held ~5 s with a gap:

1. 4:6:7 tonic (`1/1, 3/2, 7/4`), skeleton spectrum, S=2.0
2. 16:19:24 (`1/1, 19/16, 3/2`), skeleton only (no 19-partials) — expect dark but less fused
3. 16:19:24 with 19-family color partials added — expect audible fusion vs. (2)
4. Neutral subdominant (`4/3, 49/30, 2/1`) with 49-family partials
5–8. The same four chords with **both** note frequencies and partial lists mapped through `stretch_ratio(·, 2.07)`

**Steps:**

- [ ] Build the module skeleton: `PieceDefinition(name="anneal_fusion_sketch", output_name="anneal_fusion_sketch", build_score=..., study=True)`; register in `pieces/__init__.py`. Verify with `make list`.
- [ ] Author the sketch score: one `additive` voice, `phase_disperse` morph (`spectral_morph_type="phase_disperse"`, modest `spectral_morph_amount`), per-note `synth={"partials": ...}` overrides carrying the four spectrum variants; notes placed via `freq=F0 * stretched_degree` so tuning and spectrum use the same stretch value. Soft reverb send (`bricasti_or_reverb`), `DEFAULT_MASTER_EFFECTS`.
- [ ] Smoke: `make test-selected TESTS=tests/test_pieces_smoke.py` (auto-collects the new score-backed piece).
- [ ] Render full sketch: `make render PIECE=anneal_fusion_sketch` (timeout ≥ 240 s; capture full output). Inspect the chromagram + spectrogram analysis: fusion should appear as partial alignment (fewer independent chroma smears) in variants 3/4 vs 2; stretched variants should show uniformly displaced partials, not smeared ones. Check artifact-risk warnings.
- [ ] `make all`, then commit: `git commit -m "feat(pieces): anneal_fusion_sketch co-design audition study"`.
- [ ] **AUDITION GATE 1:** user listens; confirm (a) fused chords sound fused, (b) the 19/49 color-partial variants read as intended, (c) S=2.07 sounds "warped world," not "out of tune." Do not start Task 4 without sign-off. Fold any verdicts (rolloff, morph amount, stretch depth) back into the module constants.

---

### Task 4: Palette sketch — `anneal_palette_sketch` (MAIN CONTEXT + AUDITION GATE)

**Files:**
- Modify: `code_musics/pieces/anneal.py` (second `PieceDefinition` in the same `PIECES` dict, `study=True`)

**Interfaces:**
- Consumes: Task 3's constants and spectrum helpers.
- Produces: the full voice-construction functions (`_add_bass`, `_add_bell_arp`, `_add_pad`, `_add_drums`, `_add_atmosphere`, plus the hall `SendBusSpec` and drum bus) that Task 5 reuses verbatim.

**Content:** ~60 s, 8–16 bars at 110 BPM, home tuning only (S=2.0), all roles sounding together over a simple I–IV–V–I pass of the harmonic skeleton:

- **Bass:** `synth_voice` — sub osc + `partials_partials` `{1, 2, 3}` (+ faint 7), ladder filter, `max_polyphony=1`, `legato=True`, kick-duck sidechain compressor (ninth_wave recipe: `threshold_db=-21.0, ratio=4.0, attack_ms=2.0, release_ms=140.0, lookahead_ms=5.0, sidechain_source="kick"`), notes at octave −1 (≈49 Hz region).
- **Bell arp:** `additive` — skeleton spectrum with per-partial `"envelope"` decays (higher partials decay faster: e.g. partial ratio r gets keyframes `[{"time": 0.0, "value": 1.0}, {"time": 0.9 / r**0.5, "value": 0.1}]`), light `phase_disperse`, velocity phrasing, vibrato `PitchMotionSpec`, `Groove.sixteenths_swing()`-shifted 16ths.
- **Pad:** `additive` — skeleton spectrum, slow attack, `ratio_glide` between changes, mild `spectral_gravity` with `gravity_targets` set to the Colundi degrees (not the 5-limit defaults!), band-split kick sidechain like ninth_wave's pad.
- **Drums:** `setup_drum_bus(style="electronic")`; kick via `add_drum_voice(engine="drum_voice", preset="909_techno", ...)` with notes at `freq=49.0`; hats/toms as `drum_voice` with `metallic_type="modal_bank"`, `modal_mode_table="custom"` semantics via `modal_ratios=[1.0, 2.375, 3.267, 4.9, 6.533]` (scale-degree modes).
- **Atmosphere:** `synth_voice` noise slot, `preset="found_empty_room"` + a second quiet `found_city_at_night` voice, low `mix_db`, no stretch ever.
- Score: `f0_hz=98.0`, `timing_humanize=TimingHumanizeSpec(preset="tight_ensemble")`, `add_drift_bus("kiln_drift", rate_hz=0.06, depth_cents=3.5, seed=...)` with tonal voices subscribed, hall send bus, `DEFAULT_MASTER_EFFECTS`.

**Steps:**

- [ ] Author voice-construction functions + the sketch's `build_score`; register `anneal_palette_sketch`.
- [ ] Smoke test (`make test-selected TESTS=tests/test_pieces_smoke.py`), then `make render PIECE=anneal_palette_sketch` (timeout ≥ 300 s); read loudness/artifact output and the mel/chromagram analysis; iterate mix (`mix_db`, sends) until clean.
- [ ] `make all`, commit: `git commit -m "feat(pieces): anneal_palette_sketch full-palette audition study"`.
- [ ] **AUDITION GATE 2:** user listens; sign off on each voice's character and the mix balance before any form-building. Fold verdicts into the voice functions.

---

### Task 5: The full piece — `anneal` (MAIN CONTEXT, iterative)

**Files:**
- Modify: `code_musics/pieces/anneal.py` (main `PieceDefinition(name="anneal", study=False, sections=(...))`)
- Modify: `tests/test_pieces_smoke.py` (add `"anneal"` to `_RENDER_SMOKE_PIECE_NAMES`)

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: the finished piece; `PieceSection` markers for the three acts.

**Fixed scaffolding (authored first):**

- [ ] The master stretch curve — single source of truth, drums led by 8 s:

```python
def pseudo_octave_at(t: float) -> float:
    """S(t): 2.0 through Act I, smoothstep up to 2.07 across Act II,
    slower ease back home across Act III (anneal = slow cool)."""
    peak = 2.07
    if t < 120.0:
        return 2.0
    if t < 250.0:
        x = (t - 120.0) / 130.0
        return 2.0 + (peak - 2.0) * (3 * x**2 - 2 * x**3)
    if t < 270.0:
        return peak
    if t < 390.0:
        x = (t - 270.0) / 120.0
        return peak - (peak - 2.0) * x**2  # ease-in: slow start to the cooling
    return 2.0


def drum_pseudo_octave_at(t: float) -> float:
    return pseudo_octave_at(t + 8.0)  # percussion warps first
```

Every note-placement helper computes `p = pseudo_octave_at(onset)` and derives BOTH `freq = F0 * stretch_ratio(degree, p) * stretch_ratio(2.0, p) ** octave_shift` (note octave shifts must stretch too — a pseudo-octave, not 2.0; `stretch_ratio(r, 2.0)` is already the identity at home tuning) AND the note's partial list mapped through the same `p`.

- [ ] Harmonic spine constants: root cycle `1/1 → 4/3 → 3/2 → 1/1` with `7/4` turnaround; Act II dark turn through 16:19:24 territory and the neutral subdominant; exactly one 11/10 spice chord at the Act II climax (~4:10); 11-family partials only in the Act II counter-voice.

**Composition loop (creative, iterative — the repo's normal workflow):**

- [ ] Act I (0:00–2:00): found-sound room fade-in → pad breath → arp motif statement → beat assembly. Iterate with `make render-window PIECE=anneal START=0 DUR=30` style windows + analysis after each meaningful change.
- [ ] Act II (2:00–4:30): stretch ramp; darkening harmony; 11-carrier counter-voice entry; full kit; climax spice chord at max stretch. Iterate with windows around 2:00 (warp onset legibility) and 4:00–4:30 (climax).
- [ ] Act III (4:30–6:30): slow cool; beat thinning; motif return at home tuning; final maximum-fusion 4:6:7; found-sound outro.
- [ ] Automation passes per the spec + AGENTS.md automation ideas: filter rides, send rides, dropout bars, transition swells, pan motion on hats.
- [ ] Full render `make render PIECE=anneal` (timeout ≥ 900 s); read complete output — loudness, artifact-risk warnings, timing-drift diagnostics, analysis manifest. Fix anything flagged.
- [ ] 11-limit audit: grep the module for `11 / 10` and `49 / 30` uses; confirm each matches its designated job from the spec.
- [ ] Add `"anneal"` to `_RENDER_SMOKE_PIECE_NAMES`; `make all`; commit `feat(pieces): anneal — colundi stretch-drift piece`.
- [ ] User audition of the full piece; iterate on feedback; optionally `make evaluate PIECE=anneal`.

**Risk playbook (from the spec):** stretch reads as "out of tune" → stretch drums earlier / lean on fused pads / re-check that note freq and partials share the same S sample. Render time balloons from per-note partial lists → quantize S(t) per bar. Fusion inaudible → raise `octaves` in the fused spectrum, reduce `rolloff_alpha`, revisit `phase_disperse` amount.

---

## Revision pass — 2026-07-07 (audition 4 feedback)

User audition of the full render surfaced these; all applied in `anneal.py`:

- **Stretch cut ~75%**: `PIECE_PEAK_PSEUDO_OCTAVE = 2.0175` (sketch keeps 2.07
  as a demo). At 2.07 the warp read as "the pitch is changing" rather than
  "subtly alien". S(t) is now two-staged: slow simmer to 40% of span by bar 88
  (the dark turn stays barely-warped), faster push to full span at the bar-114
  climax. Drum stretch lead reduced 4 → 2 bars — with the smaller span, a big
  lead just read as drums out of tune with the tonal voices.
- **2:11 dissonance**: the arp was playing 16:19:24 (and later neutral-triad)
  tones over the NEUTRAL_SUBDOMINANT pad. Audition-2 rule now applied in the
  full piece everywhere: neutral pad ⇒ arp on FOURTH_OPEN tones (bars 60, 80,
  92).
- **"Steady clap" percussion**: the every-odd-bar open hat at step 14 and the
  every-4th-bar tom double read as a metronomic clap. Both are now
  intermittent on an 8-bar variation cycle with varied steps/velocities, and
  both send into the echo bus (dub throws instead of dry backbeat).
- **Arp repetition by 1:30**: two new cycles — `_ARP_CYCLE_A2` (contour
  variation, mixed note lengths, 32nd-note pickup, second-bar breath) and
  `_ARP_CYCLE_HALF` (ringing half-time gear). Patterns now rotate every
  section: half-time intro, A2 alternation from bar 24, theme A trades bars
  main/A2, afterglow and ending drop to half-time. `_place_arp` accepts float
  steps for sub-16th placement.
- **Noticeable arp delay**: echo bus upgraded `delay` → `mod_delay` (dotted
  8th, feedback 0.45, dark 3.2 kHz feedback LPF, slow stereo wander) and the
  send ride now starts at bar 8 (−16 dB), peaks −7 dB at the climax.
- **More patch automation**: arp gained a sample-accurate `analog_filter` SVF
  LP with a piece-length cutoff ride (1.7 kHz veiled entrance → 6.8 kHz climax
  → 1.9 kHz close) plus a `release` ride (0.28 tight ↔ 0.55–0.7 washy in the
  dark turn / cooling / ending).

## Revision pass — 2026-07-07 (audition 5 feedback)

- **"Clap" identified as the hats**: too long a decay, nowhere near enough
  treble to read "hat" at freq=784. Fixed at the source — the `drum_voice`
  `closed_hat` / `open_hat` presets get clickier/brighter/shorter (global
  preset improvement, delegated).
- **2:11 still dissonant**: it was the neutral triad itself (49/40 ≈ 351-cent
  thirds), not just the arp over it. Bar 60 pad now takes FOURTH_OPEN; the
  neutral color survives only inside denser textures (bars 80, 92).
- **Arp louder**: mix_db −8 → −5.5 (delay send had eaten its prominence).
- **Stretch stays (bending is the point) but mellowed**: all partial builders
  now tilt high partials down progressively with stretch depth
  (`_STRETCH_TILT = 0.35`, ~−8 dB on the 14th partial at full stretch — high
  partials deviate most in Hz and carry most of the roughness), and pad
  color-partial weight fades up to 50% at full stretch. The world gets
  darker/softer-edged as it melts instead of rougher.
- **Patch automation everywhere**: pad `spectral_morph_amount` 0.3→0.6→0.2
  arc + attack ride (slow blooms in Act III), bass `resonance_q` ride,
  11-carrier morph ride, arp attack ride (hard climax bells, soft cooling
  bells), plus `_WASH_BELLS` (phase-disperse) per-note patch for all
  half-time arp sections. No new plumbing needed: voice-level synth
  automation applies note-onset scalars to any engine param.
- **Analysis tier retune** (delegated): continuous compressor GR >1 dB →
  warning; >2 dB sustained → severe.

## Revision pass — 2026-07-07 (audition 6 feedback)

- Stretch halved again: peak S = 2.009 (+7.8 ¢/octave; +15.5 ¢ two octaves
  up; kick −7.8 ¢). The liked "sea melody" dark turn stays at 0–2 ¢.
- Piece tightened 176 → 142 bars (~5:14): Act I kick-entrance block cut,
  Act II bridge halved, Act III cooling 28 → 12 bars. ABA form unchanged,
  just legible now.
- drum_voice hats crisped further (closed −20 dB ≈ 35 ms; open ≈ 300 ms,
  noise mix rebalanced for the voicing test bound).
- Arp `decay` added to the automated patch surface (tight at climax,
  singing in the dark turn and cooling).
