# Rhythm Studies Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Three study pieces (study_groove, study_aksak, study_rhythm_transforms) showcasing new rhythm/groove features.

**Architecture:** Each study is a single Python module in `code_musics/pieces/studies/`, following the existing study pattern: module-level constants, synth specs, a `build_score()` function, and a `PIECES` dict. All registered through `studies/__init__.py`.

**Tech Stack:** Python, code_musics score API, new rhythm tools (Groove, prob_rhythm, AksakPattern, ca_rhythm, mutate_rhythm, augment/diminish/rotate/displace/rhythmic_retrograde, polyrhythm, cross_rhythm).

---

### Task 1: study_groove — "Same Beat, Different Feel"

**Files:**

- Create: `code_musics/pieces/studies/study_groove.py`
- Modify: `code_musics/pieces/studies/__init__.py`
- Test: `make render PIECE=study_groove PLOT=0 ANALYSIS=0` + `make all`

**Design reference:** `docs/plans/2026-04-14-rhythm-studies-design.md` section 1.

**Pattern to follow:** `code_musics/pieces/studies/study_euclidean.py` for imports, voice setup, Score construction, PIECES dict, and registration.

**Key implementation details:**

1. ~100s at 92 BPM. Five sections, each with a different groove applied to the same musical material.

2. Section structure: For each section, create a `Timeline(bpm=92, groove=...)` with the appropriate groove preset. Use `grid_line()` with beat-relative durations to build phrases through each groove's timing/velocity warp. Place phrases with `grid_sequence()` or `sequence()`.

3. Voices and synth specs (module-level dicts):
   - Kick: `engine="kick_tom"`, `preset="808_hiphop"`. `normalize_peak_db=-6.0`.
   - Hi-hat: `engine="metallic_perc"`, `preset="closed_hat"`. `normalize_peak_db=-6.0`.
   - Chord pad: `engine="organ"`, `preset="septimal"`. `normalize_lufs=-24.0`.
   - Bass: `engine="polyblep"`, `preset="sub_bass"`. `normalize_lufs=-24.0`.
   - Clap: `engine="clap"`, `preset="909_clap"`. `normalize_peak_db=-6.0`.

4. Pattern generation:
   - Kick pattern: `prob_rhythm(16, onset_weights=[1.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0, 1.0, 0.0, 0.0, 0.0], span=S16_DUR, seed=1)`
   - Hat pattern: `prob_rhythm(16, onset_weights=[0.9, 0.4, 0.7, 0.4], span=S16_DUR, seed=2)` — 4-entry cycling pattern emphasizing beats
   - Evolve hat per section via `mutate_rhythm(hat_phrase, add_prob=0.05*i, shift_amount=0.003*i, seed=100+i)` where i is section index

5. Section placement: Compute bar boundaries in seconds. For each section:
   - Build a Timeline with the section's groove
   - Build phrases through that Timeline's grid helpers
   - Place at the section's start offset
   - Voices enter/exit at section boundaries per the design (kick+hat from bar 1, pad+bass from bar 13, clap from bar 27)

6. Chord material: Two-chord vamp. Otonal (1/1, 5/4, 3/2, 7/4) and utonal (1/1, 8/7, 4/3, 8/5). Two bars each, alternating. Use `add_note()` directly for sustained chord tones.

7. Effects: Shared reverb send bus ("room"). Master effects from `DEFAULT_MASTER_EFFECTS`. EQ highpass on master.

8. Humanization: `TimingHumanizeSpec(preset="tight_ensemble")`.

9. Registration: `study=True` in PieceDefinition.

**Step 1:** Write the complete study module.

**Step 2:** Add import and merge to `studies/__init__.py`:

```python
from code_musics.pieces.studies.study_groove import PIECES as _GROOVE_PIECES
# add _GROOVE_PIECES to merge_piece_maps()
```

**Step 3:** Run `make all` — must pass (0 errors, all tests pass).

**Step 4:** Run `make render PIECE=study_groove PLOT=0 ANALYSIS=0` — must produce WAV without error. Capture full output (do NOT pipe through tail). Use timeout of at least 300s.

---

### Task 2: study_aksak — "Unequal Pulses"

**Files:**

- Create: `code_musics/pieces/studies/study_aksak.py`
- Modify: `code_musics/pieces/studies/__init__.py`
- Test: `make render PIECE=study_aksak PLOT=0 ANALYSIS=0` + `make all`

**Design reference:** `docs/plans/2026-04-14-rhythm-studies-design.md` section 2.

**Key implementation details:**

1. ~105s. The BPM concept is tricky with aksak — define a pulse rate (140 pulses/min) and derive the pulse duration: `PULSE = 60.0 / 140.0`. The aksak bar is `AksakPattern.balkan_7(pulse=PULSE)` giving (2,2,3) groups = 7 pulses per bar. Bar duration = `aksak.total_duration`.

2. Use `AksakPattern.balkan_7(pulse=PULSE)` throughout. Two key rhythmic cells:
   - `aksak.to_rhythm()` — 3 spans (the group-level rhythm for kick/bass)
   - `aksak.to_pulses()` — 7 equal spans (the pulse-level rhythm for metallic perc)

3. Voices and synth specs:
   - Kick: `engine="kick_tom"`, `preset="909_house"`. `normalize_peak_db=-6.0`.
   - Metallic perc: `engine="metallic_perc"`, `preset="closed_hat"`. `normalize_peak_db=-6.0`.
   - Bass: `engine="polyblep"`, `preset="moog_bass"`. `normalize_lufs=-24.0`.
   - Melody: `engine="fm"`, `preset="bell"`. `normalize_lufs=-24.0`.
   - Counter: `engine="additive"`, `preset="soft_pad"`. `normalize_lufs=-24.0`.
   - Pad: `engine="filtered_stack"`, `preset="warm_pad"`. `normalize_lufs=-24.0`.

4. Pitch material: F0=110 Hz, 7-limit JI. Key ratios: 1/1, 7/6, 5/4, 3/2, 7/4.

5. Section placement: Compute section boundaries in aksak bars. Place phrases with `sequence()` using computed start times.

6. Groove section (bars 9-16): Use `cross_rhythm([(7, melody_tones), (3, counter_tones)], span=aksak.total_duration)` to build two interlocking phrases over one aksak bar, then sequence them.

7. Mutation section (bars 17-24): Apply `mutate_rhythm()` to the hat and kick phrases with increasing mutation parameters per bar (or per 2-bar group). Use different seeds for each repetition.

8. Poly peak (bars 25-30): `polyrhythm(3, 7, span=aksak.total_duration)` gives two RhythmCells. Use them with `line()` to build two new rhythmic layers that fight against the aksak pulse.

9. Release (bars 31-35): Only kick + pad, fading out.

10. Effects: Shared reverb send. Master effects. Humanization: `chamber`.

**Step 1:** Write the complete study module.

**Step 2:** Add import and merge to `studies/__init__.py`.

**Step 3:** `make all` — must pass.

**Step 4:** `make render PIECE=study_aksak PLOT=0 ANALYSIS=0` — timeout 300s.

---

### Task 3: study_rhythm_transforms — "Motif Development"

**Files:**

- Create: `code_musics/pieces/studies/study_rhythm_transforms.py`
- Modify: `code_musics/pieces/studies/__init__.py`
- Test: `make render PIECE=study_rhythm_transforms PLOT=0 ANALYSIS=0` + `make all`

**Design reference:** `docs/plans/2026-04-14-rhythm-studies-design.md` section 3.

**Key implementation details:**

1. ~110s at 108 BPM. Timeline with no groove (straight time — the rhythmic interest comes from transforms, not feel).

2. The motif: Build with `grid_line(tl, tones, durations)`. Use a rhythmically interesting 2-bar phrase:

   ```python
   MOTIF_DURATIONS = [Q, E, E, Q, E, E, Q, Q]  # 2 bars of 4/4
   MOTIF_TONES = [1.0, 9/8, 5/4, 3/2, 7/4, 5/3, 3/2, 1.0]  # harmonic series + descent
   motif = grid_line(tl, MOTIF_TONES, MOTIF_DURATIONS, pitch_kind="partial", ...)
   ```

3. Transforms: Apply each to the motif phrase:

   ```python
   motif_aug = augment(motif, 2.0)
   motif_dim = diminish(motif, 2.0)
   motif_retro = rhythmic_retrograde(motif)
   motif_disp = displace(motif, tl.duration(E))
   motif_rot1 = rotate(motif, 1)
   motif_rot2 = rotate(motif, 2)
   # etc.
   ```

4. Voices and synth specs:
   - V1 (original): `engine="harpsichord"`, `preset="baroque"`. `normalize_lufs=-24.0`.
   - V2 (augmented): `engine="organ"`, `preset="warm"`. `normalize_lufs=-24.0`.
   - V3 (diminished): `engine="fm"`, `preset="glass_lead"`. `normalize_lufs=-24.0`.
   - V4 (retrograde): `engine="piano"`, `preset="warm"`. `normalize_lufs=-24.0`.
   - V5 (displaced): `engine="additive"`, `preset="soft_pad"`. `normalize_lufs=-24.0`.
   - Perc layers: `engine="metallic_perc"` and `engine="noise_perc"`. `normalize_peak_db=-6.0`.

5. Percussion: Generate 3 interlocking layers from `ca_rhythm_layers(rule=30, steps=16, layers=3, span=tl.duration(S))`. Assign to metallic_perc voices with different presets/pans. These run throughout.

6. Section placement: Use `sequence()` to repeat each transformed phrase at the right section starts. Voices enter and exit at section boundaries:
   - Bars 1-8: V1 + perc
   - Bars 9-16: V1 + V2 + perc
   - Bars 17-22: V1 + V2 + V3 + perc
   - Bars 23-30: V1-V5 + perc (peak density)
   - Bars 31-38: V1 + V1 with rotation variants
   - Bars 39-40: V1 alone (coda)

7. Different pan positions for each voice to maintain clarity in the dense sections.

8. Effects: Shared reverb send. Master effects. Humanization: `chamber`.

**Step 1:** Write the complete study module.

**Step 2:** Add import and merge to `studies/__init__.py`.

**Step 3:** `make all` — must pass.

**Step 4:** `make render PIECE=study_rhythm_transforms PLOT=0 ANALYSIS=0` — timeout 300s.

---

### Task 4: Final verification

**Step 1:** Run `make all` — must pass with 0 errors.

**Step 2:** Render all three studies:

```
make render PIECE=study_groove
make render PIECE=study_aksak
make render PIECE=study_rhythm_transforms
```

Capture full output including loudness stats and artifact warnings. Use generous timeouts (300s each).

**Step 3:** Review render output for any artifact-risk warnings and address if needed.

---

## Delegation plan

Tasks 1-3 are independent and can be parallelized. Each study is a self-contained file with no shared dependencies beyond the existing library.

- **Tasks 1-3**: Parallel subagents, inherit model (creative composition needs strong judgment)
- **Task 4**: Main context, sequential verification

All three touch `studies/__init__.py` — use worktree isolation to avoid conflicts, or run sequentially and merge the **init**.py changes.
