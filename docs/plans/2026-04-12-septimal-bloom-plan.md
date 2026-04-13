# Septimal Bloom Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a ~2:50 piece for two Surge XT voices in 7-limit JI with A/B/A' form, slow bloom effects arc, and an elegiac melody.

**Architecture:** Single piece file with hand-composed chords and melody using `score.add_note()`. Two Surge XT voices (warbly pad + cleaner melody). Shared send bus for reverb/delay. Master-bus reverb wet automated for the bloom arc.

**Tech Stack:** Surge XT via `engine="surge_xt"` + `surge_params`, native effect chain (bricasti/reverb, delay, saturation), score-time automation.

---

## Reference files

- Design doc: `docs/plans/2026-04-12-septimal-bloom-design.md`
- Study (existing template): `code_musics/pieces/studies/study_surge_xt.py`
- Full piece example: `code_musics/pieces/slow_glass.py`
- Piece with send bus + automation: `code_musics/pieces/colundi_sequence.py`
- Piece registration: `code_musics/pieces/__init__.py`
- Shared effects: `code_musics/pieces/_shared.py`
- Automation types: `code_musics/automation.py`
- Score API: `code_musics/score.py` (EffectSpec, SendBusSpec, VoiceSend, Score)
- Synth API: `docs/synth_api.md`, `docs/score_api.md`

## Surge XT patch constants

Pad (warbly):

```python
_PAD_PARAMS = {
    "a_osc_1_type": 0.0,           # Classic saw
    "a_osc_1_unison_voices": 0.15,  # 3 voices
    "a_osc_1_unison_detune": 0.12,  # ~12 cents
    "a_filter_1_type": 0.295,       # LP Vintage Ladder
    "a_filter_1_cutoff": 0.52,      # ~500 Hz
    "a_filter_1_resonance": 0.12,   # 12%
    "a_amp_eg_attack": 0.50,        # ~350 ms
    "a_amp_eg_decay": 0.55,         # ~550 ms
    "a_amp_eg_sustain": 0.80,       # 80%
    "a_amp_eg_release": 0.58,       # ~700 ms
}
```

Melody (clean shimmer):

```python
_MELODY_PARAMS = {
    "a_osc_1_type": 0.05,           # Sine
    "a_osc_1_unison_voices": 0.05,  # 2 voices
    "a_osc_1_unison_detune": 0.04,  # ~4 cents -- shimmer
    "a_filter_1_type": 0.295,       # LP Vintage Ladder
    "a_filter_1_cutoff": 0.60,      # ~1 kHz
    "a_filter_1_resonance": 0.06,   # 6%
    "a_amp_eg_attack": 0.40,        # ~140 ms
    "a_amp_eg_decay": 0.50,         # ~350 ms
    "a_amp_eg_sustain": 0.85,       # 85%
    "a_amp_eg_release": 0.55,       # ~550 ms
}
```

---

### Task 1: Skeleton — file, voices, effects, registration

**Files:**

- Create: `code_musics/pieces/septimal_bloom.py`
- Modify: `code_musics/pieces/__init__.py`

**Step 1:** Create `septimal_bloom.py` with:

- Module docstring (form, tuning, voice layout)
- Constants: `F0_HZ = 110.0`, pad/melody surge_params dicts, timing constants for section boundaries (`A_START = 0.0`, `B_START = 60.0`, `A2_START = 110.0`, `TOTAL = 160.0`)
- `build_score() -> Score` that:
  - Creates a shared send bus `"hall"` with bricasti reverb (or fallback native reverb)
  - Adds a delay effect to the send bus alongside the reverb
  - Adds master_effects with saturation
  - Creates `"pad"` voice: engine=surge_xt, surge_params=_PAD_PARAMS, normalize_lufs=-20, sends to "hall" at send_db=-6
  - Creates `"melody"` voice: engine=surge_xt, surge_params=_MELODY_PARAMS, normalize_lufs=-18, sends to "hall" at send_db=-4 (more reverb on melody)
  - Placeholder comment blocks for sections A, B, A'
  - Returns score
- `PIECES` dict with `PieceDefinition` including `sections` tuple

**Step 2:** Register in `pieces/__init__.py`: import and add to `merge_piece_maps`.

**Step 3:** Verify: `make list | grep septimal_bloom` and `make render PIECE=septimal_bloom` (should render silence or near-silence, no crash).

---

### Task 2: Section A — chords (0:00–1:00)

**File:** `code_musics/pieces/septimal_bloom.py`

**Step 1:** Write the A section chord progression. Four chords, each ~8s, with the first chord entering sparse (bass only for 4s, then upper voices join). Chords overlap slightly via release tails.

Chord voicings (partials of f0=110 Hz):

```
t=0.0   I bass:    partial=1     dur=8    amp_db=-10  vel=0.7
t=4.0   I upper:   partial=5/2   dur=4    amp_db=-9   vel=0.7
t=4.0   I upper:   partial=3     dur=4    amp_db=-9   vel=0.7
t=8.0   iv7:       4/3, 7/3, 8/3          dur=8    amp_db=-9
t=16.0  vi:        5/3, 2, 7/2            dur=8    amp_db=-9
t=24.0  I7:        1, 5/4, 3/2, 7/4       dur=10   amp_db=-9
```

Repeat the progression a second time (t=34 onward) with fuller voicings — add octave doublings or extra chord tones:

```
t=34.0  I:         1, 5/4, 5/2, 3         dur=8    amp_db=-8
t=42.0  iv7:       4/3, 7/3, 8/3, 4       dur=8    amp_db=-8
t=50.0  vi:        5/3, 2, 7/2, 10/3      dur=8    amp_db=-8
t=58.0  I7:        1, 5/4, 3/2, 7/4       dur=8    amp_db=-8
                   (held into B section)
```

**Step 2:** Render snippet: `make render-window PIECE=septimal_bloom START=0 DUR=30` to hear the opening bloom.

---

### Task 3: Section A — melody (enters ~15s)

**File:** `code_musics/pieces/septimal_bloom.py`

**Step 1:** Write the A melody on the `"melody"` voice. Sparse, descending, 3-4 notes per chord. Range: partials 5/2 to 7/2 (~275-385 Hz). The melody enters gently during the second chord (iv7) and grows slightly more present.

Sketch (exact timing/pitches to be refined):

```
# First pass (chords 2-4, sparse introduction)
t=10.0  partial=3     dur=3.0   amp_db=-8   vel=0.75   # E4, over iv7
t=14.0  partial=7/2   dur=2.5   amp_db=-7   vel=0.8    # ~G4 (sept 7th)
t=17.0  partial=3     dur=2.0   amp_db=-8   vel=0.75   # E4, stepping down
t=20.0  partial=5/2   dur=3.5   amp_db=-8   vel=0.7    # C#4, resting
t=25.0  partial=7/4   dur=3.0   amp_db=-7   vel=0.8    # sept 7th, lower
t=29.0  partial=3/2   dur=2.5   amp_db=-9   vel=0.7    # E3, descending

# Second pass (fuller, more confident)
t=35.0  partial=4     dur=2.0   amp_db=-6   vel=0.85   # A4, reaching up
t=38.0  partial=7/2   dur=2.5   amp_db=-6   vel=0.85   # sept 7th
t=41.0  partial=3     dur=3.0   amp_db=-7   vel=0.8    # E4
t=45.0  partial=8/3   dur=2.0   amp_db=-7   vel=0.8    # D4
t=48.0  partial=5/2   dur=2.5   amp_db=-7   vel=0.75   # C#4
t=52.0  partial=7/4   dur=3.0   amp_db=-6   vel=0.85   # sept 7th — held into B
t=56.0  partial=7/4   dur=6.0   amp_db=-7   vel=0.8    # bridge tone, carries into B
```

**Step 2:** Render snippet: `make render-window PIECE=septimal_bloom START=8 DUR=25` to hear melody entrance.

---

### Task 4: Section B — chords (1:00–1:50)

**File:** `code_musics/pieces/septimal_bloom.py`

**Step 1:** Write B section chords. The utonal/drift section. Each chord ~10-12s for maximum spaciousness.

```
t=60.0  I utonal:  1, 8/5, 4/3                dur=12   amp_db=-9   vel=0.7
t=72.0  VII:       7/4, 35/16, 21/8            dur=12   amp_db=-8   vel=0.75
t=84.0  bVI:       8/5, 2, 12/5                dur=12   amp_db=-9   vel=0.7
t=96.0  V7:        3/2, 15/8, 9/4, 21/8        dur=14   amp_db=-8   vel=0.75
                   (held into A')
```

**Step 2:** Render snippet: `make render-window PIECE=septimal_bloom START=58 DUR=30` to hear the A-to-B transition.

---

### Task 5: Section B — melody

**File:** `code_musics/pieces/septimal_bloom.py`

**Step 1:** Write B melody. More exposed, higher, slower. Fewer notes, longer durations. Lingers on 7/4 and 7/6.

Sketch:

```
# Melody sits higher, breathes more
t=63.0  partial=7/2   dur=5.0   amp_db=-5   vel=0.85   # sept 7th, exposed
t=69.0  partial=3     dur=4.0   amp_db=-6   vel=0.8    # fifth
t=74.0  partial=35/16 dur=5.0   amp_db=-5   vel=0.85   # 5/4 of 7/4 — the reaching note
t=80.0  partial=7/2   dur=4.0   amp_db=-6   vel=0.8    # back to sept 7th
t=85.0  partial=12/5  dur=4.0   amp_db=-6   vel=0.8    # neutral, resting
t=90.0  partial=5/2   dur=3.5   amp_db=-6   vel=0.8    # C#4
t=95.0  partial=9/4   dur=4.0   amp_db=-5   vel=0.85   # pulling home
t=100.0 partial=7/4   dur=6.0   amp_db=-6   vel=0.8    # sept 7th — bridge back
```

**Step 2:** Render snippet: `make render-window PIECE=septimal_bloom START=60 DUR=30` to hear B melody.

---

### Task 6: Section A' — return and dissolve (1:50–2:40)

**File:** `code_musics/pieces/septimal_bloom.py`

**Step 1:** Write A' chords. Same progression as A pass 2 but the final chord resolves differently — I without 7/4, or I with just root+fifth trailing off.

```
t=110.0  I:         1, 5/4, 5/2, 3         dur=8    amp_db=-8
t=118.0  iv7:       4/3, 7/3, 8/3, 4       dur=8    amp_db=-9
t=126.0  vi:        5/3, 2, 7/2, 10/3      dur=8    amp_db=-9
t=134.0  I7:        1, 5/4, 3/2, 7/4       dur=8    amp_db=-10  (quieter)
t=142.0  I (bare):  1, 3                    dur=12   amp_db=-12  (dissolving)
```

**Step 2:** Write A' melody. Restates the A theme but lands on 7/6 instead of 1. Thins out and fades.

```
t=112.0 partial=4     dur=2.0   amp_db=-6   vel=0.85
t=115.0 partial=7/2   dur=2.5   amp_db=-7   vel=0.8
t=118.0 partial=3     dur=3.0   amp_db=-7   vel=0.8
t=122.0 partial=5/2   dur=2.5   amp_db=-8   vel=0.75
t=126.0 partial=7/4   dur=4.0   amp_db=-7   vel=0.8
t=131.0 partial=7/6   dur=6.0   amp_db=-8   vel=0.7    # lands on 7/6 — home but askew
t=140.0 partial=1     dur=8.0   amp_db=-14  vel=0.5    # barely there, dissolving
```

**Step 3:** Render: `make render PIECE=septimal_bloom` (full piece, generous timeout).

---

### Task 7: Effects — reverb bloom automation + saturation

**File:** `code_musics/pieces/septimal_bloom.py`

**Step 1:** Add reverb wet automation to the send bus's bricasti/reverb effect. The bloom arc:

- A start (t=0): wet=0.18
- A peak (t=50): wet=0.25
- B peak (t=85): wet=0.38
- A' (t=130): wet=0.28
- Dissolve (t=155): wet=0.35 (reverb stays present as notes fade)

Use `AutomationSpec` with `AutomationTarget(kind="control", name="wet")` (or `"wet_level"` depending on which reverb is used) on the send bus effect's `automation` list.

**Step 2:** Add melody voice send_db automation — send level increases through B for more wash:

- A (t=0): send_db=-4
- B (t=60): send_db=-1
- A' (t=110): send_db=-3

**Step 3:** Render full piece: `make render PIECE=septimal_bloom`.

---

### Task 8: Polish — listen, adjust, render final

**Step 1:** Render the full piece and examine:

- Piano roll for structure
- Spectrogram for frequency content / harmonic movement
- Analysis manifest for any artifact warnings
- Duration / loudness stats

**Step 2:** Adjust based on render output:

- Chord voicing balance (amp_db adjustments)
- Melody phrasing and timing
- Effect wetness levels
- Section transition smoothness
- Overall loudness / dynamics

**Step 3:** Final render: `make render PIECE=septimal_bloom`

**Step 4:** Run `make test` to verify no regressions (the piece smoke test should auto-discover).

---

## Rendering notes

- Full piece (~170s) will be slow to render — use `make render-window` for section iteration
- Set generous Bash timeout (600s) for full renders
- Two Surge XT voice passes at ~170s each + master effects = expect ~60-90s wall time
- The plugin is cached — melody params overwrite pad params between render passes, which is fine since voices render sequentially
