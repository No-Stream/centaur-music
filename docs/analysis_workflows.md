# Analysis Workflows

This document describes the timestamp-oriented analysis workflow used by the repo.

The goal is to make comments like "2:10 in `ji_chorale`" easy to resolve for both
humans and coding agents without requiring a DAW or manually reading raw score
code.

## Goals

- make timestamp references first-class throughout the render workflow
- produce machine-readable analysis artifacts, not just images
- support fast iterative feedback on arrangement, form, texture, and timing feel
- verify that score-level timing humanization stays within musically plausible
  ensemble drift bounds

## Non-Goals

- building a full DAW or piano-roll editor inside this repo
- replacing the current render path with a heavy GUI-first workflow
- overfitting drift checks to exact classical quantization rather than human feel

## Timestamp Inspector

Use the inspector when you want score context around a timestamp rather than a
full render.

CLI:

- `make inspect PIECE=ji_chorale AT=2:10`
- `make inspect-window PIECE=ji_chorale AT=2:10 WINDOW=8`
- `make snippet PIECE=ji_chorale AT=2:10 WINDOW=12`
- `make render-window PIECE=ji_chorale START=130 DUR=12`

The inspector reports:

- normalized timestamp in seconds
- containing section label when the piece exposes section metadata
- notes starting near the target time
- notes active through the target time
- active voices and rough register/frequency ranges
- nearby density information and any drift warnings
- paths to relevant render artifacts when available

This is the fastest path from conversational references to score context.

## Snippet Rendering

Use snippet rendering when you want to hear a local passage without paying the
cost of a full score render.

CLI:

- `make snippet PIECE=ji_chorale AT=2:10 WINDOW=12`
- `make render-window PIECE=ji_chorale START=130 DUR=12`

Behavior:

- snippet rendering is currently supported for score-backed pieces
- the render path adds a small hidden pre/post margin so note attacks, releases,
  and reverb feel natural at the exported boundaries
- the exported WAV is still trimmed to the exact requested window
- snippet renders write separate artifacts and metadata so they do not overwrite
  the latest full-piece render

## Timeline Artifacts

Render analysis emits a queryable timeline JSON alongside the existing analysis
manifest and plots.

Initial artifact contents:

- section boundaries with labels
- note onsets and offsets by voice
- voice activity windows
- time-binned onset density and active-note density
- optional rough registral summaries per time window

Current shape:

- one stable latest timeline artifact path recorded in the analysis manifest
- versioned copies alongside existing versioned render artifacts
- plain JSON that can be read directly in Python

The timeline JSON is the substrate used by the inspector and is the preferred
machine-readable entry point for any future UI work.

## Timing Drift Analysis

Score analysis explicitly measures render-time score drift so timing
humanization remains human and ensemble-like rather than becoming sloppy.

The key requirement is not "no drift." Global and local drift are both fine.
What matters is that voices remain plausibly synced relative to each other.

### Measured Drift Stats

Core metrics:

- mean absolute drift offset across placed note attacks
- median absolute drift offset
- max absolute drift offset
- percentile summaries such as p95 absolute drift
- max pairwise inter-voice attack spread for near-simultaneous note groups
- mean pairwise inter-voice attack spread for near-simultaneous note groups
- windowed inter-voice spread summaries so we can detect sections where sync
  degrades even if whole-piece averages still look fine

Interpretation split:

- absolute drift: how far notes moved from authored score time
- inter-voice spread: how far nominally aligned voices diverged from each other

The second category is the more important musical safety metric.

### Matching Strategy

The current implementation stays symbolic and deterministic:

- analyze note attacks from the symbolic score after timing humanization is
  resolved
- group near-simultaneous authored attacks into onset clusters
- compare the resolved attacks inside each cluster
- compute spread statistics across the full piece and optionally per voice pair

This avoids trying to infer timing drift from audio.

### Output Surface

Drift diagnostics are recorded in the score section of the analysis manifest and
included in warnings.

Initial manifest additions:

- global drift summary
- time-windowed drift summary across the piece
- optional per-voice or per-voice-pair spread summaries
- warning list specific to timing drift

Current warnings include:

- drift regularly exceeds a human ensemble feel
- large inter-voice timing spreads may blur coordinated attacks
- some score windows drift more than the surrounding texture
- one voice pair shows a persistent lead-lag bias

Thresholds are intentionally conservative and should stay documented rather than
hidden in code comments.

## Artifact Surfaces

Current analysis outputs for score-backed renders include:

- piano-roll PNG when requested through `make render`
- score-density PNG
- analysis manifest JSON
- timeline JSON
- export-time log lines with peak, true-peak, and integrated LUFS diagnostics
- drift summary and drift-window stats inside the score analysis payload

## Documentation Split

Keep the doc split lightweight and consistent:

- `AGENTS.md`: high-level note that timestamp inspection and drift-aware analysis
  are part of the normal workflow
- this file: concrete commands, artifact surfaces, and drift semantics
- `docs/score_api.md`: score-domain API details when the public score surface changes
