# Visualization Pipeline

Turn a rendered piece into a music video: a semantic JSON export drives a
deterministic browser scene, captured headless into an H.264 file with the
piece's audio muxed in.

```
make render PIECE=<name>      # 1. render audio (writes output/<name>/<name>.wav)
make viz PIECE=<name>         # 2. export output/<name>/<name>.viz.json
make viz-video PIECE=<name>   # 3. capture viz/<name>/ scene -> mp4
```

One-time tooling bootstrap: `make viz-setup` (static ffmpeg into `tools/`
(gitignored), vendored three.js in `viz/vendor/`, `playwright` dev dep
driving the system Chrome — no browser download).

## Viz JSON export (`code_musics/viz_export.py`)

`export_piece_viz(piece_name)` (`render.py`; CLI `main.py <piece>
--export-viz`) builds the score and writes `output/<name>/<name>.viz.json`,
schema version `VIZ_SCHEMA_VERSION = 1`:

- `notes[]` — the full resolved timeline (sorted by resolved start), each
  with `voice_name`, `note_index`, `start_seconds` / `end_seconds` /
  `duration_seconds` (resolved), `authored_start_seconds`, `freq_hz`,
  `partial`, `velocity`, `amp`, `amp_db`, raw `label`, and `semantics` —
  the parsed label. If the score carries a `Timeline`, each note also includes
  `musical_location` and `authored_musical_location` with bar, beat, absolute
  beat, and seconds.
- `voices{}` — `pan`, `mix_db`, `is_percussive`, `note_count` per voice.
- `sections[]`, `f0_hz`, `sample_rate`, `total_duration_seconds`.
- `musical_time{}` — present when `Score.timeline` is set; includes meter,
  pickup, groove, and tempo-map anchors for tempo-aware visual scenes.
- `annotations{}` — opaque piece-provided blob (see below).
- `envelope{}` — mix RMS envelope (`hop_seconds` grid, `rms` + `rms_db`
  arrays) computed from the rendered WAV. The exporter fails fast if the
  piece has not been rendered.

### Structured labels

`NoteEvent.label` doubles as the semantic channel. Format:
`kind;key=value;key=value` — e.g.
`walker;deg=3;oct=4;phase=replay;leap=polar;grid=straight`.
`parse_viz_label` coerces int/float values, turns bare tokens into boolean
tags, treats a label with no `;`/`=` as a legacy `{"kind": label}`, skips
malformed segments, and never raises (labels are a creative authoring
surface; a bad label must not break a render). Labels are audio-neutral
metadata — attach them at `add_note` time without touching RNG draw order.

### Piece annotations

`PieceDefinition.build_viz_annotations: Callable[[], dict] | None` — an
optional registry hook returning a JSON-serializable blob of piece-level
structure (chord tables, section bars, tuning geometry, timing grid).
`hexany_garden` is the reference implementation: degree ratios/factors,
polar pairs, region table with otonal/utonal polarity, full chord-slot
timeline in bars and seconds, riff phrase starts, and landmark times.

## Browser scene contract (`viz/lib/`)

A scene lives in `viz/<name>/` with an `index.html` that loads
`../lib/frame_driver.js` as an ES module and a sibling `scene.js` exporting:

```js
export async function createScene({ data, width, height, canvas, hud })
// -> { renderFrame(timeSeconds) }
```

`renderFrame` MUST be a pure function of time: no requestAnimationFrame,
`Date.now()`, `performance.now()`, or frame-to-frame state (stateful
smoothing produces seams at capture-shard boundaries). Use
`mulberry32(seed)` from `viz/lib/viz_data.js` for deterministic randomness;
`notesInRange` / `envelopeAt` help with time-indexed lookups. The driver
exposes `window.__viz.renderFrame(frameIndex, fps)` plus a `ready` flag,
and surfaces scene errors via `window.__viz_error`. `?hud=1` requests a
debug overlay. `viz/_smoke/` is a minimal Canvas2D scene used by the
capture tests.

Three.js gotcha worth repeating: additive materials must set `fog: false`,
or scene fog *adds* its color to black (invisible) geometry, producing
ghost shapes.

## Capture (`viz/capture.py`)

`make viz-video PIECE=<name> [VIZ_WIDTH= VIZ_HEIGHT= VIZ_FPS= VIZ_WORKERS=
VIZ_CRF= START= END= VIZ_OUT=]` — defaults 1920x1080 @ 30 fps, 4 workers,
CRF 17, full piece, `output/<name>/<name>_<W>x<H>.mp4`.

The driver shards the frame range across worker processes; each launches
headless Chrome (`channel="chrome"`, SwiftShader WebGL), serves the repo
root over a local ephemeral-port HTTP server, screenshots each frame, and
pipes PNGs straight into a per-shard `ffmpeg -f image2pipe` libx264 encode
(no frame files on disk). Shard segments are concat'ed losslessly and the
piece WAV is muxed (AAC 320k, offset = start frame / fps).

Throughput on this machine (SwiftShader, software GL): ~0.65 s/frame/worker
at 1080p, ~4.7 s/frame/worker at 4K; both scale near-linearly with workers
up to CPU count. A full 5:46 piece is ~35-60 min at 1080p/6 workers and
~1.5-2 h at 4K/6-8 workers.

Tests: `tests/test_viz_export.py` (exporter, always runs) and
`tests/test_viz_capture.py` (pure functions always; the end-to-end smoke
capture is gated behind `VIZ_SMOKE=1` plus tool availability so `make test`
stays hermetic).
