# Hexany Garden — video / visualization seed

Seed notes for a future session: build a rendered video for
`hexany_garden` (see `code_musics/pieces/hexany_garden.py`, whose module
docstring is the authoritative description of the piece's structure).

## Why this piece visualizes itself

The 1-3-5-7 hexany **is an octahedron**: 6 vertices (the six notes,
each a product of two factors) and 8 triangular faces (the eight
triads — 4 otonal, 4 utonal). Every structural device in the piece has
a direct geometric meaning:

| Musical object | Geometry |
|---|---|
| The six scale degrees | The six vertices |
| A triad region (`_CHORD_SLOTS`) | A face |
| Otonal vs utonal region | Opposite faces (light / shadow side) |
| Factor-adjacent walker move | Travel along an edge |
| Polar leap (1·3↔5·7, 1·5↔3·7, 1·7↔3·5) | A jump through the solid's center (the three polar pairs are the three vertex-antipode pairs) |
| The closing 4:7 dyad (degrees 0 & 5) | A single edge, alone |

## Concept pitches

1. **The Octahedron** (strongest single idea). Slowly orbiting
   wireframe octahedron. Vertices flash as the walker sounds them
   (voice → size/color, velocity → brightness); the active chord
   region's face glows — warm gold for otonal, cool blue-violet for
   utonal; polar leaps draw arcs through the interior. Section
   lighting: S1 dew — dawn fog, geometry barely visible; S2 sunlight;
   S3 the same solid lit from behind (shadow side toward camera); S4
   both light sources fighting; S5 fade until only the 0–5 edge glows,
   then that too dims.
2. **The Garden**. Procedural growth (L-system vines): each walker
   note extends a vine; riff replays re-trace the same branch
   (thickening it — repetition literally becomes structure); the thumb
   line grows a second plant on a 7-step phyllotaxis; glint notes =
   fireflies; the bar-93 cluster chord = one flower opening; S5 =
   petals fall, one seed remains. Night palette for utonal stretches.
3. **The Web** (cheapest bespoke idea). Wilson lattice drawn as a
   spider web with dew droplets (S1 is literally "Dew"); notes light
   droplets; chord regions tension web sectors.
4. **Data-driven abstract** (cheapest overall): particle/typography
   render straight off the timeline JSON. Fine fallback, least soul.

**Recommendation: 1 + 2 hybrid** — the octahedron hangs in the sky
(sun/moon of the scene) doing the harmonic geometry, while the garden
below grows from the note stream. Structure above, life below; the
piece's whole thesis in one frame.

## Data already available (no new infra needed to start)

- `output/hexany_garden/hexany_garden.timeline.json` — 1805 notes with
  `voice`, `freq_hz`, `partial`, resolved start/end seconds; `sections`
  with labeled boundaries; per-second `windows` with active-voice and
  onset density. Degree recovery: `ratio = partial / 2**floor(log2(partial))`,
  match against `code_musics.tuning.hexany()` values.
- `output/hexany_garden/hexany_garden.wav` — the audio to mux.
- `hexany_garden.analysis.json` + mel spectrogram / chromagram PNGs —
  usable as texture layers or for onset-reactive effects.
- Piece source constants for exact structure: `_CHORD_SLOTS` (bar →
  region), `_DEGREE_FACTORS` (vertex identity), `_RIFF_PHRASE_STARTS`
  (when the lead is repeating vs free — visualize riff-lock as the
  walker re-tracing a lit path), `_SEVEN_GRID_BARS`, `_QUOTE_BARS`,
  section bar boundaries.

Section times (seconds): S1 0–41.7 · S2 41.7–125.2 · S3 125.2–198.3 ·
S4 198.3–281.7 · S5 281.7–345.8. BPM 92, bar ≈ 2.609 s, F0 = 92.5 Hz.

## Suggested first step in the video session

Add a small `--viz-json` exporter (or a scratch script) that re-walks
`build_score()` and emits per-note **semantic** annotations the
timeline lacks: scale degree, octave, factor pair, active region +
otonal/utonal flag, riff-phase (record/replay/free), and whether a note
is a motif quote, dyad hit, or polar leap. Everything is deterministic
(seeded), so this is a pure re-derivation.

## Technical routes (pick per ambition)

- **Three.js + headless capture**: write a self-contained HTML page
  that loads the viz JSON and renders with `requestAnimationFrame`
  driven by a fixed timestep; capture frames headless via the
  playwright MCP tooling or `ffmpeg` screen pipe; mux with the WAV
  (`ffmpeg -i frames -i wav`). Best-looking route; GPU shading for the
  glow/fog moods.
- **Python/matplotlib or moderngl frame renderer**: slower, zero new
  deps beyond the repo's matplotlib; fine for concept 3/4, painful for
  1/2.
- **Manim**: good for the octahedron concept specifically (camera
  moves, geometry), heavier dependency.

Render at 1920×1080, 30 fps (~10.4k frames for 5:46); prototype at
480p/12fps on a 30 s window first (`make render-window` boundaries
align with the timeline JSON's absolute times).

## Palette / mood references

Berlin-era Bowie ambient sides ("Warszawa"), Boards of Canada haze,
Four Tet organic warmth. Dawn → day → dusk-shadow → golden hour → night
maps S1→S5. Grain and bloom over clean lines; the geometry should feel
found, not CGI.
