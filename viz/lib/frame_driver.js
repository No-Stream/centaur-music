// Generic deterministic capture harness. Loaded by each scene's index.html
// as an ES module (`<script type="module" src="../lib/frame_driver.js">`).
//
// Reads query params:
//   viz  - URL of a viz.json data file (optional; data is null if absent)
//   hud  - "1" to request scenes render a debug overlay
//
// Creates a full-viewport canvas, imports the sibling ./scene.js, calls
// createScene, and exposes window.__viz for the Python capture driver to
// drive frame-by-frame. Scenes must be pure functions of timeSeconds -- no
// requestAnimationFrame, Date.now(), or performance.now() are used here.

import { loadVizData } from "./viz_data.js";

/** @type {{ready: boolean, renderFrame: (frameIndex: number, fps: number) => Promise<void>}} */
window.__viz = { ready: false, renderFrame: async () => {} };

async function boot() {
  const params = new URLSearchParams(window.location.search);
  const vizUrl = params.get("viz");
  const hud = params.get("hud") === "1";

  document.body.style.margin = "0";
  document.body.style.overflow = "hidden";

  const canvas = document.createElement("canvas");
  canvas.style.display = "block";
  document.body.appendChild(canvas);

  const width = window.innerWidth;
  const height = window.innerHeight;
  canvas.width = width;
  canvas.height = height;

  const data = vizUrl === null ? null : await loadVizData(vizUrl);

  // Resolve relative to the *page* URL (not this module's URL) so the same
  // generic frame_driver.js works for any scene directory's sibling scene.js.
  const sceneModuleUrl = new URL("./scene.js", window.location.href).href;
  const { createScene } = await import(sceneModuleUrl);
  const scene = await createScene({ data, width, height, canvas, hud });

  window.__viz = {
    ready: true,
    renderFrame: async (frameIndex, fps) => {
      await scene.renderFrame(frameIndex / fps);
    },
  };
}

boot().catch((error) => {
  window.__viz_error = {
    message: error && error.message ? error.message : String(error),
    stack: error && error.stack ? error.stack : "",
  };
});
