// Trivial Canvas2D scene for end-to-end capture-pipe validation. No three.js
// dependency, so capture tests stay hermetic and fast.
//
// renderFrame paints the full canvas a solid color derived deterministically
// from floor(timeSeconds * 10) (a hue rotation) plus a centered frame-time
// readout, giving verifiable per-frame variation.

/**
 * @param {{data: any, width: number, height: number, canvas: HTMLCanvasElement, hud?: boolean}} args
 */
export async function createScene({ width, height, canvas }) {
  const ctx = canvas.getContext("2d");
  if (ctx === null) {
    throw new Error("smoke scene: failed to acquire 2d rendering context");
  }

  return {
    renderFrame(timeSeconds) {
      const step = Math.floor(timeSeconds * 10);
      const hue = (step * 37) % 360;

      ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
      ctx.fillRect(0, 0, width, height);

      ctx.fillStyle = "#ffffff";
      ctx.font = `${Math.round(height * 0.08)}px monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(timeSeconds.toFixed(3), width / 2, height / 2);
    },
  };
}
