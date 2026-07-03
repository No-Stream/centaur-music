// hexany_garden scene: octahedron above (the harmony's geometry), crystal
// garden below (the note stream made visible), in an open alien-void
// landscape. Pure function of timeSeconds — see viz/lib/frame_driver.js.

import * as THREE from "three";
import { prepareMusic } from "./music.js";
import { createWorld } from "./world.js";
import { createOctahedron } from "./octahedron.js";
import { createGarden } from "./garden.js";
import { createPost } from "./post.js";

export async function createScene({ data, width, height, canvas, hud }) {
  if (!data) {
    throw new Error("hexany_garden scene requires ?viz=<viz.json url>");
  }
  const music = prepareMusic(data);

  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(1);
  renderer.setSize(width, height, false);

  const scene = new THREE.Scene();
  const world = createWorld(scene, music, width, height);
  const octahedron = createOctahedron(scene, world.mirror, music);
  const garden = createGarden(scene, world.mirror, music);
  const post = createPost(renderer, scene, world.camera, width, height);

  let hudEl = null;
  if (hud) {
    hudEl = document.createElement("div");
    hudEl.style.cssText =
      "position:fixed;top:8px;left:8px;color:#8f8;font:12px monospace;" +
      "background:rgba(0,0,0,0.6);padding:6px;white-space:pre;z-index:10";
    document.body.appendChild(hudEl);
  }

  function renderFrame(t) {
    world.update(t);
    octahedron.update(t);
    garden.update(t);
    post.render(t, music.glowAt(t));
    if (hudEl) {
      const slot = music.slotAt(t);
      const section = music.sections[music.sectionIndexAt(t)];
      const bar = Math.floor(t / music.barSeconds) + 1;
      hudEl.textContent =
        `t=${t.toFixed(2)}s bar=${bar}\n` +
        `${section.label}\n` +
        `region=${slot.name} otonal=${slot.otonal}\n` +
        `rms=${music.rmsAt(t).toFixed(4)} glow=${music.glowAt(t).toFixed(3)}`;
    }
  }

  return { renderFrame };
}
