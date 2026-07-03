// The void-landscape: gradient sky dome, dark glass ground, drifting motes,
// stars, fog, and the single continuous camera journey across the sections.

import * as THREE from "three";
import { mulberry32 } from "../lib/viz_data.js";
import { blendKeyframes, smoothstep } from "./music.js";

// Section mood keyframes. v = [skyTopR,G,B, horizonR,G,B, fogR,G,B,
// fogDensity, starAlpha, tintStrength]
function moodKeyframes(sections) {
  const s = sections.map((x) => x.start_seconds);
  return [
    // S1 dew: pale pre-dawn glow low in heavy fog, world barely there.
    { t: s[0], v: [0.010, 0.012, 0.030, 0.145, 0.105, 0.095, 0.050, 0.045, 0.060, 0.030, 0.05, 0.35] },
    // S2 first bloom: open airy, teal zenith, warm horizon band.
    { t: s[1], v: [0.016, 0.038, 0.065, 0.200, 0.155, 0.115, 0.050, 0.060, 0.080, 0.012, 0.0, 0.55] },
    // S3 the turn: backlit dusk, violet sky, cold rose horizon.
    { t: s[2], v: [0.012, 0.008, 0.032, 0.095, 0.048, 0.100, 0.030, 0.020, 0.050, 0.018, 0.15, 0.85] },
    // S4 full garden: golden hour fighting the violet.
    { t: s[3], v: [0.020, 0.014, 0.040, 0.165, 0.100, 0.080, 0.040, 0.030, 0.050, 0.014, 0.25, 0.85] },
    // S5 seed: deep night.
    { t: s[4], v: [0.002, 0.003, 0.008, 0.028, 0.021, 0.048, 0.008, 0.008, 0.018, 0.022, 0.85, 0.5] },
  ];
}

const SKY_VERT = `
varying vec3 vWorld;
void main() {
  vWorld = (modelMatrix * vec4(position, 1.0)).xyz;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const SKY_FRAG = `
varying vec3 vWorld;
uniform vec3 topColor;
uniform vec3 horizonColor;
uniform vec3 tintColor;
uniform float tintStrength;
void main() {
  float h = normalize(vWorld).y;
  float horizonBand = exp(-max(h, 0.0) * 4.5);
  vec3 sky = mix(topColor, horizonColor, horizonBand);
  // Harmonic tint breathes into the lower sky around the horizon.
  sky += tintColor * tintStrength * exp(-abs(h) * 6.0) * 0.45;
  gl_FragColor = vec4(sky, 1.0);
}`;

export function createWorld(scene, music, width, height) {
  const moods = moodKeyframes(music.sections);

  // Sky dome.
  const skyUniforms = {
    topColor: { value: new THREE.Color(0, 0, 0) },
    horizonColor: { value: new THREE.Color(0, 0, 0) },
    tintColor: { value: new THREE.Color(0, 0, 0) },
    tintStrength: { value: 0 },
  };
  const sky = new THREE.Mesh(
    new THREE.SphereGeometry(420, 32, 20),
    new THREE.ShaderMaterial({
      uniforms: skyUniforms,
      vertexShader: SKY_VERT,
      fragmentShader: SKY_FRAG,
      side: THREE.BackSide,
      depthWrite: false,
    }),
  );
  sky.renderOrder = -10;
  scene.add(sky);

  // Stars: static seeded points high in the dome, alpha ramps with night.
  const starRng = mulberry32(1357);
  const starCount = 900;
  const starPositions = new Float32Array(starCount * 3);
  for (let i = 0; i < starCount; i += 1) {
    const az = starRng() * Math.PI * 2;
    const el = Math.asin(0.12 + 0.88 * starRng());
    const r = 400;
    starPositions[i * 3] = r * Math.cos(el) * Math.cos(az);
    starPositions[i * 3 + 1] = r * Math.sin(el);
    starPositions[i * 3 + 2] = r * Math.cos(el) * Math.sin(az);
  }
  const starGeom = new THREE.BufferGeometry();
  starGeom.setAttribute("position", new THREE.BufferAttribute(starPositions, 3));
  const starMat = new THREE.PointsMaterial({
    color: 0xcfd8ff,
    size: 1.6,
    sizeAttenuation: false,
    transparent: true,
    opacity: 0,
    depthWrite: false,
  });
  const stars = new THREE.Points(starGeom, starMat);
  stars.renderOrder = -9;
  scene.add(stars);

  // Ground: dark glass disc. Mirrored scene elements render beneath it
  // (renderOrder -2) and bleed through its not-quite-full opacity.
  const ground = new THREE.Mesh(
    new THREE.CircleGeometry(430, 180),
    new THREE.MeshBasicMaterial({
      color: 0x05060b,
      transparent: true,
      opacity: 0.78,
      depthWrite: false,
    }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.renderOrder = -1;
  scene.add(ground);

  // Mirror group: octahedron + garden add reflected copies here.
  const mirror = new THREE.Group();
  mirror.scale.y = -1;
  mirror.renderOrder = -2;
  scene.add(mirror);

  // Drifting ambient motes: slow deterministic float in a box around the
  // garden. Positions are a pure function of t.
  const moteRng = mulberry32(92);
  const moteCount = 220;
  const moteSeeds = [];
  for (let i = 0; i < moteCount; i += 1) {
    moteSeeds.push({
      x: (moteRng() - 0.5) * 60,
      y: 0.4 + moteRng() * 12,
      z: (moteRng() - 0.5) * 60 - 4,
      phase: moteRng() * Math.PI * 2,
      speed: 0.05 + moteRng() * 0.1,
      amp: 0.8 + moteRng() * 2.2,
    });
  }
  const motePositions = new Float32Array(moteCount * 3);
  const moteGeom = new THREE.BufferGeometry();
  moteGeom.setAttribute("position", new THREE.BufferAttribute(motePositions, 3));
  const moteMat = new THREE.PointsMaterial({
    color: 0xbfd4e8,
    size: 0.09,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.35,
    blending: THREE.AdditiveBlending,
    fog: false,
    depthWrite: false,
  });
  const motes = new THREE.Points(moteGeom, moteMat);
  scene.add(motes);

  // Fog.
  scene.fog = new THREE.FogExp2(0x101018, 0.03);

  // Camera path: one continuous journey. Keyframes at section boundaries,
  // eased with smootherstep; a slow drift wobble keeps it alive between.
  const camera = new THREE.PerspectiveCamera(46, width / height, 0.1, 900);
  const s = music.sections.map((x) => x.start_seconds);
  const end = music.totalDur;
  const camKeys = [
    { t: s[0], pos: [3.0, 1.1, 27.0], look: [0.0, 3.2, -2.0] },
    { t: s[1] * 0.6, pos: [1.5, 1.8, 26.0], look: [0.0, 4.0, -3.0] },
    { t: s[1], pos: [0.0, 4.8, 31.0], look: [0.0, 5.6, -8.0] },
    { t: (s[1] + s[2]) / 2, pos: [-4.0, 5.6, 29.0], look: [0.0, 6.0, -9.0] },
    { t: s[2], pos: [-15.0, 6.5, 15.0], look: [1.0, 8.5, -14.0] },
    { t: (s[2] + s[3]) / 2, pos: [-18.0, 7.0, 4.0], look: [2.0, 8.5, -14.0] },
    { t: s[3], pos: [-9.0, 6.0, 24.0], look: [0.0, 6.5, -9.0] },
    { t: (s[3] + s[4]) / 2, pos: [7.5, 5.2, 30.0], look: [0.0, 6.0, -8.0] },
    { t: s[4], pos: [2.0, 3.4, 22.0], look: [0.0, 7.0, -14.0] },
    { t: (s[4] + end) / 2, pos: [0.5, 2.4, 14.0], look: [0.0, 9.0, -18.0] },
    { t: end, pos: [0.0, 1.6, 9.0], look: [0.0, 10.0, -20.0] },
  ];

  function cameraAt(t) {
    let i = 0;
    while (i < camKeys.length - 1 && t >= camKeys[i + 1].t) i += 1;
    const a = camKeys[i];
    const b = camKeys[Math.min(i + 1, camKeys.length - 1)];
    const span = Math.max(1e-6, b.t - a.t);
    const u = smoothstep(0, 1, Math.max(0, Math.min(1, (t - a.t) / span)));
    const lerp = (p, q) => p + (q - p) * u;
    const pos = [0, 1, 2].map((k) => lerp(a.pos[k], b.pos[k]));
    const look = [0, 1, 2].map((k) => lerp(a.look[k], b.look[k]));
    // Slow breathing drift.
    pos[0] += Math.sin(t * 0.043) * 0.5;
    pos[1] += Math.sin(t * 0.031 + 1.7) * 0.25;
    look[0] += Math.sin(t * 0.027 + 0.6) * 0.3;
    camera.position.set(pos[0], pos[1], pos[2]);
    camera.lookAt(look[0], look[1], look[2]);
  }

  const otonalTint = new THREE.Color(1.0, 0.72, 0.38);
  const utonalTint = new THREE.Color(0.55, 0.47, 1.0);

  function update(t) {
    const mood = blendKeyframes(moods, t, 9.0);
    skyUniforms.topColor.value.setRGB(mood[0], mood[1], mood[2]);
    skyUniforms.horizonColor.value.setRGB(mood[3], mood[4], mood[5]);
    scene.fog.color.setRGB(mood[6], mood[7], mood[8]);
    scene.fog.density = mood[9];
    starMat.opacity = mood[10];

    // Harmonic tint: the active region's polarity colors the horizon.
    const slot = music.slotAt(t);
    const tint = slot.otonal === false ? utonalTint : otonalTint;
    skyUniforms.tintColor.value.copy(tint);
    skyUniforms.tintStrength.value = mood[11] * (0.35 + 0.25 * music.glowAt(t));

    // Motes drift; brightness follows the mix.
    for (let i = 0; i < moteCount; i += 1) {
      const m = moteSeeds[i];
      motePositions[i * 3] = m.x + Math.sin(t * m.speed + m.phase) * m.amp;
      motePositions[i * 3 + 1] =
        m.y + Math.sin(t * m.speed * 0.7 + m.phase * 2.1) * m.amp * 0.4;
      motePositions[i * 3 + 2] = m.z + Math.cos(t * m.speed * 0.83 + m.phase) * m.amp;
    }
    moteGeom.attributes.position.needsUpdate = true;
    moteMat.opacity = 0.14 + 0.3 * music.glowAt(t);

    cameraAt(t);
  }

  return { camera, mirror, update };
}
