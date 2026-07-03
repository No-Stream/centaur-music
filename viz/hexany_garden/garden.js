// The crystalline garden: light-flora growing from the note stream on the
// glass plane. All geometry is precomputed with birth times at init;
// renderFrame just repaints vertex colors (additive blending: black = off).
//
// - Walker record/free notes grow segments of the main crystal plant;
//   replay notes re-trace and *thicken* (brighten) the segments they echo.
// - The thumb line grows a second plant on 7-fold phyllotaxis.
// - Glints rise as spark-motes; kicks ripple the plane; hats shimmer.
// - Bar 93's cluster chord opens a radial crystal bloom whose petals fall
//   during S5 until one seed-light remains.

import * as THREE from "three";
import { mulberry32 } from "../lib/viz_data.js";

const PLANT_BASE = new THREE.Vector3(-1.5, 0, -2);
const THUMB_BASE = new THREE.Vector3(6.5, 0, 2.5);
const GOLD = new THREE.Color(1.0, 0.78, 0.42);
const VIOLET = new THREE.Color(0.62, 0.52, 1.0);
const ICE = new THREE.Color(0.72, 0.88, 1.0);

function additiveLineMaterial() {
  return new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    blending: THREE.AdditiveBlending,
    fog: false,
    depthWrite: false,
  });
}

function additivePointsMaterial(size) {
  return new THREE.PointsMaterial({
    vertexColors: true,
    size,
    sizeAttenuation: true,
    transparent: true,
    blending: THREE.AdditiveBlending,
    fog: false,
    depthWrite: false,
  });
}

// --- Garden field growth (precomputed) -------------------------------------
// Each 8-bar riff phrase grows its own crystal plant; free-walk stretches
// grow smaller tufts in 4-bar groups. Plants stand on a golden-angle spiral
// around the garden center — the piece's form plants a field. Record notes
// grow a plant's stem, replays re-trace it (boost brightness), free notes
// add twigs.
const GOLDEN_ANGLE = 2.399963229728653;

function growGardenField(music) {
  const rng = mulberry32(1357);
  const segments = []; // {ax..bz, birth, boosts:[t...], quote, stem}
  const plants = new Map(); // key -> {tip, base, segCount, recorded, replayIdx}
  let plantCount = 0;
  const barSec = music.barSeconds;
  const starts = music.ann.riff_phrase_starts;

  for (const w of music.lead) {
    const bar = Math.floor(w.t / barSec) + 1;
    let key = null;
    for (const s of starts) if (bar >= s && bar < s + 8) key = `p${s}`;
    if (key === null) key = `f${Math.floor((bar - 1) / 4)}`;

    if (!plants.has(key)) {
      const angle = plantCount * GOLDEN_ANGLE;
      const radius = 2.6 + plantCount * 0.95;
      const base = new THREE.Vector3(
        PLANT_BASE.x + Math.cos(angle) * radius,
        0,
        PLANT_BASE.z + Math.sin(angle) * radius * 0.8,
      );
      plants.set(key, {
        base,
        tip: base.clone(),
        segCount: 0,
        recorded: [],
        replayIdx: 0,
      });
      plantCount += 1;
    }
    const plant = plants.get(key);

    if (w.phase === "replay") {
      // Re-trace: boost the matching recorded segment in sequence.
      if (plant.recorded.length > 0) {
        plant.recorded[plant.replayIdx % plant.recorded.length].boosts.push(w.t);
        plant.replayIdx += 1;
      }
      continue;
    }

    // Grow a segment: young segments climb (stem), older arc outward (frond).
    const outward = Math.atan2(
      plant.base.z - PLANT_BASE.z,
      plant.base.x - PLANT_BASE.x,
    );
    const stemness = Math.max(0, 1 - plant.segCount / 10);
    const az =
      outward +
      ((w.deg / 6) * Math.PI * 2 - outward) * 0.25 +
      (rng() - 0.5) * 1.1;
    const climb = 0.75 * stemness + (0.2 + 0.06 * w.oct) * (1 - stemness);
    const len = (0.34 + Math.min(0.6, w.dur * 0.5)) * (0.8 + w.vel * 0.5);
    const dir = new THREE.Vector3(
      Math.cos(az) * (1 - climb * 0.85),
      climb,
      Math.sin(az) * (1 - climb * 0.85),
    )
      .normalize()
      .multiplyScalar(len);
    const next = plant.tip.clone().add(dir);
    // Keep each plant compact around its own base.
    const r = Math.hypot(next.x - plant.base.x, next.z - plant.base.z);
    const maxR = 2.1 + plant.segCount * 0.025;
    if (r > maxR) {
      next.x = plant.base.x + ((next.x - plant.base.x) * maxR) / r;
      next.z = plant.base.z + ((next.z - plant.base.z) * maxR) / r;
    }
    const maxH = 4.4 + (plantCount % 3) * 1.1;
    if (next.y > maxH) next.y = maxH - (next.y - maxH) * 0.6;
    if (next.y < 0.08) next.y = 0.16 - next.y;
    const seg = {
      ax: plant.tip.x,
      ay: plant.tip.y,
      az: plant.tip.z,
      bx: next.x,
      by: next.y,
      bz: next.z,
      birth: w.t,
      boosts: [],
      quote: w.quote,
      stem: plant.segCount < 10,
    };
    segments.push(seg);
    if (w.phase === "record") plant.recorded.push(seg);
    plant.segCount += 1;
    // Fronds re-anchor near the stem every so often instead of snaking on.
    if (plant.segCount % 9 === 0 && plant.recorded.length > 2) {
      const node =
        plant.recorded[Math.floor(rng() * plant.recorded.length)];
      plant.tip = new THREE.Vector3(node.bx, node.by, node.bz);
    } else {
      plant.tip = next;
    }
  }
  return segments;
}

export function createGarden(scene, mirrorGroup, music) {
  // --- Garden field ---------------------------------------------------------
  const segments = growGardenField(music);
  const segCount = segments.length;
  const linePositions = new Float32Array(segCount * 6);
  const lineColors = new Float32Array(segCount * 6);
  for (let i = 0; i < segCount; i += 1) {
    const s = segments[i];
    linePositions.set([s.ax, s.ay, s.az, s.bx, s.by, s.bz], i * 6);
  }
  const lineGeom = new THREE.BufferGeometry();
  lineGeom.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
  lineGeom.setAttribute("color", new THREE.BufferAttribute(lineColors, 3));
  const plant = new THREE.LineSegments(lineGeom, additiveLineMaterial());
  scene.add(plant);
  mirrorGroup.add(new THREE.LineSegments(lineGeom, additiveLineMaterial()));

  // Faceted crystal shards: one elongated octahedron per segment, oriented
  // along it. Instance colors repaint each frame with the same brightness
  // field as the lines, so the flora reads as glassy volume, not wire.
  const shardGeom = new THREE.OctahedronGeometry(1, 0);
  const shardMat = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    blending: THREE.AdditiveBlending,
    fog: false,
    depthWrite: false,
  });
  const shards = new THREE.InstancedMesh(shardGeom, shardMat, segCount);
  const shardDummy = new THREE.Object3D();
  const yAxis = new THREE.Vector3(0, 1, 0);
  const segDir = new THREE.Vector3();
  for (let i = 0; i < segCount; i += 1) {
    const s = segments[i];
    shardDummy.position.set((s.ax + s.bx) / 2, (s.ay + s.by) / 2, (s.az + s.bz) / 2);
    segDir.set(s.bx - s.ax, s.by - s.ay, s.bz - s.az);
    const segLen = Math.max(0.05, segDir.length());
    shardDummy.quaternion.setFromUnitVectors(yAxis, segDir.normalize());
    const girth = s.stem ? 0.085 : 0.055;
    shardDummy.scale.set(girth, segLen * 0.62, girth);
    shardDummy.updateMatrix();
    shards.setMatrixAt(i, shardDummy.matrix);
    shards.setColorAt(i, new THREE.Color(0, 0, 0));
  }
  shards.instanceMatrix.needsUpdate = true;
  scene.add(shards);

  // Node sparkles at segment tips.
  const nodeColors = new Float32Array(segCount * 3);
  const nodePositions = new Float32Array(segCount * 3);
  for (let i = 0; i < segCount; i += 1) {
    const s = segments[i];
    nodePositions.set([s.bx, s.by, s.bz], i * 3);
  }
  const nodeGeom = new THREE.BufferGeometry();
  nodeGeom.setAttribute("position", new THREE.BufferAttribute(nodePositions, 3));
  nodeGeom.setAttribute("color", new THREE.BufferAttribute(nodeColors, 3));
  const nodes = new THREE.Points(nodeGeom, additivePointsMaterial(0.16));
  scene.add(nodes);

  // --- Thumb plant: 7-fold phyllotaxis shards -----------------------------
  const thumbRng = mulberry32(753);
  const thumbShards = music.thumb.map((ev, i) => {
    const angle = (ev.tags.step / 7) * Math.PI * 2 + i * 0.13;
    const radius = 0.5 + (i / Math.max(1, music.thumb.length)) * 2.6;
    const h = 0.25 + (i / Math.max(1, music.thumb.length)) * 4.2 + thumbRng() * 0.3;
    const x = THUMB_BASE.x + Math.cos(angle) * radius;
    const z = THUMB_BASE.z + Math.sin(angle) * radius;
    return { x, y: h, z, birth: ev.t, vel: ev.vel, rank: ev.tags.rank };
  });
  const thumbCount = thumbShards.length;
  const thumbLinePos = new Float32Array(thumbCount * 6);
  const thumbLineCol = new Float32Array(thumbCount * 6);
  for (let i = 0; i < thumbCount; i += 1) {
    const sh = thumbShards[i];
    // Short stem from the spiral column toward the shard.
    thumbLinePos.set(
      [THUMB_BASE.x, Math.max(0, sh.y - 0.8), THUMB_BASE.z, sh.x, sh.y, sh.z],
      i * 6,
    );
  }
  const thumbGeom = new THREE.BufferGeometry();
  thumbGeom.setAttribute("position", new THREE.BufferAttribute(thumbLinePos, 3));
  thumbGeom.setAttribute("color", new THREE.BufferAttribute(thumbLineCol, 3));
  const thumbPlant = new THREE.LineSegments(thumbGeom, additiveLineMaterial());
  scene.add(thumbPlant);
  mirrorGroup.add(new THREE.LineSegments(thumbGeom, additiveLineMaterial()));

  // --- Glints: rising spark motes ------------------------------------------
  const glintRng = mulberry32(93);
  const glintSeeds = music.glint.map((ev) => ({
    t: ev.t,
    vel: ev.vel,
    x: PLANT_BASE.x + (glintRng() - 0.5) * 8,
    z: PLANT_BASE.z + (glintRng() - 0.5) * 8,
    y0: 1.5 + glintRng() * 3.5,
    drift: (glintRng() - 0.5) * 0.6,
  }));
  const glintPos = new Float32Array(glintSeeds.length * 3);
  const glintCol = new Float32Array(glintSeeds.length * 3);
  const glintGeom = new THREE.BufferGeometry();
  glintGeom.setAttribute("position", new THREE.BufferAttribute(glintPos, 3));
  glintGeom.setAttribute("color", new THREE.BufferAttribute(glintCol, 3));
  const glints = new THREE.Points(glintGeom, additivePointsMaterial(0.22));
  scene.add(glints);

  // --- Hat/shaker shimmer ---------------------------------------------------
  const shimmerRng = mulberry32(4242);
  const shimmerEvents = music.hats.concat(music.shaker).sort((a, b) => a.t - b.t);
  const shimmerSeeds = shimmerEvents.map((ev) => ({
    t: ev.t,
    vel: ev.vel,
    x: (shimmerRng() - 0.5) * 22,
    y: 0.1 + shimmerRng() * 1.4,
    z: (shimmerRng() - 0.5) * 22 - 2,
  }));
  const shimPos = new Float32Array(shimmerSeeds.length * 3);
  const shimCol = new Float32Array(shimmerSeeds.length * 3);
  for (let i = 0; i < shimmerSeeds.length; i += 1) {
    const s = shimmerSeeds[i];
    shimPos.set([s.x, s.y, s.z], i * 3);
  }
  const shimGeom = new THREE.BufferGeometry();
  shimGeom.setAttribute("position", new THREE.BufferAttribute(shimPos, 3));
  shimGeom.setAttribute("color", new THREE.BufferAttribute(shimCol, 3));
  const shimmer = new THREE.Points(shimGeom, additivePointsMaterial(0.1));
  scene.add(shimmer);

  // --- Kick ripples ----------------------------------------------------------
  const RIPPLE_POOL = 6;
  const ripples = [];
  for (let i = 0; i < RIPPLE_POOL; i += 1) {
    const mat = new THREE.MeshBasicMaterial({
      color: 0x8899cc,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
    fog: false,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(new THREE.RingGeometry(0.96, 1.0, 48), mat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.set(0, 0.02, -2);
    scene.add(mesh);
    ripples.push({ mesh, mat });
  }

  // --- The bar-93 bloom -------------------------------------------------------
  const bloomT = music.bloomTime;
  // Crown position: mean of late segment tips.
  const crown = new THREE.Vector3();
  const lateSegs = segments.filter((s) => s.birth < bloomT).slice(-40);
  let crownTop = 0;
  for (const s of lateSegs) crownTop = Math.max(crownTop, s.by);
  crown.set(PLANT_BASE.x, Math.max(4.6, crownTop * 0.9), PLANT_BASE.z);
  const PETALS = 18;
  const petalRng = mulberry32(93_93);
  const petals = [];
  const petalPos = new Float32Array(PETALS * 6);
  const petalCol = new Float32Array(PETALS * 6);
  for (let i = 0; i < PETALS; i += 1) {
    const angle = (i / PETALS) * Math.PI * 2;
    const spread = 3.6 + petalRng() * 1.4;
    const tipY = crown.y + 2.2 + petalRng() * 1.4;
    petals.push({
      birth: bloomT + i * 0.14,
      x0: crown.x,
      y0: crown.y,
      z0: crown.z,
      x1: crown.x + Math.cos(angle) * spread,
      y1: tipY,
      z1: crown.z + Math.sin(angle) * spread,
      fallStart: music.s5Time + 6 + petalRng() * 26,
    });
  }
  const petalGeom = new THREE.BufferGeometry();
  petalGeom.setAttribute("position", new THREE.BufferAttribute(petalPos, 3));
  petalGeom.setAttribute("color", new THREE.BufferAttribute(petalCol, 3));
  const petalLines = new THREE.LineSegments(petalGeom, additiveLineMaterial());
  scene.add(petalLines);

  // Bloom crown flash: a burst of light at the cluster-chord moment.
  const bloomFlashMat = new THREE.MeshBasicMaterial({
    color: 0xffe9c0,
    transparent: true,
    opacity: 0,
    blending: THREE.AdditiveBlending,
    fog: false,
    depthWrite: false,
  });
  const bloomFlash = new THREE.Mesh(
    new THREE.SphereGeometry(0.6, 12, 8),
    bloomFlashMat,
  );
  bloomFlash.position.copy(crown);
  scene.add(bloomFlash);

  // The seed: a single point of light that remains at the end.
  const seedMat = new THREE.MeshBasicMaterial({
    color: 0xfff6de,
    transparent: true,
    opacity: 0,
    blending: THREE.AdditiveBlending,
    fog: false,
    depthWrite: false,
  });
  const seed = new THREE.Mesh(new THREE.SphereGeometry(0.12, 10, 8), seedMat);
  seed.position.set(PLANT_BASE.x, 0.35, PLANT_BASE.z);
  scene.add(seed);

  const colorA = new THREE.Color();
  const colorB = new THREE.Color();

  function segmentBrightness(s, t, dimS5) {
    if (s.birth > t) return 0;
    const age = t - s.birth;
    const grow = Math.min(1, age / 0.5);
    let heat = 0;
    for (const bt of s.boosts) {
      if (bt > t) break;
      heat += Math.exp(-(t - bt) / 6) * 0.55;
    }
    const flash = Math.exp(-age / 0.6) * 1.4;
    const base = (s.stem ? 0.36 : 0.24) + (s.quote ? 0.1 : 0);
    return (base + flash + Math.min(1.4, heat)) * grow * dimS5;
  }

  function update(t) {
    const slot = music.slotAt(t);
    const polarity = slot.otonal === false ? VIOLET : GOLD;
    // S5 dimming: garden fades toward darkness, envelope-following.
    const dimS5 =
      t < music.s5Time
        ? 1
        : Math.max(0.06, 0.25 + 0.75 * music.glowAt(t)) *
          Math.max(0.05, 1 - (t - music.s5Time) / (music.totalDur - music.s5Time));

    // Garden field: lines, shards, and tip sparkles share one brightness field.
    for (let i = 0; i < segCount; i += 1) {
      const b = segmentBrightness(segments[i], t, dimS5);
      colorA.copy(ICE).lerp(polarity, 0.55).multiplyScalar(b);
      lineColors.set([colorA.r, colorA.g, colorA.b, colorA.r, colorA.g, colorA.b], i * 6);
      shards.setColorAt(i, colorB.copy(colorA).multiplyScalar(0.85));
      const nb = b * 0.8;
      nodeColors.set([colorA.r * nb, colorA.g * nb, colorA.b * nb], i * 3);
    }
    lineGeom.attributes.color.needsUpdate = true;
    shards.instanceColor.needsUpdate = true;
    nodeGeom.attributes.color.needsUpdate = true;

    // Thumb plant: shards glow on their 7-cycle hits, cool ice tone.
    for (let i = 0; i < thumbCount; i += 1) {
      const sh = thumbShards[i];
      let b = 0;
      if (sh.birth <= t) {
        const age = t - sh.birth;
        b = (0.12 + Math.exp(-age / 0.5) * 0.85 * sh.vel) * dimS5;
        if (sh.rank === 0) b *= 1.3;
      }
      colorA.copy(ICE).multiplyScalar(b);
      thumbLineCol.set(
        [colorA.r * 0.4, colorA.g * 0.4, colorA.b * 0.4, colorA.r, colorA.g, colorA.b],
        i * 6,
      );
    }
    thumbGeom.attributes.color.needsUpdate = true;

    // Glints rise and fade over ~3.5 s.
    for (let i = 0; i < glintSeeds.length; i += 1) {
      const g = glintSeeds[i];
      const age = t - g.t;
      let b = 0;
      let y = g.y0;
      if (age >= 0 && age < 3.5) {
        b = Math.exp(-age / 1.2) * g.vel * 1.6;
        y = g.y0 + age * 0.9;
      }
      glintPos.set([g.x + Math.sin(age * 1.7) * g.drift, y, g.z], i * 3);
      glintCol.set([1.0 * b, 0.85 * b, 0.55 * b], i * 3);
    }
    glintGeom.attributes.position.needsUpdate = true;
    glintGeom.attributes.color.needsUpdate = true;

    // Shimmer: brief twinkles.
    for (let i = 0; i < shimmerSeeds.length; i += 1) {
      const s = shimmerSeeds[i];
      const age = t - s.t;
      const b = age >= 0 && age < 0.3 ? (1 - age / 0.3) * s.vel * 0.5 * dimS5 : 0;
      shimCol.set([0.7 * b, 0.8 * b, 1.0 * b], i * 3);
    }
    shimGeom.attributes.color.needsUpdate = true;

    // Kick ripples.
    let ri = 0;
    for (let i = music.kicks.length - 1; i >= 0 && ri < RIPPLE_POOL; i -= 1) {
      const k = music.kicks[i];
      const age = t - k.t;
      if (age < 0) continue;
      if (age > 1.6) break;
      const u = age / 1.6;
      const r = ripples[ri];
      r.mesh.scale.setScalar(0.5 + u * 9);
      r.mat.opacity = (1 - u) * (1 - u) * 0.28 * k.vel;
      ri += 1;
    }
    for (; ri < RIPPLE_POOL; ri += 1) ripples[ri].mat.opacity = 0;

    // Bloom petals; in S5 they fall and fade.
    for (let i = 0; i < PETALS; i += 1) {
      const p = petals[i];
      let b = 0;
      let dy = 0;
      let fade = 1;
      if (p.birth <= t) {
        const age = t - p.birth;
        const open = Math.min(1, age / 1.8);
        b = (0.35 + Math.exp(-age / 3) * 0.9) * open;
        if (t > p.fallStart) {
          const fallAge = t - p.fallStart;
          dy = -fallAge * fallAge * 0.35;
          fade = Math.max(0, 1 - fallAge / 7);
        }
        b *= fade * (t >= music.s5Time ? Math.max(0.1, dimS5 * 1.6) : 1);
        const openU = Math.min(1, age / 1.8);
        const x1 = p.x0 + (p.x1 - p.x0) * openU;
        const y1 = Math.max(0.05, p.y0 + (p.y1 - p.y0) * openU + dy);
        const z1 = p.z0 + (p.z1 - p.z0) * openU;
        petalPos.set([p.x0, Math.max(0.05, p.y0 + dy * 0.6), p.z0, x1, y1, z1], i * 6);
      }
      colorA.copy(GOLD).lerp(ICE, 0.25).multiplyScalar(b);
      petalCol.set([colorA.r, colorA.g, colorA.b, colorA.r, colorA.g, colorA.b], i * 6);
    }
    petalGeom.attributes.position.needsUpdate = true;
    petalGeom.attributes.color.needsUpdate = true;

    // Bloom flash: swells over the first beats of the cluster, then settles.
    const bloomAge = t - bloomT;
    if (bloomAge >= 0 && bloomAge < 6) {
      bloomFlashMat.opacity = Math.min(1, bloomAge / 0.8) * Math.exp(-bloomAge / 2.2) * 0.45;
      bloomFlash.scale.setScalar(1 + bloomAge * 1.0);
    } else {
      bloomFlashMat.opacity = 0;
    }

    // The seed glows once the garden has gone dark.
    if (t > music.dyadTime) {
      const u = Math.min(1, (t - music.dyadTime) / 8);
      seedMat.opacity = u * (0.35 + 0.5 * music.glowAt(t));
    } else {
      seedMat.opacity = 0;
    }
  }

  return { update };
}
