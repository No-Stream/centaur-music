// The hexany octahedron: six degree-vertices, twelve factor-edges, eight
// triad-faces. It hangs above the horizon like a slow instrument and plays
// the piece's harmony: vertices flash with the walker, edge pulses travel
// factor-adjacent moves, faces glow with the active region (otonal gold /
// utonal violet), and polar leaps fire beams through the solid's center.

import * as THREE from "three";

const CENTER = new THREE.Vector3(0, 12.0, -13);
const RADIUS = 5.5;

// Degree -> vertex direction. Polar pairs (0,3) (1,2) (4,5) are antipodes.
const VERTEX_DIRS = [
  [1, 0, 0],
  [0, 1, 0],
  [0, -1, 0],
  [-1, 0, 0],
  [0, 0, 1],
  [0, 0, -1],
];

const OTONAL_COLOR = new THREE.Color(1.0, 0.72, 0.35);
const UTONAL_COLOR = new THREE.Color(0.52, 0.44, 1.0);
const WIRE_COLOR = new THREE.Color(0.62, 0.72, 0.86);

function edgeKey(a, b) {
  return a < b ? `${a}-${b}` : `${b}-${a}`;
}

export function createOctahedron(scene, mirrorGroup, music) {
  const group = new THREE.Group();
  group.position.copy(CENTER);
  scene.add(group);
  const mirrored = new THREE.Group();
  mirrored.position.set(CENTER.x, CENTER.y, CENTER.z);
  mirrorGroup.add(mirrored);

  const vertexPos = VERTEX_DIRS.map(
    (d) => new THREE.Vector3(d[0], d[1], d[2]).multiplyScalar(RADIUS),
  );

  // --- Edges (12: every non-antipodal vertex pair) --------------------
  const edges = [];
  const edgeIndex = new Map();
  for (let a = 0; a < 6; a += 1) {
    for (let b = a + 1; b < 6; b += 1) {
      const antipodal = VERTEX_DIRS[a].every((v, i) => v === -VERTEX_DIRS[b][i]);
      if (antipodal) continue;
      const geom = new THREE.BufferGeometry().setFromPoints([
        vertexPos[a],
        vertexPos[b],
      ]);
      const mat = new THREE.LineBasicMaterial({
        color: WIRE_COLOR.clone(),
        transparent: true,
        opacity: 0.3,
        blending: THREE.AdditiveBlending,
    fog: false,
        depthWrite: false,
      });
      const line = new THREE.Line(geom, mat);
      group.add(line);
      mirrored.add(new THREE.Line(geom, mat));
      const idx = edges.length;
      edges.push({ a, b, mat, events: [] });
      edgeIndex.set(edgeKey(a, b), idx);
    }
  }
  for (const trav of music.traversals) {
    if (trav.polar) continue;
    const idx = edgeIndex.get(edgeKey(trav.from, trav.to));
    if (idx !== undefined) edges[idx].events.push(trav);
  }

  // --- Faces (8 triads) ------------------------------------------------
  // Region degrees from annotations; each triad is one degree per polar
  // pair, i.e. one octahedron face. The closing dyad (2 degrees) has none.
  const faces = [];
  for (const region of music.ann.regions) {
    if (region.degrees.length !== 3) continue;
    const [d0, d1, d2] = region.degrees;
    const geom = new THREE.BufferGeometry().setFromPoints([
      vertexPos[d0],
      vertexPos[d1],
      vertexPos[d2],
    ]);
    geom.computeVertexNormals();
    const mat = new THREE.MeshBasicMaterial({
      color: (region.otonal ? OTONAL_COLOR : UTONAL_COLOR).clone(),
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
    fog: false,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(geom, mat);
    group.add(mesh);
    mirrored.add(new THREE.Mesh(geom, mat));
    faces.push({ name: region.name, mat });
  }

  // --- Vertices ----------------------------------------------------------
  const vertexHits = Array.from({ length: 6 }, () => []);
  for (const w of music.lead) vertexHits[w.deg].push({ t: w.t, vel: w.vel, kind: "walker" });
  for (const m of music.motif) {
    if (m.deg !== null) vertexHits[m.deg].push({ t: m.t, vel: m.vel, kind: "motif" });
  }
  for (const hits of vertexHits) hits.sort((x, y) => x.t - y.t);

  const vertexMeshes = [];
  const vertexGeom = new THREE.SphereGeometry(0.22, 12, 8);
  for (let d = 0; d < 6; d += 1) {
    const mat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
    fog: false,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(vertexGeom, mat);
    mesh.position.copy(vertexPos[d]);
    group.add(mesh);
    const mirroredMesh = new THREE.Mesh(vertexGeom, mat);
    mirroredMesh.position.copy(vertexPos[d]);
    mirrored.add(mirroredMesh);
    vertexMeshes.push({ mesh, mirroredMesh, mat });
  }

  // --- Edge pulses (traveling dots) ---------------------------------------
  const PULSE_POOL = 8;
  const pulseGeom = new THREE.SphereGeometry(0.13, 8, 6);
  const pulses = [];
  for (let i = 0; i < PULSE_POOL; i += 1) {
    const mat = new THREE.MeshBasicMaterial({
      color: 0xfff3d0,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
    fog: false,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(pulseGeom, mat);
    group.add(mesh);
    pulses.push({ mesh, mat });
  }

  // --- Polar beams --------------------------------------------------------
  const polarEvents = music.traversals.filter((tr) => tr.polar);
  const beams = polarEvents.map((ev) => {
    const geom = new THREE.CylinderGeometry(0.06, 0.06, RADIUS * 2, 8, 1, true);
    const mat = new THREE.MeshBasicMaterial({
      color: 0xe8dcff,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
    fog: false,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(geom, mat);
    // Orient along from->to axis (through the center).
    const dir = vertexPos[ev.to].clone().sub(vertexPos[ev.from]).normalize();
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    group.add(mesh);
    return { ev, mat };
  });
  const coreFlash = new THREE.Mesh(
    new THREE.SphereGeometry(0.5, 12, 8),
    new THREE.MeshBasicMaterial({
      color: 0xf4ecff,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
    fog: false,
      depthWrite: false,
    }),
  );
  group.add(coreFlash);

  // The final 4:7 edge (degrees 0-5): the last thing left glowing.
  const finalEdgeIdx = edgeIndex.get(edgeKey(0, 5));

  function recentEnergy(events, t, tau, horizon) {
    let e = 0;
    for (const ev of events) {
      if (ev.t > t) break;
      const age = t - ev.t;
      if (age < horizon) e += Math.exp(-age / tau) * (ev.vel || 0.7);
    }
    return e;
  }

  function update(t) {
    // Section presence: ghostly in the dew, full once the walker enters,
    // dimming into the night (additive materials ignore scene fog, so the
    // atmosphere is applied by hand here).
    const s = music.sections.map((x) => x.start_seconds);
    let presence;
    if (t < s[1]) {
      presence = 0.3 + 0.7 * Math.max(0, (t - (s[1] - 10)) / 10);
    } else if (t < music.s5Time) {
      presence = 1.0;
    } else {
      presence = 1.0 - 0.25 * Math.min(1, (t - music.s5Time) / 30);
    }

    // Slow rotation, shared with the reflection.
    const rotY = t * 0.045;
    const rotX = Math.sin(t * 0.021) * 0.12;
    group.rotation.set(rotX, rotY, 0);
    mirrored.rotation.set(rotX, rotY, 0);
    const bob = Math.sin(t * 0.09) * 0.35;
    group.position.y = CENTER.y + bob;
    mirrored.position.y = CENTER.y + bob;

    const slot = music.slotAt(t);
    const slotFade =
      Math.min(1, (t - slot.start_seconds) / 1.2) *
      (1 - 0.4 * Math.max(0, 1 - (slot.end_seconds - t) / 1.2));

    // Faces: only the active region's face glows. Pure function of t —
    // stateful smoothing would break determinism across capture shards.
    for (const face of faces) {
      const active = face.name === slot.name;
      face.mat.opacity = active
        ? (0.06 + 0.14 * music.glowAt(t)) * presence * slotFade
        : 0;
    }

    // Vertices: flash on walker/motif hits, faint base glow otherwise.
    for (let d = 0; d < 6; d += 1) {
      const flash = recentEnergy(vertexHits[d], t, 0.35, 3.0);
      const scale = 1 + Math.min(2.2, flash * 1.6);
      vertexMeshes[d].mesh.scale.setScalar(scale);
      vertexMeshes[d].mirroredMesh.scale.setScalar(scale);
      vertexMeshes[d].mat.opacity = (0.3 + Math.min(0.7, flash * 0.5)) * presence;
    }

    // Edges: heat from traversals; replays keep paths warm.
    for (const edge of edges) {
      let heat = 0;
      for (const ev of edge.events) {
        if (ev.t > t) break;
        const age = t - ev.t;
        const tau = ev.phase === "replay" ? 7.0 : 3.0;
        if (age < 30) heat += Math.exp(-age / tau) * 0.5;
      }
      edge.mat.opacity = Math.min(0.85, 0.16 + heat * 0.4) * presence;
    }

    // Traveling pulses: most recent traversals animate along their edge.
    let pulseSlot = 0;
    for (let i = music.traversals.length - 1; i >= 0 && pulseSlot < PULSE_POOL; i -= 1) {
      const tr = music.traversals[i];
      if (tr.t > t) continue;
      const age = t - tr.t;
      if (age > 0.4) break;
      const u = age / 0.4;
      const p = pulses[pulseSlot];
      p.mesh.position.lerpVectors(vertexPos[tr.from], vertexPos[tr.to], u);
      p.mat.opacity = (1 - u) * 0.9;
      pulseSlot += 1;
    }
    for (; pulseSlot < PULSE_POOL; pulseSlot += 1) pulses[pulseSlot].mat.opacity = 0;

    // Polar beams: bright flash through the center, slow decay.
    let core = 0;
    for (const beam of beams) {
      const age = t - beam.ev.t;
      if (age < 0 || age > 4) {
        beam.mat.opacity = 0;
        continue;
      }
      const a = Math.exp(-age / 1.4) * 0.85;
      beam.mat.opacity = a;
      core = Math.max(core, a);
    }
    coreFlash.material.opacity = core * 0.8;
    coreFlash.scale.setScalar(1 + core * 2.5);

    // The ending: after the dyad begins the rest of the solid recedes over
    // ~10 s until only the 0-5 edge (the bare 4:7) is left, riding the mix
    // envelope down to darkness.
    if (t >= music.dyadTime && finalEdgeIdx !== undefined) {
      const recede = 1 - 0.94 * Math.min(1, (t - music.dyadTime) / 10);
      for (const edge of edges) {
        if (edge !== edges[finalEdgeIdx]) edge.mat.opacity *= recede;
      }
      for (let d = 0; d < 6; d += 1) {
        if (d !== 0 && d !== 5) vertexMeshes[d].mat.opacity *= recede;
      }
      const dyadGlow = 0.2 + 0.8 * music.glowAt(t);
      edges[finalEdgeIdx].mat.opacity = Math.max(
        edges[finalEdgeIdx].mat.opacity,
        0.8 * dyadGlow,
      );
    }
  }

  return { update, center: CENTER };
}
