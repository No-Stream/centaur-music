// Musical data layer: turns hexany_garden.viz.json into time-indexed state.
// Everything here is precomputed at init so renderFrame stays a pure f(t).

export function prepareMusic(data) {
  const notes = data.notes;
  const ann = data.annotations;

  const byKind = {};
  const byVoice = {};
  for (const note of notes) {
    const kind = note.semantics ? note.semantics.kind : note.voice_name;
    (byKind[kind] = byKind[kind] || []).push(note);
    (byVoice[note.voice_name] = byVoice[note.voice_name] || []).push(note);
  }

  // Chord slots: [{start_seconds, end_seconds, degrees, otonal, name}]
  const slots = ann.chord_slots;

  // Mix RMS envelope lookup (linear index into the hop grid).
  const env = data.envelope;
  const rmsAt = (t) => {
    const i = Math.max(
      0,
      Math.min(env.rms.length - 1, Math.floor(t / env.hop_seconds)),
    );
    return env.rms[i];
  };
  // Smoothed glow drive: short attack window average, normalized to ~[0,1].
  let rmsPeak = 1e-6;
  for (const v of env.rms) rmsPeak = Math.max(rmsPeak, v);
  const glowAt = (t) => {
    const a = rmsAt(t);
    const b = rmsAt(t - 0.05);
    return Math.min(1, (0.5 * (a + b)) / rmsPeak);
  };

  const slotAt = (t) => {
    let active = slots[0];
    for (const slot of slots) {
      if (t >= slot.start_seconds) active = slot;
      else break;
    }
    return active;
  };

  const sections = ann.sections;
  const sectionIndexAt = (t) => {
    let idx = 0;
    for (let i = 0; i < sections.length; i += 1) {
      if (t >= sections[i].start_seconds) idx = i;
    }
    return idx;
  };

  // Walker events, annotated for octahedron + garden use.
  const walker = (byKind.walker || []).map((n) => ({
    t: n.start_seconds,
    dur: n.duration_seconds,
    vel: n.velocity,
    deg: n.semantics.tags.deg,
    oct: n.semantics.tags.oct,
    phase: n.semantics.tags.phase,
    leap: n.semantics.tags.leap,
    grid: n.semantics.tags.grid,
    quote: Boolean(n.semantics.tags.quote),
    dyad: Boolean(n.semantics.tags.dyad),
  }));
  // Edge traversals: consecutive non-dyad walker notes with an edge/polar leap.
  const lead = walker.filter((w) => !w.dyad);
  const traversals = [];
  for (let i = 1; i < lead.length; i += 1) {
    const prev = lead[i - 1];
    const cur = lead[i];
    if (cur.leap !== "none" && cur.deg !== prev.deg) {
      traversals.push({
        t: cur.t,
        from: prev.deg,
        to: cur.deg,
        polar: cur.leap === "polar",
        phase: cur.phase,
        vel: cur.vel,
      });
    }
  }

  const simpleEvents = (kind) =>
    (byKind[kind] || []).map((n) => ({
      t: n.start_seconds,
      dur: n.duration_seconds,
      vel: n.velocity,
      deg: n.semantics ? n.semantics.tags.deg : null,
      oct: n.semantics ? n.semantics.tags.oct : null,
      tags: n.semantics ? n.semantics.tags : {},
    }));

  const percEvents = (voice) =>
    (byVoice[voice] || []).map((n) => ({
      t: n.start_seconds,
      vel: n.velocity,
    }));

  const time = ann.time;
  return {
    ann,
    totalDur: data.total_duration_seconds,
    barSeconds: time.bar_seconds,
    walker,
    lead,
    traversals,
    thumb: simpleEvents("thumb"),
    motif: simpleEvents("motif"),
    glint: simpleEvents("glint"),
    bass: simpleEvents("bass"),
    pad: simpleEvents("pad"),
    kicks: percEvents("kick"),
    hats: percEvents("hat_c").concat(percEvents("hat_o")),
    shaker: percEvents("shaker"),
    slotAt,
    sectionIndexAt,
    sections,
    rmsAt,
    glowAt,
    bloomTime: ann.bloom.start_seconds,
    s5Time: sections[4].start_seconds,
    // Bar 121: the closing 4:7 dyad slot (degrees 0 and 5).
    dyadTime: ann.chord_slots[ann.chord_slots.length - 1].start_seconds,
  };
}

// Smootherstep between section moods; used by world + post.
export function smoothstep(edge0, edge1, x) {
  const u = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return u * u * (3 - 2 * u);
}

// Blend an array of {t, ...values} keyframes at time t. Values must be
// numeric arrays of equal length; returns the interpolated array.
export function blendKeyframes(keyframes, t, easeSeconds) {
  let prev = keyframes[0];
  let next = keyframes[keyframes.length - 1];
  for (let i = 0; i < keyframes.length; i += 1) {
    if (t >= keyframes[i].t) {
      prev = keyframes[i];
      next = keyframes[Math.min(i + 1, keyframes.length - 1)];
    }
  }
  if (prev === next) return prev.v.slice();
  const mix = smoothstep(next.t - easeSeconds, next.t, t);
  return prev.v.map((a, i) => a + (next.v[i] - a) * mix);
}
