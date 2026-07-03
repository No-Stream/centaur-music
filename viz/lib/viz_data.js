// Small helpers for loading viz.json data and deterministic scene randomness.
// Kept dependency-free so scenes can import it directly as an ES module.

/**
 * Fetch and parse a viz.json data file.
 * @param {string} url
 * @returns {Promise<any>}
 */
export async function loadVizData(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`loadVizData: failed to fetch ${url} (status ${response.status})`);
  }
  return response.json();
}

/**
 * Mulberry32 seeded PRNG. Returns a function that yields floats in [0, 1).
 * Deterministic for a given 32-bit integer seed.
 * @param {number} seed
 * @returns {() => number}
 */
export function mulberry32(seed) {
  let state = seed >>> 0;
  return function next() {
    state |= 0;
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Binary-search-backed lookup of notes overlapping [t0, t1) from a list of
 * notes sorted ascending by start_seconds.
 * @param {Array<{start_seconds: number}>} notes
 * @param {number} t0
 * @param {number} t1
 * @returns {Array<any>}
 */
export function notesInRange(notes, t0, t1) {
  if (notes.length === 0) {
    return [];
  }
  let low = 0;
  let high = notes.length;
  while (low < high) {
    const mid = (low + high) >>> 1;
    if (notes[mid].start_seconds < t0) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  const result = [];
  for (let i = low; i < notes.length && notes[i].start_seconds < t1; i += 1) {
    result.push(notes[i]);
  }
  return result;
}

/**
 * Linear-interpolate an RMS/level value from an envelope array of
 * {time_seconds, value} points at time t. Clamps to the first/last point
 * outside the envelope's range.
 * @param {Array<{time_seconds: number, value: number}>} envelope
 * @param {number} t
 * @returns {number}
 */
export function envelopeAt(envelope, t) {
  if (envelope.length === 0) {
    return 0;
  }
  if (t <= envelope[0].time_seconds) {
    return envelope[0].value;
  }
  const last = envelope[envelope.length - 1];
  if (t >= last.time_seconds) {
    return last.value;
  }
  let low = 0;
  let high = envelope.length - 1;
  while (low < high - 1) {
    const mid = (low + high) >>> 1;
    if (envelope[mid].time_seconds <= t) {
      low = mid;
    } else {
      high = mid;
    }
  }
  const a = envelope[low];
  const b = envelope[high];
  const span = b.time_seconds - a.time_seconds;
  const frac = span === 0 ? 0 : (t - a.time_seconds) / span;
  return a.value + (b.value - a.value) * frac;
}
