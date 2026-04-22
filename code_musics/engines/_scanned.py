"""Scanned synthesis (Verplank): mass-spring ring traversal as the oscillator.

A ring of ``n_nodes`` point masses coupled by springs evolves mechanically
at its own sub-audio rate; the audio waveform is read by scanning around
the ring at ``freq`` Hz with linear node interpolation.  The shape
breathes and warps on its own.  One mechanical step per audio sample —
the choice that gives the characteristic Verplank "breathing" character.
``motion`` scales that mechanical step.

Stability is first-class: symplectic Euler + ``sqrt(k)``-bounded ``dt``
keep the ring numerically safe across any ``0..1`` knob setting.
"""

from __future__ import annotations

import math

import numba
import numpy as np


@numba.njit(cache=True)
def _scan_ring_loop(
    out: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    phase_increment_per_sample: np.ndarray,
    neighbor_stiffness: float,
    restore_stiffness: float,
    damping: float,
    mechanical_dt_per_sample: float,
) -> None:
    """Co-evolve the mass-spring ring and scan its displacement at audio rate.

    Symplectic Euler (updates velocities then positions) preserves energy
    for weakly-damped systems, keeping long notes stable where plain
    explicit Euler would drift at musical stiffness settings.
    """
    n_nodes = positions.shape[0]
    n_samples = out.shape[0]
    inv_n_nodes = 1.0 / float(n_nodes)
    scan_phase = 0.0

    for i in range(n_samples):
        # Force = neighbor springs (ring) + rest-position restore - damping.
        for k in range(n_nodes):
            left = positions[(k - 1) % n_nodes]
            right = positions[(k + 1) % n_nodes]
            spring_force = neighbor_stiffness * (left + right - 2.0 * positions[k])
            restore_force = -restore_stiffness * positions[k]
            damping_force = -damping * velocities[k]
            accel = spring_force + restore_force + damping_force
            velocities[k] += accel * mechanical_dt_per_sample
        for k in range(n_nodes):
            positions[k] += velocities[k] * mechanical_dt_per_sample

        phase = scan_phase - math.floor(scan_phase)
        node_pos = phase * n_nodes
        node_idx = int(node_pos)
        frac = node_pos - float(node_idx)
        next_idx = (node_idx + 1) % n_nodes
        out[i] = (1.0 - frac) * positions[node_idx] + frac * positions[next_idx]

        scan_phase += phase_increment_per_sample[i] * inv_n_nodes * n_nodes
        if scan_phase > 1e9:
            scan_phase -= math.floor(scan_phase)


def _excite_ring(
    *,
    n_nodes: int,
    position: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build the initial displacement distribution around the ring.

    ``position`` in ``[0, 1]`` controls how concentrated the excitation is.
    0 = fully broadband (independent random displacement per node) —
    smoother, pad-like shapes.  1 = fully concentrated (spike on one
    node) — sharp, harmonic-rich shapes.  Values between interpolate
    between those two via a Gaussian window whose width contracts as
    position increases.
    """
    position = float(max(0.0, min(1.0, position)))
    broadband = rng.standard_normal(n_nodes).astype(np.float64)

    if position <= 0.0:
        shaped = broadband
    else:
        # Gaussian window around node 0: width=n_nodes at position=0
        # (fully broad), width=0.5 at position=1 (single mass).
        width = max(0.5, float(n_nodes) * (1.0 - position))
        idx = np.arange(n_nodes, dtype=np.float64)
        centered = np.minimum(idx, float(n_nodes) - idx)
        window = np.exp(-0.5 * (centered / width) ** 2)
        shaped = broadband * window

    peak = float(np.max(np.abs(shaped)))
    if peak > 0.0:
        shaped = shaped / peak
    return shaped


def render_scanned(
    *,
    freq: float,
    n_samples: int,
    sample_rate: int,
    n_nodes: int,
    motion: float,
    tension: float,
    damping: float,
    position: float,
    seed: int,
    freq_profile: np.ndarray | None = None,
) -> np.ndarray:
    """Render a scanned-synthesis waveform.

    Args:
        freq: Fundamental scan frequency in Hz (positive).  Used to build
            a constant profile when ``freq_profile`` is not supplied.
        n_samples: Number of output samples.
        sample_rate: Audio sample rate.
        n_nodes: Number of point masses in the ring.  8-64 is the useful
            range (hard ceiling ``128``); low counts give brittle,
            high-harmonic shapes, high counts give smooth pad-like shapes.
        motion: Scalar in ``[0, 1]`` controlling how fast the mechanical
            system evolves.  0 = nearly frozen (almost a static wavetable),
            1 = aggressively morphing/chaotic.
        tension: Scalar in ``[0, 1]`` controlling neighbor-spring
            stiffness.  Higher tension = faster mechanical frequency =
            brighter, more harmonic motion.
        damping: Scalar in ``[0, 1]`` controlling velocity damping.  0 =
            undamped (motion persists indefinitely), 1 = heavily damped
            (excitation dies quickly).
        position: Scalar in ``[0, 1]`` controlling excitation spatial
            distribution.  0 = broadband (pad-like), 1 = concentrated
            (spiky / harmonic-rich).
        seed: Deterministic seed for the excitation RNG.
        freq_profile: Optional per-sample frequency trajectory (Hz,
            positive).  Length must equal ``n_samples``.  When ``None`` a
            constant profile at ``freq`` is used.

    Returns:
        Mono ``float64`` audio array of length ``n_samples``.
    """
    if n_nodes < 4:
        raise ValueError(f"n_nodes must be >= 4, got {n_nodes}")
    if n_nodes > 128:
        raise ValueError(f"n_nodes must be <= 128, got {n_nodes}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if freq <= 0.0:
        raise ValueError(f"freq must be positive, got {freq}")
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}")
    if not 0.0 <= motion <= 1.0:
        raise ValueError(f"motion must be in [0, 1], got {motion}")
    if not 0.0 <= tension <= 1.0:
        raise ValueError(f"tension must be in [0, 1], got {tension}")
    if not 0.0 <= damping <= 1.0:
        raise ValueError(f"damping must be in [0, 1], got {damping}")
    if not 0.0 <= position <= 1.0:
        raise ValueError(f"position must be in [0, 1], got {position}")

    if freq_profile is None:
        resolved_profile = np.full(n_samples, float(freq), dtype=np.float64)
    else:
        resolved_profile = np.asarray(freq_profile, dtype=np.float64)
        if resolved_profile.ndim != 1:
            raise ValueError("freq_profile must be 1-D")
        if resolved_profile.shape[0] != n_samples:
            raise ValueError(
                f"freq_profile length {resolved_profile.shape[0]} != "
                f"n_samples {n_samples}"
            )
        if np.any(resolved_profile <= 0.0):
            raise ValueError("freq_profile values must be positive")

    if n_samples == 0:
        return np.zeros(0, dtype=np.float64)

    rng = np.random.default_rng(seed)
    positions = _excite_ring(n_nodes=n_nodes, position=position, rng=rng)
    velocities = np.zeros(n_nodes, dtype=np.float64)

    # Map perceptual params to physical coefficients.  Tuned by ear to
    # span a musical range without blowing up across 0..1 on each knob.
    # Neighbor stiffness sets the ring's mechanical oscillation rate
    # (sqrt(k) is the natural rate).  50..10000 gives an audible
    # modulation spectrum from ~1 Hz to ~16 Hz mechanical breathing.
    neighbor_stiffness = 50.0 + 9950.0 * (tension**2)
    # Small elastic restoration keeps the center of mass from drifting
    # off indefinitely — independent of neighbor stiffness.
    restore_stiffness = 5.0 + 20.0 * tension
    # Damping maps to the time constant of velocity decay.
    # Calibrated so per-second velocity multiplier ≈ exp(-damping_coeff).
    # damping=0 => ~0.05/s (rings for 20s); damping=1 => ~15/s (~65 ms tau).
    damping_coeff = 0.05 + 15.0 * (damping**1.5)
    # Mechanical dt scales how fast the ring evolves in audio time.
    # At motion=0, dt*sample_rate ≈ 0.1 (ring advances at 1/10x real time);
    # at motion=0.5, dt*sample_rate ≈ 1 (real-time mechanical evolution);
    # at motion=1, dt*sample_rate ≈ 10 (10x faster => bubbling texture).
    # Stability: Euler needs dt*sqrt(k) < 2, so max dt ≤ 2e-4 at k=10000.
    inv_sr = 1.0 / float(sample_rate)
    motion_mult = 0.1 + 9.9 * motion
    stability_dt = 1.8 / max(math.sqrt(neighbor_stiffness), 1.0)
    mechanical_dt = min(motion_mult * inv_sr, stability_dt)
    # Damping is applied as ``v *= (1 - c*dt)`` per mechanical step, and
    # we do one step per audio sample, so the per-audio-second decay
    # factor becomes ``(1 - c*dt)^sample_rate`` ≈ ``exp(-c*dt*sample_rate)``.
    # To honor the damping_coeff calibration above regardless of the
    # motion-scaled dt, we convert damping_coeff (per-second) into a
    # per-step coefficient: per_step_damping = damping_coeff * dt.
    damping_coeff = damping_coeff / max(mechanical_dt * float(sample_rate), 1e-12)

    phase_increment_per_sample = (resolved_profile / float(sample_rate)).astype(
        np.float64
    )

    out = np.zeros(n_samples, dtype=np.float64)
    _scan_ring_loop(
        out,
        positions,
        velocities,
        phase_increment_per_sample,
        neighbor_stiffness,
        restore_stiffness,
        damping_coeff,
        mechanical_dt,
    )

    peak = float(np.max(np.abs(out)))
    if peak > 1e-12:
        out = out / peak
    return out
