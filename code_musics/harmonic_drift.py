"""JI-aware harmonic drift: smooth pitch trajectories shaped by consonance.

Generates pitch_ratio automation lanes that glide between chords while
lingering near pure JI intervals and moving quickly through rough zones.
"""

from __future__ import annotations

import numpy as np

from code_musics.automation import (
    AutomationSegment,
    AutomationSpec,
    AutomationTarget,
)
from code_musics.tuning import enumerate_ji_ratios, tenney_height


def harmonic_drift(
    start_chord: list[float],
    end_chord: list[float],
    duration: float,
    attraction: float = 0.5,
    prime_limit: int = 7,
    wander: float = 0.0,
    smoothness: float = 0.8,
    resolution_ms: float = 50.0,
    seed: int = 0,
) -> list[AutomationSpec]:
    """Generate pitch_ratio automation lanes that drift between two JI chords.

    The trajectory lingers near consonant JI intervals (controlled by attraction)
    and moves quickly through rough zones.

    Returns one AutomationSpec per voice (chord tone), each targeting pitch_ratio
    in multiply mode.
    """
    if len(start_chord) != len(end_chord):
        raise ValueError(
            f"start_chord and end_chord must be the same length, "
            f"got {len(start_chord)} and {len(end_chord)}"
        )
    if duration <= 0:
        raise ValueError("duration must be positive")
    if not 0.0 <= attraction <= 1.0:
        raise ValueError("attraction must be between 0 and 1")
    if not 0.0 <= wander <= 1.0:
        raise ValueError("wander must be between 0 and 1")
    if not 0.0 <= smoothness <= 1.0:
        raise ValueError("smoothness must be between 0 and 1")

    rng = np.random.default_rng(seed)
    lanes: list[AutomationSpec] = []

    for start_ratio, end_ratio in zip(start_chord, end_chord, strict=True):
        trajectory = _build_voice_trajectory(
            start_ratio=start_ratio,
            end_ratio=end_ratio,
            duration=duration,
            attraction=attraction,
            prime_limit=prime_limit,
            wander=wander,
            smoothness=smoothness,
            resolution_ms=resolution_ms,
            rng=rng,
        )
        lane = _trajectory_to_automation(
            trajectory=trajectory,
            start_ratio=start_ratio,
            duration=duration,
            resolution_ms=resolution_ms,
        )
        lanes.append(lane)

    return lanes


def _build_voice_trajectory(
    *,
    start_ratio: float,
    end_ratio: float,
    duration: float,
    attraction: float,
    prime_limit: int,
    wander: float,
    smoothness: float,
    resolution_ms: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build a pitch trajectory (in absolute ratio space) for one voice."""
    n_samples = max(2, int(duration / (resolution_ms / 1000.0)))

    log_start = np.log2(start_ratio)
    log_end = np.log2(end_ratio)

    # Base path: linear in log-pitch space (geometric interpolation).
    base_pitch = np.linspace(log_start, log_end, n_samples)

    if attraction > 0.0 and not np.isclose(start_ratio, end_ratio):
        base_pitch = _apply_attraction(
            base_pitch=base_pitch,
            start_ratio=start_ratio,
            end_ratio=end_ratio,
            attraction=attraction,
            prime_limit=prime_limit,
        )

    if wander > 0.0 and not np.isclose(start_ratio, end_ratio):
        base_pitch = _apply_wander(
            pitch=base_pitch,
            wander=wander,
            prime_limit=prime_limit,
            rng=rng,
        )

    if smoothness > 0.0:
        base_pitch = _apply_smoothing(base_pitch, smoothness)

    # Clamp endpoints exactly.
    base_pitch[0] = log_start
    base_pitch[-1] = log_end

    return np.power(2.0, base_pitch)


def _apply_attraction(
    *,
    base_pitch: np.ndarray,
    start_ratio: float,
    end_ratio: float,
    attraction: float,
    prime_limit: int,
) -> np.ndarray:
    """Time-warp the trajectory to linger near consonant JI waypoints."""
    n_samples = len(base_pitch)
    low = min(start_ratio, end_ratio)
    high = max(start_ratio, end_ratio)
    waypoints = enumerate_ji_ratios(low, high, prime_limit=prime_limit)

    if not waypoints:
        return base_pitch

    # Compute consonance density at each pitch point.
    density = np.ones(n_samples, dtype=np.float64)
    for wp_ratio in waypoints:
        wp_pitch = np.log2(wp_ratio)
        wp_height = tenney_height(wp_ratio)
        sigma = 0.02 + 0.01 * wp_height
        weight = 1.0 / (1.0 + wp_height)
        density += (
            attraction * weight * np.exp(-0.5 * ((base_pitch - wp_pitch) / sigma) ** 2)
        )

    # Time-remap: higher density = slower traversal = more time near consonances.
    cum_density = np.cumsum(density)
    cum_density /= cum_density[-1]

    warped_pitch = np.interp(np.linspace(0.0, 1.0, n_samples), cum_density, base_pitch)
    return warped_pitch


def _apply_wander(
    *,
    pitch: np.ndarray,
    wander: float,
    prime_limit: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add smooth noise biased toward nearby JI ratios."""
    n_samples = len(pitch)

    # Brownian motion (cumulative sum of small random steps).
    steps = rng.standard_normal(n_samples) * 0.005 * wander
    brownian = np.cumsum(steps)
    # Remove drift so endpoints aren't shifted.
    brownian -= np.linspace(brownian[0], brownian[-1], n_samples)

    wandered = pitch + brownian

    # Pull toward nearest JI ratio at each point.
    pitch_low = float(np.min(np.power(2.0, pitch)))
    pitch_high = float(np.max(np.power(2.0, pitch)))
    margin = 0.1 * (pitch_high - pitch_low) if pitch_high > pitch_low else 0.1
    ji_ratios = enumerate_ji_ratios(
        max(0.01, pitch_low - margin),
        pitch_high + margin,
        prime_limit=prime_limit,
    )

    if ji_ratios:
        ji_log = np.array([np.log2(r) for r in ji_ratios])
        ji_heights = np.array([tenney_height(r) for r in ji_ratios])
        ji_weights = 1.0 / (1.0 + ji_heights)

        for i in range(1, n_samples - 1):
            distances = np.abs(wandered[i] - ji_log)
            nearest_idx = int(np.argmin(distances))
            pull_strength = 0.3 * wander * ji_weights[nearest_idx]
            pull_amount = (ji_log[nearest_idx] - wandered[i]) * pull_strength
            wandered[i] += pull_amount

    return wandered


def _apply_smoothing(pitch: np.ndarray, smoothness: float) -> np.ndarray:
    """Exponential moving average smoothing on the pitch trajectory."""
    alpha = 1.0 - 0.95 * smoothness
    smoothed = np.copy(pitch)
    for i in range(1, len(smoothed)):
        smoothed[i] = alpha * smoothed[i] + (1.0 - alpha) * smoothed[i - 1]
    return smoothed


def _trajectory_to_automation(
    *,
    trajectory: np.ndarray,
    start_ratio: float,
    duration: float,
    resolution_ms: float,
) -> AutomationSpec:
    """Convert an absolute-ratio trajectory into a pitch_ratio AutomationSpec."""
    n_samples = len(trajectory)
    time_step = duration / max(n_samples - 1, 1)

    target = AutomationTarget(kind="pitch_ratio", name="pitch_ratio")
    segments: list[AutomationSegment] = []

    for i in range(n_samples - 1):
        seg_start = i * time_step
        seg_end = (i + 1) * time_step
        # Automation values are ratios relative to the note's base pitch.
        ratio_start = trajectory[i] / start_ratio
        ratio_end = trajectory[i + 1] / start_ratio

        segments.append(
            AutomationSegment(
                start=seg_start,
                end=seg_end,
                shape="linear",
                start_value=ratio_start,
                end_value=ratio_end,
            )
        )

    return AutomationSpec(
        target=target,
        segments=tuple(segments),
        default_value=1.0,
        mode="multiply",
    )
