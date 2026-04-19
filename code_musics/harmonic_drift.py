"""JI-aware harmonic drift: smooth pitch trajectories shaped by consonance.

Generates pitch_ratio automation lanes that glide between chords while
lingering near pure JI intervals and moving quickly through rough zones.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from code_musics.automation import (
    AutomationSegment,
    AutomationSpec,
    AutomationTarget,
)
from code_musics.score import NoteEvent
from code_musics.tuning import enumerate_ji_ratios, tenney_height

# Chord voicing callable: returns a list of (partial_ratio, amp_db) tuples.
ChordVoicingFn = Callable[[], list[tuple[float, float]]]


def drifted_chord_events(
    chord_partials_db: list[tuple[float, float]],
    duration: float,
    drift_lanes: list[AutomationSpec] | None,
    amp_db_offset: float = 0.0,
) -> tuple[NoteEvent, ...]:
    """Build NoteEvents for a chord, optionally attaching per-note pitch drift.

    Chord tones are sorted low-to-high by partial so that drift lane indices
    match the sorted order used in :func:`progression_drift_lanes`.
    """
    sorted_pairs = sorted(chord_partials_db, key=lambda x: x[0])
    events: list[NoteEvent] = []
    for i, (partial, amp_db) in enumerate(sorted_pairs):
        auto: list[AutomationSpec] = []
        if drift_lanes and i < len(drift_lanes):
            auto.append(drift_lanes[i])
        events.append(
            NoteEvent(
                start=0.0,
                duration=duration,
                partial=partial,
                amp_db=amp_db + amp_db_offset,
                automation=auto,
            )
        )
    return tuple(events)


# Default glide window in ms. Traditional portamento/slide durations are tens
# to a few hundred ms — multi-second glides sound like risers, not voice leading.
DEFAULT_GLIDE_MS = 250.0

# Default max interval (in cents) for a voice to glide. Larger intervals are
# treated as leaps and skip pitch_ratio automation entirely so a pad voice
# doesn't impersonate a theremin. 700 cents ~= perfect fifth.
DEFAULT_MAX_INTERVAL_CENTS = 700.0


def progression_drift_lanes(
    progression: list[ChordVoicingFn],
    chord_dur: float,
    attraction: float,
    wander: float,
    smoothness: float = 0.85,
    seed_base: int = 0,
    glide_ms: float | None = DEFAULT_GLIDE_MS,
    max_interval_cents: float = DEFAULT_MAX_INTERVAL_CENTS,
    target_time: float | None = None,
    glide_transitions: set[int] | None = None,
) -> list[list[AutomationSpec] | None]:
    """Compute drift lanes for consecutive chords in a progression.

    Returns one entry per chord: a list of :class:`AutomationSpec` (drift toward
    the next chord) or ``None`` for the last chord (or for chords whose
    transition is not selected by ``glide_transitions``). Chord tones are
    sorted low-to-high by partial before lanes are generated so that lane
    indices match the sorted order used in :func:`drifted_chord_events`.

    Args:
        glide_transitions: Optional set of chord indices whose outgoing
            transition should glide. If ``None``, every transition glides
            (the default). Use this to place drift as an accent on specific
            transitions rather than a continuous effect across the section.

    See :func:`harmonic_drift` for ``glide_ms``, ``max_interval_cents``, and
    ``target_time``.
    """
    result: list[list[AutomationSpec] | None] = []
    chords_data = [ch() for ch in progression]

    for i, chord_a in enumerate(chords_data):
        if i >= len(chords_data) - 1:
            result.append(None)
            continue
        if glide_transitions is not None and i not in glide_transitions:
            result.append(None)
            continue
        chord_b = chords_data[i + 1]

        sorted_a = sorted([p for p, _db in chord_a])
        sorted_b = sorted([p for p, _db in chord_b])
        n = min(len(sorted_a), len(sorted_b))

        lanes = harmonic_drift(
            start_chord=sorted_a[:n],
            end_chord=sorted_b[:n],
            duration=chord_dur,
            attraction=attraction,
            wander=wander,
            smoothness=smoothness,
            prime_limit=7,
            seed=seed_base + i,
            glide_ms=glide_ms,
            max_interval_cents=max_interval_cents,
            target_time=target_time,
        )
        result.append(lanes)

    return result


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
    glide_ms: float | None = DEFAULT_GLIDE_MS,
    max_interval_cents: float = DEFAULT_MAX_INTERVAL_CENTS,
    target_time: float | None = None,
) -> list[AutomationSpec]:
    """Generate pitch_ratio automation lanes that drift between two JI chords.

    The trajectory lingers near consonant JI intervals (controlled by attraction)
    and moves quickly through rough zones.

    Returns one AutomationSpec per voice (chord tone), each targeting pitch_ratio
    in multiply mode.

    Args:
        glide_ms: Glide window in ms before the glide target time. The voice
            holds at ``start_ratio`` for most of the note and only glides in
            the final ``glide_ms`` before the target. Pass ``None`` to glide
            across the entire duration (the legacy behavior).
        target_time: Time (in seconds, from the note's start) at which the
            glide must reach ``end_ratio``. After this time the lane holds
            at ``end_ratio`` for the rest of the note. Defaults to the note's
            ``duration`` — i.e. glide completes at note end. For overlapping
            chord transitions where the next chord attacks before this note
            ends, pass the boundary time so the current note matches the next
            chord's pitch during the overlap and doesn't beat against it.
        max_interval_cents: Voices whose pitch change exceeds this magnitude
            skip pitch drift and emit a flat unity automation lane. This avoids
            multi-octave "riser" slides when chord voicings pair voices from
            very different registers.
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
    if glide_ms is not None and glide_ms <= 0:
        raise ValueError("glide_ms must be positive or None")
    if max_interval_cents < 0:
        raise ValueError("max_interval_cents must be non-negative")
    if target_time is not None and (target_time <= 0 or target_time > duration):
        raise ValueError("target_time must be in (0, duration]")

    effective_target_time = duration if target_time is None else target_time

    rng = np.random.default_rng(seed)
    lanes: list[AutomationSpec] = []

    for start_ratio, end_ratio in zip(start_chord, end_chord, strict=True):
        interval_cents = abs(1200.0 * np.log2(end_ratio / start_ratio))
        if interval_cents > max_interval_cents:
            lanes.append(_static_unity_lane(duration=duration))
            continue

        glide_dur = (
            effective_target_time
            if glide_ms is None
            else min(glide_ms / 1000.0, effective_target_time)
        )
        glide_start_time = effective_target_time - glide_dur

        trajectory = _build_voice_trajectory(
            start_ratio=start_ratio,
            end_ratio=end_ratio,
            duration=glide_dur,
            attraction=attraction,
            prime_limit=prime_limit,
            wander=wander,
            smoothness=smoothness,
            resolution_ms=resolution_ms,
            rng=rng,
        )
        # Hold at the end ratio after the glide completes, so overlapping
        # next-chord notes at that pitch don't beat against this one.
        hold_after_ratio = (
            end_ratio / start_ratio if effective_target_time < duration - 1e-9 else None
        )
        lane = _trajectory_to_automation(
            trajectory=trajectory,
            start_ratio=start_ratio,
            duration=glide_dur,
            time_offset=glide_start_time,
            hold_after=(
                (effective_target_time, duration, hold_after_ratio)
                if hold_after_ratio is not None
                else None
            ),
        )
        lanes.append(lane)

    return lanes


def _static_unity_lane(*, duration: float) -> AutomationSpec:
    """A single automation segment that holds pitch_ratio at 1.0 for the note."""
    return AutomationSpec(
        target=AutomationTarget(kind="pitch_ratio", name="pitch_ratio"),
        segments=(AutomationSegment(start=0.0, end=duration, shape="hold", value=1.0),),
        default_value=1.0,
        mode="multiply",
    )


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
    time_offset: float = 0.0,
    hold_after: tuple[float, float, float] | None = None,
) -> AutomationSpec:
    """Convert an absolute-ratio trajectory into a pitch_ratio AutomationSpec.

    If ``time_offset`` is positive, a leading hold segment at unity (1.0) is
    inserted so the voice stays at its base pitch until the glide window begins.

    If ``hold_after`` is provided as ``(start, end, ratio)``, a trailing hold
    segment at the given ratio is appended — useful when the glide completes
    before the note ends (e.g., for overlapping chord transitions where the
    note must match the next chord's pitch during the overlap window).
    """
    n_samples = len(trajectory)
    time_step = duration / max(n_samples - 1, 1)

    target = AutomationTarget(kind="pitch_ratio", name="pitch_ratio")
    segments: list[AutomationSegment] = []

    if time_offset > 0:
        segments.append(
            AutomationSegment(start=0.0, end=time_offset, shape="hold", value=1.0)
        )

    for i in range(n_samples - 1):
        seg_start = time_offset + i * time_step
        seg_end = time_offset + (i + 1) * time_step
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

    if hold_after is not None:
        hold_start, hold_end, hold_ratio = hold_after
        if hold_end > hold_start + 1e-9:
            segments.append(
                AutomationSegment(
                    start=hold_start, end=hold_end, shape="hold", value=hold_ratio
                )
            )

    return AutomationSpec(
        target=target,
        segments=tuple(segments),
        default_value=1.0,
        mode="multiply",
    )
