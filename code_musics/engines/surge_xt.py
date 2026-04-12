"""Surge XT instrument engine -- renders voices through pedalboard's VSTi hosting."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from mido import Message

from code_musics.synth import SAMPLE_RATE, has_external_plugin, load_external_plugin

logger: logging.Logger = logging.getLogger(__name__)

BEND_RANGE_SEMITONES = 24.0
MAX_MPE_CHANNELS = 15  # channels 1-15; channel 0 is MPE manager/global


def _build_mpe_config_messages(
    num_member_channels: int = MAX_MPE_CHANNELS,
    bend_range_semitones: int = int(BEND_RANGE_SEMITONES),
) -> list[Message]:
    """Build MCM (MPE Configuration Message) + per-channel bend range RPNs.

    MCM tells Surge XT to enter MPE Lower Zone mode so that per-channel pitch
    bend is routed per-note instead of globally.  RPN 0 on each member channel
    sets the pitch bend range to match ``BEND_RANGE_SEMITONES``.
    """
    msgs: list[Message] = [
        # MCM on manager channel 0: declare MPE Lower Zone
        Message("control_change", channel=0, control=101, value=0, time=0.0),
        Message("control_change", channel=0, control=100, value=6, time=0.0),
        Message(
            "control_change", channel=0, control=6, value=num_member_channels, time=0.0
        ),
        Message("control_change", channel=0, control=38, value=0, time=0.0),
    ]
    # RPN 0 on each member channel: set pitch bend range
    for ch in range(1, num_member_channels + 1):
        msgs.extend(
            [
                Message("control_change", channel=ch, control=101, value=0, time=0.0),
                Message("control_change", channel=ch, control=100, value=0, time=0.0),
                Message(
                    "control_change",
                    channel=ch,
                    control=6,
                    value=bend_range_semitones,
                    time=0.0,
                ),
                Message("control_change", channel=ch, control=38, value=0, time=0.0),
            ]
        )
    return msgs


def _resolve_glide_bend(target_midi_note: int, freq_hz: float) -> int:
    """Compute pitch bend value to reach *freq_hz* from a given MIDI note.

    This is the glide counterpart to ``_resolve_note_and_bend``: given a
    fixed MIDI note (the target note of the glide), compute what pitch-bend
    value would tune that note to an arbitrary frequency.  Returns a 14-bit
    signed bend value in [-8191, 8191].
    """
    if freq_hz <= 0:
        raise ValueError(f"freq_hz must be positive, got {freq_hz}")
    target_semitones = 69.0 + 12.0 * math.log2(freq_hz / 440.0)
    residual = target_semitones - target_midi_note
    normalized = residual / BEND_RANGE_SEMITONES
    return int(round(max(-1.0, min(1.0, normalized)) * 8191.0))


def _resolve_note_and_bend(freq_hz: float) -> tuple[int, int]:
    """Find nearest MIDI note and pitch bend value for an arbitrary frequency.

    Uses a 24-semitone pitch bend range for sub-cent microtonal accuracy.
    """
    midi_note_float = 69.0 + (12.0 * math.log2(freq_hz / 440.0))
    midi_note = int(round(midi_note_float))
    midi_note = max(0, min(127, midi_note))
    residual_semitones = midi_note_float - midi_note
    if abs(residual_semitones) > BEND_RANGE_SEMITONES:
        raise ValueError(
            f"frequency {freq_hz} Hz requires pitch bend beyond "
            f"{BEND_RANGE_SEMITONES} semitones"
        )
    normalized_bend = residual_semitones / BEND_RANGE_SEMITONES
    bend_value = int(round(max(-1.0, min(1.0, normalized_bend)) * 8191.0))
    return midi_note, bend_value


PARAM_CURVE_CHUNK_SECONDS = 0.5


def _interpolate_param_curves(
    param_curves: list[dict[str, Any]], time: float
) -> dict[str, float]:
    """Interpolate all param curves at a given time, returning {param_name: raw_value}.

    Linear interpolation between breakpoints.  Holds the first value for times
    before the first breakpoint and the last value for times after the last
    breakpoint.  Values are clamped to [0.0, 1.0].
    """
    result: dict[str, float] = {}
    for curve in param_curves:
        param_name: str = curve["param"]
        points: list[tuple[float, float]] = sorted(curve["points"], key=lambda p: p[0])

        if not points:
            continue

        if len(points) == 1 or time <= points[0][0]:
            value = points[0][1]
        elif time >= points[-1][0]:
            value = points[-1][1]
        else:
            # Find the segment containing *time*
            seg_idx = 0
            for i in range(len(points) - 1):
                if points[i + 1][0] >= time:
                    seg_idx = i
                    break
            t0, v0 = points[seg_idx]
            t1, v1 = points[seg_idx + 1]
            frac = (time - t0) / (t1 - t0) if t1 != t0 else 1.0
            value = v0 + frac * (v1 - v0)

        result[param_name] = max(0.0, min(1.0, value))
    return result


CC_UPDATE_INTERVAL_SECONDS = 0.01  # 10ms = 100 Hz update rate for CC automation


def _build_cc_messages(
    cc_curves: list[dict[str, Any]], total_duration: float
) -> list[Message]:
    """Generate MIDI CC messages from automation curves.

    Each entry in *cc_curves* is a dict with:
    - ``cc``: MIDI CC number (0--127)
    - ``channel``: MIDI channel (default 0)
    - ``points``: list of ``(time_seconds, value_0_to_1)`` breakpoints

    Breakpoints are linearly interpolated at ``CC_UPDATE_INTERVAL_SECONDS``
    resolution.  Values are clamped to [0.0, 1.0] and mapped to MIDI 0--127.
    """
    messages: list[Message] = []

    for curve in cc_curves:
        cc_number = int(curve["cc"])
        if not 0 <= cc_number <= 127:
            logger.warning(
                "CC number %d outside valid range 0-127 -- skipping curve", cc_number
            )
            continue

        channel = int(curve.get("channel", 0))
        raw_points: list[tuple[float, float]] = list(curve.get("points", []))

        if not raw_points:
            continue

        # Sort breakpoints by time
        raw_points.sort(key=lambda p: p[0])

        # Single breakpoint: hold the value for the entire duration
        if len(raw_points) == 1:
            hold_value = max(0.0, min(1.0, raw_points[0][1]))
            midi_value = int(round(hold_value * 127))
            num_steps = max(1, int(total_duration / CC_UPDATE_INTERVAL_SECONDS))
            for step in range(num_steps + 1):
                t = step * CC_UPDATE_INTERVAL_SECONDS
                if t > total_duration:
                    t = total_duration
                messages.append(
                    Message(
                        "control_change",
                        channel=channel,
                        control=cc_number,
                        value=midi_value,
                        time=t,
                    )
                )
            continue

        # Multi-point: linearly interpolate between breakpoints
        curve_start = raw_points[0][0]
        curve_end = raw_points[-1][0]
        span = curve_end - curve_start
        if span <= 0:
            # All points at the same time: use the last value
            hold_value = max(0.0, min(1.0, raw_points[-1][1]))
            midi_value = int(round(hold_value * 127))
            messages.append(
                Message(
                    "control_change",
                    channel=channel,
                    control=cc_number,
                    value=midi_value,
                    time=curve_start,
                )
            )
            continue

        num_steps = max(1, int(span / CC_UPDATE_INTERVAL_SECONDS))
        segment_idx = 0

        for step in range(num_steps + 1):
            t = curve_start + step * (span / num_steps)

            # Advance to the correct segment
            while (
                segment_idx < len(raw_points) - 2
                and t >= raw_points[segment_idx + 1][0]
            ):
                segment_idx += 1

            t0, v0 = raw_points[segment_idx]
            t1, v1 = raw_points[segment_idx + 1]
            seg_span = t1 - t0
            if seg_span > 0:
                frac = (t - t0) / seg_span
                frac = max(0.0, min(1.0, frac))
                value = v0 + frac * (v1 - v0)
            else:
                value = v1

            value = max(0.0, min(1.0, value))
            midi_value = int(round(value * 127))
            messages.append(
                Message(
                    "control_change",
                    channel=channel,
                    control=cc_number,
                    value=midi_value,
                    time=t,
                )
            )

    return messages


def _render_chunked(
    *,
    plugin: Any,
    messages: list[Message],
    param_curves: list[dict[str, Any]],
    render_duration: float,
    sample_rate: int,
    buffer_size: int,
) -> np.ndarray:
    """Render audio in chunks, updating plugin parameters between chunks.

    Divides *render_duration* into ``PARAM_CURVE_CHUNK_SECONDS``-long chunks.
    Before each chunk, interpolates *param_curves* at the chunk start time and
    sets the corresponding ``plugin.parameters[name].raw_value``.  MIDI messages
    are filtered per-chunk and time-offset to chunk-relative coordinates.
    """
    chunk_dur = PARAM_CURVE_CHUNK_SECONDS
    num_full_chunks = int(render_duration / chunk_dur)
    remainder = render_duration - num_full_chunks * chunk_dur

    chunk_boundaries: list[tuple[float, float]] = []
    for i in range(num_full_chunks):
        chunk_boundaries.append((i * chunk_dur, chunk_dur))
    if remainder > 1e-9:
        chunk_boundaries.append((num_full_chunks * chunk_dur, remainder))

    # Validate param names once up front
    valid_curves: list[dict[str, Any]] = []
    for curve in param_curves:
        param_name = curve["param"]
        if param_name in plugin.parameters:
            valid_curves.append(curve)
        else:
            logger.warning(
                "param_curves: unknown Surge XT parameter %r -- skipping curve",
                param_name,
            )

    chunks: list[np.ndarray] = []
    for chunk_start, this_chunk_dur in chunk_boundaries:
        chunk_end = chunk_start + this_chunk_dur

        # Set parameter values for this chunk
        param_values = _interpolate_param_curves(valid_curves, chunk_start)
        for param_name, raw_value in param_values.items():
            plugin.parameters[param_name].raw_value = raw_value

        # Filter messages to this chunk's time window: [chunk_start, chunk_end)
        chunk_messages: list[Message] = []
        for msg in messages:
            msg_time: float = msg.time  # type: ignore[reportAttributeAccessIssue]
            if chunk_start <= msg_time < chunk_end:
                chunk_messages.append(msg.copy(time=msg_time - chunk_start))
            elif chunk_start == 0.0 and msg_time == 0.0:
                # RPN setup messages at time=0 go in the first chunk
                # (already captured by the condition above)
                pass

        chunk_audio = plugin(
            chunk_messages,
            sample_rate=sample_rate,
            duration=this_chunk_dur,
            num_channels=2,
            buffer_size=buffer_size,
        )
        chunks.append(np.asarray(chunk_audio, dtype=np.float64))

    return np.concatenate(chunks, axis=1)


def render_voice(
    *,
    notes: list[dict[str, Any]],
    total_duration: float,
    sample_rate: int = SAMPLE_RATE,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Render a full voice through Surge XT using MPE-style per-note pitch bend.

    Parameters
    ----------
    notes
        Each dict has: ``freq`` (Hz), ``start`` (seconds), ``duration`` (seconds),
        ``velocity`` (0.0--1.0), ``amp`` (linear amplitude, used for velocity scaling).
        Optional glide fields: ``glide_from`` (Hz, starting frequency for a pitch
        sweep toward ``freq``) and ``glide_time`` (seconds, defaults to full
        ``duration``).  The glide is linear in pitch-bend space at ~200 Hz
        update rate (5 ms steps).  If the glide span exceeds
        ``BEND_RANGE_SEMITONES`` a warning is logged and the glide is skipped.
    total_duration
        Total duration of the voice in seconds (the latest note-off time).
    sample_rate
        Sample rate for rendering.
    params
        Optional engine params.  Recognised keys:

        - ``preset_path`` -- path to a ``.vstpreset`` or ``.fxp`` file
        - ``raw_state`` -- serialised plugin state (bytes)
        - ``tail_seconds`` -- extra render time after last note-off (default 2.0)
        - ``release_padding`` -- seconds to keep an MPE channel reserved after
          note-off so pitch bend from new notes does not bleed into release
          tails (default 1.0)
        - ``mpe`` -- when True (default), send MCM to enable MPE Lower Zone
          so pitch bend is per-note.  When False, pitch bend is global
          (scene-level), meaning new notes repitch existing voices -- useful
          as a creative effect but not for accurate polyphonic tuning.
        - ``buffer_size`` -- pedalboard processing block size in samples
          (default 256).  MIDI events are delivered at block boundaries, so
          smaller blocks give finer-grained pitch bend / CC timing.  The
          pedalboard library default of 8192 (~186 ms) produces audible
          staircase artifacts on pitch glides; 256 (~5.8 ms) matches typical
          DAW host granularity.
    """
    resolved_params = params or {}

    if not has_external_plugin("surge_xt"):
        logger.warning(
            "SKIPPING Surge XT voice render (%d notes, %.1fs): Surge XT VST3 "
            "is not installed on this machine. The voice will be silent. "
            "Install Surge XT and place at ~/.vst3/Surge XT.vst3",
            len(notes),
            total_duration,
        )
        return np.zeros((2, int(total_duration * sample_rate)), dtype=np.float64)

    plugin = load_external_plugin(plugin_name="surge_xt")

    if not getattr(plugin, "is_instrument", False):
        raise RuntimeError("Loaded Surge XT plugin is not an instrument")

    # Restore preset/state if provided
    preset_path = resolved_params.get("preset_path")
    raw_state = resolved_params.get("raw_state")
    if preset_path is not None:
        plugin.load_preset(str(preset_path))
    elif raw_state is not None:
        plugin.raw_state = raw_state

    use_mpe = bool(resolved_params.get("mpe", True))

    # Set scene-level pitch bend range via parameter API.  In non-MPE mode
    # this is the only bend range control (RPN is ignored outside MPE).
    # In MPE mode the per-note range is set via RPN in the MIDI stream, but
    # the scene-level setting is still applied as a harmless safety net.
    _bend_range_raw = 1.0  # raw_value 1.0 = 24 semitones (Surge XT maximum)
    for prefix in ("a_", "b_"):
        for direction in ("up", "down"):
            param_name = f"{prefix}pitch_bend_{direction}_range"
            if param_name in plugin.parameters:
                plugin.parameters[param_name].raw_value = _bend_range_raw

    # Apply per-voice synthesis parameters (oscillator type, filter, envelope,
    # etc.) so pieces can configure the sound from the score level.
    surge_params: dict[str, float] = resolved_params.get("surge_params", {})
    for param_name, raw_value in surge_params.items():
        if param_name in plugin.parameters:
            plugin.parameters[param_name].raw_value = float(raw_value)
        else:
            logger.warning("Unknown Surge XT parameter %r -- skipping", param_name)

    tail_seconds = float(resolved_params.get("tail_seconds", 2.0))
    render_duration = total_duration + tail_seconds

    # Release padding: after note-off, the synth's amp envelope release phase
    # keeps producing sound.  If we reuse the MPE channel during that window
    # and send a new pitch bend, the old note's release tail gets pitch-shifted
    # -- audible as an unwanted glide.  Pad channel_free_at so channels are not
    # reused while releases are still audible.
    release_padding = float(resolved_params.get("release_padding", 1.0))

    # -- Build MIDI messages -------------------------------------------------
    messages: list[Message] = _build_mpe_config_messages() if use_mpe else []

    # Assign notes to MPE member channels 1-15, preferring a channel whose
    # previous note's release tail has died away.  Falls back to round-robin
    # when all 15 channels are occupied (with a warning).
    channel_free_at = [0.0] * (MAX_MPE_CHANNELS + 1)  # index 0 unused
    for i, note_data in enumerate(notes):
        freq = float(note_data["freq"])
        start = float(note_data["start"])
        duration = float(note_data["duration"])
        velocity_raw = float(note_data.get("velocity", 0.8))
        amp = float(note_data.get("amp", 1.0))

        midi_velocity = max(1, min(127, int(round(velocity_raw * amp * 127))))
        for ch_offset in range(MAX_MPE_CHANNELS):
            candidate = ((i + ch_offset) % MAX_MPE_CHANNELS) + 1
            if channel_free_at[candidate] <= start:
                channel = candidate
                break
        else:
            channel = (i % MAX_MPE_CHANNELS) + 1
            logger.warning(
                "MPE channel collision: >%d overlapping notes at t=%.3f "
                "(release_padding=%.2fs)",
                MAX_MPE_CHANNELS,
                start,
                release_padding,
            )
        channel_free_at[channel] = start + duration + release_padding

        midi_note, bend_value = _resolve_note_and_bend(freq)

        # -- Glide support: optionally start at a different pitch and sweep --
        glide_from = note_data.get("glide_from")
        initial_bend = bend_value  # default: start at target pitch
        if glide_from is not None:
            glide_from_hz = float(glide_from)
            glide_from_semitones = 69.0 + 12.0 * math.log2(glide_from_hz / 440.0)
            glide_distance = abs(
                glide_from_semitones
                - (midi_note + bend_value / 8191.0 * BEND_RANGE_SEMITONES)
            )
            if glide_distance > BEND_RANGE_SEMITONES:
                logger.warning(
                    "Glide from %.1f Hz to %.1f Hz spans %.1f semitones, "
                    "beyond bend range of %.0f -- skipping glide",
                    glide_from_hz,
                    freq,
                    glide_distance,
                    BEND_RANGE_SEMITONES,
                )
                glide_from = None  # fall through to normal non-glide behavior
            else:
                initial_bend = _resolve_glide_bend(midi_note, glide_from_hz)

        # Pitch bend BEFORE note-on so the plugin sees the tuning first.
        messages.append(
            Message("pitchwheel", channel=channel, pitch=initial_bend, time=start)
        )
        messages.append(
            Message(
                "note_on",
                channel=channel,
                note=midi_note,
                velocity=midi_velocity,
                time=start,
            )
        )

        # Generate intermediate pitch bend messages for glide
        if glide_from is not None:
            glide_time = float(note_data.get("glide_time", duration))
            glide_time = min(glide_time, duration)
            glide_step_interval = 0.005  # 5ms steps (~200 Hz update rate)
            num_steps = max(1, int(glide_time / glide_step_interval))
            for step in range(1, num_steps + 1):
                t_fraction = step / num_steps
                t_absolute = start + t_fraction * glide_time
                # Linear interpolation in bend space
                intermediate_bend = int(
                    round(initial_bend + t_fraction * (bend_value - initial_bend))
                )
                intermediate_bend = max(-8191, min(8191, intermediate_bend))
                messages.append(
                    Message(
                        "pitchwheel",
                        channel=channel,
                        pitch=intermediate_bend,
                        time=t_absolute,
                    )
                )

        messages.append(
            Message(
                "note_off",
                channel=channel,
                note=midi_note,
                velocity=0,
                time=start + duration,
            )
        )

    # -- CC automation curves --------------------------------------------------
    cc_curves: list[dict[str, Any]] = resolved_params.get("cc_curves", [])
    if cc_curves:
        messages.extend(_build_cc_messages(cc_curves, total_duration))

    messages.sort(key=lambda m: m.time)  # type: ignore[reportAttributeAccessIssue]

    logger.info(
        "Rendering %d notes through Surge XT (%.1fs + %.1fs tail)",
        len(notes),
        total_duration,
        tail_seconds,
    )

    # Use a small buffer size so MIDI events (especially pitch bend glides)
    # are delivered at close to their requested timestamps.  The pedalboard
    # default of 8192 samples (~186 ms at 44.1 kHz) quantises all messages
    # within a block to the block start, turning smooth pitch glides into
    # audible staircases.  256 samples (~5.8 ms) matches typical DAW host
    # granularity and keeps each 5 ms glide step in its own processing block.
    buffer_size = int(resolved_params.get("buffer_size", 256))

    param_curves: list[dict[str, Any]] = resolved_params.get("param_curves", [])

    if param_curves:
        audio = _render_chunked(
            plugin=plugin,
            messages=messages,
            param_curves=param_curves,
            render_duration=render_duration,
            sample_rate=sample_rate,
            buffer_size=buffer_size,
        )
    else:
        audio = plugin(
            messages,
            sample_rate=sample_rate,
            duration=render_duration,
            num_channels=2,
            buffer_size=buffer_size,
        )

    # pedalboard returns shape (channels, samples) -- trim silent tail.
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        tail_start = int(total_duration * sample_rate)
        if tail_start < audio.shape[1]:
            tail = audio[:, tail_start:]
            tail_energy = np.max(np.abs(tail), axis=0)
            silent_threshold = 1e-6
            non_silent = np.where(tail_energy > silent_threshold)[0]
            if len(non_silent) > 0:
                trim_point = tail_start + non_silent[-1] + 1
                audio = audio[:, :trim_point]
            else:
                audio = audio[:, :tail_start]

    return np.asarray(audio, dtype=np.float64)
