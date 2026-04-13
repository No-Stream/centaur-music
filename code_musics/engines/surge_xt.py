"""Surge XT instrument engine -- renders voices through pedalboard's VSTi hosting."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from mido import Message

from code_musics.engines._mpe_utils import (
    DEFAULT_GLOBAL_GLIDE_TIME_SECONDS,
    build_cc_messages,
    build_global_bend_messages,
    build_mpe_config_messages,
    build_mpe_note_messages,
)
from code_musics.synth import SAMPLE_RATE, has_external_plugin, load_external_plugin

logger: logging.Logger = logging.getLogger(__name__)

PARAM_CURVE_CHUNK_SECONDS = 0.05  # 50ms -- small enough that per-step parameter
# changes are inaudible.  At 0.5s the steps were clearly audible as 2 Hz beating
# and timbral jumps, especially on clean (sine-like) voices.
CROSSFADE_SAMPLES = 128  # ~2.9ms at 44100 Hz -- smooths chunk boundary discontinuities


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

    .. warning:: **EXPERIMENTAL / BROKEN** -- this function produces audible
       clicking and popping artifacts at chunk boundaries and should not be
       used in production pieces.

       The root cause is fundamental: when a plugin parameter changes as a
       step function between chunks, the plugin's internal DSP state (IIR
       filter feedback lines, oscillator phase accumulators, delay buffers,
       etc.) cannot smoothly transition.  The discontinuity is generated
       *inside* the plugin, so output-level crossfading between chunks does
       not fix it -- by the time we see the audio, the artifact is already
       baked in.

       Prefer native post-processing effects with score-time automation
       instead, which update parameters sample-accurately within a single
       continuous render pass.

       This code is retained for experimentation (it may be useful with
       plugins that have smoother internal parameter interpolation, or as a
       starting point for future improvements) but should not be relied on
       for finished pieces.

    Divides *render_duration* into ``PARAM_CURVE_CHUNK_SECONDS``-long chunks.
    Before each chunk, interpolates *param_curves* at the chunk start time and
    sets the corresponding ``plugin.parameters[name].raw_value``.  MIDI messages
    are filtered per-chunk and time-offset to chunk-relative coordinates.

    To eliminate audible clicks at chunk boundaries (caused by instantaneous
    parameter jumps), each non-final chunk is rendered with a short overlap
    tail.  Adjacent chunks are then joined with a linear crossfade over the
    overlap region.
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

    overlap_samples = min(CROSSFADE_SAMPLES, int(chunk_dur * sample_rate) // 2)
    overlap_dur = overlap_samples / sample_rate

    chunks: list[np.ndarray] = []
    for chunk_idx, (chunk_start, this_chunk_dur) in enumerate(chunk_boundaries):
        chunk_end = chunk_start + this_chunk_dur
        is_last = chunk_idx == len(chunk_boundaries) - 1

        # Non-final chunks render extra overlap samples so we can crossfade
        # at the boundary with the next chunk.
        render_dur = this_chunk_dur if is_last else this_chunk_dur + overlap_dur

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
            duration=render_dur,
            num_channels=2,
            buffer_size=buffer_size,
        )
        chunks.append(np.asarray(chunk_audio, dtype=np.float64))

    if len(chunks) <= 1:
        return np.concatenate(chunks, axis=1) if chunks else np.empty((2, 0))

    # Build crossfade ramps once (linear fade, shape broadcastable over channels)
    fade_out = np.linspace(1.0, 0.0, overlap_samples)[np.newaxis, :]
    fade_in = 1.0 - fade_out

    # Stitch chunks with crossfade at boundaries.
    # Non-final chunks were rendered with an overlap tail; we split each chunk
    # into its nominal body and its overlap tail, crossfade adjacent tails/heads,
    # and concatenate.
    parts: list[np.ndarray] = []

    # First chunk: nominal body (exclude overlap tail)
    parts.append(chunks[0][:, :-overlap_samples])

    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][:, -overlap_samples:]
        curr_head = chunks[i][:, :overlap_samples]
        parts.append(prev_tail * fade_out + curr_head * fade_in)

        is_last_chunk = i == len(chunks) - 1
        if is_last_chunk:
            # Last chunk has no overlap tail -- take everything after the head
            if chunks[i].shape[1] > overlap_samples:
                parts.append(chunks[i][:, overlap_samples:])
        else:
            # Intermediate chunk: exclude overlap tail (will be crossfaded next iter)
            body_end = chunks[i].shape[1] - overlap_samples
            if body_end > overlap_samples:
                parts.append(chunks[i][:, overlap_samples:body_end])

    return np.concatenate(parts, axis=1)


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
          so pitch bend is per-note with sub-cent accuracy.  When False,
          use global-bend chord mode: notes are grouped into chords by
          start time, each chord shares a single pitch bend derived from
          the bass note, and consecutive chords glide smoothly between
          reference pitches (like a tremolo bar).  Non-bass notes have up
          to ~50 cent rounding error from MIDI note quantisation; the bass
          is always perfectly tuned.
        - ``global_glide_time`` -- seconds for the pitch-bend glide between
          consecutive chords in global-bend mode (default 0.4).  Only used
          when ``mpe=False``.
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
    messages: list[Message] = build_mpe_config_messages() if use_mpe else []

    if use_mpe:
        messages.extend(build_mpe_note_messages(notes, release_padding))
    else:
        global_glide_time = float(
            resolved_params.get("global_glide_time", DEFAULT_GLOBAL_GLIDE_TIME_SECONDS)
        )
        messages.extend(build_global_bend_messages(notes, global_glide_time))

    # -- CC automation curves --------------------------------------------------
    cc_curves: list[dict[str, Any]] = resolved_params.get("cc_curves", [])
    if cc_curves:
        messages.extend(build_cc_messages(cc_curves, total_duration))

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
        logger.warning(
            "param_curves is experimental and produces audible artifacts (clicks/pops) "
            "at chunk boundaries. Prefer native post-processing effects with score-time "
            "automation instead. See _render_chunked() docstring for details."
        )
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
