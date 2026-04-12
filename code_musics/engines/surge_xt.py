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


CHORD_GROUP_TOLERANCE_SECONDS = 0.1  # notes within this window are one chord
GLIDE_STEP_INTERVAL_SECONDS = 0.005  # 5ms steps (~200 Hz update rate)
DEFAULT_GLOBAL_GLIDE_TIME_SECONDS = 0.4


def _resolve_chord_notes(freqs: list[float]) -> list[tuple[int, int]]:
    """Compute MIDI note numbers and a shared pitch bend for a chord.

    In global-bend mode all notes share a single pitchwheel value.  The bass
    (lowest frequency) is used as the reference: its MIDI note and residual
    pitch bend are computed with full precision.  Every other note gets a MIDI
    note number chosen so that the shared bend places it as close as possible
    to its target frequency.

    Returns a list of ``(midi_note, shared_bend)`` tuples, one per input
    frequency, in the same order as *freqs*.
    """
    if not freqs:
        return []

    sorted_freqs = sorted(enumerate(freqs), key=lambda p: p[1])
    bass_idx, bass_freq = sorted_freqs[0]

    # Bass note: full-precision pitch bend
    bass_midi, bass_bend = _resolve_note_and_bend(bass_freq)

    # Global shift in semitones implied by the shared bend
    global_shift_semitones = (bass_bend / 8191.0) * BEND_RANGE_SEMITONES

    results: list[tuple[int, int] | None] = [None] * len(freqs)
    results[bass_idx] = (bass_midi, bass_bend)

    for orig_idx, freq in sorted_freqs[1:]:
        target_semitones = 69.0 + 12.0 * math.log2(freq / 440.0)
        # What MIDI note, when shifted by global_shift_semitones, hits target?
        ideal_midi = target_semitones - global_shift_semitones
        midi_note = int(round(ideal_midi))
        midi_note = max(0, min(127, midi_note))
        results[orig_idx] = (midi_note, bass_bend)

    return results  # type: ignore[return-value]


def _group_notes_into_chords(
    notes: list[dict[str, Any]],
    tolerance: float = CHORD_GROUP_TOLERANCE_SECONDS,
) -> list[list[dict[str, Any]]]:
    """Group notes into chord clusters by start time.

    Notes whose start times are within *tolerance* seconds of each other are
    grouped together.  Returns a list of chord groups sorted by start time.
    """
    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda n: float(n["start"]))
    chords: list[list[dict[str, Any]]] = []
    current_chord: list[dict[str, Any]] = [sorted_notes[0]]
    chord_start = float(sorted_notes[0]["start"])

    for note in sorted_notes[1:]:
        note_start = float(note["start"])
        if note_start - chord_start <= tolerance:
            current_chord.append(note)
        else:
            chords.append(current_chord)
            current_chord = [note]
            chord_start = note_start

    chords.append(current_chord)
    return chords


def _build_mpe_note_messages(
    notes: list[dict[str, Any]], release_padding: float
) -> list[Message]:
    """Build per-note MPE MIDI messages (the original MPE path).

    Each note gets its own MIDI channel with independent pitch bend for
    sub-cent microtonal accuracy.
    """
    messages: list[Message] = []
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
            num_steps = max(1, int(glide_time / GLIDE_STEP_INTERVAL_SECONDS))
            for step in range(1, num_steps + 1):
                t_fraction = step / num_steps
                t_absolute = start + t_fraction * glide_time
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

    return messages


def _build_global_bend_messages(
    notes: list[dict[str, Any]], glide_time: float
) -> list[Message]:
    """Build MIDI messages for global-bend chord mode (mpe=False).

    Groups notes into chords by start time.  Each chord shares a single
    pitchwheel value derived from its bass (lowest) note.  Non-bass notes get
    MIDI note numbers chosen relative to the shared bend so that interval
    structure is preserved (with up to ~50 cent rounding error per non-bass
    note).

    Between consecutive chord groups, intermediate pitchwheel messages glide
    the global bend from the old chord's reference to the new chord's over
    *glide_time* seconds.  All messages use MIDI channel 0.
    """
    messages: list[Message] = []
    chords = _group_notes_into_chords(notes)

    if not chords:
        return messages

    prev_bend: int | None = None

    for chord_group in chords:
        freqs = [float(n["freq"]) for n in chord_group]
        chord_notes = _resolve_chord_notes(freqs)
        chord_start = float(chord_group[0]["start"])
        chord_bend = chord_notes[0][1]  # shared bend from bass

        # -- Chord-to-chord glide from previous chord's bend ----------------
        if prev_bend is not None and prev_bend != chord_bend:
            # Start the glide from the old bend at the new chord's onset.
            # The glide sweeps to chord_bend over glide_time seconds.
            messages.append(
                Message("pitchwheel", channel=0, pitch=prev_bend, time=chord_start)
            )
            num_steps = max(1, int(glide_time / GLIDE_STEP_INTERVAL_SECONDS))
            for step in range(1, num_steps + 1):
                t_fraction = step / num_steps
                t_absolute = chord_start + t_fraction * glide_time
                intermediate_bend = int(
                    round(prev_bend + t_fraction * (chord_bend - prev_bend))
                )
                intermediate_bend = max(-8191, min(8191, intermediate_bend))
                messages.append(
                    Message(
                        "pitchwheel",
                        channel=0,
                        pitch=intermediate_bend,
                        time=t_absolute,
                    )
                )
        else:
            # First chord or no bend change: set the target bend directly.
            messages.append(
                Message("pitchwheel", channel=0, pitch=chord_bend, time=chord_start)
            )

        # -- Note-on for every note in this chord --------------------------
        for note_data, (midi_note, _bend) in zip(chord_group, chord_notes, strict=True):
            velocity_raw = float(note_data.get("velocity", 0.8))
            amp = float(note_data.get("amp", 1.0))
            midi_velocity = max(1, min(127, int(round(velocity_raw * amp * 127))))
            start = float(note_data["start"])
            duration = float(note_data["duration"])

            messages.append(
                Message(
                    "note_on",
                    channel=0,
                    note=midi_note,
                    velocity=midi_velocity,
                    time=start,
                )
            )
            messages.append(
                Message(
                    "note_off",
                    channel=0,
                    note=midi_note,
                    velocity=0,
                    time=start + duration,
                )
            )

        prev_bend = chord_bend

    return messages


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
    messages: list[Message] = _build_mpe_config_messages() if use_mpe else []

    if use_mpe:
        messages.extend(_build_mpe_note_messages(notes, release_padding))
    else:
        global_glide_time = float(
            resolved_params.get("global_glide_time", DEFAULT_GLOBAL_GLIDE_TIME_SECONDS)
        )
        messages.extend(_build_global_bend_messages(notes, global_glide_time))

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
