"""Shared MPE and MIDI utility functions for instrument engine rendering."""

from __future__ import annotations

import logging
import math
from typing import Any

from mido import Message

logger: logging.Logger = logging.getLogger(__name__)

BEND_RANGE_SEMITONES = 24.0
MAX_MPE_CHANNELS = 15  # channels 1-15; channel 0 is MPE manager/global

CHORD_GROUP_TOLERANCE_SECONDS = 0.1  # notes within this window are one chord
GLIDE_STEP_INTERVAL_SECONDS = 0.005  # 5ms steps (~200 Hz update rate)
DEFAULT_GLOBAL_GLIDE_TIME_SECONDS = 0.4

CC_UPDATE_INTERVAL_SECONDS = 0.01  # 10ms = 100 Hz update rate for CC automation


def build_mpe_config_messages(
    num_member_channels: int = MAX_MPE_CHANNELS,
    bend_range_semitones: int = int(BEND_RANGE_SEMITONES),
) -> list[Message]:
    """Build MCM (MPE Configuration Message) + per-channel bend range RPNs.

    MCM declares an MPE Lower Zone so that per-channel pitch bend is routed
    per-note instead of globally.  RPN 0 on each member channel sets the
    pitch bend range to match ``BEND_RANGE_SEMITONES``.
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


def resolve_glide_bend(target_midi_note: int, freq_hz: float) -> int:
    """Compute pitch bend value to reach *freq_hz* from a given MIDI note.

    This is the glide counterpart to ``resolve_note_and_bend``: given a
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


def resolve_note_and_bend(freq_hz: float) -> tuple[int, int]:
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


def resolve_chord_notes(freqs: list[float]) -> list[tuple[int, int]]:
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
    bass_midi, bass_bend = resolve_note_and_bend(bass_freq)

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


def group_notes_into_chords(
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


def build_mpe_note_messages(
    notes: list[dict[str, Any]], release_padding: float
) -> list[Message]:
    """Build per-note MPE MIDI messages.

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

        midi_note, bend_value = resolve_note_and_bend(freq)

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
                initial_bend = resolve_glide_bend(midi_note, glide_from_hz)

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


def build_global_bend_messages(
    notes: list[dict[str, Any]], glide_time: float
) -> list[Message]:
    """Build MIDI messages for global-bend chord mode (non-MPE).

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
    chords = group_notes_into_chords(notes)

    if not chords:
        return messages

    prev_bend: int | None = None

    for chord_group in chords:
        freqs = [float(n["freq"]) for n in chord_group]
        chord_notes = resolve_chord_notes(freqs)
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
        for note_data, (midi_note, _) in zip(chord_group, chord_notes, strict=True):
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


def build_cc_messages(
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
