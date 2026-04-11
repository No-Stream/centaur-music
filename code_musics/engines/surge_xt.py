"""Surge XT instrument engine -- renders voices through pedalboard's VSTi hosting."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from mido import Message

from code_musics.synth import SAMPLE_RATE, has_external_plugin, load_external_plugin

logger: logging.Logger = logging.getLogger(__name__)

BEND_RANGE_SEMITONES = 48.0
MAX_MPE_CHANNELS = 15  # channels 1-15; channel 0 is MPE manager


def _resolve_note_and_bend(freq_hz: float) -> tuple[int, int]:
    """Find nearest MIDI note and pitch bend value for an arbitrary frequency.

    Uses a 48-semitone pitch bend range so that even extreme microtonal
    deviations from 12-TET are representable.
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
    total_duration
        Total duration of the voice in seconds (the latest note-off time).
    sample_rate
        Sample rate for rendering.
    params
        Optional engine params.  Recognised keys:

        - ``preset_path`` -- path to a ``.vstpreset`` or ``.fxp`` file
        - ``raw_state`` -- serialised plugin state (bytes)
        - ``tail_seconds`` -- extra render time after last note-off (default 2.0)
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

    tail_seconds = float(resolved_params.get("tail_seconds", 2.0))
    render_duration = total_duration + tail_seconds

    # -- Build MIDI messages -------------------------------------------------
    messages: list[Message] = []

    # Set pitch bend range on member channels via RPN 0 (MSB=0, LSB=0).
    n_channels_needed = min(len(notes), MAX_MPE_CHANNELS)
    for ch in range(1, n_channels_needed + 1):
        messages.extend(
            [
                Message("control_change", channel=ch, control=101, value=0, time=0.0),
                Message("control_change", channel=ch, control=100, value=0, time=0.0),
                Message(
                    "control_change",
                    channel=ch,
                    control=6,
                    value=int(BEND_RANGE_SEMITONES),
                    time=0.0,
                ),
                Message("control_change", channel=ch, control=38, value=0, time=0.0),
            ]
        )

    # Assign notes to MPE member channels 1-15, preferring a channel whose
    # previous note has already ended.  Falls back to round-robin when all
    # 15 channels are occupied (with a warning).
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
                "MPE channel collision: >%d overlapping notes at t=%.3f",
                MAX_MPE_CHANNELS,
                start,
            )
        channel_free_at[channel] = start + duration

        midi_note, bend_value = _resolve_note_and_bend(freq)

        # Pitch bend BEFORE note-on so the plugin sees the tuning first.
        messages.append(
            Message("pitchwheel", channel=channel, pitch=bend_value, time=start)
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
        messages.append(
            Message(
                "note_off",
                channel=channel,
                note=midi_note,
                velocity=0,
                time=start + duration,
            )
        )

    messages.sort(key=lambda m: m.time)  # type: ignore[reportAttributeAccessIssue]

    logger.info(
        "Rendering %d notes through Surge XT (%.1fs + %.1fs tail)",
        len(notes),
        total_duration,
        tail_seconds,
    )

    audio = plugin(
        messages, sample_rate=sample_rate, duration=render_duration, num_channels=2
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
