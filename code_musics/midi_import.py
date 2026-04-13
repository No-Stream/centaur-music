"""Lightweight MIDI file reader that extracts note events with absolute timing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import mido

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_TEMPO = 500000  # microseconds per beat (120 BPM)


@dataclass(frozen=True)
class MidiNote:
    start: float  # absolute seconds
    duration: float  # seconds
    midi_note: int  # 0-127
    velocity: int  # 0-127
    channel: int  # 0-15
    track: int  # track index


@dataclass(frozen=True)
class MidiImportResult:
    notes: list[MidiNote]
    track_names: dict[int, str]  # track index -> name from track_name meta events
    duration: float  # total duration in seconds


def read_midi(path: str | Path) -> MidiImportResult:
    """Read a MIDI file and return structured note data with absolute timing.

    For Type 1 (multi-track) files, tempo changes from any track are merged
    into a single global tempo map and applied when converting ticks to
    seconds on every track.  This matches the standard MIDI convention where
    tempo events on the conductor track govern all tracks.
    """
    midi_file = mido.MidiFile(path)
    ticks_per_beat = midi_file.ticks_per_beat

    tempo_map = _build_global_tempo_map(midi_file)

    all_notes: list[MidiNote] = []
    track_names: dict[int, str] = {}

    for track_index, track in enumerate(midi_file.tracks):
        notes, track_name = _process_track(
            track=track,
            track_index=track_index,
            ticks_per_beat=ticks_per_beat,
            tempo_map=tempo_map,
        )
        all_notes.extend(notes)
        if track_name is not None:
            track_names[track_index] = track_name

    all_notes.sort(key=lambda n: (n.start, n.midi_note))

    total_duration = max((n.start + n.duration for n in all_notes), default=0.0)

    return MidiImportResult(
        notes=all_notes,
        track_names=track_names,
        duration=total_duration,
    )


def _build_global_tempo_map(midi_file: mido.MidiFile) -> list[tuple[int, int]]:
    """Collect tempo events from all tracks into a sorted (abs_tick, tempo) list.

    For Type 1 files the conductor track (usually track 0) carries the tempo
    events, but we scan all tracks defensively.
    """
    tempo_events: list[tuple[int, int]] = []
    for track in midi_file.tracks:
        abs_tick = 0
        for message in track:
            abs_tick += message.time
            if message.is_meta and message.type == "set_tempo":
                tempo_events.append((abs_tick, message.tempo))

    if not tempo_events:
        return [(0, DEFAULT_TEMPO)]

    tempo_events.sort(key=lambda pair: pair[0])

    # Ensure there's an entry at tick 0.
    if tempo_events[0][0] != 0:
        tempo_events.insert(0, (0, DEFAULT_TEMPO))

    return tempo_events


def _tick_to_seconds(
    abs_tick: int,
    ticks_per_beat: int,
    tempo_map: list[tuple[int, int]],
) -> float:
    """Convert an absolute tick position to seconds using a global tempo map."""
    seconds = 0.0
    prev_tick = 0
    prev_tempo = tempo_map[0][1]

    for map_tick, map_tempo in tempo_map:
        if map_tick >= abs_tick:
            break
        if map_tick > prev_tick:
            delta_ticks = map_tick - prev_tick
            seconds += mido.tick2second(delta_ticks, ticks_per_beat, prev_tempo)
            prev_tick = map_tick
        prev_tempo = map_tempo

    remaining_ticks = abs_tick - prev_tick
    if remaining_ticks > 0:
        seconds += mido.tick2second(remaining_ticks, ticks_per_beat, prev_tempo)

    return seconds


def _process_track(
    *,
    track: mido.MidiTrack,
    track_index: int,
    ticks_per_beat: int,
    tempo_map: list[tuple[int, int]],
) -> tuple[list[MidiNote], str | None]:
    """Process a single MIDI track, returning notes and optional track name."""
    track_name: str | None = None
    # Pending note-ons: (note_number, channel) -> (abs_tick, velocity)
    pending: dict[tuple[int, int], tuple[int, int]] = {}
    completed_notes: list[MidiNote] = []

    current_tick = 0

    for message in track:
        current_tick += message.time

        if message.is_meta:
            if message.type == "track_name":
                track_name = message.name
            continue

        is_note_on = message.type == "note_on" and message.velocity > 0
        is_note_off = message.type == "note_off" or (
            message.type == "note_on" and message.velocity == 0
        )

        if is_note_on:
            key = (message.note, message.channel)
            if key in pending:
                _close_note(
                    pending,
                    key,
                    current_tick,
                    track_index,
                    ticks_per_beat,
                    tempo_map,
                    completed_notes,
                )
            pending[key] = (current_tick, message.velocity)

        elif is_note_off:
            key = (message.note, message.channel)
            if key in pending:
                _close_note(
                    pending,
                    key,
                    current_tick,
                    track_index,
                    ticks_per_beat,
                    tempo_map,
                    completed_notes,
                )

    # Close any notes still pending at end of track.
    for key in list(pending):
        _close_note(
            pending,
            key,
            current_tick,
            track_index,
            ticks_per_beat,
            tempo_map,
            completed_notes,
        )
        logger.warning(
            f"note {key[0]} on channel {key[1]} in track {track_index} "
            f"was never closed; used end-of-track time"
        )

    return completed_notes, track_name


def _close_note(
    pending: dict[tuple[int, int], tuple[int, int]],
    key: tuple[int, int],
    end_tick: int,
    track_index: int,
    ticks_per_beat: int,
    tempo_map: list[tuple[int, int]],
    completed_notes: list[MidiNote],
) -> None:
    start_tick, velocity = pending.pop(key)
    start_seconds = _tick_to_seconds(start_tick, ticks_per_beat, tempo_map)
    end_seconds = _tick_to_seconds(end_tick, ticks_per_beat, tempo_map)
    note_duration = end_seconds - start_seconds
    if note_duration > 0:
        completed_notes.append(
            MidiNote(
                start=start_seconds,
                duration=note_duration,
                midi_note=key[0],
                velocity=velocity,
                channel=key[1],
                track=track_index,
            )
        )
