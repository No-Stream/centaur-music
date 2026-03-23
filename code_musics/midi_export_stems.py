"""Stem-note collection and MIDI file writers for MIDI export."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import mido

from code_musics.automation import AutomationSpec
from code_musics.midi_export_types import (
    MONO_BEND_CHANNEL,
    MPE_GLOBAL_CHANNEL,
    MPE_MEMBER_CHANNELS,
    POLY_BEND_CHANNELS,
    MidiBundleExportSpec,
    MidiStemExportResult,
    MidiStemFormat,
    MidiStemNote,
    TuningAnalysisResult,
)
from code_musics.score import NoteEvent, Score


def collect_stem_notes(
    score: Score,
    *,
    window_start_seconds: float | None = None,
    window_end_seconds: float | None = None,
) -> dict[str, list[MidiStemNote]]:
    export_start_seconds = 0.0 if window_start_seconds is None else window_start_seconds
    export_end_seconds = math.inf if window_end_seconds is None else window_end_seconds
    if export_end_seconds <= export_start_seconds:
        raise ValueError("window_end_seconds must be greater than window_start_seconds")

    timing_offsets = score.resolve_timing_offsets()
    stem_notes: dict[str, list[MidiStemNote]] = {}
    for voice_name, voice in score.voices.items():
        if not voice_name:
            raise ValueError("all exported voices must have non-empty names")
        if any(_has_pitch_ratio_automation(spec) for spec in voice.automation):
            raise ValueError(
                f"voice {voice_name!r} uses pitch_ratio automation, which MIDI export does not support yet"
            )
        resolved_voice_notes: list[MidiStemNote] = []
        for note_index, note in enumerate(voice.notes):
            _validate_note_is_exportable(
                voice_name=voice_name,
                note_index=note_index,
                note=note,
            )
            resolved_start_seconds = note.start + timing_offsets.get(
                (voice_name, note_index), 0.0
            )
            resolved_end_seconds = resolved_start_seconds + note.duration
            clipped_start_seconds = max(export_start_seconds, resolved_start_seconds)
            clipped_end_seconds = min(export_end_seconds, resolved_end_seconds)
            if clipped_end_seconds <= clipped_start_seconds:
                continue
            exported_start_seconds = clipped_start_seconds - export_start_seconds
            duration_seconds = clipped_end_seconds - clipped_start_seconds
            resolved_voice_notes.append(
                MidiStemNote(
                    voice_name=voice_name,
                    note_index=note_index,
                    start_seconds=exported_start_seconds,
                    duration_seconds=duration_seconds,
                    end_seconds=exported_start_seconds + duration_seconds,
                    freq_hz=score._resolve_freq(note),
                    velocity=_resolve_midi_velocity(note),
                    label=note.label,
                )
            )
        stem_notes[voice_name] = sorted(
            resolved_voice_notes,
            key=lambda item: (item.start_seconds, item.end_seconds, item.note_index),
        )
    return stem_notes


def _validate_note_is_exportable(
    *,
    voice_name: str,
    note_index: int,
    note: NoteEvent,
) -> None:
    if note.pitch_motion is not None:
        raise ValueError(
            f"voice {voice_name!r} note {note_index} uses pitch_motion, which MIDI export does not support yet"
        )
    if note.automation is not None and any(
        _has_pitch_ratio_automation(spec) for spec in note.automation
    ):
        raise ValueError(
            f"voice {voice_name!r} note {note_index} uses pitch_ratio automation, which MIDI export does not support yet"
        )


def _has_pitch_ratio_automation(spec: AutomationSpec) -> bool:
    return spec.target.kind == "pitch_ratio"


def _resolve_midi_velocity(note: NoteEvent) -> int:
    resolved_amp = 1.0 if note.amp is None else note.amp
    return int(round(max(1.0, min(127.0, resolved_amp * note.velocity * 96.0))))


def write_stem_files(
    *,
    stems_dir: Path,
    stem_notes: dict[str, list[MidiStemNote]],
    tuning_analysis: TuningAnalysisResult,
    spec: MidiBundleExportSpec,
) -> list[MidiStemExportResult]:
    stem_results: list[MidiStemExportResult] = []
    for voice_name, voice_notes in stem_notes.items():
        emitted_files: dict[str, str] = {}
        max_simultaneous_notes = _max_simultaneous_notes(voice_notes)
        for stem_format in spec.stem_formats:
            emitted_files[stem_format] = _write_stem_format(
                stem_format=stem_format,
                stems_dir=stems_dir,
                voice_name=voice_name,
                voice_notes=voice_notes,
                max_simultaneous_notes=max_simultaneous_notes,
                tuning_analysis=tuning_analysis,
                spec=spec,
            )
        stem_results.append(
            MidiStemExportResult(
                voice_name=voice_name,
                note_count=len(voice_notes),
                emitted_files=emitted_files,
                max_simultaneous_notes=max_simultaneous_notes,
            )
        )
    return stem_results


def _write_stem_format(
    *,
    stem_format: MidiStemFormat,
    stems_dir: Path,
    voice_name: str,
    voice_notes: list[MidiStemNote],
    max_simultaneous_notes: int,
    tuning_analysis: TuningAnalysisResult,
    spec: MidiBundleExportSpec,
) -> str:
    if stem_format == "scala":
        output_path = stems_dir / f"{voice_name}_scala.mid"
        _write_plain_midi_stem(
            output_path=output_path,
            notes=voice_notes,
            midi_note_resolver=lambda note: resolve_shared_tuning_midi_note(
                note=note,
                tuning_analysis=tuning_analysis,
            ),
            ticks_per_beat=spec.ticks_per_beat,
            export_bpm=spec.export_bpm,
            track_name=f"{voice_name} scala",
        )
        return str(output_path)

    if stem_format == "tun":
        output_path = stems_dir / f"{voice_name}_tun.mid"
        _write_plain_midi_stem(
            output_path=output_path,
            notes=voice_notes,
            midi_note_resolver=lambda note: resolve_shared_tuning_midi_note(
                note=note,
                tuning_analysis=tuning_analysis,
            ),
            ticks_per_beat=spec.ticks_per_beat,
            export_bpm=spec.export_bpm,
            track_name=f"{voice_name} tun",
        )
        return str(output_path)

    if stem_format == "mpe_48st":
        if max_simultaneous_notes > len(MPE_MEMBER_CHANNELS):
            raise ValueError(
                f"voice {voice_name!r} requires {max_simultaneous_notes} simultaneous notes but "
                f"mpe_48st supports at most {len(MPE_MEMBER_CHANNELS)}"
            )
        output_path = stems_dir / f"{voice_name}_mpe_48st.mid"
        _write_bend_midi_stem(
            output_path=output_path,
            notes=voice_notes,
            ticks_per_beat=spec.ticks_per_beat,
            export_bpm=spec.export_bpm,
            bend_range_semitones=48.0,
            channels=MPE_MEMBER_CHANNELS,
            global_channel=MPE_GLOBAL_CHANNEL,
            mono_mode=False,
            track_name=f"{voice_name} mpe",
        )
        return str(output_path)

    if stem_format == "poly_bend_12st":
        if max_simultaneous_notes > len(POLY_BEND_CHANNELS):
            raise ValueError(
                f"voice {voice_name!r} requires {max_simultaneous_notes} simultaneous notes but "
                f"poly_bend_12st supports at most {len(POLY_BEND_CHANNELS)}"
            )
        output_path = stems_dir / f"{voice_name}_poly_bend_12st.mid"
        _write_bend_midi_stem(
            output_path=output_path,
            notes=voice_notes,
            ticks_per_beat=spec.ticks_per_beat,
            export_bpm=spec.export_bpm,
            bend_range_semitones=12.0,
            channels=POLY_BEND_CHANNELS,
            global_channel=None,
            mono_mode=False,
            track_name=f"{voice_name} poly bend",
        )
        return str(output_path)

    if not _notes_are_monophonic(voice_notes):
        raise ValueError(
            f"voice {voice_name!r} contains overlapping notes and cannot be exported as mono_bend_12st"
        )
    output_path = stems_dir / f"{voice_name}_mono_bend_12st.mid"
    _write_bend_midi_stem(
        output_path=output_path,
        notes=voice_notes,
        ticks_per_beat=spec.ticks_per_beat,
        export_bpm=spec.export_bpm,
        bend_range_semitones=12.0,
        channels=(MONO_BEND_CHANNEL,),
        global_channel=None,
        mono_mode=True,
        track_name=f"{voice_name} mono bend",
    )
    return str(output_path)


def resolve_shared_tuning_midi_note(
    *,
    note: MidiStemNote,
    tuning_analysis: TuningAnalysisResult,
) -> int:
    total_cents = 1200.0 * math.log2(
        note.freq_hz / tuning_analysis.reference_frequency_hz
    )
    scale_size = len(tuning_analysis.pitch_class_cents)
    base_period_index = math.floor(total_cents / tuning_analysis.period_cents)
    best_period_index = 0
    best_degree_index = 0
    best_error_cents = math.inf

    for period_adjustment in (-1, 0, 1):
        candidate_period_index = base_period_index + period_adjustment
        period_base_cents = candidate_period_index * tuning_analysis.period_cents
        for degree_index, pitch_class_cents in enumerate(
            tuning_analysis.pitch_class_cents
        ):
            candidate_total_cents = period_base_cents + pitch_class_cents
            error_cents = abs(candidate_total_cents - total_cents)
            if error_cents < best_error_cents:
                best_error_cents = error_cents
                best_period_index = candidate_period_index
                best_degree_index = degree_index

    midi_note = (
        tuning_analysis.reference_midi_note
        + (best_period_index * scale_size)
        + best_degree_index
    )
    if not 0 <= midi_note <= 127:
        raise ValueError(
            f"shared tuning mapping produced out-of-range MIDI note {midi_note}"
        )
    return midi_note


def _write_plain_midi_stem(
    *,
    output_path: Path,
    notes: list[MidiStemNote],
    midi_note_resolver: Callable[[MidiStemNote], int],
    ticks_per_beat: int,
    export_bpm: float,
    track_name: str,
) -> None:
    midi_file = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    _append_common_track_header(
        track=track, track_name=track_name, export_bpm=export_bpm
    )

    events: list[tuple[int, int, mido.Message]] = []
    for note in notes:
        midi_note = int(midi_note_resolver(note))
        start_tick = _seconds_to_ticks(
            seconds=note.start_seconds,
            ticks_per_beat=ticks_per_beat,
            export_bpm=export_bpm,
        )
        end_tick = _seconds_to_ticks(
            seconds=note.end_seconds,
            ticks_per_beat=ticks_per_beat,
            export_bpm=export_bpm,
        )
        events.append(
            (
                start_tick,
                1,
                mido.Message(
                    "note_on",
                    channel=0,
                    note=midi_note,
                    velocity=note.velocity,
                    time=0,
                ),
            )
        )
        events.append(
            (
                end_tick,
                0,
                mido.Message(
                    "note_off",
                    channel=0,
                    note=midi_note,
                    velocity=0,
                    time=0,
                ),
            )
        )

    _append_sorted_events(track=track, events=events)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi_file.save(output_path)


def _write_bend_midi_stem(
    *,
    output_path: Path,
    notes: list[MidiStemNote],
    ticks_per_beat: int,
    export_bpm: float,
    bend_range_semitones: float,
    channels: tuple[int, ...],
    global_channel: int | None,
    mono_mode: bool,
    track_name: str,
) -> None:
    midi_file = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    _append_common_track_header(
        track=track, track_name=track_name, export_bpm=export_bpm
    )
    if global_channel is not None:
        _append_pitch_bend_range_setup(
            track=track,
            channels=(global_channel, *channels),
            bend_range_semitones=bend_range_semitones,
        )
    else:
        _append_pitch_bend_range_setup(
            track=track,
            channels=channels,
            bend_range_semitones=bend_range_semitones,
        )

    events: list[tuple[int, int, mido.Message]] = []
    available_channels = list(channels)
    active_channels: dict[int, int] = {}
    release_events: list[tuple[int, int]] = sorted(
        [
            (
                _seconds_to_ticks(
                    seconds=note.end_seconds,
                    ticks_per_beat=ticks_per_beat,
                    export_bpm=export_bpm,
                ),
                index,
            )
            for index, note in enumerate(notes)
        ],
        key=lambda item: (item[0], item[1]),
    )
    release_index = 0

    for note_index, note in enumerate(notes):
        start_tick = _seconds_to_ticks(
            seconds=note.start_seconds,
            ticks_per_beat=ticks_per_beat,
            export_bpm=export_bpm,
        )
        while (
            release_index < len(release_events)
            and release_events[release_index][0] <= start_tick
        ):
            _, released_note_index = release_events[release_index]
            if released_note_index in active_channels:
                available_channels.append(active_channels.pop(released_note_index))
            release_index += 1

        if mono_mode:
            if active_channels:
                raise ValueError("mono bend export cannot handle overlapping notes")
            channel = channels[0]
        else:
            if not available_channels:
                raise ValueError("not enough MIDI channels for bend-based export")
            channel = min(available_channels)
            available_channels.remove(channel)
        active_channels[note_index] = channel

        end_tick = _seconds_to_ticks(
            seconds=note.end_seconds,
            ticks_per_beat=ticks_per_beat,
            export_bpm=export_bpm,
        )
        midi_note, bend_value = _resolve_nearest_12tet_note_and_bend(
            freq_hz=note.freq_hz,
            bend_range_semitones=bend_range_semitones,
        )
        events.append(
            (
                start_tick,
                1,
                mido.Message(
                    "pitchwheel",
                    channel=channel,
                    pitch=bend_value,
                    time=0,
                ),
            )
        )
        events.append(
            (
                start_tick,
                2,
                mido.Message(
                    "note_on",
                    channel=channel,
                    note=midi_note,
                    velocity=note.velocity,
                    time=0,
                ),
            )
        )
        events.append(
            (
                end_tick,
                0,
                mido.Message(
                    "note_off",
                    channel=channel,
                    note=midi_note,
                    velocity=0,
                    time=0,
                ),
            )
        )

    _append_sorted_events(track=track, events=events)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi_file.save(output_path)


def _append_common_track_header(
    *,
    track: mido.MidiTrack,
    track_name: str,
    export_bpm: float,
) -> None:
    track.append(mido.MetaMessage("track_name", name=track_name, time=0))
    track.append(
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(export_bpm), time=0)
    )


def _append_pitch_bend_range_setup(
    *,
    track: mido.MidiTrack,
    channels: tuple[int, ...],
    bend_range_semitones: float,
) -> None:
    coarse_value = int(round(bend_range_semitones))
    for channel in channels:
        track.append(
            mido.Message(
                "control_change", channel=channel, control=101, value=0, time=0
            )
        )
        track.append(
            mido.Message(
                "control_change", channel=channel, control=100, value=0, time=0
            )
        )
        track.append(
            mido.Message(
                "control_change",
                channel=channel,
                control=6,
                value=coarse_value,
                time=0,
            )
        )
        track.append(
            mido.Message("control_change", channel=channel, control=38, value=0, time=0)
        )
        track.append(
            mido.Message(
                "control_change",
                channel=channel,
                control=101,
                value=127,
                time=0,
            )
        )
        track.append(
            mido.Message(
                "control_change",
                channel=channel,
                control=100,
                value=127,
                time=0,
            )
        )


def _append_sorted_events(
    *,
    track: mido.MidiTrack,
    events: list[tuple[int, int, mido.Message]],
) -> None:
    current_tick = 0
    for absolute_tick, _priority, message in sorted(
        events,
        key=lambda item: (item[0], item[1], str(item[2])),
    ):
        message.time = absolute_tick - current_tick
        track.append(message)
        current_tick = absolute_tick
    track.append(mido.MetaMessage("end_of_track", time=0))


def _seconds_to_ticks(
    *,
    seconds: float,
    ticks_per_beat: int,
    export_bpm: float,
) -> int:
    beats = seconds / (60.0 / export_bpm)
    return int(round(beats * ticks_per_beat))


def _resolve_nearest_12tet_note_and_bend(
    *,
    freq_hz: float,
    bend_range_semitones: float,
) -> tuple[int, int]:
    midi_note_float = 69.0 + (12.0 * math.log2(freq_hz / 440.0))
    midi_note = int(round(midi_note_float))
    if not 0 <= midi_note <= 127:
        raise ValueError(f"frequency {freq_hz} resolves to out-of-range MIDI note")
    residual_semitones = midi_note_float - midi_note
    if abs(residual_semitones) > bend_range_semitones:
        raise ValueError("required pitch bend exceeds the configured bend range")
    normalized_bend = residual_semitones / bend_range_semitones
    bend_value = int(round(max(-1.0, min(1.0, normalized_bend)) * 8191.0))
    return midi_note, bend_value


def _notes_are_monophonic(notes: list[MidiStemNote]) -> bool:
    if not notes:
        return True
    previous_end = notes[0].end_seconds
    for note in notes[1:]:
        if note.start_seconds < previous_end - 1e-9:
            return False
        previous_end = max(previous_end, note.end_seconds)
    return True


def _max_simultaneous_notes(notes: list[MidiStemNote]) -> int:
    events: list[tuple[float, int]] = []
    for note in notes:
        events.append((note.start_seconds, 1))
        events.append((note.end_seconds, -1))
    active_count = 0
    max_active = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active_count += delta
        max_active = max(max_active, active_count)
    return max_active
