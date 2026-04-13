"""MIDI import tests."""

from __future__ import annotations

from pathlib import Path

import mido
import pytest
from code_musics.midi_import import MidiImportResult, read_midi

BWV_846_PATH = Path(
    "midi_references/bach/well-tempered-clavier-i_bwv-846_(c)sankey.mid"
)


class TestSyntheticRoundTrip:
    """Build a simple MIDI file programmatically, write it, read it back."""

    def test_round_trip_triad(self, tmp_path: Path) -> None:
        midi_file = mido.MidiFile(type=1, ticks_per_beat=480)
        track = mido.MidiTrack()
        midi_file.tracks.append(track)

        track.append(mido.MetaMessage("track_name", name="piano", time=0))
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))

        # C4 at t=0, duration=0.5s = 1 beat at 120bpm = 480 ticks
        track.append(mido.Message("note_on", note=60, velocity=100, channel=0, time=0))
        track.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=480))
        # E4 at t=0.5s, duration=0.5s
        track.append(mido.Message("note_on", note=64, velocity=80, channel=0, time=0))
        track.append(mido.Message("note_off", note=64, velocity=0, channel=0, time=480))
        # G4 at t=1.0s, duration=0.5s
        track.append(mido.Message("note_on", note=67, velocity=90, channel=0, time=0))
        track.append(mido.Message("note_off", note=67, velocity=0, channel=0, time=480))

        track.append(mido.MetaMessage("end_of_track", time=0))

        midi_path = tmp_path / "triad.mid"
        midi_file.save(midi_path)

        result = read_midi(midi_path)

        assert isinstance(result, MidiImportResult)
        assert len(result.notes) == 3

        assert result.notes[0].midi_note == 60
        assert result.notes[0].velocity == 100
        assert result.notes[0].start == pytest.approx(0.0, abs=0.01)
        assert result.notes[0].duration == pytest.approx(0.5, abs=0.01)

        assert result.notes[1].midi_note == 64
        assert result.notes[1].velocity == 80
        assert result.notes[1].start == pytest.approx(0.5, abs=0.01)
        assert result.notes[1].duration == pytest.approx(0.5, abs=0.01)

        assert result.notes[2].midi_note == 67
        assert result.notes[2].velocity == 90
        assert result.notes[2].start == pytest.approx(1.0, abs=0.01)
        assert result.notes[2].duration == pytest.approx(0.5, abs=0.01)

        assert result.track_names == {0: "piano"}
        assert result.duration == pytest.approx(1.5, abs=0.01)

    def test_round_trip_simultaneous_notes(self, tmp_path: Path) -> None:
        """Simultaneous notes (chord) should all be captured."""
        midi_file = mido.MidiFile(type=1, ticks_per_beat=480)
        track = mido.MidiTrack()
        midi_file.tracks.append(track)

        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))

        # Two notes starting at the same time
        track.append(mido.Message("note_on", note=60, velocity=100, channel=0, time=0))
        track.append(mido.Message("note_on", note=64, velocity=100, channel=0, time=0))
        track.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=480))
        track.append(mido.Message("note_off", note=64, velocity=0, channel=0, time=0))

        track.append(mido.MetaMessage("end_of_track", time=0))

        midi_path = tmp_path / "chord.mid"
        midi_file.save(midi_path)

        result = read_midi(midi_path)

        assert len(result.notes) == 2
        assert result.notes[0].midi_note == 60
        assert result.notes[1].midi_note == 64
        assert result.notes[0].start == pytest.approx(0.0, abs=0.01)
        assert result.notes[1].start == pytest.approx(0.0, abs=0.01)


class TestBWV846Smoke:
    """Smoke test against the real BWV 846 reference MIDI file."""

    @pytest.fixture()
    def bwv846(self) -> MidiImportResult:
        assert BWV_846_PATH.exists(), f"Reference MIDI not found: {BWV_846_PATH}"
        return read_midi(BWV_846_PATH)

    def test_parses_without_error(self, bwv846: MidiImportResult) -> None:
        assert isinstance(bwv846, MidiImportResult)

    def test_has_many_notes(self, bwv846: MidiImportResult) -> None:
        assert len(bwv846.notes) > 100

    def test_duration_is_reasonable(self, bwv846: MidiImportResult) -> None:
        assert 60.0 < bwv846.duration < 300.0

    def test_midi_notes_in_valid_range(self, bwv846: MidiImportResult) -> None:
        for note in bwv846.notes:
            assert 0 <= note.midi_note <= 127

    def test_velocities_in_valid_range(self, bwv846: MidiImportResult) -> None:
        for note in bwv846.notes:
            assert 0 <= note.velocity <= 127

    def test_all_durations_positive(self, bwv846: MidiImportResult) -> None:
        for note in bwv846.notes:
            assert note.duration > 0


class TestTempoChanges:
    """Verify that mid-file tempo changes produce correctly adjusted timing."""

    def test_tempo_change_adjusts_timing(self, tmp_path: Path) -> None:
        midi_file = mido.MidiFile(type=1, ticks_per_beat=480)
        track = mido.MidiTrack()
        midi_file.tracks.append(track)

        # Start at 120 BPM (500000 us/beat)
        track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))

        # Note 1 at t=0, duration = 480 ticks = 1 beat = 0.5s at 120 BPM
        track.append(mido.Message("note_on", note=60, velocity=100, channel=0, time=0))
        track.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=480))

        # Change to 60 BPM (1000000 us/beat) at tick 480
        track.append(mido.MetaMessage("set_tempo", tempo=1000000, time=0))

        # Note 2 at tick 480, duration = 480 ticks = 1 beat = 1.0s at 60 BPM
        track.append(mido.Message("note_on", note=64, velocity=100, channel=0, time=0))
        track.append(mido.Message("note_off", note=64, velocity=0, channel=0, time=480))

        track.append(mido.MetaMessage("end_of_track", time=0))

        midi_path = tmp_path / "tempo_change.mid"
        midi_file.save(midi_path)

        result = read_midi(midi_path)

        assert len(result.notes) == 2

        # Note 1: starts at 0.0s, lasts 0.5s (120 BPM)
        assert result.notes[0].start == pytest.approx(0.0, abs=0.01)
        assert result.notes[0].duration == pytest.approx(0.5, abs=0.01)

        # Note 2: starts at 0.5s, lasts 1.0s (60 BPM)
        assert result.notes[1].start == pytest.approx(0.5, abs=0.01)
        assert result.notes[1].duration == pytest.approx(1.0, abs=0.01)

        assert result.duration == pytest.approx(1.5, abs=0.01)


class TestNoteOnVelocityZeroAsNoteOff:
    """Some MIDI files use note_on with velocity=0 instead of note_off."""

    def test_velocity_zero_treated_as_note_off(self, tmp_path: Path) -> None:
        midi_file = mido.MidiFile(type=1, ticks_per_beat=480)
        track = mido.MidiTrack()
        midi_file.tracks.append(track)

        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))

        # Note on
        track.append(mido.Message("note_on", note=60, velocity=100, channel=0, time=0))
        # Note off via note_on velocity=0
        track.append(mido.Message("note_on", note=60, velocity=0, channel=0, time=480))

        track.append(mido.MetaMessage("end_of_track", time=0))

        midi_path = tmp_path / "vel_zero.mid"
        midi_file.save(midi_path)

        result = read_midi(midi_path)

        assert len(result.notes) == 1
        assert result.notes[0].midi_note == 60
        assert result.notes[0].velocity == 100
        assert result.notes[0].start == pytest.approx(0.0, abs=0.01)
        assert result.notes[0].duration == pytest.approx(0.5, abs=0.01)

    def test_mixed_noteoff_styles(self, tmp_path: Path) -> None:
        """File uses both real note_off and velocity=0 note_on."""
        midi_file = mido.MidiFile(type=1, ticks_per_beat=480)
        track = mido.MidiTrack()
        midi_file.tracks.append(track)

        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))

        # Note 1: real note_off
        track.append(mido.Message("note_on", note=60, velocity=100, channel=0, time=0))
        track.append(mido.Message("note_off", note=60, velocity=0, channel=0, time=480))
        # Note 2: velocity=0 note_off
        track.append(mido.Message("note_on", note=64, velocity=80, channel=0, time=0))
        track.append(mido.Message("note_on", note=64, velocity=0, channel=0, time=480))

        track.append(mido.MetaMessage("end_of_track", time=0))

        midi_path = tmp_path / "mixed.mid"
        midi_file.save(midi_path)

        result = read_midi(midi_path)

        assert len(result.notes) == 2
        assert result.notes[0].midi_note == 60
        assert result.notes[0].velocity == 100
        assert result.notes[1].midi_note == 64
        assert result.notes[1].velocity == 80
