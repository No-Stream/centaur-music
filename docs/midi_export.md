# MIDI Export

Score-backed pieces can now export a MIDI bundle with shared tuning files plus per-voice MIDI stems.

## Commands

- `make midi PIECE=<name>` exports a full bundle
- `make midi-snippet PIECE=<name> AT=<timestamp> WINDOW=<seconds>` exports a centered snippet bundle
- `make midi-window PIECE=<name> START=<timestamp> DUR=<seconds>` exports an exact-window bundle
- `MIDI_FORMATS=scala,tun` optionally limits which stem formats are requested

## Bundle Layout

A bundle contains:

- `manifest.json`
- `README.md`
- `tuning/*.scl`
- `tuning/*.kbm`
- `tuning/*.tun`
- `stems/*_scala.mid`
- `stems/*_tun.mid`
- `stems/*_mpe_48st.mid`
- `stems/*_poly_bend_12st.mid`
- `stems/*_mono_bend_12st.mid` for eligible monophonic voices

## Tuning Modes

The exporter classifies each piece as one of:

- `static_periodic_tuning`: shared `SCL + KBM + TUN` are treated as exact
- `exact_note_tuning`: bend-based MIDI stems are exact, while shared tuning files are emitted as convenience approximations and suffixed with `WARNING_APPROX`

Automatic classification is conservative:

- small pitch-class inventories default to `static_periodic_tuning`
- larger inventories default to `exact_note_tuning`

## Timing and Pitch Rules

- MIDI timing is encoded at `60 BPM`, so `1 beat = 1 second`
- `*_scala.mid` and `*_tun.mid` are plain note stems intended to be used with the shared tuning files
- `*_mpe_48st.mid` uses per-note-channel pitch bend with a `48` semitone bend range
- `*_poly_bend_12st.mid` uses channel-per-note pitch bend with a `12` semitone bend range
- `*_mono_bend_12st.mid` uses a single bend channel and is only valid for non-overlapping stems

Requested formats fail fast:

- if a requested bend-based format exceeds channel/polyphony constraints, export raises
- if `mono_bend_12st` is requested for overlapping material, export raises
- use `MIDI_FORMATS=...` or `--midi-formats ...` to request only the formats you want

## Current Limitations

- `pitch_motion` is not exported yet
- `pitch_ratio` automation is not exported yet
- those cases fail fast rather than silently writing misleading MIDI

## Python API

The types below are frozen dataclasses exposed from
`code_musics.midi_export_types`. The entry points live in
`code_musics.midi_export` and `code_musics.midi_export_stems`.

### `export_midi_bundle(score, out_dir, *, spec, window_start_seconds=None, window_end_seconds=None) -> MidiBundleExportResult`

Top-level bundle entry point. Writes `manifest.json`, `README.md`,
`tuning/`, and `stems/` under `out_dir`.

- `score: Score` — the score to export.
- `out_dir: str | Path` — destination bundle directory (created if missing).
- `spec: MidiBundleExportSpec` — configuration (see below).
- `window_start_seconds: float | None` — optional window start in score seconds; notes before this are clipped.
- `window_end_seconds: float | None` — optional window end in score seconds; notes after this are clipped. Must be greater than `window_start_seconds` when both are set.

### `MidiBundleExportSpec`

Configuration for a MIDI bundle export.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `piece_name` | `str` | required | Name of the piece; recorded in the manifest. |
| `output_name` | `str` | required | Name of the bundle; recorded in the manifest. |
| `ticks_per_beat` | `int` | `960` | MIDI resolution. Must be positive. |
| `export_bpm` | `float` | `60.0` | Tempo written into the MIDI files. At the default, `1 beat = 1 second`. Must be positive. |
| `tuning_mode` | `Literal["auto", "static_periodic_tuning", "exact_note_tuning"]` | `"auto"` | Override the shared-tuning classifier. `"auto"` picks based on pitch-class inventory size. |
| `reference_midi_note` | `int` | `60` | MIDI note mapped to the tuning reference. Must be in `[0, 127]`. |
| `reference_frequency_hz` | `float \| None` | `None` | Optional explicit reference pitch. Falls back to a score-derived pitch when `None`. Must be positive when set. |
| `period_ratio` | `float` | `2.0` | Tuning period as a frequency ratio (octave = `2.0`). Must be `> 1.0`. |
| `static_scale_max_size` | `int` | `24` | Max pitch-class count before auto-classification drops from `static_periodic_tuning` into `exact_note_tuning`. |
| `quantization_cents` | `float` | `0.01` | Cents granularity for pitch-class deduplication. Must be positive. |
| `stem_formats` | `tuple[MidiStemFormat, ...]` | all five formats | Which stem formats to emit. Duplicates are dropped; unknown formats raise. |
| `chromatic_scl` | `bool` | `True` | Also emit a chromatic-fill SCL/KBM pair for DAW workflows. |
| `chromatic_warning_threshold_cents` | `float` | `35.0` | Per-slot error above which the chromatic fill emits a warning. Must be positive. |

`MidiStemFormat` is `Literal["scala", "tun", "mpe_48st", "poly_bend_12st", "mono_bend_12st"]`.

### `MidiBundleManifest`

Serialized to `manifest.json` via `to_dict()`.

| Field | Type | Description |
| --- | --- | --- |
| `schema_version` | `int` | Manifest schema version (currently `1`). |
| `piece_name` | `str` | From the export spec. |
| `output_name` | `str` | From the export spec. |
| `tuning_mode` | `Literal["static_periodic_tuning", "exact_note_tuning"]` | Resolved shared-tuning mode. |
| `shared_tuning_status` | `Literal["exact", "approximate"]` | Whether the shared `SCL + KBM + TUN` files represent the piece exactly or are convenience approximations. |
| `warning_suffix` | `str \| None` | Suffix applied to approximate tuning filenames (e.g. `WARNING_APPROX`), else `None`. |
| `requested_stem_formats` | `list[MidiStemFormat]` | Stem formats actually emitted. |
| `timing_encoding` | `dict[str, float \| int]` | `ticks_per_beat`, `export_bpm`, and derived `seconds_per_beat`. |
| `tuning` | `dict[str, Any]` | Resolved tuning summary: `period_ratio`, `period_cents`, `reference_midi_note`, `reference_frequency_hz`, `pitch_class_cents`, `scale_entry_cents`, `quantization_cents`. |
| `stem_exports` | `list[MidiStemExportResult]` | Per-voice export results. |
| `tuning_files` | `dict[str, str]` | Map from tuning-file kind (e.g. `scl`, `kbm`, `tun`, `chromatic_scl`, `chromatic_kbm`) to path. |
| `chromatic_tuning` | `dict[str, Any] \| None` | Chromatic-fill summary (slot assignments and warnings), or `{"skipped_reason": ...}` when skipped, or `None` when `chromatic_scl=False`. |
| `warnings` | `list[str]` | Non-fatal warnings collected during export. |

### `MidiStemExportResult`

One entry per voice in `MidiBundleManifest.stem_exports`.

| Field | Type | Description |
| --- | --- | --- |
| `voice_name` | `str` | Voice name from the score. |
| `note_count` | `int` | Number of notes exported for this voice (after window clipping). |
| `emitted_files` | `dict[str, str]` | Map from `MidiStemFormat` to the written `.mid` path. |
| `max_simultaneous_notes` | `int` | Peak concurrent-note count, used to validate bend-format channel budgets. |

### `MidiBundleExportResult`

Return value from `export_midi_bundle`.

| Field | Type | Description |
| --- | --- | --- |
| `bundle_dir` | `Path` | Top-level bundle directory. |
| `manifest_path` | `Path` | Path to `manifest.json`. |
| `readme_path` | `Path` | Path to the generated `README.md`. |
| `tuning_dir` | `Path` | Path to the `tuning/` subdirectory. |
| `stems_dir` | `Path` | Path to the `stems/` subdirectory. |
| `manifest` | `MidiBundleManifest` | In-memory manifest (matches the JSON file). |

### Stem-builder entry points

Lower-level helpers from `code_musics.midi_export_stems` for code that wants
to drive stem construction directly rather than going through
`export_midi_bundle`.

#### `collect_stem_notes(score, *, window_start_seconds=None, window_end_seconds=None) -> dict[str, list[MidiStemNote]]`

Resolves every exportable voice into a list of `MidiStemNote` entries,
applying score timing offsets and clipping to the optional window. Raises
`ValueError` if a voice uses `pitch_motion` or `pitch_ratio` automation
(neither is supported by MIDI export yet), or if `window_end_seconds` is
not greater than `window_start_seconds`.

#### `write_stem_files(*, stems_dir, stem_notes, tuning_analysis, spec) -> list[MidiStemExportResult]`

Writes every requested stem format for every voice under `stems_dir`, using
the resolved `TuningAnalysisResult` for shared-tuning note mapping. Fails
fast when a bend-based format exceeds its channel/polyphony budget or when
`mono_bend_12st` is requested for overlapping material.

- `stems_dir: Path` — output directory for `.mid` files.
- `stem_notes: dict[str, list[MidiStemNote]]` — usually produced by `collect_stem_notes`.
- `tuning_analysis: TuningAnalysisResult` — from `code_musics.midi_export_tuning.analyze_tuning`.
- `spec: MidiBundleExportSpec` — drives timing, tuning, and requested formats.

### `MidiStemNote`

Internal per-note record shared between `collect_stem_notes` and
`write_stem_files`. Stable enough to be useful when writing custom exporters.

| Field | Type | Description |
| --- | --- | --- |
| `voice_name` | `str` | Source voice. |
| `note_index` | `int` | Index of the note within `voice.notes`. |
| `start_seconds` | `float` | Onset in exported-bundle seconds (after window shift). |
| `duration_seconds` | `float` | Clipped duration. |
| `end_seconds` | `float` | Release in exported-bundle seconds. |
| `freq_hz` | `float` | Resolved absolute frequency in Hz. |
| `velocity` | `int` | Resolved MIDI velocity in `[1, 127]`. |
| `label` | `str \| None` | Optional note label carried through from `NoteEvent.label`. |

### `TuningAnalysisResult`

Resolved shared-tuning summary produced by
`code_musics.midi_export_tuning.analyze_tuning` and consumed by
`write_stem_files`.

| Field | Type | Description |
| --- | --- | --- |
| `tuning_mode` | `Literal["static_periodic_tuning", "exact_note_tuning"]` | Resolved mode. |
| `is_approximate` | `bool` | `True` when shared tuning files are convenience approximations. |
| `period_ratio` | `float` | Period as a frequency ratio. |
| `period_cents` | `float` | Period in cents. |
| `reference_midi_note` | `int` | MIDI note mapped to the reference pitch. |
| `reference_frequency_hz` | `float` | Reference pitch in Hz. |
| `pitch_class_cents` | `tuple[float, ...]` | Resolved pitch-class offsets in cents. |
| `scale_entry_cents` | `tuple[float, ...]` | Scale entries in cents (as written into `.scl`). |
| `quantization_cents` | `float` | Cents granularity used for pitch-class deduplication. |
| `warning_suffix` | `str \| None` | Filename suffix when approximate, else `None`. |
