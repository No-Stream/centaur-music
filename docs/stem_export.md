# Audio Stem Export

Score-backed pieces can export per-voice audio stem WAVs, send-bus returns, and an
optional mastered reference mix to a bundle directory with a JSON manifest.

## Commands

- `make stems PIECE=<name>` exports a full wet bundle
- `make stems PIECE=<name> DRY=1` exports a dry (pre-effects) bundle
- `make stems-snippet PIECE=<name> AT=<timestamp> WINDOW=<seconds>` exports a centered snippet bundle
- `make stems-window PIECE=<name> START=<timestamp> DUR=<seconds>` exports an exact-window bundle

## Bundle Layout

A bundle contains:

- `manifest.json`
- `voices/*.wav` — per-voice stems (one file per score `Voice`)
- `sends/*.wav` — per-send-bus returns (wet mode only)
- `mix.wav` — mastered reference mix (when `include_mix=True`)

All WAVs in a bundle share the same sample count (shorter stems are zero-padded
to the longest signal) so they line up exactly in a DAW.

## Export Modes

The exporter has two modes controlled by `StemExportSpec.dry`:

- **Wet** (`dry=False`, default): voice stems include normalization, `pre_fx_gain`,
  pan, voice effects, and the `mix_db` fader. Send stems include bus summing,
  bus effects, `return_db`, and bus pan. Voice stems plus send returns sum to
  the pre-master mix (before auto gain staging and master effects). The
  reference mix includes full mastering.
- **Dry** (`dry=True`): voice stems are post-normalization only — pre-effects,
  pre-pan, pre-fader, and mono. No send returns are included. The reference
  mix is still the full wet mastered mix so the dry stems can be aligned
  against a finished target.

## Ceiling Gain Behavior

To prevent inter-sample clipping on the individual stems while preserving their
summation property, the exporter computes the global peak across all stems and
send returns, then applies a single uniform gain such that the loudest sample
sits at `-0.5 dBFS`. Because every stem and send is scaled by the same factor,
their relative levels and the property "stems + sends = pre-master mix" are
preserved exactly. The applied gain in dB is recorded in the manifest as
`stem_gain_db` (`0.0` when no attenuation was needed).

The reference `mix.wav` is mastered independently (LUFS + true-peak ceiling)
and is not subject to the stem ceiling gain.

## Python API

### `export_stem_bundle(score, bundle_dir, *, spec) -> StemBundleResult`

Primary entry point. Renders the score, writes stem/send/mix WAVs plus
`manifest.json` into `bundle_dir`, and returns a `StemBundleResult`.

Parameters:

- `score: Score` — the score to export. Must have at least one voice.
- `bundle_dir: str | Path` — destination directory. Created if missing.
- `spec: StemExportSpec` — configuration (see below).

Returns a `StemBundleResult` with the bundle directory, manifest path, and
in-memory manifest.

### `StemExportSpec` (frozen dataclass)

Configuration for a stem export bundle.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `piece_name` | `str` | required | Name of the piece; recorded in the manifest. |
| `output_name` | `str` | required | Name of the bundle/output; recorded in the manifest. |
| `bit_depth` | `int` | `24` | WAV bit depth; must be `16`, `24`, or `32`. |
| `include_mix` | `bool` | `True` | Write the mastered reference `mix.wav` into the bundle. |
| `dry` | `bool` | `False` | Export dry stems (pre-effects/pan/fader, mono) instead of wet stems. |

Invalid `bit_depth` values raise `ValueError` at construction time.

### `StemFileInfo` (frozen dataclass)

Metadata for a single exported stem WAV, one entry per file in `voices/` or
`sends/`.

| Field | Type | Description |
| --- | --- | --- |
| `name` | `str` | Stem name (voice name or send bus name). |
| `kind` | `Literal["voice", "send"]` | Which subdirectory the file lives in. |
| `path` | `str` | Path to the WAV, relative to the bundle directory's parent. |
| `channels` | `int` | `1` for mono stems (always true for dry voice stems), `2` for stereo. |
| `sample_count` | `int` | Sample count after zero-padding (identical across all stems in a bundle). |
| `peak_dbfs` | `float` | Peak level of the written file in dBFS, rounded to 2 decimals. |

### `StemBundleManifest` (frozen dataclass)

Describes the contents of an exported bundle. Serialized to `manifest.json`
via `to_dict()`.

| Field | Type | Description |
| --- | --- | --- |
| `schema_version` | `int` | Manifest schema version (currently `1`). |
| `piece_name` | `str` | From `StemExportSpec.piece_name`. |
| `output_name` | `str` | From `StemExportSpec.output_name`. |
| `sample_rate` | `int` | Score sample rate in Hz. |
| `bit_depth` | `int` | WAV bit depth used for all files. |
| `dry` | `bool` | `True` when the bundle was exported in dry mode. |
| `total_duration_seconds` | `float` | Duration of the shared stem length (rounded to 4 decimals). |
| `total_samples` | `int` | Shared stem length in samples. |
| `voice_stems` | `list[StemFileInfo]` | Entries for files written to `voices/`. |
| `send_stems` | `list[StemFileInfo]` | Entries for files written to `sends/` (empty in dry mode). |
| `mix_path` | `str \| None` | Relative path to `mix.wav`, or `None` when the mix was not written. |
| `processing_notes` | `str` | Human-readable summary of which processing stages the stems include (differs between wet and dry mode). |
| `stem_gain_db` | `float` | Uniform gain applied to all stems to hit the `-0.5 dBFS` ceiling (rounded to 2 decimals; `0.0` when none applied). |
| `warnings` | `list[str]` | Non-fatal warnings captured during export. |

### `StemBundleResult` (frozen dataclass)

Return value from `export_stem_bundle`.

| Field | Type | Description |
| --- | --- | --- |
| `bundle_dir` | `Path` | Directory the bundle was written to. |
| `manifest_path` | `Path` | Full path to `manifest.json`. |
| `manifest` | `StemBundleManifest` | In-memory manifest object (same content as the JSON file). |

## Usage Example

```python
from pathlib import Path

from code_musics.stem_export import export_stem_bundle
from code_musics.stem_export_types import StemExportSpec

score = build_score()  # from a piece module

spec = StemExportSpec(
    piece_name="slow_glass",
    output_name="slow_glass_stems",
    bit_depth=24,
    include_mix=True,
    dry=False,
)

result = export_stem_bundle(
    score=score,
    bundle_dir=Path("out/slow_glass_stems"),
    spec=spec,
)

print(f"Wrote {len(result.manifest.voice_stems)} voice stems")
print(f"Manifest: {result.manifest_path}")
```

## Current Limitations

- Dry voice stems are always mono (pre-pan); stereo dry stems are not supported.
- `include_mix=False` skips the reference mix but does not change how the
  stem ceiling gain is computed.
- The ceiling gain is computed over stems only; the reference mix is mastered
  independently and may land at a different relative level than the summed
  stems.
