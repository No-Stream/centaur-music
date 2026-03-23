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
