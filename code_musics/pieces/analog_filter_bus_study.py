"""analog_filter_bus_study — the new `analog_filter` EffectSpec in action.

Until now the Diva-tier analog filter palette (ladder, K35, Jupiter, diode,
SEM, Sallen-Key, cascade, SVF) lived strictly inside the synth engines.  You
could not drop a ladder filter on a send bus or put a screaming K35 scream
on the master.  This piece is the proof-of-concept for the new
``EffectSpec("analog_filter", ...)`` surface:

* Master bus: a slow **ladder** LP filter that opens up over 30 s with an
  exponential cutoff sweep (200 Hz -> 8 kHz), expressed as an ``AutomationSpec``
  on the effect's ``cutoff_hz`` param — the score-time automation surface
  converts that spec into a per-sample curve at render time.  The whole mix
  emerges from underneath the filter like an acid-house stem unmuting.
* ``k35_scream`` send bus: a Korg MS-20-flavoured K35 filter in front of a
  reverb, so the pad reverb tail inherits the bite-y MS-20 snarl.
  ``k35_feedback_asymmetry=0.7`` pushes it into the screamer region.
* ``jupiter_bright`` send bus: a Jupiter-8-flavoured HPF on a second
  reverb/delay return, producing a parallel bright air band that keeps its
  top-end even when the master ladder is still shut.

The musical material is intentionally understated — a three-voice texture
(F# minor pad + sub-bass + tick pattern) that gives the filter moves room
to breathe.  The point is to hear the filters do something expressive on
the bus, not to impress with the tune.
"""

from __future__ import annotations

from code_musics.automation import (
    AutomationSegment,
    AutomationSpec,
    AutomationTarget,
)
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score, VoiceSend

# Piece duration — long enough for a slow ladder sweep to breathe but short
# enough that a full render stays well under a minute on an M-series laptop.
_DURATION_S: float = 30.0
_F0_HZ: float = 185.0  # F#3 — kick-friendly key window per AGENTS.md
_CUTOFF_HOLD_SECONDS: float = _DURATION_S * 0.10


def _build_cutoff_automation() -> AutomationSpec:
    """Exponential master-ladder cutoff sweep from 200 Hz to 8 kHz over the piece.

    Frequency perception is logarithmic, so a linear ramp would rush through
    the bottom and crawl at the top — the classic "filter sweeps wrong" bug.
    ``shape="exp"`` gives us geometric interpolation with a minimum of fuss;
    the static hold at the top of the piece is a zero-crossing constant
    segment so the opening 10% stays fully closed.
    """
    return AutomationSpec(
        target=AutomationTarget(kind="control", name="cutoff_hz"),
        segments=(
            AutomationSegment(
                start=0.0,
                end=_CUTOFF_HOLD_SECONDS,
                shape="linear",
                start_value=200.0,
                end_value=200.0,
            ),
            AutomationSegment(
                start=_CUTOFF_HOLD_SECONDS,
                end=_DURATION_S,
                shape="exp",
                start_value=200.0,
                end_value=8_000.0,
            ),
        ),
    )


def build_analog_filter_bus_study() -> Score:
    # Master chain: default preamp + glue compressor, then the new analog
    # filter on top.  The master ladder is the slow-opening identity of the
    # piece.  `filter_drive=0.35` gives just enough saturation to make the
    # closed filter sound "pressed" rather than merely LP'd.  The cutoff
    # sweep is expressed as an ``AutomationSpec`` on the effect — the
    # effect-chain pipeline converts it to a per-sample curve at render time.
    master_effects = [
        *DEFAULT_MASTER_EFFECTS,
        EffectSpec(
            "analog_filter",
            {
                "filter_topology": "ladder",
                "cutoff_hz": 200.0,  # base value; automation spec below drives it
                "resonance_q": 2.4,
                "filter_drive": 0.35,
                "bass_compensation": 0.5,  # preserve sub weight
                "mode": "lp",
                "quality": "great",
                "mix": 1.0,
            },
            automation=[_build_cutoff_automation()],
        ),
    ]

    score = Score(f0_hz=_F0_HZ, master_effects=master_effects)

    # Send bus 1: K35 screamer in front of a reverb.
    # The filter is driven hard with a moderate Q; the MS-20 diode-clipped
    # feedback gives the reverb tail a biting, vocal character especially
    # around sibilant pad notes.  Return trimmed -3 dB so the screamer stays
    # a colour, not the lead.
    score.add_send_bus(
        "k35_scream",
        effects=[
            EffectSpec(
                "analog_filter",
                {
                    "filter_topology": "k35",
                    "cutoff_hz": 1_800.0,
                    "resonance_q": 5.5,
                    "filter_drive": 0.55,
                    "k35_feedback_asymmetry": 0.7,  # the screamer knob
                    "mode": "lp",
                    "quality": "great",
                    "mix": 1.0,
                },
            ),
            EffectSpec(
                "reverb",
                {"room_size": 0.78, "damping": 0.45, "wet_level": 1.0},
            ),
        ],
        return_db=-3.0,
    )

    # Send bus 2: Jupiter-flavoured HP + delay for a parallel bright air band.
    # With the master ladder closed, this bus is the only thing keeping the
    # mix from feeling too dark in the opening.  `hpf_cutoff_hz=0` + mode='hp'
    # would also work, but using the engine-native Jupiter topology with mode
    # hp here gives the characteristic 24 dB/oct slope.
    score.add_send_bus(
        "jupiter_bright",
        effects=[
            EffectSpec(
                "analog_filter",
                {
                    "filter_topology": "jupiter",
                    "cutoff_hz": 1_500.0,
                    "resonance_q": 1.1,
                    "mode": "hp",
                    "filter_drive": 0.15,
                    "quality": "great",
                    "mix": 1.0,
                },
            ),
            EffectSpec(
                "delay",
                {"delay_seconds": 0.375, "feedback": 0.28, "mix": 0.45},
            ),
            EffectSpec(
                "reverb",
                {"room_size": 0.5, "damping": 0.35, "wet_level": 0.6},
            ),
        ],
        return_db=-6.0,
    )

    # --- Voices ---

    # Pad: the main harmonic body.  Two supersaw stacks with a slow chord
    # motion; feeds both send buses for character returns.
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "saw",
            "filter_topology": "svf",
            "cutoff_hz": 2_200.0,
            "resonance_q": 0.9,
            "osc2_level": 0.5,
            "osc2_detune_cents": 9.0,
            "attack": 0.6,
            "decay": 1.2,
            "sustain_level": 0.85,
            "release": 2.2,
        },
        pan=-0.05,
        mix_db=-4.0,
        sends=[
            VoiceSend(target="k35_scream", send_db=-12.0),
            VoiceSend(target="jupiter_bright", send_db=-10.0),
        ],
        automation=[
            # Pad vibrato depth rises through the piece to add life as the
            # master filter opens.
            AutomationSpec(
                target=AutomationTarget(kind="synth", name="vibrato_depth"),
                segments=(
                    AutomationSegment(
                        start=0.0,
                        end=_DURATION_S,
                        shape="linear",
                        start_value=0.002,
                        end_value=0.006,
                    ),
                ),
            ),
        ],
    )

    # Sub-bass: simple pulse line anchoring the bottom.  Does not feed the
    # k35_scream bus — that would muddy the return with low-end.
    score.add_voice(
        "sub",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "triangle",
            "cutoff_hz": 600.0,
            "resonance_q": 0.7,
            "attack": 0.01,
            "decay": 0.3,
            "sustain_level": 0.7,
            "release": 0.4,
        },
        pan=0.0,
        mix_db=-6.0,
    )

    # Tick: short bright pings to give transient contrast to the filter
    # motion.  Pan alternates, feeds the jupiter_bright bus so the pings
    # bloom into the parallel HP delay/reverb.
    score.add_voice(
        "tick",
        synth_defaults={
            "engine": "polyblep",
            "waveform": "square",
            "pulse_width": 0.35,
            "cutoff_hz": 4_500.0,
            "resonance_q": 1.8,
            "filter_env_amount": 1.2,
            "filter_env_decay": 0.14,
            "attack": 0.003,
            "decay": 0.11,
            "sustain_level": 0.0,
            "release": 0.22,
        },
        pan=0.25,
        mix_db=-12.0,
        sends=[VoiceSend(target="jupiter_bright", send_db=-6.0)],
    )

    # --- Notes ---

    # Pad: three chords across 30 s.  F#m -> D -> C#7 -> F#m (tonic return).
    # Expressed in 7-limit JI against f0 = F#3 = 185 Hz.
    # F# minor triad voicing (1, 6/5, 3/2), then D major (4/3 root going up
    # to a 5/3 / 2 / 9/4 voicing), then C#7 (5/6 root -- i.e. perfect fifth
    # below F# -- using 3/4, 15/16 * 2, etc).  Kept simple.
    pad_chords: list[tuple[float, float, list[float]]] = [
        (0.0, 10.0, [1.0, 6 / 5, 3 / 2, 2.0]),  # F#m
        (10.0, 10.0, [4 / 3, 5 / 3, 2.0, 5 / 2]),  # D-ish
        (20.0, 10.0, [3 / 4, 15 / 8, 9 / 4, 3.0]),  # bright tension chord
    ]
    for start, dur, voicing in pad_chords:
        for partial in voicing:
            score.add_note(
                "pad",
                start=start,
                duration=dur + 0.5,
                partial=partial,
                amp_db=-16.0,
            )

    # Sub: eighth notes on the root, dropped an octave (partial=0.5).
    sub_interval = 0.5  # 2 Hz at 120 BPM
    n_sub_steps = int(_DURATION_S / sub_interval)
    for step in range(n_sub_steps):
        t = step * sub_interval
        # Follow the chord-root motion: F# (0-10s), D (10-20s), C# (20-30s).
        if t < 10.0:
            partial = 0.5
        elif t < 20.0:
            partial = 4 / 3 * 0.5
        else:
            partial = 3 / 4 * 0.5
        score.add_note(
            "sub",
            start=t,
            duration=0.4,
            partial=partial,
            amp_db=-12.0,
            velocity=0.9 if step % 2 == 0 else 0.7,
        )

    # Tick: irregular sparse pings to give the filter motion something to
    # articulate.  Avoid the downbeats so the sub has room.
    tick_events: list[tuple[float, float]] = [
        (0.75, 2.5),
        (1.75, 3.0),
        (3.25, 2.0),
        (5.5, 4.0),
        (7.25, 2.5),
        (9.5, 3.0),
        (11.75, 2.5),
        (13.5, 4.0),
        (15.25, 3.0),
        (17.75, 2.0),
        (19.5, 4.0),
        (21.25, 3.0),
        (23.5, 2.5),
        (25.75, 3.5),
        (27.5, 2.0),
        (29.25, 2.5),
    ]
    for start, partial in tick_events:
        score.add_note(
            "tick",
            start=start,
            duration=0.25,
            partial=partial,
            amp_db=-18.0,
            velocity=0.8,
        )

    return score


PIECES: dict[str, PieceDefinition] = {
    "analog_filter_bus_study": PieceDefinition(
        name="analog_filter_bus_study",
        output_name="analog_filter_bus_study",
        build_score=build_analog_filter_bus_study,
        sections=(
            PieceSection(
                label="Filter closed (F#m)",
                start_seconds=0.0,
                end_seconds=10.0,
            ),
            PieceSection(
                label="Opening (D)",
                start_seconds=10.0,
                end_seconds=20.0,
            ),
            PieceSection(
                label="Fully open (C# tension)",
                start_seconds=20.0,
                end_seconds=_DURATION_S,
            ),
        ),
        study=True,
    ),
}
