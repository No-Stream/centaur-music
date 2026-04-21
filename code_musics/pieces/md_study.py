"""MD Study — Machinedrum-inspired kernel showcase at 96 BPM.

A short percussion-forward study in F, 7-limit just intonation, exercising
every new drum_voice kernel added in the Machinedrum pass:

  - PI modal kick (``pi_kick_shell``):    physical-body 4-on-the-floor
  - PI modal bells (``pi_metal_bell``):   sparse JI-tuned accent melody
  - PI wood block (``pi_wood_block``):    off-beat clack texture
  - EFM cymbal (``efm_cymbal_china``):    backbeat shimmer
  - EFM snare (``efm_snare_bright``):     clap-position crack
  - PI bowl (``pi_bowl_shimmer``):        long tail on the climax
  - Digital-character kick (``kick_bitcrush``): lo-fi stab section

Tuning: f0 = 87.31 Hz (F2).  Partial set keeps everything inside a
7-limit harmonic world:
  1       — F2    root / kick anchor
  3/2     — C3    fifth
  5/4     — A     major third (bell)
  7/4     — Eb    septimal seventh (bell — the dark accent)
  2       — F3    octave (bell)
  9/4     — G3    whole-step-above-octave (bell spice)
  3       — C4    high fifth (chime)

BPM = 96.  1 bar = 2.5 s.  1 beat = 0.625 s.  Total: 16 bars = 40 s.

Structure:
  bars  1- 4   intro:  modal kick alone + wood block off-beats
  bars  5- 8   add EFM cymbal + first bell entries (A & Eb)
  bars  9-12   pre-drop:  EFM snare joins on 2/4,
               kick switches to bitcrushed variant bar 11
  bars 13-16   climax:  full kit, bowl sustains over the top,
               bells reach the 9/4 and 3-octave spice notes
"""

from __future__ import annotations

from code_musics.automation import AutomationSegment, AutomationSpec, AutomationTarget
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS, SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.score import EffectSpec, Score, VoiceSend

BPM: float = 96.0
BEAT: float = 60.0 / BPM  # 0.625 s
BAR: float = 4.0 * BEAT  # 2.5 s
S16: float = BEAT / 4.0  # 0.15625 s
S8: float = BEAT / 2.0

F0: float = 87.31  # F2 — root

# 7-limit partial vocabulary for bell / bowl melody
P1: float = 1.0  # F2
P54: float = 5 / 4  # A3 (major 3rd, 386¢)
P32: float = 3 / 2  # C3 (perfect 5th)
P74: float = 7 / 4  # Eb3 (septimal 7th, 969¢ — the signature color)
P2: float = 2.0  # F3 (octave)
P94: float = 9 / 4  # G3 (whole step above octave)
P3: float = 3.0  # C4 (high fifth)


def _pos(bar: int, beat: int = 1, n16: int = 0) -> float:
    """Absolute seconds at (1-indexed) bar:beat:sixteenth."""
    return (bar - 1) * BAR + (beat - 1) * BEAT + n16 * S16


def build_md_study() -> Score:
    """Build the MD Study score."""
    score = Score(f0_hz=F0, master_effects=DEFAULT_MASTER_EFFECTS)

    # ------------------------------------------------------------------
    # Shared send bus: soft reverb for bells, bowl, cymbal
    # ------------------------------------------------------------------
    score.add_send_bus(
        "room",
        effects=[SOFT_REVERB_EFFECT],
    )

    # ------------------------------------------------------------------
    # Kick — PI modal shell for bars 1-10 and 13-16; bit-crushed variant
    # on bars 11-12 for the lofi pre-drop stab.
    # ------------------------------------------------------------------
    score.add_voice(
        "kick",
        synth_defaults={"engine": "drum_voice", "preset": "pi_kick_shell"},
        effects=[EffectSpec("compressor", {"preset": "kick_punch"})],
        normalize_peak_db=-6.0,
        mix_db=-4.0,
        velocity_humanize=None,
    )
    score.add_voice(
        "kick_crush",
        synth_defaults={"engine": "drum_voice", "preset": "kick_bitcrush"},
        effects=[EffectSpec("compressor", {"preset": "kick_punch"})],
        normalize_peak_db=-6.0,
        mix_db=-5.0,
        velocity_humanize=None,
    )

    for bar in range(1, 17):
        voice = "kick_crush" if bar in (11, 12) else "kick"
        for beat in range(1, 5):
            score.add_note(
                voice,
                start=_pos(bar, beat),
                duration=0.6,
                freq=55.0,  # A1 ≈ f0 * (5/8) — punchy sub below the scale root
                amp_db=-4.0 if bar >= 13 else -6.0,
            )

    # ------------------------------------------------------------------
    # Wood block — off-beat clacks throughout, brighter from bar 5
    # ------------------------------------------------------------------
    score.add_voice(
        "wood",
        synth_defaults={"engine": "drum_voice", "preset": "pi_wood_block"},
        normalize_peak_db=-12.0,
        mix_db=-12.0,
        velocity_humanize=None,
        pan=-0.35,
    )
    for bar in range(1, 17):
        base = -14.0 if bar <= 4 else -11.0
        for beat in [2, 4]:
            score.add_note(
                "wood",
                start=_pos(bar, beat, 2),  # "&" of 2 and "&" of 4
                duration=0.15,
                freq=880.0,
                amp_db=base,
            )

    # ------------------------------------------------------------------
    # EFM cymbal — bars 5-16, driving backbeat shimmer
    # ------------------------------------------------------------------
    score.add_voice(
        "hat",
        synth_defaults={"engine": "drum_voice", "preset": "efm_cymbal_china"},
        sends=[VoiceSend(target="room", send_db=-12.0)],
        normalize_peak_db=-10.0,
        mix_db=-16.0,
        velocity_humanize=None,
        pan=0.25,
    )
    for bar in range(5, 17):
        for beat in range(1, 5):
            # 8th-note hat pattern, emphasizing off-beats
            for n16 in [0, 2]:
                amp = -15.0 if n16 == 2 else -19.0  # off-beats louder
                score.add_note(
                    "hat",
                    start=_pos(bar, beat, n16),
                    duration=0.12,
                    freq=440.0,
                    amp_db=amp,
                )

    # ------------------------------------------------------------------
    # EFM snare — clap-position on 2 and 4, bars 9-16
    # ------------------------------------------------------------------
    score.add_voice(
        "snare",
        synth_defaults={"engine": "drum_voice", "preset": "efm_snare_bright"},
        sends=[VoiceSend(target="room", send_db=-15.0)],
        effects=[EffectSpec("compressor", {"preset": "snare_punch"})],
        normalize_peak_db=-6.0,
        mix_db=-9.0,
        velocity_humanize=None,
    )
    for bar in range(9, 17):
        for beat in [2, 4]:
            # Slight push on 4 in the climax bars — snare crack drives the groove
            accent = -3.0 if (bar >= 13 and beat == 4) else -6.0
            score.add_note(
                "snare",
                start=_pos(bar, beat),
                duration=0.25,
                freq=220.0,
                amp_db=accent,
            )

    # ------------------------------------------------------------------
    # Bell melody — PI metal-bell modal bank.  Sparse, JI-tuned.
    # Each note rings across the bar; bells overlap and create the
    # xenharmonic hook.
    # ------------------------------------------------------------------
    score.add_voice(
        "bell",
        synth_defaults={"engine": "drum_voice", "preset": "pi_metal_bell"},
        sends=[VoiceSend(target="room", send_db=-8.0)],
        normalize_peak_db=-9.0,
        mix_db=-10.0,
        velocity_humanize=None,
        pan=0.15,
    )
    # (bar, beat, n16, partial, amp_db)
    bell_notes: list[tuple[int, int, int, float, float]] = [
        # Section 2 (bars 5-8): introduce A and Eb — major 3rd against septimal 7th
        (5, 1, 0, P54, -12.0),  # A3
        (6, 3, 0, P74, -13.0),  # Eb3 — septimal
        (7, 1, 0, P54, -12.0),
        (7, 3, 2, P2, -14.0),  # F3 grounding
        (8, 2, 0, P74, -11.0),  # Eb accent on 2
        (8, 4, 2, P32, -13.0),  # C — push into section 3
        # Section 3 (bars 9-12): more motion
        (9, 1, 0, P2, -11.0),
        (9, 3, 0, P54, -12.0),
        (10, 1, 0, P74, -10.0),  # septimal hit
        (10, 2, 2, P32, -13.0),
        (10, 4, 0, P54, -11.0),
        (11, 1, 0, P2, -10.0),
        (11, 3, 2, P74, -11.0),
        (12, 2, 0, P54, -10.0),
        (12, 3, 2, P32, -12.0),
        (12, 4, 2, P2, -9.0),  # run up to the drop
        # Section 4 (bars 13-16): spice notes arrive
        (13, 1, 0, P94, -8.0),  # 9/4 — the "wrong" interval, the hook
        (13, 3, 0, P74, -9.0),
        (14, 1, 0, P3, -8.0),  # high fifth, 3rd octave
        (14, 2, 2, P2, -11.0),
        (14, 4, 0, P54, -10.0),
        (15, 1, 0, P94, -8.0),  # repeat the 9/4 — let it land
        (15, 3, 0, P74, -9.0),
        (15, 4, 2, P32, -11.0),
        (16, 1, 0, P3, -7.0),  # final high chime
        (16, 3, 0, P2, -10.0),
    ]
    for bar, beat, n16, partial, amp_db in bell_notes:
        score.add_note(
            "bell",
            start=_pos(bar, beat, n16),
            duration=1.4,
            partial=partial,
            amp_db=amp_db,
        )

    # ------------------------------------------------------------------
    # Bowl — singing PI bowl shimmer over the climax, one long sustain
    # per bar from bar 13.  Tuned to the octave so it glues everything.
    # ------------------------------------------------------------------
    score.add_voice(
        "bowl",
        synth_defaults={"engine": "drum_voice", "preset": "pi_bowl_shimmer"},
        sends=[VoiceSend(target="room", send_db=-6.0)],
        normalize_peak_db=-12.0,
        mix_db=-14.0,
        velocity_humanize=None,
    )
    for bar in (13, 15):
        score.add_note(
            "bowl",
            start=_pos(bar, 1),
            duration=4.5,
            partial=P2,
            amp_db=-10.0,
        )

    # ------------------------------------------------------------------
    # Hat mix_db automation: gentle ride across the piece so the cymbal
    # grows from the section-2 entry through the climax.
    # ------------------------------------------------------------------
    score.voices["hat"].automation = [
        AutomationSpec(
            target=AutomationTarget(kind="control", name="mix_db"),
            segments=(
                AutomationSegment(
                    start=_pos(5),
                    end=_pos(9),
                    shape="linear",
                    start_value=-18.0,
                    end_value=-14.0,
                ),
                AutomationSegment(
                    start=_pos(13),
                    end=_pos(17),
                    shape="linear",
                    start_value=-12.0,
                    end_value=-10.0,
                ),
            ),
        ),
    ]

    return score


PIECES: dict[str, PieceDefinition] = {
    "md_study": PieceDefinition(
        name="md_study",
        output_name="md_study",
        build_score=build_md_study,
        sections=(
            PieceSection(
                label="Intro (kick+wood)", start_seconds=0.0, end_seconds=10.0
            ),
            PieceSection(label="Cymbal+bells", start_seconds=10.0, end_seconds=20.0),
            PieceSection(label="Pre-drop", start_seconds=20.0, end_seconds=30.0),
            PieceSection(label="Climax", start_seconds=30.0, end_seconds=40.0),
        ),
        study=True,
    ),
}
