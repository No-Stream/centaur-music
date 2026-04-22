"""chaos_weather — changing weather-systems built from chaotic attractors.

A short "storm" piece driven by ``synth_voice`` chaotic-oscillator presets
(``osc_type="chaotic"``).  Each voice maps to a different attractor, and
a ``ChaoticSource`` modulates the filter cutoff on the lead voice so the
weather stays moving within each note, not just between them.

Aesthetic: a duffing bass groove establishes the ground weather, a lorenz
wobble lead sings above it with wind-like filter motion, and a chua
scatter glitches in as lightning.  Builds and recedes like a storm.

Sections (target wall-clock):

  1. Clearing                   (0:00 - 0:20)  Duffing bass alone, slow.
  2. Wind rising                (0:20 - 0:45)  Lorenz lead enters with
     chaotic cutoff modulation; ChaoticSource on the lead.
  3. Storm                      (0:45 - 1:05)  Chua scatter accents fire;
     rossler smear pad widens.
  4. Passing                    (1:05 - 1:25)  Scatter subsides, lead
     falls, duffing settles.

Key: F (F#1=46.25 Hz for bass / F3=174.614 Hz for lead) — 7-limit JI.  ~85 s.
"""

from __future__ import annotations

from code_musics.automation import (
    AutomationSegment,
    AutomationSpec,
    AutomationTarget,
)
from code_musics.modulation import ChaoticSource, ModConnection
from code_musics.pieces._shared import DEFAULT_MASTER_EFFECTS, SOFT_REVERB_EFFECT
from code_musics.pieces.registry import PieceDefinition, PieceSection
from code_musics.pitch_motion import PitchMotionSpec
from code_musics.score import Score, SendBusSpec, VoiceSend

F0_HZ = 174.614  # F3
TOTAL_DUR = 85.0

S1_END = 20.0
S2_END = 45.0
S3_END = 65.0


def build_chaos_weather() -> Score:
    """Build the Chaos Weather score."""
    score = Score(
        f0_hz=F0_HZ,
        master_effects=DEFAULT_MASTER_EFFECTS,
        send_buses=[
            SendBusSpec(
                name="hall",
                effects=[SOFT_REVERB_EFFECT],
                return_db=-3.0,
            )
        ],
    )

    # ------------------------------------------------------------------
    # Bass: duffing_bass — growly analog-style low end, irregular pulses.
    # Using normalize_peak_db keeps the chaotic-attractor transients
    # predictable, so the master limiter has less work to do.
    # ------------------------------------------------------------------
    score.add_voice(
        "bass",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "duffing_bass",
            "attack": 0.008,
            "release": 0.25,
        },
        effects=[],
        pan=0.0,
        mix_db=-8.0,
        normalize_peak_db=-10.0,
    )

    # ------------------------------------------------------------------
    # Lead: lorenz_wobble — slow vocal-ish drone through ladder filter.
    # ChaoticSource drives hpf_cutoff_hz so the filter breathes with the
    # attractor's own motion — weather within each note.
    # ------------------------------------------------------------------
    score.add_voice(
        "lead",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "lorenz_wobble",
            # Slow the chaos enough that the attractor's x-oscillation is sub-audible.
            "osc_chaos_rate_hz": 1.2,
            "osc_chaos_amount": 0.5,
            "hpf_cutoff_hz": 220.0,
            "attack": 0.35,
            "release": 1.6,
        },
        sends=[VoiceSend(target="hall", send_db=-3.0)],
        pan=0.12,
        mix_db=-8.0,
        normalize_peak_db=-8.0,
        modulations=[
            # ChaoticSource rides the HPF cutoff for irregular,
            # "weather-like" timbre motion.  Base 220 Hz + unipolar add of
            # up to 30 Hz keeps the cutoff comfortably above the bassy
            # chaotic-motion band, so the Rössler drift shapes timbre
            # rather than leaking sub-audio rumble into the right channel.
            ModConnection(
                source=ChaoticSource(
                    system="rossler",
                    rate_hz=0.6,
                    amount=0.5,
                    symmetry=0.1,
                    seed=71,
                ),
                target=AutomationTarget(kind="synth", name="hpf_cutoff_hz"),
                amount=30.0,
                bipolar=False,
                mode="add",
                name="weather_hpf",
            ),
        ],
    )

    # ------------------------------------------------------------------
    # Pad: supporting drone — scanned-synthesis pad that behaves
    # predictably under the limiter (unlike a chaotic pad).  This keeps
    # the piece's LUFS floor up without inflating peaks — the chaotic
    # character lives in the lead and bass, not the supporting layer.
    # ------------------------------------------------------------------
    score.add_voice(
        "pad",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "scanned_breathing_string",
            # Slow the scanned ring's mechanical evolution so the
            # amplitude envelope doesn't modulate at audible rates.
            "osc_scan_motion": 0.08,
            "attack": 2.5,
            "release": 3.5,
        },
        sends=[VoiceSend(target="hall", send_db=-2.0)],
        pan=-0.22,
        mix_db=-4.0,
    )

    # ------------------------------------------------------------------
    # Scatter: chua_scatter — glitchy lightning accents.
    # Sparse & bright, opposite stereo side to the pad for lightning feel.
    # ------------------------------------------------------------------
    score.add_voice(
        "scatter",
        synth_defaults={
            "engine": "synth_voice",
            "preset": "chua_scatter",
            "attack": 0.002,
            "release": 0.3,
        },
        sends=[VoiceSend(target="hall", send_db=-4.0)],
        pan=0.3,
        mix_db=-15.0,
        normalize_peak_db=-12.0,
    )

    # ==================================================================
    # Section 1 (0-20s): Duffing bass alone, slow irregular groove.
    # Weather-as-pulse: uneven gaps create natural rhythm.
    # ==================================================================
    bass_pulse_s1 = [
        # (start, duration, partial, velocity, amp_db)
        (0.8, 1.3, 0.5, 0.9, -6.0),  # F2
        (2.8, 1.2, 0.5, 0.85, -7.0),  # F2
        (5.0, 1.5, 2 / 3, 1.0, -6.0),  # Bb1-ish (sub 4/3 of root)
        (7.6, 1.4, 0.5, 0.95, -6.0),
        (10.0, 1.6, 3 / 4, 1.0, -6.0),  # C2 — sub fifth
        (12.8, 1.4, 0.5, 0.9, -7.0),
        (15.0, 2.0, 2 / 3, 1.05, -5.0),
        (17.8, 1.8, 0.5, 1.0, -6.0),
    ]
    for start, dur, partial, vel, amp_db in bass_pulse_s1:
        score.add_note(
            "bass",
            start=start,
            duration=dur,
            partial=partial,
            amp_db=amp_db,
            velocity=vel,
        )

    # ==================================================================
    # Section 2 (20-45s): Lead enters, wind rising.
    # Bass continues at similar pulse.  Pad sneaks in.
    # ==================================================================
    # Bass pulse continues into S2.
    bass_pulse_s2 = [
        (20.5, 1.4, 0.5, 1.0, -6.0),
        (22.8, 1.2, 2 / 3, 0.95, -6.0),
        (25.0, 1.5, 0.5, 1.0, -5.0),
        (27.6, 1.4, 3 / 4, 1.0, -6.0),
        (30.0, 1.6, 0.5, 1.05, -5.0),
        (32.6, 1.4, 2 / 3, 0.95, -6.0),
        (35.0, 1.5, 3 / 4, 1.0, -5.0),
        (37.6, 1.4, 0.5, 1.0, -6.0),
        (40.0, 1.6, 2 / 3, 1.05, -5.0),
        (42.8, 1.4, 3 / 4, 0.95, -6.0),
    ]
    for start, dur, partial, vel, amp_db in bass_pulse_s2:
        score.add_note(
            "bass",
            start=start,
            duration=dur,
            partial=partial,
            amp_db=amp_db,
            velocity=vel,
        )

    # Lead line — slow, vocal, always moving somewhere.
    lead_line_s2 = [
        (21.5, 5.0, 3 / 2, -12.0),
        (27.0, 4.0, 7 / 4, -11.0),
        (31.5, 4.5, 2.0, -10.0),
        (36.5, 4.5, 5 / 4 * 2, -11.0),  # 5/2 — major 3rd up octave
        (41.5, 3.0, 3 / 2 * 2, -10.0),  # 3 — fifth up octave
    ]
    for start, dur, partial, amp_db in lead_line_s2:
        score.add_note(
            "lead",
            start=start,
            duration=dur,
            partial=partial,
            amp_db=amp_db,
            pitch_motion=PitchMotionSpec.vibrato(depth_ratio=0.004, rate_hz=4.2),
        )

    # Pad: continuous quiet drone through S1 for ambient texture + LUFS
    # floor.  Root + fifth only — never steps on the bass pulses.
    for partial, amp_db in [(1.0, -14.0), (3 / 2, -16.0)]:
        score.add_note(
            "pad", start=0.5, duration=S1_END - 0.5, partial=partial, amp_db=amp_db
        )
    # S2: shift to 4/3, 5/3, 2.0 — a chord that complements the bass's sub motion.
    for partial, amp_db in [(4 / 3, -12.0), (5 / 3, -14.0), (2.0, -14.0)]:
        score.add_note(
            "pad",
            start=S1_END,
            duration=S2_END - S1_END,
            partial=partial,
            amp_db=amp_db,
        )

    # ==================================================================
    # Section 3 (45-65s): Storm.  Scatter accents fire unpredictably.
    # Bass densifies.  Pad widens harmonically.
    # ==================================================================
    # Denser bass — storm grows.
    for start in [
        45.2,
        46.6,
        47.8,
        49.2,
        50.6,
        51.9,
        53.4,
        54.8,
        56.2,
        57.6,
        59.0,
        60.4,
        61.8,
        63.2,
    ]:
        partial = 0.5 if int(start * 10) % 3 == 0 else 2 / 3
        score.add_note(
            "bass",
            start=start,
            duration=1.1,
            partial=partial,
            amp_db=-8.0,
            velocity=1.0,
        )

    # Lead lifts and holds a few bright notes in the storm.
    storm_lead = [
        (46.0, 6.0, 3.0, -9.0, 0.005),
        (52.5, 5.0, 7 / 4 * 2, -9.0, 0.006),
        (58.0, 5.5, 2.0 * 2, -8.0, 0.006),
    ]
    for start, dur, partial, amp_db, depth in storm_lead:
        score.add_note(
            "lead",
            start=start,
            duration=dur,
            partial=partial,
            amp_db=amp_db,
            pitch_motion=PitchMotionSpec.vibrato(depth_ratio=depth, rate_hz=5.0),
        )

    # Pad widens — 4 voices in the middle of the storm.
    for partial, amp_db in [
        (1.0, -12.0),
        (4 / 3, -13.0),
        (5 / 3, -15.0),
        (7 / 4, -14.0),
        (2.0, -15.0),
    ]:
        score.add_note(
            "pad", start=45.0, duration=S3_END - 45.0, partial=partial, amp_db=amp_db
        )

    # Scatter: lightning accents — sparse, chaotic timing, short bursts.
    scatter_strikes = [
        (46.3, 0.25, 3 / 2 * 4),  # very high
        (48.7, 0.2, 7 / 4 * 2),
        (51.1, 0.3, 9 / 4),
        (53.6, 0.15, 5 / 2),
        (55.2, 0.4, 3.0 * 2),  # brightest strike
        (57.9, 0.22, 7 / 4),
        (60.5, 0.25, 9 / 4 * 2),
        (62.8, 0.3, 5 / 2 * 2),
    ]
    for start, dur, partial in scatter_strikes:
        score.add_note(
            "scatter",
            start=start,
            duration=dur,
            partial=partial,
            amp_db=-12.0,
            velocity=1.1,
        )

    # ==================================================================
    # Section 4 (65-85s): Passing.  Everything thins out.
    # ==================================================================
    # Sparse final bass pulses, dying away.
    for start, amp_db in [
        (65.5, -8.0),
        (68.0, -9.0),
        (71.0, -10.0),
        (74.5, -11.0),
        (78.5, -13.0),
    ]:
        score.add_note(
            "bass",
            start=start,
            duration=1.8,
            partial=0.5,
            amp_db=amp_db,
            velocity=0.85,
        )

    # Lead falls through 3/2 -> 5/4 -> 1.0.
    falling_lead = [
        (66.0, 5.0, 3 / 2 * 2, -11.0),
        (71.5, 5.5, 5 / 4 * 2, -13.0),
        (77.0, 8.0, 2.0, -14.0),
    ]
    for start, dur, partial, amp_db in falling_lead:
        score.add_note(
            "lead",
            start=start,
            duration=dur,
            partial=partial,
            amp_db=amp_db,
            pitch_motion=PitchMotionSpec.vibrato(depth_ratio=0.003, rate_hz=3.8),
        )

    # Pad settles back to root triad, quieter.
    for partial, amp_db in [(1.0, -13.0), (3 / 2, -15.0), (2.0, -15.0)]:
        score.add_note(
            "pad",
            start=S3_END,
            duration=TOTAL_DUR - S3_END,
            partial=partial,
            amp_db=amp_db,
        )

    # Final scatter echo — one last distant flash.
    score.add_note(
        "scatter",
        start=74.0,
        duration=0.25,
        partial=7 / 4 * 2,
        amp_db=-16.0,
        velocity=0.9,
    )

    # Cross-section scatter send ride — louder in the storm, softer outside.
    # Automate scatter mix_db so it punches in during S3 only.
    scatter_voice = score.voices["scatter"]
    scatter_voice.automation.append(
        AutomationSpec(
            target=AutomationTarget(kind="control", name="mix_db"),
            segments=(
                AutomationSegment(start=0.0, end=S2_END, shape="hold", value=-24.0),
                AutomationSegment(
                    start=S2_END,
                    end=S2_END + 2.0,
                    shape="linear",
                    start_value=-24.0,
                    end_value=-8.0,
                ),
                AutomationSegment(
                    start=S2_END + 2.0, end=S3_END, shape="hold", value=-8.0
                ),
                AutomationSegment(
                    start=S3_END,
                    end=S3_END + 6.0,
                    shape="linear",
                    start_value=-8.0,
                    end_value=-24.0,
                ),
            ),
            default_value=-12.0,
            mode="replace",
        )
    )

    return score


PIECES: dict[str, PieceDefinition] = {
    "chaos_weather": PieceDefinition(
        name="chaos_weather",
        output_name="chaos_weather_01",
        build_score=build_chaos_weather,
        sections=(
            PieceSection(label="Clearing", start_seconds=0.0, end_seconds=S1_END),
            PieceSection(label="Wind", start_seconds=S1_END, end_seconds=S2_END),
            PieceSection(label="Storm", start_seconds=S2_END, end_seconds=S3_END),
            PieceSection(label="Passing", start_seconds=S3_END, end_seconds=TOTAL_DUR),
        ),
    ),
}
