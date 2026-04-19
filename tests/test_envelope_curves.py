"""Tests for per-stage ADSR curve power shaping.

Covers the exponent-per-stage attack/decay/release curves plus the VCV-style
attack overshoot target. Defaults are linear (power=1.0, target=1.0), which
must remain byte-identical to the pre-curve-power ADSR for backward compat.
"""

from __future__ import annotations

import numpy as np
import pytest

from code_musics import synth

SR = 44100


def _unit_tone(dur: float = 2.0) -> np.ndarray:
    """Constant-amplitude sine long enough to exercise all stages fully."""
    t = np.arange(int(SR * dur)) / SR
    return np.sin(2 * np.pi * 440 * t)


def _raw_envelope(
    *,
    n: int,
    attack: float,
    decay: float,
    sustain_level: float,
    release: float,
    attack_power: float = 1.0,
    decay_power: float = 1.0,
    release_power: float = 1.0,
    attack_target: float = 1.0,
    hold_duration: float | None = None,
) -> np.ndarray:
    """Recover the envelope by applying ADSR to a flat unity signal."""
    flat = np.ones(n, dtype=np.float64)
    return synth.adsr(
        flat,
        attack=attack,
        decay=decay,
        sustain_level=sustain_level,
        release=release,
        sample_rate=SR,
        hold_duration=hold_duration,
        vca_nonlinearity=0.0,
        attack_power=attack_power,
        decay_power=decay_power,
        release_power=release_power,
        attack_target=attack_target,
    )


class TestDefaultsAreLinear:
    """Power=1.0 and target=1.0 must preserve the pre-change linear behavior."""

    def test_flat_signal_envelope_matches_linpace(self) -> None:
        """Default call reproduces the analytic linear ADSR exactly."""
        attack = 0.04
        decay = 0.1
        sustain_level = 0.7
        release = 0.3
        total_dur = 1.5
        n = int(SR * total_dur)
        recovered = _raw_envelope(
            n=n,
            attack=attack,
            decay=decay,
            sustain_level=sustain_level,
            release=release,
        )

        n_attack = int(attack * SR)
        n_decay = int(decay * SR)
        n_release = int(release * SR)
        hold_samples = max(0, n - n_release)
        release_samples = max(0, min(n_release, n - hold_samples))

        expected = np.zeros(n, dtype=np.float64)
        cursor = 0
        attack_samples = min(n_attack, hold_samples)
        if attack_samples > 0:
            expected[cursor : cursor + attack_samples] = np.linspace(
                0.0, 1.0, attack_samples, endpoint=False
            )
            cursor += attack_samples
        decay_samples = min(n_decay, hold_samples - cursor)
        if decay_samples > 0:
            expected[cursor : cursor + decay_samples] = np.linspace(
                1.0, sustain_level, decay_samples, endpoint=False
            )
            cursor += decay_samples
        sustain_samples = hold_samples - cursor
        if sustain_samples > 0:
            expected[cursor : cursor + sustain_samples] = sustain_level
            cursor += sustain_samples
        release_start_level = float(expected[cursor - 1]) if cursor > 0 else 0.0
        if release_samples > 0:
            expected[cursor : cursor + release_samples] = np.linspace(
                release_start_level, 0.0, release_samples, endpoint=True
            )

        np.testing.assert_allclose(recovered, expected, atol=1e-12)

    def test_audio_signal_matches_original_adsr(self) -> None:
        """Applying default ADSR to a sine matches calling without the new kwargs."""
        sig = _unit_tone()
        baseline = synth.adsr(
            sig,
            attack=0.03,
            decay=0.2,
            sustain_level=0.6,
            release=0.25,
            sample_rate=SR,
        )
        with_defaults = synth.adsr(
            sig,
            attack=0.03,
            decay=0.2,
            sustain_level=0.6,
            release=0.25,
            sample_rate=SR,
            attack_power=1.0,
            decay_power=1.0,
            release_power=1.0,
            attack_target=1.0,
        )
        np.testing.assert_allclose(with_defaults, baseline, atol=1e-12)


class TestAttackPower:
    def test_convex_attack_slow_start(self) -> None:
        """power=2.0 on attack should produce a convex (slow-start) ramp.

        At the halfway point of attack, env should be < 0.5 (linear midpoint).
        """
        attack = 0.4
        decay = 0.0
        sustain_level = 1.0
        release = 0.0
        total_dur = attack + 0.1
        n = int(SR * total_dur)
        env = _raw_envelope(
            n=n,
            attack=attack,
            decay=decay,
            sustain_level=sustain_level,
            release=release,
            attack_power=2.0,
        )
        mid_sample = int((attack * 0.5) * SR)
        assert env[mid_sample] < 0.5

    def test_concave_attack_fast_start(self) -> None:
        """power=0.5 on attack should produce a concave (fast-start) ramp."""
        attack = 0.4
        total_dur = attack + 0.1
        n = int(SR * total_dur)
        env = _raw_envelope(
            n=n,
            attack=attack,
            decay=0.0,
            sustain_level=1.0,
            release=0.0,
            attack_power=0.5,
        )
        mid_sample = int((attack * 0.5) * SR)
        assert env[mid_sample] > 0.5

    def test_attack_reaches_one_with_target_one(self) -> None:
        """With attack_target=1.0, the final attack sample must approach 1.0."""
        attack = 0.1
        total_dur = attack + 0.05
        n = int(SR * total_dur)
        env = _raw_envelope(
            n=n,
            attack=attack,
            decay=0.0,
            sustain_level=1.0,
            release=0.0,
            attack_power=2.0,
            attack_target=1.0,
        )
        end_of_attack = int(attack * SR) - 1
        assert 0.95 <= env[end_of_attack] <= 1.0

    def test_attack_target_overshoot_clamps_at_one(self) -> None:
        """attack_target>1.0 never exceeds 1.0 in the output envelope."""
        attack = 0.1
        total_dur = attack + 0.05
        n = int(SR * total_dur)
        env = _raw_envelope(
            n=n,
            attack=attack,
            decay=0.0,
            sustain_level=1.0,
            release=0.0,
            attack_target=1.2,
        )
        assert env.max() <= 1.0 + 1e-12

    def test_attack_target_reaches_one_earlier(self) -> None:
        """With attack_target>1, the attack reaches 1.0 before the stage ends.

        This is the defining property of the VCV ATT_TARGET idiom: the ramp
        aims past 1.0 and clamps there, so full sustain begins slightly earlier
        in the attack segment. A target=1.0 attack ramp approaches 1.0 only at
        the last sample; a target=1.2 ramp hits 1.0 at p = 1/1.2 (~83% through).
        """
        attack = 0.2
        total_dur = attack + 0.05
        n = int(SR * total_dur)
        linear_env = _raw_envelope(
            n=n,
            attack=attack,
            decay=0.0,
            sustain_level=1.0,
            release=0.0,
            attack_target=1.0,
        )
        overshoot_env = _raw_envelope(
            n=n,
            attack=attack,
            decay=0.0,
            sustain_level=1.0,
            release=0.0,
            attack_target=1.2,
        )
        n_attack = int(attack * SR)

        def _first_reach_one(env: np.ndarray, end: int) -> int:
            """Sample index where env first equals 1.0 within the attack stage."""
            for idx in range(end):
                if env[idx] >= 1.0 - 1e-12:
                    return idx
            return end

        linear_hits = _first_reach_one(linear_env, n_attack)
        overshoot_hits = _first_reach_one(overshoot_env, n_attack)
        assert overshoot_hits < linear_hits


class TestDecayPower:
    def test_decay_power_changes_midpoint(self) -> None:
        """power=2.0 on decay should make the midpoint closer to 1.0 than linear.

        Decay goes from 1.0 down to sustain_level; with power=2, shaped
        position s = p**2 stays small longer, so start + (end-start)*s is
        closer to start (which is the higher value).
        """
        attack = 0.01
        decay = 0.4
        sustain_level = 0.2
        release = 0.0
        total_dur = attack + decay + 0.05
        n = int(SR * total_dur)
        linear_env = _raw_envelope(
            n=n,
            attack=attack,
            decay=decay,
            sustain_level=sustain_level,
            release=release,
            decay_power=1.0,
        )
        curved_env = _raw_envelope(
            n=n,
            attack=attack,
            decay=decay,
            sustain_level=sustain_level,
            release=release,
            decay_power=2.0,
        )
        mid_sample = int(attack * SR) + int(decay * 0.5 * SR)
        assert curved_env[mid_sample] > linear_env[mid_sample]

    def test_decay_reaches_sustain(self) -> None:
        """End of decay stage must always reach sustain_level (for any power)."""
        attack = 0.01
        decay = 0.2
        sustain_level = 0.4
        total_dur = attack + decay + 0.3
        n = int(SR * total_dur)
        env = _raw_envelope(
            n=n,
            attack=attack,
            decay=decay,
            sustain_level=sustain_level,
            release=0.0,
            decay_power=3.0,
        )
        sustain_start = int(attack * SR) + int(decay * SR)
        assert abs(env[sustain_start] - sustain_level) < 0.05


class TestReleasePower:
    def test_release_power_half_is_concave(self) -> None:
        """power=0.5 on release should make a natural-decay-ish (concave) shape.

        Release goes from sustain_level to 0. With power=0.5, shaped
        position s = p**0.5 grows faster, so start + (end-start)*s drops
        faster early: env should be BELOW the linear envelope at midpoint.
        """
        attack = 0.01
        decay = 0.0
        sustain_level = 1.0
        release = 0.4
        total_dur = attack + release + 0.05
        n = int(SR * total_dur)
        hold_duration = attack
        linear_env = _raw_envelope(
            n=n,
            attack=attack,
            decay=decay,
            sustain_level=sustain_level,
            release=release,
            release_power=1.0,
            hold_duration=hold_duration,
        )
        curved_env = _raw_envelope(
            n=n,
            attack=attack,
            decay=decay,
            sustain_level=sustain_level,
            release=release,
            release_power=0.5,
            hold_duration=hold_duration,
        )
        mid_sample = int(attack * SR) + int(release * 0.5 * SR)
        assert curved_env[mid_sample] < linear_env[mid_sample]

    def test_release_reaches_zero(self) -> None:
        """End of release stage must always reach 0 for any power."""
        attack = 0.01
        decay = 0.0
        sustain_level = 1.0
        release = 0.25
        total_dur = attack + release + 0.05
        n = int(SR * total_dur)
        env = _raw_envelope(
            n=n,
            attack=attack,
            decay=decay,
            sustain_level=sustain_level,
            release=release,
            release_power=2.0,
            hold_duration=attack,
        )
        final_sample = int(attack * SR) + int(release * SR) - 1
        assert abs(env[final_sample]) < 1e-6


class TestStageContinuity:
    def test_no_discontinuity_across_stages(self) -> None:
        """With curvy powers, stage boundaries must remain continuous.

        Specifically: attack->decay transition should end near 1.0 and start
        decay at that same level; decay->sustain meets at sustain_level;
        sustain->release starts at sustain_level.
        """
        attack = 0.05
        decay = 0.1
        sustain_level = 0.5
        release = 0.2
        hold_duration = attack + decay + 0.1
        total_dur = hold_duration + release
        n = int(SR * total_dur)
        env = _raw_envelope(
            n=n,
            attack=attack,
            decay=decay,
            sustain_level=sustain_level,
            release=release,
            attack_power=2.5,
            decay_power=2.0,
            release_power=0.5,
            attack_target=1.2,
            hold_duration=hold_duration,
        )
        n_attack = int(attack * SR)
        n_decay = int(decay * SR)
        n_hold = int(hold_duration * SR)
        n_release = int(release * SR)

        attack_end = env[n_attack - 1]
        decay_start = env[n_attack]
        decay_end = env[n_attack + n_decay - 1]
        sustain_start = env[n_attack + n_decay]
        sustain_end = env[n_hold - 1]
        release_start = env[n_hold]
        release_end = env[n_hold + n_release - 1]

        assert attack_end > 0.9
        # decay starts from 1.0 -- attack_end may be <1 due to being one sample
        # before completion; decay always begins from fixed level 1.0
        assert abs(decay_start - 1.0) < 0.01
        assert abs(decay_end - sustain_level) < 0.05
        assert abs(sustain_start - sustain_level) < 0.05
        assert abs(sustain_end - sustain_level) < 0.05
        assert abs(release_start - sustain_level) < 0.05
        assert abs(release_end) < 1e-6


class TestParamClamping:
    @pytest.mark.parametrize("power", [0.0, -1.0, 50.0, 1e9])
    def test_extreme_powers_are_clamped_and_finite(self, power: float) -> None:
        """Out-of-range powers should be clamped internally; output remains finite."""
        sig = _unit_tone(dur=0.5)
        out = synth.adsr(
            sig,
            attack=0.02,
            decay=0.05,
            sustain_level=0.7,
            release=0.1,
            sample_rate=SR,
            attack_power=power,
            decay_power=power,
            release_power=power,
        )
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("target", [0.5, -1.0, 5.0])
    def test_extreme_targets_are_clamped(self, target: float) -> None:
        sig = _unit_tone(dur=0.5)
        out = synth.adsr(
            sig,
            attack=0.05,
            decay=0.05,
            sustain_level=0.7,
            release=0.1,
            sample_rate=SR,
            attack_target=target,
        )
        assert np.all(np.isfinite(out))
        # With target clamped to [1.0, 1.5], env still should not exceed 1.0.
        flat = np.ones_like(sig, dtype=np.float64)
        env = synth.adsr(
            flat,
            attack=0.05,
            decay=0.05,
            sustain_level=0.7,
            release=0.1,
            sample_rate=SR,
            attack_target=target,
        )
        assert env.max() <= 1.0 + 1e-12


class TestScoreSurfaceFlowthrough:
    """Verify the power params flow from synth_defaults through the score render.

    These are lightweight integration smoke tests: render a minimal score with
    a curvy-attack synth_defaults setting and confirm the output differs from
    a linear-default render on the same score.
    """

    def test_attack_power_changes_rendered_audio(self) -> None:
        from code_musics.score import Score

        linear_score = Score(f0_hz=220.0, auto_master_gain_stage=False)
        linear_score.add_voice(
            "pad",
            synth_defaults={"engine": "additive", "attack": 0.2},
            normalize_lufs=None,
        )
        linear_score.add_note("pad", start=0.0, duration=0.4, partial=1.0, amp=0.2)

        curvy_score = Score(f0_hz=220.0, auto_master_gain_stage=False)
        curvy_score.add_voice(
            "pad",
            synth_defaults={
                "engine": "additive",
                "attack": 0.2,
                "attack_power": 3.0,
            },
            normalize_lufs=None,
        )
        curvy_score.add_note("pad", start=0.0, duration=0.4, partial=1.0, amp=0.2)

        linear_audio = linear_score.render()
        curvy_audio = curvy_score.render()
        assert not np.allclose(linear_audio, curvy_audio)

    def test_attack_target_changes_rendered_audio(self) -> None:
        from code_musics.score import Score

        linear_score = Score(f0_hz=220.0, auto_master_gain_stage=False)
        linear_score.add_voice(
            "pad",
            synth_defaults={"engine": "additive", "attack": 0.2},
            normalize_lufs=None,
        )
        linear_score.add_note("pad", start=0.0, duration=0.4, partial=1.0, amp=0.2)

        pokey_score = Score(f0_hz=220.0, auto_master_gain_stage=False)
        pokey_score.add_voice(
            "pad",
            synth_defaults={
                "engine": "additive",
                "attack": 0.2,
                "attack_target": 1.2,
            },
            normalize_lufs=None,
        )
        pokey_score.add_note("pad", start=0.0, duration=0.4, partial=1.0, amp=0.2)

        linear_audio = linear_score.render()
        pokey_audio = pokey_score.render()
        assert not np.allclose(linear_audio, pokey_audio)

    def test_attack_power_automation_reshapes_rendered_attack(self) -> None:
        """Automating attack_power from 1.0 to 4.0 must measurably reshape the rendered attack.

        This is the real integration guard: it proves the param is wired as an
        automation target, accepted by the score, and actually drives the
        rendered audio. Registry membership alone is not enough — the name
        must flow through AutomationTarget -> AutomationSpec -> per-note
        synth param resolution -> the engine's ADSR call.
        """
        from code_musics.automation import (
            AutomationSegment,
            AutomationSpec,
            AutomationTarget,
        )
        from code_musics.score import Score

        dur = 0.5
        attack = 0.4  # long attack so the reshaped region dominates the waveform

        # Rebuild both scores against the same tone / timing. Only the attack_power
        # differs: the automated score ramps from 1.0 (linear) to 4.0 (very convex).
        def _build(
            attack_power_spec: AutomationSpec | None,
        ) -> Score:
            score = Score(f0_hz=220.0, auto_master_gain_stage=False)
            note_automation = [attack_power_spec] if attack_power_spec else None
            score.add_voice(
                "pad",
                synth_defaults={
                    "engine": "additive",
                    "attack": attack,
                    "release": 0.05,
                },
                normalize_lufs=None,
                velocity_humanize=None,
            )
            score.add_note(
                "pad",
                start=0.0,
                duration=dur,
                partial=1.0,
                amp=0.3,
                automation=note_automation,
            )
            return score

        # attack_power is sampled once at note start (it's a scalar passed to
        # adsr), so automation here is a constant hold that proves the param
        # name flows through AutomationTarget -> synth_params -> PreparedNote
        # -> adsr(..., attack_power=...). A ramp would sample start_value only.
        convex = AutomationSpec(
            target=AutomationTarget(kind="synth", name="attack_power"),
            segments=(
                AutomationSegment(
                    start=0.0,
                    end=dur,
                    shape="hold",
                    value=4.0,
                ),
            ),
            mode="replace",
        )

        linear_audio = _build(None).render()
        ramped_audio = _build(convex).render()

        # Both renders must share shape and finiteness.
        assert linear_audio.shape == ramped_audio.shape
        assert np.all(np.isfinite(linear_audio))
        assert np.all(np.isfinite(ramped_audio))

        # The overall audio must measurably differ.
        assert not np.allclose(linear_audio, ramped_audio, atol=1e-6)

        # Attack energy should drop meaningfully when attack_power is convex:
        # a convex ramp stays near zero for longer, so RMS in the first ~20%
        # of the attack region is lower than the linear version.
        mono_linear = (
            linear_audio.mean(axis=0) if linear_audio.ndim == 2 else linear_audio
        )
        mono_ramped = (
            ramped_audio.mean(axis=0) if ramped_audio.ndim == 2 else ramped_audio
        )
        n_probe = int(0.2 * attack * SR)
        rms_linear = float(np.sqrt(np.mean(mono_linear[:n_probe] ** 2)))
        rms_ramped = float(np.sqrt(np.mean(mono_ramped[:n_probe] ** 2)))
        assert rms_ramped < rms_linear * 0.9, (
            f"convex-attack automation should reduce early-attack RMS: "
            f"linear={rms_linear:.6f} ramped={rms_ramped:.6f}"
        )
