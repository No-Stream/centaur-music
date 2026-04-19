"""Tests for the per-connection modulation matrix."""

from __future__ import annotations

import numpy as np
import pytest

from code_musics.automation import (
    AutomationSegment,
    AutomationSpec,
    AutomationTarget,
)
from code_musics.modulation import (
    ConstantSource,
    DriftAdapter,
    EnvelopeSource,
    LFOSource,
    MacroDefinition,
    MacroSource,
    ModConnection,
    RandomSource,
    SourceSamplingContext,
    VelocitySource,
    build_macro_lookup,
    combine_connections_on_curve,
    combine_connections_scalar,
    iter_connections_for_target,
)
from code_musics.score import EffectSpec, Score


def _ctx(
    *,
    total_dur: float = 4.0,
    note_velocity: float | None = None,
    note_start: float | None = None,
    note_duration: float | None = None,
    macro_lookup: dict[str, np.ndarray] | None = None,
) -> SourceSamplingContext:
    return SourceSamplingContext(
        sample_rate=44100,
        total_dur=total_dur,
        macro_lookup=macro_lookup or {},
        note_velocity=note_velocity,
        note_start=note_start,
        note_duration=note_duration,
    )


class TestLFOSource:
    def test_sine_is_bounded_and_periodic(self) -> None:
        source = LFOSource(rate_hz=1.0, waveshape="sine")
        times = np.linspace(0.0, 4.0, 4001)
        curve = source.sample(times, _ctx())
        assert curve.shape == times.shape
        assert curve.min() >= -1.0 - 1e-9
        assert curve.max() <= 1.0 + 1e-9
        zero_crossings = np.where(np.diff(np.signbit(curve)))[0]
        # 1 Hz over 4 s => roughly 8 zero crossings.
        assert 6 <= zero_crossings.size <= 10

    def test_square_is_plus_or_minus_one(self) -> None:
        source = LFOSource(rate_hz=2.0, waveshape="square")
        times = np.linspace(0.0, 1.0, 1000)
        curve = source.sample(times, _ctx())
        assert set(np.unique(curve).tolist()) == {-1.0, 1.0}

    def test_smoothed_random_is_deterministic(self) -> None:
        source = LFOSource(rate_hz=0.5, waveshape="smoothed_random", seed=42)
        times = np.linspace(0.0, 4.0, 2000)
        first = source.sample(times, _ctx())
        second = source.sample(times, _ctx())
        np.testing.assert_allclose(first, second)

    def test_retrigger_shifts_phase(self) -> None:
        source = LFOSource(rate_hz=1.0, waveshape="sine", retrigger=True)
        times = np.array([2.0, 2.25, 2.5])
        curve = source.sample(times, _ctx(note_start=2.0))
        # Retrigger means at note_start the phase is 0 => value ~ 0.
        assert abs(curve[0]) < 1e-9


class TestEnvelopeSource:
    def test_adsr_shape_peaks_then_decays(self) -> None:
        source = EnvelopeSource(
            attack=0.1, hold=0.0, decay=0.2, sustain=0.5, release=0.3
        )
        times = np.linspace(0.0, 1.5, 1500)
        curve = source.sample(times, _ctx(note_start=0.0, note_duration=1.0))
        assert curve[0] == pytest.approx(0.0)
        assert curve[100] == pytest.approx(1.0, abs=0.05)  # end of attack
        # Sustain plateau between decay_end (0.3) and release_start (1.0)
        sustain_region = curve[(times > 0.4) & (times < 0.9)]
        assert np.all(np.abs(sustain_region - 0.5) < 1e-9)
        # Release reaches zero by note_end + release
        assert curve[-1] == pytest.approx(0.0, abs=0.05)

    def test_requires_note_info(self) -> None:
        source = EnvelopeSource()
        times = np.array([0.0, 0.1, 0.2])
        with pytest.raises(ValueError):
            source.sample(times, _ctx())


class TestMacroSource:
    def test_uses_macro_lookup(self) -> None:
        source = MacroSource(name="brightness")
        times = np.array([0.0, 0.5, 1.0])
        ctx = _ctx(macro_lookup={"brightness": np.array([0.2, 0.4, 0.6])})
        curve = source.sample(times, ctx)
        np.testing.assert_allclose(curve, [0.2, 0.4, 0.6])

    def test_unregistered_macro_raises(self) -> None:
        source = MacroSource(name="missing")
        with pytest.raises(ValueError, match="MacroSource"):
            source.sample(np.array([0.0]), _ctx())


class TestVelocitySource:
    def test_scales_by_velocity_scale(self) -> None:
        source = VelocitySource(velocity_scale=1.25)
        ctx = _ctx(note_velocity=1.0)
        curve = source.sample(np.array([0.0]), ctx)
        assert curve[0] == pytest.approx(1.0 / 1.25)

    def test_clipped_at_unity(self) -> None:
        source = VelocitySource(velocity_scale=1.0)
        ctx = _ctx(note_velocity=2.0)
        curve = source.sample(np.array([0.0]), ctx)
        assert curve[0] == pytest.approx(1.0)


class TestRandomSource:
    def test_seeded_determinism(self) -> None:
        source = RandomSource(rate_hz=4.0, seed=7)
        times = np.linspace(0.0, 2.0, 400)
        first = source.sample(times, _ctx())
        second = source.sample(times, _ctx())
        np.testing.assert_allclose(first, second)

    def test_bucketed_hold(self) -> None:
        source = RandomSource(rate_hz=1.0, seed=7)
        times = np.array([0.0, 0.5, 0.99, 1.01, 1.5])
        curve = source.sample(times, _ctx())
        # First three share bucket 0; last two share bucket 1.
        assert curve[0] == curve[1] == curve[2]
        assert curve[3] == curve[4]


class TestConstantSource:
    def test_broadcasts_value(self) -> None:
        source = ConstantSource(value=0.7)
        curve = source.sample(np.linspace(0, 1, 5), _ctx())
        np.testing.assert_allclose(curve, 0.7)


class TestDriftAdapter:
    def test_parity_with_drift_spec(self) -> None:
        from code_musics.humanize import DriftSpec, _sample_drift_curve

        adapter = DriftAdapter(
            style="random_walk", rate_hz=0.3, smoothness=0.8, seed=11
        )
        spec = DriftSpec(style="random_walk", rate_hz=0.3, smoothness=0.8, seed=11)
        times = np.linspace(0.0, 4.0, 400)
        from_adapter = adapter.sample(times, _ctx(total_dur=4.0))
        from_spec = _sample_drift_curve(spec, times=times, total_dur=4.0, seed=11)
        np.testing.assert_allclose(from_adapter, from_spec)


class TestModConnectionShaping:
    def test_bipolar_false_rectifies_bipolar_source(self) -> None:
        source = LFOSource(rate_hz=1.0, waveshape="sine")
        connection = ModConnection(
            source=source,
            target=AutomationTarget(kind="control", name="mix_db"),
            bipolar=False,
        )
        raw = np.array([-0.5, 0.0, 0.5])
        shaped = connection.shape(raw)
        np.testing.assert_allclose(shaped, [0.0, 0.0, 0.5])

    def test_power_shapes_curve(self) -> None:
        source = MacroSource(name="m")
        concave = ModConnection(
            source=source,
            target=AutomationTarget(kind="control", name="mix_db"),
            power=4.0,
        )
        convex = ModConnection(
            source=source,
            target=AutomationTarget(kind="control", name="mix_db"),
            power=-4.0,
        )
        raw = np.array([0.5])
        # power=4 -> exponent = exp(-1) -> 0.5 ** exp(-1) ~ 0.687
        assert concave.shape(raw)[0] == pytest.approx(0.5 ** np.exp(-1.0))
        assert convex.shape(raw)[0] == pytest.approx(0.5 ** np.exp(1.0))

    def test_breakpoints_warp_unipolar(self) -> None:
        source = MacroSource(name="m")
        connection = ModConnection(
            source=source,
            target=AutomationTarget(kind="control", name="mix_db"),
            breakpoints=((0.0, 0.0), (0.5, 0.1), (1.0, 1.0)),
        )
        raw = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        shaped = connection.shape(raw)
        # breakpoint at 0.5 -> 0.1; linear interp on either side.
        assert shaped[2] == pytest.approx(0.1)

    def test_amount_scales_output(self) -> None:
        source = ConstantSource(value=1.0)
        connection = ModConnection(
            source=source,
            target=AutomationTarget(kind="control", name="mix_db"),
            amount=3.0,
        )
        shaped = connection.shape(np.array([1.0]))
        assert shaped[0] == pytest.approx(3.0)

    def test_power_out_of_range_rejected(self) -> None:
        source = ConstantSource(value=1.0)
        with pytest.raises(ValueError):
            ModConnection(
                source=source,
                target=AutomationTarget(kind="control", name="mix_db"),
                power=25.0,
            )


class TestCombine:
    def test_add_mode_sums(self) -> None:
        source = ConstantSource(value=0.5)
        connection = ModConnection(
            source=source,
            target=AutomationTarget(kind="control", name="mix_db"),
            amount=1.0,
            mode="add",
        )
        base = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 0.1, 0.2])
        combined = combine_connections_on_curve(
            base=base,
            connections=[connection],
            times=times,
            context=_ctx(),
        )
        np.testing.assert_allclose(combined, [1.5, 2.5, 3.5])

    def test_multiply_mode_scales(self) -> None:
        source = ConstantSource(value=2.0)
        connection = ModConnection(
            source=source,
            target=AutomationTarget(kind="control", name="mix_db"),
            amount=1.0,
            mode="multiply",
        )
        base = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 0.1, 0.2])
        combined = combine_connections_on_curve(
            base=base,
            connections=[connection],
            times=times,
            context=_ctx(),
        )
        np.testing.assert_allclose(combined, [2.0, 4.0, 6.0])

    def test_replace_overrides_base(self) -> None:
        source = ConstantSource(value=7.0)
        connection = ModConnection(
            source=source,
            target=AutomationTarget(kind="control", name="mix_db"),
            amount=1.0,
            mode="replace",
        )
        base = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 0.1, 0.2])
        combined = combine_connections_on_curve(
            base=base,
            connections=[connection],
            times=times,
            context=_ctx(),
        )
        np.testing.assert_allclose(combined, [7.0, 7.0, 7.0])

    def test_scalar_combine_picks_note_start(self) -> None:
        source = ConstantSource(value=0.25)
        connection = ModConnection(
            source=source,
            target=AutomationTarget(kind="synth", name="cutoff_hz"),
            amount=100.0,
            mode="add",
        )
        combined = combine_connections_scalar(
            base=1000.0,
            connections=[connection],
            context=_ctx(note_start=0.5),
        )
        assert combined == pytest.approx(1025.0)

    def test_order_is_replace_then_multiply_then_add(self) -> None:
        replace_conn = ModConnection(
            source=ConstantSource(value=10.0),
            target=AutomationTarget(kind="control", name="mix_db"),
            amount=1.0,
            mode="replace",
        )
        multiply_conn = ModConnection(
            source=ConstantSource(value=2.0),
            target=AutomationTarget(kind="control", name="mix_db"),
            amount=1.0,
            mode="multiply",
        )
        add_conn = ModConnection(
            source=ConstantSource(value=1.0),
            target=AutomationTarget(kind="control", name="mix_db"),
            amount=1.0,
            mode="add",
        )
        base = np.zeros(3)
        combined = combine_connections_on_curve(
            base=base,
            connections=[add_conn, multiply_conn, replace_conn],
            times=np.array([0.0, 0.1, 0.2]),
            context=_ctx(),
        )
        # 10 (replace) -> *2 (multiply) -> +1 (add) = 21
        np.testing.assert_allclose(combined, [21.0, 21.0, 21.0])


class TestMacroLookup:
    def test_constant_macro_broadcast(self) -> None:
        macros = {"m": MacroDefinition(name="m", default=0.5)}
        times = np.linspace(0.0, 1.0, 5)
        lookup = build_macro_lookup(macros, times=times)
        np.testing.assert_allclose(lookup["m"], 0.5)

    def test_timeline_macro_samples(self) -> None:
        automation = AutomationSpec(
            target=AutomationTarget(kind="control", name="mix_db"),
            segments=(
                AutomationSegment(
                    start=0.0,
                    end=1.0,
                    shape="linear",
                    start_value=0.0,
                    end_value=1.0,
                ),
            ),
            default_value=0.0,
        )
        macros = {
            "brightness": MacroDefinition(
                name="brightness", default=0.0, automation=automation
            )
        }
        times = np.array([0.0, 0.5, 1.0])
        lookup = build_macro_lookup(macros, times=times)
        np.testing.assert_allclose(lookup["brightness"], [0.0, 0.5, 1.0])


class TestScoreIntegration:
    def test_iter_connections_for_target(self) -> None:
        a = ModConnection(
            source=ConstantSource(value=1.0),
            target=AutomationTarget(kind="control", name="mix_db"),
        )
        b = ModConnection(
            source=ConstantSource(value=1.0),
            target=AutomationTarget(kind="control", name="pan"),
        )
        found = iter_connections_for_target([a, b], kind="control", name="pan")
        assert found == [b]

    def test_score_renders_with_matrix(self) -> None:
        """Each matrix variant must materially change the rendered audio.

        Renders baseline vs. macro->cutoff, lfo->cutoff, and pan rides,
        each in isolation so we can assert which matrix contribution
        actually reached the audio.  ``Score.render()`` returns mono
        1D when no voice is panned and stereo ``(2, N)`` otherwise; we
        normalize to stereo for cross-variant comparison.
        """

        def _render(modulations: list[ModConnection]) -> np.ndarray:
            score = Score(f0_hz=220.0, sample_rate=22050)
            score.add_macro("brightness", default=0.6)
            score.add_voice(
                "lead",
                synth_defaults={
                    "engine": "polyblep",
                    "cutoff_hz": 800.0,
                    "waveform": "saw",
                },
                modulations=modulations,
            )
            score.add_note("lead", start=0.0, duration=0.4, partial=1.0)
            audio = score.render()
            assert audio.size > 0
            assert np.all(np.isfinite(audio))
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=0)
            return audio

        baseline = _render([])

        macro_mod = ModConnection(
            source=MacroSource(name="brightness"),
            target=AutomationTarget(kind="synth", name="cutoff_hz"),
            amount=1500.0,
            bipolar=False,
            mode="add",
        )
        macro_audio = _render([macro_mod])
        assert macro_audio.shape == baseline.shape
        macro_diff = float(np.mean(np.abs(macro_audio - baseline)))
        assert macro_diff > 1e-3, (
            f"macro->cutoff matrix did not alter audio (diff={macro_diff})"
        )

        lfo_mod = ModConnection(
            source=LFOSource(rate_hz=8.0, waveshape="sine"),
            target=AutomationTarget(kind="synth", name="cutoff_hz"),
            amount=1200.0,
            mode="add",
        )
        lfo_audio = _render([lfo_mod])
        assert lfo_audio.shape == baseline.shape
        lfo_diff = float(np.mean(np.abs(lfo_audio - baseline)))
        assert lfo_diff > 1e-3, (
            f"lfo->cutoff matrix did not alter audio (diff={lfo_diff})"
        )

        pan_mod = ModConnection(
            source=ConstantSource(value=1.0),
            target=AutomationTarget(kind="control", name="pan"),
            amount=0.8,
            mode="add",
        )
        pan_audio = _render([pan_mod])
        assert pan_audio.shape == baseline.shape
        pan_diff = float(np.mean(np.abs(pan_audio - baseline)))
        assert pan_diff > 1e-3, f"pan matrix did not alter audio (diff={pan_diff})"
        # Pan fully right should strongly decorrelate L vs R (baseline is centered).
        channel_spread = float(np.mean(np.abs(pan_audio[0] - pan_audio[1])))
        assert channel_spread > 1e-3, (
            "pan matrix did not produce an L/R difference "
            f"(channel_spread={channel_spread})"
        )

    def test_describe_modulations_lists_connections(self) -> None:
        score = Score(f0_hz=220.0, sample_rate=22050)
        connection = ModConnection(
            source=ConstantSource(value=1.0),
            target=AutomationTarget(kind="control", name="mix_db"),
            name="glue",
        )
        score.add_voice("v", modulations=[connection])
        rows = score.describe_modulations()
        assert len(rows) == 1
        assert rows[0]["scope"] == "voice:v"
        assert rows[0]["source"] == "ConstantSource"
        assert rows[0]["target_name"] == "mix_db"

    def test_param_profile_shifts_cutoff(self) -> None:
        """Matrix-driven cutoff produces different audio than baseline."""
        rng = np.random.default_rng(0)
        _ = rng  # keep signature stable if we later extend randomization

        def _render(with_matrix: bool) -> np.ndarray:
            score = Score(f0_hz=220.0, sample_rate=22050)
            synth_defaults = {
                "engine": "polyblep",
                "cutoff_hz": 400.0,
                "waveform": "saw",
            }
            modulations: list[ModConnection] = []
            if with_matrix:
                modulations.append(
                    ModConnection(
                        source=ConstantSource(value=1.0),
                        target=AutomationTarget(kind="synth", name="cutoff_hz"),
                        amount=3000.0,
                        bipolar=False,
                        mode="add",
                    )
                )
            score.add_voice(
                "lead",
                synth_defaults=synth_defaults,
                modulations=modulations,
            )
            score.add_note("lead", start=0.0, duration=0.1, partial=1.0)
            return score.render()

        baseline = _render(False)
        matrixed = _render(True)
        assert baseline.shape == matrixed.shape
        # Different filter cutoff must produce materially different audio.
        diff = float(np.mean(np.abs(baseline - matrixed)))
        assert diff > 1e-3

    def test_param_profile_time_varying_cutoff(self) -> None:
        """LFO-driven cutoff must produce time-varying filter behavior.

        A constant source would bake a single scalar cutoff into the
        engine.  Only the per-sample param_profiles path can surface
        LFO-shaped modulation at audio time, so we verify the rendered
        note's segment-wise RMS is not flat when an LFO rides cutoff.
        """

        def _render(with_matrix: bool) -> np.ndarray:
            score = Score(f0_hz=220.0, sample_rate=22050)
            synth_defaults = {
                "engine": "polyblep",
                "cutoff_hz": 600.0,
                "waveform": "saw",
            }
            modulations: list[ModConnection] = []
            if with_matrix:
                modulations.append(
                    ModConnection(
                        source=LFOSource(rate_hz=15.0, waveshape="sine"),
                        target=AutomationTarget(kind="synth", name="cutoff_hz"),
                        amount=1800.0,
                        bipolar=False,
                        mode="add",
                    )
                )
            score.add_voice(
                "lead",
                synth_defaults=synth_defaults,
                modulations=modulations,
            )
            # Long enough note to sample several LFO cycles at 15 Hz.
            score.add_note("lead", start=0.0, duration=0.5, partial=1.0)
            return score.render()

        baseline = _render(False)
        matrixed = _render(True)
        assert baseline.shape == matrixed.shape
        diff = float(np.mean(np.abs(baseline - matrixed)))
        assert diff > 1e-3

        # Split the modulated render into 8 contiguous segments and
        # confirm RMS varies noticeably — flat RMS would indicate the
        # LFO was folded to a single scalar at note onset rather than
        # applied per-sample.  ``render()`` may return mono 1D or
        # stereo (2, N); normalize to mono either way.
        mono = matrixed if matrixed.ndim == 1 else np.mean(matrixed, axis=0)
        segment_count = 8
        segment_size = mono.size // segment_count
        assert segment_size > 0
        segments = mono[: segment_count * segment_size].reshape(
            segment_count, segment_size
        )
        segment_rms = np.sqrt(np.mean(segments**2, axis=1))
        rms_spread = float(np.max(segment_rms) - np.min(segment_rms))
        rms_mean = float(np.mean(segment_rms))
        # A constant-cutoff filter on a saw will have stable segment RMS.
        # With a 15 Hz LFO of 1800 Hz depth, the spectral shape changes
        # enough that segment-to-segment RMS drifts by at least ~5% of
        # the average segment RMS.
        assert rms_mean > 1e-6
        assert rms_spread / rms_mean > 0.05, (
            f"expected LFO cutoff to modulate segment RMS; "
            f"rms_mean={rms_mean}, rms_spread={rms_spread}"
        )

    def test_macro_timeline_drives_cutoff_brightness(self) -> None:
        """A macro ramp 0->1 on cutoff_hz increases late-half brightness.

        End-to-end test that wires a macro with an AutomationSpec
        timeline through a ModConnection into synth cutoff, then
        measures spectral centroid between early and late halves of
        the rendered note to prove the macro motion reached the audio.
        The ``AutomationSpec.target`` is nominal (macros index by
        name, not by target); we reuse ``mix_db`` as a valid control
        target.
        """
        score = Score(f0_hz=220.0, sample_rate=22050)
        ramp = AutomationSpec(
            target=AutomationTarget(kind="control", name="mix_db"),
            segments=(
                AutomationSegment(
                    start=0.0,
                    end=3.0,
                    shape="linear",
                    start_value=0.0,
                    end_value=1.0,
                ),
            ),
            default_value=0.0,
        )
        score.add_macro("brightness", default=0.0, automation=ramp)
        score.add_voice(
            "lead",
            synth_defaults={
                "engine": "polyblep",
                "cutoff_hz": 400.0,
                "waveform": "saw",
            },
            modulations=[
                ModConnection(
                    source=MacroSource(name="brightness"),
                    target=AutomationTarget(kind="synth", name="cutoff_hz"),
                    amount=3500.0,
                    bipolar=False,
                    mode="add",
                )
            ],
        )
        score.add_note("lead", start=0.0, duration=3.0, partial=1.0)
        audio = score.render()
        mono = audio if audio.ndim == 1 else np.mean(audio, axis=0)
        assert mono.size > 0
        assert np.any(np.abs(mono) > 1e-9)

        def _spectral_centroid(segment: np.ndarray) -> float:
            windowed = segment * np.hanning(segment.size)
            spectrum = np.abs(np.fft.rfft(windowed))
            freqs = np.fft.rfftfreq(segment.size, d=1.0 / score.sample_rate)
            energy = float(spectrum.sum())
            if energy < 1e-12:
                return 0.0
            return float((spectrum * freqs).sum() / energy)

        half = mono.size // 2
        early_centroid = _spectral_centroid(mono[:half])
        late_centroid = _spectral_centroid(mono[half:])
        assert late_centroid > early_centroid * 1.15, (
            "macro timeline ramp should raise late-half spectral centroid; "
            f"early={early_centroid}, late={late_centroid}"
        )


class TestPitchRatioMatrix:
    """End-to-end coverage for kind='pitch_ratio' matrix fold."""

    def test_pitch_ratio_multiply_shifts_audio(self) -> None:
        """Multiplicative pitch_ratio modulation alters audio and FFT peak."""

        def _render(with_matrix: bool) -> np.ndarray:
            score = Score(f0_hz=220.0, sample_rate=22050)
            modulations: list[ModConnection] = []
            if with_matrix:
                modulations.append(
                    ModConnection(
                        source=ConstantSource(value=1.5),
                        target=AutomationTarget(kind="pitch_ratio", name="pitch_ratio"),
                        amount=1.0,
                        bipolar=False,
                        mode="multiply",
                    )
                )
            score.add_voice(
                "lead",
                synth_defaults={
                    "engine": "polyblep",
                    "cutoff_hz": 4000.0,
                    "waveform": "sine",
                },
                modulations=modulations,
            )
            score.add_note("lead", start=0.0, duration=0.5, partial=1.0)
            return score.render()

        baseline = _render(False)
        shifted = _render(True)
        assert baseline.shape == shifted.shape
        diff = float(np.mean(np.abs(baseline - shifted)))
        assert diff > 1e-3, f"pitch_ratio multiply did not alter audio (diff={diff})"

        # Coarse FFT peak comparison: a 1.5x ratio should push the
        # fundamental upward.  Use the held portion only to avoid the
        # release tail biasing the FFT.  ``render()`` may return mono
        # 1D or stereo (2, N) depending on pan state — normalize.
        sample_rate = 22050
        held_samples = int(0.5 * sample_rate)

        def _mono_slice(audio: np.ndarray) -> np.ndarray:
            if audio.ndim == 1:
                return audio[:held_samples]
            return np.mean(audio[:, :held_samples], axis=0)

        base_mono = _mono_slice(baseline)
        shift_mono = _mono_slice(shifted)
        base_spectrum = np.abs(np.fft.rfft(base_mono * np.hanning(base_mono.size)))
        shift_spectrum = np.abs(np.fft.rfft(shift_mono * np.hanning(shift_mono.size)))
        freqs = np.fft.rfftfreq(base_mono.size, d=1.0 / sample_rate)
        # Restrict to a reasonable fundamental band to avoid DC/top-end noise.
        band_mask = (freqs > 100.0) & (freqs < 1000.0)
        base_peak_hz = float(freqs[band_mask][np.argmax(base_spectrum[band_mask])])
        shift_peak_hz = float(freqs[band_mask][np.argmax(shift_spectrum[band_mask])])
        assert shift_peak_hz > base_peak_hz * 1.2, (
            f"expected FFT peak to rise with 1.5x ratio; "
            f"base={base_peak_hz}, shifted={shift_peak_hz}"
        )

    def test_pitch_ratio_non_positive_raises(self) -> None:
        """Modulation that drives pitch_ratio <= 0 must raise ValueError."""
        score = Score(f0_hz=220.0, sample_rate=22050)
        score.add_voice(
            "lead",
            synth_defaults={"engine": "polyblep", "waveform": "sine"},
            modulations=[
                # Base ratio starts at 1.0; adding -2.0 drives it to -1.0.
                ModConnection(
                    source=ConstantSource(value=1.0),
                    target=AutomationTarget(kind="pitch_ratio", name="pitch_ratio"),
                    amount=-2.0,
                    bipolar=True,
                    mode="add",
                )
            ],
        )
        score.add_note("lead", start=0.0, duration=0.2, partial=1.0)
        with pytest.raises(ValueError, match="strictly positive"):
            score.render()


class TestControlMatrixFold:
    """End-to-end coverage for _apply_db_control / _apply_pan_control."""

    def test_mix_db_matrix_lowers_amplitude(self) -> None:
        """A -12 dB mix_db matrix contribution halves the rendered peak.

        Disables both voice and master-bus normalization so the matrix
        dB contribution reaches the final render rather than being
        re-normalized away.
        """

        def _render(with_matrix: bool) -> np.ndarray:
            score = Score(
                f0_hz=220.0,
                sample_rate=22050,
                auto_master_gain_stage=False,
            )
            modulations: list[ModConnection] = []
            if with_matrix:
                modulations.append(
                    ModConnection(
                        source=ConstantSource(value=1.0),
                        target=AutomationTarget(kind="control", name="mix_db"),
                        amount=-12.0,
                        mode="add",
                    )
                )
            score.add_voice(
                "lead",
                synth_defaults={
                    "engine": "polyblep",
                    "cutoff_hz": 4000.0,
                    "waveform": "sine",
                },
                mix_db=0.0,
                normalize_peak_db=-6.0,
                modulations=modulations,
            )
            score.add_note("lead", start=0.0, duration=0.3, partial=1.0)
            return score.render()

        baseline = _render(False)
        modulated = _render(True)
        baseline_peak = float(np.max(np.abs(baseline)))
        modulated_peak = float(np.max(np.abs(modulated)))
        assert baseline_peak > 1e-4
        # -12 dB is ~0.25 linear; allow room for any residual ceiling.
        assert modulated_peak < baseline_peak * 0.6, (
            f"mix_db -12 dB matrix did not attenuate render "
            f"(baseline={baseline_peak}, modulated={modulated_peak})"
        )

    def test_pan_matrix_pushes_signal_right(self) -> None:
        """A full-right pan matrix leaves the left channel near silent."""
        score = Score(f0_hz=220.0, sample_rate=22050)
        score.add_voice(
            "lead",
            synth_defaults={
                "engine": "polyblep",
                "cutoff_hz": 4000.0,
                "waveform": "sine",
            },
            pan=0.0,
            modulations=[
                ModConnection(
                    source=ConstantSource(value=1.0),
                    target=AutomationTarget(kind="control", name="pan"),
                    amount=1.0,
                    mode="add",
                )
            ],
        )
        score.add_note("lead", start=0.0, duration=0.3, partial=1.0)
        audio = score.render()
        assert audio.ndim == 2
        left_peak = float(np.max(np.abs(audio[0])))
        right_peak = float(np.max(np.abs(audio[1])))
        assert right_peak > 1e-4
        assert left_peak < right_peak * 0.2, (
            f"pan matrix right should silence left channel "
            f"(left={left_peak}, right={right_peak})"
        )


class TestEffectWetMatrixFold:
    """F6: matrix connections must ride effect mix/wet/wet_level."""

    def test_saturation_mix_matrix_changes_output(self) -> None:
        """A matrix connection on saturation mix should measurably
        change the rendered audio versus an inactive connection.
        """

        def _render(mix_override: float) -> np.ndarray:
            score = Score(
                f0_hz=220.0,
                sample_rate=22050,
                auto_master_gain_stage=False,
            )
            score.add_voice(
                "lead",
                synth_defaults={
                    "engine": "polyblep",
                    "cutoff_hz": 4000.0,
                    "waveform": "saw",
                },
                normalize_peak_db=-6.0,
                effects=[
                    EffectSpec(
                        "delay",
                        {"delay_seconds": 0.08, "feedback": 0.4, "mix": 0.0},
                    )
                ],
                modulations=[
                    ModConnection(
                        source=ConstantSource(value=1.0),
                        target=AutomationTarget(kind="control", name="mix"),
                        amount=mix_override,
                        bipolar=False,
                        mode="add",
                    )
                ],
            )
            score.add_note("lead", start=0.0, duration=0.3, partial=1.0)
            return score.render()

        dry = _render(mix_override=0.0)
        wet = _render(mix_override=0.85)
        diff = float(np.mean(np.abs(dry - wet)))
        assert diff > 1e-3, (
            f"matrix contribution on saturation 'mix' did not alter audio "
            f"(mean abs diff={diff})"
        )


class TestCombineOrderTimeVarying:
    """The replace -> multiply -> add order must hold per-sample too."""

    def test_time_varying_combine_matches_manual_order(self) -> None:
        """Manually compose replace/multiply/add per-sample and compare."""
        replace_source = LFOSource(rate_hz=3.0, waveshape="sine")
        multiply_source = LFOSource(rate_hz=5.0, waveshape="triangle")
        add_source = LFOSource(rate_hz=7.0, waveshape="sine")

        replace_conn = ModConnection(
            source=replace_source,
            target=AutomationTarget(kind="control", name="mix_db"),
            amount=4.0,
            mode="replace",
        )
        multiply_conn = ModConnection(
            source=multiply_source,
            target=AutomationTarget(kind="control", name="mix_db"),
            amount=1.5,
            mode="multiply",
        )
        add_conn = ModConnection(
            source=add_source,
            target=AutomationTarget(kind="control", name="mix_db"),
            amount=2.0,
            mode="add",
        )

        times = np.linspace(0.0, 1.5, 512)
        context = _ctx(total_dur=1.5)
        base = np.sin(times * 2.0) + 0.5  # non-trivial non-zero base curve

        combined = combine_connections_on_curve(
            base=base.copy(),
            # Deliberately out of combine order; the function must
            # re-sort by mode (replace -> multiply -> add).
            connections=[add_conn, multiply_conn, replace_conn],
            times=times,
            context=context,
        )

        # Build expected manually in the correct order per-sample.
        replace_raw = replace_source.sample(times, context)
        multiply_raw = multiply_source.sample(times, context)
        add_raw = add_source.sample(times, context)
        expected = base.copy()
        expected = replace_conn.shape(replace_raw).astype(np.float64)
        expected = expected * multiply_conn.shape(multiply_raw)
        expected = expected + add_conn.shape(add_raw)
        np.testing.assert_allclose(combined, expected, rtol=1e-9, atol=1e-12)


class TestEnvelopeReleaseTail:
    """Pin that the envelope tail is exactly zero after release completes."""

    def test_envelope_tail_is_zero_after_release(self) -> None:
        attack = 0.05
        decay = 0.1
        sustain = 0.4
        release = 0.25
        note_duration = 0.8
        source = EnvelopeSource(
            attack=attack,
            hold=0.0,
            decay=decay,
            sustain=sustain,
            release=release,
        )
        times = np.linspace(0.0, note_duration + release + 0.5, 4000)
        curve = source.sample(times, _ctx(note_start=0.0, note_duration=note_duration))
        release_end = note_duration + release
        # Use a small epsilon so we don't catch samples that fall
        # exactly on the last release ramp point.
        tail_mask = times > (release_end + 1e-6)
        assert tail_mask.any()
        tail_values = curve[tail_mask]
        np.testing.assert_array_equal(tail_values, 0.0)


class TestMacroReRegistration:
    """Document add_macro overwrite-silently semantics.

    Mirrors the behavior of ``Score.add_drift_bus``, which also
    overwrites an existing entry with the same name.  If this ever
    needs to raise, update the behavior and this test together.
    """

    def test_re_registration_overwrites_silently(self) -> None:
        score = Score(f0_hz=220.0, sample_rate=22050)
        first = score.add_macro("brightness", default=0.5)
        assert score.macros["brightness"] is first
        assert score.macros["brightness"].default == pytest.approx(0.5)

        second = score.add_macro("brightness", default=0.8)
        assert score.macros["brightness"] is second
        assert score.macros["brightness"].default == pytest.approx(0.8)
        # No stale copy left behind.
        assert len(score.macros) == 1
