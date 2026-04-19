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
from code_musics.score import Score


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
        score = Score(f0_hz=220.0, sample_rate=22050)
        macro = score.add_macro("brightness", default=0.6)
        mod = ModConnection(
            source=MacroSource(name=macro.name),
            target=AutomationTarget(kind="synth", name="cutoff_hz"),
            amount=1500.0,
            bipolar=False,
            mode="add",
        )
        lfo_mod = ModConnection(
            source=LFOSource(rate_hz=2.0, waveshape="sine"),
            target=AutomationTarget(kind="synth", name="cutoff_hz"),
            amount=200.0,
            mode="add",
        )
        pan_mod = ModConnection(
            source=LFOSource(rate_hz=0.5, waveshape="sine"),
            target=AutomationTarget(kind="control", name="pan"),
            amount=0.4,
            mode="add",
        )
        score.add_voice(
            "lead",
            synth_defaults={
                "engine": "polyblep",
                "cutoff_hz": 800.0,
                "waveform": "saw",
            },
            modulations=[mod, lfo_mod, pan_mod],
        )
        score.add_note("lead", start=0.0, duration=0.1, partial=1.0)
        audio = score.render()
        assert audio.ndim == 2
        assert audio.shape[1] > 0
        assert np.all(np.isfinite(audio))

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
