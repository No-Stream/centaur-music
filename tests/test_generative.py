"""Tests for the generative composition tools."""

from __future__ import annotations

import pytest

from code_musics.composition import HarmonicContext, RhythmCell, line
from code_musics.generative import (
    AksakPattern,
    LatticeWalker,
    RatioMarkov,
    TonePool,
    TuringMachine,
    ca_rhythm,
    ca_rhythm_layers,
    euclidean_line,
    euclidean_pattern,
    euclidean_rhythm,
    mutate_rhythm,
    prob_gate,
    prob_rhythm,
    stochastic_cloud,
)
from code_musics.meter import Timeline
from code_musics.score import Phrase, Score

# --- TonePool ---


def test_tone_pool_uniform_creates_equal_weights() -> None:
    pool = TonePool.uniform([1.0, 1.25, 1.5])
    assert pool.ratios == (1.0, 1.25, 1.5)
    assert pool.weights == pytest.approx((1 / 3, 1 / 3, 1 / 3))


def test_tone_pool_weighted_normalizes_weights() -> None:
    pool = TonePool.weighted({1.0: 2.0, 1.5: 3.0, 2.0: 5.0})
    assert pool.weights == pytest.approx((0.2, 0.3, 0.5))


def test_tone_pool_from_harmonics_computes_correct_ratios() -> None:
    pool = TonePool.from_harmonics([4, 5, 6, 7])
    assert pool.ratios == pytest.approx((1.0, 1.25, 1.5, 1.75))
    assert pool.weights == pytest.approx((0.25, 0.25, 0.25, 0.25))


def test_tone_pool_draw_is_deterministic() -> None:
    pool = TonePool.uniform([1.0, 1.25, 1.5, 1.75])
    a = pool.draw(10, seed=42)
    b = pool.draw(10, seed=42)
    assert a == b


def test_tone_pool_draw_different_seeds_differ() -> None:
    pool = TonePool.uniform([1.0, 1.25, 1.5, 1.75])
    a = pool.draw(20, seed=1)
    b = pool.draw(20, seed=2)
    assert a != b


def test_tone_pool_draw_no_replace_returns_unique_values() -> None:
    pool = TonePool.uniform([1.0, 1.25, 1.5, 1.75])
    drawn = pool.draw(4, seed=0, replace=False)
    assert len(drawn) == 4
    assert len(set(drawn)) == 4
    assert set(drawn) == {1.0, 1.25, 1.5, 1.75}


def test_tone_pool_draw_no_replace_raises_when_n_exceeds_pool() -> None:
    pool = TonePool.uniform([1.0, 1.5])
    with pytest.raises(ValueError, match="cannot draw 3 without replacement"):
        pool.draw(3, seed=0, replace=False)


def test_tone_pool_rejects_empty_ratios() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        TonePool.uniform([])
    with pytest.raises(ValueError, match="non-empty"):
        TonePool.weighted({})
    with pytest.raises(ValueError, match="non-empty"):
        TonePool.from_harmonics([])


def test_tone_pool_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="positive"):
        TonePool.weighted({1.0: -1.0, 1.5: 2.0})


# --- Euclidean ---


def test_euclidean_pattern_tresillo() -> None:
    pattern = euclidean_pattern(3, 8)
    assert sum(pattern) == 3
    assert len(pattern) == 8


def test_euclidean_pattern_five_of_eight() -> None:
    pattern = euclidean_pattern(5, 8)
    assert sum(pattern) == 5
    assert len(pattern) == 8


def test_euclidean_pattern_five_of_thirteen() -> None:
    pattern = euclidean_pattern(5, 13)
    assert sum(pattern) == 5
    assert len(pattern) == 13


def test_euclidean_pattern_rotation() -> None:
    base = euclidean_pattern(3, 8)
    rotated = euclidean_pattern(3, 8, rotation=1)
    assert rotated != base
    assert sum(rotated) == 3
    assert len(rotated) == 8


def test_euclidean_pattern_zero_hits() -> None:
    pattern = euclidean_pattern(0, 8)
    assert pattern == (False,) * 8


def test_euclidean_pattern_all_hits() -> None:
    pattern = euclidean_pattern(5, 5)
    assert pattern == (True,) * 5


def test_euclidean_pattern_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="steps must be positive"):
        euclidean_pattern(1, 0)
    with pytest.raises(ValueError, match="hits must satisfy"):
        euclidean_pattern(-1, 8)
    with pytest.raises(ValueError, match="hits must satisfy"):
        euclidean_pattern(9, 8)


def test_euclidean_rhythm_produces_correct_span_count() -> None:
    rhythm = euclidean_rhythm(3, 8, span=0.25)
    assert rhythm is not None
    assert len(rhythm.spans) == 3
    assert sum(rhythm.spans) == pytest.approx(8 * 0.25)


def test_euclidean_rhythm_raises_for_zero_hits() -> None:
    with pytest.raises(ValueError, match="hits must be positive"):
        euclidean_rhythm(0, 8)


def test_euclidean_line_produces_phrase_with_correct_event_count() -> None:
    phrase = euclidean_line([1.0, 1.25, 1.5], hits=3, steps=8, span=0.25)
    assert isinstance(phrase, Phrase)
    assert len(phrase.events) == 3


def test_euclidean_line_with_context_uses_ratio_line_path() -> None:
    ctx = HarmonicContext(tonic=220.0)
    phrase = euclidean_line([1.0, 5 / 4], hits=3, steps=8, context=ctx)
    for event in phrase.events:
        assert event.freq is not None
        assert event.partial is None


def test_euclidean_line_rejects_empty_tones() -> None:
    with pytest.raises(ValueError, match="tones must not be empty"):
        euclidean_line([], hits=3, steps=8)


# --- Probability Gate ---


def test_prob_gate_returns_phrase() -> None:
    source = line(tones=[4.0, 5.0, 6.0, 7.0], rhythm=(0.25, 0.25, 0.25, 0.25), amp=0.3)
    result = prob_gate(source, density=0.5, seed=42)
    assert isinstance(result, Phrase)


def test_prob_gate_density_one_keeps_all_notes() -> None:
    source = line(tones=[4.0, 5.0, 6.0], rhythm=(0.5, 0.5, 0.5), amp=0.3)
    result = prob_gate(source, density=1.0, seed=0)
    assert len(result.events) == len(source.events)


def test_prob_gate_density_zero_removes_all_notes() -> None:
    source = line(tones=[4.0, 5.0, 6.0], rhythm=(0.5, 0.5, 0.5), amp=0.3)
    result = prob_gate(source, density=0.0, seed=0)
    assert len(result.events) == 0


def test_prob_gate_is_deterministic() -> None:
    source = line(tones=[4.0, 5.0, 6.0, 7.0, 8.0], rhythm=(0.2,) * 5, amp=0.3)
    a = prob_gate(source, density=0.5, seed=99)
    b = prob_gate(source, density=0.5, seed=99)
    assert [e.partial for e in a.events] == [e.partial for e in b.events]


def test_prob_gate_position_weights_affect_survival() -> None:
    source = line(tones=[4.0, 5.0, 6.0, 7.0], rhythm=(0.25,) * 4, amp=0.3)
    heavy_first = prob_gate(
        source, density=0.5, position_weights=[10.0, 0.01, 0.01, 0.01], seed=0
    )
    heavy_last = prob_gate(
        source, density=0.5, position_weights=[0.01, 0.01, 0.01, 10.0], seed=0
    )
    assert [e.partial for e in heavy_first.events] != [
        e.partial for e in heavy_last.events
    ]


def test_prob_gate_empty_phrase_returns_empty() -> None:
    empty = Phrase(events=())
    result = prob_gate(empty, density=0.5, seed=0)
    assert len(result.events) == 0


# --- RatioMarkov ---


def test_ratio_markov_from_transitions_builds_valid_chain() -> None:
    chain = RatioMarkov.from_transitions(
        {
            1.0: {1.25: 1.0, 1.5: 1.0},
            1.25: {1.0: 1.0},
            1.5: {1.0: 1.0},
        }
    )
    assert chain.order == 1


def test_ratio_markov_generate_is_deterministic() -> None:
    chain = RatioMarkov.from_transitions(
        {
            1.0: {1.25: 1.0, 1.5: 1.0},
            1.25: {1.0: 2.0, 1.5: 1.0},
            1.5: {1.0: 1.0},
        }
    )
    a = chain.generate(20, seed=42)
    b = chain.generate(20, seed=42)
    assert a == b


def test_ratio_markov_generate_outputs_only_valid_ratios() -> None:
    valid_ratios = {1.0, 1.25, 1.5}
    chain = RatioMarkov.from_transitions(
        {
            1.0: {1.25: 1.0, 1.5: 1.0},
            1.25: {1.0: 1.0, 1.5: 1.0},
            1.5: {1.0: 1.0, 1.25: 1.0},
        }
    )
    result = chain.generate(50, seed=7)
    assert all(r in valid_ratios for r in result)


def test_ratio_markov_from_phrase_extracts_transitions() -> None:
    phrase = line(tones=[1.0, 1.25, 1.5, 1.0, 1.25], rhythm=(0.5,) * 5, amp=0.3)
    chain = RatioMarkov.from_phrase(phrase, order=1)
    assert chain.order == 1
    result = chain.generate(10, seed=0)
    assert len(result) == 10


def test_ratio_markov_from_table_order_2() -> None:
    chain = RatioMarkov.from_table(
        {
            (1.0, 1.25): {1.5: 1.0},
            (1.25, 1.5): {1.0: 1.0},
            (1.5, 1.0): {1.25: 1.0},
        },
        order=2,
    )
    assert chain.order == 2
    result = chain.generate(10, start=(1.0, 1.25), seed=0)
    assert len(result) == 10


def test_ratio_markov_to_phrase_produces_correct_event_count() -> None:
    chain = RatioMarkov.from_transitions(
        {
            1.0: {1.25: 1.0},
            1.25: {1.0: 1.0},
        }
    )
    phrase = chain.to_phrase(4, rhythm=RhythmCell(spans=(0.5,)), seed=0)
    assert isinstance(phrase, Phrase)
    assert len(phrase.events) == 4


def test_ratio_markov_rejects_empty_transitions() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        RatioMarkov.from_transitions({})


# --- TuringMachine ---


def test_turing_machine_zero_flip_repeats_with_period() -> None:
    tones = TonePool.uniform([1.0, 1.25, 1.5, 1.75])
    tm = TuringMachine(length=4, flip_probability=0.0, tones=tones, seed=42)
    seq = tm.generate(20)
    period = seq[:4]
    for i in range(4, 20, 4):
        assert seq[i : i + 4] == period


def test_turing_machine_is_deterministic() -> None:
    tones = [1.0, 1.25, 1.5]
    tm = TuringMachine(length=6, flip_probability=0.3, tones=tones, seed=99)
    a = tm.generate(30)
    b = tm.generate(30)
    assert a == b


def test_turing_machine_to_phrase() -> None:
    tones = [1.0, 1.5, 2.0]
    tm = TuringMachine(length=4, flip_probability=0.1, tones=tones, seed=0)
    phrase = tm.to_phrase(5, rhythm=RhythmCell(spans=(0.25,)))
    assert isinstance(phrase, Phrase)
    assert len(phrase.events) == 5


def test_turing_machine_rejects_empty_tones() -> None:
    with pytest.raises(ValueError, match="tones must be non-empty"):
        TuringMachine(length=4, tones=[], seed=0)


def test_turing_machine_rejects_invalid_flip_probability() -> None:
    with pytest.raises(ValueError, match="flip_probability must be in"):
        TuringMachine(length=4, flip_probability=1.5, tones=[1.0], seed=0)


# --- LatticeWalker ---


def test_lattice_walker_octave_reduce_keeps_ratios_in_range() -> None:
    walker = LatticeWalker(axes=(3, 5, 7), octave_reduce=True, seed=42)
    ratios = walker.walk(30)
    assert all(1.0 <= r < 2.0 for r in ratios)


def test_lattice_walker_no_octave_reduce_can_exceed_range() -> None:
    walker = LatticeWalker(axes=(3, 5), octave_reduce=False, max_distance=3, seed=42)
    ratios = walker.walk(50)
    assert any(r < 1.0 or r >= 2.0 for r in ratios)


def test_lattice_walker_is_deterministic() -> None:
    walker = LatticeWalker(axes=(3, 5, 7), seed=123)
    a = walker.walk(20)
    b = walker.walk(20)
    assert a == b


def test_lattice_walker_gravity_clusters_near_origin() -> None:
    walker_gravity = LatticeWalker(
        axes=(3, 5, 7), gravity=1.0, octave_reduce=False, seed=42
    )
    walker_free = LatticeWalker(
        axes=(3, 5, 7), gravity=0.0, octave_reduce=False, seed=42
    )
    ratios_gravity = walker_gravity.walk(100)
    ratios_free = walker_free.walk(100)
    mean_dist_gravity = sum(abs(r - 1.0) for r in ratios_gravity) / len(ratios_gravity)
    mean_dist_free = sum(abs(r - 1.0) for r in ratios_free) / len(ratios_free)
    assert mean_dist_gravity < mean_dist_free


def test_lattice_walker_to_phrase() -> None:
    walker = LatticeWalker(axes=(3, 5), seed=0)
    phrase = walker.to_phrase(6, rhythm=RhythmCell(spans=(0.5,)))
    assert isinstance(phrase, Phrase)
    assert len(phrase.events) == 6


def test_lattice_walker_rejects_empty_axes() -> None:
    with pytest.raises(ValueError, match="axes must be non-empty"):
        LatticeWalker(axes=(), seed=0)


def test_lattice_walker_rejects_negative_gravity() -> None:
    with pytest.raises(ValueError, match="gravity must be non-negative"):
        LatticeWalker(axes=(3,), gravity=-0.1, seed=0)


# --- Stochastic Cloud ---


def test_stochastic_cloud_produces_approximately_expected_event_count() -> None:
    phrase = stochastic_cloud(tones=[1.0, 1.5, 2.0], duration=10.0, density=5.0, seed=0)
    assert isinstance(phrase, Phrase)
    assert len(phrase.events) == 50


def test_stochastic_cloud_events_in_time_range() -> None:
    phrase = stochastic_cloud(tones=[1.0, 1.5], duration=4.0, density=10.0, seed=7)
    for event in phrase.events:
        assert 0.0 <= event.start <= 4.0


def test_stochastic_cloud_breakpoint_density_produces_more_notes_where_dense() -> None:
    sparse_then_dense = [
        (0.0, 1.0),
        (0.5, 1.0),
        (0.5, 20.0),
        (1.0, 20.0),
    ]
    phrase = stochastic_cloud(
        tones=[1.0, 1.5],
        duration=10.0,
        density=sparse_then_dense,
        seed=42,
    )
    midpoint = 5.0
    early = sum(1 for e in phrase.events if e.start < midpoint)
    late = sum(1 for e in phrase.events if e.start >= midpoint)
    assert late > early


def test_stochastic_cloud_is_deterministic() -> None:
    a = stochastic_cloud(tones=[1.0, 1.25, 1.5], duration=5.0, density=3.0, seed=77)
    b = stochastic_cloud(tones=[1.0, 1.25, 1.5], duration=5.0, density=3.0, seed=77)
    assert [e.start for e in a.events] == [e.start for e in b.events]
    assert [e.partial for e in a.events] == [e.partial for e in b.events]


def test_stochastic_cloud_with_context_uses_freq() -> None:
    ctx = HarmonicContext(tonic=220.0)
    phrase = stochastic_cloud(
        tones=[1.0, 5 / 4, 3 / 2],
        duration=2.0,
        density=5.0,
        context=ctx,
        seed=0,
    )
    for event in phrase.events:
        assert event.freq is not None
        assert event.partial is None


def test_stochastic_cloud_rejects_non_positive_duration() -> None:
    with pytest.raises(ValueError, match="duration must be positive"):
        stochastic_cloud(tones=[1.0], duration=0.0, seed=0)
    with pytest.raises(ValueError, match="duration must be positive"):
        stochastic_cloud(tones=[1.0], duration=-1.0, seed=0)


def test_stochastic_cloud_rejects_bad_amp_range() -> None:
    with pytest.raises(ValueError, match="amp_db_range"):
        stochastic_cloud(tones=[1.0], duration=1.0, amp_db_range=(0.0, -6.0), seed=0)


# --- Integration ---


def test_euclidean_line_integrates_with_score() -> None:
    score = Score(f0_hz=220.0)
    score.add_voice("rhythm", synth_defaults={"engine": "additive"})
    phrase = euclidean_line([4.0, 5.0, 6.0], hits=3, steps=8, span=0.25)
    score.add_phrase("rhythm", phrase, start=0.0)
    assert len(score.voices["rhythm"].notes) == 3


def test_generator_chain_tone_pool_turing_prob_gate_to_score() -> None:
    pool = TonePool.from_harmonics([4, 5, 6, 7])
    tm = TuringMachine(length=6, flip_probability=0.1, tones=pool, seed=42)
    phrase = tm.to_phrase(8, rhythm=RhythmCell(spans=(0.25,)))
    gated = prob_gate(phrase, density=0.7, seed=0)
    score = Score(f0_hz=110.0)
    score.add_voice("seq", synth_defaults={"engine": "additive"})
    score.add_phrase("seq", gated, start=0.0)
    assert len(score.voices["seq"].notes) == len(gated.events)


def test_stochastic_cloud_integrates_with_score() -> None:
    phrase = stochastic_cloud(
        tones=[1.0, 1.25, 1.5],
        duration=2.0,
        density=3.0,
        seed=0,
    )
    score = Score(f0_hz=220.0)
    score.add_voice("cloud", synth_defaults={"engine": "additive"})
    score.add_phrase("cloud", phrase, start=0.0)
    assert len(score.voices["cloud"].notes) == len(phrase.events)


# --- prob_rhythm ---


def test_prob_rhythm_uniform_density() -> None:
    cell = prob_rhythm(16, onset_weights=0.5, seed=42)
    assert 1 <= len(cell.spans) <= 16
    assert sum(cell.spans) == pytest.approx(16 * 0.25)


def test_prob_rhythm_cycling_onset_weights() -> None:
    cell = prob_rhythm(16, onset_weights=[1.0, 0.3, 0.5, 0.3], seed=7)
    assert 1 <= len(cell.spans) <= 16
    assert sum(cell.spans) == pytest.approx(16 * 0.25)


def test_prob_rhythm_zero_density_forces_one_onset() -> None:
    cell = prob_rhythm(8, onset_weights=0.0, seed=0)
    assert len(cell.spans) == 1
    assert sum(cell.spans) == pytest.approx(8 * 0.25)


def test_prob_rhythm_spans_sum_to_total() -> None:
    cell = prob_rhythm(12, onset_weights=0.6, span=0.5, seed=99)
    assert sum(cell.spans) == pytest.approx(12 * 0.5)


def test_prob_rhythm_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        prob_rhythm(8, onset_weights=-0.1, seed=0)
    with pytest.raises(ValueError, match="non-negative"):
        prob_rhythm(8, onset_weights=[0.5, -0.1], seed=0)


def test_prob_rhythm_rejects_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="steps must be positive"):
        prob_rhythm(0, seed=0)
    with pytest.raises(ValueError, match="steps must be positive"):
        prob_rhythm(-1, seed=0)


# --- AksakPattern ---


def test_aksak_balkan_7_rhythm() -> None:
    pat = AksakPattern.balkan_7(pulse=0.125)
    assert pat.grouping == (2, 2, 3)
    rhythm = pat.to_rhythm()
    assert len(rhythm.spans) == 3
    assert sum(rhythm.spans) == pytest.approx(7 * 0.125)


def test_aksak_turkish_9_meter() -> None:
    pat = AksakPattern.turkish_9(pulse=0.1)
    assert pat.to_meter() == (9, 8)


def test_aksak_from_timeline() -> None:
    tl = Timeline(bpm=120.0)
    pat = AksakPattern.from_timeline((2, 2, 3), tl)
    assert pat.grouping == (2, 2, 3)
    assert pat.pulse == pytest.approx(tl.duration(0.25))


def test_aksak_to_pulses() -> None:
    pat = AksakPattern(grouping=(3, 2), pulse=0.2)
    pulses = pat.to_pulses()
    assert len(pulses.spans) == 5
    assert all(s == pytest.approx(0.2) for s in pulses.spans)


def test_aksak_rejects_empty_grouping() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        AksakPattern(grouping=(), pulse=0.1)


def test_aksak_rejects_non_positive_groups() -> None:
    with pytest.raises(ValueError, match="positive"):
        AksakPattern(grouping=(2, 0, 3), pulse=0.1)


def test_aksak_rejects_non_positive_pulse() -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        AksakPattern(grouping=(2, 3), pulse=0.0)
    with pytest.raises(ValueError, match="positive and finite"):
        AksakPattern(grouping=(2, 3), pulse=-0.1)


def test_aksak_total_pulses_and_duration() -> None:
    pat = AksakPattern(grouping=(3, 3, 2), pulse=0.125)
    assert pat.total_pulses == 8
    assert pat.total_duration == pytest.approx(1.0)


def test_aksak_to_pulses_accent_first_produces_group_accents() -> None:
    pat = AksakPattern(grouping=(2, 3), pulse=0.2)
    accented = pat.to_pulses(accent_first=True)
    unaccented = pat.to_pulses(accent_first=False)
    # Both have 5 spans of equal duration
    assert len(accented.spans) == 5
    assert len(unaccented.spans) == 5
    assert accented.spans == unaccented.spans
    # Unaccented: uniform gates (scalar 1.0)
    assert unaccented.gates == 1.0
    # Accented: first pulse of each group gets 1.0, rest get 0.7
    # Group (2): pulse 0 = 1.0, pulse 1 = 0.7
    # Group (3): pulse 2 = 1.0, pulse 3 = 0.7, pulse 4 = 0.7
    assert accented.gates == pytest.approx((1.0, 0.7, 1.0, 0.7, 0.7))


# --- ca_rhythm ---


def test_ca_rule_30_produces_nontrivial_pattern() -> None:
    cell = ca_rhythm(30, 16)
    assert len(cell.spans) > 1
    assert sum(cell.spans) == pytest.approx(16 * 0.25)


def test_ca_rule_0_all_zero_fallback() -> None:
    cell = ca_rhythm(0, 8)
    assert len(cell.spans) == 1
    assert sum(cell.spans) == pytest.approx(8 * 0.25)


def test_ca_custom_init_state() -> None:
    cell_default = ca_rhythm(30, 8, seed=0)
    cell_custom = ca_rhythm(30, 8, init=0b11001100, seed=0)
    assert cell_default.spans != cell_custom.spans


def test_ca_different_rows_differ() -> None:
    cell_a = ca_rhythm(30, 16, row=-1)
    cell_b = ca_rhythm(30, 16, row=3)
    # Both produce valid rhythms; different rows usually differ for rule 30
    assert sum(cell_a.spans) == pytest.approx(16 * 0.25)
    assert sum(cell_b.spans) == pytest.approx(16 * 0.25)


def test_ca_rhythm_layers_count() -> None:
    layers = ca_rhythm_layers(30, 16, layers=4)
    assert len(layers) == 4
    for layer in layers:
        assert sum(layer.spans) == pytest.approx(16 * 0.25)


def test_ca_rejects_invalid_rule() -> None:
    with pytest.raises(ValueError, match="rule must be in 0-255"):
        ca_rhythm(-1, 8)
    with pytest.raises(ValueError, match="rule must be in 0-255"):
        ca_rhythm(256, 8)


def test_ca_seed_reproducibility() -> None:
    """Same seed produces identical output."""
    a = ca_rhythm(30, 16, seed=42)
    b = ca_rhythm(30, 16, seed=42)
    assert a.spans == b.spans


def test_ca_different_seeds_differ() -> None:
    """Different seeds produce different initial states and thus different rhythms."""
    a = ca_rhythm(30, 16, seed=0)
    b = ca_rhythm(30, 16, seed=99)
    assert a.spans != b.spans


def test_ca_layers_seed_reproducibility() -> None:
    """Same seed produces identical layers output."""
    a = ca_rhythm_layers(30, 16, layers=3, seed=7)
    b = ca_rhythm_layers(30, 16, layers=3, seed=7)
    assert [layer.spans for layer in a] == [layer.spans for layer in b]


def test_ca_layers_different_seeds_differ() -> None:
    """Different seeds produce different layers."""
    a = ca_rhythm_layers(30, 16, layers=3, seed=0)
    b = ca_rhythm_layers(30, 16, layers=3, seed=99)
    assert [layer.spans for layer in a] != [layer.spans for layer in b]


# --- mutate_rhythm ---


def test_mutate_identity() -> None:
    source = line(tones=[4.0, 5.0, 6.0, 7.0], rhythm=(0.25, 0.25, 0.25, 0.25), amp=0.3)
    result = mutate_rhythm(source, seed=0)
    assert len(result.events) == len(source.events)
    assert [e.start for e in result.events] == [e.start for e in source.events]
    assert [e.velocity for e in result.events] == [e.velocity for e in source.events]


def test_mutate_drop_removes_events() -> None:
    source = line(tones=[4.0, 5.0, 6.0, 7.0, 8.0] * 4, rhythm=(0.25,) * 20, amp=0.3)
    result = mutate_rhythm(source, drop_prob=0.5, seed=42)
    assert len(result.events) < len(source.events)
    assert len(result.events) >= 1


def test_mutate_subdivide_increases_count() -> None:
    source = line(tones=[4.0, 5.0, 6.0, 7.0], rhythm=(0.5, 0.5, 0.5, 0.5), amp=0.3)
    result = mutate_rhythm(source, subdivide_prob=1.0, seed=0)
    assert len(result.events) == len(source.events) * 2


def test_mutate_merge_decreases_count() -> None:
    source = line(tones=[4.0, 5.0, 6.0, 7.0], rhythm=(0.25, 0.25, 0.25, 0.25), amp=0.3)
    result = mutate_rhythm(source, merge_prob=1.0, seed=0)
    assert len(result.events) < len(source.events)


def test_mutate_shift_changes_starts() -> None:
    source = line(tones=[4.0, 5.0, 6.0, 7.0], rhythm=(0.5, 0.5, 0.5, 0.5), amp=0.3)
    result = mutate_rhythm(source, shift_amount=0.1, seed=42)
    original_starts = [e.start for e in source.events]
    mutated_starts = [e.start for e in result.events]
    assert original_starts != mutated_starts


def test_mutate_accent_drift_changes_velocities() -> None:
    source = line(tones=[4.0, 5.0, 6.0, 7.0], rhythm=(0.25, 0.25, 0.25, 0.25), amp=0.3)
    result = mutate_rhythm(source, accent_drift=0.3, seed=42)
    original_vels = [e.velocity for e in source.events]
    mutated_vels = [e.velocity for e in result.events]
    assert original_vels != mutated_vels


def test_mutate_deterministic_with_same_seed() -> None:
    source = line(tones=[4.0, 5.0, 6.0], rhythm=(0.25, 0.25, 0.25), amp=0.3)
    a = mutate_rhythm(source, drop_prob=0.3, shift_amount=0.05, seed=99)
    b = mutate_rhythm(source, drop_prob=0.3, shift_amount=0.05, seed=99)
    assert [e.start for e in a.events] == [e.start for e in b.events]
    assert [e.velocity for e in a.events] == [e.velocity for e in b.events]


def test_mutate_different_seeds_differ() -> None:
    source = line(tones=[4.0, 5.0, 6.0, 7.0] * 3, rhythm=(0.25,) * 12, amp=0.3)
    a = mutate_rhythm(source, drop_prob=0.3, accent_drift=0.2, seed=1)
    b = mutate_rhythm(source, drop_prob=0.3, accent_drift=0.2, seed=2)
    a_vels = [e.velocity for e in a.events]
    b_vels = [e.velocity for e in b.events]
    assert a_vels != b_vels


def test_mutate_empty_phrase_returns_empty() -> None:
    empty = Phrase(events=())
    result = mutate_rhythm(empty, drop_prob=0.5, seed=0)
    assert len(result.events) == 0
