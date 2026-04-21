"""Tests for the evaluate module's pure functions."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from code_musics.eval_rubric import DIMENSIONS
from code_musics.evaluate import (
    DimensionScore,
    EvalResult,
    JudgeResponse,
    _eval_to_dict,
    _format_delta,
    aggregate_responses,
    synthesize_feedback,
)


@pytest.fixture()
def dim_keys() -> list[str]:
    return [d.key for d in DIMENSIONS]


def _make_judge_response(
    model: str,
    scores: dict[str, int],
    overall_notes: str = "",
) -> JudgeResponse:
    dims = {
        key: DimensionScore(score=scores.get(key, 50), notes=f"{key} notes")
        for key in (d.key for d in DIMENSIONS)
    }
    return JudgeResponse(
        model=model,
        backend="test",
        dimensions=dims,
        overall_notes=overall_notes,
    )


class TestFormatDelta:
    def test_positive_delta(self) -> None:
        assert _format_delta(75.0, 60.0) == "+15"

    def test_negative_delta(self) -> None:
        assert _format_delta(45.0, 60.0) == "-15"

    def test_near_zero_delta(self) -> None:
        assert _format_delta(60.2, 60.0) == "~"
        assert _format_delta(60.0, 60.4) == "~"

    def test_exactly_one_point_delta(self) -> None:
        assert _format_delta(61.0, 60.0) == "+1"

    def test_exactly_negative_one_point_delta(self) -> None:
        assert _format_delta(59.0, 60.0) == "-1"


class TestAggregateResponses:
    def test_median_and_spread_with_two_judges(self, dim_keys: list[str]) -> None:
        r1 = _make_judge_response("a", {k: 70 for k in dim_keys})
        r2 = _make_judge_response("b", {k: 80 for k in dim_keys})

        result = aggregate_responses([r1, r2], "test_piece", {})

        for key in dim_keys:
            assert result.dimension_medians[key] == pytest.approx(75.0)
            assert result.dimension_spreads[key] == 10

    def test_median_with_three_judges(self, dim_keys: list[str]) -> None:
        r1 = _make_judge_response("a", {k: 60 for k in dim_keys})
        r2 = _make_judge_response("b", {k: 70 for k in dim_keys})
        r3 = _make_judge_response("c", {k: 90 for k in dim_keys})

        result = aggregate_responses([r1, r2, r3], "test_piece", {})

        for key in dim_keys:
            assert result.dimension_medians[key] == pytest.approx(70.0)
            assert result.dimension_spreads[key] == 30

    def test_high_confidence_when_spread_at_most_15(self, dim_keys: list[str]) -> None:
        r1 = _make_judge_response("a", {k: 70 for k in dim_keys})
        r2 = _make_judge_response("b", {k: 80 for k in dim_keys})

        result = aggregate_responses([r1, r2], "test_piece", {})

        assert result.confidence == "high"

    def test_medium_confidence_when_spread_between_16_and_25(
        self, dim_keys: list[str]
    ) -> None:
        scores_a = {k: 60 for k in dim_keys}
        scores_b = {k: 80 for k in dim_keys}
        r1 = _make_judge_response("a", scores_a)
        r2 = _make_judge_response("b", scores_b)

        result = aggregate_responses([r1, r2], "test_piece", {})

        assert result.confidence == "medium"

    def test_low_confidence_when_spread_exceeds_25(self, dim_keys: list[str]) -> None:
        scores_a = {k: 40 for k in dim_keys}
        scores_b = {k: 70 for k in dim_keys}
        r1 = _make_judge_response("a", scores_a)
        r2 = _make_judge_response("b", scores_b)

        result = aggregate_responses([r1, r2], "test_piece", {})

        assert result.confidence == "low"

    def test_overall_score_is_weighted_sum(self, dim_keys: list[str]) -> None:
        r1 = _make_judge_response("a", {k: 80 for k in dim_keys})

        result = aggregate_responses([r1], "test_piece", {})

        expected = round(sum(d.weight * 80 for d in DIMENSIONS), 1)
        assert result.overall_score == pytest.approx(expected)


class TestSynthesizeFeedback:
    def test_improved_dimensions_appear_in_delta(self, dim_keys: list[str]) -> None:
        responses = [
            _make_judge_response("a", {k: 80 for k in dim_keys}, "Good piece.")
        ]
        current_medians = {k: 80.0 for k in dim_keys}
        previous = EvalResult(
            piece_name="p",
            evaluated_at_utc="2025-01-01T00:00:00Z",
            render_ref={},
            judges=[],
            overall_score=50.0,
            dimension_medians={k: 60.0 for k in dim_keys},
            dimension_spreads={k: 0 for k in dim_keys},
            confidence="high",
            synthesized_feedback="",
        )

        feedback = synthesize_feedback(responses, current_medians, previous)

        assert "Improved:" in feedback
        assert "+20" in feedback

    def test_regressed_dimensions_appear_in_delta(self, dim_keys: list[str]) -> None:
        responses = [_make_judge_response("a", {k: 40 for k in dim_keys}, "Meh.")]
        current_medians = {k: 40.0 for k in dim_keys}
        previous = EvalResult(
            piece_name="p",
            evaluated_at_utc="2025-01-01T00:00:00Z",
            render_ref={},
            judges=[],
            overall_score=70.0,
            dimension_medians={k: 70.0 for k in dim_keys},
            dimension_spreads={k: 0 for k in dim_keys},
            confidence="high",
            synthesized_feedback="",
        )

        feedback = synthesize_feedback(responses, current_medians, previous)

        assert "Regressed:" in feedback
        assert "-30" in feedback

    def test_no_delta_when_previous_is_none(self, dim_keys: list[str]) -> None:
        responses = [_make_judge_response("a", {k: 70 for k in dim_keys}, "Nice.")]

        feedback = synthesize_feedback(responses, {k: 70.0 for k in dim_keys}, None)

        assert "Delta" not in feedback
        assert "Nice." in feedback

    def test_small_changes_omitted_from_delta(self, dim_keys: list[str]) -> None:
        responses = [_make_judge_response("a", {k: 55 for k in dim_keys}, "Ok.")]
        current_medians = {k: 55.0 for k in dim_keys}
        previous = EvalResult(
            piece_name="p",
            evaluated_at_utc="2025-01-01T00:00:00Z",
            render_ref={},
            judges=[],
            overall_score=50.0,
            dimension_medians={k: 50.0 for k in dim_keys},
            dimension_spreads={k: 0 for k in dim_keys},
            confidence="high",
            synthesized_feedback="",
        )

        feedback = synthesize_feedback(responses, current_medians, previous)

        assert "Delta" not in feedback

    def test_fallback_when_no_notes(self) -> None:
        responses = [_make_judge_response("a", {}, "")]

        feedback = synthesize_feedback(responses)

        assert feedback == "No qualitative feedback available."

    def test_deduplication_of_similar_notes(self) -> None:
        r1 = _make_judge_response(
            "a", {}, "This piece has interesting harmony and form."
        )
        r2 = _make_judge_response(
            "b", {}, "This piece has interesting harmony and form. Also cool rhythm."
        )
        r3 = _make_judge_response("c", {}, "Completely different observation.")

        feedback = synthesize_feedback([r1, r2, r3])

        assert "Completely different observation." in feedback
        assert feedback.count("This piece has interesting harmony") == 1


class TestEvalRoundTrip:
    def test_write_and_load_schema_v2(self, dim_keys: list[str]) -> None:
        r1 = _make_judge_response("opus", {k: 75 for k in dim_keys}, "Thoughtful.")
        r2 = _make_judge_response("sonnet", {k: 65 for k in dim_keys}, "Interesting.")

        original = aggregate_responses([r1, r2], "test_piece", {"git_commit": "abc123"})

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval.json"
            data = _eval_to_dict(original)
            path.write_text(json.dumps(data, indent=2))

            loaded = json.loads(path.read_text())

        assert loaded["schema_version"] == 2
        assert loaded["piece_name"] == "test_piece"
        assert loaded["render_ref"]["git_commit"] == "abc123"
        assert loaded["aggregate"]["overall_score"] == original.overall_score
        assert loaded["aggregate"]["confidence"] == original.confidence
        assert len(loaded["judges"]) == 2
        assert loaded["judges"][0]["model"] == "opus"
        assert loaded["judges"][0]["dimensions"]["musical_substance"]["score"] == 75

    def test_round_trip_preserves_dimension_medians(self, dim_keys: list[str]) -> None:
        r1 = _make_judge_response("a", {k: 80 for k in dim_keys})
        original = aggregate_responses([r1], "piece", {})

        data = _eval_to_dict(original)
        agg = data["aggregate"]

        for key in dim_keys:
            assert agg["dimension_medians"][key] == original.dimension_medians[key]
            assert agg["dimension_spreads"][key] == original.dimension_spreads[key]
