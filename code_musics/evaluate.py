"""Piece evaluation via LLM judges.

Assembles a rich data packet from render artifacts, dispatches it to multiple
LLM judges running in parallel via Claude Code headless, and aggregates their
scores into a single evaluation result.  Designed as a standalone tool that
can also serve as the evaluation step in a future autonomous compose-evaluate
loop.
"""

from __future__ import annotations

import json
import logging
import statistics
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from code_musics.eval_rubric import (
    DEFAULT_JUDGE_MODELS,
    DIMENSIONS,
    build_judge_system_prompt,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JudgePacket:
    """Everything a judge needs to evaluate a piece."""

    piece_name: str
    score_summary: dict[str, Any]
    score_snapshot: dict[str, Any]
    timeline: dict[str, Any]
    analysis: dict[str, Any]
    image_paths: list[Path]
    render_ref: dict[str, Any]


@dataclass(frozen=True)
class DimensionScore:
    """One judge's score for one dimension."""

    score: int
    notes: str


@dataclass(frozen=True)
class JudgeResponse:
    """Structured response from a single judge."""

    model: str
    backend: str
    dimensions: dict[str, DimensionScore]
    overall_notes: str


@dataclass(frozen=True)
class EvalResult:
    """Aggregated evaluation across all judges."""

    piece_name: str
    evaluated_at_utc: str
    render_ref: dict[str, Any]
    judges: list[JudgeResponse]
    overall_score: float
    dimension_medians: dict[str, float]
    dimension_spreads: dict[str, int]
    confidence: str
    synthesized_feedback: str


# ---------------------------------------------------------------------------
# Judge backend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class JudgeBackend(Protocol):
    """Interface for dispatching a judge prompt and collecting the response."""

    def invoke(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path],
    ) -> str:
        """Send the evaluation prompt and return raw text output."""
        ...


# ---------------------------------------------------------------------------
# Headless Claude Code backend
# ---------------------------------------------------------------------------


class HeadlessBackend:
    """Run judges via ``claude --print``."""

    def invoke(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path],
    ) -> str:
        # image_paths used by API backends; headless embeds paths in user_prompt
        _ = image_paths
        cmd = [
            "claude",
            "--model",
            model,
            "--print",
            "--system-prompt",
            system_prompt,
            "--allowedTools",
            "Read",
            user_prompt,
        ]
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if completed.returncode != 0:
            logger.error(
                "Judge %s failed (rc=%d): %s",
                model,
                completed.returncode,
                completed.stderr[:500],
            )
            raise RuntimeError(f"Judge {model} exited with code {completed.returncode}")
        return completed.stdout


# ---------------------------------------------------------------------------
# Packet assembly
# ---------------------------------------------------------------------------


def _find_piece_output_dir(piece_name: str) -> Path:
    """Locate the output directory for a rendered piece."""
    from code_musics.pieces import PIECES

    if piece_name not in PIECES:
        raise ValueError(f"Unknown piece: {piece_name}")

    definition = PIECES[piece_name]
    if definition.study:
        candidate = Path("output/studies") / piece_name
    else:
        candidate = Path("output") / piece_name

    if not candidate.is_dir():
        raise FileNotFoundError(
            f"No rendered output for {piece_name!r} at {candidate}.  "
            f"Run `make render PIECE={piece_name}` first."
        )
    return candidate


def _read_json(path: Path) -> dict[str, Any]:
    """Read and parse a JSON file."""
    return json.loads(path.read_text())


def assemble_judge_packet(piece_name: str) -> JudgePacket:
    """Build a judge packet from existing render artifacts."""
    output_dir = _find_piece_output_dir(piece_name)

    from code_musics.pieces import PIECES

    definition = PIECES[piece_name]
    stem = definition.output_name

    render_json_path = output_dir / f"{stem}.render.json"
    timeline_path = output_dir / f"{stem}.timeline.json"
    analysis_path = output_dir / f"{stem}.analysis.json"

    missing = [
        p for p in (render_json_path, timeline_path, analysis_path) if not p.exists()
    ]
    if missing:
        missing_names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Missing render artifacts for {piece_name!r}: {missing_names}.  "
            f"Run `make render PIECE={piece_name}` first."
        )

    render_data = _read_json(render_json_path)
    timeline_data = _read_json(timeline_path)
    analysis_data = _read_json(analysis_path)

    score_summary = render_data.get("score_summary", {})
    score_snapshot = render_data.get("score_snapshot", {})

    render_ref = {
        "render_json_path": str(render_json_path),
        "git_commit": render_data.get("provenance", {}).get("git_commit"),
        "rendered_at_utc": render_data.get("rendered_at_utc"),
    }

    image_paths: list[Path] = []
    for suffix in (
        ".png",
        ".mix_spectrogram.png",
        ".score_density.png",
        ".mix_spectrum.png",
        ".mix_band_energy.png",
    ):
        candidate = output_dir / f"{stem}{suffix}"
        if candidate.exists():
            image_paths.append(candidate)

    return JudgePacket(
        piece_name=piece_name,
        score_summary=score_summary,
        score_snapshot=score_snapshot,
        timeline=timeline_data,
        analysis=analysis_data,
        image_paths=image_paths,
        render_ref=render_ref,
    )


def _build_user_prompt(packet: JudgePacket, packet_file: Path) -> str:
    """Build the user prompt that tells the judge what to read."""
    image_block = "\n".join(f"  - {p.resolve()}" for p in packet.image_paths)
    return f"""\
Evaluate the piece "{packet.piece_name}".

First, read the evaluation data packet:
  {packet_file.resolve()}

Then read each of these image files to see the visual analysis:
{image_block}

After reading all files, provide your evaluation as a JSON object following
the format described in your instructions."""


# ---------------------------------------------------------------------------
# Invocation and parsing
# ---------------------------------------------------------------------------


def _write_packet_file(packet: JudgePacket, directory: Path) -> Path:
    """Serialize the judge packet to a JSON file."""
    packet_data = {
        "piece_name": packet.piece_name,
        "score_summary": packet.score_summary,
        "timeline": packet.timeline,
        "analysis": packet.analysis,
        "score_snapshot": packet.score_snapshot,
    }
    path = directory / "judge_packet.json"
    path.write_text(json.dumps(packet_data, indent=2))
    return path


def parse_judge_response(raw_output: str, model: str) -> JudgeResponse:
    """Extract structured scores from raw judge output."""
    text = raw_output.strip()

    # Strip markdown code fencing if present
    if text.startswith("```"):
        first_newline = text.index("\n")
        text = text[first_newline + 1 :]
    if text.endswith("```"):
        text = text[: text.rfind("```")]
    text = text.strip()

    # Find the JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in judge output from {model}")

    data = json.loads(text[start : end + 1])

    dimensions: dict[str, DimensionScore] = {}
    raw_dims = data.get("dimensions", {})
    for dim in DIMENSIONS:
        dim_data = raw_dims.get(dim.key, {})
        dimensions[dim.key] = DimensionScore(
            score=int(dim_data.get("score", 5)),
            notes=str(dim_data.get("notes", "")),
        )

    return JudgeResponse(
        model=model,
        backend="headless",
        dimensions=dimensions,
        overall_notes=str(data.get("overall_notes", "")),
    )


def _run_single_judge(
    *,
    model: str,
    backend: JudgeBackend,
    system_prompt: str,
    user_prompt: str,
    image_paths: list[Path],
) -> JudgeResponse:
    """Invoke one judge and parse the response."""
    logger.info("Starting judge: %s", model)
    raw = backend.invoke(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_paths=image_paths,
    )
    response = parse_judge_response(raw, model)
    logger.info(
        "Judge %s complete — overall_notes: %s", model, response.overall_notes[:80]
    )
    return response


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_responses(
    responses: list[JudgeResponse], piece_name: str, render_ref: dict[str, Any]
) -> EvalResult:
    """Combine multiple judge responses into a single result."""
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    dimension_medians: dict[str, float] = {}
    dimension_spreads: dict[str, int] = {}

    for dim in DIMENSIONS:
        scores = [r.dimensions[dim.key].score for r in responses]
        dimension_medians[dim.key] = statistics.median(scores)
        dimension_spreads[dim.key] = max(scores) - min(scores)

    overall_score = round(
        sum(dim.weight * dimension_medians[dim.key] for dim in DIMENSIONS),
        1,
    )

    max_spread = max(dimension_spreads.values())
    if max_spread <= 2:
        confidence = "high"
    elif max_spread > 3:
        confidence = "low"
    else:
        confidence = "medium"

    feedback = synthesize_feedback(responses)

    return EvalResult(
        piece_name=piece_name,
        evaluated_at_utc=now,
        render_ref=render_ref,
        judges=responses,
        overall_score=overall_score,
        dimension_medians=dimension_medians,
        dimension_spreads=dimension_spreads,
        confidence=confidence,
        synthesized_feedback=feedback,
    )


def synthesize_feedback(responses: list[JudgeResponse]) -> str:
    """Distill judge notes into brief generator-facing feedback.

    V1 uses simple template-based concatenation of judge overall notes.
    Future: LLM synthesis pass for more natural prose.
    """
    notes = [r.overall_notes.strip() for r in responses if r.overall_notes.strip()]
    if not notes:
        return "No qualitative feedback available."

    # Deduplicate near-identical notes (same model family might echo)
    seen: list[str] = []
    for note in notes:
        if not any(_similar(note, s) for s in seen):
            seen.append(note)

    return "  ".join(seen)


def _similar(a: str, b: str) -> bool:
    """Rough duplicate check — same first 40 chars."""
    return a[:40].lower() == b[:40].lower()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _eval_to_dict(result: EvalResult) -> dict[str, Any]:
    """Serialize an EvalResult to a JSON-writable dict."""
    return {
        "schema_version": 1,
        "evaluated_at_utc": result.evaluated_at_utc,
        "piece_name": result.piece_name,
        "render_ref": result.render_ref,
        "judges": [
            {
                "model": j.model,
                "backend": j.backend,
                "dimensions": {
                    key: {"score": ds.score, "notes": ds.notes}
                    for key, ds in j.dimensions.items()
                },
                "overall_notes": j.overall_notes,
            }
            for j in result.judges
        ],
        "aggregate": {
            "overall_score": result.overall_score,
            "dimension_medians": result.dimension_medians,
            "dimension_spreads": result.dimension_spreads,
            "confidence": result.confidence,
            "synthesized_feedback": result.synthesized_feedback,
        },
    }


def save_eval_manifest(result: EvalResult, output_dir: Path) -> Path:
    """Write the full evaluation manifest to eval.json."""
    from code_musics.pieces import PIECES

    definition = PIECES[result.piece_name]
    stem = definition.output_name
    path = output_dir / f"{stem}.eval.json"
    path.write_text(json.dumps(_eval_to_dict(result), indent=2) + "\n")
    logger.info("Saved evaluation manifest: %s", path)
    return path


def append_eval_log(result: EvalResult, log_path: Path | None = None) -> None:
    """Append a summary line to the experiment log."""
    if log_path is None:
        log_path = Path("output/eval_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "piece": result.piece_name,
        "ts": result.evaluated_at_utc,
        "score": result.overall_score,
        "confidence": result.confidence,
        "feedback": result.synthesized_feedback,
        "git_commit": result.render_ref.get("git_commit"),
    }
    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def format_stdout_summary(result: EvalResult) -> str:
    """Build a human-readable summary for terminal output."""
    dim_lines: list[str] = []
    for dim in DIMENSIONS:
        median = result.dimension_medians[dim.key]
        spread = result.dimension_spreads[dim.key]
        flag = "  <- low agreement" if spread > 3 else ""
        dim_lines.append(f"  {dim.name:<24s} {median:4.1f}  (spread: {spread}){flag}")

    dim_block = "\n".join(dim_lines)
    return f"""\

{"=" * 50}
 Evaluation: {result.piece_name}
{"=" * 50}
Overall: {result.overall_score} / 10  (confidence: {result.confidence})

{dim_block}

Feedback:
  {result.synthesized_feedback}
"""


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    """Configuration for an evaluation run."""

    models: tuple[str, ...] = DEFAULT_JUDGE_MODELS
    backend: JudgeBackend = field(default_factory=HeadlessBackend)
    max_workers: int = 4


def evaluate_piece(
    piece_name: str,
    config: EvalConfig | None = None,
) -> EvalResult:
    """Evaluate a rendered piece end-to-end.

    1. Assemble judge packet from render artifacts.
    2. Dispatch to judges in parallel.
    3. Aggregate and synthesize.
    4. Save manifest and append experiment log.
    5. Print summary.
    """
    if config is None:
        config = EvalConfig()

    packet = assemble_judge_packet(piece_name)
    system_prompt = build_judge_system_prompt()

    with tempfile.TemporaryDirectory(prefix="centaur_eval_") as tmp:
        tmp_dir = Path(tmp)
        packet_file = _write_packet_file(packet, tmp_dir)
        user_prompt = _build_user_prompt(packet, packet_file)

        responses: list[JudgeResponse] = []
        errors: list[str] = []

        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            futures = {
                pool.submit(
                    _run_single_judge,
                    model=model,
                    backend=config.backend,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_paths=packet.image_paths,
                ): model
                for model in config.models
            }
            for future in as_completed(futures):
                model = futures[future]
                try:
                    responses.append(future.result())
                except Exception:
                    logger.exception("Judge %s failed", model)
                    errors.append(model)

    if not responses:
        raise RuntimeError(f"All judges failed for {piece_name!r}: {', '.join(errors)}")
    if errors:
        logger.warning(
            "%d/%d judges failed: %s",
            len(errors),
            len(config.models),
            ", ".join(errors),
        )

    result = aggregate_responses(responses, packet.piece_name, packet.render_ref)

    output_dir = _find_piece_output_dir(piece_name)
    save_eval_manifest(result, output_dir)
    append_eval_log(result)
    print(format_stdout_summary(result))

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_models(models_arg: str | None) -> tuple[str, ...]:
    """Parse the MODELS= argument into a tuple of model identifiers."""
    if models_arg is None:
        return DEFAULT_JUDGE_MODELS
    return tuple(m.strip() for m in models_arg.split(",") if m.strip())


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``python -m code_musics.evaluate``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate a rendered piece with LLM judges.",
    )
    parser.add_argument("piece", help="Piece name (or 'all' for every rendered piece)")
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated model identifiers "
            "(default: opus,sonnet,claude-opus-4-5-20250301,claude-sonnet-4-5-20241022)"
        ),
    )
    args = parser.parse_args(argv)

    models = _parse_models(args.models)
    config = EvalConfig(models=models)

    if args.piece == "all":
        from code_musics.pieces import PIECES

        for name in sorted(PIECES):
            output_dir = Path("output") / name
            if not output_dir.is_dir():
                study_dir = Path("output/studies") / name
                if not study_dir.is_dir():
                    logger.info("Skipping %s (not rendered)", name)
                    continue
            try:
                evaluate_piece(name, config)
            except Exception:
                logger.exception("Failed to evaluate %s", name)
    else:
        evaluate_piece(args.piece, config)


if __name__ == "__main__":
    main()
