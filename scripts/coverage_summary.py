"""Emit a one-line coverage summary with delta vs. a committed baseline.

Reads `.coverage` (produced by pytest-cov) via `coverage json`, compares totals
against `.coverage-baseline` at the repo root, and prints a single informational
line to stdout. Exit code is always 0 on normal runs — regression is signalled
visually with a leading `⚠` and `↓`, not with a nonzero exit.

With `--update-baseline`, writes the current totals to `.coverage-baseline` as
a small JSON blob suitable for committing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COVERAGE_FILE = REPO_ROOT / ".coverage"
BASELINE_FILE = REPO_ROOT / ".coverage-baseline"

# Classification rule (documented per spec):
#   regression  = (new_covered < baseline_covered) OR (new_percent < baseline_percent - 0.1)
#   improvement = (new_covered > baseline_covered) OR (new_percent > baseline_percent + 0.1)
#   flat        = otherwise (within floating-point noise)
# The percent epsilon (0.1 pp) is slightly looser than the "same counts" check
# so small ratio shifts from `omit` filters don't trigger false improvements.
PERCENT_EPSILON = 0.1


def _run_coverage_json() -> dict:
    """Invoke `coverage json -o -` and parse the totals. Fail-fast on errors."""
    if not COVERAGE_FILE.exists():
        print(
            f"error: {COVERAGE_FILE} not found; run pytest with --cov first",
            file=sys.stderr,
        )
        sys.exit(1)

    result = subprocess.run(
        ["coverage", "json", "-o", "-", "--quiet"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _current_totals() -> dict[str, float | int]:
    data = _run_coverage_json()
    totals = data["totals"]
    return {
        "percent": float(totals["percent_covered"]),
        "covered": int(totals["covered_lines"]),
        "missing": int(totals["missing_lines"]),
        "statements": int(totals["num_statements"]),
    }


def _load_baseline() -> dict[str, float | int] | None:
    if not BASELINE_FILE.exists():
        return None
    try:
        raw = json.loads(BASELINE_FILE.read_text())
    except json.JSONDecodeError as exc:
        print(f"error: malformed {BASELINE_FILE}: {exc}", file=sys.stderr)
        sys.exit(1)
    required = {"percent", "covered", "missing", "statements"}
    missing_keys = required - raw.keys()
    if missing_keys:
        print(
            f"error: {BASELINE_FILE} missing keys: {sorted(missing_keys)}",
            file=sys.stderr,
        )
        sys.exit(1)
    return {
        "percent": float(raw["percent"]),
        "covered": int(raw["covered"]),
        "missing": int(raw["missing"]),
        "statements": int(raw["statements"]),
    }


def _format_signed_pct(delta: float) -> str:
    return f"{delta:+.1f}%"


def _format_signed_int(delta: int) -> str:
    return f"{delta:+d}"


def _classify(current: dict[str, float | int], baseline: dict[str, float | int]) -> str:
    pct_diff = float(current["percent"]) - float(baseline["percent"])
    covered_diff = int(current["covered"]) - int(baseline["covered"])

    same_counts = int(current["covered"]) == int(baseline["covered"]) and int(
        current["missing"]
    ) == int(baseline["missing"])
    if abs(pct_diff) < 0.05 and same_counts:
        return "flat"

    if covered_diff < 0 or pct_diff < -PERCENT_EPSILON:
        return "regression"
    if covered_diff > 0 or pct_diff > PERCENT_EPSILON:
        return "improvement"
    return "flat"


def _print_summary(
    current: dict[str, float | int], baseline: dict[str, float | int] | None
) -> None:
    pct = float(current["percent"])

    if baseline is None:
        print(
            f"coverage: {pct:.1f}% (no baseline; run 'make coverage-baseline' to set)"
        )
        return

    kind = _classify(current, baseline)
    if kind == "flat":
        print(f"coverage: {pct:.1f}%")
        return

    base_pct = float(baseline["percent"])
    pct_diff = pct - base_pct
    covered_diff = int(current["covered"]) - int(baseline["covered"])
    missing_diff = int(current["missing"]) - int(baseline["missing"])

    detail = (
        f"({_format_signed_int(covered_diff)} covered, "
        f"{_format_signed_int(missing_diff)} missing) vs baseline {base_pct:.1f}%"
    )

    if kind == "improvement":
        print(f"coverage: {pct:.1f}% ↑ {_format_signed_pct(pct_diff)} {detail}")
    else:
        print(f"⚠ coverage: {pct:.1f}% ↓ {_format_signed_pct(pct_diff)} {detail}")


def _update_baseline(current: dict[str, float | int]) -> None:
    payload = {
        "percent": round(float(current["percent"]), 2),
        "covered": int(current["covered"]),
        "missing": int(current["missing"]),
        "statements": int(current["statements"]),
    }
    BASELINE_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Baseline updated: coverage {payload['percent']:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write current totals to .coverage-baseline and exit.",
    )
    args = parser.parse_args()

    current = _current_totals()

    if args.update_baseline:
        _update_baseline(current)
        return

    baseline = _load_baseline()
    _print_summary(current, baseline)


if __name__ == "__main__":
    main()
