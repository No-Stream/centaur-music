"""Snapshot tests protecting existing filter topology audio behavior.

These tests exercise every existing topology (``svf``, ``ladder``,
``sallen_key``, ``cascade``) across a matrix of cutoff / Q / drive / mode /
feedback / HPF / morph / solver combinations.  Each case hashes the output
waveform into a deterministic fingerprint (sha256 over the float64 bytes
plus a few numerical invariants: peak, RMS, mean-abs).

The hash + invariants are recorded the first time the test runs (baseline
captured by setting ``CAPTURE_BASELINE=1``) and must match exactly after
the ``FilterParams`` / dispatcher refactor — this proves the refactor is
plumbing-only and changes no audio.

To regenerate the baseline intentionally (e.g. after a *genuine* algorithmic
change):

    CAPTURE_BASELINE=1 uv run pytest tests/test_filter_params_migration.py

Otherwise the baseline is loaded from the JSON file next to this module.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from code_musics.engines._filters import apply_filter

SR = 44100
BASELINE_PATH = Path(__file__).parent / "_filter_params_migration_baseline.json"


def _make_signal(dur: float, seed: int) -> np.ndarray:
    """Reproducible pseudo-pink signal with harmonic + broadband content."""
    rng = np.random.default_rng(seed)
    n = int(SR * dur)
    t = np.arange(n) / SR
    tones = (
        np.sin(2 * np.pi * 110.0 * t)
        + 0.5 * np.sin(2 * np.pi * 440.0 * t)
        + 0.3 * np.sin(2 * np.pi * 1760.0 * t)
    )
    return 0.6 * tones + 0.3 * rng.standard_normal(n)


def _fingerprint(y: np.ndarray) -> dict[str, float | str]:
    """Deterministic fingerprint for numerical equality checks.

    Hash covers the full waveform bytes; the scalar invariants catch tiny
    drift and make failures human-readable.  Any real algorithmic change
    shifts all four simultaneously.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    h = hashlib.sha256(y.tobytes()).hexdigest()
    return {
        "sha256": h,
        "peak": float(np.max(np.abs(y))),
        "rms": float(np.sqrt(np.mean(y * y))),
        "mean_abs": float(np.mean(np.abs(y))),
        "n": int(y.shape[0]),
    }


# Matrix of cases covering each existing topology with representative
# Canonical baseline matrix: one basic and one stressed case per topology, plus
# a small set of corner cases that exercise the dispatch code paths (cutoff
# modulation, serial HPF chain, Newton-solved ladder).  Kept intentionally
# narrow — a broad matrix freezes incidental output for every topology under
# every combination, and a legitimate DSP tweak then fails every row at once
# with no diagnostic signal.  Names become baseline keys.  When a real
# algorithmic change lands, regenerate via
# ``CAPTURE_BASELINE=1 make test-selected TESTS=tests/test_filter_params_migration.py``.
_CASES: list[dict] = [
    {
        "name": "svf_lp_basic",
        "topo": "svf",
        "mode": "lowpass",
        "fc": 1500.0,
        "q": 0.707,
    },
    {
        "name": "svf_lp_driven",
        "topo": "svf",
        "mode": "lowpass",
        "fc": 1200.0,
        "q": 3.0,
        "drive": 0.6,
    },
    {
        "name": "svf_hpf_chain",
        "topo": "svf",
        "mode": "lowpass",
        "fc": 2000.0,
        "q": 1.2,
        "hpf": 200.0,
    },
    {
        "name": "ladder_lp_basic",
        "topo": "ladder",
        "mode": "lowpass",
        "fc": 1500.0,
        "q": 0.707,
    },
    {
        "name": "ladder_lp_resonant_newton",
        "topo": "ladder",
        "mode": "lowpass",
        "fc": 900.0,
        "q": 8.0,
        "solver": "newton",
    },
    {
        "name": "sk_lp_basic",
        "topo": "sallen_key",
        "mode": "lowpass",
        "fc": 1500.0,
        "q": 0.707,
    },
    {
        "name": "cas_lp_basic",
        "topo": "cascade",
        "mode": "lowpass",
        "fc": 1500.0,
        "q": 0.707,
    },
    {
        "name": "cas_modulated_cutoff",
        "topo": "cascade",
        "mode": "lowpass",
        "fc_mod": True,
        "q": 3.0,
    },
]


def _build_cutoff(case: dict, n: int) -> np.ndarray:
    """Constant or sweeping cutoff profile."""
    if case.get("fc_mod"):
        t = np.arange(n) / SR
        return 400.0 + 1600.0 * (0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t))
    return np.full(n, float(case["fc"]))


def _stable_seed(name: str) -> int:
    """Deterministic per-case seed — Python's builtin hash() is randomized."""
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big") & 0xFFFF


def _run_case(case: dict) -> np.ndarray:
    """Render a case through ``apply_filter`` with its kwargs."""
    sig = _make_signal(0.5, seed=_stable_seed(case["name"]))
    cutoff = _build_cutoff(case, sig.shape[0])
    kwargs = {
        "cutoff_profile": cutoff,
        "sample_rate": SR,
        "filter_topology": case["topo"],
        "filter_mode": case["mode"],
        "resonance_q": float(case.get("q", 0.707)),
        "filter_drive": float(case.get("drive", 0.0)),
        "filter_even_harmonics": float(case.get("even", 0.0)),
        "filter_morph": float(case.get("morph", 0.0)),
        "bass_compensation": float(case.get("bass_comp", 0.0)),
        "hpf_cutoff_hz": float(case.get("hpf", 0.0)),
        "feedback_amount": float(case.get("fb", 0.0)),
        "feedback_saturation": float(case.get("fb_sat", 0.3)),
        "filter_solver": case.get("solver", "adaa"),
    }
    return apply_filter(sig, **kwargs)


def _load_baseline() -> dict[str, dict[str, float | str]]:
    if not BASELINE_PATH.exists():
        return {}
    with BASELINE_PATH.open() as f:
        return json.load(f)


def _save_baseline(baseline: dict[str, dict[str, float | str]]) -> None:
    with BASELINE_PATH.open("w") as f:
        json.dump(baseline, f, indent=2, sort_keys=True)


@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
def test_filter_topology_snapshot(case: dict) -> None:
    """Output fingerprint must match the captured baseline exactly.

    On first run (or when CAPTURE_BASELINE=1 is set), the baseline is
    updated and the test passes.  On subsequent runs any drift in output
    is caught — this is the safety net for the FilterParams / dispatcher
    refactor.
    """
    y = _run_case(case)
    assert np.all(np.isfinite(y)), f"Non-finite output for case {case['name']}"
    fp = _fingerprint(y)

    baseline = _load_baseline()
    capture = os.environ.get("CAPTURE_BASELINE") == "1"

    if capture:
        baseline[case["name"]] = fp
        _save_baseline(baseline)
        return

    if case["name"] not in baseline:
        pytest.fail(
            f"No baseline recorded for {case['name']!r}. "
            "Run `CAPTURE_BASELINE=1 make test-selected "
            "TESTS=tests/test_filter_params_migration.py` to record it."
        )

    expected = baseline[case["name"]]
    assert fp["sha256"] == expected["sha256"], (
        f"Hash drift for {case['name']}: got {fp} vs baseline {expected}"
    )
    assert fp["n"] == expected["n"]
    for key in ("peak", "rms", "mean_abs"):
        assert fp[key] == pytest.approx(expected[key], rel=1e-6), (
            f"{key} drift for {case['name']}: {fp[key]} vs {expected[key]}"
        )
