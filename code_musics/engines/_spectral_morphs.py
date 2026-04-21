"""Vital-style spectral morph operators for the additive engine.

Five frequency-domain morph operators that act on the per-partial
``(ratio, amp, [phase])`` array *before* resynthesis.  The morphs are
re-implementations of the standard transforms popularised by Vital's
wavetable oscillator (``WavetableOscillator``'s spectrum-space frame
operators), adapted to operate on this library's explicit partial bank
rather than a wavetable frame.  No verbatim GPL-3 source was copied;
the transforms are classic spectral-domain operations described in
Vital's documentation and in general signal-processing literature.

Also provides sigma-approximation band-limiting (Lanczos sigma factors),
following the standard Fourier-series approach described in MZ2SYNTH's
``SOURCE/wvecmp.f90`` (applying ``sinc(k/(K+1))`` factors before the
inverse DFT to reduce Gibbs ringing).  The transform itself is textbook
MIT-compatible.

All morphs:

- Are pure functions that return a *new* list of partial dicts.
- Accept and preserve arbitrary optional keys on the partial (``noise``,
  ``envelope``, etc).  The explicit keys that matter to the morphs are
  ``ratio``, ``amp``, and (for ``phase_disperse``) ``phase``.
- Reduce to identity at ``amount == 0``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

MORPH_TYPES: tuple[str, ...] = (
    "none",
    "inharmonic_scale",
    "phase_disperse",
    "smear",
    "shepard",
    "random_amplitudes",
)

_RANDOM_AMPLITUDE_STAGES: int = 16


def _copy_partials(partials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shallow-copy each partial dict so morphs do not mutate inputs."""
    return [dict(entry) for entry in partials]


_INHARMONIC_AMOUNT_FLOOR: float = -1.0 + 1e-6


def apply_inharmonic_scale(
    partials: list[dict[str, Any]],
    *,
    amount: float,
) -> list[dict[str, Any]]:
    """Stretch the harmonic grid by a log-weighted amount.

    At ``amount == 0`` the spectrum is unchanged.  Positive ``amount`` stretches
    higher partials away from a pure harmonic ratio (piano-style stiffness at
    small amounts; inharmonic drift at larger).  Negative ``amount`` compresses
    upper partials.  The fundamental (ratio closest to 1.0) stays in place.

    Formula::

        new_ratio[k] = ratio[k] * (1 + amount * log2(k_rank) / log2(k_max))

    where ``k_rank`` is the 1-indexed rank of each partial's ratio (smallest
    ratio = rank 1) and ``k_max`` is the count of partials.  Rank-based
    weighting ensures the stretching curve is applied to spectral position
    regardless of the input list's order; partials are returned in their
    original input order.  ``log2(1) = 0`` ensures the bottom partial is
    unaffected, and higher-ranked partials scale progressively more.

    ``amount`` is clamped at ``-1 + 1e-6`` from below to guarantee strictly
    positive output ratios (``math.log2`` downstream in ``apply_shepard`` would
    otherwise crash on zero/negative ratios).  There is no upper clamp.
    """
    if amount == 0.0 or len(partials) <= 1:
        return _copy_partials(partials)

    clamped_amount = max(_INHARMONIC_AMOUNT_FLOOR, float(amount))

    result = _copy_partials(partials)
    n = len(result)
    if n <= 1:
        return result

    log2_kmax = math.log2(n)
    if log2_kmax <= 0.0:
        return result

    ratios = np.asarray([float(entry["ratio"]) for entry in result], dtype=np.float64)
    # argsort -> position in sorted order; invert to get rank of each input index.
    sort_order = np.argsort(ratios, kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    ranks[sort_order] = np.arange(n, dtype=np.int64)

    for input_index, entry in enumerate(result):
        k_rank = int(ranks[input_index]) + 1  # 1-indexed rank
        weight = math.log2(k_rank) / log2_kmax
        entry["ratio"] = float(entry["ratio"]) * (1.0 + clamped_amount * weight)

    return result


def apply_phase_disperse(
    partials: list[dict[str, Any]],
    *,
    amount: float,
    center_k: int = 24,
) -> list[dict[str, Any]]:
    """Add a quadratic per-partial phase offset centered on ``center_k``.

    Produces a "spread" character in the waveform without changing the
    magnitude spectrum.  Tuned for small ``amount`` values (0-0.05 typical);
    larger values wrap around and create more chaotic-sounding dispersions.

    Formula::

        phase[k] += sin((k - center_k)^2 * amount) * 2 * pi

    The ``phase`` key is written onto each partial (zeroed where absent).
    """
    result = _copy_partials(partials)
    if amount == 0.0:
        for entry in result:
            entry.setdefault("phase", 0.0)
        return result

    two_pi = 2.0 * math.pi
    for k_index, entry in enumerate(result):
        # k-1 relative to the center (partial index is 0-based; center_k in
        # partial-number units).  center_k - 1 is the 0-indexed center.
        delta = k_index - (center_k - 1)
        phase_offset = math.sin((delta * delta) * amount) * two_pi
        base_phase = float(entry.get("phase", 0.0))
        entry["phase"] = base_phase + phase_offset

    return result


def apply_smear(
    partials: list[dict[str, Any]],
    *,
    amount: float,
) -> list[dict[str, Any]]:
    """Recursively smear amplitude energy toward upper partials.

    A first-order running mixer that leaks a fraction of each partial's
    amplitude into its upper neighbour.  Creates a softly-pink-shifted spread
    of overtones without touching ratios.

    Formula (applied in ascending ratio order)::

        new_amp[k+1] = (1 - amount) * amp[k+1] + amount * amp[k] * (1 + 0.25/k)

    ``amount == 0`` returns the input unchanged.  ``amount == 1`` gives a fully
    propagated spread (equivalent to a low-pass moving average in harmonic
    space).
    """
    if amount == 0.0 or len(partials) <= 1:
        return _copy_partials(partials)

    amount = max(0.0, min(1.0, amount))
    result = _copy_partials(partials)
    for k_index in range(1, len(result)):
        prev_amp = float(result[k_index - 1]["amp"])
        current_amp = float(result[k_index]["amp"])
        scale = 1.0 + 0.25 / float(k_index)
        new_amp = (1.0 - amount) * current_amp + amount * prev_amp * scale
        result[k_index]["amp"] = max(0.0, new_amp)

    return result


def apply_shepard(
    partials: list[dict[str, Any]],
    *,
    amount: float,
    shift: float,
) -> list[dict[str, Any]]:
    """Amplitude crossfade toward an octave-shifted ghost copy.

    For each partial at ``ratio[k]`` we find the partner at ``ratio[k] * 2^shift``
    (in the *existing* partial set) and blend::

        new_amp[k] = (1 - amount) * amp[k] + amount * ghost_amp[k]

    When no exact partner exists the nearest-ratio partial is used.  Shift
    outside the partial range resolves to silence on the ghost side, so
    sustained blends with out-of-bounds shifts just fade that portion to 0.

    Simple v1: skips fancy amplitude-ratio-aware phase interpolation; does an
    amplitude-only crossfade.  Partners sufficiently close in amplitude may
    null at their crossover — document that and revisit later.
    """
    result = _copy_partials(partials)
    if amount == 0.0 or shift == 0.0 or len(result) == 0:
        return result

    amount = max(0.0, min(1.0, amount))
    shift_factor = 2.0 ** float(shift)
    ratios = np.asarray([float(entry["ratio"]) for entry in result], dtype=np.float64)
    amps = np.asarray([float(entry["amp"]) for entry in result], dtype=np.float64)

    for k_index, entry in enumerate(result):
        ghost_ratio = ratios[k_index] * shift_factor
        # Nearest partner in log-space (octave-aware).
        log_diffs = np.abs(np.log2(ratios) - math.log2(ghost_ratio))
        nearest_index = int(np.argmin(log_diffs))
        # Only use the ghost if the partner ratio is reasonably close in log-space
        # (within a minor third, ~0.25 octaves).  Otherwise treat ghost as silent.
        if log_diffs[nearest_index] <= 0.25:
            ghost_amp = float(amps[nearest_index])
        else:
            ghost_amp = 0.0

        blended = (1.0 - amount) * float(amps[k_index]) + amount * ghost_amp
        entry["amp"] = max(0.0, blended)

    return result


def apply_random_amplitudes(
    partials: list[dict[str, Any]],
    *,
    amount: float,
    shift: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Apply a stable, seeded 16-stage interpolated random amplitude mask.

    Generates ``_RANDOM_AMPLITUDE_STAGES`` random amplitude vectors (one per
    partial per stage) under a fixed ``seed``, then linearly interpolates
    between two adjacent stages based on ``shift in [0, 1]`` (wrapped
    circularly so shift=1.0 maps back to shift=0.0).  Blends the resulting
    random mask against a flat mask (= 1.0) by ``amount`` so that
    ``amount == 0`` is identity and ``amount == 1`` uses the full mask.

    Formula per partial::

        mask = lerp(stage[floor(s * N)], stage[ceil(s * N)], frac)
        new_amp = amp * ((1 - amount) * 1.0 + amount * mask)

    This mirrors Vital's stable-random wavetable morphs, adapted for our
    explicit partial bank.
    """
    result = _copy_partials(partials)
    if amount == 0.0 or len(result) == 0:
        return result

    amount = max(0.0, min(1.0, amount))
    shift = float(shift) % 1.0
    rng = np.random.default_rng(np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF))
    stages = rng.uniform(0.0, 1.0, size=(_RANDOM_AMPLITUDE_STAGES, len(result)))

    scaled = shift * _RANDOM_AMPLITUDE_STAGES
    base_index = int(math.floor(scaled)) % _RANDOM_AMPLITUDE_STAGES
    next_index = (base_index + 1) % _RANDOM_AMPLITUDE_STAGES
    frac = scaled - math.floor(scaled)
    mask = (1.0 - frac) * stages[base_index] + frac * stages[next_index]

    for k_index, entry in enumerate(result):
        base_amp = float(entry["amp"])
        blended = (1.0 - amount) * base_amp + amount * base_amp * float(mask[k_index])
        entry["amp"] = max(0.0, blended)

    return result


def apply_sigma_approximation(
    partials: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Multiply each partial's amplitude by a Lanczos sigma factor.

    Reduces Gibbs ringing from hard truncation of a Fourier series at
    near-zero cost.  The factor is::

        sigma[k] = sin(pi * k / (K + 1)) / (pi * k / (K + 1)) == sinc(k / (K + 1))

    where ``K`` is the highest partial index and ``k`` runs from 1 up.  The
    fundamental is attenuated least; the top partial is attenuated most.

    This is the standard Lanczos sigma approximation; see MZ2SYNTH
    ``SOURCE/wvecmp.f90`` for a reference Fortran implementation.
    """
    if len(partials) == 0:
        return _copy_partials(partials)

    result = _copy_partials(partials)
    max_k = len(result)
    denom = float(max_k + 1)
    for k_index, entry in enumerate(result, start=1):
        x = k_index / denom
        # np.sinc(y) == sin(pi*y)/(pi*y); use as-is since numpy normalises by pi.
        sigma = float(np.sinc(x))
        entry["amp"] = max(0.0, float(entry["amp"]) * sigma)
    return result


def _dispatch_none(
    partials: list[dict[str, Any]],
    *,
    amount: float,
    shift: float,
    center_k: int,
    seed: int,
) -> list[dict[str, Any]]:
    del amount, shift, center_k, seed
    return _copy_partials(partials)


def _dispatch_inharmonic_scale(
    partials: list[dict[str, Any]],
    *,
    amount: float,
    shift: float,
    center_k: int,
    seed: int,
) -> list[dict[str, Any]]:
    del shift, center_k, seed
    return apply_inharmonic_scale(partials, amount=amount)


def _dispatch_phase_disperse(
    partials: list[dict[str, Any]],
    *,
    amount: float,
    shift: float,
    center_k: int,
    seed: int,
) -> list[dict[str, Any]]:
    del shift, seed
    return apply_phase_disperse(partials, amount=amount, center_k=center_k)


def _dispatch_smear(
    partials: list[dict[str, Any]],
    *,
    amount: float,
    shift: float,
    center_k: int,
    seed: int,
) -> list[dict[str, Any]]:
    del shift, center_k, seed
    return apply_smear(partials, amount=amount)


def _dispatch_shepard(
    partials: list[dict[str, Any]],
    *,
    amount: float,
    shift: float,
    center_k: int,
    seed: int,
) -> list[dict[str, Any]]:
    del center_k, seed
    return apply_shepard(partials, amount=amount, shift=shift)


def _dispatch_random_amplitudes(
    partials: list[dict[str, Any]],
    *,
    amount: float,
    shift: float,
    center_k: int,
    seed: int,
) -> list[dict[str, Any]]:
    del center_k
    return apply_random_amplitudes(partials, amount=amount, shift=shift, seed=seed)


_MORPH_DISPATCH: dict[str, Any] = {
    "none": _dispatch_none,
    "inharmonic_scale": _dispatch_inharmonic_scale,
    "phase_disperse": _dispatch_phase_disperse,
    "smear": _dispatch_smear,
    "shepard": _dispatch_shepard,
    "random_amplitudes": _dispatch_random_amplitudes,
}


def apply_spectral_morph(
    partials: list[dict[str, Any]],
    *,
    morph_type: str,
    amount: float = 0.0,
    shift: float = 0.0,
    center_k: int = 24,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Dispatch to the requested spectral morph.

    ``morph_type == "none"`` is identity (ignoring amount).  Unknown types
    raise ``ValueError`` so misconfigured params fail fast.  This is the single
    authoritative validation point for ``morph_type``.
    """
    dispatcher = _MORPH_DISPATCH.get(morph_type)
    if dispatcher is None:
        raise ValueError(
            f"Unsupported spectral_morph_type: {morph_type!r}. "
            f"Expected one of {list(MORPH_TYPES)}."
        )
    return dispatcher(
        partials,
        amount=amount,
        shift=shift,
        center_k=center_k,
        seed=seed,
    )
