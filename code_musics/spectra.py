"""Helpers for constructing additive spectral profiles."""

from __future__ import annotations

import math
from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Physical model mode tables
# ---------------------------------------------------------------------------

# Circular drumhead modes: zeros of Bessel functions (normalized to fundamental).
_MEMBRANE_MODES: list[float] = [
    1.000,
    1.594,
    2.136,
    2.296,
    2.653,
    2.918,
    3.156,
    3.501,
    3.600,
    3.652,
    4.060,
    4.154,
    4.601,
    4.680,
    4.904,
    5.132,
]

# Euler-Bernoulli free-free bar modes (marimba/xylophone ratios).
_BAR_MODES: list[float] = [
    1.000,
    2.757,
    5.404,
    8.933,
    13.344,
    18.637,
    24.812,
    31.869,
]

_BAR_MATERIAL_DAMPING: dict[str, float] = {
    "wood": 0.4,
    "metal": 0.15,
    "glass": 0.25,
}

# Singing bowl / Tibetan bowl modes.
_BOWL_MODES: list[float] = [
    1.000,
    2.71,
    5.04,
    7.92,
    11.34,
    15.30,
    19.81,
    24.86,
]

# Stopped-pipe (neither end open) slightly shifted partials.
_STOPPED_PIPE_MODES: list[float] = [
    1.0,
    2.08,
    3.15,
    4.22,
    5.30,
    6.37,
    7.44,
    8.52,
    9.59,
    10.67,
    11.74,
    12.81,
    13.89,
    14.96,
    16.04,
    17.11,
]


def ratio_spectrum(
    ratios: Sequence[float],
    amps: Sequence[float] | None = None,
) -> list[dict[str, float]]:
    """Build an explicit additive spectrum from ratio and amplitude lists."""
    if not ratios:
        raise ValueError("ratios must not be empty")
    if amps is not None and len(amps) != len(ratios):
        raise ValueError("amps must match ratios length")

    resolved_amps = (
        [1.0] * len(ratios) if amps is None else [float(amp) for amp in amps]
    )
    spectrum: list[dict[str, float]] = []
    for ratio, amp in zip(ratios, resolved_amps, strict=True):
        resolved_ratio = float(ratio)
        if resolved_ratio <= 0.0:
            raise ValueError("ratios must be strictly positive")
        if amp < 0.0:
            raise ValueError("amps must be non-negative")
        spectrum.append({"ratio": resolved_ratio, "amp": amp})
    return spectrum


def harmonic_spectrum(
    *,
    n_partials: int,
    harmonic_rolloff: float = 0.5,
    brightness_tilt: float = 0.0,
    odd_even_balance: float = 0.0,
) -> list[dict[str, float]]:
    """Build a harmonic-series spectrum matching the additive engine defaults."""
    if n_partials < 1:
        raise ValueError("n_partials must be at least 1")

    clamped_odd_even_balance = max(-0.95, min(0.95, float(odd_even_balance)))
    spectrum: list[dict[str, float]] = []
    for harmonic_index in range(1, n_partials + 1):
        amp = float(harmonic_rolloff) ** (harmonic_index - 1)
        if brightness_tilt != 0.0:
            amp *= harmonic_index ** float(brightness_tilt)
        if harmonic_index % 2 == 0:
            amp *= 1.0 - clamped_odd_even_balance
        else:
            amp *= 1.0 + clamped_odd_even_balance
        if amp <= 0.0:
            continue
        spectrum.append({"ratio": float(harmonic_index), "amp": amp})
    return spectrum


def stretched_spectrum(
    *,
    n_partials: int,
    stretch_exponent: float,
    harmonic_rolloff: float = 0.5,
    brightness_tilt: float = 0.0,
) -> list[dict[str, float]]:
    """Build a stretched or compressed overtone family."""
    if n_partials < 1:
        raise ValueError("n_partials must be at least 1")
    if stretch_exponent <= 0.0:
        raise ValueError("stretch_exponent must be positive")

    return ratio_spectrum(
        ratios=[
            float(partial_index) ** float(stretch_exponent)
            for partial_index in range(1, n_partials + 1)
        ],
        amps=[
            (float(harmonic_rolloff) ** (partial_index - 1))
            * (partial_index ** float(brightness_tilt))
            for partial_index in range(1, n_partials + 1)
        ],
    )


# ---------------------------------------------------------------------------
# Physical model spectra
# ---------------------------------------------------------------------------


def membrane_spectrum(
    *,
    n_modes: int = 12,
    damping: float = 0.3,
) -> list[dict[str, float]]:
    """Build a circular drumhead spectrum from Bessel function zeros.

    Mode ratios are the well-known zeros of Bessel functions for an ideal
    circular membrane, normalized so the fundamental is 1.0.

    ``damping`` controls exponential amplitude rolloff across modes:
    ``amp_i = exp(-damping * i)`` where *i* is the 0-indexed mode number.
    """
    if n_modes < 1:
        raise ValueError("n_modes must be at least 1")
    if damping < 0.0:
        raise ValueError("damping must be non-negative")
    if n_modes > len(_MEMBRANE_MODES):
        raise ValueError(
            f"n_modes must be at most {len(_MEMBRANE_MODES)} (available membrane modes)"
        )

    spectrum: list[dict[str, float]] = []
    for i in range(n_modes):
        amp = math.exp(-damping * i)
        spectrum.append({"ratio": _MEMBRANE_MODES[i], "amp": amp})
    return spectrum


def bar_spectrum(
    *,
    n_modes: int = 8,
    material: str = "wood",
) -> list[dict[str, float]]:
    """Build a vibrating bar spectrum from Euler-Bernoulli beam theory.

    Mode ratios follow the classic marimba/xylophone free-free bar pattern.

    ``material`` controls amplitude rolloff:

    - ``"wood"``: fast decay (wood absorbs high modes)
    - ``"metal"``: slow decay (metal sustains high modes)
    - ``"glass"``: moderate decay
    """
    if n_modes < 1:
        raise ValueError("n_modes must be at least 1")
    if n_modes > len(_BAR_MODES):
        raise ValueError(
            f"n_modes must be at most {len(_BAR_MODES)} (available bar modes)"
        )
    if material not in _BAR_MATERIAL_DAMPING:
        valid = ", ".join(sorted(_BAR_MATERIAL_DAMPING))
        raise ValueError(f"material must be one of: {valid}")

    damping = _BAR_MATERIAL_DAMPING[material]
    spectrum: list[dict[str, float]] = []
    for i in range(n_modes):
        amp = math.exp(-damping * i)
        spectrum.append({"ratio": _BAR_MODES[i], "amp": amp})
    return spectrum


def plate_spectrum(
    *,
    n_modes: int = 12,
    aspect_ratio: float = 1.0,
) -> list[dict[str, float]]:
    """Build a rectangular plate spectrum from plate vibration theory.

    For a free rectangular plate, mode frequencies follow:
    ``f_{m,n} = C * (m^2/a^2 + n^2/b^2)`` where ``a/b = aspect_ratio``.

    ``aspect_ratio=1.0`` gives a square plate with maximally degenerate modes.
    Other values break the degeneracy for richer spectra.

    Amplitude: ``1.0 / (m * n)`` (higher mode pairs are weaker).
    """

    if n_modes < 1:
        raise ValueError("n_modes must be at least 1")
    if aspect_ratio <= 0.0:
        raise ValueError("aspect_ratio must be positive")

    search_limit = math.ceil(math.sqrt(n_modes * 2))
    a = aspect_ratio
    b = 1.0

    raw_modes: list[tuple[float, float]] = []
    for m in range(1, search_limit + 1):
        for n in range(1, search_limit + 1):
            freq = (m**2) / (a**2) + (n**2) / (b**2)
            amp = 1.0 / (m * n)
            raw_modes.append((freq, amp))

    raw_modes.sort(key=lambda pair: pair[0])

    if not raw_modes:
        raise ValueError("no plate modes generated")

    min_freq = raw_modes[0][0]
    spectrum: list[dict[str, float]] = []
    for freq, amp in raw_modes[:n_modes]:
        spectrum.append({"ratio": freq / min_freq, "amp": amp})
    return spectrum


def tube_spectrum(
    *,
    n_modes: int = 8,
    open_ends: str = "both",
) -> list[dict[str, float]]:
    """Build a cylindrical tube spectrum for wind instrument modeling.

    ``open_ends`` controls the harmonic structure:

    - ``"both"`` (flute-like): all harmonics ``[1, 2, 3, 4, ...]``
    - ``"one"`` (clarinet-like): odd harmonics only ``[1, 3, 5, 7, ...]``
    - ``"neither"`` (stopped pipe): slightly shifted partials
    """
    valid_open_ends = {"both", "one", "neither"}
    if n_modes < 1:
        raise ValueError("n_modes must be at least 1")
    if open_ends not in valid_open_ends:
        valid = ", ".join(sorted(valid_open_ends))
        raise ValueError(f"open_ends must be one of: {valid}")

    if open_ends == "both":
        ratios = [float(i + 1) for i in range(n_modes)]
    elif open_ends == "one":
        ratios = [float(2 * i + 1) for i in range(n_modes)]
    else:
        if n_modes > len(_STOPPED_PIPE_MODES):
            raise ValueError(
                f"n_modes must be at most {len(_STOPPED_PIPE_MODES)} "
                f"for open_ends='neither'"
            )
        ratios = _STOPPED_PIPE_MODES[:n_modes]

    spectrum: list[dict[str, float]] = []
    for i, ratio in enumerate(ratios):
        amp = 1.0 / (1.0 + 0.3 * i)
        spectrum.append({"ratio": ratio, "amp": amp})
    return spectrum


def bowl_spectrum(
    *,
    n_modes: int = 8,
) -> list[dict[str, float]]:
    """Build a singing bowl / Tibetan bowl spectrum.

    Mode ratios are from well-documented singing bowl acoustics measurements.
    Amplitude rolloff: ``1.0 / (1 + 0.5 * i)`` (moderate rolloff).
    """
    if n_modes < 1:
        raise ValueError("n_modes must be at least 1")
    if n_modes > len(_BOWL_MODES):
        raise ValueError(
            f"n_modes must be at most {len(_BOWL_MODES)} (available bowl modes)"
        )

    spectrum: list[dict[str, float]] = []
    for i in range(n_modes):
        amp = 1.0 / (1.0 + 0.5 * i)
        spectrum.append({"ratio": _BOWL_MODES[i], "amp": amp})
    return spectrum


# ---------------------------------------------------------------------------
# Merge utility
# ---------------------------------------------------------------------------


def _cents_distance(ratio_a: float, ratio_b: float) -> float:
    """Absolute interval distance in cents between two frequency ratios."""

    if ratio_a <= 0.0 or ratio_b <= 0.0:
        return float("inf")
    return abs(1200.0 * math.log2(ratio_a / ratio_b))


_MERGE_CENTS_THRESHOLD = 10.0


def _merge_partials(
    partials: list[dict[str, float]],
    *,
    cents_threshold: float = _MERGE_CENTS_THRESHOLD,
    max_partials: int | None = None,
) -> list[dict[str, float]]:
    """Merge near-coincident partials and optionally cap the count.

    Partials within *cents_threshold* cents are combined by summing amplitudes
    and amplitude-weighted geometric mean of ratios.  The result is sorted by
    ratio and pruned to *max_partials* loudest entries when given.
    """
    if not partials:
        return []

    sorted_partials = sorted(partials, key=lambda p: p["ratio"])
    merged: list[dict[str, float]] = [
        {"ratio": sorted_partials[0]["ratio"], "amp": sorted_partials[0]["amp"]}
    ]

    for partial in sorted_partials[1:]:
        prev = merged[-1]
        if _cents_distance(partial["ratio"], prev["ratio"]) <= cents_threshold:
            total_amp = prev["amp"] + partial["amp"]
            if total_amp > 0.0:
                weighted_log = (
                    prev["amp"] * math.log2(prev["ratio"])
                    + partial["amp"] * math.log2(partial["ratio"])
                ) / total_amp
                weighted_ratio = 2.0**weighted_log
            else:
                weighted_ratio = prev["ratio"]
            prev["ratio"] = weighted_ratio
            prev["amp"] = total_amp
        else:
            merged.append({"ratio": partial["ratio"], "amp": partial["amp"]})

    if max_partials is not None and len(merged) > max_partials:
        merged.sort(key=lambda p: p["amp"], reverse=True)
        merged = merged[:max_partials]

    merged.sort(key=lambda p: p["ratio"])
    return merged


# ---------------------------------------------------------------------------
# Spectral convolution
# ---------------------------------------------------------------------------


def spectral_convolve(
    spec_a: list[dict[str, float]],
    spec_b: list[dict[str, float]],
    *,
    max_partials: int = 32,
    merge_tolerance_cents: float = _MERGE_CENTS_THRESHOLD,
    min_amp_db: float = -60.0,
) -> list[dict[str, float]]:
    """Compute the spectral convolution (cross-product) of two partial lists.

    Every pair ``(a, b)`` produces a product partial at ``a.ratio * b.ratio``
    with amplitude ``a.amp * b.amp``.  Near-coincident partials (within
    ``merge_tolerance_cents``) are merged via amplitude-weighted geometric mean
    of their ratios and summed amplitudes.

    This is musically powerful for JI spectra: cross-products of JI ratios
    remain in the same ratio family.

    The result is pruned below ``min_amp_db`` relative to the peak, capped at
    ``max_partials``, and normalized so peak amplitude is 1.0.
    """

    if not spec_a or not spec_b:
        raise ValueError("both input spectra must be non-empty")

    products: list[dict[str, float]] = []
    for a in spec_a:
        for b in spec_b:
            products.append(
                {"ratio": a["ratio"] * b["ratio"], "amp": a["amp"] * b["amp"]}
            )

    merged = _merge_partials(products, cents_threshold=merge_tolerance_cents)

    peak_amp = max((p["amp"] for p in merged), default=0.0)
    if peak_amp <= 0.0:
        raise ValueError("spectral convolution produced no audible partials")

    amp_floor = peak_amp * (10.0 ** (min_amp_db / 20.0))
    pruned = [p for p in merged if p["amp"] >= amp_floor]

    pruned.sort(key=lambda p: p["amp"], reverse=True)
    pruned = pruned[:max_partials]

    pruned.sort(key=lambda p: p["ratio"])

    return [{"ratio": p["ratio"], "amp": p["amp"] / peak_amp} for p in pruned]


# ---------------------------------------------------------------------------
# Fractal spectra
# ---------------------------------------------------------------------------


def fractal_spectrum(
    seed: list[dict[str, float]],
    *,
    depth: int = 2,
    level_rolloff: float = 0.5,
    max_partials: int = 32,
) -> list[dict[str, float]]:
    """Build a self-similar spectrum via iterated self-convolution.

    Starting from *seed* (level 0), each subsequent level convolves *seed*
    with the previous level and scales amplitudes by ``level_rolloff ** level``.
    All levels are merged into a single spectrum, normalized so peak amp = 1.0,
    and capped at *max_partials*.
    """
    if not seed:
        raise ValueError("seed must not be empty")
    if depth < 0:
        raise ValueError("depth must be non-negative")
    if level_rolloff <= 0.0:
        raise ValueError("level_rolloff must be positive")
    if max_partials < 1:
        raise ValueError("max_partials must be at least 1")

    if depth == 0:
        result = [{"ratio": p["ratio"], "amp": p["amp"]} for p in seed]
        peak = max(p["amp"] for p in result)
        if peak > 0.0:
            for p in result:
                p["amp"] /= peak
        return result[:max_partials]

    all_partials: list[dict[str, float]] = [
        {"ratio": p["ratio"], "amp": p["amp"]} for p in seed
    ]
    previous_level = list(seed)

    for level in range(1, depth + 1):
        new_level = spectral_convolve(seed, previous_level)
        scale = level_rolloff**level
        for p in new_level:
            p["amp"] *= scale
        all_partials.extend(new_level)
        previous_level = new_level

    result = _merge_partials(all_partials, max_partials=max_partials)

    peak = max((p["amp"] for p in result), default=0.0)
    if peak > 0.0:
        for p in result:
            p["amp"] /= peak

    return result


# ---------------------------------------------------------------------------
# Formant shaping
# ---------------------------------------------------------------------------

_VOWEL_FORMANTS: dict[str, list[tuple[float, float, float]]] = {
    "a": [(800, 1.0, 80), (1200, 0.8, 90), (2500, 0.5, 120)],
    "i": [(270, 1.0, 60), (2300, 0.7, 100), (3000, 0.4, 120)],
    "e": [(530, 1.0, 70), (1850, 0.7, 100), (2500, 0.4, 120)],
    "o": [(500, 1.0, 70), (800, 0.8, 80), (2800, 0.3, 120)],
    "u": [(300, 1.0, 60), (870, 0.6, 90), (2250, 0.3, 120)],
}


def vowel_formants(name: str) -> list[tuple[float, float, float]]:
    """Return formant data (center_hz, gain, bandwidth_hz) for a named vowel."""
    key = name.lower().strip()
    if key not in _VOWEL_FORMANTS:
        raise ValueError(
            f"unknown vowel {name!r}, expected one of {sorted(_VOWEL_FORMANTS)}"
        )
    return list(_VOWEL_FORMANTS[key])


def formant_weight(
    abs_freq: float,
    formants: list[tuple[float, float, float]],
) -> float:
    """Sum of Gaussian resonance peaks at *abs_freq* for the given formants."""
    weight = 0.0
    for center_hz, gain, bandwidth_hz in formants:
        z = (abs_freq - center_hz) / bandwidth_hz
        weight += gain * math.exp(-0.5 * z * z)
    return weight


def formant_shape(
    partials: list[dict[str, float]],
    f0: float,
    formants: list[tuple[float, float, float]] | str,
    *,
    bandwidth_hz: float = 100.0,
) -> list[dict[str, float]]:
    """Shape a partial spectrum through formant resonance peaks.

    *formants* is either a vowel name string (looked up via
    :func:`vowel_formants`) or an explicit list of
    ``(center_hz, gain, bandwidth_hz)`` tuples.  When *formants* is a
    string, the per-formant bandwidths are overridden by *bandwidth_hz*.
    """
    if isinstance(formants, str):
        resolved = [
            (center, gain, bandwidth_hz)
            for center, gain, _bw in vowel_formants(formants)
        ]
    else:
        resolved = list(formants)

    shaped: list[dict[str, float]] = []
    for partial in partials:
        abs_freq = partial["ratio"] * f0
        weight = formant_weight(abs_freq, resolved)
        shaped.append({"ratio": partial["ratio"], "amp": partial["amp"] * weight})
    return shaped


def formant_morph(
    partials: list[dict[str, float]],
    f0: float,
    vowel_sequence: list[str | list[tuple[float, float, float]]],
    morph_times: list[float] | None = None,
) -> list[dict[str, float | list]]:
    """Generate per-partial envelopes that morph between formant shapes.

    Each entry in *vowel_sequence* is either a vowel name string or an
    explicit formant list.  The returned partials carry an ``"envelope"``
    key with time/value keyframes suitable for the additive engine's
    per-partial envelope feature.  The ``"amp"`` field is preserved at its
    original value (the envelope modulates it).
    """
    if not vowel_sequence:
        raise ValueError("vowel_sequence must not be empty")
    if morph_times is not None and len(morph_times) != len(vowel_sequence):
        raise ValueError("morph_times length must match vowel_sequence length")

    resolved_formants: list[list[tuple[float, float, float]]] = []
    for entry in vowel_sequence:
        if isinstance(entry, str):
            resolved_formants.append(vowel_formants(entry))
        else:
            resolved_formants.append(list(entry))

    n_vowels = len(vowel_sequence)
    if morph_times is not None:
        times = [float(t) for t in morph_times]
    elif n_vowels == 1:
        times = [0.0]
    else:
        times = [i / (n_vowels - 1) for i in range(n_vowels)]

    weight_table: list[list[float]] = []
    for formant_set in resolved_formants:
        weights_for_vowel: list[float] = []
        for partial in partials:
            abs_freq = partial["ratio"] * f0
            weights_for_vowel.append(formant_weight(abs_freq, formant_set))
        weight_table.append(weights_for_vowel)

    result: list[dict[str, float | list]] = []
    for partial_idx, partial in enumerate(partials):
        envelope: list[dict[str, float]] = []
        for vowel_idx, t in enumerate(times):
            envelope.append({"time": t, "value": weight_table[vowel_idx][partial_idx]})
        result.append(
            {
                "ratio": partial["ratio"],
                "amp": partial["amp"],
                "envelope": envelope,
            }
        )
    return result
