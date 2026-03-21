"""Helpers for constructing additive spectral profiles."""

from __future__ import annotations

from collections.abc import Sequence


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
