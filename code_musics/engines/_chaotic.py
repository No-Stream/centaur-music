"""Chaotic audio-rate oscillators.

Four RK4-integrated ODE attractors — Lorenz, Rössler, Duffing, Chua —
projected to a 1-D waveform and peak-normalized.  ``rate_hz`` scales the
integration step so the same attractor can play at sub-audio drift rates
or up to near-Nyquist evolution.  ``amount`` blends between a periodic /
stable regime (0) and the canonical chaotic regime (1); ``symmetry`` adds
a DC bias that breaks the attractor's reflective symmetry.
"""

from __future__ import annotations

from typing import Final

import numba
import numpy as np

from code_musics.humanize import seed_or_default

SUPPORTED_SYSTEMS: Final[frozenset[str]] = frozenset(
    {"lorenz", "rossler", "duffing", "chua"}
)

# Empirical per-system ``rate_hz`` ceilings where RK4 integration stays finite
# under ``amount=1`` for 10-second runs. The attractors accelerate into
# numerical instability past these values; callers get a finite peak but the
# spectrum degrades well before the limit. Published as the documented
# ceiling on ``rate_hz`` so boundary tests can pin the contract.
RATE_HZ_CEILINGS: Final[dict[str, float]] = {
    "lorenz": 200.0,
    "rossler": 400.0,
    "duffing": 800.0,
    "chua": 150.0,
}


# ---------------------------------------------------------------------------
# Integration kernels (numba-njit).  Each kernel writes the projected
# output coordinate into ``out`` over ``n_samples`` steps.  The caller
# supplies ``dt`` (integration step in seconds) plus system params.
# ---------------------------------------------------------------------------


@numba.njit(cache=True, fastmath=True)
def _lorenz_loop(
    out: np.ndarray,
    x0: float,
    y0: float,
    z0: float,
    sigma: float,
    rho: float,
    beta: float,
    symmetry: float,
    dt: float,
    n_samples: int,
) -> None:
    """RK4 integration of the Lorenz attractor, projecting x."""
    x = x0
    y = y0
    z = z0
    for i in range(n_samples):
        dx1 = sigma * (y - x) + symmetry
        dy1 = x * (rho - z) - y
        dz1 = x * y - beta * z
        x2 = x + 0.5 * dt * dx1
        y2 = y + 0.5 * dt * dy1
        z2 = z + 0.5 * dt * dz1
        dx2 = sigma * (y2 - x2) + symmetry
        dy2 = x2 * (rho - z2) - y2
        dz2 = x2 * y2 - beta * z2
        x3 = x + 0.5 * dt * dx2
        y3 = y + 0.5 * dt * dy2
        z3 = z + 0.5 * dt * dz2
        dx3 = sigma * (y3 - x3) + symmetry
        dy3 = x3 * (rho - z3) - y3
        dz3 = x3 * y3 - beta * z3
        x4 = x + dt * dx3
        y4 = y + dt * dy3
        z4 = z + dt * dz3
        dx4 = sigma * (y4 - x4) + symmetry
        dy4 = x4 * (rho - z4) - y4
        dz4 = x4 * y4 - beta * z4

        x += dt * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4) / 6.0
        y += dt * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4) / 6.0
        z += dt * (dz1 + 2.0 * dz2 + 2.0 * dz3 + dz4) / 6.0

        out[i] = x


@numba.njit(cache=True, fastmath=True)
def _rossler_loop(
    out: np.ndarray,
    x0: float,
    y0: float,
    z0: float,
    a: float,
    b: float,
    c: float,
    symmetry: float,
    dt: float,
    n_samples: int,
) -> None:
    """RK4 integration of the Rössler system, projecting x."""
    x = x0
    y = y0
    z = z0
    for i in range(n_samples):
        dx1 = -y - z + symmetry
        dy1 = x + a * y
        dz1 = b + z * (x - c)

        x2 = x + 0.5 * dt * dx1
        y2 = y + 0.5 * dt * dy1
        z2 = z + 0.5 * dt * dz1
        dx2 = -y2 - z2 + symmetry
        dy2 = x2 + a * y2
        dz2 = b + z2 * (x2 - c)

        x3 = x + 0.5 * dt * dx2
        y3 = y + 0.5 * dt * dy2
        z3 = z + 0.5 * dt * dz2
        dx3 = -y3 - z3 + symmetry
        dy3 = x3 + a * y3
        dz3 = b + z3 * (x3 - c)

        x4 = x + dt * dx3
        y4 = y + dt * dy3
        z4 = z + dt * dz3
        dx4 = -y4 - z4 + symmetry
        dy4 = x4 + a * y4
        dz4 = b + z4 * (x4 - c)

        x += dt * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4) / 6.0
        y += dt * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4) / 6.0
        z += dt * (dz1 + 2.0 * dz2 + 2.0 * dz3 + dz4) / 6.0

        out[i] = x


@numba.njit(cache=True, fastmath=True)
def _duffing_loop(
    out: np.ndarray,
    x0: float,
    v0: float,
    damping: float,
    alpha: float,
    beta: float,
    drive_amp: float,
    drive_freq: float,
    symmetry: float,
    dt: float,
    n_samples: int,
) -> None:
    """RK4 integration of the driven Duffing oscillator, projecting x.

    Equation:
    ``x'' + damping*x' + alpha*x + beta*x^3 = drive_amp*cos(drive_freq*t) + symmetry``
    """
    x = x0
    v = v0
    t = 0.0
    for i in range(n_samples):
        dx1 = v
        dv1 = (
            -damping * v
            - alpha * x
            - beta * x * x * x
            + drive_amp * np.cos(drive_freq * t)
            + symmetry
        )
        x2 = x + 0.5 * dt * dx1
        v2 = v + 0.5 * dt * dv1
        t2 = t + 0.5 * dt
        dx2 = v2
        dv2 = (
            -damping * v2
            - alpha * x2
            - beta * x2 * x2 * x2
            + drive_amp * np.cos(drive_freq * t2)
            + symmetry
        )
        x3 = x + 0.5 * dt * dx2
        v3 = v + 0.5 * dt * dv2
        dx3 = v3
        dv3 = (
            -damping * v3
            - alpha * x3
            - beta * x3 * x3 * x3
            + drive_amp * np.cos(drive_freq * t2)
            + symmetry
        )
        x4 = x + dt * dx3
        v4 = v + dt * dv3
        t4 = t + dt
        dx4 = v4
        dv4 = (
            -damping * v4
            - alpha * x4
            - beta * x4 * x4 * x4
            + drive_amp * np.cos(drive_freq * t4)
            + symmetry
        )

        x += dt * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4) / 6.0
        v += dt * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4) / 6.0
        t += dt

        out[i] = x


@numba.njit(cache=True, fastmath=True)
def _chua_nonlinearity(x: float, m0: float, m1: float, breakpoint: float) -> float:
    """Piecewise-linear Chua diode characteristic."""
    return m1 * x + 0.5 * (m0 - m1) * (abs(x + breakpoint) - abs(x - breakpoint))


@numba.njit(cache=True, fastmath=True)
def _chua_loop(
    out: np.ndarray,
    x0: float,
    y0: float,
    z0: float,
    alpha: float,
    beta: float,
    m0: float,
    m1: float,
    breakpoint: float,
    symmetry: float,
    dt: float,
    n_samples: int,
) -> None:
    """RK4 integration of Chua's circuit, projecting x (double-scroll)."""
    x = x0
    y = y0
    z = z0
    for i in range(n_samples):
        hx1 = _chua_nonlinearity(x, m0, m1, breakpoint)
        dx1 = alpha * (y - x - hx1) + symmetry
        dy1 = x - y + z
        dz1 = -beta * y
        x2 = x + 0.5 * dt * dx1
        y2 = y + 0.5 * dt * dy1
        z2 = z + 0.5 * dt * dz1
        hx2 = _chua_nonlinearity(x2, m0, m1, breakpoint)
        dx2 = alpha * (y2 - x2 - hx2) + symmetry
        dy2 = x2 - y2 + z2
        dz2 = -beta * y2
        x3 = x + 0.5 * dt * dx2
        y3 = y + 0.5 * dt * dy2
        z3 = z + 0.5 * dt * dz2
        hx3 = _chua_nonlinearity(x3, m0, m1, breakpoint)
        dx3 = alpha * (y3 - x3 - hx3) + symmetry
        dy3 = x3 - y3 + z3
        dz3 = -beta * y3
        x4 = x + dt * dx3
        y4 = y + dt * dy3
        z4 = z + dt * dz3
        hx4 = _chua_nonlinearity(x4, m0, m1, breakpoint)
        dx4 = alpha * (y4 - x4 - hx4) + symmetry
        dy4 = x4 - y4 + z4
        dz4 = -beta * y4

        x += dt * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4) / 6.0
        y += dt * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4) / 6.0
        z += dt * (dz1 + 2.0 * dz2 + 2.0 * dz3 + dz4) / 6.0

        out[i] = x


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_chaotic(
    *,
    system: str,
    rate_hz: float,
    amount: float,
    symmetry: float,
    n_samples: int,
    sample_rate: int,
    seed: int,
    initial_conditions: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Render a chaotic oscillator as a peak-normalized waveform.

    Args:
        system: One of ``"lorenz"``, ``"rossler"``, ``"duffing"``, ``"chua"``.
        rate_hz: Nominal evolution rate in Hz.  Scales the integrator's step
            size so that the attractor's fundamental motion lands near this
            frequency.  Higher values move through phase space faster and
            produce brighter output.  Must be positive.
        amount: ``[0, 1]`` — 0 collapses toward the system's stable/periodic
            regime; 1 uses the fully chaotic canonical parameters.
        symmetry: ``[-1, +1]`` — DC-like offset injected into the drive term
            of each system.  Non-zero values break the attractor's natural
            reflective symmetry and bias the output waveform.
        n_samples: Output length in samples.
        sample_rate: Host sample rate (Hz).
        seed: Seed for deterministic initial conditions.
        initial_conditions: Optional explicit state override.  When ``None``
            the initial state is derived deterministically from ``seed``.

    Returns:
        1-D ``float64`` array of length ``n_samples``, peak-normalized to
        ``<= 1.0`` in magnitude.  For ``n_samples == 0`` returns an empty
        array.

    Raises:
        ValueError: Unknown ``system`` name, non-positive ``rate_hz`` or
            ``sample_rate``, ``amount`` outside ``[0, 1]``, ``symmetry``
            outside ``[-1, 1]``, or initial conditions of the wrong length.
    """
    if system not in SUPPORTED_SYSTEMS:
        raise ValueError(
            f"chaotic system must be one of {sorted(SUPPORTED_SYSTEMS)}, got {system!r}"
        )
    if rate_hz <= 0.0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz!r}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
    if not 0.0 <= amount <= 1.0:
        raise ValueError(f"amount must be in [0, 1], got {amount!r}")
    if not -1.0 <= symmetry <= 1.0:
        raise ValueError(f"symmetry must be in [-1, 1], got {symmetry!r}")
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float64)

    out = np.empty(n_samples, dtype=np.float64)

    # Deterministic initial conditions from seed.
    rng = np.random.default_rng(seed)
    default_ic = rng.uniform(-0.1, 0.1, size=3).astype(np.float64)

    # Each system normalizes rate_hz to its own natural circulation frequency.
    # Step size chosen so that one "characteristic revolution" spans roughly
    # ``sample_rate / rate_hz`` samples.

    if system == "lorenz":
        # Canonical params at amount=1.  At amount=0 we dial rho down under
        # the Hopf bifurcation threshold so trajectories spiral into a fixed
        # point (periodic/damped).  sigma and beta remain canonical.
        sigma = 10.0
        rho = 5.0 + amount * 23.0  # 5 -> 28
        beta = 8.0 / 3.0
        dt = rate_hz / sample_rate
        ic = _resolve_ic(initial_conditions, default_ic, length=3)
        # Kick the state off the stable equilibrium so low-amount still moves.
        _lorenz_loop(
            out,
            float(ic[0]) + 0.5,
            float(ic[1]),
            float(ic[2]) + 0.5,
            sigma,
            rho,
            beta,
            symmetry * 5.0,
            dt,
            n_samples,
        )

    elif system == "rossler":
        # Canonical params at amount=1: a=0.2, b=0.2, c=5.7.  At amount=0
        # pull c down to 2.0 where the system is a clean limit cycle.
        a = 0.1 + amount * 0.1  # 0.1 -> 0.2
        b = 0.2
        c = 2.0 + amount * 3.7  # 2.0 -> 5.7
        dt = rate_hz / sample_rate
        ic = _resolve_ic(initial_conditions, default_ic, length=3)
        _rossler_loop(
            out,
            float(ic[0]) + 1.0,
            float(ic[1]),
            float(ic[2]),
            a,
            b,
            c,
            symmetry * 2.0,
            dt,
            n_samples,
        )

    elif system == "duffing":
        # Driven Duffing.  At amount=0 we use high damping + small drive
        # so the well-settled response is near-periodic; at amount=1 we get
        # the classic chaotic regime (damping=0.3, alpha=-1, beta=1,
        # drive_amp=0.5, drive_freq=1.2).
        damping = 0.7 - amount * 0.4  # 0.7 -> 0.3
        alpha = -1.0
        beta = 1.0
        drive_amp = 0.2 + amount * 0.3  # 0.2 -> 0.5
        drive_freq = 1.2
        dt = rate_hz / sample_rate
        ic = _resolve_ic(initial_conditions, default_ic, length=2)
        _duffing_loop(
            out,
            float(ic[0]),
            float(ic[1]),
            damping,
            alpha,
            beta,
            drive_amp,
            drive_freq,
            symmetry * 0.4,
            dt,
            n_samples,
        )

    else:  # "chua"
        # Chua's double-scroll.  Canonical (amount=1): alpha=15.6, beta=28,
        # m0=-1.143, m1=-0.714, breakpoint=1.  Lower amount pulls alpha
        # down, which collapses the attractor to a period-1 limit cycle.
        alpha = 6.0 + amount * 9.6  # 6 -> 15.6
        beta = 28.0
        m0 = -1.143
        m1 = -0.714
        breakpoint = 1.0
        dt = rate_hz / sample_rate
        ic = _resolve_ic(initial_conditions, default_ic, length=3)
        _chua_loop(
            out,
            float(ic[0]) + 0.1,
            float(ic[1]),
            float(ic[2]) + 0.1,
            alpha,
            beta,
            m0,
            m1,
            breakpoint,
            symmetry * 2.0,
            dt,
            n_samples,
        )

    if not np.all(np.isfinite(out)):
        raise RuntimeError(
            f"chaotic system {system!r} diverged (rate_hz={rate_hz}, "
            f"amount={amount}, symmetry={symmetry}); try a smaller rate_hz"
        )

    peak = float(np.max(np.abs(out)))
    if peak > 1e-12:
        out = out / peak
    return out


def _resolve_ic(
    user_ic: tuple[float, ...] | None,
    default_ic: np.ndarray,
    *,
    length: int,
) -> np.ndarray:
    """Validate and promote the caller-supplied initial conditions."""
    if user_ic is None:
        return default_ic[:length].astype(np.float64)
    if len(user_ic) != length:
        raise ValueError(
            f"initial_conditions must have length {length}, got {len(user_ic)}"
        )
    return np.asarray(user_ic, dtype=np.float64)


def default_seed(seed: int | None, *parts: object) -> int:
    """Public wrapper around ``seed_or_default`` for module-level callers."""
    return seed_or_default(seed, *parts)
