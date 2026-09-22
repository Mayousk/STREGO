"""Synthetic benchmarks used in the paper, with their shifted variants.

Four multimodal functions (rastrigin, alpine01, ackley, schwefel) plus the
"full-shift" variants that move the optimum off the domain centre.

Why shifting matters
--------------------
Rastrigin, alpine01 and ackley all put their optimum at the origin, which is the
exact centre of their canonical boxes. Any method with a centre bias -- and that
includes IVR, whose integration grid is uniform over the box, so interior
candidates score higher than edge candidates purely from geometry -- gets an
unearned advantage there. The full-shift variants relocate the optimum to a
random interior point, per function and per dimension but deterministic given
the seed, so the comparison measures search rather than luck.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Base functions (all minimized; all take a 1-D array and return a float)
# ---------------------------------------------------------------------------


def ackley(x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi) -> float:
    x = np.asarray(x, dtype=float)
    dim = x.size
    term1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / dim))
    term2 = -np.exp(np.sum(np.cos(c * x)) / dim)
    return float(term1 + term2 + a + np.e)


def rastrigin(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(10.0 * x.size + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))


def alpine(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(np.abs(x * np.sin(x) + 0.1 * x)))


def schwefel(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(418.9829 * x.size - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


BOUNDS: dict[str, tuple[float, float]] = {
    "rastrigin": (-5.12, 5.12),
    "alpine01": (-10.0, 10.0),
    "ackley": (-32.768, 32.768),
    "schwefel": (-500.0, 500.0),
}

BASE_FUNCTIONS: dict[str, Callable[[np.ndarray], float]] = {
    "rastrigin": rastrigin,
    "alpine01": alpine,
    "ackley": ackley,
    "schwefel": schwefel,
}

# Canonical optimum location, per coordinate. Needed because the shift is derived
# from the *desired* optimum: schwefel's optimum sits at 420.9687, not 0, so
# treating the shift itself as the new optimum would push it out of bounds.
_CANONICAL_OPT = {"rastrigin": 0.0, "alpine01": 0.0, "ackley": 0.0, "schwefel": 420.9687}
_SHIFT_IDX = {"rastrigin": 0, "alpine01": 1, "schwefel": 2, "ackley": 3}
_SHIFT_SEED = 12345


# ---------------------------------------------------------------------------
# Shifts
# ---------------------------------------------------------------------------


def make_full_shift(
    func_name: str, dim: int, lower: np.ndarray, upper: np.ndarray, band: float = 0.2
) -> np.ndarray:
    """Shift every coordinate so the optimum lands at a random interior point.

    The target optimum ``p`` is drawn from the central ``[band, 1-band]`` region
    of the box and the shift is ``s = p - x_canonical_opt``, so ``f(x - s)``
    attains its minimum exactly at ``p``, in bounds.

    Safe for rastrigin/alpine01/ackley only: those are non-negative with minimum
    0, so the excursion ``z = x - s`` outside the canonical domain cannot
    manufacture a lower minimum. Schwefel is unbounded below and gets its own
    shift (see :func:`make_schwefel_boundary_shift`).
    """
    rng = np.random.RandomState(_SHIFT_SEED + _SHIFT_IDX.get(func_name, 0) * 1000 + dim)
    span = upper - lower
    p = rng.uniform(lower + band * span, upper - band * span)
    return p - _CANONICAL_OPT.get(func_name, 0.0)


# Schwefel's optimum sits at normalized position 0.921 of its box -- in every
# coordinate at once, i.e. jammed into a corner. That systematically penalises
# the [mu, -IVR] objective, whose integration grid is uniform on the box: a
# corner candidate has its kernel neighbourhood truncated by the boundary and so
# scores a structurally lower IVR than an interior one. The fix is a negative
# shift sliding the optimum inward.
#
# Unlike the other three, schwefel's per-dimension term is unbounded below, so
# the excursion beyond the canonical box can create a *better* minimum. The
# binding constraint is z_max = 640: the minimum of f over the excursion
# (500, 640] is 313.6, comfortably above the box's own second-best per-dimension
# local minimum (118.4 at z = -302.5), so the excursion is no more attractive
# than structure the box already contains. That gives s >= -140; the band below
# keeps a margin and lands the optimum at normalized 0.781-0.796 -- inside the
# same central band the other three draw from, and ~2.7x further from the wall
# than the unshifted 0.079.
_SCHWEFEL_SHIFT_BAND = (-140.0, -125.0)


def make_schwefel_boundary_shift(dim: int, band: tuple[float, float] = _SCHWEFEL_SHIFT_BAND) -> np.ndarray:
    """Inward shift for schwefel, drawn per coordinate.

    Per coordinate rather than as a single scalar, so the shifted optimum loses
    schwefel's degenerate "same value in every coordinate" structure -- matching
    how the other three functions are treated.
    """
    rng = np.random.RandomState(_SHIFT_SEED + _SHIFT_IDX["schwefel"] * 1000 + dim)
    return rng.uniform(band[0], band[1], size=dim)


# ---------------------------------------------------------------------------
# Public lookup
# ---------------------------------------------------------------------------

FUNCTIONS = tuple(BASE_FUNCTIONS) + tuple(f"{n}_fullshift" for n in BASE_FUNCTIONS)


def get_function_spec(name: str, dim: int) -> tuple[Callable[[np.ndarray], float], np.ndarray, np.ndarray]:
    """Return ``(objective_fn, lower_bounds, upper_bounds)`` for a benchmark."""
    if name in BASE_FUNCTIONS:
        lo, hi = BOUNDS[name]
        return BASE_FUNCTIONS[name], np.full(dim, lo), np.full(dim, hi)

    if name.endswith("_fullshift"):
        base_name = name[: -len("_fullshift")]
        if base_name not in BASE_FUNCTIONS:
            raise ValueError(f"Unsupported function: {name}")
        base_fn = BASE_FUNCTIONS[base_name]
        lo, hi = BOUNDS[base_name]
        lower, upper = np.full(dim, lo), np.full(dim, hi)
        shift = (
            make_schwefel_boundary_shift(dim)
            if base_name == "schwefel"
            else make_full_shift(base_name, dim, lower, upper)
        )
        return (lambda x, f=base_fn, s=shift: f(np.asarray(x, dtype=float) - s)), lower, upper

    raise ValueError(f"Unsupported function: {name}")


def wrap_noisy(base_fn: Callable, noise_type: str, noise_std: float) -> Callable:
    """Wrap an objective in observation noise.

    ``additive`` models a fixed-scale measurement error; ``multiplicative``
    models relative error, which is the realistic one when the objective spans
    orders of magnitude (a simulator's runtime, say).
    """
    if noise_type == "none":
        return lambda x: float(base_fn(x))
    if noise_type == "additive":
        return lambda x: float(base_fn(x) + noise_std * np.random.randn())
    if noise_type == "multiplicative":
        return lambda x: float(base_fn(x) * (1.0 + noise_std * np.random.randn()))
    raise ValueError(f"Unsupported noise_type: {noise_type}")
