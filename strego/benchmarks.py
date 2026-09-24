"""Synthetic benchmarks used in the paper, with their shifted variants.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


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


_CANONICAL_OPT = {"rastrigin": 0.0, "alpine01": 0.0, "ackley": 0.0, "schwefel": 420.9687}
_SHIFT_IDX = {"rastrigin": 0, "alpine01": 1, "schwefel": 2, "ackley": 3}
_SHIFT_SEED = 12345



def make_full_shift(
    func_name: str, dim: int, lower: np.ndarray, upper: np.ndarray, band: float = 0.2
) -> np.ndarray:
 
    rng = np.random.RandomState(_SHIFT_SEED + _SHIFT_IDX.get(func_name, 0) * 1000 + dim)
    span = upper - lower
    p = rng.uniform(lower + band * span, upper - band * span)
    return p - _CANONICAL_OPT.get(func_name, 0.0)

_SCHWEFEL_SHIFT_BAND = (-140.0, -125.0)


def make_schwefel_boundary_shift(dim: int, band: tuple[float, float] = _SCHWEFEL_SHIFT_BAND) -> np.ndarray:
    rng = np.random.RandomState(_SHIFT_SEED + _SHIFT_IDX["schwefel"] * 1000 + dim)
    return rng.uniform(band[0], band[1], size=dim)


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
