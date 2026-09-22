"""Small numerical helpers shared by the optimizer, the GP models and the runners.

Everything here is deliberately dependency-light: the optimizer spends most of
its time inside BoTorch/NSMA, so these helpers are the parts we want to stay
obvious and cheap.
"""

from __future__ import annotations

import os
import random
from typing import Iterable

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Seed every RNG the optimizer touches.

    We deliberately do *not* enable ``torch.use_deterministic_algorithms``: it
    makes some BoTorch fitting paths raise instead of falling back, and the
    campaign numbers were produced without it.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize(X: np.ndarray, lower: np.ndarray, upper: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Map points from the problem box into the unit cube."""
    scale = np.maximum(upper - lower, eps)
    return (np.asarray(X, dtype=float) - lower) / scale


def denormalize(Xn: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Inverse of :func:`normalize`."""
    return np.asarray(Xn, dtype=float) * (upper - lower) + lower


def distance_to_box(X: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """L2 distance from each row of ``X`` to the box, 0 for points inside it."""
    below = np.clip(lower - X, 0.0, None)
    above = np.clip(X - upper, 0.0, None)
    return np.linalg.norm(below + above, axis=1)


def unique_rows_tol(X: np.ndarray, y: np.ndarray, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Drop duplicate rows (within ``tol``) while preserving the original order.

    Duplicates come from the trust region revisiting points; feeding them to a GP
    makes the covariance matrix singular, so we strip them before every fit.
    """
    if X.size == 0:
        return X, y
    keys = np.round(np.asarray(X, dtype=float) / tol).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx = np.sort(idx)
    return X[idx], y[idx]


def is_close_to_any(x: np.ndarray, points: Iterable[np.ndarray], tol: float = 1e-6) -> bool:
    """True when ``x`` duplicates a point we have already evaluated.

    Used to avoid burning budget re-evaluating the incumbent once the trust
    region has collapsed around it.
    """
    x = np.asarray(x, dtype=float)
    for point in points:
        if np.linalg.norm(x - np.asarray(point, dtype=float)) <= tol:
            return True
    return False
