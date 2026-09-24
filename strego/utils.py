
from __future__ import annotations

import os
import random
from typing import Iterable

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """
    Seed every random number generator to ensure reproducibility.
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
    scale = np.maximum(upper - lower, eps)
    return (np.asarray(X, dtype=float) - lower) / scale


def denormalize(Xn: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.asarray(Xn, dtype=float) * (upper - lower) + lower


def distance_to_box(X: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    below = np.clip(lower - X, 0.0, None)
    above = np.clip(X - upper, 0.0, None)
    return np.linalg.norm(below + above, axis=1)


def unique_rows_tol(X: np.ndarray, y: np.ndarray, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Drop duplicate rows (within ``tol``) while preserving the original order.
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
