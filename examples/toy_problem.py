"""
Tests STREGO on a toy problem
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strego import STREGO


def my_objective(x: np.ndarray) -> float:
    """Any callable taking a 1-D array and returning a float works.

    STREGO minimizes, so return a loss. The objective may be noisy -- the
    trust-region loop is built for it.
    """
    return float(np.sum((x - 0.7) ** 2) + 0.1 * np.sum(np.cos(8.0 * x)))


def main() -> None:
    dim = 8

    optimizer = STREGO(
        objective_fn=my_objective,
        lower_bounds=np.full(dim, -2.0),
        upper_bounds=np.full(dim, 2.0),
        budget=60,       # total objective evaluations, including the design
        n_init=15,       # initial Latin-hypercube design
        seed=0,
        log_path="logs/quickstart.csv",
    )

    result = optimizer.run()

    print(f"best_y      : {result.best_y:.6f}")
    print(f"best_x      : {np.round(result.best_x, 3)}")
    print(f"evaluations : {result.n_evaluations}")
    print(f"trace       : {result.log_path}")


if __name__ == "__main__":
    main()
