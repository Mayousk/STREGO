"""Solving the global-phase bi-objective problem.
1. :func:`solve_biobjective` -- run NSMA on ``[mu, -IVR]`` starting from a Sobol
   pool, and return the non-dominated front.
2. :func:`select_batch` -- choose ``batch_size`` points *from* that front.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.cluster import KMeans

from .biobjective import BiObjectiveProblem

# NSMA search hyperparameters. These are the reference implementation's values
# and the ones every campaign in the paper used.
NSMA_MAX_ITER = 20
NSMA_POP_SIZE = 100
NSMA_CROSSOVER_PROBABILITY = 0.9
NSMA_ETA = 20.0


def _import_nsma():
    """Import NSMA. Must be handled on its own since NSMA requires graph mode and modern TensorFlow starts in eager mode."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()

    from nsma.algorithms.memetic.nsma import NSMA
    from nsma.general_utils.pareto_utils import pareto_efficient

    return NSMA, pareto_efficient


def solve_biobjective(
    model,
    Xn_pool: np.ndarray,
    objective_pair: str = "MU_IVR",
    ivr_integration_points: int = 64,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run NSMA on the GP's bi-objective problem, seeded with ``Xn_pool``.

    Returns ``(pareto_x, pareto_f)`` in normalized [0, 1]^d coordinates. The
    front can be empty if the search degenerates.
    """
    NSMA, pareto_efficient = _import_nsma()
    dim = Xn_pool.shape[1]

    problem = BiObjectiveProblem(
        dim=dim,
        model=model,
        objective_pair=objective_pair,
        ivr_integration_points=ivr_integration_points,
        seed=seed,
    )

    
    F = np.array([problem.evaluate_functions(Xn_pool[i]) for i in range(len(Xn_pool))])

    solver = NSMA(
        max_iter=NSMA_MAX_ITER,
        max_time=None,
        max_f_evals=None,
        verbose=False,
        verbose_interspace=10,
        plot_pareto_front=False,
        plot_pareto_solutions=False,
        plot_dpi=100,
        pop_size=NSMA_POP_SIZE,
        crossover_probability=NSMA_CROSSOVER_PROBABILITY,
        crossover_eta=NSMA_ETA,
        mutation_eta=NSMA_ETA,
        shift=np.inf,
        crowding_quantile=0.9,
        n_opt=5,
        FMOPG_max_iter=5,
        theta_for_stationarity=-1e-10,
        theta_tol=-1e-1,
        theta_dec_factor=10 ** (-0.5),
        gurobi=False,
        gurobi_method=1,
        gurobi_verbose=False,
        ALS_alpha_0=1,
        ALS_delta=0.5,
        ALS_beta=1e-4,
        ALS_min_alpha=1e-7,
    )

    result_x, result_f, _ = solver.search(np.asarray(Xn_pool, dtype=float), F, problem)
    result_x = np.asarray(result_x, dtype=float)
    result_f = np.asarray(result_f, dtype=float)

    efficient = pareto_efficient(result_f)
    return result_x[efficient], result_f[efficient]


def select_batch(
    pareto_x: np.ndarray,
    pareto_f: np.ndarray,
    batch_size: int,
    Xn_pool: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Pick ``batch_size`` points from the Pareto front.
    """
    n_pareto = pareto_x.shape[0]

    if n_pareto == 0:
        # NSMA returned nothing usable; keep the run alive with random points.
      
        idx = np.random.choice(len(Xn_pool), size=batch_size, replace=False)
        return Xn_pool[idx]

    if batch_size == 1:
        best = int(np.argmin(pareto_f[:, 0]))
        return pareto_x[best : best + 1]

    if batch_size >= n_pareto:
        return pareto_x


    # k-means the front in design space, then snap each centroid to its nearest
    # real front member (centroids themselves are not Pareto-optimal).
    km = KMeans(n_clusters=batch_size, n_init=10, random_state=seed)
    km.fit(pareto_x)
    selected = [
        pareto_x[int(np.argmin(np.linalg.norm(pareto_x - c, axis=1)))] for c in km.cluster_centers_
    ]
    return np.asarray(selected, dtype=float)
