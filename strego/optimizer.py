"""STREGO: Scalable Efficient Global Optimization.

STREGO runs a trust-region loop (see :mod:`strego.trust_region`) whose two
phases are both GP-driven, but which are deliberately given different jobs:

**Global phase** -- fit a GP on every observation so far, draw a Sobol candidate
pool over the whole box, and solve the bi-objective problem ``[mu, -IVR]`` on
that pool with NSMA. The resulting Pareto front is the set of exploit/explore
compromises the surrogate considers defensible; we k-means it and evaluate a
small batch (3 by default). This phase is where exploration happens, and it is
*not* restricted to the trust region.

**Local phase** -- only runs when the global batch failed to improve
sufficiently. Collect the observations inside the trust region, relaxing the
region until it holds at least ``n_min`` data points, fit an independant GP on that
local design, and minimize the posterior mean inside the (unrelaxed) trust
region. Pure exploitation, because the global phase already covered exploration.

Example
-------
>>> import numpy as np
>>> from strego import STREGO
>>> opt = STREGO(
...     objective_fn=lambda x: float(np.sum(x ** 2)),
...     lower_bounds=np.full(10, -5.0),
...     upper_bounds=np.full(10, 5.0),
...     budget=120,
... )
>>> result = opt.run()
>>> print(result.best_y)"""


from __future__ import annotations

import csv
import os
import time
from typing import Callable, Optional

import numpy as np
import torch
from botorch.optim import optimize_acqf
from scipy.stats import qmc
from torch.quasirandom import SobolEngine

from .acquisition import NegativePosteriorMean
from .models import fit_gp
from .selection import select_batch, solve_biobjective
from .trust_region import OptimizationResult, TrustRegionBO
from .utils import (
    denormalize,
    distance_to_box,
    ensure_directory,
    is_close_to_any,
    normalize,
    set_all_seeds,
    unique_rows_tol,
)

DEFAULTS: dict = {
    "n_init": 10,
    "global_batch_size": 3,
    "min_local_points": 10,
    "relaxation_step": 0.5,
    "candidate_pool_size": 250,
    "local_num_restarts": None,
    "local_raw_samples": None,
    "objective_pair": "MU_IVR",
    "ivr_integration_points": 64,
    "beta_1": 0.7,
    "beta_2": 0.5,
    "gamma": 2.0,
    "kappa": 1.0,
    "d_max": 1.0,
}

#Default values used in the article
def default_num_restarts(dim: int) -> int:
    """Multi-start restarts for the local acquisition: ``2d + 4``(This is the default value for the article. Recommend lowering for test)"""
    return 2 * dim + 4


def default_raw_samples(dim: int) -> int:
    """Prescreen pool for the local acquisition: ``(2d + 4)^2``(This is the default value for the article. Recommend lowering for test)."""
    return (2 * dim + 4) ** 2


def default_sigma_0(dim: int) -> float:
    """Initial radius: ``0.5 * (1/5)^(1/d)``."""

    return 0.5 * ((1.0 / 5.0) ** (1.0 / dim))


class STREGO(TrustRegionBO):
    """The STREGO optimizer.

    Parameters
    ----------
    objective_fn : callable
        ``f(x: np.ndarray) -> float``, minimized. May be noisy.
    lower_bounds, upper_bounds : array-like, shape (d,)
    budget : int
        Total objective evaluations, including the initial design.
    n_init : int
        Size of the initial Latin-hypercube design.
    global_batch_size : int
        Points drawn from the Pareto front per global phase. 3 by default.
    min_local_points : int
        ``n_min``: the minimum size of the local design the local GP is fitted
        on. If the trust region holds fewer observations, it is relaxed until
        it holds at least this many. Must satisfy ``2 <= n_min <= n_init``.
    relaxation_step : float
        Increment (> 0) of the relaxation factor. At relaxation step ``j`` the
        local-design box has radius ``(1 + j * relaxation_step)`` times the
        trust-region radius: 1x (the trust region itself), 1.5x, 2.0x, ... with
        the default 0.5.
    local_num_restarts, local_raw_samples : int, optional
        Local acquisition optimizer: a Sobol pool of ``local_raw_samples``
        points is scored and the best ``local_num_restarts`` seed L-BFGS.
        Default ``2d + 4`` and ``(2d + 4)^2``. 
    candidate_pool_size : int
        Sobol pool seeding NSMA. Default ``8 * d``.
    objective_pair : {"MU_IVR", "MU_SIGMA"}
        Global-phase bi-objective. See :mod:`strego.biobjective`.
    doe_points, doe_values : optional
        Inject a precomputed initial design instead of sampling one. Passing
        ``doe_values`` as well reuses the stored objective values rather than
        re-evaluating, which is how campaigns share one design across solvers.
    deterministic : bool
        Skip the certain/uncertain failure split (noise-free objectives).
    """

    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], float],
        lower_bounds,
        upper_bounds,
        budget: int,
        n_init: int = DEFAULTS["n_init"],
        *,
        global_batch_size: int = DEFAULTS["global_batch_size"],
        min_local_points: int = DEFAULTS["min_local_points"],
        relaxation_step: float = DEFAULTS["relaxation_step"],
        candidate_pool_size: int = DEFAULTS["candidate_pool_size"],
        local_num_restarts: Optional[int] = DEFAULTS["local_num_restarts"],
        local_raw_samples: Optional[int] = DEFAULTS["local_raw_samples"],
        objective_pair: str = DEFAULTS["objective_pair"],
        ivr_integration_points: int = DEFAULTS["ivr_integration_points"],
        sigma_0: Optional[float] = None,
        beta_1: float = DEFAULTS["beta_1"],
        beta_2: float = DEFAULTS["beta_2"],
        gamma: float = DEFAULTS["gamma"],
        kappa: float = DEFAULTS["kappa"],
        d_max: float = DEFAULTS["d_max"],
        deterministic: bool = False,
        doe_points: Optional[np.ndarray] = None,
        doe_values: Optional[np.ndarray] = None,
        seed: int = 0,
        log_path: Optional[str] = None,
    ):
        self.objective_fn = objective_fn
        self.lower_bounds = np.asarray(lower_bounds, dtype=float)
        self.upper_bounds = np.asarray(upper_bounds, dtype=float)
        self.dim = self.lower_bounds.size

        if self.upper_bounds.size != self.dim:
            raise ValueError("lower_bounds and upper_bounds must have the same length")
        if np.any(self.lower_bounds > self.upper_bounds):
            raise ValueError("lower_bounds must be <= upper_bounds")
        if budget < n_init:
            raise ValueError("budget must be >= n_init")
        if not 2 <= min_local_points <= n_init:
            raise ValueError("min_local_points must satisfy 2 <= min_local_points <= n_init.")
        # The incrementation value of the relaxation factor    
        if relaxation_step <= 0.0:
            raise ValueError("relaxation_step must be > 0")

        self.budget = int(budget)
        self.n_init = int(n_init)
        self.global_batch_size = int(global_batch_size)
        self.min_local_points = int(min_local_points)
        self.relaxation_step = float(relaxation_step)
        self.candidate_pool_size = max(int(candidate_pool_size), 8 * self.dim)
        self.local_num_restarts = (
            default_num_restarts(self.dim) if local_num_restarts is None else int(local_num_restarts)
        )
        self.local_raw_samples = (
            default_raw_samples(self.dim) if local_raw_samples is None else int(local_raw_samples)
        )
        if self.local_num_restarts < 1:
            raise ValueError("local_num_restarts must be >= 1")
        if self.local_raw_samples < self.local_num_restarts:
            raise ValueError("local_raw_samples must be >= local_num_restarts")
        self.objective_pair = objective_pair
        self.ivr_integration_points = int(ivr_integration_points)
        self.seed = int(seed)

        self.X_obs: list[np.ndarray] = []
        self.y_obs: list[float] = []
        self.best_y = float("inf")

        self._run_start = time.perf_counter()
        self._log_handle = None
        self._log_writer = None
        if log_path is not None:
            self._open_log(log_path)
        self.log_path = log_path

        x_0, f_0 = self._initial_doe(doe_points, doe_values)

        super().__init__(
            x_0=x_0,
            f_0=f_0,
            sigma_0=default_sigma_0(self.dim) if sigma_0 is None else float(sigma_0),
            beta_1=beta_1,
            beta_2=beta_2,
            gamma=gamma,
            kappa=kappa,
            d_max=d_max,
            deterministic=deterministic,
        )



    def _open_log(self, log_path: str) -> None:
        """Create the run's trace CSV (one row per evaluation, appended by _observe) and write its header."""
        ensure_directory(os.path.dirname(log_path) or ".")
        self._log_handle = open(log_path, "w", newline="")
        self._log_writer = csv.DictWriter(
            self._log_handle,
            fieldnames=[
                "seed", "iteration", "phase", "value",
                "best_so_far", "sigma", "elapsed_seconds", "cum_seconds",
            ],
        )
        self._log_writer.writeheader()
        self._log_handle.flush()

    def close(self) -> None:
        """Called at the end of the run to close the logging file"""
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
            self._log_writer = None

    def _budget_reached(self) -> bool:
        return len(self.y_obs) >= self.budget

    def _observe(self, x: np.ndarray, phase: str, precomputed: Optional[float] = None) -> float:
        """Evaluate the objective at ``x``, store it, and log the row.
        """
        x = np.asarray(x, dtype=float).flatten()
        if precomputed is not None:
            f, elapsed = float(precomputed), 0.0
        else:
            started = time.time()
            f = float(self.objective_fn(x))
            elapsed = time.time() - started

        self.X_obs.append(x.copy())
        self.y_obs.append(f)
        self.best_y = min(self.best_y, f)

        if self._log_writer is not None:
            self._log_writer.writerow({
                "seed": self.seed,
                "iteration": len(self.y_obs),
                "phase": phase,
                "value": f,
                "best_so_far": self.best_y,
                "sigma": getattr(self, "sigma_k", 0.0),
                "elapsed_seconds": elapsed,
                "cum_seconds": time.perf_counter() - self._run_start,
            })
            self._log_handle.flush()

        return f

    def _initial_doe(
        self, doe_points: Optional[np.ndarray], doe_values: Optional[np.ndarray]
    ) -> tuple[np.ndarray, float]:
        if doe_points is not None:
            X_init = np.asarray(doe_points, dtype=float)[: self.n_init]
            if X_init.shape[0] != self.n_init:
                raise ValueError(f"DoE has {X_init.shape[0]} points, expected {self.n_init}")
            values = np.asarray(doe_values, dtype=float) if doe_values is not None else None
            for i, x in enumerate(X_init):
                self._observe(x, phase="doe", precomputed=None if values is None else values[i])
        else:
            sampler = qmc.LatinHypercube(d=self.dim, seed=self.seed)
            X_init = qmc.scale(sampler.random(n=self.n_init), self.lower_bounds, self.upper_bounds)
            for x in X_init:
                self._observe(x, phase="doe")

        best_idx = int(np.argmin(self.y_obs))
        return self.X_obs[best_idx].copy(), float(self.y_obs[best_idx])


    def _build_trust_region(self) -> dict:
        """Trust-region box around the incumbent, in normalized and raw coords."""
        center_n = np.clip(normalize(self.x_k, self.lower_bounds, self.upper_bounds), 0.0, 1.0)
        radius_n = float(self.d_max * self.sigma_k)

        lower_n = np.clip(center_n - radius_n, 0.0, 1.0)
        upper_n = np.clip(center_n + radius_n, 0.0, 1.0)

        return {
            "center_n": center_n,
            "radius_n": radius_n,
            "lower_n": lower_n,
            "upper_n": upper_n,
            "lower": denormalize(lower_n, self.lower_bounds, self.upper_bounds),
            "upper": denormalize(upper_n, self.lower_bounds, self.upper_bounds),
        }

    def step(self) -> dict:
        """One STREGO run"""
      
        
        self.sigma_k = min(self.sigma_k, 1.0)

        if self._budget_reached():
            return self._record("budget_stop")

        x_glb, f_glb = self.global_phase()

        if self.sufficient_decrease(f_glb):
            self.x_k, self.f_k = np.asarray(x_glb, dtype=float).copy(), float(f_glb)
            self.sigma_k = self.gamma * self.sigma_k
            return self._record("global_success")

        if self._budget_reached():
            return self._record("budget_stop")

        x_lcl, f_lcl = self.local_phase(self._build_trust_region())

        if self.sufficient_decrease(f_lcl):
            self.x_k, self.f_k = np.asarray(x_lcl, dtype=float).copy(), float(f_lcl)
            self.sigma_k = self.gamma * self.sigma_k
            return self._record("local_success")

     
        if self.deterministic or self.certain_failure(f_lcl):
            self.sigma_k = self.beta_2 * self.sigma_k
            return self._record("unsuccessful" if self.deterministic else "certain_unsuccessful")
        self.sigma_k = self.beta_1 * self.sigma_k
        return self._record("uncertain_unsuccessful")


    def global_phase(self) -> tuple[np.ndarray, float]:
        """Propose and evaluate a batch from the bi-objective Pareto front."""
        n_batch = min(self.global_batch_size, self.budget - len(self.y_obs))
        #fallout if we don't have enough points for the global phase
        if len(self.X_obs) < 2:
            selected_x = np.random.uniform(
                self.lower_bounds, self.upper_bounds, size=(n_batch, self.dim)
            )
        else:
            #Fit the global GP on normalized data
            Xn = np.clip(normalize(np.array(self.X_obs), self.lower_bounds, self.upper_bounds), 0.0, 1.0)
            model = fit_gp(Xn, np.array(self.y_obs), self.dim)

            #Prepare initial population for NSMA
            sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.seed + len(self.X_obs))
            Xn_pool = sobol.draw(self.candidate_pool_size).cpu().numpy()

            pareto_x, pareto_f = solve_biobjective(
                model,
                Xn_pool,
                objective_pair=self.objective_pair,
                ivr_integration_points=self.ivr_integration_points,
                seed=self.seed,
            )
            selected_n = select_batch(pareto_x, pareto_f, n_batch, Xn_pool, seed=self.seed)
            selected_x = denormalize(
                np.asarray(selected_n, dtype=float).reshape(-1, self.dim),
                self.lower_bounds,
                self.upper_bounds,
            )

        batch_x, batch_f = [], []
        for x_i in selected_x:
            x_i = np.clip(np.asarray(x_i, dtype=float).flatten(), self.lower_bounds, self.upper_bounds)
            batch_f.append(self._observe(x_i, phase="global"))
            batch_x.append(x_i)

        best = int(np.argmin(batch_f))
        return batch_x[best], float(batch_f[best])

    def _gather_local_data(self, trust_region: dict) -> tuple[np.ndarray, np.ndarray]:
        """Local design for the local GP, via the relaxation rule.
        """
        #Make sure we have identical points to ensure correct GP behavior. This is a fallout in case one of the phases produced very close points.
        X_all, y_all = unique_rows_tol(
            np.array(self.X_obs, dtype=float), np.array(self.y_obs, dtype=float)
        )
        Xn_all = np.clip(normalize(X_all, self.lower_bounds, self.upper_bounds), 0.0, 1.0)
         
        center_n = trust_region["center_n"]
        radius_tr = trust_region["radius_n"]

        j = 0
        while True:
            radius = (1.0 + j * self.relaxation_step) * radius_tr
            lower_n = np.clip(center_n - radius, 0.0, 1.0)
            upper_n = np.clip(center_n + radius, 0.0, 1.0)
            inside = distance_to_box(Xn_all, lower_n, upper_n) == 0
            if inside.sum() >= self.min_local_points or radius >= 1.0:
                return X_all[inside], y_all[inside]
            j += 1

    def local_phase(self, trust_region: dict) -> tuple[np.ndarray, float]:
        """Minimize the local GP's posterior mean inside the trust region."""
        X_local, y_local = self._gather_local_data(trust_region)
        # In case we don't have enough data or we reached our maximum budget, the local phase doesn't take place and we retrun the current incumbent.
        if X_local.shape[0] < 2 or self._budget_reached():
            return self.x_k.copy(), float(self.f_k)
        Xn_local = np.clip(normalize(X_local, self.lower_bounds, self.upper_bounds), 0.0, 1.0)
        model = fit_gp(Xn_local, y_local, self.dim)
        acq = NegativePosteriorMean(model=model, maximize=False)
        #Extract tbe original trust region's bounds
        z_lower = np.clip(trust_region["lower_n"], 0.0, 1.0)
        z_upper = np.clip(trust_region["upper_n"], 0.0, 1.0)
     
        z_upper = np.maximum(z_upper, z_lower)
        bounds_t = torch.stack([
            torch.tensor(z_lower, dtype=torch.float64),
            torch.tensor(z_upper, dtype=torch.float64),
        ])
         # Pre-screen: score a Sobol pool of local_raw_samples points inside the
        # trust region with the acquisition (-mu) and keep the best
        # local_num_restarts (default 2d + 4) as starting points for L-BFGS.
        # BoTorch's standard procedure would work too (drop
        # batch_initial_conditions and pass raw_samples to optimize_acqf): it
        # scores a similar pool but samples the starts at random, weighted toward
        # high scores, and then discards the pool. We keep our ranked pool so that,
        # if the optimizer returns an already-evaluated point, we can fall back to
        # the best unseen point of the pool (see below).

        pool_unit = SobolEngine(
            dimension=self.dim, scramble=True, seed=self.seed + len(self.X_obs) + 1
        ).draw(self.local_raw_samples).to(torch.float64)
        pool_t = bounds_t[0] + pool_unit * (bounds_t[1] - bounds_t[0])

        with torch.no_grad():
            acq_values = acq(pool_t.unsqueeze(1)).detach().cpu().numpy().reshape(-1)

        ranked = np.argsort(-acq_values)
        warm_starts = pool_t[ranked[: self.local_num_restarts]].unsqueeze(1)

        candidate_n, _ = optimize_acqf(
            acq_function=acq,
            bounds=bounds_t,
            q=1,
            num_restarts=len(warm_starts),
            batch_initial_conditions=warm_starts,
        )

        lower, upper = trust_region["lower"], trust_region["upper"]
        x_new = np.clip(
            denormalize(candidate_n.detach().cpu().numpy().reshape(-1), self.lower_bounds, self.upper_bounds),
            lower,
            upper,
        )

   
        if is_close_to_any(x_new, self.X_obs, tol=1e-6):
            pool_sorted = pool_t[ranked].cpu().numpy()
            x_new = next(
                (
                    cand
                    for cand in (
                        np.clip(denormalize(c, self.lower_bounds, self.upper_bounds), lower, upper)
                        for c in pool_sorted
                    )
                    if not is_close_to_any(cand, self.X_obs, tol=1e-6)
                ),
                np.random.uniform(lower, upper),
            )

        f_new = self._observe(x_new, phase="local-mean")

        all_x = np.vstack([X_local, x_new[None, :]])
        all_f = np.concatenate([y_local, [f_new]])
        best = int(np.argmin(all_f))
        return all_x[best], float(all_f[best])


    def run(self) -> OptimizationResult:
        """Run until the evaluation budget is exhausted."""
        #Fix the seed of every random number generator to ensure reproducibility.
        set_all_seeds(self.seed)
        results = self.optimize(
            max_iterations=max(1, self.budget - self.n_init),
            stopping_criterion=lambda opt: opt._budget_reached(),
        )
        self.close()

        best_idx = int(np.argmin(self.y_obs))
        return OptimizationResult(
            best_x=self.X_obs[best_idx].copy(),
            best_y=float(self.y_obs[best_idx]),
            n_evaluations=len(self.y_obs),
            n_iterations=results["n_iterations"],
            history=results["history"],
            log_path=self.log_path,
        )
