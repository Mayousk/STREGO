"""STREGO: Surrogate TRust-region Efficient Global Optimization.

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
region until it holds at least ``n_min`` of them, fit the same GP class on that
local design, and minimize the posterior mean inside the (unrelaxed) trust
region. Pure exploitation, because the global phase already covered exploration.

The trust region therefore controls only the *local* phase's extent, while the
global phase always sees the full domain. That is what keeps STREGO from
stalling in a collapsed region: even at tiny ``sigma_k``, the global phase can
still propose anywhere, and a global success re-centres and re-expands the region.

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
>>> result.best_y < 1.0
True
"""

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

from .acquisition import qNegativePosteriorMean
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

# Every default, in one place. The command-line scripts read their defaults
# from here, so changing a value below changes it everywhere.
DEFAULTS: dict = {
    "n_init": 10,
    "global_batch_size": 3,
    "min_local_points": 10,
    "relaxation_step": 0.5,
    "candidate_pool_size": 250,
    # Local acquisition optimizer. None = scale with the dimension, see
    # default_num_restarts / default_raw_samples below.
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


def default_num_restarts(dim: int) -> int:
    """Multi-start restarts for the local acquisition: ``2d + 4``.

    Grows linearly with the dimension, since the number of local optima of the
    posterior mean inside the trust region grows with it.
    """
    return 2 * dim + 4


def default_raw_samples(dim: int) -> int:
    """Prescreen pool for the local acquisition: ``(2d + 4)^2``.

    The square of the restart count, so the warm starts are always the top
    ``1 / (2d + 4)`` of the pool -- a fixed selectivity at every dimension.
    """
    return (2 * dim + 4) ** 2


def default_sigma_0(dim: int) -> float:
    """Initial radius: ``0.5 * (1/5)^(1/d)``.

    The trust region covers a fixed *fraction* of the box volume regardless of
    dimension, rather than a fixed side length -- at d = 100 a fixed side length
    would be either the whole box or a speck.
    """
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
        Points drawn from the Pareto front per global phase. 3 by default: enough
        to span the trade-off, cheap enough to re-fit often.
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
        Default ``2d + 4`` and ``(2d + 4)^2``. Used exactly as given, with no
        floor or cap.
    candidate_pool_size : int
        Sobol pool seeding NSMA. Raised to ``8 * d`` when that is larger, so
        high-dimensional runs keep adequate coverage.
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
        # n_min <= n_init guarantees the relaxation terminates with a local design
        # of at least n_min points: at worst the box grows to the whole domain,
        # which contains the full initial design.
        if not 2 <= min_local_points <= n_init:
            raise ValueError("min_local_points must satisfy 2 <= min_local_points <= n_init")
        if relaxation_step <= 0.0:
            raise ValueError("relaxation_step must be > 0")

        self.budget = int(budget)
        self.n_init = int(n_init)
        self.global_batch_size = int(global_batch_size)
        self.min_local_points = int(min_local_points)
        self.relaxation_step = float(relaxation_step)
        self.candidate_pool_size = max(int(candidate_pool_size), 8 * self.dim)
        # Used exactly as given (or as the dimension-scaled default): no floor,
        # no cap. The one hard requirement is that the prescreen pool can
        # supply every warm start.
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

        # Observation store. Every evaluation lands here, in raw problem coords.
        self.X_obs: list[np.ndarray] = []
        self.y_obs: list[float] = []
        self.best_y = float("inf")

        self._run_start = time.perf_counter()
        self._log_handle = None
        self._log_writer = None
        if log_path is not None:
            self._open_log(log_path)
        self.log_path = log_path

        # Build the initial design *before* super().__init__, which needs the
        # starting incumbent (x_0, f_0) that the design produces.
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

    # -- bookkeeping --------------------------------------------------------

    def _open_log(self, log_path: str) -> None:
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
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
            self._log_writer = None

    def _budget_reached(self) -> bool:
        return len(self.y_obs) >= self.budget

    def _observe(self, x: np.ndarray, phase: str, precomputed: Optional[float] = None) -> float:
        """Evaluate the objective at ``x``, store it, and log the row.

        ``precomputed`` reuses a value from an injected design instead of
        spending an evaluation.
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
                # sigma_k does not exist yet while the initial design runs.
                "sigma": getattr(self, "sigma_k", 0.0),
                "elapsed_seconds": elapsed,
                "cum_seconds": time.perf_counter() - self._run_start,
            })
            # Flushed per row: campaigns are long, and a killed run should still
            # leave a readable trace.
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

    # -- trust region -------------------------------------------------------

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
        """One trust-region iteration: global, then local only if global failed."""
        # Beyond sigma_k = 1 the region already covers the box; expanding further
        # is a no-op that only delays the first contraction.
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

        # Failure: contract. How hard depends on whether the failure is
        # attributable to noise (see strego.trust_region).
        if self.deterministic or self.certain_failure(f_lcl):
            self.sigma_k = self.beta_2 * self.sigma_k
            return self._record("unsuccessful" if self.deterministic else "certain_unsuccessful")
        self.sigma_k = self.beta_1 * self.sigma_k
        return self._record("uncertain_unsuccessful")

    # -- phases -------------------------------------------------------------

    def global_phase(self) -> tuple[np.ndarray, float]:
        """Propose and evaluate a batch from the bi-objective Pareto front."""
        n_batch = min(self.global_batch_size, self.budget - len(self.y_obs))

        if len(self.X_obs) < 2:
            # Too little data to fit a surrogate worth trusting.
            selected_x = np.random.uniform(
                self.lower_bounds, self.upper_bounds, size=(n_batch, self.dim)
            )
        else:
            Xn = np.clip(normalize(np.array(self.X_obs), self.lower_bounds, self.upper_bounds), 0.0, 1.0)
            model = fit_gp(Xn, np.array(self.y_obs), self.dim)

            # Re-seeded per call so successive pools do not repeat.
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

        Start from the observations inside the trust region. While the local
        design holds fewer than ``n_min`` points, relax the region: at step
        ``j`` its radius is ``(1 + j * relaxation_step)`` times the trust-region
        radius, still centred on the incumbent and clipped to the domain.

        Relaxation only decides what the local GP is *trained on*. The next
        point is still searched inside the unrelaxed trust region, so
        ``sigma_k`` keeps full control of the local phase's extent -- a sparse
        region borrows data from its neighbourhood, not search territory.

        Termination is guaranteed: the growth is unbounded, and once the box
        covers the whole domain it holds every observation, of which there are
        at least ``n_init >= n_min``.
        """
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
            # radius >= 1 means the box already spans the whole unit cube: every
            # observation is in, and relaxing further cannot add any.
            if inside.sum() >= self.min_local_points or radius >= 1.0:
                return X_all[inside], y_all[inside]
            j += 1

    def local_phase(self, trust_region: dict) -> tuple[np.ndarray, float]:
        """Minimize the local GP's posterior mean inside the trust region."""
        X_local, y_local = self._gather_local_data(trust_region)

        if X_local.shape[0] < 2 or self._budget_reached():
            # Nothing to fit, or nothing left to spend: report the incumbent so
            # the caller records a (non-improving) iteration and contracts.
            return self.x_k.copy(), float(self.f_k)

        # The local GP is normalized to the FULL problem box, not the local one.
        # Only the *search* is restricted to the trust region; keeping the input
        # scaling global means the local and global models are directly
        # comparable and the kernel's lengthscale prior stays meaningful.
        Xn_local = np.clip(normalize(X_local, self.lower_bounds, self.upper_bounds), 0.0, 1.0)
        model = fit_gp(Xn_local, y_local, self.dim)
        acq = qNegativePosteriorMean(model=model, maximize=False)

        z_lower = np.clip(trust_region["lower_n"], 0.0, 1.0)
        z_upper = np.clip(trust_region["upper_n"], 0.0, 1.0)
        # Guard zero-width dimensions, which make optimize_acqf ill-posed.
        z_upper = np.maximum(z_upper, z_lower + 1e-9)
        bounds_t = torch.stack([
            torch.tensor(z_lower, dtype=torch.float64),
            torch.tensor(z_upper, dtype=torch.float64),
        ])

        # Prescreen: score a Sobol pool of local_raw_samples points in one batched
        # forward pass and use the best local_num_restarts of them as warm starts.
        # Far more reliable than letting optimize_acqf start from raw random
        # samples, because in high dimensions most random starts sit on a flat
        # part of the posterior mean.
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

        # A collapsed trust region will keep returning the incumbent. Re-evaluating
        # it wastes budget, so walk down the prescreened pool for the best point
        # we have not seen yet.
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

        # Report the best point in the local set *including* history, not just the
        # new one: the trust-region test asks whether the local region contains an
        # improvement, and an already-evaluated neighbour is a valid answer.
        all_x = np.vstack([X_local, x_new[None, :]])
        all_f = np.concatenate([y_local, [f_new]])
        best = int(np.argmin(all_f))
        return all_x[best], float(all_f[best])

    # -- driver -------------------------------------------------------------

    def run(self) -> OptimizationResult:
        """Run until the evaluation budget is exhausted."""
        set_all_seeds(self.seed)

        # Each iteration spends at least one evaluation, so the remaining budget
        # is a safe upper bound on the iteration count.
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
