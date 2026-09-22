"""The bi-objective problem solved in STREGO's global phase.

This is the heart of the method. Instead of scalarising exploitation and
exploration into a single acquisition, the global phase poses them as two
objectives and hands the trade-off to a multi-objective solver:

    minimize  [ mu(x),  -IVR(x) ]        (``MU_IVR``, the default)
    minimize  [ mu(x),  -var(x) ]        (``MU_SIGMA``, the classic pairing)

``mu`` is the GP posterior mean (exploit) and the second objective is the
exploration term (negated, so that both are minimized).

Why IVR rather than the posterior variance
------------------------------------------
``-var(x)`` rewards a candidate for being uncertain *at itself*. In high
dimensions that is nearly every unobserved point at once, so the Pareto front
degenerates: the exploration axis saturates and the front collapses onto the
exploitation axis.

The integrated variance reduction asks a better question -- how much would
sampling ``c`` reduce the posterior variance *everywhere else*? For a GP that
has a closed form. Conditioning on a new observation at ``c`` reduces the
variance at any ``z`` by

    Delta(z; c) = cov(z, c | D)^2 / var(c | D)

so, averaged over an integration grid Z drawn on the domain,

    IVR(c) = (1/N) * sum_z cov(z, c | D)^2 / var(c | D).

Both terms come straight out of the joint posterior covariance over ``[Z; c]``,
which makes the whole objective a single posterior call. ``observation_noise=True``
makes the denominator the predictive variance of the look-ahead observation,
which also keeps it strictly positive.

Attribution
-----------
The ``[mu, -var]`` bi-objective acquisition solved with NSMA, and the problem
interface used here, come from

    F. Carciaghi, S. Magistri, P. Mansueto, F. Schoen. "A Bi-Objective
    Optimization Based Acquisition Strategy for Batch Bayesian Global
    Optimization." Computational Optimization and Applications, 2025.

This file is adapted from their implementation,
https://github.com/FranciC19/biobj_acquistion_function_for_BO, distributed
under the Apache License 2.0. Modifications: their problem wrapper classes are
merged into one self-contained class, the objective pairs are reduced to the two
STREGO uses, and the integrated-variance-reduction objective (``MU_IVR``) is
added.
"""

from __future__ import annotations

import numpy as np
import torch
from nsma.problems.problem import Problem

TORCH_TYPE = torch.float64
DEVICE = "cpu"

OBJECTIVE_PAIRS = ("MU_IVR", "MU_SIGMA")

# Jitter on the IVR denominator. var(c | D) is strictly positive once observation
# noise is included, but it can still underflow for a candidate sitting on top of
# an existing observation.
_JITTER = 1e-12


class BiObjectiveProblem(Problem):
    """NSMA-facing view of the GP's [exploit, explore] trade-off.

    The solver drives this through ``evaluate_functions`` (values) and
    ``evaluate_functions_jacobian`` (gradients, used by NSMA's memetic descent
    steps), both in the GP's normalized [0, 1]^d coordinates.
    """

    def __init__(
        self,
        dim: int,
        model,
        objective_pair: str = "MU_IVR",
        ivr_integration_points: int = 64,
        seed: int = 0,
    ):
        super().__init__(dim)

        if objective_pair not in OBJECTIVE_PAIRS:
            raise ValueError(f"objective_pair must be one of {OBJECTIVE_PAIRS}, got {objective_pair!r}")

        self.model = model
        self.objective_pair = objective_pair
        self._posterior = model.posterior

        # NSMA searches the normalized cube, matching the GP's input space.
        self.lb = np.zeros((dim,), dtype=float)
        self.ub = np.ones((dim,), dtype=float)

        # Fixed Sobol integration grid for IVR. Fixed (not resampled per call) so
        # that the two objectives stay consistent across one NSMA search -- a
        # moving grid makes the front jitter and defeats the memetic descent.
        # 64 points is the campaign default: it keeps NSMA tractable at d = 100,
        # where every jacobian call costs a joint posterior over N + 1 points.
        self._ivr_Z = None
        if objective_pair == "MU_IVR":
            engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)
            self._ivr_Z = engine.draw(ivr_integration_points).to(device=DEVICE, dtype=TORCH_TYPE)

        self._objective_fns = (
            [self.mu_x, self.ivr_x] if objective_pair == "MU_IVR" else [self.mu_x, self.sigma_x]
        )

    # -- objectives ---------------------------------------------------------
    # Each takes a (1, d) tensor and returns a 1-element tensor, kept connected
    # to the input so autograd can supply the jacobian.

    def mu_x(self, x0: torch.Tensor) -> torch.Tensor:
        """Posterior mean -- the exploitation objective (minimized)."""
        return self._posterior(x0).mvn.mean

    def sigma_x(self, x0: torch.Tensor) -> torch.Tensor:
        """Negated posterior variance. No square root: monotone, so the Pareto
        front is identical and we skip a needless nonlinearity."""
        return -self._posterior(x0).mvn.variance

    def ivr_x(self, x0: torch.Tensor) -> torch.Tensor:
        """Negated integrated variance reduction (see the module docstring)."""
        Z = self._ivr_Z
        n = Z.shape[0]
        joint = torch.cat([Z, x0.reshape(1, -1)], dim=0)
        cov = self._posterior(joint, observation_noise=True).mvn.covariance_matrix
        cov_zc = cov[:n, n]  # cov(z, c | D)
        var_c = cov[n, n]  # var(c | D) + noise
        ivr = (cov_zc ** 2).sum() / (var_c + _JITTER) / n
        return -ivr.reshape(1)

    # -- NSMA Problem contract ---------------------------------------------

    @property
    def objectives(self) -> list:
        return self._objective_fns

    @property
    def m(self) -> int:
        return len(self._objective_fns)

    def evaluate_functions(self, x: np.ndarray) -> np.ndarray:
        x_t = torch.tensor(x.reshape((1, self.n)), device=DEVICE, dtype=TORCH_TYPE)
        with torch.no_grad():
            return np.array([float(f(x_t).reshape(-1)[0]) for f in self._objective_fns], dtype=float)

    def evaluate_functions_tensor(self, x: torch.Tensor) -> tuple:
        return tuple(f(x) for f in self._objective_fns)

    def evaluate_functions_jacobian(self, x: np.ndarray) -> np.ndarray:
        x_t = torch.tensor(
            x.reshape((1, self.n)), device=DEVICE, dtype=TORCH_TYPE, requires_grad=True
        )
        rows = []
        for i, f in enumerate(self._objective_fns):
            value = f(x_t).reshape(-1)[0]
            # retain_graph for every objective but the last: they share the one
            # posterior graph built from x_t.
            (grad,) = torch.autograd.grad(
                value, x_t, retain_graph=(i < len(self._objective_fns) - 1)
            )
            rows.append(grad.detach().cpu().numpy().reshape(-1))
        return np.vstack(rows)

    def evaluate_constraints(self, x: np.ndarray) -> list:
        return []  # box bounds only, handled by lb/ub

    @staticmethod
    def name() -> str:
        return "BiObjectiveProblem"

    @staticmethod
    def family_name() -> str:
        return "BiObjectiveProblem"
