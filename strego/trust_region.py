"""The trust-region framework STREGO's two phases plug into.

This is a noise-aware variant of the TREGO scheme. The loop per iteration is:

1. propose globally; accept and expand if the proposal decreases sufficiently;
2. otherwise propose inside the trust region; accept and expand if *that*
   decreases sufficiently;
3. otherwise contract.

The one substantive change over TREGO is in step 3. Under noise, a failed
iteration is ambiguous: the candidate may genuinely be worse, or it may be a
good point that evaluated badly. Collapsing both cases into one contraction
factor makes the region shrink too fast on noisy problems and strand the search.
So failure is split in two:

    certain    f_lcl >= f_k + kappa * sigma_k^2   -- worse beyond what noise
                                                     explains; contract hard (beta_2)
    uncertain  otherwise                          -- could be noise; contract
                                                     gently (beta_1)

with ``beta_2 <= beta_1``. A deterministic problem has no ambiguity to model, so
``deterministic=True`` uses the single factor ``beta_2`` throughout.

Subclasses supply :meth:`global_phase` and :meth:`local_phase`; see
:class:`strego.optimizer.STREGO`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class OptimizationResult:
    best_x: np.ndarray
    best_y: float
    n_evaluations: int
    n_iterations: int
    history: dict = field(default_factory=dict)
    log_path: Optional[str] = None


class TrustRegionBO:
    """Trust-region bookkeeping: acceptance tests, radius updates, the outer loop.

    Parameters
    ----------
    sigma_0 : initial trust-region radius, as a fraction of the box width.
    beta_1, beta_2 : contraction factors for uncertain / certain failure,
        with ``0 < beta_2 <= beta_1 < 1``.
    gamma : expansion factor on success (``gamma > 1`` and ``gamma * beta_2 >= 1``,
        so that a contract-then-expand cycle never loses ground).
    kappa : sufficient-decrease threshold; the required improvement is
        ``kappa * sigma_k^2``, so the bar falls as the region shrinks.
    d_max : radius scaling at ``sigma_k = 1``.
    deterministic : collapse the two failure modes into one (see module docstring).
    """

    def __init__(
        self,
        x_0: np.ndarray,
        f_0: float,
        sigma_0: float,
        beta_1: float,
        beta_2: float,
        gamma: float,
        kappa: float,
        d_max: float = 1.0,
        deterministic: bool = False,
    ):
        if sigma_0 <= 0:
            raise ValueError("sigma_0 must be positive")
        if not 0 < beta_2 <= beta_1 < 1:
            raise ValueError("0 < beta_2 <= beta_1 < 1 required")
        if gamma <= 1:
            raise ValueError("gamma must be > 1")
        if gamma * beta_2 < 1:
            raise ValueError("gamma * beta_2 >= 1 required")
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        if d_max <= 0:
            raise ValueError("d_max must be positive")

        self.sigma_0 = float(sigma_0)
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.gamma = float(gamma)
        self.kappa = float(kappa)
        self.d_max = float(d_max)
        self.deterministic = bool(deterministic)

        # Iterate state: x_k / f_k are the incumbent the trust region centres on.
        self.k = 0
        self.x_k = np.asarray(x_0, dtype=float)
        self.f_k = float(f_0)
        self.sigma_k = float(sigma_0)

        self.history: dict[str, list] = {
            "x": [self.x_k.copy()],
            "f": [self.f_k],
            "sigma": [self.sigma_k],
            "iteration_type": [],
        }

    # -- acceptance tests ---------------------------------------------------

    def sufficient_decrease(self, f_candidate: float) -> bool:
        """``f_candidate <= f_k - kappa * sigma_k^2``."""
        return f_candidate <= self.f_k - self.kappa * self.sigma_k ** 2

    def certain_failure(self, f_candidate: float) -> bool:
        """``f_candidate >= f_k + kappa * sigma_k^2`` -- worse beyond the noise band."""
        return f_candidate >= self.f_k + self.kappa * self.sigma_k ** 2

    # -- phases (implemented by subclasses) ---------------------------------

    def global_phase(self) -> tuple[np.ndarray, float]:
        raise NotImplementedError

    def local_phase(self, trust_region: dict) -> tuple[np.ndarray, float]:
        raise NotImplementedError

    def step(self) -> dict[str, Any]:
        raise NotImplementedError

    # -- outer loop ---------------------------------------------------------

    def _record(self, iteration_type: str) -> dict[str, Any]:
        self.history["x"].append(self.x_k.copy())
        self.history["f"].append(self.f_k)
        self.history["sigma"].append(self.sigma_k)
        self.history["iteration_type"].append(iteration_type)
        self.k += 1
        return {
            "type": iteration_type,
            "x": self.x_k.copy(),
            "f": self.f_k,
            "sigma": self.sigma_k,
            "k": self.k - 1,
        }

    def optimize(
        self,
        max_iterations: int,
        callback: Optional[Callable[[dict], None]] = None,
        stopping_criterion: Optional[Callable[["TrustRegionBO"], bool]] = None,
    ) -> dict[str, Any]:
        for _ in range(max_iterations):
            if stopping_criterion is not None and stopping_criterion(self):
                break
            info = self.step()
            if callback is not None:
                callback(info)

        best_idx = int(np.argmin(self.history["f"]))
        return {
            "best_x": self.history["x"][best_idx],
            "best_f": self.history["f"][best_idx],
            "history": self.history,
            "n_iterations": self.k,
        }
