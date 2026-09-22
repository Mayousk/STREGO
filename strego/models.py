"""The GP surrogate used by both STREGO phases.

STREGO fits exactly one kind of model, in one place: a ``SingleTaskGP`` with an
ARD RBF kernel whose lengthscale prior is scaled with the dimension, and a
``Standardize`` outcome transform. The global phase fits it on every observation
so far; the local phase fits the same model class on the local design gathered
around the trust-region centre.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior


def make_dim_scaled_covar(dim: int) -> RBFKernel:
    """ARD RBF kernel with the dimensionality-scaled lengthscale prior.

    From Hvarfner, Hellsten & Nardi, "Vanilla Bayesian Optimization Performs
    Great in High Dimensions" (ICML 2024):

        ell_i ~ LogNormal(sqrt(2) + log(D)/2, sqrt(3))

    so the median lengthscale grows like sqrt(D). That growth cancels the sqrt(D)
    growth of pairwise distances in the unit cube, which is what stops the kernel
    from collapsing onto its prior in high dimensions.
    There is deliberately no ``ScaleKernel``: the signal variance stays pinned at
    1,  since ``Standardize(m=1)`` already puts y on unit scale.
    """
    loc = math.sqrt(2.0) + 0.5 * math.log(dim)
    kernel = RBFKernel(ard_num_dims=dim, lengthscale_prior=LogNormalPrior(loc=loc, scale=math.sqrt(3.0)))
    kernel.lengthscale = math.exp(loc)  
    return kernel


def fit_gp(Xn: np.ndarray, y: np.ndarray, dim: int) -> SingleTaskGP:
    """Fit the STREGO surrogate on unit-cube inputs ``Xn`` and raw outputs ``y``.
    """
    train_X = torch.as_tensor(Xn, dtype=torch.float64)
    train_Y = torch.as_tensor(np.asarray(y, dtype=float), dtype=torch.float64).unsqueeze(-1)

    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        outcome_transform=Standardize(m=1),
        covar_module=make_dim_scaled_covar(dim),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
   
    model.eval()
    return model
