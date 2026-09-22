"""The mean-only local phase acquisition function..
"""

from __future__ import annotations

import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform


class NegativePosteriorMean(AnalyticAcquisitionFunction):
    """Negated GP posterior mean at a single point.

    ``optimize_acqf`` maximizes the acquisition, so maximizing ``-mu(x)``
    minimizes the posterior mean -- which is what STREGO's local phase does.
    """

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return -self.model.posterior(X).mean.view(X.shape[:-2])
