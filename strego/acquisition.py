"""The local-phase acquisition function.

STREGO's local phase is pure exploitation: all of the exploration is handled by
the global phase's bi-objective front, so once we are inside the trust region we
simply descend the posterior mean. That is what makes the two phases
complementary rather than redundant -- an EI-style local phase would re-explore
inside a region the global phase already chose for its information content.
"""

from __future__ import annotations

import torch
from botorch.acquisition import AcquisitionFunction


class qNegativePosteriorMean(AcquisitionFunction):
    """Negated posterior mean, averaged over a q-batch.

    BoTorch acquisitions take ``batch_shape x q x d`` and return one scalar per
    batch element, and ``optimize_acqf`` *maximizes* what they return. We are
    minimizing the objective, so we return ``-mean(mu)`` over the q candidates.
    """

    def __init__(self, model, maximize: bool = False):
        super().__init__(model=model)
        self.maximize = maximize

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        mean = self.model.posterior(X).mean.squeeze(-1)  # batch_shape x q
        value = mean.mean(dim=-1)
        return value if self.maximize else -value
