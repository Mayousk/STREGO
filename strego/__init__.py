"""STREGO -- trust-region Bayesian optimization with a bi-objective global phase.

See :mod:`strego.optimizer` for the algorithm and README.md for usage.
"""

from .benchmarks import FUNCTIONS, get_function_spec, wrap_noisy
from .optimizer import DEFAULTS, STREGO, default_num_restarts, default_raw_samples, default_sigma_0
from .trust_region import OptimizationResult, TrustRegionBO

__version__ = "1.0.0"

__all__ = [
    "STREGO",
    "OptimizationResult",
    "TrustRegionBO",
    "FUNCTIONS",
    "DEFAULTS",
    "default_num_restarts",
    "default_raw_samples",
    "default_sigma_0",
    "get_function_spec",
    "wrap_noisy",
]
