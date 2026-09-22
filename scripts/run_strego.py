#!/usr/bin/env python
"""Run STREGO once on a benchmark function.

    python scripts/run_strego.py --function rastrigin_fullshift --dim 50 --budget 400

Writes a per-evaluation CSV trace (one row per objective call) to --log-path.
For multi-trial, multi-function sweeps use scripts/run_campaign.py instead.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strego import STREGO, get_function_spec, wrap_noisy
from strego.benchmarks import FUNCTIONS
from strego.optimizer import DEFAULTS
from strego.utils import set_all_seeds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--function", default="rastrigin_fullshift", choices=FUNCTIONS)
    p.add_argument("--dim", type=int, default=50)
    p.add_argument("--budget", type=int, default=400, help="Total evaluations, including the initial design")
    p.add_argument("--n-init", type=int, default=DEFAULTS["n_init"], help="Initial Latin-hypercube design size")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--noise-type", default="none", choices=["none", "additive", "multiplicative"])
    p.add_argument("--noise-std", type=float, default=0.0)

    p.add_argument("--objective-pair", default=DEFAULTS["objective_pair"], choices=["MU_IVR", "MU_SIGMA"],
                   help="Global-phase bi-objective: [mu, -IVR] (default) or [mu, -variance]")
    p.add_argument("--ivr-integration-points", type=int, default=DEFAULTS["ivr_integration_points"])
    p.add_argument("--global-batch-size", type=int, default=DEFAULTS["global_batch_size"])
    p.add_argument("--min-local-points", type=int, default=DEFAULTS["min_local_points"],
                   help="n_min: the trust region is relaxed until the local design holds this many points")
    p.add_argument("--relaxation-step", type=float, default=DEFAULTS["relaxation_step"],
                   help="Relaxation factor increment: radius goes 1x, 1.5x, 2.0x, ... with 0.5")
    p.add_argument("--candidate-pool-size", type=int, default=DEFAULTS["candidate_pool_size"])
    p.add_argument("--local-num-restarts", type=int, default=DEFAULTS["local_num_restarts"],
                   help="Default: 2d + 4")
    p.add_argument("--local-raw-samples", type=int, default=DEFAULTS["local_raw_samples"],
                   help="Default: (2d + 4)^2")

    p.add_argument("--beta-1", type=float, default=DEFAULTS["beta_1"])
    p.add_argument("--beta-2", type=float, default=DEFAULTS["beta_2"])
    p.add_argument("--gamma", type=float, default=DEFAULTS["gamma"])
    p.add_argument("--kappa", type=float, default=DEFAULTS["kappa"])
    p.add_argument("--sigma-0", type=float, default=None, help="Default: 0.5 * (1/5)^(1/d)")
    p.add_argument("--deterministic", action="store_true",
                   help="Noise-free objective: use a single contraction factor")

    p.add_argument("--log-path", default="logs/strego_run.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Seed before building the objective: the shifted benchmarks are themselves
    # deterministic, but the noise wrapper draws from the global numpy RNG.
    set_all_seeds(args.seed)

    base_fn, lower, upper = get_function_spec(args.function, args.dim)
    objective = wrap_noisy(base_fn, args.noise_type, args.noise_std)

    optimizer = STREGO(
        objective_fn=objective,
        lower_bounds=lower,
        upper_bounds=upper,
        budget=args.budget,
        n_init=args.n_init,
        global_batch_size=args.global_batch_size,
        relaxation_step=args.relaxation_step,
        min_local_points=args.min_local_points,
        candidate_pool_size=args.candidate_pool_size,
        local_num_restarts=args.local_num_restarts,
        local_raw_samples=args.local_raw_samples,
        objective_pair=args.objective_pair,
        ivr_integration_points=args.ivr_integration_points,
        sigma_0=args.sigma_0,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        gamma=args.gamma,
        kappa=args.kappa,
        deterministic=args.deterministic,
        seed=args.seed,
        log_path=args.log_path,
    )

    print(f"STREGO  {args.function} d={args.dim}  budget={args.budget}  seed={args.seed}")
    print(f"global=[{args.objective_pair}] batch={args.global_batch_size}  local=mean n_min={args.min_local_points}")

    result = optimizer.run()

    print(f"\nbest_y        : {result.best_y:.6f}")
    print(f"evaluations   : {result.n_evaluations}")
    print(f"iterations    : {result.n_iterations}")
    print(f"trace         : {result.log_path}")


if __name__ == "__main__":
    main()
