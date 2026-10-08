# STREGO
 
**S**calable **TR**ust-region **E**fficient **G**lobal **O**ptimization: trust-region Bayesian optimization for expensive, noisy black-box functions in high dimensions.
 
STREGO alternates a global and a local phase. The global phase selects points from the Pareto front of the bi-objective problem $\min\,[\mu(x), -\mathrm{IVR}(x)]$, where IVR (integrated variance reduction) replaces the posterior variance, which saturates in high dimensions. The local phase minimizes the posterior mean inside a trust region, and the trust region is updated with a noise-aware rule. Details are in the [paper](LINK).
## Installation

```bash
git clone https://github.com/Mayousk/STREGO.git
cd STREGO
pip install -e .
```

Or pin the exact versions the paper's campaigns used:

```bash
pip install -r requirements.txt
```

**Note on dependencies.** The global phase's multi-objective solver is `nsma`,
which depends on TensorFlow and uses it in graph mode. TensorFlow is imported only when a global phase actually runs. Everything else is the standard BoTorch/GPyTorch stack.

Tested on Python 3.9 with botorch 0.10.0 and torch 2.8.0.

---

## Quickstart

```python
import numpy as np
from strego import STREGO

def objective(x: np.ndarray) -> float:   # minimized; may be noisy
    return float(np.sum(x ** 2))

optimizer = STREGO(
    objective_fn=objective,
    lower_bounds=np.full(20, -5.0),
    upper_bounds=np.full(20, 5.0),
    budget=400,        # total evaluations, including the initial design
    n_init=10,         # initial Latin-hypercube design
    seed=42,
    log_path="logs/run.csv",
)

result = optimizer.run()
print(result.best_y, result.best_x)
```

`log_path` receives one row per objective evaluation — `phase`, `value`,
`best_so_far`, `sigma`, and wall-clock .

Runnable version: [`examples/toy_problem.py`](examples/toy_problem.py).

---

## Command line

Single run on a built-in benchmark:

```bash
python scripts/run_strego.py --function rastrigin_fullshift --dim 50 --budget 400
```

Multi-trial campaign (functions x dimensions x repeats, parallelized):

```bash
python scripts/run_campaign.py \
    --output-dir results/baseline \
    --functions rastrigin_fullshift,alpine01_fullshift,ackley_fullshift,schwefel_fullshift \
    --dims 100,50,20,4 --budget 400 --num-trials 20 \
    --max-workers 4
```

This writes:

```
results/baseline/
├── doe/          shared Latin-hypercube designs (points + values)
├── details/      one per-evaluation CSV per run
├── config.json   the settings the campaign was started with
└── summary.csv   one row per completed run
```

**Resuming.** If a campaign is interrupted, rerun the same command: runs whose
logs already holds the full budget are skipped, and the rest are redone.




Noisy campaigns:

```bash
# additive noise
python scripts/run_campaign.py --output-dir results/add  --noise-type additive       --noise-std 1.0  ...
# multiplicative (relative) noise
python scripts/run_campaign.py --output-dir results/mult --noise-type multiplicative --noise-std 0.10 ...
```

---

## Configuration

Defaults are the values used for the results in the paper.

| Parameter | Default | Meaning |
|---|---|---|
| `budget` | — | total evaluations, including the initial design |
| `n_init` | 10 | initial Latin-hypercube design size |
| `objective_pair` | `MU_IVR` | global bi-objective; `MU_SIGMA` for `[mu, -var]` |
| `ivr_integration_points` | 64 | Sobol grid size for IVR; keeps NSMA tractable at d = 100 |
| `global_batch_size` | 3 | points taken from the Pareto front per global phase |
| `min_local_points` | 10 | `n_min`: minimum local-design size; the TR is relaxed until reached |
| `relaxation_step` | 0.5 | relaxation factor increment (radius 1×, 1.5×, 2.0×, …) |
| `candidate_pool_size` | 250 | Sobol pool seeding NSMA; raised to `8*d` when larger |
| `local_num_restarts` | `2d+4` | warm starts for the local acquisition |
| `local_raw_samples` | `(2d+4)^2` | prescreen pool the warm starts are picked from |
| `sigma_0` | `0.5*(1/5)^(1/d)` | initial radius — constant box *fraction*, not side length |
| `beta_1` | 0.7 | gentle contraction (uncertain failure) |
| `beta_2` | 0.5 | hard contraction (certain failure) |
| `gamma` | 2.0 | expansion on success |
| `kappa` | 1.0 | sufficient-decrease threshold scale |



### The surrogate

Both phases fit a `SingleTaskGP` with an ARD RBF kernel whose lengthscale prior
is scaled with dimension, following Hvarfner, Hellsten & Nardi, *Vanilla Bayesian
Optimization Performs Great in High Dimensions* (ICML 2024):

```
ell_i ~ LogNormal(sqrt(2) + log(D)/2, sqrt(3))
```

---



## Repository layout

```
strego/
├── optimizer.py      STREGO: the two phases and the iteration
├── trust_region.py   trust-region framework: acceptance tests, radius updates
├── biobjective.py    the [mu, -IVR] / [mu, -var] problem  
├── global_selection.py     NSMA search + Pareto-front -> batch selection
├── models.py         the GP surrogate and dimension-scaled kernel
├── local_acquisition.py      local-phase posterior-mean acquisition
├── benchmarks.py     benchmark functions, shifts, noise wrappers
└── utils.py          numerical helpers
scripts/
├── run_strego.py     single run
└── run_campaign.py   multi-trial campaign with shared designs
```

Tests: `pytest -q`.

---

## Citation



## Acknowledgements

- [NSMA](https://github.com/pierlumanzu/nsma) — the memetic multi-objective
  solver used for the global phase.
- The `[mu, -var]` bi-objective acquisition solved with NSMA, from
  F. Carciaghi, S. Magistri, P. Mansueto & F. Schoen, *A Bi-Objective
  Optimization Based Acquisition Strategy for Batch Bayesian Global
  Optimization*, Computational Optimization and Applications (2025).
  [`strego/biobjective.py`](strego/biobjective.py) is adapted from
  [their implementation](https://github.com/FranciC19/biobj_acquistion_function_for_BO)
  (Apache License 2.0); the modifications are listed in that file's header.
- [BoTorch](https://botorch.org/) and [GPyTorch](https://gpytorch.ai/).
