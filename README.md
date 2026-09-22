# STREGO

**S**urrogate **TR**ust-region **E**fficient **G**lobal **O**ptimization — trust-region
Bayesian optimization whose global phase poses exploitation and exploration as a
*bi-objective problem* rather than scalarising them into a single acquisition.

STREGO targets expensive, possibly noisy, black-box objectives in moderate to
high dimension (tested up to d = 100).

---

## The method

Each iteration runs up to two phases against the same kind of GP surrogate.

### Global phase — where exploration happens

1. Fit a GP on every observation so far, over the whole domain.
2. Draw a Sobol candidate pool.
3. Solve a **bi-objective problem** on that pool with [NSMA](https://github.com/pierlumanzu/nsma):

   ```
   minimize  [ mu(x),  -IVR(x) ]
   ```

   `mu` is the posterior mean (exploit); `IVR` is the *integrated variance
   reduction* (explore). This builds on the `[mu, -var]` bi-objective
   acquisition of Carciaghi et al. (2025), with the variance replaced by IVR.
4. The result is a Pareto front of defensible exploit/explore compromises.
   k-means it in design space and evaluate a small batch (3 by default).

The global phase always sees the **full domain** — it is never restricted to the
trust region.

### Why IVR instead of the posterior variance

The usual exploration term, `-var(x)`, rewards a candidate for being uncertain
*at itself*. In high dimensions that describes nearly every unobserved point, so
the exploration axis saturates and the Pareto front collapses onto the
exploitation axis — the bi-objective formulation stops buying anything.

IVR asks a better question: how much would sampling `c` reduce the posterior
variance **everywhere else**? For a GP this is closed-form. Conditioning on a new
observation at `c` reduces the variance at any `z` by
`cov(z, c | D)^2 / var(c | D)`, so averaged over an integration grid `Z`:

```
IVR(c) = (1/N) * sum_z  cov(z, c | D)^2 / var(c | D)
```

Both terms come out of a single joint posterior over `[Z; c]`, so the objective
costs one posterior call. Set `--objective-pair MU_SIGMA` to use the classic
`[mu, -var]` pairing instead.

### Local phase — pure exploitation

Runs **only** when the global batch failed to improve sufficiently. It fits the
same GP class on a *local design* around the incumbent and minimizes the
posterior mean inside the trust region. No exploration term: the global phase
already covered that, and an EI-style local phase would re-explore a region that
was chosen for its information content.

The local design is built with a **relaxation rule**. Start from the
observations inside the trust region; while there are fewer than `n_min` of
them, relax the region — at step `j` its radius becomes
`(1 + j * relaxation_step)` times the trust-region radius (1× — the trust region itself — then 1.5×, 2.0×, …
with the default step of 0.5) — until the local design holds at least `n_min`
points.

Relaxation only decides what the local GP is **trained on**. The next point is
still searched inside the unrelaxed trust region, so `sigma_k` keeps full
control of the local phase's extent: a sparse region borrows data from its
neighbourhood, not search territory. Because `n_min <= n_init`, the rule always
terminates — at worst the box covers the whole domain and holds every
observation.

### Trust-region updates under noise

Standard TREGO expands on sufficient decrease and contracts otherwise. Under
noise a failure is ambiguous — genuinely worse, or a good point that evaluated
badly — and collapsing both into one factor makes the region shrink too fast.
STREGO splits failure in two:

| Case | Condition | Action |
|---|---|---|
| Success | `f <= f_k - kappa * sigma_k^2` | expand by `gamma` |
| Certain failure | `f >= f_k + kappa * sigma_k^2` | contract by `beta_2` (hard) |
| Uncertain failure | otherwise | contract by `beta_1` (gentle) |

with `beta_2 <= beta_1`. Pass `--deterministic` for noise-free objectives to use
the single factor `beta_2`.

Because the trust region gates only the *local* phase, a collapsed region never
strands the search: the global phase can still propose anywhere, and a global
success re-centres and re-expands the region.

---

## Installation

```bash
git clone https://github.com/<your-username>/STREGO.git
cd STREGO
pip install -e .
```

Or pin the exact versions the paper's campaigns used:

```bash
pip install -r requirements.txt
```

**Note on dependencies.** The global phase's multi-objective solver is `nsma`,
which depends on TensorFlow and uses it in graph mode. TensorFlow is imported
lazily — only when a global phase actually runs — so `import strego` stays fast
and side-effect-free. Everything else is the standard BoTorch/GPyTorch stack.

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
`best_so_far`, `sigma`, and wall-clock — flushed as it goes, so a killed run
still leaves a readable trace.

Runnable version: [`examples/quickstart.py`](examples/quickstart.py).

---

## Command line

Single run on a built-in benchmark:

```bash
python scripts/run_strego.py --function rastrigin_fullshift --dim 50 --budget 400
```

Multi-trial campaign (functions x dimensions x repeats, parallelised):

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
trace already holds the full budget are skipped, and the rest are redone. A
resume with *different* settings is refused, since the summary would otherwise
mix two configurations; use a fresh `--output-dir` for a new configuration.
Failed runs are reported at the end (exit code 1) and retried on the next rerun.

### Reproducing an ablation

Every run reads its initial design from the shared DoE catalog, so trial `t` of
any configuration starts from byte-identical initial data. Point a second
campaign at the first one's `doe/` and the flag under study becomes the only
thing that differs. Catalogs are matched on function, dimension *and*
`--noise-type`, so a deterministic design is never reused for a noisy campaign:

```bash
python scripts/run_campaign.py \
    --output-dir results/batch_b1 --global-batch-size 1 \
    --prebuilt-doe-dir results/baseline/doe \
    --functions rastrigin_fullshift,alpine01_fullshift,ackley_fullshift,schwefel_fullshift \
    --dims 100,50,20,4 --budget 400 --num-trials 20 --max-workers 4
```

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

All defaults are defined once, in the `DEFAULTS` dictionary at the top of
[`strego/optimizer.py`](strego/optimizer.py); both command-line scripts read
from it.

### The surrogate

Both phases fit a `SingleTaskGP` with an ARD RBF kernel whose lengthscale prior
is scaled with dimension, following Hvarfner, Hellsten & Nardi, *Vanilla Bayesian
Optimization Performs Great in High Dimensions* (ICML 2024):

```
ell_i ~ LogNormal(sqrt(2) + log(D)/2, sqrt(3))
```

The median lengthscale grows like `sqrt(D)`, cancelling the `sqrt(D)` growth of
pairwise distances in the unit cube. Without it, at D = 100 the kernel collapses
onto its prior, the posterior mean flattens, and the Pareto front the global
phase depends on goes with it.

---

## Benchmarks

Four multimodal functions — `rastrigin`, `alpine01`, `ackley`, `schwefel` — each
with a `_fullshift` variant.

The shifts matter. Rastrigin, alpine01 and ackley all put their optimum at the
origin, the exact centre of their boxes. Any method with a centre bias gets an
unearned advantage there — and IVR has one, since its integration grid is uniform
over the box, so interior candidates outscore edge candidates on geometry alone.
The `_fullshift` variants relocate the optimum to a random interior point
(deterministic given the seed), so the comparison measures search rather than
luck. Schwefel needs separate handling because it is unbounded below; see the
derivation in [`strego/benchmarks.py`](strego/benchmarks.py).

---

## Repository layout

```
strego/
├── optimizer.py      STREGO: the two phases and the iteration
├── trust_region.py   trust-region framework: acceptance tests, radius updates
├── biobjective.py    the [mu, -IVR] / [mu, -var] problem  <- the core idea
├── selection.py      NSMA search + Pareto-front -> batch selection
├── models.py         the GP surrogate and dimension-scaled kernel
├── acquisition.py    local-phase posterior-mean acquisition
├── benchmarks.py     benchmark functions, shifts, noise wrappers
└── utils.py          numerical helpers
scripts/
├── run_strego.py     single run
└── run_campaign.py   multi-trial campaign with shared designs
```

Tests: `pytest -q`.

---

## Citation

```bibtex
@article{strego,
  title   = {STREGO: Trust-Region Bayesian Optimization with a Bi-Objective Global Phase},
  author  = {Kadri, Meissem},
  year    = {2026}
}
```

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
