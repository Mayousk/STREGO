#!/usr/bin/env python
"""STREGO campaign over multiple functions, dimensions and trials.

Two phases:

1. **Shared design.** One Latin-hypercube DoE catalog per (function, dimension),
   holding the points and their objective values for every trial. Every run
   reads its design from there instead of sampling its own.

2. **Runs.** One process per (function, dimension, trial), pooled across
   ``--max-workers``.

Outputs, under ``--output-dir``:

    doe/                 shared DoE catalogs
    details/             one per-evaluation CSV trace per run
    config.json          the settings the campaign was started with
    summary.csv          one row per completed run (final best_y)

**Resuming.** Rerunning the same command skips every run whose trace already
holds the full budget, and redoes the rest so a killed campaign picks up
where it stopped. Resuming with *different* STREGO or noise settings is refused
(compared against ``config.json``), because the summary would silently mix two
configurations. Use a fresh ``--output-dir`` for a new configuration.

Example:

    python scripts/run_campaign.py \\
        --output-dir results/batch_b1 --global-batch-size 1 \\
        --functions rastrigin_fullshift,alpine01_fullshift,ackley_fullshift,schwefel_fullshift \\
        --dims 100,50,20,4 --budget 400 --num-trials 20 \\
        --prebuilt-doe-dir results/baseline_b3/doe --max-workers 4
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strego import STREGO, get_function_spec, wrap_noisy
from strego.benchmarks import FUNCTIONS
from strego.optimizer import DEFAULTS
from strego.utils import ensure_directory, set_all_seeds

SUMMARY_FIELDS = ["solver", "function", "dimension", "trial", "seed", "best_y", "n_evaluations", "log_path"]




def doe_filename(func_name: str, dim: int, noise_type: str, num_trials: int, seed: int, n_init: int) -> str:
    return f"{func_name}_{noise_type}_{dim}_trials{num_trials}_seed{seed}_lhs_n{n_init}.csv"


def generate_doe_catalog(
    path: str, func_name: str, dim: int, n_init: int, num_trials: int, base_seed: int,
    noise_type: str, noise_std: float,
) -> None:
    """Write one DoE catalog covering every trial of a (function, dimension) case.

    Values are stored alongside the points so that runs reuse them rather than
    re-evaluating which is important under noise, where re-evaluating the same design
    would give each run a different starting picture.
    """
    ensure_directory(os.path.dirname(path))
    base_fn, lower, upper = get_function_spec(func_name, dim)

    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial", "seed", "y", *[f"x{i + 1}" for i in range(dim)]])
        for trial in range(num_trials):
            trial_seed = base_seed + trial
            set_all_seeds(trial_seed)
            sampler = qmc.LatinHypercube(d=dim, seed=trial_seed)
            X = qmc.scale(sampler.random(n=n_init), lower, upper)
            objective = wrap_noisy(base_fn, noise_type, noise_std)
            for row in X:
                writer.writerow([trial, trial_seed, float(objective(row)), *row.tolist()])


def find_prebuilt_doe(doe_dir: str, func_name: str, dim: int, noise_type: str) -> str:
    """Checks if a doe hasn't been generated yet. If so, extracts it.

    Matched on the full ``{function}_{noise}_{dim}_trials`` prefix. The full
    function name must agree`.
    """
    pattern = os.path.join(doe_dir, f"{func_name}_{noise_type}_{dim}_trials*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No DoE catalog for {func_name} d={dim} noise={noise_type} in {doe_dir}"
        )
    return matches[0]


def check_doe_catalog(path: str, dim: int, num_trials: int, n_init: int) -> None:
     """Check that a DoE catalog can supply every run of the campaign.

    Verifies three things, and raises ValueError on the first that fails:
      - the catalog has `dim` coordinate columns;
      - it contains every trial 0 .. num_trials-1;
      - each of those trials has at least `n_init` points..
    """
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        x_fields = [name for name in (reader.fieldnames or []) if name.startswith("x")]
        if len(x_fields) != dim:
            raise ValueError(f"{os.path.basename(path)} has dimension {len(x_fields)}, expected {dim}")
        points_per_trial = Counter(int(row["trial"]) for row in reader)

    missing = [t for t in range(num_trials) if t not in points_per_trial]
    if missing:
        raise ValueError(
            f"{os.path.basename(path)} holds {len(points_per_trial)} trial(s); "
            f"the campaign needs {num_trials} (missing trial {missing[0]})"
        )
    short = [t for t in range(num_trials) if points_per_trial[t] < n_init]
    if short:
        raise ValueError(
            f"{os.path.basename(path)}: trial {short[0]} has {points_per_trial[short[0]]} points, "
            f"--n-init is {n_init}"
        )


def load_doe_trial(path: str, trial: int, expected_dim: int) -> tuple[int, np.ndarray, np.ndarray | None]:
    

    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        x_fields = [name for name in fields if name.startswith("x")]
        has_y = "y" in fields

        if len(x_fields) != expected_dim:
            raise ValueError(f"{path} has dimension {len(x_fields)}, expected {expected_dim}")

        rows, values, trial_seed = [], [], None
        for row in reader:
            if int(row["trial"]) != trial:
                continue
            if trial_seed is None:
                trial_seed = int(row["seed"])
            rows.append([float(row[f]) for f in x_fields])
            if has_y:
                values.append(float(row["y"]))

    if trial_seed is None:
        raise ValueError(f"{path} has no trial {trial}")

    return trial_seed, np.asarray(rows, dtype=float), (np.asarray(values, dtype=float) if has_y else None)




def detail_log_path(detail_dir: str, func_name: str, dim: int, trial: int) -> str:
   """Return the path of a run's trace CSV, used both to write it and to find it again on resume."""
    return os.path.join(detail_dir, f"strego_{func_name}_d{dim}_trial{trial}.csv")


def completed_run(log_path: str, budget: int) -> dict | None:
    """Check whether a run already finished, i.e. its trace holds all `budget` evaluations."""
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != budget:
        return None
    return {"seed": int(rows[-1]["seed"]), "best_y": float(rows[-1]["best_so_far"])}


def run_one(task: dict) -> dict:
    """Execute a single STREGO run. Runs in a worker process."""
    cfg = task["config"]
    func_name, dim, trial = task["func_name"], task["dim"], task["trial"]

    trial_seed, doe_points, doe_values = load_doe_trial(task["doe_path"], trial, dim)
    set_all_seeds(trial_seed)

    base_fn, lower, upper = get_function_spec(func_name, dim)
    objective = wrap_noisy(base_fn, cfg["noise_type"], cfg["noise_std"])

    optimizer = STREGO(
        objective_fn=objective,
        lower_bounds=lower,
        upper_bounds=upper,
        budget=cfg["budget"],
        n_init=cfg["n_init"],
        global_batch_size=cfg["global_batch_size"],
        min_local_points=cfg["min_local_points"],
        relaxation_step=cfg["relaxation_step"],
        candidate_pool_size=cfg["candidate_pool_size"],
        local_num_restarts=cfg["local_num_restarts"],
        local_raw_samples=cfg["local_raw_samples"],
        objective_pair=cfg["objective_pair"],
        ivr_integration_points=cfg["ivr_integration_points"],
        beta_1=cfg["beta_1"],
        beta_2=cfg["beta_2"],
        gamma=cfg["gamma"],
        kappa=cfg["kappa"],
        deterministic=cfg["deterministic"],
        doe_points=doe_points,
        doe_values=doe_values,
        seed=trial_seed,
        log_path=task["log_path"],
    )
    result = optimizer.run()

    return summary_row(task, trial_seed, float(result.best_y), result.n_evaluations)


def summary_row(task: dict, seed: int, best_y: float, n_evaluations: int) -> dict:
    return {
        "solver": "strego",
        "function": task["func_name"],
        "dimension": task["dim"],
        "trial": task["trial"],
        "seed": seed,
        "best_y": best_y,
        "n_evaluations": n_evaluations,
        "log_path": task["log_path"],
    }


def task_label(task: dict) -> str:
   """Short name of a run, used in progress and failure messages."""
    return f"{task['func_name']}_d{task['dim']} trial {task['trial']}"




def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--output-dir", required=True)
    p.add_argument("--functions", default="rastrigin_fullshift,alpine01_fullshift,ackley_fullshift,schwefel_fullshift")
    p.add_argument("--dims", default="100,50,20,4")
    p.add_argument("--num-trials", type=int, default=20)
    p.add_argument("--budget", type=int, default=200)
    p.add_argument("--n-init", type=int, default=DEFAULTS["n_init"])
    p.add_argument("--seed", type=int, default=42, help="Base seed; trial t uses seed + t")
    p.add_argument("--max-workers", type=int, default=4)

    p.add_argument("--noise-type", default="none", choices=["none", "additive", "multiplicative"])
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument("--deterministic", action="store_true",
                   help="Noise-free objective: use a single contraction factor")

    p.add_argument("--prebuilt-doe-dir", default=None,
                   help="Reuse DoE catalogs from a previous campaign instead of generating new ones. "
                        "Catalogs are matched on function, dimension and --noise-type")

    p.add_argument("--objective-pair", default=DEFAULTS["objective_pair"], choices=["MU_IVR", "MU_SIGMA"])
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

    return p.parse_args()


def main() -> None:
    args = parse_args()

    functions = [f.strip() for f in args.functions.split(",") if f.strip()]
    dims = [int(d) for d in args.dims.split(",") if d.strip()]

    unknown = [f for f in functions if f not in FUNCTIONS]
    if unknown:
        raise SystemExit(f"Unknown function(s): {', '.join(unknown)}\nAvailable: {', '.join(FUNCTIONS)}")

    doe_dir = os.path.join(args.output_dir, "doe")
    detail_dir = os.path.join(args.output_dir, "details")
    ensure_directory(detail_dir)

    config = {
        "seed": args.seed,
        "noise_type": args.noise_type,
        "noise_std": args.noise_std,
        "budget": args.budget,
        "n_init": args.n_init,
        "objective_pair": args.objective_pair,
        "ivr_integration_points": args.ivr_integration_points,
        "global_batch_size": args.global_batch_size,
        "min_local_points": args.min_local_points,
        "relaxation_step": args.relaxation_step,
        "candidate_pool_size": args.candidate_pool_size,
        "local_num_restarts": args.local_num_restarts,
        "local_raw_samples": args.local_raw_samples,
        "beta_1": args.beta_1,
        "beta_2": args.beta_2,
        "gamma": args.gamma,
        "kappa": args.kappa,
        "deterministic": args.deterministic,
        "prebuilt_doe_dir": os.path.abspath(args.prebuilt_doe_dir) if args.prebuilt_doe_dir else None,
    }

    config_path = os.path.join(args.output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as handle:
            previous = json.load(handle)
        changed = sorted(k for k in set(config) | set(previous) if config.get(k) != previous.get(k))
        if changed:
            diff = "\n    ".join(f"{k}: {previous.get(k)!r} -> {config.get(k)!r}" for k in changed)
            raise SystemExit(
                f"{args.output_dir} was started with different settings:\n    {diff}\n"
                f"Resuming would mix two configurations in one summary. "
                f"Use a fresh --output-dir, or rerun with the original settings."
            )
    else:
        with open(config_path, "w") as handle:
            json.dump(config, handle, indent=2)

    print(f"STREGO campaign -> {args.output_dir}")
    print(f"  functions : {', '.join(functions)}")
    print(f"  dims      : {dims}   trials: {args.num_trials}   budget: {args.budget}")
    print(f"  noise     : {args.noise_type} (std={args.noise_std})")
    print(f"  global    : [{args.objective_pair}] batch={args.global_batch_size}")

    print("\n[1/2] Shared DoE")
    doe_paths: dict[tuple[str, int], str] = {}
    try:
        for func_name in functions:
            for dim in dims:
                if args.prebuilt_doe_dir:
                    path = find_prebuilt_doe(args.prebuilt_doe_dir, func_name, dim, args.noise_type)
                    print(f"  reuse    {os.path.basename(path)}")
                else:
                    path = os.path.join(
                        doe_dir,
                        doe_filename(func_name, dim, args.noise_type, args.num_trials, args.seed, args.n_init),
                    )
                    # On resume, keep the catalog the finished runs started from.
                    if os.path.exists(path):
                        print(f"  keep     {os.path.basename(path)}")
                    else:
                        generate_doe_catalog(
                            path, func_name, dim, args.n_init, args.num_trials, args.seed,
                            args.noise_type, args.noise_std,
                        )
                        print(f"  generate {os.path.basename(path)}")
                check_doe_catalog(path, dim, args.num_trials, args.n_init)
                doe_paths[(func_name, dim)] = path
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"DoE error: {exc}")

    tasks = [
        {
            "func_name": func_name,
            "dim": dim,
            "trial": trial,
            "doe_path": doe_paths[(func_name, dim)],
            "log_path": detail_log_path(detail_dir, func_name, dim, trial),
            "config": config,
        }
        for func_name in functions
        for dim in dims
        for trial in range(args.num_trials)
    ]

    results: list[dict] = []
    todo: list[dict] = []
    for task in tasks:
        done = completed_run(task["log_path"], args.budget)
        if done is None:
            todo.append(task)
        else:
            results.append(summary_row(task, done["seed"], done["best_y"], args.budget))

    print(f"\n[2/2] {len(todo)} run(s) on {args.max_workers} worker(s)"
          + (f"  ({len(results)} already complete, skipped)" if results else ""))

    failures: list[tuple[dict, Exception]] = []

    def record(task: dict, row: dict | None, exc: Exception | None) -> None:
.
        if exc is not None:
            failures.append((task, exc))
            print(f"  [FAILED] {task_label(task)}: {exc}")
        else:
            results.append(row)
            print(f"  {task_label(task)} -> {row['best_y']:.6f}")

    if args.max_workers <= 1:
        for task in todo:
            try:
                record(task, run_one(task), None)
            except Exception as exc:
                record(task, None, exc)
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(run_one, task): task for task in todo}
            for future in as_completed(futures):
                try:
                    record(futures[future], future.result(), None)
                except Exception as exc:
                    record(futures[future], None, exc)

    summary_path = os.path.join(args.output_dir, "summary.csv")
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in sorted(results, key=lambda r: (r["function"], r["dimension"], r["trial"])):
            writer.writerow(row)

    print(f"\n{len(results)}/{len(tasks)} runs complete -> {summary_path}")
    if failures:
        print(f"{len(failures)} run(s) failed; rerun the same command to retry them:")
        for task, exc in failures:
            print(f"  {task_label(task)}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
