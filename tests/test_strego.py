"""Smoke tests: the loop runs end to end and respects its contracts.

Deliberately small budgets -- these check wiring, not optimization quality.
Run with:  pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest

from strego import STREGO, get_function_spec, wrap_noisy
from strego.benchmarks import FUNCTIONS
from strego.trust_region import TrustRegionBO


def sphere(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x, dtype=float) ** 2))


def make_optimizer(**kwargs) -> STREGO:
    dim = kwargs.pop("dim", 4)
    defaults = dict(
        objective_fn=sphere,
        lower_bounds=np.full(dim, -5.0),
        upper_bounds=np.full(dim, 5.0),
        budget=30,
        n_init=12,
        seed=0,
    )
    defaults.update(kwargs)
    return STREGO(**defaults)


def test_respects_budget():
    result = make_optimizer(budget=30, n_init=12).run()
    assert result.n_evaluations == 30


def test_improves_on_initial_design():
    optimizer = make_optimizer(budget=40, n_init=12)
    initial_best = min(optimizer.y_obs)  # design is evaluated in __init__
    result = optimizer.run()
    assert result.best_y <= initial_best


def test_best_x_matches_best_y():
    result = make_optimizer().run()
    assert sphere(result.best_x) == pytest.approx(result.best_y)


def test_injected_doe_is_used_without_reevaluation():
    dim, n_init = 4, 10
    rng = np.random.default_rng(0)
    doe_points = rng.uniform(-5.0, 5.0, size=(n_init, dim))
    doe_values = np.array([sphere(x) for x in doe_points])

    optimizer = make_optimizer(
        dim=dim, budget=20, n_init=n_init, doe_points=doe_points, doe_values=doe_values
    )
    assert np.allclose(optimizer.y_obs[:n_init], doe_values)
    assert np.allclose(np.array(optimizer.X_obs[:n_init]), doe_points)


def test_mu_sigma_objective_pair_runs():
    result = make_optimizer(objective_pair="MU_SIGMA").run()
    assert np.isfinite(result.best_y)


def test_noisy_objective_runs():
    noisy = wrap_noisy(sphere, "additive", 0.5)
    result = make_optimizer(objective_fn=noisy).run()
    assert np.isfinite(result.best_y)


def test_log_written(tmp_path):
    log_path = tmp_path / "trace.csv"
    result = make_optimizer(log_path=str(log_path)).run()
    lines = log_path.read_text().strip().splitlines()
    assert lines[0].startswith("seed,iteration,phase")
    assert len(lines) == result.n_evaluations + 1  # + header


@pytest.mark.parametrize("name", FUNCTIONS)
def test_benchmark_specs_are_well_formed(name):
    fn, lower, upper = get_function_spec(name, 6)
    assert lower.shape == upper.shape == (6,)
    assert np.all(lower < upper)
    assert np.isfinite(fn(np.zeros(6)))


def test_shifted_optimum_is_interior_and_better_than_centre():
    """The full-shift variants must move the optimum off the domain centre."""
    fn, lower, upper = get_function_spec("rastrigin_fullshift", 8)
    centre = (lower + upper) / 2.0
    assert fn(centre) > 1.0  # centre is no longer the optimum


def test_trust_region_rejects_invalid_hyperparameters():
    # gamma * beta_2 >= 1 keeps a contract-then-expand cycle from losing ground.
    with pytest.raises(ValueError):
        TrustRegionBO(x_0=np.zeros(3), f_0=1.0, sigma_0=0.5,
                      beta_1=0.5, beta_2=0.3, gamma=2.0, kappa=1.0)
    # beta_2 <= beta_1 < 1
    with pytest.raises(ValueError):
        TrustRegionBO(x_0=np.zeros(3), f_0=1.0, sigma_0=0.5,
                      beta_1=0.2, beta_2=0.9, gamma=3.5, kappa=1.0)


def test_budget_below_n_init_is_rejected():
    with pytest.raises(ValueError):
        make_optimizer(budget=5, n_init=10)


# -- relaxation rule ---------------------------------------------------------
# These call the local-design builder directly, so they need no NSMA search:
# constructing STREGO only evaluates the initial design.


def test_relaxation_reaches_n_min_from_a_collapsed_region():
    optimizer = make_optimizer(dim=4, budget=30, n_init=12, min_local_points=8)
    optimizer.sigma_k = 1e-6  # a trust region far too small to hold any point
    X_local, y_local = optimizer._gather_local_data(optimizer._build_trust_region())
    assert X_local.shape[0] >= 8
    assert X_local.shape[0] == y_local.shape[0]


def test_no_relaxation_when_region_already_holds_n_min():
    optimizer = make_optimizer(dim=4, budget=30, n_init=12, min_local_points=2)
    optimizer.sigma_k = 1.0  # region covers the whole domain
    X_local, _ = optimizer._gather_local_data(optimizer._build_trust_region())
    assert X_local.shape[0] == 12  # every initial point, untouched


def test_relaxation_grows_additively_and_stops_at_first_sufficient_box():
    """The accepted box is the first one on the 1x, 1.5x, 2.0x, ... ladder
    holding n_min points -- not a larger one."""
    optimizer = make_optimizer(dim=4, budget=30, n_init=12, min_local_points=6, relaxation_step=0.5)
    optimizer.sigma_k = 0.05
    tr = optimizer._build_trust_region()
    X_local, _ = optimizer._gather_local_data(tr)
    assert X_local.shape[0] >= 6

    # Walk the ladder independently and find the first sufficient rung; the
    # optimizer must have stopped exactly there.
    Xn = (np.array(optimizer.X_obs) + 5.0) / 10.0
    radius_tr = tr["radius_n"]
    radii = [(1.0 + j * 0.5) * radius_tr for j in range(10_000)]
    counts = [
        int(np.all(np.abs(Xn - tr["center_n"]) <= r + 1e-12, axis=1).sum()) if r < 1.0 else len(Xn)
        for r in radii
    ]
    first = next(j for j, c in enumerate(counts) if c >= 6 or radii[j] >= 1.0)
    assert X_local.shape[0] == counts[first]


def test_n_min_above_n_init_is_rejected():
    # n_min <= n_init is what guarantees the relaxation terminates.
    with pytest.raises(ValueError):
        make_optimizer(n_init=12, min_local_points=13)


def test_non_positive_relaxation_step_is_rejected():
    with pytest.raises(ValueError):
        make_optimizer(relaxation_step=0.0)


# -- local optimizer settings ------------------------------------------------


def test_restarts_and_raw_samples_scale_with_dimension():
    optimizer = make_optimizer(dim=6, budget=20, n_init=10)
    assert optimizer.local_num_restarts == 2 * 6 + 4
    assert optimizer.local_raw_samples == (2 * 6 + 4) ** 2


def test_explicit_restarts_and_raw_samples_are_used_as_given():
    # No floor and no cap: small explicit values must survive untouched.
    optimizer = make_optimizer(budget=20, n_init=10, local_num_restarts=2, local_raw_samples=3)
    assert (optimizer.local_num_restarts, optimizer.local_raw_samples) == (2, 3)
    with pytest.raises(ValueError):
        make_optimizer(budget=20, n_init=10, local_num_restarts=5, local_raw_samples=4)
