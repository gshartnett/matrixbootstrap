"""
Feasibility-based bootstrap lower bound for the d=2 two-matrix model.

Instead of minimizing ⟨H⟩ subject to bootstrap constraints (which is
ill-posed because the SDP has recession directions), this script fixes
⟨H⟩ = E_trial as a hard equality constraint and asks CLARABEL: feasible
or not?  For E_trial < E_0 the constraints are inconsistent and CLARABEL
returns infeasible; for E_trial ≥ E_0 a consistent solution exists.
Binary-searching on E_trial yields the rigorous lower bound

    E_bound(L) ≤ E_0

with no regularization required.

Convention mapping (same as two_matrix_bo_comparison.py):
    lambda = g^2 * N  <->  g4 = 2 * lambda   (g4/4 = g^2/2)
    E_HKK = 2 * E_ours

Usage:
    python scripts/two_matrix_feasibility_bound.py           # run scan + plot
    python scripts/two_matrix_feasibility_bound.py --plot    # plot from saved results
"""

import json
import logging
import os

import fire
import matplotlib.pyplot as plt
import numpy as np
import yaml

from matrixbootstrap.algebra import SingleTraceOperator
from matrixbootstrap.bootstrap import (
    BootstrapSystem,
    BootstrapSystemReal,
)
from matrixbootstrap.born_oppenheimer import BornOppenheimer
from matrixbootstrap.config_utils import (
    _struct_hash,
    generate_config_two_matrix,
)
from matrixbootstrap.models import TwoMatrix
from matrixbootstrap.solver_newton import solve_bootstrap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── parameters ────────────────────────────────────────────────────────────────
L = 3
G2 = 1.0

# FAST=True: 6 lambda points, 8 bisection steps (~48 feasibility checks/lambda)
# FAST=False: 13 lambda points, 12 bisection steps
FAST = True

if FAST:
    LAMBDA_VALUES = np.round(np.linspace(0.25, 8.0, 6), decimals=4).tolist()
    N_BISECT = 8
    MAXITERS = 20
else:
    LAMBDA_VALUES = np.round(np.linspace(0.0, 8.0, 13), decimals=4).tolist()
    N_BISECT = 12
    MAXITERS = 30

MAXITERS_CVXPY = 500
EPS = 1e-5
TOL_QUAD = 1e-6  # max quadratic-constraint violation to call a run "converged"

# BO grids
BO_X_GRID_LOWER = np.linspace(-12, 12, 200)
BO_X_GRID_UPPER = np.linspace(-12, 12, 80)

HKK_CSV = "HKK_extracted_data_Fig3a.csv"
RESULTS_DIR = f"runs/TwoMatrix_feasibility_L{L}_g2_{G2}"

# Reference config dir — one config per (g2, g4), used only for cache paths.
# The per-config cache (linear constraints, null space, quadratic constraints)
# is shared with any optimization run having the same physics parameters.
_REF_CONFIG_DIR = f"TwoMatrix_feasibility_ref_L{L}_g2_{G2}"


# ── bootstrap construction ────────────────────────────────────────────────────


def _build_bootstrap(g2: float, g4: float) -> tuple:
    """Build a BootstrapSystem for the given couplings, using disk caches.

    Generates a reference config (if absent) to pin the cache paths, then
    constructs and loads the BootstrapSystem.  Subsequent calls for the same
    (g2, g4) reuse the on-disk cache — no redundant computation.
    """
    # Generate a reference config to get a stable config_id and cache path.
    # reg/maxiters don't affect the bootstrap structure, only the optimizer run.
    config_id = generate_config_two_matrix(
        config_dir=_REF_CONFIG_DIR,
        g2=g2,
        g4=g4,
        st_operator_to_minimize="energy",
        st_operators_evs_to_set=None,
        max_degree_L=L,
        impose_symmetries=True,
        load_from_previously_computed=True,
        optimization_method="newton",
        cvxpy_solver="CLARABEL",
        maxiters=MAXITERS,
        maxiters_cvxpy=MAXITERS_CVXPY,
        init_scale=1e-2,
        reg=1e-3,
        eps_abs=EPS,
        eps_rel=EPS,
        tol=TOL_QUAD,
        radius=1e10,
        use_factorization_block=True,
    )

    with open(f"runs/{_REF_CONFIG_DIR}/configs/{config_id}.yaml") as f:
        config = yaml.safe_load(f)

    structural_cache_path = f"cache/structural/{_struct_hash(config)}"
    config_cache_path = f"cache/per_config/{config_id}"

    model = TwoMatrix(couplings=config["model"]["couplings"])

    bootstrap = BootstrapSystemReal(
        matrix_system=model.matrix_system,
        hamiltonian=model.hamiltonian,
        gauge_generator=model.gauge_generator,
        max_degree_L=L,
        odd_degree_vanish=True,
        simplify_quadratic=True,
        symmetry_generators=model.symmetry_generators,
        structural_cache_path=structural_cache_path,
        config_cache_path=config_cache_path,
    )
    bootstrap.load_structural_cache(structural_cache_path)
    bootstrap.load_config_cache(config_cache_path)

    return bootstrap, model


# ── feasibility check ─────────────────────────────────────────────────────────


def check_feasibility(
    bootstrap: BootstrapSystem,
    energy_op: SingleTraceOperator,
    E_trial: float,
    init: np.ndarray = None,
) -> tuple:
    """Return (feasible, param) for E_trial at level L.

    Fixes ⟨H⟩ = E_trial as a hard equality constraint and minimizes ‖x‖
    (no energy objective, so no recession direction to escape along).
    feasible=True iff Newton converges with small EOM violation.

    radius=500 prevents Newton blow-up: with no energy objective, step 1
    (no EOM) finds the minimum-norm point on the energy hyperplane, which
    without a radius bound can be arbitrarily large and destabilise the
    subsequent Newton steps.  Comparison runs show ‖x‖ < 30 at reg=1e-3
    for all lambda values tested, so radius=500 safely covers the feasible
    region while keeping the Newton iterates bounded.

    init : optional warm-start (e.g. from a nearby feasible check).
    """
    param, result = solve_bootstrap(
        bootstrap=bootstrap,
        st_operator_to_minimize=None,  # pure feasibility: minimize ‖x‖
        st_operator_inhomo_constraints=[
            (SingleTraceOperator(data={(): 1}), 1.0),  # tr(1) = 1
            (energy_op, float(E_trial)),  # ⟨H⟩ = E_trial  (hard)
        ],
        maxiters=MAXITERS,
        maxiters_cvxpy=MAXITERS_CVXPY,
        tol=TOL_QUAD,
        eps_abs=EPS,
        eps_rel=EPS,
        reg=0.0,
        radius=500.0,  # bounded: prevents Newton blow-up (see docstring)
        init=init,
        init_scale=1e-2,
        cvxpy_solver="CLARABEL",
        use_factorization_block=True,
    )
    if param is None:
        return False, None
    feasible = result["max_quad_constraint_violation"] < TOL_QUAD
    return feasible, (param if feasible else None)


# ── binary search ─────────────────────────────────────────────────────────────


def find_lower_bound(lam: float, E_lo: float, E_hi: float) -> dict:
    """Binary-search for the bootstrap lower bound at a given 't Hooft coupling.

    Parameters
    ----------
    lam   : 't Hooft coupling lambda = g^2 * N
    E_lo  : starting infeasible value (e.g. 0)
    E_hi  : starting feasible value (e.g. BO variational upper bound)

    Returns
    -------
    dict with keys: lam, E_bound, E_lo_final, E_hi_final, history
        E_bound = E_hi after binary search = lowest known feasible E_trial
                = rigorous lower bound on E_0.
    """
    g4 = 2.0 * lam
    bootstrap, model = _build_bootstrap(G2, g4)
    energy_op = model.operators_to_track["energy"]

    logger.info("lambda=%.4f: verifying bracket [%.4f, %.4f]", lam, E_lo, E_hi)

    # verify bracket: E_hi must be feasible, E_lo infeasible
    feasible_hi, init = check_feasibility(bootstrap, energy_op, E_hi)
    if not feasible_hi:
        logger.warning("  E_hi=%.4f is INFEASIBLE — widening", E_hi)
        E_hi *= 1.5
        feasible_hi, init = check_feasibility(bootstrap, energy_op, E_hi)
        if not feasible_hi:
            logger.error("  Cannot find feasible upper bracket for lambda=%.4f", lam)
            return {
                "lam": lam,
                "E_bound": None,
                "E_lo_final": E_lo,
                "E_hi_final": E_hi,
                "history": [],
            }

    feasible_lo, _ = check_feasibility(bootstrap, energy_op, E_lo)
    if feasible_lo:
        logger.warning(
            "  E_lo=%.4f is unexpectedly FEASIBLE for lambda=%.4f", E_lo, lam
        )

    # init = param from E_hi check; warm-start each bisection from last feasible solution
    history = []
    for i in range(N_BISECT):
        E_mid = 0.5 * (E_lo + E_hi)
        feasible, param_mid = check_feasibility(bootstrap, energy_op, E_mid, init=init)
        history.append({"E_trial": float(E_mid), "feasible": bool(feasible)})
        if feasible:
            E_hi = E_mid  # tighten the feasible upper end
            init = param_mid  # warm-start next check from this solution
        else:
            E_lo = (
                E_mid  # raise the infeasible lower end (keep init from last feasible)
            )
        logger.info(
            "  bisect %d/%d: E=%.6f -> %s  [lo=%.6f, hi=%.6f]",
            i + 1,
            N_BISECT,
            E_mid,
            "FEASIBLE" if feasible else "infeasible",
            E_lo,
            E_hi,
        )

    # E_hi = lowest known feasible E_trial = rigorous lower bound E_0 >= E_hi
    logger.info(
        "lambda=%.4f: lower bound = %.6f  (bracket width %.2e)",
        lam,
        E_hi,
        E_hi - E_lo,
    )
    return {
        "lam": float(lam),
        "E_bound": float(E_hi),
        "E_lo_final": float(E_lo),
        "E_hi_final": float(E_hi),
        "history": history,
    }


# ── scan ──────────────────────────────────────────────────────────────────────


def run_scan() -> dict:
    """Compute BO bounds to bracket the scan, then binary-search each lambda."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_file = os.path.join(RESULTS_DIR, "feasibility_bounds.json")

    logger.info("Computing BO bounds for initial bracket...")
    bo_data = {}
    for lam in LAMBDA_VALUES:
        g4 = 2.0 * lam
        bo = BornOppenheimer(g2=G2, g4=g4)
        e_lower = bo.solve(BO_X_GRID_LOWER).fun
        e_upper = bo.solve_upper(BO_X_GRID_UPPER).fun
        bo_data[lam] = {"E_bo_lower": float(e_lower), "E_bo_upper": float(e_upper)}
        logger.info(
            "  lambda=%.4f: BO lower=%.4f, BO upper=%.4f", lam, e_lower, e_upper
        )

    all_results = {}
    for lam in LAMBDA_VALUES:
        E_lo = 0.0
        E_hi = bo_data[lam]["E_bo_upper"] * 1.1  # comfortably above E_0
        result = find_lower_bound(lam, E_lo, E_hi)
        result["E_bo_lower"] = bo_data[lam]["E_bo_lower"]
        result["E_bo_upper"] = bo_data[lam]["E_bo_upper"]
        all_results[str(lam)] = result

    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved results to %s", results_file)
    return {float(k): v for k, v in all_results.items()}


# ── plotting ──────────────────────────────────────────────────────────────────


def make_plot(all_results: dict = None):
    results_file = os.path.join(RESULTS_DIR, "feasibility_bounds.json")

    if all_results is None:
        if not os.path.exists(results_file):
            print(f"No results file found: {results_file}")
            return
        with open(results_file) as f:
            raw = json.load(f)
        all_results = {float(k): v for k, v in raw.items()}

    lams_sorted = sorted(all_results.keys())
    bo_upper = [all_results[lam]["E_bo_upper"] for lam in lams_sorted]
    bo_lower = [all_results[lam]["E_bo_lower"] for lam in lams_sorted]
    bounds = [all_results[lam].get("E_bound") for lam in lams_sorted]

    # HKK extracted data
    hkk = {}
    if os.path.exists(HKK_CSV):
        raw_csv = np.genfromtxt(
            HKK_CSV, delimiter=",", skip_header=2, filling_values=np.nan
        )

        def clean(x, y):
            mask = ~(np.isnan(x) | np.isnan(y))
            return x[mask], y[mask] / 2

        hkk["L3_x"], hkk["L3_y"] = clean(raw_csv[:, 0], raw_csv[:, 1])
        print(f"Loaded HKK data from {HKK_CSV}")
    else:
        print("HKK CSV not found, skipping.")

    COLOR_HKK = "0.35"
    COLOR_OURS = "#1f77b4"

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.fill_between(lams_sorted, bo_upper, bo_lower, alpha=0.15, color=COLOR_OURS)
    ax.plot(
        lams_sorted,
        bo_upper,
        "--",
        color=COLOR_OURS,
        lw=1.5,
        label=r"Our BO upper ($E_\mathrm{var}$)",
    )
    ax.plot(
        lams_sorted,
        bo_lower,
        "-",
        color=COLOR_OURS,
        lw=1.5,
        label=r"Our BO lower ($E_\mathrm{BO}$)",
    )

    if hkk:
        ax.plot(
            hkk["L3_x"],
            hkk["L3_y"],
            "s",
            ms=5,
            color=COLOR_HKK,
            alpha=0.8,
            label="HKK bootstrap $L=3$",
        )

    valid = [(lam, b) for lam, b in zip(lams_sorted, bounds) if b is not None]
    if valid:
        lams_v, bounds_v = zip(*valid)
        ax.plot(
            lams_v,
            bounds_v,
            "o-",
            color=COLOR_OURS,
            ms=6,
            lw=2,
            label=f"Feasibility lower bound $L={L}$",
        )

    ax.set_xlabel(r"$\lambda = g^2 N$", fontsize=13)
    ax.set_ylabel(r"$E$  (our conventions, $= E_\mathrm{HKK}/2$)", fontsize=12)
    ax.set_title(
        rf"Two-matrix model, $L={L}$, $m=1$ — feasibility lower bound", fontsize=12
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "feasibility_bound.pdf")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()
    plt.close()


# ── entrypoint ────────────────────────────────────────────────────────────────


def main(plot: bool = False):
    if plot:
        make_plot()
        return
    all_results = run_scan()
    make_plot(all_results)


if __name__ == "__main__":
    fire.Fire(main)
